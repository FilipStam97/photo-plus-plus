import base64
from io import BytesIO
import json
import cv2
import face_recognition
import hdbscan
from flask import Flask, app, jsonify, request
from minio import Minio
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import psycopg2

app = Flask(__name__)

client = Minio(
    "localhost:9000",              
    access_key="admin",      
    secret_key="admin123",   
    secure=False                  
)
BUCKET_NAME = "test"


try:
    conn = psycopg2.connect(
        host="localhost",
        database="postgresql",
        user="admin",
        password="admin123",
        port="5432"  # Default PostgreSQL port
    )
    print("Connection to PostgreSQL successful!")

except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")
    conn = None

def list_all_images():
    objects = client.list_objects(BUCKET_NAME, recursive=True)
    image_files = [
        obj.object_name
        for obj in objects
        if obj.object_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return image_files

def list_all_images_folder(bucket_name, folder_name):
    objects = client.list_objects(bucket_name, prefix= folder_name, recursive=True)
    image_files = [
        obj.object_name
        for obj in objects
        if obj.object_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return image_files

def get_faces_and_embeddings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    print('nadjene karakteristike lica za  sliku')
    return face_encodings, face_locations

@app.route('/save-face-embeddings', methods=['POST'])
def  get_face_embeddings_minio(bucket_name, folder_name):
    metadata = []

    image_files = list_all_images_folder(bucket_name, folder_name)
    if not image_files:
        return jsonify({"error": "No images found in bucket"}), 400

    for img_name in image_files:
        try:
            response = client.get_object(bucket_name, img_name)
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Failed to read {img_name}: {e}")
            continue

        face_encodings, face_locations = get_faces_and_embeddings(image)

        for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
            encoding_str = base64.b64encode(embedding.tobytes()).decode("utf-8")
            metadata.append({
                "image_name": img_name,
                "face_index": idx,
                "bbox": bbox,
                "encoding_b64": encoding_str
            })
    return jsonify({"metadata": metadata})

# gets face embeddings from uploaded images

def get_embeddings_db(bucket_name):
    if(conn):
        cur = conn.cursor()
        cur.execute('SELECT id, bucket, fileName, boundingBox, faceIndex, encoding_b64 FROM face_embeddings WHERE bucket = ' + bucket_name)
        rows = cur.fetchall()
        return rows
    return []   

@app.route('/predict-photo', methods=['POST'])
def cluster_faces_minio():
    embeddings = []
    metadata = []

    image_files = list_all_images()
    if not image_files:
        return jsonify({"error": "No images found in bucket"}), 400

    # Step 1. Extract embeddings and metadata for all faces
    for img_name in image_files:
        try:
            response = client.get_object(BUCKET_NAME, img_name)
            print('nadjen bucket')
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Failed to read {img_name}: {e}")
            continue

        face_encodings, face_locations = get_faces_and_embeddings(image)

        for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
            embeddings.append(embedding)
            metadata.append({
                "image_name": img_name,
                "face_index": idx,
                "bbox": bbox
            })
        print('napravljeni metapodaci za slike')

    if not embeddings:
        return jsonify({"error": "No faces found in bucket"}), 400

    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dbscan = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    # dbscan.fit(embeddings)
    labels = dbscan.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)
        bbox = metadata[idx]["bbox"]
        img_name = metadata[idx]["image_name"]
        top, right, bottom, left = bbox

        try:
            # Reopen the source image
            response = client.get_object(BUCKET_NAME, img_name)
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()

            # Crop face
            face_crop = image[top:bottom, left:right]
            pil_face = Image.fromarray(face_crop)
            buf = BytesIO()
            pil_face.save(buf, format="JPEG")
            buf.seek(0)
            print('napravljen isecak lica')

            # Convert face encoding to base64 for metadata (MinIO metadata must be strings)
            encoding_str = base64.b64encode(embeddings[idx].tobytes()).decode("utf-8")

            # Define new filename and metadata
            face_key = f"faces/{label}_{img_name.replace('/', '_')}_{idx}.jpg"
            face_meta = {
                "source_image": img_name,
                "face_index": str(idx),
                "bbox": json.dumps(bbox),
                "cluster_label": str(label),
                "encoding_b64": encoding_str[:2000]  # optional: truncate to avoid metadata size limits
            }

            # Upload to MinIO with metadata
            client.put_object(
                BUCKET_NAME,
                face_key,
                buf,
                length=buf.getbuffer().nbytes,
                content_type="image/jpeg",
                metadata=face_meta
            )
            print('ubacen isecak na minio')


            # Prepare response entry
            clusters.setdefault(label, []).append({
                "face_key": face_key,
                "image_name": img_name,
                "bbox": bbox,
                "minio_metadata": face_meta
            })

        except Exception as e:
            print(f"Failed to crop/upload face for {img_name}: {e}")
            continue
    return jsonify({"clusters": clusters})

@app.route('/find-similar-faces-batch', methods=['POST'])
def find_similar_faces_batch():
    """
    More efficient version: finds similar faces by comparing against 
    all face embeddings at once using vectorized operations.
    
    Expected input: JSON with 'face_key' or file upload with 'image'
    """
    
    # Get reference embedding (same as above)
    reference_embedding = None
    reference_face_key = None
    
    if 'face_key' in request.form or ('face_key' in request.json if request.json else False):
        reference_face_key = request.form.get('face_key') or request.json.get('face_key')
        
        try:
            # Get the face image from MinIO and stat to get metadata
            stat = client.stat_object(BUCKET_NAME, reference_face_key)
            metadata = stat.metadata
            
            # Try to get encoding from metadata first (faster)
            print('nadjen encoding u metapodacima')
            encoding_b64 = metadata['x-amz-meta-encoding_b64']
            reference_embedding = np.frombuffer(
                base64.b64decode(encoding_b64), 
                dtype=np.float64
            )
          
        except Exception as e:
            return jsonify({"error": f"Failed to load reference face: {str(e)}"}), 400
  
    else:
        return jsonify({"error": "Must provide either 'image' file or 'face_key' parameter"}), 400
    
    # Normalize reference embedding
    reference_embedding = reference_embedding / np.linalg.norm(reference_embedding)
    
    threshold = float(request.form.get('threshold', 0.6) if 'threshold' in request.form 
                     else request.json.get('threshold', 0.6) if request.json else 0.6)
    
    # Build embedding database from all images
    all_embeddings = []
    all_metadata = []
    
    image_files = list_all_images()
    
    for img_name in image_files:
        if img_name.startswith('faces/'):
            continue
        
        try:
            response = client.get_object(BUCKET_NAME, img_name)
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
            
            face_encodings, face_locations = get_faces_and_embeddings(image)
            
            for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
                all_embeddings.append(embedding)
                all_metadata.append({
                    "image_name": img_name,
                    "face_index": idx,
                    "bbox": bbox
                })
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
            continue
    
    if not all_embeddings:
        return jsonify({"error": "No faces found in bucket"}), 400
    
    # Vectorized similarity computation
    all_embeddings = np.array(all_embeddings)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(all_embeddings, reference_embedding)
    
    # Find matches
    matching_indices = np.where(similarities >= threshold)[0]
    
    matching_images = []
    seen_images = set()
    
    for idx in matching_indices:
        img_name = all_metadata[idx]["image_name"]
        
        # Only include each image once (first matching face)
        if img_name not in seen_images:
            matching_images.append({
                "image_name": img_name,
                "face_index": all_metadata[idx]["face_index"],
                "bbox": all_metadata[idx]["bbox"],
                "similarity": float(similarities[idx])
            })
            seen_images.add(img_name)
    
    matching_images.sort(key=lambda x: x['similarity'], reverse=True)
    
    return jsonify({
        "reference_face_key": reference_face_key,
        "threshold": threshold,
        "total_matches": len(matching_images),
        "matching_images": matching_images
    })


def face_distance(enc1, enc2):
    return np.linalg.norm(enc1 - enc2)

if __name__ == "__main__":
    app.run(debug = True)