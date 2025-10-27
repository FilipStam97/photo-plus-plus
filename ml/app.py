import base64
from io import BytesIO
import json
import cv2
import os
import face_recognition
import hdbscan
from flask import Flask, app, jsonify, request
from minio import Minio, S3Error
import numpy as np
from PIL import Image
import psycopg2
from psycopg2.extras import execute_values

from ml.helpers import get_faces_and_embeddings, is_similar, list_all_images_folder

app = Flask(__name__)
host = os.getenv("MINIO_HOST", "localhost") #lokalno je localhost, a ovako preko dockera treba da bude "minio"
port = os.getenv("MINIO_PORT", "9000")
access = os.getenv("MINIO_ACCESS_KEY", "admin")
secret = os.getenv("MINIO_SECRET_KEY", "admin123")
secure = os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes")

endpoint = f"{host}:{port}"
print(endpoint)

client = Minio(
    endpoint,
    access_key=access,
    secret_key=secret,
    secure=secure
)
BUCKET_NAME = "test"

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"), #localhost radi kad se python pokrene lokalno, a "db" kad je docker pokrenut
            port=os.getenv("DB_PORT", 5332), #5432 preko dockera
            database=os.getenv("DB_NAME", "bank"),
            user=os.getenv("DB_USER", "admin"),
            password=os.getenv("DB_PASSWORD", "admin123")
        )
        print("Connection to PostgreSQL successful!")

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        conn = None
    return conn

def list_all_images():
    objects = client.list_objects(BUCKET_NAME, recursive=True)
    image_files = [
        obj.object_name
        for obj in objects
        if obj.object_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return image_files



@app.route('/get-face-embeddings', methods=['GET'])
def get_face_embeddings():
    bucket_name = request.args.get('bucketName') #ne radi za general bucket iz nekog razloga, ja sam napravila poseban "test" bucket

    metadata = []

    image_files = list_all_images_folder(bucket_name, client)
    if not image_files:
        return jsonify({"error": "No images found in bucket"}), 400

    for img_name in image_files:
        if img_name.startswith('faces/'):
            continue
        try:
            response = client.get_object(bucket_name, img_name)
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Failed to read {img_name}: {e}")
            continue

        try:
            face_encodings, face_locations = get_faces_and_embeddings(image)

            for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
                print('encoding?')
                encoding_str = base64.b64encode(embedding.tobytes()).decode("utf-8")

                metadata.append({
                    "image_name": img_name,
                    "face_index": idx,
                    "bbox": bbox,
                    "encoding_b64": encoding_str
                })
        except Exception as e:
            print('Failed to extract face embeddings ', e)
            continue
    return jsonify({"metadata": metadata})

# gets face embeddings from uploaded images

def get_embeddings_db(bucket_name):
    conn = get_db_connection()
    if(conn):
        cur = conn.cursor()
        cur.execute('SELECT id, bucket, fileName, boundingBox, faceIndex, encoding_b64 FROM FaceEmbedding WHERE bucket = ' + bucket_name)
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

    for img_name in image_files:
        try:
            response = client.get_object(BUCKET_NAME, img_name)
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Failed to read {img_name}: {e}")
            continue

        face_encodings, face_locations = get_faces_and_embeddings(image)
        embeddings = insert_all_face_embeddings(metadata, img_name, face_encodings, face_locations)

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

def insert_all_face_embeddings(embeddings, metadata, img_name, face_encodings, face_locations):
    values = []
    embeddings = []
    for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
        encoding_str= base64.b64encode(embedding.tobytes()).decode("utf-8")
        embeddings.append(embedding)
        metadata.append({
                "image_name": img_name,
                "face_index": idx,
                "bbox": bbox
            })
        values.append((
                img_name,
                psycopg2.extras.Json(embedding),
                psycopg2.extras.Json(bbox),
                False
            ))
        
        # Batch insert face embeddings into database
    conn = get_db_connection()
    cursor = conn.cursor()
        
    execute_values(cursor, """
            INSERT INTO face_embeddings 
            (image_name, embedding, bbox, is_representative)
            VALUES %s
            RETURNING id
        """, values)
        
    inserted_ids = [row[0] for row in cursor.fetchall()]
    conn.commit()
    conn.close()
        
    face_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    conn = get_db_connection()
    cursor = conn.cursor()
    return embeddings

@app.route('/find-similar-faces-batch', methods=['POST'])
def find_similar_faces_batch():
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

def cluster_faces_with_hdbscan(embeddings, min_cluster_size=2, min_samples=1):
    """Cluster new face embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embeddings)
    return labels

def load_existing_embeddings(client, BUCKET_NAME, dtype):
    """Load all face embeddings already in MinIO metadata."""
    embeddings = []
    try:
        for obj in client.list_objects(BUCKET_NAME, prefix="faces/"):
            stat = client.stat_object(BUCKET_NAME, obj.object_name)
            meta = stat.metadata
            if "x-amz-meta-encoding_b64" in meta:
                try:
                    emb_bytes = base64.b64decode(meta["x-amz-meta-encoding_b64"])
                    emb = np.frombuffer(emb_bytes, dtype=dtype)
                    embeddings.append(emb)
                except Exception as decode_err:
                    print(f"Skipping {obj.object_name} — invalid metadata: {decode_err}")
    except S3Error as e:
        print(f"Error loading existing embeddings: {e}")
    return embeddings

def upload_unique_faces(client, BUCKET_NAME, img_name, embeddings, bboxes, coords):
    """
    embeddings: numpy array of shape (n_faces, embedding_dim)
    bboxes: list of bounding boxes for each face
    coords: list of (top, bottom, left, right) for each face
    """
    labels = cluster_faces_with_hdbscan(embeddings)
    print(f"Clustered {len(embeddings)} faces into {len(set(labels))} clusters")

    existing_embeddings = load_existing_embeddings(client, BUCKET_NAME, dtype=embeddings[0].dtype)
    print(f"Loaded {len(existing_embeddings)} existing embeddings from MinIO")

    for idx, (emb, bbox, label, (top, bottom, left, right)) in enumerate(zip(embeddings, bboxes, labels, coords)):
        # Compare against existing embeddings
        is_duplicate = False
        for existing_emb in existing_embeddings:
            if is_similar(emb, existing_emb):
                is_duplicate = True
                print(f"Face {idx} is similar to existing one — skipping upload.")
                break

        if is_duplicate:
            continue  # skip only this one

        try:
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

            # Encode embedding as base64 for MinIO metadata
            encoding_str = base64.b64encode(emb.tobytes()).decode("utf-8")

            # Define unique key & metadata
            face_key = f"faces/{label}_{img_name.replace('/', '_')}_{idx}.jpg"
            face_meta = {
                "source_image": img_name,
                "face_index": str(idx),
                "bbox": json.dumps(bbox),
                "cluster_label": str(label),
                "encoding_b64": encoding_str[:2000],
            }

            # Upload
            client.put_object(
                BUCKET_NAME,
                face_key,
                buf,
                length=buf.getbuffer().nbytes,
                content_type="image/jpeg",
                metadata=face_meta,
            )

            # Add this new embedding to prevent duplicates in the same run
            existing_embeddings.append(emb)
            print(f"✅ Uploaded new unique face for cluster {label}: {face_key}")

        except S3Error as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)