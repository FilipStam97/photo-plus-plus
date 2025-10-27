

import base64
import datetime
from io import BytesIO
import io
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
from psycopg2.extras import RealDictCursor, execute_values, Json

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


def get_face_embeddings_db(bucket_name):
    """
    Izvlači imena fotografija koje već imaju face embeddings u bazi
    
    Args:
        bucket_name: Ime MinIO bucket-a
        
    Returns:
        set: Skup imena slika koje imaju embeddings
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema='public';
    """)
    print(cursor.fetchall())
    cursor.execute("""
        SELECT DISTINCT image_name 
        FROM public.face_embeddings 
        WHERE bucket_name = %s AND image_name IS NOT NULL
    """, (bucket_name,))
    
    existing_images = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    return existing_images

def get_faces_minio(bucket_name):
    """
    Izvlači sve face embeddings iz MinIO faces/ foldera
    
    Args:
        bucket_name: Ime MinIO bucket-a
        
    Returns:
        list: Lista dict-ova sa face embeddingima i metapodacima
    """
    faces = []
    
    try:
        # Lista svih objekata u faces/ folderu
        objects = client.list_objects(bucket_name, prefix='faces/', recursive=True)
        
        for obj in objects:
            # Preskačemo foldere i fajlove koji nisu slike
            if obj.object_name.endswith('/') or not obj.object_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Učitaj metadata
            stat = client.stat_object(bucket_name, obj.object_name)
            metadata = stat.metadata
            
            if 'x-amz-meta-embedding' in metadata:
                try:
                    embedding = json.loads(metadata['x-amz-meta-embedding'])
                    
                    faces.append({
                        'face_name': obj.object_name,
                        'embedding': np.array(embedding),
                        'cluster_id': metadata.get('x-amz-meta-cluster-id'),
                        'source_image': metadata.get('x-amz-meta-source-image'),
                        'bbox': json.loads(metadata.get('x-amz-meta-bbox', '{}')),
                    })
                except json.JSONDecodeError:
                    print(f"Failed to parse metadata for {obj.object_name}")
                    continue
    
    except S3Error as e:
        print(f"Error reading faces from MinIO: {e}")
    
    return faces

def insert_face_embeddings(conn, embeddings_data):
    """
    Upisuje nove face embeddings u PostgreSQL
    
    Args:
        conn: Database connection
        embeddings_data: Lista dict-ova sa podacima o embeddingima
            [{
                'bucket_name': str,
                'image_name': str,
                'embedding': json,
                'bbox': json,
                'cluster_id': int or None
            }]
    
    Returns:
        list: Lista ID-jeva upisanih face embeddings
    """
    cursor = conn.cursor()
    
    values = []

    print("embeddings: ",embeddings_data)
    for data in embeddings_data:
        values.append((
            data['bucket_name'],
            data['image_name'],
            Json(data['embedding']),
            Json(data['bbox']),
            data.get('cluster_id')
        ))
    execute_values(cursor, """
        INSERT INTO face_embeddings 
        (bucket_name, image_name, embedding, bbox, cluster_id)
        VALUES %s
        RETURNING id
    """, values)
    
    inserted_ids = [row[0] for row in cursor.fetchall()]
    conn.commit()
    print("saved faces.")
    
    return inserted_ids


def insert_faces(conn, bucket_name, new_faces_data, existing_faces_embeddings, tolerance=0.6):
    """
    Upisuje nove representative face slike u MinIO i bazu
    Proverava da li slično lice već postoji
    
    Args:
        conn: Database connection
        bucket_name: MinIO bucket ime
        new_faces_data: Lista novih lica za proveru
            [{
                'image_name': str,
                'face_image': PIL.Image,
                'embedding': np.array,
                'bbox': dict,
                'cluster_id': int
            }]
        existing_faces_embeddings: Lista postojećih face embeddings iz MinIO
        tolerance: Prag za prepoznavanje sličnih lica (default 0.6)
        
    Returns:
        dict: Statistika o upisanim licima
    """
    saved_faces = []
    skipped_faces = 0
    
    for face_data in new_faces_data:
        print("face",face_data)
        new_embedding = face_data['embedding']
        is_new_face = True
        
        # Proveri da li slično lice već postoji
        if existing_faces_embeddings:
            existing_embeddings = np.array([f['embedding'] for f in existing_faces_embeddings])
            
            # Uporedi sa svim postojećim licima
            matches = face_recognition.compare_faces(
                existing_embeddings, 
                new_embedding, 
                tolerance=tolerance
            )
            
            if any(matches):
                is_new_face = False
                skipped_faces += 1

        if not is_new_face:
            continue
        if is_new_face:
            # Generiši jedinstveno ime za face sliku
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            cluster_id = face_data.get('cluster_id', 'unknown')
            face_filename = f"faces/face_{cluster_id}_{timestamp}.jpg"
            
            # Konvertuj PIL Image u bytes
            img_byte_arr = io.BytesIO()
            face_data['face_image'].save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Pripremi metadata
            metadata = {
                'embedding': json.dumps(new_embedding.tolist()),
                'cluster-id': str(cluster_id),
                'image_name': face_data['image_name'],
                'bbox': json.dumps(face_data['bbox'])
            }
            
            try:
                # Upload u MinIO
                client.put_object(
                    bucket_name,
                    face_filename,
                    img_byte_arr,
                    length=img_byte_arr.getbuffer().nbytes,
                    content_type='image/jpeg',
                    metadata=metadata
                )
                
                # Sačuvaj u bazu kao representative face
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO face_embeddings 
                    (bucket_name, image_name, embedding, bbox, cluster_id, is_representative, face_image_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    bucket_name,
                    face_data['image_name'],
                    json(new_embedding.tolist()),
                    json(face_data['bbox']),
                    cluster_id if cluster_id != 'unknown' else None,
                    True,
                    face_filename
                ))
                
                face_id = cursor.fetchone()[0]
                conn.commit()
                
                saved_faces.append({
                    'face_id': face_id,
                    'face_path': face_filename,
                    'cluster_id': cluster_id
                })
                
                # Dodaj u listu postojećih za sledeću iteraciju
                existing_faces_embeddings.append({
                    'embedding': new_embedding,
                    'face_name': face_filename
                })
                
            except S3Error as e:
                print(f"Error uploading face to MinIO: {e}")
                continue
    
    return {
        'saved': len(saved_faces),
        'skipped': skipped_faces,
        'faces': saved_faces
    }

def cluster_images(bucket_name, min_cluster_size=2, tolerance=0.6):
    """
    Glavna funkcija koja:
    1. Učitava sve slike iz bucket-a
    2. Izvlači face embeddings samo za nove slike
    3. Klasterizuje lica pomoću HDBSCAN
    4. Čuva embeddings u bazu
    5. Čuva representative faces u MinIO
    
    Args:
        bucket_name: MinIO bucket ime
        min_cluster_size: Minimalan broj lica za klaster
        tolerance: Prag za prepoznavanje sličnih lica
        
    Returns:
        dict: Rezultati klasterizacije
    """
    conn = get_db_connection()
    
    try:
        # 1. Izvuci imena slika koje vec imaju embeddings
        print("Checking existing face embeddings in database...")
        existing_images = get_face_embeddings_db(bucket_name)
        print(f"Found {len(existing_images)} images with existing embeddings")
        
        # 2. Izvuci sve slike iz bucket-a
        print("Fetching all images from MinIO...")
        all_objects = client.list_objects(bucket_name, recursive=True)
        
        new_images = []
        for obj in all_objects:
            # Preskoci faces/ folder i fajlove koji nisu slike
            if obj.object_name.startswith('faces/') or obj.object_name.endswith('/'):
                continue
            
            if not obj.object_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            # Dodaj samo nove slike
            if obj.object_name not in existing_images:
                new_images.append(obj.object_name)
        
        print(f"Found {len(new_images)} new images to process")
        
        if not new_images:
            return {
                'success': True,
                'message': 'No new images to process',
                'stats': {
                    'new_images': 0,
                    'faces_detected': 0,
                    'faces_saved': 0,
                    'clusters': 0
                }
            }
        
        # 3. Izvuci face embeddings za nove slike
        print("Extracting face embeddings from new images...")
        all_faces = []
        
        for image_name in new_images:
            try:
                # Preuzmi sliku sa MinIO
                response = client.get_object(bucket_name, image_name)
                image_data = response.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Konvertuj u RGB ako nije
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Konvertuj PIL u numpy array
                image_array = np.array(image)
                
                # Detektuj lica
                face_locations = face_recognition.face_locations(image_array)
                face_encodings = face_recognition.face_encodings(image_array, face_locations)
                
                for encoding, location in zip(face_encodings, face_locations):
                    top, right, bottom, left = location
                    
                    # Iseci lice iz slike
                    face_image = image.crop((left, top, right, bottom))
                    
                    all_faces.append({
                        'image_name': image_name,
                        'encoding': encoding,
                        'face_image': face_image,
                        'bbox': {
                            'x': left,
                            'y': top,
                            'width': right - left,
                            'height': bottom - top
                        }
                    })
                
                response.close()
                response.release_conn()
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        print(f"Detected {len(all_faces)} faces in new images")
        
        if not all_faces:
            return {
                'success': True,
                'message': 'No faces detected in new images',
                'stats': {
                    'new_images': len(new_images),
                    'faces_detected': 0,
                    'faces_saved': 0,
                    'clusters': 0
                }
            }
        
        # 4. Klasterizacija pomocu HDBSCAN
        print("Clustering faces...")
        encodings = np.array([face['encoding'] for face in all_faces])
        num_points = len(encodings)

        if num_points == 0:
            cluster_labels = []
        elif num_points == 1:
            cluster_labels = [0]  # single cluster for one point
        else:
            cluster_size = min(2, num_points)
            clusterer = hdbscan.HDBSCAN(cluster_size)
            cluster_labels = clusterer.fit_predict(encodings)
                
        # Dodaj cluster ID svakom licu
        for face, cluster_id in zip(all_faces, cluster_labels):
            face['cluster_id'] = int(cluster_id) if cluster_id != -1 else None
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {num_clusters} clusters")
        
        # 5. Sacuvaj face embeddings u bazu
        print("Saving face embeddings to database...")
        embeddings_data = [{
            'bucket_name': bucket_name,
            'image_name': face['image_name'],
            'embedding': face['encoding'].tolist(),
            'bbox': face['bbox'],
            'cluster_id': face['cluster_id']
        } for face in all_faces]
        print('krece')
        insert_face_embeddings(conn, embeddings_data)
        
        # 6. Izvuci postojece representative faces iz MinIO
        print("Loading existing representative faces from MinIO...")
        existing_faces = get_faces_minio(bucket_name)
        print(len(existing_faces), " postojecig faca ")
        # 7. Grupiši lica po klasterima i pronađi representative faces
        print("Finding representative faces for each cluster...")
        clusters = {}
        noise_faces = []
        
        for face in all_faces:
            cluster_id = face.get('cluster_id')
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(face)
            else:
                noise_faces.append(face)
        
        # Uzmi po jedno reprezentativno lice iz svakog klastera
        representative_faces = []
        for cluster_id, faces in clusters.items():
            if faces:
                # Možeš uzeti prvo lice ili lice najbliže centroidu
                representative_faces.append(faces[0])

        face_stats = {
            'saved': 0,
            'skipped': 0,
            'faces': 0
        }
        if len(representative_faces) > 0:
            face_stats = insert_faces(
                conn,
                bucket_name,
                representative_faces,
                existing_faces,
                tolerance=tolerance
            )
        
        conn.close()
        
        return {
            'success': True,
            'message': 'Clustering completed successfully',
            'stats': {
                'new_images': len(new_images),
                'faces_detected': len(all_faces),
                'faces_saved': face_stats['saved'],
                'faces_skipped': face_stats['skipped'],
                'clusters': num_clusters,
                'noise_faces': len(noise_faces)
            },
            'representative_faces': face_stats['faces']
        }
        
    except Exception as e:
        conn.close()
        print(f"Error during clustering: {e}")
        raise e
    
@app.route('/api/cluster-images', methods=['POST'])
def api_cluster_images():
    """
    API endpoint za klasterizaciju slika
    
    Expected JSON:
    {
        "bucket_name": "my-photos",
        "min_cluster_size": 2,  # optional
        "tolerance": 0.6  # optional
    }
    """
    try:
        data = request.json
        bucket_name = data.get('bucket_name')
        
        if not bucket_name:
            return jsonify({'error': 'bucket_name is required'}), 400
        
        min_cluster_size = 2
        tolerance = 0.6
        
        result = cluster_images(bucket_name, min_cluster_size, tolerance)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/faces/find-similar', methods=['POST'])
def find_similar_faces():
    """
    Pronalazi sve slike sa sličnim licem
    
    Expected JSON:
    {
        "bucket_name": "my-photos",
        "embedding": [0.123, ...],  # 128-dim array
        "tolerance": 0.6  # optional
    }
    """
    try:
        data = request.json
        bucket_name = data.get('bucket_name')
        reference_embedding = np.array(data.get('embedding'))
        tolerance = data.get('tolerance', 0.6)
        
        if not bucket_name or reference_embedding is None:
            return jsonify({'error': 'bucket_name and embedding are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Preuzmi sve embeddings iz baze za taj bucket
        cursor.execute("""
            SELECT 
                id,
                image_name,
                embedding,
                bbox,
                cluster_id,
                is_representative,
                face_image_path
            FROM face_embeddings
            WHERE bucket_name = %s
        """, (bucket_name,))
        
        all_faces = cursor.fetchall()
        conn.close()
        
        matches = []
        
        for face in all_faces:
            stored_embedding = np.array(face['embedding'])
            
            is_match = face_recognition.compare_faces(
                [stored_embedding],
                reference_embedding,
                tolerance=tolerance
            )[0]
            
            if is_match:
                distance = face_recognition.face_distance(
                    [stored_embedding],
                    reference_embedding
                )[0]
                
                matches.append({
                    'face_id': face['id'],
                    'image_name': face['image_name'],
                    'bbox': face['bbox'],
                    'cluster_id': face['cluster_id'],
                    'is_representative': face['is_representative'],
                    'face_image_path': face['face_image_path'],
                    'distance': float(distance)
                })
        
        # Sortiraj po distance
        matches.sort(key=lambda x: x['distance'])
        
        # Grupiši po slikama
        unique_images = {}
        for match in matches:
            img_name = match['image_name']
            if img_name not in unique_images:
                unique_images[img_name] = {
                    'image_name': img_name,
                    'faces': []
                }
            unique_images[img_name]['faces'].append({
                'face_id': match['face_id'],
                'bbox': match['bbox'],
                'distance': match['distance']
            })
        
        return jsonify({
            'success': True,
            'total_matches': len(matches),
            'total_images': len(unique_images),
            'images': list(unique_images.values())
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/faces/representative', methods=['GET'])
def get_representative_faces():
    """
    Vraća sve representative faces za bucket
    
    Query params:
    - bucket_name: MinIO bucket
    """
    try:
        bucket_name = request.args.get('bucket_name')
        
        if not bucket_name:
            return jsonify({'error': 'bucket_name is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id,
                image_name,
                embedding,
                bbox,
                cluster_id,
                face_image_path,
                created_at
            FROM face_embeddings
            WHERE bucket_name = %s AND is_representative = true
            ORDER BY cluster_id, created_at DESC
        """, (bucket_name,))
        
        faces = cursor.fetchall()
        conn.close()
        
        # Grupiši po klasterima
        clusters = {}
        for face in faces:
            cluster_id = face['cluster_id'] or 'unclustered'
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append({
                'face_id': face['id'],
                'image_name': face['image_name'],
                'face_image_path': face['face_image_path'],
                'bbox': face['bbox'],
                'embedding': face['embedding']
            })
        
        return jsonify({
            'success': True,
            'bucket_name': bucket_name,
            'total_clusters': len(clusters),
            'clusters': clusters
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def embedding_to_base64(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.tobytes()).decode('utf-8')

def base64_to_embedding(b64_string: str) -> np.ndarray:
    bytes_data = base64.b64decode(b64_string)
    return np.frombuffer(bytes_data, dtype=np.float64)

def list_all_images_folder(bucket_name, client):
    objects = client.list_objects(bucket_name, recursive=True)
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_similar(new_emb, existing_emb, threshold=0.75):
    similarity = cosine_similarity(new_emb, existing_emb)
    return similarity >= threshold

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)