

import base64
import datetime
from io import BytesIO
import io
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
import threading
import logging

logger = logging.getLogger(__name__)

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
    Vraća samo putanje i cluster_id, embedding se čita iz baze
    
    Args:
        bucket_name: Ime MinIO bucket-a
        
    Returns:
        list: Lista dict-ova sa putanjama i cluster_id
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
            
            faces.append({
                'face_path': obj.object_name,
                'cluster_id': int(metadata.get('x-amz-meta-cluster-id', -1)),
                'source_image': metadata.get('x-amz-meta-source-image', ''),
            })
    
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


def insert_faces(conn, bucket_name, new_faces_data, existing_cluster_embeddings, tolerance=0.6):
    """
    Upisuje nove representative face slike u MinIO i bazu
    Proverava da li novi klaster već postoji među postojećim klasterima
    
    Args:
        conn: Database connection
        bucket_name: MinIO bucket ime
        new_faces_data: Lista novih lica za proveru
            [{
                'image_name': str,
                'face_image': PIL.Image (already cropped),
                'embedding': np.array,
                'bbox': dict,
                'cluster_id': int
            }]
        existing_cluster_embeddings: Dict {cluster_id: [embeddings]} postojećih klastera
        tolerance: Prag za prepoznavanje sličnih lica (default 0.6)
        
    Returns:
        dict: Statistika o upisanim licima
    """
    saved_faces = []
    skipped_faces = 0
    cluster_mapping = {}  # new_cluster_id -> existing_cluster_id
    
    for face_data in new_faces_data:
        print(face_data)
        new_embedding = face_data['embedding']
        new_cluster_id = face_data.get('cluster_id')
        
        # Proveri da li ovaj novi klaster odgovara nekom postojećem klasteru
        matched_existing_cluster = None
        
        if existing_cluster_embeddings:
            for existing_cluster_id, embeddings in existing_cluster_embeddings.items():
                # Uporedi sa svim embeddingima iz tog postojećeg klastera
                matches = face_recognition.compare_faces(
                    embeddings,
                    new_embedding,
                    tolerance=tolerance
                )
                
                if any(matches):
                    matched_existing_cluster = existing_cluster_id
                    cluster_mapping[new_cluster_id] = existing_cluster_id
                    break
        
        # Ako se poklapa sa postojećim klasterom, preskači
        if matched_existing_cluster is not None:
            skipped_faces += 1
            continue
        
        # Ovo je novo lice - sačuvaj ga
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        face_filename = f"faces/face_{new_cluster_id}_{timestamp}.jpg"
        
        # Konvertuj PIL Image u bytes (slika je već cropovana)
        img_byte_arr = io.BytesIO()
        face_data['face_image'].save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Pripremi metadata (bez embeddinga!)
        metadata = {
            'cluster-id': str(new_cluster_id),
            'source-image': face_data['image_name'],
            'bbox-x': str(face_data['bbox']['x']),
            'bbox-y': str(face_data['bbox']['y']),
            'bbox-width': str(face_data['bbox']['width']),
            'bbox-height': str(face_data['bbox']['height'])
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
                Json(new_embedding.tolist()),
                Json(face_data['bbox']),
                new_cluster_id,
                True,
                face_filename
            ))
            
            face_id = cursor.fetchone()[0]
            conn.commit()
            
            saved_faces.append({
                'face_id': face_id,
                'face_path': face_filename,
                'cluster_id': new_cluster_id
            })
            
            # Dodaj u postojeće klastere za sledeću iteraciju
            if new_cluster_id not in existing_cluster_embeddings:
                existing_cluster_embeddings[new_cluster_id] = []
            existing_cluster_embeddings[new_cluster_id].append(new_embedding)
            
        except S3Error as e:
            print(f"Error uploading face to MinIO: {e}")
            continue
    
    return {
        'saved': len(saved_faces),
        'skipped': skipped_faces,
        'faces': saved_faces,
        'cluster_mapping': cluster_mapping
    }
def update_cluster_ids(conn, bucket_name, cluster_mapping):
    """
    Ažuriraj cluster_id za nova lica koja pripadaju postojećim klasterima
    
    Args:
        conn: Database connection
        bucket_name: Bucket name
        cluster_mapping: Dict {new_cluster_id: existing_cluster_id}
    """
    if not cluster_mapping:
        return
    
    cursor = conn.cursor()
    
    for new_cluster_id, existing_cluster_id in cluster_mapping.items():
        cursor.execute("""
            UPDATE face_embeddings
            SET cluster_id = %s
            WHERE bucket_name = %s 
            AND cluster_id = %s
            AND is_representative = false
        """, (existing_cluster_id, bucket_name, new_cluster_id))
    
    conn.commit()
    print(f"Updated cluster IDs: {cluster_mapping}")

def get_representative_embeddings_from_db(conn, bucket_name):
    """
    Učitaj sve representative face embeddings iz baze
    
    Returns:
        dict: {cluster_id: [embeddings]}
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT cluster_id, embedding
        FROM face_embeddings
        WHERE bucket_name = %s 
        AND is_representative = true
        AND cluster_id IS NOT NULL
    """, (bucket_name,))
    
    rows = cursor.fetchall()
    
    # Grupiši embeddings po cluster_id
    cluster_embeddings = {}
    for row in rows:
        cluster_id = row['cluster_id']
        embedding = np.array(row['embedding'])
        
        if cluster_id not in cluster_embeddings:
            cluster_embeddings[cluster_id] = []
        cluster_embeddings[cluster_id].append(embedding)
    
    return cluster_embeddings
def cluster_images(bucket_name, min_cluster_size=2, tolerance=0.6):
    """
    Glavna funkcija koja:
    1. Učitava sve slike iz bucket-a
    2. Izvlači face embeddings samo za nove slike
    3. Učitava postojeće embeddings iz baze
    4. Klasterizuje SVA lica (nova + postojeća) pomoću HDBSCAN
    5. Čuva nove embeddings u bazu
    6. Čuva nove representative faces u MinIO
    """
    conn = get_db_connection()
    
    try:
        # 1. Izvuci imena slika koje vec imaju embeddings
        print("Checking existing face embeddings in database...")
        existing_images = get_face_embeddings_db(bucket_name)
        print(f"Found {len(existing_images)} images with existing embeddings")
        
        # 2. Učitaj postojeće embeddings iz baze
        print("Loading existing face embeddings from database...")
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT image_name, embedding, bbox, cluster_id
            FROM face_embeddings
            WHERE bucket_name = %s
        """, (bucket_name,))
        
        existing_face_data = cursor.fetchall()
        print(f"Found {len(existing_face_data)} existing face embeddings")
        
        # 3. Izvuci sve slike iz bucket-a
        print("Fetching all images from MinIO...", bucket_name)
        all_objects = client.list_objects(bucket_name, recursive=True)
        
        new_images = []
        for obj in all_objects:
            if obj.object_name.startswith('faces/') or obj.object_name.endswith('/'):
                continue
            
            if obj.object_name not in existing_images:
                new_images.append(obj.object_name)
        
        print(f"Found {len(new_images)} new images to process")
        
        # 4. Izvuci face embeddings za nove slike
        print("Extracting face embeddings from new images...")
        new_faces = []
        
        for image_name in new_images:
            try:
                response = client.get_object(bucket_name, image_name)
                image_data = response.read()
                response.close()
                response.release_conn()

                # Decode image from bytes using OpenCV (works even without extension)
                np_bytes = np.frombuffer(image_data, np.uint8)
                image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    print(f"Could not decode image {image_name}")
                    continue

                # Convert BGR → RGB (face_recognition expects RGB)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Detect faces and compute embeddings
                face_locations = face_recognition.face_locations(image_rgb)
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

                if not face_encodings:
                    print(f"No faces found in {image_name}")
                    continue

                # For each detected face
                for encoding, location in zip(face_encodings, face_locations):
                    top, right, bottom, left = location

                    # Crop the face (convert NumPy → PIL for easy cropping)
                    image_pil = Image.fromarray(image_rgb)
                    face_image = image_pil.crop((left, top, right, bottom))

                    new_faces.append({
                        'image_name': image_name,
                        'embedding': encoding,
                        'face_image': face_image,
                        'bbox': {
                            'x': left,
                            'y': top,
                            'width': right - left,
                            'height': bottom - top
                        },
                        'is_new': True
                    })

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                continue
        
        print(f"Detected {len(new_faces)} faces in new images")
        
        # 5. Kombinuj postojeća i nova lica za klasterizaciju
        all_faces_for_clustering = []
        
        # Dodaj postojeća lica
        for face_data in existing_face_data:
            all_faces_for_clustering.append({
                'image_name': face_data['image_name'],
                'embedding': np.array(face_data['embedding']),
                'bbox': face_data['bbox'],
                'is_new': False,
                'old_cluster_id': face_data['cluster_id']
            })
        
        # Dodaj nova lica
        all_faces_for_clustering.extend(new_faces)
        
        print(f"Total faces for clustering: {len(all_faces_for_clustering)}")
        
        if len(all_faces_for_clustering) == 0:
            return {
                'success': True,
                'message': 'No faces to process',
                'stats': {
                    'new_images': 0,
                    'faces_detected': 0,
                    'faces_saved': 0,
                    'clusters': 0
                }
            }
        
        # 6. HDBSCAN klasterizacija na SVIM licima
        print("Clustering all faces with HDBSCAN...")
        encodings = np.array([face['embedding'] for face in all_faces_for_clustering])
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(encodings)
        
        # Dodaj cluster ID svakom licu
        for face, cluster_id in zip(all_faces_for_clustering, cluster_labels):
            face['cluster_id'] = int(cluster_id) if cluster_id != -1 else None
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {num_clusters} clusters")
        
        # 7. Sačuvaj embeddings samo za NOVA lica
        print("Saving new face embeddings to database...")
        new_embeddings_data = [{
            'bucket_name': bucket_name,
            'image_name': face['image_name'],
            'embedding': face['embedding'].tolist(),
            'bbox': face['bbox'],
            'cluster_id': face['cluster_id']
        } for face in all_faces_for_clustering if face['is_new']]
        
        if new_embeddings_data:
            insert_face_embeddings(conn, new_embeddings_data)
        
        # 8. Učitaj postojeće representative embeddings
        print("Loading existing representative faces...")
        existing_cluster_embeddings = get_representative_embeddings_from_db(conn, bucket_name)
        
        # 9. Grupiši nova lica po klasterima
        print("Finding new representative faces...")
        new_clusters = {}
        
        for face in all_faces_for_clustering:
            if not face['is_new']:
                continue
            
            cluster_id = face.get('cluster_id')
            if cluster_id is not None and cluster_id >= 0:
                if cluster_id not in new_clusters:
                    new_clusters[cluster_id] = []
                new_clusters[cluster_id].append(face)
        
        # Uzmi po jedno reprezentativno lice iz svakog NOVOG klastera
        new_representative_faces = []
        for cluster_id, faces in new_clusters.items():
            if faces:
                new_representative_faces.append(faces[0])
        
        print(f"Found {len(new_representative_faces)} potential new representative faces")
        
        # 10. Sačuvaj samo NOVE representative faces
        face_stats = {'saved': 0, 'skipped': 0, 'faces': [], 'cluster_mapping': {}}
        
        if new_representative_faces:
            face_stats = insert_faces(
                conn,
                bucket_name,
                new_representative_faces,
                existing_cluster_embeddings,
                tolerance=tolerance
            )
            
            # Ažuriraj cluster_id za lica koja pripadaju postojećim klasterima
            if face_stats['cluster_mapping']:
                update_cluster_ids(conn, bucket_name, face_stats['cluster_mapping'])
        
        conn.close()
        
        return {
            'success': True,
            'message': 'Clustering completed successfully',
            'stats': {
                'new_images': len(new_images),
                'new_faces_detected': len(new_faces),
                'total_faces_clustered': len(all_faces_for_clustering),
                'faces_saved': face_stats['saved'],
                'faces_skipped': face_stats['skipped'],
                'clusters': num_clusters,
                'new_people': face_stats['saved']
            },
            'representative_faces': face_stats['faces']
        }
        
    except Exception as e:
        conn.close()
        print(f"Error during clustering: {e}")
        raise e
    
def start_cluster_job(bucket_name: str, min_cluster_size: int = 2, tolerance: float = 0.6):
    """
    Runs cluster_images() in its own thread.
    This can be reused by any route or scheduled job.
    """
    def _worker():
        try:
            logger.info(f"Starting cluster_images for bucket={bucket_name}")
            result = cluster_images(bucket_name, min_cluster_size, tolerance)
            logger.info(f"Cluster job done for bucket={bucket_name}: {result}")
        except Exception as e:
            logger.exception(f"Cluster job failed for bucket={bucket_name}: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
    
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

        # fire-and-forget
        # start_cluster_job(bucket_name, min_cluster_size, tolerance)
        try:
            logger.info(f"Starting cluster_images for bucket={bucket_name}")
            result = cluster_images(bucket_name, min_cluster_size, tolerance)
            logger.info(f"Cluster job done for bucket={bucket_name}: {result}")
        except Exception as e:
            logger.exception(f"Cluster job failed for bucket={bucket_name}: {e}")
        
        # 202 Accepted = work started, still processing
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
        face_image_path = data.get('face_image_path')
        tolerance = data.get('tolerance', 0.6)
        
        if not bucket_name or not face_image_path:
            return jsonify({'error': 'bucket_name and face_image_path are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Preuzmi embedding za kliknuto lice iz baze
        cursor.execute("""
            SELECT 
                id,
                embedding,
                cluster_id,
                image_name as source_image
            FROM face_embeddings
            WHERE bucket_name = %s 
            AND face_image_path = %s
            AND is_representative = true
        """, (bucket_name, face_image_path))
        
        clicked_face = cursor.fetchone()
        if not clicked_face:
            conn.close()
            return jsonify({'error': 'Face not found'}), 404
        
        reference_embedding = np.array(clicked_face['embedding'])
        clicked_cluster_id = clicked_face['cluster_id']
        
        print("pokusaj brze pretrage po cluster idju ", clicked_cluster_id)
        # 2. Ako postoji cluster_id, prvo probaj brzu pretragu po cluster_id
        # cursor.execute("""
        #     SELECT 
        #         id,
        #         image_name,
        #         bbox,
        #         cluster_id,
        #         face_image_path
        #     FROM face_embeddings
        #     WHERE bucket_name = %s 
        #     AND cluster_id = %s
        #     AND is_representative = false
        # """, (bucket_name, clicked_cluster_id))
        
        # cluster_faces = cursor.fetchall()
        
        # if cluster_faces:
        #     conn.close()
            
        #     # Grupiši u Python-u
        #     images_dict = {}
        #     for face in cluster_faces:
        #         img_name = face['image_name']
        #         if img_name not in images_dict:
        #             images_dict[img_name] = {
        #                 'image_name': img_name,
        #                 'faces': []
        #             }
        #         images_dict[img_name]['faces'].append({
        #             'face_id': face['id'],
        #             'bbox': face['bbox'],
        #             'cluster_id': face['cluster_id'],
        #             'face_image_path': face['face_image_path']
        #         })
            
        #     images = list(images_dict.values())
        #     return jsonify({
        #         'success': True,
        #         'method': 'cluster',
        #         'cluster_id': clicked_cluster_id,
        #         'total_images': len(images),
        #         'images': images
        #     }), 200
        print("to najverovatnije nije uspelo, tako da cemo sad da probamo uporedjivanje slika...")
        # 3. Ako nema cluster_id ili želiš preciznije rezultate, uporedi embeddings
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
            AND is_representative = false
        """, (bucket_name,))
        
        all_faces = cursor.fetchall()
        conn.close()
        
        matches = []
        
        # Uporedi svaki embedding sa referentnim
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
                matches.append(face['image_name'])
                # matches.append({
                #     'face_id': face['id'],
                #     'image_name': face['image_name'],
                #     'bbox': face['bbox'],
                #     'cluster_id': face['cluster_id'],
                #     'face_image_path': face['face_image_path'],
                #     'distance': float(distance)
                # })
        print("nadjeni matches")
        # Sortiraj po distance (najsličniji prvo)
        # matches.sort(key=lambda x: x['distance'])
        
        # Grupiši po slikama
        unique_images = {}
        # final_images = []
        # for match in matches:
        #     img_name = match['image_name']
            #final_images.append(img_name)
            # if img_name not in unique_images:
            #      unique_images[img_name] = {
            #          'image_name': img_name,
            #         'faces': []
                # }
            # unique_images[img_name]['faces'].append({
            #     'face_id': match['face_id'],
            #     'bbox': match['bbox'],
            #     'distance': match['distance']
            # })
        
        return jsonify({
            'success': True,
            'method': 'embedding_comparison',
            'total_matches': len(matches),
            'total_images': len(unique_images),
            'images': list(matches)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/get-embeddings', methods=['GET'])
def get_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
        
    cursor.execute("""
        SELECT 
            id,
            image_name,
            bbox,
            cluster_id,
            face_image_path,
            created_at,
            is_representative
        FROM face_embeddings
     
    """)
    res = cursor.fetchall()
    conn.close()
    return jsonify({
        'res': res
    })

@app.route('/api/delete-embeddings', methods=['DELETE'])
def delete_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
        
    cursor.execute("""
        TRUNCATE TABLE face_embeddings
    """)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({
        'status': 'ok'
    })

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
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)