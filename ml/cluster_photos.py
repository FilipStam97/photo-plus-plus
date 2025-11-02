import base64
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import face_recognition
import hdbscan

from PIL import Image
import io
import os
from minio import Minio
import cv2
import datetime
import uuid
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
# MinIO client
client = Minio(
    os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    access_key=os.getenv('MINIO_ACCESS_KEY', "admin"),
    secret_key=os.getenv('MINIO_SECRET_KEY', "admin123"),
    secure=os.getenv('MINIO_SECURE', 'false').lower() == 'true'
)

# Qdrant client
qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "face_embeddings"

BUCKET_NAME = "test"

# Initialize collection
try:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )
except:
    pass


def get_cluster_centroids(bucket_name):
    """
    Izračunaj centroide za sve postojeće klastere
    Returns: dict {cluster_id: centroid_vector}
    """
    try:
        # Učitaj sve embeddings grupisane po cluster_id
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name))
                ]
            ),
            limit=100000,
            with_vectors=True,
            with_payload=True
        )
        
        if not results:
            return {}
        
        # Grupiši po cluster_id
        clusters = {}
        for point in results:
            cluster_id = point.payload.get('cluster_id')
            if cluster_id is not None and cluster_id >= 0:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(np.array(point.vector))
        
        # Izračunaj centroide
        centroids = {}
        for cluster_id, vectors in clusters.items():
            if len(vectors) > 0:
                centroids[cluster_id] = np.mean(vectors, axis=0)
        
        print(f"Calculated centroids for {len(centroids)} clusters")
        return centroids
        
    except Exception as e:
        print(f"Error calculating centroids: {e}")
        return {}


def assign_to_cluster(embedding, centroids, tolerance=0.6):
    """
    Dodeli novo lice najbližem klasteru ili vrati None ako nije dovoljno slično
    
    Args:
        embedding: numpy array (128,)
        centroids: dict {cluster_id: centroid_vector}
        tolerance: prag sličnosti (0.6 za face_recognition)
    
    Returns:
        (cluster_id, distance) ili (None, None) ako ne odgovara nijednom klasteru
    """
    if not centroids:
        return None, None
    
    best_cluster = None
    best_distance = float('inf')
    
    for cluster_id, centroid in centroids.items():
        # Cosine distance
        similarity = cosine_similarity([embedding], [centroid])[0][0]
        distance = 1 - similarity
        
        if distance < best_distance:
            best_distance = distance
            best_cluster = cluster_id
    
    # Proveri da li je dovoljno slično (tolerance za face_recognition)
    # tolerance 0.6 za Euclidean, za cosine koristimo 1-tolerance
    cosine_threshold = 1 - tolerance
    
    if best_distance < (1 - cosine_threshold):
        return best_cluster, best_distance
    
    return None, None


def insert_face_embeddings(bucket_name, embeddings_data):
    """Upisuje face embeddings u Qdrant"""
    points = []
    
    for data in embeddings_data:
        point_id = str(uuid.uuid4())
        
        points.append(PointStruct(
            id=point_id,
            vector=data['embedding'],
            payload={
                'bucket_name': bucket_name,
                'image_name': data['image_name'],
                'bbox': data['bbox'],
                'cluster_id': data.get('cluster_id'),
                'created_at': datetime.datetime.now().isoformat()
            }
        ))
    
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Saved {len(points)} face embeddings to Qdrant")
    
    return [p.id for p in points]


def check_if_first_run(bucket_name):
    """Proveri da li bucket ima bilo kakve embeddings"""
    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name))]
            ),
            limit=1
        )
        return len(results) == 0
    except:
        return True


def get_existing_images(bucket_name):
    """Dobavi listu slika koje već imaju embeddings"""
    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name))]
            ),
            limit=100000
        )
        return {point.payload['image_name'] for point in results}
    except:
        return set()


def recluster_all_faces(bucket_name, min_cluster_size=2):
    """Re-klasterizuj SVA lica u bucket-u"""
    print(" Re-clustering all faces...")
    
    try:
        # Učitaj SVE embeddings
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name))]
            ),
            limit=100000,
            with_vectors=True,
            with_payload=True
        )
        
        if len(results) == 0:
            print("No faces to recluster")
            return 0
        
        print(f"Re-clustering {len(results)} faces...")
        
        # HDBSCAN
        embeddings = np.array([point.vector for point in results])
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {num_clusters} clusters")
        
        # Update Qdrant sa novim cluster_id
        updates = []
        for point, cluster_id in zip(results, cluster_labels):
            new_cluster_id = int(cluster_id) if cluster_id != -1 else None
            
            payload = point.payload.copy()
            payload['cluster_id'] = new_cluster_id
            
            updates.append(PointStruct(
                id=point.id,
                vector=point.vector,
                payload=payload
            ))
        
        if updates:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=updates)
            print(f" Updated {len(updates)} embeddings with new cluster IDs")
        
        return num_clusters
        
    except Exception as e:
        print(f" Error during re-clustering: {e}")
        raise e


def cluster_images(bucket_name, min_cluster_size=2, tolerance=0.6):
    """
    Glavna funkcija za klasterizaciju:
    1. Prvi put: HDBSCAN na svim licima
    2. Sledeći put: Dodeli nove face najbližem klasteru, recluster samo ako treba
    """
    try:
        is_first_run = check_if_first_run(bucket_name)
        print(f"First run: {is_first_run}")
        
        all_objects = client.list_objects(bucket_name)
        # 1. Izvuci postojeće slike
        existing_images = get_existing_images(bucket_name)
        existing_set = set(existing_images)
        print(f"Found {len(existing_images)} images with embeddings")
        
        new_images = [
            obj.object_name
            for obj in all_objects
            if not obj.object_name.startswith('faces/') and not obj.object_name.endswith('/')
            and obj.object_name not in existing_set
        ]

        print(f"Found {len(new_images)} new images to process")
        
        if len(new_images) == 0 and not is_first_run:
            return {
                'success': True,
                'message': 'No new images',
                'stats': {'new_images': 0, 'faces_detected': 0}
            }
        
        # 3. Izvuci embeddings za nove slike
        print("Extracting face embeddings...")
        new_faces = []
        
        for image_name in new_images:
            try:
                response = client.get_object(bucket_name, image_name)
                image_data = response.read()
                response.close()
                response.release_conn()

                np_bytes = np.frombuffer(image_data, np.uint8)
                image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(image_rgb)
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

                for encoding, location in zip(face_encodings, face_locations):
                    top, right, bottom, left = location
                    
                    new_faces.append({
                        'image_name': image_name,
                        'embedding': encoding,
                        'bbox': {
                            'x': left,
                            'y': top,
                            'width': right - left,
                            'height': bottom - top
                        }
                    })

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        print(f"Detected {len(new_faces)} faces")
        
        if len(new_faces) == 0:
            return {
                'success': True,
                'message': 'No faces detected',
                'stats': {'new_images': len(new_images), 'faces_detected': 0}
            }
        
        # 4. PRVI PUT: Klasterizuj sve
        if is_first_run:
            print("INITIAL CLUSTERING")
            
            embeddings = np.array([face['embedding'] for face in new_faces])
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom',
                min_samples=1,
                cluster_selection_epsilon=0.0,
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"Found {num_clusters} clusters")
            
            # Sačuvaj embeddings
            embeddings_data = []
            for face, cluster_id in zip(new_faces, cluster_labels):
                embeddings_data.append({
                    'image_name': face['image_name'],
                    'embedding': face['embedding'].tolist(),
                    'bbox': face['bbox'],
                    'cluster_id': int(cluster_id) if cluster_id != -1 else None
                })
            
            insert_face_embeddings(bucket_name, embeddings_data)
            
            return {
                'success': True,
                'message': 'Initial clustering completed',
                'stats': {
                    'new_images': len(new_images),
                    'faces_detected': len(new_faces),
                    'clusters': num_clusters
                }
            }
        
        # 5. NIJE PRVI PUT: Incremental clustering
        else:
            print(" INCREMENTAL CLUSTERING")
            
            # Dobavi centroide postojećih klastera
            centroids = get_cluster_centroids(bucket_name)
            print(f"Working with {len(centroids)} existing clusters")
            
            # Dodeli nova lica klasterima
            assigned_faces = []
            unassigned_faces = []
            
            for face in new_faces:
                cluster_id, distance = assign_to_cluster(
                    face['embedding'],
                    centroids,
                    tolerance
                )
                
                if cluster_id is not None:
                    face['cluster_id'] = cluster_id
                    assigned_faces.append(face)
                    print(f" Assigned to cluster {cluster_id} (distance: {distance:.4f})")
                else:
                    face['cluster_id'] = None
                    unassigned_faces.append(face)
                    print(f" No matching cluster (new person?)")
            
            print(f"Assigned: {len(assigned_faces)}, Unassigned: {len(unassigned_faces)}")
            
            # Sačuvaj SVA nova lica
            embeddings_data = [{
                'image_name': face['image_name'],
                'embedding': face['embedding'].tolist(),
                'bbox': face['bbox'],
                'cluster_id': face['cluster_id']
            } for face in new_faces]
            
            insert_face_embeddings(bucket_name, embeddings_data)
            
            # Ako ima nedodeljenih lica -> RECLUSTER
            needs_reclustering = len(unassigned_faces) > 0
            
            if needs_reclustering:
                print(f"Found {len(unassigned_faces)} unassigned faces - re-clustering all...")
                num_clusters = recluster_all_faces(bucket_name, min_cluster_size)
            else:
                num_clusters = len(centroids)
                print("All faces assigned to existing clusters")
            
            return {
                'success': True,
                'message': 'Incremental clustering completed',
                'stats': {
                    'new_images': len(new_images),
                    'faces_detected': len(new_faces),
                    'assigned_to_existing': len(assigned_faces),
                    'new_people': len(unassigned_faces),
                    'reclustered': needs_reclustering,
                    'clusters': num_clusters
                }
            }
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        raise e


# ============= API ENDPOINTS =============

@app.route('/api/cluster-images', methods=['POST'])
def api_cluster_images():
    """Klasterizuj slike"""
    try:
        data = request.json
        bucket_name = data.get('bucket_name')
        
        print(bucket_name)
        if not bucket_name:
            return jsonify({'error': 'bucket_name is required'}), 400
        
        min_cluster_size = data.get('min_cluster_size', 2)
        tolerance = data.get('tolerance', 0.6)
        
        result = cluster_images(bucket_name, min_cluster_size, tolerance)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/faces/find-similar', methods=['POST'])
def find_similar_faces():
    """
    Pronađi sve slike gde se pojavljuje osoba
    
    Expected JSON:
    {
        "bucket_name": "my-photos",
        "cluster_id": 5  // cluster_id osobe
    }
    """
    try:
        data = request.json
        bucket_name = data.get('bucket_name')
        cluster_id = data.get('cluster_id')
        
        if not bucket_name or cluster_id is None:
            return jsonify({'error': 'bucket_name and cluster_id are required'}), 400
        
        # Pronađi SVA lica u tom klasteru
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name)),
                    FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))
                ]
            ),
            limit=10000
        )
        
        # Grupiši po slikama
        images_dict = {}
        for point in results:
            img_name = point.payload['image_name']
            if img_name not in images_dict:
                images_dict[img_name] = {
                    'image_name': img_name,
                    'faces': []
                }
            images_dict[img_name]['faces'].append({
                'point_id': point.id,
                'bbox': point.payload['bbox']
            })
        
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'total_images': len(images_dict),
            'images': list(images_dict.values())
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/faces/clusters', methods=['GET'])
def get_all_clusters():
    """
    Dobavi sve klastere (people) sa po jednom sample slikom
    
    Query params:
    - bucket_name: MinIO bucket
    """
    try:
        bucket_name = request.args.get('bucket_name')
        
        if not bucket_name:
            return jsonify({'error': 'bucket_name is required'}), 400
        
        # Učitaj sve embeddings
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="bucket_name", match=MatchValue(value=bucket_name))]
            ),
            limit=100000
        )
        
        # Grupiši po cluster_id
        clusters = {}
        for point in results:
            cluster_id = point.payload.get('cluster_id')
            if cluster_id is not None and cluster_id >= 0:
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'cluster_id': cluster_id,
                        'face_count': 0,
                        'sample_image': None,
                        'sample_bbox': None,
                        'sample_image': None,
                    }
                
                clusters[cluster_id]['face_count'] += 1
                
                # Uzmi prvu sliku kao sample
                if clusters[cluster_id]['sample_image'] is None:
                    image_name = point.payload['image_name']
                    bbox = point.payload['bbox']  # assuming [x_min, y_min, x_max, y_max]
                    
                    response = client.get_object(bucket_name, image_name)
                    img_data = response.read()
                    img = Image.open(io.BytesIO(img_data))
                    
                    x_min = float(bbox['x'])
                    y_min = float(bbox['y'])
                    width = float(bbox['width'])
                    height = float(bbox['height'])
                    x_max = x_min + width
                    y_max = y_min + height

                    cropped = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                    if cropped.mode in ("RGBA", "LA"):
                        cropped = cropped.convert("RGB")
                    # Convert to base64
                    buffered = io.BytesIO()
                    cropped.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    clusters[cluster_id]['sample_image'] = image_name
                    clusters[cluster_id]['sample_bbox'] = bbox
                    clusters[cluster_id]['cropped_face'] = img_str
        
        return jsonify({
            'success': True,
            'bucket_name': bucket_name,
            'total_clusters': len(clusters),
            'clusters': list(clusters.values())
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
