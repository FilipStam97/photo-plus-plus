import base64
from io import BytesIO
import cv2
import face_recognition
from flask import Flask, app, jsonify, request
from minio import Minio
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


app = Flask(__name__)

client = Minio(
    "localhost:9000",              
    access_key="admin",      
    secret_key="admin123",   
    secure=False                  
)
BUCKET_NAME = "test"

def list_all_images():
    print("trazimo slike")
    objects = client.list_objects(BUCKET_NAME, recursive=True)
    # Filter only image files (jpg, png, jpeg)
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
    return face_encodings, face_locations

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
        for idx, (embedding, bbox) in enumerate(zip(face_encodings, face_locations)):
            embeddings.append(embedding)
            metadata.append({
                "image_name": img_name,
                "face_index": idx,
                "bbox": bbox
            })

    if not embeddings:
        return jsonify({"error": "No faces found in bucket"}), 400

    embeddings = np.array(embeddings)

    dbscan = DBSCAN(eps=0.6, min_samples=1, metric="euclidean")
    dbscan.fit(embeddings)
    labels = dbscan.labels_

    clusters = {}
    seen_labels = set()
    for idx, label in enumerate(labels):
        label = int(label)
        if label in seen_labels:
            continue  
        seen_labels.add(label)

        bbox = metadata[idx]["bbox"]
        try:
            response = client.get_object(BUCKET_NAME, metadata[idx]["image_name"])
            image_bytes = BytesIO(response.read())
            image = np.array(Image.open(image_bytes))
            response.close()
            response.release_conn()
            top, right, bottom, left = bbox
            face_crop = image[top:bottom, left:right]
            pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            pil_face.save(buf, format="JPEG")
            buf.seek(0)
            face_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Failed to crop face for {metadata[idx]['image_name']}: {e}")
            continue

        clusters[label] = {
            "image_name": metadata[idx]["image_name"],
            "face_index": metadata[idx]["face_index"],
            "face_base64": face_base64
        }

    return jsonify({"clusters": clusters})

if __name__ == "__main__":
    app.run(debug = True)