import cv2
import face_recognition
from flask import Flask, app, jsonify, request
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


app = Flask(__name__)

def get_face_embeddings(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations) 
    
    return face_encodings

@app.route('/predict-photo', methods=['POST'])
def cluster_faces():
  
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    #convert file to image file 
    image_file = request.files['image_file']
    pillow_image = Image.open(image_file)
    #convert image to 2d array
    image = np.asarray(pillow_image)

    face_embeddings = get_face_embeddings(image)

    #create clusters of faces
    dbscan = DBSCAN(eps=0.6, min_samples=1) 
    dbscan.fit(face_embeddings)
    
    labels = dbscan.labels_
    
    clusters = {}
    #create JSON response with clusters and face indices 
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return jsonify({'clusters': clusters})

if __name__ == "__main__":
    app.run(debug = True)