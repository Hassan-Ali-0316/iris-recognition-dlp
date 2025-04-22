import cv2
import numpy as np
import tensorflow as tf
import json
import os
from src.preprocessing import normalize_image

model = tf.keras.models.load_model('models/iris_model.h5')

with open('utils/label_map.json','r') as f:
    label_map = json.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def predict_identity(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f'Could not read image: {img_path}')
        return 'Unknown'
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No face detected.")
        return "Unknown"
    
    # Process the first detected face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    # Detect eyes inside the face
    eyes = eye_cascade.detectMultiScale(face_roi)
    if len(eyes) == 0:
        print("No eyes detected inside face.")
        return "Unknown"
    
    # Take the first detected eye
    (ex, ey, ew, eh) = eyes[0]
    eye_img = face_roi[ey:ey+eh, ex:ex+ew]

    # Resize and normalize
    eye_img = cv2.resize(eye_img, (128, 128))
    eye_img = normalize_image(eye_img)
    eye_img = eye_img.reshape(1, 128, 128, 1)  # (batch_size, height, width, channels)

    # Predict
    predictions = model.predict(eye_img)
    predicted_label = np.argmax(predictions)

    confidence = np.max(predictions)

    if confidence < 0.70:  # Threshold for low confidence
        return "Unknown"

    person_name = label_map[str(predicted_label)]
    return person_name

