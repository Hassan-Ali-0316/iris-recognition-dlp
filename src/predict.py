import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocessing import normalize_image


def build_base_network(input_shape=(128, 128, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    return tf.keras.models.Model(inputs, x)


def load_embeddings(path="embeddings/embeddings.npz"):
    data = np.load(path)
    return data["embeddings"], data["labels"]


def load_label_map(path="embeddings/label_map.json"):
    with open(path, "r") as f:
        return json.load(f)


def predict_identity(img_path, threshold=0.5):
    base_network = build_base_network()
    base_network.load_weights("models/siamese_model.keras", by_name=True)
    embeddings, labels = load_embeddings()
    label_map = load_label_map()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return "Unknown"
    img = normalize_image(cv2.resize(img, (128, 128)))
    img = img.reshape(1, 128, 128, 1)
    query_embedding = base_network.predict(img)
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    if min_distance > threshold:
        return "Unknown"

    label_id = int(labels[min_index])
    return label_map[str(label_id)]
