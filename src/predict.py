import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
<<<<<<< HEAD
from tensorflow.keras.utils import register_keras_serializable

# Define constant paths
MODEL_PATH = "src/models/siamese_model.keras"
EMBEDDINGS_PATH = "embeddings/embeddings.npz"
LABEL_MAP_PATH = "embeddings/label_map.json"

@register_keras_serializable()
def euclidean_distance(vectors):
    """Calculate Euclidean distance between two vectors"""
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    """Contrastive loss function for Siamese network"""
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def normalize_image(image):
    """Normalize image to 0-1 range"""
    return image / 255.0

def load_model_and_base_network():
    """Load the trained model and extract base network"""
    try:
        model = load_model(
            MODEL_PATH, 
            custom_objects={
                "contrastive_loss": contrastive_loss,
                "euclidean_distance": euclidean_distance
            }
        )
        base_network = model.get_layer("base_network")
        return base_network
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_embeddings(path=EMBEDDINGS_PATH):
    """Load stored embeddings from file"""
    
=======
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
>>>>>>> 3f942bb85303d61437080beb9365eb1b055e18b8
