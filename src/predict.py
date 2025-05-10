import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
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
    