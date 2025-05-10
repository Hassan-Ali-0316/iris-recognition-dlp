import os
import json
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

# Define output directories
MODEL_DIR = "src/models"
EMBEDDINGS_DIR = "embeddings"
PROCESSED_DIR = "data/processed"

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@register_keras_serializable()
def euclidean_distance(vectors):
    """Calculate Euclidean distance between two vectors"""
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def normalize_image(image):
    """Normalize image to 0-1 range"""
    return image / 255.0

def load_processed_dataset(processed_dir=PROCESSED_DIR):
    """Load all processed images from the dataset directory"""
    X = []
    y = []
    label_map = {}
    label_counter = 0

    # Check if directory exists
    if not os.path.exists(processed_dir):
        print(f"Directory not found: {processed_dir}")
        return np.array([]), np.array([]), {}

    # Get list of person directories
    person_dirs = [d for d in os.listdir(processed_dir) 
                   if os.path.isdir(os.path.join(processed_dir, d))]
    
    if not person_dirs:
        print(f"No person directories found in {processed_dir}")
        return np.array([]), np.array([]), {}
    
    print(f"Found {len(person_dirs)} person directories")
    
    for person_name in person_dirs:
        person_path = os.path.join(processed_dir, person_name)
        if os.path.isdir(person_path):
            person_images = [f for f in os.listdir(person_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not person_images:
                print(f"No images found for {person_name}")
                continue
                
            print(f"Loading {len(person_images)} images for {person_name}")
            label_map[str(label_counter)] = person_name  # Store as string keys for JSON compatibility
            
            for img_name in person_images:
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = normalize_image(img)
                X.append(img)
                y.append(label_counter)
            label_counter += 1

    # Save label map for later use
    label_map_path = os.path.join(EMBEDDINGS_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)
    print(f"Saved label map to {label_map_path}")

    if not X:
        print("No valid images found")
        return np.array([]), np.array([]), {}
        
    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)
    print(f"Dataset loaded: {X.shape[0]} images, {len(label_map)} classes")
    return X, y, label_map

def build_base_network(input_shape=(128, 128, 1)):
    """Build base network for feature extraction"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    return Model(inputs, x, name="base_network")

@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    """Contrastive loss function for Siamese network"""
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(X, y):
    """Create positive and negative pairs for training"""
    pairs = []
    labels = []

    y = np.array(y)
    unique_classes = np.unique(y)
    
    if len(unique_classes) < 2:
        raise ValueError("Need at least 2 classes to create pairs")
        
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    # Count total samples per class
    for cls, indices in class_indices.items():
        print(f"Class {cls}: {len(indices)} samples")

    # Create pairs
    for class_id in unique_classes:
        indices = class_indices[class_id]
        if len(indices) < 2:
            print(f"Skipping class {class_id} - not enough samples")
            continue
            
        # Create positive pairs (same class)
        for i in range(len(indices) - 1):
            a, b = X[indices[i]], X[indices[i + 1]]
            pairs.append([a, b])
            labels.append(1)
            
            # Create negative pairs (different classes)
            other_classes = [
                cls
                for cls in unique_classes
                if cls != class_id and len(class_indices[cls]) > 0
            ]
            if not other_classes:
                print("Warning: No other classes available for negative pairs")
                continue

            neg_class = random.choice(other_classes)
            c = X[indices[i]]
            d = X[random.choice(class_indices[neg_class])]
            pairs.append([c, d])
            labels.append(0)

    if not pairs:
        raise ValueError(
            "No valid pairs could be created. Check if your dataset has enough samples per class."
        )

    pairs = np.array(pairs)
    labels = np.array(labels).astype("float32")
    print(f"Created {len(pairs)} pairs ({np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative)")
    return [pairs[:, 0], pairs[:, 1]], labels

def save_embeddings(base_network, X, y, label_map, output_path=os.path.join(EMBEDDINGS_DIR, "embeddings.npz")):
    """Save embeddings and label map to disk"""
    if len(X) == 0:
        print("No data to create embeddings from")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating embeddings for {len(X)} images...")
    embeddings = base_network.predict(X, verbose=1)
    
    np.savez(output_path, embeddings=embeddings, labels=y)

    # Save human-readable label map
    label_map_path = os.path.join(EMBEDDINGS_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)

    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Saved label map to {label_map_path}")

def train_model():
    """Train the Siamese network"""
    print("Loading dataset...")
    X, y, label_map = load_processed_dataset(PROCESSED_DIR)
    
    if len(X) == 0:
        print("No data available to train the model")
        return
        
    if len(np.unique(y)) < 2:
        print(f"You need at least 2 people to train a similarity model. Found {len(np.unique(y))}.")
        return

    print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Creating training pairs...")
    try:
        pair_train, label_train = create_pairs(X_train, y_train)
        print("Creating validation pairs...")
        pair_val, label_val = create_pairs(X_val, y_val)
    except ValueError as e:
        print(f"Error creating pairs: {e}")
        return
    
    print("Building model...")
    input_shape = (128, 128, 1)
    base_network = build_base_network(input_shape)
    
    # Create siamese network
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    feat_a = base_network(input_a)
    feat_b = base_network(input_b)
    distance = Lambda(euclidean_distance)([feat_a, feat_b])
    
    model = Model(inputs=[input_a, input_b], outputs=distance)
    
    # Compile model
    print("Compiling model...")
    model.compile(loss=contrastive_loss, optimizer=Adam())
    
    # Train model
    print("Training model...")
    history = model.fit(
        [pair_train[0], pair_train[1]],
        label_train,
        validation_data=([pair_val[0], pair_val[1]], label_val),
        batch_size=16,
        epochs=10,
        verbose=1
    )
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "siamese_model.keras")
    model.save(model_path)
    print(f"Siamese model trained and saved to {model_path}")
    
    # Save model summary
    with open(os.path.join(MODEL_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Save embeddings for all data
    print("Saving embeddings...")
    save_embeddings(base_network, X, y, label_map)
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()