import os
import json
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import register_keras_serializable

from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))


def normalize_image(image):
    return image / 255.0


def load_processed_dataset(processed_dir):
    X = []
    y = []
    label_map = {}
    label_counter = 0

    for person_name in os.listdir(processed_dir):
        person_path = os.path.join(processed_dir, person_name)
        if os.path.isdir(person_path):
            label_map[label_counter] = person_name
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = normalize_image(img)
                X.append(img)
                y.append(label_counter)
            label_counter += 1

    with open("../utils/label_map.json", "w") as f:
        json.dump(label_map, f)

    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)
    return X, y, label_map


def build_base_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    return Model(inputs, x, name="base_network")


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(X, y):
    pairs = []
    labels = []

    y = np.array(y)
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    for class_id in unique_classes:
        indices = class_indices[class_id]
        if len(indices) < 2:
            continue  
        for i in range(len(indices) - 1):
            a, b = X[indices[i]], X[indices[i + 1]]
            pairs.append([a, b])
            labels.append(1)
            other_classes = [
                cls
                for cls in unique_classes
                if cls != class_id and len(class_indices[cls]) > 0
            ]
            if not other_classes:
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
    return [pairs[:, 0], pairs[:, 1]], labels
def save_embeddings(
    base_network, X, y, label_map, output_path="embeddings/embeddings.npz"
):
    os.makedirs("embeddings", exist_ok=True)
    embeddings = base_network.predict(X)
    np.savez(output_path, embeddings=embeddings, labels=y)

    # Save human-readable label map
    with open("embeddings/label_map.json", "w") as f:
        json.dump(label_map, f)

    print(
        f"Saved embeddings to {output_path} and label map to embeddings/label_map.json"
    )
def train_model():
    X, y, label_map = load_processed_dataset("../data/processed")
    if len(label_map) < 2:
        print("You need at least 2 people to train a similarity model.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    (pair_train, label_train) = create_pairs(X_train, y_train)
    print("heahs")
    print(len(X_val), len(y_val))
    (pair_val, label_val) = create_pairs(X_val, y_val)
    input_shape = (128, 128, 1)
    base_network = build_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    feat_a = base_network(input_a)
    feat_b = base_network(input_b)
    distance = Lambda(euclidean_distance)([feat_a, feat_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer=Adam())
    history = model.fit(
        [pair_train[0], pair_train[1]],
        label_train,
        validation_data=([pair_val[0], pair_val[1]], label_val),
        batch_size=16,
        epochs=10,
    )
    os.makedirs("models", exist_ok=True)
    model.save("models/siamese_model.keras")
    print("Siamese model trained and saved!")
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Save embeddings for all data
    save_embeddings(base_network, X, y, label_map)


if __name__ == "__main__":
    train_model()
