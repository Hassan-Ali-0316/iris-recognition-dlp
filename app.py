import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocessing import normalize_image, save_processed_image

from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


# === Load model and base network ===
model = load_model(
    "./src/models/siamese_model.keras", custom_objects={"contrastive_loss": contrastive_loss}
)
base_network = model.get_layer("base_network")

# === Load embeddings ===
emb_path = "embeddings/embeddings.npz"
label_map_path = "embeddings/label_map.json"

if os.path.exists(emb_path):
    emb_data = np.load(emb_path)
    stored_embeddings = emb_data["embeddings"]
    stored_labels = emb_data["labels"]
else:
    stored_embeddings = np.empty((0, 4096))  # Adjust if needed
    stored_labels = np.empty((0,))

if os.path.exists(label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
else:
    label_map = {}

label_counter = max([int(k) for k in label_map.keys()], default=-1) + 1


def save_embeddings():
    np.savez(emb_path, embeddings=stored_embeddings, labels=stored_labels)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)


def get_identity_from_embedding(embedding, threshold=0.5):
    if len(stored_embeddings) == 0:
        return "Unknown", float("inf")

    distances = np.linalg.norm(stored_embeddings - embedding, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    if min_distance > threshold:
        return "Unknown", min_distance

    predicted_label = int(stored_labels[min_index])
    return label_map.get(str(predicted_label), "Unknown"), min_distance


# === Signup (Upload Images) ===
def upload_and_register():
    global stored_embeddings, stored_labels, label_map, label_counter

    person_name = simpledialog.askstring("Signup", "Enter the person's name:")
    if not person_name:
        return

    files = filedialog.askopenfilenames(
        title="Select eye images",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")),
    )
    if not files:
        return

    save_path = os.path.join("data/processed", person_name)
    os.makedirs(save_path, exist_ok=True)

    embeddings = []
    for i, file_path in enumerate(files):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        norm_img = normalize_image(img)
        save_processed_image(norm_img, save_path, i)
        input_img = norm_img.reshape(1, 128, 128, 1)
        embedding = base_network.predict(input_img)[0]
        embeddings.append(embedding)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        stored_embeddings = np.vstack([stored_embeddings, avg_embedding])
        stored_labels = np.append(stored_labels, label_counter)
        label_map[str(label_counter)] = person_name
        label_counter += 1
        save_embeddings()
        messagebox.showinfo("Success", f"Stored average embedding for {person_name}.")
    else:
        messagebox.showwarning("Warning", "No valid images processed.")


# === Signup (Webcam) ===
def signup_from_webcam():
    global stored_embeddings, stored_labels, label_map, label_counter

    person_name = simpledialog.askstring("Signup", "Enter Person Name:")
    if not person_name:
        return

    save_path = os.path.join("data/processed", person_name)
    os.makedirs(save_path, exist_ok=True)

    embeddings = []
    for i in range(3):  # Fewer than before
        messagebox.showinfo(
            "Capture", f"Capture #{i+1}: Press SPACE to save, ESC to stop."
        )
        eye_img = capture_eye_from_webcam()
        if eye_img is None:
            break
        save_processed_image(eye_img, save_path, i)
        input_img = eye_img.reshape(1, 128, 128, 1)
        embedding = base_network.predict(input_img)[0]
        embeddings.append(embedding)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        stored_embeddings = np.vstack([stored_embeddings, avg_embedding])
        stored_labels = np.append(stored_labels, label_counter)
        label_map[str(label_counter)] = person_name
        label_counter += 1
        save_embeddings()
        messagebox.showinfo("Success", f"Stored average embedding for {person_name}.")
    else:
        messagebox.showwarning("No Images", "No eye images captured.")


# === Webcam Capture Utility ===
def capture_eye_from_webcam():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    eye_img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            for ex, ey, ew, eh in eyes:
                eye_img = face_roi[ey : ey + eh, ex : ex + ew]
                cv2.rectangle(
                    display,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (0, 255, 0),
                    2,
                )
                break

        cv2.imshow("Press SPACE to capture, ESC to cancel", display)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:
            cap.release()
            cv2.destroyAllWindows()
            break

    if eye_img is not None:
        eye_img = cv2.resize(eye_img, (128, 128))
        return normalize_image(eye_img)
    return None


# === Login (Image Upload) ===
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")),
    )
    if not file_path:
        return

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", "Invalid image selected.")
        return

    img = cv2.resize(img, (128, 128))
    norm_img = normalize_image(img)
    eye_input = norm_img.reshape(1, 128, 128, 1)
    embedding = base_network.predict(eye_input)[0]

    identity, distance = get_identity_from_embedding(embedding)
    messagebox.showinfo(
        "Prediction", f"Predicted: {identity}\nDistance: {round(distance, 4)}"
    )

    if identity != "Unknown":
        messagebox.showinfo(
            "Access Granted", f"Welcome, {identity}!\nOpening secured file..."
        )
        os.system("notepad notes.txt")
    else:
        messagebox.showwarning("Access Denied", "Could not verify identity.")


# === Login (Webcam) ===
def login_from_webcam():
    eye_img = capture_eye_from_webcam()
    if eye_img is None:
        messagebox.showwarning("Failed", "No eye captured.")
        return

    eye_input = eye_img.reshape(1, 128, 128, 1)
    embedding = base_network.predict(eye_input)[0]

    identity, distance = get_identity_from_embedding(embedding)
    messagebox.showinfo(
        "Prediction", f"Predicted: {identity}\nDistance: {round(distance, 4)}"
    )

    if identity != "Unknown":
        messagebox.showinfo(
            "Access Granted", f"Welcome, {identity}!\nOpening secured file..."
        )
        os.system("notepad notes.txt")
    else:
        messagebox.showwarning("Access Denied", "Could not verify identity.")


# === GUI ===
root = tk.Tk()
root.title("Iris Recognition App")
root.geometry("500x300")

tk.Label(root, text="Iris Recognition System", font=("Helvetica", 16)).pack(pady=20)
tk.Button(
    root, text="Login (Upload Image)", command=upload_and_predict, width=25, height=2
).pack(pady=10)
tk.Button(
    root, text="Signup (Upload Images)", command=upload_and_register, width=25, height=2
).pack(pady=10)
tk.Button(
    root, text="Login (Webcam)", command=login_from_webcam, width=25, height=2
).pack(pady=5)
tk.Button(
    root, text="Signup (Webcam)", command=signup_from_webcam, width=25, height=2
).pack(pady=5)
tk.Button(root, text="Exit", command=root.destroy, width=25, height=2).pack(pady=10)

root.mainloop()
