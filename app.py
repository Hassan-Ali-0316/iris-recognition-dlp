import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocessing import normalize_image, save_processed_image

# Register custom functions for model loading
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Create necessary directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("src/models", exist_ok=True)

# Define constant paths
MODEL_PATH = "src/models/siamese_model.keras"
EMBEDDINGS_PATH = "embeddings/embeddings.npz"
LABEL_MAP_PATH = "embeddings/label_map.json"

# === Load model and base network ===
try:
    model = load_model(
        MODEL_PATH, 
        custom_objects={
            "contrastive_loss": contrastive_loss,
            "euclidean_distance": euclidean_distance
        }
    )
    base_network = model.get_layer("base_network")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    messagebox.showerror("Error", f"Failed to load model: {e}\nPlease train the model first.")
    base_network = None
    model = None

# === Load embeddings ===
if os.path.exists(EMBEDDINGS_PATH):
    try:
        emb_data = np.load(EMBEDDINGS_PATH)
        stored_embeddings = emb_data["embeddings"]
        stored_labels = emb_data["labels"]
        print(f"Loaded {len(stored_embeddings)} embeddings")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        stored_embeddings = np.empty((0, 128))
        stored_labels = np.empty((0,))
else:
    stored_embeddings = np.empty((0, 128))
    stored_labels = np.empty((0,))

if os.path.exists(LABEL_MAP_PATH):
    try:
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        print(f"Loaded {len(label_map)} labels")
    except Exception as e:
        print(f"Error loading label map: {e}")
        label_map = {}
else:
    label_map = {}

label_counter = max([int(k) for k in label_map.keys()], default=-1) + 1

def save_embeddings():
    """Save embeddings and label map to disk"""
    try:
        np.savez(EMBEDDINGS_PATH, embeddings=stored_embeddings, labels=stored_labels)
        with open(LABEL_MAP_PATH, "w") as f:
            json.dump(label_map, f)
        print("Embeddings and label map saved successfully")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        messagebox.showerror("Error", f"Failed to save embeddings: {e}")

def get_identity_from_embedding(embedding, threshold=0.5):
    """Find the closest identity to the given embedding"""
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
    """Register a new user by uploading multiple eye images"""
    global stored_embeddings, stored_labels, label_map, label_counter
    
    if base_network is None:
        messagebox.showerror("Error", "Model not loaded. Please train the model first.")
        return

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
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showwarning("Warning", f"Failed to load image: {file_path}")
                continue
                
            img = cv2.resize(img, (128, 128))
            norm_img = normalize_image(img)
            save_processed_image(norm_img, save_path, i)
            
            input_img = norm_img.reshape(1, 128, 128, 1)
            embedding = base_network.predict(input_img, verbose=0)[0]
            embeddings.append(embedding)
        except Exception as e:
            messagebox.showwarning("Error", f"Error processing {file_path}: {e}")
            continue

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
    """Register a new user using webcam images"""
    global stored_embeddings, stored_labels, label_map, label_counter
    
    if base_network is None:
        messagebox.showerror("Error", "Model not loaded. Please train the model first.")
        return

    person_name = simpledialog.askstring("Signup", "Enter Person Name:")
    if not person_name:
        return

    save_path = os.path.join("data/processed", person_name)
    os.makedirs(save_path, exist_ok=True)

    embeddings = []
    for i in range(3):  # Take 3 images for better accuracy
        messagebox.showinfo(
            "Capture", f"Capture #{i+1}: Press SPACE to save, ESC to stop."
        )
        eye_img = capture_eye_from_webcam()
        if eye_img is None:
            break
            
        save_processed_image(eye_img, save_path, i)
        input_img = eye_img.reshape(1, 128, 128, 1)
        embedding = base_network.predict(input_img, verbose=0)[0]
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
    """Capture an eye image from webcam"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam")
            return None
            
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

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
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # SPACE
                cap.release()
                cv2.destroyAllWindows()
                break

        if eye_img is not None:
            eye_img = cv2.resize(eye_img, (128, 128))
            return normalize_image(eye_img)
        return None
    
    except Exception as e:
        messagebox.showerror("Error", f"Webcam error: {e}")
        return None

# === Login (Image Upload) ===
def upload_and_predict():
    """Verify identity using uploaded eye image"""
    if base_network is None:
        messagebox.showerror("Error", "Model not loaded. Please train the model first.")
        return

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")),
    )
    if not file_path:
        return

    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Invalid image selected.")
            return

        img = cv2.resize(img, (128, 128))
        norm_img = normalize_image(img)
        eye_input = norm_img.reshape(1, 128, 128, 1)
        embedding = base_network.predict(eye_input, verbose=0)[0]

        identity, distance = get_identity_from_embedding(embedding)
        messagebox.showinfo(
            "Prediction", f"Predicted: {identity}\nDistance: {round(distance, 4)}"
        )

        if identity != "Unknown":
            messagebox.showinfo(
                "Access Granted", f"Welcome, {identity}!\nOpening secured file..."
            )
            try:
                # Create a placeholder notes file if it doesn't exist
                if not os.path.exists("notes.txt"):
                    with open("notes.txt", "w") as f:
                        f.write("Your secure notes go here.")
                os.system("notepad notes.txt")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open notes: {e}")
        else:
            messagebox.showwarning("Access Denied", "Could not verify identity.")
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction: {e}")

# === Login (Webcam) ===
def login_from_webcam():
    """Verify identity using webcam eye image"""
    if base_network is None:
        messagebox.showerror("Error", "Model not loaded. Please train the model first.")
        return
        
    eye_img = capture_eye_from_webcam()
    if eye_img is None:
        messagebox.showwarning("Failed", "No eye captured.")
        return

    try:
        eye_input = eye_img.reshape(1, 128, 128, 1)
        embedding = base_network.predict(eye_input, verbose=0)[0]

        identity, distance = get_identity_from_embedding(embedding)
        messagebox.showinfo(
            "Prediction", f"Predicted: {identity}\nDistance: {round(distance, 4)}"
        )

        if identity != "Unknown":
            messagebox.showinfo(
                "Access Granted", f"Welcome, {identity}!\nOpening secured file..."
            )
            try:
                # Create a placeholder notes file if it doesn't exist
                if not os.path.exists("notes.txt"):
                    with open("notes.txt", "w") as f:
                        f.write("Your secure notes go here.")
                os.system("notepad notes.txt")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open notes: {e}")
        else:
            messagebox.showwarning("Access Denied", "Could not verify identity.")
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction: {e}")

# === Train model button ===
def open_train_dialog():
    """Open dialog to confirm model training"""
    if messagebox.askyesno("Train Model", "Do you want to train the model? This may take some time."):
        try:
            from src.model import train_model
            train_model()
            messagebox.showinfo("Success", "Model trained successfully. Restart the application to use the new model.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

# === GUI ===
def create_gui():
    """Create the main GUI"""
    root = tk.Tk()
    root.title("Iris Recognition App")
    root.geometry("500x380")

    tk.Label(root, text="Iris Recognition System", font=("Helvetica", 16)).pack(pady=20)
    
    # Create a frame for login buttons
    login_frame = tk.Frame(root)
    login_frame.pack(pady=10)
    tk.Label(login_frame, text="Login", font=("Helvetica", 12)).pack()
    tk.Button(
        login_frame, text="Upload Image", command=upload_and_predict, width=15, height=2
    ).pack(side=tk.LEFT, padx=5)
    tk.Button(
        login_frame, text="Use Webcam", command=login_from_webcam, width=15, height=2
    ).pack(side=tk.RIGHT, padx=5)
    
    # Create a frame for signup buttons
    signup_frame = tk.Frame(root)
    signup_frame.pack(pady=10)
    tk.Label(signup_frame, text="Signup", font=("Helvetica", 12)).pack()
    tk.Button(
        signup_frame, text="Upload Images", command=upload_and_register, width=15, height=2
    ).pack(side=tk.LEFT, padx=5)
    tk.Button(
        signup_frame, text="Use Webcam", command=signup_from_webcam, width=15, height=2
    ).pack(side=tk.RIGHT, padx=5)
    
    # Add a train model button
    tk.Button(
        root, text="Train Model", command=open_train_dialog, width=20, height=2
    ).pack(pady=10)
    
    # Add exit button
    tk.Button(root, text="Exit", command=root.destroy, width=20, height=2).pack(pady=10)

    return root

if __name__ == "__main__":
    root = create_gui()
    root.mainloop()