import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os
from src.predict import predict_identity, label_map, model
import cv2
from src.preprocessing import normalize_image, save_processed_image
from src.model import train_model
import numpy as np

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*"))
    )
    if file_path:
        result = predict_identity(file_path)
        messagebox.showinfo("Prediction Result", f"Predicted Identity: {result}")
        
    if result != "Unknown":
        messagebox.showinfo("Access Granted", f"Welcome, {result}!\nOpening secured file...")
        os.system("notepad notes.txt")
    else:
        messagebox.showwarning("Access Denied", "Could not verify identity.")


def upload_and_register():
    person_name = simpledialog.askstring("Signup", "Enter the person's name:")
    if not person_name:
        return
        
    files = filedialog.askopenfilenames(
        title='Select eye images for signup (upload 3-5 good eye images)',
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*"))
        )
    if not files:
        return
        
    save_path = os.path.join('data/processed', person_name)
    os.makedirs(save_path, exist_ok=True)

    count = 0

    for file_path in files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_resized = cv2.resize(img, (128, 128))   
        norm_img = normalize_image(img_resized)
        save_processed_image(norm_img, save_path, count)
        count+=1

    messagebox.showinfo('Success', f'{count} images saved for {person_name}.\n Retraining model now...')
    train_model()
    messagebox.showinfo('Success', 'Model retrained successfully!') 

def capture_eye_from_webcam():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  

    eye_img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)

            for ex, ey, ew, eh in eyes:
                eye_img = face_roi[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(frame_display, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                break

        cv2.imshow('Press SPACE to capture, ESC to cancel', frame_display)

        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE key
            cap.release()
            cv2.destroyAllWindows()
            break

    if eye_img is not None:
        eye_img = cv2.resize(eye_img, (128, 128))
        norm_img = normalize_image(eye_img)
        return norm_img
    return None


def login_from_webcam():
    eye_img = capture_eye_from_webcam()
    if eye_img is None:
        messagebox.showwarning("Failed", "No eye captured.")
        return

    eye_input = eye_img.reshape(1, 128, 128, 1)
    predictions = model.predict(eye_input)
    predicted_label = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    if confidence < 0.70:
        result = "Unknown"
    else:
        result = label_map[str(predicted_label)]

    messagebox.showinfo("Prediction", f"Predicted: {result}\nConfidence: {round(confidence*100, 2)}%")

    if result != "Unknown":
        messagebox.showinfo("Access Granted", f"Welcome, {result}!\nOpening secured file...")
        os.system("notepad notes.txt")
    else:
        messagebox.showwarning("Access Denied", "Could not verify identity.")


def signup_from_webcam():
    person_name = simpledialog.askstring("Signup", "Enter Person Name:")
    if not person_name:
        return

    save_path = os.path.join("data/processed", person_name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    while count < 5:
        messagebox.showinfo("Capture", f"Capture #{count+1}: Press SPACE to save, ESC to stop.")
        eye_img = capture_eye_from_webcam()
        if eye_img is None:
            break
        save_processed_image(eye_img, save_path, count)
        count += 1

    if count > 0:
        messagebox.showinfo("Saved", f"Saved {count} images for {person_name}.\nRetraining model now...")
        train_model()
        messagebox.showinfo("Training Done", "Model retrained and saved!")



root = tk.Tk()
root.title("Iris Recognition App")
root.geometry("400x250")

title_label = tk.Label(root, text="Iris Recognition System", font=("Helvetica", 16))
title_label.pack(pady=20)

login_button = tk.Button(root, text="Login (Upload Image)", command=upload_and_predict, width=25, height=2)
login_button.pack(pady=10)

signup_button = tk.Button(root, text="Signup (Upload Images)", command=upload_and_register, width=25, height=2)
signup_button.pack(pady=10)

webcam_login_button = tk.Button(root, text="Login (Webcam)", command=login_from_webcam, width=25, height=2)
webcam_login_button.pack(pady=5)

webcam_signup_button = tk.Button(root, text="Signup (Webcam)", command=signup_from_webcam, width=25, height=2)
webcam_signup_button.pack(pady=5)


exit_button = tk.Button(root, text="Exit", command=root.destroy, width=25, height=2)
exit_button.pack(pady=10)

root.mainloop()