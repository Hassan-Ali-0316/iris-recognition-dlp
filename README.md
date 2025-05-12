# 👁️ Iris Recognition System using Deep Learning

This project is a real-time biometric iris recognition system built with **Convolutional Neural Networks (CNNs)**. It includes a clean **Tkinter GUI** that supports both image upload and **webcam-based iris recognition**, along with user registration and dynamic model retraining. The system simulates secure access — only verified individuals can unlock a protected file.

## 📌 Features

- 🔍 **Iris-based Identity Verification** using CNN  
- 🖼️ **Upload** or 📷 **Capture via Webcam**  
- 🧠 **Trainable CNN Model** with data augmentation  
- 👤 **Signup Flow** for new users (with automatic retraining)  
- 🧪 **Login Flow** with unknown-user detection  
- 📊 Visual outputs:  
  - Accuracy / Loss plots  
  - Confusion Matrix  
  - Model Summary + Training Logs  
- 🔒 Opens Notepad file only on successful login

## 📁 Folder Structure

```
iris-recognition/
├── app.py                        # GUI app (main file)
├── data/
│   ├── raw/                      # Original MMU iris dataset
│   ├── processed/                # Cropped, normalized images per person
│   └── test/                     # User-provided test images
├── models/
│   └── iris_model.h5             # Trained CNN model
├── src/
│   ├── model.py                  # Training and model building
│   ├── predict.py                # Prediction logic
│   ├── preprocessing.py          # Eye detection, normalization, saving
│   └── preprocess_raw_dataset.py # Converts MMU to cropped eye images
├── utils/
│   └── label_map.json            # Maps class index to person names
├── accuracy_plot.png             # Training accuracy graph
├── loss_plot.png                 # Training loss graph
├── confusion_matrix.png          # Model evaluation matrix
├── model_summary.txt             # Architecture summary
├── training_metrics.csv          # Epoch-wise logs
└── README.md                     # This file
```

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Preprocess MMU Dataset
```bash
python src/preprocess_raw_dataset.py
```

### 3. Train the CNN Model
```python
from src.model import train_model
train_model()
```

### 4. Launch the GUI
```bash
python app.py
```

## 🖥️ GUI Features

- **Login (Upload / Webcam):** Predict identity from image and grant access  
- **Signup (Upload / Webcam):** Add a new person and retrain the model  
- **Protected Access:** Only known users can open the secured file (e.g., Notepad)


Model summary is saved to `model_summary.txt`.

