import cv2
import cv2.data
import numpy as np
import os

def detect_eyes(path):
    """
    Detects eyes from a given image.
    
    Parameters:
        image_path (str): Path to the input image.
    
    Returns:
        list: A list of cropped eye images.
    """
    # load haar cascade identifier 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    img = cv2.imread(path)
    if img is None:
        print(f'Could not read image with path: {path}')
        return []
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_cropped = []

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            eye = roi_gray[ey:ey+eh,ex:ex+ew]
            eye_resized = cv2.resize(eye,(128,128))
            eyes_cropped.append(eye_resized)

    return eyes_cropped


def normalize_image(image): 
    return image/255.0


def save_processed_image(image,save_path,count):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path,f'eye_{count}.jpg')
    cv2.imwrite(file_path,image*255)