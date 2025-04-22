import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.preprocessing import detect_eyes, normalize_image
import cv2

def load_processed_dataset(processed_dir):
    
    X = []
    y = []
    label_map = {}
    label_counter = 0

    for person_name in os.listdir(processed_dir):
        person_path = os.path.join(processed_dir,person_name)
        if os.path.isdir(person_path):
            label_map[label_counter] = person_name
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path,img_name)
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = normalize_image(img)
                X.append(img)
                y.append(label_counter)
            label_counter += 1

    with open('utils/label_map.json','w') as f:
        json.dump(label_map,f)

    X = np.array(X).reshape(-1,128,128,1)
    y = np.array(y)
    return X,y,label_map


def build_model(input_shape,num_classes):
    model = Sequential()

    model.add(Conv2D(32,(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model():
    X, y, label_map = load_processed_dataset('data/processed/')
    if len(label_map) < 2:
        print('You need atleast 2 people to train a classification model')
        return
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((128,128,1), len(label_map))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val,y_val), batch_size=16)

    model.save('models/iris_model.h5')
    print('Model is trained and saved!')