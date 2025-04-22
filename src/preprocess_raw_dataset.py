import os
import cv2
from preprocessing import normalize_image, save_processed_image

def preprocess_raw_dataset(raw_data_dir='data/raw', processed_data_dir='data/processed'):
    """
    Processes raw iris dataset images by resizing and saving them
    into processed folders.
    """
    person_folders = os.listdir(raw_data_dir)
    person_folders.sort()  # Sort for consistent labeling

    for idx, person_id in enumerate(person_folders):
        person_path = os.path.join(raw_data_dir, person_id)

        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        person_name = f"Person_{idx+1}"
        image_counter = 0

        for eye_side in ['left', 'right']:
            eye_folder = os.path.join(person_path, eye_side)
            if not os.path.exists(eye_folder):
                continue

            for img_name in os.listdir(eye_folder):
                if img_name.lower().endswith(('.jpg', '.png', '.bmp')):  # allow bmp
                    img_path = os.path.join(eye_folder, img_name)
                    
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue

                    # Resize to 128x128 directly
                    img_resized = cv2.resize(img, (128, 128))
                    norm_img = normalize_image(img_resized)

                    # Save normalized resized image
                    save_path = os.path.join(processed_data_dir, person_name)
                    save_processed_image(norm_img, save_path, image_counter)
                    image_counter += 1
        
        print(f"Processed {image_counter} images for {person_name}")

    print("✅ Preprocessing complete! All data saved to:", processed_data_dir)

if __name__ == "__main__":
    preprocess_raw_dataset()
