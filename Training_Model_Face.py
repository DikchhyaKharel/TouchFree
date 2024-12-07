import os
import cv2
import numpy as np

# Label dictionary mapping names to IDs
label = {'Dikchhya': 0, 'Roshan': 1, 'Swasthik': 2, 'Nischal': 3}

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_model(label_dict):
    faces = []
    labels = []

    # Ensure the 'Faces' directory exists
    if not os.path.exists('Faces'):
        print("Error: 'Faces' directory not found!")
        return

    # Loop through the images in the 'Faces' folder
    for file_name in os.listdir('Faces'):
        if file_name.endswith('.jpg'):
            # Extract the name from the file name
            name = file_name.split('_')[0]
            if name not in label_dict:
                print(f"Warning: {name} not found in label dictionary. Skipping {file_name}.")
                continue
            
            # Load the image and convert to grayscale
            image_path = os.path.join('Faces', file_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(detected_faces) > 0:
                # Crop the face region
                x, y, w, h = detected_faces[0]
                face_crop = gray[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (100, 100))  # Resize for consistency

                faces.append(face_crop)
                labels.append(label_dict[name])
            else:
                print(f"Warning: No face detected in {file_name}. Skipping.")
    
    # Train the recognizer
    if len(faces) < 2:
        print("Error: Not enough training data. Need at least 2 samples.")
        return
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save('trained_model.xml')
    print("Model trained and saved as 'trained_model.xml'.")

# Call the training function
train_model(label)
