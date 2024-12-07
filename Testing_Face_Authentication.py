import os
import cv2
import numpy as np
import subprocess  # Import subprocess module to run external scripts

# Label dictionary mapping IDs to names
label = {'Dikchhya': 0, 'Roshan': 1, 'Swasthik': 2, 'Nischal': 3}
label_name = {v: k for k, v in label.items()}

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_faces():
    # Load the trained model
    if not os.path.exists('trained_model.xml'):
        print("Error: Trained model file not found! Please train the model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.xml')

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame. Exiting.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))  # Resize for consistency
            label_id, confidence = recognizer.predict(face_roi)
            
            # If confidence is above threshold, authentication is successful
            if confidence > 50:  # Adjust threshold as needed
                name = label_name.get(label_id, "Unknown")
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Prompt user for selecting virtual keyboard or mouse
                #cv2.putText(frame, "Press 'K' for Keyboard, 'M' for Mouse", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                #cv2.imshow('Face Recognition', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('k'):
                    print("Virtual Keyboard selected.")
                    subprocess.Popen(['python', 'virtual_keyboard.py'])  # Run virtual_keyboard.py
                    cap.release()  # Release camera and stop the face recognition loop
                    cv2.destroyAllWindows()  # Close the OpenCV window
                    break  # Exit after selection
                elif key == ord('m'):
                    print("Virtual Mouse selected.")
                    subprocess.Popen(['python', 'virtualMouse.py'])  # Run virtualMouse.py
                    cap.release()  # Release camera and stop the face recognition loop
                    cv2.destroyAllWindows()  # Close the OpenCV window
                    break  # Exit after selection

            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Face Recognition', frame)

        # Break loop on 'q' key press (this is now more of a fallback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the recognition function
recognize_faces()
