import cv2
import os
import numpy as np
from train import train_model
# Function to recognize faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def recognize_faces(recognizer, label):
	# Open the camera
	cap = cv2.VideoCapture(0)
	
	# Reverse keys and values in the dictionary
	label_name = {value: key for key, value in label.items()}
	while True:
		# Read a frame from the camera
		ret, frame = cap.read()

		# Convert the frame to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect faces in the grayscale frame
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		
		# Recognize and label the faces
		for (x, y, w, h) in faces:
			# Recognize the face using the trained model
			label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
			#print(confidence)
			if confidence > 50:
				# Display the recognized label and confidence level
				cv2.putText(frame, label_name[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	
				# Draw a rectangle around the face
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			else:
				print('Unrecognized')

		# Display the frame with face recognition
		cv2.imshow('Recognize Faces', frame)

		# Break the loop if the 'q' key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the camera and close windows
	cap.release()
	cv2.destroyAllWindows()
label = {'Dikchhya':0,'Roshan':1,'Swasthik':2,'Nischal':3}
recognizer = cv2.face.LBPHFaceRecognizer_create()
Recognizer =train_model(label)
recognize_faces(Recognizer, label)