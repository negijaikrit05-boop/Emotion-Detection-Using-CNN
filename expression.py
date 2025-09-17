import cv2
import numpy as np
from keras.models import load_model
import os

# Update to the available model file path
model_path = 'fer2013_mini_XCEPTION.00-0.47.hdf5'

# Check if the model file exists
if not os.path.isfile(model_path):
    print(f"Error: The model file '{model_path}' was not found.")
    print("Listing files in the current directory:")
    print(os.listdir('.'))
    exit(1)

# Load the pre-trained emotion recognition model
print("Loading model...")
emotion_model = load_model(model_path, compile=False)
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
print("Model loaded successfully!")

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)
print("Starting video capture...")

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Count the number of faces detected
        num_faces = len(faces)
        
        # Perform emotion recognition on detected faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize the face ROI to match the input size of the model
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi.astype('float') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            predicted_emotion = emotion_model.predict(face_roi)[0]
            emotion_label = emotion_labels[np.argmax(predicted_emotion)]
            
            # Draw bounding box around the face and show emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the number of faces detected
        cv2.putText(frame, f'Number of faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Break the loop if 'a' is pressed
        if cv2.waitKey(10) & 0xFF == ord('a'):
            break

    except Exception as e:
        print(f"Error during processing: {e}")

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
print("Video capture stopped.")
