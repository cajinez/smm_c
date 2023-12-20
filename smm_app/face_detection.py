import cv2
import tensorflow as tf
import numpy as np

# Load the trained models
gender_model = tf.keras.models.load_model('gender_model.h5')
emotion_model = tf.keras.models.load_model('emotion_model.h5')

# Define the labels for gender and emotions
gender_labels = ['Male', 'Female']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop through each face and predict gender and emotion
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to 48x48 pixels
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the face ROI
        face_roi = face_roi / 255.0

        # Reshape the face ROI to a 4D tensor
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict the gender of the face
        gender_prediction = gender_model.predict(face_roi)
        gender_label = gender_labels[np.argmax(gender_prediction)]

        # Predict the emotion of the face
        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Draw a rectangle around the face and display the gender and emotion labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()