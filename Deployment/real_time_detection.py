import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained facial expression recognition model
model = tf.keras.models.load_model("E:\courses\Orange & Amit\Projects\Facial Expression\model_fer2013_optimized.h5")

# Define emotion labels
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (as the model was trained on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Extract face region
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model input
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Expand dimensions for model
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = roi_gray / 255.0  # Normalize

        # Predict emotion
        prediction = model.predict(roi_gray)[0]
        emotion_index = np.argmax(prediction)
        emotion_label = label_dict[emotion_index]

        # Draw rectangle around face and put emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the real-time video with detections
    cv2.imshow("Facial Expression Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
