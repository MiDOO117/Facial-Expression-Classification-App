import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from mtcnn import MTCNN
import time

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("E:\courses\Orange & Amit\Projects\Facial Expression\model_fer2013_optimized.h5")

model = load_model()

# Emotion labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load MTCNN for face detection
detector = MTCNN()

# Streamlit UI
st.title("😊 Real-Time & Image-Based Facial Expression Recognition 😠")
st.write("Detect emotions from a live webcam feed or uploaded images.")

# Choose between real-time detection and image upload
option = st.radio("Choose an option:", ["📷 Real-Time Webcam", "📁 Upload an Image"])

# Function to predict emotion
def predict_emotion(face):
    face = cv2.resize(face, (48, 48))  # Resize to model input
    face = np.expand_dims(face, axis=0)  # Expand dimensions for model
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = face / 255.0  # Normalize

    prediction = model.predict(face)[0]
    emotion_index = np.argmax(prediction)
    
    return label_dict[emotion_index], prediction

# 🟢 **Option 1: Real-Time Webcam Detection**
if option == "📷 Real-Time Webcam":
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_placeholder = st.empty()

        prev_time = 0  # For FPS calculation

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not found!")
                break

            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)

            for face in faces:
                x, y, w, h = face['box']
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]  # Extract face
                emotion_label, _ = predict_emotion(roi_gray)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show video in Streamlit
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            if stop_button:
                cap.release()
                cv2.destroyAllWindows()
                break

# 🔵 **Option 2: Upload an Image**
elif option == "📁 Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")  # Keep RGB format
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        # Convert to numpy array
        img_array = np.array(image_pil)

        # Face detection using MTCNN
        faces = detector.detect_faces(img_array)

        if len(faces) == 0:
            st.warning("No face detected. Try another image!")
        else:
            for face in faces:
                x, y, w, h = face['box']
                roi_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)[y:y+h, x:x+w]  # Extract face
                emotion_label, confidence = predict_emotion(roi_gray)

                st.subheader(f"Predicted Emotion: {emotion_label} 😃")

                # Show confidence scores
                st.bar_chart(confidence)
                st.write("Model Confidence Scores:")
                for i, label in label_dict.items():
                    st.write(f"{label}: {confidence[i]*100:.2f}%")
