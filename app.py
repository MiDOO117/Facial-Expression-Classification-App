import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("E:/courses/Orange & Amit/Projects/Facial Expression/model_fer2013_optimized.h5")

# Define emotion labels
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Streamlit UI
st.title("😊 Facial Expression Recognition App 😠")
st.write("Upload an image, and the model will predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to predict emotion
def predict_emotion(img):
    img = img.resize((48, 48)).convert("L")  # Resize to 48x48 and convert to grayscale
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).reshape(1, 48, 48, 1)

    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    
    return label_dict[class_index], prediction

if uploaded_file:
    image_pil = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Predict emotion
    emotion, confidence = predict_emotion(image_pil)

    # Show prediction
    st.subheader(f"Predicted Emotion: {emotion} 😃")
    
    # Show confidence scores
    st.bar_chart(confidence)

    # Explainability
    st.write("Model Prediction Scores:")
    for i, label in label_dict.items():
        st.write(f"{label}: {confidence[i]*100:.2f}%")

