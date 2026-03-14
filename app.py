!pip install tensorflow
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load your trained model
# Ensure 'plant_disease_model.h5' is in the same directory as this script
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    return model

model = load_model()

# 2. Define labels (from your notebook)
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# 3. Preprocessing function
def preprocess_image(image):
    # Resize to match your model's input shape (assuming 224x224 based on common defaults)
    image = image.resize((224, 224)) 
    img_array = np.array(image) / 255.0  # Normalize if your model expects it
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# --- UI Layout ---
st.title("🌿 Plant Disease Detector")
st.write("Upload an image of a leaf to identify its health status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Run Prediction
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # Display Result
    result = labels[predicted_class]
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: {confidence:.2f}%")
