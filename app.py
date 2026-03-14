pip install -r requirements.txt
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load trained model
model = load_model("model.h5")

# Class labels (same order as training folders)
classes = ["Healthy", "Powdery", "Rust"]

# Image preprocessing function
def preprocess_image(image, target_size=(225,225)):
    img = image.resize(target_size)
    img = img_to_array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to detect the disease")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Analyzing image...")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    st.success(f"Disease: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

       
