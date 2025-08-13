import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

@st.cache_resource
def load_face_model():
    model_path = "face_detection/model.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please download it manually.")
        return None
    return YOLO(model_path)

def run_face_detection():
    st.title("Face Detection")

    model = load_face_model()
    if model is None:
        return  # stop if model not loaded

    uploaded_file = st.file_uploader("Upload an image for face detection", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to NumPy array for model
        image_np = np.array(image)

        # Run prediction
        results = model.predict(image_np)

        # Get annotated image with bounding boxes
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Face Detection Result", use_column_width=True)
