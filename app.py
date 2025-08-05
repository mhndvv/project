import os
import streamlit as st
from PIL import Image
from io import BytesIO
from model.load_model import load_model
from detection.inference import run_inference
st.title(":soccer: Football Object Detection - YOLO Streamlit App")
st.write("Upload an image to detect players on the field using a custom YOLO model.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Open uploaded image directly from memory
    image = Image.open(BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image")

# Load model only once
@st.cache_resource
def cached_load_model():
    return load_model()

model = cached_load_model()
st.success("✅ YOLO model loaded successfully.")

# Detection parameters
conf = st.number_input("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.number_input("Intersection Over Union (IoU)", 0.0, 1.0, 0.7, 0.01)
max_det = st.number_input("Maximum Detections", 1, 3000, 100, 1)
imgsz = st.number_input("Image Size", 64, 2048, 1024, 32)
augment = st.checkbox("Enable Augmentation", value=False)
agnostic_nms = st.checkbox("Agnostic NMS", value=True)

if st.button("Detect Players"):

    result_img = run_inference(
        model, image, conf, iou, augment, agnostic_nms, max_det, imgsz
    )

    st.image(result_img, caption="Detected Players")
