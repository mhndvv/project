import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from model.load_model import load_model
from detection.inference import run_inference
from blip_integration.blip import generate_caption
import os
from ultralytics import YOLO

@st.cache_resource
def cached_load_model():
    return load_model()

@st.cache_resource
def load_face_model():
    model_path = "face_detection/model.pt"  # Adjust path as needed
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please download it manually.")
        return None
    return YOLO(model_path)

def run_inference_streamlit():
    st.title(":soccer: Football Detection & Captioning")

    model = cached_load_model()

    uploaded_file = st.file_uploader("Upload a football image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.01)

        if st.button("Detect and Caption"):
            result_img, _ = run_inference(
                model, image, conf, iou,
                augment=False, agnostic_nms=True, max_det=100, imgsz=1024
            )
            st.image(Image.fromarray(result_img), caption="Detected Players", use_container_width=True)

            caption = generate_caption(Image.fromarray(result_img))
            st.success(f"Caption: {caption}")

def run_face_detection():
    st.title("Face Detection")

    model = load_face_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("Upload an image for face detection", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        results = model.predict(image_np)

        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Face Detection Result", use_container_width=True)


from face_detection.face_caption import run_face_caption

mode = st.sidebar.selectbox("Detection Mode", ["Football Detection", "Face Detection", "Face Caption"])

if mode == "Football Detection":
    run_inference_streamlit()
elif mode == "Face Detection":
    run_face_detection()
elif mode == "Face Caption":
    run_face_caption()
