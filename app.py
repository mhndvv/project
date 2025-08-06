import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np

from model.load_model import load_model
from detection.inference import run_inference
from blip_integration.blip import generate_caption

st.title(":soccer: Football Detection & Captioning App")
st.write("Upload an image → Detect players → Generate automatic caption using BLIP.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

@st.cache_resource
def cached_load_model():
    return load_model()

model = cached_load_model()
st.success("✅ YOLO model loaded successfully.")

if uploaded_file:

    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded Image")

    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.01)

    if st.button("Detect and Caption"):
        # Updated to unpack two values from run_inference
        result_img, result = run_inference(
            model, image,
            conf, iou,
            augment=False, agnostic_nms=True, max_det=100, imgsz=1024
        )

        # Display detection image (NumPy RGB) as PIL for Streamlit
        st.image(Image.fromarray(result_img), caption="Detected Players")

        # Generate caption from the RGB PIL image
        caption = generate_caption(Image.fromarray(result_img))
        st.success(f" Caption: {caption}")
