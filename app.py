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

        result_img = run_inference(
            model, image,
            conf, iou,
            augment=False, agnostic_nms=True, max_det=100, imgsz=1024
        )
        st.image(result_img, caption="Detected Players")


        if isinstance(result_img, np.ndarray):
            result_pil = Image.fromarray(result_img)
        else:
            result_pil = result_img

        caption = generate_caption(result_pil)
        st.success(f" Caption: {caption}")
