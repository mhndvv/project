import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import numpy as np

# Load YOLO face detection model
@st.cache_resource
def load_face_model():
    return YOLO("face_detection/model.pt")  # update path if needed

# Load BLIP model for captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(img_pil, processor, model):
    inputs = processor(images=img_pil, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def run_face_caption():
    st.title("🧑 Face Detection + Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        model = load_face_model()
        processor, blip_model = load_blip_model()

        # Run YOLO face detection
        results = model.predict(np.array(img))
        result_img = results[0].plot()[:, :, ::-1]  # BGR to RGB
        st.image(result_img, caption="Detection Result", use_container_width=True)

        st.subheader("Detected Faces with Captions:")
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img.crop((x1, y1, x2, y2))

            caption = generate_caption(face_crop, processor, blip_model)
            st.image(face_crop, caption=f"Caption: {caption}", use_container_width=False)
