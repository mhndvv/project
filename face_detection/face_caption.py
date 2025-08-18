import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import numpy as np

# Load YOLO face detection model
@st.cache_resource
def load_face_model():
    """
    Load and cache the YOLO face detection model.

    This function ensures the face detection model is loaded only once and
    cached across Streamlit reruns to improve performance.

    Returns
    -------
    YOLO
        A YOLO model instance loaded with the face detection weights.
    """
    return YOLO("face_detection/model.pt")

# Load BLIP model for captioning
@st.cache_resource
def load_blip_model():
    """
    Load and cache the BLIP image captioning model and processor.

    Downloads and initializes the BLIP processor and model from Hugging Face
    (`Salesforce/blip-image-captioning-base`). The objects are cached to 
    prevent reloading on every Streamlit rerun.

    Returns
    -------
    tuple
        processor : BlipProcessor
            Pretrained BLIP processor for preparing image and text inputs.
        model : BlipForConditionalGeneration
            Pretrained BLIP model for generating captions from images.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(img_pil: Image.Image, processor: BlipProcessor, model: BlipForConditionalGeneration) -> str:
    """
    Generate a text caption for an input image using the BLIP model.

    Parameters
    ----------
    img_pil : PIL.Image.Image
        The input image for which a caption is generated.
    processor : BlipProcessor
        Pretrained BLIP processor used to prepare image inputs.
    model : BlipForConditionalGeneration
        Pretrained BLIP model used to generate captions.

    Returns
    -------
    str
        A text caption describing the content of the image.
    """
    inputs = processor(images=img_pil, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def run_face_caption() -> None:
    """
    Streamlit app for face detection and caption generation.

    Workflow
    --------
    1. Upload an image (JPG, JPEG, PNG).
    2. Detect faces using a YOLO model.
    3. For each detected face:
       - Crop the region.
       - Generate a caption using BLIP.
    4. Display results with bounding boxes and captions.

    Returns
    -------
    None
        The function runs the Streamlit UI and displays outputs directly.
    """
    st.title("🧑 Face Detection + Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Load models (cached via @st.cache_resource)
        face_model = load_face_model()
        processor, blip_model = load_blip_model()

        # Run YOLO face detection
        results = face_model.predict(np.array(img))
        result_img = results[0].plot()[:, :, ::-1]  # Convert BGR → RGB
        st.image(result_img, caption="Detection Result", use_container_width=True)

        st.subheader("Detected Faces with Captions:")

        # Loop through detected faces
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img.crop((x1, y1, x2, y2))

            with torch.no_grad():  # disable gradient calc for inference
                caption = generate_caption(face_crop, processor, blip_model)

            st.image(face_crop, caption=f"Caption: {caption}", use_container_width=False)
