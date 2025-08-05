from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_blip_model()

def generate_caption(image: Image.Image, prompt: str = "a photo of") -> str:
    inputs = processor(image, prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)
