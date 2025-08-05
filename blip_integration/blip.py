# blip.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image_pil, prompt: str = None):
    processor, model = load_blip_model()
    
    if prompt:
        inputs = processor(image_pil, prompt, return_tensors="pt")
    else:
        inputs = processor(image_pil, return_tensors="pt")
    
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
