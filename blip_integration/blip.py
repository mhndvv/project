from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip_model():
    """
    Load and cache the BLIP image captioning model and processor.

    This function downloads the pretrained BLIP processor and model
    from Hugging Face ("Salesforce/blip-image-captioning-base") and
    caches them with Streamlit so they are loaded only once per session.

    Returns
    -------
    tuple
        processor : BlipProcessor
            Pretrained BLIP processor used for preparing inputs.
        model : BlipForConditionalGeneration
            Pretrained BLIP model for generating captions.

    Raises
    ------
    OSError
        If the model or processor cannot be downloaded from Hugging Face.


    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model


processor, model = load_blip_model()
def generate_caption(image: Image.Image, prompt: str = "a photo of") -> str:
    """
    Generate a descriptive caption for an input image using the BLIP model.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to be captioned. Must be a valid RGB image.
    prompt : str, optional
        Initial text prompt to guide caption generation.
        Default is "a photo of".

    Returns
    -------
    str
        Generated caption describing the content of the image.

    Raises
    ------
    RuntimeError
        If the model fails during the generation process.
    ValueError
        If the provided image is not valid or not supported.


    """
    inputs = processor(image, prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)
