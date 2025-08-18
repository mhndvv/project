import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

@st.cache_resource
def load_face_model():
    """
    Load the YOLO face detection model.

    Returns
    -------
    YOLO
        The loaded YOLO model if available.
    None
        If the model file is missing.
    """
    model_path = "face_detection/model.pt"

    if not os.path.exists(model_path):
        st.error(
            f"❌ Face detection model not found at: `{model_path}`\n"
            f"➡ Please place the model file in the correct folder."
        )
        st.stop()  # prevents rest of the app from running

    return YOLO(model_path)

def run_face_detection():
    """
    Launch a Streamlit interface for face detection on uploaded images.

    The function loads a YOLO face detection model, allows the user to upload
    an image (JPG/PNG), runs inference to detect faces, and displays both the
    uploaded image and the annotated detection result with bounding boxes.

    Parameters
    ----------
    None
        This function takes no input parameters directly. All inputs come
        interactively from the Streamlit UI.

    Returns
    -------
    None
        The function outputs results directly to the Streamlit app interface
        (uploaded image preview, detection results).

    Raises
    ------
    FileNotFoundError
        If the face detection model file is missing when `load_face_model()` is called.
    RuntimeError
        If the YOLO model fails during prediction.
    """
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

