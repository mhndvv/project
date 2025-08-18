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
    """
    Load and cache the ML model so it is not reloaded on every app rerun.

    Returns
    -------
    model : object
        The loaded machine learning model.

    Notes
    -----
    - Uses Streamlit's @st.cache_resource to persist the model across reruns.
    - Only reloads if code or dependencies change.
    """
    return load_model()

@st.cache_resource
def load_face_model():
    """
    Load and cache the YOLO face detection model.
    
    Returns
    -------
    YOLO
        A YOLO model instance loaded from face_detection/model.pt.
    
    Raises
    ------
    FileNotFoundError
        If the model file is not found at the specified path.
    """
    model_path = "face_detection/model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please download it manually."
        )
    return YOLO(model_path)


def run_inference_streamlit():
    """
    Run the Streamlit UI for football detection and captioning.

    Displays an uploader, confidence/IoU sliders, and a button to run
    YOLO detection followed by BLIP captioning. Shows the uploaded image,
    the detection result, and the generated caption.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Renders results directly in the Streamlit app.

    Raises
    ------
    RuntimeError
        If the detection model fails to load or inference fails.
    ValueError
        If the uploaded file is not a supported image type.
    """
    st.title(":soccer: Football Detection & Captioning")

    model = cached_load_model()
    if model is None:
        st.error("Model not loaded. Please ensure the weights are available.")
        return

    uploaded_file = st.file_uploader(
        "Upload a football image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Guard: treat non-image uploads early
        if uploaded_file.type.startswith("image/"):
            image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.error("Video files are not supported in this workflow. Please upload an image.")
            return

        conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.01)

        if st.button("Detect and Caption"):
            try:
                result_img, _ = run_inference(
                    model, image, conf, iou,
                    augment=False, agnostic_nms=True, max_det=100, imgsz=1024
                )
                st.image(
                    Image.fromarray(result_img),
                    caption="Detected Players",
                    use_container_width=True
                )

                caption = generate_caption(Image.fromarray(result_img))
                st.success(f"Caption: {caption}")
            except Exception as e:
                st.error(f"Inference failed: {e}")


def run_face_detection():
    """
    Run the Streamlit UI for face detection.

    Shows an image uploader, loads a cached YOLO face model, runs
    prediction on the uploaded image, and displays the annotated result.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Renders results directly in the Streamlit app.

    Raises
    ------
    RuntimeError
        If the face model fails to load or inference fails.
    ValueError
        If the uploaded file is not a valid image.
    """
    st.title("Face Detection")

    model = load_face_model()
    if model is None:
        st.error("Face model not loaded. Make sure weights exist at the expected path.")
        return

    uploaded_file = st.file_uploader(
        "Upload an image for face detection",
        type=["jpg", "jpeg", "png"]
    )
    if not uploaded_file:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    try:
        image_np = np.array(image)
        results = model.predict(image_np)
        annotated_img = results[0].plot()
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return

    st.image(annotated_img, caption="Face Detection Result", use_container_width=True)



from face_detection.face_caption import run_face_caption

def main():
    """
    Streamlit entry point for running multiple detection modes.

    Allows user to choose between:
    - Football Detection
    - Face Detection
    - Face Caption
    """
    st.sidebar.title("⚙️ Settings")

    mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Football Detection", "Face Detection", "Face Caption"]
    )

    if mode == "Football Detection":
        run_inference_streamlit()
    elif mode == "Face Detection":
        run_face_detection()
    elif mode == "Face Caption":
        run_face_caption()

if __name__ == "__main__":
    main()