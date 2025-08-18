import numpy as np
from PIL import Image
from typing import Tuple, Any

def run_inference_streamlit():
    """
    Run the Streamlit UI for football detection and captioning.

    Shows an image uploader, sliders for confidence/IoU, and a button to run
    YOLO detection followed by caption generation. Displays the uploaded image,
    the detection result, and the caption.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Renders outputs directly in the Streamlit app.

    Raises
    ------
    RuntimeError
        If the detection model fails to load or inference fails.
    ValueError
        If the uploaded file is missing or not a valid image.
    """

    
    pass  


def run_inference(model, img, conf, iou, augment, agnostic_nms, max_det, imgsz) -> Tuple[np.ndarray, Any]:
    """
    Run YOLO inference on an input image.

    Converts a PIL image to a NumPy array if needed, runs detection,
    and returns both the annotated image (RGB) and the first results object.

    Parameters
    ----------
    model : object
        The YOLO model instance used for inference.
    img : PIL.Image.Image or np.ndarray
        The input image for inference.
    conf : float
        Confidence threshold for detection.
    iou : float
        Intersection-over-Union (IoU) threshold for NMS.
    augment : bool
        Whether to apply test-time augmentation.
    agnostic_nms : bool
        Whether to apply class-agnostic NMS.
    max_det : int
        Maximum number of detections per image.
    imgsz : int
        Target image size for inference.

    Returns
    -------
    annotated_image_rgb : np.ndarray
        The annotated image in RGB (ready to display).
    result : object
        The YOLO Results object for this image.

    Raises
    ------
    ValueError
        If the input image type is not supported.
    RuntimeError
        If model inference fails unexpectedly.
    """
    img_np = np.array(img) if isinstance(img, Image.Image) else img

    results = model(
        img_np,
        conf=conf,
        iou=iou,
        augment=augment,          # <-- fixed missing comma
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        imgsz=imgsz,
    )

    # Handle both shapes: single Results vs list of Results
    res0 = results[0] if isinstance(results, list) else results

    # Ultralytics .plot() returns BGR; convert to RGB for display
    annotated_bgr = res0.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    return annotated_rgb, res0