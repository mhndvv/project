from ultralytics import YOLO

def load_model():
    """
    Load the YOLO model.

    Returns
    -------
    YOLO
        A YOLO object detection model loaded from the specified file path.
    """
    model = YOLO("model/best (1).pt")
    return model


