def run_inference(model, img, conf, iou, augment, agnostic_nms, max_det, imgsz):
    """
    Run object detection on the image.

    Parameters
    ----------
    model : YOLO
        The loaded YOLO model.
    img : PIL.Image
        The image to run detection on.
    conf : float
        Confidence threshold.
    iou : float
        Intersection over Union threshold.
    augment : bool
        Whether to apply data augmentation.
    agnostic_nms : bool
        Whether to use class-agnostic NMS.
    max_det : int
        Maximum number of detections.
    imgsz : int
        Image size to use for inference.

    Returns
    -------
    result_img : np.ndarray
        The annotated image with detections drawn.

    """
    results = model(
        img,
        conf=conf,
        iou=iou,
        augment=augment,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        imgsz=imgsz,
        classes=None
    )

    result = results[0]
    result_img = result.plot()



    return result_img
