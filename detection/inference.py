import numpy as np
import cv2

def run_inference(model, image, conf, iou, augment, agnostic_nms, max_det, imgsz):
    
    if isinstance(image, np.ndarray):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model.predict(
        image_bgr,
        conf=conf,
        iou=iou,
        augment=augment,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        imgsz=imgsz
    )

    result_img = results[0].plot()  

  
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)