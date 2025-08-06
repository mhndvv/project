import numpy as np
from PIL import Image

def run_inference(model, img, conf, iou, augment, agnostic_nms, max_det, imgsz):


    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    results = model(
        img_np,
        conf=conf,
        iou=iou,
        augment=augment,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        imgsz=imgsz,
    )

    result = results[0]

    result_img_bgr = result.plot()

  
    result_img_rgb = result_img_bgr[:, :, ::-1]

    return result_img_rgb, result
