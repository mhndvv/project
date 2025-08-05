# Football Player Detection & Captioning App

A Streamlit web application that detects football players in images using YOLO and generates descriptive captions with BLIP. .

---

##  Features

- Upload football field images for player detection.
- YOLOv8-based player detection with adjustable confidence and IoU thresholds.
- BLIP-powered image captioning for automatic description generation.
- Real-time visualization of detected players with bounding boxes.
-

---

## ⚙ Requirements

- Python 3.12+
- PyTorch
- Transformers
- Streamlit
- OpenCV
- Docker (optional for containerized deployment)

---





project/
├── app.py                     # Main Streamlit app
├── detection/
│   └── inference.py           # YOLO detection logic
├── model/
│   └── load_model.py          # YOLO model loader
├── blip_integration/
│   └── blip.py                # BLIP captioning module
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container setup
└── README.md                  # This file


