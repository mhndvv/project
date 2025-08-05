# ⚽ Football Object Detection - YOLO Streamlit App

A real-time football player detection application built with YOLO (You Only Look Once) and Streamlit. This application allows users to upload football field images and detect players using a custom-trained YOLO model.

## Features

- **Real-time Object Detection**: Detect football players on uploaded images
- **Interactive Web Interface**: User-friendly Streamlit interface
- **Customizable Parameters**: Adjust confidence threshold, IoU, and other detection parameters
- **Image Processing**: Support for JPG, JPEG, and PNG image formats
- **Model Caching**: Efficient model loading with Streamlit caching

## Project Structure

```
yolo_app/
├── app.py                 # Main Streamlit application
├── pyproject.toml        # Project dependencies and configuration
├── README.md             # Project documentation
├── detection/
│   └── inference.py      # YOLO inference logic
├── model/
│   ├── load_model.py     # Model loading utilities
│   └── best (1).pt      # Custom-trained YOLO model weights
└── .venv/               # Virtual environment (not tracked)
```

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yolo_app
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Launch the Application**: Run `streamlit run app.py` in your terminal
2. **Upload Image**: Click "Browse files" to upload a football field image
3. **Adjust Parameters** (optional)
4. **Detect Players**: Click "Detect Players" button to run inference
5. **View Results**: The annotated image with detected players will be displayed

## Configuration

### Detection Parameters

- **Confidence Threshold**: Controls the minimum confidence score for detections (default: 0.25)
- **IoU Threshold**: Controls overlap detection for non-maximum suppression (default: 0.7)
- **Maximum Detections**: Limits the number of detected objects (default: 100)
- **Image Size**: Sets the input image size for the model (default: 1024)
- **Augmentation**: Enables data augmentation during inference
- **Agnostic NMS**: Uses class-agnostic non-maximum suppression

##  Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **Ultralytics**: YOLO model implementation
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical computations

### Model Information

- **Model Type**: Custom-trained YOLO model
- **Model File**: `model/best (1).pt`
- **Input Format**: RGB images (JPG, JPEG, PNG)
- **Output**: Annotated images with bounding boxes


## License

This project is licensed under the MIT License - see the LICENSE file for details.
