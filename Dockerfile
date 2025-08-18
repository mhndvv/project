FROM python:3.12-slim

WORKDIR /app

COPY . /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install only the needed Python packages
RUN python -m pip install --upgrade pip \
    && pip install streamlit opencv-python-headless

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
