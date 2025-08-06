# Use official Python slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything to /app inside container
COPY . /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip
RUN pip install streamlit torch torchvision transformers pillow requests jinja2==3.1.4

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app when container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
