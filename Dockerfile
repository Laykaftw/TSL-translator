# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Configure pip to use a more stable mirror and increase timeout
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.timeout 1000

# Install Python dependencies in stages to avoid timeouts
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    pip install --no-cache-dir fastapi==0.109.2 uvicorn[standard]==0.27.1 python-multipart==0.0.9 pydantic==2.6.1 && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 numpy==1.26.4 mediapipe==0.10.9 && \
    pip install --no-cache-dir tqdm==4.66.2 filelock==3.13.1 typing-extensions>=4.10.0 && \
    pip install --no-cache-dir networkx==3.2.1 jinja2==3.1.3 fsspec==2024.2.0 && \
    pip install --no-cache-dir matplotlib==3.8.3

# Create necessary directories
RUN mkdir -p server_side_debug_frames saved_models

# Copy only the essential files needed for running the FastAPI app
COPY fastapi_app.py .
COPY configs/ ./configs/
COPY models/ ./models/
COPY utils/ ./utils/
COPY saved_models/best_model.pth ./saved_models/
COPY saved_models/class_names.txt ./saved_models/

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the application with debug logging
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 