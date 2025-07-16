# Use the correct RunPod base image for PyTorch
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with explicit protobuf installation
RUN pip install --upgrade pip && \
    pip install --no-cache-dir protobuf>=3.20.0 && \
    pip install --no-cache-dir sentencepiece>=0.1.99 && \
    pip install --no-cache-dir tokenizers>=0.13.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install runpod

# Copy all files
COPY . .

# Skip download for now - will download on first request with token
# RUN python download_weights.py

# Set the handler
CMD ["python", "-u", "handler.py"]