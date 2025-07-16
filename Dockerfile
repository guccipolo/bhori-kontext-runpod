# Use the correct RunPod base image for PyTorch
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with force reinstall for tokenizers
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall sentencepiece tokenizers && \
    pip install runpod

# Copy all files
COPY . .

# Skip download for now - will download on first request with token
# RUN python download_weights.py

# Set the handler
CMD ["python", "-u", "handler.py"]