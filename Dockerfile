# Use the correct RunPod base image for PyTorch
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

# Copy all files
COPY . .

# Download model weights during build (like SDXL template)
RUN python download_weights.py

# Set the handler
CMD ["python", "-u", "handler.py"]