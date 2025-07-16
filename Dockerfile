# Start completely fresh with clean Python
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first with CUDA support
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install AI/ML dependencies in order
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    accelerate \
    safetensors \
    diffusers \
    huggingface_hub \
    sentencepiece \
    tokenizers \
    Pillow \
    numpy \
    runpod

# Copy your code
COPY . .

# Run handler
CMD ["python", "-u", "handler.py"]