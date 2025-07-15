FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY . /app

# Install system dependencies (for PIL, etc.)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# RunPod expects this
CMD ["python", "handler.py"]