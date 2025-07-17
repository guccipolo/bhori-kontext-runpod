FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

COPY requirements.txt .

# Install key packages in order to avoid conflicts
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir protobuf==3.20.3
RUN python3 -m pip install --no-cache-dir transformers==4.44.0
RUN python3 -m pip install --no-cache-dir diffusers==0.25.0
RUN python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-u", "handler.py"]