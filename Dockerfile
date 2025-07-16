FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip --no-cache-dir
RUN pip3 install --upgrade setuptools --no-cache-dir  
RUN pip3 install -r requirements.txt --no-cache-dir

COPY . .

CMD ["python3", "-u", "handler.py"]