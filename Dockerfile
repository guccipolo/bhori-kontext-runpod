FROM runpod/python:3.10-ubuntu

WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Required: install 'runpod' serverless package
RUN pip install runpod

# Set the serverless handler
CMD ["python", "handler.py"]
