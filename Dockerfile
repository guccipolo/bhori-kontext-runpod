# Use their container but run your handler
FROM valyriantech/comfyui-with-flux:latest

WORKDIR /app

# Copy your handler
COPY handler.py .
COPY . .

# Install runpod 
RUN pip install runpod

# Run your handler instead of ComfyUI
CMD ["python", "-u", "handler.py"]