dockerfile# Use their container but run your handler
FROM valyriantech/comfyui-with-flux:latest

# Remove ComfyUI startup, add your handler  
COPY handler.py .
RUN pip install runpod

# Your handler uses the pre-installed Flux model
CMD ["python", "-u", "handler.py"]