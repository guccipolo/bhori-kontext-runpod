import base64
import os
from io import BytesIO
import runpod
import torch
from diffusers import FluxPipeline  # Note: Using FluxPipeline for 0.25.0
from PIL import Image, ImageOps
import numpy as np

# Load Flux pipeline - NOTE: FluxKontextPipeline might not exist in diffusers 0.25.0
# We'll use regular FluxPipeline and modify for image-to-image
try:
    # Try FluxKontextPipeline first (if available in 0.25.0)
    from diffusers import FluxKontextPipeline
    pipeline_class = FluxKontextPipeline
    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    print("Using FluxKontextPipeline")
except ImportError:
    # Fallback to regular FluxPipeline
    from diffusers import FluxPipeline
    pipeline_class = FluxPipeline
    model_name = "black-forest-labs/FLUX.1-dev"
    print("Using FluxPipeline (FluxKontext not available in diffusers 0.25.0)")

pipe = pipeline_class.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ.get("CACHE_DIR", "/runpod-volume/models")
)

# Load LoRA weights if they exist
try:
    pipe.load_lora_weights(".", weight_name="Bh0r12.safetensors")
    print("✅ LoRA weights loaded from ./Bh0r12.safetensors")
except Exception as e:
    print(f"⚠️ Failed to load LoRA weights: {str(e)}")

pipe = pipe.to("cuda")

def handler(job):
    job_input = job["input"]
    
    # Required inputs
    prompt = job_input["prompt"]
    
    # Optional parameters
    guidance_scale = job_input.get("guidance_scale", 4.0)
    num_inference_steps = job_input.get("num_inference_steps", 35)
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    
    # Fake run for testing
    fake_run = job_input.get("fake_run", False)
    if fake_run:
        return {"output": "fake_run"}
    
    # Handle image input for Kontext (if available)
    image_b64 = job_input.get("image")
    input_image = None
    
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            input_image = ImageOps.exif_transpose(input_image)
            print(f"🖼️ Input image size: {input_image.size}")
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}
    
    # Generate with appropriate method
    try:
        if input_image and hasattr(pipe, '__call__') and 'image' in pipe.__call__.__code__.co_varnames:
            # FluxKontextPipeline - image-to-image
            output = pipe(
                prompt=prompt,
                image=input_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        else:
            # FluxPipeline - text-to-image
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512
            ).images[0]
        
        # Hard clamp pixel values to prevent NaN/black outputs
        output_array = np.array(output)
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        output = Image.fromarray(output_array)
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
    
    # Encode output
    buffered = BytesIO()
    output.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

runpod.serverless.start({"handler": handler})