import base64
import os
from io import BytesIO
import runpod
import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageOps
import numpy as np

# Load Flux Kontext pipeline with LoRA
pipe = FluxKontextPipeline.from_pretrained(
    os.environ.get("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-Kontext-dev"),
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
    
    # Required inputs for Flux Kontext (image-to-image)
    prompt = job_input["prompt"]
    image_b64 = job_input["image"]
    
    # Optional parameters
    guidance_scale = job_input.get("guidance_scale", 4.0)
    num_inference_steps = job_input.get("num_inference_steps", 35)
    
    # Fake run for testing
    fake_run = job_input.get("fake_run", False)
    if fake_run:
        return {"output": "fake_run"}
    
    # Decode input image
    try:
        image_bytes = base64.b64decode(image_b64)
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_image = ImageOps.exif_transpose(input_image)
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}
    
    # Generate with Flux Kontext
    try:
        output = pipe(
            prompt=prompt,
            image=input_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
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