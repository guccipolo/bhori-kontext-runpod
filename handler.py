import os
import base64
import torch
from diffusers import FluxKontextPipeline
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup

torch.cuda.empty_cache()

class ModelHandler:
    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        print("🚀 Initializing Flux Kontext pipeline...")
        
        # Load base model from Hugging Face without fp16 variant
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            # Remove variant="fp16" - not available for Flux Kontext
        )

        # Load LoRA weights from local files
        try:
            self.pipe.load_lora_weights(
                ".",  # Current directory where your .safetensors files are
                weight_name="Bh0r12.safetensors"
            )
            print("✅ LoRA weights loaded from ./Bh0r12.safetensors.")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights: {str(e)}")

        # Move pipeline to GPU if available
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable memory optimizations (if available for Flux)
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory efficient attention enabled")
            
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.pipe.enable_model_cpu_offload()
            print("✅ Model CPU offload enabled")
            
        print("✅ Model ready with LoRA applied.")

# Global model instance
MODELS = ModelHandler()

def _save_and_upload_image(image, job_id):
    """Save and upload image using RunPod utilities"""
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_path = os.path.join(f"/{job_id}", "output.png")
    image.save(image_path)

    if os.environ.get("BUCKET_ENDPOINT_URL", False):
        image_url = rp_upload.upload_image(job_id, image_path)
    else:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:image/png;base64,{image_data}"

    rp_cleanup.clean([f"/{job_id}"])
    return image_url

@torch.inference_mode()
def generate_image(job):
    """
    Generate an image using Flux Kontext model
    """
    # Debug logging like the SDXL template
    import json
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        import pprint
        pprint.pprint(job, depth=4, compact=False)

    # Get input data
    job_input = job["input"]
    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        import pprint
        pprint.pprint(job_input, depth=4, compact=False)

    # Extract required parameters
    prompt = job_input.get("prompt")
    image_input = job_input.get("image")
    
    # Handle nested inputs (HF format compatibility)
    if not prompt and not image_input:
        inputs = job_input.get("inputs", {})
        prompt = inputs.get("prompt")
        image_input = inputs.get("image")

    if not prompt:
        return {"error": "Missing 'prompt' in input data."}
    if not image_input:
        return {"error": "Missing 'image' (base64) in input data."}

    # Optional parameters with defaults
    num_inference_steps = job_input.get("num_inference_steps", 35)
    guidance_scale = job_input.get("guidance_scale", 4.0)
    seed = job_input.get("seed")

    # Set seed if provided
    generator = None
    if seed is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device).manual_seed(seed)

    # Decode image from base64
    try:
        image_bytes = base64.b64decode(image_input)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Correct EXIF orientation
        print(f"🖼️ Image size: {image.size}")
    except Exception as e:
        return {"error": f"Failed to decode 'image' as base64: {str(e)}"}

    print(f"📝 Final prompt: {prompt}")

    try:
        # Generate image using Flux Kontext
        with torch.inference_mode():
            result = MODELS.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            output = result.images[0]
            print("🎨 Image generated.")

        # Hard clamp pixel values to prevent NaN/black outputs
        output_array = np.array(output)
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        output = Image.fromarray(output_array)
        print("🛑 Hard clamped output pixel values to [0, 255].")

        # Save and upload using RunPod utilities
        image_url = _save_and_upload_image(output, job["id"])

        results = {
            "image": image_url,
            "image_url": image_url,  # For compatibility
            "prompt": prompt,
        }

        if seed is not None:
            results["seed"] = seed

        print("✅ Returning image.")
        return results

    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}",
            "refresh_worker": True,
        }
    except Exception as err:
        print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }

# Start the RunPod serverless worker
runpod.serverless.start({"handler": generate_image})