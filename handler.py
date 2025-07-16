import runpod
from typing import Dict
import torch
from diffusers import FluxKontextPipeline
from io import BytesIO
import base64
from PIL import Image, ImageOps
import numpy as np
import os

# Initialize the model globally so it persists across requests
pipe = None

def initialize_model():
    """Initialize the Flux Kontext pipeline"""
    global pipe
    
    if pipe is not None:
        return pipe
        
    print("🚀 Initializing Flux Kontext pipeline...")

    # Load base model from Hugging Face
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # Load LoRA weights from local file in the root directory
    try:
        pipe.load_lora_weights(
            ".",  # Current directory
            weight_name="Bh0r12.safetensors"
        )
        print("✅ LoRA weights loaded from ./Bh0r12.safetensors.")
    except Exception as e:
        print(f"⚠️ Failed to load LoRA weights: {str(e)}")

    # Move model to GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    print(f"✅ Model ready with LoRA applied on {device}.")
    
    return pipe

def handler(event):
    """RunPod serverless handler"""
    try:
        print("🔧 Received event:", event)
        
        # Initialize model if not already done
        model = initialize_model()
        
        # Extract input data
        input_data = event.get("input", {})
        print("🔧 Input data:", input_data)
        
        # Handle different input formats
        prompt = input_data.get("prompt")
        image_input = input_data.get("image")
        
        # Check if it's nested under 'inputs'
        if not prompt and not image_input:
            nested_inputs = input_data.get("inputs", {})
            prompt = nested_inputs.get("prompt")
            image_input = nested_inputs.get("image")
        
        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not image_input:
            return {"error": "Missing 'image' (base64) in input"}

        print(f"📝 Prompt: {prompt}")
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = ImageOps.exif_transpose(image)
            print(f"🖼️ Image size: {image.size}")
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

        # Generate image
        try:
            print("🎨 Starting generation...")
            output = model(
                prompt=prompt,
                image=image,
                num_inference_steps=35,
                guidance_scale=4.0
            ).images[0]

            # Convert to proper format
            output_array = np.array(output)
            output_array = np.clip(output_array, 0, 255).astype(np.uint8)
            output = Image.fromarray(output_array)
            
            print("✅ Generation completed")
        except Exception as e:
            return {"error": f"Model inference failed: {str(e)}"}

        # Encode result as base64
        try:
            buffer = BytesIO()
            output.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            print("📤 Returning generated image")
            return {"image": base64_image}
            
        except Exception as e:
            return {"error": f"Failed to encode output image: {str(e)}"}

    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        return {"error": f"Handler failed: {str(e)}"}

if __name__ == "__main__":
    # Start the RunPod serverless worker
    print("🚀 Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})