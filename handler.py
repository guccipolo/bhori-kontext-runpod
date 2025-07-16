# handler.py

from typing import Dict
import torch
from diffusers import FluxKontextPipeline
from io import BytesIO
import base64
from PIL import Image, ImageOps
import numpy as np

class EndpointHandler:
    def __init__(self, path: str = ""):
        print("🚀 Initializing Flux Kontext pipeline...")

        # Load base model from Hugging Face
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        # Load LoRA weights from local file in the root directory
        try:
            self.pipe.load_lora_weights(
                ".",  # Current directory
                weight_name="Bh0r12.safetensors"
            )
            print("✅ LoRA weights loaded from ./Bh0r12.safetensors.")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights: {str(e)}")

        # Move model to GPU or CPU
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Model ready with LoRA applied.")

    def __call__(self, data: Dict) -> Dict:
        print("🔧 Received raw data type:", type(data))
        print("🔧 Received raw data content:", data)

        if isinstance(data, dict):
            prompt = data.get("prompt")
            image_input = data.get("image")

            if prompt is None and image_input is None:
                inputs = data.get("inputs")
                if isinstance(inputs, dict):
                    prompt = inputs.get("prompt")
                    image_input = inputs.get("image")
                else:
                    return {"error": "Expected 'inputs' to be a JSON object with 'prompt' and 'image'."}
        else:
            return {"error": "Input must be a JSON object."}

        if not prompt:
            return {"error": "Missing 'prompt'."}
        if not image_input:
            return {"error": "Missing 'image' (base64)."}

        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

        print(f"📝 Prompt: {prompt}")
        print(f"🖼️ Image size: {image.size}")

        try:
            output = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=35,
                guidance_scale=4.0
            ).images[0]

            output_array = np.array(output)
            output_array = np.clip(output_array, 0, 255).astype(np.uint8)
            output = Image.fromarray(output_array)
        except Exception as e:
            return {"error": f"Model inference failed: {str(e)}"}

        try:
            buffer = BytesIO()
            output.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {"image": base64_image}
        except Exception as e:
            return {"error": f"Failed to encode output image: {str(e)}"}
