from typing import Dict
import torch
from diffusers import FluxKontextPipeline
from io import BytesIO
import base64
from PIL import Image, ImageOps
import numpy as np  # Added import

class EndpointHandler:
    def __init__(self, path: str = ""):
        print("🚀 Initializing Flux Kontext pipeline...")

        # Load base model from Hugging Face
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.float16,
        )

        # Debug available methods on pipeline
        print("🔍 Available methods on pipeline:", dir(self.pipe))

        # Load your LoRA weights from your Hugging Face repo
        try:
            self.pipe.load_lora_weights(
                "Texttra/BhoriKontext",
                weight_name="Bh0r1.safetensors"
            )
            print("✅ LoRA weights loaded from Texttra/BhoriKontext/Bh0r1.safetensors.")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights: {str(e)}")

        # Move pipeline to GPU if available
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Model ready with LoRA applied.")

    def __call__(self, data: Dict) -> Dict:
        print("🔧 Received raw data type:", type(data))
        print("🔧 Received raw data content:", data)

        # Defensive parsing
        if isinstance(data, dict):
            prompt = data.get("prompt")
            image_input = data.get("image")

            # If 'inputs' key is used (HF Inference schema)
            if prompt is None and image_input is None:
                inputs = data.get("inputs")
                if isinstance(inputs, dict):
                    prompt = inputs.get("prompt")
                    image_input = inputs.get("image")
                else:
                    return {"error": "Expected 'inputs' to be a JSON object containing 'prompt' and 'image'."}
        else:
            return {"error": "Input payload must be a JSON object."}

        if not prompt:
            return {"error": "Missing 'prompt' in input data."}
        if not image_input:
            return {"error": "Missing 'image' (base64) in input data."}

        # Decode image from base64 and correct orientation
        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = ImageOps.exif_transpose(image)  # Correct EXIF orientation here
        except Exception as e:
            return {"error": f"Failed to decode 'image' as base64: {str(e)}"}

        # Debug prints for prompt and image size
        print(f"📝 Final prompt: {prompt}")
        print(f"🖼️ Image size: {image.size}")

        # Generate edited image with Kontext
        try:
            output = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                guidance_scale=2.0
            ).images[0]
            print("🎨 Image generated.")

            # 💡 HARD CLAMP pixel values to [0, 255] to prevent NaN/black outputs
            output_array = np.array(output)
            output_array = np.clip(output_array, 0, 255).astype(np.uint8)
            output = Image.fromarray(output_array)
            print("🛑 Hard clamped output pixel values to [0, 255].")

        except Exception as e:
            return {"error": f"Model inference failed: {str(e)}"}

        # Encode output image to base64
        try:
            buffer = BytesIO()
            output.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            print("✅ Returning image.")
            return {"image": base64_image}
        except Exception as e:
            return {"error": f"Failed to encode output image: {str(e)}"}
