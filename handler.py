from typing import Dict
import torch
from diffusers import FluxKontextPipeline
from io import BytesIO
import base64
from PIL import Image

class EndpointHandler:
    def __init__(self, path: str = ""):
        print("🚀 Initializing Flux Kontext pipeline...")

        # Load Flux Kontext model from Hugging Face Hub
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",  # replace if using your own model repo
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Model ready.")

    def __call__(self, data: Dict) -> Dict:
        print("🔧 Received data:", data)

        # Validate data structure
        inputs = data.get("inputs")
        if not inputs or not isinstance(inputs, dict):
            return {"error": "'inputs' must be a JSON object containing 'prompt' and 'image'."}

        prompt = inputs.get("prompt")
        image_input = inputs.get("image")

        if not prompt:
            return {"error": "'prompt' is required in 'inputs'."}
        if not image_input:
            return {"error": "'image' (base64 encoded string) is required in 'inputs'."}

        # Decode image from base64
        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to decode 'image' input as base64: {str(e)}"}

        # Generate edited image with Kontext
        try:
            output = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=28,  # Kontext standard
                guidance_scale=3.5
            ).images[0]
            print("🎨 Image generated.")
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
