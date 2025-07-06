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
            "black-forest-labs/FLUX.1-Kontext-dev",  # replace with your specific Kontext model if different
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Model ready.")

    def __call__(self, data: Dict) -> Dict:
        print("🔧 Received data:", data)

        inputs = data.get("inputs", {})
        prompt = inputs.get("prompt")
        image_base64 = inputs.get("image")

        if not prompt or not image_base64:
            return {"error": "Both 'prompt' and 'image' inputs are required."}

        # Decode input image from base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Generate edited image with Kontext
        output = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=28,  # context standard
            guidance_scale=3.5
        ).images[0]

        print("🎨 Image generated.")

        # Encode output image to base64
        buffer = BytesIO()
        output.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("✅ Returning image.")
        return {"image": base64_image}
