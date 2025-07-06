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

        inputs = data.get("inputs")
        if not inputs:
            return {"error": "'inputs' key missing. Payload must include an 'inputs' dictionary."}

        if not isinstance(inputs, dict):
            return {"error": "'inputs' must be a JSON object with 'prompt' and optionally 'image'."}

        prompt = inputs.get("prompt")
        image_input = inputs.get("image")

        if not prompt:
            return {"error": "Prompt is required in 'inputs'."}

        # Process image input if provided
        image = None
        if image_input:
            if isinstance(image_input, str):
                try:
                    # Assume it's base64 encoded
                    image_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    return {"error": f"Failed to decode base64 image input: {str(e)}"}
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return {"error": "'image' must be a base64 string or a PIL.Image object."}

        # Generate edited image with Kontext
        try:
            output = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=28,  # context standard
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
