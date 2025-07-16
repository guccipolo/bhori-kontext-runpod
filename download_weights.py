import torch
from diffusers import FluxKontextPipeline
import os

def fetch_pretrained_model(model_class, model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise

def get_flux_pipeline():
    """
    Fetches the Flux Kontext pipeline from the HuggingFace model hub.
    """
    print("🚀 Downloading Flux Kontext pipeline...")
    
    # Download the main model
    pipe = fetch_pretrained_model(
        FluxKontextPipeline,
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        variant="fp16",
        use_safetensors=True,
    )
    
    print("✅ Flux Kontext pipeline downloaded successfully")
    
    # Check if LoRA files exist locally
    lora_files = ["Bh0r1.safetensors", "Bh0r12.safetensors"]
    for lora_file in lora_files:
        if os.path.exists(lora_file):
            print(f"✅ Found LoRA file: {lora_file}")
        else:
            print(f"⚠️ LoRA file not found: {lora_file}")
    
    return pipe

if __name__ == "__main__":
    get_flux_pipeline()
    print("🎉 All models downloaded and ready!")