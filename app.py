!pip install gradio torch diffusers transformers Pillow

import os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr


os.environ['HUGGINGFACE_TOKEN'] = "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"
auth_token = os.getenv('HUGGINGFACE_TOKEN')
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=auth_token
)
pipe.to(device)

def generate_image(prompt, seed, guidance_scale, steps):
    generator = torch.manual_seed(seed) if seed is not None else None

    with torch.autocast("cuda"):
        output = pipe(
            prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=steps
        )

    image = output.images[0]
    return image
