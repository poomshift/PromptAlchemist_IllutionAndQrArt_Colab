import gradio as gr
from PIL import Image, ImageOps
import torch
from transformers import pipeline
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    #UniPCMultistepScheduler,
)

from diffusers.utils import load_image

# Model loading (do this only once at the start)
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_single_file(
    "/content/CyberRealistic_V4.1_FP16.safetensors", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to("cuda")  # Assuming you have a CUDA-enabled GPU

def invert_image(image):
    """Inverts the colors of a PIL Image."""
    return ImageOps.invert(image)

# Define the main Gradio interface function
def generate_image(prompt, negative_prompt, qrcode_image, width=768, height=768, 
                   guidance_scale=7.5, controlnet_conditioning_scale=1.8, num_inference_steps=30, 
                   seed=33, invert_colors=False):

    # Load and resize the QR code image
    #qrcode_image = load_image(qrcode_image).resize((width, height)) 

    # Invert colors if the checkbox is selected
    if invert_colors:
        qrcode_image = invert_image(qrcode_image)
                                    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    return output.images[0]

# Gradio interface creation
with gr.Blocks(title="Illusion|QR Code Art") as demo:
    gr.Markdown("""# Illusion|QR Code Art""")   
    gr.Markdown("สร้างภาพลวงตาและ QrCode ด้วย Ai | fb.com/PromptAlchemist")  
    with gr.Row():
        prompt = gr.Textbox(label="Text Prompt", value="Vegetable cheese pizza, melted mozzarella, golden-brown crust, bubbling cheese surface, freshly baked, Italian cuisine, added vegetables: bell peppers, onions, black olives, cherry tomatoes, spinach, photorealistic")
        negative_prompt= gr.Textbox(label="Negative Prompt", value="Low quality, bad quality, worst quality, 3d, cartoon, painting, bad anatomy, NSFW, Nudity")
        width = gr.Number(label="Width (กว้าง)", value=768)
        height = gr.Number(label="Height(สูง)", value=768) 
    with gr.Row():   
        guidance_scale = gr.Slider(label="CFG", value=7.5, minimum=0, maximum=10.0)
        controlnet_conditioning_scale = gr.Slider(label="ControlNet Weight", value=1.7, minimum=0, maximum=2.0)
        seed = gr.Number(label="Seed", value=1234)
        num_inference_steps = gr.Number(label="Steps", value=30)
        invert_colors = gr.Checkbox(label="สลับสีภาพ Input", value=True)
    generate_button = gr.Button("Generate Image")

    with gr.Row():
        qrcode_image = gr.Image(label="QR Code Image (Required)", type="pil", height=768, width=768)
        output_image = gr.Image(label="Output", type="pil", height=768, width=768) 
    
    generate_button.click(fn=generate_image, 
                      inputs=[prompt, negative_prompt, qrcode_image, width, height, guidance_scale, controlnet_conditioning_scale,num_inference_steps, seed,invert_colors,],
                      outputs=output_image) 

    demo.launch(share=True)
