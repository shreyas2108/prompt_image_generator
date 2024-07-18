
def gradio_interface(prompt, seed, guidance_scale, steps):
    image = generate_image(prompt, seed, guidance_scale, steps)
    return image

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(0, 10000, step=1, value=42, label="Seed"),
        gr.Slider(0.1, 20.0, step=0.1, value=7.5, label="Guidance Scale"),
        gr.Slider(1, 100, step=1, value=50, label="Steps")
    ],
    outputs=gr.Image(type="pil"),
    title="Prompt Image Generator"
)

interface.launch()
