import streamlit as st
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

st.title("Stable Video Diffusion Demo")

# Load the model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    image = image.resize((1024, 576))

    # Generate video frames
    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

    # Display generated video
    st.video(frames)

    # Save the video (optional)
    if st.button("Save Video"):
        export_to_video(frames, "generated_video.mp4", fps=7)
        st.success("Video saved successfully!")
