import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🎨 AI Text to Image Generator")
st.write("Enter a prompt and generate an image!")

# Input
prompt = st.text_input("Enter your prompt:")

# Button
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        try:
            with st.spinner("Loading model... ⏳ (first time may take time)"):
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16
                )
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            with st.spinner("Generating image... 🎨"):
                image = pipe(prompt).images[0]

            st.success("Here is your generated image:")
            st.image(image, caption=prompt)

        except Exception as e:
            st.error(f"Error: {e}")
