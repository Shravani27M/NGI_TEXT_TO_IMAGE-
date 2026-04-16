import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.title("🎨 AI Text to Image Generator")
st.write("Enter a prompt and generate an image!")

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        try:
            with st.spinner("Loading model... ⏳"):
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo",   # latest fast model
                    torch_dtype=torch.float16
                )
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            with st.spinner("Generating image... 🎨"):
                image = pipe(prompt, num_inference_steps=1).images[0]

            st.success("Here is your generated image:")
            st.image(image, caption=prompt)

        except Exception as e:
            st.error(f"Error: {e}")
