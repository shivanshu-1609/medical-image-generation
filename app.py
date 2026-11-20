"""Streamlit interface for synthetic medical image generation."""

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from generate import MEDIGAN_MODELS, generate
from PIL import Image


def main():
    st.title("Synthetic Medical Images")
    st.caption("Educational image synthesis; outputs are not clinically validated.")
    source = st.sidebar.selectbox("Generator", ["Trained GAN", "MediGAN"])
    checkpoint, model_id = None, None
    if source == "Trained GAN":
        checkpoint = Path(st.sidebar.text_input("Checkpoint", "outputs/gan/gan.pt"))
    else:
        model_id = st.sidebar.selectbox("Model", MEDIGAN_MODELS)
    count = st.sidebar.slider("Samples", 1, 8, 4)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=2147483647, value=42)
    install = (
        st.sidebar.checkbox("Install model dependencies", value=False)
        if source == "MediGAN"
        else False
    )
    if st.button("Generate"):
        try:
            with (
                st.spinner("Generating images"),
                tempfile.TemporaryDirectory() as temporary,
            ):
                paths = generate(
                    Path(temporary), count, int(seed), checkpoint, model_id, install
                )
                st.session_state["images"] = [
                    (path.name, path.read_bytes()) for path in paths
                ]
        except Exception as error:
            st.error(str(error))
    images = st.session_state.get("images", [])
    if images:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            for index, (name, content) in enumerate(images):
                with Image.open(io.BytesIO(content)) as image:
                    st.image(image.copy(), caption=name)
                handle.writestr(f"{index:03d}_{name}", content)
        st.download_button(
            "Download images",
            archive.getvalue(),
            "synthetic_images.zip",
            "application/zip",
        )


if __name__ == "__main__":
    main()
