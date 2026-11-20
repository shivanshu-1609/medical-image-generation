"""Generate images with a trained GAN or a MediGAN model."""

import argparse
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from gan import Generator, device
from torchvision.utils import save_image

MEDIGAN_MODELS = (
    "00001_DCGAN_MMG_CALC_ROI",
    "00002_DCGAN_MMG_MASS_ROI",
    "00003_CYCLEGAN_MMG_DENSITY_FULL",
    "00004_PIX2PIX_MMG_MASSES_W_MASKS",
    "00019_PGGAN_CHEST_XRAY",
)


@lru_cache(maxsize=1)
def medigan_models():
    from medigan import Generators

    return Generators()


def generate(
    output,
    count=4,
    seed=42,
    checkpoint=None,
    medigan_model=None,
    install_dependencies=False,
):
    if count < 1 or (checkpoint is None) == (medigan_model is None):
        raise ValueError("Choose exactly one model source and a positive sample count.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    if medigan_model is not None:
        model_id = int(medigan_model) if str(medigan_model).isdigit() else medigan_model
        output.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        medigan_models().generate(
            model_id=model_id,
            num_samples=count,
            output_path=str(output),
            install_dependencies=install_dependencies,
        )
    else:
        target_device = device()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = Generator(state["latent_size"]).to(target_device).eval()
        model.load_state_dict(state["generator"])
        noise = torch.randn(
            count, state["latent_size"], generator=torch.Generator().manual_seed(seed)
        ).to(target_device)
        output.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for index, image in enumerate(model(noise)):
                save_image(
                    image,
                    output / f"sample_{index:04d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise RuntimeError("The model did not write supported image files.")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--checkpoint", type=Path)
    models.add_argument("--medigan-model")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--install-dependencies", action="store_true")
    args = parser.parse_args()
    try:
        paths = generate(
            args.output,
            args.count,
            args.seed,
            args.checkpoint,
            args.medigan_model,
            args.install_dependencies,
        )
        print(f"Generated {len(paths)} image files in {args.output}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
