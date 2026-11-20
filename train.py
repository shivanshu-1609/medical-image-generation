"""Train a vanilla GAN on authorized grayscale medical training images."""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from gan import Discriminator, Generator, device
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


class ImageDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise ValueError("No JPG or PNG images found in the data directory.")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def train(args):
    if (
        min(args.epochs, args.batch_size, args.latent_size) < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("Training settings must be positive and finite.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    torch.manual_seed(args.seed)
    target_device = device()
    dataset = ImageDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    generator, discriminator = (
        Generator(args.latent_size).to(target_device),
        Discriminator().to(target_device),
    )
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    criterion = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    fixed_noise = torch.randn(16, args.latent_size, device=target_device)
    history = []
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_total, d_total = 0.0, 0.0
        for real in loader:
            real = real.to(target_device)
            count = len(real)
            ones, zeros = (
                torch.ones(count, 1, device=target_device),
                torch.zeros(count, 1, device=target_device),
            )
            noise = torch.randn(count, args.latent_size, device=target_device)
            fake = generator(noise)
            d_optimizer.zero_grad(set_to_none=True)
            d_loss = (
                criterion(discriminator(real), ones)
                + criterion(discriminator(fake.detach()), zeros)
            ) / 2
            d_loss.backward()
            d_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad = False
            g_optimizer.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake), ones)
            g_loss.backward()
            g_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad = True
            g_total += g_loss.item() * count
            d_total += d_loss.item() * count
        g_loss, d_loss = g_total / len(dataset), d_total / len(dataset)
        if not math.isfinite(g_loss) or not math.isfinite(d_loss):
            raise ValueError("Training produced non-finite losses.")
        history.append(
            {"epoch": epoch, "generator_loss": g_loss, "discriminator_loss": d_loss}
        )
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        torch.save(
            {
                "latent_size": args.latent_size,
                "epoch": epoch,
                "seed": args.seed,
                "generator": {
                    key: value.detach().cpu()
                    for key, value in generator.state_dict().items()
                },
                "discriminator": {
                    key: value.detach().cpu()
                    for key, value in discriminator.state_dict().items()
                },
            },
            args.output / "gan.pt",
        )
        generator.eval()
        with torch.inference_mode():
            save_image(
                generator(fixed_noise),
                args.output / "samples.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
        print(f"Epoch {epoch}: generator={g_loss:.4f}, discriminator={d_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/gan"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
