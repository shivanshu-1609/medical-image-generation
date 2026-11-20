"""Vanilla GAN for 64 by 64 grayscale images."""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 64 * 64),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise).reshape(-1, 1, 64, 64)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        return self.network(images)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
