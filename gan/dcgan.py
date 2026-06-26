"""
DCGAN — the original (2015) convolutional GAN, sized for 64x64 RGB faces.

Generator: maps a latent vector (latent_dim, 1, 1) -> image (3, 64, 64) via a stack
of transposed convolutions, with a tanh output (range [-1, 1]).
Discriminator: a convolutional classifier, image -> probability of being real (sigmoid).

Train with the standard non-saturating GAN objective (binary cross-entropy).
"""
import torch
import torch.nn as nn


def weights_init(m):
    """DCGAN init: conv weights ~ N(0, 0.02), batchnorm weight ~ N(1, 0.02), bias 0."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class Generator(nn.Module):
    def __init__(self, latent_dim=100, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, kernel_size=4, stride=1, padding=0, bias=False),  # -> 4x4
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1, bias=False),  # -> 8x8
            nn.BatchNorm2d(base * 4), 
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1, bias=False),  # -> 16x16
            nn.BatchNorm2d(base * 2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1, bias=False),  # -> 32x32
            nn.BatchNorm2d(base), 
            nn.ReLU(True),
            nn.ConvTranspose2d(base, 3, kernel_size=4, stride=2, padding=1, bias=False),  # -> 64x64
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1, bias=False),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),  # 32 -> 16
            nn.BatchNorm2d(base * 2), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),  # 16 -> 8
            nn.BatchNorm2d(base * 4), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),  # 8 -> 4
            nn.BatchNorm2d(base * 8), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 0, bias=False),  # 4 -> 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)
        