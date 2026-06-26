"""
VQ-GAN pieces. The generator is the VQ-VAE - VQ-GAN is a VQ-VAE trained with two extra losses: 
a perceptual loss (LPIPS) and an adversarial loss from a PatchGAN discriminator. 
This module supplies the discriminator and the GAN loss helpers.

PatchGAN discriminator: instead of one real/fake score per image, it outputs a small
grid of scores, one per overlapping patch - it judges local texture, which is what
makes reconstructions look sharp.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for i in range(1, n_layers):
            ch_next = min(base * (2 ** i), base * 8)
            layers += [
                nn.Conv2d(ch, ch_next, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        # one stride-1 layer, then collapse to a 1-channel patch-score map
        ch_next = min(ch * 2, base * 8)
        layers += [
            nn.Conv2d(ch, ch_next, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ch_next),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_next, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (N, 1, h, w) patch logits

def hinge_d_loss(real_logits, fake_logits):
    """Discriminator hinge loss: push real scores > +1 and fake scores < -1"""
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

def hinge_g_loss(fake_logits):
    """Generator hinge loss: maximize the discriminator's score on fakes."""
    return -fake_logits.mean()
