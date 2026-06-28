"""
ESRGAN - Enhanced Super-Resolution GAN. Takes a low-resolution image and produces 
a high-resolution one with realistic texture.

Generator (RRDBNet): a stack of Residual-in-Residual Dense Blocks. Two changes 
from SRGAN that the ESRGAN: no BatchNorm (more stable, fewer artifacts) and 
the RRDB block (residual-in-residual with dense connections). Upsampling is done with PixelShuffle.

Discriminator: a VGG-style classifier, used as a relativistic discriminator (RaGAN) - it predicts
how much more realistic a real image is than a fake one, rather than an absolute real/fake score.

Generator loss = L1 (content) + perceptual (VGG features, before activation) + relativistic adversarial.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Generator (RRDBNet)
class DenseBlock(nn.Module):
    """5-conv dense block with residual scaling."""
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 dense blocks chained, plus an outer residual."""
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.db1 = DenseBlock(nf, gc, res_scale)
        self.db2 = DenseBlock(nf, gc, res_scale)
        self.db3 = DenseBlock(nf, gc, res_scale)

    def forward(self, x):
        out = self.db3(self.db2(self.db1(x)))
        return x + out * self.res_scale


class RRDBNet(nn.Module):
    """ESRGAN generator"""
    def __init__(self, in_ch=3, out_ch=3, nf=64, n_blocks=16, gc=32, scale=4):
        super().__init__()
        assert scale in (2, 4, 8), "scale must be 2, 4 or 8"
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(n_blocks)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        ups = []
        for _ in range(int(math.log2(scale))):  # one PixelShuffle(x2) per factor of 2
            ups += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True)]
        self.upsampling = nn.Sequential(*ups)
        self.conv_hr   = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))  # global residual over the trunk
        feat = self.upsampling(feat)
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class Discriminator(nn.Module):  # (VGG-style, 128x128)
    """VGG-style discriminator for 128x128 inputs. Returns a raw logit per image."""
    def __init__(self, in_ch=3, nf=64):
        super().__init__()

        def block(i, o, s):
            return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True)]

        layers = [nn.Conv2d(in_ch, nf, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        layers += block(nf, nf, 2)  # 128 -> 64
        layers += block(nf, nf * 2, 1)
        layers += block(nf * 2, nf * 2, 2)  # 64 -> 32
        layers += block(nf * 2, nf * 4, 1)
        layers += block(nf * 4, nf * 4, 2)  # 32 -> 16
        layers += block(nf * 4, nf * 8, 1)
        layers += block(nf * 8, nf * 8, 2)  # 16 -> 8
        layers += block(nf * 8, nf * 8, 1)
        layers += block(nf * 8, nf * 8, 2)  # 8 -> 4
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(nf * 8 * 4 * 4, 100), nn.LeakyReLU(0.2, inplace=True), nn.Linear(100, 1))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)).flatten()


# Perceptual loss (VGG19, pre-activation)
class VGGPerceptual(nn.Module):
    """VGG19 feature extractor for perceptual loss. ESRGAN uses features *before* the final
    activation; layer_index=34 ends at conv5_4 (the conv before its ReLU). Input in [0, 1]."""
    def __init__(self, layer_index=34):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer_index + 1]
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.vgg((x - self.mean) / self.std)


def perceptual_loss(vgg, sr, hr):
    return F.l1_loss(vgg(sr), vgg(hr))


# Relativistic GAN losses (RaGAN)
def ragan_d_loss(real_logits, fake_logits):
    """Discriminator: real should look more realistic than the average fake (and vice-versa)."""
    rl = real_logits - fake_logits.mean()
    fl = fake_logits - real_logits.mean()
    return 0.5 * (F.binary_cross_entropy_with_logits(rl, torch.ones_like(rl)) +
                  F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl)))


def ragan_g_loss(real_logits, fake_logits):
    """Generator: push fakes to look more realistic than the average real."""
    rl = real_logits - fake_logits.mean()
    fl = fake_logits - real_logits.mean()
    return 0.5 * (F.binary_cross_entropy_with_logits(rl, torch.zeros_like(rl)) +
                  F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl)))
