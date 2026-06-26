"""
Image-quality metrics for benchmarking generative models: PSNR, SSIM, LPIPS, FID.

Canonical input convention
---------------------------
Every metric takes images as a float tensor/array in [0, 1], shape (N, 3, H, W).
Each metric converts internally to whatever scale it needs (PSNR on [0,1], LPIPS on
[-1,1], FID through InceptionV3 preprocessing), so callers normalize once and scores
stay comparable across models.

PSNR / SSIM -> reference metrics, compare paired images (e.g. original vs reconstruction).
LPIPS -> learned perceptual distance, also paired (needs the `lpips` package).
FID -> compares two *distributions* of images (real set vs generated set).
"""
import numpy as np
import torch
import torch.nn.functional as F
import scipy.linalg
import lpips
from torchvision.models import inception_v3, Inception_V3_Weights

from .utils import return_device


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def psnr(x, y, data_range=1.0):
    """Mean PSNR (dB) over a batch of paired images. Higher is better.
    
    Peak Signal-to-Noise Ratio (PSNR) — a logarithmic quantity measured in decibels that quantifies 
    the ratio of the maximum possible signal power to the noise level.

    PSNR focuses on differences measured at the pixel level, which is why it correlates poorly with visual perception. 
    For example, a small shift in pixels can cause a large degradation in the metric, even though the shift is barely noticeable to the eye.
    """
    x, y = _to_numpy(x), _to_numpy(y)
    mse = ((x - y) ** 2).reshape(x.shape[0], -1).mean(axis=1)
    mse = np.clip(mse, 1e-12, None)                  # avoid log(0) on identical images
    return float(np.mean(10 * np.log10(data_range ** 2 / mse)))


def ssim(x, y, data_range=1.0, c1=0.01, c2=0.03, c3=0.015):
    """Mean (global) SSIM over a batch of paired images. Higher is better.
    
    Structural Similarity Index (SSIM) — a metric that evaluates the degree of similarity between 
    two images, taking into account their luminance, contrast, and structure.

    It better reflects human visual perception and more accurately assesses the structural characteristics of 
    images compared to, for example, PSNR.

    SSIM estimates luminance as the mean of the image's pixels, and contrast as the standard deviation.
    """
    x, y = _to_numpy(x), _to_numpy(y)
    c1, c2, c3 = c1 * data_range, c2 * data_range, c3 * data_range
    scores = []
    for a, b in zip(x, y):
        mu_a, mu_b = a.mean(), b.mean()
        sd_a = np.sqrt(((a - mu_a) ** 2).sum() / (a.size - 1))
        sd_b = np.sqrt(((b - mu_b) ** 2).sum() / (b.size - 1))
        cov = ((a - mu_a) * (b - mu_b)).sum() / (a.size - 1)
        luminance = (2 * mu_a * mu_b + c1) / (mu_a ** 2 + mu_b ** 2 + c1)
        contrast  = (2 * sd_a * sd_b + c2) / (sd_a ** 2 + sd_b ** 2 + c2)
        structure = (cov + c3) / (sd_a * sd_b + c3)
        scores.append(luminance * contrast * structure)
    return float(np.mean(scores))

@torch.no_grad()
def lpips_distance(x, y, net='alex'):
    """Mean LPIPS perceptual distance over paired images. Lower is better.
    x, y: (N, 3, H, W) in [0, 1]. Requires the `lpips` package."""
    device = return_device()
    model = lpips.LPIPS(net=net).to(device).eval()
    
    # [0,1] -> [-1,1]
    x = torch.as_tensor(x, dtype=torch.float32, device=device) * 2 - 1
    y = torch.as_tensor(y, dtype=torch.float32, device=device) * 2 - 1
    return float(model(x, y).mean().item())

def reconstruction_metrics(originals, reconstructions):
    """Dict of paired metrics on (N,3,H,W) [0,1] images: PSNR, SSIM, LPIPS."""
    return {
        'PSNR': psnr(originals, reconstructions),
        'SSIM': ssim(originals, reconstructions),
        'LPIPS': lpips_distance(originals, reconstructions),
    }

@torch.no_grad()
def inception_features(images, batch_size=64):
    """Extract 2048-d InceptionV3 features. images: (N, 3, H, W) in [0,1] -> (N, 2048) array."""
    device = return_device()

    m = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    m.fc = torch.nn.Identity()
    model = m.eval().to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = torch.as_tensor(images, dtype=torch.float32)
    feats = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)

def fid(real_images, fake_images, batch_size=64):
    """Fréchet distance between two feature sets, each (N, D) array. Lower is better."""
    fr = inception_features(real_images, batch_size)
    ff = inception_features(fake_images, batch_size)

    mu1, mu2 = fr.mean(0), ff.mean(0)
    cov1, cov2 = np.cov(fr, rowvar=False), np.cov(ff, rowvar=False)
    diff = ((mu1 - mu2) ** 2).sum()
    covmean = scipy.linalg.sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return float(diff + np.trace(cov1 + cov2 - 2 * covmean))
