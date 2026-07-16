"""
StyleCLIP-style latent mapper: a small network trained to produce a W+ offset that applies
one specific facial edit (e.g. "make the person smile").

The pretrained StyleGAN2 stays frozen - only the mapper trains. Supervision comes from CLIP:
the shift the edit causes in CLIP *image* space (delta_I) should point the same way as the
shift between two text prompts in CLIP *text* space (delta_T), e.g.
"a photo of a person" -> "a photo of a smiling person".

Following StyleCLIP (section 5), the mapper is split into three sub-networks acting on
different parts of W+ - coarse, medium and fine latents - because those groups control
different aspects of the image (pose/shape, features/hair, color/lighting).

Simplifications vs. the paper: cosine similarity between delta_I and delta_T is optimized
directly, and the ArcFace identity loss is dropped.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
        

class PixelNorm(nn.Module):
    """Normalize each latent vector to unit RMS - StyleGAN's standard input normalization."""
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)


class SubMapper(nn.Module):
    """MLP applied to every W vector of one group (coarse / medium / fine)."""
    def __init__(self, w_dim=512, num_layers=4):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(num_layers):
            layers += [nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)
        # Start near zero: an untrained mapper should leave the face alone, not wreck it.
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, x):  # (B, n, w_dim) -> same shape
        return self.net(x)


class LatentMapper(nn.Module):
    """Maps a W+ latent to the offset delta_wplus that performs the learned edit.

    Defaults match StyleGAN2 at 256x256 (num_ws = 14): 6 coarse + 4 medium + 4 fine latents.
    """
    def __init__(self, num_coarse=3 * 2, num_medium=2 * 2, num_fine=2 * 2, num_latents=14, num_layers=4, w_dim=512):
        super().__init__()
        assert num_coarse + num_medium + num_fine == num_latents, 'group sizes must sum to num_latents'
        self.num_coarse, self.num_medium, self.num_fine = num_coarse, num_medium, num_fine
        self.coarse = SubMapper(w_dim, num_layers)
        self.medium = SubMapper(w_dim, num_layers)
        self.fine   = SubMapper(w_dim, num_layers)

    def forward(self, wplus):  # (B, num_latents, w_dim) -> (B, num_latents, w_dim)
        a, b = self.num_coarse, self.num_coarse + self.num_medium
        return torch.cat([
            self.coarse(wplus[:, :a]),
            self.medium(wplus[:, a:b]),
            self.fine(wplus[:, b:]),
        ], dim=1)


def clip_direction_loss(delta_I, delta_T):
    """Push the CLIP image-space shift to align with the CLIP text-space shift."""
    return (1.0 - F.cosine_similarity(delta_T, delta_I)).mean()


def magnitude_loss(delta_wplus):
    """Keep the W+ offset small, so the edit changes the attribute and not the identity."""
    scaled = delta_wplus / math.sqrt(delta_wplus.shape[1] * delta_wplus.shape[2])
    return scaled.flatten(1).abs().sum(dim=1).mean()


class CLIPEmbedder:
    """Wraps CLIP for the two embeddings the mapper needs: images and text prompts."""
    def __init__(self, device, model_id='openai/clip-vit-base-patch32'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_id).eval().to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id).eval().to(device)
        self.mean = torch.tensor(self.processor.image_mean, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(self.processor.image_std,  device=device).view(1, 3, 1, 1)

    def embed_image(self, tensor):
        """(B, 3, H, W) in [-1, 1] (StyleGAN2's output range) -> CLIP image embedding.
        Kept differentiable: gradients flow back through the generator into the mapper."""
        x = 0.5 * (tensor + 1.0)  # -> [0, 1]
        x = (x - self.mean) / self.std  # CLIP normalization
        x = F.interpolate(x, (224, 224), mode='area')  # CLIP input size
        return self.vision_model(pixel_values=x).image_embeds

    @torch.no_grad()
    def embed_text(self, text):
        tokens = self.tokenizer([text], padding=True, return_tensors='pt').to(self.device)
        return self.text_model(**tokens).text_embeds

    @torch.no_grad()
    def text_direction(self, text_before, text_after):
        """delta_T - the direction of the edit in CLIP text space."""
        return self.embed_text(text_after) - self.embed_text(text_before)
