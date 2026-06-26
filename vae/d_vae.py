import torch
import torch.nn as nn


class GumbelQuantizer(nn.Module):
    """
    Differentiable discrete bottleneck via Gumbel-Softmax.

    Instead of picking the nearest codebook vector with argmin (not differentiable),
    the encoder outputs logits over the n_e codes. We add Gumbel noise, apply a
    temperature-scaled softmax, and return a weighted mix of codebook vectors — so
    gradients flow straight to the encoder.

    Temperature sets how hard the choice is: low -> near one-hot, high -> near uniform.
    Usually annealed from high to low during training.
    """
    def __init__(self, n_e=256, e_dim=8):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, logits, temperature=1.0):
        # logits: (N, n_e) - encoder scores over codebook entries, one row per position.
        # Gumbel noise: -log(-log(U)); the eps terms keep the logs finite.
        U = torch.rand_like(logits)
        G = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        # Gumbel-Softmax weights. Subtracting the row max prevents exp() overflow;
        # softmax is shift-invariant, so the result is unchanged.
        z = logits + G
        weights = ((z - z.max(dim=-1, keepdim=True)[0]) / temperature).softmax(dim=-1)
        # Soft combination of codebook vectors.
        z_q = weights @ self.embedding.weight  # (N, n_e) @ (n_e, e_dim) -> (N, e_dim)
        return z_q


class TinnyDVAE(nn.Module):
    def __init__(self, n_e=256, e_dim=8):
        super().__init__()
        tmp_ch = 64

        self.encoder = nn.ModuleDict()
        self.encoder['in_conv'] = nn.Conv2d(3, tmp_ch, 1)
        for i in range(2):
            self.encoder[f'block_{i}'] = nn.Sequential(
                nn.Conv2d(tmp_ch, tmp_ch * 2, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(tmp_ch * 2)
            )
            tmp_ch = tmp_ch * 2
        self.encoder['out_conv'] = nn.Conv2d(tmp_ch, n_e, 1)

        self.gumbel_quantize = GumbelQuantizer(n_e=n_e, e_dim=e_dim)

        self.decoder = nn.ModuleDict()
        self.decoder['in_conv'] = nn.Conv2d(e_dim, tmp_ch, 1)
        for i in range(2):
            self.decoder[f'block_{i}'] = nn.Sequential(
                nn.Conv2d(tmp_ch, tmp_ch // 2, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(tmp_ch // 2),
                nn.Upsample(scale_factor=2)
            )
            tmp_ch = tmp_ch // 2
        self.decoder['out_conv'] = nn.Conv2d(tmp_ch, 3, 1)

    def encode(self, x):
        z = x
        for block in self.encoder.values():
            z = block(z)
        z = z.permute(0, 2, 3, 1)  # (BS, H, W, n_e)
        B, H, W, _ = z.shape
        return z.reshape(-1, self.gumbel_quantize.n_e), (B, H, W)

    def decode(self, z_q, shape):
        B, H, W = shape
        e_dim = self.gumbel_quantize.e_dim
        z_q = z_q.reshape(B, H, W, e_dim).permute(0, 3, 1, 2)  # (BS, e_dim, H, W)
        for block in self.decoder.values():
            z_q = block(z_q)
        return z_q

    def forward(self, x, temperature=1.0):
        logits, shape = self.encode(x)
        z_q = self.gumbel_quantize(logits, temperature)
        reconstruction = self.decode(z_q, shape)
        return reconstruction