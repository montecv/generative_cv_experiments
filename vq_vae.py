import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class _VectorQuantizerBase(nn.Module):
    """
    Shared codebook storage + nearest-neighbour lookup for both VQ variants.
    The codebook is always an nn.Embedding, so the public interface (.embedding.weight, .init_from_data, .get_code_indices) is identical for the gradient and the EMA version.
    That lets the notebook stay the same either way.
    """
    def __init__(self, n_e=512, e_dim=8, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
 
    def get_code_indices(self, z_e):
        """Nearest codebook index for each row of z_e -> (BS*H*W,)"""
        w = self.embedding.weight
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b^T  (matrix form, no python loops)
        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)
        emb_sq = (w ** 2).sum(dim=1)
        dot    = z_e @ w.t()
        distances = z_e_sq + emb_sq - 2 * dot
        return distances.argmin(dim=1)
 
    @torch.no_grad()
    def init_from_data(self, vectors):
        """Initialize the codebook from real encoder outputs. vectors: (n_e, e_dim)."""
        self.embedding.weight.data.copy_(vectors)
 
 
class VectorQuantizer(_VectorQuantizerBase):
    """Original VQ-VAE: codebook is trained by gradient (two-term loss)."""
    def forward(self, z_e):
        indices = self.get_code_indices(z_e)
        z_q = self.embedding(indices)
 
        # codebook_loss -> pull codebook toward encoder output (stop-grad on z_e)
        # commitment_loss -> pull encoder output toward codebook (stop-grad on z_q)
        codebook_loss   = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
 
        # Straight-through: forward value is z_q, gradient flows to z_e.
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss
 
 
class VectorQuantizerEMA(_VectorQuantizerBase):
    """
    VQ-VAE with EMA codebook updates. 
    The codebook is NOT trained by the optimizer; each code tracks the exponential moving average of the encoder vectors assigned to it. 
    Robust against codebook collapse. The loss keeps only the commitment term.
    """
    def __init__(self, n_e=512, e_dim=8, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__(n_e, e_dim, beta)
        self.decay = decay
        self.eps = eps
        # Codebook updated by EMA, not by Adam.
        self.embedding.weight.requires_grad_(False)
        self.register_buffer('cluster_size', torch.ones(n_e))  # EMA of counts
        self.register_buffer('ema_w', self.embedding.weight.data.clone())  # EMA of summed vectors
 
    @torch.no_grad()
    def init_from_data(self, vectors):
        # Keep embedding, ema_w and cluster_size consistent so the data-init survives
        # the first EMA step (embedding = ema_w / cluster_size = vectors / 1).
        self.embedding.weight.data.copy_(vectors)
        self.ema_w.copy_(vectors)
        self.cluster_size.fill_(1.0)
 
    def forward(self, z_e):
        indices = self.get_code_indices(z_e)
        z_q = self.embedding(indices)
 
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.n_e).type_as(z_e)  # (N, n_e)
                counts = encodings.sum(dim=0)  # (n_e,)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                dw = encodings.t() @ z_e  # (n_e, e_dim)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                # Laplace smoothing so empty codes don't blow up the division.
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
 
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commitment_loss
 
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss
 
    @torch.no_grad()
    def reset_dead_codes(self, z_e, threshold=1.0):
        """
        Move barely-used codes onto random encoder vectors from the current batch.
        Returns the number of codes reset. Call occasionally during training if collapse persists.
        """
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        perm = torch.randperm(z_e.shape[0])[:n_dead]
        self.embedding.weight.data[dead] = z_e[perm]
        self.ema_w[dead] = z_e[perm]
        self.cluster_size[dead] = 1.0
        return n_dead
 
 
class TinnyVQVAE(nn.Module):
    def __init__(self, n_e=512, e_dim=8, use_ema=True):
        super().__init__()
        tmp_ch = 6
 
        self.encoder = nn.ModuleDict()
        self.encoder['in_conv'] = nn.Conv2d(3, tmp_ch, 1)
        for i in range(2):
            self.encoder[f'block_{i}'] = nn.Sequential(
                nn.Conv2d(tmp_ch, tmp_ch * 2, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(tmp_ch * 2)
            )
            tmp_ch = tmp_ch * 2
        self.encoder['out_conv'] = nn.Conv2d(tmp_ch, e_dim, 1)
 
        Quant = VectorQuantizerEMA if use_ema else VectorQuantizer
        self.quantize = Quant(n_e=n_e, e_dim=e_dim)
 
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
        """Run the encoder and flatten to (BS*H*W, e_dim). Returns z_e and the (B,H,W,C) shape."""
        z_e = x
        for block in self.encoder.values():
            z_e = block(z_e)
        z_e = z_e.permute(0, 2, 3, 1)  # (BS, H, W, C)
        B, H, W, C = z_e.shape
        return z_e.reshape(-1, C), (B, H, W, C)
 
    def decode(self, z_q, shape):
        B, H, W, C = shape
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (BS, C, H, W)
        for block in self.decoder.values():
            z_q = block(z_q)
        return z_q
 
    def forward(self, x):
        z_e, shape = self.encode(x)
        z_q, vq_loss = self.quantize(z_e)
        reconstruction = self.decode(z_q, shape)
        return reconstruction, vq_loss
 
 
def loss_vq_vae(x, reconstruction, vq_loss):
    mse_loss = nn.MSELoss(reduction='sum')(x, reconstruction)
    return mse_loss + vq_loss