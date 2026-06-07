"""
A compact decoder-only transformer (GPT) used as an autoregressive prior over
VQ-VAE codebook indices.
 
Training: given a sequence [start, c0, c1, ..., c_{N-1}], the model predicts each
next code from the preceding ones (standard next-token cross-entropy).
Sampling: starting from just the start token, sample codes one at a time; the
resulting index map is looked up in the codebook and decoded into an image.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (a position attends only to itself
    and earlier positions)."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
 
        self.qkv = nn.Linear(n_embd, 3*n_embd)  # query, key, value in one projection
        self.proj = nn.Linear(n_embd, n_embd)  # output projection
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
 
        # Lower-triangular mask, registered as a buffer so it moves with .to(device).
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer('mask', mask)
 
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
 
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(dim=-1)
        att = self.attn_drop(att)
 
        y = att @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B, T, C)
        return self.resid_drop(self.proj(y))
 
 
class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention + MLP, each with a residual."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
 
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
 
 
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=8, n_embd=384, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # token -> vector
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))  # learned positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # vector -> logits over codes
        self.apply(self._init_weights)
        # pos_emb is a bare Parameter, so _init_weights (Linear/Embedding only) skips it;
        # give it a small-normal start instead of all-zeros.
        nn.init.normal_(self.pos_emb, std=0.02)
 
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
 
    def forward(self, idx):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size)."""
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        x = self.drop(self.tok_emb(idx) + self.pos_emb[:, :T, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
 
    @staticmethod
    def _top_p_filter(logits, top_p):
        """Nucleus filtering: keep the smallest set of tokens whose cumulative prob >= top_p,
        set the rest to -inf. Operates per row (batch)."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        # shift right so the first token crossing the threshold is kept (always keep >=1 token)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        # map the removal flags back to the original (unsorted) token order
        remove_orig = torch.zeros_like(remove)
        remove_orig.scatter_(1, sorted_idx, remove)
        return logits.masked_fill(remove_orig, float('-inf'))
 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, forbidden=None):
        """Autoregressively sample `max_new_tokens` codes given a prefix idx (B, T).
 
        temperature: <1 sharpens (safer/smoother), >1 flattens (more diverse).
        top_k:  keep only the k most likely codes at each step (e.g. 100).
        top_p:  nucleus sampling, keep the top codes summing to prob top_p (e.g. 0.9).
        forbidden: list of token ids the model is not allowed to emit (e.g. the start
                   token, which is a valid vocab entry but not a real codebook index).
        Use one of top_k / top_p (or neither for full sampling)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # never exceed block_size
            logits = self(idx_cond)[:, -1, :] / temperature  # logits for the next token
            if forbidden is not None:
                logits[:, forbidden] = float('-inf')  # block e.g. the start token
            if top_k is not None:
                k = min(top_k, logits.size(-1))  # don't exceed vocab size
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p is not None:
                logits = self._top_p_filter(logits, top_p)
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
        