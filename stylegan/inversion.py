"""
GAN inversion: find a latent that makes a *pretrained* StyleGAN2 reproduce a given photo,
then (optionally) tune the generator itself so it reproduces it exactly.

Two stages:
  1. `invert_w` - optimize a single W latent (starting from w_avg) so the generated image
     matches the target. Loss = LPIPS + MSE. A real face rarely lies exactly in W, so the
     result is recognizable but imperfect - that's expected.
  2. `pivotal_tuning` - PTI (Roich et al., 2021). Keep the latent from step 1 fixed as the
     "pivot" and fine-tune the generator's synthesis network so that this pivot decodes to
     the target photo. To stop the generator from drifting everywhere else, the same step
     regularizes it on other latents sampled around the pivot: on those, the tuned generator
     must keep producing what the original generator produced.

The target image is expected as a (1, 3, H, W) tensor in [-1, 1] (StyleGAN2's range),
already FFHQ-aligned (see stylegan_utils.align_face).
"""
import random

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# optimize W
def invert_w(G, target, lpips_fn, n_steps=1500, lr=0.05, w_mse=1.0, w_lpips=1.0, log_every=100):
    """Optimize one W latent so G reproduces `target`. Returns (w_pivot (1,1,512), history)."""
    w_pivot = G.mapping.w_avg.reshape(1, 1, -1).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w_pivot], lr=lr)

    history = []
    pbar = tqdm(range(n_steps))
    for step in pbar:
        wplus = w_pivot.repeat(1, G.num_ws, 1)  # broadcast W -> W+
        img = G.synthesis(wplus, noise_mode='const')

        loss_lpips = lpips_fn(img, target).mean()
        loss_mse = F.mse_loss(img, target)
        loss = w_lpips * loss_lpips + w_mse * loss_mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if step % log_every == 0:
            pbar.set_postfix({'lpips': round(loss_lpips.item(), 4), 'mse': round(loss_mse.item(), 4)})
    pbar.close()
    return w_pivot.detach(), history


# PTI
@torch.no_grad()
def sample_regularization_latents(G, wplus_pivot, n=300, alpha=120.0, seed=42, device='cuda'):
    """Latents for the PTI regularization term: random W+ vectors pulled to a fixed distance
    `alpha` from the pivot (eq. 4 in the PTI paper). These are the neighbourhood on which the
    tuned generator must keep behaving like the original one.

    NB: a fresh z is drawn every iteration - reusing one z would make every 'neighbour' the
    same face and the regularizer meaningless."""
    random.seed(seed)
    gen = torch.Generator(device).manual_seed(seed)
    out = []
    for _ in range(n):
        z = torch.randn([1, G.z_dim], generator=gen, device=device)
        wplus = G.mapping(z, None, truncation_psi=random.random(), update_emas=False)
        direction = wplus - wplus_pivot
        wplus = wplus_pivot + alpha * (direction / direction.norm())
        out.append(wplus.cpu())
    return torch.cat(out, dim=0)


def pivotal_tuning(G_pti, G_orig, wplus_pivot, target, reg_latents, lpips_fn,
                   n_steps=350, batch_size=4, lr=2e-3, w_mse=1.0, w_lpips=1.0, log_every=25):
    """Fine-tune G_pti.synthesis so wplus_pivot -> target, while its output on `reg_latents`
    stays equal to the original generator's. G_orig is frozen and used as the teacher."""
    for name, p in G_pti.synthesis.named_parameters():
        p.requires_grad_('noise_const' not in name)  # don't train the constant noise
    optimizer = torch.optim.Adam(G_pti.synthesis.parameters(), lr=lr)

    device = wplus_pivot.device
    n_reg = batch_size - 1  # rest of the batch = regularization
    history = []
    pbar = tqdm(range(n_steps))
    for step in pbar:
        idx = torch.randint(0, reg_latents.shape[0], (n_reg,))
        wplus_reg = reg_latents[idx].to(device)

        with torch.no_grad():  # teacher targets for the neighbours
            img_reg_target = G_orig.synthesis(wplus_reg, noise_mode='const')

        # one batch: the pivot (target = the photo) + neighbours (target = original generator)
        wplus_batch = torch.cat([wplus_pivot, wplus_reg], dim=0)
        img_target = torch.cat([target, img_reg_target], dim=0)
        img_pred = G_pti.synthesis(wplus_batch, noise_mode='const')

        loss_lpips = lpips_fn(img_pred, img_target).mean()
        loss_mse = F.mse_loss(img_pred, img_target)
        loss = w_lpips * loss_lpips + w_mse * loss_mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if step % log_every == 0:
            pbar.set_postfix({'lpips': round(loss_lpips.item(), 4), 'mse': round(loss_mse.item(), 4)})
    pbar.close()
    return G_pti.eval(), history
