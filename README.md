# generative_cv_experiments

A personal collection of from-scratch implementations and experiments with
**generative models for computer vision**. Each subproject is self-contained:
a small PyTorch module plus a notebook that trains it and visualizes the results.
The code is heavily reworked and annotated for clarity rather than copied as-is.

This repo grows over time as I explore more architectures and ideas.

## Contents

### VAE — `vae.py`, `vae_experiments.ipynb`
A Variational Autoencoder. Includes a convolutional variant and a minimal linear
variant, the reparametrization trick, a KL term against a standard normal prior,
and notebooks that show reconstructions and the structure of the latent space
(sampling from `N(0, I)` vs. the encoder's posterior).

### VQ-VAE — `vq_vae.py`, `vq_vae_experiments.ipynb`
A Vector-Quantized VAE with a discrete codebook. Two codebook update rules behind
one shared interface, switchable with a single `use_ema` flag:
- **Gradient codebook** — the original two-term VQ loss (codebook + commitment).
- **EMA codebook** — codebook updated by an exponential moving average of the
  encoder outputs; more robust against codebook collapse.

The notebook also includes codebook-from-data initialization, straight-through
gradient estimation, reconstructions, and a codebook-usage diagnostic that shows
how many codes are actually active (a direct way to spot collapse).

### dVAE — `dvae.py`, `dvae_experiments.ipynb`
A discrete VAE that replaces hard nearest-neighbour quantization with the
**Gumbel-Softmax** relaxation. The encoder emits logits over the codebook entries;
adding Gumbel noise and taking a temperature-scaled softmax yields a differentiable
soft selection over codes, so gradients reach the encoder directly — no
straight-through trick needed. The codebook is learned jointly.

Temperature is annealed from high (soft, near-uniform mixing) to low (peaky,
near one-hot) over training, with evaluation done at low temperature for
near-discrete codes. The notebook opens with a Gumbel-Softmax visualization
showing how the sampled distribution sharpens as temperature drops, followed by
training and reconstructions.

## Dataset

Experiments are trained on **LFW (Labeled Faces in the Wild)**, center-cropped to
the face and resized to 64×64. The dataset is **not** included in this repo —
download it separately and point `PATH_TO_PHOTOS` in the notebooks at your local
copy
