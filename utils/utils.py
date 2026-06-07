import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tqdm import tqdm
import torch

from PIL import Image


def plot_train_val_losses(train_losses, val_losses, loss_name='Loss'):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel(loss_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

def reconstruct(model, val_loader, device, n_show=10, **model_kwargs):
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader)).permute(0, 3, 1, 2).to(device).to(torch.float32) / 255
        out = model(sample_batch, **model_kwargs)
        reconstruction = out[0] if isinstance(out, tuple) else out

    fig, axes = plt.subplots(2, n_show, figsize=(16, 4))
    for i in range(n_show):
        axes[0, i].imshow(sample_batch[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstruction[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
        axes[1, i].axis('off')
    axes[0, 0].set_title('Original', fontsize=9)
    axes[1, 0].set_title('Reconstruction', fontsize=9)
    plt.tight_layout()
    plt.show()

def return_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Device is {device}')
    return device


def return_photos(path2photos):
    paths = []
    for dirpath, _, filenames in os.walk(path2photos):
        for fname in filenames:
            if fname.endswith('.jpg'):
                paths.append(os.path.join(dirpath, fname))

    all_photos = []
    for path in tqdm(paths):
        with open(path, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB'))[80:-80, 80:-80]
            image = np.array(Image.fromarray(image).resize((64, 64), Image.Resampling.LANCZOS))
            all_photos.append(image)

    print(f'Photos uploaded: {len(all_photos)}')
    return all_photos
