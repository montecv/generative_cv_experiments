from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class TinnyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        tmp_ch = 6
        
        self.encoder = nn.ModuleDict()
        self.encoder['in_conv'] = nn.Conv2d(3, tmp_ch, 1)
        for i in range(2):
            self.encoder[f'block_{i}'] = nn.Sequential(
                nn.Conv2d(tmp_ch, tmp_ch*2, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(tmp_ch*2)
            )
            tmp_ch = tmp_ch*2
        self.encoder['out_conv'] = nn.Conv2d(tmp_ch, tmp_ch*2, 1)
        
        self.decoder = nn.ModuleDict()
        self.decoder['in_conv'] = nn.Conv2d(tmp_ch, tmp_ch, 1)
        for i in range(2):
            self.decoder[f'block_{i}'] = nn.Sequential(
                nn.Conv2d(tmp_ch, tmp_ch//2, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(tmp_ch//2),
                nn.Upsample(scale_factor=2)
            )
            tmp_ch = tmp_ch//2
        self.decoder['out_conv'] = nn.Conv2d(tmp_ch, 3, 1)

    def forward(self, x):
        for block in self.encoder.values():
            x = block(x) 
    
        mu, logvar = x.split(x.shape[1]//2, dim=1)  # splits the tensor in half along the channel dim: first half is mu, second is logvar
    
        # Reparametrization trick
        std = (0.5*logvar).exp()  # ensures positivity
        epsilon = torch.randn_like(mu)
        z = (std * epsilon + mu).contiguous()
    
        for block in self.decoder.values():
            z = block(z) 
        reconstruction = z
        
        return reconstruction, mu, std


class TinnyLinearVAE(nn.Module):
    def __init__(self):
        super().__init__()
        latent_space_dim = 2

        self.encoder = nn.Sequential(
            nn.Linear(in_features=64*64*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=latent_space_dim*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_space_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64*64*3)
        )
    
    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        
        x = self.encoder(x)

        mu, logvar = x.split(x.shape[1]//2, dim=1)
        std = (0.5*logvar).exp()
        epsilon = torch.randn_like(mu)
        z = mu + std * epsilon

        reconstruction = self.decoder(z)
        reconstruction = reconstruction.reshape((-1, 3, 64, 64))

        return reconstruction, mu[..., None, None], std[..., None, None]


def reversed_KL_divergence(mu, std):
    '''KL(N(mu, std^2), N(0, 1))'''
    loss = (- std.log() + 0.5 * (std.pow(2) + mu.pow(2)) - 0.5)
    return loss

def loss_vae(x, mu, std, reconstruction):
    kl_loss = reversed_KL_divergence(mu, std).sum(dim=(1,2,3)).mean()
    mse_loss = nn.MSELoss(reduction='sum')(x, reconstruction)
    return mse_loss + kl_loss

def train(model, n_epochs, train_loader, val_loader, device, optimizer, ):
    train_losses, val_losses = [], []
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_losses_epoch = []
        for batch in train_loader:
            batch = batch.permute(0, 3, 1, 2).to(device).to(torch.float32)/255
        
            optimizer.zero_grad()
            reconstruction, mu, std = model(batch)
            loss = loss_vae(batch, mu, std, reconstruction)
            loss.backward()
            optimizer.step()
            train_losses_epoch.append(loss.item())
            
        train_loss_epoch = np.mean(train_losses_epoch)
        train_losses.append(train_loss_epoch)
        

        model.eval()
        val_losses_epoch = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.permute(0, 3, 1, 2).to(device).to(torch.float32)/255
            
                reconstruction, mu, std = model(batch)
                loss = loss_vae(batch, mu, std, reconstruction)
                val_losses_epoch.append(loss.item())
                
        val_loss_epoch = np.mean(val_losses_epoch)
        val_losses.append(val_loss_epoch)

        # tqdm.write(f'Epoch {epoch+1:>3}/{n_epochs} | train loss: {train_loss_epoch:.2f} | val loss: {val_loss_epoch:.2f}')
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return train_losses, val_losses


def reconstruct(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = sample_batch.permute(0, 3, 1, 2).to(device).to(torch.float32) / 255
        reconstruction, mu, std = model(sample_batch)

    n_show = 10
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
