# ============================================
# Script 7: Variational Autoencoder (VAE)
# ============================================
# What is a VAE?
# Remember our basic Autoencoder?
# It compressed data into a fixed point.
#
# A VAE is smarter — instead of compressing
# to a fixed point it compresses to a
# RANGE (distribution).
#
# Think of it like this:
# Basic Autoencoder: "This cell is exactly
# at point X=3, Y=5 on the map"
#
# VAE: "This cell is somewhere around
# X=3±0.5, Y=5±0.3 on the map"
#
# This makes the VAE:
# - Better at learning biology
# - Able to generate new synthetic cells
# - More robust to noise in data
#
# VAEs are used in drug discovery to
# generate new molecular structures!
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("Step 7: Variational Autoencoder (VAE)")
print("=" * 50)

# Load data
adata = sc.read_h5ad("data/single_cell_markers.h5ad")
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Prepare data
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)

# Labels for colouring plots
cell_types = adata.obs['cell_type'].values
labels = adata.obs['leiden'].astype(int).values

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X_tensor.shape[1]
latent_dim = 10

# ----------------------------------------
# Define VAE
# Two key differences from basic autoencoder:
# 1. Encoder outputs mean AND variance
# 2. Reparameterization trick for sampling
# ----------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Mean and variance layers
        # Like saying "the cell is around HERE"
        # instead of "exactly HERE"
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_var(h)

    def reparameterize(self, mean, log_var):
        # This is the magic trick!
        # Instead of using mean directly
        # we sample from the distribution
        # Like saying "pick a random point
        # near the mean"
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var

# ----------------------------------------
# VAE Loss Function
# Two parts:
# 1. Reconstruction loss - how well does
#    the VAE reconstruct the input?
# 2. KL divergence - how close is the
#    latent space to a normal distribution?
# ----------------------------------------
def vae_loss(reconstructed, original, mean, log_var):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mean.pow(2) - log_var.exp()
    )
    kl_loss = kl_loss / original.shape[0]
    return reconstruction_loss + 0.001 * kl_loss

# Initialize and train
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel architecture:")
print(f"  Input dim  : {input_dim}")
print(f"  Latent dim : {latent_dim}")
print(f"\nTraining VAE...")

losses = []
recon_losses = []
epochs = 150

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        reconstructed, mean, log_var = model(x)
        loss = vae_loss(reconstructed, x, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)

    if (epoch+1) % 15 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# ----------------------------------------
# Extract latent representations
# ----------------------------------------
model.eval()
with torch.no_grad():
    mean, log_var = model.encode(X_tensor)
    z = model.reparameterize(mean, log_var)

latent = z.numpy()

# ----------------------------------------
# Plot 1: Training loss curve
# ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('VAE Training Loss')
axes[0].grid(True, alpha=0.3)

# ----------------------------------------
# Plot 2: Latent space coloured by cell type
# ----------------------------------------
colors_map = {'Immune': 'steelblue', 'Cancer': 'coral', 'Stromal': 'green'}
for cell_type, color in colors_map.items():
    mask = cell_types == cell_type
    axes[1].scatter(
        latent[mask, 0],
        latent[mask, 1],
        c=color, label=cell_type,
        alpha=0.7, s=30
    )
axes[1].set_xlabel('Latent Dimension 1')
axes[1].set_ylabel('Latent Dimension 2')
axes[1].set_title('VAE Latent Space\n(coloured by cell type)')
axes[1].legend()

# ----------------------------------------
# Plot 3: Latent space coloured by cluster
# ----------------------------------------
cluster_colors = ['red', 'blue', 'green']
for cluster in [0, 1, 2]:
    mask = labels == cluster
    axes[2].scatter(
        latent[mask, 0],
        latent[mask, 1],
        c=cluster_colors[cluster],
        label=f'Cluster {cluster}',
        alpha=0.7, s=30
    )
axes[2].set_xlabel('Latent Dimension 1')
axes[2].set_ylabel('Latent Dimension 2')
axes[2].set_title('VAE Latent Space\n(coloured by cluster)')
axes[2].legend()

plt.tight_layout()
plt.savefig("figures/08_vae_latent_space.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nVAE latent space plot saved!")

# Save latent representations
latent_df = pd.DataFrame(
    latent,
    index=adata.obs_names,
    columns=[f"VAE_{i}" for i in range(latent_dim)]
)
latent_df['cell_type'] = cell_types
latent_df['cluster'] = labels
latent_df.to_csv("results/vae_latent.csv")

print("Latent representations saved!")
print("\nVAE training complete!")
print("Done!")
