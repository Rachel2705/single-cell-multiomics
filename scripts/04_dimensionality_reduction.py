# ============================================
# Script 4: Dimensionality Reduction
# ============================================
# Why do we need this?
# We have 588 interesting genes per cell.
# That is 588 dimensions — impossible to
# visualise or understand!
#
# We use two steps to simplify:
#
# Step 1: PCA
# Like summarising a 500 page book into
# 50 key points. Keeps the most important
# information, throws away the noise.
#
# Step 2: UMAP
# Like drawing a map of a city.
# Cells that are similar end up close
# together on the map.
# Cells that are different end up far apart.
# ============================================

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

sc.settings.verbosity = 1

print("=" * 50)
print("Step 4: Dimensionality Reduction")
print("=" * 50)

# Load normalised data
adata = sc.read_h5ad("data/single_cell_normalised.h5ad")
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# ----------------------------------------
# Use only highly variable genes
# Like only using the interesting questions
# for our analysis
# ----------------------------------------
adata = adata[:, adata.var['highly_variable']]
print(f"Using {adata.n_vars} highly variable genes")

# ----------------------------------------
# Step 1: Scale data
# Make sure no single gene dominates
# Like making sure no single question
# is worth more than others
# ----------------------------------------
print("\nScaling data...")
sc.pp.scale(adata, max_value=10)

# ----------------------------------------
# Step 2: PCA
# Compress 588 genes into 50 key components
# Like summarising a book into chapters
# ----------------------------------------
print("Running PCA...")
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

# ----------------------------------------
# Plot variance explained by PCA
# Shows how much information each
# component captures
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

variance_ratio = adata.uns['pca']['variance_ratio']
axes[0].plot(
    range(1, 21),
    variance_ratio[:20],
    'bo-', linewidth=2, markersize=6
)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('PCA Variance Explained\n(Elbow Plot)')
axes[0].axvline(x=10, color='red', linestyle='--', label='Selected PCs')
axes[0].legend()

# Cumulative variance
cumsum = np.cumsum(variance_ratio[:20])
axes[1].plot(range(1, 21), cumsum, 'ro-', linewidth=2, markersize=6)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance Explained')
axes[1].axhline(y=0.8, color='blue', linestyle='--', label='80% threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/03_pca_variance.png", dpi=300, bbox_inches='tight')
plt.close()
print("PCA variance plot saved!")

# ----------------------------------------
# Step 3: Build neighbourhood graph
# Like finding which cells are neighbours
# on our map before drawing it
# ----------------------------------------
print("\nBuilding neighbourhood graph...")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# ----------------------------------------
# Step 4: UMAP
# Draw the final 2D map of all cells
# This is the most beautiful plot!
# ----------------------------------------
print("Running UMAP...")
sc.tl.umap(adata)

# ----------------------------------------
# Plot UMAP coloured by cell type
# Each colour = one cell type
# Clusters = groups of similar cells
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# UMAP coloured by cell type
colors = {'Immune': 'steelblue', 'Cancer': 'coral', 'Stromal': 'green'}
for cell_type, color in colors.items():
    mask = adata.obs['cell_type'] == cell_type
    axes[0].scatter(
        adata.obsm['X_umap'][mask, 0],
        adata.obsm['X_umap'][mask, 1],
        c=color, label=cell_type, alpha=0.7, s=30
    )
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].set_title('UMAP - Cell Types')
axes[0].legend()

# PCA plot coloured by cell type
for cell_type, color in colors.items():
    mask = adata.obs['cell_type'] == cell_type
    axes[1].scatter(
        adata.obsm['X_pca'][mask, 0],
        adata.obsm['X_pca'][mask, 1],
        c=color, label=cell_type, alpha=0.7, s=30
    )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('PCA - Cell Types')
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/04_umap_pca.png", dpi=300, bbox_inches='tight')
plt.close()
print("UMAP plot saved to figures/04_umap_pca.png")

# Save
adata.write_h5ad("data/single_cell_reduced.h5ad")
print("\nDimensionality reduction complete!")
print("Done!")
