# ============================================
# Script 5: Leiden Clustering
# ============================================
# What is clustering?
# Imagine you throw 500 coloured balls
# into a room. Similar coloured balls
# naturally group together.
#
# Leiden algorithm does the same thing
# with cells — it finds groups of cells
# that are similar to each other WITHOUT
# being told what the groups are.
#
# This is UNSUPERVISED learning — the
# computer finds the patterns by itself!
# ============================================

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sc.settings.verbosity = 1

print("=" * 50)
print("Step 5: Leiden Clustering")
print("=" * 50)

# Load reduced data
adata = sc.read_h5ad("data/single_cell_reduced.h5ad")
print(f"Loaded: {adata.n_obs} cells")

# ----------------------------------------
# Run Leiden clustering
# resolution controls how many clusters:
# - Low resolution = fewer big clusters
# - High resolution = many small clusters
# Like zooming in or out on a map
# ----------------------------------------
print("\nRunning Leiden clustering...")
sc.tl.leiden(adata, resolution=0.5, random_state=42)

n_clusters = adata.obs['leiden'].nunique()
print(f"Found {n_clusters} clusters!")
print("\nCluster sizes:")
print(adata.obs['leiden'].value_counts().sort_index())

# ----------------------------------------
# Compare clusters with known cell types
# Let's see if Leiden found the same
# groups as our true cell types
# ----------------------------------------
comparison = pd.crosstab(
    adata.obs['leiden'],
    adata.obs['cell_type']
)
print("\nCluster vs Cell Type comparison:")
print(comparison)

# ----------------------------------------
# Plot 1: UMAP coloured by Leiden clusters
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Get cluster colours
n_clusters = adata.obs['leiden'].nunique()
colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
cluster_colors = {str(i): colors[i] for i in range(n_clusters)}

for cluster in adata.obs['leiden'].unique():
    mask = adata.obs['leiden'] == cluster
    axes[0].scatter(
        adata.obsm['X_umap'][mask, 0],
        adata.obsm['X_umap'][mask, 1],
        c=[cluster_colors[cluster]],
        label=f'Cluster {cluster}',
        alpha=0.7, s=30
    )
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].set_title('UMAP - Leiden Clusters')
axes[0].legend()

# Plot 2: Cell type vs cluster heatmap
im = axes[1].imshow(comparison.values, cmap='Blues', aspect='auto')
axes[1].set_xticks(range(len(comparison.columns)))
axes[1].set_xticklabels(comparison.columns, rotation=45)
axes[1].set_yticks(range(len(comparison.index)))
axes[1].set_yticklabels([f'Cluster {i}' for i in comparison.index])
axes[1].set_title('Cluster vs Cell Type Heatmap')
plt.colorbar(im, ax=axes[1])

# Add numbers to heatmap
for i in range(len(comparison.index)):
    for j in range(len(comparison.columns)):
        axes[1].text(j, i, comparison.values[i, j],
                    ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig("figures/05_leiden_clustering.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nClustering plot saved to figures/05_leiden_clustering.png")

# Save
adata.write_h5ad("data/single_cell_clustered.h5ad")
print("Done!")
