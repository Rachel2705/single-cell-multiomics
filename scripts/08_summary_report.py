# ============================================
# Script 8: Summary Report
# ============================================
# This script creates a final summary
# of everything we did in this project
# and generates a final comparison plot
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scanpy as sc

print("=" * 50)
print("Step 8: Summary Report")
print("=" * 50)

# Load final data
adata = sc.read_h5ad("data/single_cell_markers.h5ad")
vae_latent = pd.read_csv("results/vae_latent.csv", index_col=0)
markers = pd.read_csv("results/marker_genes.csv")
clustering = pd.read_csv("results/clustering_results.csv") \
    if __import__('os').path.exists("results/clustering_results.csv") else None

print("\n=== PROJECT SUMMARY ===")
print(f"Total cells analysed    : {adata.n_obs}")
print(f"Total genes analysed    : {adata.n_vars}")
print(f"Cell types identified   : {adata.obs['cell_type'].nunique()}")
print(f"Clusters found          : {adata.obs['leiden'].nunique()}")
print(f"VAE latent dimensions   : {vae_latent.shape[1]-2}")

print("\n=== CELL TYPE DISTRIBUTION ===")
print(adata.obs['cell_type'].value_counts())

print("\n=== CLUSTER DISTRIBUTION ===")
print(adata.obs['leiden'].value_counts().sort_index())

# ----------------------------------------
# Create final summary figure
# A beautiful multi-panel plot showing
# the entire analysis pipeline
# ----------------------------------------
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Panel 1: Cell type distribution pie chart
ax1 = fig.add_subplot(gs[0, 0])
cell_counts = adata.obs['cell_type'].value_counts()
colors = ['steelblue', 'coral', 'green']
ax1.pie(
    cell_counts.values,
    labels=cell_counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90
)
ax1.set_title('Cell Type Distribution', fontsize=14, fontweight='bold')

# Panel 2: UMAP coloured by cell type
ax2 = fig.add_subplot(gs[0, 1])
colors_map = {'Immune': 'steelblue', 'Cancer': 'coral', 'Stromal': 'green'}
for cell_type, color in colors_map.items():
    mask = adata.obs['cell_type'] == cell_type
    ax2.scatter(
        adata.obsm['X_umap'][mask, 0],
        adata.obsm['X_umap'][mask, 1],
        c=color, label=cell_type, alpha=0.7, s=20
    )
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_title('UMAP - Cell Types', fontsize=14, fontweight='bold')
ax2.legend()

# Panel 3: VAE latent space
ax3 = fig.add_subplot(gs[0, 2])
for cell_type, color in colors_map.items():
    mask = vae_latent['cell_type'] == cell_type
    ax3.scatter(
        vae_latent[mask]['VAE_0'],
        vae_latent[mask]['VAE_1'],
        c=color, label=cell_type, alpha=0.7, s=20
    )
ax3.set_xlabel('VAE Dimension 1')
ax3.set_ylabel('VAE Dimension 2')
ax3.set_title('VAE Latent Space', fontsize=14, fontweight='bold')
ax3.legend()

# Panel 4: Cluster sizes bar chart
ax4 = fig.add_subplot(gs[1, 0])
cluster_counts = adata.obs['leiden'].value_counts().sort_index()
bars = ax4.bar(
    [f'Cluster {i}' for i in cluster_counts.index],
    cluster_counts.values,
    color=colors, edgecolor='black'
)
ax4.set_ylabel('Number of Cells')
ax4.set_title('Leiden Cluster Sizes', fontsize=14, fontweight='bold')
for bar, val in zip(bars, cluster_counts.values):
    ax4.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 2,
        str(val), ha='center', fontweight='bold'
    )

# Panel 5: Top marker genes heatmap
ax5 = fig.add_subplot(gs[1, 1])
top_3_markers = []
type_labels = []
for col in markers.columns:
    top_3_markers.extend(markers[col][:3].tolist())
    type_labels.extend([col]*3)

expr_data = []
cell_type_order = ['Cancer', 'Immune', 'Stromal']
for gene in top_3_markers:
    if gene in adata.var_names:
        gene_idx = list(adata.var_names).index(gene)
        row = []
        for ct in cell_type_order:
            mask = adata.obs['cell_type'] == ct
            expr = adata.X[mask, gene_idx]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            row.append(float(np.mean(expr)))
        expr_data.append(row)

if expr_data:
    im = ax5.imshow(expr_data, cmap='RdYlBu_r', aspect='auto')
    ax5.set_xticks(range(3))
    ax5.set_xticklabels(cell_type_order)
    ax5.set_yticks(range(len(top_3_markers)))
    ax5.set_yticklabels(top_3_markers, fontsize=7)
    ax5.set_title('Top Marker Genes', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax5)

# Panel 6: Pipeline summary text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
pipeline_text = """
ANALYSIS PIPELINE SUMMARY

1. Data Generation
   500 cells × 2000 genes
   3 cell types simulated

2. Quality Control
   All 500 cells passed QC
   2000 genes retained

3. Normalisation
   10,000 counts/cell
   588 HVGs selected

4. Dimensionality Reduction
   PCA: 50 components
   UMAP: 2D visualisation

5. Leiden Clustering
   3 perfect clusters found
   100% accuracy vs cell types

6. Marker Gene Analysis
   Wilcoxon rank-sum test
   Top 10 markers per cluster

7. Variational Autoencoder
   588 → 10 latent dimensions
   150 epochs trained
"""
ax6.text(
    0.05, 0.95, pipeline_text,
    transform=ax6.transAxes,
    fontsize=10, verticalalignment='top',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
)
ax6.set_title('Pipeline Summary', fontsize=14, fontweight='bold')

plt.suptitle(
    'Single Cell Multi-omics Analysis\nComplete Pipeline Summary',
    fontsize=16, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig(
    "figures/09_final_summary.png",
    dpi=300, bbox_inches='tight'
)
plt.close()

print("\nFinal summary plot saved to figures/09_final_summary.png")
print("\nProject complete!")
