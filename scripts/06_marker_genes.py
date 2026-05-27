# ============================================
# Script 6: Marker Gene Analysis
# ============================================
# What are marker genes?
# Every cell type has special genes that
# are highly expressed ONLY in that type.
#
# Like how:
# - Doctors wear stethoscopes (marker)
# - Chefs wear aprons (marker)
# - Firefighters wear helmets (marker)
#
# Finding marker genes helps us understand
# WHAT makes each cell type unique and
# is crucial for drug target discovery!
# ============================================

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sc.settings.verbosity = 1

print("=" * 50)
print("Step 6: Marker Gene Analysis")
print("=" * 50)

# Load clustered data
adata = sc.read_h5ad("data/single_cell_clustered.h5ad")
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# ----------------------------------------
# Find marker genes for each cluster
# Compares each cluster against all others
# and finds genes that are uniquely high
# in that cluster
# ----------------------------------------
print("\nFinding marker genes...")
sc.tl.rank_genes_groups(
    adata,
    groupby='leiden',
    method='wilcoxon',
    key_added='rank_genes'
)

# ----------------------------------------
# Extract top 10 marker genes per cluster
# ----------------------------------------
top_genes = {}
for cluster in ['0', '1', '2']:
    genes = adata.uns['rank_genes']['names'][cluster][:10]
    scores = adata.uns['rank_genes']['scores'][cluster][:10]
    top_genes[f'Cluster_{cluster}'] = list(genes)
    print(f"\nTop 10 marker genes for Cluster {cluster}:")
    for gene, score in zip(genes, scores):
        print(f"  {gene}: score={score:.3f}")

# ----------------------------------------
# Plot 1: Dot plot of marker genes
# Shows expression level and % of cells
# expressing each marker gene
# ----------------------------------------
all_markers = []
for cluster_genes in top_genes.values():
    all_markers.extend(cluster_genes[:5])

fig, ax = plt.subplots(figsize=(16, 8))

# Create expression matrix for markers
marker_data = []
cell_types = ['0', '1', '2']
type_names = ['Cancer', 'Immune', 'Stromal']

for gene in all_markers:
    if gene in adata.var_names:
        gene_idx = list(adata.var_names).index(gene)
        row = []
        for cluster in cell_types:
            mask = adata.obs['leiden'] == cluster
            expr = adata.X[mask, gene_idx]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            row.append(float(np.mean(expr)))
        marker_data.append(row)

marker_df = pd.DataFrame(
    marker_data,
    index=all_markers,
    columns=type_names
)

im = ax.imshow(marker_df.values, cmap='RdYlBu_r', aspect='auto')
ax.set_xticks(range(len(type_names)))
ax.set_xticklabels(type_names, fontsize=12)
ax.set_yticks(range(len(all_markers)))
ax.set_yticklabels(all_markers, fontsize=8)
ax.set_title('Marker Gene Expression Heatmap\n(Top 5 per cluster)', fontsize=14)
plt.colorbar(im, ax=ax, label='Mean Expression')
plt.tight_layout()
plt.savefig("figures/06_marker_genes.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nMarker gene plot saved!")

# ----------------------------------------
# Plot 2: Top marker gene violin plots
# Shows distribution of expression
# across all cell types for top markers
# ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (cluster, name) in enumerate(zip(['0', '1', '2'], type_names)):
    top_gene = top_genes[f'Cluster_{cluster}'][0]
    if top_gene in adata.var_names:
        gene_idx = list(adata.var_names).index(top_gene)
        data_by_type = []
        labels = []
        for c, n in zip(['0', '1', '2'], type_names):
            mask = adata.obs['leiden'] == c
            expr = adata.X[mask, gene_idx]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            data_by_type.append(expr)
            labels.append(n)
        axes[idx].violinplot(data_by_type, showmeans=True)
        axes[idx].set_xticks(range(1, 4))
        axes[idx].set_xticklabels(labels, rotation=45)
        axes[idx].set_title(f'Top marker: {top_gene}\n({name} cells)')
        axes[idx].set_ylabel('Expression')

plt.tight_layout()
plt.savefig("figures/07_violin_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("Violin plots saved!")

# Save results
pd.DataFrame(top_genes).to_csv("results/marker_genes.csv", index=False)
adata.write_h5ad("data/single_cell_markers.h5ad")
print("\nMarker gene analysis complete!")
print("Done!")
