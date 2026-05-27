# ============================================
# Script 3: Normalisation and Feature Selection
# ============================================
# Why do we normalise?
# Imagine two students took the same exam:
# Student A answered 1000 questions
# Student B answered 500 questions
# Student A will naturally have higher scores
# just because they answered more questions!
#
# Normalisation makes it FAIR by adjusting
# all cells to have the same total counts.
#
# Feature selection = picking the most
# interesting genes (like picking the most
# useful questions for comparison)
# ============================================

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

sc.settings.verbosity = 1

print("=" * 50)
print("Step 3: Normalisation and Feature Selection")
print("=" * 50)

# Load filtered data
adata = sc.read_h5ad("data/single_cell_filtered.h5ad")
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# ----------------------------------------
# Step 1: Normalise total counts
# Make every cell have exactly 10,000 counts
# Like grading all students out of 100
# regardless of how many questions they did
# ----------------------------------------
print("\nNormalising counts...")
sc.pp.normalize_total(adata, target_sum=1e4)
print("Total counts normalised to 10,000 per cell")

# ----------------------------------------
# Step 2: Log transform
# Gene expression values can be huge (0-10000)
# Log transform squishes them into a smaller
# range that is easier to work with
# Like converting a ruler from meters to
# a more manageable scale
# ----------------------------------------
sc.pp.log1p(adata)
print("Log transformation applied")

# ----------------------------------------
# Step 3: Find highly variable genes
# Not all 2000 genes are interesting!
# Some genes are the same in every cell
# (boring!) We want genes that differ
# between cell types (interesting!)
# Like finding the questions where students
# gave very different answers
# ----------------------------------------
print("\nFinding highly variable genes...")
sc.pp.highly_variable_genes(
    adata,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5
)

n_hvg = adata.var['highly_variable'].sum()
print(f"Found {n_hvg} highly variable genes out of {adata.n_vars}")

# ----------------------------------------
# Plot highly variable genes
# Shows which genes vary the most
# across cells
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Mean vs Dispersion
axes[0].scatter(
    adata.var['means'],
    adata.var['dispersions'],
    c=adata.var['highly_variable'].map({True: 'red', False: 'gray'}),
    alpha=0.5, s=10
)
axes[0].set_xlabel('Mean Expression')
axes[0].set_ylabel('Dispersion')
axes[0].set_title('Highly Variable Genes\n(red = selected)')

# Plot 2: HVG counts
hvg_counts = adata.var['highly_variable'].value_counts()
axes[1].bar(
    ['Variable Genes\n(interesting)', 'Stable Genes\n(boring)'],
    [hvg_counts.get(True, 0), hvg_counts.get(False, 0)],
    color=['red', 'gray'],
    edgecolor='black'
)
axes[1].set_ylabel('Number of Genes')
axes[1].set_title('Gene Variability Distribution')

plt.tight_layout()
plt.savefig("figures/02_normalisation.png", dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to figures/02_normalisation.png")

# Save normalised data
adata.write_h5ad("data/single_cell_normalised.h5ad")
print("\nNormalised data saved!")
print("Done!")
