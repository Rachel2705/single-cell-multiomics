# ============================================
# Script 2: Quality Control
# ============================================
# Why do we need QC?
# Imagine you took photos of 500 cells but
# some photos are blurry or broken.
# QC helps us throw away the bad photos
# and keep only the good ones.
#
# We check 3 things:
# 1. Total genes per cell (too few = dead cell)
# 2. Total counts per cell (too low = empty)
# 3. Mitochondrial genes % (too high = dying)
# ============================================

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sc.settings.verbosity = 1

print("=" * 50)
print("Step 2: Quality Control")
print("=" * 50)

# ----------------------------------------
# Load our raw data
# ----------------------------------------
adata = sc.read_h5ad("data/single_cell_raw.h5ad")
print(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")

# ----------------------------------------
# Calculate QC metrics
# Like checking each student's answer sheet:
# - How many questions did they answer? (n_genes)
# - What is their total score? (total_counts)
# ----------------------------------------
sc.pp.calculate_qc_metrics(adata, inplace=True)

print("\nQC metrics calculated!")
print(f"Average genes per cell: {adata.obs['n_genes_by_counts'].mean():.0f}")
print(f"Average counts per cell: {adata.obs['total_counts'].mean():.0f}")

# ----------------------------------------
# Plot QC metrics
# Let's visualise what our data looks like
# before filtering
# ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Genes per cell
axes[0].hist(adata.obs['n_genes_by_counts'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Number of Genes per Cell')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Genes per Cell Distribution')
axes[0].axvline(x=200, color='red', linestyle='--', label='Min threshold')
axes[0].legend()

# Plot 2: Total counts per cell
axes[1].hist(adata.obs['total_counts'], bins=50, color='green', edgecolor='black')
axes[1].set_xlabel('Total Counts per Cell')
axes[1].set_ylabel('Number of Cells')
axes[1].set_title('Total Counts Distribution')
axes[1].axvline(x=500, color='red', linestyle='--', label='Min threshold')
axes[1].legend()

# Plot 3: Cell type distribution
cell_counts = adata.obs['cell_type'].value_counts()
axes[2].bar(cell_counts.index, cell_counts.values,
            color=['steelblue', 'coral', 'green'], edgecolor='black')
axes[2].set_xlabel('Cell Type')
axes[2].set_ylabel('Number of Cells')
axes[2].set_title('Cell Type Distribution')

plt.tight_layout()
plt.savefig("figures/01_qc_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nQC plot saved to figures/01_qc_metrics.png")

# ----------------------------------------
# Filter cells
# Remove bad cells like a teacher removing
# incomplete answer sheets
# Keep cells with:
# - More than 200 genes expressed
# - More than 500 total counts
# ----------------------------------------
print("\nFiltering cells...")
print(f"Before filtering: {adata.n_obs} cells")

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_genes(adata, min_cells=10)

print(f"After filtering : {adata.n_obs} cells")
print(f"After filtering : {adata.n_vars} genes")

# Save filtered data
adata.write_h5ad("data/single_cell_filtered.h5ad")
print("\nFiltered data saved to data/single_cell_filtered.h5ad")
print("Done!")
