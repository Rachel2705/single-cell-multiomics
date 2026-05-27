# ============================================
# Script 1: Generate Single Cell Data
# ============================================
# What is single cell data?
# Imagine you have 500 cells from a tumor.
# Each cell has its own gene expression profile.
# This script creates fake but realistic data
# that looks just like real single cell data.
# ============================================

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Tell scanpy not to show too many messages
sc.settings.verbosity = 1

print("=" * 50)
print("Step 1: Creating Single Cell Data")
print("=" * 50)

# ----------------------------------------
# Set random seed
# Think of this like shuffling a deck of
# cards the same way every time so we get
# the same result each run
# ----------------------------------------
np.random.seed(42)

# ----------------------------------------
# Define our experiment
# 500 cells, 2000 genes
# Like having 500 students each answering
# 2000 questions differently
# ----------------------------------------
n_cells = 500
n_genes = 2000

# ----------------------------------------
# Create 3 cell types
# Like 3 types of students in a class:
# Type A: immune cells (150 cells)
# Type B: cancer cells (200 cells)
# Type C: stromal cells (150 cells)
# ----------------------------------------
cell_types = (
    ['Immune'] * 150 +
    ['Cancer'] * 200 +
    ['Stromal'] * 150
)

# ----------------------------------------
# Generate gene expression for each type
# Each cell type expresses genes differently
# Like how doctors, engineers and teachers
# use different words in their daily work
# ----------------------------------------
expression_data = np.zeros((n_cells, n_genes))

# Immune cells - high expression of immune genes
expression_data[:150, :500] = np.random.negative_binomial(
    5, 0.3, size=(150, 500))
expression_data[:150, 500:] = np.random.negative_binomial(
    1, 0.8, size=(150, 1500))

# Cancer cells - high expression of proliferation genes
expression_data[150:350, 500:1000] = np.random.negative_binomial(
    8, 0.2, size=(200, 500))
expression_data[150:350, :500] = np.random.negative_binomial(
    1, 0.9, size=(200, 500))
expression_data[150:350, 1000:] = np.random.negative_binomial(
    2, 0.7, size=(200, 1000))

# Stromal cells - high expression of structural genes
expression_data[350:, 1000:] = np.random.negative_binomial(
    6, 0.25, size=(150, 1000))
expression_data[350:, :1000] = np.random.negative_binomial(
    1, 0.85, size=(150, 1000))

# ----------------------------------------
# Create cell and gene names
# Like giving each student and question a name
# ----------------------------------------
cell_names = [f"Cell_{i}" for i in range(n_cells)]
gene_names = [f"Gene_{i}" for i in range(n_genes)]

# ----------------------------------------
# Create AnnData object
# AnnData is the standard format for
# single cell data - think of it like
# a smart Excel sheet that stores:
# - expression matrix (X)
# - cell information (obs)
# - gene information (var)
# ----------------------------------------
adata = ad.AnnData(
    X=expression_data,
    obs=pd.DataFrame(
        {'cell_type': cell_types},
        index=cell_names
    ),
    var=pd.DataFrame(index=gene_names)
)

print(f"Created AnnData object:")
print(f"  Cells : {adata.n_obs}")
print(f"  Genes : {adata.n_vars}")
print(f"  Cell types: {adata.obs['cell_type'].unique().tolist()}")

# Save the data
adata.write_h5ad("data/single_cell_raw.h5ad")
print("\nRaw data saved to data/single_cell_raw.h5ad")
print("Done!")
