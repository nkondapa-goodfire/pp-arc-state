# CellOracle Dataset Generation Plan

## Goal

Use pre-computed GRNs from CellOracle to generate synthetic single-cell gene expression data with paired perturbation conditions, formatted for State model training.

---

## `generate_dataset.py` — Steps

1. **Load pre-computed GRNs** from the CellOracle repo
  - CellOracle stores inferred GRNs as cell-cluster-specific regression models
  - Each GRN defines TF → target gene regulatory weights for one cell type
2. **Generate unperturbed gene expression vectors**
  - Simulate steady-state expression for each GRN (control condition)
3. **Perturb the top 10 genes and generate perturbed expression vectors**
  - For each GRN, identify the top 10 hub genes (by network centrality score)
  - Apply in silico knockout of each gene, knockdown at 0.75, knockdown at 0.25, upregulate at 1.25, and 1.75 and propagate through the GRN
  - Produces one perturbed expression vector per gene per GRN
4. **Save data**
  - Output as AnnData (`.h5ad`) with control and perturbed cells
  - Include `obs` columns: `cell_type` (GRN source), `gene` (perturbation), `gem_group` (batch)
  - Compatible with `state tx train` data kwargs (`pert_col=gene`, `cell_type_key=cell_type`, `control_pert=non-targeting`)

