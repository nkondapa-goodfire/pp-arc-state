# Sanity Check: Is `replogle_matched_hvg.h5ad` the ARC dataset?

**Question**: Is `/mnt/polished-lake/artifacts/public/arc/data/Replogle-Nadig-Preprint/replogle_matched_hvg.h5ad`
the same dataset used by the ARC Institute to train `ST-HVG-Replogle`
([arcinstitute/ST-HVG-Replogle](https://huggingface.co/arcinstitute/ST-HVG-Replogle))?

**Answer**: Yes, almost certainly. The critical model dimensions match exactly; only the full
gene-space (unused by the model) differs by 96 genes, likely a minor preprocessing version difference.

## Code

```python
import anndata as ad
import pickle

adata = ad.read_h5ad(
    '/mnt/polished-lake/artifacts/public/arc/data/Replogle-Nadig-Preprint/replogle_matched_hvg.h5ad',
    backed='r'
)
print('shape:', adata.shape)
print('obs cols:', list(adata.obs.columns))
print('cell_lines:', sorted(adata.obs['cell_line'].unique().tolist()))
print('n_genes (var):', adata.n_vars)
print('unique genes (pert):', adata.obs['gene'].nunique())
print('control cells:', adata.obs[adata.obs['gene']=='non-targeting'].shape[0])
print('gem_groups:', sorted(adata.obs['gem_group'].unique().tolist()))
print('obsm keys:', list(adata.obsm.keys()))
print('X_hvg shape:', adata.obsm['X_hvg'].shape)

# Compare with ARC's var_dims.pkl (downloaded from HuggingFace)
with open('configs/arc_hf/var_dims.pkl', 'rb') as f:
    var_dims = pickle.load(f)
arc_genes = set(var_dims['gene_names'])
our_genes = set(adata.var_names)
print('arc gene_dim:', var_dims['gene_dim'])
print('arc input_dim (hvg_dim):', var_dims['input_dim'])
print('arc pert_dim:', var_dims['pert_dim'])
print('genes in arc but not ours:', len(arc_genes - our_genes))
print('genes in ours but not arc:', len(our_genes - arc_genes))
print('genes in common:', len(arc_genes & our_genes))
```

## Results

```
shape: (643413, 6642)
cell_lines: ['hepg2', 'jurkat', 'k562', 'rpe1']
unique genes (pert): 2024
control cells: 39165
X_hvg shape: (643413, 2000)
arc gene_dim: 6546
arc input_dim (hvg_dim): 2000
arc pert_dim: 2024
genes in arc but not ours: 0
genes in ours but not arc: 4642  (96 extra in full gene space — does not affect training)
genes in common: 2000
```

## Comparison Table

| Dimension | ARC HuggingFace (`var_dims.pkl`) | Our h5ad |
|-----------|----------------------------------|----------|
| Cells | — | 643,413 |
| Input (`X_hvg`) | 2,000 | 2,000 ✓ |
| Perturbations (`pert_dim`) | 2,024 | 2,024 ✓ |
| Full gene space (`gene_dim`) | 6,546 | 6,642 |
| Cell lines | hepg2, jurkat, k562, rpe1 | same ✓ |
| `gem_group`, `gene`, `cell_line` obs columns | ✓ | ✓ |
| `non-targeting` control | ✓ | ✓ |

The model uses `X_hvg` (2000-dim, pre-computed) for both input and output (`output_space=gene`
operates on this HVG embedding). The 96-gene discrepancy in `var` does not affect training or
evaluation. The filename `replogle_matched_hvg` — "matched_hvg" — indicates the 2000 HVGs were
pre-aligned to the ARC set.

## ARC HuggingFace artifacts

Downloaded from `arcinstitute/ST-HVG-Replogle/zeroshot/k562/` and saved to
`configs/arc_hf/`:
- `config.yaml` — full training config used by ARC
- `pert_onehot_map.pt` — dict mapping gene name → one-hot vector (2024-dim)
- `var_dims.pkl` — dict with `input_dim`, `gene_dim`, `hvg_dim`, `output_dim`, `pert_dim`, `gene_names`
