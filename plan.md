# Engineering Plan: SERGIO Synthetic Pre-Pre-Training for State ST

## Summary

Train the State Transition (ST) model from scratch on SERGIO-generated synthetic scRNA-seq data using the `X_hvg` input path (log-normalized expression counts, no PLM). Evaluate on a fixed synthetic test set. The goal is to establish that State learns meaningful perturbation structure from synthetic data and identify the best training configuration to use as an initializer before real-data fine-tuning.

**Scope of this plan:** Build the data pipeline, format synthetic data for State, run baseline training, and sweep key hyperparameters. REPTILE meta-learning is deferred to a follow-on plan.

---

## 1. Input Format: What State Expects

State ST with `embed_key=X_hvg` takes:

```
.obsm["X_hvg"]    float32  (n_cells, n_hvgs)   — log-normalized HVG expression
.obs["gene"]      str                           — perturbation name; "non-targeting" for control
.obs["cell_type"] str                           — cell type / bin label
.obs["gem_group"] str                           — batch label (can be a constant like "synthetic")
```

The preprocessing pipeline (`state tx preprocess_train`) does:
1. `sc.pp.normalize_total(adata)` — library-size normalize to 10k counts
2. `sc.pp.log1p(adata)` — log1p transform
3. `sc.pp.highly_variable_genes(adata, n_top_genes=N)` — select HVGs
4. `adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()`

For synthetic data we run this pipeline on each GRN instance individually (HVG selection makes sense per-GRN since gene count is fixed) or skip HVG selection and use all genes (if n_genes is small enough, e.g. ≤ 400).

**Perturbation encoding**: `pert_rep: onehot` (default) — one-hot over unique perturbation names within a dataset. Since each GRN instance has its own synthetic gene names, we use *positional* perturbation names: `"gene_0_KO"`, `"gene_1_KO"`, ..., `"gene_{k}_KO"` where k is the index of the perturbed gene within the GRN. This makes the encoding consistent across GRN instances.

---

## 2. SERGIO Data Generation Pipeline

### 2.1 Per-GRN instance

```python
# Pseudocode — implement in scripts/generate_sergio_dataset.py
for grn_seed in range(N_GRN_INSTANCES):
    # 1. Generate scale-free GRN
    grn = generate_scale_free_grn(n_genes=N_GENES, m=M, seed=grn_seed)

    # 2. Simulate control cells
    sim = sergio(number_genes=N_GENES, number_bins=N_BINS, number_sc=N_SC,
                 noise_params=NOISE_PARAMS, ...)
    sim.build_graph(targets, regs)
    sim.simulate()
    control_expr = sim.getExpressions()   # (n_bins, n_genes, n_sc)

    # 3. For each master regulator to perturb:
    for gene_idx, gene_name in enumerate(master_regulators[:K_PERTS]):
        sim_p = sergio(...)
        sim_p.build_graph(targets, regs)
        apply_ko_perturbation(sim_p.graph_, gene_idx)
        sim_p.simulate()
        pert_expr = sim_p.getExpressions()   # (n_bins, n_genes, n_sc)

        # 4. Build AnnData and save as h5ad
        adata = build_anndata(
            control_expr=control_expr,
            pert_expr=pert_expr,
            pert_name=f"gene_{gene_idx}_KO",
            n_bins=N_BINS,
            grn_seed=grn_seed,
        )
        adata.write_h5ad(f"output/grn_{grn_seed}/gene_{gene_idx}_KO.h5ad")
```

### 2.2 AnnData construction

Each H5AD contains **both** control and perturbed cells for one (GRN instance, perturbation) pair:

```
adata.X               — raw SERGIO expression (n_cells, n_genes), float32
adata.obs["gene"]     — "non-targeting" for control cells, "gene_{k}_KO" for perturbed
adata.obs["cell_type"]— "bin_0", "bin_1", ..., "bin_{n_bins-1}"
adata.obs["gem_group"]— "grn_{seed}" (serves as batch identifier)
adata.var.index       — "gene_0", "gene_1", ..., "gene_{n_genes-1}"
```

After construction, run preprocessing in-place:

```python
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# For small n_genes: skip HVG selection, use all genes as HVGs
adata.obsm["X_hvg"] = adata.X.toarray() if sparse else adata.X
# For larger n_genes: standard HVG selection
# sc.pp.highly_variable_genes(adata, n_top_genes=min(N_GENES, 400))
# adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()
adata.write_h5ad(output_path)
```

### 2.3 KO perturbation implementation

SERGIO has no built-in perturbation API. Modify `sim.graph_` after `build_graph` but before `simulate`:

```python
def apply_ko_perturbation(graph, gene_idx):
    """Knockout: set basal rate to 0 for master regulators,
    remove gene from regulation params of downstream targets."""
    gene_name = list(graph.keys())[gene_idx]
    # Zero basal rates (master regulators)
    if 'rates' in graph[gene_name]:
        graph[gene_name]['rates'] = [0.0] * len(graph[gene_name]['rates'])
    # Remove this gene as a regulator in all targets
    for other_gene, info in graph.items():
        if 'params' in info:
            info['params'] = [p for p in info['params'] if p['reg'] != gene_name]
```

### 2.4 Output directory structure

```
data/sergio_synthetic/
  train/
    grn_0000/
      gene_0_KO.h5ad     # control + gene_0_KO cells
      gene_1_KO.h5ad
      ...
    grn_0001/
      ...
  test/
    grn_5000/            # seeds disjoint from train
      gene_0_KO.h5ad
      ...
```

Each file contains ~`N_BINS × N_SC × 2` cells (control + perturbed), e.g. 9 bins × 200 cells × 2 = 3,600 cells per file.

---

## 3. SLURM Generation Job

Generation is CPU-bound and embarrassingly parallel. Use a SLURM array:

```bash
# scripts/submit_generate.sh
sbatch --array=0-499 --cpus-per-task=4 --mem=8G \
  --wrap="python scripts/generate_sergio_dataset.py \
    --grn-seed \$SLURM_ARRAY_TASK_ID \
    --n-genes 400 \
    --n-bins 9 \
    --n-sc 200 \
    --k-perts 20 \
    --output-dir data/sergio_synthetic/train \
    --noise-params dpd_low"
```

Estimated time: ~3 min per GRN instance (400 genes, 20 perturbations) × 500 instances = ~25 CPU-hours total.

---

## 4. Fixed Test Set (generate once, never modify)

- 40 GRN instances (seeds 5000–5039), disjoint from training
- Stratified: {100 genes, 400 genes} × {no noise, dpd_low noise} = 4 sub-groups × 10 instances each
- 9 cell types, 200 cells/type, 20 KO perturbations per GRN (master regulators)
- Hold-out split: 2 of 9 cell types withheld entirely from training within each test GRN

Generate and preprocess once:

```bash
python scripts/generate_sergio_dataset.py \
  --grn-seed-range 5000 5040 \
  --output-dir data/sergio_synthetic/test \
  --stratified
```

---

## 5. TOML Configuration for State Training

State's data module reads a TOML that maps dataset names to directories of h5ad files.

```toml
# configs/sergio_train.toml
[datasets]
sergio = "data/sergio_synthetic/train"

[training]
sergio = "train"

[zeroshot]
# No zeroshot split for initial baseline — test set is separate
```

For train/val split: hold out a random 10% of GRN instances as validation (no cell types withheld — that's reserved for the fixed test set evaluation).

---

## 6. Training Command

```bash
state tx train \
  data.kwargs.toml_config_path=configs/sergio_train.toml \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  training.max_steps=40000 \
  training.batch_size=8 \
  model=state \
  output_dir="$HOME/state_runs" \
  name="sergio_baseline"
```

Key differences from real-data training:
- `control_pert=non-targeting` (SERGIO doesn't have "non-targeting" guide; we label unperturbed cells this way)
- `output_space=gene` (predict back to HVG space, n_hvgs = n_genes for synthetic data)
- `pert_dim` is set automatically from the one-hot map size = n_unique_perturbation_names in the training set

---

## 7. Model Dimensions

With 400 genes (all used as HVGs), 20 perturbation types per GRN × 500 GRN instances = 10,000 unique pert names across the full training set. The `pert_encoder` MLP input dim will be 10,000 (one-hot). This is large but manageable (the first linear layer maps 10k → 768).

Alternative: use a **gene-position** encoding instead of name-based one-hot. Set `pert_dim = n_genes` and encode each perturbation as a binary vector with 1 at the knocked-out gene's index. This makes pert encoding consistent and compact regardless of GRN count:
- `pert_dim = 400` (same as n_genes)
- `pert_emb[gene_idx] = 1`, all others = 0
- Requires a custom `pert_onehot_map` builder in the data pipeline

**Recommendation**: start with name-based one-hot (simpler, uses existing machinery). Switch to gene-position encoding if pert_encoder convergence is slow.

---

## 8. Evaluation

Use `cell-eval` via `state tx predict`. Point at the fixed test set TOML:

```bash
state tx predict \
  --output-dir $HOME/state_runs/sergio_baseline \
  --checkpoint final.ckpt \
  --toml configs/sergio_test.toml
```

Metrics (from cell-eval):
- Perturbation discrimination score
- Pearson correlation of expression changes
- AUPRC for DE gene prediction
- Spearman correlation of log2FC
- Effect size Spearman

Synthetic advantage: ground-truth DEGs and log2FC are computed exactly from SERGIO outputs.

---

## 9. Experiment Sweeps

Run all sweeps against the fixed test set. Gate on baseline first.

### 9.1 Baseline (run first)
- 400 genes, no noise (dpd=0), 9 bins, 200 cells/bin, 20 KO perts on master regs
- 500 GRN training instances, 40k steps

### 9.2 Noise sweep (after baseline clears)
- Conditions: none | dpd_low | dpd_high | outlier_low | lib_size_variation
- Fix: 400 genes, 500 GRN instances

### 9.3 GRN size sweep
- 100 vs 400 genes
- Fix: dpd_low, 500 GRN instances

### 9.4 Data quantity sweep
- 100 / 250 / 500 / 1000 GRN instances
- Fix: 400 genes, dpd_low

### 9.5 Perturbation diversity sweep
- KO only | KO + KD | KO + OE
- Fix: 400 genes, dpd_low, 500 GRN instances

---

## 10. Checkpoint Transfer to Real Data

Once a best synthetic config is found, load it as an initializer for real-data training:

```bash
state tx train \
  data.kwargs.toml_config_path=configs/replogle_train.toml \
  data.kwargs.embed_key=X_hvg \
  model.kwargs.init_from=$HOME/state_runs/sergio_best/checkpoints/final.ckpt \
  name="replogle_from_sergio"
```

State handles mismatched `pert_encoder` dimensions automatically (rebuilds the layer if pert_dim differs). The transformer backbone and basal_encoder weights transfer directly.

---

## 11. Questions

1. **HVG selection on synthetic data**: Use all genes as HVGs (simplest, no info loss for small GRNs) or apply standard HVG selection? For 400 genes, using all is likely fine. For 100 genes, definitely use all. Recommend: skip HVG selection for synthetic, set `n_hvgs = n_genes`.

2. **Perturbation encoding strategy**: Name-based one-hot (one entry per "gene_k_KO" × GRN instance, large but simple) vs. gene-position binary vector (pert_dim = n_genes, consistent across GRNs). The gene-position encoding requires a small custom data loader change. Which do you prefer?

3. **Number of GRN training instances**: Plan proposes 500. Compute budget? At ~3 min/instance on CPU, 500 = ~25 CPU-hours (trivially parallelizable on SLURM). Training time is the bottleneck — more instances = more steps needed.

4. **Noise level for baseline**: Start with zero noise (cleanest signal to verify learning) or dpd_low immediately? Zero noise risks overfitting to clean synthetic patterns that don't generalize.

5. **Test set size**: Plan proposes 40 GRN instances (10 per stratification cell). Is this sufficient, or do you want more instances per sub-group?

6. **How many bins (cell types) per GRN**: Plan uses 9 bins, holding out 2 for the generalization test. Should the held-out bins be consistent positions (e.g., always bins 0 and 1) or random per GRN instance?
