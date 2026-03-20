---
name: state_repo_overview
description: Architecture, data format, and training interface for the ARC State repo (arc-state PyPI package)
type: project
---

# ARC State Repo Overview

**Why:** Context for engineering work on synthetic pre-pre-training using SERGIO-generated data.
**How to apply:** Reference when writing data pipelines, training configs, or evaluation scripts for State.

## Two models

- **ST (State Transition)**: Perturbation prediction. `state tx train/predict/infer`. Main target for synthetic pre-pre-training.
- **SE (State Embedding)**: Cell embedding via PLM. `state emb fit/transform`. Not relevant for synthetic work — we use `X_hvg` path only.

## ST Input Format

State ST operates in two modes, controlled by `embed_key`:
- **`embed_key: X_hvg`** (our path): Takes log-normalized HVG expression as input. **No PLM needed.** `input_dim = n_HVGs`.
- **`embed_key: X_uce` etc.**: Takes cell embeddings from SE model. Requires PLM pre-embedding.

Preprocessing pipeline (via `state tx preprocess_train`):
1. `sc.pp.normalize_total(adata)` — library-size normalize
2. `sc.pp.log1p(adata)` — log1p transform
3. `sc.pp.highly_variable_genes(adata, n_top_genes=N)` — select HVGs
4. `adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()`

## H5AD Schema Required by cell-load

Each H5AD file needs:
- `.X`: raw counts (or log-normalized; depends on transform setting)
- `.obsm["X_hvg"]`: log-normalized HVG expression (shape: n_cells × n_hvgs)
- `.obs["gene"]`: perturbation column (default `pert_col`)
- `.obs["cell_type"]`: cell type column (default `cell_type_key`)
- `.obs["gem_group"]`: batch column (default `batch_col`)

Control cells have `.obs["gene"] == "non-targeting"` (configurable via `control_pert`).

Data is organized as a directory of H5AD files. The TOML config maps dataset names to those directories.

## Perturbation Encoding

Default `pert_rep: onehot` — one-hot over all unique perturbation names in the training set. `pert_dim = n_unique_perts`.

For genetic perturbations: `pert_flags` is a binary vector (length = n_genes) with 1 at the perturbed gene's position. Used by scGPT-genetic model; not the primary path for State ST.

## Architecture (state model)

- `hidden_dim: 768`
- `cell_set_len: 512` (cells per "sentence")
- Transformer backbone: Llama, 8 layers, 12 heads, bidirectional attention
- `basal_encoder`: MLP (n_HVG → 768)
- `pert_encoder`: MLP (n_perts → 768)
- Combined = pert_encoding + basal_encoding → transformer → project_out → output
- Loss: energy distance (geomloss `SamplesLoss`)
- `predict_residual: True` — predicts delta from control, not absolute expression

## Training Command

```bash
state tx train \
  data.kwargs.toml_config_path=examples/fewshot.toml \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  training.max_steps=40000 \
  model=state \
  output_dir="$HOME/state" \
  name="test"
```

Key overrides for synthetic data: `data.kwargs.control_pert=non-targeting`, `data.kwargs.output_space=gene`.

## Checkpoint Transfer

`model.kwargs.init_from` loads a checkpoint for fine-tuning. Mismatched `pert_encoder` dimensions are automatically handled by rebuilding that layer. This enables: train on synthetic → load checkpoint → fine-tune on real data with different pert set.
