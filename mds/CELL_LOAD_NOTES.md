# cell_load Notes

`cell_load` is a third-party library (installed as a uv dependency) that handles all H5/AnnData dataloading for the State TX training and prediction pipelines. The State codebase wraps it via `scGPTPerturbationDataset` and `PerturbationDataModule`.

Package path: `~/.cache/uv/archive-v0/kan-3WK8lthpbPs59ofkW/lib/python3.13/site-packages/cell_load/`

---

## Key classes

| Class | File | Role |
|---|---|---|
| `PerturbationDataModule` | `data_modules/perturbation_dataloader.py` | Lightning DataModule; owns setup, splits, samplers |
| `PerturbationDataset` | `dataset/_perturbation.py` | `__getitem__` — returns one sample dict per cell |
| `GlobalH5MetadataCache` | (imported into dataloader) | Per-file cache of pert/cell_type/batch codes; shared across workers |
| `scGPTPerturbationDataset` | `state/tx/data/dataset/scgpt_perturbation_dataset.py` | State's subclass of `PerturbationDataset` |

---

## TOML config schema

`PerturbationDataModule` is configured by a TOML file passed as `toml_config_path`. The TOML has four sections:

```toml
[datasets]
<name> = "/path/to/file.h5ad"   # or a glob / list of files

[training]
<name> = "train"                 # dataset used for train/val

[zeroshot]
"<name>.<cell_type>" = "test"   # entire cell type held out for test only

[fewshot]
["<name>.<cell_type>"]
test = ["GENE_A", "GENE_B", ...]  # specific perturbations held out for test
# all other perturbations in this cell type go to train
```

For SERGIO data (`sergio_ppt_train.toml`, `sergio_tgt_train.toml`): only `[training]` is used — all cell types train, no holdout.

For Replogle-Nadig all-pert eval (`rpnd_train.toml`): `[zeroshot]` holds out k562 entirely for test.

For Replogle-Nadig fewshot eval (`rpnd_fewshot.toml`): `[fewshot]` holds out 858 specific k562 perturbations for test; the rest of k562 trains.

---

## Setup / scanning phase

During `setup()`, `PerturbationDataModule` scans every H5 file to collect:

- All perturbation names → `pert_onehot_map` (one-hot or custom featurization from `perturbation_features_file`)
- All batch names → `batch_onehot_map`
- All cell type names → `cell_type_onehot_map`

These global maps are then passed into every `PerturbationDataset` instance.

---

## Train/val/test splitting

Splitting happens inside `setup()`, **per cell type** within each file:

```
for ct in cache.cell_type_categories:
    ct_mask = cache.cell_type_codes == ct_idx
    # → route ctrl/pert indices to train/val/test subsets
    # based on [zeroshot] / [fewshot] / [training] config
```

- **Training cell type**: ctrl+pert indices go to train/val.
- **Zeroshot cell type**: all indices go to test.
- **Fewshot cell type**: indices for held-out perturbations → test; rest → train/val.

---

## `__getitem__` return dict

Each sample is a dict:

```python
{
    "pert_cell_emb":   Tensor,        # perturbed cell expression / embedding
    "ctrl_cell_emb":   Tensor,        # matched control cell (by mapping_strategy)
    "pert_emb":        Tensor,        # one-hot or custom featurization of perturbation
    "pert_name":       str,
    "dataset_name":    str,
    "batch_name":      str,
    "batch":           Tensor,        # one-hot of batch (gem_group)
    "cell_type":       str,           # raw string, e.g. "grn_0000" or "k562"
    "cell_type_onehot": Tensor,       # one-hot of cell type
    # optional if store_raw_expression=True:
    "pert_cell_counts": Tensor,
}
```

`cell_type_onehot` is looked up from the global `cell_type_onehot_map` built during setup. Its dimensionality = total number of unique cell types across all datasets.

---

## Batch sampler

`samplers.py` groups indices by `(cell_type, perturbation)` (or `(batch, cell_type, perturbation)`) to form balanced batches. Downsampling (`downsample_cells`) is applied per `(cell_type, perturbation[, batch])` group during this step.

---

## How `cell_type` is used in the State model

- `cell_type_key` kwarg (default `"cell_type"`) tells `cell_load` which `obs` column to read.
- The resulting `cell_type_onehot` tensor in each batch is consumed by `StateTransitionModel` as a conditioning signal (alongside `pert_emb` and `batch`).
- For SERGIO data: `cell_type = grn_{seed:04d}` — each GRN seed is treated as a distinct cell line.
- For Replogle-Nadig: `cell_type = cell_line` column (k562, rpe1, etc.).

---

## Control cell mapping strategies

`mapping_strategy` determines how control cells are paired to perturbed cells:

- `"batch"`: match control cells within the same batch/bin (used for SERGIO, where `gem_group` = bin)
- `"random"`: random control cell from the same cell type
- `"nearest"`: nearest-neighbor in embedding space

---

## Key kwargs passed from State train CLI

```
data.kwargs.toml_config_path   # path to TOML
data.kwargs.embed_key          # obsm key or None (X_hvg, X_uce, etc.)
data.kwargs.pert_col           # obs column for perturbation identity (gene)
data.kwargs.cell_type_key      # obs column for cell type (cell_type / cell_line)
data.kwargs.batch_col          # obs column for batch (gem_group)
data.kwargs.control_pert       # name of control perturbation (non-targeting)
data.kwargs.output_space       # "gene" | "embedding" | "all"
```
