# Plan 2: Wiring SERGIO Synthetic Data into State Transition Training

---

## Overview

The State Transition (ST) model lives at `/mnt/polished-lake/shared/models/state/state/`.
It uses `cell-load`'s `PerturbationDataModule` to load H5AD files, configured via a TOML
pointing to dataset directories. Training is launched with `state tx train [hydra overrides]`.

This plan covers everything needed to go from the generated SERGIO H5AD files to a trained
ST model checkpoint, including a first baseline, a validation strategy, and the full ablation
sweep.

---

## 1. Column Mapping

Our SERGIO H5AD columns map directly to what the data module expects:


| SERGIO `obs` column | `state tx train` kwarg | Value           | Semantics |
| ------------------- | ---------------------- | --------------- | --------- |
| `gene`              | `pert_col`             | `gene`          | Perturbation identity (`SYN_0042_KD_010` or `non-targeting`) |
| `cell_type`         | `cell_type_key`        | `cell_type`     | GRN seed — the "cell line" analog (distinct regulatory topology) |
| `gem_group`         | `batch_col`            | `gem_group`     | Bin — minor variation of the same GRN (different master-regulator rates) |
| `non-targeting`     | `control_pert`         | `non-targeting` | Control perturbation label |
| `X_hvg` (obsm)      | `embed_key`            | `X_hvg`         | Log-normalized expression used as model input |


Model dimensions:


| Parameter    | Value | Source                              |
| ------------ | ----- | ----------------------------------- |
| `input_dim`  | 2000  | `X` shape (full 2000-gene pool)     |
| `output_dim` | 2000  | gene-space prediction               |
| `pert_dim`   | 2000  | gene-position magnitude vector      |
| `hidden_dim` | 768   | default `state.yaml`                |


---

## 2. Perturbation Map

The data module requires a `perturbation_features_file`: a `Dict[str, torch.Tensor]` mapping
every value that appears in `obs["gene"]` to a float32 vector of length `pert_dim=2000`.

This matches the plan.md perturbation encoding exactly — position `k` holds the perturbation
strength for gene `SYN_{k:04d}`:

```python
# scripts/build_pert_map.py
import torch, numpy as np, json, pathlib

POOL_SIZE = 2000
PERT_STRENGTHS = {"KO": 1.0, "KD_010": 0.1, "KD_050": 0.5, "KD_080": 0.8}

pert_map = {"non-targeting": torch.zeros(POOL_SIZE, dtype=torch.float32)}
for k in range(POOL_SIZE):
    for label, strength in PERT_STRENGTHS.items():
        v = torch.zeros(POOL_SIZE, dtype=torch.float32)
        v[k] = strength
        pert_map[f"SYN_{k:04d}_{label}"] = v

torch.save(pert_map, "configs/pert_onehot_map.pt")
print(f"Saved {len(pert_map)} entries.")  # 1 + 2000*4 = 8001
```

Build once; reuse for all training runs. Place at `simulate/sergio/configs/pert_onehot_map.pt`.

---

## 3. TOML Config

`cell-load` discovers H5AD files by scanning the directory listed in `[datasets]`.
Each unique value of `obs[batch_col]` (`gem_group`) acts as a mini-batch label;
each unique value of `obs[cell_type_key]` (`cell_type` = `grn_XXXX`) acts as a cell type (the "cell line" analog — seed = GRN topology; bin = minor variation within same GRN, stored in `gem_group`).

The `*_merged` directories (see §8.1) contain one H5AD per condition
`(grn_type, grn_size, noise_level, pert_type)`. Ablation runs point at a symlink
subdirectory containing only the condition files matching the ablation filter.

### 3.1 TOML structure

The TOML `[zeroshot]` section is used to hold out entire cell types (bins) for validation.
For training data we hold out `bin_4` (the last bin for `n_bins=5` instances) as a zeroshot
val split. The test set uses a separate TOML pointing at `test_mini_merged/`.

```toml
# configs/sergio_mini_train.toml

[datasets]
sergio_mini = "/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio/data/sergio_synthetic/mini_merged"

[training]
sergio_mini = "train"

# Hold out bin_4 across all instances for validation
[zeroshot]
"sergio_mini.bin_4" = "val"
```

```toml
# configs/sergio_mini_test.toml

[datasets]
sergio_mini_test = "/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio/data/sergio_synthetic/test_mini_merged"

[training]
sergio_mini_test = "test"
```

---

## 4. Training Command

**State model location**: `/mnt/polished-lake/shared/models/state/state/`

```bash
cd /mnt/polished-lake/shared/models/state/state

uv run state tx train \
  data.kwargs.toml_config_path=/abs/path/to/configs/sergio_mini_train.toml \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=grn_seed \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=all \
  data.kwargs.perturbation_features_file=/abs/path/to/configs/pert_onehot_map.pt \
  training.max_steps=10000 \
  training.batch_size=8 \
  model=state \
  model.kwargs.hidden_dim=768 \
  output_dir=$HOME/state_runs \
  name=sergio_mini_baseline
```

For the full dataset (once mini baseline works):

```bash
  data.kwargs.toml_config_path=.../sergio_train.toml \
  training.max_steps=40000 \
  training.batch_size=16 \
  name=sergio_baseline
```

---

## 5. Step-by-Step Execution Plan

### 5.1 Build merged H5ADs

```bash
cd /mnt/polished-lake/home/nkondapaneni/state/simulate/sergio

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/mini \
    --dst data/sergio_synthetic/mini_merged

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/test_mini \
    --dst data/sergio_synthetic/test_mini_merged
```

### 5.2 Build pert_onehot_map.pt

```bash
uv run python scripts/build_pert_map.py
# → configs/pert_onehot_map.pt  (8001 entries, each float32 shape [2000])
```

### 5.3 Verify cell-load compatibility

```bash
cd /mnt/polished-lake/shared/models/state/state

uv run python -c "
from cell_load.utils.modules import get_datamodule

dm = get_datamodule('PerturbationDataModule', {
    'toml_config_path': '/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio/configs/sergio_mini_train.toml',
    'pert_col': 'gene',
    'cell_type_key': 'cell_type',
    'batch_col': 'gem_group',
    'control_pert': 'non-targeting',
    'output_space': 'all',
    'perturbation_features_file': '/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio/configs/pert_onehot_map.pt',
}, batch_size=8, cell_sentence_len=512)
dm.setup('fit')
print('var_dims:', dm.get_var_dims())
print('n perts:', len(dm.pert_onehot_map))
batch = next(iter(dm.train_dataloader()))
print('batch keys:', list(batch.keys()))
print('pert_emb shape:', batch['pert_emb'].shape)
print('ctrl_cell_emb shape:', batch['ctrl_cell_emb'].shape)
"
```

Expected:

- `var_dims`: `{input_dim: 2000, output_dim: 2000, pert_dim: 2000, ...}`
- `pert_emb shape`: `[batch, 2000]`
- `ctrl_cell_emb shape`: `[batch, 2000]`

### 5.4 Run mini baseline

Use `dataset_mini` (675 tasks, ~30 min on SLURM) + `test_mini` (10 tasks, ~8 min).
Both should be finished before starting this step.

```bash
cd /mnt/polished-lake/shared/models/state/state
uv run state tx train \
  [kwargs from Section 4, mini variant] \
  name=sergio_mini_baseline
```

Check that:

- Training loss decreases over 10k steps
- Val loss tracks training loss (not diverging)
- `wandb` logs show energy distance going down

### 5.5 Evaluate on test_mini

```bash
uv run state tx infer \
  --model-dir $HOME/state_runs/sergio_mini_baseline \
  --toml-config /abs/path/to/configs/sergio_mini_test.toml \
  --output-dir $HOME/state_runs/sergio_mini_baseline/eval
```

Metrics to compute per prediction file:

- **Pearson Δ Corr**: Pearson r between predicted and observed pseudobulk expression delta (perturbed − control), averaged across perturbations
- **Log2FC Spearman**: Spearman correlation of log fold changes restricted to true-significant DE genes

Stratify by the 2-condition evaluation framework (see Section 7) and by:

- `grn_type` (ER / BA / BA-VM)
- `pert_type` (KO / KD_010 / KD_050 / KD_080)
- `ko_out_degree` (hub vs leaf perturbations)

Note: ignore the `obs["split"]` column — all test seeds (5000–5024) are out-of-sample regardless of bin label.

### 5.6 Full dataset training

Once mini baseline looks healthy, submit full `dataset1` + `test_set`:

```bash
sbatch submit_generate_dataset.sh   # 3300 tasks, ~4 hrs
sbatch submit_generate_test_set.sh  # 100 tasks

# After completion, build merged dirs:
uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/train \
    --dst data/sergio_synthetic/train_merged

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/test \
    --dst data/sergio_synthetic/test_merged

# Then train:
uv run state tx train \
  [kwargs from Section 4, full variant] \
  training.max_steps=40000 \
  name=sergio_baseline
```

---

## 6. Ablation Sweeps

Each ablation is a pair of training runs (base vs. base+addition) evaluated on the same fixed
synthetic test set. The design principle: the base condition is the one closest to real data
(noisy, graded, complex); the addition is a cleaner synthetic condition unavailable in real data.
If adding the cleaner condition improves test performance, it suggests synthetic data's clean
signal scaffolds learning — a positive indicator for synthetic→real transfer.

All runs use the same fixed test set for evaluation (see Section 7).

The merged directory contains files named `{grn_type}_size{n}_noise{label}_{pert_type}.h5ad`.
For each ablation, create a symlink directory containing only the matching files and point
the TOML `[datasets]` at it. The `manifest.csv` written by `build_merged_h5ads.py` enables
programmatic filtering.


| # | Run name | Train filter | Question |
|---|----------|-------------|----------|
| 0 | `baseline_all` | all files | Ceiling: does max diversity win? |
| 1a | `ablation_ba_only` | `grn_type == "BA"` | Base for #1 |
| 1b | `ablation_ba_plus_er` | `grn_type in ["BA", "ER"]` | Does homogeneous (ER) cascade structure scaffold hub-heavy (BA) learning? |
| 3a | `ablation_kd_only` | `pert_type.str.startswith("KD")` | Base: graded perturbations (closest to CRISPRi real data) |
| 3b | `ablation_kd_plus_ko` | `pert_type.str.startswith("KD") or pert_type == "KO"` | Does clean full-KO signal scaffold learning graded effects? |
| 4a | `ablation_noise_high` | `noise_label == "high"` | Base: noisy (closest to real scRNA-seq) |
| 4b | `ablation_noise_high_plus_clean` | `noise_label in ["high", "low", "none"]` | Does clean/low-noise signal scaffold noisy learning? |
| 5a | `ablation_size_100` | `grn_size == 100` | Base: complex GRNs (closest to real network complexity) |
| 5b | `ablation_size_100_plus_010` | `grn_size in [100, 10]` | Does simple GRN structure (isolated cascades) scaffold complex? |


```python
# scripts/build_ablation_dir.py  (to be written)
# Filter manifest, symlink matching files into ablation dir, write TOML

import pandas as pd, pathlib

manifest = pd.read_csv("data/sergio_synthetic/train_merged/manifest.csv")
filtered = manifest[manifest.grn_type == "ER"]

ablation_dir = pathlib.Path("data/sergio_synthetic/ablation_er_only")
ablation_dir.mkdir(exist_ok=True)
merged_dir = pathlib.Path("data/sergio_synthetic/train_merged")

for _, row in filtered.iterrows():
    src = (merged_dir / row.merged_file).resolve()
    dst = ablation_dir / row.merged_file
    if not dst.exists():
        dst.symlink_to(src)
```

---

## 7. Open Questions


| #   | Question                                                                                    | How to resolve                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1   | What is the minimum `cell_sentence_len` that fits in GPU memory at 2000-gene input?         | Profile memory with `batch_size=8, cell_sentence_len=512` on H200                                                                   |
| 2   | Does energy distance loss converge well on SERGIO's structured expression (mostly zeros)?   | Watch train loss in Step 5.4 — if stuck, try `loss=mse` first                                                                       |
| 3   | Does `obs["split"]` (train/held_out) need to be communicated to the TOML, or is it ignored? | `cell-load` uses only TOML zeroshot/fewshot sections for splits — `obs["split"]` is for our own eval filtering, not the data module |


---

## 7. Evaluation Framework

### Rationale

Each GRN seed is a distinct regulatory network — the "cell line" analog. Bins are stochastic
expression realizations of the same GRN (different master regulator basal rates), not
independent cell lines. Out-of-GRN generalization is not meaningfully learnable: the same
gene can be a hub activator in one GRN and a leaf repressor in another, and neither the
perturbation embedding nor the control expression can resolve this for an unseen GRN. This is addressed by the
State Embedding model which can represent cells with similar gene expression counts, but different underlying 
GRNs differently. 

The Replogle zeroshot task (predict perturbation effects in a held-out cell line, given the
model has seen those perturbations in other cell lines) is also not reproducible: it requires
contexts with genuinely different regulatory topologies and genes with consitent relationships to other genes.
This simulation is constructed with completely random networks where genes do not preserve a consistent relationship

The meaningful evaluation is **in-cell-line perturbation holdout** — analogous to Replogle's
fewshot setup. All 4 training seeds are used; the question is whether the model can predict
perturbation effects it was not trained on within a GRN it has seen.

Dataset design: **4 seeds × 10 bins** per graph type. All 10 bins used for training (no bin
holdout). Perturbation holdout handled at training time via TOML `[fewshot]`.

---

### 7.1 Evaluation conditions

For each of the 4 seeds, the K=10 perturbable genes are split at training time:

| Condition | # genes | Training treatment | Tests |
|-----------|---------|-------------------|-------|
| pert-type-holdout | 3 | 3 of 4 pert_types seen (e.g. KO + KD_050 + KD_080); KD_010 held out | Interpolation across perturbation strength |
| in-CL-common | 7 | All pert_types seen normally (~50 cells) | Upper bound / interpolation |

`in-CL-excluded` and `in-CL-underrepresented` are not used. Excluded requires causal
discovery from observational data — not learnable when GRN roles are random across seeds.
Underrepresented has no native cell-load TOML support (no per-perturbation fraction key).

Holdout specified via TOML `[fewshot]` per seed, generated deterministically from the
per-seed manifest by `scripts/build_inCL_holdout_toml.py`.

```toml
[datasets]
sergio_mini = ".../mini_incl_merged"

[training]
sergio_mini = "train"

[zeroshot]

[fewshot]
# pert-type-holdout: 3 genes seen with KO/KD_050/KD_080 only; KD_010 held out as test
# Repeated for each bin (bin_0 … bin_9) since fewshot operates at (dataset, cell_type) level
[fewshot."sergio_mini.bin_0"]
test = ["SYN_0042_KD_010", "SYN_0019_KD_010", "SYN_0077_KD_010"]
[fewshot."sergio_mini.bin_1"]
test = ["SYN_0042_KD_010", "SYN_0019_KD_010", "SYN_0077_KD_010"]
# ... generated for bin_0 through bin_9 by scripts/build_inCL_holdout_toml.py
```

---

### 7.2 Out-of-GRN sanity check (seed 4, optional)

One additional seed never seen during training. Expected near-random performance — confirms
the model is not somehow generalizing across random GRNs. Not a primary evaluation target.

---

## 8. Known Issues

### 8.1 cell-load does not support recursive directory scanning

**Root cause**: `cell_load._find_dataset_files()` calls `glob.glob(pattern)` without
`recursive=True` (line 834 of `perturbation_dataloader.py`), so `**/*.h5ad` in a TOML
path silently matches zero files.

Additionally, `_find_dataset_files()` deduplicates by `fpath.stem`, so multiple h5ad
files with the same name (e.g. `SYN_0033_KO.h5ad` appearing once per GRN seed) would
overwrite each other in the dict, leaving only one file per perturbation name.

**Fix**: Merge source files into condition-grouped H5ADs — one per
`(grn_type, grn_size, noise_level, pert_type)` — using `scripts/build_merged_h5ads.py`.
This produces ~165 files for the full dataset (fewer for mini). Each file contains all
GRN seeds for that condition. Provenance columns (`grn_type`, `grn_size`, `noise_label`,
`grn_seed`) are added to `obs` during the merge. A `manifest.csv` is written alongside
the merged files for programmatic ablation filtering.

This grouping aligns with the ablation axes in plan.md §5.3 — every ablation filter
(grn_type, grn_size, noise_level, pert_type) maps cleanly to a file-level filename filter.

**Script**: `scripts/build_merged_h5ads.py`

```bash
# Run once after dataset generation, before training
uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/mini \
    --dst data/sergio_synthetic/mini_merged

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/test_mini \
    --dst data/sergio_synthetic/test_mini_merged

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/train \
    --dst data/sergio_synthetic/train_merged

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/test \
    --dst data/sergio_synthetic/test_merged
```

The merge is idempotent — re-running skips already-written files. TOML `[datasets]`
paths point at the `*_merged` directories (or ablation symlink subdirs).