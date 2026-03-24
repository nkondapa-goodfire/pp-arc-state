# Dataset Fixes

## 2026-03-24 — `ER_size010_seed0101` missing from SERGIO_TGT (unrecoverable)

**Cause**: `generate_er_grn` with `n_genes=10`, `p_edge=0.08`, `seed=101` produces a graph with **zero edges**. `grn_utils.py:73` calls `rng.choice(list(G.edges()), ...)` which raises `ValueError: a cannot be empty` when the edge list is empty.

**Why it happened**: Small ER graphs with low p_edge have high variance in edge count. Expected edges = 10×9×0.08 = 7.2, but seed 101 draws zero. Seeds 100, 102, 103 all generated non-empty graphs.

**Resolution**: This seed cannot be generated with the current config. SERGIO_TGT legitimately has **35 cell types** (not 36). `sergio_tgt_test.toml` already omits `ER_size010_seed0101`.

---

## 2026-03-22 — `cell_eval` broken import: `parallel_differential_expression` removed in `pdex 0.2.0`

**Error**:
```
ImportError: cannot import name 'parallel_differential_expression' from 'pdex'
```

**Root cause**: `cell_eval 0.6.8` imports `parallel_differential_expression` from `pdex`, but `pdex 0.2.0` (pinned in `uv.lock` for Python >= 3.11) renamed this function to just `pdex`. The two packages are out of sync.

**Files patched** (both in `.venv/lib/python3.11/site-packages/cell_eval/`):
- `_baseline.py:8`
- `_evaluator.py:10`

**Change** (same in both files):
```python
# Before
from pdex import parallel_differential_expression

# After
from pdex import pdex as parallel_differential_expression
```

**Justification**: `pdex.pdex()` has the same signature as the old `parallel_differential_expression()` — same parameters (`adata`, `groupby`, `mode`, `threads`, `is_log1p`, `geometric_mean`, `as_pandas`, `**kwargs`), same return type (`pl.DataFrame | pd.DataFrame`). The rename was cosmetic. All call sites in `cell_eval` continue to work unchanged.

**Note**: This is a venv patch — it will be lost if the venv is rebuilt. The proper fix is to update `cell_eval` to a version that targets `pdex >= 0.2.0`.

---

## 2026-03-24 — `cell_type` only encodes `grn_seed`, drops `grn_type` and `grn_size`

**Bug**: In `generate_dataset.py:build_anndata`, `cell_type` was set to `grn_{grn_seed:04d}` — encoding only the seed. Two GRNs from different types or sizes but the same seed number (e.g., `ER/size_010/grn_0000` and `BA/size_100/grn_0000`) would both get `cell_type = "grn_0000"`, collapsing distinct regulatory topologies into the same identity.

**Fix**: Changed to `{grn_type}_size{grn_size:03d}_seed{grn_seed:04d}` (e.g., `ER_size010_seed0000`), which uniquely identifies each GRN instance across all three axes.

**Files changed**: `generate_dataset.py:195`

**Retroactive data fix**: Run `scripts/fix_cell_type.py` to patch all existing H5AD files in `SERGIO_PPT`, `SERGIO_PPT_merged`, `SERGIO_TGT`, `SERGIO_TGT_train_merged`, `SERGIO_TGT_test_merged`. See script for usage.

**Downstream**: After patching the merged H5ADs, `sergio_tgt_test.toml` zeroshot keys must be updated to match the new format (e.g., `grn_0100` → `ER_size010_seed0100`). Check what cell types are present in `SERGIO_TGT_test_merged` after the fix and update the TOML accordingly.

---

## 2026-03-21 — `output_space=all` vs `output_space=gene`

**Context**: Arc's HuggingFace config uses `output_space=gene`. Our original training scripts used `output_space=all`. We changed to `gene` to match Arc.

**What `output_space` controls** (`_train.py:133-136`):
- `gene`: decoder output dim = `hvg_dim` (predict HVG-space only)
- `all`: decoder output dim = `gene_dim` (predict full transcriptome)

**For Replogle real data** (`var_dims.pkl`): `hvg_dim=2000`, `gene_dim=6546` — genuinely different decoders.

**For our SERGIO data**: `hvg_dim == gene_dim == 2000`, so both settings produce **identical model architecture**. The change has no numerical effect on pretraining.

**The `gene_decoder` wrinkle** (`base.py:269-270`): an extra decoder layer is created only when:
- `embed_key != "X_hvg"` and `output_space == "gene"`, OR
- `embed_key` is set and `output_space == "all"`

Our old combo (`embed_key=null, output_space=all`) → no extra decoder.
Our new combo (`embed_key=X_hvg, output_space=gene`) → no extra decoder (`embed_key == "X_hvg"` makes line 269 false).
The combo `embed_key=X_hvg, output_space=all` → **would** create an extra gene_decoder — avoid this.

**Verdict**: change to `output_space=gene` is correct for matching Arc's convention and ensuring clean checkpoint transfer to real-data fine-tuning, but does not affect SERGIO pretraining behavior.

---

## 2026-03-21 — Corrupted ER/size_010 files

**Cause**: 11 h5ad files in `ER/size_010` (noise_low and noise_none) had bad HDF5 object headers, likely from SLURM jobs killed mid-write during the original mini generation run.

**Detected via**:
```bash
uv run python -c "
import anndata, pathlib
for p in sorted(pathlib.Path('data/sergio_synthetic/mini').rglob('*.h5ad')):
    try:
        anndata.read_h5ad(p, backed='r').file.close()
    except OSError as e:
        print(p)
"
```

**Corrupted files**:
```
data/sergio_synthetic/mini/ER/size_010/noise_low/grn_0010/SYN_0414_KO.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_low/grn_0012/SYN_0398_KD_050.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_low/grn_0018/SYN_0560_KD_050.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_low/grn_0020/SYN_0818_KD_050.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_low/grn_0021/SYN_0687_KD_010.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0011/SYN_1200_KD_080.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0014/SYN_1718_KO.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0015/SYN_0687_KD_080.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0016/SYN_0187_KD_050.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0018/SYN_0560_KO.h5ad
data/sergio_synthetic/mini/ER/size_010/noise_none/grn_0021/SYN_1208_KD_010.h5ad
```

**Fix**: deleted corrupted files and regenerated the affected seeds:
```bash
cd /mnt/polished-lake/home/nkondapaneni/state/simulate/sergio

for seed in 10 12 18 20 21; do
  uv run python generate_dataset.py --config generation_configs/dataset_mini.json \
    --grn-type ER --grn-size 10 --noise-label low --seed $seed &
done
for seed in 11 14 15 16 18 21; do
  uv run python generate_dataset.py --config generation_configs/dataset_mini.json \
    --grn-type ER --grn-size 10 --noise-label none --seed $seed &
done
wait
```

---

## 2026-03-22 — `is_log1p mode is ENABLED by configuration/default, but no uns/log1p metadata was detected`

**Where it comes from**: `cell_load/data_modules/perturbation_dataloader.py` (in the uv cache), not from the `state` repo itself.

**Why it fires**: `PerturbationDataLoader` has `is_log1p=True` as its default. During init it scans every h5 dataset for a `uns/log1p` key — the metadata scanpy writes when you call `sc.pp.log1p(adata)`. Our SERGIO h5 files were log1p'd in `generate_dataset.py` (line 225: `sc.pp.log1p(adata)`), but the h5 export did not preserve `adata.uns['log1p']`, so the loader can't verify it from file metadata.

**Is it a problem?** No. The loader still treats data as log1p-transformed, which is correct. The warning is purely informational — it fires when `is_log1p=True` but no `uns/log1p` key is found to confirm it.

**To silence it** (optional): write the log1p uns key back into the h5 files during generation, after `sc.pp.log1p(adata)`:
```python
adata.uns["log1p"] = {"base": None}  # add before writing h5ad
```

---

## 2026-03-22 — Checkpoint loading in `submit_train_spptv1_last_stgt.sh`

**Context**: `spptv1_last_stgt` fine-tunes from `sergio_ppt_v1/checkpoints/last.ckpt` on `SERGIO_TGT_train_merged` for 25k steps.

**Does it enter the manual-init block?** Yes. The block in `src/state/_cli/_tx/_train.py` (lines 306–392) is entered when:
1. No `last.ckpt` exists in the output run dir (fresh run → `checkpoint_path = None`)
2. `model.kwargs.init_from` is set (the script passes `PPT_CKPT`)

**What happens inside the block**:
- Weights from `sergio_ppt_v1/last.ckpt` are loaded into the new model.
- **Output space**: both checkpoint and config use `output_space=gene` → no decoder rebuild.
- **Pert encoder**: `embed_key=X_hvg` is passed, so the `pert_encoder` is **rebuilt with the new `pert_dim`** and the checkpoint's pert_encoder weights are discarded.
- Shape-mismatched parameters are skipped (not loaded).
- `trainer.fit(..., ckpt_path=None)`: optimizer and scheduler state are **reset** — this is a fine-tune from weights, not a training resume.
