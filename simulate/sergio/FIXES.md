# Dataset Fixes

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
