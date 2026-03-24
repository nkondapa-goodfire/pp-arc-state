#!/usr/bin/env bash
#SBATCH --job-name=build_merged_tgt
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/build_merged_tgt_%j.out
#SBATCH --error=slurm_logs/build_merged_tgt_%j.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"

mkdir -p "${SERGIO_DIR}/slurm_logs"

cd "${SERGIO_DIR}"

# Train split: bins 0–4
uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/SERGIO_TGT \
    --dst data/sergio_synthetic/SERGIO_TGT_train_merged \
    --bins bin_0,bin_1,bin_2,bin_3,bin_4

# Test split: bins 5–9
uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/SERGIO_TGT \
    --dst data/sergio_synthetic/SERGIO_TGT_test_merged \
    --bins bin_5,bin_6,bin_7,bin_8,bin_9
