#!/usr/bin/env bash
#SBATCH --job-name=build_merged_ppt_v2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/build_merged_ppt_v2_%j.out
#SBATCH --error=slurm_logs/build_merged_ppt_v2_%j.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"

mkdir -p "${SERGIO_DIR}/slurm_logs"

cd "${SERGIO_DIR}"

uv run python scripts/build_merged_h5ads.py \
    --src data/sergio_synthetic/SERGIO_PPTD2 \
    --dst data/sergio_synthetic/SERGIO_PPTD2_merged
