#!/usr/bin/env bash
#SBATCH --job-name=sergio_tgt
#SBATCH --array=0-107
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/sergio_tgt_%A_%a.out
#SBATCH --error=slurm_logs/sergio_tgt_%A_%a.err

# SERGIO_TGT dataset: 4 seeds x 10 bins x 9 GRN combos x 3 noise = 108 tasks
# Seeds 100-103 (disjoint from SERGIO_PPT seeds 0-3)
# Spec: plan3.md Section 2.1

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="${SCRIPT_DIR}/generation_configs/dataset_tgt.json"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

cd "${SCRIPT_DIR}"
uv run python generate_dataset.py \
    --config  "${CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
