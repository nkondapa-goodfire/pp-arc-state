#!/usr/bin/env bash
#SBATCH --job-name=sergio_ppt
#SBATCH --array=0-107
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/sergio_ppt_%A_%a.out
#SBATCH --error=slurm_logs/sergio_ppt_%A_%a.err

# SERGIO_PPT dataset: 4 seeds x 5 bins x 9 GRN combos x 3 noise = 108 tasks
# Spec: plan3.md Section 1

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="${SCRIPT_DIR}/generation_configs/dataset_ppt_v2.json"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

cd "${SCRIPT_DIR}"
uv run python generate_dataset.py \
    --config  "${CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
