#!/usr/bin/env bash
#SBATCH --job-name=sergio_mini
#SBATCH --array=0-674
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/sergio_mini_%A_%a.out
#SBATCH --error=slurm_logs/sergio_mini_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="${SCRIPT_DIR}/generation_configs/dataset_mini.json"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

cd "${SCRIPT_DIR}"
uv run python generate_dataset.py \
    --config  "${CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
