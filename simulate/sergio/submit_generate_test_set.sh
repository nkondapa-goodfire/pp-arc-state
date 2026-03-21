#!/usr/bin/env bash
#SBATCH --job-name=sergio_test
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/sergio_test_%A_%a.out
#SBATCH --error=slurm_logs/sergio_test_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="${SCRIPT_DIR}/generation_configs/test_set.json"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

cd "${SCRIPT_DIR}"
uv run python generate_test_set.py \
    --config  "${CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
