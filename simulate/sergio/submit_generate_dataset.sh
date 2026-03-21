#!/usr/bin/env bash
# submit_generate_dataset.sh — Submit SERGIO dataset generation as a SLURM array job.
#
# Each array task maps to one (grn_type, grn_size, noise_label, seed) combination
# via --task-id, which generate_dataset.py looks up from the config.
#
# Usage:
#   sbatch submit_generate_dataset.sh
#   sbatch --array=0-9 submit_generate_dataset.sh   # subset for testing

#SBATCH --job-name=sergio_gen
#SBATCH --array=0-3299
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/sergio_gen_%A_%a.out
#SBATCH --error=slurm_logs/sergio_gen_%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="${SCRIPT_DIR}/generation_configs/dataset1.json"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

cd "${SCRIPT_DIR}"
uv run python generate_dataset.py \
    --config  "${CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
