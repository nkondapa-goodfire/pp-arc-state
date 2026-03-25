#!/usr/bin/env bash
#SBATCH --job-name=eval_best
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/eval_best_%j.out
#SBATCH --error=slurm_logs/eval_best_%j.err

# Evaluates best.ckpt for a given run.
# Required env vars (passed via --export):
#   RUN_DIR          path to the model's state_runs directory
#   TEST_TOML        path to the eval TOML config
#   COMPLETION_MARKER  filename that indicates eval is complete

set -euo pipefail

CKPT_NAME="best.ckpt"
CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT_NAME}"
EVAL_DIR="${RUN_DIR}/eval_${CKPT_NAME}"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

echo "Evaluating ${CKPT_NAME} for $(basename ${RUN_DIR})"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "Checkpoint not found: ${CKPT_PATH}, exiting."
    exit 1
fi

if [ -f "${EVAL_DIR}/${COMPLETION_MARKER}" ]; then
    echo "Already evaluated, skipping."
    exit 0
fi

cd "${STATE_DIR}"
uv run state tx predict \
    --output-dir "${RUN_DIR}" \
    --checkpoint "${CKPT_NAME}" \
    --toml "${TEST_TOML}" \
    --profile minimal

echo "Done: ${CKPT_NAME}"
