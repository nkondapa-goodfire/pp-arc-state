#!/usr/bin/env bash
#SBATCH --job-name=eval_worker
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/eval_worker_%A_%a.out
#SBATCH --error=slurm_logs/eval_worker_%A_%a.err

# Generic array worker — evaluates one checkpoint per task.
# Required env vars (passed via --export):
#   RUN_DIR          path to the model's state_runs directory
#   TEST_TOML        path to the eval TOML config
#   COMPLETION_MARKER  filename that indicates eval is complete (e.g. k562_agg_results.csv)

set -euo pipefail

CHECKPOINTS=(
    step=step=200.ckpt
    step=step=400.ckpt
    step=step=600.ckpt
    step=step=800.ckpt
    step=step=1000.ckpt
    step=step=1200.ckpt
    step=step=1400.ckpt
    step=step=2000.ckpt
    step=step=2200.ckpt
    step=step=2400.ckpt
    step=step=2600.ckpt
    step=step=2800.ckpt
    step=step=3000.ckpt
    step=step=8000.ckpt
    step=step=20000.ckpt
    step=step=30000.ckpt
    step=step=40000.ckpt
    step=step=50000.ckpt
)

CKPT_NAME="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT_NAME}"
EVAL_DIR="${RUN_DIR}/eval_${CKPT_NAME}"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CKPT_NAME} for $(basename ${RUN_DIR})"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "Checkpoint not found: ${CKPT_PATH}, skipping."
    exit 0
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
