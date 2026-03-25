#!/usr/bin/env bash
#SBATCH --job-name=eval_sample_eff_nca
#SBATCH --array=0-4
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/eval_sample_eff_nca_%A_%a.out
#SBATCH --error=slurm_logs/eval_sample_eff_nca_%A_%a.err

# Evaluate sample efficiency runs at step=8000 for the nca condition.
# Task layout:
#   0: sample_eff_nca_cpp10
#   1: sample_eff_nca_cpp25
#   2: sample_eff_nca_cpp50
#   3: sample_eff_nca_cpp100
#   4: nca_stgt_v2  (full data)

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/sergio_tgt_test.toml"
CKPT="step=step=8000.ckpt"
COMPLETION_MARKER="ER_size100_seed0103_agg_results.csv"

RUN_DIRS=(
    "${STATE_RUNS}/sample_eff_nca_cpp10"
    "${STATE_RUNS}/sample_eff_nca_cpp25"
    "${STATE_RUNS}/sample_eff_nca_cpp50"
    "${STATE_RUNS}/sample_eff_nca_cpp100"
    "${STATE_RUNS}/nca_stgt_v2"
)

RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT}"
EVAL_DIR="${RUN_DIR}/eval_${CKPT}"

echo "Task ${SLURM_ARRAY_TASK_ID}: $(basename ${RUN_DIR}) @ ${CKPT}"

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
    --checkpoint "${CKPT}" \
    --toml "${TEST_TOML}" \
    --profile minimal

echo "Done: $(basename ${RUN_DIR}) @ ${CKPT}"
