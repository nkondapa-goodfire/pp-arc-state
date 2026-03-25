#!/usr/bin/env bash
#SBATCH --job-name=state_eval_tgt
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_eval_tgt_%j.out
#SBATCH --error=slurm_logs/state_eval_tgt_%j.err

# Evaluate sergio_tgt, spptv1_last_stgt, nca_stgt up to step=8000
# Uses sergio_tgt_test.toml (SERGIO_TGT_test_merged, grn_0100-0103 zeroshot)

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/sergio_tgt_test.toml"
MAX_STEP=8000

eval_run() {
    local RUN_DIR="$1"
    echo "=== Evaluating run: $(basename ${RUN_DIR}) ==="
    for CKPT in "${RUN_DIR}"/checkpoints/step=step=*.ckpt; do
        CKPT_NAME=$(basename "${CKPT}")
        STEP=$(echo "${CKPT_NAME}" | grep -oP '(?<=step=step=)\d+')
        if [ "${STEP}" -gt "${MAX_STEP}" ]; then
            echo "Skipping ${CKPT_NAME} (step ${STEP} > ${MAX_STEP})"
            continue
        fi
        EVAL_DIR="${RUN_DIR}/eval_${CKPT_NAME}"
        if [ -d "${EVAL_DIR}" ] && [ -f "${EVAL_DIR}/metrics.csv" ]; then
            echo "Skipping ${CKPT_NAME} (already evaluated)"
            continue
        fi
        echo "Evaluating ${CKPT_NAME}..."
        uv run state tx predict \
            --output-dir "${RUN_DIR}" \
            --checkpoint "${CKPT_NAME}" \
            --toml "${TEST_TOML}" \
            --profile minimal
    done
}

cd "${STATE_DIR}"
eval_run "${STATE_RUNS}/sergio_tgt"
eval_run "${STATE_RUNS}/spptv1_last_stgt"
eval_run "${STATE_RUNS}/nca_stgt"

echo "All done."
