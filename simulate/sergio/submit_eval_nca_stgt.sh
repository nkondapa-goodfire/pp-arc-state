#!/usr/bin/env bash
#SBATCH --job-name=eval_nca_stgt
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/eval_nca_stgt_%j.out
#SBATCH --error=slurm_logs/eval_nca_stgt_%j.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
RUN_DIR="/mnt/polished-lake/home/nkondapaneni/state_runs/nca_stgt"
TEST_TOML="${SERGIO_DIR}/configs/sergio_tgt_test.toml"
MAX_STEP=8000

cd "${STATE_DIR}"

for CKPT in $(ls "${RUN_DIR}"/checkpoints/step=step=*.ckpt | sort -t= -k3 -n); do
    CKPT_NAME=$(basename "${CKPT}")
    STEP=$(echo "${CKPT_NAME}" | grep -oP '(?<=step=step=)\d+')
    if [ "${STEP}" -gt "${MAX_STEP}" ]; then
        echo "Skipping ${CKPT_NAME} (step ${STEP} > ${MAX_STEP})"
        continue
    fi
    EVAL_DIR="${RUN_DIR}/eval_${CKPT_NAME}"
    if [ -f "${EVAL_DIR}/grn_0103_agg_results.csv" ]; then
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

echo "Done: nca_stgt"
