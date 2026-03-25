#!/usr/bin/env bash
# Submit eval array job for spptv2_last_stgt.
# Evaluates checkpoints at steps 200-8000 (15 checkpoints, indices 0-14).
# Usage: bash submit_eval_spptv2_last_stgt.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/sergio_tgt_test.toml"
WORKER="${SERGIO_DIR}/scripts/eval/eval_array_worker_spptv2_stgt.sh"
ARRAY="0-14"  # 15 checkpoints, indices 0-14

cd "${SERGIO_DIR}"

JOB=$(sbatch --array="${ARRAY}" \
    --export="RUN_DIR=${STATE_RUNS}/spptv2_last_stgt,TEST_TOML=${TEST_TOML}" \
    "${WORKER}" | awk '{print $4}')
echo "spptv2_last_stgt eval: ${JOB}"
