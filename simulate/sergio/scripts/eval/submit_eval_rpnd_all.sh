#!/usr/bin/env bash
# Submit eval array jobs for rpnd_baseline, spptv1_rpnd, nca_rpnd.
# Each eval job starts after its own training job finishes.
# Training job IDs: rpnd_baseline=375329, spptv1_rpnd=375330, nca_rpnd=375331
# Usage: bash submit_eval_rpnd_all.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/rpnd_train.toml"
WORKER="${SERGIO_DIR}/scripts/eval/eval_array_worker.sh"
COMPLETION_MARKER="k562_agg_results.csv"
ARRAY="0-17"  # 18 checkpoints, indices 0-17

cd "${SERGIO_DIR}"

JOB1=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:375329" \
    --export="RUN_DIR=${STATE_RUNS}/rpnd_baseline,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_rpnd_baseline" \
    "${WORKER}" | awk '{print $4}')
echo "rpnd_baseline eval: ${JOB1} (after train 375329)"

JOB2=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:375330" \
    --export="RUN_DIR=${STATE_RUNS}/spptv1_rpnd,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_spptv1_rpnd" \
    "${WORKER}" | awk '{print $4}')
echo "spptv1_rpnd eval:   ${JOB2} (after train 375330)"

JOB3=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:375331" \
    --export="RUN_DIR=${STATE_RUNS}/nca_rpnd,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_nca_rpnd" \
    "${WORKER}" | awk '{print $4}')
echo "nca_rpnd eval:      ${JOB3} (after train 375331)"
