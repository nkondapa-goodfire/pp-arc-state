#!/usr/bin/env bash
# Submit eval array jobs for rpnd_fewshot, spptv1_rpnd_fewshot, nca_rpnd_fewshot.
# Each eval job starts after its own training job finishes.
# Training job IDs: rpnd_fewshot=377870, spptv1_rpnd_fewshot=378530, nca_rpnd_fewshot=378529
# Usage: bash submit_eval_rpnd_fewshot.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/rpnd_fewshot.toml"
WORKER="${SERGIO_DIR}/scripts/eval/eval_array_worker_fewshot.sh"
COMPLETION_MARKER="k562_agg_results.csv"
ARRAY="0-22"  # 23 checkpoints, indices 0-22

cd "${SERGIO_DIR}"

JOB1=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:377870" \
    --export="RUN_DIR=${STATE_RUNS}/rpnd_fewshot,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_rpnd_fewshot" \
    "${WORKER}" | awk '{print $4}')
echo "rpnd_fewshot eval:         ${JOB1} (after train 377870)"

JOB2=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:382250" \
    --export="RUN_DIR=${STATE_RUNS}/spptv2_rpnd_fewshot,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_spptv2_rpnd_fewshot" \
    "${WORKER}" | awk '{print $4}')
echo "spptv2_rpnd_fewshot eval:  ${JOB2} (after train 382250)"

JOB3=$(sbatch --array="${ARRAY}" \
    --dependency="afterany:378529" \
    --export="RUN_DIR=${STATE_RUNS}/nca_rpnd_fewshot,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_nca_rpnd_fewshot" \
    "${WORKER}" | awk '{print $4}')
echo "nca_rpnd_fewshot eval:     ${JOB3} (after train 378529)"
