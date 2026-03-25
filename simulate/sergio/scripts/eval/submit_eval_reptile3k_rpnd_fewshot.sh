#!/usr/bin/env bash
# Submit eval array job for reptile3k_rpnd_fewshot.
# Runs after training job 386152 finishes (afterany dependency).
# Usage: bash submit_eval_reptile3k_rpnd_fewshot.sh
#
# The worker skips checkpoints that don't exist yet, so this is safe to
# submit while training is still in progress — completed checkpoints are
# evaluated immediately and missing ones are skipped gracefully.

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/rpnd_fewshot.toml"
WORKER="${SERGIO_DIR}/scripts/eval/eval_array_worker_fewshot.sh"
COMPLETION_MARKER="k562_agg_results.csv"
ARRAY="0-22"  # 23 checkpoints, indices 0-22

cd "${SERGIO_DIR}"

JOB=$(sbatch --array="${ARRAY}" \
    --export="RUN_DIR=${STATE_RUNS}/reptile3k_rpnd_fewshot,TEST_TOML=${TEST_TOML},COMPLETION_MARKER=${COMPLETION_MARKER}" \
    --job-name="eval_reptile3k_rpnd_fewshot" \
    "${WORKER}" | awk '{print $4}')
echo "reptile3k_rpnd_fewshot eval: ${JOB}"
