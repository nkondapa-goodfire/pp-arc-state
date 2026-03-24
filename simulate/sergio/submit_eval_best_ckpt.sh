#!/usr/bin/env bash
# Submit eval jobs for best.ckpt for nca_rpnd, rpnd_baseline, spptv1_rpnd.
# Usage: bash submit_eval_best_ckpt.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"
TEST_TOML="${SERGIO_DIR}/configs/rpnd_train.toml"
WORKER="${SERGIO_DIR}/eval_best_worker.sh"

cd "${SERGIO_DIR}"

for MODEL in nca_rpnd rpnd_baseline spptv1_rpnd; do
    JOB=$(sbatch \
        --export="RUN_DIR=${STATE_RUNS}/${MODEL},TEST_TOML=${TEST_TOML},COMPLETION_MARKER=k562_agg_results.csv" \
        --job-name="eval_best_${MODEL}" \
        "${WORKER}" | awk '{print $4}')
    echo "${MODEL} best.ckpt eval: ${JOB}"
done
