#!/usr/bin/env bash
# Submit eval array for sergio_tgtv2 (SERGIO_TGT scratch, 25k steps).
# Chains after training job 381020 (state_train_sergio_tgtv2).
# Usage: bash submit_eval_sergio_tgtv2.sh [--after <job_id>]
#
# By default chains after job 381020. Pass --after <id> to override.

set -euo pipefail

AFTER_JOB=381020

while [[ $# -gt 0 ]]; do
    case "$1" in
        --after) AFTER_JOB="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"

mkdir -p "${SERGIO_DIR}/slurm_logs"

sbatch \
    --array=0-17 \
    --dependency=afterok:${AFTER_JOB} \
    --export=ALL,\
RUN_DIR=/mnt/polished-lake/home/nkondapaneni/state_runs/sergio_tgtv2,\
TEST_TOML=${SERGIO_DIR}/configs/sergio_tgt_test.toml,\
COMPLETION_MARKER=ER_size100_seed0103_agg_results.csv \
    "${SERGIO_DIR}/scripts/eval/eval_array_worker_stgt_v2.sh"
