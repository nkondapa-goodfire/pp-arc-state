#!/usr/bin/env bash
# Submit eval array for nca_stgt_v2 (NCA backbone fine-tuned on SERGIO_TGT, 25k steps).
# Chains after training job 381021 (state_train_nca_stgt_v2).
# Usage: bash submit_eval_nca_stgt_v2.sh [--after <job_id>]
#
# By default chains after job 381021. Pass --after <id> to override.

set -euo pipefail

AFTER_JOB=381021

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
RUN_DIR=/mnt/polished-lake/home/nkondapaneni/state_runs/nca_stgt_v2,\
TEST_TOML=${SERGIO_DIR}/configs/sergio_tgt_test.toml,\
COMPLETION_MARKER=ER_size100_seed0103_agg_results.csv \
    "${SERGIO_DIR}/scripts/eval/eval_array_worker_stgt_v2.sh"
