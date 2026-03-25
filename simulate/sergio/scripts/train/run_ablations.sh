#!/usr/bin/env bash
# Build ablation dirs and submit all training jobs.
# Usage: bash run_ablations.sh [run_name]
#   With no argument: builds dirs and submits all 9 ablations.
#   With a run_name:  builds and submits only that ablation.

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
cd "${SERGIO_DIR}"

ONLY="${1:-}"

# Step 1: build ablation symlink dirs + TOML configs
if [[ -n "${ONLY}" ]]; then
    uv run python scripts/build_ablation_dirs.py --only "${ONLY}"
else
    uv run python scripts/build_ablation_dirs.py
fi

# Step 2: submit SLURM jobs
RUNS=(
    baseline_all
    ablation_ba_only
    ablation_ba_plus_er
    ablation_kd_only
    ablation_kd_plus_ko
    ablation_noise_high
    ablation_noise_high_plus_clean
    ablation_size_100
    ablation_size_100_plus_010
)

if [[ -n "${ONLY}" ]]; then
    RUNS=("${ONLY}")
fi

for RUN_NAME in "${RUNS[@]}"; do
    JOB_ID=$(sbatch --job-name="${RUN_NAME}" "${SERGIO_DIR}/submit_train_ablation.sh" "${RUN_NAME}" | awk '{print $4}')
    echo "Submitted ${RUN_NAME} → job ${JOB_ID}"
done
