#!/usr/bin/env bash
# Generate filelist and submit SLURM array for test_mini_merged evaluation.
# Usage: bash run_eval_mini.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
TEST_MERGED_DIR="${SERGIO_DIR}/data/sergio_synthetic/test_mini_merged"
FILELIST="${SERGIO_DIR}/configs/test_mini_filelist.txt"

mkdir -p "${SERGIO_DIR}/configs"

# Build filelist from test_mini_merged
ls "${TEST_MERGED_DIR}"/*.h5ad > "${FILELIST}"
N=$(wc -l < "${FILELIST}")
echo "Found ${N} test files. Submitting array job 0-$((N - 1))."

sbatch --array="0-$((N - 1))%8" "${SERGIO_DIR}/submit_eval_mini.sh"
