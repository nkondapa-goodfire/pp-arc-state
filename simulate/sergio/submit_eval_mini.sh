#!/usr/bin/env bash
#SBATCH --job-name=state_eval_mini
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_eval_mini_%A_%a.out
#SBATCH --error=slurm_logs/state_eval_mini_%A_%a.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
MODEL_DIR="/mnt/polished-lake/home/nkondapaneni/state_runs/sergio_mini_replogle_config_8gpu"
TEST_MERGED_DIR="${SERGIO_DIR}/data/sergio_synthetic/test_mini_merged"
EVAL_DIR="${MODEL_DIR}/eval"
FILELIST="${SERGIO_DIR}/configs/test_mini_filelist.txt"

mkdir -p "${EVAL_DIR}" "${SERGIO_DIR}/slurm_logs"

# Each array task processes one test h5ad file
ADATA=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${FILELIST}")
BASENAME=$(basename "${ADATA}" .h5ad)
OUTPUT="${EVAL_DIR}/${BASENAME}_simulated.h5ad"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${ADATA}"

cd "${STATE_DIR}"
uv run state tx infer \
  --model-dir "${MODEL_DIR}" \
  --adata "${ADATA}" \
  --embed-key X_hvg \
  --pert-col gene \
  --celltype-col cell_type \
  --batch-col gem_group \
  --control-pert non-targeting \
  --output "${OUTPUT}"

# Add pert_type column derived from obs["gene"]
cd "${STATE_DIR}"
uv run python - "${OUTPUT}" <<'EOF'
import sys, anndata

path = sys.argv[1]
ad = anndata.read_h5ad(path)

def parse_pert_type(gene):
    if gene == "non-targeting":
        return "non-targeting"
    parts = gene.split("_")   # SYN_0033_KD_010 -> ['SYN', '0033', 'KD', '010']
    return "_".join(parts[2:])

ad.obs["pert_type"] = ad.obs["gene"].map(parse_pert_type)
ad.write_h5ad(path)
EOF

echo "Done: ${OUTPUT}"
