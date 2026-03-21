#!/usr/bin/env bash
#SBATCH --job-name=state_ablation
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/ablation_%x_%j.out
#SBATCH --error=slurm_logs/ablation_%x_%j.err

# Usage: sbatch --job-name=<run_name> submit_train_ablation.sh <run_name>
#   e.g. sbatch --job-name=ablation_ba_only submit_train_ablation.sh ablation_ba_only

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch submit_train_ablation.sh <run_name>}"

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
TOML="${SERGIO_DIR}/configs/ablations/${RUN_NAME}.toml"

if [[ ! -f "${TOML}" ]]; then
    echo "ERROR: TOML not found: ${TOML}"
    echo "Run: uv run python scripts/build_ablation_dirs.py first"
    exit 1
fi

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sergio_ablations
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Starting ablation run: ${RUN_NAME}"
echo "TOML: ${TOML}"

cd "${STATE_DIR}"
uv run state tx train \
  data.kwargs.toml_config_path="${TOML}" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.perturbation_features_file="${SERGIO_DIR}/configs/pert_onehot_map.pt" \
  data.kwargs.num_workers=12 \
  data.kwargs.pin_memory=true \
  model=replogle \
  training=replogle \
  training.devices=8 \
  training.strategy=ddp \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name="${RUN_NAME}" \
  wandb.entity=goodfire \
  use_wandb=true
