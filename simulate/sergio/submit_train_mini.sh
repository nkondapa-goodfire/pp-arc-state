#!/usr/bin/env bash
#SBATCH --job-name=state_train_mini
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_mini_%j.out
#SBATCH --error=slurm_logs/state_train_mini_%j.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sergio_simulations
export WANDB_START_METHOD=thread

# Single SLURM task; Lightning spawns one DDP process per GPU internally
unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"
uv run state tx train \
  data.kwargs.toml_config_path="${SERGIO_DIR}/configs/sergio_mini_train.toml" \
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
  name=sergio_mini_replogle_config_8gpu \
  wandb.entity=goodfire \
  use_wandb=true
