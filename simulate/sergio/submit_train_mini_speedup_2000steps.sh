#!/usr/bin/env bash
# Speedup comparison run: same 2000 steps / val_freq=200 as the original
# sergio_mini_replogle_config_8gpu run, but with all speedup changes applied:
#   - embed_key=X_hvg (dense obsm instead of sparse X)
#   - pin_memory=true
#   - num_workers=12
#   - strategy=ddp (not ddp_find_unused_parameters_true)
#   - output_space=gene
#SBATCH --job-name=state_train_mini_speedup
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_mini_speedup_%j.out
#SBATCH --error=slurm_logs/state_train_mini_speedup_%j.err

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sergio_simulations
export WANDB_START_METHOD=thread

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
  training.max_steps=2000 \
  training.val_freq=200 \
  training.devices=8 \
  training.strategy=ddp \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=sergio_mini_speedup_2000steps \
  wandb.entity=goodfire \
  use_wandb=true
