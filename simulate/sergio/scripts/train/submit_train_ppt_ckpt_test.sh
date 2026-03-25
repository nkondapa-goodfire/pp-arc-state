#!/usr/bin/env bash
#SBATCH --job-name=state_ckpt_test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_ckpt_test_%j.out
#SBATCH --error=slurm_logs/state_ckpt_test_%j.err

# Quick checkpoint test: 60 steps total, checkpoint every 20 steps
# Expect: step=20.ckpt, step=40.ckpt, step=60.ckpt + last.ckpt

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"
uv run state tx train \
  data.kwargs.toml_config_path="${SERGIO_DIR}/configs/sergio_ppt_train.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.perturbation_features_file="${SERGIO_DIR}/configs/pert_onehot_map_ppt.pt" \
  data.kwargs.num_workers=12 \
  data.kwargs.pin_memory=true \
  model=replogle \
  training=replogle \
  training.max_steps=60 \
  training.val_freq=20 \
  training.ckpt_every_n_steps=20 \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=sergio_ppt_ckpt_test \
  use_wandb=false
