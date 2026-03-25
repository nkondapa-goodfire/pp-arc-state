#!/usr/bin/env bash
#SBATCH --job-name=state_train_pptcte
#SBATCH --time=02:40:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_pptcte_%j.out
#SBATCH --error=slurm_logs/state_train_pptcte_%j.err

# sergio_pptcte: SERGIO_PPT pre-pre-training with cell-type encoder enabled.
# Identical to sergio_ppt_v2 except model.kwargs.cell_type_encoder=true.

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sergio_pptcte
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
  model.kwargs.cell_type_encoder=true \
  training=replogle \
  training.max_steps=20000 \
  training.val_freq=2000 \
  training.ckpt_every_n_steps=2000 \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=sergio_pptcte \
  wandb.entity=goodfire \
  use_wandb=true
