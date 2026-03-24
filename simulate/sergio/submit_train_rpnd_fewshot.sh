#!/usr/bin/env bash
#SBATCH --job-name=state_train_rpnd_fewshot
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_rpnd_fewshot_%j.out
#SBATCH --error=slurm_logs/state_train_rpnd_fewshot_%j.err

# rpnd_fewshot: fewshot baseline on Replogle-Nadig dataset, 50k steps, no pre-pre-training
# k562 is split: 858 held-out perturbations for test, remaining perturbations used in training.
# Architecture: hidden_dim=328, 8 layers, 12 heads, cell_set_len=64 (model=replogle)
# Data: embed_key=X_hvg, output_space=gene, pert_col=gene, cell_type_key=cell_line,
#       batch_col=gem_group, control_pert=non-targeting, perturbation_features_file=null

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=rpnd_fewshot
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"
uv run state tx train \
  data.kwargs.toml_config_path="${SERGIO_DIR}/configs/rpnd_fewshot.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_line \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.num_workers=12 \
  data.kwargs.pin_memory=true \
  model=replogle \
  training=replogle \
  training.max_steps=50000 \
  training.val_freq=2000 \
  "training.ckpt_steps=[200,400,600,800,1000,1200,1400,2000,2200,2400,2600,2800,3000,4000,8000,14000,20000,25000,30000,35000,40000,45000,50000]" \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=rpnd_fewshot \
  wandb.entity=goodfire \
  use_wandb=true
