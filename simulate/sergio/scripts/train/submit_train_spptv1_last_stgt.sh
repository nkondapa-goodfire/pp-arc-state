#!/usr/bin/env bash
#SBATCH --job-name=state_train_spptv1_last_stgt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_spptv1_last_stgt_%j.out
#SBATCH --error=slurm_logs/state_train_spptv1_last_stgt_%j.err

# spptv1_last_stgt: fine-tune from sergio_ppt_v1 last checkpoint on SERGIO_TGT_train_merged, 25k steps
# Spec: plan3.md Section 3.2 (spptv1_last_stgt)

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
PPT_CKPT="/mnt/polished-lake/home/nkondapaneni/state_runs/sergio_ppt_v1/checkpoints/last.ckpt"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=spptv1_last_stgt
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"
uv run state tx train \
  data.kwargs.toml_config_path="${SERGIO_DIR}/configs/sergio_tgt_train.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.perturbation_features_file="${SERGIO_DIR}/configs/pert_onehot_map_tgt.pt" \
  data.kwargs.num_workers=12 \
  data.kwargs.pin_memory=true \
  model=replogle \
  model.kwargs.init_from="${PPT_CKPT}" \
  training=replogle \
  training.max_steps=25000 \
  training.val_freq=2000 \
  "training.ckpt_steps=[200,400,600,800,1000,1200,1400,2000,2200,2400,2600,2800,3000,4000,8000,14000,20000,25000]" \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=spptv1_last_stgt \
  wandb.entity=goodfire \
  use_wandb=true
