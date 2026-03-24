#!/usr/bin/env bash
#SBATCH --job-name=state_train_nca_stgt_v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/state_train_nca_stgt_v2_%j.out
#SBATCH --error=slurm_logs/state_train_nca_stgt_v2_%j.err

# nca_stgt_v2: fine-tune from nca_ppt backbone on SERGIO_TGT_train_merged, 25k steps
# Uses corrected cell_type encoding (grn_type + grn_size + grn_seed)
# Spec: plan3.md Section 3.2 (nca_stgt)
#
# Step 1: remap NCA checkpoint keys to STATE's transformer_backbone namespace (idempotent)
# Step 2: warm-start training from remapped backbone (fresh optimizer)

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
NCA_CKPT="/mnt/polished-lake/home/nkondapaneni/nca-pre-pretraining/checkpoints/model_100.pth"
REMAPPED_CKPT="/mnt/polished-lake/home/nkondapaneni/state_runs/nca_ppt/nca_ppt_backbone_init.pt"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=nca_stgt_v2
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"

# Step 1: remap NCA checkpoint → STATE backbone namespace (skipped if already done)
if [ ! -f "${REMAPPED_CKPT}" ]; then
  echo "Remapping NCA checkpoint..."
  uv run python -m state.tx.models.nca_ckpt_remap \
      --nca_ckpt "${NCA_CKPT}" \
      --out      "${REMAPPED_CKPT}" \
      --verify
else
  echo "Remapped checkpoint already exists at ${REMAPPED_CKPT}, skipping remap."
fi

# Step 2: fine-tune on SERGIO_TGT_train_merged, init from remapped backbone
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
  model.kwargs.init_from="${REMAPPED_CKPT}" \
  training=replogle \
  training.max_steps=25000 \
  training.val_freq=2000 \
  "training.ckpt_steps=[200,400,600,800,1000,1200,1400,2000,2200,2400,2600,2800,3000,4000,8000,14000,20000,25000]" \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name=nca_stgt_v2 \
  wandb.entity=goodfire \
  use_wandb=true
