#!/usr/bin/env bash
#SBATCH --job-name=sample_efficiency
#SBATCH --array=0-11
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/sample_efficiency_%A_%a.out
#SBATCH --error=slurm_logs/sample_efficiency_%A_%a.err

# Sample efficiency sweep: 3 conditions x 4 cells_per_pert = 12 array tasks.
#
# Task index layout (condition_idx * 4 + cpp_idx):
#   0-3:  spptv2  (finetune from sergio_ppt_v2 last ckpt)
#   4-7:  scratch (train from scratch)
#   8-11: nca     (finetune from nca_ppt backbone)
#
#   cpp_idx 0-3 -> cells_per_pert 10, 25, 50, 100
#
# Usage: sbatch sample_efficiency_worker.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"

SPPTV2_CKPT="${STATE_RUNS}/sergio_ppt_v2/checkpoints/last.ckpt"
NCA_CKPT="${STATE_RUNS}/nca_ppt/nca_ppt_backbone_init.pt"

CONDITIONS=(spptv2 scratch nca)
INIT_CKPTS=("${SPPTV2_CKPT}" "" "${NCA_CKPT}")
CELLS_PER_PERT_VALUES=(10 25 50 100)

CONDITION_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
CPP_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

CONDITION="${CONDITIONS[$CONDITION_IDX]}"
INIT_CKPT="${INIT_CKPTS[$CONDITION_IDX]}"
CELLS_PER_PERT="${CELLS_PER_PERT_VALUES[$CPP_IDX]}"
RUN_NAME="sample_eff_${CONDITION}_cpp${CELLS_PER_PERT}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (cells_per_pert=${CELLS_PER_PERT}, init=${INIT_CKPT:-scratch})"

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sample_efficiency
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

INIT_OVERRIDE=""
if [ -n "${INIT_CKPT}" ]; then
    INIT_OVERRIDE="model.kwargs.init_from=${INIT_CKPT}"
fi

cd "${STATE_DIR}"
uv run state tx train \
  data.name=BalancedPerturbationDataModule \
  data.kwargs.toml_config_path="${SERGIO_DIR}/configs/sergio_tgt_train.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.perturbation_features_file="${SERGIO_DIR}/configs/pert_onehot_map_tgt.pt" \
  +data.kwargs.cells_per_pert="${CELLS_PER_PERT}" \
  data.kwargs.num_workers=12 \
  data.kwargs.pin_memory=true \
  model=replogle \
  ${INIT_OVERRIDE} \
  training=replogle \
  training.max_steps=8000 \
  training.val_freq=1000 \
  "training.ckpt_steps=[200,400,600,800,1000,2000,4000,8000]" \
  training.devices=8 \
  training.strategy=ddp_find_unused_parameters_true \
  output_dir=/mnt/polished-lake/home/nkondapaneni/state_runs \
  name="${RUN_NAME}" \
  wandb.entity=goodfire \
  use_wandb=true
