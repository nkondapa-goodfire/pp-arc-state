#!/usr/bin/env bash
#SBATCH --job-name=sample_eff_reptile10k
#SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/sample_eff_reptile10k_%A_%a.out
#SBATCH --error=slurm_logs/sample_eff_reptile10k_%A_%a.err

# Sample efficiency sweep initialised from sergio_ppt_reptile step=10000 checkpoint.
#
# Task index layout:
#   0: cells_per_pert=10   (BalancedPerturbationDataModule)
#   1: cells_per_pert=25   (BalancedPerturbationDataModule)
#   2: cells_per_pert=50   (BalancedPerturbationDataModule)
#   3: cells_per_pert=100  (BalancedPerturbationDataModule)
#   4: full data           (PerturbationDataModule, no subsampling)
#
# Usage: sbatch sample_efficiency_reptile10k_worker.sh

set -euo pipefail

SERGIO_DIR="/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio"
STATE_DIR="/mnt/polished-lake/home/nkondapaneni/state"
STATE_RUNS="/mnt/polished-lake/home/nkondapaneni/state_runs"

INIT_CKPT="${STATE_RUNS}/sergio_ppt_reptile/checkpoints/step=step=10000.ckpt"

CELLS_PER_PERT_VALUES=(10 25 50 100)

mkdir -p "${SERGIO_DIR}/slurm_logs"

export WANDB_RUN_GROUP=sample_efficiency_reptile
export WANDB_START_METHOD=thread

unset SLURM_NTASKS
unset SLURM_PROCID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd "${STATE_DIR}"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 4 ]; then
    # Full data — no subsampling
    RUN_NAME="sample_eff_reptile10k_full"
    echo "Task 4: ${RUN_NAME} (full data, init=${INIT_CKPT})"
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
      "model.kwargs.init_from='${INIT_CKPT}'" \
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
else
    CELLS_PER_PERT="${CELLS_PER_PERT_VALUES[$SLURM_ARRAY_TASK_ID]}"
    RUN_NAME="sample_eff_reptile10k_cpp${CELLS_PER_PERT}"
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (cells_per_pert=${CELLS_PER_PERT}, init=${INIT_CKPT})"
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
      "model.kwargs.init_from='${INIT_CKPT}'" \
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
fi
