# NK Work Trial — Entry Points

Notes and scripts for SERGIO synthetic pre-pre-training of the State ST model. Detailed docs in `mds/`.

---

## Data Generation

**Config**: `simulate/sergio/generation_configs/dataset_ppt.json`
- 3 GRN types (ER, BA, BA-VM) × 3 sizes (10, 50, 100) × 3 noise levels × 4 seeds = 108 SLURM tasks
- 7 perturbations (top by out-degree) × 3 pert types (KO, KD_020, KD_060) per GRN = 2,268 h5ads

```bash
# Generate PPT dataset (108-task array)
sbatch simulate/sergio/scripts/generate/submit_generate_dataset_ppt.sh

# Merge into condition-grouped h5ads + add obsm["X_hvg"]
sbatch simulate/sergio/scripts/merge/submit_build_merged_ppt.sh
```

Output: `simulate/sergio/data/sergio_synthetic/SERGIO_PPT_merged/`
Pert map: `simulate/sergio/configs/pert_onehot_map_ppt.pt`

---

## Standard Pre-Pre-Training (joint)

Trains on all GRN tasks mixed together (effective k=1, no meta-learning signal).

```bash
sbatch simulate/sergio/scripts/train/submit_train_ppt_v2.sh       # standard PPT
sbatch simulate/sergio/scripts/train/submit_train_pptcte.sh        # + cell-type encoder
```

Checkpoints: `state_runs/sergio_ppt_v2/`, `state_runs/sergio_pptcte/`

---

## Reptile Pre-Pre-Training

Reptile runs k inner steps on the same GRN task (varying noise/pert/bin), then does one outer update. This maximizes gradient alignment within a task (AvgGradInner), producing initializations that fine-tune faster.

| Concept | SERGIO analog |
|---------|--------------|
| Task τ | One GRN instance (`cell_type` = `{grn_type}_size{n}_seed{k}`) |
| Inner mini-batch | One `(h5ad file, gem_group bin)` draw from that GRN |
| k inner steps | k steps with fresh Adam (β₁=0), new (file, bin) each step |
| Outer update | `θ ← θ_t + ε·(θ̃ − θ_t)`, all-reduced across GPUs |

```bash
sbatch simulate/sergio/scripts/train/submit_reptile_test.sh        # Reptile PPT
sbatch simulate/sergio/scripts/train/submit_reptile_cell_type.sh   # + cell-type encoder
```

Checkpoints: `state_runs/sergio_ppt_reptile/`, `state_runs/sergio_ppt_reptile_cte/`

Pseudocode: `mds/reptile_loop_pseudocode.md`
Algorithm notes: `mds/REPTILE_NOTES.md`, `mds/REPTILE_TRAINING.md`

---

## Fine-tuning on Target Data

```bash
# Fine-tune reptile checkpoint on SERGIO target dataset
sbatch simulate/sergio/scripts/train/submit_train_reptile3k_rpnd_fewshot.sh

# Sample efficiency sweep (cells-per-pert ∈ {10,25,50,100,full})
sbatch simulate/sergio/scripts/sample_efficiency/sample_efficiency_reptile3k_worker.sh
```

`model.kwargs.init_from` loads weights only; optimizer resets for a fresh fine-tune start.

---

## Evaluation

```bash
# Eval on SERGIO target test set (array over checkpoints)
sbatch simulate/sergio/scripts/eval/submit_eval_sergio_tgtv2.sh

# Eval on Replogle-Nadig fewshot
sbatch simulate/sergio/scripts/eval/submit_eval_reptile3k_rpnd_fewshot.sh

# Sample efficiency eval
sbatch simulate/sergio/scripts/sample_efficiency/eval_sample_eff_reptile3k.sh
```

Primary metrics: `pearson_delta`, `overlap_at_N`, `discrimination_score_l1`

---

## Key Configs

| File | Purpose |
|------|---------|
| `simulate/sergio/generation_configs/dataset_ppt.json` | Dataset generation spec |
| `simulate/sergio/configs/sergio_ppt_train.toml` | PPT training TOML |
| `simulate/sergio/configs/sergio_tgt_train.toml` | Fine-tune training TOML |
| `simulate/sergio/configs/sergio_tgt_test.toml` | Test set TOML |
| `simulate/sergio/configs/pert_onehot_map_ppt.pt` | Perturbation feature map |
| `mds/model_log.md` | All runs, checkpoints, SLURM job IDs, status |



* I moved the scripts to clean up the repo, some of the paths for running models may have issues. Haven't had a chance to test all the scripts.