## Pre-pre-training & Baselines


| Model                    | Description                                                           | SLURM                                     | Checkpoint                                                                                                                         | Status    |
| ------------------------ | --------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `sergio_ppt_v2`          | Pre-pre-training on steady-state data from synthetic GRNs             | `sbatch submit_train_ppt_v2.sh`           | `state_runs/sergio_ppt_v2/checkpoints/last.ckpt`                                                                                   | done      |
| `sergio_pptcte`          | Pre-pre-training with cell-type encoder                               | `sbatch submit_train_pptcte.sh`           | `state_runs/sergio_pptcte/checkpoints/last.ckpt`                                                                                   | done      |
| `sergio_ppt_reptile`     | Reptile pre-pre-training on synthetic GRNs                            | `sbatch submit_reptile_test.sh`           | `state_runs/sergio_ppt_reptile/checkpoints/last.ckpt`                                                                              | done      |
| `sergio_ppt_reptile_cte` | Reptile pre-pre-training with cell-type encoder                       | `sbatch submit_reptile_cell_type.sh`      | `state_runs/sergio_ppt_reptile_cte/checkpoints/last.ckpt`                                                                          | done      |
| `nca_ppt`                | Pre-pre-training on NCA data                                          | `sbatch nca_prepretraining.sh`            | `nca-pre-pretraining/checkpoints/model_100.pth` → `state_runs/nca_ppt/nca_ppt_backbone_init.pt` (remapped via `nca_ckpt_remap.py`) | done      |
| `sergio_tgtv2`           | Baseline — no PPT, trained directly on target data                    | `sbatch submit_train_tgtv2.sh`            | `state_runs/sergio_tgtv2/checkpoints/last.ckpt`                                                                                    | done      |
| `sergio_tgtv2_cte`       | Baseline — no PPT, cell-type encoder, trained directly on target data | `sbatch submit_train_sergio_tgtv2_cte.sh` | `state_runs/sergio_tgtv2_cte/checkpoints/last.ckpt`                                                                                | scheduled |


## Fine-tuning on Target Data (sergio_tgt)


| Model              | Description                                           | SLURM                                     | Checkpoint                                          | Status |
| ------------------ | ----------------------------------------------------- | ----------------------------------------- | --------------------------------------------------- | ------ |
| `spptv2_last_stgt` | `sergio_ppt_v2` (last ckpt) finetuned on `sergio_tgt` | `sbatch submit_train_spptv2_last_stgt.sh` | `state_runs/spptv2_last_stgt/checkpoints/last.ckpt` | done   |
| `nca_stgt_v2`      | `nca_ppt` finetuned on `sergio_tgt`                   | `sbatch submit_train_nca_stgt_v2.sh`      | `state_runs/nca_stgt_v2/checkpoints/last.ckpt`      | done   |


## Sample Efficiency

Array job sweeping cells-per-perturbation ∈ {10, 25, 50, 100} × 3 init conditions at 8k steps.
Submitted via `sbatch sample_efficiency_worker.sh` (`--array=0-11%2`).


| Model                       | Description                               | Checkpoint                                                             | Status |
| --------------------------- | ----------------------------------------- | ---------------------------------------------------------------------- | ------ |
| `sample_eff_spptv2_cpp{N}`  | `sergio_ppt_v2` finetuned on N cells/pert | `state_runs/sample_eff_spptv2_cpp{N}/checkpoints/step=step=8000.ckpt`  | done   |
| `sample_eff_scratch_cpp{N}` | Trained from scratch on N cells/pert      | `state_runs/sample_eff_scratch_cpp{N}/checkpoints/step=step=8000.ckpt` | done   |
| `sample_eff_nca_cpp{N}`     | `nca_ppt` finetuned on N cells/pert       | `state_runs/sample_eff_nca_cpp{N}/checkpoints/step=step=8000.ckpt`     | done   |


Reptile backbones evaluated at 1k, 3k, and 10k steps as init for the same sweep.
cells-per-perturbation ∈ {10, 25, 50, 100, full}.


| Model                          | Description                                               | Checkpoint                                                                | Status |
| ------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------- | ------ |
| `sample_eff_reptile1k_cpp{N}`  | `sergio_ppt_reptile` (ckpt 1k) finetuned on N cells/pert  | `state_runs/sample_eff_reptile1k_cpp{N}/checkpoints/step=step=8000.ckpt`  | done   |
| `sample_eff_reptile3k_cpp{N}`  | `sergio_ppt_reptile` (ckpt 3k) finetuned on N cells/pert  | `state_runs/sample_eff_reptile3k_cpp{N}/checkpoints/step=step=8000.ckpt`  | done   |
| `sample_eff_reptile10k_cpp{N}` | `sergio_ppt_reptile` (ckpt 10k) finetuned on N cells/pert | `state_runs/sample_eff_reptile10k_cpp{N}/checkpoints/step=step=8000.ckpt` | done   |


## Sample Efficiency — Eval @ step=8000

Each condition is a single array job (`--array=0-4`) evaluating `step=step=8000.ckpt` across
5 run dirs: cpp ∈ {10, 25, 50, 100} + full. The full-data run dirs for spptv2/scratch/nca
point to their respective training runs.


| Condition   | Script                          | Run dirs                                                                | Status           |
| ----------- | ------------------------------- | ----------------------------------------------------------------------- | ---------------- |
| spptv2      | `eval_sample_eff_spptv2.sh`     | `sample_eff_spptv2_cpp{10,25,50,100}`, `spptv2_last_stgt`               | done             |
| scratch     | `eval_sample_eff_scratch.sh`    | `sample_eff_scratch_cpp{10,25,50,100}`, `sergio_tgtv2`                  | done             |
| nca         | `eval_sample_eff_nca.sh`        | `sample_eff_nca_cpp{10,25,50,100}`, `nca_stgt_v2`                       | done             |
| reptile 1k  | `eval_sample_eff_reptile1k.sh`  | `sample_eff_reptile1k_cpp{10,25,50,100}`, `sample_eff_reptile1k_full`   | done             |
| reptile 3k  | `eval_sample_eff_reptile3k.sh`  | `sample_eff_reptile3k_cpp{10,25,50,100}`, `sample_eff_reptile3k_full`   | done             |
| reptile 10k | `eval_sample_eff_reptile10k.sh` | `sample_eff_reptile10k_cpp{10,25,50,100}`, `sample_eff_reptile10k_full` | done             |


## Sample Efficiency — Eval @ step=2000

Each condition is a single array job (`--array=0-4`) evaluating `step=step=2000.ckpt` across
5 run dirs: cpp ∈ {10, 25, 50, 100} + full.


| Condition   | Script                             | Run dirs                                                                | Status |
| ----------- | ---------------------------------- | ----------------------------------------------------------------------- | ------ |
| spptv2      | `eval_sample_eff_spptv2_2k.sh`     | `sample_eff_spptv2_cpp{10,25,50,100}`, `spptv2_last_stgt`               | done   |
| scratch     | `eval_sample_eff_scratch_2k.sh`    | `sample_eff_scratch_cpp{10,25,50,100}`, `sergio_tgtv2`                  | done   |
| nca         | `eval_sample_eff_nca_2k.sh`        | `sample_eff_nca_cpp{10,25,50,100}`, `nca_stgt_v2`                       | done   |
| reptile 1k  | `eval_sample_eff_reptile1k_2k.sh`  | `sample_eff_reptile1k_cpp{10,25,50,100}`, `sample_eff_reptile1k_full`   | todo   |
| reptile 3k  | `eval_sample_eff_reptile3k_2k.sh`  | `sample_eff_reptile3k_cpp{10,25,50,100}`, `sample_eff_reptile3k_full`   | done   |
| reptile 10k | `eval_sample_eff_reptile10k_2k.sh` | `sample_eff_reptile10k_cpp{10,25,50,100}`, `sample_eff_reptile10k_full` | done   |


## Fine-tuning on Replogle-Nadig


| Model                 | Description                                               | SLURM                                 | Checkpoint                                       | Status |
| --------------------- | --------------------------------------------------------- | ------------------------------------- | ------------------------------------------------ | ------ |
| `rpnd_fewshot`        | Baseline — no PPT, reproducing ARC ST-HVG-Replogle        | `sbatch submit_train_rpnd_fewshot.sh` | `state_runs/rpnd_baseline/checkpoints/last.ckpt` | done   |
| `spptv2_rpnd_fewshot` | `sergio_ppt_v2` (last ckpt) finetuned on `replogle-nadig` | `sbatch submit_train_spptv2_rpnd.sh`  | `state_runs/spptv2_rpnd/checkpoints/last.ckpt`   | done   |
| `nca_rpnd_fewshot`    | `nca_ppt` finetuned on `replogle-nadig`                   | `sbatch submit_train_nca_rpnd.sh`     | `state_runs/nca_rpnd/checkpoints/last.ckpt`      | done   |
| `reptile3k_rpnd_fewshot`    | `reptile3k` finetuned on `replogle-nadig`                   | `sbatch submit_train_reptile3k_rpnd_fewshot.sh` | `state_runs/reptile3k_rpnd_fewshot/checkpoints/last.ckpt` | training (386152), eval (386794) |

