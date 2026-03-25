# Plan 3: Synthetic Pre-Pre-Training for Perturbation Prediction

---

## Overview

**Central hypothesis**: Pre-pre-training the State Transition model on formally defined
SERGIO synthetic data improves sample efficiency and peak performance on target datasets,
compared to training on the target dataset alone.

**Experimental design**:

```
Pre-pre-train (SERGIO synthetic)
        ↓
Fine-tune on N target samples    ← sweep N
        ↓
Eval on held-out target samples

vs.

Train from scratch on N target samples   ← baseline, same N sweep
        ↓
Eval on held-out target samples
```

**Target datasets**:

- **Synthetic cell lines** (held-out SERGIO seeds never seen in pre-pre-training)
- **Real data** (Replogle-Dang perturbation dataset)

**Primary metric**: Pearson Δ Corr on held-out perturbations, plotted vs. N training samples.

---

## 1. Pre-Pre-Training Data

SERGIO synthetic dataset, formally specified by a generation config JSON (reproducible).


| Axis        | Values                     |
| ----------- | -------------------------- |
| GRN type    | ER, BA, BA-VM              |
| GRN size    | 10, 50, 100                |
| Noise level | none, low, high            |
| Pert type   | KO, KD_010, KD_050, KD_080 |
| Seeds       | 0–3 (4 per graph type)     |
| Bins        | 5 per seed                 |


Checkpoint: train for 25k steps, save for fine-tuning initialization.

---

## 2. Target Datasets

### 2.1 Held-Out Synthetic Cell Lines

New SERGIO seeds **never seen** during pre-pre-training, forming held-out "cell lines."

- Seeds: 100–103 (or similar offset, clearly disjoint from pre-pre-train seeds 0–3)
- Same generation config as pre-pre-training (same GRN types, sizes, noise)
- All 4 pert_types generated (KO, KD_010, KD_050, KD_080)

Fine-tuning uses all pert_types, varying N cells/perturbations. Evaluation is on held-out
perturbations from the same seeds (see Section 4.1).

### 2.2 Real Data: Replogle-Dang

Genome-scale Perturb-seq dataset.

- Fine-tune on N cells from training cell lines
- Evaluate on held-out perturbations within a training cell line (fewshot) or a fully
held-out cell line (zeroshot), following the Replogle evaluation protocol

Dataset path and preprocessing: TBD.

---

## 3. Training Protocol

### 3.1 Pre-Pre-Training

SERGIO obs column mapping (consistent with plan2.md §1):


| `obs` column | `state tx train` kwarg | Semantics                             |
| ------------ | ---------------------- | ------------------------------------- |
| `cell_type`  | `cell_type_key`        | GRN seed — cell line analog           |
| `gem_group`  | `batch_col`            | Bin — minor variation within same GRN |
| `gene`       | `pert_col`             | Perturbation identity string          |


```bash
uv run state tx train \
  data.kwargs.toml_config_path=configs/sergio_ppt_train.toml \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.pert_col=gene \
  data.kwargs.control_pert=non-targeting \
  training.max_steps=50000 \
  name=sergio_ppt_v1
```

Single run. Checkpoint saved at `state_runs/sergio_ppt_v1/`.

### 3.2 N Sweep via Checkpoint Intervals

Since State uses a learning rate independent of the number of training steps (ADAM, no scheduler), 
we will train on the full dataset for T steps and reading intermediate checkpoints at steps t_1 < t_2 < ... < T

Checkpoint interval: save every `ckpt_every_n_steps` (default 4000). X-axis of the
learning curve = cumulative training samples seen = `step × batch_size × n_gpus`.

**With pre-pre-training** (two runs total per target dataset):

```bash
# Run 1: warm-start from PPT checkpoint (weights only, fresh optimizer)
uv run state tx train \
  data.kwargs.toml_config_path=configs/{target}_train.toml \
  training.max_steps=T \
  training.ckpt_every_n_steps=4000 \
  model.kwargs.init_from=.../sergio_ppt_v1/checkpoints/last.ckpt \
  name={target}_ppt

# Run 2: scratch baseline
uv run state tx train \
  data.kwargs.toml_config_path=configs/{target}_train.toml \
  training.max_steps=T \
  training.ckpt_every_n_steps=4000 \
  name={target}_scratch
```

`model.kwargs.init_from` loads model weights only — optimizer state is reset, so fine-tuning
starts with a fresh Adam from step 0. This is the intended pretrain→finetune path
(`_train.py:305`).

Eval runs against every saved checkpoint for both runs, producing the full learning curve.
Plot: x = cumulative samples seen (`step × batch_size × n_gpus`), y = metric, two lines
(PPT vs. scratch).

### 3.3 Open: Warm-Start Caveats

`init_from` filters out mismatched-size parameters and rebuilds the decoder/pert_encoder if
`output_space` or `pert_dim` differs between PPT and fine-tune configs. Ensure PPT and
fine-tune use the same `output_space` and `embed_key` to avoid silent weight drops.

---

## 4. Evaluation

### 4.1 Synthetic Cell Lines

**Held-out cell line**: Eval on a new seed with varying N fine-tuning samples (including N=0).

- N=0 tests zeroshot transfer from pre-pre-training alone
- N>0 tests sample efficiency vs. scratch baseline

### 4.2 Replogle-Dang — Two Conditions

Following the Replogle evaluation protocol:

**Fewshot**: Hold out perturbations within a seen cell line. Fine-tune on remaining
perturbations, eval on held-out ones.

**Zeroshot**: Hold out an entire cell line. Fine-tune on other cell lines, eval on
held-out cell line with no fine-tuning data from it.

### 4.3 Metrics

**Primary** (reported in main figures):


| Metric              | What it measures                       | Computation                                                                                                                    |
| ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Pearson Δ Corr**  | Profile shape of predicted effect      | `pearsonr(mean(pred_pert) − mean(pred_ctrl), mean(true_pert) − mean(true_ctrl))` across all genes, averaged over perturbations |
| **Log2FC Spearman** | Direction + magnitude on true DE genes | Spearman r of LFCs restricted to genes significant in the true condition                                                       |


**Secondary** (Cell-Eval suite, reported in supplementary / appendix):


| Metric                                        | What it measures                         | Computation                                                                                                              |
| --------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Perturbation Discrimination Score (PDisc)** | Can model rank perturbations correctly   | Manhattan distance of predicted pseudobulk vs. all true pseudobulks; report normalized rank of the matching perturbation |
| **DE Overlap @ K**                            | Gene-set recovery                        | Wilcoxon on predicted vs. ctrl; intersect top-K predicted DEGs with true top-K DEGs; report Jaccard @ K={20, 50, 100}    |
| **Effect Size Correlation**                   | Does model predict strong vs. weak perts | Spearman of n_DEGs(pred) vs. n_DEGs(true) across perturbations in a file                                                 |


All metrics computed per perturbation then averaged per file. Both primary and secondary metrics
plotted vs. log N training samples — one curve per condition (PPT vs. scratch), per dataset.

---

## 5. Key Questions


| #   | Question                                                                                          |
| --- | ------------------------------------------------------------------------------------------------- |
| 1   | Does pre-pre-training improve **sample efficiency** — same performance with fewer target samples? |
| 2   | Does pre-pre-training improve **peak performance** at large N?                                    |
| 3   | Does the benefit generalize from synthetic to real (Replogle-Dang)?                               |
| 4   | Does zeroshot (N=0) show any off-the-shelf benefit from synthetic pre-pre-training?               |


---

## 6. Open Items


| #   | Item                                                                       |
| --- | -------------------------------------------------------------------------- |
| 1   | Replogle-Dang dataset path, preprocessing, and TOML setup                  |
| 2   | Decide held-out cell line for Replogle-Dang zeroshot condition             |
| 3   | How many seeds for synthetic held-out cell lines (Section 2.1)?            |
| 4   | Choose T (max_steps) for fine-tuning runs — depends on target dataset size |


---

## 7. Model Log


| Model              | Description                                                                           | SLURM                                                                                             | CKPT                                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `sergio_ppt_v1`    | Pre-pre-training on steady state data from synthetic GRNs                             | `sbatch submit_train_ppt.sh` (job 372755)                                                         | `state_runs/sergio_ppt_v1/checkpoints/last.ckpt`                                                                                   |
| `nca_ppt`          | Pre-pre-training on nca data                                                          | `sbatch nca_prepretraining.sh` (job 371416)                                                       | `nca-pre-pretraining/checkpoints/model_100.pth` → `state_runs/nca_ppt/nca_ppt_backbone_init.pt` (remapped via `nca_ckpt_remap.py`) |
| `sergio_tgt`       | Baseline model with no PPT                                                            | `sbatch submit_train_tgt.sh` (job 372756 → rerun job 373828)                                      | `state_runs/sergio_tgt/checkpoints/last.ckpt`                                                                                      |
| `spptv1_last_stgt` | PPT model (`sergio_ppt_v1 last ckpt`) finetuned for N steps on `sergio_tgt_tr`        | `sbatch submit_train_spptv1_last_stgt.sh` (job 372760 → rerun job 373830, afterany:373828:373829) | `state_runs/spptv1_last_stgt/checkpoints/last.ckpt` (init from `sergio_ppt_v1/checkpoints/last.ckpt`)                              |
| `nca_stgt`         | PPT model (`nca_ppt`) finetuned for N steps on `sergio_tgt_tr`                        | `sbatch submit_train_nca_stgt.sh` (job 372758 → rerun job 373829)                                 | `state_runs/nca_stgt/checkpoints/last.ckpt` (init from `state_runs/nca_ppt/nca_ppt_backbone_init.pt`)                              |
| `rpnd_baseline`    | Baseline model with no PPT, reproducing ARC ST-HVG-Replogle                           | `sbatch submit_train_rpnd_baseline.sh` (job 375329)                                               | `state_runs/rpnd_baseline/checkpoints/last.ckpt`                                                                                   |
| `spptv1_rpnd`      | PPT model (`sergio_ppt_v1 last ckpt`) finetuned for N steps on `replogle-nadig`       | `sbatch submit_train_spptv1_rpnd.sh` (job 375330)                                                 | `state_runs/spptv1_rpnd/checkpoints/last.ckpt` (init from `state_runs/sergio_ppt_v1/checkpoints/last.ckpt`)                        |
| `nca_rpnd`         | PPT model (`nca_ppt`) finetuned for N steps on `replogle-nadig`                       | `sbatch submit_train_nca_rpnd.sh` (job 375331, afterany:375329:375330)                            | `state_runs/nca_rpnd/checkpoints/last.ckpt` (init from `state_runs/nca_ppt/nca_ppt_backbone_init.pt`)                              |
| `rpnd_fewshot`     | Fewshot baseline on `replogle-nadig`; k562 split: 858 test perts held out, rest train | `sbatch submit_train_rpnd_fewshot.sh` (job 377870)                                                | `state_runs/rpnd_fewshot/checkpoints/last.ckpt`                                                                                    |
| `spptv1_rpnd_fewshot` | PPT model (`sergio_ppt_v1 last ckpt`) finetuned on `replogle-nadig` fewshot split  | `sbatch submit_train_spptv1_rpnd_fewshot.sh` (job 378530)                                         | `state_runs/spptv1_rpnd_fewshot/checkpoints/last.ckpt` (init from `state_runs/sergio_ppt_v1/checkpoints/last.ckpt`)                |
| `nca_rpnd_fewshot` | PPT model (`nca_ppt`) finetuned on `replogle-nadig` fewshot split                    | `sbatch submit_train_nca_rpnd_fewshot.sh` (job 378529)                                            | `state_runs/nca_rpnd_fewshot/checkpoints/last.ckpt` (init from `state_runs/nca_ppt/nca_ppt_backbone_init.pt`)                      |
| `spptv2_last_stgt` | PPT model (`sergio_ppt_v2 last ckpt`) finetuned for N steps on `sergio_tgt_tr`       | `sbatch submit_train_spptv2_last_stgt.sh` (job 381878), eval job 381880                           | `state_runs/spptv2_last_stgt/checkpoints/last.ckpt` (init from `state_runs/sergio_ppt_v2/checkpoints/last.ckpt`)                   |


