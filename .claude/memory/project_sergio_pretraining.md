---
name: sergio_synthetic_pretraining
description: Project goal — synthetic pre-pre-training of State ST model using SERGIO-generated scRNA-seq data
type: project
---

# SERGIO Synthetic Pre-Pre-Training

**Goal:** Train State ST on SERGIO-generated perturbation data, evaluate on synthetic test set, then use as init for real-data fine-tuning.

**Why:** SERGIO provides ground-truth GRN, exact perturbation effects, and noise-free DEG labels — removes label noise as confound, enables controlled experiments.

**Key decisions (as of 2026-03-20):**
- No PLM embeddings. Use `embed_key=X_hvg` (normalized expression counts) for both training and eval. SE model not involved.
- Synthetically generated PLMs for both training and evaluation (no real data yet).
- Start with KO perturbations on master regulators only.
- Fixed test set generated once, never touched during sweeps.

**Plan document:** `/mnt/polished-lake/home/nkondapaneni/state/plan.md`
**Reference plan:** `/mnt/polished-lake/home/nkondapaneni/test-sergio-pypi/plan.md`

**How to apply:** When the user asks about the SERGIO pipeline, data format, or training config, refer to plan.md for the current engineering spec.
