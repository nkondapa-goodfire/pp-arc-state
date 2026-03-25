# Reptile Training on SERGIO Synthetic Data

Reptile requires a notion of **task**: a distribution of data over which the inner loop
runs for k steps before the outer update. This document maps the abstract Reptile algorithm
onto the SERGIO dataset structure and describes how to implement it concretely.

---

## obs column reference

Before anything else — the column names are counterintuitive:

| `obs` column | values | role in State | role in Reptile |
|---|---|---|---|
| `cell_type` | `grn_0000`, `grn_0001`, … | cell line identifier | **task identifier** |
| `gem_group` | `bin_0`, `bin_1`, …, `bin_4` | technical batch | **within-task variation (cell state)** |
| `gene` | `non-targeting`, `SYN_0033_KO`, … | perturbation identity | within-task variation |

`cell_type` = GRN seed (the biology / topology). `gem_group` = bin (the cell state within
that topology). Bins are **not** batches in the usual sense — each bin has a distinct
steady-state expression profile driven by different master-regulator basal rates, even though
they share the same GRN edges, K, and hill coefficients.

---

## Concept mapping

| Reptile concept | SERGIO / State analog |
|---|---|
| Task τ | One GRN instance: `(grn_type, grn_size, grn_seed)` identified by `cell_type` |
| Task distribution p(τ) | Uniform over all GRN instances in the dataset |
| Inner-loop mini-batch | One `(noise_level, pert_gene, pert_type, bin)` draw from that GRN |
| k inner steps | k gradient steps, each drawing a **new** `(noise_level, pert_gene, pert_type, bin)` |
| θ_t (init snapshot) | Model weights before the inner loop |
| θ̃ (adapted weights) | Model weights after k inner steps |
| Outer update | `θ ← θ + ε(θ̃ − θ_t)` |

**Why noise level, pert type, and bin are mini-batch variation, not task variation:**
The GRN topology (edges, K, hill coefficients) is fixed by `(grn_type, grn_size, grn_seed)`.
Noise level and perturbation type are different interventions on the same topology. Bins are
different cell states (different basal rates) that share the same topology. All three are
observations of the same underlying GRN — so k gradients taken across them all point into
the same "GRN's solution manifold," which is the structure AvgGradInner exploits.

**Bin holdout:** `bin_4` (`gem_group == "bin_4"`) is reserved for zero-shot evaluation —
never seen during training. Inner-loop steps sample only from bins 0–3.

---

## Algorithm

```
θ ← initial parameters
TRAIN_BINS = {bin_0, bin_1, bin_2, bin_3}    # bin_4 held out for zero-shot eval

for each outer iteration:
    Sample task τ = (grn_type, grn_size, grn_seed) uniformly
    θ_t ← θ                                            # snapshot

    for step in 1..k:
        Sample h5ad from τ's files                     # new (noise_level, pert_gene, pert_type)
        Sample bin b from TRAIN_BINS                   # new cell state
        Load mini-batch: rows where gem_group == b, from control + pert cells
        g ← ∇_θ L(θ, mini-batch)
        θ ← θ − α·g                                    # inner-loop Adam/SGD

    θ̃ ← θ                                              # adapted weights
    θ ← θ_t + ε·(θ̃ − θ_t)                            # outer update (restore + move)
```

The outer update direction `(θ̃ − θ_t)` can be plugged directly into an outer-loop Adam
optimizer (treat it as a gradient, negate for minimization).

---

## Task structure in the dataset

Each GRN instance produces files at:
```
SERGIO_PPT/{grn_type}/size_{n:03d}/noise_{label}/grn_{seed:04d}/
    SYN_{k:04d}_KO.h5ad
    SYN_{k:04d}_KD_020.h5ad
    SYN_{k:04d}_KD_060.h5ad
```

Each h5ad contains 250 rows: 5 bins × 25 cells × 2 conditions (control + perturbed).
Filtering to one bin gives 50 rows (25 ctrl + 25 pert) — the mini-batch for one inner step.

From `dataset_ppt.json`: 3 grn_types × 3 sizes × 3 noise_levels × 4 seeds = **108 GRN
instances** (tasks). Each task has:

```
7 pert_genes × 3 pert_types × 3 noise_levels = 63 h5ad files
63 files × 4 training bins = 252 distinct (file, bin) pairs
```

With 252 unique mini-batches per task, k can be set comfortably in the range 10–50 without
any within-loop overlap.

**Task key in obs:** `cell_type` column (`grn_0000`, …). To get all h5ad files for a task,
group by directory path `{grn_type}/size_{n:03d}/…/grn_{seed:04d}/`.

---

## Key design decisions

**k (inner steps):** With k=1 Reptile reduces to joint training — no meta-learning signal.
The AvgGradInner coefficient is `½k(k−1)α`, so k=5–20 is the practical sweet spot. Draw
a fresh `(h5ad, bin)` pair at each step without replacement to keep gradients independent.
With 252 pairs per task, this is trivially satisfied up to k=252.

**α (inner learning rate):** Adam with lr=1e-3, **β₁=0**. Removing momentum restores
gradient independence across inner steps — momentum would carry signal from step i into
step i+1, coupling g_i and g_{i+1} and weakening the AvgGradInner term.

**ε (outer step size):** 0.1–1.0, linearly annealed to 0. Outer optimizer can be plain
SGD on `(θ̃ − θ_t)` or Adam treating `(θ̃ − θ_t)` as a gradient.

**Outer optimizer state:** The outer Adam accumulates moments across tasks (this is
correct). The inner Adam is re-initialized from scratch each inner loop — do not carry
inner Adam state between tasks.

---

## Relationship to current State training

Current pretraining (`sergio_ppt_v1`) trains on a merged h5ad with all tasks mixed together
and samples mini-batches uniformly across the entire dataset. This is **joint training**
(effective k=1), not Reptile. It optimizes AvgGrad but not AvgGradInner.

True Reptile takes k>1 steps on the same GRN task, varying `(noise_level, pert_type, bin)`
within each task. This explicitly maximizes gradient alignment within a task — the quantity
that enables fast fine-tuning on new cell lines.

```
E[g_Reptile] = k · AvgGrad  −  ½k(k−1)α · AvgGradInner
E[g_joint]   = 1 · AvgGrad
```

The AvgGradInner term is absent in joint training and grows quadratically with k in Reptile.

---

## Minimal implementation sketch

```python
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
import anndata
import torch
from torch.optim import Adam

TRAIN_BINS = ["bin_0", "bin_1", "bin_2", "bin_3"]   # bin_4 = zero-shot holdout

# Precompute: group h5ad paths by task key (grn_type, grn_size, grn_seed)
task_files: dict[tuple, list[Path]] = defaultdict(list)
for path in all_h5ad_paths:
    key = (path.grn_type, path.grn_size, path.grn_seed)
    task_files[key].append(path)
task_keys = list(task_files.keys())

outer_optimizer = Adam(model.parameters(), lr=outer_lr, betas=(0.9, 0.999))

for outer_step in range(max_outer_steps):
    τ = random.choice(task_keys)
    files = list(task_files[τ])

    θ_snapshot = deepcopy(model.state_dict())
    inner_optimizer = Adam(model.parameters(), lr=inner_lr, betas=(0.0, 0.999))

    # Sample k distinct (h5ad, bin) pairs without replacement
    pool = [(f, b) for f in files for b in TRAIN_BINS]
    random.shuffle(pool)
    inner_samples = pool[:k]

    for h5ad_path, bin_label in inner_samples:
        adata = anndata.read_h5ad(h5ad_path)
        mask = adata.obs["gem_group"] == bin_label      # filter to one bin
        batch = make_batch(adata[mask])                 # control + pert cells for this bin
        loss = model(batch)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()

    # Outer update: move init toward adapted weights
    reptile_grad = {
        name: θ_snapshot[name] - param.data
        for name, param in model.named_parameters()
    }
    model.load_state_dict(θ_snapshot)                   # restore before outer step
    outer_optimizer.zero_grad()
    for name, param in model.named_parameters():
        param.grad = reptile_grad[name]
    outer_optimizer.step()
```

Note: `outer_lr` controls ε. When using plain SGD for the outer loop, `ε = outer_lr`
directly. When using Adam, outer_lr is the Adam step size applied to the normalized
`(θ̃ − θ_t)` direction.
