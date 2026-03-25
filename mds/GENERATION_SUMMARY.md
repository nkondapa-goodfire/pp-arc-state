# SERGIO Synthetic Dataset — Generation Summary

Based strictly on: `generate_dataset.py`, `grn_utils.py`, `build_merged_h5ads.py`,
`generation_configs/dataset_ppt.json`, `generation_configs/dataset_ppt_d2.json`,
TOML configs, and training scripts.

---

## Pipeline Overview

Three sequential stages:

1. **Generate** (`generate_dataset.py`) — simulate per-instance h5ads, one per (grn_type, grn_size, noise_level, seed)
2. **Merge** (`build_merged_h5ads.py`) — concatenate instances by condition into training-ready h5ads, add `obsm["X_hvg"]`
3. **Train** (`submit_train_ppt_v2.sh` / `submit_train_tgt.sh`) — run `state tx train` against the merged dataset

---

## Stage 1: Generation

### Gene Pool
- Fixed vocabulary of **2000 synthetic genes** (`SYN_0000`–`SYN_1999`)
- Each GRN instance draws its active genes via **random draw without replacement** (`gene_strategy = "random_draw"`)
- Inactive genes receive zero expression; pool indices stored in `adata.uns["grn_gene_indices"]`

### GRN Topology Types

**ER (Erdős–Rényi)**
- `nx.erdos_renyi_graph(n, p_edge, directed=True)`, self-loops removed
- DAG conversion: random permutation rank → keep edge `(u,v)` only if `rank[u] < rank[v]`; drops ~50% of edges; produces balanced depth (avoids DFS deep-chain bias)
- `p_edge = 0.08` (pre-DAG; effective density ~0.04)

**BA (Barabási–Albert, fixed-m)**
- `nx.barabasi_albert_graph(n, m=2)` (undirected), then orient all edges `lower-index → higher-index`
- DAG by construction; no cycle removal needed; preserves power-law degree distribution
- `m = 2` → mean in-degree ≈ 2

**BA-VM (Barabási–Albert, variable-m)**
- Custom preferential attachment starting from a 3-node fully-connected seed graph
- Each new node samples `m ∈ {1, 2, 3}` from weights `[0.57, 0.29, 0.14]` (geometric-like, p=0.5)
- Attachment probability `∝ out_degree(u) + 1` for existing nodes
- Edges oriented `lower-index → higher-index`, DAG by construction
- Mean in-degree ≈ 1.57; ~56% of nodes get exactly 1 regulator

**Connectivity guarantee (all types)**
- Any node with degree 0 is attached to a random existing node as a target (`_ensure_connected`), so all genes participate in the network

### Edge Attributes
- **Sign**: `frac_repression = 0.3` (hardcoded); `n_rep = max(1, floor(0.3 × n_edges))` edges chosen uniformly at random as repressive; remainder activating
- **K (interaction strength)**: `K = sign × Uniform(0.5, 2.0)`; signed float passed directly to SERGIO
- **Hill coefficient**: `hill = Uniform(1.5, 3.0)` per edge, independent of sign

### SERGIO Simulation

**Basal rates** (master regulators only — nodes with in-degree 0):
- Sampled per bin: `Uniform(basal_low=0.2, basal_high=1.5)`, independently for each bin
- This is the **sole source of cell-type variation** between bins within a GRN instance

**Fixed SDE parameters** (`sergio_kwargs`):
| Parameter | Value |
|-----------|-------|
| `noise_type` | `"dpd"` (Dropout-Poisson-Degradation) |
| `decays` | 0.8 |
| `dt` | 0.01 |
| `sampling_state` | 15 |
| `dynamics` | `false` (steady-state, not time-series) |

**Noise levels** (applied as `noise_params`):
| Label | Value |
|-------|-------|
| `none` | 0.0 |
| `low` | 0.1 |
| `high` | 0.5 |

**Bins and cells per bin**:
- `n_bins = 5` → 5 cell types per GRN instance (labeled `bin_0`…`bin_4`)
- `dataset_ppt.json`: `n_sc = 25` cells per bin → 125 cells per condition
- `dataset_ppt_d2.json`: `n_sc = 200` cells per bin → 1000 cells per condition

### Perturbation Implementation

**Gene selection**: top-K by out-degree, `k_perts = 7` (hub genes have the highest downstream cascade impact; works across all graph types)

**Perturbation mechanism** (applied to `G_sim` before running SERGIO):
1. Remove all incoming and outgoing edges from the target gene
2. Scale the gene's basal rate in the Regs file by `(1 − strength)`

| Type | Strength | Basal rate multiplier |
|------|----------|-----------------------|
| KO | 1.0 | 0.0 (fully silenced) |
| KD_060 | 0.6 | 0.4× |
| KD_020 | 0.2 | 0.8× |

**Control simulation** is shared across all perturbations for a given GRN instance (run once, reused for all 21 perturbed h5ads).

**Per instance**: 1 control + 7 genes × 3 pert types = **22 SERGIO simulations**

### Per-Instance AnnData Schema

| Field | Description |
|-------|-------------|
| `X` | Raw SERGIO expression → normalize_total(1e4) → log1p → sparse CSR float32, shape `(n_cells, 2000)` |
| `obs["gene"]` | `"non-targeting"` (control) or `"SYN_{k:04d}_{pert_label}"` |
| `obs["cell_type"]` | `"{grn_type}_size{n:03d}_seed{seed:04d}"` — one unique value per GRN instance |
| `obs["gem_group"]` | `"bin_0"`…`"bin_4"` — cell type within the GRN (driven by basal rate variation) |
| `obs["ko_out_degree"]` | Out-degree of the perturbed gene; −1 for control cells |
| `obs["pert_strength"]` | 0.0 (control), 0.2 / 0.6 / 1.0 (pert) |
| `adata.var.index` | `SYN_0000`…`SYN_1999` (always full 2000-gene pool) |
| `adata.uns["grn_type/seed/size/params/gene_indices"]` | GRN provenance metadata |

**Note**: `obsm["X_hvg"]` is **not** written by `generate_dataset.py`. It is added in Stage 2.

---

## Stage 2: Merge (`build_merged_h5ads.py`)

- Groups per-instance h5ads by `(grn_type, grn_size, noise_label, pert_type)` and concatenates all seeds into one file
- Output filename pattern: `{grn_type}_size{n}_noise{label}_{pert_type}.h5ad` (e.g. `BA_size010_noiselow_KO.h5ad`)
- **Adds `obsm["X_hvg"] = X.toarray()`** at this step — this is what the training `embed_key=X_hvg` reads
- Writes `manifest.csv` with columns: `merged_file, grn_type, grn_size, noise_label, pert_type, n_cells, n_instances`
- Supports optional `--bins` flag to filter to a subset of `gem_group` values at merge time

---

## Stage 3: Training

**Model**: `model=replogle`, `training=replogle`

**Key data kwargs**:
| Kwarg | Value |
|-------|-------|
| `embed_key` | `X_hvg` |
| `pert_col` | `gene` |
| `cell_type_key` | `cell_type` (= `{grn_type}_size{n}_seed{k}`) |
| `batch_col` | `gem_group` (= `bin_0`…`bin_4`) |
| `control_pert` | `non-targeting` |
| `output_space` | `gene` |
| `perturbation_features_file` | `pert_onehot_map_ppt.pt` |

**Training config**: 25k steps, val every 2k steps, 8 GPUs, DDP (`ddp_find_unused_parameters_true`)

### TOML Configs

| Config | Dataset | Zeroshot |
|--------|---------|---------|
| `sergio_ppt_train.toml` | `SERGIO_PPT_merged` | none |
| `sergio_tgt_train.toml` | `SERGIO_TGT_train_merged` | none |
| `sergio_tgt_test.toml` | `SERGIO_TGT_test_merged` | all entries → `"test"` (seeds 100–103) |
| `sergio_train.toml` (older) | `train_merged` | `bin_4 = "val"` |

Test set uses seeds **100–103**, disjoint from training seeds **0–3**.

---

## Dataset Scale (`dataset_ppt.json`)

| Axis | Values | Count |
|------|--------|-------|
| Graph type | ER, BA, BA-VM | 3 |
| GRN size | 10, 50, 100 | 3 |
| Noise level | none (0.0), low (0.1), high (0.5) | 3 |
| Seeds | 0, 1, 2, 3 | 4 |
| Perturbation genes | top-7 by out-degree | 7 |
| Perturbation types | KO, KD_020, KD_060 | 3 |

**Total h5ad files (raw)**: 3 × 3 × 3 × 4 × 7 × 3 = **2,268**
**Total SLURM tasks**: 3 × 3 × 3 × 4 = **108**

`dataset_ppt_d2.json` is identical except `n_sc = 200` (vs 25) → output at `SERGIO_PPTD2`.

---

## Presentation Slide: 10 Design Choices

1. **Three graph topologies** — Erdős–Rényi (uniform random edges), Barabási–Albert fixed-m (power-law hubs, m=2), and BA variable-m (geometric m ∈ {1,2,3}) — covering a spectrum from flat to highly heterogeneous regulatory structure.

2. **All graphs are enforced as DAGs** — ER via random-permutation rank orientation (balanced depth); BA via lower→higher node index (DAG by construction, no pruning needed).

3. **Edge parameters drawn per edge independently** — sign (30% repressive, 70% activating), interaction strength K ~ Uniform(0.5, 2.0), Hill coefficient ~ Uniform(1.5, 3.0).

4. **Cell-type diversity from basal rate variation only** — 5 bins per GRN instance each draw independent master-regulator basal rates ~ Uniform(0.2, 1.5); topology and edge weights are identical across bins.

5. **Steady-state DPD noise model** — Dropout-Poisson-Degradation at three levels (0.0, 0.1, 0.5); ablates model sensitivity to technical noise; mRNA decay 0.8, dt=0.01, 15-step burn-in.

6. **Perturbations target top-7 hub genes by out-degree** — three strengths (KO=1.0, KD_060=0.6, KD_020=0.2) implemented by edge removal + basal rate scaling by (1−strength); control simulation shared across all perturbations per instance.

7. **Fixed 2000-gene pool with sparse embedding** — each GRN activates 10/50/100 genes drawn randomly; inactive positions are zero; ~99.5% sparsity at generation time, matching real scRNA-seq dropout.

8. **Two-stage output pipeline** — raw per-instance h5ads (22 files per GRN) are merged by condition across seeds by `build_merged_h5ads.py`, which also materializes `obsm["X_hvg"]` for the training dataloader.

9. **108 SLURM generation tasks, 2,268 raw h5ads** — 4 seeds × 3 graph types × 3 sizes × 3 noise levels; merged into condition-grouped files enabling ablation by TOML config swap.

10. **Test set uses fully disjoint seeds** — training seeds 0–3, test seeds 100–103; test split registered via `[zeroshot]` in `sergio_tgt_test.toml`, stratified across all graph types and sizes.
