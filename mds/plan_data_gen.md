# Engineering Plan: SERGIO Synthetic Pre-Pre-Training for State ST

## Summary

Train the State Transition (ST) model from scratch on SERGIO-generated synthetic scRNA-seq data using the `X_hvg` input path (log-normalized expression counts, no PLM). Evaluate on a fixed synthetic test set. The goal is to establish that State learns meaningful perturbation structure from synthetic data and identify the best training configuration to use as an initializer before real-data fine-tuning.

---

## 1. Fixed Gene Pool1

Define a vocabulary of 2000 synthetic gene names, fixed for the lifetime of this project:

```
SYN_0000, SYN_0001, ..., SYN_1999
```

Each GRN instance draws a subset of **10, 50, or 100 genes** from this pool (see Section 5 for the size grid). Genes not selected for a given GRN instance have zero expression for all cells in that instance. `X_hvg` is always `(n_cells, 2000)` across all GRN instances — consistent dimensionality, no alignment needed.

**Why the 2000-gene pool**: State's real training data (e.g., Replogle-Nadig) has all cells measuring the same fixed gene panel, with `preprocess_train` producing a consistent `X_hvg` shape across the full dataset. The fixed pool replicates this. Using 2000 matches the standard HVG count used for Replogle, so `basal_encoder` weights transfer directly at fine-tuning time.

**HVG selection**: **Skip** — do not run `sc.pp.highly_variable_genes`. With ≤100 active genes per GRN and 1900+ zeros, variance across the dataset is dominated by gene presence/absence rather than biological signal, making HVG selection unreliable. Use all 2000 dimensions directly as `X_hvg`.

**Sparsity**: Each cell vector is ~90% zeros. This is normal for scRNA-seq (high dropout) and does not require special handling.

---

## 2. Gene Sampling Strategy

Each GRN instance samples its 200 active genes from the 2000-gene pool. The strategy is a configurable flag, defaulting to `random_draw`.

```python
def sample_grn_genes(pool_size: int, grn_size: int, seed: int, strategy: str = "random_draw") -> list[int]:
    """Return indices into the global gene pool for this GRN instance."""
    rng = np.random.default_rng(seed)
    if strategy == "random_draw":
        return sorted(rng.choice(pool_size, size=grn_size, replace=False).tolist())
    # Additional strategies can be added here (e.g., "fixed_blocks", "overlap_core")
    raise ValueError(f"Unknown gene sampling strategy: {strategy}")
```

The sampled indices are stored in `adata.uns["grn_gene_indices"]` for reproducibility.

---

## 3. GRN Topology Types

**SERGIO requires a DAG.** SERGIO processes genes in topological order (master regulators first, then downstream layers). Cycles make topological ordering impossible and will break simulation. Both ER and BA graphs can contain cycles and must be converted to DAGs before use.

```python
def enforce_dag(G: nx.DiGraph, seed: int) -> nx.DiGraph:
    """Keep only edges consistent with a random topological ordering."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(list(G.nodes())).tolist()
    rank = {node: i for i, node in enumerate(order)}
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes())
    dag.add_edges_from((u, v) for u, v in G.edges() if rank[u] < rank[v])
    return dag
```

This drops ~50% of edges on average, so target `p_edge` and `m` should be set at **2× the desired post-DAG density**.

### 3.1 Erdős–Rényi (ER) — true random graph

Each directed edge is included independently with probability `p`, then `enforce_dag` is applied. Degree distribution is binomial, no hubs.

```python
def generate_er_grn(n_genes: int, p_edge: float, seed: int) -> nx.DiGraph:
    G = nx.erdos_renyi_graph(n_genes, p_edge, seed=seed, directed=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return enforce_dag(G, seed)
```

Actual parameter used: `p_edge = 0.08`.

### 3.2 Barabási–Albert (BA) — power-law degree distribution

Preferential attachment produces hub regulators with many targets — biologically more realistic. BA node construction order (0, 1, ..., n-1) is used as the natural topological ordering — orient all edges from lower to higher index, which avoids most cycle removal and preserves the power-law structure.

```python
def generate_ba_grn(n_genes: int, m: int, seed: int) -> nx.DiGraph:
    G = nx.barabasi_albert_graph(n_genes, m, seed=seed)
    DG = nx.DiGraph()
    for u, v in G.edges():
        # Orient by construction order: lower index = earlier node = regulator
        if u < v:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)
    return DG  # already a DAG by construction
```

`m=2` by default (2 edges added per new node). No DAG pruning needed for BA since construction order is acyclic.

### 3.3 Barabási–Albert Variable-m (BA-VM) — sparse heterogeneous in-degree

A variant of BA where each new node draws its number of regulators `m` from `{1, 2, 3}` according to a geometric distribution (p=0.5) truncated and renormalized:

| m | probability |
|---|-------------|
| 1 | ~57%        |
| 2 | ~29%        |
| 3 | ~14%        |

This produces a sparser, more heterogeneous network than fixed-m BA: mean in-degree ≈ 1.57 vs 2.0, with ~56% of genes receiving only a single regulator. Preferential attachment is preserved — edges are still drawn proportional to existing out-degree, so a hub structure still emerges. Edges are oriented lower → higher index as in BA.

```python
def generate_ba_vm_grn(n_genes, m_weights=(0.57, 0.29, 0.14), seed=None):
    # seed graph: 3 nodes fully connected
    # for each new node: sample m ~ Categorical(m_weights), attach to m
    # existing nodes via preferential attachment (∝ out-degree + 1)
    # orient all edges lower-index → higher-index
```

Compared to fixed-m BA at simulation time:
- Lower mean expression (~1.0–1.2 vs ~1.5) due to weaker aggregate regulatory input
- Higher baseline CV (~0.4–0.6 vs ~0.2–0.3) — sparser regulation creates more heterogeneous expression across cell types
- Slightly higher sparsity at all noise levels — genes with only 1 regulator are more easily driven to zero by the SDE noise floor

### 3.4 Metadata tracking

```python
adata.uns["grn_type"] = "ER" | "BA" | "BA-VM"
adata.uns["grn_seed"] = <int>
adata.uns["grn_params"] = {"p_edge": 0.08} | {"m": 2} | {"m_weights": [0.57, 0.29, 0.14]}
adata.uns["grn_gene_indices"] = [42, 107, 883, ...]  # indices into the 2000-gene pool
adata.uns["grn_n_levels"] = nx.dag_longest_path_length(dag) + 1  # depth of the DAG
adata.uns["grn_n_edges"] = dag.number_of_edges()
```

`grn_n_levels` enables post-hoc analysis of whether perturbation signal strength correlates with GRN depth — shallow graphs (2–3 levels) have few downstream cascade targets while deeper graphs produce richer effects.

---

## 4. H5AD Schema

```
adata.X                       float32  (n_cells, 2000)   raw SERGIO expression (zeros for inactive genes)
adata.obsm["X_hvg"]           float32  (n_cells, 2000)   log-normalized expression (all 2000 dims, no HVG selection)
adata.obs["gene"]             str      "non-targeting" | "SYN_{k:04d}_KO" | "SYN_{k:04d}_KD"  (k = index in global pool)
adata.obs["cell_type"]        str      "bin_0" ... "bin_{n_bins-1}"
adata.obs["gem_group"]        str      "grn_{seed:04d}"
adata.obs["ko_out_degree"]    int      out-degree of KO gene in the GRN (-1 for non-targeting cells)
adata.var.index               str      "SYN_0000" ... "SYN_1999"  (always full 2000-gene pool)
adata.uns["grn_type"]         str      "ER" | "BA"
adata.uns["grn_seed"]         int
adata.uns["grn_params"]       dict
adata.uns["grn_gene_indices"] list[int]
```

Preprocessing (applied in-place before saving):

```python
sc.pp.normalize_total(adata)                       # library-size normalize to 10k
sc.pp.log1p(adata)                                 # log1p transform
adata.obsm["X_hvg"] = adata.X.toarray()           # all 2000 genes, no HVG selection
```

---

## 5. Training Dataset

### 5.1 Ablation grid

The training dataset is organized along three orthogonal axes so any subset can be isolated for ablation without regenerating data:

| Axis              | Values                              |
|-------------------|-------------------------------------|
| Graph type        | `ER`, `BA`, `BA-VM`                 |
| GRN size          | `10`, `50`, `100` genes             |
| Noise level       | `none` (0.0), `low` (0.1), `high` (0.5) |
| Perturbation type | `KO`, `KD_020`, `KD_060`            |

KD strengths encode the fraction of gene silencing applied to basal rate and regulatory edge weights:

| Label     | `pert_emb[k]` | Description              |
|-----------|---------------|--------------------------|
| `KO`      | 1.0           | Full knockout            |
| `KD_020`  | 0.2           | 20% knockdown            |
| `KD_060`  | 0.6           | 60% knockdown            |

**4 seeds** (seeds 0–3) × 3 types × 3 sizes × 3 noise levels = **108 GRN instances** total.
Each instance generates **7 perturbations** (top-K by out-degree) × **3 perturbation types** = 21 h5ad files per instance, ~2268 files total.

### 5.2 Directory structure

```
data/sergio_synthetic/SERGIO_PPT/
  {grn_type}/          # ER | BA | BA_VM
    size_{n}/          # size_010 | size_050 | size_100
      noise_{level}/   # noise_000 | noise_010 | noise_050
        grn_{seed:04d}/
          SYN_{k:04d}_KO.h5ad
          SYN_{k:04d}_KD_020.h5ad
          SYN_{k:04d}_KD_060.h5ad
          ...
  manifest.csv           # one row per h5ad: path + all group attributes
```

`manifest.csv` columns: `path, grn_type, grn_size, noise_level, grn_seed, pert_gene, pert_type, pert_strength, pert_out_degree`

### 5.3 Ablation subsets

Any ablation is defined by filtering `manifest.csv` and passing the resulting paths to the dataloader. Example subsets:

| Ablation                        | Filter                                                      |
|---------------------------------|-------------------------------------------------------------|
| Graph type (ER only)            | `grn_type == "ER"`                                          |
| Size sweep                      | `grn_type == "BA" & noise_level == "low"`                   |
| Noise sweep                     | `grn_type == "BA" & grn_size == 100`                        |
| KO only                         | `pert_type == "KO"`                                         |
| KD only (all strengths)         | `pert_type.str.startswith("KD")`                            |
| KD strength sweep               | `pert_type in ["KD_020", "KD_060"]`                         |
| KO + all KD                     | `pert_type in ["KO", "KD_020", "KD_060"]`                   |
| Baseline (full)                 | no filter                                                   |

The TOML config references a pre-filtered manifest rather than a directory, so switching ablations requires only swapping the manifest path.

---

## 6. Perturbation Encoding

Use a **gene-position magnitude vector** keyed to the global 2000-gene pool:

```
pert_emb ∈ [0, 1]^2000
pert_emb[k] = perturbation strength at gene k  (0 = no effect, 1 = full KO)
```

Perturbation strength by type:

| Type            | `pert_emb[k]` | `obs["gene"]`           |
|-----------------|---------------|-------------------------|
| Non-targeting   | 0.0           | `"non-targeting"`       |
| Knockout (KO)   | 1.0           | `"SYN_{k:04d}_KO"`     |
| Knockdown 20%   | 0.2           | `"SYN_{k:04d}_KD_020"` |
| Knockdown 60%   | 0.6           | `"SYN_{k:04d}_KD_060"` |

KO and KD of the same gene share position k — the magnitude encodes perturbation strength. The `pert_encoder` MLP receives float32 input so continuous values are natively supported; no architecture changes are needed. This allows the model to learn that larger values at position k correspond to stronger perturbation of gene k, and to interpolate between KD and KO.

Build a fixed `pert_onehot_map` once before any training run:

```python
pert_onehot_map = {"non-targeting": np.zeros(2000, dtype=np.float32)}
for k in range(2000):
    for label, strength in [("KO", 1.0), ("KD_020", 0.2), ("KD_060", 0.6)]:
        v = np.zeros(2000, dtype=np.float32)
        v[k] = strength
        pert_onehot_map[f"SYN_{k:04d}_{label}"] = v

torch.save(pert_onehot_map, "configs/pert_onehot_map.pt")
```

`pert_dim = 2000` in the model config. Size is fixed regardless of how many GRN instances are in the training set.

---

## 6. Data Generation Pipeline

### 6.1 Per-GRN instance

```python
# scripts/generate_sergio_dataset.py
POOL_SIZE = 2000
GRN_NAMES = [f"SYN_{i:04d}" for i in range(POOL_SIZE)]

def generate_instance(grn_seed, grn_type, grn_size, output_dir, params,
                      gene_strategy="random_draw"):
    # 1. Sample active genes from pool
    gene_indices = sample_grn_genes(POOL_SIZE, grn_size, grn_seed, gene_strategy)
    active_gene_names = [GRN_NAMES[i] for i in gene_indices]

    # 2. Generate GRN topology over the active genes
    if grn_type == "ER":
        G = generate_er_grn(grn_size, p_edge=params["p_edge"], seed=grn_seed)
    elif grn_type == "BA":
        G = generate_ba_grn(grn_size, m=params["m"], seed=grn_seed)
    elif grn_type == "BA-VM":
        G = generate_ba_vm_grn(grn_size, m_weights=params["m_weights"], seed=grn_seed)

    targets, regs = grn_to_sergio_format(G, active_gene_names)
    # Use top-K by out-degree (robust across all graph types)
    top_regs = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)[:K_PERTS]

    # SERGIO simulation parameters (from dataset_ppt.json)
    # n_bins=5, n_sc=25, noise_type="dpd", decays=0.8, sampling_state=15, dt=0.01, dynamics=False

    # 3. Simulate control
    sim_ctrl = sergio(number_genes=grn_size, number_bins=5, number_sc=25,
                      noise_type="dpd", decays=0.8, sampling_state=15, dt=0.01, dynamics=False)
    sim_ctrl.build_graph(targets, regs)
    sim_ctrl.simulate()
    ctrl_expr = sim_ctrl.getExpressions()   # (5, grn_size, 25)

    # 4. Simulate KO and KD for each selected gene (k_perts=7, top-7 by out-degree)
    for local_idx in top_regs:
        global_idx = gene_indices[local_idx]   # index in the 2000-gene pool

        for pert_type, strength in [("KO", 1.0), ("KD_020", 0.2), ("KD_060", 0.6)]:
            sim_p = sergio(number_genes=grn_size, number_bins=5, number_sc=25,
                           noise_type="dpd", decays=0.8, sampling_state=15, dt=0.01, dynamics=False)
            sim_p.build_graph(targets, regs)
            apply_perturbation(sim_p.graph_, local_idx, strength)  # scales basal/edges by (1 - strength)
            sim_p.simulate()
            pert_expr = sim_p.getExpressions()

            # 5. Embed into 2000-gene space and write h5ad
            adata = build_anndata_embedded(
                ctrl_expr, pert_expr,
                global_gene_idx=global_idx,
                pert_type=pert_type,
                gene_indices=gene_indices,
                all_gene_names=GRN_NAMES,
                grn_seed=grn_seed, grn_type=grn_type, grn_params=params,
            )
            preprocess(adata)
            adata.write_h5ad(f"{output_dir}/grn_{grn_seed:04d}/SYN_{global_idx:04d}_{pert_type}.h5ad")
# output_dir = "data/sergio_synthetic/SERGIO_PPT/{grn_type}/size_{grn_size:03d}/noise_{noise_level}"
```

`build_anndata_embedded` places SERGIO expression values at the active gene columns and leaves the other 1800 columns as zero.

### 6.2 Perturbation selection

Use **top-K by out-degree** for both ER and BA graphs. This is robust — ER graphs may have few or no nodes with in-degree 0, but every graph has high-out-degree nodes. **K=7** (`k_perts=7` in `dataset_ppt.json`).

Each KO perturbation's out-degree is stored in `adata.obs["ko_out_degree"]` (set to -1 for non-targeting cells). This enables post-hoc analysis of whether model prediction quality correlates with KO gene influence — hub KOs (high out-degree) produce strong cascades while low-degree KOs have minimal effect. Useful diagnostic for understanding what the model has actually learned before fine-tuning on real data.

### 6.3 Directory structure

```
data/sergio_synthetic/SERGIO_PPT/
  ER/
    size_010/noise_000/grn_0000/  SYN_0042_KO.h5ad  SYN_0042_KD_020.h5ad  SYN_0042_KD_060.h5ad ...
    size_010/noise_010/...
    size_050/...
    size_100/...
  BA/
    size_010/...
    ...
  BA-VM/
    ...
```

### 6.4 SLURM array job

```bash
# Driven by dataset_ppt.json (seeds 0–3, sizes 10/50/100, all three GRN types)
python scripts/generate_sergio_dataset.py \
  --config generation_configs/dataset_ppt.json \
  --output-dir data/sergio_synthetic/SERGIO_PPT

# Key parameters from dataset_ppt.json:
#   pool_size=2000, n_bins=5, n_sc=25, k_perts=7, gene_strategy=random_draw
#   n_seeds=4 (seeds 0–3), seed_offset=0
#   ER: sizes=[10,50,100], p_edge=0.08
#   BA: sizes=[10,50,100], m=2
#   BA-VM: sizes=[10,50,100], m_weights=[0.57,0.29,0.14]
#   noise_levels: none=0.0, low=0.1, high=0.5
#   pert_strengths: KO=1.0, KD_020=0.2, KD_060=0.6
#   sergio_kwargs: noise_type=dpd, decays=0.8, sampling_state=15, dt=0.01, dynamics=false
```

---

## 7. Fixed Test Set

- Seeds disjoint from training (training uses seeds 0–3 per `dataset_ppt.json`)
- Stratified: **ER** × **BA** × **BA-VM**, matched sizes and noise levels to training
- **n_bins = 5**, **n_sc = 25** per bin (matching `dataset_ppt.json`)
- Perturbations per GRN: **KO + KD_020 + KD_060** (matching `dataset_ppt.json`)
- `bin_4` is the zero-shot held-out cell type — withheld from training via TOML `[zeroshot]`
- Preprocessed and stored at generation time; never modified

**Note on bin semantics:** Bins represent different cell types within a single GRN seed. All bins
share the same GRN topology (edges, K, hill coefficients) but differ in their master-regulator
basal rates, producing different steady-state expression profiles. Bin labels (bin_0…bin_4) are
ordinal within a seed — `bin_0` in seed 0 and `bin_0` in seed 5000 are unrelated (different
topology, different basal rates). There is no cross-seed correspondence between same-named bins.

Metrics computed:

- Overall (all instances)
- Per graph type: ER / BA / BA-VM
- Per perturbation type: KO / KD_020 / KD_060
- Zero-shot generalization: `bin_4` (never seen during training) vs bins 0–3 (seen cell types, new seeds)

---

## 8. TOML and Training Command

```toml
# configs/sergio_train.toml
[datasets]
sergio_er = "data/sergio_synthetic/train/ER"
sergio_ba = "data/sergio_synthetic/train/BA"

[training]
sergio_er = "train"
sergio_ba = "train"
```

```bash
state tx train \
  data.kwargs.toml_config_path=configs/sergio_train.toml \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.output_space=gene \
  data.kwargs.perturbation_features_file=configs/pert_onehot_map.pt \
  training.max_steps=40000 \
  training.batch_size=8 \
  model=state \
  model.kwargs.hidden_dim=768 \
  output_dir="$HOME/state_runs" \
  name="sergio_baseline"
```

Model dimensions:

- `input_dim = 2000` (basal_encoder input)
- `output_dim = 2000` (project_out output)
- `pert_dim = 2000` (pert_encoder input, gene-position binary)
- `hidden_dim = 768`

---

## 9. Experiment Sweeps


| Sweep                  | Variable                   | Fixed                       |
| ---------------------- | -------------------------- | --------------------------- |
| Baseline               | —                          | BA, no noise, 500 instances |
| Graph type             | ER vs BA training data     | no noise, 500 instances     |
| Noise                  | dpd ∈ {0, low, high}       | BA, 500 instances           |
| Data quantity          | 100/250/500/1000 instances | BA, no noise                |
| Perturbation diversity | KO / KO+KD / KO+OE         | BA, no noise, 500 instances |


All sweeps evaluated on the fixed test set; metrics stratified by GRN type.

---

## 10. Checkpoint Transfer to Real Data

With 2000 synthetic genes matching Replogle's ~2000 HVGs, `basal_encoder` weights transfer directly:

```bash
state tx train \
  data.kwargs.toml_config_path=configs/replogle.toml \
  data.kwargs.embed_key=X_hvg \
  model.kwargs.init_from=$HOME/state_runs/sergio_baseline/checkpoints/final.ckpt \
  name="replogle_from_sergio"
```

`pert_encoder` is automatically rebuilt when pert_dim differs (real data uses name-based one-hot over ~750 perturbations). Transformer backbone and basal_encoder transfer directly.


