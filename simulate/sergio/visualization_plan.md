# Visualization Plan: SERGIO Synthetic → Cell-Eval Evaluation

## Data Layout Reminder

Each eval file (`{grn_type}_{size}_{noise}_{pert_type}_simulated.h5ad`) contains:
- `adata.X` — ground truth log-normalized expression (2000 HVGs)
- `adata.obsm['X_hvg']` — model predictions (written by `state tx infer --embed_key X_hvg`)
- `adata.obs['split']` — `train` (bins 0–3, 10k cells) vs `held_out` (bin_4, 2.5k cells)
- `adata.obs['gene']` — perturbation identity (`non-targeting` = control)
- `adata.obs['grn_type']` — BA, BA-VM, ER (3 groups × 36 files each = 108)
- `adata.obs['grn_size']` — 10, 50, 100 (genes in GRN)
- `adata.obs['noise_label']` — none, low, high
- `adata.obs['pert_type']` — KO, KD_010, KD_050, KD_080, non-targeting

**All evaluations are zero-shot.** Test files use GRN seeds 5000–5024 — completely new topologies
and basal rates, never seen during training (which used seeds 0–24). The `split` column can be
ignored entirely.

**Evaluation unit:** ALL perturbed cells, excluding `non-targeting` controls from metric
calculation (controls serve as the pseudobulk reference for Δ computation).

The meaningful axes are: `grn_type`, `noise_label`, `pert_type`, and `grn_size`.

---

## Core Metrics (from Cell-Eval, Section 4.7)

For each file, compute per-perturbation then aggregate across perturbations:

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Pearson Δ Corr** | Profile shape of effect | `pearsonr(mean(pred_pert) - mean(pred_ctrl), mean(true_pert) - mean(true_ctrl))` across all 2000 genes |
| **Perturbation Discrimination Score (PDisc)** | Can model rank perturbations correctly | Manhattan dist of pred vs all true pseudobulks; normalized rank |
| **DE Overlap @ N** | Biological relevance (gene sets) | Wilcoxon on pred vs ctrl; intersect top-N with true DEGs |
| **Log2FC Spearman** | Direction + magnitude of DE genes | Spearman on LFC for true-significant genes |
| **Effect Size Correlation** | Does model predict strong vs weak perts correctly | Spearman of n_DEGs(pred) vs n_DEGs(true) across perturbations |

**Minimal viable set for a first pass:** Pearson Δ Corr + Log2FC Spearman (both gene-level, fast to compute).

---

## Factor Structure (Experimental Axes)

The 108 test files form a balanced factorial design with one exception:

```
grn_type:   BA (3 sizes × 3 noises × 4 pert_types = 36)
            BA-VM (3 noises × 4 pert_types = 12, no size variation)
            ER  (3 sizes × 3 noises × 4 pert_types = 36)

grn_size:   10, 50, 100  [BA and ER only]
noise:      none, low, high
pert_type:  KO, KD_010, KD_050, KD_080
```

BA-VM has no size variation (only one network topology), so size comparisons must exclude it or treat it separately.

---

## Visualization Plan

### Plot 1 — Overall Performance Heatmap (Summary)
**Purpose:** Single-view scorecard, easy to read.
**Type:** 4-panel 2D heatmap (one per pert_type: KO, KD_010, KD_050, KD_080)
- Rows: `noise_label` (none → low → high)
- Columns: `grn_type` × `grn_size` (ER_010, ER_050, ER_100, BA_010, BA_050, BA_100, BA-VM)
- Cell color: **Pearson Δ Corr** (diverging colormap centered at 0.5)
- Annotate cells with numeric value
- Expect: noise ↑ → harder; KD_010 harder than KO; ER may be harder than BA

### Plot 2 — Noise Effect (Violin / Box Strip)
**Purpose:** Show whether model degrades gracefully with noise (mirrors real data's key challenge).
**Type:** Violin + strip plot, 3 panels (one per grn_type: BA, BA-VM, ER)
- X-axis: `noise_label` (none, low, high)
- Y-axis: Pearson Δ Corr per perturbation (held_out)
- Hue: `pert_type`
- Each violin is all perturbations in that grn_type × noise stratum
- **Balance:** BA-VM only appears in its own panel; BA and ER use all 3 sizes pooled
- Expected story: noise degrades performance; KO should be easier to predict

### Plot 3 — Perturbation Strength Effect
**Purpose:** KD_010 (weakest KD, 10% residual expression) vs KD_080 (near-normal) vs KO — does model handle partial knockdowns?
**Type:** Line plot
- X-axis: pert_type ordered by strength: KO < KD_010 < KD_050 < KD_080
- Y-axis: mean Pearson Δ Corr
- Lines: one per `grn_type` (3 lines), with shaded ±SE
- **Balance:** Pool all noise levels and sizes within each grn_type for each pert_type
- Expected: KO easiest (largest effect), KD_080 hardest (smallest effect signal)

### Plot 4 — GRN Size Effect (BA and ER only)
**Purpose:** Larger GRNs → more complex regulation → harder to predict?
**Type:** Box plot
- X-axis: `grn_size` (10, 50, 100)
- Y-axis: Pearson Δ Corr
- Hue: `grn_type` (BA vs ER, 2 colors)
- **Balance:** Pool all noise levels and pert_types within each size × grn_type cell
- Note: BA-VM excluded (no size variation)

### Plot 5 — DE Gene Recovery (Log2FC Spearman by condition)
**Purpose:** Biological relevance beyond expression correlation — does the model get direction right?
**Type:** Scatter + colorbar
- X-axis: Pearson Δ Corr (per file, aggregated)
- Y-axis: Log2FC Spearman (per file, mean over held_out perturbations)
- Color: `grn_type`
- Shape: `noise_label`
- Each point = one of the 108 test files
- **Balance:** All 108 files plotted; helps spot which grn_type/noise combos are outliers
- Expected: positive correlation between metrics; ER might cluster differently from BA

### Plot 6 — Per-Perturbation Score Distribution (Calibration)
**Purpose:** Are a few perturbations dragging down the average, or is the model uniformly bad/good?
**Type:** Cumulative distribution (ECDF) plot
- X-axis: per-perturbation Pearson Δ Corr
- Y-axis: fraction of perturbations ≤ x
- Lines: one per `grn_type` (pool all conditions)
- **Balance:** Each grn_type has ~36 files × ~50–200 perturbations = thousands of points
- Helps answer: does model fail on a tail of hard perturbations or uniformly?

### Plot 7 — Ablation Comparison (once ablation runs finish)
**Purpose:** Main ablation result — does training data composition matter?
**Type:** Grouped bar chart
- X-axis: ablation run name (9 runs from `run_ablations.sh`)
- Y-axis: mean Pearson Δ Corr on the fixed test set
- Bars grouped by `noise_label` (3 bars per ablation)
- Error bars: SE across perturbations
- **Balance:** Test set is fixed — same 108 files for all ablations; only training data changes

---

## Balancing Strategy

The main imbalance is **BA-VM has no size variation** while BA and ER have 3 sizes. To handle this:

1. **Size-stratified analyses (Plots 1, 4):** Exclude BA-VM or show it separately with a note.
2. **Noise/pert analyses (Plots 2, 3, 6):** Include BA-VM, pool across sizes for BA and ER.
3. **Summary analyses (Plots 5, 7):** All 108 files treated equally (file-level aggregation).
4. **Per-file metrics:** Always compute at the file level first (one number per h5ad), then aggregate. This avoids cell count imbalance (some files have more perturbations than others).

---

## Implementation Notes

### Scoring Script Skeleton

```python
# scripts/score_eval.py
import anndata as ad, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

EVAL_DIR = Path(".../state_runs/sergio_mini_replogle_config_8gpu/eval")

def score_file(path: Path) -> dict:
    adata = ad.read_h5ad(path)
    # Use ALL cells — test files use seeds 5000-5024, entirely separate from training seeds 0-24
    ctrl_mask = adata.obs['gene'] == 'non-targeting'
    ctrl_true = adata[ctrl_mask].X.mean(0)   # pseudobulk control, ground truth
    ctrl_pred = adata[ctrl_mask].obsm['X_hvg'].mean(0)  # pseudobulk control, predicted

    records = []
    for pert in adata.obs['gene'].unique():
        if pert == 'non-targeting':
            continue
        p = adata[adata.obs['gene'] == pert]
        true_delta = p.X.mean(0) - ctrl_true
        pred_delta = p.obsm['X_hvg'].mean(0) - ctrl_pred
        r, _ = pearsonr(true_delta, pred_delta)
        # log2fc spearman on true-significant genes (requires DE step, see below)
        records.append({'perturbation': pert, 'pearson_delta': r, **dict(adata.obs.iloc[0][
            ['grn_type','grn_size','noise_label','pert_type']
        ])})
    return records
```

### Dependencies
- `anndata`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- All available in the `sergio` project venv

---

## Output Files

| File | Description |
|------|-------------|
| `results/scores_per_perturbation.csv` | Per-perturbation metrics for all 108 files |
| `results/scores_per_file.csv` | Per-file aggregated metrics |
| `figures/fig1_heatmap.pdf` | Plot 1 |
| `figures/fig2_noise_violin.pdf` | Plot 2 |
| `figures/fig3_pert_strength.pdf` | Plot 3 |
| `figures/fig4_grn_size.pdf` | Plot 4 |
| `figures/fig5_scatter.pdf` | Plot 5 |
| `figures/fig6_ecdf.pdf` | Plot 6 |
| `figures/fig7_ablation.pdf` | Plot 7 (after ablation runs complete) |
