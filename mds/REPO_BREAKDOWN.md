# State Repo Breakdown

Notes on how the codebase works, discovered through code reading.

---

## Eval profiles (`state tx predict --profile`)

Source files:

- `.venv/lib/python3.11/site-packages/cell_eval/metrics/_impl.py` — metric registry (all registered metrics)
- `.venv/lib/python3.11/site-packages/cell_eval/_pipeline/_runner.py` — profile definitions (`MINIMAL_METRICS`, profile match block)

### `minimal`

Hardcoded list in `_runner.py:11-19`:


| Metric                    | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `pearson_delta`           | Pearson correlation of predicted vs real Δ from control |
| `mse`                     | Mean squared error                                      |
| `mae`                     | Mean absolute error                                     |
| `discrimination_score_l1` | L1 normalized rank similarity of pred to real           |
| `overlap_at_N`            | Top-N DE gene overlap between predicted and real        |
| `precision_at_N`          | Precision of top-N predicted DE genes                   |
| `de_nsig_counts`          | Count of significant DE genes recovered                 |


### `full`

Everything in `minimal` plus all `MetricType.DE` and `MetricType.ANNDATA_PAIR` registered metrics:


| Extra DE metrics                        | Extra ANNDATA_PAIR metrics       |
| --------------------------------------- | -------------------------------- |
| `overlap/precision_at_{50,100,200,500}` | `mse_delta`, `mae_delta`         |
| `de_spearman_sig`                       | `discrimination_score_l2/cosine` |
| `de_direction_match`                    | `pearson_edistance`              |
| `de_spearman_lfc_sig`                   | `clustering_agreement`           |
| `de_sig_genes_recall`                   |                                  |
| `pr_auc`, `roc_auc`                     |                                  |


### Other profiles


| Profile   | What it runs                                     |
| --------- | ------------------------------------------------ |
| `de`      | Only `MetricType.DE` metrics                     |
| `anndata` | Only `MetricType.ANNDATA_PAIR` metrics           |
| `vcc`     | `mae`, `discrimination_score_l1`, `overlap_at_N` |


### Notes

- `minimal` is appropriate for checkpoint sweeps (fast)
- `full` runs DE testing and clustering per perturbation — expensive at many checkpoints
- `pearson_edistance` and `clustering_agreement` are skipped in the pseudobulk path (`_predict.py:602`)

---

## Metric descriptions

All metrics compare predicted vs real post-perturbation gene expression, relative to a shared control population.

### Cell-level (ANNDATA_PAIR) — no DE required


| Metric                    | Best | What it measures                                                                                                                                                                                                                                                                                                 |
| ------------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pearson_delta`           | ↑ 1  | Pearson correlation between mean(pred) − mean(ctrl) and mean(real) − mean(ctrl) across all genes. High = model correctly ranks which genes go up/down and by how much. Most interpretable single number.                                                                                                         |
| `mse`                     | ↓ 0  | Mean squared error between predicted and real expression per perturbation, averaged across genes. Penalises large errors heavily.                                                                                                                                                                                |
| `mae`                     | ↓ 0  | Mean absolute error. Like MSE but linear — less sensitive to outlier genes.                                                                                                                                                                                                                                      |
| `mse_delta`               | ↓ 0  | MSE computed on the Δ (perturbation − control) rather than raw expression. Focuses on the perturbation effect, not the baseline.                                                                                                                                                                                 |
| `mae_delta`               | ↓ 0  | MAE on the Δ.                                                                                                                                                                                                                                                                                                    |
| `discrimination_score_l1` | ↑ 1  | For each perturbation, rank all perturbations by L1 distance to the predicted centroid; discrimination score = normalised rank of the true real centroid. 1 = predicted centroid is closest to its own real centroid. Measures whether predictions are directionally correct across the full perturbation space. |
| `pearson_edistance`       | ↑ 1  | Pearson correlation between predicted and real energy distances from the control distribution. Measures distributional similarity, not just means.                                                                                                                                                               |
| `clustering_agreement`    | ↑ 1  | Agreement between clusters of predicted and real perturbation centroids. Measures whether the model preserves the topology of the perturbation space.                                                                                                                                                            |


### DE-based — requires differential expression (pdex)

DE is computed per perturbation vs control (Wilcoxon, FDR-corrected). Significant genes = FDR < 0.05.


| Metric                | Best | What it measures                                                                                                                                                                  |
| --------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `overlap_at_N`        | ↑ 1  | Jaccard-style overlap between top-N predicted DE genes and top-N real DE genes. N = number of significant genes in real data. Tests whether the model identifies the right genes. |
| `precision_at_N`      | ↑ 1  | Fraction of top-N predicted DE genes that are truly significant. Similar to overlap_at_N but focuses on predicted precision.                                                      |
| `de_nsig_counts`      | —    | Raw count of significant DE genes in predicted vs real. Not a quality metric per se — large gap between pred/real counts indicates over- or under-prediction of effect magnitude. |
| `de_spearman_sig`     | ↑ 1  | Spearman correlation of per-gene significance scores (−log10 p-values) between pred and real.                                                                                     |
| `de_spearman_lfc_sig` | ↑ 1  | Spearman correlation of log fold-changes among significant genes only. Checks whether predicted effect sizes are ordered correctly.                                               |
| `de_direction_match`  | ↑ 1  | Fraction of significant DE genes where predicted and real agree on direction (up vs down).                                                                                        |
| `de_sig_genes_recall` | ↑ 1  | Recall of real significant genes in the predicted significant set.                                                                                                                |
| `pr_auc`              | ↑ 1  | Area under precision-recall curve for recovering significant genes.                                                                                                               |
| `roc_auc`             | ↑ 1  | Area under ROC curve for recovering significant genes.                                                                                                                            |


### Practical priority

For a perturbation biology model, the most biologically meaningful metrics are:

1. `pearson_delta` — does the model know which genes change?
2. `overlap_at_N` / `precision_at_N` — does it identify the right DE genes?
3. `discrimination_score_l1` — does it distinguish between perturbations?
4. `mse`/`mae` — how accurate are the predicted expression values?

