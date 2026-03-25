"""
Sample efficiency curves comparing spptv2, scratch, and nca.

X-axis: cells_per_pert ∈ {10, 25, 50, 100, full}
Y-axis: metric value
Lines: one per model type (spptv2, scratch, nca)
Error bars: SEM across perturbations (pooled across GRNs)

Usage:
    uv run python scripts/plot_sample_efficiency.py \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output results/sample_efficiency.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    "pearson_delta",
    "overlap_at_N",
    "precision_at_N",
    "discrimination_score_l1",
    "mse",
    "de_nsig_counts_pred",
]

METRIC_LABELS = {
    "pearson_delta": "Pearson Δ (↑)",
    "overlap_at_N": "Overlap@N (↑)",
    "precision_at_N": "Precision@N (↑)",
    "discrimination_score_l1": "Discrimination L1 (↑)",
    "mse": "MSE (↓)",
    "de_nsig_counts_pred": "DE Sig Counts Pred",
}

# model_type -> list of (x_label, run_dir_name) in order
CONDITIONS = {
    "spptv2": [
        ("10",   "sample_eff_spptv2_cpp10"),
        ("25",   "sample_eff_spptv2_cpp25"),
        ("50",   "sample_eff_spptv2_cpp50"),
        ("100",  "sample_eff_spptv2_cpp100"),
        ("full", "spptv2_last_stgt"),
    ],
    "scratch": [
        ("10",   "sample_eff_scratch_cpp10"),
        ("25",   "sample_eff_scratch_cpp25"),
        ("50",   "sample_eff_scratch_cpp50"),
        ("100",  "sample_eff_scratch_cpp100"),
        ("full", "sergio_tgtv2"),
    ],
    "nca": [
        ("10",   "sample_eff_nca_cpp10"),
        ("25",   "sample_eff_nca_cpp25"),
        ("50",   "sample_eff_nca_cpp50"),
        ("100",  "sample_eff_nca_cpp100"),
        ("full", "nca_stgt_v2"),
    ],
    "reptile1k": [
        ("10",   "sample_eff_reptile1k_cpp10"),
        ("25",   "sample_eff_reptile1k_cpp25"),
        ("50",   "sample_eff_reptile1k_cpp50"),
        ("100",  "sample_eff_reptile1k_cpp100"),
        ("full", "sample_eff_reptile1k_full"),
    ],
    "reptile3k": [
        ("10",   "sample_eff_reptile3k_cpp10"),
        ("25",   "sample_eff_reptile3k_cpp25"),
        ("50",   "sample_eff_reptile3k_cpp50"),
        ("100",  "sample_eff_reptile3k_cpp100"),
        ("full", "sample_eff_reptile3k_full"),
    ],
    "reptile10k": [
        ("10",   "sample_eff_reptile10k_cpp10"),
        ("25",   "sample_eff_reptile10k_cpp25"),
        ("50",   "sample_eff_reptile10k_cpp50"),
        ("100",  "sample_eff_reptile10k_cpp100"),
        ("full", "sample_eff_reptile10k_full"),
    ],
}

STEP_OVERRIDES: dict[str, str] = {}

MODEL_COLORS = {
    "spptv2":     "#2166ac",
    "scratch":    "#d6604d",
    "nca":        "#4dac26",
    "reptile1k":  "#01bfc4",
    "reptile3k":  "#8073ac",
    "reptile10k": "#e08214",
}

MODEL_LABELS = {
    "spptv2":     "SERGIO-PPT",
    "scratch":    "BASELINE",
    "nca":        "NCA",
    "reptile1k":  "REPT-1k",
    "reptile3k":  "REPT-3k",
    "reptile10k": "REPT-10k",
}

EVAL_STEP = "eval_step=step=8000.ckpt"


def load_eval_dir(eval_dir: Path) -> dict[str, dict]:
    """
    Returns {metric: {"mean": float, "std": float, "sem": float, "ci95": float}}
    pooled across GRNs.
    """
    agg_files = list(eval_dir.glob("*_agg_results.csv"))
    if not agg_files:
        return {}

    grn_means, grn_stds, grn_counts = [], [], []
    for f in agg_files:
        df = pd.read_csv(f, index_col=0)
        if "mean" in df.index:
            grn_means.append(df.loc["mean"])
        if "std" in df.index:
            grn_stds.append(df.loc["std"])
        if "count" in df.index:
            grn_counts.append(df.loc["count"])

    if not grn_means:
        return {}

    means = pd.DataFrame(grn_means).mean()
    pooled_std = pd.DataFrame(grn_stds).pow(2).mean().pow(0.5) if grn_stds else pd.Series(dtype=float)
    total_n = pd.DataFrame(grn_counts).sum() if grn_counts else pd.Series(dtype=float)

    result = {}
    for metric in METRICS + ["de_nsig_counts_real"]:
        if metric in means.index:
            n = total_n.get(metric, float("nan"))
            std = pooled_std.get(metric, float("nan"))
            sem = std / np.sqrt(n) if n > 0 else float("nan")
            ci95 = 1.96 * sem if not np.isnan(sem) else float("nan")
            result[metric] = {"mean": means[metric], "std": std, "sem": sem, "ci95": ci95}
    return result


ERROR_BAR_LABEL = {
    "std":  "±1 SD",
    "sem":  "±SEM",
    "ci95": "±95% CI",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-runs",
                        default="/mnt/polished-lake/home/nkondapaneni/state_runs")
    parser.add_argument("--output", default="results/sample_efficiency/sample_efficiency.png")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--error-bars", default="sem", choices=["std", "sem", "ci95"],
                        help="Error bar type: std, sem, or ci95 (default: sem)")
    parser.add_argument("--step", default=EVAL_STEP,
                        help="Eval dir name to load (default: eval_step=step=8000.ckpt)")
    args = parser.parse_args()

    state_runs = Path(args.state_runs)
    metrics_to_plot = args.metrics
    error_bar_type = args.error_bars

    # Append error bar type to output filename
    output = Path(args.output)
    output = output.with_name(output.stem + f"_{error_bar_type}" + output.suffix)

    # Load all data: data[model_type][x_label] = {metric: {mean, sem}}
    data = {}
    for model_type, conditions in CONDITIONS.items():
        data[model_type] = {}
        step = STEP_OVERRIDES.get(model_type, args.step)
        for x_label, run_name in conditions:
            eval_dir = state_runs / run_name / step
            if not eval_dir.exists():
                print(f"Warning: {eval_dir} not found, skipping {model_type} @ {x_label}")
                continue
            metrics = load_eval_dir(eval_dir)
            if metrics:
                data[model_type][x_label] = metrics
                print(f"  {model_type:8s} cpp={x_label:4s}: {run_name}")
            else:
                print(f"Warning: no data in {eval_dir}")

    x_labels = ["10", "25", "50", "100", "full"]
    x_pos = np.arange(len(x_labels))

    n_metrics = len(metrics_to_plot)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        for model_type in CONDITIONS:
            model_data = data.get(model_type, {})
            xs, ys, errs = [], [], []
            for i, x_label in enumerate(x_labels):
                if x_label in model_data and metric in model_data[x_label]:
                    xs.append(i)
                    ys.append(model_data[x_label][metric]["mean"])
                    errs.append(model_data[x_label][metric][error_bar_type])

            if not xs:
                continue

            color = MODEL_COLORS[model_type]
            ax.errorbar(xs, ys, yerr=errs,
                        label=MODEL_LABELS[model_type],
                        color=color,
                        marker="o", markersize=5,
                        linewidth=1.8,
                        capsize=4,
                        elinewidth=1.2,
                        capthick=1.2)

        # Overlay de_nsig_counts_real as a dashed reference line in the pred subplot
        if metric == "de_nsig_counts_pred":
            # Average real counts across all models (it's a test-set property)
            real_xs, real_ys = [], []
            for model_type in CONDITIONS:
                model_data = data.get(model_type, {})
                for i, x_label in enumerate(x_labels):
                    if x_label in model_data and "de_nsig_counts_real" in model_data[x_label]:
                        real_xs.append(i)
                        real_ys.append(model_data[x_label]["de_nsig_counts_real"]["mean"])
            if real_xs:
                # Average across models per x position
                real_by_x = {}
                for xi, yi in zip(real_xs, real_ys):
                    real_by_x.setdefault(xi, []).append(yi)
                xs_r = sorted(real_by_x)
                ys_r = [np.mean(real_by_x[xi]) for xi in xs_r]
                ax.plot(xs_r, ys_r, color="black", linewidth=1.5,
                        linestyle="--", label="real (ground truth)", zorder=3)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Cells per perturbation")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"Sample Efficiency  ({ERROR_BAR_LABEL[error_bar_type]})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
