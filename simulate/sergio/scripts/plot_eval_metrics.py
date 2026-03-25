"""
Plot eval metrics vs checkpoint step for a list of models.

Usage:
    uv run python scripts/plot_eval_metrics.py \
        --models sergio_tgtv2 spptv2_last_stgt nca_stgt_v2 \
        --state-runs /path/to/state_runs \
        --output metrics_vs_step.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import METRIC_LABELS, run_color, run_label

METRICS = [
    "pearson_delta",
    "overlap_at_N",
    "precision_at_N",
    "discrimination_score_l1",
    "mse",
    "de_nsig_counts_pred",
]


def _agg_dir_to_row(eval_dir: Path, step: int) -> dict | None:
    """Read *_agg_results.csv files from eval_dir and return mean + error stats per metric."""
    agg_files = list(eval_dir.glob("*_agg_results.csv"))
    if not agg_files:
        return None
    grn_means, grn_stds, grn_counts = [], [], []
    for agg_file in agg_files:
        df = pd.read_csv(agg_file, index_col=0)
        if "mean" in df.index:
            grn_means.append(df.loc["mean"])
        if "std" in df.index:
            grn_stds.append(df.loc["std"])
        if "count" in df.index:
            grn_counts.append(df.loc["count"])
    if not grn_means:
        return None
    means      = pd.DataFrame(grn_means).mean()
    pooled_std = pd.DataFrame(grn_stds).pow(2).mean().pow(0.5) if grn_stds else pd.Series(dtype=float)
    total_n    = pd.DataFrame(grn_counts).sum() if grn_counts else pd.Series(dtype=float)

    row = {"step": step}
    for metric in METRICS:
        if metric not in means.index:
            continue
        n    = total_n.get(metric, float("nan"))
        std  = pooled_std.get(metric, float("nan"))
        sem  = std / np.sqrt(n) if n > 0 else float("nan")
        ci95 = 1.96 * sem if not np.isnan(sem) else float("nan")
        row[metric]              = means[metric]
        row[f"{metric}_std"]     = std
        row[f"{metric}_ci95"]    = ci95
    return row


def load_model_metrics(run_dir: Path) -> pd.DataFrame:
    """Load mean metrics across all GRNs for each evaluated checkpoint."""
    records = []
    for eval_dir in sorted(run_dir.glob("eval_step=step=*.ckpt")):
        m = re.search(r"step=step=(\d+)", eval_dir.name)
        if not m:
            continue
        row = _agg_dir_to_row(eval_dir, int(m.group(1)))
        if row:
            records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("step").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names (subdirs of --state-runs)")
    parser.add_argument("--state-runs",
                        default="/mnt/polished-lake/home/nkondapaneni/state_runs")
    parser.add_argument("--output", default="metrics_vs_step.png")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--error-bars", default="both", choices=["ci95", "std", "both"])
    args = parser.parse_args()

    state_runs = Path(args.state_runs)

    model_data = {}
    for model in args.models:
        run_dir = state_runs / model
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping.")
            continue
        df = load_model_metrics(run_dir)
        if df.empty:
            print(f"Warning: no eval results found for {model}, skipping.")
            continue
        model_data[model] = df
        print(f"{model}: {len(df)} checkpoints, steps: {df['step'].tolist()}")

    if not model_data:
        print("No data found.")
        return

    metrics_to_plot = args.metrics
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eb_modes = ["ci95", "std"] if args.error_bars == "both" else [args.error_bars]
    eb_labels = {"ci95": "±95% CI", "std": "±1 SD"}

    for eb in eb_modes:
        _save_plot(model_data, metrics_to_plot, eb, eb_labels[eb], out_path)


def _save_plot(model_data, metrics_to_plot, eb, eb_label, base_path):
    n_metrics = len(metrics_to_plot)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        err_col = f"{metric}_{eb}"
        for model, df in model_data.items():
            if metric not in df.columns:
                continue
            color = run_color(model)
            xs = df["step"].values
            ys = df[metric].values
            errs = df[err_col].values if err_col in df.columns else None
            ax.errorbar(xs, ys, yerr=errs,
                        label=run_label(model), color=color,
                        marker="o", markersize=5, linewidth=2.0,
                        capsize=4, elinewidth=1.4, capthick=1.4)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))
        ax.set_xscale("log")
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"Eval Metrics vs Training Step  ({eb_label})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    stem = base_path.stem
    suffix = base_path.suffix
    out_path = base_path.with_name(f"{stem}_{eb}{suffix}")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
