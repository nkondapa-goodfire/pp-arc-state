"""
Plot eval metrics vs checkpoint step for a list of models.

Usage:
    uv run python scripts/plot_eval_metrics.py \
        --models sergio_tgt spptv1_last_stgt nca_stgt \
        --state-runs /path/to/state_runs \
        --output metrics_vs_step.pdf

Each model must have eval dirs named eval_step=step=<N>.ckpt/ inside its
state_runs directory, containing grn_*_agg_results.csv files.
Metrics are averaged across all GRNs found in each eval dir.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

METRICS = [
    "pearson_delta",
    "overlap_at_N",
    "precision_at_N",
    "discrimination_score_l1",
    "mse",
    "mae",
    "de_nsig_counts_pred",
]

METRIC_LABELS = {
    "pearson_delta": "Pearson Δ (↑)",
    "overlap_at_N": "Overlap@N (↑)",
    "precision_at_N": "Precision@N (↑)",
    "discrimination_score_l1": "Discrimination L1 (↑)",
    "mse": "MSE (↓)",
    "mae": "MAE (↓)",
    "de_nsig_counts_pred": "DE Sig Counts Pred",
}

MODEL_COLORS = [
    "#2166ac",  # blue
    "#d6604d",  # red
    "#4dac26",  # green
    "#8073ac",  # purple
    "#f4a582",  # orange
]


def _agg_dir_to_row(eval_dir: Path, step: int) -> dict | None:
    """Read *_agg_results.csv files from eval_dir and return averaged metric row."""
    agg_files = list(eval_dir.glob("*_agg_results.csv"))
    if not agg_files:
        return None
    grn_means = []
    for agg_file in agg_files:
        df = pd.read_csv(agg_file, index_col=0)
        if "mean" not in df.index:
            continue
        grn_means.append(df.loc["mean"])
    if not grn_means:
        return None
    combined = pd.DataFrame(grn_means).mean()
    row = {"step": step}
    for metric in METRICS:
        if metric in combined.index:
            row[metric] = combined[metric]
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


def load_best_ckpt_metrics(run_dir: Path) -> dict | None:
    """Load metrics from eval_best.ckpt and resolve the step from checkpoint metadata."""
    eval_dir = run_dir / "eval_best.ckpt"
    if not eval_dir.exists():
        return None

    # Resolve step from checkpoint global_step metadata
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    step = None
    if ckpt_path.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            step = ckpt.get("global_step")
        except Exception as e:
            print(f"  Warning: could not load best.ckpt metadata: {e}")

    if step is None:
        return None

    row = _agg_dir_to_row(eval_dir, step)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names (subdirs of --state-runs)",
    )
    parser.add_argument(
        "--state-runs",
        default="/mnt/polished-lake/home/nkondapaneni/state_runs",
        help="Path to state_runs directory",
    )
    parser.add_argument(
        "--output", default="metrics_vs_step.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=METRICS,
        help="Metrics to plot (default: all)",
    )
    args = parser.parse_args()

    state_runs = Path(args.state_runs)
    metrics_to_plot = args.metrics

    # Load data for all models
    model_data = {}
    best_data = {}
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
        print(f"{model}: {len(df)} checkpoints evaluated, steps: {df['step'].tolist()}")

        best = load_best_ckpt_metrics(run_dir)
        if best:
            best_data[model] = best
            print(f"  best.ckpt → step {best['step']}")

    if not model_data:
        print("No data found.")
        return

    n_metrics = len(metrics_to_plot)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        for i, (model, df) in enumerate(model_data.items()):
            if metric not in df.columns:
                continue
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(df["step"], df[metric], marker="o", markersize=4,
                    label=model, color=color, linewidth=1.5)
            # Overlay best.ckpt as a star
            if model in best_data and metric in best_data[model]:
                # print(f"  best.ckpt → step {best_data[model]['step']}")
                bx = best_data[model]["step"]
                by = best_data[model][metric]
                ax.plot(bx, by, marker="*", markersize=3, color="black",
                        linestyle="none", zorder=5)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("Step")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Eval Metrics vs Training Step", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
