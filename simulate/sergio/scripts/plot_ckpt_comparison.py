"""
Compare checkpoint evaluations as grouped bar charts with error bars.

Each bar = one checkpoint eval dir. Error bars = 95% CI across perturbations,
averaged across GRNs. Pairwise Wilcoxon signed-rank tests (Bonferroni-corrected)
are annotated on each subplot. One subplot per metric.

Usage:
    uv run python scripts/plot_ckpt_comparison.py \
        --evals model_a:path/to/eval_dir model_b:path/to/eval_dir \
        --output results/comparison.png

    # Or use shorthand <model>:<step> to resolve from --state-runs:
    uv run python scripts/plot_ckpt_comparison.py \
        --evals nca_stgt:best sergio_tgt:step=40000 \
        --state-runs /mnt/polished-lake/home/nkondapaneni/state_runs \
        --output results/comparison.png

Each eval dir must contain *_agg_results.csv and *_results.csv files (one per GRN).
"""

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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

BAR_COLORS = [
    "#2166ac",
    "#d6604d",
    "#4dac26",
    "#8073ac",
    "#f4a582",
    "#1a9850",
    "#d73027",
]


def load_eval_dir(eval_dir: Path) -> tuple[dict[str, dict], dict[str, pd.Series]]:
    """
    Returns:
      summary: {metric: {"mean": float, "ci95": float}}
      per_pert: {metric: Series indexed by perturbation (concatenated across GRNs)}
    """
    agg_files = sorted(eval_dir.glob("*_agg_results.csv"))
    results_files = sorted(eval_dir.glob("*_results.csv"))

    if not agg_files:
        return {}, {}

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
        return {}, {}

    means = pd.DataFrame(grn_means).mean()
    pooled_std = pd.DataFrame(grn_stds).pow(2).mean().pow(0.5) if grn_stds else pd.Series(dtype=float)
    total_n = pd.DataFrame(grn_counts).sum() if grn_counts else pd.Series(dtype=float)

    summary = {}
    for metric in METRICS:
        if metric in means.index:
            n = total_n[metric] if metric in total_n.index else float("nan")
            std = pooled_std[metric] if metric in pooled_std.index else float("nan")
            ci95 = 1.96 * std / np.sqrt(n) if n > 0 else float("nan")
            summary[metric] = {"mean": means[metric], "ci95": ci95}

    # Load per-perturbation values, using grn_name+perturbation as index
    per_pert_frames = []
    for rf in results_files:
        grn_name = rf.stem.replace("_results", "")
        df = pd.read_csv(rf)
        if "perturbation" not in df.columns:
            continue
        df["_key"] = grn_name + "/" + df["perturbation"].astype(str)
        df = df.set_index("_key")
        per_pert_frames.append(df)

    per_pert: dict[str, pd.Series] = {}
    if per_pert_frames:
        all_perts = pd.concat(per_pert_frames)
        for metric in METRICS:
            if metric in all_perts.columns:
                per_pert[metric] = all_perts[metric].dropna()

    return summary, per_pert


def resolve_eval_dir(spec: str, state_runs: Path) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(f"Each --evals entry must be label:path_or_shorthand, got: {spec!r}")

    label, path_or_shorthand = spec.split(":", 1)

    candidate = Path(path_or_shorthand)
    if candidate.is_absolute() or candidate.exists():
        return label, candidate

    run_dir = state_runs / label
    if path_or_shorthand == "best":
        return label, run_dir / "eval_best.ckpt"

    m = re.fullmatch(r"(?:step=)?(\d+)", path_or_shorthand)
    if m:
        return label, run_dir / f"eval_step=step={m.group(1)}.ckpt"

    raise ValueError(
        f"Cannot resolve shorthand {path_or_shorthand!r} for label {label!r}. "
        "Use 'best', 'step=<N>', '<N>', or a full path."
    )


def pvalue_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_significance(ax, pairs, bar_positions, pvalues, bar_height_map, y_top):
    """Draw bracket + stars above bars for each significant pair."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    y_range = y_top - ax.get_ylim()[0]
    step = y_range * 0.08
    base_y = y_top + y_range * 0.02

    for k, ((i, j), p) in enumerate(zip(pairs, pvalues)):
        stars = pvalue_stars(p)
        x1, x2 = bar_positions[i], bar_positions[j]
        y = base_y + k * step

        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                color="black", linewidth=0.8)
        ax.text((x1 + x2) / 2, y + step * 0.35, stars,
                ha="center", va="bottom", fontsize=8, color="black")

    # Expand y limit to fit annotations
    ax.set_ylim(ax.get_ylim()[0], base_y + n_pairs * step + step * 0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Bar chart comparison of checkpoint eval results with significance."
    )
    parser.add_argument("--evals", nargs="+", required=True,
                        metavar="LABEL:PATH_OR_SHORTHAND")
    parser.add_argument("--state-runs",
                        default="/mnt/polished-lake/home/nkondapaneni/state_runs")
    parser.add_argument("--output", default="ckpt_comparison.png")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    args = parser.parse_args()

    state_runs = Path(args.state_runs)
    metrics_to_plot = args.metrics

    entries: list[tuple[str, dict, dict]] = []
    for spec in args.evals:
        label, eval_dir = resolve_eval_dir(spec, state_runs)
        if not eval_dir.exists():
            print(f"Warning: {eval_dir} does not exist, skipping {label!r}.")
            continue
        summary, per_pert = load_eval_dir(eval_dir)
        if not summary:
            print(f"Warning: no data found in {eval_dir}, skipping {label!r}.")
            continue
        entries.append((label, summary, per_pert))
        print(f"Loaded {label!r} from {eval_dir} "
              f"({len(summary)} metrics, {len(next(iter(per_pert.values()), pd.Series()))} perturbations)")

    if not entries:
        print("No data loaded. Exiting.")
        return

    labels = [e[0] for e in entries]
    n_models = len(labels)
    n_metrics = len(metrics_to_plot)
    n_pairs = n_models * (n_models - 1) // 2
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    x = np.arange(n_models)
    bar_width = 0.65

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        means, ci95s = [], []
        for _, summary, _ in entries:
            if metric in summary:
                means.append(summary[metric]["mean"])
                ci95s.append(summary[metric]["ci95"])
            else:
                means.append(float("nan"))
                ci95s.append(float("nan"))

        colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(n_models)]
        bars = ax.bar(x, means, width=bar_width, yerr=ci95s, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2})

        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(s for s in ci95s if not np.isnan(s)) * 0.05
                                        if any(not np.isnan(s) for s in ci95s) else 0),
                    f"{mean_val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#222222",
                )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, max(ymax, 0.5))

        # Pairwise Wilcoxon tests on matched perturbations, Bonferroni-corrected
        pairs = list(itertools.combinations(range(n_models), 2))
        raw_pvalues = []
        valid_pairs = []
        for i, j in pairs:
            s_i = entries[i][2].get(metric)
            s_j = entries[j][2].get(metric)
            if s_i is None or s_j is None:
                continue
            shared = s_i.index.intersection(s_j.index)
            if len(shared) < 10:
                continue
            a, b = s_i[shared].values, s_j[shared].values
            if np.allclose(a, b):
                continue
            _, p = wilcoxon(a, b, alternative="two-sided")
            raw_pvalues.append(p)
            valid_pairs.append((i, j))

        # Bonferroni correction
        corrected = [min(p * len(raw_pvalues), 1.0) for p in raw_pvalues]

        bar_tops = [bar.get_height() + ci for bar, ci in zip(bars, ci95s)
                    if not np.isnan(bar.get_height())]
        y_top = max(bar_tops) if bar_tops else ax.get_ylim()[1]
        annotate_significance(ax, valid_pairs, x, corrected, None, y_top)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Checkpoint Eval Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
