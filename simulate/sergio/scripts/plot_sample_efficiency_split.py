"""
Sample efficiency grid: one figure per metric.
Rows = GRN type (BA, BA-VM, ER), Columns = GRN size (010, 050, 100).
Each cell shows sample efficiency curves (lines per model, x = cells_per_pert).

Usage:
    uv run python scripts/plot_sample_efficiency_split.py
    uv run python scripts/plot_sample_efficiency_split.py --error-bars ci95
    uv run python scripts/plot_sample_efficiency_split.py --metrics pearson_delta mse
"""

import argparse
import re
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

GRN_TYPES = ["BA", "BA-VM", "ER"]
GRN_SIZES = ["010", "050", "100"]

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
    "reptile1k": [
        ("10",   "sample_eff_reptile1k_cpp10"),
        ("25",   "sample_eff_reptile1k_cpp25"),
        ("50",   "sample_eff_reptile1k_cpp50"),
        ("100",  "sample_eff_reptile1k_cpp100"),
        ("full", "sample_eff_reptile1k_full"),
    ],
}

STEP_OVERRIDES = {}

MODEL_COLORS = {
    "spptv2":     "#2166ac",
    "scratch":    "#d6604d",
    "nca":        "#4dac26",
    "reptile3k":  "#8073ac",
    "reptile10k": "#e08214",
    "reptile1k":  "#01bfc4",
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

ERROR_BAR_LABEL = {
    "std":  "±1 SD",
    "sem":  "±SEM",
    "ci95": "±95% CI",
}


def parse_stem(stem: str) -> tuple[str, str]:
    """BA_size010_seed0100 -> (grn_type='BA', grn_size='010')"""
    grn_type = stem.split("_size")[0]
    m = re.search(r"_size(\d+)_", stem)
    grn_size = m.group(1) if m else "unknown"
    return grn_type, grn_size


def load_eval_dir(eval_dir: Path) -> dict[tuple, dict[str, dict]]:
    """
    Returns {(grn_type, grn_size): {metric: {mean, std, sem, ci95}}}
    one entry per GRN file, grouped by (type, size).
    """
    agg_files = list(eval_dir.glob("*_agg_results.csv"))
    if not agg_files:
        return {}

    # Group by (grn_type, grn_size)
    groups: dict[tuple, list] = {}
    for f in agg_files:
        stem = f.stem.replace("_agg_results", "")
        key = parse_stem(stem)
        groups.setdefault(key, []).append(f)

    result = {}
    for key, files in groups.items():
        grn_means, grn_stds, grn_counts = [], [], []
        for f in files:
            df = pd.read_csv(f, index_col=0)
            if "mean" in df.index:
                grn_means.append(df.loc["mean"])
            if "std" in df.index:
                grn_stds.append(df.loc["std"])
            if "count" in df.index:
                grn_counts.append(df.loc["count"])

        if not grn_means:
            continue

        means = pd.DataFrame(grn_means).mean()
        pooled_std = pd.DataFrame(grn_stds).pow(2).mean().pow(0.5) if grn_stds else pd.Series(dtype=float)
        total_n = pd.DataFrame(grn_counts).sum() if grn_counts else pd.Series(dtype=float)

        metrics_out = {}
        for metric in set(METRICS) | {"de_nsig_counts_real"}:
            if metric in means.index:
                n = total_n.get(metric, float("nan"))
                std = pooled_std.get(metric, float("nan"))
                sem = std / np.sqrt(n) if n > 0 else float("nan")
                ci95 = 1.96 * sem if not np.isnan(sem) else float("nan")
                metrics_out[metric] = {"mean": means[metric], "std": std, "sem": sem, "ci95": ci95}
        result[key] = metrics_out

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-runs",
                        default="/mnt/polished-lake/home/nkondapaneni/state_runs")
    parser.add_argument("--output-dir", default="results/grid/grn_type")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--error-bars", default="sem", choices=["std", "sem", "ci95"])
    parser.add_argument("--step", default=EVAL_STEP)
    args = parser.parse_args()

    state_runs = Path(args.state_runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # data[model_type][x_label][(grn_type, grn_size)] = {metric: stats}
    data = {}
    for model_type, conditions in CONDITIONS.items():
        data[model_type] = {}
        step = STEP_OVERRIDES.get(model_type, args.step)
        for x_label, run_name in conditions:
            eval_dir = state_runs / run_name / step
            if not eval_dir.exists():
                print(f"Warning: {eval_dir} not found, skipping {model_type} @ {x_label}")
                continue
            grn_data = load_eval_dir(eval_dir)
            if grn_data:
                data[model_type][x_label] = grn_data
            else:
                print(f"Warning: no data in {eval_dir}")

    x_labels = ["10", "25", "50", "100", "full"]
    x_pos = np.arange(len(x_labels))
    step_label = args.step.replace("eval_", "").replace(".ckpt", "")
    eb = args.error_bars

    for metric in args.metrics:
        fig, axes = plt.subplots(
            len(GRN_TYPES), len(GRN_SIZES),
            figsize=(6 * len(GRN_SIZES), 5 * len(GRN_TYPES)),
            sharex=True, sharey=True,
        )

        for row, grn_type in enumerate(GRN_TYPES):
            for col, grn_size in enumerate(GRN_SIZES):
                ax = axes[row][col]
                key = (grn_type, grn_size)

                for model_type in CONDITIONS:
                    model_data = data.get(model_type, {})
                    xs, ys, errs = [], [], []
                    for i, x_label in enumerate(x_labels):
                        entry = model_data.get(x_label, {}).get(key)
                        if entry and metric in entry:
                            xs.append(i)
                            ys.append(entry[metric]["mean"])
                            errs.append(entry[metric][eb])

                    if not xs:
                        continue

                    ax.errorbar(xs, ys, yerr=errs,
                                label=MODEL_LABELS[model_type],
                                color=MODEL_COLORS[model_type],
                                marker="o", markersize=8,
                                linewidth=2.5, capsize=5,
                                elinewidth=1.6, capthick=1.6)

                # Real DE reference line
                if metric == "de_nsig_counts_pred":
                    real_by_x = {}
                    for model_type in CONDITIONS:
                        for i, x_label in enumerate(x_labels):
                            entry = data.get(model_type, {}).get(x_label, {}).get(key)
                            if entry and "de_nsig_counts_real" in entry:
                                real_by_x.setdefault(i, []).append(
                                    entry["de_nsig_counts_real"]["mean"]
                                )
                    if real_by_x:
                        xs_r = sorted(real_by_x)
                        ys_r = [np.mean(real_by_x[xi]) for xi in xs_r]
                        ax.plot(xs_r, ys_r, color="black", linewidth=1.6,
                                linestyle="--", label="real", zorder=3)

                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=14)
                ax.tick_params(axis="y", labelsize=13)
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)

                if col == 0:
                    ax.set_ylabel(grn_type, fontsize=17, fontweight="bold")
                if row == 0:
                    ax.set_title(f"Size {grn_size}", fontsize=17, fontweight="bold")
                if row == len(GRN_TYPES) - 1:
                    ax.set_xlabel("Cells per perturbation", fontsize=15)
                if row == 0 and col == len(GRN_SIZES) - 1:
                    ax.legend(fontsize=12, loc="upper right")

        fig.suptitle(
            f"{METRIC_LABELS.get(metric, metric)}  ({ERROR_BAR_LABEL[eb]})",
            fontsize=20, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"sample_efficiency_grid_{metric}_{eb}.png"
        out_path = output_dir / fname
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
