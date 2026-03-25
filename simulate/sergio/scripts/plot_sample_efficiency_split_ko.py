"""
Sample efficiency grid split by knock-out type: one figure per metric.
Rows = KO type (KO, KD_020, KD_060), Columns = GRN size (010, 050, 100).
Each cell shows sample efficiency curves (lines per model, x = cells_per_pert),
with metrics computed only over perturbations of that KO type, pooled across
all GRN types and seeds.

Usage:
    uv run python scripts/plot_sample_efficiency_split_ko.py
    uv run python scripts/plot_sample_efficiency_split_ko.py --error-bars ci95
    uv run python scripts/plot_sample_efficiency_split_ko.py --metrics pearson_delta mse
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

KO_TYPES = ["KO", "KD_060", "KD_020"]
KO_LABELS = {
    "KO":     "Full KO",
    "KD_060": "60% Knockdown",
    "KD_020": "20% Knockdown",
}

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

STEP_OVERRIDES = {}

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

ERROR_BAR_LABEL = {
    "std":  "±1 SD",
    "sem":  "±SEM",
    "ci95": "±95% CI",
}


def get_ko_type(perturbation: str) -> str | None:
    """Extract KO type from perturbation name like SYN_0085_KO or SYN_0085_KD_020."""
    m = re.search(r"_(KO|KD_\d+)$", perturbation)
    return m.group(1) if m else None


def get_grn_size(filename_stem: str) -> str | None:
    """Extract grn_size from stem like BA-VM_size050_seed0100."""
    m = re.search(r"_size(\d+)_", filename_stem)
    return m.group(1) if m else None


def load_eval_dir_by_ko(eval_dir: Path) -> dict[tuple, dict[str, dict]]:
    """
    Returns {(ko_type, grn_size): {metric: {mean, std, sem, ci95}}}
    by reading per-perturbation *_results.csv files and grouping by KO type + GRN size.
    """
    result_files = list(eval_dir.glob("*_results.csv"))
    # Exclude agg files
    result_files = [f for f in result_files if "_agg_results" not in f.name]
    if not result_files:
        return {}

    # Collect all rows grouped by (ko_type, grn_size)
    groups: dict[tuple, list] = {}
    for f in result_files:
        stem = f.stem.replace("_results", "")
        grn_size = get_grn_size(stem)
        if grn_size is None:
            continue

        df = pd.read_csv(f)
        if "perturbation" not in df.columns:
            continue

        for _, row in df.iterrows():
            ko_type = get_ko_type(str(row["perturbation"]))
            if ko_type is None:
                continue
            key = (ko_type, grn_size)
            groups.setdefault(key, []).append(row)

    result = {}
    for key, rows in groups.items():
        if not rows:
            continue
        df_group = pd.DataFrame(rows)
        metrics_out = {}
        all_metrics = set(METRICS) | {"de_nsig_counts_real"}
        for metric in all_metrics:
            if metric not in df_group.columns:
                continue
            vals = df_group[metric].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            n = len(vals)
            sem = std / np.sqrt(n)
            ci95 = 1.96 * sem
            metrics_out[metric] = {"mean": mean, "std": std, "sem": sem, "ci95": ci95}
        if metrics_out:
            result[key] = metrics_out

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-runs",
                        default="/mnt/polished-lake/home/nkondapaneni/state_runs")
    parser.add_argument("--output-dir", default="results/grid/ko_type")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--error-bars", default="ci95", choices=["std", "sem", "ci95"])
    parser.add_argument("--step", default=EVAL_STEP)
    args = parser.parse_args()

    state_runs = Path(args.state_runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # data[model_type][x_label][(ko_type, grn_size)] = {metric: stats}
    data = {}
    for model_type, conditions in CONDITIONS.items():
        data[model_type] = {}
        step = STEP_OVERRIDES.get(model_type, args.step)
        for x_label, run_name in conditions:
            eval_dir = state_runs / run_name / step
            if not eval_dir.exists():
                print(f"Warning: {eval_dir} not found, skipping {model_type} @ {x_label}")
                continue
            ko_data = load_eval_dir_by_ko(eval_dir)
            if ko_data:
                data[model_type][x_label] = ko_data
            else:
                print(f"Warning: no data in {eval_dir}")

    x_labels = ["10", "25", "50", "100", "full"]
    x_pos = np.arange(len(x_labels))
    step_label = args.step.replace("eval_", "").replace(".ckpt", "")
    eb = args.error_bars

    for metric in args.metrics:
        fig, axes = plt.subplots(
            len(KO_TYPES), len(GRN_SIZES),
            figsize=(6 * len(GRN_SIZES), 5 * len(KO_TYPES)),
            sharex=True, sharey=True,
        )

        for row, ko_type in enumerate(KO_TYPES):
            for col, grn_size in enumerate(GRN_SIZES):
                ax = axes[row][col]
                key = (ko_type, grn_size)

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
                    ax.set_ylabel(KO_LABELS[ko_type], fontsize=17, fontweight="bold")
                if row == 0:
                    ax.set_title(f"Size {grn_size}", fontsize=17, fontweight="bold")
                if row == len(KO_TYPES) - 1:
                    ax.set_xlabel("Cells per perturbation", fontsize=15)
                if row == 0 and col == len(GRN_SIZES) - 1:
                    ax.legend(fontsize=12, loc="upper right")

        fig.suptitle(
            f"{METRIC_LABELS.get(metric, metric)}  by KO Type  ({ERROR_BAR_LABEL[eb]})",
            fontsize=20, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"sample_efficiency_grid_ko_{metric}_{eb}.png"
        out_path = output_dir / fname
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
