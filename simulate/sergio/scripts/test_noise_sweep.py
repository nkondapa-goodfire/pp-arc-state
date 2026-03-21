"""
test_noise_sweep.py — Sweep SERGIO dpd noise levels across a few ER and BA GRNs.

For each graph × noise level, simulate control + one KO (top out-degree gene)
and record:
  - sparsity      : fraction of cells with zero expression per gene, averaged
  - mean_expr     : mean expression across all genes and cells
  - expr_cv       : coefficient of variation (std/mean) across cells, per gene
  - ko_effect     : mean |log2FC| across all genes except the KO gene

Outputs
-------
test_outputs/noise_sweep/
  summary.csv                         — per (grn, noise_level) statistics
  {er,ba}_seed{N}_noise_sweep.png     — 4-panel summary per graph
  {er,ba}_seed{N}_pca_sweep.png       — PCA grid: one panel per noise level
  all_graphs_noise_summary.png        — aggregate across all graphs

Usage
-----
    uv run python test_noise_sweep.py
"""

import csv
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from SERGIO.sergio import sergio
from grn_utils import (
    generate_er_grn,
    generate_scale_free_grn,
    generate_ba_vm_grn,
    grn_to_sergio_files,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_GENES  = 50
N_BINS   = 5
N_SC     = 100

ER_SEEDS      = [0, 1]
BA_SEEDS      = [100, 101]
BA_VM_SEEDS   = [200, 201]
ER_P_EDGE     = 0.08
BA_M          = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]

BASE_KWARGS = dict(
    noise_type="dpd",
    decays=0.8,
    sampling_state=15,
    dt=0.01,
    dynamics=False,
)

OUT_DIR = "test_outputs/noise_sweep"


# ---------------------------------------------------------------------------
# SERGIO helpers
# ---------------------------------------------------------------------------

def run_simulation(G: nx.DiGraph, noise_params: float,
                   n_genes: int, n_bins: int, n_sc: int,
                   tmpdir: str, ko_gene: int | None = None) -> np.ndarray:
    """Run one SERGIO simulation (control or KO). Returns (n_genes, n_bins*n_sc)."""
    G_sim = G.copy()
    if ko_gene is not None:
        for succ in list(G_sim.successors(ko_gene)):
            G_sim.remove_edge(ko_gene, succ)
        for pred in list(G_sim.predecessors(ko_gene)):
            G_sim.remove_edge(pred, ko_gene)

    seed = 1 if ko_gene is not None else 0
    targets_path, regs_path = grn_to_sergio_files(G_sim, tmpdir, n_bins=n_bins, seed=seed)

    # Zero basal rate for KO gene if it appears in the Regs file (master-reg case)
    if ko_gene is not None:
        rows = []
        with open(regs_path, newline="") as f:
            for row in csv.reader(f):
                if int(float(row[0])) == ko_gene:
                    row = [row[0]] + ["0.0"] * n_bins
                rows.append(row)
        with open(regs_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    sim = sergio(
        number_genes=n_genes,
        number_bins=n_bins,
        number_sc=n_sc,
        noise_params=noise_params,
        **BASE_KWARGS,
    )
    sim.build_graph(targets_path, regs_path, shared_coop_state=2)
    sim.simulate()
    expr = sim.getExpressions()
    return np.concatenate(expr, axis=1)  # (n_genes, n_bins * n_sc)


def top_out_degree_gene(G: nx.DiGraph) -> int:
    return max(G.nodes(), key=lambda n: G.out_degree(n))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ctrl: np.ndarray, ko: np.ndarray, ko_gene: int) -> dict:
    sparsity   = (ctrl == 0).mean()
    mean_expr  = ctrl.mean()
    # CV per gene, then averaged (exclude genes with mean=0 to avoid div/0)
    gene_mean  = ctrl.mean(axis=1)
    gene_std   = ctrl.std(axis=1)
    valid      = gene_mean > 0
    cv         = (gene_std[valid] / gene_mean[valid]).mean() if valid.any() else 0.0

    ctrl_mean  = ctrl.mean(axis=1)
    ko_mean    = ko.mean(axis=1)
    log2fc     = np.log2((ko_mean + 1e-6) / (ctrl_mean + 1e-6))
    other      = np.arange(len(log2fc)) != ko_gene
    ko_effect  = np.abs(log2fc[other]).mean()

    return dict(sparsity=sparsity, mean_expr=mean_expr, cv=cv, ko_effect=ko_effect)


# ---------------------------------------------------------------------------
# Per-graph noise sweep
# ---------------------------------------------------------------------------

def sweep_graph(G: nx.DiGraph, label: str, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    ko_gene   = top_out_degree_gene(G)
    n_genes   = G.number_of_nodes()
    records   = []

    metrics_by_noise = []

    for noise in NOISE_LEVELS:
        print(f"  {label}  noise={noise}")
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = run_simulation(G, noise, n_genes, N_BINS, N_SC, tmpdir)
            ko   = run_simulation(G, noise, n_genes, N_BINS, N_SC, tmpdir, ko_gene=ko_gene)

        m = compute_metrics(ctrl, ko, ko_gene)
        m["label"]      = label
        m["noise"]      = noise
        m["ko_gene"]    = ko_gene
        m["ko_outdeg"]  = G.out_degree(ko_gene)
        records.append(m)
        metrics_by_noise.append((noise, m, ctrl, ko))

    _plot_graph_sweep(label, ko_gene, metrics_by_noise, out_dir)
    plot_pca_sweep(label, metrics_by_noise, out_dir)
    return records


def plot_pca_sweep(label: str, metrics_by_noise: list, out_dir: str) -> None:
    """One PCA scatter per noise level, coloured by cell type."""
    n_noise = len(metrics_by_noise)
    fig, axes = plt.subplots(1, n_noise, figsize=(4 * n_noise, 4), squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("tab10", N_BINS)
    cell_labels = np.repeat(np.arange(N_BINS), N_SC)

    for ax, (noise, _, ctrl, _ko) in zip(axes, metrics_by_noise):
        X = np.log1p(ctrl.T)                          # (n_cells, n_genes)
        X = StandardScaler().fit_transform(X)
        n_comp = min(2, X.shape[1])
        coords = PCA(n_components=n_comp, random_state=42).fit_transform(X)

        for b in range(N_BINS):
            mask = cell_labels == b
            y = coords[mask, 1] if n_comp > 1 else np.zeros(mask.sum())
            ax.scatter(coords[mask, 0], y, c=[cmap(b)], s=8,
                       alpha=0.6, linewidths=0, label=f"bin {b}")

        ax.set_title(f"noise={noise}", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    axes[-1].legend(fontsize=7, markerscale=2, loc="best")
    fig.suptitle(f"{label}  —  PCA by noise level (control expression)", fontsize=11)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("=", "")
    fig.savefig(os.path.join(out_dir, f"{safe_label}_pca_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_graph_sweep(label: str, ko_gene: int,
                      metrics_by_noise: list, out_dir: str) -> None:
    """4-panel plot: sparsity / mean_expr / cv / ko_effect vs noise level."""
    noises     = [x[0] for x in metrics_by_noise]
    sparsities = [x[1]["sparsity"]  for x in metrics_by_noise]
    means      = [x[1]["mean_expr"] for x in metrics_by_noise]
    cvs        = [x[1]["cv"]        for x in metrics_by_noise]
    effects    = [x[1]["ko_effect"] for x in metrics_by_noise]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles  = ["Sparsity (frac zeros)", "Mean expression", "Mean CV (expr)", "Mean |log2FC| KO"]
    series  = [sparsities, means, cvs, effects]
    colors  = ["#E74C3C", "#4A90D9", "#F4A827", "#27AE60"]

    for ax, title, vals, color in zip(axes, titles, series, colors):
        ax.plot(noises, vals, "o-", color=color, linewidth=2, markersize=6)
        ax.set_xlabel("noise_params (dpd)")
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label}  —  noise sweep  (KO gene {ko_gene})", fontsize=11)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("=", "")
    fig.savefig(os.path.join(out_dir, f"{safe_label}_noise_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate summary plot
# ---------------------------------------------------------------------------

def plot_aggregate(records: list[dict], out_dir: str) -> None:
    """Line plot of each metric vs noise, one line per graph, coloured by type."""
    noises = sorted(set(r["noise"] for r in records))
    labels = sorted(set(r["label"] for r in records))
    metrics = ["sparsity", "mean_expr", "cv", "ko_effect"]
    metric_titles = ["Sparsity (frac zeros)", "Mean expression",
                     "Mean CV (expr)", "Mean |log2FC| KO"]

    color_map = {"ER": "#4A90D9", "BA": "#F4A827", "BA-VM": "#27AE60"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric, title in zip(axes, metrics, metric_titles):
        for lbl in labels:
            grn_type = lbl.split(" ")[0]
            color = color_map.get(grn_type, "#888888")
            vals  = [r[metric] for r in records if r["label"] == lbl and r["noise"] in noises]
            ax.plot(noises, vals, "o-", color=color, alpha=0.75, linewidth=1.8,
                    markersize=5, label=lbl)
        ax.set_xlabel("noise_params (dpd)")
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)

    # Unified legend (deduplicated by type)
    handles, seen = [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append((h, l))
                seen.add(l)
    axes[-1].legend(*zip(*handles), fontsize=7, loc="upper right")

    fig.suptitle("Noise sweep — all graphs", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_graphs_noise_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {len(ER_SEEDS)} ER graphs (n_genes={N_GENES}, p_edge={ER_P_EDGE})...")
    er_graphs = [(generate_er_grn(N_GENES, ER_P_EDGE, seed=s), f"ER seed={s}")
                 for s in ER_SEEDS]

    print(f"Generating {len(BA_SEEDS)} BA graphs (n_genes={N_GENES}, m={BA_M})...")
    ba_graphs = [(generate_scale_free_grn(N_GENES, BA_M, seed=s), f"BA seed={s}")
                 for s in BA_SEEDS]

    print(f"Generating {len(BA_VM_SEEDS)} BA-VM graphs (n_genes={N_GENES}, weights={BA_VM_WEIGHTS})...")
    ba_vm_graphs = [(generate_ba_vm_grn(N_GENES, m_weights=BA_VM_WEIGHTS, seed=s), f"BA-VM seed={s}")
                    for s in BA_VM_SEEDS]

    all_records = []
    for G, label in er_graphs + ba_graphs + ba_vm_graphs:
        print(f"\nSweeping noise for {label}...")
        records = sweep_graph(G, label, OUT_DIR)
        all_records.extend(records)

    # Summary CSV
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = ["label", "noise", "ko_gene", "ko_outdeg",
                  "sparsity", "mean_expr", "cv", "ko_effect"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    plot_aggregate(all_records, OUT_DIR)

    print(f"\nOutputs written to: {os.path.abspath(OUT_DIR)}")
    print(f"Summary CSV:        {csv_path}")

    # Print table
    print()
    print(f"{'label':<14} {'noise':>6} {'sparsity':>10} {'mean_expr':>10} {'cv':>8} {'ko_effect':>10}")
    print("─" * 64)
    for r in all_records:
        print(f"{r['label']:<14} {r['noise']:>6.2f} {r['sparsity']:>10.3f} "
              f"{r['mean_expr']:>10.3f} {r['cv']:>8.3f} {r['ko_effect']:>10.3f}")


if __name__ == "__main__":
    main()
