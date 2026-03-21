"""
test_generation.py — Sample ER and BA GRNs, run SERGIO simulations, and write
visualizations + statistics to test_outputs/.

Outputs
-------
test_outputs/
  stats_summary.csv              — per-graph statistics table
  er/
    grn_{seed}_network.png       — GRN network (nodes coloured by role)
    grn_{seed}_degrees.png       — in/out-degree distributions
    grn_{seed}_expression.png    — expression heatmap + PCA
    grn_{seed}_ko_effect.png     — control vs KO mean expression
  ba/
    (same structure)

Usage
-----
    uv run python test_generation.py
    uv run python test_generation.py --n-samples 5 --n-genes 100
"""

import argparse
import csv
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA

from SERGIO.sergio import sergio
from grn_utils import (
    generate_er_grn,
    generate_scale_free_grn,
    generate_ba_vm_grn,
    grn_summary,
    grn_to_sergio_files,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_SAMPLES  = 3    # graphs per type
N_GENES    = 50   # keep small for fast testing
N_BINS     = 5    # cell types
N_SC       = 100  # cells per cell type
K_PERTS    = 3    # KO perturbations per GRN

ER_P_EDGE  = 0.08  # ~2× target post-DAG density of ~0.04 for 50 genes
BA_M       = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)  # geometric(p=0.5) truncated to {1,2,3}

SERGIO_KWARGS = dict(
    noise_params=0.1,
    noise_type="dpd",
    decays=0.8,
    sampling_state=15,
    dt=0.01,
    dynamics=False,
)


# ---------------------------------------------------------------------------
# SERGIO helpers
# ---------------------------------------------------------------------------

def run_control(G: nx.DiGraph, n_genes: int, n_bins: int, n_sc: int,
                tmpdir: str) -> np.ndarray:
    """Simulate control expression. Returns (n_genes, n_bins * n_sc)."""
    targets_path, regs_path = grn_to_sergio_files(G, tmpdir, n_bins=n_bins, seed=0)
    sim = sergio(number_genes=n_genes, number_bins=n_bins, number_sc=n_sc,
                 **SERGIO_KWARGS)
    sim.build_graph(targets_path, regs_path, shared_coop_state=2)
    sim.simulate()
    expr = sim.getExpressions()          # list of (n_genes, n_sc) per bin
    return np.concatenate(expr, axis=1)  # (n_genes, n_bins * n_sc)


def run_ko(G: nx.DiGraph, ko_gene: int, n_genes: int, n_bins: int, n_sc: int,
           tmpdir: str) -> np.ndarray:
    """
    Simulate KO of ``ko_gene``. Returns (n_genes, n_bins * n_sc).

    A proper knockout does three things:
      1. Removes the gene's outgoing regulatory edges (stops driving downstream genes).
      2. Removes the gene's incoming regulatory edges (its production rate becomes 0
         since SERGIO computes non-master-reg expression purely from regulators).
      3. Zeros its basal rate in the Regs file (handles the master-regulator case
         where the gene has no regulators but has a sampled basal rate).

    Removing both in- and out-edges is necessary: removing only outgoing edges
    leaves the gene's own expression unchanged when it has upstream regulators.
    """
    G_ko = G.copy()
    for succ in list(G_ko.successors(ko_gene)):
        G_ko.remove_edge(ko_gene, succ)
    for pred in list(G_ko.predecessors(ko_gene)):
        G_ko.remove_edge(pred, ko_gene)

    targets_path, regs_path = grn_to_sergio_files(G_ko, tmpdir, n_bins=n_bins, seed=1)

    # Zero basal rate for the KO gene in the Regs file
    rows = []
    with open(regs_path, newline="") as f:
        for row in csv.reader(f):
            if int(float(row[0])) == ko_gene:
                row = [row[0]] + ["0.0"] * n_bins
            rows.append(row)
    with open(regs_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    sim = sergio(number_genes=n_genes, number_bins=n_bins, number_sc=n_sc,
                 **SERGIO_KWARGS)
    sim.build_graph(targets_path, regs_path, shared_coop_state=2)
    sim.simulate()
    expr = sim.getExpressions()
    return np.concatenate(expr, axis=1)


def top_k_out_degree(G: nx.DiGraph, k: int) -> list[int]:
    """Return the k nodes with highest out-degree."""
    return sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)[:k]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def hierarchical_layout(G: nx.DiGraph) -> dict:
    master_regs = {n for n in G.nodes() if G.in_degree(n) == 0}
    layers: dict[int, int] = {}
    visited: set[int] = set()
    queue = list(master_regs & set(G.nodes()))
    depth = 0
    while queue:
        next_q = []
        for n in queue:
            if n not in visited:
                visited.add(n)
                layers[n] = depth
                next_q.extend(G.successors(n))
        queue = next_q
        depth += 1
    max_depth = max(layers.values(), default=0)
    for n in G.nodes():
        if n not in layers:
            layers[n] = max_depth + 1

    by_layer: dict[int, list] = defaultdict(list)
    for n, d in layers.items():
        by_layer[d].append(n)

    pos = {}
    for d, nodes in by_layer.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (i - len(nodes) / 2.0, -d)
    return pos


def plot_network(G: nx.DiGraph, title: str, out_path: str) -> None:
    master_regs = {n for n in G.nodes() if G.in_degree(n) == 0}
    act_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) > 0]
    rep_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) < 0]

    pos = hierarchical_layout(G) if G.number_of_nodes() <= 200 else \
          nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

    node_color = ["#F4A827" if n in master_regs else "#4A90D9" for n in G.nodes()]
    n = G.number_of_nodes()
    node_size = max(60, 1000 // n * 5)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#F8F8F8")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                           node_size=node_size, alpha=0.9)
    arrow_kw = dict(ax=ax, arrows=True, arrowstyle="-|>", arrowsize=8,
                    connectionstyle="arc3,rad=0.05",
                    min_source_margin=4, min_target_margin=4)
    nx.draw_networkx_edges(G, pos, edgelist=act_edges,
                           edge_color="#27AE60", alpha=0.6, width=1.0, **arrow_kw)
    nx.draw_networkx_edges(G, pos, edgelist=rep_edges,
                           edge_color="#E74C3C", alpha=0.6, width=1.0, **arrow_kw)
    if n <= 80:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#1a1a1a")

    legend_items = [
        mpatches.Patch(color="#F4A827", label="Master regulator"),
        mpatches.Patch(color="#4A90D9", label="Target gene"),
        mpatches.Patch(color="#27AE60", label="Activating"),
        mpatches.Patch(color="#E74C3C", label="Repressing"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_degrees(G: nx.DiGraph, title: str, out_path: str) -> None:
    in_deg  = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, degrees, label, color in [
        (axes[0], in_deg,  "In-degree",  "#4A90D9"),
        (axes[1], out_deg, "Out-degree", "#F4A827"),
    ]:
        max_d = max(degrees) if degrees else 1
        bins = range(0, max_d + 2)
        ax.hist(degrees, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} distribution")
        ax.axvline(np.mean(degrees), color="red", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(degrees):.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expression(expr: np.ndarray, n_bins: int, n_sc: int,
                    title: str, out_path: str) -> None:
    n_genes, n_cells = expr.shape
    cell_labels = np.repeat(np.arange(n_bins), n_sc)
    cmap_ct = plt.get_cmap("tab10", n_bins)

    # Top variable genes for heatmap
    n_show = min(30, n_genes)
    top_idx = np.argsort(expr.var(axis=1))[::-1][:n_show]
    cell_order = np.argsort(cell_labels, kind="stable")
    expr_sorted = expr[np.ix_(top_idx, cell_order)]
    labels_sorted = cell_labels[cell_order]

    expr_log = np.log1p(expr_sorted)
    row_std = expr_log.std(axis=1, keepdims=True) + 1e-8
    expr_z = (expr_log - expr_log.mean(axis=1, keepdims=True)) / row_std

    # PCA
    from sklearn.preprocessing import StandardScaler
    X_log = np.log1p(expr.T)
    X_scaled = StandardScaler().fit_transform(X_log)
    n_comp = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(expr_z, aspect="auto", cmap="RdBu_r",
                        vmin=-2.5, vmax=2.5, interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="Z-score")

    bar_h = max(1, n_show // 15)
    ax_heat.imshow(np.array([labels_sorted]), aspect="auto",
                   cmap=cmap_ct, vmin=0, vmax=n_bins - 1,
                   extent=[-0.5, n_cells - 0.5, -bar_h - 0.5, -0.5],
                   interpolation="nearest")
    for b in range(n_bins - 1):
        ax_heat.axvline((b + 1) * n_sc - 0.5, color="white", lw=0.6, alpha=0.6)
    ax_heat.set_xlim(-0.5, n_cells - 0.5)
    ax_heat.set_ylim(n_show - 0.5, -bar_h - 0.5)
    ax_heat.set_xlabel("Cells (sorted by cell type)")
    ax_heat.set_ylabel(f"Top {n_show} variable genes")
    ax_heat.set_yticks([])
    ax_heat.legend(
        handles=[mpatches.Patch(color=cmap_ct(b), label=f"Type {b}") for b in range(n_bins)],
        loc="upper right", fontsize=7, framealpha=0.8,
    )
    ax_heat.set_title("Expression heatmap (log1p, z-scored)")

    ax_pca = fig.add_subplot(gs[1])
    for b in range(n_bins):
        mask = cell_labels == b
        y = coords[mask, 1] if n_comp > 1 else np.zeros(mask.sum())
        ax_pca.scatter(coords[mask, 0], y, c=[cmap_ct(b)], s=8,
                       alpha=0.6, label=f"Type {b}", linewidths=0)
    ax_pca.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)" if n_comp > 1 else "")
    ax_pca.set_title("PCA — coloured by cell type")
    ax_pca.legend(fontsize=7, markerscale=2, framealpha=0.8)
    ax_pca.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ko_effect(ctrl: np.ndarray, ko: np.ndarray, ko_gene: int,
                   n_bins: int, n_sc: int, title: str, out_path: str) -> None:
    """Bar chart of mean expression change (KO - ctrl) per gene, plus heatmap."""
    # Mean per gene across all cells
    ctrl_mean = ctrl.mean(axis=1)
    ko_mean   = ko.mean(axis=1)
    delta     = ko_mean - ctrl_mean
    log2fc    = np.log2((ko_mean + 1e-6) / (ctrl_mean + 1e-6))

    n_genes = ctrl.shape[0]
    gene_idx = np.arange(n_genes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: log2FC bar chart
    ax = axes[0]
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in log2fc]
    ax.bar(gene_idx, log2fc, color=colors, alpha=0.8, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(ko_gene, color="orange", lw=1.5, linestyle="--",
               label=f"KO gene ({ko_gene})")
    ax.set_xlabel("Gene index")
    ax.set_ylabel("log2FC (KO / ctrl)")
    ax.set_title("Per-gene log2 fold change")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: ctrl vs KO mean expression scatter
    ax2 = axes[1]
    ax2.scatter(ctrl_mean, ko_mean, s=20, alpha=0.6, color="#4A90D9")
    ax2.scatter(ctrl_mean[ko_gene], ko_mean[ko_gene], s=80,
                color="orange", zorder=5, label=f"KO gene ({ko_gene})")
    lim = max(ctrl_mean.max(), ko_mean.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="y=x")
    ax2.set_xlabel("Control mean expression")
    ax2.set_ylabel("KO mean expression")
    ax2.set_title("Control vs KO expression")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_graph_type(grn_type: str, graphs: list[nx.DiGraph], seeds: list[int],
                   out_dir: str, n_bins: int, n_sc: int) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for G, seed in zip(graphs, seeds):
        print(f"  [{grn_type}] seed={seed}")
        stats = grn_summary(G)
        stats["grn_type"] = grn_type
        stats["seed"] = seed

        prefix = os.path.join(out_dir, f"grn_{seed}")
        n_genes = G.number_of_nodes()

        # Network plot
        plot_network(
            G,
            title=f"{grn_type} GRN  seed={seed} | {stats['n_genes']} genes  "
                  f"{stats['n_edges']} edges  {stats['n_master_regulators']} master regs  "
                  f"{stats['n_levels']} levels",
            out_path=f"{prefix}_network.png",
        )

        # Degree distribution
        plot_degrees(
            G,
            title=f"{grn_type} GRN  seed={seed} — degree distributions",
            out_path=f"{prefix}_degrees.png",
        )

        # SERGIO simulation
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"    running control simulation...")
            ctrl = run_control(G, n_genes, n_bins, n_sc, tmpdir)

            plot_expression(
                ctrl, n_bins, n_sc,
                title=f"{grn_type} GRN  seed={seed} — control expression",
                out_path=f"{prefix}_expression.png",
            )

            # KO perturbation for the top out-degree gene
            ko_gene = top_k_out_degree(G, 1)[0]
            print(f"    running KO simulation (gene {ko_gene})...")
            ko_expr = run_ko(G, ko_gene, n_genes, n_bins, n_sc, tmpdir)

        plot_ko_effect(
            ctrl, ko_expr, ko_gene, n_bins, n_sc,
            title=f"{grn_type} GRN  seed={seed} — KO gene {ko_gene} effect",
            out_path=f"{prefix}_ko_effect.png",
        )

        stats["ko_gene"] = ko_gene
        n_deg = len([g for g in G.nodes() if G.in_degree(g) == 0])
        stats["n_true_master_regs"] = n_deg
        records.append(stats)
        print(f"    done. levels={stats['n_levels']}  edges={stats['n_edges']}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-genes",   type=int, default=N_GENES)
    parser.add_argument("--n-bins",    type=int, default=N_BINS)
    parser.add_argument("--n-sc",      type=int, default=N_SC)
    parser.add_argument("--out-dir",   default="test_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Generate graphs ---
    er_seeds    = list(range(args.n_samples))
    ba_seeds    = list(range(100, 100 + args.n_samples))
    ba_vm_seeds = list(range(200, 200 + args.n_samples))

    print(f"Generating {args.n_samples} ER graphs (n_genes={args.n_genes}, p_edge={ER_P_EDGE})...")
    er_graphs = [
        generate_er_grn(args.n_genes, p_edge=ER_P_EDGE, seed=s)
        for s in er_seeds
    ]

    print(f"Generating {args.n_samples} BA graphs (n_genes={args.n_genes}, m={BA_M})...")
    ba_graphs = [
        generate_scale_free_grn(args.n_genes, n_edges_per_new_node=BA_M, seed=s)
        for s in ba_seeds
    ]

    print(f"Generating {args.n_samples} BA-VM graphs (n_genes={args.n_genes}, weights={BA_VM_WEIGHTS})...")
    ba_vm_graphs = [
        generate_ba_vm_grn(args.n_genes, m_weights=BA_VM_WEIGHTS, seed=s)
        for s in ba_vm_seeds
    ]

    # --- Run simulations and produce plots ---
    print("\nRunning ER simulations...")
    er_records = run_graph_type(
        "ER", er_graphs, er_seeds,
        out_dir=os.path.join(args.out_dir, "er"),
        n_bins=args.n_bins, n_sc=args.n_sc,
    )

    print("\nRunning BA simulations...")
    ba_records = run_graph_type(
        "BA", ba_graphs, ba_seeds,
        out_dir=os.path.join(args.out_dir, "ba"),
        n_bins=args.n_bins, n_sc=args.n_sc,
    )

    print("\nRunning BA-VM simulations...")
    ba_vm_records = run_graph_type(
        "BA-VM", ba_vm_graphs, ba_vm_seeds,
        out_dir=os.path.join(args.out_dir, "ba_vm"),
        n_bins=args.n_bins, n_sc=args.n_sc,
    )

    # --- Summary statistics CSV ---
    all_records = er_records + ba_records + ba_vm_records
    csv_path = os.path.join(args.out_dir, "stats_summary.csv")
    fieldnames = [
        "grn_type", "seed", "n_genes", "n_edges", "n_master_regulators",
        "n_true_master_regs", "n_activating", "n_repressing",
        "mean_in_degree", "max_in_degree", "max_out_degree",
        "n_levels", "is_dag", "ko_gene",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # --- Print summary table ---
    print(f"\n{'─'*80}")
    print(f"{'type':<6} {'seed':<6} {'genes':<7} {'edges':<7} {'levels':<8} "
          f"{'m_regs':<8} {'mean_in':<9} {'is_dag'}")
    print(f"{'─'*80}")
    for r in all_records:
        print(f"{r['grn_type']:<6} {r['seed']:<6} {r['n_genes']:<7} "
              f"{r['n_edges']:<7} {r['n_levels']:<8} "
              f"{r['n_master_regulators']:<8} {r['mean_in_degree']:<9.2f} "
              f"{r['is_dag']}")
    print(f"{'─'*80}")
    print(f"\nOutputs written to: {os.path.abspath(args.out_dir)}")
    print(f"Stats CSV:          {csv_path}")


if __name__ == "__main__":
    main()
