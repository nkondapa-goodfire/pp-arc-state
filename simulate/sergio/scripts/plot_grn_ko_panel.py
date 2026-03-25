"""
plot_grn_ko_panel.py — 3×3 GRN network panel after knockout of the highest
out-degree gene.

Same 9 GRNs as plot_grn_panel.py (seed=0, grn_0000).  For each GRN the
top hub gene is removed and the resulting network is drawn. The removed gene
is annotated in the title of each panel.

Usage:
    uv run python scripts/plot_grn_ko_panel.py
    uv run python scripts/plot_grn_ko_panel.py --out results/datasets/grn_ko_panel.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grn_utils import generate_er_grn, generate_scale_free_grn, generate_ba_vm_grn
from generate_dataset import sample_grn_genes

# ---------------------------------------------------------------------------
# Constants — must match generate_dataset.py and plot_grn_panel.py
# ---------------------------------------------------------------------------
SEED          = 0
POOL_SIZE     = 2000
ER_P_EDGE     = 0.08
BA_M          = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)

GRN_TYPES      = ["BA", "BA-VM", "ER"]
GRN_SIZES      = [10, 50, 100]
GRN_SIZE_LABELS = ["10 genes", "50 genes", "100 genes"]

BG_COLOR     = "#0f1117"
MASTER_COLOR = "#f7c948"
TARGET_COLOR = "#5b9bd5"
ACT_COLOR    = "#56c596"
REP_COLOR    = "#e05c6a"
KO_COLOR     = "#ff3b3b"
LABEL_COLOR  = "#e8eaf0"
TITLE_COLOR  = "#ffffff"
BORDER_COLOR = "#2a2d3a"

TYPE_ACCENT = {"BA": "#f7c948", "BA-VM": "#a78bfa", "ER": "#38bdf8"}
TYPE_FULLNAME = {
    "BA":    "Barabási–Albert",
    "BA-VM": "BA Variable-m",
    "ER":    "Erdős–Rényi",
}


# ---------------------------------------------------------------------------
# Graph generation + KO selection
# ---------------------------------------------------------------------------

def make_grn(grn_type: str, n_genes: int) -> nx.DiGraph:
    if grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=BA_M, seed=SEED)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=BA_VM_WEIGHTS, seed=SEED)
    elif grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=ER_P_EDGE, seed=SEED)
    raise ValueError(grn_type)


def top_hub(G: nx.DiGraph) -> int:
    return max(G.nodes(), key=lambda n: G.out_degree(n))


def node_to_syn(local_idx: int, n_genes: int) -> str:
    gene_indices = sample_grn_genes(POOL_SIZE, n_genes, SEED)
    return f"SYN_{gene_indices[local_idx]:04d}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def hierarchical_layout(G: nx.DiGraph, spread: float = 1.8) -> dict:
    master_regs = {n for n in G.nodes() if G.in_degree(n) == 0}
    layers: dict = {}
    visited: set = set()
    queue = list(master_regs)
    depth = 0
    while queue:
        nxt = []
        for n in queue:
            if n not in visited:
                visited.add(n)
                layers[n] = depth
                nxt.extend(G.successors(n))
        queue = nxt
        depth += 1
    max_d = max(layers.values(), default=0)
    for n in G.nodes():
        if n not in layers:
            layers[n] = max_d + 1

    by_layer: dict = defaultdict(list)
    for n, d in layers.items():
        by_layer[d].append(n)

    pos = {}
    for d, nodes in by_layer.items():
        n_layer = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            pos[n] = ((i - (n_layer - 1) / 2.0) * spread, -d * spread)
    return pos


def layout_for(G: nx.DiGraph, n_genes: int) -> dict:
    if n_genes <= 20:
        return hierarchical_layout(G, spread=2.5)
    elif n_genes <= 60:
        return hierarchical_layout(G, spread=1.6)
    else:
        return nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n_genes), iterations=60)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_grn_ko(ax: plt.Axes, G: nx.DiGraph, ko_node: int,
                ko_syn: str, n_genes: int, accent: str) -> None:
    G_ko = G.copy()
    G_ko.remove_node(ko_node)

    # Compute layout on the reduced graph
    pos = layout_for(G_ko, n_genes)

    master_regs = {n for n in G_ko.nodes() if G_ko.in_degree(n) == 0}
    act_edges   = [(u, v) for u, v, d in G_ko.edges(data=True) if d.get("sign", 1) > 0]
    rep_edges   = [(u, v) for u, v, d in G_ko.edges(data=True) if d.get("sign", 1) < 0]

    node_colors = [MASTER_COLOR if n in master_regs else TARGET_COLOR for n in G_ko.nodes()]
    base_size   = max(30, int(2200 / n_genes))
    node_sizes  = [base_size * 1.8 if n in master_regs else base_size for n in G_ko.nodes()]

    # Glow on master regulators
    master_nodes = [n for n in G_ko.nodes() if n in master_regs]
    if master_nodes:
        nx.draw_networkx_nodes(G_ko, pos, ax=ax, nodelist=master_nodes,
                               node_color=MASTER_COLOR,
                               node_size=[base_size * 5] * len(master_nodes),
                               alpha=0.15)

    nx.draw_networkx_nodes(G_ko, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92,
                           linewidths=0.6, edgecolors="#ffffff30")

    edge_kw = dict(ax=ax, arrows=True, arrowstyle="-|>",
                   connectionstyle="arc3,rad=0.08",
                   min_source_margin=4, min_target_margin=4)
    edge_width  = max(0.4, 1.8 - n_genes / 80)
    arrow_size  = max(6, 14 - n_genes // 10)

    if act_edges:
        nx.draw_networkx_edges(G_ko, pos, edgelist=act_edges,
                               edge_color=ACT_COLOR, alpha=0.55,
                               width=edge_width, arrowsize=arrow_size, **edge_kw)
    if rep_edges:
        nx.draw_networkx_edges(G_ko, pos, edgelist=rep_edges,
                               edge_color=REP_COLOR, alpha=0.55,
                               width=edge_width, arrowsize=arrow_size, **edge_kw)

    if n_genes <= 15:
        nx.draw_networkx_labels(G_ko, pos, ax=ax,
                                font_size=7, font_color=BG_COLOR,
                                font_weight="bold")

    # KO annotation in corner
    ko_text = f"KO: {ko_syn}\n(out-deg {G.out_degree(ko_node)})"
    ax.text(0.03, 0.97, ko_text,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=KO_COLOR, va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)])

    # Stats
    n_edges = G_ko.number_of_edges()
    n_mr    = len(master_regs)
    ax.text(0.03, 0.03, f"{G_ko.number_of_nodes()} genes · {n_edges} edges · {n_mr} master regs",
            transform=ax.transAxes, fontsize=9, color="#aaaacc", va="bottom", ha="left",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)])

    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(KO_COLOR)
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/datasets/grn_ko_panel.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(GRN_TYPES), len(GRN_SIZES)
    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows), facecolor=BG_COLOR)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.12, wspace=0.08,
                          left=0.08, right=0.98, top=0.92, bottom=0.04)

    for row, grn_type in enumerate(GRN_TYPES):
        for col, (n_genes, size_label) in enumerate(zip(GRN_SIZES, GRN_SIZE_LABELS)):
            ax = fig.add_subplot(gs[row, col])

            G       = make_grn(grn_type, n_genes)
            ko_node = top_hub(G)
            ko_syn  = node_to_syn(ko_node, n_genes)

            draw_grn_ko(ax, G, ko_node, ko_syn, n_genes, TYPE_ACCENT[grn_type])

            if row == 0:
                ax.set_title(size_label, fontsize=17, fontweight="bold",
                             color=LABEL_COLOR, pad=10)
            if col == 0:
                x_fig = gs[row, 0].get_position(fig).x0 - 0.055
                y_fig = gs[row, 0].get_position(fig).get_points().mean(axis=0)[1]
                fig.text(x_fig, y_fig,
                         f"{grn_type}\n{TYPE_FULLNAME[grn_type]}",
                         fontsize=14, fontweight="bold",
                         color=TYPE_ACCENT[grn_type],
                         ha="center", va="center", rotation=90)

    # Shared legend
    legend_items = [
        mpatches.Patch(facecolor=MASTER_COLOR, label="Master regulator", linewidth=0),
        mpatches.Patch(facecolor=TARGET_COLOR, label="Target gene",      linewidth=0),
        mpatches.Patch(facecolor=ACT_COLOR,    label="Activating edge",  linewidth=0),
        mpatches.Patch(facecolor=REP_COLOR,    label="Repressing edge",  linewidth=0),
        mpatches.Patch(facecolor=KO_COLOR,     label="KO gene (removed)", linewidth=0),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=12,
               framealpha=0.15, facecolor=BORDER_COLOR, edgecolor="#ffffff30",
               labelcolor=LABEL_COLOR, handlelength=1.2, handleheight=1.0,
               borderpad=0.8, labelspacing=0.5)

    fig.text(0.5, 0.965,
             "SERGIO Synthetic GRN Topologies — Hub Gene Knockout",
             ha="center", va="top", fontsize=22, fontweight="bold", color=TITLE_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
