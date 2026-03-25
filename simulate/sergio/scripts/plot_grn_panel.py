"""
plot_grn_panel.py — 3×3 presentation panel of GRN network visualizations.

Rows = GRN type  (BA, BA-VM, ER)
Cols = GRN size  (10, 50, 100 genes)

One representative GRN is drawn per cell (seed=0 / grn_0000 from SERGIO_PPT).
Uses the same generator functions as the dataset pipeline to reproduce the
exact topologies stored in SERGIO_PPT.

Usage:
    uv run python scripts/plot_grn_panel.py
    uv run python scripts/plot_grn_panel.py --out results/grn_panel.png
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

# Add parent dir so grn_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grn_utils import generate_er_grn, generate_scale_free_grn, generate_ba_vm_grn

# ---------------------------------------------------------------------------
# Constants matching the dataset pipeline
# ---------------------------------------------------------------------------
SEED        = 0
ER_P_EDGE   = 0.08
BA_M        = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)

GRN_TYPES = ["BA", "BA-VM", "ER"]
GRN_SIZES = [10, 50, 100]
GRN_SIZE_LABELS = ["10 genes", "50 genes", "100 genes"]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG_COLOR        = "#0f1117"   # near-black background
MASTER_COLOR    = "#f7c948"   # warm gold  — master regulators
TARGET_COLOR    = "#5b9bd5"   # steel blue — target genes
ACT_COLOR       = "#56c596"   # teal-green — activating edges
REP_COLOR       = "#e05c6a"   # coral-red  — repressing edges
TITLE_COLOR     = "#ffffff"
LABEL_COLOR     = "#e8eaf0"
BORDER_COLOR    = "#2a2d3a"

TYPE_ACCENT = {
    "BA":    "#f7c948",
    "BA-VM": "#a78bfa",
    "ER":    "#38bdf8",
}

TYPE_FULLNAME = {
    "BA":    "Barabási–Albert",
    "BA-VM": "BA Variable-m",
    "ER":    "Erdős–Rényi",
}

# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def make_grn(grn_type: str, n_genes: int) -> nx.DiGraph:
    if grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=BA_M, seed=SEED)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=BA_VM_WEIGHTS, seed=SEED)
    elif grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=ER_P_EDGE, seed=SEED)
    raise ValueError(grn_type)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def hierarchical_layout(G: nx.DiGraph, spread: float = 1.8) -> dict:
    """Top-down hierarchical layout with horizontal spread."""
    from collections import defaultdict
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
            x = (i - (n_layer - 1) / 2.0) * spread
            pos[n] = (x, -d * spread)
    return pos


def layout_for(G: nx.DiGraph, n_genes: int) -> dict:
    if n_genes <= 20:
        return hierarchical_layout(G, spread=2.5)
    elif n_genes <= 60:
        return hierarchical_layout(G, spread=1.6)
    else:
        # For large graphs spring layout is cleaner
        return nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n_genes), iterations=60)


# ---------------------------------------------------------------------------
# Single-cell draw
# ---------------------------------------------------------------------------

def draw_grn(ax: plt.Axes, G: nx.DiGraph, n_genes: int, accent: str) -> None:
    pos = layout_for(G, n_genes)

    master_regs = {n for n in G.nodes() if G.in_degree(n) == 0}
    act_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) > 0]
    rep_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) < 0]

    node_colors = [MASTER_COLOR if n in master_regs else TARGET_COLOR for n in G.nodes()]

    # Scale node sizes inversely with gene count
    base_size = max(30, int(2200 / n_genes))
    node_sizes = [base_size * 1.8 if n in master_regs else base_size for n in G.nodes()]

    # Node glow for master regulators (draw twice: large+transparent then normal)
    master_nodes = [n for n in G.nodes() if n in master_regs]
    if master_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=master_nodes,
                               node_color=MASTER_COLOR,
                               node_size=[base_size * 5] * len(master_nodes),
                               alpha=0.15)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.92,
                           linewidths=0.6,
                           edgecolors="#ffffff30")

    edge_kw = dict(ax=ax, arrows=True, arrowstyle="-|>",
                   connectionstyle="arc3,rad=0.08",
                   min_source_margin=4, min_target_margin=4)

    edge_width = max(0.4, 1.8 - n_genes / 80)
    arrow_size = max(6, 14 - n_genes // 10)

    if act_edges:
        nx.draw_networkx_edges(G, pos, edgelist=act_edges,
                               edge_color=ACT_COLOR, alpha=0.55,
                               width=edge_width, arrowsize=arrow_size, **edge_kw)
    if rep_edges:
        nx.draw_networkx_edges(G, pos, edgelist=rep_edges,
                               edge_color=REP_COLOR, alpha=0.55,
                               width=edge_width, arrowsize=arrow_size, **edge_kw)

    # Labels only for small graphs
    if n_genes <= 15:
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_size=7, font_color=BG_COLOR,
                                font_weight="bold")

    ax.set_facecolor(BG_COLOR)
    # Accent border
    for spine in ax.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/grn_panel.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(GRN_TYPES), len(GRN_SIZES)
    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows), facecolor=BG_COLOR)

    # Add a thin margin so column/row labels have room
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.12, wspace=0.08,
                          left=0.08, right=0.98, top=0.92, bottom=0.04)

    for row, grn_type in enumerate(GRN_TYPES):
        accent = TYPE_ACCENT[grn_type]
        for col, (n_genes, size_label) in enumerate(zip(GRN_SIZES, GRN_SIZE_LABELS)):
            ax = fig.add_subplot(gs[row, col])

            G = make_grn(grn_type, n_genes)
            draw_grn(ax, G, n_genes, accent)

            # Stats annotation (bottom-left corner of axes)
            n_edges = G.number_of_edges()
            n_mr    = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
            stats_txt = f"{n_edges} edges · {n_mr} master regs"
            ax.text(0.03, 0.03, stats_txt,
                    transform=ax.transAxes,
                    fontsize=9, color="#aaaacc", va="bottom", ha="left",
                    path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)])

            # Column headers (top row only)
            if row == 0:
                ax.set_title(size_label, fontsize=17, fontweight="bold",
                             color=LABEL_COLOR, pad=10)

            # Row labels (left column only) — drawn as figure text
            if col == 0:
                # Place row label to the left of the subplot
                x_fig = gs[row, 0].get_position(fig).x0 - 0.055
                y_fig = gs[row, 0].get_position(fig).get_points().mean(axis=0)[1]
                fig.text(x_fig, y_fig,
                         f"{grn_type}\n{TYPE_FULLNAME[grn_type]}",
                         fontsize=14, fontweight="bold",
                         color=accent,
                         ha="center", va="center", rotation=90)

    # Legend (shared, top-right)
    legend_items = [
        mpatches.Patch(facecolor=MASTER_COLOR, label="Master regulator", linewidth=0),
        mpatches.Patch(facecolor=TARGET_COLOR, label="Target gene",       linewidth=0),
        mpatches.Patch(facecolor=ACT_COLOR,    label="Activating edge",   linewidth=0),
        mpatches.Patch(facecolor=REP_COLOR,    label="Repressing edge",   linewidth=0),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.99, 0.99),
               fontsize=12, framealpha=0.15,
               facecolor=BORDER_COLOR, edgecolor="#ffffff30",
               labelcolor=LABEL_COLOR,
               handlelength=1.2, handleheight=1.0,
               borderpad=0.8, labelspacing=0.5)

    # Main title
    fig.text(0.5, 0.965,
             "SERGIO Synthetic GRN Topologies",
             ha="center", va="top",
             fontsize=22, fontweight="bold", color=TITLE_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
