"""
plot_grn_ko_expression_panel.py — 3×3 expression heatmap after hub-gene KO.

Same 9 GRNs and same KO gene as plot_grn_ko_panel.py (highest out-degree hub,
seed=0 / grn_0000 / noise_none).  Shows KO cells sorted by bin with the
knocked-out gene's row annotated with a red ✕.

Usage:
    uv run python scripts/plot_grn_ko_expression_panel.py
    uv run python scripts/plot_grn_ko_expression_panel.py --out results/datasets/grn_ko_expression_panel.png
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grn_utils import generate_er_grn, generate_scale_free_grn, generate_ba_vm_grn
from generate_dataset import sample_grn_genes

# ---------------------------------------------------------------------------
# Constants — must match generate_dataset.py and plot_grn_ko_panel.py
# ---------------------------------------------------------------------------
SEED          = 0
POOL_SIZE     = 2000
ER_P_EDGE     = 0.08
BA_M          = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)

GRN_TYPES     = ["BA", "BA-VM", "ER"]
GRN_SIZES     = [10, 50, 100]
GRN_SIZE_DIRS = ["size_010", "size_050", "size_100"]
SIZE_LABELS   = ["10 genes", "50 genes", "100 genes"]

DATA_ROOT = Path(__file__).resolve().parent.parent / \
            "data/sergio_synthetic/SERGIO_PPT"

BG_COLOR    = "#0f1117"
LABEL_COLOR = "#e8eaf0"
TITLE_COLOR = "#ffffff"
BORDER_COLOR = "#2a2d3a"
KO_COLOR    = "#ff3b3b"

TYPE_ACCENT = {"BA": "#f7c948", "BA-VM": "#a78bfa", "ER": "#38bdf8"}
TYPE_FULLNAME = {
    "BA":    "Barabási–Albert",
    "BA-VM": "BA Variable-m",
    "ER":    "Erdős–Rényi",
}


# ---------------------------------------------------------------------------
# KO gene selection — same logic as plot_grn_ko_panel.py
# ---------------------------------------------------------------------------

def make_grn(grn_type: str, n_genes: int):
    if grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=BA_M, seed=SEED)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=BA_VM_WEIGHTS, seed=SEED)
    elif grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=ER_P_EDGE, seed=SEED)
    raise ValueError(grn_type)


def get_ko_syn(grn_type: str, n_genes: int) -> tuple[str, int]:
    """Return (SYN name, out-degree) for the top hub KO gene."""
    G = make_grn(grn_type, n_genes)
    ko_node = max(G.nodes(), key=lambda n: G.out_degree(n))
    ko_deg  = G.out_degree(ko_node)
    gene_indices = sample_grn_genes(POOL_SIZE, n_genes, SEED)
    ko_syn = f"SYN_{gene_indices[ko_node]:04d}"
    return ko_syn, ko_deg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ko_expression(grn_type: str, size_dir: str, n_genes: int):
    """
    Returns:
        X      : (n_grn_genes, n_ko_cells) log1p z-scored, rows sorted by
                 control-cell variance (high → low)
        bins   : bin labels for each cell
        ko_row : row index of the KO gene in X (None if not found)
        ko_syn : SYN name of the KO gene
        ko_deg : out-degree of the KO gene
    """
    ko_syn, ko_deg = get_ko_syn(grn_type, n_genes)

    ko_file = DATA_ROOT / grn_type / size_dir / "noise_none" / "grn_0000" / f"{ko_syn}_KO.h5ad"
    if not ko_file.exists():
        raise FileNotFoundError(f"KO file not found: {ko_file}")

    adata = ad.read_h5ad(ko_file)

    # Identify GRN genes from control cells
    ctrl_mask = adata.obs["gene"] == "non-targeting"
    X_ctrl = adata[ctrl_mask].X
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = X_ctrl.astype(np.float32)
    nonzero_cols = np.where(X_ctrl.sum(axis=0) > 0)[0]
    gene_names   = [adata.var_names[i] for i in nonzero_cols]

    # Sort genes by control-cell variance (before z-scoring)
    X_ctrl_grn = np.log1p(X_ctrl[:, nonzero_cols])
    gene_var   = X_ctrl_grn.var(axis=0)
    gene_order = np.argsort(gene_var)[::-1]
    gene_names_sorted = [gene_names[i] for i in gene_order]

    # KO row index
    ko_row = gene_names_sorted.index(ko_syn) if ko_syn in gene_names_sorted else None

    # KO cells
    ko_mask = adata.obs["gene"] != "non-targeting"
    X_ko = adata[ko_mask].X
    if hasattr(X_ko, "toarray"):
        X_ko = X_ko.toarray()
    X_ko = X_ko.astype(np.float32)
    ko_obs = adata[ko_mask].obs

    # Subset to GRN genes in variance-sorted order
    X_ko = np.log1p(X_ko[:, nonzero_cols])[:, gene_order]  # (n_ko_cells, n_grn_genes)

    # Sort cells by bin
    bin_order = np.argsort(ko_obs["gem_group"].values, kind="stable")
    X_ko  = X_ko[bin_order]
    bins  = ko_obs["gem_group"].values[bin_order]

    # Z-score per gene
    X_ko = X_ko.T  # (n_grn_genes, n_ko_cells)
    mu   = X_ko.mean(axis=1, keepdims=True)
    std  = X_ko.std(axis=1,  keepdims=True) + 1e-8
    X_ko = (X_ko - mu) / std

    return X_ko, bins, ko_row, ko_syn, ko_deg


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_ko_expression(ax: plt.Axes, X: np.ndarray, bins: np.ndarray,
                       ko_row: int | None, ko_syn: str, ko_deg: int,
                       accent: str) -> object:
    n_genes, n_cells = X.shape

    im = ax.imshow(X, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")

    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR, labelsize=8)

    # X-ticks: bin centres + dividers
    unique_bins  = sorted(set(bins))
    bin_centers  = [np.where(bins == b)[0].mean() for b in unique_bins]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(unique_bins, color=LABEL_COLOR, fontsize=8)
    for b in unique_bins[:-1]:
        boundary = np.where(bins == b)[0][-1] + 0.5
        ax.axvline(boundary, color="#ffffff", linewidth=0.5, alpha=0.4)

    ax.set_yticks([0, n_genes - 1])
    ax.set_yticklabels(["high var", "low var"], color=LABEL_COLOR, fontsize=8)
    ax.set_xlabel("Cell type (bin)", fontsize=10, color=LABEL_COLOR, labelpad=4)
    ax.set_ylabel("Genes (↑ variance)", fontsize=10, color=LABEL_COLOR, labelpad=4)

    # KO gene annotation
    if ko_row is not None:
        ax.axhline(ko_row, color=KO_COLOR, linewidth=1.4, alpha=0.7, linestyle="--")
        # ✕ on y-axis
        ax.annotate(
            "✕",
            xy=(0, ko_row), xycoords=("axes fraction", "data"),
            xytext=(-0.08, ko_row), textcoords=("axes fraction", "data"),
            fontsize=14, fontweight="bold", color=KO_COLOR,
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)],
        )
        # label on right
        ax.text(
            1.01, ko_row,
            f"{ko_syn}  (deg={ko_deg})",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color=KO_COLOR, va="center", ha="left",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)],
        )

    # Accent border
    for spine in ax.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2.0)
        spine.set_visible(True)

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/datasets/grn_ko_expression_panel.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(GRN_TYPES), len(GRN_SIZES)
    fig = plt.figure(figsize=(7 * n_cols, 5.5 * n_rows), facecolor=BG_COLOR)
    gs_outer = gridspec.GridSpec(
        n_rows, n_cols, hspace=0.18, wspace=0.28,
        left=0.09, right=0.93, top=0.92, bottom=0.05,
    )

    last_im = None
    for row, grn_type in enumerate(GRN_TYPES):
        accent = TYPE_ACCENT[grn_type]
        for col, (n_genes, size_dir, size_label) in enumerate(
            zip(GRN_SIZES, GRN_SIZE_DIRS, SIZE_LABELS)
        ):
            ax = fig.add_subplot(gs_outer[row, col])
            X, bins, ko_row, ko_syn, ko_deg = load_ko_expression(grn_type, size_dir, n_genes)
            im = draw_ko_expression(ax, X, bins, ko_row, ko_syn, ko_deg, accent)
            last_im = im

            if row == 0:
                ax.set_title(size_label, fontsize=17, fontweight="bold",
                             color=LABEL_COLOR, pad=10)
            if col == 0:
                x_fig = gs_outer[row, 0].get_position(fig).x0 - 0.055
                y_fig = gs_outer[row, 0].get_position(fig).get_points().mean(axis=0)[1]
                fig.text(x_fig, y_fig,
                         f"{grn_type}\n{TYPE_FULLNAME[grn_type]}",
                         fontsize=14, fontweight="bold", color=accent,
                         ha="center", va="center", rotation=90)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.70])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Z-score (log1p)", color=LABEL_COLOR, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    fig.text(0.5, 0.965,
             "SERGIO Synthetic Gene Expression — Hub Gene KO (highest out-degree)",
             ha="center", va="top", fontsize=20, fontweight="bold", color=TITLE_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
