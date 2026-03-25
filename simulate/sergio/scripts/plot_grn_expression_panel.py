"""
plot_grn_expression_panel.py — 3×3 gene expression heatmap panel.

Rows = GRN type  (BA, BA-VM, ER)
Cols = GRN size  (10, 50, 100 genes)

For each cell, loads the control (non-targeting) cells from grn_0000/noise_none,
subsets to the GRN genes (first N vars), log1p + z-scores, and plots a heatmap
with cells sorted by PC1.

Usage:
    uv run python scripts/plot_grn_expression_panel.py
    uv run python scripts/plot_grn_expression_panel.py --out results/datasets/grn_expression_panel.png
"""

import argparse
import glob
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRN_TYPES  = ["BA", "BA-VM", "ER"]
GRN_SIZES  = [10, 50, 100]
GRN_SIZE_DIRS = ["size_010", "size_050", "size_100"]
SEED       = 0
N_CELLS    = 100   # cells to display (subsampled from control)

DATA_ROOT = Path(__file__).resolve().parent.parent / \
            "data/sergio_synthetic/SERGIO_PPT"

# Dark palette matching the network panel
BG_COLOR     = "#0f1117"
LABEL_COLOR  = "#e8eaf0"
TITLE_COLOR  = "#ffffff"
BORDER_COLOR = "#2a2d3a"

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

SIZE_LABELS = ["10 genes", "50 genes", "100 genes"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_control_expression(grn_type: str, size_dir: str, n_genes: int) -> np.ndarray:
    """
    Returns (n_genes × n_cells) log1p z-scored expression matrix
    using control (non-targeting) cells, GRN genes only (identified as
    nonzero columns in control cells — GRN genes are sparse in the 2000-gene space).
    """
    grn_dir = DATA_ROOT / grn_type / size_dir / "noise_none" / "grn_0000"
    h5ads = sorted(grn_dir.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No h5ad files in {grn_dir}")

    # Load first file (all share the same control cells)
    adata = ad.read_h5ad(h5ads[0])

    # Control cells only
    ctrl_mask = adata.obs["gene"] == "non-targeting"
    X = adata[ctrl_mask].X  # (n_ctrl_cells, 2000)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    # Identify GRN genes as columns with any nonzero expression
    nonzero_cols = np.where(X.sum(axis=0) > 0)[0]
    X = X[:, nonzero_cols]  # (n_ctrl_cells, n_grn_genes)

    # Subsample cells
    rng = np.random.default_rng(42)
    n_avail = X.shape[0]
    idx = rng.choice(n_avail, size=min(N_CELLS, n_avail), replace=False)
    X = X[idx]  # (N_CELLS, n_genes)

    # log1p
    X = np.log1p(X)

    # Sort genes by variance (before z-scoring) — used for ordering rows
    gene_var = X.var(axis=0)  # variance across cells, per gene
    gene_order = np.argsort(gene_var)[::-1]  # high var first
    X = X[:, gene_order]  # (N_CELLS, n_genes) sorted by variance

    # Sort cells by bin (gem_group)
    ctrl_obs = adata[ctrl_mask].obs.iloc[idx]
    bin_order = np.argsort(ctrl_obs["gem_group"].values, kind="stable")
    X = X[bin_order]
    bins = ctrl_obs["gem_group"].values[bin_order]

    # Z-score per gene (row = gene)
    X = X.T  # (n_genes, N_CELLS)
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1,  keepdims=True) + 1e-8
    X   = (X - mu) / std

    return X, bins  # (n_genes, N_CELLS), rows sorted high->low variance


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_expression(ax: plt.Axes, X: np.ndarray, bins: np.ndarray,
                    n_genes: int, accent: str) -> None:
    """Draw gene-expression heatmap on ax."""
    n_show_genes = min(n_genes, X.shape[0])
    X_show = X[:n_show_genes]  # already sorted high->low variance before z-scoring

    im = ax.imshow(
        X_show,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2.5, vmax=2.5,
        interpolation="nearest",
    )

    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR, labelsize=8)

    # Axis labels
    ax.set_xlabel("Cell type (bin)", fontsize=10, color=LABEL_COLOR, labelpad=4)
    ax.set_ylabel("Genes (↑ variance)", fontsize=10, color=LABEL_COLOR, labelpad=4)

    # X-ticks: one per bin, placed at bin centre
    unique_bins = sorted(set(bins))
    bin_centers = [np.where(bins == b)[0].mean() for b in unique_bins]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(unique_bins, color=LABEL_COLOR, fontsize=8)

    # Vertical dividers between bins
    for b in unique_bins[:-1]:
        boundary = np.where(bins == b)[0][-1] + 0.5
        ax.axvline(boundary, color="#ffffff", linewidth=0.5, alpha=0.4)

    ax.set_yticks([0, n_show_genes - 1])
    ax.set_yticklabels(["high var", "low var"], color=LABEL_COLOR, fontsize=8)

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
    parser.add_argument("--out", default="results/datasets/grn_expression_panel.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(GRN_TYPES), len(GRN_SIZES)

    fig = plt.figure(figsize=(6.5 * n_cols, 5.5 * n_rows), facecolor=BG_COLOR)
    gs_outer = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.18, wspace=0.20,
        left=0.08, right=0.94, top=0.92, bottom=0.05,
    )

    last_im = None
    for row, grn_type in enumerate(GRN_TYPES):
        accent = TYPE_ACCENT[grn_type]
        for col, (n_genes, size_dir, size_label) in enumerate(
            zip(GRN_SIZES, GRN_SIZE_DIRS, SIZE_LABELS)
        ):
            ax = fig.add_subplot(gs_outer[row, col])

            X, bins = load_control_expression(grn_type, size_dir, n_genes)
            im = draw_expression(ax, X, bins, n_genes, accent)
            last_im = im

            # Column headers
            if row == 0:
                ax.set_title(size_label, fontsize=17, fontweight="bold",
                             color=LABEL_COLOR, pad=10)

            # Row labels
            if col == 0:
                x_fig = gs_outer[row, 0].get_position(fig).x0 - 0.055
                y_fig = gs_outer[row, 0].get_position(fig).get_points().mean(axis=0)[1]
                fig.text(x_fig, y_fig,
                         f"{grn_type}\n{TYPE_FULLNAME[grn_type]}",
                         fontsize=14, fontweight="bold",
                         color=accent,
                         ha="center", va="center", rotation=90)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.955, 0.12, 0.012, 0.70])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Z-score (log1p)", color=LABEL_COLOR, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    # Main title
    fig.text(0.5, 0.965,
             "SERGIO Synthetic Gene Expression  (control cells, GRN genes)",
             ha="center", va="top",
             fontsize=20, fontweight="bold", color=TITLE_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
