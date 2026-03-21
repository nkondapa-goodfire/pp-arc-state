"""
inspect_h5ad.py — Load one merged H5AD and produce diagnostic visualizations.

Checks that expression values, obs columns, sparsity, control/pert split, and
cell-type structure all look reasonable after generation and merging.

Outputs
-------
test_outputs/inspect_h5ad/
  obs_distributions.png  — cell_type, gene, split column histograms
  expression_ctrl.png    — heatmap + PCA for control cells
  pert_effect.png        — log2FC and ctrl-vs-pert scatter for one pert gene
  summary.csv            — scalar sanity metrics

Usage
-----
    uv run python scripts/inspect_h5ad.py
    uv run python scripts/inspect_h5ad.py --src data/sergio_synthetic/mini_merged/BA_size050_noisehigh_KO.h5ad
    uv run python scripts/inspect_h5ad.py --src data/sergio_synthetic/mini_merged/ER_size100_noiselow_KD_010.h5ad --out-dir test_outputs/inspect_h5ad
"""

import argparse
import csv
import os

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_SRC = "data/sergio_synthetic/mini_merged/BA_size050_noiselow_KO.h5ad"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def sparsity(X) -> float:
    if sp.issparse(X):
        nnz = X.nnz
    else:
        nnz = int(np.count_nonzero(X))
    return 1.0 - nnz / (X.shape[0] * X.shape[1])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_obs_distributions(ad: anndata.AnnData, out_path: str) -> None:
    """Bar charts for cell_type, pert gene, and split columns."""
    cols = [c for c in ["cell_type", "gene", "split"] if c in ad.obs.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    MAX_BARS = 20
    for ax, col in zip(axes, cols):
        counts = ad.obs[col].value_counts()
        if len(counts) > MAX_BARS:
            top = counts.iloc[:MAX_BARS]
            other = counts.iloc[MAX_BARS:].sum()
            import pandas as pd
            counts = pd.concat([top, pd.Series({"other (N=%d)" % (len(counts) - MAX_BARS): other})])
        else:
            counts = counts.sort_index()
        ax.bar(range(len(counts)), counts.values, color="#4A90D9", alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"obs['{col}'] ({ad.obs[col].nunique()} unique)")
        ax.set_ylabel("Cell count")
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(f"obs column distributions — {os.path.basename(out_path.replace('_distributions.png', ''))}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expression(X: np.ndarray, cell_labels: np.ndarray, n_bins: int,
                    title: str, out_path: str) -> None:
    """Heatmap of top variable genes + PCA coloured by cell type."""
    n_genes, n_cells = X.shape
    cmap_ct = plt.get_cmap("tab10", n_bins)

    n_show = min(30, n_genes)
    top_idx = np.argsort(X.var(axis=1))[::-1][:n_show]
    cell_order = np.argsort(cell_labels, kind="stable")
    X_sorted = X[np.ix_(top_idx, cell_order)]
    labels_sorted = cell_labels[cell_order]

    X_log = np.log1p(X_sorted)
    row_std = X_log.std(axis=1, keepdims=True) + 1e-8
    X_z = (X_log - X_log.mean(axis=1, keepdims=True)) / row_std

    X_log_pca = np.log1p(X.T)
    X_scaled = StandardScaler().fit_transform(X_log_pca)
    n_comp = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(X_z, aspect="auto", cmap="RdBu_r",
                        vmin=-2.5, vmax=2.5, interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="Z-score")

    bar_h = max(1, n_show // 15)
    n_sc = n_cells // n_bins if n_bins > 0 else n_cells
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
        handles=[mpatches.Patch(color=cmap_ct(b), label=f"bin_{b}") for b in range(n_bins)],
        loc="upper right", fontsize=7, framealpha=0.8,
    )
    ax_heat.set_title("Expression heatmap (log1p, z-scored)")

    ax_pca = fig.add_subplot(gs[1])
    for b in range(n_bins):
        mask = cell_labels == b
        y = coords[mask, 1] if n_comp > 1 else np.zeros(mask.sum())
        ax_pca.scatter(coords[mask, 0], y, c=[cmap_ct(b)], s=8,
                       alpha=0.6, label=f"bin_{b}", linewidths=0)
    ax_pca.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)" if n_comp > 1 else "")
    ax_pca.set_title("PCA — coloured by cell type")
    ax_pca.legend(fontsize=7, markerscale=2, framealpha=0.8)
    ax_pca.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pert_effect(X_ctrl: np.ndarray, X_pert: np.ndarray,
                     pert_gene_name: str, title: str, out_path: str) -> None:
    """log2FC bar chart and ctrl-vs-pert scatter."""
    ctrl_mean = X_ctrl.mean(axis=0)  # (n_genes,)
    pert_mean = X_pert.mean(axis=0)
    log2fc = np.log2((pert_mean + 1e-6) / (ctrl_mean + 1e-6))

    # Parse pert gene index from name e.g. "SYN_0033_KO" -> 33
    pert_gene_idx = None
    parts = pert_gene_name.split("_")
    if len(parts) >= 2 and parts[0] == "SYN":
        try:
            pert_gene_idx = int(parts[1])
        except ValueError:
            pass

    # Filter to genes with any expression signal to avoid empty plot
    has_signal = (ctrl_mean > 1e-4) | (pert_mean > 1e-4)
    sig_idx = np.where(has_signal)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if len(sig_idx) > 0:
        fc_sig = log2fc[sig_idx]
        colors = ["orange" if (pert_gene_idx is not None and i == pert_gene_idx)
                  else ("#E74C3C" if v < 0 else "#27AE60")
                  for i, v in zip(sig_idx, fc_sig)]
        ax.bar(range(len(sig_idx)), fc_sig, color=colors, alpha=0.8, width=0.8)
        ax.set_xticks([])
        if pert_gene_idx is not None and pert_gene_idx in sig_idx:
            pos = np.where(sig_idx == pert_gene_idx)[0][0]
            ax.annotate(f"gene {pert_gene_idx}", xy=(pos, fc_sig[pos]),
                        xytext=(pos, fc_sig[pos] + 0.3 * np.sign(fc_sig[pos] or 1)),
                        fontsize=7, ha="center", color="orange")
    else:
        ax.text(0.5, 0.5, "no expressed genes", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Expressed genes ({len(sig_idx)}/{len(ctrl_mean)} with signal)")
    ax.set_ylabel("log2FC (pert / ctrl)")
    ax.set_title(f"Per-gene log2FC  [{pert_gene_name}]")
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.scatter(ctrl_mean, pert_mean, s=10, alpha=0.5, color="#4A90D9")
    if pert_gene_idx is not None and pert_gene_idx < len(ctrl_mean):
        ax2.scatter(ctrl_mean[pert_gene_idx], pert_mean[pert_gene_idx],
                    s=80, color="orange", zorder=5, label=f"pert gene ({pert_gene_idx})")
    lim = max(ctrl_mean.max(), pert_mean.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="y=x")
    ax2.set_xlabel("Control mean expression")
    ax2.set_ylabel("Pert mean expression")
    ax2.set_title("Control vs Pert expression")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect one merged H5AD file.")
    parser.add_argument("--src",     default=DEFAULT_SRC, help="Path to H5AD file.")
    parser.add_argument("--out-dir", default="test_outputs/inspect_h5ad")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]

    print(f"Loading: {args.src}")
    ad = anndata.read_h5ad(args.src)

    n_cells, n_genes = ad.shape
    spar = sparsity(ad.X)
    n_instances = ad.obs["grn_seed"].nunique() if "grn_seed" in ad.obs else "?"
    n_bins = ad.obs["cell_type"].nunique() if "cell_type" in ad.obs else "?"

    print(f"  Shape:      {n_cells} cells × {n_genes} genes")
    print(f"  Sparsity:   {spar*100:.1f}%")
    print(f"  Instances:  {n_instances}")
    print(f"  Cell types: {n_bins}")
    print(f"  obs cols:   {list(ad.obs.columns)}")

    # -- obs distributions --
    plot_obs_distributions(
        ad,
        out_path=os.path.join(args.out_dir, f"{stem}_obs_distributions.png"),
    )
    print("  [1/3] obs_distributions.png")

    # -- expression heatmap (control cells only, one instance) --
    ctrl_mask = ad.obs["gene"] == "non-targeting"
    # Use first grn_seed for a clean single-instance view
    first_seed = ad.obs["grn_seed"].iloc[0]
    seed_ctrl_mask = ctrl_mask & (ad.obs["grn_seed"] == first_seed)

    X_dense = to_dense(ad.X)
    X_ctrl_single = X_dense[seed_ctrl_mask.values]  # (n_ctrl_cells, n_genes)

    bin_labels_raw = ad.obs.loc[seed_ctrl_mask, "cell_type"].values
    unique_bins = sorted(set(bin_labels_raw))
    bin_to_int = {b: i for i, b in enumerate(unique_bins)}
    bin_labels_int = np.array([bin_to_int[b] for b in bin_labels_raw])
    n_bins_int = len(unique_bins)

    plot_expression(
        X_ctrl_single.T,  # (n_genes, n_cells) convention
        bin_labels_int,
        n_bins_int,
        title=f"{stem} — control expression (seed {first_seed})",
        out_path=os.path.join(args.out_dir, f"{stem}_expression_ctrl.png"),
    )
    print("  [2/3] expression_ctrl.png")

    # -- pert effect (first non-targeting gene found, same seed) --
    pert_genes = [g for g in ad.obs["gene"].unique() if g != "non-targeting"]
    if pert_genes:
        first_pert = pert_genes[0]
        pert_mask = (ad.obs["gene"] == first_pert) & (ad.obs["grn_seed"] == first_seed)
        ctrl_match = seed_ctrl_mask

        X_pert_cells = X_dense[pert_mask.values]
        X_ctrl_cells = X_dense[ctrl_match.values]

        # Align cell count (pert may have fewer if bins differ)
        n_min = min(len(X_ctrl_cells), len(X_pert_cells))
        plot_pert_effect(
            X_ctrl_cells[:n_min],
            X_pert_cells[:n_min],
            pert_gene_name=first_pert,
            title=f"{stem} — pert effect [{first_pert}] vs ctrl (seed {first_seed})",
            out_path=os.path.join(args.out_dir, f"{stem}_pert_effect.png"),
        )
        print("  [3/3] pert_effect.png")
    else:
        print("  [3/3] no pert genes found, skipping pert_effect.png")

    # -- summary CSV --
    n_ctrl_cells = int(ctrl_mask.sum())
    n_pert_cells = n_cells - n_ctrl_cells
    n_pert_types = len(pert_genes)

    has_split = "split" in ad.obs.columns
    n_held_out = int((ad.obs["split"] == "held_out").sum()) if has_split else 0

    rows = [{
        "file":          os.path.basename(args.src),
        "n_cells":       n_cells,
        "n_genes":       n_genes,
        "n_ctrl_cells":  n_ctrl_cells,
        "n_pert_cells":  n_pert_cells,
        "n_pert_types":  n_pert_types,
        "n_held_out":    n_held_out,
        "n_instances":   n_instances,
        "n_bins":        n_bins,
        "sparsity_pct":  round(spar * 100, 2),
        "X_dtype":       str(ad.X.dtype),
        "X_format":      type(ad.X).__name__,
        "grn_type":      ad.obs["grn_type"].iloc[0] if "grn_type" in ad.obs else "?",
        "grn_size":      int(ad.obs["grn_size"].iloc[0]) if "grn_size" in ad.obs else "?",
        "noise_label":   ad.obs["noise_label"].iloc[0] if "noise_label" in ad.obs else "?",
    }]

    csv_path = os.path.join(args.out_dir, f"{stem}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary:")
    for k, v in rows[0].items():
        print(f"  {k:<18} {v}")
    print(f"\nOutputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
