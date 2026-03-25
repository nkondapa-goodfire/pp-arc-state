"""
test_TGT_splits.py — Distributional sanity check for the SERGIO_TGT train/test bin split.

For each (grn_type, grn_size) combination, pools cells across all noise levels and
pert types, subsamples up to 5k from the train split (bins 0–4) and 5k from the
test split (bins 5–9), then:

  Page 1 — PCA grid: side-by-side scatter plots per (GRN type, size).
            Title shows mean Wasserstein-1 distance averaged across genes —
            a per-gene marginal distribution comparison. Lower = more similar.

  Page 2 — Gene histograms: pools all conditions, ranks genes by W1 distance,
            plots 10 most similar / 10 least similar / 10 random gene expression
            histograms (log1p), overlaid train vs test.

Usage
-----
    uv run python scripts/test_TGT_splits.py
    uv run python scripts/test_TGT_splits.py \\
        --train-dir data/sergio_synthetic/SERGIO_TGT_train_merged \\
        --test-dir  data/sergio_synthetic/SERGIO_TGT_test_merged \\
        --out       test_outputs/TGT_splits_pca.pdf
"""

import argparse
import pathlib
import re

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
from sklearn.decomposition import PCA

SUBSAMPLE = 5_000
SEED = 42
RNG = np.random.default_rng(SEED)

GRN_TYPES = ["BA", "BA-VM", "ER"]
SIZES = [10, 50, 100]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(stem: str):
    m = re.match(r"^(.+?)_size(\d+)_", stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_X(path: pathlib.Path) -> np.ndarray:
    ad = anndata.read_h5ad(path)
    return ad.obsm["X_hvg"].astype(np.float32)


def subsample(X: np.ndarray, n: int) -> np.ndarray:
    if X.shape[0] > n:
        idx = RNG.choice(X.shape[0], size=n, replace=False)
        return X[idx]
    return X


def collect_by_condition(directory: pathlib.Path) -> dict[tuple, list[pathlib.Path]]:
    groups: dict[tuple, list[pathlib.Path]] = {}
    for p in sorted(directory.glob("*.h5ad")):
        grn_type, grn_size = parse_filename(p.stem)
        if grn_type is None:
            continue
        groups.setdefault((grn_type, grn_size), []).append(p)
    return groups




# ---------------------------------------------------------------------------
# Page 1: PCA grid
# ---------------------------------------------------------------------------

def make_pca_page(train_groups, test_groups, subsample_n: int) -> plt.Figure:
    nrows = len(GRN_TYPES)
    ncols = len(SIZES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.0), squeeze=False)
    alpha = max(0.02, min(0.15, 600 / subsample_n))

    for row, grn_type in enumerate(GRN_TYPES):
        for col_idx, size in enumerate(SIZES):
            key = (grn_type, size)
            ax = axes[row][col_idx]

            if key not in train_groups or key not in test_groups:
                ax.set_visible(False)
                continue

            X_train = subsample(np.vstack([load_X(p) for p in train_groups[key]]), subsample_n)
            X_test  = subsample(np.vstack([load_X(p) for p in test_groups[key]]),  subsample_n)

            pca = PCA(n_components=2, random_state=SEED)
            pca.fit(np.vstack([X_train, X_test]))
            var = pca.explained_variance_ratio_ * 100
            tr = pca.transform(X_train)
            te = pca.transform(X_test)

            # Overlay both splits in the same panel
            ax.scatter(tr[:, 0], tr[:, 1], s=1, alpha=alpha, color="#2166ac",
                       rasterized=True, label="train (bins 0–4)")
            ax.scatter(te[:, 0], te[:, 1], s=1, alpha=alpha, color="#d6604d",
                       rasterized=True, label="test (bins 5–9)")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"PC1 {var[0]:.0f}%", fontsize=7)
            ax.set_ylabel(f"PC2 {var[1]:.0f}%", fontsize=7)
            ax.set_title(f"{grn_type}  size={size}", fontsize=8)

    handles = [
        matplotlib.patches.Patch(color="#2166ac", label="train (bins 0–4)"),
        matplotlib.patches.Patch(color="#d6604d", label="test  (bins 5–9)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig.suptitle(
        "SERGIO_TGT — train/test bin split PCA (overlaid)\n"
        "Shared PCA per (GRN type, size), pooled across noise levels & pert types",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Page 2: Gene activity histograms
# ---------------------------------------------------------------------------

def make_histogram_page(train_groups, test_groups, subsample_n: int) -> plt.Figure:
    # Pool all conditions
    all_keys = sorted(set(train_groups) & set(test_groups))
    X_train_all = subsample(
        np.vstack([np.vstack([load_X(p) for p in train_groups[k]]) for k in all_keys]),
        subsample_n,
    )
    X_test_all = subsample(
        np.vstack([np.vstack([load_X(p) for p in test_groups[k]]) for k in all_keys]),
        subsample_n,
    )

    # 30 most active genes by mean expression (pooled)
    mean_activity = np.concatenate([X_train_all, X_test_all], axis=0).mean(axis=0)
    top30 = np.argsort(mean_activity)[-30:][::-1]

    fig, axes = plt.subplots(3, 10, figsize=(22, 7), squeeze=False)

    for idx, g in enumerate(top30):
        row, col = divmod(idx, 10)
        ax = axes[row][col]
        # Non-zero values only so the tail structure is visible
        tr_vals = np.log1p(X_train_all[:, g][X_train_all[:, g] > 0])
        te_vals = np.log1p(X_test_all[:, g][X_test_all[:, g] > 0])
        combined = np.concatenate([tr_vals, te_vals])
        bins = np.linspace(combined.min(), combined.max(), 40) if len(combined) > 1 else 10
        ax.hist(tr_vals, bins=bins, density=True, alpha=0.7, color="#2166ac", label="train")
        ax.hist(te_vals, bins=bins, density=True, alpha=0.7, color="#d6604d", label="test")
        ax.set_title(f"gene {g}\nµ={mean_activity[g]:.3f}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [
        matplotlib.patches.Patch(color="#2166ac", label="train (bins 0–4)"),
        matplotlib.patches.Patch(color="#d6604d", label="test  (bins 5–9)"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Top 30 most active genes — log1p expression histograms, pooled across all conditions",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/sergio_synthetic/SERGIO_TGT_train_merged")
    parser.add_argument("--test-dir",  default="data/sergio_synthetic/SERGIO_TGT_test_merged")
    parser.add_argument("--out",       default="test_outputs/TGT_splits_pca.pdf")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE)
    args = parser.parse_args()

    train_dir = pathlib.Path(args.train_dir)
    test_dir  = pathlib.Path(args.test_dir)
    out_path  = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_groups = collect_by_condition(train_dir)
    test_groups  = collect_by_condition(test_dir)

    print("Building PCA page ...")
    fig_pca  = make_pca_page(train_groups, test_groups, args.subsample)
    print("Building histogram page ...")
    fig_hist = make_histogram_page(train_groups, test_groups, args.subsample)

    with pdf_backend.PdfPages(out_path) as pdf:
        pdf.savefig(fig_pca,  bbox_inches="tight")
        pdf.savefig(fig_hist, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
