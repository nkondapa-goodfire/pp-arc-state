"""
report_merged_stats.py — Report per-file statistics for a merged H5AD directory.

Writes a CSV to test_outputs/ with one row per merged file, covering cell counts,
sparsity, perturbation diversity, and GRN instance coverage.

Usage
-----
    uv run python scripts/report_merged_stats.py \
        --src data/sergio_synthetic/mini_merged \
        --out test_outputs/mini_merged_stats.csv

    uv run python scripts/report_merged_stats.py \
        --src data/sergio_synthetic/test_mini_merged \
        --out test_outputs/test_mini_merged_stats.csv
"""

import argparse
import pathlib

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp


def file_size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / 1024**2


def sparsity(X) -> float:
    """Fraction of zero entries in X."""
    if sp.issparse(X):
        n_nonzero = X.nnz
    else:
        n_nonzero = int(np.count_nonzero(X))
    return 1.0 - n_nonzero / (X.shape[0] * X.shape[1])


def summarize_file(path: pathlib.Path) -> dict:
    ad = anndata.read_h5ad(path)

    n_cells        = ad.n_obs
    n_genes        = ad.n_vars
    n_instances    = ad.obs["grn_seed"].nunique()
    unique_perts   = ad.obs["gene"].unique()
    n_perts        = sum(p != "non-targeting" for p in unique_perts)
    n_control_cells = (ad.obs["gene"] == "non-targeting").sum()
    n_pert_cells   = n_cells - n_control_cells
    cells_per_pert = n_pert_cells / n_perts if n_perts > 0 else 0

    return {
        "file":             path.name,
        "grn_type":         ad.obs["grn_type"].iloc[0],
        "grn_size":         int(ad.obs["grn_size"].iloc[0]),
        "noise_label":      ad.obs["noise_label"].iloc[0],
        # filename: {grn_type}_size{n}_noise{label}_{pert_type} — pert_type is everything after the 3rd token
        "pert_type":        "_".join(path.stem.split("_")[3:]),
        "n_cells":          n_cells,
        "n_genes":          n_genes,
        "n_instances":      n_instances,
        "n_perts":          n_perts,
        "n_control_cells":  int(n_control_cells),
        "n_pert_cells":     int(n_pert_cells),
        "cells_per_pert":   round(cells_per_pert, 1),
        "sparsity_pct":     round(sparsity(ad.X) * 100, 2),
        "file_size_mb":     round(file_size_mb(path), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Report stats for merged H5AD files.")
    parser.add_argument("--src", required=True, help="Directory of merged H5AD files.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    src = pathlib.Path(args.src).resolve()
    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    h5ad_files = sorted(src.glob("*.h5ad"))
    print(f"Found {len(h5ad_files)} files in {src}")

    rows = []
    for i, path in enumerate(h5ad_files, 1):
        row = summarize_file(path)
        rows.append(row)
        if i % 10 == 0 or i == len(h5ad_files):
            print(f"  [{i}/{len(h5ad_files)}] {path.name}")

    df = pd.DataFrame(rows).sort_values(["grn_type", "grn_size", "noise_label", "pert_type"])
    df.to_csv(out, index=False)

    print(f"\nSummary across all files:")
    print(f"  Total cells:     {df['n_cells'].sum():,}")
    print(f"  Total files:     {len(df)}")
    print(f"  Avg sparsity:    {df['sparsity_pct'].mean():.1f}%")
    print(f"  Avg file size:   {df['file_size_mb'].mean():.1f} MB")
    print(f"  Total size:      {df['file_size_mb'].sum():.0f} MB")
    print(f"\nCSV written to: {out}")


if __name__ == "__main__":
    main()
