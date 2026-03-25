"""
build_merged_h5ads.py — Merge per-instance H5AD files into condition-grouped H5ADs.

Groups source files by (grn_type, grn_size, noise_level, pert_type) and concatenates
all GRN seeds for each condition into a single H5AD. This produces ~165 files for the
full dataset, enabling clean ablation runs by pointing a TOML at a subset directory.

Input layout:
  {grn_type}/size_{n}/noise_{label}/grn_{seed}/{gene}_{pert_type}.h5ad

Output layout (flat dir):
  {grn_type}_size{n}_noise{label}_{pert_type}.h5ad
  e.g. BA_size010_noiselow_KO.h5ad

Also writes manifest.csv alongside the merged files with columns:
  merged_file, grn_type, grn_size, noise_label, pert_type, n_cells, n_instances

Usage
-----
    uv run python scripts/build_merged_h5ads.py \\
        --src data/sergio_synthetic/mini \\
        --dst data/sergio_synthetic/mini_merged

    uv run python scripts/build_merged_h5ads.py \\
        --src data/sergio_synthetic/train \\
        --dst data/sergio_synthetic/train_merged

    uv run python scripts/build_merged_h5ads.py \\
        --src data/sergio_synthetic/test_mini \\
        --dst data/sergio_synthetic/test_mini_merged

    uv run python scripts/build_merged_h5ads.py \\
        --src data/sergio_synthetic/test \\
        --dst data/sergio_synthetic/test_merged
"""

import argparse
import pathlib
from collections import defaultdict

import anndata
import pandas as pd


def collect_paths(src: pathlib.Path) -> dict[tuple, list[pathlib.Path]]:
    """
    Walk src and group h5ad paths by (grn_type, grn_size, noise_label, pert_type).

    src layout:  {grn_type}/size_{n}/noise_{label}/grn_{seed}/{gene}_{pert_type}.h5ad
    Returns:     {("BA", "010", "low", "KO"): [path1, path2, ...], ...}
    """
    groups: dict[tuple, list[pathlib.Path]] = defaultdict(list)
    for h5ad_path in sorted(src.rglob("*.h5ad")):
        parts = h5ad_path.relative_to(src).parts
        if len(parts) != 5:
            print(f"  WARNING: unexpected path depth, skipping: {h5ad_path}")
            continue
        grn_type, size_dir, noise_dir, _grn_dir, fname = parts
        size_val  = size_dir.replace("size_", "")
        noise_val = noise_dir.replace("noise_", "")
        stem = pathlib.Path(fname).stem  # e.g. SYN_0033_KO or SYN_0033_KD_010
        # pert_type is everything after the gene token: SYN_XXXX_<pert_type>
        pert_type = "_".join(stem.split("_")[2:])  # KO | KD_010 | KD_050 | KD_080
        if not pert_type:
            pert_type = stem.split("_")[-1]
        key = (grn_type, size_val, noise_val, pert_type)
        groups[key].append(h5ad_path)
    return groups


def merge_group(
    key: tuple,
    paths: list[pathlib.Path],
    src: pathlib.Path,
    bins: list[str] | None = None,
) -> anndata.AnnData:
    """Load and concatenate all instances for one condition, adding provenance columns.

    If bins is provided, only cells whose gem_group is in bins are retained.
    """
    grn_type, size_val, noise_val, pert_type = key
    adatas = []
    for p in paths:
        parts = p.relative_to(src).parts
        _gt, _sd, _nd, grn_dir, _ = parts
        grn_seed = int(grn_dir.replace("grn_", ""))

        ad = anndata.read_h5ad(p)
        if bins is not None:
            ad = ad[ad.obs["gem_group"].isin(bins)].copy()
        ad.obs["grn_type"]    = grn_type
        ad.obs["grn_size"]    = int(size_val)
        ad.obs["noise_label"] = noise_val
        ad.obs["grn_seed"]    = grn_seed
        adatas.append(ad)

    merged = anndata.concat(adatas, join="outer")
    merged.obs_names_make_unique()
    merged.obsm["X_hvg"] = merged.X.toarray()
    return merged


def condition_filename(grn_type: str, size_val: str, noise_val: str, pert_type: str) -> str:
    return f"{grn_type}_size{size_val}_noise{noise_val}_{pert_type}.h5ad"


def build_merged_h5ads(src: pathlib.Path, dst: pathlib.Path, bins: list[str] | None = None) -> pd.DataFrame:
    dst.mkdir(parents=True, exist_ok=True)

    groups = collect_paths(src)
    n_source = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} conditions across {n_source} source files.")
    if bins is not None:
        print(f"Filtering to bins: {bins}")

    manifest_rows = []
    created = skipped = 0

    for key, paths in sorted(groups.items()):
        grn_type, size_val, noise_val, pert_type = key
        fname = condition_filename(grn_type, size_val, noise_val, pert_type)
        out_path = dst / fname

        if out_path.exists():
            skipped += 1
            ad = anndata.read_h5ad(out_path, backed="r")
            n_cells = ad.n_obs
            n_instances = ad.obs["grn_seed"].nunique() if "grn_seed" in ad.obs else len(paths)
            ad.file.close()
        else:
            merged = merge_group(key, paths, src, bins=bins)
            merged.write_h5ad(out_path)
            n_cells = merged.n_obs
            n_instances = len(paths)
            created += 1
            print(f"  [{created + skipped}/{len(groups)}] {fname}: {n_cells} cells, {n_instances} instances")

        manifest_rows.append({
            "merged_file": fname,
            "grn_type": grn_type,
            "grn_size": int(size_val),
            "noise_label": noise_val,
            "pert_type": pert_type,
            "n_cells": n_cells,
            "n_instances": n_instances,
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(dst / "manifest.csv", index=False)
    print(f"\nCreated: {created}  Skipped (already exist): {skipped}")
    print(f"Total merged files: {len(list(dst.glob('*.h5ad')))}")
    print(f"manifest.csv written: {len(manifest)} rows")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-instance H5ADs into condition-grouped files for ablation training."
    )
    parser.add_argument("--src", required=True, help="Source dataset directory (nested).")
    parser.add_argument("--dst", required=True, help="Destination merged H5AD directory.")
    parser.add_argument(
        "--bins", default=None,
        help="Comma-separated bin labels to retain (e.g. bin_0,bin_1,bin_2,bin_3,bin_4). "
             "Filters on obs['gem_group']. If omitted, all bins are kept.",
    )
    args = parser.parse_args()

    src = pathlib.Path(args.src).resolve()
    dst = pathlib.Path(args.dst).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    bins = [b.strip() for b in args.bins.split(",")] if args.bins else None

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    build_merged_h5ads(src, dst, bins=bins)


if __name__ == "__main__":
    main()
