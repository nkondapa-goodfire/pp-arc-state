"""
build_flat_symlinks.py — Create a flat symlink directory for a SERGIO dataset split.

cell-load's _find_dataset_files() uses glob.glob() without recursive=True, so
the **/*.h5ad pattern does not work on nested directory trees. It also deduplicates
by stem, so files with the same name across different GRN instances would collide.

This script creates a flat directory of uniquely-named symlinks pointing to the
original h5ad files. Run once after generation before training.

Output name format:
  {grn_type}_{size}_{noise}_{grn}_{stem}.h5ad
  e.g. BA_s010_nlow_g0000_SYN_0033_KO.h5ad

Usage
-----
    uv run python scripts/build_flat_symlinks.py \\
        --src data/sergio_synthetic/mini \\
        --dst data/sergio_synthetic/mini_flat

    uv run python scripts/build_flat_symlinks.py \\
        --src data/sergio_synthetic/train \\
        --dst data/sergio_synthetic/train_flat

    uv run python scripts/build_flat_symlinks.py \\
        --src data/sergio_synthetic/test_mini \\
        --dst data/sergio_synthetic/test_mini_flat

    uv run python scripts/build_flat_symlinks.py \\
        --src data/sergio_synthetic/test \\
        --dst data/sergio_synthetic/test_flat
"""

import argparse
import pathlib


def build_flat_symlinks(src: pathlib.Path, dst: pathlib.Path) -> int:
    """
    Create uniquely-named symlinks in dst pointing to all h5ad files under src.

    src layout:  {grn_type}/size_{n}/noise_{label}/grn_{seed}/{stem}.h5ad
    dst layout:  {grn_type}_s{n}_n{label}_g{seed}_{stem}.h5ad  (flat)
    """
    dst.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for h5ad_path in sorted(src.rglob("*.h5ad")):
        parts = h5ad_path.relative_to(src).parts
        if len(parts) != 5:
            print(f"  WARNING: unexpected path depth, skipping: {h5ad_path}")
            continue

        grn_type, size_dir, noise_dir, grn_dir, fname = parts
        size  = size_dir.replace("size_",  "s")
        noise = noise_dir.replace("noise_", "n")
        grn   = grn_dir.replace("grn_",   "g")
        stem  = pathlib.Path(fname).stem

        unique_name = f"{grn_type}_{size}_{noise}_{grn}_{stem}.h5ad"
        link = dst / unique_name

        if link.exists():
            skipped += 1
            continue

        link.symlink_to(h5ad_path.resolve())
        created += 1

    return created, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Build flat symlink directory for cell-load compatibility."
    )
    parser.add_argument("--src", required=True, help="Source dataset directory (nested).")
    parser.add_argument("--dst", required=True, help="Destination flat symlink directory.")
    args = parser.parse_args()

    src = pathlib.Path(args.src).resolve()
    dst = pathlib.Path(args.dst).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"Source: {src}")
    print(f"Dest:   {dst}")

    created, skipped = build_flat_symlinks(src, dst)
    total = len(list(dst.glob("*.h5ad")))

    print(f"Created: {created}  Skipped (already exist): {skipped}")
    print(f"Total symlinks: {total}")


if __name__ == "__main__":
    main()
