"""
fix_cell_type.py — Retroactively fix obs['cell_type'] in existing SERGIO H5AD files.

Old format: grn_{seed:04d}                        e.g. "grn_0000"
New format: {grn_type}_size{grn_size:03d}_seed{grn_seed:04d}  e.g. "ER_size010_seed0000"

Two modes:
  --raw   Nested source dirs (SERGIO_PPT, SERGIO_TGT). Infers grn_type and grn_size
          from the file path; reads grn_seed from existing cell_type value.
  --merged  Flat merged dirs (SERGIO_PPT_merged, SERGIO_TGT_train_merged,
            SERGIO_TGT_test_merged). Uses grn_type, grn_size, grn_seed obs columns
            already present in the file.

Usage
-----
# Fix all five directories:
cd /mnt/polished-lake/home/nkondapaneni/state/simulate/sergio

uv run python scripts/fix_cell_type.py --raw     data/sergio_synthetic/SERGIO_PPT
uv run python scripts/fix_cell_type.py --raw     data/sergio_synthetic/SERGIO_TGT
uv run python scripts/fix_cell_type.py --merged  data/sergio_synthetic/SERGIO_PPT_merged
uv run python scripts/fix_cell_type.py --merged  data/sergio_synthetic/SERGIO_TGT_train_merged
uv run python scripts/fix_cell_type.py --merged  data/sergio_synthetic/SERGIO_TGT_test_merged

# Dry-run (print what would change, don't write):
uv run python scripts/fix_cell_type.py --merged  data/sergio_synthetic/SERGIO_PPT_merged --dry-run
"""

import argparse
import pathlib
import re

import anndata
import numpy as np


def new_cell_type(grn_type: str, grn_size: int, grn_seed: int) -> str:
    return f"{grn_type}_size{grn_size:03d}_seed{grn_seed:04d}"


# ── Raw nested dirs ──────────────────────────────────────────────────────────
# Layout: {grn_type}/size_{n:03d}/noise_{label}/grn_{seed:04d}/{gene}_{pert}.h5ad

def fix_raw(src: pathlib.Path, dry_run: bool) -> None:
    h5ads = sorted(src.rglob("*.h5ad"))
    print(f"Found {len(h5ads)} h5ad files in {src}")
    changed = skipped = errors = 0

    for p in h5ads:
        parts = p.relative_to(src).parts
        if len(parts) != 5:
            print(f"  SKIP (unexpected depth): {p}")
            skipped += 1
            continue

        grn_type, size_dir, _noise_dir, grn_dir, _fname = parts
        grn_size = int(size_dir.replace("size_", ""))
        grn_seed = int(grn_dir.replace("grn_", ""))
        target = new_cell_type(grn_type, grn_size, grn_seed)

        try:
            ad = anndata.read_h5ad(p)
        except Exception as e:
            print(f"  ERROR reading {p}: {e}")
            errors += 1
            continue

        current = ad.obs["cell_type"].iloc[0]
        if current == target:
            skipped += 1
            continue

        if dry_run:
            print(f"  DRY-RUN {p.name}: '{current}' → '{target}'")
            changed += 1
            continue

        ad.obs["cell_type"] = target
        ad.write_h5ad(p)
        changed += 1

    print(f"Done: changed={changed}  skipped={skipped}  errors={errors}")


# ── Merged flat dirs ─────────────────────────────────────────────────────────
# Layout: {grn_type}_size{n}_noise{label}_{pert}.h5ad
# obs columns already present: grn_type, grn_size, grn_seed

def fix_merged(src: pathlib.Path, dry_run: bool) -> None:
    h5ads = sorted(src.glob("*.h5ad"))
    print(f"Found {len(h5ads)} h5ad files in {src}")
    changed = skipped = errors = 0

    for p in h5ads:
        try:
            ad = anndata.read_h5ad(p)
        except Exception as e:
            print(f"  ERROR reading {p}: {e}")
            errors += 1
            continue

        if not all(c in ad.obs.columns for c in ("grn_type", "grn_size", "grn_seed")):
            print(f"  SKIP (missing obs columns grn_type/grn_size/grn_seed): {p.name}")
            skipped += 1
            continue

        expected = ad.obs.apply(
            lambda r: new_cell_type(str(r["grn_type"]), int(r["grn_size"]), int(r["grn_seed"])),
            axis=1,
        )

        if (ad.obs["cell_type"] == expected).all():
            skipped += 1
            continue

        n_changed = (ad.obs["cell_type"] != expected).sum()
        if dry_run:
            mapping = (
                ad.obs[["cell_type", "grn_type", "grn_size", "grn_seed"]]
                .drop_duplicates()
                .assign(new_cell_type=lambda r: r.apply(
                    lambda row: new_cell_type(str(row["grn_type"]), int(row["grn_size"]), int(row["grn_seed"])), axis=1
                ))
            )
            pairs = [f"'{r.cell_type}' → '{r.new_cell_type}'" for _, r in mapping.iterrows()]
            print(f"  DRY-RUN {p.name}: {n_changed} cells")
            for pair in pairs:
                print(f"    {pair}")
            changed += 1
            continue

        ad.obs["cell_type"] = expected.values
        ad.write_h5ad(p)
        print(f"  Fixed {p.name}: {n_changed} cells updated")
        changed += 1

    print(f"Done: changed={changed}  skipped={skipped}  errors={errors}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fix obs['cell_type'] in SERGIO H5AD files.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw",    metavar="DIR", help="Nested raw source directory.")
    mode.add_argument("--merged", metavar="DIR", help="Flat merged directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    src = pathlib.Path(args.raw or args.merged).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Directory not found: {src}")

    print(f"Mode: {'raw' if args.raw else 'merged'}  |  dry_run={args.dry_run}")
    print(f"Path: {src}\n")

    if args.raw:
        fix_raw(src, args.dry_run)
    else:
        fix_merged(src, args.dry_run)


if __name__ == "__main__":
    main()
