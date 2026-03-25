#!/usr/bin/env python
"""
Build the [fewshot] section of sergio_mini_train.toml for the pert-type-holdout condition.

For each (grn_type, grn_size, seed) combination, 3 of the 10 perturbed genes
have KD_010 held out as test. The other 7 genes see all 4 pert_types during training.

Strategy: for each grn_seed within each KD_010 merged file, sort the genes by index
and hold out the first 3.

Output: updates configs/sergio_mini_train.toml with the [fewshot] section.
"""

import re
import toml
from pathlib import Path
import anndata

SERGIO_DIR  = Path(__file__).parent.parent
MERGED_DIR  = SERGIO_DIR / "data/sergio_synthetic/mini_incl_merged"
TOML_PATH   = SERGIO_DIR / "configs/sergio_mini_train.toml"
N_BINS      = 10
HOLDOUT_FRAC = 3  # genes per seed to hold out


def gene_index(name: str) -> int:
    """SYN_0042_KD_010 -> 42"""
    return int(name.split("_")[1])


def collect_holdout_genes() -> set[str]:
    """
    Scan all KD_010 merged files. For each grn_seed within each file,
    sort the 10 perturbed gene names by index, take the first 3 as holdout.
    Returns a set of perturbation names like 'SYN_0042_KD_010'.
    """
    holdout = set()
    kd010_files = sorted(MERGED_DIR.glob("*_KD_010.h5ad"))
    print(f"Scanning {len(kd010_files)} KD_010 merged files...")

    for fpath in kd010_files:
        ad = anndata.read_h5ad(fpath)
        for seed in sorted(ad.obs["grn_seed"].unique()):
            seed_mask = (ad.obs["grn_seed"] == seed) & (ad.obs["gene"] != "non-targeting")
            genes = sorted(ad.obs[seed_mask]["gene"].unique(), key=gene_index)
            for g in genes[:HOLDOUT_FRAC]:
                holdout.add(g)

    return holdout


def main():
    holdout_genes = collect_holdout_genes()
    print(f"\nTotal holdout perturbations: {len(holdout_genes)}")
    for g in sorted(holdout_genes, key=gene_index):
        print(f"  {g}")

    # Load existing TOML
    cfg = toml.load(TOML_PATH)

    # Build fewshot section: one entry per bin
    fewshot = {}
    holdout_list = sorted(holdout_genes, key=gene_index)
    for b in range(N_BINS):
        key = f"sergio_mini.bin_{b}"
        fewshot[key] = {"test": holdout_list}

    cfg["fewshot"] = fewshot

    with open(TOML_PATH, "w") as f:
        toml.dump(cfg, f)

    print(f"\nWrote fewshot section ({N_BINS} bins × {len(holdout_list)} holdout genes) to {TOML_PATH}")


if __name__ == "__main__":
    main()
