"""
sweep_downsample_cells.py — Sweep cells_per_pert values and report dataloader sizes.

For each value of cells_per_pert, instantiates BalancedPerturbationDataModule, runs setup(),
and reports true train cells (post-subsample) and number of batches in the train dataloader.

Usage
-----
cd /mnt/polished-lake/home/nkondapaneni/state

uv run python simulate/sergio/scripts/sweep_downsample_cells.py \
    --toml simulate/sergio/configs/sergio_ppt_train.toml \
    --cell-type-key cell_type \
    --pert-col gene \
    --batch-col gem_group \
    --control-pert non-targeting \
    --embed-key X_hvg \
    --output-space gene
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

DOWNSAMPLE_VALUES = [None, 10, 25, 50, 100, 250, 500, 1000]


def probe(kwargs, cells_per_pert, batch_size, cell_sentence_len):
    from balanced_datamodule import BalancedPerturbationDataModule

    dm = BalancedPerturbationDataModule(
        cells_per_pert=cells_per_pert,
        batch_size=batch_size,
        cell_sentence_len=cell_sentence_len,
        **kwargs,
    )
    dm.setup(stage="fit")

    # Count train cells per cell type from the (possibly subsampled) subset indices
    ct_counts: dict[str, int] = collections.Counter()
    for subset in dm.train_datasets:
        base = subset.dataset if hasattr(subset, "dataset") else subset
        indices = np.asarray(subset.indices) if hasattr(subset, "indices") else base.all_indices
        for ct_code, ct_name in enumerate(base.cell_type_categories):
            mask = base.metadata_cache.cell_type_codes[indices] == ct_code
            ct_counts[ct_name] += int(mask.sum())

    total_cells = sum(ct_counts.values())
    n_batches = len(dm.train_dataloader())

    return total_cells, n_batches, ct_counts


def main():
    parser = argparse.ArgumentParser(description="Sweep downsample_cells and report dataloader sizes.")
    parser.add_argument("--toml",              required=True)
    parser.add_argument("--cell-type-key",     default="cell_type")
    parser.add_argument("--pert-col",          default="gene")
    parser.add_argument("--batch-col",         default="gem_group")
    parser.add_argument("--control-pert",      default="non-targeting")
    parser.add_argument("--embed-key",         default="X_hvg")
    parser.add_argument("--output-space",      default="gene")
    parser.add_argument("--batch-size",        type=int, default=64)
    parser.add_argument("--cell-sentence-len", type=int, default=64)
    parser.add_argument("--downsample-values", type=str, default=None,
                        help="Comma-separated list of ints (e.g. 50,100,500). Defaults to preset sweep.")
    args = parser.parse_args()

    base_kwargs = {
        "toml_config_path":           args.toml,
        "embed_key":                  args.embed_key,
        "output_space":               args.output_space,
        "pert_col":                   args.pert_col,
        "cell_type_key":              args.cell_type_key,
        "batch_col":                  args.batch_col,
        "control_pert":               args.control_pert,
        "num_workers":                0,
        "pin_memory":                 False,
        "n_basal_samples":            1,
        "basal_mapping_strategy":     "random",
        "should_yield_control_cells": True,
        "map_controls":               True,
        "perturbation_features_file": None,
        "store_raw_basal":            False,
        "downsample":                 None,
        "downsample_cells":           None,
        "additional_obs":             [],
        "val_subsample_fraction":     1.0,
    }

    if args.downsample_values:
        sweep = [int(x) for x in args.downsample_values.split(",")]
    else:
        sweep = DOWNSAMPLE_VALUES

    print(f"TOML: {args.toml}")
    print(f"batch_size={args.batch_size}  cell_sentence_len={args.cell_sentence_len}")
    print()
    print(f"{'cells_per_pert':>18}  {'total_cells':>12}  {'train_batches':>13}")
    print("-" * 50)

    results = []
    for val in sweep:
        label = str(val) if val is not None else "None (full)"
        total_cells, n_batches, ct_counts = probe(
            base_kwargs, val, args.batch_size, args.cell_sentence_len
        )
        print(f"{label:>18}  {total_cells:>12,}  {n_batches:>13,}")
        results.append((val, total_cells, n_batches, ct_counts))

    # Per-cell-type breakdown for each sweep value
    print()
    print("=== Per-cell-type breakdown ===")
    all_cts = sorted({ct for _, _, _, ct_counts in results for ct in ct_counts})
    header = f"{'cell_type':40s}" + "".join(
        f"  {str(v) if v is not None else 'full':>10}" for v, *_ in results
    )
    print(header)
    print("-" * len(header))
    for ct in all_cts:
        row = f"{ct:40s}" + "".join(
            f"  {ct_counts.get(ct, 0):>10,}" for _, _, _, ct_counts in results
        )
        print(row)


if __name__ == "__main__":
    main()
