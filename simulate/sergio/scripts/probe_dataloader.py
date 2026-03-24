"""
probe_dataloader.py — Dry-run the state PerturbationDataModule and print cell_type details.

Instantiates the dataloader exactly as training does (via get_datamodule), runs setup(),
then reports:
  - All unique cell_type values seen across train/val/test splits
  - How many cells per cell_type per split
  - A few sample batch entries showing cell_type, pert_name, dataset_name

Usage
-----
cd /mnt/polished-lake/home/nkondapaneni/state

# SERGIO PPT training toml:
uv run python simulate/sergio/scripts/probe_dataloader.py \\
    --toml simulate/sergio/configs/sergio_ppt_train.toml \\
    --cell-type-key cell_type \\
    --pert-col gene \\
    --batch-col gem_group \\
    --control-pert non-targeting \\
    --embed-key X_hvg \\
    --output-space gene

# SERGIO TGT test toml:
uv run python simulate/sergio/scripts/probe_dataloader.py \\
    --toml simulate/sergio/configs/sergio_tgt_test.toml \\
    --cell-type-key cell_type \\
    --pert-col gene \\
    --batch-col gem_group \\
    --control-pert non-targeting \\
    --embed-key X_hvg \\
    --output-space gene
"""

import argparse
import collections
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Probe state PerturbationDataModule cell_type info.")
    parser.add_argument("--toml",             required=True, help="Path to TOML config.")
    parser.add_argument("--cell-type-key",    default="cell_type")
    parser.add_argument("--pert-col",         default="gene")
    parser.add_argument("--batch-col",        default="gem_group")
    parser.add_argument("--control-pert",     default="non-targeting")
    parser.add_argument("--embed-key",        default="X_hvg")
    parser.add_argument("--output-space",     default="gene")
    parser.add_argument("--num-workers",      type=int, default=0)
    parser.add_argument("--n-sample-batches", type=int, default=3,
                        help="Number of train batches to sample for per-batch cell_type reporting.")
    args = parser.parse_args()

    from cell_load.utils.modules import get_datamodule

    kwargs = {
        "toml_config_path":    args.toml,
        "embed_key":           args.embed_key,
        "output_space":        args.output_space,
        "pert_col":            args.pert_col,
        "cell_type_key":       args.cell_type_key,
        "batch_col":           args.batch_col,
        "control_pert":        args.control_pert,
        "num_workers":         args.num_workers,
        "pin_memory":          False,
        "n_basal_samples":     1,
        "basal_mapping_strategy": "random",
        "should_yield_control_cells": True,
        "map_controls":        True,
        "perturbation_features_file": None,
        "store_raw_basal":     False,
        "downsample":          None,
        "downsample_cells":    None,
        "additional_obs":      [],
        "val_subsample_fraction": 1.0,
    }

    print(f"TOML:          {args.toml}")
    print(f"cell_type_key: {args.cell_type_key}")
    print(f"embed_key:     {args.embed_key}")
    print(f"output_space:  {args.output_space}")
    print()

    dm = get_datamodule("PerturbationDataModule", kwargs, batch_size=64, cell_sentence_len=64)
    dm.setup(stage="fit")

    # ── Global one-hot map ────────────────────────────────────────────────────
    ct_map = dm.cell_type_onehot_map or {}
    print(f"=== cell_type_onehot_map ({len(ct_map)} unique cell types) ===")
    for ct in sorted(ct_map):
        print(f"  {ct}")
    print()

    # ── Per-split cell_type counts from datasets ──────────────────────────────
    for split_name, datasets in [
        ("train", dm.train_datasets),
        ("val",   dm.val_datasets),
        ("test",  dm.test_datasets),
    ]:
        if not datasets:
            continue
        ct_counts: dict[str, int] = collections.Counter()
        for ds in datasets:
            # Unwrap Subset if needed
            base = ds.dataset if hasattr(ds, "dataset") else ds
            for ct_code, ct_name in enumerate(base.cell_type_categories):
                mask = base.metadata_cache.cell_type_codes[base.all_indices] == ct_code
                ct_counts[ct_name] += int(mask.sum())

        print(f"=== {split_name} split: {sum(ct_counts.values())} cells across {len(ct_counts)} cell types ===")
        for ct, n in sorted(ct_counts.items()):
            print(f"  {ct:40s}  {n:>8,} cells")
        print()

    # ── Sample a few train batches ────────────────────────────────────────────
    # Infer sentence_len from the datamodule
    try:
        sentence_len = dm.cell_sentence_len
    except AttributeError:
        sentence_len = 64

    # Sample from test (eval targets = perturbed cells) if available,
    # otherwise fall back to train. For zeroshot TOMLs, train_datasets
    # contains only control/basal cells — test_datasets has the perturbed targets.
    if dm.test_datasets:
        dl = dm.test_dataloader()
        sample_split = "test (perturbed eval targets)"
    elif dm.train_datasets:
        dl = dm.train_dataloader()
        sample_split = "train"
    else:
        print("No datasets available — skipping batch sampling.")
        return

    print(f"=== Sampling {args.n_sample_batches} {sample_split} batch(es)  (sentence_len={sentence_len}) ===")
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.n_sample_batches:
            break
        cell_types = batch.get("cell_type", [])
        pert_names = batch.get("pert_name", [])
        dataset_names = batch.get("dataset_name", [])
        n_cells = len(cell_types)
        n_sentences = n_cells // sentence_len

        ct_in_batch = collections.Counter(cell_types)
        print(f"\nBatch {batch_idx}  ({n_cells} cells, {n_sentences} sentences of {sentence_len})")
        print("  cell_type counts in batch:")
        for ct, n in sorted(ct_in_batch.items()):
            print(f"    {ct:40s}  {n}")

        # Per-sentence cell_type + pert composition
        if n_sentences > 0 and sentence_len > 1:
            print(f"  cell_type counts per sentence (first 3 sentences):")
            for s in range(min(3, n_sentences)):
                sl = slice(s * sentence_len, (s + 1) * sentence_len)
                ct_in_sent = collections.Counter(cell_types[sl])
                pt_in_sent = collections.Counter(pert_names[sl]) if pert_names else {}
                unique_cts = "  ".join(f"{ct}:{n}" for ct, n in sorted(ct_in_sent.items()))
                unique_pts = "  ".join(f"{pt}:{n}" for pt, n in sorted(pt_in_sent.items()))
                print(f"    sentence {s}: cell_type=[{unique_cts}]  pert=[{unique_pts}]")

            # Summarise across all sentences: avg unique cell types per sentence
            unique_per_sent = []
            mixed_sentences = 0
            for s in range(n_sentences):
                sl = slice(s * sentence_len, (s + 1) * sentence_len)
                u = len(set(cell_types[sl]))
                unique_per_sent.append(u)
                if u > 1:
                    mixed_sentences += 1
            print(f"  avg unique cell_types per sentence: {sum(unique_per_sent)/len(unique_per_sent):.2f}")
            print(f"  mixed sentences (>1 cell_type):     {mixed_sentences}/{n_sentences}")

        if pert_names:
            unique_perts = sorted(set(pert_names))[:5]
            print(f"  example pert_names (first 5): {unique_perts}")
        if dataset_names:
            print(f"  dataset_names: {sorted(set(dataset_names))}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
