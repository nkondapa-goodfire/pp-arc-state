"""
generate_test_set.py — Generate the fixed SERGIO test set.

100 instances, stratified by GRN type and balanced across sizes, noise levels,
and n_bins choices. Seeds are disjoint from the training set (offset 5000+).

Instance assignment (deterministic, round-robin within each type):
  - Seed:        seed_offset + global_index
  - GRN type:    ER (34 instances), BA (33), BA-VM (33)
  - GRN size:    cycles through type's size list
  - Noise level: cycles through [none, low, high]
  - n_bins:      cycles through n_bins_choices [3, 5, 8]

Held-out split:
  The last bin (bin_{n_bins-1}) is flagged as held-out:
    obs["split"] = "train"    — bins 0 .. n_bins-2
    obs["split"] = "held_out" — bin n_bins-1

Output per instance:
  {output_dir}/{grn_type}/size_{n:03d}/noise_{label}/grn_{seed:04d}/
      SYN_{k:04d}_{pert_type}.h5ad
      grn_{seed:04d}_manifest.jsonl

Usage
-----
Single instance (SLURM array task):
    uv run python generate_test_set.py --config generation_configs/test_set.json --task-id 0

All instances (local, sequential):
    uv run python generate_test_set.py --config generation_configs/test_set.json --all

Dry-run:
    uv run python generate_test_set.py --config generation_configs/test_set.json --all --dry-run
"""

import argparse
import csv
import io
import json
import os
import tempfile

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from generate_dataset import (
    make_gene_pool,
    make_grn,
    preprocess,
    run_simulation,
    sample_grn_genes,
    suppress_stdout,
)


# ---------------------------------------------------------------------------
# Task enumeration
# ---------------------------------------------------------------------------

def enumerate_test_tasks(cfg: dict) -> list[dict]:
    """
    Build the test task list using the same cartesian product as enumerate_tasks()
    in generate_dataset.py: grn_type × grn_size × noise_label × seed.

    n_bins is fixed (cfg["n_bins"]). Seeds start at cfg["seed_offset"] and are
    disjoint from the training set.
    """
    n_bins      = cfg["n_bins"]
    n_seeds     = cfg["n_seeds"]
    seed_offset = cfg["seed_offset"]

    tasks = []
    for grn_type, type_cfg in cfg["grn_types"].items():
        for grn_size in type_cfg["sizes"]:
            for noise_label in cfg["noise_levels"]:
                for seed in range(seed_offset, seed_offset + n_seeds):
                    tasks.append({
                        "grn_type":    grn_type,
                        "grn_size":    grn_size,
                        "noise_label": noise_label,
                        "n_bins":      n_bins,
                        "seed":        seed,
                    })
    return tasks


# ---------------------------------------------------------------------------
# AnnData construction (test-set variant: adds obs["split"])
# ---------------------------------------------------------------------------

def build_anndata_test(
    ctrl_expr: np.ndarray,
    pert_expr: np.ndarray,
    gene_indices: list[int],
    global_gene_idx: int,
    pert_label: str,
    pert_strength: float,
    pool_genes: list[str],
    n_bins: int,
    n_sc: int,
    grn_seed: int,
    grn_type: str,
    grn_size: int,
    noise_label: str,
    noise_params: float,
    ko_out_degree: int,
) -> anndata.AnnData:
    """
    Build an AnnData for one perturbation, identical to the train variant but
    with an additional obs["split"] column marking the last bin as held-out.
    """
    pool_size = len(pool_genes)
    n_cells_per_cond = n_bins * n_sc

    def embed(expr):  # expr: (n_genes, n_cells)
        mat = np.zeros((expr.shape[1], pool_size), dtype=np.float32)
        for local_i, global_i in enumerate(gene_indices):
            mat[:, global_i] = expr[local_i].astype(np.float32)
        return mat

    X_ctrl = embed(ctrl_expr)
    X_pert = embed(pert_expr)
    X = np.vstack([X_ctrl, X_pert])

    # obs columns
    cell_type = [f"bin_{b}" for b in range(n_bins) for _ in range(n_sc)]
    split     = ["held_out" if b == n_bins - 1 else "train"
                 for b in range(n_bins) for _ in range(n_sc)]

    obs = {
        "gene":          ["non-targeting"] * n_cells_per_cond
                         + [f"SYN_{global_gene_idx:04d}_{pert_label}"] * n_cells_per_cond,
        "cell_type":     cell_type + cell_type,
        "gem_group":     [f"grn_{grn_seed:04d}"] * (2 * n_cells_per_cond),
        "ko_out_degree": [-1] * n_cells_per_cond + [ko_out_degree] * n_cells_per_cond,
        "pert_strength": [0.0] * n_cells_per_cond + [pert_strength] * n_cells_per_cond,
        "split":         split + split,
    }

    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(obs),
        var=pd.DataFrame(index=pool_genes),
    )

    adata.uns["grn_type"]         = grn_type
    adata.uns["grn_seed"]         = grn_seed
    adata.uns["grn_size"]         = grn_size
    adata.uns["grn_params"]       = {}
    adata.uns["grn_gene_indices"] = gene_indices
    adata.uns["noise_label"]      = noise_label
    adata.uns["noise_params"]     = noise_params
    adata.uns["pert_label"]       = pert_label
    adata.uns["pert_gene"]        = f"SYN_{global_gene_idx:04d}"
    adata.uns["n_bins"]           = n_bins
    adata.uns["held_out_bin"]     = f"bin_{n_bins - 1}"

    return adata


# ---------------------------------------------------------------------------
# Per-instance generation
# ---------------------------------------------------------------------------

def generate_test_instance(
    grn_type: str,
    grn_size: int,
    n_bins: int,
    noise_label: str,
    seed: int,
    cfg: dict,
    output_dir: str,
) -> list[dict]:
    pool_size    = cfg["pool_size"]
    n_sc         = cfg["n_sc"]
    k_perts      = cfg["k_perts"]
    strategy     = cfg["gene_strategy"]
    noise_params = cfg["noise_levels"][noise_label]
    sergio_kw    = dict(cfg["sergio_kwargs"])
    grn_params   = cfg["grn_types"][grn_type]["params"]
    pert_strengths = cfg["pert_strengths"]

    pool_genes   = make_gene_pool(pool_size)
    gene_indices = sample_grn_genes(pool_size, grn_size, seed, strategy)

    G = make_grn(grn_type, grn_size, grn_params, seed)

    top_genes = sorted(
        G.nodes(), key=lambda n: G.out_degree(n), reverse=True
    )[:min(k_perts, G.number_of_nodes())]

    inst_dir = os.path.join(
        output_dir, grn_type,
        f"size_{grn_size:03d}",
        f"noise_{noise_label}",
        f"grn_{seed:04d}",
    )
    os.makedirs(inst_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctrl_expr = run_simulation(
            G, grn_size, n_bins, n_sc, noise_params, sergio_kw, tmpdir
        )

        manifest_records = []

        for local_idx in top_genes:
            global_idx    = gene_indices[local_idx]
            ko_out_degree = int(G.out_degree(local_idx))

            for pert_label, strength in pert_strengths.items():
                pert_expr = run_simulation(
                    G, grn_size, n_bins, n_sc, noise_params, sergio_kw, tmpdir,
                    pert_gene=local_idx, strength=strength,
                )

                adata = build_anndata_test(
                    ctrl_expr=ctrl_expr,
                    pert_expr=pert_expr,
                    gene_indices=gene_indices,
                    global_gene_idx=global_idx,
                    pert_label=pert_label,
                    pert_strength=strength,
                    pool_genes=pool_genes,
                    n_bins=n_bins,
                    n_sc=n_sc,
                    grn_seed=seed,
                    grn_type=grn_type,
                    grn_size=grn_size,
                    noise_label=noise_label,
                    noise_params=noise_params,
                    ko_out_degree=ko_out_degree,
                )
                adata.uns["grn_params"] = grn_params

                preprocess(adata)

                fname = f"SYN_{global_idx:04d}_{pert_label}.h5ad"
                fpath = os.path.join(inst_dir, fname)
                adata.write_h5ad(fpath)

                manifest_records.append({
                    "path":           fpath,
                    "grn_type":       grn_type,
                    "grn_size":       grn_size,
                    "noise_label":    noise_label,
                    "noise_params":   noise_params,
                    "grn_seed":       seed,
                    "n_bins":         n_bins,
                    "held_out_bin":   f"bin_{n_bins - 1}",
                    "pert_gene":      f"SYN_{global_idx:04d}",
                    "pert_type":      pert_label,
                    "pert_strength":  strength,
                    "pert_out_degree": ko_out_degree,
                })

    sidecar = os.path.join(inst_dir, f"grn_{seed:04d}_manifest.jsonl")
    with open(sidecar, "w") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    return manifest_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SERGIO fixed test set.")
    parser.add_argument("--config",     required=True, help="Path to test set JSON config.")
    parser.add_argument("--output-dir", default=None,  help="Override output_dir from config.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",     action="store_true", help="Run all instances sequentially.")
    mode.add_argument("--task-id", type=int, help="SLURM array mode: 0-based task index.")

    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = args.output_dir or cfg["output_dir"]
    tasks = enumerate_test_tasks(cfg)

    if args.task_id is not None:
        t = tasks[args.task_id]
        print(f"Task {args.task_id}: {t['grn_type']} size={t['grn_size']} "
              f"n_bins={t['n_bins']} noise={t['noise_label']} seed={t['seed']}")
        if not args.dry_run:
            generate_test_instance(
                grn_type=t["grn_type"],
                grn_size=t["grn_size"],
                n_bins=t["n_bins"],
                noise_label=t["noise_label"],
                seed=t["seed"],
                cfg=cfg,
                output_dir=output_dir,
            )
    else:
        print(f"Total test instances: {len(tasks)}")
        for i, t in enumerate(tasks):
            print(f"  [{i:>3}] {t['grn_type']:<6} size={t['grn_size']:>3}  "
                  f"n_bins={t['n_bins']}  noise={t['noise_label']:<4}  seed={t['seed']}")
        if args.dry_run:
            return
        for i, t in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {t['grn_type']} size={t['grn_size']} "
                  f"n_bins={t['n_bins']} noise={t['noise_label']} seed={t['seed']}")
            generate_test_instance(
                grn_type=t["grn_type"],
                grn_size=t["grn_size"],
                n_bins=t["n_bins"],
                noise_label=t["noise_label"],
                seed=t["seed"],
                cfg=cfg,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
