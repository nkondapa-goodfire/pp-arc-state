"""
generate_dataset.py — Generate one GRN instance of the SERGIO synthetic dataset.

Designed to be run as a SLURM array job where each task handles one
(grn_type, grn_size, noise_label, seed) combination, or locally to generate
the full dataset sequentially.

Usage
-----
Single instance (SLURM array task):
    uv run python generate_dataset.py \\
        --config generation_configs/dataset1.json \\
        --grn-type BA --grn-size 200 --noise-label low --seed 42

Enumerate all combinations (local):
    uv run python generate_dataset.py \\
        --config generation_configs/dataset1.json \\
        --all

Dry-run (print tasks without running):
    uv run python generate_dataset.py \\
        --config generation_configs/dataset1.json \\
        --all --dry-run

Output per instance
-------------------
{output_dir}/{grn_type}/size_{n:03d}/noise_{label}/grn_{seed:04d}/
    SYN_{global_idx:04d}_{pert_type}.h5ad   (one per gene × pert_type)
    grn_{seed:04d}_manifest.jsonl           (one line per h5ad)
"""

import argparse
import contextlib
import csv
import io
import json
import os
import tempfile
from collections import defaultdict

import anndata
import numpy as np
import scanpy as sc

from SERGIO.sergio import sergio
from grn_utils import (
    generate_er_grn,
    generate_scale_free_grn,
    generate_ba_vm_grn,
    grn_to_sergio_files,
)

# ---------------------------------------------------------------------------
# Gene pool
# ---------------------------------------------------------------------------

def make_gene_pool(pool_size: int) -> list[str]:
    return [f"SYN_{i:04d}" for i in range(pool_size)]


def sample_grn_genes(pool_size: int, grn_size: int, seed: int,
                     strategy: str = "random_draw") -> list[int]:
    """Return sorted indices into the global gene pool for this GRN instance."""
    rng = np.random.default_rng(seed)
    if strategy == "random_draw":
        return sorted(rng.choice(pool_size, size=grn_size, replace=False).tolist())
    raise ValueError(f"Unknown gene sampling strategy: {strategy}")


# ---------------------------------------------------------------------------
# GRN generation
# ---------------------------------------------------------------------------

def make_grn(grn_type: str, n_genes: int, params: dict, seed: int):
    if grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=params["p_edge"], seed=seed)
    elif grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=params["m"], seed=seed)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=tuple(params["m_weights"]), seed=seed)
    raise ValueError(f"Unknown grn_type: {grn_type}")


# ---------------------------------------------------------------------------
# SERGIO simulation
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def run_simulation(G, n_genes: int, n_bins: int, n_sc: int,
                   noise_params: float, sergio_kwargs: dict,
                   tmpdir: str,
                   pert_gene: int | None = None,
                   strength: float = 1.0) -> np.ndarray:
    """
    Run one SERGIO simulation.  Returns (n_genes, n_bins * n_sc).

    If pert_gene is given, apply a perturbation:
      - Remove all incoming + outgoing edges (silences regulatory input/output).
      - Scale the gene's basal rate by (1 - strength) in the Regs file.
        strength=1.0 → KO (basal=0), strength<1.0 → partial KD.
    """
    G_sim = G.copy()
    if pert_gene is not None:
        for s in list(G_sim.successors(pert_gene)):
            G_sim.remove_edge(pert_gene, s)
        for p in list(G_sim.predecessors(pert_gene)):
            G_sim.remove_edge(p, pert_gene)

    file_seed = 0 if pert_gene is None else 1
    targets_path, regs_path = grn_to_sergio_files(
        G_sim, tmpdir, n_bins=n_bins, seed=file_seed
    )

    if pert_gene is not None:
        rows = []
        with open(regs_path, newline="") as f:
            for row in csv.reader(f):
                if int(float(row[0])) == pert_gene:
                    scaled = [str((1.0 - strength) * float(v)) for v in row[1:]]
                    row = [row[0]] + scaled
                rows.append(row)
        with open(regs_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    with suppress_stdout():
        sim = sergio(
            number_genes=n_genes,
            number_bins=n_bins,
            number_sc=n_sc,
            noise_params=noise_params,
            **sergio_kwargs,
        )
        sim.build_graph(targets_path, regs_path, shared_coop_state=2)
        sim.simulate()

    expr_list = sim.getExpressions()   # list of (n_genes, n_sc) per bin
    return np.concatenate(expr_list, axis=1)  # (n_genes, n_bins * n_sc)


# ---------------------------------------------------------------------------
# AnnData construction
# ---------------------------------------------------------------------------

def build_anndata(
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
    Build an AnnData for one perturbation, embedding expression into the
    full 2000-gene pool space.

    obs rows: control cells first, then perturbed cells.
    X: raw SERGIO expression embedded in pool (float32, n_cells × pool_size).
    obsm["X_hvg"]: log-normalized expression (same shape, set by preprocess()).
    """
    pool_size = len(pool_genes)
    n_cells_per_cond = n_bins * n_sc

    # Embed into pool space: (n_cells, pool_size)
    def embed(expr):  # expr: (n_genes, n_cells)
        mat = np.zeros((expr.shape[1], pool_size), dtype=np.float32)
        for local_i, global_i in enumerate(gene_indices):
            mat[:, global_i] = expr[local_i].astype(np.float32)
        return mat

    X_ctrl = embed(ctrl_expr)
    X_pert = embed(pert_expr)
    X = np.vstack([X_ctrl, X_pert])

    # obs metadata
    cell_type_ctrl = [f"bin_{b}" for b in range(n_bins) for _ in range(n_sc)]
    cell_type_pert = cell_type_ctrl[:]
    gene_ctrl = ["non-targeting"] * n_cells_per_cond
    gene_pert = [f"SYN_{global_gene_idx:04d}_{pert_label}"] * n_cells_per_cond
    out_deg_ctrl = [-1] * n_cells_per_cond
    out_deg_pert = [ko_out_degree] * n_cells_per_cond

    obs = {
        "gene":          gene_ctrl + gene_pert,
        "cell_type":     cell_type_ctrl + cell_type_pert,
        "gem_group":     [f"grn_{grn_seed:04d}"] * (2 * n_cells_per_cond),
        "ko_out_degree": out_deg_ctrl + out_deg_pert,
        "pert_strength": [0.0] * n_cells_per_cond + [pert_strength] * n_cells_per_cond,
    }

    import pandas as pd
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(obs),
        var=pd.DataFrame(index=pool_genes),
    )

    adata.uns["grn_type"]         = grn_type
    adata.uns["grn_seed"]         = grn_seed
    adata.uns["grn_size"]         = grn_size
    adata.uns["grn_params"]       = {}   # filled by caller
    adata.uns["grn_gene_indices"] = gene_indices
    adata.uns["noise_label"]      = noise_label
    adata.uns["noise_params"]     = noise_params
    adata.uns["pert_label"]       = pert_label
    adata.uns["pert_gene"]        = f"SYN_{global_gene_idx:04d}"

    return adata


def preprocess(adata: anndata.AnnData) -> None:
    """Normalize → log1p → convert X to sparse CSR float32. Modifies in-place."""
    import scipy.sparse as sp
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Store X as sparse CSR — SERGIO output is ~99.5% zeros, so sparse gives ~200x
    # smaller files vs dense. cell-load reads CSR X directly when embed_key=None.
    x = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    adata.X = sp.csr_matrix(x.astype(np.float32))


# ---------------------------------------------------------------------------
# Per-instance generation
# ---------------------------------------------------------------------------

def generate_instance(
    grn_type: str,
    grn_size: int,
    noise_label: str,
    seed: int,
    cfg: dict,
    output_dir: str,
) -> list[dict]:
    """
    Generate all h5ad files for one GRN instance.
    Returns a list of manifest records (one per h5ad).
    """
    pool_size    = cfg["pool_size"]
    n_bins       = cfg["n_bins"]
    n_sc         = cfg["n_sc"]
    k_perts      = cfg["k_perts"]
    strategy     = cfg["gene_strategy"]
    noise_params = cfg["noise_levels"][noise_label]
    sergio_kw    = dict(cfg["sergio_kwargs"])
    grn_params   = cfg["grn_types"][grn_type]["params"]
    pert_strengths = cfg["pert_strengths"]  # {label: strength}

    pool_genes   = make_gene_pool(pool_size)
    gene_indices = sample_grn_genes(pool_size, grn_size, seed, strategy)

    G = make_grn(grn_type, grn_size, grn_params, seed)

    top_genes = sorted(
        G.nodes(), key=lambda n: G.out_degree(n), reverse=True
    )[:min(k_perts, G.number_of_nodes())]

    noise_str = noise_label
    inst_dir = os.path.join(
        output_dir, grn_type,
        f"size_{grn_size:03d}",
        f"noise_{noise_str}",
        f"grn_{seed:04d}",
    )
    os.makedirs(inst_dir, exist_ok=True)

    # Control simulation (shared across all perturbations)
    with tempfile.TemporaryDirectory() as tmpdir:
        ctrl_expr = run_simulation(
            G, grn_size, n_bins, n_sc, noise_params, sergio_kw, tmpdir
        )

        manifest_records = []

        for local_idx in top_genes:
            global_idx   = gene_indices[local_idx]
            ko_out_degree = int(G.out_degree(local_idx))

            for pert_label, strength in pert_strengths.items():
                pert_expr = run_simulation(
                    G, grn_size, n_bins, n_sc, noise_params, sergio_kw, tmpdir,
                    pert_gene=local_idx, strength=strength,
                )

                adata = build_anndata(
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
                    "path":         fpath,
                    "grn_type":     grn_type,
                    "grn_size":     grn_size,
                    "noise_label":  noise_label,
                    "noise_params": noise_params,
                    "grn_seed":     seed,
                    "pert_gene":    f"SYN_{global_idx:04d}",
                    "pert_type":    pert_label,
                    "pert_strength": strength,
                    "pert_out_degree": ko_out_degree,
                })

    # Write per-instance manifest sidecar
    sidecar = os.path.join(inst_dir, f"grn_{seed:04d}_manifest.jsonl")
    with open(sidecar, "w") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    return manifest_records


# ---------------------------------------------------------------------------
# Task enumeration
# ---------------------------------------------------------------------------

def enumerate_tasks(cfg: dict) -> list[dict]:
    """Return all (grn_type, grn_size, noise_label, seed) combinations."""
    tasks = []
    n_seeds     = cfg["n_seeds"]
    seed_offset = cfg["seed_offset"]
    for grn_type, type_cfg in cfg["grn_types"].items():
        for grn_size in type_cfg["sizes"]:
            for noise_label in cfg["noise_levels"]:
                for seed in range(seed_offset, seed_offset + n_seeds):
                    tasks.append({
                        "grn_type":   grn_type,
                        "grn_size":   grn_size,
                        "noise_label": noise_label,
                        "seed":       seed,
                    })
    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SERGIO synthetic dataset.")
    parser.add_argument("--config",      required=True, help="Path to generation JSON config.")
    parser.add_argument("--output-dir",  default=None,  help="Override output_dir from config.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",      action="store_true", help="Run all tasks sequentially.")
    mode.add_argument("--task-id",  type=int, help="SLURM array mode: 0-based task index into enumerate_tasks().")
    mode.add_argument("--grn-type", help="Single-instance mode: GRN type.")

    parser.add_argument("--grn-size",    type=int, help="Single-instance mode: GRN size.")
    parser.add_argument("--noise-label", help="Single-instance mode: noise label (none/low/high).")
    parser.add_argument("--seed",        type=int, help="Single-instance mode: GRN seed.")
    parser.add_argument("--dry-run",     action="store_true", help="Print tasks without running.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = args.output_dir or cfg["output_dir"]

    if args.task_id is not None:
        tasks = enumerate_tasks(cfg)
        t = tasks[args.task_id]
        print(f"Task {args.task_id}: {t['grn_type']} size={t['grn_size']} "
              f"noise={t['noise_label']} seed={t['seed']}")
        generate_instance(
            grn_type=t["grn_type"],
            grn_size=t["grn_size"],
            noise_label=t["noise_label"],
            seed=t["seed"],
            cfg=cfg,
            output_dir=output_dir,
        )
    elif args.all:
        tasks = enumerate_tasks(cfg)
        print(f"Total tasks: {len(tasks)}")
        if args.dry_run:
            for t in tasks:
                print(f"  {t['grn_type']:<6} size={t['grn_size']:>3}  "
                      f"noise={t['noise_label']:<4}  seed={t['seed']}")
            return
        for i, t in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {t['grn_type']} size={t['grn_size']} "
                  f"noise={t['noise_label']} seed={t['seed']}")
            generate_instance(
                grn_type=t["grn_type"],
                grn_size=t["grn_size"],
                noise_label=t["noise_label"],
                seed=t["seed"],
                cfg=cfg,
                output_dir=output_dir,
            )
    else:
        # Single-instance mode
        for arg, name in [(args.grn_size, "--grn-size"),
                          (args.noise_label, "--noise-label"),
                          (args.seed, "--seed")]:
            if arg is None:
                parser.error(f"{name} is required in single-instance mode.")
        if args.dry_run:
            print(f"Would generate: {args.grn_type} size={args.grn_size} "
                  f"noise={args.noise_label} seed={args.seed}")
            return
        generate_instance(
            grn_type=args.grn_type,
            grn_size=args.grn_size,
            noise_label=args.noise_label,
            seed=args.seed,
            cfg=cfg,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
