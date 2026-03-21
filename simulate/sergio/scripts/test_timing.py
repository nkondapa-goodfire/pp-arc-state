"""
test_timing.py — Time one SERGIO simulation at each (grn_size, grn_type) and
extrapolate to the full dataset generation cost.

Strategy: time with small probe values (n_sc_probe, n_bins_probe, k_perts_probe),
then scale up linearly to the real generation parameters. All three dimensions
scale linearly with simulation time.

Usage
-----
    uv run python test_timing.py
    uv run python test_timing.py --n-cpus 128
"""

import argparse
import contextlib
import csv
import io
import os
import tempfile
import time

import numpy as np
from tqdm import tqdm

from SERGIO.sergio import sergio
from grn_utils import (
    generate_er_grn,
    generate_ba_vm_grn,
    generate_scale_free_grn,
    grn_to_sergio_files,
)

# ---------------------------------------------------------------------------
# Config — probe (fast) vs real (extrapolation target)
# ---------------------------------------------------------------------------

GRN_SIZES_BY_TYPE = {
    "ER":    [10, 50, 100],
    "BA":    [10, 50, 100, 200],
    "BA-VM": [10, 50, 100, 200],
}

# Probe: small values so each cell runs in ~1-2s
N_BINS_PROBE  = 2
N_SC_PROBE    = 10
K_PERTS_PROBE = 2     # just 2 perturbations to capture per-sim overhead

# Real: actual generation parameters to extrapolate to
N_BINS_REAL   = 5
N_SC_REAL     = 200
K_PERTS_REAL  = 10
N_PERT_TYPES  = 4     # KO + KD_010 + KD_050 + KD_080

# Full dataset size
INSTANCES_PER_CELL = 100   # seeds
NOISE_LEVELS       = 3
N_CPUS_SLURM       = 64

NOISE       = 0.1
ER_P_EDGE   = 0.08
BA_M        = 2
BA_VM_WEIGHTS = (0.57, 0.29, 0.14)

BASE_KWARGS = dict(
    noise_type="dpd",
    noise_params=NOISE,
    decays=0.8,
    sampling_state=15,
    dt=0.01,
    dynamics=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph(grn_type: str, n_genes: int, seed: int):
    if grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=ER_P_EDGE, seed=seed)
    elif grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=BA_M, seed=seed)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=BA_VM_WEIGHTS, seed=seed)
    raise ValueError(grn_type)


@contextlib.contextmanager
def suppress_stdout():
    """Silence SERGIO's level-by-level print statements."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def run_sim(G, n_genes, n_bins, n_sc, tmpdir, ko_gene=None, strength=1.0):
    G_sim = G.copy()
    if ko_gene is not None:
        for s in list(G_sim.successors(ko_gene)):
            G_sim.remove_edge(ko_gene, s)
        for p in list(G_sim.predecessors(ko_gene)):
            G_sim.remove_edge(p, ko_gene)

    seed = 0 if ko_gene is None else 1
    targets, regs = grn_to_sergio_files(G_sim, tmpdir, n_bins=n_bins, seed=seed)

    if ko_gene is not None:
        rows = []
        with open(regs, newline="") as f:
            for row in csv.reader(f):
                if int(float(row[0])) == ko_gene:
                    row = [row[0]] + [str((1.0 - strength) * float(v)) for v in row[1:]]
                rows.append(row)
        with open(regs, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    with suppress_stdout():
        sim = sergio(number_genes=n_genes, number_bins=n_bins, number_sc=n_sc, **BASE_KWARGS)
        sim.build_graph(targets, regs, shared_coop_state=2)
        sim.simulate()
    return sim.getExpressions()


def time_probe(grn_type, n_genes):
    """Time 1 control + K_PERTS_PROBE perturbations with small probe params."""
    G = make_graph(grn_type, n_genes, seed=0)
    k = min(K_PERTS_PROBE, G.number_of_nodes())
    top = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)[:k]
    n_sims = 1 + k  # 1 control + k perts (1 strength only; pert types scale linearly)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_sim(G, n_genes, N_BINS_PROBE, N_SC_PROBE, tmpdir)
        for gene in top:
            run_sim(G, n_genes, N_BINS_PROBE, N_SC_PROBE, tmpdir, ko_gene=gene, strength=1.0)
    elapsed = time.perf_counter() - t0

    s_per_sim = elapsed / n_sims
    return s_per_sim, n_sims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cpus", type=int, default=N_CPUS_SLURM)
    args = parser.parse_args()

    # Scale factors from probe → real
    sc_scale   = N_SC_REAL   / N_SC_PROBE
    bin_scale  = N_BINS_REAL / N_BINS_PROBE
    pert_scale = (1 + K_PERTS_REAL * N_PERT_TYPES) / (1 + K_PERTS_PROBE)

    print(f"Probe params:  n_sc={N_SC_PROBE}  n_bins={N_BINS_PROBE}  k_perts={K_PERTS_PROBE}")
    print(f"Real params:   n_sc={N_SC_REAL}   n_bins={N_BINS_REAL}   k_perts={K_PERTS_REAL} × {N_PERT_TYPES} types")
    print(f"Scale factors: n_sc×{sc_scale:.0f}  n_bins×{bin_scale:.1f}  perts×{pert_scale:.1f}\n")

    cells = [(gt, ng) for gt, sizes in GRN_SIZES_BY_TYPE.items() for ng in sizes]
    records = []

    for grn_type, n_genes in tqdm(cells, desc="Timing cells", unit="cell"):
        s_per_sim, n_sims = time_probe(grn_type, n_genes)

        # Extrapolate to real single-instance time
        real_s = s_per_sim * sc_scale * bin_scale * pert_scale

        total_instances = INSTANCES_PER_CELL * NOISE_LEVELS
        total_cpu_hr    = real_s * total_instances / 3600
        wall_hr         = total_cpu_hr / args.n_cpus

        records.append({
            "grn_type":     grn_type,
            "n_genes":      n_genes,
            "probe_s_per_sim": s_per_sim,
            "real_s":       real_s,
            "instances":    total_instances,
            "total_cpu_hr": total_cpu_hr,
            "wall_hr":      wall_hr,
        })

    # --- Summary table ---
    print()
    print(f"{'─'*80}")
    print(f"{'type':<8} {'genes':>6} {'real s/inst':>12} {'instances':>10} "
          f"{'cpu-hrs':>9} {'wall-hrs':>10}  (@{args.n_cpus} CPUs)")
    print(f"{'─'*80}")
    total_cpu = 0.0
    for r in records:
        print(f"{r['grn_type']:<8} {r['n_genes']:>6} {r['real_s']:>12.1f} "
              f"{r['instances']:>10} {r['total_cpu_hr']:>9.2f} {r['wall_hr']:>10.2f}")
        total_cpu += r["total_cpu_hr"]
    print(f"{'─'*80}")
    print(f"{'TOTAL':<8} {'':>6} {'':>12} {'':>10} "
          f"{total_cpu:>9.2f} {total_cpu/args.n_cpus:>10.2f}")
    print(f"{'─'*80}")

    # --- CSV ---
    os.makedirs("test_outputs", exist_ok=True)
    csv_path = "test_outputs/timing_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
