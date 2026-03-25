"""
test_graph_noise_invariance.py — Verify that the same GRN seed produces an
identical graph topology and edge attributes regardless of noise level.

Noise (`noise_params`) is passed only to the SERGIO simulator, not to the
graph generator.  This test confirms that property holds for all GRN types,
sizes, and seeds defined in a generation config.

Usage
-----
    cd /path/to/sergio
    uv run python scripts/test_graph_noise_invariance.py
    uv run python scripts/test_graph_noise_invariance.py --config generation_configs/dataset_ppt.json
    uv run python scripts/test_graph_noise_invariance.py --seeds 0 1 2 --out-dir test_outputs/noise_invariance

Outputs
-------
test_outputs/noise_invariance/
    results.txt              — pass/fail table (also printed to stdout)
    <grn_type>_size<n>_seed<s>_graph.png  — graph drawing with edge attributes
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# Ensure the parent directory (where grn_utils lives) is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grn_utils import generate_er_grn, generate_scale_free_grn, generate_ba_vm_grn


# ---------------------------------------------------------------------------
# Defaults (match dataset_ppt.json)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generation_configs", "dataset_ppt.json",
)
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_outputs", "noise_invariance",
)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def make_grn(grn_type: str, n_genes: int, params: dict, seed: int) -> nx.DiGraph:
    if grn_type == "ER":
        return generate_er_grn(n_genes, p_edge=params["p_edge"], seed=seed)
    elif grn_type == "BA":
        return generate_scale_free_grn(n_genes, n_edges_per_new_node=params["m"], seed=seed)
    elif grn_type == "BA-VM":
        return generate_ba_vm_grn(n_genes, m_weights=tuple(params["m_weights"]), seed=seed)
    raise ValueError(f"Unknown grn_type: {grn_type}")


# ---------------------------------------------------------------------------
# Graph comparison
# ---------------------------------------------------------------------------

def graphs_equal(G1: nx.DiGraph, G2: nx.DiGraph) -> tuple[bool, list[str]]:
    """
    Return (equal, reasons) where reasons is a list of failure descriptions.
    Checks: same node set, same edge set, same edge attributes (K, hill, sign).
    """
    failures = []

    if set(G1.nodes()) != set(G2.nodes()):
        failures.append(
            f"node sets differ: {set(G1.nodes()) ^ set(G2.nodes())} extra/missing"
        )

    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    if edges1 != edges2:
        only_in_1 = edges1 - edges2
        only_in_2 = edges2 - edges1
        if only_in_1:
            failures.append(f"edges only in G1: {sorted(only_in_1)[:5]}{'...' if len(only_in_1) > 5 else ''}")
        if only_in_2:
            failures.append(f"edges only in G2: {sorted(only_in_2)[:5]}{'...' if len(only_in_2) > 5 else ''}")

    if not failures:
        for u, v in edges1:
            for attr in ("K", "hill", "sign"):
                v1 = G1[u][v].get(attr)
                v2 = G2[u][v].get(attr)
                if v1 != v2:
                    failures.append(f"edge ({u},{v}) attr '{attr}' differs: {v1} vs {v2}")
                    if len(failures) >= 5:
                        failures.append("... (truncated)")
                        return False, failures

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_graph(G: nx.DiGraph, title: str, ax: plt.Axes) -> None:
    """Draw the GRN on ax with master regulators highlighted."""
    master_regs = {n for n in G.nodes() if G.in_degree(n) == 0}

    node_colors = ["#e74c3c" if n in master_regs else "#3498db" for n in G.nodes()]

    edge_colors = []
    for u, v, d in G.edges(data=True):
        edge_colors.append("#27ae60" if d.get("sign", 1) > 0 else "#e74c3c")

    pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, G.number_of_nodes() ** 0.5))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=120, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           arrows=True, arrowsize=10, alpha=0.7,
                           connectionstyle="arc3,rad=0.1")
    if G.number_of_nodes() <= 30:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_graph_figure(G: nx.DiGraph, noise_labels: list[str],
                      grn_type: str, grn_size: int, seed: int,
                      all_equal: bool, out_dir: str) -> str:
    """
    Draw the graph once (they should be identical) with a noise-level annotation,
    and annotate with PASS/FAIL.
    """
    fig, axes = plt.subplots(1, len(noise_labels), figsize=(4 * len(noise_labels), 4))
    if len(noise_labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, noise_labels):
        draw_graph(G, f"noise={label}", ax)

    status = "PASS — graphs identical across noise levels" if all_equal else "FAIL — graphs differ"
    status_color = "#27ae60" if all_equal else "#e74c3c"
    fig.suptitle(
        f"{grn_type}  size={grn_size}  seed={seed}\n{status}",
        fontsize=10, color=status_color, fontweight="bold"
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="master regulator"),
        mpatches.Patch(color="#3498db", label="downstream gene"),
        mpatches.Patch(color="#27ae60", label="activation edge"),
        mpatches.Patch(color="#e74c3c", label="repression edge"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fname = f"{grn_type}_size{grn_size:03d}_seed{seed:04d}_graph.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to generation config JSON (default: dataset_ppt.json)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds to test (default: all seeds in config)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Directory for output figures and results.txt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    noise_labels = list(cfg["noise_levels"].keys())   # ["none", "low", "high"]
    n_seeds = cfg["n_seeds"]
    seed_offset = cfg.get("seed_offset", 0)
    seeds = args.seeds if args.seeds is not None else list(range(seed_offset, seed_offset + n_seeds))

    os.makedirs(args.out_dir, exist_ok=True)

    results = []   # list of dicts for the summary table
    n_pass = 0
    n_fail = 0

    for grn_type, type_cfg in cfg["grn_types"].items():
        params = type_cfg["params"]
        for grn_size in type_cfg["sizes"]:
            for seed in seeds:
                # Generate the graph once per noise label — should be identical
                graphs = {}
                for label in noise_labels:
                    graphs[label] = make_grn(grn_type, grn_size, params, seed)

                # Compare each pair against the first
                ref_label = noise_labels[0]
                G_ref = graphs[ref_label]
                all_equal = True
                failure_msgs = []

                for label in noise_labels[1:]:
                    equal, failures = graphs_equal(G_ref, graphs[label])
                    if not equal:
                        all_equal = False
                        failure_msgs.extend(
                            [f"  noise={ref_label} vs noise={label}: {f}" for f in failures]
                        )

                status = "PASS" if all_equal else "FAIL"
                if all_equal:
                    n_pass += 1
                else:
                    n_fail += 1

                results.append({
                    "grn_type": grn_type,
                    "grn_size": grn_size,
                    "seed": seed,
                    "n_nodes": G_ref.number_of_nodes(),
                    "n_edges": G_ref.number_of_edges(),
                    "status": status,
                    "failures": failure_msgs,
                })

                # Save figure
                fig_path = save_graph_figure(
                    G_ref, noise_labels, grn_type, grn_size, seed,
                    all_equal, args.out_dir,
                )

                line = (f"{status}  {grn_type:<6}  size={grn_size:3d}  seed={seed:4d}"
                        f"  nodes={G_ref.number_of_nodes():3d}  edges={G_ref.number_of_edges():3d}"
                        f"  → {os.path.basename(fig_path)}")
                print(line)
                for msg in failure_msgs:
                    print(msg)

    # Summary
    total = n_pass + n_fail
    summary_lines = [
        "",
        f"{'='*60}",
        f"Results: {n_pass}/{total} passed  ({n_fail} failed)",
        f"{'='*60}",
    ]
    for line in summary_lines:
        print(line)

    # Write results.txt
    results_path = os.path.join(args.out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Graph noise-invariance test\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Noise labels tested: {noise_labels}\n\n")
        for r in results:
            f.write(f"{r['status']}  {r['grn_type']:<6}  size={r['grn_size']:3d}"
                    f"  seed={r['seed']:4d}  nodes={r['n_nodes']:3d}  edges={r['n_edges']:3d}\n")
            for msg in r["failures"]:
                f.write(msg + "\n")
        f.write("\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nResults written to: {results_path}")
    print(f"Figures written to: {args.out_dir}/")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
