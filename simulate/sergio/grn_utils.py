"""
grn_utils.py - Utilities for generating Gene Regulatory Networks (GRNs)
and writing them to SERGIO-compatible input files.

Two topology types are supported:
  - Erdős–Rényi (ER): uniform random edges, binomial degree distribution.
  - Barabási–Albert (BA): preferential attachment, power-law degree distribution.

Both are converted to DAGs before use (SERGIO requires topological ordering).
"""

import os
import csv
import numpy as np
import networkx as nx
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core GRN generation
# ---------------------------------------------------------------------------

def generate_er_grn(
    n_genes: int,
    p_edge: float,
    frac_repression: float = 0.3,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Generate an Erdős–Rényi random GRN as a DAG.

    Edges are oriented according to a random permutation of nodes — any edge
    (u, v) is kept only if rank[u] < rank[v].  This preserves ~50% of edges
    while producing a balanced depth distribution (not the deep chains created
    by DFS back-edge removal).

    Parameters
    ----------
    n_genes : int
        Total number of genes (nodes).
    p_edge : float
        Edge probability *before* DAG pruning.  ~50% of edges are kept after
        orientation, so set ``p_edge`` at ~2× the desired post-DAG density.
    frac_repression : float
        Fraction of edges that are repressive (K < 0).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        DAG with edge attributes ``sign``, ``K``, ``hill``.
    """
    rng = np.random.default_rng(seed)
    nx_seed = int(rng.integers(0, 2**31))

    G_raw = nx.erdos_renyi_graph(n_genes, p_edge, seed=nx_seed, directed=True)
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

    # Orient edges by a random permutation: keep (u,v) only if rank[u] < rank[v].
    # This gives a balanced DAG depth unlike DFS back-edge removal.
    order = rng.permutation(n_genes).tolist()
    rank = {node: i for i, node in enumerate(order)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))
    G.add_edges_from((u, v) for u, v in G_raw.edges() if rank[u] < rank[v])

    G = _ensure_connected(G, rng)

    n_rep = max(1, int(frac_repression * G.number_of_edges()))
    rep_edges = set(
        map(tuple, rng.choice(list(G.edges()), size=n_rep, replace=False))
    )
    for u, v in G.edges():
        sign = -1 if (u, v) in rep_edges else +1
        K = sign * rng.uniform(0.5, 2.0)
        hill = rng.uniform(1.5, 3.0)
        G[u][v]["sign"] = sign
        G[u][v]["K"] = float(K)
        G[u][v]["hill"] = float(hill)

    return G


def generate_scale_free_grn(
    n_genes: int,
    n_edges_per_new_node: int = 2,
    frac_repression: float = 0.3,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Generate a directed scale-free GRN using preferential attachment.

    Uses NetworkX's ``barabasi_albert_graph`` (undirected BA model), then
    orients each edge from the lower-index node to the higher-index node.
    Node construction order (0, 1, ..., n-1) is the natural topological
    ordering for BA: early nodes are hubs that later nodes attach to, so
    lower-index = regulator, higher-index = target.  This produces a valid
    DAG by construction with no back-edge removal needed, and preserves the
    power-law degree distribution.

    Parameters
    ----------
    n_genes : int
        Total number of genes (nodes).
    n_edges_per_new_node : int
        ``m`` parameter of BA growth: edges added per new node (controls
        density).  Default 2 gives mean in-degree ≈ 2.
    frac_repression : float
        Fraction of edges that are repressive (K < 0).  Range [0, 1].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        Directed graph where each edge has attributes:
          - ``sign``  : +1 (activation) or -1 (repression)
          - ``K``     : interaction strength (float, signed)
          - ``hill``  : Hill coefficient (float >= 1)
    """
    rng = np.random.default_rng(seed)
    nx_seed = int(rng.integers(0, 2**31))

    # Undirected BA graph; orient edges by construction order (lower → higher).
    # This is already a DAG — no cycle removal needed.
    raw = nx.barabasi_albert_graph(n_genes, n_edges_per_new_node, seed=nx_seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))
    for u, v in raw.edges():
        if u < v:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    # Ensure connectivity: if any gene has zero total degree, attach it.
    G = _ensure_connected(G, rng)

    # Assign interaction parameters to every edge
    n_rep = max(1, int(frac_repression * G.number_of_edges()))
    rep_edges = set(
        map(tuple, rng.choice(list(G.edges()), size=n_rep, replace=False))
    )

    for u, v in G.edges():
        sign = -1 if (u, v) in rep_edges else +1
        K = sign * rng.uniform(0.5, 2.0)
        hill = rng.uniform(1.5, 3.0)
        G[u][v]["sign"] = sign
        G[u][v]["K"] = float(K)
        G[u][v]["hill"] = float(hill)

    return G


def generate_ba_vm_grn(
    n_genes: int,
    m_weights: tuple[float, float, float] = (0.57, 0.29, 0.14),
    frac_repression: float = 0.3,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Generate a directed GRN using BA preferential attachment with variable m.

    Each new node draws its number of regulators m from {1, 2, 3} according to
    ``m_weights``.  The default weights follow a geometric distribution (p=0.5)
    truncated and renormalized to {1,2,3}: ~57% of nodes get 1 regulator, ~29%
    get 2, ~14% get 3.  Edges are oriented lower-index → higher-index, producing
    a DAG by construction.  This gives a more sparse and heterogeneous in-degree
    distribution than the fixed-m BA model.

    Parameters
    ----------
    n_genes : int
        Total number of genes (nodes).
    m_weights : tuple of three floats
        Probabilities for m ∈ {1, 2, 3}.  Must sum to 1.
    frac_repression : float
        Fraction of edges that are repressive (K < 0).  Range [0, 1].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        DAG with edge attributes ``sign``, ``K``, ``hill``.
    """
    rng = np.random.default_rng(seed)
    m_choices = [1, 2, 3]
    weights = np.array(m_weights, dtype=float)
    weights /= weights.sum()

    # Seed graph: 3 nodes, fully connected (lower → higher)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])
    out_deg = {0: 2, 1: 1, 2: 0}

    for new_node in range(3, n_genes):
        m_k = int(rng.choice(m_choices, p=weights))
        existing = list(range(new_node))
        pa_weights = np.array([out_deg.get(u, 0) + 1 for u in existing], dtype=float)
        pa_weights /= pa_weights.sum()
        targets = rng.choice(
            existing, size=min(m_k, len(existing)), replace=False, p=pa_weights
        )
        for t in targets:
            G.add_edge(t, new_node)
            out_deg[t] = out_deg.get(t, 0) + 1
        out_deg.setdefault(new_node, 0)

    G = _ensure_connected(G, rng)

    n_rep = max(1, int(frac_repression * G.number_of_edges()))
    rep_edges = set(
        map(tuple, rng.choice(list(G.edges()), size=n_rep, replace=False))
    )
    for u, v in G.edges():
        sign = -1 if (u, v) in rep_edges else +1
        K = sign * rng.uniform(0.5, 2.0)
        hill = rng.uniform(1.5, 3.0)
        G[u][v]["sign"] = sign
        G[u][v]["K"] = float(K)
        G[u][v]["hill"] = float(hill)

    return G


# ---------------------------------------------------------------------------
# SERGIO file writers
# ---------------------------------------------------------------------------

def grn_to_sergio_files(
    G: nx.DiGraph,
    out_dir: str,
    n_bins: int = 2,
    basal_low: float = 0.2,
    basal_high: float = 1.5,
    seed: int | None = None,
) -> tuple[str, str]:
    """
    Write SERGIO-compatible Interaction (targets) and Regs files from a GRN.

    Parameters
    ----------
    G : nx.DiGraph
        Scale-free GRN produced by :func:`generate_scale_free_grn`.
    out_dir : str
        Directory to write the files into.
    n_bins : int
        Number of cell types / bins (sets basal rate columns in Regs file).
    basal_low, basal_high : float
        Range for sampling master regulator basal expression rates.
    seed : int or None
        Random seed for basal rate sampling.

    Returns
    -------
    (targets_path, regs_path) : tuple[str, str]
        Paths to the written files.

    File formats
    ------------
    Interaction file (one row per regulated gene):
        target_id, n_regs, reg_id_0, ..., K_0, ..., hill_0, ...

    Regs file (one row per master regulator):
        reg_id, basal_bin0, basal_bin1, ...
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    master_regs = _get_master_regulators(G)
    # Group edges by target
    tgt_regs: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        tgt_regs[v].append((u, data["K"], data["hill"]))

    targets_path = os.path.join(out_dir, "Interaction.txt")
    regs_path = os.path.join(out_dir, "Regs.txt")

    with open(targets_path, "w", newline="") as f:
        writer = csv.writer(f)
        for tgt, reg_list in sorted(tgt_regs.items()):
            reg_ids = [r[0] for r in reg_list]
            ks = [r[1] for r in reg_list]
            hills = [r[2] for r in reg_list]
            writer.writerow([tgt, len(reg_list)] + reg_ids + ks + hills)

    with open(regs_path, "w", newline="") as f:
        writer = csv.writer(f)
        for reg in sorted(master_regs):
            basal = rng.uniform(basal_low, basal_high, size=n_bins).tolist()
            writer.writerow([reg] + basal)

    return targets_path, regs_path


# ---------------------------------------------------------------------------
# Convenience: generate + write in one call
# ---------------------------------------------------------------------------

def make_scale_free_sergio_inputs(
    n_genes: int,
    n_bins: int,
    out_dir: str,
    n_edges_per_new_node: int = 2,
    frac_repression: float = 0.3,
    basal_low: float = 0.2,
    basal_high: float = 1.5,
    seed: int | None = None,
) -> tuple[nx.DiGraph, str, str]:
    """
    Generate a scale-free GRN and write SERGIO input files in one call.

    Returns
    -------
    (G, targets_path, regs_path)
    """
    G = generate_scale_free_grn(
        n_genes=n_genes,
        n_edges_per_new_node=n_edges_per_new_node,
        frac_repression=frac_repression,
        seed=seed,
    )
    targets_path, regs_path = grn_to_sergio_files(
        G,
        out_dir=out_dir,
        n_bins=n_bins,
        basal_low=basal_low,
        basal_high=basal_high,
        seed=seed,
    )
    return G, targets_path, regs_path


# ---------------------------------------------------------------------------
# GRN inspection helpers
# ---------------------------------------------------------------------------

def get_master_regulators(G: nx.DiGraph) -> list[int]:
    """Return nodes with no incoming edges (root transcription factors)."""
    return _get_master_regulators(G)


def degree_distribution(G: nx.DiGraph) -> dict[str, np.ndarray]:
    """
    Return in- and out-degree sequences for power-law verification.

    Example::

        dd = degree_distribution(G)
        # dd["in_degrees"], dd["out_degrees"]
    """
    return {
        "in_degrees": np.array([d for _, d in G.in_degree()]),
        "out_degrees": np.array([d for _, d in G.out_degree()]),
    }


def grn_summary(G: nx.DiGraph) -> dict:
    """Return a dict with basic GRN statistics."""
    dd = degree_distribution(G)
    master_regs = _get_master_regulators(G)
    n_act = sum(1 for _, _, d in G.edges(data=True) if d.get("sign", 1) > 0)
    n_rep = G.number_of_edges() - n_act
    n_levels = nx.dag_longest_path_length(G) + 1 if nx.is_directed_acyclic_graph(G) else None
    return {
        "n_genes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_master_regulators": len(master_regs),
        "n_activating": n_act,
        "n_repressing": n_rep,
        "max_in_degree": int(dd["in_degrees"].max()),
        "max_out_degree": int(dd["out_degrees"].max()),
        "mean_in_degree": float(dd["in_degrees"].mean()),
        "n_levels": n_levels,
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_master_regulators(G: nx.DiGraph) -> list[int]:
    return [n for n in G.nodes() if G.in_degree(n) == 0]


def _make_dag(G: nx.DiGraph) -> nx.DiGraph:
    """Remove back-edges (based on DFS finish order) to produce a DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes())
    visited = set()
    in_stack = set()

    def dfs(node):
        visited.add(node)
        in_stack.add(node)
        for nbr in list(G.successors(node)):
            if nbr not in visited:
                dag.add_edge(node, nbr)
                dfs(nbr)
            elif nbr in in_stack:
                pass  # back-edge: skip to break cycle
            else:
                dag.add_edge(node, nbr)
        in_stack.discard(node)

    for node in G.nodes():
        if node not in visited:
            dfs(node)

    return dag


def _ensure_connected(G: nx.DiGraph, rng: np.random.Generator) -> nx.DiGraph:
    """
    Attach isolated nodes (degree 0) to a random existing node as a target,
    so every gene participates in the network.
    """
    candidates = [n for n in G.nodes() if G.degree(n) > 0]
    if not candidates:
        return G
    for node in list(G.nodes()):
        if G.degree(node) == 0:
            parent = int(rng.choice(candidates))
            G.add_edge(parent, node)
    return G
