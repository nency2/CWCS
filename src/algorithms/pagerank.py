"""
Matrix PageRank algorithm for gene prioritization.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional


def calculate_pagerank(
    G: nx.DiGraph,
    target_id: str,
    seed_genes: Optional[List[str]] = None,
    alpha: float = 0.85
) -> Dict[str, float]:
    """
    Calculate Matrix PageRank scores on reversed graph.

    Args:
        G: NetworkX DiGraph
        target_id: Disease node ID to use as personalization target
        seed_genes: Optional list of seed genes for personalization
        alpha: Damping factor (default 0.85)

    Returns:
        Dictionary mapping node IDs to normalized PageRank scores
    """
    if G is None or target_id not in G:
        return {}

    # Reverse graph for PageRank
    G_rev = G.reverse()

    # Add self-loops
    for n in G_rev.nodes():
        if not G_rev.has_edge(n, n):
            G_rev.add_edge(n, n, weight=0.01)

    nodes = list(G_rev.nodes())
    n_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build transition matrix
    adj = nx.to_numpy_array(G_rev, nodelist=nodes, weight='weight')
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    W = adj / row_sums[:, np.newaxis]

    # Build personalization vector
    p = np.zeros(n_nodes)

    if seed_genes:
        valid = [s for s in seed_genes if s in node_to_idx]
        if valid:
            if target_id in node_to_idx:
                p[node_to_idx[target_id]] = 0.5
            sw = 0.5 / len(valid)
            for s in valid:
                p[node_to_idx[s]] = sw
        elif target_id in node_to_idx:
            p[node_to_idx[target_id]] = 1.0
    elif target_id in node_to_idx:
        p[node_to_idx[target_id]] = 1.0

    if p.sum() > 0:
        p = p / p.sum()

    # Solve PageRank equation
    try:
        r = np.linalg.solve(
            np.eye(n_nodes) - alpha * W.T,
            (1 - alpha) * p
        )

        scores = {nodes[i]: float(r[i]) for i in range(n_nodes)}
        scores.pop(target_id, None)

        # Normalize scores
        max_score = max(scores.values()) if scores else 1
        return {k: v / max_score for k, v in scores.items() if max_score > 0}

    except np.linalg.LinAlgError:
        return {}
