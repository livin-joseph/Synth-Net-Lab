"""Graph sub-sampling used for size caps and the scaling benchmark."""
from __future__ import annotations

import networkx as nx
import numpy as np


def snowball_nodes(G: nx.Graph, n_target: int, seed: int = 0) -> list:
    """The first ``n_target`` nodes (original labels) reached by a randomised BFS.

    Starts from a random node; if the component is exhausted before ``n_target``
    nodes are collected it continues from another random unvisited node.
    """
    n = G.number_of_nodes()
    if n_target >= n:
        return list(G.nodes())
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    seen, order = set(), []
    unvisited = list(rng.permutation(len(nodes)))
    while len(order) < n_target and unvisited:
        start = nodes[unvisited.pop()]
        if start in seen:
            continue
        seen.add(start); order.append(start)
        frontier = [start]
        while frontier and len(order) < n_target:
            nxt = []
            for u in frontier:
                nb = [v for v in G.neighbors(u) if v not in seen]
                rng.shuffle(nb)
                for v in nb:
                    if len(order) >= n_target:
                        break
                    seen.add(v); order.append(v); nxt.append(v)
            frontier = nxt
    return order[:n_target]


def snowball_sample(G: nx.Graph, n_target: int, seed: int = 0) -> nx.Graph:
    """Induced subgraph on the snowball nodes, relabelled to 0..n-1."""
    if n_target >= G.number_of_nodes():
        return G.copy()
    return nx.convert_node_labels_to_integers(G.subgraph(snowball_nodes(G, n_target, seed)).copy())
