"""Network portraits and Portrait Divergence (Bagrow & Bollt, 2019).

The *portrait* B of a graph has B[l, k] = number of nodes that have exactly k
nodes at shortest-path distance l.  The portrait divergence is the
Jensen-Shannon divergence between the two graphs' (size-weighted) portrait
distributions; it is a single number in [0, 1] (log base 2) that is sensitive
to degree distribution, path lengths, clustering and component structure.

* :func:`portrait` is a fast scipy/C-backed re-implementation that returns the
  *same matrix* as ``portrait_py`` of the reference code (checked in
  ``tests/test_portrait.py``) but is 20-100x faster than the pure-python BFS.
* The divergence itself is the original function from
  https://github.com/bagrow/network-portrait-divergence (MIT, vendored in
  ``third_party/portrait_divergence_bagrow.py``).
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from third_party.portrait_divergence_bagrow import portrait_divergence as _pd_matrices


def _csr(G: nx.Graph) -> sp.csr_matrix:
    n = G.number_of_nodes()
    nodes = {u: i for i, u in enumerate(G.nodes())}
    if G.number_of_edges() == 0:
        return sp.csr_matrix((n, n))
    e = np.array([(nodes[u], nodes[v]) for u, v in G.edges() if u != v], dtype=np.int64)
    data = np.ones(len(e) * 2, dtype=np.float32)
    A = sp.coo_matrix((data, (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def portrait(G: nx.Graph, chunk: int = 256) -> np.ndarray:
    """Portrait matrix B (rows: distance l = 0..max_path, cols: k = 0..N-1)."""
    N = G.number_of_nodes()
    A = _csr(G)
    pairs = Counter()          # (l, k) -> number of starting nodes
    ecc = np.zeros(N, dtype=np.int64)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(N, s + chunk))
        D = shortest_path(A, method="D", directed=False, unweighted=True, indices=idx)
        D[~np.isfinite(D)] = -1
        D = D.astype(np.int64)
        for row_i, row in zip(idx, D):
            counts = np.bincount(row[row >= 0])
            ecc[row_i] = len(counts) - 1
            for l, c in enumerate(counts):
                pairs[(l, int(c))] += 1
    max_path = max(1, int(ecc.max()))
    # reference quirk: sources whose eccentricity is < l contribute to B[l, 0]
    for l in range(1, max_path + 1):
        n_short = int((ecc < l).sum())
        if n_short:
            pairs[(l, 0)] += n_short
    B = np.zeros((max_path + 1, max(N, 2)))
    for (l, c), cnt in pairs.items():
        if l <= max_path:
            B[l, c] += cnt
    return B


def portrait_divergence(G, H) -> float:
    """Portrait divergence between two graphs *or* two precomputed portraits."""
    BG = portrait(G) if isinstance(G, nx.Graph) else G
    BH = portrait(H) if isinstance(H, nx.Graph) else H
    return float(_pd_matrices(BG, BH))
