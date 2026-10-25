"""Classical random-graph baselines: Erdős–Rényi, Barabási–Albert, configuration model."""
from __future__ import annotations

import time

import networkx as nx
import numpy as np

from netsynth.generators.base import GeneratorResult, ProgressFn


def _ident(n):
    return np.arange(n)


def generate_er(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None, **_) -> GeneratorResult:
    """G(n, m): same number of nodes and edges as the real graph."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    t0 = time.perf_counter()
    graphs = []
    for b in range(k):
        graphs.append(nx.gnm_random_graph(n, m, seed=seed + b))
        if progress: progress("ER", (b + 1) / k)
    dt = (time.perf_counter() - t0) / k
    return GeneratorResult("er", "Erdős–Rényi", graphs, [_ident(n)] * k, 0.0, dt)


def generate_ba(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None, **_) -> GeneratorResult:
    """Preferential attachment with m_BA = round(m / n) edges per new node."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    m_ba = int(min(max(1, round(m / max(n, 1))), n - 1))
    t0 = time.perf_counter()
    graphs = []
    for b in range(k):
        graphs.append(nx.barabasi_albert_graph(n, m_ba, seed=seed + b))
        if progress: progress("BA", (b + 1) / k)
    dt = (time.perf_counter() - t0) / k
    return GeneratorResult("ba", "Barabási–Albert", graphs, [_ident(n)] * k, 0.0, dt, info={"m_ba": m_ba})


def generate_cm(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None, **_) -> GeneratorResult:
    """Configuration model with the real degree sequence, collapsed to a simple graph."""
    n = G.number_of_nodes()
    seq = [d for _, d in G.degree()]
    if sum(seq) % 2:
        seq[int(np.argmax(seq))] -= 1
    t0 = time.perf_counter()
    graphs, lost = [], []
    for b in range(k):
        M = nx.configuration_model(seq, seed=seed + b)
        H = nx.Graph(M)                                 # collapse parallel edges
        H.remove_edges_from(nx.selfloop_edges(H))
        graphs.append(H)
        lost.append(1 - H.number_of_edges() / max(G.number_of_edges(), 1))
        if progress: progress("CM", (b + 1) / k)
    dt = (time.perf_counter() - t0) / k
    return GeneratorResult("cm", "Configuration model", graphs, [_ident(n)] * k, 0.0, dt,
                           info={"edge_loss_from_simplification": float(np.mean(lost))})
