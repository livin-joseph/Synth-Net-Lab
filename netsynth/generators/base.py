"""Common types for all generators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import networkx as nx
import numpy as np

ProgressFn = Optional[Callable[[str, float], None]]   # (message, fraction 0..1)


@dataclass
class GeneratorResult:
    """K synthetic graphs from one generator plus timing / diagnostics."""

    key: str
    label: str
    graphs: list                      # list[nx.Graph] on nodes 0..n-1
    node_maps: list                   # per graph: np.ndarray (synthetic->real node) or None
    fit_seconds: float                # "training" time (0 for pure samplers)
    gen_seconds: float                # mean time to draw one graph
    info: dict = field(default_factory=dict)


def graph_from_edges(n: int, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if len(edges):
        G.add_edges_from(map(tuple, np.asarray(edges).tolist()))
    return G
