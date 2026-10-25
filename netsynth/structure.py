"""Structural fidelity metrics: scalar summaries + distribution distances."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import ks_2samp, wasserstein_distance

from netsynth.portrait import _csr

# name -> (label, error kind).  "rel": |syn-real|/|real| ; "abs": |syn-real|
SCALAR_METRICS = {
    "mean_degree": ("Mean degree", "rel"),
    "max_degree": ("Max degree", "rel"),
    "degree_second_moment": ("<k²> (degree heterogeneity)", "rel"),
    "avg_clustering": ("Avg. local clustering", "abs"),
    "transitivity": ("Global clustering (transitivity)", "abs"),
    "assortativity": ("Degree assortativity", "abs"),
    "aspl": ("Avg. shortest path (LCC)", "rel"),
    "diameter": ("Diameter (LCC)", "rel"),
    "lcc_fraction": ("Largest component fraction", "abs"),
}


def degree_array(G: nx.Graph) -> np.ndarray:
    return np.fromiter((d for _, d in G.degree()), dtype=np.int64, count=G.number_of_nodes())


def lcc_nodes(G: nx.Graph) -> np.ndarray:
    A = _csr(G)
    _, lab = connected_components(A, directed=False)
    return np.flatnonzero(lab == np.bincount(lab).argmax())


def path_stats(G: nx.Graph, max_sources: int = 300, seed: int = 0) -> tuple[float, float, bool]:
    """(average shortest path, diameter, exact?) on the largest component.

    BFS from at most ``max_sources`` random sources (all of them when the
    component is small) - the diameter is then a lower bound.
    """
    A = _csr(G)
    keep = lcc_nodes(G)
    if len(keep) < 2:
        return 0.0, 0.0, True
    A = A[keep][:, keep]
    rng = np.random.default_rng(seed)
    exact = len(keep) <= max_sources
    src = np.arange(len(keep)) if exact else rng.choice(len(keep), max_sources, replace=False)
    D = shortest_path(A, method="D", directed=False, unweighted=True, indices=src)
    D = D[np.isfinite(D)]
    nz = D[D > 0]
    return float(nz.mean()) if len(nz) else 0.0, float(D.max()), exact


def clustering_array(G: nx.Graph) -> np.ndarray:
    c = nx.clustering(G)
    return np.fromiter((c[u] for u in G.nodes()), dtype=float, count=G.number_of_nodes())


@dataclass
class StructuralSummary:
    scalars: dict
    degrees: np.ndarray
    clustering: np.ndarray


def summarize(G: nx.Graph, seed: int = 0) -> StructuralSummary:
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = degree_array(G)
    clu = clustering_array(G)
    try:
        assort = nx.degree_assortativity_coefficient(G) if m > 0 else 0.0
        assort = 0.0 if not np.isfinite(assort) else float(assort)
    except Exception:
        assort = 0.0
    aspl, diam, exact = path_stats(G, seed=seed)
    sc = {
        "n": n,
        "m": m,
        "density": 2 * m / (n * (n - 1)) if n > 1 else 0.0,
        "mean_degree": float(deg.mean()) if n else 0.0,
        "max_degree": float(deg.max()) if n else 0.0,
        "degree_second_moment": float((deg.astype(float) ** 2).mean()) if n else 0.0,
        "avg_clustering": float(clu.mean()) if n else 0.0,
        "transitivity": float(nx.transitivity(G)),
        "assortativity": assort,
        "aspl": aspl,
        "diameter": diam,
        "lcc_fraction": len(lcc_nodes(G)) / n if n else 0.0,
        "n_triangles": int(sum(nx.triangles(G).values()) // 3),
        "n_components": int(nx.number_connected_components(G)),
    }
    return StructuralSummary(sc, deg, clu)


def metric_errors(real: dict, syn: dict) -> dict:
    """Per-metric error of one synthetic graph against the real graph."""
    out = {}
    for k, (_, kind) in SCALAR_METRICS.items():
        d = abs(syn[k] - real[k])
        out[k] = d / max(abs(real[k]), 1e-12) if kind == "rel" else d
    return out


def degree_distances(real_deg: np.ndarray, syn_deg: np.ndarray) -> dict:
    """KS statistic, normalised 1-Wasserstein and Jensen-Shannon distance of degree distributions."""
    ks = float(ks_2samp(real_deg, syn_deg).statistic)
    w = float(wasserstein_distance(real_deg, syn_deg) / max(real_deg.mean(), 1e-12))
    kmax = int(max(real_deg.max(), syn_deg.max())) + 1
    p = np.bincount(real_deg, minlength=kmax) / len(real_deg)
    q = np.bincount(syn_deg, minlength=kmax) / len(syn_deg)
    mm = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    jsd = 0.5 * kl(p, mm) + 0.5 * kl(q, mm)
    return {"degree_ks": ks, "degree_wasserstein": w, "degree_jsd": jsd}


def clustering_ks(real_c: np.ndarray, syn_c: np.ndarray) -> float:
    return float(ks_2samp(real_c, syn_c).statistic)


def edge_overlap(real: nx.Graph, edges: np.ndarray, node_map) -> float | None:
    """Fraction of synthetic edges that are also real edges, via a node map.

    ``node_map[i]`` is the real node synthetic node ``i`` corresponds to
    (``None`` -> not defined for this generator).  Pairs mapped to the same real
    node are dropped.  Real nodes are assumed to be labelled 0..n-1.
    """
    if node_map is None or len(edges) == 0:
        return None
    u, v = node_map[edges[:, 0]], node_map[edges[:, 1]]
    ok = u != v
    if not ok.any():
        return 0.0
    n = real.number_of_nodes()
    A = _csr(real)
    return float(np.asarray(A[u[ok], v[ok]]).ravel().mean())
