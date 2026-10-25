"""Scalability benchmark: training + generation time as the network grows."""
from __future__ import annotations

import time
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from netsynth.pipeline import _registry
from netsynth.sampling import snowball_sample


def default_sizes(n: int) -> list:
    base = [100, 200, 400, 800, 1600, 3200]
    sizes = [s for s in base if s < n] + [n]
    return sizes[-5:] if len(sizes) > 5 else sizes


def scaling_benchmark(G: nx.Graph, sizes: list, generators: list, params: Optional[dict] = None,
                      seed: int = 0, progress: Optional[Callable[[str, float], None]] = None) -> pd.DataFrame:
    """Run every generator on snowball sub-samples of ``G`` and time fit / generate.

    Each generator draws a single graph (k=1) with a fixed training budget, so
    the numbers show how the *cost per iteration / per sample* scales with n.
    """
    reg = _registry()
    params = dict(params or {})
    rows, total, done = [], len(sizes) * len(generators), 0
    for n_s in sizes:
        H = snowball_sample(G, n_s, seed=seed)
        for key in generators:
            label, fn = reg[key]
            row = {"size": H.number_of_nodes(), "edges": H.number_of_edges(), "generator": label, "key": key}
            try:
                t0 = time.perf_counter()
                res = fn(H, 1, seed=seed, **params)
                row.update(fit_s=res.fit_seconds, gen_s=res.gen_seconds,
                           total_s=time.perf_counter() - t0, error=None)
            except Exception as e:
                row.update(fit_s=np.nan, gen_s=np.nan, total_s=np.nan, error=f"{type(e).__name__}: {e}")
            rows.append(row)
            done += 1
            if progress:
                progress(f"{label} @ n={H.number_of_nodes()}", done / total)
    return pd.DataFrame(rows)


def scaling_exponents(df: pd.DataFrame, col: str = "total_s") -> pd.DataFrame:
    """Empirical exponent b in time ≈ a · n^b from a log-log least-squares fit."""
    out = []
    for key, d in df.dropna(subset=[col]).groupby("key"):
        d = d[d[col] > 0]
        if len(d) >= 3:
            b, a = np.polyfit(np.log(d["size"]), np.log(d[col]), 1)
            out.append({"key": key, "generator": d["generator"].iloc[0], "exponent": float(b)})
    return pd.DataFrame(out)
