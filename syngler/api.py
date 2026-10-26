"""High-level, object-oriented API for SyNGLER.

The original repository exposes SyNGLER as loose functions plus batch runner
scripts.  This module wraps the pieces into one object::

    model = SyNGLER(r=6).fit(G)                 # stage 1: latent space model
    graphs = model.sample(10, method="res")     # stage 2: resample + reconstruct

Additions w.r.t. the reference implementation
---------------------------------------------
* ``fit`` takes a networkx graph / scipy sparse / dense adjacency and uses the
  fast fitting routine in :mod:`syngler.fit`.
* Sampling is *chunked*: the n x n probability matrix is never materialised,
  only ``block x n`` slices of it, and the output is a sparse edge list.
  Memory is O(block * n + m) instead of O(n^2).
* Every sample records ``node_map`` - for SyNG-R, the real node each synthetic
  node was bootstrapped from.  This enables an edge-overlap ("leakage") check.
* Three stage-2 samplers: ``"res"`` (bootstrap, SyNG-R), ``"diff-mlp"``
  (DDPM over latents, CPU friendly) and ``"diff-forest"`` (ForestDiffusion).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from syngler.fit import LSMFit, _calibrate_rho, fit_lsm


@dataclass
class SyntheticSample:
    """One synthetic graph as an undirected edge list over nodes 0..n-1."""

    n: int
    edges: np.ndarray                       # (m, 2) int64, u < v
    node_map: Optional[np.ndarray] = None   # (n,) synthetic node -> real node (if defined)
    seconds: float = 0.0

    def to_networkx(self):
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(map(tuple, self.edges.tolist()))
        return G


def sample_edges_from_latents(Z, alpha, rho, rng, block: int = 1024) -> np.ndarray:
    """Bernoulli-sample the upper triangle of ``sigmoid(Z Z^T + a_i + a_j + rho)``.

    Equivalent to :func:`syngler.utils.source.reconstruct_adjacency` but processes
    ``block`` rows at a time and returns an ``(m, 2)`` edge array.
    """
    Z = np.asarray(Z, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    n = Z.shape[0]
    rows, cols = [], []
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        logits = Z[i0:i1] @ Z.T
        logits += alpha[i0:i1, None]
        logits += alpha[None, :]
        logits += float(rho)
        P = expit(logits)
        U = rng.random(P.shape)
        hit = U < P
        # keep strictly-upper-triangular entries only (j > i)
        ii = np.arange(i0, i1)[:, None]
        jj = np.arange(n)[None, :]
        hit &= jj > ii
        r_, c_ = np.nonzero(hit)
        rows.append(r_ + i0)
        cols.append(c_)
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1).astype(np.int64)


class SyNGLER:
    """Two-stage synthetic network generator (fit LSM -> resample latents -> reconstruct)."""

    def __init__(self, r: int = 6, eta0: float = 0.5, n_iter_init: int = 600,
                 n_iter_refine: int = 300, seed: int = 0):
        self.r, self.eta0 = r, eta0
        self.n_iter_init, self.n_iter_refine = n_iter_init, n_iter_refine
        self.seed = seed
        self.fit_: Optional[LSMFit] = None
        self.fit_seconds: float = 0.0
        self._diff_cache: dict = {}

    # ---------------------------------------------------------------- stage 1
    def fit(self, graph, progress=None) -> "SyNGLER":
        A = _to_dense_adjacency(graph)
        self.fit_ = fit_lsm(A, r=self.r, eta0=self.eta0, n_iter_init=self.n_iter_init,
                            n_iter_refine=self.n_iter_refine, seed=self.seed, progress=progress)
        self.fit_seconds = self.fit_.seconds
        self._diff_cache = {}
        return self

    # ---------------------------------------------------------------- stage 2
    def sample(self, k: int = 1, method: str = "res", seed: Optional[int] = None,
               mlp_steps: int = 2000, forest_kw: Optional[dict] = None, latent_noise: float = 0.0) -> list:
        """Draw ``k`` synthetic graphs.

        latent_noise (SyNG-R only): standard deviation, in units of the spread of the fitted latents, of Gaussian
        jitter added to the resampled (Z, alpha), rescaled by 1/sqrt(1+sigma^2) so the spread (and with it the degree heterogeneity) is preserved.  The fitted model can be nearly deterministic on small graphs
        (edge probabilities close to 0/1 for real edges), which makes SyNG-R copy real edges.  Jitter smooths
        this; the intercept rho is re-calibrated per graph so the expected density stays equal to the real one.
        0 reproduces plain SyNG-R.

        method = "res"          bootstrap-resample (Z, alpha)  (SyNG-R, no extra training)
                 "diff-mlp"     DDPM over (Z, alpha), trained once on CPU/GPU
                 "diff-forest"  ForestDiffusion over (Z, alpha), trained once
        """
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        f = self.fit_
        seed = self.seed if seed is None else seed
        out = []
        if method == "res":
            for b in range(k):
                t0 = time.perf_counter()
                rng = np.random.default_rng([seed, b])
                idx = rng.integers(0, f.n, size=f.n)                 # bootstrap nodes
                Zb, ab, rho = f.Z[idx], f.alpha[idx], f.rho
                if latent_noise > 0:
                    sc = 1.0 / np.sqrt(1.0 + latent_noise ** 2)      # variance-preserving: keeps the degree spread
                    Zb = sc * (Zb + latent_noise * f.Z.std(axis=0, keepdims=True) * rng.standard_normal(Zb.shape))
                    ab = sc * (ab + latent_noise * f.alpha.std() * rng.standard_normal(ab.shape))
                    rho = _calibrate_rho(Zb @ Zb.T + ab[:, None] + ab[None, :], f.density)
                edges = sample_edges_from_latents(Zb, ab, rho, rng)
                out.append(SyntheticSample(f.n, edges, node_map=idx, seconds=time.perf_counter() - t0))
            return out

        latents = self._diffusion_latents(method, k, seed, mlp_steps, forest_kw or {})
        for b, (Zb, ab) in enumerate(latents):
            t0 = time.perf_counter()
            rng = np.random.default_rng([seed, 10_000 + b])
            edges = sample_edges_from_latents(Zb, ab, f.rho, rng)
            out.append(SyntheticSample(f.n, edges, node_map=None, seconds=time.perf_counter() - t0))
        return out

    def _diffusion_latents(self, method, k, seed, mlp_steps, forest_kw):
        f = self.fit_
        if method == "diff-mlp":
            from syngler.diff.mlp import _sample_mlp_diffusion, _train_mlp_diffusion

            key = ("mlp", seed, mlp_steps)
            if key not in self._diff_cache:
                X = np.hstack([f.Z, f.alpha.reshape(-1, 1)]).astype(np.float32)
                t0 = time.perf_counter()
                st = _train_mlp_diffusion(X, n_steps=mlp_steps, device="cpu", seed=seed)
                self._diff_cache[key] = (st, time.perf_counter() - t0)
            st, self.diff_train_seconds = self._diff_cache[key]
            res = []
            for _ in range(k):
                x = _sample_mlp_diffusion(st, n=f.n)
                res.append((x[:, : f.r], x[:, f.r:f.r + 1].ravel()))
            return res
        if method == "diff-forest":
            from syngler.diff.forest import build_forest

            kw = dict(n_t=30, duplicate_K=30)
            kw.update(forest_kw)
            key = ("forest", seed, tuple(sorted(kw.items())))
            if key not in self._diff_cache:
                t0 = time.perf_counter()
                model, _ = build_forest(f.Z, f.alpha, seed=seed, **kw)
                self._diff_cache[key] = (model, time.perf_counter() - t0)
            model, self.diff_train_seconds = self._diff_cache[key]
            res = []
            for _ in range(k):
                x = model.generate(batch_size=f.n)[:, :-1]
                res.append((x[:, : f.r], x[:, f.r:f.r + 1].ravel()))
            return res
        raise ValueError(f"unknown method {method!r}")


def _to_dense_adjacency(graph) -> np.ndarray:
    """networkx graph | scipy sparse | ndarray -> symmetric 0/1 float adjacency."""
    if isinstance(graph, np.ndarray):
        A = graph
    elif sp.issparse(graph):
        A = graph.toarray()
    else:
        import networkx as nx

        A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()), weight=None)
    A = (np.asarray(A) != 0).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A
