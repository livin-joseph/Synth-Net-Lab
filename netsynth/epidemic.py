"""Functional fidelity: SIR epidemics (EoN) on real vs synthetic networks."""
from __future__ import annotations

import random
from dataclasses import dataclass

import EoN
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

from netsynth.structure import degree_array


def threshold_beta(G: nx.Graph, gamma: float) -> float:
    """Mean-field epidemic threshold for SIR: beta_c = gamma <k> / (<k^2> - <k>)."""
    k = degree_array(G).astype(float)
    denom = (k ** 2).mean() - k.mean()
    return float(gamma * k.mean() / denom) if denom > 0 else float("inf")


@dataclass
class SIRConfig:
    beta: float                  # per-edge transmission rate
    gamma: float = 1.0           # recovery rate
    init_frac: float = 0.01      # fraction of nodes infected at t = 0
    n_runs: int = 20             # runs per graph
    big_outbreak: float = 0.05   # final size above which a run counts as an outbreak


def simulate(G: nx.Graph, cfg: SIRConfig, seed: int = 0) -> list:
    """Run ``cfg.n_runs`` SIR simulations; returns list of (t, I, R) step-function arrays."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    k0 = max(1, int(round(cfg.init_frac * n)))
    nodes = list(G.nodes())
    runs = []
    for _ in range(cfg.n_runs):
        init = [nodes[i] for i in rng.choice(n, size=k0, replace=False)]
        t, S, I, R = EoN.fast_SIR(G, cfg.beta, cfg.gamma, initial_infecteds=init, return_full_data=False)
        runs.append((np.asarray(t), np.asarray(I), np.asarray(R)))
    return runs


def _on_grid(run, grid):
    t, I, R = run
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    return I[idx], R[idx]


def curves_on_grid(runs: list, grid: np.ndarray, n: int) -> tuple:
    """Prevalence I(t)/n and cumulative R(t)/n per run, shape (runs, len(grid))."""
    I = np.empty((len(runs), len(grid)))
    R = np.empty_like(I)
    for i, run in enumerate(runs):
        a, b = _on_grid(run, grid)
        I[i], R[i] = a / n, b / n
    return I, R


def common_grid(runs: list, num: int = 200) -> np.ndarray:
    """Time grid up to the point where 99% of the real runs have ended."""
    ends = np.array([r[0][-1] for r in runs])
    return np.linspace(0, max(float(np.quantile(ends, 0.99)), 1e-6) * 1.05, num)


def run_stats(runs: list, grid: np.ndarray, n: int, cfg: SIRConfig) -> dict:
    I, R = curves_on_grid(runs, grid, n)
    final = np.array([r[2][-1] / n for r in runs])
    peak = I.max(axis=1)
    tpeak = grid[I.argmax(axis=1)]
    return {
        "I": I, "R": R, "final": final, "peak": peak, "tpeak": tpeak,
        "outbreak_prob": float((final > cfg.big_outbreak).mean()),
    }


def _nrmse(a, b, scale):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(scale, 1e-9))


def compare_to_real(real: dict, syn: dict) -> dict:
    """Functional errors of pooled synthetic runs against pooled real runs."""
    mI_r, mI_s = real["I"].mean(0), syn["I"].mean(0)
    scale = mI_r.max()
    return {
        "final_size_err": float(abs(syn["final"].mean() - real["final"].mean())),
        "final_size_wasserstein": float(wasserstein_distance(real["final"], syn["final"])),
        "peak_err_rel": float(abs(syn["peak"].mean() - real["peak"].mean()) / max(real["peak"].mean(), 1e-9)),
        "tpeak_err_rel": float(abs(syn["tpeak"].mean() - real["tpeak"].mean()) / max(real["tpeak"].mean(), 1e-9)),
        "curve_nrmse": _nrmse(mI_r, mI_s, scale),
        "outbreak_prob_err": float(abs(syn["outbreak_prob"] - real["outbreak_prob"])),
    }


def noise_floor(real: dict, rng: np.random.Generator) -> dict:
    """Real-vs-real error from splitting the real runs in two halves (sampling-noise reference)."""
    n = real["I"].shape[0]
    if n < 4:
        return {}
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    pick = lambda d, ix: {k: (v[ix] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    A, B = pick(real, a), pick(real, b)
    A["outbreak_prob"] = float((A["final"] > 0.05).mean()); B["outbreak_prob"] = float((B["final"] > 0.05).mean())
    return compare_to_real(A, B)


