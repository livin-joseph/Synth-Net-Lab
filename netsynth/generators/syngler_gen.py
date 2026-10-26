"""SyNGLER wrappers (SyNG-R / SyNG-D) behind the common generator interface."""
from __future__ import annotations

import time

import networkx as nx
import numpy as np

from netsynth.generators.base import GeneratorResult, ProgressFn, graph_from_edges
from syngler.api import SyNGLER
from netsynth.structure import edge_overlap

NOISE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)   # candidate latent-noise levels for auto-tuning


def tune_latent_noise(model: SyNGLER, G: nx.Graph, target: float, seed: int = 0) -> tuple:
    """Smallest latent noise whose SyNG-R samples copy at most ``margin * target`` of the real edges *beyond chance*.

    Leakage is measured exactly as in the experiment (edge overlap through the bootstrap node map), on a few probe
    graphs per candidate level, scanning from no noise upwards.  If even the largest level misses the target, that
    level is used (more noise would degrade the structure too far).  Returns (noise, excess_at_that_noise).
    """
    d = float(nx.density(G))
    n = G.number_of_nodes()
    margin = 0.65 if n <= 100 else 0.8 if n <= 300 else 0.95    # small graphs: noisy estimate -> aim lower
    n_probe = 16 if n <= 100 else 8 if n <= 300 else 4 if n <= 1000 else 2
    last = (NOISE_GRID[-1], float("nan"))
    for noise in NOISE_GRID:
        ex = []
        for smp in model.sample(n_probe, method="res", seed=seed + 7919, latent_noise=noise):
            ov = edge_overlap(G, smp.edges, smp.node_map)
            ex.append(0.0 if ov is None else max(0.0, (ov - d) / max(1.0 - d, 1e-9)))
        excess = float(np.mean(ex))
        last = (noise, excess)
        if excess <= margin * target:
            break
    return last

LABELS = {"syng_r": "SyNGLER (SyNG-R)", "syng_d_mlp": "SyNGLER (SyNG-D, MLP)", "syng_d_forest": "SyNGLER (SyNG-D, Forest)"}
_METHOD = {"syng_r": "res", "syng_d_mlp": "diff-mlp", "syng_d_forest": "diff-forest"}


def _make(key: str):
    def run(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None, r: int = 6,
            n_iter_init: int = 300, n_iter_refine: int = 500, mlp_steps: int = 2000,
            forest_n_t: int = 30, forest_dup: int = 30, latent_noise: float | None = None, leak_target: float = 0.25, shared_fit: dict | None = None, **_) -> GeneratorResult:
        """Fit the LSM once (cached across SyNG variants via ``shared_fit``), then sample ``k`` graphs."""
        shared_fit = shared_fit if shared_fit is not None else {}
        if "model" not in shared_fit:
            model = SyNGLER(r=r, n_iter_init=n_iter_init, n_iter_refine=n_iter_refine, seed=seed)
            def cb(phase, it, total, loss):
                if progress:
                    base = 0.0 if phase == "init" else 0.5
                    progress(f"LSM fit ({phase}) {it}/{total}", 0.7 * (base + 0.5 * it / max(total, 1)))
            model.fit(G, progress=cb)
            shared_fit["model"] = model
        model = shared_fit["model"]
        noise, probe = 0.0, None
        if key == "syng_r":
            if latent_noise is None:                    # auto: calibrate to the leakage target
                if progress: progress("tuning privacy noise", 0.72)
                if "noise" not in shared_fit:
                    shared_fit["noise"] = tune_latent_noise(model, G, leak_target, seed)
                noise, probe = shared_fit["noise"]
            else:
                noise = float(latent_noise)
        if progress: progress("sampling", 0.75)
        t0 = time.perf_counter()
        samples = model.sample(k, method=_METHOD[key], seed=seed, mlp_steps=mlp_steps,
                               forest_kw=dict(n_t=forest_n_t, duplicate_K=forest_dup),
                               latent_noise=noise)
        total = time.perf_counter() - t0
        extra_train = getattr(model, "diff_train_seconds", 0.0) if key != "syng_r" else 0.0
        graphs = [graph_from_edges(s.n, s.edges) for s in samples]
        fit_s = model.fit_seconds + extra_train
        return GeneratorResult(
            key, LABELS[key], graphs, [s.node_map for s in samples],
            fit_seconds=fit_s, gen_seconds=(total - extra_train) / k,
            info={"r": model.fit_.r, "latent_noise": noise, "leak_probe": probe, "leak_target": leak_target if latent_noise is None else None, "rho": model.fit_.rho,
                  "loss_init": model.fit_.loss_init, "loss_refine": model.fit_.loss_refine,
                  "lsm_seconds": model.fit_seconds, "diffusion_train_seconds": extra_train},
        )
    return run


generate_syng_r = _make("syng_r")
generate_syng_d_mlp = _make("syng_d_mlp")
generate_syng_d_forest = _make("syng_d_forest")
