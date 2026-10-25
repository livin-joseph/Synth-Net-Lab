"""SyNGLER wrappers (SyNG-R / SyNG-D) behind the common generator interface."""
from __future__ import annotations

import time

import networkx as nx

from netsynth.generators.base import GeneratorResult, ProgressFn, graph_from_edges
from syngler.api import SyNGLER

LABELS = {"syng_r": "SyNGLER (SyNG-R)", "syng_d_mlp": "SyNGLER (SyNG-D, MLP)", "syng_d_forest": "SyNGLER (SyNG-D, Forest)"}
_METHOD = {"syng_r": "res", "syng_d_mlp": "diff-mlp", "syng_d_forest": "diff-forest"}


def _make(key: str):
    def run(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None, r: int = 6,
            n_iter_init: int = 300, n_iter_refine: int = 500, mlp_steps: int = 2000,
            forest_n_t: int = 30, forest_dup: int = 30, shared_fit: dict | None = None, **_) -> GeneratorResult:
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
        if progress: progress("sampling", 0.75)
        t0 = time.perf_counter()
        samples = model.sample(k, method=_METHOD[key], seed=seed, mlp_steps=mlp_steps,
                               forest_kw=dict(n_t=forest_n_t, duplicate_K=forest_dup))
        total = time.perf_counter() - t0
        extra_train = getattr(model, "diff_train_seconds", 0.0) if key != "syng_r" else 0.0
        graphs = [graph_from_edges(s.n, s.edges) for s in samples]
        fit_s = model.fit_seconds + extra_train
        return GeneratorResult(
            key, LABELS[key], graphs, [s.node_map for s in samples],
            fit_seconds=fit_s, gen_seconds=(total - extra_train) / k,
            info={"r": model.fit_.r, "rho": model.fit_.rho,
                  "loss_init": model.fit_.loss_init, "loss_refine": model.fit_.loss_refine,
                  "lsm_seconds": model.fit_seconds, "diffusion_train_seconds": extra_train},
        )
    return run


generate_syng_r = _make("syng_r")
generate_syng_d_mlp = _make("syng_d_mlp")
generate_syng_d_forest = _make("syng_d_forest")
