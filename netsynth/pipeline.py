"""End-to-end experiment: generate -> structural fidelity -> epidemic fidelity -> timing."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import networkx as nx
import numpy as np

from netsynth import epidemic as epi
from netsynth import structure as st
from netsynth.generators.base import GeneratorResult
from netsynth.generators.classical import generate_ba, generate_cm, generate_er
from netsynth.portrait import portrait, portrait_divergence
from netsynth.structure import StructuralSummary

# key -> (label, function, needs torch/LSM)
def _registry():
    from netsynth.generators import netgan, syngler_gen
    return {
        "er": ("Erdős–Rényi", generate_er),
        "ba": ("Barabási–Albert", generate_ba),
        "cm": ("Configuration model", generate_cm),
        "netgan": ("NetGAN", netgan.generate_netgan),
        "syng_r": ("SyNGLER (SyNG-R)", syngler_gen.generate_syng_r),
        "syng_d_mlp": ("SyNGLER (SyNG-D, MLP)", syngler_gen.generate_syng_d_mlp),
        "syng_d_forest": ("SyNGLER (SyNG-D, Forest)", syngler_gen.generate_syng_d_forest),
    }


def generator_labels() -> dict:
    return {k: v[0] for k, v in _registry().items()}


@dataclass
class ExperimentConfig:
    generators: list = field(default_factory=lambda: ["er", "ba", "cm", "syng_r"])
    k: int = 5                                  # synthetic graphs per generator
    seed: int = 0
    gen_params: dict = field(default_factory=dict)
    beta_mult: float = 2.0                      # SIR beta = beta_mult * beta_c(real)
    gamma: float = 1.0
    init_frac: float = 0.01
    runs_per_graph: int = 40
    big_outbreak: float = 0.05
    compute_portrait: bool = True


@dataclass
class GenReport:
    key: str
    label: str
    ok: bool = True
    error: Optional[str] = None
    result: Optional[GeneratorResult] = None
    summaries: list = field(default_factory=list)
    errors: list = field(default_factory=list)          # per graph: scalar-metric errors
    degree_d: list = field(default_factory=list)        # per graph: degree distances
    cluster_ks: list = field(default_factory=list)
    pd: list = field(default_factory=list)
    overlap: list = field(default_factory=list)         # per graph: edge overlap or None
    sir_pooled: Optional[dict] = None
    sir_cmp: Optional[dict] = None                      # pooled synthetic vs real
    sir_cmp_each: list = field(default_factory=list)    # per graph vs real
    beta_threshold: list = field(default_factory=list)  # per graph epidemic threshold
    trace: Optional[str] = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    real: nx.Graph
    real_summary: StructuralSummary
    real_portrait: Optional[np.ndarray]
    sir_cfg: epi.SIRConfig
    beta_c: float
    grid: np.ndarray
    real_sir: dict
    floor: dict
    reports: dict
    seconds: float = 0.0


def run_experiment(G: nx.Graph, cfg: ExperimentConfig,
                   progress: Optional[Callable[[str, float], None]] = None) -> ExperimentResult:
    t_start = time.perf_counter()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    reg = _registry()
    stages = ["real"] + list(cfg.generators)
    wf = 1.0 / len(stages)

    def prog(i, msg, f=1.0):
        if progress:
            progress(msg, min(1.0, (i + f) * wf))

    # ---------------------------------------------------------------- real network
    prog(0, "Analysing the real network…", 0.0)
    real_sum = st.summarize(G, seed=cfg.seed)
    BG = portrait(G) if cfg.compute_portrait else None
    gamma = cfg.gamma
    beta_c = epi.threshold_beta(G, gamma)
    beta = cfg.beta_mult * beta_c if np.isfinite(beta_c) else gamma
    sir_cfg = epi.SIRConfig(beta=beta, gamma=gamma, init_frac=cfg.init_frac,
                            n_runs=cfg.runs_per_graph * cfg.k, big_outbreak=cfg.big_outbreak)
    real_runs = epi.simulate(G, sir_cfg, seed=cfg.seed)
    grid = epi.common_grid(real_runs)
    real_stats = epi.run_stats(real_runs, grid, n, sir_cfg)
    floor = epi.noise_floor(real_stats, np.random.default_rng(cfg.seed))
    prog(0, "Real network done", 1.0)

    per_graph_cfg = epi.SIRConfig(beta=beta, gamma=gamma, init_frac=cfg.init_frac,
                                  n_runs=cfg.runs_per_graph, big_outbreak=cfg.big_outbreak)
    reports = {}
    shared_fit: dict = {}
    for gi, key in enumerate(cfg.generators, start=1):
        label, fn = reg[key]
        rep = GenReport(key, label)
        reports[key] = rep
        try:
            gen_prog = lambda msg, f, _gi=gi, _l=label: prog(_gi, f"{_l}: {msg}", 0.6 * f)
            params = dict(cfg.gen_params)
            res = fn(G, cfg.k, seed=cfg.seed, progress=gen_prog, shared_fit=shared_fit, **params)
            rep.result = res
            all_runs = []
            for b, H in enumerate(res.graphs):
                prog(gi, f"{label}: evaluating graph {b + 1}/{len(res.graphs)}", 0.6 + 0.4 * b / len(res.graphs))
                s = st.summarize(H, seed=cfg.seed)
                rep.summaries.append(s)
                rep.errors.append(st.metric_errors(real_sum.scalars, s.scalars))
                rep.degree_d.append(st.degree_distances(real_sum.degrees, s.degrees))
                rep.cluster_ks.append(st.clustering_ks(real_sum.clustering, s.clustering))
                if BG is not None:
                    rep.pd.append(portrait_divergence(BG, portrait(H)))
                edges = np.array(list(H.edges()), dtype=np.int64).reshape(-1, 2)
                rep.overlap.append(st.edge_overlap(G, edges, res.node_maps[b]))
                rep.beta_threshold.append(epi.threshold_beta(H, gamma))
                runs = epi.simulate(H, per_graph_cfg, seed=cfg.seed + 1000 * (b + 1))
                all_runs.extend(runs)
                rep.sir_cmp_each.append(epi.compare_to_real(real_stats, epi.run_stats(runs, grid, n, per_graph_cfg)))
            rep.sir_pooled = epi.run_stats(all_runs, grid, n, sir_cfg)
            rep.sir_cmp = epi.compare_to_real(real_stats, rep.sir_pooled)
        except Exception as e:                                   # keep going: other generators still count
            rep.ok = False
            rep.error = f"{type(e).__name__}: {e}"
            rep.trace = traceback.format_exc()
        prog(gi, f"{label} done", 1.0)

    return ExperimentResult(cfg, G, real_sum, BG, sir_cfg, beta_c, grid, real_stats, floor, reports,
                            seconds=time.perf_counter() - t_start)
