import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ba(n=150, m=3, seed=1):
    return nx.barabasi_albert_graph(n, m, seed=seed)


def test_fit_matches_reference_pgd_trajectory():
    """Fast covariate-free phase 1 == reference Model.PGD_single_step (float64)."""
    from syngler.fit import fit_lsm
    from syngler.lsm import source as LSM
    from syngler.lsm.source import center_and_rotate, topk_sqrt_eig_embedding
    A = nx.to_numpy_array(_ba()); n, r = 150, 3
    Z0 = center_and_rotate(topk_sqrt_eig_embedding(A - A.mean(), r))
    deg = A.sum(1); a0 = np.log(deg + 1) - np.log(deg + 1).mean()
    p = A.sum() / (n * (n - 1)); rho = np.log(p / (1 - p))
    m = LSM.Model(A, np.zeros((n, n, 1)), alpha=a0.copy(), beta=np.zeros(1), Z=Z0.copy(), sparsity=rho,
                  sparsity_estimation=True)
    ea, eZ = 0.5 / (2 * n), 0.5 / (2 * max(np.sum(Z0 ** 2) / r, 1.0))
    ref = np.array([m.PGD_single_step(ea, eZ, 0.0, if_init=True)[0] for _ in range(25)])
    f = fit_lsm(A, r=r, n_iter_init=25, n_iter_refine=0, sigma_init=0.0, dtype=np.float64, rtol=0)
    assert np.allclose(ref, f.loss_init, rtol=1e-9)


def test_rho_calibration_matches_density():
    from syngler.fit import fit_lsm
    G = _ba(); A = nx.to_numpy_array(G)
    f = fit_lsm(A, r=3, n_iter_init=60, n_iter_refine=30)
    assert abs(f.expected_density() - f.density) < 1e-4


def test_chunked_sampler_matches_reference_distribution():
    from syngler.api import sample_edges_from_latents
    from syngler.utils.source import reconstruct_adjacency
    rng = np.random.default_rng(0)
    Z, a = rng.normal(size=(80, 3)) * 0.5, rng.normal(size=80) * 0.3
    mine = np.mean([len(sample_edges_from_latents(Z, a, -2.0, np.random.default_rng(s), block=17)) for s in range(200)])
    ref = np.mean([reconstruct_adjacency(Z, a, -2.0, seed=s).sum() / 2 for s in range(200)])
    assert abs(mine - ref) / ref < 0.03


def test_sampler_edges_are_simple_and_upper():
    from syngler.api import sample_edges_from_latents
    e = sample_edges_from_latents(np.zeros((30, 2)), np.zeros(30), 0.0, np.random.default_rng(1), block=7)
    assert (e[:, 0] < e[:, 1]).all() and len({tuple(x) for x in e}) == len(e)


@pytest.mark.parametrize("make", [lambda: nx.karate_club_graph(), lambda: _ba(120, 2, 3),
                                  lambda: nx.disjoint_union(nx.complete_graph(5), nx.cycle_graph(8)),
                                  lambda: nx.empty_graph(5), lambda: nx.path_graph(30)])
def test_fast_portrait_equals_reference(make):
    from netsynth.portrait import portrait
    from third_party.portrait_divergence_bagrow import portrait_py
    g = make()
    assert np.allclose(portrait(g), portrait_py(g))


def test_portrait_divergence_properties():
    from netsynth.portrait import portrait_divergence
    g, h = _ba(100, 2, 1), nx.gnm_random_graph(100, 198, seed=1)
    assert portrait_divergence(g, g) == pytest.approx(0, abs=1e-12)
    assert portrait_divergence(g, h) == pytest.approx(portrait_divergence(h, g))
    assert 0 < portrait_divergence(g, h) <= 1


def test_classical_generators():
    from netsynth.generators.classical import generate_ba, generate_cm, generate_er
    G = _ba(200, 3, 2)
    er = generate_er(G, 2); assert all(g.number_of_edges() == G.number_of_edges() for g in er.graphs)
    cm = generate_cm(G, 2); assert cm.info["edge_loss_from_simplification"] < 0.2
    ba = generate_ba(G, 2); assert ba.graphs[0].number_of_nodes() == 200


def test_netgan_runs_and_assembles_graph():
    from netsynth.generators.netgan import generate_netgan
    G = nx.karate_club_graph()
    r = generate_netgan(G, 2, netgan_iters=15, netgan_rw_len=6)
    assert len(r.graphs) == 2 and r.graphs[0].number_of_edges() <= G.number_of_edges()
    assert r.graphs[0].number_of_edges() > 0


def test_syngler_wrapper_and_node_map():
    from netsynth.generators.syngler_gen import generate_syng_r
    G = _ba(120, 3, 5)
    r = generate_syng_r(G, 3, r=3, n_iter_init=80, n_iter_refine=40)
    m = np.mean([g.number_of_edges() for g in r.graphs])
    assert 0.6 * G.number_of_edges() < m < 1.5 * G.number_of_edges()
    assert r.node_maps[0].shape == (120,)


def test_sir_threshold_and_runs():
    from netsynth import epidemic as epi
    G = _ba(300, 4, 1)
    bc = epi.threshold_beta(G, 1.0)
    cfg = epi.SIRConfig(beta=3 * bc, n_runs=10)
    runs = epi.simulate(G, cfg, seed=0)
    grid = epi.common_grid(runs); s = epi.run_stats(runs, grid, 300, cfg)
    assert s["I"].shape == (10, len(grid)) and (s["final"] >= 0).all() and s["final"].max() <= 1


def test_io_formats_roundtrip():
    import gzip, io
    from netsynth.io import load_graph, prepare_graph
    G = nx.karate_club_graph()
    el = "# c\nsource,target\n" + "\n".join(f"{u},{v}" for u, v in G.edges())
    assert load_graph(el.encode(), "a.csv")[0].number_of_edges() == 78
    assert load_graph(gzip.compress(el.encode()), "a.csv.gz")[0].number_of_edges() == 78
    A = nx.to_numpy_array(G, dtype=int)
    assert load_graph("\n".join(",".join(map(str, r)) for r in A).encode(), "m.csv")[0].number_of_edges() == 78
    b = io.BytesIO(); np.save(b, A)
    assert load_graph(b.getvalue(), "m.npy")[0].number_of_edges() == 78
    D = nx.DiGraph(G); D.add_edge(0, 0)
    P, rep = prepare_graph(D)
    assert P.number_of_edges() == 78 and any("self-loop" in n for n in rep["notes"])


def test_pipeline_survives_generator_failure(monkeypatch):
    from netsynth import pipeline
    reg = pipeline._registry()
    def boom(*a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(pipeline, "_registry", lambda: {**reg, "netgan": ("NetGAN", boom)})
    res = pipeline.run_experiment(nx.karate_club_graph(),
                                  pipeline.ExperimentConfig(generators=["er", "netgan", "cm"], k=2, runs_per_graph=5))
    assert res.reports["netgan"].ok is False and res.reports["er"].ok and res.reports["cm"].ok
