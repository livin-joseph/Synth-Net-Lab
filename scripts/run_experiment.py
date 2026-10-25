"""Command-line runner (no Streamlit): python scripts/run_experiment.py --input net.txt --out results/"""
import argparse, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netsynth import io as nio, reporting as rp
from netsynth.pipeline import ExperimentConfig, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--generators", default="er,ba,cm,syng_r")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-nodes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--netgan-iters", type=int, default=600)
    a = ap.parse_args()
    p = Path(a.input)
    G0, _ = nio.load_graph(p.read_bytes(), p.name)
    G, rep = nio.prepare_graph(G0, max_nodes=a.max_nodes)
    print(f"network: {rep['nodes']} nodes, {rep['edges']} edges", *rep["notes"], sep="\n  ")
    cfg = ExperimentConfig(generators=a.generators.split(","), k=a.k, seed=a.seed,
                           gen_params=dict(netgan_iters=a.netgan_iters))
    res = run_experiment(G, cfg, progress=lambda m, f: print(f"[{f:5.0%}] {m}", flush=True))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, f in [("scalar_metrics", rp.scalar_table), ("metric_errors", rp.error_table), ("fidelity", rp.fidelity_table),
                    ("epidemic", rp.sir_table), ("timing", rp.timing_table), ("structure_vs_function", rp.structure_function_points)]:
        f(res).to_csv(out / f"{name}.csv")
    print(rp.fidelity_table(res).round(3).to_string())
    print(rp.sir_table(res).round(3).to_string())
    for t in rp.takeaways(res, rp.Tolerances()):
        print("-", t)


if __name__ == "__main__":
    main()
