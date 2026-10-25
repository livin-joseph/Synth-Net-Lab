# Synth-Net Lab

**When is a synthetic network a safe substitute for simulation or for data sharing?**

Synth-Net Lab generates synthetic copies of a real network with five generators — **Erdős–Rényi, Barabási–Albert, the configuration model, NetGAN and SyNGLER** — and tests them on:

* **Structure** — degree distribution, clustering, path lengths, assortativity, and **Portrait Divergence** as a single similarity score.
* **Function** — **SIR epidemics** (simulated with EoN) on the real and synthetic networks: outbreak curves, final size, peak, outbreak probability.
* **Cost** — fitting and generation time as the network grows.
* **Safety to share** — how many synthetic edges are copies of real edges.

A Streamlit dashboard lets you load a network, run the whole study and read a verdict per generator with tolerances you control.

📖 **Full technical documentation — architecture, every metric, portrait divergence, configuration, limits: [DOCUMENTATION.md](DOCUMENTATION.md)**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # PyTorch is large; the CPU build is enough
streamlit run app.py
```

1. In the sidebar choose a bundled example (political blogs, co-authorship, social, Karate club, Les Misérables) or **upload your own network**, or paste a URL.
2. Select the generators to compare (ER, BA and the configuration model are instant; SyNGLER takes seconds; NetGAN minutes).
3. Press **Run experiment** and explore the tabs.

**Accepted inputs:** edge list (space/comma/tab separated, comments and header allowed), adjacency matrix (`.csv`, `.npy`), `.npz`, `.mtx`, GraphML, GML, GEXF, Pajek `.net`, and `.gz` versions of these. Directed edges are treated as undirected; self-loops and duplicates are removed; the largest component is kept by default; very large networks are capped by snowball sampling (default 1,500 nodes).

### Without the dashboard

```bash
python scripts/run_experiment.py --input mynet.txt --out results/ --generators er,ba,cm,netgan,syng_r --k 5
```

### Tests

```bash
pytest -q tests          # 16 tests, a few seconds
```

## Dashboard tabs

| Tab | What you get |
|---|---|
| **Verdict** | ✔/✘ per generator for *safe simulation substitute* and *safe for data sharing*; adjustable tolerances; auto-generated key findings |
| **Structure** | degree CCDF, clustering distributions, error heat-map over all structural metrics, metric tables |
| **Portrait divergence** | divergence per generator; side-by-side portrait heat-maps |
| **Epidemics** | mean outbreak curves with bands, final-size distributions, epidemic thresholds, noise floor |
| **Structure ↔ function** | does structural similarity predict epidemic similarity? (scatter + Spearman ρ) |
| **Scalability** | fit / generation time vs network size, empirical scaling exponents |
| **Diagnostics & export** | NetGAN and SyNGLER training curves; ZIP of metrics (CSV) and synthetic edge lists |

## Project layout

```
app.py                      Streamlit dashboard
netsynth/                   io · sampling · generators/ · structure · portrait · epidemic ·
                            pipeline · reporting · plots · benchmark
syngler/                    SyNGLER implementation (latent space model fit, resampling, diffusion variants,
                            fast fitting routine and high-level API)
third_party/                portrait-divergence routine and licence texts
scripts/run_experiment.py   command-line runner
tests/                      automated tests
data/samples/               example networks
examples/                   CSV output of a complete example run
```

## Example result

Political-blogs network (1,222 nodes, 16.7k edges); 5 synthetic graphs per generator, 40 SIR runs per graph, β = 2 × the real network's reference threshold, one seed.

| Generator | Portrait divergence ↓ | Degree KS ↓ | Epidemic-curve error ↓ | Δ final size ↓ | Real edges copied |
|---|---|---|---|---|---|
| Erdős–Rényi | 0.380 | 0.569 | 0.406 | 0.192 | 2% (chance) |
| Barabási–Albert | 0.362 | 0.512 | 0.317 | 0.143 | 6% |
| Configuration model | 0.398 | 0.022 | 0.123 | 0.051 | 15% |
| NetGAN (600 iterations, CPU) | 0.422 | 0.054 | 0.033 | 0.015 | 14% |
| **SyNGLER (SyNG-R)** | **0.315** | 0.076 | **0.019** | **0.004** | **40%** |
| SyNGLER (SyNG-D, MLP) | 0.367 | 0.134 | 0.230 | 0.071 | n/a |

The real network's own run-to-run noise gives a curve error of about 0.03. Three observations from this single run: degree heterogeneity drives the epidemic (ER and BA fail); structural distance did **not** predict epidemic error (Spearman ρ ≈ 0); and the best simulation substitute copies about 40% of real edges, so *safe for simulation* is not *safe for sharing*. These are one data point — repeat with several networks and seeds before drawing conclusions. The CSV files are in `examples/polblogs_results/`.

## Things to know

* NetGAN is trained with a small CPU-friendly budget by default (600 iterations / 5-minute cap); more iterations improve it.
* SyNGLER's fit uses dense `n × n` matrices, so it is intended for networks up to a few thousand nodes.
* SyNG-D (Forest) needs the optional `ForestDiffusion` and `xgboost` packages and a reasonable budget (≥ 30 time levels).
* Verdict tolerances are rules of thumb; set them for your application. Edge overlap is a leakage proxy, not a formal privacy guarantee.

More detail on all of these, and on threats to validity, is in [DOCUMENTATION.md](DOCUMENTATION.md).

## References

* Jiang, Bu, Wu, Xu, Zhu — *Efficient Synthetic Network Generation via Latent Embedding Reconstruction*, ICML 2026.
* Bojchevski, Shchur, Zügner, Günnemann — *NetGAN: Generating Graphs via Random Walks*, ICML 2018.
* Bagrow, Bollt — *An information-theoretic, all-scales approach to comparing networks*, Applied Network Science 2019.
* Miller, Ting — *EoN (Epidemics on Networks)*, Journal of Open Source Software 2019.

Licence texts of bundled components are in `third_party/`.
