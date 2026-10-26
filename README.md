# Synth-Net Lab

**When is a synthetic network a safe substitute for simulation or for data sharing?**

Synth-Net Lab generates synthetic copies of a real network with five generators — **Erdős–Rényi, Barabási–Albert, the configuration model, NetGAN and SyNGLER** — and tests them on:

* **Structure** — degree distribution, clustering, path lengths, assortativity, and **Portrait Divergence** as a single similarity score.
* **Function** — **SIR epidemics** (simulated with EoN) on the real and synthetic networks: outbreak curves, final size, peak, outbreak probability.
* **Cost** — fitting and generation time as the network grows.
* **Safety to share** — how many synthetic edges are copies of real edges.

A Streamlit dashboard lets you load a network, run the whole study and read a verdict per generator (safe for simulation? safe for data sharing?), look at the real and synthetic networks side by side, and see the structure, epidemic and privacy results. A run can be saved as one ZIP and reopened later.

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
4. Or switch to **Open saved results** and open one of the bundled runs (`examples/*.zip`).

**Accepted inputs:** edge list (space/comma/tab separated, comments and header allowed), adjacency matrix (`.csv`, `.npy`), `.npz`, `.mtx`, GraphML, GML, GEXF, Pajek `.net`, and `.gz` versions of these. Directed edges are treated as undirected; self-loops and duplicates are removed; the largest component is kept by default; very large networks are capped by snowball sampling (default 1,500 nodes).

### Network views

The **Networks** tab draws the real network next to a synthetic graph with [pyvis](https://pyvis.readthedocs.io): drag, zoom, hover, colour by degree / clustering / community / k-core, and see which synthetic edges are copies of real edges (red). Graphs above the node limit (default 200) are shown as a snowball sample.

### Without the dashboard

```bash
python scripts/run_experiment.py --input mynet.txt --out results/ --generators er,ba,cm,netgan,syng_r --k 5
```

This writes all tables, epidemic curves and graphs to `results/` plus `<name>_results.zip`, which opens in the dashboard via **Open saved results**.

### Tests

```bash
pytest -q tests          # 23 tests, a few seconds
```

## Dashboard tabs

| Tab | What you get |
|---|---|
| **Overview** | per generator: safe substitute *for simulation*? *for data sharing*? (adjustable tolerances), a summary table and generated key findings |
| **Networks** | interactive side-by-side views of the real and a synthetic graph, with copied edges highlighted |
| **Structure** | scalar metrics, portrait divergence, degree distribution, clustering |
| **Epidemics** | outbreak curves, summary bars, errors against the simulation noise floor |
| **Privacy & cost** | share of real edges copied, fit / generation time, scalability benchmark |
| **Data** | every exported table, CSV and full-results ZIP downloads |

### How the verdict works

* **Safe for simulation:** the degree distribution is close (KS ≤ 0.25) and four epidemic errors (final size, peak, outbreak curve, outbreak probability) are within tolerance. Epidemic tolerances are never tighter than the real network's own run-to-run noise.
* **Safe for data sharing:** the above, and the synthetic graph does not copy real edges beyond chance (≤ 25 % of its edges, above what a random graph would share).
* Portrait divergence is shown but is not part of the verdict — on the bundled run it did not predict epidemic error.
* **SyNGLER privacy noise:** on small graphs the fitted model nearly memorises the network and SyNG-R would copy most real edges. The pipeline therefore adds just enough noise to the resampled latents to hit a copied-edge target (default 0.25; *Advanced settings*), and reports the noise it used.

## Project layout

```
app.py                      Streamlit dashboard
netsynth/                   io · sampling · generators/ · structure · portrait · epidemic ·
                            pipeline · bundle · reporting · plots · netviz · benchmark
syngler/                    SyNGLER implementation (latent space model fit, resampling, diffusion variants,
                            fast fitting routine and high-level API)
third_party/                portrait-divergence routine and licence texts
scripts/run_experiment.py   command-line runner
tests/                      automated tests
data/samples/               example networks
examples/                   saved result bundles (*.zip, open in the dashboard) and the CSV/JSON of a complete run
```

## Example result

Political-blogs network (1,222 nodes, 16.7k edges); 5 synthetic graphs per generator, 40 SIR runs per graph, β = 2 × the real network's reference threshold, one seed.

| Generator | Portrait divergence ↓ | Degree KS ↓ | Epidemic-curve error ↓ | Δ final size ↓ | Real edges copied | Safe: simulation / sharing |
|---|---|---|---|---|---|---|
| Erdős–Rényi | 0.380 | 0.569 | 0.388 | 0.184 | 2% (chance) | ✘ / ✘ |
| Barabási–Albert | 0.362 | 0.512 | 0.290 | 0.133 | 6% | ✘ / ✘ |
| Configuration model | 0.398 | 0.022 | 0.105 | 0.039 | 15% | ✔ / ✔ |
| NetGAN (600 iterations, CPU) | 0.422 | 0.054 | 0.046 | 0.019 | 14% | ✔ / ✔ |
| **SyNGLER (SyNG-R)** | **0.356** | 0.144 | 0.088 | 0.028 | 23% | ✔ / ✔ |
| SyNGLER (SyNG-D, MLP) | 0.367 | 0.134 | 0.210 | 0.057 | n/a | ✘ / ✘ |

The real network's own run-to-run noise gives a curve error of about 0.03. Observations from this single run: degree heterogeneity drives the epidemic (ER and BA fail); SyNG-R has the lowest portrait divergence and is a safe substitute for both purposes, with its privacy noise auto-tuned to 0.75; and structural distance did **not** predict epidemic error. Without the noise (σ = 0) SyNG-R would copy about 40% of the real edges (about 18 times chance) and fail the sharing check; on very small graphs the unprotected model copies 95–98%. These are one data point per network — repeat with several networks and seeds before drawing conclusions. The full run is bundled as `examples/Political_blogs_full.zip` (open it with **Open saved results**) and as plain files in `examples/polblogs_results/`.

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
