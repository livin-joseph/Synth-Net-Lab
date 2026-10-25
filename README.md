# Synth-Net-Lab

**When is a synthetic network a safe substitute for simulation or data sharing?**

A Network Science course project that combines two papers —

1. **SyNGLER** – *Efficient Synthetic Network Generation via Latent Embedding Reconstruction* (Jiang, Bu, Wu, Xu, Zhu; ICML 2026). Fit a latent space model `P_ij = σ(Z_iᵀZ_j + α_i + α_j + ρ)` to the real graph, resample the latents `(Z, α)`, rebuild a graph.
2. **NetGAN** – *Generating Graphs via Random Walks* (Bojchevski et al., ICML 2018). A WGAN whose generator emits random walks; a score matrix of walk transitions is turned into a graph.

— and benchmarks them against **Erdős–Rényi, Barabási–Albert and the configuration model** on two axes:

| Axis | What is measured |
|---|---|
| **Structural fidelity** | degree distribution (KS / W₁ / JSD), local & global clustering, average path length & diameter, degree assortativity, largest-component share, and **Portrait Divergence** (Bagrow & Bollt) as the single similarity score |
| **Functional fidelity** | **SIR epidemic** simulated with **EoN** on the real and synthetic networks: outbreak curves, final size, peak, time-to-peak, outbreak probability, epidemic threshold β_c — with a real-vs-real *noise floor* |
| **Cost** | fit/training time and per-graph generation time vs. network size (log–log exponents) |
| **Safety to share** | edge-overlap ("leakage") between synthetic and real edges, for generators where synthetic nodes map to real nodes |

A Streamlit dashboard lets you upload any network, run all of this, and read a verdict per generator with tolerances you control.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch is large; CPU build is fine
streamlit run app.py
```

Pick a bundled example (PolBlogs, DBLP, YouTube from the SyNGLER repo, Karate club, Les Misérables) or upload your own, choose generators, press **Run experiment**.
No Streamlit? `python scripts/run_experiment.py --input mynet.txt --generators er,ba,cm,netgan,syng_r --out results/`.
Tests: `pytest -q tests` (16 tests, ~5 s).

**Input formats:** edge list (whitespace/comma/tab; `#`/`%` comments; header auto-skipped; extra columns ignored), dense adjacency matrix (`.csv`, `.npy`), sparse `.npz`, MatrixMarket `.mtx`, GraphML, GML, GEXF, Pajek `.net`, any of them `.gz`; or paste a URL (e.g. a SNAP `.txt.gz`). Graphs are made simple/undirected; options keep the largest component and cap the node count by snowball sampling.

## Dashboard tabs

1. **Verdict** – per-generator ✔/✘ for *safe simulation substitute* and *safe for data sharing*, from tolerances you set (portrait divergence, final-size error, peak error, curve NRMSE, outbreak-probability error, max edge overlap). Auto-generated key findings.
2. **Structure** – degree CCDF, clustering distributions, error heat-map across all structural metrics, metric tables.
3. **Portrait divergence** – bar chart with spread over samples; side-by-side portrait heat-maps real vs. generated.
4. **Epidemics** – mean I(t) with 10–90 % bands, final-size distributions, summary table, threshold β_c.
5. **Structure ↔ function** – one point per synthetic graph: does structural similarity predict epidemic similarity? (Spearman ρ)
6. **Scalability** – timing on snowball sub-samples, fit vs. generate, empirical exponent.
7. **Diagnostics & export** – NetGAN critic loss / walk validity, LSM loss, CSV + edge-list ZIP download.

## Layout

```
app.py                     Streamlit dashboard
netsynth/                  project code
  io.py sampling.py          loading / cleaning / snowball sampling
  generators/                classical.py  netgan.py  syngler_gen.py  (common interface)
  structure.py portrait.py   structural metrics, fast portraits + divergence
  epidemic.py                SIR with EoN, threshold, curves, noise floor
  pipeline.py reporting.py   experiment orchestration, tables, verdict
  benchmark.py plots.py
syngler/                   SyNGLER package, vendored from the repo and extended (below)
third_party/               Bagrow's portrait_divergence.py (MIT, unmodified) + licences
scripts/run_experiment.py  CLI;   tests/ pytest;   examples/ sample output;   data/samples/ example networks
```

## What was developed on top of the SyNGLER codebase

The repository ships SyNGLER as batch scripts. Its Cora runner imports modules (`LSM_source`, `SyNG_source`) that are not in the package, `run_real` is referenced but missing, and the fit loop prints every iteration. For this project:

* **`syngler/fit.py`** – a covariate-free re-implementation of the same PGD updates (spectral init → Phase 1 on the Gram matrix → Phase 2 on `Z, α` → ρ calibrated to the observed density, following `run_cora_syngr.fit_lsm`). Float32, in place, relative stopping rule, progress callback. **~9× faster** (n = 1,222: 13.5 s → 1.5 s per 100 iterations). `tests/test_core.py` checks the Phase-1 loss trajectory equals the reference `Model.PGD_single_step` to 1e-9.
* **`syngler/api.py`** – `SyNGLER(r).fit(G).sample(k, method="res" | "diff-mlp" | "diff-forest")`. Sampling is **chunked** (never builds the n×n probability matrix → memory O(block·n + m)) and records a `node_map` (which real node each bootstrapped node came from) used for the leakage check. Chunked sampler is tested against `reconstruct_adjacency`.
* The `res`, `diff` (forest + MLP) and `lsm.source` modules are used unchanged.

`syngler/fit.py` still needs dense `n×n` matrices, so SyNGLER is capped at a few thousand nodes by default (the app warns and can snowball-sample).

## Example result (PolBlogs, 1,222 nodes, 16.7k edges; 5 graphs × 40 SIR runs; β = 2·β_c; one seed)

| Generator | Portrait div. ↓ | Degree KS ↓ | Curve NRMSE ↓ | Δ final size ↓ | Edge overlap |
|---|---|---|---|---|---|
| Erdős–Rényi | 0.380 | 0.569 | 0.406 | 0.192 | 2% (chance) |
| Barabási–Albert | 0.362 | 0.512 | 0.317 | 0.143 | 6% |
| Configuration model | 0.398 | 0.022 | 0.123 | 0.051 | 15% |
| NetGAN (600 it, CPU) | 0.422 | 0.054 | 0.033 | 0.015 | 14% |
| **SyNGLER SyNG-R** | **0.315** | 0.076 | **0.019** | **0.004** | **40%** |
| SyNGLER SyNG-D (MLP) | 0.367 | 0.134 | 0.230 | 0.071 | n/a |

Real-network noise floor of the curve NRMSE ≈ 0.03. Full CSVs in `examples/polblogs_results/`. Take this as one data point: single seed, default tolerances.
What it illustrates: (i) degree heterogeneity drives the epidemic – ER/BA fail badly; (ii) **structural distance did not predict epidemic error** (Spearman ρ ≈ 0 across graphs: NetGAN has the worst portrait divergence yet near-perfect dynamics); (iii) the best simulator, SyNG-R, **copies ~40 % of real edges**, so "safe for simulation" ≠ "safe for sharing".

## Design decisions and caveats — please read

* **NetGAN is a PyTorch re-implementation** (LSTM generator with Gumbel-softmax straight-through, LSTM critic, WGAN-GP, score matrix → graph with the real edge count and ≥ 1 edge per node). The original is TensorFlow 1.x. It is under-trained by default (600 iterations / 5-minute cap, no early stopping on edge overlap) so that it runs on a CPU; more iterations improve it.
* **SyNG-D (Forest)** needs a decent budget: with `n_t`, `duplicate K` ≲ 10 the latents are degenerate (we saw ~12× too many edges); 30+ behaves.
* **Portrait** is computed with a fast SciPy implementation that reproduces the reference `portrait_py` matrix exactly (tested); the divergence is Bagrow's original code. Path statistics use BFS from ≤ 300 sources on big components (diameter is then a lower bound).
* **Edge overlap** is meaningful only where synthetic nodes correspond to real ones: identity (ER/BA/CM/NetGAN — ER's value is the chance level = density) or the bootstrap index (SyNG-R). SyNG-D has no correspondence → n/a. It is a simple leakage proxy, **not a formal privacy guarantee**.
* **Tolerances are rules of thumb**, not standards. The default portrait-divergence limit (0.35) was set after seeing typical values; change it to your use case. The SIR comparison uses the *same* β, γ for all graphs (β = multiple of the real network's mean-field threshold).
* SNAP could not be reached from the sandbox where this was built, so URL loading was tested only on local files/formats.
* ER uses `G(n, m)`; BA uses `m_BA = round(m/n)` (edge count only approximately matched); the configuration model collapses multi-edges, reporting the lost share.

## Suggested write-up angles

Does a lower portrait divergence mean a better epidemic proxy? How does degree heterogeneity (⟨k²⟩) set β_c in the real vs. synthetic graphs? Training-cost curves (NetGAN grows with the n-way softmax; LSM fitting with n²). Where does bootstrapping nodes (SyNG-R) lose structure (clustering, assortativity)? Sweep β/β_c to find where each generator stops being a good substitute.

## Credits & licences
SyNGLER (MIT) – github.com/FeifanJiang/syngler · Network Portrait Divergence (MIT) – github.com/bagrow/network-portrait-divergence · NetGAN – Bojchevski et al. 2018 · EoN – Miller & Ting. Licences in `third_party/`.
