# Synth-Net Lab — Technical Documentation

> **Project question:** *When is a synthetic network a safe substitute for simulation or for data sharing?*

This document describes the design, algorithms, metrics, configuration and limits of Synth-Net Lab. For installation and a quick tour see [README.md](README.md).

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [System architecture](#2-system-architecture)
3. [Data handling](#3-data-handling)
4. [Generators](#4-generators)
5. [Structural fidelity metrics](#5-structural-fidelity-metrics)
6. [Portrait divergence](#6-portrait-divergence)
7. [Functional fidelity: SIR epidemics](#7-functional-fidelity-sir-epidemics)
8. [Safe-substitute verdict and leakage check](#8-safe-substitute-verdict-and-leakage-check)
9. [Scalability benchmark](#9-scalability-benchmark)
10. [Dashboard guide](#10-dashboard-guide)
11. [Configuration reference](#11-configuration-reference)
12. [Outputs and file formats](#12-outputs-and-file-formats)
13. [Command-line use and Python API](#13-command-line-use-and-python-api)
14. [Testing and validation](#14-testing-and-validation)
15. [Performance, memory and limits](#15-performance-memory-and-limits)
16. [Threats to validity](#16-threats-to-validity)
17. [Extending the project](#17-extending-the-project)
18. [Glossary](#18-glossary)
19. [References and licences](#19-references-and-licences)

---

## 1. Project overview

Synthetic copies of a network are used in two ways:

* **As a stand-in for simulation** — run an epidemic, a diffusion or a cascade on the synthetic graph instead of the real one.
* **As shareable data** — publish the synthetic graph because the real one is private.

A generator that matches many static statistics may still behave differently when something *runs on it*, and a generator that behaves correctly may reproduce too much of the original. Synth-Net Lab therefore evaluates every generator on four axes:

| Axis | Question | Main measurements |
|---|---|---|
| **Structure** | Does the synthetic graph look like the real one? | degree distribution, clustering, path lengths, assortativity, component structure, **portrait divergence** |
| **Function** | Does a process behave the same on it? | SIR outbreak curves, final size, peak, time to peak, outbreak probability (simulated with **EoN**) |
| **Cost** | What does it take to generate it? | fitting/training time and per-graph generation time versus network size |
| **Safety to share** | Does it copy the original? | share of synthetic edges that are real edges ("leakage"), compared with chance |

Generators compared:

| Key | Generator | Type |
|---|---|---|
| `er` | Erdős–Rényi G(n, m) | classical baseline |
| `ba` | Barabási–Albert | classical baseline |
| `cm` | Configuration model (real degree sequence) | classical baseline |
| `netgan` | NetGAN (random-walk WGAN) | deep generative |
| `syng_r` | SyNGLER, SyNG-R (bootstrap resampling of latents) | latent-space model |
| `syng_d_mlp` | SyNGLER, SyNG-D with a DDPM (MLP) over latents | latent-space model + diffusion |
| `syng_d_forest` | SyNGLER, SyNG-D with ForestDiffusion over latents | latent-space model + diffusion |

The classical baselines have no dependencies beyond NetworkX, so a complete comparison is always available even if a learned generator fails (see [§2.4](#24-failure-isolation)).

---

## 2. System architecture

### 2.1 Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  app.py  — Streamlit dashboard (sidebar inputs, 6 result tabs)        │
│  scripts/run_experiment.py — command-line runner                      │
└───────────────┬──────────────────────────────────────────────────────┘
                │ ExperimentConfig                       ▲ ExperimentResult
┌───────────────▼────────────────────────────────────────┴─────────────┐
│  netsynth/pipeline.py      orchestration                              │
│  netsynth/bundle.py        ResultBundle: save / load / export         │
│  netsynth/reporting.py     verdicts, key findings                     │
│  netsynth/plots.py         Plotly figure builders (from a bundle)     │
│  netsynth/netviz.py        interactive network views (pyvis)          │
│  netsynth/benchmark.py     scalability benchmark                      │
└──────┬──────────────┬───────────────────┬─────────────────┬──────────┘
       │              │                   │                 │
┌──────▼─────┐ ┌──────▼───────┐ ┌─────────▼───────┐ ┌───────▼────────┐
│ io.py      │ │ generators/  │ │ structure.py    │ │ epidemic.py    │
│ sampling.py│ │  classical   │ │ portrait.py     │ │ (EoN SIR)      │
│ load/clean │ │  netgan      │ │ (metrics)       │ │                │
│            │ │  syngler_gen │ │                 │ │                │
└────────────┘ └──────┬───────┘ └─────────┬───────┘ └────────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼──────────────────────────┐
               │ syngler/     │    │ third_party/                    │
               │  fit.py      │    │  portrait divergence routine    │
               │  api.py      │    │  + licence texts                │
               │  lsm/ res/   │    └─────────────────────────────────┘
               │  diff/ utils/│
               └──────────────┘
```

### 2.2 Packages

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI: two modes (*Run an experiment* / *Open saved results*), progress bar, six result tabs, downloads. Contains no analysis logic; every tab reads a `ResultBundle`. |
| `netsynth/io.py` | Parse many file formats into a NetworkX graph; clean the graph (`prepare_graph`); list/load bundled example networks. |
| `netsynth/sampling.py` | `snowball_nodes` / `snowball_sample`: randomised-BFS sub-sampling used for the node cap, the scalability benchmark and the network views of large graphs. |
| `netsynth/generators/base.py` | `GeneratorResult` dataclass and `graph_from_edges`. The common return type of all generators. |
| `netsynth/generators/classical.py` | ER, BA and configuration-model generators. |
| `netsynth/generators/netgan.py` | PyTorch NetGAN: walk sampler, generator, critic, WGAN-GP training loop, score matrix, graph assembly. |
| `netsynth/generators/syngler_gen.py` | Adapters that expose the SyNGLER variants through the common generator interface. |
| `netsynth/structure.py` | Scalar structural summaries, error definitions, degree/clustering distribution distances, edge overlap; per-node profiles and pooled profiles over several graphs (`pooled_profile`) for the distribution plots. |
| `netsynth/portrait.py` | Fast portrait computation and the portrait divergence. |
| `netsynth/epidemic.py` | SIR simulation with EoN, epidemic threshold, curve alignment on a common time grid, functional error metrics, noise floor. |
| `netsynth/pipeline.py` | `run_experiment`: runs everything for one real network and returns an `ExperimentResult`. |
| `netsynth/bundle.py` | `ResultBundle`: one object holding everything the dashboard shows (tables, per-graph metrics, epidemic curves and runs, diagnostics, graphs, node maps). Built from a fresh `ExperimentResult` (`bundle_from_result`) or loaded from a ZIP, a folder or uploaded files (`bundle_from_zip`, `bundle_from_dir`, `bundle_from_files`); written with `bundle_to_zip` / `write_dir`. |
| `netsynth/reporting.py` | Result tables, `Tolerances`, `verdicts` and `takeaways` (the last two take a bundle). |
| `netsynth/plots.py` | The dashboard's charts, each taking a bundle, with a fixed colour per generator. |
| `netsynth/netviz.py` | Interactive pyvis network pages (`network_html`), node colouring modes, copied-edge highlighting and `viz_subgraph` (snowball sample above the node limit). |
| `netsynth/benchmark.py` | `scaling_benchmark` and `scaling_exponents`. |
| `syngler/` | The SyNGLER implementation (see [§4.6](#46-syngler)). |
| `third_party/` | Third-party portrait-divergence routine used for the divergence formula, and the licence texts of bundled components. |
| `tests/test_core.py` | Automated tests (see [§14](#14-testing-and-validation)). |
| `data/samples/` | Bundled example networks (edge lists). |
| `examples/` | Two ready-to-open result bundles (`*.zip`, for *Open saved results*) and the CSV/JSON output of the political-blogs run in `polblogs_results/`. |

### 2.3 Data flow of one experiment

```
 file / example / URL
        │  io.load_graph                       parse any supported format
        ▼
   nx.Graph (raw)
        │  io.prepare_graph                    simple, undirected, LCC, node cap, relabel 0..n-1
        ▼
   real graph G ───────────────────────────────────────────────────────────────┐
        │                                                                      │
        │  structure.summarize(G)    scalars, degree array, clustering array   │
        │  portrait(G)               real portrait B_G                         │
        │  epidemic.threshold_beta   β_c ;  β = β_mult · β_c                   │
        │  epidemic.simulate(G)      K·runs_per_graph SIR runs  → real_stats   │
        │  epidemic.common_grid      time grid from the real runs              │
        │  epidemic.noise_floor      real-vs-real reference error              │
        ▼                                                                      │
   for each selected generator:                                                │
        │  fn(G, K, …) → GeneratorResult   K graphs, node maps, timings        │
        │  for each synthetic graph H:                                         │
        │      summarize(H) → scalars, errors vs G                             │
        │      degree / clustering distribution distances                      │
        │      portrait(H) → portrait_divergence(B_G, B_H)                     │
        │      edge_overlap(G, edges(H), node_map)                             │
        │      simulate(H) (runs_per_graph runs) → compare to real_stats       │
        │  pool all runs → generator-level epidemic statistics                 │
        ▼                                                                      │
   ExperimentResult ◄───────────────────────────────────────────────────────────┘
        │  reporting.*   tables, verdict, key findings
        ▼
   dashboard tabs / CSV + edge-list ZIP
```

SyNGLER variants share one fitted latent space model per experiment (`shared_fit` dictionary), so selecting SyNG-R and SyNG-D together fits the LSM once. The reported fit time of each variant still includes the LSM fit, because each variant needs it.

### 2.4 Failure isolation

`run_experiment` wraps each generator in `try/except`. A failing generator (missing optional dependency, out-of-memory, numerical problem) is recorded in its `GenReport` (`ok=False`, `error`, `trace`) and the remaining generators continue. The dashboard shows an error box with the traceback for the failed generator and renders everything else. This is why the classical baselines always yield a complete result.

### 2.5 Key data structures

| Type | Defined in | Contents |
|---|---|---|
| `GeneratorResult` | `generators/base.py` | `key`, `label`, `graphs` (list of `nx.Graph`, nodes `0..n-1`), `node_maps` (per graph: array mapping synthetic node → real node, or `None`), `fit_seconds`, `gen_seconds`, `info` (diagnostics such as NetGAN history or LSM loss curves). |
| `StructuralSummary` | `structure.py` | `scalars` (dict), `degrees` (array), `clustering` (array of local coefficients). |
| `ExperimentConfig` | `pipeline.py` | generator list, `k`, seed, generator parameters, SIR parameters, `compute_portrait`. |
| `GenReport` | `pipeline.py` | per generator: summaries, per-graph metric errors, degree distances, clustering KS, portrait divergences, edge overlaps, pooled SIR statistics and their comparison to the real network. |
| `ExperimentResult` | `pipeline.py` | real graph, real summary and portrait, SIR config, β_c, time grid, real SIR statistics, noise floor, reports, wall-clock time. |
| `ResultBundle` | `bundle.py` | `meta` (settings, real-network scalars, β, β_c, node-map kinds), `generators` (key, label), `scalar`, `errors`, `fidelity`, `epidemic`, `timing`, `sf` (structure-vs-function) tables, `per_graph`, `curves`, `runs`, `floor`, `diagnostics`, `failed`, and optionally `real` and `graphs` / `node_maps`. Tables are indexed by generator label. |
| `Tolerances` | `reporting.py` | thresholds used by the verdict. |
| `LSMFit` | `syngler/fit.py` | `Z`, `alpha`, `rho`, `r`, `n`, observed `density`, loss curves, convergence flags, seconds. |
| `SyntheticSample` | `syngler/api.py` | `n`, `edges` (m×2 array, u<v), `node_map`, `seconds`. |

---

## 3. Data handling

### 3.1 Supported input formats

| Format | Extensions | Notes |
|---|---|---|
| Edge list | `.txt .csv .tsv .edges .el` and any other extension | Delimiters: whitespace, comma, tab, semicolon. Lines starting with `#`, `%` or `/` are comments. A header row is skipped if its tokens look like column names (`source`, `target`, `from`, `to`, …) or if the first row is non-numeric while the next is numeric. Only the first two columns are used; additional columns (e.g. weights) are ignored. Node labels may be integers or strings. |
| Dense adjacency matrix | `.csv .txt` (square numeric), `.npy` | Auto-detected when all rows have the same length ≥ 3 and the number of rows equals the number of columns (an optional header row is allowed). Any non-zero entry is an edge. A `.npy` of shape (m, 2) is read as an edge array. |
| Sparse matrix | `.npz` (SciPy), `.mtx` (MatrixMarket) | Non-zero = edge. |
| Graph files | `.graphml .gml .gexf .net` (Pajek) | Read with NetworkX. |
| Compressed | any of the above with `.gz` | Decompressed in memory. |
| URL | sidebar → "Download from URL" | Downloads the bytes and then follows the same path as an upload. |

The sidebar "Text format" selector (`auto`, `edgelist`, `adjacency`) overrides detection for delimited text files.

### 3.2 Cleaning (`prepare_graph`)

Applied in this order; every change is reported in the notes line under the metric tiles.

1. Convert to a simple undirected graph: direction is dropped, parallel edges merged, **self-loops removed**.
2. If *Largest component* is on (default), keep only the largest connected component. SIR epidemics and NetGAN walks are meaningful on a connected graph, and path-based metrics are defined on it.
3. If the node count exceeds *Max nodes* (default 1500; 0 = no cap), sub-sample with `snowball_sample`.
4. Relabel nodes to `0..n-1`.
5. Reject graphs with fewer than 10 nodes or 10 edges.

### 3.3 Snowball sampling

`snowball_sample(G, n_target, seed)` picks a random start node, expands breadth-first visiting neighbours in random order, and stops when `n_target` nodes have been collected; if a component is exhausted first, it restarts from another random unvisited node. The result is the induced subgraph on the collected nodes. It is used (a) for the node cap and (b) for the scalability benchmark. Snowball samples over-represent the neighbourhood of the start node, so a sub-sampled network is *not* a statistically faithful miniature of the original; keep the cap high enough, or disable it, when fidelity to the full network matters.

### 3.4 Bundled examples

`data/samples/` holds edge lists for three networks of 1,222 (political blogs), 1,481 (co-authorship) and 1,991 (social) nodes. Two small networks (Zachary karate club, Les Misérables) come from NetworkX.

---

## 4. Generators

All generators share the signature

```python
generate_<name>(G, k, seed=0, progress=None, **params) -> GeneratorResult
```

and receive unknown keyword arguments harmlessly (`**_`), so one parameter dictionary can serve every generator.

For every generator, `fit_seconds` is the training/fitting time and `gen_seconds` is the **mean time to draw one graph** after fitting.

### 4.1 Erdős–Rényi (`er`)

`networkx.gnm_random_graph(n, m)` — the G(n, m) model with the same number of nodes and the **same number of edges** as the real graph. Every pair is equally likely; degrees are approximately Poisson. No fit; node maps are the identity (for the leakage metric, ER's overlap with the real graph equals chance = density, which serves as the reference level).

### 4.2 Barabási–Albert (`ba`)

`networkx.barabasi_albert_graph(n, m_BA)` with `m_BA = max(1, round(m / n))`. Each new node attaches to `m_BA` existing nodes with probability proportional to degree, giving a power-law tail and very little clustering. The edge count is `m_BA·(n − m_BA)`, which only approximately matches the real `m`. Identity node map.

### 4.3 Configuration model (`cm`)

`networkx.configuration_model` with the real degree sequence, collapsed to a simple graph (parallel edges merged, self-loops removed). Degrees are preserved up to the collapse; everything else (clustering, correlations, community structure) is randomised. The fraction of edges lost by simplification is stored in `info["edge_loss_from_simplification"]` and shown in the diagnostics tab. Identity node map (node *i* keeps the real degree of node *i*).

### 4.4 NetGAN (`netgan`)

An implementation of the NetGAN idea in PyTorch.

**Training data.** Uniform random walks of `rw_len` nodes (default 16) on the real graph. Start nodes are drawn uniformly among non-isolated nodes; each step moves to a uniformly chosen neighbour. Walks are sampled in vectorised form from the CSR adjacency.

**Generator.**

```
z ~ N(0, I_16)
h0 = tanh(W_h z),  c0 = tanh(W_c z)
x_0 = 0
for t = 1..T:
    (h_t, c_t) = LSTMCell(x_{t-1}, (h_{t-1}, c_{t-1}))       hidden size 40
    logits_t   = W_up h_t                                     (n logits)
    y_t        = onehot(argmax(logits_t + Gumbel))  with straight-through gradient of
                 softmax((logits_t + Gumbel) / τ),            τ = 0.5
    x_t        = W_down y_t                                   embedding size 64
output: one-hot walk (T × n)
```

**Critic.** `W_down` (n → 64) → LSTM (hidden 30) over the T steps → linear layer on the last hidden state → scalar.

**Objective.** Wasserstein GAN with gradient penalty (λ = 10): the critic minimises `E[D(fake)] − E[D(real)] + λ·E[(‖∇D(x̂)‖₂ − 1)²]` on random interpolates `x̂` between real and generated one-hot walks; the generator minimises `−E[D(fake)]`. `n_critic = 3` critic steps per generator step, batch size 64, Adam (lr 3·10⁻⁴, β = (0.5, 0.9)).

**Stopping.** Fixed iteration budget (default 600) *or* a wall-clock cap (default 300 s), whichever comes first. There is no early stopping on edge overlap. Every `eval_every = 50` iterations the **walk validity** is logged: the share of generated steps that follow an edge of the real graph (a random generator scores ≈ the graph density). It is displayed in *Data & downloads* together with the critic loss.

**Sampling a graph.**

1. Generate `n_walks = min(max(20·m, 20 000), 200 000)` walks from the trained generator.
2. Count every consecutive pair `(u, v)` of every walk into a sparse matrix, symmetrise (`S + Sᵀ`) and zero the diagonal → *score matrix* `S`.
3. Assemble a graph with the real number of edges `m`: first, for every node with a non-empty row of `S`, sample one neighbour with probability proportional to its score (guarantees min-degree ≥ 1 where scores exist); then fill the remaining edges by sampling **without replacement** among the remaining upper-triangle entries with probability proportional to `S`.
4. The K graphs of an experiment are K independent draws from the **same** score matrix (one training run per experiment).

Node identities are those of the real graph, so the identity node map is used for leakage. If `S` has fewer than `m` non-zero pairs, the graph has fewer edges than the real one.

Device: CUDA if available, else CPU. Gradient-penalty double back-propagation through the LSTM is run with cuDNN disabled (a requirement of PyTorch for double backward through RNNs).

### 4.5 Parameter summary (NetGAN)

| Parameter | Default | Meaning |
|---|---|---|
| `rw_len` | 16 | nodes per walk |
| `batch_size` | 64 | walks per step |
| `noise_dim` | 16 | generator noise size |
| `gen_hidden` / `disc_hidden` | 40 / 30 | LSTM sizes |
| `emb_dim` | 64 | node embedding size |
| `n_critic` | 3 | critic steps per generator step |
| `lr` | 3·10⁻⁴ | Adam learning rate |
| `gp_lambda` | 10 | gradient penalty weight |
| `temperature` | 0.5 | Gumbel-softmax temperature |
| `iters` | 600 | maximum iterations |
| `max_seconds` | 300 | wall-clock cap |
| `n_gen_walks` | `min(max(20m, 20000), 200000)` | walks for the score matrix |

### 4.6 SyNGLER

SyNGLER is a two-stage generator.

**Stage 1 — latent space model (LSM).** Each node *i* gets a latent position `Z_i ∈ ℝʳ` and an intercept `α_i` (degree heterogeneity). The probability of an edge is

```
P_ij = σ( Z_iᵀ Z_j + α_i + α_j + ρ ),        σ(x) = 1 / (1 + e^{−x})
```

with a global sparsity intercept `ρ`. The parameters are fitted by projected gradient descent (PGD) to the observed adjacency matrix `A`.

**Stage 2 — resample and reconstruct.** New latent pairs `(Z*, α*)` are drawn, `P*` is computed with the same link and the adjacency is sampled entry-wise from `Bernoulli(P*_ij)` (upper triangle, symmetrised, no self-loops). Variants differ in stage 2:

| Variant | Key | Stage-2 sampler |
|---|---|---|
| SyNG-R | `syng_r` | **bootstrap**: draw `n` node indices with replacement; use `(Z_idx, α_idx)`. Distribution-free, no extra training. |
| SyNG-D (MLP) | `syng_d_mlp` | a small residual-MLP DDPM (hidden 128, 1 000 diffusion steps, EMA 0.999, Adam 10⁻³, batch 64) trained on the stacked latents `[Z | α]`, then ancestral sampling of `n` new rows. Default 2 000 training steps, run on CPU. |
| SyNG-D (Forest) | `syng_d_forest` | ForestDiffusion (variance-preserving diffusion with gradient-boosted trees) on `[Z | α]`. Defaults `n_t = 30`, `duplicate_K = 30`. Requires the optional `ForestDiffusion` and `xgboost` packages. With a very small budget (≲ 10) the generated latents are degenerate and produce far too many edges. |

#### Fitting algorithm (`syngler/fit.py`)

The fit is a covariate-free implementation of the LSM PGD. With `A` the adjacency matrix (zero diagonal), `n` nodes, density `d = m / C(n,2)`:

1. **Initialisation.**
   * `Z⁰ = center_and_rotate( U·√max(Λ,0) )` using the top-`r` eigenpairs of the centred adjacency `A − mean(A)`, plus Gaussian noise (σ = 0.01).
   * `α⁰ = log(deg+1) − mean(log(deg+1))` plus noise (σ = 0.01).
   * `ρ⁰ = logit(d)`, `G⁰ = Z⁰Z⁰ᵀ`.
   * Step sizes `η_α = η₀ / 2n` and `η_Z = η₀ / (2·max(‖Z⁰‖²_F / r, 1))`, with `η₀ = 0.5`.
2. **Phase 1 — PGD on the Gram matrix `G`** (up to `n_iter_init` iterations):
   `Θ = G + α 1ᵀ + 1 αᵀ + ρ⁰`, `P = σ(Θ)` with zero diagonal, `D = A − P`, loss `‖D‖²_F`;
   `G ← J ( G + 2η_Z (D − 10⁻³ G − 10⁻³ I) ) J` (J = centring matrix), `α ← α + 2η_α D·1`.
   Then `Z = center_and_rotate( top-r factor of G )`.
3. **Phase 2 — PGD on `(Z, α)`** (up to `n_iter_refine` iterations): `Z ← center_and_rotate²( Z + 2η_Z D Z )`, `α ← α + 2η_α D·1`.
4. **Intercept calibration.** `α` is re-centred and `ρ` is found by root-finding (Brent) so that the mean of `σ(ZZᵀ + α_i + α_j + ρ)` over off-diagonal pairs equals the observed density `d`.

Each phase stops early when the relative change of the loss falls below `rtol = 10⁻⁶`. Computation uses float32 with in-place updates; eigen-decompositions use float64. Defaults: `r = 6`, 300 phase-1 and 500 phase-2 iterations. The dashboard presets *Fast / Standard / Thorough* are (150, 250), (300, 500) and (600, 1 500) iterations. Phase 2 contributes most of the fidelity gain: on PolBlogs, 600 phase-1 iterations alone give portrait divergence ≈ 0.45 and average clustering 0.175 (real 0.32), whereas adding 300 phase-2 iterations gives ≈ 0.32 and 0.25; further iterations bring only small gains.

#### Chunked sampling (`syngler/api.py`)

`sample_edges_from_latents` computes the logits for `block = 1 024` rows at a time (`block × n` matrix), samples Bernoulli entries, keeps only `j > i` and returns an `(m, 2)` edge array. Peak memory is `O(block·n + m)` instead of `O(n²)`.

#### Node map for SyNG-R

For graph *b*, `idx = rng.integers(0, n, n)`; synthetic node *i* is a bootstrap copy of real node `idx[i]`. The array `idx` is stored as `node_map`, which allows the leakage metric ([§8.2](#82-edge-overlap-leakage)). SyNG-D samplers draw latents from a learned distribution without any correspondence, so their node map is `None` and leakage is reported as *n/a*.

#### Privacy noise for SyNG-R

A latent space model fitted to a small graph can be nearly deterministic: with a handful of nodes and `r = 6` dimensions it assigns probabilities close to 1 to the real edges and close to 0 elsewhere, so resampling reproduces the real graph (on Les Misérables and the karate club, plain SyNG-R reproduces 95–98 % of the real edges). To control this, `SyNGLER.sample(..., latent_noise=σ)` jitters each resampled latent vector and node intercept,

```
Z'_i = (Z_{φ(i)} + σ · sd(Z) · ε_i) / √(1 + σ²) ,     α'_i = (α_{φ(i)} + σ · sd(α) · ε'_i) / √(1 + σ²) ,     ε, ε' ~ N(0, 1)
```

where `sd` is the spread of the fitted values, and **re-calibrates ρ** for every graph so that the expected density equals the real density (Brent root-finding, as in the fit). `σ = 0` is plain SyNG-R. The factor `1/√(1+σ²)` keeps the variance of the latents unchanged, so the noise blurs *which* nodes connect without inflating the spread of node propensities; without it, large σ widens the degree distribution and the epidemic becomes far too strong (on Les Misérables, σ = 2 without the factor gives degree KS 0.97 and an epidemic-curve error about 50× the noise floor; with it, 0.13 and about the noise floor).

In the experiment pipeline `σ` is **chosen automatically**: `tune_latent_noise` scans `σ ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3}` and takes the smallest value for which the mean edge overlap of a few probe graphs, measured above chance ([§8.2](#82-edge-overlap-leakage)), is at most the *copied-edge target* (default 0.25, the verdict's leakage tolerance) times a margin (0.65 up to 100 nodes, 0.8 up to 300, 0.95 above) that absorbs the noisier estimate of small graphs. Several probes are averaged (16, 8, 4 or 2 depending on size) because the overlap of a small graph varies by about ±0.08 from draw to draw. If even σ = 3 misses the target it is used anyway. The tuned σ and the probe measurement are stored in `diagnostics.json` and shown on the *Overview* tab.

Measured excess overlap (above chance) at several σ, 12 graphs each: Les Misérables 0.31 / 0.17 / 0.09 at σ = 1 / 1.5 / 2 with degree KS 0.12 throughout; karate club 0.48 / 0.33 / 0.25 / 0.17 at σ = 1 / 1.5 / 2 / 3 with degree KS ≈ 0.2. Clustering is the structural property that suffers (clustering KS rises, for example, from 0.07 to 0.24 on political blogs at σ = 1 and to about 0.35 on Les Misérables), and the trade-off is real: a very small graph may not reach the target without degrading, in which case the dashboard reports what it finds rather than hiding it.

#### The `SyNGLER` class

```python
from syngler.api import SyNGLER
model = SyNGLER(r=6, n_iter_init=300, n_iter_refine=500, seed=0).fit(G)   # G: nx.Graph | scipy sparse | ndarray
samples = model.sample(10, method="res")          # or "diff-mlp", "diff-forest"
H = samples[0].to_networkx()
```

---

## 5. Structural fidelity metrics

All structural metrics compare the **real** graph G with each synthetic graph H; results are averaged over the K synthetic samples (the dashboard also shows the spread where relevant).

### 5.1 Scalar summaries

| Key | Name | Definition |
|---|---|---|
| `n`, `m` | nodes, edges | counts |
| `density` | density | `2m / (n(n−1))` |
| `mean_degree` | mean degree | `⟨k⟩ = 2m / n` |
| `max_degree` | max degree | `max_i k_i` |
| `degree_second_moment` | `⟨k²⟩` | `mean_i k_i²`. Measures degree heterogeneity; governs the epidemic threshold. |
| `avg_clustering` | average local clustering | `mean_i C_i`, `C_i = 2·t_i / (k_i(k_i−1))` with `t_i` the number of triangles at *i*; `C_i = 0` if `k_i < 2`. |
| `transitivity` | global clustering | `3 × (#triangles) / (#connected triples)` |
| `assortativity` | degree assortativity | Pearson correlation between the degrees at the two ends of an edge; in [−1, 1]; non-finite values (e.g. regular graphs) are reported as 0. |
| `aspl` | average shortest path length | mean distance over reachable ordered node pairs (excluding self-pairs) in the largest component |
| `diameter` | diameter | maximum distance found in the largest component |
| `lcc_fraction` | largest-component share | `|LCC| / n` |
| `n_triangles`, `n_components` | triangles, components | counts (informational) |

**Path statistics** (`path_stats`) use unweighted shortest paths on the largest component, computed by BFS from at most **300 randomly chosen source nodes** (all nodes if the component has ≤ 300). With sampled sources the average is an unbiased estimate and the diameter is a **lower bound** (maximum eccentricity of the sampled sources). The source sample is seeded, so the real and synthetic graphs are measured identically.

### 5.2 Error definitions

Every scalar metric has an error of the synthetic value *s* against the real value *r*, either **relative** or **absolute** as appropriate to its scale:

| Metric | Error |
|---|---|
| mean degree, max degree, ⟨k²⟩, average shortest path, diameter | relative: `|s − r| / |r|` |
| average clustering, transitivity, assortativity, largest-component share | absolute: `|s − r|` (these are bounded quantities; assortativity can be near zero or negative, which would make a relative error unstable) |

These errors populate the heat-map in the *Structure* tab (colour scale capped at 1; the printed number is the true value).

### 5.3 Distribution distances

These compare entire distributions rather than a single number. Synthetic degrees/clustering values of the K samples are pooled in the plots; the tabulated values are averaged per graph.

| Metric | Definition | Range |
|---|---|---|
| **Degree KS** | two-sample Kolmogorov–Smirnov statistic between the real and synthetic degree samples: `sup_k |F_G(k) − F_H(k)|` | [0, 1] |
| **Degree W₁ (normalised)** | 1-Wasserstein (earth-mover) distance between the two degree samples divided by the real mean degree — a unit-free "typical degree shift" | ≥ 0 |
| **Degree JSD** | Jensen–Shannon divergence (base 2) between the two integer-degree probability mass functions | [0, 1] |
| **Clustering KS** | two-sample KS statistic between the real and synthetic *local clustering coefficient* samples | [0, 1] |

KS is sensitive to the bulk of the distribution; W₁ to how far mass has to move (it is more sensitive to the tail than KS); JSD to differences in the probability of individual degree values.

### 5.4 Edge overlap

See [§8.2](#82-edge-overlap-leakage).

---

## 6. Portrait divergence

### 6.1 The network portrait

For a graph with *N* nodes, the **portrait** is the matrix **B** with

> `B[ℓ, k]` = the number of nodes that have exactly `k` nodes at shortest-path distance `ℓ`.

Row `ℓ = 0` has `B[0, 1] = N` (every node has exactly one node at distance 0 — itself). Row `ℓ = 1` is the degree distribution (`B[1, k]` = number of nodes of degree `k`). Higher rows describe how the neighbourhood grows with distance, so a portrait encodes degree distribution, neighbourhood expansion, path lengths and diameter, and — through the number of nodes reached — component structure. It does not require a node correspondence between graphs and can be computed for graphs of different sizes.

Rows go from 0 up to the largest distance found in the graph (at least 1). In a disconnected graph, distances to unreachable nodes do not exist; a start node whose farthest reachable node is closer than ℓ contributes to `B[ℓ, 0]` (zero nodes at that distance), so that every row sums to the number of start nodes.

### 6.2 From portrait to a probability distribution

With `V[ℓ, k] = k` (the column index), define

```
P(ℓ, k) = k · B[ℓ, k] / Σ_{ℓ', k'} k' · B[ℓ', k']
```

Weighting by `k` makes `P(ℓ, k)` the probability that, choosing a node *pair* at random, the pair is at distance `ℓ` and the first node has `k` nodes at that distance. The same is done for the second graph, giving `Q(ℓ, k)`. The two matrices are first padded with zeros to a common shape (and trimmed of empty trailing columns).

### 6.3 Portrait divergence

```
M = ½ (P + Q)
D_JS(G, H) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),          KL measured with log base 2
```

| Property | Statement |
|---|---|
| Range | 0 ≤ D_JS ≤ 1 (log base 2) |
| Identity | 0 if and only if the two portrait distributions are identical (identical graphs give 0) |
| Symmetry | `D_JS(G, H) = D_JS(H, G)` |
| Metric | its square root is a metric |
| Sensitivity | degree distribution, path-length distribution, neighbourhood growth, diameter, component sizes |
| Insensitivity | node labelling; it needs no node alignment |

It is used in the dashboard as the **single structural similarity score** (*Portrait divergence* tab and *Overview*).

### 6.4 Implementation (`netsynth/portrait.py`)

* `portrait(G)` computes **B** with SciPy's compiled unweighted shortest-path routine, processing 256 source nodes at a time. For every source it takes `bincount` of its distances, accumulates `(ℓ, k) → count` in a counter, and builds the final matrix once the global maximum distance is known. Time `O(N·(N + M))`, extra memory `O(256·N)`.
* `portrait_divergence(G, H)` accepts graphs or precomputed portraits and applies the formula above (the divergence itself is computed by the bundled routine in `third_party/`).
* The real graph's portrait is computed once and reused for every synthetic graph.
* The fast portrait is verified to be **identical** to a pure-Python breadth-first reference portrait on several graph types, including disconnected graphs, an empty graph and a path (see §14).

### 6.5 Reading values

* A synthetic graph compared with itself gives exactly 0; two independent samples of the same stable generator typically give small values.
* Values are not on an absolute "good/bad" scale: they depend on network size, density and structure. Use them to *rank* generators on the same real network, and judge the absolute level against the ER baseline on that network. On the bundled 1,222-node network the ER graph has ≈ 0.38 and the best generator ≈ 0.32.
* The dashboard shows the mean and standard deviation over the K samples. *Portrait divergence* also lets you inspect the portraits of the real and a synthetic graph side by side (log-scaled `k·B`).
* A good score is not sufficient for a good simulation substitute — see [§8](#8-safe-substitute-verdict-and-leakage-check) and the *Structure ↔ function* tab.

---

## 7. Functional fidelity: SIR epidemics

### 7.1 Model

Discrete nodes are Susceptible, Infected or Recovered. Over every edge between an infected and a susceptible node, infection is transmitted at rate **β**; infected nodes recover at rate **γ** (default 1, which sets the time unit). Simulation uses the event-driven `fast_SIR` of the EoN library, which draws exactly the continuous-time Markovian process without time-stepping.

At the start `round(init_frac · n)` nodes (at least 1; default 1 %) are infected, chosen uniformly at random for every run.

### 7.2 Choosing β: epidemic threshold

For a fair comparison **the same β and γ are used on the real and all synthetic graphs.** β is expressed relative to the real network's mean-field reference value

```
β_c = γ · ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)
```

which is the condition for early exponential growth of the number of infected nodes in a degree-heterogeneous (uncorrelated) network. It is a **reference scale**, not an exact critical point (the exact final-size threshold of SIR in terms of transmissibility `T = β/(β+γ)` differs slightly, and real networks have degree correlations and clustering). The user sets `β = β_mult · β_c` (default `β_mult = 2`, a clearly supercritical epidemic on the real network). The dashboard also tabulates each synthetic graph's own `β_c`, which shows why degree heterogeneity matters: a graph with smaller ⟨k²⟩ (such as ER) has a higher threshold, so the same β may be subcritical there.

### 7.3 Simulation design

* Real network: `K × runs_per_graph` runs (default 5 × 40 = 200).
* Each synthetic graph: `runs_per_graph` runs (default 40), different seed per graph. The runs of a generator's K graphs are **pooled** for the generator-level statistics; each graph's runs are also compared with the real runs separately (used in *Structure ↔ function*).
* All runs are resampled onto a **common time grid** of 200 points from 0 to 1.05 × the 99th percentile of the real runs' end times. Step functions `I(t)`, `R(t)` are evaluated by `searchsorted`; runs that ended earlier hold their last value (0 infected).

### 7.4 Per-run quantities

| Quantity | Definition |
|---|---|
| **Final size** | `R(∞) / n` — fraction ever infected (attack rate) |
| **Peak prevalence** | `max_t I(t) / n` on the common grid |
| **Time to peak** | the grid time at which `I(t)` is maximal |
| **Outbreak** | a run whose final size exceeds `big_outbreak` (default 5 %) |
| **Outbreak probability** | fraction of runs that are outbreaks (small stochastic die-outs make final size bimodal) |

### 7.5 Functional error metrics

Computed between the pooled synthetic runs and the real runs:

| Metric | Definition |
|---|---|
| **Δ final size** (`final_size_err`) | `| mean final size(syn) − mean final size(real) |` — absolute, in fraction of nodes |
| **Final-size Wasserstein** | 1-Wasserstein distance between the two distributions of final size |
| **Peak error (rel.)** | `| mean peak(syn) − mean peak(real) | / mean peak(real)` |
| **Time-to-peak error (rel.)** | same, for the time to peak |
| **Curve NRMSE** | `RMSE( mean I_syn(t), mean I_real(t) ) / max_t mean I_real(t)` — root-mean-square gap between the mean infection curves, normalised by the real peak height |
| **Outbreak-probability error** | `| p_syn − p_real |` |

### 7.6 Noise floor

Even the real network does not reproduce its own epidemic exactly, because the process is stochastic. The **noise floor** randomly splits the real runs into two halves and computes the same error metrics between the halves. It is displayed as a reference in the *Overview* and *Epidemics* tabs: tolerances much tighter than the floor cannot be met by chance. Because each half has only half the runs, the floor is slightly pessimistic relative to a comparison with all runs.

### 7.7 Plots

* **Outbreak curves:** mean `I(t)` per network with a 10–90 percentile band across runs.
* **Summary bars:** final size, peak prevalence, time to peak and outbreak probability per network, with the real value as a dashed line.
* **Error bars against the noise floor:** the four functional errors per generator, the real-vs-real noise floor (dotted) and the user's tolerance (dashed).

---

## 8. Safe-substitute verdict and leakage check

### 8.1 Tolerances and checks

The *Overview* tab combines structural, functional and leakage checks against tolerances that the user can change. Defaults (`Tolerances` in `reporting.py`) are **rules of thumb**, not standards:

| Check | Value used | Default tolerance |
|---|---|---|
| Structure: degree distribution | KS distance, mean over the K graphs | ≤ 0.25 |
| Epidemic: final-size error | `Δ final size` (absolute) | ≤ 0.05 |
| Epidemic: peak error | relative | ≤ 0.25 |
| Epidemic: curve NRMSE | normalised | ≤ 0.20 |
| Epidemic: outbreak-probability error | absolute | ≤ 0.15 |
| Leakage: real edges copied beyond chance | `leak_excess` ([§8.2](#82-edge-overlap-leakage)) | ≤ 0.25 |

Per generator:

* **Safe simulation substitute** = the degree-distribution check *and* all four epidemic checks pass.
* **Safe for data sharing** = safe simulation substitute *and* the leakage check passes (or leakage is *n/a* because the generator has no node correspondence — then edge-level leakage cannot be measured and this is stated in the findings).

Three design decisions, each made on evidence rather than to favour a generator:

1. **Structure is judged on the degree distribution; portrait divergence is reported but does not gate.** The degree distribution is what drives an SIR epidemic, and on the bundled political-blogs run portrait divergence was uncorrelated with epidemic error (Spearman ρ ≈ 0). An absolute portrait-divergence cut-off is also not meaningful across networks: on small graphs even the best generator sits near 0.4–0.5, so any fixed threshold either passes everything or nothing.
2. **Epidemic tolerances are never tighter than the noise floor.** The effective tolerance is `max(tolerance, real-vs-real noise floor)` ([§7.6](#76-noise-floor), `effective_tolerances`). A generator cannot be required to match the real epidemic more closely than the real network matches itself.
3. **Leakage is measured above chance.** In a dense small graph a random graph already shares many edges with the real one (the karate club has density 0.14), so a raw overlap threshold penalises the graph, not the generator.

The classical baselines are held to the same rules: ER and BA fail the degree-distribution check on every bundled network, so neither is ever reported as a safe substitute.

The *Key findings* list is generated from the numbers: closest generator in structure and in epidemic behaviour, the Spearman correlation between portrait divergence and epidemic error, which generators are safe for simulation/sharing, and which reproduce the dynamics but copy too many edges.

The tolerances should reflect the use case. A sensitivity study with a threshold close to the epidemic's critical point will need a much stricter final-size and outbreak-probability tolerance than a rough what-if analysis.

### 8.2 Edge overlap (leakage)

For a synthetic graph with node map `φ` (synthetic node → real node):

```
overlap = mean over synthetic edges (u, v) with φ(u) ≠ φ(v) of  1[ (φ(u), φ(v)) is an edge of the real graph ]
```

* **Chance level** is the real graph's density; the table also gives `overlap / density` ("× chance").
* **Excess over chance** (used by the verdict): `leak_excess = (overlap − chance) / (1 − chance)`, 0 for a graph that shares only as many edges as luck would, 1 for a graph made only of real edges.
* Node maps: identity for ER, BA, CM and NetGAN (the synthetic graph lives on the real node set; for ER and BA the identity map carries no information, so their overlap equals chance and acts as a reference); the bootstrap index vector for SyNG-R; none for SyNG-D.
* It is a **proxy** for memorisation. It is not a formal privacy guarantee (no differential-privacy accounting, no analysis of de-anonymisation attacks that use structure rather than exact edges).

### 8.3 Structure ↔ function

`structure_function_points` yields one row per synthetic graph with its portrait divergence, degree KS, final-size error, curve NRMSE and peak error (exported as `structure_vs_function.csv`). The key findings quote the Spearman rank correlation between portrait divergence and curve error over all synthetic graphs: a high value would mean that static similarity is a good proxy for dynamical similarity.

---

## 9. Scalability benchmark

`scaling_benchmark(G, sizes, generators, params)`:

1. For each size, take `snowball_sample(G, size)`.
2. For each selected generator, run it with `k = 1` using a **fixed small training budget** (benchmark defaults: NetGAN 100 iterations, SyNGLER "Fast" preset, MLP 500 steps) and record `fit_s`, `gen_s` and `total_s`.
3. `scaling_exponents` fits `time ≈ a · nᵇ` by least squares on log–log data (needs ≥ 3 sizes) and reports `b`.

Interpretation notes:

* With a fixed iteration budget, the exponent reflects the **cost per iteration or per sample** as the network grows, not the cost of reaching a given quality.
* Classical generators run in milliseconds; their curves are clipped at 10⁻⁴ s on the log axes.
* The default sizes are powers of two up to the network size (`default_sizes`), at most five.
* Snowball sub-samples change density and degree structure with size, so exponents are empirical and specific to the sampled family.
* A failed run is recorded with an error message and shown as a warning; other rows are unaffected.

Theoretical expectations (for orientation): LSM fitting costs `O(n²)` per iteration (dense `n × n` operations) and chunked sampling `O(n² r)`; NetGAN costs `O(batch · T · n · H)` per iteration (the softmax over `n` nodes) plus the walk generation for the score matrix; ER/BA/CM are near-linear in `n + m`.

---

## 10. Dashboard guide

The sidebar has two modes.

* **Run an experiment** — choose a network (bundled example or uploaded file), the generators and the number of graphs; *Advanced settings* holds the seed, node cap, SyNGLER latent dimension and copied-edge target, NetGAN iterations, epidemic strength and number of simulations. Settings not shown use the defaults of §11.
* **Open saved results** — open a bundled run or a results ZIP (or the exported CSV/JSON files). The dashboard is rebuilt from the file; nothing is recomputed. Views that need graphs are disabled when the file has none.

Both modes feed the same `ResultBundle` (§2.5), so a fresh run and a reopened file look identical.

### 10.1 Tabs

| Tab | Contents |
|---|---|
| **Overview** | A card per generator saying whether it is a safe substitute *for simulation* and *for data sharing*, with the reasons it fails; tolerance sliders; one summary table (degree distance, portrait divergence, final-size and curve error, share of real edges copied, fit time); generated key findings. |
| **Networks** | Interactive side-by-side views of the real network and a chosen synthetic graph (§10.2). |
| **Structure** | Scalar metrics against the real value, portrait divergence per generator, degree distribution (CCDF), clustering per node. |
| **Epidemics** | Mean outbreak curves with percentile bands, summary bars, errors against the noise floor and tolerances. |
| **Privacy & cost** | Share of synthetic edges that are real edges against chance and tolerance; fit and generation time; scalability benchmark (in an expander). |
| **Data** | Every exported table, per-table CSV download and the full results ZIP. |

If the loaded network or settings change after a run, a banner reminds that the displayed results belong to the previous setup.

### 10.2 Network views

The views are **pyvis** (vis-network) pages embedded in the app: drag nodes, zoom, hover for degree and clustering. Node size is proportional to degree.

* **Size limit.** At most 40–300 nodes are drawn (slider, default 200). Larger graphs are shown as a **snowball sample** (breadth-first from a random seed node). Clustering shown under a view describes the drawn graph, so a sample's value is not that of the full network.
* **Colour** by degree, local clustering, community (greedy modularity) or k-core. Each panel has its own colour scale; the legend states the range.
* **Same node set.** For generators whose synthetic node *i* is real node *i* (ER, BA, configuration model, NetGAN) the synthetic view shows the subgraph induced on the same nodes as the real view, so the two are directly comparable. Otherwise each graph is sampled independently.
* **Copied edges.** Edges of the synthetic graph that are also real edges (through the node map, §8.2) are drawn **red**, with their count and share. Not available for SyNG-D, which has no node correspondence.

### 10.3 Colours

Each generator keeps the same colour in every chart (blue SyNG-R, orange ER, green BA, pink CM, violet NetGAN, red SyNG-D MLP, yellow SyNG-D Forest); the real network is always dark neutral.

---

## 11. Configuration reference

### 11.1 Experiment

| Parameter | UI | Default | Notes |
|---|---|---|---|
| `generators` | Compare | ER, BA, CM, SyNG-R | any subset of the seven keys |
| `k` | Synthetic graphs per generator | 5 | 1–20 |
| `seed` | Random seed | 0 | base seed; graph *b* uses derived seeds |
| `beta_mult` | β as multiple of β_c | 2.0 | 1.0–6.0 |
| `gamma` | Recovery rate | 1.0 | sets the time scale |
| `init_frac` | Initially infected | 0.01 | fraction of nodes |
| `runs_per_graph` | Simulations per synthetic graph | 40 | real network gets `k ×` this |
| `big_outbreak` | Outbreak threshold | 0.05 | final-size fraction |
| `compute_portrait` | — (API only) | True | disable to skip portraits on very large graphs |

### 11.2 Data preparation

| Parameter | Default |
|---|---|
| Largest component only | on |
| Max nodes | 1 500 (0 = no cap) |
| Text format | auto |

### 11.3 SyNGLER

| Parameter | Default | Notes |
|---|---|---|
| `r` | 6 | latent dimension; clipped to `n − 1` |
| `n_iter_init`, `n_iter_refine` | 300, 500 | presets: Fast (150, 250), Standard (300, 500), Thorough (600, 1 500) |
| `eta0` | 0.5 | base step size |
| `rtol` | 10⁻⁶ | relative loss-change stopping rule |
| `mlp_steps` | 2 000 | SyNG-D (MLP) training steps |
| `forest_n_t`, `forest_dup` | 30, 30 | SyNG-D (Forest) budget |

### 11.4 NetGAN

See §4.5. In the UI: iterations (default 600), time cap (300 s), walk length (16).

### 11.5 Benchmark

| Parameter | Default |
|---|---|
| Sizes | powers of two below `n`, plus `n` (≤ 5 values) |
| NetGAN iterations | 100 |
| SyNGLER effort | Fast |
| MLP steps | 500 |

---

## 12. Outputs and file formats

*Data & downloads* offers every table separately as CSV and the whole run as one ZIP (the command-line runner writes the same files to a folder and also produces the ZIP). The ZIP contains:

| File | Contents |
|---|---|
| `meta.json` | settings (K, seed, β, β_c, γ, runs, tolerances-independent run parameters), real-network scalars, generator keys/labels, node-map kinds, noise floor, failed generators (if any), timing of the run |
| `scalar_metrics.csv` | real value and per-generator mean of every scalar structural metric |
| `metric_errors.csv` | per-generator mean error (§5.2) for every scalar metric |
| `fidelity.csv` | portrait divergence (mean, s.d.), degree KS / W₁ / JSD, clustering KS, edge overlap and overlap × chance |
| `epidemic.csv` | final size, peak prevalence, time to peak, outbreak probability, and the functional errors (§7.5) |
| `timing.csv` | fit/training time and per-graph generation time |
| `structure_vs_function.csv` | one row per synthetic graph (§8.3) |
| `per_graph_metrics.csv` | one row per synthetic graph: all scalar metrics, β_c, portrait divergence, degree and clustering KS |
| `epidemic_curves.csv` | per network and time: mean, 10th and 90th percentile of `I(t)` and `R(t)` (`network,time,I_mean,I_p10,I_p90,R_mean,R_p10,R_p90`) |
| `epidemic_runs.csv` | per simulation run: `network,run,final_size,peak_prevalence,time_to_peak` |
| `diagnostics.json` | NetGAN training history, LSM fitting losses and ρ, tuned SyNG-R privacy noise and its leakage probe, diffusion training time, configuration-model edge loss |
| `real/real.edgelist` | the (cleaned) real network, one `u v` pair per line |
| `synthetic/<generator>/graph_<i>.edgelist` | synthetic graphs, nodes `0..n-1` |
| `synthetic/<generator>/node_map_<i>.txt` | synthetic node → real node (only where defined) |

The first group (`scalar_metrics.csv` … `structure_vs_function.csv`) keeps the layout of earlier versions, so existing scripts that read them continue to work. Graph files are optional: *Download everything* can omit them, giving a small file that still opens in the dashboard, with the graph-dependent views disabled.

**Reopening.** *Open saved results* accepts the ZIP, or any subset of these files uploaded together. Missing optional files only disable the views that need them; the minimum is the six summary tables.

`examples/` contains `Les_Miserables_small.zip` (77 nodes, six generators, all graphs — use it to try the network explorer) and `Political_blogs_full.zip` (1,222 nodes, six generators), both openable with *Open saved results*; `examples/polblogs_results/` holds the same political-blogs run as plain files.

---

## 13. Command-line use and Python API

### 13.1 Command line

```bash
python scripts/run_experiment.py --input mynet.txt --out results/ \
       --generators er,ba,cm,netgan,syng_r --k 5 --max-nodes 1500 --seed 0 --netgan-iters 600
```

Prints progress, the fidelity and epidemic tables and the key findings, and writes the complete export of §12 to `--out`, including `<name>_results.zip`, which opens directly in the dashboard (*Open saved results*).

### 13.2 Python

```python
import networkx as nx
from netsynth.io import load_graph, prepare_graph
from netsynth.pipeline import ExperimentConfig, run_experiment
from netsynth import reporting as rp
from netsynth.bundle import bundle_from_result, bundle_to_zip, bundle_from_zip

G0, _ = load_graph(open("net.txt", "rb").read(), "net.txt")
G, info = prepare_graph(G0, take_lcc=True, max_nodes=1500)

cfg = ExperimentConfig(generators=["er", "ba", "cm", "syng_r"], k=5, runs_per_graph=40)
res = run_experiment(G, cfg, progress=lambda msg, frac: print(f"{frac:.0%} {msg}"))

b = bundle_from_result(res, "mynet")          # everything the dashboard shows
print(b.fidelity)
print(b.epidemic)
print(rp.verdicts(b, rp.Tolerances())[["simulation_ok", "sharing_ok"]])
for line in rp.takeaways(b, rp.Tolerances()):
    print("-", line)

open("mynet_results.zip", "wb").write(bundle_to_zip(b))   # reopen later, or in the dashboard
b2 = bundle_from_zip(open("mynet_results.zip", "rb").read())
```

Single pieces can be used directly: `netsynth.portrait.portrait_divergence(G, H)`, `netsynth.structure.summarize(G)`, `netsynth.epidemic.simulate(G, SIRConfig(...))`, `syngler.api.SyNGLER`.

---

## 14. Testing and validation

`pytest -q tests` runs 23 tests in a few seconds:

| Test | What it verifies |
|---|---|
| Fit equals baseline PGD | the Phase-1 loss trajectory of `syngler/fit.py` equals the step-by-step baseline LSM implementation (`syngler/lsm/source.py`) to relative 10⁻⁹ in float64 |
| ρ calibration | after fitting, the model's expected density equals the observed density (|Δ| < 10⁻⁴) |
| Chunked sampler | expected edge count matches the dense sampler `reconstruct_adjacency` within 3 % over 200 draws |
| Sampler output | edges are strictly upper-triangular and unique |
| Portrait equality (5 graph types) | the fast portrait equals the pure-Python breadth-first reference portrait exactly (karate club, BA, disconnected graph, empty graph, path) |
| Portrait divergence properties | `D(G,G) = 0`, symmetry, `0 < D ≤ 1` |
| Classical generators | ER edge count exact; CM simplification loss small; BA node count |
| NetGAN | trains, produces a non-empty graph with ≤ m edges |
| SyNGLER wrapper | edge count within a plausible range; node map has length n |
| SIR | threshold finite, runs have the expected shape, final sizes in [0, 1] |
| I/O | CSV edge list with header, `.gz`, adjacency CSV, `.npy`; directed + self-loop clean-up |
| Failure isolation | a crashing generator is recorded while the others complete |
| Bundle round-trip | a bundle written to ZIP and read back has the same tables, graphs, curves and runs; a graph-free export and a set of loose CSV files also load |
| Verdicts and findings | verdict columns present, key findings produced from a bundle |
| Leakage excess and effective tolerance | excess is 0 at chance and 1 for a copy; an epidemic tolerance is never below the noise floor |
| SyNG-R latent noise | noise lowers edge copying and the density stays near the real density |
| Plot builders | every chart function builds on a real small bundle |
| Network views | pyvis HTML builds for every colour mode with copied-edge highlighting; snowball cap respected |

The dashboard was additionally exercised through Streamlit's headless test harness (a fresh run and both bundled examples) and in a real browser, screenshotting every tab, and a complete run of six generators on a 1,222-node network was executed end to end.

---

## 15. Performance, memory and limits

Measured on a 2-core CPU, no GPU:

| Task | Network | Time |
|---|---|---|
| LSM fit, 300 + 500 iterations | 1,222 nodes | ≈ 8 s |
| 100 fit iterations (previous general-purpose loop for comparison) | 1,222 nodes | 13.5 s vs 1.5 s |
| SyNG-R sampling, per graph | 1,222 nodes | well under a second |
| NetGAN training | 1,222 nodes | ≈ 0.18 s per iteration |
| Portrait | 1,200 nodes | ≈ 0.2 s |
| 20 SIR runs | 1,222 nodes, 16.7k edges | ≈ 0.07 s |
| Complete experiment (ER, BA, CM, SyNG-R; K = 3, 30 runs) | 1,222 nodes | ≈ 20–30 s |

Memory: the LSM fit holds about five dense `n × n` float32 matrices (adjacency, Gram matrix, probabilities, residual and temporaries), i.e. roughly `20·n²` bytes (≈ 30 MB at n = 1,222; ≈ 0.2 GB at n = 3,000; ≈ 2 GB at n = 10,000, with time growing quadratically per iteration). Sampling is chunked and does not add `n²` memory. NetGAN keeps `batch × T × n` activations.

Limits and guidance:

* **Large graphs:** use the node cap (snowball sampling) or run only the classical generators; set `compute_portrait=False` through the Python API when `n·(n+m)` becomes too large (portrait cost is all-pairs BFS).
* **NetGAN quality** depends on the training budget. The default is deliberately small for CPUs; increase iterations or the time cap for better walks, and watch the walk-validity curve.
* **SyNG-D (Forest)** time grows with `n_t × (r+1)` boosted models trained on `n × duplicate_K` rows.
* Pickled or cached objects are not used; every run starts from the uploaded data.

---

## 16. Threats to validity

* **One real network, one realisation.** Conclusions from a single network and seed do not generalise; repeat with several networks and seeds. The dashboard's K samples and many SIR runs reduce *generator* and *process* noise, not network-selection bias.
* **Same-size synthetic graphs.** All generators produce graphs with the real node count. Behaviour for up- or down-scaling is not evaluated.
* **One process.** SIR with a single β/γ pair probes one dynamical regime. Behaviour close to the critical point, other processes (SIS, threshold models, random-walk dynamics) and temporal networks may rank generators differently.
* **Threshold reference.** `β_c` is a mean-field scale (§7.2); "twice the threshold" is not an exactly matched epidemic regime across graphs.
* **Tolerances are subjective.** Defaults are heuristics and some were chosen after looking at typical values; report the tolerances with any verdict. The verdict gates on the degree distribution rather than on portrait divergence, a choice based on one network's correlation result — check it on yours.
* **Privacy noise is a trade-off, not a guarantee.** Latent noise lowers edge copying and is tuned against the same overlap proxy it is measured with; it gives no differential-privacy guarantee, and structure such as clustering degrades as it grows.
* **Leakage proxy.** Edge overlap measures exact-edge copying under a node map; it does not capture attribute inference, structural re-identification or formal privacy loss. SyNG-D has no map and is not assessed.
* **Snowball artefacts.** Capped or benchmark sub-samples are biased towards the start node's neighbourhood.
* **Sampled path statistics.** Average path length is estimated from ≤ 300 sources and diameter is a lower bound on large components.
* **Under-trained deep models.** NetGAN and the SyNG-D variants are run with CPU-friendly budgets; their scores are lower bounds on what a longer training could achieve.
* **Bootstrap duplicates.** SyNG-R can duplicate latent positions, so its graphs contain structurally similar nodes; this affects clustering and assortativity and is one reason its portrait divergence is not zero.

---

## 17. Extending the project

**Add a generator.** Write a function returning a `GeneratorResult` and register it in `netsynth/pipeline.py`:

```python
# netsynth/generators/my_gen.py
def generate_mygen(G, k, seed=0, progress=None, **params):
    graphs = [...]                     # k nx.Graph objects on nodes 0..n-1
    return GeneratorResult("mygen", "My generator", graphs, [None] * k,
                           fit_seconds=..., gen_seconds=..., info={})

# netsynth/pipeline.py  -> _registry():  "mygen": ("My generator", my_gen.generate_mygen)
# netsynth/plots.py     -> COLORS["mygen"] = "#......"
```

It then appears in the sidebar, the metrics, the epidemic comparison and the benchmark automatically. Pass a node map (array or identity) instead of `None` to enable the leakage check.

**Add a metric.** Scalar metrics: add to `SCALAR_METRICS` (label, `"rel"`/`"abs"`) and compute it in `structure.summarize`. Distribution metrics: extend `structure.degree_distances` or add a function and call it in `pipeline.run_experiment`, then surface it in `reporting.py`.

**Add another dynamical process.** Implement a simulate/stats/compare trio like `epidemic.py` and call it from the pipeline loop; add a tab or extend *Epidemics*.

**New input format.** Extend `netsynth/io.load_graph`.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **LSM** | Latent space model: edge probabilities from latent positions and node intercepts. |
| **PGD** | Projected gradient descent. |
| **SyNG-R / SyNG-D** | SyNGLER with resampling (bootstrap) / diffusion in stage 2. |
| **Score matrix** | NetGAN's count matrix of how often node pairs were traversed by generated walks. |
| **Portrait** | Matrix `B[ℓ, k]`: number of nodes with `k` nodes at distance `ℓ`. |
| **Portrait divergence** | Jensen–Shannon divergence between the portrait distributions of two graphs. |
| **CCDF** | Complementary cumulative distribution function, `P(K ≥ k)`. |
| **KS** | Kolmogorov–Smirnov statistic. |
| **W₁** | 1-Wasserstein (earth-mover) distance. |
| **JSD** | Jensen–Shannon divergence. |
| **SIR** | Susceptible–Infected–Recovered epidemic model. |
| **β, γ** | Per-edge transmission rate; recovery rate. |
| **β_c** | Mean-field epidemic reference threshold `γ⟨k⟩/(⟨k²⟩−⟨k⟩)`. |
| **Attack rate / final size** | Fraction of nodes ever infected. |
| **NRMSE** | Root-mean-square error normalised by a reference magnitude. |
| **Noise floor** | Error between two halves of the real runs; the minimum achievable discrepancy. |
| **Leakage / edge overlap** | Share of synthetic edges that are also real edges, under the node map. |
| **LCC** | Largest connected component. |

---

## 19. References and licences

**Methods implemented or used**

* Jiang, F., Bu, Y., Wu, S., Xu, G., Zhu, J. *Efficient Synthetic Network Generation via Latent Embedding Reconstruction.* Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. — latent space model fitting and the SyNG-R / SyNG-D generators.
* Bojchevski, A., Shchur, O., Zügner, D., Günnemann, S. *NetGAN: Generating Graphs via Random Walks.* ICML, 2018.
* Bagrow, J. P., Bollt, E. M. *An information-theoretic, all-scales approach to comparing networks.* Applied Network Science 4, 45 (2019). — network portrait divergence.
* Miller, J. C., Ting, T. *EoN (Epidemics on Networks): a fast, flexible Python package for simulation, analytic approximation, and analysis of epidemics on networks.* Journal of Open Source Software 4(44), 1731 (2019).
* Erdős, P., Rényi, A. *On random graphs I.* (1959); Barabási, A.-L., Albert, R. *Emergence of scaling in random networks.* Science 286 (1999); Molloy, M., Reed, B. *A critical point for random graphs with a given degree sequence.* (1995).
* Gulrajani, I. et al. *Improved Training of Wasserstein GANs.* NeurIPS, 2017. — gradient penalty.
* Jolicoeur-Martineau, A., Fatras, K., Kachman, T. *Generating and imputing tabular data via diffusion and flow-based gradient-boosted trees.* AISTATS, 2024. — ForestDiffusion.
* Hagberg, A., Schult, D., Swart, P. *NetworkX*; SciPy; PyTorch; Streamlit; Plotly.

**Licences.** Components that are included in the source tree under their own licences keep their licence texts in `third_party/`. Dependencies are used under their respective licences.
