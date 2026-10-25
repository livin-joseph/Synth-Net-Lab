"""Synth-Net Lab - when is a synthetic network a safe substitute?  (Streamlit dashboard)

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from netsynth import io as nio
from netsynth import plots
from netsynth import reporting as rp
from netsynth.benchmark import default_sizes, scaling_benchmark, scaling_exponents
from netsynth.pipeline import ExperimentConfig, generator_labels, run_experiment
from netsynth.portrait import portrait
from netsynth.structure import summarize

st.set_page_config(page_title="Synth-Net Lab", page_icon="🕸️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1300px}
div[data-testid="stMetric"]{background:#f1f0ed;border-radius:10px;padding:10px 14px}
.ok{color:#0a7d3b;font-weight:600}.bad{color:#b3261e;font-weight:600}
</style>""", unsafe_allow_html=True)

LABELS = generator_labels()
PRESETS = {"Fast": (150, 250), "Standard": (300, 500), "Thorough": (600, 1500)}
HELP = {
    "er": "Same n and m, edges placed uniformly at random. Baseline with no structure.",
    "ba": "Preferential attachment, m_BA = round(m/n). Heavy tail, but almost no clustering.",
    "cm": "Keeps the exact degree sequence, randomises everything else.",
    "netgan": "WGAN over random walks (PyTorch re-implementation). Slow to train; many iterations needed.",
    "syng_r": "SyNGLER: fit a latent space model, bootstrap-resample the latents, rebuild the graph.",
    "syng_d_mlp": "SyNGLER with a small DDPM over the latents (CPU).",
    "syng_d_forest": "SyNGLER with ForestDiffusion over the latents (slow, needs `ForestDiffusion`).",
}


# --------------------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def parse_and_prepare(raw: bytes, filename: str, fmt: str, lcc: bool, max_nodes: int):
    G0, notes = nio.load_graph(raw, filename, fmt)
    G, rep = nio.prepare_graph(G0, take_lcc=lcc, max_nodes=max_nodes or None)
    rep["notes"] = notes + rep["notes"]
    return G, rep


@st.cache_data(show_spinner=False)
def sample_bytes(ref: str):
    if ref.startswith("nx:"):
        G = nio.load_sample(ref)
        txt = "\n".join(f"{u} {v}" for u, v in G.edges())
        return txt.encode(), ref.replace("nx:", "") + ".txt"
    p = Path(ref)
    return p.read_bytes(), p.name


@st.cache_data(show_spinner=False)
def real_overview(_G, key):
    s = summarize(_G)
    return s


with st.sidebar:
    st.title("🕸️ Synth-Net Lab")
    st.caption("Is a synthetic network a safe substitute for simulation or data sharing?")
    st.subheader("1 · Real network")
    src = st.radio("Source", ["Bundled example", "Upload file", "Download from URL"], horizontal=False)
    raw, fname = None, None
    fmt = "auto"
    if src == "Bundled example":
        samples = nio.sample_networks()
        choice = st.selectbox("Example", list(samples), index=2 if len(samples) > 2 else 0)
        st.caption(samples[choice][1])
        raw, fname = sample_bytes(samples[choice][0])
    elif src == "Upload file":
        up = st.file_uploader("Network file", help="Supported: " + nio.SUPPORTED)
        fmt = st.selectbox("Text format", ["auto", "edgelist", "adjacency"],
                           help="Only matters for delimited text files; 'auto' detects square matrices.")
        if up is not None:
            raw, fname = up.getvalue(), up.name
    else:
        url = st.text_input("URL", placeholder="https://snap.stanford.edu/data/ca-GrQc.txt.gz")
        if url and st.button("Fetch"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "synth-net-lab"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    st.session_state["url_raw"] = (r.read(), url.split("/")[-1].split("?")[0] or "download.txt")
            except Exception as e:
                st.error(f"Download failed: {e}")
        if "url_raw" in st.session_state:
            raw, fname = st.session_state["url_raw"]
            st.caption(f"Loaded {fname} ({len(raw) / 1e6:.1f} MB)")
    c1, c2 = st.columns(2)
    lcc = c1.checkbox("Largest component", True, help="Recommended: epidemics and NetGAN walks need a connected graph.")
    max_nodes = c2.number_input("Max nodes", 0, 20000, 1500, 100,
                                help="Larger networks are snowball-sampled down. SyNGLER's fit uses dense n×n matrices "
                                     "(~n² memory); 0 = no cap.")

    st.subheader("2 · Generators")
    keys = st.multiselect("Compare", list(LABELS), default=["er", "ba", "cm", "syng_r"],
                          format_func=lambda k: LABELS[k])
    for k in keys:
        st.caption(f"**{LABELS[k]}** - {HELP[k]}")
    K = st.slider("Synthetic graphs per generator", 1, 20, 5)
    seed = st.number_input("Random seed", 0, 10_000, 0)

    with st.expander("SyNGLER settings"):
        preset = st.select_slider("Fit effort", list(PRESETS), "Standard",
                                  help="PGD iterations for LSM fitting (phase 1 / phase 2).")
        r_dim = st.slider("Latent dimension r", 1, 16, 6)
        mlp_steps = st.number_input("SyNG-D (MLP) training steps", 200, 30000, 2000, 200)
        forest_nt = st.number_input("SyNG-D (Forest) time levels", 2, 50, 30, help="Too small a budget (< ~20) gives degenerate latents and far too many edges.")
        forest_dup = st.number_input("SyNG-D (Forest) duplicate K", 1, 100, 30)
    with st.expander("NetGAN settings"):
        gan_iters = st.number_input("Training iterations", 20, 20000, 600, 20)
        gan_cap = st.number_input("Time cap (seconds)", 10, 7200, 300, 10)
        gan_len = st.slider("Random-walk length", 4, 24, 16)
    with st.expander("Epidemic (SIR) settings"):
        beta_mult = st.slider("β as multiple of real threshold β_c", 1.0, 6.0, 2.0, 0.1,
                              help="β_c = γ⟨k⟩/(⟨k²⟩−⟨k⟩) of the real network. 1 = critical; larger = bigger outbreaks.")
        gamma = st.number_input("Recovery rate γ", 0.1, 5.0, 1.0, 0.1)
        init_frac = st.slider("Initially infected fraction", 0.001, 0.2, 0.01, 0.001, format="%.3f")
        runs = st.slider("Simulations per synthetic graph", 5, 200, 40)
        big = st.slider("'Outbreak' = final size above", 0.01, 0.5, 0.05, 0.01)

    run_clicked = st.button("▶ Run experiment", type="primary", width="stretch",
                            disabled=(raw is None or not keys))

# --------------------------------------------------------------------------- main
st.title("Synth-Net Lab")
st.caption("Generate synthetic copies of a real network with ER · BA · configuration model · NetGAN · SyNGLER, "
           "then test them on **structure** (degree, clustering, paths, assortativity, portrait divergence) and "
           "**function** (SIR epidemics via EoN).")

if raw is None:
    st.info("Pick a bundled example or upload a network in the sidebar to begin. "
            f"Accepted: {nio.SUPPORTED}.")
    st.stop()

try:
    G, prep = parse_and_prepare(raw, fname, fmt, lcc, int(max_nodes))
except Exception as e:
    st.error(f"Could not read the network: {e}")
    st.stop()

ov = real_overview(G, (fname, len(raw), lcc, int(max_nodes)))
m = st.columns(5)
m[0].metric("Nodes", f"{prep['nodes']:,}")
m[1].metric("Edges", f"{prep['edges']:,}")
m[2].metric("Mean degree", f"{ov.scalars['mean_degree']:.2f}")
m[3].metric("Avg clustering", f"{ov.scalars['avg_clustering']:.3f}")
m[4].metric("Assortativity", f"{ov.scalars['assortativity']:.3f}")
if prep["notes"]:
    st.caption("Preparation: " + " ".join(prep["notes"]))
if prep["nodes"] > 3000 and any(k.startswith("syng") for k in keys):
    st.warning("SyNGLER's LSM fit holds several dense n×n float32 matrices (≈ 5 × 4n² bytes) - "
               f"for n = {prep['nodes']:,} that is about {20 * prep['nodes'] ** 2 / 1e9:.1f} GB and minutes of compute.")

if run_clicked:
    bar = st.progress(0.0, text="Starting…")
    cfg = ExperimentConfig(
        generators=keys, k=int(K), seed=int(seed),
        gen_params=dict(r=int(r_dim), n_iter_init=PRESETS[preset][0], n_iter_refine=PRESETS[preset][1],
                        mlp_steps=int(mlp_steps), forest_n_t=int(forest_nt), forest_dup=int(forest_dup),
                        netgan_iters=int(gan_iters), netgan_max_seconds=float(gan_cap), netgan_rw_len=int(gan_len)),
        beta_mult=float(beta_mult), gamma=float(gamma), init_frac=float(init_frac),
        runs_per_graph=int(runs), big_outbreak=float(big))
    t0 = time.time()
    st.session_state["result"] = run_experiment(G, cfg, progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
    st.session_state["result_for"] = (fname, len(raw), lcc, int(max_nodes))
    st.session_state.pop("bench", None)
    bar.empty()

res = st.session_state.get("result")
if res is None:
    st.subheader("Real network preview")
    fig = go.Figure(go.Scatter(x=plots.ccdf(ov.degrees)[0], y=plots.ccdf(ov.degrees)[1], mode="lines",
                               line=dict(color=plots.REAL, width=3)))
    fig = plots._layout(fig, "Degree distribution (CCDF, log–log)", xt="degree k", yt="P(K ≥ k)", legend=False)
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.info("Choose generators in the sidebar and press **Run experiment**.")
    st.stop()

if st.session_state.get("result_for") != (fname, len(raw), lcc, int(max_nodes)):
    st.warning("These results belong to a previously loaded network or setting - press Run to refresh.")

ok = rp.ok_reports(res)
color_of_label = {r.label: plots.COLORS[k] for k, r in ok.items()}
st.caption(f"Experiment finished in {res.seconds:.0f}s · SIR: β = {res.sir_cfg.beta:.4f} "
           f"({res.config.beta_mult:.1f} × β_c = {res.beta_c:.4f}), γ = {res.sir_cfg.gamma}, "
           f"{res.sir_cfg.init_frac:.1%} initially infected.")
for k, r in res.reports.items():
    if not r.ok:
        st.error(f"{r.label} failed: {r.error}. The other generators are unaffected.")
        with st.expander(f"Traceback - {r.label}"):
            st.code(r.trace or "")
if not ok:
    st.stop()

tabs = st.tabs(["✅ Verdict", "📐 Structure", "🧬 Portrait divergence", "🦠 Epidemics",
                "🔗 Structure ↔ function", "⏱ Scalability", "🔬 Diagnostics & export"])

# ---- Verdict
with tabs[0]:
    st.subheader("When is a synthetic network a safe substitute?")
    with st.expander("Tolerances (what counts as 'close enough')", expanded=False):
        a, b, c = st.columns(3)
        tol = rp.Tolerances(
            portrait_divergence=a.slider("Portrait divergence ≤", 0.02, 0.8, 0.35, 0.01),
            final_size_abs=a.slider("Final-size error ≤ (abs.)", 0.01, 0.3, 0.05, 0.01),
            peak_rel=b.slider("Peak-prevalence error ≤ (rel.)", 0.05, 1.0, 0.25, 0.05),
            curve_nrmse=b.slider("Outbreak-curve NRMSE ≤", 0.05, 1.0, 0.20, 0.01),
            outbreak_prob_abs=c.slider("Outbreak-probability error ≤", 0.02, 0.5, 0.15, 0.01),
            max_edge_overlap=c.slider("Max share of synthetic edges that are real", 0.0, 1.0, 0.25, 0.01,
                                      help="Leakage check for data sharing; only measurable where synthetic nodes "
                                           "correspond to real nodes."))
    st.caption(f"Reference: re-running the *real* network gives an epidemic-curve NRMSE of about "
               f"{res.floor.get('curve_nrmse', float('nan')):.3f} and final-size error "
               f"{res.floor.get('final_size_err', float('nan')):.3f} - tolerances tighter than this cannot be met by chance.")
    v = rp.verdicts(res, tol)
    cols = st.columns(len(v))
    for col, (g, row) in zip(cols, v.iterrows()):
        with col:
            st.markdown(f"**{g}**")
            st.markdown(f"Simulation: {'<span class=ok>✔ safe</span>' if row.simulation_ok else '<span class=bad>✘ not safe</span>'}",
                        unsafe_allow_html=True)
            st.markdown(f"Data sharing: {'<span class=ok>✔ safe</span>' if row.sharing_ok else '<span class=bad>✘ not safe</span>'}",
                        unsafe_allow_html=True)
            if row.failed:
                st.caption("Failed: " + "; ".join(row.failed))
    st.markdown("#### Key findings")
    for t in rp.takeaways(res, tol):
        st.markdown("- " + t)
    detail = []
    for g, row in v.iterrows():
        for name, val, thr, _ in row.checks:
            detail.append({"Generator": g, "Check": name, "Value": val, "Tolerance": thr, "Pass": val <= thr})
        if row.leak is not None:
            detail.append({"Generator": g, "Check": "Leakage: edge overlap", "Value": row.leak,
                           "Tolerance": tol.max_edge_overlap, "Pass": row.leak <= tol.max_edge_overlap})
    with st.expander("All checks"):
        st.dataframe(pd.DataFrame(detail).style.format({"Value": "{:.3f}", "Tolerance": "{:.3f}"}), width="stretch")

# ---- Structure
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.degree_ccdf_fig(res, ok), width="stretch")
    c2.plotly_chart(plots.clustering_fig(res, ok), width="stretch")
    err = rp.error_table(res)
    st.plotly_chart(plots.error_heatmap(err), width="stretch")
    st.caption("Relative error for degree-based and path metrics, absolute error for clustering, assortativity "
               "and component fraction. Tables show mean over the synthetic samples.")
    st.markdown("**Scalar metrics (mean over samples)**")
    st.dataframe(rp.scalar_table(res).T.style.format("{:.3f}"), width="stretch")
    st.markdown("**Distribution distances to the real degree / clustering distributions**")
    fid = rp.fidelity_table(res)
    st.dataframe(fid[["Degree KS", "Degree W1 (norm.)", "Degree JSD", "Clustering KS"]].style.format("{:.3f}"),
                 width="stretch")

# ---- Portrait divergence
with tabs[2]:
    fid = rp.fidelity_table(res)
    if fid["Portrait divergence"].notna().any():
        st.plotly_chart(plots.pd_bar(fid), width="stretch")
        st.caption("Portrait divergence (Bagrow & Bollt) is the Jensen–Shannon divergence between the graphs' "
                   "shortest-path 'portraits'. One number that mixes degree, path-length, clustering and "
                   "component information.")
        g = st.selectbox("Inspect portraits of", list(ok), format_func=lambda k: ok[k].label)
        Bs = portrait(ok[g].result.graphs[0])
        st.plotly_chart(plots.portrait_heatmaps(res.real_portrait, Bs, ok[g].label), width="stretch")
        st.caption("Rows: distance ℓ. Columns: k, number of nodes at that distance from a start node. "
                   "Colour: log-scaled mass of start nodes (weighted by k, as in the divergence).")
    else:
        st.info("Portrait computation was disabled.")

# ---- Epidemics
with tabs[3]:
    st.plotly_chart(plots.epidemic_curves(res, ok), width="stretch")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(plots.final_size_fig(res, ok), width="stretch")
    with c2:
        st.markdown("**Epidemic summary**")
        st.dataframe(rp.sir_table(res).style.format("{:.3f}", na_rep="-"), width="stretch")
        st.caption("Real-network noise floor (two halves of the real runs): " +
                   ", ".join(f"{k} {v:.3f}" for k, v in res.floor.items()))
    thr = pd.DataFrame({r.label: [res.beta_c, float(np.mean(r.beta_threshold))] for r in ok.values()},
                       index=["Real β_c", "Synthetic β_c"]).T
    st.markdown("**Epidemic threshold β_c** (mean-field, from ⟨k⟩ and ⟨k²⟩) - the simplest reason why degree "
                "heterogeneity matters for epidemics")
    st.dataframe(thr.style.format("{:.4f}"), width="stretch")

# ---- Structure <-> function
with tabs[4]:
    pts = rp.structure_function_points(res)
    c1, c2 = st.columns(2)
    xcol = c1.selectbox("Structural score (x)", ["Portrait divergence", "Degree KS"])
    ycol = c2.selectbox("Functional error (y)", ["Curve NRMSE", "Δ final size", "Peak err (rel.)"])
    d = pts.dropna(subset=[xcol])
    st.plotly_chart(plots.structure_function_fig(d, xcol, ycol, color_of_label), width="stretch")
    if len(d) >= 4:
        from scipy.stats import spearmanr
        rho, p = spearmanr(d[xcol], d[ycol])
        st.markdown(f"Spearman ρ = **{rho:.2f}** (p = {p:.2g}, n = {len(d)} graphs). A high ρ would mean structural "
                    "similarity is a good proxy for dynamical similarity; points far from the trend show where it is not.")

# ---- Scalability
with tabs[5]:
    st.markdown("Time the generators on snowball sub-samples of increasing size (one synthetic graph each, "
                "fixed training budget per generator).")
    c1, c2 = st.columns(2)
    sizes = c1.multiselect("Network sizes", sorted(set(default_sizes(prep["nodes"]) + [50, 100, 200, 400, 800, 1600, 3200])),
                           default=default_sizes(prep["nodes"]))
    bgens = c2.multiselect("Generators", list(LABELS), default=list(res.config.generators), format_func=lambda k: LABELS[k],
                           key="bench_gens")
    b1, b2, b3 = st.columns(3)
    bi = b1.number_input("NetGAN iterations (benchmark)", 10, 5000, 100, 10)
    bs = b2.select_slider("SyNGLER fit effort (benchmark)", list(PRESETS), "Fast")
    bm = b3.number_input("MLP steps (benchmark)", 100, 10000, 500, 100)
    if st.button("Run scalability benchmark", disabled=not sizes or not bgens):
        pbar = st.progress(0.0, text="Benchmarking…")
        df = scaling_benchmark(G, sorted(sizes), bgens, dict(
            r=int(r_dim), n_iter_init=PRESETS[bs][0], n_iter_refine=PRESETS[bs][1], mlp_steps=int(bm),
            netgan_iters=int(bi), netgan_max_seconds=600.0, netgan_rw_len=int(gan_len),
            forest_n_t=int(forest_nt), forest_dup=int(forest_dup)), seed=int(seed),
            progress=lambda msg, f: pbar.progress(min(f, 1.0), text=msg))
        st.session_state["bench"] = df
        pbar.empty()
    df = st.session_state.get("bench")
    if df is not None:
        if df["error"].notna().any():
            st.warning("Some runs failed: " + "; ".join(sorted(set(df["error"].dropna()))))
        cc = {k: plots.COLORS[k] for k in plots.COLORS}
        c1, c2 = st.columns(2)
        c1.plotly_chart(plots.scaling_fig(df, "fit_s", "Fit / training time vs network size", cc), width="stretch")
        c2.plotly_chart(plots.scaling_fig(df, "gen_s", "Generation time per graph vs network size", cc), width="stretch")
        st.plotly_chart(plots.scaling_fig(df, "total_s", "Total time (fit + generate) vs network size", cc), width="stretch")
        ex = scaling_exponents(df)
        if len(ex):
            st.markdown("**Empirical scaling exponent** (time ∝ nᵇ, log–log fit; needs ≥ 3 sizes)")
            st.dataframe(ex[["generator", "exponent"]].set_index("generator").style.format("{:.2f}"), width="stretch")
        st.dataframe(df.drop(columns=["key"]).style.format({"fit_s": "{:.3f}", "gen_s": "{:.3f}", "total_s": "{:.3f}"},
                                                           na_rep="-"), width="stretch")
    else:
        st.info("Run the benchmark to see timing curves.")
    st.markdown("**Timing of the main experiment**")
    st.dataframe(rp.timing_table(res).style.format("{:.3f}"), width="stretch")

# ---- Diagnostics & export
with tabs[6]:
    if "netgan" in ok:
        h = ok["netgan"].result.info["history"]
        st.plotly_chart(plots.training_fig({"NetGAN": h}), width="stretch")
        if h.get("stopped_early_at"):
            st.caption(f"Training stopped at the time cap after {h['stopped_early_at']} iterations.")
        st.caption(f"NetGAN device: {ok['netgan'].result.info['device']}. Walk validity = share of generated "
                   f"steps that follow a real edge (chance ≈ density = {ov.scalars['density']:.4f}).")
    for k in ("syng_r", "syng_d_mlp", "syng_d_forest"):
        if k in ok:
            st.plotly_chart(plots.lsm_loss_fig(ok[k].result.info), width="stretch")
            st.caption(f"Fitted ρ = {ok[k].result.info['rho']:.3f}, latent dimension r = {ok[k].result.info['r']}.")
            break
    if "cm" in ok:
        st.caption(f"Configuration model: {ok['cm'].result.info['edge_loss_from_simplification']:.1%} of edges "
                   "lost when collapsing multi-edges/self-loops.")
    st.markdown("#### Download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, f in [("scalar_metrics", rp.scalar_table), ("metric_errors", rp.error_table),
                        ("fidelity", rp.fidelity_table), ("epidemic", rp.sir_table), ("timing", rp.timing_table),
                        ("structure_vs_function", rp.structure_function_points)]:
            z.writestr(f"{name}.csv", f(res).to_csv())
        for k, r in ok.items():
            for i, H in enumerate(r.result.graphs):
                z.writestr(f"synthetic/{k}/graph_{i}.edgelist",
                           "\n".join(f"{u} {v}" for u, v in H.edges()))
    st.download_button("⬇ Metrics (CSV) + synthetic edge lists (ZIP)", buf.getvalue(), "synth_net_results.zip",
                       mime="application/zip")
