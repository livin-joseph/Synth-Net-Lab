"""Synth-Net Lab - when is a synthetic network a safe substitute?  (Streamlit dashboard)

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from netsynth import io as nio
from netsynth import netviz, plots
from netsynth import reporting as rp
from netsynth.benchmark import default_sizes, scaling_benchmark, scaling_exponents
from netsynth.bundle import bundle_from_files, bundle_from_result, bundle_from_zip, bundle_to_zip
from netsynth.pipeline import ExperimentConfig, generator_labels, run_experiment
from netsynth.structure import pooled_profile, summarize

st.set_page_config(page_title="Synth-Net Lab", page_icon="🕸️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1300px}
div[data-testid="stMetric"]{background:#f1f0ed;border-radius:10px;padding:10px 14px}
.ok{color:#0a7d3b;font-weight:600}.bad{color:#b3261e;font-weight:600}
.card{border:1px solid #e3e1dc;border-radius:12px;padding:12px 14px;background:#fff;height:100%}
.card h4{margin:0 0 6px 0;font-size:1rem}.small{color:#6b6b66;font-size:.85rem}
</style>""", unsafe_allow_html=True)

LABELS = generator_labels()
EXAMPLES_DIR = Path(__file__).parent / "examples"
DEFAULT_GENS = ["er", "ba", "cm", "syng_r"]


@st.cache_data(show_spinner=False)
def prepare(raw: bytes, filename: str, max_nodes: int):
    G0, notes = nio.load_graph(raw, filename)
    G, rep = nio.prepare_graph(G0, take_lcc=True, max_nodes=max_nodes or None)
    rep["notes"] = notes + rep["notes"]
    return G, rep


@st.cache_data(show_spinner=False)
def sample_bytes(ref: str):
    if ref.startswith("nx:"):
        G = nio.load_sample(ref)
        return "\n".join(f"{u} {v}" for u, v in G.edges()).encode(), ref.replace("nx:", "") + ".txt"
    p = Path(ref)
    return p.read_bytes(), p.name


def table(df, fmt="{:.3f}", per_col=None):
    num = df.apply(pd.to_numeric, errors="coerce")
    return num.style.format({c: (per_col or {}).get(c, fmt) for c in num.columns}, na_rep="n/a")


# =========================================================================== sidebar
with st.sidebar:
    st.title("🕸️ Synth-Net Lab")
    st.caption("Is a synthetic network a safe substitute for simulation or data sharing?")
    mode = st.radio("Mode", ["Run an experiment", "Open saved results"])
    raw = fname = None
    run_clicked = False
    if mode == "Run an experiment":
        src = st.radio("Real network", ["Example", "Upload file"], horizontal=True)
        if src == "Example":
            samples = nio.sample_networks()
            choice = st.selectbox("Example", list(samples), index=1)
            raw, fname = sample_bytes(samples[choice][0])
        else:
            up = st.file_uploader("Network file", help="Edge list, adjacency matrix, GraphML, GML, GEXF, Pajek, .mtx, .npz (also .gz).")
            if up is not None:
                raw, fname = up.getvalue(), up.name
        keys = st.multiselect("Generators", list(LABELS), default=DEFAULT_GENS, format_func=lambda k: LABELS[k])
        K = st.slider("Synthetic graphs per generator", 1, 20, 5)
        with st.expander("Advanced settings"):
            seed = st.number_input("Random seed", 0, 10_000, 0)
            max_nodes = st.number_input("Max nodes (larger networks are sampled down)", 0, 20000, 1500, 100)
            r_dim = st.slider("SyNGLER latent dimension", 1, 16, 6)
            leak_target = st.slider("SyNG-R: copied-edge target", 0.0, 0.6, 0.25, 0.05,
                                    help="Plain SyNGLER can copy most real edges on small graphs. SyNG-R adds just enough noise to its "
                                         "resampled latents to keep real edges copied beyond chance at or below this share.")
            gan_iters = st.number_input("NetGAN training iterations", 20, 5000, 600, 20)
            beta_mult = st.slider("Epidemic strength β / β_c", 1.0, 6.0, 2.0, 0.1,
                                  help="Transmission rate as a multiple of the real network's epidemic threshold.")
            runs = st.slider("Epidemic simulations per graph", 5, 100, 40)
        run_clicked = st.button("▶ Run experiment", type="primary", width="stretch", disabled=(raw is None or not keys))
    else:
        ex = {p.stem.replace("_", " "): p for p in sorted(EXAMPLES_DIR.glob("*.zip"))}
        how = st.radio("Source", ["Bundled run", "Upload"], horizontal=True)
        loaded = None
        if how == "Bundled run" and ex:
            pick = st.selectbox("Run", list(ex))
            if st.button("Open", type="primary", width="stretch"):
                loaded = (ex[pick].read_bytes(), None, pick)
        elif how == "Upload":
            ups = st.file_uploader("Results ZIP (or exported CSV/JSON files)", accept_multiple_files=True, type=["zip", "csv", "json", "edgelist", "txt"])
            if ups and st.button("Open", type="primary", width="stretch"):
                loaded = (ups[0].getvalue(), None, ups[0].name) if len(ups) == 1 and ups[0].name.endswith(".zip") \
                    else (None, {u.name: u.getvalue() for u in ups}, "uploaded files")
        if loaded:
            try:
                data, files, nm = loaded
                st.session_state["bundle"] = bundle_from_zip(data, nm) if data else bundle_from_files(files, nm)
                st.session_state["origin"] = "saved"
            except Exception as e:
                st.error(f"Could not open results: {e}")

st.title("Synth-Net Lab")

# =========================================================================== run / preview
if mode == "Run an experiment":
    if raw is None:
        st.info("Pick an example or upload a network in the sidebar.")
        st.stop()
    try:
        G, prep = prepare(raw, fname, int(max_nodes))
    except Exception as e:
        st.error(f"Could not read the network: {e}")
        st.stop()
    if run_clicked:
        bar = st.progress(0.0, text="Starting…")
        cfg = ExperimentConfig(generators=keys, k=int(K), seed=int(seed),
                               gen_params=dict(r=int(r_dim), leak_target=float(leak_target), netgan_iters=int(gan_iters)),
                               beta_mult=float(beta_mult), runs_per_graph=int(runs))
        res = run_experiment(G, cfg, progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
        st.session_state["bundle"] = bundle_from_result(res, Path(fname).name.split(".")[0])
        st.session_state["origin"] = "run"
        st.session_state["run_for"] = (fname, int(max_nodes))
        st.session_state.pop("bench", None)
        bar.empty()

b = st.session_state.get("bundle")
origin = st.session_state.get("origin")
if mode == "Run an experiment" and origin == "saved":
    b = None
if b is None:
    if mode == "Run an experiment":
        ov = summarize(G)
        c = st.columns(4)
        c[0].metric("Nodes", f"{prep['nodes']:,}"); c[1].metric("Edges", f"{prep['edges']:,}")
        c[2].metric("Mean degree", f"{ov.scalars['mean_degree']:.2f}"); c[3].metric("Avg clustering", f"{ov.scalars['avg_clustering']:.3f}")
        if prep["notes"]:
            st.caption(" ".join(prep["notes"]))
        st.info("Choose generators in the sidebar and press **Run experiment**.")
    else:
        st.info("Open a bundled run or upload a results ZIP in the sidebar.")
    st.stop()

meta = b.meta
if origin == "saved":
    st.success(f"Showing saved results: **{meta.get('name')}**" + ("" if b.has_graphs else " (no graphs stored: network views are unavailable)"))
elif mode == "Run an experiment" and st.session_state.get("run_for") != (fname, int(max_nodes)):
    st.warning("These results belong to a different network or setting. Press Run to refresh.")
for k, err in b.failed.items():
    st.error(f"{k} failed: {err}. The other generators are unaffected.")
if not b.generators:
    st.stop()

c = st.columns(4)
c[0].metric("Nodes", meta.get("n", "–")); c[1].metric("Edges", meta.get("m", "–"))
rs = meta.get("real_scalars") or {}
c[2].metric("Mean degree", f"{rs['mean_degree']:.2f}" if rs else "–"); c[3].metric("Avg clustering", f"{rs['avg_clustering']:.3f}" if rs else "–")

cm = plots.color_map(b)
labels = [g["label"] for g in b.generators]
tok = f"{meta.get('name')}|{meta.get('created')}|{len(b.graphs)}"
tabs = st.tabs(["Overview", "Networks", "Structure", "Epidemics", "Privacy & cost", "Data"])

# =========================================================================== Overview
with tabs[0]:
    with st.expander("Tolerances (what counts as 'close enough')"):
        a, c2, c3 = st.columns(3)
        tol = rp.Tolerances(
            degree_ks=a.slider("Degree-distribution distance (KS) ≤", 0.05, 0.6, 0.25, 0.01),
            final_size_abs=a.slider("Final-size error ≤", 0.01, 0.3, 0.05, 0.01),
            curve_nrmse=c2.slider("Outbreak-curve error ≤", 0.05, 1.0, 0.20, 0.01),
            peak_rel=c2.slider("Peak error (relative) ≤", 0.05, 1.0, 0.25, 0.05),
            outbreak_prob_abs=c3.slider("Outbreak-probability error ≤", 0.02, 0.5, 0.15, 0.01),
            max_leak_excess=c3.slider("Real edges copied, beyond chance ≤", 0.0, 1.0, 0.25, 0.01,
                                      help="0 = no more real edges than a random graph would contain; 1 = every edge is a real edge."))
        st.caption("Epidemic tolerances are never applied tighter than the real network's own run-to-run noise "
                   f"(curve error ≈ {b.floor.get('curve_nrmse', float('nan')):.2f}).")
    v = rp.verdicts(b, tol)
    sr = b.diagnostics.get("syng_r", {})
    if sr.get("latent_noise") is not None:
        st.caption(f"SyNG-R privacy noise was auto-tuned to {sr['latent_noise']:.2f}"
                   + (f" (target: at most {sr['leak_target']:.0%} of edges copied beyond chance; measured on a probe graph: {sr['leak_probe']:.0%})."
                      if sr.get("leak_target") is not None and sr.get("leak_probe") is not None else "."))
    st.markdown("#### Is it a safe substitute?")
    cols = st.columns(min(4, len(v)))
    for i, (g, r) in enumerate(v.iterrows()):
        with cols[i % len(cols)]:
            sim = '<span class=ok>✔ yes</span>' if r.simulation_ok else '<span class=bad>✘ no</span>'
            shr = '<span class=ok>✔ yes</span>' if r.sharing_ok else '<span class=bad>✘ no</span>'
            leak = "" if r.leak is None or pd.isna(r.leak) else f"<div class=small>{r.leak:.0%} of its edges are real edges</div>"
            why = f"<div class=small>Fails: {'; '.join(r.failed)}</div>" if r.failed else ""
            st.markdown(f"<div class=card style='border-top:4px solid {cm[g]}'><h4>{g}</h4>For simulation: {sim}<br>"
                        f"For data sharing: {shr}{leak}{why}</div>", unsafe_allow_html=True)
            st.write("")
    st.markdown("#### Summary")
    summ = pd.DataFrame({
        "Degree distance (KS)": b.fidelity["Degree KS"], "Portrait divergence": b.fidelity["Portrait divergence"],
        "Final-size error": b.epidemic["Δ final size"], "Outbreak-curve error": b.epidemic["Curve NRMSE"],
        "Real edges copied": b.fidelity["Edge overlap with real"], "Fit time (s)": b.timing["Fit / training (s)"] if not b.timing.empty else None,
    }).loc[labels]
    st.dataframe(table(summ, per_col={"Real edges copied": "{:.0%}"}), width="stretch")
    st.markdown("#### Key findings")
    for t in rp.takeaways(b, tol):
        st.markdown("- " + t)

# =========================================================================== Networks
with tabs[1]:
    if not b.has_graphs:
        st.info("This results file has no graphs.")
    else:
        st.caption("Real network (left) and a synthetic graph (right). Drag, zoom, hover. Graphs above the node limit are shown as a snowball sample.")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        gsel = c1.selectbox("Generator", [g for g in labels if b.key_of(g) in b.graphs])
        gkey = b.key_of(gsel)
        gi = c2.number_input("Graph #", 0, len(b.graphs[gkey]) - 1, 0)
        maxn = c3.slider("Max nodes shown", 40, 300, 200, 10)
        color_by = c4.selectbox("Colour by", netviz.COLOR_MODES)
        kind = meta.get("node_map_kind", {}).get(gkey, "none")
        copy_hl = st.checkbox("Highlight edges copied from the real network (red)", value=kind != "none", disabled=kind == "none")

        @st.cache_data(show_spinner="Drawing…", max_entries=8)
        def views(_tok, key, gi, maxn, color_by, copy_hl):
            Rs, r_s = netviz.viz_subgraph(b.real, maxn, 0)
            H = b.graphs[key][gi]
            same = meta.get("node_map_kind", {}).get(key) == "identity" and r_s
            Hs, h_s = netviz.viz_subgraph(H, maxn, 0, nodes=list(Rs.nodes())) if same else netviz.viz_subgraph(H, maxn, 0)
            is_copy = netviz.real_edge_set_fn(b.real, b.node_maps[key][gi]) if copy_hl else None
            hr, lr, _ = netviz.network_html(Rs, color_by)
            hs, ls, n_copy = netviz.network_html(Hs, color_by, is_copy)
            return hr, lr, Rs, r_s, hs, ls, n_copy, Hs, h_s

        hr, lr, Rs, r_s, hs, ls, n_copy, Hs, h_s = views(tok, gkey, int(gi), int(maxn), color_by, bool(copy_hl))
        l, r = st.columns(2)
        with l:
            st.markdown(f"**Real** — {Rs.number_of_nodes()} nodes, {Rs.number_of_edges()} edges" + (" (sample)" if r_s else ""))
            st.caption(f"{lr} · clustering {nx.average_clustering(Rs):.2f}")
            st.iframe(hr, height=540)
        with r:
            st.markdown(f"**{gsel}** — {Hs.number_of_nodes()} nodes, {Hs.number_of_edges()} edges" + (" (sample)" if h_s else ""))
            st.caption(f"{ls} · clustering {nx.average_clustering(Hs):.2f}" + (f" · {n_copy} red edges are real ({n_copy / max(Hs.number_of_edges(), 1):.0%})" if copy_hl else ""))
            st.iframe(hs, height=540)

# =========================================================================== Structure
with tabs[2]:
    st.plotly_chart(plots.scalar_small_multiples(b), width="stretch")
    st.caption("Bars: mean over synthetic graphs (whiskers: s.d.). Dashed line: real network.")
    if b.fidelity["Portrait divergence"].notna().any():
        st.plotly_chart(plots.pd_figure(b), width="stretch")
        st.caption("Portrait divergence: one number comparing the networks' shortest-path structure (0 = identical). "
                   "Reported for reference; it is not used in the verdict because it did not predict epidemic error.")
    if b.has_graphs:
        cache = st.session_state.setdefault("_prof", {})
        if tok not in cache:
            with st.spinner("Computing distributions…"):
                profs = {"Real": pooled_profile([b.real])}
                for g in b.generators:
                    if b.graphs.get(g["key"]):
                        profs[g["label"]] = pooled_profile(b.graphs[g["key"]][:3])
                cache[tok] = profs
        c1, c2 = st.columns(2)
        c1.plotly_chart(plots.degree_ccdf_fig(cache[tok], cm), width="stretch")
        c2.plotly_chart(plots.clustering_box(cache[tok], cm), width="stretch")

# =========================================================================== Epidemics
with tabs[3]:
    if b.curves is None:
        st.info("This results file has no epidemic curves.")
    else:
        st.plotly_chart(plots.epidemic_curves(b, "I", False, labels), width="stretch")
        st.caption(f"SIR with β = {meta.get('beta', float('nan')):.3f} ({meta.get('beta_mult', float('nan')):.1f} × the real epidemic threshold), "
                   f"γ = {meta.get('gamma')}. Line: mean of the runs; band: 10–90 percentile.")
        st.plotly_chart(plots.epidemic_bars(b), width="stretch")
        st.plotly_chart(plots.epidemic_errors(b, tol), width="stretch")
        st.caption("Dotted grey: error of the real network against itself (simulation noise). Dashed green: your tolerance. "
                   "A bar under the grey line cannot be told apart from noise.")

# =========================================================================== Privacy & cost
with tabs[4]:
    ch = float(b.density) if b.density == b.density else 0.0
    st.plotly_chart(plots.leakage_bars(b, ch + tol.max_leak_excess * (1 - ch)), width="stretch")
    st.caption("Share of a synthetic graph's edges that are also edges of the real graph (via the node correspondence). Dotted: what a random graph "
               "would share. Dashed: your tolerance. A leakage proxy, not a formal privacy guarantee. "
               + ("Not measurable for: " + ", ".join(g for g in labels if pd.isna(b.fidelity.loc[g, 'Edge overlap with real'])) + "." if b.fidelity["Edge overlap with real"].isna().any() else ""))
    if not b.timing.empty:
        st.plotly_chart(plots.timing_fig(b), width="stretch")
    if b.real is not None:
        with st.expander("Scalability: time vs network size"):
            n_real = b.real.number_of_nodes()
            sizes = st.multiselect("Sizes", sorted(set(default_sizes(n_real) + [50, 100, 200, 400, 800, 1600])), default=default_sizes(n_real))
            bg = st.multiselect("Generators", list(LABELS), default=[k for k in b.keys() if k in LABELS], format_func=lambda k: LABELS[k], key="bg")
            if st.button("Run benchmark", disabled=not sizes or not bg):
                pbar = st.progress(0.0)
                st.session_state["bench"] = scaling_benchmark(b.real, sorted(sizes), bg, dict(
                    r=6, n_iter_init=150, n_iter_refine=250, mlp_steps=500, netgan_iters=100, netgan_max_seconds=600.0, forest_n_t=10, forest_dup=10),
                    seed=0, progress=lambda m, f: pbar.progress(min(f, 1.0), text=m))
                pbar.empty()
            df = st.session_state.get("bench")
            if df is not None:
                ck = {k: plots.COLORS.get(k, "#888") for k in LABELS}
                c1, c2 = st.columns(2)
                c1.plotly_chart(plots.scaling_fig(df, "fit_s", "Fit / training time", ck), width="stretch")
                c2.plotly_chart(plots.scaling_fig(df, "gen_s", "Generation time per graph", ck), width="stretch")
                ex = scaling_exponents(df)
                if len(ex):
                    st.dataframe(ex[["generator", "exponent"]].set_index("generator").style.format("{:.2f}"), width="stretch")

# =========================================================================== Data
with tabs[5]:
    tables = {"scalar_metrics": b.scalar, "metric_errors": b.errors, "fidelity": b.fidelity, "epidemic": b.epidemic, "timing": b.timing,
              "structure_vs_function": b.sf, "per_graph_metrics": b.per_graph, "epidemic_runs": b.runs, "epidemic_curves": b.curves}
    tables = {k: v for k, v in tables.items() if v is not None and len(v)}
    name = st.selectbox("Table", list(tables))
    st.dataframe(tables[name], width="stretch", height=340)
    idx = name in ("scalar_metrics", "metric_errors", "fidelity", "epidemic", "timing")
    st.download_button(f"⬇ {name}.csv", tables[name].to_csv(index=idx).encode(), f"{name}.csv", mime="text/csv")
    st.download_button("⬇ Everything (ZIP: tables, curves, graphs)", bundle_to_zip(b, include_graphs=b.has_graphs),
                       f"synth_net_{meta.get('name', 'run')}.zip", mime="application/zip")
    st.caption("Reopen this ZIP any time with *Open saved results*.")
