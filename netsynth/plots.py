"""Plotly figure builders. Every function reads a ResultBundle (or DataFrames from it).

Colours follow the entity (one fixed colour per generator, real network = dark neutral), never its rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from netsynth.structure import SCALAR_METRICS, _logbin

REAL = "#2b2b2b"
COLORS = {
    "syng_r": "#2a78d6", "er": "#eb6834", "ba": "#1baf7a", "cm": "#e87ba4",
    "netgan": "#6250d6", "syng_d_mlp": "#e34948", "syng_d_forest": "#eda100",
}
GRID = "rgba(120,120,120,0.18)"
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#e34948", "#eda100", "#008300"]


def color_map(b) -> dict:
    """display label -> colour (plus 'Real')."""
    m = {"Real": REAL}
    for i, g in enumerate(b.generators):
        m[g["label"]] = COLORS.get(g["key"], FALLBACK[i % len(FALLBACK)])
    return m


def _layout(fig, title=None, height=380, xt=None, yt=None, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None, height=height,
        margin=dict(l=55, r=20, t=55 if title else 20, b=50), template="plotly_white",
        legend=dict(orientation="h", y=-0.22, x=0) if legend else None, showlegend=legend,
        font=dict(size=12), hovermode="closest")
    fig.update_xaxes(title=xt, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=yt, gridcolor=GRID, zeroline=False)
    return fig


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def ccdf(x):
    x = np.sort(np.asarray(x))
    u, c = np.unique(x, return_counts=True)
    return u, 1.0 - (np.cumsum(c) - c) / len(x)


def empty_fig(msg: str, height=200):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#777"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(height=height, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ----------------------------------------------------------------------------- overview




# ----------------------------------------------------------------------------- structure
def scalar_small_multiples(b):
    keys = ["m", "mean_degree", "degree_second_moment", "avg_clustering", "transitivity", "assortativity",
            "aspl", "diameter", "lcc_fraction"]
    titles = {"m": "Edges", **{k: v[0] for k, v in SCALAR_METRICS.items()}}
    real = b.meta.get("real_scalars") or {}
    cm = color_map(b)
    fig = make_subplots(rows=3, cols=3, subplot_titles=[titles[k] for k in keys], vertical_spacing=0.17, horizontal_spacing=0.07)
    pg = b.per_graph
    for i, k in enumerate(keys):
        r, c = divmod(i, 3)
        for g in b.generators:
            lab = g["label"]
            if pg is not None and k in pg:
                v = pg.loc[pg["key"] == g["key"], k]
                mean, sd = float(v.mean()), float(v.std(ddof=0))
            else:
                col = "edges (m)" if k == "m" else SCALAR_METRICS[k][0]
                mean, sd = float(b.scalar.loc[lab, col]), 0.0
            fig.add_bar(x=[lab], y=[mean], error_y=dict(type="data", array=[sd], color="#555", thickness=1),
                        marker_color=cm[lab], showlegend=False, row=r + 1, col=c + 1,
                        hovertemplate=f"{lab}<br>{titles[k]}: %{{y:.3f}}<extra></extra>")
        rv = real.get(k, b.scalar.loc["Real", "edges (m)" if k == "m" else SCALAR_METRICS[k][0]] if "Real" in b.scalar.index else None)
        if rv is not None:
            fig.add_hline(y=float(rv), line=dict(color=REAL, dash="dash", width=2), row=r + 1, col=c + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=620, template="plotly_white", margin=dict(l=45, r=15, t=50, b=15), bargap=0.25)
    fig.update_annotations(font_size=12)
    return fig








def degree_ccdf_fig(profiles: dict, cm: dict):
    fig = go.Figure()
    for name, p in profiles.items():
        u, s = ccdf(p["deg"])
        fig.add_scatter(x=u, y=s, mode="lines", name=name,
                        line=dict(color=cm.get(name, REAL), width=3.5 if name == "Real" else 2))
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return _layout(fig, "Degree distribution (CCDF, log–log)", xt="degree k", yt="P(K ≥ k)")










def clustering_box(profiles, cm):
    fig = go.Figure()
    for name, p in profiles.items():
        fig.add_box(y=p["clu"], name=name, marker_color=cm.get(name, REAL), boxpoints=False)
    return _layout(fig, "Local clustering coefficient per node", yt="C_i", legend=False)


# ----------------------------------------------------------------------------- portrait divergence
def pd_figure(b):
    cm = color_map(b)
    fig = go.Figure()
    fid = b.fidelity.sort_values("Portrait divergence")
    pg = b.per_graph
    for lab, row in fid.iterrows():
        fig.add_bar(x=[lab], y=[row["Portrait divergence"]], marker_color=cm.get(lab, "#888"), showlegend=False,
                    error_y=dict(type="data", array=[row.get("PD std", 0)], color="#555"),
                    text=[f"{row['Portrait divergence']:.3f}"], textposition="outside", opacity=0.55)
        if pg is not None:
            v = pg.loc[pg["generator"] == lab, "portrait_divergence"]
            fig.add_scatter(x=[lab] * len(v), y=v, mode="markers", showlegend=False,
                            marker=dict(size=9, color=cm.get(lab, "#888"), line=dict(color="white", width=1.5)),
                            hovertemplate="graph %{pointNumber}: %{y:.3f}<extra>" + lab + "</extra>")
    fig.update_yaxes(rangemode="tozero")
    return _layout(fig, "Portrait divergence to the real network (bars: mean ± sd, dots: individual graphs)",
                   yt="D_JS (0 = identical, 1 = maximally different)", legend=False)






# ----------------------------------------------------------------------------- epidemics
def epidemic_curves(b, which="I", log=False, show=None):
    cm = color_map(b)
    df = b.curves
    fig = go.Figure()
    nets = ["Real"] + [g["label"] for g in b.generators]
    for name in nets:
        if show is not None and name not in show and name != "Real":
            continue
        d = df[df["network"] == name]
        if d.empty:
            continue
        col = cm[name]
        fig.add_scatter(x=np.r_[d["time"], d["time"][::-1]], y=np.r_[d[f"{which}_p90"], d[f"{which}_p10"][::-1]], fill="toself",
                        mode="lines", line=dict(width=0), fillcolor=_rgba(col, 0.12), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=d["time"], y=d[f"{which}_mean"], mode="lines", name=name, line=dict(color=col, width=3.5 if name == "Real" else 2))
    if log:
        fig.update_yaxes(type="log")
    t = "Infected fraction I(t)" if which == "I" else "Cumulative recovered fraction R(t)"
    return _layout(fig, f"{t}: mean with 10–90 percentile band", xt="time", yt="fraction of nodes")


def epidemic_bars(b):
    cols = [("Final size", "final_size"), ("Peak prevalence", "peak_prevalence"), ("Time to peak", "time_to_peak"), ("Outbreak prob.", None)]
    cm = color_map(b)
    fig = make_subplots(rows=1, cols=4, subplot_titles=[c[0] for c in cols], horizontal_spacing=0.06)
    for i, (title, rc) in enumerate(cols, start=1):
        for g in b.generators:
            lab = g["label"]
            if lab not in b.epidemic.index:
                continue
            v = float(b.epidemic.loc[lab, title])
            sd = 0.0
            if b.runs is not None and rc:
                sd = float(b.runs.loc[b.runs["network"] == lab, rc].std(ddof=0))
            fig.add_bar(x=[lab], y=[v], marker_color=cm[lab], showlegend=False, row=1, col=i,
                        error_y=dict(type="data", array=[sd], color="#555", thickness=1) if sd else None,
                        hovertemplate=f"{lab}<br>{title}: %{{y:.3f}}<extra></extra>")
        if "Real" in b.epidemic.index:
            fig.add_hline(y=float(b.epidemic.loc["Real", title]), line=dict(color=REAL, dash="dash", width=2), row=1, col=i)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=320, template="plotly_white", margin=dict(l=45, r=15, t=45, b=15), bargap=0.25)
    return fig


def epidemic_errors(b, tol=None):
    items = [("Δ final size", "final_size_err", "final_size_abs"), ("Peak err (rel.)", "peak_err_rel", "peak_rel"),
             ("Curve NRMSE", "curve_nrmse", "curve_nrmse"), ("Outbreak-prob err", "outbreak_prob_err", "outbreak_prob_abs")]
    cm = color_map(b)
    fig = make_subplots(rows=1, cols=4, subplot_titles=[i[0] for i in items], horizontal_spacing=0.07)
    for i, (col, fk, tk) in enumerate(items, start=1):
        for g in b.generators:
            lab = g["label"]
            if lab in b.epidemic.index and col in b.epidemic:
                fig.add_bar(x=[lab], y=[float(b.epidemic.loc[lab, col])], marker_color=cm[lab], showlegend=False, row=1, col=i,
                            hovertemplate=f"{lab}<br>{col}: %{{y:.3f}}<extra></extra>")
        fl = b.floor.get(fk)
        if fl is not None:
            fig.add_hline(y=fl, line=dict(color="#888", dash="dot", width=2), row=1, col=i)
        if tol is not None:
            fig.add_hline(y=getattr(tol, tk), line=dict(color="#0a7d3b", dash="dash", width=2), row=1, col=i)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=330, template="plotly_white", margin=dict(l=45, r=15, t=45, b=15), bargap=0.25)
    return fig








# ----------------------------------------------------------------------------- leakage / utility
def leakage_bars(b, tol_overlap=None):
    cm = color_map(b)
    f = b.fidelity
    fig = go.Figure()
    for lab, row in f.iterrows():
        ov = row.get("Edge overlap with real")
        if pd.isna(ov):
            continue
        fig.add_bar(x=[lab], y=[ov], marker_color=cm.get(lab, "#888"), showlegend=False, text=[f"{ov:.1%}"], textposition="outside",
                    hovertemplate=f"{lab}<br>{ov:.1%} of synthetic edges are real ({row['Overlap vs chance (×)']:.1f}× chance)<extra></extra>")
    if np.isfinite(b.density):
        fig.add_hline(y=b.density, line=dict(color="#888", dash="dot"), annotation_text="chance (= density)", annotation_position="top left")
    if tol_overlap is not None:
        fig.add_hline(y=tol_overlap, line=dict(color="#0a7d3b", dash="dash"), annotation_text="tolerance", annotation_position="top right")
    fig.update_yaxes(rangemode="tozero", tickformat=".0%")
    return _layout(fig, "Share of synthetic edges that are also real edges", yt="edge overlap", legend=False)




# ----------------------------------------------------------------------------- structure <-> function




# ----------------------------------------------------------------------------- cost
def timing_fig(b):
    t = b.timing
    cm = color_map(b)
    fig = go.Figure()
    fig.add_bar(y=list(t.index), x=t["Fit / training (s)"].clip(lower=1e-3), orientation="h", name="Fit / training",
                marker_color="#4b5563", hovertemplate="%{y}<br>fit %{x:.3f}s<extra></extra>")
    fig.add_bar(y=list(t.index), x=t["Per-graph generation (s)"].clip(lower=1e-3), orientation="h", name="Generate one graph",
                marker_color="#c7ccd4", hovertemplate="%{y}<br>generate %{x:.3f}s<extra></extra>")
    fig.update_xaxes(type="log"); fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group")
    return _layout(fig, "Cost of each generator (log scale; values below 1 ms clipped)", height=130 + 55 * len(t), xt="seconds")




def scaling_fig(df: pd.DataFrame, col: str, title: str, cm_key: dict):
    fig = go.Figure()
    for key, d in df.dropna(subset=[col]).groupby("key", sort=False):
        d = d.sort_values("size")
        fig.add_scatter(x=d["size"], y=d[col].clip(lower=1e-4), mode="lines+markers", name=d["generator"].iloc[0],
                        line=dict(color=cm_key.get(key, "#888"), width=2), marker=dict(size=9, line=dict(color="white", width=1.5)))
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return _layout(fig, title, xt="nodes n", yt="seconds")


# ----------------------------------------------------------------------------- diagnostics


