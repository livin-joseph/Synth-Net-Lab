"""Plotly figure builders for the dashboard (fixed colour per entity, real = neutral ink)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REAL = "#2b2b2b"
COLORS = {  # validated categorical slots, assigned per generator and never re-ordered
    "syng_r": "#2a78d6", "er": "#eb6834", "ba": "#1baf7a", "cm": "#e87ba4",
    "netgan": "#6250d6", "syng_d_mlp": "#e34948", "syng_d_forest": "#eda100",
}
GRID = "rgba(120,120,120,0.18)"


def _layout(fig, title=None, height=380, xt=None, yt=None, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None, height=height,
        margin=dict(l=50, r=20, t=50 if title else 20, b=45), template="plotly_white",
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


def degree_ccdf_fig(res, ok):
    fig = go.Figure()
    u, p = ccdf(res.real_summary.degrees)
    fig.add_scatter(x=u, y=p, mode="lines", name="Real", line=dict(color=REAL, width=3))
    for k, r in ok.items():
        u, p = ccdf(np.concatenate([s.degrees for s in r.summaries]))
        fig.add_scatter(x=u, y=p, mode="lines", name=r.label, line=dict(color=COLORS[k], width=2))
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return _layout(fig, "Degree distribution (complementary CDF, log–log)", xt="degree k", yt="P(K ≥ k)")


def clustering_fig(res, ok):
    fig = go.Figure()
    fig.add_box(y=res.real_summary.clustering, name="Real", marker_color=REAL, boxpoints=False)
    for k, r in ok.items():
        fig.add_box(y=np.concatenate([s.clustering for s in r.summaries]), name=r.label,
                    marker_color=COLORS[k], boxpoints=False)
    return _layout(fig, "Local clustering coefficient per node", yt="C_i", legend=False)


def error_heatmap(err: pd.DataFrame):
    z = err.values
    txt = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=np.minimum(z, 1.0), x=list(err.columns), y=list(err.index), text=txt,
                               texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
                               colorbar=dict(title="error<br>(capped at 1)"),
                               hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Error of each structural metric vs the real network (lower is better)",
                   height=120 + 55 * len(err), legend=False)


def pd_bar(fid: pd.DataFrame):
    fig = go.Figure()
    for name, row in fid.sort_values("Portrait divergence").iterrows():
        fig.add_bar(x=[name], y=[row["Portrait divergence"]], name=name, marker_color=COLORS[row["key"]],
                    error_y=dict(type="data", array=[row["PD std"]], color="#555"),
                    text=[f"{row['Portrait divergence']:.3f}"], textposition="outside")
    fig.update_yaxes(rangemode="tozero")
    return _layout(fig, "Portrait divergence to the real network (0 = identical, 1 = maximally different)",
                   yt="D_JS (mean ± sd over samples)", legend=False)


def portrait_heatmaps(B_real, B_syn, label):
    def prep(B):
        K = B.shape[1]
        return np.log10(1 + B * np.arange(K)[None, :])
    L = max(B_real.shape[0], B_syn.shape[0]); K = max(B_real.shape[1], B_syn.shape[1])
    def pad(B):
        out = np.zeros((L, K)); out[:B.shape[0], :B.shape[1]] = B; return out
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Real", label), shared_yaxes=True)
    zr, zs = prep(pad(B_real)), prep(pad(B_syn))
    zmax = max(zr.max(), zs.max())
    for c, z in enumerate((zr, zs), start=1):
        fig.add_trace(go.Heatmap(z=z, zmin=0, zmax=zmax, colorscale="Viridis", showscale=(c == 2),
                                 colorbar=dict(title="log₁₀(1+k·B)")), row=1, col=c)
    fig.update_xaxes(title="k = nodes at distance ℓ", type="log" if K > 50 else "linear")
    fig.update_yaxes(title="distance ℓ", row=1, col=1)
    fig.update_layout(height=380, margin=dict(l=50, r=20, t=50, b=45), template="plotly_white")
    return fig


def epidemic_curves(res, ok, band=(10, 90)):
    g = res.grid
    fig = go.Figure()
    def add(name, I, color, width, band_on=True):
        m = I.mean(0)
        if band_on:
            lo, hi = np.percentile(I, band[0], axis=0), np.percentile(I, band[1], axis=0)
            fig.add_scatter(x=np.r_[g, g[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="lines",
                            line=dict(width=0), fillcolor=_rgba(color, 0.12), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=g, y=m, mode="lines", name=name, line=dict(color=color, width=width))
    add("Real", res.real_sir["I"], REAL, 3.5)
    for k, r in ok.items():
        add(r.label, r.sir_pooled["I"], COLORS[k], 2)
    return _layout(fig, f"Infected fraction I(t): mean and {band[0]}–{band[1]}th percentile band across runs",
                   xt="time", yt="fraction infected")


def final_size_fig(res, ok):
    fig = go.Figure()
    fig.add_box(y=res.real_sir["final"], name="Real", marker_color=REAL, boxmean=True)
    for k, r in ok.items():
        fig.add_box(y=r.sir_pooled["final"], name=r.label, marker_color=COLORS[k], boxmean=True)
    return _layout(fig, "Final outbreak size (fraction ever infected)", yt="final size", legend=False)


def structure_function_fig(pts: pd.DataFrame, xcol: str, ycol: str, color_of: dict):
    fig = go.Figure()
    for gname, d in pts.groupby("Generator", sort=False):
        fig.add_scatter(x=d[xcol], y=d[ycol], mode="markers", name=gname,
                        marker=dict(size=11, color=color_of[gname], line=dict(color="white", width=1.5)))
    return _layout(fig, f"{ycol} vs {xcol} (one point per synthetic graph)", xt=xcol, yt=ycol)


def scaling_fig(df: pd.DataFrame, col: str, title: str, color_of: dict):
    fig = go.Figure()
    for key, d in df.dropna(subset=[col]).groupby("key", sort=False):
        d = d.sort_values("size")
        fig.add_scatter(x=d["size"], y=d[col].clip(lower=1e-4), mode="lines+markers", name=d["generator"].iloc[0],
                        line=dict(color=color_of[key], width=2), marker=dict(size=9, line=dict(color="white", width=1.5)))
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return _layout(fig, title, xt="nodes n", yt="seconds")


def training_fig(info_by_label: dict):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("NetGAN critic loss", "NetGAN walk validity (share of steps on real edges)"))
    for label, h in info_by_label.items():
        fig.add_scatter(x=h["iter"], y=h["critic_loss"], mode="lines", name="critic loss",
                        line=dict(color=COLORS["netgan"], width=1.5), row=1, col=1, showlegend=False)
        fig.add_scatter(x=h["val_iter"], y=h["walk_validity"], mode="lines+markers", name="validity",
                        line=dict(color=COLORS["netgan"], width=2), row=1, col=2, showlegend=False)
    fig.update_layout(height=320, margin=dict(l=50, r=20, t=50, b=45), template="plotly_white")
    fig.update_xaxes(title="iteration"); fig.update_yaxes(rangemode="tozero", row=1, col=2)
    return fig


def lsm_loss_fig(info):
    fig = go.Figure()
    li, lr = info.get("loss_init", []), info.get("loss_refine", [])
    fig.add_scatter(y=li, x=list(range(len(li))), name="Phase 1 (Gram matrix G)", line=dict(color=COLORS["syng_r"]))
    fig.add_scatter(y=lr, x=[len(li) + i for i in range(len(lr))], name="Phase 2 (Z, α)", line=dict(color=COLORS["syng_d_mlp"]))
    fig.update_yaxes(type="log")
    return _layout(fig, "SyNGLER stage 1: LSM fitting loss ‖A − P‖²", xt="PGD iteration", yt="loss")
