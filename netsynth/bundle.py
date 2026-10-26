"""ResultBundle: everything an experiment produces, in one object that can be saved and re-opened.

The dashboard reads *only* a bundle, so a fresh run and a previously exported ZIP (or a folder of the
exported CSV files) are displayed identically.  Files in the exported ZIP:

    meta.json                      network + experiment settings, noise floor, generators
    scalar_metrics.csv             real value and per-generator mean of every scalar structural metric
    metric_errors.csv              per-generator mean error of every scalar metric
    fidelity.csv                   portrait divergence, degree/clustering distances, edge overlap
    epidemic.csv                   SIR summary and functional errors
    timing.csv                     fit / generation time
    structure_vs_function.csv      one row per synthetic graph
    per_graph_metrics.csv          every metric for every synthetic graph
    epidemic_curves.csv            mean I(t) with 10-90 % band and mean R(t) per network
    epidemic_runs.csv              final size / peak / time-to-peak of every SIR run
    diagnostics.json               NetGAN history, LSM loss curves, fitted rho
    real/real.edgelist             the (cleaned) real network
    synthetic/<key>/graph_<i>.edgelist, synthetic/<key>/node_map_<i>.txt
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from netsynth import reporting as rp
from netsynth.structure import SCALAR_METRICS


@dataclass
class ResultBundle:
    meta: dict
    scalar: pd.DataFrame
    errors: pd.DataFrame
    fidelity: pd.DataFrame
    epidemic: pd.DataFrame
    timing: pd.DataFrame
    sf: pd.DataFrame
    per_graph: Optional[pd.DataFrame] = None
    curves: Optional[pd.DataFrame] = None
    runs: Optional[pd.DataFrame] = None
    real: Optional[nx.Graph] = None
    graphs: dict = field(default_factory=dict)       # key -> list[nx.Graph]
    node_maps: dict = field(default_factory=dict)    # key -> list[np.ndarray | None]
    diagnostics: dict = field(default_factory=dict)  # key -> dict
    failed: dict = field(default_factory=dict)       # key -> error message

    # ------------------------------------------------------------------ helpers
    @property
    def generators(self) -> list:
        return list(self.meta.get("generators", []))

    def label(self, key: str) -> str:
        return {g["key"]: g["label"] for g in self.generators}.get(key, key)

    def key_of(self, label: str) -> str:
        return {g["label"]: g["key"] for g in self.generators}.get(label, label)

    def keys(self) -> list:
        return [g["key"] for g in self.generators]

    @property
    def has_graphs(self) -> bool:
        return self.real is not None and any(self.graphs.get(k) for k in self.graphs)

    @property
    def floor(self) -> dict:
        return self.meta.get("floor", {})

    @property
    def density(self) -> float:
        return float(self.meta.get("real_density", np.nan))


# ----------------------------------------------------------------------------- from a run
def _curves_and_runs(res) -> tuple:
    g = res.grid
    cur, runs = [], []
    nets = [("Real", res.real_sir)] + [(r.label, r.sir_pooled) for r in res.reports.values() if r.ok and r.sir_pooled]
    for name, s in nets:
        I, R = s["I"], s["R"]
        cur.append(pd.DataFrame({"network": name, "time": g, "I_mean": I.mean(0),
                                 "I_p10": np.percentile(I, 10, axis=0), "I_p90": np.percentile(I, 90, axis=0),
                                 "R_mean": R.mean(0), "R_p10": np.percentile(R, 10, axis=0),
                                 "R_p90": np.percentile(R, 90, axis=0)}))
        runs.append(pd.DataFrame({"network": name, "run": np.arange(len(s["final"])), "final_size": s["final"],
                                  "peak_prevalence": s["peak"], "time_to_peak": s["tpeak"]}))
    return pd.concat(cur, ignore_index=True), pd.concat(runs, ignore_index=True)


def _per_graph(res) -> pd.DataFrame:
    rows = []
    for k, r in rp.ok_reports(res).items():
        for i in range(len(r.summaries)):
            row = {"generator": r.label, "key": k, "graph": i}
            row.update({c: r.summaries[i].scalars[c] for c in ["n", "m", "density", *SCALAR_METRICS, "n_triangles", "n_components"]})
            row.update({f"err_{c}": r.errors[i][c] for c in SCALAR_METRICS})
            row["portrait_divergence"] = r.pd[i] if r.pd else np.nan
            row.update({"degree_ks": r.degree_d[i]["degree_ks"], "degree_w1": r.degree_d[i]["degree_wasserstein"],
                        "degree_jsd": r.degree_d[i]["degree_jsd"], "clustering_ks": r.cluster_ks[i],
                        "edge_overlap": r.overlap[i] if r.overlap[i] is not None else np.nan,
                        "beta_c": r.beta_threshold[i]})
            row.update(r.sir_cmp_each[i])
            rows.append(row)
    return pd.DataFrame(rows)


def bundle_from_result(res, name: str = "network") -> ResultBundle:
    ok = rp.ok_reports(res)
    cfg = res.config
    curves, runs = _curves_and_runs(res)
    diag = {}
    for k, r in ok.items():
        info = dict(r.result.info)
        d = {}
        if "history" in info:
            d["netgan_history"] = {a: [float(x) for x in b] if isinstance(b, list) else b for a, b in info["history"].items()}
            d["device"] = info.get("device")
        for a in ("loss_init", "loss_refine", "rho", "r", "lsm_seconds", "diffusion_train_seconds",
                  "edge_loss_from_simplification", "m_ba", "latent_noise", "leak_probe", "leak_target"):
            if a in info:
                d[a] = [float(x) for x in info[a]] if isinstance(info[a], list) else info[a]
        diag[k] = d
    meta = {
        "name": name, "version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": res.real.number_of_nodes(), "m": res.real.number_of_edges(),
        "real_density": res.real_summary.scalars["density"], "real_scalars": res.real_summary.scalars,
        "beta": res.sir_cfg.beta, "beta_c": res.beta_c, "beta_mult": cfg.beta_mult, "gamma": cfg.gamma,
        "init_frac": cfg.init_frac, "runs_per_graph": cfg.runs_per_graph, "k": cfg.k, "seed": cfg.seed,
        "big_outbreak": cfg.big_outbreak, "floor": res.floor, "seconds": res.seconds,
        "generators": [{"key": k, "label": r.label} for k, r in ok.items()],
        "node_map_kind": {},
    }
    nm = {}
    for k, r in ok.items():
        nm[k] = list(r.result.node_maps)
        first = nm[k][0] if nm[k] else None
        n = res.real.number_of_nodes()
        meta["node_map_kind"][k] = ("none" if first is None else
                                    "identity" if np.array_equal(np.asarray(first), np.arange(n)) else "bootstrap")
    return ResultBundle(
        meta=meta, scalar=rp.scalar_table(res), errors=rp.error_table(res), fidelity=rp.fidelity_table(res),
        epidemic=rp.sir_table(res), timing=rp.timing_table(res), sf=rp.structure_function_points(res),
        per_graph=_per_graph(res), curves=curves, runs=runs, real=res.real,
        graphs={k: list(r.result.graphs) for k, r in ok.items()}, node_maps=nm, diagnostics=diag,
        failed={k: r.error for k, r in res.reports.items() if not r.ok})


# ----------------------------------------------------------------------------- saving
def _edgelist(G: nx.Graph) -> str:
    return "\n".join(f"{u} {v}" for u, v in G.edges())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def bundle_files(b: ResultBundle, include_graphs: bool = True) -> dict:
    """filename -> bytes of every file of the export."""
    f = {}
    csv = lambda df, **kw: df.to_csv(**kw).encode()
    f["scalar_metrics.csv"] = csv(b.scalar)
    f["metric_errors.csv"] = csv(b.errors)
    f["fidelity.csv"] = csv(b.fidelity)
    f["epidemic.csv"] = csv(b.epidemic)
    f["timing.csv"] = csv(b.timing)
    f["structure_vs_function.csv"] = csv(b.sf, index=False)
    if b.per_graph is not None:
        f["per_graph_metrics.csv"] = csv(b.per_graph, index=False)
    if b.curves is not None:
        f["epidemic_curves.csv"] = csv(b.curves, index=False)
    if b.runs is not None:
        f["epidemic_runs.csv"] = csv(b.runs, index=False)
    f["meta.json"] = json.dumps(b.meta, indent=2, default=_json_default).encode()
    f["diagnostics.json"] = json.dumps(b.diagnostics, default=_json_default).encode()
    if include_graphs and b.real is not None:
        f["real/real.edgelist"] = _edgelist(b.real).encode()
        for k, gl in b.graphs.items():
            for i, H in enumerate(gl):
                f[f"synthetic/{k}/graph_{i}.edgelist"] = _edgelist(H).encode()
                if b.meta.get("node_map_kind", {}).get(k) == "bootstrap":
                    f[f"synthetic/{k}/node_map_{i}.txt"] = "\n".join(map(str, np.asarray(b.node_maps[k][i]).tolist())).encode()
    return f


def bundle_to_zip(b: ResultBundle, include_graphs: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in bundle_files(b, include_graphs).items():
            z.writestr(name, data)
    return buf.getvalue()


def write_dir(b: ResultBundle, path) -> None:
    path = Path(path)
    for name, data in bundle_files(b).items():
        p = path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# ----------------------------------------------------------------------------- loading
class _Source:
    """Uniform read access to a ZIP or a directory."""
    def __init__(self, zf=None, root=None, files=None):
        self.zf, self.root, self.files = zf, root, files
        if files is not None:
            self.names = set(files)
        else:
            self.names = set(zf.namelist()) if zf else {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}

    def has(self, n): return n in self.names
    def read(self, n):
        if self.files is not None:
            return self.files[n]
        return self.zf.read(n) if self.zf else (self.root / n).read_bytes()
    def csv(self, n, index=True):
        return pd.read_csv(io.BytesIO(self.read(n)), index_col=0 if index else None) if self.has(n) else None


def _read_graph(text: str, n: Optional[int]) -> nx.Graph:
    G = nx.Graph()
    if n:
        G.add_nodes_from(range(n))
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 2:
            G.add_edge(int(t[0]), int(t[1]))
    return G


def _load(src: _Source, default_name: str) -> ResultBundle:
    need = ["scalar_metrics.csv", "fidelity.csv", "epidemic.csv"]
    missing = [n for n in need if not src.has(n)]
    if missing:
        raise ValueError(f"Not a results export: missing {', '.join(missing)}")
    meta = json.loads(src.read("meta.json")) if src.has("meta.json") else {}
    scalar, errors = src.csv("scalar_metrics.csv"), src.csv("metric_errors.csv")
    fid, epi, timing = src.csv("fidelity.csv"), src.csv("epidemic.csv"), src.csv("timing.csv")
    sf = src.csv("structure_vs_function.csv", index=False)
    if "generators" not in meta:                       # older exports without meta.json
        keys = fid["key"].tolist() if "key" in fid.columns else [str(i) for i in fid.index]
        meta["generators"] = [{"key": k, "label": lab} for k, lab in zip(keys, fid.index)]
    meta.setdefault("name", default_name)
    if "real_density" not in meta and "Overlap vs chance (×)" in fid.columns:
        ratio = (fid["Edge overlap with real"] / fid["Overlap vs chance (×)"]).dropna()
        meta["real_density"] = float(ratio.iloc[0]) if len(ratio) else np.nan
    if "n" not in meta and "Real" in scalar.index:
        meta["m"] = int(scalar.loc["Real", "edges (m)"])
    b = ResultBundle(meta=meta, scalar=scalar, errors=errors if errors is not None else pd.DataFrame(),
                     fidelity=fid, epidemic=epi, timing=timing if timing is not None else pd.DataFrame(),
                     sf=sf if sf is not None else pd.DataFrame(), per_graph=src.csv("per_graph_metrics.csv", index=False),
                     curves=src.csv("epidemic_curves.csv", index=False), runs=src.csv("epidemic_runs.csv", index=False))
    if src.has("diagnostics.json"):
        b.diagnostics = json.loads(src.read("diagnostics.json"))
    if src.has("real/real.edgelist"):
        n = meta.get("n")
        b.real = _read_graph(src.read("real/real.edgelist").decode(), n)
        for g in b.generators:
            k, i, gl, nml = g["key"], 0, [], []
            while src.has(f"synthetic/{k}/graph_{i}.edgelist"):
                gl.append(_read_graph(src.read(f"synthetic/{k}/graph_{i}.edgelist").decode(), n))
                kind = meta.get("node_map_kind", {}).get(k, "none")
                if kind == "identity":
                    nml.append(np.arange(n))
                elif kind == "bootstrap" and src.has(f"synthetic/{k}/node_map_{i}.txt"):
                    nml.append(np.array(src.read(f"synthetic/{k}/node_map_{i}.txt").decode().split(), dtype=int))
                else:
                    nml.append(None)
                i += 1
            if gl:
                b.graphs[k], b.node_maps[k] = gl, nml
    return b


def bundle_from_zip(data: bytes, name: str = "saved results") -> ResultBundle:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return _load(_Source(zf=z), name)


def bundle_from_dir(path) -> ResultBundle:
    path = Path(path)
    return _load(_Source(root=path), path.name)


def bundle_from_files(files: dict, name: str = "uploaded results") -> ResultBundle:
    """files: {filename: bytes} e.g. several uploaded CSV/JSON files (path separators allowed)."""
    return _load(_Source(files={k.replace("\\", "/").split("/")[-1] if "/" not in k else k: v for k, v in files.items()}), name)
