"""Turn an ExperimentResult into tables and a 'safe substitute?' verdict."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from netsynth.structure import SCALAR_METRICS


def _ms(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (float(np.mean(x)), float(np.std(x))) if x else (np.nan, np.nan)


def ok_reports(res):
    return {k: r for k, r in res.reports.items() if r.ok and r.result is not None}


def scalar_table(res) -> pd.DataFrame:
    """Mean over the K synthetic graphs of every scalar metric, plus the real value."""
    cols = list(SCALAR_METRICS)
    rows = [{"Network": "Real", **{c: res.real_summary.scalars[c] for c in cols}, "edges (m)": res.real_summary.scalars["m"]}]
    for k, r in ok_reports(res).items():
        row = {"Network": r.label, "edges (m)": _ms([s.scalars["m"] for s in r.summaries])[0]}
        for c in cols:
            row[c] = _ms([s.scalars[c] for s in r.summaries])[0]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Network")
    return df.rename(columns={c: SCALAR_METRICS[c][0] for c in cols})


def error_table(res) -> pd.DataFrame:
    """Mean error of each scalar metric against the real network (rel. or abs., see SCALAR_METRICS)."""
    rows = []
    for k, r in ok_reports(res).items():
        row = {"Generator": r.label}
        for c in SCALAR_METRICS:
            row[SCALAR_METRICS[c][0]] = _ms([e[c] for e in r.errors])[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Generator")


def fidelity_table(res) -> pd.DataFrame:
    rows = []
    real_density = res.real_summary.scalars["density"]
    for k, r in ok_reports(res).items():
        pdm, pds = _ms(r.pd)
        ov = _ms(r.overlap)[0]
        rows.append({
            "Generator": r.label, "key": k,
            "Portrait divergence": pdm, "PD std": pds,
            "Degree KS": _ms([d["degree_ks"] for d in r.degree_d])[0],
            "Degree W1 (norm.)": _ms([d["degree_wasserstein"] for d in r.degree_d])[0],
            "Degree JSD": _ms([d["degree_jsd"] for d in r.degree_d])[0],
            "Clustering KS": _ms(r.cluster_ks)[0],
            "Edge overlap with real": ov,
            "Overlap vs chance (×)": ov / real_density if np.isfinite(ov) and real_density > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("Generator")


def sir_table(res) -> pd.DataFrame:
    rr = res.real_sir
    rows = [{"Network": "Real", "Final size": rr["final"].mean(), "Peak prevalence": rr["peak"].mean(),
             "Time to peak": rr["tpeak"].mean(), "Outbreak prob.": rr["outbreak_prob"]}]
    for k, r in ok_reports(res).items():
        s = r.sir_pooled
        rows.append({"Network": r.label, "Final size": s["final"].mean(), "Peak prevalence": s["peak"].mean(),
                     "Time to peak": s["tpeak"].mean(), "Outbreak prob.": s["outbreak_prob"],
                     "Δ final size": r.sir_cmp["final_size_err"], "Peak err (rel.)": r.sir_cmp["peak_err_rel"],
                     "Curve NRMSE": r.sir_cmp["curve_nrmse"], "Outbreak-prob err": r.sir_cmp["outbreak_prob_err"]})
    return pd.DataFrame(rows).set_index("Network")


def timing_table(res) -> pd.DataFrame:
    rows = []
    for k, r in ok_reports(res).items():
        rows.append({"Generator": r.label, "Fit / training (s)": r.result.fit_seconds,
                     "Per-graph generation (s)": r.result.gen_seconds})
    return pd.DataFrame(rows).set_index("Generator")


def structure_function_points(res) -> pd.DataFrame:
    """One row per synthetic graph: structural scores vs functional (epidemic) errors."""
    rows = []
    for k, r in ok_reports(res).items():
        for i in range(len(r.summaries)):
            rows.append({
                "Generator": r.label, "graph": i,
                "Portrait divergence": r.pd[i] if r.pd else np.nan,
                "Degree KS": r.degree_d[i]["degree_ks"],
                "Δ final size": r.sir_cmp_each[i]["final_size_err"],
                "Curve NRMSE": r.sir_cmp_each[i]["curve_nrmse"],
                "Peak err (rel.)": r.sir_cmp_each[i]["peak_err_rel"],
            })
    return pd.DataFrame(rows)


@dataclass
class Tolerances:
    """What 'close enough' means. Defaults are rules of thumb - tune them to your use case."""
    portrait_divergence: float = 0.35
    final_size_abs: float = 0.05        # absolute difference in attack rate (5 percentage points)
    peak_rel: float = 0.25
    curve_nrmse: float = 0.20
    outbreak_prob_abs: float = 0.15
    max_edge_overlap: float = 0.25      # share of synthetic edges that are real edges


def verdicts(res, tol: Tolerances) -> pd.DataFrame:
    """Per generator: each check (value, threshold, pass) and an overall verdict."""
    rows = []
    for k, r in ok_reports(res).items():
        pdm = _ms(r.pd)[0]
        ov = _ms(r.overlap)[0]
        c = r.sir_cmp
        checks = [
            ("Structure: portrait divergence", pdm, tol.portrait_divergence, True) if r.pd else None,
            ("Epidemic: final-size error", c["final_size_err"], tol.final_size_abs, True),
            ("Epidemic: peak error (rel.)", c["peak_err_rel"], tol.peak_rel, True),
            ("Epidemic: curve NRMSE", c["curve_nrmse"], tol.curve_nrmse, True),
            ("Epidemic: outbreak-prob. error", c["outbreak_prob_err"], tol.outbreak_prob_abs, True),
        ]
        sim_checks = [x for x in checks if x]
        sim_pass = all(v <= t for _, v, t, _ in sim_checks)
        leak = None if not np.isfinite(ov) else ov
        share_pass = sim_pass and (leak is None or leak <= tol.max_edge_overlap)
        rows.append({
            "Generator": r.label,
            "simulation_ok": sim_pass,
            "sharing_ok": share_pass,
            "leak": leak,
            "failed": [n for n, v, t, _ in sim_checks if v > t] + (["Leakage: edge overlap"] if leak is not None and leak > tol.max_edge_overlap else []),
            "checks": sim_checks,
        })
    return pd.DataFrame(rows).set_index("Generator")


def takeaways(res, tol: Tolerances) -> list:
    """Plain-language findings derived from the numbers (no hand-written conclusions)."""
    out = []
    ok = ok_reports(res)
    bad = {k: r for k, r in res.reports.items() if not r.ok}
    for k, r in bad.items():
        out.append(f"⚠️ **{r.label}** failed ({r.error}); the remaining generators are unaffected.")
    if not ok:
        return out
    fid = fidelity_table(res)
    if fid["Portrait divergence"].notna().any():
        b = fid["Portrait divergence"].idxmin()
        out.append(f"**Closest in structure:** {b} (portrait divergence {fid.loc[b, 'Portrait divergence']:.3f}).")
    sir = {r.label: r.sir_cmp["curve_nrmse"] for r in ok.values()}
    b = min(sir, key=sir.get)
    out.append(f"**Closest epidemic behaviour:** {b} (outbreak-curve NRMSE {sir[b]:.2f}; "
               f"sampling-noise floor of the real network ≈ {res.floor.get('curve_nrmse', float('nan')):.2f}).")
    pts = structure_function_points(res)
    if len(pts) >= 6 and pts["Portrait divergence"].notna().all():
        from scipy.stats import spearmanr
        rho, p = spearmanr(pts["Portrait divergence"], pts["Curve NRMSE"])
        if np.isfinite(rho):
            out.append(f"**Does structure predict function?** Across all synthetic graphs, portrait divergence and "
                       f"epidemic-curve error have Spearman ρ = {rho:.2f} (p = {p:.2g}).")
    v = verdicts(res, tol)
    sim = [g for g in v.index if v.loc[g, "simulation_ok"]]
    shr = [g for g in v.index if v.loc[g, "sharing_ok"]]
    leaky = [g for g in sim if g not in shr]
    out.append("**Safe simulation substitute (under your tolerances):** " + (", ".join(sim) if sim else "none"))
    out.append("**Safe for data sharing (also passes the edge-leakage check):** " + (", ".join(shr) if shr else "none"))
    if leaky:
        out.append("🔒 " + ", ".join(leaky) + (" reproduces" if len(leaky) == 1 else " reproduce") +
                   " the dynamics but copies too many real edges to be shared as 'anonymous' data.")
    return out
