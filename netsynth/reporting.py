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
    """What 'close enough' means. Defaults are rules of thumb - tune them to your use case.

    Epidemic tolerances are never applied tighter than the real network's own run-to-run noise (the noise floor):
    a generator cannot be required to match the real epidemic better than the real network matches itself.
    """
    degree_ks: float = 0.25             # Kolmogorov-Smirnov distance between degree distributions
    final_size_abs: float = 0.05        # absolute difference in attack rate (5 percentage points)
    peak_rel: float = 0.25
    curve_nrmse: float = 0.20
    outbreak_prob_abs: float = 0.15
    max_leak_excess: float = 0.25       # share of synthetic edges that are real edges, beyond chance (see leak_excess)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def leak_excess(overlap, ratio_to_chance):
    """Edge overlap above chance, rescaled to 0..1:  (overlap - chance) / (1 - chance).

    ``chance`` is the overlap a random graph would have (the real density); it is recovered from the exported
    'overlap vs chance' ratio. 0 = no more real edges than luck gives, 1 = every synthetic edge is a real edge.
    """
    overlap, ratio = _num(overlap), _num(ratio_to_chance)
    if not (np.isfinite(overlap) and np.isfinite(ratio)) or ratio <= 0:
        return np.nan
    chance = min(overlap / ratio, 0.999)
    return max(0.0, (overlap - chance) / (1.0 - chance))


def effective_tolerances(b, tol: Tolerances) -> dict:
    """User tolerances, raised to the real-vs-real noise floor where the floor is higher."""
    fl = b.floor or {}
    def eff(t, key):
        f = _num(fl.get(key))
        return max(t, f) if np.isfinite(f) else t
    return {"degree_ks": tol.degree_ks, "final_size": eff(tol.final_size_abs, "final_size_err"), "peak": eff(tol.peak_rel, "peak_err_rel"),
            "curve": eff(tol.curve_nrmse, "curve_nrmse"), "outbreak": eff(tol.outbreak_prob_abs, "outbreak_prob_err")}


def verdicts(b, tol: Tolerances) -> pd.DataFrame:
    """Per generator: each check (value, threshold) and an overall verdict. Works on a ResultBundle.

    safe for simulation = degree distribution close AND the four epidemic errors within tolerance;
    safe for sharing    = safe for simulation AND not copying real edges beyond the leakage tolerance.
    Portrait divergence is reported but is not a gate: on the bundled networks it did not predict epidemic error.
    """
    et = effective_tolerances(b, tol)
    rows = []
    for g in b.generators:
        lab = g["label"]
        if lab not in b.epidemic.index or lab not in b.fidelity.index:
            continue
        e, f = b.epidemic.loc[lab], b.fidelity.loc[lab]
        checks = []
        dks = _num(f.get("Degree KS"))
        if np.isfinite(dks):
            checks.append(("Structure: degree distribution (KS)", dks, et["degree_ks"], True))
        checks += [("Epidemic: final-size error", _num(e["Δ final size"]), et["final_size"], True),
                   ("Epidemic: peak error (rel.)", _num(e["Peak err (rel.)"]), et["peak"], True),
                   ("Epidemic: curve NRMSE", _num(e["Curve NRMSE"]), et["curve"], True),
                   ("Epidemic: outbreak-prob. error", _num(e["Outbreak-prob err"]), et["outbreak"], True)]
        sim_pass = all(v <= t for _, v, t, _ in checks)
        ov = _num(f.get("Edge overlap with real"))
        ex = leak_excess(ov, f.get("Overlap vs chance (×)"))
        leak = ov if np.isfinite(ov) else None
        leak_fail = np.isfinite(ex) and ex > tol.max_leak_excess
        share_pass = sim_pass and not leak_fail
        rows.append({"Generator": lab, "simulation_ok": sim_pass, "sharing_ok": share_pass, "leak": leak,
                     "leak_excess": ex if np.isfinite(ex) else None,
                     "failed": [n for n, v, t, _ in checks if v > t] + (["Leakage: real edges copied"] if leak_fail else []),
                     "checks": checks})
    return pd.DataFrame(rows).set_index("Generator")


def takeaways(b, tol: Tolerances) -> list:
    """Plain-language findings derived from the numbers (no hand-written conclusions)."""
    out = []
    for k, err in b.failed.items():
        out.append(f"⚠️ **{k}** failed ({err}); the remaining generators are unaffected.")
    if not b.generators:
        return out
    fid = b.fidelity
    if "Portrait divergence" in fid and fid["Portrait divergence"].notna().any():
        best = fid["Portrait divergence"].idxmin()
        out.append(f"**Closest in structure:** {best} (portrait divergence {fid.loc[best, 'Portrait divergence']:.3f}).")
    curve = b.epidemic.drop(index="Real", errors="ignore")["Curve NRMSE"].dropna()
    if len(curve):
        best = curve.idxmin()
        out.append(f"**Closest epidemic behaviour:** {best} (outbreak-curve NRMSE {curve[best]:.2f}; "
                   f"sampling-noise floor of the real network ≈ {b.floor.get('curve_nrmse', float('nan')):.2f}).")
    sf = b.sf
    if len(sf) >= 6 and "Portrait divergence" in sf and sf["Portrait divergence"].notna().all():
        from scipy.stats import spearmanr
        rho, p = spearmanr(sf["Portrait divergence"], sf["Curve NRMSE"])
        if np.isfinite(rho):
            out.append(f"**Does structure predict function?** Across all synthetic graphs, portrait divergence and "
                       f"epidemic-curve error have Spearman ρ = {rho:.2f} (p = {p:.2g}).")
    v = verdicts(b, tol)
    sim = [g for g in v.index if v.loc[g, "simulation_ok"]]
    shr = [g for g in v.index if v.loc[g, "sharing_ok"]]
    leaky = [g for g in sim if g not in shr]
    out.append("**Safe simulation substitute (under your tolerances):** " + (", ".join(sim) if sim else "none"))
    out.append("**Safe for data sharing (also passes the edge-leakage check):** " + (", ".join(shr) if shr else "none"))
    if leaky:
        out.append("🔒 " + ", ".join(leaky) + (" reproduces" if len(leaky) == 1 else " reproduce") +
                   " the dynamics but copies too many real edges to be shared as 'anonymous' data.")
    return out
