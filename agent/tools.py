"""
agent/tools.py
==============
Typed tool layer wrapping every methodology capability built in Steps 1-7.

Each tool is a pure function with:
  * a typed signature
  * a one-line docstring suitable for an LLM tool description
  * a stable JSON-serialisable return value

This is the boundary between the deterministic methodology code and the
agent orchestration / optional LLM driver.

Tools return: {"ok": bool, "message": str, "result": ..., ...optional handles}
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reduced_model      import simulate, Protocol, DEFAULT_PARAMS, CORE_PARAM_ORDER
from core.diagnostics        import detect_instability
from data_io.loader          import load_dataset
from data_io.preprocess      import preprocess
from calibration.calibrate   import calibrate_de, calibrate_staged
from analysis.identifiability import fisher_information, profile_likelihood
from analysis.sensitivity    import morris_screening, sobol_indices, time_resolved_sobol
from analysis.validation     import (
    within_trace_holdout, parametric_bootstrap_predictive_check,
)


def _ok(message="", **payload):
    return {"ok": True, "message": message, **payload}

def _err(message, **payload):
    return {"ok": False, "message": message, **payload}

def _to_jsonable(o):
    if isinstance(o, np.ndarray):       return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.bool_):          return bool(o)
    if isinstance(o, dict):              return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):     return [_to_jsonable(v) for v in o]
    if is_dataclass(o):                  return _to_jsonable(asdict(o))
    return o


def load_data(path: str, chamber_index: int = 0):
    """Load a respirometry Excel/CSV/NPY file; returns event boundaries and
    an in-memory chamber handle for downstream tools."""
    if not os.path.exists(path):
        return _err(f"file not found: {path}")
    try:
        ds = load_dataset(path)
    except Exception as e:
        return _err(f"loader failed: {e!r}")
    if chamber_index >= len(ds.chambers):
        return _err(f"chamber_index {chamber_index} out of range "
                     f"(have {len(ds.chambers)})")
    ch = ds.chambers[chamber_index]
    return _ok(f"Loaded {os.path.basename(path)} "
                f"(chamber {ch.label}, n={len(ch.t)})",
                result={
                    "n_chambers": len(ds.chambers),
                    "chamber_label": ch.label,
                    "n_samples": int(len(ch.t)),
                    "t_start":   float(ch.t[0]),
                    "t_end":     float(ch.t[-1]),
                    "t_oligo":   ch.t_oligo,
                    "t_fccp":    [float(x) for x in ch.t_fccp],
                    "t_inhibit": ch.t_inhibit,
                    "n_fccp":    len(ch.t_fccp),
                    "noise_sd_estimate": float(ch.sigma_obs_est)
                                          if ch.sigma_obs_est else None,
                },
                _chamber_obj=ch)


def preprocess_data(chamber, do_outliers=True, n_sigma=4.0, do_smooth=False):
    """Preprocess a chamber: outlier rejection, event validation."""
    try:
        ch_p, issues = preprocess(chamber, do_outliers=do_outliers,
                                    n_sigma=n_sigma, do_smooth=do_smooth)
    except Exception as e:
        return _err(f"preprocessing failed: {e!r}")
    return _ok(f"Preprocessed: {len(ch_p.t)}/{len(chamber.t)} samples retained, "
                f"{len(issues)} issue(s)",
                result={
                    "n_in":        int(len(chamber.t)),
                    "n_out":       int(len(ch_p.t)),
                    "n_dropped":   int(len(chamber.t) - len(ch_p.t)),
                    "validation_issues": list(issues),
                },
                _chamber_obj=ch_p)


def simulate_default(chamber, params=None, k_step=2.0):
    """Forward-simulate the 3-state OCR-informed model on a chamber's protocol."""
    proto = chamber.to_protocol(k_step=k_step)
    p = dict(DEFAULT_PARAMS)
    if params:
        for k, v in params.items():
            if k == "alphas":
                p["alphas"] = list(v)
            else:
                p[k] = v
    if "alphas" not in p or len(p["alphas"]) != len(proto.t_fccp):
        p["alphas"] = [1.0] * len(proto.t_fccp)
    res = simulate(p, proto, o2_init=float(chamber.o[0]),
                   t_eval=np.linspace(proto.t_start, proto.t_end, 400))
    if not res.converged:
        return _err("simulation failed to converge", params=_to_jsonable(p))
    return _ok("Simulation OK",
                result={"t":   res.t.tolist(),
                          "o":   res.o.tolist(),
                          "OCR": res.OCR.tolist(),
                          "metrics": _to_jsonable(res.seahorse_metrics())})


def calibrate(chamber, method="de", n_data=250, **kwargs):
    """Calibrate the model. method ∈ {'de','staged'}."""
    proto = chamber.to_protocol()
    n = len(chamber.t)
    idx = np.round(np.linspace(0, n-1, min(n_data, n))).astype(int)
    t_d, o_d = chamber.t[idx], chamber.o[idx]
    if method == "de":
        kwargs.setdefault("maxiter", 80); kwargs.setdefault("popsize", 10)
        res = calibrate_de(t_d, o_d, proto, **kwargs)
    elif method == "staged":
        kwargs.setdefault("maxiter_stageA", 40); kwargs.setdefault("maxiter_stageB", 60)
        res = calibrate_staged(t_d, o_d, proto, **kwargs)
    else:
        return _err(f"unknown method: {method}")
    rmse = (res.objective / len(t_d)) ** 0.5
    return _ok(f"Calibration done ({method}): RMSE={rmse:.3f} nmol/mL",
                result={"method": res.method,
                          "params": _to_jsonable(res.params),
                          "rmse_calib": rmse,
                          "objective": float(res.objective),
                          "n_data": int(len(t_d))},
                _chamber_obj=chamber, _calib_t=t_d, _calib_o=o_d,
                _params=res.params)


def check_stability(chamber, params):
    """Stability diagnostic: stiffness + conservation + tolerance robustness."""
    proto = chamber.to_protocol()
    p = dict(DEFAULT_PARAMS); p.update(params or {})
    if "alphas" not in p:
        p["alphas"] = [1.0] * len(proto.t_fccp)
    rep = detect_instability(p, proto, o2_init=float(chamber.o[0]))
    return _ok(rep.summary(),
                result={
                    "is_healthy":            rep.is_healthy(),
                    "max_jacobian_eig":      float(rep.max_jacobian_eig),
                    "stiffness_ratio":       float(rep.stiffness_ratio),
                    "cytc_drift":            float(rep.cytc_conservation_drift),
                    "oxygen_monotone":       bool(rep.oxygen_monotone),
                    "tolerance_robust":      bool(rep.solver_tolerance_robust),
                    "warnings":              list(rep.warnings),
                })


def analyze_identifiability(chamber, params, method="fim",
                              grid_n=9, grid_span_log=0.6,
                              param_subset=None):
    """Identifiability: 'fim' (cheap), 'profile' (slow), 'both'."""
    proto = chamber.to_protocol()
    n = len(chamber.t)
    idx = np.round(np.linspace(0, n-1, 150)).astype(int)
    t_d, o_d = chamber.t[idx], chamber.o[idx]
    p = dict(params)
    if "alphas" not in p:
        p["alphas"] = [1.0] * len(proto.t_fccp)
    out = {}
    if method in ("fim", "both"):
        try:
            rep = fisher_information(p, t_d, o_d, proto,
                                      o2_init=float(o_d[0]),
                                      sigma_obs=p.get("sigma_obs"))
            out["fim"] = {
                "param_names":       rep.param_names,
                "eigvals_raw":       rep.eigvals_raw.tolist(),
                "eigvals_clipped":   rep.eigvals_clipped.tolist(),
                "condition_raw":     float(rep.condition_raw),
                "condition_clipped": float(rep.condition_clipped),
                # Legacy alias for older readers
                "condition":         float(rep.condition_raw),
                "warnings":          list(rep.warnings),
            }
        except Exception as e:
            out["fim_error"] = str(e)
    if method in ("profile", "both"):
        keys = param_subset or list(CORE_PARAM_ORDER)
        out["profiles"] = {}
        for nm in keys:
            try:
                pr = profile_likelihood(nm, p, t_d, o_d, proto,
                                          o2_init=float(o_d[0]),
                                          n_grid=grid_n,
                                          grid_span_log=grid_span_log,
                                          maxiter=12)
                out["profiles"][nm] = {"ci_low": pr.ci_low,
                                         "ci_high": pr.ci_high,
                                         "verdict": pr.practical_id}
            except Exception as e:
                out["profiles"][nm] = {"error": str(e)}
    summary = []
    if "fim" in out:
        summary.append(f"FIM cond ≈ {out['fim']['condition']:.1e}")
    if "profiles" in out:
        verdicts = [v.get("verdict", "") for v in out["profiles"].values()]
        summary.append(f"profiles: {verdicts.count('identifiable')} id / "
                        f"{verdicts.count('one-sided')} 1-side / "
                        f"{verdicts.count('non-identifiable')} non-id")
    return _ok(" | ".join(summary), result=out)


def analyze_sensitivity(chamber, method="morris", N=30):
    """Sensitivity analysis: 'morris','sobol','time_sobol'."""
    proto = chamber.to_protocol()
    o2_init = float(chamber.o[0])
    try:
        if method == "morris":
            S = morris_screening(proto, o2_init=o2_init, N_trajectories=N, num_levels=4)
            order = np.argsort(-S["mu_star"])
            ranked = [(S["names"][i], float(S["mu_star"][i]), float(S["sigma"][i]))
                       for i in order]
            return _ok(f"Morris (n_eval={S['n_evals']})",
                        result={"method": "morris", "ranking": ranked,
                                  "names": S["names"],
                                  "mu_star": S["mu_star"].tolist(),
                                  "sigma":   S["sigma"].tolist()})
        if method == "sobol":
            S = sobol_indices(proto, o2_init=o2_init, N=N)
            order = np.argsort(-S["ST"])
            ranked = [(S["names"][i], float(S["S1"][i]), float(S["ST"][i]))
                       for i in order]
            return _ok(f"Sobol (n_eval={S['n_evals']})",
                        result={"method": "sobol", "ranking": ranked,
                                  "names": S["names"],
                                  "S1": S["S1"].tolist(),
                                  "ST": S["ST"].tolist(),
                                  "interpretation_note": S.get("interpretation_note", "")})
        if method == "time_sobol":
            S = time_resolved_sobol(proto, o2_init=o2_init, N=N, n_t_eval=20)
            return _ok(f"Time-resolved Sobol (n_eval={S['n_evals']})",
                        result={"method": "time_sobol",
                                  "names": S["names"],
                                  "t_grid": S["t_grid"].tolist(),
                                  "S1_t": S["S1_t"].tolist(),
                                  "ST_t": S["ST_t"].tolist(),
                                  "variance_degenerate_mask": S["variance_degenerate_mask"].tolist(),
                                  "interpretation_note": S.get("interpretation_note", "")})
    except RuntimeError as e:
        return _ok("Sensitivity analysis skipped: optional dependency unavailable",
                   result={"method": method, "status": "skipped",
                           "reason": str(e),
                           "recommended_install": "python -m pip install -r requirements-sensitivity.txt"})
    return _err(f"unknown method: {method}")


def validate(chamber, params, method="ppc", **kwargs):
    """Validation: 'ppc' (parametric-bootstrap predictive check) or 'within_trace'."""
    proto = chamber.to_protocol()
    n = len(chamber.t)
    idx = np.round(np.linspace(0, n-1, 200)).astype(int)
    t_d, o_d = chamber.t[idx], chamber.o[idx]
    p = dict(params)
    if "alphas" not in p:
        p["alphas"] = [1.0] * len(proto.t_fccp)
    if method == "ppc":
        ppc = parametric_bootstrap_predictive_check(
            t_d, o_d, proto, p,
            n_boot=kwargs.get("n_boot", 200),
            refit=False)
        cov = float(ppc.get("parametric_bootstrap_coverage_90",
                             ppc.get("coverage_90", float("nan"))))
        # Note: the [0.80, 0.95] band is a CONFIGURABLE warning threshold,
        # not a validated band of biologically-correct calibration. The
        # `within_configurable_band` field is reported alongside but
        # callers should treat it as advisory.
        within_band = bool(0.80 <= cov <= 0.95)
        return _ok(
            f"parametric-bootstrap predictive check coverage at 90% PI: "
            f"{cov*100:.1f}%",
            result={
                "parametric_bootstrap_coverage_90": cov,
                "coverage_90": cov,    # legacy alias
                "n_boot": int(ppc.get("n_boot", 0)),
                "within_configurable_band": within_band,
                "configurable_threshold_low":  0.80,
                "configurable_threshold_high": 0.95,
                "explicit_disclaimer": ppc.get(
                    "explicit_disclaimer",
                    "Parametric-bootstrap predictive check, "
                    "NOT posterior-predictive."),
            })
    if method == "within_trace":
        wt = within_trace_holdout(t_d, o_d, proto,
                                    train_frac=kwargs.get("train_frac", 0.7),
                                    calibrate_fn=calibrate_staged,
                                    maxiter_stageA=20, maxiter_stageB=30)
        return _ok(f"Within-trace: RMSE_train={wt['rmse_train']:.2f}, "
                    f"RMSE_test={wt['rmse_test']:.2f}",
                    result={"rmse_train": float(wt["rmse_train"]),
                              "rmse_test":  float(wt["rmse_test"]),
                              "n_train": int(wt["n_train"]),
                              "n_test":  int(wt["n_test"])})
    return _err(f"unknown method: {method}")


TOOLS = {
    "load_data":               load_data,
    "preprocess_data":         preprocess_data,
    "simulate_default":        simulate_default,
    "calibrate":               calibrate,
    "check_stability":         check_stability,
    "analyze_identifiability": analyze_identifiability,
    "analyze_sensitivity":     analyze_sensitivity,
    "validate":                validate,
}

TOOL_DESCRIPTIONS = {
    "load_data":               "Load Excel/CSV/NPY trace; returns event boundaries.",
    "preprocess_data":         "Preprocess a chamber (outlier rejection, validation).",
    "simulate_default":        "Forward-simulate the 3-state OCR-informed model.",
    "calibrate":               "Fit model parameters; method='de' or 'staged'.",
    "check_stability":         "Numerical-stability diagnostic.",
    "analyze_identifiability": "Profile-likelihood and/or FIM identifiability.",
    "analyze_sensitivity":     "Morris/Sobol/time-resolved global sensitivity.",
    "validate":                "Validation: 'ppc' or 'within_trace'.",
}
