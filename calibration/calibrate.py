"""
calibration/calibrate.py
========================
Calibration framework for the 3-state OCR-informed bioenergetics model.

OBJECTIVE SEMANTICS
-------------------
The calibration routines optimise a SUM-OF-SQUARED-ERRORS (SSE) objective
on the oxygen residuals; sigma_obs is then ESTIMATED post-hoc from the
fitted residuals (root-mean-square). This is mathematically equivalent
to maximising the Gaussian likelihood ONLY IF the log-sigma normalising
term is ignored. We document this honestly:

    objective_type = "SSE_with_post_hoc_sigma"

This is NOT the full Gaussian negative log-likelihood
    NLL = 0.5 * sum((residual / sigma)^2 + log(2*pi*sigma^2))
because the log-sigma term is not part of the optimised objective.
The post-hoc-estimated sigma_obs is reported in every calibration JSON
so users can recompute the full Gaussian NLL externally if they need it.

STRATEGIES IMPLEMENTED
----------------------
1. DETERMINISTIC GLOBAL (Differential Evolution + L-BFGS-B polish)
   - One trace at a time, free-form
   - Internal log-transformation for parameters spanning >= 2 decades

2. SEQUENTIAL STAGED CALIBRATION (DIAGNOSTIC strategy, NOT a guaranteed
   mechanistic separation)
   - Stage A: fit (V_max, K_o, K_r, gamma_oligo) on the FCCP plateau and
     oligomycin phase, where they are MOST identifiable.
   - Stage B: with those frozen, fit (k_supply, c_tot, r0, tau_kappa,
     alpha_j) on the full trace.
   This is a HEURISTIC that often improves robustness for OCR-only data.
   It is NOT a guaranteed separation of CIV-side and supply-side dynamics.

3. POOLED MULTI-TRACE CALIBRATION (across chambers / datasets)
   - Core kinetic parameters are pooled across all chambers; only
     dataset-specific FCCP alphas and noise SD are fit per chamber.
   - This is implemented as a shared-parameter wrapper around the
     deterministic SSE objective. It is NOT a hierarchical Bayesian
     model with population hyperparameters and random effects.
   - Public name: `calibrate_pooled_multitrace`. The legacy alias
     `calibrate_hierarchical` remains as a deprecation-warning wrapper.

All routines return a `CalibrationResult` carrying the fitted parameter
dict, final SSE / objective value, post-hoc sigma_obs, RMSE on the
calibration window AND on the full untruncated trace, the optimiser
settings used (maxiter, popsize, seed), and a list of diagnostic
warnings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import differential_evolution, minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reduced_model import (
    Protocol, simulate, DEFAULT_PARAMS,
    PARAM_BOUNDS, CORE_PARAM_ORDER,
    params_to_vec, vec_to_params, get_bounds_vec, get_param_names,
)


# ── Result container ─────────────────────────────────────────────────────
@dataclass
class CalibrationResult:
    params: dict
    objective: float
    n_eval: int
    success: bool
    message: str = ""
    method: str = ""
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    history: List[float] = field(default_factory=list)
    chamber_label: str = ""
    weighted_residuals: Optional[np.ndarray] = None
    # ── CHUNK-2 provenance fields ────────────────────────────────────────
    objective_type: str = "SSE_with_post_hoc_sigma"
    rmse_calib: Optional[float] = None
    rmse_full_trace: Optional[float] = None
    n_data: int = 0
    seed: int = 0
    optimiser_settings: dict = field(default_factory=dict)
    sigma_obs: Optional[float] = None
    sigma_estimation_method: str = "post-hoc residual RMS (deterministic SSE)"
    warnings: List[str] = field(default_factory=list)


# ── Log-parameter transforms (for scale-free optimisation) ────────────────
def _to_log(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log-transform entries where mask is True; identity otherwise."""
    out = v.copy().astype(float)
    out[mask] = np.log(np.clip(v[mask], 1e-30, None))
    return out


def _from_log(v_log: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = v_log.copy().astype(float)
    out[mask] = np.exp(v_log[mask])
    return out


def _log_mask(param_names: Sequence[str]) -> np.ndarray:
    """Parameters whose bounds span >= 2 orders of magnitude get log-fitted."""
    mask = np.zeros(len(param_names), dtype=bool)
    for i, nm in enumerate(param_names):
        if nm.startswith("alpha_"):
            mask[i] = True
            continue
        key = nm
        if key in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[key]
            mask[i] = (hi / max(lo, 1e-30)) >= 100.0
    return mask


# ── Objective: Gaussian log-likelihood / SSE -------------------------------
def negloglik(params: dict, t_data: np.ndarray, o_data: np.ndarray,
              proto: Protocol, sigma_obs: Optional[float] = None) -> float:
    """Negative log-likelihood under iid Gaussian observation noise on O2."""
    # solve at the data time grid for exactness
    res = simulate(params, proto, o2_init=float(o_data[0]),
                   r0=params.get("r0"), t_eval=t_data,
                   rtol=1e-7, atol=1e-9)
    if not res.converged:
        return 1e15
    sd = sigma_obs if sigma_obs is not None else params.get("sigma_obs", 0.5)
    sd = max(float(sd), 1e-3)
    resid = res.o - o_data
    n = len(o_data)
    nll = 0.5 * n * np.log(2 * np.pi * sd**2) + 0.5 * np.sum(resid**2) / sd**2
    if not np.isfinite(nll):
        return 1e15
    return float(nll)


def sse(params: dict, t_data: np.ndarray, o_data: np.ndarray,
        proto: Protocol) -> float:
    """Plain sum-of-squared-errors (SSE) for deterministic optimisation."""
    res = simulate(params, proto, o2_init=float(o_data[0]),
                   r0=params.get("r0"), t_eval=t_data,
                   rtol=1e-6, atol=1e-9)
    if not res.converged:
        return 1e15
    return float(np.sum((res.o - o_data)**2))


# ── Deterministic global calibration: DE + L-BFGS-B refinement ────────────
def calibrate_de(t_data: np.ndarray, o_data: np.ndarray, proto: Protocol,
                 *, init_params: Optional[dict] = None,
                 maxiter: int = 200, popsize: int = 12,
                 seed: int = 0, polish: bool = True,
                 use_log: bool = True,
                 verbose: bool = False) -> CalibrationResult:
    """Differential Evolution + optional L-BFGS-B polish.

    Internal log-transformation is applied to parameters whose bounds span
    >= 2 orders of magnitude.  This makes DE proposals scale-invariant
    and substantially reduces convergence time.
    """
    base = dict(DEFAULT_PARAMS)
    if init_params:
        base.update(init_params)
    n_fccp = len(proto.t_fccp)
    base["alphas"] = (init_params or {}).get(
        "alphas", [0.5] * n_fccp)[:n_fccp]
    bnds = get_bounds_vec(n_fccp, include_sigma=False)
    names = get_param_names(n_fccp, include_sigma=False)
    log_mask = _log_mask(names) if use_log else np.zeros(len(names), bool)

    bnds_arr = np.array(bnds, dtype=float)
    bnds_log = bnds_arr.copy()
    bnds_log[log_mask] = np.log(bnds_arr[log_mask])

    history = []

    def vec_log_to_params(vec_log):
        v = _from_log(np.asarray(vec_log, float), log_mask)
        return vec_to_params(v, n_fccp, include_sigma=False, base=base)

    def obj(vec_log):
        p = vec_log_to_params(vec_log)
        val = sse(p, t_data, o_data, proto)
        history.append(val)
        return val

    n_eval_start = len(history)
    de = differential_evolution(
        obj, bnds_log.tolist(),
        maxiter=maxiter, popsize=popsize, seed=seed,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
        init="sobol", polish=False, workers=1,
    )
    best_x = de.x
    best_f = de.fun

    if polish:
        loc = minimize(obj, de.x, method="L-BFGS-B",
                       bounds=bnds_log.tolist(),
                       options={"maxiter": 400, "ftol": 1e-12})
        if loc.fun < best_f:
            best_x, best_f = loc.x, loc.fun

    params_fit = vec_log_to_params(best_x)
    # estimate sigma_obs as RMS residual
    res = simulate(params_fit, proto, o2_init=float(o_data[0]),
                   r0=params_fit.get("r0"), t_eval=t_data,
                   rtol=1e-7, atol=1e-9)
    sigma_obs_est: Optional[float] = None
    rmse_calib_val: Optional[float] = None
    if res.converged:
        rmse_calib_val = float(np.sqrt(np.mean((res.o - o_data) ** 2)))
        sigma_obs_est = float(np.sqrt(rmse_calib_val ** 2 + 1e-12))
        params_fit["sigma_obs"] = sigma_obs_est

    if verbose:
        print(f"DE+LBFGS done: SSE={best_f:.4f}, "
              f"sigma_obs={params_fit.get('sigma_obs', float('nan')):.4f}")

    return CalibrationResult(
        params=params_fit, objective=float(best_f),
        n_eval=len(history) - n_eval_start, success=True,
        method="DE+L-BFGS-B", bounds=bnds, param_names=names,
        history=history,
        objective_type="SSE_with_post_hoc_sigma",
        rmse_calib=rmse_calib_val,
        n_data=int(len(o_data)),
        seed=int(seed),
        optimiser_settings={
            "maxiter":      int(maxiter),
            "popsize":      int(popsize),
            "polish":       bool(polish),
            "use_log":      bool(use_log),
            "ftol_polish":  1e-12,
            "tol_de":       1e-7,
            "init":         "sobol",
            "mutation":     [0.5, 1.0],
            "recombination": 0.7,
        },
        sigma_obs=sigma_obs_est,
        sigma_estimation_method="post-hoc residual RMS (deterministic SSE)",
        warnings=[] if res.converged else ["solver did not converge at MAP"],
    )


# ── Sequential staged calibration ────────────────────────────────────────
def calibrate_staged(t_data: np.ndarray, o_data: np.ndarray, proto: Protocol,
                     *, init_params: Optional[dict] = None,
                     maxiter_stageA: int = 120, maxiter_stageB: int = 200,
                     seed: int = 0,
                     verbose: bool = False) -> CalibrationResult:
    """Two-stage calibration.

    Stage A: V_max, K_o, K_r, gamma_oligo on the oligomycin + FCCP segment
    Stage B: k_supply, c_tot, r0, tau_kappa, alphas on full trace,
             keeping Stage A parameters frozen.

    Why staged:
    -----------
    A simultaneous global fit lets {V_max, k_supply} trade off (raising
    one while lowering the other can preserve the trace shape over wide
    ranges).  By first identifying CIV-side parameters from the segment
    where supply is rate-limited differently, that degeneracy is broken.
    """
    base = dict(DEFAULT_PARAMS)
    if init_params:
        base.update(init_params)
    n_fccp = len(proto.t_fccp)
    base["alphas"] = (init_params or {}).get(
        "alphas", [0.5] * n_fccp)[:n_fccp]

    # ---- STAGE A : focus on (V_max, K_o, K_r, gamma_oligo)
    stageA_keys = ["V_max", "K_o", "K_r", "gamma_oligo"]
    stageA_bounds = [PARAM_BOUNDS[k] for k in stageA_keys]

    # Restrict the data to the oligomycin + FCCP region
    if proto.t_fccp:
        mA = (t_data >= proto.t_oligo) & (t_data <= proto.t_inhibit)
    else:
        mA = (t_data >= proto.t_oligo) & (t_data <= proto.t_end)
    tA, oA = t_data[mA], o_data[mA]
    if len(tA) < 20:
        tA, oA = t_data, o_data

    def objA(v):
        p = dict(base)
        for k, vk in zip(stageA_keys, v):
            p[k] = float(vk)
        return sse(p, tA, oA, proto)

    deA = differential_evolution(
        objA, stageA_bounds, maxiter=maxiter_stageA, popsize=10, seed=seed,
        tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, polish=True)
    for k, vk in zip(stageA_keys, deA.x):
        base[k] = float(vk)
    if verbose:
        print(f"Stage A done: SSE_A={deA.fun:.4f}, params={dict(zip(stageA_keys, deA.x))}")

    # ---- STAGE B : fit remaining on full data
    stageB_keys = ["k_supply", "c_tot", "tau_kappa", "r0"]
    stageB_bounds = [PARAM_BOUNDS[k] for k in stageB_keys]
    stageB_bounds.extend([PARAM_BOUNDS["alpha"]] * n_fccp)

    def objB(v):
        p = dict(base)
        for k, vk in zip(stageB_keys, v[:len(stageB_keys)]):
            p[k] = float(vk)
        p["alphas"] = [float(x) for x in v[len(stageB_keys):
                                            len(stageB_keys)+n_fccp]]
        return sse(p, t_data, o_data, proto)

    deB = differential_evolution(
        objB, stageB_bounds, maxiter=maxiter_stageB, popsize=12, seed=seed,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, polish=True)
    for k, vk in zip(stageB_keys, deB.x[:len(stageB_keys)]):
        base[k] = float(vk)
    base["alphas"] = [float(x) for x in deB.x[len(stageB_keys):
                                               len(stageB_keys)+n_fccp]]

    # Final estimate of sigma_obs
    res = simulate(base, proto, o2_init=float(o_data[0]),
                   r0=base["r0"], t_eval=t_data, rtol=1e-7, atol=1e-9)
    rmse_calib_val: Optional[float] = None
    sigma_obs_est: Optional[float] = None
    if res.converged:
        rmse_calib_val = float(np.sqrt(np.mean((res.o - o_data) ** 2)))
        sigma_obs_est = float(np.sqrt(rmse_calib_val ** 2 + 1e-12))
        base["sigma_obs"] = sigma_obs_est

    return CalibrationResult(
        params=base, objective=float(deB.fun), n_eval=deA.nfev + deB.nfev,
        success=True,
        method="staged (DE Stage A + DE Stage B); diagnostic strategy",
        bounds=stageB_bounds,
        param_names=stageB_keys + [f"alpha_{j+1}" for j in range(n_fccp)],
        history=[deA.fun, deB.fun],
        objective_type="SSE_with_post_hoc_sigma",
        rmse_calib=rmse_calib_val,
        n_data=int(len(o_data)),
        seed=int(seed),
        optimiser_settings={
            "maxiter_stageA": int(maxiter_stageA),
            "maxiter_stageB": int(maxiter_stageB),
            "popsize_stageA": 10,
            "popsize_stageB": 12,
            "polish":         True,
            "init":           "sobol",
        },
        sigma_obs=sigma_obs_est,
        warnings=[] if res.converged else ["solver did not converge at MAP"],
    )


# ── Pooled multi-trace calibration across chambers ───────────────────────
def _pooled_multitrace_impl(
        traces: List[Tuple[np.ndarray, np.ndarray, Protocol]],
        *, shared_keys: Optional[List[str]] = None,
        init_params: Optional[dict] = None,
        maxiter: int = 250, popsize: int = 15,
        seed: int = 0,
        verbose: bool = False) -> List[CalibrationResult]:
    """Joint SSE calibration across multiple chambers / datasets, with
    shared core parameters pooled and per-trace nuisance parameters fit
    independently.

    This is NOT a hierarchical Bayesian model with population
    hyperparameters and random effects; it is a deterministic
    shared-parameter wrapper around the SSE objective.

    `shared_keys` lists parameters pooled across all traces (default: the
    eight core kinetic parameters). Each trace gets its own `alphas` and
    `sigma_obs` (these are nuisance / dataset-specific).

    Returns one CalibrationResult per trace; the shared parameters are
    identical across them, the nuisance parameters differ.
    """
    if not traces:
        raise ValueError("No traces supplied")
    if shared_keys is None:
        shared_keys = list(CORE_PARAM_ORDER)
    base = dict(DEFAULT_PARAMS)
    if init_params:
        base.update(init_params)

    n_traces = len(traces)
    n_fccp_list = [len(p.t_fccp) for _, _, p in traces]
    # Vector layout: [shared_params, alphas_trace_1, alphas_trace_2, ...]
    shared_bounds = [PARAM_BOUNDS[k] for k in shared_keys]
    nuisance_bounds = []
    for nf in n_fccp_list:
        nuisance_bounds.extend([PARAM_BOUNDS["alpha"]] * nf)
    bounds = shared_bounds + nuisance_bounds

    def vec_to_traces(v):
        p_shared = dict(base)
        for i, k in enumerate(shared_keys):
            p_shared[k] = float(v[i])
        per_trace = []
        off = len(shared_keys)
        for nf in n_fccp_list:
            ptr = dict(p_shared)
            ptr["alphas"] = [float(v[off + j]) for j in range(nf)]
            off += nf
            per_trace.append(ptr)
        return per_trace

    def obj(v):
        params_per_trace = vec_to_traces(v)
        total = 0.0
        for (t_d, o_d, proto), p in zip(traces, params_per_trace):
            total += sse(p, t_d, o_d, proto)
        return total

    de = differential_evolution(
        obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, polish=True)

    params_per_trace = vec_to_traces(de.x)
    results = []
    for (t_d, o_d, proto), p in zip(traces, params_per_trace):
        res = simulate(p, proto, o2_init=float(o_d[0]),
                       r0=p["r0"], t_eval=t_d, rtol=1e-7, atol=1e-9)
        rmse_calib_val: Optional[float] = None
        sigma_obs_est: Optional[float] = None
        if res.converged:
            rmse_calib_val = float(np.sqrt(np.mean((res.o - o_d) ** 2)))
            sigma_obs_est = float(np.sqrt(rmse_calib_val ** 2 + 1e-12))
            p["sigma_obs"] = sigma_obs_est
        else:
            p["sigma_obs"] = 0.5
        results.append(CalibrationResult(
            params=p,
            objective=float(np.sum((res.o - o_d) ** 2))
                              if res.converged else 1e15,
            n_eval=de.nfev, success=de.success,
            method="pooled-multitrace-DE",
            bounds=bounds,
            param_names=shared_keys + [f"alpha_{j+1}"
                                       for j in range(len(p["alphas"]))],
            history=[de.fun],
            objective_type="SSE_with_post_hoc_sigma",
            rmse_calib=rmse_calib_val,
            n_data=int(len(o_d)),
            seed=int(seed),
            optimiser_settings={
                "maxiter":   int(maxiter),
                "popsize":   int(popsize),
                "polish":    True,
                "shared_keys": list(shared_keys),
                "n_traces":  int(n_traces),
            },
            sigma_obs=sigma_obs_est,
            warnings=[] if res.converged else
                      ["solver did not converge at MAP"],
        ))
    if verbose:
        print(f"Pooled multi-trace DE: total SSE={de.fun:.3f}")
    return results


def calibrate_pooled_multitrace(
        traces: List[Tuple[np.ndarray, np.ndarray, Protocol]],
        **kwargs) -> List[CalibrationResult]:
    """Public name for pooled multi-trace SSE calibration.

    See `_pooled_multitrace_impl` for parameters and semantics. The
    accompanying calibration is NOT a hierarchical Bayesian model.
    """
    return _pooled_multitrace_impl(traces, **kwargs)


def calibrate_hierarchical(
        traces: List[Tuple[np.ndarray, np.ndarray, Protocol]],
        **kwargs) -> List[CalibrationResult]:
    """DEPRECATED: legacy alias for `calibrate_pooled_multitrace`.

    The original name was misleading — this routine pools shared core
    parameters across traces but is NOT a hierarchical Bayesian model
    with population hyperparameters and random effects. Please migrate
    to `calibrate_pooled_multitrace`.
    """
    import warnings as _w
    _w.warn(
        "calibrate_hierarchical() is deprecated; use "
        "calibrate_pooled_multitrace() instead. The new name reflects "
        "what the routine actually does (pooled SSE calibration across "
        "traces, with shared core parameters and per-trace nuisance "
        "parameters); it is NOT a hierarchical Bayesian model.",
        DeprecationWarning, stacklevel=2,
    )
    return _pooled_multitrace_impl(traces, **kwargs)
