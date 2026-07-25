"""
core/diagnostics.py
===================
Numerical stability diagnostics for the 3-state OCR-informed bioenergetics model.

Provides:
  * stiffness_index(): max-eigenvalue Jacobian estimate over the trace
  * conservation_check(): cyt-c pool drift and oxygen monotonicity
  * tolerance_sensitivity(): OCR shift across rtol/atol grid
  * parameter_robustness_sweep(): convergence at parameter-bound corners
  * detect_instability(): single-call diagnostic used by the AI agent

All routines operate on the 3-state OCR-informed model in core.reduced_model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reduced_model import (
    Protocol, simulate, rhs, DEFAULT_PARAMS,
    PARAM_BOUNDS, CORE_PARAM_ORDER,
)


# ── Result containers ────────────────────────────────────────────────────
@dataclass
class StabilityReport:
    converged: bool = True
    max_jacobian_eig: float = 0.0           # max real eigenvalue magnitude
    stiffness_ratio: float = 1.0            # |lambda_max| / |lambda_min|
    cytc_conservation_drift: float = 0.0    # max |r + c_ox - c_tot| over trace
    oxygen_monotone: bool = True            # do/dt <= 0 at all times
    kappa_in_range: bool = True             # kappa stays in a finite, data-dependent range
                                            # determined by gamma_oligo and cumulative alpha_j;
                                            # NOT bounded to [0, 1] -- FCCP can push it above 1.
    nan_count: int = 0
    negative_state_count: int = 0
    solver_tolerance_robust: bool = True
    warnings: List[str] = field(default_factory=list)
    n_steps: int = 0

    def is_healthy(self) -> bool:
        return (self.converged
                and self.cytc_conservation_drift < 1.0
                and self.oxygen_monotone
                and self.kappa_in_range
                and self.nan_count == 0
                and self.negative_state_count == 0)

    def summary(self) -> str:
        flag = "OK " if self.is_healthy() else "FAIL"
        return (f"[{flag}] eig_max={self.max_jacobian_eig:.2e}  "
                f"stiff_ratio={self.stiffness_ratio:.1f}  "
                f"drift={self.cytc_conservation_drift:.3f}  "
                f"o-mono={self.oxygen_monotone}  "
                f"NaN={self.nan_count}  neg={self.negative_state_count}")


# ── Numerical Jacobian by central differences ───────────────────────────
def numerical_jacobian(t: float, y: np.ndarray, params: dict, proto: Protocol,
                       eps: float = 1e-6) -> np.ndarray:
    """3x3 Jacobian of the RHS at (t, y)."""
    n = len(y)
    J = np.zeros((n, n))
    f0 = rhs(t, y, params, proto)
    for i in range(n):
        h = max(eps * abs(y[i]), eps)
        yp = y.copy(); yp[i] += h
        ym = y.copy(); ym[i] -= h
        fp = rhs(t, yp, params, proto)
        fm = rhs(t, ym, params, proto)
        J[:, i] = (fp - fm) / (2 * h)
    return J


# ── Stiffness analysis ───────────────────────────────────────────────────
def stiffness_analysis(params: dict, proto: Protocol,
                       o2_init: float = 170.0,
                       n_samples: int = 30) -> Tuple[float, float, np.ndarray]:
    """Estimate stiffness across the trace.

    Returns (max_|Re(lambda)|, stiffness_ratio, eigenvalue_array_over_time).
    Stiffness ratio = max|Re(lambda)| / min|Re(lambda)| over all sampled t.
    A ratio < 1e3 means non-stiff and any explicit RK works; > 1e6 means
    we should stick with LSODA / BDF.
    """
    res = simulate(params, proto, o2_init=o2_init,
                   t_eval=np.linspace(proto.t_start, proto.t_end, n_samples))
    if not res.converged:
        return np.inf, np.inf, np.array([])

    eig_per_t = []
    for i, t in enumerate(res.t):
        y = np.array([res.r[i], res.o[i], res.kappa[i]])
        J = numerical_jacobian(t, y, params, proto)
        eigs = np.linalg.eigvals(J)
        eig_per_t.append(np.abs(eigs.real))
    eig_arr = np.array(eig_per_t)              # (n_samples, 3)
    eig_arr = eig_arr[np.isfinite(eig_arr).all(axis=1)]
    if eig_arr.size == 0:
        return np.inf, np.inf, np.array([])
    max_eig = float(eig_arr.max())
    nonzero = eig_arr[eig_arr > 1e-15]
    min_eig = float(nonzero.min()) if nonzero.size else 0.0
    ratio = (max_eig / min_eig) if min_eig > 0 else np.inf
    return max_eig, ratio, eig_arr


# ── Conservation check ───────────────────────────────────────────────────
def conservation_check(res, params: dict) -> Dict[str, float]:
    """Check cyt-c pool conservation, oxygen monotonicity, kappa boundedness.

    IMPORTANT: this diagnostic inspects the RAW (unclipped) solver states
    (`res.r_raw`, `res.o_raw`, `res.kappa_raw`). The clipped presentation
    states (`res.r`, `res.o`, `res.kappa`) are sanitized to physical ranges
    and would make `drift` and the negative-state checks vacuously pass.
    The whole point of this check is to detect non-physical excursions in
    the actual integrator output, so it must see the raw arrays.

    Conservation should hold exactly only before t_inhibit (since the
    smooth supply switch leaks tiny amounts during the transition).
    """
    if not res.converged:
        return {"drift": np.inf, "o_monotone": False, "kappa_in_range": False,
                "raw_negative_r_count": -1, "raw_negative_o_count": -1,
                "raw_kappa_min": np.nan, "kappa_upper": np.nan,
                "kappa_max_observed": np.nan}

    # Use RAW solver states for every physical check below.
    r_raw = np.asarray(res.r_raw, dtype=float)
    o_raw = np.asarray(res.o_raw, dtype=float)
    k_raw = np.asarray(res.kappa_raw, dtype=float)

    # cyt c oxidized = c_tot - r should be in [0, c_tot]. The drift is the
    # largest excursion of the RAW oxidized pool outside its physical band.
    cyt_ox_raw = params["c_tot"] - r_raw
    drift = float(np.max(np.abs(
        np.clip(cyt_ox_raw, 0.0, params["c_tot"]) - cyt_ox_raw)))

    # oxygen monotone non-increasing (raw output, noise-tolerant)
    o_diff = np.diff(o_raw)
    o_monotone = bool(np.all(o_diff <= 1e-3))

    # kappa upper-bound is data-dependent: 1 + sum(alpha_j) is the
    # equilibrium value with all FCCP steps active.  We tolerate +20% for
    # transient overshoot during smooth transitions, and require >= 0.
    alpha_sum = sum(params.get("alphas", []))
    k_upper   = 1.0 + alpha_sum + 0.2 * (1.0 + alpha_sum)
    kappa_ok  = bool(np.all(k_raw >= -1e-6) and np.all(k_raw <= k_upper))

    # Honest negative-state counts on the RAW arrays (a small tolerance
    # absorbs solver round-off; genuine excursions are still caught).
    raw_neg_r = int(np.sum(r_raw < -1e-6))
    raw_neg_o = int(np.sum(o_raw < -1e-6))

    return {"drift": drift, "o_monotone": o_monotone,
            "kappa_in_range": kappa_ok, "kappa_upper": k_upper,
            "kappa_max_observed": float(np.max(k_raw)),
            "raw_kappa_min": float(np.min(k_raw)),
            "raw_negative_r_count": raw_neg_r,
            "raw_negative_o_count": raw_neg_o}


# ── Solver-tolerance robustness ──────────────────────────────────────────
def tolerance_sensitivity(params: dict, proto: Protocol,
                          o2_init: float = 170.0) -> Dict[str, float]:
    """Compare OCR(t) at three tolerance levels.

    Reports the max(|OCR_loose - OCR_tight|) / max(OCR_tight) over the trace.
    A robust integration shows < 1% relative deviation across 4 orders of
    magnitude in atol.
    """
    t_eval = np.linspace(proto.t_start, proto.t_end, 400)
    res_tight = simulate(params, proto, o2_init=o2_init, t_eval=t_eval,
                         rtol=1e-9, atol=1e-12)
    res_loose = simulate(params, proto, o2_init=o2_init, t_eval=t_eval,
                         rtol=1e-4, atol=1e-7)
    if not (res_tight.converged and res_loose.converged):
        return {"rel_diff": np.inf, "robust": False}
    scale = max(float(np.max(res_tight.OCR)), 1e-12)
    rel_diff = float(np.max(np.abs(res_loose.OCR - res_tight.OCR)) / scale)
    return {"rel_diff": rel_diff, "robust": rel_diff < 0.01}


# ── Parameter-robustness sweep ───────────────────────────────────────────
def parameter_robustness_sweep(proto: Protocol, o2_init: float = 170.0,
                               n_samples: int = 50,
                               seed: int = 0) -> Dict[str, float]:
    """Sample parameters log-uniformly across their full bounds and check
    that the integrator converges and produces finite traces.

    A robust model converges on >99% of samples; this is reviewer-defensible
    proof that the reduction did not introduce a region of pathological
    behaviour.
    """
    rng = np.random.default_rng(seed)
    n_fccp = len(proto.t_fccp)
    n_ok = 0
    n_fail = 0
    n_finite = 0
    for _ in range(n_samples):
        p = dict(DEFAULT_PARAMS)
        for k in CORE_PARAM_ORDER:
            lo, hi = PARAM_BOUNDS[k]
            p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        a_lo, a_hi = PARAM_BOUNDS["alpha"]
        p["alphas"] = [float(np.exp(rng.uniform(np.log(a_lo), np.log(a_hi))))
                       for _ in range(n_fccp)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = simulate(p, proto, o2_init=o2_init, rtol=1e-6, atol=1e-9)
            # Finiteness is judged on the RAW solver output so that a
            # diverging trajectory cannot be masked by presentation clipping.
            o_raw = np.asarray(res.o_raw, dtype=float)
            r_raw = np.asarray(res.r_raw, dtype=float)
            if res.converged and np.isfinite(o_raw).all() \
                    and np.isfinite(r_raw).all():
                n_ok += 1
                if np.isfinite(res.OCR).all() and (res.OCR >= -1e-6).all():
                    n_finite += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    return {"convergence_rate": n_ok / n_samples,
            "finite_OCR_rate":  n_finite / n_samples,
            "n_failures":       n_fail,
            "n_samples":        n_samples}


# ── Top-level instability detector (used by the AI agent) ─────────────────
def detect_instability(params: dict, proto: Protocol,
                       o2_init: float = 170.0) -> StabilityReport:
    """Single-call diagnostic that returns a StabilityReport.

    Used by the AI agent to flag suspect calibrations or simulations
    before reporting them to the user.
    """
    rep = StabilityReport()
    try:
        res = simulate(params, proto, o2_init=o2_init,
                       t_eval=np.linspace(proto.t_start, proto.t_end, 400),
                       rtol=1e-7, atol=1e-9)
    except Exception as e:
        rep.converged = False
        rep.warnings.append(f"simulate raised: {e!r}")
        return rep

    rep.converged = res.converged
    rep.n_steps = len(res.t)

    if not res.converged:
        rep.warnings.append("solver failed to converge")
        return rep

    # NaN and negative-state counts are computed on the RAW solver output.
    # The clipped presentation arrays (res.o/res.r/res.kappa) would make
    # the negative-state count vacuously zero. A small -1e-6 tolerance
    # absorbs solver round-off; genuine non-physical excursions are caught.
    r_raw = np.asarray(res.r_raw, dtype=float)
    o_raw = np.asarray(res.o_raw, dtype=float)
    k_raw = np.asarray(res.kappa_raw, dtype=float)
    rep.nan_count = int(np.sum(~np.isfinite(o_raw))) \
                  + int(np.sum(~np.isfinite(r_raw))) \
                  + int(np.sum(~np.isfinite(k_raw)))
    rep.negative_state_count = int(np.sum(o_raw < -1e-6)) \
                              + int(np.sum(r_raw < -1e-6))
    if rep.negative_state_count > 0:
        rep.warnings.append(
            f"{rep.negative_state_count} raw solver state value(s) went "
            f"negative beyond tolerance; trajectory is non-physical "
            f"(presentation arrays are clipped, raw arrays are not)")

    cons = conservation_check(res, params)
    rep.cytc_conservation_drift = cons["drift"]
    rep.oxygen_monotone        = cons["o_monotone"]
    rep.kappa_in_range          = cons["kappa_in_range"]
    if not cons["kappa_in_range"]:
        rep.warnings.append(
            f"raw kappa left its tolerated range "
            f"[{-1e-6:.0e}, {cons['kappa_upper']:.3g}] "
            f"(min={cons.get('raw_kappa_min', float('nan')):.3g}, "
            f"max={cons.get('kappa_max_observed', float('nan')):.3g})")
    if cons["drift"] > 1.0:
        rep.warnings.append(
            f"cyt-c oxidized pool drifted {cons['drift']:.3g} nmol/mL "
            f"outside [0, c_tot] in the raw solver output")

    try:
        max_eig, ratio, _ = stiffness_analysis(params, proto, o2_init,
                                               n_samples=10)
        rep.max_jacobian_eig  = max_eig
        rep.stiffness_ratio    = ratio
    except Exception as e:
        rep.warnings.append(f"stiffness analysis failed: {e!r}")

    try:
        tol = tolerance_sensitivity(params, proto, o2_init)
        rep.solver_tolerance_robust = tol["robust"]
        if not tol["robust"]:
            rep.warnings.append(
                f"OCR shifts {tol['rel_diff']*100:.2f}% across tolerances")
    except Exception as e:
        rep.warnings.append(f"tolerance test failed: {e!r}")

    return rep


# ── Audit table generator (for manuscript Methods section) ────────────────
STABILITY_AUDIT = [
    {
        "issue": r"Unmeasured proton-gradient/proton-ratio terms",
        "design_risk": "Division or multiplicative dependence on unmeasured proton states can become numerically singular and biologically underdetermined from OCR alone.",
        "resolution": r"Represent unmeasured protonmotive effects with the latent effective respiratory-drive factor $\kappa(t)$; do not interpret $\kappa$ as measured membrane potential or true protonmotive force.",
    },
    {
        "issue": "CIV activation terms with near-zero denominators",
        "design_risk": "Terms that divide by a state approaching zero can destabilize integration and profile-likelihood optimization.",
        "resolution": "Use bounded separable saturation functions for oxygen and the reduced pool: $o/(K_o+o)$ and $r/(K_r+r)$.",
    },
    {
        "issue": "Explicit proton-leak state equations without proton observables",
        "design_risk": "OCR-only data cannot uniquely estimate detailed proton-leak or pH dynamics.",
        "resolution": r"Fold protocol-dependent OCR permissiveness into $\kappa_{\mathrm{eq}}(t)$ and report the resulting parameters as phenomenological unless supported by identifiability diagnostics.",
    },
    {
        "issue": "Discontinuous injection functions",
        "design_risk": "Hard event switches break smooth trajectories and complicate sensitivity/profile-likelihood calculations.",
        "resolution": r"Represent each intervention by a smooth $\sigma_k(t-t_*) = \tfrac{1}{2}(1+\tanh(k(t-t_*)))$ transition.",
    },
    {
        "issue": "Discontinuous state resets at inhibition",
        "design_risk": "Instantaneous state rescaling introduces non-differentiability and a parameter not supported by OCR-only data.",
        "resolution": "Avoid state resets; the inhibition event smoothly gates supply and CIV-mediated OCR terms.",
    },
    {
        "issue": "Penalty-based constraints for unobserved physiology",
        "design_risk": "Large ad-hoc penalties can obscure fit quality and create discontinuous objectives.",
        "resolution": "Use explicit parameter bounds, smooth dynamics, diagnostic warnings, and identifiability flags rather than treating unmeasured physiological quantities as directly inferred.",
    },
    {
        "issue": "Hidden initial conditions for unmeasured states",
        "design_risk": "Initial values for unavailable biochemical states are not inferable from a single oxygen trace.",
        "resolution": r"Use only the measured initial oxygen value, an estimated initial reduced-pool value $r_0$, and the convention $\kappa(0)=1$ for the protocol baseline.",
    },
]
