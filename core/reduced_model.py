"""
core/reduced_model.py
=====================
3-state OCR-informed mitochondrial stress-test model.

This module implements the 3-state OCR-informed model used by the current manuscript.
The model is intentionally centred on the observable available from
Oroboros/Seahorse-style stress-test respirometry: oxygen concentration
and the derived oxygen-consumption rate (OCR). It is not a full
mechanistic reconstruction of the electron-transport chain, proton
transport, membrane potential, redox state, pH, or isolated Complex IV
enzyme kinetics.

Scientific framing
------------------
The model represents CIV-mediated oxygen consumption in intact
respiratory-chain context. Upstream electron supply and unmeasured
protonmotive effects are represented phenomenologically so that the
resulting equations can be calibrated and diagnostically interrogated
from OCR-only data. Parameter interpretation is conditional on explicit
calibration, numerical-stability, practical-identifiability, sensitivity,
and validation diagnostics; no parameter is treated as biologically
interpretable by assumption.

State variables
---------------
r(t)      [nmol/mL]  reduced cytochrome-c pool or effective reductant pool
o(t)      [nmol/mL]  oxygen concentration, the direct observable
kappa(t)  [-]        latent effective respiratory-drive / OCR-permissiveness
                     factor. kappa is not measured membrane potential,
                     not true protonmotive force, and not a direct pH,
                     redox, or proton-gradient readout. It can fall below
                     1 after oligomycin and can exceed 1 after FCCP.

Core kinetic parameters
-----------------------
k_supply      [1/s]        lumped upstream supply rate for the effective
                           reduced pool
c_tot         [nmol/mL]    total effective cytochrome-c/reductant pool
V_max         [nmol/mL/s]  maximal effective CIV-mediated OCR capacity
K_o           [nmol/mL]    oxygen half-saturation constant
K_r           [nmol/mL]    reduced-pool half-activation constant
gamma_oligo   [-]          oligomycin attenuation factor for kappa_eq
tau_kappa     [s]          relaxation time of kappa toward protocol state
r0            [nmol/mL]    initial reduced-pool value

Protocol-specific nuisance parameters
-------------------------------------
alpha_j       [-]          FCCP response amplitude for injection j
sigma_obs     [nmol/mL]    within-trace observational noise used by
                           likelihood/reporting code; not used in the ODE

The module deliberately avoids discontinuous state resets and uses smooth
tanh injection functions for differentiable ODE trajectories. The formulation is phenomenological and non-invertible
with respect to any more detailed biochemical model because unmeasured
states have been collapsed by design.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp


# ── Smooth Heaviside ──────────────────────────────────────────────────────
def smooth_step(t: float, t0: float, k: float = 2.0) -> float:
    """Smooth approximation to the Heaviside step.

    sigma(t-t0; k) = 0.5 * (1 + tanh(k*(t-t0)))

    k controls steepness; k=2 gives a transition width of ~1 s,
    fast enough to capture injections without breaking the ODE solver.
    """
    return 0.5 * (1.0 + np.tanh(k * (t - t0)))


# ── Protocol descriptor ───────────────────────────────────────────────────
@dataclass
class Protocol:
    """Experimental protocol with injection time points.

    Times are in seconds (absolute, in the recording's clock).
    """
    t_oligo: float
    t_fccp: List[float]
    t_inhibit: float
    t_end: float
    t_start: float = 0.0
    k_step: float = 2.0   # smoothness of step transitions

    @property
    def n_fccp(self) -> int:
        return len(self.t_fccp)


# ── ODE right-hand side ───────────────────────────────────────────
def rhs(t: float, y: np.ndarray, p: dict, proto: Protocol) -> np.ndarray:
    """Right-hand side of the 3-state ODE model.

    The drive factor kappa is a LATENT effective respiratory-drive factor
    (NOT a measurement of membrane potential or true protonmotive force).
    It is NOT bounded to [0, 1]: kappa = 1 at baseline (State 3 with ADP),
    drops below 1 under oligomycin to gamma_oligo, and can rise above 1
    after FCCP additions if the cumulative alpha_j is large.

    Sign conventions (consistent with Seahorse-style respirometry):
      * Baseline (State 3, with ADP):  kappa_eq = 1
      * Oligomycin blocks ATP synthase -> proton-gradient back-pressure
        slows respiration: kappa_eq = gamma_oligo, in (0, 1]
      * Each FCCP step partially uncouples the membrane, releasing
        back-pressure: kappa_eq increases by alpha_j (cumulative may
        exceed 1)
      * Rotenone+antimycin block electron supply -> all fluxes are
        smoothly switched off via s_inh

    Numerical stability:
      * Michaelis denominators K + x have constant K > 0, avoiding
        division by unmeasured or near-zero latent biochemical quantities.
      * All step functions are tanh-smoothed (C^infinity).
      * Defensive non-negativity is applied to states inside the RHS;
        smooth dynamics already keep them physical.
    """
    r, o, kappa = y[0], y[1], y[2]
    r_pos = max(r, 0.0)
    o_pos = max(o, 0.0)
    k_pos = max(kappa, 0.0)

    k_step = proto.k_step

    # Inhibition switch  -> 1 before, 0 after t_inhibit
    s_inh = 1.0 - smooth_step(t, proto.t_inhibit, k_step)

    # Equilibrium drive factor as a function of injections
    s_oligo = smooth_step(t, proto.t_oligo, k_step)
    fccp_sum = 0.0
    for tj, aj in zip(proto.t_fccp, p["alphas"]):
        fccp_sum += aj * smooth_step(t, tj, k_step)
    # Baseline = 1, oligomycin reduces to gamma_oligo, FCCP adds alpha_j.
    # kappa_eq is NOT bounded to [0, 1] -- the cumulative FCCP contribution
    # can push it above 1 if alpha_j sums above (1 - gamma_oligo*s_oligo).
    kappa_eq = 1.0 - (1.0 - p["gamma_oligo"]) * s_oligo + fccp_sum

    # CIV flux: Michaelis in O2, Hill-1 activation in r, modulated by kappa.
    # f_o, f_r are bounded fractions in [0, 1]; k_pos is non-negative but
    # NOT capped at 1.
    f_o = o_pos / (p["K_o"] + o_pos + 1e-12)
    f_r = r_pos / (p["K_r"] + r_pos + 1e-12)
    v_CIV = p["V_max"] * f_o * f_r * k_pos * s_inh

    # Supply of reduced cyt c by Complexes I-III (lumped pseudo-first-order)
    cyt_ox = max(p["c_tot"] - r_pos, 0.0)
    v_supply = p["k_supply"] * cyt_ox * s_inh

    # State derivatives
    dr = 2.0 * v_supply - 2.0 * v_CIV
    do = -0.5 * v_CIV
    dkappa = (kappa_eq - kappa) / max(p["tau_kappa"], 1e-3)

    return np.array([dr, do, dkappa], dtype=float)


# ── Simulation result container ───────────────────────────────────────────
@dataclass
class SimulationResult:
    t: np.ndarray
    r: np.ndarray              # presentation states: clipped to physical ranges
    o: np.ndarray
    kappa: np.ndarray
    OCR: np.ndarray            # nmol/mL/s (positive); computed from clipped states
    params: dict
    protocol: Protocol
    converged: bool = True
    # ── RAW solver output, BEFORE any clipping ───────────────────────────
    # Diagnostics (conservation drift, negative-state counts, monotonicity)
    # MUST inspect these raw arrays. The clipped `r`/`o`/`kappa` above are
    # sanitized for presentation and calibration and would otherwise mask
    # genuine non-physical solver excursions. When the solver did not
    # converge, the raw fields are NaN-filled like the clipped fields.
    r_raw:     Optional[np.ndarray] = None
    o_raw:     Optional[np.ndarray] = None
    kappa_raw: Optional[np.ndarray] = None

    def __post_init__(self):
        # Back-fill raw fields for any caller / legacy path that constructed
        # a SimulationResult without them, so downstream code can always
        # rely on r_raw/o_raw/kappa_raw being present.
        if self.r_raw is None:
            self.r_raw = np.asarray(self.r, dtype=float)
        if self.o_raw is None:
            self.o_raw = np.asarray(self.o, dtype=float)
        if self.kappa_raw is None:
            self.kappa_raw = np.asarray(self.kappa, dtype=float)

    def seahorse_metrics(self) -> dict:
        """Standard respirometry summary statistics computed from the
        analytic OCR trace (no finite differences)."""
        proto = self.protocol
        if not proto.t_fccp:
            return {}
        win = max((proto.t_fccp[0] - proto.t_oligo) * 0.3, 5.0)

        def _mean(lo, hi):
            m = (self.t >= lo) & (self.t < hi)
            return float(np.mean(self.OCR[m])) if m.any() else 0.0

        basal = _mean(proto.t_oligo - 2*win, proto.t_oligo)
        oligo_ocr = _mean(proto.t_oligo + win, proto.t_fccp[0])
        maximal = _mean(proto.t_fccp[-1] + win, proto.t_inhibit - win)
        nonmito = _mean(proto.t_inhibit + win, proto.t_end)
        atp = max(basal - oligo_ocr, 0.0)
        leak = max(oligo_ocr - nonmito, 0.0)
        spare = max(maximal - basal, 0.0)
        coupling = (atp / basal * 100.0) if basal > 0 else 0.0
        return {
            "Basal OCR": basal,
            "ATP-linked OCR": atp,
            "Proton Leak OCR": leak,
            "Maximal OCR": maximal,
            "Spare Respiratory Capacity": spare,
            "Non-Mitochondrial OCR": nonmito,
            "Coupling Efficiency (%)": coupling,
        }


# ── Forward simulation ────────────────────────────────────────────────────
def simulate(params: dict, proto: Protocol,
             o2_init: float = 170.0,
             r0: Optional[float] = None,
             kappa0: float = 1.0,
             t_eval: Optional[np.ndarray] = None,
             rtol: float = 1e-7,
             atol: float = 1e-9,
             method: str = "LSODA") -> SimulationResult:
    """Integrate the ODE system over the protocol.

    Parameters
    ----------
    params : dict   -- must contain the 8 core kinetic parameters
        k_supply, c_tot, V_max, K_o, K_r, gamma_oligo, tau_kappa, r0
        plus alphas (list, length n_fccp) and (optionally) sigma_obs.
        sigma_obs is used by the likelihood, NOT by the RHS.
    proto : Protocol
    o2_init : initial O2 concentration in nmol/mL
    r0 : initial reduced cyt c (overrides params["r0"] if given)
    kappa0 : initial value of the effective respiratory-drive factor.
        Default 1.0 corresponds to the State-3 baseline.
    """
    if t_eval is None:
        t_eval = np.linspace(proto.t_start, proto.t_end, 500)

    r_init = float(r0 if r0 is not None else params["r0"])
    y0 = np.array([r_init, float(o2_init), float(kappa0)])

    sol = solve_ivp(
        rhs, (proto.t_start, proto.t_end), y0,
        method=method, t_eval=t_eval, args=(params, proto),
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        nan = np.full_like(t_eval, np.nan, dtype=float)
        return SimulationResult(t_eval, nan, nan, nan, nan,
                                params, proto, converged=False,
                                r_raw=nan.copy(), o_raw=nan.copy(),
                                kappa_raw=nan.copy())

    # RAW solver output — kept verbatim for numerical diagnostics. These
    # arrays may legitimately contain small negative excursions or
    # conservation drift; the diagnostics in core/diagnostics.py inspect
    # exactly these arrays to detect non-physical solver behaviour.
    r_raw = np.asarray(sol.y[0], dtype=float)
    o_raw = np.asarray(sol.y[1], dtype=float)
    k_raw = np.asarray(sol.y[2], dtype=float)

    # CLIPPED presentation states — used for plotting, calibration residuals,
    # and the analytic OCR. Clipping keeps the reported trajectory physical;
    # it does NOT hide solver problems because the raw arrays are preserved.
    r_arr = np.clip(r_raw, 0.0, params["c_tot"])
    o_arr = np.clip(o_raw, 0.0, None)
    k_arr = np.clip(k_raw, 0.0, None)

    # Analytic OCR = 0.5 * v_CIV(r, o, kappa) using post-hoc evaluation,
    # not finite differences -- exact to within solver tolerance.
    s_inh_arr = np.array([1.0 - smooth_step(t, proto.t_inhibit, proto.k_step)
                          for t in t_eval])
    f_o = o_arr / (params["K_o"] + o_arr + 1e-12)
    f_r = r_arr / (params["K_r"] + r_arr + 1e-12)
    OCR = 0.5 * params["V_max"] * f_o * f_r * k_arr * s_inh_arr

    return SimulationResult(t_eval, r_arr, o_arr, k_arr, OCR,
                            params, proto, converged=True,
                            r_raw=r_raw, o_raw=o_raw, kappa_raw=k_raw)


# ── Default parameter dictionary (for testing/initialisation) ─────────────
DEFAULT_PARAMS = {
    "k_supply":   0.05,    # 1/s
    "c_tot":      200.0,   # nmol/mL
    "V_max":      1.5,     # nmol/mL/s
    "K_o":        5.0,     # nmol/mL
    "K_r":        50.0,    # nmol/mL
    "gamma_oligo":0.30,    # dimensionless, in (0,1]
    "tau_kappa":  5.0,     # s
    "r0":         100.0,   # nmol/mL
    "alphas":     [1.0],   # one FCCP step
    "sigma_obs":  0.5,     # nmol/mL  (for likelihood)
}


# ── Parameter bounds for calibration --------------------------------------
PARAM_BOUNDS = {
    "k_supply":    (1e-3,  10.0),    # very wide, log scale
    "c_tot":       (50.0,  1000.0),
    "V_max":       (1e-2,  20.0),
    "K_o":         (0.1,   100.0),
    "K_r":         (1.0,   500.0),
    "gamma_oligo": (1e-2,  1.0),
    "tau_kappa":   (0.5,   60.0),
    "r0":          (1.0,   600.0),
    "alpha":       (1e-3,  100.0),   # per FCCP step
    "sigma_obs":   (1e-3,  10.0),
}

# Order of parameters in the calibration vector.  Alphas and sigma_obs
# are appended at the end (variable count per dataset).
CORE_PARAM_ORDER = ["k_supply", "c_tot", "V_max", "K_o", "K_r",
                    "gamma_oligo", "tau_kappa", "r0"]


def params_to_vec(p: dict, n_fccp: int, include_sigma: bool = True) -> np.ndarray:
    v = [p[k] for k in CORE_PARAM_ORDER]
    v.extend(p["alphas"][:n_fccp])
    if include_sigma:
        v.append(p.get("sigma_obs", 0.5))
    return np.array(v, dtype=float)


def vec_to_params(v: np.ndarray, n_fccp: int, include_sigma: bool = True,
                  base: Optional[dict] = None) -> dict:
    p = dict(base) if base else {}
    for i, k in enumerate(CORE_PARAM_ORDER):
        p[k] = float(v[i])
    off = len(CORE_PARAM_ORDER)
    p["alphas"] = [float(v[off + j]) for j in range(n_fccp)]
    if include_sigma:
        p["sigma_obs"] = float(v[off + n_fccp])
    return p


def get_bounds_vec(n_fccp: int, include_sigma: bool = True) -> List[Tuple[float, float]]:
    bounds = [PARAM_BOUNDS[k] for k in CORE_PARAM_ORDER]
    bounds.extend([PARAM_BOUNDS["alpha"]] * n_fccp)
    if include_sigma:
        bounds.append(PARAM_BOUNDS["sigma_obs"])
    return bounds


def get_param_names(n_fccp: int, include_sigma: bool = True) -> List[str]:
    names = list(CORE_PARAM_ORDER)
    names.extend([f"alpha_{j+1}" for j in range(n_fccp)])
    if include_sigma:
        names.append("sigma_obs")
    return names
