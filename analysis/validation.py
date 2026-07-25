"""
analysis/validation.py
======================
Out-of-sample diagnostic strategies for the 3-state OCR-informed bioenergetics model.

CAUTIOUS TERMINOLOGY
--------------------
Each routine below is named for what it ACTUALLY tests, not for what it
might colloquially be called:

1. CHAMBER HOLDOUT (technical-replicate transfer check)
   Fit on Chamber A of one preparation, predict Chamber B of the SAME
   preparation. This is a TECHNICAL-REPLICATE TRANSFER CHECK, NOT
   independent biological validation. Two chambers run simultaneously
   on the same biological sample share substrate, mitochondria, and
   experimental conditions; agreement between them does not establish
   inter-individual / inter-preparation generalisability.

2. WITHIN-TRACE HOLDOUT (intervention-phase extrapolation diagnostic)
   Fit on the first f% of a trace, predict the remainder. Useful for
   asking "does fitting the early phase let us predict the FCCP and
   inhibition phases?" — i.e. an INTERVENTION-PHASE EXTRAPOLATION
   DIAGNOSTIC. NOT full biological validation; the data are not
   statistically independent of the training data (same chamber, same
   noise process, same operator).

3. LEAVE-ONE-DATASET-OUT (LODO)
   Pooled-multitrace fit on K-1 datasets, predict the K-th. NOT
   hierarchical-Bayesian LOO. Whether LODO RMSE on real biological
   replicates approaches within-chamber noise is an OPEN QUESTION
   that real replicate data must answer; this codebase does not
   make that prediction.

4. PARAMETRIC-BOOTSTRAP PREDICTIVE CHECK
   Resample observation noise around the MAP-fitted trajectory; report
   the empirical coverage of a 90% predictive envelope. Coverage near
   90% is COMPATIBLE with the fitted iid-Gaussian observation-noise
   assumption — it is NOT evidence that the noise model is "correct"
   or "well-calibrated" in any stronger sense, and it is NOT a
   posterior-predictive check (no posterior parameter samples are
   drawn). The function is named `parametric_bootstrap_predictive_check`;
   the legacy alias `parametric_bootstrap_ppc` emits a DeprecationWarning.

NOTE ON THE FCCP PHASE
----------------------
Removing the FCCP phase from the calibration window removes the data
that lets the FCCP-amplitude parameters alpha_j be estimated. We do
NOT claim that "every kinetic phase carries information that cannot
be reconstructed"; we DO claim that the FCCP phase contains the
information needed to estimate the FCCP/uncoupling parameters.

NOTE ON COVERAGE BANDS
----------------------
The default reporting band [0.80, 0.95] used by the agent for
flagging is a CONFIGURABLE WARNING THRESHOLD, not a validated band of
biologically-correct calibration.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import sys, os
import warnings as _warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reduced_model import simulate, Protocol, DEFAULT_PARAMS, CORE_PARAM_ORDER


# ── Chamber holdout (technical-replicate transfer check) ────────────────
def chamber_holdout_rmse(t_train, o_train, t_test, o_test,
                         proto_train, proto_test,
                         calibrate_fn, **calib_kwargs):
    """Fit on (t_train, o_train) under proto_train, predict (t_test, o_test).

    Reported as a TECHNICAL-REPLICATE TRANSFER CHECK, not as biological
    validation. Chamber A and Chamber B share preparation, substrate,
    operator, and noise process.
    """
    res_calib = calibrate_fn(t_train, o_train, proto_train, **calib_kwargs)
    p = res_calib.params
    sim_train = simulate(p, proto_train, o2_init=float(o_train[0]), t_eval=t_train)
    sim_test  = simulate(p, proto_test,  o2_init=float(o_test[0]),  t_eval=t_test)
    return {
        "rmse_train": float(np.sqrt(np.mean((sim_train.o - o_train)**2)))
                        if sim_train.converged else np.nan,
        "rmse_test":  float(np.sqrt(np.mean((sim_test.o  - o_test )**2)))
                        if sim_test.converged else np.nan,
        "params":     p,
        "interpretation_note": (
            "Technical-replicate transfer check (Chamber A → Chamber B). "
            "NOT independent biological validation."
        ),
    }


# ── Within-trace holdout (intervention-phase extrapolation diagnostic) ──
def within_trace_holdout(t, o, proto, train_frac=0.7,
                         calibrate_fn=None, **calib_kwargs):
    """Time-split: fit first ``train_frac`` of the trace, predict the rest.

    This is a GENUINE refit-based holdout: parameters are re-optimised on the
    training segment ONLY (via ``calibrate_fn``), then the fitted model
    predicts the held-out segment. It is NOT a residual split of an
    already-fitted parameter vector.

    Framed as an INTERVENTION-PHASE EXTRAPOLATION DIAGNOSTIC: train and test
    are not statistically independent (same chamber, same noise process, same
    operator). Note also that ``rmse_test < rmse_train`` is legitimately
    possible here -- the held-out tail of a stress-test trace (post-FCCP
    plateau, post-inhibition residual) is often intrinsically lower-variance
    and therefore easier to fit than the event-rich training segment. To make
    the comparison interpretable, the returned dict reports which protocol
    events fall in the training vs test segments.
    """
    if calibrate_fn is None:
        from calibration.calibrate import calibrate_de
        calibrate_fn = calibrate_de
    n = len(t)
    cut = int(round(train_frac * n))
    t_train, o_train = t[:cut], o[:cut]
    t_test,  o_test  = t[cut:], o[cut:]
    t_split = float(t[cut]) if cut < n else float(t[-1])

    # Report which protocol events are in train vs test. The training fit
    # only "sees" data up to t_split, so any event at or after t_split is
    # being extrapolated to, not fitted.
    events = {"oligomycin": proto.t_oligo, "inhibition": proto.t_inhibit}
    for j, tf in enumerate(proto.t_fccp or [], start=1):
        events[f"FCCP_{j}"] = tf
    events_in_train = sorted(k for k, tv in events.items()
                             if tv is not None and tv < t_split)
    events_in_test = sorted(k for k, tv in events.items()
                            if tv is not None and tv >= t_split)

    proto_train = Protocol(t_oligo=proto.t_oligo, t_fccp=proto.t_fccp,
                            t_inhibit=proto.t_inhibit,
                            t_end=float(t_train[-1]),
                            t_start=proto.t_start, k_step=proto.k_step)
    res = calibrate_fn(t_train, o_train, proto_train, **calib_kwargs)
    p = res.params

    sim_full = simulate(p, proto, o2_init=float(o[0]), t_eval=t)
    base = {
        "method": "within_trace_holdout",
        "refit_based": True,
        "train_frac": train_frac, "n_train": cut, "n_test": n - cut,
        "t_split": t_split,
        "events_in_train": events_in_train,
        "events_in_test": events_in_test,
        "framing": "intervention-phase extrapolation diagnostic",
    }
    if not sim_full.converged:
        base.update({
            "rmse_train": np.nan, "rmse_test": np.nan, "params": p,
            "interpretation_note": (
                "Intervention-phase extrapolation diagnostic (refit-based); "
                "solver failed at the MAP fitted on the training segment."),
        })
        return base
    rmse_train = float(np.sqrt(np.mean((sim_full.o[:cut] - o_train)**2)))
    rmse_test  = float(np.sqrt(np.mean((sim_full.o[cut:] - o_test )**2)))
    base.update({
        "rmse_train": rmse_train, "rmse_test": rmse_test,
        "params": p, "t_full": t, "o_data": o, "o_pred": sim_full.o,
        "interpretation_note": (
            "Genuine refit-based intervention-phase extrapolation diagnostic: "
            "parameters were re-optimised on the training segment only, then "
            "used to predict the held-out segment. Train and test data are "
            "NOT statistically independent. rmse_test below rmse_train is "
            "expected when the held-out tail is intrinsically lower-variance "
            "than the event-rich training segment; see events_in_train / "
            "events_in_test."),
    })
    return base


# ── LODO ────────────────────────────────────────────────────────────────
def leave_one_dataset_out(traces, calibrate_pooled=None, **calib_kwargs):
    """Pooled-multitrace leave-one-dataset-out CV.

    `traces` is a list of (name, t, o, proto) tuples. The non-held datasets
    are calibrated with shared core parameters via `calibrate_pooled_multitrace`
    (the deprecated alias `calibrate_hierarchical` still works); only the
    held-out dataset's FCCP alphas are then estimated.

    NOTE: This is NOT hierarchical Bayesian leave-one-out. It is pooled-SSE
    leave-one-out cross-validation under shared kinetic parameters. Whether
    LODO RMSE on real biological replicates approaches within-chamber noise
    is an OPEN QUESTION that real replicate data must answer.
    """
    if calibrate_pooled is None:
        from calibration.calibrate import calibrate_pooled_multitrace
        calibrate_pooled = calibrate_pooled_multitrace
    from scipy.optimize import differential_evolution
    n_traces = len(traces)
    out = []
    for k in range(n_traces):
        train_set = [(t, o, p) for i, (_, t, o, p) in enumerate(traces) if i != k]
        held_name, t_held, o_held, proto_held = traces[k]
        results_train = calibrate_pooled(train_set, **calib_kwargs)
        shared = {kk: results_train[0].params[kk] for kk in CORE_PARAM_ORDER}

        n_fccp = len(proto_held.t_fccp)
        def obj_alpha(alphas):
            p = dict(DEFAULT_PARAMS); p.update(shared)
            p["alphas"] = [float(a) for a in alphas]
            sim = simulate(p, proto_held, o2_init=float(o_held[0]), t_eval=t_held)
            if not sim.converged: return 1e15
            return float(np.sum((sim.o - o_held)**2))
        bnds = [(1e-3, 100.0)] * n_fccp
        de = differential_evolution(obj_alpha, bnds, maxiter=40, popsize=8,
                                      seed=0, tol=1e-6, polish=True)
        p_held = dict(DEFAULT_PARAMS); p_held.update(shared)
        p_held["alphas"] = [float(a) for a in de.x]
        sim = simulate(p_held, proto_held, o2_init=float(o_held[0]),
                        t_eval=t_held)
        rmse = float(np.sqrt(np.mean((sim.o - o_held)**2))) \
                if sim.converged else np.nan
        out.append({
            "held_out": held_name, "shared_params": shared,
            "alphas_held": p_held["alphas"], "rmse_held": rmse,
            "n_train_traces": n_traces - 1,
            "t_held": t_held, "o_held": o_held,
            "o_pred_held": (sim.o if sim.converged else
                              np.full_like(t_held, np.nan, float)),
        })
    return out


# ── Parametric-bootstrap predictive check ───────────────────────────────
EXPLICIT_DISCLAIMER = (
    "This is a PARAMETRIC-BOOTSTRAP predictive check, NOT a "
    "posterior-predictive check (no posterior parameter samples are drawn). "
    "Coverage assumes the iid-Gaussian observation-noise model fitted at the "
    "MAP. Coverage near the nominal level is COMPATIBLE with that noise model "
    "for this trace; it is NOT evidence that the noise model is correct or "
    "well-calibrated in any stronger sense."
)


def parametric_bootstrap_predictive_check(
        t, o, proto, params_hat,
        n_boot=80, refit=False,
        calibrate_fn=None, seed=0, **calib_kwargs):
    """Parametric-bootstrap predictive check.

    refit=False: fast, observation-noise-only resampling (default).
    refit=True : full bootstrap (slow); refits parameters on each
                 simulated draw to incorporate parameter uncertainty.

    Returns a dict carrying both the canonical key
    `parametric_bootstrap_coverage_90` AND the legacy alias `coverage_90`,
    plus an `explicit_disclaimer` field. This function is the canonical
    name; the deprecated alias `parametric_bootstrap_ppc` emits a
    DeprecationWarning when called.
    """
    rng = np.random.default_rng(seed)
    sigma = float(params_hat.get("sigma_obs", 0.5))
    sim_hat = simulate(params_hat, proto, o2_init=float(o[0]), t_eval=t)
    if not sim_hat.converged:
        return {}
    o_hat = sim_hat.o

    if refit:
        if calibrate_fn is None:
            from calibration.calibrate import calibrate_de
            calibrate_fn = calibrate_de
        boot_sims = np.zeros((n_boot, len(t)))
        for b in range(n_boot):
            y_b = o_hat + rng.normal(0, sigma, size=len(t))
            res = calibrate_fn(t, y_b, proto, **calib_kwargs)
            sim_b = simulate(res.params, proto,
                              o2_init=float(y_b[0]), t_eval=t)
            boot_sims[b] = sim_b.o if sim_b.converged else o_hat
        # Add observation noise on top of refitted predictive mean
        boot_sims = boot_sims + rng.normal(0, sigma, size=boot_sims.shape)
    else:
        boot_sims = o_hat[None, :] + rng.normal(0, sigma,
                                                  size=(n_boot, len(t)))
    lo = np.quantile(boot_sims, 0.05, axis=0)
    hi = np.quantile(boot_sims, 0.95, axis=0)
    md = np.quantile(boot_sims, 0.50, axis=0)
    coverage = float(np.mean((o >= lo) & (o <= hi)))
    return {
        "t": t, "o_data": o, "o_hat": o_hat,
        "lo90": lo, "median": md, "hi90": hi,
        # Canonical key (CHUNK 5 rename)
        "parametric_bootstrap_coverage_90": coverage,
        # Legacy alias (still emitted so old callers don't break)
        "coverage_90": coverage,
        "n_boot": n_boot, "refit": refit,
        "explicit_disclaimer": EXPLICIT_DISCLAIMER,
        "interpretation_note": (
            "Coverage near the nominal level is COMPATIBLE with the fitted "
            "iid-Gaussian observation-noise model for this trace. It is NOT "
            "a posterior-predictive check, NOT evidence that the noise model "
            "is well-calibrated in any stronger sense, and NOT independent "
            "biological validation."
        ),
    }


def parametric_bootstrap_ppc(*args, **kwargs):
    """DEPRECATED: legacy alias for parametric_bootstrap_predictive_check.

    The original name suggested a posterior-predictive check, which it is
    NOT. Please migrate to `parametric_bootstrap_predictive_check`.
    """
    _warnings.warn(
        "parametric_bootstrap_ppc() is deprecated; use "
        "parametric_bootstrap_predictive_check() instead. The new name "
        "reflects that this is a parametric-bootstrap resampling under "
        "the fitted iid-Gaussian noise model, NOT a posterior-predictive "
        "check (no posterior parameter samples are drawn).",
        DeprecationWarning, stacklevel=2,
    )
    return parametric_bootstrap_predictive_check(*args, **kwargs)
