"""
analysis/sensitivity.py
=======================
Global sensitivity analysis for the 3-state OCR-informed bioenergetics model.

PARAMETER SET
-------------
The sensitivity parameter set comprises:
  * 8 core kinetic parameters (CORE_PARAM_ORDER)
  * Dataset-specific FCCP step amplitudes alpha_j (one per FCCP injection,
    so dataset_I has 1, dataset_II has 2, dataset_III has 4)
This is recorded in every result file as `parameter_set` so downstream
code knows exactly what was sampled.

OUTPUT METRICS
--------------
  'AUC_OCR'    : integrated OCR over the trace          (scalar)
  'final_O2'   : O2 concentration at t_end              (scalar)
  'OCR(t)'     : full time-resolved OCR                 (vector)
The metric used is recorded in every result file as `metric`.

INTERPRETATION RULES (must be respected when reporting / writing manuscripts)
-----------------------------------------------------------------------------
1. **ST INCLUDES INTERACTIONS** and is NOT additive variance. Sums of ST
   over parameters can exceed 1; only first-order S1 satisfies sum(S1) <= 1.
   Do NOT report ST values as "exclusive explained variance" or treat the
   sum of ST as 100%.

2. Sensitivity COMPLEMENTS identifiability — high sensitivity is NECESSARY
   but not SUFFICIENT for identifiability. Sensitivity analysis does NOT
   "confirm" identifiability of any parameter.

3. **Variance-degenerate phases must be flagged**: when the model output
   has near-zero variance over the parameter sample at a given time point
   (e.g. the post-Rotenone/Antimycin phase where OCR is forced to 0 by
   the inhibition switch), the Sobol indices are mathematically undefined
   and must NOT be interpreted. `time_resolved_sobol` flags these via
   `variance_degenerate_mask`.

4. **Small-N runs are diagnostic-level only.** Report the sample size
   (`n_evals`) in every result file. Publication-grade Sobol typically
   requires N_base >= 1024 for stable indices; below that, S1 estimates
   can be slightly negative due to Monte-Carlo error.

Methods:
  morris_screening      -- cheap screening (factor-fixing)
  sobol_indices         -- variance-based, first-order S1 + total-order ST
  time_resolved_sobol   -- S1, ST as functions of time

Uses SALib for sampling and analysis.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reduced_model import (
    Protocol, simulate, DEFAULT_PARAMS,
    PARAM_BOUNDS, CORE_PARAM_ORDER,
    vec_to_params,
)


# ── Module-level constants ──────────────────────────────────────────────
VAR_DEGEN_TOL = 1e-12   # output-variance threshold below which time-resolved
                        # Sobol indices are flagged as variance-degenerate.

INTERPRETATION_NOTE_SOBOL = (
    "Total-order indices (ST) include parameter INTERACTIONS and are NOT "
    "additive / exclusive variance contributions. Sums of ST across "
    "parameters can exceed 1. Only first-order S1 indices satisfy "
    "sum(S1) <= 1. Do not interpret ST as a fraction of variance attributable "
    "exclusively to that parameter."
)


def _salib_version() -> str:
    """Return the installed SALib version.

    SALib does not reliably expose a module-level ``__version__`` attribute,
    so ``getattr(SALib, "__version__", ...)`` previously always fell through
    to the literal string "unknown", silently breaking provenance capture.
    importlib.metadata reads the installed-package metadata directly.
    """
    try:
        import importlib.metadata as _md
        return str(_md.version("SALib"))
    except Exception:
        # Fall back to the module attribute if metadata lookup somehow fails;
        # only report "unknown" if there is genuinely nothing to report.
        try:
            import SALib
            return str(getattr(SALib, "__version__", "unknown"))
        except ImportError:
            return "not-installed"


def build_salib_problem(n_fccp: int, log_scale: bool = True
                          ) -> Tuple[dict, List[str]]:
    """Build a SALib `problem` dict for the 8 core params + N_fccp alphas."""
    names = list(CORE_PARAM_ORDER) + [f"alpha_{j+1}" for j in range(n_fccp)]
    bounds = []
    log_mask = []
    for nm in names:
        key = "alpha" if nm.startswith("alpha_") else nm
        lo, hi = PARAM_BOUNDS[key]
        if log_scale and (hi / max(lo, 1e-30)) >= 100:
            bounds.append([np.log10(lo), np.log10(hi)])
            log_mask.append(True)
        else:
            bounds.append([lo, hi])
            log_mask.append(False)
    problem = {"num_vars": len(names), "names": names, "bounds": bounds,
               "_log_mask": log_mask}
    return problem, names


def vec_to_params_salib(v: np.ndarray, problem: dict, n_fccp: int) -> dict:
    """Convert a SALib parameter row to a model parameter dict."""
    actual = np.array([10**v[i] if problem["_log_mask"][i] else v[i]
                        for i in range(len(v))])
    return vec_to_params(actual, n_fccp, include_sigma=False,
                          base=DEFAULT_PARAMS)




def _require_salib() -> None:
    """Raise a friendly error if optional SALib is unavailable."""
    try:
        import SALib  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Optional sensitivity dependency SALib is not installed. "
            "Morris and Sobol analyses were skipped. Install UI/science "
            "extras with `python -m pip install -r requirements-sensitivity.txt` "
            "or include SALib in your environment."
        ) from e

def output_final_O2(params, proto, o2_init, t_end):
    res = simulate(params, proto, o2_init=o2_init,
                   t_eval=np.linspace(proto.t_start, t_end, 100))
    return float(res.o[-1]) if res.converged else np.nan


def output_AUC_OCR(params, proto, o2_init):
    t_eval = np.linspace(proto.t_start, proto.t_end, 200)
    res = simulate(params, proto, o2_init=o2_init, t_eval=t_eval)
    if not res.converged:
        return np.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(res.OCR, t_eval))
    return float(np.trapz(res.OCR, t_eval))


def output_OCR_trace(params, proto, o2_init, t_grid):
    res = simulate(params, proto, o2_init=o2_init, t_eval=t_grid)
    return res.OCR if res.converged else np.full_like(t_grid, np.nan, float)


def morris_screening(proto, o2_init=170.0, N_trajectories=30, num_levels=4,
                     output="AUC_OCR", seed=0):
    """Morris elementary-effects screening.

    Total evals: N_trajectories * (num_vars + 1).

    Returns a dict carrying the parameter set, output metric, sample size,
    seed, and SALib version, so the result can be reproduced and cited.
    """
    _require_salib()
    from SALib.sample.morris  import sample  as morris_sample
    from SALib.analyze.morris import analyze as morris_analyze

    problem, names = build_salib_problem(len(proto.t_fccp), log_scale=True)
    log_mask = list(problem["_log_mask"])
    X = morris_sample(problem, N=N_trajectories, num_levels=num_levels,
                       seed=seed)
    Y = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        p = vec_to_params_salib(row, problem, len(proto.t_fccp))
        Y[i] = output_AUC_OCR(p, proto, o2_init) if output == "AUC_OCR" \
                else output_final_O2(p, proto, o2_init, proto.t_end)
    finite = np.isfinite(Y)
    n_nan = int((~finite).sum())
    if not finite.all():
        Y[~finite] = np.nanmean(Y[finite]) if finite.any() else 0.0
    Si = morris_analyze(problem, X, Y, num_levels=num_levels,
                         print_to_console=False)
    return {
        "method":         "morris",
        "metric":         output,
        "parameter_set":  list(names),
        "log_mask":       log_mask,
        "names":          list(names),
        "mu":             np.array(Si["mu"]),
        "mu_star":        np.array(Si["mu_star"]),
        "sigma":          np.array(Si["sigma"]),
        "mu_star_conf":   np.array(Si["mu_star_conf"]),
        "n_evals":        int(X.shape[0]),
        "n_trajectories": int(N_trajectories),
        "num_levels":     int(num_levels),
        "seed":           int(seed),
        "salib_version":  _salib_version(),
        "n_nan_outputs":  n_nan,
    }


def sobol_indices(proto, o2_init=170.0, N=128, output="AUC_OCR", seed=0):
    """First and total-order Sobol indices (scalar output).

    The result dict carries an explicit `interpretation_note` warning that
    ST INCLUDES interactions and is NOT additive variance.
    """
    _require_salib()
    from SALib.sample.sobol  import sample  as sobol_sample
    from SALib.analyze.sobol import analyze as sobol_analyze

    problem, names = build_salib_problem(len(proto.t_fccp), log_scale=True)
    log_mask = list(problem["_log_mask"])
    X = sobol_sample(problem, N, calc_second_order=False, seed=seed)
    Y = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        p = vec_to_params_salib(row, problem, len(proto.t_fccp))
        Y[i] = output_AUC_OCR(p, proto, o2_init) if output == "AUC_OCR" \
                else output_final_O2(p, proto, o2_init, proto.t_end)
    finite = np.isfinite(Y)
    n_nan = int((~finite).sum())
    if not finite.all():
        Y[~finite] = np.nanmean(Y[finite]) if finite.any() else 0.0
    Si = sobol_analyze(problem, Y, calc_second_order=False,
                        print_to_console=False)
    return {
        "method":         "sobol",
        "metric":         output,
        "parameter_set":  list(names),
        "log_mask":       log_mask,
        "names":          list(names),
        "S1":             np.array(Si["S1"]),
        "S1_conf":        np.array(Si["S1_conf"]),
        "ST":             np.array(Si["ST"]),
        "ST_conf":        np.array(Si["ST_conf"]),
        "n_evals":        int(X.shape[0]),
        "N_base":         int(N),
        "seed":           int(seed),
        "salib_version":  _salib_version(),
        "n_nan_outputs":  n_nan,
        "interpretation_note": INTERPRETATION_NOTE_SOBOL,
    }


def time_resolved_sobol(proto, o2_init=170.0, N=64, n_t_eval=30, seed=0):
    """Compute Sobol indices on OCR(t) at a grid of time points.

    Variance-degenerate time points (output variance below VAR_DEGEN_TOL,
    typically post-inhibition where OCR(t) ~ 0) are explicitly flagged
    via `variance_degenerate_mask`. Their S1/ST entries are forced to zero
    and must NOT be interpreted as sensitivity indices — sensitivity is
    mathematically undefined at zero variance.
    """
    _require_salib()
    from SALib.sample.sobol  import sample  as sobol_sample
    from SALib.analyze.sobol import analyze as sobol_analyze

    problem, names = build_salib_problem(len(proto.t_fccp), log_scale=True)
    log_mask = list(problem["_log_mask"])
    X = sobol_sample(problem, N, calc_second_order=False, seed=seed)
    t_grid = np.linspace(proto.t_start, proto.t_end, n_t_eval)
    Y_all = np.zeros((X.shape[0], n_t_eval))
    n_fccp = len(proto.t_fccp)
    for i, row in enumerate(X):
        p = vec_to_params_salib(row, problem, n_fccp)
        Y_all[i] = output_OCR_trace(p, proto, o2_init, t_grid)

    n_params = len(names)
    S1_t = np.zeros((n_params, n_t_eval))
    ST_t = np.zeros((n_params, n_t_eval))
    output_variance = np.zeros(n_t_eval)
    var_degen_mask = np.zeros(n_t_eval, dtype=bool)
    n_nan_total = 0

    for k in range(n_t_eval):
        Y_k = Y_all[:, k].copy()
        finite = np.isfinite(Y_k)
        n_nan_total += int((~finite).sum())
        if not finite.all():
            Y_k[~finite] = np.nanmean(Y_k[finite]) if finite.any() else 0.0
        var_k = float(np.var(Y_k))
        output_variance[k] = var_k
        if var_k < VAR_DEGEN_TOL:
            var_degen_mask[k] = True
            # Leave S1_t[:, k] and ST_t[:, k] at zero; flagged as
            # variance-degenerate so consumers know not to interpret them.
            continue
        try:
            Si = sobol_analyze(problem, Y_k, calc_second_order=False,
                                 print_to_console=False)
            S1_t[:, k] = np.array(Si["S1"])
            ST_t[:, k] = np.array(Si["ST"])
        except Exception:
            var_degen_mask[k] = True

    return {
        "method":          "time_resolved_sobol",
        "metric":          "OCR(t)",
        "parameter_set":   list(names),
        "log_mask":        log_mask,
        "names":           list(names),
        "t_grid":          t_grid,
        "S1_t":            S1_t,
        "ST_t":            ST_t,
        "output_variance": output_variance,
        "variance_degenerate_mask": var_degen_mask,
        "var_degen_tol":   VAR_DEGEN_TOL,
        "n_evals":         int(X.shape[0]),
        "N_base":          int(N),
        "n_t_eval":        int(n_t_eval),
        "seed":            int(seed),
        "salib_version":   _salib_version(),
        "n_nan_outputs":   n_nan_total,
        "interpretation_note": (
            INTERPRETATION_NOTE_SOBOL + " Time points flagged in "
            "variance_degenerate_mask have output variance below "
            f"{VAR_DEGEN_TOL:g} (typically post-Rotenone/Antimycin where "
            "OCR(t) is forced to ~0 by the inhibition switch); their "
            "Sobol indices are mathematically undefined and have been "
            "set to zero. Do NOT interpret them."
        ),
    }
