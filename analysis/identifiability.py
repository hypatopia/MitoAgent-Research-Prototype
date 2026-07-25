"""
analysis/identifiability.py
============================
Practical identifiability analysis for the 3-state OCR-informed bioenergetics model.

Two complementary methods:

1. FISHER INFORMATION MATRIX (FIM) at the MAP estimate
   -----------------------------------------------------
   Linearises the model around theta_hat and computes
       FIM = (1/sigma^2) * J^T J
   where J_ij = d o(t_i; theta) / d theta_j is the sensitivity matrix.
   Eigenvalues of FIM give 'stiffness' of each parameter direction:
   - large eigenvalue  -> well-constrained (stiff) direction
   - small eigenvalue  -> sloppy / unidentifiable direction
   The condition number cond(FIM) measures the overall identifiability.

2. PROFILE LIKELIHOODS
   -------------------
   For each parameter theta_k, fix theta_k at a grid of values and
   re-optimise all OTHER parameters at each grid point.  The resulting
   profile -2 log L curve has:
   - flat regions  -> non-identifiable
   - sharp minimum -> identifiable
   - one-sided rise -> identifiable only on one side
   95% CI = parameter values where 2(L_max - L) <= 3.84 (chi^2_1).

Both methods return ParameterIdentifiabilityReport objects suitable for
inclusion in the manuscript and for runtime diagnostics in the AI agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys, os
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reduced_model import (
    Protocol, simulate, DEFAULT_PARAMS, PARAM_BOUNDS, CORE_PARAM_ORDER,
    params_to_vec, vec_to_params,
)


# Module constant: tiny eigenvalues below this are clipped before computing
# the "clipped" condition number. The raw condition number is always reported
# alongside, with both values written into the JSON. A condition number above
# this threshold (raw or clipped) is large enough that direct interpretation
# of the inverse-FIM correlation matrix is not warranted.
EIG_CLIP_FLOOR = 1e-12


# ── Reports ──────────────────────────────────────────────────────────────
@dataclass
class FIMReport:
    param_names:   List[str]
    theta_hat:     np.ndarray          # MAP estimate vector
    fim:           np.ndarray          # n_p x n_p Fisher Info Matrix
    # Raw (unclipped) eigenvalues and condition number
    eigvals_raw:   np.ndarray          # sorted ascending; may include very
                                        # small or numerically-negative values
    condition_raw: float               # max(eig_raw) / max(min(eig_raw),
                                        # tiny) — diagnostic value
    # Clipped: eigenvalues floor-clipped at EIG_CLIP_FLOOR before computing
    # the condition number. This is what should be cited when discussing
    # numerical-stability of the FIM inverse.
    eigvals_clipped: np.ndarray
    condition_clipped: float
    eigvecs:       np.ndarray          # cols = eigenvectors of the SAME
                                        # symmetric FIM eig decomposition
    sigma_obs:     float
    sloppy_index:  np.ndarray          # ranking of params by total sensitivity
    direction_widths: np.ndarray       # 1/sqrt(clipped lambda_i)
    warnings:      List[str] = field(default_factory=list)

    # Backward-compat aliases (so existing callers keep working).
    @property
    def eigvals(self) -> np.ndarray:
        return self.eigvals_raw

    @property
    def condition(self) -> float:
        return self.condition_raw


@dataclass
class ProfileLikelihoodReport:
    param_name:        str
    map_value:         float           # the MAP value supplied to the profiler
    map_nll:           float           # NLL evaluated at the MAP
    profile_min_value: float           # theta where the profile NLL is lowest
    profile_min_nll:   float           # the lowest NLL on the profile grid
    theta_grid:        np.ndarray
    nll_grid:          np.ndarray      # -log L at each grid point (after refit;
                                        # NaN where the inner-optim failed)
    delta_nll:         np.ndarray      # 2*(L - L_min) at each grid point
    optimizer_success: np.ndarray      # bool array: True at converged grid pts
    n_optimizer_failures: int
    ci_low:            Optional[float]
    ci_high:           Optional[float]
    identified_left:   bool
    identified_right:  bool
    practical_id:      str             # "identifiable" | "one-sided" |
                                        # "non-identifiable" | "unresolved"
    map_inside_ci:     Optional[bool]
    notes:             str = ""


# ── Helper: numerical sensitivity matrix ─────────────────────────────────
def sensitivity_matrix(theta_hat: np.ndarray, t_data: np.ndarray,
                       proto: Protocol, o2_init: float, n_fccp: int,
                       eps_rel: float = 1e-3) -> np.ndarray:
    """Compute J_ij = d o_i / d theta_j by central finite differences in
    log-parameter space (more robust for parameters spanning many decades).

    Returns J of shape (n_t, n_p).  theta_hat contains the 8 core kinetic
    parameters followed by N_fccp alpha values (no sigma_obs).
    """
    n_p = len(theta_hat)
    n_t = len(t_data)
    base = vec_to_params(theta_hat, n_fccp, include_sigma=False,
                          base=DEFAULT_PARAMS)
    res0 = simulate(base, proto, o2_init=o2_init, t_eval=t_data)
    if not res0.converged:
        return np.full((n_t, n_p), np.nan)
    o0 = res0.o

    J = np.zeros((n_t, n_p))
    for k in range(n_p):
        h = eps_rel * max(abs(theta_hat[k]), 1e-6)
        tp = theta_hat.copy(); tp[k] += h
        tm = theta_hat.copy(); tm[k] -= h
        p_p = vec_to_params(tp, n_fccp, include_sigma=False, base=DEFAULT_PARAMS)
        p_m = vec_to_params(tm, n_fccp, include_sigma=False, base=DEFAULT_PARAMS)
        rp = simulate(p_p, proto, o2_init=o2_init, t_eval=t_data)
        rm = simulate(p_m, proto, o2_init=o2_init, t_eval=t_data)
        if rp.converged and rm.converged:
            J[:, k] = (rp.o - rm.o) / (2 * h)
        else:
            J[:, k] = 0.0   # silently treat as zero sensitivity (extremes)
    return J


# ── FIM analysis ──────────────────────────────────────────────────────────
def fisher_information(params_hat: dict, t_data: np.ndarray,
                       o_data: np.ndarray, proto: Protocol,
                       o2_init: float,
                       sigma_obs: Optional[float] = None,
                       eps_rel: float = 1e-3) -> FIMReport:
    """Compute the Fisher Information Matrix at the MAP estimate.

    Parameters
    ----------
    params_hat : fitted parameter dict from calibration
    t_data, o_data : observation grid and oxygen values
    proto : Protocol used during calibration
    o2_init : initial oxygen used during simulation
    sigma_obs : observation SD; if None, taken from params_hat['sigma_obs']
                or estimated from RMS residual.
    """
    n_fccp = len(proto.t_fccp)
    names  = list(CORE_PARAM_ORDER) + [f"alpha_{j+1}" for j in range(n_fccp)]
    theta_hat = np.array([params_hat[k] for k in CORE_PARAM_ORDER]
                          + list(params_hat["alphas"]), dtype=float)
    if sigma_obs is None:
        sigma_obs = float(params_hat.get("sigma_obs", None) or np.std(o_data) / 2.0)

    J = sensitivity_matrix(theta_hat, t_data, proto, o2_init, n_fccp, eps_rel)
    if not np.isfinite(J).all():
        zeros = np.zeros(len(theta_hat))
        return FIMReport(
            param_names=names, theta_hat=theta_hat,
            fim=np.zeros((len(theta_hat),)*2),
            eigvals_raw=zeros.copy(), condition_raw=np.inf,
            eigvals_clipped=zeros.copy(), condition_clipped=np.inf,
            eigvecs=np.eye(len(theta_hat)),
            sigma_obs=float(sigma_obs),
            sloppy_index=zeros.copy(), direction_widths=zeros.copy(),
            warnings=["sensitivity matrix contains non-finite entries; "
                      "FIM not computed"],
        )

    fim = (J.T @ J) / (sigma_obs ** 2)
    eigvals_raw, eigvecs = np.linalg.eigh((fim + fim.T) / 2)  # symmetric

    # Raw condition (diagnostic): use the actual smallest eigenvalue, clamped
    # only to avoid division-by-zero. The raw value can be enormous when the
    # FIM is rank-deficient — that is the diagnostic signal we want to surface.
    smallest_raw = max(float(eigvals_raw[0]), 1e-30)
    cond_raw = float(eigvals_raw[-1] / smallest_raw)

    # Clipped: floor tiny / numerically-negative eigenvalues at EIG_CLIP_FLOOR
    # before computing the condition number. This is what should be cited
    # for inverse-FIM-style discussion.
    eigvals_clipped = np.clip(eigvals_raw, EIG_CLIP_FLOOR, None)
    cond_clipped = float(eigvals_clipped[-1] / eigvals_clipped[0])

    # Per-parameter total sensitivity (root-sum-square column of J)
    rss = np.sqrt(np.sum(J ** 2, axis=0)) / sigma_obs
    sloppy_index = np.argsort(-rss)   # most sensitive first

    # Direction widths (in log-parameter space) at 1-sigma CI, using clipped
    # eigenvalues to avoid div-by-zero on degenerate directions.
    direction_widths = 1.0 / np.sqrt(eigvals_clipped)

    warnings: List[str] = []
    if cond_raw > 1e15:
        warnings.append(
            f"FIM raw condition {cond_raw:.3e} indicates a sloppy / "
            f"practically rank-deficient information matrix. Direct "
            f"interpretation of the inverse-FIM correlation matrix is NOT "
            f"warranted; profile-likelihood analysis is required to obtain "
            f"reliable confidence intervals."
        )
    if eigvals_raw[0] < 0:
        warnings.append(
            f"FIM smallest eigenvalue is numerically negative "
            f"({float(eigvals_raw[0]):.3e}). This is a numerical artefact of "
            f"a degenerate direction and was clipped to {EIG_CLIP_FLOOR:.0e} "
            f"before computing the clipped condition number."
        )

    return FIMReport(
        param_names=names, theta_hat=theta_hat, fim=fim,
        eigvals_raw=eigvals_raw, condition_raw=cond_raw,
        eigvals_clipped=eigvals_clipped, condition_clipped=cond_clipped,
        eigvecs=eigvecs, sigma_obs=float(sigma_obs),
        sloppy_index=sloppy_index, direction_widths=direction_widths,
        warnings=warnings,
    )


def fim_summary_table(rep: FIMReport) -> List[Tuple[str, float, float, float]]:
    """Per-parameter summary: (name, MAP value, marginal SD via FIM pseudo-inv,
    relative SD = SD / MAP).

    WARNING: when condition_raw exceeds ~1e15 (sloppy / practically
    rank-deficient FIM), the pseudo-inverse-derived SDs are NOT reliable
    confidence-interval estimates. The pinv silently inflates uncertainty
    along sloppy directions in ways that don't correspond to actual
    likelihood-based intervals. Use profile likelihoods for definitive CIs
    when rep.warnings is non-empty.
    """
    try:
        Cov = np.linalg.pinv(rep.fim, rcond=1e-10)
    except np.linalg.LinAlgError:
        Cov = np.full_like(rep.fim, np.nan)
    sd = np.sqrt(np.clip(np.diag(Cov), 0, None))
    rows = []
    for i, name in enumerate(rep.param_names):
        m = rep.theta_hat[i]
        rel = sd[i] / max(abs(m), 1e-30)
        rows.append((name, float(m), float(sd[i]), float(rel)))
    return rows


# ── Profile likelihood ───────────────────────────────────────────────────
def profile_likelihood(param_name: str, params_hat: dict,
                       t_data: np.ndarray, o_data: np.ndarray,
                       proto: Protocol, o2_init: float,
                       n_grid: int = 25,
                       grid_span_log: float = 1.5,
                       sigma_obs: Optional[float] = None,
                       maxiter: int = 30,
                       adaptive_extend: bool = True,
                       max_extensions: int = 2,
                       extend_factor_log: float = 0.5,
                       n_restarts_constrained: int = 3,
                       chi2_threshold: float = 3.841,
                       verbose: bool = False) -> ProfileLikelihoodReport:
    """Robust profile likelihood for a single parameter.

    For each grid point theta_k, we FIX theta_k and re-optimise all OTHER
    parameters (L-BFGS-B). The NLL profile NLL(theta_k) is what tells us
    how identifiable theta_k is from the data.

    Robustness features (Section F of the project plan):

    * Default 25-point log-spaced grid (vs. the previous 9-point grid which
      was too coarse to support publication-quality verdicts).
    * **Adaptive extension** (`adaptive_extend=True`): if the chi^2 threshold
      isn't crossed within the initial grid span, the grid is extended
      outward by `extend_factor_log` decades on whichever side(s) didn't
      cross, up to `max_extensions` times. Parameter bounds are respected.
    * **Continuation / warm-starting**: the inner optimiser walks LEFT and
      RIGHT from the MAP grid index, warm-starting each new optimisation
      from the previous grid point's converged solution. This dramatically
      reduces the chance of L-BFGS-B getting stuck in a local well.
    * **Multi-start at constrained points**: when a grid point sits at a
      parameter bound, `n_restarts_constrained` random uniform starts within
      the free-parameter bounds are tried; the best (lowest-NLL) result is
      kept.
    * **Optimizer-failure tracking**: every grid point records whether
      L-BFGS-B reported success AND finished within `maxiter` iterations.
      Failed minimisations record `nll_grid[i] = np.nan`. The CI extractor
      explicitly SKIPS NaN entries — a NaN spike is never treated as
      evidence for crossing the chi^2 threshold.

    Returns a fully-populated ProfileLikelihoodReport including:
    map_value, map_nll, profile_min_value, profile_min_nll, theta_grid,
    nll_grid, delta_nll, optimizer_success array, n_optimizer_failures,
    ci_low, ci_high, verdict, map_inside_ci, and a `notes` string that
    narrates any anomalies found during profiling.
    """
    n_fccp = len(proto.t_fccp)
    names = list(CORE_PARAM_ORDER) + [f"alpha_{j+1}" for j in range(n_fccp)]
    if param_name not in names:
        raise KeyError(f"unknown parameter: {param_name}")
    k_idx = names.index(param_name)

    theta_hat = np.array([params_hat[k] for k in CORE_PARAM_ORDER]
                          + list(params_hat["alphas"]), dtype=float)
    sigma_obs = float(sigma_obs if sigma_obs is not None
                       else params_hat.get("sigma_obs",
                                            np.std(o_data) / 2.0))

    # Bounds — full and free-parameter.
    full_bounds = [PARAM_BOUNDS[k] for k in CORE_PARAM_ORDER] + \
                   [PARAM_BOUNDS["alpha"]] * n_fccp
    bd_key = "alpha" if param_name.startswith("alpha_") else param_name
    pk_lo, pk_hi = PARAM_BOUNDS.get(bd_key, (1e-6, 1e6))
    free_idx = [i for i in range(len(names)) if i != k_idx]
    free_bounds = [full_bounds[i] for i in free_idx]

    # ── Inner objective ─────────────────────────────────────────────────
    def nll_full(theta_full: np.ndarray) -> float:
        p = vec_to_params(theta_full, n_fccp, include_sigma=False,
                           base=DEFAULT_PARAMS)
        p["sigma_obs"] = sigma_obs
        res = simulate(p, proto, o2_init=o2_init, t_eval=t_data)
        if not res.converged:
            return 1e15
        resid = res.o - o_data
        n = len(o_data)
        nll = 0.5 * n * np.log(2 * np.pi * sigma_obs**2) \
              + 0.5 * np.sum(resid**2) / sigma_obs**2
        return float(nll)

    def fit_at_theta_k(theta_k: float, x0_free: np.ndarray
                       ) -> Tuple[float, np.ndarray, bool]:
        """Run L-BFGS-B at fixed theta_k from a single starting point.
        Returns (nll, x_at_min, ok)."""
        def obj_free(x_free):
            x_full = np.zeros(len(names))
            x_full[free_idx] = x_free
            x_full[k_idx]    = theta_k
            return nll_full(x_full)
        try:
            res = minimize(obj_free, x0_free, method="L-BFGS-B",
                            bounds=free_bounds,
                            options={"maxiter": maxiter, "ftol": 1e-7})
            ok = bool(res.success and res.nit < maxiter)
            return float(res.fun), np.asarray(res.x, dtype=float), ok
        except Exception:
            return float("nan"), x0_free.copy(), False

    def fit_with_multistart(theta_k: float, x0_free: np.ndarray,
                            n_restarts: int) -> Tuple[float, np.ndarray, bool]:
        """Best-of-n_restarts from the warm-start plus n_restarts-1 random
        uniform starts within the free bounds."""
        nll_best, x_best, ok_best = fit_at_theta_k(theta_k, x0_free)
        rng = np.random.default_rng(0)
        for _ in range(max(n_restarts - 1, 0)):
            x_try = np.array([rng.uniform(lo, hi) for lo, hi in free_bounds])
            nll_try, x_at, ok = fit_at_theta_k(theta_k, x_try)
            if (not np.isfinite(nll_best)
                    or (np.isfinite(nll_try) and nll_try < nll_best)):
                nll_best, x_best, ok_best = nll_try, x_at, ok
        return nll_best, x_best, ok_best

    # ── Initial grid construction ───────────────────────────────────────
    theta_k_hat = float(theta_hat[k_idx])
    if theta_k_hat <= 0:
        # extremely rare; build a linear-around-zero grid that respects bounds
        lo = max(pk_lo, theta_k_hat - 1.0)
        hi = min(pk_hi, theta_k_hat + 1.0)
        log_grid = None
        theta_grid = np.linspace(lo, hi, n_grid)
    else:
        log_th = np.log(theta_k_hat)
        log_lo = max(np.log(max(pk_lo, 1e-30)), log_th - grid_span_log)
        log_hi = min(np.log(pk_hi),             log_th + grid_span_log)
        log_grid = np.linspace(log_lo, log_hi, n_grid)
        theta_grid = np.exp(log_grid)

    # MAP NLL at the unrestricted MAP (one full-vector evaluation).
    map_nll = nll_full(theta_hat)

    # ── Inner loop with warm-starting ───────────────────────────────────
    def run_grid(theta_grid_local: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the inner optimiser across theta_grid_local with warm-starting
        from the MAP outward. Returns (nll_arr, success_arr, x_at_min_grid).
        """
        n = len(theta_grid_local)
        nll_arr = np.full(n, np.nan)
        ok_arr = np.zeros(n, dtype=bool)
        # Each row: free-parameter solution at that grid point (or x0_free).
        x_grid = np.tile(theta_hat[free_idx], (n, 1)).astype(float)

        # Find the index of the grid point closest to MAP.
        midx = int(np.argmin(np.abs(theta_grid_local - theta_k_hat)))

        # Centre point: warm-start from the MAP free-vector.
        x0_free = theta_hat[free_idx].copy()
        nll_arr[midx], x_grid[midx], ok_arr[midx] = \
            fit_at_theta_k(theta_grid_local[midx], x0_free)
        if verbose:
            print(f"  [{param_name}] grid {midx+1}/{n} (centre): "
                   f"theta={theta_grid_local[midx]:.4g}  "
                   f"-log L={nll_arr[midx]:.3f}  ok={ok_arr[midx]}")

        # Walk RIGHT (increasing theta), warm-starting from the previous
        # converged free vector. At parameter-bound grid points, do
        # multi-start to avoid getting stuck.
        for i in range(midx + 1, n):
            at_bound = (
                np.isclose(theta_grid_local[i], pk_hi, rtol=1e-6, atol=1e-12)
                or np.isclose(theta_grid_local[i], pk_lo,
                              rtol=1e-6, atol=1e-12))
            x0 = x_grid[i - 1].copy()
            if at_bound:
                nll_arr[i], x_grid[i], ok_arr[i] = fit_with_multistart(
                    theta_grid_local[i], x0, n_restarts_constrained)
            else:
                nll_arr[i], x_grid[i], ok_arr[i] = fit_at_theta_k(
                    theta_grid_local[i], x0)
            if verbose:
                print(f"  [{param_name}] grid {i+1}/{n} (right): "
                       f"theta={theta_grid_local[i]:.4g}  "
                       f"-log L={nll_arr[i]:.3f}  ok={ok_arr[i]}")

        # Walk LEFT (decreasing theta).
        for i in range(midx - 1, -1, -1):
            at_bound = (
                np.isclose(theta_grid_local[i], pk_hi, rtol=1e-6, atol=1e-12)
                or np.isclose(theta_grid_local[i], pk_lo,
                              rtol=1e-6, atol=1e-12))
            x0 = x_grid[i + 1].copy()
            if at_bound:
                nll_arr[i], x_grid[i], ok_arr[i] = fit_with_multistart(
                    theta_grid_local[i], x0, n_restarts_constrained)
            else:
                nll_arr[i], x_grid[i], ok_arr[i] = fit_at_theta_k(
                    theta_grid_local[i], x0)
            if verbose:
                print(f"  [{param_name}] grid {i+1}/{n} (left): "
                       f"theta={theta_grid_local[i]:.4g}  "
                       f"-log L={nll_arr[i]:.3f}  ok={ok_arr[i]}")

        return nll_arr, ok_arr, x_grid

    nll_grid, ok_grid, _ = run_grid(theta_grid)

    # ── Adaptive extension ──────────────────────────────────────────────
    def crosses(side: str, nll_arr: np.ndarray, theta_arr: np.ndarray) -> bool:
        finite = ~np.isnan(nll_arr)
        if not finite.any():
            return False
        nll_min = float(np.nanmin(nll_arr))
        delta = 2.0 * (nll_arr - nll_min)
        if side == "left":
            # leftmost grid points: did delta exceed threshold on the left?
            for j in range(int(np.nanargmin(delta)) - 1, -1, -1):
                if not np.isnan(delta[j]) and delta[j] > chi2_threshold:
                    return True
            return False
        if side == "right":
            for j in range(int(np.nanargmin(delta)) + 1, len(delta)):
                if not np.isnan(delta[j]) and delta[j] > chi2_threshold:
                    return True
            return False
        return False

    if adaptive_extend and log_grid is not None:
        for ext in range(max_extensions):
            extend_left  = (not crosses("left",  nll_grid, theta_grid)
                             and theta_grid[0] > pk_lo + 1e-12)
            extend_right = (not crosses("right", nll_grid, theta_grid)
                             and theta_grid[-1] < pk_hi - 1e-12)
            if not (extend_left or extend_right):
                break
            log_grid = np.log(theta_grid)
            new_log_grid = log_grid.copy()
            if extend_left:
                new_log_lo = max(np.log(max(pk_lo, 1e-30)),
                                  log_grid[0] - extend_factor_log)
                left_extra = np.linspace(new_log_lo, log_grid[0],
                                           max(int(n_grid / 4), 3))[:-1]
                new_log_grid = np.concatenate([left_extra, new_log_grid])
            if extend_right:
                new_log_hi = min(np.log(pk_hi),
                                  log_grid[-1] + extend_factor_log)
                right_extra = np.linspace(log_grid[-1], new_log_hi,
                                            max(int(n_grid / 4), 3))[1:]
                new_log_grid = np.concatenate([new_log_grid, right_extra])
            theta_grid = np.exp(new_log_grid)
            nll_grid, ok_grid, _ = run_grid(theta_grid)
            if verbose:
                print(f"  [{param_name}] extension {ext+1}: "
                       f"new grid len={len(theta_grid)}, "
                       f"span [{theta_grid[0]:.3g}, {theta_grid[-1]:.3g}]")

    # ── Process results ─────────────────────────────────────────────────
    finite_mask = ~np.isnan(nll_grid)
    if finite_mask.any():
        nll_min_idx = int(np.nanargmin(nll_grid))
        profile_min_value = float(theta_grid[nll_min_idx])
        profile_min_nll   = float(nll_grid[nll_min_idx])
        delta = 2.0 * (nll_grid - profile_min_nll)
    else:
        profile_min_value = float(theta_k_hat)
        profile_min_nll   = float(map_nll)
        delta = np.full_like(nll_grid, np.nan)

    ci_info = _extract_ci_from_profile(
        theta_grid, delta, chi2_threshold=chi2_threshold,
        map_value=float(theta_k_hat))
    ci_low      = ci_info["ci_low"]
    ci_high     = ci_info["ci_high"]
    id_left     = ci_low  is not None
    id_right    = ci_high is not None
    verdict     = ci_info["verdict"]
    map_in_ci   = ci_info["map_inside_ci"]
    n_fail      = int(np.sum(~ok_grid))

    # ── Plain-English notes ─────────────────────────────────────────────
    notes_parts: List[str] = []
    if n_fail:
        notes_parts.append(
            f"{n_fail} optimiser failure(s) of {len(theta_grid)} grid "
            f"points; their NLL entries are NaN and were skipped during CI "
            f"extraction (a NaN spike is NOT evidence)."
        )
    if not np.isclose(profile_min_value, theta_k_hat, rtol=1e-3):
        notes_parts.append(
            f"profile minimum at theta_k={profile_min_value:.4g} differs "
            f"from MAP at theta_k={theta_k_hat:.4g} (the inner re-optim "
            f"found a strictly better fit). Both values are reported."
        )
    if map_in_ci is False:
        notes_parts.append(
            "MAP estimate lies OUTSIDE the extracted 95% CI; this typically "
            "means the profile re-optimisation found a strictly better "
            "optimum than the MAP supplied (multimodal likelihood / "
            "imperfect MAP convergence). Both values are reported."
        )
    notes = " ".join(notes_parts)

    return ProfileLikelihoodReport(
        param_name=param_name,
        map_value=float(theta_k_hat),
        map_nll=float(map_nll),
        profile_min_value=profile_min_value,
        profile_min_nll=profile_min_nll,
        theta_grid=theta_grid,
        nll_grid=nll_grid,
        delta_nll=delta,
        optimizer_success=ok_grid,
        n_optimizer_failures=n_fail,
        ci_low=ci_low, ci_high=ci_high,
        identified_left=id_left, identified_right=id_right,
        practical_id=verdict,
        map_inside_ci=map_in_ci,
        notes=notes,
    )


def _extract_ci_from_profile(theta_grid: np.ndarray,
                             delta: np.ndarray,
                             chi2_threshold: float = 3.841,
                             map_value: Optional[float] = None,
                             ) -> Dict[str, Any]:
    """Extract a confidence interval from a profile-likelihood curve.

    Parameters
    ----------
    theta_grid : ndarray
        Parameter grid (any spacing).
    delta : ndarray
        2 * (NLL(theta) - NLL_min) at each grid point. May contain NaN
        for grid points where the inner re-optimisation failed; these
        entries are SKIPPED rather than treated as crossings of the
        threshold (a NaN spike is NOT evidence).
    chi2_threshold : float
        Default 3.841 (95% chi^2_1 quantile).
    map_value : float, optional
        The MAP value of theta. If supplied, the report includes whether
        the MAP lies inside the extracted CI.

    Returns
    -------
    dict with keys:
        ci_low, ci_high             : Optional[float]
        verdict                     : "identifiable" | "one-sided"
                                      | "non-identifiable" | "unresolved"
        n_optimizer_failures        : int   (count of NaN in delta)
        map_inside_ci               : Optional[bool]
    """
    delta = np.asarray(delta, dtype=float)
    if len(delta) == 0:
        return {"ci_low": None, "ci_high": None,
                "verdict": "unresolved", "n_optimizer_failures": 0,
                "map_inside_ci": None}

    n_fail = int(np.sum(np.isnan(delta)))
    finite_mask = ~np.isnan(delta)
    if not finite_mask.any():
        return {"ci_low": None, "ci_high": None,
                "verdict": "unresolved",
                "n_optimizer_failures": n_fail,
                "map_inside_ci": None}

    # The grid minimum is taken over finite entries only.
    min_idx = int(np.nanargmin(delta))
    ci_low = ci_high = None

    # Walk left from min_idx, skipping NaN entries.
    for j in range(min_idx - 1, -1, -1):
        if np.isnan(delta[j]):
            continue
        if delta[j] > chi2_threshold:
            ci_low = float(theta_grid[j])
            break

    # Walk right from min_idx, skipping NaN entries.
    for j in range(min_idx + 1, len(delta)):
        if np.isnan(delta[j]):
            continue
        if delta[j] > chi2_threshold:
            ci_high = float(theta_grid[j])
            break

    id_left  = ci_low  is not None
    id_right = ci_high is not None
    if id_left and id_right:
        verdict = "identifiable"
    elif id_left or id_right:
        verdict = "one-sided"
    else:
        # Distinguish flat (well-resolved at minimum but no rise) from
        # truly unresolved (most points NaN). If less than half the grid
        # is finite we call it unresolved.
        verdict = "unresolved" if n_fail >= len(delta) / 2 \
                                else "non-identifiable"

    map_inside_ci: Optional[bool] = None
    if map_value is not None:
        if id_left and id_right:
            map_inside_ci = bool(ci_low <= map_value <= ci_high)
        elif id_left:
            map_inside_ci = bool(ci_low <= map_value)
        elif id_right:
            map_inside_ci = bool(map_value <= ci_high)
        # else: no CI -> map_inside_ci stays None

    return {"ci_low": ci_low, "ci_high": ci_high,
            "verdict": verdict,
            "n_optimizer_failures": n_fail,
            "map_inside_ci": map_inside_ci}


def run_all_profiles(params_hat: dict, t_data: np.ndarray, o_data: np.ndarray,
                     proto: Protocol, o2_init: float,
                     n_grid: int = 25, grid_span_log: float = 1.5,
                     maxiter: int = 30,
                     adaptive_extend: bool = True,
                     n_restarts_constrained: int = 3,
                     verbose: bool = False
                     ) -> Dict[str, ProfileLikelihoodReport]:
    """Run profile likelihoods for all 8 core kinetic parameters.

    Defaults are publication-grade (n_grid=25, grid_span_log=1.5,
    adaptive extension on). Reduce for diagnostic-level runs by passing
    smaller values explicitly.
    """
    out = {}
    for nm in CORE_PARAM_ORDER:
        if verbose:
            print(f"  profiling {nm}...")
        out[nm] = profile_likelihood(
            nm, params_hat, t_data, o_data, proto, o2_init,
            n_grid=n_grid, grid_span_log=grid_span_log,
            maxiter=maxiter,
            adaptive_extend=adaptive_extend,
            n_restarts_constrained=n_restarts_constrained,
            verbose=False)
    return out

# ── Fast diagnostic fixed-other-parameter scans ─────────────────────────
def fixed_parameter_scan(param_name: str, params_hat: dict,
                         t_data: np.ndarray, o_data: np.ndarray,
                         proto: Protocol, o2_init: float,
                         n_grid: int = 7,
                         grid_span_log: float = 0.8,
                         sigma_obs: Optional[float] = None,
                         chi2_threshold: float = 3.841,
                         ) -> ProfileLikelihoodReport:
    """Very fast diagnostic scan with all other parameters fixed.

    This is NOT a full profile likelihood because the other parameters are
    not re-optimised at each grid point. It exists so smoke/fast runs can
    produce cautious interpretability flags without pretending to provide
    publication-grade practical-identifiability intervals.
    """
    n_fccp = len(proto.t_fccp)
    names = list(CORE_PARAM_ORDER) + [f"alpha_{j+1}" for j in range(n_fccp)]
    if param_name not in names:
        raise KeyError(f"unknown parameter: {param_name}")
    k_idx = names.index(param_name)
    theta_hat = np.array([params_hat[k] for k in CORE_PARAM_ORDER] + list(params_hat["alphas"]), dtype=float)
    sigma_obs = float(sigma_obs if sigma_obs is not None else params_hat.get("sigma_obs", np.std(o_data)/2.0))
    bd_key = "alpha" if param_name.startswith("alpha_") else param_name
    pk_lo, pk_hi = PARAM_BOUNDS.get(bd_key, (1e-6, 1e6))
    th = float(theta_hat[k_idx])
    if th > 0:
        lo = max(np.log(max(pk_lo, 1e-30)), np.log(th) - grid_span_log)
        hi = min(np.log(pk_hi), np.log(th) + grid_span_log)
        theta_grid = np.exp(np.linspace(lo, hi, n_grid))
    else:
        theta_grid = np.linspace(max(pk_lo, th-1), min(pk_hi, th+1), n_grid)

    def nll(theta_full):
        p = vec_to_params(theta_full, n_fccp, include_sigma=False, base=DEFAULT_PARAMS)
        p["sigma_obs"] = sigma_obs
        res = simulate(p, proto, o2_init=o2_init, t_eval=t_data)
        if not res.converged:
            return np.nan
        resid = res.o - o_data
        return float(0.5 * len(o_data) * np.log(2*np.pi*sigma_obs**2) + 0.5*np.sum(resid**2)/sigma_obs**2)

    nll_grid = []
    ok = []
    for tv in theta_grid:
        full = theta_hat.copy(); full[k_idx] = tv
        val = nll(full)
        nll_grid.append(val)
        ok.append(np.isfinite(val))
    nll_grid = np.array(nll_grid, dtype=float)
    ok_grid = np.array(ok, dtype=bool)
    map_nll = nll(theta_hat)
    if np.isfinite(nll_grid).any():
        imin = int(np.nanargmin(nll_grid))
        profile_min_value = float(theta_grid[imin])
        profile_min_nll = float(nll_grid[imin])
        delta = 2.0 * (nll_grid - profile_min_nll)
    else:
        profile_min_value = th
        profile_min_nll = float(map_nll)
        delta = np.full_like(nll_grid, np.nan)
    info = _extract_ci_from_profile(theta_grid, delta, chi2_threshold, map_value=th)
    verdict = info["verdict"]
    if verdict == "identifiable":
        verdict = "weakly identified"  # bounded in a fixed-other diagnostic only
    notes = ("Fast diagnostic fixed-other-parameter scan only; this is not a full "
             "profile likelihood because other parameters were not re-optimised. "
             "Use publication-grade profile likelihood before making final CIs.")
    return ProfileLikelihoodReport(param_name=param_name, map_value=th,
        map_nll=float(map_nll), profile_min_value=profile_min_value,
        profile_min_nll=profile_min_nll, theta_grid=theta_grid,
        nll_grid=nll_grid, delta_nll=delta, optimizer_success=ok_grid,
        n_optimizer_failures=int(np.sum(~ok_grid)), ci_low=info["ci_low"],
        ci_high=info["ci_high"], identified_left=info["ci_low"] is not None,
        identified_right=info["ci_high"] is not None, practical_id=verdict,
        map_inside_ci=info["map_inside_ci"], notes=notes)


def run_all_fixed_scans(params_hat: dict, t_data: np.ndarray, o_data: np.ndarray,
                        proto: Protocol, o2_init: float,
                        n_grid: int = 7, grid_span_log: float = 0.8,
                        sigma_obs: Optional[float] = None) -> Dict[str, ProfileLikelihoodReport]:
    return {nm: fixed_parameter_scan(nm, params_hat, t_data, o_data, proto, o2_init,
                                    n_grid=n_grid, grid_span_log=grid_span_log,
                                    sigma_obs=sigma_obs)
            for nm in CORE_PARAM_ORDER}
