"""
core/run_settings.py
====================
Centralized numerical settings for MitoAgent.

Tiers
-----
* smoke: import/sanity testing only; not scientific.
* fast: development and UI iteration; diagnostic only; not manuscript-reportable.
* publication_real_data: recommended tier for real Oroboros results intended for tables, figures, and Results text.

The publication_real_data tier uses stronger deterministic calibration, genuine profile likelihoods with nuisance-parameter re-optimization, Morris/Sobol sensitivity budgets, and refit-based validation settings. Exact budgets are recorded in result provenance. The legacy tier name ``publication`` is retained as an alias for ``publication_real_data``.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class RunSettings:
    """Numerical budgets for one pipeline tier."""
    tier: str

    # ── Calibration (differential evolution + L-BFGS-B polish) ───────────
    de_maxiter: int
    de_popsize: int
    de_polish: bool
    # Downsample the trace to this many points before calibration (None =
    # use the full preprocessed trace). Downsampling is recorded in provenance;
    # publication_real_data uses a documented 200-point grid for optimizer stability/cost,
    # while full-trace RMSE is still reported on the original data.
    calib_n_downsample: int | None

    # ── Identifiability ──────────────────────────────────────────────────
    # If True, run genuine profile likelihoods (all other parameters
    # re-optimised at every grid point). If False, run the cheap
    # fixed-parameter scan (clearly labelled, fast tier only).
    profile_real: bool
    profile_n_grid: int
    profile_maxiter: int
    profile_grid_span_log: float
    profile_adaptive_extend: bool
    profile_n_restarts_constrained: int
    # Optional downsample for the identifiability time grid (None = full).
    identif_n_downsample: int | None

    # ── Sensitivity ──────────────────────────────────────────────────────
    morris_trajectories: int
    sobol_n_base: int
    time_resolved_sobol_n_base: int
    time_resolved_sobol_n_t: int

    # ── Validation ───────────────────────────────────────────────────────
    bootstrap_n_boot: int
    within_trace_refit: bool          # real refit-based holdout?
    within_trace_de_maxiter: int
    within_trace_de_popsize: int

    # ── Scope ────────────────────────────────────────────────────────────
    # Whether this tier is allowed to produce numbers that appear in the
    # manuscript. Only the publication tier is.
    reportable: bool

    def as_dict(self) -> Dict:
        return asdict(self)


# ── Canonical tiers ───────────────────────────────────────────────────────
# diagnostic and fast tiers are for workflow checks only. publication_real_data
# is the recommended tier for real Oroboros results intended for a manuscript.
SMOKE = RunSettings(
    tier="smoke",
    de_maxiter=1, de_popsize=1, de_polish=False, calib_n_downsample=25,
    profile_real=False, profile_n_grid=3, profile_maxiter=5,
    profile_grid_span_log=0.6, profile_adaptive_extend=False,
    profile_n_restarts_constrained=1, identif_n_downsample=25,
    morris_trajectories=2, sobol_n_base=4,
    time_resolved_sobol_n_base=4, time_resolved_sobol_n_t=8,
    bootstrap_n_boot=5, within_trace_refit=False,
    within_trace_de_maxiter=2, within_trace_de_popsize=2,
    reportable=False,
)

FAST = RunSettings(
    tier="fast",
    de_maxiter=5, de_popsize=3, de_polish=False, calib_n_downsample=120,
    profile_real=False, profile_n_grid=7, profile_maxiter=15,
    profile_grid_span_log=0.8, profile_adaptive_extend=False,
    profile_n_restarts_constrained=1, identif_n_downsample=120,
    morris_trajectories=4, sobol_n_base=16,
    time_resolved_sobol_n_base=8, time_resolved_sobol_n_t=12,
    bootstrap_n_boot=40, within_trace_refit=False,
    within_trace_de_maxiter=5, within_trace_de_popsize=3,
    reportable=False,
)

PUBLICATION_REAL_DATA = RunSettings(
    tier="publication_real_data",
    de_maxiter=60, de_popsize=15, de_polish=True, calib_n_downsample=200,
    profile_real=True, profile_n_grid=25, profile_maxiter=40,
    profile_grid_span_log=1.5, profile_adaptive_extend=True,
    profile_n_restarts_constrained=3, identif_n_downsample=200,
    morris_trajectories=20, sobol_n_base=512,
    time_resolved_sobol_n_base=128, time_resolved_sobol_n_t=24,
    bootstrap_n_boot=500, within_trace_refit=True,
    within_trace_de_maxiter=40, within_trace_de_popsize=12,
    reportable=True,
)

# Backward-compatible alias used by older scripts.
PUBLICATION = PUBLICATION_REAL_DATA

TIERS: Dict[str, RunSettings] = {
    "smoke": SMOKE,
    "fast": FAST,
    "publication": PUBLICATION,
    "publication_real_data": PUBLICATION_REAL_DATA,
}


def get_settings(tier: str) -> RunSettings:
    """Look up a settings tier by name.

    Valid tiers: ``smoke``, ``fast``, ``publication``, and
    ``publication_real_data``. ``publication`` is retained as an alias for
    ``publication_real_data``.
    """
    key = (tier or "").strip().lower()
    if key not in TIERS:
        raise ValueError(
            f"unknown run tier {tier!r}; expected one of {sorted(TIERS)}")
    return TIERS[key]
