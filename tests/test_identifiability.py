"""Tests for analysis.identifiability._extract_ci_from_profile.

Covers Section F point 9 (CHUNK 3 preview, but the helper landed in CHUNK 1):
  * two-sided identifiable
  * one-sided
  * flat / non-identifiable
  * optimizer-failure entries (NaN) handled correctly (no false crossings)
  * MAP-inside-CI checks
"""
from __future__ import annotations
import numpy as np
import pytest

pytestmark = pytest.mark.slow  # FIM/profile utilities are computational diagnostics.

from analysis.identifiability import _extract_ci_from_profile


def test_two_sided_identifiable():
    """A clear V-shape: delta crosses 3.84 on both sides."""
    theta = np.linspace(0.5, 1.5, 11)
    delta = 50.0 * (theta - 1.0) ** 2  # max ~ 12.5 > 3.841
    out = _extract_ci_from_profile(theta, delta, chi2_threshold=3.841,
                                    map_value=1.0)
    assert out["verdict"] == "identifiable"
    assert out["ci_low"] is not None and out["ci_low"] < 1.0
    assert out["ci_high"] is not None and out["ci_high"] > 1.0
    assert out["map_inside_ci"] is True
    assert out["n_optimizer_failures"] == 0


def test_one_sided_bound():
    """Profile rises only on one side; hits a flat boundary on the other."""
    theta = np.linspace(0.5, 2.0, 16)
    # left side flat at 0; right side rises sharply above 1.0
    delta = np.where(theta < 1.0, 0.0, 20.0 * (theta - 1.0) ** 2)
    out = _extract_ci_from_profile(theta, delta, map_value=1.0)
    assert out["verdict"] == "one-sided"
    assert out["ci_low"] is None
    assert out["ci_high"] is not None


def test_flat_is_non_identifiable():
    """A perfectly flat profile yields 'non-identifiable' (NOT 'unresolved'
    because the entries are finite — the profile is well-resolved, just flat)."""
    theta = np.linspace(0.5, 2.0, 16)
    delta = np.zeros_like(theta)
    out = _extract_ci_from_profile(theta, delta, map_value=1.0)
    assert out["verdict"] == "non-identifiable"
    assert out["ci_low"] is None
    assert out["ci_high"] is None


def test_optimizer_failures_are_skipped_not_treated_as_crossings():
    """A spike of NaN entries (from inner-optimisation failure) must NOT
    be misread as the profile crossing the threshold.

    Construction: a basin around index 5, then NaN at index 7, then small
    finite rise at index 8 that does NOT cross 3.84. The walk to the right
    should NOT report a crossing at index 7 just because that entry is NaN.
    """
    theta = np.linspace(0.5, 1.5, 11)
    delta = np.array([5.0, 4.0, 2.5, 1.0, 0.5, 0.0, 0.5, np.nan,
                      1.0, 2.0, 3.0])
    out = _extract_ci_from_profile(theta, delta, chi2_threshold=3.841,
                                    map_value=1.0)
    # Right-side walk: skips index 7 (NaN), then sees indices 8/9/10
    # which are 1.0/2.0/3.0 — none above 3.84 — so ci_high must be None.
    assert out["ci_high"] is None
    # Left-side walk should find a crossing at index 0 (5.0 > 3.841).
    assert out["ci_low"] is not None
    assert out["n_optimizer_failures"] == 1


def test_map_outside_ci_is_flagged():
    """If the inner re-optimisation finds a strictly better fit at theta != MAP,
    map_inside_ci must be False (the helper reports the discrepancy honestly)."""
    # Centre the parabola at 1.0, MAP claimed at 1.8 (far outside CI).
    theta = np.linspace(0.5, 1.5, 21)
    delta = 80.0 * (theta - 1.0) ** 2          # crosses at ~ 1.0 ± 0.22
    out = _extract_ci_from_profile(theta, delta, map_value=1.8)
    assert out["verdict"] == "identifiable"
    assert out["map_inside_ci"] is False


def test_all_nan_returns_unresolved():
    theta = np.linspace(0.5, 2.0, 11)
    delta = np.full_like(theta, np.nan)
    out = _extract_ci_from_profile(theta, delta, map_value=1.0)
    assert out["verdict"] == "unresolved"
    assert out["ci_low"] is None
    assert out["ci_high"] is None


# ── New CHUNK-3 tests ─────────────────────────────────────────────────────
def test_fim_report_has_raw_and_clipped_fields():
    """FIMReport must populate eigvals_raw, eigvals_clipped, condition_raw,
    condition_clipped, and warnings; backward-compat aliases must work."""
    from core.reduced_model import simulate, DEFAULT_PARAMS, Protocol
    from analysis.identifiability import fisher_information, EIG_CLIP_FLOOR

    proto = Protocol(t_start=0.0, t_oligo=300.0, t_fccp=[480.0],
                     t_inhibit=660.0, t_end=820.0, k_step=2.0)
    t = np.linspace(0.0, proto.t_end, 100)
    res = simulate(DEFAULT_PARAMS, proto, o2_init=200.0, t_eval=t)
    rng = np.random.default_rng(0)
    o = res.o + rng.normal(0, 0.5, size=res.o.shape)
    rep = fisher_information(DEFAULT_PARAMS, t, o, proto, o2_init=200.0)

    # New fields exist
    assert hasattr(rep, "eigvals_raw")
    assert hasattr(rep, "eigvals_clipped")
    assert hasattr(rep, "condition_raw")
    assert hasattr(rep, "condition_clipped")
    assert hasattr(rep, "warnings")

    # Eigvals are sorted ascending
    assert np.all(np.diff(rep.eigvals_raw) >= -1e-12)
    assert np.all(np.diff(rep.eigvals_clipped) >= -1e-12)

    # Clipped eigvals respect the floor
    assert np.all(rep.eigvals_clipped >= EIG_CLIP_FLOOR - 1e-30)

    # Backward-compat aliases
    assert np.allclose(rep.eigvals, rep.eigvals_raw)
    assert rep.condition == rep.condition_raw

    # Both condition numbers are positive
    assert rep.condition_raw > 0
    assert rep.condition_clipped > 0


@pytest.mark.slow
def test_profile_likelihood_carries_provenance():
    """A profile-likelihood run on a single parameter must populate all
    new provenance fields."""
    from core.reduced_model import simulate, DEFAULT_PARAMS, Protocol
    from analysis.identifiability import profile_likelihood

    proto = Protocol(t_start=0.0, t_oligo=300.0, t_fccp=[480.0],
                     t_inhibit=660.0, t_end=820.0, k_step=2.0)
    t = np.linspace(0.0, proto.t_end, 80)
    sim = simulate(DEFAULT_PARAMS, proto, o2_init=200.0, t_eval=t)
    rng = np.random.default_rng(0)
    o = sim.o + rng.normal(0, 0.5, size=sim.o.shape)

    rep = profile_likelihood(
        "V_max", DEFAULT_PARAMS, t, o, proto, o2_init=200.0,
        n_grid=7, grid_span_log=0.6, maxiter=10,
        adaptive_extend=False, n_restarts_constrained=1,
    )

    # Provenance fields present
    for f in ("map_value", "map_nll", "profile_min_value", "profile_min_nll",
              "optimizer_success", "n_optimizer_failures",
              "map_inside_ci", "notes"):
        assert hasattr(rep, f), f"missing field {f}"

    # Re-optimisation NLL must be <= MAP NLL (re-optim can only improve)
    assert rep.profile_min_nll <= rep.map_nll + 1e-6

    # optimizer_success has the right shape
    assert len(rep.optimizer_success) == len(rep.theta_grid)
    assert rep.n_optimizer_failures == int(np.sum(~rep.optimizer_success))


def test_extract_ci_handles_nan_failures_explicitly():
    """A NaN entry must be counted as an optimizer failure and NOT
    treated as a crossing of the chi^2 threshold (no spike-as-evidence)."""
    theta = np.linspace(0.5, 1.5, 11)
    # Smooth low basin around the centre, NaN at index 7, gentle rise
    # at indices 8-10 that does NOT cross 3.841.
    delta = np.array([5.0, 4.0, 2.5, 1.0, 0.5, 0.0, 0.5, np.nan,
                      1.0, 2.0, 3.0])
    out = _extract_ci_from_profile(theta, delta, chi2_threshold=3.841,
                                    map_value=1.0)
    # Right walk should NOT report a crossing at index 7 (NaN) or
    # anywhere in indices 8-10 (none > 3.841).
    assert out["ci_high"] is None
    # Left walk finds index 0 (delta=5.0 > 3.841)
    assert out["ci_low"] is not None
    assert out["n_optimizer_failures"] == 1
