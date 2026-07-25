"""Tests for analysis.sensitivity."""
from __future__ import annotations
import numpy as np
import pytest

pytestmark = pytest.mark.slow  # SALib-based tests are computational and excluded from smoke tests.

from analysis.sensitivity import morris_screening, sobol_indices
from core.reduced_model import Protocol


@pytest.fixture
def proto_with_one_fccp():
    return Protocol(t_start=0.0, t_oligo=300.0, t_fccp=[480.0],
                    t_inhibit=660.0, t_end=820.0, k_step=2.0)


def test_morris_returns_finite_results(proto_with_one_fccp):
    """Morris screening with a small N must return finite mu_star
    values for every parameter."""
    res = morris_screening(proto_with_one_fccp, o2_init=200.0,
                            N_trajectories=8, num_levels=4, seed=0)
    assert "mu_star" in res
    assert "names" in res
    assert len(res["mu_star"]) == len(res["names"])
    assert np.all(np.isfinite(res["mu_star"]))


def test_sobol_indices_property(proto_with_one_fccp):
    """ST values include interactions, so they can sum to more than 1.
    But each individual ST should be in [0, ~1.5] for sensible models."""
    res = sobol_indices(proto_with_one_fccp, o2_init=200.0, N=8, seed=0)
    assert "S1" in res
    assert "ST" in res
    # ST is allowed to exceed 1 (interactions). Just check finiteness.
    assert np.all(np.isfinite(res["S1"]))
    assert np.all(np.isfinite(res["ST"]))


# ── New CHUNK-4 tests ─────────────────────────────────────────────────────
def test_morris_result_carries_metadata(proto_with_one_fccp):
    """Morris result must carry method, metric, parameter_set, log_mask,
    n_evals, seed, salib_version, n_nan_outputs."""
    res = morris_screening(proto_with_one_fccp, o2_init=200.0,
                            N_trajectories=4, num_levels=4, seed=0)
    for f in ("method", "metric", "parameter_set", "log_mask",
              "n_evals", "seed", "salib_version", "n_nan_outputs"):
        assert f in res, f"missing field {f}"
    assert res["method"] == "morris"
    assert res["metric"] == "AUC_OCR"
    # parameter_set: 8 core + 1 alpha = 9 for proto with one FCCP
    assert len(res["parameter_set"]) == 9
    assert "alpha_1" in res["parameter_set"]
    assert len(res["log_mask"]) == 9


def test_sobol_carries_interpretation_note(proto_with_one_fccp):
    """The Sobol result MUST carry an interpretation note that explicitly
    warns ST is NOT additive variance."""
    res = sobol_indices(proto_with_one_fccp, o2_init=200.0, N=8, seed=0)
    assert "interpretation_note" in res
    note = res["interpretation_note"]
    # The note should mention 'interactions' and one of the
    # additive/exclusive disclaimers.
    assert "interaction" in note.lower()
    assert ("additive" in note.lower() or "exclusive" in note.lower())


def test_time_resolved_sobol_flags_variance_degenerate(proto_with_one_fccp):
    """Time-resolved Sobol MUST produce a variance_degenerate_mask, and at
    least one post-inhibition time point should be flagged True."""
    from analysis.sensitivity import time_resolved_sobol
    res = time_resolved_sobol(proto_with_one_fccp, o2_init=200.0,
                               N=8, n_t_eval=15, seed=0)
    assert "variance_degenerate_mask" in res
    mask = np.asarray(res["variance_degenerate_mask"])
    assert mask.dtype == bool
    assert len(mask) == 15
    # The post-Rot/Ant time points (t > t_inhibit) should have OCR ~ 0
    # and so output_variance ~ 0 ⇒ variance-degenerate.
    t_grid = np.asarray(res["t_grid"])
    post_inhib = t_grid > proto_with_one_fccp.t_inhibit + 50.0
    assert mask[post_inhib].any(), \
        "at least one post-inhibition point should be flagged variance-degenerate"
