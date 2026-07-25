"""Tests for analysis.validation.

Verifies:
  * the canonical `parametric_bootstrap_predictive_check` returns coverage
  * the deprecated alias `parametric_bootstrap_ppc` still works AND emits
    a DeprecationWarning
  * the dual result keys (canonical + legacy) and explicit_disclaimer
    are present
"""
from __future__ import annotations
import warnings
import numpy as np
import pytest

pytestmark = pytest.mark.slow  # bootstrap/holdout validation is computational.

from core.reduced_model import simulate, DEFAULT_PARAMS, Protocol
from analysis.validation import (
    parametric_bootstrap_predictive_check,
    parametric_bootstrap_ppc,    # deprecated alias
    within_trace_holdout,
    EXPLICIT_DISCLAIMER,
)


@pytest.fixture
def synthetic_setup():
    proto = Protocol(t_start=0.0, t_oligo=300.0, t_fccp=[480.0],
                     t_inhibit=660.0, t_end=820.0, k_step=2.0)
    t = np.linspace(0.0, proto.t_end, 200)
    res = simulate(DEFAULT_PARAMS, proto, o2_init=200.0, t_eval=t)
    rng = np.random.default_rng(0)
    o = res.o + rng.normal(0, 0.5, size=res.o.shape)
    return t, o, proto


def test_canonical_name_returns_coverage_and_disclaimer(synthetic_setup):
    """The canonical function returns both result keys and the disclaimer."""
    t, o, proto = synthetic_setup
    out = parametric_bootstrap_predictive_check(
        t, o, proto, DEFAULT_PARAMS, n_boot=20, seed=0)
    # Both keys must be present
    assert "parametric_bootstrap_coverage_90" in out
    assert "coverage_90" in out
    cov = float(out["parametric_bootstrap_coverage_90"])
    assert 0.0 <= cov <= 1.0
    assert out["lo90"].shape == t.shape
    assert out["hi90"].shape == t.shape
    assert np.all(out["lo90"] <= out["median"])
    assert np.all(out["median"] <= out["hi90"])
    # Explicit disclaimer must be present and mention "NOT" + "posterior"
    assert "explicit_disclaimer" in out
    assert "NOT" in out["explicit_disclaimer"]
    assert "posterior" in out["explicit_disclaimer"].lower()


def test_deprecated_alias_still_works_and_warns(synthetic_setup):
    """The legacy `parametric_bootstrap_ppc` must keep working but emit
    a DeprecationWarning explaining the rename."""
    t, o, proto = synthetic_setup
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = parametric_bootstrap_ppc(
            t, o, proto, DEFAULT_PARAMS, n_boot=10, seed=0)
        # Must include exactly one DeprecationWarning naming both
        # functions
        deprec = [wi for wi in w
                   if issubclass(wi.category, DeprecationWarning)]
        assert len(deprec) >= 1
        msg = str(deprec[0].message)
        assert "parametric_bootstrap_ppc" in msg
        assert "parametric_bootstrap_predictive_check" in msg
    assert "parametric_bootstrap_coverage_90" in out


@pytest.mark.slow
def test_within_trace_holdout_runs(synthetic_setup):
    """within_trace_holdout calls calibrate_de internally → slow."""
    t, o, proto = synthetic_setup
    out = within_trace_holdout(t, o, proto, train_frac=0.7,
                                maxiter=10, popsize=6, seed=0)
    assert isinstance(out, dict)
    assert "rmse_train" in out
    assert "rmse_test" in out
    assert "interpretation_note" in out
