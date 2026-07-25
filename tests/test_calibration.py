"""Tests for calibration/calibrate.py."""
from __future__ import annotations
import numpy as np
import pytest

pytestmark = pytest.mark.slow  # calibration tests invoke optimization and are integration-level.

from core.reduced_model import simulate, DEFAULT_PARAMS, Protocol
from calibration.calibrate import calibrate_de


@pytest.fixture
def synthetic_trace():
    proto = Protocol(t_start=0.0, t_oligo=300.0, t_fccp=[480.0],
                     t_inhibit=660.0, t_end=820.0, k_step=2.0)
    t = np.linspace(0.0, proto.t_end, 200)
    res = simulate(DEFAULT_PARAMS, proto, o2_init=200.0, t_eval=t)
    rng = np.random.default_rng(0)
    o = res.o + rng.normal(0, 0.5, size=res.o.shape)
    return t, o, proto


def test_calibration_objective_finite(synthetic_trace):
    """The SSE objective must return a finite value at the default
    parameter setting (sanity check)."""
    t, o, proto = synthetic_trace
    sim = simulate(DEFAULT_PARAMS, proto, o2_init=float(o[0]), t_eval=t)
    sse = float(np.sum((sim.o - o) ** 2))
    assert np.isfinite(sse)
    assert sse >= 0


@pytest.mark.slow
def test_calibrate_de_runs_end_to_end(synthetic_trace):
    """A short DE run on synthetic data must converge and return a
    populated CalibrationResult."""
    t, o, proto = synthetic_trace
    res = calibrate_de(t, o, proto,
                       maxiter=20, popsize=8, seed=0)
    assert res.success
    assert np.isfinite(res.objective)
    assert "V_max" in res.params
    assert "alphas" in res.params
    assert len(res.params["alphas"]) == len(proto.t_fccp)
