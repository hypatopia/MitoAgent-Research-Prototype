"""Tests for the reduced bioenergetics model (core/reduced_model.py)."""
from __future__ import annotations
import numpy as np
import pytest

from core.reduced_model import (
    simulate, Protocol, DEFAULT_PARAMS, smooth_step,
)


@pytest.fixture
def basic_proto():
    return Protocol(t_start=0.0, t_oligo=300.0,
                    t_fccp=[480.0], t_inhibit=660.0, t_end=820.0,
                    k_step=2.0)


def test_simulate_integrates_without_error(basic_proto):
    """Plain integration with default parameters returns a converged
    SimResult with the expected array shapes."""
    t_eval = np.linspace(0.0, basic_proto.t_end, 200)
    res = simulate(DEFAULT_PARAMS, basic_proto, o2_init=200.0,
                   t_eval=t_eval)
    assert res.converged, f"integration failed: {getattr(res, 'message', '')}"
    assert res.t.shape == t_eval.shape
    assert res.o.shape == t_eval.shape
    assert np.all(np.isfinite(res.o))


def test_oxygen_is_non_increasing_when_civ_flux_nonneg(basic_proto):
    """With a non-negative CIV flux the oxygen trace must be monotonically
    non-increasing (allowing a small numerical-noise tolerance)."""
    t_eval = np.linspace(0.0, basic_proto.t_end, 400)
    res = simulate(DEFAULT_PARAMS, basic_proto, o2_init=200.0,
                   t_eval=t_eval)
    assert res.converged
    diffs = np.diff(res.o)
    # tolerance scaled to the noise of a stiff ODE solution
    assert (diffs <= 1e-6).all(), \
        f"oxygen has {(diffs > 1e-6).sum()} positive jumps > 1e-6"


def test_reduced_cytochrome_pool_admissible(basic_proto):
    """The reduced cytochrome-c pool variable must remain in [0, c_tot]
    throughout the simulated trace (admissibility, not full conservation)."""
    t_eval = np.linspace(0.0, basic_proto.t_end, 400)
    res = simulate(DEFAULT_PARAMS, basic_proto, o2_init=200.0,
                   t_eval=t_eval)
    assert res.converged
    c_tot = float(DEFAULT_PARAMS["c_tot"])
    assert (res.r >= -1e-6).all()
    assert (res.r <= c_tot + 1e-3).all()


def test_kappa_remains_finite_and_can_exceed_one(basic_proto):
    """kappa is the LATENT effective respiratory-drive factor.
    It is NOT bounded to [0, 1]: FCCP can push it above 1, and oligomycin
    can drag it below 1. We assert only finiteness and a sane upper bound
    (kappa < 5 is comfortable for the protocols we use)."""
    t_eval = np.linspace(0.0, basic_proto.t_end, 400)
    res = simulate(DEFAULT_PARAMS, basic_proto, o2_init=200.0,
                   t_eval=t_eval)
    assert res.converged
    assert np.all(np.isfinite(res.kappa))
    assert (res.kappa >= 0).all()
    assert (res.kappa < 5.0).all()


def test_smooth_step_endpoints():
    """smooth_step must equal 0 well before the step time and 1 well after."""
    t = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = smooth_step(t, t0=0.0, k=5.0)
    assert y[0] < 1e-3
    assert y[-1] > 1 - 1e-3
    assert 0.4 < y[2] < 0.6  # half-height at the step time
