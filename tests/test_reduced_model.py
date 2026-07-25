"""CHUNK 4 tests for the current reduced OCR-informed model formulation."""
from __future__ import annotations

import numpy as np

from core.reduced_model import DEFAULT_PARAMS, Protocol, rhs, simulate, smooth_step
from core.protocols import make_protocol, protocol_metadata


def test_protocol_helper_sorts_fccp_and_records_scope():
    proto = make_protocol(t_oligo=100, t_fccp=[250, 200], t_inhibit=400, t_end=500)
    assert proto.t_fccp == [200.0, 250.0]
    meta = protocol_metadata(proto)
    assert meta["n_fccp"] == 2
    assert meta["model_scope"] == "reduced_ocr_informed_civ_mediated_ocr"


def test_kappa_is_latent_and_can_exceed_one_after_fccp():
    proto = Protocol(t_start=0, t_oligo=50, t_fccp=[100], t_inhibit=180, t_end=220)
    p = dict(DEFAULT_PARAMS)
    p["alphas"] = [2.0]
    res = simulate(p, proto, o2_init=180, t_eval=np.linspace(0, 220, 300))
    assert res.converged
    assert np.nanmax(res.kappa) > 1.0
    assert np.all(np.isfinite(res.kappa))


def test_rhs_has_no_discontinuous_state_reset_at_inhibition():
    proto = Protocol(t_start=0, t_oligo=50, t_fccp=[100], t_inhibit=150, t_end=200)
    p = dict(DEFAULT_PARAMS)
    y = np.array([p["r0"], 170.0, 1.0])
    before = rhs(149.9, y, p, proto)
    after = rhs(150.1, y, p, proto)
    assert np.all(np.isfinite(before))
    assert np.all(np.isfinite(after))
    assert "r_attenuate" not in p


def test_smooth_step_half_height_and_bounds():
    assert 0.49 < smooth_step(10.0, 10.0, 2.0) < 0.51
    assert smooth_step(-100.0, 0.0, 2.0) < 1e-6
    assert smooth_step(100.0, 0.0, 2.0) > 1 - 1e-6
