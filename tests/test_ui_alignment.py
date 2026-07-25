"""Tests verifying the Streamlit UI is aligned with `core.run_settings`.

These check the contract — not the rendered HTML — that the engineering
remediation requires of the UI:

  * The publication tier opts in to genuine profile likelihoods
    (`profile_real=True`); fast and smoke do not.
  * Only the publication tier is reportable.
  * Within-trace holdout is refit-based whenever budgets allow.
  * Sobol N_base at the publication tier is at the 512+ threshold above
    which first-order indices are stable.

The Streamlit page functions themselves read `run_cfg` (set in
`_desktop_sidebar`) and pass these fields to the analysis routines. If
the contract holds in `core/run_settings.py`, the UI pages are aligned.
"""
from __future__ import annotations

import importlib


def test_streamlit_imports_run_settings():
    """The Streamlit app must import run_settings — not hardcode budgets."""
    mod = importlib.import_module("app.streamlit_app")
    # Both the lookup helper and the tier table must be importable from the
    # app module (re-exported via `from core.run_settings import ...`).
    assert hasattr(mod, "get_settings")
    assert hasattr(mod, "TIERS")


def test_only_publication_tier_is_reportable():
    """The reportable flag is the gating predicate the UI shows as a
    'Reportable' / 'NOT reportable' badge. Only `publication` is."""
    from core.run_settings import get_settings
    assert get_settings("smoke").reportable is False
    assert get_settings("fast").reportable is False
    assert get_settings("publication").reportable is True


def test_publication_tier_runs_genuine_profile_likelihoods():
    """The Identifiability page now branches on `run_cfg.profile_real`."""
    from core.run_settings import get_settings
    assert get_settings("publication").profile_real is True
    assert get_settings("fast").profile_real is False
    assert get_settings("smoke").profile_real is False


def test_publication_sobol_n_base_is_above_stability_threshold():
    """Publication Sobol N_base must be >= 512 (the benchmarked threshold
    below which first-order indices can be negative on this model)."""
    from core.run_settings import get_settings
    pub = get_settings("publication")
    assert pub.sobol_n_base >= 512


def test_publication_within_trace_holdout_refits():
    """The Validation page now reads `within_trace_refit` from run_cfg."""
    from core.run_settings import get_settings
    assert get_settings("publication").within_trace_refit is True
    assert get_settings("publication").within_trace_de_maxiter >= 25


def test_status_card_shows_tier_when_present():
    """The status card module reads `tier` and `reportable` fields from
    the structured report. Make sure the keys are documented in the
    render function path."""
    src = importlib.import_module("app.components.status_card")
    code = open(src.__file__).read()
    # Must look for 'tier' and 'reportable' keys.
    assert "tier" in code and "reportable" in code, (
        "status_card.render must surface 'tier' and 'reportable' so users "
        "can see whether displayed numbers are reportable.")


def test_manuscript_figures_page_is_wired():
    """The Streamlit app must expose a Manuscript Figures page that can
    regenerate Figs 1-10 from the current `results/` tree."""
    mod = importlib.import_module("app.streamlit_app")
    # Page function exists.
    assert hasattr(mod, "page_manuscript_figures")
    # Figure inventory contains the manuscript-figure scripts.
    assert hasattr(mod, "_MANUSCRIPT_FIGS")
    scripts = {script for _, script, _ in mod._MANUSCRIPT_FIGS}
    for n in range(1, 11):
        assert f"make_fig_step{n}.py" in scripts, (
            f"Manuscript-figures page must include make_fig_step{n}.py")
    # Status helper returns the same set of scripts.
    status = mod._figure_input_status()
    for n in range(1, 11):
        assert f"make_fig_step{n}.py" in status


def test_manuscript_figure_script_runner_works():
    """The UI's `_run_figure_script` helper must actually invoke a figure
    script as a subprocess and return its return code + log. We use a
    schematic figure (no data inputs) so the test is fast and robust."""
    mod = importlib.import_module("app.streamlit_app")
    ok, log = mod._run_figure_script("make_fig_step8.py", timeout_s=60)
    assert ok, f"make_fig_step8.py failed: {log[-400:]}"
    assert "Saved" in log or "saved" in log.lower()
