"""Tests for agent/orchestrator.py and agent/cli.py.

The full run_pipeline + JSON-report tests are deferred to CHUNK 5 once the
new run_pipeline signature (with warnings_by_category) is in place.
"""
from __future__ import annotations
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_agent_imports_cleanly():
    """The agent package and its modules must import without raising."""
    import agent
    import agent.tools           # noqa: F401
    import agent.orchestrator    # noqa: F401
    from agent.orchestrator import MitoAgent
    a = MitoAgent(verbose=False)
    assert a is not None


def test_agent_state_is_initialised():
    from agent.orchestrator import MitoAgent, AgentState
    a = MitoAgent(verbose=False)
    assert isinstance(a.state, AgentState)
    assert a.state.chamber is None
    assert a.state.last_calib is None
    assert isinstance(a.state.log, list)


# ── New CHUNK-5 tests ─────────────────────────────────────────────────────
def test_agent_state_has_warning_categories():
    """AgentState must initialise warnings_by_category with the four
    orthogonal categories."""
    from agent.orchestrator import MitoAgent, WARN_CATEGORIES
    a = MitoAgent(verbose=False)
    assert set(a.state.warnings_by_category.keys()) == set(WARN_CATEGORIES)
    for cat in WARN_CATEGORIES:
        assert a.state.warnings_by_category[cat] == []
    assert isinstance(a.state.skipped_analyses, list)


def test_warn_records_into_correct_category():
    """_warn must append to state.warnings_by_category[category] and to
    the tool log."""
    from agent.orchestrator import MitoAgent
    a = MitoAgent(verbose=False)
    a._warn("identifiability", "FIM cond is large")
    a._warn("validation_noise_model", "coverage outside band")
    assert a.state.warnings_by_category["identifiability"] == ["FIM cond is large"]
    assert a.state.warnings_by_category["validation_noise_model"] \
            == ["coverage outside band"]
    # An unknown category falls back to data_pipeline
    a._warn("not_a_category", "fallback test")
    assert "fallback test" in a.state.warnings_by_category["data_pipeline"]
    # Tool log has the warn entries
    warn_entries = [e for e in a.state.log if e.get("tool") == "_warn"]
    assert len(warn_entries) == 3


def test_final_report_carries_metadata():
    """A bare _final_report() (no pipeline) must still carry mode,
    skipped_analyses, diagnostic_thresholds, warnings_by_category,
    warning_counts."""
    from agent.orchestrator import MitoAgent
    a = MitoAgent(verbose=False)
    rep = a._final_report(
        mode="fast",
        skipped_analyses=["profile_likelihoods (publication-mode only)"],
        coverage_band=(0.80, 0.95),
        fim_sloppy_threshold=1e15,
    )
    for k in ("mode", "skipped_analyses", "diagnostic_thresholds",
              "warnings_by_category", "warning_counts"):
        assert k in rep, f"missing {k} in final report"
    assert rep["mode"] == "fast"
    assert rep["diagnostic_thresholds"]["coverage_band"] == [0.8, 0.95]
    assert rep["diagnostic_thresholds"]["fim_sloppy_threshold"] == 1e15
    assert "profile_likelihoods (publication-mode only)" \
            in rep["skipped_analyses"]


def test_save_report_caches_last_report(tmp_path):
    """save_report must serialise the cached _last_report (no re-run)."""
    import json
    from agent.orchestrator import MitoAgent
    a = MitoAgent(verbose=False)
    rep = a._final_report(mode="fast", skipped_analyses=["x"])
    out = tmp_path / "rep.json"
    a.save_report(str(out))
    saved = json.loads(out.read_text())
    assert saved["mode"] == "fast"
    assert "x" in saved["skipped_analyses"]
    assert "warnings_by_category" in saved
