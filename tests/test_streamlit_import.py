"""Smoke test for the Streamlit UI (added in CHUNK 6).

The test is structured to AUTO-SKIP cleanly until both:
  * Streamlit is installed, AND
  * app/streamlit_app.py exists.

This way the suite stays green during CHUNKs 1-5 and starts exercising the
UI as soon as CHUNK 6 lands.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import pytest


def test_streamlit_app_imports():
    streamlit = pytest.importorskip(
        "streamlit",
        reason="Streamlit not installed; install with -r requirements-ui.txt",
    )
    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    if not app_path.exists():
        pytest.skip("app/streamlit_app.py not present (added in CHUNK 6).")

    # Streamlit's bare-import path is fragile (it triggers ScriptRunContext).
    # We compile-check instead: this confirms the file is syntactically valid
    # without running its top-level code.
    import py_compile
    py_compile.compile(str(app_path), doraise=True)
