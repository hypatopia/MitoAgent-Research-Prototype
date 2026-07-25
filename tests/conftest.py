"""Pytest configuration for lightweight smoke checks.

The execution environment used for this project preloads tracing/async plugins
that can leave non-daemon background hooks alive after Streamlit/backend import
smoke tests. Exiting explicitly at session finish keeps
`pytest -q -m "not slow"` deterministic for the reproducibility smoke command.
"""
from __future__ import annotations
import os

def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - pytest hook
    os._exit(int(exitstatus))
