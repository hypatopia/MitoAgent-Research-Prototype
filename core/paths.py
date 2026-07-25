"""
core/paths.py
=============
Single source of truth for package directory locations.

Historically the bundled demonstration Excel files lived under ``data_excel/``.
They were renamed to ``data_samples/`` but the rename was applied inconsistently
across the codebase, tests, figure scripts, and documentation, which broke the
test suite, the documented CLI command, and the figure-regeneration pipeline.

To prevent that class of bug from recurring, every module that needs a package
path should import it from here rather than hardcoding a string. If the layout
changes again, only this file needs to change.

All paths are absolute and computed relative to the repository root (the parent
of the directory containing this file).
"""
from __future__ import annotations

from pathlib import Path

# Repository root = parent of core/
ROOT: Path = Path(__file__).resolve().parent.parent

# Bundled demonstration datasets (synthetic / representative parser fixtures;
# NOT real biological measurements). Real proprietary OCR datasets are supplied
# by the user at run time and are never committed to this archive.
DATA_SAMPLES: Path = ROOT / "data_samples"

# Drop-in location for the user's own real Oroboros exports.
DATA_REAL: Path = ROOT / "data_real"

# Pipeline outputs.
RESULTS: Path = ROOT / "results"
FIGURES: Path = ROOT / "figures"
FIGURES_FINAL: Path = FIGURES / "final"
MANUSCRIPT: Path = ROOT / "manuscript"
EXECUTION_LOGS: Path = ROOT / "execution_logs"

# Canonical names of the three bundled demonstration datasets.
DEMO_DATASETS = ("dataset_I", "dataset_II", "dataset_III")


def demo_dataset(name: str) -> Path:
    """Absolute path to a bundled demonstration .xlsx by stem name."""
    return DATA_SAMPLES / f"{name}.xlsx"


def ensure_dirs() -> None:
    """Create the output directory tree if it does not yet exist."""
    for sub in ("diagnostics", "calibration", "identifiability",
                "sensitivity", "validation", "agent_reports",
                "calibration_ready", "_legacy_archive"):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    FIGURES_FINAL.mkdir(parents=True, exist_ok=True)
    EXECUTION_LOGS.mkdir(parents=True, exist_ok=True)
    DATA_REAL.mkdir(parents=True, exist_ok=True)
