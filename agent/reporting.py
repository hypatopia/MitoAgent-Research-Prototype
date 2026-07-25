"""Report assembly helpers for MitoAgent."""
from __future__ import annotations
from typing import Any, Dict
from agent.hypothesis import generate_hypothesis_summary
from agent.design_guidance import generate_design_guidance

def build_analysis_status(report: Dict[str, Any] | None) -> Dict[str, str]:
    report = report or {}
    warnings = report.get("warnings_by_category") or {}
    def has(cat): return bool(warnings.get(cat))
    ident = report.get("identifiability") or {}
    status = {
        "Data parsing": "warning" if has("data_pipeline") else ("passed" if report.get("data") else "not run"),
        "Calibration": "passed" if report.get("calibration") else "not run",
        "Numerical stability": "warning" if has("numerical_stability") else ("passed" if report.get("stability") else "not run"),
        "Identifiability": "weak" if has("identifiability") or ident else "not run",
        "Sensitivity": "completed" if report.get("sensitivity") else "not run",
        "Validation": "warning" if has("validation_noise_model") else ("completed" if report.get("validation") else "not run"),
        "Hypothesis summary": "exploratory only" if report else "not generated",
        "Experimental-design guidance": "generated" if report else "not generated",
        "Ask MitoAgent": "deterministic/offline",
    }
    return status

def enrich_report(report: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(report or {})
    out.setdefault("analysis_status", build_analysis_status(out))
    out.setdefault("hypothesis_prioritization", generate_hypothesis_summary(out))
    out.setdefault("experimental_design_guidance", generate_design_guidance(out))
    return out
