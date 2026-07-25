"""Deterministic keyword router for Ask MitoAgent."""
from __future__ import annotations

def route_question(question: str) -> str:
    q = (question or "").lower()
    if any(w in q for w in ["identif", "trust", "profile", "fim", "interpretable", "weak", "one-sided"]):
        return "identifiability"
    if any(w in q for w in ["sensitivity", "sobol", "morris", "total-order", "phase informative"]):
        return "sensitivity"
    if any(w in q for w in ["validation", "bootstrap", "holdout", "transfer", "predictive"]):
        return "validation"
    if any(w in q for w in ["calibration", "fit", "rmse", "failed", "residual"]):
        return "calibration"
    if any(w in q for w in ["experiment", "follow-up", "measurement", "design", "reduce uncertainty"]):
        return "design_guidance"
    if any(w in q for w in ["hypothesis", "fccp", "oligomycin", "basal", "rot", "antimycin", "complex iv", "civ"]):
        return "hypothesis"
    if any(w in q for w in ["llm", "agent", "api key", "natural language"]):
        return "llm_role"
    if any(w in q for w in ["help", "workflow", "run", "export", "cli", "api", "dashboard", "ui"]):
        return "workflow"
    return "overview"
