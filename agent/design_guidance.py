"""Deterministic experimental-design guidance from MitoAgent diagnostics."""
from __future__ import annotations
from typing import Any, Dict, List

def generate_design_guidance(report: Dict[str, Any] | None) -> Dict[str, Any]:
    if not report:
        return {
            "status": "not_generated",
            "main_uncertainty_sources": ["No completed analysis report was supplied."],
            "recommendations": [{"recommendation": "Run the deterministic analysis pipeline first.", "why_it_helps": "Design guidance requires parsed events, calibration, identifiability, sensitivity, and validation outputs."}],
        }
    ident = report.get("identifiability") or {}
    validation = report.get("validation") or {}
    data = report.get("data") or {}
    sources: List[str] = []
    recs: List[Dict[str, str]] = []
    if ident and "fim" in ident:
        cond = ident["fim"].get("condition_raw", ident["fim"].get("condition", 0.0))
        if cond and float(cond) > 1e12:
            sources.append("Practical sloppiness / weak identifiability from OCR-only data.")
            recs.append({"recommendation": "Add membrane-potential proxy or redox-state measurement.", "why_it_helps": "These observables help separate coupling/drive changes from electron-supply or CIV-mediated OCR capacity."})
            recs.append({"recommendation": "Run profile likelihoods with nuisance-parameter re-optimization.", "why_it_helps": "Profiles are needed before treating fitted parameters as interpretable biological endpoints."})
    if data.get("n_fccp", 0) in (0, 1):
        sources.append("Limited FCCP titration information.")
        recs.append({"recommendation": "Use a longer FCCP plateau or denser sampling near FCCP transitions.", "why_it_helps": "Additional transition/plateau information can improve separation of FCCP-response parameters."})
    cov = None
    if validation:
        cov = validation.get("parametric_bootstrap_coverage_90", validation.get("coverage_90"))
    if cov is not None and not (0.80 <= float(cov) <= 0.95):
        sources.append("Predictive-interval coverage suggests possible observation-noise mismatch.")
        recs.append({"recommendation": "Inspect residuals and consider heteroscedastic or phase-specific noise models.", "why_it_helps": "Residual structure can reveal whether a single iid observation-noise estimate is adequate."})
    recs.extend([
        {"recommendation": "Use targeted Complex IV activity assay for CIV-specific claims.", "why_it_helps": "OCR stress tests do not directly measure isolated Complex IV enzymatic activity."},
        {"recommendation": "Add biological and technical replicates for disease/control or treatment/control studies.", "why_it_helps": "Replicates are required to assess biological generalization rather than single-trace fit quality."},
        {"recommendation": "Record clear event labels for oligomycin, each FCCP dose, and rotenone/antimycin.", "why_it_helps": "Accurate event parsing controls phase-level summaries, model inputs, and diagnostics."},
    ])
    if not sources:
        sources.append("No single dominant uncertainty source was detected in the available diagnostic report; OCR-only limitations still apply.")
    return {"status": "generated", "main_uncertainty_sources": sources, "recommendations": recs}
