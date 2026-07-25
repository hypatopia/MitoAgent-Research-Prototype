"""Deterministic hypothesis-prioritization summaries for OCR traces."""
from __future__ import annotations
from typing import Any, Dict, List

def _warnings(report: Dict[str, Any]) -> List[str]:
    out=[]
    for msgs in (report.get("warnings_by_category") or {}).values():
        out.extend(msgs or [])
    return out

def generate_hypothesis_summary(report: Dict[str, Any] | None) -> Dict[str, Any]:
    """Generate cautious candidate hypotheses from structured outputs only."""
    if not report:
        return {
            "status": "not_generated",
            "label": "Candidate hypothesis requiring experimental confirmation",
            "observed_phenotype": [],
            "candidate_interpretations": [],
            "identifiability_caveats": ["No analysis report was supplied."],
            "recommended_follow_up": ["Run data loading, preprocessing, calibration, identifiability, sensitivity, and validation first."],
        }
    data = report.get("data") or {}
    calib = report.get("calibration") or {}
    ident = report.get("identifiability") or {}
    sens = report.get("sensitivity") or {}
    validation = report.get("validation") or {}
    phenotype = [
        f"Trace contains {data.get('n_samples', 'unknown')} samples and {data.get('n_fccp', 'unknown')} FCCP injection(s).",
    ]
    if calib:
        phenotype.append(f"Calibration RMSE is {calib.get('rmse_calib', 'not reported')} nmol/mL in the diagnostic calibration window.")
    if validation:
        cov = validation.get("parametric_bootstrap_coverage_90", validation.get("coverage_90"))
        if cov is not None:
            phenotype.append(f"Parametric-bootstrap 90% predictive-interval coverage is {float(cov)*100:.1f}%.")
    interpretations = [
        "Dominant phase-level OCR differences may indicate altered respiratory capacity, altered uncoupling/FCCP response, or upstream supply effects.",
        "Effective CIV-mediated OCR limitation cannot be uniquely separated from upstream supply or coupling limitations using OCR alone.",
    ]
    if sens:
        interpretations.append("Sensitivity results can prioritize which parameters or phases are high-information, but do not prove identifiability.")
    caveats = [
        "All hypotheses are exploratory and require experimental confirmation.",
        "OCR-only limitation applies; additional observables are needed for mechanistic separation.",
    ]
    if ident and "fim" in ident:
        cond = ident["fim"].get("condition_raw", ident["fim"].get("condition"))
        caveats.append(f"FIM condition number is approximately {float(cond):.2e}; direct inverse-FIM interpretation is not warranted if the matrix is ill-conditioned.")
    caveats.extend(_warnings(report))
    follow = [
        "Add membrane-potential proxy or redox-state measurement if coupling/supply ambiguity is central.",
        "Use a targeted Complex IV activity assay if CIV dysfunction is the biological question.",
        "Increase biological and technical replicates before disease/control generalization.",
        "Inspect FCCP plateau duration and sampling density around transitions.",
    ]
    return {
        "status": "generated",
        "label": "Candidate hypothesis requiring experimental confirmation",
        "observed_phenotype": phenotype,
        "candidate_interpretations": interpretations,
        "identifiability_caveats": caveats,
        "recommended_follow_up": follow,
    }
