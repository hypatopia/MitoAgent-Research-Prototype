"""Template-based interpretation of structured MitoAgent backend outputs."""
from __future__ import annotations
from typing import Any, Dict, List
from agent.safety_rules import CAVEATS, screen_question, unsupported_claim_messages
from agent.question_router import route_question
from agent.hypothesis import generate_hypothesis_summary
from agent.design_guidance import generate_design_guidance


def summarize_evidence(report: Dict[str, Any] | None) -> List[str]:
    if not report:
        return []
    mapping = [
        ("data", "parsed events"),
        ("phase_summary", "phase-level OCR summary"),
        ("calibration", "calibration result"),
        ("stability", "numerical diagnostics"),
        ("identifiability", "identifiability summary"),
        ("sensitivity", "sensitivity summary"),
        ("validation", "validation summary"),
        ("warnings_by_category", "warnings"),
    ]
    return [label for key, label in mapping if report.get(key)]


def _missing(report: Dict[str, Any] | None, key: str) -> bool:
    return not bool((report or {}).get(key))


def _fmt_float(x: Any, digits: int = 3) -> str:
    try:
        x = float(x)
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}"
    except Exception:
        return "not available"


def _fim_condition(report: Dict[str, Any]) -> str:
    fim = ((report.get("identifiability") or {}).get("fim") or {})
    return _fmt_float(fim.get("condition_raw") or fim.get("condition"))


def _rmse(report: Dict[str, Any]) -> str:
    c = report.get("calibration") or {}
    return _fmt_float(c.get("rmse_calib") or c.get("rmse") or c.get("rmse_full_trace"))


def interpret_question(question: str, report: Dict[str, Any] | None = None, *, answer_mode: str = "deterministic_offline") -> Dict[str, Any]:
    route = route_question(question)
    refused = screen_question(question)
    if refused:
        answer = " ".join(unsupported_claim_messages(refused))
        next_action = "Review identifiability and add targeted follow-up measurements before making mechanistic or disease claims."
    elif report is None:
        answer = "No completed analysis report was provided, so I can only give workflow guidance. Run the deterministic analysis pipeline first."
        next_action = "Run loading, preprocessing, calibration, identifiability, sensitivity, and validation as needed."
    elif route == "identifiability":
        if _missing(report, "identifiability"):
            answer = "Identifiability analysis has not been run yet. I cannot determine whether this parameter is interpretable from this trace. Please run FIM/profile-likelihood diagnostics first."
            next_action = "Run FIM and publication-grade profile likelihoods."
        else:
            answer = (
                f"Use the parameter badges and profile/FIM outputs to decide what to trust. The current FIM condition number is approximately {_fim_condition(report)}, so any ill-conditioned result should be treated as practical sloppiness rather than a biological endpoint. Parameters with weak, one-sided, flat, or unresolved flags should not be used as standalone biological conclusions."
            )
            next_action = "Inspect profile likelihoods and parameter interpretability badges; rerun publication-grade profiles before reporting parameter-specific claims."
    elif route == "sensitivity":
        if _missing(report, "sensitivity"):
            answer = "Sensitivity analysis has not been run yet. I cannot rank high-information parameters or protocol phases."
            next_action = "Run Morris screening and Sobol diagnostics."
        else:
            answer = "Sensitivity tells you which parameters or phases the simulated OCR output responds to most strongly. It is useful for prioritizing interpretation and experiment design, but it does not prove identifiability. Sobol total-order indices include interactions and are not additive."
            next_action = "Compare sensitivity rankings with profile-likelihood identifiability flags and avoid interpreting variance-degenerate time regions."
    elif route == "validation":
        if _missing(report, "validation"):
            answer = "Validation diagnostics have not been run yet. I cannot assess technical transfer or predictive-envelope compatibility."
            next_action = "Run technical-replicate transfer, intervention-phase holdout, and parametric-bootstrap predictive checks where applicable."
        else:
            answer = "Validation outputs are workflow diagnostics, not biological generalization. Chamber A to B is technical-replicate transfer; the bootstrap check is parametric-bootstrap, not posterior predictive. Coverage warnings should be read as possible noise-model or residual-structure issues."
            next_action = "Inspect residuals, technical transfer metrics, and coverage warnings before using the fit for hypothesis prioritization."
    elif route == "calibration":
        if _missing(report, "calibration"):
            answer = "Calibration has not been run yet, so fitted-parameter interpretation is unavailable."
            next_action = "Run deterministic calibration and inspect residuals and phase-level summaries."
        else:
            answer = (
                f"Calibration currently reports an RMSE of about {_rmse(report)} nmol/mL for the fitted window. If some phases fit poorly, users should not manually force the model to match that region by overinterpreting parameters. They can improve the calibration attempt by increasing downsampled points, DE iterations/population, checking event labels, rerunning with a different seed, using publication mode, and inspecting whether the poor-fit region reflects a model limitation or data/event issue."
            )
            next_action = "Use the Calibration tab's residual plot and 'Improve calibration' checklist; then rerun identifiability before interpreting parameters."
    elif route == "design_guidance":
        dg = generate_design_guidance(report)
        top = dg.get("recommendations", [{}])[0]
        answer = "Experimental-design guidance translates uncertainty into concrete follow-up actions. The recommendations are driven by weak identifiability, missing observables, poor validation diagnostics, unclear event labels, or OCR-only limitations."
        next_action = top.get("recommendation", "Export design guidance.")
    elif route == "hypothesis":
        hyp = generate_hypothesis_summary(report)
        answer = "MitoAgent can prioritize candidate hypotheses from phase-level OCR summaries, calibration, identifiability warnings, sensitivity rankings, and validation diagnostics. A low FCCP response, for example, may be consistent with altered maximal OCR capacity, altered uncoupling/FCCP response, upstream supply limitation, or combinations of these. OCR alone cannot uniquely separate them."
        next_action = hyp.get("recommended_follow_up", ["Export the hypothesis summary."])[0]
    elif route == "llm_role":
        answer = "The optional LLM layer is not the scientific engine. It can help with language, routing, and explanation if configured with an API/provider, but it does not estimate parameters, create numerical results, or override backend diagnostics. Deterministic/offline mode remains the reproducible default."
        next_action = "Use deterministic/offline mode for reproducibility; enable LLM-assisted mode only for language explanation after configuring a provider."
    elif route == "workflow":
        answer = "The UI is best for interactive inspection and guided analysis. The CLI/API are better for batch runs, exact reproducibility, scripted real-data reruns, automated testing, and regenerating manuscript figures/results."
        next_action = "Open the Help Hub for UI versus CLI/API guidance and runbook steps."
    else:
        answer = "MitoAgent reports deterministic backend analyses and cautious interpretation. Use the dashboard/status card to see what has been run, then ask targeted questions about calibration, identifiability, sensitivity, validation, hypothesis prioritization, or design guidance."
        next_action = "Run missing analyses, then ask a targeted question such as 'Which parameters should I trust?' or 'What follow-up experiment would reduce uncertainty?'"
    return {
        "answer_mode": answer_mode,
        "route": route,
        "answer": answer,
        "backend_evidence_used": summarize_evidence(report),
        "unsupported_claims_refused": refused,
        "caveats": list(CAVEATS),
        "recommended_next_action": next_action,
    }
