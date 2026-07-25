"""Safety and interpretation guardrails for MitoAgent.

The deterministic backend performs all numerical analyses. The optional
natural-language layer may only route questions, summarize structured backend
outputs, and explain caveats. It must not invent results, diagnose disease, or
turn weak/non-identifiable parameters into biological endpoints.
"""
from __future__ import annotations
from typing import Dict, List

UNSUPPORTED_CLAIMS: Dict[str, str] = {
    "disease_diagnosis": (
        "No. MitoAgent cannot diagnose Alzheimer’s disease, neurodegeneration, "
        "or any disease from OCR-only traces. It can summarize OCR phenotype "
        "patterns and suggest candidate hypotheses requiring experimental confirmation."
    ),
    "complex_iv_proof": (
        "No. OCR stress-test traces cannot prove isolated Complex IV dysfunction. "
        "They can support cautious CIV-mediated OCR hypotheses only when calibration, "
        "identifiability, sensitivity, and validation diagnostics support that framing."
    ),
    "kappa_membrane_potential": (
        "kappa is a latent effective respiratory-drive / OCR-permissiveness factor. "
        "It is not measured membrane potential, true protonmotive force, pH, redox "
        "state, or a proton-gradient measurement."
    ),
    "ocr_only_mechanism": (
        "OCR-only data cannot uniquely separate CIV-mediated OCR capacity, upstream "
        "supply, coupling, membrane-potential, redox, pH, or proton-gradient effects "
        "without additional measurements."
    ),
    "llm_numerical_inference": (
        "The optional LLM layer does not estimate parameters, compute diagnostics, "
        "or produce numerical scientific results."
    ),
}

CAVEATS = [
    "Exploratory interpretation only",
    "Candidate hypothesis requiring experimental confirmation",
    "OCR-only limitation applies",
]

WARNING_TYPES = {
    "data_pipeline": "Data parsing, chamber selection, event labels, or preprocessing warning.",
    "numerical_stability": "Solver convergence, finite-state, tolerance, or stability warning.",
    "identifiability": "FIM/profile-likelihood warning; do not overinterpret parameters.",
    "validation_noise_model": "Validation or observation-noise-model warning.",
    "unsupported_claim": "Unsupported disease/mechanistic claim was refused or qualified.",
}

CALIBRATION_INTERPRETATION_RULE = (
    "Calibration results use deterministic SSE with post-hoc within-trace "
    "observational-noise estimation. Do not describe these results as MCMC, "
    "posterior, hierarchical Bayesian, or biological validation unless those "
    "analyses were explicitly run and documented."
)

MODEL_SCOPE_STATEMENT = (
    "The 3-state OCR-informed model represents CIV-mediated OCR in intact "
    "respiratory-chain context. It is interpretable but phenomenological and "
    "does not reconstruct full ETC/proton dynamics."
)

def screen_question(question: str) -> List[str]:
    """Return guardrail flags triggered by a user question."""
    q = (question or "").lower()
    flags: List[str] = []
    if any(w in q for w in ["diagnose", "diagnosis", "alzheimer", "disease", "neurodegeneration"]):
        flags.append("disease_diagnosis")
    if "complex iv" in q and any(w in q for w in ["prove", "confirmed", "diagnosis", "dysfunction", "defect"]):
        flags.append("complex_iv_proof")
    if "kappa" in q and any(w in q for w in ["membrane", "delta", "ψ", "potential", "protonmotive", "ph", "redox", "gradient"]):
        flags.append("kappa_membrane_potential")
    if any(w in q for w in ["mechanism", "distinguish", "separate", "identify all", "prove"]):
        flags.append("ocr_only_mechanism")
    if "llm" in q and any(w in q for w in ["calculate", "estimate", "infer parameter", "produce result", "compute"]):
        flags.append("llm_numerical_inference")
    return list(dict.fromkeys(flags))

def unsupported_claim_messages(flags: List[str]) -> List[str]:
    return [UNSUPPORTED_CLAIMS[f] for f in flags if f in UNSUPPORTED_CLAIMS]
