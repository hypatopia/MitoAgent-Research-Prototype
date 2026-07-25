"""Hypothesis-prioritization Streamlit component."""
from __future__ import annotations
from agent.hypothesis import generate_hypothesis_summary


def _bullets(st, items):
    for item in items or []:
        st.markdown(f"- {item}")


def render(st, report=None):
    st.header("Hypothesis prioritization")
    st.caption("Runbook and usage guidance are in the Help Hub. This page shows the current session's candidate hypotheses in human-readable form.")
    h = generate_hypothesis_summary(report)
    st.markdown('<span class="mito-badge badge-yellow">Candidate hypothesis requiring experimental confirmation</span> <span class="mito-badge badge-blue">OCR-only limitation applies</span>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Observed phenotype")
        _bullets(st, h.get("observed_phenotype"))
        st.subheader("Identifiability caveats")
        _bullets(st, h.get("identifiability_caveats"))
    with c2:
        st.subheader("Candidate interpretations")
        _bullets(st, h.get("candidate_interpretations"))
        st.subheader("Recommended follow-up")
        _bullets(st, h.get("recommended_follow_up"))

    st.caption("Use this section to prioritize experiments, not to make final biological claims. Disease/control claims require appropriate replicate design and additional observables.")
    with st.expander("Machine-readable export payload", expanded=False):
        st.caption("This JSON is for reproducibility/export only. The main panel above is the human-readable view.")
        st.code(__import__("json").dumps(h, indent=2, default=str), language="json")
