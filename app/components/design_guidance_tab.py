"""Experimental-design guidance Streamlit component."""
from __future__ import annotations
from agent.design_guidance import generate_design_guidance


def render(st, report=None):
    st.header("Experimental-design guidance")
    st.caption("Runbook and usage guidance are in the Help Hub. This page converts uncertainty sources into practical follow-up design actions.")
    g = generate_design_guidance(report)
    st.subheader("Main uncertainty sources")
    for src in g.get("main_uncertainty_sources", []):
        st.markdown(f"- {src}")

    st.subheader("Recommended design actions")
    for i, rec in enumerate(g.get("recommendations", []), start=1):
        with st.container():
            st.markdown(f"""
<div class="mito-card">
<h4>{i}. {rec.get('recommendation','Recommendation')}</h4>
<p><strong>Why it helps:</strong> {rec.get('why_it_helps','')}</p>
</div>
            """, unsafe_allow_html=True)
    st.warning("Design guidance is not a biological conclusion. It is a planning aid for reducing uncertainty in future OCR or multi-observable experiments.")
    with st.expander("Machine-readable export payload", expanded=False):
        st.caption("This JSON is for reproducibility/export only. The main panel above is the human-readable view.")
        st.code(__import__("json").dumps(g, indent=2, default=str), language="json")
