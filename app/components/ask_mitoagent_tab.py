"""Ask MitoAgent Streamlit component."""
from __future__ import annotations
from agent.ask_agent import ask

EXAMPLES = [
    "Which parameters should I trust?",
    "Why is my FCCP response low?",
    "Can this trace suggest Complex IV dysfunction?",
    "What follow-up experiment would reduce uncertainty?",
    "Why did calibration fail or fit poorly?",
    "Does the LLM produce scientific results?",
]


def render(st, report=None):
    st.header("Ask MitoAgent / Interpretation Assistant")
    st.caption("Ask questions about the current deterministic backend outputs. Full purpose and guardrails are documented in the Help Hub.")
    st.caption("The optional LLM layer does not perform numerical inference. All simulations, estimates, diagnostics, sensitivity indices, validation outputs, and figures are produced by deterministic Python backend modules.")
    example = st.selectbox("Example questions", EXAMPLES, help="Choose an example or type your own question below.")
    question = st.text_input("Your question", value=example, help="Ask about interpretation, missing analyses, parameter trustworthiness, uncertainty, follow-up design, or workflow steps.")
    mode = st.selectbox("Answer mode", ["deterministic/offline", "LLM-assisted (if configured)"], help="Deterministic/offline mode is fully reproducible and does not need an API key. LLM-assisted mode is only for language explanation if a provider is configured.")
    if st.button("Ask MitoAgent", type="primary", help="Generate an interpretation using only structured backend outputs and safety rules."):
        out = ask(question, report, llm_assisted=mode.startswith("LLM"))
        st.subheader("Answer")
        st.markdown(f"<div class='mito-card'><p>{out['answer']}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='mito-badge badge-blue'>Mode: {out['answer_mode']}</span> <span class='mito-badge badge-yellow'>Route: {out.get('route','overview')}</span>", unsafe_allow_html=True)
        if out.get("backend_evidence_used"):
            st.markdown("**Backend evidence used**")
            for item in out["backend_evidence_used"]:
                st.markdown(f"- {item}")
        if out.get("caveats"):
            st.markdown("**Caveats**")
            for item in out["caveats"]:
                st.markdown(f"- {item}")
        if out.get("unsupported_claims_refused"):
            st.warning("Unsupported claim(s) refused: " + ", ".join(out["unsupported_claims_refused"]))
        st.success("Recommended next action: " + out["recommended_next_action"])
        with st.expander("Machine-readable answer payload", expanded=False):
            st.caption("This JSON is for reproducibility/export only. The answer above is the user-facing interpretation.")
            st.code(__import__("json").dumps(out, indent=2, default=str), language="json")
