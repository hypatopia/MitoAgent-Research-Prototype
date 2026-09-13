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
    st.caption("External LLM providers are disabled in the approved public release. Interpretation is deterministic and uses structured backend outputs only.")
    example = st.selectbox("Example questions", EXAMPLES, help="Choose an example or type your own question below.")
    question = st.text_input("Your question", value=example, help="Ask about interpretation, missing analyses, parameter trustworthiness, uncertainty, follow-up design, or workflow steps.")
    mode = "deterministic/offline"
    st.caption("Answer mode: deterministic/offline")
    if st.button("Ask MitoAgent", type="primary", help="Generate an interpretation using only structured backend outputs and safety rules."):
        out = ask(question, report, llm_assisted=False)
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
