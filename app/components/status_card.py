"""Analysis status card component."""
from __future__ import annotations
from agent.reporting import build_analysis_status


def _badge_class(status: str) -> str:
    s = (status or "").lower()
    if "passed" in s or "completed" in s or "generated" in s or "deterministic" in s:
        return "badge-pass"
    if "warning" in s or "weak" in s or "exploratory" in s:
        return "badge-yellow"
    if "failed" in s or "unresolved" in s:
        return "badge-red"
    return "badge-blue"


def render(st, report=None):
    st.subheader("Analysis Status")
    st.caption("Dashboard-style progress summary for the current session.")

    # Tier badge — surfaces whether numbers shown elsewhere on this page
    # may legitimately be quoted in a report. Belt-and-braces with the
    # sidebar badge; reviewers will read the page they are reading.
    tier = (report or {}).get("tier") or (report or {}).get("mode")
    reportable = (report or {}).get("reportable")
    if tier is not None:
        badge_cls = "badge-pass" if reportable else "badge-yellow"
        rep_text = "Reportable" if reportable else "NOT reportable"
        st.markdown(
            f"<div class='mito-card' style='margin-bottom:0.5rem'>"
            f"<div class='small-muted'>Run tier</div>"
            f"<span class='mito-badge {badge_cls}'>{tier} · {rep_text}</span>"
            f"</div>", unsafe_allow_html=True)

    status = build_analysis_status(report)
    cols = st.columns(3)
    for i, (name, val) in enumerate(status.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
<div class="mito-card">
  <div class="small-muted">{name}</div>
  <span class="mito-badge {_badge_class(val)}">{val}</span>
</div>
                """,
                unsafe_allow_html=True,
            )
