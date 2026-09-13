from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_public_streamlit_has_no_upload_or_persistent_temp_path():
    text = _read("app/streamlit_app.py")
    assert "st.file_uploader" not in text
    assert "NamedTemporaryFile" not in text
    assert "_save_uploaded_file" not in text
    assert "Private or measured datasets must not be uploaded" in text


def test_public_ui_is_demo_only_and_nonreportable():
    text = _read("app/streamlit_app.py")
    assert "PUBLIC DEMO · SYNTHETIC/DEMO DATA · NOT MANUSCRIPT-REPORTABLE" in text
    assert '["fast", "smoke"]' in text
    assert '"Manuscript Figures": page_manuscript_figures' not in text


def test_external_llm_is_disabled_even_if_api_key_exists(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-enable-provider")
    import agent.llm_driver as driver

    assert driver.LLM_AVAILABLE is False
    assert driver.LLM_RELEASE_STATUS == "disabled_excluded_from_approved_release"

    dummy_agent = object()
    with pytest.raises(RuntimeError, match="External LLM providers"):
        driver.NaturalLanguageDriver(dummy_agent, provider="anthropic")

    nl = driver.NaturalLanguageDriver(dummy_agent, provider="auto")
    assert nl.provider == "offline"


def test_ask_agent_rejects_llm_assisted_mode():
    from agent.ask_agent import ask

    with pytest.raises(RuntimeError, match="LLM-assisted mode is disabled"):
        ask("test", {}, llm_assisted=True)


def test_readme_states_public_boundary():
    text = _read("README.md")
    normalized = " ".join(text.split())

    assert "bundled-demo-only" in text
    assert "Do not upload private" in text
    assert "not the canonical manuscript calculation source" in normalized
    assert "disabled and excluded from the approved public release" in normalized
