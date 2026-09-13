"""Ask MitoAgent deterministic/offline entry point."""
from __future__ import annotations
from typing import Any, Dict
from agent.interpretation import interpret_question

def ask(question: str, report: Dict[str, Any] | None = None, *, llm_assisted: bool = False) -> Dict[str, Any]:
    """Answer using deterministic structured backend outputs only."""
    if llm_assisted:
        raise RuntimeError(
            "LLM-assisted mode is disabled and excluded from the approved public release."
        )
    return interpret_question(question, report, answer_mode="deterministic_offline")
