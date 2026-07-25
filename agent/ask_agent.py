"""Ask MitoAgent deterministic/offline entry point."""
from __future__ import annotations
from typing import Any, Dict
from agent.interpretation import interpret_question

def ask(question: str, report: Dict[str, Any] | None = None, *, llm_assisted: bool = False) -> Dict[str, Any]:
    """Answer a natural-language question using structured backend outputs only."""
    mode = "llm_assisted" if llm_assisted else "deterministic_offline"
    return interpret_question(question, report, answer_mode=mode)
