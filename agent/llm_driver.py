"""
agent/llm_driver.py
===================
Optional LLM driver for natural-language interaction with the MitoAgent.

Design: ports & adapters.  Users plug in any provider that supports
tool/function calling.  One reference hosted-LLM adapter is implemented:
  * AnthropicAdapter  (Claude family)

An offline deterministic keyword router is used when no supported provider is
configured.

If no API key is configured, the driver falls back to a deterministic
keyword-routing heuristic so the agent works fully offline.

Usage
-----
    from agent.orchestrator import MitoAgent
    from agent.llm_driver    import NaturalLanguageDriver

    agent = MitoAgent()
    nl = NaturalLanguageDriver(agent, provider="anthropic")  # or "offline"
    response = nl.ask("Calibrate the model on dataset_I and tell me which "
                      "parameters I can trust.")
    print(response)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, json, re

from agent.tools         import TOOLS, TOOL_DESCRIPTIONS
from agent.orchestrator  import MitoAgent


# Module-level flag so consumers (e.g. the Streamlit UI) can probe whether
# a supported hosted LLM provider is configured. Set to True only when
# an Anthropic API key is configured. Detection is intentionally
# cheap (env-vars only) — actual provider import is deferred until ask().
LLM_AVAILABLE = False
LLM_RELEASE_STATUS = "disabled_excluded_from_approved_release"



def tool_schemas_anthropic():
    """JSON schemas for the Anthropic tool-use API.  Kept here rather than
    in tools.py so that tools.py has no provider dependency."""
    schemas = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        # Hand-curated minimal schemas; users can extend
        if name == "load_data":
            schema = {
                "name": name, "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path":          {"type": "string"},
                        "chamber_index": {"type": "integer", "default": 0},
                    }, "required": ["path"],
                }
            }
        elif name == "calibrate":
            schema = {
                "name": name, "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string",
                                    "enum": ["de", "staged"], "default": "de"},
                        "n_data": {"type": "integer", "default": 250},
                    },
                }
            }
        elif name == "analyze_identifiability":
            schema = {
                "name": name, "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string",
                                    "enum": ["fim", "profile", "both"]},
                    }, "required": ["method"],
                }
            }
        elif name == "analyze_sensitivity":
            schema = {
                "name": name, "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string",
                                    "enum": ["morris", "sobol", "time_sobol"]},
                        "N":      {"type": "integer", "default": 20},
                    }, "required": ["method"],
                }
            }
        else:
            schema = {"name": name, "description": desc,
                       "input_schema": {"type": "object", "properties": {}}}
        schemas.append(schema)
    return schemas


# ── Offline keyword router (fallback when no LLM is configured) ────────
class OfflineRouter:
    """Keyword-based router used when no LLM provider is configured.
    Surprisingly capable for a focused domain.
    """
    def __init__(self, agent: MitoAgent):
        self.agent = agent

    def ask(self, query: str) -> str:
        q = query.lower()
        outputs = []

        # Match commands
        if re.search(r"\bload\b|\bopen\b|\bread\b", q):
            m = re.search(r"['\"]([^'\"]+\.(?:xlsx|xls|csv|npy))['\"]", query)
            path = m.group(1) if m else None
            if not path:
                outputs.append("Please specify a file path in quotes.")
            else:
                r = self.agent.load(path)
                outputs.append(r["message"])
        if re.search(r"\bcalibrat", q):
            method = "staged" if "fast" in q or "stage" in q else "de"
            r = self.agent.calibrate(method=method)
            outputs.append(r["message"])
        if re.search(r"\bidentifiabilit|\bprofile|\bfim\b", q):
            method = ("profile" if "profile" in q
                       else "both" if "both" in q else "fim")
            r = self.agent.identifiability(method=method)
            outputs.append(r["message"])
        if re.search(r"\bsensitivit|morris|sobol", q):
            method = ("sobol" if "sobol" in q
                       else "time_sobol" if "time" in q else "morris")
            r = self.agent.sensitivity(method=method)
            outputs.append(r["message"])
        if re.search(r"\bvalid|\bppc\b|coverage", q):
            r = self.agent.validate(method="ppc")
            outputs.append(r["message"])
        if re.search(r"\bstability|\bstiff|\bdiagnostic", q):
            r = self.agent.check_stability()
            outputs.append(r["message"])
        if re.search(r"\bpipeline|\bfull|\beverything|\banalys\w+\s+all", q):
            m = re.search(r"['\"]([^'\"]+)['\"]", query)
            if m:
                self.agent.run_pipeline(m.group(1), fast=True)
                outputs.append("Pipeline complete; see report.")
        if not outputs:
            outputs.append("I can: load data, calibrate, check identifiability,"
                            " analyze sensitivity, validate, or run the full"
                            " pipeline. What would you like?")
        return "\n".join(outputs)


# ── Anthropic adapter (sketch; requires `anthropic` package + key) ───────
class AnthropicAdapter:
    """Disabled compatibility stub for the approved public release."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "External LLM providers/tool-calling are disabled and excluded "
            "from the approved public MitoAgent research prototype."
        )


# ── Top-level natural-language driver ─────────────────────────────────────
class NaturalLanguageDriver:
    """Deterministic-only public-release driver."""

    def __init__(self, agent: MitoAgent, provider: str = "auto"):
        if provider not in {"auto", "offline"}:
            raise RuntimeError(
                "External LLM providers/tool-calling are disabled and excluded "
                "from the approved public MitoAgent research prototype."
            )
        self.agent = agent
        self.provider = "offline"
        self.adapter = OfflineRouter(agent)

    def ask(self, query: str) -> str:
        return self.adapter.ask(query)
