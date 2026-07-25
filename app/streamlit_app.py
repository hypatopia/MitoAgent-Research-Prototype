"""
app/streamlit_app.py
====================
Graphical interface for the MitoAgent backend.

This is a FRONT END over the reproducible backend pipeline. Every tab
calls the same functions used by `run_all.py` and the CLI. There is no
separate scientific logic in this file.

Run with:
    streamlit run app/streamlit_app.py

Tabs (Section J of the project plan):
    A. Data Upload / Load Dataset
    B. Preprocessing
    C. Model Simulation
    D. Calibration
    E. Numerical Diagnostics
    F. Identifiability
    G. Sensitivity Analysis
    H. Validation
    I. Report Export
    J. Optional Natural-Language Agent
"""
from __future__ import annotations
import io
import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))
import sys
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# In normal use this file is executed by `streamlit run`, where
# __name__ == "__main__". During smoke tests we import the module with
# `python -c "import app.streamlit_app"`; in that mode a lightweight dummy
# Streamlit object prevents top-level UI construction from starting
# Streamlit runtime threads that can keep pytest alive after tests finish.
class _DummyContext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            if name == "columns":
                n = args[0] if args and isinstance(args[0], int) else 2
                return [_DummyContext() for _ in range(n)]
            if name == "tabs":
                labels = args[0] if args else []
                return [_DummyContext() for _ in labels]
            if name in {"radio", "selectbox"}:
                opts = kwargs.get("options") or (args[1] if len(args) > 1 else [])
                return opts[0] if opts else None
            if name == "number_input": return kwargs.get("value", 0)
            if name == "multiselect": return kwargs.get("default", [])
            if name == "checkbox": return kwargs.get("value", False)
            if name == "text_input": return kwargs.get("value", "")
            if name == "button": return False
            if name == "download_button": return False
            if name == "file_uploader": return None
            if name in {"expander", "container", "spinner"}: return _DummyContext()
            return None
        return _method

class _DummySessionState(dict):
    def __getattr__(self, key): return self.get(key)
    def __setattr__(self, key, value): self[key] = value

class _DummyComponents:
    class v1:
        @staticmethod
        def html(*args, **kwargs):
            return None

class _DummyStreamlit(_DummyContext):
    session_state = _DummySessionState()
    sidebar = _DummyContext()
    components = _DummyComponents()

if __name__ == "__main__":
    import streamlit as st
else:
    st = _DummyStreamlit()

# Backend imports — UI code MUST NOT duplicate this logic
from data_io.loader import load_excel, load_dataset
from data_io.preprocess import preprocess
from core.reduced_model import (
    simulate, Protocol, DEFAULT_PARAMS, PARAM_BOUNDS, CORE_PARAM_ORDER,
)
from core.diagnostics import detect_instability
from core.run_settings import get_settings, TIERS
from calibration.calibrate import calibrate_de
from calibration.phase import compute_phase_summary
from analysis.identifiability import (
    fisher_information, profile_likelihood, fixed_parameter_scan, EIG_CLIP_FLOOR,
)
from analysis.sensitivity import (
    morris_screening, sobol_indices, time_resolved_sobol,
)
from analysis.validation import (
    parametric_bootstrap_predictive_check, within_trace_holdout,
    EXPLICIT_DISCLAIMER,
)
from agent.orchestrator import MitoAgent
from agent.reporting import enrich_report, build_analysis_status
from app.components.status_card import render as render_status_card
from app.components.hypothesis_tab import render as render_hypothesis_tab
from app.components.design_guidance_tab import render as render_design_guidance_tab
from app.components.ask_mitoagent_tab import render as render_ask_mitoagent_tab
from app.components.help_tab import render as render_help_tab
from app.components.faq_tab import render as render_faq_tab
from app.ui_utils import (
    display_param, display_param_verbose, display_param_help, label_params_in_df, display_section,
    add_event_markers_ax, add_event_markers_plotly, inject_professional_css,
    report_to_html, report_to_yaml, report_to_pdf_bytes, humanize_value,
    trace_plotly, render_plot, go, make_subplots, _dark_plot_layout,
    DATA_BLUE, MODEL_ORANGE, ACCENT_PURPLE, WARN_ORANGE,
)


# ── Page setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MitoAgent — diagnostic-gated bioenergetics analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_professional_css(st)


# ── Session state initialisation ────────────────────────────────────────
def _init_state():
    defaults = {
        "chamber_raw":   None,   # pre-preprocess chamber
        "chamber":       None,   # post-preprocess chamber
        "preprocess_issues": [],
        "current_params": None,  # parameters currently in the simulation tab
        "calib_result":   None,  # most recent CalibrationResult dict
        "fim_report":     None,
        "profile_reports": {},
        "sens_morris":    None,
        "sens_sobol":     None,
        "sens_trs":       None,
        "validation_ppc": None,
        "validation_wt":  None,
        "stability":      None,
        "loaded_path":    None,
        "loaded_label":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Desktop sidebar/navigation is built after helper functions ───────────


# ── Helpers ─────────────────────────────────────────────────────────────
def _proto_from_chamber(ch) -> Protocol:
    return ch.to_protocol()


def _params_from_calib(calib: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a model-parameter dict from a CalibrationResult-like dict."""
    p = dict(calib.get("params", {}))
    if "alphas" not in p:
        p["alphas"] = list(calib.get("alphas", []))
    if "sigma_obs" not in p and "sigma_obs" in calib:
        p["sigma_obs"] = float(calib["sigma_obs"])
    return p


def _save_uploaded_file(uploaded_file) -> Optional[str]:
    """Write a Streamlit UploadedFile to a temp path and return the path."""
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix or ".xlsx"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def _trace_fig(t: np.ndarray, o: np.ndarray, *,
                title: str = "", events: Optional[Dict[str, Any]] = None,
                events_label: str = "intervention",
                extra_lines: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = None,
                ):
    """Interactive trace figure with zoom/pan/export toolbar."""
    fig = trace_plotly(t, o, title=title, events=events, extra_lines=extra_lines)
    if fig is not None:
        return fig
    # Fallback if Plotly is unavailable.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, o, color=DATA_BLUE, lw=0.8, label="O₂ data")
    if extra_lines:
        for tt, oo, lbl in extra_lines:
            ax.plot(tt, oo, lw=1.4, label=lbl)
    if events:
        add_event_markers_ax(ax, events)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"O$_2$ [nmol/mL]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if extra_lines:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def _preprocess_plot(ch_raw, ch):
    """Native Plotly raw/preprocessed comparison with event labels and toolbar."""
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ch_raw.t), y=list(ch_raw.o), mode="lines", name="Raw trace", line=dict(color="#94a3b8", width=1.2)))
    fig.add_trace(go.Scatter(x=list(ch.t), y=list(ch.o), mode="lines", name="Preprocessed trace", line=dict(color=DATA_BLUE, width=1.8)))
    add_event_markers_plotly(fig, {"oligo": ch_raw.t_oligo, "fccp": list(ch_raw.t_fccp), "inhib": ch_raw.t_inhibit})
    _dark_plot_layout(fig, title="Raw vs preprocessed O₂ trace", y_title="O₂ [nmol/mL]")
    return fig


def _simulation_plot(res, proto):
    """Native Plotly simulation panels with event labels and toolbar."""
    if go is None or make_subplots is None:
        return None
    fig = make_subplots(rows=1, cols=3, subplot_titles=("O₂(t)", "OCR(t)", "κ(t) effective drive"))
    fig.add_trace(go.Scatter(x=list(res.t), y=list(res.o), mode="lines", name="O₂", line=dict(color=DATA_BLUE, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(res.t), y=list(res.OCR), mode="lines", name="OCR", line=dict(color=MODEL_ORANGE, width=1.8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(res.t), y=list(res.kappa), mode="lines", name="κ", line=dict(color=ACCENT_PURPLE, width=1.8)), row=1, col=3)
    for c in (1, 2, 3):
        add_event_markers_plotly(fig, {"oligo": proto.t_oligo, "fccp": list(proto.t_fccp), "inhib": proto.t_inhibit}, row=1, col=c)
    _dark_plot_layout(fig, title="Model simulation")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Time [s]", tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"), linecolor="#64748b")
    fig.update_yaxes(title="O₂ [nmol/mL]", row=1, col=1, tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"), linecolor="#64748b")
    fig.update_yaxes(title="OCR [nmol/mL/s]", row=1, col=2, tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"), linecolor="#64748b")
    fig.update_yaxes(title="κ", row=1, col=3, tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"), linecolor="#64748b")
    return fig


def _validation_ppc_plot(ppc, proto):
    if go is None:
        return None
    t_b = np.asarray(ppc["t"], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_b, y=ppc["hi90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t_b, y=ppc["lo90"], mode="lines", fill="tonexty", fillcolor="rgba(0,114,189,.16)", line=dict(width=0), name="90% predictive envelope"))
    fig.add_trace(go.Scatter(x=t_b, y=ppc["o_hat"], mode="lines", name="Model prediction", line=dict(color=DATA_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=t_b, y=ppc["o_data"], mode="markers", name="Data", marker=dict(color="#e5e7eb", size=4, opacity=.65)))
    add_event_markers_plotly(fig, {"oligo": getattr(proto, "t_oligo", None), "fccp": list(getattr(proto, "t_fccp", [])), "inhib": getattr(proto, "t_inhibit", None)})
    _dark_plot_layout(fig, title="Parametric-bootstrap predictive check", y_title="O₂ [nmol/mL]")
    return fig


def _validation_holdout_plot(wt, ch, proto):
    if go is None:
        return None
    t_full = np.asarray(wt.get("t_full", ch.t), dtype=float)
    o_data = np.asarray(wt.get("o_data", ch.o), dtype=float)
    o_pred = np.asarray(wt.get("o_pred", []), dtype=float)
    fig = go.Figure()
    cut = int(wt.get("n_train", int(0.7 * len(t_full))))
    if 0 < cut < len(t_full):
        fig.add_vrect(x0=t_full[0], x1=t_full[cut], fillcolor="rgba(148,163,184,.18)", line_width=0, annotation_text="training window", annotation_position="top left")
    fig.add_trace(go.Scatter(x=t_full, y=o_data, mode="markers", name="Data", marker=dict(color="#e5e7eb", size=4, opacity=.65)))
    if len(o_pred) == len(t_full):
        fig.add_trace(go.Scatter(x=t_full, y=o_pred, mode="lines", name="Prediction", line=dict(color=MODEL_ORANGE, width=2)))
    add_event_markers_plotly(fig, {"oligo": getattr(proto, "t_oligo", None), "fccp": list(getattr(proto, "t_fccp", [])), "inhib": getattr(proto, "t_inhibit", None)})
    _dark_plot_layout(fig, title="Within-trace holdout diagnostic", y_title="O₂ [nmol/mL]")
    return fig



def _fim_eigen_plot(rep):
    if go is None:
        return None
    ev_raw = np.asarray(rep.eigvals_raw, dtype=float)
    ev_clip = np.asarray(rep.eigvals_clipped, dtype=float)
    ranks = np.arange(1, len(ev_raw) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[display_param(n) for n in rep.param_names],
                         y=np.log10(np.maximum(np.abs(ev_raw), 1e-30)),
                         name="Raw eigenvalues", marker_color=DATA_BLUE,
                         hovertemplate="%{x}<br>log10(|λ|)=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Bar(x=[display_param(n) for n in rep.param_names],
                         y=np.log10(ev_clip), name="Clipped eigenvalues",
                         marker_color=MODEL_ORANGE,
                         hovertemplate="%{x}<br>log10(λ clipped)=%{y:.3g}<extra></extra>"))
    fig.update_layout(template="plotly_white", title="FIM eigenvalue spectrum", barmode="group",
                      hovermode="x unified", paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(color="#0f172a"),
                      legend=dict(orientation="h", y=1.04, x=0), margin=dict(l=45,r=22,t=60,b=70))
    fig.update_xaxes(title="Parameter", tickangle=-35, gridcolor="#e5e7eb")
    fig.update_yaxes(title="log₁₀ |λ|", gridcolor="#e5e7eb")
    return fig


def _profile_plotly(pr, param_name: str):
    if go is None:
        return None
    ok_mask = pr.optimizer_success.astype(bool)
    finite_ok = ok_mask & ~np.isnan(pr.delta_nll)
    fail_mask = ~ok_mask | np.isnan(pr.delta_nll)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr.theta_grid[finite_ok], y=pr.delta_nll[finite_ok],
                             mode="lines+markers", name="Profile scan",
                             line=dict(color=DATA_BLUE, width=2), marker=dict(size=7),
                             hovertemplate=f"{display_param(param_name)}=%{{x:.4g}}<br>2ΔlogL=%{{y:.4g}}<extra></extra>"))
    if fail_mask.any():
        y_fail = np.where(np.isnan(pr.delta_nll), 0, pr.delta_nll)[fail_mask]
        fig.add_trace(go.Scatter(x=pr.theta_grid[fail_mask], y=y_fail, mode="markers",
                                 name="Optimizer failed / unresolved", marker=dict(color="#d62728", size=9, symbol="x")))
    fig.add_hline(y=3.841, line=dict(color=MODEL_ORANGE, width=1.2, dash="dash"),
                  annotation_text="95% χ² threshold", annotation_position="top right")
    fig.add_vline(x=pr.map_value, line=dict(color=ACCENT_PURPLE, width=1.5),
                  annotation_text="MAP", annotation_position="top")
    if pr.ci_low is not None and pr.ci_high is not None:
        fig.add_vrect(x0=pr.ci_low, x1=pr.ci_high, fillcolor="rgba(0,114,189,.12)", line_width=0,
                      annotation_text="CI", annotation_position="top left")
    fig.update_layout(template="plotly_white", title=f"Profile for {display_param(param_name)} — {pr.practical_id}",
                      hovermode="x unified", paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(color="#0f172a"),
                      legend=dict(orientation="h", y=1.04, x=0), margin=dict(l=45,r=22,t=60,b=45))
    fig.update_xaxes(title=display_param(param_name), gridcolor="#e5e7eb")
    fig.update_yaxes(title="2Δlog L", gridcolor="#e5e7eb")
    return fig


def _morris_plotly(m):
    if go is None:
        return None
    order = np.argsort(m["mu_star"])[::-1]
    names = [display_param(m["names"][i]) for i in order]
    vals = np.asarray(m["mu_star"], dtype=float)[order]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=DATA_BLUE, name="μ*"))
    fig.update_layout(template="plotly_white", title="Morris screening", hovermode="x unified",
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color="#0f172a"), margin=dict(l=45,r=22,t=60,b=75))
    fig.update_xaxes(title="Parameter", tickangle=-35, gridcolor="#e5e7eb")
    fig.update_yaxes(title="μ* (Morris)", gridcolor="#e5e7eb")
    return fig


def _sobol_auc_plotly(s):
    if go is None:
        return None
    names = [display_param(n) for n in s["names"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=np.asarray(s["S1"], dtype=float), name="S1 first-order", marker_color=DATA_BLUE))
    fig.add_trace(go.Bar(x=names, y=np.asarray(s["ST"], dtype=float), name="ST total-order", marker_color=MODEL_ORANGE))
    fig.update_layout(template="plotly_white", title="Sobol sensitivity — AUC(OCR)", barmode="group",
                      hovermode="x unified", paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(color="#0f172a"),
                      legend=dict(orientation="h", y=1.04, x=0), margin=dict(l=45,r=22,t=60,b=75))
    fig.update_xaxes(title="Parameter", tickangle=-35, gridcolor="#e5e7eb")
    fig.update_yaxes(title="Sobol index", gridcolor="#e5e7eb")
    return fig


def _time_resolved_sobol_plotly(ts, proto):
    if go is None:
        return None
    t_grid = np.asarray(ts["t_grid"], dtype=float)
    ST_t = np.asarray(ts["ST_t"], dtype=float)
    mask = np.asarray(ts["variance_degenerate_mask"], dtype=bool)
    colors = [DATA_BLUE, MODEL_ORANGE, ACCENT_PURPLE, WARN_ORANGE, "#A2142F", "#4DBEEE", "#9467bd", "#8c564b", "#e377c2"]
    fig = go.Figure()
    for i, nm in enumerate(ts["names"]):
        fig.add_trace(go.Scatter(x=t_grid, y=ST_t[i], mode="lines", name=display_param(nm),
                                 line=dict(color=colors[i % len(colors)], width=1.8)))
    for k, is_bad in enumerate(mask):
        if is_bad:
            x0 = t_grid[max(k-1, 0)] if k > 0 else t_grid[k]
            x1 = t_grid[min(k+1, len(t_grid)-1)] if k < len(t_grid)-1 else t_grid[k]
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(148,163,184,.18)", line_width=0)
    add_event_markers_plotly(fig, {"oligo": getattr(proto, "t_oligo", None), "fccp": list(getattr(proto, "t_fccp", [])), "inhib": getattr(proto, "t_inhibit", None)})
    fig.update_layout(template="plotly_white", title="Time-resolved Sobol total-order sensitivity",
                      hovermode="x unified", paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(color="#0f172a"),
                      legend=dict(orientation="h", y=1.04, x=0), margin=dict(l=45,r=22,t=60,b=45))
    fig.update_xaxes(title="Time [s]", gridcolor="#e5e7eb")
    fig.update_yaxes(title="ST(t)", gridcolor="#e5e7eb")
    return fig

def _current_structured_report() -> Dict[str, Any]:
    """Build a lightweight report from current Streamlit session state.

    This is for interpretation/display only. It does not rerun analyses.
    """
    tier_name = (run_cfg.tier if "run_cfg" in globals()
                 else (diagnostic_level if "diagnostic_level" in globals()
                       else "fast"))
    is_reportable = bool(run_cfg.reportable) if "run_cfg" in globals() \
                                              else False
    report: Dict[str, Any] = {
        "ok": True,
        "mode": tier_name,
        "tier": tier_name,
        "reportable": is_reportable,
        "data": None,
        "calibration": st.session_state.get("calib_result"),
        "stability": None,
        "identifiability": None,
        "sensitivity": None,
        "validation": st.session_state.get("validation_ppc"),
        "warnings_by_category": {
            "data_pipeline": list(st.session_state.get("preprocess_issues", []) or []),
            "numerical_stability": [],
            "identifiability": ([] if is_reportable else
                                 ["Session is in a non-reportable tier "
                                  f"('{tier_name}'); identifiability "
                                  "numbers shown are diagnostic only."]),
            "validation_noise_model": [],
            "unsupported_claim": [],
        },
    }
    ch = st.session_state.get("chamber") or st.session_state.get("chamber_raw")
    if ch is not None:
        report["data"] = {
            "chamber_label": getattr(ch, "label", "unknown"),
            "n_samples": int(len(ch.t)),
            "t_start": float(ch.t[0]),
            "t_end": float(ch.t[-1]),
            "t_oligo": getattr(ch, "t_oligo", None),
            "t_fccp": [float(x) for x in getattr(ch, "t_fccp", [])],
            "t_inhibit": getattr(ch, "t_inhibit", None),
            "n_fccp": int(len(getattr(ch, "t_fccp", []))),
            "noise_sd_estimate": float(ch.sigma_obs_est) if getattr(ch, "sigma_obs_est", None) else None,
        }
    fim = st.session_state.get("fim_report")
    if fim is not None:
        report["identifiability"] = {
            "fim": {
                "param_names": list(getattr(fim, "param_names", [])),
                "condition_raw": float(getattr(fim, "condition_raw", 0.0)),
                "condition_clipped": float(getattr(fim, "condition_clipped", 0.0)),
                "warnings": list(getattr(fim, "warnings", [])),
            }
        }
    # Per-parameter profile verdicts (genuine or fixed-scan, depending on
    # which path was used at run time).
    prof_reports = st.session_state.get("profile_reports", {}) or {}
    if prof_reports:
        is_genuine = bool(run_cfg.profile_real) if "run_cfg" in globals() \
                                                 else False
        ident = report.get("identifiability") or {}
        params_calib = (_params_from_calib(st.session_state.get("calib_result"))
                        if st.session_state.get("calib_result") else {})
        rows = []
        for nm, pr in prof_reports.items():
            map_v = params_calib.get(nm)
            lo = pr.ci_low if pr.ci_low is not None else float("-inf")
            hi = pr.ci_high if pr.ci_high is not None else float("+inf")
            in_ci = (map_v is not None) and (lo <= float(map_v) <= hi)
            rows.append({
                "parameter": nm,
                "verdict": pr.practical_id,
                "ci_low": pr.ci_low, "ci_high": pr.ci_high,
                "map_value": map_v, "map_in_ci": in_ci,
                "source": ("genuine_profile_likelihood" if is_genuine
                           else "fixed_other_parameter_scan_diagnostic_only"),
            })
        ident["profile_verdicts"] = rows
        ident["profile_source"] = ("genuine_profile_likelihood" if is_genuine
                                    else "fixed_other_parameter_scan")
        report["identifiability"] = ident
    if st.session_state.get("sens_morris") is not None:
        report["sensitivity"] = st.session_state.get("sens_morris")
    if st.session_state.get("stability") is not None:
        rep = st.session_state.get("stability")
        try:
            report["stability"] = asdict(rep)
        except Exception:
            report["stability"] = {"summary": str(rep)}
    if ch is not None and st.session_state.get("calib_result") is not None:
        try:
            report["phase_summary"] = compute_phase_summary(ch, _params_from_calib(st.session_state.get("calib_result")))
        except Exception:
            pass
    return enrich_report(report)



# ── Modern scientific desktop-app shell ─────────────────────────────────
def _nav_style_status(label: str, ok: bool) -> str:
    return "complete" if ok else "pending"


def _state_status_items() -> list[tuple[str, bool]]:
    return [
        ("Data", st.session_state.get("chamber_raw") is not None),
        ("Preprocess", st.session_state.get("chamber") is not None),
        ("Calibration", st.session_state.get("calib_result") is not None),
        ("Diagnostics", st.session_state.get("stability") is not None),
        ("Identifiability", st.session_state.get("fim_report") is not None),
        ("Sensitivity", st.session_state.get("sens_morris") is not None or st.session_state.get("sens_sobol") is not None),
        ("Validation", st.session_state.get("validation_ppc") is not None or st.session_state.get("validation_wt") is not None),
        ("Report", True),
    ]


def _desktop_sidebar():
    global diagnostic_level, cov_lo, cov_hi, fim_thr, run_cfg
    with st.sidebar:
        st.markdown("<div class='desktop-brand'><div class='brand-icon'>🧬</div><div><h2>MitoAgent</h2><span>Scientific OCR Workbench</span></div></div>", unsafe_allow_html=True)
        diagnostic_level = st.selectbox(
            "Analysis mode",
            ["fast", "publication", "smoke"], index=0,
            help=("Tiers are defined in core/run_settings.py.\n\n"
                  "• smoke — CI / import-sanity only. NEVER reportable.\n"
                  "• fast — development / iteration. NEVER reportable.\n"
                  "• publication — benchmark-validated, scientifically "
                  "defensible budgets. The only tier whose UI outputs "
                  "may be carried into a manuscript table."),
        )
        run_cfg = get_settings(diagnostic_level)
        # Tier badge — clear visual signal of whether displayed numbers
        # may be quoted in a report. badge-pass = reportable;
        # badge-yellow = explicitly non-reportable.
        badge_class = "badge-pass" if run_cfg.reportable else "badge-yellow"
        reportable_text = ("Reportable" if run_cfg.reportable
                           else "NOT reportable")
        st.markdown(
            f"<span class='mito-badge {badge_class}'>"
            f"Tier: {run_cfg.tier} · {reportable_text}</span>",
            unsafe_allow_html=True)
        with st.expander("Tier budgets (see core/run_settings.py)", expanded=False):
            st.markdown(
                f"- DE calibration: maxiter={run_cfg.de_maxiter}, "
                f"popsize={run_cfg.de_popsize}, "
                f"polish={'on' if run_cfg.de_polish else 'off'}\n"
                f"- Calibration downsample: {run_cfg.calib_n_downsample or 'full trace'}\n"
                f"- Profile likelihoods: "
                f"{'GENUINE (inner re-opt)' if run_cfg.profile_real else 'fixed-other-parameter scan only'}, "
                f"n_grid={run_cfg.profile_n_grid}, "
                f"maxiter={run_cfg.profile_maxiter}\n"
                f"- Morris: N_traj={run_cfg.morris_trajectories}\n"
                f"- Sobol AUC: N_base={run_cfg.sobol_n_base}\n"
                f"- Time-resolved Sobol: N_base="
                f"{run_cfg.time_resolved_sobol_n_base} × "
                f"{run_cfg.time_resolved_sobol_n_t} t-points\n"
                f"- Parametric bootstrap: n_boot={run_cfg.bootstrap_n_boot}\n"
                f"- Within-trace holdout refit: "
                f"{'YES (DE maxiter='+str(run_cfg.within_trace_de_maxiter)+')' if run_cfg.within_trace_refit else 'NO (residual split)'}"
            )
        with st.expander("Warning thresholds", expanded=False):
            cov_lo = st.number_input("Coverage warning lower bound", 0.0, 1.0, 0.80, 0.01)
            cov_hi = st.number_input("Coverage warning upper bound", 0.0, 1.0, 0.95, 0.01)
            fim_thr = st.number_input("FIM sloppiness threshold", min_value=1e6, max_value=1e30, value=1e15, step=1e14, format="%.1e")

        if "nav_page" not in st.session_state:
            st.session_state.nav_page = "Dashboard"
        groups = {
            "PROJECT & DATA": ["Dashboard", "Load Data", "Event Parsing & Preprocessing"],
            "MODEL & ANALYSIS": ["Model Simulation", "Calibration", "Numerical Diagnostics", "Identifiability", "Sensitivity", "Validation"],
            "INTERPRETATION": ["Hypothesis Prioritization", "Experimental Design Guidance", "Ask MitoAgent"],
            "OUTPUTS": ["Report Builder", "Manuscript Figures", "Export Results"],
            "LEARNING & SETTINGS": ["Help / Runbook", "FAQ", "Optional NL Agent / LLM Settings"],
        }
        for group, pages in groups.items():
            st.markdown(f"<div class='nav-group-title'>{group}</div>", unsafe_allow_html=True)
            for page in pages:
                selected = st.session_state.nav_page == page
                if st.button(("● " if selected else "○ ") + page, key="nav_" + page, use_container_width=True):
                    st.session_state.nav_page = page
                    selected = True
        st.markdown("<div class='workflow-rail-title'>Workflow status</div>", unsafe_allow_html=True)
        html = ["<div class='desktop-rail'>"]
        for i, (label, ok) in enumerate(_state_status_items(), start=1):
            cls = "done" if ok else "todo"
            state = "Complete" if ok else "Not run"
            html.append(f"<div class='rail-item {cls}'><span>{i}</span><div><b>{label}</b><small>{state}</small></div></div>")
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)
        st.caption("UI = guided interactive workbench. CLI/API = exact reruns, automation, and reproducibility. See Help / Runbook.")


def _hero(title: str, subtitle: str):
    st.markdown(f"""
    <div class="mito-hero compact-hero">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def _metric_cards(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(f"<div class='metric-card'><span>{label}</span><b>{value}</b><small>{note}</small></div>", unsafe_allow_html=True)


def _active_chamber():
    return st.session_state.get("chamber") or st.session_state.get("chamber_raw")


def page_dashboard():
    _hero("MitoAgent Scientific Workbench", "Desktop-style interface for reproducible mitochondrial OCR modeling, diagnostics, interpretation, and report generation.")
    ch_raw = st.session_state.get("chamber_raw")
    ch = st.session_state.get("chamber")
    report = _current_structured_report()
    render_status_card(st, report)
    st.markdown("### Project snapshot")
    _metric_cards([
        ("Loaded trace", st.session_state.get("loaded_label") or "None", "Current chamber"),
        ("Samples", str(len(_active_chamber().t)) if _active_chamber() is not None else "—", "Active trace"),
        ("FCCP injections", str(len(getattr(_active_chamber(), "t_fccp", []))) if _active_chamber() is not None else "—", "Parsed protocol"),
        ("Tier", run_cfg.tier,
         "Reportable" if run_cfg.reportable else "NOT reportable"),
    ])
    st.markdown("### Recommended next action")
    if ch_raw is None:
        st.info("Start with **Load Data**. Use a demo Excel file or upload a real Oroboros-style CSV/Excel file.")
    elif ch is None:
        st.info("Next: go to **Event Parsing & Preprocessing** and generate the calibration-ready trace.")
    elif st.session_state.get("calib_result") is None:
        st.info("Next: go to **Calibration** and fit the 3-state OCR-informed model.")
    elif st.session_state.get("fim_report") is None:
        st.info("Next: run **Numerical Diagnostics** and **Identifiability** before interpreting parameter values.")
    else:
        st.success("Core workflow outputs are available. You can proceed to Sensitivity, Validation, Interpretation, or Report Builder.")
    if _active_chamber() is not None:
        ch0 = _active_chamber()
        render_plot(st, _trace_fig(ch0.t, ch0.o, title="Active O₂ trace", events={"oligo": ch0.t_oligo, "fccp": list(ch0.t_fccp), "inhib": ch0.t_inhibit}))


def page_load_data():
    _hero("Load Data", "Load demo or real Oroboros-style Excel/CSV data and inspect detected protocol events.")
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        upl = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"], help="Expected columns include time, O₂ trace columns, and recognized event labels.")
    with col2:
        demo_dir = ROOT / "data_samples"
        demo_files = sorted(demo_dir.glob("*.xlsx")) if demo_dir.exists() else []
        demo_choice = st.selectbox("Demo file", ["(none)"] + [f.name for f in demo_files], help="Demo fixtures are synthetic/parser examples, not biological validation data.")
    with col3:
        chamber_idx = st.number_input("Chamber", min_value=0, max_value=8, value=0, step=1, help="0 = Chamber A, 1 = Chamber B, etc.")
    if st.button("Load selected dataset", type="primary"):
        path = _save_uploaded_file(upl) if upl is not None else (str(demo_dir / demo_choice) if demo_choice != "(none)" else None)
        if path is None:
            st.error("No file selected.")
        else:
            try:
                ds = load_dataset(path)
                if chamber_idx >= len(ds.chambers):
                    st.error(f"Chamber index {chamber_idx} out of range; file has {len(ds.chambers)} chamber(s).")
                else:
                    ch = ds.chambers[chamber_idx]
                    st.session_state.chamber_raw = ch
                    st.session_state.chamber = None
                    st.session_state.loaded_path = path
                    st.session_state.loaded_label = ch.label
                    for key in ["calib_result","fim_report","sens_morris","sens_sobol","sens_trs","validation_ppc","validation_wt","stability"]:
                        st.session_state[key] = None
                    st.success(f"Loaded {Path(path).name}, chamber {ch.label} ({len(ch.t)} samples).")
            except Exception as e:
                st.exception(e)
    ch = st.session_state.get("chamber_raw")
    if ch is not None:
        _metric_cards([
            ("Samples", str(len(ch.t)), "Raw trace"),
            ("FCCP injections", str(len(ch.t_fccp)), "Detected events"),
            ("Noise estimate", f"{ch.sigma_obs_est:.3f}" if ch.sigma_obs_est is not None else "—", "Within-trace"),
        ])
        event_rows = [{"Event":"Start","Time [s]":ch.t_start},{"Event":"Oligomycin","Time [s]":ch.t_oligo}] + [{"Event":f"FCCP {i+1}","Time [s]":t} for i,t in enumerate(ch.t_fccp)] + [{"Event":"Rotenone/Antimycin","Time [s]":ch.t_inhibit},{"Event":"End","Time [s]":ch.t_end}]
        st.dataframe(pd.DataFrame(event_rows), hide_index=True, use_container_width=True)
        render_plot(st, _trace_fig(ch.t, ch.o, title=f"Raw trace — chamber {ch.label}", events={"oligo":ch.t_oligo,"fccp":list(ch.t_fccp),"inhib":ch.t_inhibit}))


def page_preprocess():
    _hero("Event Parsing & Preprocessing", "Confirm detected interventions, reject artifacts, and prepare a calibration-ready trace.")
    if st.session_state.chamber_raw is None:
        st.info("Load a dataset first.")
        return
    ch_raw = st.session_state.chamber_raw
    st.markdown("### Parsed events")
    st.dataframe(pd.DataFrame([{"Event":"Oligomycin","Time [s]":ch_raw.t_oligo}] + [{"Event":f"FCCP {i+1}","Time [s]":t} for i,t in enumerate(ch_raw.t_fccp)] + [{"Event":"Rotenone/Antimycin","Time [s]":ch_raw.t_inhibit}]), hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1: do_outliers = st.checkbox("Reject outliers", value=True, help="Exclude extreme points while preserving intervention windows.")
    with c2: n_sigma = st.number_input(display_param_verbose("n_sigma"), min_value=2.0, max_value=8.0, value=4.0, step=0.5, help="Threshold for outlier filtering expressed in σ units.")
    with c3: do_smooth = st.checkbox("Light smoothing", value=False, help="Optional visual/noise smoothing. Use cautiously before calibration.")
    if st.button("Run preprocessing", type="primary"):
        try:
            ch_p, issues = preprocess(ch_raw, do_outliers=do_outliers, n_sigma=n_sigma, do_smooth=do_smooth)
            st.session_state.chamber = ch_p
            st.session_state.preprocess_issues = list(issues)
            st.success(f"Preprocessed {len(ch_p.t)}/{len(ch_raw.t)} samples; {len(issues)} issue(s).")
        except Exception as e:
            st.exception(e)
    if st.session_state.chamber is not None:
        ch = st.session_state.chamber
        render_plot(st, _preprocess_plot(ch_raw, ch))
        st.download_button("Download calibration-ready CSV", pd.DataFrame({"t": ch.t, "o": ch.o}).to_csv(index=False).encode(), file_name=f"{ch.label}_preprocessed.csv", mime="text/csv")


def page_simulate():
    _hero("Model Simulation", "Forward-simulate the 3-state OCR-informed model using the active protocol.")
    ch = st.session_state.get("chamber")
    if ch is None:
        st.info("Preprocess a dataset first.")
        return
    proto = _proto_from_chamber(ch)
    params = {}
    cols = st.columns(4)
    for i, k in enumerate(CORE_PARAM_ORDER):
        with cols[i % 4]:
            lo, hi = PARAM_BOUNDS.get(k, (1e-6, 1e6))
            params[k] = st.number_input(display_param_verbose(k), value=float(DEFAULT_PARAMS[k]), min_value=float(lo), max_value=float(hi), format="%.4g", key=f"desktop_sim_{k}", help=display_param_help(k))
    n_fccp = len(proto.t_fccp)
    params["alphas"] = [st.number_input(display_param_verbose(f"alpha_{j+1}"), value=1.0, min_value=0.0, max_value=10.0, step=0.1, key=f"desktop_alpha_{j}") for j in range(n_fccp)]
    if st.button("Run simulation", type="primary"):
        try:
            res = simulate(params, proto, o2_init=float(ch.o[0]))
            st.session_state.current_params = params
            render_plot(st, _simulation_plot(res, proto))
        except Exception as e:
            st.exception(e)


def page_calibration():
    _hero("Calibration", "Fit the 3-state model to the active O₂ trace with deterministic optimization.")
    ch = st.session_state.get("chamber")
    if ch is None:
        st.info("Preprocess a dataset first.")
        return
    c1, c2, c3 = st.columns(3)
    with c1: maxiter = st.number_input(
        "Global optimizer iterations",
        min_value=2, max_value=500, value=int(run_cfg.de_maxiter),
        key=f"calib_maxiter_{run_cfg.tier}",
        help=f"Tier default for '{run_cfg.tier}' is {run_cfg.de_maxiter} "
             f"(from core/run_settings.py). Increase for tighter fits.")
    with c2: popsize = st.number_input(
        "Global optimizer population size",
        min_value=2, max_value=30, value=int(run_cfg.de_popsize),
        key=f"calib_popsize_{run_cfg.tier}",
        help=f"Tier default for '{run_cfg.tier}' is {run_cfg.de_popsize}. "
             f"Larger populations explore more broadly but take longer.")
    with c3: seed = st.number_input(
        "Random seed", min_value=0, max_value=999999, value=123,
        key="calib_seed")
    polish = st.checkbox(
        "Refine with local polishing", value=bool(run_cfg.de_polish),
        key=f"calib_polish_{run_cfg.tier}",
        help="Local polishing (L-BFGS-B) tightens the final fit.")
    if not run_cfg.reportable:
        st.warning(
            f"You are running tier '{run_cfg.tier}', which is NOT reportable. "
            f"Switch to 'publication' in the sidebar for tier-defensible "
            f"numbers before quoting results in a manuscript.")
    st.info("If a local section fits poorly, first inspect event labels and preprocessing. Then run a longer calibration, compare residuals, and use identifiability before treating parameter changes as biological endpoints.")
    if st.button("Run calibration", type="primary"):
        try:
            with st.spinner("Calibrating..."):
                proto = _proto_from_chamber(ch)
                res = calibrate_de(ch.t, ch.o, proto, maxiter=int(maxiter), popsize=int(popsize), seed=int(seed), polish=bool(polish))
            st.session_state.calib_result = asdict(res)
            st.success(f"Calibration complete. RMSE = {(res.rmse_calib if res.rmse_calib is not None else res.rmse_full_trace):.4g} nmol/mL")
        except Exception as e:
            st.exception(e)
    if st.session_state.calib_result:
        calib = st.session_state.calib_result
        params = _params_from_calib(calib)
        proto = _proto_from_chamber(ch)
        sim = simulate(params, proto, o2_init=float(ch.o[0]))
        render_plot(st, _trace_fig(ch.t, ch.o, title="Observed vs fitted O₂", events={"oligo":ch.t_oligo,"fccp":list(ch.t_fccp),"inhib":ch.t_inhibit}, extra_lines=[(sim.t, sim.o, "Fitted model")]))
        rows=[]
        for k in CORE_PARAM_ORDER:
            if k in params: rows.append({"Parameter": display_param_verbose(k), "Symbol": display_param(k), "Value": params[k]})
        for j,a in enumerate(params.get("alphas", []), 1): rows.append({"Parameter": display_param_verbose(f"alpha_{j}"), "Symbol": display_param(f"alpha_{j}"), "Value": a})
        rows.append({"Parameter": display_param_verbose("sigma_obs"), "Symbol": display_param("sigma_obs"), "Value": calib.get("sigma_obs")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        try:
            ps = compute_phase_summary(ch, params)
            st.markdown("### Phase-level OCR summary")
            st.dataframe(pd.DataFrame(ps.get("phases", [])), hide_index=True, use_container_width=True)
        except Exception:
            pass


def page_diagnostics():
    _hero("Numerical Diagnostics", "Audit solver status, state ranges, monotonic oxygen behavior, and numerical warnings.")
    ch = st.session_state.get("chamber")
    calib = st.session_state.get("calib_result")
    if ch is None or calib is None:
        st.info("Run preprocessing and calibration first.")
        return
    if st.button("Run numerical audit", type="primary"):
        try:
            rep = detect_instability(_params_from_calib(calib), _proto_from_chamber(ch), o2_init=float(ch.o[0]))
            st.session_state.stability = rep
            st.success("Numerical audit complete.")
        except Exception as e:
            st.exception(e)
    rep = st.session_state.get("stability")
    if rep is not None:
        try: d = asdict(rep)
        except Exception: d = {"summary": str(rep)}
        items = [("Solver success", "Yes" if d.get("converged") else "No", "integration"), ("Finite states", "Yes" if int(d.get("nan_count", 0)) == 0 and int(d.get("negative_state_count", 0)) == 0 else "No", "state audit"), ("O₂ non-increasing", "Yes" if d.get("oxygen_monotone") else "No", "mass-balance check"), ("κ finite/range", "Yes" if d.get("kappa_in_range") else "No", "latent-drive check")]
        _metric_cards(items)
        warnings = d.get("warnings", []) or []
        if warnings:
            st.warning("\n".join(f"- {w}" for w in warnings))
        else:
            st.success("No numerical-stability warning was reported for the tested settings.")
        with st.expander("Machine-readable diagnostics", expanded=False):
            st.json(d)


def page_identifiability():
    _hero("Identifiability",
          "Assess whether fitted parameters are interpretable from OCR-only data.")
    ch = st.session_state.get("chamber"); calib = st.session_state.get("calib_result")
    if ch is None or calib is None:
        st.info("Run preprocessing and calibration first.")
        return
    params = _params_from_calib(calib); proto = _proto_from_chamber(ch)

    # ── FIM ──────────────────────────────────────────────────────────────
    if st.button("Run FIM diagnostic", type="primary"):
        try:
            rep = fisher_information(params, ch.t, ch.o, proto,
                                     o2_init=float(ch.o[0]),
                                     sigma_obs=params.get("sigma_obs"))
            st.session_state.fim_report = rep
            st.success(f"FIM complete. "
                       f"Raw condition number ≈ {rep.condition_raw:.2e} "
                       f"(clipped: {rep.condition_clipped:.2e})")
        except Exception as e: st.exception(e)
    rep = st.session_state.get("fim_report")
    if rep is not None:
        render_plot(st, _fim_eigen_plot(rep))
        # The FIM is a LOCAL diagnostic; without profile likelihoods we
        # cannot assign per-parameter identifiability verdicts. Be honest
        # about that here rather than printing "weak" against every row.
        st.caption(
            "FIM is a local diagnostic. Per-parameter verdicts require a "
            "profile-likelihood run below. The FIM table alone cannot "
            "distinguish identifiable / one-sided / non-identifiable.")
        st.dataframe(pd.DataFrame({
            "Parameter": [display_param_verbose(p) for p in rep.param_names],
            "Symbol":    [display_param(p) for p in rep.param_names],
            "Status":    ["FIM-only; run profile likelihood for verdict"]
                          * len(rep.param_names),
        }), hide_index=True, use_container_width=True)

    # ── Profile likelihood / fixed-parameter scan ───────────────────────
    if run_cfg.profile_real:
        st.markdown("### Genuine profile likelihood (inner re-optimisation)")
        st.caption(
            f"Tier '{run_cfg.tier}': all other parameters re-optimised at "
            f"every grid point (n_grid={run_cfg.profile_n_grid}, "
            f"maxiter={run_cfg.profile_maxiter}, "
            f"n_restarts_constrained={run_cfg.profile_n_restarts_constrained}). "
            f"This is the publication-grade analysis the manuscript Table 3 "
            f"is built from.")
    else:
        st.markdown("### Fixed-other-parameter scan (NOT a profile likelihood)")
        st.warning(
            f"Tier '{run_cfg.tier}' runs a fixed-other-parameter scan — all "
            f"other parameters are held at the MAP rather than re-optimised. "
            f"On a sloppy likelihood surface (FIM condition "
            f"{getattr(rep, 'condition_clipped', float('nan')):.1e} for "
            f"this dataset) this produces CIs that are too narrow by "
            f"construction. Switch the sidebar tier to 'publication' for "
            f"genuine profile likelihoods before quoting verdicts.")

    pnames = list(CORE_PARAM_ORDER)
    psel = st.selectbox("Parameter to profile", pnames,
                        format_func=display_param_verbose)
    btn_label = ("Run profile likelihood (genuine)" if run_cfg.profile_real
                 else "Run fixed-parameter scan (diagnostic only)")
    if st.button(btn_label):
        try:
            if run_cfg.profile_real:
                pr = profile_likelihood(
                    psel, params, ch.t, ch.o, proto,
                    o2_init=float(ch.o[0]),
                    n_grid=run_cfg.profile_n_grid,
                    grid_span_log=run_cfg.profile_grid_span_log,
                    maxiter=run_cfg.profile_maxiter,
                    adaptive_extend=run_cfg.profile_adaptive_extend,
                    n_restarts_constrained=run_cfg.profile_n_restarts_constrained,
                    sigma_obs=params.get("sigma_obs"), verbose=False)
            else:
                pr = fixed_parameter_scan(
                    psel, params, ch.t, ch.o, proto,
                    o2_init=float(ch.o[0]),
                    n_grid=run_cfg.profile_n_grid,
                    sigma_obs=params.get("sigma_obs"))
            st.session_state.profile_reports[psel] = pr
            tag = "PROFILE" if run_cfg.profile_real else "FIXED-SCAN"
            st.success(f"{tag} complete for {display_param(psel)}: "
                       f"verdict = {pr.practical_id}")
        except Exception as e: st.exception(e)

    # ── Verdict table (every parameter that has been profiled) ──────────
    if st.session_state.profile_reports:
        rows = []
        for nm, pr in st.session_state.profile_reports.items():
            map_v = float(params.get(nm, float("nan")))
            lo = pr.ci_low if pr.ci_low is not None else float("-inf")
            hi = pr.ci_high if pr.ci_high is not None else float("+inf")
            in_ci = bool(lo <= map_v <= hi)
            rows.append({
                "Parameter": display_param_verbose(nm),
                "Verdict":   pr.practical_id,
                "MAP":       f"{map_v:.4g}",
                "CI low":    "—" if pr.ci_low is None else f"{pr.ci_low:.4g}",
                "CI high":   "—" if pr.ci_high is None else f"{pr.ci_high:.4g}",
                "MAP in CI": "✓" if in_ci else "✗",
                "Source":    "genuine profile" if run_cfg.profile_real
                              else "fixed scan (diagnostic)",
            })
        df = pd.DataFrame(rows)
        # Sort: profiled in this session, then by verdict, then by param.
        verdict_order = {"identifiable": 0, "one-sided": 1,
                         "weakly identified": 2,
                         "non-identifiable": 3, "unresolved": 4}
        df["__order"] = df["Verdict"].map(verdict_order).fillna(5)
        df = df.sort_values(["__order", "Parameter"]).drop(columns="__order")
        st.markdown("### Per-parameter verdicts (from this session's runs)")
        # MAP-outside-CI is a key non-identifiability signal that the
        # engine remediation surfaced — flag it prominently.
        n_outside = int((df["MAP in CI"] == "✗").sum())
        if n_outside > 0:
            st.warning(
                f"{n_outside} parameter(s) have MAP outside the profile "
                f"CI. On a sloppy OCR-only likelihood surface this is a "
                f"strong signal of practical non-identifiability: the "
                f"calibration optimum sits on a different point of the "
                f"flat valley than profile re-optimisation finds.")
        st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Plot for the currently-selected parameter ───────────────────────
    if psel in st.session_state.profile_reports:
        render_plot(st, _profile_plotly(
            st.session_state.profile_reports[psel], psel))


def page_sensitivity():
    _hero("Sensitivity", "Identify high-information parameters and protocol phases. Sensitivity does not prove identifiability.")
    ch = st.session_state.get("chamber"); calib = st.session_state.get("calib_result")
    if ch is None or calib is None:
        st.info("Run preprocessing and calibration first.")
        return
    params = _params_from_calib(calib); proto = _proto_from_chamber(ch)
    if not run_cfg.reportable:
        st.warning(
            f"Tier '{run_cfg.tier}' Sobol N_base={run_cfg.sobol_n_base} is "
            f"below the publication threshold (512). First-order indices may "
            f"be unstable or negative at this sample size. Numbers shown "
            f"here are not manuscript-reportable.")
    st.caption(
        f"Tier '{run_cfg.tier}': Morris N_traj={run_cfg.morris_trajectories}, "
        f"Sobol N_base={run_cfg.sobol_n_base}, time-resolved "
        f"N_base={run_cfg.time_resolved_sobol_n_base} × "
        f"{run_cfg.time_resolved_sobol_n_t} t-points.")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Run Morris screening", type="primary"):
            st.session_state.sens_morris = morris_screening(
                proto, o2_init=float(ch.o[0]),
                N_trajectories=run_cfg.morris_trajectories, seed=123)
    with c2:
        if st.button("Run Sobol AUC"):
            st.session_state.sens_sobol = sobol_indices(
                proto, o2_init=float(ch.o[0]),
                N=run_cfg.sobol_n_base, seed=123)
    with c3:
        if st.button("Run time-resolved Sobol"):
            st.session_state.sens_trs = time_resolved_sobol(
                proto, o2_init=float(ch.o[0]),
                N=run_cfg.time_resolved_sobol_n_base,
                n_t_eval=run_cfg.time_resolved_sobol_n_t, seed=123)
    if st.session_state.sens_morris: render_plot(st, _morris_plotly(st.session_state.sens_morris))
    if st.session_state.sens_sobol: render_plot(st, _sobol_auc_plotly(st.session_state.sens_sobol))
    if st.session_state.sens_trs: render_plot(st, _time_resolved_sobol_plotly(st.session_state.sens_trs, proto))


def page_validation():
    _hero("Validation", "Run diagnostics that separate technical transfer from biological generalization.")
    ch = st.session_state.get("chamber"); calib = st.session_state.get("calib_result")
    if ch is None or calib is None:
        st.info("Run preprocessing and calibration first.")
        return
    params = _params_from_calib(calib); proto = _proto_from_chamber(ch)
    st.caption(
        f"Tier '{run_cfg.tier}': parametric bootstrap n_boot="
        f"{run_cfg.bootstrap_n_boot}; within-trace holdout = "
        f"{'REFIT-BASED (DE maxiter='+str(run_cfg.within_trace_de_maxiter)+', popsize='+str(run_cfg.within_trace_de_popsize)+')' if run_cfg.within_trace_refit else 'residual split (no refit, diagnostic only)'}."
    )
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Run parametric-bootstrap predictive check", type="primary"):
            st.session_state.validation_ppc = parametric_bootstrap_predictive_check(
                ch.t, ch.o, proto, params,
                n_boot=run_cfg.bootstrap_n_boot, seed=123)
    with c2:
        if st.button("Run within-trace holdout"):
            # Always refit-based at publication tier. Fast/smoke get the
            # same code path but with a much smaller DE budget; refit
            # actually happens, just cheaply.
            st.session_state.validation_wt = within_trace_holdout(
                ch.t, ch.o, proto, train_frac=0.70,
                maxiter=run_cfg.within_trace_de_maxiter,
                popsize=run_cfg.within_trace_de_popsize,
                seed=123, polish=run_cfg.de_polish)
    if st.session_state.validation_ppc:
        ppc = st.session_state.validation_ppc
        # The result JSON uses several keys across versions; accept any.
        cov = (ppc.get("coverage_band_observed")
               or ppc.get("coverage90") or ppc.get("coverage_90")
               or float("nan"))
        st.metric("Empirical 90% coverage", f"{cov:.1%}")
        # Surface a band-warning consistent with the sidebar thresholds.
        try:
            cov_f = float(cov)
            if not (cov_lo <= cov_f <= cov_hi):
                st.warning(
                    f"Empirical coverage {cov_f:.1%} is outside the "
                    f"configurable warning band "
                    f"[{cov_lo:.0%}, {cov_hi:.0%}].")
        except (TypeError, ValueError):
            pass
        render_plot(st, _validation_ppc_plot(ppc, proto))
        st.caption(EXPLICIT_DISCLAIMER)
    if st.session_state.validation_wt:
        wt = st.session_state.validation_wt
        # Show RMSE_train and RMSE_test honestly and explain the
        # legitimate test<train pattern for stress-test traces.
        c1, c2 = st.columns(2)
        with c1: st.metric("RMSE train (refit window)",
                            f"{float(wt.get('rmse_train', float('nan'))):.3f}")
        with c2: st.metric("RMSE test (held-out tail)",
                            f"{float(wt.get('rmse_test', float('nan'))):.3f}")
        rt = wt.get("rmse_train"); rs = wt.get("rmse_test")
        if rt is not None and rs is not None and rs < rt:
            st.info(
                "RMSE_test < RMSE_train is legitimate here: the held-out "
                "post-inhibition tail has near-zero OCR (rotenone + "
                "antimycin block the chain), so its observational variance "
                "is intrinsically low. This is intervention-phase "
                "extrapolation, not predictive-performance comparison.")
        render_plot(st, _validation_holdout_plot(wt, ch, proto))


def page_report_builder():
    _hero("Report Builder", "Build a customized human-readable or machine-readable report from analyses run in this session.")
    rep_obj = _current_structured_report()
    rep_obj["report_provenance"] = {"source":"current Streamlit session state","loaded_dataset":st.session_state.loaded_path,"loaded_chamber":st.session_state.loaded_label,"note":"The report builder does not invent results; empty/not-run sections are hidden by default."}
    all_sections = [k for k,v in rep_obj.items() if v not in (None, {}, [])]
    default_sections = [s for s in ["analysis_status","data","phase_summary","calibration","stability","identifiability","sensitivity","validation","hypothesis_prioritization","experimental_design_guidance","warnings_by_category","report_provenance"] if s in all_sections]
    selected = st.multiselect("Report sections", options=all_sections, default=default_sections or all_sections, format_func=display_section)
    fmt = st.selectbox("Report format", ["HTML","PDF"], help="Report Builder creates polished human-readable reports. Use Export Results for JSON/YAML machine-readable bundles.")
    title = st.text_input("Report title", "MitoAgent session report")
    report_for_export = {k:rep_obj[k] for k in selected if k in rep_obj}
    if fmt == "HTML":
        html = report_to_html(report_for_export, title=title, sections=selected)
        st.download_button("Download HTML report", html.encode(), "mitoagent_report.html", "text/html")
        st.components.v1.html(html, height=480, scrolling=True)
    elif fmt == "PDF":
        pdf = report_to_pdf_bytes(report_for_export, title=title, sections=selected)
        st.download_button("Download PDF report", pdf, "mitoagent_report.pdf", "application/pdf")
    with st.expander("Structured source used to build this report", expanded=False): st.json(report_for_export)


def page_export_results():
    _hero("Export Results", "Download machine-readable session outputs for reproducible reruns and external analysis.")
    rep_obj = _current_structured_report()
    rep_obj["export_provenance"] = {"source": "current Streamlit session state", "loaded_dataset": st.session_state.loaded_path, "loaded_chamber": st.session_state.loaded_label}
    st.info("Use Report Builder for human-readable HTML/PDF reports. Use this page for compact JSON/YAML exports that preserve structured backend outputs.")
    available = [k for k, v in rep_obj.items() if v not in (None, {}, [])]
    selected = st.multiselect("Export sections", options=available, default=available, format_func=display_section)
    payload = {k: rep_obj[k] for k in selected if k in rep_obj}
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download JSON bundle", json.dumps(payload, indent=2, default=str).encode(), "mitoagent_results_bundle.json", "application/json", type="primary")
    with c2:
        st.download_button("Download YAML bundle", report_to_yaml(payload).encode(), "mitoagent_results_bundle.yaml", "text/yaml")
    with st.expander("Machine-readable preview", expanded=False):
        st.json(payload)


# ── Manuscript figure inventory (script -> output base name) ──────────
# This list mirrors the script set under figures/ and the canonical
# output names under figures/final/ and manuscript/final/. Edit here if
# the figure inventory changes.
_MANUSCRIPT_FIGS = [
    ("Figure 1 — Model overview",        "make_fig_step1.py",  "fig1_model_overview"),
    ("Figure 2 — 3-state model",         "make_fig_step2.py",  "fig2_reduced_model"),
    ("Figure 3 — Data pipeline",         "make_fig_step3.py",  "fig3_data_pipeline"),
    ("Figure 4 — Calibration",           "make_fig_step4.py",  "fig4_calibration"),
    ("Figure 5 — Identifiability",       "make_fig_step5.py",  "fig5_identifiability"),
    ("Figure 6 — Sensitivity",           "make_fig_step6.py",  "fig6_sensitivity"),
    ("Figure 7 — Validation",            "make_fig_step7.py",  "fig7_validation"),
    ("Figure 8 — Agent architecture",    "make_fig_step8.py",  "fig8_agent_architecture"),
    ("Figure 9 — Streamlit UI overview", "make_fig_step9.py",  "fig9_streamlit_ui_overview"),
    ("Figure 10 — Ask MitoAgent",       "make_fig_step10.py", "fig10_ask_mitoagent"),
]


def _figure_input_status() -> dict:
    """For each manuscript figure, return which result files exist on disk.

    Returned dict maps script-name -> {available: bool, expected_inputs: [..],
    missing_inputs: [..], tier: 'publication' | 'fast' | 'smoke' | '?' | None}.
    Tier is taken from the calibration JSON when relevant.
    """
    root = Path(__file__).resolve().parent.parent
    res = root / "results"
    # Map each figure to its expected inputs (None = no data inputs, schematic).
    expected = {
        "make_fig_step1.py":  [],  # schematic
        "make_fig_step2.py":  [],  # schematic
        "make_fig_step3.py":  [
            res / "calibration" / "calib_dataset_I.json",
        ],
        "make_fig_step4.py":  [
            res / "calibration" / "calib_dataset_I.json",
            res / "calibration" / "calib_dataset_II.json",
            res / "calibration" / "calib_dataset_III.json",
        ],
        "make_fig_step5.py":  [
            res / "identifiability" / "fim_dataset_I.json",
            res / "identifiability" / "profiles_dataset_I.json",
        ],
        "make_fig_step6.py":  [
            res / "sensitivity" / "morris_dataset_I.json",
            res / "sensitivity" / "sobol_auc_dataset_I.json",
            res / "sensitivity" / "time_resolved_sobol_dataset_I.npz",
        ],
        "make_fig_step7.py":  [
            res / "validation" / "parametric_bootstrap_predictive_check_dataset_I.json",
            res / "validation" / "within_trace_holdout_dataset_I.json",
        ],
        "make_fig_step8.py":  [],  # schematic
        "make_fig_step9.py":  [],  # schematic
        "make_fig_step10.py": [],  # schematic
    }
    out = {}
    # Try to read the calibration tier (a useful hint about what figures
    # 3-7 will be drawn from).
    tier = None
    try:
        cal = json.loads((res / "calibration" / "calib_dataset_I.json").read_text())
        tier = cal.get("diagnostic_level")
    except Exception:
        pass
    for script, inputs in expected.items():
        missing = [str(p.relative_to(root)) for p in inputs if not p.exists()]
        out[script] = {
            "available": not missing,
            "expected_inputs": [str(p.relative_to(root)) for p in inputs],
            "missing_inputs": missing,
            "tier": tier if inputs else None,
        }
    return out


def _run_figure_script(script_name: str, timeout_s: int = 120) -> tuple[bool, str]:
    """Run one figure script as a subprocess, capturing stdout+stderr."""
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(root / "figures" / script_name)],
        cwd=str(root), capture_output=True, text=True, timeout=timeout_s)
    ok = (proc.returncode == 0)
    log = (proc.stdout or "") + (proc.stderr or "")
    return ok, log


def page_manuscript_figures():
    _hero("Manuscript Figures",
          "Regenerate the publication figures (Figs 1–10) from the current results/ tree, then preview them inline.")
    st.markdown(
        "Each figure script reads its inputs from `results/` and writes "
        "the output to both `figures/final/` (the canonical figure tree) "
        "and `manuscript/final/` (mirror used by the LaTeX manuscript). "
        "Figures 4–7 depend on result JSONs produced by the Calibration, "
        "Identifiability, Sensitivity, and Validation pages or by "
        "`python run_all.py --publication` on the command line. "
        "Schematic figures (1, 2, 3, 8, 9, 10) have no data dependencies.")
    if not run_cfg.reportable:
        st.warning(
            f"Sidebar tier is '{run_cfg.tier}' (NOT reportable). The figure "
            f"scripts will use whatever is currently on disk in `results/`. "
            f"If those results came from a non-publication run, the figures "
            f"will inherit that tier. Switch to 'publication' and run the "
            f"underlying analyses (or run `python run_all.py --publication` "
            f"in a terminal) before regenerating figures for a manuscript.")

    status = _figure_input_status()
    root = Path(__file__).resolve().parent.parent
    figs_final = root / "figures" / "final"

    st.markdown("### Figure inventory")
    rows = []
    for label, script, base in _MANUSCRIPT_FIGS:
        s = status[script]
        existing_png = figs_final / f"{base}.png"
        existing_pdf = figs_final / f"{base}.pdf"
        rows.append({
            "Figure": label,
            "Script": script,
            "Inputs ready": "—" if not s["expected_inputs"]
                                 else ("✓" if s["available"] else "✗"),
            "Tier on disk":  s["tier"] or "—",
            "PNG on disk":   "✓" if existing_png.exists() else "—",
            "PDF on disk":   "✓" if existing_pdf.exists() else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)

    # ── Bulk-regenerate buttons ─────────────────────────────────────
    st.markdown("### Regenerate")
    sel_labels = [r["Figure"] for r in rows]
    # Pre-select figures 4-6 because they are the most-asked-about set
    # (calibration / identifiability / sensitivity).
    default_sel = [r["Figure"] for r in rows
                   if "Figure 4" in r["Figure"]
                   or "Figure 5" in r["Figure"]
                   or "Figure 6" in r["Figure"]]
    chosen = st.multiselect("Select figures to regenerate", sel_labels,
                            default=default_sel,
                            help="Tip: figures 4-6 are the calibration, "
                                  "identifiability, and sensitivity figures "
                                  "shown in the manuscript.")
    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("Regenerate selected figures", type="primary",
                            disabled=not chosen)
    with c2:
        run_all_btn = st.button("Regenerate all 10 figures")

    if run_btn or run_all_btn:
        targets = ([(label, script, base) for label, script, base
                    in _MANUSCRIPT_FIGS if label in chosen]
                   if run_btn else _MANUSCRIPT_FIGS)
        with st.spinner(f"Regenerating {len(targets)} figure(s)..."):
            results = []
            for label, script, base in targets:
                # Block obviously-missing data figures from running and
                # explain why; let schematics through unconditionally.
                s = status[script]
                if s["expected_inputs"] and not s["available"]:
                    results.append((label, False,
                                    f"missing inputs: "
                                    f"{', '.join(s['missing_inputs'])}"))
                    continue
                try:
                    ok, log = _run_figure_script(script)
                    tail = log.strip().splitlines()
                    last = tail[-1] if tail else ""
                    results.append((label, ok, last or
                                     ("regenerated" if ok else "failed")))
                except subprocess.TimeoutExpired:
                    results.append((label, False, "timeout (>120s)"))
                except Exception as e:
                    results.append((label, False, str(e)))
        # Summary
        n_ok = sum(1 for _, ok, _ in results if ok)
        if n_ok == len(results):
            st.success(f"Regenerated {n_ok}/{len(results)} figure(s).")
        else:
            st.error(f"Regenerated {n_ok}/{len(results)} figure(s); "
                     f"{len(results) - n_ok} failed.")
        st.dataframe(pd.DataFrame([
            {"Figure": l, "Status": "OK" if ok else "FAILED", "Detail": msg}
            for l, ok, msg in results
        ]), hide_index=True, use_container_width=True)

    # ── Inline preview of the figures currently on disk ────────────
    st.markdown("### Preview")
    preview_choice = st.selectbox(
        "Preview figure", sel_labels,
        index=(sel_labels.index("Figure 4 — Calibration")
               if "Figure 4 — Calibration" in sel_labels else 0))
    for label, script, base in _MANUSCRIPT_FIGS:
        if label != preview_choice:
            continue
        png_path = figs_final / f"{base}.png"
        pdf_path = figs_final / f"{base}.pdf"
        if png_path.exists():
            st.image(str(png_path), use_column_width=True, caption=label)
        else:
            st.info(f"No PNG yet at `{png_path.relative_to(root)}`. "
                    f"Regenerate it above.")
        c1, c2 = st.columns(2)
        if png_path.exists():
            with c1:
                st.download_button(f"Download {base}.png",
                                    png_path.read_bytes(),
                                    file_name=f"{base}.png",
                                    mime="image/png")
        if pdf_path.exists():
            with c2:
                st.download_button(f"Download {base}.pdf",
                                    pdf_path.read_bytes(),
                                    file_name=f"{base}.pdf",
                                    mime="application/pdf")
        break

    # ── Equivalent CLI commands, for reproducibility ────────────────
    st.markdown("### Equivalent CLI commands")
    st.code(
        "# Regenerate all manuscript figures from current results/:\n"
        "python run_all.py --figures-only\n"
        "\n"
        "# Or run individual figure scripts:\n"
        "python figures/make_fig_step4.py   # Figure 4 — Calibration\n"
        "python figures/make_fig_step5.py   # Figure 5 — Identifiability\n"
        "python figures/make_fig_step6.py   # Figure 6 — Sensitivity\n"
        "\n"
        "# Full publication-tier pipeline (analyses + figures, ~30-60 min):\n"
        "python run_all.py --publication\n",
        language="bash")



def page_optional_agent():
    _hero("Optional NL Agent / LLM Settings", "Configure optional natural-language explanation. All numerical results remain deterministic backend outputs.")
    has_llm = False
    try:
        from agent.llm_driver import LLM_AVAILABLE
        has_llm = bool(LLM_AVAILABLE)
    except Exception:
        has_llm = False
    st.markdown(f"<span class='mito-badge {'badge-pass' if has_llm else 'badge-yellow'}'>{'LLM provider configured' if has_llm else 'No LLM provider configured'}</span>", unsafe_allow_html=True)
    st.info("LLM-assisted mode is optional. It can explain structured backend outputs, route questions, and summarize caveats. It cannot estimate parameters, create results, diagnose disease, or override diagnostics.")
    st.caption(
        f"Pipeline run tier follows the sidebar selector: currently "
        f"'{run_cfg.tier}' ({'reportable' if run_cfg.reportable else 'NOT reportable'}). "
        f"Switch the sidebar to 'publication' before generating any report "
        f"intended for a manuscript.")
    if st.session_state.loaded_path and st.button("Run deterministic full pipeline", type="primary"):
        try:
            a = MitoAgent(verbose=False)
            # MitoAgent.run_pipeline uses a fast: bool flag. The
            # publication tier (reportable=True) maps to fast=False;
            # smoke and fast both map to fast=True (publication budgets
            # are only honoured when reportable).
            rep = a.run_pipeline(
                st.session_state.loaded_path,
                fast=not run_cfg.reportable,
                coverage_band=(float(cov_lo), float(cov_hi)),
                fim_sloppy_threshold=float(fim_thr))
            st.success(f"Pipeline complete: mode={rep.get('mode')} (tier={run_cfg.tier})")
            st.dataframe(pd.DataFrame([
                {"Item": "Tier",              "Value": run_cfg.tier},
                {"Item": "Reportable",        "Value": "yes" if run_cfg.reportable else "no"},
                {"Item": "Warnings",          "Value": str(rep.get('warning_counts', {}))},
                {"Item": "Skipped analyses",  "Value": ', '.join(rep.get('skipped_analyses', [])) or 'none'},
            ]), hide_index=True, use_container_width=True)
        except Exception as e: st.exception(e)


def render_page():
    _desktop_sidebar()
    page = st.session_state.get("nav_page", "Dashboard")
    dispatch = {
        "Dashboard": page_dashboard,
        "Load Data": page_load_data,
        "Event Parsing & Preprocessing": page_preprocess,
        "Model Simulation": page_simulate,
        "Calibration": page_calibration,
        "Numerical Diagnostics": page_diagnostics,
        "Identifiability": page_identifiability,
        "Sensitivity": page_sensitivity,
        "Validation": page_validation,
        "Hypothesis Prioritization": lambda: (render_status_card(st, _current_structured_report()), render_hypothesis_tab(st, _current_structured_report())),
        "Experimental Design Guidance": lambda: (render_status_card(st, _current_structured_report()), render_design_guidance_tab(st, _current_structured_report())),
        "Ask MitoAgent": lambda: (render_status_card(st, _current_structured_report()), render_ask_mitoagent_tab(st, _current_structured_report())),
        "Report Builder": page_report_builder,
        "Manuscript Figures": page_manuscript_figures,
        "Export Results": page_export_results,
        "Help / Runbook": lambda: render_help_tab(st),
        "FAQ": lambda: render_faq_tab(st),
        "Optional NL Agent / LLM Settings": page_optional_agent,
    }
    dispatch.get(page, page_dashboard)()


render_page()
