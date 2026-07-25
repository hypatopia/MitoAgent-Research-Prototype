"""UI helper utilities for MitoAgent Streamlit app.

These helpers keep visual formatting, parameter labels, and report export
logic out of the scientific backend.
"""
from __future__ import annotations
from html import escape
from io import BytesIO
import base64
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json
import math
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # Plotly is part of requirements-ui but keep import-safe smoke tests.
    go = None
    make_subplots = None

PARAM_SYMBOLS: Dict[str, str] = {
    # Symbols only. These are used in figures, tables, dropdowns, and compact
    # labels so raw code names such as k_supply or V_max never appear to users.
    "k_supply": "kₛ",
    "c_tot": "cₜ",
    "V_max": "Vₘₐₓ",
    "K_o": "Kₒ",
    "K_r": "Kᵣ",
    "gamma_oligo": "γₒ",
    "tau_kappa": "τₖ",
    "r0": "r₀",
    "alpha": "α",
    "sigma_obs": "σₒᵦₛ",
    "n_sigma": "nσ",
}

PARAM_LABELS: Dict[str, str] = {
    # User-facing labels for form controls/body text: expressive phrase + symbol.
    "k_supply": "Effective supply rate kₛ",
    "c_tot": "Total reduced-pool scale cₜ",
    "V_max": "Maximum OCR capacity Vₘₐₓ",
    "K_o": "Oxygen affinity scale Kₒ",
    "K_r": "Reduced-pool affinity scale Kᵣ",
    "gamma_oligo": "Oligomycin response factor γₒ",
    "tau_kappa": "Drive relaxation time τₖ",
    "r0": "Initial reduced-pool state r₀",
    "alpha": "FCCP response amplitude α",
    "sigma_obs": "Within-trace observational noise σₒᵦₛ",
    "n_sigma": "Outlier threshold nσ",
}

PARAM_HELP: Dict[str, str] = {
    "k_supply": "Effective upstream reductant-supply rate. It can correlate with CIV-mediated OCR-capacity parameters in OCR-only fits.",
    "c_tot": "Effective total cytochrome-c/reductant pool scaling parameter in the model.",
    "V_max": "Effective maximum CIV-mediated OCR capacity. It is not an isolated Complex IV activity assay.",
    "K_o": "Oxygen half-saturation/scale parameter in the bounded OCR term.",
    "K_r": "Reduced-pool half-saturation/scale parameter in the bounded OCR term.",
    "gamma_oligo": "Oligomycin response factor in the latent effective respiratory-drive term.",
    "tau_kappa": "Relaxation time for the latent effective respiratory-drive/OCR-permissiveness factor κ.",
    "r0": "Initial effective reduced cytochrome-c/reductant pool state.",
    "alpha": "FCCP response amplitude. This is an FCCP-response indicator, not a direct proton-leak measurement.",
    "sigma_obs": "Post-hoc within-trace observational-noise estimate from residuals.",
}

EVENT_LABELS = {
    "start": "Start",
    "oligo": "Oligomycin",
    "fccp": "FCCP",
    "inhib": "Rot/Ant",
    "inhibit": "Rot/Ant",
    "end": "End",
}

DATA_BLUE = "#0072BD"  # MATLAB default blue: [0, 0.4470, 0.7410]
MODEL_ORANGE = "#D95319"
ACCENT_PURPLE = "#7E2F8E"
WARN_ORANGE = "#EDB120"

EVENT_COLORS = {
    "start": "#94a3b8",
    "oligo": "#EDB120",
    "fccp": "#0072BD",
    "inhib": "#A2142F",
    "inhibit": "#A2142F",
    "end": "#94a3b8",
}


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def display_param_symbol(name: str) -> str:
    """Return compact mathematical parameter notation with no underscores."""
    name = str(name)
    if name in PARAM_SYMBOLS:
        return PARAM_SYMBOLS[name]
    if name.startswith("alpha_"):
        suffix = name.split("_", 1)[1].translate(_SUBSCRIPT_DIGITS)
        return "α" + suffix
    return name.replace("_", " ")

def display_param(name: str) -> str:
    """Backward-compatible alias: compact symbol only for plots/tables."""
    return display_param_symbol(name)

def display_param_verbose(name: str) -> str:
    """Return an expressive user-facing parameter label plus book-style symbol."""
    name = str(name)
    if name in PARAM_LABELS:
        return PARAM_LABELS[name]
    if name.startswith("alpha_"):
        suffix = name.split("_", 1)[1].translate(_SUBSCRIPT_DIGITS)
        return "FCCP response amplitude α" + suffix
    return name.replace("_", " ").title()


def display_param_help(name: str) -> str:
    base = "alpha" if name.startswith("alpha_") else name
    return PARAM_HELP.get(base, "Model parameter. Interpret only with the identifiability flag and OCR-only caveats.")


def label_params_in_df(df: pd.DataFrame, col: str = "parameter") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str).map(display_param)
    return out


def event_items(events: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not events:
        return []
    out: List[Dict[str, Any]] = []
    for key, value in events.items():
        if value is None:
            continue
        vals = value if isinstance(value, list) else [value]
        for idx, ti in enumerate(vals, start=1):
            label = EVENT_LABELS.get(key, key)
            if key == "fccp":
                label = f"FCCP {idx}"
            try:
                out.append({"time": float(ti), "label": label, "key": key})
            except Exception:
                continue
    return sorted(out, key=lambda d: d["time"])


def add_event_markers_ax(ax, events: Mapping[str, Any] | None, *, y_text: float = 0.98) -> None:
    """Add dashed event lines and vertical labels to a matplotlib axis."""
    for item in event_items(events):
        color = EVENT_COLORS.get(item["key"], "#6c757d")
        ax.axvline(item["time"], color=color, linestyle="--", linewidth=0.9, alpha=0.85)
        ax.text(
            item["time"], y_text, item["label"], rotation=90,
            va="top", ha="right", fontsize=7, color=color,
            transform=ax.get_xaxis_transform(),
        )



def _dark_plot_layout(fig, *, title: str = "", x_title: str = "Time [s]", y_title: str = ""):
    """Apply the standard UI plot style.

    The app shell remains dark, but the figures themselves use a white plotting
    canvas so screenshots/exports are publication- and report-friendly.
    """
    if go is None:
        return fig
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0f172a", size=16)),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=12),
        hovermode="x unified",
        margin=dict(l=52, r=24, t=54, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,0.85)", font=dict(color="#0f172a")),
    )
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                     linecolor="#64748b", tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"))
    if y_title:
        fig.update_yaxes(title=y_title, showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                         linecolor="#64748b", tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a"))
    return fig


def add_event_markers_plotly(fig, events: Mapping[str, Any] | None, *, row=None, col=None) -> None:
    """Add dashed event lines and plain vertical text labels to a Plotly figure.

    Labels are intentionally plain text (no background box) and are placed inside
    the plotting area so they remain visible with Plotly's toolbar enabled.
    """
    if go is None:
        return
    for item in event_items(events):
        color = EVENT_COLORS.get(item["key"], "#94a3b8")
        if row is not None and col is not None:
            try:
                # For the one-row subplot layouts used in the UI, Plotly x-axis
                # ids are x, x2, x3, ... by column.
                suffix = "" if col == 1 else str(col)
                fig.add_vline(x=item["time"], line=dict(color=color, width=1.2, dash="dash"), row=row, col=col)
                fig.add_annotation(x=item["time"], y=0.96, xref=f"x{suffix}", yref="paper",
                                   text=item["label"], showarrow=False, textangle=-90,
                                   xanchor="left", yanchor="top", font=dict(color=color, size=12))
                continue
            except Exception:
                pass
        fig.add_shape(type="line", x0=item["time"], x1=item["time"], y0=0, y1=1,
                      xref="x", yref="paper", line=dict(color=color, width=1.2, dash="dash"))
        fig.add_annotation(x=item["time"], y=0.96, xref="x", yref="paper", text=item["label"],
                           showarrow=False, textangle=-90, xanchor="left", yanchor="top",
                           font=dict(color=color, size=12))


def trace_plotly(t: Sequence[float], o: Sequence[float], *, title: str = "", events: Mapping[str, Any] | None = None,
                 extra_lines: Sequence[tuple] | None = None, y_title: str = "O₂ [nmol/mL]"):
    """Interactive O2 trace plot with zoom/pan/export toolbar."""
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(t), y=list(o), mode="lines", name="O₂ data",
                             line=dict(color=DATA_BLUE, width=1.7)))
    if extra_lines:
        colors = [MODEL_ORANGE, ACCENT_PURPLE, WARN_ORANGE, "#A2142F"]
        for idx, (tt, oo, lbl) in enumerate(extra_lines):
            fig.add_trace(go.Scatter(x=list(tt), y=list(oo), mode="lines", name=str(lbl),
                                     line=dict(color=colors[idx % len(colors)], width=2)))
    add_event_markers_plotly(fig, events)
    _dark_plot_layout(fig, title=title, y_title=y_title)
    return fig


def render_plot(st, fig, *, use_container_width: bool = True) -> None:
    """Render Plotly figures with toolbar; fall back safely to explicit Matplotlib figures.

    Streamlit deprecates ``st.pyplot()`` without a figure.  Some plot helper
    functions return ``None`` when optional plotting backends are unavailable,
    so this renderer now handles that case explicitly instead of passing
    ``None`` to ``st.pyplot``.
    """
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": "mitoagent_plot", "scale": 2},
    }
    if fig is None:
        st.error("Interactive plotting requires Plotly. Install the UI dependencies with `python -m pip install -r requirements-ui.txt` and restart Streamlit.")
        return
    if go is not None and hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=use_container_width, config=config)
        return
    # Static fallback only for non-interactive Matplotlib figures. All main UI
    # figures are created as native Plotly objects so zoom/pan/download tools
    # remain available when requirements-ui.txt is installed.
    st.pyplot(fig)

def inject_professional_css(st) -> None:
    st.markdown(
        """
<style>
:root {
  color-scheme: dark;
  --mito-ink:#f8fafc;
  --mito-muted:#cbd5e1;
  --mito-soft:#94a3b8;
  --mito-line:rgba(148,163,184,.28);
  --mito-accent:#107AB0;
  --mito-accent-soft:#0B5F8F;
  --mito-blue:#107AB0;
  --mito-bg:#020617;
  --mito-panel:#0f172a;
  --mito-panel2:#111827;
  --mito-card:#0b1220;
  --mito-card2:#111c2e;
  --mito-sidebar-bg:#000000;
  --mito-sidebar-bg2:#020617;
  --mito-sidebar-text:#ffffff;
  --mito-sidebar-muted:#cbd5e1;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(circle at top left, #111827 0, #06121f 36%, #020617 78%) !important;
  color: var(--mito-ink) !important;
}
.block-container {padding-top: 1.25rem; max-width: 1580px;}
/* Global text visibility in dark mode */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
label, label p, .stText, .stCaptionContainer, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *, div, p, li, span {
  color: var(--mito-ink);
}
h1, h2, h3, h4, h5, h6, strong, b {color:#ffffff !important;}
a {color:#8bd3ff !important;}
hr {border-color:rgba(148,163,184,.25) !important;}
/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--mito-sidebar-bg) 0%, var(--mito-sidebar-bg2) 58%, #020617 100%) !important;
  border-right:1px solid rgba(148,163,184,.28) !important;
}
[data-testid="stSidebar"] * {color: var(--mito-sidebar-text) !important;}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: var(--mito-sidebar-muted) !important;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] b {color: var(--mito-sidebar-text) !important;}
[data-testid="stSidebar"] button, [data-testid="stSidebar"] button * {color: white !important;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] select {
  color:#f8fafc !important; background:#111827 !important; border-color:rgba(148,163,184,.35) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#111827 !important; border-color:rgba(148,163,184,.35) !important;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#f8fafc !important;}
[data-testid="stSidebar"] [role="radiogroup"] {
  background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10); border-radius:14px; padding:.35rem;
}

/* Desktop navigation buttons: left aligned and visually organized */
[data-testid="stSidebar"] .stButton > button {
  justify-content:flex-start !important;
  text-align:left !important;
  width:100% !important;
  padding-left:.85rem !important;
  border-radius:12px !important;
}
[data-testid="stSidebar"] .stButton > button p {
  text-align:left !important;
  width:100% !important;
}
.nav-group-title {
  text-align:left !important;
  letter-spacing:.08em;
  font-size:.72rem;
  font-weight:800;
  color:#93c5fd !important;
  margin:1.0rem 0 .35rem 0;
}

/* Main panels */
.mito-hero {padding: 1.35rem 1.45rem; border-radius: 24px; background: linear-gradient(135deg,#111827 0%,#107AB0 48%,#0B5F8F 100%); color: white; box-shadow: 0 18px 44px rgba(0,0,0,0.38); margin-bottom: 1.05rem; border:1px solid rgba(255,255,255,.22);} 
.mito-hero h1 {margin: 0 0 .4rem 0; font-size: 2.08rem; letter-spacing:-.02em; color:white !important;}
.mito-hero p {margin: 0; opacity: .96; font-size: 1.05rem; color:white !important;}
.mito-card {padding: 1rem 1.05rem; border: 1px solid var(--mito-line); border-radius: 18px; background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(11,18,32,.98)); box-shadow: 0 10px 30px rgba(0,0,0,.28); margin-bottom: .9rem; color:var(--mito-ink);} 
.mito-card h4, .mito-card h3, .mito-card h2 {margin-top: 0; color: #ffffff !important; letter-spacing:-.01em;}
.mito-card p, .mito-card li, .mito-card span {color: var(--mito-muted) !important;}
.mito-section-title {font-size:1.05rem; font-weight:800; color:#ffffff !important; margin:.4rem 0 .35rem 0;}
.small-muted {color:#94a3b8 !important; font-size:.9rem;}
.tooltip-note {border-left: 4px solid var(--mito-accent); padding: .7rem .85rem; background: rgba(16,122,176,.14); border-radius: 12px; margin: .55rem 0; color:#dbeafe !important;}
.tooltip-note * {color:#dbeafe !important;}
.mito-badge {display: inline-block; padding: .22rem .62rem; border-radius: 999px; font-size: .78rem; font-weight: 800; margin: .12rem .2rem .12rem 0; letter-spacing:.01em;}
.badge-pass {background: rgba(59,130,246,.18); color: #bfdbfe !important; border:1px solid rgba(59,130,246,.35)}
.badge-yellow {background: rgba(245,158,11,.18); color: #fcd34d !important; border:1px solid rgba(245,158,11,.35)}
.badge-red {background: rgba(239,68,68,.18); color: #fca5a5 !important; border:1px solid rgba(239,68,68,.35)}
.badge-blue {background: rgba(59,130,246,.18); color: #8bd3ff !important; border:1px solid rgba(59,130,246,.35)}
.badge-slate {background: rgba(148,163,184,.16); color: #cbd5e1 !important; border:1px solid rgba(148,163,184,.30)}

.mito-breadcrumb{margin:.55rem 0 .85rem .15rem;padding-left:.15rem;border-left:2px solid rgba(148,163,184,.35)}
.mito-breadcrumb .crumb{position:relative;display:flex;gap:.65rem;align-items:flex-start;margin:.55rem 0 .65rem -0.82rem;color:#cbd5e1}
.mito-breadcrumb .dot{width:1.45rem;height:1.45rem;border-radius:999px;display:inline-flex;align-items:center;justify-content:center;font-size:.78rem;font-weight:800;border:1px solid rgba(148,163,184,.45);background:#111827;color:#cbd5e1}
.mito-breadcrumb .done .dot{background:#0B5F8F;color:#E0F2FE;border-color:#107AB0}
.mito-breadcrumb .pending .dot{background:#020617;color:#94a3b8}
.mito-breadcrumb b{display:block;color:#f8fafc!important;line-height:1.05}
.mito-breadcrumb small{display:block;color:#94a3b8!important;font-size:.76rem;margin-top:.12rem}


/* Desktop scientific app shell */
.desktop-brand{display:flex;gap:.75rem;align-items:center;padding:.8rem .75rem 1rem .75rem;margin-bottom:.4rem;border-bottom:1px solid rgba(148,163,184,.25)}
.desktop-brand .brand-icon{width:2.35rem;height:2.35rem;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#107AB0,#1d4ed8);box-shadow:0 8px 22px rgba(16,122,176,.28);font-size:1.25rem}
.desktop-brand h2{margin:0!important;font-size:1.22rem!important;letter-spacing:-.02em}
.desktop-brand span{display:block;color:#cbd5e1!important;font-size:.78rem;margin-top:.12rem}
.nav-group-title{font-size:.72rem;font-weight:900;letter-spacing:.08em;color:#94a3b8!important;margin:1.05rem .25rem .4rem;text-transform:uppercase}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start!important;text-align:left!important;background:rgba(15,23,42,.70)!important;border:1px solid rgba(148,163,184,.22)!important;box-shadow:none!important;margin:.08rem 0!important;padding:.55rem .65rem!important;border-radius:12px!important;color:#f8fafc!important}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(16,122,176,.22)!important;border-color:#107AB0!important}
.workflow-rail-title{font-size:.78rem;font-weight:900;color:#f8fafc!important;margin:1.25rem .25rem .5rem;letter-spacing:.04em}
.desktop-rail{border-left:2px solid rgba(148,163,184,.28);margin-left:.72rem;padding-left:.85rem;margin-bottom:.75rem}
.rail-item{position:relative;display:flex;gap:.62rem;align-items:flex-start;margin:.52rem 0}
.rail-item span{position:absolute;left:-1.6rem;width:1.35rem;height:1.35rem;border-radius:99px;display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:900;border:1px solid rgba(148,163,184,.38);background:#020617;color:#94a3b8}
.rail-item.done span{background:#107AB0;color:white;border-color:#60a5fa}
.rail-item b{color:#f8fafc!important;font-size:.84rem}
.rail-item small{display:block;color:#94a3b8!important;font-size:.72rem;margin-top:.08rem}
.metric-card{background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(11,18,32,.98));border:1px solid rgba(148,163,184,.28);border-radius:18px;padding:1rem;min-height:7.4rem;box-shadow:0 10px 26px rgba(0,0,0,.24);margin-bottom:.75rem}
.metric-card span{display:block;color:#94a3b8!important;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;font-weight:800}
.metric-card b{display:block;color:#ffffff!important;font-size:1.45rem;line-height:1.25;margin:.35rem 0 .2rem}
.metric-card small{color:#cbd5e1!important;font-size:.82rem}
.compact-hero{padding:1.05rem 1.25rem!important;margin-bottom:.85rem!important}
.compact-hero h1{font-size:1.7rem!important}

/* Streamlit widgets and data containers */
div[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stExpander"] {
  background: rgba(15,23,42,.92) !important; border:1px solid var(--mito-line) !important; border-radius:16px !important; box-shadow:0 8px 24px rgba(0,0,0,.22) !important;
}
[data-testid="stMetric"] * {color:#f8fafc !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color:#f8fafc !important;}
[data-testid="stDataFrame"] * {color:inherit;}
.stTabs [data-baseweb="tab-list"] {gap:.35rem; flex-wrap:wrap; background:transparent;}
.stTabs [data-baseweb="tab"] {background:rgba(15,23,42,.92); border:1px solid rgba(148,163,184,.32); border-radius:14px 14px 0 0; padding:.62rem .9rem; color:#cbd5e1; min-height:2.65rem; min-width:6.1rem; font-weight:700;}
.stTabs [data-baseweb="tab"] p {color:#cbd5e1 !important;}
.stTabs [aria-selected="true"] {background:rgba(16,122,176,.22) !important; border-color:#107AB0 !important; color:#e0f2fe !important;}
.stTabs [aria-selected="true"] p {color:#dbeafe !important;}
button[kind="primary"], .stButton>button {border-radius:12px !important; border:1px solid rgba(16,122,176,.58) !important; background:linear-gradient(135deg,#107AB0,#0B5F8F) !important; color:white !important; font-weight:700 !important;}
.stDownloadButton>button {border-radius:12px !important; border:1px solid rgba(16,122,176,.58) !important; background:#0f172a !important; color:#f8fafc !important;}
input, textarea, select, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  background:#0f172a !important; color:#f8fafc !important; border-color:rgba(148,163,184,.35) !important;
}
[data-baseweb="select"] > div {background:#0f172a !important; border-color:rgba(148,163,184,.35) !important;}
[data-baseweb="select"] * {color:#f8fafc !important;}
/* Alerts keep readable contrast */
[data-testid="stAlert"] {background:rgba(15,23,42,.96) !important; border:1px solid rgba(148,163,184,.30) !important; color:#f8fafc !important;}
[data-testid="stAlert"] * {color:#f8fafc !important;}
/* Plot containers */
[data-testid="stImage"], [data-testid="stPlotlyChart"], [data-testid="stPyplot"] {
  background:#ffffff !important; border-radius:18px; padding:.45rem; border:1px solid rgba(148,163,184,.35);
}
.modebar, .modebar-container {opacity:1 !important; display:block !important; visibility:visible !important;}
.modebar-btn svg {fill:#0f172a !important;}
.modebar-btn:hover svg {fill:#107AB0 !important;}

/* Plotly SVG text readability on white plot backgrounds */
[data-testid="stPlotlyChart"] .xtick text,
[data-testid="stPlotlyChart"] .ytick text,
[data-testid="stPlotlyChart"] .gtitle,
[data-testid="stPlotlyChart"] .xtitle,
[data-testid="stPlotlyChart"] .ytitle {
  fill:#0f172a !important;
}

/* Code/JSON is intentionally dark but not visually dominant */
code, pre, [data-testid="stCodeBlock"] {background:#020617 !important; color:#dbeafe !important; border-color:rgba(148,163,184,.25) !important;}
</style>
        """,
        unsafe_allow_html=True,
    )


def humanize_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(humanize_value(x) for x in v)
    return str(v)



SECTION_LABELS: Dict[str, str] = {
    "analysis_status": "Analysis Status",
    "diagnostic_thresholds": "Diagnostic Thresholds",
    "diagnostic_level": "Diagnostic Level",
    "loaded_path": "Loaded Dataset Path",
    "loaded_label": "Loaded Chamber / Dataset Label",
    "preprocess_issues": "Preprocessing Issues",
    "data": "Data and Event Parsing",
    "phase_summary": "Phase-Level OCR Summary",
    "calibration": "Calibration Results",
    "stability": "Numerical Diagnostics",
    "identifiability": "Identifiability Analysis",
    "identifiability_profiles": "Profile-Likelihood / Scan Results",
    "sensitivity": "Sensitivity Analysis",
    "validation": "Validation Diagnostics",
    "hypothesis_prioritization": "Hypothesis Prioritization",
    "experimental_design_guidance": "Experimental-Design Guidance",
    "warnings_by_category": "Warnings by Category",
    "figure_inventory": "Figure Inventory",
}


def display_section(name: str) -> str:
    return SECTION_LABELS.get(str(name), str(name).replace("_", " ").title())


def _clean_key(k: Any) -> str:
    s = display_section(str(k)) if str(k) in SECTION_LABELS else str(k).replace("_", " ").title()
    return display_param_symbol(k) if str(k) in PARAM_SYMBOLS or str(k).startswith("alpha_") else s


def _json_ready(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_ready(x) for x in v]
    try:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
    except Exception:
        pass
    return v


def _html_table_from_mapping(data: Mapping[str, Any]) -> str:
    rows = []
    for k, v in data.items():
        if isinstance(v, Mapping):
            val = _html_table_from_mapping(v)
        elif isinstance(v, list):
            val = _html_list(v)
        else:
            val = escape(humanize_value(v))
        rows.append(f"<tr><th>{escape(_clean_key(k))}</th><td>{val}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _html_list(items: Sequence[Any]) -> str:
    if not items:
        return "<span class='muted'>None</span>"
    out = ["<ul>"]
    for item in items:
        if isinstance(item, Mapping):
            out.append("<li>" + _html_table_from_mapping(item) + "</li>")
        elif isinstance(item, list):
            out.append("<li>" + _html_list(item) + "</li>")
        else:
            out.append("<li>" + escape(humanize_value(item)) + "</li>")
    out.append("</ul>")
    return "".join(out)


def _html_section_value(val: Any) -> str:
    val = _json_ready(val)
    if isinstance(val, Mapping):
        return _html_table_from_mapping(val)
    if isinstance(val, list):
        return _html_list(val)
    return f"<p>{escape(humanize_value(val))}</p>"


def report_to_html(report: Mapping[str, Any], *, title: str = "MitoAgent report", sections: Sequence[str] | None = None, figure_paths: Sequence[str] | None = None) -> str:
    sections = list(sections or report.keys())
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        """
<style>
body{font-family:Arial,Helvetica,sans-serif;line-height:1.48;margin:0;color:#111827;background:#ffffff}
.report{max-width:1100px;margin:0 auto;background:#ffffff;min-height:100vh;padding:34px 42px;box-shadow:0 12px 34px rgba(15,23,42,.10)}
h1{color:#1e3a8a;margin-bottom:4px} h2{margin-top:28px;border-bottom:2px solid #dbeafe;padding-bottom:7px;color:#1e3a8a} h3{color:#111827}
.caveat{background:#fffbeb;border-left:5px solid #f59e0b;padding:12px 14px;border-radius:10px;margin:16px 0}.muted{color:#475569}
table{border-collapse:collapse;width:100%;margin:10px 0 16px 0;font-size:14px} th{width:28%;text-align:left;background:#eff6ff;color:#1e3a8a} td,th{border:1px solid #dbeafe;padding:8px 10px;vertical-align:top} td table{font-size:13px;margin:4px 0} ul{margin-top:6px}.figure{margin:22px 0;padding:12px;border:1px solid rgba(148,163,184,.28);border-radius:12px;background:#f8fafc}.figure img{max-width:100%;border-radius:8px}
.badge{display:inline-block;background:#dbeafe;color:#1e3a8a;padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700;margin-right:5px}
</style>
""",
        "</head><body><main class='report'>", f"<h1>{escape(title)}</h1>",
        "<p class='muted'>Generated from the current MitoAgent session. Sections are exported only when the corresponding UI/backend output exists in this session. HTML/PDF reports are formatted for human review; JSON/YAML remain available for machine-readable reproducibility.</p>",
        "<p class='caveat'><strong>Caveat:</strong> exploratory interpretation only; candidate hypotheses require experimental confirmation; OCR-only limitation applies.</p>",
    ]
    for sec in sections:
        if sec not in report:
            continue
        parts.append(f"<section><h2>{escape(display_section(sec))}</h2>")
        parts.append(_html_section_value(report.get(sec)))
        parts.append("</section>")
    if figure_paths:
        parts.append("<section><h2>Figures</h2>")
        for fp in figure_paths:
            path = Path(fp)
            if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(f"<div class='figure'><h3>{escape(path.stem.replace('_',' ').title())}</h3><img alt='{escape(path.name)}' src='data:{mime};base64,{data}'></div>")
        parts.append("</section>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def report_to_yaml(report: Mapping[str, Any]) -> str:
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(_json_ready(report), sort_keys=False, allow_unicode=True)
    except Exception:
        def _emit(obj, indent=0):
            sp = "  " * indent
            if isinstance(obj, dict):
                lines = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{sp}{k}:")
                        lines.extend(_emit(v, indent+1))
                    else:
                        lines.append(f"{sp}{k}: {humanize_value(v)}")
                return lines
            if isinstance(obj, list):
                lines = []
                for v in obj:
                    if isinstance(v, (dict, list)):
                        lines.append(f"{sp}-")
                        lines.extend(_emit(v, indent+1))
                    else:
                        lines.append(f"{sp}- {humanize_value(v)}")
                return lines
            return [f"{sp}{humanize_value(obj)}"]
        return "\n".join(_emit(_json_ready(report))) + "\n"


def _pdf_lines_for_value(val: Any, indent: int = 0) -> List[str]:
    val = _json_ready(val)
    prefix = "  " * indent
    lines: List[str] = []
    if isinstance(val, Mapping):
        for k, v in val.items():
            key = _clean_key(k)
            if isinstance(v, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_pdf_lines_for_value(v, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {humanize_value(v)}")
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}•")
                lines.extend(_pdf_lines_for_value(item, indent + 1))
            elif isinstance(item, list):
                lines.extend(_pdf_lines_for_value(item, indent + 1))
            else:
                lines.append(f"{prefix}• {humanize_value(item)}")
    else:
        lines.append(f"{prefix}{humanize_value(val)}")
    return lines


def report_to_pdf_bytes(report: Mapping[str, Any], *, title: str = "MitoAgent report", sections: Sequence[str] | None = None, figure_paths: Sequence[str] | None = None) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import textwrap
    sections = list(sections or report.keys())
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        fig.text(0.07, y, title, fontsize=18, weight="bold", color="#1e3a8a")
        y -= 0.035
        fig.text(0.07, y, "Exploratory interpretation only; candidate hypotheses require experimental confirmation; OCR-only limitation applies.", fontsize=8.5, color="#9a3412")
        y -= 0.04
        for sec in sections:
            if sec not in report:
                continue
            if y < 0.12:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.95
            fig.text(0.07, y, display_section(sec), fontsize=12.5, weight="bold", color="#1e3a8a")
            y -= 0.026
            for raw_line in _pdf_lines_for_value(report.get(sec)):
                for line in textwrap.wrap(raw_line, width=105)[:3]:
                    if y < 0.06:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                        fig = plt.figure(figsize=(8.27, 11.69))
                        y = 0.95
                    fig.text(0.08, y, line, fontsize=7.6, color="#0f172a")
                    y -= 0.015
            y -= 0.012
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        if figure_paths:
            import matplotlib.image as mpimg
            for fp in figure_paths:
                path = Path(fp)
                if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                img = mpimg.imread(str(path))
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.06, 0.96, path.stem.replace("_", " ").title(), fontsize=12, weight="bold", color="#1e3a8a")
                ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return buf.getvalue()
