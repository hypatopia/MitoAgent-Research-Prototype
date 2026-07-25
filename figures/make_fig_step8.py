"""Figure for Step 8: AI agent architecture."""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np, os

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

def box(x, y, w, h, label, color, fontsize=9, edgecolor='black', alpha=0.85):
    rect = mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                              fc=color, ec=edgecolor, lw=1.3, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', wrap=True)

def arrow(x1, y1, x2, y2, color='black', lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                  arrowprops=dict(arrowstyle='->', lw=lw, color=color))

# ── User-facing layer ───────────────────────────────────────────────
box(0.5, 8.6, 3, 1.0, 'User\n(scientist / clinician)', '#fef3bd')
box(4.5, 8.6, 3, 1.0, 'CLI\n(python -m agent.cli)', '#cdeac0')
box(8.5, 8.6, 3, 1.0, 'Python script\n(programmatic API)', '#cdeac0')
arrow(2.0, 8.6, 6.0, 8.6, color='gray')
arrow(2.0, 8.6, 10.0, 8.6, color='gray')

# Optional LLM
box(12.0, 8.6, 1.5, 1.0, 'LLM\n(optional)', '#f3c5c5')
arrow(2.0, 8.6, 12.5, 8.6, color='gray', lw=0.8)

# ── Driver layer ────────────────────────────────────────────────────
box(2.0, 7.0, 4, 1.1, 'NaturalLanguageDriver\n(offline router OR LLM adapter)',
     '#d6eaf8')
box(7.5, 7.0, 4, 1.1, 'MitoAgent.run_pipeline()\nstate machine + diagnostic gates',
     '#d6eaf8')
arrow(6.0, 8.4, 4.0, 8.1)
arrow(10.0, 8.4, 9.5, 8.1)
arrow(12.5, 8.4, 4.0, 8.1, color='gray', lw=0.8)
arrow(4.0, 7.0, 7.5, 7.0)

# ── Agent state ──────────────────────────────────────────────────────
box(0.3, 4.4, 2.5, 2.0,
    'AgentState\n• chamber\n• params\n• last_calib\n• last_stability\n• last_identif\n• last_sens\n• last_validation\n• log',
    '#fce5cd', fontsize=8)
arrow(4.0, 7.0, 2.0, 6.4)

# ── Tools layer ─────────────────────────────────────────────────────
tools = [
    ('load_data',               '#f9d4d4', 3.5, 5.0, 1.95, 0.55),
    ('preprocess_data',         '#f9d4d4', 5.55, 5.0, 1.95, 0.55),
    ('simulate_default',        '#f9d4d4', 7.6,  5.0, 1.95, 0.55),
    ('calibrate',               '#fdebd0', 9.65, 5.0, 1.95, 0.55),

    ('check_stability',         '#d4efdf', 3.5, 4.3, 1.95, 0.55),
    ('analyze_identifiability', '#d4efdf', 5.55, 4.3, 1.95, 0.55),
    ('analyze_sensitivity',     '#d4efdf', 7.6, 4.3, 1.95, 0.55),
    ('validate',                '#d4efdf', 9.65, 4.3, 1.95, 0.55),
]
for name, color, x, y, w, h in tools:
    box(x, y, w, h, name, color, fontsize=8)

# Tools header
ax.text(7.6, 5.85, 'Typed tool layer  (agent/tools.py)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         color='dimgray')

# Connecting arrows: orchestrator -> tools
arrow(9.5, 7.0, 4.5, 5.6, color='gray', lw=0.8)
arrow(9.5, 7.0, 6.5, 5.6, color='gray', lw=0.8)
arrow(9.5, 7.0, 8.5, 5.6, color='gray', lw=0.8)
arrow(9.5, 7.0, 10.5, 5.6, color='gray', lw=0.8)

# ── Methodology layer ────────────────────────────────────────────────
modules = [
    ('core/reduced_model.py\n(Step 1 — reduced 3-state ODE)',           '#bbdefb', 0.5, 2.6, 3.0, 0.95),
    ('core/diagnostics.py\n(Step 2 — stability)',                        '#bbdefb', 4.0, 2.6, 2.5, 0.95),
    ('data_io/loader.py + preprocess.py\n(Step 3 — data pipeline)',     '#bbdefb', 7.0, 2.6, 3.5, 0.95),
    ('calibration/calibrate.py\n(Step 4 — DE / staged / hierarchical)', '#bbdefb', 11.0, 2.6, 2.5, 0.95),
    ('analysis/identifiability.py\n(Step 5 — FIM + profiles)',          '#bbdefb', 0.5, 1.4, 3.0, 0.95),
    ('analysis/sensitivity.py\n(Step 6 — Morris/Sobol/time-resolved)',  '#bbdefb', 4.0, 1.4, 4.5, 0.95),
    ('analysis/validation.py\n(Step 7 — within-trace, LODO, PPC)',      '#bbdefb', 9.0, 1.4, 4.5, 0.95),
]
for label, color, x, y, w, h in modules:
    box(x, y, w, h, label, color, fontsize=8)
ax.text(7.0, 3.85, 'Methodology layer  (paper-grade modules)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         color='dimgray')

# Connecting arrow: tools -> methodology
arrow(7.6, 4.3, 7.6, 3.6, color='gray', lw=0.8)

# ── External resources ──────────────────────────────────────────────
box(0.5, 0.1, 5, 0.8, 'Data files (xlsx / csv / npy)', '#e8e8e8')
box(8.5, 0.1, 5, 0.8, 'Saved results / reports (JSON / NPZ / PNG)', '#e8e8e8')
arrow(3.0, 0.9, 3.0, 1.4, color='gray', lw=0.8)
arrow(11.0, 1.4, 11.0, 0.9, color='gray', lw=0.8)

# Title
ax.text(7.0, 9.85, "MitoAgent — architecture",
         ha="center", va="center", fontsize=14, fontweight='bold')

plt.tight_layout()
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_legacy_path = os.path.join(_root, "figures", "fig_step8_agent_architecture.png")
os.makedirs(os.path.dirname(_legacy_path), exist_ok=True)
plt.savefig(_legacy_path, dpi=160, bbox_inches="tight")
for _dest in (os.path.join(_root, "figures", "final"),
              os.path.join(_root, "manuscript", "final")):
    os.makedirs(_dest, exist_ok=True)
    plt.savefig(os.path.join(_dest, "fig8_agent_architecture.png"),
                dpi=220, bbox_inches="tight")
    plt.savefig(os.path.join(_dest, "fig8_agent_architecture.pdf"),
                bbox_inches="tight")
plt.close(fig)
print(f"Saved: {_legacy_path} + figures/final/fig8_agent_architecture.{{png,pdf}}"
      f" + manuscript/final/fig8_agent_architecture.{{png,pdf}}")
