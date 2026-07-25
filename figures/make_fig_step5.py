"""Publication-oriented diagnostic identifiability figure.

Inputs:
  results/identifiability/fim_dataset_I.json
  results/identifiability/profiles_dataset_I.json
  results/identifiability/parameter_interpretability_flags.csv

The figure deliberately labels the run level. Diagnostic outputs are not
publication-grade profile-likelihood claims until rerun with n_grid >= 25.
"""
from __future__ import annotations
import csv, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "identifiability"
FIG_DIR = ROOT / "figures"
FINAL_DIR = FIG_DIR / "final"
MANUSCRIPT_FINAL = ROOT / "manuscript" / "final"
for d in (FIG_DIR, FINAL_DIR, MANUSCRIPT_FINAL): d.mkdir(parents=True, exist_ok=True)

fim_path = RES_DIR / "fim_dataset_I.json"
prof_path = RES_DIR / "profiles_dataset_I.json"
flags_path = RES_DIR / "parameter_interpretability_flags.csv"
if not fim_path.exists():
    raise FileNotFoundError(f"missing {fim_path}; run python run_all.py --fast first")
with open(fim_path) as f: fim = json.load(f)
profiles = {}
prof_meta = {}
if prof_path.exists():
    with open(prof_path) as f:
        prof_meta = json.load(f); profiles = prof_meta.get("profiles", {}) or {}
flags = []
if flags_path.exists():
    with open(flags_path, newline="") as f:
        flags = list(csv.DictReader(f))

# 4-panel explanatory layout: FIM, profiles, verdict table, OCR-only interpretability.
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.45, height_ratios=[1.05, 1.15, 1.1])
ax_eig = fig.add_subplot(gs[0, 0:2])
ax_verdict = fig.add_subplot(gs[0, 2:4])
profile_axes = [fig.add_subplot(gs[1 + i//4, i % 4]) for i in range(8)]

# A. FIM eigenvalues
raw = np.array(fim.get("eigvals_raw", []), dtype=float)
clip = np.array(fim.get("eigvals_clipped", []), dtype=float)
ranks = np.arange(1, len(raw)+1)
floor = float(fim.get("eig_clip_floor", 1e-12))
if len(raw):
    ax_eig.bar(ranks - 0.18, np.log10(np.maximum(np.abs(raw), 1e-30)), width=0.34, label="raw eigenvalues")
    ax_eig.bar(ranks + 0.18, np.log10(np.maximum(clip, floor)), width=0.34, label="clipped for inverse-FIM discussion")
    ax_eig.axhline(np.log10(floor), linestyle="--", linewidth=0.8)
ax_eig.set_title("A. FIM eigenvalue spectrum", loc="left", fontweight="bold")
ax_eig.set_xlabel("eigenvalue rank (1 = smallest)")
ax_eig.set_ylabel(r"$\log_{10}$ eigenvalue")
ax_eig.grid(True, alpha=0.25, axis="y")
ax_eig.legend(fontsize=8)
cond_text = f"raw condition = {float(fim.get('condition_raw', np.nan)):.2e}\nclipped condition = {float(fim.get('condition_clipped', np.nan)):.2e}\nFIM is local; profiles are required for CIs"
ax_eig.text(0.98, 0.04, cond_text, transform=ax_eig.transAxes, ha="right", va="bottom", fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5"))

# B. Verdict table
ax_verdict.axis("off")
lines = ["B. Profile-likelihood verdicts / flags", ""]
if flags:
    for row in flags[:12]:
        lines.append(f"{row['parameter']:<13s} {row['interpretability_flag']:<18s} {row['profile_verdict']}")
else:
    lines += ["No profile flag CSV found.", "Only FIM diagnostics available."]
lines += ["", f"run level: {prof_meta.get('diagnostic_level', fim.get('diagnostic_level', 'unknown'))}", f"profile grid: {prof_meta.get('n_grid_used', 'not run')}", "Diagnostic profiles are not final publication CIs."]
ax_verdict.text(0, 1, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)

# C. Profile curves
core_names = ['k_supply', 'c_tot', 'V_max', 'K_o', 'K_r', 'gamma_oligo', 'tau_kappa', 'r0']
label_map = {'k_supply':r'$k_{supply}$','c_tot':r'$c_{tot}$','V_max':r'$V_{max}$','K_o':r'$K_o$','K_r':r'$K_r$','gamma_oligo':r'$\gamma_{oligo}$','tau_kappa':r'$\tau_\kappa$','r0':r'$r_0$'}
for ax, nm in zip(profile_axes, core_names):
    pr = profiles.get(nm)
    if not pr:
        ax.text(0.5,0.5,"profile not run",ha="center",va="center")
        ax.set_title(label_map.get(nm,nm), loc="left", fontweight="bold", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    grid = np.array(pr.get("theta_grid", []), dtype=float)
    delta = np.array(pr.get("delta_nll", []), dtype=float)
    ok = np.array(pr.get("optimizer_success", [True]*len(grid)), dtype=bool)
    finite = np.isfinite(delta) & ok
    if finite.any():
        ax.plot(grid[finite], delta[finite], marker="o", linewidth=1.2, markersize=3)
    fail = (~finite)
    if fail.any() and len(grid):
        ax.scatter(grid[fail], np.zeros(np.sum(fail)), marker="x", s=25)
    ax.axhline(3.841, linestyle="--", linewidth=0.8)
    ax.axvline(float(pr.get("map_value", np.nan)), linewidth=0.8)
    if pr.get("ci_low") is not None: ax.axvline(float(pr["ci_low"]), linestyle=":", linewidth=0.8)
    if pr.get("ci_high") is not None: ax.axvline(float(pr["ci_high"]), linestyle=":", linewidth=0.8)
    if len(grid) and np.nanmin(grid) > 0 and np.nanmax(grid)/np.nanmin(grid) > 10: ax.set_xscale("log")
    verdict = pr.get("practical_id", "")
    ax.set_title(f"{label_map.get(nm,nm)} — {verdict}", loc="left", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=-0.25)
    if np.isfinite(delta).any():
        ax.set_ylim(top=min(max(5, float(np.nanmax(delta))*1.05), 30))
    if ax in profile_axes[4:]: ax.set_xlabel(nm, fontsize=8)
    if ax in (profile_axes[0], profile_axes[4]): ax.set_ylabel(r"$2\Delta$NLL")

fig.suptitle("Identifiability diagnostics for the 3-state OCR-informed model — dataset_I", fontweight="bold", y=0.99)
fig.text(0.5, 0.01, "FIM is a local diagnostic; profile likelihoods assess practical identifiability. OCR-only data cannot support direct interpretation of all parameters. Demo/synthetic data are diagnostic only.", ha="center", fontsize=9)

for out in [FIG_DIR/"fig_step5_identifiability.png", FINAL_DIR/"fig5_identifiability.png", MANUSCRIPT_FINAL/"fig5_identifiability.png"]:
    fig.savefig(out, dpi=180, bbox_inches="tight")
for out in [FIG_DIR/"fig_step5_identifiability.pdf", FINAL_DIR/"fig5_identifiability.pdf", MANUSCRIPT_FINAL/"fig5_identifiability.pdf"]:
    fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("Saved identifiability figure outputs")
