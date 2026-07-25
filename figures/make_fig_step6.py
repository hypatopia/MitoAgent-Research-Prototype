"""Figure for Step 6: sensitivity analysis.

Reads from the structured CHUNK-4 outputs:
    results/sensitivity/morris_dataset_I.json     (always required)
    results/sensitivity/sobol_auc_dataset_I.json  (always required)
    results/sensitivity/time_resolved_sobol_dataset_I.npz +
    results/sensitivity/time_resolved_sobol_dataset_I.meta.json
                                                   (publication only)

Layout adapts to which files are present:
  * Morris + Sobol AUC only      -> 2-panel layout
  * + time-resolved Sobol        -> 3-panel layout with variance-degenerate
                                     intervals shaded on panel (c)
"""
from __future__ import annotations
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results", "sensitivity")
FIG_DIR = os.path.join(ROOT, "figures")

morris_path = os.path.join(RES_DIR, "morris_dataset_I.json")
sobol_path  = os.path.join(RES_DIR, "sobol_auc_dataset_I.json")
trs_npz     = os.path.join(RES_DIR, "time_resolved_sobol_dataset_I.npz")
trs_meta    = os.path.join(RES_DIR, "time_resolved_sobol_dataset_I.meta.json")

if not os.path.exists(morris_path):
    raise FileNotFoundError(
        f"Morris JSON not found at {morris_path}. "
        "Run `python run_all.py --fast` first.")
if not os.path.exists(sobol_path):
    raise FileNotFoundError(
        f"Sobol JSON not found at {sobol_path}. "
        "Run `python run_all.py --fast` first.")

with open(morris_path) as f:
    M = json.load(f)
with open(sobol_path) as f:
    S = json.load(f)

has_trs = os.path.exists(trs_npz) and os.path.exists(trs_meta)
if has_trs:
    npz  = np.load(trs_npz)
    with open(trs_meta) as f:
        meta = json.load(f)

# Pretty parameter labels
def pretty(nm: str) -> str:
    LBL = {
        "k_supply":    r"$k_{\mathrm{sup}}$",
        "c_tot":       r"$c_{\mathrm{tot}}$",
        "V_max":       r"$V_{\max}$",
        "K_o":         r"$K_o$",
        "K_r":         r"$K_r$",
        "gamma_oligo": r"$\gamma_{\mathrm{oligo}}$",
        "tau_kappa":   r"$\tau_{\kappa}$",
        "r0":          r"$r_0$",
    }
    if nm in LBL:
        return LBL[nm]
    if nm.startswith("alpha_"):
        return rf"$\alpha_{{{nm.split('_')[1]}}}$"
    return nm


# ── Layout ───────────────────────────────────────────────────────────────
if has_trs:
    fig = plt.figure(figsize=(15, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
else:
    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

# ── (a) Morris mu_star vs sigma ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
names_m  = M["names"]
mu_star  = np.array(M["mu_star"], dtype=float)
sigma_m  = np.array(M["sigma"], dtype=float)
labels_m = [pretty(n) for n in names_m]
order    = np.argsort(-mu_star)  # most sensitive first
for i, idx in enumerate(order):
    ax.barh(len(order) - 1 - i, mu_star[idx], color="C0", alpha=0.85)
    ax.text(mu_star[idx], len(order) - 1 - i, f" {labels_m[idx]}",
             va="center", fontsize=9)
ax.set_yticks([])
ax.set_xlabel(r"$\mu^*$  (Morris)")
ax.set_title(f"(a) Morris screening — {len(names_m)} parameters\n"
              f"metric: {M.get('metric')}, n_eval={M.get('n_evals')}, "
              f"diagnostic level: {M.get('diagnostic_level')}",
              loc="left", fontweight="bold", fontsize=10)
ax.grid(True, alpha=0.3, axis="x")

# ── (b) Sobol AUC: S1 + ST with explicit caveat ──────────────────────────
ax = fig.add_subplot(gs[0, 1])
names_s = S["names"]
S1     = np.array(S["S1"], dtype=float)
S1_cf  = np.array(S["S1_conf"], dtype=float)
ST     = np.array(S["ST"], dtype=float)
ST_cf  = np.array(S["ST_conf"], dtype=float)
labels_s = [pretty(n) for n in names_s]
x = np.arange(len(names_s))
w = 0.4
ax.bar(x - w/2, S1, w, yerr=S1_cf, color="C0", label="$S_1$ (first-order)",
        capsize=2, error_kw={"alpha": 0.5})
ax.bar(x + w/2, ST, w, yerr=ST_cf, color="C3",
        label="$S_T$ (total; INCLUDES interactions)",
        capsize=2, error_kw={"alpha": 0.5})
ax.set_xticks(x)
ax.set_xticklabels(labels_s, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylabel("Sobol index")
ax.set_title(f"(b) Sobol indices on {S.get('metric')}\n"
              f"n_eval={S.get('n_evals')}, "
              f"diagnostic level: {S.get('diagnostic_level')}",
              loc="left", fontweight="bold", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.text(0.02, 0.98,
         "$S_T$ is NOT a partition of variance; sums of $S_T$ can exceed 1.",
         transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))
ax.grid(True, alpha=0.3, axis="y")

# ── (c) Time-resolved Sobol (publication only) ───────────────────────────
if has_trs:
    ax = fig.add_subplot(gs[0, 2])
    t_grid     = npz["t_grid"]
    S1_t       = npz["S1_t"]
    var_degen  = npz["variance_degenerate_mask"].astype(bool)
    names_trs  = meta.get("names", [])
    n_params   = S1_t.shape[0]
    cmap       = plt.cm.tab10
    for i in range(min(n_params, 10)):
        # Mask variance-degenerate points to NaN so the line is broken there
        y = S1_t[i].copy()
        y[var_degen] = np.nan
        ax.plot(t_grid, y, color=cmap(i % 10),
                 label=pretty(names_trs[i] if i < len(names_trs) else f"p{i}"),
                 lw=1.2)
    # Shade variance-degenerate intervals
    if var_degen.any():
        # Find contiguous intervals of variance-degenerate time points
        in_int = False
        i0 = 0
        for k in range(len(var_degen)):
            if var_degen[k] and not in_int:
                i0 = k; in_int = True
            elif (not var_degen[k]) and in_int:
                ax.axvspan(t_grid[i0], t_grid[k - 1], color="gray",
                            alpha=0.25)
                in_int = False
        if in_int:
            ax.axvspan(t_grid[i0], t_grid[-1], color="gray", alpha=0.25)
        # Label
        ax.text(0.98, 0.02,
                 "grey: variance-degenerate\n(post-Rot/AntA);\nnot interpreted",
                 transform=ax.transAxes, ha="right", va="bottom",
                 fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$S_1(t)$")
    ax.set_title(f"(c) Time-resolved Sobol $S_1(t)$\n"
                  f"n_eval={meta.get('n_evals')}, "
                  f"diagnostic level: {meta.get('diagnostic_level')}",
                  loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle("Sensitivity analysis — dataset_I",
              fontweight="bold", y=1.02)
fig_path = os.path.join(FIG_DIR, "fig_step6_sensitivity.png")
plt.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig_step6_sensitivity.pdf"), bbox_inches="tight")
final_dir = os.path.join(FIG_DIR, "final")
os.makedirs(final_dir, exist_ok=True)
plt.savefig(os.path.join(final_dir, "fig6_sensitivity.png"), dpi=220, bbox_inches="tight")
plt.savefig(os.path.join(final_dir, "fig6_sensitivity.pdf"), bbox_inches="tight")
# manuscript mirror
man_final = os.path.join(ROOT, "manuscript", "final")
os.makedirs(man_final, exist_ok=True)
plt.savefig(os.path.join(man_final, "fig6_sensitivity.png"), dpi=220, bbox_inches="tight")
plt.savefig(os.path.join(man_final, "fig6_sensitivity.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig_path}")
print(f"Saved: {os.path.join(final_dir, 'fig6_sensitivity.png')}")
