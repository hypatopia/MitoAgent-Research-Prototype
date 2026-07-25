"""Figure for Step 7: validation diagnostics.

This figure distinguishes four validation diagnostics:
  (a) Chamber A -> Chamber B technical-replicate transfer
  (b) Within-trace intervention-phase extrapolation diagnostic
  (c) Leave-one-dataset-out pooled-transfer diagnostic status
  (d) Parametric-bootstrap predictive check

None of these panels is presented as proof of biological disease/generalisation.
"""
from __future__ import annotations
import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results", "validation")
FIG_DIR = os.path.join(ROOT, "figures")
FINAL_DIR = os.path.join(FIG_DIR, "final")
MAN_FINAL = os.path.join(ROOT, "manuscript", "final")
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(MAN_FINAL, exist_ok=True)

bs_path = os.path.join(RES_DIR, "parametric_bootstrap_predictive_check_dataset_I.json")
wt_path = os.path.join(RES_DIR, "within_trace_holdout_dataset_I.json")
ch_path = os.path.join(RES_DIR, "chamber_holdout_summary.csv")
lodo_path = os.path.join(RES_DIR, "lodo_summary.csv")

if not os.path.exists(bs_path):
    raise FileNotFoundError(f"Parametric-bootstrap JSON not found: {bs_path}")
with open(bs_path) as f:
    BS = json.load(f)
WT = None
if os.path.exists(wt_path):
    with open(wt_path) as f:
        WT = json.load(f)
CH_ROWS = []
if os.path.exists(ch_path):
    with open(ch_path, newline="") as f:
        CH_ROWS = list(csv.DictReader(f))
LODO_ROWS = []
if os.path.exists(lodo_path):
    with open(lodo_path, newline="") as f:
        LODO_ROWS = list(csv.DictReader(f))

fig = plt.figure(figsize=(12.5, 9.0))
gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

# Panel A: chamber transfer summary
axA = fig.add_subplot(gs[0, 0])
axA.set_title("(a) Technical-replicate transfer", loc="left", fontweight="bold")
if CH_ROWS and CH_ROWS[0].get("rmse_test") not in (None, ""):
    row = CH_ROWS[0]
    vals = [float(row.get("rmse_train", "nan")), float(row.get("rmse_test", "nan"))]
    labels = ["fit chamber", "predicted chamber"]
    axA.bar(labels, vals)
    axA.set_ylabel("RMSE [nmol/mL]")
    axA.text(0.02, 0.95,
             "Chamber A fitted parameters -> Chamber B prediction\nnot independent biological validation",
             transform=axA.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
else:
    axA.text(0.5, 0.5, "Chamber-transfer diagnostic not run",
             ha="center", va="center", transform=axA.transAxes)
    axA.set_axis_off()
axA.grid(True, alpha=0.3, axis="y")

# Panel B: within-trace holdout
axB = fig.add_subplot(gs[0, 1])
axB.set_title("(b) Intervention-phase extrapolation", loc="left", fontweight="bold")
if WT is not None and WT.get("t_full"):
    t_full = np.asarray(WT["t_full"], dtype=float)
    o_data = np.asarray(WT["o_data"], dtype=float)
    o_pred = np.asarray(WT["o_pred"], dtype=float)
    n_train = int(WT.get("n_train", 0))
    if 0 < n_train < len(t_full):
        axB.axvspan(t_full[0], t_full[n_train], alpha=0.25, label="train segment")
    axB.scatter(t_full, o_data, s=5, alpha=0.55, label="data")
    if len(o_pred) == len(t_full):
        axB.plot(t_full, o_pred, lw=1.4, label="prediction")
    axB.set_xlabel("time [s]")
    axB.set_ylabel("O2 [nmol/mL]")
    txt = f"RMSE_train={WT.get('rmse_train'):.3f}\nRMSE_test={WT.get('rmse_test'):.3f}\n{WT.get('diagnostic_level', 'diagnostic')}"
    axB.text(0.02, 0.03, txt, transform=axB.transAxes, ha="left", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    axB.legend(fontsize=8)
else:
    axB.text(0.5, 0.5, "Within-trace holdout not run", ha="center", va="center", transform=axB.transAxes)
    axB.set_axis_off()
axB.grid(True, alpha=0.3)

# Panel C: LODO status
axC = fig.add_subplot(gs[1, 0])
axC.set_title("(c) LODO pooled-transfer diagnostic", loc="left", fontweight="bold")
axC.set_axis_off()
if LODO_ROWS:
    row = LODO_ROWS[0]
    status = row.get("status", "not_run")
    interpretation = row.get("interpretation", "LODO status unavailable.")
    limitation = row.get("limitation", "Not proof of biological generalization.")
    text = f"Status: {status}\n\n{interpretation}\n\nLimitation: {limitation}"
else:
    text = "LODO summary file not found."
axC.text(0.02, 0.98, text, ha="left", va="top", transform=axC.transAxes,
         fontsize=9, wrap=True, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))

# Panel D: parametric-bootstrap predictive check
axD = fig.add_subplot(gs[1, 1])
t = np.asarray(BS["envelope_t"], dtype=float)
o_data = np.asarray(BS["o_data"], dtype=float)
o_hat = np.asarray(BS["o_hat"], dtype=float)
lo90 = np.asarray(BS["envelope_lo90"], dtype=float)
hi90 = np.asarray(BS["envelope_hi90"], dtype=float)
cov = float(BS["parametric_bootstrap_coverage_90"])
diag_level = BS.get("diagnostic_level", "fast")
n_boot = int(BS.get("n_boot", 0))
axD.fill_between(t, lo90, hi90, alpha=0.25, label="90% envelope")
axD.plot(t, o_hat, lw=1.5, label="MAP prediction")
axD.scatter(t, o_data, s=4, alpha=0.55, label="data")
axD.set_xlabel("time [s]")
axD.set_ylabel("O2 [nmol/mL]")
axD.set_title(f"(d) Parametric-bootstrap predictive check ({diag_level})", loc="left", fontweight="bold")
axD.legend(fontsize=8)
axD.grid(True, alpha=0.3)
axD.text(0.02, 0.98,
         f"coverage_90={cov:.1%}\nn_boot={n_boot}\nnot posterior predictive",
         transform=axD.transAxes, ha="left", va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

fig.suptitle("Validation diagnostics — dataset_I", fontweight="bold", y=0.995)
fig.text(0.5, 0.01,
         "Validation diagnostics assess workflow behavior and technical transfer; they do not prove biological disease/generalization.",
         ha="center", va="bottom", fontsize=8, style="italic")

for name in ["fig_step7_validation", os.path.join("final", "fig7_validation")]:
    stem = os.path.join(FIG_DIR, name)
    fig.savefig(stem + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
# manuscript copy
fig.savefig(os.path.join(MAN_FINAL, "fig7_validation.png"), dpi=180, bbox_inches="tight")
fig.savefig(os.path.join(MAN_FINAL, "fig7_validation.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {os.path.join(FINAL_DIR, 'fig7_validation.png')}")
