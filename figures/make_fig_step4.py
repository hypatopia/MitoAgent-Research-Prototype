"""Publication-oriented calibration figure with residual diagnostics.

Reads structured calibration JSONs and phase summaries. The figure is a
methodological diagnostic based on bundled demo/synthetic traces unless real
Oroboros data are substituted by the user.
"""
from __future__ import annotations
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_io.loader import load_excel
from data_io.preprocess import preprocess
from core.reduced_model import simulate, CORE_PARAM_ORDER
from core.paths import DATA_SAMPLES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results", "calibration")
DATA_DIR = str(DATA_SAMPLES)
FIG_DIR = os.path.join(ROOT, "figures")
FINAL_DIR = os.path.join(FIG_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)

datasets = ["dataset_I", "dataset_II", "dataset_III"]
calibs = {}
for name in datasets:
    p = os.path.join(RES_DIR, f"calib_{name}.json")
    if os.path.exists(p):
        with open(p) as f:
            calibs[name] = json.load(f)

if not calibs:
    raise FileNotFoundError(
        f"No calibration JSONs found in {RES_DIR}. Run `python run_all.py --fast` first.")

present = [n for n in datasets if n in calibs]
fig = plt.figure(figsize=(5.4 * len(present), 7.2))
gs = GridSpec(3, len(present), figure=fig, height_ratios=[2.2, 1.0, 1.0], hspace=0.25, wspace=0.30)
all_resids = []

for k, name in enumerate(present):
    cal = calibs[name]
    ds = load_excel(os.path.join(DATA_DIR, f"{name}.xlsx"))
    ch_raw = ds.chambers[0]
    ch, prep_warnings = preprocess(ch_raw)
    proto = ch.to_protocol()
    params = {kk: cal["params"][kk] for kk in CORE_PARAM_ORDER if kk in cal["params"]}
    params["alphas"] = list(cal.get("alphas", []))
    t = np.asarray(ch_raw.t, dtype=float)
    o = np.asarray(ch_raw.o, dtype=float)
    sim = simulate(params, proto, o2_init=float(o[0]), t_eval=t)
    residual = sim.o - o
    all_resids.extend(residual[np.isfinite(residual)].tolist())

    ax = fig.add_subplot(gs[0, k])
    ax.scatter(t, o, s=3, alpha=0.42, label="observed O$_2$")
    ax.plot(sim.t, sim.o, lw=1.5, label="fitted O$_2$")
    for label, ti in [("oligomycin", ch_raw.t_oligo), ("Rot/Ant", ch_raw.t_inhibit)]:
        if ti is not None:
            ax.axvline(ti, ls="--", lw=0.8)
    for ti in ch_raw.t_fccp:
        ax.axvline(ti, ls=":", lw=0.8)
    ax.set_title(f"{name}: fit", loc="left", fontweight="bold")
    ax.set_xlabel("time [s]")
    if k == 0:
        ax.set_ylabel("O$_2$ [nmol/mL]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    warn = cal.get("warnings", []) or []
    txt = (
        f"objective: {cal.get('objective_type', '')}\n"
        f"RMSE: {cal.get('rmse_calib', float('nan')):.3f} nmol/mL\n"
        f"within-trace σ: {cal.get('sigma_obs', float('nan')):.3f}\n"
        f"warnings: {len(warn)}"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            family="monospace", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    axr = fig.add_subplot(gs[1, k], sharex=ax)
    axr.axhline(0.0, lw=0.8)
    axr.plot(t, residual, lw=0.8)
    axr.set_xlabel("time [s]")
    if k == 0:
        axr.set_ylabel("residual\nfit − observed")
    axr.grid(True, alpha=0.25)

    axt = fig.add_subplot(gs[2, k])
    ph_path = os.path.join(RES_DIR, f"phase_summary_{name}.json")
    if os.path.exists(ph_path):
        with open(ph_path) as f:
            ph = json.load(f).get("phases", [])
        labels = [p["phase"].replace("/", "/\n") for p in ph]
        values = [p.get("observed_ocr_nmol_ml_s") or 0.0 for p in ph]
        axt.bar(range(len(values)), values)
        axt.set_xticks(range(len(labels)), labels, rotation=0, fontsize=8)
        axt.set_ylabel("observed OCR\nfrom −slope")
        axt.set_title("phase OCR summary", loc="left", fontsize=9)
        axt.grid(True, axis="y", alpha=0.25)
    else:
        axt.text(0.5, 0.5, "phase summary missing", ha="center", va="center")
        axt.axis("off")

fig.suptitle(
    "Calibration fits, residuals, and phase-level OCR summaries\n"
    "Deterministic SSE objective with post-hoc within-trace observational-noise estimate",
    fontweight="bold", y=0.995)

_MAN_FINAL = os.path.join(ROOT, "manuscript", "final")
os.makedirs(_MAN_FINAL, exist_ok=True)
for out in [
    os.path.join(FIG_DIR, "fig_step4_calibration.png"),
    os.path.join(FIG_DIR, "fig_step4_calibration.pdf"),
    os.path.join(FINAL_DIR, "fig4_calibration.png"),
    os.path.join(FINAL_DIR, "fig4_calibration.pdf"),
    os.path.join(FINAL_DIR, "fig_calibration.png"),
    os.path.join(FINAL_DIR, "fig_calibration.pdf"),
    os.path.join(_MAN_FINAL, "fig4_calibration.png"),
    os.path.join(_MAN_FINAL, "fig4_calibration.pdf"),
]:
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
