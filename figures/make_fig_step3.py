"""Figure for Step 3: visual confirmation of parsing correctness.
Shows all six chambers (3 datasets x 2 chambers each) with detected
event boundaries overlaid as vertical lines.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from data_io.loader import load_dataset
from data_io.preprocess import preprocess
from core.paths import DATA_SAMPLES

EXCEL_DIR = str(DATA_SAMPLES)

datasets = ["dataset_I", "dataset_II", "dataset_III"]
fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=False)

for i, name in enumerate(datasets):
    ds = load_dataset(os.path.join(EXCEL_DIR, f"{name}.xlsx"))
    for j, ch in enumerate(ds.chambers):
        ax = axes[i, j]
        # Raw vs preprocessed
        ax.plot(ch.t, ch.o, color="lightgray", lw=0.6, label="raw")
        ch_p, _ = preprocess(ch, do_outliers=True, do_smooth=False)
        ax.plot(ch_p.t, ch_p.o, color="C0", lw=1.0,
                label=f"after preproc (n={len(ch_p.t)})")

        # Event vertical lines
        for tj, lab, c in [(ch.t_oligo, "Oligo", "C1"),
                           *[(t_, f"FCCP{k+1}", "C2") for k, t_ in enumerate(ch.t_fccp)],
                           (ch.t_inhibit, "Rot/AntA", "C3")]:
            if tj is None: continue
            ax.axvline(tj, color=c, lw=1.0, ls="--", alpha=0.85)
            ax.text(tj, ax.get_ylim()[1] if i == 0 else ch.o.max()*0.99,
                    f" {lab}", rotation=90, fontsize=7, va="top",
                    color=c, alpha=0.85)

        ax.set_title(f"{name} • {ch.label}", fontsize=9, loc="left",
                     fontweight="bold")
        ax.set_ylabel(r"O$_2$ (nmol/mL)")
        if i == 2:
            ax.set_xlabel("time (s)")
        ax.legend(loc="lower left", fontsize=7)
        ax.grid(True, alpha=0.3)
        # Annotation
        ax.text(0.99, 0.04,
                f"σ̂={ch.sigma_obs_est:.2f}  N_FCCP={len(ch.t_fccp)}",
                transform=ax.transAxes, ha="right", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.85))

plt.tight_layout()
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_legacy_path = os.path.join(_root, "figures", "fig_step3_parsing.png")
os.makedirs(os.path.dirname(_legacy_path), exist_ok=True)
plt.savefig(_legacy_path, dpi=160, bbox_inches="tight")
for _dest in (os.path.join(_root, "figures", "final"),
              os.path.join(_root, "manuscript", "final")):
    os.makedirs(_dest, exist_ok=True)
    plt.savefig(os.path.join(_dest, "fig3_data_pipeline.png"),
                dpi=220, bbox_inches="tight")
    plt.savefig(os.path.join(_dest, "fig3_data_pipeline.pdf"),
                bbox_inches="tight")
plt.close(fig)
print(f"Saved: {_legacy_path} + figures/final/fig3_data_pipeline.{{png,pdf}}"
      f" + manuscript/final/fig3_data_pipeline.{{png,pdf}}")
