"""Generate Figure 1: model-overview visualization.

Shows the four observable/state panels of the 3-state OCR-informed
model on the dataset_I protocol: oxygen concentration o(t), analytic
OCR(t), reduced cytochrome-c pool r(t), and the latent effective
respiratory-drive factor kappa(t). The figure is used in the
manuscript to introduce the model states and the smooth-protocol
event structure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from core.reduced_model import simulate, Protocol, DEFAULT_PARAMS

# Shared protocol: dataset_I shape
proto = Protocol(t_oligo=300, t_fccp=[480], t_inhibit=660, t_end=826,
                 t_start=210, k_step=2.0)

# Model: simulate the 3-state OCR-informed model
params = dict(DEFAULT_PARAMS)
params["alphas"] = [1.0]
res = simulate(params, proto, o2_init=170.0,
               t_eval=np.linspace(proto.t_start, proto.t_end, 800))

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
event_lines = [(proto.t_oligo, "Oligomycin", "C1"),
               *((tj, f"FCCP {j+1}", "C2") for j, tj in enumerate(proto.t_fccp)),
               (proto.t_inhibit, "Rot/AntA", "C3")]

# Top-left: O2 trajectory
ax = axes[0, 0]
ax.plot(res.t, res.o, lw=2, color="C0")
ax.set_ylabel(r"O$_2$  (nmol/mL)")
ax.set_title("Observable: $O_2(t)$", loc="left", fontweight="bold")
ax.grid(True, alpha=0.3)

# Top-right: OCR (computed analytically from C4)
ax = axes[0, 1]
ax.plot(res.t, res.OCR, lw=2, color="C4")
ax.set_ylabel("OCR  (nmol/mL/s)")
ax.set_title(r"Analytic OCR $= \frac{1}{2}\,v_{CIV}$", loc="left",
             fontweight="bold")
ax.grid(True, alpha=0.3)

# Bottom-left: cyt c reduced
ax = axes[1, 0]
ax.plot(res.t, res.r, lw=2, color="C5")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$r$ (nmol/mL)")
ax.set_title("State: cyt c reduced", loc="left", fontweight="bold")
ax.grid(True, alpha=0.3)

# Bottom-right: kappa effective respiratory-drive factor
ax = axes[1, 1]
ax.plot(res.t, res.kappa, lw=2, color="C6")
ax.axhline(1.0, color="gray", lw=0.6, ls="--")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\kappa$ (–)")
ax.set_title(r"State: effective respiratory-drive factor $\kappa(t)$",
             loc="left", fontweight="bold")
ax.grid(True, alpha=0.3)

# Vertical event lines
for ax in axes.flat:
    for tj, lab, c in event_lines:
        ax.axvline(tj, color=c, lw=0.8, ls=":", alpha=0.7)

# Legend strip
fig.suptitle("3-state, 8-parameter OCR-informed model on dataset_I protocol",
             y=0.995, fontweight="bold")
plt.tight_layout()
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_legacy_path = os.path.join(_root, "figures", "fig_step1_reduced_model.png")
os.makedirs(os.path.dirname(_legacy_path), exist_ok=True)
plt.savefig(_legacy_path, dpi=160, bbox_inches="tight")
# Canonical destinations the manuscript references.
for _dest in (os.path.join(_root, "figures", "final"),
              os.path.join(_root, "manuscript", "final")):
    os.makedirs(_dest, exist_ok=True)
    plt.savefig(os.path.join(_dest, "fig1_model_overview.png"),
                dpi=220, bbox_inches="tight")
    plt.savefig(os.path.join(_dest, "fig1_model_overview.pdf"),
                bbox_inches="tight")
plt.close(fig)
print(f"Saved: {_legacy_path} + figures/final/fig1_model_overview.{{png,pdf}}"
      f" + manuscript/final/fig1_model_overview.{{png,pdf}}")
