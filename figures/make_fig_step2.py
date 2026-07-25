"""Figure for Step 2: numerical stability evidence.

Three panels:
  (a) Jacobian spectrum |Re(λ)| over the trace for one protocol
      (proves the system is mildly stiff, justifies LSODA, not catastrophic)
  (b) OCR(t) at three solver tolerances (proves robust to tolerance choice)
  (c) Parameter-sweep convergence histogram across all three protocols
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from core.reduced_model import simulate, Protocol, DEFAULT_PARAMS, PARAM_BOUNDS, CORE_PARAM_ORDER
from core.diagnostics import stiffness_analysis, parameter_robustness_sweep

PROTOCOLS = {
    "I":  Protocol(t_oligo=300, t_fccp=[480], t_inhibit=660, t_end=826, t_start=210),
    "II": Protocol(t_oligo=300, t_fccp=[480, 600], t_inhibit=660, t_end=686, t_start=210),
    "III":Protocol(t_oligo=300, t_fccp=[480, 540, 600, 660], t_inhibit=720, t_end=954, t_start=210),
}

# Default parameters per protocol
def base_params(proto):
    p = dict(DEFAULT_PARAMS)
    p["alphas"] = [1.0] * len(proto.t_fccp)
    return p

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# ─── (a) Jacobian spectrum over time, dataset_I ──────────────────────────
ax = axes[0]
proto = PROTOCOLS["I"]; p = base_params(proto)
max_eig, ratio, eig_arr = stiffness_analysis(p, proto, n_samples=80)
t_grid = np.linspace(proto.t_start, proto.t_end, eig_arr.shape[0])
labels = [r"$|\mathrm{Re}(\lambda_1)|$", r"$|\mathrm{Re}(\lambda_2)|$",
          r"$|\mathrm{Re}(\lambda_3)|$"]
for k in range(3):
    ax.semilogy(t_grid, eig_arr[:, k] + 1e-15, lw=1.5, label=labels[k])
for tj, c, name in [(proto.t_oligo, "C1", "Oligo"),
                    (proto.t_fccp[0], "C2", "FCCP"),
                    (proto.t_inhibit, "C3", "Rot/AntA")]:
    ax.axvline(tj, color=c, lw=0.7, ls=":", alpha=0.7)
ax.set_xlabel("time (s)"); ax.set_ylabel(r"$|\mathrm{Re}(\lambda)|$  (s$^{-1}$)")
ax.set_title("(a)  Jacobian spectrum over trace", loc="left", fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, f"max |λ| = {max_eig:.2e} s⁻¹\nstiffness ratio ≈ {ratio:.0f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

# ─── (b) OCR at three solver tolerances ──────────────────────────────────
ax = axes[1]
proto = PROTOCOLS["II"]; p = base_params(proto)
t_eval = np.linspace(proto.t_start, proto.t_end, 600)
tols = [(1e-9, 1e-12, "tight"), (1e-6, 1e-9, "default"),
        (1e-4, 1e-7, "loose")]
for rt, at, lab in tols:
    res = simulate(p, proto, o2_init=170.0, t_eval=t_eval, rtol=rt, atol=at)
    ax.plot(res.t, res.OCR, lw=1.6,
            label=f"{lab}  (rtol={rt:.0e}, atol={at:.0e})")
for tj, c in [(proto.t_oligo, "C1"),
              *((tj, "C2") for tj in proto.t_fccp),
              (proto.t_inhibit, "C3")]:
    ax.axvline(tj, color=c, lw=0.7, ls=":", alpha=0.7)
ax.set_xlabel("time (s)"); ax.set_ylabel("OCR  (nmol/mL/s)")
ax.set_title("(b)  Solver-tolerance robustness", loc="left", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# ─── (c) Parameter-sweep convergence rates per protocol ──────────────────
ax = axes[2]
names = list(PROTOCOLS.keys())
sweeps = []
for n, pr in PROTOCOLS.items():
    s = parameter_robustness_sweep(pr, o2_init=170.0, n_samples=80, seed=7)
    sweeps.append(s)
conv = [100*s["convergence_rate"] for s in sweeps]
finite = [100*s["finite_OCR_rate"] for s in sweeps]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, conv, w, label="Convergence", color="C0", alpha=0.85)
ax.bar(x + w/2, finite, w, label="Finite OCR", color="C2", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"dataset {n}\n({len(PROTOCOLS[n].t_fccp)} FCCP)"
                    for n in names])
ax.set_ylabel("rate (%)")
ax.set_ylim(0, 110)
ax.set_title("(c)  Robustness across parameter bounds (n=80)", loc="left",
             fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
for i, (c, f) in enumerate(zip(conv, finite)):
    ax.text(i - w/2, c + 1.5, f"{c:.0f}%", ha="center", fontsize=8)
    ax.text(i + w/2, f + 1.5, f"{f:.0f}%", ha="center", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_legacy_path = os.path.join(_root, "figures", "fig_step2_stability.png")
os.makedirs(os.path.dirname(_legacy_path), exist_ok=True)
plt.savefig(_legacy_path, dpi=160, bbox_inches="tight")
for _dest in (os.path.join(_root, "figures", "final"),
              os.path.join(_root, "manuscript", "final")):
    os.makedirs(_dest, exist_ok=True)
    plt.savefig(os.path.join(_dest, "fig2_reduced_model.png"),
                dpi=220, bbox_inches="tight")
    plt.savefig(os.path.join(_dest, "fig2_reduced_model.pdf"),
                bbox_inches="tight")
plt.close(fig)
print(f"Saved: {_legacy_path} + figures/final/fig2_reduced_model.{{png,pdf}}"
      f" + manuscript/final/fig2_reduced_model.{{png,pdf}}")
