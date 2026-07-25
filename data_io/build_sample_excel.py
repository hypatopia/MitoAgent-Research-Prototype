"""
data_io/build_sample_excel.py
=============================
Build REPRESENTATIVE Oroboros-style Excel exports generated to test parser
behaviour. These files are NOT real Oroboros measurements and DO NOT prove
compatibility with all real Oroboros export variants. They demonstrate the
parser pipeline end-to-end with one specific column / event-label layout.

Replace the bundled files with real measured exports before drawing any
biological conclusions.

Output: data_samples/dataset_I.xlsx, dataset_II.xlsx, dataset_III.xlsx
Schema:
    column "Time [s]"          : seconds
    column "O2 conc.A [nmol/mL]"
    column "O2 conc.B [nmol/mL]"
    column "Event"             : sparse string column with intervention labels
                                 at injection rows (start, oligomycin, FCCP_*,
                                 rotenone+antimycin, end)
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_io.loader import SAMPLE_EVENTS

DATA_NPY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "mito_agent", "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_samples")


def make_chamber_b(t: np.ndarray, o_a: np.ndarray, seed: int) -> np.ndarray:
    """Create a Chamber-B trace by perturbing Chamber A with realistic
    chamber-to-chamber differences:
      - small constant offset (~0-2 nmol/mL)
      - mild gain difference (~5%)
      - independent noise of similar magnitude
    This is a synthetic technical-replicate-style perturbation used only to
    exercise the parser and transfer-check pipeline; it is not biological variability.
    """
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-1.5, 1.5)
    gain   = 1.0 + rng.uniform(-0.05, 0.05)
    noise  = rng.normal(0.0, 0.25, size=o_a.shape)
    return np.clip(offset + gain * o_a + noise, 0.0, None)


def build_one(name: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas + openpyxl required: pip install pandas openpyxl")

    t = np.load(os.path.join(DATA_NPY, f"{name}_t.npy"))
    o_a = np.load(os.path.join(DATA_NPY, f"{name}_o.npy"))
    seed = {"dataset_I": 101, "dataset_II": 202, "dataset_III": 303}[name]
    o_b = make_chamber_b(t, o_a, seed=seed)

    events = SAMPLE_EVENTS[name]
    label_col = [""] * len(t)
    def _set(t_target: float, lab: str):
        idx = int(np.argmin(np.abs(t - t_target)))
        label_col[idx] = lab

    if events.get("t_start") is not None:    _set(events["t_start"], "start")
    if events.get("t_oligo") is not None:    _set(events["t_oligo"], "oligomycin")
    for j, tj in enumerate(events["t_fccp"]):
        _set(tj, f"FCCP_{j+1}")
    if events.get("t_inhibit") is not None:  _set(events["t_inhibit"],
                                                  "rotenone+antimycin")
    if events.get("t_end") is not None:      _set(events["t_end"], "end")

    df = pd.DataFrame({
        "Time [s]":             t,
        "O2 conc.A [nmol/mL]":  o_a,
        "O2 conc.B [nmol/mL]":  o_b,
        "Event":                label_col,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
    df.to_excel(out_path, index=False)
    return out_path


if __name__ == "__main__":
    for name in ["dataset_I", "dataset_II", "dataset_III"]:
        path = build_one(name)
        print(f"Built {path}")


def build_demo_files(out_dir):
    """Build all bundled demo files into `out_dir`. Best-effort: returns the
    list of files actually built. Skips datasets whose source .npy traces
    cannot be located.
    """
    import os as _os
    built = []
    for name in ["dataset_I", "dataset_II", "dataset_III"]:
        try:
            built.append(build_one(name))
        except Exception as e:  # pragma: no cover
            print(f"  could not build {name}: {e!r}")
    return built
