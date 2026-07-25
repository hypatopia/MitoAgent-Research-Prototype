"""Phase-level OCR/O2 summaries for mitochondrial stress-test traces.

The summaries are deterministic descriptive diagnostics used to support
calibration review and MitoAgent interpretation. Values are within-trace
summaries only; they are not biological replicate statistics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from data_io.loader import ChamberTrace
from core.reduced_model import simulate


@dataclass
class PhaseSummary:
    phase: str
    start_s: float
    end_s: float
    n_samples: int
    observed_slope_nmol_ml_s: Optional[float]
    observed_ocr_nmol_ml_s: Optional[float]
    observed_o2_start_nmol_ml: Optional[float]
    observed_o2_end_nmol_ml: Optional[float]
    fitted_ocr_mean_nmol_ml_s: Optional[float]
    residual_mean_nmol_ml: Optional[float]
    residual_rmse_nmol_ml: Optional[float]
    warning: str


def _finite_float(x: Any) -> Optional[float]:
    try:
        xf = float(x)
    except Exception:
        return None
    if not np.isfinite(xf):
        return None
    return xf


def _linear_slope(t: np.ndarray, o: np.ndarray) -> Optional[float]:
    if len(t) < 3:
        return None
    if float(np.nanmax(t) - np.nanmin(t)) <= 0:
        return None
    try:
        slope = np.polyfit(t - t[0], o, 1)[0]
    except Exception:
        return None
    return _finite_float(slope)


def _phase_windows(ch: ChamberTrace) -> List[tuple[str, float, float]]:
    t0 = float(ch.t_start if ch.t_start is not None else ch.t[0])
    tend = float(ch.t_end if ch.t_end is not None else ch.t[-1])
    tol = ch.t_oligo
    tinhib = ch.t_inhibit
    fccps = list(ch.t_fccp or [])
    windows: List[tuple[str, float, float]] = []
    if tol is not None:
        windows.append(("basal", t0, float(tol)))
    if tol is not None and fccps:
        windows.append(("oligomycin", float(tol), float(fccps[0])))
    if fccps and tinhib is not None:
        windows.append(("FCCP/uncoupled", float(fccps[0]), float(tinhib)))
    if tinhib is not None:
        windows.append(("Rot/Ant residual", float(tinhib), tend))
    return [(nm, lo, hi) for nm, lo, hi in windows if hi > lo]


def compute_phase_summary(ch: ChamberTrace, params: Optional[dict] = None) -> Dict[str, Any]:
    """Return phase-level descriptive summaries for a chamber trace.

    If fitted params are supplied, fitted OCR/residual summaries are included.
    The observed OCR is approximated as the negative local oxygen slope, so it
    is a descriptive derivative estimate and not a separate measured variable.
    """
    t = np.asarray(ch.t, dtype=float)
    o = np.asarray(ch.o, dtype=float)
    sim = None
    if params is not None:
        try:
            proto = ch.to_protocol()
            sim = simulate(params, proto, o2_init=float(o[0]), t_eval=t)
            if not sim.converged:
                sim = None
        except Exception:
            sim = None

    phases: List[PhaseSummary] = []
    warnings: List[str] = []
    for name, lo, hi in _phase_windows(ch):
        m = (t >= lo) & (t < hi)
        n = int(np.sum(m))
        warn = "stable"
        slope = None
        obs_ocr = None
        start_o = None
        end_o = None
        fit_ocr = None
        res_mean = None
        res_rmse = None
        if n < 3:
            warn = "too few samples for slope/OCR summary"
        else:
            tt = t[m]
            oo = o[m]
            slope = _linear_slope(tt, oo)
            obs_ocr = _finite_float(-slope) if slope is not None else None
            start_o = _finite_float(oo[0])
            end_o = _finite_float(oo[-1])
            if obs_ocr is not None and obs_ocr < -1e-6:
                warn = "oxygen increasing in this phase; inspect trace/event parsing"
            if name == "FCCP/uncoupled" and n < 10:
                warn = "FCCP window has limited samples; plateau may be unclear"
            if sim is not None:
                fit_ocr = _finite_float(np.mean(sim.OCR[m]))
                rr = sim.o[m] - oo
                res_mean = _finite_float(np.mean(rr))
                res_rmse = _finite_float(np.sqrt(np.mean(rr**2)))
        if warn != "stable":
            warnings.append(f"{name}: {warn}")
        phases.append(PhaseSummary(
            phase=name,
            start_s=float(lo),
            end_s=float(hi),
            n_samples=n,
            observed_slope_nmol_ml_s=slope,
            observed_ocr_nmol_ml_s=obs_ocr,
            observed_o2_start_nmol_ml=start_o,
            observed_o2_end_nmol_ml=end_o,
            fitted_ocr_mean_nmol_ml_s=fit_ocr,
            residual_mean_nmol_ml=res_mean,
            residual_rmse_nmol_ml=res_rmse,
            warning=warn,
        ))

    return {
        "analysis_type": "phase_level_ocr_summary",
        "model": "reduced_ocr_informed_v2",
        "chamber": ch.label,
        "data_type": ch.metadata.get("data_type", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phases": [asdict(p) for p in phases],
        "warnings": warnings,
        "terminology_note": "Observed OCR is approximated from within-trace oxygen slope; sigma/noise terms are within-trace observational summaries, not biological variability.",
    }


def write_phase_summary(summary: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
