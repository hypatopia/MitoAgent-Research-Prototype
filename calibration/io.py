"""
calibration/io.py
=================
Structured I/O helpers for calibration results.

Each writer emits a JSON file carrying full provenance metadata so the
result can be cited definitively in a manuscript and reproduced from a
clean checkout. The schema is stable across runs (`schema_version = "1"`).
"""
from __future__ import annotations
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.reduced_model import CORE_PARAM_ORDER, PARAM_BOUNDS

SCHEMA_VERSION   = "1"
MODEL_VERSION    = "reduced_v2.1"
PIPELINE_VERSION = "mito_v2.1"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _platform_str() -> str:
    return (f"{platform.system()} {platform.release()} "
            f"python {platform.python_version()}")


def _bounds_dict(n_fccp: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        k: [float(PARAM_BOUNDS[k][0]), float(PARAM_BOUNDS[k][1])]
        for k in CORE_PARAM_ORDER
    }
    a_lo, a_hi = PARAM_BOUNDS["alpha"]
    for j in range(n_fccp):
        out[f"alpha_{j+1}"] = [float(a_lo), float(a_hi)]
    return out


def write_calibration_result_json(
        result: Any,
        out_path: Path | str,
        *,
        dataset: str,
        chamber: str,
        diagnostic_level: str = "fast",
        rmse_full_trace: Optional[float] = None,
        extra_warnings: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
    """Serialise a `CalibrationResult` to a structured JSON file.

    Parameters
    ----------
    result : calibration.calibrate.CalibrationResult
    out_path : path to write
    dataset, chamber : identifiers for the run
    diagnostic_level : "fast" | "publication"
    rmse_full_trace : optional RMSE on the un-truncated trace (computed
        externally because it requires the raw chamber data)
    extra_warnings : additional preprocess / pipeline warnings to merge
        into the result.warnings list

    Returns the payload dict (also written to `out_path`).
    """
    p = dict(getattr(result, "params", {}) or {})
    alphas = list(p.pop("alphas", []) or [])
    sigma_from_params = p.pop("sigma_obs", None)

    n_fccp = len(alphas)
    bounds = _bounds_dict(n_fccp)

    warnings: List[str] = list(getattr(result, "warnings", []) or [])
    if extra_warnings:
        warnings.extend(list(extra_warnings))

    payload: Dict[str, Any] = {
        "schema_version":   SCHEMA_VERSION,
        "model_version":    MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset":          str(dataset),
        "chamber":          str(chamber),
        "method":           str(getattr(result, "method", "unknown")),
        "objective_type":   str(getattr(result, "objective_type",
                                          "SSE_with_post_hoc_sigma")),
        "rmse_calib":       _f(getattr(result, "rmse_calib", None)),
        "rmse_full_trace":  _f(rmse_full_trace
                                if rmse_full_trace is not None
                                else getattr(result, "rmse_full_trace", None)),
        "n_data":           int(getattr(result, "n_data", 0)),
        "n_eval":           int(getattr(result, "n_eval", -1)),
        "seed":             int(getattr(result, "seed", 0)),
        "diagnostic_level": str(diagnostic_level),
        "params":           {k: float(p[k]) for k in CORE_PARAM_ORDER
                              if k in p},
        "alphas":           [float(a) for a in alphas],
        "sigma_obs":        _f(getattr(result, "sigma_obs",
                                         sigma_from_params)),
        "sigma_estimation_method": str(getattr(
            result, "sigma_estimation_method",
            "post-hoc residual RMS (deterministic SSE)")),
        "parameter_bounds": bounds,
        "optimiser_settings": dict(getattr(result, "optimiser_settings",
                                            {}) or {}),
        "objective_value":  _f(getattr(result, "objective", None)),
        "warnings":         warnings,
        "run_timestamp_utc": _now_utc(),
        "platform":          _platform_str(),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return payload


def calibration_summary_csv(
        payloads: Iterable[Dict[str, Any]],
        out_path: Path | str,
        ) -> None:
    """Write a flat CSV summary across a collection of calibration JSONs."""
    rows: List[Dict[str, Any]] = []
    for pl in payloads:
        rows.append({
            "dataset":          pl.get("dataset", ""),
            "chamber":          pl.get("chamber", ""),
            "method":           pl.get("method", ""),
            "objective_type":   pl.get("objective_type", ""),
            "diagnostic_level": pl.get("diagnostic_level", ""),
            "rmse_calib":       pl.get("rmse_calib", ""),
            "rmse_full_trace":  pl.get("rmse_full_trace", ""),
            "n_data":           pl.get("n_data", ""),
            "n_eval":           pl.get("n_eval", ""),
            "seed":             pl.get("seed", ""),
            "sigma_obs":        pl.get("sigma_obs", ""),
            "n_warnings":       len(pl.get("warnings", []) or []),
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── helpers ──────────────────────────────────────────────────────────────
def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Cannot serialise {type(o)}")
