"""
analysis/validation_io.py
=========================
Structured I/O helpers for validation-stage results.

Writers:
    write_parametric_bootstrap_check_json(d, out_path, ...)
    write_within_trace_holdout_json(d, out_path, ...)
    write_chamber_holdout_summary_csv(rows, out_path)
    write_lodo_summary_csv(rows, out_path)

All carry the standard provenance metadata (schema_version, model_version,
pipeline_version, dataset, chamber, diagnostic_level, run_timestamp_utc,
platform) plus the configurable warning thresholds and explicit disclaimers.
"""
from __future__ import annotations
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.validation import EXPLICIT_DISCLAIMER


SCHEMA_VERSION   = "1"
MODEL_VERSION    = "reduced_v2.1"
PIPELINE_VERSION = "mito_v2.1"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _platform_str() -> str:
    return f"{platform.system()} {platform.release()} python {platform.python_version()}"


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _provenance(dataset: str, chamber: str, diagnostic_level: str
                 ) -> Dict[str, Any]:
    return {
        "schema_version":   SCHEMA_VERSION,
        "model_version":    MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset":          dataset,
        "chamber":          chamber,
        "diagnostic_level": diagnostic_level,
        "run_timestamp_utc": _now_utc(),
        "platform":          _platform_str(),
    }


def write_parametric_bootstrap_check_json(
        result: Dict[str, Any],
        out_path: Path,
        *,
        dataset: str,
        chamber: str = "",
        diagnostic_level: str = "fast",
        configurable_threshold_low: float = 0.80,
        configurable_threshold_high: float = 0.95,
        ) -> Path:
    """Serialise a parametric-bootstrap predictive-check result to JSON.

    Records the configurable warning band used by the agent (default
    [0.80, 0.95]) explicitly so downstream code knows the gate applied
    when interpreting the coverage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cov = float(result.get("parametric_bootstrap_coverage_90",
                            result.get("coverage_90", float("nan"))))
    within_band = bool(configurable_threshold_low <= cov
                        <= configurable_threshold_high) \
                  if np.isfinite(cov) else False

    payload: Dict[str, Any] = {
        "method":           "parametric_bootstrap_predictive_check",
        "metric":           "coverage_90",
        "n_boot":           int(result.get("n_boot", 0)),
        "refit":            bool(result.get("refit", False)),
        "parametric_bootstrap_coverage_90": cov,
        # Legacy alias for older readers
        "coverage_90":      cov,
        "configurable_threshold_low":  float(configurable_threshold_low),
        "configurable_threshold_high": float(configurable_threshold_high),
        "within_configurable_band":    within_band,
        "envelope_t":     np.asarray(result.get("t",      [])).tolist(),
        "envelope_lo90":  np.asarray(result.get("lo90",   [])).tolist(),
        "envelope_hi90":  np.asarray(result.get("hi90",   [])).tolist(),
        "envelope_median": np.asarray(result.get("median", [])).tolist(),
        "o_data":         np.asarray(result.get("o_data", [])).tolist(),
        "o_hat":          np.asarray(result.get("o_hat",  [])).tolist(),
        "explicit_disclaimer": result.get("explicit_disclaimer",
                                            EXPLICIT_DISCLAIMER),
        "interpretation_note": result.get("interpretation_note", ""),
    }
    payload.update(_provenance(dataset, chamber, diagnostic_level))

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return out_path


def write_within_trace_holdout_json(
        result: Dict[str, Any],
        out_path: Path,
        *,
        dataset: str,
        chamber: str = "",
        diagnostic_level: str = "publication",
        ) -> Path:
    """Serialise a within-trace-holdout result to JSON.

    The interpretation note is preserved from the input dict and the
    diagnostic is explicitly framed as 'intervention-phase extrapolation
    diagnostic', NOT biological validation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result.get("params", {})
    payload: Dict[str, Any] = {
        "method":         "within_trace_holdout",
        "framing":        "intervention-phase extrapolation diagnostic",
        "train_frac":     float(result.get("train_frac", 0.7)),
        "n_train":        int(result.get("n_train", 0)),
        "n_test":         int(result.get("n_test", 0)),
        "rmse_train":     float(result["rmse_train"])
                              if result.get("rmse_train") is not None
                              and np.isfinite(result["rmse_train"])
                              else None,
        "rmse_test":      float(result["rmse_test"])
                              if result.get("rmse_test") is not None
                              and np.isfinite(result["rmse_test"])
                              else None,
        "params":         {k: float(v) for k, v in p.items()
                            if isinstance(v, (int, float, np.floating,
                                                np.integer))},
        "alphas":         [float(a) for a in p.get("alphas", [])],
        "refit_based":    bool(result.get("refit_based", True)),
        "t_split":        (float(result["t_split"])
                            if result.get("t_split") is not None else None),
        "events_in_train": list(result.get("events_in_train", [])),
        "events_in_test":  list(result.get("events_in_test", [])),
        "t_full":         np.asarray(result.get("t_full", [])).tolist(),
        "o_data":         np.asarray(result.get("o_data", [])).tolist(),
        "o_pred":         np.asarray(result.get("o_pred", [])).tolist(),
        "interpretation_note": result.get("interpretation_note",
            "Intervention-phase extrapolation diagnostic. "
            "Train and test data are NOT statistically independent."),
        "explicit_disclaimer": (
            "Within-trace holdout is a GENUINE refit-based "
            "INTERVENTION-PHASE EXTRAPOLATION DIAGNOSTIC: parameters are "
            "re-optimised on the training segment only, then used to "
            "predict the held-out segment. It is NOT biological validation "
            "-- the held-out test segment shares chamber, noise process, "
            "and operator with the training segment. rmse_test below "
            "rmse_train is expected when the held-out tail is intrinsically "
            "lower-variance; see events_in_train / events_in_test."
        ),
    }
    payload.update(_provenance(dataset, chamber, diagnostic_level))

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return out_path



def write_chamber_holdout_summary_csv(rows: Sequence[Dict[str, Any]],
                                      out_csv: Path) -> Path:
    """Write Chamber A -> Chamber B technical-replicate transfer summary.

    This CSV intentionally uses the phrase "technical-replicate transfer"
    rather than biological validation. Each row should contain dataset,
    train_chamber, test_chamber, rmse_train, rmse_test, and interpretation.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{
            "dataset": "", "validation_mode": "technical-replicate transfer",
            "status": "not_run", "interpretation":
            "No chamber-transfer diagnostic was run in this execution."
        }]
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def write_lodo_summary_csv(rows: Sequence[Dict[str, Any]],
                           out_csv: Path) -> Path:
    """Write leave-one-dataset-out pooled-transfer diagnostic summary.

    Rows may report actual LODO metrics or an explicit not-run/deferred status.
    LODO is framed as a pooled-transfer diagnostic, not hierarchical Bayesian
    leave-one-out and not biological generalization proof.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{
            "validation_mode": "pooled-transfer diagnostic",
            "status": "not_run_diagnostic_mode",
            "metric": "rmse_held",
            "value": "",
            "interpretation":
            "LODO was deferred in diagnostic mode; run publication mode on real replicate datasets for a meaningful pooled-transfer diagnostic.",
            "limitation":
            "Not hierarchical Bayesian leave-one-out and not proof of biological generalization."
        }]
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def validation_summary_csv(json_paths: Sequence[Path],
                            out_csv: Path) -> Path:
    """Aggregate validation JSONs into a flat per-method CSV."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for jp in json_paths:
        with open(jp) as f:
            d = json.load(f)
        method = d.get("method")
        row = {
            "dataset":          d.get("dataset"),
            "method":           method,
            "diagnostic_level": d.get("diagnostic_level"),
        }
        if method == "parametric_bootstrap_predictive_check":
            row.update({
                "metric":   "coverage_90",
                "value":    d.get("parametric_bootstrap_coverage_90"),
                "within_configurable_band": d.get("within_configurable_band"),
                "n_boot":   d.get("n_boot"),
            })
        elif method == "within_trace_holdout":
            row.update({
                "metric":   "rmse_test",
                "value":    d.get("rmse_test"),
                "rmse_train": d.get("rmse_train"),
                "train_frac": d.get("train_frac"),
            })
        rows.append(row)

    if not rows:
        return out_csv
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out_csv
