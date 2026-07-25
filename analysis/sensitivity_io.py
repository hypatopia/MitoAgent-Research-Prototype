"""
analysis/sensitivity_io.py
==========================
Structured I/O helpers for sensitivity-analysis results.

Writers:
    write_morris_json(d, out_path, ...)
    write_sobol_json(d, out_path, ...)
    write_time_resolved_sobol_npz(d, out_npz_path, ...)
        Time-resolved arrays go into NPZ for size; a small companion
        `.meta.json` is written next to the NPZ for inspection without
        loading the array.
    sensitivity_summary_csv(json_paths, out_csv)
        Flat aggregate CSV across Morris + Sobol JSONs.
"""
from __future__ import annotations
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


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


# ── Morris ───────────────────────────────────────────────────────────────
def write_morris_json(result: Dict[str, Any],
                       out_path: Path,
                       *,
                       dataset: str,
                       chamber: str = "",
                       diagnostic_level: str = "fast",
                       ) -> Dict[str, Any]:
    """Serialise a morris_screening() result dict to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {**_provenance(dataset, chamber, diagnostic_level)}
    payload.update({
        "method":         result.get("method", "morris"),
        "metric":         result.get("metric"),
        "parameter_set":  list(result.get("parameter_set", [])),
        "log_mask":       list(result.get("log_mask", [])),
        "names":          list(result.get("names", [])),
        "mu":             np.asarray(result.get("mu", [])).tolist(),
        "mu_star":        np.asarray(result.get("mu_star", [])).tolist(),
        "sigma":          np.asarray(result.get("sigma", [])).tolist(),
        "mu_star_conf":   np.asarray(result.get("mu_star_conf", [])).tolist(),
        "n_evals":        int(result.get("n_evals", -1)),
        "n_trajectories": int(result.get("n_trajectories", -1)),
        "num_levels":     int(result.get("num_levels", -1)),
        "seed":           int(result.get("seed", 0)),
        "salib_version":  result.get("salib_version"),
        "n_nan_outputs":  int(result.get("n_nan_outputs", 0)),
    })
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return payload


# ── Sobol scalar (AUC_OCR / final_O2) ────────────────────────────────────
def write_sobol_json(result: Dict[str, Any],
                      out_path: Path,
                      *,
                      dataset: str,
                      chamber: str = "",
                      diagnostic_level: str = "fast",
                      ) -> Dict[str, Any]:
    """Serialise a sobol_indices() result dict to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {**_provenance(dataset, chamber, diagnostic_level)}
    payload.update({
        "method":         result.get("method", "sobol"),
        "metric":         result.get("metric"),
        "parameter_set":  list(result.get("parameter_set", [])),
        "log_mask":       list(result.get("log_mask", [])),
        "names":          list(result.get("names", [])),
        "S1":             np.asarray(result.get("S1", [])).tolist(),
        "S1_conf":        np.asarray(result.get("S1_conf", [])).tolist(),
        "ST":             np.asarray(result.get("ST", [])).tolist(),
        "ST_conf":        np.asarray(result.get("ST_conf", [])).tolist(),
        "n_evals":        int(result.get("n_evals", -1)),
        "N_base":         int(result.get("N_base", -1)),
        "seed":           int(result.get("seed", 0)),
        "salib_version":  result.get("salib_version"),
        "n_nan_outputs":  int(result.get("n_nan_outputs", 0)),
        "interpretation_note": result.get("interpretation_note", ""),
    })
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return payload


# ── Time-resolved Sobol ──────────────────────────────────────────────────
def write_time_resolved_sobol_npz(result: Dict[str, Any],
                                    out_npz_path: Path,
                                    *,
                                    dataset: str,
                                    chamber: str = "",
                                    diagnostic_level: str = "fast",
                                    ) -> Path:
    """Write time-resolved Sobol arrays to NPZ + small companion meta JSON.

    Why split: NPZ is the right format for the (n_params, n_t) S1_t / ST_t
    arrays (size). The meta JSON next to it carries the parameter names,
    seeds, sample size, variance-degenerate mask, etc., so users can
    inspect provenance without np.load-ing the arrays.
    """
    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz_path,
        t_grid=np.asarray(result.get("t_grid", [])),
        S1_t=np.asarray(result.get("S1_t", [])),
        ST_t=np.asarray(result.get("ST_t", [])),
        output_variance=np.asarray(result.get("output_variance", [])),
        variance_degenerate_mask=np.asarray(
            result.get("variance_degenerate_mask", []), dtype=bool),
    )
    meta_path = out_npz_path.with_suffix(".meta.json")
    meta: Dict[str, Any] = {**_provenance(dataset, chamber, diagnostic_level)}
    meta.update({
        "method":         result.get("method", "time_resolved_sobol"),
        "metric":         result.get("metric", "OCR(t)"),
        "parameter_set":  list(result.get("parameter_set", [])),
        "log_mask":       list(result.get("log_mask", [])),
        "names":          list(result.get("names", [])),
        "n_evals":        int(result.get("n_evals", -1)),
        "N_base":         int(result.get("N_base", -1)),
        "n_t_eval":       int(result.get("n_t_eval", -1)),
        "seed":           int(result.get("seed", 0)),
        "var_degen_tol":  float(result.get("var_degen_tol", 1e-12)),
        "salib_version":  result.get("salib_version"),
        "n_nan_outputs":  int(result.get("n_nan_outputs", 0)),
        "interpretation_note": result.get("interpretation_note", ""),
        "npz_filename":   out_npz_path.name,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_to_jsonable)
    return out_npz_path


# ── Summary CSV ──────────────────────────────────────────────────────────
def sensitivity_summary_csv(json_paths: Sequence[Path],
                              out_csv: Path) -> Path:
    """Aggregate Morris + Sobol JSONs into a flat CSV.

    Each row reports per-parameter sensitivity values for one method on one
    dataset. Time-resolved Sobol is NOT included (it is a vector quantity;
    consult the NPZ + meta JSON directly).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for jp in json_paths:
        with open(jp) as f:
            d = json.load(f)
        method  = d.get("method")
        metric  = d.get("metric")
        dataset = d.get("dataset")
        diag    = d.get("diagnostic_level")
        n_evals = d.get("n_evals")
        names   = d.get("names", [])
        if method == "morris":
            mu_star = d.get("mu_star", [])
            sigma   = d.get("sigma", [])
            for nm, ms, sg in zip(names, mu_star, sigma):
                rows.append({
                    "dataset":          dataset,
                    "method":           method,
                    "metric":           metric,
                    "parameter":        nm,
                    "primary_index":    ms,   # mu_star
                    "secondary_index":  sg,   # sigma
                    "diagnostic_level": diag,
                    "n_evals":          n_evals,
                })
        elif method == "sobol":
            S1 = d.get("S1", [])
            ST = d.get("ST", [])
            for nm, s1v, stv in zip(names, S1, ST):
                rows.append({
                    "dataset":          dataset,
                    "method":           method,
                    "metric":           metric,
                    "parameter":        nm,
                    "primary_index":    s1v,  # S1
                    "secondary_index":  stv,  # ST (NOT additive!)
                    "diagnostic_level": diag,
                    "n_evals":          n_evals,
                })
    if not rows:
        return out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv


# ── Interpretation summary for hypothesis/design-guidance layers ─────────
def write_sensitivity_interpretation_json(
    morris_payload: Dict[str, Any],
    sobol_payload: Dict[str, Any],
    out_path: Path,
    *,
    time_resolved_meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a cautious interpretation summary for sensitivity analysis.

    This file is intentionally textual/structured rather than inferential:
    it ranks parameters by diagnostic sensitivity and records caveats for
    downstream MitoAgent reports, Help/FAQ text, and manuscript tables.
    It MUST NOT be used as an identifiability verdict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(morris_payload.get("names", []))
    mu_star = list(morris_payload.get("mu_star", []))
    S1 = list(sobol_payload.get("S1", []))
    ST = list(sobol_payload.get("ST", []))
    rows = []
    for nm in names:
        i_m = names.index(nm)
        if nm in sobol_payload.get("names", []):
            i_s = list(sobol_payload.get("names", [])).index(nm)
            s1 = float(S1[i_s]) if i_s < len(S1) else None
            st = float(ST[i_s]) if i_s < len(ST) else None
        else:
            s1 = None; st = None
        rows.append({
            "parameter": nm,
            "morris_mu_star": float(mu_star[i_m]) if i_m < len(mu_star) else None,
            "sobol_S1": s1,
            "sobol_ST": st,
        })
    rows_sorted = sorted(rows, key=lambda r: (r["morris_mu_star"] is None, -(r["morris_mu_star"] or 0.0)))
    top = rows_sorted[:3]
    time_meta = None
    if time_resolved_meta_path and Path(time_resolved_meta_path).exists():
        with open(time_resolved_meta_path) as f:
            time_meta = json.load(f)
    caveats = [
        "Sensitivity analysis complements identifiability but does not prove parameter identifiability.",
        "Sobol total-order indices include interactions and are not additive/exclusive variance fractions.",
        "Diagnostic-mode sample sizes are intended for workflow verification, not stable publication-level estimates.",
    ]
    if time_meta:
        caveats.append(
            "Time points flagged as variance-degenerate in the time-resolved Sobol metadata are not interpreted."
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": morris_payload.get("dataset"),
        "chamber": morris_payload.get("chamber"),
        "run_timestamp_utc": _now_utc(),
        "analysis_type": "sensitivity_interpretation",
        "diagnostic_level": morris_payload.get("diagnostic_level"),
        "metric": "AUC_OCR plus optional OCR(t)",
        "top_morris_parameters": top,
        "all_parameter_sensitivity": rows_sorted,
        "time_resolved_sobol_available": bool(time_meta),
        "time_resolved_sobol_meta": time_meta,
        "hypothesis_support_role": "Sensitivity can prioritize which model features and protocol phases are informative, but generated hypotheses remain candidate hypotheses requiring experimental confirmation.",
        "experimental_design_role": "High sensitivity paired with weak identifiability motivates additional measurements or protocol refinements to reduce uncertainty.",
        "caveats": caveats,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return payload
