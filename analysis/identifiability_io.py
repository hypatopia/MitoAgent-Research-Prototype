"""
analysis/identifiability_io.py
==============================
Structured I/O helpers for FIM and profile-likelihood results.

Each writer emits a JSON file carrying full provenance metadata. The schema
is stable across runs (`schema_version = "1"`).

Functions:
    write_fim_json(rep, out_path, dataset, chamber, diagnostic_level)
    write_profiles_json(profiles, out_path, dataset, chamber, diagnostic_level)
    identifiability_summary_csv(profile_jsons, out_csv)
"""
from __future__ import annotations
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from analysis.identifiability import (
    FIMReport, ProfileLikelihoodReport, EIG_CLIP_FLOOR,
)


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


# ── FIM ──────────────────────────────────────────────────────────────────
def write_fim_json(rep: FIMReport,
                    out_path: Path,
                    *,
                    dataset: str,
                    chamber: str = "",
                    diagnostic_level: str = "fast",
                    ) -> Dict[str, Any]:
    """Write a FIMReport to JSON with full provenance.

    The payload distinguishes RAW vs CLIPPED eigenvalue spectra and
    condition numbers. The clipping floor (`eig_clip_floor`) is recorded
    explicitly so the values can be reproduced.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version":   SCHEMA_VERSION,
        "model_version":    MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset":          dataset,
        "chamber":          chamber,
        "diagnostic_level": diagnostic_level,
        "method":           "fisher_information",
        "param_names":      list(rep.param_names),
        "theta_hat":        np.asarray(rep.theta_hat).tolist(),
        "sigma_obs_used":   float(rep.sigma_obs),
        "eigvals_raw":      np.asarray(rep.eigvals_raw).tolist(),
        "eigvals_clipped":  np.asarray(rep.eigvals_clipped).tolist(),
        "condition_raw":    float(rep.condition_raw),
        "condition_clipped": float(rep.condition_clipped),
        "eig_clip_floor":   float(EIG_CLIP_FLOOR),
        "condition_raw_note": (
            "raw condition number is the diagnostic value: max(eig) divided "
            "by the actual smallest eigenvalue (clamped only to avoid "
            "division by zero). The clipped condition number floors tiny "
            "or numerically-negative eigenvalues at eig_clip_floor and is "
            "more appropriate for inverse-FIM-style discussion."
        ),
        "warnings":         list(rep.warnings),
        "run_timestamp_utc": _now_utc(),
        "platform":          _platform_str(),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return payload


# ── Profile likelihoods ──────────────────────────────────────────────────
def _profile_to_dict(rep: ProfileLikelihoodReport) -> Dict[str, Any]:
    """Serialise one ProfileLikelihoodReport to a JSON-compatible dict."""
    return {
        "param_name":          rep.param_name,
        "map_value":           float(rep.map_value),
        "map_nll":             float(rep.map_nll),
        "profile_min_value":   float(rep.profile_min_value),
        "profile_min_nll":     float(rep.profile_min_nll),
        "theta_grid":          np.asarray(rep.theta_grid).tolist(),
        "nll_grid":            np.asarray(rep.nll_grid).tolist(),
        "delta_nll":           np.asarray(rep.delta_nll).tolist(),
        "optimizer_success":   [bool(b) for b in rep.optimizer_success],
        "n_optimizer_failures": int(rep.n_optimizer_failures),
        "ci_low":              (float(rep.ci_low)
                                  if rep.ci_low is not None else None),
        "ci_high":             (float(rep.ci_high)
                                  if rep.ci_high is not None else None),
        "identified_left":     bool(rep.identified_left),
        "identified_right":    bool(rep.identified_right),
        "practical_id":        rep.practical_id,
        "map_inside_ci":       (bool(rep.map_inside_ci)
                                  if rep.map_inside_ci is not None else None),
        "notes":               rep.notes,
    }


def write_profiles_json(profiles: Dict[str, ProfileLikelihoodReport],
                         out_path: Path,
                         *,
                         dataset: str,
                         chamber: str = "",
                         diagnostic_level: str = "publication",
                         n_grid_used: Optional[int] = None,
                         ) -> Dict[str, Any]:
    """Write a dict {param_name: ProfileLikelihoodReport} to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version":   SCHEMA_VERSION,
        "model_version":    MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset":          dataset,
        "chamber":          chamber,
        "diagnostic_level": diagnostic_level,
        "method":           "profile_likelihood",
        "n_grid_used":      n_grid_used,
        "chi2_threshold":   3.841,
        "profiles":         {nm: _profile_to_dict(rep)
                              for nm, rep in profiles.items()},
        "interpretation_note": (
            "Verdicts: 'identifiable' = CI bounded on both sides after re-optimisation; "
            "'weakly identified' = bounded in a fast fixed-other diagnostic only; "
            "'one-sided' = CI bounded on only one side (the parameter is "
            "constrained from one direction only); 'non-identifiable' = "
            "profile is essentially flat within the explored grid; "
            "'unresolved' = too many optimiser failures to draw a "
            "conclusion. When map_inside_ci is False, the profile "
            "re-optimisation found a strictly better fit than the supplied "
            "MAP — both values are reported."
        ),
        "run_timestamp_utc": _now_utc(),
        "platform":          _platform_str(),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    return payload


# ── Summary CSV ──────────────────────────────────────────────────────────
def identifiability_summary_csv(profile_json_paths: Sequence[Path],
                                  out_csv: Path) -> Path:
    """Aggregate per-dataset profile-likelihood JSONs into a flat CSV."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for jp in profile_json_paths:
        with open(jp) as f:
            d = json.load(f)
        dataset = d.get("dataset")
        diag = d.get("diagnostic_level")
        for nm, prof in d.get("profiles", {}).items():
            rows.append({
                "dataset":               dataset,
                "diagnostic_level":      diag,
                "parameter":             nm,
                "map_value":             prof.get("map_value"),
                "profile_min_value":     prof.get("profile_min_value"),
                "ci_low":                prof.get("ci_low"),
                "ci_high":               prof.get("ci_high"),
                "verdict":               prof.get("practical_id"),
                "map_inside_ci":         prof.get("map_inside_ci"),
                "n_optimizer_failures":  prof.get("n_optimizer_failures"),
            })
    if not rows:
        return out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv


# ── Interpretability flags ───────────────────────────────────────────────
def _flag_from_verdict(verdict: str, n_fail: int = 0) -> tuple[str, str]:
    """Map profile-likelihood verdicts to user-facing interpretability flags."""
    v = (verdict or "").lower()
    if n_fail and v == "unresolved":
        return "optimizer failure", "Optimizer failures prevent a reliable profile-likelihood verdict. Do not interpret this parameter as a biological endpoint."
    if v == "identifiable":
        return "interpretable", "Profile likelihood is bounded on both sides. Interpret only within the OCR-only model scope."
    if v == "weakly identified":
        return "weak", "Diagnostic scan is bounded but not publication-grade profile likelihood; treat as weak until full profiling confirms it."
    if v == "one-sided":
        return "one-sided", "Profile likelihood constrains the parameter from only one side; avoid using it as a standalone biological endpoint."
    if v in ("non-identifiable", "flat"):
        return "flat", "Profile likelihood is flat or unbounded over the explored range; OCR-only data do not support direct interpretation."
    return "unresolved", "Profile likelihood did not support a reliable verdict; rerun publication-grade profiles or add observables."


def write_parameter_interpretability_flags(profile_json_path: Path,
                                           fim_json_path: Path,
                                           out_csv: Path) -> Path:
    """Write per-parameter interpretability flags using profile verdicts.

    Parameters absent from the profile JSON but present in the FIM JSON, such
    as dataset-specific alpha parameters when only core parameters are profiled,
    are flagged as ``weak`` because FIM alone is diagnostic and local.
    """
    profile_json_path = Path(profile_json_path)
    fim_json_path = Path(fim_json_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    profiles: Dict[str, Any] = {}
    dataset = ""
    diagnostic_level = ""
    if profile_json_path.exists():
        with open(profile_json_path) as f:
            pdat = json.load(f)
        profiles = pdat.get("profiles", {}) or {}
        dataset = pdat.get("dataset", "")
        diagnostic_level = pdat.get("diagnostic_level", "")
    fim_names: List[str] = []
    theta_hat: List[Any] = []
    fim_condition = None
    if fim_json_path.exists():
        with open(fim_json_path) as f:
            fdat = json.load(f)
        fim_names = list(fdat.get("param_names", []) or [])
        theta_hat = list(fdat.get("theta_hat", []) or [])
        fim_condition = fdat.get("condition_raw")
        dataset = dataset or fdat.get("dataset", "")
        diagnostic_level = diagnostic_level or fdat.get("diagnostic_level", "")
    rows: List[Dict[str, Any]] = []
    names = fim_names or list(profiles.keys())
    value_by_name = {nm: theta_hat[i] if i < len(theta_hat) else None for i, nm in enumerate(fim_names)}
    for nm in names:
        pr = profiles.get(nm)
        if pr:
            flag, caveat = _flag_from_verdict(pr.get("practical_id"), int(pr.get("n_optimizer_failures") or 0))
            rows.append({
                "dataset": dataset,
                "diagnostic_level": diagnostic_level,
                "parameter": nm,
                "map_value": pr.get("map_value", value_by_name.get(nm)),
                "profile_min_value": pr.get("profile_min_value"),
                "ci_low": pr.get("ci_low"),
                "ci_high": pr.get("ci_high"),
                "profile_verdict": pr.get("practical_id"),
                "interpretability_flag": flag,
                "map_inside_ci": pr.get("map_inside_ci"),
                "n_optimizer_failures": pr.get("n_optimizer_failures"),
                "fim_condition_raw": fim_condition,
                "caveat": caveat,
            })
        else:
            rows.append({
                "dataset": dataset,
                "diagnostic_level": diagnostic_level,
                "parameter": nm,
                "map_value": value_by_name.get(nm),
                "profile_min_value": None,
                "ci_low": None,
                "ci_high": None,
                "profile_verdict": "not_profiled",
                "interpretability_flag": "weak",
                "map_inside_ci": None,
                "n_optimizer_failures": None,
                "fim_condition_raw": fim_condition,
                "caveat": "Only local FIM information is available for this parameter in this run. Treat as weak until profile likelihood is run.",
            })
    if not rows:
        out_csv.write_text("")
        return out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return out_csv
