"""
run_all.py
==========
Reproducible end-to-end pipeline for mito_v2.

Stages
------
1.  ensure output folders
2.  ensure demo data (Excel files in data_samples/)
3.  numerical diagnostics (per dataset)
4.  calibration (per dataset)
5.  identifiability (FIM in --fast; FIM + profile likelihoods in --publication)
6.  sensitivity (Morris in --fast; Morris + Sobol AUC + time-resolved Sobol
    in --publication)
7.  validation (parametric-bootstrap predictive check; within-trace holdout
    added in --publication)
8.  generate figures
9.  write run summary
10. optionally compile manuscript if LaTeX is available
11. fail-soft logging throughout

Modes
-----
--smoke           : minimal deterministic import/sanity check (NOT scientific)
--fast            : quick-turn development settings (NOT reportable)
--publication     : scientifically defensible budgets; THIS IS THE DEFAULT
--skip-identifiability : omit stage 5
--figures-only    : run only stage 8
--manuscript-only : run only stage 10
--datasets ...    : restrict to a subset of dataset names

IMPORTANT
---------
``--publication`` is the default. You must explicitly pass ``--fast`` to get
the cheap development tier. Only publication-tier outputs are marked
``reportable`` and may legitimately populate manuscript tables/figures. All
numerical budgets live in ``core/run_settings.py`` (single source of truth).
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.run_settings import RunSettings, get_settings

DATASETS = ["dataset_I", "dataset_II", "dataset_III"]
SCHEMA_VERSION = "1"
MODEL_VERSION = "reduced_v2.1"
PIPELINE_VERSION = "mito_v2.1"

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA_SAMPLES = ROOT / "data_samples"
MANUSCRIPT = ROOT / "manuscript"


# ── Helpers ──────────────────────────────────────────────────────────────
LOG: List[str] = []


def _log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _platform_str() -> str:
    return f"{platform.system()} {platform.release()} python {platform.python_version()}"


def _package_versions() -> Dict[str, str]:
    """Capture installed versions of the scientific dependencies.

    Uses importlib.metadata (authoritative) rather than module __version__
    attributes, several of which (e.g. SALib) are absent and previously
    caused provenance to be recorded as the literal string 'unknown'.
    """
    import importlib.metadata as _md
    pkgs = ["numpy", "scipy", "pandas", "matplotlib", "openpyxl",
            "SALib", "pytest"]
    out: Dict[str, str] = {"python": platform.python_version()}
    for name in pkgs:
        try:
            out[name] = _md.version(name)
        except Exception:
            out[name] = "not-installed"
    return out


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_conv)


# ── Stage 1: folders ─────────────────────────────────────────────────────
def stage_folders() -> None:
    for sub in ("diagnostics", "calibration", "identifiability",
                "sensitivity", "validation", "agent_reports",
                "_legacy_archive"):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    (FIGURES / "final").mkdir(parents=True, exist_ok=True)
    _log("[folders]    OK")


# ── Stage 2: demo data ───────────────────────────────────────────────────
def stage_demo_data() -> None:
    missing = [name for name in DATASETS
               if not (DATA_SAMPLES / f"{name}.xlsx").exists()]
    if missing:
        _log(f"[demo data]  MISSING: {missing}; "
             f"replace with real Oroboros exports in data_samples/")
    else:
        _log(f"[demo data]  OK ({len(DATASETS)} datasets present)")


# ── Stage 3: diagnostics ─────────────────────────────────────────────────
def stage_diagnostics(datasets: List[str]) -> Dict[str, Any]:
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from core.diagnostics import detect_instability

    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version":  MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_timestamp_utc": _now_utc(),
        "platform": _platform_str(),
        "datasets": {},
    }
    for name in datasets:
        path = DATA_SAMPLES / f"{name}.xlsx"
        if not path.exists():
            _log(f"[diagnostics] {name}: SKIPPED (no file)")
            continue
        try:
            ds = load_excel(str(path))
            ch_a = ds.chambers[0]
            ch, issues = preprocess(ch_a)
            ok = (len(issues) == 0)
            summary["datasets"][name] = {
                "n_samples": int(len(ch.t)),
                "preprocess_warnings": list(issues),
                "ok": ok,
            }
            _log(f"[diagnostics] {name}: {'OK' if ok else 'WARN'} "
                 f"({len(ch.t)} samples)")
        except Exception as e:
            summary["datasets"][name] = {"error": str(e)}
            _log(f"[diagnostics] {name}: ERROR {e}")

    _json_dump(summary, RESULTS / "diagnostics" / "diagnostics_summary.json")
    return summary


# ── Stage 4: calibration ─────────────────────────────────────────────────
def stage_calibration(datasets: List[str], cfg: RunSettings) -> Dict[str, Any]:
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from calibration.calibrate import calibrate_de
    from calibration.io import (
        write_calibration_result_json, calibration_summary_csv,
    )
    from calibration.phase import compute_phase_summary, write_phase_summary
    from core.reduced_model import simulate

    payloads: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {}
    for name in datasets:
        path = DATA_SAMPLES / f"{name}.xlsx"
        if not path.exists():
            continue
        try:
            ds = load_excel(str(path))
            ch_raw = ds.chambers[0]
            ch, prep_warnings = preprocess(ch_raw)
            proto = ch.to_protocol()
            t_full = np.asarray(ch.t, dtype=float)
            o_full = np.asarray(ch.o, dtype=float)
            # Downsampling is governed entirely by the tier settings. For the
            # publication tier this is a fixed, documented 400-point grid;
            # the fast/smoke tiers use smaller grids. None => full trace.
            nds = cfg.calib_n_downsample
            if nds is not None and len(t_full) > nds:
                idx = np.round(np.linspace(0, len(t_full) - 1,
                                            nds)).astype(int)
                t_data = t_full[idx]
                o_data = o_full[idx]
            else:
                t_data = t_full
                o_data = o_full

            res = calibrate_de(
                t_data, o_data, proto,
                maxiter=cfg.de_maxiter, popsize=cfg.de_popsize,
                polish=cfg.de_polish, seed=0,
            )

            # Full-trace RMSE on raw chamber for reference.
            sim_full = simulate(res.params, proto,
                                 o2_init=float(ch_raw.o[0]),
                                 t_eval=np.asarray(ch_raw.t, dtype=float))
            rmse_full = float(np.sqrt(np.mean(
                (sim_full.o - np.asarray(ch_raw.o, dtype=float))**2)))

            payload = write_calibration_result_json(
                res,
                RESULTS / "calibration" / f"calib_{name}.json",
                dataset=name,
                chamber=ch_raw.label,
                diagnostic_level=cfg.tier,
                rmse_full_trace=rmse_full,
                extra_warnings=list(prep_warnings or []),
            )
            phase_summary = compute_phase_summary(ch_raw, res.params)
            write_phase_summary(
                phase_summary,
                RESULTS / "calibration" / f"phase_summary_{name}.json",
            )

            payloads.append(payload)
            out[name] = payload
            _log(f"[calibration] {name}: RMSE={payload['rmse_calib']:.3f} "
                 f"(full={payload['rmse_full_trace']:.3f}, "
                 f"n_eval={payload['n_eval']}, tier={cfg.tier}; "
                 f"phase summary written)")
        except Exception as e:
            _log(f"[calibration] {name}: ERROR {e}")

    if payloads:
        calibration_summary_csv(
            payloads,
            RESULTS / "calibration" / "calibration_summary.csv")
    return out


# ── Stage 5: identifiability (FIM + profile likelihoods) ────────────────
def stage_identifiability(datasets: List[str], cfg: RunSettings,
                           calib: Dict[str, Any]) -> Dict[str, Any]:
    """FIM diagnostics for every calibrated dataset, plus profile likelihoods.

    Profile semantics depend on the tier:
      * publication tier (`cfg.profile_real == True`): GENUINE profile
        likelihoods via `run_all_profiles` -- all other parameters are
        re-optimised at every grid point, with warm-starting, adaptive
        extension, and multi-start at constrained points.
      * fast/smoke tiers (`cfg.profile_real == False`): cheap
        FIXED-parameter scans via `run_all_fixed_scans`. These are written
        to a *separate* filename (`fixed_scans_<dataset>.json`) and are
        never called "profile likelihoods" anywhere, because they are not.

    Only publication-tier outputs are written to `profiles_<dataset>.json`
    and may populate manuscript Table 3.
    """
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from analysis.identifiability import (
        fisher_information, run_all_profiles, run_all_fixed_scans,
    )
    from analysis.identifiability_io import (
        write_fim_json, write_profiles_json,
        identifiability_summary_csv, write_parameter_interpretability_flags,
    )

    out: Dict[str, Any] = {}
    profile_paths: List[Path] = []
    for target in datasets:
        if target not in calib:
            _log(f"[identifiability] {target}: SKIPPED (no calibration result)")
            continue
        try:
            ds = load_excel(str(DATA_SAMPLES / f"{target}.xlsx"))
            ch, _ = preprocess(ds.chambers[0])
            proto = ch.to_protocol()
            params = dict(calib[target]["params"])
            params["alphas"] = list(calib[target]["alphas"])
            sigma_obs = float(calib[target]["sigma_obs"])
            chamber_label = str(calib[target].get("chamber", ""))

            t_arr = np.asarray(ch.t, dtype=float)
            o_arr = np.asarray(ch.o, dtype=float)

            # Downsample the identifiability time grid per tier settings.
            nds = cfg.identif_n_downsample
            if nds is not None and len(t_arr) > nds:
                idx = np.round(np.linspace(0, len(t_arr) - 1,
                                            nds)).astype(int)
                t_arr = t_arr[idx]
                o_arr = o_arr[idx]

            # FIM (computed in every tier).
            rep = fisher_information(
                params, t_arr, o_arr, proto,
                o2_init=float(o_arr[0]), sigma_obs=sigma_obs,
            )
            write_fim_json(rep,
                            RESULTS / "identifiability" / f"fim_{target}.json",
                            dataset=target, chamber=chamber_label,
                            diagnostic_level=cfg.tier)
            out[target] = {"fim_condition_raw": float(rep.condition_raw),
                            "fim_condition_clipped": float(rep.condition_clipped),
                            "warnings": list(rep.warnings)}

            if cfg.profile_real:
                # GENUINE profile likelihoods: re-optimise all other
                # parameters at every grid point.
                _log(f"[identifiability] {target}: running GENUINE profile "
                     f"likelihoods (n_grid={cfg.profile_n_grid}, "
                     f"re-optimised inner fits, adaptive extension)...")
                profiles = run_all_profiles(
                    params, t_arr, o_arr, proto, o2_init=float(o_arr[0]),
                    n_grid=cfg.profile_n_grid,
                    grid_span_log=cfg.profile_grid_span_log,
                    maxiter=cfg.profile_maxiter,
                    adaptive_extend=cfg.profile_adaptive_extend,
                    n_restarts_constrained=cfg.profile_n_restarts_constrained,
                    verbose=False,
                )
                profile_path = (RESULTS / "identifiability"
                                / f"profiles_{target}.json")
                write_profiles_json(
                    profiles, profile_path,
                    dataset=target, chamber=chamber_label,
                    diagnostic_level=cfg.tier,
                    n_grid_used=cfg.profile_n_grid,
                )
                profile_paths.append(profile_path)
                write_parameter_interpretability_flags(
                    profile_path,
                    RESULTS / "identifiability" / f"fim_{target}.json",
                    RESULTS / "identifiability"
                    / f"parameter_interpretability_flags_{target}.csv")
            else:
                # Cheap FIXED-parameter scans. NOT profile likelihoods.
                # Written to a clearly distinct filename.
                _log(f"[identifiability] {target}: running cheap FIXED-"
                     f"parameter scans (n_grid={cfg.profile_n_grid}; "
                     f"NOT profile likelihoods, {cfg.tier} tier only)...")
                profiles = run_all_fixed_scans(
                    params, t_arr, o_arr, proto, o2_init=float(o_arr[0]),
                    n_grid=cfg.profile_n_grid,
                    grid_span_log=cfg.profile_grid_span_log,
                    sigma_obs=sigma_obs,
                )
                profile_path = (RESULTS / "identifiability"
                                / f"fixed_scans_{target}.json")
                write_profiles_json(
                    profiles, profile_path,
                    dataset=target, chamber=chamber_label,
                    diagnostic_level=cfg.tier,
                    n_grid_used=cfg.profile_n_grid,
                )
                profile_paths.append(profile_path)
                write_parameter_interpretability_flags(
                    profile_path,
                    RESULTS / "identifiability" / f"fim_{target}.json",
                    RESULTS / "identifiability"
                    / f"parameter_interpretability_flags_{target}.csv")

            n_id = sum(1 for p in profiles.values()
                        if p.practical_id == "identifiable")
            n_one = sum(1 for p in profiles.values()
                         if p.practical_id == "one-sided")
            n_flat = sum(1 for p in profiles.values()
                          if p.practical_id == "non-identifiable")
            n_unres = sum(1 for p in profiles.values()
                           if p.practical_id == "unresolved")
            n_weak = sum(1 for p in profiles.values()
                          if p.practical_id == "weakly identified")
            kind = "profiles" if cfg.profile_real else "fixed-scans"
            _log(f"[identifiability] {target}: {kind} done "
                 f"({n_id} identifiable, {n_weak} weak, {n_one} one-sided, "
                 f"{n_flat} flat, {n_unres} unresolved of {len(profiles)})")
            out[target].update({"profiles_path": str(profile_path),
                                "profile_kind":
                                    "genuine_profile_likelihood"
                                    if cfg.profile_real
                                    else "fixed_parameter_scan",
                                "n_profiles": len(profiles),
                                "n_identifiable": n_id,
                                "n_weak": n_weak,
                                "n_one_sided": n_one,
                                "n_flat": n_flat,
                                "n_unresolved": n_unres})
        except Exception as e:
            _log(f"[identifiability] {target}: ERROR {e}")

    # One combined summary CSV across all datasets processed.
    if profile_paths:
        identifiability_summary_csv(
            profile_paths,
            RESULTS / "identifiability" / "identifiability_summary.csv")
    return out


# ── Stage 6: sensitivity ─────────────────────────────────────────────────
def stage_sensitivity(datasets: List[str], cfg: RunSettings,
                       calib: Dict[str, Any]) -> Dict[str, Any]:
    """Sensitivity analysis stage (Morris + Sobol AUC + time-resolved Sobol).

    All sample sizes come from `core/run_settings.py`. The publication tier
    uses Morris N_trajectories=20, Sobol N_base=1024, and time-resolved
    Sobol N_base=256 -- large enough that first-order Sobol indices are
    non-negative and confidence intervals are usable.
    """
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from analysis.sensitivity import (
        morris_screening, sobol_indices, time_resolved_sobol,
    )
    from analysis.sensitivity_io import (
        write_morris_json, write_sobol_json,
        write_time_resolved_sobol_npz, sensitivity_summary_csv,
        write_sensitivity_interpretation_json,
    )

    out: Dict[str, Any] = {}
    for target in datasets:
        if target not in calib:
            _log(f"[sensitivity] {target}: SKIPPED (no calibration result)")
            continue
        try:
            ds = load_excel(str(DATA_SAMPLES / f"{target}.xlsx"))
            ch, _ = preprocess(ds.chambers[0])
            proto = ch.to_protocol()
            chamber_label = str(calib[target].get("chamber", ""))

            json_paths: List[Path] = []

            # Morris screening
            m = morris_screening(proto, o2_init=float(ch.o[0]),
                                  N_trajectories=cfg.morris_trajectories,
                                  seed=0)
            morris_path = RESULTS / "sensitivity" / f"morris_{target}.json"
            write_morris_json(m, morris_path,
                               dataset=target, chamber=chamber_label,
                               diagnostic_level=cfg.tier)
            json_paths.append(morris_path)
            _log(f"[sensitivity] {target}: Morris done "
                 f"(n_eval={m['n_evals']}, params={len(m['names'])}, "
                 f"tier={cfg.tier})")

            # Sobol AUC
            s = sobol_indices(proto, o2_init=float(ch.o[0]),
                               N=cfg.sobol_n_base, seed=0)
            sobol_path = RESULTS / "sensitivity" / f"sobol_auc_{target}.json"
            write_sobol_json(s, sobol_path,
                              dataset=target, chamber=chamber_label,
                              diagnostic_level=cfg.tier)
            json_paths.append(sobol_path)
            _log(f"[sensitivity] {target}: Sobol AUC done "
                 f"(n_eval={s['n_evals']}, N_base={cfg.sobol_n_base})")

            # Time-resolved Sobol
            ts = time_resolved_sobol(proto, o2_init=float(ch.o[0]),
                                       N=cfg.time_resolved_sobol_n_base,
                                       n_t_eval=cfg.time_resolved_sobol_n_t,
                                       seed=0)
            ts_path = (RESULTS / "sensitivity"
                       / f"time_resolved_sobol_{target}.npz")
            write_time_resolved_sobol_npz(ts, ts_path,
                                            dataset=target,
                                            chamber=chamber_label,
                                            diagnostic_level=cfg.tier)
            n_var_degen = int(np.sum(ts["variance_degenerate_mask"]))
            _log(f"[sensitivity] {target}: time-resolved Sobol done "
                 f"(n_eval={ts['n_evals']}, "
                 f"N_base={cfg.time_resolved_sobol_n_base}, "
                 f"{n_var_degen}/{ts['n_t_eval']} time points "
                 "flagged variance-degenerate)")

            # Summary CSV and cautious interpretation file
            sensitivity_summary_csv(
                json_paths,
                RESULTS / "sensitivity" / f"sensitivity_summary_{target}.csv")
            interp_path = (RESULTS / "sensitivity"
                           / f"sensitivity_interpretation_{target}.json")
            write_sensitivity_interpretation_json(
                m, s, interp_path,
                time_resolved_meta_path=ts_path.with_suffix(".meta.json"),
            )

            out[target] = {"n_params": len(m["names"]),
                            "morris_path": str(morris_path),
                            "sobol_path": str(sobol_path),
                            "time_resolved_path": str(ts_path),
                            "interpretation_path": str(interp_path)}
        except Exception as e:
            _log(f"[sensitivity] {target}: ERROR {e}")

    # Combined sensitivity summary across datasets.
    all_json: List[Path] = []
    for target in datasets:
        for nm in (f"morris_{target}.json", f"sobol_auc_{target}.json"):
            p = RESULTS / "sensitivity" / nm
            if p.exists():
                all_json.append(p)
    if all_json:
        from analysis.sensitivity_io import sensitivity_summary_csv as _scsv
        _scsv(all_json, RESULTS / "sensitivity" / "sensitivity_summary.csv")
    return out


# ── Stage 7: validation ──────────────────────────────────────────────────
def stage_validation(datasets: List[str], cfg: RunSettings,
                      calib: Dict[str, Any]) -> Dict[str, Any]:
    """Validation stage.

    All three diagnostics are run for every calibrated dataset:
      * parametric-bootstrap predictive check (n_boot from tier settings);
      * Chamber A -> B technical-replicate transfer (when 2 chambers exist);
      * within-trace intervention-phase holdout.

    The within-trace holdout ALWAYS uses the real refit-based routine
    `analysis.validation.within_trace_holdout`, which re-optimises model
    parameters on the training segment and predicts the held-out segment.
    The previous fast-mode shortcut (splitting residuals of already-fitted
    parameters, which leaked the test segment into the fit and produced the
    impossible rmse_test < rmse_train artefact) has been removed entirely.
    """
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from core.reduced_model import simulate
    from analysis.validation import (
        parametric_bootstrap_predictive_check, within_trace_holdout,
    )
    from analysis.validation_io import (
        write_parametric_bootstrap_check_json,
        write_within_trace_holdout_json,
        validation_summary_csv,
        write_chamber_holdout_summary_csv,
        write_lodo_summary_csv,
    )

    out: Dict[str, Any] = {}
    all_json_paths: List[Path] = []
    chamber_rows: List[Dict[str, Any]] = []
    lodo_rows: List[Dict[str, Any]] = []

    for target in datasets:
        if target not in calib:
            _log(f"[validation] {target}: SKIPPED (no calibration result)")
            continue
        json_paths: List[Path] = []
        try:
            ds = load_excel(str(DATA_SAMPLES / f"{target}.xlsx"))
            ch_a, _ = preprocess(ds.chambers[0])
            proto_a = ch_a.to_protocol()
            params = dict(calib[target]["params"])
            params["alphas"] = list(calib[target]["alphas"])
            params["sigma_obs"] = float(calib[target]["sigma_obs"])
            chamber_label = str(calib[target].get("chamber", ch_a.label))

            # 1) Parametric-bootstrap predictive check
            ppc = parametric_bootstrap_predictive_check(
                np.asarray(ch_a.t, dtype=float),
                np.asarray(ch_a.o, dtype=float),
                proto_a, params_hat=params,
                n_boot=cfg.bootstrap_n_boot, seed=0)
            coverage = float(ppc.get("parametric_bootstrap_coverage_90",
                                       np.nan))
            ppc_path = (RESULTS / "validation"
                        / f"parametric_bootstrap_predictive_check_{target}.json")
            write_parametric_bootstrap_check_json(
                ppc, ppc_path, dataset=target, chamber=chamber_label,
                diagnostic_level=cfg.tier,
            )
            json_paths.append(ppc_path)
            _log(f"[validation] {target}: parametric-bootstrap coverage_90 "
                 f"= {coverage:.1%}  (n_boot={cfg.bootstrap_n_boot})")

            # 2) Chamber A -> B technical-replicate transfer
            if len(ds.chambers) >= 2:
                ch_b, _ = preprocess(ds.chambers[1])
                proto_b = ch_b.to_protocol()
                sim_a = simulate(params, proto_a, o2_init=float(ch_a.o[0]),
                                 r0=params.get("r0"),
                                 t_eval=np.asarray(ch_a.t, float))
                sim_b = simulate(params, proto_b, o2_init=float(ch_b.o[0]),
                                 r0=params.get("r0"),
                                 t_eval=np.asarray(ch_b.t, float))
                rmse_a = (float(np.sqrt(np.mean((sim_a.o - ch_a.o)**2)))
                          if sim_a.converged else float("nan"))
                rmse_b = (float(np.sqrt(np.mean((sim_b.o - ch_b.o)**2)))
                          if sim_b.converged else float("nan"))
                chamber_rows.append({
                    "dataset": target,
                    "validation_mode": "technical-replicate transfer",
                    "train_chamber": ch_a.label,
                    "test_chamber": ch_b.label,
                    "metric": "rmse_test",
                    "rmse_train": rmse_a,
                    "rmse_test": rmse_b,
                    "status": "completed",
                    "interpretation": (
                        "Chamber A fitted parameters predicted Chamber B; "
                        "this is technical-replicate transfer, not "
                        "independent biological validation."),
                    "limitation": (
                        "Same preparation/export family; does not establish "
                        "disease/control or biological generalization."),
                })
                _log(f"[validation] {target}: Chamber A->B technical-transfer "
                     f"rmse_train={rmse_a:.3f}, rmse_test={rmse_b:.3f}")

            # 3) Within-trace intervention-phase holdout -- ALWAYS the real
            #    refit-based routine. No residual-split shortcut.
            t_arr = np.asarray(ch_a.t, dtype=float)
            o_arr = np.asarray(ch_a.o, dtype=float)
            _log(f"[validation] {target}: within-trace holdout "
                 f"(refit-based, DE maxiter={cfg.within_trace_de_maxiter})...")
            wt = within_trace_holdout(
                t_arr, o_arr, proto_a, train_frac=0.7,
                maxiter=cfg.within_trace_de_maxiter,
                popsize=cfg.within_trace_de_popsize,
                seed=0,
            )
            wt_path = (RESULTS / "validation"
                       / f"within_trace_holdout_{target}.json")
            write_within_trace_holdout_json(
                wt, wt_path, dataset=target, chamber=chamber_label,
                diagnostic_level=cfg.tier,
            )
            json_paths.append(wt_path)
            rt = wt.get("rmse_train")
            rs = wt.get("rmse_test")
            rt_s = f"{rt:.3f}" if isinstance(rt, (int, float)) else "nan"
            rs_s = f"{rs:.3f}" if isinstance(rs, (int, float)) else "nan"
            _log(f"[validation] {target}: within-trace holdout done "
                 f"(refit-based; rmse_train={rt_s}, rmse_test={rs_s})")

            # 4) LODO row (deferred -- needs real replicate datasets)
            lodo_rows.append({
                "validation_mode": "pooled-transfer diagnostic",
                "dataset": target,
                "status": "deferred",
                "metric": "rmse_held",
                "value": "",
                "interpretation": (
                    "LODO requires multiple real biological replicate "
                    "datasets and is deferred until those are supplied."),
                "limitation": (
                    "Not hierarchical Bayesian leave-one-out and not proof "
                    "of biological disease/generalization."),
            })

            out[target] = {"coverage_90": coverage,
                           "n_boot": cfg.bootstrap_n_boot,
                           "within_trace_rmse_train": rt,
                           "within_trace_rmse_test": rs}
            all_json_paths.extend(json_paths)
        except Exception as e:
            _log(f"[validation] {target}: ERROR {e}")

    # Combined summary artifacts across all datasets processed.
    if chamber_rows:
        write_chamber_holdout_summary_csv(
            chamber_rows,
            RESULTS / "validation" / "chamber_holdout_summary.csv")
    if lodo_rows:
        write_lodo_summary_csv(
            lodo_rows, RESULTS / "validation" / "lodo_summary.csv")
    if all_json_paths:
        validation_summary_csv(
            all_json_paths,
            RESULTS / "validation" / "validation_summary.csv")
    return out


# ── Stage 8: figures ─────────────────────────────────────────────────────
def stage_figures(force: bool = True) -> int:
    """Regenerate every figure from its script.

    The previous implementation skipped regeneration whenever PNG/PDF assets
    already existed ("existing assets verified"), which meant the shipped
    figures could silently drift from the current code and results. That
    behaviour is removed: by default every `make_fig_step*.py` script is
    actually executed, so figures are always reproducible from a clean
    checkout. A generous per-script timeout guards against a hang.
    """
    n_ok = 0
    n_fail = 0
    figs_dir = ROOT / "figures"
    scripts = sorted(figs_dir.glob("make_fig_step*.py"))
    timeout_s = 300
    for sp in scripts:
        try:
            r = subprocess.run([sys.executable, str(sp)], cwd=str(ROOT),
                               text=True, capture_output=True,
                               timeout=timeout_s)
            if r.returncode == 0:
                n_ok += 1
                _log(f"[figures]    {sp.name}: regenerated")
            else:
                n_fail += 1
                stderr = (r.stderr or r.stdout or "")[-700:]
                _log(f"[figures]    {sp.name}: FAILED (rc={r.returncode})\n"
                     f"{stderr}")
        except subprocess.TimeoutExpired:
            n_fail += 1
            _log(f"[figures]    {sp.name}: TIMEOUT after {timeout_s}s")
        except Exception as e:
            n_fail += 1
            _log(f"[figures]    {sp.name}: ERROR {e}")
    _log(f"[figures]    {n_ok}/{len(scripts)} regenerated"
         + (f", {n_fail} FAILED" if n_fail else ""))
    return n_ok


# ── Stage 9: summary ─────────────────────────────────────────────────────
def stage_summary(diag, calib, identif, sens, valid, cfg: RunSettings) -> None:
    summary = {
        "mode": cfg.tier,
        "reportable": bool(cfg.reportable),
        "run_settings": cfg.as_dict(),
        "schema_version":   SCHEMA_VERSION,
        "model_version":    MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_timestamp_utc": _now_utc(),
        "platform":          _platform_str(),
        "package_versions":  _package_versions(),
        "n_datasets_diagnostics": len(diag.get("datasets", {})),
        "n_datasets_calibrated":  len(calib),
        "identifiability_run":    bool(identif),
        "sensitivity_run":        bool(sens),
        "validation_run":         bool(valid),
        "log": list(LOG),
    }
    _json_dump(summary, RESULTS / "_run_summary.json")


# ── Stage 10: manuscript ─────────────────────────────────────────────────
def stage_manuscript() -> None:
    main_tex = MANUSCRIPT / "main.tex"
    if not main_tex.exists():
        _log("[manuscript] SKIPPED: manuscript/main.tex not present.")
        return

    # CHUNK 2 repair: manuscript/main.tex expects figures under
    # manuscript/final/. The canonical generated assets live under
    # figures/final/, so mirror them before compilation.
    src_final = FIGURES / "final"
    dst_final = MANUSCRIPT / "final"
    if src_final.exists():
        dst_final.mkdir(parents=True, exist_ok=True)
        for fp in src_final.glob("fig*.*"):
            try:
                shutil.copy2(fp, dst_final / fp.name)
            except Exception as e:
                _log(f"[manuscript] WARN: could not copy {fp.name}: {e}")

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        _log("[manuscript] SKIPPED: pdflatex not found in PATH.")
        return
    try:
        # Run twice for cross-references; do not claim publication readiness
        # if LaTeX warnings remain, but smoke compilation should produce PDF.
        rcodes = []
        for _ in range(2):
            r = subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"],
                               cwd=str(MANUSCRIPT), check=False, timeout=120,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            rcodes.append(r.returncode)
        if all(code == 0 for code in rcodes):
            _log("[manuscript] compiled.")
        else:
            _log(f"[manuscript] compiled with nonzero return codes: {rcodes}")
    except Exception as e:
        _log(f"[manuscript] ERROR {e}")



# ── Smoke mode: minimal deterministic reproducibility check ──────────────
def stage_smoke() -> None:
    """Very fast diagnostic smoke run.

    Avoids expensive optimisation, SALib sampling, and profile likelihoods.
    Verifies demo-data loading, preprocessing, simulation, objective evaluation,
    numerical diagnostics, a quick FIM diagnostic, report export, and expected
    figure/manuscript paths.
    """
    from data_io.loader import load_excel
    from data_io.preprocess import preprocess
    from core.reduced_model import simulate, DEFAULT_PARAMS
    from core.diagnostics import detect_instability
    from calibration.calibrate import sse
    from analysis.identifiability import fisher_information

    stage_folders()
    (ROOT / "execution_logs").mkdir(exist_ok=True)
    target = "dataset_I"
    path = DATA_SAMPLES / f"{target}.xlsx"
    report: Dict[str, Any] = {
        "mode": "smoke",
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_timestamp_utc": _now_utc(),
        "platform": _platform_str(),
        "dataset": target,
        "checks": {},
        "warnings": [],
        "skipped_analyses": [
            "calibration optimization skipped in smoke mode",
            "Morris/Sobol skipped in smoke mode",
            "profile likelihoods skipped in smoke mode",
            "validation bootstrap skipped in smoke mode",
        ],
    }
    ds = load_excel(str(path))
    ch_raw = ds.chambers[0]
    ch, issues = preprocess(ch_raw)
    report["checks"]["load_preprocess"] = {"ok": True, "n_samples_raw": int(len(ch_raw.t)), "n_samples_preprocessed": int(len(ch.t)), "warnings": list(issues)}
    proto = ch.to_protocol()
    params = dict(DEFAULT_PARAMS); params["alphas"] = [1.0] * len(proto.t_fccp)
    res = simulate(params, proto, o2_init=float(ch.o[0]), t_eval=np.asarray(ch.t[:120], dtype=float))
    report["checks"]["simulate"] = {"ok": bool(res.converged), "n_eval": int(len(res.t))}
    objective = sse(params, np.asarray(ch.t[:120], dtype=float), np.asarray(ch.o[:120], dtype=float), proto)
    report["checks"]["objective"] = {"ok": bool(np.isfinite(objective)), "sse": float(objective)}
    diag = detect_instability(params, proto, o2_init=float(ch.o[0]))
    report["checks"]["numerical_diagnostics"] = {"ok": bool(diag.is_healthy()), "warnings": list(diag.warnings)}
    idx = np.round(np.linspace(0, len(ch.t)-1, min(40, len(ch.t)))).astype(int)
    fim = fisher_information(params, np.asarray(ch.t[idx], dtype=float), np.asarray(ch.o[idx], dtype=float), proto, o2_init=float(ch.o[idx][0]), sigma_obs=1.0)
    report["checks"]["fim_diagnostic"] = {"ok": True, "condition_raw": float(fim.condition_raw), "condition_clipped": float(fim.condition_clipped), "warnings": list(fim.warnings)}
    report_path = RESULTS / "agent_reports" / f"{target}_smoke_report.json"
    report["checks"]["paths"] = {"figures_dir_exists": bool(FIGURES.exists()), "final_figures_dir_exists": bool((FIGURES / "final").exists()), "manuscript_main_exists": bool((MANUSCRIPT / "main.tex").exists())}
    _json_dump(report, report_path)
    _log(f"[smoke]      OK: wrote {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="mito_v2 reproducible pipeline. "
                    "DEFAULT MODE IS --publication.")
    p.add_argument("--fast", action="store_true",
                   help="Quick-turn DEVELOPMENT settings. NOT reportable: "
                        "fast-tier outputs must never populate manuscript "
                        "tables or figures.")
    p.add_argument("--smoke", action="store_true",
                   help="Minimal deterministic import/sanity check; no "
                        "expensive optimization. NOT scientific.")
    p.add_argument("--publication", action="store_true",
                   help="Scientifically defensible budgets (genuine profile "
                        "likelihoods, Sobol N_base>=1024, Morris>=20 "
                        "trajectories, refit-based holdout). THIS IS THE "
                        "DEFAULT if no mode flag is given.")
    p.add_argument("--skip-identifiability", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--manuscript-only", action="store_true")
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    args = p.parse_args()

    if args.smoke:
        stage_smoke()
        return 0

    # Mode resolution: --publication is the DEFAULT. --fast must be passed
    # explicitly to get the cheap development tier. --fast and --publication
    # together is a user error; publication wins and we say so.
    if args.fast and args.publication:
        _log("[mode] both --fast and --publication given; using "
             "--publication (the safe default).")
        cfg = get_settings("publication")
    elif args.fast:
        cfg = get_settings("fast")
    else:
        cfg = get_settings("publication")
    _log(f"[mode] tier='{cfg.tier}'  reportable={cfg.reportable}")

    if args.figures_only:
        stage_folders()
        stage_figures()
        return 0

    if args.manuscript_only:
        stage_manuscript()
        return 0

    stage_folders()
    stage_demo_data()
    # The fast development tier intentionally restricts to dataset_I unless
    # the caller explicitly overrides --datasets. The publication tier
    # always processes every requested dataset.
    if cfg.tier == "fast" and args.datasets == DATASETS:
        _log("[fast]       development tier uses dataset_I only; "
             "pass --datasets to override, or use --publication for all.")
        args.datasets = ["dataset_I"]
    diag = stage_diagnostics(args.datasets)
    calib = stage_calibration(args.datasets, cfg)
    identif: Dict[str, Any] = {}
    if not args.skip_identifiability:
        identif = stage_identifiability(args.datasets, cfg, calib)
    sens = stage_sensitivity(args.datasets, cfg, calib)
    valid = stage_validation(args.datasets, cfg, calib)
    stage_figures()
    stage_summary(diag, calib, identif, sens, valid, cfg)
    stage_manuscript()
    return 0


if __name__ == "__main__":
    sys.exit(main())
