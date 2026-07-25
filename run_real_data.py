"""Batch runner for real Oroboros OCR datasets.

Primary analysis: chambers are analysed separately. Optional averaged-chamber
analysis is written as supplementary sensitivity analysis only.

Example:
    python run_real_data.py --input-dir data_real --mode publication_real_data
    python run_real_data.py --input-dir data_samples --mode smoke --include-averaged
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.run_settings import get_settings
from core.reduced_model import simulate, DEFAULT_PARAMS
from core.diagnostics import detect_instability
from data_io.loader import load_excel, _load_csv, ChamberTrace, ExperimentDataset, downsample
from data_io.preprocess import preprocess
from calibration.calibrate import calibrate_de
from calibration.io import write_calibration_result_json, calibration_summary_csv
from calibration.phase import compute_phase_summary, write_phase_summary
from analysis.identifiability import fisher_information, fixed_parameter_scan, profile_likelihood
from analysis.identifiability_io import write_fim_json, write_profiles_json, identifiability_summary_csv, write_parameter_interpretability_flags
from analysis.validation import parametric_bootstrap_predictive_check, within_trace_holdout

try:
    from analysis.sensitivity import morris_screening, sobol_indices, time_resolved_sobol
    from analysis.sensitivity_io import write_morris_json, write_sobol_json, write_time_resolved_sobol_npz, sensitivity_summary_csv
except Exception:  # pragma: no cover
    morris_screening = sobol_indices = time_resolved_sobol = None


def _json_default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "trace"


def _load_dataset(path: Path) -> ExperimentDataset:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return load_excel(str(path))
    if path.suffix.lower() == ".csv":
        return _load_csv(str(path))
    raise ValueError(f"Unsupported real-data file type: {path}")


def _rmse_full(ch: ChamberTrace, params: Dict[str, Any]) -> float | None:
    try:
        proto = ch.to_protocol()
        sim = simulate(params, proto, o2_init=float(ch.o[0]), t_eval=ch.t)
        if not sim.converged: return None
        return float(np.sqrt(np.mean((sim.o - ch.o) ** 2)))
    except Exception:
        return None


def _write_diagnostics(rep, out_path: Path, dataset: str, chamber: str, mode: str) -> Dict[str, Any]:
    d = {
        "dataset": dataset,
        "chamber": chamber,
        "diagnostic_level": mode,
        "solver_success": bool(rep.converged),
        "finite_states": bool(rep.nan_count == 0),
        "oxygen_monotone_non_increasing": bool(rep.oxygen_monotone),
        "kappa_finite_range": bool(rep.kappa_in_range),
        "is_healthy": bool(rep.is_healthy()),
        "max_jacobian_eig": float(rep.max_jacobian_eig),
        "stiffness_ratio": float(rep.stiffness_ratio),
        "cytc_conservation_drift": float(rep.cytc_conservation_drift),
        "negative_state_count": int(rep.negative_state_count),
        "nan_count": int(rep.nan_count),
        "warnings": list(rep.warnings),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, indent=2, default=_json_default))
    return d


def _save_calibration_figure(ch: ChamberTrace, params: Dict[str, Any], out_path: Path, title: str):
    import matplotlib.pyplot as plt
    proto = ch.to_protocol()
    sim = simulate(params, proto, o2_init=float(ch.o[0]), t_eval=ch.t)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(ch.t, ch.o, label="Observed O₂", color="#0072BD", lw=1.6)
    if sim.converged:
        ax.plot(sim.t, sim.o, label="Fitted O₂", color="#D95319", lw=1.6)
    event_pairs = [(ch.t_oligo, "Oligomycin"), *[(t, f"FCCP {i+1}") for i,t in enumerate(ch.t_fccp)], (ch.t_inhibit, "Rot/Ant")]
    ymin, ymax = ax.get_ylim()
    for t, lab in event_pairs:
        if t is None: continue
        ax.axvline(float(t), color="0.25", ls="--", lw=0.9)
        ax.text(float(t), ymax, lab, rotation=90, va="top", ha="right", fontsize=8, color="black")
    ax.set_title(title, color="black")
    ax.set_xlabel("Time (s)", color="black")
    ax.set_ylabel("O₂ concentration", color="black")
    ax.tick_params(colors="black")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _average_chambers(ds: ExperimentDataset) -> ChamberTrace | None:
    if len(ds.chambers) < 2:
        return None
    a, b = ds.chambers[0], ds.chambers[1]
    lo = max(float(np.min(a.t)), float(np.min(b.t)))
    hi = min(float(np.max(a.t)), float(np.max(b.t)))
    if hi <= lo:
        return None
    n = int(min(len(a.t), len(b.t)))
    t = np.linspace(lo, hi, n)
    oa = np.interp(t, a.t, a.o)
    ob = np.interp(t, b.t, b.o)
    o = 0.5 * (oa + ob)
    return ChamberTrace(
        label="A_B_average_supplementary",
        t=t, o=o,
        t_start=a.t_start,
        t_oligo=a.t_oligo,
        t_fccp=list(a.t_fccp),
        t_inhibit=a.t_inhibit,
        t_end=a.t_end if a.t_end is not None else float(t[-1]),
        sigma_obs_est=float(np.nanstd(oa-ob)/math.sqrt(2)) if len(t) else None,
        metadata={"data_type": "supplementary_averaged_chamber", "source_chambers": [a.label, b.label], "note": "Averaged after interpolation to a shared grid; use as supplementary sensitivity analysis only."},
    )


def process_chamber(ch: ChamberTrace, dataset: str, chamber_id: str, cfg, root: Path, *, profiles: bool=True, sensitivity: bool=True) -> Dict[str, Any]:
    run_id = f"{_safe(dataset)}_{_safe(chamber_id)}"
    out: Dict[str, Any] = {"dataset": dataset, "chamber": chamber_id, "run_id": run_id}
    ch, prep_warnings = preprocess(ch)
    proto = ch.to_protocol()
    t_cal, o_cal = ch.t, ch.o
    if cfg.calib_n_downsample:
        t_cal, o_cal = downsample(t_cal, o_cal, cfg.calib_n_downsample)
    cal = calibrate_de(t_cal, o_cal, proto, maxiter=cfg.de_maxiter, popsize=cfg.de_popsize, seed=0, polish=cfg.de_polish)
    rmse_full = _rmse_full(ch, cal.params)
    cal_json = write_calibration_result_json(cal, root / "results_real" / "calibration" / f"calib_{run_id}.json", dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier, rmse_full_trace=rmse_full, extra_warnings=prep_warnings)
    out["calibration_json"] = str(root / "results_real" / "calibration" / f"calib_{run_id}.json")
    phase = compute_phase_summary(ch, cal.params)
    write_phase_summary(phase, root / "results_real" / "calibration" / f"phase_summary_{run_id}.json")
    _save_calibration_figure(ch, cal.params, root / "figures_real" / f"calibration_{run_id}.png", f"{dataset} — {chamber_id}")
    diag = detect_instability(cal.params, proto, o2_init=float(ch.o[0]))
    _write_diagnostics(diag, root / "results_real" / "diagnostics" / f"numerical_diagnostics_{run_id}.json", dataset, chamber_id, cfg.tier)
    # Identifiability
    t_id, o_id = ch.t, ch.o
    if cfg.identif_n_downsample:
        t_id, o_id = downsample(t_id, o_id, cfg.identif_n_downsample)
    fim = fisher_information(cal.params, t_id, o_id, proto, o2_init=float(o_id[0]), sigma_obs=cal.params.get("sigma_obs"))
    fim_path = root / "results_real" / "identifiability" / f"fim_{run_id}.json"
    write_fim_json(fim, fim_path, dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier)
    prof_path = root / "results_real" / "identifiability" / f"profiles_{run_id}.json"
    profs = {}
    if profiles:
        names = list(fim.param_names[:min(8, len(fim.param_names))])
        for nm in names:
            try:
                if cfg.profile_real:
                    profs[nm] = profile_likelihood(nm, cal.params, t_id, o_id, proto, o2_init=float(o_id[0]), n_grid=cfg.profile_n_grid, maxiter=cfg.profile_maxiter, grid_span_log=cfg.profile_grid_span_log, n_restarts_constrained=cfg.profile_n_restarts_constrained)
                else:
                    profs[nm] = fixed_parameter_scan(nm, cal.params, t_id, o_id, proto, o2_init=float(o_id[0]), n_grid=cfg.profile_n_grid)
            except Exception as e:
                print(f"[profiles] {run_id} {nm}: {e}")
    write_profiles_json(profs, prof_path, dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier, n_grid_used=cfg.profile_n_grid)
    write_parameter_interpretability_flags(prof_path, fim_path, root / "results_real" / "identifiability" / f"parameter_flags_{run_id}.csv")
    # Sensitivity (optional SALib)
    if sensitivity and morris_screening is not None:
        sens_dir = root / "results_real" / "sensitivity"
        try:
            m = morris_screening(proto, o2_init=float(ch.o[0]), N_trajectories=cfg.morris_trajectories, seed=0)
            write_morris_json(m, sens_dir / f"morris_{run_id}.json", dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier)
            s = sobol_indices(proto, o2_init=float(ch.o[0]), N=cfg.sobol_n_base, seed=0)
            write_sobol_json(s, sens_dir / f"sobol_auc_{run_id}.json", dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier)
            ts = time_resolved_sobol(proto, o2_init=float(ch.o[0]), N=cfg.time_resolved_sobol_n_base, n_t_eval=cfg.time_resolved_sobol_n_t, seed=0)
            write_time_resolved_sobol_npz(ts, sens_dir / f"time_resolved_sobol_{run_id}.npz", dataset=dataset, chamber=chamber_id, diagnostic_level=cfg.tier)
        except Exception as e:
            (sens_dir / f"sensitivity_skipped_{run_id}.json").parent.mkdir(parents=True, exist_ok=True)
            (sens_dir / f"sensitivity_skipped_{run_id}.json").write_text(json.dumps({"dataset": dataset, "chamber": chamber_id, "status": "skipped", "reason": str(e), "install": "python -m pip install -r requirements-sensitivity.txt"}, indent=2))
    # Validation
    val_dir = root / "results_real" / "validation"
    ppc = parametric_bootstrap_predictive_check(ch.t, ch.o, proto, cal.params, n_boot=cfg.bootstrap_n_boot, seed=0, refit=False)
    (val_dir / f"parametric_bootstrap_predictive_check_{run_id}.json").parent.mkdir(parents=True, exist_ok=True)
    (val_dir / f"parametric_bootstrap_predictive_check_{run_id}.json").write_text(json.dumps(ppc, indent=2, default=_json_default))
    try:
        wt = within_trace_holdout(ch.t, ch.o, proto, maxiter=cfg.within_trace_de_maxiter, popsize=cfg.within_trace_de_popsize, seed=0, polish=cfg.de_polish)
        (val_dir / f"within_trace_holdout_{run_id}.json").write_text(json.dumps(wt, indent=2, default=_json_default))
    except Exception as e:
        (val_dir / f"within_trace_holdout_{run_id}.json").write_text(json.dumps({"status":"failed", "error":str(e)}, indent=2))
    out.update({"rmse_calib": cal_json.get("rmse_calib"), "rmse_full_trace": cal_json.get("rmse_full_trace"), "sigma_obs": cal_json.get("sigma_obs"), "n_warnings": len(cal_json.get("warnings", []))})
    return out


def write_manuscript_exports(root: Path, rows: List[Dict[str, Any]]):
    tables = root / "manuscript" / "tables_real"
    tables.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(tables / "real_calibration_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    md = root / "manuscript" / "real_results_paragraphs.md"
    md.write_text("""# Draft real-data results paragraphs\n\nReplace bracketed values after reviewing all generated CSV/JSON outputs.\n\n## Calibration\nThe reduced 3-state OCR-informed model was calibrated to paired chamber traces from the real Oroboros datasets. Chambers A and B were analysed separately as the primary analysis to preserve technical-replicate information and avoid masking chamber-specific artifacts.\n\n## Identifiability\nFisher-information and profile-likelihood diagnostics were used to classify parameters as interpretable, weak, one-sided, flat/non-identifiable, or unresolved. Parameters without adequate profile support should not be interpreted as standalone biological endpoints.\n\n## Validation\nValidation diagnostics were interpreted as technical-transfer and noise-model checks, not biological generalization or disease validation.\n""")
    (root / "reports_real").mkdir(exist_ok=True)
    (root / "exports_real").mkdir(exist_ok=True)
    (root / "exports_real" / "real_data_run_summary.json").write_text(json.dumps({"runs": rows}, indent=2, default=_json_default))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run MitoAgent real-data batch analysis.")
    ap.add_argument("--input-dir", default="data_real", help="Directory containing real Excel/CSV files.")
    ap.add_argument("--mode", default="publication_real_data", choices=["smoke","fast","publication","publication_real_data"])
    ap.add_argument("--include-averaged", action="store_true", help="Also run supplementary averaged A/B chamber analysis.")
    ap.add_argument("--skip-profiles", action="store_true", help="Skip profile likelihoods/scans.")
    ap.add_argument("--skip-sensitivity", action="store_true", help="Skip optional SALib sensitivity analyses.")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent
    cfg = get_settings(args.mode)
    inp = root / args.input_dir
    files = sorted([*inp.glob("*.xlsx"), *inp.glob("*.xls"), *inp.glob("*.csv")])
    if not files:
        print(f"No Excel/CSV files found in {inp}. Place real datasets there first.")
        return 1
    rows: List[Dict[str, Any]] = []
    for fp in files:
        print(f"\n=== {fp.name} ===", flush=True)
        ds = _load_dataset(fp)
        chambers = list(ds.chambers)
        if args.include_averaged:
            avg = _average_chambers(ds)
            if avg is not None: chambers.append(avg)
        for idx, ch in enumerate(chambers):
            chamber_id = ch.metadata.get("detected_chamber") or ch.label or f"chamber_{idx+1}"
            if ch.metadata.get("data_type") == "supplementary_averaged_chamber":
                chamber_id = "A_B_average_supplementary"
            print(f"Processing {ds.name} — {chamber_id} ({cfg.tier})", flush=True)
            try:
                rows.append(process_chamber(ch, ds.name, chamber_id, cfg, root, profiles=not args.skip_profiles, sensitivity=not args.skip_sensitivity))
            except Exception as e:
                print(f"FAILED {ds.name} — {chamber_id}: {e}", flush=True)
                rows.append({"dataset": ds.name, "chamber": chamber_id, "run_id": f"{_safe(ds.name)}_{_safe(chamber_id)}", "status": "failed", "error": str(e)})
    write_manuscript_exports(root, rows)
    print("\nReal-data batch run complete.", flush=True)
    print("Primary outputs: results_real/, figures_real/, manuscript/tables_real/, reports_real/, exports_real/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
