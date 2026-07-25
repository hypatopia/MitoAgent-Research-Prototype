"""
agent/orchestrator.py
=====================
The MitoAgent: a DIAGNOSTIC-GATED orchestrator for the methodology pipeline.

This is NOT a broad autonomous scientific reasoning agent. It automatically
sequences the analysis tools according to predefined diagnostic rules:

  load -> preprocess -> stability sanity -> calibrate -> stability of fit
       -> identifiability -> sensitivity -> validation -> report

At each stage, diagnostic flags are inspected and the pipeline either
continues, escalates a warning, or aborts. The natural-language LLM driver
is OPTIONAL — the scientific workflow runs deterministically without an
LLM via the `run_pipeline` Python entry point and the `agent.cli` script.

Warning categories
------------------
Section I requires that warnings be classified into three orthogonal
buckets so consumers can act on them appropriately:

  * `numerical_stability`     — solver / integration / preprocessing
                                stability issues (NOT identifiability)
  * `identifiability`         — high FIM condition (sloppiness),
                                non-identifiable parameters, MAP outside CI
  * `validation_noise_model`  — coverage outside the configurable band,
                                noise-model misspecification cues
  * `data_pipeline`           — loader/preprocess issues that surfaced
                                during ingestion

The agent records both `warnings_by_category` (mapping category → list
of strings) and `warning_counts` (mapping category → int) in the final
report.

Configurable diagnostic thresholds
----------------------------------
`run_pipeline` accepts:

  * `coverage_band=(low, high)` — the configurable warning band for
    parametric-bootstrap predictive-check coverage. Default (0.80, 0.95).
    This is a CONFIGURABLE WARNING THRESHOLD, not a validated band of
    biologically-correct calibration.
  * `fim_sloppy_threshold` — FIM raw-condition value above which an
    `identifiability` warning is raised. Default 1e15.

Both thresholds are recorded in the report's `diagnostic_thresholds`
field.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.tools import TOOLS, TOOL_DESCRIPTIONS
from agent.reporting import enrich_report


WARN_CATEGORIES = (
    "numerical_stability", "identifiability",
    "validation_noise_model", "data_pipeline", "unsupported_claim",
)


@dataclass
class AgentState:
    """All in-memory state the agent maintains across a session."""
    chamber:        Optional[Any]   = None
    params:         Optional[Dict]  = None
    proto_summary:  Optional[Dict]  = None
    last_calib:     Optional[Dict]  = None
    last_stability: Optional[Dict]  = None
    last_identif:   Optional[Dict]  = None
    last_sens:      Optional[Dict]  = None
    last_validation: Optional[Dict] = None
    log:            List[Dict]      = field(default_factory=list)
    warnings_by_category: Dict[str, List[str]] = field(
        default_factory=lambda: {c: [] for c in WARN_CATEGORIES})
    skipped_analyses: List[str] = field(default_factory=list)


class MitoAgent:
    """Diagnostic-gated orchestrator.

    Public methods (`load`, `preprocess`, `calibrate`, `check_stability`,
    `identifiability`, `sensitivity`, `validate`, `run_pipeline`,
    `save_report`) are independent of any LLM. Use `run_pipeline()` for
    the bundled diagnostic-gated sequence.
    """

    def __init__(self, verbose: bool = True):
        self.state   = AgentState()
        self.verbose = verbose
        self._last_report: Optional[Dict[str, Any]] = None

    # ── Tool dispatch (logs everything) ────────────────────────────────
    def _call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in TOOLS:
            return {"ok": False, "message": f"unknown tool: {tool_name}"}
        t0 = time.perf_counter()
        result = TOOLS[tool_name](**kwargs)
        elapsed = time.perf_counter() - t0
        log_entry = {
            "tool":     tool_name,
            "kwargs":   {k: v for k, v in kwargs.items()
                          if not k.startswith("_") and not hasattr(v, "to_protocol")},
            "ok":       result.get("ok"),
            "message":  result.get("message"),
            "elapsed":  elapsed,
        }
        self.state.log.append(log_entry)
        if self.verbose:
            flag = "OK " if result.get("ok") else "FAIL"
            print(f"  [{flag}] {tool_name}({elapsed:.1f}s):  {result.get('message','')}")
        return result

    # ── Warning recorder (orthogonal categories) ───────────────────────
    def _warn(self, category: str, text: str) -> None:
        """Record a warning in the chosen orthogonal category and the
        tool log. `category` must be one of WARN_CATEGORIES."""
        if category not in WARN_CATEGORIES:
            category = "data_pipeline"   # safe fallback
        self.state.warnings_by_category[category].append(text)
        self.state.log.append({
            "tool":    "_warn",
            "category": category,
            "message": text,
        })
        if self.verbose:
            print(f"  WARN [{category}]: {text}")

    # Backward-compat: a few existing callers use _note for non-warning
    # messages. Keep it so they still work.
    def _note(self, text: str) -> None:
        if self.verbose:
            print(f"  NOTE: {text}")
        self.state.log.append({"tool": "_note", "message": text})

    # ── Public methods (also LLM-callable) ─────────────────────────────
    def load(self, path: str, chamber_index: int = 0):
        r = self._call("load_data", path=path, chamber_index=chamber_index)
        if not r["ok"]:
            self._warn("data_pipeline", r.get("message", "load failed"))
            return r
        self.state.chamber       = r.get("_chamber_obj")
        self.state.proto_summary = r["result"]
        return r

    def preprocess(self, do_outliers=True, n_sigma=4.0):
        if self.state.chamber is None:
            return {"ok": False, "message": "no chamber loaded; call .load() first"}
        r = self._call("preprocess_data", chamber=self.state.chamber,
                        do_outliers=do_outliers, n_sigma=n_sigma)
        if r["ok"]:
            self.state.chamber = r.get("_chamber_obj", self.state.chamber)
            issues = r.get("result", {}).get("validation_issues", [])
            for issue in issues:
                self._warn("data_pipeline",
                            f"preprocess flagged: {issue}")
        return r

    def calibrate(self, method="de", n_data=250, **kwargs):
        if self.state.chamber is None:
            return {"ok": False, "message": "no chamber; call .load()"}
        r = self._call("calibrate", chamber=self.state.chamber,
                        method=method, n_data=n_data, **kwargs)
        if r["ok"]:
            self.state.params     = r.get("_params")
            self.state.last_calib = r["result"]
        return r

    def check_stability(self):
        if self.state.params is None:
            return {"ok": False, "message": "no params; call .calibrate() first"}
        r = self._call("check_stability", chamber=self.state.chamber,
                        params=self.state.params)
        if r["ok"]:
            self.state.last_stability = r["result"]
            if not r["result"].get("is_healthy", True):
                self._warn("numerical_stability",
                            "fitted-parameter integration shows "
                            "numerical-stability flags; results may be "
                            "less reliable.")
        return r

    def identifiability(self, method="fim", **kwargs):
        if self.state.params is None:
            return {"ok": False, "message": "no params; call .calibrate() first"}
        r = self._call("analyze_identifiability", chamber=self.state.chamber,
                        params=self.state.params, method=method, **kwargs)
        if r["ok"]:
            self.state.last_identif = r["result"]
        return r

    def sensitivity(self, method="morris", N=20):
        if self.state.chamber is None:
            return {"ok": False, "message": "no chamber; call .load()"}
        r = self._call("analyze_sensitivity", chamber=self.state.chamber,
                        method=method, N=N)
        if r["ok"]:
            self.state.last_sens = r["result"]
        return r

    def validate(self, method="ppc", **kwargs):
        if self.state.params is None:
            return {"ok": False, "message": "no params; call .calibrate()"}
        r = self._call("validate", chamber=self.state.chamber,
                        params=self.state.params, method=method, **kwargs)
        if r["ok"]:
            self.state.last_validation = r["result"]
        return r

    # ── Autonomous (diagnostic-gated) pipeline ─────────────────────────
    def run_pipeline(self,
                      path: str,
                      *,
                      fast: bool = True,
                      coverage_band: Tuple[float, float] = (0.80, 0.95),
                      fim_sloppy_threshold: float = 1e15,
                      chamber_index: int = 0,
                      ) -> Dict[str, Any]:
        """Run the diagnostic-gated pipeline.

        Parameters
        ----------
        path : str
            Path to an Excel/CSV file containing one or more chambers.
        fast : bool, default True
            Fast mode: small DE settings, FIM-only identifiability,
            small sensitivity samples, n_boot=100 for the predictive
            check, profile likelihoods skipped.
            Publication mode (fast=False): larger DE settings, FIM +
            profile likelihoods, larger sensitivity samples, n_boot=500.
        coverage_band : (low, high), default (0.80, 0.95)
            Configurable warning threshold for the parametric-bootstrap
            predictive-check 90% coverage. NOT a validated band of
            biologically-correct calibration.
        fim_sloppy_threshold : float, default 1e15
            FIM raw-condition value above which an `identifiability`
            warning is raised.
        chamber_index : int, default 0
            Which chamber of the loaded file to analyse.

        Both thresholds are recorded in the report's
        `diagnostic_thresholds` field.
        """
        mode = "fast" if fast else "publication"
        if self.verbose:
            print(f"\n{'='*72}")
            print(f"MitoAgent diagnostic-gated pipeline:  {path}  (mode: {mode})")
            print(f"{'='*72}")

        skipped: List[str] = []

        # Stage 1 — load
        r = self.load(path, chamber_index)
        if not r["ok"]:
            return self._final_report(error=r["message"], mode=mode,
                skipped_analyses=skipped, coverage_band=coverage_band,
                fim_sloppy_threshold=fim_sloppy_threshold)

        # Stage 2 — preprocess
        r = self.preprocess()
        if not r["ok"]:
            return self._final_report(error=r["message"], mode=mode,
                skipped_analyses=skipped, coverage_band=coverage_band,
                fim_sloppy_threshold=fim_sloppy_threshold)

        # Stage 3 — stability sanity check (default params)
        if self.verbose: print("\n-- Stability sanity check (default params) --")
        s0 = self._call("check_stability", chamber=self.state.chamber,
                         params={"alphas": [1.0]*self.state.proto_summary["n_fccp"]})
        if not s0["ok"] or not s0["result"].get("is_healthy", False):
            self._warn("numerical_stability",
                        "default-parameter integration unhealthy; "
                        "aborting pipeline before calibration.")
            return self._final_report(
                error="default-param stability failed",
                mode=mode, skipped_analyses=skipped,
                coverage_band=coverage_band,
                fim_sloppy_threshold=fim_sloppy_threshold)

        # Stage 4 — calibrate
        if self.verbose: print("\n-- Calibration --")
        method = "staged" if fast else "de"
        kw: Dict[str, Any] = {}
        if fast:
            kw.update({"maxiter_stageA": 3, "maxiter_stageB": 3})
        r = self.calibrate(method=method, n_data=60 if fast else 200, **kw)
        if not r["ok"]:
            return self._final_report(error=r["message"], mode=mode,
                skipped_analyses=skipped, coverage_band=coverage_band,
                fim_sloppy_threshold=fim_sloppy_threshold)

        # Stage 5 — stability of the fit
        if self.verbose: print("\n-- Stability check (fitted params) --")
        self.check_stability()

        # Stage 6 — identifiability
        if self.verbose: print("\n-- Identifiability --")
        if fast:
            self.identifiability(method="fim")
            skipped.append("profile_likelihoods (publication-mode only)")
        else:
            self.identifiability(method="both",
                                  param_subset=["V_max", "K_r", "gamma_oligo",
                                                "tau_kappa", "r0"])
        # FIM sloppiness -> identifiability warning (NOT numerical_stability)
        if self.state.last_identif and "fim" in self.state.last_identif:
            cond_raw = float(self.state.last_identif["fim"].get(
                "condition_raw",
                self.state.last_identif["fim"].get("condition", 0.0)))
            if cond_raw > fim_sloppy_threshold:
                self._warn(
                    "identifiability",
                    f"FIM raw condition cond_raw={cond_raw:.2e} exceeds the "
                    f"configurable sloppiness threshold "
                    f"{fim_sloppy_threshold:.0e}. The information matrix is "
                    f"practically rank-deficient; direct interpretation of "
                    f"the inverse-FIM correlation matrix is NOT warranted."
                )

        # Stage 7 — sensitivity
        if self.verbose: print("\n-- Sensitivity (Morris) --")
        self.sensitivity(method="morris", N=2 if fast else 30)

        # Stage 8 — validation (parametric-bootstrap predictive check)
        if self.verbose: print(
            "\n-- Validation (parametric-bootstrap predictive check) --")
        n_boot = 5 if fast else 500
        r = self.validate(method="ppc", n_boot=n_boot)
        if r["ok"]:
            cov_key = ("parametric_bootstrap_coverage_90"
                        if "parametric_bootstrap_coverage_90"
                            in r["result"]
                        else "coverage_90")
            cov = float(r["result"].get(cov_key, float("nan")))
            lo, hi = coverage_band
            if not (lo <= cov <= hi):
                self._warn(
                    "validation_noise_model",
                    f"parametric-bootstrap predictive-check coverage "
                    f"{cov*100:.1f}% lies outside the configurable warning "
                    f"band [{lo:.2f}, {hi:.2f}]. This is a HINT of possible "
                    f"noise-model misspecification or "
                    f"heteroscedasticity, NOT proof of either."
                )

        return self._final_report(
            mode=mode,
            skipped_analyses=skipped,
            coverage_band=coverage_band,
            fim_sloppy_threshold=fim_sloppy_threshold,
        )

    # ── Final report assembly ──────────────────────────────────────────
    def _final_report(self,
                       *,
                       error: Optional[str] = None,
                       mode: str = "fast",
                       skipped_analyses: Optional[List[str]] = None,
                       coverage_band: Tuple[float, float] = (0.80, 0.95),
                       fim_sloppy_threshold: float = 1e15,
                       ) -> Dict[str, Any]:
        skipped = list(skipped_analyses or [])
        skipped.extend(self.state.skipped_analyses)
        warnings_by_category = {
            c: list(self.state.warnings_by_category.get(c, []))
            for c in WARN_CATEGORIES
        }
        warning_counts = {c: len(v) for c, v in warnings_by_category.items()}

        report = {
            "ok":                  error is None,
            "error":               error,
            "mode":                mode,
            "skipped_analyses":    skipped,
            "diagnostic_thresholds": {
                "coverage_band":         [float(coverage_band[0]),
                                            float(coverage_band[1])],
                "fim_sloppy_threshold":  float(fim_sloppy_threshold),
            },
            "warnings_by_category": warnings_by_category,
            "warning_counts":       warning_counts,
            "data":                 self.state.proto_summary,
            "calibration":          self.state.last_calib,
            "stability":            self.state.last_stability,
            "identifiability":      self.state.last_identif,
            "sensitivity":          self.state.last_sens,
            "validation":           self.state.last_validation,
            "tool_log":             list(self.state.log),
        }
        # Add deterministic interpretation/reporting sections without rerunning analyses.
        report = enrich_report(report)
        # Cache the report so save_report() doesn't recompute / re-warn.
        self._last_report = copy.deepcopy(report)
        if self.verbose:
            self._print_summary(report)
        return report

    def _print_summary(self, r: Dict):
        print("\n" + "="*72)
        print("PIPELINE SUMMARY")
        print("="*72)
        print(f"mode:               {r.get('mode')}")
        print(f"skipped_analyses:   {r.get('skipped_analyses')}")
        print(f"warning_counts:     {r.get('warning_counts')}")
        print(f"diagnostic_thresholds: {r.get('diagnostic_thresholds')}")
        if r.get("error"):
            print(f"Pipeline error: {r['error']}")
            return
        d = r.get("data") or {}
        print(f"Data:          n={d.get('n_samples')}  "
               f"FCCP injections: {d.get('n_fccp')}  "
               f"noise SD≈{d.get('noise_sd_estimate')}")
        c = r.get("calibration") or {}
        if c:
            print(f"Calibration:   {c.get('method')}  RMSE={c.get('rmse_calib'):.3f}")
        i = r.get("identifiability") or {}
        if i and "fim" in i:
            cond_raw = i["fim"].get("condition_raw",
                                     i["fim"].get("condition"))
            print(f"Identifiability: cond_raw(FIM) = {cond_raw:.2e}")
        v = r.get("validation") or {}
        if v:
            cov = v.get("parametric_bootstrap_coverage_90",
                         v.get("coverage_90", 0))
            print(f"Validation:    parametric-bootstrap coverage_90 = "
                   f"{cov*100:.1f}%")
        print("="*72)

    # ── Save / load report ─────────────────────────────────────────────
    def save_report(self, path: str) -> str:
        """Serialise the cached `_final_report()` to disk.

        If `run_pipeline` has been called, the cached report is reused —
        we do not re-run anything or re-emit warnings.
        """
        report = (self._last_report
                   if self._last_report is not None
                   else self._final_report())
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path
