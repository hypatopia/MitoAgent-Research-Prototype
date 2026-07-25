from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "results" / "validation"


def test_chunk8_validation_artifacts_exist_after_fast_run():
    expected = [
        VAL / "parametric_bootstrap_predictive_check_dataset_I.json",
        VAL / "within_trace_holdout_dataset_I.json",
        VAL / "chamber_holdout_summary.csv",
        VAL / "lodo_summary.csv",
        VAL / "validation_summary.csv",
    ]
    for path in expected:
        assert path.exists(), f"missing {path}"


def test_parametric_bootstrap_not_posterior_predictive():
    d = json.loads((VAL / "parametric_bootstrap_predictive_check_dataset_I.json").read_text())
    assert d["method"] == "parametric_bootstrap_predictive_check"
    assert "posterior" in d["explicit_disclaimer"].lower()
    assert "NOT" in d["explicit_disclaimer"]
    assert 0.0 <= float(d["parametric_bootstrap_coverage_90"]) <= 1.0


def test_within_trace_has_diagnostic_caveat():
    d = json.loads((VAL / "within_trace_holdout_dataset_I.json").read_text())
    assert d["method"] == "within_trace_holdout"
    assert "intervention-phase" in d["framing"]
    assert "not biological validation" in d["explicit_disclaimer"].lower()
    assert d["rmse_train"] is not None
    assert d["rmse_test"] is not None


def test_chamber_and_lodo_summaries_are_cautious():
    with open(VAL / "chamber_holdout_summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "technical-replicate" in rows[0]["validation_mode"]
    assert "not independent biological validation" in rows[0]["interpretation"].lower()

    with open(VAL / "lodo_summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "pooled-transfer" in rows[0]["validation_mode"]
    assert "not proof" in rows[0]["limitation"].lower()
