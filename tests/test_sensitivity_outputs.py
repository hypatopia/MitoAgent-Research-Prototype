
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def test_sensitivity_output_files_exist_after_fast_run():
    root = Path(__file__).resolve().parents[1]
    sens = root / "results" / "sensitivity"
    required = [
        sens / "morris_dataset_I.json",
        sens / "sobol_auc_dataset_I.json",
        sens / "sensitivity_summary.csv",
        sens / "time_resolved_sobol_dataset_I.npz",
        sens / "time_resolved_sobol_dataset_I.meta.json",
        sens / "sensitivity_interpretation_dataset_I.json",
    ]
    for path in required:
        assert path.exists(), f"missing sensitivity output: {path}"


def test_sobol_interpretation_note_and_parameter_set():
    root = Path(__file__).resolve().parents[1]
    with open(root / "results" / "sensitivity" / "sobol_auc_dataset_I.json") as f:
        d = json.load(f)
    assert d["metric"] == "AUC_OCR"
    assert "alpha_1" in d["parameter_set"]
    note = d.get("interpretation_note", "").lower()
    assert "interaction" in note
    assert "not additive" in note or "exclusive" in note


def test_time_resolved_sobol_has_variance_degenerate_mask():
    root = Path(__file__).resolve().parents[1]
    npz = np.load(root / "results" / "sensitivity" / "time_resolved_sobol_dataset_I.npz")
    mask = npz["variance_degenerate_mask"]
    assert mask.dtype == bool
    assert mask.size > 0
    with open(root / "results" / "sensitivity" / "time_resolved_sobol_dataset_I.meta.json") as f:
        meta = json.load(f)
    assert meta["metric"] == "OCR(t)"
    assert "not interpret" in meta.get("interpretation_note", "").lower()


def test_sensitivity_interpretation_caveats():
    root = Path(__file__).resolve().parents[1]
    with open(root / "results" / "sensitivity" / "sensitivity_interpretation_dataset_I.json") as f:
        d = json.load(f)
    caveats = " ".join(d.get("caveats", [])).lower()
    assert "does not prove" in caveats
    assert "not additive" in caveats
    assert d["time_resolved_sobol_available"] is True
