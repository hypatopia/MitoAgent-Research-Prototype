import json
from pathlib import Path

from calibration.phase import compute_phase_summary
from core.paths import demo_dataset
from data_io.loader import load_excel

ROOT = Path(__file__).resolve().parents[1]


def test_phase_summary_contains_required_phases():
    ds = load_excel(str(demo_dataset("dataset_I")))
    summary = compute_phase_summary(ds.chambers[0])
    phases = {p["phase"] for p in summary["phases"]}
    assert "basal" in phases
    assert "oligomycin" in phases
    assert "FCCP/uncoupled" in phases
    assert "Rot/Ant residual" in phases
    assert "within-trace observational" in summary["terminology_note"]


def test_existing_calibration_json_uses_sse_not_mcmc():
    path = ROOT / "results" / "calibration" / "calib_dataset_I.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["objective_type"] == "SSE_with_post_hoc_sigma"
    assert "posterior" not in json.dumps(payload).lower()
