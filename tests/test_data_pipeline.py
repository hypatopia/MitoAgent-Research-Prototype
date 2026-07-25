"""Tests for real-data parser behavior and preprocessing safeguards.

These are lightweight enough for the diagnostic smoke suite and cover:
  * Excel/CSV event labels
  * sparse-comment rejection
  * arbitrary FCCP parsing
  * chamber A/B detection
  * injection-window protection during outlier rejection
  * calibration-ready export
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from core.paths import demo_dataset
from data_io.loader import (
    ChamberTrace,
    export_calibration_ready,
    load_dataset,
    load_excel,
    parse_events,
)
from data_io.preprocess import reject_outliers, preprocess


def test_parse_events_recognises_all_intervention_labels():
    t = np.array([0., 100., 200., 300., 400., 500., 600., 700.])
    labels = ["start", "", "oligomycin", "", "FCCP", "", "rotenone+antimycin", "end"]
    ev = parse_events(t, labels)
    assert ev["t_start"] == 0.0
    assert ev["t_oligo"] == 200.0
    assert ev["t_fccp"] == [400.0]
    assert ev["t_inhibit"] == 600.0
    assert ev["t_end"] == 700.0


def test_parse_events_handles_arbitrary_multiple_fccp_in_order():
    t = np.array([0., 100., 200., 300., 400., 500., 600., 700.])
    labels = ["", "FCCP_3", "", "FCCP_1", "", "uncoupler", "", "FCCP_4"]
    ev = parse_events(t, labels)
    assert ev["t_fccp"] == [100.0, 300.0, 500.0, 700.0]


def _demo_df(n=200):
    return pd.DataFrame({
        "Time [s]": np.arange(0, n * 5, 5, dtype=float),
        "O2 conc.A [nmol/mL]": 200.0 - np.arange(n) * 0.5,
        "O2 conc.B [nmol/mL]": 199.0 - np.arange(n) * 0.48,
        "Event": [""] * n,
    })


def _add_events(df):
    df.loc[10, "Event"] = "start"
    df.loc[40, "Event"] = "oligomycin"
    df.loc[80, "Event"] = "FCCP_1"
    df.loc[110, "Event"] = "FCCP_2"
    df.loc[140, "Event"] = "rotenone+antimycin"
    df.loc[180, "Event"] = "end"
    return df


def test_excel_loader_accepts_sparse_column_with_intervention_labels_and_detects_chambers(tmp_path):
    p = tmp_path / "realistic_export.xlsx"
    _add_events(_demo_df()).to_excel(p, index=False)
    ds = load_excel(str(p))
    assert len(ds.chambers) == 2
    assert ds.chambers[0].t_oligo == 200.0
    assert ds.chambers[0].t_fccp == [400.0, 550.0]
    assert ds.chambers[0].t_inhibit == 700.0
    assert ds.chambers[0].metadata["detected_chamber"] == "A"
    assert ds.chambers[1].metadata["detected_chamber"] == "B"


def test_csv_loader_uses_same_autodetection_as_excel(tmp_path):
    p = tmp_path / "realistic_export.csv"
    _add_events(_demo_df()).to_csv(p, index=False)
    ds = load_dataset(str(p))
    ch = ds.chambers[0]
    assert ch.t_start == 50.0
    assert ch.t_oligo == 200.0
    assert ch.t_fccp == [400.0, 550.0]
    assert ch.t_inhibit == 700.0
    assert ch.metadata["event_column"] == "Event"


def test_loader_rejects_sparse_comment_column_without_keywords(tmp_path):
    df = pd.DataFrame({
        "Time [s]": np.arange(0, 1000, 5, dtype=float),
        "O2 conc.A": 200.0 - np.arange(0, 1000, 5) * 0.1,
        "Comments": [""] * 200,
    })
    df.loc[5, "Comments"] = "calibration check"
    df.loc[20, "Comments"] = "operator note"
    df.loc[80, "Comments"] = "lid opened briefly"
    p = tmp_path / "sparse_no_keywords.xlsx"
    df.to_excel(p, index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = load_excel(str(p))
    ch = ds.chambers[0]
    assert ch.t_oligo is None
    assert ch.t_inhibit is None
    assert ch.t_fccp == []
    assert any("rejected as event column" in str(wi.message) for wi in w)
    assert any("oligomycin" in msg for msg in ch.metadata["event_warnings"])


def test_outlier_rejection_protects_injection_windows():
    t = np.linspace(0, 1000, 201)
    o = 200.0 - 0.05 * t
    rng = np.random.default_rng(0)
    o = o + rng.normal(0, 0.1, size=t.shape)
    inj = 300.0
    inj_idx = int(np.argmin(np.abs(t - inj)))
    o[inj_idx] += 4.0
    o[inj_idx + 1] += 4.0

    _, _, mask1 = reject_outliers(t, o, n_sigma=3.0, event_times=None, exclude_window_sec=5.0)
    assert (~mask1[inj_idx:inj_idx + 2]).any()

    _, _, mask2 = reject_outliers(t, o, n_sigma=3.0, event_times=[inj], exclude_window_sec=5.0)
    assert mask2[inj_idx]
    assert mask2[inj_idx + 1]


def test_preprocess_pipeline_passes_event_times_to_outlier_rejection():
    ds = load_excel(str(demo_dataset("dataset_I")))
    ch_raw = ds.chambers[0]
    inj_idx = int(np.argmin(np.abs(ch_raw.t - ch_raw.t_oligo)))
    o2 = np.array(ch_raw.o, copy=True)
    o2[inj_idx] += 6.0
    ch2 = ChamberTrace(label=ch_raw.label, t=ch_raw.t.copy(), o=o2,
                       t_start=ch_raw.t_start, t_oligo=ch_raw.t_oligo,
                       t_fccp=list(ch_raw.t_fccp), t_inhibit=ch_raw.t_inhibit,
                       t_end=ch_raw.t_end, metadata=dict(ch_raw.metadata))
    ch_clean, issues = preprocess(ch2, do_outliers=True, n_sigma=3.0)
    nearest_t = ch_clean.t[np.argmin(np.abs(ch_clean.t - ch_raw.t_oligo))]
    assert abs(nearest_t - ch_raw.t_oligo) <= 5.0
    assert isinstance(issues, list)


def test_calibration_ready_export_round_trips(tmp_path):
    ds = load_excel(str(demo_dataset("dataset_I")))
    out = tmp_path / "dataset_I_A_calibration_ready.csv"
    export_calibration_ready(ds.chambers[0], str(out), dataset="dataset_I")
    df = pd.read_csv(out)
    assert {"time_s", "oxygen_nmol_ml", "chamber", "event", "dataset"}.issubset(df.columns)
    assert "oligomycin" in set(df["event"].fillna(""))
    assert "FCCP_1" in set(df["event"].fillna(""))
