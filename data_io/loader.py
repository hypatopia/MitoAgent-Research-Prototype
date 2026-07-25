"""
data_io/loader.py
=================
Data pipeline for mitochondrial respirometry traces.

Handles:
  * Excel files (representative Oroboros / Seahorse-style exports)
  * CSV files with the same detection logic as Excel
  * Numpy .npy files used by the bundled synthetic/demo traces
  * Event-label parsing -> core.reduced_model.Protocol objects
  * Multiple FCCP injections, including 1, 2, 4, or arbitrary counts
  * Calibration-ready export for downstream model fitting

Important terminology
---------------------
Bundled Excel files are DEMO/SYNTHETIC parser fixtures generated from demo
traces. They are not real Oroboros measurements and must not be used for
biological inference. Noise estimates are within-trace observational-noise
estimates, not biological variability.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


# ── Public dataclasses ────────────────────────────────────────────────────
@dataclass
class ChamberTrace:
    """A single chamber's time/oxygen trace plus its detected protocol."""
    label: str
    t: np.ndarray
    o: np.ndarray
    t_oligo: Optional[float] = None
    t_fccp: List[float] = field(default_factory=list)
    t_inhibit: Optional[float] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    sigma_obs_est: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_protocol(self, k_step: float = 2.0):
        """Construct a core.reduced_model.Protocol from detected events."""
        from core.reduced_model import Protocol
        if self.t_oligo is None or self.t_inhibit is None:
            raise ValueError(
                f"Chamber {self.label}: oligomycin or inhibition event not "
                "detected. Cannot build Protocol. Inspect event parsing first."
            )
        return Protocol(
            t_oligo=float(self.t_oligo),
            t_fccp=[float(x) for x in self.t_fccp],
            t_inhibit=float(self.t_inhibit),
            t_end=float(self.t_end if self.t_end is not None else self.t[-1]),
            t_start=float(self.t_start if self.t_start is not None else self.t[0]),
            k_step=k_step,
        )


@dataclass
class ExperimentDataset:
    """Top-level container. May contain multiple chambers per file."""
    name: str
    chambers: List[ChamberTrace] = field(default_factory=list)
    raw_path: Optional[str] = None

    def __len__(self):
        return len(self.chambers)

    def __iter__(self):
        return iter(self.chambers)


# ── Event-label parsing ───────────────────────────────────────────────────
_EVENT_PATTERNS = {
    "start": re.compile(r"^\s*(start|begin|t0|baseline|substrate)\s*$", re.I),
    "oligo": re.compile(r"olig", re.I),
    "fccp": re.compile(r"fccp|uncoupler", re.I),
    "inhibit": re.compile(
        r"(rot|rotenone|antim|antimycin|rot[/+\s-]*ant|rotenone[/+\s-]*antimycin|^\s*ra\s*$)",
        re.I,
    ),
    "end": re.compile(r"^\s*(end|stop|finish)\s*$", re.I),
}


def _is_missing_label(x) -> bool:
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def parse_events(t_arr: np.ndarray, label_arr) -> dict:
    """Parse a time array + parallel label column into intervention events.

    Returns keys ``t_start``, ``t_oligo``, ``t_fccp`` (list), ``t_inhibit``,
    and ``t_end``. FCCP events are sorted in temporal order and may have any
    count: 1, 2, 4, or arbitrary titration sequences.
    """
    events = {"t_start": None, "t_oligo": None, "t_fccp": [],
              "t_inhibit": None, "t_end": None}
    if label_arr is None:
        return events
    n = min(len(t_arr), len(label_arr))
    for i in range(n):
        lab = label_arr[i]
        if _is_missing_label(lab):
            continue
        lab_str = str(lab).strip()
        if not lab_str:
            continue
        ti = float(t_arr[i])
        if _EVENT_PATTERNS["start"].search(lab_str) and events["t_start"] is None:
            events["t_start"] = ti
        elif _EVENT_PATTERNS["oligo"].search(lab_str) and events["t_oligo"] is None:
            events["t_oligo"] = ti
        elif _EVENT_PATTERNS["fccp"].search(lab_str):
            events["t_fccp"].append(ti)
        elif _EVENT_PATTERNS["inhibit"].search(lab_str) and events["t_inhibit"] is None:
            events["t_inhibit"] = ti
        elif _EVENT_PATTERNS["end"].search(lab_str) and events["t_end"] is None:
            events["t_end"] = ti
    events["t_fccp"] = sorted(events["t_fccp"])
    return events


def event_warnings(events: dict) -> List[str]:
    """Return parser warnings for missing/ambiguous intervention labels."""
    out: List[str] = []
    if events.get("t_start") is None:
        out.append("start/baseline event label not detected; first sample will be used when needed")
    if events.get("t_oligo") is None:
        out.append("oligomycin event label not detected")
    if not events.get("t_fccp"):
        out.append("FCCP event label not detected")
    if events.get("t_inhibit") is None:
        out.append("rotenone/antimycin inhibition event label not detected")
    if events.get("t_end") is None:
        out.append("end event label not detected; last sample will be used when needed")
    ordered = [events.get("t_oligo"), *(events.get("t_fccp") or []), events.get("t_inhibit")]
    ordered = [x for x in ordered if x is not None]
    if any(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
        out.append("intervention labels are not in expected temporal order: oligomycin < FCCP(s) < rotenone/antimycin")
    return out


# ── Detection helpers ─────────────────────────────────────────────────────
def _numeric_series(df, c) -> bool:
    try:
        import pandas as pd
        s = pd.to_numeric(df[c], errors="coerce")
        return bool(s.notna().sum() >= max(3, int(0.5 * len(s))))
    except Exception:
        return False


def _detect_time_column(df, columns: List[str]) -> str:
    for c in columns:
        cl = str(c).lower()
        if "time" in cl or "[s]" in cl or "sec" in cl or cl.strip() in {"t", "time_s"}:
            return c
    for c in columns:
        if _numeric_series(df, c):
            vals = np.asarray(__import__("pandas").to_numeric(df[c], errors="coerce"), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 2 and np.all(np.diff(vals) >= 0):
                return c
    raise ValueError("Could not auto-detect time column")


def _detect_o2_columns(df, columns: List[str], time_col, event_col=None) -> List[str]:
    out = []
    for c in columns:
        if c == time_col or c == event_col:
            continue
        cl = str(c).lower().replace(" ", "")
        looks_o2 = any(tok in cl for tok in ["o2", "oxygen", "oxygenconc", "chamber", "ch_a", "ch_b", "conc.a", "conc.b"])
        if looks_o2 and _numeric_series(df, c):
            out.append(c)
    if out:
        return out
    return [c for c in columns if c != time_col and c != event_col and _numeric_series(df, c)]


def _detect_chamber_label(column_name) -> str:
    s = str(column_name)
    sl = s.lower().replace(" ", "")
    if re.search(r"(?:chamber|ch|conc\.)[_\-. ]*a\b", s, re.I) or sl.endswith("a[nmol/ml]") or ".a" in sl:
        return "A"
    if re.search(r"(?:chamber|ch|conc\.)[_\-. ]*b\b", s, re.I) or sl.endswith("b[nmol/ml]") or ".b" in sl:
        return "B"
    return s


def _recognised_event_strings(values: Iterable) -> List[str]:
    hits = []
    for v in values:
        if _is_missing_label(v):
            continue
        s = str(v).strip()
        if s and any(pat.search(s) for pat in _EVENT_PATTERNS.values()):
            hits.append(s)
    return hits


def _detect_event_column(df, columns: List[str], time_col, o2_cols: Optional[List[str]] = None):
    import pandas as pd
    o2_cols = set(o2_cols or [])
    for c in columns:
        if c == time_col or c in o2_cols:
            continue
        vals = df[c]
        non_na = vals[pd.notna(vals)]
        non_empty = [str(v).strip() for v in non_na if str(v).strip()]
        if not non_empty:
            continue
        if not (1 <= len(non_empty) <= max(50, int(0.25 * len(df)))):
            continue
        hits = _recognised_event_strings(non_empty)
        if hits:
            return c
        # sparse string/comment column with no recognised interventions must not be accepted
        if getattr(df[c].dtype, "kind", "") in {"O", "U", "S"} or df[c].dtype == object:
            warnings.warn(
                f"Sparse text column '{c}' rejected as event column: no recognised intervention labels "
                f"(start/oligo/fccp/rot/antimycin/end) found among {non_empty[:5]!r}",
                stacklevel=2,
            )
    return None


def _build_chambers(df, path: str, sheet, time_col, o2_cols: List[str], event_col) -> List[ChamberTrace]:
    import pandas as pd
    t_arr = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    labels = df[event_col].tolist() if event_col is not None else None
    events = parse_events(t_arr, labels)
    warnings_list = event_warnings(events)
    chambers: List[ChamberTrace] = []
    for oc in o2_cols:
        o_arr = pd.to_numeric(df[oc], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(o_arr) & np.isfinite(t_arr)
        t_c, o_c = t_arr[m], o_arr[m]
        order = np.argsort(t_c)
        t_c, o_c = t_c[order], o_c[order]
        metadata = {
            "path": path,
            "sheet": sheet,
            "column": str(oc),
            "detected_chamber": _detect_chamber_label(oc),
            "time_column": str(time_col),
            "event_column": None if event_col is None else str(event_col),
            "data_type": "demo/synthetic" if "data_samples" in os.path.normpath(path).split(os.sep) else "real_or_user_supplied",
            "event_warnings": list(warnings_list),
        }
        ch = ChamberTrace(
            label=str(oc),
            t=t_c,
            o=o_c,
            t_oligo=events["t_oligo"],
            t_fccp=list(events["t_fccp"]),
            t_inhibit=events["t_inhibit"],
            t_start=events["t_start"],
            t_end=events["t_end"],
            sigma_obs_est=estimate_noise_sd(o_c),
            metadata=metadata,
        )
        chambers.append(ch)
    return chambers


# ── Pre-processing helpers ────────────────────────────────────────────────
def estimate_noise_sd(o: np.ndarray, window: int = 21) -> float:
    """Estimate within-trace observational noise SD using rolling residual MAD.

    This is not biological variability and is not a substitute for replicate
    uncertainty. It is a single-trace high-frequency observation-noise estimate.
    """
    o = np.asarray(o, dtype=float)
    o = o[np.isfinite(o)]
    if len(o) == 0:
        return float("nan")
    if len(o) < window:
        return float(np.std(o))
    kernel = np.ones(window) / window
    pad = window // 2
    o_pad = np.pad(o, pad, mode="edge")
    smooth = np.convolve(o_pad, kernel, mode="valid")[:len(o)]
    resid = o - smooth
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(1.4826 * mad)


def downsample(t: np.ndarray, o: np.ndarray, n_target: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly down-sample to approximately ``n_target`` points."""
    if len(t) <= n_target:
        return t, o
    idx = np.round(np.linspace(0, len(t) - 1, n_target)).astype(int)
    return t[idx], o[idx]


# ── Excel / CSV / NPY loaders ─────────────────────────────────────────────
def load_excel(path: str, sheet: Union[str, int, None] = 0,
               time_col: str = None, o2_cols: List[str] = None,
               event_col: str = None) -> ExperimentDataset:
    """Load an Excel respirometry export with automatic column detection."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pandas is required for Excel loading. pip install pandas openpyxl") from e
    df = pd.read_excel(path, sheet_name=sheet)
    cols = list(df.columns)
    time_col = time_col or _detect_time_column(df, cols)
    if o2_cols is None:
        provisional_event = event_col if event_col in cols else None
        o2_cols = _detect_o2_columns(df, cols, time_col, provisional_event)
    if event_col is None:
        event_col = _detect_event_column(df, cols, time_col, o2_cols)
    if not o2_cols:
        raise ValueError(f"Could not auto-detect oxygen/O2 columns in {path}")
    chambers = _build_chambers(df, path, sheet, time_col, o2_cols, event_col)
    return ExperimentDataset(name=os.path.splitext(os.path.basename(path))[0], chambers=chambers, raw_path=path)


def _load_csv(path: str, time_col: str = None,
              o2_cols: Optional[List[str]] = None,
              event_col: Optional[str] = None) -> ExperimentDataset:
    """Load CSV with the same detection rules as Excel."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = list(df.columns)
    time_col = time_col or _detect_time_column(df, cols)
    if o2_cols is None:
        provisional_event = event_col if event_col in cols else None
        o2_cols = _detect_o2_columns(df, cols, time_col, provisional_event)
    if event_col is None:
        event_col = _detect_event_column(df, cols, time_col, o2_cols)
    elif event_col not in df.columns:
        event_col = None
    if not o2_cols:
        raise ValueError(f"Could not auto-detect oxygen/O2 columns in {path}")
    chambers = _build_chambers(df, path, None, time_col, o2_cols, event_col)
    return ExperimentDataset(name=os.path.splitext(os.path.basename(path))[0], chambers=chambers, raw_path=path)


def load_npy(t_path: str, o_path: str, events: Optional[dict] = None, label: str = "chamber") -> ExperimentDataset:
    """Load a t/o pair of .npy files. Events can be supplied externally."""
    t = np.load(t_path)
    o = np.load(o_path)
    events = events or {}
    ch = ChamberTrace(
        label=label,
        t=t,
        o=o,
        t_oligo=events.get("t_oligo"),
        t_fccp=list(events.get("t_fccp", [])),
        t_inhibit=events.get("t_inhibit"),
        t_start=events.get("t_start", float(t[0])),
        t_end=events.get("t_end", float(t[-1])),
        sigma_obs_est=estimate_noise_sd(o),
        metadata={"t_path": t_path, "o_path": o_path, "data_type": "demo/synthetic"},
    )
    ch.metadata["event_warnings"] = event_warnings({
        "t_start": ch.t_start, "t_oligo": ch.t_oligo, "t_fccp": ch.t_fccp,
        "t_inhibit": ch.t_inhibit, "t_end": ch.t_end,
    })
    return ExperimentDataset(name=os.path.basename(t_path).replace("_t.npy", ""), chambers=[ch], raw_path=t_path)


def load_dataset(path: str, **kwargs) -> ExperimentDataset:
    """Detect file type and dispatch to Excel, CSV, or paired NPY loader."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return load_excel(path, **kwargs)
    if ext == ".csv":
        return _load_csv(path, **kwargs)
    if ext == ".npy":
        t_path = path
        o_path = path.replace("_t.npy", "_o.npy")
        if not os.path.exists(o_path):
            raise FileNotFoundError(f"Companion file not found: {o_path}")
        return load_npy(t_path, o_path, **kwargs)
    raise ValueError(f"Unsupported file extension: {ext}")


def export_calibration_ready(ch: ChamberTrace, out_path: str, dataset: Optional[str] = None) -> str:
    """Export one chamber to a calibration-ready CSV with metadata columns.

    Columns: ``time_s``, ``oxygen_nmol_ml``, ``chamber``, ``event``. Event labels
    are placed at nearest sampled rows for inspection and reproducibility.
    """
    import pandas as pd
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame({
        "time_s": np.asarray(ch.t, dtype=float),
        "oxygen_nmol_ml": np.asarray(ch.o, dtype=float),
        "chamber": ch.metadata.get("detected_chamber", ch.label),
        "source_column": ch.label,
        "dataset": dataset or ch.metadata.get("dataset", "unknown"),
        "event": [""] * len(ch.t),
    })
    def set_event(t_event, label):
        if t_event is None or len(ch.t) == 0:
            return
        idx = int(np.argmin(np.abs(ch.t - float(t_event))))
        df.loc[idx, "event"] = label
    set_event(ch.t_start, "start")
    set_event(ch.t_oligo, "oligomycin")
    for j, tf in enumerate(ch.t_fccp or [], start=1):
        set_event(tf, f"FCCP_{j}")
    set_event(ch.t_inhibit, "rotenone+antimycin")
    set_event(ch.t_end, "end")
    df.to_csv(out_path, index=False)
    return out_path


# ── Helper event definitions for bundled demo traces ──────────────────────
SAMPLE_EVENTS = {
    "dataset_I": {"t_start": 210.0, "t_oligo": 300.0, "t_fccp": [480.0], "t_inhibit": 660.0, "t_end": 826.2},
    "dataset_II": {"t_start": 210.0, "t_oligo": 300.0, "t_fccp": [480.0, 560.0], "t_inhibit": 720.0, "t_end": 826.2},
    "dataset_III": {"t_start": 210.0, "t_oligo": 300.0, "t_fccp": [420.0, 480.0, 540.0, 600.0], "t_inhibit": 720.0, "t_end": 826.2},
}


def add_event_labels_to_npy(ds_name: str, t_array: np.ndarray, out_csv: str) -> str:
    """Create a sparse event-label CSV next to bundled NPY demo traces."""
    import pandas as pd
    events = SAMPLE_EVENTS[ds_name]
    label_col = [""] * len(t_array)
    def _set(t_target: float, lab: str):
        idx = int(np.argmin(np.abs(t_array - t_target)))
        label_col[idx] = lab
    _set(events["t_start"], "start")
    _set(events["t_oligo"], "oligomycin")
    for j, tj in enumerate(events["t_fccp"], start=1):
        _set(tj, f"FCCP_{j}")
    _set(events["t_inhibit"], "rotenone+antimycin")
    _set(events["t_end"], "end")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame({"time_s": t_array, "event": label_col}).to_csv(out_csv, index=False)
    return out_csv
