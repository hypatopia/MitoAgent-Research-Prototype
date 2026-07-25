"""
data_io/preprocess.py
=====================
Pre-processing utilities for respirometry traces.

Functions
---------
trim_pre_injection(ch, lead_seconds=0.0)
    Remove sample points before t_start - lead.
reject_outliers(o, n_sigma=4.0, window=51, event_times=None,
                exclude_window_sec=5.0)
    Median-absolute-deviation rejection on rolling residuals. Samples within
    ±exclude_window_sec of any event time are protected from rejection
    because sharp injection transitions are structurally meaningful, not
    electrode glitches.
smooth_signal(o, window=5)
    Light moving-average smoothing (off by default; preserved as option).
align_to_first_event(ch)
    Shift t-axis so t_start = 0. Valid only if event times are also shifted
    consistently with the trace (handled internally).
validate_chamber(ch)
    Sanity checks: monotone time, decreasing oxygen, all events ordered.
"""
from __future__ import annotations
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io.loader import ChamberTrace


# ── Outlier rejection ───────────────────────────────────────────────────
def reject_outliers(t: np.ndarray, o: np.ndarray,
                    n_sigma: float = 4.0, window: int = 51,
                    event_times: Optional[Sequence[float]] = None,
                    exclude_window_sec: float = 5.0,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject points whose residual from a rolling median exceeds
    n_sigma robust standard deviations.

    Samples within ±exclude_window_sec of any time in `event_times` are
    NEVER rejected — sharp transitions around oligomycin / FCCP /
    rotenone+antimycin injections are structurally meaningful and must
    not be confused with electrode glitches.

    Returns (t_clean, o_clean, mask_kept).
    """
    if len(o) < window:
        return t, o, np.ones_like(o, dtype=bool)
    pad = window // 2
    o_pad = np.pad(o, pad, mode="edge")
    medians = np.array([np.median(o_pad[i:i+window]) for i in range(len(o))])
    resid = o - medians
    mad = np.median(np.abs(resid - np.median(resid)))
    sd = 1.4826 * mad if mad > 0 else np.std(resid)
    keep = np.abs(resid) <= n_sigma * sd

    # Protect injection windows.
    if event_times:
        protected = np.zeros_like(o, dtype=bool)
        for te in event_times:
            if te is None:
                continue
            protected |= (np.abs(t - float(te)) <= exclude_window_sec)
        keep |= protected
    return t[keep], o[keep], keep


# ── Smoothing (off by default; preserve raw data) ───────────────────────
def smooth_signal(o: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving-average smoother. Use sparingly; default is no smoothing
    so calibration sees the raw observations and the noise-SD estimate is
    consistent with the likelihood model."""
    if window <= 1:
        return o
    pad = window // 2
    o_pad = np.pad(o, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(o_pad, kernel, mode="valid")[:len(o)]


# ── Baseline trim ───────────────────────────────────────────────────────
def trim_pre_injection(ch: ChamberTrace, lead_seconds: float = 0.0
                       ) -> ChamberTrace:
    """Remove samples before (t_start - lead_seconds).
    Useful when an Oroboros recording starts well before the substrates
    are added and the leading equilibration phase is non-informative.
    """
    if ch.t_start is None:
        return ch
    cutoff = ch.t_start - lead_seconds
    mask = ch.t >= cutoff
    return replace(ch, t=ch.t[mask], o=ch.o[mask])


# ── Time-axis alignment ─────────────────────────────────────────────────
def align_to_first_event(ch: ChamberTrace) -> ChamberTrace:
    """Shift t-axis so the recording begins at t=0.
    All event time points are shifted by the same offset.
    """
    if len(ch.t) == 0:
        return ch
    t0 = float(ch.t[0])
    return replace(ch, t=ch.t - t0,
                   t_start=0.0 if ch.t_start is None else ch.t_start - t0,
                   t_oligo=None if ch.t_oligo is None else ch.t_oligo - t0,
                   t_fccp=[float(x - t0) for x in ch.t_fccp],
                   t_inhibit=None if ch.t_inhibit is None else ch.t_inhibit - t0,
                   t_end=None if ch.t_end is None else ch.t_end - t0)


# ── Sanity validation ───────────────────────────────────────────────────
def validate_chamber(ch: ChamberTrace) -> List[str]:
    """Return a list of validation warnings (empty list = OK)."""
    issues = []
    if len(ch.t) < 30:
        issues.append(f"Too few samples ({len(ch.t)}); need >= 30.")
    if len(ch.t) != len(ch.o):
        issues.append(f"t and o length mismatch: {len(ch.t)} vs {len(ch.o)}.")
    if not np.all(np.diff(ch.t) > 0):
        issues.append("Time axis is not strictly increasing.")
    # Oxygen decreasing? (allowed local upticks within noise)
    n_up = int(np.sum(np.diff(ch.o) > 1.0))    # nmol/mL jumps
    if n_up > 5:
        issues.append(f"{n_up} samples show implausible oxygen jumps > 1 nmol/mL.")
    if ch.t_oligo is None:
        issues.append("t_oligo not detected.")
    if ch.t_inhibit is None:
        issues.append("t_inhibit not detected.")
    if not ch.t_fccp:
        issues.append("No FCCP events detected.")
    # Event ordering
    times = ([ch.t_oligo] + list(ch.t_fccp) + [ch.t_inhibit])
    times = [x for x in times if x is not None]
    if any(times[i] >= times[i+1] for i in range(len(times)-1)):
        issues.append("Event times not in increasing order: oligo < FCCPs < inhibit.")
    return issues


# ── Bundled preprocess pipeline ─────────────────────────────────────────
def preprocess(ch: ChamberTrace,
               do_outliers: bool = True, n_sigma: float = 4.0,
               do_smooth: bool = False, smooth_window: int = 5,
               do_trim: bool = True, lead_seconds: float = 0.0,
               do_align: bool = False
               ) -> Tuple[ChamberTrace, List[str]]:
    """Full preprocessing pipeline; returns (cleaned_chamber, warnings)."""
    if do_trim:
        ch = trim_pre_injection(ch, lead_seconds=lead_seconds)
    if do_outliers:
        # Collect event times so the bundled pipeline protects injection
        # windows from outlier rejection (sharp transitions are signal,
        # not glitches).
        event_times: List[Optional[float]] = []
        for et in (ch.t_oligo, ch.t_inhibit, ch.t_start, ch.t_end):
            if et is not None:
                event_times.append(float(et))
        for et in (ch.t_fccp or []):
            event_times.append(float(et))
        t, o, _ = reject_outliers(ch.t, ch.o, n_sigma=n_sigma,
                                  event_times=event_times)
        ch = replace(ch, t=t, o=o)
    if do_smooth:
        ch = replace(ch, o=smooth_signal(ch.o, smooth_window))
    if do_align:
        ch = align_to_first_event(ch)
    issues = validate_chamber(ch)
    return ch, issues
