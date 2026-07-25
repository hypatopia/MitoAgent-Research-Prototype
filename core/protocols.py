"""
Protocol utilities for the 3-state OCR-informed mitochondrial stress-test model.

The canonical Protocol dataclass lives in ``core.reduced_model`` for backward
compatibility with existing calibration and analysis modules. This file provides
a stable import location for manuscript-facing and UI-facing code that needs to
construct or summarize protocol event schedules.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Mapping, Any

from core.reduced_model import Protocol, smooth_step


def make_protocol(
    *,
    t_oligo: float,
    t_fccp: Iterable[float],
    t_inhibit: float,
    t_end: float,
    t_start: float = 0.0,
    k_step: float = 2.0,
) -> Protocol:
    """Create a Protocol with sorted FCCP times and basic consistency checks."""
    fccp = sorted(float(t) for t in t_fccp)
    proto = Protocol(
        t_start=float(t_start),
        t_oligo=float(t_oligo),
        t_fccp=fccp,
        t_inhibit=float(t_inhibit),
        t_end=float(t_end),
        k_step=float(k_step),
    )
    validate_protocol(proto)
    return proto


def validate_protocol(proto: Protocol) -> None:
    """Raise ValueError if event order is incompatible with a stress test."""
    if proto.t_end <= proto.t_start:
        raise ValueError("t_end must be after t_start")
    if not (proto.t_start <= proto.t_oligo <= proto.t_end):
        raise ValueError("t_oligo must lie within the recorded time range")
    for t in proto.t_fccp:
        if not (proto.t_oligo <= t <= proto.t_inhibit):
            raise ValueError("FCCP injections should occur after oligomycin and before inhibition")
    if not (proto.t_oligo <= proto.t_inhibit <= proto.t_end):
        raise ValueError("t_inhibit must occur after oligomycin and before t_end")
    if proto.k_step <= 0:
        raise ValueError("k_step must be positive")


def protocol_metadata(proto: Protocol) -> dict[str, Any]:
    """Return JSON-serializable protocol metadata for result files."""
    d = asdict(proto)
    d["n_fccp"] = proto.n_fccp
    d["event_model"] = "smooth_tanh_injections"
    d["model_scope"] = "reduced_ocr_informed_civ_mediated_ocr"
    return d
