"""
core/model.py
=============
Stable public import path for the 3-state OCR-informed model.

Historically the implementation lived in ``core/reduced_model.py``. To
preserve every existing import in the package, that module is kept
unchanged. New code should prefer::

    from core.model import simulate, Protocol, DEFAULT_PARAMS, ...

which re-exports the same public API from ``core.reduced_model``. If the
implementation is ever moved or restructured, this alias is the only
file callers need to depend on.
"""
from core.reduced_model import (  # noqa: F401
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    CORE_PARAM_ORDER,
    Protocol,
    SimulationResult,
    rhs,
    simulate,
    smooth_step,
    get_bounds_vec,
    get_param_names,
    params_to_vec,
    vec_to_params,
)
