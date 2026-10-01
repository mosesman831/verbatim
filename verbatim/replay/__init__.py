"""Replay laboratory (SPEC_V3 §43, T27).

Offline replay of recorded decisions inside a sandbox store, pinned by
manifests, reporting what-if divergences. The ``evidence.replay`` module
builds trajectory replay manifests; this package executes policy replay.

Boundary: sandbox store only (V3-43.01); local rules only — model_calls
is always 0 (V3-43.06); decisions without pinned ``state_json`` inputs
report ``not_replayable`` rather than fabricating inputs.
"""

from .lab import ReplayLab
from .report import DecisionDivergence, WhatIfReport

__all__ = ["ReplayLab", "WhatIfReport", "DecisionDivergence"]
