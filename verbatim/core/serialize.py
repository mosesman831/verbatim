"""Canonical JSON serialization seam (SPEC_V5 V5-06.01).

The versioned wire/durability format lives here so consumers never grow
private ``json`` callsites: canonical dumps are sorted-key, compact,
UTF-8-safe, and reject non-finite floats; loads are bounded, reject
NaN/Infinity and duplicate keys, and fail with typed ``VALIDATION``
errors.

The implementations are the single canonical pair defined in
``verbatim.core.types`` (``json_dumps``/``safe_json_loads``) — this
module re-exports them so the serialization seam is importable as its
own module without duplicating the codec, plus ``json_loads`` as the
strict-by-default reader name callers reach for first.
"""

from __future__ import annotations

from typing import Any

from .types import json_dumps, safe_json_loads


def json_loads(text: str, *, max_bytes: int = 1 << 20, max_depth: int = 32) -> Any:
    """Strict JSON read — the ``safe_json_loads`` contract under the
    conventional ``*_loads`` name."""
    return safe_json_loads(text, max_bytes=max_bytes, max_depth=max_depth)


__all__ = ["json_dumps", "json_loads", "safe_json_loads"]
