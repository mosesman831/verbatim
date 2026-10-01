"""§32.17 ``stage_profile/v7`` — per-call stage records + envelope aggregation.

Two surfaces:

* **Producers.** ``record_search`` / ``record_add`` validate and build the
  frozen ``StageRecord`` contract (``verbatim.core.types_v7``), and
  ``ProfiledCall`` is the measurement context a pipeline — or a test —
  wraps around one call: ``mark(stage)`` opens a named stage interval,
  ``mark_lane(name)`` opens a per-lane interval, ``set_field`` /
  ``set_candidates`` fill scalar and per-lane fields, and injected
  counters supply ``sql_statements`` / ``snapshots`` / ``bytes_read``.
* **Aggregation.** ``aggregate(records)`` produces the §32.17 envelope:
  ``{field: {n, p50, p95, p99, mean, max}}`` per numeric field
  (nearest-rank percentiles, the ``eval/v5`` convention). Non-numeric
  fields (``status``, ``coverage_digest``) aggregate to an honest value
  histogram ``{n, counts: {value: n}}`` — a percentile over strings would
  be a fabricated number.

Field names follow §32.17 exactly; the spec's ``t_lane[<name>]`` /
``candidates[<lane>]`` materialize in aggregates as the flat keys
``t_lane.<name>`` / ``candidates.<name>``. All ``t_*`` values are
milliseconds; ``bytes_read`` is bytes; the other counters are unitless.
Fields left unset stay absent — a stage that never ran contributes no
sample, so every aggregate row carries its own denominator ``n`` (the
v5 rule: report nothing rather than fabricate a zero).
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Optional

from verbatim.core.types_v7 import (
    ADD_STAGE_FIELDS,
    SEARCH_STAGE_FIELDS,
    StageRecord,
)

#: Artifact tag pinned on every serialized record (§32.17 format id; all
#: §32 constants are ``provisional/v7-r0`` — the tag is recorded, not
#: implied, so envelopes stay comparable once constants re-freeze).
RECORD_FORMAT = "stage_profile/v7"

_KINDS = ("search", "add")
_FIELDS_FOR = {
    "search": frozenset(SEARCH_STAGE_FIELDS),
    "add": frozenset(ADD_STAGE_FIELDS),
}

#: Declared fields that are labels, not measurements — they aggregate to
#: histograms, never to percentiles.
_NONNUMERIC_FIELDS = frozenset({"status", "coverage_digest"})

#: Scalar fields the context itself owns: the wall-clock total is measured
#: at __exit__, so ``mark("total")`` / ``mark("ack")`` are not stages.
_RESERVED_MARKS = frozenset({"total", "ack"})


def _markable_stages(kind: str) -> frozenset:
    """Stage names ``mark()`` accepts for ``kind`` — the declared ``t_*``
    fields minus the context-owned totals."""
    return frozenset(
        f[2:] for f in _FIELDS_FOR[kind] if f.startswith("t_")
    ) - _RESERVED_MARKS


_MARKABLE = {kind: _markable_stages(kind) for kind in _KINDS}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_scalar(kind: str, name: str, value: Any) -> None:
    allowed = _FIELDS_FOR[kind]
    if name not in allowed:
        raise ValueError(
            f"unknown {kind} stage field {name!r}; "
            f"declared fields: {sorted(allowed)}"
        )
    if name in _NONNUMERIC_FIELDS:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be str, got {type(value).__name__}"
            )
    elif not _is_number(value) or not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{name} must be a finite number >= 0, got {value!r}"
        )


def _check_lane_name(lane: Any) -> None:
    if not isinstance(lane, str) or not lane:
        raise ValueError(
            f"lane keys must be non-empty strings, got {lane!r}"
        )


def _build(
    kind: str,
    fields: Optional[Mapping[str, Any]],
    t_lane: Optional[Mapping[str, Any]],
    candidates: Optional[Mapping[str, Any]],
) -> StageRecord:
    rec = StageRecord(kind=kind)
    for name, value in (fields or {}).items():
        _check_scalar(kind, name, value)
        rec.fields[name] = value
    for lane, ms in (t_lane or {}).items():
        _check_lane_name(lane)
        if not _is_number(ms) or not math.isfinite(ms) or ms < 0:
            raise ValueError(
                f"t_lane[{lane!r}] must be a finite number >= 0, "
                f"got {ms!r}"
            )
        rec.t_lane[lane] = float(ms)
    for lane, n in (candidates or {}).items():
        _check_lane_name(lane)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(
                f"candidates[{lane!r}] must be an int >= 0, got {n!r}"
            )
        rec.candidates[lane] = int(n)
    return rec


# ---------------------------------------------------------------------------
# producers
# ---------------------------------------------------------------------------


def record_search(
    fields: Optional[Mapping[str, Any]] = None,
    *,
    t_lane: Optional[Mapping[str, Any]] = None,
    candidates: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> StageRecord:
    """Build a §32.17 *search* record.

    Scalar fields arrive as a mapping and/or keyword arguments (keywords
    win on collision); every name is validated against the frozen field
    set — ``t_total`` … ``t_post``, ``sql_statements``, ``snapshots``,
    ``bytes_read``, ``pool_R``, ``items``, ``tokens``, ``status``,
    ``coverage_digest``. ``t_lane`` maps lane name -> milliseconds and
    ``candidates`` maps lane name -> produced-candidate count (the spec's
    ``t_lane[<name>]`` / ``candidates[<lane>]``). Fields left out stay
    absent: absent is honest, zero is a claim.
    """
    merged = dict(fields or {})
    merged.update(kwargs)
    return _build("search", merged, t_lane, candidates)


def record_add(
    fields: Optional[Mapping[str, Any]] = None,
    *,
    t_lane: Optional[Mapping[str, Any]] = None,
    candidates: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> StageRecord:
    """Build a §32.17 *add* record — ``t_ack``, ``t_screen``, ``t_tx``,
    ``t_enqueue``, and the async ``t_visible`` / ``t_t0`` (fill those when
    the post-ack signal arrives; they are absent, not zero, until then).
    Same scalar/t_lane/candidates validation as ``record_search``.
    """
    merged = dict(fields or {})
    merged.update(kwargs)
    return _build("add", merged, t_lane, candidates)


class ProfiledCall:
    """Context manager: measure one search/add call into a ``StageRecord``.

    ::

        with ProfiledCall("search", sql_counter=sql) as pc:
            pc.mark("barrier")
            snap = acquire_snapshot()
            pc.mark("analyze")
            norm = analyze(q)
            pc.mark_lane("lexical")
            lex = lane_lexical(ctx, qv, slice_)
            pc.mark("union")
            ...
            res = pc(run_search, ctx, qv, deadline_ms)
            pc.set_field("status", res.status)
        rec = pc.record

    ``mark(stage)`` opens the named stage interval — elapsed time accrues
    to it until the next ``mark``/``mark_lane`` or context exit. Markable
    names are the declared ``t_*`` suffixes (``barrier``, ``analyze``,
    ``union``, ``rrf``, ``rerank_feat``, ``rerank_ce``, ``boost``,
    ``verdict``, ``pack``, ``post`` for search; ``screen``, ``tx``,
    ``enqueue``, ``visible``, ``t0`` for add). Time before the first mark
    is honest residual inside the total — never smeared into a named
    stage. One interval is open at a time; a repeated lane accumulates.

    ``clock`` is injectable (defaults to ``time.perf_counter``, seconds)
    so tests are deterministic. ``sql_counter`` / ``snapshot_counter`` /
    ``bytes_counter`` accept the injected instrumentation hooks (which
    land in a later wave): a callable ``() -> number`` or an object with
    a ``.count``/``.value`` attribute is read at entry and exit and the
    *delta* is recorded; a plain number is recorded as the absolute
    value. Counter deltas overwrite a same-named field set mid-call —
    instrumentation is authoritative.

    The context owns the wall-clock total: ``t_total`` for search,
    ``t_ack`` for add (the ack boundary is the call's return). A value
    the caller already set is kept. ``t_visible``/``t_t0`` on adds are
    async — ``set_field`` stays usable after exit for exactly that
    purpose. An exception inside the block still yields a finalized
    record and is never suppressed.
    """

    def __init__(
        self,
        kind: str,
        *,
        clock: Optional[Callable[[], float]] = None,
        sql_counter: Any = None,
        snapshot_counter: Any = None,
        bytes_counter: Any = None,
    ) -> None:
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.kind = kind
        self._clock = clock if clock is not None else time.perf_counter
        self._counters = {
            "sql_statements": sql_counter,
            "snapshots": snapshot_counter,
            "bytes_read": bytes_counter,
        }
        self._rec: Optional[StageRecord] = None
        self._open: Optional[str] = None
        self._open_is_lane = False
        self._enter_at = 0.0
        self._mark_at = 0.0
        self._counter_start: dict[str, Any] = {}
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ProfiledCall":
        self._rec = StageRecord(kind=self.kind)
        self._closed = False
        self._open = None
        self._open_is_lane = False
        self._enter_at = self._mark_at = self._clock()
        self._counter_start = {
            name: _counter_read(c)
            for name, c in self._counters.items()
            if c is not None and not _counter_is_static(c)
        }
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._rec is None:
            return False
        now = self._clock()
        self._accrue(now)
        self._open = None
        total_field = "t_total" if self.kind == "search" else "t_ack"
        self._rec.fields.setdefault(
            total_field, (now - self._enter_at) * 1000.0
        )
        for name, counter in self._counters.items():
            if counter is None:
                continue
            if _counter_is_static(counter):
                self._rec.fields[name] = counter
            else:
                self._rec.fields[name] = (
                    _counter_read(counter) - self._counter_start[name]
                )
        self._closed = True
        return False

    # -- in-call API --------------------------------------------------------

    @property
    def record(self) -> StageRecord:
        if self._rec is None:
            raise RuntimeError(
                "ProfiledCall has no record before __enter__"
            )
        return self._rec

    def __call__(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn`` — the 'wraps a callable' half of the contract."""
        return fn(*args, **kwargs)

    def mark(self, stage: str) -> None:
        """Open stage ``stage`` (a declared ``t_*`` suffix for this kind).

        Time accrues to it until the next mark or exit. Unknown names —
        and the context-owned ``total``/``ack`` — are rejected so a typo
        can never mint an undeclared field.
        """
        self._require_open()
        allowed = _MARKABLE[self.kind]
        if stage not in allowed:
            raise ValueError(
                f"unknown {self.kind} stage {stage!r}; "
                f"markable stages: {sorted(allowed)}"
            )
        self._advance(stage, is_lane=False)

    def mark_lane(self, lane: str) -> None:
        """Open a per-lane interval — accrues to ``t_lane[lane]``."""
        self._require_open()
        _check_lane_name(lane)
        self._advance(lane, is_lane=True)

    def set_field(self, name: str, value: Any) -> None:
        """Set a declared scalar field — in-call (``status``, ``pool_R``,
        ``items``, ``tokens`` …) or post-exit for the async add fields
        (``t_visible``, ``t_t0``). Validated like the producers."""
        if self._rec is None:
            raise RuntimeError(
                "ProfiledCall has no record before __enter__"
            )
        _check_scalar(self.kind, name, value)
        self._rec.fields[name] = value

    def set_candidates(self, lane: str, n: int) -> None:
        """Record lane ``lane``'s produced-candidate count
        (``candidates[<lane>]``)."""
        if self._rec is None:
            raise RuntimeError(
                "ProfiledCall has no record before __enter__"
            )
        _check_lane_name(lane)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(
                f"candidates[{lane!r}] must be an int >= 0, got {n!r}"
            )
        self._rec.candidates[lane] = int(n)

    # -- internals -----------------------------------------------------------

    def _require_open(self) -> None:
        if self._rec is None:
            raise RuntimeError(
                "ProfiledCall has no record before __enter__"
            )
        if self._closed:
            raise RuntimeError("ProfiledCall is closed (after __exit__)")

    def _advance(self, label: str, *, is_lane: bool) -> None:
        now = self._clock()
        self._accrue(now)
        self._open, self._open_is_lane = label, is_lane
        self._mark_at = now

    def _accrue(self, now: float) -> None:
        if self._open is None:
            return
        dt_ms = (now - self._mark_at) * 1000.0
        if self._open_is_lane:
            prev = self._rec.t_lane.get(self._open, 0.0)
            self._rec.t_lane[self._open] = prev + dt_ms
        else:
            key = "t_" + self._open
            self._rec.fields[key] = self._rec.fields.get(key, 0.0) + dt_ms


def _counter_is_static(counter: Any) -> bool:
    return _is_number(counter)


def _counter_read(counter: Any) -> Any:
    if callable(counter):
        return counter()
    for attr in ("count", "value"):
        if hasattr(counter, attr):
            return getattr(counter, attr)
    return counter


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _percentile_stats(vals: Iterable[float]) -> dict[str, Any]:
    """Nearest-rank p50/p95/p99 — the ``eval.v5.harness.percentiles``
    convention (``sorted[round(q * (n - 1))]``) with the same key set."""
    s = sorted(float(v) for v in vals)
    n = len(s)

    def _p(q: float) -> float:
        return s[min(n - 1, max(0, int(round(q * (n - 1)))))]

    return {
        "n": n,
        "p50": _p(0.50),
        "p95": _p(0.95),
        "p99": _p(0.99),
        "mean": sum(s) / n,
        "max": s[-1],
    }


def aggregate(records: Iterable[StageRecord]) -> dict[str, dict[str, Any]]:
    """Aggregate §32.17 records into the per-field envelope.

    Returns ``{field: stats}`` where ``stats`` is
    ``{n, p50, p95, p99, mean, max}`` for fields whose samples are all
    finite numbers, and ``{n, counts: {value: n}}`` for label fields
    (``status``, ``coverage_digest`` — determinism shows up as a single
    digest bucket) or for any field whose samples are mixed-type. Per-lane
    channels flatten to ``t_lane.<name>`` / ``candidates.<name>``. Keys are
    sorted for deterministic artifacts; every field reports its own ``n``
    so sparse instrumentation keeps honest denominators. Empty input
    yields ``{}`` — no fabricated rows.
    """
    series: dict[str, list[Any]] = {}
    for rec in records:
        for name, value in rec.fields.items():
            series.setdefault(name, []).append(value)
        for lane, ms in rec.t_lane.items():
            series.setdefault(f"t_lane.{lane}", []).append(ms)
        for lane, n in rec.candidates.items():
            series.setdefault(f"candidates.{lane}", []).append(n)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(series):
        vals = series[name]
        if all(_is_number(v) and math.isfinite(v) for v in vals):
            out[name] = _percentile_stats(vals)
        else:
            counts: dict[str, int] = {}
            for v in vals:
                key = v if isinstance(v, str) else repr(v)
                counts[key] = counts.get(key, 0) + 1
            out[name] = {
                "n": len(vals),
                "counts": {k: counts[k] for k in sorted(counts)},
            }
    return out


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def record_to_dict(rec: StageRecord) -> dict[str, Any]:
    """The canonical JSON-able shape of one record."""
    return {
        "format": RECORD_FORMAT,
        "kind": rec.kind,
        "fields": dict(rec.fields),
        "t_lane": dict(rec.t_lane),
        "candidates": dict(rec.candidates),
    }


def record_from_dict(doc: Mapping[str, Any]) -> StageRecord:
    """Rebuild a record from ``record_to_dict`` output — fields are
    re-validated, so a hand-edited artifact fails loudly, not silently."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"record document must be a mapping, got {doc!r}")
    kind = doc.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"record kind must be one of {_KINDS}, got {kind!r}")
    return _build(
        kind,
        doc.get("fields") or {},
        doc.get("t_lane") or {},
        doc.get("candidates") or {},
    )


def to_json(obj: Any) -> str:
    """Canonical JSON for a ``StageRecord``, an iterable of records, or a
    plain mapping (e.g. an ``aggregate()`` envelope passed through
    unchanged). Sorted keys + strict JSON (``allow_nan=False``) so
    artifacts are byte-deterministic and non-finite values fail loudly.
    """
    if isinstance(obj, StageRecord):
        payload: Any = record_to_dict(obj)
    elif isinstance(obj, Mapping):
        payload = dict(obj)
    else:
        payload = [record_to_dict(rec) for rec in obj]
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def from_json(text: str) -> Any:
    """Inverse of ``to_json`` for records: a JSON object yields one
    ``StageRecord``, a JSON array a list. Anything else (including an
    aggregate envelope, which is already a plain dict) is rejected —
    aggregates never re-enter as records."""
    doc = json.loads(text)
    if isinstance(doc, list):
        return [record_from_dict(d) for d in doc]
    if isinstance(doc, dict) and "kind" in doc:
        return record_from_dict(doc)
    raise ValueError("not a stage_profile/v7 record document")


__all__ = [
    "RECORD_FORMAT",
    "ProfiledCall",
    "aggregate",
    "from_json",
    "record_add",
    "record_from_dict",
    "record_search",
    "record_to_dict",
    "to_json",
]
