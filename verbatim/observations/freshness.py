"""Freshness classes and anchor-based staleness (SPEC_V3 §24).

V3-17.08 / V3-24.03..24.08: every derived object carries a freshness class;
``volatile`` objects are anchored to ``environment_state`` rows, so when an
anchor moves (the environment row changes) the dependent objects are marked
stale at a recorded sequence. ``revalidate_after`` objects go stale when
their policy deadline passes. Staleness produces warnings and ``verify``
routing — never an inferred contradiction (V3-18.11).

The ``freshness`` row is the staleness ledger: ``anchor_refs_json`` stores
``[{"ref": <anchor_id>, "stale_since_seq": <int|null>}]`` so any object kind
can carry anchor-based staleness without schema changes. For
``observation`` objects the marker is additionally mirrored onto the
``observations.stale_since_seq`` column (§39), which retrieval reads.

Sequence numbers here are logical creation sequences — the same counter
domain used for ``derivations.seq`` and ``observations.recorded_from`` —
not wall-clock microseconds. ``current_seq``/``next_seq`` keep every
module in this package on one monotonic scale.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional, Tuple

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import FreshnessClass
from ..storage import repos_v3

# An object reference is the freshness primary key:
# (scope_id, object_kind, object_id, revision). Dicts with those keys are
# also accepted so callers can pass row snapshots through.
ObjectRef = Tuple[str, str, str, int]


def _ref_parts(object_ref: Any) -> ObjectRef:
    if isinstance(object_ref, dict):
        return (
            object_ref["scope_id"],
            object_ref["object_kind"],
            object_ref["object_id"],
            int(object_ref["revision"]),
        )
    scope_id, object_kind, object_id, revision = object_ref
    return scope_id, object_kind, object_id, int(revision)


# ---------------------------------------------------------------------------
# logical sequence helpers
# ---------------------------------------------------------------------------


def next_seq(conn: sqlite3.Connection, scope_id: str) -> int:
    """Next logical creation sequence for this scope.

    Draws from the same monotonic domains derivations and observations use —
    the global ``events`` counter plus per-scope derivation/stale markers —
    so a later write always lands on a strictly larger seq without needing
    an event row of its own.
    """
    require_id(scope_id, "scope_id")
    candidates = [
        conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM events"
        ).fetchone()[0],
        conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM derivations WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()[0],
        conn.execute(
            "SELECT COALESCE(MAX(recorded_from), 0) FROM observations"
            " WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()[0],
        conn.execute(
            "SELECT COALESCE(MAX(stale_since_seq), 0) FROM observations"
            " WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()[0],
    ]
    # Stale marks also live inside freshness.anchor_refs_json entries —
    # fold them into the seq domain so a mark never sits "in the future".
    for (raw,) in conn.execute(
        "SELECT anchor_refs_json FROM freshness WHERE scope_id = ?",
        (scope_id,),
    ).fetchall():
        for entry in decode_anchors(raw):
            stale = entry.get("stale_since_seq")
            if stale is not None:
                candidates.append(int(stale))
    return int(max(candidates)) + 1


def current_seq(conn: sqlite3.Connection, scope_id: str) -> int:
    """Largest seq already consumed in this scope (0 when nothing wrote)."""
    return next_seq(conn, scope_id) - 1


# ---------------------------------------------------------------------------
# anchor_refs_json payload
# ---------------------------------------------------------------------------


def _encode_anchors(anchor_refs: Iterable[Any]) -> str:
    entries = []
    for ref in anchor_refs:
        if isinstance(ref, dict):
            rid = require_id(ref.get("ref"), "anchor_ref")
            stale = ref.get("stale_since_seq")
            entries.append(
                {"ref": rid, "stale_since_seq": None if stale is None else int(stale)}
            )
        else:
            entries.append(
                {"ref": require_id(ref, "anchor_ref"), "stale_since_seq": None}
            )
    return json_dumps(entries)


def decode_anchors(raw: Any) -> list[dict[str, Any]]:
    """Decode ``anchor_refs_json`` into ``[{"ref", "stale_since_seq"}]``.

    Tolerates a plain list of strings (entries written by other producers).
    """
    if raw is None:
        return []
    val = raw if isinstance(raw, list) else safe_json_loads(raw)
    if not isinstance(val, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in val:
        if isinstance(entry, dict) and "ref" in entry:
            out.append(
                {
                    "ref": entry["ref"],
                    "stale_since_seq": entry.get("stale_since_seq"),
                }
            )
        elif isinstance(entry, str):
            out.append({"ref": entry, "stale_since_seq": None})
    return out


# ---------------------------------------------------------------------------
# freshness rows
# ---------------------------------------------------------------------------


def set_freshness(
    conn: sqlite3.Connection,
    object_ref: Any,
    freshness_class: Any,
    *,
    revalidate_after_us: Optional[int] = None,
    anchor_refs: Iterable[Any] = (),
) -> None:
    """Declare the freshness class of one object revision (§24).

    Upserts the ``freshness`` row keyed by (scope, kind, id, revision).
    ``anchor_refs`` binds ``volatile`` objects to environment anchors; each
    ref is an ``environment_state.anchor_id`` (or ``state_anchors`` id) whose
    movement marks this object stale. Passing ``anchor_refs`` replaces the
    ref list and clears prior stale markers — refreshing freshness is the
    documented way to clear a stale mark after revalidation (V3-24.07:
    retrieval alone never extends freshness; an explicit owner refresh does).
    """
    scope_id, object_kind, object_id, revision = _ref_parts(object_ref)
    klass = (
        freshness_class
        if isinstance(freshness_class, FreshnessClass)
        else FreshnessClass(freshness_class)
    )
    if klass == FreshnessClass.REVALIDATE_AFTER and revalidate_after_us is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "revalidate_after class requires revalidate_after_us",
        )
    where = {
        "scope_id": scope_id,
        "object_kind": object_kind,
        "object_id": object_id,
        "revision": revision,
    }
    payload = {
        "class": klass.value,
        "revalidate_after_us": revalidate_after_us,
        "anchor_refs_json": _encode_anchors(anchor_refs),
    }
    if repos_v3.get(conn, "freshness", where) is None:
        repos_v3.insert(conn, "freshness", {**where, **payload})
    else:
        repos_v3.update(conn, "freshness", payload, where)


def get_freshness(
    conn: sqlite3.Connection, object_ref: Any
) -> Optional[dict[str, Any]]:
    """Row snapshot for one object revision; anchors decoded into
    ``row["anchors"]`` as ``[{"ref", "stale_since_seq"}]``."""
    scope_id, object_kind, object_id, revision = _ref_parts(object_ref)
    row = repos_v3.get(
        conn,
        "freshness",
        {
            "scope_id": scope_id,
            "object_kind": object_kind,
            "object_id": object_id,
            "revision": revision,
        },
    )
    if row is None:
        return None
    row["anchors"] = decode_anchors(row.get("anchor_refs_json"))
    return row


def anchors_for(conn: sqlite3.Connection, object_ref: Any) -> list[str]:
    """Anchor ids this object revision is bound to."""
    row = get_freshness(conn, object_ref)
    if row is None:
        return []
    return [a["ref"] for a in row["anchors"]]


def mark_anchored_stale(
    conn: sqlite3.Connection,
    scope_id: str,
    anchor_ref: str,
    seq: Optional[int] = None,
) -> int:
    """Mark every object in ``scope_id`` anchored to ``anchor_ref`` stale.

    Called when an ``environment_state`` row changes: each freshness row
    whose anchor list contains the moved anchor records ``stale_since_seq``
    on that anchor entry (the earliest mark is kept — an object is stale
    *since* the first unhandled move). ``observation`` objects additionally
    mirror the marker onto ``observations.stale_since_seq`` (V3-23.03).
    Returns the number of object revisions marked.
    """
    require_id(scope_id, "scope_id")
    require_id(anchor_ref, "anchor_ref")
    if seq is None:
        seq = next_seq(conn, scope_id)
    marked = 0
    for row in repos_v3.query(conn, "freshness", {"scope_id": scope_id}):
        anchors = decode_anchors(row.get("anchor_refs_json"))
        hit = False
        for entry in anchors:
            if entry["ref"] != anchor_ref:
                continue
            hit = True
            prior = entry["stale_since_seq"]
            if prior is None or seq < int(prior):
                entry["stale_since_seq"] = seq
        if not hit:
            continue
        repos_v3.update(
            conn,
            "freshness",
            {"anchor_refs_json": json_dumps(anchors)},
            {
                "scope_id": row["scope_id"],
                "object_kind": row["object_kind"],
                "object_id": row["object_id"],
                "revision": row["revision"],
            },
        )
        if row["object_kind"] == "observation":
            conn.execute(
                "UPDATE observations SET stale_since_seq = ?"
                " WHERE observation_id = ? AND revision = ?"
                " AND (stale_since_seq IS NULL OR stale_since_seq > ?)",
                (seq, row["object_id"], row["revision"], seq),
            )
        marked += 1
    return marked


def is_stale(
    conn: sqlite3.Connection,
    object_ref: Any,
    current_seq_value: int,
    *,
    now: Optional[int] = None,
) -> bool:
    """Whether the object revision is stale at ``current_seq_value``.

    True when any anchor entry carries ``stale_since_seq <= current_seq``,
    when a ``revalidate_after`` deadline has passed, or — for observations —
    the row's own ``stale_since_seq`` is set at or before ``current_seq``.
    A ``volatile`` class alone is not stale: it is delivered with a
    ``verify`` recommendation until an anchor actually moves (§24 table).
    Objects with no freshness row are not stale (nothing marked them).
    """
    row = get_freshness(conn, object_ref)
    if row is None:
        return False
    if row.get("class") == FreshnessClass.REVALIDATE_AFTER.value:
        deadline = row.get("revalidate_after_us")
        if deadline is not None and (now if now is not None else now_us()) >= int(
            deadline
        ):
            return True
    for entry in row["anchors"]:
        stale = entry.get("stale_since_seq")
        if stale is not None and int(stale) <= int(current_seq_value):
            return True
    if row["object_kind"] == "observation":
        obs = repos_v3.get(
            conn, "observations", {"observation_id": row["object_id"]}
        )
        if obs is not None and obs.get("stale_since_seq") is not None:
            return int(obs["stale_since_seq"]) <= int(current_seq_value)
    return False
