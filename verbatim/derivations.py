"""Materialized derivation graph: child → exact parent inputs (SPEC_V3 §06.06,
§17.02, §36.02, §39.07).

Every learning-plane object (claim interpretation, episode, transition,
procedure, observation, working item) is produced FROM exact evidence. The
``derivations`` table records one immutable edge per
``(child_kind, child_id, child_revision) → (parent_kind, parent_id,
parent_revision)`` pair, with producer identity and creation sequence
(V3-17.02). This module is the only writer/reader of that table — purge
closure (``privacy.deletion``), invalidation, influence tracing, and rollback
planning all traverse it here.

Conventions (docs/v3_contracts.md):

* Object references are ``(kind, object_id, revision)`` triples. ``revision``
  must be an ``int >= 1`` when recording edges; traversal APIs accept
  ``revision=None`` meaning "any revision of this object".
* Mutating helpers take the caller's transaction ``conn``; reads return
  plain tuples/dicts, never row objects.
* Edges are append-only: there is deliberately **no update path** — an
  incorrect edge is superseded by recording correct ancestry on the child's
  next revision, and physical edge removal happens only inside authorized
  deletion closure (V3-36.02).
* Traversals run as parameterized recursive CTEs (the seeds are the only
  bound values — no caller text ever becomes SQL structure) with a depth
  bound and a node bound, so a cyclic or pathological graph cannot turn a
  query into an unbounded scan (V3-39.07: forward, reverse, and
  time-bounded traversal with bounded fan-out and cycle protection).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from .core.types import ErrorCode, VerbatimError, require_id
from .storage import repos_v3

#: Revision-precise object reference: (kind, object_id, revision|None).
ObjectRef = tuple[str, str, Optional[int]]

#: Evidence-plane kinds — objects produced by capture, not by derivation
#: from other memory (§06, §12–§13). These are the natural roots of the
#: graph: everything else is derived.
EVIDENCE_KINDS = frozenset({
    "source",
    "source_revision",
    "span",
    "envelope",
    "artifact",
    "trajectory",
    "trajectory_step",
    "state_anchor",
})

#: Learning-plane kinds (§17). ``fact`` is accepted as an alias of ``claim``.
LEARNING_KINDS = frozenset({
    "claim",
    "episode",
    "transition",
    "procedure",
    "observation",
    "profile",
    "plan",
    "working",
    "social",
    "environment",
    "branch",
    "source_state",
    "memory_card",
})

_KIND_ALIASES = {"fact": "claim"}

#: Every kind a derivation edge may name. Unknown kinds are rejected at
#: write time so provenance can never dangle into an unresolvable table
#: (V3-17.01: kinds must be declared).
OBJECT_KINDS = EVIDENCE_KINDS | LEARNING_KINDS

#: Traversal bounds (V3-39.07). The depth bound is also what makes cyclic
#: graphs terminate: a cycle can only re-emit a node at a greater depth, so
#: recursion stops at ``_MAX_DEPTH``; the node bound caps total fan-out.
_MAX_DEPTH = 64
_MAX_NODES = 50_000


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ----------------------------------------------------------------------
# reference normalization
# ----------------------------------------------------------------------


def normalize_kind(kind: Any) -> str:
    """Validate a memory/evidence kind token; ``fact`` maps to ``claim``."""
    if not isinstance(kind, str):
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid object kind {kind!r}")
    k = _KIND_ALIASES.get(kind, kind)
    if k not in OBJECT_KINDS:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown object kind {kind!r}"
        )
    return k


def normalize_ref(item: Any) -> ObjectRef:
    """Normalize one object reference to ``(kind, id, revision|None)``.

    Accepts ``(kind, id)`` / ``(kind, id, revision)`` tuples/lists or dicts
    with ``kind``/``object_kind``, ``id``/``object_id``, and optional
    ``revision`` keys. ``revision=None`` means "any revision" and is legal
    for queries and revision-agnostic purge roots — never for ``record_edge``.
    """
    kind: Any
    oid: Any
    rev: Any = None
    if isinstance(item, dict):
        kind = item.get("kind", item.get("object_kind"))
        oid = item.get("id", item.get("object_id"))
        rev = item.get("revision", item.get("object_revision"))
    else:
        try:
            parts = tuple(item)
        except TypeError:
            parts = ()
        if len(parts) == 2:
            kind, oid = parts
        elif len(parts) == 3:
            kind, oid, rev = parts
        else:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "object refs must be (kind, id[, revision])",
            )
    k = normalize_kind(kind)
    require_id(str(oid), "object_id")
    if rev is not None:
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, "object revision must be an int >= 1"
            )
    return (k, str(oid), rev)


def _require_concrete(ref: ObjectRef, what: str) -> tuple[str, str, int]:
    kind, oid, rev = ref
    if rev is None:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{what} requires an explicit revision"
        )
    return (kind, oid, int(rev))


# ----------------------------------------------------------------------
# edge recording — immutable inserts (V3-17.02)
# ----------------------------------------------------------------------


def record_edge(
    conn: sqlite3.Connection,
    child: Any,
    parent: Any,
    producer_kind: str,
    producer_id: str,
    scope_id: str,
    seq: int,
) -> bool:
    """Append one child → parent derivation edge.

    Returns ``True`` when the edge was inserted, ``False`` when the exact
    edge already existed (job redelivery replay is a no-op — §40.01). An
    edge present with *different* producer or scope attribution is an
    INTEGRITY failure, never a silent overwrite: edges are immutable.
    """
    ck, cid, crev = _require_concrete(normalize_ref(child), "child")
    pk, pid, prev = _require_concrete(normalize_ref(parent), "parent")
    require_id(producer_kind, "producer_kind")
    require_id(producer_id, "producer_id")
    require_id(scope_id, "scope_id")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, "seq must be an int >= 0"
        )
    existing = repos_v3.get(
        conn,
        "derivations",
        {
            "child_kind": ck,
            "child_id": cid,
            "child_revision": crev,
            "parent_kind": pk,
            "parent_id": pid,
            "parent_revision": prev,
        },
    )
    if existing is not None:
        if (
            existing["producer_kind"] != producer_kind
            or existing["producer_id"] != producer_id
            or existing["scope_id"] != scope_id
        ):
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                "derivation edge exists with different producer/scope — "
                "edges are immutable (V3-17.02)",
            )
        return False
    repos_v3.insert(
        conn,
        "derivations",
        {
            "child_kind": ck,
            "child_id": cid,
            "child_revision": crev,
            "parent_kind": pk,
            "parent_id": pid,
            "parent_revision": prev,
            "producer_kind": producer_kind,
            "producer_id": producer_id,
            "seq": seq,
            "scope_id": scope_id,
        },
    )
    return True


# ----------------------------------------------------------------------
# one-hop lookups
# ----------------------------------------------------------------------


def _refs(rows: list[dict[str, Any]], prefix: str) -> list[tuple[str, str, int]]:
    out = [
        (r[f"{prefix}_kind"], r[f"{prefix}_id"], int(r[f"{prefix}_revision"]))
        for r in rows
    ]
    out.sort()
    return out


def parents_of(conn: sqlite3.Connection, child: Any) -> list[tuple[str, str, int]]:
    """Exact recorded inputs of one object revision (reverse traversal)."""
    kind, oid, rev = normalize_ref(child)
    where: dict[str, Any] = {"child_kind": kind, "child_id": oid}
    if rev is not None:
        where["child_revision"] = rev
    return _refs(repos_v3.query(conn, "derivations", where), "parent")


def children_of(conn: sqlite3.Connection, parent: Any) -> list[tuple[str, str, int]]:
    """Objects recorded as derived from this revision (forward traversal)."""
    kind, oid, rev = normalize_ref(parent)
    where: dict[str, Any] = {"parent_kind": kind, "parent_id": oid}
    if rev is not None:
        where["parent_revision"] = rev
    return _refs(repos_v3.query(conn, "derivations", where), "child")


def producers_of(conn: sqlite3.Connection, child: Any) -> list[dict[str, Any]]:
    """Producer identity per parent edge (V3-17.02), for receipts/audit."""
    kind, oid, rev = normalize_ref(child)
    where: dict[str, Any] = {"child_kind": kind, "child_id": oid}
    if rev is not None:
        where["child_revision"] = rev
    rows = repos_v3.query(conn, "derivations", where, order="seq")
    return [
        {
            "parent": (r["parent_kind"], r["parent_id"], int(r["parent_revision"])),
            "producer_kind": r["producer_kind"],
            "producer_id": r["producer_id"],
            "seq": int(r["seq"]),
            "scope_id": r["scope_id"],
        }
        for r in rows
    ]


# ----------------------------------------------------------------------
# recursive traversal — parameterized recursive CTEs (V3-39.07)
# ----------------------------------------------------------------------
#
# The seed set is a non-recursive CTE of bound parameters (one SELECT per
# seed); recursion joins derivations against the frontier. Depth rides in
# the CTE row, so UNION dedup alone cannot stop a cycle — each lap re-emits
# the node at a greater depth — but the ``w.depth < ?`` term guarantees
# termination, and ``_MAX_NODES`` caps the materialized result.


def _seed_cte(seeds: list[ObjectRef]) -> tuple[str, list[Any]]:
    """``seed(sk, si, sr)`` CTE text + params; ``sr`` NULL = any revision."""
    selects = " UNION ALL ".join("SELECT ?, ?, ?" for _ in seeds)
    params: list[Any] = []
    for k, i, r in seeds:
        params.extend([k, i, r])
    return f"seed(sk, si, sr) AS ({selects})", params


def _walk(
    conn: sqlite3.Connection,
    seeds_in: Iterable[Any],
    *,
    direction: str,
    max_depth: int,
) -> dict[tuple[str, str, int], int]:
    """BFS over derivations; returns ``{ref: min_depth}`` (excludes seeds).

    ``direction='down'`` follows parent→child edges (descendants);
    ``'up'`` follows child→parent edges (ancestors). A ``rev=None`` seed
    matches any revision of that object.
    """
    seeds = [normalize_ref(s) for s in seeds_in]
    if not seeds:
        return {}
    if direction == "down":
        frm, to = "parent", "child"
    elif direction == "up":
        frm, to = "child", "parent"
    else:  # pragma: no cover - internal guard
        raise VerbatimError(ErrorCode.VALIDATION, f"bad direction {direction!r}")
    if isinstance(max_depth, bool) or not (1 <= int(max_depth) <= _MAX_DEPTH):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"max_depth must be 1..{_MAX_DEPTH}"
        )
    seed_sql, params = _seed_cte(seeds)
    sql = (
        f"WITH RECURSIVE {seed_sql}, "
        f"walk(kind, id, rev, depth) AS ("
        f"  SELECT d.{to}_kind, d.{to}_id, d.{to}_revision, 1"
        f"  FROM derivations d JOIN seed s"
        f"    ON d.{frm}_kind = s.sk AND d.{frm}_id = s.si"
        f"   AND (s.sr IS NULL OR d.{frm}_revision = s.sr)"
        f"  UNION"
        f"  SELECT d.{to}_kind, d.{to}_id, d.{to}_revision, w.depth + 1"
        f"  FROM derivations d JOIN walk w"
        f"    ON d.{frm}_kind = w.kind AND d.{frm}_id = w.id"
        f"   AND d.{frm}_revision = w.rev"
        f"  WHERE w.depth < ?"
        f") "
        f"SELECT kind, id, rev, MIN(depth) AS depth FROM walk"
        f" GROUP BY kind, id, rev LIMIT ?"
    )
    rows = _rows(conn.execute(sql, [*params, int(max_depth), _MAX_NODES + 1]))
    if len(rows) > _MAX_NODES:
        raise VerbatimError(
            ErrorCode.VALIDATION, "derivation traversal exceeds node bound"
        )
    out: dict[tuple[str, str, int], int] = {}
    for r in rows:
        out[(r["kind"], r["id"], int(r["rev"]))] = int(r["depth"])
    return out


def ancestors_of(
    conn: sqlite3.Connection, ref: Any, *, max_depth: int = _MAX_DEPTH
) -> set[tuple[str, str, int]]:
    """All recorded inputs, transitively — the full provenance of ``ref``."""
    return set(_walk(conn, [ref], direction="up", max_depth=max_depth))


def descendants_of(
    conn: sqlite3.Connection, ref: Any, *, max_depth: int = _MAX_DEPTH
) -> dict[tuple[str, str, int], int]:
    """All objects derived from ``ref``, transitively — the impact set.

    Returns ``{ref: min_depth}`` where depth counts derivation hops from
    the seed; cycle-safe and bounded (V3-39.07).
    """
    return _walk(conn, [ref], direction="down", max_depth=max_depth)


def roots(conn: sqlite3.Connection, ref: Any) -> list[tuple[str, str, int]]:
    """Evidence-plane ancestors: frontier nodes with no recorded parents.

    These are the exact pieces of evidence a derived object stands on —
    typically ``source_revisions`` and ``spans`` (V3-17.04 links to exact
    evidence). An object with no recorded ancestry is its own root; an
    object with no edges at all returns itself as the sole root.
    """
    me = normalize_ref(ref)
    if me[2] is not None:
        seeds = [me]
    else:
        # Expand "any revision" to the revisions that appear in the graph —
        # as a derived child first, else as a parent (it is its own root).
        revs = {
            int(r[0])
            for r in conn.execute(
                "SELECT child_revision FROM derivations"
                " WHERE child_kind = ? AND child_id = ?"
                " UNION SELECT parent_revision FROM derivations"
                " WHERE parent_kind = ? AND parent_id = ?",
                (me[0], me[1], me[0], me[1]),
            ).fetchall()
        }
        seeds = [(me[0], me[1], r) for r in sorted(revs)]
        if not seeds:
            return [(me[0], me[1], 0)]
    seed_sql, params = _seed_cte(seeds)
    sql = (
        f"WITH RECURSIVE {seed_sql}, "
        f"walk(kind, id, rev, depth) AS ("
        f"  SELECT s.sk, s.si, s.sr, 0 FROM seed s"
        f"  UNION"
        f"  SELECT d.parent_kind, d.parent_id, d.parent_revision, w.depth + 1"
        f"  FROM derivations d JOIN walk w"
        f"    ON d.child_kind = w.kind AND d.child_id = w.id"
        f"   AND d.child_revision = w.rev"
        f"  WHERE w.depth < ?"
        f") "
        f"SELECT DISTINCT kind, id, rev FROM walk w"
        f" WHERE NOT EXISTS ("
        f"   SELECT 1 FROM derivations d"
        f"   WHERE d.child_kind = w.kind AND d.child_id = w.id"
        f"   AND d.child_revision = w.rev"
        f" ) ORDER BY kind, id, rev LIMIT ?"
    )
    rows = _rows(conn.execute(sql, [*params, _MAX_DEPTH, _MAX_NODES + 1]))
    if len(rows) > _MAX_NODES:
        raise VerbatimError(
            ErrorCode.VALIDATION, "derivation traversal exceeds node bound"
        )
    return [(r["kind"], r["id"], int(r["rev"])) for r in rows]


def affected_by_purge(
    conn: sqlite3.Connection, object_refs: Iterable[Any]
) -> dict[str, Any]:
    """Full descendant set needing closure when ``object_refs`` are purged.

    Returns ``{"roots": [...normalized inputs...], "derived": {ref: depth}}``.
    The derived map is deduplicated across seeds (diamond graphs collapse to
    one entry at the shallowest depth). A concrete seed that appears in the
    derived set reached itself through a cycle — the planner treats it as
    already-covered by the roots, not as a separate derived object.
    """
    seeds = [normalize_ref(r) for r in (object_refs or ())]
    derived = _walk(conn, seeds, direction="down", max_depth=_MAX_DEPTH)
    concrete_seeds = {s for s in seeds if s[2] is not None}
    for s in concrete_seeds:
        derived.pop(s, None)
    return {"roots": seeds, "derived": derived}


__all__ = [
    "EVIDENCE_KINDS",
    "LEARNING_KINDS",
    "OBJECT_KINDS",
    "ObjectRef",
    "affected_by_purge",
    "ancestors_of",
    "children_of",
    "descendants_of",
    "normalize_kind",
    "normalize_ref",
    "parents_of",
    "producers_of",
    "record_edge",
    "roots",
]
