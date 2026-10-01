"""Typed graph-edge derivation for the V7 unit graph (V7-08.07, V7-08.10).

``build_edges`` is the job handler behind the graph projection: given the
units added (or re-projected) at a generation it derives the typed,
weighted, generation-fenced edge rows of §30 ``graph_edges`` and returns
the number of rows written. All derivation is deterministic T0 work —
no models, no inference beyond the explicit connective lexicon.

Edge families (V7-08.07):

* ``co_mention`` — units sharing an entity canon; weight factor is the
  inverse canon df (``1/df`` over the scope's member units).
* ``same_session`` — units sharing ``session_id``; bounded fan-out.
* ``adjacent_turn`` — session-mates with ``|Δseq| ∈ {1, 2}`` (t±1, t±2).
* ``temporal_near`` — occurred intervals within the declared window
  ``TEMPORAL_NEAR_WINDOW_US_V1``; weight factor decays linearly with the
  inter-interval gap (1.0 = overlapping).
* ``causal`` — ONLY explicit connectives (``because``, ``so``, ``due to``,
  ``which is why``, ``therefore``, ``as a result``) with BOTH clauses
  byte-pinned into the unit's source bytes; never inferred. Direction is
  cause-unit → effect-unit; a self-contained statement produces a
  self-loop (``self_loop: true`` in evidence). Cross-unit resolution is a
  literal (folded) substring match of the cause clause against units
  already in the build context — nothing weaker is accepted.
* ``supersedes`` / ``contradicts`` / ``refines`` — passthrough hooks:
  rows supplied by the update machinery via ``write_supplied_edges`` or
  a unit row's ``relations`` list; never derived here.
* ``semantic_knn`` — precomputed neighbour lists accepted via a unit
  row's ``knn`` list or ``write_supplied_edges`` (k ≤ 8 enforced, floor
  ``KNN_SIM_FLOOR_V1``); the producing job lands with the dense lane.
* ``causal_candidate`` — model-proposed rows accepted with mandatory
  quote pins for both clauses and stamped ``eligible_input: false``
  (V7-08.14): stored for later ablation, never a traversal or
  eligibility input.

Stored ``weight`` is the *type-local factor* (e.g. ``1/df`` for
``co_mention``, similarity for ``semantic_knn``); the lane multiplies by
the versioned per-type weight (``EDGE_WEIGHTS_V1``, §32.6). Edges between
the same ``(src, type, dst)`` merge by summing factors and unioning
evidence contributors.

Incremental rebuild (V7-08.10): only pairs incident to the supplied units
are recomputed — derived-type edges touching a rebuilt unit are deleted
and re-emitted inside the same transaction; supplied-type rows are left
to their owning machinery (a unit's ``knn`` list replaces only that
unit's outgoing ``semantic_knn`` rows). Existing edges between untouched
units are not renormalized when a new member enters a bounded fan-out
window — that renormalization is deferred to a full rebuild pass
(documented limitation of the ``graph_edges/v1`` incremental policy).

Mirror-DDL note: this module intentionally looks up ``units``,
``entity_mentions``, ``entity_canon`` and ``source_revisions`` by name
through ``has_table`` so the wave-A mirror DDL in
``tests/retrieval/v7/test_graph.py`` works before ``schema_v7`` lands;
swap to ``schema_v7.ensure_v7_additive`` at integration (V7-30.01).
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps
from ..core.types_v7 import EdgeType
from ..storage.repos import has_table

GRAPH_EDGES_VERSION = "graph_edges/v1"
FORMULA_STATUS = "provisional/v7-r0"

# ---- declared fan-out / window policy (provisional/v7-r0) -----------------

CO_MENTION_NEIGHBORS_V1 = 8        # banded-clique radius per canon
SAME_SESSION_NEIGHBORS_V1 = 8      # banded-clique radius per session
TEMPORAL_NEAR_NEIGHBORS_V1 = 8     # nearest-in-time partners per unit
TEMPORAL_NEAR_WINDOW_US_V1 = 48 * 3600 * 1_000_000  # declared window: ±48 h
ADJACENT_TURN_RADIUS_V1 = 2        # t±1, t±2 (V7-08.07)
KNN_MAX_V1 = 8                     # semantic_knn top-k bound (V7-08.07)
KNN_SIM_FLOOR_V1 = 0.0             # calibrated floor placeholder — producer-owned
CAUSAL_MAX_PER_UNIT_V1 = 8         # connective statements per unit
CAUSAL_DST_MAX_V1 = 4              # cross-unit cause resolutions per statement
CAUSAL_CONTEXT_SCAN_CAP_V1 = 64    # candidate texts scanned per cause clause
MIN_CAUSE_MATCH_CHARS_V1 = 3       # alnum floor for cross-unit clause match

DERIVED_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.CO_MENTION,
    EdgeType.SAME_SESSION,
    EdgeType.ADJACENT_TURN,
    EdgeType.TEMPORAL_NEAR,
    EdgeType.CAUSAL,
)
SUPPLIED_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.SUPERSEDES,
    EdgeType.CONTRADICTS,
    EdgeType.REFINES,
    EdgeType.SEMANTIC_KNN,
    EdgeType.CAUSAL_CANDIDATE,
)
# Relevance-symmetric relations: stored once per unordered pair, canonical
# (lower unit_id) → (higher unit_id). The lane traverses every declared
# type bidirectionally regardless.
SYMMETRIC_EDGE_TYPES = frozenset(
    {
        EdgeType.CO_MENTION,
        EdgeType.SAME_SESSION,
        EdgeType.ADJACENT_TURN,
        EdgeType.TEMPORAL_NEAR,
        EdgeType.SEMANTIC_KNN,
    }
)

_UNIT_COLS = (
    "unit_id",
    "source_id",
    "revision",
    "scope_id",
    "kind",
    "parent_unit_id",
    "session_id",
    "seq",
    "speaker_canon",
    "perspective",
    "recorded_at_us",
    "occurred_start_us",
    "occurred_end_us",
    "occurred_precision",
    "occurred_source",
    "byte_start",
    "byte_end",
    "generation",
)

# ---- causal connective lexicon (V7-08.07 — explicit only, never inferred) -
#
# (rule_id, pattern, cause_side): ``cause_side`` locates the cause clause
# relative to the connective. Longest forms are tried first so
# "as a result of" wins over "as a result".
_CAUSAL_RULES: tuple[tuple[str, "re.Pattern[str]", str], ...] = (
    (
        "which is why",
        re.compile(r"\bwhich\s+is\s+why\b", re.IGNORECASE),
        "before",
    ),
    (
        "as a result of",
        re.compile(r"\bas\s+a\s+result\s+of\b", re.IGNORECASE),
        "after",
    ),
    (
        "as a result",
        re.compile(r"\bas\s+a\s+result\b", re.IGNORECASE),
        "before",
    ),
    ("because", re.compile(r"\bbecause\b", re.IGNORECASE), "after"),
    ("due to", re.compile(r"\bdue\s+to\b", re.IGNORECASE), "after"),
    ("therefore", re.compile(r"\btherefore\b", re.IGNORECASE), "before"),
    (
        "so",
        re.compile(
            r"\bso\b(?!\s+(?:that|this|much|many|few|little|far|long|often)\b)",
            re.IGNORECASE,
        ),
        "before",
    ),
)

# Clause delimiters: sentence/clause punctuation and hard breaks. Quotes,
# apostrophes and hyphens are deliberately absent — they occur inside
# tokens and must not cut a clause mid-word.
_CLAUSE_BREAK = re.compile(r"[.!?,;:\n\r\t()\[\]{}\"“”—–]")
# Strip chars around a clause span include sentence terminators: in
# "It rained. Therefore, we stayed" the '.' is the boundary the left
# clause ends at, not part of it.
_STRIP_CHARS = " \t,;:.!?—–-"
_ALNUM = re.compile(r"[A-Za-z0-9]")
_FUNC_LEAD = re.compile(
    r"^(?:of|the|a|an|to|for|on|in|that|this|it|its|his|her|their|our|my|your)\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------------


def _coerce_row(item: Any) -> Optional[dict]:
    """Accept ``dict``/``sqlite3.Row`` unit rows; a bare ``str`` is a unit_id
    to be loaded from the ``units`` table (the ``unit_ids`` contract form)."""
    if isinstance(item, str):
        return {"unit_id": item}
    if isinstance(item, sqlite3.Row):
        return {k: item[k] for k in item.keys()}
    if isinstance(item, Mapping):
        return dict(item)
    raise VerbatimError(
        ErrorCode.VALIDATION, f"unit_rows item must be mapping or id, got {type(item)}"
    )


def _rows_dicts(cur: sqlite3.Cursor) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _in_clause(name: str, ids: list[str]) -> tuple[str, list[str]]:
    return f"{name} IN ({','.join('?' for _ in ids)})", list(ids)


def _load_units(
    conn: sqlite3.Connection, scope_id: str, generation: int, unit_ids: Iterable[str]
) -> dict[str, dict]:
    """Load ``units`` rows (generation-fenced) keyed by unit_id. Absent table
    → empty map (callers fall back to provided rows only)."""
    ids = sorted(set(unit_ids))
    if not ids or not has_table(conn, "units"):
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        where, params = _in_clause("unit_id", chunk)
        cur = conn.execute(
            f"SELECT {','.join(_UNIT_COLS)} FROM units"
            f" WHERE scope_id=? AND generation<=? AND {where}"
            f" ORDER BY generation",
            (scope_id, generation, *params),
        )
        for row in _rows_dicts(cur):
            # (unit_id, generation) PK: ascending scan ⇒ the last row per
            # unit is its latest visible projection (V7-30.02).
            out[row["unit_id"]] = row
    return out


def _unit_text(conn: sqlite3.Connection, rec: Mapping[str, Any]) -> Optional[str]:
    """The unit's own text: an explicit ``text`` key wins; otherwise the
    byte-pinned slice of ``source_revisions.payload`` (§30 units carry
    ``byte_start``/``byte_end`` into source bytes). Strict UTF-8; undecodable
    payloads yield ``None`` — the unit simply derives no causal edges."""
    text = rec.get("text")
    if isinstance(text, str):
        return text
    src = rec.get("source_id")
    rev = rec.get("revision")
    bs = rec.get("byte_start")
    be = rec.get("byte_end")
    if (
        src is None
        or rev is None
        or bs is None
        or be is None
        or not has_table(conn, "source_revisions")
    ):
        return None
    row = conn.execute(
        "SELECT payload FROM source_revisions WHERE source_id=? AND revision=?",
        (src, rev),
    ).fetchone()
    if row is None:
        return None
    payload = row[0]
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    try:
        return bytes(payload)[int(bs) : int(be)].decode("utf-8")
    except (UnicodeDecodeError, ValueError, TypeError):
        return None


def _merge_db_fields(rec: dict, db_row: Optional[Mapping[str, Any]]) -> dict:
    """Provided rows are authoritative; absent keys backfill from the
    persisted ``units`` row (explicit ``None`` stays ``None``)."""
    if db_row is None:
        return rec
    for col in _UNIT_COLS:
        if col not in rec:
            rec[col] = db_row.get(col)
    return rec


# ---------------------------------------------------------------------------
# Causal extraction — explicit connectives with both clauses byte-pinned
# ---------------------------------------------------------------------------


def _clause_bounds(text: str, start: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Clause spans around the connective occupying ``[start, end)``.

    The left clause ends at the nearest delimiter before the connective's
    left context (the delimiter itself separates clause from connective);
    both sides are whitespace/punctuation-trimmed back to real text.
    """
    left_end = start
    while left_end > 0 and text[left_end - 1] in _STRIP_CHARS:
        left_end -= 1
    left_start = 0
    m = None
    for m in _CLAUSE_BREAK.finditer(text, 0, left_end):
        pass
    if m is not None:
        left_start = m.end()

    right_start = end
    while right_start < len(text) and text[right_start] in _STRIP_CHARS:
        right_start += 1
    m2 = _CLAUSE_BREAK.search(text, right_start)
    right_end = m2.start() if m2 else len(text)

    return (left_start, left_end), (right_start, right_end)


def _nonempty_clause(text: str, span: tuple[int, int]) -> bool:
    frag = text[span[0] : span[1]].strip(_STRIP_CHARS)
    return bool(frag) and bool(_ALNUM.search(frag))


def _trim_span(text: str, span: tuple[int, int]) -> tuple[int, int]:
    a, b = span
    while a < b and text[a] in _STRIP_CHARS:
        a += 1
    while b > a and text[b - 1] in _STRIP_CHARS:
        b -= 1
    return a, b


def _extract_causals(text: str) -> list[dict]:
    """Explicit-connective causal statements in ``text``.

    Returns up to ``CAUSAL_MAX_PER_UNIT_V1`` dicts with ``connective``,
    char offsets of both clause spans, and which side the cause sits on.
    Both clause spans must contain alphanumeric content — a connective
    with an empty side (e.g. sentence-initial "So …") yields nothing.
    """
    found: list[tuple[int, int, str, str]] = []  # start,end,rule,cause_side
    for rule, pat, side in _CAUSAL_RULES:
        for m in pat.finditer(text):
            found.append((m.start(), m.end(), rule, side))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, str, str]] = []
    for item in found:
        if all(item[0] >= k[1] or item[1] <= k[0] for k in kept):
            kept.append(item)
        if len(kept) >= CAUSAL_MAX_PER_UNIT_V1:
            break
    out = []
    for start, end, rule, side in kept:
        left, right = _clause_bounds(text, start, end)
        left, right = _trim_span(text, left), _trim_span(text, right)
        if not (_nonempty_clause(text, left) and _nonempty_clause(text, right)):
            continue
        cause_span, effect_span = (right, left) if side == "after" else (left, right)
        out.append(
            {
                "connective": rule,
                "conn_start": start,
                "conn_end": end,
                "effect_span": effect_span,
                "cause_span": cause_span,
            }
        )
    return out


def _byte_off(text: str, char_idx: int) -> int:
    return len(text[:char_idx].encode("utf-8"))


def _clause_variants(clause: str) -> list[str]:
    """Literal-match candidates for a cause clause: the clause itself plus
    forms with leading function words stripped ("of the promotion news" →
    "the promotion news" → "promotion news"). Every variant is matched as
    an exact folded substring — this is lexical lookup, not inference."""
    clause = clause.strip(_STRIP_CHARS)
    variants = [clause]
    rest = clause
    while True:
        m = _FUNC_LEAD.match(rest)
        if not m:
            break
        rest = rest[m.end() :].strip(_STRIP_CHARS)
        if rest:
            variants.append(rest)
    # longest first; dedupe preserving order
    seen: list[str] = []
    for v in variants:
        if v and v not in seen:
            seen.append(v)
    return seen


def _pin(rec: Mapping[str, Any], text: str, span: tuple[int, int]) -> dict:
    """A clause pin in source-byte coordinates (unit.byte_start + relative
    byte offset when the unit's byte range is known)."""
    a, b = _trim_span(text, span)
    rel_a, rel_b = _byte_off(text, a), _byte_off(text, b)
    base = rec.get("byte_start")
    pin = {
        "unit_id": rec["unit_id"],
        "byte_start": (base + rel_a) if isinstance(base, int) else rel_a,
        "byte_end": (base + rel_b) if isinstance(base, int) else rel_b,
        "text": text[a:b],
    }
    if not isinstance(base, int):
        pin["relative"] = True
    return pin


# ---------------------------------------------------------------------------
# Edge accumulation
# ---------------------------------------------------------------------------


def _edge_key(src: str, etype: EdgeType, dst: str) -> tuple[str, str, str]:
    if etype in SYMMETRIC_EDGE_TYPES and dst < src:
        src, dst = dst, src
    return (src, etype.value, dst)


def _merge_edge(
    edges: dict,
    src: str,
    dst: str,
    etype: EdgeType,
    weight: float,
    evidence: dict,
    contributor: Optional[str] = None,
) -> None:
    """Merge rule for the ``(src, type, dst)`` primary key: contributions
    carrying a ``contributor`` (co_mention canons) SUM factors and union
    contributors; everything else keeps max weight and preserves displaced
    evidence under ``also`` so no pin is silently dropped."""
    if src == dst and etype in SYMMETRIC_EDGE_TYPES:
        return  # self-loops are only meaningful for directed causal rows
    key = _edge_key(src, etype, dst)
    ent = edges.get(key)
    if ent is None:
        evidence = dict(evidence)
        evidence["rule"] = GRAPH_EDGES_VERSION
        edges[key] = {"w": float(weight), "ev": evidence, "contrib": set()}
        ent = edges[key]
        if contributor is not None:
            ent["contrib"].add(contributor)
    elif contributor is not None:
        ent["w"] += float(weight)  # shared-canon factors sum
        ent["contrib"].add(contributor)
    else:
        ent["w"] = max(ent["w"], float(weight))
        also = ent["ev"].setdefault("also", [])
        if len(also) < 8:
            also.append(evidence)


# ---------------------------------------------------------------------------
# Context loading for incremental pairing
# ---------------------------------------------------------------------------


def _canon_map(
    conn: sqlite3.Connection, scope_id: str, generation: int, recs: dict[str, dict]
) -> tuple[dict[str, set], dict[str, set]]:
    """``unit_id -> canons`` and ``canon -> member unit_ids`` for the units
    sharing a canon with any new unit. Sources: ``entity_mentions`` (when
    present) merged with explicit ``canons`` keys on provided rows."""
    unit_canons: dict[str, set] = {
        uid: set(r.get("canons") or ()) for uid, r in recs.items()
    }
    if has_table(conn, "entity_mentions"):
        ids = sorted(recs)
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            where, params = _in_clause("unit_id", chunk)
            for row in conn.execute(
                f"SELECT canon, unit_id FROM entity_mentions"
                f" WHERE scope_id=? AND generation<=? AND {where}",
                (scope_id, generation, *params),
            ):
                unit_canons.setdefault(row[1], set()).add(row[0])
    canons = sorted({c for cs in unit_canons.values() for c in cs})
    canon_members: dict[str, set] = {c: set() for c in canons}
    for uid, cs in unit_canons.items():
        for c in cs:
            canon_members[c].add(uid)
    if canons and has_table(conn, "entity_mentions"):
        for i in range(0, len(canons), 400):
            chunk = canons[i : i + 400]
            where, params = _in_clause("canon", chunk)
            for row in conn.execute(
                f"SELECT canon, unit_id FROM entity_mentions"
                f" WHERE scope_id=? AND generation<=? AND {where}",
                (scope_id, generation, *params),
            ):
                canon_members.setdefault(row[0], set()).add(row[1])
    return unit_canons, canon_members


def _session_mates(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    session_ids: Iterable[str],
) -> dict[str, dict]:
    sids = sorted(s for s in set(session_ids) if s)
    if not sids or not has_table(conn, "units"):
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(sids), 200):
        chunk = sids[i : i + 200]
        where, params = _in_clause("session_id", chunk)
        cur = conn.execute(
            f"SELECT {','.join(_UNIT_COLS)} FROM units"
            f" WHERE scope_id=? AND generation<=? AND {where}"
            f" ORDER BY generation",
            (scope_id, generation, *params),
        )
        for row in _rows_dicts(cur):
            out[row["unit_id"]] = row
    return out


def _temporal_neighborhood(
    conn: sqlite3.Connection, scope_id: str, generation: int, recs: dict[str, dict]
) -> dict[str, dict]:
    """Units whose occurred interval sits within the declared window of any
    new unit's interval. Unknown intervals never participate."""
    out: dict[str, dict] = {}
    if not has_table(conn, "units"):
        return out
    w = TEMPORAL_NEAR_WINDOW_US_V1
    for rec in recs.values():
        s, e = rec.get("occurred_start_us"), rec.get("occurred_end_us")
        if s is None or e is None:
            continue
        cur = conn.execute(
            f"SELECT {','.join(_UNIT_COLS)} FROM units WHERE scope_id=?"
            f" AND generation<=? AND occurred_start_us IS NOT NULL"
            f" AND occurred_end_us IS NOT NULL"
            f" AND occurred_start_us<=? AND occurred_end_us>=?"
            f" ORDER BY generation",
            (scope_id, generation, e + w, s - w),
        )
        for row in _rows_dicts(cur):
            out[row["unit_id"]] = row
    return out


# ---------------------------------------------------------------------------
# Supplied edges — update machinery / knn job / model candidates
# ---------------------------------------------------------------------------


def _norm_supplied(
    row: Mapping[str, Any], default_src: Optional[str]
) -> dict:
    try:
        etype = EdgeType(row["type"])
    except (KeyError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"supplied edge bad type: {row.get('type')!r}"
        ) from exc
    if etype in DERIVED_EDGE_TYPES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"edge type {etype.value} is T0-derived; supply via unit rows",
        )
    if etype not in SUPPLIED_EDGE_TYPES:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown edge type {etype.value}")
    src = row.get("src_unit") or row.get("src") or default_src
    dst = row.get("dst_unit") or row.get("dst")
    if not src or not dst:
        raise VerbatimError(
            ErrorCode.VALIDATION, "supplied edge needs src_unit and dst_unit"
        )
    ev = row.get("evidence") or row.get("evidence_ref") or {}
    if isinstance(ev, str):
        ev = {"note": ev}
    if not isinstance(ev, Mapping):
        raise VerbatimError(ErrorCode.VALIDATION, "supplied edge evidence not a mapping")
    ev = dict(ev)
    if etype is EdgeType.CAUSAL_CANDIDATE:
        # V7-08.14: stored only with quote pins for BOTH clauses; flagged
        # non-eligibility so no consumer mistakes it for the T0 floor.
        pins = ev.get("pins") if isinstance(ev.get("pins"), Mapping) else ev
        cause, effect = pins.get("cause"), pins.get("effect")
        for name, p in (("cause", cause), ("effect", effect)):
            if not (
                isinstance(p, Mapping)
                and isinstance(p.get("byte_start"), int)
                and isinstance(p.get("byte_end"), int)
                and p["byte_end"] > p["byte_start"]
            ):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"causal_candidate requires pinned {name} clause bytes",
                )
        ev["eligible_input"] = False
    elif etype is EdgeType.SEMANTIC_KNN:
        sim = row.get("similarity", row.get("weight"))
        if sim is None or not isinstance(sim, (int, float)):
            raise VerbatimError(
                ErrorCode.VALIDATION, "semantic_knn edge needs a similarity/weight"
            )
        if float(sim) < KNN_SIM_FLOOR_V1:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"semantic_knn similarity {sim} below floor {KNN_SIM_FLOOR_V1}",
            )
        ev.setdefault("similarity", float(sim))
        ev.setdefault("floor", KNN_SIM_FLOOR_V1)
        return {
            "src": str(src),
            "dst": str(dst),
            "type": etype,
            "weight": float(sim),
            "evidence": ev,
        }
    ev["supplied"] = True
    w = row.get("weight", 1.0)
    if not isinstance(w, (int, float)):
        raise VerbatimError(ErrorCode.VALIDATION, "supplied edge weight not numeric")
    return {
        "src": str(src),
        "dst": str(dst),
        "type": etype,
        "weight": float(w),
        "evidence": ev,
    }


def _insert_edges(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    edges: dict[tuple[str, str, str], dict],
) -> int:
    n = 0
    for (src, type_v, dst) in sorted(edges):
        ent = edges[(src, type_v, dst)]
        ev = dict(ent["ev"])
        if ent["contrib"]:
            ev["contributors"] = sorted(ent["contrib"])
        conn.execute(
            "INSERT OR REPLACE INTO graph_edges"
            " (scope_id, src_unit, type, dst_unit, weight, evidence_ref, generation)"
            " VALUES (?,?,?,?,?,?,?)",
            (scope_id, src, type_v, dst, ent["w"], json_dumps(ev), generation),
        )
        n += 1
    return n


def write_supplied_edges(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    """Passthrough hook for update machinery: ``supersedes``/``contradicts``/
    ``refines`` relations, precomputed ``semantic_knn`` lists, and pinned
    ``causal_candidate`` rows. Derived T0 types are rejected — they are
    computed, never supplied. Rows upsert on the §30 primary key."""
    edges: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        n = _norm_supplied(row, default_src=None)
        _merge_edge(
            edges, n["src"], n["dst"], n["type"], n["weight"], n["evidence"]
        )
    return _insert_edges(conn, scope_id, generation, edges)


# ---------------------------------------------------------------------------
# The job handler
# ---------------------------------------------------------------------------


def build_edges(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_rows: Iterable[Any],
) -> int:
    """Derive typed graph edges for ``unit_rows`` at ``generation``.

    ``unit_rows`` items are ``units``-shaped mappings (§30 columns), or bare
    unit_id strings resolved against the ``units`` table. Optional extra
    keys per row: ``text`` (the unit's own text for causal extraction —
    else sliced from ``source_revisions.payload``), ``canons`` (entity
    canons — else read from ``entity_mentions``), ``knn`` (precomputed
    ``(dst, similarity)`` neighbours, top-8 kept), and ``relations``
    (supplied edge dicts; the row's unit is the default ``src``).

    Returns the number of edge rows written. Derived-type rows incident to
    the supplied units are replaced; supplied-type rows upsert.
    """
    recs: dict[str, dict] = {}
    supplied_rows: list[tuple[str, Mapping[str, Any]]] = []
    knn_replace: set[str] = set()
    for item in unit_rows:
        row = _coerce_row(item)
        assert row is not None
        uid = row.get("unit_id")
        if not uid:
            raise VerbatimError(ErrorCode.VALIDATION, "unit row lacks unit_id")
        rec = dict(row)
        for extra in ("canons", "text", "knn", "relations"):
            rec.setdefault(extra, row.get(extra))
        recs[str(uid)] = rec
        for rel in row.get("relations") or ():
            supplied_rows.append((str(uid), rel))
        if row.get("knn") is not None:
            knn_replace.add(str(uid))

    # Backfill absent fields from the persisted units rows, then fold in any
    # persisted units the caller referenced only by id.
    db = _load_units(conn, scope_id, generation, recs.keys())
    for uid, rec in recs.items():
        _merge_db_fields(rec, db.get(uid))

    written = 0
    if recs:
        new_ids = sorted(recs)
        # Derived-type edges incident to the rebuilt units are replaced
        # inside this transaction (incremental rebuild, V7-08.10).
        derived = tuple(t.value for t in DERIVED_EDGE_TYPES)
        for i in range(0, len(new_ids), 300):
            chunk = new_ids[i : i + 300]
            # both endpoints, chunked once for src and once for dst
            conn.execute(
                f"DELETE FROM graph_edges WHERE scope_id=?"
                f" AND type IN ({','.join('?' for _ in derived)})"
                f" AND (src_unit IN ({','.join('?' for _ in chunk)})"
                f"      OR dst_unit IN ({','.join('?' for _ in chunk)}))",
                (scope_id, *derived, *chunk, *chunk),
            )
        if knn_replace:
            for uid in sorted(knn_replace):
                conn.execute(
                    "DELETE FROM graph_edges WHERE scope_id=?"
                    " AND type=? AND src_unit=?",
                    (scope_id, EdgeType.SEMANTIC_KNN.value, uid),
                )

        edges: dict[tuple[str, str, str], dict] = {}

        # --- context: partners needed for pair derivation -----------------
        unit_canons, canon_members = _canon_map(
            conn, scope_id, generation, recs
        )
        sessions = {
            r.get("session_id") for r in recs.values() if r.get("session_id")
        }
        ctx: dict[str, dict] = {}
        ctx.update(
            _session_mates(conn, scope_id, generation, sessions)
        )
        ctx.update(_temporal_neighborhood(conn, scope_id, generation, recs))
        member_ids = {m for ms in canon_members.values() for m in ms}
        ctx.update(
            _load_units(
                conn, scope_id, generation, member_ids - set(ctx) - set(recs)
            )
        )
        # provided rows are authoritative members of the context
        all_recs = dict(ctx)
        all_recs.update(recs)

        # --- co_mention: banded cliques per shared canon -------------------
        for canon in sorted(canon_members):
            members = canon_members[canon]
            if not members & set(recs):
                continue
            df = len(members)
            if df < 2:
                continue
            factor = 1.0 / df  # inverse canon df (V7-08.07)
            ordered = sorted(
                members,
                key=lambda u: (
                    all_recs.get(u, {}).get("recorded_at_us") or 0,
                    u,
                ),
            )
            for i, a in enumerate(ordered):
                for j in range(i + 1, min(i + 1 + CO_MENTION_NEIGHBORS_V1, df)):
                    b = ordered[j]
                    if a not in recs and b not in recs:
                        continue  # incremental: emit only new-incident pairs
                    _merge_edge(
                        edges,
                        a,
                        b,
                        EdgeType.CO_MENTION,
                        factor,
                        {"canon": canon, "df": df},
                        contributor=canon,
                    )

        # --- session families ---------------------------------------------
        for sid in sorted(sessions):
            members = [
                r for r in all_recs.values() if r.get("session_id") == sid
            ]
            if not any(m["unit_id"] in recs for m in members):
                continue
            members.sort(
                key=lambda r: (
                    r.get("seq") is None,
                    r.get("seq") or 0,
                    r["unit_id"],
                )
            )
            seq_of = {m["unit_id"]: m.get("seq") for m in members}
            for i, a in enumerate(members):
                aid = a["unit_id"]
                for j in range(i + 1, len(members)):
                    b = members[j]
                    bid = b["unit_id"]
                    if aid not in recs and bid not in recs:
                        continue
                    band = j - i
                    if band <= SAME_SESSION_NEIGHBORS_V1:
                        _merge_edge(
                            edges,
                            aid,
                            bid,
                            EdgeType.SAME_SESSION,
                            1.0,
                            {"session_id": sid},
                        )
                    sa, sb = seq_of[aid], seq_of[bid]
                    if (
                        sa is not None
                        and sb is not None
                        and 1 <= abs(sb - sa) <= ADJACENT_TURN_RADIUS_V1
                    ):
                        _merge_edge(
                            edges,
                            aid,
                            bid,
                            EdgeType.ADJACENT_TURN,
                            1.0,
                            {
                                "session_id": sid,
                                "seq_a": min(sa, sb),
                                "seq_b": max(sa, sb),
                                "distance": abs(sb - sa),
                            },
                        )

        # --- temporal_near --------------------------------------------------
        w = TEMPORAL_NEAR_WINDOW_US_V1
        seen_pairs: set[tuple[str, str]] = set()
        for uid in new_ids:
            rec = recs[uid]
            s, e = rec.get("occurred_start_us"), rec.get("occurred_end_us")
            if s is None or e is None:
                continue
            cands = []
            for other, orec in all_recs.items():
                if other == uid:
                    continue
                os_, oe = orec.get("occurred_start_us"), orec.get(
                    "occurred_end_us"
                )
                if os_ is None or oe is None:
                    continue
                gap = max(0, max(s, os_) - min(e, oe))
                if gap <= w:
                    cands.append((gap, other))
            cands.sort(key=lambda t: (t[0], t[1]))
            for gap, other in cands[:TEMPORAL_NEAR_NEIGHBORS_V1]:
                # New↔new pairs surface in both endpoints' scans (either
                # side's K-window suffices); emit each unordered pair once
                # so merge-by-sum never double-counts a within-batch edge.
                pair = (uid, other) if uid < other else (other, uid)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                _merge_edge(
                    edges,
                    uid,
                    other,
                    EdgeType.TEMPORAL_NEAR,
                    max(0.0, 1.0 - gap / w),
                    {"gap_us": int(gap), "window_us": w},
                )

        # --- causal: explicit connectives only ------------------------------
        ctx_sorted = sorted(all_recs)
        text_cache: dict[str, Optional[str]] = {}

        def ctx_text(u: str) -> Optional[str]:
            if u not in text_cache:
                text_cache[u] = _unit_text(conn, all_recs.get(u, {}))
            return text_cache[u]

        for uid in new_ids:
            rec = all_recs.get(uid, recs[uid])
            text = ctx_text(uid)
            if not text:
                continue
            for st in _extract_causals(text):
                cause_txt = text[st["cause_span"][0] : st["cause_span"][1]]
                eff_pin = _pin(rec, text, st["effect_span"])
                cause_pin_self = _pin(rec, text, st["cause_span"])
                dsts: list[tuple[str, tuple[int, int], str]] = []
                if (
                    sum(ch.isalnum() for ch in cause_txt)
                    >= MIN_CAUSE_MATCH_CHARS_V1
                ):
                    scanned = 0
                    for other in ctx_sorted:
                        if other == uid:
                            continue
                        if scanned >= CAUSAL_CONTEXT_SCAN_CAP_V1:
                            break
                        otext = ctx_text(other)
                        scanned += 1
                        if not otext:
                            continue
                        ofold = otext.casefold()
                        for variant in _clause_variants(cause_txt):
                            vi = ofold.find(variant.casefold())
                            if vi >= 0:
                                dsts.append((other, (vi, vi + len(variant)), variant))
                                break
                    dsts.sort(key=lambda t: t[0])
                for other, (va, vb), variant in dsts[:CAUSAL_DST_MAX_V1]:
                    orec = all_recs.get(other, {})
                    otext = ctx_text(other) or ""
                    # matched span in the OTHER unit's bytes (char→byte)
                    oa = _byte_off(otext, va)
                    ob = _byte_off(otext, vb)
                    base = orec.get("byte_start")
                    cause_pin = {
                        "unit_id": other,
                        "byte_start": (base + oa) if isinstance(base, int) else oa,
                        "byte_end": (base + ob) if isinstance(base, int) else ob,
                        "text": otext[va:vb],
                    }
                    _merge_edge(
                        edges,
                        other,
                        uid,
                        EdgeType.CAUSAL,
                        1.0,
                        {
                            "connective": st["connective"],
                            "effect": eff_pin,
                            "cause": cause_pin,
                            "statement_unit": uid,
                            "statement_cause_span": {
                                "byte_start": cause_pin_self["byte_start"],
                                "byte_end": cause_pin_self["byte_end"],
                                "text": cause_pin_self["text"],
                            },
                            "self_loop": False,
                        },
                    )
                if not dsts:
                    _merge_edge(
                        edges,
                        uid,
                        uid,
                        EdgeType.CAUSAL,
                        1.0,
                        {
                            "connective": st["connective"],
                            "effect": eff_pin,
                            "cause": cause_pin_self,
                            "statement_unit": uid,
                            "self_loop": True,
                        },
                    )

        written += _insert_edges(conn, scope_id, generation, edges)

    # --- supplied rows attached to unit records ---------------------------
    if supplied_rows or knn_replace:
        edges2: dict[tuple[str, str, str], dict] = {}
        for uid in sorted(knn_replace):
            rec = recs[uid]
            raw = rec.get("knn") or []
            normed = []
            for it in raw:
                if isinstance(it, Mapping):
                    did, sim = it.get("unit") or it.get("dst_unit") or it.get("dst"), it.get("similarity", it.get("weight"))
                else:
                    did, sim = it[0], it[1]
                if did is None or not isinstance(sim, (int, float)):
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        "knn entry needs neighbor id + numeric similarity",
                    )
                normed.append((float(sim), str(did)))
            normed.sort(key=lambda t: (-t[0], t[1]))
            for sim, did in normed[:KNN_MAX_V1]:
                n = _norm_supplied(
                    {"type": EdgeType.SEMANTIC_KNN.value, "dst_unit": did,
                     "similarity": sim},
                    default_src=uid,
                )
                _merge_edge(
                    edges2, n["src"], n["dst"], n["type"], n["weight"], n["evidence"]
                )
        for uid, rel in supplied_rows:
            n = _norm_supplied(rel, default_src=uid)
            _merge_edge(
                edges2, n["src"], n["dst"], n["type"], n["weight"], n["evidence"]
            )
        written += _insert_edges(conn, scope_id, generation, edges2)

    return written


__all__ = [
    "ADJACENT_TURN_RADIUS_V1",
    "CAUSAL_CONTEXT_SCAN_CAP_V1",
    "CAUSAL_DST_MAX_V1",
    "CAUSAL_MAX_PER_UNIT_V1",
    "CO_MENTION_NEIGHBORS_V1",
    "DERIVED_EDGE_TYPES",
    "FORMULA_STATUS",
    "GRAPH_EDGES_VERSION",
    "KNN_MAX_V1",
    "KNN_SIM_FLOOR_V1",
    "MIN_CAUSE_MATCH_CHARS_V1",
    "SAME_SESSION_NEIGHBORS_V1",
    "SUPPLIED_EDGE_TYPES",
    "SYMMETRIC_EDGE_TYPES",
    "TEMPORAL_NEAR_NEIGHBORS_V1",
    "TEMPORAL_NEAR_WINDOW_US_V1",
    "build_edges",
    "write_supplied_edges",
]
