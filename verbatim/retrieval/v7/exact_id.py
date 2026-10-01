"""Exact-identifier lane for the V7 read path (V7-05.11, D7-20).

L-exact in the §04.2 pipeline: the lane exists because the *identifier
channel* (``norm/v2`` — versions, file paths, ticket ids, hashes, handles,
tags, measures, URLs, emails, quoted code spans) is precisely the channel
the folded text index cannot serve.  ``unicode61`` folds case and splits
on punctuation, so ``v2`` vs ``v3`` — and ``v2.4.1`` vs ``v2-4-1`` — are
indistinguishable downstream.  This lane matches identifier surfaces
**byte-exact** and resolves direct id lookups (D7-20: "find me the doc
with id X").

Two match mechanisms, both deterministic:

1. **Byte-exact substring** — each identifier surface is searched in the
   unit's ``text`` payload bytes (``unit_fts_content.text`` reached
   through the ``unit_fts_rows`` carrier, fenced by
   ``(scope_id, generation)``).  The bytes are compared, never the
   tokens: ``CAST(text AS BLOB)`` + substring test is the same
   ``INSTR`` semantics, done in Python so the scan keeps per-row
   deadline/eligibility control and honest ``examined`` counts.

   *Why not FTS5 phrase queries* (measured on this build, sqlite
   3.45.1 + unicode61): ``MATCH '"v2.4.1"'`` tokenizes to the token
   sequence ``v2 4 1`` and therefore (a) **false-positives** —
   ``v2-4-1`` and folded ``V2.4.1`` match equally — and (b)
   **false-negatives** — a byte-exact occurrence glued to a token char
   (``xv2.4.1``) tokenizes as ``xv2 4 1`` and is missed.  FTS5 cannot
   express byte-exact; INSTR-style substring is the deterministic
   choice (the brief's fallback, promoted to primary).  The signal
   value is ``"instr"`` rather than the sketch's ``"fts_phrase"`` so
   rerank inputs never mislabel the mechanism.

   Backtick code-span surfaces (`` `foo()` ``) match both the wrapped
   surface and the inner core — the backticks are quotation, not
   content.

2. **Direct id lookup** — a surface equal to ``units.unit_id``, a
   ``units.source_id``, a ``<source_id>[@:#]<revision>`` pattern
   (``source_revisions``-shaped), or a ``sources.external_id`` resolves
   to real unit rows and is emitted at the top of the lane (D7-20).
   Id lookups only need ``units`` (+ ``sources`` for external_id), so
   they still run when the text index is absent.

Honesty rules honored here (V7-04.03, LaneV7 protocol):

- no identifiers in ``query.norm.identifiers`` → ``skipped`` /
  ``reason="no_identifiers"`` — this lane never runs off-channel;
- eligibility is evaluated while candidates are produced — a held unit
  containing the identifier is never emitted; ``df`` is counted on the
  fenced corpus *before* eligibility (it is a corpus statistic);
- a missing text index does not fabricate matches: id-lookup hits are
  still emitted under ``partial`` / ``"text_index_missing"``, and with
  no hits at all the lane reports ``unavailable``;
- deadline → ``partial``/``"deadline"``; scan bound →
  ``partial``/``"scan_bound"``; pool cap recorded via
  ``stats["cap_truncated"]``.

Scoring (``provisional/v7-r0`` — declared constants, not tuned):
``raw_score = 10.0 + 1.0/(1+df)`` per identifier (df = fenced units
containing the surface, usually ~1), ``+ ID_LOOKUP_BONUS`` when any
match is a direct id resolution, ``+ MULTI_BONUS`` per additional
identifier matched.  Deterministic order: ``(-score, unit_id)``.

The lane is stdlib + sqlite3 only and codes against the §30 column
contract (``units`` / ``unit_fts_rows`` / ``unit_fts_content`` /
``unit_fts`` and the v1 ``sources`` table) plus the frozen ``types_v7``
API.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    CandidateV7,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)
from ...storage.repos import has_table as _has_table

LANE_NAME = LaneName.EXACT_ID.value  # "exact_id" — stable coverage key
LANE_VERSION = "exact_id_lane/v7-r0"  # provisional constants (V7-32.01)

# Scoring constants — provisional/v7-r0, declared not tuned (§32.0).
BASE_SCORE = 10.0       # spec: 10.0 + 1.0/(1+df)
ID_LOOKUP_BONUS = 2.0   # direct id hits lead the lane (D7-20)
MULTI_BONUS = 0.25      # per extra identifier matched by the same unit

MAX_IDENTIFIERS = 32    # beyond that the channel is truncated honestly
SCAN_PAGE = 512         # fetchmany page size for the fenced text scan
SCAN_ROW_LIMIT = 8192   # bound on fenced rows scanned; -> partial/scan_bound
DEADLINE_CHECK_ROWS = 64

UNITS_TABLE = "units"
ROWS_TABLE = "unit_fts_rows"
CONTENT_TABLE = "unit_fts_content"
FTS_TABLE = "unit_fts"
SOURCES_TABLE = "sources"

#: ``<source_id><sep><digits>`` — the source_revisions (source_id, revision)
#: pattern. Requires an explicit separator so "v2"/"ABC-123" never parse.
_REV_SUFFIX_RE = re.compile(r"^(?P<sid>.+?)[@:#](?P<rev>\d{1,12})$")

#: Monotonic clock seam — module-level so deterministic tests (and a
#: replay-time pipeline, if it ever pins one) can substitute a scripted
#: clock. Defaults to ``time.monotonic``.
_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _resolve_conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Find the caller's pinned read snapshot. Lanes never open a new
    transaction (LaneV7 protocol), so a bare Store without a pinned conn
    resolves to None -> ``unavailable`` rather than a fresh snapshot."""
    conn = getattr(ctx, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    return None


def _eligibility(elig: Any) -> Optional[Callable[[dict], bool]]:
    """Normalize ``ctx.eligible`` into ``row -> bool``. Accepted forms per
    the frozen contract: ``callable(unit_row)`` (falling back to a unit_id
    argument on signature mismatch), objects exposing ``unit_ids`` or
    ``is_eligible(row)``, and set-like containers of unit_ids. Anything
    else — including a missing handle — returns None so the lane fails
    closed."""
    if elig is None:
        return None
    if callable(elig):

        def _call(row: dict) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row["unit_id"]))

        return _call
    if getattr(elig, "unit_ids", None) is not None:
        ids = {str(u) for u in elig.unit_ids}
        return lambda row: row["unit_id"] in ids
    if hasattr(elig, "is_eligible"):
        return lambda row: bool(elig.is_eligible(row))
    if hasattr(elig, "__contains__"):
        return lambda row: row["unit_id"] in elig
    return None


def _surfaces(qv: QueryViewV7) -> tuple[tuple[str, ...], bool]:
    """Identifier-channel surfaces, deduped in first-appearance order.

    Returns ``(surfaces, truncated)`` — the channel is capped at
    ``MAX_IDENTIFIERS`` and the truncation is reported in stats."""
    norm = getattr(qv, "norm", None)
    raw = getattr(norm, "identifiers", None) or ()
    seen: list[str] = []
    for t in raw:
        s = getattr(t, "term", t)
        if not isinstance(s, str):
            s = str(s)
        if s and s.strip() and s not in seen:
            seen.append(s)
    truncated = len(seen) > MAX_IDENTIFIERS
    return tuple(seen[:MAX_IDENTIFIERS]), truncated


def _needles(surface: str) -> tuple[bytes, ...]:
    """UTF-8 needle(s) for the byte-exact substring test of ``surface``.

    A backtick-wrapped code span (`` `foo()` ``) also yields its inner
    core — the backticks are query quotation, not content; the wrapped
    surface still matches when the source kept its backticks.
    """
    try:
        needles = [surface.encode("utf-8", "strict")]
    except UnicodeEncodeError:
        return ()
    if len(surface) >= 3 and surface.startswith("`") and surface.endswith("`"):
        core = surface[1:-1]
        if core and core.strip():
            try:
                needles.append(core.encode("utf-8", "strict"))
            except UnicodeEncodeError:
                pass
    return tuple(dict.fromkeys(needles))


class _Deadline:
    """Relative per-lane slice budget (``LaneSlice.deadline_ms``)."""

    __slots__ = ("t_end",)

    def __init__(self, ms: float) -> None:
        try:
            budget = float(ms)
        except (TypeError, ValueError):
            budget = 0.0
        self.t_end = _monotonic() + max(budget, 0.0) / 1000.0

    def expired(self) -> bool:
        return _monotonic() >= self.t_end


def _row_dict(cols: list[str], row: tuple) -> dict:
    """Eligibility view of a unit row: whatever columns the mirror/real
    table actually stored, keyed by column name."""
    return {c: row[i] for i, c in enumerate(cols)}


# ---------------------------------------------------------------------------
# candidate accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Cand:
    unit_id: str
    source_id: str = ""
    revision: int = 0
    # per-match records: {"identifier", "match" ("instr"|"id_lookup"),
    #                     "df", "id_kind"?}
    matches: list = field(default_factory=list)
    score: float = 0.0

    def order_key(self) -> tuple:
        return (-self.score, self.unit_id)


def _merge(
    pool: dict[str, _Cand],
    row: dict,
    surface: str,
    match: str,
    df: int,
    id_kind: Optional[str] = None,
) -> None:
    """Record ``surface`` hitting ``row``'s unit; merge per unit_id."""
    uid = row["unit_id"]
    cand = pool.get(uid)
    if cand is None:
        cand = _Cand(
            unit_id=uid,
            source_id=str(row.get("source_id") or ""),
            revision=int(row.get("revision") or 0),
        )
        pool[uid] = cand
    entry: dict[str, Any] = {"identifier": surface, "match": match, "df": df}
    if id_kind is not None:
        entry["id_kind"] = id_kind
    cand.matches.append(entry)


# ---------------------------------------------------------------------------
# direct id lookups (D7-20)
# ---------------------------------------------------------------------------


def _fetch_units(
    conn: sqlite3.Connection,
    where: str,
    params: list,
    ctx: LaneContextV7,
    eligible_fn: Callable[[dict], bool],
    out: LaneOutput,
) -> list[dict]:
    """Point-lookup fenced unit rows; each fetched row counts as examined
    and passes the caller's eligibility predicate before it can emit.

    V7-30.02: ``units`` is keyed ``(unit_id, generation)`` — the fence is
    ``generation <= pinned`` and the *latest* row per unit_id supplies the
    row, so ``where`` predicates (unit_id / source_id / revision) bind the
    newest visible projection, never a superseded one."""
    rows: list[dict] = []
    cur = conn.execute(
        f"SELECT u.* FROM {UNITS_TABLE} u WHERE u.scope_id = ?"
        f" AND u.generation <= ? AND {where}"
        " AND u.generation = (SELECT MAX(u2.generation) FROM units u2"
        "                    WHERE u2.unit_id = u.unit_id"
        "                      AND u2.scope_id = u.scope_id"
        "                      AND u2.generation <= ?)"
        " ORDER BY u.unit_id, u.revision",
        [ctx.scope_id, ctx.generation, *params, ctx.generation],
    )
    cols = [d[0] for d in cur.description]
    for r in cur.fetchall():
        out.examined += 1
        rd = _row_dict(cols, r)
        if eligible_fn(rd):
            rows.append(rd)
    return rows


def _id_lookups(
    conn: sqlite3.Connection,
    surface: str,
    ctx: LaneContextV7,
    eligible_fn: Callable[[dict], bool],
    out: LaneOutput,
    have_sources: bool,
) -> list[tuple[dict, str]]:
    """Resolve ``surface`` as a direct id. Returns ``(row, id_kind)``
    pairs for every distinct unit the identifier names:

    - ``unit_id`` — the unit itself ("the doc with id X");
    - ``source_id`` — every fenced unit of that source;
    - ``source_revision`` — ``<sid>[@:#]<rev>`` naming one revision;
    - ``external_id`` — ``sources.external_id`` -> source_id -> units.
    """
    hits: list[tuple[dict, str]] = []

    for rd in _fetch_units(
        conn, "u.unit_id = ?", [surface], ctx, eligible_fn, out
    ):
        hits.append((rd, "unit_id"))

    for rd in _fetch_units(
        conn, "u.source_id = ?", [surface], ctx, eligible_fn, out
    ):
        hits.append((rd, "source_id"))

    m = _REV_SUFFIX_RE.match(surface)
    if m is not None:
        try:
            rev = int(m.group("rev"))
        except ValueError:
            rev = -1
        if rev >= 0:
            for rd in _fetch_units(
                conn,
                "u.source_id = ? AND u.revision = ?",
                [m.group("sid"), rev],
                ctx,
                eligible_fn,
                out,
            ):
                hits.append((rd, "source_revision"))

    if have_sources:
        try:
            sids = [
                str(r[0])
                for r in conn.execute(
                    f"SELECT source_id FROM {SOURCES_TABLE}"
                    " WHERE scope_id = ? AND external_id = ?"
                    " ORDER BY source_id",
                    (ctx.scope_id, surface),
                ).fetchall()
            ]
        except sqlite3.Error:
            sids = []
            out.stats["external_id_error"] = True
        for sid in sids:
            for rd in _fetch_units(
                conn, "u.source_id = ?", [sid], ctx, eligible_fn, out
            ):
                hits.append((rd, "external_id"))

    # one (unit, kind) pair each — a unit reachable by two id kinds keeps both
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[dict, str]] = []
    for rd, kind in hits:
        key = (rd["unit_id"], kind)
        if key not in seen:
            seen.add(key)
            deduped.append((rd, kind))
    return deduped


# ---------------------------------------------------------------------------
# byte-exact text scan
# ---------------------------------------------------------------------------


def _text_scan_sql(conn: sqlite3.Connection) -> tuple[Optional[str], str]:
    """Pick the fenced unit-text scan for the wiring actually present.

    ``(sql, path)`` — path names the wiring used:

    - ``"fts_content"``: canonical §30 wiring — ``unit_fts_rows`` carrier
      joined to ``unit_fts_content`` on ``fts_row_id`` (works even on
      FTS5-less builds: both are plain tables);
    - ``"fts_table"``: carrier present but content absent — read the
      stored ``text`` column through ``unit_fts`` (external-content
      read-through or standalone index);
    - ``"none"``: no carrier — the lane cannot fence a text scan and
      reports the text channel missing rather than guess rowid
      conventions (``unit_fts.rowid`` == ``units.rowid`` holds only
      under mirror/standalone wiring, never the real schema).
    """
    # V7-30.02: both the carrier (``UNIQUE(unit_id, generation)``) and
    # ``units`` (``PRIMARY KEY (unit_id, generation)``) are versioned — the
    # scan reads each unit's *latest* row at/below the pinned generation on
    # both sides of the pair, so superseded text/metadata never match.
    _latest = (
        "JOIN (SELECT unit_id, MAX(generation) AS mg FROM {t}"
        "      WHERE scope_id = ? AND generation <= ?"
        "      GROUP BY unit_id) {a}"
        "   ON {a}.unit_id = {r}.unit_id AND {a}.mg = {r}.generation"
    )
    latest_rows = _latest.format(t=ROWS_TABLE, a="rl", r="r")
    latest_units = _latest.format(t=UNITS_TABLE, a="ul", r="u")
    if _has_table(conn, ROWS_TABLE) and _has_table(conn, CONTENT_TABLE):
        return (
            f"SELECT u.*, CAST(c.text AS BLOB) AS unit_text"
            f" FROM {ROWS_TABLE} r"
            f" {latest_rows}"
            f" JOIN {CONTENT_TABLE} c ON c.fts_row_id = r.row_id"
            f" JOIN {UNITS_TABLE} u ON u.unit_id = r.unit_id"
            f"   AND u.scope_id = r.scope_id"
            f" {latest_units}"
            " WHERE r.scope_id = ? AND r.generation <= ?"
            " ORDER BY u.unit_id",
            "fts_content",
        )
    if _has_table(conn, ROWS_TABLE) and _has_table(conn, FTS_TABLE):
        return (
            f"SELECT u.*, CAST(c.text AS BLOB) AS unit_text"
            f" FROM {ROWS_TABLE} r"
            f" {latest_rows}"
            f" JOIN {FTS_TABLE} c ON c.rowid = r.row_id"
            f" JOIN {UNITS_TABLE} u ON u.unit_id = r.unit_id"
            f"   AND u.scope_id = r.scope_id"
            f" {latest_units}"
            " WHERE r.scope_id = ? AND r.generation <= ?"
            " ORDER BY u.unit_id",
            "fts_table",
        )
    return None, "none"


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_exact_id(
    ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
) -> LaneOutput:
    """L-exact (V7-05.11, D7-20): byte-exact identifier substring +
    direct id resolution."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = FORMULA_STATUS_PROVISIONAL

    deadline = _Deadline(getattr(slice, "deadline_ms", 0.0) or 0.0)

    # Channel gate first (V7-05.11): the lane ONLY runs when the
    # identifier channel has entries — it never touches the store to
    # report itself off-channel.
    surfaces, idents_truncated = _surfaces(query)
    stats["identifiers"] = list(surfaces)
    stats["n_identifiers"] = len(surfaces)
    if idents_truncated:
        stats["identifiers_truncated"] = True
    if not surfaces:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_identifiers"
        return out

    conn = _resolve_conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    if ctx.generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out
    if not _has_table(conn, UNITS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "units_table_missing"
        return out

    eligible_fn = _eligibility(getattr(ctx, "eligible", None))
    if eligible_fn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_handle_missing"
        return out

    cap = max(0, int(getattr(slice, "cap", 0) or 0))
    if cap == 0:
        stats["mode"] = "capped"
        return out
    if deadline.expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["mode"] = "none"
        return out

    pool: dict[str, _Cand] = {}
    df_map: dict[str, int] = {s: 0 for s in surfaces}
    modes: list[str] = []
    deadline_hit = False
    truncated = False

    # -- phase 1: direct id lookups (D7-20) ----------------------------------
    have_sources = _has_table(conn, SOURCES_TABLE)
    stats["sources_table"] = have_sources
    id_kinds: dict[str, list[str]] = {}
    try:
        for surface in surfaces:
            if deadline.expired():
                deadline_hit = True
                break
            hits = _id_lookups(
                conn, surface, ctx, eligible_fn, out, have_sources
            )
            if hits:
                modes.append("id_lookup")
                id_kinds.setdefault(surface, [])
                # df for an id lookup = the distinct units it resolved
                df = len({rd["unit_id"] for rd, _k in hits})
                for rd, kind in hits:
                    if kind not in id_kinds[surface]:
                        id_kinds[surface].append(kind)
                    _merge(pool, rd, surface, "id_lookup", df, id_kind=kind)
    except sqlite3.Error as exc:
        stats["id_lookup_error"] = repr(exc)[:200]
    stats["id_kinds"] = id_kinds

    # -- phase 2: byte-exact substring over the fenced unit texts ------------
    needles = {s: _needles(s) for s in surfaces}
    sql, text_path = _text_scan_sql(conn)
    stats["text_path"] = text_path
    if sql is None:
        stats["text_channel"] = "missing"
    elif not deadline_hit:
        stats["text_channel"] = "ok"
        modes.append("substring")
        n = 0
        try:
            cur = conn.execute(
                sql,
                [
                    ctx.scope_id, ctx.generation,  # latest carrier rows
                    ctx.scope_id, ctx.generation,  # latest units rows
                    ctx.scope_id, ctx.generation,  # outer fence
                ],
            )
            cols = [d[0] for d in cur.description][:-1]  # unit_text last
            while True:
                batch = cur.fetchmany(SCAN_PAGE)
                if not batch:
                    break
                for row in batch:
                    n += 1
                    out.examined += 1
                    blob = row[-1]
                    if blob is not None:
                        text = bytes(blob)
                        matched = [
                            s
                            for s in surfaces
                            if any(p in text for p in needles[s])
                        ]
                        if matched:
                            # df is a corpus statistic — counted before
                            # eligibility
                            for s in matched:
                                df_map[s] += 1
                            rd = _row_dict(cols, row[:-1])
                            if eligible_fn(rd):
                                for s in matched:
                                    _merge(pool, rd, s, "instr", df_map[s])
                    if n % DEADLINE_CHECK_ROWS == 0 and deadline.expired():
                        deadline_hit = True
                        break
                    if n >= SCAN_ROW_LIMIT:
                        truncated = True
                        break
                if deadline_hit or truncated:
                    break
        except sqlite3.Error as exc:
            stats["text_channel"] = "error"
            stats["text_error"] = repr(exc)[:200]
        stats["scan_rows"] = n
    else:
        stats["text_channel"] = "deadline_skip"

    stats["df"] = {s: df_map[s] for s in surfaces if df_map[s]}
    stats["mode"] = "+".join(dict.fromkeys(modes)) if modes else "none"
    stats["matched_identifiers"] = len(
        {m["identifier"] for c in pool.values() for m in c.matches}
    )
    out.eligible = len(pool)

    # -- scoring: 10.0 + 1/(1+df), id bonus, multi-identifier bonus ----------
    for cand in pool.values():
        for m in cand.matches:
            if m["match"] == "instr":
                m["df"] = df_map.get(m["identifier"], 0)
        best_df = min(m["df"] for m in cand.matches)
        n_idents = len({m["identifier"] for m in cand.matches})
        has_id = any(m["match"] == "id_lookup" for m in cand.matches)
        cand.score = (
            BASE_SCORE
            + 1.0 / (1.0 + best_df)
            + (ID_LOOKUP_BONUS if has_id else 0.0)
            + MULTI_BONUS * (n_idents - 1)
        )

    final = sorted(pool.values(), key=lambda c: c.order_key())[:cap]
    stats["overflow"] = max(0, len(pool) - len(final))
    if stats["overflow"]:
        stats["cap_truncated"] = stats["overflow"]

    for rank, cand in enumerate(final, start=1):
        # headline match: prefer id_lookup, then rarest df, then surface
        best = min(
            cand.matches,
            key=lambda m: (m["match"] != "id_lookup", m["df"], m["identifier"]),
        )
        signals: dict[str, Any] = {
            "identifier": best["identifier"],
            "match": best["match"],
            "df": best["df"],
        }
        if best["match"] == "id_lookup" and best.get("id_kind"):
            signals["id_kind"] = best["id_kind"]
        idents = sorted({m["identifier"] for m in cand.matches})
        if len(idents) > 1:
            signals["identifiers"] = idents
            signals["n_identifiers"] = len(idents)
        out.candidates.append(
            CandidateV7(
                unit_id=cand.unit_id,
                source_id=cand.source_id,
                revision=cand.revision,
                lane=LANE_NAME,
                rank=rank,
                raw_score=cand.score,
                signals=signals,
            )
        )

    if deadline_hit:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
    elif truncated:
        out.status = LaneStatus.PARTIAL
        out.reason = "scan_bound"
    elif text_path == "none" or stats.get("text_channel") == "error":
        # the byte-exact channel could not run; id-lookup hits are real
        # evidence and still ship, honestly labeled partial.
        if out.candidates:
            out.status = LaneStatus.PARTIAL
        else:
            out.status = LaneStatus.UNAVAILABLE
        out.reason = "text_index_missing"
    return out


class ExactIdLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
        return lane_exact_id(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the pipeline
# never imports this module itself — the registrar/importer seam is owned
# by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.EXACT_ID, lane_exact_id)


__all__ = [
    "BASE_SCORE",
    "ID_LOOKUP_BONUS",
    "ExactIdLane",
    "LANE_NAME",
    "LANE_VERSION",
    "MAX_IDENTIFIERS",
    "MULTI_BONUS",
    "lane_exact_id",
]
