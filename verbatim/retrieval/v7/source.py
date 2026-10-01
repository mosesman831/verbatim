"""V7 source lane — the governed coverage floor (V4-08.07 carried).

L-source in the §04.2 pipeline surfaces *raw source payloads* — the
retained bytes a memory aliases — through the ordinary governed lane
machinery: eligibility before rank, the ``(scope, generation)`` fence,
quarantine/purge/lifecycle gates, honest degradation.  Raw sources ship
ONLY through this lane — never through a post-denial bypass
(V4-08.07).  It is the honest coverage floor: it runs for EVERY intent
(never intent-gated — the D7-09 ``thin``/``run_source`` gate is dead by
construction, V7-05.01/05.03), and it is what makes V7 retrieval work on
a store whose T0 unit projection has not run yet (or never ran): a
source-only store still retrieves through the V7 pipeline.

Two sub-modes produce one ranked list under one pool cap:

- **unit-backed** — folded query terms probe ``unit_fts`` (the ``text``
  field, OR'd terms + a phrase probe) the way the primary lex lane
  nominates, then the lane re-scores on the stored field bytes with a
  coarse single-field BM25 (``bm25-src/v1`` — this is the fallback, not
  the primary lexical lane).  Candidates carry the real ``unit_id`` and
  ``signals["unit_level"] = True``.  Both index conventions are
  supported: the §30 carrier pair (``unit_fts.rowid`` →
  ``unit_fts_rows.row_id`` → ``unit_id``) and the standalone
  external-content convention (``unit_fts.rowid`` → ``units.rowid``)
  used by wave-A mirror schemas.  Stale-generation carrier rowids are
  resolved through an alias map so a posting hit on an older rebuild
  row still nominates the unit — scoring always reads the latest
  visible row's bytes (V7-30.02).

- **source-level** — terms probe the v5 source projection
  (``source_lexical_projection.tokens``/``doc_len`` — the norm/v1
  space-joined token projection; ``source_fts_idx`` +
  ``source_fts_rows`` nomination when the FTS shadow exists) to catch
  sources whose units were never projected.  ``CandidateV7.unit_id`` is
  required, so a source-level hit emits the synthetic
  ``unit_id = "{source_id}:{revision}"`` with
  ``signals["unit_level"] = False`` — verified against the consumers:
  ``fusion.rrf_fuse`` keys on ``str(unit_id)`` (a stable per-
  (source, revision) identity merges correctly; ``""`` would collapse
  every orphaned source into one fused entry), ``pack._as_item`` copies
  it into ``PackItemV7.unit_id``/``ref`` without dereferencing a
  ``units`` row, and ``verdict_v2`` groups it under
  ``src:<source_id>``.  The pack layer renders these from payload
  spans.  A source already represented by an emitted unit-backed
  candidate is suppressed at source granularity (counted in
  ``stats["source_covered_by_units"]``) — the lane covers what
  unit-level artifacts did not answer, it never double-represents.

Eligibility (V7-05.08) — the caller's ``ctx.eligible`` handle is
unit-granular by contract.  For unit-backed hits it evaluates the real
unit row.  For source-level hits a unit-level handle cannot name the
object, so the lane first applies the same governance the eligibility
adapter composes — scope+generation fence, live quarantine holds on the
``source`` ref plus the covering ``source_envelope`` cascade
(V3-14.10), suppressing ``purges`` targets (``source`` and
``source_revision`` object kinds), and ``source_state`` currency
(V5-14.12/14.16) — and THEN still offers the caller's handle a
synthesized row (``unit_id`` = synthetic, ``kind="source"``, plus the
``sources`` row fields when loadable): a callable may deny it; a
unit-id container fails closed (the synthetic id is never a member) and
the drop is counted in ``stats["eligible_dropped"]``.  Source-level
candidates carry ``signals["eligibility"] = "source_gates"`` so the gap
between unit-level and source-granular eligibility is visible in
coverage/explain instead of silent.  A quarantine/purge read failure on
this path fails closed — the hit is withheld, never leaked.

Generation fence (V7-30.02, docs/v7_contracts.md): every read is
``generation <= ctx.generation`` with latest-row-per-natural-key
resolution — rebuild coexistence means several stamped generations can
sit side by side and only the newest at/below the pinned snapshot is
visible.  ``ctx.generation is None`` → ``unavailable`` /
``"generation_unpinned"`` (the same convention the entity lane uses) —
a lane never scans unfenced.

Scoring — ``bm25-src/v1`` (provisional/v7-r0): single-field BM25, the
same k1/b/idf family as ``bm25f/v1`` restricted to the ``text`` field,
plus a declared additive phrase bonus when the consecutive query-term
sequence occurs contiguously in the matched token stream.
``df(t)``/``N``/``avgdl`` are measured over the *eligible* fenced set
(F4-11 discipline carried — the index nominates, stored bytes verify);
when the eligible universe exceeds the declared statistics bound the
lane reports ``stats["stats"] = "nominated"`` instead of silently
relabeling a candidate-set statistic as corpus truth.

Honesty (V7-04.03): ``unavailable`` reasons name the missing piece
(``no_read_snapshot``, ``generation_unpinned``, ``no_unit_fts``,
``no_source_projection``, ``no_indexes``); a term-less query reports
``skipped``/``"no_terms"``; deadline-cut scans report
``partial``/``"deadline"`` with real ``examined``/``eligible`` counts;
declared row bounds report ``partial``/``"scan_bound"``; pool overflow
is counted in ``stats["cap_truncated"]``.

The lane is stdlib + sqlite3 only, consumes the caller's pinned read
snapshot, and never opens a transaction (LaneV7 protocol).
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
import unicodedata
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    CandidateV7,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    POOLS,
    QueryViewV7,
)
from ...storage.repos import has_table

try:  # norm/v1 — the pinned projection space of source_lexical_projection
    from ...enrichment.normalize import NORMALIZATION_VERSION, normalize_text
except Exception:  # pragma: no cover - partial checkout resilience
    NORMALIZATION_VERSION = "norm/v1"
    normalize_text = None

LANE_NAME = LaneName.SOURCE.value  # "source" — stable coverage key
LANE_VERSION = "source_lane/v7-r0"  # provisional constants (V7-32.01)
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL
FORMULA_ID = "bm25-src/v1"

# ---------------------------------------------------------------------------
# declared constants (provisional/v7-r0 — same ranking-contract family as
# bm25f/v1; single 'text' field only, deliberately coarser than L-lex)
# ---------------------------------------------------------------------------

K1 = 1.2          # BM25 k1 (ranking-contract value, V5 carried)
B = 0.75          # BM25 length normalization (text field, §32.2 b_text)
W_PHRASE = 0.5    # additive raw-score bonus for a contiguous phrase hit

UNITS_TABLE = "units"
UNIT_FTS = "unit_fts"                # §30 fielded index (MATCH target)
UNIT_FTS_ROWS = "unit_fts_rows"      # §30 rowid carrier (real schema)
UNIT_FTS_DOCSIZE = "unit_fts_docsize"  # FTS5 shadow: per-field token counts
PROJ_TABLE = "source_lexical_projection"  # v5: tokens + doc_len per revision
SRC_FTS = "source_fts"               # v5 content table (normalized tokens)
SRC_FTS_IDX = "source_fts_idx"       # v5 FTS5 virtual table (MATCH target)
SRC_FTS_ROWS = "source_fts_rows"     # v5 rowid carrier (fence metadata)
SOURCES_TABLE = "sources"
STATE_TABLE = "source_state"
QUAR_TABLE = "quarantine"
ENV_TABLE = "source_envelopes"
PURGES_TABLE = "purges"
PURGE_TARGETS_TABLE = "purge_targets"
LEX_STATS_TABLE = "lex_stats"

MAX_TERMS = 24             # query-term bound (spec envelope is <= 12)
POSTING_ROW_LIMIT = 8192   # per-term FTS nomination bound
UNIVERSE_SCAN_CAP = 16384  # unit-carrier / units fence scan bound
SOURCE_SCAN_CAP = 16384    # source-projection fence scan bound
STATS_ROW_CAP = 8192       # eligible rows contributing to avgdl
IN_CHUNK = 400             # < SQLITE_MAX_VARIABLE_NUMBER
_PAGE = 512                # streaming fetch page

_LIVE_HOLD = ("pending", "suppressed")
_SUPPRESSING = ("suppressed", "purging", "completed")
#: ``source_state`` dispositions that can never be the current answer
#: (V5-14.09): ``recorded`` is pre-admission registration; the rest carry
#: no current head.  ``superseded`` answers only until its declared
#: ``effective_at`` boundary (V5-14.12), evaluated at read time.
_NEVER_CURRENT = frozenset(
    {"recorded", "corrected", "retracted", "archived", "erased"}
)

#: unit columns handed to a callable eligibility predicate / merged into
#: signals — intersected with the live ``units`` schema before selecting
#: (mirror DDLs may carry a reduced column set).
_UNITS_WANT_COLS = (
    "unit_id", "source_id", "revision", "scope_id", "kind",
    "parent_unit_id", "session_id", "seq", "speaker_canon", "perspective",
    "recorded_at_us", "occurred_start_us", "occurred_end_us",
    "occurred_precision", "occurred_source", "byte_start", "byte_end",
    "generation",
)


def _monotonic() -> float:
    """Indirection so tests drive the cooperative clock deterministically."""
    return time.monotonic()


class _Deadline:
    """Relative per-lane slice budget (LaneSlice.deadline_ms)."""

    __slots__ = ("_end",)

    def __init__(self, deadline_ms: Optional[float]) -> None:
        if deadline_ms is None:
            self._end = None
            return
        try:
            ms = float(deadline_ms)
        except (TypeError, ValueError):
            ms = 0.0
        self._end = (
            None
            if math.isinf(ms)
            else _monotonic() + max(ms, 0.0) / 1000.0
        )

    def expired(self) -> bool:
        return self._end is not None and _monotonic() >= self._end


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Resolve the caller-pinned read connection (the wave-A/B lane
    convention): explicit ``ctx.conn``, then ``ctx.store`` itself when it
    is a connection, then ``ctx.store.conn``, then the Store's
    thread-local reader.  The lane never opens a transaction of its own
    (V7-05.06) — no resolvable snapshot is ``unavailable``."""
    conn = getattr(ctx, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    reader = getattr(store, "_reader", None)
    if callable(reader):
        try:
            conn = reader()
        except Exception:
            return None
        if isinstance(conn, sqlite3.Connection):
            return conn
    if hasattr(store, "execute"):
        return store  # duck-typed connection-like
    return None


# --- unicode61 + remove_diacritics-2 mirror (the §32.1 matching projection,
# --- identical semantics to the lexical lane's local copy) -----------------


def _fold(text: str) -> str:
    """NFKC -> casefold -> strip combining marks (the unit_fts matching
    projection; idempotent over already-folded input)."""
    s = str(text)
    if s.isascii():
        return s.casefold()
    norm = unicodedata.normalize("NFKC", s).casefold()
    decomp = unicodedata.normalize("NFKD", norm)
    return "".join(
        c for c in decomp if not unicodedata.category(c).startswith("M")
    )


_TOKEN_RE = re.compile(r"[^\W_]+")


def _tokenize(text: str) -> list:
    """unicode61-equivalent token stream over the folded projection —
    maximal runs of Unicode L*/N* characters.  Used to count tf on stored
    field bytes (the match authority; the index only nominates)."""
    if not text:
        return []
    if text.isascii():
        return _TOKEN_RE.findall(text.casefold())
    return _TOKEN_RE.findall(_fold(text))


def _norm_tokens(text: str) -> list:
    """norm/v1 token stream — the ``source_lexical_projection.tokens`` /
    ``source_fts.text`` space (punctuation/separators already folded to
    spaces by the projection, so ``split()`` is exact; the local fallback
    reproduces norm/v1 when ``enrichment.normalize`` is absent)."""
    if not text:
        return []
    if normalize_text is not None:
        return normalize_text(text).split()
    # local norm/v1-equivalent fallback (NFKC -> NFKD drop marks ->
    # casefold -> P/S/C/Z fold to space -> collapse)
    s = unicodedata.normalize("NFKC", str(text))
    s = unicodedata.normalize("NFKD", s)
    out = []
    for ch in s:
        g = unicodedata.category(ch)[0]
        if g == "M":
            continue
        out.append(" " if g in "PSCZ" else ch)
    return "".join(out).casefold().split()


def _fts_quote(term: str) -> str:
    """One MATCH-safe literal: double-quoted, inner quotes doubled."""
    return '"' + str(term).replace('"', '""') + '"'


def _match_rowids(
    conn: sqlite3.Connection, table: str, match: str
) -> Optional[set]:
    """Rowids matching an FTS5 query; ``None`` on error (syntax or a
    missing/shadowless table) — callers degrade honestly, never crash."""
    try:
        cur = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? LIMIT ?",
            (match, POSTING_ROW_LIMIT),
        )
        return {int(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return None


def _table_cols(conn: sqlite3.Connection, table: str) -> list:
    """Declared column order (PRAGMA table_info); [] on error."""
    try:
        return [
            str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")
        ]
    except sqlite3.Error:
        return []


def _decode_sizes(blob: bytes) -> list:
    """FTS5 ``%_docsize.sz`` — one 7-bit varint per column."""
    out: list[int] = []
    val = 0
    for byte in blob:
        val = (val << 7) | (byte & 0x7F)
        if not byte & 0x80:
            out.append(val)
            val = 0
    if val:
        out.append(val)
    return out


# ---------------------------------------------------------------------------
# bm25-src/v1 — single-field BM25 (the coarse fallback scorer)
# ---------------------------------------------------------------------------


def _idf(n_docs: float, df: float) -> float:
    """``ln(1 + (N - df + 0.5) / (df + 0.5))`` — the bm25f/v1 idf."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _bm25_text(tf: float, dl: float, avgdl: float, idf_value: float) -> float:
    """One term's BM25 contribution on the single ``text`` field —
    identical to ``bm25f/v1`` restricted to ``{text: 1.0, b=0.75}``."""
    norm = 1.0 - B + (B * float(dl) / avgdl if avgdl > 0.0 else 0.0)
    tf_t = float(tf) / norm if norm > 0.0 else float(tf)
    return idf_value * tf_t * (K1 + 1.0) / (tf_t + K1)


def _phrase_hit(seq: list, doc_tokens: list) -> bool:
    """``seq`` (the consecutive query token stream, len >= 2) occurs
    contiguously in ``doc_tokens`` — deterministic phrase probe."""
    n = len(seq)
    if n < 2 or len(doc_tokens) < n:
        return False
    first = seq[0]
    for i, tok in enumerate(doc_tokens):
        if tok == first and doc_tokens[i : i + n] == seq:
            return True
    return False


# ---------------------------------------------------------------------------
# query terms
# ---------------------------------------------------------------------------


def _query_terms(qv: QueryViewV7, stats: dict) -> tuple:
    """Folded query terms in analysis order, deduped.

    Returns ``(terms, ident_flags)``: ``terms`` covers the ``text`` and
    ``identifier`` channels (stem twins are skipped — they duplicate the
    text term they shadow); ``ident_flags`` marks identifier-channel
    terms.  Term text is folded into the unit_fts matching projection.
    """
    norm = getattr(qv, "norm", None)
    terms = tuple(getattr(norm, "terms", ()) or ())
    idents = tuple(getattr(norm, "identifiers", ()) or ())
    ident_surfaces = {str(getattr(t, "term", "")) for t in idents}

    out: list[str] = []
    flags: dict[str, bool] = {}
    seen: set = set()
    truncated = 0
    for t in (*terms, *idents):
        ch = getattr(t, "channel", None) or "text"
        term = str(getattr(t, "term", "") or "")
        if ch == "stem" or not term:
            continue
        folded = _fold(term)
        if not folded:
            continue
        is_ident = ch == "identifier" or term in ident_surfaces
        if folded in seen:
            flags[folded] = flags[folded] or is_ident
            continue
        if len(out) >= MAX_TERMS:
            truncated += 1
            continue
        seen.add(folded)
        out.append(folded)
        flags[folded] = is_ident
    if truncated:
        stats["terms_truncated"] = truncated
    return out, flags


def _source_term_groups(terms: list) -> list:
    """Per-term norm/v1 token groups for the source-projection space —
    ``"abc-123"`` projects to ``["abc", "123"]`` there.  A term is
    *covered* by a source doc iff ALL its projection tokens occur in it."""
    groups = []
    for t in terms:
        toks = _norm_tokens(t)
        groups.append(tuple(toks) if toks else (t,))
    return groups


def _term_tf(term_toks: list, counter: Mapping) -> int:
    """tf of a possibly multi-token term inside a token Counter —
    min over subtokens (contiguous-or-not is a phrase question; tf
    measures presence)."""
    if not term_toks:
        return 0
    return min(int(counter.get(t, 0)) for t in term_toks)


# ---------------------------------------------------------------------------
# eligibility — ctx.eligible normalization (types_v7 contract forms)
# ---------------------------------------------------------------------------


def _eligibility(ctx: LaneContextV7) -> tuple:
    """``(fn, via)`` — normalize ``ctx.eligible`` into ``row -> bool``.

    Contract forms: ``callable(unit_row)`` (falling back to a bare
    unit_id argument on signature mismatch), objects exposing
    ``eligible(row)``/``is_eligible(row)``/``unit_ids``/
    ``contains(unit_id)``, containers of unit_ids, ``None`` =
    unrestricted.  Anything unrecognized — and any predicate that
    raises — fails CLOSED (eligibility-before-rank is the security
    property, V7-05.08).
    """
    elig = getattr(ctx, "eligible", None)
    if elig is None:
        return (lambda row: True), "all"
    if callable(elig):

        def _call(row: Mapping) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row.get("unit_id")))

        fn = _call
        via = "callable"
    elif callable(getattr(elig, "is_eligible", None)):
        fn = lambda row: bool(elig.is_eligible(row))
        via = "is_eligible"
    elif callable(getattr(elig, "eligible", None)):
        fn = lambda row: bool(elig.eligible(row))
        via = "eligible_attr"
    elif getattr(elig, "unit_ids", None) is not None:
        ids = elig.unit_ids
        fn = lambda row: row.get("unit_id") in ids
        via = "unit_ids"
    elif callable(getattr(elig, "contains", None)):
        contains = elig.contains
        fn = lambda row: bool(contains(row.get("unit_id")))
        via = "contains"
    elif hasattr(elig, "__contains__"):
        fn = lambda row: row.get("unit_id") in elig
        via = "set"
    else:
        return (lambda row: False), "unrecognized"

    def guarded(row: Mapping) -> bool:
        try:
            return bool(fn(row))
        except Exception:
            return False

    return guarded, via


# ---------------------------------------------------------------------------
# source-level governance gates — the eligibility-adapter machinery mirrored
# locally (retrieval/v7/eligibility.py has not landed; the lane applies the
# same hold/purge/lifecycle semantics itself and marks candidates
# ``signals["eligibility"]="source_gates"``)
# ---------------------------------------------------------------------------


def _held_source_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> set:
    """Source ids under a live quarantine hold — mirrors
    ``typed_lane._held_source_ids`` / the claim-lane cascade (V3-14.10):
    direct ``source`` holds plus the covering ``source_envelope`` hold.
    Raises propagate (the caller's policy is fail-closed); a missing
    table means no hold machinery exists on this store."""
    ids = sorted({str(i) for i in ids if i})
    if not ids or not has_table(conn, QUAR_TABLE):
        return set()
    if conn.execute(
        "SELECT 1 FROM quarantine"
        " WHERE state IN ('pending','suppressed') LIMIT 1"
    ).fetchone() is None:
        return set()  # no active holds anywhere — skip the ref probes
    ph = _ph(len(ids))
    held = {
        str(r[0])
        for r in conn.execute(
            "SELECT object_id FROM quarantine"
            f" WHERE object_kind = 'source' AND object_id IN ({ph})"
            " AND state IN ('pending','suppressed')",
            ids,
        )
    }
    if has_table(conn, ENV_TABLE):
        held |= {
            str(r[0])
            for r in conn.execute(
                "SELECT se.source_id FROM quarantine q"
                " JOIN source_envelopes se"
                "  ON se.envelope_id = q.object_id"
                " WHERE q.object_kind = 'source_envelope'"
                f"  AND se.source_id IN ({ph})"
                "  AND q.state IN ('pending','suppressed')",
                ids,
            )
        }
    return held


def _suppressed_pairs(conn: sqlite3.Connection, pairs: Iterable[tuple]) -> set:
    """``(source_id, revision)`` keys under a suppressing purge — the
    ``source`` and ``source_revision`` target forms, mirroring
    ``candidates._suppressed``'s states and object-id conventions
    (``suppressed``|``purging``|``completed``; previewed purges do not
    suppress).  Missing purge tables mean nothing can be suppressed."""
    wanted = sorted({(str(s), int(r)) for s, r in pairs})
    if not wanted:
        return set()
    if not has_table(conn, PURGES_TABLE) or not has_table(
        conn, PURGE_TARGETS_TABLE
    ):
        return set()
    ph_states = _ph(len(_SUPPRESSING))
    try:
        live = conn.execute(
            f"SELECT 1 FROM purges p WHERE p.state IN ({ph_states}) LIMIT 1",
            list(_SUPPRESSING),
        ).fetchone()
    except sqlite3.Error:
        live = None
    if live is None:
        return set()
    out: set = set()
    sids = sorted({s for s, _ in wanted})
    suppressed_sids: set = set()
    for part in _chunks(sids, IN_CHUNK):
        suppressed_sids.update(
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE pt.object_kind = ?"
                f" AND p.state IN ({ph_states})"
                f" AND pt.object_id IN ({_ph(len(part))})",
                ["source", *_SUPPRESSING, *part],
            )
        )
    out.update(k for k in wanted if k[0] in suppressed_sids)
    rev_ids = sorted({f"{s}:{r}" for s, r in wanted})
    suppressed_revs: set = set()
    for part in _chunks(rev_ids, IN_CHUNK):
        suppressed_revs.update(
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE pt.object_kind = ?"
                f" AND p.state IN ({ph_states})"
                f" AND pt.object_id IN ({_ph(len(part))})",
                ["source_revision", *_SUPPRESSING, *part],
            )
        )
    out.update(k for k in wanted if f"{k[0]}:{k[1]}" in suppressed_revs)
    return out


def _rfc3339_ts(value: Any) -> Optional[float]:
    """RFC3339/ISO-8601 → epoch seconds; ``None`` when unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def _window_ok(disp: str, eff: Any, vf: Any, vt: Any, now_ts: float) -> bool:
    """Read-time currency test for one ``source_state`` row — identical
    verdicts to the v5 lane's ``_current_window_ok`` (V5-14.12):
    ``active`` answers inside ``[valid_from, valid_to)``; ``superseded``
    answers only until its declared ``effective_at`` boundary; every
    other known disposition never answers; unparseable bounds are absent
    for ``superseded`` (successor already owns the answer) and open for
    ``active``; unknown dispositions are admissible — a lifecycle state
    this build does not recognize is not an assertion of non-currency."""
    if disp == "active":
        start = _rfc3339_ts(vf)
        if start is not None and now_ts < start:
            return False
        end = _rfc3339_ts(vt)
        if end is not None and now_ts >= end:
            return False
        return True
    if disp == "superseded":
        boundary = _rfc3339_ts(eff)
        return boundary is not None and now_ts < boundary
    if disp in _NEVER_CURRENT:
        return False
    return True


def _lifecycle_map(conn: sqlite3.Connection, ids: Iterable[str]) -> Optional[dict]:
    """``{source_id: admissible}`` from ``source_state``, or ``None`` when
    the table is absent — unprovisioned stores keep the pre-v5 behavior:
    no lifecycle evidence exists to apply and a missing control row is
    unresolved state, not an assertion of suppression (V5-14.16)."""
    if not has_table(conn, STATE_TABLE):
        return None
    wanted = sorted({str(i) for i in ids if i})
    if not wanted:
        return {}
    now_ts = datetime.now(timezone.utc).timestamp()
    out: dict = {}
    for part in _chunks(wanted, IN_CHUNK):
        for row in conn.execute(
            "SELECT source_id, disposition, effective_at, valid_from,"
            " valid_to FROM source_state"
            f" WHERE source_id IN ({_ph(len(part))})",
            list(part),
        ):
            out[str(row[0])] = _window_ok(
                str(row[1] or ""), row[2], row[3], row[4], now_ts
            )
    return out


def _source_scope_map(
    conn: sqlite3.Connection, ids: Iterable[str], scope_id: str
) -> Optional[dict]:
    """``{source_id: scope_ok}`` — the ``sources`` join is a veto only:
    an explicit scope mismatch can never surface (foreign-scope
    evidence), while a missing row stays admissible on the projection's
    own fenced ``scope_id`` (the projection row was only writable past
    the publication gate).  ``None`` when ``sources`` is absent."""
    if not has_table(conn, SOURCES_TABLE):
        return None
    wanted = sorted({str(i) for i in ids if i})
    out: dict = {}
    for part in _chunks(wanted, IN_CHUNK):
        for row in conn.execute(
            "SELECT source_id, scope_id FROM sources"
            f" WHERE source_id IN ({_ph(len(part))})",
            list(part),
        ):
            out[str(row[0])] = str(row[1]) == str(scope_id)
    return out


def _sources_meta(conn: sqlite3.Connection, ids: Iterable[str]) -> dict:
    """``{source_id: {source_kind, speaker_id, created_us}}`` — folded
    into source-level eligibility rows; absent table → ``{}``."""
    if not has_table(conn, SOURCES_TABLE):
        return {}
    cols = [
        c
        for c in ("source_kind", "speaker_id", "created_us")
        if c in _table_cols(conn, SOURCES_TABLE)
    ]
    wanted = sorted({str(i) for i in ids if i})
    if not cols or not wanted:
        return {}
    out: dict = {}
    for part in _chunks(wanted, IN_CHUNK):
        try:
            rows = conn.execute(
                f"SELECT source_id, {', '.join(cols)} FROM sources"
                f" WHERE source_id IN ({_ph(len(part))})",
                list(part),
            ).fetchall()
        except sqlite3.Error:
            return out
        for row in rows:
            out[str(row[0])] = {c: row[i + 1] for i, c in enumerate(cols)}
    return out


# ---------------------------------------------------------------------------
# unit-backed sub-mode
# ---------------------------------------------------------------------------


def _unit_universe(
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    elig_fn,
    deadline: _Deadline,
    stats: dict,
) -> tuple:
    """``(by_rowid, by_unit, alias, truncated)`` — the fenced, eligible
    unit universe.

    ``by_rowid``: latest-visible ``fts_rowid -> unit row`` for eligible
    units (the scoring/emission authority).  ``by_unit``:
    ``unit_id -> that rowid``.  ``alias``: EVERY fenced FTS rowid
    (including stale generations) -> ``unit_id`` — a posting hit on an
    older rebuild row still nominates the unit; scoring then reads the
    latest row's bytes (V7-30.02 rebuild coexistence).

    Carrier convention (``unit_fts_rows`` present — the §30 schema): the
    searchable universe is the carrier set joined to ``units`` for the
    §30 row.  Standalone convention (no carrier — wave-A mirror schema):
    ``unit_fts.rowid == units.rowid`` directly.
    """
    scope_id = str(getattr(ctx, "scope_id", ""))
    generation = getattr(ctx, "generation", None)
    by_rowid: dict[int, dict] = {}
    by_unit: dict[str, int] = {}
    alias: dict[int, str] = {}
    truncated = False

    carrier = has_table(conn, UNIT_FTS_ROWS)
    stats["unit_rowid_space"] = "carrier" if carrier else "units.rowid"

    if carrier:
        avail = set(_table_cols(conn, UNITS_TABLE))
        unit_cols = [c for c in _UNITS_WANT_COLS if c in avail]
        ucol_sql = ", ".join("u." + c for c in unit_cols) or "u.unit_id"
        sql = (
            f"SELECT r.row_id, r.unit_id, r.generation, {ucol_sql}"
            f" FROM {UNIT_FTS_ROWS} r"
            f" LEFT JOIN {UNITS_TABLE} u"
            "   ON u.unit_id = r.unit_id AND u.scope_id = r.scope_id"
            "   AND u.generation <= ?"
            f" WHERE r.scope_id = ? AND r.generation <= ?"
            " ORDER BY r.unit_id, r.generation, u.generation"
        )
        params = [generation, scope_id, generation]
    else:
        avail = set(_table_cols(conn, UNITS_TABLE))
        unit_cols = [c for c in _UNITS_WANT_COLS if c in avail]
        sql = (
            f"SELECT rowid, {', '.join(unit_cols)} FROM {UNITS_TABLE}"
            " WHERE scope_id = ? AND generation <= ?"
            " ORDER BY unit_id, generation"
        )
        params = [scope_id, generation]

    try:
        cur = conn.execute(sql, params)
    except sqlite3.Error:
        stats["universe"] = "query_error"
        return by_rowid, by_unit, alias, True

    n_rows = 0
    # unit_id -> latest (carrier gen, unit gen) entry at/below the fence
    latest: dict[str, dict] = {}
    while True:
        if deadline.expired():
            truncated = True
            stats["universe_scan"] = "deadline"
            break
        batch = cur.fetchmany(_PAGE)
        if not batch:
            break
        for row in batch:
            n_rows += 1
            if n_rows > UNIVERSE_SCAN_CAP:
                truncated = True
                stats["universe_scan"] = "scan_bound"
                break
            if carrier:
                rid = int(row[0])
                uid = str(row[1])
                rgen = int(row[2] or 0)
                urow = {c: row[3 + i] for i, c in enumerate(unit_cols)}
                ugen = int(urow.get("generation") or 0)
                alias[rid] = uid
                ent = latest.get(uid)
                if ent is None or (rgen, ugen) >= (ent["rgen"], ent["ugen"]):
                    latest[uid] = {
                        "rid": rid, "rgen": rgen, "ugen": ugen, "row": urow,
                    }
            else:
                rid = int(row[0])
                urow = {c: row[1 + i] for i, c in enumerate(unit_cols)}
                uid = str(urow.get("unit_id"))
                alias[rid] = uid
                ent = latest.get(uid)
                ugen = int(urow.get("generation") or 0)
                if ent is None or ugen >= ent["ugen"]:
                    latest[uid] = {"rid": rid, "ugen": ugen, "row": urow}
        if truncated:
            break
    try:
        cur.close()
    except sqlite3.Error:
        pass

    stats["unit_universe_rows"] = n_rows
    stats["unit_universe"] = len(latest)

    orphans = 0
    for uid in sorted(latest):
        ent = latest[uid]
        urow = dict(ent["row"])
        if carrier and urow.get("unit_id") is None:
            # carrier row with no visible units row at/below the fence —
            # an orphan that cannot carry source_id; never emitted.
            orphans += 1
            continue
        urow.setdefault("unit_id", uid)
        urow.setdefault("scope_id", scope_id)
        if str(urow.get("scope_id")) != scope_id:
            continue  # foreign-scope row — ineligible by fence
        if elig_fn(urow):
            rid = int(ent["rid"])
            by_rowid[rid] = urow
            by_unit[uid] = rid
    stats["carrier_orphans"] = orphans
    stats["unit_eligible"] = len(by_rowid)
    return by_rowid, by_unit, alias, truncated


def _unit_postings(
    conn: sqlite3.Connection,
    terms: list,
    text_scoped: bool,
    alias: Mapping,
    by_unit: Mapping,
    stats: dict,
) -> dict:
    """``term -> latest-eligible fts-rowid set``.

    ``unit_fts`` only *nominates* — a matched rowid (any fenced
    generation) resolves through ``alias`` to its unit, then through
    ``by_unit`` to the latest eligible rowid.  Stored field bytes remain
    the match authority downstream.
    """
    posts: dict[str, set] = {}
    match_errors = 0
    for t in terms:
        q = _fts_quote(t)
        match = f"text : {q}" if text_scoped else q
        rows = _match_rowids(conn, UNIT_FTS, match)
        if rows is None:
            match_errors += 1
            posts[t] = set()
            continue
        hit_rids: set = set()
        for rid in rows:
            uid = alias.get(rid)
            if uid is None:
                continue  # unfenced generation or foreign scope
            elig_rid = by_unit.get(uid)
            if elig_rid is not None:
                hit_rids.add(elig_rid)
        posts[t] = hit_rids
    if match_errors:
        stats["match_errors"] = match_errors
    return posts


def _unit_texts(conn: sqlite3.Connection, rowids: list) -> dict:
    """``rowid -> text field bytes`` — the match authority for tf.
    External-content and standalone ``unit_fts`` both answer this read;
    NULLs (contentless shadow) stay None and are counted honestly."""
    out: dict[int, Any] = {}
    for part in _chunks(list(rowids), IN_CHUNK):
        try:
            cur = conn.execute(
                f"SELECT rowid, text FROM {UNIT_FTS}"
                f" WHERE rowid IN ({_ph(len(part))})",
                list(part),
            )
            for rid, text in cur.fetchall():
                out[int(rid)] = text
        except sqlite3.Error:
            continue
    return out


def _avgdl_units(
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    elig_rowids: set,
    stats: dict,
) -> tuple:
    """``(avgdl, mode)`` over the eligible unit universe.

    Precedence: maintained ``lex_stats`` ``field='text'`` aggregate —
    exact only when the eligible set IS the fenced universe (the lane
    never applies universe statistics to a subset silently); then the
    ``unit_fts_docsize`` shadow's first-column varints over eligible
    rowids within the statistics bound; else ``(0, "nominated")`` and the
    caller measures the nominated set itself — disclosed, never
    relabeled (F4-11)."""
    scope_id = str(getattr(ctx, "scope_id", ""))
    generation = getattr(ctx, "generation", None)

    if has_table(conn, LEX_STATS_TABLE):
        try:
            cols = set(_table_cols(conn, LEX_STATS_TABLE))
            order = "generation DESC"
            if "stats_version" in cols:
                order += ", stats_version DESC"
            row = conn.execute(
                "SELECT n_units, total_len FROM lex_stats"
                " WHERE scope_id = ? AND generation <= ?"
                " AND field = 'text'"
                f" ORDER BY {order} LIMIT 1",
                [scope_id, generation],
            ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None and int(row[0] or 0) > 0:
            n_maint = int(row[0])
            if n_maint == len(elig_rowids):
                return float(row[1]) / n_maint, "maintained"
            stats["stats_note"] = "maintained_ignored_subset"

    cols = _table_cols(conn, UNIT_FTS)
    text_idx = cols.index("text") if "text" in cols else 0
    elig = sorted(elig_rowids)
    if elig and has_table(conn, UNIT_FTS_DOCSIZE):
        if len(elig) <= STATS_ROW_CAP:
            total = 0
            got = 0
            for part in _chunks(elig, IN_CHUNK):
                try:
                    for _rid, blob in conn.execute(
                        f"SELECT id, sz FROM {UNIT_FTS_DOCSIZE}"
                        f" WHERE id IN ({_ph(len(part))})",
                        list(part),
                    ):
                        sizes = _decode_sizes(bytes(blob))
                        total += (
                            sizes[text_idx] if text_idx < len(sizes) else 0
                        )
                        got += 1
                except sqlite3.Error:
                    break
            if got:
                if got < len(elig):
                    stats["docsize_missing"] = len(elig) - got
                return total / got, "docsize"
        else:
            stats["stats_bound"] = "eligible_over_cap"
    return 0.0, "nominated"


# ---------------------------------------------------------------------------
# source-level sub-mode
# ---------------------------------------------------------------------------


def _source_rows(
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    deadline: _Deadline,
    stats: dict,
) -> tuple:
    """``(rows, truncated, mode)`` — fenced source-revision rows.

    ``rows``: ``[(source_id, revision, doc_len, tokens_text)]`` at
    ``scope_id = ctx.scope_id AND generation <= ctx.generation`` — the
    v5 incremental fence (projection rows are stamped at write time;
    lifecycle bumps mint a new generation without rewriting unaffected
    rows, so equality would strand them).  Preferred source is
    ``source_lexical_projection`` (the ``tokens``/``doc_len`` authority);
    when only the FTS carrier exists, ``source_fts.text`` supplies the
    same normalized token stream and ``doc_len`` is counted from it.
    """
    scope_id = str(getattr(ctx, "scope_id", ""))
    generation = getattr(ctx, "generation", None)
    have_proj = has_table(conn, PROJ_TABLE)
    have_carrier = has_table(conn, SRC_FTS_ROWS) and has_table(
        conn, SRC_FTS
    )
    if not (have_proj or have_carrier):
        return [], False, "absent"

    rows: list = []
    truncated = False
    if have_proj:
        sql = (
            "SELECT source_id, revision, doc_len, tokens"
            f" FROM {PROJ_TABLE} WHERE scope_id = ? AND generation <= ?"
            " ORDER BY source_id, revision"
        )
        try:
            cur = conn.execute(sql, [scope_id, generation])
        except sqlite3.Error:
            stats["source_scan"] = "query_error"
            return [], False, "error"
        n = 0
        while True:
            if deadline.expired():
                truncated = True
                stats["source_scan"] = "deadline"
                break
            batch = cur.fetchmany(_PAGE)
            if not batch:
                break
            for sid, rev, dl, tokens in batch:
                n += 1
                if n > SOURCE_SCAN_CAP:
                    truncated = True
                    stats["source_scan"] = "scan_bound"
                    break
                rows.append(
                    (str(sid), int(rev), int(dl or 0), tokens or "")
                )
            if truncated:
                break
        try:
            cur.close()
        except sqlite3.Error:
            pass
        stats["source_rows"] = n
        return rows, truncated, "projection"

    # carrier-only fallback: latest carrier row per (source_id, revision)
    # at/below the fence, tokens fetched from source_fts by rowid.
    sql = (
        "SELECT row_id, source_id, revision, generation"
        f" FROM {SRC_FTS_ROWS} WHERE scope_id = ? AND generation <= ?"
        " ORDER BY source_id, revision, generation"
    )
    try:
        cur = conn.execute(sql, [scope_id, generation])
    except sqlite3.Error:
        stats["source_scan"] = "query_error"
        return [], False, "error"
    n = 0
    latest: dict[tuple, int] = {}
    while True:
        if deadline.expired():
            truncated = True
            stats["source_scan"] = "deadline"
            break
        batch = cur.fetchmany(_PAGE)
        if not batch:
            break
        for rid, sid, rev, _gen in batch:
            n += 1
            if n > SOURCE_SCAN_CAP:
                truncated = True
                stats["source_scan"] = "scan_bound"
                break
            latest[(str(sid), int(rev))] = int(rid)
        if truncated:
            break
    try:
        cur.close()
    except sqlite3.Error:
        pass
    stats["source_rows"] = n
    pairs = sorted(latest)
    texts: dict[int, Any] = {}
    for part in _chunks([latest[p] for p in pairs], IN_CHUNK):
        if deadline.expired():
            truncated = True
            stats["source_scan"] = "deadline"
            break
        try:
            for rid, text in conn.execute(
                f"SELECT fts_row_id, text FROM {SRC_FTS}"
                f" WHERE fts_row_id IN ({_ph(len(part))})",
                list(part),
            ):
                texts[int(rid)] = text
        except sqlite3.Error:
            continue
    for sid, rev in pairs:
        rid = latest[(sid, rev)]
        text = texts.get(rid)
        if text is None:
            stats["carrier_content_missing"] = (
                stats.get("carrier_content_missing", 0) + 1
            )
            continue
        toks = str(text).split()
        rows.append((sid, rev, len(toks), str(text)))
    return rows, truncated, "carrier"


def _source_nominated(
    conn: sqlite3.Connection,
    stoks: list,
    scope_id: str,
    generation: Any,
    stats: dict,
) -> Optional[set]:
    """``{(source_id, revision)}`` nominated through ``source_fts_idx``
    (OR'd quoted terms), fenced by the carrier's ``(scope, gen<=)`` —
    or ``None`` when the shadow is unusable (missing/erroring), in which
    case the projection scan itself decides matching."""
    if not has_table(conn, SRC_FTS_IDX) or not has_table(
        conn, SRC_FTS_ROWS
    ):
        stats["source_nomination"] = "no_fts_idx"
        return None
    match = " OR ".join(_fts_quote(t) for t in stoks[:MAX_TERMS])
    if not match:
        return None
    rowids = _match_rowids(conn, SRC_FTS_IDX, match)
    if rowids is None:
        stats["source_nomination"] = "match_error"
        return None
    stats["source_nomination"] = "fts_idx"
    stats["source_nominated_rowids"] = len(rowids)
    out: set = set()
    for part in _chunks(sorted(rowids), IN_CHUNK):
        try:
            rows = conn.execute(
                f"SELECT row_id, source_id, revision FROM {SRC_FTS_ROWS}"
                f" WHERE row_id IN ({_ph(len(part))})"
                " AND scope_id = ? AND generation <= ?",
                [*part, scope_id, generation],
            ).fetchall()
        except sqlite3.Error:
            stats["source_nomination"] = "carrier_join_error"
            return None
        for _rid, sid, rev in rows:
            out.add((str(sid), int(rev)))
    return out


def _source_gates(
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    pairs: list,
    stats: dict,
) -> tuple:
    """``(admitted_pairs, drop_counts)`` — the governance fence applied
    to source-level hits: quarantine holds (source + covering envelope),
    suppressing purges, ``source_state`` currency, and a foreign-scope
    veto through ``sources``.  A read failure inside a gate fails
    closed — every examined pair is withheld, never leaked."""
    sids = sorted({s for s, _ in pairs})
    drops = {
        "held": 0, "purged": 0, "lifecycle": 0, "scope": 0, "gate_error": 0,
    }
    try:
        held = _held_source_ids(conn, sids)
        suppressed = _suppressed_pairs(conn, pairs)
        lifecycle = _lifecycle_map(conn, sids)
        scope_map = _source_scope_map(
            conn, sids, str(getattr(ctx, "scope_id", ""))
        )
    except sqlite3.Error:
        drops["gate_error"] = len(pairs)
        return [], drops
    out: list = []
    for pair in pairs:
        sid, rev = pair
        if sid in held:
            drops["held"] += 1
            continue
        if pair in suppressed:
            drops["purged"] += 1
            continue
        if lifecycle is not None and lifecycle.get(sid, True) is False:
            drops["lifecycle"] += 1
            continue
        if scope_map is not None and scope_map.get(sid, True) is False:
            drops["scope"] += 1
            continue
        out.append(pair)
    return out, drops


# ---------------------------------------------------------------------------
# candidate accumulator
# ---------------------------------------------------------------------------


class _Hit:
    """One produced candidate before ranking."""

    __slots__ = (
        "key", "source_id", "revision", "unit_level", "score",
        "terms_hit", "coverage", "phrase", "doc_len", "row", "extra",
    )

    def __init__(self, key: str, source_id: str, revision: int,
                 unit_level: bool) -> None:
        self.key = key
        self.source_id = source_id
        self.revision = revision
        self.unit_level = unit_level
        self.score = 0.0
        self.terms_hit: list = []
        self.coverage = 0.0
        self.phrase = 0.0
        self.doc_len = 0
        self.row: dict = {}
        self.extra: dict = {}


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_source(
    ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice
) -> LaneOutput:
    """L-source (V4-08.07 carried): governed raw-source fallback —
    unit-backed via ``unit_fts``, source-level via the v5 source
    projection, never intent-gated."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula"] = FORMULA_ID
    stats["formula_status"] = FORMULA_STATUS

    deadline = _Deadline(getattr(slice, "deadline_ms", None))

    conn = _conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    generation = getattr(ctx, "generation", None)
    if generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out

    norm = getattr(qv, "norm", None)
    if norm is None:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_analysis"
        return out
    terms, _ident_flags = _query_terms(qv, stats)
    if not terms:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_terms"
        return out

    cap = getattr(slice, "cap", None)
    if cap is None:
        pool = POOLS.get(getattr(ctx, "budget", None))
        cap = pool.lane_cap if pool is not None else 200
    else:
        cap = max(0, int(cap))
    stats["cap"] = cap
    if cap == 0:
        # a zero-budget slice emits nothing — and never scans (temporal
        # lane convention); honest empty ``ok``.
        stats["mode"] = "capped"
        return out

    if deadline.expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["deadline"] = True
        return out

    eligible_fn, elig_via = _eligibility(ctx)
    stats["eligible_via"] = elig_via
    if elig_via == "unrecognized":
        # an eligibility shape the lane cannot evaluate fails CLOSED:
        # emitting without it would widen the caller's eligible set.
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_shape_unknown"
        return out

    scope_id = str(getattr(ctx, "scope_id", ""))
    partial = False
    deadline_cut = False
    hits: dict[str, _Hit] = {}

    # ======================= unit-backed mode ===========================
    unit_fts_ok = has_table(conn, UNIT_FTS)
    units_ok = has_table(conn, UNITS_TABLE)
    unit_cols = _table_cols(conn, UNIT_FTS) if unit_fts_ok else []
    text_scoped = "text" in unit_cols
    unit_emitted_pairs: set = set()

    if not unit_fts_ok or not units_ok:
        missing = [t for t, ok in
                   (("unit_fts", unit_fts_ok), ("units", units_ok))
                   if not ok]
        stats["unit_mode"] = "unavailable:" + "+".join(missing)
        # ``units`` populated but unindexed is a real coverage gap; an
        # absent units table on a pre-T0 store is the expected shape.
        if units_ok:
            try:
                n_units = conn.execute(
                    f"SELECT COUNT(*) FROM {UNITS_TABLE}"
                    " WHERE scope_id = ? AND generation <= ?",
                    [scope_id, generation],
                ).fetchone()[0]
            except sqlite3.Error:
                n_units = 0
            if n_units:
                stats["unit_mode"] += ";populated_units"
                stats["degraded"] = "unit_fts_missing"
                out.status = LaneStatus.PARTIAL
                out.reason = "unit_fts_missing"
    else:
        uni, by_unit, alias, uni_trunc = _unit_universe(
            conn, ctx, eligible_fn, deadline, stats
        )
        if uni_trunc:
            partial = True
            deadline_cut = deadline_cut or deadline.expired()
        elig_rowids = set(uni)
        n_elig = len(elig_rowids)
        out.eligible += n_elig
        out.examined += int(stats.get("unit_universe_rows", 0))

        posts = _unit_postings(
            conn, terms, text_scoped, alias, by_unit, stats
        )
        nominated = sorted({r for rs in posts.values() for r in rs})
        stats["unit_nominated"] = len(nominated)
        out.examined += len(nominated)

        if nominated and not deadline.expired():
            # df(t) is the exact eligible count — the postings are
            # already fenced + eligibility-filtered (F4-11 carried).
            df_of = {t: len(posts.get(t, ())) for t in terms}
            avgdl, stats_mode = _avgdl_units(
                conn, ctx, elig_rowids, stats
            )
            texts = _unit_texts(conn, nominated)
            if stats_mode == "nominated":
                # disclosed fallback: tokenize the nominated set itself
                lens = [
                    len(_tokenize(texts.get(r) or "")) or 1
                    for r in nominated
                ]
                avgdl = sum(lens) / len(lens) if lens else 1.0
            stats["stats"] = stats_mode
            stats["avgdl"] = round(avgdl, 4)
            stats["df"] = {t: df_of[t] for t in terms if df_of.get(t)}
            stats["n_units_eligible"] = n_elig

            no_content = 0
            term_toks = {t: _tokenize(t) for t in terms}
            qseq: list = []
            for t in terms:
                qseq.extend(term_toks[t])

            for rid in nominated:
                if deadline.expired():
                    partial = True
                    deadline_cut = True
                    stats["scoring"] = "deadline_cut"
                    break
                urow = uni[rid]
                text = texts.get(rid)
                if text is None:
                    no_content += 1
                    # the index nominated this row; token bytes are
                    # unreadable so tf is nominal (1 per matched term)
                    # — flagged in stats, never silently full-scored.
                    toks: list = []
                    counter: Optional[dict] = None
                else:
                    toks = _tokenize(text)
                    counter = {}
                    for tok in toks:
                        counter[tok] = counter.get(tok, 0) + 1
                dl = len(toks) or 1
                hit = _Hit(
                    key=str(urow.get("unit_id")),
                    source_id=str(urow.get("source_id") or ""),
                    revision=int(urow.get("revision") or 0),
                    unit_level=True,
                )
                hit.row = urow
                hit.doc_len = dl
                score = 0.0
                matched: list = []
                for t in terms:
                    if rid not in posts.get(t, ()):
                        continue
                    tf = (
                        _term_tf(term_toks[t], counter)
                        if counter is not None
                        else 1
                    )
                    if tf <= 0:
                        continue
                    score += _bm25_text(
                        tf, dl, avgdl, _idf(n_elig, df_of[t])
                    )
                    matched.append(t)
                if not matched:
                    continue  # nominated but nothing verifies
                if _phrase_hit(qseq, toks):
                    hit.phrase = 1.0
                    score += W_PHRASE
                hit.score = score
                hit.terms_hit = matched
                hit.coverage = len(matched) / len(terms) if terms else 0.0
                hits[hit.key] = hit
                unit_emitted_pairs.add((hit.source_id, hit.revision))
            if no_content:
                stats["no_field_content"] = no_content
        stats["unit_mode"] = "ok"

    # ======================= source-level mode ==========================
    srows, src_trunc, src_mode = _source_rows(conn, ctx, deadline, stats)
    if src_trunc:
        partial = True
        deadline_cut = deadline_cut or deadline.expired()
    stats["source_mode"] = src_mode
    if src_mode not in ("absent", "error"):
        stats["projection"] = NORMALIZATION_VERSION

    if srows:
        sterm_groups = _source_term_groups(terms)
        group_of = dict(zip(terms, sterm_groups))
        stoks = sorted({tok for g in sterm_groups for tok in g})
        nom = _source_nominated(conn, stoks, scope_id, generation, stats)
        if nom is not None:
            stats["source_rows_prefilter"] = len(srows)
            srows = [r for r in srows if (r[0], r[1]) in nom]
            stats["source_nominated_in"] = len(srows)

        pairs = [(r[0], r[1]) for r in srows]
        admitted_pairs, drops = _source_gates(conn, ctx, pairs, stats)
        stats["source_drops"] = drops
        admitted = set(admitted_pairs)

        # caller's eligible handle on a synthesized source-granular row —
        # a callable may still deny; a unit-id container fails closed.
        src_meta = _sources_meta(conn, [s for s, _ in admitted_pairs])
        elig_dropped = 0
        corpus: list = []
        for (sid, rev, dl, tokens) in srows:
            if (sid, rev) not in admitted:
                continue
            meta = src_meta.get(sid) or {}
            row = {
                "unit_id": f"{sid}:{rev}",
                "source_id": sid,
                "revision": rev,
                "scope_id": scope_id,
                "kind": "source",
                "unit_level": False,
                "source_kind": meta.get("source_kind"),
                "speaker_id": meta.get("speaker_id"),
                "created_us": meta.get("created_us"),
            }
            if eligible_fn(row):
                corpus.append((sid, rev, dl, tokens))
            else:
                elig_dropped += 1
        stats["eligible_dropped"] = elig_dropped
        out.eligible += len(corpus)

        # which admitted sources carry ANY fenced units row — the honest
        # "was T0 done for this source" signal behind
        # stats["source_only_orphaned"].
        have_units: set = set()
        if units_ok and admitted:
            sid_list = sorted({s for s, _ in admitted})
            for part in _chunks(sid_list, IN_CHUNK):
                try:
                    for r in conn.execute(
                        f"SELECT DISTINCT source_id FROM {UNITS_TABLE}"
                        f" WHERE source_id IN ({_ph(len(part))})"
                        " AND scope_id = ? AND generation <= ?",
                        [*part, scope_id, generation],
                    ).fetchall():
                        have_units.add(str(r[0]))
                except sqlite3.Error:
                    break

        # exact eligible-corpus statistics over the gated set (F4-11).
        n_src = len(corpus)
        total_len = sum(dl for _, _, dl, _ in corpus)
        avgdl_src = (total_len / n_src) if n_src else 0.0
        stats["n_sources_eligible"] = n_src
        stats["avgdl_source"] = round(avgdl_src, 4)
        qseq_src: list = []
        for t in terms:
            qseq_src.extend(_norm_tokens(t) or [t])

        df_src = {t: 0 for t in terms}
        scored_rows: list = []
        for (sid, rev, dl, tokens) in corpus:
            if deadline.expired():
                partial = True
                deadline_cut = True
                stats["source_scoring"] = "deadline_cut"
                break
            out.examined += 1
            doc_toks = tokens.split()
            counter: dict = {}
            for tok in doc_toks:
                counter[tok] = counter.get(tok, 0) + 1
            term_hit: list = []
            for t in terms:
                if _term_tf(list(group_of[t]), counter) > 0:
                    df_src[t] += 1
                    term_hit.append(t)
            if term_hit:
                scored_rows.append((sid, rev, dl, doc_toks, counter, term_hit))
        stats["df_source"] = {t: df_src[t] for t in terms if df_src[t]}

        covered_by_units = 0
        orphan_sources = 0
        for (sid, rev, dl, doc_toks, counter, term_hit) in scored_rows:
            if (sid, rev) in unit_emitted_pairs:
                covered_by_units += 1
                continue
            score = 0.0
            for t in term_hit:
                tf = _term_tf(list(group_of[t]), counter)
                score += _bm25_text(
                    tf, dl or len(doc_toks) or 1, avgdl_src,
                    _idf(n_src, df_src[t]),
                )
            phrase = 1.0 if _phrase_hit(qseq_src, doc_toks) else 0.0
            score += W_PHRASE * phrase
            syn = f"{sid}:{rev}"
            hit = _Hit(key=syn, source_id=sid, revision=rev,
                       unit_level=False)
            hit.score = score
            hit.phrase = phrase
            hit.doc_len = int(dl or len(doc_toks))
            hit.terms_hit = term_hit
            hit.coverage = len(term_hit) / len(terms) if terms else 0.0
            hit.row = {
                "unit_id": syn,
                "source_id": sid,
                "revision": rev,
                "scope_id": scope_id,
                "kind": "source",
            }
            if sid not in have_units:
                orphan_sources += 1
                hit.extra["no_units"] = True
            hits[syn] = hit
        stats["source_covered_by_units"] = covered_by_units
        stats["source_only_orphaned"] = orphan_sources

    # --------------------------- emit ----------------------------------
    ordered = sorted(hits.values(), key=lambda h: (-h.score, h.key))
    final = ordered[:cap]
    overflow = len(ordered) - len(final)
    if overflow:
        stats["cap_truncated"] = overflow
    stats["pool"] = len(ordered)

    nominated_via_idx = stats.get("source_nomination") == "fts_idx"
    for rank, hit in enumerate(final, start=1):
        signals: dict[str, Any] = {
            "match": (
                "unit_fts" if hit.unit_level
                else ("source_fts" if nominated_via_idx
                      else "source_projection")
            ),
            "terms_hit": list(hit.terms_hit),
            "matched_terms": list(hit.terms_hit),
            "coverage": round(hit.coverage, 6),
            "unit_level": hit.unit_level,
            "eligibility": "unit" if hit.unit_level else "source_gates",
            "doc_len": hit.doc_len,
            "bm25": round(hit.score - W_PHRASE * hit.phrase, 6),
            "phrase": hit.phrase,
            "formula": FORMULA_ID,
        }
        if hit.unit_level:
            for k in (
                "kind", "session_id", "seq", "speaker_canon",
                "perspective", "recorded_at_us", "occurred_start_us",
                "occurred_end_us",
            ):
                v = hit.row.get(k)
                if v is not None:
                    signals[k] = v
        else:
            signals["source_granular"] = True
            if hit.extra.get("no_units"):
                signals["no_units"] = True
        out.candidates.append(
            CandidateV7(
                unit_id=hit.key,
                source_id=hit.source_id,
                revision=hit.revision,
                lane=LANE_NAME,
                rank=rank,
                raw_score=hit.score,
                signals=signals,
            )
        )

    if deadline.expired():
        deadline_cut = True
    if deadline_cut:
        stats["deadline"] = True
        partial = True
    if partial and out.status == LaneStatus.OK:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline" if deadline_cut else "scan_bound"

    if out.status == LaneStatus.OK and not hits:
        # nothing produced — decide whether that is honest emptiness or
        # missing substrate (reason names the absent tables, V7-04.03).
        unit_missing = stats.get("unit_mode", "").startswith("unavailable")
        source_missing = src_mode in ("absent", "error")
        if unit_missing and source_missing:
            out.status = LaneStatus.UNAVAILABLE
            missing = []
            if not unit_fts_ok:
                missing.append("unit_fts")
            if src_mode == "absent":
                missing.append("source_projection")
            else:
                missing.append("source_scan")
            out.reason = "no_indexes:" + "+".join(missing)
    elif out.status == LaneStatus.UNAVAILABLE and not out.reason:
        out.reason = "unavailable"
    return out


class SourceLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(
        self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
    ) -> LaneOutput:
        return lane_source(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the
# pipeline's lane-module table is owned by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.SOURCE, lane_source)


__all__ = [
    "FORMULA_ID",
    "FORMULA_STATUS",
    "LANE_NAME",
    "LANE_VERSION",
    "SourceLane",
    "lane_source",
]
