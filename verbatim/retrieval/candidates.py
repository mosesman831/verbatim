"""Candidate generation and eligibility (SPEC §29).

Every source is scoped *before* ranking — filtering only after global
retrieval is forbidden (SPEC §6). Scope, purge suppression, lifecycle, and
valid-time predicates run on the union before fusion, never after.

Optional sources degrade with warnings instead of raising: a missing FTS5
index or absent embeddings must never turn recall into an error — numpy
is optional acceleration only, never required (SPEC §4, §43).
"""

from __future__ import annotations

import math
import re
import sqlite3
import time as _time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.identity import can_read
from ..embeddings.codec import Float32Codec
from ..embeddings import vectors as _vec
from ..core.types import (
    Condition,
    EndpointKind,
    MemoryKind,
    Precision,
    RecallMode,
    Scope,
    TimeInterval,
    Visibility,
    safe_json_loads,
)
from ..storage.repos import has_table as _has_table

LEXICAL_LIMIT = 40
SEMANTIC_LIMIT = 40
STRUCTURED_LIMIT = 20
GRAPH_LIMIT = 20
EPISODE_LIMIT = 20
UNION_CAP = 128
# Hard bound on the members of one bundle (SPEC_V2 §26.08).
GROUP_MEMBER_CAP = 32
# Total span bound across one result (SPEC_V2 §30.06).
SPAN_CAP = 64

# Authorization-local BM25 (V4-28.03/04): the FTS5 index only generates
# candidate identifiers; ranking uses document frequency, corpus size and
# average length measured over the query's eligible authorized corpus so
# unauthorized writes can never move an authorized score (F4-11).
_BM25_K1 = 1.2
_BM25_B = 0.75
# SQLite host-variable bound used for every chunked IN()/OR() query.
_IN_CHUNK = 400
_PAIR_CHUNK = 200   # pairs carry 2 params each
_REF_CHUNK = 150    # quarantine refs carry 3 params each

_GRAPH_EDGE_TYPES = ("context_of", "supports", "conflicts_with")

# Lifecycle states each recall mode may surface (SPEC §15, SPEC_V2 §26).
# Erased is never retrievable — purge suppression overrides every mode
# regardless (V2-16.17). ARCHIVE is the explicit operator list mode: it may
# expose authorized pending/rejected evidence, always with warnings
# (V2-13.04); quarantine stays out through suppression and state policy.
_MODE_STATES: dict[RecallMode, frozenset] = {
    RecallMode.CURRENT: frozenset({"active", "disputed"}),
    RecallMode.EXPANDED: frozenset({"active", "disputed"}),
    RecallMode.HISTORICAL: frozenset({"active", "disputed", "superseded", "archived"}),
    RecallMode.TIMELINE: frozenset(
        {"active", "disputed", "superseded", "archived", "pending", "rejected"}
    ),
    RecallMode.ARCHIVE: frozenset(
        {"pending", "active", "disputed", "superseded", "rejected", "archived"}
    ),
}

# Modes where "past knowledge" is first-class: inapplicable-but-known
# evidence stays, labeled, instead of being filtered out (V2-16.08).
_LENIENT_TIME_MODES = frozenset(
    {RecallMode.HISTORICAL, RecallMode.TIMELINE, RecallMode.ARCHIVE}
)

# Modes that may list eligible memory without a keyword query (V2-25.11).
_BROWSE_MODES = frozenset({RecallMode.TIMELINE, RecallMode.ARCHIVE})

# Purge states that suppress retrieval (previewed purges do not).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

# Prospective-record statuses that count as open plans in current recall.
_PLAN_OPEN = ("planned", "in_progress", "overdue", "unknown")


def _monotonic() -> float:
    """Indirection so tests can drive the cooperative clock deterministically."""
    return _time.monotonic()


class Deadline:
    """Cooperative millisecond work budget (SPEC_V2 §25).

    ``expired()`` is checked between candidate lanes and inside bounded
    loops; crossing the budget flips ``exceeded`` so callers can stop
    gathering and return partial results instead of running unbounded.
    ``None`` means no budget (direct internal calls).
    """

    def __init__(self, deadline_ms: Optional[float]) -> None:
        self._end = (
            None if deadline_ms is None else _monotonic() + deadline_ms / 1000.0
        )
        self.exceeded = False

    def expired(self) -> bool:
        if self._end is None:
            return False
        if _monotonic() >= self._end:
            self.exceeded = True
            return True
        return False

    def remaining_ms(self) -> Optional[float]:
        if self._end is None:
            return None
        return max(0.0, (self._end - _monotonic()) * 1000.0)


def _columns(conn: sqlite3.Connection, table: str) -> set:
    """Column names of ``table`` — additive v2 columns need detection, not
    assumption (v1 databases legitimately lack them)."""
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


@dataclass
class CandidateHit:
    """One claim surfaced by ≥1 retrieval source.

    ``claim_revision`` starts as the source-hit revision and is rewritten to
    the latest-known revision during eligibility so downstream stages always
    package the belief the store held at the ``known_at`` cutoff (§14).
    ``source_ranks`` keeps one-based ranks per source for RRF; ``reasons``
    carries internal flags mapped to reason codes at packaging time.
    ``applicability`` is the three-valued condition verdict on the resolved
    revision: "applies" | "unknown" | "does_not_apply" (V2-17).
    """

    claim_id: str
    claim_revision: int = 0
    source_ranks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    recorded_from: int = 0
    applicability: Optional[str] = None
    cond_keys: tuple = ()


class CandidateMap(dict):
    """``dict[claim_id, CandidateHit]`` plus gather-stage diagnostics.

    ``overflow`` records how many eligible claims exceeded UNION_CAP;
    ``degraded`` lists sources that were requested but inactive.
    ``kind_units`` holds pre-assembled non-claim memory-kind bundle plans
    (episodes, procedures, prospective records) for the packager.
    """

    def __init__(self) -> None:
        super().__init__()
        self.overflow = 0
        self.warnings: list = []
        self.degraded: list = []
        self.semantic_active = False
        self.deadline_exceeded = False
        self.kind_units: list = []
        self.capability_notes: dict = {}


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _chunks(seq: list, n: int):
    """Fixed-size slices — keeps IN()/OR() clauses under SQLite's
    host-variable limit at corpus scale."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class _FtsRows(list):
    """``[(claim_id, claim_revision, text_or_None)]`` plus scan diagnostics.

    Stays a plain list for every consumer (v2 ``_lexical``, v3
    ``lane_lexical`` paging, the f27 instrumenter); ``diag`` carries the
    authorization-local statistics report for capability notes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.diag: dict = {}


def _hit(out: CandidateMap, claim_id: str) -> CandidateHit:
    hit = out.get(claim_id)
    if hit is None:
        hit = CandidateHit(claim_id)
        out[claim_id] = hit
    return hit


def _repo(store: Any, name: str) -> Any:
    """Instantiate a storage repo when the parallel-built module exists.

    Call sites prefer the real repo; any construction/import failure falls
    back to the equivalent parameterized SQL below so retrieval keeps the
    same contract either way.
    """
    try:
        from ..storage import repos  # type: ignore
    except Exception:
        return None
    cls = getattr(repos, name, None)
    if cls is None:
        return None
    try:
        return cls(store)
    except Exception:
        return None


def _allowed_scopes(conn: sqlite3.Connection, reader: Scope) -> list:
    """Scope ids whose rows the requesting scope may read (SPEC §9).

    ``can_read`` decides per stored scope row; the row's own ``profile_id``
    is used so a stray foreign-profile row can never widen access.
    """
    rows = conn.execute(
        "SELECT scope_id, profile_id, principal_id, workspace_id,"
        " conversation_id, visibility FROM scopes ORDER BY scope_id"
    ).fetchall()
    allowed = []
    for (scope_id, profile_id, principal_id, workspace_id,
         conversation_id, visibility) in rows:
        owner = Scope(
            profile_id=profile_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            visibility=Visibility(visibility),
        )
        if can_read(reader, owner):
            allowed.append(scope_id)
    return allowed


def _fts5_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='facts_fts_idx'"
    ).fetchone()
    return row is not None


# Scripts unicode61 cannot segment (no whitespace between words): CJK
# ideographs/kana, Hangul, Thai, Lao, Khmer, Myanmar. A MATCH for such a
# term can never hit the glued index token, so the substring probe in
# ``_term_posting`` covers them instead (V2-25.08).
_UNSEGMENTED_RE = re.compile(
    "[゠-ヿ㐀-䶿一-鿿가-힯ก-๿ກ-໿ក-៿ဈ-ၟ]"
)


# Document tokens for the local scorer: letters/digits/marks like the
# unicode61 tokenizer, underscore excluded (it is a separator in FTS5).
_DOC_TOK_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _norm_text(text: str) -> str:
    """unicode61-style fold: NFKD, combining marks dropped, casefolded."""
    folded = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(
        c for c in folded if unicodedata.category(c) != "Mn"
    )
    return stripped.casefold()


def _fts_quote(token: str) -> str:
    """FTS5 quoted literal — same escaping ``query._fts_quote`` applies."""
    return '"' + token.replace('"', '""') + '"'


def _query_units(plan: Any) -> list:
    """Scoring units for local BM25: one per phrase/term, in plan order.

    ``toks`` is the normalized token tuple the doc matcher counts (adjacent
    occurrences for multi-token units, token membership for single ones).
    Unsegmented scripts carry no tokens — they score by folded substring
    count, mirroring the bounded-scan semantics (V2-25.08).
    """
    units: list = []
    seen: set = set()
    for raw in (*plan.phrases, *plan.terms)[:64]:
        raw = (raw or "").strip()
        if not raw:
            continue
        key = _norm_text(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        toks: tuple = ()
        if not _UNSEGMENTED_RE.search(raw):
            toks = tuple(_DOC_TOK_RE.findall(key))
            if not toks:
                # Punctuation-only residue — match it as a raw substring.
                pass
        units.append({"raw": raw, "key": key, "toks": toks})
    return units


# Bound on identifier generation per unit: a pathological common-term
# posting cannot pull the whole table. The cap is rowid-ordered — never
# bm25-ordered — so it cannot leak global corpus statistics, and reaching
# it is reported through ``diag["match_capped"]`` (F4-11).
_POSTING_CAP = 200_000


def _fts_match(conn: sqlite3.Connection, scope_ids: list,
               match_query: str, generation: int, limit: int) -> list:
    """Identifier rows for one FTS5 MATCH expression.

    Returns ``(claim_id, claim_revision, row_id)`` triples in
    deterministic rowid order — a pure postings lookup. ``bm25()`` never
    enters this pipeline: the index generates candidate identifiers only,
    and ``limit`` is a bounded-scan guard, not a rank-order truncation
    (F4-11, V4-28.04).
    """
    # Two phases: the bare MATCH emits rowids in ascending order (the
    # index's native doc order — no sort needed); scope/generation filters
    # and the claim triples come from chunked INTEGER-PRIMARY-KEY probes
    # on fts_rows. A flat JOIN+ORDER BY instead lets the planner drive
    # from fts_rows and re-evaluate MATCH per row — O(corpus × probe).
    # ``limit`` caps AFTER the scope filter so it measures authorized
    # matches, never the global posting.
    rowids = [
        r[0] for r in conn.execute(
            "SELECT rowid FROM facts_fts_idx"
            " WHERE facts_fts_idx MATCH ?",
            (match_query,),
        ).fetchall()
    ]
    out: list = []
    scope_ph = _ph(len(scope_ids))
    for chunk in _chunks(rowids, _IN_CHUNK):
        out.extend(
            conn.execute(
                "SELECT claim_id, claim_revision, row_id FROM fts_rows"
                f" WHERE row_id IN ({_ph(len(chunk))})"
                " AND projection_generation = ?"
                f" AND scope_id IN ({scope_ph})"
                " ORDER BY row_id",
                [*chunk, generation, *scope_ids],
            ).fetchall()
        )
        if len(out) >= limit:
            return out[:limit]
    return out


def _fold_sensitive(raw: str) -> bool:
    """Whether the doc side must be NFKD-folded to match this term.

    ``instr(lower(text), lower(needle))`` only folds ASCII case: an
    accented or compatibility-foldable document (``néovim``, halfwidth
    kana) can never match a folded needle, and a folded needle hides the
    fact. Non-ASCII raw terms (unsegmented scripts included) take the
    Python-fold path so doc-side compat folds match too (V4-17.07).
    """
    raw = raw or ""
    return _norm_text(raw) != raw.casefold().strip() or any(
        ord(c) > 0x7F for c in raw
    )


def _fts_fallback_scan(conn: sqlite3.Connection, scope_ids: list,
                       plan: Any, generation: int, limit: int,
                       fold_docs: bool = False) -> list:
    """Identifier rows via a bounded substring probe (index-free path).

    Used for units the FTS index cannot serve — unsegmented scripts
    (V2-25.08), punctuation-only residue — and for every unit when the
    index is absent. Membership is raw folded-substring containment:
    content-free of corpus statistics, so no statistics channel exists
    here either. Returns ``(claim_id, claim_revision, row_id)`` triples
    in deterministic rowid order, ``limit``-bounded.

    ``fold_docs`` runs the containment check in Python over
    ``_norm_text``-folded document text — the correct semantics when the
    needle or the corpus can carry combining marks or compatibility-
    foldable characters that SQL ``lower()`` cannot fold.
    """
    needles = sorted({
        _norm_text(t)
        for t in (*getattr(plan, "phrases", ()), *getattr(plan, "terms", ()))
        if (t or "").strip()
    })
    if not needles:
        return []
    if fold_docs:
        # Rowid-ordered window; Python folded containment decides, so the
        # LIMIT applies to *matches* exactly like the SQL path.
        cur = conn.execute(
            "SELECT fr.claim_id, fr.claim_revision, fr.row_id, ft.text"
            " FROM fts_rows fr"
            " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
            " WHERE fr.projection_generation = ?"
            f" AND fr.scope_id IN ({_ph(len(scope_ids))})"
            " ORDER BY fr.row_id",
            [generation, *scope_ids],
        )
        out: list = []
        seen: set = set()
        for claim_id, claim_rev, row_id, text in cur:
            folded = _norm_text(text)
            if any(n in folded for n in needles):
                key = (claim_id, claim_rev)
                if key not in seen:
                    seen.add(key)
                    out.append((claim_id, claim_rev, row_id))
                    if len(out) >= limit:
                        break
        return out
    where = " OR ".join(
        "instr(lower(ft.text), lower(?)) > 0" for _ in needles
    )
    sql = (
        "SELECT DISTINCT fr.claim_id, fr.claim_revision, fr.row_id"
        " FROM fts_rows fr"
        " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
        " WHERE fr.projection_generation = ?"
        f" AND fr.scope_id IN ({_ph(len(scope_ids))})"
        f" AND ({where})"
        " ORDER BY fr.row_id LIMIT ?"
    )
    return conn.execute(
        sql, [generation, *scope_ids, *needles, limit]
    ).fetchall()


class _UnitPlan:
    """Plan-shaped carrier so one scoring unit can flow through
    ``_fts_fallback_scan``'s plan signature."""

    __slots__ = ("phrases", "terms")

    def __init__(self, key: str) -> None:
        self.phrases: tuple = ()
        self.terms: tuple = (key,)


def _term_posting(conn: sqlite3.Connection, scope_ids: list,
                  generation: int, unit: dict, fts_ok: bool) -> tuple:
    """``(row_id set, capped)`` — rows whose sanctioned text contains ``unit``.

    Segmented units use the FTS index's own postings — a pure identifier
    lookup, never a score source (V4-28.04). Unsegmented terms (and every
    unit when the index is absent) use the parameterized substring probe;
    ``instr``+``lower()`` folds ASCII case, and the tf pass re-verifies
    candidates against the fully-folded text so recall can only gain.
    ``capped`` marks a posting that reached ``_POSTING_CAP``.
    """
    if fts_ok and unit["toks"]:
        try:
            rows = _fts_match(
                conn, scope_ids, _fts_quote(unit["raw"]),
                generation, _POSTING_CAP,
            )
        except sqlite3.OperationalError:
            rows = []
        else:
            # The index posting is authoritative for segmented units —
            # an empty match is a real df=0, not a cue to substring-scan
            # the whole corpus (F4-11 must not re-add an O(corpus) probe
            # per absent term).
            return {r[2] for r in rows}, len(rows) >= _POSTING_CAP
    try:
        rows = _fts_fallback_scan(
            conn, scope_ids, _UnitPlan(unit["key"]),
            generation, _POSTING_CAP,
            fold_docs=_fold_sensitive(unit["raw"]),
        )
    except sqlite3.Error:
        return set(), False
    return {r[2] for r in rows}, len(rows) >= _POSTING_CAP


class _StatsRequest:
    """Minimal request for corpus eligibility when the caller (e.g. the v3
    lexical lane, which applies its own admission eligibility) did not pass
    one — CURRENT-mode states, empty condition context."""

    mode = RecallMode.CURRENT
    context: dict = {}


_STATS_REQUEST = _StatsRequest()


def _corpus_fast_path(conn: sqlite3.Connection, scope_ids: list,
                      generation: int, plan: Any,
                      request: Any) -> Optional[dict]:
    """Corpus eligibility resolved in one SQL pass — the common case.

    Only valid when every Python-side eligibility stage is provably a
    no-op for this query: no suppressing purge, no active quarantine hold,
    no condition-bearing revision anywhere, and no temporal constraint.
    The remaining checks — scope, claim existence, latest-known revision
    resolution, lifecycle state — are expressed relationally, with the
    resolved revision coming from the same ``MAX(revision) over known``
    rule ``_latest_known_revisions`` applies. Returns ``None`` whenever a
    skipped stage could matter (or a probe fails) so the caller falls
    back to the full pipeline — never an approximation.
    """
    try:
        live_purge = conn.execute(
            "SELECT 1 FROM purges p"
            f" WHERE p.state IN ({_ph(len(_SUPPRESSING_STATES))}) LIMIT 1",
            list(_SUPPRESSING_STATES),
        ).fetchone() is not None
        live_hold = (
            _has_table(conn, "quarantine")
            and conn.execute(
                "SELECT 1 FROM quarantine"
                " WHERE state IN ('pending','suppressed') LIMIT 1"
            ).fetchone() is not None
        )
        has_conditions = conn.execute(
            "SELECT 1 FROM claim_revisions"
            " WHERE condition_json IS NOT NULL"
            " AND condition_json NOT IN ('', '{}') LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        return None  # cannot prove the skipped stages are no-ops
    if live_purge or live_hold or has_conditions:
        return None
    if (plan.valid_at_us is not None
            or getattr(plan, "valid_until_us", None) is not None):
        return None
    allowed = _MODE_STATES.get(request.mode)
    if not allowed:
        return None
    res_params: list = []
    known_params: list = []
    if plan.known_at_seq is None:
        known_pred = "cr.recorded_until IS NULL"
        res_pred = "recorded_until IS NULL"
    else:
        known_pred = ("cr.recorded_from <= ? AND (cr.recorded_until IS NULL"
                      " OR cr.recorded_until > ?)")
        res_pred = ("recorded_from <= ? AND (recorded_until IS NULL"
                    " OR recorded_until > ?)")
        res_params = [plan.known_at_seq, plan.known_at_seq]
        known_params = [plan.known_at_seq, plan.known_at_seq]
    params = [
        *res_params, generation, *scope_ids, *allowed, *known_params,
    ]
    # One pass returns the corpus map AND per-row document length — the
    # facts_fts join is a rowid probe per eligible row, far cheaper than
    # re-scanning the eligible set afterwards for avgdl/tf passes.
    rows = conn.execute(
        "SELECT fr.row_id, fr.claim_id, fr.claim_revision,"
        " LENGTH(ft.text) FROM fts_rows fr"
        " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
        # No claims join: an fts_row whose claim has no revision can
        # never satisfy the resolved-revision join below, so claim
        # existence is already enforced — one fewer probe per row.
        " JOIN claim_revisions cr"
        "   ON cr.claim_id = fr.claim_id AND cr.revision = fr.claim_revision"
        " JOIN (SELECT claim_id, MAX(revision) AS r FROM claim_revisions"
        f"       WHERE {res_pred} GROUP BY claim_id) res"
        "   ON res.claim_id = fr.claim_id AND res.r = fr.claim_revision"
        " WHERE fr.projection_generation = ?"
        f" AND fr.scope_id IN ({_ph(len(scope_ids))})"
        f" AND cr.state IN ({_ph(len(allowed))})"
        f" AND {known_pred}",
        params,
    ).fetchall()
    corpus: dict = {}
    dl_map: dict = {}
    for row_id, cid, rev, dlen in rows:
        corpus[row_id] = (cid, rev)
        dl_map[row_id] = int(dlen or 0)
    return corpus, dl_map


def _lexical_corpus(conn: sqlite3.Connection, store: Any,
                    scope_ids: list, generation: int, plan: Any,
                    request: Any) -> tuple:
    """``(corpus, dl_map)`` for the eligible corpus.

    ``corpus`` maps ``row_id -> (claim_id, resolved_revision)`` — the set
    of sanctioned documents the query could possibly surface: fts rows in
    authorized scopes at this projection generation, restricted to each
    claim's resolved (latest-known) revision and run through the SAME
    eligibility pipeline that later filters candidates — lifecycle states,
    purge suppression, quarantine cascade, valid time, and conditions
    (V4-28.03). Ranking statistics measured over exactly this set cannot
    be moved by unauthorized or invisible documents.

    ``dl_map`` maps ``row_id -> character length`` when it was produced
    alongside the corpus (the SQL fast path); the fallback pipeline
    returns ``None`` and the caller measures lengths itself.

    ``_corpus_fast_path`` covers the common case in one relational pass;
    whenever suppression/holds/conditions/temporal filters could matter,
    the full per-claim pipeline below is authoritative.
    """
    fast = _corpus_fast_path(conn, scope_ids, generation, plan, request)
    if fast is not None:
        return fast
    rows = conn.execute(
        "SELECT row_id, claim_id, claim_revision FROM fts_rows"
        " WHERE projection_generation = ?"
        f" AND scope_id IN ({_ph(len(scope_ids))}) ORDER BY row_id",
        [generation, *scope_ids],
    ).fetchall()
    if not rows:
        return {}, None
    corpus_hits = CandidateMap()
    for _row_id, claim_id, _rev in rows:
        _hit(corpus_hits, claim_id)
    _apply_eligibility(conn, store, corpus_hits, plan, request, scope_ids)
    eligible_keys = {
        (hit.claim_id, hit.claim_revision) for hit in corpus_hits.values()
    }
    corpus = {
        row_id: (claim_id, rev)
        for row_id, claim_id, rev in rows
        if (claim_id, rev) in eligible_keys
    }
    return corpus, None


def _fts_search(store: Any, conn: sqlite3.Connection, scope_ids: list,
                plan: Any, generation: int, limit: int, *,
                request: Any = None,
                deadline: Optional[Deadline] = None) -> list:
    """Return ``[(claim_id, claim_revision, text_or_None)]`` best-first.

    The FTS5 MATCH index (or the bounded substring scan when it is absent
    or unservable) only GENERATES candidate identifiers. Ranks come from
    BM25 computed over the query's eligible authorized corpus — document
    frequency, corpus size, and average document length all measured on
    that set — so writes outside the caller's authorization cannot move
    an authorized ordering (F4-11, V4-28.03/04). ``_FtsRows.diag`` reports
    the statistics provenance, coverage, and any bounded/deadline cut.
    """
    out = _FtsRows()
    diag = out.diag
    diag.update(
        path="none", matched=0, candidates=0, scored=0, eligible=0,
        stats_complete=True, match_capped=False, scores={},
    )
    if not scope_ids:
        return out
    units = _query_units(plan)
    if not units:
        return out
    fts_ok = _fts5_available(conn)

    # ---- identifier generation only (never the rank source) ----------
    # Per-unit postings BEFORE the corpus build: a fully-empty union means
    # no document can score, so the eligibility pass is skipped entirely.
    # Segmented units use phrase-aware MATCH rowids (identifiers only —
    # bm25() and bm25-ordered caps are global statistics and never enter
    # this pipeline); unsegmented scripts and index-absent builds use a
    # parameterized substring probe (F4-11).
    raw_units: dict = {}        # unit key -> set(row_id)
    paths: set = set()
    capped = False
    for unit in units:
        posting, hit_cap = _term_posting(
            conn, scope_ids, generation, unit, fts_ok
        )
        raw_units[unit["key"]] = posting
        capped = capped or hit_cap
        paths.add("fts_sql" if (fts_ok and unit["toks"]) else "scan")
    diag["path"] = "+".join(sorted(paths)) or "none"
    diag["match_capped"] = capped
    if not any(raw_units.values()):
        return out

    # ---- the authorization-equivalent corpus for this query ----
    req = request if request is not None else _STATS_REQUEST
    corpus, dl_map = _lexical_corpus(
        conn, store, scope_ids, generation, plan, req
    )
    diag["eligible"] = len(corpus)
    if not corpus:
        return out
    elig_rows = set(corpus)

    # df over the eligible set: posting membership ∩ corpus, exact.
    elig_postings = {
        unit["key"]: raw_units[unit["key"]] & elig_rows for unit in units
    }
    df: dict = {}
    cand_rows: set = set()
    for unit in units:
        df[unit["key"]] = len(elig_postings[unit["key"]])
        cand_rows |= elig_postings[unit["key"]]
    diag["matched"] = len(cand_rows)
    if not cand_rows:
        return out

    # Corpus statistics: N is exact (set membership); average document
    # length is measured over the same eligible rows, in character units.
    # The fast path already produced per-row lengths; the fallback path
    # measures them here (chunked, deadline-checked).
    complete = True
    if dl_map is None:
        dl_map = {}
        total_len = 0
        counted = 0
        for chunk in _chunks(sorted(elig_rows), _IN_CHUNK):
            if deadline is not None and deadline.expired():
                complete = False
                break
            cur = conn.execute(
                "SELECT fts_row_id, LENGTH(ft.text) FROM facts_fts ft"
                f" WHERE ft.fts_row_id IN ({_ph(len(chunk))})",
                chunk,
            )
            for row_id, dlen in cur.fetchall():
                dl_map[row_id] = int(dlen or 0)
                counted += 1
                total_len += int(dlen or 0)
        avgdl = (total_len / counted) if counted else 0.0
    else:
        counted = len(dl_map)
        avgdl = (sum(dl_map.values()) / counted) if counted else 0.0
    n_docs = len(elig_rows)

    # Term frequency, one chunked SQL pass over the candidate rows:
    # ``str.count`` on the SQL-``LOWER``ed text counts non-overlapping
    # folded-substring occurrences — exactly the ``REPLACE``-delta count
    # the previous formulation computed per unit (each REPLACE removal
    # consumes len(needle) chars, so delta/len == count) — while letting
    # SQLite evaluate ``LOWER`` once per row instead of once per unit.
    # Masked by each unit's eligible-posting membership (a substring
    # alone never grants membership — a doc whose only hit is inside a
    # larger token, e.g. 'deployment', still needs the index posting).
    # Posting members floor at tf=1, covering fold mismatches (diacritics
    # the index normalized but ``LOWER`` did not). All of it stays
    # read-only — it runs under ``query_only`` snapshot connections
    # where fts5vocab creation is impossible.
    unit_tf: dict = {unit["key"]: {} for unit in units}
    needles = [unit["key"] for unit in units]
    for chunk in _chunks(sorted(cand_rows), _IN_CHUNK):
        if deadline is not None and deadline.expired():
            complete = False
            break
        cur = conn.execute(
            "SELECT fts_row_id, LOWER(ft.text)"
            " FROM facts_fts ft"
            f" WHERE ft.fts_row_id IN ({_ph(len(chunk))})",
            chunk,
        )
        for row in cur.fetchall():
            row_id = row[0]
            low = row[1] or ""
            if not isinstance(low, str):
                # A BLOB that slipped into the TEXT column would make
                # LOWER return bytes; treat as uncountable rather than
                # raise — the posting-member tf=1 floor still applies.
                low = ""
            for i, unit in enumerate(units):
                if row_id in elig_postings[unit["key"]]:
                    needle = needles[i]
                    # REPLACE(x, '', '') is a no-op (count 0) — guard the
                    # str.count('') == len(x)+1 divergence explicitly.
                    count = low.count(needle) if needle else 0
                    unit_tf[unit["key"]][row_id] = max(count, 1)

    # BM25 over eligible candidate documents only (V4-28.03/04). idf is
    # per-unit and the length adjustment is per-document — both hoisted
    # out of the unit loop.
    unit_idf = {
        unit["key"]: math.log(1.0 + (n_docs - df[unit["key"]] + 0.5)
                            / (df[unit["key"]] + 0.5))
        for unit in units if df[unit["key"]]
    }
    scored: list = []
    for row_id in sorted(cand_rows):
        dl = dl_map.get(row_id)
        if dl is None:
            complete = False  # deadline cut left this row unmeasured
            continue
        len_adj = (
            1.0 - _BM25_B + _BM25_B * dl / avgdl
            if avgdl > 0 else 1.0
        )
        denom_base = _BM25_K1 * len_adj
        score = 0.0
        for key, idf in unit_idf.items():
            tfv = unit_tf[key].get(row_id, 0)
            if tfv:
                score += (
                    idf * (tfv * (_BM25_K1 + 1.0)) / (tfv + denom_base)
                )
        scored.append((row_id, score))
    # Deterministic order: score desc, rowid asc (V4-28.10).
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    top_ids = [row_id for row_id, _s in scored[:limit]]
    texts: dict = {}
    for chunk in _chunks(top_ids, _IN_CHUNK):
        if deadline is not None and deadline.expired():
            complete = False
            break
        cur = conn.execute(
            "SELECT fts_row_id, text FROM facts_fts ft"
            f" WHERE ft.fts_row_id IN ({_ph(len(chunk))})",
            chunk,
        )
        texts.update(cur.fetchall())
    for row_id, s in scored[:limit]:
        cid, rev = corpus[row_id]
        out.append((cid, rev, texts.get(row_id)))
        diag["scores"].setdefault(cid, s)
    diag["candidates"] = len(cand_rows)
    diag["scored"] = len(scored)
    diag["stats_complete"] = complete
    diag["n_docs"] = n_docs
    diag["avgdl"] = avgdl
    return out


def _fts_text(conn: sqlite3.Connection, claim_id: str, revision: int,
              generation: int) -> Optional[str]:
    row = conn.execute(
        "SELECT ft.text FROM fts_rows fr"
        " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
        " WHERE fr.claim_id=? AND fr.claim_revision=?"
        " AND fr.projection_generation=? LIMIT 1",
        (claim_id, revision, generation),
    ).fetchone()
    return row[0] if row else None


def _lexical(store: Any, conn: sqlite3.Connection, scope_ids: list,
             plan: Any, generation: int, out: CandidateMap, *,
             request: Any = None,
             deadline: Optional[Deadline] = None) -> None:
    rows = _fts_search(store, conn, scope_ids, plan, generation,
                       LEXICAL_LIMIT, request=request, deadline=deadline)
    diag = getattr(rows, "diag", None) or {}
    if diag:
        out.capability_notes["lexical"] = {
            k: v for k, v in diag.items() if k != "scores"
        }
        out.capability_notes["lexical"]["scores"] = {
            cid: round(s, 9) for cid, s in diag.get("scores", {}).items()
        }
        if not diag.get("stats_complete", True):
            out.warnings.append("lexical_stats_partial")
        if diag.get("match_capped"):
            out.warnings.append("lexical_match_capped")
    for rank, (claim_id, claim_revision, text) in enumerate(rows, start=1):
        hit = _hit(out, claim_id)
        if "lexical" in hit.source_ranks:
            continue  # a later revision row of the same claim keeps rank 1
        hit.source_ranks["lexical"] = rank
        if plan.phrases:
            if text is None:
                text = _fts_text(conn, claim_id, claim_revision, generation)
            lowered = (text or "").lower()
            if any(p.lower() in lowered for p in plan.phrases):
                if "phrase_hit" not in hit.reasons:
                    hit.reasons.append("phrase_hit")


def _structured(conn: sqlite3.Connection, scope_ids: list, plan: Any,
                out: CandidateMap) -> None:
    """Exact "all these entities" join via INTERSECT semantics (§29).

    Entity ids can narrow the authorized scope's claims, never widen it —
    the claims join keeps ``scope_id`` inside the allowed set.
    """
    entity_ids = sorted(set(plan.entity_ids))
    sql = (
        "SELECT ce.claim_id FROM claim_entities ce"
        " JOIN claims c ON c.claim_id = ce.claim_id"
        f" WHERE ce.entity_id IN ({_ph(len(entity_ids))})"
        f" AND c.scope_id IN ({_ph(len(scope_ids))})"
        " GROUP BY ce.claim_id"
        " HAVING COUNT(DISTINCT ce.entity_id) = ?"
        " ORDER BY ce.claim_id LIMIT ?"
    )
    rows = conn.execute(
        sql, [*entity_ids, *scope_ids, len(entity_ids), STRUCTURED_LIMIT]
    ).fetchall()
    for rank, (claim_id,) in enumerate(rows, start=1):
        hit = _hit(out, claim_id)
        hit.source_ranks.setdefault("structured", rank)


def _predicate(conn: sqlite3.Connection, scope_ids: list, plan: Any,
               out: CandidateMap) -> None:
    predicates = sorted(set(plan.predicates))
    rows = conn.execute(
        "SELECT claim_id FROM claims"
        f" WHERE predicate IN ({_ph(len(predicates))})"
        f" AND scope_id IN ({_ph(len(scope_ids))})"
        " ORDER BY claim_id LIMIT ?",
        [*predicates, *scope_ids, STRUCTURED_LIMIT],
    ).fetchall()
    for rank, (claim_id,) in enumerate(rows, start=1):
        hit = _hit(out, claim_id)
        hit.source_ranks.setdefault("structured", rank)


def _query_encoder(store: Any):
    """Locate an optional query encoder without importing heavyweight deps.

    Returns ``(encode_fn, encoder_id)`` or None. Probes, in order: a
    ``store.encode_query`` callable, a ``store.encoder`` object exposing the
    ``Encoder`` protocol (``encode(texts) -> [bytes]`` + ``encoder_id``),
    then ``get_encoder(store.cfg)`` when the store carries a configured
    backend. Absence simply disables the semantic source — lexical recall
    still works (§28).
    """
    encoder = getattr(store, "encode_query", None)
    if callable(encoder):
        # Normalize to the (query, *, broker, scope_ids, caller) contract —
        # a host-supplied callable ignores the egress context (it manages
        # its own transport posture, e.g. a pre-authorized host encoder).
        def _host_encode(
            query: str, *, broker=None, scope_ids=(), caller=""
        ):
            return encoder(query)

        return _host_encode, getattr(store, "encoder_id", None)
    enc = getattr(store, "encoder", None)
    if enc is None:
        # cfg-bearing stores (Engine, open_store, replay sandboxes)
        # resolve the configured backend directly — a bare Store with a
        # hashing/artifact backend configured still answers semantic
        # queries without Engine wiring.
        cfg = getattr(store, "cfg", None)
        backend = getattr(getattr(cfg, "embedding", None), "backend", "none")
        if cfg is not None and backend != "none":
            try:
                from ..embeddings import get_encoder

                enc = get_encoder(cfg)
            except Exception:
                enc = None
    if enc is not None and callable(getattr(enc, "encode", None)):
        encoder_id = getattr(enc, "encoder_id", None)
        if callable(encoder_id):
            encoder_id = encoder_id()

        def _encode(
            query: str, *, broker=None, scope_ids=(), caller=""
        ):
            # F4-04: a transport-bound encoder dispatches only through a
            # scoped permit minted by the broker — the context (scopes,
            # caller) arrives at call time from the lane that knows the
            # request. Unwired probes raise EGRESS_DENIED → the lane
            # reports semantic_unavailable (honest degrade, never bypass).
            if getattr(enc, "requires_transport_permit", False):
                if broker is None:
                    raise VerbatimError(
                        ErrorCode.EGRESS_DENIED,
                        "no transport broker for remote encoder",
                    )
                blobs = broker.encode_permitted(
                    enc,
                    [query],
                    scope_ids=tuple(scope_ids),
                    purpose="embed_query",
                    caller=caller,
                )
                return blobs[0] if blobs else None
            vectors = enc.encode([query])
            return vectors[0] if vectors else None

        return _encode, encoder_id
    return None


def _encoder_preprocessing(store: Any) -> Optional[str]:
    """Best-effort ``preprocessing_version`` of the query encoder.

    ``None`` means the lane could not verify a preprocessing generation —
    rows are then scored without that guard (the encoder_id + dtype +
    dimensions checks still apply) and the scan reports it.
    """
    enc = getattr(store, "encoder", None)
    if enc is None:
        cfg = getattr(store, "cfg", None)
        backend = getattr(getattr(cfg, "embedding", None), "backend", "none")
        if cfg is not None and backend != "none":
            try:
                from ..embeddings import get_encoder

                enc = get_encoder(cfg)
            except Exception:
                enc = None
    manifest = getattr(enc, "manifest", None)
    if callable(manifest):
        try:
            m = manifest()
        except Exception:
            return None
        if isinstance(m, dict):
            return m.get("preprocessing_version") or None
    return None


def _semantic(store: Any, conn: sqlite3.Connection, scope_ids: list,
              request: Any, out: CandidateMap,
              deadline: Optional[Deadline] = None) -> None:
    """Exact cosine over the authorized span→claim join (§28, V4-28.05).

    Skipped entirely when no embedding rows exist. The vector math lives in
    ``verbatim.embeddings.vectors`` — a pure-Python reference path with
    optional numpy acceleration — so the lane works without the
    ``semantic`` extra installed.

    The scan streams bounded batches of eligible vectors through a bounded
    top-k heap, so arbitrarily large authorized sets are searched exactly
    (no fixed input bound — F4-12). Deadline expiry stops the scan and is
    reported as ``semantic_partial`` with examined/eligible coverage in
    ``capability_notes["semantic_scan"]``; stale-input, generation, and
    malformed exclusions are counted there too (V4-29.05/06).
    """
    has_rows = conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
    if not has_rows:
        out.degraded.append("semantic")
        return
    resolved = _query_encoder(store)
    if resolved is None:
        out.degraded.append("semantic")
        out.warnings.append("semantic_unavailable")
        return
    encoder, encoder_id = resolved

    if not isinstance(encoder_id, str) or not encoder_id:
        encoder_id = getattr(store, "encoder_id", None)
    if not isinstance(encoder_id, str) or not encoder_id:
        # Unknown query-encoder identity is only safe when the table holds
        # a single embedding space — mixing spaces is meaningless cosine.
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT encoder_id FROM embeddings"
            ).fetchall()
        ]
        if len(ids) != 1:
            out.degraded.append("semantic")
            out.warnings.append("semantic_unavailable")
            return
        encoder_id = ids[0]

    try:
        raw = encoder(
            request.query,
            broker=getattr(store, "transport_broker", None),
            scope_ids=scope_ids,
            caller=getattr(getattr(request, "scope", None), "principal_id", "")
            or "",
        )
        qblob = (
            bytes(raw) if isinstance(raw, (bytes, bytearray))
            else Float32Codec.pack(list(raw))
        )
    except Exception:
        out.degraded.append("semantic")
        out.warnings.append("semantic_unavailable")
        return
    if not qblob or len(qblob) % 4:
        out.degraded.append("semantic")
        out.warnings.append("semantic_unavailable")
        return
    qvec = Float32Codec.unpack(qblob, len(qblob) // 4)
    if not all(map(math.isfinite, qvec)) or not any(qvec):
        # malformed or directionless query vector — unavailable, not silent
        out.degraded.append("semantic")
        out.warnings.append("semantic_unavailable")
        return

    # Authorized span→claim map (scoped before ranking, §6/§29); the
    # streaming exact scan comes from the embeddings module.
    pairs = conn.execute(
        "SELECT DISTINCT ce.span_id, ce.claim_id"
        " FROM claim_evidence ce"
        " JOIN claims c ON c.claim_id = ce.claim_id"
        f" WHERE c.scope_id IN ({_ph(len(scope_ids))})",
        list(scope_ids),
    ).fetchall()
    span_claims: dict[str, list[str]] = {}
    for span_id, claim_id in pairs:
        span_claims.setdefault(span_id, []).append(claim_id)

    # Claim-level score = best evidence-span score; the sink observes every
    # scored vector so claim order is exact over the whole examined set.
    best: dict[str, tuple] = {}

    def _sink(span_id: str, score: float) -> None:
        for claim_id in span_claims.get(span_id, ()):
            cur = best.get(claim_id)
            if (cur is None or score > cur[0]
                    or (score == cur[0] and span_id < cur[1])):
                best[claim_id] = (score, span_id)

    _hits, cov = _vec.search(
        conn, list(span_claims), encoder_id, qblob,
        top_k=SEMANTIC_LIMIT,
        deadline=deadline,
        preprocessing_version=_encoder_preprocessing(store),
        known_at_seq=getattr(request, "known_at_seq", None),
        sink=_sink,
    )
    out.semantic_active = True
    excluded = {
        "stale": cov.excluded_stale,
        "generation": cov.excluded_generation,
        "malformed": cov.excluded_malformed,
        "missing": cov.missing,
    }
    out.capability_notes["semantic_scan"] = {
        "eligible": cov.eligible,
        "examined": cov.examined,
        "scored": cov.scored,
        "returned": cov.returned,
        "partial": cov.partial,
        "excluded": excluded,
    }
    if cov.partial:
        out.deadline_exceeded = True
        out.warnings.append("semantic_partial")
    if any(excluded.values()):
        out.warnings.append("semantic_rows_excluded")

    # Deterministic claim order: best span score desc, then span id, then
    # claim id — same ordering the span-stream walk produced (V4-28.10).
    ordered = sorted(
        best.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0])
    )
    rank = 0
    for claim_id, _best_score in ordered:
        if rank >= SEMANTIC_LIMIT:
            break
        rank += 1
        hit = _hit(out, claim_id)
        hit.source_ranks.setdefault("semantic", rank)


def _graph(conn: sqlite3.Connection, scope_ids: list, out: CandidateMap) -> None:
    """One-hop claim adjacency from lexical/structured seeds (§29)."""
    seeds = {
        cid
        for cid, hit in out.items()
        if {"lexical", "structured"} & set(hit.source_ranks)
    }
    if not seeds:
        return
    seed_list = sorted(seeds)
    sql = (
        "SELECT source_id, target_id, source_kind, target_kind FROM edges"
        f" WHERE edge_type IN ({_ph(len(_GRAPH_EDGE_TYPES))})"
        " AND retired_event IS NULL"
        f" AND scope_id IN ({_ph(len(scope_ids))})"
        f" AND ((source_kind='claim' AND source_id IN ({_ph(len(seed_list))}))"
        f"  OR (target_kind='claim' AND target_id IN ({_ph(len(seed_list))})))"
        " ORDER BY created_event, edge_id"
    )
    params = [
        *_GRAPH_EDGE_TYPES, *scope_ids, *seed_list, *seed_list,
    ]
    neighbors: list = []
    for source_id, target_id, source_kind, target_kind in conn.execute(sql, params):
        if source_kind == "claim" and source_id in seeds and target_kind == "claim":
            neighbors.append(target_id)
        if target_kind == "claim" and target_id in seeds and source_kind == "claim":
            neighbors.append(source_id)
    rank = 0
    seen: set = set()
    for claim_id in neighbors:
        # Dedupe and enforce the limit *before* minting a hit so neighbors
        # beyond the cap never enter the union with empty source_ranks.
        if claim_id in seen:
            continue
        seen.add(claim_id)
        if rank >= GRAPH_LIMIT:
            break
        rank += 1
        _hit(out, claim_id).source_ranks["graph"] = rank


def _suppressed(store: Any, conn: sqlite3.Connection, kind: str,
                object_ids: list) -> set:
    """Suppressed object ids of one kind (SPEC §15, §40).

    Only purges that passed preview suppress retrieval.
    """
    ids = list(dict.fromkeys(object_ids))
    if not ids:
        return set()
    try:
        live = conn.execute(
            "SELECT 1 FROM purges p"
            f" WHERE p.state IN ({_ph(len(_SUPPRESSING_STATES))}) LIMIT 1",
            list(_SUPPRESSING_STATES),
        ).fetchone()
        if live is None:
            return set()  # no suppressing purge exists at all
    except sqlite3.Error:
        pass  # fail through to the real query (which may raise → caller)
    repo = _repo(store, "PurgesRepo")
    if repo is not None:
        try:
            return set(repo.suppressed_ids(kind, ids))
        except Exception:
            pass  # fall through to equivalent SQL
    out: set = set()
    for part in _chunks(ids, _IN_CHUNK):
        sql = (
            "SELECT DISTINCT pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            " WHERE pt.object_kind = ?"
            f" AND p.state IN ({_ph(len(_SUPPRESSING_STATES))})"
            f" AND pt.object_id IN ({_ph(len(part))})"
        )
        rows = conn.execute(
            sql, [kind, *_SUPPRESSING_STATES, *part]
        ).fetchall()
        out.update(row[0] for row in rows)
    return out


def _quarantine_cascade(
    conn: sqlite3.Connection, out: CandidateMap, spans_of: dict
) -> bool:
    """Withhold claims under a quarantine hold — claim ref, any evidence
    span ref, that span's source ref, or its source-envelope ref
    (V3-14.10, §34). Mutates ``out``; returns True when it ran.

    Holds are compared on the same (kind, object_id, revision) triples
    the screening paths write: spans carry their *source* revision, so a
    span ref is (span_id, source revision) and an envelope ref is
    (envelope_id, source revision)."""
    try:
        live = conn.execute(
            "SELECT 1 FROM quarantine"
            " WHERE state IN ('pending','suppressed') LIMIT 1"
        ).fetchone()
        if live is None:
            return True  # no active holds — skip the ref build entirely
        span_ids = [s for ids in spans_of.values() for s in ids]
        refs: set = set()
        env_ids_by_src: dict = {}
        for cid, hit in out.items():
            rev = hit.claim_revision if hit.claim_revision is not None else 0
            refs.add(("claim", cid, int(rev)))
        span_src: dict = {}
        if span_ids:
            src_rev: dict = {}
            for part in _chunks(sorted(set(span_ids)), _IN_CHUNK):
                rows = conn.execute(
                    "SELECT span_id, source_id, revision FROM spans"
                    f" WHERE span_id IN ({_ph(len(part))})",
                    part,
                ).fetchall()
                for span_id, source_id, revision in rows:
                    span_src[span_id] = (source_id, revision)
                    src_rev[(source_id, revision)] = source_id
                    refs.add(("span", span_id, int(revision)))
                    refs.add(("source", source_id, int(revision)))
            if _has_table(conn, "source_envelopes") and src_rev:
                for part in _chunks(sorted(src_rev), _PAIR_CHUNK):
                    env_where = " OR ".join(
                        "(source_id = ? AND revision = ?)" for _ in part
                    )
                    env_rows = conn.execute(
                        "SELECT envelope_id, source_id, revision"
                        " FROM source_envelopes WHERE " + env_where,
                        [v for pair in part for v in pair],
                    ).fetchall()
                    for env_id, sid, revision in env_rows:
                        env_ids_by_src.setdefault(
                            (sid, revision), []
                        ).append(env_id)
                        refs.add(("source_envelope", env_id, int(revision)))
        if not refs:
            return True
        held: set = set()
        for part in _chunks(sorted(refs), _REF_CHUNK):
            where = " OR ".join(
                "(object_kind = ? AND object_id = ? AND revision = ?)"
                for _ in part
            )
            flat = [v for triple in part for v in triple]
            held.update(
                (r[0], r[1], int(r[2]))
                for r in conn.execute(
                    "SELECT object_kind, object_id, revision FROM quarantine"
                    " WHERE state IN ('pending','suppressed')"
                    " AND (" + where + ")",
                    flat,
                ).fetchall()
            )
        if not held:
            return True
        for cid in list(out.keys()):
            hit = out[cid]
            rev = hit.claim_revision if hit.claim_revision is not None else 0
            if ("claim", cid, int(rev)) in held:
                del out[cid]
                continue
            for span_id in spans_of.get(cid, []):
                src = span_src.get(span_id)
                span_rev = src[1] if src else 0
                env_held = False
                if src is not None:
                    env_held = ("source", src[0], int(src[1])) in held or any(
                        ("source_envelope", eid, int(src[1])) in held
                        for eid in env_ids_by_src.get(src, ())
                    )
                if ("span", span_id, int(span_rev)) in held or env_held:
                    del out[cid]
                    break
        return True
    except sqlite3.Error:
        # Fail closed like v3 union's _should_exclude: an unreadable
        # hold table must not become a silent disclosure.
        out.clear()
        return True


def _latest_known_revisions(conn: sqlite3.Connection, claim_ids: list,
                            known_at_seq: Optional[int]) -> dict:
    """Resolve each claim's revision visible at the ``known_at`` cutoff.

    A revision is known at K iff ``recorded_from <= K`` and
    ``recorded_until`` is absent or ``> K`` (SPEC §14). With no cutoff, only
    currently-open revisions count.
    """
    rows: list = []
    for part in _chunks(list(claim_ids), _IN_CHUNK):
        rows.extend(
            conn.execute(
                "SELECT claim_id, revision, state, recorded_from,"
                " recorded_until FROM claim_revisions"
                f" WHERE claim_id IN ({_ph(len(part))})",
                part,
            ).fetchall()
        )
    resolved: dict = {}
    for claim_id, revision, state, recorded_from, recorded_until in rows:
        if known_at_seq is None:
            known = recorded_until is None
        else:
            known = recorded_from <= known_at_seq and (
                recorded_until is None or recorded_until > known_at_seq
            )
        if not known:
            continue
        current = resolved.get(claim_id)
        if current is None or revision > current[0]:
            resolved[claim_id] = (revision, state, recorded_from)
    return resolved


def _apply_eligibility(conn: sqlite3.Connection, store: Any,
                       out: CandidateMap, plan: Any, request: Any,
                       allowed_scope_ids: list) -> None:
    """Hard filters before ranking (SPEC §29): scope, known-at revision,
    lifecycle state, purge suppression (claim and span), valid time."""
    if not out:
        return
    allowed_set = set(allowed_scope_ids)

    # 1. Claim must exist in a readable scope.
    ids = list(out.keys())
    scope_of: dict = {}
    for part in _chunks(ids, _IN_CHUNK):
        rows = conn.execute(
            "SELECT claim_id, scope_id FROM claims"
            f" WHERE claim_id IN ({_ph(len(part))})",
            part,
        ).fetchall()
        scope_of.update(rows)
    for cid in ids:
        if scope_of.get(cid) not in allowed_set:
            del out[cid]
    if not out:
        return

    # 2. Latest-known revision must exist and pass the mode's state set.
    resolved = _latest_known_revisions(conn, list(out.keys()), plan.known_at_seq)
    allowed_states = _MODE_STATES[request.mode]
    for cid in list(out.keys()):
        meta = resolved.get(cid)
        if meta is None or meta[1] not in allowed_states:
            del out[cid]
            continue
        out[cid].claim_revision = meta[0]
        out[cid].recorded_from = meta[2]
    if not out:
        return

    # 3. Claim-level purge suppression.
    suppressed_claims = _suppressed(store, conn, "claim", list(out.keys()))
    for cid in suppressed_claims:
        out.pop(cid, None)
    if not out:
        return

    # 4. Span suppression + quarantine cascade both need the evidence map —
    #    skip the corpus-scale fetch entirely when neither a suppressing
    #    purge nor an active hold exists anywhere in the store.
    try:
        live_purge = conn.execute(
            "SELECT 1 FROM purges p"
            f" WHERE p.state IN ({_ph(len(_SUPPRESSING_STATES))}) LIMIT 1",
            list(_SUPPRESSING_STATES),
        ).fetchone() is not None
        live_hold = (
            _has_table(conn, "quarantine")
            and conn.execute(
                "SELECT 1 FROM quarantine"
                " WHERE state IN ('pending','suppressed') LIMIT 1"
            ).fetchone() is not None
        )
    except sqlite3.Error:
        live_purge = live_hold = True  # fail closed — run the checks
    spans_of: dict = {}
    if live_purge or live_hold:
        # A claim is withheld when ANY of its evidence spans is suppressed
        # — partial evidence could still leak the suppressed quotation's
        # content (SPEC §14, §40).
        pairs = [(cid, out[cid].claim_revision) for cid in out.keys()]
        rows: list = []
        for part in _chunks(pairs, _PAIR_CHUNK):
            where = " OR ".join("(claim_id=? AND revision=?)" for _ in part)
            flat = [v for pair in part for v in pair]
            rows.extend(
                conn.execute(
                    "SELECT claim_id, span_id FROM claim_evidence"
                    f" WHERE {where}",
                    flat,
                ).fetchall()
            )
        all_spans: list = []
        for cid, span_id in rows:
            spans_of.setdefault(cid, []).append(span_id)
            all_spans.append(span_id)
        suppressed_spans = _suppressed(store, conn, "span", all_spans)
        if suppressed_spans:
            for cid, span_ids in spans_of.items():
                if suppressed_spans & set(span_ids):
                    del out[cid]
        if not out:
            return

        # 4b. Quarantine holds (V3-14.10 binds every read surface): a claim
        #     is withheld when itself, any evidence span, or that span's
        #     source / source-envelope is held — the same cascade v3's
        #     union admission applies (§34). Quarantine rows are checked in
        #     the pending/suppressed states only.
        if _has_table(conn, "quarantine") and _quarantine_cascade(
            conn, out, spans_of
        ):
            if not out:
                return

    # 5. Valid-time applicability, only when the query pins a time or an
    #    explicit range (SPEC_V2 §16: three-valued — False is a hard
    #    constraint for current recall, None stays labeled-uncertain).
    point = plan.valid_at_us
    until = getattr(plan, "valid_until_us", None)
    if point is None and until is None:
        pass  # no temporal constraint — condition pass still applies below
    else:
        _apply_time_eligibility(conn, out, plan, request, point, until)
        if not out:
            return

    # 6. Condition applicability on the resolved revision (V2-17, V2-26.03):
    #    three-valued — does_not_apply is a hard eligibility constraint for
    #    current recall; unknown stays labeled, never silently satisfied.
    verdicts = _eval_conditions(conn, out, request)
    for cid, verdict in verdicts.items():
        hit = out.get(cid)
        if hit is None:
            continue
        hit.applicability = verdict
        if verdict == "does_not_apply" and request.mode not in _LENIENT_TIME_MODES:
            del out[cid]


def _interval_rows(
    conn: sqlite3.Connection, pairs: list
) -> dict:
    """(claim_id) → [TimeInterval] for resolved revisions.

    Reads the v2 endpoint columns when present (uncertain ranges, unbounded,
    unknown); on v1 stores the plain bounds map onto EXACT endpoints, which
    ``applicability_at`` treats with the legacy shorthand semantics.
    """
    if not pairs:
        return {}
    extended = {"start_kind", "end_kind", "from_us_hi", "until_us_hi"} <= _columns(
        conn, "valid_intervals"
    )
    rows: list = []
    for part in _chunks(list(pairs), _PAIR_CHUNK):
        where = " OR ".join("(claim_id=? AND revision=?)" for _ in part)
        flat = [v for pair in part for v in pair]
        if extended:
            rows.extend(
                conn.execute(
                    "SELECT claim_id, from_us, until_us, precision, timezone,"
                    " basis, start_kind, end_kind, from_us_hi, until_us_hi"
                    f" FROM valid_intervals WHERE {where}"
                    " ORDER BY interval_no",
                    flat,
                ).fetchall()
            )
        else:
            rows.extend(
                conn.execute(
                    "SELECT claim_id, from_us, until_us, precision, timezone,"
                    f" basis FROM valid_intervals WHERE {where}"
                    " ORDER BY interval_no",
                    flat,
                ).fetchall()
            )
    intervals_of: dict = {}
    if extended:
        for (cid, f_us, u_us, precision, timezone, basis,
             s_kind, e_kind, f_hi, u_hi) in rows:
            intervals_of.setdefault(cid, []).append(
                TimeInterval(
                    f_us, u_us, Precision(precision), timezone, basis,
                    EndpointKind(s_kind), EndpointKind(e_kind), f_hi, u_hi,
                )
            )
        return intervals_of
    for cid, f_us, u_us, precision, timezone, basis in rows:
        intervals_of.setdefault(cid, []).append(
            TimeInterval(f_us, u_us, Precision(precision), timezone, basis)
        )
    return intervals_of


def _apply_time_eligibility(
    conn: sqlite3.Connection, out: CandidateMap, plan: Any, request: Any,
    point: Optional[int], until: Optional[int],
) -> None:
    """Hard valid-time predicates before ranking (V2-26.03, V2-29.05)."""
    pairs = [(cid, out[cid].claim_revision) for cid in out.keys()]
    intervals_of = _interval_rows(conn, pairs)
    for cid in list(out.keys()):
        intervals = intervals_of.get(cid, [])
        if until is not None:
            verdicts = [
                _range_applicability(iv, point, until) for iv in intervals
            ]
        else:
            verdicts = [iv.applicability_at(point) for iv in intervals]
        hit = out[cid]
        if any(v is True for v in verdicts):
            hit.reasons.append("time_compatible")
        elif not intervals or any(v is None for v in verdicts):
            # Unknown applicability stays as uncertain evidence (V2-16.12).
            hit.reasons.append("time_unknown")
        elif request.mode in _LENIENT_TIME_MODES:
            hit.reasons.append("time_inapplicable")
        else:
            del out[cid]


def _range_applicability(
    iv: TimeInterval, a_us: Optional[int], b_us: Optional[int]
) -> Optional[bool]:
    """Three-valued "was the interval applicable somewhere in [a, b)".

    True  — a covered point provably exists inside the range.
    False — the interval is provably disjoint from the range.
    None  — possible but not certain (uncertain/unknown endpoints).
    """
    if iv.start_kind in (EndpointKind.UNBOUNDED, EndpointKind.UNKNOWN):
        s_lo = s_hi = None
    else:
        s_lo = iv.from_us
        s_hi = iv.from_us_hi if iv.from_us_hi is not None else iv.from_us
    if iv.end_kind in (EndpointKind.UNBOUNDED, EndpointKind.UNKNOWN):
        e_lo = e_hi = None
    else:
        e_lo = iv.until_us
        e_hi = iv.until_us_hi if iv.until_us_hi is not None else iv.until_us

    # Provable disjointness: interval certainly ends at/before the range
    # start, or certainly starts at/after the range end.
    if e_hi is not None and a_us is not None and e_hi <= a_us:
        return False
    if s_lo is not None and b_us is not None and s_lo >= b_us:
        return False

    # Certainty needs a provable lower bound below the range end and a
    # provable upper bound above the range start. UNBOUNDED sides impose no
    # constraint; UNKNOWN sides can never prove membership.
    start_unknown = iv.start_kind == EndpointKind.UNKNOWN or (
        iv.start_kind == EndpointKind.EXACT and s_lo is None
    )
    lower_certain = (
        (s_hi is None or b_us is None or s_hi < b_us) and not start_unknown
    )
    if iv.end_kind == EndpointKind.UNKNOWN:
        upper_certain = False
    elif e_lo is None:
        upper_certain = True  # open-ended
    else:
        upper_certain = a_us is None or e_lo > a_us
    if lower_certain and upper_certain:
        return True
    return None


def _eval_conditions(
    conn: sqlite3.Connection, out: CandidateMap, request: Any
) -> dict:
    """claim_id → "applies" | "unknown" | "does_not_apply" (V2-17).

    Evaluates ``condition_json`` of each candidate's resolved revision via
    ``Condition.evaluate`` — a pure three-valued function. Unparseable or
    missing-key conditions yield "unknown" (APPLICABILITY_UNKNOWN semantics:
    unknown is not failure, and never silently satisfies).
    """
    pairs = [(cid, hit.claim_revision) for cid, hit in out.items()]
    if not pairs:
        return {}
    context = getattr(request, "context", None) or {}
    try:
        has_conditions = conn.execute(
            "SELECT 1 FROM claim_revisions"
            " WHERE condition_json IS NOT NULL"
            " AND condition_json NOT IN ('', '{}') LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        has_conditions = True  # cannot prove emptiness — evaluate fully
    if not has_conditions:
        # Every candidate's condition is absent/empty → "applies" without
        # the corpus-scale row fetch.
        return {cid: "applies" for cid, _ in pairs}
    rows: list = []
    for part in _chunks(pairs, _PAIR_CHUNK):
        where = " OR ".join("(claim_id=? AND revision=?)" for _ in part)
        flat = [v for pair in part for v in pair]
        rows.extend(
            conn.execute(
                "SELECT claim_id, condition_json FROM claim_revisions"
                f" WHERE {where}",
                flat,
            ).fetchall()
        )
    verdicts: dict = {}
    for cid, condition_json in rows:
        hit = out[cid]
        verdict = "applies"
        keys: tuple = ()
        if condition_json:
            cond = None
            try:
                cond = Condition.from_json(safe_json_loads(condition_json))
            except Exception:
                verdict = "unknown"
                if "cond_unparseable" not in hit.reasons:
                    hit.reasons.append("cond_unparseable")
            if cond is not None:
                keys = cond.required_keys()
                try:
                    value = cond.evaluate(context)
                except Exception:
                    value = None
                verdict = (
                    "applies" if value is True
                    else "does_not_apply" if value is False
                    else "unknown"
                )
        hit.cond_keys = keys
        verdicts[cid] = verdict
    return verdicts


def _cap_union(out: CandidateMap) -> None:
    """Keep the strongest UNION_CAP eligible claims; record overflow (§29)."""
    if len(out) <= UNION_CAP:
        return
    weights = {
        "lexical": 1.0, "semantic": 1.0, "structured": 1.0,
        "graph": 0.5, "browse": 0.5, "episode": 0.5,
    }

    def score(hit: CandidateHit) -> float:
        return sum(
            weights.get(src, 0.0) / (60 + rank)
            for src, rank in hit.source_ranks.items()
        )

    ordered = sorted(
        out.values(), key=lambda h: (-score(h), -h.recorded_from, h.claim_id)
    )
    keep = {h.claim_id for h in ordered[:UNION_CAP]}
    out.overflow = len(ordered) - UNION_CAP
    for cid in list(out.keys()):
        if cid not in keep:
            del out[cid]
    out.warnings.append("candidate_overflow")


# ----------------------------------------------------------------------
# memory-kind lanes (SPEC_V2 §21-§24, §26): episodes, procedures, plans
# ----------------------------------------------------------------------


def _text_hit(text: Optional[str], plan: Any) -> bool:
    """Bounded lexical match over a row's own text fields (V2-26.01).

    The same term/phrase containment used by the sanctioned FTS fallback —
    stored text is matched, never generated.
    """
    if not text:
        return False
    lowered = text.lower()
    if any((t or "").lower() in lowered for t in plan.terms):
        return True
    return any((p or "").lower() in lowered for p in plan.phrases)


def _known_pred(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return (
        f"{p}recorded_from <= ? AND "
        f"({p}recorded_until IS NULL OR {p}recorded_until > ?)"
    )


def _gather_episodes(
    conn: sqlite3.Connection, store: Any, scope_ids: list, plan: Any,
    request: Any, out: CandidateMap, browse: bool,
) -> None:
    """Episode units: label item + member claims resolved at the cutoff."""
    known = plan.known_at_seq
    if known is None:
        pred = "recorded_until IS NULL"
        params: list = [*scope_ids]
    else:
        pred = _known_pred()
        params = [*scope_ids, known, known]
    rows = conn.execute(
        "SELECT episode_id, revision, kind, label, recorded_from FROM episodes"
        f" WHERE scope_id IN ({_ph(len(scope_ids))}) AND {pred}"
        " ORDER BY episode_id LIMIT ?",
        [*params, EPISODE_LIMIT],
    ).fetchall()
    if not rows:
        return
    suppressed = _suppressed(store, conn, "episode", [r[0] for r in rows])
    member_rows: dict = {}
    ep_ids = [r[0] for r in rows]
    if _has_table(conn, "episode_members"):
        if known is None:
            mpred = "recorded_until IS NULL"
            mparams: list = ep_ids
        else:
            mpred = _known_pred()
            mparams = [*ep_ids, known, known]
        for ep_id, okind, oid in conn.execute(
            "SELECT episode_id, object_kind, object_id FROM episode_members"
            f" WHERE episode_id IN ({_ph(len(ep_ids))}) AND {mpred}"
            " ORDER BY ord, object_id",
            mparams,
        ).fetchall():
            member_rows.setdefault(ep_id, []).append((okind, oid))
    for ep_id, revision, kind, label, recorded_from in rows:
        if ep_id in suppressed:
            continue
        members = member_rows.get(ep_id, [])
        match = _text_hit(label, plan)
        if not match and not browse:
            continue
        out.kind_units.append({
            "kind": "episode",
            "id": ep_id,
            "revision": revision,
            "text": label,
            "state": "active",
            "recorded_from": recorded_from,
            "member_claim_ids": [
                oid for okind, oid in members if okind == "claim"
            ][:GROUP_MEMBER_CAP],
            "extra": {"episode_kind": kind},
        })


def _gather_procedures(
    conn: sqlite3.Connection, store: Any, scope_ids: list, plan: Any,
    request: Any, out: CandidateMap, browse: bool,
) -> None:
    """Procedure units: task label + ordered steps; condition-aware."""
    known = plan.known_at_seq
    if request.mode in _LENIENT_TIME_MODES:
        state_pred = "state != 'retired'"
        sparams: list = []
    else:
        state_pred = "state = 'active'"
        sparams = []
    if known is None:
        kpred = "recorded_until IS NULL"
        kparams: list = []
    else:
        kpred = _known_pred()
        kparams = [known, known]
    rows = conn.execute(
        "SELECT procedure_id, revision, task_label, state, condition_json,"
        " recorded_from FROM procedures"
        f" WHERE scope_id IN ({_ph(len(scope_ids))}) AND {state_pred}"
        f" AND {kpred} ORDER BY procedure_id LIMIT ?",
        [*scope_ids, *sparams, *kparams, EPISODE_LIMIT],
    ).fetchall()
    if not rows:
        return
    suppressed = _suppressed(store, conn, "procedure", [r[0] for r in rows])
    context = getattr(request, "context", None) or {}
    step_rows: dict = {}
    pids = [r[0] for r in rows]
    if _has_table(conn, "procedure_steps"):
        for pid, rev, step_no, span_id, desc in conn.execute(
            "SELECT procedure_id, revision, step_no, span_id, description"
            f" FROM procedure_steps WHERE procedure_id IN ({_ph(len(pids))})"
            " ORDER BY step_no",
            pids,
        ).fetchall():
            step_rows.setdefault((pid, rev), []).append((step_no, span_id, desc))
    for pid, revision, label, state, cond_json, recorded_from in rows:
        if pid in suppressed:
            continue
        if not _text_hit(label, plan) and not browse:
            continue
        app = "applies"
        keys: tuple = ()
        if cond_json:
            try:
                cond = Condition.from_json(safe_json_loads(cond_json))
                keys = cond.required_keys()
                value = cond.evaluate(context)
                app = (
                    "applies" if value is True
                    else "does_not_apply" if value is False
                    else "unknown"
                )
            except Exception:
                app = "unknown"
        if app == "does_not_apply" and request.mode not in _LENIENT_TIME_MODES:
            continue
        out.kind_units.append({
            "kind": "procedure",
            "id": pid,
            "revision": revision,
            "text": label,
            "state": state,
            "recorded_from": recorded_from,
            "steps": step_rows.get((pid, revision), [])[:GROUP_MEMBER_CAP],
            "applicability": app,
            "cond_keys": keys,
        })


def _gather_prospective(
    conn: sqlite3.Connection, store: Any, scope_ids: list, plan: Any,
    request: Any, out: CandidateMap, browse: bool,
) -> None:
    """Prospective records: open plans/deadlines — never completed facts."""
    known = plan.known_at_seq
    if request.mode in _LENIENT_TIME_MODES:
        stat_pred = "1=1"
        sparams: list = []
    else:
        stat_pred = f"status IN ({_ph(len(_PLAN_OPEN))})"
        sparams = list(_PLAN_OPEN)
    if known is None:
        kpred = "recorded_until IS NULL"
        kparams: list = []
    else:
        kpred = _known_pred()
        kparams = [known, known]
    rows = conn.execute(
        "SELECT record_id, revision, owner_id, intention_text, due_us,"
        " status, recorded_from FROM prospective_records"
        f" WHERE scope_id IN ({_ph(len(scope_ids))}) AND {stat_pred}"
        f" AND {kpred} ORDER BY due_us IS NULL, due_us, record_id LIMIT ?",
        [*scope_ids, *sparams, *kparams, EPISODE_LIMIT],
    ).fetchall()
    if not rows:
        return
    suppressed = _suppressed(
        store, conn, "prospective_record", [r[0] for r in rows]
    )
    for rid, revision, owner, text, due_us, status, recorded_from in rows:
        if rid in suppressed:
            continue
        if not _text_hit(text, plan) and not browse:
            continue
        out.kind_units.append({
            "kind": "plan",
            "id": rid,
            "revision": revision,
            "text": text,
            "state": status,
            "recorded_from": recorded_from,
            "speaker_id": owner,
            "due_us": due_us,
        })


def _gather_kind_units(
    conn: sqlite3.Connection, store: Any, scope_ids: list, plan: Any,
    request: Any, out: CandidateMap, browse: bool,
) -> None:
    """Non-claim memory kinds (V2-25.02): each lane is guarded; a missing
    v2 relation is reported as an unavailable capability, not an error."""
    kinds = set(getattr(request, "memory_kinds", ()) or ())
    if not kinds:
        # An explicit empty tuple means "no non-claim kinds" — kinds are
        # opt-in; expanding () to every kind would fire deferred-
        # capability warnings on a request that asked for none.
        return
    lanes = (
        (MemoryKind.EPISODE, "episodes", _gather_episodes),
        (MemoryKind.PROCEDURE, "procedures", _gather_procedures),
        (MemoryKind.PLAN, "prospective_records", _gather_prospective),
    )
    for kind, table, fn in lanes:
        if kind not in kinds:
            continue
        if not _has_table(conn, table):
            out.degraded.append(kind.value)
            out.warnings.append(f"{kind.value}_unavailable")
            continue
        try:
            fn(conn, store, scope_ids, plan, request, out, browse)
        except Exception:
            out.degraded.append(kind.value)
            out.warnings.append(f"{kind.value}_unavailable")
    if MemoryKind.WORKING_SET in kinds:
        # No canonical working-set relation exists yet — report honestly
        # rather than scanning claim rows under a wrong label (V2-05).
        out.degraded.append("working_set")
        out.warnings.append("working_set_unavailable")


def _browse(conn: sqlite3.Connection, scope_ids: list, out: CandidateMap) -> None:
    """Bounded authorized listing for explicit timeline/archive queries
    (V2-25.11): no keyword required; eligibility still decides inclusion."""
    rows = conn.execute(
        "SELECT claim_id FROM claims"
        f" WHERE scope_id IN ({_ph(len(scope_ids))})"
        " ORDER BY created_event DESC, claim_id LIMIT ?",
        [*scope_ids, UNION_CAP],
    ).fetchall()
    for rank, (claim_id,) in enumerate(rows, start=1):
        hit = _hit(out, claim_id)
        hit.source_ranks.setdefault("browse", rank)
        if "browse" not in hit.reasons:
            hit.reasons.append("browse")


def gather(conn: sqlite3.Connection, store: Any, plan: Any,
           request: Any, generation: int,
           deadline: Optional[Deadline] = None,
           scope_ids: Optional[list] = None) -> CandidateMap:
    """Collect scoped candidates from all requested sources, then filter.

    Source order: lexical → structured → semantic → graph (graph seeds only
    from lexical/structured hits per SPEC §29). Each source is individually
    guarded: its failure degrades to a warning, never an exception. The
    cooperative ``deadline`` stops remaining lanes on expiry — partial
    results are packaged with a DEADLINE_EXCEEDED warning (V2-25).

    ``scope_ids`` may carry a pre-computed authorization set so a caller
    that already resolved scopes (e.g. for term-coverage checks) does not
    pay for the scan twice.
    """
    out = CandidateMap()
    allowed = scope_ids if scope_ids is not None else _allowed_scopes(conn, request.scope)
    if not allowed:
        return out
    deadline = deadline if deadline is not None else Deadline(None)
    include = set(request.include_sources)
    kinds = set(getattr(request, "memory_kinds", ()) or ())
    want_claims = (not kinds) or (MemoryKind.CLAIM in kinds)
    browse = request.mode in _BROWSE_MODES

    # Readiness probe (V2-25.14): an unmet declared sequence is reported,
    # never silently waited on.
    if getattr(request, "min_ready_seq", None) is not None:
        row = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM events"
        ).fetchone()
        if row is None or int(row[0]) < request.min_ready_seq:
            out.warnings.append("PROCESSING_PENDING")
            out.capability_notes["min_ready_seq"] = {
                "requested": request.min_ready_seq,
                "observed": int(row[0]) if row else 0,
            }

    def _stop() -> bool:
        if deadline.expired():
            out.deadline_exceeded = True
            return True
        return False

    if want_claims and not _stop():
        if "lexical" in include and plan.match_query:
            try:
                _lexical(store, conn, allowed, plan, generation, out,
                         request=request, deadline=deadline)
            except Exception:
                out.degraded.append("lexical")
                out.warnings.append("lexical_unavailable")

    if want_claims and not _stop():
        if "structured" in include and (plan.entity_ids or plan.predicates):
            try:
                if plan.entity_ids:
                    _structured(conn, allowed, plan, out)
                if plan.predicates:
                    _predicate(conn, allowed, plan, out)
            except Exception:
                out.degraded.append("structured")
                out.warnings.append("structured_unavailable")

    if want_claims and not _stop():
        if "semantic" in include:
            try:
                _semantic(store, conn, allowed, request, out, deadline)
            except Exception:
                out.degraded.append("semantic")
                out.warnings.append("semantic_unavailable")

    if want_claims and not _stop():
        if "graph" in include:
            try:
                _graph(conn, allowed, out)
            except Exception:
                out.degraded.append("graph")
                out.warnings.append("graph_unavailable")

    if want_claims and not _stop() and browse:
        try:
            _browse(conn, allowed, out)
        except Exception:
            out.degraded.append("browse")
            out.warnings.append("browse_unavailable")

    if not _stop():
        try:
            _gather_kind_units(conn, store, allowed, plan, request, out, browse)
        except Exception:
            out.warnings.append("kind_lanes_unavailable")

    if not _stop():
        _apply_eligibility(conn, store, out, plan, request, allowed)
    _cap_union(out)
    return out
