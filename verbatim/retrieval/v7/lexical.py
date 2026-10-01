"""V7 lexical lane — fielded BM25F over ``unit_fts`` (SPEC_V7 §32.2, V7-06).

The lane scores units with the ``bm25f/v1`` formula *exactly* as specified:

    tf~(t,d) = sum_f  w_f * tf(t,d,f) / (1 - b_f + b_f * len(d,f) / avglen(f))
    score(d) = sum_t  idf(t) * tf~(t,d) * (k1 + 1) / (tf~(t,d) + k1)
    idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    k1 = 1.2
    w:   text 1.0, entities 0.8, when 0.6, speaker 0.3, session 0.2
    b:   text 0.75, entities 0.3, when 0.0, speaker 0.0, session 0.5
    stem-channel matches contribute at w * 0.6; exact at w * 1.0; a token
    matching both channels counts once at the higher weight.

Discipline carried from the v3 source lane: the FTS indexes *nominate*
candidate rowids (exact channel ``unit_fts``, stem shadow ``unit_fts_stem``);
term frequencies and field lengths are recomputed from stored field bytes
(``SELECT <fields> FROM unit_fts``) and the ``unit_fts_docsize`` shadow, so
the index is never the match authority.  ``offsets()`` is unavailable in the
pinned SQLite build, which is why tf is counted in Python rather than read
from the auxiliary function.

Corpus statistics (V7-06.05/06/07): N, df, avglen are computed over the
*request-eligible* set.  Maintained ``lex_stats`` / ``lex_df`` rows keyed by
``(scope_id, generation)`` are the universe aggregates; when the caller's
eligible set is a proper subset the lane subtracts the ineligible units'
``docsize`` contributions (bounded by the ineligible count) and reports
``stats.stats = "corrected"``; when no maintained aggregates exist it
recomputes exactly and reports ``"exact_recomputed"``.  Universe statistics
are never silently applied to a subset.  df(t) itself is always the exact
eligible count — the per-term postings are already in hand, so intersecting
them with the eligible set is cheaper and strictly correct.

Phrase / proximity (V7-06.08): quoted query spans and adjacent term pairs
are probed once per query with FTS5 phrase / ``NEAR(…, 3)`` queries over the
``text`` field; admitted candidates carry ``signals["phrase"]`` in
{1.0, 0.5, 0.0} for the feature reranker (the lane does not score it).

Eligibility inside production (V7-05.08): posting rows are fenced by
``(scope_id, generation)`` and the caller's eligible set *before* field
bytes are fetched or scores computed; ineligible rows are never ranked.
Fetch/scoring proceeds in pages so a deadline stop reports
``partial(examined, eligible)`` honestly (V7-04.03).

Nomination budget (V75-04.02): which terms may *nominate* is selected
identifiers-first, then content terms by ascending eligible df (the df
the collected postings already measure — no extra corpus scan), capped
at ``ctx.policy.nominate_terms_max`` (r0 = ``NOMINATE_TERMS_MAX_R0`` = 32,
a Q1 constant).  This replaces the old positional ``_MAX_TERMS``
truncation that could drop the rarest term before it ever nominated.
Every parsed term still collects its posting and scores every nominated
document — the budget bounds only nomination-set membership, exactly as
the content/stopword split already does.  An optional ``df > θ·N_E``
exclusion (``ctx.policy.nominate_df_theta``, disarmed by default) can bar
over-common content terms from nominating.  Terms excluded from
nomination are reported in ``stats["nomination_dropped"]`` with their df
and the reason (``budget`` / ``df_threshold``).

Integration note (§30 mirror): the lane maps ``unit_fts.rowid`` to
``units.rowid`` (the external-content / standalone rowid convention).  Field
column order is discovered via ``PRAGMA table_info(unit_fts)``; ``docsize``
varints decode in the same order.  If ``unit_fts`` is created contentless
(field SELECT returns NULLs) the lane degrades to ``unavailable`` rather
than fabricate term frequencies.

V8 (SPEC_V8 §07): the pre-fetch df gate is an arm
(``lexical.df_floor`` = {1400, 2000, off}; V8-07.02) with declared
exemptions — identifiers, resolved entity/speaker canons, the query's
rarest content term, and any term whose gating would leave zero
nominating terms — reported as ``coverage.lexical.df_gate`` =
``{floor, gated, exempt}`` on ``stats["coverage_lexical"]`` (the
``coverage_dense`` convention; the pipeline lifts the block verbatim).
Thin nomination triggers the bounded rescue (``lexical.K_rescue`` = 64,
``lexical.rescue_rows`` = 4096; V8-07.03, §21.4) whose produced rows
carry ``signals["rescue"]`` and are reported as
``coverage.lexical.rescue`` = ``{fired, terms, rows, produced}``.
Maintained ``unit_doclen`` rows replace ``unit_fts_docsize`` reads when
present (V8-07.04); ``porter_stem``/``_stem_of`` are memoized in a
bounded process LRU (``lexical.stem_lru`` = 65536; V8-07.05); and
``lexical.single_match`` (off; V8-07.06) collapses nomination to ≤1
statement per channel with byte-identical term→rowid sets.  All five
arms resolve off ``ctx.policy`` (attribute, ``params``/``knobs``/``arms``
mapping, or manifest — the dense-lane carrier convention); absent
everywhere they hold the §23 defaults.

V8.5 (SPEC_V8_5 §05): the nominated set then passes the co-occurrence
filter — a unit must match ≥ ``lexical.cooc_min`` (2) nominated terms
(``lexical.nom_cooc`` = on; V85-05.02), with identifiers and terms whose
measured eligible df is below ``lexical.rare_df`` (4; V85-05.03)
exempted.  An empty filtered set falls back to the union and reports
``coverage_lexical.cooc.fallback = "union"``.

All constants are ``provisional/v7-r0`` unless tagged otherwise.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    NOMINATE_TERMS_MAX_R0,
    CandidateV7,
    LaneContextV7,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    POOLS,
    QueryViewV7,
)
from ...storage.repos import has_table

LANE = "lex"

# ---------------------------------------------------------------------------
# bm25f/v1 constants — provisional/v7-r0 (§32.2)
# ---------------------------------------------------------------------------

FORMULA_ID = "bm25f/v1"
K1 = 1.2
FIELD_W: dict[str, float] = {
    "text": 1.0,
    "entities": 0.8,
    "when": 0.6,
    "speaker": 0.3,
    "session": 0.2,
}
FIELD_B: dict[str, float] = {
    "text": 0.75,
    "entities": 0.3,
    "when": 0.0,
    "speaker": 0.0,
    "session": 0.5,
}
STEM_SCALE = 0.6  # stem-channel field-weight multiplier (§32.2)

#: §23 ``context.ctx_field_weight`` (V8-06.03) — BM25F weight of the
#: optional indexed ``ctx`` field (±W same-session neighbor-turn text).
#: Off by default; ``True`` selects the 0.5 prior. The field enters
#: scoring only when armed *and* present in the ``unit_fts`` schema —
#: an armed arm on a schema without the column is reported honestly,
#: never fabricated.
CTX_FIELD_NAME = "ctx"
CTX_FIELD_W_DEFAULT = 0.5

UNITS_TABLE = "units"
FTS_TABLE = "unit_fts"
STEM_TABLE = "unit_fts_stem"
TRI_TABLE = "unit_fts_tri"
VOCAB_TABLE = "unit_fts_vocab"  # fts5vocab shadow used by the fuzzy lane
STATS_TABLE = "lex_stats"
DF_TABLE = "lex_df"
DOCLEN_TABLE = "unit_doclen"  # V8-07.04 maintained field lengths (§19)

_PAGE = 256          # field-fetch page size (bounded IN chunks)
_IN_CHUNK = 400      # < SQLITE_MAX_VARIABLE_NUMBER
# Pre-fetch df gate (beat-it r8): a term whose maintained df floods the
# eligible corpus costs a multi-thousand-rowid MATCH for near-zero
# discrimination.  The gate is ``max(_DF_PREFETCH_FLOOR, θ·N_eligible)``
# — the floor keeps the gate corpus-size-safe (a 1400-doc term is a flood
# on any scale this lane serves) and ``θ`` rides the policy's
# ``nominate_df_theta`` when armed, else this default.
_DF_PREFETCH_FLOOR = 1400
_DF_PREFETCH_THETA = 0.2
# V8 §23 arm defaults — resolved off ctx.policy, these are the shipped
# priors (V8-00.11: priors, not constants).
_K_RESCUE_DEFAULT = 64        # lexical.K_rescue     (V8-07.03)
_RESCUE_ROWS_DEFAULT = 4096   # lexical.rescue_rows  (V8-07.03)
_STEM_LRU_DEFAULT = 65536     # lexical.stem_lru     (V8-07.05)
_RESCUE_CHECK = 512           # §21.4: deadline check every 512 rows
# V8.5 §05.02/05.03 ship-set arms — the measured co-occurrence
# nomination priors (``policy.PARAM_DEFAULTS`` declares the same values;
# the read site keeps its own §23 priors like every other arm).
_NOM_COOC_DEFAULT = True      # lexical.nom_cooc   (V85-05.02)
_COOC_MIN_DEFAULT = 2         # lexical.cooc_min   (V85-05.02)
_RARE_DF_DEFAULT = 4          # lexical.rare_df    (V85-05.03)
# The nomination budget default lives on the contract —
# ``types_v7.NOMINATE_TERMS_MAX_R0`` (32, the pre-V7.5 ``_MAX_TERMS``
# envelope) — so the policy channel is the single source; never hardcode
# a second value here.
_UNITS_COLS = (
    "unit_id", "source_id", "revision", "scope_id", "kind",
    "parent_unit_id", "session_id", "seq", "speaker_canon", "perspective",
    "recorded_at_us", "occurred_start_us", "occurred_end_us",
    "occurred_precision", "occurred_source", "byte_start", "byte_end",
    "generation",
)


# ---------------------------------------------------------------------------
# Small shared machinery (also imported by fuzzy.py — same owner)
# ---------------------------------------------------------------------------


class _Deadline:
    """Relative per-lane slice budget (LaneSlice.deadline_ms)."""

    __slots__ = ("t_end",)

    def __init__(self, ms: float) -> None:
        try:
            budget = float(ms)
        except (TypeError, ValueError):
            budget = 0.0
        self.t_end = time.monotonic() + max(budget, 0.0) / 1000.0

    def expired(self) -> bool:
        return time.monotonic() >= self.t_end


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Resolve the caller-pinned read connection.

    Order: an explicit ``ctx.conn`` (pipeline-pinned snapshot), then
    ``ctx.store`` itself when it is already a connection, then
    ``ctx.store.conn``, then the Store's thread-local reader.  The lane
    never opens a transaction of its own (V7-05.06).
    """
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


def _fts_quote(term: str) -> str:
    """One MATCH-safe literal: double-quoted, inner quotes doubled."""
    return '"' + str(term).replace('"', '""') + '"'


def _match_rowids(conn: sqlite3.Connection, table: str,
                  match: str) -> Optional[set]:
    """Universe-wide rowids matching an FTS5 query; ``None`` on error
    (syntax or missing table) — callers degrade honestly, never crash."""
    try:
        cur = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (match,)
        )
        return {int(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return None


def _fts_columns(conn: sqlite3.Connection, table: str) -> list:
    """Declared FTS5 column order — docsize varints decode against it."""
    try:
        return [str(r[1]) for r in conn.execute(
            f"PRAGMA table_info({table})")]
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


def _field_lens(conn: sqlite3.Connection, docsize_table: str,
                rowids: list, columns: list) -> dict:
    """``rowid -> {field: token_count}`` from the docsize shadow.

    Missing docsize rows contribute zeros.  When the shadow itself is
    absent callers must fall back to counting tokens from field text.
    """
    out: dict[int, dict] = {}
    if not rowids:
        return out
    if not has_table(conn, docsize_table):
        return out
    for chunk in _chunks(list(rowids), _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        try:
            cur = conn.execute(
                f"SELECT id, sz FROM {docsize_table} WHERE id IN ({ph})",
                chunk,
            )
        except sqlite3.Error:
            return out
        for rid, blob in cur.fetchall():
            sizes = _decode_sizes(bytes(blob))
            out[int(rid)] = {
                f: (sizes[i] if i < len(sizes) else 0)
                for i, f in enumerate(columns)
            }
    return out


def _fetch_fields(conn: sqlite3.Connection, table: str, rowids: list,
                  columns: list) -> dict:
    """``rowid -> {field: str}`` — the stored field bytes that are the
    match authority for tf.  NULLs (contentless table) stay None so the
    caller can detect an unreadable index honestly."""
    out: dict[int, dict] = {}
    if not rowids or not columns:
        return out
    cols = ",".join('"' + c.replace('"', '""') + '"' for c in columns)
    for chunk in _chunks(list(rowids), _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        try:
            cur = conn.execute(
                f"SELECT rowid, {cols} FROM {table} WHERE rowid IN ({ph})",
                chunk,
            )
        except sqlite3.Error:
            return out
        for row in cur.fetchall():
            out[int(row[0])] = {
                c: row[i + 1] for i, c in enumerate(columns)
            }
    return out


def _doclens(conn: sqlite3.Connection, uni: _Universe, rowids: list,
             columns: list, scope_id: Optional[str],
             generation: Optional[int]) -> dict:
    """``rowid -> {field: len}`` from maintained ``unit_doclen`` (V8-07.04).

    The table is keyed ``(unit_id, generation, field)`` — rowids map
    through ``uni.by_rowid`` — and fenced exactly like ``units`` readers
    (V8-19.04): ``generation <= pinned``, latest row per
    ``(unit_id, field)`` wins (DESC + first-seen).  ``scope_id``/``len``
    are read but only same-scope rows qualify.  One batched statement per
    ``_IN_CHUNK`` of candidate units; units with no row stay absent — the
    caller's per-unit fallback counts field bytes (the match authority)
    rather than mixing deriver versions inside a field (V8-19.05).
    """
    out: dict[int, dict] = {}
    if not rowids or not has_table(conn, DOCLEN_TABLE):
        return out
    rid_of: dict[str, int] = {}
    for rid in rowids:
        u = uni.by_rowid.get(int(rid))
        if u is not None:
            rid_of.setdefault(str(u["unit_id"]), int(rid))
    if not rid_of:
        return out
    wanted = {str(f) for f in columns}
    for chunk in _chunks(list(rid_of), _IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        if generation is None:
            sql = (f"SELECT unit_id, field, len FROM {DOCLEN_TABLE}"
                   f" WHERE scope_id = ? AND unit_id IN ({ph})"
                   " ORDER BY generation DESC")
            params: list = [scope_id, *chunk]
        else:
            sql = (f"SELECT unit_id, field, len FROM {DOCLEN_TABLE}"
                   f" WHERE scope_id = ? AND generation <= ?"
                   f" AND unit_id IN ({ph}) ORDER BY generation DESC")
            params = [scope_id, generation, *chunk]
        try:
            cur = conn.execute(sql, params)
        except sqlite3.Error:
            return out
        for uid, f, ln in cur.fetchall():
            f = str(f)
            rid = rid_of.get(str(uid))
            if rid is None or f not in wanted:
                continue
            # first-seen at DESC order = the latest generation at/below pin
            out.setdefault(rid, {}).setdefault(f, int(ln))
    return out


def _lens_for(conn: sqlite3.Connection, uni: _Universe, rowids: list,
              columns: list, scope_id: Optional[str],
              generation: Optional[int]) -> dict:
    """V8-07.04: field lengths from ``unit_doclen`` when the maintained
    table exists — one batched read per chunk, zero ``unit_fts_docsize``
    statements — else the docsize shadow decode it replaces.  When
    neither exists the empty map leaves the caller's field-byte counts as
    the (identical) fallback."""
    if has_table(conn, DOCLEN_TABLE):
        return _doclens(conn, uni, rowids, columns, scope_id, generation)
    return _field_lens(conn, FTS_TABLE + "_docsize", rowids, columns)


# --- unicode61 + remove_diacritics 2 mirror (matching projection) ---------


def _fold(text: str) -> str:
    """NFKC -> casefold -> strip combining marks (the §32.1 matching
    projection; mirrors norm_v2.fold so query/document terms compare in
    the same space).  ASCII fast path: NFKC is the identity on ASCII and
    no combining marks exist, so casefold alone is the projection."""
    s = str(text)
    if s.isascii():
        return s.casefold()
    norm = unicodedata.normalize("NFKC", s).casefold()
    decomp = unicodedata.normalize("NFKD", norm)
    return "".join(c for c in decomp if not unicodedata.category(c).startswith("M"))


_TOKEN_RE = re.compile(r"[^\W_]+")


def _tokenize(text: str) -> list:
    """unicode61-equivalent token stream: fold first (the §32.1 matching
    projection), then maximal runs of Unicode L*/N* characters — Python's
    ``[^\\W_]`` is ``str.isalnum`` minus underscore, which is exactly the
    unicode61 token-character class.  Deterministic and stdlib-only; used
    to count tf on stored field bytes (the match authority)."""
    if not text:
        return []
    if text.isascii():
        return _TOKEN_RE.findall(text.casefold())
    return _TOKEN_RE.findall(_fold(text))


# --- Porter stemmer (stem-channel tf; validated against FTS5 'porter') ----


def _p_cons(w: str, i: int) -> bool:
    ch = w[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == 0 or not _p_cons(w, i - 1)
    return True


def _p_measure(w: str) -> int:
    """Porter m of w = count of VC sequences."""
    n = 0
    i = 0
    j = len(w) - 1
    while True:
        if i > j:
            return n
        if not _p_cons(w, i):
            break
        i += 1
    i += 1
    while True:
        while True:
            if i > j:
                return n
            if _p_cons(w, i):
                break
            i += 1
        i += 1
        n += 1
        while True:
            if i > j:
                return n
            if not _p_cons(w, i):
                break
            i += 1
        i += 1


def _p_has_vowel(w: str) -> bool:
    return any(not _p_cons(w, i) for i in range(len(w)))


def _p_dbl_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _p_cons(w, len(w) - 1)


def _p_cvc(w: str) -> bool:
    n = len(w)
    return (n >= 3 and _p_cons(w, n - 3) and not _p_cons(w, n - 2)
            and _p_cons(w, n - 1) and w[-1] not in "wxy")


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("abli", "able"), ("bli", "ble"),
    ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
    ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def porter_stem(word: str) -> str:
    """Classic Porter (1980) stemmer; byte-identical to the FTS5 'porter'
    tokenizer on the validation wordlist used in tests."""
    w = str(word).lower()
    if len(w) <= 2:
        return w
    # step 1a — plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # step 1b — -eed / -ed / -ing
    flag = False
    if w.endswith("eed"):
        if _p_measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _p_has_vowel(w[:-2]):
        w = w[:-2]
        flag = True
    elif w.endswith("ing") and _p_has_vowel(w[:-3]):
        w = w[:-3]
        flag = True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _p_dbl_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _p_measure(w) == 1 and _p_cvc(w):
            w += "e"
    # step 1c — terminal y
    if w.endswith("y") and _p_has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # step 2 — suffixes, m > 0 on the stem
    for suf, rep in _STEP2:
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if _p_measure(stem) > 0:
                w = stem + rep
            break
    # step 3 — m > 0
    for suf, rep in _STEP3:
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if _p_measure(stem) > 0:
                w = stem + rep
            break
    # step 4 — m > 1
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if _p_measure(stem) > 1:
                w = stem
            break
    if w.endswith("ion"):
        stem = w[:-3]
        if stem and stem[-1] in "st" and _p_measure(stem) > 1:
            w = stem
    # step 5a — terminal e
    if w.endswith("e"):
        stem = w[:-1]
        m1 = _p_measure(stem)
        if m1 > 1 or (m1 == 1 and not _p_cvc(stem)):
            w = stem
    # step 5b — terminal double l
    if _p_measure(w) > 1 and _p_dbl_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w


# --- stem memoization (V8-07.05) -------------------------------------------
#
# ``porter_stem`` is a pure function of the token, so its results are
# memoized in a bounded process-level LRU keyed by the exact surface
# form.  ``lexical.stem_lru`` (default 65,536) sets the bound per
# request — 0 disables caching; memoization never changes a score.

_STEM_LRU: OrderedDict = OrderedDict()
_STEM_LRU_LOCK = threading.Lock()
_STEM_LRU_CAP = _STEM_LRU_DEFAULT


def _configure_stem_lru(cap: int) -> None:
    """Apply the ``lexical.stem_lru`` arm: resize + trim the process LRU."""
    global _STEM_LRU_CAP
    _STEM_LRU_CAP = int(cap)
    with _STEM_LRU_LOCK:
        if _STEM_LRU_CAP <= 0:
            _STEM_LRU.clear()
        else:
            while len(_STEM_LRU) > _STEM_LRU_CAP:
                _STEM_LRU.popitem(last=False)


def _porter_memo(tok: str) -> str:
    """``porter_stem`` through the bounded LRU (V8-07.05).

    Deterministic and thread-safe: the compute runs outside the lock —
    a racing duplicate writes the identical value.  ``cap <= 0`` bypasses
    the cache entirely (the unmemoized arm for parity measurement)."""
    if _STEM_LRU_CAP <= 0:
        return porter_stem(tok)
    key = str(tok)
    with _STEM_LRU_LOCK:
        hit = _STEM_LRU.get(key)
        if hit is not None:
            _STEM_LRU.move_to_end(key)
            return hit
    stem = porter_stem(key)
    with _STEM_LRU_LOCK:
        _STEM_LRU[key] = stem
        _STEM_LRU.move_to_end(key)
        while len(_STEM_LRU) > _STEM_LRU_CAP:
            _STEM_LRU.popitem(last=False)
    return stem


# --- universe / eligibility / corpus statistics ----------------------------


@dataclass
class _Universe:
    """``(scope_id, generation)``-fenced unit rows keyed by FTS rowid."""

    by_rowid: dict  # int -> unit row dict


#: Per-context memo for ``_universe`` — the fenced unit set is a pure
#: function of the caller's pinned read snapshot, and every lane
#: invocation under one ``ctx`` (peers + V7-05.13 facet calls) re-runs
#: the identical scan.  ``(id(share), id(conn), scope_id, generation,
#: need_full) -> (share, conn, _Universe, scan_seconds)``; holding the
#: share object and the connection keeps both ``id``s unrecyclable while
#: the entry lives.  ``share=None`` (direct calls, tests) always scans
#: fresh.  A hit is served only when the caller's deadline could not
#: plausibly cut an identical fresh scan — an expired or tight deadline
#: falls through to a real scan — and only completed (non-truncated)
#: scans are ever cached.
_UNI_MEMO: dict = {}
_UNI_MEMO_MAX = 32
_UNI_SERVE_MARGIN_S = 0.001


def _universe(conn: sqlite3.Connection, scope_id: str,
              generation: Optional[int], deadline: _Deadline,
              need_full: bool = False, share: Any = None) -> tuple:
    """(by_rowid, truncated) — one indexed scan of the fenced unit set.
    Minimal ``(unit_id, source_id, revision)`` rows suffice for fencing +
    output; the full §30 column set is fetched only when the caller's
    eligibility is a callable that must see the whole unit row."""
    key = (id(share), id(conn), scope_id, generation, bool(need_full))
    ent = _UNI_MEMO.get(key) if share is not None else None
    t_end = getattr(deadline, "t_end", None)
    if ent is not None and ent[0] is share and ent[1] is conn \
            and t_end is not None:
        remaining = t_end - time.monotonic()
        if remaining <= 0:
            # A fresh scan whose deadline is already expired executes the
            # SELECT, then stops at the first ``expired()`` check and
            # reports ``({}, truncated=True)`` — replay that verdict.
            return _Universe({}), True
        if remaining > ent[3] + _UNI_SERVE_MARGIN_S:
            # The budget could host an identical fresh scan end-to-end —
            # it would complete and return exactly this universe.
            return ent[2], False
        # Marginal budget: a fresh scan could honestly truncate
        # mid-read — run it for real.

    t0 = time.monotonic()
    cols = ",".join(_UNITS_COLS if need_full else
                    ("unit_id", "source_id", "revision"))
    sel_cols = _UNITS_COLS if need_full else ("unit_id", "source_id", "revision")
    # Retrieval surface is leaf units only: ``session``/``episode`` rows
    # restate their member turns byte-for-byte — they double candidate-pool
    # occupancy and document frequency without adding evidence
    # (sentence windows are covered by their parent turn's text).
    if generation is None:
        sql = (f"SELECT rowid, {cols} FROM {UNITS_TABLE}"
               " WHERE scope_id = ? AND kind = 'turn'")
        params: list = [scope_id]
    else:
        # V7-30.02: the fence is ``generation <= pinned`` — re-projection
        # bumps the pinned generation while a unit's older rows stay
        # valid.  Latest row per unit_id wins (ORDER BY generation DESC
        # seeds the newest rowid first; first-seen keeps it).
        sql = (f"SELECT rowid, {cols} FROM {UNITS_TABLE}"
               " WHERE scope_id = ? AND generation <= ?"
               " AND kind = 'turn'"
               " ORDER BY unit_id, generation DESC")
        params = [scope_id, generation]
    out: dict[int, dict] = {}
    seen_uid: set = set()
    truncated = False
    try:
        cur = conn.execute(sql, params)
    except sqlite3.Error:
        return _Universe(out), truncated
    while True:
        if deadline.expired():
            truncated = True
            break
        batch = cur.fetchmany(1024)
        if not batch:
            break
        for row in batch:
            rec = {c: row[i + 1] for i, c in enumerate(sel_cols)}
            uid = str(rec.get("unit_id"))
            if uid in seen_uid:
                continue
            seen_uid.add(uid)
            out[int(row[0])] = rec
    uni = _Universe(out)
    if not truncated and share is not None:
        _UNI_MEMO[key] = (share, conn, uni, time.monotonic() - t0)
        while len(_UNI_MEMO) > _UNI_MEMO_MAX:
            _UNI_MEMO.pop(next(iter(_UNI_MEMO)))
    return uni, truncated


def _elig_mode(elig: Any) -> str:
    """Dispatch order shared by ``_universe`` (need_full) and
    ``_resolve_eligibility`` — keep the two in lockstep."""
    if elig is None:
        return "all"
    if getattr(elig, "unit_ids", None) is not None:
        return "set"
    if isinstance(elig, (set, frozenset)) or (
        hasattr(elig, "__iter__") and hasattr(elig, "__contains__")
        and not callable(elig)
    ):
        return "set"
    if callable(elig):
        return "callable"
    if hasattr(elig, "__contains__"):
        return "contains"
    return "unrecognized"


def _resolve_eligibility(ctx: LaneContextV7, uni: _Universe) -> tuple:
    """``(eligible_rowids, via)`` — normalizes ctx.eligible, which the
    contract defines as *callable(unit_row) -> bool, or eligible-set
    object*.  Supported set objects: set/frozenset of unit_ids, an object
    exposing ``.unit_ids``, or a bare ``__contains__`` over unit_ids.  A
    callable receives the units row as a dict (falling back to the bare
    unit_id on TypeError); exceptions fail closed — the row is ineligible.
    """
    elig = getattr(ctx, "eligible", None)
    mode = _elig_mode(elig)
    if mode == "all":
        return set(uni.by_rowid), "all"
    if mode == "set":
        wanted = ({str(u) for u in elig.unit_ids}
                  if getattr(elig, "unit_ids", None) is not None
                  else {str(u) for u in elig})
        return ({r for r, u in uni.by_rowid.items()
                 if str(u["unit_id"]) in wanted}, "set")
    if mode == "callable":
        out: set = set()
        errors = 0
        for r, u in uni.by_rowid.items():
            try:
                ok = bool(elig(u))
            except TypeError:
                try:
                    ok = bool(elig(str(u["unit_id"])))
                except Exception:
                    ok = False
                    errors += 1
            except Exception:
                ok = False
                errors += 1
            if ok:
                out.add(r)
        return out, "callable" if not errors else "callable_errors"
    if mode == "contains":
        return ({r for r, u in uni.by_rowid.items() if str(u["unit_id"]) in elig},
                "contains")
    # Unrecognized shape: fail open to the fenced universe would widen
    # authorization — treat as nothing eligible and report honestly.
    return set(), "unrecognized"


def _read_maintained_stats(conn: sqlite3.Connection, scope_id: str,
                           generation: Optional[int]) -> dict:
    """``field -> {"n_units": int, "total_len": int}`` from lex_stats."""
    out: dict[str, dict] = {}
    if not has_table(conn, STATS_TABLE):
        return out
    if generation is None:
        sql = (f"SELECT field, n_units, total_len FROM {STATS_TABLE}"
               " WHERE scope_id = ?")
        params: list = [scope_id]
    else:
        # V7-30.02: ``generation <= pinned`` — a generation bump does not
        # rewrite stats rows, so older snapshots stay valid. Rows are
        # versioned per (field, stats_version); the newest row per field
        # wins (DESC order + first-seen), so a re-scored stats_version
        # supersedes its older rows instead of mixing with them.
        order = "field, generation DESC"
        if "stats_version" in _fts_columns(conn, STATS_TABLE):
            order += ", stats_version DESC"
        sql = (f"SELECT field, n_units, total_len FROM {STATS_TABLE}"
               " WHERE scope_id = ? AND generation <= ?"
               f" ORDER BY {order}")
        params = [scope_id, generation]
    try:
        if generation is None:
            for field, n, tl in conn.execute(sql, params):
                out[str(field)] = {
                    "n_units": int(n), "total_len": int(tl)
                }
        else:
            for field, n, tl in conn.execute(sql, params):
                out.setdefault(
                    str(field), {"n_units": int(n), "total_len": int(tl)}
                )
    except sqlite3.Error:
        return {}
    return out


#: Per-share memo for ``_eligible_corpus_stats`` — ``(N, avglen)`` is a
#: pure function of the pinned snapshot + ctx eligibility, and lexical /
#: fuzzy + their facet calls each recompute it identically.
#: ``(id(share), id(conn), scope_id, generation, cols) -> (share, conn,
#: n_eligible, avglen, stats_writes, cost)``.  A hit is served only when
#: the caller's remaining budget exceeds the measured recompute cost —
#: where a fresh run could honestly return ``None`` on the deadline, the
#: recompute runs for real.
_ECS_MEMO: dict = {}
_ECS_MEMO_MAX = 16
_ECS_MARGIN_S = 0.001


def _eligible_corpus_stats(conn: sqlite3.Connection, uni: _Universe,
                           elig_rowids: set, columns: list,
                           scope_id: str, generation: Optional[int],
                           deadline: _Deadline, stats: dict,
                           share: Any = None) -> tuple:
    """``(N, avglen[field])`` over the request-eligible set (V7-06.05/07).

    Maintained ``lex_stats`` aggregates are used verbatim when the
    eligible set equals the fenced universe (``maintained``); otherwise
    the ineligible rows' docsize contributions are subtracted — bounded
    by the ineligible count (``corrected``).  Without maintained rows the
    lane recomputes exactly over the eligible rowids
    (``exact_recomputed``).  Returns ``None`` when the deadline cut the
    correction scan.
    """
    key = (id(share), id(conn), scope_id, generation, tuple(columns))
    ent = _ECS_MEMO.get(key) if share is not None else None
    t_end = getattr(deadline, "t_end", None)
    if ent is not None and ent[0] is share and ent[1] is conn \
            and t_end is not None \
            and t_end - time.monotonic() > ent[5] + _ECS_MARGIN_S:
        stats.update(ent[4])
        return ent[2], dict(ent[3])
    t0 = time.monotonic()
    before = {k: stats.get(k, None) for k in
              ("stats", "ineligible_subtracted")}
    out = _ecs_compute(conn, uni, elig_rowids, columns, scope_id,
                       generation, deadline, stats)
    if out is not None and share is not None:
        writes = {k: stats[k] for k in before
                  if k in stats and stats[k] != before[k]}
        _ECS_MEMO[key] = (share, conn, out[0], out[1], writes,
                          time.monotonic() - t0)
        while len(_ECS_MEMO) > _ECS_MEMO_MAX:
            _ECS_MEMO.pop(next(iter(_ECS_MEMO)))
    return out


def _ecs_compute(conn: sqlite3.Connection, uni: _Universe,
                 elig_rowids: set, columns: list,
                 scope_id: str, generation: Optional[int],
                 deadline: _Deadline, stats: dict) -> tuple:
    docsize = FTS_TABLE + "_docsize"
    maintained = _read_maintained_stats(conn, scope_id, generation)
    n_eligible = len(elig_rowids)
    n_universe = len(uni.by_rowid)

    def _avg(total: dict) -> dict:
        return {f: (total.get(f, 0.0) / n_eligible if n_eligible else 0.0)
                for f in columns}

    if maintained and n_eligible == n_universe:
        totals = {f: float(maintained.get(f, {}).get("total_len", 0))
                  for f in columns}
        stats["stats"] = "maintained"
        return n_eligible, _avg(totals)

    # Ineligible contributions are subtracted from the maintained
    # aggregates — work bounded by the ineligible count (V7-06.07).
    ineligible = [r for r in uni.by_rowid if r not in elig_rowids]
    lens: dict[int, dict] = {}
    have_lens = has_table(conn, DOCLEN_TABLE) or has_table(conn, docsize)
    if have_lens:
        # V8-07.04: unit_doclen preferred, docsize shadow else (equal
        # values — the same field lengths either way).
        lens = _lens_for(conn, uni, ineligible, columns, scope_id,
                         generation)
    elif ineligible:
        # No maintained lengths at all: count tokens from field bytes.
        texts = _fetch_fields(conn, FTS_TABLE, ineligible, columns)
        for rid, fields in texts.items():
            lens[rid] = {
                f: len(_tokenize(fields.get(f) or "")) for f in columns
            }
    if deadline.expired():
        return None
    if maintained:
        totals = {}
        for f in columns:
            base = float(maintained.get(f, {}).get("total_len", 0))
            sub = sum(l.get(f, 0) for l in lens.values())
            totals[f] = max(base - sub, 0.0)
        stats["stats"] = "corrected"
        stats["ineligible_subtracted"] = len(ineligible)
        return n_eligible, _avg(totals)

    # Exact recompute over the eligible set (V7-06.07 fallback).
    totals = {}
    elig_list = sorted(elig_rowids)
    if have_lens:
        el_lens = _lens_for(conn, uni, elig_list, columns, scope_id,
                            generation)
    else:
        texts = _fetch_fields(conn, FTS_TABLE, elig_list, columns)
        el_lens = {
            rid: {f: len(_tokenize(fields.get(f) or "")) for f in columns}
            for rid, fields in texts.items()
        }
    for f in columns:
        totals[f] = float(sum(l.get(f, 0) for l in el_lens.values()))
    stats["stats"] = "exact_recomputed"
    if deadline.expired():
        return None
    return n_eligible, _avg(totals)


def _maintained_df(conn: sqlite3.Connection, scope_id: str,
                   generation: Optional[int], terms: list) -> dict:
    """``term -> max df`` across fields from maintained lex_df (coverage
    transparency + OOV detection; scoring df is always exact)."""
    out: dict[str, int] = {}
    if not terms or not has_table(conn, DF_TABLE):
        return out
    if generation is None:
        sql = (f"SELECT term, MAX(df) FROM {DF_TABLE}"
               " WHERE scope_id = ? AND term IN ({}) GROUP BY term")
        extra: list = []
    else:
        # V7-30.02: ``generation <= pinned`` — a generation bump does not
        # rewrite df rows. Each row is a cumulative snapshot per
        # (field, term, stats_version): resolve every key's newest row
        # at/below the pin (DESC + first-seen), then MAX(df) across
        # fields/versions per term exactly as the grouped read did.
        has_sver = "stats_version" in _fts_columns(conn, DF_TABLE)
        if has_sver:
            sel = "term, field, stats_version, generation, df"
            order = "term, field, stats_version, generation DESC"
        else:
            # Simplified mirrors may lack stats_version — the natural
            # key degrades to (field, term).
            sel = "term, field, generation, df"
            order = "term, field, generation DESC"
        sql = (f"SELECT {sel} FROM {DF_TABLE}"
               " WHERE scope_id = ? AND generation <= ?"
               " AND term IN ({})"
               f" ORDER BY {order}")
        extra = [generation]
    for chunk in _chunks(list(terms), _IN_CHUNK):
        try:
            cur = conn.execute(
                sql.format(",".join("?" * len(chunk))),
                [scope_id, *extra, *chunk],
            )
        except sqlite3.Error:
            return out
        if generation is None:
            for term, df in cur.fetchall():
                out[str(term)] = int(df)
        else:
            seen: set = set()
            for row in cur.fetchall():
                if has_sver:
                    term, field, sver, _gen, df = row
                else:
                    term, field, _gen, df = row
                    sver = ""
                key = (str(term), str(field), str(sver))
                if key in seen:
                    continue  # older snapshot of the same df key
                seen.add(key)
                t, d = str(term), int(df)
                if t not in out or d > out[t]:
                    out[t] = d
    return out


# ---------------------------------------------------------------------------
# bm25f/v1 — the hand-computable core (§32.2)
# ---------------------------------------------------------------------------


def idf(n_docs: float, df: float) -> float:
    """``ln(1 + (N − df + 0.5) / (df + 0.5))`` — the bm25f/v1 idf."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25f_score(tf_by_field: dict, len_by_field: dict, avglen: dict, *,
                idf_value: float, k1: float = K1,
                weights: Optional[dict] = None,
                b: Optional[dict] = None) -> float:
    """One term's BM25F contribution to one unit (§32.2, provisional/v7-r0).

    ``tf_by_field[f]`` is the channel-combined effective tf (exact counts
    plus stem-channel extras already weighted by the caller); the function
    applies field length normalization, field weights, and the k1
    saturation exactly as written in the spec.
    """
    weights = FIELD_W if weights is None else weights
    b = FIELD_B if b is None else b
    tf_tilde = 0.0
    for f, w_f in weights.items():
        tf = float(tf_by_field.get(f, 0.0))
        if tf <= 0.0:
            continue
        b_f = b.get(f, 0.0)
        avg = float(avglen.get(f, 0.0))
        norm = 1.0 - b_f + (b_f * float(len_by_field.get(f, 0)) / avg
                          if avg > 0.0 else 0.0)
        tf_tilde += w_f * tf / norm
    return idf_value * tf_tilde * (k1 + 1.0) / (tf_tilde + k1)


# ---------------------------------------------------------------------------
# Query-term model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QTerm:
    """One scored query term. ``via``: ``exact`` (text-channel term),
    ``identifier`` (exact-byte identifier, nominal tf=1 text hit), or
    ``respell`` (fuzzy lane's reissued alternative at ``scale`` weight)."""

    term: str
    via: str = "exact"
    stem: Optional[str] = None   # stem used for the stem channel
    scale: float = 1.0           # respelled terms carry a declared weight
    source: Optional[str] = None  # originating OOV term for respells
    content: bool = True         # False for stop/meta words — see below


#: Query-side content filter for *nomination* (V7-11.01/D7-04 honest
#: refusal).  Stopwords and meta-terms stay indexed and still contribute
#: idf-derived score to a nominated document — §32.1 rule 7 — but a
#: document that shares ONLY function words with a content-bearing query
#: carries no real support, so it must never nominate a candidate on its
#: own.  Mirroring ``querying.analyze``'s stop/meta set keeps the lane's
#: notion of "content" identical to the analyzer's.
def _content_filter() -> frozenset:
    try:
        from ...querying.analyze import _META_TERMS, _STOPWORDS
        return _STOPWORDS | _META_TERMS
    except Exception:
        return _NOM_STOP


#: Offline copy of the canonical stop/meta set — used only if the import
#: above ever fails; kept in lockstep with ``querying.analyze``.
_NOM_STOP = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have he her hers him his how i if in into is it its me my of on or our
    ours she so such than that the their theirs them then there these they
    this those to too was we were what when where which who whom why will
    with would you your yours about after again against all also am any
    because before being between both each few further here once only other
    out over own same should some under until up very
    tell told remember recall remind show find search look list mention
    mentioned say said ask asked talk talked speak spoke know knew think
    thought wondering want wanted need needed give gave get got please help
    mean meant happen happened anything something everything everyone
    someone anybody somebody thing things stuff way ways kind kinda sort
    really actually just even still yet maybe perhaps probably definitely
    basically literally honestly hey hi hello ok okay yeah yes nope thanks
    thank dude btw fyi
    """.split()
)


def _query_terms(qv: QueryViewV7, stats: Optional[dict] = None) -> tuple:
    """Split the NormAnalysis terms into (text terms, identifiers).

    Returns ``(qterms, ident_terms)``; stem twins from the analyzer (same
    byte span, channel ``stem``) are preferred over the local Porter for
    the stem channel so the lane stays consistent with ``norm/v2``.

    Every parsed term is kept: all of them collect postings and score
    (V75-04.02).  The nomination budget is applied at *nomination* time
    (``_select_nomination``) — the positional ``_MAX_TERMS`` truncation
    that used to live here could drop the rarest term entirely.

    ``stats`` is accepted-and-ignored: sibling lanes sharing this helper
    (``retrieval/v7/scope.py``) still pass it positionally from the
    truncation era; nothing is recorded here any longer.
    """
    del stats
    norm = getattr(qv, "norm", None)
    terms = tuple(getattr(norm, "terms", ()) or ())
    idents = tuple(getattr(norm, "identifiers", ()) or ())
    ident_surfaces = {str(t.term) for t in idents}

    stem_twins = {}
    for t in terms:
        if getattr(t, "channel", None) == "stem":
            stem_twins[(t.byte_start, t.byte_end)] = t.term

    noncontent = _content_filter()
    qterms: list[_QTerm] = []
    ident_terms: list[_QTerm] = []
    seen: set = set()
    for t in (*terms, *idents):
        ch = getattr(t, "channel", None) or "text"
        term = str(getattr(t, "term", "") or "")
        if ch == "stem" or not term or term in seen:
            continue
        if ch == "identifier" or term in ident_surfaces:
            ident_terms.append(_QTerm(term=term, via="identifier"))
        else:
            qterms.append(_QTerm(
                term=term, via="exact",
                stem=stem_twins.get((t.byte_start, t.byte_end))
                or _porter_memo(term),   # V8-07.05: once per request
                content=term not in noncontent,
            ))
        seen.add(term)
    return qterms, ident_terms


def _select_nomination(qterms: list, ident_terms: list, posts: dict,
                       df_unknown: set, stats: dict, *,
                       budget: int, df_theta: Optional[float],
                       n_eligible: int,
                       df_gated: Optional[dict] = None,
                       nom_cooc: bool = _NOM_COOC_DEFAULT,
                       cooc_min: int = _COOC_MIN_DEFAULT,
                       rare_df: int = _RARE_DF_DEFAULT) -> set:
    """The V75-04.02 nomination set — which terms may *nominate* docs.

    Selection order: identifiers first (query order), then
    nomination-eligible query terms by *ascending eligible df* — the df
    the collected postings already measured (``len(posts[t])``), so the
    ordering adds no corpus scan.  Content terms only when the query
    carries any; on a facetless query every term is eligible to nominate
    (the V7-11.01 fallback, unchanged).  Terms whose df could not be
    measured (a MATCH channel errored) sort *after* every known-df term;
    ties break on term text, so selection is deterministic.

    Two exclusions, both reported in ``stats["nomination_dropped"]`` as
    ``{"term", "df", "reason"}``:

    - ``df_threshold``: when ``df_theta`` is armed, a content term with
      ``df > θ·N_E`` may not nominate (identifiers are exempt — they
      nominate by identity, not topicality).  Excluded terms consume no
      budget slot.
    - ``budget``: terms past ``budget`` slots in the selection order.
      Dropped terms still collect postings and still *score* nominated
      docs — only their nominating power is withheld.
    """
    has_content = any(q.content for q in qterms)
    pool = [q for q in qterms if q.content or not has_content]

    def _df_key(qt: _QTerm) -> tuple:
        if qt.term in df_unknown or qt.term not in posts:
            return (1, 0, str(qt.term))
        return (0, len(posts[qt.term]), str(qt.term))

    ordered = [*ident_terms, *sorted(pool, key=_df_key)]
    theta_cut = float(df_theta) * n_eligible if df_theta is not None else None

    kept: list = []
    dropped: list = []
    gated = df_gated or {}
    for qt in ordered:
        if qt.term in gated:
            # Pre-fetch df gate: postings were never collected — report
            # the maintained df honestly and consume no budget slot.
            dropped.append({"term": qt.term, "df": gated[qt.term],
                            "reason": "df_threshold"})
            continue
        known = qt.term in posts and qt.term not in df_unknown
        df = len(posts[qt.term]) if known else None
        if (theta_cut is not None and qt.via != "identifier"
                and df is not None and df > theta_cut):
            dropped.append({"term": qt.term, "df": df,
                            "reason": "df_threshold"})
            continue
        if len(kept) >= budget:
            dropped.append({"term": qt.term, "df": df,
                            "reason": "budget"})
            continue
        kept.append(qt)

    # V85-05.02 — the union of kept terms' postings passes through the
    # co-occurrence filter (≥ ``cooc_min`` kept terms per unit, with the
    # identifier + rare-df exemptions of V85-05.03); an empty filtered
    # set falls back to the union, reported in ``coverage_lexical.cooc``.
    nominated = _cooc_filter(kept, posts, df_unknown, stats,
                             nom_cooc=nom_cooc, cooc_min=cooc_min,
                             rare_df=rare_df)
    stats["nominate_terms_max"] = int(budget)
    if df_theta is not None:
        stats["nominate_df_theta"] = float(df_theta)
    stats["nomination_eligible_terms"] = len(ordered)
    stats["nominated_terms"] = len(kept)
    stats["nomination_dropped"] = dropped
    return nominated


# --- V85-05.02/05.03 co-occurrence nomination -------------------------------


def _cooc_filter(kept: list, posts: dict, df_unknown: set,
                 stats: dict, *, nom_cooc: bool, cooc_min: int,
                 rare_df: int) -> set:
    """The V8.5 co-occurrence nomination filter (``lexical.nom_cooc``).

    A unit is nominated when it matches ≥ ``cooc_min`` *kept* terms —
    the AND-of-postings the b5 arm measured (union 674→110 mean,
    +0.064 any@10).  Two exemption classes nominate without meeting
    the threshold:

    - **identifiers** — they nominate by identity, not topicality (the
      same discipline as the ``df_theta`` exclusion; a df-1
      identifier's one unit could never satisfy a ≥2-term
      requirement);
    - **rare terms** — a kept term whose measured eligible df
      (``len(posts[term])`` — the postings this call already collected,
      exact ∪ stem channels) sits below ``lexical.rare_df`` nominates
      its whole posting set (V85-05.03; the pack's ``df_floor`` arm
      renamed — ``lexical.df_floor`` is the *pre-fetch* gate).  A term
      whose df could not be fully measured (``df_unknown`` — a channel
      MATCH errored) is never treated as rare.

    An empty filtered set falls back to the plain union — reported
    honestly as ``coverage_lexical.cooc.fallback = "union"``.  The
    report lands on ``stats["coverage_lexical"]["cooc"]``:
    ``{enabled, min, rare_df, union, filtered, exempt_terms,
    fallback?}``.
    """
    cov = stats.setdefault("coverage_lexical", {})
    union: set = set()
    for qt in kept:
        union |= posts.get(qt.term, set())
    info: dict[str, Any] = {
        "enabled": bool(nom_cooc),
        "min": int(cooc_min),
        "rare_df": int(rare_df),
        "union": len(union),
    }
    cov["cooc"] = info
    if not nom_cooc or cooc_min <= 1 or not union:
        # ``cooc_min`` <= 1 makes the filter the identity on the union;
        # an empty union has nothing to filter.
        info["filtered"] = len(union)
        return union

    counts: dict[int, int] = {}
    exempt_posts: set = set()
    exempt_terms: dict[str, dict] = {}
    for qt in kept:
        p = posts.get(qt.term) or set()
        for rid in p:
            counts[rid] = counts.get(rid, 0) + 1
        if qt.via == "identifier":
            exempt_terms[qt.term] = {"term": qt.term,
                                     "reason": "identifier"}
            exempt_posts |= p
        elif qt.term not in df_unknown and len(p) < rare_df:
            exempt_terms[qt.term] = {"term": qt.term,
                                     "reason": "rare",
                                     "df": len(p)}
            exempt_posts |= p
    filtered = {rid for rid, c in counts.items() if c >= cooc_min}
    filtered |= exempt_posts
    info["filtered"] = len(filtered)
    if exempt_terms:
        info["exempt_terms"] = [
            exempt_terms[t] for t in sorted(exempt_terms)]
    if not filtered:
        # The co-occurrence requirement starved every unit — nominate
        # the union instead, and say so (the pack measured ~4/989).
        info["fallback"] = "union"
        return union
    return filtered


# --- V8-07.02 df-gate exemptions --------------------------------------------


def _canon_gate_reasons(qv: QueryViewV7) -> dict:
    """``term -> reason`` for resolved canons (V8-07.02b): every folded
    token of ``qv.speaker_canon`` / ``qv.entity_canons`` is a protected
    nominating surface — a canon flood is identity, not topicality."""
    reasons: dict[str, str] = {}
    spk = getattr(qv, "speaker_canon", None)
    if spk:
        for tok in _tokenize(str(spk)):
            reasons.setdefault(tok, "speaker_canon")
    for canon in getattr(qv, "entity_canons", ()) or ():
        for tok in _tokenize(str(canon)):
            reasons.setdefault(tok, "entity_canon")
    return reasons


def _df_gate_exemptions(qterms: list, ident_terms: list, over_cut: dict,
                        df_maintained: dict, fetch_cut: Optional[float],
                        canon_reasons: dict) -> dict:
    """``term -> exemption reason`` (V8-07.02, K38).

    The gate MUST NOT gate (a) identifiers — they nominate by identity,
    listed here only when their maintained df actually exceeds the cut;
    (b) resolved entity/speaker canons; (c) the query's rarest content
    term; (d) any term whose gating would leave zero nominating terms.
    ``over_cut`` holds only terms whose maintained df exceeds
    ``fetch_cut`` — an exemption is meaningful (and reported) only there.
    """
    exempt: dict[str, str] = {}
    for qt in qterms:
        if qt.term in over_cut:
            reason = canon_reasons.get(qt.term)
            if reason is not None:
                exempt[qt.term] = reason
    content = [q for q in qterms if q.content]
    if content:
        # The rarest content term is the query's strongest nominating
        # signal — never gated.  Best-known df orders it: maintained
        # count, absent → 0 (OOV is as rare as it gets); term text breaks
        # ties deterministically.
        rarest = min(content,
                     key=lambda q: (df_maintained.get(q.term, 0), q.term))
        if rarest.term in over_cut and rarest.term not in exempt:
            exempt[rarest.term] = "rarest_content_term"
    for qt in ident_terms:
        df = df_maintained.get(qt.term)
        if fetch_cut is not None and df is not None and df > fetch_cut:
            exempt.setdefault(qt.term, "identifier")
    # (d): a query made only of gateable terms keeps >= 1 nominating
    # term.  The pool mirrors _select_nomination — content terms when any
    # exist else every term; identifiers always nominate (they are never
    # gated), so the safety fires only without them.
    pool = content if content else list(qterms)
    nominating = [q for q in pool
                  if q.term not in over_cut or q.term in exempt]
    if not nominating and not ident_terms:
        cand = [q for q in pool
                if q.term in over_cut and q.term not in exempt]
        if cand:
            pick = min(cand, key=lambda q: (over_cut[q.term], q.term))
            exempt[pick.term] = "last_nominating_term"
    return exempt


# --- V8-07.06 single-statement nomination (arm) -----------------------------


def _vocab_for(conn: sqlite3.Connection, fts_table: str) -> Optional[str]:
    """A usable fts5vocab table over ``fts_table`` — ``instance`` mode
    only.  ``doc`` is a real rowid only in instance mode
    (``term, doc, col, offset``); in ``row``/``col`` modes a column
    *named* ``doc`` carries a document count, so accepting either would
    fabricate posting rowids — those modes are rejected and the caller
    degrades to the union-MATCH + field-byte attribution path instead
    (V8-07.06 requires byte-identical term→rowid sets)."""
    name = fts_table + "_vocab"
    if not has_table(conn, name):
        return None
    cols = _fts_columns(conn, name)
    if "term" in cols and "doc" in cols and "offset" in cols:
        return name
    return None


def _or_match(terms: list) -> str:
    """One FTS5 query matching any quoted term — the union of the
    per-term MATCHes (same tokenizer applied to each literal)."""
    return " OR ".join(_fts_quote(t) for t in terms)


def _posts_single(conn: sqlite3.Connection, fetch_qts: list,
                  uni: _Universe, elig_rowids: set, stem_ok: bool,
                  deadline: _Deadline, stats: dict,
                  probe) -> Any:
    """V8-07.06 collapsed nomination: ``(posts, df_unknown, n_rows)``.

    ``posts`` is ``term -> eligible∩universe rowids`` — byte-identical to
    the per-term MATCH sets.  Per-term attribution is recovered from the
    maintained postings (an fts5vocab table in ``instance`` mode — the
    only mode whose ``doc`` is a rowid: one ``term IN (...)`` read per
    channel) or, when no usable vocab exists,
    from ≤1 union MATCH per channel plus field-byte attribution over the
    matched rows.  ``"fallback"`` return means neither path could
    honestly attribute — the caller runs the per-term loop, so the arm
    degrades to identical sets rather than fabricate them.  ``None``
    means the deadline cut the pass.
    """
    posts: dict[str, set] = {qt.term: set() for qt in fetch_qts}
    df_unknown: set = set()
    n_rows = 0
    if not fetch_qts:
        return posts, df_unknown, n_rows
    fence = uni.by_rowid.keys() & elig_rowids
    terms = [qt.term for qt in fetch_qts]

    exact_voc = _vocab_for(conn, FTS_TABLE)
    stem_voc = _vocab_for(conn, STEM_TABLE) if stem_ok else None
    if exact_voc is not None and (not stem_ok or stem_voc is not None):
        # Maintained postings: one batched instance read per channel.
        ex_by: dict[str, set] = {t: set() for t in terms}
        ok_exact = True
        for chunk in _chunks(terms, _IN_CHUNK):
            if deadline.expired():
                return None
            ph = ",".join("?" * len(chunk))
            try:
                cur = conn.execute(
                    f"SELECT term, doc FROM {exact_voc}"
                    f" WHERE term IN ({ph})", chunk)
            except sqlite3.Error:
                ok_exact = False
                break
            for term, doc in cur.fetchall():
                bucket = ex_by.get(str(term))
                if bucket is not None:
                    bucket.add(int(doc))
        st_by: dict[str, set] = {t: set() for t in terms}
        ok_stem = True
        if stem_ok:
            # stem index holds stems: several query terms may share one
            stem_terms: dict[str, list] = {}
            for qt in fetch_qts:
                if qt.stem:
                    stem_terms.setdefault(qt.stem, []).append(qt.term)
            for chunk in _chunks(list(stem_terms), _IN_CHUNK):
                if deadline.expired():
                    return None
                ph = ",".join("?" * len(chunk))
                try:
                    cur = conn.execute(
                        f"SELECT term, doc FROM {stem_voc}"
                        f" WHERE term IN ({ph})", chunk)
                except sqlite3.Error:
                    ok_stem = False
                    break
                for stem, doc in cur.fetchall():
                    for qt in stem_terms.get(str(stem), ()):
                        st_by[qt].add(int(doc))
        if not ok_exact or not ok_stem:
            # An unreadable postings channel degrades like a failed
            # MATCH: every term's df is unmeasurable, never fabricated.
            df_unknown.update(terms)
            stats["match_errors"] = stats.get("match_errors", 0) + 1
        n_rows += sum(len(ex_by[t]) + len(st_by[t]) for t in terms)
        for t in terms:
            posts[t] = (ex_by[t] | st_by[t]) & fence
        stats["single_match"] = "vocab"
        return posts, df_unknown, n_rows

    # Field-byte attribution over one union MATCH per channel.
    or_terms = _or_match(terms)
    union_exact = probe(conn, FTS_TABLE, or_terms)
    union_stem = probe(conn, STEM_TABLE, or_terms) if stem_ok else None
    if union_exact is None:
        df_unknown.update(terms)
        stats["match_errors"] = stats.get("match_errors", 0) + 1
    if stem_ok and union_stem is None:
        df_unknown.update(terms)
        stats["match_errors"] = stats.get("match_errors", 0) + 1
    n_rows += len(union_exact or set()) + len(union_stem or set())
    u_exact = (union_exact or set()) & fence
    u_stem = (union_stem or set()) & fence
    u_all = sorted(u_exact | u_stem)
    all_cols = _fts_columns(conn, FTS_TABLE)
    term_set = set(terms)
    stem_of = {qt.term: qt.stem for qt in fetch_qts if qt.stem}
    stem_wanted = set(stem_of.values())
    unreadable = 0
    attributed_any = False
    for page in _chunks(u_all, _PAGE):
        if deadline.expired():
            return None
        fields = _fetch_fields(conn, FTS_TABLE, page, all_cols)
        if page and not fields:
            # Content fetch failed entirely — attribution is impossible.
            return "fallback"
        for rid in page:
            rowf = fields.get(rid)
            if rowf is None:
                continue
            if all(rowf.get(c) is None for c in all_cols):
                unreadable += 1
                continue
            attributed_any = True
            if rid in u_exact:
                toks: set = set()
                for c in all_cols:
                    txt = rowf.get(c)
                    if txt:
                        toks.update(_tokenize(txt))
                for t in toks & term_set:
                    posts[t].add(rid)
            if rid in u_stem and stem_wanted:
                hit_stems: set = set()
                for tok in _tokenize(rowf.get("text") or ""):
                    st = _porter_memo(tok)
                    if st in stem_wanted:
                        hit_stems.add(st)
                for t, s in stem_of.items():
                    if s in hit_stems:
                        posts[t].add(rid)
    if u_all and unreadable == len(u_all) and not attributed_any:
        # Contentless index — per-term MATCH is the only honest measure.
        return "fallback"
    stats["single_match"] = "fields"
    return posts, df_unknown, n_rows


# --- V8-07.03 bounded exhaustive rescue (§21.4) ------------------------------


def _rescue_candidates(conn: sqlite3.Connection, qterms: list, posts: dict,
                       nominated: set, uni: _Universe, elig_rowids: set,
                       stem_ok: bool, deadline: _Deadline, stats: dict, *,
                       k_rescue: int, rescue_rows: int) -> set:
    """Bounded rescue for starved nomination (V8-07.03, §21.4).

    Fires when ``|nominated ∩ E| < K_rescue`` or every content term was
    gated; scans the eligible postings of the ≤2 rarest gated or dropped
    content terms — df-ascending terms, rowid-ascending rows — capped at
    ``rescue_rows`` posting rows and deadline-checked every
    ``_RESCUE_CHECK`` rows.  Rescued rows are scored by the same BM25F
    pass as nominated candidates.  Returns the newly-produced rowids
    (for ``signals["rescue"]``); ``coverage_lexical.rescue`` reports
    ``{fired, terms, rows, produced}`` exactly as measured.
    """
    cov = stats.setdefault("coverage_lexical", {})
    info: dict[str, Any] = {"fired": False, "terms": [], "rows": 0,
                            "produced": 0}
    cov["rescue"] = info
    gated = {e["term"]: e["df"]
             for e in cov.get("df_gate", {}).get("gated", ())}
    dropped = stats.get("nomination_dropped") or ()
    content = [q for q in qterms if q.content]
    all_gated = bool(content) and all(q.term in gated for q in content)
    if len(nominated) >= k_rescue and not all_gated:
        return set()
    info["fired"] = True
    # ≤2 rarest gated-or-dropped terms by best-known df: exact eligible
    # df where a dropped term measured it, maintained df for gated terms,
    # unknown → +inf (sorted last); term text is the deterministic tie.
    df_of: dict[str, Optional[float]] = {}
    for e in dropped:
        df_of.setdefault(e["term"], e.get("df"))
    df_of.update(gated)
    pool = content if content else list(qterms)
    cand = sorted(
        ((float(df_of[q.term]) if df_of.get(q.term) is not None
          else math.inf, q.term)
         for q in pool if q.term in df_of),
    )
    terms = [t for _, t in cand[:2]]
    info["terms"] = terms
    if not terms or deadline.expired():
        return set()
    rescued: set = set()
    visited = 0
    stop = False
    for t in terms:
        if stop:
            break
        if posts.get(t):
            # Dropped terms already hold eligible-fenced postings — reuse
            # them (sorted for the rowid-ascending order), no re-MATCH.
            sources: list = [iter(sorted(int(r) for r in posts[t]))]
        else:
            sources = []
            for table in ((FTS_TABLE, STEM_TABLE) if stem_ok
                          else (FTS_TABLE,)):
                remaining = rescue_rows - visited
                if remaining <= 0:
                    break
                try:
                    cur = conn.execute(
                        f"SELECT rowid FROM {table}"
                        f" WHERE {table} MATCH ? ORDER BY rowid LIMIT ?",
                        (_fts_quote(t), remaining))
                    stats["match_calls"] = stats.get("match_calls", 0) + 1
                    sources.append(cur)
                except sqlite3.Error:
                    stats["match_errors"] = stats.get("match_errors", 0) + 1
        for it in sources:
            for row in it:
                if visited >= rescue_rows:
                    stop = True
                    break
                if visited and visited % _RESCUE_CHECK == 0 \
                        and deadline.expired():
                    stop = True
                    break
                visited += 1
                rid = int(row[0]) if isinstance(row, tuple) else int(row)
                if rid in elig_rowids and rid in uni.by_rowid:
                    rescued.add(rid)
            if stop:
                break
    info["rows"] = visited
    produced = rescued - set(nominated)
    info["produced"] = len(produced)
    stats["posting_rows"] = stats.get("posting_rows", 0) + visited
    return produced


def _collect_postings(conn: sqlite3.Connection, qterms: list,
                      ident_terms: list, uni: _Universe,
                      elig_rowids: set, stem_ok: bool,
                      deadline: _Deadline, stats: dict, *,
                      nominate_budget: int = NOMINATE_TERMS_MAX_R0,
                      df_theta: Optional[float] = None,
                      n_eligible: int = 0,
                      df_maintained: Optional[dict] = None,
                      df_floor: Optional[int] = _DF_PREFETCH_FLOOR,
                      single_match: bool = False,
                      canon_reasons: Optional[dict] = None,
                      nom_cooc: bool = _NOM_COOC_DEFAULT,
                      cooc_min: int = _COOC_MIN_DEFAULT,
                      rare_df: int = _RARE_DF_DEFAULT
                      ) -> Optional[tuple]:
    """``(posts, nominated)`` — per-term eligible-fenced rowid sets plus
    the nominated doc set under the V75-04.02 selection.

    Exact channel: ``unit_fts MATCH "<term>"`` (all fields).  Stem channel:
    ``unit_fts_stem MATCH "<term>"`` (porter tokenizer stems the query
    term itself).  Identifier channel: quoted MATCH on ``unit_fts`` —
    FTS5 re-tokenizes the identifier into a token phrase, an exact-order
    proxy; the ``exact_id`` lane remains the precise surface.

    Nomination discipline (V7-11.01 honest refusal + V75-04.02 budget):
    postings are collected for *every* term — stopwords/meta-terms carry
    idf-derived score on a nominated doc and feed the df coverage report,
    and budget-dropped terms still score — but a document enters the pool
    only via a term ``_select_nomination`` keeps.  ``df_unknown`` marks
    terms whose df could not be fully measured (a channel's MATCH
    errored); they sort last in the selection order.  ``None`` return
    means the deadline cut the scan.

    V8: ``df_floor`` is the §23 arm — ``None`` disables the pre-fetch
    gate entirely (``lexical.df_floor = off``); ``single_match`` runs the
    collapsed ≤1-statement-per-channel path (V8-07.06) with
    byte-identical term→rowid sets; ``canon_reasons`` maps protected
    canon surfaces to their exemption reason (V8-07.02).
    ``stats["coverage_lexical"]["df_gate"]`` reports
    ``{floor, gated, exempt}`` (V8-20.03).
    """
    posts: dict[str, set] = {}
    df_unknown: set = set()
    universe = uni.by_rowid
    n_universe_rows = 0
    df_maintained = df_maintained or {}
    # Pre-fetch df gate (beat-it r8): a term flooding the eligible corpus
    # costs a multi-thousand-rowid MATCH for near-zero discrimination.
    # Maintained ``lex_df`` decides *before* the fetch; V8-07.02 adds the
    # declared exemptions and the ``lexical.df_floor`` arm — ``None``
    # switches the gate off entirely (the nominate_df_theta exclusion is
    # a separate, later stage and still applies).
    theta_eff = float(df_theta) if df_theta is not None else _DF_PREFETCH_THETA
    if df_floor is None:
        fetch_cut: Optional[float] = None
    else:
        fetch_cut = (
            max(float(df_floor), theta_eff * n_eligible)
            if n_eligible > 0 else float(df_floor)
        )
    over_cut = {
        qt.term: int(df_maintained[qt.term])
        for qt in qterms
        if fetch_cut is not None
        and df_maintained.get(qt.term) is not None
        and df_maintained[qt.term] > fetch_cut
    }
    exempt = _df_gate_exemptions(
        qterms, ident_terms, over_cut, df_maintained, fetch_cut,
        canon_reasons or {})
    df_gated = {t: d for t, d in over_cut.items() if t not in exempt}
    stats.setdefault("coverage_lexical", {})["df_gate"] = {
        "floor": int(df_floor) if df_floor is not None else "off",
        "gated": [{"term": t, "df": df_gated[t]} for t in sorted(df_gated)],
        "exempt": [{"term": t, "reason": exempt[t]} for t in sorted(exempt)],
    }
    for t in df_gated:
        posts[t] = set()
    fetch_qts = [qt for qt in qterms if qt.term not in df_gated]

    def _probe(*args):
        stats["match_calls"] = stats.get("match_calls", 0) + 1
        return _match_rowids(*args)

    if single_match:
        res = _posts_single(conn, fetch_qts, uni, elig_rowids, stem_ok,
                            deadline, stats, _probe)
        if res is None:
            return None
        if res != "fallback":
            s_posts, s_unknown, s_rows = res
            posts.update(s_posts)
            df_unknown |= s_unknown
            n_universe_rows += s_rows
        else:
            stats["single_match"] = "fallback"
            # The union pass may have marked terms unmeasurable — the
            # per-term probes below re-measure them for real.
            df_unknown.difference_update(qt.term for qt in fetch_qts)
            single_match = False
    if not single_match:
        stats.setdefault("single_match", "off")
        for qt in fetch_qts:
            if deadline.expired():
                return None
            exact = _probe(conn, FTS_TABLE, _fts_quote(qt.term))
            stem = (_probe(conn, STEM_TABLE, _fts_quote(qt.term))
                    if stem_ok else None)
            ex = (exact or set()) & universe.keys() & elig_rowids
            st = (stem or set()) & universe.keys() & elig_rowids
            n_universe_rows += len(exact or set()) + len(stem or set())
            posts[qt.term] = ex | st
            if exact is None or (stem_ok and stem is None):
                df_unknown.add(qt.term)
            if exact is None:
                stats["match_errors"] = stats.get("match_errors", 0) + 1
    for qt in ident_terms:
        if deadline.expired():
            return None
        exact = _probe(conn, FTS_TABLE, _fts_quote(qt.term))
        n_universe_rows += len(exact or set())
        posts[qt.term] = (exact or set()) & universe.keys() & elig_rowids
        if exact is None:
            df_unknown.add(qt.term)
            stats["match_errors"] = stats.get("match_errors", 0) + 1
    stats["posting_rows"] = n_universe_rows
    if df_gated:
        stats["df_prefetch_gated"] = [
            {"term": t, "df": df_gated[t]} for t in sorted(df_gated)
        ]
    nominated = _select_nomination(
        qterms, ident_terms, posts, df_unknown, stats,
        budget=nominate_budget, df_theta=df_theta, n_eligible=n_eligible,
        df_gated=df_gated, nom_cooc=nom_cooc, cooc_min=cooc_min,
        rare_df=rare_df)
    return posts, nominated


#: Per-share porter-stem memo for ``_score_candidates`` — stems are pure
#: functions of the token string, so one cache serves every facet call
#: under a shared ``ctx``.  ``id(share) -> (share, cache)``; holding the
#: share object keeps its id unrecyclable while the entry lives.
_SC_MEMO: dict = {}
_SC_MEMO_MAX = 8


def _score_candidates(conn: sqlite3.Connection, qterms: list,
                      ident_terms: list, posts: dict, nominated_set: set,
                      uni: _Universe,
                      columns: list, avglen: dict, n_eligible: int,
                      deadline: _Deadline, stats: dict,
                      stem_cache: Optional[dict] = None,
                      share: Any = None,
                      df_override: Optional[dict] = None,
                      scope_id: Optional[str] = None,
                      generation: Optional[int] = None) -> Optional[dict]:
    """Score every eligible nominated unit.  ``rowid -> {"score": float,
    "matched": {term: detail}}``.  ``nominated_set`` is the content/identifier
    nomination from ``_collect_postings`` — a doc enters scoring only via a
    real query term, never a lone stopword.  Field bytes + maintained
    lengths (``unit_doclen`` when present — V8-07.04 — else docsize) are
    fetched in pages; a deadline stop returns ``None`` with the caller
    reporting ``partial`` over honest counts."""
    nominated = sorted(nominated_set)
    stats["nominated"] = len(nominated)
    idf_of: dict[str, float] = {}
    df_of: dict[str, int] = {}
    for qt in (*qterms, *ident_terms):
        # A pre-fetch-gated term has no postings by construction — report
        # its maintained df, not a misleading zero.
        df = len(posts.get(qt.term, ()))
        df = int((df_override or {}).get(qt.term, df))
        df_of[qt.term] = df
        idf_of[qt.term] = idf(n_eligible, df)
    stats["df"] = df_of
    if not nominated or not columns:
        return {}

    # V8-06.03 — an armed ``context.ctx_field_weight`` contributes only
    # when the caller's column set carries the ``ctx`` field. ``share``
    # is the lane-context carrier (``None`` for detached callers → the
    # arm stays off); the weight joins ``FIELD_W`` as a real BM25F
    # field weight, never a post-hoc score multiplier.
    field_w = FIELD_W
    if share is not None and CTX_FIELD_NAME in columns:
        ctx_w = _ctx_field_weight_of(share)
        if ctx_w is not None:
            field_w = dict(FIELD_W)
            field_w[CTX_FIELD_NAME] = ctx_w
            stats["ctx_field_weight_applied"] = ctx_w

    docsize = FTS_TABLE + "_docsize"
    if scope_id is None:
        scope_id = getattr(share, "scope_id", None)
    if generation is None:
        generation = getattr(share, "generation", None)
    have_lens = has_table(conn, DOCLEN_TABLE) or has_table(conn, docsize)
    null_rows = 0
    stems_wanted = {q.stem for q in qterms if q.stem}
    term_surfaces = {q.term for q in qterms}
    cache: dict = stem_cache if stem_cache is not None else {}
    if share is not None:
        sc_ent = _SC_MEMO.get(id(share))
        if sc_ent is not None and sc_ent[0] is share:
            cache = sc_ent[1]
        else:
            _SC_MEMO[id(share)] = (share, cache)
            while len(_SC_MEMO) > _SC_MEMO_MAX:
                _SC_MEMO.pop(next(iter(_SC_MEMO)))

    def _stem_of(tok: str) -> str:
        st = cache.get(tok)
        if st is None:
            st = _porter_memo(tok)   # V8-07.05 bounded process LRU
            cache[tok] = st
        return st

    scores: dict[int, dict] = {}
    examined = 0
    for page in _chunks(nominated, _PAGE):
        if deadline.expired():
            stats["scored_docs"] = examined
            return None
        fields = _fetch_fields(conn, FTS_TABLE, page, columns)
        lens = (_lens_for(conn, uni, page, columns, scope_id, generation)
                if have_lens else {})
        for rid in page:
            examined += 1
            row_fields = fields.get(rid)
            if row_fields is None:
                continue
            if all(row_fields.get(f) is None for f in columns):
                null_rows += 1
                continue
            # Targeted counting: ``counters[f]`` is only ever read back via
            # ``.get(qt.term, 0)`` for query terms, so count just those
            # surfaces plus the field token total in one pass instead of
            # materializing a full ``Counter`` of every distinct token.
            counters: dict[str, dict] = {}
            field_total: dict[str, int] = {}
            # stem counts are needed only for the query's stem set —
            # stem each text token once, memoized across docs
            stem_counter: Counter = Counter()
            want_stem = bool(stems_wanted)
            for f in columns:
                ftext = row_fields.get(f) or ""
                cnt: dict = {}
                total = 0
                if ftext:
                    if f == "text":
                        for tok in _tokenize(ftext):
                            total += 1
                            if tok in term_surfaces:
                                cnt[tok] = cnt.get(tok, 0) + 1
                            if want_stem:
                                st = _stem_of(tok)
                                if st in stems_wanted:
                                    stem_counter[st] += 1
                    else:
                        # exact-only fields: a substring precheck on the
                        # casefolded bytes decides whether tokenizing can
                        # produce any query-term hit at all (cheap C scan;
                        # conservative — a hit just means tokenize anyway).
                        probe = (ftext.casefold() if ftext.isascii()
                                 else _fold(ftext))
                        if any(t in probe for t in term_surfaces):
                            for tok in _tokenize(ftext):
                                total += 1
                                if tok in term_surfaces:
                                    cnt[tok] = cnt.get(tok, 0) + 1
                counters[f] = cnt
                field_total[f] = total
            row_lens = lens.get(rid)
            if row_lens is None:
                # maintained row missing: field text is authoritative
                row_lens = {f: float(field_total[f]) for f in columns}
            elif len(row_lens) < len(columns):
                # partial maintained row (a pending doclen field): the
                # field-byte count fills the gap with the same value the
                # deriver will write (V8-19.05 honest partial).
                row_lens = {
                    f: (row_lens[f] if f in row_lens
                        else float(field_total[f]))
                    for f in columns
                }
            text_cnt = counters.get("text")
            acc = scores.setdefault(rid, {"score": 0.0, "matched": {}})
            for qt in qterms:
                tf_eff: dict[str, float] = {}
                exact_fields: dict[str, int] = {}
                for f in columns:
                    cnt = counters[f].get(qt.term, 0)
                    if cnt:
                        exact_fields[f] = cnt
                        tf_eff[f] = float(cnt)
                stem_extra = 0
                if qt.stem:
                    stem_extra = max(
                        stem_counter.get(qt.stem, 0)
                        - (text_cnt.get(qt.term, 0) if text_cnt else 0),
                        0)
                    if stem_extra:
                        tf_eff["text"] = tf_eff.get(
                            "text", 0.0) + STEM_SCALE * stem_extra
                if not tf_eff:
                    stats["tf_miss"] = stats.get("tf_miss", 0) + 1
                    continue
                acc["score"] += qt.scale * bm25f_score(
                    tf_eff, row_lens, avglen, idf_value=idf_of[qt.term],
                    weights=field_w)
                detail = {
                    "fields": exact_fields,
                    "stem_extra": stem_extra,
                    "channel": "exact" if exact_fields else "stem",
                }
                if qt.scale != 1.0 or qt.via != "exact":
                    detail["weight"] = qt.scale
                    if qt.source is not None:
                        detail["of"] = qt.source
                acc["matched"][qt.term] = detail
            for qt in ident_terms:
                if rid in posts.get(qt.term, ()):
                    # Identifier postings are phrase-level matches; a
                    # nominal single text-field hit keeps the weight
                    # declared without fabricating a token count.
                    acc["score"] += bm25f_score(
                        {"text": 1.0}, row_lens, avglen,
                        idf_value=idf_of[qt.term], weights=field_w)
                    acc["matched"][qt.term] = {
                        "fields": {"text": 1}, "stem_extra": 0,
                        "channel": "identifier",
                    }
    stats["scored_docs"] = examined
    if null_rows:
        stats["no_field_content"] = null_rows
    return scores


_QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def _quoted_spans(qv: QueryViewV7) -> list:
    """Quoted spans of the raw query mapped to ordered norm terms via
    byte offsets (V7-06.08).  Each entry is a tuple of ≥2 terms."""
    norm = getattr(qv, "norm", None)
    query = getattr(qv, "query", "") or ""
    terms = [t for t in (getattr(norm, "terms", ()) or ())
             if getattr(t, "channel", "text") == "text"]
    if not terms:
        return []
    spans: list[tuple] = []
    for match in _QUOTE_RE.finditer(query):
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        if not inner or not inner.strip():
            continue
        # byte range of the quoted text inside the query
        pre = query[:match.start(0)] + ('"' if match.group(1) is not None else "'")
        start_b = len(pre.encode("utf-8", "surrogatepass"))
        end_b = start_b + len(inner.encode("utf-8", "surrogatepass"))
        inside = [t.term for t in terms
                  if t.byte_start >= start_b and t.byte_end <= end_b]
        if len(inside) >= 2:
            spans.append(tuple(inside))
    return spans


def _phrase_flags(conn: sqlite3.Connection, qterms: list,
                  quoted: list, cand_rowids: set,
                  stats: dict) -> dict:
    """``rowid -> 1.0 | 0.5`` phrase/proximity flag (V7-06.08).

    1.0 — a quoted span or an adjacent term pair occurs contiguously in
    the ``text`` field (FTS5 phrase query); 0.5 — within NEAR(…, 3).
    One grouped MATCH each; absent on every candidate → all-zero map.
    """
    flags: dict[int, float] = {}
    if not cand_rowids or "text" not in _fts_columns(conn, FTS_TABLE):
        return flags
    phrases: list[tuple] = list(quoted)
    seq = [q.term for q in qterms]
    pairs = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)
             if seq[i] != seq[i + 1]]
    phrases.extend(pairs)
    if not phrases:
        return flags
    stats["phrase_probes"] = len(phrases)

    contiguous = " OR ".join(
        '"' + " ".join(str(t).replace('"', "") for t in ph) + '"'
        for ph in phrases)
    near = " OR ".join(
        "NEAR({}, 3)".format(
            " ".join(str(t).replace('"', "") for t in ph))
        for ph in phrases)
    exact_rows = _match_rowids(conn, FTS_TABLE, f"text : ({contiguous})")
    near_rows = _match_rowids(conn, FTS_TABLE, f"text : ({near})")
    exact_rows = (exact_rows or set()) & cand_rowids
    near_rows = (near_rows or set()) & cand_rowids
    for rid in near_rows:
        flags[rid] = 0.5
    for rid in exact_rows:
        flags[rid] = 1.0
    stats["phrase_hits"] = len(flags)
    return flags


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def _nomination_knobs(ctx: LaneContextV7) -> tuple:
    """``(nominate_terms_max, df_theta)`` resolved off ``ctx.policy`` —
    the ``retrieval_policy/v7`` per-profile channel (V75-04.02; Q1
    constants).  An absent policy or absent attrs yields the r0 snapshot
    (``NOMINATE_TERMS_MAX_R0``, θ disarmed).  Malformed values on a
    hand-built policy fail loudly — a mistyped arm must never silently
    reconfigure nomination (same discipline as ``graph_max_hops``)."""
    pol = getattr(ctx, "policy", None)
    raw_budget = getattr(pol, "nominate_terms_max", None)
    if raw_budget is None:
        budget = NOMINATE_TERMS_MAX_R0
    else:
        if isinstance(raw_budget, bool) or \
                not isinstance(raw_budget, (int, float)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "nominate_terms_max must be a positive integer, "
                f"got {raw_budget!r}")
        if isinstance(raw_budget, float):
            if not math.isfinite(raw_budget) or not raw_budget.is_integer():
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "nominate_terms_max must be an integral value, "
                    f"got {raw_budget!r}")
            raw_budget = int(raw_budget)
        if raw_budget < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"nominate_terms_max must be >= 1, got {raw_budget!r}")
        budget = int(raw_budget)
    raw_theta = getattr(pol, "nominate_df_theta", None)
    theta: Optional[float] = None
    if raw_theta is not None:
        if isinstance(raw_theta, bool) or \
                not isinstance(raw_theta, (int, float)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "nominate_df_theta must be null or a number, "
                f"got {raw_theta!r}")
        theta = float(raw_theta)
        if not math.isfinite(theta) or not 0.0 < theta <= 1.0:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "nominate_df_theta must be finite and in (0, 1], "
                f"got {raw_theta!r}")
    return budget, theta


def _arm_carrier_value(ctx: Any, attr: str, *keys: str) -> Any:
    """Carrier walk for a §23 arm off a lane context.

    Policy attribute (``attr``), then the policy's ``params``/``knobs``/
    ``arms`` maps under each spelling, then ``ctx.manifest``. ``None``
    is absent everywhere — a declared ``None`` never masks an outer
    carrier. Callers own domain validation.
    """
    pol = getattr(ctx, "policy", None)
    val = getattr(pol, attr, None)
    if val is not None:
        return val
    for carrier in ("params", "knobs", "arms"):
        mapping = getattr(pol, carrier, None)
        if isinstance(mapping, dict):
            for key in keys:
                val = mapping.get(key)
                if val is not None:
                    return val
    manifest = getattr(ctx, "manifest", None) or {}
    if isinstance(manifest, dict):
        for key in keys:
            val = manifest.get(key)
            if val is not None:
                return val
    return None


def _ctx_field_weight_of(ctx: Any) -> Optional[float]:
    """``context.ctx_field_weight`` (V8-06.03) — resolved weight or off.

    ``None`` = the arm is off (default). ``True`` selects the 0.5
    prior; ``False``/``"off"``/``"none"`` disarm explicitly; a finite
    number ≥ 0 weights the field (``0.0`` = indexed but contributing
    nothing — an honest arm value, not an absence). A mistyped
    declaration raises VALIDATION — it must never read as off.
    """
    raw = _arm_carrier_value(
        ctx,
        "context_ctx_field_weight",
        "context.ctx_field_weight",
        "context_ctx_field_weight",
        "ctx_field_weight",
    )
    if raw is None or raw is False or (
        isinstance(raw, str)
        and raw.strip().lower() in ("off", "none", "disabled")
    ):
        return None
    if raw is True:
        return CTX_FIELD_W_DEFAULT
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"context.ctx_field_weight must be a number >= 0, true, or "
            f"off — got {raw!r}",
        )
    out = float(raw)
    if not math.isfinite(out) or out < 0.0:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"context.ctx_field_weight must be finite and >= 0, "
            f"got {raw!r}",
        )
    return out


@dataclass(frozen=True)
class _LexArms:
    """The §23 lexical arm block (V8-07.02–07.06 + context arm).  ``df_floor``
    is ``None`` when the arm is ``off`` (the pre-fetch gate disarmed);
    ``ctx_field`` is ``None`` when ``context.ctx_field_weight`` is off."""

    df_floor: Optional[int]
    k_rescue: int
    rescue_rows: int
    stem_lru: int
    single_match: bool
    ctx_field: Optional[float]


def _lexical_arms(ctx: LaneContextV7) -> _LexArms:
    """Resolve the §23 lexical arms off ``ctx.policy``.

    Carrier order follows the dense-lane convention (V8 §23 plumbing):
    a direct attribute on the policy object (``lexical_df_floor``,
    ``lexical_K_rescue``, ``lexical_rescue_rows``, ``lexical_stem_lru``,
    ``lexical_single_match``), then a ``params``/``knobs``/``arms``
    mapping keyed by the dotted or underscored flag name, then
    ``ctx.manifest`` — absent everywhere, the spec defaults hold.
    Malformed values fail loudly (VALIDATION): a mistyped arm must never
    silently reconfigure the lane.
    """
    pol = getattr(ctx, "policy", None)
    manifest = getattr(ctx, "manifest", None) or {}

    def _raw(attr: str, *keys: str) -> Any:
        val = getattr(pol, attr, None)
        if val is not None:
            return val
        for carrier in ("params", "knobs", "arms"):
            mapping = getattr(pol, carrier, None)
            if isinstance(mapping, dict):
                for key in keys:
                    if mapping.get(key) is not None:
                        return mapping[key]
        for key in keys:
            if manifest.get(key) is not None:
                return manifest[key]
        return None

    def _int(raw: Any, name: str, default: int, minimum: int) -> int:
        if raw is None:
            return default
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"{name} must be an integer >= {minimum}, got {raw!r}")
        if isinstance(raw, float):
            if not math.isfinite(raw) or not raw.is_integer():
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"{name} must be an integral value, got {raw!r}")
            raw = int(raw)
        if raw < minimum:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"{name} must be >= {minimum}, got {raw!r}")
        return int(raw)

    raw_floor = _raw("lexical_df_floor", "lexical.df_floor",
                     "lexical_df_floor", "df_floor")
    if raw_floor is None:
        df_floor: Optional[int] = _DF_PREFETCH_FLOOR
    elif raw_floor is False or (
            isinstance(raw_floor, str)
            and raw_floor.strip().lower() in ("off", "none", "disabled")):
        df_floor = None
    else:
        df_floor = _int(raw_floor, "lexical_df_floor",
                        _DF_PREFETCH_FLOOR, 1)

    raw_sm = _raw("lexical_single_match", "lexical.single_match",
                  "lexical_single_match", "single_match")
    if raw_sm is None:
        single_match = False
    elif isinstance(raw_sm, bool):
        single_match = raw_sm
    else:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"lexical_single_match must be a boolean, got {raw_sm!r}")

    return _LexArms(
        df_floor=df_floor,
        k_rescue=_int(_raw("lexical_K_rescue", "lexical.K_rescue",
                           "lexical_k_rescue", "K_rescue"),
                      "lexical_K_rescue", _K_RESCUE_DEFAULT, 0),
        rescue_rows=_int(_raw("lexical_rescue_rows",
                              "lexical.rescue_rows", "lexical_rescue_rows",
                              "rescue_rows"),
                         "lexical_rescue_rows", _RESCUE_ROWS_DEFAULT, 1),
        stem_lru=_int(_raw("lexical_stem_lru", "lexical.stem_lru",
                           "lexical_stem_lru", "stem_lru"),
                      "lexical_stem_lru", _STEM_LRU_DEFAULT, 0),
        single_match=single_match,
        # V8-06.03 — the optional indexed ``ctx`` field's BM25F weight;
        # ``None`` = off (the field contributes nothing and is not
        # fetched). Resolved through the same carrier walk.
        ctx_field=_ctx_field_weight_of(ctx),
    )


def _cooc_int(raw: Any, name: str, default: int, minimum: int) -> int:
    """``_lexical_arms._int`` shape for the V8.5 nomination arms."""
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{name} must be an integer >= {minimum}, got {raw!r}")
    if isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"{name} must be an integral value, got {raw!r}")
        raw = int(raw)
    if raw < minimum:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{name} must be >= {minimum}, got {raw!r}")
    return int(raw)


def _cooc_arms(ctx: LaneContextV7) -> tuple:
    """``(nom_cooc, cooc_min, rare_df)`` — the V8.5 §05.02/05.03 arms.

    Same carrier convention as ``_lexical_arms`` (a
    ``lexical_nom_cooc``/``lexical_cooc_min``/``lexical_rare_df``
    attribute on the policy, then a ``params``/``knobs``/``arms``
    mapping keyed by the dotted or underscored flag — the channel a
    forensic ``PolicyPatch`` document drives — then ``ctx.manifest``);
    absent everywhere the V8.5 register priors hold
    (``policy.PARAM_DEFAULTS``: on / 2 / 4).  ``cooc_min`` >= 1 and
    ``rare_df`` >= 0 (``0`` disarms the rare-term rescue).  Malformed
    values fail loudly (VALIDATION): a mistyped arm must never
    silently reconfigure nomination.
    """
    raw_cooc = _arm_carrier_value(
        ctx, "lexical_nom_cooc", "lexical.nom_cooc",
        "lexical_nom_cooc", "nom_cooc")
    if raw_cooc is None:
        nom_cooc = _NOM_COOC_DEFAULT
    elif isinstance(raw_cooc, bool):
        nom_cooc = raw_cooc
    else:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"lexical.nom_cooc must be a boolean, got {raw_cooc!r}")
    cooc_min = _cooc_int(
        _arm_carrier_value(
            ctx, "lexical_cooc_min", "lexical.cooc_min",
            "lexical_cooc_min", "cooc_min"),
        "lexical.cooc_min", _COOC_MIN_DEFAULT, 1)
    rare_df = _cooc_int(
        _arm_carrier_value(
            ctx, "lexical_rare_df", "lexical.rare_df",
            "lexical_rare_df", "rare_df"),
        "lexical.rare_df", _RARE_DF_DEFAULT, 0)
    return nom_cooc, cooc_min, rare_df


def lane_lexical(ctx: LaneContextV7, qv: QueryViewV7,
                 slice: LaneSlice) -> LaneOutput:
    """Fielded BM25F over ``unit_fts`` — the L-lex workhorse (V7-06).

    Returns raw BM25F scores (never normalized) with matched-term lists
    and the phrase flag per candidate (V7-06.09).  Degradation is honest:
    ``unavailable`` when the fielded index is absent or unreadable,
    ``skipped`` when the query carries no usable terms, ``partial`` on a
    deadline-cut scan.
    """
    out = LaneOutput(lane=LANE, status=LaneStatus.OK)
    stats = out.stats
    stats["formula"] = FORMULA_ID
    stats["formula_status"] = FORMULA_STATUS_PROVISIONAL

    # V8 §23: resolve the lexical arm block first — ``stem_lru`` bounds
    # the process LRU before ``_query_terms`` computes query-side stems.
    arms = _lexical_arms(ctx)
    _configure_stem_lru(arms.stem_lru)

    deadline = _Deadline(getattr(slice, "deadline_ms", 0.0))
    if deadline.expired():
        out.status = LaneStatus.DEADLINE
        out.reason = "deadline"
        return out

    conn = _conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    if not has_table(conn, UNITS_TABLE) or not has_table(conn, FTS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_unit_fts"
        return out
    stem_ok = has_table(conn, STEM_TABLE)
    if not stem_ok:
        stats["stem_channel"] = "unavailable"
    # V8-06.03 ``context.ctx_field_weight``: the optional indexed ``ctx``
    # field joins the scored column set only when the arm is armed *and*
    # the schema carries it — armed-but-absent is reported, not faked.
    columns = [
        c
        for c in _fts_columns(conn, FTS_TABLE)
        if c in FIELD_W
        or (c == CTX_FIELD_NAME and arms.ctx_field is not None)
    ]
    stats["ctx_field_weight"] = {
        "armed": arms.ctx_field,
        "indexed": CTX_FIELD_NAME in columns,
    }
    if not columns:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_field_columns"
        return out

    norm = getattr(qv, "norm", None)
    if norm is None:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_analysis"
        return out
    qterms, ident_terms = _query_terms(qv)
    if not qterms and not ident_terms:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_terms"
        return out

    scope_id = str(getattr(ctx, "scope_id", ""))
    generation = getattr(ctx, "generation", None)

    uni, uni_trunc = _universe(
        conn, scope_id, generation, deadline,
        need_full=_elig_mode(getattr(ctx, "eligible", None)) == "callable",
        share=ctx)
    stats["n_universe"] = len(uni.by_rowid)
    if uni_trunc:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["universe_scan"] = "truncated"
        return out
    elig_rowids, elig_via = _resolve_eligibility(ctx, uni)
    stats["eligible_via"] = elig_via
    if elig_via == "unrecognized":
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_shape_unknown"
        return out

    corp = _eligible_corpus_stats(
        conn, uni, elig_rowids, columns, scope_id, generation, deadline,
        stats, share=ctx)
    if corp is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["stats_phase"] = "deadline_cut"
        return out
    n_eligible, avglen = corp
    stats["n_eligible"] = n_eligible
    stats["avglen"] = {f: round(v, 6) for f, v in avglen.items()}
    stats["df_maintained"] = _maintained_df(
        conn, scope_id, generation,
        [q.term for q in (*qterms, *ident_terms)])
    out.eligible = n_eligible

    nominate_budget, df_theta = _nomination_knobs(ctx)
    nom_cooc, cooc_min, rare_df = _cooc_arms(ctx)
    posts_nom = _collect_postings(
        conn, qterms, ident_terms, uni, elig_rowids, stem_ok, deadline,
        stats, nominate_budget=nominate_budget, df_theta=df_theta,
        n_eligible=n_eligible, df_maintained=stats["df_maintained"],
        df_floor=arms.df_floor, single_match=arms.single_match,
        canon_reasons=_canon_gate_reasons(qv),
        nom_cooc=nom_cooc, cooc_min=cooc_min, rare_df=rare_df)
    # ``examined`` counts posting rows the lane probed (pre-fence); the
    # per-doc scoring count rides in stats["scored_docs"].
    if posts_nom is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        out.examined = int(stats.get("posting_rows", 0))
        return out
    posts, nominated = posts_nom

    # V8-07.03 bounded rescue — thin nomination (or every content term
    # gated) earns a bounded exhaustive pass over the rarest gated or
    # dropped terms; produced rows merge and carry signals["rescue"].
    rescue_produced = _rescue_candidates(
        conn, qterms, posts, nominated, uni, elig_rowids, stem_ok,
        deadline, stats, k_rescue=arms.k_rescue,
        rescue_rows=arms.rescue_rows)
    if rescue_produced:
        nominated = set(nominated) | rescue_produced
    out.examined = int(stats.get("posting_rows", 0))

    scores = _score_candidates(
        conn, qterms, ident_terms, posts, nominated, uni, columns, avglen,
        n_eligible, deadline, stats, stem_cache={}, share=ctx,
        df_override={
            e["term"]: e["df"]
            for e in stats.get("df_prefetch_gated", ())
        } or None,
        scope_id=scope_id, generation=generation)
    if scores is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        out.examined += int(stats.get("scored_docs", 0))
        return out
    stats["scored"] = len(scores)

    cap = int(getattr(slice, "cap", 0) or 0)
    if cap <= 0:
        pool = POOLS.get(getattr(ctx, "budget", None))
        cap = pool.lane_cap if pool is not None else 200

    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["score"], str(uni.by_rowid[kv[0]]["unit_id"]),
                        kv[0]),
    )
    admitted = ordered[:cap]
    stats["cap"] = cap
    stats["overflow"] = max(0, len(ordered) - cap)

    cand_rowids = {rid for rid, _ in admitted}
    phrase = _phrase_flags(conn, qterms, _quoted_spans(qv), cand_rowids,
                           stats)

    rank = 0
    for rid, acc in admitted:
        unit = uni.by_rowid[rid]
        rank += 1
        signals = {
            "bm25f": acc["score"],
            "matched_terms": acc["matched"],
            "phrase": float(phrase.get(rid, 0.0)),
            "formula": FORMULA_ID,
        }
        if rid in rescue_produced:
            signals["rescue"] = True   # V8-07.03 (V8-20.04 explain)
        out.candidates.append(CandidateV7(
            unit_id=str(unit["unit_id"]),
            source_id=str(unit["source_id"]),
            revision=int(unit["revision"]),
            lane=LANE,
            rank=rank,
            raw_score=float(acc["score"]),
            signals=signals,
        ))

    if out.status == LaneStatus.OK and stats.get("no_field_content"):
        # Some matched rows carried no readable field bytes — their tf
        # was undercounted; honest partial coverage of the index.
        out.status = LaneStatus.PARTIAL
        out.reason = "no_field_content"
    return out


__all__ = [
    "lane_lexical", "bm25f_score", "idf", "porter_stem",
    "FIELD_W", "FIELD_B", "K1", "STEM_SCALE", "FORMULA_ID",
    "CTX_FIELD_NAME", "CTX_FIELD_W_DEFAULT",
]
