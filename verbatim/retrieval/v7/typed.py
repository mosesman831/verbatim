"""Typed lane for the V7 read path (V7-13/16, §32.11/12).

L-typed in the §04.2 pipeline: the grounded-fact index lane. Where L-lex
and L-time retrieve *units*, this lane retrieves units through the
structured fact records projected at write time:

- ``state_facts`` — ``(scope, state_key) -> value`` rows with a
  ``current | historical | disputed`` lifecycle (V7-16.03/04);
- ``preferences`` — first-person preference records with polarity and
  strength (V7-13.08, §32.12);
- ``events_v7`` — the deterministic event calendar (V7-09.08), consulted
  here for fact-shaped intents (L-time owns the window-driven use);
- ``t2_facts`` — optional quote-verified grounded facts (V7-13.12/13).

Match discipline (V7-06.03 — the typed lane's token prefilter is FTS, not
``LIKE '%term%'``):

- subject path: ``query.entity_canons`` (+ speaker canon) match
  ``subject_canon`` exactly; for ``state_facts`` the subject rides inside
  ``state_key`` as ``<subject canon>/<family>`` (V7-16.03), matched as an
  indexable prefix range;
- normalized-key path: query terms equality-match ``value_norm``,
  ``predicate_lemma``/``predicate`` and the state-key family segment —
  all folded forms, never substring scans;
- FTS path: open-domain terms run through ``unit_fts`` MATCH; the
  candidate set is the intersection of FTS postings with typed-record
  keys (fact rows whose supporting ``unit_id`` is a posting);
- ``t2_facts`` additionally get a *bounded* Python-side scan matching
  ``statement``/``object`` tokens: the statement is the fact's own text
  (T2 paraphrases can diverge from the unit surface) and the table is
  small by design (capped per-window extraction, V7-13.15).

Intent gate: the lane runs only for fact-shaped intents — current_value,
history_of, preference, comparison, open_domain, temporal_point,
temporal_range. Identifier lookups are L-exact-id's job; anything else
reports ``skipped`` / ``reason="intent_not_typed"``.

Intent-conditioned ordering (declared, deterministic):

- ``current_value``: units carrying a ``status='current'`` state fact
  lead (V7-16.04); disputed ranks over plain history inside the rest;
- ``history_of``: all statuses rank by fact anchor (valid_from /
  occurred) descending — history, never recency-by-recorded;
- ``preference``: preference facts lead, polarity/strength in signals.

Honesty rules honored (V7-04.03, LaneV7 protocol): eligibility before
rank on every lane (a held unit's facts are never emitted — and a
``t2_facts`` row is withheld when *any* pinned support unit is missing or
ineligible, the V7-13.13 closure rule); ``verified=0`` facts still emit
with ``signals["verified"]=0`` so the verdict can weigh them — never
hidden; deadline/scan-bound cuts report ``partial`` with real counts;
all four tables missing reports ``unavailable``/``no_typed_tables`` while
present-but-empty tables are an honest ``ok`` with zero candidates.

raw_score = typed prior (2.0 verified t2 · 1.5 current state · 1.3
disputed state · 1.2 preference · 1.1 unverified t2 · 1.0 event /
historical state) + subject-match and term/FTS bonuses.
``formula_status: provisional/v7-r0`` — priors and bonuses are declared
constants awaiting the §32.0 formula search, not tuned values.

The lane is stdlib + sqlite3 only and codes against the frozen
``types_v7`` contract; it never opens a transaction on the caller's
pinned read snapshot.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from verbatim.core.types_v7 import (
    CandidateV7,
    IntentClass,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)

LANE_NAME = "typed"  # LaneName.TYPED — stable coverage key
LANE_VERSION = "typed_lane/v7-r0"  # provisional constants (V7-32.01)

TYPED_TABLES: tuple[str, ...] = (
    "state_facts",
    "preferences",
    "events_v7",
    "t2_facts",
)

# Fact-shaped intents (brief item 1). Identifier lookups route to
# L-exact-id; ``lookup``/``multi_hop``/``why``/``abstain``/``duration``/
# ``count_aggregate``/``temporal_order`` are served by peer lanes.
_TYPED_INTENTS = frozenset(
    {
        IntentClass.CURRENT_VALUE,
        IntentClass.HISTORY_OF,
        IntentClass.PREFERENCE,
        IntentClass.COMPARISON,
        IntentClass.OPEN_DOMAIN,
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_RANGE,
    }
)

ROW_LIMIT = 4096  # per-table bounded scan/fetch; hitting it -> partial/scan_bound
FTS_TERM_LIMIT = 24
FTS_ROW_LIMIT = 1000
CANON_LIMIT = 32
TERM_LIMIT = 24
IN_CHUNK = 200
DEADLINE_CHECK_ROWS = 64

# Typed priors — provisional/v7-r0, declared not tuned. Unnamed cells in
# the brief (unverified t2, disputed state) interpolate inside the
# mandated orderings: verified t2 > current state > disputed > preference
# > unverified t2 > event / historical.
P_T2_VERIFIED = 2.0
P_STATE_CURRENT = 1.5
P_STATE_DISPUTED = 1.3
P_PREFERENCE = 1.2
P_T2_UNVERIFIED = 1.1
P_STATE_OTHER = 1.0  # historical / NULL / unknown status
P_EVENT = 1.0

B_SUBJECT = 1.0  # subject-canon match bonus — strictly above B_TERM's
# maximum so an exact canon match always outranks a term-only fallback
B_TERM = 0.5  # x fraction of query terms matched on typed columns
B_FTS = 0.25  # unit text matched the FTS postings

_STATE_STATUS_LIFECYCLE = {
    "current": "current",
    "historical": "historical",
    "disputed": "disputed",
}

_TERM_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# small pure helpers (deterministic; local copies keep the lane free of
# concurrent-worker imports — same convention as temporal.py)
# ---------------------------------------------------------------------------


def _fold(text: Any) -> str:
    """``norm/v2``-equivalent matching fold (NFKC + casefold + diacritic
    strip); idempotent over already-folded input."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", str(text)).casefold()
    if norm.isascii():
        # No ASCII codepoint has a nonzero combining class — the filter
        # below is the identity on ASCII.
        return norm
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _tokens(text: Any) -> tuple[str, ...]:
    return tuple(_TERM_RE.findall(_fold(text or "")))


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return row is not None


def _parse_json(text: Any, default: Any) -> Any:
    if not text:
        return default
    # Most row payloads are the empty container — skip the parser.
    if text == "{}":
        return {}
    if text == "[]":
        return []
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _coerce_json(value: Any, default: Any) -> Any:
    """Resolve a lazily-held JSON payload.  Scans store the raw column text
    in ``_Fact.pins``/``quotes`` rather than parsing per row (most scanned
    rows are dropped by the match filter before the payload is ever read);
    this parses on first use and passes already-materialized values
    (dicts/lists from direct construction) through unchanged."""
    if isinstance(value, str) or value is None:
        return _parse_json(value, default)
    return value


def _state_subject(state_key: Optional[str]) -> str:
    """Subject canon segment of a ``<subject>/<family>`` state key
    (V7-16.03); empty when the key carries no subject segment."""
    if not state_key or "/" not in state_key:
        return ""
    return state_key.split("/", 1)[0]


# ---------------------------------------------------------------------------
# context resolution (same contract surface as the sibling lanes)
# ---------------------------------------------------------------------------


def _resolve_conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
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


def _eligibility(elig: Any) -> Optional[Callable[[dict], bool]]:
    """Normalize ``ctx.eligible`` into ``unit_row -> bool`` (frozen
    contract: callable, ``is_eligible`` object, or set of unit_ids).
    Anything else — including a missing handle — fails closed."""
    if elig is None:
        return None
    if callable(elig):

        def _call(row: dict) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row["unit_id"]))

        return _call
    if hasattr(elig, "is_eligible"):
        return lambda row: bool(elig.is_eligible(row))
    if hasattr(elig, "__contains__"):
        return lambda row: row["unit_id"] in elig
    return None


def _intent_classes(qv: QueryViewV7) -> set[IntentClass]:
    """Primary + secondary classes, coerced to ``IntentClass`` (tolerates
    raw-string fixtures)."""
    intent = qv.intent
    if intent is None:
        return set()
    raw = list(intent.classes or ()) + [intent.primary]
    out: set[IntentClass] = set()
    for c in raw:
        if isinstance(c, IntentClass):
            out.add(c)
            continue
        try:
            out.add(IntentClass(str(c)))
        except (ValueError, TypeError):
            continue
    out.discard(None)  # type: ignore[arg-type]
    return out


def _query_terms(qv: QueryViewV7) -> tuple[str, ...]:
    """Folded text/stem query terms, deduped, ordered, bounded."""
    seen: list[str] = []
    norm = qv.norm
    for t in (norm.terms if norm is not None else ()) or ():
        if getattr(t, "channel", "text") not in ("text", "stem"):
            continue
        term = _fold(getattr(t, "term", t))
        if term and term not in seen:
            seen.append(term)
    return tuple(seen[:TERM_LIMIT])


def _subject_canons(qv: QueryViewV7) -> frozenset:
    """Canons eligible for subject equality: query entities plus the
    speaker ("what is *my* job"), folded, bounded."""
    cans: list[str] = []
    for c in (qv.entity_canons or ()) + (
        (qv.speaker_canon,) if qv.speaker_canon else ()
    ):
        fc = _fold(c)
        if fc and fc not in cans:
            cans.append(fc)
    return frozenset(cans[:CANON_LIMIT])


def _fts_quote(term: str) -> str:
    return '"' + str(term).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# fact model — one row of one typed table, normalized across kinds
# ---------------------------------------------------------------------------


@dataclass
class _Fact:
    kind: str  # state | preference | event | t2
    fact_id: str  # row identifier (event_id / fact_id / natural key)
    unit_id: str = ""
    unit_ids: tuple = ()  # t2 closure set (all pinned supports)
    subject: str = ""
    state_key: str = ""
    value_text: str = ""
    value_norm: str = ""
    status: str = ""
    polarity: str = ""
    strength: str = ""
    predicate: str = ""
    object_text: str = ""
    statement: str = ""
    # ``pins``/``quotes`` may hold the raw column text during a scan and are
    # resolved through ``_coerce_json`` only on the emitted best fact.
    pins: Any = field(default_factory=dict)
    quotes: Any = None
    valid_from_us: Optional[int] = None
    valid_to_us: Optional[int] = None
    occurred_start_us: Optional[int] = None
    occurred_end_us: Optional[int] = None
    producer: str = ""
    model_id: str = ""
    verified: Optional[int] = None
    # match bookkeeping (filled during evaluation)
    subj_match: bool = False
    term_hits: frozenset = frozenset()
    fts_match: bool = False
    score: float = 0.0

    @property
    def anchor_us(self) -> Optional[int]:
        """Ordering instant for history intents: the fact's own validity
        start — never the row's recorded time."""
        if self.kind == "state":
            return self.valid_from_us
        return self.occurred_start_us

    def prior(self) -> float:
        if self.kind == "state":
            if self.status == "current":
                return P_STATE_CURRENT
            if self.status == "disputed":
                return P_STATE_DISPUTED
            return P_STATE_OTHER
        if self.kind == "preference":
            return P_PREFERENCE
        if self.kind == "t2":
            return P_T2_VERIFIED if self.verified else P_T2_UNVERIFIED
        return P_EVENT

    def term_token_set(self) -> frozenset:
        """Folded matchable tokens for the term path (typed columns only —
        the unit's prose text is the FTS path's job, V7-06.03)."""
        if self.kind == "state":
            toks = set(_tokens(self.state_key))
            toks.update(_tokens(self.value_text))
            vn = _fold(self.value_norm)
            if vn:
                toks.add(vn)
                toks.update(_tokens(vn))
            return frozenset(toks)
        if self.kind == "preference":
            return frozenset(_tokens(self.object_text))
        if self.kind == "event":
            toks = set(_tokens(self.object_text))
            pred = _fold(self.predicate)
            if pred:
                toks.add(pred)
            return frozenset(toks)
        # t2: statement + object + predicate + state_key
        toks = set(_tokens(self.statement))
        toks.update(_tokens(self.object_text))
        toks.update(_tokens(self.state_key))
        pred = _fold(self.predicate)
        if pred:
            toks.add(pred)
        return frozenset(toks)

    def subject_match(self, canons: frozenset) -> bool:
        if not canons:
            return False
        if self.kind == "state":
            return (
                _fold(_state_subject(self.state_key)) in canons
                or _fold(self.state_key) in canons
            )
        return _fold(self.subject) in canons


#: Fields each fact kind feeds to ``term_token_set`` — the substring probe
#: is built from exactly these so a miss proves zero term hits.
_TERM_PROBE_FIELDS = {
    "state": ("state_key", "value_text", "value_norm"),
    "preference": ("object_text",),
    "event": ("object_text", "predicate"),
    "t2": ("statement", "object_text", "state_key", "predicate"),
}


def _term_maybe_present(fact: "_Fact", term_set: frozenset) -> bool:
    """Conservative possibility test for ``term_token_set``.

    ``False`` only when no matchable field can produce a query-term token
    — a token equal to a term requires that term as a substring of the
    field's folded form, and for ASCII the fold is just ``casefold`` — so
    a clean scan proves zero hits.  ``True`` when a hit is possible or
    undecidable (a non-ASCII field can merge marks under ``_fold``, so a
    raw substring test is not a safe superset) — the caller tokenizes."""
    fields = _TERM_PROBE_FIELDS.get(fact.kind)
    if fields is None:
        return True
    for name in fields:
        text = getattr(fact, name) or ""
        if not text:
            continue
        if not text.isascii():
            return True
        cf = text.casefold()
        for t in term_set:
            if t in cf:
                return True
    return False


@dataclass
class _Cand:
    """One emitted candidate: a unit plus every matched fact on it."""

    unit_id: str
    source_id: str = ""
    revision: int = 0
    speaker_canon: str = ""
    unit_occurred_start_us: Optional[int] = None
    unit_occurred_end_us: Optional[int] = None
    unit_recorded_at_us: Optional[int] = None
    facts: list = field(default_factory=list)

    def best(self) -> _Fact:
        """Primary fact for signal payload: highest score, then latest
        anchor, then id — total, deterministic."""
        return max(
            self.facts,
            key=lambda f: (f.score, f.anchor_us if f.anchor_us is not None else -1, f.fact_id),
        )

    @property
    def score(self) -> float:
        return max(f.score for f in self.facts)

    @property
    def anchor_us(self) -> Optional[int]:
        anchors = [f.anchor_us for f in self.facts if f.anchor_us is not None]
        return max(anchors) if anchors else None

    @property
    def has_current_state(self) -> bool:
        return any(f.kind == "state" and f.status == "current" for f in self.facts)

    @property
    def has_preference(self) -> bool:
        return any(f.kind == "preference" for f in self.facts)


# ---------------------------------------------------------------------------
# table scans — each returns (facts, examined, truncated) and honors the
# deadline through the caller's ``expired`` probe
# ---------------------------------------------------------------------------

_STATE_COLS = (
    "state_key, unit_id, value_text, value_norm, valid_from_us,"
    " valid_to_us, status, producer, pins_json"
)


def _state_row(r: tuple, fts_units: Optional[set]) -> _Fact:
    (
        state_key,
        unit_id,
        value_text,
        value_norm,
        vf,
        vt,
        status,
        producer,
        pins_json,
    ) = r
    return _Fact(
        kind="state",
        fact_id=f"state:{state_key}:{unit_id}:{value_norm or ''}",
        unit_id=unit_id,
        state_key=state_key or "",
        subject=_state_subject(state_key),
        value_text=value_text or "",
        value_norm=value_norm or "",
        status=status or "",
        producer=producer or "",
        pins=pins_json,
        valid_from_us=vf,
        valid_to_us=vt,
        fts_match=unit_id in fts_units if fts_units else False,
    )


def _state_where(canons: frozenset, terms: frozenset) -> tuple[str, list]:
    """Indexed match predicates for ``state_facts``: subject prefix range
    on ``<canon>/`` (BINARY collation: '0' is '/' + 1), whole-key canon
    equality, family-segment term equality, ``value_norm`` term equality.
    No LIKE anywhere (V7-06.03)."""
    parts: list[str] = []
    params: list = []
    canon_list = sorted(canons)
    if canon_list:
        parts.append(f"state_key IN ({_ph(len(canon_list))})")
        params.extend(canon_list)
        range_parts = []
        for c in canon_list:
            range_parts.append("(state_key >= ? AND state_key < ?)")
            params.extend((c + "/", c + "0"))
        parts.append("(" + " OR ".join(range_parts) + ")")
    term_list = sorted(terms)
    if term_list:
        ph = _ph(len(term_list))
        parts.append(f"substr(state_key, instr(state_key, '/') + 1) IN ({ph})")
        params.extend(term_list)
        parts.append(f"value_norm IN ({ph})")
        params.extend(term_list)
    return ("(" + " OR ".join(parts) + ")" if parts else "0"), params


def _scan_state_facts(
    conn,
    scope_id: str,
    generation: int,
    canons: frozenset,
    terms: frozenset,
    fts_units: Optional[set],
    expired: Callable[[], bool],
    out: LaneOutput,
) -> tuple[list, bool]:
    facts: list = []
    truncated = False

    def _fetch(where: str, params: list, fts_flag: bool) -> bool:
        nonlocal truncated
        # V7-30.02: the fence is ``generation <= pinned`` and the newest
        # row per natural key (scope, state_key, unit_id, value_norm) is
        # authoritative — MAX(generation) + GROUP BY resolves it inline.
        sql = (
            f"SELECT {_STATE_COLS}, MAX(generation) FROM state_facts"
            " WHERE scope_id = ? AND generation <= ?"
            f" AND {where}"
            " GROUP BY scope_id, state_key, unit_id, value_norm"
            " ORDER BY state_key, unit_id, value_norm LIMIT ?"
        )
        n = 0
        for row in conn.execute(sql, [scope_id, generation, *params, ROW_LIMIT + 1]):
            n += 1
            out.examined += 1
            if n > ROW_LIMIT:
                truncated = True
                break
            fact = _state_row(row[:-1], fts_units)
            fact.fts_match = fact.fts_match or fts_flag
            facts.append(fact)
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                return True  # deadline
        return False

    where, params = _state_where(canons, terms)
    if where != "0" and _fetch(where, params, False):
        return facts, True  # deadline encoded as truncated-by-caller
    if fts_units:
        for chunk in _chunks(sorted(fts_units), IN_CHUNK):
            if expired():
                return facts, True
            if _fetch(f"unit_id IN ({_ph(len(chunk))})", list(chunk), True):
                return facts, True
    if terms:
        # residual term candidacy on free-text columns (value_text) that
        # the normalized-column predicates cannot reach — bounded,
        # deterministic ORDER BY; candidacy is decided row-side in the
        # shared match pass.
        if _fetch("1", [], False):
            return facts, True
    return facts, truncated


def _scan_preferences(
    conn,
    scope_id: str,
    generation: int,
    canons: frozenset,
    terms: frozenset,
    fts_units: Optional[set],
    expired: Callable[[], bool],
    out: LaneOutput,
) -> tuple[list, bool]:
    facts: list = []
    truncated = False

    def _fetch(where: str, params: list, fts_flag: bool) -> bool:
        nonlocal truncated
        # V7-30.02: ``generation <= pinned`` + newest row per natural key
        # (scope, subject_canon, unit_id, object_text) via MAX+GROUP BY.
        sql = (
            "SELECT subject_canon, unit_id, object_text, polarity, strength,"
            " occurred_start_us, pins_json, MAX(generation) FROM preferences"
            " WHERE scope_id = ? AND generation <= ?"
            f" AND {where}"
            " GROUP BY scope_id, subject_canon, unit_id, object_text"
            " ORDER BY subject_canon, unit_id, object_text LIMIT ?"
        )
        n = 0
        for row in conn.execute(sql, [scope_id, generation, *params, ROW_LIMIT + 1]):
            n += 1
            out.examined += 1
            if n > ROW_LIMIT:
                truncated = True
                break
            (subj, uid, obj, pol, strength, occ, pins_json, _gen) = row
            facts.append(
                _Fact(
                    kind="preference",
                    fact_id=f"pref:{subj}:{uid}:{obj or ''}",
                    unit_id=uid,
                    subject=subj or "",
                    object_text=obj or "",
                    polarity=pol or "",
                    strength=strength or "",
                    occurred_start_us=occ,
                    pins=pins_json,
                    fts_match=fts_flag or (uid in fts_units if fts_units else False),
                )
            )
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                return True
        return False

    canon_list = sorted(canons)
    if canon_list and _fetch(
        f"subject_canon IN ({_ph(len(canon_list))})", canon_list, False
    ):
        return facts, True
    if fts_units:
        for chunk in _chunks(sorted(fts_units), IN_CHUNK):
            if expired():
                return facts, True
            if _fetch(f"unit_id IN ({_ph(len(chunk))})", list(chunk), True):
                return facts, True
    if terms:
        # object_text term candidacy (open-domain): bounded scan, exact
        # token membership row-side — never a LIKE substring prefilter.
        if _fetch("1", [], False):
            return facts, True
    return facts, truncated


def _scan_events(
    conn,
    scope_id: str,
    generation: int,
    canons: frozenset,
    terms: frozenset,
    fts_units: Optional[set],
    expired: Callable[[], bool],
    out: LaneOutput,
) -> tuple[list, bool]:
    facts: list = []
    truncated = False

    def _fetch(where: str, params: list, fts_flag: bool) -> bool:
        nonlocal truncated
        sql = (
            "SELECT event_id, unit_id, subject_canon, predicate_lemma,"
            " object_text, polarity, occurred_start_us, occurred_end_us,"
            " pins_json FROM events_v7"
            # event_id is a minted key (Rule B): rows stay valid at/below
            # the pinned generation — ``<=``, no natural-key resolution.
            " WHERE scope_id = ? AND generation <= ?"
            f" AND {where}"
            " ORDER BY event_id LIMIT ?"
        )
        n = 0
        for row in conn.execute(sql, [scope_id, generation, *params, ROW_LIMIT + 1]):
            n += 1
            out.examined += 1
            if n > ROW_LIMIT:
                truncated = True
                break
            (eid, uid, subj, pred, obj, pol, os_, oe, pins_json) = row
            facts.append(
                _Fact(
                    kind="event",
                    fact_id=f"event:{eid}",
                    unit_id=uid,
                    subject=subj or "",
                    predicate=pred or "",
                    object_text=obj or "",
                    polarity=pol or "",
                    occurred_start_us=os_,
                    occurred_end_us=oe,
                    pins=pins_json,
                    fts_match=fts_flag or (uid in fts_units if fts_units else False),
                )
            )
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                return True
        return False

    parts: list[str] = []
    params: list = []
    canon_list = sorted(canons)
    if canon_list:
        parts.append(f"subject_canon IN ({_ph(len(canon_list))})")
        params.extend(canon_list)
    term_list = sorted(terms)
    if term_list:
        parts.append(f"predicate_lemma IN ({_ph(len(term_list))})")
        params.extend(term_list)
    if parts and _fetch("(" + " OR ".join(parts) + ")", params, False):
        return facts, True
    if fts_units:
        for chunk in _chunks(sorted(fts_units), IN_CHUNK):
            if expired():
                return facts, True
            if _fetch(f"unit_id IN ({_ph(len(chunk))})", list(chunk), True):
                return facts, True
    if terms:
        # object_text term candidacy — same bounded-scan discipline.
        if _fetch("1", [], False):
            return facts, True
    return facts, truncated


def _scan_t2(
    conn,
    scope_id: str,
    generation: int,
    expired: Callable[[], bool],
    out: LaneOutput,
    stats: dict,
) -> tuple[list, bool]:
    """Bounded whole-table scan for T2 facts: ``unit_ids_json`` closure and
    ``statement``/``object`` token matching are evaluated row-side (the
    table is capped-small by design — V7-13.15)."""
    facts: list = []
    truncated = False
    sql = (
        'SELECT fact_id, unit_ids_json, statement, quotes_json,'
        ' subject_canon, predicate, "object", occurred_start_us,'
        ' occurred_end_us, state_key, model_id, verified FROM t2_facts'
        # fact_id is a minted key (Rule B) — ``<=`` at/below the pin.
        " WHERE scope_id = ? AND generation <= ?"
        " ORDER BY fact_id LIMIT ?"
    )
    n = 0
    for row in conn.execute(sql, [scope_id, generation, ROW_LIMIT + 1]):
        n += 1
        out.examined += 1
        if n > ROW_LIMIT:
            truncated = True
            break
        (
            fact_id,
            uids_json,
            statement,
            quotes_json,
            subj,
            pred,
            obj,
            os_,
            oe,
            state_key,
            model_id,
            verified,
        ) = row
        unit_ids_raw = _parse_json(uids_json, [])
        unit_ids = (
            tuple(str(u) for u in unit_ids_raw if u)
            if isinstance(unit_ids_raw, list)
            else ()
        )
        if unit_ids_raw and not unit_ids:
            stats["t2_bad_unit_json"] = stats.get("t2_bad_unit_json", 0) + 1
        facts.append(
            _Fact(
                kind="t2",
                fact_id=str(fact_id),
                unit_id=unit_ids[0] if unit_ids else "",
                unit_ids=unit_ids,
                subject=subj or "",
                predicate=pred or "",
                object_text=obj or "",
                statement=statement or "",
                state_key=state_key or "",
                occurred_start_us=os_,
                occurred_end_us=oe,
                model_id=model_id or "",
                verified=int(verified or 0),
                quotes=quotes_json,
            )
        )
        if n % DEADLINE_CHECK_ROWS == 0 and expired():
            return facts, True
    return facts, truncated


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_typed(ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice) -> LaneOutput:
    """L-typed: grounded-fact index lane over state_facts / preferences /
    events_v7 / t2_facts (V7-13/16, §32.11/12)."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = "provisional/v7-r0"

    t0 = time.monotonic()
    deadline_s = max(0.0, float(slice.deadline_ms or 0.0)) / 1000.0

    def expired() -> bool:
        return (time.monotonic() - t0) > deadline_s

    conn = _resolve_conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    if ctx.generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out
    eligible_fn = _eligibility(getattr(ctx, "eligible", None))
    if eligible_fn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_handle_missing"
        return out

    classes = _intent_classes(qv)
    stats["intent_classes"] = sorted(c.value for c in classes)
    if not (classes & _TYPED_INTENTS):
        out.status = LaneStatus.SKIPPED
        out.reason = "intent_not_typed"
        return out

    present = {name: _has_table(conn, name) for name in TYPED_TABLES}
    stats["tables"] = {name: ("ok" if ok else "missing") for name, ok in present.items()}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        stats["missing_tables"] = missing
    if not any(present.values()):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_typed_tables"
        return out
    if not _has_table(conn, "units"):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "units_table_missing"
        return out

    cap = max(0, int(slice.cap or 0))
    if cap == 0:
        stats["mode"] = "capped"
        return out
    if expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["mode"] = "none"
        return out

    terms = _query_terms(qv)
    term_set = frozenset(terms)
    canons = _subject_canons(qv)
    stats["terms"] = len(terms)
    stats["canons"] = len(canons)

    # FTS postings -> unit_ids (V7-06.03): the term-level candidate filter
    # for unit text. None = index absent/match error (degraded, honest);
    # empty set = index live, no hits.
    fts_units: Optional[set] = None
    if term_set:
        if _has_table(conn, "unit_fts"):
            match = " OR ".join(_fts_quote(t) for t in terms[:FTS_TERM_LIMIT])
            try:
                rows = conn.execute(
                    "SELECT rowid FROM unit_fts WHERE unit_fts MATCH ? LIMIT ?",
                    (match, FTS_ROW_LIMIT),
                ).fetchall()
            except sqlite3.Error:
                stats["fts"] = "match_error"
            else:
                rowids = [int(r[0]) for r in rows]
                fts_units = set()
                carrier = (
                    "unit_fts_rows" if _has_table(conn, "unit_fts_rows") else "units"
                )
                col = "row_id" if carrier == "unit_fts_rows" else "rowid"
                # V7-30.02: the carrier/units table is versioned
                # (unit_id, generation) — a posting only nominates a unit
                # when the hit row IS that unit's latest row at/below the
                # pin; a stale-generation hit must not resurrect it.
                for chunk in _chunks(rowids, IN_CHUNK):
                    for (uid,) in conn.execute(
                        f"SELECT r.unit_id FROM {carrier} r"
                        f" JOIN (SELECT unit_id, MAX(generation) AS mg"
                        f"        FROM {carrier}"
                        "        WHERE scope_id = ? AND generation <= ?"
                        "        GROUP BY unit_id) lm"
                        "   ON lm.unit_id = r.unit_id"
                        "   AND lm.mg = r.generation"
                        " WHERE r.scope_id = ? AND r.generation <= ?"
                        f" AND r.{col} IN ({_ph(len(chunk))})",
                        [ctx.scope_id, ctx.generation,
                         ctx.scope_id, ctx.generation, *chunk],
                    ):
                        fts_units.add(uid)
                stats["fts"] = "ok"
                stats["fts_units"] = len(fts_units)
        else:
            stats["fts"] = "unavailable"
    else:
        stats["fts"] = "no_terms"

    if not canons and not term_set and not fts_units:
        stats["match_keys"] = "none"
        stats["mode"] = "none"
        if missing:
            # nothing to key on, but the absent indexes are still a real
            # coverage degradation — report it.
            out.status = LaneStatus.PARTIAL
            out.reason = "typed_table_missing"
        return out  # 0 candidates — the lane ran, nothing to key on

    deadline_hit = False
    truncated = False
    modes: list[str] = []
    facts: list = []

    def _scan(name: str, fn) -> None:
        nonlocal deadline_hit, truncated
        if deadline_hit or not present[name]:
            return
        modes.append(name)
        got, hit = fn()
        facts.extend(got)
        if hit:
            if expired():
                deadline_hit = True
            else:
                truncated = True
        if expired():
            deadline_hit = True

    _scan(
        "state_facts",
        lambda: _scan_state_facts(
            conn, ctx.scope_id, ctx.generation, canons, term_set,
            fts_units, expired, out,
        ),
    )
    _scan(
        "preferences",
        lambda: _scan_preferences(
            conn, ctx.scope_id, ctx.generation, canons, term_set,
            fts_units, expired, out,
        ),
    )
    _scan(
        "events_v7",
        lambda: _scan_events(
            conn, ctx.scope_id, ctx.generation, canons, term_set,
            fts_units, expired, out,
        ),
    )
    _scan(
        "t2_facts",
        lambda: _scan_t2(
            conn, ctx.scope_id, ctx.generation, expired, out, stats,
        ),
    )
    stats["mode"] = "+".join(modes) if modes else "none"

    # -- match evaluation ---------------------------------------------------
    # A row can arrive through both the predicate path and an FTS-unit
    # chunk — dedupe on the fact's natural key, merging the fts flag.
    seen: dict[str, _Fact] = {}
    deduped: list = []
    for fact in facts:
        prev = seen.get(fact.fact_id)
        if prev is None:
            seen[fact.fact_id] = fact
            deduped.append(fact)
        else:
            prev.fts_match = prev.fts_match or fact.fts_match
    facts = deduped

    matched: list = []
    for fact in facts:
        fact.subj_match = fact.subject_match(canons)
        hits: frozenset = frozenset()
        if term_set and _term_maybe_present(fact, term_set):
            hits = frozenset(fact.term_token_set() & term_set)
        fact.term_hits = hits
        if fact.kind == "t2" and fts_units:
            fact.fts_match = bool(set(fact.unit_ids) & fts_units)
        if fact.subj_match or fact.term_hits or fact.fts_match:
            matched.append(fact)
    stats["facts_matched"] = len(matched)

    # -- unit join + eligibility (before rank, V7-05.08) ---------------------
    needed: set[str] = set()
    for fact in matched:
        if fact.kind == "t2":
            needed.update(fact.unit_ids)
        elif fact.unit_id:
            needed.add(fact.unit_id)

    umap: dict[str, tuple] = {}
    if needed and not deadline_hit:
        uid_list = sorted(needed)
        for chunk in _chunks(uid_list, IN_CHUNK):
            if expired():
                deadline_hit = True
                break
            # V7-30.02: ``generation <= pinned`` with the newest row per
            # unit_id winning — ORDER BY generation DESC + first-seen.
            rows = conn.execute(
                "SELECT u.rowid, u.unit_id, u.source_id, u.revision, u.kind,"
                " u.session_id, u.speaker_canon, u.recorded_at_us,"
                " u.occurred_start_us, u.occurred_end_us,"
                " u.occurred_precision, u.occurred_source"
                " FROM units u WHERE u.scope_id = ? AND u.generation <= ?"
                f" AND u.unit_id IN ({_ph(len(chunk))})"
                " ORDER BY u.unit_id, u.generation DESC",
                [ctx.scope_id, ctx.generation, *chunk],
            ).fetchall()
            out.examined += len(rows)
            for ur in rows:
                umap.setdefault(ur[1], ur)

    def _unit_row(ur: tuple) -> dict:
        return {
            "unit_id": ur[1],
            "source_id": ur[2],
            "revision": ur[3],
            "kind": ur[4],
            "session_id": ur[5],
            "speaker_canon": ur[6],
            "recorded_at_us": ur[7],
            "occurred_start_us": ur[8],
            "occurred_end_us": ur[9],
            "occurred_precision": ur[10],
            "occurred_source": ur[11],
        }

    pool: dict[str, _Cand] = {}
    dropped_ineligible = 0
    closure_withheld = 0
    orphaned = 0
    for fact in matched:
        if deadline_hit:
            break
        if fact.kind == "t2":
            # V7-13.13 closure: the fact lives only while EVERY pinned
            # support unit is in-fence and eligible — a held/superseded
            # support retires the fact, never silently re-points it.
            if not fact.unit_ids:
                closure_withheld += 1
                continue
            rows = [umap.get(u) for u in fact.unit_ids]
            if any(ur is None for ur in rows):
                orphaned += 1
                continue
            if not all(eligible_fn(_unit_row(ur)) for ur in rows):
                closure_withheld += 1
                continue
            ur = rows[0]
        else:
            ur = umap.get(fact.unit_id)
            if ur is None:
                orphaned += 1
                continue
            if not eligible_fn(_unit_row(ur)):
                dropped_ineligible += 1
                continue
        cand = pool.get(fact.unit_id)
        if cand is None:
            cand = _Cand(
                unit_id=fact.unit_id,
                source_id=ur[2] or "",
                revision=int(ur[3] or 0),
                speaker_canon=ur[6] or "",
                unit_recorded_at_us=ur[7],
                unit_occurred_start_us=ur[8],
                unit_occurred_end_us=ur[9],
            )
            pool[fact.unit_id] = cand
        cand.facts.append(fact)

    stats["facts_admitted"] = sum(len(c.facts) for c in pool.values())
    if dropped_ineligible:
        stats["dropped_ineligible"] = dropped_ineligible
    if closure_withheld:
        stats["closure_withheld"] = closure_withheld
    if orphaned:
        stats["orphaned_facts"] = orphaned
    out.eligible = len(pool)

    # -- scoring --------------------------------------------------------------
    n_terms = len(term_set)
    for cand in pool.values():
        for fact in cand.facts:
            fact.score = (
                fact.prior()
                + B_SUBJECT * (1.0 if fact.subj_match else 0.0)
                + B_TERM * (len(fact.term_hits) / n_terms if n_terms else 0.0)
                + B_FTS * (1.0 if fact.fts_match else 0.0)
            )

    # -- intent-conditioned ordering (brief item 3) ---------------------------
    if IntentClass.CURRENT_VALUE in classes:
        tier_of = lambda c: 0 if c.has_current_state else 1  # noqa: E731
    elif IntentClass.PREFERENCE in classes:
        tier_of = lambda c: 0 if c.has_preference else 1  # noqa: E731
    else:
        tier_of = lambda c: 0  # noqa: E731
    history = IntentClass.HISTORY_OF in classes

    def order_key(c: _Cand) -> tuple:
        if history:
            # all statuses, fact anchor descending (V7-16.04 chain order)
            return (
                tier_of(c),
                0 if c.anchor_us is not None else 1,
                -(c.anchor_us or 0),
                -c.score,
                c.unit_id,
            )
        return (tier_of(c), -c.score, c.unit_id)

    final = sorted(pool.values(), key=order_key)[:cap]
    stats["overflow"] = max(0, len(pool) - len(final))
    stats["scan_truncated"] = truncated
    if expired():
        deadline_hit = True

    for rank, cand in enumerate(final, start=1):
        best = cand.best()
        signals: dict[str, Any] = {
            "fact_kind": best.kind,
            "n_facts": len(cand.facts),
            "subject_match": int(best.subj_match),
            "fts": int(best.fts_match),
            "lifecycle": (
                _STATE_STATUS_LIFECYCLE.get(best.status, "current")
                if best.kind == "state"
                else "current"
            ),
        }
        if best.term_hits:
            signals["matched_terms"] = sorted(best.term_hits)
        if best.state_key:
            signals["state_key"] = best.state_key
        if best.value_text or best.object_text or best.statement:
            signals["value_text"] = (
                best.value_text or best.object_text or best.statement
            )
        if best.kind == "state":
            signals["status"] = best.status or None
            signals["state_value"] = best.value_text or None
            if best.valid_from_us is not None:
                signals["valid_from_us"] = best.valid_from_us
            if best.valid_to_us is not None:
                signals["valid_to_us"] = best.valid_to_us
            if best.producer:
                signals["producer"] = best.producer
        if best.polarity:
            signals["polarity"] = best.polarity
        if best.strength:
            signals["strength"] = best.strength
        if best.subject:
            signals["subject_canon"] = best.subject
        if best.predicate:
            signals["predicate"] = best.predicate
        if best.kind == "event":
            signals["event_id"] = best.fact_id.split(":", 1)[-1]
        if best.kind == "t2":
            signals["t2_fact"] = best.fact_id
            signals["verified"] = int(best.verified or 0)
            signals["derived"] = 1
            if best.statement:
                signals["statement"] = best.statement
            if best.model_id:
                signals["model_id"] = best.model_id
            _quotes = _coerce_json(best.quotes, None)
            if _quotes is not None:
                signals["quotes"] = _quotes
        # fact pins into retained source bytes (V7-13.06) — the pack layer
        # renders typed lines from these (V7-12.10).
        signals["pins"] = _coerce_json(best.pins, {}) or {}
        # supporting ids of every matched fact on the unit (rerank/pack
        # provenance — the primary fact's fields above stay scalar).
        state_keys = sorted(
            {f.state_key for f in cand.facts if f.kind == "state" and f.state_key}
        )
        if state_keys:
            signals["state_keys"] = state_keys
        event_ids = sorted(
            f.fact_id.split(":", 1)[-1] for f in cand.facts if f.kind == "event"
        )
        if event_ids:
            signals["event_ids"] = event_ids
        t2_ids = sorted(f.fact_id for f in cand.facts if f.kind == "t2")
        if t2_ids:
            signals["t2_fact_ids"] = t2_ids
        # time features for rerank: prefer the fact's own anchor fields,
        # fall back to the unit's occurred interval.
        os_ = best.occurred_start_us
        oe = best.occurred_end_us
        if os_ is None:
            os_ = cand.unit_occurred_start_us
        if oe is None:
            oe = cand.unit_occurred_end_us
        if os_ is not None:
            signals["occurred_start_us"] = os_
        if oe is not None:
            signals["occurred_end_us"] = oe
        if cand.unit_recorded_at_us is not None:
            signals["recorded_at_us"] = cand.unit_recorded_at_us
        if cand.speaker_canon:
            signals["speaker_canon"] = cand.speaker_canon

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
    elif missing:
        # some fact indexes absent — the lane ran on what exists and says
        # so honestly (temporal lane's events_table_missing convention).
        out.status = LaneStatus.PARTIAL
        out.reason = "typed_table_missing"
    return out


class TypedLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
        return lane_typed(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the pipeline
# never imports this module itself — the registrar/importer seam is owned
# by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.TYPED, lane_typed)


__all__ = [
    "LANE_NAME",
    "LANE_VERSION",
    "TYPED_TABLES",
    "TypedLane",
    "lane_typed",
]
