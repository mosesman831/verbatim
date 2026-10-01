"""Advisory update-candidate detection (SPEC_V5 §30.5, V5-30.18–30.21;
worker contract docs/v5_contracts.md §8).

On acceptance, ``detect_update_candidates`` scans prior *live* records in
the same namespace (``source_state.disposition = 'active'``) that share a
subject — entity, identifier, or topic terms — with the newly admitted
record but differ in value, polarity, version, or time. Each pair yields
at most one ``update_candidates`` row, ``state='open'``, carrying a
relation class:

    contradicts   same subject/type, opposite polarity or conflicting value
    newer_value   same key, different version/number/date
    negates       explicit negation of a prior affirmative
    refines       strictly more specific restatement of the same claim

Everything here is advisory (V5-30.19): detection never mutates
lifecycle, ``AddResult.possible_updates`` is a serialized view, and a
candidate resolves only via ``adopt_candidate`` / ``dismiss_candidate``
(or an explicit ``replaces=`` operation elsewhere). Hypothetical, hedged,
quoted, hearsay, and future-plan texts produce NO candidates on either
side (V5-14.05 / E81 adversarial twins), and shared-subject proof is
required — an identifier mention alone never binds two records.

Schema note: ``source_state``/``enrichment``/``update_candidates`` are
the contracts-§3 v5 tables, provisioned by the storage worker's
``schema_v5.py``. Until that lands, this module addresses them directly
against the frozen DDL; ``repos_v5.py`` CRUD can replace the inline SQL
later without changing behavior. A store without the tables raises
``SCHEMA_UNSUPPORTED`` — never a silent empty result.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, replace as _replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..core.time import now_us, rfc3339
from ..core.types import ErrorCode, VerbatimError
from ..memory.types import (
    ENRICHMENT_VERSION,
    MemoryRef,
    MemoryType,
    Polarity,
    UpdateCandidate,
)
from ..storage.repos import has_table
from ..storage import repos_v5
from ..storage.schema_v5 import ensure_additive_tables
from .analyze import (
    Mention,
    _HEDGE_RE,
    _HEARSAY_RE,
    _HYPOTHETICAL_RE,
    _QUOTED_SPAN_RE,
    _enum_value,
    classify_type,
    extract_entities,
    extract_identifiers,
    parse_temporal,
    text_polarity,
)

#: Producer label for this detector — versioned like every deterministic
#: producer, so a rule change is a measurable new version.
UPDATE_DETECTOR_VERSION = "update_candidates/v1"

#: Advisory bound (V5-30.20 recall floor is measured within top-3).
MAX_CANDIDATES = 3

#: Deterministic scan bound over namespace-active priors. Namespaces are
#: personal-scale; the cap bounds worst-case cost while the ORDER BY
#: keeps the evaluated set deterministic regardless of rowid order.
PRIOR_SCAN_LIMIT = 512

#: ``source_state`` dispositions considered *live* for update detection.
#: ``recorded`` (accepted but not admitted) is deliberately excluded —
#: an undeliverable record cannot be contradicted into view (V5-14.09).
_LIVE_DISPOSITIONS = ("active",)

#: Bound on the IN() fan-out while batch-resolving priors — keeps SQLite
#: variable counts and peak payload memory bounded; evaluation order is
#: preserved because chunks are sliced in ORDER BY order.
_RESOLVE_CHUNK = 200

#: Persisted subject-term postings (``update_term_postings``): each
#: projected revision records the exact ``prior.terms`` set the detector
#: would re-derive from its payload, plus one ``_UTP_MARKER`` row whose
#: existence attests the revision is covered. The scan can then prove
#: the ~95% of term-disjoint priors with one indexed lookup instead of
#: resolving every live source — identical candidates, since a prior the
#: index does not return shares no term and ``_relate`` always returns
#: ``None`` for term-disjoint pairs (the belt-and-suspenders recheck is
#: kept regardless). Coverage is proven per scan: a store projected
#: before the table existed — or any partially-projected prior set —
#: falls back to the full resolve, never a pruned guess.
#:
#: Correctness hinges on the persisted terms matching the resolver's:
#: they are computed at write time from the *resolver's own pick* — the
#: lexicographically smallest ``enrich/%`` producer row — and producer
#: versions are monotonically increasing within a build, so that pick is
#: stable for the store's lifetime (a lower-producer row can never be
#: inserted later; a downgrade open refuses on schema_version).
_UTP_TABLE = "update_term_postings"
_UTP_MARKER = ""
_UTP_IN_CHUNK = 200

#: Per-store memo of the *derived* prior view, keyed by
#: ``(source_id, revision, payload_digest, enrichment_digest)``. Revision
#: bytes are content-immutable, so the persisted ``payload_hmac`` is the
#: invalidation: rewritten bytes or a rewritten enrichment row mint a new
#: key and recompute. The memo lives on the Store object — never across
#: profiles — and only memoizes the deterministic re-derivation; it never
#: decides relations and never mutates lifecycle (V5-30.19 preserved).
_RESOLVED_CACHE_ATTR = "_update_detect_resolved_v1"
_RESOLVED_CACHE_MAX = 8192

#: Polarities that never participate in update relations — either side.
#: A hedged/hypothetical/quoted prior is not an asserted current value,
#: and a non-assertive new record asserts nothing about one (V5-14.05).
_NONASSERTIVE = frozenset(
    {
        Polarity.HEDGED.value,
        Polarity.HYPOTHETICAL.value,
        Polarity.QUOTED.value,
    }
)

_RELATIONS = frozenset({"contradicts", "newer_value", "negates", "refines"})

#: Deterministic score model: relation base + shared-subject strength +
#: type agreement. Signals are additive, bounded, and clamped to [0, 1].
_SCORE_BASE = {
    "negates": 0.60,
    "newer_value": 0.60,
    "contradicts": 0.55,
    "refines": 0.45,
}
_SCORE_SHARED_IDENT = 0.15
_SCORE_SHARED_ENTITY = 0.10
_SCORE_TERM_OVERLAP = 0.15  # scaled by Jaccard of content terms
_SCORE_SAME_TYPE = 0.05

#: Enum ``.value`` lookups hoist — ``enum.__get__`` is ~0.3µs and these
#: constants sit inside the per-pair ``_relate`` hot loop.
_POL_NEGATED = Polarity.NEGATED.value
_POL_AFFIRMATIVE = Polarity.AFFIRMATIVE.value
_MT_UNTYPED = MemoryType.UNTYPED.value
_MT_ABSENCE = MemoryType.ABSENCE.value
_NEUTRAL_TYPES = frozenset({_MT_UNTYPED, _MT_ABSENCE})


# ---------------------------------------------------------------------
# record views
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NewRecord:
    """The accepted record under examination (contract §8 shape).

    Carries the enrichment fields produced at acceptance; any field left
    ``None``/empty is recomputed deterministically from ``text`` via the
    §7 seam so tests and early integrations need only supply the source
    identity and bytes.
    """

    source_id: str = ""
    revision: int = 1
    text: Optional[str] = None
    type: Optional[str] = None
    polarity: Optional[str] = None
    identifiers: Tuple[Any, ...] = ()
    entities: Tuple[Any, ...] = ()
    event_at: Optional[str] = None
    anchor_at: Optional[str] = None
    producer: str = ENRICHMENT_VERSION


@dataclass(frozen=True, slots=True)
class _Resolved:
    """Normalized record view used by the relation rules."""

    source_id: str
    revision: int
    text: str
    type: str
    polarity: str
    idents: frozenset  # normalized identifier values
    entities: frozenset  # normalized entity values
    terms: frozenset  # subject/content terms (valueish tokens excluded)
    values: frozenset  # valueish tokens: numeric/date/identifier
    event_at: Optional[str]
    control_version: int = 0
    # Derived-once fields consumed by ``_relate``. ``None`` means "not
    # precomputed — derive from ``text``", so a record built without them
    # (e.g. a test fixture) gets byte-identical legacy behavior.
    nonassertive: Optional[bool] = None
    env: Optional[frozenset] = None


@dataclass(frozen=True)
class DetectedCandidate:
    """One open update candidate — mirrors the ``update_candidates`` row
    plus the serialized ``UpdateCandidate`` view for ``AddResult``."""

    candidate_id: str
    namespace: str
    new_source_id: str
    new_revision: int
    prior_source_id: str
    prior_revision: int
    prior_control_version: int
    relation: str
    score: float
    reason: str
    state: str = "open"
    created_at: str = ""

    def as_update_candidate(self, store_tag: str = "") -> UpdateCandidate:
        """Serialize to the frozen ``UpdateCandidate`` shape (§30.5) — the
        prior record's MemoryRef, relation, reason, score."""
        ref = MemoryRef(
            store_tag=store_tag,
            namespace=self.namespace,
            source_id=self.prior_source_id,
            expected_revision=self.prior_revision,
            control_version=self.prior_control_version,
        )
        return UpdateCandidate(
            ref=ref.to_string(),
            relation=self.relation,
            reason=self.reason,
            score=self.score,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "namespace": self.namespace,
            "new_source_id": self.new_source_id,
            "new_revision": self.new_revision,
            "prior_source_id": self.prior_source_id,
            "prior_revision": self.prior_revision,
            "relation": self.relation,
            "score": self.score,
            "reason": self.reason,
            "state": self.state,
            "created_at": self.created_at,
            "detector": UPDATE_DETECTOR_VERSION,
        }


# ---------------------------------------------------------------------
# record-side language helpers
# ---------------------------------------------------------------------

#: Function words with no subject signal in *records*. Negation markers
#: (no/not/never/without/longer) stay load-bearing — dropping them would
#: invert polarity. Temporal adverbs are also excluded: they carry no
#: subject identity (they are compared as *values* instead).
_RECORD_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have he her hers him his how i if in into is it its me my of on or our
    ours she so such than that the their theirs them then there these they
    this those to too was we were what when where which who whom why will
    with would you your yours about after again against all also am any
    because before being between both each few further here once only other
    out over own same should some under until up very per via
    """.split()
)

#: Time adverbs/expressions — excluded from subject terms; tokens like
#: "tuesday" remain *valueish* (see _valueish) so a weekday change reads
#: as ``newer_value``.
_TEMPORAL_TOKENS = frozenset(
    """
    now today yesterday tomorrow currently recently lately soon
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december morning afternoon evening night tonight daily weekly
    monthly yearly annual quarterly weekend weekday q1 q2 q3 q4
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['_./-][a-z0-9]+)*", re.IGNORECASE)
_NUMERICISH_RE = re.compile(r"\d")

#: Environment/context qualifiers — disjoint contexts mean the two
#: records describe different worlds and cannot update each other
#: (V5-14.05 "environment differences" twin).
_ENV_RE = re.compile(
    r"\b(?:in|on|for|under|within|at)\s+(production|prod|staging|stage"
    r"|development|develop|dev|testing|test|ci|qa|local|sandbox"
    r"|personal|work|home|office|macos|mac|osx|linux|windows|ubuntu"
    r"|debian|android|ios|eu|us|emea|apac|amer|latam)\b",
    re.IGNORECASE,
)

#: Future-intent markers — a plan is not a current-state assertion
#: (V5-14.05 "future plans" twin). Complements the polarity guard:
#: applies even when a caller supplied an ``affirmative`` polarity.
_FUTURE_RE = re.compile(
    r"\b(?:plan(?:s|ned|ning)? to|intend(?:s|ed|ing)? to|hope(?:s|d)? to"
    r"|hoping to|going to|aim(?:s|ing)? to|want(?:s|ed)? to|wish(?:es)?"
    r"|thinking about|considering|contemplating|someday|one day"
    r"|next quarter|next year|soon we|eventually)\b",
    re.IGNORECASE,
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _record_terms(
    text: str, idents: frozenset
) -> Tuple[frozenset, frozenset]:
    """Split content tokens into ``(subject_terms, values)``.

    Valueish tokens — anything carrying a digit, a temporal expression,
    or an extracted identifier — move to ``values`` so a version/date
    change is visible to the differ while subject matching stays on
    stable topic vocabulary.
    """
    terms: set = set()
    values: set = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'._-/")
        if not tok or len(tok) < 2 or tok in _RECORD_STOPWORDS:
            continue
        if _valueish(tok, idents):
            values.add(tok)
        else:
            terms.add(tok)
    return frozenset(terms), frozenset(values)


def _valueish(token: str, idents: frozenset) -> bool:
    """A token that carries a version/number/date/identifier value —
    differing valueish tokens under a shared key read as ``newer_value``
    rather than ``contradicts``."""
    if _NUMERICISH_RE.search(token):
        return True
    if token in _TEMPORAL_TOKENS:
        return True
    if token in idents:
        return True
    return False


def _nonassertive_text(text: str) -> bool:
    """Local twin guard (V5-14.05): hypothetical/hedged/quoted/hearsay/
    future-plan/interrogative text asserts nothing, regardless of any
    supplied label."""
    t = str(text or "")
    if not t.strip():
        return False
    if t.rstrip().endswith("?"):
        return True  # a question is not an assertion
    if _HYPOTHETICAL_RE.search(t):
        return True
    if _FUTURE_RE.search(t):
        return True
    if _HEDGE_RE.search(t):
        return True
    # quoted-majority or hearsay + quotes — mirrors _quoted_majority in
    # analyze.py's fallback but applied unconditionally as a guard.
    total = sum(1 for c in t if c.isalnum())
    quoted = 0
    for m in _QUOTED_SPAN_RE.finditer(t):
        quoted += sum(1 for c in m.group(0) if c.isalnum())
    if total and quoted * 2 >= total and quoted > 0:
        return True
    if quoted and _HEARSAY_RE.search(t):
        return True
    return False


def _env_qualifiers(text: str) -> frozenset:
    return frozenset(m.group(1).lower() for m in _ENV_RE.finditer(text))


# ---------------------------------------------------------------------
# record resolution
# ---------------------------------------------------------------------


def _field_of(rec: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(rec, Mapping) and n in rec:
            return rec[n]
        if hasattr(rec, n):
            val = getattr(rec, n)
            if val is not None:
                return val
    return default


def _mentions_of(raw: Iterable[Any], default_kind: str) -> List[Mention]:
    out: List[Mention] = []
    for item in raw or ():
        if isinstance(item, Mention):
            out.append(item)
            continue
        if isinstance(item, str):
            if item:
                out.append(Mention(default_kind, item, 0, 0))
            continue
        value = _field_of(item, "value", "text", "name", "label")
        if value is None and isinstance(item, (tuple, list)) and len(item) >= 2:
            value = item[1]
        if not value:
            continue
        kind = _field_of(item, "kind", "type")
        if kind is None and isinstance(item, (tuple, list)):
            kind = item[0]
        out.append(Mention(str(kind or default_kind), str(value), 0, 0))
    return out


def _resolve_new(conn: sqlite3.Connection, new_record: Any) -> _Resolved:
    source_id = _field_of(new_record, "source_id")
    if not source_id:
        raise VerbatimError(
            ErrorCode.VALIDATION, "new_record.source_id is required"
        )
    revision = int(_field_of(new_record, "revision", default=1) or 1)
    text = _field_of(new_record, "text", "content", "payload")
    if text is None:
        text = _source_text(conn, source_id, revision)
    if text is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"new_record.text absent and no retained payload for "
            f"{source_id}@{revision}",
        )
    text = str(text)

    idents = _mentions_of(
        _field_of(new_record, "identifiers", default=()), "identifier"
    ) or extract_identifiers(text)
    entities = _mentions_of(
        _field_of(new_record, "entities", default=()), "entity"
    ) or extract_entities(text)
    polarity = _field_of(new_record, "polarity") or text_polarity(text)
    mtype = _field_of(new_record, "type", "memory_type") or classify_type(text)
    event_at = _field_of(new_record, "event_at")
    anchor_at = _field_of(new_record, "anchor_at")
    if event_at is None:
        event_at = parse_temporal(text, anchor_at).event_at

    ident_norm = frozenset(_norm(m.value) for m in idents if m.value)
    terms, values = _record_terms(text, ident_norm)
    return _Resolved(
        source_id=str(source_id),
        revision=revision,
        text=text,
        type=_enum_value(mtype) or MemoryType.UNTYPED.value,
        polarity=_enum_value(polarity) or Polarity.AFFIRMATIVE.value,
        idents=ident_norm,
        entities=frozenset(_norm(m.value) for m in entities if m.value),
        terms=terms,
        values=values,
        event_at=event_at,
        control_version=int(
            _field_of(new_record, "control_version", default=0) or 0
        ),
        nonassertive=_nonassertive_text(text),
        env=_env_qualifiers(text),
    )


def _source_text(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> Optional[str]:
    """Canonical bytes of the live revision; ``None`` when absent or
    undecodable — a purged/corrupt payload contributes nothing rather
    than feeding detection partial bytes."""
    row = conn.execute(
        "SELECT payload FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        return bytes(row[0]).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


def _head_revision(conn: sqlite3.Connection, source_id: str, head: Any) -> int:
    """``source_state.mutation_head`` is the approved head revision id —
    stored TEXT; parse numerically, else fall back to the latest retained
    revision (unknown head → newest retained, never a fabricated id)."""
    try:
        return int(str(head))
    except (TypeError, ValueError):
        row = conn.execute(
            "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def _prior_enrichment(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> Optional[Dict[str, Any]]:
    """Stored ``enrichment`` row when the table exists (schema v5);
    ``None`` → the caller recomputes deterministically from bytes."""
    if not has_table(conn, "enrichment"):
        return None
    row = conn.execute(
        "SELECT type, polarity, event_at, anchor_at, fields_json"
        " FROM enrichment WHERE source_id = ? AND revision = ?"
        " AND producer LIKE 'enrich/%'"
        " ORDER BY producer LIMIT 1",
        (source_id, revision),
    ).fetchone()
    if row is None:
        return None
    try:
        fields = json.loads(row[4]) if row[4] else {}
        if not isinstance(fields, dict):
            fields = {}
    except (ValueError, TypeError):
        fields = {}
    return {
        "type": row[0],
        "polarity": row[1],
        "event_at": row[2],
        "anchor_at": row[3],
        "identifiers": fields.get("identifiers") or (),
        "entities": fields.get("entities") or (),
    }


def _derive_prior(
    source_id: str,
    revision: int,
    text: str,
    enr: Optional[Dict[str, Any]],
    control_version: int,
) -> _Resolved:
    """Deterministic derivation shared by the per-row and batched
    resolvers — ``enr`` is the ``_prior_enrichment``-shaped row (or
    None), everything else falls back to the §7 text extractors."""
    if enr:
        polarity = enr["polarity"] or text_polarity(text)
        mtype = enr["type"] or classify_type(text)
        idents = _mentions_of(enr["identifiers"], "identifier")
        entities = _mentions_of(enr["entities"], "entity")
        event_at = enr["event_at"] or parse_temporal(
            text, enr.get("anchor_at")
        ).event_at
    else:
        polarity = text_polarity(text)
        mtype = classify_type(text)
        idents = extract_identifiers(text)
        entities = extract_entities(text)
        event_at = parse_temporal(text, None).event_at
    if not idents:
        idents = extract_identifiers(text)
    if not entities:
        entities = extract_entities(text)
    ident_norm = frozenset(_norm(m.value) for m in idents if m.value)
    terms, values = _record_terms(text, ident_norm)
    return _Resolved(
        source_id=source_id,
        revision=revision,
        text=text,
        type=_enum_value(mtype) or MemoryType.UNTYPED.value,
        polarity=_enum_value(polarity) or Polarity.AFFIRMATIVE.value,
        idents=ident_norm,
        entities=frozenset(_norm(m.value) for m in entities if m.value),
        terms=terms,
        values=values,
        event_at=event_at,
        control_version=int(control_version or 0),
        nonassertive=_nonassertive_text(text),
        env=_env_qualifiers(text),
    )


def _resolve_prior(
    conn: sqlite3.Connection,
    source_id: str,
    mutation_head: Any,
    control_version: int,
) -> Optional[_Resolved]:
    revision = _head_revision(conn, source_id, mutation_head)
    if revision <= 0:
        return None
    text = _source_text(conn, source_id, revision)
    if not text or not text.strip():
        return None
    enr = _prior_enrichment(conn, source_id, revision)
    return _derive_prior(source_id, revision, text, enr, control_version)


# ---------------------------------------------------------------------
# batched prior resolution
# ---------------------------------------------------------------------
#
# The detector scans up to PRIOR_SCAN_LIMIT live priors per acceptance;
# resolving each one with its own statements is the dominant add-ack and
# drain cost. The batched path below performs the identical derivation
# with two set queries per chunk (payload digests + enrichment rows) plus
# one payload read per cache miss. Skip rules are byte-for-byte the
# ``_resolve_prior`` ones: unparseable head → MAX(revision) fallback,
# absent/NULL/undecodable/blank payload → no candidate, min-producer
# enrichment row wins.


def _resolved_cache(store: Any) -> Optional["OrderedDict"]:
    """Store-scoped resolved-view memo; ``None`` when no store was
    supplied (raw-conn callers still get the batched scan, just no
    memoization)."""
    if store is None:
        return None
    cache = getattr(store, _RESOLVED_CACHE_ATTR, None)
    if cache is None:
        try:
            cache = OrderedDict()
            setattr(store, _RESOLVED_CACHE_ATTR, cache)
        except Exception:
            return None
    return cache


def _chunks(rows: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _enr_view(row: Tuple[Any, ...]) -> Dict[str, Any]:
    """``_prior_enrichment``-shaped dict from a batched row
    ``(type, polarity, event_at, anchor_at, fields_json, producer)``."""
    mtype, pol, event_at, anchor_at, fields_json, _producer = row
    try:
        fields = json.loads(fields_json) if fields_json else {}
        if not isinstance(fields, dict):
            fields = {}
    except (ValueError, TypeError):
        fields = {}
    return {
        "type": mtype,
        "polarity": pol,
        "event_at": event_at,
        "anchor_at": anchor_at,
        "identifiers": fields.get("identifiers") or (),
        "entities": fields.get("entities") or (),
    }


def _payload_text(conn: sqlite3.Connection, source_id: str, revision: int) -> Optional[str]:
    """Lazy payload fetch for cache-miss priors — same acceptance rules
    as ``_source_text``."""
    return _source_text(conn, source_id, revision)


def _resolve_priors(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, Any, Any]],
    *,
    cache: Optional["OrderedDict"],
    enr_table: bool,
    overlap: Optional[frozenset] = None,
) -> List[_Resolved]:
    """Batch equivalent of ``_resolve_prior`` over the ordered
    ``(source_id, mutation_head, control_version)`` rows — identical
    skip rules and derived fields, identical output order.

    ``overlap`` — optional term set the caller will AND-test anyway
    (``new.terms.isdisjoint(prior.terms)`` is ``_relate``'s necessary
    condition, so a disjoint prior can never be a candidate). For
    memoized priors the check runs on the cached term set and skips the
    ``_Resolved`` construction entirely — same emitted list, no object
    churn for the ~95% of namespace rows that share no terms."""
    sids = [r[0] for r in rows]
    marks = ",".join("?" for _ in sids)
    # Payload digests first: the persisted hmac IS the content identity,
    # so a memo hit never needs the payload bytes at all. Revisions and
    # enrichment rows come back in ONE join round-trip — the maps this
    # builds are identical to the old two-query version: orphan
    # enrichment rows (no matching revision) were fetched but never
    # consulted before, and multi-producer rows still collapse to the
    # lexicographically smallest producer below.
    revisions: Dict[str, Dict[int, bytes]] = {}
    enrichment: Dict[Tuple[str, int], Tuple[Tuple[Any, ...], bytes]] = {}
    if enr_table:
        join_sql = (
            f"SELECT r.source_id, r.revision, r.payload_hmac,"
            f" e.type, e.polarity, e.event_at, e.anchor_at, e.fields_json,"
            f" e.producer"
            f" FROM source_revisions r"
            f" LEFT JOIN enrichment e"
            f" ON e.source_id = r.source_id AND e.revision = r.revision"
            f" AND e.producer LIKE 'enrich/%'"
            f" WHERE r.source_id IN ({marks})"
        )
    else:
        join_sql = (
            f"SELECT r.source_id, r.revision, r.payload_hmac,"
            f" NULL, NULL, NULL, NULL, NULL, NULL"
            f" FROM source_revisions r"
            f" WHERE r.source_id IN ({marks})"
        )
    for r in conn.execute(join_sql, sids):
        sid_r, rev_r = r[0], int(r[1])
        revisions.setdefault(sid_r, {})[rev_r] = (
            bytes(r[2]) if r[2] is not None else None
        )
        if enr_table and r[8] is not None:
            key = (sid_r, rev_r)
            enr_t = (r[3], r[4], r[5], r[6], r[7], r[8])
            prev = enrichment.get(key)
            # _prior_enrichment picks ORDER BY producer LIMIT 1 — the
            # lexicographically smallest producer wins. The row tuple
            # itself is the cache-key component — equality is a strictly
            # stronger identity than a digest of its repr, and free.
            if prev is None or enr_t[5] < prev[0][5]:
                enrichment[key] = (enr_t, enr_t)
    out: List[_Resolved] = []
    for source_id, mutation_head, control_version in rows:
        revs = revisions.get(source_id) or {}
        try:
            revision = int(str(mutation_head))
        except (TypeError, ValueError):
            revision = max(revs) if revs else 0
        if revision <= 0:
            continue
        if revision not in revs:
            continue  # no retained revision row — same as _source_text None
        enr_pair = enrichment.get((source_id, revision))
        enr_key = enr_pair[1] if enr_pair is not None else b""
        hmac_key = revs[revision] or b""
        ckey = (source_id, revision, hmac_key, enr_key)
        entry = cache.get(ckey) if cache is not None else None
        if entry is not None:
            cache.move_to_end(ckey)
            if overlap is not None and overlap.isdisjoint(entry.terms):
                continue  # necessary-condition miss — same as caller skip
            cv_i = int(control_version or 0)
            # The cached object is immutable — reused verbatim while the
            # control_version is unchanged (the common case); a bumped cv
            # mints a field-identical copy via ``replace`` — the same
            # fields the old dict-entry rebuild produced.
            out.append(
                entry
                if entry.control_version == cv_i
                else _replace(entry, control_version=cv_i)
            )
            continue
        text = _payload_text(conn, source_id, revision)
        if not text or not text.strip():
            continue
        prior = _derive_prior(
            source_id,
            revision,
            text,
            _enr_view(enr_pair[0]) if enr_pair is not None else None,
            control_version,
        )
        if cache is not None:
            # ``text`` is the one field the cache drops (payload bytes —
            # large, and never read for cached priors: every consumer of
            # a cached entry goes through precomputed fields).
            cache[ckey] = _replace(prior, text="")
            while len(cache) > _RESOLVED_CACHE_MAX:
                cache.popitem(last=False)
        if overlap is not None and overlap.isdisjoint(prior.terms):
            continue  # necessary-condition miss — same as caller skip
        out.append(prior)
    return out


# ---------------------------------------------------------------------
# relation rules
# ---------------------------------------------------------------------


def _relate(
    new: _Resolved, prior: _Resolved,
    ctx: Optional[Tuple[frozenset, frozenset]] = None,
) -> Optional[Tuple[str, float, str]]:
    """The deterministic relation decision. Returns ``(relation, score,
    reason)`` or ``None`` when the pair cannot be an update candidate.

    ``ctx`` — ``(new_all, env_new)`` precomputed once per scan: the
    new-side union set and environment qualifiers are prior-independent,
    so the scan loop does not rebuild them per pair. ``None`` recomputes
    exactly what the monolith always did (test fixtures, raw callers).
    """
    if ctx is None:
        ctx = (
            new.terms | new.values,
            new.env if new.env is not None else _env_qualifiers(new.text),
        )
    new_all, env_new = ctx
    # Both sides must assert. A hedged/hypothetical/quoted prior is not a
    # settled value to update; the same guard covers the new record.
    prior_nonassertive = (
        prior.nonassertive
        if prior.nonassertive is not None
        else _nonassertive_text(prior.text)
    )
    if prior.polarity in _NONASSERTIVE or prior_nonassertive:
        return None

    # Environment/context qualifiers must be compatible — "on macOS" and
    # "on linux" describe different worlds (V5-14.05 twin).
    env_prior = (
        prior.env if prior.env is not None else _env_qualifiers(prior.text)
    )
    if env_new and env_prior and env_new.isdisjoint(env_prior):
        return None

    shared_terms = new.terms & prior.terms
    shared_idents = new.idents & prior.idents
    shared_entities = new.entities & prior.entities

    # Shared-subject proof: ≥2 shared content terms, or ≥1 term plus a
    # shared entity/identifier anchor. A bare identifier mention with no
    # topic overlap never binds (ambiguous-subject / unrelated twins).
    if not shared_terms:
        return None
    if len(shared_terms) < 2 and not (shared_idents or shared_entities):
        return None

    # Type agreement: interpretation labels only (never authority) —
    # ``untyped`` and ``absence`` are neutral (an absence assertion can
    # update any typed claim); two known-but-different real types cannot
    # share a subject.
    known_types = {t for t in (new.type, prior.type)
                   if t not in _NEUTRAL_TYPES}
    if len(known_types) > 1:
        return None

    prior_all = prior.terms | prior.values
    new_only = new_all - prior_all
    prior_only = prior_all - new_all
    neg_flip = (
        new.polarity == _POL_NEGATED
        and prior.polarity == _POL_AFFIRMATIVE
    )
    pos_flip = (
        new.polarity == _POL_AFFIRMATIVE
        and prior.polarity == _POL_NEGATED
    )

    relation: Optional[str] = None
    detail = ""
    if neg_flip:
        # explicit negation of a prior affirmative on the same subject
        relation = "negates"
        detail = "new record negates the prior affirmative"
    elif pos_flip:
        # new affirmative asserts against a prior negation
        relation = "contradicts"
        detail = "opposite polarity on shared subject"
    elif prior_only and prior.terms <= new_all:
        # ``all(_valueish(t, prior.idents) for t in prior_only)`` —
        # valueish classification is the deterministic derive-time
        # predicate that built ``prior.terms``/``prior.values`` over the
        # same ``prior.idents``, so a term reclassified now returns the
        # same verdict it did then. ``prior_only`` draws only from
        # ``prior.terms ∪ prior.values``: every element is valueish iff
        # none comes from ``prior.terms`` — i.e. every prior subject
        # term already sits inside ``new_all`` — one subset test instead
        # of re-running the regex/membership probe per surviving token.
        # same key, different version/number/date — extra new detail does
        # not change that a pinned value moved
        relation = "newer_value"
        detail = "value/version/date moved: " + ", ".join(
            sorted(prior_only)[:3]
        ) + " -> " + ", ".join(sorted(new_only)[:3] or ["(none)"])
    elif prior_only and new_only:
        relation = "contradicts"
        detail = "conflicting value: " + ", ".join(sorted(prior_only)[:3])
    elif new_only and not prior_only:
        relation = "refines"
        detail = "more specific: +" + ", ".join(sorted(new_only)[:3])
    elif not new_only and not prior_only:
        if (new.event_at or "") != (prior.event_at or ""):
            relation = "newer_value"
            detail = "event time differs"
        else:
            return None  # same claim — corroboration, not an update
    else:
        return None  # narrower restatement — less specific, not an update

    overlap = len(shared_terms) / max(1, len(new.terms | prior.terms))
    score = _SCORE_BASE[relation]
    if shared_idents:
        score += _SCORE_SHARED_IDENT
    if shared_entities:
        score += _SCORE_SHARED_ENTITY
    score += _SCORE_TERM_OVERLAP * overlap
    if new.type == prior.type and new.type != _MT_UNTYPED:
        score += _SCORE_SAME_TYPE
    score = round(min(1.0, max(0.0, score)), 4)

    reason = (
        f"shared subject ({', '.join(sorted(shared_terms)[:4])}); {detail}"
    )
    return relation, score, reason


def _candidate_id(
    namespace: str,
    new_source_id: str,
    new_revision: int,
    prior_source_id: str,
    prior_revision: int,
    relation: str,
) -> str:
    """Deterministic id — re-detection of the same pair+relation is an
    idempotent INSERT OR IGNORE, and a resolved candidate never silently
    resurrects under the same id."""
    key = "|".join(
        (
            UPDATE_DETECTOR_VERSION,
            namespace,
            new_source_id,
            str(new_revision),
            prior_source_id,
            str(prior_revision),
            relation,
        )
    )
    return "uc-" + hashlib.blake2b(
        key.encode("utf-8"), digest_size=16
    ).hexdigest()


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------


#: One scored prior tuple — ``(prior_source_id, prior_revision,
#: prior_control_version, relation, score, reason)``. Identity-free on
#: the new side so a pre-commit prescan can run before the new record's
#: ``source_id``/``revision`` are minted inside the transaction.
_ScoreTuple = Tuple[str, int, int, str, float, str]


#: Whole-scan memo — ``(namespace, new-key, exclude_sid, scan_limit,
#: deps_fingerprint) -> sorted scored tuples``. The add path's fused
#: detect and the source-job prescan run the identical scan ~30ms
#: apart; with no interleaving commit the second call is a pure replay,
#: so the change-counter fingerprint replays it for free. Entries are
#: stored only under a *verifiable* fingerprint — when the v5_cv_*
#: triggers are absent the memo stays off and every call scans (never
#: a wrong answer, just the old cost).
_SCAN_MEMO_ATTR = "_update_detect_scan_memo_v1"
_SCAN_MEMO_MAX = 256
_CV_OK_ATTR = "_update_detect_cv_ok_v1"

#: The dependency tables ``_scan_scored`` reads — triggers on all three
#: must exist before the ``cv:*`` counter rows prove anything.
_SCAN_DEPS_TABLES = ("source_state", "source_revisions", "enrichment")


def _scan_memo(store: Any) -> Optional["OrderedDict"]:
    if store is None:
        return None
    memo = getattr(store, _SCAN_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _SCAN_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _scan_deps_fp(conn: sqlite3.Connection, store: Any) -> Optional[tuple]:
    """Change-counter fingerprint over the scan's dependency tables.

    The ``v5_cv_*`` triggers (installed once by the source-job prescan
    path) bump a ``cv:<table>`` meta row on every INSERT/UPDATE/DELETE
    of every dependency table — counter equality proves the scan inputs
    are byte-for-byte unchanged. Verified once per store object through
    ``sqlite_master``: without the triggers there is no fingerprint
    (counters could be silently absent forever), so the memo stays off.
    """
    if store is None:
        return None
    verified = getattr(store, _CV_OK_ATTR, None)
    if not verified:
        try:
            marks = ",".join("?" for _ in _SCAN_DEPS_TABLES)
            row = conn.execute(
                "SELECT COUNT(DISTINCT tbl_name) FROM sqlite_master"
                f" WHERE type = 'trigger' AND tbl_name IN ({marks})"
                " AND name LIKE 'v5\\_cv\\_%' ESCAPE '\\'",
                list(_SCAN_DEPS_TABLES),
            ).fetchone()
            verified = bool(row and int(row[0]) == len(_SCAN_DEPS_TABLES))
        except sqlite3.Error:
            verified = False
        if verified:
            try:
                setattr(store, _CV_OK_ATTR, True)
            except Exception:
                pass
    if not verified:
        return None
    try:
        from ..jobs.source_jobs import _deps_fingerprint
    except Exception:
        return None
    try:
        return _deps_fingerprint(conn, "counter")
    except Exception:
        return None


def _new_scan_key(new: "_Resolved") -> bytes:
    """Content key of the resolved new record — every field the scan
    consumes. ``text`` is folded in only when a fallback could read it
    (``env``/``nonassertive`` unset — the ``_resolve_new`` path always
    precomputes both)."""
    return hashlib.blake2b(
        repr((
            new.source_id, new.revision, new.type, new.polarity,
            new.event_at, new.nonassertive,
            sorted(new.terms), sorted(new.idents),
            sorted(new.entities), sorted(new.values),
            sorted(new.env) if new.env is not None else None,
            new.text if new.env is None or new.nonassertive is None else None,
        )).encode("utf-8"),
        digest_size=16,
    ).digest()


def ensure_term_postings(conn: sqlite3.Connection) -> bool:
    """Create the term-postings table on first use — inside the caller's
    write tx, so the DDL commits atomically with the first rows."""
    if has_table(conn, _UTP_TABLE):
        return True
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS update_term_postings ("
            " namespace TEXT NOT NULL,"
            " term TEXT NOT NULL,"
            " source_id TEXT NOT NULL,"
            " revision INTEGER NOT NULL,"
            " PRIMARY KEY (namespace, term, source_id, revision))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_utp_source"
            " ON update_term_postings(source_id, revision)"
        )
        return True
    except sqlite3.Error:
        return False


def write_term_postings(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    source_id: str,
    revision: int,
    text: str,
) -> None:
    """Persist the resolver-equivalent subject terms for one revision.

    Called inside the projection commit (after ``_write_enrichment``):
    the identifier set comes from the resolver's own pick — the smallest
    ``enrich/%`` producer row — via ``_prior_enrichment``, then
    ``_record_terms`` splits exactly as ``_derive_prior`` does (including
    the extract-on-empty fallback), so the persisted set IS what the
    batch resolver computes for this revision.
    """
    if not ensure_term_postings(conn):
        return
    enr = _prior_enrichment(conn, source_id, revision)
    if enr:
        idents = _mentions_of(enr["identifiers"], "identifier")
    else:
        idents = extract_identifiers(text)
    if not idents:
        idents = extract_identifiers(text)
    ident_norm = frozenset(_norm(m.value) for m in idents if m.value)
    terms, _values = _record_terms(text, ident_norm)
    conn.execute(
        "DELETE FROM update_term_postings"
        " WHERE source_id = ? AND revision = ?",
        (source_id, int(revision)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO update_term_postings"
        " (namespace, term, source_id, revision) VALUES (?, ?, ?, ?)",
        (namespace, _UTP_MARKER, source_id, int(revision)),
    )
    for term in sorted(terms):
        conn.execute(
            "INSERT OR REPLACE INTO update_term_postings"
            " (namespace, term, source_id, revision) VALUES (?, ?, ?, ?)",
            (namespace, term, source_id, int(revision)),
        )


def _utp_prefilter(
    conn: sqlite3.Connection,
    namespace: str,
    new: "_Resolved",
    priors: List[Tuple[str, Any, Any]],
) -> Optional[List[Tuple[str, Any, Any]]]:
    """Term-postings prefilter — the subset of ``priors`` that can share
    a subject term with ``new``, or ``None`` when coverage cannot be
    proven for the whole prior set (uncovered → caller's full scan).

    The index returns every revision whose persisted terms intersect
    ``new.terms``; intersecting that with each prior's *resolved* head
    revision reproduces the disjoint check ``_relate`` performs — priors
    not returned are provably term-disjoint and would have been skipped.

    Head handling mirrors ``_resolve_priors``: a head that parses to a
    revision is proven via that revision's marker; a non-numeric head
    (e.g. the ``unresolved`` sentinel) resolves to ``MAX(revision)``
    downstream, which is unprovable here — those priors are scanned
    unconditionally rather than disabling the index for the namespace.
    Heads parsing to ``<= 0`` are skipped by the resolver outright and
    are simply dropped here.
    """
    if not new.terms or not has_table(conn, _UTP_TABLE):
        return None
    prior_revs: Dict[str, int] = {}
    forced: List[Tuple[str, Any, Any]] = []
    skipped: set = set()
    for row in priors:
        sid, head, _cv = row
        try:
            rev = int(str(head))
        except (TypeError, ValueError):
            forced.append(row)
            continue
        if rev <= 0:
            skipped.add(sid)
            continue
        prior_revs[sid] = rev
    if not prior_revs:
        return forced  # nothing provable — all rows are forced/skipped
    # Coverage probes only the candidate sids through idx_utp_source —
    # O(#priors) point lookups, never an O(#namespace) marker scan.
    covered: set = set()
    sid_list = sorted(prior_revs)
    for chunk in _chunks(sid_list, _UTP_IN_CHUNK):
        marks = ",".join("?" for _ in chunk)
        for sid, rev in conn.execute(
            "SELECT source_id, revision FROM update_term_postings"
            f" WHERE namespace = ? AND term = ?"
            f" AND source_id IN ({marks})",
            [namespace, _UTP_MARKER, *chunk],
        ):
            covered.add((sid, int(rev)))
    if not all((sid, rev) in covered for sid, rev in prior_revs.items()):
        return None
    hits: set = set()
    term_list = sorted(new.terms)
    for chunk in _chunks(term_list, _UTP_IN_CHUNK):
        marks = ",".join("?" for _ in chunk)
        for sid, rev in conn.execute(
            "SELECT source_id, revision FROM update_term_postings"
            f" WHERE namespace = ? AND term IN ({marks})",
            [namespace, *chunk],
        ):
            hits.add((sid, int(rev)))
    hit_sids = {
        sid for sid, rev in hits if prior_revs.get(sid) == rev
    }
    return [
        row
        for row in priors
        if row[0] in hit_sids or (row[0] not in prior_revs and row[0] not in skipped)
    ]


def _scan_scored(
    conn: sqlite3.Connection,
    namespace: str,
    new: "_Resolved",
    *,
    exclude_sid: Optional[str],
    scan_limit: int,
    store: Any,
) -> List[_ScoreTuple]:
    """Live-prior scan + resolution + relation verdicts — the shared
    core of ``plan_update_candidates`` and the pre-commit prescan.
    ``exclude_sid=None`` prescans without the ``source_id <> new``
    predicate (the not-yet-minted id is dropped at materialization —
    see ``plan_update_scores``); the caller then widens ``scan_limit``
    by one so the fused path's window is covered exactly."""
    memo = _scan_memo(store)
    fp = _scan_deps_fp(conn, store) if memo is not None else None
    key = None
    if memo is not None and fp is not None:
        key = (
            namespace, _new_scan_key(new), exclude_sid,
            int(scan_limit), fp,
        )
        ent = memo.get(key)
        if ent is not None:
            memo.move_to_end(key)
            return list(ent)

    marks = ",".join("?" for _ in _LIVE_DISPOSITIONS)
    excl = " AND source_id <> ?" if exclude_sid is not None else ""
    params: List[Any] = [namespace, *_LIVE_DISPOSITIONS]
    if exclude_sid is not None:
        params.append(exclude_sid)
    params.append(int(scan_limit))
    priors = conn.execute(
        f"SELECT source_id, mutation_head, control_version"
        f" FROM source_state"
        f" WHERE namespace = ? AND disposition IN ({marks}){excl}"
        f" ORDER BY source_id LIMIT ?",
        params,
    ).fetchall()

    scored: List[_ScoreTuple] = []
    cache = _resolved_cache(store)
    enr_table = has_table(conn, "enrichment")
    ctx = (
        new.terms | new.values,
        new.env if new.env is not None else _env_qualifiers(new.text),
    )
    # Term-postings prefilter: when every live prior's head revision is
    # covered, the ~95% of term-disjoint priors are proven by one indexed
    # lookup — they resolve + relate to nothing, identical to the full
    # scan's outcome. Uncovered stores keep the full window.
    filtered = _utp_prefilter(conn, namespace, new, priors)
    scan_rows = filtered if filtered is not None else priors
    for chunk in _chunks(list(scan_rows), _RESOLVE_CHUNK):
        for prior in _resolve_priors(
            conn, chunk, cache=cache, enr_table=enr_table,
            overlap=new.terms,
        ):
            # Necessary-condition prefilter: ``_relate`` returns ``None``
            # whenever ``shared_terms`` is empty, so a term-disjoint
            # prior can never be a candidate — the check is ~6× cheaper
            # than the full verdict and changes no outcome. Kept as a
            # belt-and-suspenders recheck; ``overlap`` already applies it
            # inside resolution so disjoint priors never materialize.
            if new.terms.isdisjoint(prior.terms):
                continue
            verdict = _relate(new, prior, ctx)
            if verdict is None:
                continue
            relation, score, reason = verdict
            scored.append(
                (
                    prior.source_id,
                    prior.revision,
                    prior.control_version,
                    relation,
                    score,
                    reason,
                )
            )
    scored.sort(key=lambda t: (-t[4], t[0], t[1], t[3]))
    if memo is not None and key is not None:
        memo[key] = tuple(scored)
        while len(memo) > _SCAN_MEMO_MAX:
            memo.popitem(last=False)
    return scored


def _guard_new(new: "_Resolved") -> bool:
    """The twin guards ``plan_update_candidates`` applies before the
    scan — hedged/nonassertive/term-less new records yield no
    candidates (V5-14.05)."""
    if new.polarity in _NONASSERTIVE or _nonassertive_text(new.text):
        return False
    if not new.terms:
        return False
    return True


def _require_detector_schema(conn: sqlite3.Connection) -> None:
    if not has_table(conn, "source_state") or not has_table(
        conn, "update_candidates"
    ):
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            "update-candidate detection needs the v5 source_state/"
            "update_candidates tables (schema_v5 not provisioned)",
        )


def materialize_update_candidates(
    namespace: str,
    new_source_id: str,
    new_revision: int,
    scored: Iterable[_ScoreTuple],
    *,
    created_at: Optional[str] = None,
) -> List[DetectedCandidate]:
    """Mint ``DetectedCandidate`` rows from scored prior tuples —
    deterministic ``candidate_id``s, ``state='open'``, top
    ``MAX_CANDIDATES`` by the sort order ``_scan_scored`` already
    applied. Tuples naming ``new_source_id`` as their own prior are
    skipped: the fused path's ``source_id <> new`` scan predicate can
    never produce a self-candidate, so a prescan that saw the source's
    committed row drops exactly those tuples (and compensates with a
    one-wider scan window — see ``plan_update_scores``)."""
    stamp = created_at or rfc3339(now_us())
    out: List[DetectedCandidate] = []
    for psid, prev, pcv, relation, score, reason in scored:
        if psid == new_source_id:
            continue
        out.append(
            DetectedCandidate(
                candidate_id=_candidate_id(
                    namespace, new_source_id, new_revision,
                    psid, prev, relation,
                ),
                namespace=namespace,
                new_source_id=new_source_id,
                new_revision=int(new_revision),
                prior_source_id=psid,
                prior_revision=prev,
                prior_control_version=pcv,
                relation=relation,
                score=score,
                reason=reason,
                state="open",
                created_at=stamp,
            )
        )
        if len(out) >= MAX_CANDIDATES:
            break
    return out


def plan_update_candidates(
    conn: sqlite3.Connection,
    namespace: str,
    new_record: Any,
    *,
    created_at: Optional[str] = None,
    store: Any = None,
) -> List[DetectedCandidate]:
    """Scan/score phase of ``detect_update_candidates`` — the same
    candidate selection with no writes, so the namespace scan can run
    on a read snapshot ahead of the committing transaction. Returns
    the top-``MAX_CANDIDATES`` candidates, highest score first — the
    exact list the monolith persists.

    Requires the contracts-§3 v5 tables; ``SCHEMA_UNSUPPORTED`` when the
    store predates schema v5 — never a silent empty answer.
    """
    _require_detector_schema(conn)
    if not namespace:
        raise VerbatimError(ErrorCode.VALIDATION, "namespace is required")

    new = _resolve_new(conn, new_record)

    # Twin guard (V5-14.05): hedged/hypothetical/quoted/future-plan text
    # asserts nothing — no candidates, on the new record's side first.
    if not _guard_new(new):
        return []

    scored = _scan_scored(
        conn,
        namespace,
        new,
        exclude_sid=new.source_id,
        scan_limit=PRIOR_SCAN_LIMIT,
        store=store,
    )
    stamp = created_at or rfc3339(now_us())
    return [
        DetectedCandidate(
            candidate_id=_candidate_id(
                namespace,
                new.source_id,
                new.revision,
                psid,
                prev,
                relation,
            ),
            namespace=namespace,
            new_source_id=new.source_id,
            new_revision=new.revision,
            prior_source_id=psid,
            prior_revision=prev,
            prior_control_version=pcv,
            relation=relation,
            score=score,
            reason=reason,
            state="open",
            created_at=stamp,
        )
        for psid, prev, pcv, relation, score, reason in scored[:MAX_CANDIDATES]
    ]


def plan_update_scores(
    conn: sqlite3.Connection,
    namespace: str,
    new_record: Any,
    *,
    store: Any = None,
) -> Tuple["_Resolved", List[_ScoreTuple]]:
    """Pre-commit prescan twin of ``plan_update_candidates``: identical
    validation, guards, resolution, and verdicts, but the new record's
    ``source_id`` may be unminted — the prior scan runs *without* the
    ``source_id <> new`` predicate and one row wider
    (``PRIOR_SCAN_LIMIT + 1``), so its scored tuples cover the fused
    path's window exactly:

    ordered live priors minus the new row's first ``PRIOR_SCAN_LIMIT``
    entries are always a subset of the full ordering's first
    ``PRIOR_SCAN_LIMIT + 1`` entries. Tuples whose prior IS the new
    source (only possible when its state row already committed — a
    dedup-replay add) are dropped by ``materialize_update_candidates``,
    reproducing the fused ``source_id <> new`` set.

    Returns ``(resolved_new, scored_tuples)``; the caller mints and
    persists inside the commit transaction via
    ``materialize_update_candidates`` + ``persist_update_candidates``.
    """
    _require_detector_schema(conn)
    if not namespace:
        raise VerbatimError(ErrorCode.VALIDATION, "namespace is required")
    new = _resolve_new(conn, new_record)
    if not _guard_new(new):
        return new, []
    scored = _scan_scored(
        conn,
        namespace,
        new,
        exclude_sid=None,
        scan_limit=PRIOR_SCAN_LIMIT + 1,
        store=store,
    )
    return new, scored


def persist_update_candidates(
    conn: sqlite3.Connection, cands: Iterable[DetectedCandidate]
) -> List[DetectedCandidate]:
    """Write phase of ``detect_update_candidates`` — the deterministic-id
    INSERTs, inside the caller's transaction. Returns the candidate
    list it was given (the monolith's return shape)."""
    out = list(cands)
    for cand in out:
        # Idempotent write: the deterministic candidate_id makes
        # re-detection a no-op, and an already-resolved row is never
        # resurrected (INSERT OR REPLACE would — explicitly avoided).
        if (
            repos_v5.get(
                conn,
                "update_candidates",
                {"candidate_id": cand.candidate_id},
            )
            is None
        ):
            repos_v5.insert(
                conn,
                "update_candidates",
                {
                    "candidate_id": cand.candidate_id,
                    "namespace": cand.namespace,
                    "new_source_id": cand.new_source_id,
                    "new_revision": cand.new_revision,
                    "prior_source_id": cand.prior_source_id,
                    "prior_revision": cand.prior_revision,
                    "relation": cand.relation,
                    "score": cand.score,
                    "state": cand.state,
                    "created_at": cand.created_at,
                },
            )
    return out


def detect_update_candidates(
    conn: sqlite3.Connection,
    namespace: str,
    new_record: Any,
    *,
    created_at: Optional[str] = None,
    store: Any = None,
) -> List[DetectedCandidate]:
    """Find prior live records that ``new_record`` may update, and persist
    them as ``update_candidates`` rows with ``state='open'``.

    Advisory only (V5-30.19): nothing about either record's lifecycle
    changes. Bounded to the top ``MAX_CANDIDATES`` by score
    (V5-30.20's recall floor is measured within top-3). Returns the
    written candidates, highest score first.

    The persist path is the first-writer-creates point for the V6
    additive objects: ``idx_source_state_ns_disp_sid`` must exist
    before the prior scan reads ``source_state`` on a fresh
    ``Store.create`` (the lazy ensure otherwise lands only on
    exposure/attestation writes or ``apply()`` — V6-02.16). The v5
    schema check runs first — a pre-v5 store gets the typed
    ``SCHEMA_UNSUPPORTED``, not a raw CREATE INDEX failure.
    """
    _require_detector_schema(conn)
    ensure_additive_tables(conn)
    return persist_update_candidates(
        conn,
        plan_update_candidates(
            conn,
            namespace,
            new_record,
            created_at=created_at,
            store=store,
        ),
    )


def list_open_candidates(
    conn: sqlite3.Connection,
    namespace: str,
    *,
    source_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Unresolved candidates feeding §13 completeness (V5-30.21): a
    current-state read that delivers the prior record while an open
    ``contradicts``/``newer_value`` candidate exists must surface or
    label it. ``source_id`` filters on either side of the pair."""
    if not has_table(conn, "update_candidates"):
        return []
    if source_id is None:
        rows = repos_v5.query(
            conn,
            "update_candidates",
            {"namespace": namespace, "state": "open"},
            order="score DESC",
            limit=int(limit),
        )
        # deterministic tie-break under equal scores
        rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        return rows
    # endpoint filter needs OR — beyond the repo's equality-only shape
    cur = conn.execute(
        "SELECT candidate_id, namespace, new_source_id, new_revision,"
        " prior_source_id, prior_revision, relation, score, state,"
        " created_at FROM update_candidates"
        " WHERE namespace = ? AND state = 'open'"
        " AND (new_source_id = ? OR prior_source_id = ?)"
        " ORDER BY score DESC, candidate_id LIMIT ?",
        (namespace, source_id, source_id, int(limit)),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _resolve_candidate(
    conn: sqlite3.Connection, candidate_id: str, target: str
) -> Dict[str, Any]:
    """open → target transition; idempotent on repeat, conflict on the
    opposite terminal state. Advisory rows never reopen silently."""
    if not candidate_id:
        raise VerbatimError(ErrorCode.VALIDATION, "candidate_id required")
    if not has_table(conn, "update_candidates"):
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            "update_candidates table absent (schema_v5 not provisioned)",
        )
    row = repos_v5.get(
        conn, "update_candidates", {"candidate_id": candidate_id}
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN,
            f"unknown update candidate {candidate_id!r}",
        )
    if row["state"] == target:
        return row
    if row["state"] != "open":
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"candidate {candidate_id!r} is {row['state']!r}, not open",
        )
    repos_v5.update(
        conn,
        "update_candidates",
        {"state": target},
        {"candidate_id": candidate_id},
    )
    return repos_v5.get(
        conn, "update_candidates", {"candidate_id": candidate_id}
    )


def adopt_candidate(
    conn: sqlite3.Connection, candidate_id: str
) -> Dict[str, Any]:
    """Mark an open candidate adopted — the owner (or an explicit
    ``replaces=`` path elsewhere) accepted the relation. Still no
    lifecycle mutation here; adoption is bookkeeping on the advisory row.
    """
    return _resolve_candidate(conn, candidate_id, "adopted")


def dismiss_candidate(
    conn: sqlite3.Connection, candidate_id: str
) -> Dict[str, Any]:
    """Mark an open candidate dismissed — reviewed and rejected. A
    dismissed id never reopens; a genuinely new pair mints a new id via
    the revision-bearing digest."""
    return _resolve_candidate(conn, candidate_id, "dismissed")


def possible_updates(
    candidates: Iterable[DetectedCandidate], store_tag: str = ""
) -> List[UpdateCandidate]:
    """Serialize detected candidates to ``AddResult.possible_updates``
    shape (frozen ``UpdateCandidate`` in ``memory/types.py``)."""
    return [c.as_update_candidate(store_tag) for c in candidates]
