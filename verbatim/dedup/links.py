"""Deduplication by linking (SPEC_V5 §30.2; docs/v5_contracts.md §9).

Duplicate groups are *links*, never merges: every record keeps its own
bytes, attribution, and receipt (V5-30.05), and nothing here deletes or
overwrites a source (V5-30.03). Links land in the ``duplicate_links``
projection table (contracts §3) carrying method and score; group identity
is the earliest live member's ``source_id``.

Link methods (contract allowlist):

- ``exact_digest`` — byte-identical content detected via the
  profile-keyed payload HMAC (``source_revisions.payload_hmac``). Score
  1.0. Exact links are unconditional under V5-30.05.
- ``normalized`` — identical ``norm/v1``-folded text detected via the
  ``source_lexical_projection.digest`` column. Because normalization can
  erase polarity/type cues (quotes, casing), normalized links pass
  through the same V5-30.07 hard guards as near links.
- ``minhash`` — Jaccard similarity over ``shingle_signature`` sets
  recomputed from ``source_lexical_projection.tokens`` (the persisted
  normalized token stream — signatures stay recomputable, never a second
  fingerprint store), above a declared threshold, gated by the V5-30.07
  guards.

Hard guards (V5-30.07): a near/normalized link MUST NOT join records
that differ in polarity, identifier sets, version/number tokens,
explicit time expressions, or ``type``. Guard evaluation runs after the
similarity threshold and vetoes regardless of score: "deploy-v1" vs
"deploy-v2" is never a duplicate; "I like X" vs "I no longer like X" is
never a duplicate. Fields missing on *both* sides are vacuously equal
and flagged ``guards_vacuous``; a field present on only one side vetoes
(fail closed).

Deletion closure (V5-30.08): links are derived data — forgetting one
member never forgets independently submitted siblings. ``drop_member``
is the closure hook: it removes the forgotten member's rows and
re-anchors the group to the new earliest live member (a group reduced
to a single row dissolves — one member is not a duplicate set). Callers
that erase without ``drop_member`` are still handled: liveness is
checked per member (missing source/revision row, emptied payload, or
``source_state.disposition='erased'`` all count as not-live), so stale
rows can never resurrect a forgotten member as representative or
corroboration.

Dependency note (parallel build): ``verbatim.enrichment`` and
``verbatim.storage.repos_v5`` are owned by other workers and were
absent at authoring time. Both are imported behind guards — when they
land they are preferred automatically; until then deterministic
``norm/v1``-compatible fallbacks (documented below) keep the guards and
signatures honest. There is no behavioral seam: candidate signatures
are always produced by calling the *same* ``shingle_signature`` symbol
the caller used, so link math cannot fork between implementations.

Namespace note: ``duplicate_links`` carries no namespace column, so
namespace scoping is enforced at candidate-scan time. The authoritative
source→namespace map is ``source_state.namespace`` (contracts §3); on
stores without the v5 control table the scan falls back to
``sources.scope_id = <namespace>`` — i.e. callers on pre-v5 stores pass
the scope partition token as ``namespace``. Where the facade namespace
and the v3 scope partition differ, the caller must pass the storage
partition token (see module report; contract ambiguity noted).

Per-namespace ``dedupe`` policy (V5-30.08): ``link`` (default) or
``none``. There is no merge/delete mode. The policy is resolved through
:func:`get_dedupe_policy`; the durable backing store is a ``meta`` row
``dedupe_policy:<namespace>`` written by :func:`set_dedupe_policy`
(no dedicated policy table exists in the contract schema — the meta
seam keeps the value durable, per-namespace, and swappable by whichever
worker owns facade policy persistence). Callers may also pass the
resolved policy explicitly via the ``policy`` kwarg.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..core.time import now_us, rfc3339
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from ..memory.types import ENRICHMENT_VERSION
from ..storage.repos import has_table

# ---------------------------------------------------------------------------
# Parallel-worker dependencies (contracts §3, §7) — guarded imports.
# ---------------------------------------------------------------------------

try:  # storage worker owns verbatim/storage/repos_v5.py (contracts §3)
    from ..storage import repos_v5 as _repos_v5  # type: ignore
except Exception:  # pragma: no cover - absent during parallel build
    _repos_v5 = None

try:  # enrichment worker owns verbatim/enrichment/ (contracts §7)
    from ..enrichment import normalize_text as _enr_normalize_text
except Exception:  # pragma: no cover - absent during parallel build
    _enr_normalize_text = None
try:
    from ..enrichment import extract_identifiers as _enr_extract_identifiers
except Exception:  # pragma: no cover
    _enr_extract_identifiers = None
try:
    from ..enrichment import polarity as _enr_polarity
except Exception:  # pragma: no cover
    _enr_polarity = None
try:
    from ..enrichment import classify_type as _enr_classify_type
except Exception:  # pragma: no cover
    _enr_classify_type = None
try:
    from ..enrichment import shingle_signature as _enr_shingle_signature
except Exception:  # pragma: no cover
    _enr_shingle_signature = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_EXACT = "exact_digest"
METHOD_NORMALIZED = "normalized"
METHOD_MINHASH = "minhash"
LINK_METHODS = frozenset({METHOD_EXACT, METHOD_NORMALIZED, METHOD_MINHASH})

#: V5-30.06 default declared near-duplicate threshold (configurable per call).
DEFAULT_THRESHOLD = 0.85
#: V5-30.06 requires a bounded candidate scan: namespace-scoped, most recent N.
DEFAULT_SCAN_LIMIT = 500

#: Batch size for the candidate liveness/enrichment lookups — keeps the
#: SQLite variable limit comfortably out of reach while collapsing the
#: per-candidate query rounds into one per chunk.
_BATCH = 200

#: Store-scoped shingle-signature memo. ``source_lexical_projection``
#: rows are keyed (source_id, revision) and the persisted ``digest`` is a
#: content address over the normalized text the tokens were folded from
#: — same digest ⇒ same tokens ⇒ same signature — so the digest alone is
#: a sound key even across sources (exact duplicates share the entry).
#: Bounded FIFO; advisory only, never an authority.
_SIG_MEMO_ATTR = "_dedup_sig_memo_v1"
_SIG_MEMO_MAX = 4096

POLICY_LINK = "link"
POLICY_NONE = "none"
_DEDUPE_POLICIES = frozenset({POLICY_LINK, POLICY_NONE})
_POLICY_META_PREFIX = "dedupe_policy:"

#: Dispositions (source_state/v1, §14.3) whose member is forgotten/erased —
#: not live for linking, anchoring, or corroboration.
_DEAD_DISPOSITIONS = frozenset({"erased"})
#: Retained-but-depreferred dispositions for representative selection
#: (V5-30.09 collapse wants the current record, not historical evidence).
_DEPREFERRED_DISPOSITIONS = frozenset(
    {"superseded", "corrected", "retracted", "archived"}
)

_SHINGLE_LEN = 3
_POLICY_VERSION = "dedupe/v1"


# ---------------------------------------------------------------------------
# Public result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupFields:
    """Guard-relevant enrichment fields for one record (V5-30.07).

    Produced from raw text via :func:`fields_from_text`, from an
    ``enrichment`` table row via :func:`fields_from_enrichment`, or
    supplied directly by the caller (``text_fields`` argument).

    All comparison values are norm/v1-folded at construction
    (``__post_init__``): the guard compares identifier *referents*, not
    incidental casing/quoting — "deploy-v1" vs "deploy-v2" still vetoes,
    while "Deploy V1." vs "deploy v1" does not produce a phantom veto
    just because one side's fields were recomputed from the normalized
    token projection. Byte-exact identifier case remains preserved in
    ``entity_postings`` (V5-30.17); only this guard's comparison is
    folded.
    """

    polarity: str = ""
    memory_type: str = ""
    identifiers: frozenset = frozenset()      # extract_identifiers values
    number_tokens: frozenset = frozenset()    # digit-bearing tokens (versions, counts)
    time_expressions: frozenset = frozenset()  # normalized explicit time exprs

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "polarity", _norm(str(self.polarity)))
        object.__setattr__(
            self, "memory_type", _norm(str(self.memory_type)))
        object.__setattr__(
            self, "identifiers",
            frozenset(
                v for v in
                (_norm(str(i)) for i in self.identifiers) if v))
        object.__setattr__(
            self, "number_tokens",
            frozenset(
                v for v in
                (_norm(str(n)) for n in self.number_tokens)
                if v))
        object.__setattr__(
            self, "time_expressions",
            frozenset(
                v for v in
                (_norm(str(t)) for t in self.time_expressions)
                if v))

    def vacuous(self) -> bool:
        return not (
            self.polarity
            or self.memory_type
            or self.identifiers
            or self.number_tokens
            or self.time_expressions
        )


@dataclass
class LinkOutcome:
    """Result of a link attempt; serializable for job receipts."""

    linked: bool = False
    group_id: Optional[str] = None
    method: Optional[str] = None
    score: float = 0.0
    linked_to: Optional[tuple] = None      # (source_id, revision) matched
    reason: str = "no_match"               # ok|no_match|policy_none|guard_veto|no_signature|no_index
    vetoes: list = field(default_factory=list)  # [{source_id, revision, dimension, score}]
    scanned: int = 0
    guards_vacuous: bool = False

    def to_dict(self) -> dict:
        return {
            "linked": self.linked,
            "group_id": self.group_id,
            "method": self.method,
            "score": self.score,
            "linked_to": list(self.linked_to) if self.linked_to else None,
            "reason": self.reason,
            "vetoes": self.vetoes,
            "scanned": self.scanned,
            "guards_vacuous": self.guards_vacuous,
        }


# ---------------------------------------------------------------------------
# Enrichment seam — prefer verbatim.enrichment (contracts §7), else local
# deterministic norm/v1-compatible fallbacks. Fallbacks are pure stdlib,
# version-pinned, and recomputable — the same contract enrichment owns.
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    if _enr_normalize_text is not None:
        return _enr_normalize_text(text)
    return _fb_normalize_text(text)


def _identifiers(text: str) -> list:
    if _enr_extract_identifiers is not None:
        return list(_enr_extract_identifiers(text))
    return _fb_extract_identifiers(text)


def _polarity_of(text: str) -> str:
    if _enr_polarity is not None:
        p = _enr_polarity(text)
        return getattr(p, "value", p) or ""
    return _fb_polarity(text)


def _type_of(text: str) -> str:
    if _enr_classify_type is not None:
        t = _enr_classify_type(text)
        return getattr(t, "value", t) or ""
    return _fb_classify_type(text)


def _signature(tokens: list) -> frozenset:
    if _enr_shingle_signature is not None:
        return frozenset(_enr_shingle_signature(list(tokens)))
    return _fb_shingle_signature(list(tokens))


_PUNCT_CATEGORIES = ("P", "S")  # punctuation + symbols fold to spaces


def _fb_normalize_text(text: str) -> str:
    """norm/v1-compatible fold: NFKC, casefold, punctuation→space, collapse."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    chars = (
        " " if unicodedata.category(ch)[0] in _PUNCT_CATEGORIES else ch
        for ch in folded
    )
    return " ".join("".join(chars).split())


_FB_IDENT_PATTERNS = (
    ("url", re.compile(r"https?://[^\s\"'<>]+")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ticket", re.compile(r"\b[A-Z]{2,}-\d+\b")),
    ("hash", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("version", re.compile(r"\bv?\d+(?:\.\d+)+(?:-[0-9A-Za-z.]+)?\b")),
    ("version", re.compile(r"\bv\d+\b")),
    ("handle", re.compile(r"(?<![\w.])@[A-Za-z0-9_]{2,}\b")),
    ("path", re.compile(r"(?<![\w.])(?:/[\w.@+-]+){2,}/?")),
    ("code", re.compile(r"`[^`\n]+`")),
    ("quoted", re.compile(r"\"[^\"\n]+\"|'[^'\n]+'")),
)


def _fb_extract_identifiers(text: str) -> list:
    """Contract-shaped Identifier tuples (kind, value, start, end)."""
    out: list[tuple[str, str, int, int]] = []
    for kind, rx in _FB_IDENT_PATTERNS:
        for m in rx.finditer(text):
            out.append((kind, m.group(0), m.start(), m.end()))
    out.sort(key=lambda t: (t[2], t[0]))
    return out


_NEGATION = re.compile(
    r"\b(?:no longer|not|never|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|no one|nobody|nothing|without|nor|"
    r"n't)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:maybe|perhaps|probably|possibly|likely|i think|i guess|seems|"
    r"appears|reportedly|allegedly|sort of|kind of)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL = re.compile(
    r"\b(?:if|would|could|might|imagine|suppose|hypothetically|"
    r"in theory|let's say)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"\"[^\"\n]+\"|'[^'\n]{3,}'|`[^`\n]+`")


def _fb_polarity(text: str) -> str:
    if _QUOTED.search(text):
        return "quoted"
    if _NEGATION.search(text):
        return "negated"
    if _HYPOTHETICAL.search(text):
        return "hypothetical"
    if _HEDGE.search(text):
        return "hedged"
    return "affirmative"


_TYPE_MARKERS = (
    ("preference", re.compile(
        r"\b(?:i like|i prefer|i love|i hate|i dislike|my favorite|"
        r"i'd rather|i want)\b", re.IGNORECASE)),
    ("decision", re.compile(
        r"\b(?:we decided|i decided|decision:|let's use|we chose|"
        r"i chose|we agreed|agreed to|decided to)\b", re.IGNORECASE)),
    ("plan", re.compile(
        r"\b(?:plan to|planning to|we will|i will|todo|next step|"
        r"going to|intend to)\b", re.IGNORECASE)),
    ("event", re.compile(
        r"\b(?:yesterday|today|last week|deployed|happened|occurred|"
        r"on \d{4}-\d{2}-\d{2})\b", re.IGNORECASE)),
)


def _fb_classify_type(text: str) -> str:
    for memory_type, rx in _TYPE_MARKERS:
        if rx.search(text):
            return memory_type
    return "untyped"


def _fb_shingle_signature(tokens: list) -> frozenset:
    """blake2b-hashed contiguous k=3 token shingles; short docs → whole."""
    toks = [t for t in tokens if t]
    if not toks:
        return frozenset()
    if len(toks) <= _SHINGLE_LEN:
        grams = [tuple(toks)]
    else:
        grams = [tuple(toks[i:i + _SHINGLE_LEN])
                 for i in range(len(toks) - _SHINGLE_LEN + 1)]
    return frozenset(
        hashlib.blake2b(
            "\x00".join(g).encode("utf-8"),
            digest_size=8,
            person=b"vbdup1",
        ).digest()
        for g in grams
    )


_NUMBER_TOKEN = re.compile(r"\S*\d\S*")

_TIME_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|"
               r"august|september|october|november|december)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:today|yesterday|tomorrow|tonight)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:last|next|this)\s+(?:week|month|year|monday|tuesday|"
        r"wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|"
        r"sunday)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
               re.IGNORECASE),
    re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+ago\b",
               re.IGNORECASE),
    re.compile(r"\bsince\s+(?:january|february|march|april|may|june|july|"
               r"august|september|october|november|december|\d{4})\b",
               re.IGNORECASE),
)


def _fb_time_expressions(text: str) -> frozenset:
    """Normalized explicit time-expression surface forms (guard dimension)."""
    out = set()
    for rx in _TIME_PATTERNS:
        for m in rx.finditer(text):
            out.add(_fb_normalize_text(m.group(0)))
    return frozenset(out)


def _number_tokens(normalized_text: str) -> frozenset:
    return frozenset(
        t for t in normalized_text.split() if _NUMBER_TOKEN.match(t)
    )


# ---------------------------------------------------------------------------
# Field extraction — raw text, enrichment rows, caller dicts
# ---------------------------------------------------------------------------


def fields_from_text(text: str) -> DedupFields:
    """Extract V5-30.07 guard fields from source text (enrichment seam)."""
    norm = _norm(text)
    ids = frozenset(
        str(getattr(i, "value", i[1] if isinstance(i, tuple) else i))
        for i in _identifiers(text)
    )
    return DedupFields(
        polarity=_polarity_of(text),
        memory_type=_type_of(text),
        identifiers=ids,
        number_tokens=_number_tokens(norm),
        time_expressions=_fb_time_expressions(text),
    )


def _values(items: Any) -> frozenset:
    """Tolerantly extract identifier/expression values from fields_json
    lists: items may be strings, dicts (value/text/entity), or tuples."""
    out = set()
    if not isinstance(items, (list, tuple)):
        return frozenset()
    for it in items:
        if isinstance(it, str):
            out.add(it)
        elif isinstance(it, dict):
            for key in ("value", "text", "entity", "expression"):
                v = it.get(key)
                if isinstance(v, str) and v:
                    out.add(v)
                    break
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            out.add(str(it[1]))
    return frozenset(out)


def fields_from_enrichment(row: dict) -> DedupFields:
    """Guard fields from an ``enrichment`` table row (contracts §3).

    ``type``/``polarity`` are columns; identifiers/entities/time exprs
    live in ``fields_json``. Key shapes are read tolerantly — the
    enrichment worker owns the exact layout; anything absent stays empty
    and is filled from the token projection by the caller when possible.
    """
    fj = safe_json_loads(row.get("fields_json") or "{}")
    if not isinstance(fj, dict):
        fj = {}
    identifiers = _values(fj.get("identifiers"))
    numbers = _values(fj.get("number_tokens")) | _values(fj.get("numbers"))
    if not numbers:
        # Identifier kinds version/number are digit-bearing by definition.
        numbers = frozenset(v for v in identifiers if _NUMBER_TOKEN.match(v))
    times = (
        _values(fj.get("time_expressions"))
        | _values(fj.get("temporal_expressions"))
        | _values(fj.get("times"))
    )
    temporal = fj.get("temporal")
    if isinstance(temporal, dict):
        times |= _values(temporal.get("expressions"))
        ev = temporal.get("event_at") or fj.get("event_at")
        if isinstance(ev, str) and ev:
            times |= {ev}
    elif isinstance(row.get("event_at"), str) and row["event_at"]:
        times |= {row["event_at"]}
    return DedupFields(
        polarity=str(row.get("polarity") or ""),
        memory_type=str(row.get("type") or ""),
        identifiers=identifiers,
        number_tokens=numbers,
        time_expressions=times,
    )


def _merge_fields(base: DedupFields, fill: DedupFields) -> DedupFields:
    """Fill base's empty dimensions from a second extraction (projection
    tokens recompute) — never overwrite a declared value."""
    return DedupFields(
        polarity=base.polarity or fill.polarity,
        memory_type=base.memory_type or fill.memory_type,
        identifiers=base.identifiers or fill.identifiers,
        number_tokens=base.number_tokens or fill.number_tokens,
        time_expressions=base.time_expressions or fill.time_expressions,
    )


def _coerce_fields(text_fields: Any) -> DedupFields:
    if text_fields is None:
        return DedupFields()
    if isinstance(text_fields, DedupFields):
        return text_fields
    if isinstance(text_fields, str):
        return fields_from_text(text_fields)
    if isinstance(text_fields, dict):
        return DedupFields(
            polarity=str(text_fields.get("polarity") or ""),
            memory_type=str(
                text_fields.get("type")
                or text_fields.get("memory_type")
                or ""
            ),
            identifiers=frozenset(
                str(v) for v in text_fields.get("identifiers") or ()
            ),
            number_tokens=frozenset(
                str(v) for v in text_fields.get("number_tokens") or ()
            ),
            time_expressions=frozenset(
                str(v) for v in text_fields.get("time_expressions") or ()
            ),
        )
    raise VerbatimError(
        ErrorCode.VALIDATION,
        f"text_fields must be DedupFields, dict, str, or None; "
        f"got {type(text_fields).__name__}",
    )


# ---------------------------------------------------------------------------
# duplicate_links I/O — prefer repos_v5 (storage worker), else the same
# allowlist-column SQL the repo generates (contracts §3 DDL is frozen).
# ---------------------------------------------------------------------------

_LINK_COLS = ("source_id", "revision", "group_id", "method", "score",
              "created_at")
_rv5_ok: Optional[bool] = None


def _rv5() -> bool:
    global _rv5_ok
    if _rv5_ok is None:
        _rv5_ok = _repos_v5 is not None and "duplicate_links" in getattr(
            _repos_v5, "_COLUMNS", {}
        )
    return _rv5_ok


def _rows(cur: sqlite3.Cursor) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _has_links_table(conn: sqlite3.Connection) -> bool:
    return has_table(conn, "duplicate_links")


def _link_rows(conn: sqlite3.Connection, source_id: str,
               revision: int) -> list:
    if not _has_links_table(conn):
        return []
    if _rv5():
        return _repos_v5.query(
            conn, "duplicate_links",
            {"source_id": source_id, "revision": revision},
        )
    return _rows(conn.execute(
        "SELECT source_id, revision, group_id, method, score, created_at"
        " FROM duplicate_links WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ))


def _group_rows(conn: sqlite3.Connection, group_id: str) -> list:
    if not _has_links_table(conn):
        return []
    if _rv5():
        return _repos_v5.query(
            conn, "duplicate_links", {"group_id": group_id}
        )
    return _rows(conn.execute(
        "SELECT source_id, revision, group_id, method, score, created_at"
        " FROM duplicate_links WHERE group_id = ?",
        (group_id,),
    ))


def _delete_links(conn: sqlite3.Connection, where: dict) -> int:
    if _rv5():
        return _repos_v5.delete(conn, "duplicate_links", where)
    clauses = " AND ".join(f"{k} = ?" for k in sorted(where))
    return conn.execute(
        f"DELETE FROM duplicate_links WHERE {clauses}",
        tuple(where[k] for k in sorted(where)),
    ).rowcount


def _insert_link(conn: sqlite3.Connection, row: dict) -> None:
    if _rv5():
        _repos_v5.insert(conn, "duplicate_links", row)
        return
    conn.execute(
        "INSERT INTO duplicate_links"
        " (source_id, revision, group_id, method, score, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (row["source_id"], row["revision"], row["group_id"],
         row["method"], row["score"], row["created_at"]),
    )


def _upsert_link(conn: sqlite3.Connection, *, source_id: str, revision: int,
                 group_id: str, method: str, score: float,
                 created_at: str) -> None:
    """Insert the member row; on method-conflict only re-anchor the
    group_id — the first admission score/method is the honest record."""
    existing = [
        r for r in _link_rows(conn, source_id, revision)
        if r["method"] == method
    ]
    if not existing:
        _insert_link(conn, {
            "source_id": source_id,
            "revision": revision,
            "group_id": group_id,
            "method": method,
            "score": score,
            "created_at": created_at,
        })
        return
    if existing[0]["group_id"] != group_id:
        if _rv5():
            _repos_v5.update(
                conn, "duplicate_links", {"group_id": group_id},
                {"source_id": source_id, "revision": revision,
                 "method": method},
            )
        else:
            conn.execute(
                "UPDATE duplicate_links SET group_id = ?"
                " WHERE source_id = ? AND revision = ? AND method = ?",
                (group_id, source_id, revision, method),
            )


def _rewrite_group(conn: sqlite3.Connection, old_gid: str,
                   new_gid: str) -> None:
    if old_gid == new_gid:
        return
    if _rv5():
        _repos_v5.update(conn, "duplicate_links",
                         {"group_id": new_gid}, {"group_id": old_gid})
        return
    conn.execute(
        "UPDATE duplicate_links SET group_id = ? WHERE group_id = ?",
        (new_gid, old_gid),
    )


# ---------------------------------------------------------------------------
# Namespace membership + liveness
# ---------------------------------------------------------------------------


def _namespace_members(conn: sqlite3.Connection, namespace: str,
                       self_id: Optional[str] = None) -> tuple:
    """(members, authority): source_id -> disposition, plus which
    authority resolved the namespace.

    ``source_state`` (contracts §3) is the source→namespace map when the
    v5 control table manages this namespace — i.e. it has rows for the
    namespace or for ``self_id`` (authority ``"state"``). Otherwise
    (pre-v5 stores, or a schema where the control artifact isn't
    populated) fall back to the sources row whose ``scope_id`` equals the
    namespace partition token (authority ``"scope"``). The empty dict
    means the namespace has no verifiable members — callers must treat
    that as "no linkable peers", never as "unfiltered". The authority
    name lets candidate scans push the same scoping into SQL so the
    bounded LIMIT applies inside the namespace, not across the store.
    """
    members: dict[str, Optional[str]] = {}
    if has_table(conn, "source_state"):
        for sid, disp in conn.execute(
            "SELECT source_id, disposition FROM source_state"
            " WHERE namespace = ?",
            (namespace,),
        ):
            members[str(sid)] = disp
        if members:
            return members, "state"
        if self_id is not None and conn.execute(
            "SELECT 1 FROM source_state WHERE source_id = ?",
            (self_id,),
        ).fetchone():
            # The control artifact manages this source; the namespace
            # genuinely has no other members.
            return members, "state"
    for (sid,) in conn.execute(
        "SELECT source_id FROM sources WHERE scope_id = ?", (namespace,)
    ):
        members.setdefault(str(sid), None)
    return members, "scope"


def _scoped_predicate(authority: str) -> str:
    """SQL fragment restricting a candidate query (aliases: ``p``/``sr``
    for the projection/revision table, ``s`` for sources) to the
    resolved namespace authority. The predicate parameter is always the
    namespace string, appended to the query params."""
    if authority == "state":
        return (" AND EXISTS (SELECT 1 FROM source_state ss"
                " WHERE ss.source_id = s.source_id"
                " AND ss.namespace = ?)")
    return " AND s.scope_id = ?"


def _member_info(conn: sqlite3.Connection, source_id: str,
                 revision: Optional[int] = None) -> Optional[dict]:
    """Live member facts or None when forgotten/absent.

    Live = source row present AND (revision=None → any non-empty
    revision; revision given → that revision present with non-empty
    payload) AND source_state disposition (when the table+row exist)
    not in _DEAD_DISPOSITIONS. An emptied payload is the erasure signal
    the closure path writes (`_empty_revision`), so it is honored even
    when duplicate_links rows linger.
    """
    srow = conn.execute(
        "SELECT origin, speaker_id, created_us FROM sources"
        " WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if srow is None:
        return None
    if revision is None:
        rrow = conn.execute(
            "SELECT revision, provenance FROM source_revisions"
            " WHERE source_id = ? AND length(payload) > 0"
            " ORDER BY revision",
            (source_id,),
        ).fetchone()
        if rrow is None:
            return None
        revision = int(rrow[0])
        provenance = rrow[1]
    else:
        rrow = conn.execute(
            "SELECT provenance FROM source_revisions"
            " WHERE source_id = ? AND revision = ?"
            " AND length(payload) > 0",
            (source_id, revision),
        ).fetchone()
        if rrow is None:
            return None
        provenance = rrow[0]
    disposition = None
    if has_table(conn, "source_state"):
        st = conn.execute(
            "SELECT disposition FROM source_state WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if st is not None:
            disposition = st[0]
            if disposition in _DEAD_DISPOSITIONS:
                return None
    return {
        "source_id": source_id,
        "revision": revision,
        "origin": srow[0],
        "speaker_id": srow[1],
        "created_us": srow[2],
        "provenance": provenance,
        "disposition": disposition,
        "live": True,
    }


def _member_info_many(conn: sqlite3.Connection,
                      pairs: Iterable) -> dict:
    """Batched ``_member_info`` for explicit ``(source_id, revision)``
    pairs — identical liveness semantics, one query round per ``_BATCH``
    pairs instead of up to three per candidate.

    ``source_state.source_id`` is a PRIMARY KEY, so the LEFT JOIN can
    never fan out; a pair absent from the result is not-live for the
    same reasons ``_member_info`` returns None (missing source row,
    missing/emptied revision payload, dead disposition).
    """
    out: dict = {}
    todo = [(str(sid), int(rev)) for sid, rev in pairs]
    if not todo:
        return out
    st_ok = has_table(conn, "source_state")
    for i in range(0, len(todo), _BATCH):
        part = todo[i:i + _BATCH]
        marks = ",".join("(?,?)" for _ in part)
        params = [v for pair in part for v in pair]
        rows = conn.execute(
            "SELECT s.source_id, s.origin, s.speaker_id, s.created_us,"
            " sr.revision, sr.provenance"
            + (", ss.disposition" if st_ok else ", NULL")
            + " FROM sources s"
            " JOIN source_revisions sr ON sr.source_id = s.source_id"
            + (" LEFT JOIN source_state ss ON ss.source_id = s.source_id"
               if st_ok else "")
            + " WHERE length(sr.payload) > 0"
            " AND (s.source_id, sr.revision) IN (" + marks + ")",
            params,
        ).fetchall()
        for r in rows:
            sid, rev = str(r[0]), int(r[4])
            disposition = r[6] if st_ok else None
            if disposition in _DEAD_DISPOSITIONS:
                continue
            out[(sid, rev)] = {
                "source_id": sid,
                "revision": rev,
                "origin": r[1],
                "speaker_id": r[2],
                "created_us": r[3],
                "provenance": r[5],
                "disposition": disposition,
                "live": True,
            }
    return out


def _enrichment_rows_many(conn: sqlite3.Connection,
                          pairs: Iterable) -> dict:
    """Batched ``_enrichment_row`` for ``(source_id, revision)`` pairs.

    Per pair the contract's ``enrich/v1`` row wins; absent that, the
    lexicographically smallest producer is acceptable evidence — the
    same rule ``_enrichment_row`` applies with two point lookups.
    """
    out: dict = {}
    todo = [(str(sid), int(rev)) for sid, rev in pairs]
    if not todo or not has_table(conn, "enrichment"):
        return out
    for i in range(0, len(todo), _BATCH):
        part = todo[i:i + _BATCH]
        marks = ",".join("(?,?)" for _ in part)
        params = [v for pair in part for v in pair]
        for r in conn.execute(
            "SELECT source_id, revision, type, polarity, time_precision,"
            " time_status, event_at, anchor_at, fields_json, producer"
            " FROM enrichment"
            " WHERE (source_id, revision) IN (" + marks + ")"
            " ORDER BY producer",
            params,
        ):
            key = (str(r[0]), int(r[1]))
            row = {
                "type": r[2], "polarity": r[3], "time_precision": r[4],
                "time_status": r[5], "event_at": r[6], "anchor_at": r[7],
                "fields_json": r[8],
            }
            prev = out.get(key)
            if prev is None or r[9] == ENRICHMENT_VERSION:
                # Rows arrive in producer order; the enrich/v1 row is
                # authoritative, otherwise the smallest producer stays.
                if prev is None or prev[1] != ENRICHMENT_VERSION:
                    out[key] = (row, r[9])
    return {k: v[0] for k, v in out.items()}


def _sig_memo(store: Any) -> Optional[OrderedDict]:
    """Store-scoped shingle-signature memo; ``None`` when no store was
    supplied (raw-conn callers still get the batched scan)."""
    if store is None:
        return None
    memo = getattr(store, _SIG_MEMO_ATTR, None)
    if memo is None:
        try:
            memo = OrderedDict()
            setattr(store, _SIG_MEMO_ATTR, memo)
        except Exception:
            return None
    return memo


def _earliest_live(member_keys: Iterable,
                   conn: sqlite3.Connection) -> Optional[str]:
    """Earliest live member's source_id — the group anchor invariant."""
    best: Optional[tuple] = None
    for sid, rev in member_keys:
        info = _member_info(conn, sid, int(rev))
        if info is None:
            continue
        key = (info["created_us"], sid)
        if best is None or key < best:
            best = key
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Per-namespace dedupe policy (V5-30.08)
# ---------------------------------------------------------------------------


def _policy_key(namespace: str) -> str:
    return _POLICY_META_PREFIX + namespace


def set_dedupe_policy(conn: sqlite3.Connection, namespace: str,
                      policy: str) -> None:
    """Persist the namespace dedupe policy: ``link`` | ``none``.

    There is deliberately no merge/delete mode (V5-30.08) — any other
    value is a caller error, not a silent mapping.
    """
    if policy not in _DEDUPE_POLICIES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"dedupe policy must be one of {sorted(_DEDUPE_POLICIES)}; "
            f"no merge/delete mode exists: {policy!r}",
        )
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (_policy_key(namespace), json_dumps(policy)),
    )


def get_dedupe_policy(conn: sqlite3.Connection, namespace: str) -> str:
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?", (_policy_key(namespace),)
    ).fetchone()
    if row is None:
        return POLICY_LINK
    val = safe_json_loads(row[0])
    return val if val in _DEDUPE_POLICIES else POLICY_LINK


def _resolve_policy(conn: sqlite3.Connection, namespace: str,
                    policy: Optional[str]) -> str:
    if policy is not None:
        if policy not in _DEDUPE_POLICIES:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"dedupe policy must be one of {sorted(_DEDUPE_POLICIES)}: "
                f"{policy!r}",
            )
        return policy
    return get_dedupe_policy(conn, namespace)


# ---------------------------------------------------------------------------
# V5-30.07 hard guards
# ---------------------------------------------------------------------------


def guard_veto(new_fields: DedupFields, cand_fields: DedupFields) -> Optional[str]:
    """The differing guard dimension, or None when the pair may link.

    Strict equality on all five dimensions; a dimension missing on both
    sides is vacuously equal, present-on-one-side vetoes (fail closed).
    """
    if new_fields.polarity != cand_fields.polarity:
        return "polarity"
    if new_fields.identifiers != cand_fields.identifiers:
        return "identifiers"
    if new_fields.number_tokens != cand_fields.number_tokens:
        return "numbers"
    if new_fields.time_expressions != cand_fields.time_expressions:
        return "time"
    if new_fields.memory_type != cand_fields.memory_type:
        return "type"
    return None


def _enrichment_row(conn: sqlite3.Connection, source_id: str,
                    revision: int) -> Optional[dict]:
    if not has_table(conn, "enrichment"):
        return None
    row = conn.execute(
        "SELECT type, polarity, time_precision, time_status, event_at,"
        " anchor_at, fields_json FROM enrichment"
        " WHERE source_id = ? AND revision = ? AND producer = ?",
        (source_id, revision, ENRICHMENT_VERSION),
    ).fetchone()
    if row is None:
        # Any single producer row is acceptable evidence (contract pins
        # enrich/v1; tolerate a differently-named producer rather than
        # reporting the source unenriched).
        row = conn.execute(
            "SELECT type, polarity, time_precision, time_status,"
            " event_at, anchor_at, fields_json FROM enrichment"
            " WHERE source_id = ? AND revision = ?"
            " ORDER BY producer LIMIT 1",
            (source_id, revision),
        ).fetchone()
    if row is None:
        return None
    return {
        "type": row[0], "polarity": row[1], "time_precision": row[2],
        "time_status": row[3], "event_at": row[4], "anchor_at": row[5],
        "fields_json": row[6],
    }


def _tokens_for(conn: sqlite3.Connection, source_id: str,
                revision: int) -> Optional[str]:
    if not has_table(conn, "source_lexical_projection"):
        return None
    row = conn.execute(
        "SELECT tokens FROM source_lexical_projection"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchone()
    return row[0] if row else None


def _fields_for(conn: sqlite3.Connection, source_id: str,
                revision: int) -> DedupFields:
    """Best available guard fields: enrichment row, gap-filled by the
    deterministic recompute over the persisted normalized tokens."""
    row = _enrichment_row(conn, source_id, revision)
    fields = fields_from_enrichment(row) if row else DedupFields()
    tokens = _tokens_for(conn, source_id, revision)
    if tokens:
        fields = _merge_fields(fields, fields_from_text(tokens))
    return fields


# ---------------------------------------------------------------------------
# Group mechanics
# ---------------------------------------------------------------------------


def _attach(conn: sqlite3.Connection, *, source_id: str, revision: int,
            target: tuple, method: str, score: float,
            created_at: str) -> str:
    """Link (source_id, revision) to target inside one group; returns gid.

    Handles join, create, and merge uniformly: the canonical group_id is
    the earliest live member's source_id across the union of both sides'
    existing groups plus this pair; all involved rows re-anchor so the
    invariant holds after every attach (V5-30.05).
    """
    t_sid, t_rev = target
    gids = {r["group_id"] for r in _link_rows(conn, source_id, revision)}
    gids |= {r["group_id"] for r in _link_rows(conn, t_sid, t_rev)}

    member_keys = {(source_id, revision), (t_sid, t_rev)}
    for gid in gids:
        member_keys |= {
            (r["source_id"], int(r["revision"])) for r in _group_rows(conn, gid)
        }
    gid = _earliest_live(member_keys, conn) or (sorted(gids)[0] if gids
                                                else source_id)
    for old in gids:
        _rewrite_group(conn, old, gid)
    _upsert_link(conn, source_id=source_id, revision=revision,
                 group_id=gid, method=method, score=score,
                 created_at=created_at)
    _upsert_link(conn, source_id=t_sid, revision=t_rev,
                 group_id=gid, method=method, score=score,
                 created_at=created_at)
    return gid


# ---------------------------------------------------------------------------
# link_exact — byte-identical / normalized-digest linking (V5-30.05/06)
# ---------------------------------------------------------------------------


def link_exact(conn: sqlite3.Connection, *, source_id: str, revision: int,
               namespace: str, digest, method: str = METHOD_EXACT,
               scan_limit: int = DEFAULT_SCAN_LIMIT,
               policy: Optional[str] = None) -> LinkOutcome:
    """Link a source revision to the earliest live namespace member with
    an identical digest.

    ``method='exact_digest'`` compares ``digest`` (hex string or bytes)
    against ``source_revisions.payload_hmac`` — the profile-keyed HMAC of
    canonical bytes, so a match IS byte-identity (V5-30.05; guards are
    vacuous for identical bytes). ``method='normalized'`` compares against
    ``source_lexical_projection.digest`` — the digest the projection
    persisted — and applies the V5-30.07 guards because norm/v1 folding
    can erase polarity/type cues (V5-30.06).

    Both records are always retained; the link is additive derived data.
    The caller's transaction owns atomicity.
    """
    if method not in (METHOD_EXACT, METHOD_NORMALIZED):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"link_exact method must be 'exact_digest' or 'normalized': "
            f"{method!r}",
        )
    if not _has_links_table(conn):
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "duplicate_links table absent — dedup requires the v5 schema",
        )
    out = LinkOutcome(method=method)
    if _resolve_policy(conn, namespace, policy) == POLICY_NONE:
        out.reason = "policy_none"
        return out

    members, authority = _namespace_members(
        conn, namespace, self_id=source_id)
    new_info = _member_info(conn, source_id, revision)
    if new_info is None:
        out.reason = "not_live"
        return out
    if source_id not in members:
        # Cannot verify the new record's namespace membership — fail
        # closed rather than write a cross-namespace group edge.
        out.reason = "not_in_namespace"
        return out

    scope_sql = _scoped_predicate(authority)
    if method == METHOD_EXACT:
        hex_digest = digest.hex() if isinstance(digest, bytes) else str(digest)
        # Bounded, earliest-first, namespace-scoped in SQL: a digest
        # flood still converges on the same earliest member and the
        # group merge stays correct.
        cand_rows = _rows(conn.execute(
            "SELECT sr.source_id, sr.revision FROM source_revisions sr"
            " JOIN sources s ON s.source_id = sr.source_id"
            " WHERE lower(hex(sr.payload_hmac)) = lower(?)"
            + scope_sql
            + " ORDER BY s.created_us, sr.source_id LIMIT ?",
            (hex_digest, namespace, scan_limit),
        ))
    else:
        if not has_table(conn, "source_lexical_projection"):
            out.reason = "no_index"
            return out
        cand_rows = _rows(conn.execute(
            "SELECT p.source_id, p.revision FROM source_lexical_projection p"
            " JOIN sources s ON s.source_id = p.source_id"
            " WHERE p.digest = ?"
            + scope_sql
            + " ORDER BY s.created_us, p.source_id LIMIT ?",
            (str(digest), namespace, scan_limit),
        ))

    new_fields: Optional[DedupFields] = None
    candidates = []
    for r in cand_rows:
        c_sid, c_rev = str(r["source_id"]), int(r["revision"])
        if c_sid == source_id:
            continue  # a source is never its own duplicate
        if c_sid not in members:
            continue  # namespace isolation — fail closed
        info = _member_info(conn, c_sid, c_rev)
        if info is None:
            continue
        if method == METHOD_NORMALIZED:
            if new_fields is None:
                new_fields = _fields_for(conn, source_id, revision)
            cand_fields = _fields_for(conn, c_sid, c_rev)
            veto = guard_veto(new_fields, cand_fields)
            if veto is not None:
                out.vetoes.append({
                    "source_id": c_sid, "revision": c_rev,
                    "dimension": veto, "score": 1.0,
                })
                continue
        candidates.append(info)

    if not candidates:
        out.reason = "guard_veto" if out.vetoes else "no_match"
        return out
    candidates.sort(key=lambda i: (i["created_us"], i["source_id"]))
    out.guards_vacuous = bool(
        method == METHOD_NORMALIZED
        and new_fields is not None and new_fields.vacuous()
    )
    created_at = rfc3339(now_us())
    # Every digest-identical live member joins the one group — not just
    # the earliest: pairwise attach merges any pre-existing groups so
    # same-digest members can never sit in separate groups.
    target = (candidates[0]["source_id"], candidates[0]["revision"])
    out.group_id = _attach(
        conn, source_id=source_id, revision=revision, target=target,
        method=method, score=1.0, created_at=created_at,
    )
    for other in candidates[1:]:
        out.group_id = _attach(
            conn, source_id=source_id, revision=revision,
            target=(other["source_id"], other["revision"]),
            method=method, score=1.0, created_at=created_at,
        )
    out.linked = True
    out.score = 1.0
    out.linked_to = target
    out.reason = "ok"
    return out


# ---------------------------------------------------------------------------
# link_near — bounded MinHash-class similarity + hard guards (V5-30.06/07)
# ---------------------------------------------------------------------------


class NearPlan:
    """``link_near``'s read-phase result: the full outcome minus the
    attach-write, plus the chosen target (``None`` for terminal
    reasons). :func:`commit_near` re-validates the tiny chosen set
    against the write snapshot before ``_attach`` — same verdicts a
    fused call produces, with the ~ms-scale namespace scan moved off
    the writer lock."""

    __slots__ = ("out", "best", "namespace", "authority", "policy_hit")

    def __init__(
        self, out: LinkOutcome, best: Optional[dict],
        namespace: str, authority: str, policy_hit: str,
    ) -> None:
        self.out = out
        self.best = best
        self.namespace = namespace
        self.authority = authority
        self.policy_hit = policy_hit


def _member_of_namespace(conn: sqlite3.Connection, authority: str,
                         namespace: str, source_id: str) -> bool:
    """Point-membership recheck — the ``_namespace_members`` verdict for
    one source under the authority the plan resolved."""
    if authority == "state":
        return conn.execute(
            "SELECT 1 FROM source_state WHERE source_id = ?"
            " AND namespace = ?",
            (source_id, namespace),
        ).fetchone() is not None
    return conn.execute(
        "SELECT 1 FROM sources WHERE source_id = ? AND scope_id = ?",
        (source_id, namespace),
    ).fetchone() is not None


def plan_near(conn: sqlite3.Connection, *, source_id: str, revision: int,
              namespace: str, signature, text_fields=None,
              threshold: float = DEFAULT_THRESHOLD,
              scan_limit: int = DEFAULT_SCAN_LIMIT,
              policy: Optional[str] = None,
              store: Any = None) -> NearPlan:
    """Read phase of ``link_near`` — identical validation, namespace
    scan, scoring, and guard vetoes, stopping before ``_attach``. The
    returned :class:`NearPlan` carries the outcome (terminal reasons
    resolved) and the chosen member so a committing transaction only
    re-validates the tiny chosen set."""
    if not (0.0 < threshold <= 1.0):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"threshold must be in (0, 1]: {threshold}"
        )
    if scan_limit < 1:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"scan_limit must be >= 1: {scan_limit}"
        )
    if not _has_links_table(conn):
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "duplicate_links table absent — dedup requires the v5 schema",
        )
    out = LinkOutcome(method=METHOD_MINHASH)
    policy_hit = _resolve_policy(conn, namespace, policy)
    if policy_hit == POLICY_NONE:
        out.reason = "policy_none"
        return NearPlan(out, None, namespace, "", policy_hit)

    sig = frozenset(signature or ())
    if not sig:
        out.reason = "no_signature"
        return NearPlan(out, None, namespace, "", policy_hit)

    new_info = _member_info(conn, source_id, revision)
    if new_info is None:
        out.reason = "not_live"
        return NearPlan(out, None, namespace, "", policy_hit)

    # Caller-supplied fields win; the stored enrichment row (then the
    # persisted token projection) fills any dimension left empty, so a
    # partial text_fields dict never disables a guard.
    new_fields = _merge_fields(
        _coerce_fields(text_fields), _fields_for(conn, source_id, revision)
    )

    members, authority = _namespace_members(
        conn, namespace, self_id=source_id)
    if source_id not in members:
        out.reason = "not_in_namespace"
        return NearPlan(out, None, namespace, authority, policy_hit)
    if not has_table(conn, "source_lexical_projection"):
        out.reason = "no_index"
        return NearPlan(out, None, namespace, authority, policy_hit)

    # Namespace-scoped recent-N scan — the bound applies inside the
    # namespace, not across the store. Liveness folds into the same
    # query: the inner ``sources``/``source_revisions`` joins reproduce
    # ``_member_info_many``'s missing-source/empty-payload verdicts and
    # the LEFT JOIN's ``ss.disposition`` reproduces its dead-state
    # verdict; ``members`` still applies the namespace check so the
    # surviving set is identical. ``p.tokens`` is deliberately NOT
    # selected — candidate signature = the shingle signature of the
    # persisted normalized tokens, content-addressed by ``digest`` in
    # the signature memo, so token bytes are fetched only for memo
    # misses and pass-2 survivors.
    st_ok = has_table(conn, "source_state")
    scan = _rows(conn.execute(
        "SELECT p.rowid, p.source_id, p.revision, p.digest,"
        " s.created_us, ss.disposition"
        " FROM source_lexical_projection p"
        " JOIN sources s ON s.source_id = p.source_id"
        " JOIN source_revisions sr ON sr.source_id = p.source_id"
        " AND sr.revision = p.revision AND length(sr.payload) > 0"
        + (" LEFT JOIN source_state ss ON ss.source_id = p.source_id"
           if st_ok else "")
        + " WHERE p.source_id != ?"
        + _scoped_predicate(authority)
        + " ORDER BY s.created_us DESC, p.source_id"
        + " LIMIT ?",
        (source_id, namespace, scan_limit),
    ))

    memo = _sig_memo(store)

    # Pass 1 — namespace + liveness filters, signatures, scores. The
    # threshold survivors' guard fields are then fetched in one batch so
    # veto evaluation stays in scan order without per-candidate queries.
    live_rows: list = []
    for row in scan:
        c_sid = str(row["source_id"])
        if c_sid not in members:
            continue  # namespace isolation — fail closed
        if st_ok and row["disposition"] in _DEAD_DISPOSITIONS:
            continue  # _member_info_many's dead-state verdict, folded
        out.scanned += 1
        live_rows.append(row)

    tokens_by_rowid: dict = {}

    def _tokens_many(rowids: list) -> None:
        """Batch token fetch — one IN(rowid) round per chunk for every
        row the signature memo or pass-2 merge actually needs."""
        todo = [r for r in rowids if r not in tokens_by_rowid]
        for i in range(0, len(todo), _BATCH):
            part = todo[i : i + _BATCH]
            marks = ",".join("?" for _ in part)
            for rid, tok in conn.execute(
                "SELECT rowid, tokens FROM source_lexical_projection"
                f" WHERE rowid IN ({marks})",
                part,
            ):
                tokens_by_rowid[int(rid)] = tok or ""

    sigs: dict = {}   # rowid -> candidate signature
    misses: list = [] # (rowid, digest) needing a signature compute
    for row in live_rows:
        digest = row["digest"]
        cand_sig = None
        if memo is not None and digest:
            cand_sig = memo.get(digest)
            if cand_sig is not None:
                memo.move_to_end(digest)
        if cand_sig is None:
            misses.append((int(row["rowid"]), digest))
        else:
            sigs[int(row["rowid"])] = cand_sig
    if misses:
        _tokens_many([r for r, _d in misses])
        for rowid, digest in misses:
            cand_sig = _signature(
                (tokens_by_rowid.get(rowid) or "").split())
            sigs[rowid] = cand_sig
            if memo is not None and digest:
                memo[digest] = cand_sig
                while len(memo) > _SIG_MEMO_MAX:
                    memo.popitem(last=False)

    scored: list = []
    for row in live_rows:
        c_sid, c_rev = str(row["source_id"]), int(row["revision"])
        rowid = int(row["rowid"])
        cand_sig = sigs.get(rowid)
        if cand_sig is None:
            # Defensive: a digest-less row whose memo is off still
            # computes its signature — identical to the uncached loop.
            _tokens_many([rowid])
            cand_sig = _signature(
                (tokens_by_rowid.get(rowid) or "").split())
            sigs[rowid] = cand_sig
        # Cardinality bound: |A∩B| ≤ min(|A|,|B|), |A∪B| ≥ max(|A|,|B|)
        # → Jaccard ≤ min/max — a pair below the bound can never reach
        # the threshold, so the two set operations are skipped. Exact:
        # the bound never under-counts a pair that could pass.
        na, nb = len(sig), len(cand_sig)
        if na and nb:
            lo, hi = (na, nb) if na <= nb else (nb, na)
            if lo / hi < threshold:
                continue
        union = sig | cand_sig
        score = (len(sig & cand_sig) / len(union)) if union else 0.0
        if score < threshold:
            continue
        scored.append({
            "source_id": c_sid, "revision": c_rev,
            "created_us": row["created_us"], "rowid": int(row["rowid"]),
            "score": score,
        })

    enr_rows = _enrichment_rows_many(
        conn,
        ((c["source_id"], c["revision"]) for c in scored),
    )
    _tokens_many([c["rowid"] for c in scored])

    # Pass 2 — V5-30.07 guards + best-member selection in scan order.
    best: Optional[dict] = None
    for cand in scored:
        c_sid, c_rev = cand["source_id"], cand["revision"]
        score = cand["score"]
        # _fields_for equivalent: enrichment row (batched) merged with the
        # deterministic recompute over the persisted normalized tokens —
        # the projection row's ``tokens`` IS what _tokens_for returns.
        enr = enr_rows.get((c_sid, c_rev))
        cand_fields = (
            fields_from_enrichment(enr) if enr is not None
            else DedupFields()
        )
        tokens = tokens_by_rowid.get(cand["rowid"]) or ""
        if tokens:
            cand_fields = _merge_fields(cand_fields, fields_from_text(tokens))
        veto = guard_veto(new_fields, cand_fields)
        if veto is not None:
            out.vetoes.append({
                "source_id": c_sid, "revision": c_rev,
                "dimension": veto, "score": score,
            })
            continue
        if (best is None
                or score > best["score"]
                or (score == best["score"]
                    and (cand["created_us"], c_sid)
                    < (best["created_us"], best["source_id"]))):
            best = {"source_id": c_sid, "revision": c_rev,
                    "score": score, "created_us": cand["created_us"]}

    out.guards_vacuous = new_fields.vacuous()
    if best is None:
        out.reason = "guard_veto" if out.vetoes else "no_match"
        return NearPlan(out, None, namespace, authority, policy_hit)
    out.reason = "ok"
    return NearPlan(out, best, namespace, authority, policy_hit)


def commit_near(conn: sqlite3.Connection, plan: NearPlan, *,
                source_id: str, revision: int,
                policy: Optional[str] = None,
                ) -> Optional[LinkOutcome]:
    """Write phase of ``link_near`` — re-validates the plan's chosen
    state against this transaction's snapshot, then ``_attach``es.

    Returns ``None`` when the plan is stale — the caller then re-runs
    :func:`link_near` inline so the outcome is whatever the serial
    order produces. Stale means any of:

    * the dedup policy flipped between plan and commit;
    * the plan ended ``not_in_namespace`` — the fenced gate may have
      adopted this source's ``source_state`` row since, changing
      membership;
    * the chosen target left the namespace, or the source itself did;
    * the chosen target is no longer live.
    """
    out = plan.out
    if plan.best is None:
        if out.reason == "not_in_namespace":
            return None
        if _resolve_policy(conn, plan.namespace, policy) != plan.policy_hit:
            return None
        return out
    if _resolve_policy(conn, plan.namespace, policy) != plan.policy_hit:
        return None
    if not _member_of_namespace(
        conn, plan.authority, plan.namespace, source_id
    ):
        return None
    best_sid = plan.best["source_id"]
    target = (best_sid, plan.best["revision"])
    if not _member_of_namespace(
        conn, plan.authority, plan.namespace, best_sid
    ):
        return None
    if _member_info(conn, best_sid, plan.best["revision"]) is None:
        return None
    out.group_id = _attach(
        conn, source_id=source_id, revision=revision, target=target,
        method=METHOD_MINHASH, score=plan.best["score"],
        created_at=rfc3339(now_us()),
    )
    out.linked = True
    out.score = plan.best["score"]
    out.linked_to = target
    out.reason = "ok"
    return out


def link_near(conn: sqlite3.Connection, *, source_id: str, revision: int,
              namespace: str, signature, text_fields=None,
              threshold: float = DEFAULT_THRESHOLD,
              scan_limit: int = DEFAULT_SCAN_LIMIT,
              policy: Optional[str] = None,
              store: Any = None) -> LinkOutcome:
    """Link a source revision to its best near-duplicate group member.

    Candidate scan is namespace-scoped and bounded to the most recent
    ``scan_limit`` projected members (V5-30.06). Each candidate's shingle
    signature is recomputed from its persisted normalized token stream —
    the same ``shingle_signature`` the caller used for ``signature``, so
    Jaccard comparison is apples-to-apples and recomputable. Candidates
    scoring ≥ ``threshold`` must then survive the V5-30.07 guards; a veto
    is recorded in the outcome regardless of score.

    ``store`` — optional store object enabling a content-addressed
    signature memo (keyed on each candidate's persisted projection
    ``digest``, which binds the exact normalized token stream); advisory
    only — scoring results are identical with or without it.
    """
    plan = plan_near(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        signature=signature,
        text_fields=text_fields,
        threshold=threshold,
        scan_limit=scan_limit,
        policy=policy,
        store=store,
    )
    out = commit_near(
        conn, plan, source_id=source_id, revision=revision, policy=policy
    )
    if out is not None:
        return out
    # Stale plan — the fused retry sees the same snapshot this caller's
    # transaction reads, so re-running the pair is idempotent-cheap and
    # produces exactly the serial-order verdict.
    plan = plan_near(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        signature=signature,
        text_fields=text_fields,
        threshold=threshold,
        scan_limit=scan_limit,
        policy=policy,
        store=store,
    )
    out = commit_near(
        conn, plan, source_id=source_id, revision=revision, policy=policy
    )
    if out is None:
        # Cannot stay stale inside one snapshot — the recheck only fails
        # on cross-snapshot drift; guard against a logic slip rather
        # than spin.
        raise VerbatimError(
            ErrorCode.INTEGRITY, "link_near plan/commit diverged"
        )
    return out


# ---------------------------------------------------------------------------
# Group reads — members, corroboration, representative, collapse
# ---------------------------------------------------------------------------


def group_members(conn: sqlite3.Connection, group_id: str) -> list:
    """All member rows of a group with live facts, earliest-first.

    Non-live members (forgotten/erased) remain listed — the group's
    derived links are honest history — flagged ``live=False`` with the
    stale row's stored metadata. Inspect reachability needs the full set.
    """
    rows = _group_rows(conn, group_id)
    out = []
    for r in rows:
        info = _member_info(conn, r["source_id"], int(r["revision"]))
        member = {
            "source_id": r["source_id"],
            "revision": int(r["revision"]),
            "group_id": r["group_id"],
            "method": r["method"],
            "score": r["score"],
            "created_at": r["created_at"],
            "live": info is not None,
        }
        if info is not None:
            member.update({
                "disposition": info["disposition"],
                "created_us": info["created_us"],
                "origin": info["origin"],
                "speaker_id": info["speaker_id"],
                "provenance": info["provenance"],
            })
        out.append(member)
    out.sort(key=lambda m: (
        not m["live"],
        m.get("created_us", 0),
        m["source_id"],
        m["revision"],
    ))
    return out


def member_refs(conn: sqlite3.Connection, group_id: str) -> list:
    """(source_id, revision) refs for every linked member — the set pack
    collapse must keep reachable for ``inspect`` (V5-30.09)."""
    return sorted(
        {(r["source_id"], int(r["revision"]))
         for r in _group_rows(conn, group_id)}
    )


def corroboration_count(conn: sqlite3.Connection, group_id: str) -> int:
    """Independent submitters among live members (V5-30.05).

    Attribution identity is (origin, speaker_id, provenance): copies —
    same submitter re-submitting identical bytes under a new source_id —
    never count twice. Distinct submitters/origins do. Forgotten members
    contribute nothing.
    """
    attributions = set()
    for r in _group_rows(conn, group_id):
        info = _member_info(conn, r["source_id"], int(r["revision"]))
        if info is None:
            continue
        attributions.add(
            (info["origin"], info["speaker_id"], info["provenance"])
        )
    return len(attributions)


def representative(conn: sqlite3.Connection,
                   group_id: str) -> Optional[dict]:
    """The canonical member: earliest live, preferring non-superseded.

    Depreferred dispositions (superseded/corrected/retracted/archived)
    sort after active/recorded/unstated members so pack collapse picks
    the current record; among peers, earliest ``created_us`` wins —
    matching the group anchor invariant.
    """
    best: Optional[tuple] = None
    best_info: Optional[dict] = None
    for r in _group_rows(conn, group_id):
        info = _member_info(conn, r["source_id"], int(r["revision"]))
        if info is None:
            continue
        penalty = 1 if info["disposition"] in _DEPREFERRED_DISPOSITIONS else 0
        key = (penalty, info["created_us"], info["source_id"])
        if best is None or key < best:
            best = key
            best_info = info
    if best_info is None:
        return None
    best_info["group_id"] = group_id
    return best_info


def collapse_for_hit(conn: sqlite3.Connection, source_id: str,
                     revision: int) -> Optional[dict]:
    """Pack-time collapse view for one hit (V5-30.09), or None when the
    revision has no duplicate links.

    ``collapsed_duplicates`` counts live members hidden under the
    representative; ``members`` keeps every linked ref (live or not)
    reachable for inspect; ``corroboration`` counts independent
    submitters only.
    """
    rows = _link_rows(conn, source_id, revision)
    if not rows:
        return None
    gid = rows[0]["group_id"]
    members = group_members(conn, gid)
    live = [m for m in members if m["live"]]
    rep = representative(conn, gid)
    return {
        "group_id": gid,
        "representative": rep,
        "collapsed_duplicates": max(0, len(live) - 1),
        "corroboration": corroboration_count(conn, gid),
        "members": members,
        "member_refs": member_refs(conn, gid),
    }


def link_group(conn: sqlite3.Connection, source_id: str,
               revision: int) -> Optional[str]:
    """The revision's duplicate-group id, or None when unlinked — the
    membership probe pack-time collapse needs per delivered hit."""
    rows = _link_rows(conn, source_id, int(revision))
    return str(rows[0]["group_id"]) if rows else None


# ---------------------------------------------------------------------------
# Acceptance-time collapse — replayed submissions that never became a member
# ---------------------------------------------------------------------------

#: ``meta`` key prefix for duplicate submissions collapsed at acceptance
#: (V5-30.05/30.09). Content-keyed ingest dedup replays a byte-identical
#: submission onto the committed record — no second (source_id, revision)
#: member row exists for ``duplicate_links`` to group, so the collapse
#: event is recorded here as derived bookkeeping. Canonical bytes,
#: receipts, and attribution are unchanged; a collapsed duplicate
#: submission is never corroboration (same submitter, same bytes — not
#: an independent source).
_DUPSUB_PREFIX = "dedup.submissions."


def record_duplicate_submission(conn: sqlite3.Connection,
                                source_id: str, revision: int) -> int:
    """Durably count one duplicate submission collapsed onto an existing
    record at acceptance; returns the running count.

    This is the ingest-replay complement to ``link_exact``: the dedup key
    collapses byte-identical submissions onto one source record before a
    second member row can exist. Write inside the accepting transaction
    so the collapse fact commits atomically with the replayed receipt.
    """
    key = _DUPSUB_PREFIX + f"{source_id}:{int(revision)}"
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?", (key,)
    ).fetchone()
    rec = safe_json_loads(row[0]) if row else None
    n = (int(rec.get("count") or 0) if isinstance(rec, dict) else 0) + 1
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (
            key,
            json_dumps({
                "v": 1,
                "count": n,
                "source_id": source_id,
                "revision": int(revision),
                "updated_us": now_us(),
            }),
        ),
    )
    return n


def duplicate_submission_count(conn: sqlite3.Connection,
                               source_id: str, revision: int) -> int:
    """Collapsed duplicate submissions recorded at acceptance (0 when
    none) — pack-time ``collapsed_duplicates`` input for records whose
    byte-identical re-adds replayed instead of minting member rows."""
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?",
        (_DUPSUB_PREFIX + f"{source_id}:{int(revision)}",),
    ).fetchone()
    if row is None:
        return 0
    rec = safe_json_loads(row[0])
    if not isinstance(rec, dict):
        return 0
    try:
        return max(0, int(rec.get("count") or 0))
    except (TypeError, ValueError):
        return 0


def duplicate_submission_counts(conn: sqlite3.Connection,
                                pairs: Iterable) -> dict:
    """``{(source_id, revision): collapsed_submission_count}`` for many
    revisions at once — one prefix scan of the ``dedup.submissions.*``
    meta keys instead of one probe per delivered hit."""
    want = {(str(s), int(r)) for s, r in pairs}
    if not want:
        return {}
    out: dict = {}
    for key, value_json in conn.execute(
        "SELECT key, value_json FROM meta WHERE key LIKE ?",
        (_DUPSUB_PREFIX + "%",),
    ):
        sid, sep, rev_s = str(key)[len(_DUPSUB_PREFIX):].rpartition(":")
        if not sep:
            continue
        try:
            pair = (sid, int(rev_s))
        except (TypeError, ValueError):
            continue
        if pair not in want:
            continue
        rec = safe_json_loads(value_json)
        if not isinstance(rec, dict):
            continue
        try:
            out[pair] = max(0, int(rec.get("count") or 0))
        except (TypeError, ValueError):
            continue
    return out


def link_groups(conn: sqlite3.Connection, pairs: Iterable) -> dict:
    """``{(source_id, revision): group_id}`` membership map — one
    chunked ``IN`` probe instead of one ``link_group`` per hit."""
    want = sorted({(str(s), int(r)) for s, r in pairs})
    if not want or not _has_links_table(conn):
        return {}
    out: dict = {}
    for i in range(0, len(want), 200):
        chunk = want[i:i + 200]
        where = " OR ".join(
            "(source_id = ? AND revision = ?)" for _ in chunk
        )
        flat = [v for pair in chunk for v in pair]
        for row in conn.execute(
            "SELECT source_id, revision, group_id FROM duplicate_links"
            f" WHERE {where} ORDER BY group_id",
            flat,
        ):
            out.setdefault((str(row[0]), int(row[1])), str(row[2]))
    return out


def drop_member(conn: sqlite3.Connection, source_id: str,
                revision: Optional[int] = None) -> int:
    """Deletion-closure hook: remove a forgotten member's link rows.

    Forgetting one member never forgets siblings (V5-30.08) — only this
    member's rows are removed. Affected groups re-anchor to the new
    earliest live member; a group reduced to a single remaining row
    dissolves (one member is not a duplicate set). Returns rows deleted.
    """
    if not _has_links_table(conn):
        return 0
    if revision is None:
        rows = _rows(conn.execute(
            "SELECT group_id FROM duplicate_links WHERE source_id = ?",
            (source_id,),
        ))
        deleted = _delete_links(conn, {"source_id": source_id})
    else:
        rows = _rows(conn.execute(
            "SELECT group_id FROM duplicate_links"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ))
        deleted = _delete_links(
            conn, {"source_id": source_id, "revision": revision})
    for gid in {r["group_id"] for r in rows}:
        remaining = _group_rows(conn, gid)
        if len(remaining) <= 1:
            # One member is not a duplicate set — the group dissolves.
            _delete_links(conn, {"group_id": gid})
            continue
        anchor = _earliest_live(
            ((r["source_id"], int(r["revision"])) for r in remaining), conn)
        if anchor and anchor != gid:
            _rewrite_group(conn, gid, anchor)
    return deleted


__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "DEFAULT_THRESHOLD",
    "DedupFields",
    "LINK_METHODS",
    "LinkOutcome",
    "METHOD_EXACT",
    "METHOD_MINHASH",
    "METHOD_NORMALIZED",
    "POLICY_LINK",
    "POLICY_NONE",
    "collapse_for_hit",
    "corroboration_count",
    "drop_member",
    "duplicate_submission_counts",
    "fields_from_enrichment",
    "fields_from_text",
    "get_dedupe_policy",
    "group_members",
    "guard_veto",
    "link_exact",
    "link_groups",
    "link_near",
    "member_refs",
    "representative",
    "set_dedupe_policy",
]
