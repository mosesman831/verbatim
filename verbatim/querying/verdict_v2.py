"""Structural support verdict for the V7 search pipeline
(``support_verdict/v3`` — SPEC_V7 §11, V7-11.01–08 + SPEC_V8 §12,
V8-12.01–12.06; worker contract docs/v7_contracts.md).

This module replaces the literal-coverage floor of
``querying/verdict.py`` (``support_verdict/v1``) that produced the
10.4% zero-result abstentions recorded as D7-04. The verdict is
*structural*: it never deletes a ranked candidate because its literal
term coverage fell under a floor. Instead it

1. labels every delivered evidence group ``supported | partial | weak``
   from support **signals** — identifier exact/normalized match,
   entity-canon coverage, temporal-window match, real lane match
   signals — with term coverage consulted only as a *labeling* signal,
   never a deletion floor (V7-11.01),
2. marks every member ``support ∈ {real, associative}`` (V8-12.05): a
   candidate counts as real support only when its own text or entity
   postings match an asked constraint — a query content term, a query
   entity canon, an asked identifier surface, or the asked temporal
   window — or when a postings lane membership (``lex``/``fuzzy``/
   ``exact_id``/``ent``/``time``/``typed``/``obs``/``source``) attests
   such a match.  Candidates supported only by graph, dense, or
   propagation signals are ``associative``; a group with no
   real-support member labels ``weak`` (corroboration is per-group —
   one real member rescues the group's label),
3. reduces the whole result to ``ready | insufficient`` using ONLY the
   four declared abstention triggers (V7-11.01, V8-12.02), and
4. reports ``answerability`` (§21.9, V8-12.03) — premise confidence as
   an advisory field: ``supported | partial | weak_only |
   unverified_premise | contradicted_premise | no_evidence``.  Premise
   doubt is *reported*, never withheld — ``status`` stays structural.

   - **(a) identifier** — an identifier-bearing query where any asked
     identifier has no exact or normalized match anywhere in the
     eligible candidates or the declared corpus (``ctx`` identifier
     peers; equivalence per ``ident_eq/v1``, §32.8, V7-11.05). A
     lane-level ``identifier_hit``-style signal without attribution
     counts as existence evidence;
   - **(b) empty** — zero eligible candidates after all lanes;
   - **(c) calibrated** — the top S4/S5 score falls under the fitted
     ``support_calibration/v2`` threshold for the (profile, encoder
     tier, CE) combination. The trigger is DISABLED when calibration is
     ``None``/unfitted or ``separates`` is false, and the disabled
     state is reported (V7-11.06) via ``calibration_status`` and the
     ``calibration=`` marker in the ``missing`` note;
   - **(d) negative-evidence** — an ``abstain_likely`` intent where the
     corpus carries explicit negative evidence (negated/contradicting
     items) and no group supports the premise; the speaker
     ``premise_mismatch`` leg of (d) fires ONLY when the
     ``verdict.premise_speaker`` flag is on (V8-12.01 — **default
     off**, measured non-discriminating in D8-02); the flag-on path is
     preserved verbatim for the ablation harness (V8-12.04
     thresholds), and a ``verifier`` hand-off reporting contradiction
     while ``ship="status"`` adds the same trigger.

Verdict evidence (V8-12.06): every group detail records
``evidence_members`` — the members NOT produced solely by lanes
declared deadline-cut (``ctx.manifest['deadline_cut_lanes']`` /
``ctx.cut_lanes``).  Status and answerability read the
``evidence_*`` aggregates only; a cut lane's candidates still deliver
and still carry group labels, but they cannot change the verdict.

``insufficient`` results always carry a ``MissingDescriptor`` naming
the query facets — ``terms``, ``entities``, ``time_window``,
``identifier`` — that found no support (V7-11.08).

Group labels (``ident_eq/v1`` + coverage measurements; the constants
are ``provisional/v7-r0`` until the formula search selects):

- ``supported`` — identifier exact/normalized match; or term coverage
  ≥ 0.6; or coverage ≥ 0.4 backed by a real lane signal; or all query
  entity canons covered with term support; or (facetless query) a real
  lane match / temporal-window match.
- ``partial`` — some verified support (term coverage ≥ 0.25, ≥ half the
  entity canons, temporal-window match, a real lane match whose text
  cannot be inspected, or — under an identifier query — topical
  coverage without the identifier).
- ``weak`` — everything else. ``weak`` groups are still delivered,
  labeled, unless the caller asks for ``strict=True`` (V7-11.03, see
  ``deliverable_groups``).

Item contract (duck-typed): ``ScoredCandidate``, ``FusedCandidate``,
``CandidateV7``, ``PackItemV7``, or plain mappings. Text is read from
``quote``/``text``/``content`` (bytes decoded utf-8/replace); lane
membership from ``lane``/``lanes``/``lane_ranks``; features and signals
from ``detail``/``signals``/``score_detail``/``features``/``stats``
mappings. Recognized signal names:

- identifier: ``identifier_exact``/``identifier_hit``/``exact_id``
  (exact) and ``identifier_match``/``ident_eq``/``identifier_normalized``
  (normalized); candidate identifier strings from
  ``identifiers``/``identifier_tokens``/``idents``.
- entities: ``entity_overlap``/``entity_match``/``entity_coverage``/
  ``canon_coverage``/``matched_entities``/``entities``/``canons``; lane
  ``ent``.
- terms: ``term_coverage``/``term_cov``/``coverage`` (numeric),
  ``matched_terms``/``term_hits``/``term_matches``/``hit_terms``
  (lists), ``lexical``/``bm25``/``bm25f``/``idf_coverage``; lanes
  ``lex``/``fuzzy``.
- temporal: ``window_match``/``temporal_match``/``in_window``/
  ``within_window``, ``temporal_proximity`` (>0), ``occurred_us``/
  ``occurred_start_us``/``event_us``/``start_us`` inside the query
  window; lane ``time``.
- negative evidence: ``negative_evidence``/``contradicts_premise``/
  ``contradicts``/``contradicted``/``negated``/``negation`` truthy, or
  ``polarity == "negate"``.

Group keys: ``group``/``group_key``/``session_id``/``session`` detail
first (a mapping session contributes its ``id``/``session_id``), else
``src:<source_id>``, else ``unit:<unit_id>``. Input order is preserved.

``ctx`` (optional) supplies ``corpus_identifiers``/``identifier_peers``
— identifier strings visible in the eligible scope — which extend the
peer set the hash-prefix uniqueness rule consults; the
``verdict.premise_speaker`` flag (read off ``ctx.policy`` first —
§23 flags travel on the policy object the pipeline threads — then
``ctx.manifest``, then a direct ``ctx`` attribute; default OFF);
``deadline_cut_lanes``/``cut_lanes`` for V8-12.06 evidence filtering;
and the ``conn``/``store`` + ``scope_id``/``generation`` read snapshot
the bounded speaker/mention probes use (``units.speaker_canon``,
``entity_mentions`` — needed for premise measurement and §21.9 row 4
independent of the flag). Nothing else of ``ctx`` is consumed; the
verdict never widens eligibility.

Determinism: pure functions of the inputs; identical inputs produce
identical outputs (V7-27 hard invariant).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlsplit

from ..core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    GroupVerdict,
    MissingDescriptor,
    QueryViewV7,
    ResultStatus,
    SupportLabel,
)

#: Version pin for this verdict contract — bumped on any rule change.
VERDICT_VERSION = "support_verdict/v3"

#: Identifier-equivalence contract implemented here (SPEC_V7 §32.8).
IDENT_EQ_VERSION = "ident_eq/v1"

#: Abstention trigger ids (V7-11.01 (a)–(d)).
TRIGGER_IDENTIFIER = "a"   # hard identifier unmatched in eligible set
TRIGGER_EMPTY = "b"        # zero eligible candidates
TRIGGER_CALIBRATED = "c"   # calibrated top-score threshold
TRIGGER_NEGATIVE = "d"     # explicit negative evidence, abstain_likely

#: ``answerability`` values (§21.9, V8-12.03) — evaluated in declared
#: order; the first matching row wins.
ANSWERABILITY_SUPPORTED = "supported"
ANSWERABILITY_PARTIAL = "partial"
ANSWERABILITY_WEAK_ONLY = "weak_only"
ANSWERABILITY_UNVERIFIED_PREMISE = "unverified_premise"
ANSWERABILITY_CONTRADICTED_PREMISE = "contradicted_premise"
ANSWERABILITY_NO_EVIDENCE = "no_evidence"
ANSWERABILITY_VALUES = frozenset({
    ANSWERABILITY_SUPPORTED, ANSWERABILITY_PARTIAL,
    ANSWERABILITY_WEAK_ONLY, ANSWERABILITY_UNVERIFIED_PREMISE,
    ANSWERABILITY_CONTRADICTED_PREMISE, ANSWERABILITY_NO_EVIDENCE,
})

#: Group-label coverage bands (provisional/v7-r0). These are *labeling*
#: thresholds only — nothing is deleted or withheld because of them.
LABEL_STRONG_COV = 0.6
LABEL_MID_COV = 0.4
LABEL_SOME_COV = 0.25

_MISSING = object()

# ---------------------------------------------------------------------------
# Text fold + tokenization (local; the analyzer seam is a different worker's
# module — this fold is NFKC + casefold + diacritic strip, deterministic)
# ---------------------------------------------------------------------------


def _fold(text: Any) -> str:
    """Matching-projection fold: NFKC, casefold, strip combining marks."""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.casefold()
    if s.isascii():
        # NFD is the identity on ASCII and no ASCII codepoint carries the
        # Mn category — the general path below would return s unchanged.
        return s
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: Optional[str]) -> frozenset:
    if not text:
        return frozenset()
    return frozenset(
        t for t in _TOKEN_RE.findall(_fold(text)) if t.strip("_")
    )


# ---------------------------------------------------------------------------
# Identifier equivalence — ident_eq/v1 (SPEC_V7 §32.8, V7-11.05)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SEMVER_RE = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)"
    r"(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)
_HEX_RE = re.compile(r"^[0-9A-Fa-f]{7,}$")
_DIGITS_RE = re.compile(r"^\d+$")
_HANDLE_RE = re.compile(r"^[@#][\w][\w.-]*$", re.UNICODE)
_TICKET_RE = re.compile(r"^[A-Za-z0-9]+(?:[-_./ ]+[A-Za-z0-9]+)+$")
_SEP_RUN_RE = re.compile(r"[-_./ ]+")
_PCT_RE = re.compile(r"%([0-9A-Fa-f]{2})")

#: Scheme -> default port dropped by URL normalization.
_DEFAULT_PORTS = {
    "http": 80, "https": 443, "ftp": 21, "ftps": 990,
    "ws": 80, "wss": 443, "ssh": 22, "telnet": 23, "gopher": 70,
}

#: Identifier-shaped tokens inside candidate text. Extraction is
#: deliberately permissive — a false positive only creates an extra
#: comparison, never a match by itself. Space-separated pairs are a
#: second pass restricted to digit-bearing chunks so "New York" is not
#: read as a product code.
_IDENT_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"'()\[\]{}]+"
    r"|[@#]\w[\w.-]*"
    r"|\bv?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b"
    r"|\b[0-9A-Fa-f]{7,64}\b"
    r"|\b[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)+\b"
    r"|\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{2,}\b"
    r"|\b\d+\b",
    re.UNICODE,
)
_IDENT_SPACE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]* +\d[A-Za-z0-9]*\b"
    r"|\b\d[A-Za-z0-9]* +[A-Za-z0-9]+\b",
    re.UNICODE,
)


def ident_kind(value: Any) -> str:
    """The §32.8 shape class of one identifier surface."""
    v = str(value).strip()
    if not v:
        return "empty"
    if _URL_RE.match(v):
        return "url"
    if _SEMVER_RE.match(v):
        return "semver"
    if _DIGITS_RE.match(v):
        return "numeric"
    if _HEX_RE.match(v):
        return "hash"
    if _HANDLE_RE.match(v):
        return "handle"
    if _TICKET_RE.match(v):
        return "ticket"
    return "token"


def _pct_norm(s: str) -> str:
    """Percent-encoding case normalization (hex digits uppercased)."""
    return _PCT_RE.sub(lambda m: "%" + m.group(1).upper(), s)


def _url_key(v: str) -> str:
    p = urlsplit(v)
    try:
        port = p.port
    except ValueError:
        port = None
    scheme = p.scheme.lower()
    host = (p.hostname or "").lower()
    netloc = host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"
    userinfo = ""
    if p.username:
        userinfo = p.username + (":" + p.password if p.password else "") + "@"
    path = _pct_norm(p.path or "")
    path = "" if path == "/" else path.rstrip("/")
    return (
        f"url|{scheme}://{userinfo}{netloc}{path}"
        f"|{_pct_norm(p.query or '')}|{_pct_norm(p.fragment or '')}"
    )


def _semver_key(v: str) -> str:
    m = _SEMVER_RE.match(v)
    assert m is not None
    pre = m.group(4)
    pre_key = ""
    if pre:
        pre_key = ".".join(
            str(int(p)) if p.isdigit() else p for p in pre.split(".")
        )
    return (
        f"semver|{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3))}"
        f"|{pre_key}"
    )


def ident_canon(value: Any) -> str:
    """Canonical comparison key for one identifier surface.

    Rules (§32.8): case-insensitive tickets/handles/tokens; separators
    ``- _ . / <space>`` interchangeable on ticket/product codes; leading
    zeros ignored on purely numeric ids of ≥ 3 digits; URL scheme/host
    case, default ports, trailing slash and percent-encoding case
    normalized; semantic versions compare component-wise with optional
    ``v`` prefix and build metadata ignored; hashes compare
    case-insensitively (the ≥7-char prefix rule lives in ``ident_eq``
    because it needs scope uniqueness).
    """
    v = str(value).strip()
    kind = ident_kind(v)
    if kind == "url":
        try:
            return _url_key(v)
        except Exception:
            pass  # fall through to the token form
    elif kind == "semver":
        return _semver_key(v)
    elif kind == "numeric":
        return "num|" + (v.lstrip("0") or "0" if len(v) >= 3 else v)
    elif kind == "hash":
        return "hash|" + v.casefold()
    elif kind == "handle":
        return "handle|" + _fold(v)
    elif kind == "ticket":
        return "tick|" + _fold(_SEP_RUN_RE.sub("-", v))
    return "tok|" + _fold(v)


def _hash_prefix_eq(a: str, b: str, peers: Iterable[str]) -> bool:
    """≥7-char hash prefix match, gated on scope uniqueness (§32.8)."""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 7 or not long.startswith(short):
        return False
    fulls = set()
    for p in list(peers or ()) + [a, b]:
        pv = str(p).strip()
        if (
            len(pv) > len(short)
            and pv.casefold().startswith(short)
            and _HEX_RE.match(pv)
        ):
            fulls.add(pv.casefold())
    return fulls == {long}


def ident_eq(a: Any, b: Any, peers: Optional[Iterable[str]] = None) -> bool:
    """Normalized identifier equivalence (``ident_eq/v1``, §32.8).

    Byte-exact surfaces always match (the exact form is preferred);
    otherwise the canonical form decides. Hex hashes additionally match
    on a ≥7-character prefix when that prefix names a unique full hash
    among ``peers`` (the identifiers visible in the eligible scope).
    """
    sa, sb = str(a).strip(), str(b).strip()
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    ka, kb = ident_canon(sa), ident_canon(sb)
    if ka == kb:
        return True
    if ka.startswith("hash|") and kb.startswith("hash|"):
        return _hash_prefix_eq(ka[5:], kb[5:], peers)
    return False


def _exact_surface(qid: str, text: str) -> bool:
    """The identifier surface appears in ``text`` as a token bounded by
    non-word characters (so ``abc1234`` is not "exact" inside
    ``abc1234ff00dd``)."""
    try:
        return bool(
            re.search(r"(?<!\w)" + re.escape(qid) + r"(?!\w)", text)
        )
    except re.error:
        return qid in text


def identifier_tokens(text: str) -> Tuple[str, ...]:
    """Identifier-shaped surfaces extracted from ``text`` (deterministic
    left-to-right order, deduplicated)."""
    if not text:
        return ()
    out = list(_IDENT_RE.findall(text))
    out.extend(_IDENT_SPACE_RE.findall(text))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return tuple(uniq)


# ---------------------------------------------------------------------------
# Item access — duck-typed ScoredCandidate / FusedCandidate / mapping
# ---------------------------------------------------------------------------

_NESTED_MAPS = (
    "detail", "signals", "score_detail", "features", "stats",
    "lane_ranks",
)

#: Signal names counted as a *real* lane-backed match. Dense/semantic
#: similarity (``similarity``/``dense``/``ce_score``) is deliberately not
#: in this set — it informs the label through measured coverage but
#: never substitutes for a verifiable match signal.
_REAL_SIGNALS = frozenset({
    "lexical", "bm25", "bm25f", "idf_coverage", "term_coverage",
    "term_cov", "matched_terms", "term_hits", "term_matches",
    "identifier_hit", "identifier_exact", "identifier_match",
    "ident_eq", "entity_overlap", "entity_match", "entity_coverage",
    "canon_coverage", "matched_entities", "temporal_match",
    "window_match", "in_window", "event_match", "phrase", "proximity",
    "graph_ppr", "typed", "source",
})
_REAL_LANES = frozenset({
    "lex", "fuzzy", "exact_id", "ent", "time", "graph", "typed",
    "obs", "source",
})

#: V8-12.05 *real-support* sets.  A candidate is real support when its
#: own text or entity postings match an asked constraint — a content
#: term, an entity canon, an identifier surface, or the asked window —
#: or when a postings-lane membership attests such a match.  Graph,
#: dense/similarity, and propagation signals are *associative* support
#: only: they never make an item real on their own.
_REAL_SUPPORT_LANES = _REAL_LANES - {"graph"}
_REAL_SUPPORT_SIGNALS = _REAL_SIGNALS - {"graph_ppr"}
_ID_EXACT_SIGNALS = frozenset({
    "identifier_exact", "identifier_hit", "exact_id",
})
_ID_NORM_SIGNALS = frozenset({
    "identifier_match", "ident_eq", "identifier_normalized",
    "identifier_equiv",
})
_NEGATIVE_SIGNALS = frozenset({
    "negative_evidence", "contradicts_premise", "contradicts",
    "contradicted", "negated", "negation",
})
_WINDOW_SIGNALS = frozenset({
    "window_match", "temporal_match", "in_window", "within_window",
})


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


#: Per-call memo for ``_maps`` — the nested-map inventory of an item is
#: invariant for the duration of one public entry point (inputs are
#: never mutated by the verdict), and ``_find``/``_signal_names`` reach
#: for it ~25× per item.  Entries are keyed by ``id(item)`` but hold the
#: item itself, so a cached entry can never alias a recycled id while it
#: exists.  Public entry points clear it defensively on entry.
_MAPS_MEMO: dict = {}


def _maps(item: Any) -> list:
    """All signal-bearing mappings on an item, breadth-first:

    the item itself (when a mapping), then each ``detail``/``signals``/
    ``score_detail``/``features``/``stats``/``lane_ranks`` member, then
    the same members one level deeper (e.g. ``detail["signals"]``).
    """
    hit = _MAPS_MEMO.get(id(item))
    if hit is not None and hit[0] is item:
        return hit[1]
    maps: list = []
    seen: set = set()

    def add(m: Any) -> None:
        if isinstance(m, Mapping) and id(m) not in seen:
            seen.add(id(m))
            maps.append(m)

    if isinstance(item, Mapping):
        add(item)
    for h in _NESTED_MAPS:
        add(_get(item, h))
    for m in list(maps):
        for h in _NESTED_MAPS:
            add(m.get(h))
    _MAPS_MEMO[id(item)] = (item, maps)
    return maps


def _find(item: Any, names: Iterable[str]) -> Any:
    """First non-None value for ``names`` at top level, then inside the
    nested signal maps (``detail``/``signals``/``score_detail``/
    ``features``/``stats``/``lane_ranks``)."""
    for n in names:
        v = _get(item, n, _MISSING)
        if v is not _MISSING and v is not None:
            return v
    for m in _maps(item):
        for n in names:
            if n in m and m[n] is not None:
                return m[n]
    return None


def _signal_names(item: Any) -> frozenset:
    """Every declared signal/lane name visible on the item."""
    names = set()
    for m in _maps(item):
        names.update(str(k) for k in m)
    lane = _get(item, "lane")
    if lane:
        names.add(str(lane))
    lanes = _find(item, ("lanes",))
    if isinstance(lanes, (list, tuple, set, frozenset)):
        names.update(str(x) for x in lanes)
    return frozenset(names)


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "none")
    return bool(v)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _text(item: Any) -> Optional[str]:
    v = _find(item, ("quote", "text", "content", "span_text", "payload"))
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v if isinstance(v, str) and v else None


def _score(item: Any) -> float:
    for n in ("score", "raw_score", "rrf", "ce_score", "fused_score"):
        f = _num(_find(item, (n,)))
        if f is not None:
            return f
    return 0.0


def _unit_id(item: Any, fallback: str) -> str:
    v = _find(item, ("unit_id", "id", "ref"))
    return str(v) if v is not None else fallback


def _group_key(item: Any, index: int) -> str:
    """Group-key precedence: explicit group key, then session, then
    source, then the unit itself (deterministic)."""
    v = _find(item, ("group", "group_key", "session_id", "session"))
    if isinstance(v, Mapping):
        v = v.get("id", v.get("session_id"))
    if v is not None and str(v):
        return f"grp:{v}"
    src = _find(item, ("source_id", "src"))
    if src is not None and str(src):
        return f"src:{src}"
    unit = _find(item, ("unit_id", "id", "ref"))
    if unit is not None and str(unit):
        return f"unit:{unit}"
    return f"idx:{index}"


def _item_idents(item: Any) -> Tuple[str, ...]:
    out = []
    v = _find(item, (
        "identifiers", "identifier_tokens", "identifier_values",
        "idents",
    ))
    if isinstance(v, (list, tuple, set, frozenset)):
        out.extend(str(x) for x in v)
    elif isinstance(v, str) and v:
        out.append(v)
    out.extend(identifier_tokens(_text(item) or ""))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return tuple(uniq)


def _str_list(v: Any) -> Tuple[str, ...]:
    if isinstance(v, str):
        return (v,)
    if isinstance(v, (list, tuple, set, frozenset)):
        return tuple(str(x) for x in v)
    return ()


def _item_lanes(item: Any) -> frozenset:
    """The lane names that produced this item — ``lane``/``lanes``/
    ``lane_ranks`` membership only (not free-floating signal names)."""
    lanes = set()
    v = _get(item, "lane")
    if v:
        lanes.add(str(v))
    v = _find(item, ("lanes",))
    if isinstance(v, str):
        lanes.add(v)
    elif isinstance(v, Mapping):
        lanes.update(str(k) for k in v)
    elif isinstance(v, (list, tuple, set, frozenset)):
        lanes.update(str(x) for x in v)
    v = _find(item, ("lane_ranks",))
    if isinstance(v, Mapping):
        lanes.update(str(k) for k in v)
    return frozenset(lanes)


# ---------------------------------------------------------------------------
# Flag / knob resolution — §23 register rows reach the verdict through the
# policy object the pipeline threads (``load_policy``/``policy_overrides``)
# or the ``ctx.manifest`` query-param channel (the same route
# ``calibration``/``explain`` take).  Never a parallel config path.
# ---------------------------------------------------------------------------

_FLAG_NAMES = ("premise_speaker", "verdict.premise_speaker")
_FLAG_MAPS = ("flags", "params", "knobs", "arms", "verdict")


def _premise_flag(ctx: Any) -> bool:
    """``verdict.premise_speaker`` — **default OFF** (V8-12.01).

    Resolution order: ``ctx.policy`` (attribute, or a ``flags``/
    ``params``/``knobs``/``arms``/``verdict`` mapping member holding
    ``premise_speaker`` or ``verdict.premise_speaker``) →
    ``ctx.manifest`` (same shapes) → ``ctx`` itself.  First declared
    value wins; absent/``None`` → ``False``.
    """
    if ctx is None:
        return False
    for holder in (
        _get(ctx, "policy"), _get(ctx, "manifest"), ctx,
    ):
        if holder is None:
            continue
        for name in _FLAG_NAMES:
            v = _get(holder, name, _MISSING)
            if v is not _MISSING and v is not None:
                return _truthy(v)
        for mapname in _FLAG_MAPS:
            sub = _get(holder, mapname)
            if isinstance(sub, Mapping):
                for name in _FLAG_NAMES:
                    if sub.get(name) is not None:
                        return _truthy(sub[name])
    return False


def _cut_lanes(ctx: Any) -> frozenset:
    """Lanes the request deadline cut (V8-12.06) — declared via
    ``ctx.deadline_cut_lanes``/``ctx.cut_lanes`` or the same keys inside
    ``ctx.manifest``.  Their candidates still deliver but cannot change
    status or answerability.  Absent → ``∅`` (every member is verdict
    evidence)."""
    if ctx is None:
        return frozenset()
    for name in ("deadline_cut_lanes", "cut_lanes"):
        v = _get(ctx, name)
        if v is None:
            man = _get(ctx, "manifest")
            if isinstance(man, Mapping):
                v = man.get(name)
        if isinstance(v, str):
            return frozenset((v,))
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(str(x) for x in v)
    return frozenset()


# ---------------------------------------------------------------------------
# Query facet extraction
# ---------------------------------------------------------------------------


def _chan(term: Any) -> str:
    return str(_get(term, "channel", "text"))


def _surface(term: Any) -> str:
    v = _get(term, "term", term if isinstance(term, str) else "")
    return str(v)


def _query_parts(query: QueryViewV7) -> Tuple[tuple, tuple, tuple]:
    """``(identifiers, content_terms, entity_canons)`` — folded, deduped,
    with decomposition facets merged in (V7-05.13 sub-queries are part
    of the asked question)."""
    views = [query]
    views.extend(getattr(query, "facets", ()) or ())
    ids, terms, ents = [], [], []
    for qv in views:
        norm = getattr(qv, "norm", None)
        for t in getattr(norm, "terms", ()) or ():
            if _chan(t) == "identifier":
                ids.append(_surface(t))
            elif _chan(t) in ("text", "stem"):
                terms.append(_fold(_surface(t)))
        for t in getattr(norm, "identifiers", ()) or ():
            ids.append(_surface(t))
        for e in getattr(qv, "entity_canons", ()) or ():
            ents.append(_fold(e))
    return (
        tuple(dict.fromkeys(i for i in ids if i)),
        tuple(dict.fromkeys(t for t in terms if t)),
        tuple(dict.fromkeys(e for e in ents if e)),
    )


def _query_window(query: QueryViewV7):
    intent = getattr(query, "intent", None)
    w = getattr(intent, "window", None)
    if w is None:
        return None
    start = _num(_get(w, "start_us"))
    end = _num(_get(w, "end_us"))
    if start is None or end is None:
        return None
    return (start, end)


def _intent_names(query: QueryViewV7) -> frozenset:
    intent = getattr(query, "intent", None)
    if intent is None:
        return frozenset()
    names = set()
    primary = getattr(intent, "primary", intent)
    names.add(str(getattr(primary, "value", primary)))
    for c in getattr(intent, "classes", ()) or ():
        names.add(str(getattr(c, "value", c)))
    return frozenset(names)


def _abstain_likely(query: QueryViewV7) -> bool:
    return "abstain_likely" in _intent_names(query)


# ---------------------------------------------------------------------------
# Per-item measurement
# ---------------------------------------------------------------------------


class _Measure:
    """Measured support evidence for one item."""

    __slots__ = (
        "group_key", "unit_id", "score", "signals", "lanes", "id_level",
        "matched_ids", "covered_terms", "term_frac", "term_measured",
        "covered_ents", "ent_frac", "ent_measured", "entity_named",
        "text_fold", "window", "negative", "support", "soft", "speaker",
        "verdict_evidence", "subject_role",
    )

    def __init__(self) -> None:
        self.group_key = ""
        self.unit_id = ""
        self.score = 0.0
        self.signals = frozenset()
        self.lanes = frozenset()
        self.id_level: Optional[str] = None  # "exact" | "normalized"
        self.matched_ids: Tuple[str, ...] = ()
        self.covered_terms = frozenset()
        self.term_frac = 0.0
        self.term_measured = False
        self.covered_ents = frozenset()
        self.ent_frac = 0.0
        self.ent_measured = False
        self.entity_named = frozenset()
        self.text_fold = ""
        self.window = False
        self.negative = False
        self.support = "associative"  # "real" | "associative" (V8-12.05)
        self.soft = False
        self.speaker: Optional[str] = None
        self.verdict_evidence = True  # V8-12.06 — cut-lane members lose it
        self.subject_role: Optional[str] = None  # author|mention|author+mention


def _measure(
    item: Any,
    index: int,
    qids: tuple,
    qterms: tuple,
    qents: tuple,
    window,
    peers: Tuple[str, ...],
) -> _Measure:
    m = _Measure()
    m.group_key = _group_key(item, index)
    m.unit_id = _unit_id(item, f"idx:{index}")
    m.score = _score(item)
    m.signals = _signal_names(item)
    m.lanes = _item_lanes(item)
    text = _text(item)
    toks = _tokens(text)
    folded_text = _fold(text) if text else ""
    m.text_fold = folded_text

    # --- identifier match (exact surface preferred, then ident_eq) ---
    if qids:
        cand_ids = _item_idents(item)
        matched = []
        level = None
        for qid in qids:
            # "exact" means the surface appears as a bounded token —
            # a prefix of a longer token is NOT exact (that is the
            # hash-prefix rule's job, gated on scope uniqueness).
            if qid in cand_ids or (
                text and _exact_surface(qid, text)
            ):
                matched.append(qid)
                level = "exact"
            elif any(ident_eq(qid, c, peers) for c in cand_ids):
                matched.append(qid)
                level = level or "normalized"
        exact_sig = ("exact_id" in m.signals) or any(
            _truthy(_find(item, (s,)))
            for s in (_ID_EXACT_SIGNALS & m.signals)
        )
        if level is None:
            if exact_sig:
                level = "exact"
            elif any(
                _truthy(_find(item, (s,)))
                for s in (_ID_NORM_SIGNALS & m.signals)
            ):
                level = "normalized"
        elif level == "normalized" and exact_sig:
            level = "exact"
        if level is not None and not matched and len(qids) == 1:
            # a lane signal attests an identifier match; with a single
            # asked identifier it is attributable for facet reporting
            matched.append(qids[0])
        m.id_level = level
        m.matched_ids = tuple(matched)

    # --- term coverage ---
    covered = set()
    if toks:
        covered.update(t for t in qterms if t in toks)
    for name in ("matched_terms", "term_hits", "term_matches", "hit_terms"):
        for x in _str_list(_find(item, (name,))):
            fx = _fold(x)
            if fx in qterms:
                covered.add(fx)
    m.covered_terms = frozenset(covered)
    frac = None
    for name in ("term_coverage", "term_cov", "coverage"):
        v = _num(_find(item, (name,)))
        if v is not None:
            frac = v if v <= 1.0 else v / max(1, len(qterms))
            break
    m.term_frac = min(1.0, frac) if frac is not None else 0.0
    m.term_measured = bool(text) or bool(covered) or frac is not None

    # --- entity-canon coverage ---
    covered_e = set()
    named_e = set()
    if folded_text:
        covered_e.update(e for e in qents if e in folded_text)
    for name in ("matched_entities", "entities", "canons", "entity_canons"):
        for x in _str_list(_find(item, (name,))):
            fx = _fold(x)
            if fx:
                named_e.add(fx)
            if fx in qents:
                covered_e.add(fx)
    m.covered_ents = frozenset(covered_e)
    m.entity_named = frozenset(named_e)
    efrac = None
    for name in ("entity_coverage", "canon_coverage", "entity_cov"):
        v = _num(_find(item, (name,)))
        if v is not None:
            efrac = v if v <= 1.0 else v / max(1, len(qents))
            break
    if efrac is not None:
        efrac = min(1.0, efrac)
    ent_sig = bool(
        {"entity_overlap", "entity_match", "ent"} & m.signals
    ) and any(
        _truthy(_find(item, (s,)))
        for s in ({"entity_overlap", "entity_match"} & m.signals)
    )
    if "ent" in m.signals:
        ent_sig = True
    if efrac is None and ent_sig and qents:
        efrac = 1.0 / len(qents)  # signal without attribution: ≥1 canon
    m.ent_frac = efrac or 0.0
    m.ent_measured = bool(text) or bool(covered_e) or efrac is not None

    # --- temporal window ---
    if window is not None:
        win = any(
            _truthy(_find(item, (s,)))
            for s in (_WINDOW_SIGNALS & m.signals)
        )
        if not win:
            tp = _num(_find(item, ("temporal_proximity",)))
            if tp is not None and tp > 0:
                win = True
        if not win:
            occ = _num(_find(item, (
                "occurred_us", "occurred_start_us", "event_us",
                "start_us",
            )))
            if occ is not None and window[0] <= occ <= window[1]:
                win = True
        m.window = win

    # --- negative evidence ---
    m.negative = any(
        _truthy(_find(item, (s,)))
        for s in (_NEGATIVE_SIGNALS & m.signals)
    ) or str(_find(item, ("polarity",)) or "").lower() == "negate"

    # --- real vs associative support (V8-12.05) ---
    # Real support = the item's own text or entity postings matched an
    # asked constraint (content term, entity canon, identifier surface,
    # temporal window), or a postings-lane membership / verifiable match
    # signal attests it.  Graph/dense/propagation-only support is
    # associative — ``m.soft`` still records the similarity presence.
    if (
        m.covered_terms or m.covered_ents
        or m.term_frac > 0 or m.ent_frac > 0
        or m.id_level is not None or m.window
        or bool(m.lanes & _REAL_SUPPORT_LANES)
        or any(
            _truthy(_find(item, (s,)))
            for s in (m.signals & _REAL_SUPPORT_SIGNALS)
        )
    ):
        m.support = "real"
    m.soft = bool(m.signals & {"dense", "similarity", "ce_score"})

    # --- authoring speaker (premise-mismatch check) -------------------
    sp = _find(item, ("speaker_canon", "speaker"))
    m.speaker = str(sp) if sp is not None else None
    return m


def _premise_speaker(query: QueryViewV7, ctx: Any) -> Optional[str]:
    """The single subject speaker the question presupposes — caller hint
    (``query.speaker_canon``) wins outright; otherwise the query-derived
    canon probe (V75-03.05's ``resolve_query_speaker_canons`` over the
    query's entity canons).  Exactly one resolution → that canon;
    ambiguous/zero/unresolvable → ``None`` and the premise check stays
    off — a guessed speaker is worse than no check.
    """
    hint = getattr(query, "speaker_canon", None)
    if isinstance(hint, str) and hint:
        return hint
    if ctx is None:
        return None
    scope_id = getattr(ctx, "scope_id", None)
    if scope_id is None:
        return None
    try:
        generation = int(getattr(ctx, "generation", None))
    except (TypeError, ValueError):
        return None
    try:
        from ..retrieval.v7.entity import (
            lane_conn,
            resolve_query_speaker_canons,
        )
    except Exception:  # noqa: BLE001 — lane module absent
        return None
    try:
        conn = lane_conn(ctx)
    except Exception:  # noqa: BLE001
        return None
    if conn is None:
        return None
    try:
        resolved = resolve_query_speaker_canons(
            conn,
            str(scope_id),
            generation,
            getattr(query, "entity_canons", None) or (),
        )
    except Exception:  # noqa: BLE001 — probe failure = no resolution
        return None
    return resolved[0] if len(resolved) == 1 else None


def _unit_speakers(ctx: Any, unit_ids: Iterable[str]) -> dict:
    """``unit_id -> speaker_canon`` for verdict-time premise checks.

    Lane signals only carry ``speaker_canon`` when the lane emitted it —
    this one bounded ``units`` lookup (called only when a query speaker
    resolved) covers every candidate regardless of lane provenance.
    Failure → ``{}``: the check degrades to signal-carried speakers.
    """
    ids = sorted({str(u) for u in unit_ids if u})
    if not ids or ctx is None:
        return {}
    scope_id = getattr(ctx, "scope_id", None)
    try:
        generation = int(getattr(ctx, "generation", None))
    except (TypeError, ValueError):
        generation = None
    try:
        from ..retrieval.v7.entity import lane_conn
        conn = lane_conn(ctx)
    except Exception:  # noqa: BLE001
        return {}
    if conn is None or scope_id is None:
        return {}
    out: dict = {}
    try:
        for i in range(0, len(ids), 400):
            part = ids[i : i + 400]
            ph = ",".join("?" for _ in part)
            if generation is None:
                rows = conn.execute(
                    "SELECT unit_id, MAX(generation), speaker_canon"
                    " FROM units WHERE scope_id = ?"
                    f" AND unit_id IN ({ph}) GROUP BY unit_id",
                    [str(scope_id), *part],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT unit_id, MAX(generation), speaker_canon"
                    " FROM units WHERE scope_id = ? AND generation <= ?"
                    f" AND unit_id IN ({ph}) GROUP BY unit_id",
                    [str(scope_id), generation, *part],
                ).fetchall()
            for uid, _g, sp in rows:
                if sp:
                    out[str(uid)] = str(sp)
    except Exception:  # noqa: BLE001 — absent units table on old stores
        return out
    return out


def _subject_canon_forms(subject: str) -> Tuple[str, ...]:
    """Canon forms an ``entity_mentions``/``entity_canon`` probe accepts
    for ``subject`` — the verbatim canon plus the enrichment fold
    (``entities_v2.canon``; the write side's own canonization)."""
    forms = [subject]
    try:
        from ..enrichment.entities_v2 import canon as _ecanon
        forms.append(_ecanon(subject))
    except Exception:  # noqa: BLE001 — enrichment module absent
        forms.append(_fold(subject))
    return tuple(dict.fromkeys(f for f in forms if f))


def _unit_mentions(
    ctx: Any, unit_ids: Iterable[str], subject: str
) -> frozenset:
    """``unit_id`` set carrying an ``entity_mentions`` row for
    ``subject`` — the "mentions it" half of §21.9 row 4.

    One bounded, generation-fenced probe (``unit_id IN`` chunks of
    400), mirroring ``_unit_speakers``.  Failure/absent table → ``∅``:
    the mention check degrades to the signal/text evidence already on
    the measures.
    """
    ids = sorted({str(u) for u in unit_ids if u})
    if not ids or ctx is None or not subject:
        return frozenset()
    scope_id = _get(ctx, "scope_id")
    try:
        generation = int(_get(ctx, "generation"))
    except (TypeError, ValueError):
        generation = None
    try:
        from ..retrieval.v7.entity import lane_conn
        conn = lane_conn(ctx)
    except Exception:  # noqa: BLE001
        return frozenset()
    if conn is None or scope_id is None:
        return frozenset()
    canons = _subject_canon_forms(subject)
    ph_c = ",".join("?" for _ in canons)
    out: set = set()
    try:
        for i in range(0, len(ids), 400):
            part = ids[i : i + 400]
            ph = ",".join("?" for _ in part)
            sql = (
                "SELECT DISTINCT unit_id FROM entity_mentions"
                f" WHERE scope_id = ? AND canon IN ({ph_c})"
            )
            params: list = [str(scope_id), *canons]
            if generation is not None:
                sql += " AND generation <= ?"
                params.append(generation)
            sql += f" AND unit_id IN ({ph})"
            params.extend(part)
            for (uid,) in conn.execute(sql, params).fetchall():
                out.add(str(uid))
    except Exception:  # noqa: BLE001 — no mentions table on old stores
        return frozenset(out)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Presupposition verifier — V8-12.04 research arm
# ---------------------------------------------------------------------------

_VERIFIER_SHIPS = frozenset({"off", "answerability", "status"})


def _verifier_ship(ctx: Any) -> str:
    """``verdict.verifier_ship`` — where the V8-12.04 verifier's
    finding may land: ``off`` (default), ``answerability``, or
    ``status``.  Same holder order as ``_premise_flag``; a bare truthy
    value means ``answerability``."""
    if ctx is None:
        return "off"
    for holder in (_get(ctx, "policy"), _get(ctx, "manifest"), ctx):
        if holder is None:
            continue
        for name in (
            "verifier_ship", "verdict.verifier_ship", "premise_verifier",
        ):
            v = _get(holder, name, _MISSING)
            if v is _MISSING or v is None:
                continue
            if isinstance(v, str) and v in _VERIFIER_SHIPS:
                return v
            return "answerability" if _truthy(v) else "off"
        for mapname in _FLAG_MAPS:
            sub = _get(holder, mapname)
            if isinstance(sub, Mapping):
                for name in ("verifier_ship", "verdict.verifier_ship"):
                    v = sub.get(name)
                    if v is None:
                        continue
                    if isinstance(v, str) and v in _VERIFIER_SHIPS:
                        return v
                    return "answerability" if _truthy(v) else "off"
    return "off"


def _asked_relation(query: QueryViewV7):
    """The ``(predicate_fold, object_fold)`` pair the question asks —
    the V8-12.04 verifier's asked relation — from
    ``query.asked_relation``/``query.relation`` (mapping or 2-sequence)
    or ``predicate`` + ``object_canon``/``object`` fields on the query
    or its intent."""
    for name in ("asked_relation", "relation"):
        v = getattr(query, name, None)
        if isinstance(v, Mapping):
            p = v.get("predicate") or v.get("relation")
            o = v.get("object") or v.get("object_canon") or v.get("value")
            if p and o:
                return (_fold(str(p)), _fold(str(o)))
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            return (_fold(str(v[0])), _fold(str(v[1])))
    intent = getattr(query, "intent", None)
    for src in (query, intent):
        if src is None:
            continue
        p = getattr(src, "predicate", None)
        o = getattr(src, "object_canon", None)
        if o is None:
            o = getattr(src, "object", None)
        if p and o:
            return (_fold(str(p)), _fold(str(o)))
    return None


def _item_assertion(item: Any):
    """``(predicate_fold, object_fold, by_canon)`` the item asserts —
    the verifier's structured-state reading (``state_facts``/
    ``events_v7``-shaped fields carried on the candidate)."""
    p = _find(item, ("predicate", "relation", "asserted_predicate"))
    o = _find(
        item, ("object_canon", "object", "asserted_object", "value_canon")
    )
    by = _find(
        item,
        (
            "asserted_by", "by_canon", "author_canon",
            "speaker_canon", "subject_canon", "speaker",
        ),
    )
    if p is None or o is None:
        return None
    return (
        _fold(str(p)),
        _fold(str(o)),
        str(by) if by is not None else None,
    )


def _verifier_report(
    query: QueryViewV7,
    ctx: Any,
    subject: Optional[str],
    items: list,
    measures: list,
) -> dict:
    """V8-12.04 presupposition verifier (research arm).

    ``contradiction`` is reported only when ALL three declared
    conditions hold: (i) exactly one asked subject canon; (ii) a
    predicate/object match for the asked relation asserted by a
    *different* canon inside the verdict-evidence candidates; (iii)
    zero verdict-evidence items by or about the asked subject match
    that predicate/object.  ``shipped`` records where the finding may
    land (``verdict.verifier_ship``: ``off``/``answerability``/
    ``status`` — the §12 exit decision's outcome).  Dev ``tpr``/``fpr``
    pass through from ``ctx.verifier_metrics`` /
    ``ctx.manifest['verdict.verifier']`` for the decision record.
    Items are never withheld under any outcome.
    """
    ship = _verifier_ship(ctx)
    rep = {
        "shipped": ship,
        "contradiction": False,
        "conditions": {
            "one_subject": False,
            "predicate_object_match": False,
            "subject_silent": False,
        },
    }
    if ctx is not None:
        mets = _get(ctx, "verifier_metrics")
        man = _get(ctx, "manifest")
        if mets is None and isinstance(man, Mapping):
            mets = man.get("verifier_metrics") or man.get(
                "verdict.verifier"
            )
        if isinstance(mets, Mapping):
            for k in ("tpr", "fpr", "dev_tpr", "dev_fpr", "n", "decision"):
                if mets.get(k) is not None:
                    rep[k] = mets[k]
    if not subject:
        return rep
    rep["conditions"]["one_subject"] = True
    rel = _asked_relation(query)
    if rel is None:
        return rep
    subject_fold = _fold(subject)
    pred, obj = rel
    diff_canon = False
    subject_hit = False
    for item, m in zip(items, measures):
        if not m.verdict_evidence:
            continue  # V8-12.06 — cut-lane candidates can't feed it
        a = _item_assertion(item)
        if a is None or a[0] != pred or a[1] != obj:
            continue
        by_fold = _fold(a[2]) if a[2] else None
        if by_fold and by_fold != subject_fold:
            diff_canon = True
        if by_fold == subject_fold or m.subject_role:
            subject_hit = True
    rep["conditions"]["predicate_object_match"] = diff_canon
    rep["conditions"]["subject_silent"] = not subject_hit
    rep["contradiction"] = all(rep["conditions"].values())
    return rep


# ---------------------------------------------------------------------------
# Group labeling — signals, never a deletion floor (V7-11.01)
# ---------------------------------------------------------------------------


def _group_label(
    agg: dict,
    *,
    has_ids: bool,
    n_ids: int,
    n_terms: int,
    n_ents: int,
) -> Tuple[SupportLabel, str]:
    """``(label, via)`` for one aggregated group."""
    cov = agg["term_cov"]
    ecov = agg["ent_cov"]
    real = agg["real"]
    measured = agg["measured"]

    if has_ids:
        if agg["id_level"]:
            # every asked identifier matched -> full support; a subset
            # -> partial (the unmatched remainder is reported as a
            # missing facet when the result abstains elsewhere)
            all_ids = agg["matched_count"] >= n_ids
            via = (
                "identifier_exact"
                if agg["id_level"] == "exact"
                else "identifier_normalized"
            )
            if all_ids:
                return SupportLabel.SUPPORTED, via
            return SupportLabel.PARTIAL, "identifier_partial"
        if (
            cov >= 0.5
            or (n_ents and ecov >= 1.0)
            or (not measured and real)
        ):
            return SupportLabel.PARTIAL, "context_without_identifier"
        return SupportLabel.WEAK, "no_identifier"

    if not n_terms and not n_ents:
        # Facetless query — nothing measurable to cover; a real lane or
        # window match is the strongest support the question admits.
        if real or agg["window"]:
            return SupportLabel.SUPPORTED, "lane_match"
        return SupportLabel.WEAK, "no_facets"

    if (
        cov >= LABEL_STRONG_COV
        or (cov >= LABEL_MID_COV and real)
        or (agg["window"] and cov >= 0.5)
        or (n_ents and ecov >= 1.0 and cov >= LABEL_SOME_COV)
        or (n_ents and not n_terms and ecov >= 1.0 and real)
    ):
        return SupportLabel.SUPPORTED, "coverage"

    if (
        cov >= LABEL_SOME_COV
        or (n_ents and ecov >= 0.5)
        or (n_ents and ecov >= 1.0)
        or agg["window"]
        or (not measured and real)
        or (real and (cov > 0 or ecov > 0))
    ):
        return SupportLabel.PARTIAL, "partial_coverage"

    return SupportLabel.WEAK, "weak_coverage"


def _ctx_peers(ctx: Any) -> Tuple[str, ...]:
    if ctx is None:
        return ()
    v = None
    for name in ("corpus_identifiers", "identifier_peers", "peers"):
        v = _get(ctx, name)
        if v:
            break
    return _str_list(v)


def classify_groups(
    items: Iterable[Any],
    query: QueryViewV7,
    ctx: Any = None,
) -> list:
    """Label each evidence group ``supported | partial | weak``.

    Groups are formed by the group-key precedence documented above and
    classified purely on measured support signals — identifier
    exact/normalized match (``ident_eq/v1``), entity-canon coverage,
    temporal-window match, real lane signals, and term coverage as a
    weak *labeling* signal. NO candidate is deleted: the returned list
    covers every input item exactly once, in input order (D7-04 closed
    by construction — the literal-coverage deletion floor is gone).

    ``GroupVerdict.trigger`` records the abstention trigger a group is
    implicated in — ``"a"`` when an identifier-query group lacks the
    identifier, ``"d"`` when it carries negative evidence under an
    ``abstain_likely`` intent — else ``None``. Per-group measurements
    live in ``detail`` (consumed by ``result_verdict``/coverage).
    """
    _MAPS_MEMO.clear()  # fresh call — never serve a stale item's maps
    item_list = list(items or ())
    qids, qterms, qents = _query_parts(query)
    window = _query_window(query)
    abstain = _abstain_likely(query)
    q_speaker = _premise_speaker(query, ctx)
    premise_on = _premise_flag(ctx)   # V8-12.01 — default off
    cut = _cut_lanes(ctx)             # V8-12.06 — deadline-cut lanes
    # §21.9 row 4 — "exactly one asked subject canon": the resolved
    # premise speaker, else the query's sole entity canon.
    subject = q_speaker
    if subject is None:
        _ents = [
            str(e) for e in getattr(query, "entity_canons", ()) or ()
            if str(e)
        ]
        if len(_ents) == 1:
            subject = _ents[0]
    subject_fold = _fold(subject) if subject else ""

    # peers for the hash-prefix uniqueness rule: every identifier-shaped
    # surface in the eligible candidates, plus any scope corpus supplied
    # through ctx. ctx peers additionally serve as the corpus-existence
    # check for trigger (a) — "no match in the eligible corpus" is a
    # statement about the corpus, not only about delivered items.
    ctx_pool = list(_ctx_peers(ctx))
    per_item_idents = [_item_idents(i) for i in item_list]
    peers = list(ctx_pool)
    for ids in per_item_idents:
        peers.extend(ids)
    peers = tuple(dict.fromkeys(peers))

    corpus_hits = set()
    if ctx_pool:
        for qid in qids:
            if any(ident_eq(qid, c, peers) for c in ctx_pool):
                corpus_hits.add(qid)

    groups: dict = {}
    order: list = []
    for idx, item in enumerate(item_list):
        m = _measure(
            item, idx, qids, qterms, qents, window, peers
        )
        # V8-12.06 — a member produced *solely* by declared
        # deadline-cut lanes stays deliverable but never enters verdict
        # evidence.  No lane record at all counts as evidence.
        if cut and m.lanes and m.lanes.issubset(cut):
            m.verdict_evidence = False
        if m.group_key not in groups:
            groups[m.group_key] = []
            order.append(m.group_key)
        groups[m.group_key].append(m)

    all_ms = [m for ms in groups.values() for m in ms]
    speaker_of = (
        _unit_speakers(ctx, (m.unit_id for m in all_ms))
        if subject else {}
    )
    mentions_of = (
        _unit_mentions(
            ctx,
            (m.unit_id for m in all_ms
             if m.verdict_evidence and m.support == "real"),
            subject,
        )
        if subject else frozenset()
    )
    # §21.9 row 4 — per-member subject roles on the effective speaker:
    # "authored by" (speaker canon) or "mentions" (entity postings,
    # text, or the generation-fenced entity_mentions probe).
    if subject_fold:
        for m in all_ms:
            if m.speaker is None:
                m.speaker = speaker_of.get(m.unit_id)
            roles = []
            if m.speaker and _fold(m.speaker) == subject_fold:
                roles.append("author")
            if (
                subject_fold in m.entity_named
                or subject_fold in m.covered_ents
                or (m.text_fold and subject_fold in m.text_fold)
                or m.unit_id in mentions_of
            ):
                roles.append("mention")
            m.subject_role = "+".join(roles) or None

    verdicts = []
    has_ids = bool(qids)
    for key in order:
        members = groups[key]
        # V8-12.06 — the verdict sees only verdict-evidence members;
        # cut-lane members still list under ``members`` for delivery.
        ev = [m for m in members if m.verdict_evidence]
        id_level = None
        matched_ids: list = []
        for m in ev:
            if m.id_level == "exact":
                id_level = "exact"
            elif m.id_level == "normalized" and id_level != "exact":
                id_level = "normalized"
            matched_ids.extend(m.matched_ids)
        covered_terms = frozenset().union(
            *(m.covered_terms for m in ev)
        ) if ev else frozenset()
        covered_ents = frozenset().union(
            *(m.covered_ents for m in ev)
        ) if ev else frozenset()
        term_cov = max(
            [len(covered_terms) / len(qterms) if qterms else 0.0]
            + [m.term_frac for m in ev]
        )
        ent_cov = max(
            [len(covered_ents) / len(qents) if qents else 0.0]
            + [m.ent_frac for m in ev]
        )
        member_speakers = {m.speaker for m in ev} - {None}
        member_speakers_fold = {_fold(s) for s in member_speakers}
        # Premise check (cat-5/adversarial): the question names exactly
        # one subject and no evidence member of this group is authored
        # by them — the group answers a different premise.  Requires a
        # majority of members to carry speaker attribution — sparse-
        # signal groups can't trip the check on a single attributed
        # member.  Advisory only unless ``premise_on`` (V8-12.01).
        n_attributed = sum(1 for m in ev if m.speaker)
        premise_mismatch = bool(
            subject
            and member_speakers
            and n_attributed * 2 >= len(ev)
            and subject_fold not in member_speakers_fold
        )
        pm_status = premise_mismatch and premise_on
        agg = {
            "id_level": id_level,
            "matched_count": len(set(matched_ids)) if qids else 0,
            "term_cov": term_cov,
            "ent_cov": ent_cov,
            "window": any(m.window for m in ev),
            "negative": any(m.negative for m in ev),
            "premise_mismatch": premise_mismatch,
            "real": any(m.support == "real" for m in ev),
            "soft": any(m.soft for m in ev),
            "measured": any(
                m.term_measured or m.ent_measured for m in ev
            ),
        }
        label, via = _group_label(
            agg, has_ids=has_ids, n_ids=len(qids),
            n_terms=len(qterms), n_ents=len(qents),
        )
        trigger = None
        if has_ids and id_level is None:
            trigger = TRIGGER_IDENTIFIER
        elif (abstain or pm_status) and (agg["negative"] or pm_status):
            trigger = TRIGGER_NEGATIVE
        detail = {
            "verdict": VERDICT_VERSION,
            "formula": FORMULA_STATUS_PROVISIONAL,
            "ident_eq": IDENT_EQ_VERSION,
            "via": via,
            "members": [m.unit_id for m in members],
            "evidence_members": [m.unit_id for m in ev],
            "real_support_members": tuple(
                m.unit_id for m in ev if m.support == "real"
            ),
            "member_details": tuple(
                {
                    "unit_id": m.unit_id,
                    "support": m.support,
                    "lanes": tuple(sorted(m.lanes)),
                    "verdict_evidence": m.verdict_evidence,
                    "speaker": m.speaker,
                    "subject_role": m.subject_role,
                }
                for m in members
            ),
            "id_match": id_level,
            "matched_ids": tuple(dict.fromkeys(matched_ids)),
            "corpus_matched_ids": tuple(sorted(corpus_hits)),
            "term_cov": term_cov,
            "covered_terms": tuple(sorted(covered_terms)),
            "entity_cov": ent_cov,
            "covered_entities": tuple(sorted(covered_ents)),
            "coverage_measured": agg["measured"],
            "window_match": agg["window"],
            "negative_evidence": agg["negative"],
            "premise_mismatch": (
                "speaker" if premise_mismatch else None
            ),
            "real_signal": agg["real"],
            "soft_signal": agg["soft"],
            "top_score": max((m.score for m in ev), default=0.0),
            "signals": tuple(sorted(
                set().union(*(m.signals for m in members))
            )) if members else (),
        }
        verdicts.append(
            GroupVerdict(
                group_key=key,
                label=label,
                trigger=trigger,
                detail=detail,
            )
        )
    # V8 shared verdict inputs — ctx is in scope only here;
    # ``result_verdict`` keeps its V7 signature ``(groups, query,
    # calibration)`` and reads these off the group details so flag,
    # deadline-cut and verifier state still reach it.
    verifier = _verifier_report(query, ctx, subject, item_list, all_ms)
    vinputs = {
        "premise_speaker": "on" if premise_on else "off",
        "deadline_cut_lanes": tuple(sorted(cut)),
        "subject_canon": subject,
        "verifier": verifier,
    }
    for gv in verdicts:
        gv.detail["verdict_inputs"] = vinputs
    return verdicts


# ---------------------------------------------------------------------------
# Result verdict — triggers (a)–(d) only (V7-11.01)
# ---------------------------------------------------------------------------


def calibration_status(calibration: Any) -> dict:
    """Report the ``support_calibration/v2`` state (V7-11.06).

    ``calibration`` is ``{threshold, separates, ...}`` (a mapping or a
    duck-typed object). The score trigger (c) is enabled iff the entry
    carries a finite ``threshold`` AND ``separates`` is truthy (the
    isotonic fit demonstrated AUROC ≥ 0.80 separation on the dev split,
    §32.16). ``None``/missing/invalid entries are ``unfitted`` — the
    trigger stays disabled and the state is reported.
    """
    if calibration is None:
        return {
            "status": "unfitted", "fitted": False, "threshold": None,
            "separates": False, "report": "calibration=unfitted",
        }
    getter = (
        calibration.get if isinstance(calibration, Mapping)
        else lambda n, d=None: getattr(calibration, n, d)
    )
    threshold = _num(getter("threshold"))
    separates = _truthy(getter("separates"))
    fitted = threshold is not None and separates
    status = "fitted" if fitted else "unfitted"
    report = {
        "status": status,
        "fitted": fitted,
        "threshold": threshold,
        "separates": separates,
        "report": f"calibration={status}",
    }
    for extra in ("precision", "recall", "n", "dev_digest", "profile",
                  "encoder_tier"):
        v = getter(extra)
        if v is not None:
            report[extra] = v
    return report


def _facet_missing(
    groups: list,
    qids: tuple,
    qterms: tuple,
    qents: tuple,
    window,
) -> dict:
    """Which query facets found no support (V7-11.08)."""
    covered_ids, covered_terms, covered_ents = set(), set(), set()
    win = False
    for g in groups:
        d = _get(g, "detail", None) or {}
        covered_ids.update(d.get("matched_ids") or ())
        covered_terms.update(d.get("covered_terms") or ())
        covered_ents.update(d.get("covered_entities") or ())
        win = win or bool(d.get("window_match"))
    facets: dict = {}
    un_ids = tuple(i for i in qids if i not in covered_ids)
    if un_ids:
        facets["identifier"] = un_ids
    un_terms = tuple(t for t in qterms if t not in covered_terms)
    if un_terms:
        facets["terms"] = un_terms
    un_ents = tuple(e for e in qents if e not in covered_ents)
    if un_ents:
        facets["entities"] = un_ents
    if window is not None and not win:
        facets["time_window"] = (f"{int(window[0])}..{int(window[1])}",)
    return facets


def _evidence_members(detail: Mapping) -> tuple:
    """Verdict-evidence member ids from a group detail (V8-12.06).
    Absent key → every member counts (pre-V8 details)."""
    ev = detail.get("evidence_members")
    if ev is None:
        return tuple(detail.get("members") or ())
    return tuple(ev)


def _verdict_inputs(group_list: list) -> dict:
    """The shared V8 verdict inputs ``classify_groups`` stamps on each
    group detail — how ``result_verdict`` sees the flag/deadline/
    verifier state without a ctx parameter."""
    for g in group_list:
        d = _get(g, "detail", None) or {}
        vi = d.get("verdict_inputs")
        if isinstance(vi, Mapping):
            return dict(vi)
    return {}


def _answerability(group_list: list, vinputs: Mapping) -> str:
    """§21.9 answerability (V8-12.03) — first matching row wins.

    Reads only verdict-evidence aggregates (V8-12.06): group labels,
    ``real_support_members`` and ``member_details`` are already
    computed over evidence members by ``classify_groups``.
    """
    details = [_get(g, "detail", None) or {} for g in group_list]
    # 1 — no eligible candidates delivered (verdict evidence).
    if not any(_evidence_members(d) for d in details):
        return ANSWERABILITY_NO_EVIDENCE
    # 2 — the V8-12.04 verifier is shipped to answerability and
    # reports a contradiction.
    verifier = vinputs.get("verifier")
    if (
        isinstance(verifier, Mapping)
        and verifier.get("shipped") == "answerability"
        and verifier.get("contradiction")
    ):
        return ANSWERABILITY_CONTRADICTED_PREMISE
    # 3 — a SUPPORTED group with at least one real-support member.
    for g, d in zip(group_list, details):
        if (
            _get(g, "label", None) == SupportLabel.SUPPORTED
            and d.get("real_support_members")
        ):
            return ANSWERABILITY_SUPPORTED
    # 4 — exactly one asked subject canon, and no real-support item
    # is authored by or mentions it.
    if vinputs.get("subject_canon"):
        if not any(
            md.get("subject_role")
            for d in details
            for md in (d.get("member_details") or ())
            if md.get("support") == "real"
            and md.get("verdict_evidence", True)
        ):
            return ANSWERABILITY_UNVERIFIED_PREMISE
    # 5 — at least one real-support item.
    if any(d.get("real_support_members") for d in details):
        return ANSWERABILITY_PARTIAL
    # 6 — only associative support.
    return ANSWERABILITY_WEAK_ONLY


class VerdictReport(tuple):
    """``(status, missing)`` two-tuple carrying the V8 verdict report.

    Unpacks exactly like the V7 return — ``status, missing =
    result_verdict(...)`` keeps working for existing callers — while
    exposing ``answerability`` (V8-12.03), ``status_trigger`` and the
    full ``triggers`` list (V8-20.03), and the coverage-shaped
    ``detail`` dict for the facade/SDK hand-off (V8-20.02)."""

    def __new__(
        cls,
        status: ResultStatus,
        missing: Optional[MissingDescriptor],
        *,
        answerability: str,
        status_trigger: Optional[str],
        triggers: Iterable[str],
        detail: dict,
    ):
        self = super().__new__(cls, (status, missing))
        self.answerability = answerability
        self.status_trigger = status_trigger
        self.triggers = tuple(triggers)
        self.detail = detail
        return self

    @property
    def status(self) -> ResultStatus:
        return self[0]

    @property
    def missing(self) -> Optional[MissingDescriptor]:
        return self[1]

    def as_dict(self) -> dict:
        """The V8-20.03 ``coverage.verdict`` block."""
        return self.detail


def result_verdict(
    groups: Iterable[GroupVerdict],
    query: QueryViewV7,
    calibration: Any = None,
    *,
    ctx: Any = None,
) -> VerdictReport:
    """Reduce the group verdicts to ``ready | insufficient`` plus the
    advisory ``answerability`` (§21.9).

    ``insufficient`` is reachable ONLY through the four declared
    structural triggers (V7-11.01, V8-12.02) — never through a
    coverage floor and never through premise doubt:

    - (b) zero verdict-evidence candidates — an empty eligible pool,
      or every delivered candidate produced solely by declared
      deadline-cut lanes (V8-12.06);
    - (a) the query carries identifiers and no evidence group recorded
      an exact/normalized identifier match;
    - (d) an ``abstain_likely`` intent with explicit negative evidence
      and no ``supported`` group; the speaker-premise leg fires only
      under ``verdict.premise_speaker=on`` (V8-12.01 — default off);
      a verifier reporting contradiction while ``ship="status"``
      (V8-12.04) adds the same trigger;
    - (c) a fitted, separating calibration whose threshold exceeds the
      top evidence-group score. Unfitted/absent calibration leaves the
      trigger disabled and is reported as ``calibration=unfitted``
      (V7-11.06).

    ``answerability`` carries premise doubt and weak support to the
    caller without changing status or delivery.  On ``insufficient``
    the ``MissingDescriptor`` names the unsupported facets; ``note``
    records the fired trigger(s) and the calibration state.
    ``status_trigger`` is the first trigger fired in evaluation order
    (b → a → d → c), ``None`` on ``ready``.
    """
    _MAPS_MEMO.clear()  # defensive — this path reads no item maps
    group_list = list(groups or ())
    qids, qterms, qents = _query_parts(query)
    window = _query_window(query)
    cal = calibration_status(calibration)
    vinputs = _verdict_inputs(group_list)
    if ctx is not None:
        premise_on = _premise_flag(ctx)
        cut = _cut_lanes(ctx)
        # The verifier's measured fields (conditions/contradiction) are
        # only computable inside ``classify_groups`` — this stage sees
        # no items.  Reuse the classify-time measurement carried in
        # ``verdict_inputs``; refresh the ctx-state fields (ship level,
        # dev metrics) so a verdict-time policy view is honored.
        verifier = vinputs.get("verifier")
        fresh = _verifier_report(
            query, ctx, vinputs.get("subject_canon"), [], []
        )
        if isinstance(verifier, Mapping):
            verifier = dict(verifier)
            verifier["shipped"] = fresh["shipped"]
            for k, v in fresh.items():
                if k not in ("shipped", "conditions", "contradiction"):
                    verifier[k] = v
        else:
            verifier = fresh
        if verifier is None:
            verifier = {"shipped": _verifier_ship(ctx),
                        "contradiction": False}
    else:
        premise_on = vinputs.get("premise_speaker") == "on"
        cut = frozenset(vinputs.get("deadline_cut_lanes") or ())
        verifier = vinputs.get("verifier")

    triggers: list = []
    if not any(
        _evidence_members(_get(g, "detail", None) or {})
        for g in group_list
    ):
        triggers.append(TRIGGER_EMPTY)
    else:
        if qids:
            # trigger (a): an asked identifier found no exact/normalized
            # match anywhere in the eligible candidates or the declared
            # corpus (ctx peers). A lane-level identifier signal without
            # attribution counts as existence evidence. Every unmatched
            # identifier is an unsupported facet (V7-11.08).
            matched_union: set = set()
            signal_match = False
            for g in group_list:
                d = _get(g, "detail", None) or {}
                matched_union.update(d.get("matched_ids") or ())
                matched_union.update(d.get("corpus_matched_ids") or ())
                if d.get("id_match") and not d.get("matched_ids"):
                    signal_match = True
            if any(
                q not in matched_union for q in qids
            ) and not signal_match:
                triggers.append(TRIGGER_IDENTIFIER)
        # Trigger (d) — negative evidence, gated per V8-12.01/12.04:
        # the speaker-premise form fires only with the flag ON; a
        # verifier contradiction fires only when shipped to ``status``.
        # The no-SUPPORTED guard bounds false positives — real support
        # is never relabeled.
        supported = any(
            _get(g, "label", None) == SupportLabel.SUPPORTED
            for g in group_list
        )
        negative = any(
            (_get(g, "detail", None) or {}).get("negative_evidence")
            for g in group_list
        )
        premise_mm = premise_on and any(
            (_get(g, "detail", None) or {}).get("premise_mismatch")
            for g in group_list
        )
        verifier_d = bool(
            isinstance(verifier, Mapping)
            and verifier.get("shipped") == "status"
            and verifier.get("contradiction")
        )
        if not supported and (
            verifier_d
            or premise_mm
            or (_abstain_likely(query) and negative)
        ):
            triggers.append(TRIGGER_NEGATIVE)
        if cal["fitted"]:
            top = max(
                _num((_get(g, "detail", None) or {}).get("top_score"))
                or 0.0
                for g in group_list
            )
            if top < cal["threshold"]:
                triggers.append(TRIGGER_CALIBRATED)

    answerability = _answerability(group_list, vinputs)
    status_trigger = triggers[0] if triggers else None

    if not triggers:
        report = VerdictReport(
            ResultStatus.READY,
            None,
            answerability=answerability,
            status_trigger=None,
            triggers=(),
            detail={},
        )
    else:
        facets = _facet_missing(group_list, qids, qterms, qents, window)
        note = (
            f"trigger={'+'.join(triggers)}; "
            f"calibration={cal['status']}; verdict={VERDICT_VERSION}"
        )
        report = VerdictReport(
            ResultStatus.INSUFFICIENT,
            MissingDescriptor(facets=facets, note=note),
            answerability=answerability,
            status_trigger=status_trigger,
            triggers=triggers,
            detail={},
        )
    # V8-20.03 coverage.verdict — emit only keys actually computed.
    detail = {
        "verdict": VERDICT_VERSION,
        "status": report.status.value,
        "status_trigger": report.status_trigger,
        "triggers": report.triggers,
        "answerability": answerability,
        "premise_speaker": "on" if premise_on else "off",
        "deadline_cut_lanes": tuple(sorted(cut)),
        "calibration": cal,
        "missing": (
            {
                "facets": report.missing.facets,
                "note": report.missing.note,
            }
            if report.missing is not None else None
        ),
    }
    if isinstance(verifier, Mapping):
        detail["verifier"] = dict(verifier)
    report.detail.update(detail)
    return report


def verdict_report(
    groups: Iterable[GroupVerdict],
    query: QueryViewV7,
    calibration: Any = None,
    *,
    ctx: Any = None,
) -> dict:
    """The V8-20.03 ``coverage.verdict`` block for ``result_verdict`` —
    ``status``, ``status_trigger``, ``answerability``,
    ``premise_speaker``, ``deadline_cut_lanes``."""
    return result_verdict(
        groups, query, calibration, ctx=ctx
    ).as_dict()


# ---------------------------------------------------------------------------
# Delivery selection — V7-11.03 strict semantics
# ---------------------------------------------------------------------------


def deliverable_groups(
    groups: Iterable[GroupVerdict],
    *,
    strict: bool = False,
    limit: Optional[int] = None,
) -> list:
    """The groups a caller may be shown (V7-11.03).

    Non-strict delivery carries ``weak``/``partial`` groups (labeled)
    whenever the result has fewer than ``limit`` supported groups; when
    supported groups alone reach the limit they are the delivery.
    ``strict=True`` restricts delivery to ``supported`` groups only.
    ``limit=None`` means no cap. Group labels and the result verdict are
    unaffected — this is a delivery filter, never a verdict input.
    """
    _MAPS_MEMO.clear()  # defensive — this path reads no item maps
    g = list(groups or ())
    if strict:
        out = [x for x in g if x.label == SupportLabel.SUPPORTED]
        return out[:limit] if limit is not None else out
    if limit is None:
        return g
    supported = [x for x in g if x.label == SupportLabel.SUPPORTED]
    if len(supported) >= limit:
        return supported[:limit]
    return g[:limit]


__all__ = [
    "IDENT_EQ_VERSION",
    "LABEL_MID_COV",
    "LABEL_SOME_COV",
    "LABEL_STRONG_COV",
    "TRIGGER_CALIBRATED",
    "TRIGGER_EMPTY",
    "TRIGGER_IDENTIFIER",
    "TRIGGER_NEGATIVE",
    "VERDICT_VERSION",
    "calibration_status",
    "classify_groups",
    "deliverable_groups",
    "ident_canon",
    "ident_eq",
    "ident_kind",
    "identifier_tokens",
    "result_verdict",
]
