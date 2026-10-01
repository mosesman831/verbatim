"""Deterministic V7 units projection (SPEC_V7 V7-30.01, V7-13.02–07, §30).

``derive_units`` is the V7 retrieval substrate: a *pure* function of one
persisted ``sources`` row + one ``source_revisions`` row + the add-time
conversational arguments stored with the source (``add_args``). No store,
no clock, no network — identical inputs reproduce byte-identical ``units``
rows, including ``unit_id`` (V7-30.01). Dropping the V7 tables and
re-running this projection at the same inputs and the same ``generation``
yields the same rows.

Input shapes (the real persisted rows — ``verbatim/storage/schema.py``):

- ``source_row``: ``{source_id, origin, external_id, source_kind,
  scope_id, speaker_id, created_us}`` — ``source_kind`` ∈
  ``user_message|assistant_message|tool_output|import|operator_record``.
- ``revision_row``: ``{source_id, revision, payload, payload_hmac,
  event_us, captured_us, timezone, provenance, metadata_json}`` —
  ``payload`` MUST be present (the units pin bytes into it);
  ``metadata_json`` is the persisted add-args channel and is merged under
  the explicit ``add_args`` (explicit keys win).
- ``add_args``: the §31 ``Memory.add`` conversational arguments —
  ``speaker``, ``role``, ``kind``, ``session_id``, ``session_started_at``,
  ``message_at``, ``occurred``/``occurred_*``, ``messages`` (list of
  dicts), ``user``/``assistant`` speaker identities, ``generation``.

Unit model (V7-13.03/04):

- ``turn`` — one per message when ``add_args["messages"]`` is present,
  else a single turn covering the whole payload. Turn byte pins locate
  the message content inside the payload by sequential byte search
  (order-stable; a content that never appears in the payload is emitted
  with ``byte_start/byte_end = NULL`` — the units-level form of
  ``unsupported_extraction``, V7-13.06: it can rank by metadata but can
  never supply a verbatim quote). A message may carry explicit
  ``byte_start``/``byte_end``; they are used only after verifying
  ``payload[s:e] == content`` — caller hints are never trusted blindly.
- ``sentence_window`` — turns/documents with more than
  ``window_sentences`` (default 3) sentences split into disjoint ≤3-
  sentence windows pinned into the parent payload. Children of their
  turn (``parent_unit_id``). Only emitted for pinned turns.
- ``session`` — the ordered turns of one session concatenated; pinned to
  ``[min member start, max member end)`` so the slice stays verbatim.
  Emitted when a session is resolvable: explicit ``session_id`` (arg or
  per-message — explicit always wins, V7-13.04), or implicit sessions
  formed by the >``session_gap_us`` (default 30 min) timestamp-gap rule
  over the messages' recorded times. Implicit ids are deterministic:
  ``sess:{source_id}:{k}`` in creation order. A lone unlabeled message
  forms no session.
- ``episode`` — deterministic lexical-cohesion segments (TextTiling
  class): adjacent-turn token-set cosine gaps, a boundary wherever a gap
  score is strictly below ``mean − pstdev`` of the session's gap scores.
  Sessions shorter than 3 turns never split; a no-valley session emits
  no episode units (a 1:1 duplicate of the session unit is noise).
  Episodes pin ``[first member start, last member end)`` and are
  ``parent_unit_id`` children of their session; member turns re-parent
  to their episode.

``seq`` semantics: ordinal within the unit's parent container — turns
are numbered in message order within their session (within the source
when sessionless), windows within their parent turn, episodes within
their session, sessions within the source. ``parent_unit_id`` wiring:
``session → episode → turn → sentence_window``; session parents are
``NULL``.

``recorded_at_us``: per-message ``message_at``/``at``/``event_us`` →
add-args ``message_at`` → revision ``event_us`` → ``captured_us``.
Aggregates take the earliest member time (``session_started_at`` wins
for the session it describes — an explicit one, or the source's only
implicit session). Timestamps accept int µs (or s/ms by
documented magnitude), or RFC3339 strings; ``*_us``-suffixed keys are
always raw µs. Unparseable → ``None``.

``occurred_*``: resolved by the caller (``temporal/v2`` is another
worker). Per-message ``occurred``/``occurred_*`` keys win; the top-level
``occurred`` applies to every turn that lacks its own. An
:class:`IntervalUs` or ``{start_us,end_us,precision,source}`` dict is
accepted; unresolved → ``NULL, NULL, "unknown", "unknown"``. Aggregates
take the covering ``[min start, max end]`` over members with known
bounds; precision/source are the uniform member value else ``"unknown"``.

``speaker_canon``: folded speaker — ``canon_fn`` param, else lazy
``verbatim.enrichment.entities_v2.canon`` (ImportError → documented
fallback ``strip().casefold()``). Aggregates keep a uniform member
canon, else ``NULL``.

``perspective`` (V7-13.07), precedence order:

1. hard kind class, message kind first then ``sources.source_kind`` —
   tool/action kinds (``tool_call``, ``tool_result``, ``tool_output``,
   ``file_diff``, ``test_result``, ``verification``, ``error``,
   ``recovery``, ``browser_state``, ``screenshot_ref``,
   ``file_snapshot_ref``, ``function*``) → ``agent_action``;
   ``system``/``system_event``/``operator*`` → ``system``;
   ``import``/``document``/``connector_item``/``legacy_import`` →
   ``document``. The source channel's hard class wins over a claimed
   speaker/role (a tool record is an action trace, not a user
   statement).
2. speaker — declared ``user``/``assistant`` identities from add_args
   win first, then marker vocabularies (``user|human|me|self|owner``,
   ``assistant|agent|ai|bot|model``, ``tool|function``, ``system``,
   ``third_party|other|participant``); a *named* speaker resolves to
   ``third_party`` when caller identities are declared, else falls
   through to the channel default (never a guessed impersonation —
   V6-01.04).
3. soft message kind — ``user*`` → ``user_stated``,
   ``assistant``/agent-authored kinds (``agent_note``, ``plan``,
   ``subgoal``, ``decision``, ``lesson``, ``handoff``, ``delegation``) →
   ``agent_stated``.
4. ``sources.source_kind`` soft class (``user_message`` →
   ``user_stated``, ``assistant_message`` → ``agent_stated``).
5. ``source_revisions.provenance`` (``direct_user`` → ``user_stated``,
   ``assistant_generated`` → ``agent_stated``, ``approved_tool`` →
   ``agent_action``, ``operator`` → ``system``, ``legacy_import`` →
   ``document``).
6. floor: ``document`` — unattributed bytes are documentary, never
   user-impersonating.

Aggregate units carry the uniform member perspective, else ``NULL``.

``unit_id`` is content-addressed: ``"u7:" + sha256(canonical_json)[:24]``
over ``{version, source_id, revision, generation, kind, seq,
parent_unit_id, session_id, speaker_canon, perspective, byte_start,
byte_end, recorded_at_us}`` — stable across runs, unique per (position,
generation); ``occurred_*`` is excluded so temporal re-resolution never
re-ids a unit. ``generation`` comes from the kwarg or
``add_args["generation"]`` (the projection fence, V7-30.02).

``segmentation``: ``"auto"`` (default — all applicable kinds),
``"turn"`` (turns + sessions, no windows/episodes), ``"document"``
(force single-document treatment even when ``messages`` is present),
``"raw"`` (one whole-payload turn only).

All §32-adjacent constants are ``provisional/v7-r0`` and named in
``UNITS_DERIVER_VERSION`` — changing any rule above changes the
derivation and MUST bump the tag (stored ids embed it).
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, safe_json_loads
from ..core.types_v7 import (
    IntervalUs,
    OccurredPrecision,
    OccurredSource,
    Perspective,
    UnitKind,
)

#: Version tag recorded in every minted unit_id (provisional per §32.0).
#: v1 → v1.1: single-turn sessions no longer mint a byte-identical
#: container unit, and ``when``/``date`` metadata parses into ``occurred``.
UNITS_DERIVER_VERSION = "units_v7/v1.1"
FORMULA_STATUS = "provisional/v7-r0"

#: V7-13.04 default session-break gap (30 minutes, configurable).
DEFAULT_SESSION_GAP_US = 30 * 60 * 1_000_000
#: V7-13.03 sentence-window width.
DEFAULT_WINDOW_SENTENCES = 3
#: Sessions shorter than this never episode-split — with ≤3 gaps the
#: strict-valley rule below cannot fire (thr = min of two scores).
EPISODE_MIN_TURNS = 4
#: Hard bound on rows one source may mint — overflow is a typed error,
#: never a silent drop (the store is the evidence; we do not fabricate a
#: partial projection as if it were complete).
MAX_UNITS_PER_SOURCE = 50_000

_SEGMENTATIONS = frozenset({"auto", "turn", "document", "raw"})

# ---------------------------------------------------------------------------
# perspective vocabularies (V7-13.07) — provisional/v7-r0
# ---------------------------------------------------------------------------

_USER_MARKERS = frozenset({"user", "human", "me", "self", "owner", "principal"})
_AGENT_MARKERS = frozenset(
    {"assistant", "agent", "ai", "bot", "model", "copilot"}
)
_TOOL_MARKERS = frozenset(
    {"tool", "function", "tool_call", "tool_result", "tool_output"}
)
_SYSTEM_MARKERS = frozenset({"system"})
_THIRD_MARKERS = frozenset({"third_party", "other", "participant"})

#: Kind classes that override speaker (a tool_result authored "by" the
#: user is still an action record; a document stays documentary).
_HARD_KIND_PERSPECTIVE = {
    "tool_call": Perspective.AGENT_ACTION,
    "tool_result": Perspective.AGENT_ACTION,
    "tool_output": Perspective.AGENT_ACTION,
    "function": Perspective.AGENT_ACTION,
    "function_call": Perspective.AGENT_ACTION,
    "file_diff": Perspective.AGENT_ACTION,
    "file_snapshot_ref": Perspective.AGENT_ACTION,
    "test_result": Perspective.AGENT_ACTION,
    "verification": Perspective.AGENT_ACTION,
    "error": Perspective.AGENT_ACTION,
    "recovery": Perspective.AGENT_ACTION,
    "browser_state": Perspective.AGENT_ACTION,
    "screenshot_ref": Perspective.AGENT_ACTION,
    "system": Perspective.SYSTEM,
    "system_event": Perspective.SYSTEM,
    "operator": Perspective.SYSTEM,
    "operator_record": Perspective.SYSTEM,
    "import": Perspective.DOCUMENT,
    "document": Perspective.DOCUMENT,
    "connector_item": Perspective.DOCUMENT,
    "legacy_import": Perspective.DOCUMENT,
}

#: Role-ish kinds that yield to a named speaker.
_SOFT_KIND_PERSPECTIVE = {
    "user": Perspective.USER_STATED,
    "user_message": Perspective.USER_STATED,
    "human": Perspective.USER_STATED,
    "assistant": Perspective.AGENT_STATED,
    "assistant_message": Perspective.AGENT_STATED,
    "agent": Perspective.AGENT_STATED,
    "agent_note": Perspective.AGENT_STATED,
    "plan": Perspective.AGENT_STATED,
    "subgoal": Perspective.AGENT_STATED,
    "decision": Perspective.AGENT_STATED,
    "lesson": Perspective.AGENT_STATED,
    "handoff": Perspective.AGENT_STATED,
    "delegation": Perspective.AGENT_STATED,
    "third_party": Perspective.THIRD_PARTY,
    "participant": Perspective.THIRD_PARTY,
    "other": Perspective.THIRD_PARTY,
}

_PROVENANCE_PERSPECTIVE = {
    "direct_user": Perspective.USER_STATED,
    "assistant_generated": Perspective.AGENT_STATED,
    "approved_tool": Perspective.AGENT_ACTION,
    "operator": Perspective.SYSTEM,
    "legacy_import": Perspective.DOCUMENT,
}

# ---------------------------------------------------------------------------
# time / occurred normalization
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"[+-]?\d+")


def _as_us(value: Any, *, assume_us: bool = False) -> Optional[int]:
    """Deterministic timestamp → µs. Ints/floats/numeric strings are read
    as µs when ``assume_us`` (``*_us`` keys) or by magnitude: ≥1e14 µs,
    ≥1e11 ms, else s. RFC3339 strings parse via ``fromisoformat`` (naive →
    UTC). Anything unparseable is ``None`` — never a guessed clock."""
    if value is None or isinstance(value, bool):
        return None
    v: Optional[int] = None
    if isinstance(value, int):
        v = value
    elif isinstance(value, float) and value.is_integer():
        v = int(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if _INT_RE.fullmatch(s):
            v = int(s)
        else:
            iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
            try:
                dt = datetime.fromisoformat(iso)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000)
    if v is None:
        return None
    if assume_us:
        return v
    a = abs(v)
    if a >= 10**14:
        return v
    if a >= 10**11:
        return v * 1_000
    return v * 1_000_000


def _first(mapping: dict, *keys: str) -> Any:
    for k in keys:
        v = mapping.get(k)
        if v is not None and v != "":
            return v
    return None


#: Human ``when``/``date`` strings — the LoCoMo corpus writes
#: ``"10:04 am on 19 December, 2023"`` and ``"8 May, 2023"``; neither is
#: RFC3339 so ``_as_us`` cannot see them.  Deterministic stdlib parse —
#: anything unparseable is ``None``, never a guessed clock.
_WHEN_MONTHS = {
    m: i
    for i, m in enumerate(
        ("january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november",
         "december"), start=1)
}
_WHEN_DMY = re.compile(
    r"^\s*(?:(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap])\.?m\.?\s+on\s+)?"
    r"(\d{1,2})\s+([A-Za-z]+)\s*,?\s*(\d{4})\s*$",
    re.IGNORECASE,
)
_WHEN_MDY = re.compile(
    r"^\s*(?:(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap])\.?m\.?\s+on\s+)?"
    r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\s*$",
    re.IGNORECASE,
)
_DAY_US = 86_400_000_000


def _parse_when_str(value: Any) -> Optional[tuple]:
    """``(start_us, end_us, precision)`` for a human date string, or
    ``None``.  Day-only forms are day-precision half-open intervals;
    time-bearing forms are instants (``start == end``)."""
    if not isinstance(value, str) or not value.strip():
        return None
    m = _WHEN_DMY.match(value) or _WHEN_MDY.match(value)
    if m is None:
        return None
    g = m.groups()
    if m.re is _WHEN_DMY:
        hh, mm, ss, ampm, dd, mon_s, yy = g
    else:
        hh, mm, ss, ampm, mon_s, dd, yy = g
    month = _WHEN_MONTHS.get(mon_s.lower())
    if month is None:
        return None
    try:
        if hh is None:
            dt = datetime(int(yy), month, int(dd), tzinfo=timezone.utc)
            start = int(dt.timestamp() * 1_000_000)
            return (start, start + _DAY_US, OccurredPrecision.DAY.value)
        hour = int(hh) % 12 + (12 if ampm.lower() == "p" else 0)
        dt = datetime(
            int(yy), month, int(dd), hour, int(mm), int(ss or 0),
            tzinfo=timezone.utc)
        start = int(dt.timestamp() * 1_000_000)
        return (start, start, OccurredPrecision.INSTANT.value)
    except (ValueError, OverflowError):
        return None


def _norm_occurred(raw: Any, flat: Optional[dict]) -> tuple:
    """Normalize an occurred input → ``(start_us, end_us, precision,
    source)``. Accepts :class:`IntervalUs`, a dict, or flat
    ``occurred_*`` keys. Unknown values fold to the spec vocabulary."""
    start = end = None
    prec = OccurredPrecision.UNKNOWN.value
    src = OccurredSource.UNKNOWN.value
    if isinstance(raw, IntervalUs):
        start, end = raw.start_us, raw.end_us
        prec = getattr(raw.precision, "value", None) or str(raw.precision)
        src = getattr(raw.source, "value", None) or str(raw.source)
    elif isinstance(raw, dict):
        start = _as_us(raw.get("start_us"), assume_us=True)
        if start is None:
            start = _as_us(raw.get("start"))
        end = _as_us(raw.get("end_us"), assume_us=True)
        if end is None:
            end = _as_us(raw.get("end"))
        if raw.get("precision") is not None:
            prec = str(raw["precision"])
        if raw.get("source") is not None:
            src = str(raw["source"])
    elif isinstance(raw, str):
        # A bare string occurred — RFC3339/epoch via ``_as_us`` first,
        # then the human ``when`` grammar ("10:04 am on 19 December,
        # 2023").  Parsed strings are declared values: explicit source.
        parsed = _as_us(raw)
        if parsed is not None:
            start = end = parsed
            prec = OccurredPrecision.INSTANT.value
            src = OccurredSource.EXPLICIT.value
        else:
            w = _parse_when_str(raw)
            if w is not None:
                start, end, prec = w
                src = OccurredSource.EXPLICIT.value
    if raw is None and isinstance(flat, dict):
        start = _as_us(flat.get("occurred_start_us"), assume_us=True)
        end = _as_us(flat.get("occurred_end_us"), assume_us=True)
        if flat.get("occurred_precision") is not None:
            prec = str(flat["occurred_precision"])
        if flat.get("occurred_source") is not None:
            src = str(flat["occurred_source"])
        if start is None and end is None:
            # ``metadata.when``/``date`` — the conversational timestamp
            # channel (V7-13 add-args).  Numeric/RFC3339 first, then the
            # human grammar; unparseable stays unknown, never guessed.
            wv = _first(flat, "when", "date")
            parsed = _as_us(wv) if wv is not None else None
            if parsed is not None:
                start = end = parsed
                prec = OccurredPrecision.INSTANT.value
                src = OccurredSource.EXPLICIT.value
            else:
                w = _parse_when_str(wv)
                if w is not None:
                    start, end, prec = w
                    src = OccurredSource.EXPLICIT.value
    if prec not in {p.value for p in OccurredPrecision}:
        prec = OccurredPrecision.UNKNOWN.value
    if src not in {s.value for s in OccurredSource}:
        src = OccurredSource.UNKNOWN.value
    # A half-known interval keeps its one real bound — the covering
    # aggregation (``_covering_occurred``) requires both.
    return (start, end, prec, src)


def _covering_occurred(occs: Iterable[tuple]) -> tuple:
    """[min start, max end] over members with known bounds; precision and
    source survive only when uniform, else ``"unknown"``."""
    known = [o for o in occs if o[0] is not None and o[1] is not None]
    if not known:
        return (
            None,
            None,
            OccurredPrecision.UNKNOWN.value,
            OccurredSource.UNKNOWN.value,
        )
    precs = {o[2] for o in known}
    srcs = {o[3] for o in known}
    return (
        min(o[0] for o in known),
        max(o[1] for o in known),
        next(iter(precs)) if len(precs) == 1 else OccurredPrecision.UNKNOWN.value,
        next(iter(srcs)) if len(srcs) == 1 else OccurredSource.UNKNOWN.value,
    )


# ---------------------------------------------------------------------------
# canonicalization / ids
# ---------------------------------------------------------------------------


def _canon_fallback(surface: str) -> str:
    """Documented fallback when ``entities_v2`` is absent: strip+casefold."""
    return surface.strip().casefold()


def _resolve_canon_fn(canon_fn: Optional[Callable[[str], str]]) -> Callable[[str], str]:
    if canon_fn is not None:
        return canon_fn
    try:
        from ..enrichment.entities_v2 import canon  # lazy: parallel worker
    except ImportError:
        return _canon_fallback
    return canon


def _unit_id(row: dict) -> str:
    """Content-addressed id over the structural+identity fields (the
    occurred fields deliberately excluded — see module docstring)."""
    canon = {
        "v": UNITS_DERIVER_VERSION,
        "source_id": row["source_id"],
        "revision": row["revision"],
        "generation": row["generation"],
        "kind": row["kind"],
        "seq": row["seq"],
        "parent_unit_id": row["parent_unit_id"],
        "session_id": row["session_id"],
        "speaker_canon": row["speaker_canon"],
        "perspective": row["perspective"],
        "byte_start": row["byte_start"],
        "byte_end": row["byte_end"],
        "recorded_at_us": row["recorded_at_us"],
    }
    return "u7:" + hashlib.sha256(
        json_dumps(canon).encode("utf-8")
    ).hexdigest()[:24]


# ---------------------------------------------------------------------------
# sentence segmentation — deterministic rule set (documented limits)
# ---------------------------------------------------------------------------

_ASCII_TERMINAL = frozenset(".!?…")
_CJK_TERMINAL = frozenset("。！？")
_CLOSERS = frozenset("\"'”’)]}»›")
_OPENERS = frozenset("\"'“‘([{<«‹")
_LIST_MARKER_RE = re.compile(r"(?:[-*•+]|#+|\d+[.)])\s*\S")

#: Deliberately small English abbreviation guard — a documented limit,
#: not a lexicon. The boundary rule below errs toward *keeping* text
#: whole (missing a boundary is safer than inventing one).
_ABBREV = frozenset(
    "mr mrs ms dr prof st jr sr vs etc fig eq cf no e.g i.e a.m p.m "
    "approx dept est inc ltd corp".split()
)


def _after_ok(text: str, j: int, hi: int) -> bool:
    """A terminator at ``j`` is a boundary iff followed by end-of-text or
    whitespace then an uppercase/digit (optionally behind an opener)."""
    k = j
    while k < hi and text[k] in " \t\n\r":
        k += 1
    if k >= hi:
        return True
    c = text[k]
    if c.isupper() or c.isdigit():
        return True
    if c in _OPENERS:
        k += 1
        while k < hi and text[k] in " \t":
            k += 1
        return k < hi and (text[k].isupper() or text[k].isdigit())
    return False


def _nl_ok(text: str, j: int, hi: int) -> bool:
    """A newline ends a segment iff the next line starts uppercase/digit
    or a list marker — chat lines and lists split, wrapped prose does not
    (documented limit: lowercase line continuations stay joined)."""
    k = j
    while k < hi and text[k] in " \t\r\n":
        k += 1
    if k >= hi:
        return False
    if _LIST_MARKER_RE.match(text, k):
        return True
    c = text[k]
    return c.isupper() or c.isdigit()


def _dot_boundary(text: str, i: int, hi: int) -> bool:
    """'.' guards: ellipses, decimals, known abbreviations, initials."""
    if i + 1 < hi and text[i + 1] == ".":
        return False
    if i > 0 and text[i - 1] == ".":
        return False
    if (
        i > 0
        and i + 1 < hi
        and text[i - 1].isdigit()
        and text[i + 1].isdigit()
    ):
        return False
    j = i - 1
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        j -= 1
    token = text[j + 1 : i]
    if token.lower() in _ABBREV:
        return False
    if len(token) == 1 and token.isupper():
        return False
    return _after_ok(text, i + 1, hi)


def _consume_close(text: str, j: int, hi: int) -> int:
    while j < hi and text[j] in _CLOSERS:
        j += 1
    return j


def _strip_span(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _sentence_spans(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Sentence char-spans inside ``text[lo:hi]`` — the V7 deterministic
    rule set: terminal punctuation + capital following, quote/paren aware
    (closing quotes/brackets fold into the sentence), a hard newline rule
    for chat/list text, and the unterminated tail kept so nothing is
    silently lost. Limits (documented): decimal/abbreviation/initial
    guards are a fixed small set; ``?!`` runs terminate as one unit; CJK
    terminators always break."""
    spans: list[tuple[int, int]] = []

    def _emit(a: int, b: int) -> None:
        a, b = _strip_span(text, a, b)
        if a < b:
            spans.append((a, b))

    start = lo
    i = lo
    while i < hi:
        ch = text[i]
        if ch == ".":
            if _dot_boundary(text, i, hi):
                end = _consume_close(text, i + 1, hi)
                _emit(start, end)
                start = end
                i = end
                continue
        elif ch in _CJK_TERMINAL:
            end = _consume_close(text, i + 1, hi)
            _emit(start, end)
            start = end
            i = end
            continue
        elif ch == "…":
            if _after_ok(text, i + 1, hi):
                end = _consume_close(text, i + 1, hi)
                _emit(start, end)
                start = end
                i = end
                continue
        elif ch in "!?":
            j = i
            while j < hi and text[j] in "!?":
                j += 1
            if _after_ok(text, j, hi):
                end = _consume_close(text, j, hi)
                _emit(start, end)
                start = end
                i = end
                continue
            i = j
            continue
        elif ch == "\n":
            if _nl_ok(text, i + 1, hi):
                _emit(start, i)
                start = i + 1
                i += 1
                continue
        i += 1
    if start < hi:
        _emit(start, hi)
    return spans


def _byte_offsets(text: str) -> list[int]:
    """Cumulative UTF-8 byte offset per char index (char-boundary exact)."""
    offsets = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        total += len(ch.encode("utf-8"))
        offsets[i + 1] = total
    return offsets


# ---------------------------------------------------------------------------
# episode segmentation — TextTiling-class lexical cohesion (documented)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _token_set(text: str) -> frozenset:
    return frozenset(_WORD_RE.findall(text.casefold()))


def _cosine_sets(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / math.sqrt(len(a) * len(b))


#: TextTiling block half-width: each gap compares the union of up to
#: ``EPISODE_BLOCK`` turns on either side (block cosine smooths the
#: sparse per-turn token overlap — the classic TextTiling choice).
EPISODE_BLOCK = 2


def _episode_segments(turn_texts: list[str]) -> list[tuple[int, int]]:
    """Contiguous turn ranges for one session. For each gap g between
    turn g-1 and g, cohesion is the token-set cosine between the union
    of the ≤``EPISODE_BLOCK`` preceding turns and the ≤``EPISODE_BLOCK``
    following turns. A boundary fires at g iff its cohesion is a
    *pronounced* valley: strictly below ``mean − pstdev`` AND at most
    half the session's mean gap score (the depth floor keeps marginal
    dips in flat sessions from splitting — under-segmenting is the safe
    failure, turn+session units already index the content).
    Deterministic: IEEE math, fixed order, strict inequalities mean
    plateaus never split. Returns a single covering segment when no
    valley exists."""
    n = len(turn_texts)
    if n < EPISODE_MIN_TURNS:
        return [(0, n)]
    sets = [_token_set(t) for t in turn_texts]
    w = EPISODE_BLOCK
    scores = []
    for g in range(1, n):
        left = frozenset().union(*sets[max(0, g - w) : g])
        right = frozenset().union(*sets[g : min(n, g + w)])
        scores.append(_cosine_sets(left, right))
    mu = sum(scores) / len(scores)
    var = sum((s - mu) ** 2 for s in scores) / len(scores)
    thr = mu - math.sqrt(var)
    depth_floor = mu / 2
    bounds = [
        i
        for i in range(1, n)
        if scores[i - 1] < thr and scores[i - 1] <= depth_floor
    ]
    if not bounds:
        return [(0, n)]
    segs: list[tuple[int, int]] = []
    prev = 0
    for b in bounds:
        segs.append((prev, b))
        prev = b
    segs.append((prev, n))
    return segs


# ---------------------------------------------------------------------------
# turn extraction
# ---------------------------------------------------------------------------

_ASCII_WORD = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _word_edge(payload: bytes, pos: int) -> bool:
    """True when the byte at ``pos`` is absent or non-word (ASCII check —
    a documented limit; multi-byte edges are treated as boundaries)."""
    if pos < 0 or pos >= len(payload):
        return True
    return payload[pos] not in _ASCII_WORD


def _locate(payload: bytes, content: bytes, cursor: int) -> Optional[tuple[int, int]]:
    """Sequential byte search with word-edge guard. Scans forward from
    ``cursor``, then wraps to 0 (out-of-order serializations). A hit
    counts only when the bytes adjacent to a word-edged content are not
    ASCII word chars — 'yes' never pins inside 'yesterday'."""
    if not content:
        return None
    n = len(content)
    edge_lo = content[0] in _ASCII_WORD
    edge_hi = content[-1] in _ASCII_WORD
    for lo in (cursor, 0):
        pos = payload.find(content, lo)
        while pos != -1:
            if (not edge_lo or _word_edge(payload, pos - 1)) and (
                not edge_hi or _word_edge(payload, pos + n)
            ):
                return (pos, pos + n)
            pos = payload.find(content, pos + 1)
    return None


def _msg_get(msg: dict, *keys: str) -> Any:
    for k in keys:
        v = msg.get(k)
        if v is not None and v != "":
            return v
    return None


def _turn_record(
    msg: Any,
    payload: bytes,
    cursor: int,
) -> tuple[dict, int]:
    """Normalize one ``messages`` element into a turn record; returns
    ``(record, new_cursor)``. ``record["text"] is None`` marks a skip
    (empty/absent content — nothing to evidence)."""
    if isinstance(msg, dict):
        content = _msg_get(msg, "content", "text", "message")
        speaker = _msg_get(msg, "speaker", "speaker_id", "name", "author")
        role = _msg_get(msg, "role", "kind", "type")
        # ``*_us`` keys are raw microseconds; human-facing keys get the
        # documented magnitude/RFC3339 heuristic.
        rec = _msg_get(msg, "recorded_at_us", "event_us")
        if rec is None:
            rec = _msg_get(msg, "message_at", "timestamp", "at", "ts")
            rec = _as_us(rec)
        else:
            rec = _as_us(rec, assume_us=True)
        sess = _msg_get(msg, "session_id", "session")
        occ_raw = msg.get("occurred")
        explicit_pin = (msg.get("byte_start"), msg.get("byte_end"))
    else:
        content = msg if isinstance(msg, str) else None
        speaker = role = rec = sess = occ_raw = None
        explicit_pin = (None, None)

    if content is None or content == "":
        return ({"text": None}, cursor)
    if not isinstance(content, str):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "message content must be a string",
        )

    cbytes = content.encode("utf-8")
    bs_i, be_i = explicit_pin
    if (
        isinstance(bs_i, int)
        and isinstance(be_i, int)
        and not isinstance(bs_i, bool)
        and not isinstance(be_i, bool)
        and 0 <= bs_i < be_i <= len(payload)
        and payload[bs_i:be_i] == cbytes
    ):
        # Caller-supplied pins are honored only after byte verification.
        bs, be = bs_i, be_i
    else:
        loc = _locate(payload, cbytes, cursor)
        bs, be = loc if loc else (None, None)
    if bs is not None:
        cursor = max(cursor, be)

    return (
        {
            "text": content,
            "speaker_raw": speaker,
            "role": role,
            "recorded_us": rec,
            "session_hint": sess,
            "occurred_raw": occ_raw,
            "occurred_flat": msg if isinstance(msg, dict) else None,
            "bs": bs,
            "be": be,
        },
        cursor,
    )


def _arg_us(args: dict, *keys: str) -> Optional[int]:
    """Read a timestamp from add-args: ``*_us`` keys are raw µs, the rest
    go through the documented magnitude/RFC3339 heuristic."""
    for k in keys:
        v = args.get(k)
        if v is None or v == "":
            continue
        if k.endswith("_us"):
            out = _as_us(v, assume_us=True)
        else:
            out = _as_us(v)
        if out is not None:
            return out
    return None


# ---------------------------------------------------------------------------
# perspective
# ---------------------------------------------------------------------------


def _fold(value: Any) -> str:
    return str(value).strip().casefold() if value is not None else ""


def _classify_perspective(
    kind: Any,
    speaker: Any,
    source_kind: Any,
    provenance: Any,
    user_spk: Any,
    agent_spk: Any,
) -> str:
    """V7-13.07 ladder — see module docstring for the precedence contract."""
    k = _fold(kind)
    if k in _HARD_KIND_PERSPECTIVE:
        return _HARD_KIND_PERSPECTIVE[k].value
    sk_hard = _fold(source_kind)
    if sk_hard in _HARD_KIND_PERSPECTIVE:
        # The source channel's hard class (tool_output, import,
        # operator_record) wins over any claimed speaker/role — a tool
        # record is an agent action, not a user statement.
        return _HARD_KIND_PERSPECTIVE[sk_hard].value

    spk = _fold(speaker)
    if spk:
        u = _fold(user_spk)
        a = _fold(agent_spk)
        if u and spk == u:
            return Perspective.USER_STATED.value
        if a and spk == a:
            return Perspective.AGENT_STATED.value
        if spk in _USER_MARKERS:
            return Perspective.USER_STATED.value
        if spk in _AGENT_MARKERS:
            return Perspective.AGENT_STATED.value
        if spk in _TOOL_MARKERS:
            return Perspective.AGENT_ACTION.value
        if spk in _SYSTEM_MARKERS:
            return Perspective.SYSTEM.value
        if spk in _THIRD_MARKERS:
            return Perspective.THIRD_PARTY.value
        # Named speaker, no marker: declared identities make it a third
        # party; otherwise the channel (kind/source_kind) decides.
        if u or a:
            return Perspective.THIRD_PARTY.value

    if k in _SOFT_KIND_PERSPECTIVE:
        return _SOFT_KIND_PERSPECTIVE[k].value
    sk = _fold(source_kind)
    if sk in _SOFT_KIND_PERSPECTIVE:
        return _SOFT_KIND_PERSPECTIVE[sk].value
    pv = _fold(provenance)
    if pv in _PROVENANCE_PERSPECTIVE:
        return _PROVENANCE_PERSPECTIVE[pv].value
    return Perspective.DOCUMENT.value


# ---------------------------------------------------------------------------
# derive_units
# ---------------------------------------------------------------------------


def derive_units(
    source_row: dict,
    revision_row: dict,
    add_args: dict,
    *,
    segmentation: str = "auto",
    canon_fn: Optional[Callable[[str], str]] = None,
    generation: Optional[int] = None,
    session_gap_us: int = DEFAULT_SESSION_GAP_US,
    window_sentences: int = DEFAULT_WINDOW_SENTENCES,
    max_units: int = MAX_UNITS_PER_SOURCE,
) -> list[dict]:
    """Derive §30 ``units`` rows from one persisted source+revision pair
    plus its stored add-args. Pure and deterministic (V7-30.01); see the
    module docstring for the full field/pin/id contract."""
    if segmentation not in _SEGMENTATIONS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"segmentation must be one of {sorted(_SEGMENTATIONS)}",
        )
    if not isinstance(source_row, dict) or not isinstance(revision_row, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "source_row/revision_row must be dicts"
        )
    source_id = source_row.get("source_id")
    if source_id is None or source_id != revision_row.get("source_id", source_id):
        raise VerbatimError(
            ErrorCode.VALIDATION, "source/revision source_id mismatch"
        )
    revision = revision_row.get("revision")
    if revision is None:
        revision = 1

    raw_payload = revision_row.get("payload")
    if isinstance(raw_payload, str):
        payload = raw_payload.encode("utf-8")
    elif isinstance(raw_payload, (bytes, bytearray, memoryview)):
        payload = bytes(raw_payload)
    else:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "derive_units requires revision payload bytes",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, "source payload is not valid UTF-8"
        ) from exc

    # Merge persisted add-args (revision metadata_json) under the call.
    meta = revision_row.get("metadata_json")
    if isinstance(meta, str):
        meta = safe_json_loads(meta) if meta.strip() else {}
    if not isinstance(meta, dict):
        meta = {}
    args = dict(meta)
    if add_args:
        if not isinstance(add_args, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "add_args must be a dict"
            )
        args.update(add_args)

    if generation is None:
        gen_raw = args.get("generation")
        generation = (
            int(gen_raw)
            if isinstance(gen_raw, int) and not isinstance(gen_raw, bool)
            else None
        )

    canon = _resolve_canon_fn(canon_fn)
    user_spk = _first(args, "user", "user_speaker", "principal", "principal_id")
    agent_spk = _first(
        args, "assistant", "assistant_speaker", "agent", "agent_id"
    )
    args_session = _first(args, "session_id", "session")
    session_started = _arg_us(
        args, "session_started_at", "session_started_us"
    )
    top_occ_raw = args.get("occurred")
    top_speaker = _first(args, "speaker", "speaker_id")
    top_role = _first(args, "role", "kind", "type", "envelope_kind")
    top_at = _arg_us(args, "message_at", "recorded_at_us", "event_us")
    src_kind = source_row.get("source_kind")
    provenance = revision_row.get("provenance")

    # ---- turn records -----------------------------------------------------
    # ``messages`` drives multi-turn sources (``auto``/``turn`` modes);
    # ``document``/``raw`` and messages-less sources take the whole
    # payload as one turn.
    raw_messages = args.get("messages") if segmentation in ("auto", "turn") else None
    if isinstance(raw_messages, (dict, str)):
        raw_messages = [raw_messages]
    if not isinstance(raw_messages, (list, tuple)):
        raw_messages = None

    turns: list[dict] = []
    if raw_messages:
        cursor = 0
        for msg in raw_messages:
            rec, cursor = _turn_record(msg, payload, cursor)
            if rec.get("text") is not None:
                turns.append(rec)
    else:
        speaker = top_speaker
        if speaker is None:
            speaker = source_row.get("speaker_id")
        turns = [
            {
                "text": text,
                "speaker_raw": speaker,
                "role": top_role,
                "recorded_us": top_at,
                "session_hint": args_session,
                "occurred_raw": top_occ_raw,
                "occurred_flat": args,
                "bs": 0,
                "be": len(payload),
            }
        ]

    if not turns:
        return []

    # recorded_at fallback chain per turn.
    rev_event = revision_row.get("event_us")
    rev_cap = revision_row.get("captured_us")
    for t in turns:
        if t["recorded_us"] is None:
            t["recorded_us"] = top_at
        if t["recorded_us"] is None:
            t["recorded_us"] = (
                int(rev_event)
                if isinstance(rev_event, int)
                else (int(rev_cap) if isinstance(rev_cap, int) else None)
            )
        # speaker/kind fallback ladder.
        if t["speaker_raw"] is None:
            t["speaker_raw"] = (
                top_speaker
                if top_speaker is not None
                else source_row.get("speaker_id")
            )
        if t["role"] is None:
            t["role"] = top_role
        t["speaker_canon"] = (
            canon(str(t["speaker_raw"])) if t["speaker_raw"] is not None else None
        )
        t["perspective"] = _classify_perspective(
            t["role"],
            t["speaker_raw"],
            src_kind,
            provenance,
            user_spk,
            agent_spk,
        )
        t["occurred"] = _norm_occurred(t["occurred_raw"], t["occurred_flat"])
        if (
            t["occurred_raw"] is None
            and t["occurred"][0] is None
            and t["occurred"][1] is None
        ):
            # Top-level occurred applies to any turn lacking its own.
            t["occurred"] = _norm_occurred(top_occ_raw, args)

    # ---- sessionize --------------------------------------------------------
    # explicit session ids win (V7-13.04); otherwise the >30min gap rule
    # over recorded times breaks implicit sessions. A lone unlabeled
    # message forms no session.
    sessions: list[dict] = []  # {id, turns:[idx], implicit:bool}
    by_sid: dict[str, dict] = {}
    implicit_n = 0
    current: Optional[dict] = None
    prev_ts: Optional[int] = None

    def _new_implicit() -> dict:
        nonlocal implicit_n, current
        s = {
            "id": f"sess:{source_id}:{implicit_n}",
            "turns": [],
            "implicit": True,
        }
        implicit_n += 1
        sessions.append(s)
        return s

    multi = len(turns) > 1
    for idx, t in enumerate(turns):
        sid = t["session_hint"] or args_session
        if sid is not None:
            sid = str(sid)
            sess = by_sid.get(sid)
            if sess is None:
                sess = {"id": sid, "turns": [], "implicit": False}
                by_sid[sid] = sess
                sessions.append(sess)
            sess["turns"].append(idx)
            current = None  # explicit labels break the implicit run
        else:
            ts = t["recorded_us"]
            broke = (
                current is None
                or (
                    ts is not None
                    and prev_ts is not None
                    and abs(ts - prev_ts) > session_gap_us
                )
            )
            if current is None or broke:
                current = _new_implicit()
            current["turns"].append(idx)
        if t["recorded_us"] is not None:
            prev_ts = t["recorded_us"]

    # A lone unlabeled message forms no session; neither does a
    # sessionless single-turn source.
    if not multi:
        if args_session is None and (not turns or turns[0]["session_hint"] is None):
            sessions = []
            by_sid.clear()
            current = None

    emit_sessions = segmentation in ("auto", "turn", "document") and bool(sessions)

    # ---- episodes per session ----------------------------------------------
    sess_eps: list[list[tuple[int, int]]] = []
    if segmentation == "auto":
        for sess in sessions:
            texts = [turns[i]["text"] for i in sess["turns"]]
            segs = _episode_segments(texts)
            sess_eps.append(segs if len(segs) >= 2 else [])
    else:
        sess_eps = [[] for _ in sessions]

    # ---- emit --------------------------------------------------------------
    scope_id = source_row.get("scope_id")
    out: list[dict] = []

    def _mk(
        kind: str,
        seq: int,
        parent: Optional[str],
        session_id: Optional[str],
        speaker_canon: Optional[str],
        perspective: Optional[str],
        rec: Optional[int],
        occ: tuple,
        bs: Optional[int],
        be: Optional[int],
    ) -> dict:
        row = {
            "unit_id": None,
            "source_id": source_id,
            "revision": revision,
            "scope_id": scope_id,
            "kind": kind,
            "parent_unit_id": parent,
            "session_id": session_id,
            "seq": seq,
            "speaker_canon": speaker_canon,
            "perspective": perspective,
            "recorded_at_us": rec,
            "occurred_start_us": occ[0],
            "occurred_end_us": occ[1],
            "occurred_precision": occ[2],
            "occurred_source": occ[3],
            "byte_start": bs,
            "byte_end": be,
            "generation": generation,
        }
        row["unit_id"] = _unit_id(row)
        out.append(row)
        if len(out) > max_units:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"source {source_id}@{revision} exceeds max_units "
                f"({max_units})",
            )
        return row

    def _agg_span(idxs: list[int]) -> tuple[Optional[int], Optional[int]]:
        pinned = [
            (turns[i]["bs"], turns[i]["be"])
            for i in idxs
            if turns[i]["bs"] is not None
        ]
        if not pinned:
            return (None, None)
        return (min(p[0] for p in pinned), max(p[1] for p in pinned))

    def _agg_persp(idxs: list[int]) -> Optional[str]:
        ps = {turns[i]["perspective"] for i in idxs}
        return next(iter(ps)) if len(ps) == 1 else None

    def _agg_speaker(idxs: list[int]) -> Optional[str]:
        ss = {turns[i]["speaker_canon"] for i in idxs}
        return next(iter(ss)) if len(ss) == 1 else None

    def _agg_rec(idxs: list[int]) -> Optional[int]:
        rs = [
            turns[i]["recorded_us"]
            for i in idxs
            if turns[i]["recorded_us"] is not None
        ]
        return min(rs) if rs else None

    def _byte_off_for(t: dict, char_pos: int) -> int:
        off = t.get("_byte_off")
        if off is None:
            off = _byte_offsets(t["text"])
            t["_byte_off"] = off
        return off[char_pos]

    def _emit_turn(
        idx: int, seq: int, parent: Optional[str], sess_id: Optional[str]
    ) -> dict:
        t = turns[idx]
        row = _mk(
            UnitKind.TURN.value,
            seq,
            parent,
            sess_id,
            t["speaker_canon"],
            t["perspective"],
            t["recorded_us"],
            t["occurred"],
            t["bs"],
            t["be"],
        )
        # sentence windows — only under a pinned parent turn.
        if segmentation in ("auto", "document") and t["bs"] is not None:
            spans = _sentence_spans(t["text"], 0, len(t["text"]))
            if len(spans) > window_sentences:
                for w in range(0, len(spans), window_sentences):
                    grp = spans[w : w + window_sentences]
                    wa = t["bs"] + _byte_off_for(t, grp[0][0])
                    wb = t["bs"] + _byte_off_for(t, grp[-1][1])
                    _mk(
                        UnitKind.SENTENCE_WINDOW.value,
                        w // window_sentences,
                        row["unit_id"],
                        sess_id,
                        t["speaker_canon"],
                        t["perspective"],
                        t["recorded_us"],
                        t["occurred"],
                        wa,
                        wb,
                    )
        return row

    if segmentation == "raw":
        _mk(
            UnitKind.TURN.value,
            0,
            None,
            None,
            turns[0]["speaker_canon"],
            turns[0]["perspective"],
            turns[0]["recorded_us"],
            turns[0]["occurred"],
            0,
            len(payload),
        )
        return out

    if not emit_sessions:
        for i, t in enumerate(turns):
            _emit_turn(i, i, None, None)
        return out

    # session-emission path — sessions in resolution order; turns keep
    # per-session seq in message order; episode rows land just before
    # their first member turn.
    turn_seq_in_sess: dict[int, int] = {}
    for sess in sessions:
        for j, idx in enumerate(sess["turns"]):
            turn_seq_in_sess[idx] = j

    # Emit each session: session unit, then member turns (with episode
    # rows inserted before their first turn).
    for s_i, sess in enumerate(sessions):
        idxs = sess["turns"]
        sbs, sbe = _agg_span(idxs)
        rec = (
            session_started
            if session_started is not None
            and (not sess["implicit"] or len(sessions) == 1)
            else _agg_rec(idxs)
        )
        srow = _mk(
            UnitKind.SESSION.value,
            s_i,
            None,
            sess["id"],
            _agg_speaker(idxs),
            _agg_persp(idxs),
            rec,
            _covering_occurred(turns[i]["occurred"] for i in idxs),
            sbs,
            sbe,
        )
        eps = sess_eps[s_i] if s_i < len(sess_eps) else []
        if len(idxs) == 1 and not eps:
            # A single-turn session restates its member turn byte-for-byte
            # — it inflates df and double-votes in fusion without adding
            # evidence.  The turn keeps ``session_id`` for grouping; the
            # container row is not minted.
            out.pop()
            srow = None
        # map turn idx -> episode ordinal
        ep_of: dict[int, int] = {}
        for e_i, (ea, eb) in enumerate(eps):
            for idx in idxs[ea:eb]:
                ep_of[idx] = e_i
        last_ep = -1
        ep_rows: list[dict] = []
        for idx in idxs:
            e_i = ep_of.get(idx)
            if e_i is not None and e_i != last_ep:
                ea, eb = eps[e_i]
                memb = idxs[ea:eb]
                ebs, ebe = _agg_span(memb)
                erow = _mk(
                    UnitKind.EPISODE.value,
                    e_i,
                    srow["unit_id"],
                    sess["id"],
                    _agg_speaker(memb),
                    _agg_persp(memb),
                    _agg_rec(memb),
                    _covering_occurred(turns[m]["occurred"] for m in memb),
                    ebs,
                    ebe,
                )
                ep_rows.append(erow)
                last_ep = e_i
            parent = (
                ep_rows[e_i]["unit_id"]
                if e_i is not None
                else (srow["unit_id"] if srow is not None else None)
            )
            _emit_turn(idx, turn_seq_in_sess[idx], parent, sess["id"])

    # sessionless turns (only reachable when sessions exist for some
    # turns — mixed explicit/implicit never produces sessionless turns,
    # but a ``document``-mode source with messages could).
    assigned = {i for sess in sessions for i in sess["turns"]}
    for i in range(len(turns)):
        if i not in assigned:
            _emit_turn(i, i, None, None)

    return out


__all__ = [
    "DEFAULT_SESSION_GAP_US",
    "DEFAULT_WINDOW_SENTENCES",
    "EPISODE_MIN_TURNS",
    "FORMULA_STATUS",
    "MAX_UNITS_PER_SOURCE",
    "UNITS_DERIVER_VERSION",
    "derive_units",
]
