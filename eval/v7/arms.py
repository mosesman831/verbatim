"""Track R retrieval arms — SPEC_V7 §22.3 (V7-22.12/22.22), `track_r/v7-a`.

The retrieval arms behind one protocol:

* :class:`VerbatimArm` — the real public ``verbatim.Memory`` facade:
  every item goes through ``Memory.add`` (``infer`` per arm config),
  settle is the durable-queue drain + ``wait_ready`` readiness frontier,
  and queries run ``Memory.search``.  Delivered hits are mapped back to
  corpus item refs through ``source_id`` (claim hits resolve through
  ``claim_evidence`` → ``spans`` — the same scoring-side id mapping the
  v5 harness uses; gold is never visible to the arm).  When the task
  supplies a question-time anchor (``as_of``/``query_time``/
  ``question_time``/``question_date`` keys — corpus metadata, never
  gold) the arm passes it to ``Memory.search(as_of=…)``; LoCoMo-style
  corpora without a per-question clock anchor on the task group's
  final-session ``when`` instead (V8-09.02).

* :class:`VerbatimClaimsArm` — ``verbatim_claims``, the V8-13.01
  claims-plane variant: same path with ``admission.require_review``
  pinned ``False`` for the arm only (product default ``True``
  untouched); the override is recorded in the ingest report's
  ``config_overrides`` block (scenarios K13/K74).

* :class:`FlatBM25Arm` — self-contained Okapi BM25 (k1=1.2, b=0.75)
  over the identical item document strings.  The honest reference arm:
  no store, no verdict, no abstention machinery.

* :class:`FTS5Arm` — plain SQLite FTS5 ``bm25()`` over the same texts.
  The naive baseline.

Corpus protocol (duck-typed — this module MUST NOT import
``eval/v7/corpora.py``, which is a concurrent worker's file):

* items via ``corpus.items`` (attribute or zero-arg method), the
  ``"items"`` key of a dict corpus, or :class:`DictCorpus`;
* tasks via ``corpus.tasks``, the ``"tasks"`` key, or ``iter(corpus)``;
* item fields: ``id``/``ref``/``item_id``/``dia_id``, ``text``,
  ``speaker``, ``session_id``, ``when``/``timestamp``, and optionally a
  ``render_text()`` producing the full indexable document (the corpora
  adapter's probe-style ``[when] speaker: text`` form — mirrored here
  for plain dicts so *every arm indexes byte-identical documents*);
* task fields an arm may read: ``task_id``, ``query``, ``category``
  only — gold stays scorer-side (V7-22.02); :func:`arm_task` is the
  handoff view.

``arm.query(task, k)`` returns a :class:`QueryOutcome` — it *is* the
ranked ref list (iterate/index it) plus the attribution diagnostics
V7-22.21 requires (``surfaced`` pool, verdict/lane detail).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import string
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence,
    Tuple,
)

#: Tokenizer shared by the flat BM25 arm (probe convention — ``\w+``
#: over the lowercased document).
TOK = re.compile(r"\w+", re.U)

#: Verbatim arm diagnostic pool: the measured ``search(limit=k)`` call
#: is followed by an unmeasured ``search(limit=min(pool, _MAX_LIMIT))``
#: so ``surfaced`` covers the post-verdict deliverable pool below k.
#: Disclosed in the ingest report — latency is always the measured call.
DEFAULT_POOL_LIMIT = 64


# ---------------------------------------------------------------------------
# duck-typed accessors (dicts AND attribute objects)
# ---------------------------------------------------------------------------


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """First present, non-None field across dict keys and attributes."""
    if isinstance(obj, Mapping):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _seq(corpus: Any, key: str) -> Optional[List[Any]]:
    """Resolve ``corpus.<key>`` / ``corpus.<key>()`` / ``iter_<key>()`` /
    ``corpus[key]`` into a list — the corpus protocol's tolerant read."""
    if isinstance(corpus, Mapping):
        v = corpus.get(key)
        return None if v is None else list(v)
    it = getattr(corpus, f"iter_{key}", None)
    if callable(it):
        return list(it())
    v = getattr(corpus, key, None)
    if v is None:
        return None
    if callable(v):
        try:
            return list(v())
        except TypeError:
            return None
    return list(v.values()) if isinstance(v, Mapping) else list(v)


def corpus_items(corpus: Any) -> List[Any]:
    """The corpus's addable items, in corpus order."""
    out = _seq(corpus, "items")
    if out is None:
        raise TypeError(
            f"corpus {type(corpus).__name__} exposes no items "
            "(.items attribute/method or 'items' key expected)"
        )
    return out


def corpus_tasks(corpus: Any) -> List[Any]:
    """The corpus's scored tasks, in corpus order."""
    out = _seq(corpus, "tasks")
    if out is not None:
        return out
    # Corpus.__iter__ yields tasks (eval/v7/corpora.py convention).
    try:
        return list(iter(corpus))
    except TypeError:
        raise TypeError(
            f"corpus {type(corpus).__name__} exposes no tasks"
        ) from None


def corpus_name(corpus: Any) -> str:
    return str(_get(corpus, "name", "dataset_id", default=type(corpus).__name__))


def corpus_digest(corpus: Any) -> Optional[str]:
    dig = getattr(corpus, "digest", None)
    if callable(dig):
        try:
            return str(dig())
        except Exception:
            return None
    if isinstance(corpus, Mapping) or hasattr(corpus, "items"):
        try:
            blob = json.dumps(
                {
                    "items": [
                        _plain(i) for i in corpus_items(corpus)
                    ],
                    "tasks": [
                        _plain(t) for t in corpus_tasks(corpus)
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()
        except Exception:
            return None
    return None


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return repr(obj)


def _jsonable(value: Any, depth: int = 0) -> Any:
    """JSON-safe projection for report blocks — scalars and mappings
    pass through; anything else degrades to ``repr`` so a report write
    never dies on an exotic override value."""
    if isinstance(value, Mapping) and depth < 4:
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and depth < 4:
        return [_jsonable(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def item_ref(item: Any, position: int = 0) -> str:
    """The item's gold-granularity ref (positional fallback is honest:
    an id-less item is addressable only by order)."""
    ref = _get(item, "ref", "id", "item_id", "dia_id", "turn_id")
    return str(ref) if ref is not None else f"item-{position:05d}"


def item_session(item: Any) -> Optional[str]:
    s = _get(item, "session_id", "session", "sessionId")
    return str(s) if s is not None else None


def item_document(item: Any) -> str:
    """The indexable document string — identical for every arm.

    Uses the item's own ``render_text()`` when present (the corpora
    adapter's canonical form); otherwise composes the same probe-style
    ``[when] speaker: text [shares image: caption]`` payload.  Arm parity
    requires byte-identical indexable text — never let one arm index a
    different rendering.
    """
    render = getattr(item, "render_text", None)
    if callable(render):
        out = render()
        if isinstance(out, str):
            return out
    text = str(_get(item, "text", "content", "body", default="") or "")
    speaker = _get(item, "speaker", "author")
    when = _get(item, "when", "timestamp", "date", "time")
    caption = _get(item, "image_caption", "blip_caption")
    head = f"[{when}] " if when else ""
    who = f"{speaker}: " if speaker else ""
    tail = f" [shares image: {caption}]" if caption else ""
    return f"{head}{who}{text}{tail}"


def _safe_document(item: Any) -> str:
    """``item_document`` for in-memory baselines — a degenerate item
    degrades to an empty document rather than killing ingest."""
    try:
        return item_document(item)
    except Exception:  # noqa: BLE001
        return ""


def item_metadata(item: Any) -> Dict[str, Any]:
    """JSON-safe metadata carried into ``Memory.add`` — preserved
    speakers/sessions/timestamps per V7-22.02."""
    meta: Dict[str, Any] = {}
    for key, names in (
        ("speaker", ("speaker",)),
        ("session_id", ("session_id", "session")),
        ("when", ("when", "timestamp", "date")),
        ("kind", ("kind",)),
    ):
        v = _get(item, *names)
        if v is not None and isinstance(v, (str, int, float, bool)):
            meta[key] = v
    return meta


# ---------------------------------------------------------------------------
# GEIC — gold-evidence-in-context delivery expansion (SPEC_V8_5 §2/§4,
# V85-02.04 + V85-04.04)
# ---------------------------------------------------------------------------
#
# The AMB provider v2 and the Track R ``session_messages`` arm measure the
# SAME delivery: hits walk in rank order, each adds itself plus its
# ±``W_r`` session neighbors, dedup'd, stopping before the token budget
# ``B`` is exceeded.  ``geic_expand`` is that rule, once, shared by both
# harnesses — Track R and AMB then measure the same delivered context.
#
# Token accounting mirrors V85-02.05's render: every delivered position
# costs the tokens of its rendered turn line (``Speaker: text`` plus an
# image caption when present); a session that contributes at least one
# position pays its header once.  The meter is the caller's — AMB's
# authoritative ``cl100k_base`` when tiktoken is importable, else the
# repo's pinned ``tok/v1`` estimator (disclosed on the record as
# ``meter`` — the offline proxy's stand-in, never a silent swap).

#: The V8-15.08/V85-04.01 budget sweep; ``None`` is the unbounded point.
GEIC_BUDGETS: Tuple[Optional[int], ...] = (1000, 2000, 4500, 9000, None)

#: ``W_r`` — the §2 neighbor-window prior (arm {0, 1, 2}).
GEIC_WINDOW = 1


def geic_meter() -> Tuple[str, Any]:
    """``(name, fn)`` — cl100k via tiktoken when importable (the AMB
    authoritative meter), else the repo's ``tok/v1`` estimator.  The
    chosen meter is named on every record — offline runs disclose the
    stand-in rather than claiming cl100k."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return (
            "cl100k_base",
            lambda text: len(enc.encode(str(text), disallowed_special=())),
        )
    except Exception:
        from eval.v7.metrics import estimate_tokens

        return ("tok/v1", lambda text: estimate_tokens(str(text)))


#: ``[verbatim · retrieved]\n`` stray-hit wrapper + the join/header
#: margins — the same constants ``eval.amb.provider._expand`` applies
#: (kept identical on purpose: this is the offline-parity replay of the
#: provider's delivery, not an independent rule).
GEIC_STRAY_PREFIX = "[verbatim · retrieved]\n"
GEIC_LINE_MARGIN = 1     # per delivered turn line (the join's newline)
GEIC_SESSION_MARGIN = 8  # once per touched session ("## Memory i\n" + header slack)
GEIC_STRAY_MARGIN = 8


def geic_expand(
    hit_blocks: Sequence[Any],
    sessions: Mapping[str, Any],
    *,
    window: int,
    budget: Optional[int],
) -> Tuple[Dict[str, Tuple[int, ...]], int]:
    """V85-02.04 delivery expansion — the offline-parity replay of
    ``VerbatimAMBProvider._expand`` (eval/amb/provider.py), shared with
    Track R so both harnesses measure the same delivered context
    (V85-04.04).

    ``hit_blocks`` — rank-ordered ``(session_key, covered, stray_cost)``
    tuples: ``covered`` is the set of session positions the hit itself
    covers (``None`` when the hit resolved to no session — a stray,
    charged ``stray_cost`` tokens and no positions).

    ``sessions`` — ``{session_key: {"costs": [int per position],
    "header": int}}``: ``costs[i]`` is the metered ``Speaker: text``
    line cost (render_turn_line, V85-02.05); ``header`` the metered
    ``[<conversation> · session <n> · <date>]`` header, charged once the
    first time the session contributes a position (plus
    ``GEIC_SESSION_MARGIN``).

    Rule, mirrored line-for-line from the provider: walk hits in rank
    order; per hit, ``needed = ∪ [c−W_r, c+W_r]`` over covered positions,
    visited coverage-first then by expanding ring (``(min |o−c|, o)``);
    each undelivered position costs ``costs[o] + 1`` (+ ``header + 8``
    when it opens the session).  When ``budget`` is not ``None``, the
    FIRST position/stray whose cost would push the total over ``B``
    stops the whole walk — the provider's per-turn stop, kept: a hit's
    block can deliver partially, nothing after the overflow point is
    added.

    Returns ``({session_key: (delivered positions, ascending)},
    tokens_spent)``.
    """
    delivered: Dict[str, set] = {}
    spent = 0
    stopped = False
    for block in hit_blocks:
        if stopped:
            break
        try:
            skey, covered, stray_cost = block
        except (TypeError, ValueError):
            continue
        sess = sessions.get(skey) if skey is not None else None
        if sess is None:
            cost = int(stray_cost or 0)
            if budget is not None and spent + cost > budget:
                stopped = True
                break
            spent += cost
            continue
        n = len(sess["costs"])
        if not n:
            continue
        cov = sorted(
            {int(o) for o in (covered or ()) if 0 <= int(o) < n}
        ) or [0]
        needed: set = set()
        for c in cov:
            lo = max(0, c - int(window))
            hi = min(n - 1, c + int(window))
            needed.update(range(lo, hi + 1))
        ordered = sorted(
            needed,
            key=lambda o: (min(abs(o - c) for c in cov), o),
        )
        got = delivered.setdefault(skey, set())
        for o in ordered:
            if o in got:
                continue
            cost = int(sess["costs"][o]) + GEIC_LINE_MARGIN
            if not got:
                cost += int(sess.get("header") or 0) + GEIC_SESSION_MARGIN
            if budget is not None and spent + cost > budget:
                stopped = True
                break
            got.add(o)
            spent += cost
    return (
        {k: tuple(sorted(v)) for k, v in delivered.items()},
        spent,
    )


class UnitSpans:
    """Generation-fenced ``units`` + ``claim_evidence``/``spans`` reads —
    the hit→position resolver shared by the ``session_messages`` arm and
    the AMB proxy (V85-04.04: same delivery, same resolution).

    One instance per bank/store; member lists and per-source turn-unit
    tables are cached for the run.  Reads are snapshot reads at the
    store's pinned projection generation — the same snapshot the
    pipeline delivered from (BRIEF invariant 3).
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._generation: Optional[int] = None
        self._unit_rows_cache: Dict[str, Dict[str, Any]] = {}
        self._turn_units_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._claim_ranges_cache: Dict[str, List[Tuple[str, int, int]]] = {}

    @property
    def generation(self) -> int:
        if self._generation is None:
            try:
                self._generation = int(self._store.projection_generation())
            except Exception:  # noqa: BLE001 — unfenced read, disclosed
                self._generation = 0
        return self._generation

    def _conn(self) -> Any:
        return self._store.read()

    def unit_rows(
        self, unit_ids: Iterable[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Latest row per unit id at/below the pinned generation —
        ``{unit_id: {kind, source_id, revision, session_id, seq,
        parent_unit_id, byte_start, byte_end}}``."""
        want = [u for u in dict.fromkeys(unit_ids) if u]
        missing = [u for u in want if u not in self._unit_rows_cache]
        if missing:
            try:
                from verbatim.storage.repos import has_table

                with self._conn() as conn:
                    if has_table(conn, "units"):
                        ph = ",".join("?" * len(missing))
                        for r in conn.execute(
                            "SELECT unit_id, source_id, revision, kind,"
                            " session_id, seq, parent_unit_id,"
                            " byte_start, byte_end"
                            f" FROM units WHERE unit_id IN ({ph})"
                            " AND generation<=? ORDER BY generation DESC",
                            (*missing, self.generation),
                        ):
                            row = {
                                "unit_id": r[0],
                                "source_id": r[1],
                                "revision": r[2],
                                "kind": r[3],
                                "session_id": r[4],
                                "seq": r[5],
                                "parent_unit_id": r[6],
                                "byte_start": r[7],
                                "byte_end": r[8],
                            }
                            self._unit_rows_cache.setdefault(r[0], row)
                    else:
                        for u in missing:
                            self._unit_rows_cache.setdefault(u, {})
            except Exception:  # noqa: BLE001 — degrade to unresolved
                for u in missing:
                    self._unit_rows_cache.setdefault(u, {})
        return {u: self._unit_rows_cache.get(u) or {} for u in want}

    def turn_units(self, source_id: str) -> List[Dict[str, Any]]:
        """The source's ``kind='turn'`` units ordered by effective
        position (``(seq, occurred_start_us, recorded_at_us,
        byte_start, unit_id)`` — V85-03.01's read-time order)."""
        if source_id not in self._turn_units_cache:
            rows: List[Dict[str, Any]] = []
            try:
                from verbatim.storage.repos import has_table

                with self._conn() as conn:
                    if has_table(conn, "units"):
                        raw = list(conn.execute(
                            "SELECT unit_id, session_id, seq,"
                            " byte_start, byte_end, generation"
                            " FROM units WHERE source_id=?"
                            " AND kind='turn' AND generation<=?"
                            " ORDER BY generation DESC, seq,"
                            " byte_start, unit_id",
                            (source_id, self.generation),
                        ))
                        if raw:
                            # latest generation only — superseded rows
                            # must not shadow the live ordinal map
                            top = max(r[5] for r in raw)
                            for r in (r for r in raw if r[5] == top):
                                rows.append({
                                    "unit_id": r[0],
                                    "session_id": r[1],
                                    "seq": r[2],
                                    "byte_start": r[3],
                                    "byte_end": r[4],
                                })
            except Exception:  # noqa: BLE001
                rows = []
            self._turn_units_cache[source_id] = rows
        return self._turn_units_cache[source_id]

    def claim_ranges(
        self, claim_ids: Iterable[str]
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """Claim hit → evidence span byte ranges:
        ``{claim_id: [(source_id, start_byte, end_byte)]}``."""
        want = [c for c in dict.fromkeys(claim_ids) if c]
        missing = [c for c in want if c not in self._claim_ranges_cache]
        if missing:
            try:
                from verbatim.storage.repos import has_table

                with self._conn() as conn:
                    if has_table(conn, "claim_evidence") and has_table(
                        conn, "spans"
                    ):
                        ph = ",".join("?" * len(missing))
                        for r in conn.execute(
                            "SELECT ce.claim_id, s.source_id,"
                            " s.start_byte, s.end_byte"
                            " FROM claim_evidence ce"
                            " JOIN spans s ON s.span_id = ce.span_id"
                            f" WHERE ce.claim_id IN ({ph})",
                            tuple(missing),
                        ):
                            self._claim_ranges_cache.setdefault(
                                r[0], []
                            ).append((r[1], r[2], r[3]))
            except Exception:  # noqa: BLE001
                pass
            for c in missing:
                self._claim_ranges_cache.setdefault(c, [])
        return {c: self._claim_ranges_cache.get(c, []) for c in want}

    @staticmethod
    def positions(
        sess: Mapping[str, Any], row: Mapping[str, Any]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Unit row → ``(lo, hi)`` member positions inside ``sess``.

        ``sess`` is ``{"n": int, "ranges": [(bs, be)]}`` — member byte
        ranges in position order.  First match wins:

        * ``turn`` units pin ``seq`` when it lands inside the member
          table (the per-session message ordinal IS the position);
        * otherwise byte-range intersection — any member whose range
          intersects the unit's ``[byte_start, byte_end)`` is covered
          (episode/session units span their member turns; a
          sentence-window unit intersects exactly its parent turn's
          range and anchors at it);
        * ``(None, None)`` when nothing resolves — the hit still
          consumed budget but covers no position.
        """
        seq = row.get("seq")
        if row.get("kind") == "turn" and isinstance(seq, int):
            if 0 <= seq < int(sess["n"]):
                return seq, seq
        bs, be = row.get("byte_start"), row.get("byte_end")
        if isinstance(bs, int) and isinstance(be, int) and be > bs:
            hits = [
                i for i, (mbs, mbe) in enumerate(sess["ranges"])
                if mbs is not None and mbe is not None
                and mbs < be and bs < mbe
            ]
            if hits:
                return min(hits), max(hits)
        return None, None


def _parse_hit_object_ref(
    object_ref: Any,
) -> Optional[Tuple[str, str, Optional[int]]]:
    """``vobj1.<kind>.<hex(utf8 id)>.<rev|->`` → ``(kind, id, rev|None)``
    — the ``Hit.object_ref`` wire form (``verbatim/memory/controls.py``),
    parsed identically to ``eval.amb.provider.parse_object_ref`` so the
    two harnesses resolve the same hit to the same object."""
    if not isinstance(object_ref, str) or not object_ref.startswith(
        "vobj1."
    ):
        return None
    parts = object_ref.split(".")
    if len(parts) not in (3, 4):
        return None
    try:
        oid = bytes.fromhex(parts[2]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    rev: Optional[int] = None
    if len(parts) == 4 and parts[3] != "-":
        try:
            rev = int(parts[3])
        except ValueError:
            return None
    return parts[1], oid, rev


# ---------------------------------------------------------------------------
# task surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmTask:
    """The ONLY task object an arm may read — gold-free by construction.

    ``as_of`` is the task's declared question-time anchor (V8-09.02 —
    corpus metadata, never gold): int µs when the corpus clock resolved
    deterministically, else the raw scalar for the facade's own
    ``as_of`` VALIDATION to judge (V8-09.01/20.05).  ``None`` means the
    corpus supplied no anchor and the arm keeps wall-clock behavior.
    """

    task_id: str
    query: str
    category: str = "unknown"
    group_id: Optional[str] = None
    as_of: Optional[Any] = None


#: Task fields that may declare the caller's question-time anchor
#: (V8-09.02 — LongMemEval ``question_date`` → ``query_time``, twin
#: ``question_time``/``question_time_us``).  ``*_us`` keys are literal
#: µs; the rest resolve by magnitude/RFC3339/LoCoMo ``when`` grammar.
#: Declared keys only — answers/evidence ids are never read here.
_TASK_TIME_KEYS = (
    ("as_of", False),
    ("as_of_us", True),
    ("query_time", False),
    ("query_time_us", True),
    ("question_time", False),
    ("question_time_us", True),
    ("question_date", False),
    ("timestamp_us", True),
)


def _as_of_us(value: Any, *, assume_us: bool = False) -> Optional[int]:
    """Deterministic question-time anchor → int µs, or ``None``.

    Reuses the store's own parsers (``units_v7._as_us`` magnitude/
    RFC3339 + ``_parse_when_str`` for the LoCoMo ``when`` grammar) so
    the arm and the index read clocks identically.  Unresolvable values
    stay ``None`` — never a guessed clock.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        from verbatim.projections import units_v7 as _u
    except Exception:  # noqa: BLE001 — store internals absent at import
        return None
    us = _u._as_us(value, assume_us=assume_us)
    if us is not None:
        return int(us)
    if isinstance(value, str):
        w = _u._parse_when_str(value)
        if w is not None:
            return int(w[0])
    return None


def _group_anchors(items: Iterable[Any]) -> Dict[str, int]:
    """``group_id`` → the group's final-session ``when`` in µs
    (V8-09.02 — the LoCoMo anchor choice).  The max parseable ``when``
    per group wins; unparseable clocks contribute nothing (never a
    guessed anchor)."""
    out: Dict[str, int] = {}
    for it in items:
        gid = _get(it, "group_id", "group")
        if gid is None:
            continue
        us = _as_of_us(_get(it, "when", "timestamp", "date", "time"))
        if us is not None and us > out.get(str(gid), -1):
            out[str(gid)] = us
    return out


def _task_as_of(task: Any) -> Any:
    """The task's declared question-time anchor, or ``None``.

    Scan order: the task's own fields, then a ``metadata``/``meta``
    mapping, then a wrapped ``raw`` task and its metadata (the scorer-
    side view passes the corpus task through).  Public views that raise
    ``AttributeError`` on ``metadata`` degrade to ``None`` via ``_get``.
    """
    raw = _get(task, "raw")
    raw_meta = _get(raw, "metadata", "meta") if raw is not None else None
    for src in (task, _get(task, "metadata", "meta"), raw, raw_meta):
        if src is None:
            continue
        for key, assume_us in _TASK_TIME_KEYS:
            v = _get(src, key)
            if v is None:
                continue
            us = _as_of_us(v, assume_us=assume_us)
            return us if us is not None else v
    return None


def arm_task(task: Any) -> ArmTask:
    """Normalize any duck-typed task into the arm-facing view."""
    return ArmTask(
        task_id=str(_get(task, "task_id", "id", default="task")),
        query=str(_get(task, "query", "question", default="")),
        category=str(_get(task, "category", "kind", default="unknown")),
        group_id=_get(task, "group_id", "group"),
        as_of=_task_as_of(task),
    )


# ---------------------------------------------------------------------------
# query outcome — the ranked ref list + attribution diagnostics
# ---------------------------------------------------------------------------


@dataclass
class QueryOutcome:
    """One ``arm.query`` result.

    ``refs`` is the ranked delivered ref list at the asked ``k``
    (iterate/index the outcome directly).  ``surfaced`` is the arm's
    observable candidate pool including below-k entries — ``None`` when
    the arm cannot observe a pool (attribution then refuses to guess).
    ``abstained`` marks a verdict-level withhold; an empty result from
    a no-verdict arm is ``status="ok"`` + ``refs=[]``, NOT an abstain.
    """

    refs: List[Any] = field(default_factory=list)
    surfaced: Optional[List[Any]] = None
    status: str = "ok"
    abstained: bool = False
    latency_ms: float = 0.0
    n_items: int = 0
    delivered_text: str = ""
    warnings: Tuple[str, ...] = ()
    error: Optional[str] = None
    diag: Dict[str, Any] = field(default_factory=dict)
    k: int = 0

    def __iter__(self) -> Iterator[Any]:
        return iter(self.refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> Any:
        return self.refs[i]


# ---------------------------------------------------------------------------
# DictCorpus — the reference adapter for plain dicts
# ---------------------------------------------------------------------------


class DictCorpus:
    """Corpus protocol over plain dicts — tests, JSONL/JSON datasets,
    and any adapter that does not want the typed records.

    Item keys: ``id``/``ref``/``item_id``, ``text``, ``speaker``,
    ``session_id``, ``when``/``timestamp``, ``image_caption``.
    Task keys: ``task_id``, ``query``, ``category``, item-granularity
    gold under ``gold_evidence``/``evidence_ids``/``gold``, session gold
    under ``evidence_session_ids``/``gold_session_ids``, ``answerable``
    (or the inverse ``expected_abstain``), ``group_id``.
    """

    def __init__(
        self,
        items: Iterable[Any],
        tasks: Iterable[Any],
        *,
        name: str = "dict-corpus",
        dataset_id: str = "dict",
        split: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.dataset_id = dataset_id
        self.split = split
        self.items = list(items)
        self.tasks = list(tasks)
        self.metadata = dict(metadata or {})

    def __iter__(self) -> Iterator[Any]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def digest(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "dataset_id": self.dataset_id,
                "split": self.split,
                "items": [_plain(i) for i in self.items],
                "tasks": [_plain(t) for t in self.tasks],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FlatBM25Arm — the honest Okapi reference (k1=1.2, b=0.75)
# ---------------------------------------------------------------------------


class FlatBM25Arm:
    """Pure-Python Okapi BM25 over the raw item documents.

    Self-contained: document order = corpus order; tie-break is
    (``-score``, corpus index) so output is deterministic.  ``surfaced``
    is the full positive-scored pool — the whole lane is observable, so
    attribution never has to guess.
    """

    name = "flat_bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._refs: List[str] = []
        self._texts: List[str] = []
        self._docs: List[Counter] = []
        self._len: List[int] = []
        self._avg = 0.0
        self._idf: Dict[str, float] = {}
        self.notes: List[str] = []
        self.last_ingest: Dict[str, Any] = {}

    # -- protocol -----------------------------------------------------

    def ingest(self, corpus: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        items = corpus_items(corpus)
        self._refs = [item_ref(it, i) for i, it in enumerate(items)]
        self._texts = [_safe_document(it) for it in items]
        self._docs = [
            Counter(TOK.findall(t.lower())) for t in self._texts
        ]
        self._len = [sum(d.values()) for d in self._docs]
        n = len(self._docs)
        self._avg = (sum(self._len) / n) if n else 0.0
        df = Counter(w for d in self._docs for w in d)
        # Robertson/Spark-Jones idf, +1 inside the log keeps it positive.
        self._idf = {
            w: math.log(1 + (n - f + 0.5) / (f + 0.5))
            for w, f in df.items()
        }
        self.last_ingest = {
            "indexed": n,
            "add_errors": 0,
            "ingest_ms": (time.perf_counter() - t0) * 1000.0,
            "settle": "synchronous (in-process index)",
        }
        return dict(self.last_ingest)

    def indexed_refs(self) -> set:
        return set(self._refs)

    def configure(self, *, lanes_disabled: Any = None, **_: Any) -> Dict[str, Any]:
        return {
            "requested": sorted(str(x) for x in (lanes_disabled or ())),
            "applied": [],
            "unapplied": sorted(str(x) for x in (lanes_disabled or ())),
            "reason": "single-lane reference arm — no lane pool to ablate",
        }

    def query(self, task: Any, k: int) -> QueryOutcome:
        t0 = time.perf_counter()
        qterms = TOK.findall(str(getattr(task, "query", task)).lower())
        scored: List[Tuple[float, int]] = []
        for i, d in enumerate(self._docs):
            s = 0.0
            for w in qterms:
                f = d.get(w)
                if f:
                    s += self._idf[w] * f * (self.k1 + 1) / (
                        f
                        + self.k1
                        * (1 - self.b + self.b * self._len[i] / self._avg)
                    )
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda p: (-p[0], p[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        surfaced = [self._refs[i] for _s, i in scored]
        refs = surfaced[: int(k)]
        text = "\n".join(
            self._texts[i] for _s, i in scored[: int(k)]
        )
        return QueryOutcome(
            refs=refs,
            surfaced=surfaced,
            status="ok",
            abstained=False,
            latency_ms=ms,
            n_items=len(refs),
            delivered_text=text,
            diag={"pool_observable": True, "pool_size": len(surfaced)},
            k=int(k),
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# FTS5Arm — the naive SQLite baseline
# ---------------------------------------------------------------------------


class FTS5Arm:
    """SQLite FTS5 ``bm25()`` over the same item documents.

    In-memory, single table, OR-match over whitespace query terms —
    the v6 ``naive_fts`` convention.  ``surfaced`` is the full MATCH
    pool (all rows matching ≥ 1 term), ranked by ``rank`` then ``rowid``
    (insertion order) — deterministic.
    """

    name = "fts5"

    #: surfaced-pool cap — bounded honesty, far above any k we score.
    POOL_CAP = 10_000

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE VIRTUAL TABLE items USING fts5(ref UNINDEXED, text)"
        )
        self._refs: List[str] = []
        self._texts: Dict[str, str] = {}
        self.notes: List[str] = []
        self.last_ingest: Dict[str, Any] = {}

    def ingest(self, corpus: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        items = corpus_items(corpus)
        self._refs = [item_ref(it, i) for i, it in enumerate(items)]
        self._texts = {}
        rows = []
        for i, it in enumerate(items):
            doc = _safe_document(it)
            self._texts[self._refs[i]] = doc
            rows.append((self._refs[i], doc))
        self._conn.executemany(
            "INSERT INTO items(ref, text) VALUES (?, ?)", rows
        )
        self._conn.commit()
        self.last_ingest = {
            "indexed": len(rows),
            "add_errors": 0,
            "ingest_ms": (time.perf_counter() - t0) * 1000.0,
            "settle": "synchronous (fts5 commit)",
        }
        return dict(self.last_ingest)

    def indexed_refs(self) -> set:
        return set(self._refs)

    def configure(self, *, lanes_disabled: Any = None, **_: Any) -> Dict[str, Any]:
        return {
            "requested": sorted(str(x) for x in (lanes_disabled or ())),
            "applied": [],
            "unapplied": sorted(str(x) for x in (lanes_disabled or ())),
            "reason": "single-lane baseline — no lane pool to ablate",
        }

    @staticmethod
    def _terms(query: str) -> List[str]:
        """Whitespace tokens carrying ≥ 1 alphanumeric — identifiers
        like ``tok-10000`` survive quoting (v6 convention)."""
        out = []
        for tok in str(query).split():
            tok = tok.strip(string.punctuation)
            if tok and any(c.isalnum() for c in tok):
                out.append(tok)
        return out

    def query(self, task: Any, k: int) -> QueryOutcome:
        t0 = time.perf_counter()
        terms = self._terms(getattr(task, "query", task))
        rows: List[Tuple[str, str]] = []
        error = None
        if terms:
            match = " OR ".join(f'"{t}"' for t in terms)
            try:
                rows = self._conn.execute(
                    "SELECT ref, text FROM items WHERE items MATCH ?"
                    " ORDER BY rank, rowid LIMIT ?",
                    (match, self.POOL_CAP),
                ).fetchall()
            except sqlite3.Error as exc:
                error = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        surfaced = [r[0] for r in rows]
        refs = surfaced[: int(k)]
        text = "\n".join(r[1] for r in rows[: int(k)])
        return QueryOutcome(
            refs=refs,
            surfaced=surfaced,
            status="error" if error else "ok",
            abstained=False,
            latency_ms=ms,
            n_items=len(refs),
            delivered_text=text,
            error=error,
            diag={"pool_observable": True, "pool_size": len(surfaced)},
            k=int(k),
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VerbatimArm — the real public Memory facade
# ---------------------------------------------------------------------------


class VerbatimArm:
    """``verbatim`` — the real ``Memory`` write + read path.

    Seeding is ``Memory.add`` per item (contention-retried through the
    eval.v5 add helper), settle is the external-worker durable-queue
    drain + ``wait_ready`` frontier — the consumer route, measured, no
    fixture shortcuts.  Querying is ``Memory.search(limit=k)``; a second
    unmeasured pool-depth query feeds attribution's ``surfaced``
    (prefix-consistent by construction — the facade cuts a fused
    ranking at ``limit``).

    ``configure(lanes_disabled=...)`` maps known lane names onto
    ``config.v3.retrieval.<lane>=False`` — the current store's gate
    surface.  Lane names with no gate are reported ``unapplied``,
    never silently ignored.
    """

    name = "verbatim"

    #: Lane names the current store can actually gate (v3.retrieval
    #: config section — the pre-W2 gate surface; v7 lane names land
    #: with the pipeline and are reported unapplied until then).
    KNOWN_LANES = ("causal", "dense", "graph", "late_interaction", "sparse")

    def __init__(
        self,
        *,
        workdir: Optional[str] = None,
        user_id: str = "track-r",
        worker: str = "external",
        infer: bool = True,
        encoder: str = "hashing",
        consistency: str = "session",
        pool_limit: int = DEFAULT_POOL_LIMIT,
        settle_timeout_s: float = 120.0,
        memory_kwargs: Optional[Dict[str, Any]] = None,
        policy_overrides: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[float] = 2000.0,
        ingest_mode: str = "item",
        geic_window: int = GEIC_WINDOW,
        geic_budgets: Optional[Sequence[Optional[int]]] = None,
        token_meter: Optional[Tuple[str, Any]] = None,
    ) -> None:
        self._owns_dir = workdir is None
        self._workdir = workdir or tempfile.mkdtemp(prefix="v7-trackr-")
        os.makedirs(self._workdir, exist_ok=True)
        self.user_id = user_id
        self.worker = worker
        self.infer = bool(infer)
        self.encoder = encoder
        self.consistency = consistency
        self.pool_limit = int(pool_limit)
        self.settle_timeout_s = float(settle_timeout_s)
        self._memory_kwargs = dict(memory_kwargs or {})
        self._policy_overrides = dict(policy_overrides or {})
        # V85-04.04 — ``session_messages`` ingests one
        # ``Memory.add(messages=[...], session_id=…)`` per corpus
        # session (the V85-02.02 payload shape), so turn units carry
        # real per-session ``seq`` ordinals and the geic@B expansion
        # (V85-02.04) walks actual session neighbors.  ``item`` is the
        # unchanged default.
        if ingest_mode not in ("item", "session_messages"):
            raise ValueError(
                f"unknown ingest_mode {ingest_mode!r} — expected "
                "'item' or 'session_messages'")
        self.ingest_mode = ingest_mode
        self.geic_window = int(geic_window)
        self.geic_budgets = (
            GEIC_BUDGETS if geic_budgets is None else tuple(
                None if b is None else int(b) for b in geic_budgets))
        self._token_meter = token_meter
        self._meter_cache: Optional[Tuple[str, Any]] = None
        # Session-delivery tables — filled by ``_ingest_session_messages``.
        self._session_members: Dict[str, List[str]] = {}
        self._session_texts: Dict[str, List[str]] = {}
        self._session_costs: Dict[str, List[int]] = {}
        self._session_headers: Dict[str, int] = {}
        self._session_sources: Dict[str, str] = {}
        self._source_session: Dict[str, str] = {}
        self._units: Optional[UnitSpans] = None
        self._member_tables_cache: Optional[Dict[str, Dict[str, Any]]] = None
        # Quality-profile deadline for measurement (beat-it r2): the
        # product default stays 500ms — the eval arm's job is to measure
        # what the pipeline can deliver, so it gets 2s headroom unless
        # the caller pins otherwise (``None`` → library default).
        self.timeout_ms = timeout_ms
        self._config_mapping: Dict[str, Any] = {}
        self._lane_report: Optional[Dict[str, Any]] = None
        self._mem: Any = None
        self._source_ref: Dict[str, str] = {}   # source_id -> item ref
        self._ref_source: Dict[str, str] = {}   # item ref -> source_id
        self._indexed: List[str] = []
        self._add_errors: Dict[str, str] = {}
        self._dedup_collisions = 0
        self._last_receipt: Any = None
        # V8-09.02 — per-group anchor (the conversation's final session
        # ``when`` in µs, built at ingest) + cached facade capability.
        self._group_as_of: Dict[str, int] = {}
        self._as_of_supported: Optional[bool] = None
        self.notes: List[str] = []
        self.last_ingest: Dict[str, Any] = {}

    # -- configuration ------------------------------------------------

    def configure(self, *, lanes_disabled: Any = None, **overrides: Any) -> Dict[str, Any]:
        """Leave-one-lane-out plumbing (V7-22.22 ablation rows).

        Known lanes map to ``config.v3.retrieval.<lane> = False`` at
        ``Memory`` construction; unknown names and any other overrides
        are recorded unapplied — the report shows exactly what took.
        Must be called before :meth:`ingest` (config binds at
        construction); calling after ingest reports ``applied=[]``.
        """
        req = sorted(str(x) for x in (lanes_disabled or ()))
        if self._mem is not None:
            self._lane_report = {
                "requested": req,
                "applied": [],
                "unapplied": req,
                "reason": "configure() after ingest — store already built",
            }
            return dict(self._lane_report)
        known = [x for x in req if x in self.KNOWN_LANES]
        unknown = [x for x in req if x not in self.KNOWN_LANES]
        if known:
            retr = self._config_mapping.setdefault("v3", {}).setdefault(
                "retrieval", {}
            )
            for lane in known:
                retr[lane] = False
        passthrough = {k: v for k, v in overrides.items()}
        if passthrough:
            self._policy_overrides.update(passthrough)
        self._lane_report = {
            "requested": req,
            "applied": known,
            "unapplied": unknown,
            "applied_via": (
                "config.v3.retrieval.<lane>=false" if known else None
            ),
            "policy_overrides": sorted(passthrough),
        }
        return dict(self._lane_report)

    # -- ingest --------------------------------------------------------

    def ingest(self, corpus: Any) -> Dict[str, Any]:
        from eval.v5.harness import add_with_retry, drain_memory
        from verbatim import Memory

        items = corpus_items(corpus)
        mapping = dict(self._config_mapping)
        for key, val in self._policy_overrides.items():
            # policy overrides are config-section mappings merged deep
            if isinstance(val, Mapping) and isinstance(
                mapping.get(key), Mapping
            ):
                mapping[key] = {**mapping[key], **val}
            else:
                mapping[key] = val
        t0 = time.perf_counter()
        self._mem = Memory(
            os.path.join(self._workdir, "mem.db"),
            user_id=self.user_id,
            worker=self.worker,
            encoder=self.encoder,
            config=mapping or None,
            **self._memory_kwargs,
        )
        env = _DrainEnv(worker=self.worker, memory=self._mem, notes=self.notes)

        # V8-09.02 — per-group question-time anchor: the timestamp of
        # the conversation's final session (corpus metadata, never
        # gold).  A task-declared anchor still beats this at query.
        self._group_as_of = _group_anchors(items)

        if self.ingest_mode == "session_messages":
            self._ingest_session_messages(items, add_with_retry)
        else:
            for i, item in enumerate(items):
                ref = item_ref(item, i)
                try:
                    doc = item_document(item)
                except Exception as exc:  # noqa: BLE001 — recorded, kept visible
                    self._add_errors[ref] = (
                        f"document: {type(exc).__name__}: {exc}"
                    )
                    continue
                try:
                    res = add_with_retry(
                        self._mem, doc, infer=self.infer,
                        metadata=item_metadata(item),
                    )
                except Exception as exc:  # noqa: BLE001 — recorded, kept visible
                    self._add_errors[ref] = f"{type(exc).__name__}: {exc}"
                    continue
                sid = getattr(res, "memory_id", "") or ""
                if not sid:
                    try:
                        from verbatim.memory.types import MemoryRef

                        sid = MemoryRef.parse(res.ref).source_id
                    except Exception:
                        sid = ""
                if sid:
                    if sid in self._source_ref and self._source_ref[sid] != ref:
                        # Byte-identical payloads dedupe to one source — a
                        # real store behavior, counted not hidden.
                        self._dedup_collisions += 1
                    else:
                        self._source_ref[sid] = ref
                        self._ref_source[ref] = sid
                self._indexed.append(ref)
                self._last_receipt = res

        # settle: external worker drains the durable queue here;
        # managed mode drains asynchronously — either way the readiness
        # frontier is confirmed through wait_ready, never assumed.
        drain_rep = drain_memory(env)
        state = "no_receipts"
        waited = 0.0
        if self._last_receipt is not None:
            deadline = time.monotonic() + self.settle_timeout_s
            while True:
                try:
                    rd = self._mem.wait_ready(
                        self._last_receipt, timeout_ms=2000
                    )
                    state = str(rd.state)
                except Exception as exc:  # noqa: BLE001
                    state = f"error:{type(exc).__name__}"
                if state in ("ready", "blocked", "unavailable") or (
                    time.monotonic() > deadline
                ):
                    break
                time.sleep(0.05)
            waited = round(time.monotonic() - (deadline - self.settle_timeout_s), 3)
        if self._dedup_collisions:
            self.notes.append(
                f"{self._dedup_collisions} byte-identical adds deduped "
                "to an earlier item's source (first ref wins)"
            )
        self.last_ingest = {
            "indexed": len(self._indexed),
            "add_errors": len(self._add_errors),
            "add_error_detail": dict(self._add_errors),
            "dedup_collisions": self._dedup_collisions,
            "drain": dict(drain_rep),
            "settle_state": state,
            "settle_waited_s": waited,
            "ingest_ms": (time.perf_counter() - t0) * 1000.0,
            "pool_query": (
                "search(limit=min(pool_limit, facade_max)) after the "
                "measured call — unmeasured, attribution-only"
            ),
            # K13/K74 — the exact config mapping handed to
            # ``Memory(config=…)``: every eval override (e.g.
            # ``admission.require_review`` in the verbatim_claims arm)
            # is named here, never implicit.
            "config_overrides": _jsonable(mapping),
            "search_defaults": {
                "timeout_ms": self.timeout_ms,
                "consistency": self.consistency,
                "pool_limit": self.pool_limit,
            },
            # V8-09.02 — the adapter documents its anchor choice.
            "as_of_policy": (
                "task-declared question time (as_of/query_time/"
                "question_time*/question_date) else the task group's "
                "final-session `when` built at ingest; no anchor → "
                "facade wall clock"
            ),
            "group_anchors": len(self._group_as_of),
        }
        if self._lane_report is not None:
            self.last_ingest["lanes_disabled"] = dict(self._lane_report)
        self.last_ingest["ingest_mode"] = self.ingest_mode
        if self.ingest_mode == "session_messages":
            meter_name, _m = self._meter()
            self.last_ingest["geic"] = {
                "sessions": len(self._session_members),
                "window": self.geic_window,
                "budgets": [
                    "unbounded" if b is None else int(b)
                    for b in self.geic_budgets
                ],
                "meter": meter_name,
            }
        return dict(self.last_ingest)

    # -- session_messages ingest + geic@B (V85-04.04) -------------------

    def _meter(self) -> Tuple[str, Any]:
        if self._token_meter is not None:
            return self._token_meter
        if self._meter_cache is None:
            self._meter_cache = geic_meter()
        return self._meter_cache

    def _ingest_session_messages(
        self, items: Sequence[Any], add_with_retry: Any
    ) -> None:
        """One ``Memory.add(messages=[...], session_id=…)`` per corpus
        session — the V85-02.02 payload shape the AMB provider v2 uses,
        so projection mints real per-session ``seq`` ordinals and the
        geic@B expansion walks true session neighbors (fixes R2 for the
        eval path; per-turn standalone adds mint ``seq=0``).

        Member lists mirror ``units_v7._turn_record``'s skip rule —
        empty-text items stay in the ``messages`` payload (faithful
        session record) but hold no position (no unit is minted for
        them), keeping ``members[seq]`` aligned.  Per-message ``at``
        rides as int µs when the corpus ``when`` parses (``_as_of_us``
        shares the store's parsers); the session's ``occurred_at`` is
        its first parseable ``when``.
        """
        _meter_name, meter = self._meter()
        groups: Dict[str, List[Tuple[int, Any]]] = {}
        order: List[str] = []
        for i, item in enumerate(items):
            skey = item_session(item) or f"__item__:{item_ref(item, i)}"
            if skey not in groups:
                groups[skey] = []
                order.append(skey)
            groups[skey].append((i, item))

        for skey in order:
            grouped = groups[skey]
            messages: List[Dict[str, Any]] = []
            refs: List[str] = []
            texts: List[str] = []
            costs: List[int] = []
            occurred_us: Optional[int] = None
            for i, item in grouped:
                ref = item_ref(item, i)
                meta = _get(item, "metadata", "meta")
                meta = meta if isinstance(meta, Mapping) else {}
                text = str(
                    _get(item, "text", "content", "body", default="")
                    or ""
                )
                speaker = _get(item, "speaker", "author")
                when = _get(item, "when", "timestamp", "date", "time")
                caption = _get(item, "image_caption", "blip_caption")
                at_us = _as_of_us(when)
                if occurred_us is None and at_us is not None:
                    occurred_us = at_us
                msg: Dict[str, Any] = {"speaker": speaker, "text": text}
                if at_us is not None:
                    msg["at"] = at_us
                elif when is not None:
                    msg["at"] = str(when)
                dia = meta.get("dia_id")
                if dia:
                    msg["dia_id"] = str(dia)
                if caption:
                    msg["blip_caption"] = str(caption)
                messages.append(msg)
                if text:
                    refs.append(ref)
                    texts.append(text)
                    # render_turn_line (V85-02.05): "Speaker: text" +
                    # " [image: caption]" — the delivered line cost.
                    line = (
                        f"{speaker}: {text}" if speaker else text
                    )
                    if caption:
                        line = f"{line} [image: {caption}]"
                    costs.append(meter(line))
            content = "\n".join(
                (f"{m.get('speaker')}: {m.get('text')}"
                 if m.get("speaker") else str(m.get("text")))
                for m in messages
                if m.get("text")
            )
            kw: Dict[str, Any] = {
                "infer": self.infer,
                "messages": messages,
                "session_id": skey,
                "metadata": {
                    "session": skey,
                    "n_messages": len(messages),
                    "ingest_mode": "session_messages",
                },
            }
            if occurred_us is not None:
                kw["occurred_at"] = occurred_us
            try:
                res = add_with_retry(self._mem, content, **kw)
            except Exception as exc:  # noqa: BLE001 — recorded, kept visible
                self._add_errors[skey] = f"{type(exc).__name__}: {exc}"
                continue
            sid = getattr(res, "memory_id", "") or ""
            if not sid:
                try:
                    from verbatim.memory.types import MemoryRef

                    sid = MemoryRef.parse(res.ref).source_id
                except Exception:
                    sid = ""
            if sid:
                if sid in self._source_session:
                    self._dedup_collisions += 1
                else:
                    self._source_session[sid] = skey
                    self._session_sources[skey] = sid
                    for ref in refs:
                        self._ref_source[ref] = sid
                    if refs:
                        self._source_ref.setdefault(sid, refs[0])
            self._session_members[skey] = refs
            self._session_texts[skey] = texts
            self._session_costs[skey] = costs
            self._session_headers[skey] = meter(f"[{skey}]")
            self._indexed.extend(refs)
            self._last_receipt = res
        if self._mem is not None:
            self._units = UnitSpans(self._mem._store)

    def _member_tables(self) -> Dict[str, Dict[str, Any]]:
        """``{session_key: {"n", "costs", "header", "ranges", "texts"}}``
        — member byte ranges resolved from the projected turn units
        (``members[i]`` ↔ ``seq == i``); unpinned ranges stay
        ``(None, None)`` and seq remains the position pin."""
        if self._member_tables_cache is None:
            tables: Dict[str, Dict[str, Any]] = {}
            for skey, refs in self._session_members.items():
                ranges: List[Tuple[Optional[int], Optional[int]]] = [
                    (None, None)
                ] * len(refs)
                source = self._session_sources.get(skey)
                if source and self._units is not None:
                    for tu in self._units.turn_units(source):
                        seq = tu.get("seq")
                        if (
                            isinstance(seq, int)
                            and 0 <= seq < len(ranges)
                            and ranges[seq] == (None, None)
                        ):
                            ranges[seq] = (
                                tu.get("byte_start"), tu.get("byte_end"))
                tables[skey] = {
                    "n": len(refs),
                    "costs": self._session_costs.get(skey) or [],
                    "header": self._session_headers.get(skey, 0),
                    "ranges": ranges,
                    "texts": self._session_texts.get(skey) or [],
                }
            self._member_tables_cache = tables
        return self._member_tables_cache

    def _ordinal_by_quote(self, skey: str, quote: Any) -> Optional[int]:
        """The SessionIndex.ordinal_by_quote rule over member texts —
        exact match first, then containment either way."""
        texts = self._session_texts.get(skey) or []
        q = str(quote or "").strip()
        if not q:
            return None
        for i, t in enumerate(texts):
            if t.strip() == q:
                return i
        for i, t in enumerate(texts):
            ts = t.strip()
            if ts and (q in ts or ts in q):
                return i
        return None

    def _hit_spans(self, hits: Sequence[Any]) -> List[Tuple[Any, ...]]:
        """Delivered hits → rank-ordered ``(session_key, covered,
        stray_cost)`` expansion inputs — the resolution ladder of
        ``VerbatimAMBProvider._resolve_hit`` (V85-02.04's "walk the hits
        in rank order"): ``vobj1.unit`` → session/seq or byte-range
        containment → quote match; ``vobj1.claim`` → claim_evidence →
        source → session; source-level hits → their session.  A hit
        that resolves to nothing known is a stray — charged the
        rendered-stray cost, covering no position."""
        if self._units is None or not self._session_members:
            return []
        _meter_name, meter = self._meter()
        members = self._member_tables()
        parsed: List[Optional[Tuple[str, str, Any]]] = []
        unit_ids: List[str] = []
        claim_ids: List[str] = []
        for h in hits:
            p = _parse_hit_object_ref(getattr(h, "object_ref", ""))
            parsed.append(p)
            if p is not None:
                if p[0] == "unit":
                    unit_ids.append(p[1])
                elif p[0] == "claim":
                    claim_ids.append(p[1])
        rows = self._units.unit_rows(unit_ids)
        claims = self._units.claim_ranges(claim_ids)

        plans: List[Tuple[Any, ...]] = []
        for h, p in zip(hits, parsed):
            quote = str(getattr(h, "quote", "") or "")
            stray_cost = meter(GEIC_STRAY_PREFIX + quote) \
                + GEIC_STRAY_MARGIN
            if p is not None and p[0] == "unit":
                row = rows.get(p[1]) or {}
                # A unit's session_key is its unit.session_id when it
                # names a member table, else its source's session.
                sess_id = row.get("session_id")
                skey = (
                    str(sess_id)
                    if sess_id is not None and str(sess_id) in members
                    else self._source_session.get(row.get("source_id"))
                )
                sess = members.get(skey) if skey is not None else None
                if sess is None:
                    plans.append((None, None, stray_cost))
                    continue
                lo, hi = UnitSpans.positions(sess, row)
                if lo is not None:
                    covered = list(range(int(lo), int(hi) + 1))
                else:
                    by_q = self._ordinal_by_quote(skey, quote)
                    covered = [by_q] if by_q is not None else [0]
                plans.append((skey, covered, 0))
            elif p is not None and p[0] == "claim":
                skey = None
                for source_id, _bs, _be in claims.get(p[1], ()):  # noqa: B007
                    skey = self._source_session.get(source_id)
                    if skey:
                        break
                if skey is not None:
                    by_q = self._ordinal_by_quote(skey, quote)
                    plans.append(
                        (skey, [by_q] if by_q is not None else [0], 0))
                else:
                    plans.append((None, None, stray_cost))
            else:
                # source-level / bare-id hit — the provider resolves to
                # the quote's turn (ordinal 0 when it can't).
                sid = str(getattr(h, "memory_id", "") or "")
                if not sid:
                    try:
                        from verbatim.memory.types import MemoryRef

                        sid = MemoryRef.parse(
                            getattr(h, "ref", "") or "").source_id
                    except Exception:
                        sid = ""
                skey = self._source_session.get(sid)
                if skey is not None:
                    by_q = self._ordinal_by_quote(skey, quote)
                    plans.append(
                        (skey, [by_q] if by_q is not None else [0], 0))
                else:
                    plans.append((None, None, stray_cost))
        return plans

    def _geic_record(self, hits: Sequence[Any]) -> Optional[Dict[str, Any]]:
        """The geic@B block for ``QueryOutcome.diag`` — the V85-02.04
        expansion replayed over the rank-ordered hit stream at every
        configured budget, scored later by track_r against gold refs."""
        if not self._session_members:
            return None
        meter_name, _m = self._meter()
        plans = self._hit_spans(hits)
        sessions = self._member_tables()
        budgets: Dict[str, Any] = {}
        for b in self.geic_budgets:
            delivered, tokens = geic_expand(
                plans, sessions, window=self.geic_window, budget=b)
            refs_out: List[str] = []
            for skey, poss in delivered.items():
                m = self._session_members.get(skey) or []
                refs_out.extend(
                    m[p] for p in poss if 0 <= p < len(m))
            key = "unbounded" if b is None else str(b)
            budgets[key] = {
                "refs": refs_out,
                "tokens": tokens,
                "n_positions": sum(len(v) for v in delivered.values()),
            }
        return {
            "window": self.geic_window,
            "meter": meter_name,
            "budgets": budgets,
            "n_hits": len(hits),
            "basis": (
                "hits of search(limit=min(pool_limit, facade_max)) — "
                "the rank-order stream V85-02.04 expands"),
        }

    def indexed_refs(self) -> set:
        return set(self._indexed)

    # -- query ---------------------------------------------------------

    def _max_limit(self) -> int:
        try:
            from verbatim.memory import facade as _fac

            return int(getattr(_fac, "_MAX_LIMIT", 64))
        except Exception:
            return 64

    def _member_refs_for_hit(self, hit: Any) -> List[str]:
        """``session_messages`` mode: a hit's covered member refs in
        position order — the same ladder ``_hit_spans`` applies (unit →
        seq/byte-range/quote, claim → source's session, source → quote),
        minus the ±``W_r`` neighbor expansion (``refs`` stays the hit's
        own coverage; neighbors land in ``diag["geic"]``)."""
        plans = self._hit_spans([hit])
        refs: List[str] = []
        for skey, covered, _stray in plans:
            if skey is None:
                continue
            m = self._session_members.get(skey) or []
            refs.extend(
                m[o] for o in (covered or ()) if 0 <= o < len(m))
        return refs

    def _ref_for_hit(self, hit: Any) -> Optional[str]:
        """Hit → corpus ref via source id; claim/view hits resolve
        through ``claim_evidence`` → ``spans`` (v5 scoring convention —
        id mapping only, never content)."""
        if self.ingest_mode == "session_messages":
            mrefs = self._member_refs_for_hit(hit)
            if mrefs:
                return mrefs[0]
        mid = str(getattr(hit, "memory_id", "") or "")
        if mid and mid in self._source_ref:
            return self._source_ref[mid]
        ref = getattr(hit, "ref", "") or ""
        if ref:
            try:
                from verbatim.memory.types import MemoryRef

                sid = MemoryRef.parse(ref).source_id
                if sid in self._source_ref:
                    return self._source_ref[sid]
            except Exception:
                pass
        obj = getattr(hit, "object_ref", "") or ""
        parts = obj.split(".")
        if len(parts) >= 4 and parts[1] == "claim" and self._mem is not None:
            try:
                with self._mem._store.read() as conn:
                    row = conn.execute(
                        "SELECT s.source_id FROM claim_evidence ce"
                        " JOIN spans s ON s.span_id = ce.span_id"
                        " WHERE ce.claim_id = ? LIMIT 1",
                        (parts[2],),
                    ).fetchone()
                if row and row[0] in self._source_ref:
                    return self._source_ref[row[0]]
            except Exception:
                pass
        return None

    def _map_result(self, res: Any) -> Tuple[List[str], List[Any], int]:
        """``SearchResult.items`` → (distinct corpus refs, hits, unmapped)."""
        refs: List[str] = []
        hits: List[Any] = []
        unmapped = 0
        for h in getattr(res, "items", []) or []:
            ref = self._ref_for_hit(h)
            hits.append(h)
            if ref is None:
                unmapped += 1
                # An unmappable delivered unit still consumed a rank
                # slot — keep it addressable, never silently dropped.
                ref = str(getattr(h, "object_ref", "") or f"unmapped:{len(refs)}")
            if ref not in refs:
                refs.append(ref)
        return refs, hits, unmapped

    def _accepts_as_of(self) -> bool:
        """``Memory.search(as_of=…)`` ships under V8-20.01 — detect the
        parameter once per store; a facade without it keeps current
        behavior and the drop is reported in diag, never silent."""
        if self._as_of_supported is None:
            try:
                import inspect

                self._as_of_supported = "as_of" in inspect.signature(
                    self._mem.search
                ).parameters
            except Exception:  # noqa: BLE001 — introspection optional
                self._as_of_supported = False
        return bool(self._as_of_supported)

    def _resolve_as_of(self, task: Any) -> Tuple[Optional[Any], Optional[str]]:
        """(anchor, source) — the task's declared question time wins
        (V8-09.02 precedence); else the task group's final-session
        ``when`` seen at ingest (the LoCoMo convention)."""
        v = _task_as_of(task)
        if v is not None:
            return v, "task"
        gid = _get(task, "group_id", "group")
        if gid is not None:
            us = self._group_as_of.get(str(gid))
            if us is not None:
                return us, "group_final_session"
        return None, None

    def query(self, task: Any, k: int) -> QueryOutcome:
        query = str(getattr(task, "query", task))
        t0 = time.perf_counter()
        error = None
        res = None
        tkw = (
            {"timeout_ms": self.timeout_ms}
            if self.timeout_ms is not None else {}
        )
        # V8-09.02 — pass the caller's anchor only when supplied; a
        # facade without the parameter keeps wall-clock behavior and
        # the unapplied anchor is reported, never silently dropped.
        as_of, as_of_src = self._resolve_as_of(task)
        as_of_applied = False
        as_of_reason = None
        if as_of is not None:
            if self._accepts_as_of():
                tkw["as_of"] = as_of
                as_of_applied = True
            else:
                as_of_reason = (
                    "Memory.search has no as_of parameter (V8-20.01 pending)"
                )
        try:
            res = self._mem.search(
                query, limit=int(k), consistency=self.consistency, **tkw
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        if res is None:
            return QueryOutcome(
                refs=[], surfaced=None, status="error", latency_ms=ms,
                error=error,
                warnings=(
                    ("as_of_unsupported",)
                    if (as_of is not None and not as_of_applied) else ()
                ),
                diag={
                    "as_of": {
                        "requested": as_of,
                        "source": as_of_src,
                        "applied": as_of_applied,
                        "reason": as_of_reason,
                    },
                },
                k=int(k),
            )

        refs, hits, unmapped = self._map_result(res)
        coverage = getattr(res, "coverage", None) or {}
        support = coverage.get("support") or {}
        status = str(getattr(res, "status", "unknown"))
        lanes = coverage.get("lanes") or {}

        # Unmeasured pool-depth probe for attribution (surfaced beyond
        # k).  Same store state → prefix-consistent deliverable pool.
        surfaced = list(refs)
        geic_hits: List[Any] = list(hits)
        geic_basis = "search(limit=k) hits"
        pool = min(self.pool_limit, self._max_limit())
        if pool > int(k):
            try:
                deep = self._mem.search(
                    query, limit=pool, consistency=self.consistency, **tkw
                )
                deep_refs, deep_hits, deep_unmapped = self._map_result(deep)
                unmapped += deep_unmapped
                surfaced = deep_refs or surfaced
                geic_hits = deep_hits
                geic_basis = f"search(limit={pool}) hits"
            except Exception:
                pass  # surfaced stays the measured result's prefix

        geic = None
        if self.ingest_mode == "session_messages":
            geic = self._geic_record(geic_hits)
            if geic is not None:
                geic["basis"] = geic_basis

        delivered = refs[: int(k)]
        delivered_text = " ".join(
            str(getattr(h, "quote", "") or "") for h in hits[: int(k)]
        )
        warnings = [str(w) for w in (getattr(res, "warnings", []) or ())]
        if as_of is not None and not as_of_applied:
            warnings.append("as_of_unsupported")
        return QueryOutcome(
            refs=delivered,
            surfaced=surfaced,
            status=status,
            abstained=(status == "insufficient"
                       or support.get("verdict") == "insufficient"),
            latency_ms=ms,
            n_items=len(delivered),
            delivered_text=delivered_text,
            warnings=tuple(warnings),
            diag={
                "lanes": lanes,
                "verdict": support.get("verdict"),
                "kept": support.get("kept"),
                "suppressed": support.get("suppressed"),
                "omitted": coverage.get("omitted"),
                "unmapped_hits": unmapped,
                "pool_observable": "post_verdict",
                "pool_limit": pool,
                "route": coverage.get("route"),
                "as_of": {
                    "requested": as_of,
                    "source": as_of_src,
                    "applied": as_of_applied,
                    "reason": as_of_reason,
                },
                "geic": geic,
            },
            k=int(k),
        )

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        try:
            if self._mem is not None:
                self._mem.close()
        except Exception:
            pass
        if self._owns_dir:
            shutil.rmtree(self._workdir, ignore_errors=True)


class VerbatimClaimsArm(VerbatimArm):
    """``verbatim_claims`` — the V8-13.01 claims-plane evaluation arm.

    Identical to :class:`VerbatimArm` except ``admission.require_review``
    is pinned ``False`` through the config mapping — the §23 eval flag.
    The product default ``True`` is untouched (invariant 1); harvested
    claims admit ``active`` instead of parking ``pending`` on the
    blanket review gate, so the claim/fact/entity lanes get real inputs
    and the arm measures their contribution (D8-18, O10 input).  The
    override lands in the ingest report's ``config_overrides`` block —
    K74's manifest record and K13's report-row naming.
    """

    name = "verbatim_claims"

    def __init__(self, **kwargs: Any) -> None:
        overrides = dict(kwargs.pop("policy_overrides", None) or {})
        admission = dict(overrides.get("admission") or {})
        admission["require_review"] = False          # V8-13.01 — the arm's pin
        overrides["admission"] = admission
        super().__init__(policy_overrides=overrides, **kwargs)
        self.notes.append(
            "eval override admission.require_review=False (V8-13.01); "
            "product default True unchanged"
        )


class _DrainEnv:
    """Minimal env for eval.v5.harness.drain_memory — the durable-queue
    drain driver needs only ``worker``/``memory``/``notes``/``drain``."""

    def __init__(self, *, worker: str, memory: Any, notes: List[str]) -> None:
        self.worker = worker
        self.memory = memory
        self.notes = notes
        self.drain: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# registry + shared plumbing
# ---------------------------------------------------------------------------


def arm_name(arm: Any) -> str:
    n = getattr(arm, "name", None)
    return str(n() if callable(n) else n) if n is not None else type(arm).__name__


class VerbatimSessionArm(VerbatimArm):
    """``verbatim`` over the session_messages ingest path (V85-04.04) —
    the AMB-parity arm that also emits the geic@B delivery metric."""

    name = "verbatim_msg"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("ingest_mode", "session_messages")
        super().__init__(**kwargs)


ARM_FACTORIES = {
    "verbatim": VerbatimArm,
    "verbatim_msg": VerbatimSessionArm,
    "verbatim_claims": VerbatimClaimsArm,
    "flat_bm25": FlatBM25Arm,
    "bm25": FlatBM25Arm,
    "fts5": FTS5Arm,
    "naive_fts": FTS5Arm,
}


def make_arm(name: str, **kwargs: Any) -> Any:
    """CLI/registry factory — unknown names fail loudly."""
    try:
        cls = ARM_FACTORIES[name]
    except KeyError:
        raise KeyError(
            f"unknown arm {name!r}; registered: {sorted(ARM_FACTORIES)}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "ARM_FACTORIES",
    "ArmTask",
    "DictCorpus",
    "FTS5Arm",
    "FlatBM25Arm",
    "QueryOutcome",
    "VerbatimArm",
    "VerbatimClaimsArm",
    "arm_name",
    "arm_task",
    "corpus_digest",
    "corpus_items",
    "corpus_name",
    "corpus_tasks",
    "item_document",
    "item_metadata",
    "item_ref",
    "item_session",
    "make_arm",
]
