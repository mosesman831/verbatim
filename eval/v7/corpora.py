"""V7 corpus model + dataset loaders (SPEC_V7 §22).

One record model serves every registered dataset:

* :class:`CorpusItem` — one addable unit (a dialogue turn, a haystack
  turn, an action step).  Speakers / sessions / timestamps ride as
  structured fields so adapters preserve them through the public
  ``Memory`` API per V7-09.02 / V7-24.04.  ``image_caption`` keeps
  machine-generated descriptions (LoCoMo ``blip_caption``) separate from
  the speaker's words — indexed as a low-weight field, labeled as
  machine-generated (V7-24.02).
* :class:`CorpusTask` — one scored query: ``query`` text, ``category``,
  gold ``evidence_ids`` (item ids the evidence lives in — LoCoMo dia
  turn ids, qualified per conversation) plus derived
  ``evidence_session_ids`` for session-level recall, and ``answerable``
  (``False`` = the correct behavior is a calibrated refusal, e.g.
  LoCoMo category 5 adversarial questions).  ``group_id`` is the
  dev/test split unit (V7-22.08 — a conversation or question-group,
  never an individual question).
* :class:`Corpus` — sealed items + tasks with a content digest for
  manifest pinning (V7-22.10) and integrity validation: every gold ref
  must resolve to a real item/session.

Gold discipline is carried from eval.v3/v5: arms receive only
:func:`public_task` views — evidence ids, ``answerable``, and
``metadata`` (which holds answer text) raise ``AttributeError`` in arm
code (V7-22.02: gold evidence to the scorer only).

Loaders
-------

``load_corpus(dataset_id, split=None)`` dispatches through the registry:
a dataset that is not ``available`` raises :class:`DatasetUnavailable`
carrying the status string — never a fabricated empty corpus.

* ``locomo_json`` — parses ``locomo10.json`` conversations into
  per-turn items and per-question tasks.  Evidence strings are messy in
  the released file (``"D9:1 D4:4"`` space-joined, ``"D:11:26"``
  double-colon, a bare ``"D"``); :func:`_parse_evidence` normalizes what
  is resolvable and records the rest in ``metadata["unresolved_evidence"]``
  rather than guessing.
* ``longmemeval_json`` — provisional adapter (gated O2): haystack
  sessions → items, ``question_date`` → ``metadata["query_time"]``
  (V7-24.04), per-turn ``has_answer`` → gold evidence ids,
  ``answer_session_ids`` → gold session ids.
* ``"twin"`` — owned corpora import their generator lazily.  Twin
  contract (sibling wave-A workers implement ``eval/v7/twins_*.py``):
  the module exposes ``generate(seed: int = <default>, ...)`` returning
  a dict ``{"units": [...], "tasks": [...], ...}`` (``"items"`` is
  accepted as an alias for ``"units"``), a :class:`Corpus`, or a
  ``(items, tasks)`` tuple; ``ITEMS`` + ``TASKS`` module constants also
  work.  Unit dicts use ``id``/``text``/``kind``/``speaker``/
  ``session_id``/``occurred_us``/``meta``; task dicts use
  ``task_id``/``query``/``kind``/``gold_unit_ids``/``expected_abstain``/
  ``meta`` — extra keys fold into ``metadata`` (scorer-side).
  ``generate`` is called without a seed unless the caller pins one, so
  module defaults apply.  A missing or entry-point-less module yields
  ``unavailable(...)`` status and :class:`DatasetUnavailable` on load —
  never a crash.

Determinism: loaders are pure functions of file bytes / generator seeds;
splits are hash-fixed (V7-22.08); no network, ever.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from eval.v7 import dataset_registry as registry


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class DatasetUnavailable(RuntimeError):
    """Raised by ``load_corpus`` when a dataset is not ``available``.

    Carries the registry status string so callers report
    ``blocked_on_authorization`` / ``missing_file`` /
    ``unavailable(...)`` verbatim (§22.4 scoreboard statuses).
    """

    def __init__(self, dataset_id: str, status: str) -> None:
        self.dataset_id = dataset_id
        self.status = status
        super().__init__(f"dataset {dataset_id!r}: {status}")


# ---------------------------------------------------------------------------
# categories — V7-22.07 source-code mapping, printed with ids in tables
# ---------------------------------------------------------------------------

#: LoCoMo numeric category id -> name (V7-22.07).
LOCOMO_CATEGORY_NAMES: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

#: Expected question counts per category id in locomo10.json (V7-22.07).
#: Adapters re-verify at build time; observed counts land in corpus
#: metadata so a drifted file is loud in the report.
LOCOMO_CATEGORY_COUNTS: Dict[int, int] = {
    1: 282,
    2: 321,
    3: 96,
    4: 841,
    5: 446,
}

#: LongMemEval-S question types (V7-24.02 table; adapter passes through).
LME_TYPES: Tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)

#: Corpus.metadata key under which ``load_locomo`` carries the raw
#: ``observation`` layer (``session_N_observation`` speaker tables of
#: ``[assertion, "D<s>:<t>"]`` pairs) for the eval-only oracle
#: (V75-04.06).  Scorer-side gold: consumed by
#: ``eval.v7.locomo_oracle`` only — never arm-facing, never runtime.
LOCOMO_ORACLE_META_KEY = "oracle_observations"


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusItem:
    """One addable unit — typically a dialogue turn.

    ``id`` is corpus-unique (external ids are qualified with their
    group, e.g. ``conv-26:D1:3``).  ``speaker``/``session_id``/``when``
    preserve the dialogue structure adapters must carry into ``add``
    calls (V7-22.02).  ``image_caption`` is machine-generated
    description text — a separate low-weight field, never the speaker's
    words (V7-24.02).  ``group_id`` is the split unit (V7-22.08); items
    without one are shared across splits.
    """

    id: str
    text: str
    speaker: Optional[str] = None
    session_id: Optional[str] = None
    when: Optional[str] = None
    image_caption: Optional[str] = None
    group_id: Optional[str] = None
    kind: str = "dialogue_turn"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("corpus item requires non-empty id")
        if not isinstance(self.text, str):
            raise ValueError(f"corpus item {self.id!r}: text must be str")

    def render_text(self) -> str:
        """Probe-style add payload: ``[when] speaker: text`` (+ image note)."""
        head = f"[{self.when}] " if self.when else ""
        who = f"{self.speaker}: " if self.speaker else ""
        tail = (
            f" [shares image: {self.image_caption}]"
            if self.image_caption
            else ""
        )
        return f"{head}{who}{self.text}{tail}"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorpusItem":
        return cls(
            id=d["id"],
            text=d["text"],
            speaker=d.get("speaker"),
            session_id=d.get("session_id"),
            when=d.get("when"),
            image_caption=d.get("image_caption"),
            group_id=d.get("group_id"),
            kind=d.get("kind", "dialogue_turn"),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "speaker": self.speaker,
            "session_id": self.session_id,
            "when": self.when,
            "image_caption": self.image_caption,
            "group_id": self.group_id,
            "kind": self.kind,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CorpusTask:
    """One scored query.

    ``evidence_ids`` are gold item refs (turn ids / unit ids);
    ``evidence_session_ids`` the session-level gold set (V7-22.12 scores
    both granularities).  ``answerable=False`` marks a correct-refusal
    probe (LoCoMo adversarial cat 5, LME ``_abs``): delivered items are
    then scored by the refusal metric, not by recall.  ``group_id`` is
    the V7-22.08 split unit.  ``metadata`` is scorer-side (answer text,
    category ids, unresolved refs, ``query_time``) — gold, never
    arm-facing.
    """

    task_id: str
    query: str
    category: str = "unknown"
    evidence_ids: Tuple[str, ...] = ()
    evidence_session_ids: Tuple[str, ...] = ()
    answerable: bool = True
    group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("corpus task requires non-empty task_id")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"{self.task_id}: query required")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError(f"{self.task_id}: duplicate evidence ids")
        if len(set(self.evidence_session_ids)) != len(
            self.evidence_session_ids
        ):
            raise ValueError(
                f"{self.task_id}: duplicate evidence session ids"
            )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorpusTask":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            category=d.get("category", "unknown"),
            evidence_ids=tuple(d.get("evidence_ids") or ()),
            evidence_session_ids=tuple(
                d.get("evidence_session_ids") or ()
            ),
            answerable=bool(d.get("answerable", True)),
            group_id=d.get("group_id"),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "category": self.category,
            "evidence_ids": list(self.evidence_ids),
            "evidence_session_ids": list(self.evidence_session_ids),
            "answerable": self.answerable,
            "group_id": self.group_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Corpus:
    """A sealed corpus: addable items + scored tasks + content digest.

    Validation is construction-time (the v3 convention — a corpus that
    cannot be scored exactly is worse than none): ids are unique and
    every gold ref resolves to a real item/session of *this* corpus.
    """

    name: str
    dataset_id: str
    items: Tuple[CorpusItem, ...]
    tasks: Tuple[CorpusTask, ...]
    split: Optional[str] = None  # None | "dev" | "test"
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set = set()
        dupes: set = set()
        for i in self.items:
            (dupes if i.id in seen else seen).add(i.id)
        if dupes:
            raise ValueError(
                f"corpus {self.name!r}: duplicate item ids "
                f"{sorted(dupes)[:5]}"
            )
        item_ids = seen
        seen_t: set = set()
        dupes_t: set = set()
        for t in self.tasks:
            (dupes_t if t.task_id in seen_t else seen_t).add(t.task_id)
        if dupes_t:
            raise ValueError(
                f"corpus {self.name!r}: duplicate task ids "
                f"{sorted(dupes_t)[:5]}"
            )
        have_items = item_ids
        have_sessions = {
            i.session_id for i in self.items if i.session_id is not None
        }
        for t in self.tasks:
            for ref in t.evidence_ids:
                if ref not in have_items:
                    raise ValueError(
                        f"{self.name}:{t.task_id}: evidence id {ref!r} "
                        "not in corpus items"
                    )
            for ref in t.evidence_session_ids:
                if ref not in have_sessions:
                    raise ValueError(
                        f"{self.name}:{t.task_id}: evidence session "
                        f"{ref!r} not in corpus sessions"
                    )

    def __iter__(self) -> Iterator[CorpusTask]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def item_by_id(self) -> Dict[str, CorpusItem]:
        return {i.id: i for i in self.items}

    def task_by_id(self) -> Dict[str, CorpusTask]:
        return {t.task_id: t for t in self.tasks}

    def of_category(self, *cats: str) -> Tuple[CorpusTask, ...]:
        want = set(cats)
        return tuple(t for t in self.tasks if t.category in want)

    def groups(self) -> Tuple[str, ...]:
        """Distinct split-group ids carried by items and tasks."""
        seen = {
            r.group_id
            for r in (*self.items, *self.tasks)
            if r.group_id is not None
        }
        return tuple(sorted(seen))

    def digest(self) -> str:
        """Content fingerprint — recorded in every run manifest
        (V7-22.10 dataset digests and split)."""
        canon = {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "items": [i.to_dict() for i in self.items],
            "tasks": [t.to_dict() for t in self.tasks],
        }
        blob = json.dumps(
            canon, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def corpus_stats(corpus: Corpus) -> Dict[str, Any]:
    """Published corpus shape — denominators a reviewer can audit."""
    cats: Dict[str, int] = {}
    for t in corpus.tasks:
        cats[t.category] = cats.get(t.category, 0) + 1
    return {
        "name": corpus.name,
        "dataset_id": corpus.dataset_id,
        "split": corpus.split,
        "items": len(corpus.items),
        "tasks": len(corpus.tasks),
        "answerable": sum(1 for t in corpus.tasks if t.answerable),
        "unanswerable": sum(1 for t in corpus.tasks if not t.answerable),
        "categories": dict(sorted(cats.items())),
        "groups": list(corpus.groups()),
        "digest": corpus.digest(),
    }


# ---------------------------------------------------------------------------
# public views — gold stays scorer-side (V7-22.02, V5 rule carried)
# ---------------------------------------------------------------------------

#: Task attributes an arm may read.  ``category`` stays public per the
#: v3/v5 convention (it names the metric bucket); the *answerable* flag
#: is gold — knowing "this one is adversarial" is the answer to the
#: refusal probe, so arms must not see it.
_TASK_PUBLIC = frozenset({"task_id", "query", "category", "group_id"})
_TASK_GOLD = frozenset(
    {
        "evidence_ids",
        "evidence_session_ids",
        "answerable",
        "metadata",
        "answer",
        "expected_abstain",
    }
)


class PublicTaskView:
    """Whitelisted-attribute proxy over :class:`CorpusTask`.

    Public names forward; gold names raise an informative
    ``AttributeError`` — a gold read in arm code is loud in review and
    fails in CI, never silent.  ``_wrapped`` stays reachable: the view
    polices the interface, not the memory model.
    """

    __slots__ = ("_wrapped",)

    def __init__(self, wrapped: CorpusTask) -> None:
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        if name in _TASK_PUBLIC:
            return getattr(self._wrapped, name)
        if name in _TASK_GOLD:
            raise AttributeError(
                f"PublicTaskView.{name} is evaluation gold — arms "
                "receive only the public task surface (V7-22.02)"
            )
        raise AttributeError(
            f"'PublicTaskView' object has no attribute {name!r} "
            f"(wraps CorpusTask)"
        )

    def __repr__(self) -> str:
        return f"PublicTaskView({self._wrapped!r})"


def public_task(task: CorpusTask) -> PublicTaskView:
    """Wrap a task for handoff to an arm — the only task object an arm
    may read (V7-22.02 gold-to-scorer rule)."""
    return PublicTaskView(task)


# ---------------------------------------------------------------------------
# loading — dispatch through the registry
# ---------------------------------------------------------------------------


def load_corpus(
    dataset_id: str,
    split: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    seed: Optional[int] = None,
) -> Corpus:
    """Load a registered dataset as a :class:`Corpus`.

    ``split`` is ``None`` (whole dataset), ``"dev"``, or ``"test"``;
    partitioning follows the entry's :class:`SplitSpec` on each record's
    ``group_id`` (records with no group are shared across splits).  A
    dataset whose status is not ``available`` raises
    :class:`DatasetUnavailable` — the status string is the honest answer.
    """
    entry = registry.get(dataset_id)
    st = registry.status(entry, env)
    if st != registry.STATUS_AVAILABLE:
        raise DatasetUnavailable(dataset_id, st)
    if split not in (None, "dev", "test"):
        raise ValueError(f"split must be None|'dev'|'test', got {split!r}")
    if split is not None and entry.splits is None:
        raise ValueError(f"{dataset_id}: no split spec registered")

    if entry.loader_kind == "twin":
        corpus = _load_twin(entry, seed=seed)
    elif entry.loader_kind == "locomo_json":
        corpus = load_locomo(
            registry.resolved_path(entry, env), dataset_id=dataset_id
        )
    elif entry.loader_kind == "longmemeval_json":
        corpus = load_longmemeval(
            registry.resolved_path(entry, env), dataset_id=dataset_id
        )
    else:
        raise DatasetUnavailable(
            dataset_id, registry.unavailable("adapter not implemented")
        )
    return _apply_split(corpus, entry, split)


def _apply_split(
    corpus: Corpus, entry: registry.DatasetEntry, split: Optional[str]
) -> Corpus:
    if split is None:
        return corpus
    keep = {
        g
        for g in corpus.groups()
        if registry.split_for_group(entry.dataset_id, g) == split
    }
    tasks = tuple(
        t for t in corpus.tasks if t.group_id is None or t.group_id in keep
    )
    # Items in a kept group stay; so do items a kept task references
    # (a multi-session task's secondary evidence must not be filtered
    # out from under it) and ungrouped shared context.
    needed_ids = {r for t in tasks for r in t.evidence_ids}
    needed_sessions = {
        s for t in tasks for s in t.evidence_session_ids
    }
    items = tuple(
        i
        for i in corpus.items
        if i.group_id is None
        or i.group_id in keep
        or i.id in needed_ids
        or i.session_id in needed_sessions
    )
    return replace(
        corpus,
        name=f"{corpus.name}:{split}",
        items=items,
        tasks=tasks,
        split=split,
    )


# ---------------------------------------------------------------------------
# twins — lazy import, ImportError never crashes (concurrent wave-A workers)
# ---------------------------------------------------------------------------


def _load_twin(
    entry: registry.DatasetEntry, seed: Optional[int] = None
) -> Corpus:
    try:
        mod = importlib.import_module(entry.loader)
    except ImportError as exc:
        raise DatasetUnavailable(
            entry.dataset_id,
            registry.unavailable(f"twin import failed: {exc}"),
        ) from exc

    gen = getattr(mod, "generate", None) or getattr(
        mod, "build_corpus", None
    )
    if callable(gen):
        # seed=None -> module default; generators own their DEFAULT_SEED.
        if seed is None:
            out = gen()
        else:
            try:
                out = gen(seed=seed)
            except TypeError:
                out = gen(seed)
    elif hasattr(mod, "ITEMS") and hasattr(mod, "TASKS"):
        out = {"items": mod.ITEMS, "tasks": mod.TASKS}
    elif hasattr(mod, "CORPUS"):
        out = mod.CORPUS
    else:
        raise DatasetUnavailable(
            entry.dataset_id,
            registry.unavailable(
                f"{entry.loader}: no generate()/ITEMS+TASKS entry point"
            ),
        )
    return _normalize_corpus(out, entry)


#: Task dict keys consumed directly; everything else folds into
#: ``metadata`` (scorer-side gold — e.g. ``gold_rule_ids``,
#: ``expected_action``, ``preference``, ``task_text``).
_TASK_KEYS = frozenset(
    {
        "task_id",
        "query",
        "task_text",
        "category",
        "kind",
        "evidence_ids",
        "evidence_session_ids",
        "gold_unit_ids",
        "gold_session_ids",
        "answerable",
        "expected_abstain",
        "group_id",
        "group",
        "question_group",
        "metadata",
        "meta",
    }
)

_ITEM_KEYS = frozenset(
    {
        "id",
        "text",
        "speaker",
        "session_id",
        "when",
        "occurred_us",
        "image_caption",
        "group_id",
        "kind",
        "metadata",
        "meta",
    }
)


def _coerce_item(d: Any) -> CorpusItem:
    if isinstance(d, CorpusItem):
        return d
    if not isinstance(d, Mapping):
        raise ValueError(f"twin item is not a mapping: {type(d).__name__}")
    meta = dict(d.get("metadata") or d.get("meta") or {})
    for k, v in d.items():
        if k not in _ITEM_KEYS:
            meta.setdefault(k, v)
    when = d.get("when")
    if when is None and d.get("occurred_us") is not None:
        meta.setdefault("occurred_us", d["occurred_us"])
    return CorpusItem(
        id=d.get("id", ""),
        text=d.get("text", ""),
        speaker=d.get("speaker"),
        session_id=d.get("session_id"),
        when=when,
        image_caption=d.get("image_caption"),
        group_id=d.get("group_id") or d.get("session_id"),
        kind=d.get("kind", "dialogue_turn"),
        metadata=meta,
    )


def _coerce_task(d: Any, item_by_id: Mapping[str, CorpusItem]) -> CorpusTask:
    if isinstance(d, CorpusTask):
        return d
    if not isinstance(d, Mapping):
        raise ValueError(f"twin task is not a mapping: {type(d).__name__}")
    meta = dict(d.get("metadata") or d.get("meta") or {})
    for k, v in d.items():
        if k not in _TASK_KEYS:
            meta.setdefault(k, v)
    ev_ids = tuple(d.get("evidence_ids") or d.get("gold_unit_ids") or ())
    ev_sessions = tuple(
        d.get("evidence_session_ids") or d.get("gold_session_ids") or ()
    )
    if not ev_sessions:
        ev_sessions = tuple(
            sorted(
                {
                    item_by_id[r].session_id
                    for r in ev_ids
                    if r in item_by_id
                    and item_by_id[r].session_id is not None
                }
            )
        )
    if "answerable" in d:
        answerable = bool(d["answerable"])
    else:
        answerable = not bool(d.get("expected_abstain"))
    # Split group (V7-22.08): explicit field, else the session this task
    # is about (first gold or distractor unit's session) — questions
    # sharing evidence stay in one group.  Ungrouped tasks share one
    # deterministic bucket rather than partitioning per question.
    group = (
        d.get("group_id")
        or d.get("group")
        or d.get("question_group")
        or meta.get("group_id")
    )
    if group is None:
        for ref in (*ev_ids, *(d.get("distractor_ids") or ())):
            unit = item_by_id.get(ref)
            if unit is not None and unit.session_id is not None:
                group = unit.session_id
                break
    return CorpusTask(
        task_id=d.get("task_id", ""),
        query=d.get("query") or d.get("task_text") or "",
        category=d.get("category") or d.get("kind") or "unknown",
        evidence_ids=ev_ids,
        evidence_session_ids=ev_sessions,
        answerable=answerable,
        group_id=group or "ungrouped",
        metadata=meta,
    )


def _locomo_twin_items_tasks(out: Mapping[str, Any]):
    """``{sessions, turns, facts, questions}`` -> (item dicts, task dicts).

    The locomo-like twin emits one conversation graph: ``turns`` are the
    addable units (``D<session>:<turn>`` ids), ``questions`` carry
    ``evidence`` refs in the same id space.
    """
    items = []
    for t in out.get("turns") or ():
        items.append(
            {
                "id": t.get("turn_id") or t.get("id", ""),
                "text": t.get("text", ""),
                "speaker": t.get("speaker"),
                "session_id": t.get("session_id"),
                "when": t.get("timestamp") or t.get("when"),
                "group_id": t.get("session_id"),
                "kind": t.get("kind", "dialogue_turn"),
                "metadata": {
                    k: v
                    for k, v in t.items()
                    if k in ("timestamp_us", "turn_index", "role", "kind")
                },
            }
        )
    tasks = []
    for q in out.get("questions") or ():
        tasks.append(
            {
                "task_id": q.get("qid") or q.get("task_id") or q.get("id", ""),
                "query": q.get("query", ""),
                "category": q.get("category") or "unknown",
                "evidence_ids": tuple(q.get("evidence") or ()),
                "answerable": bool(q.get("answerable", True)),
                "metadata": {
                    k: v
                    for k, v in q.items()
                    if k
                    in (
                        "answer",
                        "subtype",
                        "question_time",
                        "question_time_us",
                        "unresolved_evidence",
                    )
                },
            }
        )
    return items, tasks


def _lme_twin_items_tasks(out: Mapping[str, Any]):
    """``{questions:[{sessions:[{turns}]}]}`` -> (item dicts, task dicts).

    The lme-like twin emits self-contained per-question haystacks (the
    LongMemEval shape).  Each question's turns flatten into items keyed
    ``group_id=<qid>`` so a split never separates a question from its
    haystack (V7-22.08).
    """
    items = []
    tasks = []
    for q in out.get("questions") or ():
        qid = q.get("id") or q.get("qid") or q.get("task_id", "")
        for s in q.get("sessions") or ():
            for t in s.get("turns") or ():
                items.append(
                    {
                        "id": t.get("turn_id") or t.get("id", ""),
                        "text": t.get("text", ""),
                        "speaker": t.get("speaker"),
                        "session_id": s.get("session_id"),
                        "when": s.get("date") or t.get("when"),
                        "group_id": qid,
                        "kind": "dialogue_turn",
                        "metadata": {
                            "timestamp_us": t.get("timestamp_us"),
                            "session_date": s.get("date"),
                        },
                    }
                )
        tasks.append(
            {
                "task_id": qid,
                "query": q.get("query", ""),
                "category": q.get("category") or "unknown",
                "evidence_ids": tuple(q.get("gold_turn_ids") or ()),
                "evidence_session_ids": tuple(q.get("gold_session_ids") or ()),
                "answerable": bool(q.get("answerable", True)),
                "group_id": qid,
                "metadata": {
                    k: v
                    for k, v in q.items()
                    if k
                    in (
                        "answer",
                        "subtype",
                        "expected_abstain",
                        "question_date",
                        "question_time_us",
                        "checks",
                        "preference",
                        "knowledge_update",
                        "temporal",
                        "gold_evidence",
                    )
                },
            }
        )
    return items, tasks


def _normalize_corpus(out: Any, entry: registry.DatasetEntry) -> Corpus:
    """Coerce a twin generator's return into a validated Corpus."""
    meta: Dict[str, Any] = {}
    if isinstance(out, Corpus):
        return out
    if isinstance(out, Mapping):
        name = out.get("name") or entry.dataset_id
        meta = dict(out.get("metadata") or {})
        # carry generator bookkeeping for manifest pinning (V7-22.10)
        for k in ("generator", "constants", "seed", "params", "stats"):
            if k in out:
                meta.setdefault(k, out[k])
        if "digest" in out:
            meta.setdefault("generator_digest", out["digest"])
        if "units" in out or "items" in out or "tasks" in out:
            items = out.get("units", out.get("items", ()))
            tasks = out.get("tasks", ())
        elif "turns" in out and "questions" in out:
            items, tasks = _locomo_twin_items_tasks(out)
        elif "questions" in out:
            items, tasks = _lme_twin_items_tasks(out)
        else:
            items, tasks = (), ()
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        items, tasks = out
        name = entry.dataset_id
    else:
        raise DatasetUnavailable(
            entry.dataset_id,
            registry.unavailable(
                f"{entry.loader}: cannot normalize {type(out).__name__}"
            ),
        )
    items = tuple(_coerce_item(i) for i in items)
    item_by_id = {i.id: i for i in items}
    tasks = tuple(_coerce_task(t, item_by_id) for t in tasks)
    return Corpus(
        name=name,
        dataset_id=entry.dataset_id,
        items=items,
        tasks=tasks,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# LoCoMo (locomo10.json — CC BY-NC 4.0, O1-gated, local-only)
# ---------------------------------------------------------------------------

_DIA_REF = re.compile(r"D:?(\d+):(\d+)")


def _parse_evidence(raw: Iterable[Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Normalize LoCoMo evidence strings to ``D<session>:<turn>`` refs.

    The released file mixes ``["D1:3"]``, space-joined multi-refs
    (``"D9:1 D4:4"``), a double-colon form (``"D:11:26"``), and at least
    one degenerate ``"D"``.  Resolvable tokens normalize; the rest are
    returned as ``unresolved`` for the task's metadata — never dropped
    silently, never guessed.
    """
    refs: list = []
    unresolved: list = []
    for piece in raw or ():
        for tok in re.split(r"[;,\s]+", str(piece)):
            if not tok:
                continue
            m = _DIA_REF.fullmatch(tok)
            if m:
                ref = f"D{int(m.group(1))}:{int(m.group(2))}"
                if ref not in refs:
                    refs.append(ref)
            else:
                unresolved.append(tok)
    return tuple(refs), tuple(unresolved)


def _session_sort_key(name: str) -> int:
    m = re.fullmatch(r"session_(\d+)", name)
    return int(m.group(1)) if m else 1 << 30


def load_locomo(
    path: Optional[str], *, dataset_id: str = "locomo"
) -> Corpus:
    """Parse ``locomo10.json`` into items (dialogue turns) and tasks.

    Turn ids ``D<s>:<t>`` are qualified as ``{sample_id}:D<s>:<t>`` so
    they are corpus-unique; sessions are ``{sample_id}/session_<s>``.
    Category ids map through :data:`LOCOMO_CATEGORY_NAMES`; category 5
    (adversarial) sets ``answerable=False`` — its evidence ids are the
    premise-related turns used for speaker-mismatch checks (V7-24.01),
    not answer support.  Questions whose every evidence token is
    unresolvable load with empty gold and a
    ``metadata["unresolved_evidence"]`` record.

    The released per-conversation ``observation`` layer
    (``session_N_observation`` → ``{speaker: [[assertion, ref], ...]}``)
    rides into ``Corpus.metadata[LOCOMO_ORACLE_META_KEY]`` verbatim —
    the eval-only labeled oracle fact set of V75-04.06, built into
    typed facts by ``eval.v7.locomo_oracle`` without re-reading the
    gated file.  Scorer-side only; runtime code never sees it
    (V7-22.18).
    """
    if not path or not os.path.isfile(path):
        raise DatasetUnavailable(dataset_id, registry.STATUS_MISSING)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    items: list = []
    tasks: list = []
    cat_counts: Dict[str, int] = {}
    for conv in data:
        sid = conv.get("sample_id", "unknown")
        conv_obj = conv.get("conversation", {})
        session_names = sorted(
            (k for k in conv_obj if _session_sort_key(k) < 1 << 30),
            key=_session_sort_key,
        )
        item_ids: set = set()
        for sname in session_names:
            n = _session_sort_key(sname)
            when = conv_obj.get(f"{sname}_date_time")
            session_id = f"{sid}/session_{n}"
            for turn in conv_obj.get(sname) or ():
                dia = turn.get("dia_id")
                if not dia:
                    continue
                iid = f"{sid}:{dia}"
                item_ids.add(iid)
                items.append(
                    CorpusItem(
                        id=iid,
                        text=turn.get("text", ""),
                        speaker=turn.get("speaker"),
                        session_id=session_id,
                        when=when,
                        image_caption=turn.get("blip_caption"),
                        group_id=sid,
                        kind="dialogue_turn",
                        metadata={
                            "dia_id": dia,
                            "session_n": n,
                            "sample_id": sid,
                        },
                    )
                )
        for qi, q in enumerate(conv.get("qa") or ()):
            cat_id = q.get("category")
            cat = LOCOMO_CATEGORY_NAMES.get(cat_id, f"unknown_{cat_id}")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            refs, unresolved = _parse_evidence(q.get("evidence"))
            ev_ids = tuple(
                f"{sid}:{r}" for r in refs if f"{sid}:{r}" in item_ids
            )
            dropped = tuple(
                r for r in refs if f"{sid}:{r}" not in item_ids
            )
            ev_sessions = tuple(
                sorted(
                    {
                        f"{sid}/session_{int(m.group(1))}"
                        for iid in ev_ids
                        for m in [_DIA_REF.fullmatch(iid.split(":", 1)[1])]
                        if m
                    }
                )
            )
            adversarial = cat_id == 5
            answer = q.get("answer", q.get("adversarial_answer"))
            tasks.append(
                CorpusTask(
                    task_id=f"{sid}/q{qi:03d}",
                    query=q.get("question", ""),
                    category=cat,
                    evidence_ids=ev_ids,
                    evidence_session_ids=ev_sessions,
                    answerable=not adversarial,
                    group_id=sid,
                    metadata={
                        "sample_id": sid,
                        "category_id": cat_id,
                        "answer": answer,
                        "adversarial": adversarial,
                        "raw_evidence": list(q.get("evidence") or ()),
                        "unresolved_evidence": unresolved + dropped,
                    },
                )
            )
    meta = {
        "conversations": len(data),
        "category_counts_observed": dict(sorted(cat_counts.items())),
        "category_counts_expected": {
            LOCOMO_CATEGORY_NAMES[k]: v
            for k, v in sorted(LOCOMO_CATEGORY_COUNTS.items())
        },
        "speakers": {
            c.get("sample_id"): (
                c.get("conversation", {}).get("speaker_a"),
                c.get("conversation", {}).get("speaker_b"),
            )
            for c in data
        },
        # V75-04.06: raw observation layer for the eval-only oracle.
        LOCOMO_ORACLE_META_KEY: {
            c.get("sample_id"): c.get("observation") or {}
            for c in data
        },
    }
    return Corpus(
        name=dataset_id,
        dataset_id=dataset_id,
        items=tuple(items),
        tasks=tuple(tasks),
        source_path=path,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# LongMemEval (longmemeval_{s,m,oracle}.json — O2-gated; provisional adapter)
# ---------------------------------------------------------------------------


def load_longmemeval(
    path: Optional[str], *, dataset_id: str = "longmemeval_s"
) -> Corpus:
    """Parse a LongMemEval json file (provisional — the adapter is
    exercised only once O2 lands and the released file is reviewed).

    Mapping per V7-24.04: each ``haystack_sessions[i][j]`` turn becomes
    an item with ``speaker=role`` and ``when=haystack_dates[i]``;
    per-turn ``has_answer`` flags become gold evidence ids;
    ``answer_session_ids`` become gold session ids; ``question_date`` is
    recorded as ``metadata["query_time"]`` for the search call.
    Abstention questions (``*_abs`` ids / falsy ``has_answer`` at the
    question level) set ``answerable=False``.
    """
    if not path or not os.path.isfile(path):
        raise DatasetUnavailable(dataset_id, registry.STATUS_MISSING)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    items: list = []
    tasks: list = []
    cat_counts: Dict[str, int] = {}
    for qi, q in enumerate(data):
        qid = q.get("question_id") or f"q{qi:04d}"
        sessions = q.get("haystack_sessions") or ()
        sids = q.get("haystack_session_ids") or ()
        dates = q.get("haystack_dates") or ()
        ev_ids: list = []
        for si, sess in enumerate(sessions):
            sid_part = sids[si] if si < len(sids) else f"s{si}"
            session_id = f"{qid}/{sid_part}"
            when = dates[si] if si < len(dates) else None
            for ti, turn in enumerate(sess or ()):
                iid = f"{session_id}/t{ti}"
                items.append(
                    CorpusItem(
                        id=iid,
                        text=turn.get("content", ""),
                        speaker=turn.get("role"),
                        session_id=session_id,
                        when=str(when) if when is not None else None,
                        group_id=qid,
                        kind="dialogue_turn",
                        metadata={
                            "question_id": qid,
                            "session_index": si,
                            "turn_index": ti,
                            "has_answer": bool(turn.get("has_answer")),
                        },
                    )
                )
                if turn.get("has_answer"):
                    ev_ids.append(iid)
        ev_sessions = tuple(
            sorted(f"{qid}/{s}" for s in (q.get("answer_session_ids") or ()))
        )
        qtype = q.get("question_type", "unknown")
        cat_counts[qtype] = cat_counts.get(qtype, 0) + 1
        answerable = not str(qid).endswith("_abs") and bool(
            q.get("has_answer", True)
        )
        tasks.append(
            CorpusTask(
                task_id=qid,
                query=q.get("question", ""),
                category=qtype,
                evidence_ids=tuple(ev_ids),
                evidence_session_ids=ev_sessions,
                answerable=answerable,
                group_id=qid,
                metadata={
                    "question_id": qid,
                    "question_type": qtype,
                    "answer": q.get("answer"),
                    "query_time": q.get("question_date"),
                },
            )
        )
    return Corpus(
        name=dataset_id,
        dataset_id=dataset_id,
        items=tuple(items),
        tasks=tuple(tasks),
        source_path=path,
        metadata={
            "questions": len(data),
            "category_counts_observed": dict(sorted(cat_counts.items())),
        },
    )
