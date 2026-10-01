"""Deterministic consumer-memory corpus for the V5 eval harness
(SPEC_V5 §20.11, §21, §32).

The V5 slice is *personal consumer* memory — preferences, facts,
events, identifiers, corrections — driven through the public
``verbatim.Memory`` route. Unlike the v3 coding-task corpus, every item
ingests through ``Memory.add`` and every query through
``Memory.search``; there is no operator admission step (the consumer
contract auto-admits what it accepts).

Record model:

* :class:`CorpusItem` — one ``add`` payload. ``infer`` selects the
  typed-memory branch; ``supersedes`` names a prior item id the add
  replaces through the real ``replaces=`` transition; ``forget`` marks
  items deleted after seeding (held/deleted distractors, §20.11);
  ``hold`` marks items suppressed through the source-state path.
* :class:`ConsumerTask` — one scored query with gold ids
  (``expected_ids``), an optional abstention expectation, and
  ``forbidden_ids`` (items that must NOT be delivered — superseded
  predecessors, forgotten distractors, foreign content).
* :func:`seed_corpus` — deterministic generator. A fixed hand-written
  core set covers the §20.11 query mix (identifiers, ordinary lexical,
  unsupported paraphrase, no-answer, updates/conflicts, Unicode,
  temporal intent, held/deleted distractors); generated fillers scale
  the retained-item count for the latency envelopes. Same seed → same
  bytes → same digest, recorded in every report.

Gold discipline mirrors ``eval/v3/corpus.py``: arms receive only
:func:`public_task` / :func:`public_item` views — expected ids,
abstention flags, forbidden ids, and lifecycle metadata raise
``AttributeError`` in arm code. Scoring code keeps the real records.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

#: Task categories (the §20.11 query mix, consumer-corpus edition).
CATEGORIES: Tuple[str, ...] = (
    "identifier",        # exact identifier recall (tickets, versions)
    "lexical",           # ordinary keyword fact lookup
    "paraphrase",        # unsupported paraphrase — honest hard slice
    "no_answer",         # nothing stored supports an answer
    "update",            # supersession: current beats stale
    "unicode",           # multibyte UTF-8 round-trip
    "temporal",          # event-ordering / when-intent
    "forget_distractor", # forgotten items must stay gone
)


@dataclass(frozen=True)
class CorpusItem:
    """One ``Memory.add`` payload.

    ``supersedes`` names the corpus id of the item this add replaces
    (the harness issues ``add(..., replaces=<that item's ref>)`` — the
    real §14.1 transition, not a fixture shortcut). ``forget`` marks the
    item for deletion during setup — a seeded distractor that must stay
    invisible to scoring queries (§20.11 held/deleted distractors).
    ``hold`` marks an item suppressed through ``inspect``-visible
    lifecycle — reported, never silently dropped from the corpus.
    """

    id: str
    text: str
    infer: bool = True
    tags: Tuple[str, ...] = ()
    supersedes: Optional[str] = None
    forget: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusItem":
        if not isinstance(d.get("id"), str) or not d["id"]:
            raise ValueError("corpus item requires non-empty id")
        if not isinstance(d.get("text"), str) or not d["text"]:
            raise ValueError(f"corpus item {d.get('id')!r} requires text")
        return cls(
            id=d["id"],
            text=d["text"],
            infer=bool(d.get("infer", True)),
            tags=tuple(d.get("tags") or ()),
            supersedes=d.get("supersedes"),
            forget=bool(d.get("forget", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "infer": self.infer,
            "tags": list(self.tags),
            "supersedes": self.supersedes,
            "forget": self.forget,
        }


@dataclass(frozen=True)
class ConsumerTask:
    """One scored consumer-route query.

    ``expected_ids`` are corpus item ids the query should surface;
    ``expected_abstain`` declares an unanswerable probe (a delivered
    item is a spurious answer); ``forbidden_ids`` are items that must
    never be delivered for this query (superseded predecessors,
    forgotten distractors) — each delivered forbidden id is a
    correctness violation, not just a precision cost.
    """

    task_id: str
    query: str
    category: str = "lexical"
    expected_ids: Tuple[str, ...] = ()
    expected_abstain: bool = False
    forbidden_ids: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.task_id}: unknown category {self.category!r}")
        if self.expected_abstain and self.expected_ids:
            raise ValueError(
                f"{self.task_id}: expected_abstain forbids expected_ids"
            )
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task requires task_id")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"{self.task_id}: query required")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConsumerTask":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            category=d.get("category", "lexical"),
            expected_ids=tuple(d.get("expected_ids") or ()),
            expected_abstain=bool(d.get("expected_abstain", False)),
            forbidden_ids=tuple(d.get("forbidden_ids") or ()),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "category": self.category,
            "expected_ids": list(self.expected_ids),
            "expected_abstain": self.expected_abstain,
            "forbidden_ids": list(self.forbidden_ids),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ConsumerCorpus:
    """A sealed corpus: items + scored tasks + a content digest."""

    name: str
    seed: int
    items: Tuple[CorpusItem, ...]
    tasks: Tuple[ConsumerTask, ...]

    def __iter__(self) -> Iterator[CorpusItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def item_by_id(self) -> dict[str, CorpusItem]:
        return {i.id: i for i in self.items}

    def task_by_id(self) -> dict[str, ConsumerTask]:
        return {t.task_id: t for t in self.tasks}

    def of_category(self, *cats: str) -> Tuple[ConsumerTask, ...]:
        want = set(cats)
        return tuple(t for t in self.tasks if t.category in want)

    def digest(self) -> str:
        canon = {
            "name": self.name,
            "seed": self.seed,
            "items": [i.to_dict() for i in self.items],
            "tasks": [t.to_dict() for t in self.tasks],
        }
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the hand-written core — the §20.11 mix at any scale
# ---------------------------------------------------------------------------

_CORE_ITEMS: Tuple[CorpusItem, ...] = (
    # lexical facts
    CorpusItem("core-wifi", "The home wifi password is hunter2-blue."),
    CorpusItem("core-vet", "The cat's vet is Dr. Patel at Riverside Clinic; "
                           "her number is 555-0148."),
    CorpusItem("core-car", "The car's insurance renewal is due every "
                           "November; policy number INS-44170."),
    CorpusItem("core-run", "I run five kilometres every Tuesday and "
                           "Thursday morning before work."),
    # identifiers
    CorpusItem("core-ticket", "Jira ticket ENG-4821 tracks the login "
                              "redirect bug; Priya owns it."),
    CorpusItem("core-order", "Order #A-9817 from the bookstore shipped "
                             "late; refund reference RF-2201."),
    # update pair: stale then current (the harness supersedes v1→v2)
    CorpusItem("core-meeting-v1", "The weekly design sync is Tuesdays "
                                  "at 3pm in the studio room."),
    CorpusItem("core-meeting-v2", "The weekly design sync moved to "
                                  "Thursdays at 10am in the loft room.",
               supersedes="core-meeting-v1"),
    # unicode
    CorpusItem("core-cafe", "My favourite café is Café Mosaïque on "
                            "Rue de la Paix — order the crème brûlée."),
    CorpusItem("core-jp", "東京のオフィスは月曜日に休みです — the Tokyo "
                          "office is closed Mondays."),
    # temporal
    CorpusItem("core-move", "In March 2026 I moved house to 14 Willow "
                            "Lane; the lease runs twelve months."),
    CorpusItem("core-bday", "Maya's birthday is on the 14th of "
                            "February."),
    # forget distractor — seeded then deleted before queries
    CorpusItem("core-forgotten", "The garage door code is 4471 — "
                                 "replace this note after moving.",
               forget=True),
    # benign filler that should not crowd answers
    CorpusItem("core-book", "The book club is reading a history of "
                            "Byzantine trade routes this month."),
    CorpusItem("core-plant", "Water the ferns every Sunday; the "
                             "succulents every other week."),
)

_CORE_TASKS: Tuple[ConsumerTask, ...] = (
    ConsumerTask("t-id-ticket", "what is the status of ticket ENG-4821",
                 category="identifier", expected_ids=("core-ticket",)),
    ConsumerTask("t-id-order", "where is refund RF-2201",
                 category="identifier", expected_ids=("core-order",)),
    ConsumerTask("t-lex-wifi", "wifi password",
                 category="lexical", expected_ids=("core-wifi",)),
    ConsumerTask("t-lex-vet", "vet phone number for the cat",
                 category="lexical", expected_ids=("core-vet",)),
    ConsumerTask("t-lex-ins", "car insurance policy number",
                 category="lexical", expected_ids=("core-car",)),
    ConsumerTask("t-para-pet",
                 "when should I feed the feline",
                 category="paraphrase",
                 expected_ids=(),
                 notes="unsupported paraphrase of unstored cat-feeding "
                       "content — a miss is honest; scoring counts the "
                       "delivered set"),
    ConsumerTask("t-noans-1", "what is the office alarm code",
                 category="no_answer", expected_abstain=True),
    ConsumerTask("t-noans-2", "what time does the ferry leave on Sundays",
                 category="no_answer", expected_abstain=True),
    ConsumerTask("t-update-sync", "when is the weekly design sync",
                 category="update",
                 expected_ids=("core-meeting-v2",),
                 forbidden_ids=("core-meeting-v1",)),
    ConsumerTask("t-unicode-cafe", "café on Rue de la Paix",
                 category="unicode", expected_ids=("core-cafe",)),
    ConsumerTask("t-unicode-jp", "Tokyo office closed day",
                 category="unicode", expected_ids=("core-jp",)),
    ConsumerTask("t-temporal-move", "when did I move house",
                 category="temporal", expected_ids=("core-move",)),
    ConsumerTask("t-temporal-bday", "when is Maya's birthday",
                 category="temporal", expected_ids=("core-bday",)),
    ConsumerTask("t-forget-garage", "garage door code",
                 category="forget_distractor",
                 expected_ids=(),
                 forbidden_ids=("core-forgotten",),
                 notes="deleted distractor — delivery is a closure "
                       "violation, not a precision miss"),
    ConsumerTask("t-lex-book", "what is the book club reading",
                 category="lexical", expected_ids=("core-book",)),
)

#: Filler templates for scale — each yields an addressable unique token
#: ``tok-NNNN`` so generated probes have exact-match gold.
_FILLER_TOPICS: Tuple[Tuple[str, str], ...] = (
    ("receipt", "The receipt for purchase {tok} is filed under home "
                "repairs; it cost {price} euros."),
    ("contact", "Contact card {tok}: reach Sam at extension {ext} for "
                "the facilities team."),
    ("note", "Note {tok}: remember the {thing} needs attention before "
             "the {month} deadline."),
    ("event", "Calendar {tok}: {thing} review happens on the first "
              "Friday of {month}."),
    ("recipe", "Recipe card {tok}: the {thing} soup needs two onions "
               "and fresh dill."),
    ("log", "Log entry {tok}: runbook step {ext} completed with no "
            "errors on the staging host."),
)
_FILLER_THINGS: Tuple[str, ...] = (
    "garden", "tax", "insurance", "warranty", "prescription",
    "subscription", "warranty", "lease", "membership", "invoice",
)
_FILLER_MONTHS: Tuple[str, ...] = (
    "January", "March", "April", "June", "September", "November",
)


def _filler_item(rng: random.Random, i: int) -> CorpusItem:
    kind, tmpl = _FILLER_TOPICS[i % len(_FILLER_TOPICS)]
    tok = f"tok-{10000 + i}"
    text = tmpl.format(
        tok=tok,
        price=rng.randrange(4, 400),
        ext=rng.randrange(200, 999),
        thing=_FILLER_THINGS[i % len(_FILLER_THINGS)],
        month=_FILLER_MONTHS[i % len(_FILLER_MONTHS)],
    )
    return CorpusItem(f"fill-{i:05d}", text, infer=False,
                      tags=("filler", kind))


def seed_corpus(memories: int = 64, seed: int = 42,
                probes: Optional[int] = None) -> ConsumerCorpus:
    """Build the deterministic consumer corpus.

    ``memories`` counts retained items; the fixed core (~15 items) is
    always included and fillers pad to the requested scale. ``probes``
    generated tasks (default ``memories // 4``, capped by filler count)
    query unique filler tokens — exact-identifier and lexical lookups
    with per-item gold, giving the latency and quality suites a
    workload that scales with the corpus instead of reusing the same
    fifteen queries.

    Nothing here touches a store — the corpus is pure data, fully
    determined by ``(memories, seed, probes)``.
    """
    if memories < len(_CORE_ITEMS):
        raise ValueError(
            f"memories={memories} below core corpus size "
            f"{len(_CORE_ITEMS)}"
        )
    rng = random.Random(seed)
    items = list(_CORE_ITEMS)
    n_fill = memories - len(items)
    fillers = [_filler_item(rng, i) for i in range(n_fill)]
    items.extend(fillers)

    tasks = list(_CORE_TASKS)
    n_probes = probes if probes is not None else max(4, memories // 4)
    n_probes = min(n_probes, n_fill)
    probe_idx = rng.sample(range(n_fill), n_probes) if n_fill else []
    for j, fi in enumerate(probe_idx):
        item = fillers[fi]
        tok = f"tok-{10000 + fi}"
        if j % 2:
            q = f"what does note {tok} say"
            cat = "identifier"
        else:
            q = tok
            cat = "identifier"
        tasks.append(ConsumerTask(
            f"t-probe-{j:04d}", q, category=cat,
            expected_ids=(item.id,),
        ))
    return ConsumerCorpus(
        name=f"v5-consumer-{memories}s{seed}",
        seed=seed,
        items=tuple(items),
        tasks=tuple(tasks),
    )


def corpus_stats(corpus: ConsumerCorpus) -> dict[str, Any]:
    """Published corpus shape — denominators a reader can audit."""
    return {
        "name": corpus.name,
        "seed": corpus.seed,
        "items": len(corpus.items),
        "tasks": len(corpus.tasks),
        "digest": corpus.digest(),
        "categories": {
            c: sum(1 for t in corpus.tasks if t.category == c)
            for c in CATEGORIES
        },
        "forget_items": sum(1 for i in corpus.items if i.forget),
        "superseded_items": sum(
            1 for i in corpus.items if i.supersedes
        ),
    }


# ---------------------------------------------------------------------------
# public views — the arm boundary (V5-22.06 gold stays scoring-side)
# ---------------------------------------------------------------------------

_ITEM_PUBLIC = frozenset({"id", "text", "infer", "tags"})
_ITEM_GOLD = frozenset({"supersedes", "forget"})
_TASK_PUBLIC = frozenset({"task_id", "query", "category", "notes"})
_TASK_GOLD = frozenset({
    "expected_ids", "expected_abstain", "forbidden_ids",
})


class _GoldView:
    """Whitelisted-attribute proxy — the eval.v3 tripwire idiom.

    Public names forward; gold names raise an informative
    ``AttributeError`` so a gold read in arm code is loud in review and
    fails in CI. ``_wrapped`` stays reachable — the view polices the
    interface, not the memory model.
    """

    _public: frozenset = frozenset()
    _gold: frozenset = frozenset()
    __slots__ = ("_wrapped",)

    def __init__(self, wrapped: Any) -> None:
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        cls = type(self)
        if name in cls._public:
            return getattr(self._wrapped, name)
        if name in cls._gold:
            raise AttributeError(
                f"{cls.__name__}.{name} is evaluation gold — arms "
                "receive only the public task surface (V5-22.06)"
            )
        raise AttributeError(
            f"{cls.__name__!r} object has no attribute {name!r} "
            f"(wraps {type(self._wrapped).__name__})"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._wrapped!r})"


class PublicItemView(_GoldView):
    """Arm-facing item: ``id``/``text``/``infer``/``tags`` only."""

    _public = _ITEM_PUBLIC
    _gold = _ITEM_GOLD


class PublicTaskView(_GoldView):
    """Arm-facing task: ``task_id``/``query``/``category``/``notes``."""

    _public = _TASK_PUBLIC
    _gold = _TASK_GOLD


def public_item(item: CorpusItem) -> PublicItemView:
    return PublicItemView(item)


def public_task(task: ConsumerTask) -> PublicTaskView:
    return PublicTaskView(task)


# ---------------------------------------------------------------------------
# V6 consolidation fixtures (SPEC_V6 §03.2, V6-03.11/03.12) — additive.
#
# Whitelist-predicate claims in one slot from distinct evidence families:
# the two same-value phrasings "My favourite café is Café Lumière." /
# "I prefer Café Lumière." both produce structured ``preference`` claims
# (``claims.py`` extraction) that the slot-aggregate producer folds into
# one observation. The Café Noir pair exercises the rival-value path —
# same subject+predicate slot, different value, so each observation also
# carries ``contradicts`` pins. ``cons-corr-1`` is the corroboration
# probe's consumer-route attestation; the suite adds a second, distinct-
# attribution attestation of the same bytes through the envelope channel
# so the folded ``hit.corroboration ≥ 2`` group exists to measure.
#
# These items/tasks are additive: no core task references them, and
# ``seed_corpus`` never includes them unless the consolidation suite
# appends them explicitly — the generated-corpus digest is unchanged.
# ---------------------------------------------------------------------------

CONSOLIDATION_ITEMS: Tuple[CorpusItem, ...] = (
    CorpusItem("cons-cafe-1", "My favourite café is Café Lumière.",
               tags=("consolidation", "slot:preference")),
    CorpusItem("cons-cafe-2", "I prefer Café Lumière.",
               tags=("consolidation", "slot:preference")),
    CorpusItem("cons-cafe-3", "My favourite café is Café Noir.",
               tags=("consolidation", "slot:preference")),
    CorpusItem("cons-cafe-4",
               "I prefer Café Noir for afternoon espresso.",
               tags=("consolidation", "slot:preference")),
    CorpusItem("cons-corr-1",
               "Reminder: the basement dehumidifier code is DH-5520.",
               infer=False,
               tags=("consolidation", "corroborated")),
)

CONSOLIDATION_TASKS: Tuple[ConsumerTask, ...] = (
    ConsumerTask("t-cons-cafe", "café preference favourite",
                 category="lexical",
                 expected_ids=("cons-cafe-1", "cons-cafe-2"),
                 notes="consolidation fixture — the two same-slot "
                       "preference claims should deliver grounded hits"),
    ConsumerTask("t-cons-corr", "dehumidifier code DH-5520",
                 category="lexical",
                 expected_ids=("cons-corr-1",),
                 notes="folded-corroboration probe — the byte-identical "
                       "second attestation rides a duplicate group"),
)


__all__ = [
    "CATEGORIES",
    "CONSOLIDATION_ITEMS",
    "CONSOLIDATION_TASKS",
    "ConsumerCorpus",
    "ConsumerTask",
    "CorpusItem",
    "PublicItemView",
    "PublicTaskView",
    "corpus_stats",
    "public_item",
    "public_task",
    "seed_corpus",
]
