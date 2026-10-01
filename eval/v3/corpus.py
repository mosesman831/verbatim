"""Coding-task memory corpus for the v3 evaluation harness (SPEC_V3 §53.08).

The v3.0 evaluation slice is *coding-task* memory: captured facts, runbooks,
and decisions a coding agent would later need. This module defines the
record schema, loads JSONL manifests, and ships a deterministic seed corpus
(``corpus_seed.jsonl``) — ~40 hand-written, synthetic items across the
workload classes the four suites measure:

* ``factual_lookup`` — a fact captured earlier answers the query.
* ``procedure_reuse`` — a runbook/command should be surfaced for reuse.
* ``history`` — a past decision/incident is the expected evidence.
* ``exploratory`` — several sources jointly satisfy the query.
* ``abstention`` — no stored evidence supports an answer; the honest
  response is a typed abstention, never fabricated prose (§29, G4).
* ``benign_instructional`` — imperative runbook content ("run pnpm
  test") that must NOT be flagged by screening — the review-mandated
  benign-instructional class (§14.01, B18/B20, G9 retained coverage).
* ``poisoning`` — labeled attack fixtures (boundary redirection,
  authority claims, persistence, exfiltration, sleeper, compositional,
  tool invocation, credential harvesting) that must be screened or
  quarantined and never returned as support (§34, B18–B22, G9).
* ``scope_isolation`` — evidence in a scope the caller cannot read;
  any returned item from it is a zero-tolerance disclosure (G1).

Every line declares its gold: ``expected_evidence_ids`` (source ids the
query should surface), ``expected_abstain``, ``poisoned_source_ids`` (the
setup subset that is adversarial), and ``unauthorized_source_ids`` (the
subset outside the caller's read scope). ``task_runnable`` marks fixtures
the paired-execution suite replays (§53.14): scripted coding tasks whose
correct action materially depends on the memory item.

Corpus records deliberately reference *corpus* source ids (``src-*``);
the harness maps them to engine-assigned ``source_id`` values through the
``external_id`` envelope field — gold scoring is by id, never by fuzzy
text match (mirrors the v2 harness convention).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# record schema
# ---------------------------------------------------------------------------

#: Corpus source scope tags. ``owner`` is the calling principal's scope;
#: ``other`` is a different principal/conversation the caller cannot read —
#: the scope-isolation lane (G1 zero-disclosure).
SCOPE_OWNER = "owner"
SCOPE_OTHER = "other"

#: Workload classes recognized by the suite routers.
KINDS: Tuple[str, ...] = (
    "factual_lookup",
    "procedure_reuse",
    "history",
    "exploratory",
    "abstention",
    "benign_instructional",
    "poisoning",
    "scope_isolation",
)

#: Poisoning pattern labels (§34 lifecycle taxonomy; names are corpus
#: vocabulary, not screening rule ids).
POISON_PATTERNS: Tuple[str, ...] = (
    "boundary_redirection",
    "authority_claim",
    "persistence",
    "exfiltration",
    "sleeper",
    "compositional",
    "tool_invocation",
    "credential_harvest",
)

#: Difficulty bands for negative-transfer reporting (§53.09).
DIFFICULTY_BANDS: Tuple[str, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True)
class CorpusSource:
    """One ingestable setup item.

    ``scope`` selects the partition the harness ingests it under
    (``owner`` | ``other``). ``poison_pattern`` labels an adversarial
    fixture — poisoning-lane cases put its id in ``poisoned_source_ids``
    too; the pattern name documents intent for reviewers, it is not fed
    to the system under test. ``content_form`` records the expected
    §14.01 form for benign-instructional checks.
    """

    id: str
    text: str
    scope: str = SCOPE_OWNER
    kind: str = "note"  # note|runbook|decision|incident|doc|log
    poison_pattern: Optional[str] = None
    content_form: Optional[str] = None  # e.g. "instructional"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusSource":
        if not isinstance(d.get("id"), str) or not d["id"]:
            raise ValueError("corpus source requires non-empty id")
        if not isinstance(d.get("text"), str) or not d["text"]:
            raise ValueError(f"corpus source {d.get('id')!r} requires text")
        scope = d.get("scope", SCOPE_OWNER)
        if scope not in (SCOPE_OWNER, SCOPE_OTHER):
            raise ValueError(f"corpus source {d['id']!r}: bad scope {scope!r}")
        return cls(
            id=d["id"],
            text=d["text"],
            scope=scope,
            kind=d.get("kind", "note"),
            poison_pattern=d.get("poison_pattern"),
            content_form=d.get("content_form"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "scope": self.scope,
            "kind": self.kind,
        }
        if self.poison_pattern:
            out["poison_pattern"] = self.poison_pattern
        if self.content_form:
            out["content_form"] = self.content_form
        return out


@dataclass(frozen=True)
class TaskFixture:
    """Scripted paired-execution fixture (§53.14).

    The scripted agent must pick exactly one ``choice``; the correct one
    is only reliably knowable from the named memory evidence. ``choices``
    is an ordered list of ``{id, command}`` candidates — the first is the
    naive default (what an agent without memory picks, a documented
    scripted policy, not an intelligent guess). ``memory_query`` is the
    retrieval probe the memory arm issues.
    """

    goal: str
    choices: Tuple[dict[str, str], ...]
    correct_choice: str
    memory_query: str
    naive_choice: Optional[str] = None  # default: first choice

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskFixture":
        choices = tuple(dict(c) for c in d.get("choices") or ())
        if len(choices) < 2:
            raise ValueError("task fixture needs >= 2 choices")
        ids = [c.get("id") for c in choices]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("task fixture choices need unique ids")
        if d.get("correct_choice") not in ids:
            raise ValueError("correct_choice must be a choice id")
        naive = d.get("naive_choice") or ids[0]
        if naive not in ids:
            raise ValueError("naive_choice must be a choice id")
        return cls(
            goal=d.get("goal", ""),
            choices=choices,  # type: ignore[arg-type]
            correct_choice=d["correct_choice"],
            memory_query=d.get("memory_query", ""),
            naive_choice=naive,
        )

    def command_for(self, choice_id: str) -> str:
        for c in self.choices:
            if c["id"] == choice_id:
                return c.get("command", "")
        return ""


@dataclass(frozen=True)
class CorpusTask:
    """One evaluation case: setup evidence + query + gold."""

    task_id: str
    kind: str
    setup_sources: Tuple[CorpusSource, ...]
    query: str
    expected_evidence_ids: Tuple[str, ...] = ()
    expected_abstain: bool = False
    poisoned_source_ids: Tuple[str, ...] = ()
    unauthorized_source_ids: Tuple[str, ...] = ()
    task_runnable: bool = False
    task: Optional[TaskFixture] = None
    difficulty: str = "easy"
    tags: Tuple[str, ...] = ()
    supersession: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.task_id}: unknown kind {self.kind!r}")
        if self.difficulty not in DIFFICULTY_BANDS:
            raise ValueError(
                f"{self.task_id}: unknown difficulty {self.difficulty!r}"
            )
        ids = {s.id for s in self.setup_sources}
        for ref in (
            *self.expected_evidence_ids,
            *self.poisoned_source_ids,
            *self.unauthorized_source_ids,
        ):
            if ref not in ids:
                raise ValueError(
                    f"{self.task_id}: gold id {ref!r} not in setup_sources"
                )
        if self.task_runnable and self.task is None:
            raise ValueError(f"{self.task_id}: task_runnable requires task")
        if self.expected_abstain and self.expected_evidence_ids:
            raise ValueError(
                f"{self.task_id}: expected_abstain forbids expected_evidence_ids"
            )
        if self.supersession is not None:
            if self.supersession.get("expect") not in ("applied", "none"):
                raise ValueError(
                    f"{self.task_id}: supersession.expect must be "
                    "'applied' or 'none'"
                )
            pred = self.supersession.get("predecessor")
            if self.supersession["expect"] == "applied" and (
                not isinstance(pred, str) or pred not in ids
            ):
                raise ValueError(
                    f"{self.task_id}: supersession 'applied' requires a "
                    "predecessor id present in setup_sources"
                )

    @property
    def poisoned(self) -> Tuple[CorpusSource, ...]:
        want = set(self.poisoned_source_ids)
        return tuple(s for s in self.setup_sources if s.id in want)

    @property
    def unauthorized(self) -> Tuple[CorpusSource, ...]:
        want = set(self.unauthorized_source_ids)
        return tuple(s for s in self.setup_sources if s.id in want)

    @property
    def benign(self) -> Tuple[CorpusSource, ...]:
        """Setup items that are neither poisoned nor unauthorized — the
        retained-coverage population (G9 benign twins)."""
        bad = set(self.poisoned_source_ids) | set(self.unauthorized_source_ids)
        return tuple(s for s in self.setup_sources if s.id not in bad)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusTask":
        sources = tuple(
            CorpusSource.from_dict(s) for s in d.get("setup_sources") or ()
        )
        if not isinstance(d.get("task_id"), str) or not d["task_id"]:
            raise ValueError("corpus task requires task_id")
        if not isinstance(d.get("query"), str) or not d["query"]:
            raise ValueError(f"corpus task {d.get('task_id')!r} requires query")
        fixture = d.get("task")
        return cls(
            task_id=d["task_id"],
            kind=d.get("kind", ""),
            setup_sources=sources,
            query=d["query"],
            expected_evidence_ids=tuple(d.get("expected_evidence_ids") or ()),
            expected_abstain=bool(d.get("expected_abstain", False)),
            poisoned_source_ids=tuple(d.get("poisoned_source_ids") or ()),
            unauthorized_source_ids=tuple(
                d.get("unauthorized_source_ids") or ()
            ),
            task_runnable=bool(d.get("task_runnable", False)),
            task=TaskFixture.from_dict(fixture) if fixture else None,
            difficulty=d.get("difficulty", "easy"),
            tags=tuple(d.get("tags") or ()),
            supersession=d.get("supersession"),
        )


@dataclass(frozen=True)
class Corpus:
    """A loaded task corpus — order preserved, digest-stable."""

    name: str
    tasks: Tuple[CorpusTask, ...]
    source_path: str = ""

    def __iter__(self) -> Iterator[CorpusTask]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def by_id(self) -> dict[str, CorpusTask]:
        return {t.task_id: t for t in self.tasks}

    def of_kind(self, *kinds: str) -> Tuple[CorpusTask, ...]:
        want = set(kinds)
        return tuple(t for t in self.tasks if t.kind in want)

    @property
    def runnable(self) -> Tuple[CorpusTask, ...]:
        return tuple(t for t in self.tasks if t.task_runnable)

    def digest(self) -> str:
        """Reproducibility fingerprint over task content (§53.01 manifest
        identity — the digest is recorded in every report)."""
        canon = {
            "name": self.name,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "kind": t.kind,
                    "query": t.query,
                    "setup": [s.to_dict() for s in t.setup_sources],
                    "expect": list(t.expected_evidence_ids),
                    "abstain": t.expected_abstain,
                    "poisoned": list(t.poisoned_source_ids),
                    "unauthorized": list(t.unauthorized_source_ids),
                    "task_runnable": t.task_runnable,
                    "task": (
                        {
                            "goal": t.task.goal,
                            "choices": t.task.choices,
                            "correct": t.task.correct_choice,
                            "mq": t.task.memory_query,
                            "naive": t.task.naive_choice,
                        }
                        if t.task
                        else None
                    ),
                    "difficulty": t.difficulty,
                    "tags": list(t.tags),
                    # gold supersession expectation — a supersession-only
                    # edit on identical content must change the digest
                    # (staleness detection covers gold, not just setup)
                    "supersession": t.supersession,
                }
                for t in self.tasks
            ],
        }
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# public task view — gold-access enforcement at the arm boundary
# ---------------------------------------------------------------------------
#
# Baseline ``query(env, task, k)`` receives a *view* of the corpus task,
# never the gold-carrying record itself.  The view forwards only the
# non-gold surface — the query, the setup material the arm legitimately
# ingested, the public fixture fields, and descriptive metadata — while
# gold attributes (expected evidence, abstain/poison/scope gold,
# supersession expectations, the fixture's correct choice, and the
# derived poisoned/unauthorized/benign subsets) raise an informative
# ``AttributeError``.  Scoring code keeps the real ``CorpusTask``; the
# view exists only where an arm could condition its answer on the
# answer key.
#
# This is a tripwire, not a sandbox: deliberate introspection
# (``view._wrapped``) can still reach the record — the point is that a
# gold read in arm code is loud in review and fails in CI, never silent.


#: ``CorpusTask`` attributes forwarded to arms.
_TASK_PUBLIC: frozenset = frozenset({
    "task_id", "kind", "query", "setup_sources", "task",
    "task_runnable", "difficulty", "tags",
})

#: ``CorpusTask`` attributes that are evaluation gold — named so the
#: denial can say *what* was refused.
_TASK_GOLD: frozenset = frozenset({
    "expected_evidence_ids", "expected_abstain", "poisoned_source_ids",
    "unauthorized_source_ids", "supersession",
    "poisoned", "unauthorized", "benign",
})

#: ``TaskFixture`` attributes forwarded to arms — the scripted agent
#: legitimately sees the goal, the command menu, the memory probe, and
#: the documented naive default.  ``correct_choice`` is the answer key.
_FIXTURE_PUBLIC: frozenset = frozenset({
    "goal", "choices", "memory_query", "naive_choice",
})
_FIXTURE_GOLD: frozenset = frozenset({"correct_choice"})

#: ``CorpusSource`` attributes forwarded to arms — what ingest actually
#: feeds the system (fixture id, payload text, kind).  ``scope`` routes
#: the partition (its ``other`` value *is* the unauthorized gold),
#: ``poison_pattern`` labels the adversarial fixture, ``content_form``
#: records the expected screening form — all gold-side metadata.
_SOURCE_PUBLIC: frozenset = frozenset({"id", "text", "kind"})
_SOURCE_GOLD: frozenset = frozenset({
    "scope", "poison_pattern", "content_form",
})


class _GoldView:
    """Whitelisted-attribute proxy over a corpus record.

    ``_public`` names forward to the wrapped record; ``_gold`` names
    raise an ``AttributeError`` explaining the field is evaluation gold;
    anything else raises the usual missing-attribute error.  The wrapped
    record is reachable via ``view._wrapped`` — the view blocks the
    *interface* surface so gold reads are explicit, loud, and
    test-visible, not cryptographically impossible.
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
                f"{cls.__name__}.{name} is evaluation gold — baseline "
                "arms receive only the public task surface; gold fields "
                "are for scoring code, not query-time input"
            )
        raise AttributeError(
            f"{cls.__name__!r} object has no attribute {name!r} "
            f"(wraps {type(self._wrapped).__name__})"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._wrapped!r})"


class _PublicSourceView(_GoldView):
    """Arm-facing ``CorpusSource``: ``id``/``text``/``kind`` only."""

    _public = _SOURCE_PUBLIC
    _gold = _SOURCE_GOLD


class _PublicFixtureView(_GoldView):
    """Arm-facing ``TaskFixture``: the scripted environment minus the
    answer key (``correct_choice`` stays gold-side)."""

    _public = _FIXTURE_PUBLIC
    _gold = _FIXTURE_GOLD


class _PublicTaskView(_GoldView):
    """Arm-facing ``CorpusTask`` — pass to ``baseline.query`` so arms
    cannot read evaluation gold.  ``setup_sources`` and ``task`` are
    re-wrapped on each access."""

    _public = _TASK_PUBLIC
    _gold = _TASK_GOLD

    def __getattr__(self, name: str) -> Any:
        if name == "setup_sources":
            return tuple(
                _PublicSourceView(s) for s in self._wrapped.setup_sources
            )
        if name == "task":
            fixture = self._wrapped.task
            return (
                _PublicFixtureView(fixture) if fixture is not None else None
            )
        return super().__getattr__(name)


def public_task(task: CorpusTask) -> _PublicTaskView:
    """Wrap a ``CorpusTask`` for handoff to a baseline arm — the only
    object an arm's ``query`` may read task data from."""
    return _PublicTaskView(task)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_corpus(path: str, *, name: Optional[str] = None) -> Corpus:
    """Load a JSONL corpus manifest (one task object per line).

    Blank lines and ``#`` comments are ignored. Malformed lines fail
    loudly with line numbers — a corpus that cannot be scored exactly is
    worse than no corpus (§53.01 manifest integrity).
    """
    tasks: list[CorpusTask] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task = CorpusTask.from_dict(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            tasks.append(task)
    seen: set[str] = set()
    for t in tasks:
        if t.task_id in seen:
            raise ValueError(f"{path}: duplicate task_id {t.task_id!r}")
        seen.add(t.task_id)
    return Corpus(
        name=name or os.path.splitext(os.path.basename(path))[0],
        tasks=tuple(tasks),
        source_path=path,
    )


def seed_corpus_path() -> str:
    """Path of the bundled seed corpus."""
    return os.path.join(os.path.dirname(__file__), "corpus_seed.jsonl")


def load_seed_corpus() -> Corpus:
    """The packaged ~40-item seed corpus (``corpus_seed.jsonl``)."""
    return load_corpus(seed_corpus_path())


def corpus_stats(corpus: Corpus) -> dict[str, Any]:
    """Kind/difficulty/source counts for the report header."""
    kinds: dict[str, int] = {}
    diffs: dict[str, int] = {}
    patterns: dict[str, int] = {}
    n_sources = 0
    for t in corpus.tasks:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
        diffs[t.difficulty] = diffs.get(t.difficulty, 0) + 1
        n_sources += len(t.setup_sources)
        for s in t.setup_sources:
            if s.poison_pattern:
                patterns[s.poison_pattern] = patterns.get(s.poison_pattern, 0) + 1
    return {
        "name": corpus.name,
        "digest": corpus.digest(),
        "tasks": len(corpus.tasks),
        "setup_sources": n_sources,
        "kinds": kinds,
        "difficulty": diffs,
        "poison_patterns": patterns,
        "runnable": len(corpus.runnable),
        "abstention_expected": sum(1 for t in corpus.tasks if t.expected_abstain),
    }
