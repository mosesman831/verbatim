"""Gold-access enforcement tests — arm isolation at the query boundary.

``run_retrieval_suite`` hands every arm ``public_task(task)`` — a
whitelisted view over the corpus task — so an arm that reads evaluation
gold (expected evidence ids, abstain/poison/scope gold, supersession,
the fixture's correct choice) fails loudly with an informative
``AttributeError`` instead of silently scoring on the answer key.
Scoring code keeps the real ``CorpusTask``.

These tests pin the view's surface and prove the suite actually delivers
it: a cheating arm that touches gold records an error naming the blocked
field; the real shipped arms pass clean through the same boundary.
"""

from __future__ import annotations

import json

import pytest

from eval.v3.baselines import (
    BASELINE_NAMES,
    QueryOutcome,
    get_baseline,
    prepare_case,
    probe_capabilities,
)
from eval.v3.corpus import Corpus, load_corpus, public_task
from eval.v3.suite_retrieval import run_retrieval_suite


# ---------------------------------------------------------------------------
# tiny corpus — one task per gold surface the view must hide
# ---------------------------------------------------------------------------

_TASKS = [
    {
        "task_id": "t-fact", "kind": "factual_lookup",
        "setup_sources": [
            {"id": "src-a",
             "text": "The billing service uses PostgreSQL 15.",
             "kind": "note"},
            {"id": "src-b",
             "text": "The analytics pipeline writes Parquet files.",
             "kind": "note"},
        ],
        "query": "billing service postgresql",
        "expected_evidence_ids": ["src-a"],
    },
    {
        "task_id": "t-scope", "kind": "scope_isolation",
        "setup_sources": [
            {"id": "src-mine",
             "text": "Our deploy checklist requires captain approval.",
             "kind": "runbook"},
            {"id": "src-theirs",
             "text": "Their checklist marks Friday as deploy day.",
             "kind": "runbook", "scope": "other",
             "poison_pattern": "authority_claim"},
        ],
        "query": "deploy checklist",
        "expected_evidence_ids": ["src-mine"],
        "unauthorized_source_ids": ["src-theirs"],
        "poisoned_source_ids": ["src-theirs"],
    },
    {
        "task_id": "t-task", "kind": "procedure_reuse",
        "task_runnable": True,
        "setup_sources": [
            {"id": "src-reset",
             "text": "To reset the dev database run `make db-reset`.",
             "kind": "runbook"},
        ],
        "query": "reset dev database",
        "expected_evidence_ids": ["src-reset"],
        "task": {
            "goal": "reset the dev database",
            "choices": [
                {"id": "db-init", "command": "make db-init"},
                {"id": "db-reset", "command": "make db-reset"},
            ],
            "correct_choice": "db-reset",
            "memory_query": "reset dev database",
            "naive_choice": "db-init",
        },
    },
]

_GOLD_TASK_FIELDS = (
    "expected_evidence_ids", "expected_abstain", "poisoned_source_ids",
    "unauthorized_source_ids", "supersession",
    "poisoned", "unauthorized", "benign",
)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Corpus:
    path = tmp_path_factory.mktemp("corpus") / "gold.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for t in _TASKS:
            fh.write(json.dumps(t) + "\n")
    return load_corpus(str(path), name="gold")


@pytest.fixture(scope="module")
def caps():
    return probe_capabilities()


# ---------------------------------------------------------------------------
# the view itself — public surface forwards, gold surface refuses
# ---------------------------------------------------------------------------


class TestPublicTaskView:
    def test_public_fields_forward(self, tiny_corpus):
        task = tiny_corpus.by_id()["t-fact"]
        view = public_task(task)
        assert view.task_id == "t-fact"
        assert view.kind == "factual_lookup"
        assert view.query == "billing service postgresql"
        assert view.difficulty == task.difficulty
        assert view.tags == task.tags
        assert view.task_runnable is False
        # setup sources are re-wrapped: payload fields only
        sources = view.setup_sources
        assert [s.id for s in sources] == ["src-a", "src-b"]
        assert sources[0].text.startswith("The billing service")
        assert sources[0].kind == "note"

    def test_gold_fields_raise_informative_error(self, tiny_corpus):
        view = public_task(tiny_corpus.by_id()["t-scope"])
        for field in _GOLD_TASK_FIELDS:
            with pytest.raises(AttributeError, match="evaluation gold"):
                getattr(view, field)
            # probing must not reveal the field either
            assert not hasattr(view, field), field

    def test_source_gold_fields_blocked(self, tiny_corpus):
        view = public_task(tiny_corpus.by_id()["t-scope"])
        src = view.setup_sources[1]  # the scoped+poisoned fixture
        assert src.id == "src-theirs" and src.kind == "runbook"
        for field in ("scope", "poison_pattern", "content_form"):
            with pytest.raises(AttributeError, match="evaluation gold"):
                getattr(src, field)

    def test_fixture_answer_key_blocked(self, tiny_corpus):
        view = public_task(tiny_corpus.by_id()["t-task"])
        fixture = view.task
        # the scripted environment is visible — goal, menu, probe, default
        assert fixture.goal == "reset the dev database"
        assert fixture.memory_query == "reset dev database"
        assert fixture.naive_choice == "db-init"
        assert fixture.choices[0]["id"] == "db-init"
        # the answer key is not
        with pytest.raises(AttributeError, match="evaluation gold"):
            fixture.correct_choice

    def test_task_none_fixture_forwards_none(self, tiny_corpus):
        view = public_task(tiny_corpus.by_id()["t-fact"])
        assert view.task is None

    def test_unknown_attribute_is_plain_missing(self, tiny_corpus):
        view = public_task(tiny_corpus.by_id()["t-fact"])
        with pytest.raises(AttributeError, match="no attribute"):
            view.not_a_real_field


# ---------------------------------------------------------------------------
# shipped arms — every baseline answers through the public view
# ---------------------------------------------------------------------------


class TestRealArmsThroughView:
    def test_every_baseline_reads_only_public_fields(
        self, tiny_corpus, caps
    ):
        """All five shipped arms query through the view without a gold
        error — the enforcement boundary does not break honest arms."""
        task = tiny_corpus.by_id()["t-fact"]
        env = prepare_case(task, capabilities=caps)
        try:
            for name in BASELINE_NAMES:
                out = get_baseline(name).query(env, public_task(task))
                assert out.arm == name
                assert out.error is None or "gold" not in out.error, (
                    f"{name} hit the gold boundary: {out.error}"
                )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# suite plumbing — the retrieval suite hands the view to arms
# ---------------------------------------------------------------------------


class _CheatingBaseline:
    """An arm that reads the answer key — the regression the view exists
    to catch.  Named like a real arm so the suite treats it as one."""

    name = "cheating_arm"
    description = "reads expected_evidence_ids and returns them"
    ingest_mode = "v2"

    def capabilities(self):
        return {}

    def query(self, env, task, *, k=5) -> QueryOutcome:
        gold = task.expected_evidence_ids  # <- must raise AttributeError
        return QueryOutcome(
            arm=self.name, task_id=task.task_id,
            returned_ids=tuple(gold), n_items=len(gold),
        )


class _FixtureCheatingBaseline(_CheatingBaseline):
    """Cheat via the nested fixture's correct_choice instead."""

    name = "fixture_cheating_arm"

    def query(self, env, task, *, k=5) -> QueryOutcome:
        task.task.correct_choice  # <- must raise AttributeError
        return QueryOutcome(arm=self.name, task_id=task.task_id)


class TestSuiteEnforcement:
    def test_cheating_arm_records_gold_error(
        self, tiny_corpus, caps, monkeypatch
    ):
        monkeypatch.setattr(
            "eval.v3.suite_retrieval.get_baseline",
            lambda name: _CheatingBaseline(),
        )
        run = run_retrieval_suite(
            tiny_corpus, "cheating_arm", capabilities=caps
        )
        assert run.errors == len(run.records) == 3
        for rec in run.records:
            assert rec.note.startswith("error:AttributeError"), rec.note
            assert "evaluation gold" in rec.note
            assert "expected_evidence_ids" in rec.note

    def test_fixture_answer_key_cheat_records_gold_error(
        self, tiny_corpus, caps, monkeypatch
    ):
        monkeypatch.setattr(
            "eval.v3.suite_retrieval.get_baseline",
            lambda name: _FixtureCheatingBaseline(),
        )
        run = run_retrieval_suite(
            tiny_corpus, "fixture_cheating_arm",
            capabilities=caps, task_ids=["t-task"],
        )
        assert run.errors == 1
        assert "correct_choice" in run.records[0].note
        assert "evaluation gold" in run.records[0].note

    def test_real_arm_passes_clean_through_suite(
        self, tiny_corpus, caps
    ):
        run = run_retrieval_suite(
            tiny_corpus, "no_memory", capabilities=caps
        )
        assert len(run.records) == 3
        assert run.errors == 0
