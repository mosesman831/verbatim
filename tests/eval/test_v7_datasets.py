"""Durable tests for the V7 dataset registry + corpus model.

Covers: registry completeness and field validation (V7-22.01); the
O1/O2/O3 authorization gates (§37 — a gated dataset is
``blocked_on_authorization`` until its env flag is set, ``missing_file``
when authorized but absent); the fixed hash dev/test partition
(V7-22.08); the LoCoMo ``locomo10.json`` adapter (items ↔ tasks, gold
turn refs, adversarial ``answerable=False``, evidence normalization of
the file's messy ref forms); lazy owned-twin loading (ImportError →
``unavailable``, never a crash); corpus integrity validation and digest
determinism; and the scorer-side gold boundary (``public_task``).

The LoCoMo file is local-only (CC BY-NC 4.0): tests that read it skip
when it is absent, and nothing here ever touches the network.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import types

import pytest

from eval.v7 import corpora, dataset_registry as reg


LOCOMO_PATH = "/tmp/locomo/locomo10.json"
LOCOMO_PRESENT = os.path.isfile(LOCOMO_PATH)

REQUIRED_IDS = {
    "owned_locomo_like",
    "owned_lme_like",
    "owned_actions",
    "owned_prefs",
    "owned_scale",
    "locomo",
    "longmemeval_s",
    "beam",
    "dolphinbench",
}


# ---------------------------------------------------------------------------
# registry shape + statuses
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_required_entries_present(self):
        ids = set(reg.dataset_ids())
        assert REQUIRED_IDS <= ids

    def test_entries_validate(self):
        for e in reg.all_entries():
            assert e.dataset_id and e.registry_status in (
                "permitted",
                "permitted_local_only",
                "blocked",
                "unverified",
            )
            if e.gate:
                assert e.env_flag, f"{e.dataset_id}: gate without env_flag"

    def test_owned_twins_ungated(self):
        for did in (
            "owned_locomo_like",
            "owned_lme_like",
            "owned_actions",
            "owned_prefs",
            "owned_scale",
        ):
            e = reg.get(did)
            assert e.gate is None
            assert e.loader_kind == "twin"
            st = reg.status(e, env={})
            assert st == reg.STATUS_AVAILABLE or reg.is_unavailable(st), (
                f"{did}: unexpected status {st!r}"
            )

    def test_locomo_entry_fields(self):
        e = reg.get("locomo")
        assert e.gate == "O1"
        assert e.env_flag == "VERBATIM_EVAL_LOCOMO"
        assert "CC BY-NC" in e.data_license
        assert e.registry_status == "permitted_local_only"
        assert e.sha256  # pinned
        assert e.splits and e.splits.unit == "conversation"

    def test_beam_dolphinbench_blocked(self):
        for did in ("beam", "dolphinbench"):
            assert reg.get(did).gate == "O3"
            assert reg.status(did, env={}) == reg.STATUS_BLOCKED

    def test_longmemeval_gated_o2(self):
        e = reg.get("longmemeval_s")
        assert e.gate == "O2"
        assert reg.status(e, env={}) == reg.STATUS_BLOCKED

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            reg.get("no_such_dataset")

    def test_status_detail_shape(self):
        row = reg.status_detail("locomo", env={})
        assert row["status"] == reg.STATUS_BLOCKED
        assert row["gate"] == "O1"
        assert row["flag_set"] is False

    def test_render_datasets_json(self):
        doc = json.loads(reg.render_datasets_json())
        ids = {d["dataset_id"] for d in doc["datasets"]}
        assert REQUIRED_IDS <= ids
        for d in doc["datasets"]:
            assert set(
                [
                    "source_url",
                    "data_license",
                    "code_license",
                    "redistribution",
                    "local_storage_policy",
                    "status",
                ]
            ) <= set(d)


# ---------------------------------------------------------------------------
# O1 gating for LoCoMo
# ---------------------------------------------------------------------------


class TestLocomoGate:
    def test_absent_env_blocks(self):
        env = {k: v for k, v in os.environ.items() if k != "VERBATIM_EVAL_LOCOMO"}
        assert reg.status("locomo", env=env) == reg.STATUS_BLOCKED

    def test_zero_flag_still_blocked(self):
        assert (
            reg.status("locomo", env={"VERBATIM_EVAL_LOCOMO": "0"})
            == reg.STATUS_BLOCKED
        )

    def test_flag_set_missing_file(self):
        env = {
            "VERBATIM_EVAL_LOCOMO": "1",
            "VERBATIM_EVAL_LOCOMO_PATH": "/nonexistent/locomo10.json",
        }
        assert reg.status("locomo", env=env) == reg.STATUS_MISSING

    @pytest.mark.skipif(not LOCOMO_PRESENT, reason="locomo10.json absent")
    def test_flag_set_file_present_available(self):
        env = {"VERBATIM_EVAL_LOCOMO": "1"}
        assert reg.status("locomo", env=env) == reg.STATUS_AVAILABLE

    def test_load_blocked_raises_unavailable(self):
        env = {k: v for k, v in os.environ.items() if k != "VERBATIM_EVAL_LOCOMO"}
        with pytest.raises(corpora.DatasetUnavailable) as ei:
            corpora.load_corpus("locomo", env=env)
        assert ei.value.status == reg.STATUS_BLOCKED


# ---------------------------------------------------------------------------
# LoCoMo loader (needs the gated local file)
# ---------------------------------------------------------------------------

LOCOMO_ENV = {"VERBATIM_EVAL_LOCOMO": "1"}


@pytest.fixture(scope="module")
def corpus():
    if not LOCOMO_PRESENT:
        pytest.skip("locomo10.json absent")
    return corpora.load_corpus("locomo", env=LOCOMO_ENV)


@pytest.mark.skipif(not LOCOMO_PRESENT, reason="locomo10.json absent")
class TestLocomoLoad:

    def test_shape(self, corpus):
        assert len(corpus.items) > 5000  # ~5,882 turns
        assert len(corpus.tasks) > 1500  # 1,986 QA
        assert corpus.digest()

    def test_items_carry_dialogue_fields(self, corpus):
        it = corpus.items[0]
        assert it.speaker and it.session_id and it.when
        assert it.group_id and it.id.startswith(f"{it.group_id}:")
        assert ":D" in it.id  # qualified dia id
        assert it.render_text()

    def test_categories_mapped(self, corpus):
        cats = {t.category for t in corpus.tasks}
        assert cats == set(corpora.LOCOMO_CATEGORY_NAMES.values())
        for t in corpus.tasks:
            assert t.category in corpora.LOCOMO_CATEGORY_NAMES.values()

    def test_category_counts_match_spec(self, corpus):
        observed = corpus.metadata["category_counts_observed"]
        expected = corpus.metadata["category_counts_expected"]
        assert observed == expected  # V7-22.07 published counts

    def test_adversarial_unanswerable(self, corpus):
        adv = corpus.of_category("adversarial")
        assert adv, "expected adversarial tasks"
        assert all(not t.answerable for t in adv)
        others = [t for t in corpus.tasks if t.category != "adversarial"]
        assert all(t.answerable for t in others)

    def test_evidence_refs_resolve(self, corpus):
        item_ids = set(corpus.item_by_id())
        n_with_ev = 0
        for t in corpus.tasks:
            for ref in t.evidence_ids:
                assert ref in item_ids
            if t.evidence_ids:
                n_with_ev += 1
        assert n_with_ev > 1500

    def test_evidence_normalization(self, corpus):
        # 'D:11:26' must normalize to conv-43:D11:26 on the Tim question
        cand = [
            t
            for t in corpus.tasks
            if "D:11:26" in " ".join(t.metadata.get("raw_evidence", ()))
        ]
        assert cand, "expected the double-colon evidence fixture"
        assert any("conv-43:D11:26" in t.evidence_ids for t in cand)
        # degenerate 'D' token is recorded unresolved, not guessed
        bare = [
            t
            for t in corpus.tasks
            if "D" in t.metadata.get("unresolved_evidence", ())
        ]
        assert bare, "expected the bare-D unresolved fixture"

    def test_session_level_gold(self, corpus):
        sess_ids = {i.session_id for i in corpus.items}
        ok = 0
        for t in corpus.tasks:
            for s in t.evidence_session_ids:
                assert s in sess_ids
            if t.evidence_session_ids:
                ok += 1
        assert ok > 1500

    def test_split_partition_by_conversation(self, corpus):
        dev = corpora.load_corpus("locomo", split="dev", env=LOCOMO_ENV)
        test = corpora.load_corpus("locomo", split="test", env=LOCOMO_ENV)
        assert dev.split == "dev" and test.split == "test"
        dev_groups = set(dev.groups())
        test_groups = set(test.groups())
        assert dev_groups and test_groups
        assert not (dev_groups & test_groups)
        assert dev_groups | test_groups == set(corpus.groups())
        # every dev task/item group is a dev-split conversation
        for g in dev_groups:
            assert reg.split_for_group("locomo", g) == "dev"
        # items follow their conversation into the split
        assert {i.group_id for i in dev.items} <= dev_groups
        assert {t.group_id for t in test.tasks} <= test_groups
        # split corpora are still integrity-valid (post_init ran)
        assert len(dev.items) + len(test.items) == len(corpus.items)
        assert len(dev.tasks) + len(test.tasks) == len(corpus.tasks)


# ---------------------------------------------------------------------------
# split determinism (V7-22.08)
# ---------------------------------------------------------------------------


class TestSplits:
    def test_deterministic(self):
        groups = [f"g{i}" for i in range(50)]
        a = reg.split_groups("owned_scale", groups)
        b = reg.split_groups("owned_scale", groups)
        assert a == b
        assert set(a["dev"]) | set(a["test"]) == set(groups)
        assert not (set(a["dev"]) & set(a["test"]))

    def test_per_dataset_partition(self):
        # partition is per-dataset; a group may split differently across ids
        g = "shared-group"
        assert reg.split_for_group("locomo", g) in ("dev", "test")

    def test_digest_stable(self):
        d1 = reg.split_digest("locomo", ["conv-26", "conv-30"])
        d2 = reg.split_digest("locomo", ["conv-30", "conv-26"])
        assert d1 == d2  # order-insensitive

    def test_bad_split_rejected(self):
        entry = reg.get("owned_scale")
        _fake_twin_module(entry.loader, generate=lambda seed=None: {
            "items": [{"id": "i1", "text": "x", "group_id": "g"}],
            "tasks": [{"task_id": "t1", "query": "q", "group_id": "g"}],
        })
        try:
            with pytest.raises(ValueError):
                corpora.load_corpus("owned_scale", split="validation", env={})
        finally:
            del sys.modules[entry.loader]


# ---------------------------------------------------------------------------
# corpus model validation
# ---------------------------------------------------------------------------


class TestCorpusModel:
    def test_task_requires_id_and_query(self):
        with pytest.raises(ValueError):
            corpora.CorpusTask(task_id="", query="q")
        with pytest.raises(ValueError):
            corpora.CorpusTask(task_id="t1", query="  ")

    def test_item_requires_id(self):
        with pytest.raises(ValueError):
            corpora.CorpusItem(id="", text="x")

    def test_unresolved_evidence_fails_integrity(self):
        with pytest.raises(ValueError):
            corpora.Corpus(
                name="bad",
                dataset_id="bad",
                items=(corpora.CorpusItem(id="i1", text="x"),),
                tasks=(
                    corpora.CorpusTask(
                        task_id="t1", query="q", evidence_ids=("nope",)
                    ),
                ),
            )

    def test_duplicate_item_ids_fail(self):
        with pytest.raises(ValueError):
            corpora.Corpus(
                name="bad",
                dataset_id="bad",
                items=(
                    corpora.CorpusItem(id="i1", text="x"),
                    corpora.CorpusItem(id="i1", text="y"),
                ),
                tasks=(),
            )

    def test_digest_deterministic(self):
        items = (corpora.CorpusItem(id="i1", text="hello", speaker="A"),)
        tasks = (
            corpora.CorpusTask(
                task_id="t1", query="hi", evidence_ids=("i1",)
            ),
        )
        c1 = corpora.Corpus(name="n", dataset_id="d", items=items, tasks=tasks)
        c2 = corpora.Corpus(name="n", dataset_id="d", items=items, tasks=tasks)
        assert c1.digest() == c2.digest()
        c3 = corpora.Corpus(
            name="n",
            dataset_id="d",
            items=(corpora.CorpusItem(id="i1", text="HELLO"),),
            tasks=tasks,
        )
        assert c1.digest() != c3.digest()

    def test_render_text(self):
        it = corpora.CorpusItem(
            id="x",
            text="hi",
            speaker="A",
            when="1:56 pm on 8 May, 2023",
            image_caption="a photo",
        )
        assert it.render_text() == (
            "[1:56 pm on 8 May, 2023] A: hi [shares image: a photo]"
        )


# ---------------------------------------------------------------------------
# owned twins — lazy import discipline
# ---------------------------------------------------------------------------


def _fake_twin_module(name: str, generate=None):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    if generate is not None:
        mod.generate = generate
    sys.modules[name] = mod
    return mod


class TestTwinsLazy:
    def test_missing_module_unavailable_not_crash(self):
        # a twin module that has not landed reports unavailable, and the
        # load raises DatasetUnavailable rather than ImportError
        unlanded = [
            e.dataset_id
            for e in reg.all_entries()
            if e.loader_kind == "twin"
            and reg.is_unavailable(reg.status(e, env={}))
        ]
        if not unlanded:
            pytest.skip("all twins landed; unavailable path uncovered")
        for did in unlanded:
            assert "not landed" in reg.status(did, env={})
            with pytest.raises(corpora.DatasetUnavailable) as ei:
                corpora.load_corpus(did, env={})
            assert reg.is_unavailable(ei.value.status)

    def test_fake_module_routes_through_generate(self):
        entry = reg.get("owned_actions")
        calls = []

        def gen(seed=None):
            calls.append(seed)
            return {
                "items": [
                    {"id": "i1", "text": "do thing", "group_id": "g1"}
                ],
                "tasks": [
                    {
                        "task_id": "t1",
                        "query": "q",
                        "category": "action",
                        "evidence_ids": ["i1"],
                        "group_id": "g1",
                    }
                ],
            }

        _fake_twin_module(entry.loader, generate=gen)
        try:
            assert reg.status("owned_actions", env={}) == reg.STATUS_AVAILABLE
            c = corpora.load_corpus("owned_actions", env={}, seed=7)
            assert calls == [7]
            assert len(c.items) == 1 and len(c.tasks) == 1
            assert c.tasks[0].evidence_ids == ("i1",)
            assert c.dataset_id == "owned_actions"
        finally:
            del sys.modules[entry.loader]

    def test_module_without_entry_point_unavailable(self):
        entry = reg.get("owned_prefs")
        _fake_twin_module(entry.loader, generate=None)
        try:
            # module present but has no generate/ITEMS+TASKS
            assert reg.status("owned_prefs", env={}) == reg.STATUS_AVAILABLE
            with pytest.raises(corpora.DatasetUnavailable) as ei:
                corpora.load_corpus("owned_prefs", env={})
            assert "entry point" in ei.value.status
        finally:
            del sys.modules[entry.loader]

    def test_landed_twins_load_end_to_end(self):
        # whichever sibling generators have landed route through the
        # registry -> lazy import -> normalized Corpus path
        landed = [
            e.dataset_id
            for e in reg.all_entries()
            if e.loader_kind == "twin"
            and reg.status(e, env={}) == reg.STATUS_AVAILABLE
        ]
        if not landed:
            pytest.skip("no twin generators landed yet")
        for did in landed:
            c = corpora.load_corpus(did)
            assert c.dataset_id == did
            assert len(c.tasks) > 0
            item_ids = set(c.item_by_id())
            for t in c.tasks:
                assert t.group_id  # split grouping present
                for r in t.evidence_ids:
                    assert r in item_ids
            # dev/test partition is coherent
            dev = corpora.load_corpus(did, split="dev")
            test = corpora.load_corpus(did, split="test")
            assert len(dev.tasks) + len(test.tasks) == len(c.tasks)
            for t in (*dev.tasks, *test.tasks):
                want = reg.split_for_group(did, t.group_id)
                assert t.task_id in {x.task_id for x in (
                    dev.tasks if want == "dev" else test.tasks
                )}

    def test_twin_split_filtering(self):
        entry = reg.get("owned_lme_like")

        def gen(seed=None):
            items = [
                {"id": f"{g}-i", "text": "x", "group_id": g}
                for g in ("ga", "gb")
            ]
            tasks = [
                {
                    "task_id": f"{g}-t",
                    "query": "q",
                    "evidence_ids": [f"{g}-i"],
                    "group_id": g,
                }
                for g in ("ga", "gb")
            ]
            return {"items": items, "tasks": tasks}

        _fake_twin_module(entry.loader, generate=gen)
        try:
            full = corpora.load_corpus("owned_lme_like", env={})
            dev = corpora.load_corpus("owned_lme_like", split="dev", env={})
            want = {
                g
                for g in ("ga", "gb")
                if reg.split_for_group("owned_lme_like", g) == "dev"
            }
            assert {i.group_id for i in dev.items} == want
            assert {t.group_id for t in dev.tasks} == want
            assert dev.split == "dev"
            assert len(full.items) == 2
        finally:
            del sys.modules[entry.loader]


# ---------------------------------------------------------------------------
# gold boundary (V7-22.02)
# ---------------------------------------------------------------------------


class TestPublicView:
    def test_gold_fields_denied(self):
        t = corpora.CorpusTask(
            task_id="t1",
            query="q",
            category="lexical",
            evidence_ids=("i1",),
            answerable=False,
            metadata={"answer": "secret"},
        )
        v = corpora.public_task(t)
        assert v.task_id == "t1" and v.query == "q" and v.category == "lexical"
        for gold in (
            "evidence_ids",
            "evidence_session_ids",
            "answerable",
            "metadata",
        ):
            with pytest.raises(AttributeError):
                getattr(v, gold)

    def test_unknown_attr_raises(self):
        v = corpora.public_task(
            corpora.CorpusTask(task_id="t1", query="q")
        )
        with pytest.raises(AttributeError):
            v.nonsense
