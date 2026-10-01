"""Durable tests for V85-04 — the offline AMB proxy and Track R's
geic@B parity path (SPEC_V8_5 §4).

Coverage:

* ``geic_expand`` — the V85-02.04 delivery rule: coverage-first ring
  order, ±W_r neighbors, dedup, once-per-session header, stray charge,
  exact budget boundary (``spent + cost == B`` delivers; ``> B`` stops
  the whole walk, partial hit contribution kept), unbounded.
* ``UnitSpans.positions`` — the hit→position ladder: turn ``seq`` pin,
  byte-range intersection, unresolved ``(None, None)``.
* ``_parse_hit_object_ref`` — unit/claim/garbage.
* proxy doc shape — AMB-shaped session ``Document``s (id/content/
  user_id/timestamp/context) + ``query_timestamp`` from the pinned
  loader's conventions; ``_gold_dias`` prefix strip; the AMB label
  crosswalk (open-domain=4 … adversarial=5).
* ``_delivered_dias`` — exact delivered-position → dia_id mapping.
* ``run_proxy`` — the real ``VerbatimAMBProvider`` driven end-to-end on
  a synthetic two-conversation fixture: ingest via the ``messages``
  path, measured ``retrieve`` latency, GEIC scored per budget, parity
  between the provider's ``_expand`` and the shared ``geic_expand``,
  deterministic records.
* Track R parity — ``VerbatimArm(ingest_mode="session_messages")``
  ingests one ``Memory.add(messages=[...])`` per session, mints real
  ``seq`` ordinals, ships ``diag["geic"]``; ``run_track_r`` aggregates
  it; the item-mode arm stays silent (no fabricated zeros).
* No network anywhere — every fixture is inline.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from eval.v7.arms import (
    DictCorpus,
    GEIC_LINE_MARGIN,
    GEIC_SESSION_MARGIN,
    GEIC_STRAY_MARGIN,
    GEIC_STRAY_PREFIX,
    QueryOutcome,
    UnitSpans,
    VerbatimArm,
    _parse_hit_object_ref,
    geic_expand,
    geic_meter,
)
from eval.v7.track_r import render_markdown, run_track_r
from eval.v8 import amb_proxy as AP


# ---------------------------------------------------------------------------
# geic_expand — the V85-02.04 delivery rule (provider-parity replay)
# ---------------------------------------------------------------------------


def _sess(n: int, cost: int = 10, header: int = 7) -> dict:
    return {"costs": [cost] * n, "header": header}


def _spent(n_positions: int, cost: int = 10, header: int = 7) -> int:
    return header + GEIC_SESSION_MARGIN + n_positions * (
        cost + GEIC_LINE_MARGIN)


class TestGeicExpand:
    def test_coverage_then_ring_neighbors(self):
        # covered={2}, W=1 → positions {1,2,3}; header paid once.
        delivered, spent = geic_expand(
            [("s", {2}, 0)], {"s": _sess(5)}, window=1, budget=None)
        assert delivered == {"s": (1, 2, 3)}
        assert spent == _spent(3)

    def test_window_zero_no_neighbors(self):
        delivered, spent = geic_expand(
            [("s", {2}, 0)], {"s": _sess(5)}, window=0, budget=None)
        assert delivered == {"s": (2,)}
        assert spent == _spent(1)

    def test_window_two(self):
        delivered, _ = geic_expand(
            [("s", {3}, 0)], {"s": _sess(6)}, window=2, budget=None)
        assert delivered == {"s": (1, 2, 3, 4, 5)}

    def test_edges_clamp(self):
        delivered, _ = geic_expand(
            [("s", {0}, 0)], {"s": _sess(3)}, window=1, budget=None)
        assert delivered == {"s": (0, 1)}

    def test_dedup_across_hits(self):
        # two hits covering overlapping positions — second pays only
        # for genuinely new positions; the session header once.
        delivered, spent = geic_expand(
            [("s", {1}, 0), ("s", {2}, 0)],
            {"s": _sess(5)}, window=1, budget=None)
        assert delivered == {"s": (0, 1, 2, 3)}
        assert spent == _spent(4)

    def test_second_session_header_charged(self):
        delivered, spent = geic_expand(
            [("a", {0}, 0), ("b", {0}, 0)],
            {"a": _sess(2), "b": _sess(2)}, window=0, budget=None)
        assert delivered == {"a": (0,), "b": (0,)}
        assert spent == 2 * _spent(1)

    def test_budget_boundary_exact(self):
        # equality delivers; one token under stops before the last line.
        exact = _spent(3)
        d1, s1 = geic_expand(
            [("s", {2}, 0)], {"s": _sess(5)}, window=1, budget=exact)
        assert s1 == exact and d1 == {"s": (1, 2, 3)}
        d2, s2 = geic_expand(
            [("s", {2}, 0)], {"s": _sess(5)}, window=1, budget=exact - 1)
        assert d2 == {"s": (1, 2)}          # coverage + first ring pos
        assert s2 == _spent(2)

    def test_coverage_visited_before_ring(self):
        # Budget fits covered + exactly one ring position → the nearer
        # position by (dist, ordinal) wins deterministically.
        budget = _spent(2)
        delivered, spent = geic_expand(
            [("s", {3}, 0)], {"s": _sess(6)}, window=1, budget=budget)
        assert delivered == {"s": (2, 3)}   # pos 3 first, then 2
        assert spent == budget

    def test_first_overflow_stops_later_hits(self):
        # hit1 partially delivers (overflow at 3rd position); hit2's
        # session is never opened — the provider's whole-walk stop.
        budget = _spent(2)
        delivered, spent = geic_expand(
            [("a", {2}, 0), ("b", {0}, 0)],
            {"a": _sess(5), "b": _sess(3)}, window=1, budget=budget)
        assert delivered == {"a": (1, 2)}
        assert spent == budget
        assert "b" not in delivered

    def test_stray_charged_no_positions(self):
        stray = 9
        delivered, spent = geic_expand(
            [(None, None, stray), ("s", {0}, 0)],
            {"s": _sess(3)}, window=0, budget=None)
        assert delivered == {"s": (0,)}
        assert spent == stray + _spent(1)

    def test_stray_overflow_stops_walk(self):
        delivered, spent = geic_expand(
            [("s", {0}, 0), (None, None, 100), ("s", {1}, 0)],
            {"s": _sess(3)}, window=0, budget=_spent(1) + 50)
        # stray's 100 pushes spent over → stops; hit3 never walks.
        assert delivered == {"s": (0,)}
        assert spent == _spent(1)

    def test_unknown_session_is_stray(self):
        delivered, spent = geic_expand(
            [("ghost", {0}, 42)], {"s": _sess(3)}, window=0, budget=None)
        assert delivered == {}
        assert spent == 42

    def test_covered_out_of_range_falls_back_to_zero(self):
        # covered positions outside [0, n) clamp to the {0} fallback —
        # the provider's "ordinal 0 when nothing resolves" floor.
        delivered, _ = geic_expand(
            [("s", {9}, 0)], {"s": _sess(3)}, window=0, budget=None)
        assert delivered == {"s": (0,)}

    def test_unbounded_ignores_overflow(self):
        delivered, spent = geic_expand(
            [("s", {2}, 0)], {"s": _sess(5)}, window=1, budget=None)
        assert delivered == {"s": (1, 2, 3)}
        assert spent == _spent(3)

    def test_empty_hits_zero_spend(self):
        delivered, spent = geic_expand(
            [], {"s": _sess(3)}, window=1, budget=100)
        assert delivered == {} and spent == 0

    def test_empty_coverage_floors_to_zero(self):
        # an empty ``covered`` set is the [0] fallback — a resolved hit
        # always contributes its anchor.
        delivered, _ = geic_expand(
            [("s", set(), 0)], {"s": _sess(3)}, window=0, budget=None)
        assert delivered == {"s": (0,)}


# ---------------------------------------------------------------------------
# UnitSpans.positions + _parse_hit_object_ref — the resolution ladder
# ---------------------------------------------------------------------------


class TestPositions:
    def _sess(self, n: int = 4):
        # member byte ranges: turn i spans [i*10, i*10+10)
        return {"n": n, "ranges": [
            (i * 10, i * 10 + 10) for i in range(n)]}

    def test_turn_seq_pin(self):
        row = {"kind": "turn", "seq": 2, "byte_start": 0, "byte_end": 5}
        assert UnitSpans.positions(self._sess(), row) == (2, 2)

    def test_byte_range_intersection(self):
        # episode/session unit spanning members 1..2.
        row = {"kind": "episode", "seq": None,
               "byte_start": 15, "byte_end": 25}
        assert UnitSpans.positions(self._sess(), row) == (1, 2)

    def test_seq_out_of_range_falls_to_bytes(self):
        row = {"kind": "turn", "seq": 9, "byte_start": 12, "byte_end": 18}
        assert UnitSpans.positions(self._sess(), row) == (1, 1)

    def test_unresolved(self):
        row = {"kind": "turn", "seq": None,
               "byte_start": None, "byte_end": None}
        assert UnitSpans.positions(self._sess(), row) == (None, None)

    def test_inverted_range_unresolved(self):
        row = {"kind": "turn", "seq": None, "byte_start": 20,
               "byte_end": 20}
        assert UnitSpans.positions(self._sess(), row) == (None, None)


class TestParseObjectRef:
    @staticmethod
    def _ref(kind: str, oid: str, rev) -> str:
        return f"vobj1.{kind}.{oid.encode('utf-8').hex()}.{rev}"

    def test_unit(self):
        assert _parse_hit_object_ref(
            self._ref("unit", "u-abc", 3)) == ("unit", "u-abc", 3)

    def test_claim(self):
        assert _parse_hit_object_ref(
            self._ref("claim", "c-9", 1)) == ("claim", "c-9", 1)

    def test_rev_dash(self):
        assert _parse_hit_object_ref(
            self._ref("unit", "u-1", "-")) == ("unit", "u-1", None)

    def test_three_part_form(self):
        ref = f"vobj1.unit.{'u-7'.encode().hex()}"
        assert _parse_hit_object_ref(ref) == ("unit", "u-7", None)

    def test_garbage(self):
        assert _parse_hit_object_ref("nope") is None
        assert _parse_hit_object_ref("") is None
        assert _parse_hit_object_ref(None) is None
        assert _parse_hit_object_ref("vobj1.unit.nothex.1") is None


# ---------------------------------------------------------------------------
# proxy doc shape + gold mapping + crosswalk (pure — no provider)
# ---------------------------------------------------------------------------


CONV = {
    "sample_id": "conv-t1",
    "conversation": {
        "speaker_a": "Ann",
        "speaker_b": "Bob",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_1": [
            {"speaker": "Ann", "dia_id": "D1:1", "text": "Hi Bob!"},
            {"speaker": "Bob", "dia_id": "D1:2",
             "text": "Hey Ann, how are you?"},
            {"speaker": "Ann", "dia_id": "D1:3",
             "text": "I went to the support group on Monday"},
        ],
        "session_2": [
            {"speaker": "Ann", "dia_id": "D2:1",
             "text": "I bought a kayak last weekend"},
            {"speaker": "Bob", "dia_id": "D2:2",
             "text": "Wow, where will you paddle?"},
        ],
    },
}


class _Task:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestDocShape:
    def test_session_documents(self):
        docs = AP.session_documents(CONV)
        assert [d["id"] for d in docs] == [
            "conv-t1_session_1", "conv-t1_session_2"]
        turns = json.loads(docs[0]["content"])
        assert [t["dia_id"] for t in turns] == ["D1:1", "D1:2", "D1:3"]
        assert turns[0]["speaker"] == "Ann"
        assert docs[0]["user_id"] == "conv-t1"
        assert docs[0]["timestamp"] == "2023-05-08T13:56:00+00:00"
        assert docs[1]["timestamp"] == "2023-05-09T14:00:00+00:00"
        assert docs[0]["context"] == (
            "Conversation between Ann and Bob (session_1 of conv-t1)")

    def test_query_timestamp_last_session(self):
        assert AP.question_timestamp(CONV) == "2023-05-09T14:00:00+00:00"

    def test_gold_dias_prefix_strip(self):
        t = _Task(evidence_ids=("conv-t1:D1:3", "conv-t1:D1:1"),
                  metadata={"raw_evidence": ["D1:3", "D1:1", "D?:x"]})
        dias, n_raw = AP._gold_dias(t, "conv-t1")
        assert dias == ["D1:1", "D1:3"]
        assert n_raw == 3

    def test_gold_dias_wrong_prefix_dropped(self):
        t = _Task(evidence_ids=("other:D1:3", "conv-t1:D1:1"),
                  metadata={})
        dias, _ = AP._gold_dias(t, "conv-t1")
        assert dias == ["D1:1"]

    def test_amb_label_crosswalk(self):
        assert AP.AMB_LABELS == {
            1: "single-hop", 2: "temporal", 3: "multi-hop",
            4: "open-domain", 5: "adversarial"}

    def test_parse_budgets(self):
        assert AP._parse_budgets("1000,4500,unbounded") == (
            1000, 4500, None)
        assert AP._parse_budgets("none") == (None,)
        with pytest.raises(ValueError):
            AP._parse_budgets(" , ")


class _FakeIndex:
    """``SessionIndex`` duck — only ``session()`` matters to
    ``_delivered_dias``."""

    def __init__(self, sessions):
        self._sessions = sessions

    def session(self, doc_id):
        return self._sessions.get(doc_id)


class TestDeliveredDias:
    def test_exact_mapping_sorted_dedup(self):
        idx = _FakeIndex({
            "s1": {"turns": [{"dia_id": "D1:1"}, {"dia_id": "D1:2"},
                             {"dia_id": "D1:3"}]},
            "s2": {"turns": [{"dia_id": "D2:1"}, {}]},
        })
        out = AP._delivered_dias(
            {"s1": {2, 0}, "s2": {0, 1, 7}, "ghost": {0}}, idx)
        # ordinals resolve to dia_ids exactly; missing dia / out-of-range
        # / unknown session contribute nothing fabricated.
        assert out == ["D1:1", "D1:3", "D2:1"]


# ---------------------------------------------------------------------------
# run_proxy — the real provider v2 end-to-end (no network)
# ---------------------------------------------------------------------------


def _make_corpus(convs_tasks):
    tasks = []
    for tid, query, cat, cat_id, gold, sid in convs_tasks:
        tasks.append(_Task(
            task_id=tid, query=query, category=cat, answerable=True,
            evidence_ids=tuple(f"{sid}:{d}" for d in gold),
            group_id=sid,
            metadata={"category_id": cat_id,
                      "raw_evidence": list(gold)}))
    return type("C", (), {"tasks": tasks, "items": []})


@pytest.fixture()
def provider(tmp_path):
    from eval.amb.provider import VerbatimAMBProvider

    p = VerbatimAMBProvider(
        store_dir=tmp_path / "store", token_budget=4500, neighbor_w=1)
    p.prepare(tmp_path / "store", unit_ids=["conv-t1"])
    yield p
    try:
        p.cleanup()
    except Exception:
        pass


class TestRunProxy:
    def test_end_to_end_scored(self, provider):
        corpus = _make_corpus([
            ("conv-t1/q0", "when did Ann go to the support group",
             "temporal", 2, ["D1:3"], "conv-t1"),
            ("conv-t1/q1", "where will Ann paddle her kayak",
             "single-hop", 1, ["D2:1"], "conv-t1"),
        ])
        res = AP.run_proxy(corpus, [CONV], provider,
                           budgets=(1000, 4500, None), k=10)
        assert res["ingest"]["ingest_path"].startswith("messages")
        assert len(res["records"]) == 2
        for r in res["records"]:
            assert r["retrieve_ms"] is not None
            assert r["n_hits"] > 0 and r["n_resolved"] > 0
            ub = r["budgets"]["unbounded"]
            assert ub["geic_any"] is True
            assert set(ub["delivered_dias"]) >= set(r["gold_dias"])
            assert ub["tokens"] > 0 and ub["n_turns"] > 0
        # provider._expand == geic_expand on this store — parity held
        # for every checked question.
        par = res["expansion"]["parity"]
        assert par["checked"] == 2 and par["matched"] == 2
        agg = res["aggregate"]
        assert agg["overall"]["unbounded"]["geic_any"] == 1.0
        assert "temporal" in agg["by_amb_label"]
        assert "single-hop" in agg["by_amb_label"]
        assert agg["retrieve_ms"]["n"] == 2

    def test_budget_boundary_scored(self, provider):
        corpus = _make_corpus([
            ("conv-t1/q0", "when did Ann go to the support group",
             "temporal", 2, ["D1:3"], "conv-t1"),
        ])
        res = AP.run_proxy(corpus, [CONV], provider,
                           budgets=(5, 60, None), k=10)
        b = res["records"][0]["budgets"]
        assert b["5"]["geic_any"] is False       # header+line > 5
        assert b["5"]["n_turns"] == 0
        assert b["60"]["geic_any"] is True       # session_1 fits
        assert b["unbounded"]["geic_any"] is True

    def test_deterministic_records(self, tmp_path):
        # Two fresh providers on identical inputs produce identical
        # scored content (latency fields excluded by construction —
        # they're measured, not replayed).
        outs = []
        for i in range(2):
            from eval.amb.provider import VerbatimAMBProvider

            wd = tmp_path / f"run{i}"
            p = VerbatimAMBProvider(
                store_dir=wd, token_budget=4500, neighbor_w=1)
            p.prepare(wd, unit_ids=["conv-t1"])
            try:
                corpus = _make_corpus([
                    ("conv-t1/q0",
                     "when did Ann go to the support group",
                     "temporal", 2, ["D1:3"], "conv-t1"),
                ])
                outs.append(AP.run_proxy(
                    corpus, [CONV], p, budgets=(200, None), k=10))
            finally:
                p.cleanup()
        def _norm(res):
            r = copy.deepcopy(res["records"])
            for rec in r:
                rec.pop("retrieve_ms", None)
                (rec.get("provider") or {}).pop("retrieve_ms", None)
                (rec.get("provider") or {}).pop("search_ms", None)
            return r
        assert _norm(outs[0]) == _norm(outs[1])

    def test_aggregates_exclude_unanswerable(self, provider):
        tasks = _make_corpus([
            ("conv-t1/q0", "when did Ann go to the support group",
             "temporal", 2, ["D1:3"], "conv-t1"),
        ])
        tasks.tasks.append(_Task(
            task_id="conv-t1/q9", query="premise check",
            category="adversarial", answerable=False,
            evidence_ids=(), group_id="conv-t1",
            metadata={"category_id": 5, "raw_evidence": []}))
        res = AP.run_proxy(tasks, [CONV], provider,
                           budgets=(4500,), k=10)
        agg = res["aggregate"]["overall"]["4500"]
        assert agg["n"] == 1                    # answerable+gold only
        cat5 = res["records"][1]
        assert cat5["amb_label"] == "adversarial"
        assert cat5["answerable"] is False


# ---------------------------------------------------------------------------
# manifest + markdown
# ---------------------------------------------------------------------------


class TestReport:
    def _args(self, tmp_path):
        ap = AP.build_parser()
        ns = ap.parse_args(["--split", "dev"])
        ns.budgets_parsed = AP._parse_budgets(ns.budgets)
        return ns

    def test_manifest_fields(self, provider, tmp_path):
        corpus = _make_corpus([
            ("conv-t1/q0", "when did Ann go to the support group",
             "temporal", 2, ["D1:3"], "conv-t1"),
        ])
        corpus.dataset_id = "locomo"
        corpus.name = "locomo"
        corpus.split = "dev"
        corpus.source_path = ""
        res = AP.run_proxy(corpus, [CONV], provider,
                           budgets=(4500,), k=10)
        args = self._args(tmp_path)
        args.budgets_parsed = (4500,)
        man = AP._manifest(corpus, provider, args, [CONV])
        assert man["schema"] == AP.SCHEMA
        assert man["dataset"]["split"] == "dev"
        assert man["dataset"]["n_conversations"] == 1
        prov = man["provider"]
        assert prov["provider_revision"] == "verbatim-amb/2"
        assert prov["neighbor_w"] == 1
        assert prov["expansion"] == "session_neighbors/v1"
        assert man["run"]["budgets"] == [4500]
        assert man["assumptions"] and man["code_citations"]

    def test_render_markdown(self, provider, tmp_path):
        corpus = _make_corpus([
            ("conv-t1/q0", "when did Ann go to the support group",
             "temporal", 2, ["D1:3"], "conv-t1"),
        ])
        corpus.dataset_id = "locomo"
        corpus.name = "locomo"
        corpus.split = "dev"
        corpus.source_path = ""
        res = AP.run_proxy(corpus, [CONV], provider,
                           budgets=(4500, None), k=10)
        report = {
            "schema": AP.SCHEMA,
            "manifest": AP._manifest(
                corpus, provider, self._args(tmp_path), [CONV]),
            "expansion": res["expansion"],
            "aggregate": res["aggregate"],
        }
        md = AP.render_markdown(report)
        assert "GEIC@B" in md
        assert "geic_any" in md
        assert "single-hop" in md or "temporal" in md
        assert "provider._expand" in md


# ---------------------------------------------------------------------------
# Track R parity — VerbatimArm session_messages mode (V85-04.04)
# ---------------------------------------------------------------------------


SESS_ITEMS = [
    {"id": "conv-t1:D1:1", "text": "Hi Bob!",
     "speaker": "Ann", "session_id": "conv-t1/session_1",
     "when": "2023-05-08", "group_id": "conv-t1",
     "metadata": {"dia_id": "D1:1"}},
    {"id": "conv-t1:D1:2", "text": "Hey Ann, how are you?",
     "speaker": "Bob", "session_id": "conv-t1/session_1",
     "when": "2023-05-08", "group_id": "conv-t1",
     "metadata": {"dia_id": "D1:2"}},
    {"id": "conv-t1:D1:3",
     "text": "I went to the support group on Monday",
     "speaker": "Ann", "session_id": "conv-t1/session_1",
     "when": "2023-05-08", "group_id": "conv-t1",
     "metadata": {"dia_id": "D1:3"}},
    {"id": "conv-t1:D2:1", "text": "I bought a kayak last weekend",
     "speaker": "Ann", "session_id": "conv-t1/session_2",
     "when": "2023-05-09", "group_id": "conv-t1",
     "metadata": {"dia_id": "D2:1"}},
    {"id": "conv-t1:D2:2", "text": "Wow, where will you paddle?",
     "speaker": "Bob", "session_id": "conv-t1/session_2",
     "when": "2023-05-09", "group_id": "conv-t1",
     "metadata": {"dia_id": "D2:2"}},
]

SESS_TASKS = [
    {"task_id": "conv-t1/q0",
     "query": "when did Ann go to the support group",
     "category": "temporal", "gold_evidence": ["conv-t1:D1:3"],
     "group_id": "conv-t1"},
    {"task_id": "conv-t1/q1", "query": "what did Ann buy",
     "category": "single-hop", "gold_evidence": ["conv-t1:D2:1"],
     "group_id": "conv-t1"},
]


@pytest.fixture()
def sess_corpus():
    return DictCorpus(SESS_ITEMS, SESS_TASKS, name="sess",
                      dataset_id="sess-dict")


class TestSessionMessagesArm:
    def test_ingest_and_ordinals(self, sess_corpus):
        arm = VerbatimArm(ingest_mode="session_messages")
        try:
            rep = arm.ingest(sess_corpus)
            assert rep["ingest_mode"] == "session_messages"
            assert rep["add_errors"] == 0
            geic = rep["geic"]
            assert geic["sessions"] == 2
            assert geic["window"] == 1
            assert geic["meter"] in ("cl100k_base", "tok/v1")
            # real per-session seq ordinals — 3 and 2 turn units.
            s1 = arm._session_sources["conv-t1/session_1"]
            seqs = sorted(
                t["seq"] for t in arm._units.turn_units(s1))
            assert seqs == [0, 1, 2]
            s2 = arm._session_sources["conv-t1/session_2"]
            assert sorted(t["seq"] for t in arm._units.turn_units(s2)) \
                == [0, 1]
        finally:
            arm.close()

    def test_query_ships_geic_diag(self, sess_corpus):
        arm = VerbatimArm(ingest_mode="session_messages",
                          geic_budgets=(4500, None))
        try:
            arm.ingest(sess_corpus)
            out = arm.query(
                {"task_id": "conv-t1/q0",
                 "query": "when did Ann go to the support group",
                 "category": "temporal", "group_id": "conv-t1"},
                k=10)
            g = (out.diag or {}).get("geic")
            assert g is not None and g["budgets"]
            assert set(g["budgets"]) == {"4500", "unbounded"}
            refs = g["budgets"]["unbounded"]["refs"]
            assert refs                             # delivered something
            members = set()
            for m in arm._session_members.values():
                members.update(m)
            assert set(refs) <= members             # real refs only
        finally:
            arm.close()

    def test_item_mode_silent(self, sess_corpus):
        arm = VerbatimArm()                        # default item mode
        try:
            arm.ingest(sess_corpus)
            out = arm.query(
                {"task_id": "conv-t1/q0",
                 "query": "when did Ann go to the support group",
                 "category": "temporal", "group_id": "conv-t1"},
                k=10)
            assert (out.diag or {}).get("geic") is None
        finally:
            arm.close()


class _StubGeicArm:
    """Arm that ships a hand-pinned ``diag["geic"]`` — the scorer-side
    aggregation control (delivered refs vs gold, deterministic)."""

    name = "stub_geic"
    notes: list = []

    def __init__(self, refs_per_budget):
        self._budgets = refs_per_budget
        self.last_ingest = {}

    def ingest(self, corpus):
        return {"indexed": 3}

    def indexed_refs(self):
        return {"conv-t1:D1:1", "conv-t1:D1:2", "conv-t1:D1:3"}

    def query(self, task, k):
        return QueryOutcome(
            refs=["conv-t1:D1:3"], status="ok", latency_ms=1.0,
            n_items=1, delivered_text="x", k=k,
            diag={"geic": {
                "window": 1, "meter": "tok/v1", "basis": "test",
                "budgets": self._budgets}})

    def close(self):
        pass


class TestTrackRGeic:
    def test_scorer_aggregates_delivered_refs(self, sess_corpus):
        arm = _StubGeicArm({
            "1000": {"refs": ["conv-t1:D1:1", "conv-t1:D1:2"],
                     "tokens": 40},
            "unbounded": {"refs": ["conv-t1:D1:3"], "tokens": 90},
        })
        rep = run_track_r(sess_corpus, [arm], k_list=(10,))
        g = rep["arms"]["stub_geic"]["geic"]
        assert g is not None and g["scope"] == "answerable+gold"
        # gold = conv-t1:D1:3 — hit only at unbounded.
        assert g["budgets"]["1000"]["geic_any"] == 0.0
        assert g["budgets"]["unbounded"]["geic_any"] == 0.5  # q0 hits
        assert g["budgets"]["unbounded"]["geic_all"] == 0.5
        pt = {r["task_id"]: r for r in rep["per_task"]}
        assert pt["conv-t1/q0"]["geic"]["unbounded"]["n_gold_hit"] == 1
        assert pt["conv-t1/q0"]["geic"]["1000"]["n_gold_hit"] == 0

    def test_missing_budget_dropped_not_zero(self, sess_corpus):
        # a record lacking a budget key must not count as a 0 for it.
        recs = [
            {"answerable": True, "gold": {"a": 1.0},
             "budgets": {"1000": {"n_gold_hit": 1}}},
            {"answerable": True, "gold": {"b": 1.0},
             "budgets": {}},                     # arm error — no record
        ]
        from eval.v7.track_r import _geic_agg

        agg = _geic_agg(recs, "1000")
        assert agg["n"] == 1 and agg["geic_any"] == 1.0

    def test_no_geic_block_stays_null(self, sess_corpus):
        arm = _StubGeicArm({})
        arm._budgets = {}
        rep = run_track_r(sess_corpus, [arm], k_list=(10,))
        assert rep["arms"]["stub_geic"]["geic"] is None
        assert all(t["geic"] is None for t in rep["per_task"])

    def test_markdown_geic_section(self, sess_corpus):
        arm = _StubGeicArm({
            "unbounded": {"refs": ["conv-t1:D1:3"], "tokens": 90},
        })
        rep = run_track_r(sess_corpus, [arm], k_list=(10,))
        md = render_markdown(rep)
        assert "geic@B" in md and "geic_any" in md
        assert "unbounded" in md

    def test_verbatim_session_arm_end_to_end(self, sess_corpus):
        arm = VerbatimArm(ingest_mode="session_messages",
                          geic_budgets=(4500, None))
        try:
            rep = run_track_r(sess_corpus, [arm], k_list=(10,))
        finally:
            arm.close()
        g = rep["arms"]["verbatim"]["geic"]
        assert g is not None
        assert g["window"] == 1
        assert g["meter"] in ("cl100k_base", "tok/v1")
        assert set(g["budgets"]) == {"4500", "unbounded"}
        assert g["budgets"]["unbounded"]["geic_any"] >= 0.5
