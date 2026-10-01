"""Tests for the owned LoCoMo-like twin generator
(``eval/v7/twins_locomo_like.py``, V7-22.05).

Covers: byte-determinism per seed (dict + JSONL), category coverage,
gold-evidence referential integrity, the answerable/abstain contract,
planted-fact consistency (``must_contain`` lands in the turn text),
cross-session multi-hop evidence, resolvable temporal answers, and the
never-discussed / speaker-mismatch trap hygiene.
"""

from __future__ import annotations

import json
import unittest
from datetime import date

from eval.v7 import twins_locomo_like as T


def _gen(**kw):
    kw.setdefault("seed", 7)
    kw.setdefault("n_sessions", 10)
    kw.setdefault("turns_per_session", 26)
    kw.setdefault("questions_per_category", 12)
    return T.generate(**kw)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        a = _gen()
        b = _gen()
        self.assertEqual(a, b)
        self.assertEqual(a["digest"], b["digest"])

    def test_jsonl_byte_identical(self):
        import tempfile, pathlib
        a, b = _gen(), _gen()
        with tempfile.TemporaryDirectory() as d:
            pa = T.to_jsonl(a, pathlib.Path(d) / "a.jsonl")
            pb = T.to_jsonl(b, pathlib.Path(d) / "b.jsonl")
            self.assertEqual(pa.read_bytes(), pb.read_bytes())

    def test_different_seed_differs(self):
        a = _gen(seed=7)
        b = _gen(seed=8)
        self.assertNotEqual(a["digest"], b["digest"])

    def test_jsonl_roundtrip(self):
        import tempfile, pathlib
        c = _gen()
        with tempfile.TemporaryDirectory() as d:
            p = T.to_jsonl(c, pathlib.Path(d) / "c.jsonl")
            recs = list(T.iter_jsonl(p))
        kinds = [r["record"] for r in recs]
        self.assertEqual(kinds[0], "corpus")
        self.assertEqual(
            sum(1 for k in kinds if k == "turn"), len(c["turns"]))
        self.assertEqual(
            sum(1 for k in kinds if k == "question"),
            len(c["questions"]))
        self.assertEqual(
            sum(1 for k in kinds if k == "session"),
            len(c["sessions"]))
        self.assertEqual(
            sum(1 for k in kinds if k == "fact"), len(c["facts"]))


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.c = _gen()

    def test_every_category_present(self):
        cats = {q["category"] for q in self.c["questions"]}
        self.assertEqual(cats, set(T.CATEGORIES))
        for cat in T.CATEGORIES:
            self.assertGreaterEqual(
                self.c["stats"]["per_category"][cat], 10,
                f"category {cat} under-populated")

    def test_category_counts_param(self):
        c = T.generate(seed=11, n_sessions=14, turns_per_session="auto",
                       category_counts={"single_hop": 30, "multi_hop": 18,
                                        "temporal": 25, "open_domain": 15,
                                        "abstain": 12})
        self.assertEqual(
            sum(1 for q in c["questions"] if q["category"] == "single_hop"),
            30)
        self.assertEqual(
            sum(1 for q in c["questions"] if q["category"] == "abstain"),
            12)

    def test_subtype_variety(self):
        subs = {q["subtype"] for q in self.c["questions"]}
        # every multi-hop shape and abstain trap type appears at qpc=12
        for st in ("shared_hobby", "shared_city", "paired_routine",
                   "compare_race", "purchase_chain", "event_attr",
                   "repeat_visit"):
            self.assertIn(st, subs)
        for st in ("never_discussed", "speaker_mismatch",
                   "out_of_timeline"):
            self.assertIn(st, subs)
        for st in ("relative", "explicit", "ordering", "duration",
                   "session_date"):
            self.assertIn(st, subs)

    def test_scale_to_suite_size(self):
        # V7-22.05 asks for >=500 questions per owned suite.
        c = T.generate(seed=3, n_sessions=30, turns_per_session="auto",
                       questions_per_category=105)
        self.assertGreaterEqual(c["stats"]["n_questions"], 500)
        for cat in T.CATEGORIES:
            self.assertGreaterEqual(
                c["stats"]["per_category"][cat], 100)


class TestEvidenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.c = _gen()
        self.turns = {t["turn_id"]: t for t in self.c["turns"]}

    def test_evidence_refs_exist(self):
        for q in self.c["questions"]:
            for tid in q["evidence"]:
                self.assertIn(tid, self.turns, q["qid"])

    def test_answerable_contract(self):
        for q in self.c["questions"]:
            if q["category"] == "abstain":
                self.assertFalse(q["answerable"], q["qid"])
                self.assertEqual(q["evidence"], [], q["qid"])
                self.assertIsNone(q["answer"], q["qid"])
            else:
                self.assertTrue(q["answerable"], q["qid"])
                self.assertTrue(q["evidence"], q["qid"])
                self.assertIsNotNone(q["answer"], q["qid"])

    def test_multi_hop_two_plus_turns_cross_session(self):
        mh = [q for q in self.c["questions"] if q["category"] == "multi_hop"]
        self.assertTrue(mh)
        for q in mh:
            self.assertGreaterEqual(len(q["evidence"]), 2, q["qid"])
            sess = {self.turns[t]["session_id"] for t in q["evidence"]}
            self.assertGreater(len(sess), 1,
                               f"{q['qid']} evidence not cross-session")

    def test_question_fields(self):
        for q in self.c["questions"]:
            for k in ("qid", "category", "subtype", "query", "answer",
                      "answerable", "evidence", "question_time"):
                self.assertIn(k, q)
            self.assertTrue(q["qid"].startswith(q["category"]))

    def test_turn_schema(self):
        for t in self.c["turns"]:
            for k in ("turn_id", "session_id", "turn_index", "speaker",
                      "role", "timestamp", "timestamp_us", "text", "kind"):
                self.assertIn(k, t)
            self.assertIn(t["kind"], ("evidence", "distractor", "filler"))
            self.assertIn(t["speaker"],
                          {s["name"] for s in self.c["speakers"]})

    def test_session_dates_increase(self):
        starts = [s["started_at"] for s in self.c["sessions"]]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(len({s["date"] for s in self.c["sessions"]}), 1)


class TestPlantedFacts(unittest.TestCase):
    def setUp(self):
        self.c = _gen()
        self.turns = {t["turn_id"]: t for t in self.c["turns"]}

    def test_must_contain_lands_in_text(self):
        for f in self.c["facts"]:
            turn = self.turns[f["turn_id"]]
            for m in f["must_contain"]:
                self.assertIn(m, turn["text"],
                              f"{f['fact_id']} missing {m!r}")
            self.assertEqual(turn["speaker"], f["subject"],
                             f["fact_id"])

    def test_evidence_facts_point_at_evidence_turns(self):
        for f in self.c["facts"]:
            if f["kind"] == "distractor":
                self.assertEqual(
                    self.turns[f["turn_id"]]["kind"], "distractor")
            else:
                self.assertEqual(
                    self.turns[f["turn_id"]]["kind"], "evidence")


class TestTemporal(unittest.TestCase):
    def setUp(self):
        self.c = _gen()
        self.turns = {t["turn_id"]: t for t in self.c["turns"]}

    def test_relative_answers_precede_turn(self):
        rel = [q for q in self.c["questions"]
               if q["subtype"] == "relative"]
        self.assertTrue(rel)
        for q in rel:
            ans = date.fromisoformat(q["answer"])
            tdate = date.fromisoformat(
                self.turns[q["evidence"][0]]["timestamp"][:10])
            self.assertLess(ans, tdate, q["qid"])

    def test_explicit_answers_parse(self):
        exp = [q for q in self.c["questions"]
               if q["subtype"] == "explicit"]
        self.assertTrue(exp)
        for q in exp:
            date.fromisoformat(q["answer"])  # must parse

    def test_ordering_two_turns(self):
        ordering = [q for q in self.c["questions"]
                    if q["subtype"] == "ordering"]
        self.assertTrue(ordering)
        for q in ordering:
            self.assertEqual(len(q["evidence"]), 2, q["qid"])
            self.assertIn(q["answer"], ("before", "after"))

    def test_relative_phrase_in_text(self):
        rel = [q for q in self.c["questions"]
               if q["subtype"] == "relative"]
        needles = ("ago", "yesterday", "week", "last ", "past ")
        for q in rel:
            txt = self.turns[q["evidence"][0]]["text"].lower()
            self.assertTrue(any(n in txt for n in needles), q["qid"])

    def test_session_date_answers(self):
        sd = [q for q in self.c["questions"]
              if q["subtype"] == "session_date"]
        self.assertTrue(sd)
        for q in sd:
            turn = self.turns[q["evidence"][0]]
            self.assertEqual(q["answer"], turn["timestamp"][:10])


class TestAbstainTraps(unittest.TestCase):
    def setUp(self):
        self.c = _gen()
        self.turns = {t["turn_id"]: t for t in self.c["turns"]}

    def test_never_discussed_absent(self):
        nd = [q for q in self.c["questions"]
              if q["subtype"] == "never_discussed"]
        self.assertTrue(nd)
        corpus_text = " ".join(t["text"] for t in self.c["turns"]).lower()
        for q in nd:
            self.assertNotIn(q["trap"]["topic_key"], corpus_text,
                             q["qid"])

    def test_speaker_mismatch(self):
        sm = [q for q in self.c["questions"]
              if q["subtype"] == "speaker_mismatch"]
        self.assertTrue(sm)
        by_fact = {f["fact_id"]: f for f in self.c["facts"]}
        for q in sm:
            trap = q["trap"]
            self.assertIn(trap["actual_speaker"],
                          {s["name"] for s in self.c["speakers"]})
            # the related turn exists and was spoken by the other speaker
            rt = self.turns[trap["related_turn"]]
            self.assertEqual(rt["speaker"], trap["actual_speaker"])
            # the mentioned string really is in the related turn
            self.assertIn(trap["mention"], rt["text"])
            fact = by_fact[trap["related_fact"]]
            self.assertEqual(fact["subject"], trap["actual_speaker"])

    def test_out_of_timeline_predates_corpus(self):
        ot = [q for q in self.c["questions"]
              if q["subtype"] == "out_of_timeline"]
        self.assertTrue(ot)
        start = self.c["timeline"]["start"]
        for q in ot:
            self.assertLess(q["trap"]["date"], start[:10])


class TestOpenDomain(unittest.TestCase):
    def test_low_lexical_overlap(self):
        # the hard slice: question wording diverges from evidence wording
        c = _gen()
        turns = {t["turn_id"]: t for t in c["turns"]}
        od = [q for q in c["questions"] if q["category"] == "open_domain"]
        self.assertTrue(od)
        stop = {"what", "does", "the", "a", "an", "in", "to", "do", "how",
                "is", "of", "at", "on", "for", "with", "their", "who",
                "where", "when", "did", "and", "or", "by", "can", "people",
                "s", "most", "every"}
        import re
        tok = re.compile(r"[a-z]+")
        overlaps = 0
        for q in od:
            q_terms = {w for w in tok.findall(q["query"].lower())
                       if w not in stop and len(w) > 2
                       and w not in {n.lower() for n in
                                     (t["speaker"] for t in
                                      c["turns"][:1])}}
            q_terms -= {s["name"].lower() for s in c["speakers"]}
            ev_terms = {w for w in tok.findall(
                turns[q["evidence"][0]]["text"].lower()) if w not in stop}
            shared = q_terms & ev_terms
            if not shared:
                overlaps += 1
        # at least half of open-domain items share no content word
        self.assertGreaterEqual(overlaps, len(od) // 2)


if __name__ == "__main__":
    unittest.main()
