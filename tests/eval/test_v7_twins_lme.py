"""Durable tests for the owned LongMemEval-S twin generator
(``eval/v7/twins_lme_like.py``, SPEC_V7 §22 V7-22.05).

Covered invariants:

* byte-determinism for a fixed seed (canonical JSONL + digest) and
  divergence for a different seed;
* category coverage — all seven LME-mirrored categories appear, with
  exact weighted counts at the default n=500;
* gold-ref existence — every ``gold_evidence`` session/turn id resolves
  inside the question's own haystack, ids are unique, and
  answerable/abstention flags are consistent with the gold set;
* haystack structure — session dates ascend and precede
  ``question_date``, speakers are user/assistant, turn timestamps are
  strictly increasing within a session;
* knowledge-update pairs — two planted versions exist, the predecessor
  turn is strictly earlier than the superseding turn (session date and
  timestamp), the old value lives in the old turn, the new value lives
  in the new turn and is the answer, and gold covers BOTH turns;
* abstention — zero evidence refs, ``expected_abstain``/``answerable``
  set honestly, and the named topic token appears in no turn (the
  near-miss lure variant is exercised);
* ``preference`` sub-annotation — present on both preference-bearing
  categories, absent elsewhere, and its stated refs resolve to gold
  turns containing the declared value tokens;
* evidence-token uniqueness — planted check substrings occur in exactly
  one turn, so gold recall is exactly measurable.
"""

from __future__ import annotations

import datetime

import pytest

from eval.v7 import twins_lme_like as T


@pytest.fixture(scope="module")
def corpus():
    """The default-scale corpus (n=500) — generated once per module."""
    return T.generate()


@pytest.fixture(scope="module")
def small():
    """A smaller, differently-shaped corpus for structural checks."""
    return T.generate(seed=7, n_questions=140, sessions_per_user=6,
                      min_filler_pairs=1, max_filler_pairs=3)


def _turn_map(question):
    """turn_id -> (session, turn) for one question's haystack."""
    out = {}
    for s in question["sessions"]:
        for t in s["turns"]:
            out[t["turn_id"]] = (s, t)
    return out


def _all_texts(question):
    return [t["text"] for s in question["sessions"] for t in s["turns"]]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism_byte_identical():
    a = T.generate(seed=123, n_questions=60, sessions_per_user=6)
    b = T.generate(seed=123, n_questions=60, sessions_per_user=6)
    assert T.to_jsonl(a) == T.to_jsonl(b)
    assert T.corpus_digest(a) == T.corpus_digest(b)
    # different seed must actually differ
    c = T.generate(seed=124, n_questions=60, sessions_per_user=6)
    assert T.to_jsonl(a) != T.to_jsonl(c)


# ---------------------------------------------------------------------------
# category coverage
# ---------------------------------------------------------------------------


def test_category_coverage_default(corpus):
    assert corpus["corpus"] == T.CORPUS_NAME
    assert corpus["generator"] == T.GENERATOR_ID
    assert len(corpus["questions"]) == 500
    counts = corpus["category_counts"]
    assert set(counts) == set(T.CATEGORIES)
    # weights are per-100; at n=500 each category appears exactly 5*w.
    for cat, w in T.CATEGORY_WEIGHTS.items():
        assert counts[cat] == 5 * w, cat


def test_category_coverage_small(small):
    seen = {q["category"] for q in small["questions"]}
    assert seen == set(T.CATEGORIES)


# ---------------------------------------------------------------------------
# gold-ref existence + flag consistency + haystack structure
# ---------------------------------------------------------------------------


def test_gold_refs_resolve(small):
    for q in small["questions"]:
        sessions = {s["session_id"]: s for s in q["sessions"]}
        tmap = _turn_map(q)
        # unique ids
        assert len(tmap) == sum(len(s["turns"]) for s in q["sessions"])
        assert len(sessions) == len(q["sessions"])
        flat_gold = []
        for ev in q["gold_evidence"]:
            assert ev["session_id"] in sessions
            for tid in ev["turn_ids"]:
                assert tid in tmap
                s, _ = tmap[tid]
                assert s["session_id"] == ev["session_id"]
                flat_gold.append(tid)
        assert flat_gold == q["gold_turn_ids"]
        assert [e["session_id"] for e in q["gold_evidence"]] == \
            q["gold_session_ids"]
        # flag consistency
        if q["expected_abstain"]:
            assert not q["answerable"]
            assert q["answer"] is None
            assert q["gold_evidence"] == []
        else:
            assert q["answerable"]
            assert q["answer"] is not None
            assert len(flat_gold) >= 1


def test_haystack_structure(small):
    for q in small["questions"]:
        qdate = datetime.date.fromisoformat(q["question_date"])
        dates = [datetime.date.fromisoformat(d)
                 for d in q["haystack_dates"]]
        assert dates == sorted(dates)
        assert all(d < qdate for d in dates)
        for s, d in zip(q["sessions"], dates):
            assert s["date"] == d.isoformat()
            ts = [t["timestamp_us"] for t in s["turns"]]
            assert ts == sorted(ts) and len(set(ts)) == len(ts)
            for t in s["turns"]:
                assert t["speaker"] in ("user", "assistant")
                assert t["turn_id"].startswith(s["session_id"])
                assert t["text"].strip()


# ---------------------------------------------------------------------------
# knowledge_updates — real two-version pairs with correct ordering
# ---------------------------------------------------------------------------


def test_knowledge_update_pairs(corpus):
    kus = [q for q in corpus["questions"]
           if q["category"] == "knowledge_updates"]
    assert len(kus) == corpus["category_counts"]["knowledge_updates"]
    for q in kus:
        ku = q["knowledge_update"]
        assert ku is not None
        assert ku["old_value"] != ku["new_value"]
        tmap = _turn_map(q)
        s_old, t_old = tmap[ku["old_turn_id"]]
        s_new, t_new = tmap[ku["new_turn_id"]]
        # the two versions really exist in the haystack
        assert s_old["session_id"] == ku["old_session_id"]
        assert s_new["session_id"] == ku["new_session_id"]
        assert ku["old_turn_id"] != ku["new_turn_id"]
        # correct ordering: strictly earlier session date and timestamp
        assert ku["old_date"] < ku["new_date"]
        assert s_old["date"] == ku["old_date"]
        assert s_new["date"] == ku["new_date"]
        assert t_old["timestamp_us"] < t_new["timestamp_us"]
        # values live in the right turns; answer is the NEW value
        assert ku["old_value"].lower() in t_old["text"].lower()
        assert ku["new_value"].lower() in t_new["text"].lower()
        assert q["answer"] == ku["new_value"]
        # gold covers BOTH turns (recall_all of both sessions, §24.2)
        assert set(q["gold_turn_ids"]) == {
            ku["old_turn_id"], ku["new_turn_id"]}
        assert len(q["gold_session_ids"]) == 2
        # the new value is unique to its turn; the old value must appear
        # at least in the old turn (it may be echoed in the new one)
        texts = _all_texts(q)
        assert sum(ku["new_value"].lower() in x.lower()
                   for x in texts) == 1
        assert sum(ku["old_value"].lower() in x.lower()
                   for x in texts) >= 1


# ---------------------------------------------------------------------------
# abstention
# ---------------------------------------------------------------------------


def test_abstention_items(corpus):
    abs_items = [q for q in corpus["questions"]
                 if q["category"] == "abstention"]
    assert len(abs_items) == corpus["category_counts"]["abstention"]
    near_misses = 0
    for q in abs_items:
        assert q["expected_abstain"] and not q["answerable"]
        assert q["answer"] is None
        assert q["gold_evidence"] == []
        assert q["gold_turn_ids"] == []
        topic = q["checks"]["absent_topic"]
        assert topic
        for text in _all_texts(q):
            assert topic.lower() not in text.lower()
        near_misses += bool(q["checks"]["near_miss"])
    # both variants exercised (with 25 items a 0.5 coin cannot
    # plausibly land on one side; assert both occur)
    assert 0 < near_misses < len(abs_items)


# ---------------------------------------------------------------------------
# preference sub-annotation
# ---------------------------------------------------------------------------


def test_preference_annotation(corpus):
    pref_cats = {"single_session_preference", "multi_session_user"}
    for q in corpus["questions"]:
        if q["category"] not in pref_cats:
            assert q["preference"] is None, q["id"]
            continue
        p = q["preference"]
        assert p is not None
        tmap = _turn_map(q)
        gold = set(q["gold_turn_ids"])
        if q["category"] == "single_session_preference":
            assert p["mode"] == "single_session"
            assert len(p["stated"]) == 1 and len(p["values"]) == 1
        else:
            assert p["mode"] == "multi_session"
            assert len(p["stated"]) == len(p["values"]) >= 3
            # facets really span multiple sessions
            assert len({r["session_id"] for r in p["stated"]}) == \
                len(p["stated"])
        for ref, val in zip(p["stated"], p["values"]):
            assert ref["turn_id"] in gold
            s, t = tmap[ref["turn_id"]]
            assert s["session_id"] == ref["session_id"]
            assert val.lower() in t["text"].lower()


# ---------------------------------------------------------------------------
# temporal_reasoning
# ---------------------------------------------------------------------------


def test_temporal_annotations(corpus):
    trs = [q for q in corpus["questions"]
           if q["category"] == "temporal_reasoning"]
    assert len(trs) == corpus["category_counts"]["temporal_reasoning"]
    modes = set()
    for q in trs:
        an = q["temporal"]
        assert an is not None
        modes.add(an["mode"])
        edate = datetime.date.fromisoformat(an["event_date"])
        qdate = datetime.date.fromisoformat(q["question_date"])
        assert an["question_date"] == q["question_date"]
        assert an["days"] == abs((qdate - edate).days)
        assert an["weekday"] == T._WEEKDAYS[edate.weekday()]
        assert len(q["gold_turn_ids"]) == 1
        # the absolute date is stated inside the gold turn
        tmap = _turn_map(q)
        _, t = tmap[q["gold_turn_ids"][0]]
        assert T._fmt_day(edate) in t["text"]
    assert modes == {"days_ago", "weeks_ago", "weekday", "days_until"}


# ---------------------------------------------------------------------------
# evidence-token uniqueness — planted check substrings appear in
# exactly one turn, so gold recall is exactly measurable
# ---------------------------------------------------------------------------


def test_evidence_token_uniqueness(corpus):
    checked = 0
    for q in corpus["questions"]:
        subs = q["checks"].get("evidence_substrings")
        if not subs:
            continue
        checked += 1
        texts = _all_texts(q)
        gold = set(q["gold_turn_ids"])
        tmap = _turn_map(q)
        for sub in subs:
            hits = [tid for tid, (_, t) in tmap.items()
                    if sub.lower() in t["text"].lower()]
            assert len(hits) == 1, (q["id"], sub, hits)
            assert hits[0] in gold
    assert checked > 0


def test_multi_session_gold_spans_sessions(corpus):
    count_items = 0
    for q in corpus["questions"]:
        if q["category"] not in ("multi_session", "multi_session_user"):
            continue
        assert len(set(q["gold_session_ids"])) >= 2, q["id"]
        if q["category"] == "multi_session":
            tok = q["checks"]["count_token"]
            n = sum(tok.lower() in x.lower() for x in _all_texts(q))
            assert n == q["checks"]["expected_count"], q["id"]
            assert q["checks"]["expected_count"] == \
                len(q["gold_turn_ids"])
            if q["subtype"] == "multi_session_count":
                count_items += 1
                assert q["answer"] == str(q["checks"]["expected_count"])
    assert count_items > 0


def test_information_extraction_subtypes(corpus):
    ies = [q for q in corpus["questions"]
           if q["category"] == "information_extraction"]
    subs = {q["subtype"] for q in ies}
    assert subs == {"single_session_user", "single_session_assistant"}
    for q in ies:
        assert len(q["gold_turn_ids"]) == 1
        tmap = _turn_map(q)
        _, t = tmap[q["gold_turn_ids"][0]]
        if q["subtype"] == "single_session_user":
            assert t["speaker"] == "user"
        else:
            assert t["speaker"] == "assistant"


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


def test_parameter_validation():
    with pytest.raises(ValueError):
        T.generate(n_questions=0)
    with pytest.raises(ValueError):
        T.generate(sessions_per_user=2)
    with pytest.raises(ValueError):
        T.generate(sessions_per_user=T.MAX_SESSIONS + 1)
    with pytest.raises(ValueError):
        T.generate(min_filler_pairs=0)
    with pytest.raises(ValueError):
        T.generate(min_filler_pairs=4, max_filler_pairs=3)
    with pytest.raises(ValueError):
        T.generate(max_filler_pairs=len(T._FILLERS) + 1)
    # smallest legal shape still yields every category's machinery
    c = T.generate(seed=3, n_questions=35, sessions_per_user=4)
    assert len(c["questions"]) == 35
