"""Unit tests for the eval corpus generator and harness plumbing.

These guard the measurement infrastructure itself: corpus determinism,
gold-link integrity, and the honest-reporting invariants (denominators
never shrink, defects stay visible).
"""

from __future__ import annotations

import pytest

from eval.corpus import (
    corpus_stats,
    generate_corpus,
    generate_realistic_chat_corpus,
)


def test_corpus_is_deterministic():
    a = generate_corpus(200, seed=42)
    b = generate_corpus(200, seed=42)
    assert a.sha256() == b.sha256()
    assert [s.text for s in a.statements] == [s.text for s in b.statements]


def test_corpus_seed_changes_output():
    a = generate_corpus(200, seed=42)
    b = generate_corpus(200, seed=7)
    assert a.sha256() != b.sha256()


def test_corpus_scale_and_shape():
    c = generate_corpus(1000, seed=42)
    stats = corpus_stats(c)
    assert stats["statements"] == 1000
    assert stats["update_pairs"] == 300
    assert stats["queries"]["no_answer"] == 200
    # every statement carries gold metadata
    for s in c.statements:
        assert s.stmt_id and s.text.endswith(".")
        assert s.event_us > 0
        assert s.tokens, f"statement {s.stmt_id} has no salient tokens"


def test_update_pairs_are_consistent():
    c = generate_corpus(400, seed=42)
    by_id = c.by_id()
    assert len(by_id) == len(c.statements), "duplicate statement ids"
    for u in c.update_pairs:
        old_s, new_s = by_id[u.old_stmt_id], by_id[u.new_stmt_id]
        assert old_s.role == "update_old" and new_s.role == "update_new"
        assert old_s.pair_id == new_s.pair_id == u.pair_id
        assert old_s.event_us < new_s.event_us, (
            "successor must occur strictly after the predecessor"
        )
        assert u.change_us == new_s.event_us
        assert old_s.slot == new_s.slot


def test_no_answer_queries_have_no_gold_and_no_overlap():
    c = generate_corpus(400, seed=42)
    corpus_tokens = set()
    for s in c.statements:
        corpus_tokens.update(s.tokens)
    for q in c.no_answer_queries:
        assert q.expect == ()
        # the held-out subject token must not appear in any statement —
        # otherwise the query is answerable and the probe is invalid
        probe = q.text.split()[-1].lower()
        assert probe not in corpus_tokens, (
            f"no-answer probe {probe!r} leaks into corpus statements"
        )


def test_contradiction_pairs_link_both_sides():
    c = generate_corpus(400, seed=42)
    by_id = c.by_id()
    for a_id, b_id in c.contradiction_pairs:
        a, b = by_id[a_id], by_id[b_id]
        assert a.role == "contradiction_a" and b.role == "contradiction_b"
        assert a.pair_id == b.pair_id
        assert a.text != b.text


def test_realistic_chat_slice_covers_unterminated_and_structured_queries(tmp_path):
    from eval.harness import run_harness

    corpus = generate_realistic_chat_corpus()
    assert all(not stmt.text.endswith(".") for stmt in corpus.statements)
    result = run_harness(corpus, work_dir=str(tmp_path), top_k=3)
    assert result.coverage["extraction_coverage"] == 1.0
    assert result.processing["claims_created"] == len(corpus.statements)
    for kind in ("point", "current", "historical"):
        assert result.recall[kind]["hit_rate"] == 1.0
    assert result.recall["no_answer"]["false_positive_rate"] == 0.0
    assert result.defects == []


def test_harness_result_denominators_and_defects(tmp_path):
    """A small end-to-end run: every query is scored, defects surface."""
    from eval.harness import run_harness

    c = generate_corpus(60, seed=42)
    res = run_harness(c, work_dir=str(tmp_path), top_k=3)
    total = sum(
        (res.recall.get(k) or {}).get("n", 0)
        for k in ("point", "current", "historical", "no_answer")
    )
    assert total == len(c.queries), (
        "queries were dropped from scoring — denominators must hold"
    )
    assert res.ingest["accepted"] == len(c.statements)
    assert res.coverage["extraction_coverage"] == 1.0
    # The FTS projection-generation defect this harness was built to
    # surface is fixed: sequential transitions index each newly active
    # claim at the current generation, so nothing is left stranded.
    assert res.defects == [], f"unexpected harness defects: {res.defects}"
    assert res.recall["latency_ms"]["n"] == len(c.queries)
