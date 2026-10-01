"""Update-candidate detection tests (SPEC_V5 §30.5, V5-30.18–21,
V5-14.05, E80/E81).

Covers: the four relation classes on true pairs, adversarial twins
(hedged / hypothetical / quoted / hearsay / future-plan / unrelated /
ambiguous-subject / environment-difference produce nothing), the top-3
bound, determinism + idempotent writes, advisory-only lifecycle
semantics, and adopt/dismiss resolution — all through a real on-disk
``Store``.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.memory.types import MemoryRef, UpdateCandidate
from verbatim.querying import (
    MAX_CANDIDATES,
    NewRecord,
    adopt_candidate,
    detect_update_candidates,
    dismiss_candidate,
    list_open_candidates,
    possible_updates,
)

from .conftest import candidates, seed_source

NS = "ns-querying"


def _detect(store, new_text, *, source_id="src-new", revision=1, **fields):
    rec = NewRecord(
        source_id=source_id, revision=revision, text=new_text, **fields
    )
    with store.tx() as conn:
        return detect_update_candidates(conn, NS, rec)


# ---------------------------------------------------------------------
# true pairs → correct relation class
# ---------------------------------------------------------------------


def test_newer_value_version_change(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert len(cands) == 1
    c = cands[0]
    assert c.relation == "newer_value"
    assert c.prior_source_id == "src-prior"
    assert 0.0 < c.score <= 1.0
    # row persisted open
    with v5store.read() as conn:
        rows = candidates(conn, NS)
    assert len(rows) == 1 and rows[0]["state"] == "open"
    assert rows[0]["candidate_id"] == c.candidate_id


def test_newer_value_date_change(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The review meeting is on Tuesday.")
    cands = _detect(v5store, "The review meeting is on Friday.")
    assert [c.relation for c in cands] == ["newer_value"]


def test_negates_prior_affirmative(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "We use Postgres for the backend.")
    cands = _detect(v5store, "We no longer use Postgres for the backend.")
    assert [c.relation for c in cands] == ["negates"]


def test_contradicts_conflicting_value(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "My favorite color is blue.")
    cands = _detect(v5store, "My favorite color is green.")
    assert [c.relation for c in cands] == ["contradicts"]


def test_contradicts_affirmative_against_negation(v5store):
    with v5store.tx() as conn:
        seed_source(
            conn, NS, "src-prior", "We do not use the staging database."
        )
    cands = _detect(v5store, "We use the staging database.")
    assert [c.relation for c in cands] == ["contradicts"]


def test_refines_more_specific(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The retro meeting is on Tuesday.")
    cands = _detect(v5store, "The retro meeting is on Tuesday at 3pm.")
    assert [c.relation for c in cands] == ["refines"]


def test_same_claim_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "We use Postgres for the backend.")
    # restated verbatim — corroboration, not an update
    cands = _detect(v5store, "We use Postgres for the backend.")
    assert cands == []


# ---------------------------------------------------------------------
# adversarial twins (V5-14.05 / E81): produce NO candidates
# ---------------------------------------------------------------------


def test_hedged_new_record_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(v5store, "Maybe the deploy command is deploy-v2 now.")
    assert cands == []
    with v5store.read() as conn:
        assert candidates(conn, NS) == []


def test_hypothetical_new_record_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(
        v5store, "What if the deploy command were deploy-v3?"
    )
    assert cands == []


def test_future_plan_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(
        v5store, "We plan to change the deploy command to deploy-v3."
    )
    assert cands == []


def test_question_new_record_no_candidate(v5store):
    # an interrogative asserts nothing — even about a shared subject
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(v5store, "Is the deploy command still deploy-v1?")
    assert cands == []


def test_quoted_new_record_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(
        v5store, 'The runbook says "the deploy command is deploy-v3".'
    )
    assert cands == []


def test_hedged_prior_not_a_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(
            conn,
            NS,
            "src-prior",
            "I think the deploy command might be deploy-v1.",
            enrichment={"polarity": "hedged"},
        )
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands == []


def test_unrelated_topic_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "I have a cat named Whiskers.")
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands == []


def test_single_shared_term_no_candidate(v5store):
    # one shared generic word does not make two claims the same subject
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "Green paint covered the lobby wall.")
    cands = _detect(v5store, "My favorite color is green.")
    assert cands == []


def test_ambiguous_subject_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    # pronoun subject — "it" binds nothing
    cands = _detect(v5store, "It changed to deploy-v2.")
    assert cands == []


def test_environment_difference_no_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(
            conn,
            NS,
            "src-prior",
            "On macOS the install command is brew install foo.",
        )
    cands = _detect(
        v5store, "On linux the install command is apt install foo."
    )
    assert cands == []


def test_different_namespace_isolated(v5store):
    with v5store.tx() as conn:
        seed_source(conn, "ns-other", "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands == []


def test_superseded_prior_not_live(v5store):
    with v5store.tx() as conn:
        seed_source(
            conn,
            NS,
            "src-prior",
            "The deploy command is deploy-v1.",
            disposition="superseded",
        )
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands == []


def test_self_never_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-new", "The deploy command is deploy-v1.")
        rec = NewRecord(
            source_id="src-new", revision=1, text="The deploy command is deploy-v1."
        )
        cands = detect_update_candidates(conn, NS, rec)
    assert cands == []


# ---------------------------------------------------------------------
# bound + determinism + advisory lifecycle
# ---------------------------------------------------------------------


def test_bounded_top3(v5store):
    with v5store.tx() as conn:
        for i in range(5):
            seed_source(
                conn,
                NS,
                f"src-prior-{i}",
                f"The deploy command is deploy-v{i + 10}.",
            )
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert len(cands) == MAX_CANDIDATES == 3
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    with v5store.read() as conn:
        assert len(candidates(conn, NS)) == 3


def test_deterministic_repeat_detection(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    first = _detect(v5store, "The deploy command is deploy-v2.")
    second = _detect(v5store, "The deploy command is deploy-v2.")
    assert [(c.candidate_id, c.relation, c.score) for c in first] == [
        (c.candidate_id, c.relation, c.score) for c in second
    ]
    # INSERT OR IGNORE — re-detection writes no duplicates
    with v5store.read() as conn:
        assert len(candidates(conn, NS)) == 1


def test_advisory_no_lifecycle_mutation(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    _detect(v5store, "The deploy command is deploy-v2.")
    with v5store.read() as conn:
        row = conn.execute(
            "SELECT disposition, control_version, mutation_head"
            " FROM source_state WHERE source_id = 'src-prior'"
        ).fetchone()
    assert row == ("active", 1, "1")  # nothing changed


def test_serialization_to_update_candidate(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    ups = possible_updates(cands, store_tag="storetag")
    assert len(ups) == 1
    assert isinstance(ups[0], UpdateCandidate)
    ref = MemoryRef.parse(ups[0].ref)
    assert ref.source_id == "src-prior"
    assert ref.expected_revision == 1
    assert ref.namespace == NS
    assert ref.control_version == 1
    assert ups[0].relation == "newer_value"


# ---------------------------------------------------------------------
# enrichment-row path + absent-schema honesty
# ---------------------------------------------------------------------


def test_prior_enrichment_row_used(v5store):
    # the prior's entity lives only in its enrichment row — the surface
    # text never says "Postgres"; one shared term + the enrichment entity
    # is what binds the subject.
    with v5store.tx() as conn:
        seed_source(
            conn,
            NS,
            "src-prior",
            "It handles our analytics database.",
            enrichment={
                "type": "fact",
                "polarity": "affirmative",
                "fields": {"entities": [{"kind": "dictionary", "value": "Postgres"}]},
            },
        )
    cands = _detect(
        v5store, "Postgres was deprecated upstream; analytics unaffected."
    )
    assert [c.relation for c in cands] == ["contradicts"]


def test_enrichment_row_controls_polarity(v5store):
    # same text, but the enrichment row marks the prior hedged — the
    # stored producer label wins and no candidate is written.
    with v5store.tx() as conn:
        seed_source(
            conn,
            NS,
            "src-prior",
            "The deploy command is deploy-v1.",
            enrichment={"polarity": "hedged"},
        )
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands == []
    with v5store.read() as conn:
        assert candidates(conn, NS) == []


def test_missing_v5_schema_is_typed_error(store):
    # the shared TestStore shim carries only v1 DDL — no source_state /
    # update_candidates — so detection must fail with a typed error,
    # never a silent empty answer.
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            detect_update_candidates(
                conn,
                NS,
                NewRecord(source_id="s", revision=1, text="hello"),
            )
    assert ei.value.code == ErrorCode.SCHEMA_UNSUPPORTED


# ---------------------------------------------------------------------
# adopt / dismiss / open-candidate feed
# ---------------------------------------------------------------------


def _seed_pair(store):
    with store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
    return _detect(store, "The deploy command is deploy-v2.")


def test_adopt_and_dismiss(v5store):
    cands = _seed_pair(v5store)
    cid = cands[0].candidate_id
    with v5store.tx() as conn:
        row = adopt_candidate(conn, cid)
        assert row["state"] == "adopted"
        # idempotent re-adopt
        assert adopt_candidate(conn, cid)["state"] == "adopted"
        with pytest.raises(VerbatimError) as ei:
            dismiss_candidate(conn, cid)
        assert ei.value.code == ErrorCode.INVALID_TRANSITION
    # resolved candidates leave the open feed (V5-30.21)
    with v5store.read() as conn:
        assert list_open_candidates(conn, NS) == []


def test_dismiss_then_adopt_conflicts(v5store):
    cands = _seed_pair(v5store)
    cid = cands[0].candidate_id
    with v5store.tx() as conn:
        assert dismiss_candidate(conn, cid)["state"] == "dismissed"
        with pytest.raises(VerbatimError) as ei:
            adopt_candidate(conn, cid)
        assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_unknown_candidate_typed_error(v5store):
    _seed_pair(v5store)
    with v5store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            adopt_candidate(conn, "uc-does-not-exist")
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_dismissed_candidate_never_resurrected(v5store):
    cands = _seed_pair(v5store)
    cid = cands[0].candidate_id
    with v5store.tx() as conn:
        dismiss_candidate(conn, cid)
    # re-running detection must not reopen the resolved row
    again = _detect(v5store, "The deploy command is deploy-v2.")
    assert again and again[0].candidate_id == cid
    with v5store.read() as conn:
        rows = candidates(conn, NS)
    assert len(rows) == 1 and rows[0]["state"] == "dismissed"


def test_open_candidates_feed(v5store):
    _seed_pair(v5store)
    with v5store.read() as conn:
        open_rows = list_open_candidates(conn, NS)
        assert len(open_rows) == 1
        # filtered by either side of the pair
        assert list_open_candidates(conn, NS, source_id="src-prior")
        assert list_open_candidates(conn, NS, source_id="src-new")
        assert list_open_candidates(conn, NS, source_id="src-nope") == []


# ---------------------------------------------------------------------
# update_term_postings — persisted subject-term prefilter
# ---------------------------------------------------------------------


def test_term_postings_prefilter_equivalent(v5store):
    """With full index coverage the prefilter must return the identical
    candidate set as the full resolve — term-disjoint priors are proven
    out by the index rather than resolved-and-rejected."""
    from verbatim.querying import updates as U

    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
        seed_source(
            conn, NS, "src-unrelated",
            "Quixotic banana stands sell twelve varieties of fruit.",
        )
        # Real job writes index rows inside the same commit — mirror that.
        for sid, txt in (
            ("src-prior", "The deploy command is deploy-v1."),
            ("src-unrelated",
             "Quixotic banana stands sell twelve varieties of fruit."),
        ):
            U.write_term_postings(
                conn, namespace=NS, source_id=sid, revision=1, text=txt,
            )

    with v5store.read() as conn:
        rec = NewRecord(
            source_id="src-new", revision=1,
            text="The deploy command is deploy-v2.",
        )
        # prefilter must engage (coverage proven) and prune the disjoint prior
        priors = conn.execute(
            "SELECT source_id, mutation_head, control_version"
            " FROM source_state"
            " WHERE namespace = ? AND disposition IN ('active')"
            " ORDER BY source_id",
            (NS,),
        ).fetchall()
        new = U._resolve_new(conn, rec)
        filtered = U._utp_prefilter(conn, NS, new, list(priors))
        assert filtered is not None
        kept = {r[0] for r in filtered}
        assert kept == {"src-prior"}

        # full plan through the prefilter == full plan without it
        a = U.plan_update_candidates(conn, NS, rec)
        orig = U._utp_prefilter
        U._utp_prefilter = lambda *aa, **kk: None
        try:
            b = U.plan_update_candidates(conn, NS, rec)
        finally:
            U._utp_prefilter = orig
        assert [(c.prior_source_id, c.relation, c.score) for c in a] == [
            (c.prior_source_id, c.relation, c.score) for c in b
        ]
        assert len(a) == 1 and a[0].relation == "newer_value"


def test_term_postings_uncovered_falls_back(v5store):
    """Priors never projected through the index (e.g. stores written by
    older builds) have no marker — coverage fails and the scan uses the
    full resolve, never a pruned guess."""
    from verbatim.querying import updates as U

    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior", "The deploy command is deploy-v1.")
        # one covered, one not → coverage unprovable → full scan
        U.write_term_postings(
            conn, namespace=NS, source_id="src-prior", revision=1,
            text="The deploy command is deploy-v1.",
        )
        seed_source(
            conn, NS, "src-prior2", "The deploy command was deploy-v0."
        )

    with v5store.read() as conn:
        priors = conn.execute(
            "SELECT source_id, mutation_head, control_version"
            " FROM source_state"
            " WHERE namespace = ? AND disposition IN ('active')"
            " ORDER BY source_id",
            (NS,),
        ).fetchall()
        new = U._resolve_new(
            conn,
            NewRecord(
                source_id="src-new", revision=1,
                text="The deploy command is deploy-v2.",
            ),
        )
        assert U._utp_prefilter(conn, NS, new, list(priors)) is None
        # and detection still finds the real prior via the full scan
        cands = U.plan_update_candidates(
            conn, NS,
            NewRecord(
                source_id="src-new", revision=1,
                text="The deploy command is deploy-v2.",
            ),
        )
        assert {c.prior_source_id for c in cands} >= {"src-prior"}
