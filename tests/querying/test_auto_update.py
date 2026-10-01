"""auto_safe update-application tests (SPEC_V6 V6-03.03).

Covers: every guard fires and keeps the pair advisory (hedged /
intent-modal / cross-entity / no-anchor / contradicts / negates /
below-floor / wrong policy / stale candidate), the applied path lands
the identical lifecycle state the ``add(replaces=)`` coordinator effect
produces, the advisory row resolves through ``adopt_candidate``, the
``update_policy:<ns>`` meta row validates, and the published twin
corpus carries ≥4 true + ≥8 false pairs — all through real on-disk
``Store`` instances plus one ``Memory`` facade end-to-end pass.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.governance import current_epoch
from verbatim.querying.auto_update import (
    AUTO_SAFE_MIN,
    auto_safe_replace,
    evaluate_auto_safe,
    get_update_policy,
    set_update_policy,
)
from verbatim.querying.updates import (
    NewRecord,
    detect_update_candidates,
    list_open_candidates,
)
from verbatim.sourcestate import transitions as _stransitions

from .conftest import candidates, seed_source

NS = "ns-auto"


def _policy(store, policy="auto_safe", namespace=NS):
    with store.tx() as conn:
        set_update_policy(conn, namespace, policy)


def _detect(store, new_text, *, source_id="src-new", revision=1, **fields):
    rec = NewRecord(
        source_id=source_id, revision=revision, text=new_text, **fields
    )
    with store.tx() as conn:
        return detect_update_candidates(conn, NS, rec)


def _apply(store, new_text, cand, *, source_id="src-new", revision=1):
    """auto_safe_replace inside one write tx — returns (result, conn-state)."""
    rec = NewRecord(
        source_id=source_id, revision=revision, text=new_text, **{}
    )
    with store.tx() as conn:
        out = auto_safe_replace(conn, namespace=NS, new_record=rec,
                                candidate=cand, store=store)
    return out


def _decision(store, new_text, cand, *, source_id="src-new", revision=1):
    rec = NewRecord(source_id=source_id, revision=revision, text=new_text)
    with store.read() as conn:
        return evaluate_auto_safe(
            conn, namespace=NS, new_record=rec, candidate=cand
        )


def _fabricated_cand(prior_sid="src-prior", *, relation="newer_value",
                     score=0.95, state="open", new_sid="src-new",
                     new_rev=1, prior_rev=1, prior_cv=1):
    return {
        "candidate_id": None,
        "namespace": NS,
        "new_source_id": new_sid,
        "new_revision": new_rev,
        "prior_source_id": prior_sid,
        "prior_revision": prior_rev,
        "prior_control_version": prior_cv,
        "relation": relation,
        "score": score,
        "reason": "fabricated",
        "state": state,
    }


def _state(store, sid):
    with store.read() as conn:
        return conn.execute(
            "SELECT namespace, control_version, mutation_head, disposition,"
            " superseded_by, effective_at FROM source_state"
            " WHERE source_id = ?",
            (sid,),
        ).fetchone()


# ---------------------------------------------------------------------
# policy setter / getter
# ---------------------------------------------------------------------


def test_policy_defaults_advisory(v5store):
    with v5store.read() as conn:
        assert get_update_policy(conn, NS) == "advisory"


def test_policy_roundtrip(v5store):
    _policy(v5store, "auto_safe")
    with v5store.read() as conn:
        assert get_update_policy(conn, NS) == "auto_safe"
    _policy(v5store, "advisory")
    with v5store.read() as conn:
        assert get_update_policy(conn, NS) == "advisory"


def test_policy_unknown_value_typed_error(v5store):
    with v5store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            set_update_policy(conn, NS, "merge")
    assert ei.value.code == ErrorCode.VALIDATION


def test_policy_unknown_stored_value_fails_closed(v5store):
    with v5store.tx() as conn:
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            ("update_policy:" + NS, '"yolo"'),
        )
    with v5store.read() as conn:
        assert get_update_policy(conn, NS) == "advisory"


# ---------------------------------------------------------------------
# applied path — same lifecycle state as add(replaces=)
# ---------------------------------------------------------------------


def test_applies_true_pair(v5store):
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "Ticket INC-431: the deploy command is deploy-v1.")
        set_update_policy(conn, NS, "auto_safe")
    cands = _detect(v5store,
                    "Ticket INC-431: the deploy command is deploy-v2.")
    assert len(cands) == 1 and cands[0].relation == "newer_value"
    assert cands[0].score >= AUTO_SAFE_MIN

    out = _apply(
        v5store, "Ticket INC-431: the deploy command is deploy-v2.",
        cands[0],
    )
    assert out is not None and out["applied"] is True
    assert out["decision"]["disposition"] == "superseded"
    assert out["decision"]["adopted_candidate"] is True

    row = _state(v5store, "src-prior")
    assert row[3] == "superseded"
    assert row[4] == "src-new:1"
    assert row[2] == "src-new:1"  # mutation_head fenced to successor
    assert row[1] == 2  # control_version bumped once
    assert row[5] is not None  # effective_at = commit time

    # advisory row resolved — the open feed no longer lists it
    with v5store.read() as conn:
        assert list_open_candidates(conn, NS) == []
        rows = candidates(conn, NS)
        assert len(rows) == 1 and rows[0]["state"] == "adopted"


def test_applied_matches_replaces_lifecycle(v5store):
    """Auto-apply and the manual replaces= coordinator effect must land
    byte-identical lifecycle fields."""
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-auto",
                    "INC-431 deploy command is deploy-v1.")
        seed_source(conn, NS, "src-manual",
                    "INC-431 deploy command is deploy-v1.")
        set_update_policy(conn, NS, "auto_safe")
        # The manual path — exactly what Memory.add(replaces=) calls.
        _stransitions.transition(
            conn,
            "src-manual",
            expected_control_version=1,
            disposition="supersede",
            superseded_by="src-new:1",
            producer="memory/v5",
            expected_revision=1,
            epoch_vector={NS: current_epoch(conn, NS)},
            actor="owner",
            operation_id="op-manual",
        )
        cands = detect_update_candidates(
            conn, NS,
            NewRecord(source_id="src-new", revision=1,
                      text="INC-431 deploy command is deploy-v2."),
        )
        assert cands
        cand = next(c for c in cands if c.prior_source_id == "src-auto")
        out = auto_safe_replace(
            conn, namespace=NS,
            new_record=NewRecord(
                source_id="src-new", revision=1,
                text="INC-431 deploy command is deploy-v2.",
            ),
            candidate=cand,
        )
        assert out is not None and out["applied"] is True

    auto = _state(v5store, "src-auto")
    manual = _state(v5store, "src-manual")
    # (namespace, control_version, mutation_head, disposition,
    #  superseded_by, effective_at) — identical lifecycle fields;
    # effective_at is per-call commit time so compare the rest.
    assert auto[1:5] == manual[1:5]
    assert auto[5] is not None and manual[5] is not None


def test_memory_facade_end_to_end(tmp_path):
    """Through the real facade: add prior, add new, auto-apply lands
    the same supersession the replaces= path would."""
    from verbatim.memory.facade import Memory

    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        ns = mem._namespace
        with mem._store.tx() as conn:
            set_update_policy(conn, ns, "auto_safe")
        r1 = mem.add("Ticket INC-431: the deploy command is deploy-v1.")
        r2 = mem.add("Ticket INC-431: the deploy command is deploy-v2.")
        # Integrated path: auto_safe applied inside the add commit —
        # applied candidates never surface as advisory possible_updates.
        assert r2.possible_updates == []

        with mem._store.read() as conn:
            cands = list_open_candidates(conn, ns, source_id=r2.memory_id)
            assert cands == []  # resolved by the in-tx application, not open

        with mem._store.read() as conn:
            row = conn.execute(
                "SELECT disposition, superseded_by FROM source_state"
                " WHERE source_id = ?",
                (r1.memory_id,),
            ).fetchone()
        assert row[0] == "superseded"
        assert row[1] == f"{r2.memory_id}:{r2.source_revision}"

        # The persisted advisory row was resolved in the same commit —
        # candidate_open keeps any re-application idempotent.
        with mem._store.read() as conn:
            row = conn.execute(
                "SELECT state FROM update_candidates"
                " WHERE new_source_id = ?",
                (r2.memory_id,),
            ).fetchone()
        assert row is not None and row[0] != "open"
    finally:
        mem.close()


# ---------------------------------------------------------------------
# guard matrix — each guard fires, pair stays advisory
# ---------------------------------------------------------------------


def test_wrong_policy_blocks(v5store):
    """Default advisory policy → no apply even for a perfect pair."""
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "Ticket INC-431: the deploy command is deploy-v1.")
    cands = _detect(v5store,
                    "Ticket INC-431: the deploy command is deploy-v2.")
    d = _decision(v5store,
                  "Ticket INC-431: the deploy command is deploy-v2.",
                  cands[0])
    assert d["blocking_guard"] == "policy"
    assert d["safe"] is False
    assert _apply(
        v5store, "Ticket INC-431: the deploy command is deploy-v2.",
        cands[0],
    ) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_hedged_new_blocks(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cand = _fabricated_cand()
    d = _decision(
        v5store, "Maybe INC-431 deploy command is deploy-v2.", cand
    )
    assert d["blocking_guard"] == "unhedged_new"
    assert _apply(
        v5store, "Maybe INC-431 deploy command is deploy-v2.", cand
    ) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_intent_modal_blocks(v5store):
    """'We should switch…' is a REAL 0.95 newer_value candidate with a
    shared anchor — the hedge guard refuses it (the type classifier also
    tags it procedure_hint, so same_type blocks alongside)."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cands = _detect(
        v5store, "We should switch INC-431 deploy command to deploy-v2."
    )
    assert cands and cands[0].relation == "newer_value"
    assert cands[0].score >= AUTO_SAFE_MIN  # would apply but for the hedge
    d = _decision(
        v5store, "We should switch INC-431 deploy command to deploy-v2.",
        cands[0],
    )
    assert d["guards"]["unhedged_new"] is False
    assert d["blocking_guard"] in ("same_type", "unhedged_new")
    assert _apply(
        v5store, "We should switch INC-431 deploy command to deploy-v2.",
        cands[0],
    ) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_confirm_that_blocks(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cand = _fabricated_cand()
    d = _decision(
        v5store,
        "Please confirm that INC-431 deploy command is deploy-v2.",
        cand,
    )
    assert d["blocking_guard"] == "unhedged_new"
    assert _apply(
        v5store,
        "Please confirm that INC-431 deploy command is deploy-v2.",
        cand,
    ) is None


def test_cross_entity_blocks(v5store):
    """Shared subject but anchors name different entities."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cands = _detect(v5store, "INC-999 deploy command is deploy-v2.")
    assert cands and cands[0].relation == "newer_value"
    d = _decision(v5store, "INC-999 deploy command is deploy-v2.",
                  cands[0])
    assert d["blocking_guard"] == "shared_anchor"
    assert _apply(v5store, "INC-999 deploy command is deploy-v2.",
                  cands[0]) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_no_anchor_blocks(v5store):
    """A value moved on a shared subject but nothing anchors the pair."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "The deploy command is deploy-v1.")
    cands = _detect(v5store, "The deploy command is deploy-v2.")
    assert cands and cands[0].relation == "newer_value"
    d = _decision(v5store, "The deploy command is deploy-v2.", cands[0])
    assert d["blocking_guard"] == "shared_anchor"
    assert _apply(v5store, "The deploy command is deploy-v2.",
                  cands[0]) is None


def test_contradicts_blocks(v5store):
    """A conflicting value scores high and shares an anchor — the
    relation guard is what keeps it advisory."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 favorite color is blue.")
    cands = _detect(v5store, "INC-431 favorite color is green.")
    assert cands and cands[0].relation == "contradicts"
    assert cands[0].score >= AUTO_SAFE_MIN
    d = _decision(v5store, "INC-431 favorite color is green.", cands[0])
    assert d["blocking_guard"] == "relation"
    assert _apply(v5store, "INC-431 favorite color is green.",
                  cands[0]) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_negates_blocks(v5store):
    """A negation is never safe-auto — fabricated candidate with
    matching type so the relation guard is the sole blocker."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cand = _fabricated_cand(relation="negates", score=0.95)
    d = _decision(
        v5store, "INC-431 deploy command is not deploy-v2.", cand
    )
    assert d["blocking_guard"] == "relation"
    assert _apply(
        v5store, "INC-431 deploy command is not deploy-v2.", cand
    ) is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_below_floor_blocks(v5store):
    """Real refines candidate under AUTO_SAFE_MIN — only the floor
    refuses it."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-55 retro meeting is Tuesday.")
    cands = _detect(v5store, "INC-55 retro meeting is Tuesday at 3pm.")
    assert cands and cands[0].relation == "refines"
    assert cands[0].score < AUTO_SAFE_MIN
    d = _decision(v5store, "INC-55 retro meeting is Tuesday at 3pm.",
                  cands[0])
    assert d["blocking_guard"] == "score_floor"
    assert _apply(v5store, "INC-55 retro meeting is Tuesday at 3pm.",
                  cands[0]) is None


def test_type_mismatch_blocks(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(
            conn, NS, "src-prior", "INC-431 deploy command is deploy-v1.",
            enrichment={"type": "decision", "polarity": "affirmative"},
        )
    cand = _fabricated_cand()
    d = _decision(v5store, "INC-431 deploy command is deploy-v2.", cand)
    assert d["blocking_guard"] == "same_type"


def test_pairing_mismatch_blocks(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cand = _fabricated_cand(new_sid="src-other")
    d = _decision(v5store, "INC-431 deploy command is deploy-v2.", cand)
    assert d["blocking_guard"] == "pairing"


def test_superseded_prior_not_live(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.",
                    disposition="superseded")
    cand = _fabricated_cand()
    d = _decision(v5store, "INC-431 deploy command is deploy-v2.", cand)
    assert d["blocking_guard"] == "prior_live"
    assert _apply(v5store, "INC-431 deploy command is deploy-v2.",
                  cand) is None


def test_stale_candidate_returns_none(v5store):
    """A candidate pinned to a drifted control version is unsafe —
    the CAS fence catches it and the pair stays advisory."""
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "INC-431 deploy command is deploy-v1.")
    cand = _fabricated_cand(prior_cv=99)  # stale pin
    # guards pass on resolved state; the apply fence then refuses.
    out = _apply(v5store, "INC-431 deploy command is deploy-v2.", cand)
    assert out is None
    assert _state(v5store, "src-prior")[3] == "active"


def test_resolved_candidate_never_reapplies(v5store):
    _policy(v5store)
    with v5store.tx() as conn:
        seed_source(conn, NS, "src-prior",
                    "Ticket INC-431: the deploy command is deploy-v1.")
    cands = _detect(v5store,
                    "Ticket INC-431: the deploy command is deploy-v2.")
    out = _apply(v5store,
                 "Ticket INC-431: the deploy command is deploy-v2.",
                 cands[0])
    assert out is not None
    # second application: the adopted row + superseded prior both refuse
    d = _decision(v5store,
                  "Ticket INC-431: the deploy command is deploy-v2.",
                  cands[0])
    assert d["blocking_guard"] in ("candidate_open", "prior_live")
    assert _apply(v5store,
                  "Ticket INC-431: the deploy command is deploy-v2.",
                  cands[0]) is None


def test_missing_candidate_fields_typed_error(v5store):
    _policy(v5store)
    rec = NewRecord(source_id="src-new", revision=1, text="x")
    with v5store.read() as conn:
        with pytest.raises(VerbatimError) as ei:
            evaluate_auto_safe(conn, namespace=NS, new_record=rec,
                               candidate={"relation": "newer_value"})
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# twin corpus shape + full-suite smoke
# ---------------------------------------------------------------------


def test_twins_corpus_shape():
    from eval.v6.twins import TWIN_PAIRS

    true_pairs = [p for p in TWIN_PAIRS if p["kind"] == "true"]
    false_pairs = [p for p in TWIN_PAIRS if p["kind"] == "false"]
    assert len(true_pairs) >= 4
    assert len(false_pairs) >= 8
    for p in TWIN_PAIRS:
        assert p["name"] and p["prior"] and p["new"]


def test_run_twins_passes(tmp_path):
    from eval.v6.twins import run_twins

    report = run_twins(str(tmp_path))
    assert report["should_apply"] >= 4
    assert report["false_total"] >= 8
    assert report["auto_applied"] == report["should_apply"]
    assert report["false_applied"] == 0
    assert 0.0 < report["upper95_bound"] <= 1.0
    assert report["verdict"] == "pass"
