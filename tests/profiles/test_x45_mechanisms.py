"""v4.5 X4/X5 governance anchors (SPEC_V4_5 §09; V45-09.04, V45-09.05;
acceptance D16, D17; parent binding V4-16.06, V4-21.08).

X4 (D16): two session peers never receive each other's private
perspective packs — a pack binds its caller, attenuates only, and never
widens authority. The behavioral floor already lives in
``test_perspective_never_widens`` /
``test_group_session_no_merge_no_leak``; these tests pin the M5
acceptance surface.

X5 (D17): topic-directed extraction steers retention but can NEVER
suppress deletion, consent-withdrawal, correction, or security events —
a narrow positive match spec bypasses nothing for safety-bearing claims,
an exclusion-shaped spec is refused at authoring time, and the compile
report names every safety event it saw.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.lifecycle import read_claim_head
from verbatim.profiles.service import safety_event_category

from .conftest import caller, seed_claim, topic


# ---------------------------------------------------------------------------
# X4 / D16 — session peers never share private packs
# ---------------------------------------------------------------------------


def test_d16_session_peers_no_shared_private_packs(store, svc, seeded):
    """D16/V45-09.04: alice's pack serves only alice — bob in the same
    session scope gets nothing from it, and his own pack exposes none of
    alice's private entries."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "ca", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "cb", "human:bob", "prefers",
                   "emacs editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)

    # Alice builds HER pack inside the shared session scope.
    alice_pack = svc.build_perspective_pack(
        caller(), sid, subjects=["human:alice"], topics=["editor"],
        verbs=["read", "quote"], purpose="recall",
    )
    candidates = [
        {"id": "ca", "scope_id": sid, "score": 1.0,
         "topics": ["editor"]},
        {"id": "cb", "scope_id": sid, "score": 1.0,
         "topics": ["editor"]},
    ]
    # Bob applies alice's pack: packs are caller-bound — nothing.
    res = svc.apply_perspective(
        caller("human:bob"), alice_pack, candidates
    )
    assert res["items"] == []
    assert res["reason"] == "pack_not_transferable"

    # Bob's own pack ranks only through HIS authority and exposes none
    # of alice's private (subject-only) profile entries.
    bob_pack = svc.build_perspective_pack(
        caller("human:bob"), sid, topics=["editor"], verbs=["read"],
        purpose="recall",
    )
    res_b = svc.apply_perspective(
        caller("human:bob"), bob_pack, candidates
    )
    assert res_b["authorized_verbs"] == ["read"]
    # Bob may see HIS OWN subject entries — but never alice's private
    # model through the shared session.
    assert all(
        h["subject_id"] != "human:alice"
        for h in res_b["profile_hints"]
    )
    # And reading alice's subject-private entries directly still denies.
    entries = svc.entries(caller("human:bob"), sid,
                          subject_id="human:alice")
    assert entries["entries"] == []
    assert entries["withheld"] >= 1


# ---------------------------------------------------------------------------
# X5 / D17 — topic policy cannot suppress safety events
# ---------------------------------------------------------------------------


def test_d17_topic_cannot_suppress_safety_events(store, svc, seeded):
    """D17/V45-09.05, V4-16.06: a topic matched only to 'editor' keywords
    cannot drop deletion/consent/correction/security claims — they pass
    the match gate and are named in the report."""
    sid = seeded["a"]
    with store.tx() as conn:
        # Ordinary on-topic claim.
        seed_claim(conn, store, sid, "c-ed", "human:alice", "prefers",
                   "vim editor")
        # Off-topic safety-bearing claims — the topic allowlist would
        # drop these without the X5 guard.
        seed_claim(conn, store, sid, "c-del", "human:alice",
                   "deleted", "the staging database record")
        seed_claim(conn, store, sid, "c-cons", "human:alice",
                   "consent_withdrawn", "analytics retention")
        seed_claim(conn, store, sid, "c-cor", "human:alice",
                   "corrects", "the earlier date claim")
        seed_claim(conn, store, sid, "c-sec", "human:alice",
                   "security_event", "injection attempt flagged")
        # A normal off-topic claim — stays filtered (control).
        seed_claim(conn, store, sid, "c-off", "human:alice",
                   "eats", "lunch at noon")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    report = svc.compile(caller(), sid, purpose="derive")

    # Every safety event is named — the narrow topic could not hide it.
    seen = {
        (e["claim_id"], e["category"])
        for e in report["safety_events"]
    }
    assert ("c-del", "deletion") in seen
    assert ("c-cons", "consent_withdrawal") in seen
    assert ("c-cor", "correction") in seen
    assert ("c-sec", "security") in seen
    # The off-topic benign claim produced no safety label…
    assert all(e["claim_id"] != "c-off" for e in report["safety_events"])
    # …and safety events were not filtered out of derivation either:
    # claims_matched counts every (claim, topic) contribution — the
    # on-topic claim plus all four safety events pass the match gate.
    assert report["claims_matched"] == 5


def test_d17_exclusion_shaped_match_rejected(store, svc, seeded):
    """An exclusion-shaped match spec is refused at authoring: only the
    three positive filter keys may persist — anything else is a silent
    no-op the owner would believe suppresses a category."""
    sid = seeded["a"]
    for bad_key in ("exclude_predicates", "not_keywords", "drop",
                    "except_subjects"):
        with pytest.raises(VerbatimError) as ei:
            svc.put_topic(
                caller(), sid,
                topic(match={"keywords": ["x"], bad_key: ["deleted"]}),
                purpose="admin",
            )
        assert ei.value.code is ErrorCode.VALIDATION
    # The positive-only spec still works.
    svc.put_topic(caller(), sid, topic(), purpose="admin")


def test_safety_event_category_classification(store, svc, seeded):
    """The classifier reads revision-level then claims-level predicates;
    non-safety predicates classify None."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "deleted",
                   "the record")
        head = read_claim_head(conn, "c1")
        assert safety_event_category(head) == "deletion"
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "tea")
        assert safety_event_category(read_claim_head(conn, "c2")) is None
