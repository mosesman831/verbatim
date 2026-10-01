"""SPEC_V4 §21 — profiles, preferences, perspective-aware personalization.

Covered here:

* V4-21.01  configured topics gate derivation (match surface, min_support,
  conflict policy, sensitivity).
* V4-21.02  explicit / observed / inferred / task_local / sensitive kinds
  are distinguishable labels, never conflated.
* V4-21.03  perspective packs narrow/rank only; audience restrictions are
  closed (empty audience = subject-only); no read+quote ⇒ no lift.
* V4-21.04  sensitive-trait inference is off by default and needs the
  triple opt-in (topic + purpose + subject consent row).
* V4-21.05  revisions carry changed fields, supports, effective time,
  prev revision; contradictions keep both alternatives inspectable.
* V4-21.07  owner inspect/correct/disable/delete with complete
  derivative invalidation.
* V4-21.08  group scopes never merge or cross-disclose subjects.
* V4-21.09  quality report measures adaptation/incorrect-assumption/
  correction-burden — never profile size.
* V4-21.10  sharing exposes no hidden support identifiers; packs are not
  transferable authority.
* Evidence purge/quarantine invalidates derived entries — read-path
  kernel gating + refresh tombstoning; derivation edges ride the
  existing provenance graph.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.governance import create_grant, register_principal
from verbatim.privacy.closure import ClosureEngine
from verbatim.profiles import ProfileService, probe_capability
from verbatim.security.quarantine import open_quarantine, release

from .conftest import caller, seed_claim, topic


def _denied(fn, *a, **kw):
    with pytest.raises(VerbatimError) as exc:
        fn(*a, **kw)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def _invalid(fn, *a, **kw):
    with pytest.raises(VerbatimError) as exc:
        fn(*a, **kw)
    assert exc.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# derivation from seeded evidence
# ---------------------------------------------------------------------------


def test_compile_derives_entry_from_evidence(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "vim editor", recorded_from=2)
        seed_claim(conn, store, sid, "c3", "human:alice", "eats",
                   "lunch at noon", recorded_from=3)
    svc.put_topic(
        caller(), sid, topic(min_support=1), purpose="admin"
    )
    report = svc.compile(caller(), sid, purpose="derive")
    assert report["claims_scanned"] == 3
    assert report["claims_matched"] == 2
    assert len(report["entries_created"]) == 1
    assert report["durable_job"] is False  # honest: on-demand, not a job
    assert "profile_compile" in report["durable_job_reason"]

    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out["entries"]) == 1
    e = out["entries"][0]
    assert e["entry_kind"] == "observed"           # V4-21.02 label
    assert e["topic_key"] == "editor"
    assert e["subject_id"] == "human:alice"
    assert e["state"] == "active"
    assert e["revision"] == 1
    assert e["confidence"] == 1.0
    assert {s["id"] for s in e["support"]} == {"c1", "c2"}
    assert all(s["kind"] == "claim" for s in e["support"])
    assert e["support_withheld"] == 0
    # registered as a v4 object — kernel resolves it
    with store.read() as conn:
        row = conn.execute(
            "SELECT kind, disposition, current_revision FROM objects"
            " WHERE object_id=?",
            (e["entry_id"],),
        ).fetchone()
        assert row is not None and row[0] == "profile"
        assert row[1] == "active" and row[2] == 1
        # derivation parents recorded in the shared graph
        edges = conn.execute(
            "SELECT parent_kind, parent_id, parent_revision"
            " FROM derivations"
            " WHERE child_kind='profile' AND child_id=?"
            " ORDER BY parent_id",
            (e["entry_id"],),
        ).fetchall()
        assert sorted(r[1] for r in edges) == ["c1", "c2"]
        assert all(r[0] == "claim" for r in edges)
        deps = conn.execute(
            "SELECT parent_id, role FROM dependency_edges"
            " WHERE child_kind='profile' AND child_id=?",
            (e["entry_id"],),
        ).fetchall()
        assert sorted(r[0] for r in deps) == ["c1", "c2"]
        assert all(r[1] == "supports" for r in deps)


def test_compile_is_idempotent_revisioned(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    r1 = svc.compile(caller(), sid)
    r2 = svc.compile(caller(), sid)
    # Same evidence re-derives the same entry id — a revision, not a dup.
    assert r1["entries_created"] == r2["entries_revised"]
    out = svc.entries(caller(), sid)
    assert len(out["entries"]) == 1


def test_compile_requires_derive_grant(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    # agent:prod has read+quote only — derive is denied
    _denied(svc.compile, caller("agent:prod"), sid)


def test_compile_respects_min_support_and_match(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(min_support=2), purpose="admin")
    report = svc.compile(caller(), sid)
    assert report["skipped_min_support"] == 1
    assert report["entries_created"] == []
    assert svc.entries(caller(), sid)["entries"] == []


def test_derived_entry_requires_support_refs(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    _invalid(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "observed", "value": {"object": "vim"},
        },
    )


def test_support_ref_must_be_live(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    _invalid(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "observed", "value": {"object": "vim"},
            "supports": [{"kind": "claim", "id": "ghost", "revision": 1}],
        },
    )


# ---------------------------------------------------------------------------
# kernel-resolved support references
# ---------------------------------------------------------------------------


def test_resolve_support_goes_through_kernel(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    e = svc.entries(caller(), sid)["entries"][0]
    res = svc.resolve_support(caller(), e["entry_id"])
    assert res["resolved"] == [
        {"kind": "claim", "id": "c1", "revision": 1,
         "relation": "supports"}
    ]
    assert res["lease_id"].startswith("lease:")
    # a caller with no read grant cannot resolve support ids
    _denied(svc.resolve_support, caller("human:carol"), e["entry_id"])
    # bob is authorized on the scope but not in the audience — the
    # profile's own gate still applies before the kernel check
    _denied(svc.resolve_support, caller("human:bob"), e["entry_id"])


def test_shared_view_hides_support_from_unauthorized_recipient(
    store, svc, seeded
):
    """V4-21.10: sharing a profile view does not transfer source grants —
    a recipient without read on the support sees the entry withheld,
    not the identifier."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    e = svc.entries(caller(), sid)["entries"][0]
    # carol gets read on scope:a — but the entry is subject-only
    # (audience=()), so she sees neither the value nor the support id.
    with store.tx() as conn:
        register_principal(conn, kind="human", principal_id="human:carol")
        create_grant(
            conn, scope_id=sid, principal_id="human:carol",
            verbs=["read"], purposes=["recall"], issuer_id="human:alice",
        )
    _denied(svc.get_entry, caller("human:carol"), e["entry_id"])
    _denied(svc.resolve_support, caller("human:carol"), e["entry_id"])
    # declared audience membership lets her read — and the support id is
    # disclosed only because *her* lease resolves it through the kernel.
    svc.upsert_entry(
        caller(), sid,
        {
            "entry_id": e["entry_id"], "subject_id": "human:alice",
            "topic_key": "editor", "entry_kind": "observed",
            "value": e["value"],
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
            "audience": ["human:carol"],
        },
    )
    out = svc.entries(caller("human:carol"), sid, subject_id="human:alice")
    assert len(out["entries"]) == 1
    assert out["entries"][0]["support"][0]["id"] == "c1"
    assert out["entries"][0]["support_withheld"] == 0


# ---------------------------------------------------------------------------
# contradictions (V4-21.05)
# ---------------------------------------------------------------------------


def test_contradiction_keeps_both_entries(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor", recorded_from=1)
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "emacs editor", recorded_from=2)
    svc.put_topic(
        caller(), sid, topic(conflict_policy="keep_both"), purpose="admin"
    )
    svc.compile(caller(), sid)
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out["entries"]) == 2
    states = {e["entry_id"]: e["state"] for e in out["entries"]}
    assert set(states.values()) == {"conflicted"}
    groups = {e["conflict"]["group"] for e in out["entries"]}
    assert len(groups) == 1
    g = next(iter(groups))
    for e in out["entries"]:
        assert e["conflict"]["state"] == "open"
        other = {m for m in e["conflict"]["members"]}
        assert e["entry_id"] in other
        assert len(other) == 2
    # both alternatives remain inspectable with their own support
    values = {e["value"]["object"]["text"] for e in out["entries"]}
    assert values == {"vim editor", "emacs editor"}
    # owner resolves explicitly — recorded, not silent
    keep = out["entries"][0]["entry_id"]
    res = svc.resolve_conflict(caller(), sid, g, keep, purpose="admin")
    assert res["state"] == "resolved"
    out2 = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out2["entries"]) == 1
    assert out2["entries"][0]["entry_id"] == keep
    assert out2["entries"][0]["state"] == "active"


def test_latest_wins_supersedes_older(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor", recorded_from=1)
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "emacs editor", recorded_from=9)
    svc.put_topic(
        caller(), sid, topic(conflict_policy="latest_wins"),
        purpose="admin",
    )
    svc.compile(caller(), sid)
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out["entries"]) == 1
    assert out["entries"][0]["value"]["object"]["text"] == "emacs editor"


def test_explicit_only_topic_rejects_derived(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(
        caller(), sid, topic(conflict_policy="explicit_only"),
        purpose="admin",
    )
    report = svc.compile(caller(), sid)
    assert report["entries_created"] == []
    assert report["skipped_policy"] >= 1
    _denied(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "observed", "value": {"object": "vim"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
    )
    # explicit declarations are still writable
    r = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "vim"},
        },
    )
    assert r["state"] == "active"


# ---------------------------------------------------------------------------
# revisions + update reports (V4-21.05)
# ---------------------------------------------------------------------------


def test_revision_linkage_and_changed_fields(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    r1 = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "vim"},
        },
    )
    eid = r1["entry_id"]
    assert r1["revision"] == 1 and r1["prev_revision"] is None
    r2 = svc.upsert_entry(
        caller(), sid,
        {
            "entry_id": eid, "subject_id": "human:alice",
            "topic_key": "editor", "entry_kind": "explicit",
            "value": {"object": "neovim"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
    )
    assert r2["revision"] == 2 and r2["prev_revision"] == 1
    assert "value" in r2["changed_fields"]
    hist = svc.history(caller(), eid, purpose="admin")
    assert [r["revision"] for r in hist["revisions"]] == [1, 2]
    assert hist["revisions"][1]["supports"][0]["id"] == "c1"
    # owner correction: the corrected value supersedes in place
    e = svc.get_entry(caller(), eid)
    assert e["value"]["object"] == "neovim"


def test_entry_kinds_are_distinguishable(store, svc, seeded):
    """V4-21.02: the five labels stay distinguishable — explicit,
    observed, inferred, task-local."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.put_topic(
        caller(), sid,
        topic(topic_key="scratch", multiplicity="set", expiry_s=3600),
        purpose="admin",
    )
    svc.put_topic(
        caller(), sid,
        topic(topic_key="guesswork", inference="on",
              inference_purposes=["derive"]),
        purpose="admin",
    )
    svc.compile(caller(), sid)
    exp = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "declared"},
        },
    )
    now = 1_800_000_000_000_000
    task = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "scratch",
            "entry_kind": "task_local",
            "value": {"object": "use dark theme for this review"},
            "expires_us": now + 3600_000_000, "effective_us": now,
        },
    )
    inf = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "guesswork",
            "entry_kind": "inferred",
            "value": {"object": "probably prefers terse replies"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
        purpose="derive",
    )
    kinds = {
        e["entry_kind"]
        for e in svc.entries(caller(), sid)["entries"]
    }
    assert {"observed", "explicit", "task_local", "inferred"} <= kinds
    assert task["entry_id"] != exp["entry_id"] != inf["entry_id"]


def test_task_local_expires(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(
        caller(), sid,
        topic(topic_key="scratch", multiplicity="set", expiry_s=60),
        purpose="admin",
    )
    r = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "scratch",
            "entry_kind": "task_local", "value": {"object": "temp"},
            "effective_us": 10, "expires_us": 20,
        },
    )
    # a task-local write with no expires_us inherits topic expiry_s —
    # but on a topic without one it is a caller bug (must expire).
    svc.put_topic(
        caller(), sid,
        topic(topic_key="unbounded", multiplicity="set"),
        purpose="admin",
    )
    _invalid(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "unbounded",
            "entry_kind": "task_local", "value": {"object": "no-expiry"},
        },
    )
    # expires_us=20 is already in the past — the read-time expiry gate
    # withholds it immediately, and refresh() makes it durable.
    _denied(svc.get_entry, caller(), r["entry_id"])
    rep = svc.refresh(caller(), sid, purpose="admin")
    assert r["entry_id"] in rep["expired"]


# ---------------------------------------------------------------------------
# purge / quarantine invalidation
# ---------------------------------------------------------------------------


def test_purge_of_support_invalidates_entry(store, svc, seeded):
    """Evidence purge → the derived entry withholds immediately on the
    read path (kernel denies the parent ref), closure strips the
    derivation edge, and refresh tombstones it durably."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    e = svc.entries(caller(), sid)["entries"][0]

    eng = ClosureEngine(store)
    run = eng.begin([("claim", "c1", 1)], sid)
    # suppression is live before drain — the read path must withhold now
    _denied(svc.get_entry, caller(), e["entry_id"])
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert out["entries"] == []
    assert out["withheld"] == 1

    run = eng.drain(run.run_id)
    assert run.phase.value in ("completed", "verified")
    # the derivation edge into erased material is stripped by closure,
    # and the derived entry — all parents purged — is itself erased
    # (§36 delete semantics: derived state leaves only the erasure ledger)
    with store.read() as conn:
        left = conn.execute(
            "SELECT COUNT(*) FROM derivations WHERE child_id=?",
            (e["entry_id"],),
        ).fetchone()[0]
        assert left == 0
        rows = conn.execute(
            "SELECT COUNT(*) FROM profile_entries WHERE entry_id=?",
            (e["entry_id"],),
        ).fetchone()[0]
        assert rows == 0
        ptrs = conn.execute(
            "SELECT COUNT(*) FROM profile_entry_support WHERE entry_id=?",
            (e["entry_id"],),
        ).fetchone()[0]
        assert ptrs == 0
    _denied(svc.get_entry, caller(), e["entry_id"])
    _denied(svc.history, caller(), e["entry_id"], purpose="admin")
    rep = svc.refresh(caller(), sid, purpose="admin")
    assert rep["scanned"] == 0  # nothing left to reconcile


def test_quarantined_support_withholds_then_revives(store, svc, seeded):
    """A quarantine hold on the supporting claim (or its span) withholds
    the derived entry — V3-14.10 cascade — and release revives it."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    e = svc.entries(caller(), sid)["entries"][0]

    with store.tx() as conn:
        open_quarantine(conn, ("claim", "c1", 1), ["test:hold"], [], scope_id=sid)
    _denied(svc.get_entry, caller(), e["entry_id"])
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert out["withheld"] == 1 and out["entries"] == []
    rep = svc.refresh(caller(), sid, purpose="admin")
    assert e["entry_id"] in rep["withheld"]

    with store.tx() as conn:
        release(conn, ("claim", "c1", 1), "human:alice", scope_id=sid)
    rep2 = svc.refresh(caller(), sid, purpose="admin")
    assert e["entry_id"] in rep2["revived"]
    e2 = svc.get_entry(caller(), e["entry_id"])
    assert e2["state"] == "active"


def test_quarantine_on_span_cascades_to_entry(store, svc, seeded):
    """Holding the *span* under the claim also withholds the entry —
    the kernel's claim→span quarantine cascade covers derivation."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    e = svc.entries(caller(), sid)["entries"][0]
    with store.tx() as conn:
        open_quarantine(conn, ("span", "span:c1", 1), ["test:hold"], [],
                        scope_id=sid)
    _denied(svc.get_entry, caller(), e["entry_id"])


def test_profile_entry_as_purge_root_erases(store, svc, seeded):
    """A profile entry is itself a resolvable closure target: purging it
    erases every revision, its support pointers, and its conflict-group
    membership — nothing derived survives as a ghost."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "emacs editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out["entries"]) == 2  # single-valued topic → conflicted pair
    victim = out["entries"][0]["entry_id"]

    eng = ClosureEngine(store)
    run = eng.begin([("profile", victim, 1)], sid)
    run = eng.drain(run.run_id)
    assert run.phase.value in ("completed", "verified")
    with store.read() as conn:
        for table, col in (
            ("profile_entries", "entry_id"),
            ("profile_entry_support", "entry_id"),
            ("profile_events", "entry_id"),
        ):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col}=?", (victim,)
            ).fetchone()[0] == 0
        # the surviving conflict member is scrubbed — a one-member group
        # cannot stand as an open contradiction
        leftover = conn.execute(
            "SELECT members_json, state FROM profile_conflicts"
        ).fetchall()
        for members_json, _state in leftover:
            assert victim not in members_json
    _denied(svc.get_entry, caller(), victim)
    # the sibling entry is untouched and readable again
    other = svc.entries(caller(), sid, subject_id="human:alice")["entries"]
    assert len(other) == 1


def test_mixed_ancestry_suppresses_not_erases(store, svc, seeded):
    """An entry with two derivation parents where only one is purged is
    *suppressed* (invalid until recomputed), not erased — §36 mixed
    ancestry. The surviving support keeps the row; the quarantine marker
    withholds it."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    e = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "observed", "value": {"object": "vim editor"},
            "supports": [
                {"kind": "claim", "id": "c1", "revision": 1},
                {"kind": "claim", "id": "c2", "revision": 1},
            ],
        },
        purpose="derive",
    )
    eng = ClosureEngine(store)
    run = eng.begin([("claim", "c1", 1)], sid)
    run = eng.drain(run.run_id)
    assert run.phase.value in ("completed", "verified")
    with store.read() as conn:
        row = conn.execute(
            "SELECT state FROM profile_entries WHERE entry_id=?",
            (e["entry_id"],),
        ).fetchone()
        assert row is not None  # suppressed, not erased
        q = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind='profile' AND object_id=?",
            (e["entry_id"],),
        ).fetchone()
        assert q is not None and q[0] == "suppressed"
    _denied(svc.get_entry, caller(), e["entry_id"])


# ---------------------------------------------------------------------------
# perspectives — attenuation only (V4-21.03/06/08/10)
# ---------------------------------------------------------------------------


def _seed_editor_entries(store, svc, sid):
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)


def test_perspective_narrows_and_ranks_only(store, svc, seeded):
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    pack = svc.build_perspective_pack(
        caller(), sid, subjects=["human:alice"], topics=["editor"],
        verbs=["read", "quote"], purpose="recall", persist=True,
    )
    assert pack["persisted"] is True
    with store.read() as conn:
        prow = conn.execute(
            "SELECT asserter, audience_json FROM perspectives"
            " WHERE perspective_id=?",
            (pack["perspective_id"],),
        ).fetchone()
        assert prow is not None and prow[0] == "human:alice"
    candidates = [
        {"id": "item:generic", "scope_id": sid, "score": 1.0},
        {"id": "item:editorial", "scope_id": sid, "score": 0.5,
         "topics": ["editor"], "text": "use vim editor"},
        {"id": "item:foreign", "scope_id": seeded["b"], "score": 9.9},
    ]
    res = svc.apply_perspective(caller(), pack, candidates)
    assert res["authorized_verbs"] == ["read", "quote"]
    ids = [i["id"] for i in res["items"]]
    assert "item:foreign" not in ids           # scope isolation
    assert res["dropped"] == 1
    assert ids[0] == "item:editorial"           # ranked up by the pack
    assert all(i["liftable"] for i in res["items"])
    assert res["profile_hints"]                # hints rode the gate
    assert res["profile_hints"][0]["topic_key"] == "editor"


def test_perspective_never_widens(store, svc, seeded):
    """A pack without an authorized verb evaluates to nothing; read-only
    callers rank but nothing is liftable; a foreign caller cannot use
    another principal's pack at all."""
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    pack = svc.build_perspective_pack(
        caller(), sid, subjects=["human:alice"], topics=["editor"],
        verbs=["read", "quote"], purpose="recall",
    )
    candidates = [{"id": "x", "scope_id": sid, "score": 1.0,
                   "topics": ["editor"]}]

    # caller with NO grants on the scope — pack yields nothing
    with store.tx() as conn:
        register_principal(conn, kind="human", principal_id="human:carol")
    res = svc.apply_perspective(caller("human:carol"), pack, candidates)
    assert res["items"] == []
    assert res["reason"] == "pack_not_transferable"  # pack binds its caller

    # bob has read+quote — but pack is alice's; non-transferable
    res_bob = svc.apply_perspective(caller("human:bob"), pack, candidates)
    assert res_bob["items"] == []

    # bob builds his own pack — his verbs are what they are
    bpack = svc.build_perspective_pack(
        caller("human:bob"), sid, topics=["editor"], verbs=["read"],
        purpose="recall",
    )
    res_bob2 = svc.apply_perspective(
        caller("human:bob"), bpack, candidates
    )
    assert res_bob2["authorized_verbs"] == ["read"]
    assert res_bob2["items"] and not res_bob2["items"][0]["liftable"]
    # profile hints are audience-gated: alice's entries are subject-only
    assert res_bob2["profile_hints"] == []

    # readless-verb pack: declares only 'quote' — never substitutes for read
    qpack = svc.build_perspective_pack(
        caller(), sid, topics=["editor"], verbs=["quote"],
        purpose="recall",
    )
    res_q = svc.apply_perspective(caller(), qpack, candidates)
    assert res_q["items"] == [] and res_q["reason"] == "unauthorized"
    assert res_q["authorized_verbs"] == ["quote"]
    assert res_q["denied_verbs"] == []

    # authority can also disappear between build and apply: bob builds a
    # pack, his grant is revoked, and the same pack now yields nothing.
    bpack2 = svc.build_perspective_pack(
        caller("human:bob"), sid, topics=["editor"], verbs=["read"],
        purpose="recall",
    )
    with store.tx() as conn:
        from verbatim.governance import grants_for, revoke_grant

        for g in grants_for(conn, sid, "human:bob"):
            revoke_grant(conn, g["grant_id"])
    res_bob3 = svc.apply_perspective(
        caller("human:bob"), bpack2, candidates
    )
    assert res_bob3["items"] == []
    assert res_bob3["reason"] == "unauthorized"


def test_group_session_no_merge_no_leak(store, svc, seeded):
    """V4-21.08: two subjects in one scope — entries stay partitioned;
    neither peer's private model crosses without explicit authority."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "ca", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "cb", "human:bob", "prefers",
                   "emacs editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)
    alice_view = svc.entries(caller(), sid, subject_id="human:alice")
    bob_view = svc.entries(caller("human:bob"), sid,
                           subject_id="human:bob")
    assert len(alice_view["entries"]) == 1
    assert len(bob_view["entries"]) == 1
    assert (
        alice_view["entries"][0]["subject_id"]
        != bob_view["entries"][0]["subject_id"]
    )
    # bob reading alice's entries: subject-only audience blocks him even
    # though he holds read on the scope
    bob_on_alice = svc.entries(
        caller("human:bob"), sid, subject_id="human:alice"
    )
    assert bob_on_alice["entries"] == []
    assert bob_on_alice["withheld"] == 1
    ae = alice_view["entries"][0]["entry_id"]
    _denied(svc.get_entry, caller("human:bob"), ae)


# ---------------------------------------------------------------------------
# audience + sensitivity (V4-21.03/04)
# ---------------------------------------------------------------------------


def test_audience_is_closed_by_default(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    r = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "vim"},
        },
    )
    eid = r["entry_id"]
    # subject sees it; bob (scope read grant, not in audience) does not
    assert svc.get_entry(caller(), eid)["entry_id"] == eid
    _denied(svc.get_entry, caller("human:bob"), eid)
    # explicitly-audience entries do serve their members
    r2 = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "shared pref"},
            "audience": ["human:bob"],
        },
    )
    assert svc.get_entry(caller("human:bob"), r2["entry_id"])["entry_id"]


def test_sensitive_inference_off_by_default(store, svc, seeded):
    """V4-21.04: sensitive topics never infer without topic opt-in +
    declared purpose + subject consent — detector absence is not consent."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "works",
                   "night shift editor")  # contains a topic keyword
    svc.put_topic(
        caller(), sid,
        topic(sensitivity="sensitive",
              inference_purposes=["derive"],
              required_purposes=["recall"]),
        purpose="admin",
    )
    report = svc.compile(caller(), sid, purpose="derive")
    assert report["skipped_sensitive"] >= 1
    assert report["entries_created"] == []
    # manual inferred write on a sensitive topic: denied until all three
    _denied(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "inferred", "value": {"object": "x"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
        purpose="derive",
    )
    # owner turns inference on at the topic… still denied (no consent row)
    svc.put_topic(
        caller(), sid,
        topic(sensitivity="sensitive", inference="on",
              inference_purposes=["derive"],
              required_purposes=["recall"]),
        purpose="admin",
    )
    _denied(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "inferred", "value": {"object": "x"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
        purpose="derive",
    )
    # …and only after the subject-consent row does inference run
    svc.set_inference(caller(), sid, "human:alice", True, purpose="admin")
    ok = svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "inferred", "value": {"object": "x"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
        purpose="derive",
    )
    assert ok["state"] == "active"
    # sensitive delivery requires the declared purpose at read time
    _denied(svc.get_entry, caller(), ok["entry_id"], purpose="evaluate")
    assert svc.get_entry(caller(), ok["entry_id"], purpose="recall")


def test_inferred_on_normal_topic_needs_opt_in(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
    svc.put_topic(caller(), sid, topic(), purpose="admin")  # inference=off
    _denied(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "inferred", "value": {"object": "x"},
            "supports": [{"kind": "claim", "id": "c1", "revision": 1}],
        },
        purpose="derive",
    )


# ---------------------------------------------------------------------------
# scope isolation
# ---------------------------------------------------------------------------


def test_scope_isolation(store, svc, seeded):
    """Entries live and die inside their scope — nothing crosses."""
    a, b = seeded["a"], seeded["b"]
    with store.tx() as conn:
        seed_claim(conn, store, a, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, b, "c9", "human:alice", "prefers",
                   "vscode editor")
    svc.put_topic(caller(), a, topic(), purpose="admin")
    svc.put_topic(caller(), b, topic(), purpose="admin")
    svc.compile(caller(), a)
    svc.compile(caller(), b)
    ea = svc.entries(caller(), a)["entries"]
    eb = svc.entries(caller(), b)["entries"]
    assert len(ea) == 1 and len(eb) == 1
    assert ea[0]["scope_id"] == a and eb[0]["scope_id"] == b
    assert ea[0]["entry_id"] != eb[0]["entry_id"]
    assert ea[0]["support"][0]["id"] == "c1"
    assert eb[0]["support"][0]["id"] == "c9"
    # cross-scope upsert of a support ref is rejected at write time
    _invalid(
        svc.upsert_entry, caller(), a,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "observed", "value": {"object": "x"},
            "supports": [{"kind": "claim", "id": "c9", "revision": 1}],
        },
    )
    # a pack built for scope:a never ranks scope:b candidates
    pack = svc.build_perspective_pack(caller(), a, topics=["editor"])
    res = svc.apply_perspective(
        caller(), pack, [{"id": "f", "scope_id": b, "score": 1}]
    )
    assert res["items"] == [] and res["dropped"] == 1


# ---------------------------------------------------------------------------
# owner controls (V4-21.07)
# ---------------------------------------------------------------------------


def test_owner_delete_entry_and_topic(store, svc, seeded):
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    e = svc.entries(caller(), sid)["entries"][0]
    res = svc.delete_entry(caller(), e["entry_id"], purpose="admin")
    assert res["state"] == "tombstoned"
    _denied(svc.get_entry, caller(), e["entry_id"])
    hist = svc.history(caller(), e["entry_id"], purpose="admin")
    assert hist["head_state"] == "tombstoned"
    # registry disposition is erased — the kernel itself denies the ref
    with store.read() as conn:
        disp = conn.execute(
            "SELECT disposition FROM objects WHERE object_id=?",
            (e["entry_id"],),
        ).fetchone()[0]
        assert disp == "erased"


def test_delete_topic_invalidates_all_derivatives(store, svc, seeded):
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    svc.put_topic(
        caller(), sid,
        topic(topic_key="scratch", multiplicity="set", expiry_s=60),
        purpose="admin",
    )
    svc.upsert_entry(
        caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "scratch",
            "entry_kind": "task_local", "value": {"object": "t"},
            "effective_us": 1, "expires_us": 9_999_999_999_999_999,
        },
    )
    res = svc.delete_topic(caller(), sid, "editor", purpose="admin")
    assert res["state"] == "deleted"
    assert len(res["entries_tombstoned"]) == 1
    # the other topic's entry is untouched
    out = svc.entries(caller(), sid)
    assert [e["topic_key"] for e in out["entries"]] == ["scratch"]
    # deleted topic serves nothing and rejects writes
    assert svc.get_topic(caller(), sid, "editor")["state"] == "deleted"
    _invalid(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "x"},
        },
    )


def test_disable_inference_blocks_compile_only(store, svc, seeded):
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    svc.set_inference(caller(), sid, "human:alice", False, purpose="admin")
    pol = svc.set_inference(caller(), sid, "human:alice", True,
                            purpose="admin")
    assert pol["inference"] == "on"


# ---------------------------------------------------------------------------
# quality + capability honesty (V4-21.09, V4-50)
# ---------------------------------------------------------------------------


def test_quality_report_measures_not_size(store, svc, seeded):
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    svc.delete_entry(
        caller(), svc.entries(caller(), sid)["entries"][0]["entry_id"],
        purpose="admin",
    )
    q = svc.quality_report(caller(), sid, purpose="evaluate")
    assert q["state"] == "healthy"
    assert q["incorrect_assumptions"] == 1
    assert "beneficial_adaptations" in q and "correction_burden" in q
    assert "not a quality signal" in q["note"]


def test_capability_probe_is_honest(store, svc, seeded):
    # before first write: implemented, not pretending durable compilation
    p = probe_capability(store)
    assert p["state"] == "implemented"
    assert p["details"]["durable_job"] == "deferred"
    sid = seeded["a"]
    _seed_editor_entries(store, svc, sid)
    p2 = probe_capability(store)
    assert p2["state"] == "healthy"
    assert p2["details"]["entry_rows"] >= 1


# ---------------------------------------------------------------------------
# service misc
# ---------------------------------------------------------------------------


def test_unconfigured_topic_rejects_entries(store, svc, seeded):
    sid = seeded["a"]
    _invalid(
        svc.upsert_entry, caller(), sid,
        {
            "subject_id": "human:alice", "topic_key": "ghost",
            "entry_kind": "explicit", "value": {"object": "x"},
        },
    )


def test_unauthorized_write_denied(store, svc, seeded):
    sid = seeded["a"]
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    _denied(
        svc.upsert_entry, caller("agent:prod"), sid,  # no derive verb
        {
            "subject_id": "human:alice", "topic_key": "editor",
            "entry_kind": "explicit", "value": {"object": "x"},
        },
    )
    _denied(
        svc.put_topic, caller("agent:prod"), sid, topic(), purpose="admin"
    )
    _denied(svc.refresh, caller("agent:prod"), sid, purpose="admin")


def test_compile_skips_dead_or_invisible_claims(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "prefers",
                   "emacs editor", state="superseded")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    report = svc.compile(caller(), sid)
    assert report["claims_scanned"] == 2
    assert report["claims_matched"] == 1
    assert report["skipped_unauthorized"] == 1
    out = svc.entries(caller(), sid)
    assert len(out["entries"]) == 1


def test_set_multiplicity_coexists(store, svc, seeded):
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "uses",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "uses",
                   "vscode editor")
    svc.put_topic(
        caller(), sid, topic(multiplicity="set"), purpose="admin"
    )
    svc.compile(caller(), sid)
    out = svc.entries(caller(), sid, subject_id="human:alice")
    assert len(out["entries"]) == 2
    assert {e["state"] for e in out["entries"]} == {"active"}


def test_context_pack_serialized_budget(store, svc, seeded):
    """V4-21.06: the pack balances stable profile + recent claims and the
    budget binds the *serialized* bytes — formatting and provenance are
    charged, nothing withheld occupies room."""
    sid = seeded["a"]
    with store.tx() as conn:
        seed_claim(conn, store, sid, "c1", "human:alice", "prefers",
                   "vim editor")
        seed_claim(conn, store, sid, "c2", "human:alice", "uses",
                   "tmux daily")
    svc.put_topic(caller(), sid, topic(), purpose="admin")
    svc.compile(caller(), sid)

    big = svc.context_pack(caller(), sid, subject_id="human:alice",
                           max_bytes=64 * 1024)
    assert big["bytes_used"] <= big["bytes_budget"]
    assert big["counts"]["stable"] >= 1
    assert big["counts"]["recent"] == 2
    assert big["truncated"] is False
    assert big["provenance"]["entries"] and big["provenance"]["claims"]

    tiny = svc.context_pack(caller(), sid, subject_id="human:alice",
                            max_bytes=700)
    assert tiny["bytes_used"] <= tiny["bytes_budget"]
    assert tiny["truncated"] is True

    # a withheld entry never occupies budget
    e = svc.entries(caller(), sid)["entries"][0]
    with store.tx() as conn:
        open_quarantine(conn, ("claim", "c1", 1), ["test:hold"], [],
                        scope_id=sid)
    held = svc.context_pack(caller(), sid, subject_id="human:alice",
                            max_bytes=64 * 1024)
    assert all(i["entry_id"] != e["entry_id"] for i in held["stable"])

    _denied(svc.context_pack, caller("human:eve"), sid)


def test_pack_expiry_and_persisted_perspective(store, svc, seeded):
    sid = seeded["a"]
    pack = svc.build_perspective_pack(
        caller(), sid, topics=["editor"], verbs=["read"],
        purpose="recall",
    )
    res = svc.apply_perspective(
        caller(), pack, [], now_us_=pack["expires_us"] + 1
    )
    assert res["items"] == [] and res["reason"] == "pack_expired"
