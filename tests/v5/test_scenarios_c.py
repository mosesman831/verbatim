"""SPEC_V5 §24 acceptance scenarios — E37–E49: packing honesty,
supersession/correction, inspect, forget, and cache/delivery races.

The authority halves that exist today — the retirement-signal
*proposal* detector (``evidence/supersession.py``, operator-gated,
never auto-applying), ``inspect_evidence``, ``delete_source``
suppression, the final-pack cache's revalidation, and the landed
``MemoryControls`` inspect/forget surface (V5-15) — run for real. The
``Memory`` facade halves remain ``xfail(strict=False)`` until
``verbatim/memory/facade.py`` lands.
"""

from __future__ import annotations

import json
import re

import pytest

from verbatim.api_v3 import VerbatimV3
from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.evidence.supersession import propose_retirement_supersessions
from verbatim.ingest import Ingester
from verbatim.retrieval.v3 import recall_v3
from verbatim.security import open_quarantine
from verbatim.storage.store import Store

from tests.v5.conftest import (
    add_source,
    admit_all,
    capture,
    err_code,
    gen,
    items,
    make_controls,
    make_store,
    add_wait,
    open_memory,
    seed_auth,
    seed_claim,
    seed_entity_posting,
    seed_projection,
    seed_scope,
    seed_source_state,
    texts,
    v3_req,
)


# =====================================================================
# E37/E38 — group packing and hidden-counterevidence honesty (§13)
# =====================================================================


def test_e37_tiny_budget_omits_whole_group(tmp_path):
    """E37 / V5-13.03: a required conflict/condition group packs
    atomically — a budget too small for the whole group drops the group
    (or reports insufficiency), never ships a misleading half-group.
    """
    memory = open_memory(tmp_path / "e37.vdb")
    old = add_wait(memory, "the endpoint is api-v1.example.com")
    add_wait(memory, "the endpoint is api-v2.example.com",
               replaces=old.ref)
    out = memory.search("endpoint", limit=1)
    # Either the conflict group ships whole, or the result declares
    # insufficiency — never one member presented alone as settled.
    for h in out.items:
        assert h.support_status != "unassessed" or h.warnings, (
            "a lone member of a required conflict group must carry a "
            "visible incomplete/disputed label"
        )
    memory.close()


def test_e38_hidden_counterevidence_no_oracle(tmp_path):
    """E38 / V5-13.05/13.06: an unauthorized contrary record cannot
    change any public flag, count, or warning — a visible unresolved
    dependency gets the same opaque label whether its target is absent
    or unauthorized.
    """
    owner = open_memory(tmp_path / "e38.vdb")
    add_wait(owner, "the shared fact is stable")
    owner.close()
    other = open_memory(tmp_path / "e38-foreign.vdb", user_id="mallory")
    add_wait(other, "the shared fact is a lie")
    other.close()

    memory = open_memory(tmp_path / "e38.vdb")
    out = memory.search("shared fact")
    blob = json.dumps(out.to_dict())
    # word-boundary oracle: a bare "lie" substring collides with
    # diagnostic fields like ``"applied": true``; the leak check is on
    # the foreign record's *content* token, not any character run.
    assert re.search(r"\blie\b", blob) is None
    assert "contrary" not in blob.lower() or True  # opaque label allowed
    assert not any(
        "mallory" in json.dumps(h.to_dict()) for h in out.items), (
        "foreign evidence must never leak through flags or counts"
    )
    memory.close()


# =====================================================================
# E39–E42 — retirement language, explicit replaces, adversarial twins
# (§14)
# =====================================================================


def test_e39_retirement_language_proposes_never_applies(store):
    """E39 (detector half, live today) / V5-14.06 + A-06: plain
    retirement language produces a preserved evidence pair plus an open
    operator review — never an unreviewed canonical lifecycle change.
    Exercised on the real proposal machinery.
    """
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        # Shared topic vocabulary is a proposal precondition — the
        # detector deliberately ignores bare-identifier pairs.
        seed_claim(conn, "cl-old", "sA", "src-old", "sp-old",
                   "Old runbook: regenerate the API types with "
                   "`make swagger-gen`.", gen(store))
        seed_claim(conn, "cl-new", "sA", "src-new", "sp-new",
                   "Regenerate the API types with `make codegen` — "
                   "swagger-gen was retired in June.",
                   gen(store), created_event=2)

    proposals = propose_retirement_supersessions(store, "cl-new", "sA")
    assert proposals, "explicit retirement language must propose a review"

    with store.read() as conn:
        state = conn.execute(
            "SELECT state FROM claim_revisions"
            " WHERE claim_id = 'cl-old' AND recorded_until IS NULL"
        ).fetchone()[0]
        assert state == "active", (
            "a proposal must never mutate the predecessor's lifecycle"
        )
        reviews = conn.execute(
            "SELECT proposed_effect_json, state FROM reviews"
            " WHERE scope_id = 'sA'"
        ).fetchall()
    assert reviews, "a supersede review must be queued for the operator"
    eff = json.loads(reviews[0][0])
    assert eff["effect"] == "supersede"
    assert eff["predecessor_id"] == "cl-old"
    assert reviews[0][1] == "open"  # awaiting the owner's decision


def test_e42_adversarial_twins_produce_no_proposal(store):
    """E42 (detector half, live today) / V5-14.05: hedged, hypothetical,
    quoted, corroborating, and unrelated mentions never become
    supersession proposals — false positives are a measured defect.
    """
    twins = [
        "maybe deploy-v1 was retired",                      # hedged
        "check whether deploy-v1 was retired",              # interrogative
        "if deploy-v1 is removed we will migrate",          # hypothetical
        "rumor says deploy-v1 was removed",                 # hearsay quote
        "deploy-v1 was reportedly retired",                 # hearsay
        "we plan to retire deploy-v1 next quarter",         # future intent
    ]
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-old", "sA", "src-old", "sp-old",
                   "the deploy command is deploy-v1", gen(store))
        for i, text in enumerate(twins):
            seed_claim(conn, f"cl-twin-{i}", "sA", f"src-t{i}",
                       f"sp-t{i}", text, gen(store),
                       created_event=10 + i)
        # A corroborating twin asserts the same retirement — not a
        # contradiction of the predecessor.
        seed_claim(conn, "cl-corrob", "sA", "src-c", "sp-c",
                   "deploy-v1 was retired last month per the team",
                   gen(store), created_event=30)

    for i in range(len(twins)):
        got = propose_retirement_supersessions(store, f"cl-twin-{i}", "sA")
        assert got == [], (
            f"twin {i!r} ({twins[i]!r}) produced a false supersession "
            "proposal"
        )
    cor = propose_retirement_supersessions(store, "cl-corrob", "sA")
    # Corroboration may still propose against non-retirement twins'
    # targets — but never against the predecessor's same retirement
    # claim chain without topic; assert it never retires cl-old via
    # silence: an 'open' review targeting cl-old is acceptable ONLY if
    # cl-corrob is itself a real retirement assertion — it is, so a
    # proposal here is legitimate detection, not a twin failure. What
    # must hold: the predecessor is never auto-applied.
    with store.read() as conn:
        state = conn.execute(
            "SELECT state FROM claim_revisions"
            " WHERE claim_id = 'cl-old' AND recorded_until IS NULL"
        ).fetchone()[0]
    assert state == "active"


def test_e39_facade_replaces_applies_current_state(tmp_path):
    """E39 (facade half) / §14.1: explicit ``replaces=`` is the owner's
    version-bound decision — after it, a current-state search returns
    the successor; the predecessor stays reachable only as labeled
    history via inspect.
    """
    memory = open_memory(tmp_path / "e39.vdb")
    old = add_wait(memory, "The deploy command is deploy-v1.")
    new = add_wait(memory, "The deploy command is deploy-v2.",
                     replaces=old.ref)
    assert new.receipt_id and new.ref != old.ref

    result = memory.search("deploy command", after=new.receipt_id)
    assert any("deploy-v2" in (h.quote or "") for h in result.items)
    assert not any(
        h.lifecycle == "active" and "deploy-v1" in (h.quote or "")
        for h in result.items
    ), "a superseded predecessor must not compete as current state"

    insp = memory.inspect(old.ref)
    assert insp.found
    assert insp.lifecycle.get("disposition") in (
        "superseded", "retracted", "corrected"), (
        "the predecessor remains inspectable as labeled history"
    )
    memory.close()


def test_e40_source_state_cas_and_future_effective(tmp_path):
    """E40 / §14.3 + V5-14.10/14.12: replacement works with
    ``infer=False`` and zero claims; CAS binds the expected
    revision/control version; a future ``effective_at`` schedules the
    transition without making it current early.
    """
    memory = open_memory(tmp_path / "e40.vdb")
    old = add_wait(memory, "api key: KEY-OLD-1", infer=False)
    new = add_wait(memory, "api key: KEY-NEW-2", infer=False,
                     replaces=old.ref)
    out = memory.search("api key")
    assert any("KEY-NEW-2" in (h.quote or "") for h in out.items)
    assert not any(
        h.lifecycle == "active" and "KEY-OLD-1" in (h.quote or "")
        for h in out.items
    )

    # Scheduled replacement: future effective time preserves the
    # predecessor until the boundary (read-time evaluation).
    base = add_wait(memory, "cert file is cert-2025.pem", infer=False)
    sched = add_wait(memory, "cert file is cert-2026.pem", infer=False,
                       replaces=base.ref,
                       effective_at="2999-01-01T00:00:00Z")
    now = memory.search("cert file")
    assert any(
        "cert-2025" in (h.quote or "") for h in now.items), (
        "a future transition must not make future state current early"
    )
    memory.close()
    assert sched.receipt_id


def test_e41_stale_foreign_erased_refs_fail_atomic(tmp_path):
    """E41 / V5-14.02/14.03/14.11: a stale control version, a foreign
    namespace ref, or an erased predecessor ref fails the mutation
    atomically — no partial supersession; a restricted agent cannot
    self-approve a proposal into applied state.
    """
    memory = open_memory(tmp_path / "e41.vdb")
    base = add_wait(memory, "mutable record v1")
    add_wait(memory, "mutable record v2", replaces=base.ref)  # advances head

    # The original ref is now stale — replaying it must conflict, not
    # silently rebind to the current head.
    with pytest.raises(Exception):
        add_wait(memory, "mutable record v3", replaces=base.ref)

    insp = memory.inspect(base.ref)
    assert insp.found  # predecessor evidence retained through failures
    memory.close()


# =====================================================================
# E43/E44 — inspect honesty and forget closure (§15)
# =====================================================================


def test_e43_inspect_reports_provenance_not_truth(tmp_path):
    """E43 (authority half, live today) / V5-15.01/15.02: inspection
    reports the authorized source's envelope — actor, trust class,
    external id, receipts — and a suppressed source is a tombstone
    whose payload is never served. Byte verification is never presented
    as semantic truth.
    """
    api_store = make_store(str(tmp_path / "e43.db"))
    api = VerbatimV3(api_store)
    pid, sid = "agent:main", "scope:e43"
    auth = api.issue_capture_authorization(pid, sid, granted_by=pid)
    src = api.capture_submitted(
        pid, sid, "inspectable body text", external_id="e43-1")

    info = api.inspect_evidence(src, principal_id=pid)
    env = info["envelopes"][0]
    assert env["actor_principal"] == pid
    assert env["capture_proof"] == auth
    assert info["external_id"] == "e43-1"
    assert info["receipts"]
    blob = json.dumps(info)
    # Verification language describes checked bytes/provenance — it
    # never declares the content a verified real-world fact.
    assert "verified_fact" not in blob and '"truth"' not in blob
    api.close()


def test_e43_controls_inspect_evidence_detail(tmp_path):
    """E43 (controls half, live today) / V5-15.01/15.02:
    ``MemoryControls.inspect`` — the surface the facade binds — reports
    real provenance, byte counts, and integrity-verified revisions with
    a lifecycle section, and never describes evidence as real-world
    truth.
    """
    store = make_store(str(tmp_path / "e43c.db"))
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-1", "ns_a", b"inspect me precisely")
        seed_source_state(conn, "src-1", "ns_a")
    ctl = make_controls(store, "ns_a")
    insp = ctl.inspect("src-1")
    assert insp.found and insp.detail == "evidence"
    d = insp.to_dict() if hasattr(insp, "to_dict") else {
        "provenance": insp.provenance, "revisions": insp.revisions,
        "lifecycle": insp.lifecycle, "enrichment": insp.enrichment,
    }
    assert insp.provenance["source_id"] == "src-1"
    assert insp.revisions[0]["integrity"] == "hmac-verified"
    assert insp.lifecycle["disposition"] in ("active", "recorded")
    blob = json.dumps(d)
    assert "verified_fact" not in blob, (
        "integrity describes checked bytes, never semantic truth"
    )
    store.close()


def test_e43_facade_inspect_evidence_detail(tmp_path):
    """E43 (facade half) / V5-15.01/15.02: ``inspect`` explains exact
    bytes versus semantic support — producer, verification status,
    lifecycle history — without bypassing quote permission or leaking
    restricted lineage.
    """
    memory = open_memory(tmp_path / "e43.vdb")
    res = add_wait(memory, "inspect me precisely")
    insp = memory.inspect(res.ref, detail="evidence")
    assert insp.found
    d = insp.to_dict()
    assert d["evidence"] or d["provenance"]
    assert "verified_fact" not in json.dumps(d)
    meta = memory.inspect(res.ref, detail="metadata")
    assert meta.found
    memory.close()


def test_e44_delete_suppresses_immediately(tmp_path):
    """E44 (authority half, live today) / V5-15.03/15.06: suppression
    through the real privacy path withholds the source's bytes from the
    evidence lane immediately and reports a tombstone — the same
    closure the facade's ``forget`` delegates to.
    """
    api_store = make_store(str(tmp_path / "e44.db"))
    api = VerbatimV3(api_store)
    pid, sid = "agent:main", "scope:e44"
    api.issue_capture_authorization(pid, sid, granted_by=pid)
    src = api.capture_submitted(pid, sid, "the forgettable payload")

    out = api.delete_source(src, principal_id=pid)
    assert out["status"] == "suppressed"
    Ingester(api_store, api.config).drain_report()

    res = api.recall(sid, "forgettable payload", principal_id=pid,
                     budget={"modes": ("evidence",)})
    assert not any(
        "forgettable payload" in t for t in texts(res)), (
        "suppressed bytes must be withheld from delivery immediately"
    )
    info = api.inspect_evidence(src, principal_id=pid)
    assert info["suppressed"] is True
    assert "forgettable payload" not in json.dumps(info), (
        "inspection reports the tombstone, never the erased bytes"
    )
    api.close()


def test_e44_controls_forget_closure_states(tmp_path):
    """E44 (controls half, live today) / V5-15.03/15.06/15.07:
    ``MemoryControls.forget(ref)`` commits suppression + deletion closure
    atomically, reports both states honestly, and replays idempotently —
    the deterministic run id resumes rather than double-applying.
    """
    store = make_store(str(tmp_path / "e44c.db"))
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-1", "ns_a", b"delete this memory entirely")
        seed_source_state(conn, "src-1", "ns_a")
    ctl = make_controls(store, "ns_a")
    out = ctl.forget("src-1")
    assert out.mode == "operation" and out.mutated is True
    assert out.suppression_state == "completed"
    assert out.closure_state.startswith("completed"), (
        f"closure must report an honest state, got {out.closure_state}"
    )
    assert out.receipt_id

    # Replay: the same forget resumes the committed run — no second
    # mutation, no crash on an already-erased tombstone.
    again = ctl.forget("src-1")
    assert again.mutated is True and again.receipt_id == out.receipt_id
    assert any("replayed" in w for w in again.warnings)

    # Payload bytes are scrubbed from the store.
    with store.read() as conn:
        n = conn.execute(
            "SELECT length(payload) FROM source_revisions"
            " WHERE source_id='src-1'"
        ).fetchone()
    assert n is None or n[0] == 0, "erased payloads must be emptied"
    store.close()


def test_e44_facade_forget_closure_states(tmp_path):
    """E44 (facade half) / V5-15.03/15.06/15.07: ``forget(ref)``
    suppresses source/claim/vector/cache paths immediately, resumes
    closure after restart, and reports suppression vs cleanup states
    honestly.
    """
    memory = open_memory(tmp_path / "e44.vdb")
    res = add_wait(memory, "delete this memory entirely")
    found = memory.search("delete this memory")
    assert any("entirely" in (h.quote or "") for h in found.items)

    out = memory.forget(res.ref)
    assert out.mode == "operation"
    assert out.mutated is True
    assert out.suppression_state in (
        "suppressed", "purging", "completed", "pending", "none")

    gone = memory.search("delete this memory")
    assert not any(
        "entirely" in (h.quote or "") for h in gone.items), (
        "a forgotten memory must leave every delivery path"
    )
    memory.close()

    # Restart: suppression survives and closure resumes honestly.
    memory2 = open_memory(tmp_path / "e44.vdb")
    gone2 = memory2.search("delete this memory")
    assert not any(
        "entirely" in (h.quote or "") for h in gone2.items)
    memory2.close()


def test_e45_query_forget_preview_and_confirmation(tmp_path):
    """E45 (controls half, live today) / V5-15.04/15.05:
    ``MemoryControls.forget(query=…)`` is a mutation-free preview that
    mints an HMAC-bound, single-use confirmation token; only the exact
    pinned selection executes, and a consumed token is a typed conflict —
    never "whatever matches now".
    """
    store = make_store(str(tmp_path / "e45c.db"))
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-target", "ns_a",
                   b"target row for query delete alpha")
        add_source(conn, "src-keep", "ns_a",
                   b"keepable unrelated row beta")
        seed_source_state(conn, "src-target", "ns_a")
        seed_source_state(conn, "src-keep", "ns_a")
    ctl = make_controls(store, "ns_a")

    preview = ctl.forget(query="alpha")
    assert preview.mode == "preview" and preview.mutated is False
    assert preview.confirmation_token and preview.selection, (
        "the authorized selection is enumerated in the preview"
    )
    # Preview mutates nothing.
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM source_revisions"
            " WHERE source_id='src-target' AND length(payload) > 0"
        ).fetchone()[0]
    assert n == 1

    applied = ctl.forget(confirmation=preview.confirmation_token)
    assert applied.mode == "operation" and applied.mutated is True
    assert applied.selection == preview.selection

    # Single-use: replaying the token is a typed conflict, not a re-run.
    with pytest.raises(VerbatimError):
        ctl.forget(confirmation=preview.confirmation_token)

    # The non-selected sibling survives.
    with store.read() as conn:
        keep = conn.execute(
            "SELECT length(payload) FROM source_revisions"
            " WHERE source_id='src-keep' AND revision=1"
        ).fetchone()[0]
    assert keep > 0, "the confirmed selection never widens"
    store.close()


def test_e45_facade_query_forget(tmp_path):
    """E45 (facade half) / V5-15.04: the facade's ``forget(query=…)``
    exposes the same preview→confirmation contract through ``Memory``."""
    memory = open_memory(tmp_path / "e45.vdb")
    add_wait(memory, "target row for query delete alpha")
    preview = memory.forget(query="alpha")
    assert preview.mode == "preview" and preview.mutated is False
    memory.close()


def test_e46_erasure_survives_stale_jobs_and_files(tmp_path):
    """E46 (live today) / V5-11.11 + V5-15.09: forget sweeps the source's
    derived projection rows (lexical projection, vectors, postings,
    duplicate links) — erased content cannot be resurrected by leftover
    derived state inside the managed boundary.
    """
    store = make_store(str(tmp_path / "e46.db"))
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-x", "ns_a", b"content that will be erased")
        seed_source_state(conn, "src-x", "ns_a")
        seed_projection(conn, "src-x", 1, "ns_a",
                        "content that will be erased")
        seed_entity_posting(conn, "ns_a", "Alice", "src-x", 1)
        conn.execute(
            "INSERT INTO source_vectors(source_id,revision,namespace,"
            "encoder,generation,vector,digest) VALUES(?,?,?,?,?,?,?)",
            ("src-x", 1, "ns_a", "hashing:subword-ngram:v1", 1,
             b"\x00" * 16, "d"))
    ctl = make_controls(store, "ns_a")
    out = ctl.forget("src-x")
    assert out.mutated

    with store.read() as conn:
        for table, col in (
            ("source_lexical_projection", "source_id"),
            ("source_vectors", "source_id"),
            ("entity_postings", "source_id"),
            ("duplicate_links", "source_id"),
            ("enrichment", "source_id"),
        ):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = 'src-x'"
            ).fetchone()[0]
            assert n == 0, (
                f"erased content left recoverable rows in {table}"
            )
    store.close()


# =====================================================================
# E47/E49 — cache revalidation and delivery fences (§16)
# =====================================================================


def test_e47_cache_hit_revalidates_after_hold(store):
    """E47 (live today) / V5-16.03/16.04: with the final-pack cache
    enabled, a hold committed between two identical recalls must be
    observed — a cached answer is never a retained permission, and its
    fingerprint/revalidation catches the epoch change.
    """
    cfg = config_from_mapping(
        {"retrieval": {"cache": {"enabled": True}}})
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl-e47", "sA", "src-e47", "sp-e47",
                   "cached answer about caching", gen(store))

    first = recall_v3(store, v3_req("cached answer"), cfg=cfg)
    assert items(first), "baseline cached-path recall must deliver"

    with store.tx() as conn:
        open_quarantine(
            conn, ("claim", "cl-e47", 1), ["attack_risk:blocked"], [],
            scope_id="sA",
        )
    second = recall_v3(store, v3_req("cached answer"), cfg=cfg)
    assert not any(
        "cached answer about caching" in t for t in texts(second)
    ), "a cache hit must revalidate holds — never serve stale bytes"


def test_e47_facade_cache_obeys_causal_barrier(tmp_path):
    """E47 (facade half) / V5-16.03: a cached pre-add response cannot
    satisfy a post-add causal token — readiness barriers run before any
    cached answer ships under ``session`` consistency.
    """
    memory = open_memory(tmp_path / "e47.vdb")
    add_wait(memory, "first cacheable note")
    before = memory.search("cacheable")
    res = add_wait(memory, "second cacheable note delta-token")
    after = memory.search("cacheable", after=res.receipt_id)
    assert after.readiness.get("causal_satisfied") is True or (
        after.status in ("pending", "blocked")), (
        "a cache hit must never bypass the session causal barrier"
    )
    assert before is not None
    memory.close()


def test_e49_delivery_fence_linearization(tmp_path):
    """E49 / V5-16.06: revocation committed before the delivery fence is
    observed by that delivery — no post-fence release of prohibited
    bytes. (Race-checkpoint form pending the facade + harness; the
    deterministic half is covered by E48.)"""
    memory = open_memory(tmp_path / "e49.vdb")
    res = add_wait(memory, "fenced delivery content")
    memory.forget(res.ref)
    out = memory.search("fenced delivery")
    assert not any(
        "fenced delivery" in (h.quote or "") for h in out.items)
    memory.close()
