"""V5 consumer controls — inspect + forget end-to-end over a real Store.

Every test runs against ``Store.create`` on disk (schema v4 + the v5
control tables declared in docs/v5_contracts.md where exercised) — no
mocks of the kernel, no fixture store shim.  Coverage: inspect
evidence/metadata, denial indistinguishability, forget-by-ref through
the closure engine, suppression fencing, query preview → confirmation
token → operation, token tamper/expiry/rebinding/single-use, CAS
conflict, and idempotent replay.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import ClosurePhase
from verbatim.core.time import now_us, rfc3339
from verbatim.governance import CallerV3, create_grant
from verbatim.memory.controls import MemoryControls, object_ref_to_string
from verbatim.memory.types import MemoryRef
from verbatim.storage.repos import has_table
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

NS = "ns_main"
NS_OTHER = "ns_other"
ALICE = CallerV3(principal_id="alice", session_id="s1", host_id="h1")


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "v5.db"))
    yield s
    s.close()


@pytest.fixture()
def namespace(store):
    """Provision the bound namespace + a second one + alice's grants."""
    with store.tx() as conn:
        for sid in (NS, NS_OTHER):
            conn.execute(
                "INSERT OR IGNORE INTO scopes"
                " (scope_id, profile_id, principal_id, visibility)"
                " VALUES (?, 'prof', 'alice', 'owner')",
                (sid,),
            )
        create_grant(
            conn,
            scope_id=NS,
            principal_id=ALICE.principal_id,
            verbs=["read", "quote", "ingest", "admin"],
            issuer_id="owner",
        )
        create_grant(
            conn,
            scope_id=NS_OTHER,
            principal_id=ALICE.principal_id,
            verbs=["read", "quote", "ingest", "admin"],
            issuer_id="owner",
        )
    return NS


@pytest.fixture()
def controls(store, namespace):
    return MemoryControls(store, caller=ALICE, namespace=namespace)


def qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def qrow(conn, sql, params=()):
    rows = qrows(conn, sql, params)
    return rows[0] if rows else None


def make_source(store, sid, source_id, payload=b"the launch code is falcon-9",
                *, revision=1, speaker="alice"):
    """v2/v3-neutral source rows (no source_state needed — legacy head)."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
            " speaker_id, created_us) VALUES (?, 'test', 'user_message', ?, ?, 1)",
            (source_id, sid, speaker),
        )
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, timezone, provenance, metadata_json)"
            " VALUES (?, ?, ?, ?, 1000, 1000, 'UTC', 'direct_user', '{}')",
            (source_id, revision, payload, store.hmac(payload)),
        )
        conn.execute(
            "INSERT INTO source_envelopes"
            "(envelope_id, source_id, revision, scope_id, envelope_kind,"
            " actor_principal, receipt_us, trust_class, adapter_version,"
            " host_id, session_id)"
            " VALUES (?, ?, ?, ?, 'user_message', 'alice', 1000,"
            " 'principal_direct', 'adapter/test-1.0', 'host-1', 'sess-1')",
            (f"env-{source_id}", source_id, revision, sid),
        )
    return source_id


def add_revision(store, source_id, payload, revision):
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, provenance, metadata_json)"
            " VALUES (?, ?, ?, ?, 2000, 2000, 'direct_user', '{}')",
            (source_id, revision, payload, store.hmac(payload)),
        )


def make_span(store, source_id, payload, span_id, start, end, revision=1):
    excerpt = payload[start:end]
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO spans (span_id, source_id, revision, start_byte,"
            " end_byte, excerpt_hmac, harvester_version)"
            " VALUES (?, ?, ?, ?, ?, ?, 'harv/test-1')",
            (span_id, source_id, revision, start, end, store.hmac(excerpt)),
        )
    return span_id


def mint_ref(store, source_id, *, namespace=NS, rev=1, ctl=0) -> str:
    return MemoryRef(
        store_tag=store.db_id(),
        namespace=namespace,
        source_id=source_id,
        expected_revision=rev,
        control_version=ctl,
    ).to_string()


def suppressed_sources(store):
    with store.read() as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE pt.object_kind = 'source'"
                " AND p.state IN ('suppressed','purging','completed')"
            ).fetchall()
        }


def payload_of(store, source_id, revision=1):
    with store.read() as conn:
        row = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ?"
            " AND revision = ?",
            (source_id, revision),
        ).fetchone()
        return bytes(row[0]) if row and row[0] is not None else None


def seed_v5_state(store, source_id, *, namespace=NS, head=1, ctl=0):
    """Register the ``source_state/v1`` control record through the real
    artifact API (``ensure_state`` + fenced ``transition``s) so tests
    exercise the committed ledger, not a synthetic row."""
    from verbatim import sourcestate

    with store.tx() as conn:
        st = sourcestate.ensure_state(
            conn, source_id, namespace, head=head, producer="test-seed"
        )
        # Alternate non-terminal dispositions to reach the target
        # control version — every hop lands a ledger doc.
        changes = ("archive", "activate")
        i = 0
        while st.control_version < ctl:
            st = sourcestate.transition(
                conn,
                source_id,
                expected_control_version=st.control_version,
                disposition=changes[i % 2],
                producer="test-seed",
                store=store,
            )
            i += 1
        return st


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_evidence_returns_locators_and_provenance(store, controls):
    payload = b"the launch code is falcon-9"
    sid = make_source(store, NS, "src-1", payload)
    make_span(store, sid, payload, "span-1", 19, 27)
    ref = mint_ref(store, sid)

    out = controls.inspect(ref, detail="evidence")
    assert out.found is True
    assert out.ref == ref
    assert out.detail == "evidence"
    # provenance: attribution + trust class + producer
    assert "alice" in out.provenance["attribution"]
    assert "principal_direct" in out.provenance["trust_class"]
    assert "adapter/test-1.0" in out.provenance["producer"]
    # revision chain
    assert [r["revision"] for r in out.revisions] == [1]
    assert out.revisions[0]["integrity"] == "hmac-verified"
    assert out.revisions[0]["provenance"] == "direct_user"
    # lifecycle
    assert out.lifecycle["disposition"] in ("recorded", "active")
    assert out.lifecycle["mutation_head"] == 1
    assert out.lifecycle["suppressed"] is False
    # evidence locator with quote-authorized excerpt
    assert out.evidence[0]["span_id"] == "span-1"
    assert out.evidence[0]["start_byte"] == 19
    assert out.evidence[0]["end_byte"] == 27
    assert out.evidence[0]["text"] == "falcon-9"


def test_inspect_metadata_omits_excerpt_text(store, controls):
    payload = b"bob's favourite colour is teal"
    sid = make_source(store, NS, "src-md", payload)
    make_span(store, sid, payload, "span-md", 0, 18)
    ref = mint_ref(store, sid)

    out = controls.inspect(ref, detail="metadata")
    assert out.found is True
    assert out.detail == "metadata"
    assert out.evidence[0]["span_id"] == "span-md"
    assert "text" not in out.evidence[0]  # locators only, no content
    assert out.provenance["source_kind"] == "user_message"


def test_inspect_bare_memory_id(store, controls):
    sid = make_source(store, NS, "src-bare", b"bare id resolves")
    out = controls.inspect(sid)  # memory_id is a source_id alias (V5-06.11)
    assert out.found is True
    assert out.provenance["source_id"] == sid


def test_inspect_denial_indistinguishable(store, controls, namespace):
    sid = make_source(store, NS, "src-sec", b"secret payload")
    cases = []

    # unknown source
    cases.append(lambda: controls.inspect(mint_ref(store, "nope-src")))
    # foreign namespace
    cases.append(
        lambda: controls.inspect(
            mint_ref(store, sid, namespace="ns_elsewhere")
        )
    )
    # foreign store tag
    foreign = MemoryRef(
        store_tag="deadbeef" * 4,
        namespace=NS,
        source_id=sid,
        expected_revision=1,
        control_version=0,
    ).to_string()
    cases.append(lambda: controls.inspect(foreign))
    # caller without a grant on this namespace
    mallory = MemoryControls(
        store, caller=CallerV3(principal_id="mallory"), namespace=NS
    )
    cases.append(lambda: mallory.inspect(mint_ref(store, sid)))

    codes = set()
    for fn in cases:
        with pytest.raises(VerbatimError) as ei:
            fn()
        codes.add(ei.value.code)
    assert codes == {ErrorCode.NOT_FOUND_OR_UNAUTHORIZED}


def test_inspect_rejects_bad_detail(store, controls):
    sid = make_source(store, NS, "src-det", b"data")
    with pytest.raises(VerbatimError) as ei:
        controls.inspect(mint_ref(store, sid), detail="everything")
    assert ei.value.code == ErrorCode.VALIDATION


def test_inspect_object_ref_claim(store, controls):
    sid = make_source(store, NS, "src-claim", b"claimable text here")
    make_span(store, sid, b"claimable text here", "span-c", 0, 9)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
            " created_event, row_version) VALUES (?, ?, 's', 'p', 1, 1)",
            ("claim-1", NS),
        )
        conn.execute(
            "INSERT INTO claim_revisions (claim_id, revision, state,"
            " polarity, modality, recorded_from) VALUES"
            " ('claim-1', 1, 'active', 'affirmative', 'asserted', 1)",
        )
        conn.execute(
            "INSERT INTO claim_evidence (claim_id, revision, span_id,"
            " evidence_role) VALUES ('claim-1', 1, 'span-c', 'primary')",
        )
    out = controls.inspect(("claim", "claim-1", 1), detail="evidence")
    assert out.found is True
    assert out.provenance["claim_id"] == "claim-1"
    assert out.evidence[0]["source_id"] == sid
    assert out.evidence[0]["text"] == "claimable"


def test_inspect_object_ref_resolves_via_derivations(store, controls):
    """A non-claim object resolves to its evidence roots through the
    recorded derivation graph — never a guessed parent."""
    from verbatim import derivations

    sid = make_source(store, NS, "src-ep", b"episode-backed memory")
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO objects (object_id, kind, scope_id,"
            " current_revision, disposition, created_event)"
            " VALUES ('ep-1', 'episode', ?, 1, 'active', 1)",
            (NS,),
        )
        derivations.record_edge(
            conn,
            ("episode", "ep-1", 1),
            ("source_revision", sid, 1),
            "episode_index",
            "test-seed",
            NS,
            0,
        )
    out = controls.inspect(("episode", "ep-1", 1), detail="metadata")
    assert out.found is True
    assert out.provenance["via"] == "derivations"
    assert out.provenance["object_kind"] == "episode"
    assert out.provenance["resolved_source"] == sid


def test_inspect_object_ref_unknown_kind_validates(store, controls):
    for bad in (("bogus_kind", "x", 1), "vobj1.bogus_kind.7878.1"):
        with pytest.raises(VerbatimError) as ei:
            controls.inspect(bad)
        assert ei.value.code == ErrorCode.VALIDATION


def test_inspect_source_state_object_ref(store, controls):
    """The control artifact itself resolves to the source it binds via
    its recorded ``source_revision`` provenance edge."""
    sid = make_source(store, NS, "src-sso", b"state-bearing memory")
    seed_v5_state(store, sid, head=1)
    from verbatim import sourcestate

    oid = sourcestate.artifact_id(sid)
    out = controls.inspect(("source_state", oid, 1), detail="metadata")
    assert out.found is True
    assert out.provenance["resolved_source"] == sid


# ---------------------------------------------------------------------------
# forget by ref
# ---------------------------------------------------------------------------


def test_forget_by_ref_suppresses_and_erases(store, controls):
    payload = b"forget this secret entirely"
    sid = make_source(store, NS, "src-forget", payload)
    make_span(store, sid, payload, "span-f", 0, 6)
    ref = mint_ref(store, sid)

    res = controls.forget(ref)
    assert res.mode == "operation"
    assert res.mutated is True
    assert res.receipt_id
    assert res.suppression_state in ("suppressed", "purging", "completed")
    assert res.closure_state.startswith(
        ("completed", "cleaning", "verifying", "failed_cleanup")
    )
    # Suppression committed + canonical bytes emptied by the closure.
    assert sid in suppressed_sources(store)
    assert payload_of(store, sid) == b""
    # Spans skeletons may persist, but excerpt text must not resolve.
    from verbatim.storage.repos import SpansRepo

    assert SpansRepo(store).text("span-f") is None


def test_forget_excludes_from_subsequent_selection(store, controls):
    payload = b"my pet otter is named biscuit"
    sid = make_source(store, NS, "src-gone", payload)
    controls.forget(mint_ref(store, sid))
    # A fresh query preview must not select a suppressed memory (the
    # search-equivalent eligibility fence), and the purge registry fences
    # it for every delivery surface.
    preview = controls.forget(query="pet otter biscuit")
    assert all(
        MemoryRef.parse(r).source_id != sid for r in preview.selection
    )
    assert sid in suppressed_sources(store)


def test_inspect_is_honest_after_forget(store, controls):
    payload = b"inspect me after erasure"
    sid = make_source(store, NS, "src-honest", payload)
    ref = mint_ref(store, sid)
    controls.forget(ref)

    out = controls.inspect(ref, detail="evidence")
    assert out.found is True  # tombstone is inspectable, not hidden
    assert out.lifecycle["suppressed"] is True
    assert "suppressed" in " ".join(out.warnings)
    assert out.revisions[0]["integrity"] == "purged"
    # No resurrectable content anywhere in the report.
    assert payload.decode() not in repr(out.to_dict())


def test_forget_rejects_bare_claim_and_view_refs(store, controls):
    for bad in (
        ("claim", "claim-9", 1),
        ("view", "view-1", None),
        {"kind": "claim", "id": "claim-9"},
        object_ref_to_string("claim", "claim-9", 1),
    ):
        with pytest.raises(VerbatimError) as ei:
            controls.forget(bad)
        assert ei.value.code == ErrorCode.VALIDATION


def test_forget_bare_memory_id_resolves_and_cas_guards(store, controls):
    sid = make_source(store, NS, "src-bare-f", b"bare id forget path")
    res = controls.forget(sid)
    assert res.mutated is True
    assert sid in suppressed_sources(store)


def test_forget_cas_conflict_on_stale_revision(store, controls):
    sid = make_source(store, NS, "src-cas", b"v1 content")
    add_revision(store, sid, b"v2 content moved on", revision=2)
    # Ref minted at head=1 — the live head is now 2.
    with pytest.raises(VerbatimError) as ei:
        controls.forget(mint_ref(store, sid, rev=1, ctl=0))
    assert ei.value.code in (
        ErrorCode.OPERATION_CONFLICT,
        ErrorCode.STALE_PROPOSAL,
    )
    # Nothing was mutated.
    assert sid not in suppressed_sources(store)
    assert payload_of(store, sid, revision=2) == b"v2 content moved on"


def test_forget_cas_on_source_state_control_version(store, controls):
    sid = make_source(store, NS, "src-ctl", b"control-versioned")
    seed_v5_state(store, sid, head=1, ctl=3)
    # Correct pin → succeeds.
    res = controls.forget(mint_ref(store, sid, rev=1, ctl=3))
    assert res.mutated is True
    # source_state tombstone: erased + control version bumped so the old
    # pin can never CAS again (V5-14.16).
    with store.read() as conn:
        row = qrow(conn, "SELECT * FROM source_state WHERE source_id = ?", (sid,))
    assert row["disposition"] == "erased"
    assert row["control_version"] == 4


def test_forget_stale_control_version_conflicts(store, controls):
    sid = make_source(store, NS, "src-ctl2", b"control-versioned")
    seed_v5_state(store, sid, head=1, ctl=3)
    with pytest.raises(VerbatimError) as ei:
        controls.forget(mint_ref(store, sid, rev=1, ctl=2))
    assert ei.value.code in (ErrorCode.OPERATION_CONFLICT, ErrorCode.STALE_PROPOSAL)
    assert sid not in suppressed_sources(store)


def test_forget_idempotent_replay(store, controls):
    sid = make_source(store, NS, "src-idem", b"idempotent delete")
    ref = mint_ref(store, sid)
    first = controls.forget(ref, idempotency_key="k-1")
    second = controls.forget(ref, idempotency_key="k-1")
    assert second.receipt_id == first.receipt_id
    assert second.mutated is True
    # Same key + different request → typed conflict.
    sid2 = make_source(store, NS, "src-idem2", b"other memory")
    with pytest.raises(VerbatimError) as ei:
        controls.forget(mint_ref(store, sid2), idempotency_key="k-1")
    assert ei.value.code == ErrorCode.OPERATION_CONFLICT
    # Repeating without a key still resolves the same durable run.
    third = controls.forget(ref)
    assert third.receipt_id == first.receipt_id


def test_forget_denies_foreign_namespace_and_unauthorized(store, controls):
    sid = make_source(store, NS, "src-f2", b"foreign attempt")
    with pytest.raises(VerbatimError) as ei:
        controls.forget(mint_ref(store, sid, namespace=NS_OTHER))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    mallory = MemoryControls(
        store, caller=CallerV3(principal_id="mallory"), namespace=NS
    )
    with pytest.raises(VerbatimError) as ei:
        mallory.forget(mint_ref(store, sid))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    # A source living in another namespace is invisible to this binding.
    other = make_source(store, NS_OTHER, "src-other", b"not yours")
    with pytest.raises(VerbatimError) as ei:
        controls.forget(mint_ref(store, other))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# forget by query — preview + confirmation token
# ---------------------------------------------------------------------------


def test_query_preview_mutates_nothing(store, controls):
    payload = b"preview me but do not delete"
    sid = make_source(store, NS, "src-prev", payload)
    res = controls.forget(query="preview delete")
    assert res.mode == "preview"
    assert res.mutated is False
    assert res.confirmation_token.startswith("vfc1.")
    assert sid not in suppressed_sources(store)
    assert payload_of(store, sid) == payload
    # The selected ref is version-bound and names exactly this source.
    assert res.selection
    mref = MemoryRef.parse(res.selection[0])
    assert mref.source_id == sid
    assert mref.namespace == NS


def test_query_preview_never_leaks_other_namespaces(store, controls):
    make_source(store, NS_OTHER, "src-hidden", b"shared phrase payload")
    res = controls.forget(query="shared phrase")
    assert res.selection == []


def test_confirmation_executes_pinned_selection(store, controls):
    payload = b"confirm the deletion of this"
    sid = make_source(store, NS, "src-confirm", payload)
    preview = controls.forget(query="confirm deletion")
    assert preview.mutated is False
    res = controls.forget(confirmation=preview.confirmation_token)
    assert res.mode == "operation"
    assert res.mutated is True
    assert res.receipt_id
    assert sid in suppressed_sources(store)
    assert payload_of(store, sid) == b""


def test_tampered_token_rejected(store, controls):
    make_source(store, NS, "src-tamp", b"tamper target content")
    preview = controls.forget(query="tamper target")
    token = preview.confirmation_token
    head, payload_b64, mac = token.split(".")
    bad = f"{head}.{payload_b64[:-1]}{'A' if payload_b64[-1] != 'A' else 'B'}.{mac}"
    with pytest.raises(VerbatimError) as ei:
        controls.forget(confirmation=bad)
    assert ei.value.code in (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.VALIDATION,
    )
    bad_mac = f"{head}.{payload_b64}.{'0' * len(mac)}"
    with pytest.raises(VerbatimError) as ei:
        controls.forget(confirmation=bad_mac)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expired_token_rejected(store, namespace):
    ctl = MemoryControls(
        store, caller=ALICE, namespace=NS, confirm_ttl_us=-1
    )
    make_source(store, NS, "src-exp", b"expired token target")
    preview = ctl.forget(query="expired token")
    with pytest.raises(VerbatimError) as ei:
        ctl.forget(confirmation=preview.confirmation_token)
    assert ei.value.code == ErrorCode.PERMIT_EXPIRED


def test_wrong_namespace_and_caller_tokens_rejected(store, controls):
    make_source(store, NS, "src-rebind", b"rebinding target")
    preview = controls.forget(query="rebinding target")
    token = preview.confirmation_token

    other_ns = MemoryControls(store, caller=ALICE, namespace=NS_OTHER)
    with pytest.raises(VerbatimError) as ei:
        other_ns.forget(confirmation=token)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    mallory = MemoryControls(
        store, caller=CallerV3(principal_id="mallory"), namespace=NS
    )
    with pytest.raises(VerbatimError) as ei:
        mallory.forget(confirmation=token)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # A second store cannot redeem the token either (store binding).
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        s2 = Store.create(os.path.join(td, "other.db"))
        try:
            with s2.tx() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO scopes"
                    " (scope_id, profile_id, principal_id, visibility)"
                    " VALUES (?, 'prof', 'alice', 'owner')",
                    (NS,),
                )
                create_grant(
                    conn, scope_id=NS, principal_id="alice",
                    verbs=["read", "admin"], issuer_id="owner",
                )
            foreign = MemoryControls(s2, caller=ALICE, namespace=NS)
            with pytest.raises(VerbatimError) as ei:
                foreign.forget(confirmation=token)
            assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        finally:
            s2.close()


def test_confirmation_is_single_use(store, controls):
    make_source(store, NS, "src-once", b"single use token content")
    preview = controls.forget(query="single use")
    token = preview.confirmation_token
    res = controls.forget(confirmation=token)
    assert res.mutated is True
    with pytest.raises(VerbatimError) as ei:
        controls.forget(confirmation=token)
    assert ei.value.code == ErrorCode.OPERATION_CONFLICT


def test_confirmation_rejects_drifted_selection(store, controls):
    payload = b"drift target v1"
    sid = make_source(store, NS, "src-drift", payload)
    preview = controls.forget(query="drift target")
    # The head moves after the preview — the pinned selection no longer
    # matches live state and must abort rather than delete the new data.
    add_revision(store, sid, b"drift target v2", revision=2)
    with pytest.raises(VerbatimError) as ei:
        controls.forget(confirmation=preview.confirmation_token)
    assert ei.value.code == ErrorCode.STALE_PROPOSAL
    assert sid not in suppressed_sources(store)
    assert payload_of(store, sid, revision=2) == b"drift target v2"


def test_malformed_tokens_rejected(store, controls):
    for bad in ("", "nope", "vfc1.", "vfc1.notb64.sig", "vfc1.e30=.zz"):
        with pytest.raises(VerbatimError) as ei:
            controls.forget(confirmation=bad)
        assert ei.value.code in (
            ErrorCode.VALIDATION,
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        )


def test_forget_argument_validation(store, controls):
    with pytest.raises(VerbatimError) as ei:
        controls.forget()
    assert ei.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as ei:
        controls.forget("x", query="q")
    assert ei.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as ei:
        controls.forget(query="   ")
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# v5 derived-layer closure coverage
# ---------------------------------------------------------------------------


def test_forget_sweeps_v5_derived_layers(store, controls):
    payload = b"sweep every derived layer"
    sid = make_source(store, NS, "src-sweep", payload)
    seed_v5_state(store, sid, head=1, ctl=0)
    now = rfc3339(now_us())
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO source_lexical_projection"
            " (source_id, revision, scope_id, generation, tokens,"
            "  doc_len, digest) VALUES (?,1,?,0,'toks',2,'d')",
            (sid, NS),
        )
        # FTS carrier + content pair — deleting the content row drives
        # the source_fts_idx shadow via the v5 triggers.
        conn.execute(
            "INSERT INTO source_fts_rows"
            " (source_id, revision, scope_id, generation)"
            " VALUES (?,1,?,0)",
            (sid, NS),
        )
        row_id = conn.execute(
            "SELECT row_id FROM source_fts_rows WHERE source_id = ?", (sid,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO source_fts (fts_row_id, text) VALUES (?, 'toks')",
            (row_id,),
        )
        conn.execute(
            "INSERT INTO source_vectors"
            " (source_id, revision, namespace, encoder, generation,"
            "  vector, digest) VALUES (?,1,?,'enc',0,X'00','d')",
            (sid, NS),
        )
        conn.execute(
            "INSERT INTO entity_postings"
            " (namespace, entity, entity_kind, source_id, revision,"
            "  offsets, generation) VALUES (?,'entity-1','thing',?,1,'[]',0)",
            (NS, sid),
        )
        conn.execute(
            "INSERT INTO duplicate_links"
            " (source_id, revision, group_id, method, score, created_at)"
            " VALUES (?,1,'grp-1','exact_digest',1.0,?)",
            (sid, now),
        )
        # Another member anchored on this source as group head — the
        # anchor id must not outlive the erased source (contract §9:
        # group_id is the earliest live member's source_id).
        conn.execute(
            "INSERT INTO duplicate_links"
            " (source_id, revision, group_id, method, score, created_at)"
            " VALUES ('src-twin',1,?,'normalized',0.9,?)",
            (sid, now),
        )
        conn.execute(
            "INSERT INTO enrichment"
            " (source_id, revision, producer, type, polarity,"
            "  time_precision, time_status, event_at, anchor_at,"
            "  fields_json) VALUES (?,1,'enrich/v1','fact',"
            " 'affirmative','day','ongoing',NULL,NULL,'{}')",
            (sid,),
        )
        conn.execute(
            "INSERT INTO update_candidates"
            " (candidate_id, namespace, new_source_id, new_revision,"
            "  prior_source_id, prior_revision, relation, score, state,"
            "  created_at) VALUES ('cand-1',?,?,1,?,1,"
            " 'newer_value',0.9,'open',?)",
            (NS, "src-other-new", sid, now),
        )
    res = controls.forget(mint_ref(store, sid, rev=1, ctl=0))
    assert res.mutated is True
    with store.read() as conn:
        for table in (
            "source_lexical_projection",
            "source_fts_rows",
            "source_fts",
            "source_vectors",
            "entity_postings",
            "duplicate_links",
            "enrichment",
        ):
            key = "fts_row_id" if table == "source_fts" else "source_id"
            assert (
                qrows(conn, f"SELECT * FROM {table} WHERE {key} = ?",
                      (row_id if key == "fts_row_id" else sid,))
                == []
            ), table
        # group-anchor rows for the erased source are gone too.
        assert (
            qrows(conn, "SELECT * FROM duplicate_links WHERE group_id = ?",
                  (sid,))
            == []
        )
        if has_table(conn, "source_fts_idx"):
            assert (
                qrows(conn, "SELECT * FROM source_fts_idx"
                      " WHERE source_fts_idx MATCH 'toks'")
                == []
            )
        assert (
            qrows(
                conn,
                "SELECT * FROM update_candidates WHERE new_source_id = ?"
                " OR prior_source_id = ?",
                (sid, sid),
            )
            == []
        )
        st = qrow(conn, "SELECT * FROM source_state WHERE source_id = ?", (sid,))
    assert st["disposition"] == "erased"
    # The committed ledger doc proves the tombstone — it is not swept.
    with store.read() as conn:
        docs = qrow(
            conn,
            "SELECT COUNT(*) AS n FROM object_revisions"
            " WHERE kind = 'source_state'",
        )
    assert docs["n"] >= 1
    assert "derived_swept" in res.closure_state


def test_inspect_reports_v5_surfaces(store, controls):
    payload = b"enriched inspectable memory"
    sid = make_source(store, NS, "src-enrich", payload)
    seed_v5_state(store, sid, head=1, ctl=0)
    now = rfc3339(now_us())
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO enrichment"
            " (source_id, revision, producer, type, polarity,"
            "  time_precision, time_status, event_at, anchor_at,"
            "  fields_json) VALUES (?,1,'enrich/v1','fact',"
            " 'affirmative','day','ongoing','2026-01-01T00:00:00Z',NULL,"
            " '{\"entities\":[\"acme\"]}')",
            (sid,),
        )
        conn.execute(
            "INSERT INTO duplicate_links"
            " (source_id, revision, group_id, method, score, created_at)"
            " VALUES (?,1,'grp-x','exact_digest',1.0,?)",
            (sid, now),
        )
        conn.execute(
            "INSERT INTO duplicate_links"
            " (source_id, revision, group_id, method, score, created_at)"
            " VALUES ('src-twin',1,'grp-x','exact_digest',1.0,?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO update_candidates"
            " (candidate_id, namespace, new_source_id, new_revision,"
            "  prior_source_id, prior_revision, relation, score, state,"
            "  created_at) VALUES ('cand-9',?,?,1,?,1,"
            " 'contradicts',0.8,'open',?)",
            (NS, sid, "src-prior", now),
        )
    out = controls.inspect(mint_ref(store, sid, rev=1, ctl=0))
    assert out.enrichment["enrichment"][0]["type"] == "fact"
    assert "grp-x" in out.enrichment["duplicate_groups"]
    assert "src-twin" in out.enrichment["duplicate_groups"]["grp-x"]
    assert out.enrichment["update_candidates"][0]["state"] == "open"
    assert out.lifecycle["source_state"]["disposition"] == "active"
    assert out.lifecycle["control_version"] == 0
    # The committed control ledger + read-time evaluation ride inspect.
    assert out.lifecycle["history"] is not None
    assert len(out.lifecycle["history"]) == 1  # the create doc
    assert out.lifecycle["history"][0]["change"] == "create"
    assert out.lifecycle["evaluation"]["current"] is True
    assert out.lifecycle["evaluation"]["effective_revision"] == 1


def test_quote_permission_gates_evidence_text(store, namespace):
    """read-only caller: locators visible, excerpt text withheld (V5-15.01)."""
    reader = MemoryControls(
        store,
        caller=CallerV3(principal_id="reader"),
        namespace=NS,
    )
    with store.tx() as conn:
        create_grant(
            conn, scope_id=NS, principal_id="reader",
            verbs=["read"], issuer_id="owner",
        )
    payload = b"quoted text must stay gated"
    sid = make_source(store, NS, "src-quote", payload)
    make_span(store, sid, payload, "span-q", 0, 6)
    out = reader.inspect(mint_ref(store, sid), detail="evidence")
    assert out.found is True
    assert out.evidence[0]["text"] == "[unavailable]"
    assert any("quote_not_granted" in w for w in out.warnings)
