"""SPEC_V4 §20 — grounded synthesis + derived views (V4-20.01–20.10).

Every test runs against a real file-backed ``Store.create`` with the
pinned profile HMAC key (see conftest). All evidence bytes flow through
``Kernel.read_verified``/``derive_inputs`` — the module never touches a
payload column directly; these tests exercise that contract end to end:
support-bound statements, quote gating, held/purged/corrected-parent
suppression (C47–C49), restart persistence, and capability honesty.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import CapabilityRung
from verbatim.purge import suppress
from verbatim.security.quarantine import open_quarantine
from verbatim.storage import repos_v4
from verbatim.storage.store import Store
from verbatim.synthesis import SYNTHESIS_MODE, VIEW_OBJECT_KIND, Synthesizer
from verbatim.synthesis.types import StatementForm

from tests.kernel.conftest import T0, add_source, add_span, caller, grant
from tests.synthesis.conftest import add_claim, bootstrap

PAYLOAD_A = b"The flight departed Oslo at dawn. Arrival went smoothly."
PAYLOAD_B = b"Budget review moved the launch to September next year."
PAYLOAD_C = b"Unrelated notes about lunch and a bicycle."


def seed(conn, store, scope_id, *payloads):
    """One source + one whole-payload span per payload; returns
    [(src_id, span_id), ...]."""
    out = []
    for i, payload in enumerate(payloads):
        src = f"src:{scope_id.rsplit(':', 1)[-1]}{i}"
        add_source(conn, store, scope_id, src, payload)
        sp = f"sp:{src.rsplit(':', 1)[-1]}"
        add_span(conn, store, sp, src, 1, 0, len(payload), payload)
        out.append((src, sp))
    return out


def grant_all(conn, scope_id, pid, verbs, purposes=("recall", "derive")):
    return grant(conn, scope_id, pid, verbs, purposes=purposes)


# =====================================================================
# composition + support bindings (V4-20.01–05)
# =====================================================================


def test_compose_extractive_binds_exact_support(store, kernel, synth):
    """Every quote statement carries a verified ``kind:id@rev`` + byte
    range binding; the view is honestly labeled and never impersonates
    canonical evidence."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A, PAYLOAD_B)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])

    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    assert view.synthesis_mode == SYNTHESIS_MODE == "grounded_composition"
    assert view.revision == 1 and view.persisted
    assert len(view.statements) == 2
    for st in view.statements:
        assert st.form == StatementForm.QUOTE.value
        assert st.confidence == "supported"
        assert st.text  # caller holds quote → canonical bytes ship
        primary = st.support[0]
        assert primary.evidence_kind == "span"
        assert primary.evidence_revision == 1
        assert primary.verdict == "verified"
        assert primary.start_byte == 0
        # The span's covering source is bound as a second pinned parent.
        kinds = {(s.evidence_kind) for s in st.support}
        assert kinds == {"span", "source"}
    texts = sorted(st.text for st in view.statements)
    assert texts == sorted([PAYLOAD_A.decode(), PAYLOAD_B.decode()])
    # inputs_digests pin every contributing input (V4-20.02).
    assert view.inputs_digests
    assert view.contributing_scopes == ("scope:a",)

    # Persistence: objects/object_revisions + view_support + edges.
    with store.read() as conn:
        obj = repos_v4.get(
            conn, "objects",
            {"kind": VIEW_OBJECT_KIND, "object_id": view.view_id},
        )
        assert obj["disposition"] == "active"
        rev = repos_v4.get(
            conn, "object_revisions",
            {"kind": VIEW_OBJECT_KIND, "object_id": view.view_id,
             "revision": 1},
        )
        assert rev["producer_ref"] == synth._producer_id
        supports = repos_v4.query(
            conn, "view_support",
            {"view_kind": VIEW_OBJECT_KIND, "view_id": view.view_id},
        )
        # 2 statements × (span + source) supports.
        assert len(supports) == 4
        edges = repos_v4.query(
            conn, "dependency_edges",
            {"child_kind": VIEW_OBJECT_KIND, "child_id": view.view_id},
        )
        assert {(e["parent_kind"], e["parent_id"]) for e in edges} >= {
            ("span", "sp:a0"), ("source", "src:a0"),
            ("span", "sp:a1"), ("source", "src:a1"),
        }


def test_compose_via_producer_grant_inherits_restrictions(
    store, kernel, synth
):
    """Producer path: ``kernel.derive_inputs`` verifies inputs and the
    view carries the kernel-computed inherited restrictions."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read"])
        prod_grant = grant(
            conn, "scope:a", "agent:prod", ["derive", "quote"],
            purposes=["derive", "recall"],
        )
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall",
        producer_grant=prod_grant, audience=["human:alice"], now_us=T0,
    )
    assert view.producer_id == "producer:verbatim.grounded-composition.v1"
    assert view.statements and view.statements[0].confidence == "supported"
    # The producer's recall-purpose coverage flows into the inherited set.
    assert view.allowed_purposes.permits("recall")
    assert view.effective_audience == ("human:alice",)


def test_unsupported_proposition_omitted_never_ships(
    store, kernel, synth
):
    """V4-20.05/C47: an ungrounded candidate contributes nothing — no
    statement, no support row, and it cannot corroborate later output."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a",
        caller=caller(),
        purpose="recall",
        # One real ref + one that does not exist at all.
        inputs=[
            ("span", "sp:a0", 1),
            ("span", "sp:ghost", 1),
        ],
        now_us=T0,
    )
    assert len(view.statements) == 1
    assert view.omitted.get("unsupported", 0) >= 1
    assert all(
        s.evidence_id != "sp:ghost"
        for st in view.statements for s in st.support
    )
    with store.read() as conn:
        rows = repos_v4.query(
            conn, "view_support",
            {"view_id": view.view_id, "evidence_id": "sp:ghost"},
        )
    assert rows == []


def test_topic_selection_uses_verified_text(store, kernel, synth):
    """Query matching runs over verified bytes only — a scope whose other
    payloads do not contain the terms contributes no statements."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A, PAYLOAD_C)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", "flight departed", caller=caller(),
        purpose="recall", now_us=T0,
    )
    assert len(view.statements) == 1
    assert view.statements[0].text == PAYLOAD_A.decode()
    assert view.omitted.get("unmatched", 0) == 1


def test_typed_summary_composes_metadata_only(store, kernel, synth):
    """A caller with derive+read but no quote gets a deterministic
    field-rendered view — zero canonical bytes anywhere."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        add_claim(conn, "cl:1", "scope:a", "sp:a0")
        grant_all(conn, "scope:a", "human:alice", ["read", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall",
        view_kind="typed_summary", now_us=T0,
    )
    assert view.statements
    assert all(
        st.form == StatementForm.FIELD.value for st in view.statements
    )
    assert all(st.locator is None for st in view.statements)
    # Claim section ships a field statement binding the claim revision.
    claim_stmts = [s for s in view.statements if s.section == "claims"]
    assert claim_stmts and any(
        s.evidence_kind == "claim" for s in claim_stmts[0].support
    )


# =====================================================================
# quote gating + authorization boundaries (V4-08.02, C48)
# =====================================================================


def test_quote_denied_caller_gets_metadata_only(store, kernel, synth):
    """Without ``quote`` the delivered view carries locators + digests +
    support refs — never canonical bytes."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        prod = grant(conn, "scope:a", "agent:prod",
                     ["derive", "quote"], purposes=["derive", "recall"])
        grant_all(conn, "scope:a", "human:alice", ["read", "quote"])
        grant_all(conn, "scope:a", "human:bob", ["read"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall",
        producer_grant=prod, now_us=T0,
    )
    assert view.statements[0].text  # alice holds quote

    delivered = synth.deliver(
        view.view_id, caller=caller("human:bob"), purpose="recall",
        now_us=T0,
    )
    assert delivered.statements
    assert all(st.text is None for st in delivered.statements)
    st = delivered.statements[0]
    assert st.locator and st.evidence_digest
    assert st.support[0].evidence_kind == "span"
    # No canonical bytes anywhere in the delivered structure.
    assert PAYLOAD_A.decode() not in repr(delivered)


def test_unauthorized_caller_denied(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    with pytest.raises(VerbatimError) as exc:
        synth.compose(
            "scope:a", caller=caller("human:bob"), purpose="recall",
            now_us=T0,
        )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(
            view.view_id, caller=caller("human:bob"), purpose="recall",
            now_us=T0,
        )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_cross_scope_delivery_requires_every_scope(store, kernel, synth):
    """C48: a view spanning two scopes is unavailable to a caller denied
    either contributing scope — at compose AND at delivery."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a", "scope:b"))
        seed(conn, store, "scope:a", PAYLOAD_A)
        seed(conn, store, "scope:b", PAYLOAD_B)
        # Producer spans both scopes.
        prod_a = grant(conn, "scope:a", "agent:prod",
                       ["derive", "quote"], purposes=["derive", "recall"])
        grant(conn, "scope:b", "agent:prod",
              ["derive", "quote"], purposes=["derive", "recall"])
        grant_all(conn, "scope:a", "human:alice", ["read", "quote"])
        grant_all(conn, "scope:b", "human:alice", ["read", "quote"])
        grant_all(conn, "scope:a", "human:bob", ["read"])  # b denied
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall",
        producer_grant=prod_a,
        inputs=[("span", "sp:a0", 1), ("span", "sp:b0", 1)],
        now_us=T0,
    )
    assert set(view.contributing_scopes) == {"scope:a", "scope:b"}
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(
            view.view_id, caller=caller("human:bob"), purpose="recall",
            now_us=T0,
        )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_grant_check_can_only_narrow(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice",
                  ["read", "quote", "derive"])
    denied = synth_check = lambda *a, **k: False
    with pytest.raises(VerbatimError) as exc:
        synth.compose(
            "scope:a", caller=caller(), purpose="recall",
            grant_check=denied, now_us=T0,
        )
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# =====================================================================
# lifecycle invalidation — held / purged / corrected parents (V4-20.07)
# =====================================================================


def test_held_parent_suppresses_view_until_rebuild(store, kernel, synth):
    """C49: quarantining a bound parent withholds the derived view —
    delivery stales, the suppression is durable, and a rebuild mints a
    new revision only after the hold lifts (or the parent is dropped)."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A, PAYLOAD_B)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    assert len(view.statements) == 2

    with store.tx() as conn:
        open_quarantine(
            conn, ("source", "src:a1", 1), ["reason:test"], [],
            scope_id="scope:a",
        )
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(
            view.view_id, caller=caller(), purpose="recall", now_us=T0
        )
    assert exc.value.code == ErrorCode.STALE_DEPENDENCY
    st = synth.status(view.view_id, caller=caller(), purpose="recall",
                      now_us=T0)
    assert st["disposition"] == "held"

    # Rebuild over the surviving evidence mints revision 2; the withheld
    # parent contributes nothing (no stale text, no dangling support).
    view2 = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0 + 1
    )
    assert view2.revision == 2
    assert len(view2.statements) == 1
    assert view2.statements[0].text == PAYLOAD_A.decode()
    assert synth.status(
        view.view_id, caller=caller(), purpose="recall", now_us=T0 + 2
    )["disposition"] == "active"


def test_purged_parent_suppresses_view(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    suppress(store, "scope:a", [("source", "src:a0", 1)], "human:alice")
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(
            view.view_id, caller=caller(), purpose="recall", now_us=T0
        )
    assert exc.value.code == ErrorCode.STALE_DEPENDENCY


def test_corrected_parent_invalidates_then_rebuilds(store, kernel, synth):
    """A new parent revision stales the pinned view; recomputation mints
    a fresh revision bound to the corrected bytes."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    corrected = b"Corrected: the flight departed Bergen at dusk."
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, timezone, provenance, metadata_json)"
            " VALUES ('src:a0', 2, ?, ?, ?, ?, NULL, 'direct_user', '{}')",
            (corrected, store.hmac(corrected), T0, T0),
        )
        # Correction re-harvests: a new span pins the corrected revision.
        add_span(conn, store, "sp:a0r2", "src:a0", 2,
                 0, len(corrected), corrected)
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(
            view.view_id, caller=caller(), purpose="recall", now_us=T0
        )
    assert exc.value.code == ErrorCode.STALE_DEPENDENCY

    view2 = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0 + 1
    )
    assert view2.revision == 2
    # The rev-1 span quotes superseded bytes — omitted from the rebuild;
    # the rev-2 span mints a statement bound to the corrected revision.
    assert len(view2.statements) == 1
    st = view2.statements[0]
    assert st.text == corrected.decode()
    assert st.support[0].evidence_id == "sp:a0r2"
    assert st.support[0].evidence_revision == 2
    delivered = synth.deliver(
        view.view_id, caller=caller(), purpose="recall", now_us=T0 + 2
    )
    assert delivered.revision == 2


def test_kernel_invalidate_cascade_and_eager_hook(store, kernel, synth):
    """``kernel.invalidate`` traverses ``dependency_edges`` — the view
    lands in ``affected`` with a ``reevaluate`` obligation, and the
    optional eager hook flips disposition atomically in the same tx."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    with store.tx() as conn:
        report = kernel.invalidate(
            conn,
            {"kind": "correction", "scope_ids": ["scope:a"],
             "object_refs": [("source", "src:a0", 1)]},
            now_us=T0 + 1,
        )
        marked = synth.note_invalidation(conn, report)
    assert marked == 1
    affected = {(k, i) for k, i, _ in report.affected}
    assert (VIEW_OBJECT_KIND, view.view_id) in affected
    owed = {(o["object_kind"], o["object_id"]) for o in report.obligations}
    assert (VIEW_OBJECT_KIND, view.view_id) in owed
    st = synth.status(
        view.view_id, caller=caller(), purpose="recall", now_us=T0 + 2
    )
    assert st["disposition"] == "held"


def test_tampered_evidence_fails_closed(store, kernel, synth):
    """A payload whose keyed digest no longer matches can never ground a
    statement — composition fails closed, never silently downgrades."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
        conn.execute(
            "UPDATE source_revisions SET payload = X'00' WHERE source_id"
            " = 'src:a0'"
        )
    with pytest.raises(VerbatimError) as exc:
        synth.compose(
            "scope:a", caller=caller(), purpose="recall", now_us=T0
        )
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# =====================================================================
# durability — persistence, revisioning, erasure, restart
# =====================================================================


def test_idempotent_recompose_delivers_same_revision(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    v1 = synth.compose("scope:a", caller=caller(), purpose="recall",
                       now_us=T0)
    v2 = synth.compose("scope:a", caller=caller(), purpose="recall",
                       now_us=T0 + 1)
    assert v1.view_id == v2.view_id and v2.revision == 1


def test_restart_persistence(store, kernel, synth, tmp_path):
    """A minted view is a durable registered object — close/reopen and
    the same bytes + bindings deliver again."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    path = store._path if hasattr(store, "_path") else None
    store.close()
    store2 = Store.open(str(tmp_path / "syn.db"))
    try:
        synth2 = Synthesizer(store2)
        delivered = synth2.deliver(
            view.view_id, caller=caller(), purpose="recall", now_us=T0
        )
        assert delivered.statements[0].text == PAYLOAD_A.decode()
        assert delivered.output_digest == view.output_digest
    finally:
        store2.close()


def test_erase_lifecycle(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice",
                  ["read", "quote", "derive", "admin"],
                  purposes=("recall", "derive", "admin"))
    view = synth.compose(
        "scope:a", caller=caller(), purpose="recall", now_us=T0
    )
    synth.erase(view.view_id, caller=caller(), purpose="admin", now_us=T0)
    with pytest.raises(VerbatimError) as exc:
        synth.deliver(view.view_id, caller=caller(), purpose="recall",
                      now_us=T0)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    # Erased views are indistinguishable-absent for recomputation too.
    with pytest.raises(VerbatimError):
        synth.compose("scope:a", caller=caller(), purpose="recall",
                      now_us=T0)


# =====================================================================
# capability honesty (V4-20.10, V4-50)
# =====================================================================


def test_capability_unavailable_when_unprovisioned(store, kernel):
    """No registered producer → honest CAPABILITY_UNAVAILABLE, not a
    silently simulated view."""
    bare = Synthesizer(store, kernel=kernel)
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    with pytest.raises(VerbatimError) as exc:
        bare.compose("scope:a", caller=caller(), purpose="recall",
                     now_us=T0)
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_unsupported_view_kind_reports_capability(store, kernel, synth):
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "quote", "derive"])
    with pytest.raises(VerbatimError) as exc:
        synth.compose(
            "scope:a", caller=caller(), purpose="recall",
            view_kind="grounded_summary", now_us=T0,
        )
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
    cap = synth.capabilities()
    assert cap.rung == CapabilityRung.IMPLEMENTED
    assert "grounded_summary" in cap.details["unavailable_view_kinds"]


def test_extractive_requires_byte_access(store, kernel, synth):
    """Extractive digest under a metadata-only caller fails honestly —
    it cannot mint quote statements without verified bytes."""
    with store.tx() as conn:
        bootstrap(conn, ("scope:a",))
        seed(conn, store, "scope:a", PAYLOAD_A)
        grant_all(conn, "scope:a", "human:alice", ["read", "derive"])
    with pytest.raises(VerbatimError) as exc:
        synth.compose("scope:a", caller=caller(), purpose="recall",
                      now_us=T0)
    assert exc.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
