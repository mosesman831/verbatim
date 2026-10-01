"""V2 export/import: portable bundles, remapped ownership, erasure fencing."""

from __future__ import annotations

import base64
import copy

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.export import export_scope, import_bundle
from verbatim.purge import execute_purge, plan_purge, suppress
from verbatim.storage.repos_v2 import ErasureRepo
from tests.privacy.test_v2_purge import (
    make_caller,
    make_evidence,
    make_scope,
    make_store,
    qrow,
)


def test_export_manifest_counts_and_payloads(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)

    bundle = export_scope(store, scope)
    man = bundle["manifest"]
    assert man["format_version"] == 1
    assert man["scope_id"] == ev["scope_id"]
    assert man["counts"]["sources"] == 1
    assert man["counts"]["spans"] == 1
    assert man["counts"]["claims"] == 1

    src = bundle["sources"][0]
    assert base64.b64decode(src["revisions"][0]["payload_b64"]) == ev["payload"]
    assert src["digest"]

    claim = bundle["claims"][0]
    assert claim["revisions"][0]["state"] == "active"
    assert claim["revisions"][0]["evidence"][0]["span_id"] == ev["span_id"]


def test_export_excludes_suppressed(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    suppress(store, scope, [("claim", ev["claim_id"])], actor="alice")

    bundle = export_scope(store, scope)
    assert bundle["manifest"]["counts"]["claims"] == 0
    assert all(c["claim_id"] != ev["claim_id"] for c in bundle["claims"])


def test_export_omits_quarantined_source_revision_and_claim(tmp_path):
    """A ``("source", sid, rev)`` quarantine hold withholds the revision's
    payload — and the span/claim records its evidence chain taints
    (V3-14.10). Held bytes never reach the bundle."""
    import verbatim.security as security

    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    with store.tx() as conn:
        security.open_quarantine(
            conn, ("source", ev["source_id"], 1),
            ["attack_risk:blocked"], [], scope_id=ev["scope_id"],
        )

    bundle = export_scope(store, scope)
    assert bundle["sources"] == []
    assert bundle["spans"] == []
    assert bundle["claims"] == []
    assert ev["payload"].decode() not in str(bundle)


def test_export_omits_source_envelope_hold(tmp_path):
    """A hold on the covering ``source_envelopes`` row withholds the
    payload the same way — the join resolves (source_id, revision)."""
    import verbatim.security as security

    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO source_envelopes(envelope_id, source_id, revision,"
            " scope_id, envelope_kind) VALUES (?, ?, 1, ?, 'user_message')",
            ("env-1", ev["source_id"], ev["scope_id"]),
        )
        security.open_quarantine(
            conn, ("source_envelope", "env-1", 1),
            ["attack_risk:blocked"], [], scope_id=ev["scope_id"],
        )

    bundle = export_scope(store, scope)
    assert bundle["sources"] == []
    assert bundle["claims"] == []


def test_export_omits_revision_scoped_suppression(tmp_path):
    """A ``("source_revision", "sid:rev")`` tombstone — the
    suppress→execute window — withholds the revision's payload before the
    purge physically scrubs it."""
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)
    suppress(store, scope, [("source", ev["source_id"], 1)], actor="alice")

    bundle = export_scope(store, scope)
    assert bundle["sources"] == []
    assert bundle["claims"] == []


def test_export_caller_requires_export_grant(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    ev = make_evidence(store, scope)

    no_grant = make_caller(grants=())
    with pytest.raises(VerbatimError) as ei:
        export_scope(store, scope, caller=no_grant)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

    # caller holding only the EXPORT grant succeeds
    from verbatim.core.types import CallerContext, GrantKind

    exporter = CallerContext(
        profile_id="prof", principal_id="alice",
        grants=frozenset({GrantKind.EXPORT}),
    )
    bundle = export_scope(store, scope, caller=exporter)
    assert bundle["manifest"]["counts"]["claims"] == 1


def test_export_writes_file(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    make_evidence(store, scope)
    out = tmp_path / "bundle.json"
    bundle = export_scope(store, scope, str(out))
    assert out.exists()
    import json

    on_disk = json.loads(out.read_text())
    assert on_disk["manifest"]["scope_id"] == bundle["manifest"]["scope_id"]
    # octal 600 permissions
    assert (out.stat().st_mode & 0o777) == 0o600


def test_roundtrip_remaps_ownership_and_preserves_bytes(tmp_path):
    store = make_store(tmp_path, "origin.db")
    scope = make_scope()
    ev = make_evidence(store, scope)
    bundle = export_scope(store, scope)

    target = make_store(tmp_path, "target.db")
    new_scope = make_scope(principal="carol", conversation="conv9")
    receipt = import_bundle(target, None, bundle, new_scope, actor="carol")
    assert receipt["imported"]["sources"] == 1
    assert receipt["imported"]["claims"] == 1
    assert receipt["skipped_erased"] == []

    id_map = receipt["id_map"]
    new_src = id_map["source"][ev["source_id"]]
    new_claim = id_map["claim"][ev["claim_id"]]
    new_span = id_map["span"][ev["span_id"]]
    assert new_src != ev["source_id"]

    with target.read() as conn:
        # ownership remapped to the target scope
        src_row = qrow(conn, "SELECT scope_id, origin, external_id FROM sources"
                             " WHERE source_id = ?", (new_src,))
        assert src_row["scope_id"] == receipt["scope_id"]
        assert src_row["origin"] == "verbatim-import"
        assert ev["source_id"] in src_row["external_id"]
        # evidence bytes preserved exactly
        rev = qrow(conn, "SELECT payload, provenance, metadata_json FROM"
                         " source_revisions WHERE source_id = ? AND revision = 1",
                   (new_src,))
        assert bytes(rev["payload"]) == ev["payload"]
        assert rev["provenance"] == "legacy_import"
        import json as _j

        meta = _j.loads(rev["metadata_json"])
        assert meta["import_provenance"]["origin_source_id"] == ev["source_id"]
        # claim + span + evidence links remapped
        claim_row = qrow(conn, "SELECT scope_id FROM claims WHERE claim_id = ?",
                         (new_claim,))
        assert claim_row["scope_id"] == receipt["scope_id"]
        ev_link = qrow(conn, "SELECT span_id FROM claim_evidence"
                             " WHERE claim_id = ? AND revision = 1", (new_claim,))
        assert ev_link["span_id"] == new_span


def test_import_is_idempotent(tmp_path):
    store = make_store(tmp_path, "origin.db")
    scope = make_scope()
    make_evidence(store, scope)
    bundle = export_scope(store, scope)

    target = make_store(tmp_path, "target.db")
    new_scope = make_scope(principal="carol", conversation="conv9")
    first = import_bundle(target, None, bundle, new_scope, actor="carol")
    second = import_bundle(target, None, bundle, new_scope, actor="carol")
    assert second["duplicate"] is True
    assert not any(second["imported"].values())
    assert second["id_map"]["source"] == first["id_map"]["source"]

    with target.read() as conn:
        assert qrow(conn, "SELECT COUNT(*) AS n FROM sources")["n"] == 1
        assert qrow(conn, "SELECT COUNT(*) AS n FROM claims")["n"] == 1


def test_reimport_does_not_resurrect_erased(tmp_path):
    """Purge an imported object, then replay the same bundle: the operations
    ledger replays the old mapping rather than inserting a fresh row, so the
    purged claim stays erased."""
    store = make_store(tmp_path, "origin.db")
    scope = make_scope()
    ev = make_evidence(store, scope)
    bundle = export_scope(store, scope)

    target = make_store(tmp_path, "target.db")
    new_scope = make_scope(principal="carol", conversation="conv9")
    receipt1 = import_bundle(target, None, bundle, new_scope, actor="carol")
    new_claim = receipt1["id_map"]["claim"][ev["claim_id"]]

    plan = plan_purge(target, receipt1["scope_id"], [("claim", new_claim)],
                      actor="carol")
    execute_purge(target, plan["purge_id"])
    with target.read() as conn:
        assert ErasureRepo(target).is_erased(
            conn, receipt1["scope_id"], "claim", new_claim
        )

    receipt2 = import_bundle(target, None, bundle, new_scope, actor="carol")
    assert receipt2["duplicate"] is True
    with target.read() as conn:
        from verbatim.core.lifecycle import read_claim_head

        head = read_claim_head(conn, new_claim)
        assert head.state.value == "erased"


def test_purged_import_fences_origin_id_in_later_bundles(tmp_path):
    """Purging an imported object also fences its origin id, so a *different*
    bundle carrying that origin id cannot resurrect it."""
    store = make_store(tmp_path, "origin.db")
    scope = make_scope()
    ev = make_evidence(store, scope)
    bundle1 = export_scope(store, scope)

    target = make_store(tmp_path, "target.db")
    new_scope = make_scope(principal="carol", conversation="conv9")
    receipt1 = import_bundle(target, None, bundle1, new_scope, actor="carol")
    new_claim = receipt1["id_map"]["claim"][ev["claim_id"]]
    plan = plan_purge(target, receipt1["scope_id"], [("claim", new_claim)],
                      actor="carol")
    execute_purge(target, plan["purge_id"])
    with target.read() as conn:
        # the purge recorded BOTH the local id and the origin id
        assert ErasureRepo(target).is_erased(
            conn, receipt1["scope_id"], "claim", ev["claim_id"]
        )

    # a second, different bundle from the same origin still carries the claim
    make_evidence(store, scope, text="second note for later")
    bundle2 = export_scope(store, scope)
    assert bundle2["manifest"]["records_sha256"] != bundle1["manifest"]["records_sha256"]
    receipt2 = import_bundle(target, None, bundle2, new_scope, actor="carol")
    assert any(
        s["object_id"] == ev["claim_id"] for s in receipt2["skipped_erased"]
    )
    with target.read() as conn:
        # the erased claim was not recreated — one erased skeleton plus the
        # second, legitimately new claim from bundle2
        assert qrow(conn, "SELECT COUNT(*) AS n FROM claims")["n"] == 2
        other_claim_ids = [
            c["claim_id"] for c in bundle2["claims"]
            if c["claim_id"] != ev["claim_id"]
        ]
        assert len(other_claim_ids) == 1
        new_claim2 = receipt2["id_map"]["claim"].get(other_claim_ids[0])
        assert new_claim2 is not None and new_claim2 != new_claim
        assert qrow(conn, "SELECT COUNT(*) AS n FROM claims"
                        " WHERE claim_id = ?", (new_claim2,))["n"] == 1


def test_import_fenced_by_erasure_ledger(tmp_path):
    """A bundle carrying an id this profile already purged must not
    resurrect it — the restore fence keys on the object's origin id."""
    store = make_store(tmp_path, "origin.db")
    scope = make_scope()
    ev = make_evidence(store, scope)
    bundle = export_scope(store, scope)

    target = make_store(tmp_path, "target.db")
    new_scope = make_scope(principal="carol", conversation="conv9")
    # Simulate the restore scenario: the target profile already erased this
    # object (by its origin id), then a backup bundle tries to bring it back.
    from verbatim.storage.repos import ensure_scope

    with target.tx() as conn:
        sid = ensure_scope(target, conn, new_scope)
        ErasureRepo(target).record(conn, sid, "claim", ev["claim_id"])

    receipt = import_bundle(target, None, bundle, new_scope, actor="carol")
    assert receipt["imported"]["claims"] == 0
    assert any(
        s["object_kind"] == "claim" and s["object_id"] == ev["claim_id"]
        for s in receipt["skipped_erased"]
    )
    with target.read() as conn:
        assert qrow(conn, "SELECT COUNT(*) AS n FROM claims")["n"] == 0
        # evidence objects the fence did not cover still import
        assert qrow(conn, "SELECT COUNT(*) AS n FROM sources")["n"] == 1


def test_import_rejects_tampered_bundle(tmp_path):
    store = make_store(tmp_path)
    scope = make_scope()
    make_evidence(store, scope)
    bundle = export_scope(store, scope)

    bad = copy.deepcopy(bundle)
    bad["claims"][0]["revisions"][0]["state"] = "disputed"  # digest now stale
    target = make_store(tmp_path, "target.db")
    with pytest.raises(VerbatimError) as ei:
        import_bundle(target, None, bad, make_scope(principal="d"), actor="d")
    assert ei.value.code == ErrorCode.VALIDATION

    no_manifest = copy.deepcopy(bundle)
    del no_manifest["manifest"]
    with pytest.raises(VerbatimError):
        import_bundle(target, None, no_manifest, make_scope(principal="d"), actor="d")


def test_import_scope_isolation(tmp_path):
    store = make_store(tmp_path, "origin.db")
    scope_a = make_scope(principal="alice", conversation="convA")
    scope_b = make_scope(principal="bob", conversation="convB")
    ev_a = make_evidence(store, scope_a, text="alice private note")
    make_evidence(store, scope_b, text="bob private note")

    bundle_a = export_scope(store, scope_a)
    assert bundle_a["manifest"]["counts"]["claims"] == 1
    src_ids = [s["source_id"] for s in bundle_a["sources"]]
    assert src_ids == [ev_a["source_id"]]
    # bob's evidence never leaks into alice's export
    all_ids = str(bundle_a)
    assert "bob private note" not in all_ids
