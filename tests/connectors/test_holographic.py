"""holographic connector — holographic-export-v1 JSON/JSONL (V4-48.02,
V4-48.10).

The format is a DECLARED interchange, not a verified mirror of the
provider's private store — provenance must show it, and facts without
originals must land as imported assertions, never byte-exact evidence.
"""

from __future__ import annotations

import json

import pytest

from verbatim.core.types import ErrorCode, VerbatimError

from tests.connectors.conftest import PRINCIPAL, provision, qrows


def _export_jsonl(path, header, items):
    lines = []
    if header is not None:
        lines.append(json.dumps(header))
    lines += [json.dumps(i) for i in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pull(service, path, scope_id="scopeA", **kw):
    return service.pull(
        "holographic",
        {"path": str(path)},
        scope_id=scope_id,
        principal_id=PRINCIPAL,
        **kw,
    )


def test_jsonl_imports_episodes_entities_claims(
    store, service, tmp_path
):
    provision(store)
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {
            "format": "holographic-export-v1",
            "provider": "holographic",
            "exported_us": 1_700_000_000_000_000,
            "source_scope": "remote:conv-7",
        },
        [
            {
                "id": "ep-1",
                "kind": "episode",
                "text": "we deployed the fix at noon",
                "original": True,
                "actor": "bob",
                "event_us": 1_700_000_000_000_000,
            },
            {
                "id": "ent-1",
                "kind": "entity",
                "summary": "Deploy Bot — a CI agent",
                "refs": ["ep-1"],
            },
            {
                "id": "cl-1",
                "kind": "claim",
                "text": "the fix shipped in release 4.2",
                "refs": ["ep-1"],
                "unknown_field": {"provider": "specific"},
            },
        ],
    )
    report = _pull(service, export)
    assert report.state == "complete"
    assert report.inserted == 3
    assert report.rejected == 0
    # V4-48.02: only ep-1 declared an original — the entity summary and
    # the claim are imported ASSERTIONS, not byte-exact evidence.
    # external_id lives on sources, not envelopes — join via source_id.
    with store.read() as conn:
        joined = qrows(
            conn,
            "SELECT se.*, s.external_id FROM source_envelopes se"
            " JOIN sources s ON s.source_id = se.source_id"
            " WHERE se.scope_id='scopeA'",
        )
    by_ext = {r["external_id"]: r for r in joined}
    assert by_ext["ep-1"]["trust_class"] == "external_content"
    assert by_ext["ent-1"]["trust_class"] == "imported"
    assert by_ext["cl-1"]["trust_class"] == "imported"
    for ext, expect_assert in (
        ("ep-1", False),
        ("ent-1", True),
        ("cl-1", True),
    ):
        meta = json.loads(by_ext[ext]["metadata_json"])
        assert meta["imported_assertion"] is expect_assert
        assert meta["original_present"] is (not expect_assert)
        assert meta["connector_id"] == "holographic"
        assert meta["formats"] == ["holographic-export-v1"]
        assert meta["declared_not_verified"] == [
            "provider_format_fidelity"
        ]
        assert meta["cursor"]
    # Derivation provenance: ent-1/cl-1 cite ep-1.
    assert json.loads(by_ext["ent-1"]["metadata_json"])["source_ids"] == [
        "ep-1"
    ]
    # Remote actor attribution persisted on the original.
    assert by_ext["ep-1"]["actor_principal"] == "bob"
    # Unsupported field surfaced as a loss — never hidden by a clean count.
    assert report.losses == {"unknown_field": 1}
    # Missing actor+event_us provenance is reported, not fabricated.
    assert report.missing_provenance == 2  # ent-1, cl-1
    # Remote namespace → scope mapping reported.
    assert report.scope_mappings == {"remote:conv-7": "scopeA"}


def test_json_object_form(store, service, tmp_path):
    provision(store)
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {
                "format": "holographic-export-v1",
                "items": [
                    {"id": "d1", "kind": "document", "text": "doc body"},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = _pull(service, export)
    assert report.inserted == 1


def test_wrong_format_header_rejected(store, service, tmp_path):
    provision(store)
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {"format": "holographic-export-v9"},
        [{"id": "x", "kind": "episode", "text": "hi"}],
    )
    with pytest.raises(VerbatimError) as ei:
        _pull(service, export)
    assert ei.value.code is ErrorCode.VALIDATION
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0


def test_missing_id_and_kind_are_item_rejections(
    store, service, tmp_path
):
    provision(store)
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {"format": "holographic-export-v1"},
        [
            {"kind": "episode", "text": "no id"},               # missing id
            {"id": "ok", "kind": "episode", "text": "fine"},
            {"id": "bad", "kind": "vibe", "text": "unknown kind"},
            {"id": "nt", "kind": "claim"},                       # no text
        ],
    )
    report = _pull(service, export)
    assert report.inserted == 1
    assert report.rejected == 3
    reasons = {
        i["external_id"]: i["reason"] for i in report.items
    }
    assert reasons["ord:0"] == "missing_id"
    assert reasons["bad"].startswith("unknown_kind")
    assert reasons["nt"] == "missing_text"


def test_cursor_resume_is_ordinal(store, service, tmp_path):
    provision(store)
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {"format": "holographic-export-v1"},
        [
            {"id": "i1", "kind": "claim", "text": "one"},
            {"id": "i2", "kind": "claim", "text": "two"},
            {"id": "i3", "kind": "claim", "text": "three"},
        ],
    )
    r1 = _pull(service, export, batch_size=2)
    assert r1.cursor_after == "2" and r1.inserted == 3
    # Resume from ordinal 1 → only i3 (ordinal 2) is re-enumerated, and
    # the write-channel dedup recognizes it — no second source.
    r2 = _pull(service, export, resume_from="1")
    assert r2.scanned == 1
    assert r2.inserted == 0 and r2.duplicates == 1
    assert r2.items[0]["external_id"] == "i3"
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0] == 3


def test_descriptor_declared_not_verified(service):
    desc = service.describe("holographic")
    assert desc["remote"] is False
    assert "provider_format_fidelity" in desc["declared_not_verified"]
    assert desc["formats"] == ["holographic-export-v1"]


def test_malformed_export_file_is_validation(store, service, tmp_path):
    provision(store)
    export = tmp_path / "export.jsonl"
    export.write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(VerbatimError) as ei:
        _pull(service, export)
    assert ei.value.code is ErrorCode.VALIDATION


def test_d18_dry_run_reports_missing_provenance_per_item(
    store, service, tmp_path
):
    """D18 / V45-09.06: the dry-run names provenance loss per item —
    and after acceptance the receipt carries the same per-item surface,
    while non-original items stay imported assertions."""
    provision(store)
    export = tmp_path / "export.jsonl"
    _export_jsonl(
        export,
        {"format": "holographic-export-v1"},
        [
            {
                "id": "ep-1",
                "kind": "episode",
                "text": "we deployed at noon",
                "original": True,
                "actor": "bob",
                "event_us": 1_700_000_000_000_000,
            },
            {
                "id": "cl-1",
                "kind": "claim",
                "text": "deploy shipped in 4.2",
                "refs": ["ep-1"],
                "unknown_field": "x",
            },
        ],
    )
    # --- dry run: writes nothing, but the preview names the loss ------
    dry = _pull(service, export, dry_run=True)
    assert dry.dry_run is True
    assert dry.missing_provenance == 1
    items = {i["external_id"]: i for i in dry.items}
    assert items["cl-1"]["imported_assertion"] is True
    assert set(items["cl-1"]["missing_provenance"]) == {
        "actor", "event_us",
    }
    assert items["cl-1"]["losses"] == ["unknown_field"]
    assert items["ep-1"]["missing_provenance"] == []
    assert items["ep-1"]["imported_assertion"] is False
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM source_envelopes"
        ).fetchone()[0] == 0

    # --- accepted pull: the receipt keeps the same per-item honesty ---
    rep = _pull(service, export)
    assert rep.state == "complete"
    assert rep.missing_provenance == 1
    got = {i["external_id"]: i for i in rep.items}
    assert got["cl-1"]["imported_assertion"] is True
    assert set(got["cl-1"]["missing_provenance"]) == {
        "actor", "event_us",
    }
    assert got["cl-1"]["losses"] == ["unknown_field"]
    assert got["ep-1"]["imported_assertion"] is False
