"""SPEC_V4 §07/§08 — single-authority parity across public surfaces.

V4-07.07/07.12/07.13 + C86/C87: every surface that can reach evidence
bytes (the v3 facade's explicit archive lane, the kernel's verified read,
the export bundle, the inspect report, the MCP transport) must agree with
the kernel's enforcement — the same grants decide, the same suppression
and quarantine predicates withhold, the same digest verification fails
closed, and the same denial is indistinguishable everywhere. A
compatibility wrapper that answers differently is a second authority.

Fixtures run a real ``Store.create`` (the TestStore shim lacks the v3/v4
tables) and seed evidence through the real ``VerbatimV3`` write path so
``source_envelopes``, grants, and purposes exist exactly as production
capture leaves them.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from verbatim import governance, security
from verbatim.api import open_store
from verbatim.api_v3 import VerbatimV3
from verbatim.api_v3.mcp import MCP_V3_BOUND_VERBS, McpV3Server
from verbatim.config import config_from_mapping
from verbatim.core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    VerbatimError,
)
from verbatim.core.types_v3 import Verb
from verbatim.core.types_v4 import EvidenceLocator
from verbatim.export import export_scope
from verbatim.governance import CallerV3
from verbatim.host import LocalHost
from verbatim.kernel import Kernel
from verbatim.purge import suppress
from verbatim.storage.repos import SourcesRepo, SpansRepo
from verbatim.storage.resolver import require_store_path, resolve_store_path
from verbatim.storage.store import Store

SCOPE = "scope:parity"
AGENT = "agent-parity"
OTHER = "agent-other"
INTRUDER = "agent-intruder"
PAYLOAD = b"deploy moved to friday at noon"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "parity.db"))
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """One captured source (+ whole-payload span) on a bootstrapped scope."""
    facade = VerbatimV3(store)
    facade.issue_capture_authorization(AGENT, SCOPE, granted_by=AGENT)
    sid = facade.capture_submitted(AGENT, SCOPE, PAYLOAD.decode("utf-8"))
    SpansRepo(store).insert("sp-parity", sid, 1, 0, len(PAYLOAD), "parity-1")
    return facade, sid


def _browse_items(facade, query="", principal_id=AGENT):
    res = facade.browse_evidence(SCOPE, query, principal_id=principal_id)
    return [item for pack in res.packs for item in pack.items], res


def _kernel_lease(store, principal, verb, refs):
    with store.read() as conn:
        return Kernel(store).resolve_access(
            conn,
            CallerV3(principal_id=principal),
            verb,
            "recall",
            [SCOPE],
            refs,
        )


def _kernel_slice(store, lease, object_id, revision, start, end):
    with store.read() as conn:
        return Kernel(store).read_verified(
            conn,
            lease,
            [EvidenceLocator(
                object_id=object_id,
                revision=revision,
                start_byte=start,
                end_byte=end,
            )],
        )[0]


def _export_caller(granted=True, profile_id="v3"):
    return CallerContext(
        profile_id=profile_id,
        principal_id="op-parity",
        grants=frozenset({GrantKind.EXPORT}) if granted else frozenset(),
    )


def _mcp_call(store, principal, tool, arguments):
    facade = VerbatimV3(store, bound_verbs=MCP_V3_BOUND_VERBS)
    server = McpV3Server(facade, CallerV3(principal_id=principal))
    return server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
    )


# ---------------------------------------------------------------------------
# positive parity — one authorized answer everywhere
# ---------------------------------------------------------------------------


def test_authorized_bytes_agree_across_surfaces(store, seeded):
    """Same evidence → same bytes on the lane, the export bundle, and the
    kernel's verified read; each surface carries its own authenticating
    receipt (pack handle / payload_sha256 / slice digest)."""
    facade, sid = seeded

    items, _ = _browse_items(facade, "deploy")
    assert len(items) == 1
    assert items[0].text == PAYLOAD.decode("utf-8")

    bundle = export_scope(store, SCOPE)
    rev = bundle["sources"][0]["revisions"][0]
    shipped = base64.b64decode(rev["payload_b64"])
    assert shipped == PAYLOAD
    assert rev["payload_sha256"] == hashlib.sha256(PAYLOAD).hexdigest()

    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert not lease.denied
    slice_ = _kernel_slice(store, lease, sid, 1, 0, len(PAYLOAD))
    assert slice_.data == PAYLOAD
    assert slice_.verification == "verified"
    assert slice_.digest  # slice-level receipt exists

    # Span-level verified read agrees on the same excerpt.
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("span", "sp-parity", 1)])
    assert not lease.denied
    span_slice = _kernel_slice(store, lease, "sp-parity", 1, 0, len(PAYLOAD))
    assert span_slice.data == PAYLOAD
    assert span_slice.verification == "verified"

    # The inspect surface answers lineage only — never bytes.
    report = facade.inspect_evidence(sid, principal_id=AGENT)
    assert report["source_id"] == sid
    assert PAYLOAD.decode("utf-8") not in json.dumps(report, default=str)


def test_read_without_quote_is_metadata_only_everywhere(store, seeded):
    """V4-08.02: ``read`` alone ships no exact bytes — the lane emits
    empty-text items and the kernel answers ``metadata_only`` slices."""
    facade, sid = seeded
    with store.tx() as conn:
        governance.create_grant(
            conn,
            scope_id=SCOPE,
            principal_id=OTHER,
            verbs=["read"],
            purposes=["recall"],
            issuer_id=AGENT,
        )

    items, res = _browse_items(facade, "deploy", principal_id=OTHER)
    assert items and all(item.text == "" for item in items)
    assert "quote_not_granted" in res.warnings

    lease = _kernel_lease(store, OTHER, Verb.READ, [("source", sid, 1)])
    assert not lease.denied
    slice_ = _kernel_slice(store, lease, sid, 1, 0, len(PAYLOAD))
    assert slice_.data == b""
    assert slice_.verification == "metadata_only"


# ---------------------------------------------------------------------------
# denial parity — indistinguishable everywhere, no second evaluator
# ---------------------------------------------------------------------------


def test_denial_is_uniform_across_surfaces(store, seeded):
    facade, sid = seeded

    for fn in (
        lambda: facade.browse_evidence(SCOPE, "deploy", principal_id=INTRUDER),
        lambda: facade.recall(SCOPE, "deploy", principal_id=INTRUDER),
        lambda: facade.inspect_evidence(sid, principal_id=INTRUDER),
    ):
        with pytest.raises(VerbatimError) as ei:
            fn()
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    lease = _kernel_lease(store, INTRUDER, Verb.QUOTE, [("source", sid, 1)])
    assert lease.denied  # denied lease, never a partial answer

    with pytest.raises(VerbatimError) as ei:
        export_scope(store, SCOPE, caller=_export_caller(granted=False))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN

    resp = _mcp_call(
        store, INTRUDER, "v3_recall", {"scope_id": SCOPE, "query": "deploy"}
    )
    assert resp["error"]["code"] == -32000
    assert resp["error"]["message"] == "NOT_FOUND_OR_UNAUTHORIZED"
    # The denial is opaque — it must not echo the queried identifiers.
    assert SCOPE not in resp["error"]["message"]


def test_mcp_caller_cannot_be_rebound_by_arguments(store, seeded):
    """V3-48.05/C86: the transport's bound caller is fixed; argument-level
    identity minting is a typed denial, not a silent rebind."""
    facade, sid = seeded
    resp = _mcp_call(
        store,
        INTRUDER,
        "v3_recall",
        {"scope_id": SCOPE, "query": "deploy", "principal_id": AGENT},
    )
    assert "error" in resp
    assert resp["error"]["message"] == "VALIDATION"
    # And the bound-but-unauthorized caller still gets the same opaque
    # denial as every other surface.
    resp = _mcp_call(
        store, INTRUDER, "v3_inspect", {"source_id": sid}
    )
    assert resp["error"]["message"] == "NOT_FOUND_OR_UNAUTHORIZED"


# ---------------------------------------------------------------------------
# suppression parity — tombstones withhold on every surface
# ---------------------------------------------------------------------------


def test_revision_scoped_suppression_withholds_everywhere(store, seeded):
    """A ``source_revision`` tombstone is the kernel's predicate — the
    facade lane and export must withhold the same revision (V4-07.07)."""
    facade, sid = seeded
    suppress(store, SCOPE, [("source", sid, 1)], actor=AGENT)

    items, res = _browse_items(facade, "deploy")
    assert items == []

    bundle = export_scope(store, SCOPE)
    assert bundle["sources"] == []

    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert lease.denied
    # The span hangs off the suppressed revision — denied too.
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("span", "sp-parity", 1)])
    assert lease.denied


def test_source_scoped_suppression_withholds_everywhere(store, seeded):
    facade, sid = seeded
    suppress(store, SCOPE, [("source", sid)], actor=AGENT)

    items, _ = _browse_items(facade, "deploy")
    assert items == []
    bundle = export_scope(store, SCOPE)
    assert bundle["sources"] == []
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert lease.denied
    # inspect is honest metadata: suppressed state is reported, not hidden.
    report = facade.inspect_evidence(sid, principal_id=AGENT)
    assert report["suppressed"] is True


def test_quarantine_hold_withholds_on_every_surface(store, seeded):
    """A pending hold on the covering ``source_envelope`` cascades to the
    revision on the lane, in export, and in the kernel (V3-14.10)."""
    facade, sid = seeded
    with store.read() as conn:
        env_id = conn.execute(
            "SELECT envelope_id FROM source_envelopes WHERE source_id = ?",
            (sid,),
        ).fetchone()[0]
    with store.tx() as conn:
        security.open_quarantine(
            conn,
            ("source_envelope", env_id, 1),
            ["attack_risk:blocked"],
            [],
            scope_id=SCOPE,
        )

    items, res = _browse_items(facade, "deploy")
    assert items == []
    assert "held_evidence_withheld" in res.warnings

    bundle = export_scope(store, SCOPE)
    assert bundle["sources"] == []

    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert lease.denied


def test_physically_purged_bytes_are_absent_everywhere(store, seeded):
    """An emptied payload (post-purge scrub state) is 'absent' on all
    surfaces — the lane skips it, export omits it, the kernel denies."""
    facade, sid = seeded
    with store.tx() as conn:
        # Mimic the purge scrub exactly: empty bytes, resealed digest.
        conn.execute(
            "UPDATE source_revisions SET payload = X'', payload_hmac = ?"
            " WHERE source_id = ? AND revision = 1",
            (store.hmac(b""), sid),
        )

    items, _ = _browse_items(facade, "deploy")
    assert items == []
    bundle = export_scope(store, SCOPE)
    assert bundle["sources"] == []
    # The lease mints on existence; the verified read is where absent
    # bytes deny — the same indistinguishable denial a missing row gets.
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert not lease.denied
    with pytest.raises(VerbatimError) as ei:
        _kernel_slice(store, lease, sid, 1, 0, 1)
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# integrity parity — forged bytes fail closed, never ship
# ---------------------------------------------------------------------------


def test_tampered_payload_fails_closed_on_every_surface(store, seeded):
    """V4-08.03/C03: an in-place payload rewrite must be detected by the
    repo, the facade lane, export, and the kernel — the typed
    STORE_CORRUPT is the same answer everywhere (no fallback path)."""
    facade, sid = seeded
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = ? AND revision = 1",
            (b"forged replacement bytes", sid),
        )

    with pytest.raises(VerbatimError) as ei:
        SourcesRepo(store).payload(sid, 1)
    assert ei.value.code == ErrorCode.STORE_CORRUPT

    with pytest.raises(VerbatimError) as ei:
        facade.browse_evidence(SCOPE, "forged", principal_id=AGENT)
    assert ei.value.code == ErrorCode.STORE_CORRUPT

    with pytest.raises(VerbatimError) as ei:
        export_scope(store, SCOPE)
    assert ei.value.code == ErrorCode.STORE_CORRUPT

    # The lease still mints (integrity is a read-time check); the
    # verified read then fails closed on the forged bytes.
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("source", sid, 1)])
    assert not lease.denied
    with pytest.raises(VerbatimError) as ei:
        _kernel_slice(store, lease, sid, 1, 0, 10)
    assert ei.value.code == ErrorCode.STORE_CORRUPT

    # Span-level: forged parent payload breaks the excerpt binding too.
    lease = _kernel_lease(store, AGENT, Verb.QUOTE, [("span", "sp-parity", 1)])
    assert not lease.denied
    with pytest.raises(VerbatimError) as ei:
        _kernel_slice(store, lease, "sp-parity", 1, 0, len(PAYLOAD))
    assert ei.value.code == ErrorCode.STORE_CORRUPT

    # inspect never touches bytes — it still answers metadata honestly.
    report = facade.inspect_evidence(sid, principal_id=AGENT)
    assert report["source_id"] == sid


# ---------------------------------------------------------------------------
# store-resolver identity parity — one policy chooses the store
# ---------------------------------------------------------------------------


def test_one_resolver_policy_decides_for_all_surfaces(tmp_path):
    """V4-07.10/C86: ``open_store`` (engine/CLI/SDK/provider entry) and the
    resolver agree bit-for-bit on which store a directory names — same
    adoption, same conflict, same operator decision channels."""
    profile = "prof-parity"
    data = tmp_path / "data"
    data.mkdir()
    host = LocalHost(
        profile_id=profile, principal_id="alice", conversation_id="c1"
    )
    cfg = config_from_mapping(
        {"capture": {"enabled": True, "user_messages": True}}
    )

    eng = open_store(str(data), cfg, host, create=True)
    try:
        resolved = require_store_path(
            resolve_store_path(str(data), profile_id=profile, create=True)
        )
        assert eng.store.path == resolved
        assert resolved.endswith(f"{profile}.db")
    finally:
        eng.close()

    # Both conventions on disk → the same typed conflict on both paths.
    Store.create(str(data / "v3.db")).close()
    with pytest.raises(VerbatimError) as ei:
        require_store_path(
            resolve_store_path(str(data), profile_id=profile)
        )
    assert ei.value.code == ErrorCode.STORE_CONFLICT
    with pytest.raises(VerbatimError) as ei:
        open_store(str(data), cfg, host)
    assert ei.value.code == ErrorCode.STORE_CONFLICT

    # The operator decision channel resolves identically.
    decided = require_store_path(
        resolve_store_path(str(data), profile_id=profile, prefer="v3")
    )
    eng = open_store(str(data), cfg, host, prefer_store="v3")
    try:
        assert eng.store.path == decided
    finally:
        eng.close()
