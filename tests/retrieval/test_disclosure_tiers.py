"""Progressive disclosure tiers + expansion handles (SPEC_V4 §32;
V4-32.05/32.09).

Real ``Store.create`` fixtures (v3 schema) with direct-SQL seeding — the
same fixture shape as ``tests/retrieval/v3/test_retrieval_v3.py``.

Covered:
  - L0/L1/L2 projections: metadata only / bounded summary / exact bytes.
  - Preview tiers never read ``source_revisions.payload`` (proven by
    corrupting the payload so the exact path fails while previews still
    serve).
  - Monotone budget tightening across tiers (helper + end-to-end).
  - Honest L2 degradation without ``quote`` (metadata view + warning).
  - Expansion refs: roundtrip, caller binding, expiry, forgery,
    revocation, purge/quarantine, revision drift, epoch drift.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import RecallRequestV3
from verbatim.governance import (
    create_grant,
    register_principal,
    revoke_grant,
    seed_purposes,
)
from verbatim.retrieval.v3 import recall_v3
from verbatim.retrieval.v3.recall import expand_item
from verbatim.retrieval.v3 import pack as _pack
from verbatim.storage.store import Store


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


# ---------------------------------------------------------------------
# seeding helpers (mirrors tests/retrieval/v3/test_retrieval_v3.py)
# ---------------------------------------------------------------------

def seed_scope(conn, scope_id, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, scope_id, pid="human:alice", purposes=("recall",),
              verbs=("read", "quote")):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs=set(verbs),
        issuer_id=pid, purposes=list(purposes),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1",
               provenance="direct_user"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC',?,'{}')",
        (source_id, payload, _h(payload), provenance),
    )


def add_span(conn, span_id, source_id, start, end, rev=1):
    payload = bytes(
        conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, rev),
        ).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, rev, start, end, _h(payload[start:end])),
    )


def add_fts(conn, claim_id, rev, scope_id, text, gen):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text, gen,
               state="active", recorded_from=1, rev=1,
               subject=None, predicate=None):
    """Claim + source + span + FTS row. ``subject``/``predicate`` seed the
    claim's structured metadata — the preview-tier summary surface."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject, predicate, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,?,?,NULL,?,NULL,NULL,NULL)",
        (claim_id, rev, state, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def request(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def items_of(result):
    return [i for p in result.packs for i in p.items]


def bodies_of(result):
    """Parsed item bodies — the serialized JSON inside the wrapper."""
    out = []
    for i in items_of(result):
        text = i.text
        start = text.find("{")
        end = text.rfind("}")
        assert start != -1 and end > start, text
        out.append(json.loads(text[start:end + 1]))
    return out


def texts_of(result):
    return [i.text for p in result.packs for i in p.items]


# ---------------------------------------------------------------------
# L0 — navigational projection
# ---------------------------------------------------------------------

def test_l0_ships_identifiers_and_locators_only(store):
    """L0 items carry claim id, span locator, lifecycle, provenance and a
    bound ``expand`` ref — never summary text, never payload (V4-32.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "the release ships on friday", _gen(store),
                   subject="release", predicate="ships_on")
    res = recall_v3(store, request("when does the release ship"),
                    detail_tier="l0")
    assert res.capabilities["detail_tier"] == "l0"
    bodies = bodies_of(res)
    assert bodies
    for b in bodies:
        assert b["detail_tier"] == "l0"
        assert b["claim_id"] == "cl1"
        assert b["span"]["span_id"] == "sp1"
        assert b["span"]["source_id"] == "src1"
        assert b["lifecycle"] == "active"
        assert "text" not in b or b.get("text") in (None, "")
        assert "summary" not in b
        assert b["expand"].startswith("ex_")
    # raw payload bytes appear nowhere in the serialized items
    assert not any("ships on friday" in t for t in texts_of(res))


def test_l1_summary_and_expandable_marks(store):
    """L1 ships the bounded read-level summary (structured subject/
    predicate + FTS-proven matched terms), stable locators, and the
    expansion ref — payload bytes still absent (V4-32.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "the deploy window is tuesday morning", _gen(store),
                   subject="deploy_window", predicate="is")
    res = recall_v3(store, request("when is the deploy window"),
                    detail_tier="l1")
    assert res.capabilities["detail_tier"] == "l1"
    bodies = bodies_of(res)
    assert bodies
    exact = [b for b in bodies if b.get("claim_id") == "cl1"]
    assert exact
    for b in exact:
        assert b["detail_tier"] == "l1"
        assert b["text"] == ""            # no quotation bytes
        assert b.get("expandable") is True
        assert b["expand"].startswith("ex_")
        summary = b.get("summary", "")
        # the bounded summary is claim metadata + verified term coverage
        assert "deploy_window" in summary or "matched:" in summary
    assert not any("tuesday morning" in t for t in texts_of(res))


def test_l2_exact_bytes_under_quote(store):
    """L2 under read+quote ships the byte-exact slice — unchanged exact
    path (digest-verified, ``quote_authorized``)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "exact quotation body", _gen(store))
    res = recall_v3(store, request("exact quotation"),
                    detail_tier="l2")
    assert res.capabilities["detail_tier"] == "l2"
    assert any("exact quotation body" in t for t in texts_of(res))
    bodies = bodies_of(res)
    assert any(
        b.get("quote_authorized") is True and b["detail_tier"] == "l2"
        for b in bodies
    )


def test_l2_without_quote_ships_metadata_view_with_warning(store):
    """Read-only caller asking L2: every exact item renders its
    read-authorized metadata view (``quote_withheld``) and a tier-level
    warning names the collapse — never silently exact, never error-named
    (V4-32.05 honest degradation)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", verbs=("read",))
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "verbatim secret payload text", _gen(store))
    res = recall_v3(store, request("verbatim secret payload"),
                    detail_tier="l2")
    assert items_of(res)
    assert "detail_tier_collapsed:quote_unauthorized" in res.warnings
    bodies = bodies_of(res)
    for b in bodies:
        assert b.get("quote_authorized") is False
        assert b.get("quote_withheld") is True
        assert b["text"] == ""
    assert not any(
        "verbatim secret payload text" in t for t in texts_of(res)
    )


def test_preview_tiers_never_read_payload(store):
    """Destroy the stored payload after seeding: L0/L1 still serve their
    projections (no byte reads, no digest verification), while the L2
    digest check fails closed with ``STORE_CORRUPT`` — never tainted or
    partial content (C03, V4-32.05)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "tainted evidence body", _gen(store),
                   subject="tainted", predicate="is")
        # Corrupt the stored bytes so any read/verify path fails.
        conn.execute(
            "UPDATE source_revisions SET payload = X'DEADBEEF'"
            " WHERE source_id = 'src1'"
        )
    res0 = recall_v3(store, request("tainted evidence"),
                     detail_tier="l0")
    assert items_of(res0)
    res1 = recall_v3(store, request("tainted evidence"),
                     detail_tier="l1")
    assert items_of(res1)
    # L2 digest-verifies before quoting: corruption fails closed —
    # the recall aborts rather than emitting tainted or partial content
    # (C03 fail-closed integrity).
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("tainted evidence"), detail_tier="l2")
    assert exc.value.code is ErrorCode.STORE_CORRUPT


def test_tier_budgets_monotone_helper(store):
    """``detail_budget_limits`` is monotone non-decreasing in the tier
    and never widens the caller's own limits (V4-32.05)."""
    for items in (1, 2, 4, 8, 16, 32):
        for bytes_ in (256, 640, 1024, 2048, 6000, 24000):
            l0 = _pack.detail_budget_limits(
                _pack.DetailTier.L0, items, bytes_)
            l1 = _pack.detail_budget_limits(
                _pack.DetailTier.L1, items, bytes_)
            l2 = _pack.detail_budget_limits(
                _pack.DetailTier.L2, items, bytes_)
            assert l0 <= l1 <= l2
            assert l0[0] <= items and l0[1] <= bytes_
            assert l1[0] <= items and l1[1] <= bytes_
            assert l2 == (items, bytes_)


def test_tier_budgets_monotone_end_to_end(store):
    """Serialized bytes delivered grow monotonically with the declared
    tier for the same content (V4-32.03 measured on real bytes)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        for i in range(6):
            seed_claim(conn, f"cl{i}", "sA", f"src{i}", f"sp{i}",
                       f"shared budget phrase variant {i}", _gen(store),
                       subject="budget", predicate=f"variant_{i}")
    sizes = {}
    for tier in ("l0", "l1", "l2"):
        res = recall_v3(store, request("shared budget phrase"),
                        detail_tier=tier)
        sizes[tier] = sum(
            len(i.text.encode("utf-8")) for i in items_of(res)
        )
        assert items_of(res)
    assert sizes["l0"] <= sizes["l1"] <= sizes["l2"]


def test_detail_tier_invalid_token_rejected(store):
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "tier probe text", _gen(store))
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("tier probe"), detail_tier="l9")
    assert exc.value.code is ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# expansion handles (V4-32.09)
# ---------------------------------------------------------------------

def _expand_ref(result, claim_id="cl1"):
    for b in bodies_of(result):
        if b.get("claim_id") == claim_id and b.get("expand"):
            return b["expand"]
    raise AssertionError("no expand ref on claimed item")


def test_expand_roundtrip_releases_bytes_under_quote(store):
    """An L1 preview item's expand ref, re-asked at L2 by the same
    caller under unchanged authority, yields the exact bytes
    (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "expandable exact body", _gen(store))
    res = recall_v3(store, request("expandable exact"),
                    detail_tier="l1")
    ref = _expand_ref(res)
    expanded = expand_item(
        store, ref, caller_id="human:alice", detail_tier="l2"
    )
    bodies = bodies_of(expanded)
    assert any(
        b["text"] == "expandable exact body"
        and b.get("quote_authorized") is True
        for b in bodies
    )
    # the re-delivery recorded fresh exposure rows under new handles
    with store.read() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM influence WHERE object_id = 'cl1'"
        ).fetchone()[0]
    assert n >= 2  # original delivery + expansion


def test_expand_at_l0_stays_metadata(store):
    """Expansion honors the requested tier — an L0 expand ships the
    navigational projection, not bytes."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "l0 expansion body", _gen(store))
    res = recall_v3(store, request("l0 expansion"), detail_tier="l1")
    ref = _expand_ref(res)
    expanded = expand_item(
        store, ref, caller_id="human:alice", detail_tier="l0"
    )
    bodies = bodies_of(expanded)
    assert bodies
    for b in bodies:
        assert b["detail_tier"] == "l0"
        assert not b.get("text")
    assert not any("l0 expansion body" in t for t in texts_of(expanded))


def test_expand_denies_other_caller(store):
    """The ref is bound to the original caller — a different principal
    gets the indistinguishable denial (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_auth(conn, "sA", pid="human:bob")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "caller bound body", _gen(store))
    res = recall_v3(store, request("caller bound"), detail_tier="l1")
    ref = _expand_ref(res)
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, ref, caller_id="human:bob")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_forged_token(store):
    """A tampered token fails MAC verification against the stored
    influence row (the row, not the token, supplies the binding)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "forgery probe body", _gen(store))
    res = recall_v3(store, request("forgery probe"), detail_tier="l1")
    ref = _expand_ref(res)
    forged = ref[:-1] + ("A" if ref[-1] != "A" else "B")
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, forged, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_expired_token(store):
    """Time-limited: expansion windows close at ``delivery.created_us +
    _EXPAND_TTL_US`` — enforced from the stored influence row, never the
    token (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "expiry probe body", _gen(store))
    res = recall_v3(store, request("expiry probe"), detail_tier="l1")
    ref = _expand_ref(res)
    with pytest.raises(VerbatimError) as exc:
        expand_item(
            store, ref, caller_id="human:alice",
            now=now_us() + _pack._EXPAND_TTL_US + 1_000_000,
        )
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_after_grant_revocation(store):
    """Reauthorization is current-state: revoke the grant between recall
    and expansion and the same ref denies — the bound epoch moved and
    the verb check fails regardless (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        gid = seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "revocation expansion body", _gen(store))
    res = recall_v3(store, request("revocation expansion"),
                    detail_tier="l1")
    ref = _expand_ref(res)
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, ref, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_after_quarantine(store):
    """A hold opened between delivery and expansion withholds the
    object — expansion is a fresh admissibility check, not a replay of
    the old verdict (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "quarantine expansion body", _gen(store))
    res = recall_v3(store, request("quarantine expansion"),
                    detail_tier="l1")
    ref = _expand_ref(res)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO quarantine(object_kind,object_id,revision,"
            "scope_id,state,opened_event) VALUES('claim','cl1',1,'sA',"
            "'pending',2)",
        )
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, ref, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_after_purge_suppression(store):
    """A suppressing purge between delivery and expansion denies —
    erased content cannot be resurrected by a held token."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "purge expansion body", _gen(store))
    res = recall_v3(store, request("purge expansion"), detail_tier="l1")
    ref = _expand_ref(res)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO purges(purge_id,selection_digest,scope_id,state,"
            "requested_us,approved_us) VALUES('pg1',X'00','sA',"
            "'suppressed',1,2)",
        )
        conn.execute(
            "INSERT INTO purge_targets(purge_id,object_kind,object_id)"
            " VALUES('pg1','claim','cl1')",
        )
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, ref, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_after_revision_drift(store):
    """The ref pins the delivered revision — a corrected claim (new
    head) denies the old token; the caller must re-ask (V4-32.09)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "drift probe body", _gen(store))
    res = recall_v3(store, request("drift probe"), detail_tier="l1")
    ref = _expand_ref(res)
    with store.tx() as conn:
        conn.execute(
            "UPDATE claim_revisions SET recorded_until = 9"
            " WHERE claim_id = 'cl1' AND revision = 1"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "condition_json,recorded_from,recorded_until,perspective_id,"
            "freshness) VALUES('cl1',2,'active',NULL,9,NULL,NULL,NULL)",
        )
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, ref, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_expand_denies_undelivered_object(store):
    """A well-formed token over an object that was never delivered to
    this caller has no matching influence row — expansion denies
    (delivered-object binding lives in the rows, not the string)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "undelivered object body", _gen(store))
    res = recall_v3(store, request("undelivered object"),
                    detail_tier="l1")
    ref = _expand_ref(res)
    # Rebind the token's object id (hex field) to a claim never
    # delivered — no influence row exists for it under this caller.
    parts = ref[len("ex_"):].split(".")
    parts[1] = "cl_undelivered".encode("utf-8").hex()
    fake = "ex_" + ".".join(parts)
    with pytest.raises(VerbatimError) as exc:
        expand_item(store, fake, caller_id="human:alice")
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
