"""I1 counterevidence-first manifests + I3 sufficiency pack mode tests
(SPEC_V4_5 §03/§05; V45-03.01–03.04, V45-05.01–05.04; D01/D02/D05).

The manifest is opt-in (``request.manifest="counterevidence_first"``);
default requests are byte-identical v3 deliveries — the same store,
the same seed, the same query. Everything asserted here rides
``capabilities["manifest"]`` / ``capabilities["sufficiency"]`` or the
serialized item bodies — the inspectable surface, never internals.
"""

from __future__ import annotations

import json

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import PackKind, RecallRequestV3
from verbatim.retrieval.manifest import (
    COUNTEREVIDENCE_FIRST,
    PACK_MODE_SUFFICIENCY,
    normalize_manifest,
    normalize_pack_mode,
)
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.store import Store

from tests.retrieval.v3.test_retrieval_v3 import (
    _TEST_HMAC_KEY,
    _gen,
    items_of,
    request,
    seed_auth,
    seed_claim,
    seed_scope,
)


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v3.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


def _bodies(result):
    out = []
    for p in result.packs:
        for i in p.items:
            text = i.text
            start = text.find("{")
            end = text.rfind("}")
            out.append(json.loads(text[start:end + 1]))
    return out


def _ids(result):
    return {i.handle.object_id for i in items_of(result)}


def _seed_conflict(conn, gen, scope="sA", gid="cg1",
                   a="clA", b="clB", atext=None, btext=None):
    seed_claim(conn, a, scope, f"src{a}", f"sp{a}",
               atext or "deploy flag mode is ON", gen)
    seed_claim(conn, b, scope, f"src{b}", f"sp{b}",
               btext or "deploy flag mode is OFF", gen)
    conn.execute(
        "INSERT INTO conflict_groups(group_id,scope_id,status)"
        f" VALUES('{gid}','{scope}','open')",
    )
    conn.execute(
        "INSERT INTO conflict_members(group_id,claim_id)"
        f" VALUES('{gid}','{a}'),('{gid}','{b}')",
    )


# ---------------------------------------------------------------------
# opt-in / validation
# ---------------------------------------------------------------------

def test_default_no_manifest(store):
    """Default request: no manifest capability, no role fields, and the
    pack bytes are identical to the pre-flag delivery."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_conflict(conn, _gen(store))
    res = recall_v3(store, request("deploy flag mode"))
    assert "manifest" not in res.capabilities
    assert res.capabilities.get("pack_mode") == "standard"
    for b in _bodies(res):
        assert "role" not in b
    res2 = recall_v3(store, request("deploy flag mode"))
    assert [i.text for p in res.packs for i in p.items] == [
        i.text for p in res2.packs for i in p.items
    ]


def test_manifest_optin_present(store):
    """manifest='counterevidence_first' attaches the inspectable doc."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_conflict(conn, _gen(store))
    res = recall_v3(
        store, request("deploy flag mode",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    m = res.capabilities.get("manifest")
    assert m is not None
    assert m["manifest"] == "counterevidence_first"
    assert m["producer_kind"] == "deterministic_pack"
    for key in ("supporting_refs", "contrary_refs", "as_of",
                "coverage", "insufficiency_labels",
                "unresolved_obligations", "tokens"):
        assert key in m


def test_unknown_manifest_fails_closed(store):
    """Unknown tokens are caller errors, never silent defaults."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1", "probe", _gen(store))
    with pytest.raises(VerbatimError) as exc:
        recall_v3(store, request("probe", manifest="mystery_mode"))
    assert exc.value.code is ErrorCode.VALIDATION
    with pytest.raises(VerbatimError):
        normalize_manifest("bogus")
    with pytest.raises(VerbatimError):
        normalize_pack_mode("bogus")
    with pytest.raises(VerbatimError):
        RecallRequestV3(
            query="q", scope_id="sA", caller_id="human:alice",
            purpose="recall", manifest="bogus",
        )


def test_off_positions_default(store):
    """none/standard/empty are the off positions — current behavior."""
    assert normalize_manifest(None) is None
    assert normalize_manifest("none") is None
    assert normalize_manifest("standard") is None
    assert normalize_manifest("") is None
    assert normalize_pack_mode(None) == "standard"


# ---------------------------------------------------------------------
# contrary evidence — included, omitted, labeled
# ---------------------------------------------------------------------

def test_contrary_included_in_conflict_pack(store):
    """Both conflict sides land in contrary_refs as included — the
    declared slot, not an item that happens to appear (D01)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_conflict(conn, _gen(store))
    res = recall_v3(
        store, request("deploy flag mode",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    m = res.capabilities["manifest"]
    included = {
        r["object_id"] for r in m["contrary_refs"]
        if r["disposition"] == "included"
    }
    assert {"clA", "clB"} <= included
    roles = {b.get("role") for b in _bodies(res)}
    assert "contrary" in roles
    # verified bytes and semantic support are distinct fields (V45-03.03)
    for r in m["contrary_refs"]:
        assert "verified_bytes" in r and "semantic_support" in r


def test_contrary_omitted_with_reason(store):
    """Budget-excluded contrary evidence is named with its reason —
    never silently dropped (V45-03.01/03.04)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        _seed_conflict(conn, gen)
        # a state-asserting singleton that outranks the conflict pair
        seed_claim(conn, "clSum", "sA", "srcS", "spS",
                   "deploy flag mode status is ON confirmed", gen)
    res = recall_v3(
        store,
        request("deploy flag mode status", max_items=1,
                manifest=COUNTEREVIDENCE_FIRST),
    )
    m = res.capabilities["manifest"]
    omitted = [
        r for r in m["contrary_refs"] if r["disposition"] == "omitted"
    ]
    # the 2-item conflict group cannot fit in one slot: named + reason
    assert omitted
    assert all(r["reason"] for r in omitted)
    labels = set(m["insufficiency_labels"])
    assert "contrary_evidence_budget_exceeded" in labels or \
        "contrary_evidence_incomplete" in labels or \
        any(r["reason"] == "budget_exceeded" for r in omitted)


def test_corrects_edge_marks_corrected_contrary(store):
    """A ``corrects`` edge: the correction is supporting, the corrected
    stale side is contrary (declared role, not inferred)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clNew", "sA", "srcN", "spN",
                   "limit is now 42", gen)
        seed_claim(conn, "clOld", "sA", "srcO", "spO",
                   "limit was 37", gen)
        conn.execute(
            "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
            "target_kind,target_id,edge_type,created_event) VALUES('e1',"
            "'sA','claim','clNew','claim','clOld','corrects',1)",
        )
    res = recall_v3(
        store, request("what is the limit",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    m = res.capabilities["manifest"]
    contrary = {
        r["object_id"] for r in m["contrary_refs"]
    }
    assert "clOld" in contrary
    supporting = {r["object_id"] for r in m["supporting_refs"]}
    assert "clNew" in supporting


def test_unauthorized_contrary_member_inaccessible_label(store):
    """A conflict member in an unauthorized scope: the group cannot
    ship half — the caller-visible side is named, the hidden side is a
    presence-level label only (V45-03.02). No id/title/count leaks."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "disputed mode alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "disputed mode beta", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clA'),('cg1','clB')",
        )
    res = recall_v3(
        store, request("disputed mode alpha",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    m = res.capabilities["manifest"]
    # the hidden member is NEVER named anywhere on the manifest
    blob = json.dumps(m)
    assert "clB" not in blob
    assert "srcB" not in blob and "spB" not in blob
    # its existence surfaces only as a presence-level label
    assert m["inaccessible_contrary"] is True
    assert "contrary_evidence_inaccessible" in m["insufficiency_labels"]


def test_no_contrary_declares_clean_coverage(store):
    """Uncontradicted authorized view: the label says so explicitly —
    'no contrary evidence' is itself on the record."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "the release ships friday", _gen(store))
    res = recall_v3(
        store, request("when does the release ship",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    m = res.capabilities["manifest"]
    assert "no_authorized_contrary_evidence" in m["insufficiency_labels"]
    assert m["contrary_refs"] == []
    assert m["inaccessible_contrary"] is False


def test_abstained_result_still_manifests(store):
    """An insufficiency answer still declares its contrary accounting —
    the manifest covers the abstained branch too."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "ordinary evidence", _gen(store))
    res = recall_v3(
        store,
        request("fix for src/definitely/missing.py",
                manifest=COUNTEREVIDENCE_FIRST),
    )
    m = res.capabilities["manifest"]
    assert res.abstained or "missing_hard_identifiers" in res.warnings
    assert m["coverage"]["missing_identifiers"]
    assert "missing_hard_identifiers" in m["insufficiency_labels"]


def test_manifest_labels_ride_warnings(store):
    """Insufficiency labels also surface in result warnings — callers
    who never read the manifest still see contrary_* signals."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clA", "sA", "srcA", "spA",
                   "disputed flag alpha", gen)
        seed_claim(conn, "clB", "sB", "srcB", "spB",
                   "disputed flag beta", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clA'),('cg1','clB')",
        )
    res = recall_v3(
        store, request("disputed flag alpha",
                       manifest=COUNTEREVIDENCE_FIRST)
    )
    assert any(
        w.startswith("contrary_") for w in res.warnings
    )


# ---------------------------------------------------------------------
# sufficiency pack mode (V45-05)
# ---------------------------------------------------------------------

def _seed_history(conn, gen, scope="sA"):
    """Identifier fix claim + condition-bearing claim + conflicts_with
    safety caveat + depth claims sharing query terms."""
    seed_claim(conn, "clFix", scope, "srcF", "spF",
               "the fix for src/mod/x.py is a token rebind", gen)
    seed_claim(conn, "clCond", scope, "srcC", "spC",
               "rollout constraint for src/mod/x.py: prod-eu only", gen,
               condition=json.dumps({"eq": ["env", "prod-eu"]}))
    seed_claim(conn, "clSafe", scope, "srcSa", "spSa",
               "safety: the rebind drops sessions — drain first", gen)
    conn.execute(
        "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
        "target_kind,target_id,edge_type,created_event) VALUES('es',"
        f"'{scope}','claim','clFix','claim','clSafe','conflicts_with',1)",
    )
    for j in range(3):
        seed_claim(conn, f"clDep{j}", scope, f"srcD{j}", f"spD{j}",
                   f"token investigation note {j} with long filler"
                   f" history of the component incidents and fixes",
                   gen)


def test_sufficiency_coverage_declared(store):
    """The declared coverage rule and its outcome ride capabilities —
    required identifiers covered, mandatory groups admitted first."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_history(conn, _gen(store))
    res = recall_v3(
        store,
        request("fix src/mod/x.py token",
                pack_mode=PACK_MODE_SUFFICIENCY, max_items=8),
    )
    cov = res.capabilities.get("sufficiency")
    assert cov is not None
    assert cov["pack_mode"] == "sufficiency"
    assert "src/mod/x.py" in cov["required_identifiers"]
    assert "src/mod/x.py" in cov["covered_identifiers"]
    assert cov["coverage_tier"] == "l2"
    assert cov["depth_tier"] == "l0"
    # the identifier claim ships full detail; depth ships stubs
    by_claim = {b.get("claim_id"): b for b in _bodies(res)}
    assert by_claim["clFix"]["detail_tier"] == "l2"
    assert by_claim["clCond"]["detail_tier"] == "l2"  # condition-bearing
    assert by_claim["clSafe"]["detail_tier"] == "l2"  # conflicts dep
    for j in range(3):
        assert by_claim[f"clDep{j}"]["detail_tier"] == "l0"
        assert by_claim[f"clDep{j}"].get("expand")


def test_sufficiency_fewer_tokens_same_items(store):
    """Same admitted set, fewer delivered tokens — the saving is
    deferred depth, not dropped evidence (V45-05.03)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_history(conn, _gen(store))
    q = "fix src/mod/x.py token"
    flat = recall_v3(store, request(q, max_items=8))
    prog = recall_v3(
        store, request(q, pack_mode=PACK_MODE_SUFFICIENCY,
                       max_items=8),
    )
    assert _ids(flat) == _ids(prog)
    flat_tok = sum(p.tokens for p in flat.packs)
    prog_tok = sum(p.tokens for p in prog.packs)
    assert prog_tok < flat_tok
    # identifiers preserved under the savings
    assert "src/mod/x.py" in json.dumps(
        [b for b in _bodies(prog)]
    )


def test_sufficiency_expansion_roundtrip(store):
    """An L0 stub's bound expand ref re-asks to full text — depth is
    deferred, not lost (V45-05.01/05.02)."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_history(conn, _gen(store))
    from verbatim.retrieval.v3.recall import expand_item
    res = recall_v3(
        store,
        request("fix src/mod/x.py token",
                pack_mode=PACK_MODE_SUFFICIENCY, max_items=8),
    )
    ref = None
    for b in _bodies(res):
        if b.get("detail_tier") == "l0" and b.get("expand"):
            ref = b["expand"]
            break
    assert ref is not None
    expanded = expand_item(
        store, ref, caller_id="human:alice", detail_tier="l2"
    )
    assert any(b.get("text") for b in _bodies(expanded))


def test_standard_mode_no_sufficiency(store):
    """pack_mode unset → no sufficiency capability; flat tier only."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_history(conn, _gen(store))
    res = recall_v3(
        store, request("fix src/mod/x.py token", max_items=8)
    )
    assert "sufficiency" not in res.capabilities
    assert res.capabilities.get("pack_mode") == "standard"
    for b in _bodies(res):
        assert b.get("detail_tier") == "l2"


def test_manifest_and_sufficiency_compose(store):
    """Both opt-ins compose: contrary-first ordering + coverage-first
    admission + manifest doc, one request."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        _seed_conflict(conn, _gen(store))
        seed_claim(conn, "clI", "sA", "srcI", "spI",
                   "note about deploy flag mode docs/a.py", _gen(store))
    res = recall_v3(
        store,
        request("deploy flag mode docs/a.py",
                manifest=COUNTEREVIDENCE_FIRST,
                pack_mode=PACK_MODE_SUFFICIENCY, max_items=8),
    )
    assert res.capabilities["manifest"]["manifest"] == (
        "counterevidence_first"
    )
    assert res.capabilities["sufficiency"]["pack_mode"] == "sufficiency"
    cov = res.capabilities["sufficiency"]
    assert "docs/a.py" in cov["required_identifiers"]
