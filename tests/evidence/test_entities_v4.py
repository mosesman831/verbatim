"""Entity + reference resolution and multilingual coverage — SPEC_V4 §17
(V4-17.01–17.08) with scenario C38 as the ambiguity anchor.

Contract under test:

- entity ids are scope-bound; a lookup never crosses an authorized
  partition boundary (V4-17.01);
- exact identifier/entity-id matches take precedence (V4-17.02);
- same-name entities stay DISTINCT — matching surfaces every competing
  candidate as ``ambiguous``; nothing merges on name/alias similarity;
- pronouns/deictics resolve only inside a declared context window and
  stay ``unresolved`` otherwise — never a guessed winner (V4-17.03);
- UTF-8 text survives capture → index → recall byte-exact across CJK,
  accented Latin, and mixed scripts (V4-17.05–17.07). The index is
  unicode61 — no trigram tokenizer — so unsegmented-script recall is the
  bounded folded-substring path, and that is what these tests pin.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from verbatim.core.types_v3 import QueryClass, RecallRequestV3
from verbatim.evidence import entities as ent
from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.retrieval.query import analyze
from verbatim.retrieval.v3 import lanes as _lanes
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage.repos import EntitiesRepo
from verbatim.storage.store import Store

_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "e.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "e.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


def _scope(conn, sid, principal="p1", conv="c1"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (sid, "prof", principal, "ws", conv, "conversation"),
    )


def _auth(conn, sid, pid="human:alice"):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    create_grant(
        conn, scope_id=sid, principal_id=pid,
        verbs={"read", "quote"}, issuer_id=pid, purposes=["recall"],
    )


def _seed_claim(conn, claim_id, scope_id, text, gen, state="active"):
    src, sp = f"src-{claim_id}", f"sp-{claim_id}"
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (src, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,"
        "metadata_json) VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (src, payload, _h(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (sp, src, 1, 0, len(payload), _h(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "condition_json,recorded_from,recorded_until,perspective_id,"
        "freshness) VALUES(?,1,?,NULL,1,NULL,NULL,NULL)",
        (claim_id, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,1,?,'primary',NULL)",
        (claim_id, sp),
    )
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,1,?,?)",
        (claim_id, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def _entity(conn, eid, sid, label, kind="person"):
    conn.execute(
        "INSERT INTO entities(entity_id,scope_id,kind,label,created_event)"
        " VALUES(?,?,?,?,1)",
        (eid, sid, kind, label),
    )
    return eid


def _alias(conn, eid, normalized):
    conn.execute(
        "INSERT INTO entity_aliases(entity_id,normalized_alias,"
        "source_span_id,approval_event) VALUES(?,?,NULL,1)",
        (eid, normalized),
    )


def _link(conn, claim_id, eid, role="subject"):
    conn.execute(
        "INSERT OR IGNORE INTO claim_entities(claim_id,entity_id,role,"
        "span_id) VALUES(?,?,?,NULL)",
        (claim_id, eid, role),
    )


def _req(query="q", scope_id="sA", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id="human:alice", **kw)


def _ids(res):
    return {i.handle.object_id for p in res.packs for i in p.items}


# ---------------------------------------------------------------------
# V4-17.02 — precedence + ambiguity honesty
# ---------------------------------------------------------------------

def test_exact_entity_id_wins(store):
    """An entity id is the strongest identifier — resolution binds it
    directly, with exact precedence over every label collision."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
        _entity(conn, "ent-2", "sA", "Jordan")
    with store.read() as conn:
        res = ent.resolve_entity(conn, ("sA",), "ent-1")
    assert res.status == ent.STATUS_RESOLVED
    assert res.entity_id == "ent-1"
    assert res.candidates[0].via == "entity_id"


def test_same_name_entities_never_collapse(store):
    """Two 'Jordan' entities in one scope → ambiguous with BOTH recorded —
    never a silently chosen winner (V4-17.02, C38)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-j1", "sA", "Jordan")
        _entity(conn, "ent-j2", "sA", "Jordan")
        _alias(conn, "ent-j1", "jordan")
        _alias(conn, "ent-j2", "jordan")
    with store.read() as conn:
        res = ent.resolve_entity(conn, ("sA",), "Jordan")
        assert res.status == ent.STATUS_AMBIGUOUS
        assert res.entity_id is None
        assert {c.entity_id for c in res.candidates} == {"ent-j1", "ent-j2"}
        # Alias matching preserves the same ambiguity.
        res2 = ent.resolve_entity(conn, ("sA",), "jordan")
        assert res2.status == ent.STATUS_AMBIGUOUS
        assert {c.entity_id for c in res2.candidates} == {
            "ent-j1", "ent-j2"}
        # The write path itself retains both identities.
        repo = EntitiesRepo(store)
        assert sorted(repo.all_for_label(conn, "sA", "Jordan")) == [
            "ent-j1", "ent-j2"]


def test_unique_alias_resolves(store):
    """A single exact alias match resolves — evidence, not similarity."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan Mattera")
        _alias(conn, "ent-1", ent.normalize_entity_text("jordy"))
    with store.read() as conn:
        res = ent.resolve_entity(conn, ("sA",), "jordy")
    assert res.status == ent.STATUS_RESOLVED
    assert res.entity_id == "ent-1"
    assert res.candidates[0].via == "alias"


def test_unknown_name_unresolved(store):
    """No exact identifier/label/alias → unresolved, never a fuzzy guess."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
    with store.read() as conn:
        res = ent.resolve_entity(conn, ("sA",), "Jordanna")
        assert res.status == ent.STATUS_UNRESOLVED
        assert res.entity_id is None


def test_resolution_is_scope_bound(store):
    """An entity in another scope is invisible — partitions bind
    identities (V4-17.01), so the same label across scopes never
    cross-resolves."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _scope(conn, "sB", principal="p2", conv="c2")
        _auth(conn, "sA")
        _entity(conn, "ent-a", "sA", "Jordan")
        _entity(conn, "ent-b", "sB", "Jordan")
    with store.read() as conn:
        res = ent.resolve_entity(conn, ("sA",), "Jordan")
        assert res.status == ent.STATUS_RESOLVED
        assert res.entity_id == "ent-a"
        # Asked for only sB's partition: sA's entity is not a candidate.
        res_b = ent.resolve_entity(conn, ("sB",), "Jordan")
        assert res_b.entity_id == "ent-b"
        # No authorized scope → nothing resolves.
        res_none = ent.resolve_entity(conn, (), "Jordan")
        assert res_none.status == ent.STATUS_UNRESOLVED


def test_context_disambiguates_without_merging(store):
    """A declared context window can narrow an ambiguous name to the one
    candidate in-window — competitors stay recorded (V4-17.02)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-j1", "sA", "Jordan")
        _entity(conn, "ent-j2", "sA", "Jordan")
    with store.read() as conn:
        res = ent.resolve_entity(
            conn, ("sA",), "Jordan", context_entity_ids=("ent-j2",))
    assert res.status == ent.STATUS_RESOLVED
    assert res.entity_id == "ent-j2"
    assert res.reason == "context_disambiguated"
    assert {c.entity_id for c in res.candidates} == {"ent-j1", "ent-j2"}


# ---------------------------------------------------------------------
# V4-17.03 — pronoun/deictic resolution only in declared windows
# ---------------------------------------------------------------------

def test_pronoun_without_window_stays_unresolved(store):
    """'he' with no declared context window → UNRESOLVED. There is no
    salience fallback to guess from (V4-17.03)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
    with store.read() as conn:
        res = ent.resolve_reference(conn, ("sA",), "he")
        assert res.status == ent.STATUS_UNRESOLVED
        assert res.reason == "no_context_window"
        assert res.entity_id is None


def test_pronoun_unique_in_window_resolves(store):
    """One entity inside the declared window → resolved at contextual
    confidence (recorded via=context_window)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
        _seed_claim(conn, "cl1", "sA", "jordan reviewed the diff",
                    _gen(store))
        _link(conn, "cl1", "ent-1", "subject")
    with store.read() as conn:
        res = ent.resolve_reference(
            conn, ("sA",), "he", context_claim_ids=("cl1",))
        assert res.status == ent.STATUS_RESOLVED
        assert res.entity_id == "ent-1"
        assert res.candidates[0].via == "context_window"
        assert res.confidence == 0.5
        assert "subject" in res.candidates[0].roles


def test_pronoun_ambiguous_window_stays_ambiguous(store):
    """Two entities in the window → ambiguous, both recorded — the
    resolver never picks the more recent/salient one (V4-17.03)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
        _entity(conn, "ent-2", "sA", "Riley")
        _seed_claim(conn, "cl1", "sA", "jordan and riley paired",
                    _gen(store))
        _link(conn, "cl1", "ent-1", "subject")
        _link(conn, "cl1", "ent-2", "subject")
    with store.read() as conn:
        res = ent.resolve_reference(
            conn, ("sA",), "they", context_claim_ids=("cl1",))
        assert res.status == ent.STATUS_AMBIGUOUS
        assert res.entity_id is None
        assert {c.entity_id for c in res.candidates} == {
            "ent-1", "ent-2"}


def test_pronoun_window_is_scope_bound(store):
    """A window whose claims live outside the authorized scopes supplies
    no candidates — context never leaks across partitions."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _scope(conn, "sB", principal="p2", conv="c2")
        _auth(conn, "sA")
        _entity(conn, "ent-b", "sB", "Jordan")
        _seed_claim(conn, "clB", "sB", "jordan elsewhere",
                    _gen(store))
        _link(conn, "clB", "ent-b")
    with store.read() as conn:
        res = ent.resolve_reference(
            conn, ("sA",), "he", context_claim_ids=("clB",))
        assert res.status == ent.STATUS_UNRESOLVED


def test_text_references_mixed(store):
    """resolve_text_references reports pronouns AND evidence-bearing
    names; non-referential tokens produce no noise."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-1", "sA", "Jordan")
        _seed_claim(conn, "cl1", "sA", "jordan shipped it",
                    _gen(store))
        _link(conn, "cl1", "ent-1")
    with store.read() as conn:
        out = ent.resolve_text_references(
            conn, ("sA",), "jordan said he shipped it",
            context_claim_ids=("cl1",))
    by_ref = {r.reference.casefold(): r for r in out}
    assert by_ref["jordan"].status == ent.STATUS_RESOLVED
    assert by_ref["jordan"].entity_id == "ent-1"
    assert by_ref["he"].status == ent.STATUS_RESOLVED
    # No declared window → the same pronoun stays unresolved.
    out2 = ent.resolve_text_references(conn, ("sA",), "he shipped it")
    by2 = {r.reference.casefold(): r for r in out2}
    assert by2["he"].status == ent.STATUS_UNRESOLVED


def test_speaker_roles_stay_distinct(store):
    """Subject vs mention roles and entity kinds survive resolution —
    the resolver never fuses a subject with a quoted third party
    (V4-17.04)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-user", "sA", "Alice", kind="person")
        _entity(conn, "ent-quoted", "sA", "Alice", kind="person")
        _seed_claim(conn, "cl1", "sA", "alice asked about the deploy",
                    _gen(store))
        _link(conn, "cl1", "ent-user", "subject")
        _link(conn, "cl1", "ent-quoted", "mention")
    with store.read() as conn:
        res = ent.resolve_reference(
            conn, ("sA",), "she", context_claim_ids=("cl1",))
        assert res.status == ent.STATUS_AMBIGUOUS
        roles = {c.entity_id: set(c.roles) for c in res.candidates}
        assert roles["ent-user"] == {"subject"}
        assert roles["ent-quoted"] == {"mention"}


# ---------------------------------------------------------------------
# recall-level entity behavior (lane priority + C38 ambiguity)
# ---------------------------------------------------------------------

def test_entity_id_lane_priority(store):
    """entity_ids on the request take the exact-id lane: only the
    addressed entity's claims return (V4-17.02 + C38)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        _entity(conn, "ent-j1", "sA", "Jordan")
        _entity(conn, "ent-j2", "sA", "Jordan")
        _seed_claim(conn, "clJ1", "sA", "the schema was redesigned", gen)
        _seed_claim(conn, "clJ2", "sA", "the ledger was audited", gen)
        _link(conn, "clJ1", "ent-j1")
        _link(conn, "clJ2", "ent-j2")
    res = recall_v3(store, _req("jordan", entity_ids=("ent-j1",)))
    assert _ids(res) == {"clJ1"}


def test_ambiguous_label_recall_never_collapses(store):
    """A bare same-name label query may surface both or abstain — never
    a silent single-winner collapse (C38)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        for eid in ("ent-j1", "ent-j2"):
            _entity(conn, eid, "sA", "Jordan")
            _alias(conn, eid, "jordan")
        _seed_claim(conn, "clJ1", "sA", "the schema was redesigned", gen)
        _seed_claim(conn, "clJ2", "sA", "the ledger was audited", gen)
        _link(conn, "clJ1", "ent-j1")
        _link(conn, "clJ2", "ent-j2")
    res = recall_v3(store, _req("jordan"))
    ids = _ids(res)
    assert ids != {"clJ1"} and ids != {"clJ2"}
    if ids:
        assert res.abstained or {"clJ1", "clJ2"} <= ids


# ---------------------------------------------------------------------
# V4-17.05–17.07 — UTF-8 retention + multilingual capture → recall
# ---------------------------------------------------------------------

def test_cjk_capture_recall(store):
    """CJK text captures, indexes, and recalls byte-exact. unicode61
    cannot segment it, so the bounded folded-substring path serves the
    query — substring recall, pinned honestly."""
    text = "東京での会議は水曜日に移動しました"
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-jp", "sA", text, _gen(store))
    res = recall_v3(store, _req("水曜日"))
    assert _ids(res) == {"cl-jp"}
    item_texts = [i.text for p in res.packs for i in p.items]
    assert any("水曜日" in t for t in item_texts)
    # Original bytes are the stored payload — no normalization rewrite.
    with store.read() as conn:
        payload = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ?",
            ("src-cl-jp",),
        ).fetchone()[0]
    assert payload == text.encode("utf-8")


def test_accented_capture_recall(store):
    """Accented text indexes through unicode61's diacritic folding — both
    the accented and the ASCII-folded query recall it."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-fr", "sA",
                    "mon éditeur préféré est néovim", _gen(store))
    for q in ("éditeur", "editeur", "néovim"):
        res = recall_v3(store, _req(q))
        assert "cl-fr" in _ids(res), q


def test_mixed_script_capture_recall(store):
    """Mixed-script text: the ASCII segment hits the index, the CJK
    segment hits the folded substring path, and the bytes stay exact."""
    text = "deploy 東京都 cluster au siège"
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-mix", "sA", text, _gen(store))
    for q in ("cluster", "東京都", "siège"):
        res = recall_v3(store, _req(q))
        assert "cl-mix" in _ids(res), q
    with store.read() as conn:
        payload = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ?",
            ("src-cl-mix",),
        ).fetchone()[0]
    assert payload == text.encode("utf-8")


def test_entity_labels_preserve_original_bytes(store):
    """Normalization is a matching projection only — the stored label is
    the original text, never rewritten (V4-17.05)."""
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _entity(conn, "ent-jose", "sA", "José Álvarez")
        _alias(conn, "ent-jose", ent.normalize_entity_text("Jose Alvarez"))
    with store.read() as conn:
        label = conn.execute(
            "SELECT label FROM entities WHERE entity_id = 'ent-jose'",
        ).fetchone()[0]
        assert label == "José Álvarez"
        # Folded query resolves to it through the alias…
        res = ent.resolve_entity(conn, ("sA",), "jose alvarez")
        assert res.status == ent.STATUS_RESOLVED
        # …and a normalized label match competes without rewriting bytes.
        res2 = ent.resolve_entity(conn, ("sA",), "José Álvarez")
        assert res2.status == ent.STATUS_RESOLVED
        assert res2.entity_id == "ent-jose"
