"""Behavioral tests for v2 retrieval (SPEC_V2 §14, §16, §17, §25–§31).

Same contract as ``test_retrieval.py``: tests seed the real schema — v1
plus the v2 additive tables and column migrations — and exercise
``search`` end to end over an in-memory SQLite store. No mocks of the SQL
layer; only the cooperative deadline clock is injected.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import sqlite3

import pytest

from verbatim.core.time import parse_rfc3339
from verbatim.core.types import (
    Lifecycle,
    MemoryKind,
    RecallMode,
    RecallRequest,
    Scope,
)
from verbatim.retrieval import search
from verbatim.retrieval.package import serialize_group, serialize_item
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    FTS_TRIGGERS,
)


# Content digests are verified on reads (SpansRepo.text, SourcesRepo.payload,
# Store._verify_content_digests), so seeders must write real profile-keyed
# HMACs, not placeholder bytes. The key is exactly 32 bytes so a real
# ``Store.create`` can also load it from a ``*.key`` file.
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


# ------------------------------------------------------------------ store

class StoreShim:
    """Same minimal store surface as test_retrieval.py's shim."""

    def __init__(self, conn, generation: int = 1, fts_enabled: bool = True):
        self._conn = conn
        self._gen = generation
        self.fts_enabled = fts_enabled

    @contextlib.contextmanager
    def read(self):
        yield self._conn

    def projection_generation(self) -> int:
        return self._gen

    def hmac(self, data: bytes) -> bytes:
        """Same shape as ``Store.hmac`` — verified reads delegate here."""
        return _h(data)


def make_conn(fts5: bool = True, v2: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    if fts5:
        conn.executescript(DDL_FTS5)
        conn.executescript(FTS_TRIGGERS)
    if v2:
        conn.executescript(DDL_V2)
        for stmt in DDL_V2_ALTER.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES('projection_generation','1')"
    )
    return conn


def make_store(conn=None, fts5: bool = True, **kw) -> StoreShim:
    kw.setdefault("fts_enabled", fts5)
    return StoreShim(conn or make_conn(fts5=fts5), **kw)


# ----------------------------------------------------------------- seeders

def us(text: str) -> int:
    return parse_rfc3339(text)


def add_scope(conn, scope_id, principal=None, workspace=None, conv=None,
              vis="conversation", profile="prof"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, workspace, conv, vis),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="user1",
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
    # excerpt_hmac covers exactly the stored byte slice payload[start:end],
    # which may be a sub-range of the revision payload.
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


def add_claim(conn, claim_id, scope_id, created_event=1):
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, created_event),
    )


def add_revision(conn, claim_id, rev=1, state="active", recorded_from=1,
                 recorded_until=None, span_ids=(), intervals=(),
                 condition=None):
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until) VALUES(?,?,?,?,?,?)",
        (claim_id, rev, state, condition, recorded_from, recorded_until),
    )
    for sid in span_ids:
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES(?,?,?,'primary')",
            (claim_id, rev, sid),
        )
    for i, iv in enumerate(intervals):
        if len(iv) == 3:
            from_us, until_us, precision = iv
            kinds = {}
        else:
            (from_us, until_us, precision, kinds) = iv
        cols = ("claim_id,revision,interval_no,from_us,until_us,precision,"
                "basis")
        vals = [claim_id, rev, i, from_us, until_us, precision, "test"]
        if kinds:
            cols += ",start_kind,end_kind,from_us_hi,until_us_hi"
            vals += [
                kinds.get("start_kind", "exact"),
                kinds.get("end_kind", "exact"),
                kinds.get("from_us_hi"),
                kinds.get("until_us_hi"),
            ]
        conn.execute(
            f"INSERT INTO valid_intervals({cols})"
            f" VALUES({','.join('?' * len(vals))})",
            vals,
        )


def add_fts(conn, claim_id, rev, scope_id, text, gen=1):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text,
               **rev_kw):
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    add_claim(conn, claim_id, scope_id,
              created_event=rev_kw.get("recorded_from", 1))
    add_revision(conn, claim_id, span_ids=(span_id,), **rev_kw)
    add_fts(conn, claim_id, rev_kw.get("rev", 1), scope_id, text)


def add_context_group(conn, group_id, scope_id, source_id, members,
                      completeness="complete", recorded_from=1,
                      recorded_until=None):
    """members: [(span_id, role, required)]"""
    conn.execute(
        "INSERT INTO context_groups(group_id,scope_id,source_id,revision,"
        "parser_version,operation_key,completeness,recorded_from,"
        "recorded_until) VALUES(?,?,?,1,'p1',?,?,?,?)",
        (group_id, scope_id, source_id, f"op-{group_id}", completeness,
         recorded_from, recorded_until),
    )
    for i, (span_id, role, required) in enumerate(members):
        conn.execute(
            "INSERT INTO context_members(group_id,span_id,role,required,ord)"
            " VALUES(?,?,?,?,?)",
            (group_id, span_id, role, int(required), i),
        )


READER = Scope(profile_id="prof", principal_id="p1", conversation_id="c1")


def request(query="term", **kw) -> RecallRequest:
    kw.setdefault("scope", READER)
    return RecallRequest(query=query, **kw)


# ------------------------------------------------------- context bundles

def test_evidence_group_includes_required_context_members():
    """A claim whose span sits in a context group is emitted atomically with
    that group's required members — attribution, conditions, antecedents
    (V2-30.01)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = b"Alice announced: the release ships on Friday"
    add_source(conn, "src1", "sA", payload)
    # "the release ships on Friday"
    add_span(conn, "spMain", "src1", 17, len(payload))
    # "Alice announced:"
    add_span(conn, "spAttr", "src1", 0, 16)
    add_claim(conn, "clCtx", "sA")
    add_revision(conn, "clCtx", span_ids=("spMain",))
    add_fts(conn, "clCtx", 1, "sA", "release ships friday")
    add_context_group(
        conn, "cg1", "sA", "src1",
        [("spMain", "primary", 1), ("spAttr", "attribution", 1)],
    )

    result = search(make_store(conn), request("release"))
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.primary_claim_id == "clCtx"
    assert group.complete is True
    texts = [i.text for i in group.items]
    assert "the release ships on Friday" in texts
    assert "Alice announced:" in texts
    ctx = [i for i in group.items if "REQUIRED_CONTEXT" in i.reasons]
    assert len(ctx) == 1
    assert "CONTEXT_ATTRIBUTION" in ctx[0].reasons
    # flattened view carries the same members
    assert {i.span.span_id for i in result.items} == {"spMain", "spAttr"}


def test_sibling_primary_is_not_forced_into_context_bundle():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = b"Sarah wants the Q3 report. I am allergic to shellfish"
    add_source(conn, "src1", "sA", payload)
    add_span(conn, "spReport", "src1", 0, 25)
    add_span(conn, "spAllergy", "src1", 27, len(payload))
    add_claim(conn, "clReport", "sA")
    add_revision(conn, "clReport", span_ids=("spReport",))
    add_fts(conn, "clReport", 1, "sA", "Sarah wants the Q3 report")
    add_claim(conn, "clAllergy", "sA")
    add_revision(conn, "clAllergy", span_ids=("spAllergy",))
    add_fts(conn, "clAllergy", 1, "sA", "I am allergic to shellfish")
    add_context_group(
        conn, "cg1", "sA", "src1",
        [("spReport", "primary", 1), ("spAllergy", "primary", 1)],
    )

    result = search(make_store(conn), request("allergic"))
    assert [item.text for item in result.items] == ["I am allergic to shellfish"]
    assert "context_incomplete" not in result.warnings


def test_required_context_failure_drops_group_atomically():
    """A required context member that cannot be reconstructed suppresses the
    whole group — a claim never ships stripped of its condition (V2-17.14,
    V2-26.09)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = b"when on call: restart the fleet nightly"
    add_source(conn, "src1", "sA", payload)
    add_span(conn, "spMain", "src1", 14, len(payload))
    add_claim(conn, "clCtx", "sA")
    add_revision(conn, "clCtx", span_ids=("spMain",))
    add_fts(conn, "clCtx", 1, "sA", "restart fleet nightly")
    # the 'condition' member's span exists but its payload is empty —
    # the excerpt cannot be reconstructed, so it is unrepresentable
    add_source(conn, "srcGone", "sA", b"")
    add_span(conn, "spGone", "srcGone", 0, 1)
    add_context_group(
        conn, "cg1", "sA", "src1",
        [("spMain", "primary", 1), ("spGone", "condition", 1)],
    )

    result = search(make_store(conn), request("restart"))
    assert result.items == ()
    assert result.groups == ()
    assert result.omitted >= 1
    assert "context_incomplete" in result.warnings


def test_conflict_closure_pulls_unranked_sibling():
    """An open conflict group touched by one ranked claim emits the whole
    authorized closure — the unranked sibling too (V2-26.08)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "conflictterm says blue")
    # clB shares the open conflict group but does NOT match the query
    seed_claim(conn, "clB", "sA", "srcB", "spB", "unrelated says red")
    conn.execute("UPDATE claim_revisions SET state='disputed'")
    conn.execute(
        "INSERT INTO conflict_groups(group_id,scope_id,status)"
        " VALUES('g1','sA','open')"
    )
    for cid in ("clA", "clB"):
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id) VALUES('g1',?)",
            (cid,),
        )

    result = search(make_store(conn), request("conflictterm"))
    ids = {i.claim_id for i in result.items}
    assert ids == {"clA", "clB"}
    assert len(result.groups) == 1
    group = result.groups[0]
    assert {i.claim_id for i in group.items} == {"clA", "clB"}
    assert "UNRESOLVED_CONFLICT" in group.reasons
    sib = [i for i in group.items if i.claim_id == "clB"][0]
    assert "UNRESOLVED_CONFLICT" in sib.reasons
    assert "unresolved_conflict" in result.warnings


def test_conflict_group_with_unauthorized_member_drops_whole():
    """A required group that is partly unauthorized is omitted as a unit —
    a lone member is never presented as settled truth (V2-26.09)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_scope(conn, "sB", principal="p1", conv="c2")
    seed_claim(conn, "clA", "sA", "srcA", "spA", "visibleterm alpha")
    seed_claim(conn, "clB", "sB", "srcB", "spB", "othertext beta")
    conn.execute("UPDATE claim_revisions SET state='disputed'")
    conn.execute(
        "INSERT INTO conflict_groups(group_id,scope_id,status)"
        " VALUES('g1','sA','open')"
    )
    for cid in ("clA", "clB"):
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id) VALUES('g1',?)",
            (cid,),
        )

    result = search(make_store(conn), request("visibleterm"))
    assert result.items == ()
    assert result.groups == ()
    assert result.omitted >= 1
    assert "conflict_group_incomplete" in result.warnings


# ------------------------------------------------------------- bitemporal

def test_historical_cutoff_hides_later_revision():
    """``known_at_seq`` resolves the revision the store held at the cutoff;
    later revisions cannot leak into an earlier view (SPEC_V2 §14)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    add_source(conn, "src1", "sA", b"color is blue")
    add_span(conn, "sp1", "src1", 0, len(b"color is blue"))
    add_source(conn, "src2", "sA", b"color is green")
    add_span(conn, "sp2", "src2", 0, len(b"color is green"))
    add_claim(conn, "clX", "sA")
    add_revision(conn, "clX", rev=1, state="active", recorded_from=1,
                 recorded_until=7, span_ids=("sp1",))
    add_revision(conn, "clX", rev=2, state="active", recorded_from=7,
                 span_ids=("sp2",))
    add_fts(conn, "clX", 1, "sA", "color blue")
    add_fts(conn, "clX", 2, "sA", "color green")

    store = make_store(conn)
    res_now = search(store, request("color"))
    assert [(i.claim_id, i.claim_revision, i.text) for i in res_now.items] == [
        ("clX", 2, "color is green")
    ]
    res_then = search(store, request("color", known_at_seq=3))
    assert [(i.claim_id, i.claim_revision, i.text) for i in res_then.items] == [
        ("clX", 1, "color is blue")
    ]


def test_valid_range_filters_intervals():
    """``valid_until_us`` makes the filter a *range*: a claim applicable
    somewhere inside it qualifies; provably disjoint intervals do not
    (SPEC_V2 §16, §26)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(
        conn, "clIn", "sA", "srcI", "spI", "rangeterm inside",
        intervals=[(us("2024-03-05T00:00:00Z"), us("2024-03-20T00:00:00Z"),
                    "day")],
    )
    seed_claim(
        conn, "clBefore", "sA", "srcB", "spB", "rangeterm before",
        intervals=[(us("2024-01-01T00:00:00Z"), us("2024-02-01T00:00:00Z"),
                    "day")],
    )
    seed_claim(conn, "clOpen", "sA", "srcO", "spO", "rangeterm open ended",
               intervals=[(us("2024-01-01T00:00:00Z"), None, "day")])

    march = request(
        "rangeterm",
        valid_at_us=us("2024-03-01T00:00:00Z"),
        valid_until_us=us("2024-04-01T00:00:00Z"),
    )
    ids = {i.claim_id for i in search(make_store(conn), march).items}
    assert "clIn" in ids
    assert "clOpen" in ids  # open-ended interval provably covers March
    assert "clBefore" not in ids  # provably disjoint


def test_uncertain_interval_stays_unknown_not_false():
    """An UNCERTAIN_RANGE endpoint keeps applicability in the unknown lane
    rather than being treated as absent (SPEC_V2 §16, V2-16.12)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(
        conn, "clUnc", "sA", "srcU", "spU", "uncterm uncertain start",
        intervals=[(us("2024-03-01T00:00:00Z"), us("2024-06-01T00:00:00Z"),
                    "day",
                    {"start_kind": "uncertain_range",
                     "from_us_hi": us("2024-04-01T00:00:00Z")})],
    )
    res = search(
        make_store(conn),
        request("uncterm", valid_at_us=us("2024-03-15T00:00:00Z")),
    )
    assert [i.claim_id for i in res.items] == ["clUnc"]
    assert "TIME_UNKNOWN" in res.items[0].reasons
    assert "time_unknown" in res.warnings


def test_unicode_query_is_signal_not_empty():
    """Non-ASCII queries are real signal: a Japanese term must not collapse
    into a no-signal empty result (V2-25.08, §29)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    # unicode61 tokenizes a CJK run as one token, so the query term must be
    # a standalone token in the stored text
    seed_claim(conn, "clJ", "sA", "srcJ", "spJ", "東京 タワーに行った")
    store = make_store(conn)

    res = search(store, request("東京"))
    assert "no_signal" not in res.warnings
    assert [i.claim_id for i in res.items] == ["clJ"]

    res_ar = search(store, request("مرحبا"))
    assert "no_signal" not in res_ar.warnings


# ------------------------------------------------------------- conditions

def _seed_conditioned(conn):
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(
        conn, "clCond", "sA", "srcC", "spC", "condterm the office policy",
        condition=json.dumps({"op": "eq", "key": "env", "value": "work"}),
    )
    seed_claim(conn, "clPlain", "sA", "srcP", "spP", "condterm plain fact")


def test_condition_applies_with_matching_context():
    conn = make_conn()
    _seed_conditioned(conn)
    res = search(
        make_store(conn),
        request("condterm", context={"env": "work"}),
    )
    cond_items = [i for i in res.items if i.claim_id == "clCond"]
    assert cond_items and "APPLIES" in cond_items[0].reasons
    assert "APPLICABILITY_UNKNOWN" not in res.warnings


def test_condition_unknown_when_context_missing():
    """Missing context leaves the condition in the unknown lane — it is not
    treated as satisfied, and not treated as false (V2-17.03)."""
    conn = make_conn()
    _seed_conditioned(conn)
    res = search(make_store(conn), request("condterm"))
    cond_items = [i for i in res.items if i.claim_id == "clCond"]
    assert cond_items and "APPLICABILITY_UNKNOWN" in cond_items[0].reasons
    assert "APPLICABILITY_UNKNOWN" in res.warnings
    assert res.capabilities["notes"]["missing_context_keys"] == ["env"]


def test_condition_false_excluded_in_current_kept_in_historical():
    conn = make_conn()
    _seed_conditioned(conn)
    store = make_store(conn)
    res = search(store, request("condterm", context={"env": "home"}))
    ids = {i.claim_id for i in res.items}
    assert "clCond" not in ids and "clPlain" in ids

    res_h = search(
        store,
        request("condterm", context={"env": "home"},
                mode=RecallMode.HISTORICAL),
    )
    cond_items = [i for i in res_h.items if i.claim_id == "clCond"]
    assert cond_items and "DOES_NOT_APPLY" in cond_items[0].reasons


# -------------------------------------------------------- byte accounting

def test_whole_result_serialized_byte_ceiling():
    """``max_bytes`` bounds the serialized item stream; groups that would
    exceed the ceiling drop whole and the omission is reported
    (V2-30.05, V2-30.09)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    texts = [
        "budgetterm alpha " + "a" * 400,
        "budgetterm beta " + "b" * 400,
        "budgetterm gamma " + "c" * 400,
    ]
    for n, t in enumerate(texts):
        seed_claim(conn, f"cl{n}", "sA", f"src{n}", f"sp{n}", t)
    store = make_store(conn)

    full = search(store, request("budgetterm", max_bytes=24_000))
    assert len(full.items) == 3
    assert len(full.groups) == 3
    total = sum(len(serialize_item(i)) for i in full.items)
    assert total <= 24_000
    # group serializations honestly report their own cost
    for g in full.groups:
        assert g.serialized_bytes == len(serialize_group(g))
        assert g.serialized_bytes >= sum(
            len(serialize_item(i)) for i in g.items
        )

    # room for exactly one unit: the rest drop whole and are counted
    unit_bytes = max(
        sum(len(serialize_item(i)) for i in g.items) for g in full.groups
    )
    budget = unit_bytes + unit_bytes // 2
    capped = search(store, request("budgetterm", max_bytes=budget))
    assert 1 <= len(capped.items) <= 2
    assert capped.omitted >= 1
    # serialized item stream stays inside the declared ceiling
    assert sum(len(serialize_item(i)) for i in capped.items) <= budget

    # the smallest admissible budget cannot fit even one evidence bundle:
    # BUDGET_TOO_SMALL names the floor failure rather than silent emptiness
    starved = search(store, request("budgetterm", max_bytes=512))
    assert starved.items == ()
    assert starved.omitted >= 1
    assert "BUDGET_TOO_SMALL" in starved.warnings
    assert "EVIDENCE_TOO_LARGE" in starved.warnings


def test_group_dropped_atomically_when_over_budget():
    """A context bundle that cannot fit never loses a required member:
    the entire group is omitted, not split (V2-30.09)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = (
        b"speaker noted: the atomicgroup deploy window is noon "
        + b"x" * 300
    )
    add_source(conn, "src1", "sA", payload)
    add_span(conn, "spMain", "src1", 15, len(payload))
    add_span(conn, "spAttr", "src1", 0, 14)
    add_claim(conn, "clG", "sA")
    add_revision(conn, "clG", span_ids=("spMain",))
    add_fts(conn, "clG", 1, "sA", "atomicgroup deploy window")
    add_context_group(
        conn, "cg1", "sA", "src1",
        [("spMain", "primary", 1), ("spAttr", "attribution", 1)],
    )
    store = make_store(conn)

    full = search(store, request("atomicgroup"))
    assert len(full.groups) == 1
    cost = sum(len(serialize_item(i)) for i in full.groups[0].items)
    # Fits neither the group (primary + required attribution) nor splits it.
    capped = search(
        store, request("atomicgroup", max_bytes=max(512, cost - 1))
    )
    assert capped.items == ()
    assert capped.groups == ()
    assert capped.omitted >= 1


# ------------------------------------------------------------ memory kinds

def test_memory_kind_filtering_routes_to_kind_lanes():
    """``memory_kinds`` selects which lanes run: excluding CLAIM drops claim
    results entirely while episodes/procedures/plans still answer
    (V2-25.02, §21–§24)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clKind", "sA", "srcK", "spK", "kindterm claim body")
    conn.execute(
        "INSERT INTO episodes(episode_id,scope_id,revision,kind,label,"
        "recorded_from) VALUES('ep1','sA',1,'task','kindterm episode label',1)"
    )
    conn.execute(
        "INSERT INTO procedures(procedure_id,scope_id,revision,task_label,"
        "state,recorded_from)"
        " VALUES('pr1','sA',1,'kindterm procedure task','active',1)"
    )
    conn.execute(
        "INSERT INTO procedure_steps(procedure_id,revision,step_no,span_id,"
        "description) VALUES('pr1',1,1,NULL,'kindterm first step')"
    )
    conn.execute(
        "INSERT INTO prospective_records(record_id,scope_id,revision,"
        "owner_id,intention_text,due_us,status,recorded_from)"
        " VALUES('pl1','sA',1,'u1','kindterm plan intention',NULL,"
        "'planned',1)"
    )
    store = make_store(conn)

    all_kinds = search(
        store,
        request("kindterm", memory_kinds=(
            MemoryKind.EPISODE, MemoryKind.PROCEDURE, MemoryKind.PLAN)),
    )
    kinds = {i.claim_id for i in all_kinds.items}
    assert kinds == {"ep1", "pr1", "pl1"}
    # kind labels ride on the item reasons
    by_id = {i.claim_id: i for i in all_kinds.items}
    assert "MEMORY_KIND_EPISODE" in by_id["ep1"].reasons
    assert "MEMORY_KIND_PROCEDURE" in by_id["pr1"].reasons
    assert "MEMORY_KIND_PLAN" in by_id["pl1"].reasons
    # procedure steps travel inside the procedure's group
    pr_group = [g for g in all_kinds.groups if g.primary_claim_id == "pr1"][0]
    assert any("kindterm first step" == i.text for i in pr_group.items)

    claims_only = search(
        store, request("kindterm", memory_kinds=(MemoryKind.CLAIM,))
    )
    assert {i.claim_id for i in claims_only.items} == {"clKind"}

    ep_only = search(
        store, request("kindterm", memory_kinds=(MemoryKind.EPISODE,))
    )
    assert {i.claim_id for i in ep_only.items} == {"ep1"}


def test_episode_membership_items():
    """Episode units carry their member claims' evidence when those claims
    are themselves eligible (V2-21.05, §21)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clM", "sA", "srcM", "spM", "episodeterm member claim")
    conn.execute(
        "INSERT INTO episodes(episode_id,scope_id,revision,kind,label,"
        "recorded_from) VALUES('ep9','sA',1,'task','episodeterm meeting',1)"
    )
    conn.execute(
        "INSERT INTO episode_members(episode_id,object_kind,object_id,ord,"
        "recorded_from) VALUES('ep9','claim','clM',0,1)"
    )

    res = search(
        make_store(conn),
        request("episodeterm", memory_kinds=(MemoryKind.EPISODE,)),
    )
    ep_group = [g for g in res.groups if g.primary_claim_id == "ep9"]
    assert ep_group
    member = [i for i in ep_group[0].items if i.claim_id == "clM"]
    assert member and member[0].text == "episodeterm member claim"
    assert "EPISODE_MEMBER" in member[0].reasons


def test_kind_lanes_degrade_when_tables_absent():
    """On a v1 store (no episode/procedure/prospective relations), kind
    requests report the lanes as unavailable rather than failing
    (V2-05, V2-26.04)."""
    conn = make_conn(v2=False)
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clV1", "sA", "srcV", "spV", "kindterm v1 claim")
    res = search(
        make_store(conn),
        request("kindterm", memory_kinds=(MemoryKind.EPISODE,)),
    )
    assert res.items == ()
    assert "episode" in res.capabilities["degraded"]
    assert "episode_unavailable" in res.warnings


# ---------------------------------------------------------------- deadline

def test_deadline_returns_partial_results():
    """An expired cooperative budget stops the pipeline and returns whatever
    is already packaged, flagged DEADLINE_EXCEEDED (V2-25, V2-26.10)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    for n in range(4):
        seed_claim(conn, f"cl{n}", "sA", f"src{n}", f"sp{n}",
                   f"deadlineterm item {n}")

    import verbatim.retrieval as retr
    from verbatim.retrieval.candidates import Deadline, gather
    from verbatim.retrieval.fusion import rrf
    from verbatim.retrieval.package import package
    from verbatim.retrieval.query import analyze

    class FakeDeadline(Deadline):
        """Deterministic expiry after ``checks`` calls to ``expired``."""

        def __init__(self, checks: int):
            super().__init__(None)
            self._end = 1.0  # armed; real clock value irrelevant
            self._left = checks

        def expired(self) -> bool:
            if self._left <= 0:
                self.exceeded = True
                return True
            self._left -= 1
            return False

    # 1) a budget already blown at gather time: no lanes run, empty result
    #    flagged with the deadline warning rather than silent emptiness
    req = request("deadlineterm")
    store = make_store(conn)
    hits = gather(conn, store, analyze(req.query, req, 0), req, 1,
                  deadline=FakeDeadline(0))
    res = package(conn, store, rrf(hits), req, analyze(req.query, req, 0), 1,
                  deadline=FakeDeadline(0))
    assert res.items == ()
    assert "DEADLINE_EXCEEDED" in res.warnings

    # 2) expiry mid-packaging: first unit ships, the rest are omitted
    hits = gather(conn, store, analyze(req.query, req, 0), req, 1)
    res2 = package(conn, store, rrf(hits), req, analyze(req.query, req, 0), 1,
                   deadline=FakeDeadline(1))
    assert len(res2.items) == 1
    assert res2.omitted >= 3
    assert "DEADLINE_EXCEEDED" in res2.warnings

    # 3) end-to-end: an instantly-expired request through ``search``
    real_cls = retr.Deadline
    try:
        retr.Deadline = lambda _ms: FakeDeadline(0)
        res3 = search(make_store(conn), req)
        assert res3.items == ()
        assert "DEADLINE_EXCEEDED" in res3.warnings
    finally:
        retr.Deadline = real_cls


# ------------------------------------------------------------- capability

def test_capability_reports_degradation_honestly():
    """Capabilities name each lane's real state — no silent success for a
    lane that never ran or that answered on a reduced-quality path
    (V2-05)."""
    conn = make_conn(fts5=False)  # no FTS5 index at all
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA", "capterm evidence")

    res = search(make_store(conn, fts5=False), request("capterm"))
    assert res.items  # lexical fallback still answers
    caps = res.capabilities
    assert caps["semantic"] is False
    assert "semantic" in caps["degraded"]
    reports = {r["name"]: r for r in caps["reports"]}
    # The substring fallback produced hits, but lexical must still report
    # degraded — the index that defines full-quality matching is absent.
    assert reports["lexical"]["state"] == "degraded"
    assert "fts5" in reports["lexical"]["reason"]
    # The embeddings table is simply empty: coverage empty, capability
    # intact — degraded, not "unavailable".
    assert reports["semantic"]["state"] == "degraded"
    assert reports["semantic"]["reason"] == "no embeddings indexed"
    assert reports["rerank"]["state"] == "unavailable"


def test_capability_reports_broken_semantic_unavailable():
    """A broken semantic capability reports ``unavailable`` — embedding
    rows that no encoder can score are not merely empty coverage
    (V2-05)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clB", "sA", "srcB", "spB", "capterm evidence")
    conn.execute(
        "INSERT INTO embeddings(span_id,encoder_id,preprocessing_version,"
        "dimensions,vector,source_hmac) VALUES('spB','enc:test','v1',4,?,?)",
        (b"\x00" * 16, b"h"),
    )
    res = search(make_store(conn), request("capterm"))
    assert "semantic_unavailable" in res.warnings
    reports = {r["name"]: r for r in res.capabilities["reports"]}
    assert reports["semantic"]["state"] == "unavailable"
    assert reports["semantic"]["reason"] == (
        "no embedding backend or encoder configured"
    )
    # FTS5 present and healthy on this store
    assert reports["lexical"]["state"] == "healthy"


def test_min_ready_seq_reports_processing_pending():
    """An unmet declared sequence surfaces as PROCESSING_PENDING while the
    locally available results still ship (V2-25.14)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clR", "sA", "srcR", "spR", "readyterm evidence")
    conn.execute(
        "INSERT INTO events(event_id,scope_id,kind,actor_id,recorded_us,"
        "observed_wall_us,policy_version,payload_json)"
        " VALUES('ev1','sA','ingest','u1',1,1,'p','{}')"
    )
    res = search(
        make_store(conn),
        request("readyterm", min_ready_seq=5),
    )
    assert res.items  # local results still returned
    assert "PROCESSING_PENDING" in res.warnings
    note = res.capabilities["notes"]["min_ready_seq"]
    assert note["requested"] == 5 and note["observed"] == 1


# ------------------------------------------------------------------ browse

def test_timeline_browse_without_keywords():
    """Explicit timeline/archive queries may list eligible memory without a
    keyword — no fake term required (V2-25.11)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clT1", "sA", "srcT1", "spT1", "first timeline entry")
    seed_claim(conn, "clT2", "sA", "srcT2", "spT2", "second timeline entry")
    conn.execute(
        "UPDATE claim_revisions SET state='archived' WHERE claim_id='clT2'"
    )

    res = search(
        make_store(conn),
        request("", mode=RecallMode.TIMELINE),
    )
    assert {i.claim_id for i in res.items} == {"clT1", "clT2"}
    assert "no_signal" not in res.warnings

    # a non-browse mode keeps the typed empty contract
    res_cur = search(make_store(conn), request(""))
    assert res_cur.items == ()
    assert "no_signal" in res_cur.warnings
