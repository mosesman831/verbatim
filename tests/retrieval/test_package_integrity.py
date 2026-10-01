"""Packaging integrity tests — F4-06 (SPEC_V4 §05, V4-08.03, V4-13.05).

``retrieval/package.py`` reconstructs quotations from ``source_revisions``
payload bytes sliced by ``spans`` locators. Before this fix those bytes
were served on the strength of the SQL read alone; now every read path —
claim evidence, context-group members, procedure-step spans, episode
member claims — re-verifies the persisted digests exactly as
``SourcesRepo.payload``/``SpansRepo.text`` do:

* a mutated payload byte or excerpt digest fails closed:
  ``VerbatimError(STORE_CORRUPT)``, never served content (C03);
* a NULL legacy digest is served but labeled ``LEGACY_UNVERIFIED`` and
  can never become verified through a surviving sibling digest (C04);
* the happy path verifies and serves unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import sqlite3

import pytest

from verbatim.core.types import (
    ErrorCode,
    MemoryKind,
    RecallRequest,
    Scope,
    VerbatimError,
)
from verbatim.retrieval import search
from verbatim.retrieval.package import _quote, _verify_slice
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    FTS_TRIGGERS,
)


# Content digests are verified on reads, so seeders write real
# profile-keyed HMACs (same convention as test_retrieval.py /
# test_v2_retrieval.py).
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


# ------------------------------------------------------------------ store

class StoreShim:
    """Minimal store surface retrieval uses: read()/projection_generation()/
    hmac()/fts_enabled."""

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
        return _h(data)


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    conn.executescript(DDL_FTS5)
    conn.executescript(FTS_TRIGGERS)
    conn.executescript(DDL_V2)
    for stmt in DDL_V2_ALTER.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES('projection_generation','1')"
    )
    return conn


def make_legacy_conn() -> sqlite3.Connection:
    """A store whose digest columns exist but are NULLable — the shape a
    database migrated from a pre-digest schema actually takes (C04). The
    current DDL declares ``NOT NULL``, so legacy rows can only exist via
    schema drift; rebuild the two digest-bearing tables while they are
    still empty to model it."""
    conn = make_conn()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        """
        CREATE TABLE source_revisions_nl (
            source_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            payload BLOB NOT NULL,
            payload_hmac BLOB,
            event_us INTEGER NOT NULL,
            captured_us INTEGER NOT NULL,
            timezone TEXT NOT NULL DEFAULT 'UTC',
            provenance TEXT NOT NULL DEFAULT 'unknown',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (source_id, revision),
            FOREIGN KEY (source_id) REFERENCES sources(source_id)
        );
        INSERT INTO source_revisions_nl
            SELECT * FROM source_revisions;
        DROP TABLE source_revisions;
        ALTER TABLE source_revisions_nl RENAME TO source_revisions;

        CREATE TABLE spans_nl (
            span_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            start_byte INTEGER NOT NULL CHECK (start_byte >= 0),
            end_byte INTEGER NOT NULL,
            excerpt_hmac BLOB,
            harvester_version TEXT NOT NULL,
            view_id TEXT,
            operation_key TEXT,
            CHECK (end_byte > start_byte),
            FOREIGN KEY (source_id, revision)
                REFERENCES source_revisions(source_id, revision)
        );
        INSERT INTO spans_nl SELECT * FROM spans;
        DROP TABLE spans;
        ALTER TABLE spans_nl RENAME TO spans;
        CREATE INDEX idx_spans_source ON spans(source_id, revision);
        """
    )
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def make_store(conn=None) -> StoreShim:
    return StoreShim(conn or make_conn())


# ----------------------------------------------------------------- seeders

def add_scope(conn, scope_id, principal=None, conv=None):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,NULL,?,"
        "'conversation',0)",
        (scope_id, "prof", principal, conv),
    )


def add_source(conn, source_id, scope_id, payload: bytes,
               provenance="direct_user"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "user1"),
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


def add_claim(conn, claim_id, scope_id, created_event=1):
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, created_event),
    )


def add_revision(conn, claim_id, rev=1, state="active", recorded_from=1,
                 recorded_until=None, span_ids=()):
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,recorded_from,"
        "recorded_until) VALUES(?,?,?,?,?)",
        (claim_id, rev, state, recorded_from, recorded_until),
    )
    for sid in span_ids:
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES(?,?,?,'primary')",
            (claim_id, rev, sid),
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


def add_context_group(conn, group_id, scope_id, source_id, members,
                      completeness="complete"):
    """members: [(span_id, role, required)]"""
    conn.execute(
        "INSERT INTO context_groups(group_id,scope_id,source_id,revision,"
        "parser_version,operation_key,completeness,recorded_from,"
        "recorded_until) VALUES(?,?,?,1,'p1',?, ?, 1, NULL)",
        (group_id, scope_id, source_id, f"op-{group_id}", completeness),
    )
    for i, (span_id, role, required) in enumerate(members):
        conn.execute(
            "INSERT INTO context_members(group_id,span_id,role,required,ord)"
            " VALUES(?,?,?,?,?)",
            (group_id, span_id, role, int(required), i),
        )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text,
               span_bounds=None, **rev_kw):
    """One scoped claim whose single primary span quotes ``text``."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    start, end = span_bounds or (0, len(payload))
    add_span(conn, span_id, source_id, start, end)
    add_claim(conn, claim_id, scope_id)
    add_revision(conn, claim_id, span_ids=(span_id,), **rev_kw)
    add_fts(conn, claim_id, rev_kw.get("rev", 1), scope_id, text)


READER = Scope(profile_id="prof", principal_id="p1", conversation_id="c1")


def request(query="neovim", **kw) -> RecallRequest:
    kw.setdefault("scope", READER)
    return RecallRequest(query=query, **kw)


def _flip_byte(conn, source_id, index, revision=1):
    """Corrupt one persisted payload byte in place — the digest columns are
    left untouched, exactly like silent disk/bit-rot damage."""
    row = conn.execute(
        "SELECT payload FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchone()
    payload = bytearray(row[0])
    payload[index] ^= 0xFF
    conn.execute(
        "UPDATE source_revisions SET payload = ?"
        " WHERE source_id = ? AND revision = ?",
        (bytes(payload), source_id, revision),
    )


# ------------------------------------------------------------ happy path

def test_verified_evidence_serves_unchanged():
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")

    res = search(make_store(conn), request("deploy"))
    assert [i.claim_id for i in res.items] == ["clA"]
    item = res.items[0]
    assert item.text == "the deploy window is friday at noon"
    # verified slices carry no unverified labeling at all
    assert "LEGACY_UNVERIFIED" not in item.reasons
    assert "UNVERIFIED" not in item.reasons
    assert "legacy_unverified" not in res.warnings
    assert "unverified" not in res.warnings


# --------------------------------------------------------- corruption: C03

def test_mutated_payload_inside_span_fails_closed():
    """A flipped byte inside the quoted range breaks the excerpt digest —
    packaging must raise STORE_CORRUPT, never serve the bytes."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    _flip_byte(conn, "srcA", 5)  # inside the span's [0, len) range

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("deploy"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_mutated_payload_outside_span_fails_closed():
    """A flipped byte outside the span's range leaves the excerpt digest
    intact but breaks ``payload_hmac`` — the payload row is corrupt even
    when the quoted slice happens to decode identically."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = "IGNORED PREFIX. the deploy window is friday".encode()
    text = "IGNORED PREFIX. the deploy window is friday"
    add_source(conn, "srcA", "sA", payload)
    # span covers only the tail sentence, byte 0 stays outside the slice
    start = len(payload) - len("the deploy window is friday")
    add_span(conn, "spA", "srcA", start, len(payload))
    add_claim(conn, "clA", "sA")
    add_revision(conn, "clA", span_ids=("spA",))
    add_fts(conn, "clA", 1, "sA", text)
    _flip_byte(conn, "srcA", 0)  # outside [start, end) — excerpt intact

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("deploy"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_mutated_excerpt_digest_fails_closed():
    """A tampered ``excerpt_hmac`` on the span row is corruption too —
    the span binding fails closed even though the payload verifies."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute(
        "UPDATE spans SET excerpt_hmac = ? WHERE span_id = ?",
        (_h(b"forged excerpt"), "spA"),
    )

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("deploy"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_mutated_context_member_payload_fails_closed():
    """The ``_span_payloads`` path (required context-group members) is
    verified too — corrupting a member's bytes fails the whole read."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = b"Alice announced: the release ships on Friday"
    add_source(conn, "src1", "sA", payload)
    add_span(conn, "spMain", "src1", 17, len(payload))
    add_source(conn, "src2", "sA", b"attribution bytes from the host log")
    add_span(conn, "spAttr", "src2", 0, 18)
    add_claim(conn, "clCtx", "sA")
    add_revision(conn, "clCtx", span_ids=("spMain",))
    add_fts(conn, "clCtx", 1, "sA", "release ships friday")
    add_context_group(
        conn, "cg1", "sA", "src1",
        [("spMain", "primary", 1), ("spAttr", "attribution", 1)],
    )
    _flip_byte(conn, "src2", 3)

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("release"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_mutated_procedure_step_span_fails_closed():
    """Procedure-step quotations ride the same verified slice path — a
    corrupted step payload fails the packaging read closed."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    payload = b"run migrate --dry-run first"
    add_source(conn, "srcP", "sA", payload)
    add_span(conn, "spStep", "srcP", 0, len(payload))
    conn.execute(
        "INSERT INTO procedures(procedure_id,scope_id,revision,task_label,"
        "state,recorded_from)"
        " VALUES('pr1','sA',1,'deployterm procedure task','active',1)"
    )
    conn.execute(
        "INSERT INTO procedure_steps(procedure_id,revision,step_no,span_id,"
        "description) VALUES('pr1',1,1,'spStep','step description')"
    )
    # Lexical-coverage abstention gates on authorized FTS rows — seed one
    # claim covering the query term so the kind-unit lane is reached.
    seed_claim(conn, "clCov", "sA", "srcCov", "spCov",
               "deployterm coverage marker")
    _flip_byte(conn, "srcP", 2)

    with pytest.raises(VerbatimError) as exc:
        search(
            make_store(conn),
            request("deployterm", memory_kinds=(MemoryKind.PROCEDURE,)),
        )
    assert exc.value.code == ErrorCode.STORE_CORRUPT


def test_mutated_episode_member_span_fails_closed():
    """Episode member claims resolve through ``_claim_items`` — a corrupted
    member payload fails closed rather than dropping the member silently."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clM", "sA", "srcM", "spM", "episodeterm member claim")
    conn.execute(
        "INSERT INTO episodes(episode_id,scope_id,revision,kind,label,"
        "recorded_from) VALUES('ep1','sA',1,'task','episodeterm meeting',1)"
    )
    conn.execute(
        "INSERT INTO episode_members(episode_id,object_kind,object_id,ord,"
        "recorded_from) VALUES('ep1','claim','clM',0,1)"
    )
    _flip_byte(conn, "srcM", 4)

    with pytest.raises(VerbatimError) as exc:
        search(
            make_store(conn),
            request("episodeterm", memory_kinds=(MemoryKind.EPISODE,)),
        )
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# ------------------------------------------------- legacy NULL digests: C04

def test_null_digests_serve_labeled_legacy_unverified():
    """A legacy row with both digests NULL is served but reported
    ``LEGACY_UNVERIFIED`` — nothing to check against, never verified."""
    conn = make_legacy_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute("UPDATE spans SET excerpt_hmac = NULL WHERE span_id='spA'")
    conn.execute(
        "UPDATE source_revisions SET payload_hmac = NULL"
        " WHERE source_id='srcA' AND revision=1"
    )

    res = search(make_store(conn), request("deploy"))
    assert [i.claim_id for i in res.items] == ["clA"]
    item = res.items[0]
    assert item.text == "the deploy window is friday at noon"
    assert "LEGACY_UNVERIFIED" in item.reasons
    assert "legacy_unverified" in res.warnings


def test_null_excerpt_digest_not_promoted_by_valid_payload():
    """C04: a span whose own digest is NULL stays ``legacy_unverified``
    even though the payload digest verifies — no fallback promotion."""
    conn = make_legacy_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute("UPDATE spans SET excerpt_hmac = NULL WHERE span_id='spA'")

    res = search(make_store(conn), request("deploy"))
    item = res.items[0]
    assert "LEGACY_UNVERIFIED" in item.reasons
    assert "legacy_unverified" in res.warnings


def test_null_payload_digest_not_promoted_by_valid_excerpt():
    """Symmetric C04: a NULL payload digest keeps the row
    ``legacy_unverified`` even when the excerpt digest verifies."""
    conn = make_legacy_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute(
        "UPDATE source_revisions SET payload_hmac = NULL"
        " WHERE source_id='srcA' AND revision=1"
    )

    res = search(make_store(conn), request("deploy"))
    item = res.items[0]
    assert "LEGACY_UNVERIFIED" in item.reasons
    assert "legacy_unverified" in res.warnings


def test_legacy_unverified_payload_mutation_still_fails_closed():
    """A legacy row is unverified, not exempt: a surviving sibling digest
    that still mismatches is corruption — fail closed, never served."""
    conn = make_legacy_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    # legacy span (no excerpt digest) + mutated payload -> payload_hmac
    # still catches the corruption
    conn.execute("UPDATE spans SET excerpt_hmac = NULL WHERE span_id='spA'")
    _flip_byte(conn, "srcA", 7)

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("deploy"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# ------------------------------------------------------------- purge edge

def test_purged_payload_is_unavailable_not_corrupt():
    """A purge scrub empties the payload and reseals its digest — the span
    is *absent*, not corrupt: no exception, just unavailable evidence."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute(
        "UPDATE source_revisions SET payload = X'', payload_hmac = ?"
        " WHERE source_id='srcA' AND revision=1",
        (_h(b""),),
    )

    res = search(make_store(conn), request("deploy"))
    assert res.items == ()
    assert "evidence_unavailable" in res.warnings


def test_tampered_purge_marker_is_corrupt_not_absent():
    """A purge scrub reseals ``payload_hmac`` over ``b''`` — an emptied
    payload carrying a stale digest was tampered post-purge, so it fails
    closed like any other corruption (SourcesRepo.payload parity)."""
    conn = make_conn()
    add_scope(conn, "sA", principal="p1", conv="c1")
    seed_claim(conn, "clA", "sA", "srcA", "spA",
               "the deploy window is friday at noon")
    conn.execute(
        "UPDATE source_revisions SET payload = X''"
        " WHERE source_id='srcA' AND revision=1"  # stale payload_hmac kept
    )

    with pytest.raises(VerbatimError) as exc:
        search(make_store(conn), request("deploy"))
    assert exc.value.code == ErrorCode.STORE_CORRUPT


# --------------------------------------------------------- helper contracts

def test_verify_slice_labels_and_raises():
    """Unit-level contract: verified / legacy_unverified / STORE_CORRUPT."""
    store = make_store(make_conn())
    payload = b"hello world"
    good = ("src", 1, 0, 5, payload, "direct_user", "u1",
            _h(b"hello"), _h(payload))
    assert _verify_slice(store, good, span_id="sp") == "verified"

    legacy = ("src", 1, 0, 5, payload, "direct_user", "u1", None, None)
    assert _verify_slice(store, legacy, span_id="sp") == "legacy_unverified"

    # Digests still cover the ORIGINAL bytes — a byte flipped inside the
    # quoted slice breaks the excerpt digest even when the sibling
    # payload digest happens to be NULL.
    tampered = ("src", 1, 0, 5, b"h3llo world", "direct_user", "u1",
                _h(b"hello"), None)
    with pytest.raises(VerbatimError) as exc:
        _verify_slice(store, tampered, span_id="sp")
    assert exc.value.code == ErrorCode.STORE_CORRUPT

    # A tampered payload under a stale payload digest fails closed too.
    stale = ("src", 1, 0, 5, b"hello!world", "direct_user", "u1",
             _h(b"hello"), _h(payload))
    with pytest.raises(VerbatimError) as exc:
        _verify_slice(store, stale, span_id="sp")
    assert exc.value.code == ErrorCode.STORE_CORRUPT

    # _quote with a store re-checks digests and fails closed on corruption
    with pytest.raises(VerbatimError) as exc:
        _quote(tampered, store)
    assert exc.value.code == ErrorCode.STORE_CORRUPT
    # ...while the same row served without a verifier stays byte-exact
    assert _quote(good)[0] == "hello"
