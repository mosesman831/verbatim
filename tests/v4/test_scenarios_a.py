"""SPEC_V4 §58 acceptance scenarios — C01–C16.

The C01–C16 band covers the *foundational* evidence contracts: consented
capture and durable readiness, partition-local dedup, integrity-checked
reads on every byte-serving surface, honest provenance attribution,
privacy routing before indexing, purpose-aware cross-scope
authorization, delegation attenuation, identifier narrowing, hold
cascades, honest abstention, broker-gated egress, and telemetry
honesty.

Every scenario drives the REAL public path on a file-backed
``Store.create(tmp_path)`` — facade (``VerbatimV3``), kernel,
``recall_v3``, export, the durable job queue, and the egress broker.
Where a named capability does not exist (a public telemetry-export
surface), the test asserts the boundary that would govern it and marks
the gap ``TODO-CNN`` — nothing is fabricated.

Determinism: fixtures pin the profile HMAC key (``<db>.key``) so seeded
digests verify under ``store.hmac`` — the same convention as
``tests/v4/test_scenarios_c.py`` and ``tests/retrieval/v3``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from decimal import Decimal
from typing import Optional

import pytest

from verbatim.api_v3 import VerbatimV3
from verbatim.config import (
    EmbeddingConfig,
    JudgeConfig,
    V3Config,
    VaultConfig,
    VerbatimConfig,
)
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Mode,
    Scope,
    VerbatimError,
    Visibility,
)
from verbatim.core.types_v3 import (
    RecallRequestV3,
    TrustClass,
    Verb,
)
from verbatim.core.types_v4 import (
    EvidenceLocator,
    PurposeConstraint,
    PurposeTag,
)
from verbatim.embeddings.cloudflare import CloudflareEncoder
from verbatim.embeddings.encoder import encoder_requires_permit
from verbatim.embeddings.ollama import HttpResult
from verbatim.export import export_scope
from verbatim.governance import (
    CallerV3,
    create_grant,
    delegate_grant,
    register_principal,
    seed_purposes,
)
from verbatim.ingest import Ingester
from verbatim.kernel import Kernel
from verbatim.privacy.broker import TransportBroker
from verbatim.privacy.redaction import (
    lookup_placeholder,
    redact_view,
    spans_for_view,
)
from verbatim.procedures import compile_episode
from verbatim.procedures.compiler import CompilationStatus
from verbatim.retrieval.inspect import inspect_claim
from verbatim.retrieval.v3 import recall_v3
from verbatim.security import open_quarantine
from verbatim.storage.repos import SourcesRepo, SpansRepo
from verbatim.storage.store import Store
from tests.conftest import FakeClock, grant_consent


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------

# Content digests are verified on reads, so seeded rows carry real
# profile-keyed HMACs — the fixture pins the profile key so _h() ==
# store.hmac() (same convention as tests/v4/test_scenarios_c.py).
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v4.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v4.db"))
    yield s
    s.close()


@pytest.fixture
def kernel(store):
    return Kernel(store)


def _scope(conn, sid, principal="p1", conv="c1", profile="prof",
           vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (sid, profile, principal, "ws", conv, vis),
    )


def _auth(conn, sid, pid="human:alice", purposes=("recall",),
          verbs=("read", "quote"), **kw):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=sid, principal_id=pid, verbs=set(verbs),
        issuer_id=pid,
        purposes=None if purposes is None else list(purposes),
        **kw,
    )


def _add_source(conn, source_id, scope_id, payload: bytes, speaker="u1"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, _h(payload)),
    )


def _add_span(conn, span_id, source_id, start, end, rev=1):
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


def _add_fts(conn, claim_id, rev, scope_id, text, gen):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def _seed_claim(conn, claim_id, scope_id, source_id, span_id, text, gen,
                state="active", recorded_from=1, rev=1):
    """Claim + revision + primary evidence + FTS row (real digests)."""
    payload = text.encode("utf-8")
    _add_source(conn, source_id, scope_id, payload)
    _add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until) VALUES(?,?,?,?,?,?)",
        (claim_id, rev, state, None, recorded_from, None),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, span_id),
    )
    _add_fts(conn, claim_id, rev, scope_id, text, gen)


def _gen(store) -> int:
    return store.projection_generation()


def _req(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def _items(result):
    return [i for p in result.packs for i in p.items]


def _texts(result):
    return [i.text or "" for p in result.packs for i in p.items]


def _item_body(text: str) -> dict:
    """Parse one serialized pack item's untrusted-JSON body."""
    s = text
    s = s[s.index(">") + 1:] if s.startswith("<memory_evidence") else s
    if s.endswith("</memory_evidence>"):
        s = s[: -len("</memory_evidence>")]
    return json.loads(s)


def _item_reasons(result) -> list:
    return [
        r for i in _items(result) for r in _item_body(i.text)["reasons"]
    ]


def _err(fn) -> ErrorCode:
    with pytest.raises(VerbatimError) as ei:
        fn()
    return ei.value.code


# --- episode/procedure fixtures (C42 shape, tests/v4/test_scenarios_c.py) ---


def _episode(conn, episode_id, scope_id, *, closed=True):
    conn.execute(
        "INSERT INTO episodes(episode_id,scope_id,revision,kind,label,"
        "recorded_from,recorded_until,row_version)"
        " VALUES(?,?,1,'task','ep',1,? ,1)",
        (episode_id, scope_id, 10 if closed else None),
    )


def _trajectory(conn, traj_id, scope_id):
    conn.execute(
        "INSERT INTO trajectories(trajectory_id,scope_id,host_id,"
        "session_id,task_id,boundary_rule,created_event,metadata_json)"
        " VALUES(?,?,'h1','ss1','t1','task_id',1,'{}')",
        (traj_id, scope_id),
    )


def _envelope(conn, envelope_id, source_id, scope_id, kind, meta=None,
              payload=None):
    """Envelope row + backing source/revision (payload is the JSON body
    ``_envelope_payload`` decodes — HMAC'd like every seeded source)."""
    body = json.dumps(payload or {}).encode("utf-8")
    _add_source(conn, source_id, scope_id, body)
    conn.execute(
        "INSERT INTO source_envelopes(envelope_id,source_id,revision,"
        "scope_id,envelope_kind,actor_principal,event_us,receipt_us,"
        "media_type,trust_class,metadata_json)"
        " VALUES(?,?,1,?,?,?,1,1,'application/json','agent_generated',?)",
        (envelope_id, source_id, scope_id, kind, "agent:a",
         json.dumps(meta or {})),
    )


def _tool_step(conn, step_id, traj_id, scope_id, ord_, envelope_id):
    conn.execute(
        "INSERT INTO trajectory_steps(step_id,trajectory_id,scope_id,ord,"
        "action_envelope_id) VALUES(?,?,?,?,?)",
        (step_id, traj_id, scope_id, ord_, envelope_id),
    )


def _transition(conn, transition_id, episode_id, scope_id, ord_,
                action_step_id=None, checker_ref=None):
    conn.execute(
        "INSERT INTO transitions(transition_id,episode_id,scope_id,ord,"
        "action_step_id,checker_ref,edge,created_event)"
        " VALUES(?,?,?,?,?,?,'observed_after',1)",
        (transition_id, episode_id, scope_id, ord_, action_step_id,
         checker_ref),
    )


def _seed_episode(conn, episode_id, scope_id, *, tool_meta=None,
                  checker_meta=None, tool_payload=None,
                  checker_payload=None, prefix="c"):
    """A fully-evidenced C42 episode: two recognized tool ops + one
    checker receipt — compiles to ``candidate`` when evidence verifies."""
    _episode(conn, episode_id, scope_id, closed=True)
    _trajectory(conn, f"traj-{prefix}", scope_id)
    _envelope(
        conn, f"env-{prefix}1", f"src-{prefix}1", scope_id, "tool_call",
        meta=tool_meta or {"tool": "read_file", "args": {"path": "a.py"}},
        payload=tool_payload,
    )
    _envelope(
        conn, f"env-{prefix}2", f"src-{prefix}2", scope_id, "tool_call",
        meta={"tool": "run_check", "args": {"argv": ["pytest", "-x"]}},
    )
    _envelope(
        conn, f"env-{prefix}3", f"src-{prefix}3", scope_id, "verification",
        meta=checker_meta or {"checker": "pytest", "outcome": "passed"},
        payload=checker_payload,
    )
    _tool_step(conn, f"st-{prefix}1", f"traj-{prefix}", scope_id, 0,
               f"env-{prefix}1")
    _tool_step(conn, f"st-{prefix}2", f"traj-{prefix}", scope_id, 1,
               f"env-{prefix}2")
    _transition(conn, f"ts-{prefix}1", episode_id, scope_id, 0,
                action_step_id=f"st-{prefix}1")
    _transition(conn, f"ts-{prefix}2", episode_id, scope_id, 1,
                action_step_id=f"st-{prefix}2")
    _transition(conn, f"ts-{prefix}3", episode_id, scope_id, 2,
                checker_ref=f"env-{prefix}3")


# =====================================================================
# C01 — fresh setup: consent, capture, drain, recall, restart (C)
# =====================================================================


def test_c01_capture_drain_recall_and_restart_preserve_evidence(tmp_path):
    """C01 / V4-13.01 + V4-14.01: a consented capture persists the source
    envelope (actor, claimed authorship, origin, external id, media type,
    timestamps, scope, consent proof), the operation receipt and its
    readiness obligations atomically; recall returns the exact evidence;
    close/reopen preserves every record — nothing re-derives or degrades
    across restart.
    """
    db = str(tmp_path / "c01.db")
    api = VerbatimV3.open(db)
    pid, sid = "agent:main", "scope:c01"
    text = "the exact phrase survives restart"

    # §11.11 consent first — capture without it is CONSENT_REQUIRED.
    auth_id = api.issue_capture_authorization(pid, sid, granted_by=pid)
    source_id = api.capture_submitted(
        pid, sid, text,
        declared_type="agent_note", external_id="c01-ext-1", title="n1",
    )
    assert source_id

    # The accepted capture is atomic: evidence + consent binding +
    # screening disposition + receipt + readiness obligations committed
    # together (V4-14.01) — the drain reports the durable queue state.
    report = Ingester(api.store, api.config).drain_report()
    assert report["failed"] == 0

    state = api.receipt_state(
        source_id=source_id, principal_id=pid, scope_id=sid
    )
    assert state["receipt_id"]
    assert state["states"]  # per-capability durable rows, not inferred
    ready = api.wait_ready(
        source_id=source_id, principal_id=pid, scope_id=sid, timeout_s=2.0
    )
    assert ready["receipt_id"] == state["receipt_id"]
    assert ready["ready"] is True or ready["complete"] is True

    # Recall reaches the exact bytes only through the explicitly
    # declared archive/evidence lane (V4-08.07).
    res = api.recall(
        sid, "exact phrase survives", principal_id=pid,
        budget={"modes": ("evidence",)},
    )
    assert any(text in t for t in _texts(res))

    # Envelope fields (V4-13.01): actor, trust class, consent proof,
    # external identity, media type, host, timestamps.
    info = api.inspect_evidence(source_id, principal_id=pid)
    env = info["envelopes"][0]
    assert env["actor_principal"] == pid
    assert env["trust_class"] == TrustClass.AGENT_GENERATED.value
    assert env["capture_proof"] == auth_id
    assert env["media_type"] == "text/plain"
    assert info["external_id"] == "c01-ext-1"
    assert info["receipts"]  # operation receipt persisted atomically
    api.close()

    # Restart: reopen the same file — evidence, receipts, and the
    # readiness DAG survive verbatim; nothing needs re-seeding.
    api2 = VerbatimV3.open(db)
    try:
        res2 = api2.recall(
            sid, "exact phrase survives", principal_id=pid,
            budget={"modes": ("evidence",)},
        )
        assert any(text in t for t in _texts(res2))
        info2 = api2.inspect_evidence(source_id, principal_id=pid)
        assert info2["envelopes"][0]["envelope_id"] == env["envelope_id"]
        assert info2["receipts"]
        state2 = api2.receipt_state(
            source_id=source_id, principal_id=pid, scope_id=sid
        )
        assert state2["receipt_id"] == state["receipt_id"]
        assert state2["states"] == state["states"]
    finally:
        api2.close()


# =====================================================================
# C02 — same external id in two scopes = two independent sources (C)
# =====================================================================


def test_c02_external_id_stays_partition_local_across_scopes(tmp_path):
    """C02 / V4-10.03: source dedup keys on (scope, origin, external_id)
    — the same host identifier captured in two scopes yields two
    independent sources with independent purge boundaries; deleting one
    never touches the other's evidence.
    """
    api = VerbatimV3.open(str(tmp_path / "c02.db"))
    pid = "agent:main"
    sa, sb = "scope:c02-a", "scope:c02-b"
    for s in (sa, sb):
        api.issue_capture_authorization(pid, s, granted_by=pid)
    src_a = api.capture_submitted(
        pid, sa, "alpha body", external_id="shared-ext-7"
    )
    src_b = api.capture_submitted(
        pid, sb, "bravo body — same key, different scope",
        external_id="shared-ext-7",
    )
    assert src_a != src_b  # no cross-scope graft

    with api.store.read() as conn:
        rows = conn.execute(
            "SELECT source_id, scope_id FROM sources"
            " WHERE external_id = 'shared-ext-7' ORDER BY scope_id"
        ).fetchall()
    assert {r[0] for r in rows} == {src_a, src_b}
    assert {r[1] for r in rows} == {sa, sb}

    # Same key + same scope + same payload = idempotent replay, not a
    # second source (the dedup half of V4-10.03).
    again = api.capture_submitted(
        pid, sa, "alpha body", external_id="shared-ext-7"
    )
    assert again == src_a

    # Independent purge boundary: admin delete in A suppresses only A.
    out = api.delete_source(src_a, principal_id=pid)
    assert out["status"] == "suppressed"
    Ingester(api.store, api.config).drain_report()

    # A's evidence bytes are withheld — the archive lane serves nothing
    # for A and inspection reports the tombstone (never the payload).
    res_a = api.recall(
        sa, "alpha body", principal_id=pid,
        budget={"modes": ("evidence",)},
    )
    assert not any("alpha body" in t for t in _texts(res_a))
    info_a = api.inspect_evidence(src_a, principal_id=pid)
    assert info_a["suppressed"] is True
    assert "alpha body" not in json.dumps(info_a)
    # B is untouched — exact bytes still verify and serve.
    res_b = api.recall(
        sb, "bravo body", principal_id=pid,
        budget={"modes": ("evidence",)},
    )
    assert any("bravo body" in t for t in _texts(res_b))
    info_b = api.inspect_evidence(src_b, principal_id=pid)
    assert info_b["external_id"] == "shared-ext-7"
    api.close()


# =====================================================================
# C03 — mutated payload/span digest fails closed everywhere (C)
# =====================================================================


def test_c03_mutated_bytes_fail_closed_on_every_surface(store):
    """C03 / V4-08.03 + V4-13.05: rewriting stored payload bytes without
    their keyed digest is corruption, not content — recall, inspection,
    export, and procedure compilation all fail ``STORE_CORRUPT`` (or
    withhold the tainted evidence); no surface serves or derives from
    forged bytes.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-c03", "sA", "src-c03", "sp-c03",
                    "the stored sentence is the evidence", _gen(store))
        _seed_episode(conn, "ep-c03", "sA", prefix="c03")
        # Baseline: the episode compiles when its evidence verifies.
        r = compile_episode(conn, "ep-c03", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.CANDIDATE

    # Tamper: same-length forged payload, digest left stale.
    forged = b"the FORGED sentence is the evidence"
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = 'src-c03' AND revision = 1",
            (forged,),
        )

    # recall: pack assembly re-verifies slice digests → STORE_CORRUPT.
    assert _err(lambda: recall_v3(store, _req("stored sentence"))) == (
        ErrorCode.STORE_CORRUPT
    )
    # inspect: span text re-verifies excerpt_hmac → STORE_CORRUPT.
    reader = Scope(
        profile_id="prof", principal_id="p1", workspace_id="ws",
        conversation_id="c1", visibility=Visibility.CONVERSATION,
    )
    assert _err(lambda: inspect_claim(store, "cl-c03", reader)) == (
        ErrorCode.STORE_CORRUPT
    )
    # export: SourcesRepo.payload re-verifies → STORE_CORRUPT.
    assert _err(lambda: export_scope(store, "sA")) == ErrorCode.STORE_CORRUPT
    # raw byte reads: verified-read seam → STORE_CORRUPT.
    assert _err(
        lambda: SourcesRepo(store).payload("src-c03", 1)
    ) == ErrorCode.STORE_CORRUPT
    assert _err(
        lambda: SpansRepo(store).text("sp-c03")
    ) == ErrorCode.STORE_CORRUPT

    # compilation: the compiler is a verified-read surface too — a
    # tampered envelope payload is corruption, never compiler input.
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = 'src-c031' AND revision = 1",
            (b'{"tool": "shell_exec", "args": {"command": "rm -rf /"}}',),
        )
    with store.read() as conn:
        assert _err(
            lambda: compile_episode(conn, "ep-c03", hmac_fn=store.hmac)
        ) == ErrorCode.STORE_CORRUPT

    # The durable job path fails closed the same way: the compile job
    # terminally fails STORE_CORRUPT instead of minting from forged bytes.
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        _episode(conn, "ep-c03b", "sA", closed=True)
        _trajectory(conn, "traj-c03b", "sA")
        _envelope(conn, "env-c03b1", "src-c03b1", "sA", "tool_call",
                  meta={"tool": "read_file", "args": {"path": "a.py"}})
        _envelope(conn, "env-c03b2", "src-c03b2", "sA", "tool_call",
                  meta={"tool": "run_check",
                        "args": {"argv": ["pytest", "-x"]}})
        _envelope(conn, "env-c03b3", "src-c03b3", "sA", "verification",
                  meta={"checker": "pytest", "outcome": "passed"})
        _tool_step(conn, "st-c03b1", "traj-c03b", "sA", 0, "env-c03b1")
        _tool_step(conn, "st-c03b2", "traj-c03b", "sA", 1, "env-c03b2")
        _transition(conn, "ts-b1", "ep-c03b", "sA", 0,
                    action_step_id="st-c03b1")
        _transition(conn, "ts-b2", "ep-c03b", "sA", 1,
                    action_step_id="st-c03b2")
        _transition(conn, "ts-b3", "ep-c03b", "sA", 2,
                    checker_ref="env-c03b3")
        ing.jobs.enqueue(
            conn, "sA", JobKind.PROCEDURE_COMPILE,
            {"episode_id": "ep-c03b"},
            operation_key="compile:ep-c03b",
        )
        conn.execute(
            "UPDATE source_revisions SET payload = ?"
            " WHERE source_id = 'src-c03b3' AND revision = 1",
            (b'{"checker": "pytest", "outcome": "forged"}',),
        )
    with store.read() as conn:
        proc_count = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
    report = ing.drain_report()
    assert report["failed"] >= 1
    assert any(e["code"] == ErrorCode.STORE_CORRUPT.value
               for e in report["errors"])
    with store.read() as conn:
        # No new procedure minted from forged evidence.
        assert conn.execute(
            "SELECT COUNT(*) FROM procedures"
        ).fetchone()[0] == proc_count
        job = conn.execute(
            "SELECT state, error_code FROM jobs WHERE kind = ?",
            (JobKind.PROCEDURE_COMPILE.value,),
        ).fetchone()
        assert job[0] == "failed"
        assert job[1] == ErrorCode.STORE_CORRUPT.value


# =====================================================================
# C04 — legacy NULL digest = unverified, never promoted (C)
# =====================================================================


def test_c04_null_digest_is_legacy_unverified_never_verified(store):
    """C04 / V4-13.05: a source revision stored before keyed digests
    (NULL ``payload_hmac``/``excerpt_hmac``) is served labeled
    ``legacy_unverified`` — never ``verified``, and a valid sibling
    digest cannot promote it.
    """
    with store.tx() as conn:
        # Model the shape a database migrated from a pre-digest schema
        # actually takes: the digest columns exist but are NULLable. The
        # current DDL declares NOT NULL, so legacy rows can only exist
        # via schema drift — rebuild the two digest-bearing tables while
        # still empty (same convention as
        # tests/retrieval/test_package_integrity.py::make_legacy_conn,
        # here on the durable file-backed store).
        for stmt in (
            """CREATE TABLE source_revisions_nl (
                source_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                payload BLOB NOT NULL,
                payload_hmac BLOB,
                event_us INTEGER NOT NULL,
                captured_us INTEGER NOT NULL,
                timezone TEXT,
                provenance TEXT NOT NULL DEFAULT 'unknown',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (source_id, revision)
            )""",
            "INSERT INTO source_revisions_nl SELECT * FROM source_revisions",
            "DROP TABLE source_revisions",
            "ALTER TABLE source_revisions_nl RENAME TO source_revisions",
            """CREATE TABLE spans_nl (
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
            )""",
            "INSERT INTO spans_nl SELECT * FROM spans",
            "DROP TABLE spans",
            "ALTER TABLE spans_nl RENAME TO spans",
            "CREATE INDEX idx_spans_source ON spans(source_id, revision)",
        ):
            conn.execute(stmt)
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-c04", "sA", "src-c04", "sp-c04",
                    "legacy bytes without digests", _gen(store))
        # Legacy row: NULL digests on payload AND excerpt.
        conn.execute(
            "UPDATE source_revisions SET payload_hmac = NULL"
            " WHERE source_id = 'src-c04'"
        )
        conn.execute(
            "UPDATE spans SET excerpt_hmac = NULL WHERE span_id = 'sp-c04'"
        )

    res = recall_v3(store, _req("legacy bytes"))
    texts = _texts(res)
    assert any("legacy bytes without digests" in t for t in texts)
    reasons = _item_reasons(res)
    assert "LEGACY_UNVERIFIED" in reasons
    assert not any(r == "VERIFIED" for r in reasons)

    # Integrity audit counts the row as unverified — never failed,
    # never silently verified.
    report = store.check_integrity()
    assert report["content_integrity"]["unverified"] >= 1
    assert report["content_integrity"]["failed"] == 0

    # A valid payload digest cannot promote a NULL span digest either:
    # verification requires BOTH bound digests.
    with store.tx() as conn:
        payload = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = 'src-c04'"
        ).fetchone()[0]
        conn.execute(
            "UPDATE source_revisions SET payload_hmac = ?"
            " WHERE source_id = 'src-c04'",
            (store.hmac(bytes(payload)),),
        )
    res2 = recall_v3(store, _req("legacy bytes"))
    reasons2 = _item_reasons(res2)
    assert "LEGACY_UNVERIFIED" in reasons2
    # Read paths still serve the bytes (legacy = unverified, not corrupt):
    assert SourcesRepo(store).payload("src-c04", 1) is not None
    assert SpansRepo(store).text("sp-c04") == "legacy bytes without digests"


# =====================================================================
# C05 — agent text claiming human authorship stays agent-attributed (C)
# =====================================================================


def test_c05_agent_text_claiming_human_authorship_stays_agent(tmp_path):
    """C05 / V4-13.07: an agent submission declaring a principal-authored
    kind (``user_message``) is recorded as ``agent_note`` with
    ``agent_generated`` trust — the declaration is preserved in metadata
    as an agent claim, never minted into human testimony.
    """
    api = VerbatimV3.open(str(tmp_path / "c05.db"))
    pid, sid = "agent:main", "scope:c05"
    api.issue_capture_authorization(pid, sid, granted_by=pid)
    src = api.capture_submitted(
        pid, sid, "I the human user personally wrote this",
        declared_type="user_message",
    )
    info = api.inspect_evidence(src, principal_id=pid)
    env = info["envelopes"][0]
    # Downgraded kind + agent trust — the claim is recorded, not believed.
    assert env["envelope_kind"] == "agent_note"
    assert env["trust_class"] == TrustClass.AGENT_GENERATED.value
    assert env["actor_principal"] == pid
    assert env["metadata"]["declared_type"] == "user_message"
    assert env["metadata"]["submitted_via"] == "api_v3.capture_submitted"
    # The stored evidence itself carries agent provenance, and no
    # user-authored claim can be harvested from it (agent-authored kinds
    # never reach the claim lane — V3-13.11).
    Ingester(api.store, api.config).drain_report()
    with api.store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM claims WHERE scope_id = ?", (sid,)
        ).fetchone()[0] == 0
        prov = conn.execute(
            "SELECT provenance FROM source_revisions WHERE source_id = ?",
            (src,),
        ).fetchone()[0]
    assert prov != "direct_user"
    api.close()


# =====================================================================
# C06 — declared sensitive input is privacy-routed before indexing (C)
# =====================================================================


def test_c06_declared_sensitive_never_reaches_plaintext_search(store):
    """C06 / V4-37.02: a declared S2 span is substituted by a typed
    placeholder before any indexable view exists — the value lives only
    as vault ciphertext, the placeholder is the searchable token, and a
    declared S4 span is zero-retention (no vault entry at all).
    """
    keys: dict = {}

    def _provider(sid, version):
        vers = keys.get(sid) or {}
        if version is None:
            v = max(vers)
            return vers[v], v
        k = vers.get(int(version))
        return (k, int(version)) if k is not None else None

    cfg = VerbatimConfig(
        v3=V3Config(vault=VaultConfig(enabled=True, key_source="external"))
    )
    sid = "sA"
    text = "email me at alice@example.com about the plan"
    secret_at = text.index("alice@example.com")
    s4_secret = "ak-secret-key-99"
    full = text + " key " + s4_secret
    s4_at = full.index(s4_secret)

    with store.tx() as conn:
        _scope(conn, sid)
        keys[sid] = {1: b"\x77" * 32}
        result = redact_view(
            conn, sid, "view:c06", full.encode("utf-8"),
            cfg=cfg,
            declared=[
                (secret_at, secret_at + len("alice@example.com"), "s2"),
                (s4_at, s4_at + len(s4_secret), "s4"),
            ],
            key_provider=_provider,
        )
    accepted = result.accepted.decode("utf-8")
    # Placeholders, never plaintext, in the accepted (indexable) view.
    assert "alice@example.com" not in accepted
    assert s4_secret not in accepted
    assert "[VAULT:" in accepted

    with store.read() as conn:
        # The S2 value exists only as AES-256-GCM ciphertext.
        vault_rows = conn.execute(
            "SELECT ciphertext, placeholder FROM vault_entries"
            " WHERE scope_id = ?", (sid,)
        ).fetchall()
        assert len(vault_rows) == 1  # S4 stores nothing
        ct, ph = vault_rows[0]
        assert b"alice@example.com" not in bytes(ct)
        # The placeholder resolves to non-content metadata only.
        ref = lookup_placeholder(conn, sid, ph)
        assert ref is not None and ref["view_id"] == "view:c06"
        spans = spans_for_view(conn, sid, "view:c06")
        sens = {r["sensitivity"] for r in spans}
        assert sens == {"s2", "s4"}
        s4_row = [r for r in spans if r["sensitivity"] == "s4"][0]
        assert s4_row["entry_id"] is None  # zero retention — no vault row

    # Indexing consumes ONLY the accepted view — the rows below are what
    # harvest→admit would have produced for the redacted bytes: a search
    # for the secret finds nothing; the placeholder token is searchable.
    with store.tx() as conn:
        _auth(conn, sid)
        _add_source(conn, "src-c06", sid, result.accepted)
        _add_span(conn, "sp-c06", "src-c06", 0, len(result.accepted))
        conn.execute(
            "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
            "created_event,row_version) VALUES('cl-c06',?,NULL,NULL,1,1)",
            (sid,),
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "condition_json,recorded_from,recorded_until)"
            " VALUES('cl-c06',1,'active',NULL,1,NULL)"
        )
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES('cl-c06',1,'sp-c06','primary')"
        )
        _add_fts(conn, "cl-c06", 1, sid, accepted, _gen(store))
    res = recall_v3(store, _req("alice@example.com"))
    assert not any("alice@example.com" in t for t in _texts(res))
    res2 = recall_v3(store, _req("VAULT"))
    assert res2.packs  # placeholder is the searchable token


# =====================================================================
# C07 — evaluate-only authority on B cannot leak through a query on A (C)
# =====================================================================


def test_c07_evaluate_only_scope_cannot_leak_through_query_on_a(store):
    """C07 / V4-10.05 + V4-11.04: every contributing scope must pass the
    same purpose-aware authorization — a grant on B limited to purpose
    ``evaluate`` contributes nothing to a ``recall``-purpose query, and a
    query issued on A at ``evaluate`` is denied at the requesting scope
    itself. Denial is indistinguishable from absence.
    """
    with store.tx() as conn:
        _scope(conn, "sA", conv="c1")
        _scope(conn, "sB", conv="c2")
        _auth(conn, "sA", purposes=("recall",))
        _auth(conn, "sB", purposes=("evaluate",))
        _seed_claim(conn, "cl-a", "sA", "src-a", "sp-a",
                    "alpha fact readable for recall", _gen(store))
        _seed_claim(conn, "cl-b", "sB", "src-b", "sp-b",
                    "bravo secret only for evaluation", _gen(store))

    # recall-purpose request on A: B's evaluate-only grant contributes
    # nothing — B's text never appears.
    res = recall_v3(store, _req("fact readable OR secret evaluation"))
    assert any("alpha" in t for t in _texts(res))
    assert not any("bravo" in t or "secret" in t for t in _texts(res))

    # Asking at B's own purpose from A's scope fails closed at the
    # requesting scope — NOT_FOUND, indistinguishable from absence.
    assert _err(
        lambda: recall_v3(store, _req("bravo", purpose="evaluate"))
    ) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # And a request issued against B at purpose 'recall' denies too —
    # the purpose constraint binds per scope, not per request bundle.
    assert _err(
        lambda: recall_v3(store, _req("bravo", scope_id="sB"))
    ) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# =====================================================================
# C08 — read-only authority cannot reach exact bytes (C)
# =====================================================================


def test_c08_read_only_cannot_obtain_exact_bytes(store):
    """C08 / V4-08.02: ``read`` permits metadata and approved derived
    views — exact original bytes require ``quote``. A read-only caller
    gets metadata items (``quote_withheld``), an empty-byte browse lane,
    a denied quote lease at the kernel, and no raw text anywhere.
    """
    payload = "precise bytes only a quoter may read"
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA", verbs=("read",))  # no quote
        _seed_claim(conn, "cl-c08", "sA", "src-c08", "sp-c08",
                    payload, _gen(store))

    # Derived recall: the claim's exact-byte items are withheld under the
    # quote gate — read metadata may surface, raw text never.
    res = recall_v3(store, _req("precise bytes"))
    assert not any(payload in t for t in _texts(res))

    # The explicit archive lane: read-only → metadata-only items.
    api = VerbatimV3(store)
    out = api.browse_evidence("sA", "precise", principal_id="human:alice")
    assert not any(payload in (i.text or "") for p in out.packs
                   for i in p.items)
    assert "quote_not_granted" in out.warnings

    # Kernel: a QUOTE lease resolves denied; the denied lease cannot
    # authenticate any slice read.
    kernel = Kernel(store)
    caller = CallerV3(principal_id="human:alice")
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, caller, Verb.QUOTE, "recall", ["sA"]
        )
        assert lease.denied
        loc = EvidenceLocator(
            object_id="src-c08", revision=1,
            start_byte=0, end_byte=len(payload.encode("utf-8")),
        )
        assert _err(
            lambda: kernel.read_verified(conn, lease, [loc])
        ) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        # Even a live READ lease is refused byte service: the slice comes
        # back metadata-only (zero bytes) — exact bytes ship only under
        # QUOTE (V4-08.02).
        read_lease = kernel.resolve_access(
            conn, caller, Verb.READ, "recall", ["sA"]
        )
        assert not read_lease.denied
        slices = kernel.read_verified(conn, read_lease, [loc])
        assert len(slices) == 1
        assert slices[0].data == b""
        assert slices[0].verification == "metadata_only"
        # The payload bytes never appear in anything the slice carries.
        assert payload.encode("utf-8") not in repr(slices[0]).encode()
    api.close()


# =====================================================================
# C09 — empty delegated purposes grant no recall authority (C)
# =====================================================================


def test_c09_empty_purpose_delegation_grants_no_recall(store):
    """C09 / V4-11.01 + V4-11.02: an empty purpose SET normalizes to
    NONE — a delegate holding it authorizes nothing, at any purpose; a
    child may only ever narrow the parent's purposes.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        register_principal(conn, kind="agent", principal_id="agent:sub")
        _seed_claim(conn, "cl-c09", "sA", "src-c09", "sp-c09",
                    "evaluate-scoped evidence", _gen(store))
        parent = create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read", "quote"}, issuer_id="human:alice",
            purposes=["evaluate"], delegation_depth=1,
        )
        # Empty iterable → NONE constraint (V4-11.01): legal attenuation.
        did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:sub",
            verbs={"read"}, purposes=[],
        )
        row = conn.execute(
            "SELECT purposes_json FROM grants_v3 WHERE grant_id = ?",
            (child,),
        ).fetchone()
        assert row is not None

        # Purposes outside the parent set are a subset violation —
        # caller bug, VALIDATION (not a silent clamp).
        assert _err(lambda: delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:sub2",
            verbs={"read"}, purposes=["recall"],
        )) == ErrorCode.VALIDATION

    # The NONE-purpose child authorizes nothing — recall denies at the
    # requesting scope, indistinguishable from no grant at all.
    assert _err(
        lambda: recall_v3(store, _req("evaluate-scoped",
                                      caller="agent:sub"))
    ) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    assert _err(
        lambda: recall_v3(
            store, _req("evaluate-scoped", caller="agent:sub",
                        purpose="evaluate")
        )
    ) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # The constraint itself: SET(∅) → NONE, and NONE permits nothing.
    assert PurposeConstraint.set(()).tag is PurposeTag.NONE
    assert not PurposeConstraint.set(()).permits("recall")


# =====================================================================
# C10 — delegation cannot drop caveats or widen expiry/depth (C)
# =====================================================================


def test_c10_delegation_cannot_drop_caveats_or_widen(store):
    """C10 / V4-11.03 (+11.02): caveats are conjunctive — a child must
    retain every parent restriction and may only add; expiry may only
    shorten; depth decrements to zero and then denies.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        register_principal(conn, kind="agent", principal_id="agent:sub")
        t0 = now_us()
        parent = create_grant(
            conn, scope_id="sA", principal_id="human:alice",
            verbs={"read", "quote"}, issuer_id="human:alice",
            purposes=["recall"], caveats=["network=off"],
            expires_us=t0 + 3_600_000_000, delegation_depth=1,
        )
        # Dropping the parent's caveat is an authority expansion → denied.
        assert _err(lambda: delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:sub",
            verbs={"read"}, caveats=[],
        )) == ErrorCode.VALIDATION
        # Extending expiry past the parent → denied.
        assert _err(lambda: delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:sub",
            verbs={"read"}, caveats=["network=off"],
            expires_us=t0 + 7_200_000_000,
        )) == ErrorCode.VALIDATION
        # Legal: retain the caveat, add one, shorten expiry.
        _did, child = delegate_grant(
            conn, parent_grant_id=parent, delegate_id="agent:sub",
            verbs={"read"}, caveats=["network=off", "region=local"],
            expires_us=t0 + 1_800_000_000,
        )
        row = conn.execute(
            "SELECT caveats_json, expires_us, delegation_depth"
            " FROM grants_v3 WHERE grant_id = ?", (child,),
        ).fetchone()
        caveats = set(json.loads(row[0]))
        assert {"network=off", "region=local"} <= caveats
        assert int(row[1]) == t0 + 1_800_000_000
        assert int(row[2]) == 0  # depth consumed — no further delegation
        # Depth exhausted: the child cannot delegate onward — dead
        # authority denies identically to an absent grant.
        assert _err(lambda: delegate_grant(
            conn, parent_grant_id=child, delegate_id="agent:subsub",
            verbs={"read"},
        )) == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# =====================================================================
# C11 — an authorized identifier never widens scope (C)
# =====================================================================


def test_c11_authorized_identifier_never_widens_scope(store):
    """C11 / V4-10.04: identifier lookups intersect the authorized scope
    set — naming an entity that also exists in a forbidden scope
    contributes only the authorized scope's evidence. An omitted filter
    never means global.
    """
    with store.tx() as conn:
        _scope(conn, "sA", conv="c1")
        _scope(conn, "sB", conv="c2")
        _auth(conn, "sA")  # authorized on A only
        _seed_claim(conn, "cl-a11", "sA", "src-a11", "sp-a11",
                    "alice migrated the config", _gen(store))
        _seed_claim(conn, "cl-b11", "sB", "src-b11", "sp-b11",
                    "alice moved the forbidden file", _gen(store))
        for eid, cid, sid in (
            ("ent-a", "cl-a11", "sA"), ("ent-b", "cl-b11", "sB")
        ):
            conn.execute(
                "INSERT INTO entities(entity_id,scope_id,kind,label,"
                "created_event) VALUES(?,?,'person','user:alice',1)",
                (eid, sid),
            )
            conn.execute(
                "INSERT INTO claim_entities(claim_id,entity_id,role)"
                " VALUES(?,?,'subject')",
                (cid, eid),
            )

    # Naming the SHARED label resolves only inside authorized scopes.
    res = recall_v3(store, _req("user:alice", entity_ids=("ent-a",)))
    assert any("migrated the config" in t for t in _texts(res))
    assert not any("forbidden" in t for t in _texts(res))

    # Naming B's entity id directly still cannot pull B's claim — the
    # claim join is scope-constrained (lanes.py exact-id).
    res2 = recall_v3(store, _req("user:alice", entity_ids=("ent-b",)))
    assert not any("forbidden" in t for t in _texts(res2))

    # And no filter = authorized scopes only — sB stays invisible even
    # when the identifier is well-formed and the grant absent.
    res3 = recall_v3(store, _req("alice"))
    assert not any("forbidden" in t for t in _texts(res3))


# =====================================================================
# C12 — holding a span suppresses every dependent surface (C)
# =====================================================================


def test_c12_hold_cascades_to_claims_browse_export_and_producers(store):
    """C12 / V4-36.03 + V4-36.06: a quarantine hold on evidence
    immediately withholds every dependent surface — claims in recall,
    span text in inspection, the containing revision's bytes in the
    archive lane and portable export (byte lanes ship whole revisions
    and cannot splice out a held span, so the revision is withheld) —
    and producers re-check holds at drain: a held source's queued
    harvest fails QUARANTINED instead of admitting a claim, and a held
    envelope is withheld from the compiler as unresolved evidence.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-c12", "sA", "src-c12", "sp-c12",
                    "held dependent claim text", _gen(store))
        # A producer-input lane: pending harvest job on a second source.
        _add_source(conn, "src-c12b", "sA", b"queued for harvest")
        # Compile fixtures: ep-c12 is the unheld control; ep-c12h's first
        # action envelope is held before the episode ever compiles.
        _seed_episode(conn, "ep-c12", "sA", prefix="c12")
        _seed_episode(conn, "ep-c12h", "sA", prefix="c12h")

    reader = Scope(
        profile_id="prof", principal_id="p1", workspace_id="ws",
        conversation_id="c1", visibility=Visibility.CONVERSATION,
    )
    ing = Ingester(store, VerbatimConfig())
    # Control: everything serves while unheld.
    res0 = recall_v3(store, _req("held dependent"))
    assert any("held dependent claim text" in t for t in _texts(res0))
    with store.tx() as conn:
        r = compile_episode(conn, "ep-c12", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.CANDIDATE

    with store.tx() as conn:
        # Hold the evidence span, the producer-input source, and one
        # action envelope of the never-compiled episode.
        open_quarantine(conn, ("span", "sp-c12", 1),
                        ["rule:test_hold"], None, scope_id="sA")
        open_quarantine(conn, ("source", "src-c12b", 1),
                        ["rule:test_hold"], None, scope_id="sA")
        open_quarantine(conn, ("source_envelope", "env-c12h1", 1),
                        ["rule:test_hold"], None, scope_id="sA")
        job_c12 = ing.jobs.enqueue(
            conn, "sA", JobKind.HARVEST,
            {"source_id": "src-c12b", "revision": 1},
        )

    # claims: dependent claim withheld from recall (V3-14.10 cascade).
    res = recall_v3(store, _req("held dependent"))
    assert not any("held dependent claim text" in t for t in _texts(res))
    # inspection: span text withheld ("[unavailable]"), metadata stays.
    rep = inspect_claim(store, "cl-c12", reader)
    assert rep["evidence"][0]["text"] == "[unavailable]"
    # archive lane: the held span's containing revision ships no bytes;
    # the source-held revision is withheld the same way.
    api = VerbatimV3(store)
    out = api.browse_evidence("sA", "held dependent",
                              principal_id="human:alice")
    assert not any("held dependent" in (i.text or "")
                   for p in out.packs for i in p.items)
    assert "held_evidence_withheld" in out.warnings  # withheld, not absent
    out2 = api.browse_evidence("sA", "queued",
                               principal_id="human:alice")
    assert not any("queued" in (i.text or "")
                   for p in out2.packs for i in p.items)
    assert "held_evidence_withheld" in out2.warnings
    # export: held claim, held revisions, and the held span itself all
    # stay out of the bundle — held bytes never serialize (V4-36.03).
    bundle = export_scope(store, "sA")
    src_ids = {s["source_id"] for s in bundle["sources"]}
    assert "src-c12" not in src_ids  # span hold withholds the revision
    assert "src-c12b" not in src_ids
    assert not any(
        c["claim_id"] == "cl-c12" for c in bundle["claims"]
    )
    assert not any(
        s["span_id"] == "sp-c12" for s in bundle["spans"]
    )
    # producer input: the queued harvest on the held source fails
    # QUARANTINED — the hold wins over the pending job (V4-36.06).
    report = ing.drain_report()
    assert report["failed"] >= 1
    with store.read() as conn:
        st = conn.execute(
            "SELECT state, error_code FROM jobs WHERE job_id = ?",
            (job_c12,),
        ).fetchone()
        assert st[0] == "failed"
        assert st[1] == ErrorCode.QUARANTINED.value
        # The held envelope's payload is unavailable to the compiler —
        # its action step resolves as missing evidence, not as bytes.
        rh = compile_episode(conn, "ep-c12h", hmac_fn=store.hmac)
        assert rh.status is CompilationStatus.INCOMPLETE
        assert rh.procedure_id is None
    api.close()


# =====================================================================
# C13 — empty pipeline cannot bypass into raw sources or oversize (C)
# =====================================================================


def test_c13_empty_pipeline_never_bypasses_or_oversizes(store):
    """C13 / V4-08.07 + V4-32.10: when holds or budgets empty the
    derived pipeline, recall abstains — it never falls through to raw
    source bytes unless the caller explicitly declared the
    archive/evidence lane, and even then the same budgets and holds bind.
    """
    payload = "x" * 2000 + " needle"
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _seed_claim(conn, "cl-c13", "sA", "src-c13", "sp-c13",
                    payload, _gen(store))
        open_quarantine(conn, ("source", "src-c13", 1),
                        ["rule:test_hold"], None, scope_id="sA")

    api = VerbatimV3(store)
    # Held claim → derived recall abstains; WITHOUT an explicit evidence
    # mode the facade never routes into raw sources (V4-08.07).
    res = api.recall("sA", "needle", principal_id="human:alice")
    assert res.packs == ()
    assert not any("needle" in t for t in _texts(res))
    # Explicit declaration: the lane runs INSIDE the governed surface —
    # the held source stays withheld; honest abstention, no bypass.
    res2 = api.recall(
        "sA", "needle", principal_id="human:alice",
        budget={"modes": ("evidence",)},
    )
    assert not any("needle" in t for t in _texts(res2))
    assert "held_evidence_withheld" in res2.warnings or res2.abstained

    # Budget-bounded: unheld content under max_items=1/max_bytes=512 —
    # serialized output stays inside the request's bound, no oversize.
    with store.tx() as conn:
        _add_source(conn, "src-c13b", "sA", b"needle " + b"y" * 4000)
        conn.execute(
            "DELETE FROM quarantine WHERE object_id = 'src-c13'"
        )
    out = api.browse_evidence(
        "sA", "needle", principal_id="human:alice",
        budget={"max_bytes": 512, "max_items": 1},
    )
    for p in out.packs:
        assert int(p.serialized_bytes or 0) <= 512
        assert len(p.items) <= 1
    api.close()


# =====================================================================
# C14 — document embedding without consent/reservation never sends (C)
# =====================================================================


class _FakeTransport:
    def __init__(self, result: Optional[HttpResult] = None):
        self.result = result
        self.calls: list[dict] = []

    def request(self, method, url, *, body, headers, timeout_s):
        self.calls.append({"method": method, "url": url, "body": body})
        return self.result or HttpResult(status=404, body=b"")


def _remote_cfg(**kw) -> VerbatimConfig:
    return VerbatimConfig(
        mode=Mode.REMOTE_ASSISTED,
        judge=JudgeConfig(
            backend="rules", daily_budget_usd=Decimal("1.00")
        ),
        embedding=EmbeddingConfig(backend="cloudflare", account_id="acct"),
        **kw,
    )


def _cf_encoder(*, broker, transport) -> CloudflareEncoder:
    return CloudflareEncoder(
        EmbeddingConfig(backend="cloudflare", model="m", account_id="acct"),
        account_id="acct",
        secret_getter=lambda name: "tok",
        http=transport,
        broker=broker,
    )


def test_c14_document_dispatch_requires_permit_and_consent(store):
    """C14 / V4-12.01 + V4-12.05: a document-embedding dispatch needs a
    broker-minted permit backed by per-scope consent and an atomic budget
    reservation — no consent row means no permit, no permit means the
    encoder never touches the transport.
    """
    transport = _FakeTransport()
    probe = _cf_encoder(broker=None, transport=transport)
    broker = TransportBroker(
        store, _remote_cfg(), endpoints=[probe.endpoint_descriptor()],
        clock=FakeClock(),
    )
    enc = _cf_encoder(broker=broker, transport=transport)
    assert encoder_requires_permit(enc)

    # No permit argument at all → denied inside the encoder.
    assert _err(lambda: enc.encode(["doc text"])) == ErrorCode.EGRESS_DENIED
    # Permit requested without a consent row → denied before any I/O.
    assert _err(lambda: broker.open_dispatch(
        store, caller="test", recipient=enc.endpoint_id,
        purpose="embed_document",
        payload_digest=enc.payload_digest(["doc text"]),
        scope_ids=["sA"], max_spend=0.001,
    )) == ErrorCode.EGRESS_DENIED
    # The fake transport saw NOTHING — denial precedes network I/O.
    assert transport.calls == []
    # The reservation ledger carries no phantom reservation either.
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM dispatch_permits"
        ).fetchone()[0] == 0


# =====================================================================
# C15 — query embedding denied at zero budget even when healthy (C)
# =====================================================================


def test_c15_query_embedding_denied_at_zero_budget(store):
    """C15 / V4-12.04 + V4-29.10: a configured, healthy remote encoder
    still cannot dispatch a query embedding when the remote budget is
    zero — and 'nominally free' (``max_spend=0``) is not a permit class
    even with budget headroom.
    """
    transport = _FakeTransport()
    probe = _cf_encoder(broker=None, transport=transport)
    cfg = _remote_cfg()
    cfg = replace(
        cfg, judge=replace(cfg.judge, daily_budget_usd=Decimal("0"))
    )
    broker = TransportBroker(
        store, cfg, endpoints=[probe.endpoint_descriptor()],
        clock=FakeClock(),
    )
    enc = _cf_encoder(broker=broker, transport=transport)
    # The encoder is provisioned and healthy (credential resolves) — the
    # denial below is the BUDGET gate, not an availability failure.
    assert enc.available() is True
    grant_consent(store, "sA", processor=enc.endpoint_id,
                  purpose="embed_query")

    # Consent exists, budget is zero → dispatch denied before I/O.
    assert _err(lambda: broker.open_dispatch(
        store, caller="test", recipient=enc.endpoint_id,
        purpose="embed_query",
        payload_digest=enc.payload_digest(["q text"]),
        scope_ids=["sA"], max_spend=0.001,
    )) == ErrorCode.EGRESS_DENIED
    # Nominally-free is not a permit class — even WITH budget headroom.
    broker2 = TransportBroker(
        store, _remote_cfg(), endpoints=[probe.endpoint_descriptor()],
        clock=FakeClock(),
    )
    assert _err(lambda: broker2.open_dispatch(
        store, caller="test", recipient=enc.endpoint_id,
        purpose="embed_query",
        payload_digest=enc.payload_digest(["q text"]),
        scope_ids=["sA"], max_spend=0.0,
    )) == ErrorCode.EGRESS_DENIED
    # And the encoder itself never reaches the transport unpermitted.
    assert _err(lambda: enc.encode(["q text"])) == ErrorCode.EGRESS_DENIED
    assert transport.calls == []


# =====================================================================
# C16 — telemetry obeys consent; absent in offline profiles (C)
# =====================================================================


def test_c16_telemetry_egress_consent_gated_and_offline_absent(store):
    """C16 / V4-12.01 + V4-51.08: this build exposes no dedicated
    telemetry-export surface — every outbound byte crosses the single
    egress broker, so telemetry dispatch is consent-gated like all
    egress and structurally absent in offline profiles (no allowlisted
    endpoint, offline mode → EGRESS_DENIED). Local observability —
    capabilities, readiness, drain reports — works with networking off.

    TODO-C16: no public ``export_telemetry``/diagnostics-upload API
    exists to test directly; this test pins the boundary that would
    govern it (``TransportBroker.open_dispatch``) plus the offline-mode
    denial, rather than fabricating an unimplemented surface.
    """
    transport = _FakeTransport()
    probe = _cf_encoder(broker=None, transport=transport)
    desc = probe.endpoint_descriptor()

    # Offline profile: mode OFFLINE_RULES, zero budget, no endpoints —
    # every dispatch denied, and no endpoint is even allowlisted.
    offline_cfg = VerbatimConfig(mode=Mode.OFFLINE_RULES)
    broker_off = TransportBroker(
        store, offline_cfg, endpoints=[], clock=FakeClock()
    )
    assert _err(lambda: broker_off.open_dispatch(
        store, caller="telemetry", recipient=desc.endpoint_id,
        purpose="telemetry",
        payload_digest=probe.payload_digest(["diag"]),
        scope_ids=["sA"], max_spend=0.001,
    )) == ErrorCode.EGRESS_DENIED

    # Even with the endpoint allowlisted, offline mode denies remote
    # egress — telemetry has no offline path (V4-51.08).
    broker_off2 = TransportBroker(
        store, offline_cfg, endpoints=[desc], clock=FakeClock()
    )
    assert _err(lambda: broker_off2.open_dispatch(
        store, caller="telemetry", recipient=desc.endpoint_id,
        purpose="telemetry",
        payload_digest=probe.payload_digest(["diag"]),
        scope_ids=["sA"], max_spend=0.001,
    )) == ErrorCode.EGRESS_DENIED

    # Online but consentless → denied; consent for a DIFFERENT purpose
    # does not cover telemetry (per-purpose binding, V4-12.06).
    broker_on = TransportBroker(
        store, _remote_cfg(), endpoints=[desc], clock=FakeClock()
    )
    grant_consent(store, "sA", processor=desc.endpoint_id,
                  purpose="embed_document")
    assert _err(lambda: broker_on.open_dispatch(
        store, caller="telemetry", recipient=desc.endpoint_id,
        purpose="telemetry",
        payload_digest=probe.payload_digest(["diag"]),
        scope_ids=["sA"], max_spend=0.001,
    )) == ErrorCode.EGRESS_DENIED
    assert transport.calls == []

    # Local observability is unaffected by networking being off — and
    # the capability matrix reports OBSERVED rungs, not config wishes
    # (V4-50.02): no encoder bound + backend 'none' → dense reports
    # below-healthy with a stated reason; artifact-gated lanes never
    # claim usable indexes; the evidence lane is honestly healthy.
    api = VerbatimV3(store, offline_cfg)
    caps = api.capabilities()
    assert "lanes" in caps and caps["lanes"]
    dense = caps["lanes"]["dense"]
    assert dense["available"] is False
    assert dense["rung"] not in ("healthy", "measured", "recommended")
    assert dense["degraded_reason"]
    for name in ("sparse", "late_interaction"):
        assert caps["lanes"][name]["available"] is False
    assert caps["lanes"]["archive_evidence"]["available"] is True
    report = Ingester(store, offline_cfg).drain_report()
    assert report["processed"] == 0 and report["failed"] == 0
    api.close()
