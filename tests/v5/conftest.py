"""Shared fixtures for the SPEC_V5 acceptance suite (§24 + §36, E01–E96).

Conventions mirror ``tests/v4/`` and ``tests/test_v3_integration.py``:

- Real on-disk ``Store.create(tmp_path)`` instances — never mocked
  kernels or fake memory implementations (V5-24 hard rule 6).
- Seeded rows carry real profile-keyed HMACs: the fixture pins the
  profile key (``<db>.key``) so ``_h()`` digests verify under
  ``store.hmac`` — the same convention as ``tests/v4/test_scenarios_a.py``.
- The consumer path targets ``verbatim.Memory`` (lazy public export per
  docs/v5_contracts.md §10). While facade/modules land in parallel,
  tests exercise the underlying owned surfaces named in
  ``docs/v5_contracts.md`` (``retrieval/candidates.py`` F4-11 machinery,
  ``jobs/queue.py``, ``governance/``, ``security/quarantine.py``,
  ``storage/resolver.py``, ``embeddings/``) where those already carry
  the inherited contract, and are marked ``xfail(strict=False)``
  everywhere else — never a weakened assertion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import replace
from typing import Any, Optional

import pytest

from verbatim.api import Engine
from verbatim.api_v3 import VerbatimV3
from verbatim.config import config_from_mapping
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Scope,
    TransitionCommand,
    VerbatimError,
    Visibility,
)
from verbatim.core.types_v3 import (
    EnvelopeKind,
    Perspective,
    RecallRequestV3,
    SourceEnvelopeV3,
)
from verbatim.evidence import ingest_envelope
from verbatim.host import LocalHost
from verbatim.storage.store import Store


# ---------------------------------------------------------------------
# digest convention (tests/v4/test_scenarios_a.py)
# ---------------------------------------------------------------------

#: Pinned profile HMAC key so test-seeded rows verify under
#: ``store.hmac`` exactly like production rows.
_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


def make_store(path: str) -> Store:
    """Real on-disk store with the pinned test profile key."""
    key_path = path + ".key"
    if not os.path.exists(key_path):
        with open(key_path, "wb") as fh:
            fh.write(_TEST_HMAC_KEY)
    return Store.create(path)


@pytest.fixture
def store(tmp_path):
    s = make_store(str(tmp_path / "v5.db"))
    yield s
    s.close()


@pytest.fixture
def api(store):
    """The existing host-neutral public surface over the same store —
    the authority the V5 facade composes (contracts: one authority)."""
    return VerbatimV3(store)


@pytest.fixture
def engine(store):
    """V2/v3 write-path engine: capture → screen → harvest → admit."""
    cfg = config_from_mapping(
        {"capture": {"enabled": True, "user_messages": True}}
    )
    return Engine(
        store,
        cfg,
        LocalHost(profile_id="prof", principal_id="p1",
                  conversation_id="c1"),
    )


# ---------------------------------------------------------------------
# pending-surface markers (xfail, never a weakened assertion)
#
# Landed since the suite was drafted: schema_v5 (SCHEMA_VERSION=5),
# retrieval/v3/source_lane.py + fusion_v1.py, enrichment/, dedup/links.py,
# querying/{analyze,updates}.py, memory/{worker,controls,aliases,errors}.py,
# compat/mem0.py, jobs/coordinator.py. Still absent: the ``Memory`` facade
# (verbatim/memory/facade.py — ``from verbatim import Memory`` fails),
# memory/bootstrap.py imports but ``ensure_bootstrap`` is unimportable
# (CallerV3 imported from the wrong module — latent defect), and
# jobs/source_jobs.py. Tests below use the real landed surfaces directly.
# ---------------------------------------------------------------------

XFAIL_FACADE = pytest.mark.xfail(
    strict=False,
    reason="V5 facade pending — verbatim/memory/facade.py has not landed",
)
XFAIL_BOOTSTRAP = pytest.mark.xfail(
    strict=False,
    reason="V5 bootstrap pending — memory/bootstrap.py exists but "
    "ensure_bootstrap raises ImportError (CallerV3 import defect)",
)
XFAIL_SCHEMA_V5 = pytest.mark.xfail(
    strict=False,
    reason="V5 schema pending — storage/schema_v5.py / migration 4→5 "
    "has not landed",
)
XFAIL_SOURCE_JOBS = pytest.mark.xfail(
    strict=False,
    reason="V5 source jobs pending — jobs/source_jobs.py + v5 dispatch "
    "have not landed",
)
XFAIL_SOURCE_LANE = pytest.mark.xfail(
    strict=False,
    reason="V5 source lane pending — retrieval/v3/source_lane.py "
    "has not landed",
)
XFAIL_FUSION_V1 = pytest.mark.xfail(
    strict=False,
    reason="V5 fusion contract pending — retrieval/v3/fusion_v1.py "
    "(ranking/v1) has not landed",
)
XFAIL_ENRICHMENT = pytest.mark.xfail(
    strict=False,
    reason="V5 enrichment pending — verbatim/enrichment/ has not landed",
)
XFAIL_DEDUP = pytest.mark.xfail(
    strict=False,
    reason="V5 dedup wiring pending — dedup/links.py exists but the "
    "add pipeline never invokes it (jobs/source_jobs.py pending)",
)
XFAIL_QUERYING = pytest.mark.xfail(
    strict=False,
    reason="V5 query analysis pending — verbatim/querying/ "
    "(query_analysis/v1) has not landed",
)
XFAIL_ALIASES = pytest.mark.xfail(
    strict=False,
    reason="V5 aliases pending — memory/aliases.py "
    "(user_id/agent_id/run_id) has not landed",
)
XFAIL_ANN = pytest.mark.xfail(
    strict=False,
    reason="V5 ANN accelerator pending — no ANN index exists; exact "
    "streaming scan is the only vector path",
)
XFAIL_NEURAL = pytest.mark.xfail(
    strict=False,
    reason="V5 neural extra pending — no pinned neural artifact/runtime "
    "is provisioned",
)
XFAIL_COMPAT = pytest.mark.xfail(
    strict=False,
    reason="V5 compat shim pending — verbatim/compat/ has not landed",
)
XFAIL_MCP_V5 = pytest.mark.xfail(
    strict=False,
    reason="V5 consumer MCP surface pending — the two-tool consumer "
    "toolset is not shipped",
)
XFAIL_EVAL_V5 = pytest.mark.xfail(
    strict=False,
    reason="V5 evaluation harness pending — eval/v5/ portfolio, "
    "comparators, and statistical gates have not landed",
)
XFAIL_CONSOLIDATION = pytest.mark.xfail(
    strict=False,
    reason="V5 grounded consolidation pending — §32.3 derived views "
    "have not landed",
)
XFAIL_FEEDBACK = pytest.mark.xfail(
    strict=False,
    reason="V5 feedback weights pending — §35 bounded shadow tables "
    "for the consumer route have not landed",
)


def open_memory(path, **kw):
    """Construct the V5 consumer facade through the public export.

    Raises ``ImportError`` while ``verbatim.memory.facade`` is pending;
    ``XFAIL_FACADE``-marked tests then report xfail and flip to real
    assertions automatically once the facade lands.
    """
    from verbatim import Memory  # noqa: WPS433 — lazy contract export

    return Memory(str(path), **kw)


def add_wait(memory, text, *, attempts=25, sleep_s=0.1, **kw):
    """``memory.add`` tolerant of transient writer contention, then
    settled: under a managed worker the receipt's full obligation DAG is
    awaited so subsequent searches see a quiesced snapshot (E35-style
    determinism); external-worker stores skip the wait since nothing
    drains them inline.
    """
    import time

    last = None
    res = None
    for _ in range(attempts):
        try:
            res = memory.add(text, **kw)
            break
        except Exception as exc:  # noqa: BLE001 — typed or raw lock error
            last = exc
            msg = str(exc)
            if "locked" not in msg and "BACKPRESSURE" not in type(exc).__name__.upper():
                raise
            time.sleep(sleep_s)
    if res is None:
        raise last
    if getattr(memory, "_worker_mode", None) is not None and getattr(
        memory._worker_mode, "value", ""
    ) == "managed":
        try:
            memory.wait_ready(
                res.receipt_id,
                capabilities=[
                    "accepted", "screened", "lexical_ready",
                    "semantic_ready", "derived_ready",
                    "source_lexical_ready", "source_vector_ready",
                ],
                timeout_ms=15000,
            )
        except Exception:
            pass  # held/blocked/deferred are honest terminals too
    return res


# ---------------------------------------------------------------------
# direct seeding helpers (v4 convention: real digests, real tables)
# ---------------------------------------------------------------------


def seed_scope(conn, sid, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (sid, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, sid, pid="human:alice", purposes=("recall",),
              verbs=("read", "quote"), **kw):
    """Real governance: principal + scoped grant (§05 authority)."""
    from verbatim.governance import (
        create_grant,
        register_principal,
        seed_purposes,
    )

    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=sid, principal_id=pid, verbs=set(verbs),
        issuer_id=pid,
        purposes=None if purposes is None else list(purposes),
        **kw,
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1",
               created_us=1):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,?)",
        (source_id, "test", "user_message", scope_id, speaker, created_us),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, _h(payload)),
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
               recorded_until=None, created_event=1):
    """Claim + revision + primary evidence + FTS row (real digests)."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, created_event),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until) VALUES(?,?,?,?,?,?)",
        (claim_id, rev, state, None, recorded_from, recorded_until),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def gen(store) -> int:
    return store.projection_generation()


# ---------------------------------------------------------------------
# v5 projection-plane seeding (schema_v5 landed: SCHEMA_VERSION=5)
# ---------------------------------------------------------------------


def seed_source_state(conn, source_id, namespace, disposition="active",
                      control_version=0, mutation_head="1", **kw):
    """Insert a ``source_state`` control row (§14.3, contracts §3)."""
    from verbatim.core.time import now_us, rfc3339

    ts = rfc3339(now_us())
    conn.execute(
        "INSERT INTO source_state(source_id,namespace,control_version,"
        "mutation_head,disposition,superseded_by,effective_at,known_at,"
        "valid_from,valid_to,updated_at,producer)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            source_id, namespace, control_version, str(mutation_head),
            disposition, kw.get("superseded_by"), kw.get("effective_at"),
            ts, kw.get("valid_from"), kw.get("valid_to"), ts,
            kw.get("producer", "enrich/v1"),
        ),
    )


def seed_projection(conn, source_id, revision, namespace, text,
                    generation=1):
    """Insert a ``source_lexical_projection`` row with the real
    norm/v1 tokens + digest (the values a T1 worker would persist)."""
    from verbatim.enrichment import normalize_text, normalized_digest

    toks = normalize_text(text)
    conn.execute(
        "INSERT INTO source_lexical_projection(source_id,revision,scope_id,"
        "generation,tokens,doc_len,digest) VALUES(?,?,?,?,?,?,?)",
        (source_id, revision, namespace, generation, toks,
         len(toks.split()), normalized_digest(text)),
    )


def seed_entity_posting(conn, namespace, entity, source_id, revision,
                        kind="entity", offsets=None, generation=1):
    conn.execute(
        "INSERT INTO entity_postings(namespace,entity,entity_kind,"
        "source_id,revision,offsets,generation) VALUES(?,?,?,?,?,?,?)",
        (namespace, entity, kind, source_id, revision,
         json.dumps(offsets or [0, max(1, len(entity))]), generation),
    )


def make_controls(store, namespace, caller="human:alice",
                  purposes=None, **kw):
    """A real ``MemoryControls`` bound to ``namespace`` with a granted
    caller (read/quote/admin/ingest/review — the consumer verb set)."""
    from verbatim.governance import CallerV3
    from verbatim.memory.controls import MemoryControls

    with store.tx() as conn:
        seed_auth(
            conn, namespace, pid=caller, purposes=purposes,
            verbs=("read", "quote", "admin", "ingest", "review"),
        )
    return MemoryControls(
        store, caller=CallerV3(principal_id=caller),
        namespace=namespace, **kw
    )


# ---------------------------------------------------------------------
# v2-candidate-layer helpers (the F4-11 machinery the V5 source lane
# reuses — docs/v5_contracts.md §6)
# ---------------------------------------------------------------------


def v2_request(query: str, scope: Scope, **kw):
    """A minimal v2 recall request for ``candidates.gather``."""
    from verbatim.core.types import RecallRequest

    return RecallRequest(query=query, scope=scope, **kw)


def v2_plan(query: str, request):
    from verbatim.core.time import now_us
    from verbatim.retrieval.query import analyze

    return analyze(query, request, now_us())


def gather_scores(store, request, plan):
    """``(ordered claim ids, claim_id -> lexical score)`` from the real
    eligible-corpus BM25 path (``retrieval/candidates.py`` F4-11)."""
    from verbatim.retrieval import candidates as cand

    with store.read() as conn:
        out = cand.gather(
            conn, store, plan, request, gen(store)
        )
    notes = out.capability_notes.get("lexical", {})
    order = [
        h.claim_id
        for h in sorted(
            out.values(),
            key=lambda hit: min(hit.source_ranks.values()),
        )
    ]
    return order, dict(notes.get("scores", {})), out


# ---------------------------------------------------------------------
# v3 write-path + recall helpers (tests/test_v3_integration.py style)
# ---------------------------------------------------------------------


def capture(store, kind, text, *, scope_id="sA", actor="u1", ext=None,
            event_us=1000, meta=None):
    """Ingest one envelope through the real screened write channel."""
    env = SourceEnvelopeV3(
        kind=kind, scope_id=scope_id, actor_principal=actor,
        perspective=Perspective(asserter=actor),
        content=text.encode("utf-8"), media_type="text/plain",
        host_id="h1", session_id="ss1", external_id=ext,
        event_us=event_us, receipt_us=event_us + 1,
        metadata=meta or {},
    )
    with store.tx() as conn:
        return ingest_envelope(conn, store, env)


def admit_all(engine, scope_id="sA"):
    """Operator approval over pending heads + FTS rebuild
    (tests/test_v3_integration.py convention)."""
    with engine.store.read() as conn:
        heads = dict(
            conn.execute(
                "SELECT claim_id, revision FROM claim_revisions"
                " WHERE state='pending'"
            ).fetchall()
        )
    scope = Scope(profile_id="prof", principal_id="p1",
                  conversation_id="c1", visibility=Visibility.CONVERSATION)
    for cid, rev in heads.items():
        engine.apply_transition(
            TransitionCommand(
                claim_id=cid, expected_revision=rev, effect="admit",
                actor_id="op", reason="operator approval",
            ),
            scope=scope,
        )
    with engine.store.tx() as conn:
        engine._ingester.jobs.enqueue(
            conn, scope_id, JobKind.REINDEX, {}, dedup_key=b"rdx"
        )
    engine._ingester.run_pending(limit=64)


def v3_req(query="deploy", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def items(result):
    return [i for p in result.packs for i in p.items]


def texts(result):
    return [i.text or "" for p in result.packs for i in p.items]


def item_body(text: str) -> dict:
    """Parse one serialized pack item's untrusted-JSON body
    (tests/v4 convention)."""
    s = text
    s = s[s.index(">") + 1:] if s.startswith("<memory_evidence") else s
    if s.endswith("</memory_evidence>"):
        s = s[: -len("</memory_evidence>")]
    return json.loads(s)


def err_code(fn) -> ErrorCode:
    with pytest.raises(VerbatimError) as ei:
        fn()
    return ei.value.code
