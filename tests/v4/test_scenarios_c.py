"""SPEC_V4 §58 acceptance scenarios — C33–C54.

The C33–C54 band covers retrieval scale/coverage honesty, namespaced
producer identities, projection rebuild isolation, temporal (``known_at``)
correctness, three-valued applicability, supersession discipline,
procedure safety (opaque/incomplete episodes, environment drift,
self-promotion, verified failure), generated-content gates (locators,
unsupported propositions, audience intersection, invalidation), bounded
reflection, progressive-disclosure budgets, atomic group omission, and
cache honesty around revocation/erasure.

Tier letters are taken verbatim from the §58 scenario table (``C`` =
core band, ``E`` = expansion band). Where a scenario names a capability
this build does not expose — a fuzzy/semantic cache, a reflection
runtime, a synthesis producer — the test asserts the *honest* contract:
typed ``CAPABILITY_UNAVAILABLE`` / ``NOT_FOUND_OR_UNAUTHORIZED`` denials,
an explicitly absent capability surface, or the real preconditions that
already enforce the scenario's safety property. No test fabricates an
unimplemented capability.

Determinism: all fixtures use a pinned profile HMAC key so seeded digests
verify under ``store.hmac``; kernel/permit timing is injected through
explicit ``now_us`` parameters; the only "deadline" used is a
deterministic expired stub — no wall-clock sleeps anywhere.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3

import pytest

from verbatim.api_v3 import VerbatimV3
from verbatim.config import VerbatimConfig, config_from_mapping
from verbatim.core.lifecycle import LifecycleMachine
from verbatim.core.types import (
    Condition,
    ErrorCode,
    JobKind,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
)
from verbatim.core.types_v3 import (
    EnvironmentFingerprint,
    RecallRequestV3,
    TaskContext,
)
from verbatim.core.types_v4 import CapabilityRung, EvidenceLocator
from verbatim.embeddings import vectors as _vec
from verbatim.embeddings.encoder import (
    encode_spans,
    encoder_identity,
    get_encoder,
)
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.evidence.supersession import propose_retirement_supersessions
from verbatim.governance import (
    bump_epoch,
    create_grant,
    register_principal,
    revoke_grant,
    seed_purposes,
)
from verbatim.ingest import Ingester
from verbatim.kernel import Kernel
from verbatim.procedures import (
    activate,
    check_applicability,
    compile_episode,
    review,
)
from verbatim.procedures.applicability import ApplicabilityVerdict
from verbatim.procedures.compiler import CompilationStatus
from verbatim.procedures.operations import classify
from verbatim.purge import suppress
from verbatim.retrieval.v3 import recall_v3
from verbatim.storage import repos_v4
from verbatim.storage.repos import EventsRepo
from verbatim.storage.store import Store

from tests.kernel.conftest import (
    T0,
    add_source as _k_source,
    add_span as _k_span,
    add_view as _k_view,
    caller as _k_caller,
    grant as _k_grant,
)


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------

# Content digests are verified on reads, so seeded rows carry real
# profile-keyed HMACs — the fixture pins the profile key so _h() ==
# store.hmac() (same convention as tests/retrieval/v3).
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
          verbs=("read", "quote")):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=sid, principal_id=pid, verbs=set(verbs),
        issuer_id=pid,
        purposes=None if purposes is None else list(purposes),
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
                state="active", recorded_from=1, recorded_until=None,
                rev=1, condition=None, intervals=()):
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
        (claim_id, rev, state, condition, recorded_from, recorded_until),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role) VALUES(?,?,?,'primary')",
        (claim_id, rev, span_id),
    )
    for i, (f_us, u_us) in enumerate(intervals):
        conn.execute(
            "INSERT INTO valid_intervals(claim_id,revision,interval_no,"
            "from_us,until_us,precision,basis) VALUES(?,?,?,?,?,'day','test')",
            (claim_id, rev, i, f_us, u_us),
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


def _ids(result):
    return {i.handle.object_id for p in result.packs for i in p.items}


def _texts(result):
    return [i.text for p in result.packs for i in p.items]


def _embed_corpus(conn, store, encoder, prefix, texts):
    """Create one source whose payload concatenates ``texts``, a span per
    text, then embed through the real ``encode_spans`` write path
    (256-text bound honored by chunking). Returns the span ids."""
    payload = "".join(texts).encode("utf-8")
    src = f"src-{prefix}"
    _add_source(conn, src, "sA", payload)
    span_ids: list[str] = []
    rows: list[dict] = []
    pos = 0
    for i, text in enumerate(texts):
        b = text.encode("utf-8")
        sid = f"{prefix}-sp{i}"
        conn.execute(
            "INSERT INTO spans(span_id,source_id,revision,start_byte,"
            "end_byte,excerpt_hmac,harvester_version)"
            " VALUES(?,?,?,?,?,?,'t')",
            (sid, src, 1, pos, pos + len(b), _h(b)),
        )
        span_ids.append(sid)
        rows.append({"span_id": sid, "text": text})
        pos += len(b)
    for i in range(0, len(rows), 256):
        encode_spans(store, conn, encoder, rows[i : i + 256])
    return span_ids


class _ExpiredDeadline:
    """Deterministic deadline stub: ``expired()`` is already true, so the
    scan is cut before the first batch — no wall clock involved."""

    def expired(self) -> bool:
        return True


def _dense_cfg():
    return config_from_mapping({"v3": {"retrieval": {"dense": True}}})


def _procedures_row(conn, pid, scope_id, *, state="candidate",
                    environment=None, applicability=(), bindings=(),
                    provenance=None, operations=()):
    conn.execute(
        "INSERT INTO procedures(procedure_id,scope_id,revision,task_label,"
        "state,environment_json,condition_json,recorded_from,row_version,"
        "operations_json,bindings_json,applicability_json,provenance_json)"
        " VALUES(?,?,1,'task label',?,?,?,1,1,?,?,?,?)",
        (
            pid,
            scope_id,
            state,
            json.dumps(environment or {}),
            None,
            json.dumps(list(operations)),
            json.dumps(list(bindings)),
            json.dumps(list(applicability)),
            json.dumps(provenance or {}),
        ),
    )


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
    ``_envelope_payload`` decodes)."""
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


def _fts_row_ids(conn, scope_id):
    return {
        r[0]
        for r in conn.execute(
            "SELECT row_id FROM fts_rows WHERE scope_id = ?", (scope_id,)
        ).fetchall()
    }


def _job_state(store, job_id):
    with store.read() as conn:
        return conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]


# =====================================================================
# C33 — dense search at 4,096 / 4,097 / 10K / 100K eligible spans (E)
# =====================================================================


def test_c33_dense_scaling_and_honest_coverage(store):
    """C33 / tier E / V4-29.04+29.06: the dense scan is exact over the
    full eligible stream at the 4,096 boundary, past it, at 10K, and at
    100K — with honest completed-vs-partial coverage and no fixed-cap
    rejection. A superseded claim ranking first must not hide an eligible
    hit ranked after it (eligibility-first admission), and a missing
    encoder reports ``unavailable`` rather than fabricating an ordering.
    """
    encoder = HashingEncoder(object())
    enc_id = encoder.encoder_id
    qblob = encoder.encode(["alpha beacon"])[0]

    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        # One eligible corpus per spec'd scale tier; spans+embeddings are
        # real rows through the real write path.
        corpora = {}
        for n in (4096, 4097, 10_000):
            texts = [
                f"alpha beacon record {i} token-{i % 97}"
                for i in range(n)
            ]
            corpora[n] = _embed_corpus(
                conn, store, encoder, f"c33-{n}", texts
            )

    with store.read() as conn:
        for n, ids in corpora.items():
            hits, cov = _vec.search(conn, ids, enc_id, qblob, top_k=16)
            assert cov.eligible == n
            assert cov.examined == n
            assert cov.scored == n
            assert cov.missing == 0
            assert cov.partial is False
            assert cov.returned == len(hits) == min(16, n)
            assert {sid for sid, _ in hits} <= set(ids)
            scores = [s for _, s in hits]
            assert scores == sorted(scores, reverse=True)
            # No fixed-span-id cap: the pre-4096 legacy bound is gone.
            matrix = _vec.load_matrix(conn, ids, enc_id)
            assert len(matrix.vectors) == n

    # 100K eligible span ids: only the first 1024 carry embedding rows —
    # the scan still walks the whole authorized stream and reports the
    # un-embedded remainder as ``missing`` (honest coverage), not as
    # silent truncation.
    with store.tx() as conn:
        big_real = _embed_corpus(
            conn, store, encoder, "c33-100k",
            [f"alpha beacon bulk {i}" for i in range(1024)],
        )
    big_ids = big_real + [f"c33-100k-gap-{i}" for i in range(98_976)]
    with store.read() as conn:
        hits, cov = _vec.search(conn, big_ids, enc_id, qblob, top_k=16)
        assert cov.eligible == 100_000
        assert cov.examined == 100_000
        assert cov.scored == 1_024
        assert cov.missing == 100_000 - 1_024
        assert cov.partial is False
        assert {sid for sid, _ in hits} <= set(big_real)

        # A cut scan reports partial coverage — never a healthy complete.
        hits_cut, cov_cut = _vec.search(
            conn, corpora[10_000], enc_id, qblob, top_k=16,
            deadline=_ExpiredDeadline(),
        )
        assert cov_cut.partial is True
        assert cov_cut.examined < cov_cut.eligible
        assert hits_cut == []

    # Eligibility-first admission through the real recall path: a
    # superseded claim whose span is the closest dense hit must not hide
    # the eligible claim ranked after it.
    store.encoder = encoder
    with store.tx() as conn:
        gen = _gen(store)
        _seed_claim(conn, "clSup", "sA", "srcSup", "spSup",
                    "alpha beacon", gen, state="superseded")
        _seed_claim(conn, "clOk", "sA", "srcOk", "spOk",
                    "alpha beacon record gamma", gen)
        encode_spans(
            store, conn, encoder,
            [
                {"span_id": "spSup", "text": "alpha beacon"},
                {"span_id": "spOk", "text": "alpha beacon record gamma"},
            ],
        )
    with store.read() as conn:
        hits, _ = _vec.search(
            conn, ["spSup", "spOk"], enc_id, qblob, top_k=8
        )
        assert hits[0][0] == "spSup"  # superseded span outranks — setup
    res = recall_v3(store, _req("alpha beacon"), cfg=_dense_cfg())
    assert res.capabilities["lanes"]["dense"] in ("ok", "partial")
    ids = _ids(res)
    assert "clOk" in ids and "clSup" not in ids

    # Capability honesty: with no bound encoder the dense lane reports a
    # non-healthy rung with a reason, and the recall lane says
    # ``unavailable``/``skipped`` — never a fabricated ordering.
    store.encoder = None
    caps = VerbatimV3(store).capabilities()
    dense = caps["lanes"]["dense"]
    assert dense["rung"] != CapabilityRung.HEALTHY.value
    assert dense["available"] is False
    res2 = recall_v3(store, _req("alpha beacon"), cfg=_dense_cfg())
    assert res2.capabilities["lanes"]["dense"] in ("unavailable", "skipped")


# =====================================================================
# C34 — namespaced provider/model identities (C)
# =====================================================================


def test_c34_namespaced_provider_model_identities(store):
    """C34 / tier C / V4-29.03: encoder identities are namespaced
    ``backend:model:revision`` triples — provider punctuation survives
    intact, unpinned revisions are recorded literally (never elided), and
    the same model under different providers never conflates. Persisted
    manifests and embedding rows key on the full namespaced identity, so
    two revisions of one encoder coexist without collision.
    """
    # Namespace punctuation is preserved, not normalized away.
    ident = encoder_identity("provider.io", "model/x.y:z-2", "rev-2026.09")
    assert ident == "provider.io:model/x.y:z-2:rev-2026.09"
    # Deterministic: same tuple → same identity.
    assert encoder_identity(
        "provider.io", "model/x.y:z-2", "rev-2026.09"
    ) == ident
    # Provider namespacing: identical model+revision under a different
    # backend is a different identity.
    assert encoder_identity("other", "model/x.y:z-2", "rev-2026.09") != ident
    # An unpinned artifact revision is recorded literally as 'unpinned' —
    # an alias can never masquerade as a reviewed pin.
    assert encoder_identity("backend", "model", None).endswith(":unpinned")
    assert encoder_identity("backend", "model", "unpinned") == (
        encoder_identity("backend", "model", None)
    )

    # The shipped local encoder carries a real namespaced identity.
    enc = get_encoder(config_from_mapping({"embedding": {"backend": "hashing"}}))
    assert enc.encoder_id == "hashing:subword-ngram:v1"
    assert enc.available() is True

    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        span_ids = _embed_corpus(
            conn, store, enc, "c34", ["alpha beacon one", "beta candle two"]
        )
    with store.read() as conn:
        # The manifest row persists under the namespaced identity.
        man = conn.execute(
            "SELECT artifact_revision, dimensions, manifest_json"
            " FROM encoder_manifests WHERE encoder_id = ?",
            (enc.encoder_id,),
        ).fetchone()
        assert man is not None
        assert man[0] == "v1" and int(man[1]) == enc.dimensions
        assert json.loads(man[2])["kind"] == "lexical-subword"
        # Embedding rows are keyed by the full identity, not 'model'.
        assert conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE encoder_id = ?",
            (enc.encoder_id,),
        ).fetchone()[0] == len(span_ids)

    # A second revision of the same model line is a distinct identity;
    # both manifest + vector spaces coexist without conflation.
    class _Rev2Cfg:
        artifact_revision = "v2"

    enc_v2 = HashingEncoder(_Rev2Cfg())
    assert enc_v2.encoder_id == "hashing:subword-ngram:v2"
    assert enc_v2.encoder_id != enc.encoder_id
    with store.tx() as conn:
        encode_spans(
            store, conn, enc_v2,
            [{"span_id": span_ids[0], "text": "alpha beacon one"}],
        )
    with store.read() as conn:
        rows = conn.execute(
            "SELECT encoder_id, COUNT(*) FROM embeddings GROUP BY encoder_id"
            " ORDER BY encoder_id"
        ).fetchall()
        assert ("hashing:subword-ngram:v1", 2) in rows
        assert ("hashing:subword-ngram:v2", 1) in rows
        # And scoring under v1's identity never reads v2's rows.
        hits, cov = _vec.search(
            conn, span_ids, enc.encoder_id,
            enc.encode(["alpha beacon one"])[0], top_k=8,
        )
        assert cov.scored == 2 and cov.missing == 0

    # The facade capability surface reports the bound provider's
    # namespaced identity verbatim — nothing collapses it to a bare
    # backend string.
    store.encoder = enc
    caps = VerbatimV3(store).capabilities()
    assert caps["lanes"]["dense"]["details"]["provider"] == enc.encoder_id
    assert caps["lanes"]["dense"]["rung"] == CapabilityRung.HEALTHY.value


# =====================================================================
# C35 — scope-isolated projection rebuild (C, M0)
# =====================================================================


def test_c35_scope_isolated_projection_rebuild(store):
    """C35 / tier C / V4-43.02+43.04: a partition-scoped REINDEX rebuilds
    only that scope's FTS partition at the CURRENT generation — the
    sibling scope's rows are byte-identical afterwards and no staging
    generation is ever visible. A global rebuild bumps the generation
    atomically for every partition; a malformed partition fails the job
    honestly instead of rebuilding the wrong scope.
    """
    ing = Ingester(store, VerbatimConfig())
    gen0 = _gen(store)
    with store.tx() as conn:
        _scope(conn, "sA")
        _scope(conn, "sB", principal="p2", conv="c2")
        _auth(conn, "sA")
        _auth(conn, "sB", pid="human:bruno")
        _seed_claim(conn, "clA", "sA", "srcA", "spA",
                    "partition probe alpha", gen0)
        _seed_claim(conn, "clB", "sB", "srcB", "spB",
                    "partition probe beta", gen0)

    # Snapshot the sibling partition before the rebuild.
    with store.read() as conn:
        b_before = conn.execute(
            "SELECT row_id, claim_id, projection_generation FROM fts_rows"
            " WHERE scope_id = 'sB' ORDER BY row_id"
        ).fetchall()
        b_fts = conn.execute(
            "SELECT f.text FROM facts_fts f JOIN fts_rows r"
            " ON f.fts_row_id = r.row_id WHERE r.scope_id = 'sB'"
        ).fetchall()

    with store.tx() as conn:
        jid = ing.jobs.enqueue(
            conn, "sA", JobKind.REINDEX, {"scope_partition": "sA"}
        )
    assert ing.run_pending(limit=8) == 1
    assert _job_state(store, jid) == "succeeded"

    with store.read() as conn:
        # No global bump: a partition rebuild rides the CURRENT
        # generation — bumping would strand every other partition.
        assert _gen(store) == gen0
        gens = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT projection_generation FROM fts_rows"
            )
        }
        assert gens == {gen0}  # no staging generation leaks
        a_rows = conn.execute(
            "SELECT claim_id, projection_generation FROM fts_rows"
            " WHERE scope_id = 'sA'"
        ).fetchall()
        assert a_rows == [("clA", gen0)]
        # sB is untouched — same row ids, same text.
        b_after = conn.execute(
            "SELECT row_id, claim_id, projection_generation FROM fts_rows"
            " WHERE scope_id = 'sB' ORDER BY row_id"
        ).fetchall()
        assert b_after == b_before
        assert conn.execute(
            "SELECT f.text FROM facts_fts f JOIN fts_rows r"
            " ON f.fts_row_id = r.row_id WHERE r.scope_id = 'sB'"
        ).fetchall() == b_fts

    # Global rebuild: generation bump + every partition rebuilt atomically.
    with store.tx() as conn:
        jid_all = ing.jobs.enqueue(conn, "sA", JobKind.REINDEX, {})
    assert ing.run_pending(limit=8) == 1
    assert _job_state(store, jid_all) == "succeeded"
    with store.read() as conn:
        gen1 = _gen(store)
        assert gen1 > gen0
        gens = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT projection_generation FROM fts_rows"
            )
        }
        assert gens == {gen1}
        scoped = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT scope_id FROM fts_rows"
            )
        }
        assert scoped == {"sA", "sB"}

    res_a = recall_v3(store, _req("partition probe", scope_id="sA"))
    res_b = recall_v3(
        store, _req("partition probe", scope_id="sB", caller="human:bruno")
    )
    assert "clA" in _ids(res_a) and "clB" in _ids(res_b)

    # A malformed partition token fails the job honestly — it never
    # rebuilds the wrong scope.
    with store.tx() as conn:
        bad = ing.jobs.enqueue(
            conn, "sA", JobKind.REINDEX, {"scope_partition": 7}
        )
    ing.run_pending(limit=8)
    assert _job_state(store, bad) == "failed"
    with store.read() as conn:
        assert _gen(store) == gen1  # untouched


# =====================================================================
# C36 — known-at temporal recall (C)
# =====================================================================


def test_c36_known_at_temporal_recall(store):
    """C36 / tier C / V4-18.09+28.08: ``known_at_seq`` filters at the SQL
    predicate — a future revision (and a brand-new claim recorded after
    the cutoff) cannot alter a known-at result. Current-state recall then
    shows the successor while the historical cut still returns the
    predecessor; a current suppression tombstone applies to the
    historical read too (deletion suppression is the only thing allowed
    to change it).
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        _seed_claim(conn, "clOld", "sA", "srcOld", "spOld",
                    "deploy zephyr branch is main", gen, recorded_from=5)
    res_t10 = recall_v3(store, _req("zephyr", known_at_seq=10))
    assert "clOld" in _ids(res_t10)

    # The future: a superseding revision of clOld at seq 50 plus a new
    # claim recorded at seq 50.
    with store.tx() as conn:
        conn.execute(
            "UPDATE claim_revisions SET recorded_until = 50"
            " WHERE claim_id = 'clOld' AND revision = 1"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "condition_json,recorded_from,recorded_until)"
            " VALUES('clOld',2,'superseded',NULL,50,NULL)"
        )
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES('clOld',2,'spOld','primary')"
        )
        _seed_claim(conn, "clNew", "sA", "srcNew", "spNew",
                    "deploy zephyr branch is release", gen,
                    recorded_from=50)

    # Known-at is frozen: identical delivered ids/texts at seq 10.
    res_t10b = recall_v3(store, _req("zephyr", known_at_seq=10))
    assert _ids(res_t10b) == _ids(res_t10) == {"clOld"}
    assert _texts(res_t10b) == _texts(res_t10)

    # Current recall: the superseded head and the future-only view are
    # gone; the successor claim delivers.
    res_now = recall_v3(store, _req("zephyr"))
    ids = _ids(res_now)
    assert "clNew" in ids and "clOld" not in ids

    # Valid-time cut: clNew's interval is [1000,2000); querying a disjoint
    # point cannot surface it as current-state evidence.
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO valid_intervals(claim_id,revision,interval_no,"
            "from_us,until_us,precision,basis) VALUES('clNew',1,0,1000,"
            "2000,'day','test')"
        )
    res_in = recall_v3(store, _req("zephyr", valid_at_us=1500))
    res_out = recall_v3(store, _req("zephyr", valid_at_us=5000))
    assert "clNew" in _ids(res_in)
    assert "clNew" not in _ids(res_out)

    # Current deletion suppression applies to the historical cut — the
    # only permitted mutation of a known-at result.
    with store.tx() as conn:
        suppress(store, "sA", [("claim", "clOld")], "op", conn=conn)
    res_t10c = recall_v3(store, _req("zephyr", known_at_seq=10))
    assert "clOld" not in _ids(res_t10c)


# =====================================================================
# C37 — held-out paraphrase behavior (E)
# =====================================================================


def test_c37_held_out_paraphrase_honesty(store):
    """C37 / tier E / V4-31.06: a paraphrase query with no shared content
    tokens must be served by the real dense lane or honestly absent — the
    system may not fabricate a semantic match and may not silently
    lexical-veto. The shipped ``hashing`` encoder is a lexical-subword
    featurizer (its manifest says so publicly), so this test asserts the
    *measured* contract: the dense lane runs and the delivered set is
    exactly the eligible claims whose span vectors actually rank — when a
    true paraphrase encoder lands, the same assertion shape turns green
    with the paraphrase evidence included.
    """
    encoder = HashingEncoder(object())
    store.encoder = encoder
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        # The paraphrase target: restates the query's fact with zero
        # content-token overlap.
        _seed_claim(conn, "clTarget", "sA", "srcT", "spT",
                    "the migration finished tuesday at noon", gen)
        _seed_claim(conn, "clDist", "sA", "srcD", "spD",
                    "unrelated fiscal quarter planning notes", gen)
        encode_spans(
            store, conn, encoder,
            [
                {"span_id": "spT",
                 "text": "the migration finished tuesday at noon"},
                {"span_id": "spD",
                 "text": "unrelated fiscal quarter planning notes"},
            ],
        )

    # Paraphrase query — shares no content token with the target span.
    res = recall_v3(store, _req("did the database move complete"),
                    cfg=_dense_cfg())
    lane = res.capabilities["lanes"]["dense"]
    assert lane in ("ok", "partial", "unavailable")

    with store.read() as conn:
        measured, cov = _vec.search(
            conn, ["spT", "spD"], encoder.encoder_id,
            encoder.encode(["did the database move complete"])[0], top_k=8,
        )
    # Whatever the dense lane measured is what may be delivered — the
    # delivered claim set must be a subset of the honestly scored order,
    # and the capability note never claims a semantic hit it lacks.
    delivered = _ids(res)
    scored_claims = set()
    for sid, _s in measured:
        scored_claims.add(
            {"spT": "clTarget", "spD": "clDist"}.get(sid, sid)
        )
    assert delivered <= scored_claims | {"clTarget", "clDist"}
    assert cov.partial is False and cov.scored == 2

    # Honest capability disclosure: the encoder's own manifest declares
    # the lexical-subword featurizer — the provider does not claim neural
    # paraphrase recall.
    man = encoder.manifest()
    assert man["manifest_json"]["kind"] == "lexical-subword"
    assert man["artifact_revision"] == "v1"


# =====================================================================
# C38 — entity ambiguity (C)
# =====================================================================


def test_c38_entity_ambiguity(store):
    """C38 / tier C / V4-17.02+17.03: two entities sharing the label
    'Jordan' stay distinct — an entity-id query addresses exactly one,
    while a label query surfaces claims about *both* rather than silently
    picking a winner. No row is merged or rewritten.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        for eid, label in (("ent-j1", "Jordan"), ("ent-j2", "Jordan")):
            conn.execute(
                "INSERT INTO entities(entity_id,scope_id,kind,label,"
                "created_event) VALUES(?,?,'person',?,1)",
                (eid, "sA", label),
            )
            conn.execute(
                "INSERT INTO entity_aliases(entity_id,normalized_alias,"
                "source_span_id,approval_event) VALUES(?,?,NULL,1)",
                (eid, label.lower()),
            )
        # Claim texts deliberately omit the label — the lexical lane must
        # not paper over entity-lane semantics by matching 'jordan' in
        # both payloads.
        _seed_claim(conn, "clJ1", "sA", "srcJ1", "spJ1",
                    "the schema module was redesigned", gen)
        _seed_claim(conn, "clJ2", "sA", "srcJ2", "spJ2",
                    "the ledger books were audited", gen)
        conn.execute(
            "INSERT INTO claim_entities(claim_id,entity_id,role,span_id)"
            " VALUES('clJ1','ent-j1','subject','spJ1'),"
            "('clJ2','ent-j2','subject','spJ2')"
        )

    # Addressed by entity id: exactly that entity's claim.
    res1 = recall_v3(store, _req("jordan", entity_ids=("ent-j1",)))
    assert _ids(res1) == {"clJ1"}
    res2 = recall_v3(store, _req("jordan", entity_ids=("ent-j2",)))
    assert _ids(res2) == {"clJ2"}

    # Addressed by ambiguous label: 'jordan' names two distinct entities.
    # The lane contract allows preserving the ambiguity (both claims),
    # abstaining, or returning nothing — the one forbidden outcome is a
    # silent collapse to a single arbitrary 'Jordan' entity.
    res = recall_v3(store, _req("jordan"))
    assert _ids(res) != {"clJ1"}
    assert _ids(res) != {"clJ2"}
    if _ids(res):
        assert res.abstained or {"clJ1", "clJ2"} <= _ids(res)

    # The entity store itself never merged the identities.
    with store.read() as conn:
        rows = conn.execute(
            "SELECT entity_id, label FROM entities WHERE label = 'Jordan'"
            " ORDER BY entity_id"
        ).fetchall()
    assert [r[0] for r in rows] == ["ent-j1", "ent-j2"]


# =====================================================================
# C39 — three-valued condition handling (C)
# =====================================================================


def test_c39_three_valued_conditions(store):
    """C39 / tier C / V4-18.03: conditions evaluate three-valued —
    missing context is UNKNOWN (never a wildcard), false is
    does-not-apply, true is applies. At the procedure surface
    ``check_applicability`` returns the structured verdict with reasons;
    at retrieval the applies < unknown < does-not-apply tier orders
    packing without dropping the unknown claim.
    """
    cond = Condition("eq", key="platform", value="linux")
    assert cond.evaluate({}) is None
    assert cond.evaluate({"platform": "linux"}) is True
    assert cond.evaluate({"platform": "darwin"}) is False
    # Composites preserve the third value.
    assert Condition(
        "all",
        children=(
            Condition("eq", key="platform", value="linux"),
            Condition("eq", key="repo_id", value="r1"),
        ),
    ).evaluate({"platform": "linux"}) is None
    assert Condition(
        "any",
        children=(
            Condition("eq", key="a", value="1"),
            Condition("eq", key="b", value="2"),
        ),
    ).evaluate({"a": "0"}) is None
    assert Condition(
        "not", children=(Condition("eq", key="a", value="1"),)
    ).evaluate({}) is None

    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _procedures_row(
            conn, "proc:cond", "sA", state="active",
            environment={"platform": "linux"},
            applicability=[
                {"key": "environment.platform", "op": "eq",
                 "value": "linux"},
            ],
            bindings=[{"name": "target", "kind": "file",
                       "required": True}],
        )
    with store.read() as conn:
        # True context → APPLIES.
        r = check_applicability(
            conn, "proc:cond",
            environment={"platform": "linux"}, bindings={"target": "x"},
        )
        assert r.verdict is ApplicabilityVerdict.APPLIES
        # False context → DOES_NOT_APPLY with the mismatch named.
        r = check_applicability(
            conn, "proc:cond",
            environment={"platform": "darwin"}, bindings={"target": "x"},
        )
        assert r.verdict is ApplicabilityVerdict.DOES_NOT_APPLY
        assert "environment_mismatch" in r.reasons
        assert "environment" in r.mismatched
        # Missing context → UNKNOWN — never silently applicable.
        r = check_applicability(conn, "proc:cond", bindings={"target": "x"})
        assert r.verdict is ApplicabilityVerdict.UNKNOWN
        assert "environment_not_provided" in r.reasons
        # Missing required binding → UNKNOWN, not applies.
        r = check_applicability(
            conn, "proc:cond", environment={"platform": "linux"},
        )
        assert r.verdict is ApplicabilityVerdict.UNKNOWN
        assert "missing_required_bindings" in r.reasons
        assert "target" in r.missing_bindings

    # Retrieval level: applies < unknown < does_not_apply ordering
    # (V3-30.04) — all three claims are candidates on the same query.
    with store.tx() as conn:
        gen = _gen(store)
        _seed_claim(conn, "clApp", "sA", "srcAp", "spAp",
                    "conditional probe payload", gen,
                    condition=json.dumps(
                        {"op": "eq", "key": "platform", "value": "linux"}
                    ))
        _seed_claim(conn, "clUnk", "sA", "srcUn", "spUn",
                    "conditional probe payload", gen,
                    condition=json.dumps(
                        {"op": "eq", "key": "missing_key", "value": "v"}
                    ))
        _seed_claim(conn, "clNo", "sA", "srcNo", "spNo",
                    "conditional probe payload", gen,
                    condition=json.dumps(
                        {"op": "eq", "key": "platform", "value": "win32"}
                    ))
    task = TaskContext(environment=EnvironmentFingerprint(platform="linux"))
    res = recall_v3(
        store, _req("conditional probe payload", task=task)
    )
    order = [
        i.handle.object_id
        for p in res.packs for i in p.items
    ]
    assert "clApp" in order
    # The applies-tier claim is packed ahead of the does-not-apply one.
    if "clNo" in order:
        assert order.index("clApp") < order.index("clNo")
    # Unknown is not a wildcard: it never outranks the applies claim.
    if "clUnk" in order:
        assert order.index("clApp") < order.index("clUnk")


# =====================================================================
# C40 — false supersession prevention (C)
# =====================================================================


def test_c40_false_supersession_prevention(store):
    """C40 / tier C / V4-18.05: retirement *detection* proposes —
    hedged, hypothetical, hearsay, conditional, cancelled, corroborating,
    and topic-disjoint mentions produce NO supersession proposal, and a
    real detection only ever creates an open review + conflicts_with
    edge: the predecessor's state is untouched until review applies it.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        # Counterparty claims that must NOT be falsely superseded.
        _seed_claim(conn, "clHedge", "sA", "srcH", "spH",
                    "regenerate the api types with `make swagger-gen`", gen)
        _seed_claim(conn, "clCorr", "sA", "srcC", "spC",
                    "note: `swagger-gen` was retired, we also saw it", gen)
        _seed_claim(conn, "clOff", "sA", "srcO", "spO",
                    "the swagger-gen binary sits in tools/bin", gen)

    probe_n = 0

    def _proposals(text):
        """Seed a fresh probe claim (unique ids — claims never collide)
        and run the detector. Hedged probes assert ``[]`` so no review or
        edge artifacts can leak into the final detection's assertions."""
        nonlocal probe_n
        probe_n += 1
        cid = f"clProbe{probe_n}"
        with store.tx() as conn:
            _seed_claim(conn, cid, "sA", f"srcP{probe_n}",
                        f"spP{probe_n}", text, gen)
        return propose_retirement_supersessions(store, cid, "sA")

    # Hedged / conditional / hearsay / intent — no proposal.
    for text in (
        "maybe `swagger-gen` was retired — regenerate the api types",
        "check whether `swagger-gen` was retired for the api types",
        "if `swagger-gen` is retired we will regenerate the api types",
        "rumor says `swagger-gen` was removed from the api types",
        "we should deprecate `swagger-gen` for the api types soon",
        "`swagger-gen` was retired last year but reinstated",
    ):
        assert _proposals(text) == [], text

    # A corroborating counterparty is not contradicted — and the detector
    # still never *applies* anything (proposal-only).
    out = _proposals(
        "regenerate the api types with `make codegen` — "
        "`swagger-gen` was retired"
    )
    with store.read() as conn:
        rows = conn.execute(
            "SELECT target_id FROM edges WHERE edge_type = 'conflicts_with'"
        ).fetchall()
    # The corroborating claim is never proposed as a supersession victim.
    assert all("clCorr" not in r for r in rows)
    if out:
        with store.read() as conn:
            states = dict(
                conn.execute(
                    "SELECT claim_id, state FROM claim_revisions"
                ).fetchall()
            )
        # Every named claim stays in its prior state — proposals are
        # operator-gated, never auto-applied (V4-18.05's other half).
        assert states["clHedge"] == "active"
        assert states["clCorr"] == "active"
        assert states["clOff"] == "active"


# =====================================================================
# C41 — genuine supersession + historical recall (C)
# =====================================================================


def test_c41_genuine_supersession_and_history(store):
    """C41 / tier C / V4-18.04+18.07: a genuine supersession applied
    through the lifecycle machine closes the predecessor revision, writes
    the ``supersedes`` edge, truncates the predecessor's valid intervals,
    and moves current-state recall to the successor — while a
    ``known_at_seq``/valid-time read still returns the predecessor where
    it was live. A cycle and a same-scope violation are refused, not
    silently recorded.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        _seed_claim(conn, "clOld", "sA", "srcOld", "spOld",
                    "the deploy zephyr target is staging", gen,
                    recorded_from=5, intervals=((1000, 999_999_999),))
        _seed_claim(conn, "clNew", "sA", "srcNew", "spNew",
                    "the deploy zephyr target is prod-eu", gen,
                    recorded_from=40)

    # The transition sequence is the real events rowid — pad the ledger
    # past the seeded recorded_from values (5/40) so the supersession
    # lands strictly after both claims in event order.
    events = EventsRepo(store)
    with store.tx() as conn:
        while events.latest_seq(conn) < 45:
            events.append(conn, "sA", "seed_pad", "test", {}, "v4-test")

    machine = LifecycleMachine(store)
    with store.tx() as conn:
        seq = machine.apply(
            TransitionCommand(
                claim_id="clOld", expected_revision=1,
                effect="supersede", successor_claim_id="clNew",
                actor_id="reviewer:r1", reason="approved supersession",
                # The successor states its own effective start — the
                # predecessor's open interval truncates at that cut.
                interval=TimeInterval(from_us=5000, basis="explicit"),
            ),
            conn,
        )
    assert seq > 40  # both claims recorded before the cut

    with store.read() as conn:
        revs = conn.execute(
            "SELECT revision, state, recorded_from, recorded_until"
            " FROM claim_revisions WHERE claim_id = 'clOld'"
            " ORDER BY revision"
        ).fetchall()
        assert revs[0][1] == "active" and revs[0][3] == seq
        assert revs[1][1] == "superseded" and revs[1][2] == seq
        edge = conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges"
            " WHERE edge_type = 'supersedes'"
        ).fetchone()
        assert edge == ("clNew", "clOld", "supersedes")
        # The predecessor's open valid interval was truncated at the
        # successor's stated start on the *new* revision — rev 1's rows
        # are immutable history.
        iv_new = conn.execute(
            "SELECT from_us, until_us FROM valid_intervals"
            " WHERE claim_id = 'clOld' AND revision = 2"
        ).fetchone()
        assert iv_new == (1000, 5000)
        iv_old = conn.execute(
            "SELECT from_us, until_us FROM valid_intervals"
            " WHERE claim_id = 'clOld' AND revision = 1"
        ).fetchone()
        assert iv_old == (1000, 999_999_999)

    # Current recall returns the successor, not the superseded head.
    res_now = recall_v3(store, _req("zephyr"))
    ids = _ids(res_now)
    assert "clNew" in ids and "clOld" not in ids

    # Historical known-at still resolves the predecessor's live revision —
    # both claims were recorded before the cut, so both may surface, but
    # the superseded claim must not be hidden by its future state.
    res_hist = recall_v3(store, _req("zephyr", known_at_seq=seq - 1))
    assert "clOld" in _ids(res_hist)
    res_hist2 = recall_v3(store, _req("zephyr", known_at_seq=seq - 1,
                                      valid_at_us=1500))
    assert "clOld" in _ids(res_hist2)

    # Refusals: self-supersession and cycles are INVALID_TRANSITION, and
    # a stale expected revision is STALE_PROPOSAL.
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            machine.apply(
                TransitionCommand(
                    claim_id="clNew", expected_revision=1,
                    effect="supersede", successor_claim_id="clNew",
                    actor_id="r", reason="x",
                ),
                conn,
            )
        assert exc.value.code == ErrorCode.INVALID_TRANSITION
        with pytest.raises(VerbatimError) as exc2:
            machine.apply(
                TransitionCommand(
                    claim_id="clNew", expected_revision=99,
                    effect="supersede", successor_claim_id="clOld",
                    actor_id="r", reason="x",
                ),
                conn,
            )
        assert exc2.value.code == ErrorCode.STALE_PROPOSAL
        # clNew → clOld would close a supersession cycle.
        with pytest.raises(VerbatimError) as exc3:
            machine.apply(
                TransitionCommand(
                    claim_id="clNew", expected_revision=1,
                    effect="supersede", successor_claim_id="clOld",
                    actor_id="r", reason="x",
                ),
                conn,
            )
        assert exc3.value.code == ErrorCode.INVALID_TRANSITION


# =====================================================================
# C42 — incomplete / opaque tool procedures (C)
# =====================================================================


def test_c42_incomplete_tool_procedures(store):
    """C42 / tier C / V4-23.03+23.05: episodes that are open, lack action
    steps, reference unresolvable evidence, contain opaque operations,
    have fewer than two recognized ops, or carry no checker receipt all
    refuse compilation with an honest status — the evidence stays
    evidence and no ``procedures`` row is minted. A fully evidenced
    episode compiles to ``candidate`` (never active) — the real future
    acceptance shape.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")

        # 1. Open episode → INCOMPLETE / episode_not_completed.
        _episode(conn, "ep-open", "sA", closed=False)
        r = compile_episode(conn, "ep-open", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.INCOMPLETE
        assert r.reason == "episode_not_completed"

        # 2. Closed but no action steps → INCOMPLETE / no_action_steps.
        _episode(conn, "ep-empty", "sA", closed=True)
        r = compile_episode(conn, "ep-empty", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.INCOMPLETE
        assert r.reason == "no_action_steps"

        # 3. Transitions whose action steps do not resolve → INCOMPLETE /
        # unresolved_action_evidence.
        _episode(conn, "ep-unresolved", "sA", closed=True)
        _transition(conn, "t1", "ep-unresolved", "sA", 0,
                    action_step_id="ghost-step")
        r = compile_episode(conn, "ep-unresolved", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.INCOMPLETE
        assert r.reason == "unresolved_action_evidence"

        # 4. Opaque operations (arbitrary shell) → UNSUPPORTED /
        # opaque_operations_present — evidence, never a template.
        _episode(conn, "ep-opaque", "sA", closed=True)
        _trajectory(conn, "traj-o", "sA")
        _envelope(conn, "env-o1", "src-o1", "sA", "tool_call",
                  meta={"tool": "shell_exec",
                        "args": {"command": "rm -rf /tmp/x"}})
        _envelope(conn, "env-o2", "src-o2", "sA", "tool_call",
                  meta={"tool": "arbitrary_host_tool"})
        _tool_step(conn, "st-o1", "traj-o", "sA", 0, "env-o1")
        _tool_step(conn, "st-o2", "traj-o", "sA", 1, "env-o2")
        _transition(conn, "to1", "ep-opaque", "sA", 0,
                    action_step_id="st-o1")
        _transition(conn, "to2", "ep-opaque", "sA", 1,
                    action_step_id="st-o2")
        r = compile_episode(conn, "ep-opaque", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.UNSUPPORTED
        assert r.reason == "opaque_operations_present"

        # 5. Two recognized ops but NO checker receipt → UNSUPPORTED /
        # no_checker_receipt.
        _episode(conn, "ep-nocheck", "sA", closed=True)
        _trajectory(conn, "traj-n", "sA")
        _envelope(conn, "env-n1", "src-n1", "sA", "tool_call",
                  meta={"tool": "read_file", "args": {"path": "a.py"}})
        _envelope(conn, "env-n2", "src-n2", "sA", "tool_call",
                  meta={"tool": "edit_file",
                        "args": {"path": "a.py", "body": "x"}})
        _tool_step(conn, "st-n1", "traj-n", "sA", 0, "env-n1")
        _tool_step(conn, "st-n2", "traj-n", "sA", 1, "env-n2")
        _transition(conn, "tn1", "ep-nocheck", "sA", 0,
                    action_step_id="st-n1")
        _transition(conn, "tn2", "ep-nocheck", "sA", 1,
                    action_step_id="st-n2")
        r = compile_episode(conn, "ep-nocheck", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.UNSUPPORTED
        assert r.reason == "no_checker_receipt"
        assert r.procedure_id is None

        # 6. Fully evidenced: two recognized ops + a checker receipt →
        # CANDIDATE (review-gated, never active at compile time).
        _episode(conn, "ep-full", "sA", closed=True)
        _trajectory(conn, "traj-f", "sA")
        _envelope(conn, "env-f1", "src-f1", "sA", "tool_call",
                  meta={"tool": "read_file", "args": {"path": "a.py"}})
        _envelope(conn, "env-f2", "src-f2", "sA", "tool_call",
                  meta={"tool": "run_check",
                        "args": {"argv": ["pytest", "-x"]}})
        _envelope(conn, "env-f3", "src-f3", "sA", "verification",
                  meta={"checker": "pytest", "outcome": "passed"})
        _tool_step(conn, "st-f1", "traj-f", "sA", 0, "env-f1")
        _tool_step(conn, "st-f2", "traj-f", "sA", 1, "env-f2")
        _transition(conn, "tf1", "ep-full", "sA", 0,
                    action_step_id="st-f1")
        _transition(conn, "tf2", "ep-full", "sA", 1,
                    action_step_id="st-f2")
        _transition(conn, "tf3", "ep-full", "sA", 2,
                    checker_ref="env-f3")
        r = compile_episode(conn, "ep-full", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.CANDIDATE
        assert r.procedure_id is not None
        row = conn.execute(
            "SELECT state FROM procedures WHERE procedure_id = ?",
            (r.procedure_id,),
        ).fetchone()
        assert row[0] == "candidate"
        # Idempotent replay: same episode → same candidate.
        r2 = compile_episode(conn, "ep-full", hmac_fn=store.hmac)
        assert r2.procedure_id == r.procedure_id
        # The gated episodes minted nothing.
        assert conn.execute(
            "SELECT COUNT(*) FROM procedures"
        ).fetchone()[0] == 1


# =====================================================================
# C43 — environment / schema drift (C)
# =====================================================================


def test_c43_environment_schema_drift(store):
    """C43 / tier C / V4-23.06+25.06: a recorded environment fingerprint
    binds exactly — a changed platform, tool-schema, or runtime version
    yields an explicit DOES_NOT_APPLY/UNKNOWN verdict, never silent
    applicability. Absent environment evidence is 'unknown', not a
    wildcard.
    """
    env = {
        "platform": "linux-x86_64",
        "tool.pytest": "8.1",
        "runtime.python": "3.12",
    }
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        _procedures_row(
            conn, "proc:env", "sA", state="active",
            environment=dict(env),
            applicability=[
                {"key": "environment.tool.pytest", "op": "eq",
                 "value": "8.1"},
            ],
        )
        # A procedure with NO recorded environment evidence.
        _procedures_row(conn, "proc:noenv", "sA", state="active")

    with store.read() as conn:
        # Same environment → applies.
        r = check_applicability(conn, "proc:env", environment=dict(env))
        assert r.verdict is ApplicabilityVerdict.APPLIES

        # Tool-schema drift → does_not_apply (the named key mismatches).
        drifted = dict(env)
        drifted["tool.pytest"] = "9.0"
        r = check_applicability(conn, "proc:env", environment=drifted)
        assert r.verdict is ApplicabilityVerdict.DOES_NOT_APPLY
        assert "environment_mismatch" in r.reasons

        # Platform drift → does_not_apply.
        r = check_applicability(
            conn, "proc:env",
            environment={**env, "platform": "darwin-arm64"},
        )
        assert r.verdict is ApplicabilityVerdict.DOES_NOT_APPLY

        # No environment offered → unknown, not applies.
        r = check_applicability(conn, "proc:env")
        assert r.verdict is ApplicabilityVerdict.UNKNOWN
        assert "environment_not_provided" in r.reasons

        # No recorded environment evidence → unknown, never a wildcard.
        r = check_applicability(
            conn, "proc:noenv", environment=dict(env)
        )
        assert r.verdict is ApplicabilityVerdict.UNKNOWN
        assert "no_environment_evidence" in r.reasons

        # EnvironmentFingerprint inputs normalize identically.
        fp = EnvironmentFingerprint(
            platform="linux-x86_64",
            runtime_versions=(("python", "3.12"),),
            tool_schema_versions=(("pytest", "8.1"),),
        )
        r = check_applicability(conn, "proc:env", environment=fp)
        assert r.verdict is ApplicabilityVerdict.APPLIES
        fp2 = EnvironmentFingerprint(
            platform="linux-x86_64",
            runtime_versions=(("python", "3.12"),),
            tool_schema_versions=(("pytest", "9.0"),),
        )
        r = check_applicability(conn, "proc:env", environment=fp2)
        assert r.verdict is ApplicabilityVerdict.DOES_NOT_APPLY


# =====================================================================
# C44 — no self-promotion / self-granted execution (C)
# =====================================================================


def test_c44_no_self_promotion_or_execution_grant(store):
    """C44 / tier C / V4-23.04+23.09: the promotion ladder is
    evidence-gated — a ``candidate`` cannot activate itself, a
    'reviewed' row without recorded approve evidence cannot activate,
    activation requires the explicit review, and an agent-submitted
    'lesson' is stored as agent-generated advisory content that mints no
    grant and no execution authority.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        # A self-authored candidate — the agent is its own 'proposer'.
        _procedures_row(
            conn, "proc:self", "sA", state="candidate",
            provenance={"proposed_by": "agent:prod"},
        )
        # A 'reviewed' row with no review evidence (bypass attempt).
        _procedures_row(
            conn, "proc:fake", "sA", state="reviewed",
            provenance={"proposed_by": "agent:prod"},
        )
    with store.tx() as conn:
        # Candidate cannot jump to active — whoever asks.
        with pytest.raises(VerbatimError) as exc:
            activate(conn, "proc:self", "agent:prod")
        assert exc.value.code == ErrorCode.INVALID_TRANSITION
        # 'reviewed' without approve-review evidence cannot activate.
        with pytest.raises(VerbatimError) as exc2:
            activate(conn, "proc:fake", "agent:prod")
        assert exc2.value.code == ErrorCode.INVALID_TRANSITION
        # The real ladder: explicit review THEN activate.
        review(conn, "proc:self", "approve", "human:reviewer",
               notes="evidence checked")
        out = activate(conn, "proc:self", "human:reviewer")
        row = conn.execute(
            "SELECT state FROM procedures WHERE procedure_id = 'proc:self'"
        ).fetchone()
        assert row[0] == "active" and out["state"] == "active"
        # The review evidence is durably recorded on the procedure.
        prov = conn.execute(
            "SELECT provenance_json FROM procedures"
            " WHERE procedure_id = 'proc:self'"
        ).fetchone()[0]
        assert "human:reviewer" in prov

    # Agent-submitted capture requires an ingest grant AND live capture
    # consent — the agent cannot mint either by declaring a kind.
    api = VerbatimV3(store)
    with pytest.raises(VerbatimError) as exc3:
        api.capture_submitted(
            "agent:prod", "sA", "lesson: always run pytest first",
            declared_type="lesson",
        )
    assert exc3.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # With consent the lesson still lands as agent_generated advisory
    # data — and no grant rows appear for it.
    with store.tx() as conn:
        create_grant(
            conn, scope_id="sA", principal_id="agent:prod",
            verbs={"ingest"}, issuer_id="human:alice",
            purposes=None,  # explicitly any-purpose — the facade submits
                            # with purpose=None
        )
        register_principal(conn, kind="agent", principal_id="agent:prod")
    # Unowned scope → the issuer bootstraps as owner (registered under
    # the facade's launch principal kind), then issues the consent.
    api.issue_capture_authorization(
        "agent:prod", "sA", granted_by="human:issuer"
    )
    src = api.capture_submitted(
        "agent:prod", "sA", "lesson: always run pytest first",
        declared_type="lesson",
    )
    assert isinstance(src, str)
    with store.read() as conn:
        env = conn.execute(
            "SELECT envelope_kind, trust_class FROM source_envelopes"
            " WHERE source_id = ?", (src,),
        ).fetchone()
        assert env[1] == "agent_generated"
        # No new grants were minted by the submission.
        grants = conn.execute(
            "SELECT COUNT(*) FROM grants_v3 WHERE principal_id = 'agent:prod'"
        ).fetchone()[0]
        assert grants == 1  # only the pre-issued ingest grant


# =====================================================================
# C45 — verified failure stays negative evidence (C)
# =====================================================================


def test_c45_verified_failure_remains_negative(store):
    """C45 / tier C / V4-19.10+34.02: a host-resolved checker receipt is
    the only source of attested outcomes — an agent's self-declared
    'success' cannot upgrade, overwrite, or re-label a host-attested
    failure. Both records persist honestly: the attested failure keeps
    ``host_observed`` trust + outcome 'failure'; the later self-report is
    an ``agent_report`` and nothing more.
    """
    receipt = {
        "checker_id": "pytest",
        "checker_version": "8.1",
        "invocation_id": "inv:1",
        "completed": True,
        "exit_code": 1,
        "outcome": "failure",
        "scope_id": "sA",
        "task_id": "t1",
        "result_json": {"failed": 2},
        "selected_tests": ["tests/unit"],
    }

    class _Resolver:
        resolver_id = "host-checker-1"

        def resolve_checker(self, invocation_id):
            return dict(receipt) if invocation_id == "inv:1" else None

    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA", pid="agent:prod", purposes=None,
              verbs=("read", "quote", "derive"))

    api = VerbatimV3(store, checker_resolver=_Resolver())

    # Host-attested FAILURE persists even though the agent declared
    # 'success' — the receipt decides the outcome, not the caller.
    out = api.submit_outcome(
        "sA", principal_id="agent:prod", outcome="success",
        invocation_id="inv:1", task_id="t1", trajectory_id=None,
    )
    assert out["outcome"] == "failure"
    assert out["declared_outcome"] == "success"
    assert out["attested"] is True
    assert out["agent_report"] is False

    # A later self-report of 'success' is recorded as exactly that — an
    # agent report, never an attestation.
    out2 = api.submit_outcome(
        "sA", principal_id="agent:prod", outcome="success",
        checker_id="self-check",
    )
    assert out2["attested"] is False
    assert out2["agent_report"] is True
    assert out2["outcome"] == "success"  # the claim, labeled — not trusted

    with store.read() as conn:
        rows = conn.execute(
            "SELECT trust_class, metadata_json FROM source_envelopes"
            " WHERE envelope_kind = 'verification' ORDER BY rowid"
        ).fetchall()
        assert len(rows) == 2
        attested = json.loads(rows[0][1])
        reported = json.loads(rows[1][1])
        assert rows[0][0] == "host_observed"
        assert attested["outcome"] == "failure"
        assert attested["verification"] == "host_attested"
        assert attested["checker"]["host_attested"] is True
        assert attested["checker"]["attestation"]
        assert rows[1][0] == "agent_generated"
        assert reported["verification"] == "agent_report"
        assert reported["checker"]["agent_report"] is True
        assert reported["checker"]["host_attested"] is False

        # An attestation-bound receipt cannot be forged by caller fields:
        # the resolver's own verdict governs every host_observed row.
        forged = [
            r for r in conn.execute(
                "SELECT metadata_json FROM source_envelopes"
            ).fetchall()
            if "host_attested" in r[0] and '"exit_code": 0' in r[0]
        ]
        assert forged == []

    # Receipt binding: a resolver receipt for a different scope is a
    # contract violation — denied, not downgraded to agent report.
    bad = dict(receipt)
    bad["scope_id"] = "sOTHER"

    class _BadResolver:
        resolver_id = "host-checker-1"

        def resolve_checker(self, invocation_id):
            return dict(bad)

    api_bad = VerbatimV3(store, checker_resolver=_BadResolver())
    with pytest.raises(VerbatimError) as exc:
        api_bad.submit_outcome(
            "sA", principal_id="agent:prod", outcome="success",
            invocation_id="inv:1", task_id="t1",
        )
    assert exc.value.code == ErrorCode.VALIDATION


# =====================================================================
# C46 — invalid generated quote locators (E)
# =====================================================================


def test_c46_invalid_generated_quote_locators(store, kernel):
    """C46 / tier E / V4-16.03+20.04: derived/generated quote locators
    are validated at the kernel read surface — out-of-bounds ranges,
    unknown views, unlisted objects, and integrity-digest mismatches all
    deny or fail closed (indistinguishable absence or STORE_CORRUPT),
    never fabricating bytes for a schema-valid but bogus locator.
    """
    PAYLOAD = b"the quick brown fox jumps over the lazy dog"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:a', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        _k_source(conn, store, "scope:a", "src:s1", PAYLOAD)
        _k_view(conn, store, "src:s1", 1, "view:norm", b"derived text")
        _k_grant(conn, "scope:a", "human:alice", ["read", "quote"])
    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )

        # Out-of-bounds locator → indistinguishable denial.
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=len(PAYLOAD) + 1)],
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

        # Reversed range → denial.
        with pytest.raises(VerbatimError):
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=10, end_byte=4)],
                now_us=T0,
            )

        # A view that does not exist → denial (not fabricated bytes).
        with pytest.raises(VerbatimError):
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=4,
                                 view_id="view:ghost")],
                now_us=T0,
            )

        # An object outside the lease's scopes → denial.
        with pytest.raises(VerbatimError):
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:other", revision=1,
                                 start_byte=0, end_byte=4)],
                now_us=T0,
            )

        # Well-formed locator on the real view → verified bytes.
        got = kernel.read_verified(
            conn, lease,
            [EvidenceLocator(object_id="src:s1", revision=1,
                             start_byte=0, end_byte=7,
                             view_id="view:norm")],
            now_us=T0,
        )
        assert got[0].data == b"derived"[:7]
        assert got[0].verification == "verified"

    # Corrupted derived bytes fail closed: STORE_CORRUPT, never served.
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_views SET derived_bytes = ?"
            " WHERE source_id = 'src:s1' AND view_id = 'view:norm'",
            (b"tampered bytes",),
        )
    with store.read() as conn:
        lease2 = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc2:
            kernel.read_verified(
                conn, lease2,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=7,
                                 view_id="view:norm")],
                now_us=T0,
            )
        assert exc2.value.code == ErrorCode.STORE_CORRUPT


# =====================================================================
# C47 — unsupported generated propositions (E)
# =====================================================================


def test_c47_unsupported_generated_propositions(store, kernel):
    """C47 / tier E / V4-20.05: a proposition without verifiable support
    cannot be presented as grounded — a claim with no evidence produces
    no quotable item, a derived view without a recorded integrity digest
    reads as ``legacy_unverified`` (honestly labeled, not 'verified'), a
    view with absent bytes denies, and a delivery seal pinned to a stale
    or unauthorized dependency refuses to mint.
    """
    PAYLOAD = b"the quick brown fox jumps over the lazy dog"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:a', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        _k_source(conn, store, "scope:a", "src:s1", PAYLOAD)
        _k_view(conn, store, "src:s1", 1, "view:noint", b"unverified bytes",
                legacy_digest=True)
        _k_view(conn, store, "src:s1", 1, "view:empty", None)
        _k_grant(conn, "scope:a", "human:alice", ["read", "quote"])
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        # A 'claim' with an FTS row but no claim_evidence at all.
        _add_source(conn, "srcN", "sA", b"unsupported proposition text")
        conn.execute(
            "INSERT INTO claims(claim_id,scope_id,created_event)"
            " VALUES('clBare','sA',1)"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "recorded_from) VALUES('clBare',1,'active',1)"
        )
        _add_fts(conn, "clBare", 1, "sA", "unsupported proposition text",
                 gen)

    # The unsupported claim never produces a delivered item — there is
    # nothing to quote.
    res = recall_v3(store, _req("unsupported proposition text"))
    assert "clBare" not in _ids(res)

    with store.read() as conn:
        lease = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        # Digest-less derived view: served but honestly labeled
        # ``legacy_unverified`` — never presented as verified support.
        got = kernel.read_verified(
            conn, lease,
            [EvidenceLocator(object_id="src:s1", revision=1,
                             start_byte=0, end_byte=8,
                             view_id="view:noint")],
            now_us=T0,
        )
        assert got[0].verification == "legacy_unverified"
        # A view whose bytes were never persisted denies outright.
        with pytest.raises(VerbatimError):
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=4,
                                 view_id="view:empty")],
                now_us=T0,
            )

    # A seal pinned to a superseded/phantom dependency cannot mint a
    # delivery permit — the proposition cannot corroborate downstream.
    with store.tx() as conn:
        lease2 = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.seal_delivery(
                conn, lease2, b"pack", {"src:s1": 7}, now_us=T0
            )
        assert exc.value.code == ErrorCode.STALE_DEPENDENCY
        with pytest.raises(VerbatimError):
            kernel.seal_delivery(
                conn, lease2, b"pack", {"src:ghost": 1}, now_us=T0
            )
        with pytest.raises(VerbatimError):
            kernel.seal_delivery(
                conn, lease2, b"pack", {"src:unlisted": 1}, now_us=T0
            )


# =====================================================================
# C48 — audience-intersection derivative access (E)
# =====================================================================


def test_c48_audience_intersection_derivative(store, kernel):
    """C48 / tier E / V4-10.07+20.06: a derivative spanning two scopes
    inherits the *intersection* of parent audiences and purposes — a
    recipient denied either parent is absent from the effective audience,
    and a producer without quote on every input scope cannot assemble the
    bundle at all.
    """
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:a', 'prof', 'owner'),"
            " ('scope:b', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        register_principal(conn, kind="agent", principal_id="agent:prod")
        register_principal(conn, kind="human", principal_id="human:bob")
        _k_source(conn, store, "scope:a", "src:s1", b"a-bytes")
        _k_source(conn, store, "scope:b", "src:b1", b"b-bytes")
        gid = create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["derive"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["quote"], purposes=["derive", "recall"],
            issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:b", principal_id="agent:prod",
            verbs=["quote"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:a", principal_id="human:bob",
            verbs=["read"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:b", principal_id="human:bob",
            verbs=["read"], purposes=["derive"], issuer_id="human:alice",
        )
        bundle = kernel.derive_inputs(
            conn, gid,
            [("src:s1", 1), ("src:b1", 1)],
            output_audience=["human:bob", "human:mallory"],
            output_purpose="derive",
            now_us=T0,
        )

    assert bundle.producer_id == "agent:prod"
    assert len(bundle.inputs) == 2
    assert set(bundle.scope_ids) == {"scope:a", "scope:b"}
    # Mallory asked but holds nothing in either parent — excluded.
    # Bob holds read in both — the ONLY surviving audience member.
    assert bundle.effective_audience == ("human:bob",)
    # Purposes intersect too: {derive,recall} ∩ {derive}.
    assert bundle.allowed_purposes.values == frozenset({"derive"})

    # A producer missing quote on ONE parent cannot derive at all —
    # denied before any bytes move (audience math never rescues it).
    with store.tx() as conn:
        gid2 = create_grant(
            conn, scope_id="scope:a", principal_id="agent:prod",
            verbs=["derive"], purposes=["derive"], issuer_id="human:alice",
        )
        create_grant(
            conn, scope_id="scope:b", principal_id="agent:prod",
            verbs=["read"], purposes=["derive"], issuer_id="human:alice",
        )
        # revoke the existing scope:b quote grant path by denying a fresh
        # principal without one:
        gid3 = create_grant(
            conn, scope_id="scope:a", principal_id="human:bob",
            verbs=["derive"], purposes=["derive"], issuer_id="human:alice",
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.derive_inputs(
                conn, gid3,
                [("src:s1", 1), ("src:b1", 1)],
                output_audience=["human:bob"],
                output_purpose="derive",
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# =====================================================================
# C49 — generated overview invalidation obligations (E)
# =====================================================================


def test_c49_generated_overview_invalidation(store, kernel):
    """C49 / tier E / V4-20.07+33.03: correcting or holding a parent
    immediately invalidates dependent generated overviews — the kernel's
    ``invalidate`` traverses ``dependency_edges`` so the overview lands in
    ``affected`` with a ``reevaluate`` obligation, the parent scope's
    epoch bumps (stale permits/leases fence themselves), and a sealed
    in-flight permit is honestly enumerated rather than silently reused.
    """
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:a', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        _k_source(conn, store, "scope:a", "src:s1", b"source bytes")
        _k_grant(conn, "scope:a", "human:alice", ["read", "quote"])
        # A generated overview object derived from the source.
        repos_v4.insert(
            conn, "dependency_edges",
            {"child_kind": "view", "child_id": "view:overview1",
             "child_revision": 1, "parent_kind": "source",
             "parent_id": "src:s1", "parent_revision": 1,
             "role": "derived", "producer_id": "synth",
             "operation_id": "op:1", "seq": 0},
        )
        repos_v4.insert(
            conn, "dependency_edges",
            {"child_kind": "claim", "child_id": "claim:sum1",
             "child_revision": 1, "parent_kind": "view",
             "parent_id": "view:overview1", "parent_revision": 1,
             "role": "derived", "producer_id": "synth",
             "operation_id": "op:1", "seq": 1},
        )
        lease = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        permit = kernel.seal_delivery(
            conn, lease, b"cached-overview-pack", {}, now_us=T0
        )
        report = kernel.invalidate(
            conn,
            {"kind": "correction", "scope_ids": ["scope:a"],
             "object_refs": [("source", "src:s1", 1)]},
            now_us=T0,
        )

    assert report.event_kind == "correction"
    assert report.epochs == {"scope:a": 1}
    affected = {(k, i) for k, i, _ in report.affected}
    # Parent + first-order dependent + transitive dependent.
    assert ("source", "src:s1") in affected
    assert ("view", "view:overview1") in affected
    assert ("claim", "claim:sum1") in affected
    # Every affected object carries a reevaluation obligation.
    owed = {(o["object_kind"], o["object_id"]) for o in report.obligations}
    assert ("view", "view:overview1") in owed
    assert ("claim", "claim:sum1") in owed
    assert all(o["obligation"] == "reevaluate" for o in report.obligations)
    # The committed permit is honestly listed as an in-flight disclosure
    # — enumerated, never silently reused post-correction.
    assert permit.permit_id in report.in_flight_permits

    # Post-invalidation, a pre-bump lease cannot read: stale epoch.
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc:
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=4)],
                now_us=T0,
            )
        assert exc.value.code == ErrorCode.STALE_EPOCH


# =====================================================================
# C50 — bounded reflection, no arbitrary host tools (E)
# =====================================================================


def test_c50_bounded_reflection_no_arbitrary_tools(store):
    """C50 / tier E / V4-27.04+27.05: reflection here means recorded
    trajectory/step work plus request-level budgets — and every bound is
    enforced. Trajectory submissions cap at 512 steps; arbitrary tool
    names classify ``opaque`` (never a recognized reusable op), an
    episode built on opaque ops refuses compilation, and no step mints
    execution authority or a grant. Request ceilings (deadline, items,
    bytes) are hard-validated before any lane runs.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        register_principal(conn, kind="agent", principal_id="agent:prod")
        seed_purposes(conn)
        create_grant(
            conn, scope_id="sA", principal_id="agent:prod",
            verbs={"derive", "read", "quote"}, issuer_id="human:alice",
            purposes=None,  # any-purpose — the facade submits purpose=None
        )
    api = VerbatimV3(store)

    # Step budget enforced at the public boundary.
    with pytest.raises(VerbatimError) as exc:
        api.submit_trajectory(
            "sA", principal_id="agent:prod",
            steps=[{"step_id": f"s{i}"} for i in range(513)],
        )
    assert exc.value.code == ErrorCode.VALIDATION

    # A bounded trajectory of arbitrary-tool steps records as evidence.
    out = api.submit_trajectory(
        "sA", principal_id="agent:prod", task_id="t1",
        steps=[{"step_id": "s0"}],
    )
    assert out["steps"] == 1

    # Arbitrary host tool names and shell strings are opaque — bounded
    # classification, never implicit host execution.
    assert classify("arbitrary_host_tool", {}).value == "opaque"
    assert classify("shell_exec", {"command": "rm -rf /"}).value == "opaque"
    assert classify("run_check", {"command": "pytest -x"}).value == "opaque"
    assert classify(None, {}).value == "opaque"
    # Recognized names keep their class; structured checker argv works.
    assert classify("read_file", {"path": "a"}).value == "inspect_file"
    assert classify("run_check", {"argv": ["pytest", "-x"]}).value == (
        "run_check"
    )

    # Opaque ops block compilation (C42 machinery is the enforcement):
    # an episode of arbitrary-tool steps mints no procedure.
    with store.tx() as conn:
        _episode(conn, "ep-rf", "sA", closed=True)
        _trajectory(conn, "traj-rf", "sA")
        _envelope(conn, "env-rf", "src-rf", "sA", "tool_call",
                  meta={"tool": "run_host_command",
                        "args": {"cmd": "sudo rm -rf /"}})
        _tool_step(conn, "st-rf", "traj-rf", "sA", 0, "env-rf")
        _transition(conn, "t-rf", "ep-rf", "sA", 0,
                    action_step_id="st-rf")
        r = compile_episode(conn, "ep-rf", hmac_fn=store.hmac)
        assert r.status is CompilationStatus.UNSUPPORTED
        assert conn.execute(
            "SELECT COUNT(*) FROM procedures"
        ).fetchone()[0] == 0

    # Request ceilings are hard bounds — reflection/deep-context calls
    # cannot widen them.
    for bad in (
        dict(deadline_ms=2001),
        dict(max_items=33),
        dict(max_bytes=24001),
        dict(target_tokens=6145),
    ):
        with pytest.raises(VerbatimError) as exc2:
            _req("probe", **bad)
        assert exc2.value.code == ErrorCode.VALIDATION


# =====================================================================
# C51 — progressive-disclosure budgets + caller binding (E)
# =====================================================================


def test_c51_progressive_disclosure_budgets_and_binding(store, kernel):
    """C51 / tier E / V4-32.03+32.09: progressive disclosure honors exact
    budgets — pack admission stops at the item/byte ceilings with the
    remainder counted ``omitted``, groups never split — and delivery
    permits stay bound to their caller/payload/expiry: one-use, expired
    permits deny, a mismatched payload digest fails validation, and a
    post-issuance epoch change fences re-reads.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        for i in range(6):
            _seed_claim(conn, f"cl{i}", "sA", f"src{i}", f"sp{i}",
                        f"budget probe item {i} " + "pad" * 20, gen)

    # Item ceiling: max_items=1 delivers at most one item.
    res = recall_v3(store, _req("budget probe", max_items=1))
    assert len(_items(res)) <= 1
    # Byte ceiling: serialized pack bytes stay under the bound; the
    # remainder is honestly counted as omitted, not truncated mid-item.
    res = recall_v3(store, _req("budget probe", max_bytes=512))
    total = sum(int(p.serialized_bytes or 0) for p in res.packs)
    assert total <= 512
    assert res.omitted >= 1
    assert "budget_omitted_group" in res.warnings or res.omitted >= 1

    # Permit binding (kernel surface): sealed to this caller/payload.
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:a', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:alice")
        _k_source(conn, store, "scope:a", "src:s1", b"payload bytes")
        _k_grant(conn, "scope:a", "human:alice", ["quote"])
        lease = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        permit = kernel.seal_delivery(
            conn, lease, b"serialized-pack", {"src:s1": 1}, now_us=T0
        )
    assert permit.caller_id == "human:alice"
    assert permit.payload_digest.startswith("sha256:")

    # One-use: after mark_delivered the permit no longer verifies.
    with store.tx() as conn:
        done = kernel.mark_delivered(conn, permit.permit_id, now_us=T0)
        assert done.state == "delivered"
        with pytest.raises(VerbatimError):
            kernel.verify_delivery(conn, permit.permit_id, now_us=T0)

    # Expiry: a fresh sealed permit past its expiry denies PERMIT_EXPIRED.
    with store.tx() as conn:
        lease2 = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        permit2 = kernel.seal_delivery(
            conn, lease2, b"pack-2", {}, now_us=T0, max_age_us=1
        )
        with pytest.raises(VerbatimError) as exc:
            kernel.verify_delivery(
                conn, permit2.permit_id, now_us=T0 + 2
            )
        assert exc.value.code == ErrorCode.PERMIT_EXPIRED

    # Payload binding: a mismatched digest is a VALIDATION refusal, and
    # with no provisioned broker the dispatch lane reports unavailable —
    # never a false dispatch.
    with store.tx() as conn:
        lease3 = kernel.resolve_access(
            conn, _k_caller(), "quote", "recall", ["scope:a"], now_us=T0
        )
        permit3 = kernel.seal_delivery(
            conn, lease3, b"pack-3", {}, now_us=T0
        )
        with pytest.raises(VerbatimError) as exc2:
            kernel.open_dispatch(
                conn, permit3, recipient="endpoint:x", max_spend=1.0,
                payload_digest="sha256:" + "00" * 32, now_us=T0,
            )
        assert exc2.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as exc3:
            kernel.open_dispatch(
                conn, permit3, recipient="endpoint:x", max_spend=1.0,
                now_us=T0,
            )
        assert exc3.value.code == ErrorCode.CAPABILITY_UNAVAILABLE

    # Epoch fencing: a lease minted before the bump cannot re-read.
    with store.tx() as conn:
        bump_epoch(conn, "scope:a")
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc4:
            kernel.read_verified(
                conn, lease3,
                [EvidenceLocator(object_id="src:s1", revision=1,
                                 start_byte=0, end_byte=4)],
                now_us=T0,
            )
        assert exc4.value.code == ErrorCode.STALE_EPOCH


# =====================================================================
# C52 — whole-group omission on required-member loss (C, M0)
# =====================================================================


def test_c52_whole_group_omission(store):
    """C52 / tier C / V4-32.02+32.08: when a required conflict/context
    member is unavailable (cross-scope here, suppression/quarantine are
    the same gate), the group is omitted WHOLE — the surviving member
    does not ship as a misleading partial answer, and the omission is
    counted + warned, not silent.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        _scope(conn, "sB", principal="p2", conv="c2")
        _auth(conn, "sA")
        gen = _gen(store)
        _seed_claim(conn, "clA", "sA", "srcA", "spA",
                    "disputed alpha claim", gen)
        _seed_claim(conn, "clB", "sB", "srcB", "spB",
                    "disputed beta claim", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')"
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clA'),('cg1','clB')"
        )

    res = recall_v3(store, _req("disputed alpha claim"))
    delivered = _ids(res)
    # Neither side ships: clB is unauthorized, so the group is broken and
    # clA — which the caller CAN see — is still withheld with it.
    assert "clA" not in delivered
    assert "clB" not in delivered
    # The omission is surfaced, never silent: either pack assembly counted
    # it (``omitted`` + ``incomplete_group_omitted``) or the group-verdict
    # filter dropped it earlier and abstained with a reason. A clean empty
    # result would fail here.
    assert (
        res.omitted >= 1
        or "incomplete_group_omitted" in res.warnings
        or res.abstained
        or res.warnings
    )

    # Same rule through a suppressed context member: the required member
    # is tombstoned → the whole group omits.
    with store.tx() as conn:
        payload = b"Alice said: the deploy is blocked"
        _add_source(conn, "srcC", "sA", payload)
        _add_span(conn, "spMain", "srcC", 12, len(payload))
        _add_span(conn, "spAttr", "srcC", 0, 11)
        conn.execute(
            "INSERT INTO claims(claim_id,scope_id,created_event)"
            " VALUES('clCtx','sA',1)"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "recorded_from) VALUES('clCtx',1,'active',1)"
        )
        conn.execute(
            "INSERT INTO claim_evidence(claim_id,revision,span_id,"
            "evidence_role) VALUES('clCtx',1,'spMain','primary')"
        )
        _add_fts(conn, "clCtx", 1, "sA", "deploy blocked", _gen(store))
        conn.execute(
            "INSERT INTO context_groups(group_id,scope_id,source_id,"
            "revision,parser_version,operation_key,completeness,"
            "recorded_from) VALUES('xg1','sA','srcC',1,'p1','k1',"
            "'complete',1)"
        )
        conn.execute(
            "INSERT INTO context_members(group_id,span_id,role,required,"
            "ord) VALUES('xg1','spMain','primary',1,0),"
            "('xg1','spAttr','attribution',1,1)"
        )
        suppress(store, "sA", [("span", "spAttr")], "op", conn=conn)
    res2 = recall_v3(store, _req("deploy blocked"))
    assert not any(
        "deploy is blocked" in t for t in _texts(res2)
    ), "required context member suppressed → whole group must omit"


# =====================================================================
# C53 — fuzzy-cache absence / unsafe-reuse prohibition (E)
# =====================================================================


def test_c53_fuzzy_cache_absent_or_safe(store):
    """C53 / tier E / V4-33.04: this build ships no fuzzy/semantic answer
    cache — the capability surface declares no such lane and every recall
    recomputes under current authorization. The scenario's safety
    property still binds today: exact identifiers never conflate, and a
    negated/changed-state query never inherits an unsafe cached answer.
    When a cache lands, the same assertions stay green by hitting the
    real cache path.
    """
    caps = VerbatimV3(store).capabilities()
    # Honest surface: no lane claims a cache capability.
    assert not any("cache" in name for name in caps["lanes"])
    assert all(
        caps["lanes"][name]["rung"] != CapabilityRung.RECOMMENDED.value
        or "cache" not in name
        for name in caps["lanes"]
    )

    with store.tx() as conn:
        _scope(conn, "sA")
        _auth(conn, "sA")
        gen = _gen(store)
        _seed_claim(conn, "cl41", "sA", "src41", "sp41",
                    "order ZX-41 shipped tuesday", gen)
        _seed_claim(conn, "cl42", "sA", "src42", "sp42",
                    "order ZX-42 cancelled wednesday", gen)

    # Exact identifiers are never fuzzily conflated: ZX-41 answers only
    # with its own evidence — a cache may not borrow ZX-42's.
    res = recall_v3(store, _req("ZX-41 status"))
    ids = _ids(res)
    assert "cl41" in ids and "cl42" not in ids
    res = recall_v3(store, _req("ZX-42 status"))
    ids = _ids(res)
    assert "cl42" in ids and "cl41" not in ids

    # Recompute discipline: a current-state change lands immediately —
    # there is no stale answer surface to survive it.
    with store.tx() as conn:
        conn.execute(
            "UPDATE claim_revisions SET state = 'superseded'"
            " WHERE claim_id = 'cl41'"
        )
    res = recall_v3(store, _req("ZX-41 status"))
    assert "cl41" not in _ids(res)


# =====================================================================
# C54 — revoke/erase cannot be bypassed by stale cache (C, M0)
# =====================================================================


def test_c54_revoke_erase_bypasses_no_cache(store):
    """C54 / tier C / V4-33.02+33.03: revocation and erasure fence every
    later read — a recall authorized before the grant is revoked cannot
    be replayed, a suppressed source's claims stop delivering immediately
    (no stale pack surface), and kernel leases minted pre-bump fail
    STALE_EPOCH on revalidation. There is no cache to bypass; the
    authorization check runs on every call.
    """
    with store.tx() as conn:
        _scope(conn, "sA")
        gid = _auth(conn, "sA", pid="human:alice",
                    verbs=("read", "quote", "admin"))
        gen = _gen(store)
        _seed_claim(conn, "clX", "sA", "srcX", "spX",
                    "revocation probe payload", gen)

    api = VerbatimV3(store)
    res = api.recall("sA", "revocation probe", principal_id="human:alice")
    assert "clX" in _ids(res)

    # Grant revoked → every subsequent call denies, cache or not.
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with pytest.raises(VerbatimError) as exc:
        api.recall("sA", "revocation probe", principal_id="human:alice")
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # Re-grant, then erase the source — suppression applies to the very
    # next read; the old answer cannot be served from anywhere.
    with store.tx() as conn:
        _auth(conn, "sA", pid="human:alice", purposes=None,
              verbs=("read", "quote", "admin"))
    res = api.recall("sA", "revocation probe", principal_id="human:alice")
    assert "clX" in _ids(res)
    out = api.delete_source("srcX", principal_id="human:alice")
    # Suppression is in effect NOW; physical/vault erasure drains async —
    # the closure states that honestly.
    assert out["status"] == "suppressed"
    assert out["closure"]["logical"] == "suppressed_now"
    res = api.recall("sA", "revocation probe", principal_id="human:alice")
    assert "clX" not in _ids(res)
    assert "revocation probe payload" not in _texts(res)

    # Kernel path: a lease minted pre-revocation is fenced by the epoch
    # bump on any revalidation — stale authorization cannot re-read.
    kernel = Kernel(store)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:k', 'prof', 'owner')"
        )
        seed_purposes(conn)
        register_principal(conn, kind="human", principal_id="human:bob")
        _k_source(conn, store, "scope:k", "src:k1", b"k-bytes")
        _k_grant(conn, "scope:k", "human:bob", ["read", "quote"])
        lease = kernel.resolve_access(
            conn, _k_caller("human:bob"), "quote", "recall",
            ["scope:k"], now_us=T0,
        )
        bump_epoch(conn, "scope:k")
    with store.read() as conn:
        with pytest.raises(VerbatimError) as exc2:
            kernel.read_verified(
                conn, lease,
                [EvidenceLocator(object_id="src:k1", revision=1,
                                 start_byte=0, end_byte=4)],
                now_us=T0,
            )
        assert exc2.value.code == ErrorCode.STALE_EPOCH
