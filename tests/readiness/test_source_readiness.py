"""V5 source-branch readiness tests (SPEC_V5 §08, docs/v5_contracts §4).

Covers the source-projection DAG branch added to ``readiness.py``:

- ``CapabilityName`` gains ``source_lexical_ready``/``source_vector_ready``
  (registered dynamically by the readiness module — ``types_v4.py`` is a
  frozen file owned outside this package; the values are local literals
  identical to ``memory.types.CAP_SOURCE_LEXICAL``/``CAP_SOURCE_VECTOR``).
- DAG shape: ``accepted → screened → {source_lexical_ready,
  source_vector_ready}`` as siblings on ``screened``, parallel to the
  unchanged claim branch ``lexical_ready → {semantic_ready,
  derived_ready}``. A capture may declare either branch or both.
- ``plan_obligations``/``record_obligations`` opt-in ``include_source``
  (or an explicit capabilities list); existing callers are unchanged.
- ``job_stage_capability`` maps ``source_project`` →
  ``source_lexical_ready`` and ``source_embed`` → ``source_vector_ready``.
- Deferred = settled-but-undelivered, failed = honest durable failure;
  an unprovisioned sibling is skipped, never blocked.

Fixtures use real on-disk ``Store.create`` databases, per the v5 hard
rule that tests exercise the real kernel — no mocked store.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
)
from verbatim.core.types_v4 import CapabilityName
from verbatim.ingest import Ingester
from verbatim.readiness import (
    ALL_CAPS,
    CAP_SOURCE_LEXICAL,
    CAP_SOURCE_VECTOR,
    PIPELINE_CAPS,
    SOURCE_CAPS,
    ReadinessEngine,
    ingest_receipt_id,
    job_stage_capability,
    plan_obligations,
)
from verbatim.storage.store import Store


# ---------------------------------------------------------------------
# fixtures + helpers (same conventions as tests/readiness/test_readiness.py)
# ---------------------------------------------------------------------

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")


@pytest.fixture
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    c = replace(c, capture=replace(c.capture, enabled=True))
    c = replace(c, admission=replace(c.admission, require_review=False))
    return c


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v5_readiness.db"))
    yield s
    s.close()


def _env(
    text: str,
    scope: Scope = SCOPE,
    ext: str | None = None,
    kind: SourceKind = SourceKind.USER_MESSAGE,
) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test",
        source_kind=kind,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
        external_id=ext,
    )


def _states(snap: dict) -> dict[str, str]:
    return {k: v["state"] for k, v in snap["states"].items()}


def _obligation_rows(store: Store, receipt_id: str) -> dict[str, dict]:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT * FROM readiness_obligations WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchall()
    cols = [
        "obligation_id", "receipt_id", "scope_id", "capability",
        "depends_on_json", "state", "error", "created_us", "updated_us",
    ]
    return {r[3]: dict(zip(cols, r)) for r in rows}


def _insert_source_job(
    store: Store, kind: str, source_id: str, revision: int, job_id: str
) -> None:
    """Durable source_* job evidence for the convergence path.

    The ``jobs.kind`` CHECK constraint still predates the v5 kind
    migration (the schema worker owns it — V5-08.16 allows a normal
    migration). ``ignore_check_constraints`` relaxes that not-yet-landed
    check inside one transaction so the test exercises the REAL
    readiness code path against the post-migration row shape.
    """
    with store.tx() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "INSERT INTO jobs (job_id, scope_id, kind, state,"
            " input_refs_json) VALUES (?,?,?,?,?)",
            (
                job_id,
                scope_key(SCOPE),
                kind,
                "queued",
                json.dumps(
                    {"source_id": source_id, "revision": int(revision)}
                ),
            ),
        )
        conn.execute("PRAGMA ignore_check_constraints = OFF")


SRC_LEX = CapabilityName.SOURCE_LEXICAL_READY
SRC_VEC = CapabilityName.SOURCE_VECTOR_READY


# ---------------------------------------------------------------------
# capability registration + plan_obligations
# ---------------------------------------------------------------------


def test_source_capability_members_registered():
    """The v5 members behave exactly like natively-declared enum members
    — value lookup, attribute access, iteration, identity."""
    assert CapabilityName.SOURCE_LEXICAL_READY.value == CAP_SOURCE_LEXICAL
    assert CapabilityName.SOURCE_VECTOR_READY.value == CAP_SOURCE_VECTOR
    assert (
        CapabilityName("source_lexical_ready")
        is CapabilityName.SOURCE_LEXICAL_READY
    )
    assert (
        CapabilityName("source_vector_ready")
        is CapabilityName.SOURCE_VECTOR_READY
    )
    assert CapabilityName.SOURCE_LEXICAL_READY in list(CapabilityName)
    assert CapabilityName["SOURCE_VECTOR_READY"] is SRC_VEC
    # Strings stay identical to the frozen memory.types contract — and
    # readiness never needed verbatim.memory to get there (importable
    # without it; no import cycle).
    assert CAP_SOURCE_LEXICAL == "source_lexical_ready"
    assert CAP_SOURCE_VECTOR == "source_vector_ready"


def test_plan_obligations_default_is_legacy_set():
    """The default plan is byte-identical to the pre-v5 declaration —
    no source caps, so existing callers' obligation sets do not drift."""
    plan = plan_obligations()
    assert plan == ALL_CAPS
    assert not (set(plan) & set(SOURCE_CAPS))


def test_plan_obligations_include_source():
    plan = plan_obligations(include_source=True)
    # Canonical order: claim pipeline, source branch, failed last.
    assert plan == tuple(PIPELINE_CAPS) + tuple(SOURCE_CAPS) + (
        CapabilityName.FAILED,
    )
    # include_source is idempotent when caps already list the branch.
    again = plan_obligations(list(plan), include_source=True)
    assert again == plan
    # A source-only declaration (infer=False surfaces): claim stages
    # absent — intentionally not requested — while the source branch is
    # still owed.
    source_only = plan_obligations(
        [CapabilityName.ACCEPTED, CapabilityName.SCREENED, "failed"],
        include_source=True,
    )
    assert source_only == (
        CapabilityName.ACCEPTED,
        CapabilityName.SCREENED,
        CapabilityName.SOURCE_LEXICAL_READY,
        CapabilityName.SOURCE_VECTOR_READY,
        CapabilityName.FAILED,
    )
    # Single source-capability plans are expressible too (lexical only).
    lex_only = plan_obligations(
        ["accepted", "screened", "source_lexical_ready"]
    )
    assert SRC_LEX in lex_only and SRC_VEC not in lex_only
    with pytest.raises(VerbatimError) as ei:
        plan_obligations([])
    assert ei.value.code is ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as ei:
        plan_obligations(["not_a_capability"])
    assert ei.value.code is ErrorCode.VALIDATION


# ---------------------------------------------------------------------
# obligation recording — source branch DAG shape
# ---------------------------------------------------------------------


def test_source_branch_records_siblings_on_screened(store):
    eng = ReadinessEngine(store)
    rid = "rc_v5:src1:1"
    with store.tx() as conn:
        created = eng.record_obligations(
            conn, rid, scope_key(SCOPE), include_source=True
        )
    assert len(created) == 8
    rows = _obligation_rows(store, rid)
    assert len(rows) == 8
    # Both source stages gate on screened — the shared admission gate —
    # never on claim harvest/review outputs.
    assert json.loads(rows["source_lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]
    assert json.loads(rows["source_vector_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]
    # The claim branch edges are untouched.
    assert json.loads(rows["lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]
    assert json.loads(rows["semantic_ready"]["depends_on_json"]) == [
        f"ro:{rid}:lexical_ready"
    ]
    assert rows["accepted"]["state"] == "succeeded"
    assert rows["source_lexical_ready"]["state"] == "pending"
    assert rows["source_vector_ready"]["state"] == "pending"
    assert rows["failed"]["state"] == "pending"


def test_source_branch_via_explicit_capabilities_list(store):
    """The capabilities-list form declares the branch without the flag —
    'or capabilities list' per the contract."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:src2:1"
    with store.tx() as conn:
        eng.record_obligations(
            conn,
            rid,
            scope_key(SCOPE),
            list(ALL_CAPS) + [SRC_LEX, SRC_VEC],
        )
    rows = _obligation_rows(store, rid)
    assert len(rows) == 8
    assert json.loads(rows["source_lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]


def test_source_only_capture_declares_no_claim_stages(store):
    """infer=False-style source capture: claim stages are *absent*
    (intentionally not requested), not deferred — while source work is
    owed (V5-08.12/08.15)."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:src3:1"
    caps = plan_obligations(
        [CapabilityName.ACCEPTED, CapabilityName.SCREENED, "failed"],
        include_source=True,
    )
    with store.tx() as conn:
        eng.record_obligations(
            conn, rid, scope_key(SCOPE), caps, pipeline=False
        )
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st == {
        "accepted": "succeeded",
        "screened": "succeeded",
        "source_lexical_ready": "pending",
        "source_vector_ready": "pending",
        "failed": "pending",
    }
    assert "lexical_ready" not in st  # never requested


def test_pipeline_false_keeps_source_branch_pending(store):
    """V5-08.15: ``pipeline=False`` marks claim production not requested
    (deferred claim stages) but MUST NOT strand admissible source work —
    declared source obligations still start pending."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:src4:1"
    with store.tx() as conn:
        eng.record_obligations(
            conn,
            rid,
            scope_key(SCOPE),
            include_source=True,
            pipeline=False,
        )
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st["accepted"] == "succeeded"
    assert st["screened"] == "succeeded"  # inline screen committed
    for cap in ("lexical_ready", "semantic_ready", "derived_ready"):
        assert st[cap] == "deferred", cap
    assert st["source_lexical_ready"] == "pending"
    assert st["source_vector_ready"] == "pending"
    # The receipt is honestly incomplete: owed source work outstanding.
    assert snap["ready"] is False and snap["complete"] is False


def test_source_obligations_idempotent_and_additive(store):
    """Re-recording converges; a later call may declare the source branch
    onto a receipt that already carries the claim DAG (existing rows are
    never rewritten)."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:src5:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE), ALL_CAPS)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        # Second pass adds the source branch — committed claim progress
        # stays committed.
        created = eng.record_obligations(
            conn, rid, scope_key(SCOPE), ALL_CAPS, include_source=True
        )
    assert {r["capability"] for r in created} == {
        "source_lexical_ready",
        "source_vector_ready",
    }
    rows = _obligation_rows(store, rid)
    assert len(rows) == 8
    assert rows["screened"]["state"] == "succeeded"
    assert json.loads(rows["source_lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]


# ---------------------------------------------------------------------
# dependency enforcement + skipped-sibling semantics
# ---------------------------------------------------------------------


def test_source_branch_dep_enforcement(store):
    """Source obligations are real DAG members: fulfillment before
    ``screened`` settles is STALE_DEPENDENCY, not a silent pass."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:dep:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        with pytest.raises(VerbatimError) as ei:
            eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        assert ei.value.code is ErrorCode.STALE_DEPENDENCY
        with pytest.raises(VerbatimError) as ei:
            eng.fulfill(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
        assert ei.value.code is ErrorCode.STALE_DEPENDENCY
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        assert (
            eng.state_of(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
            == "succeeded"
        )


def test_source_siblings_do_not_chain(store):
    """vector does not wait on lexical: the two source stages are
    siblings on ``screened`` — the vector projection may publish while
    the lexical one is still owed, and vice versa."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:sib:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        # Vector fulfills first — it is not gated on source_lexical_ready.
        eng.fulfill(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
        assert (
            eng.state_of(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
            == "pending"
        )
        assert (
            eng.state_of(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
            == "succeeded"
        )


def test_skipped_capability_substitutes_transitively(store):
    """Skipped-capability semantics unchanged for the new branch: a
    capability omitted from the declared set is skipped, not blocking —
    dependents re-link to the nearest recorded ancestor (_PARENTS walk)."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:skip:1"
    with store.tx() as conn:
        # No screened recorded: the source cap re-links to ``accepted``.
        eng.record_obligations(
            conn,
            rid,
            scope_key(SCOPE),
            [
                CapabilityName.ACCEPTED,
                CapabilityName.SOURCE_LEXICAL_READY,
                CapabilityName.FAILED,
            ],
        )
    rows = _obligation_rows(store, rid)
    assert json.loads(rows["source_lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:accepted"
    ]
    # …and accepted is already succeeded, so the cap is fulfillable.
    with store.tx() as conn:
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
    assert (
        ReadinessEngine(store).receipt_state(rid)["states"][
            "source_lexical_ready"
        ]["state"]
        == "succeeded"
    )


def test_deferred_source_sibling_unblocks(store):
    """An unprovisioned sibling is skipped, not blocked: a deferred
    source_lexical_ready must not stall source_vector_ready."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:defsib:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.defer(
            conn, rid, CapabilityName.SOURCE_LEXICAL_READY,
            "projection_unprovisioned",
        )
        eng.fulfill(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st["source_lexical_ready"] == "deferred"
    assert st["source_vector_ready"] == "succeeded"


# ---------------------------------------------------------------------
# wait_ready on source capabilities
# ---------------------------------------------------------------------


def test_wait_ready_on_source_caps_only(store):
    """A caller may await just the source barrier (V5-08.13 shape) —
    claim stages still pending do not block a source-scoped wait."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:wait:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
    snap = eng.wait_ready(
        rid,
        capabilities=["source_lexical_ready"],  # plain strings work too
        deadline_us=now_us() + 200_000,
    )
    assert snap["ready"] is True
    assert snap["states"]["source_lexical_ready"]["state"] == "succeeded"
    # The full-receipt view stays honest about the still-owed work.
    full = eng.receipt_state(rid)
    assert full["ready"] is False
    assert "lexical_ready" in full["pending"]


def test_wait_ready_default_covers_source_branch(store):
    """The default wait is 'every recorded capability' — source rows are
    reported and awaited, never hidden from the snapshot."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:waitall:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.LEXICAL_READY)
        eng.fulfill(conn, rid, CapabilityName.SEMANTIC_READY)
        eng.fulfill(conn, rid, CapabilityName.DERIVED_READY)
        # source branch still owed → honest pending at the deadline
    snap = eng.wait_ready(rid, deadline_us=now_us() + 120_000)
    assert snap["deadline_exceeded"] is True
    assert snap["ready"] is False
    assert "source_lexical_ready" in snap["pending"]
    assert "source_vector_ready" in snap["pending"]
    st = _states(snap)
    assert st["lexical_ready"] == "succeeded"
    # Once the source jobs commit, the same wait reports ready.
    with store.tx() as conn:
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
    snap = eng.wait_ready(rid, deadline_us=now_us() + 120_000)
    assert snap["ready"] is True
    assert snap["complete"] is True


# ---------------------------------------------------------------------
# deferred / failed honesty
# ---------------------------------------------------------------------


def test_source_deferred_is_settled_not_failed(store):
    """An unprovisioned projection defers — the receipt may be ready
    while the state field still says *deferred*, never faked succeeded."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:defer:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        eng.defer(
            conn, rid, CapabilityName.SOURCE_VECTOR_READY,
            "encoder_unavailable",
        )
    snap = eng.wait_ready(
        rid,
        capabilities=[SRC_LEX, SRC_VEC],
        deadline_us=now_us() + 200_000,
    )
    assert snap["ready"] is True  # deferred settles the barrier…
    assert "source_vector_ready" in snap["deferred"]  # …but stays labeled
    assert snap["states"]["source_vector_ready"]["error"] == (
        "encoder_unavailable"
    )


def test_source_failure_fires_indicator_and_keeps_sibling(store):
    """A terminal source_project failure lands on source_lexical_ready
    and fires the receipt's failed indicator; the vector sibling is not
    cancelled — committed independence, not a linear chain."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:fail:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fail(
            conn, rid, CapabilityName.SOURCE_LEXICAL_READY,
            "utf8_decode_error",
        )
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st["source_lexical_ready"] == "failed"
    assert st["failed"] == "failed"
    assert st["source_vector_ready"] == "pending"  # sibling unaffected
    assert snap["ready"] is False
    res = eng.wait_ready(rid, deadline_us=now_us() + 50_000)
    assert res["complete"] is False  # vector still owed
    assert "source_lexical_ready" in res["failed"]
    assert res.get("deadline_exceeded") is None or res["ready"] is False


def test_screened_failure_cancels_both_branches(store):
    """The shared admission gate failing cancels the source branch too —
    an unscreened source must never publish projections (V5-08.1)."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:gatefail:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fail(conn, rid, CapabilityName.SCREENED, "screen_error")
    st = _states(eng.receipt_state(rid))
    assert st["screened"] == "failed"
    assert st["lexical_ready"] == "cancelled"
    assert st["source_lexical_ready"] == "cancelled"
    assert st["source_vector_ready"] == "cancelled"
    assert st["failed"] == "failed"


def test_succeeded_source_cap_cannot_be_failed(store):
    eng = ReadinessEngine(store)
    rid = "rc_v5:nofail:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
        eng.fulfill(conn, rid, CapabilityName.SCREENED)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        with pytest.raises(VerbatimError) as ei:
            eng.fail(
                conn, rid, CapabilityName.SOURCE_LEXICAL_READY, "late"
            )
        assert ei.value.code is ErrorCode.INVALID_TRANSITION


# ---------------------------------------------------------------------
# terminal job failure mapping (V5-08.16)
# ---------------------------------------------------------------------


def test_job_stage_capability_source_kinds():
    assert (
        job_stage_capability(JobKind.SOURCE_PROJECT)
        is CapabilityName.SOURCE_LEXICAL_READY
    )
    assert (
        job_stage_capability(JobKind.SOURCE_EMBED)
        is CapabilityName.SOURCE_VECTOR_READY
    )
    # Plain-string kinds map identically (the function normalizes).
    assert (
        job_stage_capability("source_project")
        is CapabilityName.SOURCE_LEXICAL_READY
    )
    assert (
        job_stage_capability("source_embed")
        is CapabilityName.SOURCE_VECTOR_READY
    )
    # Backfill carries its own receipt — never mapped onto a capture DAG.
    assert job_stage_capability(JobKind.SOURCE_BACKFILL) is None
    # Existing mappings preserved.
    assert job_stage_capability(JobKind.HARVEST) is CapabilityName.SCREENED
    assert job_stage_capability(JobKind.ADMIT) is (
        CapabilityName.LEXICAL_READY
    )
    assert job_stage_capability("embed") is CapabilityName.SEMANTIC_READY
    assert job_stage_capability("purge") is None


# ---------------------------------------------------------------------
# convergence — durable job evidence materializes the branch
# ---------------------------------------------------------------------


def test_ensure_for_source_materializes_source_branch(store, cfg):
    """A capture whose obligations were never recorded converges at the
    first job touch — with a durable ``source_project`` job row, the
    rebuilt DAG carries exactly the evidenced source capabilities."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("source convergence target"))
    sid = r.accepted[0]
    rid = ingest_receipt_id(sid, 1)
    eng = ReadinessEngine(store)

    # Simulate a capture that accepted + enqueued source work but whose
    # obligation rows were lost/never written (crash before the insert).
    with store.tx() as conn:
        conn.execute(
            "DELETE FROM readiness_obligations WHERE receipt_id = ?",
            (rid,),
        )
    _insert_source_job(store, "source_project", sid, 1, "sj-proj-1")

    with store.tx() as conn:
        rids = eng.ensure_for_source(conn, sid, 1)
    assert rid in rids
    rows = _obligation_rows(store, rid)
    # Only lexical evidenced → only lexical declared; vector is not
    # fabricated as owed.
    assert "source_lexical_ready" in rows
    assert "source_vector_ready" not in rows
    assert json.loads(rows["source_lexical_ready"]["depends_on_json"]) == [
        f"ro:{rid}:screened"
    ]

    # A second revision with an embed job gains the vector cap too.
    r2 = ing.ingest(_env("second convergence target", ext="e2"))
    sid2 = r2.accepted[0]
    rid2 = ingest_receipt_id(sid2, 1)
    with store.tx() as conn:
        conn.execute(
            "DELETE FROM readiness_obligations WHERE receipt_id = ?",
            (rid2,),
        )
    _insert_source_job(store, "source_embed", sid2, 1, "sj-emb-1")
    with store.tx() as conn:
        eng.ensure_for_source(conn, sid2, 1)
    rows2 = _obligation_rows(store, rid2)
    assert "source_vector_ready" in rows2
    assert "source_lexical_ready" not in rows2


def test_ensure_for_source_no_evidence_no_source_caps(store, cfg):
    """No source_* job rows → the converged DAG stays the legacy claim
    set. Absent evidence is honest absence, not a fabricated promise."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("plain convergence target"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    eng = ReadinessEngine(store)
    with store.tx() as conn:
        conn.execute(
            "DELETE FROM readiness_obligations WHERE receipt_id = ?",
            (rid,),
        )
        rids = eng.ensure_for_source(conn, r.accepted[0], 1)
    assert rid in rids
    rows = _obligation_rows(store, rid)
    assert len(rows) == 6
    assert not (set(rows) & {"source_lexical_ready", "source_vector_ready"})


# ---------------------------------------------------------------------
# mixed claim+source captures on a real ingest path
# ---------------------------------------------------------------------


def test_mixed_capture_real_ingest_then_source_branch(store, cfg):
    """A real v2 capture records the claim DAG; declaring the source
    branch on the same receipt merges without disturbing committed
    rows — then the whole DAG settles honestly through transitions."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("My backup window is 3am."))
    rid = ingest_receipt_id(r.accepted[0], 1)
    eng = ReadinessEngine(store)

    # Facade-style declaration of the source branch for the same receipt.
    with store.tx() as conn:
        eng.record_obligations(
            conn, rid, scope_key(SCOPE), include_source=True
        )
    assert len(_obligation_rows(store, rid)) == 8

    # Claim branch settles through the real drain.
    rep = ing.drain_report(scope=None, limit=64)
    assert rep["complete"] is True
    snap = eng.receipt_state(rid)
    st = _states(snap)
    assert st["lexical_ready"] == "succeeded"
    assert st["derived_ready"] == "succeeded"
    # …but the source branch is still owed — the receipt is honestly
    # not ready (partial route coverage, V5-08.02).
    assert st["source_lexical_ready"] == "pending"
    assert st["source_vector_ready"] == "pending"
    assert snap["ready"] is False
    assert "failed" not in snap["failed"] or st["failed"] == "pending"

    # The source jobs commit (source_project / source_embed handlers).
    with store.tx() as conn:
        eng.fulfill(conn, rid, CapabilityName.SOURCE_LEXICAL_READY)
        eng.fulfill(conn, rid, CapabilityName.SOURCE_VECTOR_READY)
    snap = eng.receipt_state(rid)
    assert snap["ready"] is True
    assert snap["complete"] is True
    assert _states(snap)["failed"] == "succeeded"


def test_pending_lists_source_obligations(store):
    """Outstanding source work is visible in pending() like any other
    obligation — operator/drain surfaces stay honest."""
    eng = ReadinessEngine(store)
    rid = "rc_v5:pend:1"
    with store.tx() as conn:
        eng.record_obligations(conn, rid, scope_key(SCOPE),
                               include_source=True)
    pending = eng.pending(scope_key(SCOPE))
    by_rid = [p for p in pending if p["receipt_id"] == rid]
    caps = {p["capability"] for p in by_rid}
    assert "source_lexical_ready" in caps
    assert "source_vector_ready" in caps
    src = next(
        p for p in by_rid if p["capability"] == "source_lexical_ready"
    )
    assert src["depends_on"] == [f"ro:{rid}:screened"]


# ---------------------------------------------------------------------
# legacy captures unaffected
# ---------------------------------------------------------------------


def test_legacy_capture_dag_unchanged(store, cfg):
    """A capture that never opts in keeps exactly the pre-v5 DAG: six
    rows, claim edges only, no source rows, identical semantics."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("legacy fact — no source branch declared"))
    rid = ingest_receipt_id(r.accepted[0], 1)

    rows = _obligation_rows(store, rid)
    assert len(rows) == 6
    assert "source_lexical_ready" not in rows
    assert "source_vector_ready" not in rows

    snap = ReadinessEngine(store).receipt_state(rid)
    st = _states(snap)
    assert set(st) == {
        "accepted", "screened", "lexical_ready",
        "semantic_ready", "derived_ready", "failed",
    }
    assert st["accepted"] == "succeeded"
    assert st["screened"] == "pending"

    # The legacy receipt still drains to ready with nothing source-related
    # ever appearing.
    ing.drain_report(scope=None, limit=64)
    snap = ReadinessEngine(store).receipt_state(rid)
    assert snap["ready"] is True
    assert "source_lexical_ready" not in _states(snap)


def test_wait_ready_legacy_receipt_source_cap_is_absent(store, cfg):
    """Awaiting a source capability on a receipt that never declared it
    reports 'absent' — not pending, not owed — distinguishing
    'not requested' from 'unavailable' (V5-08.1)."""
    ing = Ingester(store, cfg)
    r = ing.ingest(_env("no source obligations here"))
    rid = ingest_receipt_id(r.accepted[0], 1)
    snap = ReadinessEngine(store).wait_ready(
        rid,
        capabilities=[CapabilityName.SOURCE_LEXICAL_READY],
        deadline_us=now_us() + 100_000,
    )
    assert snap["ready"] is True
    assert snap["states"]["source_lexical_ready"]["state"] == "absent"
