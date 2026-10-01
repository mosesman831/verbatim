"""V4 windowed consolidation + bounded reflection (SPEC_V4 §24, §27).

Pinned behaviors:

- ``consolidate_windowed`` re-derives only the slots the declared window
  touched (``since_seq`` revision activity and/or explicit
  ``claim_ids``) and retires only observations citing a touched claim —
  a bounded pass can never invalidate a belief it did not recompute
  (V4-24.01, V4-24.08/09).
- A *correction* (claim superseded by a new revision with a different
  value) propagates: the aggregate revises, ``PersistResult``
  ``added``/``removed`` records exactly which evidence moved — the
  durable "why" on the pass report (V4-24.02/24.07).
- Support vanishing below the family threshold retires the observation
  prospectively (``recorded_until`` + ``stale_since_seq``), and the
  refined observation carries full provenance (derivations edges) and
  the derived label (producer column + observation kind, V4-27).
- ``reflect`` emits *labeled hypothesis* observations for rival slots —
  bounded by ``ReflectBudget.max_iterations`` (V4-27.06), honest
  ``stopped`` reasons, deterministic ids (V4-24.08), and NO host-tool
  surface: the pass touches only observations/evidence/derivations rows
  and leaves canonical source bytes byte-identical (C50, V4-24.04).
- ``handle_consolidate`` accepts ``window`` + ``reflect`` options inside
  the same generation-fenced job; a stale ``expected_generation``
  trigger fails STALE_PROPOSAL instead of writing from an abandoned
  view (V4-24.09), and both passes append audit events (V4-24.07).
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import ErrorCode, JobKind, VerbatimError, json_dumps
from verbatim.ingest import Ingester
from verbatim.observations import (
    SLOT_AGGREGATE_V1,
    ConsolidationWindow,
    ReflectBudget,
    consolidate,
    consolidate_windowed,
    reflect,
)
from verbatim.observations.aggregate import observation_id_for
from verbatim.observations.freshness import next_seq
from verbatim.observations.reflect import (
    REFLECT_PRODUCER_ID,
    hypothesis_id_for,
)
from verbatim.storage import repos_v3
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# fixtures + seeds
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v4obs.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:v4obs"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    return sid


def _claim(
    conn,
    scope_id,
    claim_id,
    *,
    subject="alice",
    predicate="residence",
    value="london",
    state="active",
    revision=1,
    polarity="affirmative",
    modality="asserted",
    condition=None,
    perspective_id=None,
    family_id=None,
    recorded_from=1,
    recorded_until=None,
):
    """Claim + head revision; ``recorded_from`` is caller-controlled so
    window tests can place revisions on either side of ``since_seq``."""
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES (?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    obj = json_dumps({"kind": "literal", "text": value})
    cond = json_dumps(condition) if condition is not None else None
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, condition_json, recorded_from,"
        " recorded_until, perspective_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            claim_id,
            revision,
            state,
            obj,
            polarity,
            modality,
            cond,
            recorded_from,
            recorded_until,
            perspective_id,
        ),
    )
    if family_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO evidence_families"
            " (family_id, scope_id, origin_kind, origin_id, created_event)"
            " VALUES (?,?,?,?,0)",
            (family_id, scope_id, "test", claim_id),
        )
        conn.execute(
            "INSERT INTO family_members (family_id, object_kind, object_id,"
            " role) VALUES (?, 'claim', ?, 'origin')",
            (family_id, claim_id),
        )


def _revise_claim(
    conn, claim_id, *, new_value, close_at, open_at, state="active"
):
    """Close the claim's head revision and open the corrected one."""
    head = conn.execute(
        "SELECT MAX(revision) FROM claim_revisions WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE claim_revisions SET recorded_until = ?"
        " WHERE claim_id = ? AND revision = ?",
        (close_at, claim_id, head),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, recorded_from)"
        " VALUES (?,?,?,?, 'affirmative','asserted',?)",
        (claim_id, head + 1, state,
         json_dumps({"kind": "literal", "text": new_value}), open_at),
    )
    return head + 1


def _observations(conn, scope_id, producer=None):
    where = {"scope_id": scope_id}
    if producer is not None:
        where["producer"] = producer
    return repos_v3.query(conn, "observations", where)


def _obs(conn, obs_id):
    return repos_v3.get(conn, "observations", {"observation_id": obs_id})


LONDON_OBS = observation_id_for(
    "scope:v4obs", "alice", "residence", None, "", "affirmative:london"
)


# ---------------------------------------------------------------------------
# windowed pass (V4-24.01): touched-slot scoping + correction propagation
# ---------------------------------------------------------------------------


def test_windowed_pass_rederives_only_touched_slots(store, scope_id):
    """A since_seq window touches only slots whose revisions moved —
    the untouched slot's observation is left at its revision."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1", value="london")
        _claim(conn, scope_id, "c2", family_id="f2", value="london")
        _claim(conn, scope_id, "c3", subject="bob", predicate="role",
               value="admin", family_id="f3")
        _claim(conn, scope_id, "c4", subject="bob", predicate="role",
               value="admin", family_id="f4")
        rep = consolidate(conn, scope_id)
        assert rep["observations_written"] == 2
        seq = next_seq(conn, scope_id)
        # Correction lands AFTER the watermark: c2 revises london→paris.
        _revise_claim(conn, "c2", new_value="paris",
                      close_at=seq, open_at=seq)
        win = ConsolidationWindow(since_seq=seq - 1)
        rep = consolidate_windowed(conn, scope_id, window=win)
    assert rep.slots_touched == 1
    assert rep.stopped is None
    with store.read() as conn:
        london = _obs(conn, LONDON_OBS)
        bob_obs = _obs(
            conn,
            observation_id_for(
                scope_id, "bob", "role", None, "", "affirmative:admin"
            ),
        )
        # london lost c2's support — proof dropped below the threshold
        # → retired; bob's observation untouched at rev 1.
        assert london["recorded_until"] is not None
        assert london["stale_since_seq"] is not None
        assert bob_obs["recorded_until"] is None
        assert bob_obs["revision"] == 1
    assert rep.retired == [LONDON_OBS]


def test_correction_revises_aggregate_with_why(store, scope_id):
    """A correction that keeps the slot above threshold revises the
    observation in place: new revision, evidence diff on the report —
    'why' is recorded, never silently rewritten (V4-24.02/24.07)."""
    with store.tx() as conn:
        for i, fam in enumerate(("f1", "f2", "f3"), start=1):
            _claim(conn, scope_id, f"c{i}", family_id=fam, value="london")
        consolidate(conn, scope_id)
        seq = next_seq(conn, scope_id)
        _revise_claim(conn, "c3", new_value="paris",
                      close_at=seq, open_at=seq)
        rep = consolidate_windowed(
            conn, scope_id, window=ConsolidationWindow(since_seq=seq - 1)
        )
    assert rep.observations_written == 1
    assert len(rep.refinements) == 1
    ref = rep.refinements[0]
    assert ref["observation_id"] == LONDON_OBS
    assert ref["revision"] == 2 and ref["previous_revision"] == 1
    # c3@1 left supports; c3@2 arrived as a rival (contradicts).
    assert ["supports", "claim", "c3", 1] in ref["removed"]
    assert ["contradicts", "claim", "c3", 2] in ref["added"]
    with store.read() as conn:
        obs = _obs(conn, LONDON_OBS)
        assert obs["revision"] == 2 and obs["proof_count"] == 2
        # rev-2 derivations record the new input set — provenance on
        # every revision, immutable (V3-17.02).
        rev2 = repos_v3.query(
            conn, "derivations",
            {"child_kind": "observation", "child_id": LONDON_OBS,
             "child_revision": 2},
        )
        parents = {(r["parent_id"], r["parent_revision"]) for r in rev2}
        assert ("c1", 1) in parents and ("c2", 1) in parents
        assert ("c3", 2) in parents
        rev1 = repos_v3.query(
            conn, "derivations",
            {"child_kind": "observation", "child_id": LONDON_OBS,
             "child_revision": 1},
        )
        assert {r["parent_id"] for r in rev1} == {"c1", "c2", "c3"}


def test_windowed_retirement_is_slot_scoped(store, scope_id):
    """Retirement only reaches observations citing a touched claim —
    a second slot whose support vanished OUTSIDE the window is not
    touched by the bounded pass (V4-24.08)."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1", value="london")
        _claim(conn, scope_id, "c2", family_id="f2", value="london")
        _claim(conn, scope_id, "c3", subject="bob", predicate="role",
               value="admin", family_id="f3")
        _claim(conn, scope_id, "c4", subject="bob", predicate="role",
               value="admin", family_id="f4")
        consolidate(conn, scope_id)
        seq = next_seq(conn, scope_id)
        # Both bob claims silently close — but the window only names
        # alice's slot, so bob's stale observation must NOT be retired
        # by this bounded pass.
        conn.execute(
            "UPDATE claim_revisions SET recorded_until = ?"
            " WHERE claim_id IN ('c3','c4')",
            (seq,),
        )
        win = ConsolidationWindow(claim_ids=("c1",))
        rep = consolidate_windowed(conn, scope_id, window=win)
    with store.read() as conn:
        bob_obs = _obs(
            conn,
            observation_id_for(
                scope_id, "bob", "role", None, "", "affirmative:admin"
            ),
        )
        assert bob_obs["recorded_until"] is None  # untouched by the pass
    # A full pass is the authority that retires it.
    with store.tx() as conn:
        consolidate(conn, scope_id)
    with store.read() as conn:
        assert _obs(conn, bob_obs["observation_id"])["recorded_until"] is not None


def test_windowed_no_material_change(store, scope_id):
    """An empty window is an honest no-op, not a full scan."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1")
        _claim(conn, scope_id, "c2", family_id="f2")
        consolidate(conn, scope_id)
        rep = consolidate_windowed(
            conn, scope_id, window=ConsolidationWindow(since_seq=10**9)
        )
    assert rep.stopped == "no_material_change"
    assert rep.observations_written == 0 and rep.retired == []


def test_windowed_max_outputs_bounds_and_defers_retirement(store, scope_id):
    """When max_outputs binds, the pass reports truncation and defers
    retirement — a partial pass never retires what it did not finish
    recomputing (V4-24.09)."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1", value="london")
        _claim(conn, scope_id, "c2", family_id="f2", value="london")
        _claim(conn, scope_id, "c3", subject="bob", predicate="role",
               value="admin", family_id="f3")
        _claim(conn, scope_id, "c4", subject="bob", predicate="role",
               value="admin", family_id="f4")
        win = ConsolidationWindow(since_seq=0, max_outputs=1)
        rep = consolidate_windowed(conn, scope_id, window=win)
    assert rep.truncated and rep.stopped == "max_outputs"
    assert rep.observations_written == 1
    assert rep.retired == []  # deferred — not recomputed fully


def test_windowed_idempotent_reprocessing(store, scope_id):
    """Redelivery converges: the same window twice writes nothing the
    second time (V4-24.08, V3-23.04)."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1")
        _claim(conn, scope_id, "c2", family_id="f2")
        win = ConsolidationWindow(since_seq=0)
        first = consolidate_windowed(conn, scope_id, window=win)
        second = consolidate_windowed(conn, scope_id, window=win)
    assert first.observations_written == 1
    assert second.observations_written == 0
    assert second.stopped == "no_material_change"


# ---------------------------------------------------------------------------
# bounded reflection (V4-24.06, §27, C50)
# ---------------------------------------------------------------------------


def _rival_slot(conn, scope_id, subject="alice", predicate="residence",
                a="london", b="paris"):
    """Two rival values in one slot from independent families."""
    _claim(conn, scope_id, f"{subject}-{a}-1", subject=subject,
           predicate=predicate, value=a, family_id=f"{subject}-{a}-fam")
    _claim(conn, scope_id, f"{subject}-{b}-1", subject=subject,
           predicate=predicate, value=b, family_id=f"{subject}-{b}-fam")


def test_reflect_emits_labeled_hypothesis(store, scope_id):
    """A rival slot yields ONE labeled hypothesis observation citing the
    exact claim revisions on every side — derived, provenance-linked,
    never merged (V4-24.04/06)."""
    with store.tx() as conn:
        _rival_slot(conn, scope_id)
        rep = reflect(conn, scope_id)
    assert rep.hypotheses_written == 1
    assert rep.stopped is None
    obs_id = rep.observation_ids[0]
    assert obs_id == hypothesis_id_for(
        scope_id, "alice", "residence", None, ""
    )
    with store.read() as conn:
        obs = _obs(conn, obs_id)
        assert obs["producer"] == REFLECT_PRODUCER_ID
        assert obs["text"].startswith("hypothesis:")
        assert "contested" in obs["text"] and "unresolved" in obs["text"]
        ev = repos_v3.query(
            conn, "observation_evidence",
            {"observation_id": obs_id, "revision": 1},
        )
        assert {e["object_id"] for e in ev} == {
            "alice-london-1", "alice-paris-1"
        }
        edges = repos_v3.query(
            conn, "derivations",
            {"child_kind": "observation", "child_id": obs_id},
        )
        assert {e["parent_id"] for e in edges} == {
            "alice-london-1", "alice-paris-1"
        }
        assert all(e["producer_id"] == f"obs:{obs_id}" for e in edges)


def test_reflect_iteration_cap_is_honest(store, scope_id):
    """max_iterations bounds the rounds; unprocessed regions are
    reported as remaining, never silently dropped (V4-27.06)."""
    with store.tx() as conn:
        _rival_slot(conn, scope_id)
        _rival_slot(conn, scope_id, subject="bob", predicate="role",
                    a="admin", b="viewer")
        _rival_slot(conn, scope_id, subject="carol", predicate="dept",
                    a="eng", b="ops")
        rep = reflect(
            conn, scope_id,
            budget=ReflectBudget(max_iterations=1, max_outputs=1),
        )
    assert rep.iterations == 1
    assert rep.hypotheses_written == 1
    assert rep.regions_remaining == 2
    assert rep.stopped == "iteration_cap"


def test_reflect_default_budget_three_rounds(store, scope_id):
    """V4-27.06 default is three rounds — one rival per round."""
    assert ReflectBudget().max_iterations == 3
    with store.tx() as conn:
        for i in range(4):
            _rival_slot(conn, scope_id, subject=f"s{i}", predicate="p",
                        a="x", b="y")
        rep = reflect(
            conn, scope_id,
            budget=ReflectBudget(max_outputs=1),
        )
    assert rep.iterations == 3
    assert rep.regions_remaining == 1
    assert rep.stopped == "iteration_cap"


def test_reflect_deadline_and_disabled(store, scope_id):
    """Deadline stops are named; a disabled producer fails loudly."""
    with store.tx() as conn:
        _rival_slot(conn, scope_id)
        ticks = iter([0.0, 10.0])  # start, then past-deadline check
        rep = reflect(
            conn, scope_id,
            budget=ReflectBudget(deadline_s=1.0),
            clock=lambda: next(ticks, 99.0),
        )
    assert rep.stopped == "deadline"
    assert rep.hypotheses_written == 0
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            reflect(conn, scope_id, enabled=False)
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_reflect_empty_scope_no_material_change(store, scope_id):
    with store.tx() as conn:
        rep = reflect(conn, scope_id)
    assert rep.hypotheses_written == 0
    assert rep.stopped == "no_material_change"


def test_reflect_no_host_tools_and_canonical_bytes_intact(store, scope_id):
    """C50: reflect performs no host work — it writes ONLY
    observations/observation_evidence/derivations rows. Canonical
    source bytes are byte-identical afterwards (V4-24.04)."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
            " created_us) VALUES ('src-canon', 'test', 'user_message', ?, 1)",
            (scope_id,),
        )
        payload = b"canonical bytes -- never rewritten by reflection"
        conn.execute(
            "INSERT INTO source_revisions (source_id, revision, payload,"
            " payload_hmac, event_us, captured_us, provenance)"
            " VALUES ('src-canon', 1, ?, X'00', 1, 1, 'direct_user')",
            (payload,),
        )
        _rival_slot(conn, scope_id)
        before = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in (
                "claims", "claim_revisions", "sources", "source_revisions",
                "spans", "purges", "quarantine", "events",
            )
        }
        rep = reflect(conn, scope_id)
        after = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in before
        }
        canon = conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = 'src-canon' AND revision = 1"
        ).fetchone()[0]
    assert rep.hypotheses_written == 1
    assert before == after  # zero writes outside the observation plane
    assert canon == payload  # canonical bytes untouched


def test_reflect_hypothesis_retires_when_conflict_resolves(store, scope_id):
    """A resolved rival slot retires its hypothesis on the next full
    pass — derived labels never outlive their evidence (V4-24.08)."""
    hyp_id = hypothesis_id_for(scope_id, "alice", "residence", None, "")
    with store.tx() as conn:
        _rival_slot(conn, scope_id)
        reflect(conn, scope_id)
        assert _obs(conn, hyp_id)["recorded_until"] is None
        # Correction resolves the conflict: paris revises to london.
        seq = next_seq(conn, scope_id)
        _revise_claim(conn, "alice-paris-1", new_value="london",
                      close_at=seq, open_at=seq)
        rep = reflect(conn, scope_id)
    assert hyp_id in rep.retired
    with store.read() as conn:
        assert _obs(conn, hyp_id)["recorded_until"] is not None


# ---------------------------------------------------------------------------
# job-handler integration (V4-24.01/24.07/24.09, §42)
# ---------------------------------------------------------------------------


def test_handle_consolidate_window_plus_reflect(store, scope_id):
    """One consolidate job can carry the window + a reflection budget;
    both passes record audit events with their reports."""
    with store.tx() as conn:
        _rival_slot(conn, scope_id)
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        job_id = ing.jobs.enqueue(
            conn, scope_id, JobKind.CONSOLIDATE,
            {"window": {"since_seq": 0}, "reflect": True},
        )
    assert ing.run_pending(scope=scope_id) == 1
    with store.read() as conn:
        state = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        assert state == "succeeded"
        kinds = {
            r[0]
            for r in conn.execute(
                "SELECT kind FROM events WHERE scope_id = ?", (scope_id,)
            ).fetchall()
        }
        assert "consolidation_pass" in kinds
        assert "reflection_pass" in kinds
        hyps = [
            o for o in _observations(conn, scope_id)
            if o["producer"] == REFLECT_PRODUCER_ID
        ]
        assert len(hyps) == 1


def test_handle_consolidate_stale_generation_fails(store, scope_id):
    """A trigger naming a superseded generation fails STALE_PROPOSAL —
    the pass never writes from an abandoned view (V4-24.09)."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1")
        _claim(conn, scope_id, "c2", family_id="f2")
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        job_id = ing.jobs.enqueue(
            conn, scope_id, JobKind.CONSOLIDATE,
            {"window": {"since_seq": 0, "expected_generation": 9999}},
        )
    assert ing.run_pending(scope=scope_id) == 1
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, error_code FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        assert row[0] == "failed"
        assert row[1] == ErrorCode.STALE_PROPOSAL.value
        assert _observations(conn, scope_id) == []


def test_handle_consolidate_matching_generation_succeeds(store, scope_id):
    """The fence opens when the trigger's generation matches the lease."""
    with store.tx() as conn:
        _claim(conn, scope_id, "c1", family_id="f1")
        _claim(conn, scope_id, "c2", family_id="f2")
    ing = Ingester(store, VerbatimConfig())
    with store.tx() as conn:
        job_id = ing.jobs.enqueue(
            conn, scope_id, JobKind.CONSOLIDATE,
            {"window": {"since_seq": 0, "expected_generation": 1}},
        )
    assert ing.run_pending(scope=scope_id) == 1
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, generation FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        assert row[0] == "succeeded"
        assert int(row[1]) == 1  # fence compared against the real lease
        assert len(_observations(conn, scope_id)) == 1
