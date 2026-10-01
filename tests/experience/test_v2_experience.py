"""Experience-package tests: episodes, procedures, prospective records.

Covers SPEC_V2 §21 (episode grouping + temporal membership), §22
(plans are not facts; overdue ≠ completed), §23-24 (evidence-backed
procedures, environment 3-valued matching, outcome drift, dependency
invalidation edges).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.experience.conftest import make_span

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.experience import (
    activate_procedure,
    attach,
    cancel,
    close_episode,
    complete,
    detach,
    drift_status,
    due,
    environment_fingerprint,
    environment_match,
    episode_summary,
    episodes_for_scope,
    match_procedure,
    members,
    open_episode,
    plan,
    plans_for_scope,
    procedure_view,
    procedures_for_scope,
    propose_procedure,
    recurrence_next,
    record_outcome,
    reschedule,
    set_procedure_state,
    start,
    plan_view,
)
from verbatim.storage.repos import EventsRepo
from verbatim.storage.repos_v2 import DependencyRepo
from verbatim.storage.store import Store

DAY_US = 86_400 * 1_000_000
T0 = 1_700_000_000_000_000  # arbitrary epoch micros


def _tick(store: Store, conn, scope_id: str, kind: str = "test.tick") -> int:
    """Append a journal event so recorded_from/until seqs advance."""
    return EventsRepo(store).append(
        conn, scope_id, kind, "tester", {}, "pol-1"
    )


def _propose(store, conn, scope_id, label="run the pytest suite", **kw):
    return propose_procedure(
        store,
        conn,
        scope_id,
        label,
        kw.pop(
            "steps",
            [
                {"description": "install dependencies"},
                {"description": "run pytest -q", "verification": "exit 0"},
            ],
        ),
        **kw,
    )


# ----------------------------------------------------------------------
# episodes (§21)
# ----------------------------------------------------------------------


def test_episode_open_attach_close_summary(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        eid = open_episode(
            store,
            conn,
            scope_id,
            kind="task",
            host_task_id="task-9",
            host_session_id="sess-1",
            label="fix flaky test",
        )
        attach(store, conn, eid, "span", "sp-a", ord=0)
        attach(store, conn, eid, "claim", "cl-1", ord=1)
        attach(store, conn, eid, "span", "sp-b", ord=2)

    s = episode_summary(store, eid)
    assert s["state"] == "open"
    assert s["episode_kind"] == "task"
    assert s["label"] == "fix flaky test"
    assert s["host_task_id"] == "task-9"
    assert s["member_count"] == 3
    assert {m["object_id"] for m in s["members"]["span"]} == {"sp-a", "sp-b"}
    assert [m["object_id"] for m in s["members"]["claim"]] == ["cl-1"]

    with store.tx() as conn:
        close_episode(store, conn, eid)
    s2 = episode_summary(store, eid)
    assert s2["state"] == "closed"
    assert s2["recorded_until"] is not None
    # history is not deleted by closing
    assert s2["member_count"] == 3

    # attaching to a closed episode fails loudly
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            attach(store, conn, eid, "span", "sp-c")
        assert ei.value.code == ErrorCode.INVALID_TRANSITION

    # unknown episode: attach, close, and summary all fail closed
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei2:
            attach(store, conn, "nope-episode", "span", "x")
        assert ei2.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
        with pytest.raises(VerbatimError) as ei3:
            close_episode(store, conn, "nope-episode")
        assert ei3.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN
    with pytest.raises(VerbatimError):
        episode_summary(store, "nope-episode")


def test_episode_membership_as_of_seq(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        eid = open_episode(store, conn, scope_id, kind="meeting")
        attach(store, conn, eid, "span", "sp-1")  # recorded_from = 1

    with store.tx() as conn:
        _tick(store, conn, scope_id)  # event_seq 1
        attach(store, conn, eid, "span", "sp-2")  # recorded_from = 2

    with store.tx() as conn:
        _tick(store, conn, scope_id)  # event_seq 2
        detach(store, conn, eid, "span", "sp-1")  # recorded_until = 3

    current = {m["object_id"] for m in members(store, eid)}
    assert current == {"sp-2"}
    at_1 = {m["object_id"] for m in members(store, eid, as_of_seq=1)}
    assert at_1 == {"sp-1"}
    at_2 = {m["object_id"] for m in members(store, eid, as_of_seq=2)}
    assert at_2 == {"sp-1", "sp-2"}
    at_3 = {m["object_id"] for m in members(store, eid, as_of_seq=3)}
    assert at_3 == {"sp-2"}


def test_episode_hierarchy_and_for_scope(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        parent = open_episode(store, conn, scope_id, kind="task", label="parent")
        child = open_episode(
            store, conn, scope_id, kind="task", parent_episode_id=parent,
            label="child",
        )
        open_episode(store, conn, scope_id, kind="meeting", label="sync")
    rows = episodes_for_scope(store, scope_id, kind="task")
    assert {r["episode_id"] for r in rows} == {parent, child}
    assert len(episodes_for_scope(store, scope_id)) == 3
    s = episode_summary(store, child)
    assert s["parent_episode_id"] == parent


# ----------------------------------------------------------------------
# procedures: proposal, evidence edges, lifecycle fence (§23)
# ----------------------------------------------------------------------


def test_propose_procedure_records_steps_and_dependencies(
    store: Store, scope_id: str, scope
) -> None:
    source_id, span_id = make_span(store, scope, b"ran pytest -q: 12 passed")
    with store.tx() as conn:
        pid = propose_procedure(
            store,
            conn,
            scope_id,
            "run the pytest suite",
            [
                {
                    "description": "install dependencies",
                    "span_id": span_id,
                    "input_revision": 1,
                },
                {
                    "description": "run pytest -q",
                    "hazard": "may write .pytest_cache",
                    "verification": "exit code 0",
                    "precondition": {"op": "eq", "key": "cwd", "value": "repo"},
                },
            ],
            environment={"os": "Linux", "python": "3.11"},
            span_evidence=[{"span_id": span_id, "revision": 1}],
        )

    v = procedure_view(store, pid)
    assert v["state"] == "proposed"
    assert v["advisory"] is True
    assert v["environment"] == {"os": "Linux", "python": "3.11"}
    assert [s["description"] for s in v["steps"]] == [
        "install dependencies",
        "run pytest -q",
    ]
    assert v["steps"][0]["span_id"] == span_id
    assert v["steps"][1]["verification"] == "exit code 0"
    # dependency edges: step span + explicit span_evidence (deduped by PK)
    inputs = {(d["input_kind"], d["input_id"]) for d in v["evidence_inputs"]}
    assert ("span", span_id) in inputs
    assert all(d["invalidation"] == "revalidate" for d in v["evidence_inputs"])
    # the edge is findable from the input side (invalidation walking)
    dependents = DependencyRepo(store).dependents_of("span", span_id)
    assert any(
        d["derived_kind"] == "procedure" and d["derived_id"] == pid
        for d in dependents
    )
    assert v["outcome_stats"]["total"] == 0
    assert v["drift"] == "no_outcomes"


def test_propose_procedure_validation(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            propose_procedure(store, conn, scope_id, "empty", [])
        assert ei.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError):
            propose_procedure(
                store, conn, scope_id, "bad", [{"span_id": "x"}]
            )  # no description
        with pytest.raises(VerbatimError):
            propose_procedure(
                store,
                conn,
                scope_id,
                "dup",
                [{"description": "a", "step_no": 1},
                 {"description": "b", "step_no": 1}],
            )
        with pytest.raises(VerbatimError):
            propose_procedure(
                store, conn, scope_id, "env", [{"description": "a"}],
                environment="not-a-dict",
            )


def test_procedure_activation_version_fence(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = _propose(store, conn, scope_id)
    rv = procedure_view(store, pid)["row_version"]

    with store.tx() as conn:
        activate_procedure(store, conn, pid, rv)
    v = procedure_view(store, pid)
    assert v["state"] == "active"
    assert v["row_version"] == rv + 1

    # A stale expected version on a legal transition must fence, not write.
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            set_procedure_state(store, conn, pid, "review", rv)
        assert ei.value.code == ErrorCode.STALE_PROPOSAL
    assert procedure_view(store, pid)["state"] == "active"

    # active -> proposed is not a legal transition at all
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei2:
            set_procedure_state(store, conn, pid, "proposed", rv + 1)
        assert ei2.value.code == ErrorCode.INVALID_TRANSITION


def test_procedure_state_transitions_and_retire(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = _propose(store, conn, scope_id)
    with store.tx() as conn:
        set_procedure_state(store, conn, pid, "retired", 1)
    assert procedure_view(store, pid)["state"] == "retired"
    # retired is terminal
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            set_procedure_state(store, conn, pid, "active", 2)
        assert ei.value.code == ErrorCode.INVALID_TRANSITION
    # retired stays inspectable but out of candidates
    assert match_procedure(store, scope_id, "pytest suite", {}) == []
    assert len(match_procedure(
        store, scope_id, "pytest suite", {}, include_retired=True
    )) == 1


def test_record_outcome_and_stats(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = _propose(store, conn, scope_id)
    for i, outcome in enumerate(
        ["success", "success", "failure", "partial", "unknown"]
    ):
        with store.tx() as conn:
            record_outcome(
                store, conn, pid, 1, outcome, checker="pytest",
                checked_artifact="junit.xml", recorded_us=T0 + i,
            )
    v = procedure_view(store, pid)
    st = v["outcome_stats"]
    assert st["success"] == 2 and st["failure"] == 1
    assert st["partial"] == 1 and st["unknown"] == 1 and st["total"] == 5
    assert st["last_outcome"] == "unknown"
    assert len(v["outcomes"]) == 5
    assert v["outcomes"][0]["checker"] == "pytest"
    assert v["outcomes"][0]["checked_artifact"] == "junit.xml"

    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            record_outcome(store, conn, pid, 1, "exploded", checker="x")
        assert ei.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError) as ei2:
            record_outcome(store, conn, "nope-proc", 1, "success", checker="x")
        assert ei2.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_drift_detection(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = _propose(store, conn, scope_id)
    seq = ["success", "success", "success", "failure", "failure", "failure"]
    for i, o in enumerate(seq):
        with store.tx() as conn:
            record_outcome(store, conn, pid, 1, o, checker="ci", recorded_us=T0 + i)
    assert procedure_view(store, pid)["drift"] == "drifted"

    # unit-level: degrade vs stable vs insufficient
    def rows(*os):
        return [{"outcome": o} for o in os]

    assert drift_status([]) == "no_outcomes"
    assert drift_status(rows("failure", "failure")) == "insufficient_data"
    assert drift_status(rows("success", "success", "success", "success")) == "stable"
    assert drift_status(rows("success", "failure", "failure", "failure")) == "drifted"
    assert drift_status(rows("success", "success", "failure", "failure")) == "degraded"


# ----------------------------------------------------------------------
# environment fingerprint + 3-valued match (§17, §23.08-09, §24.05)
# ----------------------------------------------------------------------


def test_environment_fingerprint_is_canonical() -> None:
    a = environment_fingerprint({"os": "Linux", "python": "3.11"})
    b = environment_fingerprint({"python": "3.11", "os": " linux "})
    assert a == b  # order, case (platform keys), whitespace normalized
    c = environment_fingerprint({"os": "linux", "python": "3.12"})
    assert a != c
    assert environment_fingerprint({}) == environment_fingerprint(None)
    with pytest.raises(VerbatimError):
        environment_fingerprint("not-a-dict")


def test_environment_match_three_valued() -> None:
    env = {"os": "linux", "python": "3.11"}
    verdict, detail = environment_match(env, {"os": "LINUX", "python": "3.11"})
    assert verdict is True and detail["mismatched"] == []

    verdict, detail = environment_match(env, {"os": "windows", "python": "3.11"})
    assert verdict is False and detail["mismatched"] == ["os"]

    verdict, detail = environment_match(env, {"python": "3.11"})
    assert verdict is None and detail["missing"] == ["os"]

    verdict, detail = environment_match({}, env)
    assert verdict is None and detail["reason"] == "no_recorded_environment"

    # no current env at all → every recorded key missing → unknown, not mismatch
    verdict, detail = environment_match(env, None)
    assert verdict is None and sorted(detail["missing"]) == ["os", "python"]


def test_match_procedure_labels_env_and_ranks(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = propose_procedure(
            store, conn, scope_id, "run the pytest suite",
            [{"description": "pytest -q"}],
            environment={"os": "linux"},
        )
        propose_procedure(
            store, conn, scope_id, "brew coffee",
            [{"description": "grind beans"}],
            environment={"os": "linux"},
        )

    res = match_procedure(store, scope_id, "how do I run pytest suite",
                          {"os": "linux"})
    assert len(res) == 1
    c = res[0]
    assert c["procedure_id"] == pid
    assert c["environment_match"] is True
    assert c["environment_error"] is None
    assert c["advisory"] is True
    assert c["outcome_stats"]["total"] == 0

    # mismatched environment → False + ENVIRONMENT_MISMATCH semantics
    res = match_procedure(store, scope_id, "pytest suite", {"os": "windows"})
    assert res[0]["environment_match"] is False
    assert res[0]["environment_error"] == ErrorCode.ENVIRONMENT_MISMATCH.value

    # missing env keys → unknown, never a wildcard pass
    res = match_procedure(store, scope_id, "pytest suite", {})
    assert res[0]["environment_match"] is None
    assert res[0]["environment_error"] == ErrorCode.APPLICABILITY_UNKNOWN.value


def test_procedures_for_scope_lane(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        pid = _propose(store, conn, scope_id)
        _propose(store, conn, scope_id, label="other task")
    assert len(procedures_for_scope(store, scope_id)) == 2
    assert len(procedures_for_scope(store, scope_id, state="proposed")) == 2
    assert procedures_for_scope(store, scope_id, state="active") == []
    with store.tx() as conn:
        activate_procedure(store, conn, pid, 1)
    assert [
        r["procedure_id"]
        for r in procedures_for_scope(store, scope_id, state="active")
    ] == [pid]


# ----------------------------------------------------------------------
# prospective records (§22)
# ----------------------------------------------------------------------


def test_plan_due_overdue_lifecycle(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        rid_future = plan(
            store, conn, scope_id, "alice", "renew certificate",
            due_us=T0 + 10 * DAY_US,
        )
        rid_past = plan(
            store, conn, scope_id, "alice", "submit report",
            due_us=T0 - DAY_US,
        )
        rid_open = plan(store, conn, scope_id, "alice", "someday maybe")

    # nothing is overdue yet from the perspective before T0 - due
    rows = due(store, scope_id, T0)
    due_ids = {r["record_id"] for r in rows}
    assert rid_past in due_ids and rid_future not in due_ids
    assert rid_open not in due_ids
    past_row = [r for r in rows if r["record_id"] == rid_past][0]
    assert past_row["status"] == "overdue"  # marked inside the same tx
    assert past_row["kind"] == "plan"

    # overdue persists; completing it is an explicit authorized update
    with store.tx() as conn:
        complete(store, conn, rid_past)
    assert plan_view(store, rid_past)["status"] == "completed"
    # completed plans are not "due" work anymore
    assert all(
        r["record_id"] != rid_past for r in due(store, scope_id, T0 + DAY_US)
    )
    # future plan becomes due (not overdue) once its time arrives
    rows = due(store, scope_id, T0 + 11 * DAY_US)
    fr = [r for r in rows if r["record_id"] == rid_future][0]
    assert fr["status"] == "overdue"


def test_plan_links_to_episode_evidence(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        eid = open_episode(store, conn, scope_id, kind="task")
        rid = plan(
            store, conn, scope_id, "alice", "finish the report",
            due_us=T0 + DAY_US, episode_id=eid,
        )
    v = plan_view(store, rid)
    assert v["episode_id"] == eid
    # a bogus episode reference fails via FK at the tx boundary, not silently
    with pytest.raises(VerbatimError) as ei:
        with store.tx() as conn:
            plan(store, conn, scope_id, "alice", "bad link",
                 episode_id="nope-episode")
    assert ei.value.code == ErrorCode.VALIDATION


def test_due_marks_multiple_records_atomically(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        r1 = plan(store, conn, scope_id, "alice", "a", due_us=T0 - DAY_US)
        r2 = plan(store, conn, scope_id, "alice", "b", due_us=T0 - 2 * DAY_US)
    rows = due(store, scope_id, T0)
    statuses = {r["record_id"]: r["status"] for r in rows}
    assert statuses == {r1: "overdue", r2: "overdue"}
    # ordering is by due time: the older record is first
    assert [r["record_id"] for r in rows] == [r2, r1]


def test_match_procedure_ranks_best_label_first(
    store: Store, scope_id: str
) -> None:
    with store.tx() as conn:
        best = propose_procedure(
            store, conn, scope_id, "run pytest suite",
            [{"description": "pytest -q"}],
        )
        propose_procedure(
            store, conn, scope_id, "run pytest linter checks",
            [{"description": "flake8"}],
        )
    res = match_procedure(store, scope_id, "run pytest suite", None)
    assert len(res) == 2
    assert res[0]["procedure_id"] == best
    assert res[0]["match_score"] >= res[1]["match_score"]
    # both have recorded environments absent → unknown, not match
    assert all(c["environment_match"] is None for c in res)


def test_plan_transitions_and_view_labels(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        rid = plan(store, conn, scope_id, "alice", "call the plumber")
    v = plan_view(store, rid)
    assert v["kind"] == "plan"
    assert v["is_fact"] is False  # a plan never claims the thing happened
    assert v["status"] == "planned"

    with store.tx() as conn:
        start(store, conn, rid)
    assert plan_view(store, rid)["status"] == "in_progress"
    with store.tx() as conn:
        cancel(store, conn, rid)
    assert plan_view(store, rid)["status"] == "cancelled"
    # terminal states reject further transitions
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            complete(store, conn, rid)
        assert ei.value.code == ErrorCode.INVALID_TRANSITION
    with pytest.raises(VerbatimError) as ei2:
        plan_view(store, "nope-record")
    assert ei2.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_plan_reschedule(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        rid = plan(store, conn, scope_id, "alice", "pay invoice",
                   due_us=T0 - DAY_US)
    due(store, scope_id, T0)  # marks overdue
    assert plan_view(store, rid)["status"] == "overdue"
    with store.tx() as conn:
        reschedule(store, conn, rid, T0 + 5 * DAY_US)
    v = plan_view(store, rid)
    assert v["status"] == "planned"
    assert v["due_us"] == T0 + 5 * DAY_US


def test_plan_recurrence_validation(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        rid = plan(
            store, conn, scope_id, "alice", "weekly sync",
            due_us=T0,
            recurrence={"freq": "weekly", "interval": 2, "anchor_us": T0},
        )
    assert plan_view(store, rid)["recurrence"]["freq"] == "weekly"
    with store.tx() as conn:
        with pytest.raises(VerbatimError):
            plan(store, conn, scope_id, "alice", "bad",
                 recurrence={"freq": "whenever"})
        with pytest.raises(VerbatimError):
            plan(store, conn, scope_id, "alice", "bad2",
                 recurrence={"freq": "daily", "interval": 0})


def test_recurrence_next() -> None:
    assert recurrence_next({"freq": "daily"}, T0) == T0 + DAY_US
    assert recurrence_next({"freq": "daily", "interval": 3}, T0) == T0 + 3 * DAY_US
    assert recurrence_next({"freq": "weekly"}, T0) == T0 + 7 * DAY_US
    # anchored: next occurrence strictly after now
    anchor = T0 - 10 * DAY_US
    # anchor + 10d lands exactly on T0 — strictly-after rolls to the next step
    nxt = recurrence_next({"freq": "daily", "anchor_us": anchor}, T0)
    assert nxt == T0 + DAY_US
    nxt = recurrence_next({"freq": "daily", "anchor_us": anchor}, T0 - DAY_US)
    assert nxt == T0
    # anchor still in the future → the anchor itself is next
    nxt = recurrence_next({"freq": "daily", "anchor_us": T0 + DAY_US}, T0)
    assert nxt == T0 + DAY_US
    # monthly: calendar semantics, Jan 31 → Feb 28/29 (clamped)
    from datetime import datetime, timezone
    jan31 = int(datetime(2025, 1, 31, tzinfo=timezone.utc).timestamp() * 1e6)
    feb28 = int(datetime(2025, 2, 28, tzinfo=timezone.utc).timestamp() * 1e6)
    assert recurrence_next(
        {"freq": "monthly", "anchor_us": jan31}, jan31
    ) == feb28
    # unknown/invalid schedules stay unresolved, never guessed
    assert recurrence_next({"freq": "every_tuesday"}, T0) is None
    assert recurrence_next({"freq": "daily", "interval": -1}, T0) is None
    assert recurrence_next("daily", T0) is None
    assert recurrence_next(None, T0) is None


def test_plans_for_scope_candidate_lane(store: Store, scope_id: str) -> None:
    with store.tx() as conn:
        plan(store, conn, scope_id, "alice", "a", due_us=T0)
        r2 = plan(store, conn, scope_id, "alice", "b", due_us=T0 + DAY_US)
    assert len(plans_for_scope(store, scope_id)) == 2
    open_rows = plans_for_scope(store, scope_id, statuses=("planned",))
    assert len(open_rows) == 2
    with store.tx() as conn:
        complete(store, conn, r2)
    done = plans_for_scope(store, scope_id, statuses=("completed",))
    assert [r["record_id"] for r in done] == [r2]
    assert done[0]["kind"] == "plan"
    assert done[0]["is_fact"] is False
