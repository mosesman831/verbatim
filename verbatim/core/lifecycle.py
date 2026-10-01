"""Claim lifecycle states and deterministic transitions (SPEC §15).

Every permitted transition lives in one deterministic table with required
effect and evidence predicates. ``LifecycleMachine.apply`` re-reads the
current revision inside the caller's write transaction, verifies the expected
version, checks the transition's requirements, and appends the new revision
plus audit event atomically — a savepoint guarantees unrecognized or failing
transitions make no partial changes (SPEC §15: "fail closed with
INVALID_TRANSITION").

Reversals append compensating events; they never delete decision history.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from ..storage import repos as _storage_repos

from .time import overlaps
from .types import (
    ErrorCode,
    Lifecycle,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
)

LIFECYCLE_POLICY_VERSION = "policy-1"

#: Actor identity required for erasure — only the purge workflow may drive a
#: claim into ``erased``; it is never an automatic effect (SPEC §15).
PURGE_ACTOR = "purge"


# ---------------------------------------------------------------------------
# Transition table (SPEC §15)
# ---------------------------------------------------------------------------
#: (from_state, effect) → {"to": target_state, "requires": (...)}. The table
#: is keyed on the *requested effect* — not just the destination — because
#: several effects legitimately share a target ("admit", "resolve",
#: "restore", and "reverse_supersede" all reach ``active``). Checking intent
#: is what keeps e.g. superseded→active reachable ONLY by reversing a
#: recorded supersession, never by a generic 'restore'.
TRANSITIONS: dict[tuple[Lifecycle, str], dict[str, Any]] = {
    (Lifecycle.PENDING, "admit"): {
        "to": Lifecycle.ACTIVE,
        "requires": ("admitted_evidence", "policy_or_confirmation"),
    },
    (Lifecycle.ACTIVE, "dispute"): {
        "to": Lifecycle.DISPUTED,
        "requires": ("conflict_edge",),
    },
    (Lifecycle.ACTIVE, "supersede"): {
        "to": Lifecycle.SUPERSEDED,
        "requires": ("identified_successor", "validated_update"),
    },
    (Lifecycle.DISPUTED, "resolve"): {
        "to": Lifecycle.ACTIVE,
        "requires": ("resolution_event",),
    },
    (Lifecycle.ACTIVE, "archive"): {"to": Lifecycle.ARCHIVED, "requires": ()},
    (Lifecycle.DISPUTED, "archive"): {"to": Lifecycle.ARCHIVED, "requires": ()},
    (Lifecycle.SUPERSEDED, "archive"): {"to": Lifecycle.ARCHIVED, "requires": ()},
    (Lifecycle.ARCHIVED, "restore"): {
        "to": Lifecycle.ACTIVE,
        "requires": ("revalidation",),
    },
    (Lifecycle.PENDING, "reject"): {"to": Lifecycle.REJECTED, "requires": ()},
    (Lifecycle.DISPUTED, "reject"): {"to": Lifecycle.REJECTED, "requires": ()},
    (Lifecycle.SUPERSEDED, "reverse_supersede"): {
        "to": Lifecycle.ACTIVE,
        "requires": ("prior_supersession",),
    },
    # Erasure is reachable from every non-erased state, but only through the
    # purge workflow actor — verified again in apply().
    (Lifecycle.PENDING, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
    (Lifecycle.ACTIVE, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
    (Lifecycle.DISPUTED, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
    (Lifecycle.SUPERSEDED, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
    (Lifecycle.REJECTED, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
    (Lifecycle.ARCHIVED, "erase"): {
        "to": Lifecycle.ERASED,
        "requires": ("purge_actor",),
    },
}


def can_transition(from_state: Lifecycle, effect: str) -> bool:
    """Pure check for tests/UI: is ``effect`` permitted from ``from_state``?"""
    if not isinstance(from_state, Lifecycle):
        try:
            from_state = Lifecycle(from_state)
        except ValueError:
            return False
    return (from_state, effect) in TRANSITIONS


# ---------------------------------------------------------------------------
# Read helpers (scoped, parameterized; shared with policy.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimHead:
    """The claim row joined to its latest revision — the "current version"."""

    claim_id: str
    scope_id: str
    subject_id: Optional[str]
    predicate: Optional[str]
    row_version: int
    revision: int
    state: Lifecycle
    object_json: Optional[str]
    polarity: str
    modality: str
    condition_json: Optional[str]
    interpretation_json: Optional[str]
    recorded_from: int
    recorded_until: Optional[int]
    # v2 revision metadata (SPEC_V2 §13, §16) — carried forward verbatim on
    # every transition so interpretation lineage is never silently dropped.
    interpretation_status: str = "structured"
    registry_version: Optional[int] = None
    method: Optional[str] = None
    rev_subject_id: Optional[str] = None
    rev_predicate: Optional[str] = None


def read_claim_head(conn: sqlite3.Connection, claim_id: str) -> Optional[ClaimHead]:
    """Latest claim revision by revision number, regardless of recorded_until.

    Returns ``None`` for missing claims — callers raise
    ``NOT_FOUND_OR_FORBIDDEN`` so unauthorized callers cannot distinguish a
    missing id from a forbidden one (SPEC §9).
    """
    row = conn.execute(
        "SELECT c.claim_id, c.scope_id, c.subject_id, c.predicate, c.row_version,"
        " r.revision, r.state, r.object_json, r.polarity, r.modality,"
        " r.condition_json, r.interpretation_json, r.recorded_from, r.recorded_until,"
        " r.interpretation_status, r.registry_version, r.method,"
        " r.subject_id, r.predicate"
        " FROM claims c JOIN claim_revisions r ON r.claim_id = c.claim_id"
        " WHERE c.claim_id = ? ORDER BY r.revision DESC LIMIT 1",
        (claim_id,),
    ).fetchone()
    if row is None:
        return None
    return ClaimHead(
        claim_id=row[0],
        scope_id=row[1],
        subject_id=row[2],
        predicate=row[3],
        row_version=row[4],
        revision=row[5],
        state=Lifecycle(row[6]),
        object_json=row[7],
        polarity=row[8],
        modality=row[9],
        condition_json=row[10],
        interpretation_json=row[11],
        recorded_from=row[12],
        recorded_until=row[13],
        interpretation_status=row[14],
        registry_version=row[15],
        method=row[16],
        rev_subject_id=row[17],
        rev_predicate=row[18],
    )


def read_intervals(
    conn: sqlite3.Connection, claim_id: str, revision: int
) -> list[TimeInterval]:
    """Applicability intervals with full v2 endpoint metadata (SPEC_V2 §16)."""
    rows = conn.execute(
        "SELECT from_us, until_us, precision, timezone, basis,"
        " start_kind, end_kind, from_us_hi, until_us_hi FROM valid_intervals"
        " WHERE claim_id = ? AND revision = ? ORDER BY interval_no",
        (claim_id, revision),
    ).fetchall()
    return [
        TimeInterval(
            from_us=r[0],
            until_us=r[1],
            precision=r[2],
            timezone=r[3],
            basis=r[4],
            start_kind=r[5],
            end_kind=r[6],
            from_us_hi=r[7],
            until_us_hi=r[8],
        )
        for r in rows
    ]


def read_evidence(
    conn: sqlite3.Connection, claim_id: str, revision: int
) -> list[tuple[str, str]]:
    """(span_id, evidence_role) pairs attached to a claim revision."""
    rows = conn.execute(
        "SELECT span_id, evidence_role FROM claim_evidence"
        " WHERE claim_id = ? AND revision = ?",
        (claim_id, revision),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _conflict_edge_exists(conn: sqlite3.Connection, claim_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM edges WHERE edge_type = 'conflicts_with'"
            " AND retired_event IS NULL AND source_kind = 'claim'"
            " AND target_kind = 'claim'"
            " AND (source_id = ? OR target_id = ?) LIMIT 1",
            (claim_id, claim_id),
        ).fetchone()
        is not None
    )


def _supersession_path_exists(
    conn: sqlite3.Connection, from_id: str, to_id: str
) -> bool:
    """BFS over active ``supersedes`` edges (source → target = "supersedes").

    Adding a ``successor → predecessor`` edge creates a cycle iff the
    predecessor already transitively supersedes the successor (SPEC §16:
    cycles in supersession are rejected).
    """
    seen = {from_id}
    frontier = [from_id]
    while frontier:
        node = frontier.pop()
        if node == to_id:
            return True
        rows = conn.execute(
            "SELECT target_id FROM edges WHERE edge_type = 'supersedes'"
            " AND retired_event IS NULL AND source_kind = 'claim'"
            " AND target_kind = 'claim' AND source_id = ?",
            (node,),
        ).fetchall()
        for (nxt,) in rows:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
        if len(seen) > 10_000:  # defensive bound against pathological graphs
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                "supersession graph exceeds traversal bound",
            )
    return False


def _truncate_intervals(
    intervals: list[TimeInterval], cut: Optional[TimeInterval]
) -> list[TimeInterval]:
    """Close only the applicability intervals justified by ``cut`` (SPEC §17).

    An interval overlapping (or indistinguishable from) the supersession
    interval is closed at ``cut.from_us``; intervals wholly covered by the
    new claim's stated interval are dropped; non-overlapping intervals are
    untouched. ``cut=None`` or an unbounded cut leaves history as recorded —
    unknown effective dates must not fabricate validity bounds.
    """
    if cut is None or cut.from_us is None:
        return list(intervals)
    out: list[TimeInterval] = []
    for iv in intervals:
        ov = overlaps(iv, cut)
        if ov is False:
            out.append(iv)
            continue
        if iv.from_us is not None and iv.from_us >= cut.from_us:
            continue  # fully covered by the superseding interval
        if iv.until_us is not None and iv.until_us < cut.from_us:
            until, until_hi = iv.until_us, iv.until_us_hi
            end_kind = iv.end_kind
        else:
            # The new end is the supersession's start — inherit its endpoint
            # kind so an uncertain successor start leaves an uncertain close
            # rather than fabricating exactness (SPEC_V2 §16).
            until, until_hi = cut.from_us, cut.from_us_hi
            end_kind = cut.start_kind
        try:
            out.append(
                TimeInterval(
                    iv.from_us,
                    until,
                    iv.precision,
                    iv.timezone,
                    iv.basis,
                    start_kind=iv.start_kind,
                    end_kind=end_kind,
                    from_us_hi=iv.from_us_hi,
                    until_us_hi=until_hi,
                )
            )
        except VerbatimError:
            continue  # would be empty/inverted — drop rather than persist
    return out


def _default_repos(store: Any) -> SimpleNamespace:
    """Repository facade over ``store`` (assumed storage API).

    Isolated here so a drifted storage API is adapted in exactly one place —
    never edit call sites spread through the machine.
    """
    return SimpleNamespace(
        claims=_storage_repos.ClaimsRepo(store),
        events=_storage_repos.EventsRepo(store),
        edges=_storage_repos.EdgesRepo(store),
    )


class LifecycleMachine:
    """Applies authorized lifecycle mutations inside the caller's transaction.

    The machine owns no transaction itself: ``apply`` runs entirely on the
    supplied connection so evidence insertion, transition events, edges, and
    projection changes commit together (SPEC §17). A savepoint still guards
    atomicity if the caller catches an error mid-transaction.
    """

    def __init__(
        self,
        store: Any,
        repos: Optional[SimpleNamespace] = None,
        policy_version: str = LIFECYCLE_POLICY_VERSION,
    ) -> None:
        self._repos = repos if repos is not None else _default_repos(store)
        self._policy_version = policy_version

    def apply(self, cmd: TransitionCommand, conn: sqlite3.Connection) -> int:
        """Apply ``cmd``; return the appended event sequence number.

        Raises ``NOT_FOUND_OR_FORBIDDEN`` for unknown claims/successors,
        ``STALE_PROPOSAL`` when ``expected_revision`` no longer matches the
        current revision, and ``INVALID_TRANSITION`` for any transition the
        table or its requirement predicates reject.
        """
        head = read_claim_head(conn, cmd.claim_id)
        if head is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found"
            )
        if head.revision != cmd.expected_revision:
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL,
                f"expected revision {cmd.expected_revision}, current {head.revision}",
            )
        entry = TRANSITIONS.get((head.state, cmd.effect))
        if entry is None:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"effect {cmd.effect!r} not permitted from {head.state.value}",
            )
        target = entry["to"]
        if not cmd.actor_id or not cmd.reason:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                "transitions require an authenticated actor and a reason",
            )

        evidence = read_evidence(conn, cmd.claim_id, head.revision)
        intervals = read_intervals(conn, cmd.claim_id, head.revision)
        new_intervals = list(intervals)
        new_evidence = list(evidence)

        # --- requirement predicates per effect -----------------------------
        if cmd.effect == "admit":
            if not evidence:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "admit requires admitted evidence on the pending revision",
                )
        elif cmd.effect == "dispute":
            if not _conflict_edge_exists(conn, cmd.claim_id):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "dispute requires an existing conflict edge",
                )
        elif cmd.effect == "supersede":
            if not cmd.successor_claim_id:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "supersede requires an identified successor claim",
                )
            succ = read_claim_head(conn, cmd.successor_claim_id)
            if succ is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "successor claim not found"
                )
            if succ.claim_id == cmd.claim_id or succ.scope_id != head.scope_id:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "successor must be a different claim in the same scope",
                )
            if succ.state in (Lifecycle.ERASED, Lifecycle.REJECTED, Lifecycle.SUPERSEDED):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    f"successor in state {succ.state.value} cannot supersede",
                )
            if _supersession_path_exists(conn, cmd.claim_id, cmd.successor_claim_id):
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "supersession would create a cycle",
                )
            new_intervals = _truncate_intervals(intervals, cmd.interval)
        elif cmd.effect == "restore":
            # Revalidation: the evidence spans must still exist — a purged
            # span cannot be reactivated through a stale archive.
            for span_id, _role in evidence:
                if (
                    conn.execute(
                        "SELECT 1 FROM spans WHERE span_id = ?", (span_id,)
                    ).fetchone()
                    is None
                ):
                    raise VerbatimError(
                        ErrorCode.EVIDENCE_UNAVAILABLE,
                        "restore requires surviving evidence spans",
                    )
            if not evidence:
                raise VerbatimError(
                    ErrorCode.EVIDENCE_UNAVAILABLE,
                    "restore requires surviving evidence spans",
                )
        elif cmd.effect == "reverse_supersede":
            if head.revision < 2:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "no prior interpretation to reopen",
                )
            # Reopen only the intervals closed by the recorded supersession:
            # restore exactly the prior revision's applicability — nothing
            # that later events closed independently is resurrected.
            new_intervals = read_intervals(conn, cmd.claim_id, head.revision - 1)
        elif cmd.effect == "erase":
            if cmd.actor_id != PURGE_ACTOR:
                raise VerbatimError(
                    ErrorCode.INVALID_TRANSITION,
                    "erase is restricted to the purge workflow actor",
                )
            new_intervals = []
            new_evidence = []

        # --- atomic write phase ---------------------------------------------
        payload = {
            "claim_id": cmd.claim_id,
            "effect": cmd.effect,
            "from_state": head.state.value,
            "to_state": target.value,
            "reason": cmd.reason,
            "successor_claim_id": cmd.successor_claim_id,
            "interval": _interval_payload(cmd.interval),
        }
        conn.execute("SAVEPOINT verbatim_apply")
        try:
            seq = self._repos.events.append(
                conn,
                head.scope_id,
                "claim_transition",
                cmd.actor_id,
                payload,
                self._policy_version,
            )
            self._repos.claims.set_recorded_until(
                cmd.claim_id, head.revision, seq, conn
            )
            if cmd.effect == "supersede":
                self._repos.edges.add(
                    conn,
                    head.scope_id,
                    "claim",
                    cmd.successor_claim_id,
                    "claim",
                    cmd.claim_id,
                    "supersedes",
                )
            if cmd.effect == "reverse_supersede":
                # Compensating retirement of the supersedes edge(s) created by
                # the supersession event being reversed — history is retained,
                # never deleted.
                if cmd.successor_claim_id:
                    conn.execute(
                        "UPDATE edges SET retired_event = ? WHERE edge_type = 'supersedes'"
                        " AND retired_event IS NULL AND source_kind = 'claim'"
                        " AND target_kind = 'claim' AND source_id = ? AND target_id = ?",
                        (seq, cmd.successor_claim_id, cmd.claim_id),
                    )
                else:
                    conn.execute(
                        "UPDATE edges SET retired_event = ? WHERE edge_type = 'supersedes'"
                        " AND retired_event IS NULL AND target_kind = 'claim'"
                        " AND target_id = ? AND created_event = ?",
                        (seq, cmd.claim_id, head.recorded_from),
                    )
            if cmd.effect == "erase":
                new_rev = self._repos.claims.add_revision(
                    cmd.claim_id,
                    Lifecycle.ERASED.value,
                    None,
                    "affirmative",
                    "asserted",
                    None,
                    None,
                    [],
                    [],
                    seq,
                    conn,
                )
                _stamp_revision_v2(conn, cmd.claim_id, new_rev, head, [])
            else:
                new_rev = self._repos.claims.add_revision(
                    cmd.claim_id,
                    target.value,
                    head.object_json,
                    head.polarity,
                    head.modality,
                    head.condition_json,
                    head.interpretation_json,
                    new_intervals,
                    new_evidence,
                    seq,
                    conn,
                )
                _stamp_revision_v2(conn, cmd.claim_id, new_rev, head, new_intervals)
            _resolve_conflict_groups(
                conn, cmd.claim_id, cmd.successor_claim_id, seq
            )
            conn.execute("RELEASE verbatim_apply")
        except BaseException:
            conn.execute("ROLLBACK TO verbatim_apply")
            conn.execute("RELEASE verbatim_apply")
            raise
        return seq


def _resolve_conflict_groups(
    conn: sqlite3.Connection,
    claim_id: str,
    successor_claim_id: Optional[str],
    seq: int,
) -> None:
    """Close open conflict groups this transition settles (V2-17).

    A group resolves when either:
    - the transition's successor is itself a group member (the operator
      picked the winner — e.g. supersede old→new), or
    - at most one member still asserts truth (pending/active/disputed) —
      elimination leaves no live conflict.

    Members that left standing states for unrelated reasons don't resolve
    anything on their own: a group of three with two fallen still has two
    contenders? No — one standing means no conflict remains, so it closes.
    """
    groups = conn.execute(
        "SELECT cm.group_id FROM conflict_members cm"
        " JOIN conflict_groups g ON g.group_id = cm.group_id"
        " WHERE cm.claim_id = ? AND g.status = 'open'",
        (claim_id,),
    ).fetchall()
    for (gid,) in groups:
        resolved = False
        if successor_claim_id:
            in_group = conn.execute(
                "SELECT 1 FROM conflict_members WHERE group_id = ?"
                " AND claim_id = ?",
                (gid, successor_claim_id),
            ).fetchone()
            resolved = in_group is not None
        if not resolved:
            standing = conn.execute(
                "SELECT COUNT(*) FROM conflict_members cm"
                " JOIN claim_revisions cr ON cr.claim_id = cm.claim_id"
                " AND cr.revision = (SELECT MAX(revision) FROM claim_revisions"
                "                  WHERE claim_id = cm.claim_id)"
                " WHERE cm.group_id = ?"
                " AND cr.state IN ('pending','active','disputed')",
                (gid,),
            ).fetchone()[0]
            resolved = standing <= 1
        if resolved:
            conn.execute(
                "UPDATE conflict_groups SET status = 'resolved',"
                " resolution_event = ? WHERE group_id = ?",
                (seq, gid),
            )


def _stamp_revision_v2(
    conn: sqlite3.Connection,
    claim_id: str,
    revision: int,
    head: ClaimHead,
    intervals: list[TimeInterval],
) -> None:
    """Fill v2 columns ``ClaimsRepo.add_revision`` does not yet write.

    The storage API is stable, so the machine patches the just-appended
    revision in the same transaction: revision metadata is carried forward
    from ``head`` (SPEC_V2 §13, §16 — a transition must not silently reset
    interpretation lineage) and interval endpoint kinds/bounds come from the
    persisted ``TimeInterval`` objects.
    """
    conn.execute(
        "UPDATE claim_revisions SET subject_id = ?, predicate = ?,"
        " registry_version = ?, interpretation_status = ?, method = ?"
        " WHERE claim_id = ? AND revision = ?",
        (
            head.rev_subject_id if head.rev_subject_id is not None else head.subject_id,
            head.rev_predicate if head.rev_predicate is not None else head.predicate,
            head.registry_version,
            head.interpretation_status,
            head.method,
            claim_id,
            revision,
        ),
    )
    for interval_no, iv in enumerate(intervals):
        conn.execute(
            "UPDATE valid_intervals SET start_kind = ?, end_kind = ?,"
            " from_us_hi = ?, until_us_hi = ?"
            " WHERE claim_id = ? AND revision = ? AND interval_no = ?",
            (
                iv.start_kind.value,
                iv.end_kind.value,
                iv.from_us_hi,
                iv.until_us_hi,
                claim_id,
                revision,
                interval_no,
            ),
        )


def _interval_payload(iv: Optional[TimeInterval]) -> Optional[dict[str, Any]]:
    if iv is None:
        return None
    return {
        "from_us": iv.from_us,
        "until_us": iv.until_us,
        "precision": iv.precision.value,
        "timezone": iv.timezone,
        "basis": iv.basis,
        "start_kind": iv.start_kind.value,
        "end_kind": iv.end_kind.value,
        "from_us_hi": iv.from_us_hi,
        "until_us_hi": iv.until_us_hi,
    }
