"""V3 episode building from completed trajectories (SPEC_V3 §20).

An episode groups one *completed* trajectory (``completed_event`` set)
into an ``episodes`` row plus ``episode_members`` for the member
envelopes and ``derivations`` edges recording exactly what it was built
from. Requirements pinned here:

- V3-20.01: every episode records its declared ``boundary_rule`` — the
  trajectory's own, never inferred silently.
- V3-20.08: production runs as durable jobs with idempotent keys;
  replaying a trajectory yields identical episode identity. The
  ``episode_id`` is a pure function of ``trajectory_id`` and rebuilds are
  deduped on it (and on the ``derivations`` parent edge), so a rebuild
  converges instead of duplicating.
- V3-20.09: failed and abandoned episodes are retained as negative
  experience — ``outcome='failure'`` episodes are built and kept exactly
  like successful ones.
- V3-12.03 / V3-13.05: ``episode.outcome`` resolves only from
  ``test_result``/``verification`` envelopes carrying an *identified*
  checker receipt. An agent's "it worked" claim is never an outcome;
  missing outcomes record ``unknown``, never inferred success.
- V3-17.02: derivations edges record parent revisions and producer
  identity; they are immutable, so rebuilds ``INSERT OR IGNORE`` the same
  edge set rather than rewriting it.

Membership writes use ``INSERT OR IGNORE`` so rebuilding never churns a
member's ``recorded_from`` — membership history stays stable
(V3-20.06).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v3 import OutcomeClass
from ..storage import repos_v3
from ..storage.repos import _next_event_seq
from . import repo as _repo

# Producer identity recorded on every derivations edge this module writes
# (V3-17.02: producer kind + revisioned identity).
EPISODE_PRODUCER_KIND = "producer"
EPISODE_PRODUCER_ID = "v3.episode_build.v1"

# Envelope kinds that may carry checker-attested outcome evidence
# (§12.03). Only these can move an episode outcome away from unknown.
OUTCOME_ENVELOPE_KINDS = ("test_result", "verification")

_OUTCOME_VALUES = frozenset(o.value for o in OutcomeClass)


def episode_id_for(trajectory_id: str) -> str:
    """Deterministic episode identity for one trajectory (V3-20.08)."""
    require_id(trajectory_id, "trajectory_id")
    digest = hashlib.sha256(
        f"v3:episode\x00{trajectory_id}".encode("utf-8")
    ).hexdigest()
    return f"ep_{digest[:32]}"


# ---------------------------------------------------------------------------
# checker receipts + outcome resolution (§12.03, §22.16)
# ---------------------------------------------------------------------------


def checker_receipt(metadata: Any) -> Optional[dict[str, Any]]:
    """The *identified* checker receipt inside envelope metadata, or None.

    A receipt counts only when it names its checker (``checker_id``) —
    "an identified checker" is the contract (V3-12.03); an anonymous
    pass/fail claim is not outcome evidence. The SDK descriptor writes
    ``checker_receipt``/``checker_receipts`` (``sdk.envelope
    .outcome_descriptor``); a bare ``checker`` key is accepted too. The
    payload mirrors ``types_v3.CheckerReceipt`` fields.
    """
    if not isinstance(metadata, dict):
        return None
    candidates: list[Any] = [
        metadata.get("checker_receipt"),
        metadata.get("checker"),
    ]
    extra = metadata.get("checker_receipts")
    if isinstance(extra, (list, tuple)):
        candidates.extend(extra)
    for checker in candidates:
        if not isinstance(checker, dict):
            continue
        checker_id = checker.get("checker_id")
        if isinstance(checker_id, str) and checker_id:
            return checker
    return None


def envelope_outcome(
    metadata: Any,
    *,
    task_id: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> Optional[OutcomeClass]:
    """Outcome attested by one envelope's host-executed checker receipt.

    Returns ``None`` — *not outcome evidence at all* — unless the
    receipt is host-attested: ``host_attested`` truthy, a non-empty
    ``invocation_id`` binding a concrete host-side execution, and no
    ``agent_report`` marker (V4-23.02, C19). A caller/model-supplied
    checker name or exit code is an agent report; it can never move an
    outcome off ``unknown`` through this path.

    Binding (V4-23.01, C20): when the receipt pins ``task_id`` or
    ``scope_id`` they must equal the values under resolution — a receipt
    for another task or scope cannot certify this trajectory. Unbound
    fields stay acceptable; ``None`` expectations bind to "".

    With an attested receipt, a declared ``metadata['outcome']`` wins;
    otherwise the receipt's ``completed``/``exit_code`` derive
    success/failure and anything unverifiable stays ``unknown``
    (V3-12.03, V3-42.08 — a claim without artifact support cannot
    upgrade).
    """
    checker = checker_receipt(metadata)
    if checker is None:
        return None
    # Agent reports are evidence, not attestation — an explicit
    # ``agent_report`` marker vetoes even if ``host_attested`` is also
    # (incorrectly) asserted.
    if checker.get("agent_report"):
        return None
    if not checker.get("host_attested"):
        return None
    if not checker.get("invocation_id"):
        return None
    for field, expected in (("task_id", task_id), ("scope_id", scope_id)):
        bound = checker.get(field)
        if isinstance(bound, str) and bound and bound != (expected or ""):
            return None
    declared = metadata.get("outcome") if isinstance(metadata, dict) else None
    if isinstance(declared, str) and declared in _OUTCOME_VALUES:
        return OutcomeClass(declared)
    if checker.get("completed") is False:
        return OutcomeClass.UNKNOWN
    exit_code = checker.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return OutcomeClass.SUCCESS if exit_code == 0 else OutcomeClass.FAILURE
    return OutcomeClass.UNKNOWN


# ---------------------------------------------------------------------------
# trajectory → member envelopes
# ---------------------------------------------------------------------------


def _observation_ids(conn: sqlite3.Connection, step_id: str) -> list[str]:
    rows = repos_v3.query(
        conn, "step_observations", {"step_id": step_id}, order="ord"
    )
    return [r["envelope_id"] for r in rows]


def trajectory_steps(
    conn: sqlite3.Connection, trajectory_id: str
) -> list[dict[str, Any]]:
    """Ordered steps of one trajectory (ord is the replay order, §12.02)."""
    return repos_v3.query(
        conn, "trajectory_steps", {"trajectory_id": trajectory_id}, order="ord"
    )


def member_envelopes(
    conn: sqlite3.Connection, steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The trajectory's envelopes in deterministic member order.

    Per step (ord ascending): the action envelope first, then the step's
    observation envelopes by their recorded ord. Each entry is a dict
    with ``envelope_id``, ``kind``, ``revision``, ``metadata``, and
    ``found`` — a referenced envelope missing from ``source_envelopes``
    is still recorded as a member (honest provenance: the trajectory
    referenced it) with ``found=False`` and revision 1.
    """
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        ids: list[str] = []
        action = step.get("action_envelope_id")
        if action:
            ids.append(action)
        ids.extend(_observation_ids(conn, step["step_id"]))
        for eid in ids:
            if eid in seen:
                continue
            seen.add(eid)
            row = repos_v3.get(conn, "source_envelopes", {"envelope_id": eid})
            metadata = (
                repos_v3.json_field(row, "metadata_json", {}) if row else {}
            ) or {}
            members.append(
                {
                    "envelope_id": eid,
                    "kind": row["envelope_kind"] if row else None,
                    "revision": int(row["revision"]) if row else 1,
                    "metadata": metadata,
                    "found": row is not None,
                    "step_ord": int(step["ord"]),
                }
            )
    return members


def resolve_outcome(
    members: list[dict[str, Any]],
    *,
    task_id: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> OutcomeClass:
    """Episode outcome from host-attested outcome envelopes only.

    The *last* outcome envelope in member order wins — a trajectory's
    final checker verdict is the task outcome (V3-13.05). No
    host-attested, correctly-bound receipt → ``unknown``; an agent
    claiming success or a receipt bound to another task/scope moves
    nothing (V4-23.01/23.02, C19/C20).
    """
    outcome = OutcomeClass.UNKNOWN
    for env in members:
        if env["kind"] not in OUTCOME_ENVELOPE_KINDS:
            continue
        resolved = envelope_outcome(
            env["metadata"], task_id=task_id, scope_id=scope_id
        )
        if resolved is not None:
            outcome = resolved
    return outcome


def episode_for_trajectory(
    conn: sqlite3.Connection, trajectory_id: str
) -> Optional[str]:
    """The episode already built for this trajectory, if any.

    Lookup path: the ``derivations`` edge this producer wrote
    (episode ← trajectory), then the deterministic id itself — both are
    rebuilt identically on replay. The edge lookup is producer-scoped so
    a foreign episode builder's edge can never redirect this pipeline
    into someone else's episode row.
    """
    require_id(trajectory_id, "trajectory_id")
    row = _row_first(
        conn,
        "SELECT child_id FROM derivations"
        " WHERE child_kind = 'episode' AND parent_kind = 'trajectory'"
        "   AND parent_id = ? AND producer_id = ? ORDER BY child_id",
        (trajectory_id, EPISODE_PRODUCER_ID),
    )
    if row is not None:
        return str(row[0])
    eid = episode_id_for(trajectory_id)
    if _repo.episode_row(conn, eid) is not None:
        return eid
    return None


def _row_first(conn: sqlite3.Connection, sql: str, params: tuple) -> Any:
    return conn.execute(sql, params).fetchone()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _insert_member(
    conn: sqlite3.Connection, episode_id: str, envelope_id: str, ord_: int
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO episode_members"
        " (episode_id, object_kind, object_id, ord, recorded_from)"
        " VALUES (?, 'envelope', ?, ?, ?)",
        (episode_id, envelope_id, ord_, _next_event_seq(conn)),
    )


def _insert_derivation(
    conn: sqlite3.Connection,
    *,
    child_kind: str,
    child_id: str,
    parent_kind: str,
    parent_id: str,
    parent_revision: int,
    seq: int,
    scope_id: str,
    producer_kind: str,
    producer_id: str,
    child_revision: int = 1,
) -> None:
    """One immutable derivations edge; PK dedup makes rebuilds replay the
    identical edge set (V3-17.02, V3-20.08).

    The ``'envelope'`` parent kind matches ``derivations.EVIDENCE_KINDS``
    so purge/impact traversals resolve these edges. An exact replay is a
    no-op; an identical edge claimed by a *different* producer is an
    INTEGRITY failure — edges are append-only, never silently rewritten.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO derivations"
        " (child_kind, child_id, child_revision, parent_kind, parent_id,"
        "  parent_revision, producer_kind, producer_id, seq, scope_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            child_kind,
            child_id,
            child_revision,
            parent_kind,
            parent_id,
            parent_revision,
            producer_kind,
            producer_id,
            seq,
            scope_id,
        ),
    )
    if cur.rowcount == 0:
        existing = _row_first(
            conn,
            "SELECT producer_kind, producer_id FROM derivations"
            " WHERE child_kind = ? AND child_id = ? AND child_revision = ?"
            "   AND parent_kind = ? AND parent_id = ? AND parent_revision = ?",
            (
                child_kind,
                child_id,
                child_revision,
                parent_kind,
                parent_id,
                parent_revision,
            ),
        )
        if existing is not None and (
            existing[0] != producer_kind or existing[1] != producer_id
        ):
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                "derivation edge exists under a different producer — "
                "edges are immutable (V3-17.02)",
            )


def build_episode(conn: sqlite3.Connection, trajectory_id: str) -> str:
    """Group one completed trajectory into an episode; returns episode_id.

    Refuses when the trajectory is absent (``NOT_FOUND_OR_UNAUTHORIZED``)
    or still open (``VALIDATION`` — incomplete evidence stays evidence;
    an unfinished trajectory is not an episode boundary).

    Idempotent (V3-20.08): the episode id derives deterministically from
    ``trajectory_id`` and a ``derivations`` edge records the parentage,
    so rebuilding the same trajectory returns the same episode. Members
    and edges are ``INSERT OR IGNORE``; a later checker outcome heals the
    derived ``outcome``/``environment_digest`` forward in place.
    """
    require_id(trajectory_id, "trajectory_id")
    traj = repos_v3.get(conn, "trajectories", {"trajectory_id": trajectory_id})
    if traj is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "trajectory not found"
        )
    if traj["completed_event"] is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "trajectory has no completed_event — incomplete evidence "
            "stays evidence (§20.01)",
        )

    steps = trajectory_steps(conn, trajectory_id)
    members = member_envelopes(conn, steps)
    # Receipts must bind THIS trajectory's task/scope — a receipt for
    # another task, scope, or artifact cannot certify it (V4-23.01, C20).
    outcome = resolve_outcome(
        members,
        task_id=traj["task_id"],
        scope_id=traj["scope_id"],
    )
    metadata = repos_v3.json_field(traj, "metadata_json", {}) or {}
    label = metadata.get("label") or traj["task_id"] or None
    env_digest = traj.get("environment_digest")

    episode_id = episode_for_trajectory(conn, trajectory_id) or episode_id_for(
        trajectory_id
    )
    existing = _repo.episode_row(conn, episode_id)
    if existing is None:
        conn.execute(
            "INSERT INTO episodes"
            " (episode_id, scope_id, revision, host_task_id,"
            "  host_session_id, kind, label, recorded_from,"
            "  boundary_rule, outcome, environment_digest)"
            " VALUES (?, ?, 1, ?, ?, 'task', ?, ?, ?, ?, ?)",
            (
                episode_id,
                traj["scope_id"],
                traj["task_id"] or None,
                traj["session_id"] or None,
                label,
                _next_event_seq(conn),
                traj["boundary_rule"],
                outcome.value,
                env_digest,
            ),
        )
    else:
        # Rebuild heals derived fields forward: late checker evidence may
        # legitimately refine outcome; identity and revision stay put.
        if (
            existing["outcome"] != outcome.value
            or existing.get("environment_digest") != env_digest
            or existing.get("boundary_rule") != traj["boundary_rule"]
        ):
            conn.execute(
                "UPDATE episodes SET outcome = ?, environment_digest = ?,"
                " boundary_rule = ?, row_version = row_version + 1"
                " WHERE episode_id = ?",
                (outcome.value, env_digest, traj["boundary_rule"], episode_id),
            )

    for ord_, env in enumerate(members):
        _insert_member(conn, episode_id, env["envelope_id"], ord_)

    _insert_derivation(
        conn,
        child_kind="episode",
        child_id=episode_id,
        parent_kind="trajectory",
        parent_id=trajectory_id,
        parent_revision=1,
        seq=0,
        scope_id=traj["scope_id"],
        producer_kind=EPISODE_PRODUCER_KIND,
        producer_id=EPISODE_PRODUCER_ID,
    )
    for seq, env in enumerate(members, start=1):
        _insert_derivation(
            conn,
            child_kind="episode",
            child_id=episode_id,
            parent_kind="envelope",
            parent_id=env["envelope_id"],
            parent_revision=env["revision"],
            seq=seq,
            scope_id=traj["scope_id"],
            producer_kind=EPISODE_PRODUCER_KIND,
            producer_id=EPISODE_PRODUCER_ID,
        )
    return episode_id
