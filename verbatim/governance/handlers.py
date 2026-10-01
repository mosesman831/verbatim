"""Job handlers for governance pipeline kinds (SPEC_V3 §40).

``handle_revocation_notify`` applies a revocation inside the store:
``input_refs`` carries exactly one of ``{"delegation_id", "grant_id"}``;
the handler marks the grant (or delegation edge) revoked at a bumped
authorization epoch, cascades to dependent child grants, fences the
recipient's propagated copies in the ledger (§10.05), closes their open
handoff capsules, and appends a safe-identifier audit event (§09.11).
Effects are naturally idempotent: a redelivered job re-revokes nothing —
the second pass reports ``already_revoked`` and leaves state unchanged.
Job-level dedup happens at enqueue through ``dedup_key``.
"""

from __future__ import annotations

from typing import Any

from ..core.types import ErrorCode, VerbatimError, require_id
from ..storage.repos import EventsRepo
from .propagation import mark_propagations_revoked
from .revocation import revoke_delegation, revoke_grant


def handle_revocation_notify(job: dict, owner: str, ingester) -> None:
    """Apply one revocation notice (kind ``revocation_notify``, §10.05).

    ``job`` is the leased queue row dict; ``ingester`` supplies the store
    (``tx()``/``read()``) and, when present, the job queue for lease
    fencing. Everything commits in one transaction — revocation, epoch
    bump, cascade, ledger marking, capsule closure, and the audit event.
    """
    refs = job.get("input_refs") or {}
    grant_id = refs.get("grant_id")
    delegation_id = refs.get("delegation_id")
    if (grant_id is None) == (delegation_id is None):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "revocation_notify needs exactly one of"
            " input_refs.grant_id / input_refs.delegation_id",
        )
    store = ingester.store
    with store.tx() as conn:
        jobs = getattr(ingester, "jobs", None)
        if (
            jobs is not None
            and job.get("job_id") is not None
            and job.get("generation") is not None
        ):
            jobs.assert_lease(conn, job["job_id"], owner, job["generation"])
        if grant_id is not None:
            require_id(grant_id, "grant_id")
            res = revoke_grant(conn, grant_id)
            recipient = res["principal_id"]
            scope_id = res["scope_id"]
            epoch = res["epoch"]
            target = {"grant_id": grant_id}
        else:
            require_id(delegation_id, "delegation_id")
            res = revoke_delegation(conn, delegation_id)
            recipient = res["delegate_id"]
            scope_id = res["scope_id"]
            epoch = res["epoch"]
            target = {"delegation_id": delegation_id}
        props = 0
        capsules = 0
        if scope_id is not None and recipient is not None:
            # Fleet revocation (§10.05): copies already propagated to the
            # revoked principal are fenced in the ledger at the new epoch —
            # delivered bytes are marked, never claimed to be recalled.
            props = mark_propagations_revoked(
                conn, seq=epoch, scope_id=scope_id, recipient_id=recipient
            )
            capsules = conn.execute(
                "UPDATE handoff_capsules SET status = 'revoked'"
                " WHERE scope_id = ? AND recipient_id = ?"
                " AND status = 'open'",
                (scope_id, recipient),
            ).rowcount
        EventsRepo(store).append(
            conn,
            scope_id or "scope:none",
            "revocation_applied",
            "governance",
            {
                **target,
                "recipient_id": recipient,
                "epoch": epoch,
                "grants_revoked": res.get("grants_revoked", 0),
                "delegations_revoked": res.get("delegations_revoked", 0),
                "propagations_fenced": props,
                "capsules_revoked": capsules,
                "already_revoked": res.get("already_revoked", False),
            },
            "v3-governance",
        )
