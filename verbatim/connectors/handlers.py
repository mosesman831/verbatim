"""``connector_pull`` durable-job handler (SPEC_V4 §48, V4-48.05/09).

Registered in ``ingest._V3_KIND_HANDLERS`` — the dispatcher lazily
imports this module; a missing module fails ``CAPABILITY_UNAVAILABLE``.

Lifecycle under the queue's generation fencing:

1. ``_replay_done`` — a redelivery whose operation receipt already
   committed completes without rescanning (V2-39.10).
2. ``_pre_fence`` — cheap lease check before the (unfenceable) remote
   enumeration begins.
3. ``ConnectorService.pull`` runs its page transactions; EVERY page tx
   re-asserts ``(job_id, owner, generation)`` via the passed ``fence``
   callable, so a superseded or expired lease abandons the pull at the
   next page boundary and rolls that page's writes back (V4-42.01).
   Item writes + cursor advance + batch receipt already share each
   page transaction — the ledger cannot advance past uncommitted work.
4. A terminal ``_commit_effects`` transaction journals the pull report
   as the operation receipt and completes the job atomically.

``input_refs``::

    {"connector_id": str, "source": {...}, "scope_id": str,
     "principal_id": str, "purpose": str?, "authorization_id": str?,
     "dry_run": bool?, "batch_size": int?, "max_items": int?,
     "pull_id": str?}
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.types import ErrorCode, VerbatimError


def _require_ref(refs: Mapping[str, Any], key: str) -> str:
    value = refs.get(key)
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string",
        )
    return value


def _opt_int(refs: Mapping[str, Any], key: str) -> Any:
    value = refs.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"job input_refs.{key} must be an integer"
        )
    return value


def handle_connector_pull(job: dict[str, Any], owner: str, ingester: Any) -> None:
    refs = job["input_refs"]
    if not isinstance(refs, Mapping):
        raise VerbatimError(
            ErrorCode.VALIDATION, "connector_pull input_refs must be a mapping"
        )
    connector_id = _require_ref(refs, "connector_id")
    scope_id = _require_ref(refs, "scope_id")
    principal_id = _require_ref(refs, "principal_id")
    source = refs.get("source")
    if not isinstance(source, Mapping):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "connector_pull input_refs.source must be a mapping",
        )
    # The job row's partition is authoritative: a job enqueued under one
    # scope must never pull into another (V4-10.05 isolation).
    if job["scope_id"] != scope_id:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "connector_pull scope_id does not match the job's partition",
        )
    dry_run = refs.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise VerbatimError(
            ErrorCode.VALIDATION, "connector_pull dry_run must be a boolean"
        )
    batch_size = _opt_int(refs, "batch_size")
    max_items = _opt_int(refs, "max_items")
    authorization_id = refs.get("authorization_id")
    if authorization_id is not None and (
        not isinstance(authorization_id, str) or not authorization_id
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "connector_pull authorization_id must be a non-empty string",
        )
    purpose = refs.get("purpose")
    if purpose is not None and not isinstance(purpose, str):
        raise VerbatimError(
            ErrorCode.VALIDATION, "connector_pull purpose must be a string"
        )

    # Redelivery: the committed receipt completes the job without
    # rescanning — the receipt IS the proof the pull already ran.
    if ingester._replay_done(job, owner):
        return
    # Cheap early fence before unfenceable source enumeration.
    ingester._pre_fence(job, owner)

    from .service import ConnectorService

    service = ConnectorService(ingester.store, ingester.cfg)

    def _fence(conn: Any) -> None:
        ingester.jobs.assert_lease(
            conn, job["job_id"], owner, job["generation"]
        )

    report = service.pull(
        connector_id,
        dict(source),
        scope_id=scope_id,
        principal_id=principal_id,
        purpose=purpose,
        authorization_id=authorization_id,
        dry_run=dry_run,
        batch_size=batch_size or 64,
        max_items=max_items,
        fence=_fence,
    )

    def _apply(conn: Any) -> dict[str, Any]:
        return report.to_dict()

    with ingester.store.tx() as conn:
        ingester._commit_effects(
            conn, job, owner, "connector_pull", _apply
        )


__all__ = ["handle_connector_pull"]
