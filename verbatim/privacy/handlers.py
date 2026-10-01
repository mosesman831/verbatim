"""Privacy-lane job handlers (SPEC_V3 §35, §36, §40 — handler contract per
``docs/v3_contracts.md``).

Registered targets:

- ``handle_purge_vault`` — erase ``vault_entries`` for a purge selection
  (explicit ``entry_ids``, every entry referenced from a ``view_id``, or
  all live entries in the job scope). Idempotent: replaying the job
  re-erases nothing and reports the already-erased set.
- ``handle_vault_rotate`` — re-wrap every live entry's data key under the
  scope's *current* wrap-key version (§35.12, §37.03). Value ciphertexts,
  nonces, data keys, and AAD digests are untouched — only the outer wrap
  moves. The receipt lists the wrap versions retired by the pass and the
  backup dependency that remains; dropping the old key copy is the
  provider's responsibility, reported honestly.
- ``handle_purge_derived`` — deletion closure over the ``derivations``
  graph (§36.02, V4-38 / F4-07): children of purged parents are deleted
  transitively, derivation edges are removed, influence rows redacted,
  and recorded propagations marked revoked — all through the shared,
  resumable ``ClosureEngine``. There is no cascade bound: a bounded step
  yields a durable pending continuation, never a false completion.
  Unmapped child kinds are reported, never silently skipped.

``handle_purge_vault`` and ``handle_vault_rotate`` use
``ingester._commit_effects`` so domain effects, operation receipts, lease
fencing, and job completion commit in one transaction (V2-39.01/39.10
semantics carried into §40). ``handle_purge_derived`` runs the durable
closure engine in its own committed steps (the frontier must outlive the
job), then records the operation receipt and completes the job in the
same fenced transaction — a crashed run resumes on redelivery because
``begin`` is idempotent on the job's run id.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..core.lifecycle import PURGE_ACTOR
from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v4 import ClosurePhase
from ..storage import repos_v3
from ..storage.repos import EventsRepo
from . import closure as _closure
from . import erasure as _erasure
from . import vault as _vault


def _key_provider(ingester: Any) -> Any:
    """Optional externally attached vault key provider (§35.11)."""
    return getattr(ingester, "vault_key_provider", None)


def _policy_version(ingester: Any) -> str:
    return getattr(getattr(ingester, "policy", None), "policy_version", "v3")


# ---------------------------------------------------------------------------
# purge_vault
# ---------------------------------------------------------------------------


def handle_purge_vault(job: dict, owner: str, ingester: Any) -> None:
    """Erase vault entries selected by a purge job (§36.02).

    ``input_refs`` selectors (at least one required):
    ``entry_ids`` list, ``view_id`` (entries referenced by its
    ``vault_refs``), or ``all_entries: true`` for the job scope.
    ``purge_id``/``erasure_epoch`` annotate the erasure ledger.
    """
    refs = job["input_refs"]
    scope_id = job["scope_id"]
    explicit = refs.get("entry_ids") or []
    view_id = refs.get("view_id")
    all_entries = bool(refs.get("all_entries"))
    purge_id = refs.get("purge_id")
    erasure_epoch = int(refs.get("erasure_epoch") or 0)
    if (
        not isinstance(explicit, list)
        or (view_id is not None and not isinstance(view_id, str))
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "purge_vault entry_ids must be a list and view_id a string",
        )
    if not explicit and not view_id and not all_entries:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "purge_vault requires entry_ids, view_id, or all_entries",
        )
    for eid in explicit:
        require_id(eid, "entry_id")
    if view_id is not None:
        require_id(view_id, "view_id")

    def _apply(conn: sqlite3.Connection) -> dict[str, Any]:
        targets: list[str] = list(dict.fromkeys(explicit))
        if view_id is not None:
            for row in repos_v3.query(
                conn, "vault_refs",
                {"scope_id": scope_id, "view_id": view_id},
            ):
                if row["entry_id"] not in targets:
                    targets.append(row["entry_id"])
        if all_entries:
            for row in repos_v3.query(
                conn, "vault_entries",
                {"scope_id": scope_id, "erased_event": None},
            ):
                if row["entry_id"] not in targets:
                    targets.append(row["entry_id"])

        event_seq = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
        ).fetchone()[0]
        erased: list[str] = []
        skipped: list[str] = []
        for eid in targets:
            if _erasure.erase_entry(
                conn,
                eid,
                erased_event=event_seq,
                hmac_fn=ingester.store.hmac,
                purge_id=purge_id,
                erasure_epoch=erasure_epoch,
            ):
                erased.append(eid)
            else:
                skipped.append(eid)
        seq = EventsRepo(ingester.store).append(
            conn,
            scope_id,
            "vault_purged",
            PURGE_ACTOR,
            {
                "job_id": job["job_id"],
                "purge_id": purge_id,
                "requested": len(targets),
                "newly_erased": len(erased),
                "already_erased": len(skipped),
            },
            _policy_version(ingester),
        )
        return {
            "scope_id": scope_id,
            "erased": erased,
            "already_erased": skipped,
            "event_seq": seq,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "purge_vault", _apply)


# ---------------------------------------------------------------------------
# vault_rotate
# ---------------------------------------------------------------------------


def handle_vault_rotate(job: dict, owner: str, ingester: Any) -> None:
    """Re-wrap live entries under the current scope wrap key (§35.12).

    ``input_refs.scope_id`` overrides the job scope when present. Entries
    whose ``wrap_key_version`` already equals the current provisioned
    version are skipped; ciphertexts stay byte-identical either way.
    """
    refs = job["input_refs"]
    scope_id = refs.get("scope_id") or job["scope_id"]
    require_id(scope_id, "scope_id")
    cfg = ingester.cfg
    provider = _key_provider(ingester)

    def _apply(conn: sqlite3.Connection) -> dict[str, Any]:
        rows = repos_v3.query(
            conn, "vault_entries",
            {"scope_id": scope_id, "erased_event": None},
        )
        retired: set[int] = set()
        rewapped = 0
        for row in rows:
            old_version = int(row["wrap_key_version"])
            if _vault.rewrap(
                conn, row, cfg=cfg, key_provider=provider
            ):
                retired.add(old_version)
                rewapped += 1
        seq = EventsRepo(ingester.store).append(
            conn,
            scope_id,
            "vault_wrap_rotated",
            PURGE_ACTOR,
            {
                "job_id": job["job_id"],
                "rewrapped": rewapped,
                "retired_wrap_versions": sorted(retired),
            },
            _policy_version(ingester),
        )
        return {
            "scope_id": scope_id,
            "entries": len(rows),
            "rewrapped": rewapped,
            "retired_wrap_versions": sorted(retired),
            # §35.12: the receipt states remaining obligations plainly —
            # pre-rotation backups may still need the retired versions.
            "backup_dependency": (
                "backups taken before this rotation may still require "
                "the retired wrap key versions to decrypt"
                if retired
                else "no wrap version retired"
            ),
            "event_seq": seq,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "vault_rotate", _apply)


# ---------------------------------------------------------------------------
# purge_derived
# ---------------------------------------------------------------------------


def _parents_from(refs: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the purge selection: [{kind,id,revision?}, ...]."""
    raw = refs.get("parents")
    if raw is None and refs.get("parent") is not None:
        raw = [refs["parent"]]
    if not isinstance(raw, list) or not raw:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "purge_derived requires a non-empty parents list",
        )
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "parents entries must be objects"
            )
        kind = item.get("kind") or item.get("object_kind")
        oid = item.get("id") or item.get("object_id")
        rev = item.get("revision", item.get("object_revision"))
        if not kind or not oid:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "each parent needs kind/id (revision optional)",
            )
        require_id(str(oid), "parent_id")
        out.append(
            {
                "kind": str(kind),
                "id": str(oid),
                "revision": None if rev is None else int(rev),
            }
        )
    return out


def handle_purge_derived(job: dict, owner: str, ingester: Any) -> None:
    """Delete derived rows whose parents were purged (§36.02, V4-38/F4-07).

    ``input_refs.parents`` is a list of ``{"kind", "id", "revision"?}``
    purge parents (or a single ``parent`` object); ``purge_id`` optionally
    binds the closure to the originating purge's suppression registry.

    The closure runs through ``privacy.closure.ClosureEngine`` — the same
    engine the ordinary purge path uses (V4-05.05): suppression of the
    roots is durable at ``begin``; the frontier is persisted and stepped
    in bounded batches; there is no truncation bound — a run that cannot
    finish inside the drain leaves ``pending`` work and a resumable
    ``cleaning``/``failed_cleanup`` phase, never a false ``completed``.

    The job commits only when the run reaches ``completed``; a
    ``failed_cleanup`` run surfaces as a job failure with the run id so
    the retryable obligation is never silent (V4-38.11). Job redelivery
    replays ``begin`` on the same run id — a live run continues from its
    durable frontier, a failed one is resumed.
    """
    refs = job["input_refs"]
    scope_id = job["scope_id"]
    parents = _parents_from(refs)
    if ingester._replay_done(job, owner):
        return
    # Cheap early fence: the closure does real work in its own committed
    # transactions before the fenced receipt commit.
    ingester._pre_fence(job, owner)

    engine = _closure.ClosureEngine(ingester.store)
    run_id = f"purge_derived:{job['job_id']}"
    run = engine.begin(
        [(p["kind"], p["id"], p["revision"]) for p in parents],
        scope_id,
        purge_id=refs.get("purge_id"),
        run_id=run_id,
    )
    run = engine.drain(run_id)
    if run.phase is not ClosurePhase.COMPLETED:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"closure run {run_id} did not complete "
            f"({run.phase.value}): {run.error}",
        )

    def _apply(conn: sqlite3.Connection) -> dict[str, Any]:
        receipt = engine.receipt(run_id, conn=conn)
        root_set = {tuple(r) for r in receipt["roots"]}
        deleted = [
            {"kind": k, "id": i, "revision": r}
            for k, i, r in receipt["erased"]
            if (k, i, r) not in root_set
        ]
        unhandled = [
            {
                "kind": o["ref"][0],
                "id": o["ref"][1],
                "revision": o["ref"][2],
                "reason": o["reason"],
            }
            for o in receipt["outside_boundary"]
        ] + [
            {
                "kind": u["ref"][0],
                "id": u["ref"][1],
                "revision": u["ref"][2],
                "reason": u["reason"],
            }
            for u in receipt["unaffected"]
        ]
        ev_seq = EventsRepo(ingester.store).append(
            conn,
            scope_id,
            "derived_purged",
            PURGE_ACTOR,
            {
                "job_id": job["job_id"],
                "run_id": run_id,
                "parents": len(parents),
                "children_deleted": len(deleted),
                "unhandled": len(unhandled),
                "truncated": False,
            },
            _policy_version(ingester),
        )
        return {
            "scope_id": scope_id,
            "parents": parents,
            "closure_run_id": run_id,
            "phase": receipt["phase"],
            "children_deleted": deleted,
            "unhandled": unhandled,
            "truncated": False,
            "pending_work": receipt["pending_work"],
            "external_copies": receipt["external_copies"],
            "cryptographic_erasure": receipt["cryptographic_erasure"],
            "backup_obligation": receipt["backup_obligation"],
            "event_seq": ev_seq,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "purge_derived", _apply)
