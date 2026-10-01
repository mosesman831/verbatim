"""Job handlers for the v3 experience pipeline (SPEC_V3 §40, §20.08).

Registered in ``ingest._V3_KIND_HANDLERS``:

* ``episode_build`` → :func:`handle_episode_build` —
  ``input_refs {"trajectory_id"}`` groups the completed trajectory into
  an episode (members + derivations) and chains a ``transition_build``
  job for the same episode, deduped on the episode id so replayed
  pipelines converge (V3-20.08).
* ``transition_build`` → :func:`handle_transition_build` —
  ``input_refs {"episode_id"}`` derives the per-step transitions,
  anchor links, and derivations edges.

Both effects commit inside ONE generation-fenced transaction through
``ingester._commit_effects``: a redelivered job replays its operation
receipt instead of re-deriving, and a superseded lease generation can
never commit (V2-39.05/V3-40 carried conventions). Domain rebuilds are
idempotent on top of that — deterministic episode/transition identities
plus PK-deduped member/anchor/derivation rows.
"""

from __future__ import annotations

from typing import Any

from ..core.types import ErrorCode, JobKind, VerbatimError
from . import episodes_v3, hierarchy, repo as _repo, scenes, transitions


def _require_ref(refs: dict[str, Any], key: str) -> str:
    value = refs.get(key)
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string",
        )
    return value


def handle_episode_build(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Build the episode for ``input_refs.trajectory_id``, then chain the
    transition derivation job for the same episode (background lane,
    deduped on the episode id).

    V4-22.09 hierarchy maintenance is inline and bounded: the new episode
    is assigned to its declared-family scene (one upsert, no rescan) and
    its overview — plus its scene's — is refreshed through the derived
    ``v4.hierarchy.v1`` producer. Everything is idempotent on
    deterministic ids, so redelivery converges.
    """
    refs = job["input_refs"]
    trajectory_id = _require_ref(refs, "trajectory_id")

    def _apply(conn: Any) -> dict[str, Any]:
        episode_id = episodes_v3.build_episode(conn, trajectory_id)
        ep = _repo.episode_row(conn, episode_id)
        scope_id = ep["scope_id"] if ep is not None else job["scope_id"]
        scene_res = scenes.assign_episode(conn, episode_id)
        hierarchy.build_overview(conn, scope_id, "episode", episode_id)
        scene_id = scene_res.get("scene_id")
        if scene_id:
            hierarchy.build_overview(conn, scope_id, "episode", scene_id)
        dedup = ingester.store.hmac(
            f"transition_build:{episode_id}".encode("utf-8")
        )
        ingester.jobs.enqueue(
            conn,
            job["scope_id"],
            JobKind.TRANSITION_BUILD,
            {"episode_id": episode_id},
            dedup_key=dedup,
            operation_key=(
                f"transition_build:{episode_id}"
                if ingester.jobs.supports_durability
                else None
            ),
        )
        return {
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": scene_id,
        }

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "episode_build", _apply)


def handle_transition_build(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Derive transitions for ``input_refs.episode_id``; the receipt
    reports the episode's transition count."""
    refs = job["input_refs"]
    episode_id = _require_ref(refs, "episode_id")

    def _apply(conn: Any) -> dict[str, Any]:
        count = transitions.build_transitions(conn, episode_id)
        return {"episode_id": episode_id, "transitions": count}

    with ingester.store.tx() as conn:
        ingester._commit_effects(conn, job, owner, "transition_build", _apply)


__all__ = ["handle_episode_build", "handle_transition_build"]
