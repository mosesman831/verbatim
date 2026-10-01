"""Policy-artifact graduation flow (SPEC_V6 V6-03.15/16).

``policy_artifacts`` (schema v2, ``PolicyArtifactsRepo``) has existed
since v2 with ``validation_state ∈ {unvalidated, validated, revoked}`` —
but nothing ever called it, so ``learned_active`` stayed a config-time
refusal. This package wires the graduation path:

    fit offline (``eval/v3/g8.py``)
      → ``register_artifact`` — strict ``learned_policy_v1`` validation,
        inserted ``unvalidated`` (registering directly as ``validated``
        is refused — the state is earned, not declared)
      → ``mark_validated`` — ``unvalidated → validated`` ONLY with
        executed paired-evaluation evidence ``{paired_run, gate,
        verdict: "pass"}``, attested durably in
        ``policy_artifact_attestations`` inside the same transaction
      → ``bind_for_activation`` — ``learned_active`` resolves the bound
        ``v3.retrieval.controller_policy_artifact`` id against THIS
        store's registry: missing/unvalidated/revoked/corrupt all refuse
        ``CONFIG_INVALID`` naming the artifact (never a silent
        deterministic fallback — that fallback is for shadow/replay).

``mark_revoked`` completes the lifecycle: a validated or unvalidated
artifact can be withdrawn (attested); revocation is terminal — a
revoked artifact can never validate.
"""

from .artifacts import (
    attestations,
    bind_for_activation,
    mark_revoked,
    mark_validated,
    register_artifact,
)

__all__ = [
    "attestations",
    "bind_for_activation",
    "mark_revoked",
    "mark_validated",
    "register_artifact",
]
