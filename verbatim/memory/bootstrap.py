"""Trusted-owner bootstrap for the V5 consumer facade (SPEC_V5 §05).

On a genuinely new store, construction performs one atomic trusted-owner
establishment: register the launch-bound principal, provision the first
``user`` namespace (scope row + owner grant + alias record together),
and persist a bootstrap binding — all inside a single ``store.tx()`` so
a crash mid-bootstrap leaves no half-owned store.

Reopen semantics (V5-05.02/05.09):

  * The binding is verified, never rewritten: a different owner or a
    different resolved storage profile fails closed (opaque denial —
    convergence rule: constructors either converge on the same identity
    or fail cleanly).
  * Revoked authority is never resurrected.  A namespace whose owner
    grant was revoked yields zero effective verbs → the binding is dead
    and construction denies rather than re-minting a grant.
  * ``user_id`` is an alias label only — it is never consulted for
    identity, authority, or filesystem selection, and cannot take over
    another owner's store.
  * Ownership is *never* obtained through agent-capture consent: no
    ``capture_authorizations`` row is minted here, and no blanket agent
    retention consent exists anywhere in this path (V5-05.05).

Persisted binding: ``meta["memory.bootstrap.v1"]`` — the existing
``meta`` KV facility (no dedicated schema exists in this checkout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .. import governance
from ..core.time import now_us
from ..governance import CallerV3
from ..storage.repos import EventsRepo
from . import aliases
from .errors import denied

#: Root verbs for the personal namespace (V5-05.11): exactly this set —
#: never share/act/hydrate.
ROOT_VERBS: Tuple[str, ...] = ("read", "quote", "ingest", "derive", "review", "admin")

#: The owner root grant is purpose-unconstrained (``any``) — the same
#: convention ``VerbatimV3._ensure_owner_grant`` uses.  Purpose-tagged
#: constraints exist for *delegated* grants; the partition's trusted root
#: owner is the authority delegated grants attenuate from.
ROOT_PURPOSES: Optional[Tuple[str, ...]] = None

BINDING_KEY = "memory.bootstrap.v1"
_BINDING_VERSION = 1


@dataclass(frozen=True)
class BootstrapInfo:
    """Result of owner establishment / verification."""

    owner: str
    namespace: str
    alias: dict
    created: bool
    profile_id: str


def _live_verbs(conn, owner: str, namespace: str) -> frozenset:
    caller = CallerV3(principal_id=owner)
    try:
        return frozenset(governance.effective_verbs(conn, caller, namespace))
    except Exception:
        # A namespace with no usable authority surfaces as empty rather
        # than raising — construction then denies on the check below.
        return frozenset()


def ensure_bootstrap(
    store: Any,
    *,
    owner: str,
    owner_kind: str,
    profile_id: str,
    host_name: str,
    alias_label: str,
    verbs: Tuple[str, ...] = ROOT_VERBS,
    purposes: Optional[Tuple[str, ...]] = ROOT_PURPOSES,
) -> BootstrapInfo:
    """Establish-or-verify the trusted owner + personal namespace.

    Single write transaction: concurrent first constructors serialize on
    the SQLite write lock; the loser observes the committed binding and
    either converges (same identity) or fails closed (V5-05.09).
    """
    with store.tx() as conn:
        governance.seed_purposes(conn)
        binding = store._meta_get(conn, BINDING_KEY)
        created = False
        if binding is None:
            governance.register_principal(
                conn, kind=owner_kind, principal_id=owner
            )
            alias = aliases.provision(
                conn,
                store,
                owner=owner,
                profile_id=profile_id,
                kind="user",
                label=alias_label,
                verbs=verbs,
                purposes=purposes,
            )
            binding = {
                "v": _BINDING_VERSION,
                "owner": owner,
                "owner_kind": owner_kind,
                "profile_id": profile_id,
                "host_name": host_name,
                "verbs": list(verbs),
                "purposes": list(purposes) if purposes else "any",
                "created_us": now_us(),
            }
            store._meta_set(conn, BINDING_KEY, binding)
            EventsRepo(store).append(
                conn,
                alias["namespace"],
                "memory_bootstrap",
                owner,
                {"owner": owner, "profile_id": profile_id},
                "memory/v5",
            )
            created = True
        else:
            if not isinstance(binding, dict) or binding.get("v") != _BINDING_VERSION:
                raise denied("unrecognized bootstrap binding")
            if binding.get("owner") != owner:
                # One owner per store — never a takeover, never a hint.
                raise denied("store belongs to a different trusted identity")
            if binding.get("profile_id") != profile_id:
                raise denied("store profile binding mismatch")
            principal = governance.get_principal(conn, owner)
            if principal is None or principal.get("retired"):
                raise denied("trusted identity is not usable")

        # Resolve the requested user alias inside the owner's partition.
        alias = aliases.resolve(
            conn,
            store,
            owner=owner,
            profile_id=profile_id,
            kind="user",
            label=alias_label,
        )
        if alias is None:
            if not created:
                # Explicit authorized provisioning: the bound identity IS
                # the persisted trusted owner, so minting a new personal
                # partition is an owner act — journaled, durable, and
                # never available to a foreign principal (the owner check
                # above already denied that case).
                alias = aliases.provision(
                    conn,
                    store,
                    owner=owner,
                    profile_id=profile_id,
                    kind="user",
                    label=alias_label,
                    verbs=verbs,
                    purposes=purposes,
                )
                EventsRepo(store).append(
                    conn,
                    alias["namespace"],
                    "memory_alias_provisioned",
                    owner,
                    {"kind": "user", "profile_id": profile_id},
                    "memory/v5",
                )
            else:  # pragma: no cover - establish path always provisions
                raise denied("bootstrap provisioning did not resolve")

        namespace = alias["namespace"]
        # Reopen must not restore revoked grants: if the owner holds no
        # live verbs on the namespace the binding is dead — deny, never
        # re-mint.  Partial attenuation is respected as-is; individual
        # calls authorize per-verb and fail typed.
        if not _live_verbs(conn, owner, namespace):
            raise denied("no live authority on the bound namespace")

    return BootstrapInfo(
        owner=owner,
        namespace=namespace,
        alias=alias,
        created=created,
        profile_id=profile_id,
    )
