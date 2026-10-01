"""The one profile-store resolver (SPEC_V4 §06–§07, F4-18, V4-05.10,
V4-07.10, C82).

Two on-disk store conventions historically coexist under a profile's data
directory:

- the **profile store** ``<data_dir>/<profile_id>.db`` — the engine's
  canonical name (``api.open_store``'s long-standing convention), and
- the **v3 store** ``<data_dir>/v3.db`` — the v3 adapter's independent
  default (the "version-named database" V4-07.10 prohibits adapters from
  minting silently).

Every host surface — ``api.open_store`` (registered Hermes provider, CLI,
MCP) and the native capture adapters — resolves through
:func:`resolve_store_path`; no caller may invent its own filename
(V4-07.10).

Precedence (deterministic, non-destructive):

1. ``explicit_path`` — an operator-supplied path wins outright; it is the
   explicit decision channel and also the way to reopen either side of a
   conflicted pair.
2. ``prefer`` — the operator's conflict decision without needing the
   hashed filename: ``"legacy"``/``"profile"`` selects the profile store,
   ``"v3"`` selects ``v3.db``, and a literal path equal to a discovered
   candidate selects it. When the directory holds no store yet and
   ``create`` is set, ``prefer`` also picks the convention the new store
   is created under (default: the profile store).
3. Inventory — exactly one store on disk resolves to it (either
   convention adopted as-is: an existing ``v3.db`` keeps serving its
   profile rather than being replaced by an empty profile store).
4. Both conventions present → **conflict**: ``path=None``,
   ``conflicts`` lists both, ``needs_operator_decision()`` is true and
   :func:`require_store_path` raises ``STORE_CONFLICT``. A conflict is
   never resolved by silently merging, silently picking one, or creating
   an empty replacement (V4-05.10, C82).
5. Neither present → ``NONE``; with ``create=True`` the create target is
   the profile store ``<profile_id>.db`` (the canonical convention —
   ``prefer="v3"`` is the only way to mint the version-named file, and
   only as an explicit operator decision).

``StoreResolution`` is the frozen contract type from
``core/types_v4.py`` — this module supplies the policy, not the type.
"""

from __future__ import annotations

import os
from typing import Optional

from ..core.types import ErrorCode, VerbatimError
from ..core.types_v4 import StoreResolution, StoreResolutionKind

#: The v3 adapter's historical version-named database (F4-18). Kept as a
#: discoverable convention so existing deployments keep resolving — new
#: stores default to the profile-keyed name instead.
V3_STORE_NAME = "v3.db"

#: ``prefer``/``prefer_store`` tokens accepted as operator decisions.
_LEGACY_TOKENS = frozenset({"legacy", "profile", "legacy_profile"})
_V3_TOKENS = frozenset({"v3", "v3_default"})


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


def _inventory(data_dir: str, profile_id: Optional[str]) -> dict:
    """Existing candidate stores under ``data_dir`` (deterministic order:
    profile store first, then ``v3.db``)."""
    found: dict[str, str] = {}
    legacy = (
        os.path.join(data_dir, f"{profile_id}.db") if profile_id else None
    )
    if legacy is not None and os.path.isfile(legacy):
        found[StoreResolutionKind.LEGACY_PROFILE] = legacy
    v3 = os.path.join(data_dir, V3_STORE_NAME)
    if os.path.isfile(v3):
        found[StoreResolutionKind.V3_DEFAULT] = v3
    return found, legacy, v3


def _prefer_target(
    prefer: str,
    legacy: Optional[str],
    v3: str,
) -> tuple:
    """Map a ``prefer`` decision to (kind|None, path).

    Convention tokens name a convention's canonical path; anything else is
    treated as a literal filesystem path that must match a discovered
    candidate exactly.
    """
    token = str(prefer).strip()
    low = token.lower()
    if low in _LEGACY_TOKENS:
        if legacy is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "prefer='legacy' requires a profile_id — no profile store "
                "name can be formed",
            )
        return StoreResolutionKind.LEGACY_PROFILE, legacy
    if low in _V3_TOKENS:
        return StoreResolutionKind.V3_DEFAULT, v3
    return None, _norm(token)


def resolve_store_path(
    data_dir: str,
    profile_id: Optional[str] = None,
    explicit_path: Optional[str] = None,
    create: bool = False,
    *,
    prefer: Optional[str] = None,
) -> StoreResolution:
    """Resolve the single authoritative store path for a profile.

    Returns a :class:`StoreResolution` — inspectable, never silently
    destructive. ``data_dir`` is the profile's verbatim data directory;
    ``profile_id`` names the profile-store convention (``{profile_id}.db``);
    ``explicit_path`` and ``prefer`` are the operator decision channels;
    ``create`` permits returning a not-yet-existing create target.

    Raises ``VerbatimError(VALIDATION)`` for malformed directives
    (directory as explicit path, ``prefer`` naming nothing, creation
    without a nameable store). Conflict is reported on the resolution —
    callers turn it into ``STORE_CONFLICT`` via :func:`require_store_path`.
    """
    root = _norm(data_dir)

    # -- explicit operator path wins outright (the conflict-escape hatch) --
    if explicit_path is not None:
        target = _norm(explicit_path)
        if not target or os.path.isdir(target):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"explicit store path is a directory or empty: {explicit_path!r}",
            )
        found, _, _ = _inventory(root, profile_id)
        # The bypassed inventory stays visible on the resolution.
        candidates = (target,) + tuple(
            p for p in found.values() if p != target
        )
        return StoreResolution(
            kind=StoreResolutionKind.EXPLICIT,
            path=target,
            candidates=candidates,
        )

    found, legacy, v3 = _inventory(root, profile_id)
    candidates = tuple(found.values())

    # -- operator preference (conflict decision / fresh-store convention) --
    if prefer is not None:
        want_kind, want_path = _prefer_target(prefer, legacy, v3)
        if want_path in candidates:
            kind = (
                want_kind
                if want_kind is not None
                else next(k for k, p in found.items() if p == want_path)
            )
            return StoreResolution(
                kind=kind, path=want_path, candidates=candidates
            )
        if not found and create:
            if want_kind is None:
                # A literal path prefer on an empty directory is just an
                # explicit create target.
                return StoreResolution(
                    kind=StoreResolutionKind.EXPLICIT,
                    path=want_path,
                    candidates=(want_path,),
                )
            return StoreResolution(
                kind=want_kind, path=want_path, candidates=(want_path,)
            )
        if not found:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"prefer={prefer!r} names no discovered store (none exist "
                f"under {root})",
            )
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"prefer={prefer!r} names no discovered store under {root} "
            f"(found: {sorted(candidates)}); use explicit_path to select a "
            "path outside the inventory",
        )

    # -- deterministic inventory rules --------------------------------------
    if len(found) == 1:
        kind, path = next(iter(found.items()))
        return StoreResolution(kind=kind, path=path, candidates=candidates)

    if len(found) > 1:
        # Both conventions on disk: an operator decision is required —
        # never a silent merge or empty replacement (V4-05.10, C82).
        return StoreResolution(
            kind=StoreResolutionKind.NONE,
            path=None,
            candidates=candidates,
            conflicts=candidates,
        )

    # -- nothing on disk ------------------------------------------------------
    if not create:
        return StoreResolution(
            kind=StoreResolutionKind.NONE, path=None, candidates=()
        )
    if legacy is not None:
        return StoreResolution(
            kind=StoreResolutionKind.LEGACY_PROFILE,
            path=legacy,
            candidates=(legacy,),
        )
    raise VerbatimError(
        ErrorCode.VALIDATION,
        "cannot create a store: no profile_id to name the profile store "
        "and no explicit_path/prefer decision",
    )


def require_store_path(resolution: StoreResolution) -> str:
    """Turn a resolution into a path or a typed error (single policy).

    ``conflicts``/unresolved multi-candidate → ``STORE_CONFLICT`` with the
    discovered paths and the decision channels spelled out; an empty
    inventory → ``NOT_FOUND_OR_FORBIDDEN`` (matching ``open_store``'s
    historical miss error). Never picks a store on the caller's behalf.
    """
    if resolution.path is not None and not resolution.conflicts:
        return resolution.path
    if resolution.conflicts or resolution.needs_operator_decision():
        raise VerbatimError(
            ErrorCode.STORE_CONFLICT,
            "conflicting stores discovered — an operator decision is "
            "required (pass explicit_path/store_path or prefer="
            "'legacy'|'v3'); candidates: "
            + ", ".join(resolution.conflicts or resolution.candidates),
        )
    raise VerbatimError(
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        "no profile store found (looked for {profile_id}.db and "
        f"{V3_STORE_NAME})",
    )


__all__ = [
    "V3_STORE_NAME",
    "require_store_path",
    "resolve_store_path",
]
