"""State anchors and environment fingerprints (SPEC_V3 §20.02–§20.03).

State anchors are exact references to observed environment state — file
revisions, artifact digests, DOM/screen digests, tool versions. They let
the learning plane reconstruct observation/action order without replaying
full trajectories; they never establish causation (V3-20.03, V3-20.04).

Conventions:

- ``record_anchor`` mints a deterministic ``anchor_id`` from the recorded
  fields, so replayed capture yields identical anchor identity
  (V3-20.08) and re-recording is a no-op rather than a duplicate.
- Ordering helpers join ``state_anchors`` to ``trajectory_steps`` — an
  anchor participates in before/after queries only while it is attached
  to a trajectory step (``state_anchors.step_id``); unattached anchors
  stay exact evidence but cannot be ordered.
- Every helper takes the caller's transaction ``conn``; nothing opens its
  own transaction (§21-style caller-owned writes).
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v3 import EnvironmentFingerprint
from ..storage import repos_v3
from ..storage.repos import _next_event_seq, _require_str, _row, _rows

# Documented anchor kinds (types_v3.StateAnchor). The column carries no
# CHECK — hosts may record other exact-reference kinds — so this set is
# vocabulary, not validation.
ANCHOR_KINDS = frozenset(
    {
        "file_revision",
        "artifact_digest",
        "tool_version",
        "dom_digest",
        "screen_digest",
    }
)

_NEAR_SIDES = ("before", "after")


def anchor_id_for(
    scope_id: str,
    kind: str,
    ref: str,
    digest: str,
    step_id: Optional[str] = None,
) -> str:
    """Deterministic anchor identity over the recorded fields."""
    raw = "\x00".join((scope_id, kind, ref, digest, step_id or ""))
    return "sa_" + hashlib.sha256(f"v3:anchor\x00{raw}".encode("utf-8")).hexdigest()[:32]


def record_anchor(
    conn: sqlite3.Connection,
    scope_id: str,
    kind: str,
    ref: str,
    digest: str,
    step_id: Optional[str] = None,
    *,
    anchor_id: Optional[str] = None,
) -> str:
    """Persist one state anchor inside the caller's transaction.

    Idempotent: the default ``anchor_id`` is a pure function of the
    recorded fields, so re-recording the same observation returns the
    existing anchor instead of duplicating it. A caller-supplied
    ``anchor_id`` that already exists must describe identical fields —
    divergence is ``INTEGRITY``, never a silent rewrite.
    """
    require_id(scope_id, "scope_id")
    _require_str(kind, "kind")
    _require_str(ref, "ref")
    _require_str(digest, "digest")
    if step_id is not None:
        require_id(step_id, "step_id")
    aid = anchor_id or anchor_id_for(scope_id, kind, ref, digest, step_id)
    require_id(aid, "anchor_id")
    existing = repos_v3.get(conn, "state_anchors", {"anchor_id": aid})
    if existing is not None:
        diverged = [
            name
            for name, want in (
                ("scope_id", scope_id),
                ("kind", kind),
                ("ref", ref),
                ("digest", digest),
                ("step_id", step_id),
            )
            if existing.get(name) != want
        ]
        if diverged:
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                f"anchor {aid} re-recorded with different fields: "
                + ", ".join(diverged),
            )
        return aid
    repos_v3.insert(
        conn,
        "state_anchors",
        {
            "anchor_id": aid,
            "scope_id": scope_id,
            "kind": kind,
            "ref": ref,
            "digest": digest,
            "step_id": step_id,
            "created_event": _next_event_seq(conn),
        },
    )
    return aid


def anchors_for_step(conn: sqlite3.Connection, step_id: str) -> list[dict[str, Any]]:
    """Anchors attached directly to one trajectory step, id-ordered."""
    require_id(step_id, "step_id")
    return repos_v3.query(
        conn, "state_anchors", {"step_id": step_id}, order="anchor_id"
    )


def anchors_for_ords(
    conn: sqlite3.Connection,
    trajectory_id: str,
    ords: Iterable[int],
) -> list[dict[str, Any]]:
    """Anchors attached to the steps at ``ords`` of one trajectory.

    Each returned row is the ``state_anchors`` row plus ``step_ord`` —
    the ord of the step the anchor is attached to — ordered by step ord
    then anchor id for deterministic replay.
    """
    require_id(trajectory_id, "trajectory_id")
    ord_list = sorted({int(o) for o in ords})
    if not ord_list:
        return []
    marks = ", ".join("?" for _ in ord_list)
    return _rows(
        conn.execute(
            "SELECT a.*, s.ord AS step_ord FROM state_anchors a"
            " JOIN trajectory_steps s ON s.step_id = a.step_id"
            f" WHERE s.trajectory_id = ? AND s.ord IN ({marks})"
            " ORDER BY s.ord, a.anchor_id",
            (trajectory_id, *ord_list),
        )
    )


def anchors_near(
    conn: sqlite3.Connection,
    step_id: str,
    side: str,
) -> list[dict[str, Any]]:
    """Anchors on steps strictly ``before``/``after`` the given step.

    ``before`` → anchors attached to trajectory steps with ``ord`` below
    the step's ord (the observed prior state); ``after`` → ords strictly
    above (subsequent observations). Anchors on the step itself are on
    neither side — use :func:`anchors_for_step` or
    :func:`anchors_for_ords` for the inclusive window.
    """
    require_id(step_id, "step_id")
    if side not in _NEAR_SIDES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"anchors_near side must be one of {_NEAR_SIDES}",
        )
    step = repos_v3.get(conn, "trajectory_steps", {"step_id": step_id})
    if step is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "trajectory step not found"
        )
    op = "<" if side == "before" else ">"
    return _rows(
        conn.execute(
            "SELECT a.*, s.ord AS step_ord FROM state_anchors a"
            " JOIN trajectory_steps s ON s.step_id = a.step_id"
            f" WHERE s.trajectory_id = ? AND s.ord {op} ?"
            " ORDER BY s.ord, a.anchor_id",
            (step["trajectory_id"], int(step["ord"])),
        )
    )


# ---------------------------------------------------------------------------
# environment fingerprints (§20.02, §21 environment_fingerprint)
# ---------------------------------------------------------------------------


def _pairs(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    """Normalize version pairs: accepts a mapping or an iterable of pairs."""
    if value is None:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out: list[tuple[str, str]] = []
    try:
        for k, v in items:
            out.append((str(k), str(v)))
    except (TypeError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{field} must be pairs of (name, version)"
        ) from exc
    return tuple(out)


def fingerprint_from_mapping(data: dict[str, Any]) -> EnvironmentFingerprint:
    """Rebuild an :class:`EnvironmentFingerprint` from a JSON-shaped mapping.

    Missing fields stay ``None`` — absent environment information is
    unknown, never a wildcard match (§21.07).
    """
    if not isinstance(data, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "environment fingerprint must be a mapping"
        )
    return EnvironmentFingerprint(
        repo_id=data.get("repo_id"),
        repo_revision=data.get("repo_revision"),
        runtime_versions=_pairs(data.get("runtime_versions"), "runtime_versions"),
        tool_schema_versions=_pairs(
            data.get("tool_schema_versions"), "tool_schema_versions"
        ),
        platform=data.get("platform"),
    )


def environment_digest(fingerprint: Any) -> str:
    """SHA-256 digest of an environment fingerprint.

    Accepts an :class:`EnvironmentFingerprint` or an equivalent mapping;
    the digest is the canonical identity stored on trajectories, steps,
    transitions, and procedures (§20.02, §22.05).
    """
    if isinstance(fingerprint, EnvironmentFingerprint):
        return fingerprint.digest()
    if isinstance(fingerprint, dict):
        return fingerprint_from_mapping(fingerprint).digest()
    raise VerbatimError(
        ErrorCode.VALIDATION,
        "environment_digest needs an EnvironmentFingerprint or mapping",
    )
