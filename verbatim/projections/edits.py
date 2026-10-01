"""Edit semantics for file projections (SPEC_V4 V4-47.05).

A generated Markdown file is a *derived, reviewable view* — never a
mutation channel. Editing it produces attributed input and an explicit
proposal; original evidence is never overwritten:

* ``propose_edit`` turns an edit into a ``reviews`` row whose
  ``expected_versions`` pin the base revision — the existing review
  machinery's conflict detection applies (a head that moved after the
  file was built is ``STALE_PROPOSAL``/``stale``, never silently
  merged). The claim row is untouched; approval is a separate gated
  step that runs through the store's transition path.
* ``audit_file`` re-reads a (possibly hand-edited) projection file's
  entry markers and reports each entry as ``current`` / ``stale`` /
  ``foreign`` / ``suppressed`` — the conflict-detection surface an
  operator consults before proposing edits.

Both paths authorize through ``governance.authorize`` — proposing an
edit requires the ``review`` verb; auditing a file requires ``read``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..core.lifecycle import read_claim_head
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import Verb
from ..governance import authorize
from ..storage.repos import EventsRepo, PurgesRepo, ReviewsRepo
from . import markdown as _md
from .builder import ProjectionAuthority

_PROJECTION_POLICY = "projection-1"


def _fail(code: ErrorCode, msg: str) -> VerbatimError:
    return VerbatimError(code, msg)


def propose_edit(
    store: Any,
    *,
    authority: ProjectionAuthority,
    scope_id: str,
    claim_id: str,
    base_revision: int,
    new_object: Any = None,
    note: str = "",
) -> dict[str, Any]:
    """Turn a file edit into a reviewable proposal — never a write.

    ``base_revision`` is the revision the file showed (parsed from the
    entry marker). If the live head moved, this raises
    ``STALE_PROPOSAL`` — the conflict is surfaced, not auto-merged.
    Returns ``{review_id, conflict: False, ...}``.
    """
    if not isinstance(authority, ProjectionAuthority):
        raise _fail(ErrorCode.VALIDATION, "authority must be a ProjectionAuthority")
    scope_id = require_id(scope_id, "scope_id")
    claim_id = require_id(claim_id, "claim_id")
    if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision < 1:
        raise _fail(ErrorCode.VALIDATION, "base_revision must be an int >= 1")
    caller = authority.caller()
    with store.tx() as conn:
        # The proposer must hold review on the scope AND read on the
        # claim — an edit proposal can never name invisible evidence.
        authorize(conn, caller, scope_id, Verb.REVIEW.value,
                  purpose=authority.purpose)
        authorize(conn, caller, scope_id, Verb.READ.value,
                  purpose=authority.purpose)
        head = read_claim_head(conn, claim_id)
        if head is None or head.scope_id != scope_id:
            raise _fail(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "claim not found"
            )
        if int(head.revision) != base_revision:
            raise _fail(
                ErrorCode.STALE_PROPOSAL,
                f"claim {claim_id} head is revision {head.revision}, "
                f"file pinned {base_revision} — conflict, re-render first",
            )
        if claim_id in PurgesRepo(store).suppressed_ids("claim", [claim_id]):
            raise _fail(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                "claim is under suppression",
            )
        effect = {
            "effect": "edit_claim",
            "claim_id": claim_id,
            "base_revision": base_revision,
            "object": new_object,
            "note": note or "",
            "origin": "file-projection",
            "object_sha256": (
                hashlib.sha256(
                    json_dumps(new_object).encode("utf-8")
                ).hexdigest()
                if new_object is not None
                else None
            ),
        }
        EventsRepo(store).append(
            conn,
            scope_id,
            "projection_edit_proposed",
            caller.principal_id,
            {"claim_id": claim_id, "base_revision": base_revision},
            _PROJECTION_POLICY,
        )
        review_id = ReviewsRepo(store).create(
            conn,
            scope_id,
            effect,
            {claim_id: base_revision},
        )
    return {
        "review_id": review_id,
        "claim_id": claim_id,
        "scope_id": scope_id,
        "base_revision": base_revision,
        "conflict": False,
        "applied": False,
        "note": (
            "proposal queued — approval applies a NEW claim revision "
            "through the store's transition path; the original is "
            "never overwritten"
        ),
    }


def audit_file(
    store: Any,
    *,
    authority: ProjectionAuthority,
    scope_id: str,
    text: str,
) -> dict[str, Any]:
    """Compare a file's entry markers against the live store.

    Per entry: ``current`` (head still at the pinned revision),
    ``stale`` (head moved — the file's edit base is in conflict),
    ``foreign`` (no local claim — an unverifiable imported row), or
    ``suppressed`` (live tombstone — never editable, never resurfaceable).
    """
    if not isinstance(authority, ProjectionAuthority):
        raise _fail(ErrorCode.VALIDATION, "authority must be a ProjectionAuthority")
    if not isinstance(text, str):
        raise _fail(ErrorCode.VALIDATION, "file text must be str")
    scope_id = require_id(scope_id, "scope_id")
    caller = authority.caller()
    entries: list[dict[str, Any]] = []
    with store.read() as conn:
        authorize(conn, caller, scope_id, Verb.READ.value,
                  purpose=authority.purpose)
        purges = PurgesRepo(store)
        for meta in _iter_markers(text):
            cid = meta.get("claim_id")
            rev = meta.get("revision")
            if not isinstance(cid, str) or not isinstance(rev, int):
                entries.append({"status": "malformed", "meta": None})
                continue
            if cid in purges.suppressed_ids("claim", [cid]):
                entries.append(
                    {"claim_id": cid, "revision": rev, "status": "suppressed"}
                )
                continue
            head = read_claim_head(conn, cid)
            if head is None or head.scope_id != scope_id:
                entries.append(
                    {"claim_id": cid, "revision": rev, "status": "foreign"}
                )
                continue
            entries.append(
                {
                    "claim_id": cid,
                    "revision": rev,
                    "head_revision": int(head.revision),
                    "status": (
                        "current" if int(head.revision) == rev else "stale"
                    ),
                }
            )
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"scope_id": scope_id, "entries": entries, "counts": counts}


def _iter_markers(text: str):
    open_tag = f"<!-- {_md.ENTRY_MARK} "
    close = " -->"
    pos = 0
    while True:
        start = text.find(open_tag, pos)
        if start < 0:
            return
        end = text.find(close, start + len(open_tag))
        if end < 0:
            return
        meta = safe_json_loads(text[start + len(open_tag):end])
        if isinstance(meta, dict):
            yield meta
        pos = end + len(close)


__all__ = ["audit_file", "propose_edit"]
