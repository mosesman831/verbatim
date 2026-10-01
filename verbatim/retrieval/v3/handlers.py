"""Index-artifact job handlers for the optional retrieval lanes
(SPEC_V3 §28.06, §40, V3-46 error taxonomy).

``sparse_index`` and ``late_index`` jobs are declared in the v3 queue so
the capability surface is enumerable — but no scoring/index
implementation ships in this build. Handlers therefore fail HONESTLY:
``CAPABILITY_UNAVAILABLE`` unless a published ``index_generations``
artifact row exists, and even then the missing implementation degrades
rather than fabricating index state (V3-28.12). A handler must never
silently mark an index built, never write artifacts it did not produce.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ...core.types import ErrorCode, VerbatimError


def _artifact_ready(conn: sqlite3.Connection, kind: str) -> bool:
    """A published artifact row exists for this index kind."""
    try:
        row = conn.execute(
            "SELECT 1 FROM index_generations WHERE kind = ?"
            " AND status IN ('ready','published') LIMIT 1",
            (kind,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _unavailable(kind: str, detail: str) -> VerbatimError:
    return VerbatimError(
        ErrorCode.CAPABILITY_UNAVAILABLE,
        f"{kind} index: {detail}",
    )


def handle_sparse_index(
    job: dict,
    owner: str,
    ingester: Any,
) -> None:
    """``sparse_index`` job handler (§28.06).

    Signature is the frozen v3 contract ``(job, owner, ingester)`` — the
    dispatcher calls ``handler(job, owner, self)``. Learned-sparse
    indexing is a research-to-optional capability: without provisioned
    artifacts the job fails with ``CAPABILITY_UNAVAILABLE``; with
    artifacts but no implementation it still reports the missing scorer
    rather than fabricating a build.
    """
    with ingester.store.read() as conn:
        ready = _artifact_ready(conn, "sparse")
    if not ready:
        raise _unavailable("sparse", "index artifacts not provisioned")
    raise _unavailable("sparse", "scoring implementation not installed")


def handle_late_index(
    job: dict,
    owner: str,
    ingester: Any,
) -> None:
    """``late_index`` job handler (§28.06).

    Late-interaction rerank artifacts gate the lane the same way: honest
    ``CAPABILITY_UNAVAILABLE`` until both artifact and scorer exist.
    """
    with ingester.store.read() as conn:
        ready = _artifact_ready(conn, "late")
    if not ready:
        raise _unavailable("late", "index artifacts not provisioned")
    raise _unavailable("late", "scoring implementation not installed")


# Dispatch surface for the job runner (mirrors _V3_KIND_HANDLERS).
HANDLERS = {
    "sparse_index": handle_sparse_index,
    "late_index": handle_late_index,
}
