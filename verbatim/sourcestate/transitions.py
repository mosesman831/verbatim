"""``source_revision_transition`` — fenced, CAS-guarded state transitions
for the ``source_state/v1`` control artifact (SPEC_V5 §14.3, V5-14.10 …
V5-14.16).

Every public writer here is *coordinator-effect style*: the caller opens
``store.tx()`` and passes the connection; the function re-verifies every
fence inside that transaction (expected control version, expected
predecessor head, pinned scope epochs, tombstone/scheduled-transition
state) and then commits row update + immutable doc + registry bump +
provenance edges + projection-generation invalidation in the same commit.
Nothing partially applies — any failure propagates and the transaction
rolls back.

Fences (V5-14.01/14.11):

* ``expected_control_version`` — required CAS on ``control_version``;
  mismatch → ``STALE_DEPENDENCY`` (the same code the coordinator raises
  for expected-revision drift). The ``UPDATE`` itself is additionally
  guarded on the expected version, so the compare-and-set is atomic even
  if the pre-check raced.
* ``expected_revision`` — optional CAS on ``mutation_head`` (the approved
  predecessor head a mutation binds). ``MemoryRef.expected_revision``
  maps here: stale or foreign predecessors conflict instead of silently
  rebinding (V5-14.02/14.11).
* ``epoch_vector`` — optional ``{scope_id: authz_epoch}`` pins rechecked
  through ``governance.epochs``; drift → ``STALE_EPOCH`` (V4-11.07 rule).
* Tombstone — a ``disposition="erased"`` row is terminal:
  ``INVALID_TRANSITION``. Erasure never reactivates an older revision
  (V5-14.16).
* Scheduled transitions — while a ``supersede``'s ``effective_at`` is
  still in the future the transition is *unresolved*: further changes
  require ``resolve_pending=True`` (the explicit reviewed resolution —
  V5-14.12 forbids silently replacing the schedule).

Validation (V5-14.13):

* ``effective_at`` is accepted only for an explicit ``supersede``
  (disposition ``superseded``); any other combination fails
  ``VALIDATION``. When omitted on a supersede, the commit time is the
  declared boundary — never a timestamp parsed from prose (V5-14.12).
* ``superseded_by`` (the successor revision) is required for
  ``superseded``, optional for ``corrected`` (a correction may mark the
  head without appending a replacement), and rejected for every other
  disposition.
* ``known_at`` must be nondecreasing across a source's history —
  record/system time cannot regress inside one ledger.

Erasure / closure membership (V5-14.16, V5-15.03):

* ``apply_erasure`` is the deletion-closure hook: purge/suppression paths
  call it inside their closure transaction to fence the row to the
  ``erased`` tombstone. It accepts the same CAS fences so a version-bound
  forget conflicts rather than silently deleting a newer revision.
* ``assert_publishable`` is the producer-path recheck (V5-14.14): pending
  projection/harvest jobs call it immediately before publish so a
  replacement or erasure that landed since the job was leased cannot be
  overtaken by its output.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Mapping, Optional

from ..core import time as _time
from ..core.types import ErrorCode, VerbatimError, require_id
from ..governance import epochs as _epochs
from . import state as _state
from .state import (
    CHANGE_ALIASES,
    EFFECT_KIND,
    HEAD_UNRESOLVED,
    TRANSITION_DISPOSITIONS,
    SourceState,
    _event_us,
    _norm_head,
    _norm_time,
    _rfc3339,
    _to_us,
)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------


def _normalize_disposition(value: Any) -> str:
    """Accept change-kind or disposition spelling; store the disposition."""
    if not isinstance(value, str):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid disposition {value!r}"
        )
    disp = CHANGE_ALIASES.get(value, value)
    if disp == "erased":
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            "disposition 'erased' is reachable only through apply_erasure "
            "(erasure is a closure operation, not a transition)",
        )
    if disp not in TRANSITION_DISPOSITIONS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"disposition must be one of {sorted(TRANSITION_DISPOSITIONS)} "
            f"or change kinds {sorted(CHANGE_ALIASES)}",
        )
    return disp


def _validate_fields(
    disp: str,
    superseded_by: Optional[str],
    effective_at: Any,
    valid_from: Optional[str],
    valid_to: Optional[str],
) -> None:
    """Field-combination rules (V5-14.12/14.13)."""
    if effective_at is not None and disp != "superseded":
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "effective_at is accepted only for explicit supersession "
            "(V5-14.13)",
        )
    if disp == "superseded" and superseded_by is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "supersede requires superseded_by (the successor revision)",
        )
    if superseded_by == HEAD_UNRESOLVED:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "superseded_by must name a concrete revision",
        )
    if superseded_by is not None and disp not in ("superseded", "corrected"):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"superseded_by is not meaningful for disposition {disp!r}",
        )
    if valid_from is not None and valid_to is not None:
        if _to_us(valid_from, "valid_from") >= _to_us(valid_to, "valid_to"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "valid_from must precede valid_to",
            )


def _check_fences(
    conn: sqlite3.Connection,
    current: SourceState,
    *,
    expected_control_version: Any,
    expected_revision: Any,
    epoch_vector: Optional[Mapping[str, int]],
) -> None:
    """Re-verify every fence against the live row inside the tx."""
    if isinstance(expected_control_version, bool) or not isinstance(
        expected_control_version, int
    ) or expected_control_version < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "expected_control_version must be an int >= 0",
        )
    if current.control_version != expected_control_version:
        raise VerbatimError(
            ErrorCode.STALE_DEPENDENCY,
            f"source {current.source_id} control_version is "
            f"{current.control_version}, expected "
            f"{expected_control_version}",
        )
    if expected_revision is not None:
        want = _norm_head(expected_revision, "expected_revision")
        if current.mutation_head != want:
            raise VerbatimError(
                ErrorCode.STALE_DEPENDENCY,
                f"source {current.source_id} mutation head is "
                f"{current.mutation_head!r}, expected {want!r}",
            )
    for scope_id, epoch in (epoch_vector or {}).items():
        if _epochs.current_epoch(conn, scope_id) != int(epoch):
            raise VerbatimError(
                ErrorCode.STALE_EPOCH,
                f"scope {scope_id} epoch drifted since plan computation",
            )


def _replay_lookup(
    conn: sqlite3.Connection,
    source_id: str,
    operation_id: str,
    *,
    disp: str,
    superseded_by: Optional[str],
    effective_at: Any,
    valid_from: Optional[str],
    valid_to: Optional[str],
) -> Optional[SourceState]:
    """Idempotent replay (the coordinator's V4-09.05 rule): a committed
    doc already carrying ``operation_id`` means this transition was applied
    — identical request surface returns the prior result; a different one
    is ``OPERATION_CONFLICT``, never a silent second application.

    ``effective_at`` compares only when the caller supplied it — its
    omitted form defaults to the original commit time, which is
    deliberately not reconstructible on replay.
    """
    require_id(operation_id, "operation_id")
    eff = (
        _rfc3339(_to_us(effective_at, "effective_at"))
        if effective_at is not None
        else None
    )
    for doc in _state._docs(conn, source_id):
        if doc.get("operation_id") != operation_id:
            continue
        same = (
            doc["disposition"] == disp
            and doc.get("superseded_by") == superseded_by
            and doc.get("valid_from") == valid_from
            and doc.get("valid_to") == valid_to
            and (eff is None or doc.get("effective_at") == eff)
        )
        if not same:
            raise VerbatimError(
                ErrorCode.OPERATION_CONFLICT,
                f"operation {operation_id} already applied with different "
                "input",
            )
        return SourceState.from_dict(doc)
    return None


# ----------------------------------------------------------------------
# transition — the coordinator-effect application (V5-14.10)
# ----------------------------------------------------------------------


def transition(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    expected_control_version: int,
    disposition: str,
    superseded_by: Any = None,
    effective_at: Any = None,
    valid_from: Any = None,
    valid_to: Any = None,
    producer: str,
    expected_revision: Any = None,
    epoch_vector: Optional[Mapping[str, int]] = None,
    resolve_pending: bool = False,
    actor: Optional[str] = None,
    operation_id: Optional[str] = None,
    known_at: Any = None,
    store: Any = None,
) -> SourceState:
    """Apply one fenced control transition; returns the new record.

    Must run inside ``store.tx()``. The logical coordinator effect is
    ``source_revision_transition`` (V5-14.10): the committed doc is its
    persisted encoding, and passing ``store`` bumps the projection
    generation in the same transaction so affected claims/views/index/
    cache entries invalidate atomically.

    ``disposition`` accepts the change kind (``supersede``/``correct``/
    ``retract``/``archive``/``record``/``activate``) or the stored
    disposition spelling (``superseded``/``corrected``/…). ``known_at``
    defaults to commit time; overriding it exists for replay/migration —
    it must not regress below the current record's ``known_at``.
    """
    require_id(source_id, "source_id")
    _state._require_table(conn)
    if not isinstance(producer, str) or not producer:
        raise VerbatimError(ErrorCode.VALIDATION, "producer required")

    disp = _normalize_disposition(disposition)
    vf = _norm_time(valid_from, "valid_from")
    vt = _norm_time(valid_to, "valid_to")
    succ = (
        _norm_head(superseded_by, "superseded_by")
        if superseded_by is not None
        else None
    )
    _validate_fields(disp, succ, effective_at, vf, vt)

    row = _state._fetch_row(conn, source_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "source_state not found — ensure_state binds first",
        )
    current = SourceState.from_row(row)

    if operation_id is not None:
        replayed = _replay_lookup(
            conn,
            source_id,
            operation_id,
            disp=disp,
            superseded_by=succ,
            effective_at=effective_at,
            valid_from=vf,
            valid_to=vt,
        )
        if replayed is not None:
            return replayed

    _check_fences(
        conn,
        current,
        expected_control_version=expected_control_version,
        expected_revision=expected_revision,
        epoch_vector=epoch_vector,
    )
    if current.tombstoned:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            "source_state is erased — erasure never reactivates a "
            "revision (V5-14.16)",
        )
    now = _time.now_us()
    if current.pending_scheduled(now) and not resolve_pending:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            "a scheduled supersede is unresolved — resolving it requires "
            "resolve_pending=True (explicit reviewed resolution, V5-14.12)",
        )

    known = (
        _to_us(known_at, "known_at") if known_at is not None else now
    )
    if known < _to_us(current.known_at, "known_at"):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "known_at cannot regress below the current record",
        )
    if disp == "superseded":
        # V5-14.12: omitted effective_at = this operation's commit time,
        # never a timestamp invented from prose.
        eff = (
            _rfc3339(_to_us(effective_at, "effective_at"))
            if effective_at is not None
            else _rfc3339(known)
        )
    else:
        eff = None

    new_version = current.control_version + 1
    new_head = succ if succ is not None else current.mutation_head
    new = SourceState(
        source_id=source_id,
        namespace=current.namespace,
        control_version=new_version,
        mutation_head=new_head,
        disposition=disp,
        superseded_by=succ,
        effective_at=eff,
        known_at=_rfc3339(known),
        valid_from=vf,
        valid_to=vt,
        updated_at=_rfc3339(now),
        producer=producer,
    )
    updated = _state._cas_update(
        conn,
        source_id,
        current.control_version,
        {
            "control_version": new.control_version,
            "mutation_head": new.mutation_head,
            "disposition": new.disposition,
            "superseded_by": new.superseded_by,
            "effective_at": new.effective_at,
            "known_at": new.known_at,
            "valid_from": new.valid_from,
            "valid_to": new.valid_to,
            "updated_at": new.updated_at,
            "producer": new.producer,
        },
    )
    if updated != 1:
        # Defense in depth: the guarded UPDATE is the atomic CAS — the
        # pre-check read raced a concurrent writer (single-writer stores
        # cannot reach this, but the guard keeps the semantics true).
        raise VerbatimError(
            ErrorCode.STALE_DEPENDENCY,
            f"source {source_id} control_version moved during transition",
        )

    change = {
        "superseded": "supersede",
        "corrected": "correct",
        "retracted": "retract",
        "archived": "archive",
        "recorded": "record",
        "active": "activate",
    }[disp]
    doc = new.to_dict() | {
        "change": change,
        "predecessor_head": current.mutation_head,
        "actor": actor,
        "operation_id": operation_id,
        "recorded_event": _event_us(store),
        "effect": EFFECT_KIND,
        "resolve_pending": bool(resolve_pending),
    }
    _state._write_doc(conn, store, doc, disposition=disp)
    _state._record_edges(
        conn, doc, predecessor_doc_rev=current.control_version + 1
    )
    if store is not None:
        # Invalidation handler (V5-14.10): the projection generation moves
        # inside the same commit; index/cache dependents re-fence on it.
        store.bump_generation(conn)
    return new


# ----------------------------------------------------------------------
# erasure — the closure-membership hook (V5-14.16, V5-15.03)
# ----------------------------------------------------------------------


def apply_erasure(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    producer: str,
    namespace: Optional[str] = None,
    expected_control_version: Optional[int] = None,
    expected_revision: Any = None,
    store: Any = None,
    known_at: Any = None,
) -> SourceState:
    """Fence a source's control record to the ``erased`` tombstone.

    Deletion-closure/purge paths call this inside their transaction; it
    never deletes the row (the tombstone *is* the proof of fencing) and
    never reactivates an older revision. ``expected_control_version`` /
    ``expected_revision`` implement V5-15.03's version-bound suppression:
    a stale ref conflicts (``STALE_DEPENDENCY``) rather than silently
    deleting a newer revision. Idempotent — re-erasing returns the
    tombstone unchanged. On a source with no prior record it *creates*
    the tombstone (``namespace`` then required) so suppression of an
    unregistered source still leaves a fenced artifact.
    """
    require_id(source_id, "source_id")
    _state._require_table(conn)
    if not isinstance(producer, str) or not producer:
        raise VerbatimError(ErrorCode.VALIDATION, "producer required")

    row = _state._fetch_row(conn, source_id)
    now = _time.now_us()
    known = (
        _to_us(known_at, "known_at") if known_at is not None else now
    )
    if row is not None:
        current = SourceState.from_row(row)
        if current.tombstoned:
            return current  # idempotent: already fenced
        _check_fences(
            conn,
            current,
            expected_control_version=(
                current.control_version
                if expected_control_version is None
                else expected_control_version
            ),
            expected_revision=expected_revision,
            epoch_vector=None,
        )
        if known < _to_us(current.known_at, "known_at"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "known_at cannot regress below the current record",
            )
        new = SourceState(
            source_id=source_id,
            namespace=current.namespace,
            control_version=current.control_version + 1,
            mutation_head=current.mutation_head,  # CAS anchor retained
            disposition="erased",
            superseded_by=None,
            effective_at=None,
            known_at=_rfc3339(known),
            valid_from=None,
            valid_to=None,
            updated_at=_rfc3339(now),
            producer=producer,
        )
        updated = _state._cas_update(
            conn,
            source_id,
            current.control_version,
            {
                "control_version": new.control_version,
                "disposition": new.disposition,
                "superseded_by": None,
                "effective_at": None,
                "known_at": new.known_at,
                "valid_from": None,
                "valid_to": None,
                "updated_at": new.updated_at,
                "producer": new.producer,
            },
        )
        if updated != 1:
            raise VerbatimError(
                ErrorCode.STALE_DEPENDENCY,
                f"source {source_id} control_version moved during erasure",
            )
        doc = new.to_dict() | {
            "change": "erase",
            "predecessor_head": current.mutation_head,
            "actor": None,
            "operation_id": None,
            "recorded_event": _event_us(store),
            "effect": EFFECT_KIND,
        }
        _state._write_doc(conn, store, doc, disposition="erased")
        _state._record_edges(
            conn, doc, predecessor_doc_rev=current.control_version + 1
        )
    else:
        # Erasure of a source that never had a record still leaves a
        # fenced artifact — the tombstone is the durable proof.
        if namespace is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "namespace required to fence an unregistered source",
            )
        require_id(namespace, "namespace")
        _state.register_producer(conn)
        new = SourceState(
            source_id=source_id,
            namespace=namespace,
            control_version=0,
            mutation_head=HEAD_UNRESOLVED,
            disposition="erased",
            known_at=_rfc3339(known),
            updated_at=_rfc3339(now),
            producer=producer,
        )
        _state._insert_row(
            conn, {k: v for k, v in new.to_dict().items() if k != "kind"}
        )
        doc = new.to_dict() | {
            "change": "erase",
            "predecessor_head": None,
            "actor": None,
            "operation_id": None,
            "recorded_event": _event_us(store),
            "effect": EFFECT_KIND,
        }
        _state._write_doc(conn, store, doc, disposition="erased")
        _state._record_edges(conn, doc, predecessor_doc_rev=None)
    if store is not None:
        store.bump_generation(conn)
    return new


# ----------------------------------------------------------------------
# producer-path recheck (V5-14.14) + per-revision eligibility
# ----------------------------------------------------------------------


def assert_publishable(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    expected_control_version: Optional[int] = None,
) -> SourceState:
    """Re-fence a pending producer/job before it publishes.

    Pending projection/harvest jobs call this inside their publish
    transaction: the row must still exist, not be tombstoned, and match
    ``expected_control_version`` when the job pinned one — a replacement
    or erasure that landed since the lease fails loudly instead of being
    overtaken by stale output (V5-14.14).
    """
    require_id(source_id, "source_id")
    _state._require_table(conn)
    row = _state._fetch_row(conn, source_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            "source_state not found — nothing publishable",
        )
    state = SourceState.from_row(row)
    if state.tombstoned:
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            "source_state is erased — publication fenced",
        )
    if (
        expected_control_version is not None
        and state.control_version != expected_control_version
    ):
        raise VerbatimError(
            ErrorCode.STALE_DEPENDENCY,
            f"source {source_id} control_version is "
            f"{state.control_version}, expected {expected_control_version}",
        )
    return state


def is_revision_current(
    conn: sqlite3.Connection,
    source_id: str,
    revision: Any,
    *,
    at_time: Any = None,
    known_at: Any = None,
) -> bool:
    """Whether ``revision`` is the current answer at ``at_time`` —
    the per-revision half of the publication predicate (V5-07.13)."""
    cur = _state.current_state(
        conn, source_id, at_time=at_time, known_at=known_at
    )
    if not cur.found or not cur.current or cur.effective_revision is None:
        return False
    head = _state.parse_head(revision) if not isinstance(revision, int) else revision
    return cur.effective_revision == head


__all__ = [
    "apply_erasure",
    "assert_publishable",
    "is_revision_current",
    "transition",
]
