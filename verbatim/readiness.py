"""Durable per-receipt capability readiness (SPEC_V4 §14, SPEC_V5 §08).

One row in ``readiness_obligations`` per (receipt, capability) — durable,
transactional, and queryable across restarts. ``wait_ready`` evaluates the
receipt's OWN dependency DAG; global event counters and unrelated later work
are never readiness proof (V4-14.03, C26/C88).

Capability pipeline (per capture receipt)::

    accepted → screened ─┬─ lexical_ready → ┬─ semantic_ready
                         │                  └─ derived_ready
                         ├─ source_lexical_ready   (v5 source branch —
                         └─ source_vector_ready     siblings on screened)
    failed      — independent failure indicator (V4-14.02)

``semantic_ready`` and ``derived_ready`` are siblings gated on
``lexical_ready``: embeddings enrich the semantic lane while relation/
derivation passes are structural — an embed outage must never cancel
derivation work that already committed, and a missing encoder defers
``semantic_ready`` without blocking ``derived_ready``.

The v5 source branch (``source_lexical_ready``/``source_vector_ready``)
hangs off ``screened`` as siblings of the claim branch — a source-backed
capture may declare either branch or both (``include_source``), and the
source branch never routes through claim harvest or review (V5-08.01).
``screened`` is the shared admission gate: a screening failure cancels
both branches, while a permanently deferred claim capability never
strands owed source work (V5-08.12/08.15).

States are ``ReadinessState`` values (``pending``/``running``/``succeeded``/
``failed``/``deferred``/``cancelled``), stored verbatim in the ``state``
column. Semantics fixed here:

- ``accepted`` is fulfilled in the same transaction that persists the
  capture (V4-14.01).
- ``screened`` is fulfilled when the drain-time screen gate commits (the
  ``harvest`` job) — or marked ``succeeded`` at capture for kinds excluded
  from derivation, where the inline write-channel screen is the only gate.
- ``lexical_ready`` is fulfilled when the receipt's claims commit into the
  lexical index; ``deferred`` when the pipeline settles with nothing
  admissible or the capability is unprovisioned (e.g. FTS off).
- ``semantic_ready`` is fulfilled when the last owed embed job for the
  source commits; ``deferred`` when no encoder is provisioned.
- ``derived_ready`` is fulfilled when the admit pipeline's relation/
  derivation passes commit for the receipt's claims; ``deferred`` when no
  claim exists to derive from.
- ``source_lexical_ready`` (v5) is fulfilled when the screened source
  revision's lexical projection commits for search (the ``source_project``
  job); ``deferred`` when source indexing is unprovisioned or the source
  is not admissible for indexing.
- ``source_vector_ready`` (v5) is fulfilled when the source's requested
  encoder projection commits (the ``source_embed`` job); ``deferred``
  when no encoder is provisioned. It is a sibling on ``screened`` — it
  does not wait on a claim the policy may never create (V5-08.12).
- ``failed`` fires (state ``failed``) when any stage permanently fails —
  the durable, queryable failure surface — and succeeds when the receipt
  settles with no failures.

A dependency is *settled* by ``succeeded`` OR ``deferred``: a stage whose
upstream capability was never provisioned is not thereby blocked — e.g.
``derived_ready`` may succeed while ``semantic_ready`` is deferred because
relation derivation never needed embeddings. A ``failed``/``cancelled``
dependency blocks fulfillment (``STALE_DEPENDENCY``) and cascades
cancellation to dependents.

Honesty rules (V4-14.03/05): deadline expiry returns the honest pending
snapshot (never an error, never a fabricated ready); unknown receipts raise
typed ``VerbatimError``; ``pending()`` lists only durable obligations.
"""

from __future__ import annotations

import enum
import sqlite3
import time
from typing import Any, Iterable, Optional

from .core.time import now_us
from .core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from .core.types_v4 import CapabilityName, ReadinessState
from .storage import commit_notify, repos_v4

TABLE = "readiness_obligations"

# ---------------------------------------------------------------------
# V5 source-branch capability members (SPEC_V5 §08, v5_contracts §4)
#
# ``core.types_v4.CapabilityName`` is a frozen type module owned outside
# this package, so the v5 members are registered onto the shared enum
# here at import. ``enum._proto_member.__set_name__`` performs exactly
# the registration a class-body declaration would — member maps, value
# lookup, attribute binding — so ``CapabilityName("source_lexical_ready")``,
# attribute access, iteration, and pickle round-trips all behave like a
# natively declared member. The values are LOCAL literals identical to
# ``memory.types.CAP_SOURCE_LEXICAL``/``CAP_SOURCE_VECTOR``: this module
# must stay importable without ``verbatim.memory`` (the facade package
# imports readiness; a reverse import would be a cycle).
# ---------------------------------------------------------------------

#: Capability value strings for the v5 source branch — kept identical to
#: ``verbatim.memory.types.CAP_SOURCE_LEXICAL``/``CAP_SOURCE_VECTOR``.
CAP_SOURCE_LEXICAL = "source_lexical_ready"
CAP_SOURCE_VECTOR = "source_vector_ready"


def _register_capability_member(name: str, value: str) -> CapabilityName:
    """Attach ``name``/``value`` to ``CapabilityName`` if absent.

    Idempotent — a ``types_v4`` revision that lands the member
    statically, a module reload, or a repeated import all converge on
    the same member; a conflicting pre-existing value is an INTEGRITY
    error, never a silent rebind.
    """
    existing = CapabilityName._member_map_.get(name)
    if existing is not None:
        if existing.value != value:
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                f"CapabilityName.{name} exists with value "
                f"{existing.value!r}, expected {value!r}",
            )
        return existing
    proto_type = getattr(enum, "_proto_member", None)
    if proto_type is not None:
        proto = proto_type(value)
        # ``__set_name__`` binds the real member onto the class and
        # updates _member_map_/_member_names_/_value2member_map_.
        setattr(CapabilityName, name, proto)
        proto.__set_name__(CapabilityName, name)
    else:  # pragma: no cover - fallback for enum internals without it
        member = str.__new__(CapabilityName, value)
        member._name_ = name
        member._value_ = value
        member.__objclass__ = CapabilityName
        member._sort_order_ = len(CapabilityName._member_names_)
        # setattr BEFORE _member_map_ insertion: EnumType.__setattr__
        # refuses reassigning names already registered as members.
        setattr(CapabilityName, name, member)
        CapabilityName._member_map_[name] = member
        CapabilityName._member_names_.append(name)
        CapabilityName._value2member_map_[value] = member
    return CapabilityName._member_map_[name]


_register_capability_member("SOURCE_LEXICAL_READY", CAP_SOURCE_LEXICAL)
_register_capability_member("SOURCE_VECTOR_READY", CAP_SOURCE_VECTOR)

#: The derivation-pipeline stages, in dependency order. ``failed`` is a
#: per-receipt indicator, not a pipeline stage — it is recorded alongside
#: but participates in no chain edge.
PIPELINE_CAPS: tuple[CapabilityName, ...] = (
    CapabilityName.ACCEPTED,
    CapabilityName.SCREENED,
    CapabilityName.LEXICAL_READY,
    CapabilityName.SEMANTIC_READY,
    CapabilityName.DERIVED_READY,
)
ALL_CAPS: tuple[CapabilityName, ...] = PIPELINE_CAPS + (CapabilityName.FAILED,)

#: The v5 source-projection branch (SPEC_V5 §08): siblings on
#: ``screened`` parallel to the claim branch. Opt-in per capture via
#: ``include_source``/an explicit capabilities list — deliberately NOT
#: part of ``ALL_CAPS`` so existing callers' declared obligation sets
#: stay byte-identical (legacy receipts carry no source rows).
SOURCE_CAPS: tuple[CapabilityName, ...] = (
    CapabilityName.SOURCE_LEXICAL_READY,
    CapabilityName.SOURCE_VECTOR_READY,
)

#: Every recordable capability in canonical DAG order — claim pipeline,
#: source branch, then the ``failed`` indicator last.
_KNOWN_CAPS: tuple[CapabilityName, ...] = (
    PIPELINE_CAPS + SOURCE_CAPS + (CapabilityName.FAILED,)
)

#: Claim-derivation stages the ``pipeline`` flag governs — everything
#: downstream of ``screened`` that derives claims. The source branch is
#: independent of it (V5-08.15): ``pipeline=False`` defers these while
#: declared source obligations still start ``pending``.
_CLAIM_DOWNSTREAM: frozenset[CapabilityName] = frozenset(
    {
        CapabilityName.LEXICAL_READY,
        CapabilityName.SEMANTIC_READY,
        CapabilityName.DERIVED_READY,
    }
)

#: Dependency states that permit a dependent obligation to fulfill.
_SETTLED = frozenset(
    {ReadinessState.SUCCEEDED.value, ReadinessState.DEFERRED.value}
)
#: States after which no further transition is expected.
_TERMINAL = frozenset(
    {
        ReadinessState.SUCCEEDED.value,
        ReadinessState.FAILED.value,
        ReadinessState.DEFERRED.value,
        ReadinessState.CANCELLED.value,
    }
)
#: States that can never transition again *and* cannot be re-opened —
#: ``succeeded`` work is never un-done (``_fulfill_row`` idempotents,
#: ``_fail_row``/``pend`` raise ``INVALID_TRANSITION``), ``failed`` and
#: ``cancelled`` keep their honest cause forever, and nothing in the
#: codebase ever DELETEs an obligation row. ``deferred`` is deliberately
#: excluded: ``pend`` may re-open it, so a deferred verdict is not a
#: stable fact. A snapshot restricted to capabilities whose rows are all
#: in this set — with no wanted row absent — is an immutable function of
#: immutable rows and may be memoized for the engine's lifetime.
_STABLE_STATES = frozenset(
    {
        ReadinessState.SUCCEEDED.value,
        ReadinessState.FAILED.value,
        ReadinessState.CANCELLED.value,
    }
)
#: States meaning "work is still owed" (visible in ``pending()``).
_OUTSTANDING = frozenset(
    {
        ReadinessState.PENDING.value,
        ReadinessState.RUNNING.value,
        ReadinessState.DEFERRED.value,
    }
)

#: Upper bound on the poll backoff inside ``wait_ready`` /
#: ``wait_ready_many``. ``poll_s`` remains the caller-facing cadence
#: knob; the cap only bounds how long a settle that lands *just after*
#: a poll stays undetected. Each poll is two indexed queries over the
#: receipt's own obligation rows — a 15 ms cadence is still polite —
#: while the causal barrier's budget is tens of ms, so a 50 ms tail
#: would dominate the measured wait without changing any verdict.
_POLL_CAP_S = 0.015

#: Real prerequisite edges. A stage waits on the capability whose durable
#: effects it consumes — not on every earlier stage — so a failed sibling
#: never cancels committed work (e.g. a permanent embed failure cannot
#: cancel derivation that already landed).
_PARENTS: dict[CapabilityName, tuple[CapabilityName, ...]] = {
    CapabilityName.ACCEPTED: (),
    CapabilityName.SCREENED: (CapabilityName.ACCEPTED,),
    CapabilityName.LEXICAL_READY: (CapabilityName.SCREENED,),
    CapabilityName.SEMANTIC_READY: (CapabilityName.LEXICAL_READY,),
    CapabilityName.DERIVED_READY: (CapabilityName.LEXICAL_READY,),
    # v5 source branch — siblings on ``screened`` (v5_contracts §4).
    # Source projections consume the screen gate's admission decision,
    # never the claim pipeline's outputs.
    CapabilityName.SOURCE_LEXICAL_READY: (CapabilityName.SCREENED,),
    CapabilityName.SOURCE_VECTOR_READY: (CapabilityName.SCREENED,),
    CapabilityName.FAILED: (),
}

#: Job kind → the pipeline stage obligation a permanent job failure lands on.
_JOB_STAGE: dict[str, CapabilityName] = {
    "harvest": CapabilityName.SCREENED,
    "screen": CapabilityName.SCREENED,
    "admit": CapabilityName.LEXICAL_READY,
    "embed": CapabilityName.SEMANTIC_READY,
    # v5 source lanes (V5-08.16) — a terminal projection/index failure
    # lands on the matching source-branch capability of the receipt(s)
    # for the job's (source_id, revision) refs, same convention as the
    # claim stages. ``source_backfill`` is deliberately unmapped: it is
    # maintenance work carrying its own backfill receipt (V5-08.17).
    "source_project": CapabilityName.SOURCE_LEXICAL_READY,
    "source_embed": CapabilityName.SOURCE_VECTOR_READY,
}


def job_stage_capability(kind: Any) -> Optional[CapabilityName]:
    """The receipt capability a terminal job failure lands on, or None.

    Claim stages (harvest/screen/admit/embed) and the v5 source lanes
    (``source_project``/``source_embed``) map onto the capture receipt
    DAG of the source revision the job's ``input_refs`` name — the
    commit path resolves ``(source_id, revision)`` pairs from the job
    payload either way (V5-08.16). Other kinds (purge/reindex/review/v3
    lanes, ``source_backfill``) fail their own job rows and receipts;
    they do not map onto a capture receipt's capability DAG.
    """
    try:
        return _JOB_STAGE.get(str(getattr(kind, "value", kind)))
    except Exception:
        return None


def plan_obligations(
    capabilities: Optional[Iterable[Any]] = None,
    *,
    include_source: bool = False,
) -> tuple[CapabilityName, ...]:
    """The capability set a capture declares at accept time (SPEC_V5 §08).

    The default is the legacy claim DAG — ``ALL_CAPS`` (accepted →
    screened → lexical_ready → {semantic_ready, derived_ready} plus the
    ``failed`` indicator), exactly what pre-v5 callers record.

    ``include_source=True`` adds the v5 source branch —
    ``source_lexical_ready``/``source_vector_ready`` as siblings on
    ``screened`` — so a source-backed capture's search obligations are
    declared atomically in the same transaction (V5-08.15). Callers
    decide which branch a capture carries: the claim branch, the source
    branch, or both.

    ``capabilities`` overrides the base set for captures declaring a
    subset — e.g. a source-only ``infer=False`` capture passes
    ``(ACCEPTED, SCREENED, FAILED)`` so claim stages stay *absent*
    (intentionally not requested) rather than recorded as deferred.

    Returns canonical recording order — claim pipeline, source branch,
    ``failed`` last — deduplicated. Raises ``VALIDATION`` on an empty or
    unknown capability set.
    """
    caps = (
        [_capability(c) for c in capabilities]
        if capabilities is not None
        else list(ALL_CAPS)
    )
    if include_source:
        for cap in SOURCE_CAPS:
            if cap not in caps:
                caps.append(cap)
    ordered = [c for c in _KNOWN_CAPS if c in caps]
    if not ordered:
        raise VerbatimError(
            ErrorCode.VALIDATION, "capabilities must be non-empty"
        )
    return tuple(ordered)


def _resolve_deps(
    cap: CapabilityName, recorded: set[CapabilityName]
) -> list[CapabilityName]:
    """The recorded ancestors of ``cap`` — a skipped capability's parents
    are substituted transitively so a partial ``capabilities`` list still
    yields a connected DAG."""
    out: list[CapabilityName] = []
    stack = list(_PARENTS.get(cap, ()))
    seen: set[CapabilityName] = set()
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        if parent in recorded:
            out.append(parent)
        else:
            stack.extend(_PARENTS.get(parent, ()))
    return out


def ingest_receipt_id(source_id: str, revision: int) -> str:
    """The readiness receipt id for a v2 ``Ingester.ingest`` capture.

    Deterministic over (source_id, revision) — callers derive it from the
    ``accepted`` ids on ``IngestReceipt``.
    """
    require_id(source_id, "source_id")
    if not isinstance(revision, int) or revision < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "revision must be >= 1")
    return f"rc_ingest:{source_id}:{revision}"


def envelope_pipeline_eligible(kind: Any) -> bool:
    """Whether a v3 envelope kind reaches the derivation pipeline.

    Mirrors ``evidence.envelopes._enqueue_harvest`` exactly: agent-authored
    kinds never reach claim derivation (V3-13.11), structural kinds always
    harvest, and the remainder need a harvestable coarse v2 storage class.
    """
    from .core.types_v3 import EnvelopeKind
    from .evidence.envelopes import (
        _AGENT_AUTHORED,
        _STRUCTURAL_HARVEST,
        _V2_KIND,
    )

    k = kind if isinstance(kind, EnvelopeKind) else EnvelopeKind(str(kind))
    if k in _AGENT_AUTHORED:
        return False
    if k in _STRUCTURAL_HARVEST:
        return True
    try:
        from .core.harvest import _DEFAULT_ALLOW_KINDS  # type: ignore
    except ImportError:
        return False
    return _V2_KIND.get(k) in _DEFAULT_ALLOW_KINDS


def _envelope_receipt_id(source_id: str, revision: int, kind: str) -> str:
    """The deterministic ``cr_*`` receipt id minted by ``mint_receipt``."""
    from .evidence.receipts import _receipt_id

    return _receipt_id(source_id, revision, kind)


def receipt_ids_for_source(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> list[str]:
    """Every receipt id a persisted source revision can be answered under.

    ``rc_ingest:`` for the v2 capture path plus one ``cr_*`` id per
    ``source_envelopes`` row (v3 captures). Read-only; callers filter to
    ids that actually carry obligations.
    """
    ids = [ingest_receipt_id(source_id, int(revision))]
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table'"
        " AND name='source_envelopes'"
    ).fetchone() is not None:
        rows = conn.execute(
            "SELECT DISTINCT envelope_kind FROM source_envelopes"
            " WHERE source_id = ? AND revision = ?",
            (source_id, int(revision)),
        ).fetchall()
        for (kind,) in rows:
            try:
                ids.append(_envelope_receipt_id(source_id, int(revision), kind))
            except Exception:
                # An unknown/legacy kind string is not a reason to drop the
                # receipt set — the id is derivable regardless.
                continue
    return ids


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _obligation_id(receipt_id: str, capability: str) -> str:
    return f"ro:{receipt_id}:{capability}"


def _capability(value: Any) -> CapabilityName:
    try:
        return (
            value
            if isinstance(value, CapabilityName)
            else CapabilityName(str(value))
        )
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown readiness capability {value!r}"
        ) from exc


class ReadinessEngine:
    """Durable per-receipt capability DAG over ``readiness_obligations``.

    Mutations always take the caller's ``conn`` so obligation transitions
    commit atomically with the domain effects they describe (V4-14.01).
    Reads take ``store.read()`` snapshots and return plain dicts.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._table_seen = False
        # Immutable-snapshot memo: ``(receipt_id, wanted caps, scope)``
        # → the ``_evaluate`` snapshot when every wanted capability
        # resolved to a row in ``_STABLE_STATES``. Those rows are
        # write-once facts — the snapshot is byte-identical on every
        # later observation, so re-querying is pure overhead. Anything
        # re-openable (deferred), in-flight (pending/running), or
        # not-yet-materialized (absent row / absent receipt) is never
        # memoized — it must be re-read under a fresh snapshot.
        self._snap_memo: dict[tuple, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the store carries the v4 readiness table.

        Presence is monotonic — migrations create the table and nothing
        drops it — so a True answer is memoized for the engine's
        lifetime. A False answer is re-checked every call, so a store
        migrated mid-session still becomes available.
        """
        if self._table_seen:
            return True
        with self._store.read() as conn:
            present = self._table_present(conn)
        if present:
            self._table_seen = True
        return present

    @staticmethod
    def _table_present(conn: sqlite3.Connection) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                f" AND name='{TABLE}'"
            ).fetchone()
            is not None
        )

    def _require_table(self) -> None:
        if not self.available:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"{TABLE} table absent — store schema predates v4",
            )

    # ------------------------------------------------------------------
    # recording (capture-time, inside the caller's tx)
    # ------------------------------------------------------------------

    def record_obligations(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        scope_id: str,
        capabilities: Iterable[Any] = ALL_CAPS,
        depends_on: Iterable[str] = (),
        *,
        pipeline: bool = True,
        include_source: bool = False,
    ) -> list[dict[str, Any]]:
        """Persist one obligation row per capability — atomic with capture.

        ``accepted`` commits ``succeeded`` immediately (durable acceptance
        IS the capability). Remaining pipeline stages start ``pending``
        with ``depends_on`` obligation ids following the ``_PARENTS``
        edges; ``failed`` starts ``pending`` as the receipt's failure
        indicator.

        ``pipeline=False`` covers captures excluded from derivation (e.g.
        agent-authored v3 kinds): ``screened`` is fulfilled at capture —
        the inline write-channel screen already committed — and the
        downstream CLAIM stages are ``deferred`` (never owed), so the DAG
        still terminates instead of pending forever.

        ``include_source=True`` declares the v5 source branch
        (``source_lexical_ready``/``source_vector_ready``, siblings on
        ``screened``) on top of ``capabilities`` — equivalent to passing
        them in the list; :func:`plan_obligations` computes the same
        sets. Source obligations are independent of the ``pipeline``
        flag (V5-08.15): an ``infer=False`` capture that defers claim
        production still owes source-search work, so declared source
        stages start ``pending`` under either flag. An unprovisioned or
        inadmissible projection is the jobs layer's ``defer`` decision —
        not a reason to skip the obligation row.

        Idempotent: rows are keyed ``ro:{receipt}:{capability}``; existing
        rows are left untouched, so duplicate/replayed captures converge
        and a later call may declare additional branches (e.g. source
        obligations added to a receipt that recorded the claim DAG).

        ``depends_on`` adds extra obligation ids to every non-intrinsic
        obligation (``accepted``/``failed`` never take extra deps).
        """
        require_id(receipt_id, "receipt_id")
        require_id(scope_id, "scope_id")
        caps = [_capability(c) for c in capabilities]
        if include_source:
            for cap in SOURCE_CAPS:
                if cap not in caps:
                    caps.append(cap)
        if not caps:
            raise VerbatimError(
                ErrorCode.VALIDATION, "capabilities must be non-empty"
            )
        extra = [require_id(str(d), "depends_on") for d in depends_on]
        now = now_us()
        created: list[dict[str, Any]] = []
        # Deps follow the real prerequisite edges (_PARENTS): each stage
        # waits on the capability whose effects it consumes — semantic and
        # derived are siblings on lexical, the source branch siblings on
        # screened — so a skipped capability re-links to the nearest
        # recorded ancestors (never widened).
        recorded = [c for c in PIPELINE_CAPS + SOURCE_CAPS if c in caps]
        recorded_set = set(recorded)
        for cap in recorded:
            deps = [
                _obligation_id(receipt_id, p.value)
                for p in _resolve_deps(cap, recorded_set)
            ]
            if cap is not CapabilityName.ACCEPTED:
                deps.extend(extra)
            if cap is CapabilityName.ACCEPTED:
                state = ReadinessState.SUCCEEDED.value
            elif cap is CapabilityName.SCREENED and not pipeline:
                state = ReadinessState.SUCCEEDED.value
            elif not pipeline and cap in _CLAIM_DOWNSTREAM:
                state = ReadinessState.DEFERRED.value
            else:
                state = ReadinessState.PENDING.value
            error = None
            if state == ReadinessState.DEFERRED.value:
                error = "kind_excluded_from_derivation"
            row = {
                "obligation_id": _obligation_id(receipt_id, cap.value),
                "receipt_id": receipt_id,
                "scope_id": scope_id,
                "capability": cap.value,
                "depends_on_json": json_dumps(deps),
                "state": state,
                "error": error,
                "created_us": now,
                "updated_us": now,
            }
            if (
                repos_v4.get(
                    conn, TABLE, {"obligation_id": row["obligation_id"]}
                )
                is None
            ):
                repos_v4.insert(conn, TABLE, row)
                created.append(dict(row))
        if CapabilityName.FAILED in caps:
            row = {
                "obligation_id": _obligation_id(
                    receipt_id, CapabilityName.FAILED.value
                ),
                "receipt_id": receipt_id,
                "scope_id": scope_id,
                "capability": CapabilityName.FAILED.value,
                "depends_on_json": json_dumps([]),
                "state": ReadinessState.PENDING.value,
                "error": None,
                "created_us": now,
                "updated_us": now,
            }
            if (
                repos_v4.get(
                    conn, TABLE, {"obligation_id": row["obligation_id"]}
                )
                is None
            ):
                repos_v4.insert(conn, TABLE, row)
                created.append(dict(row))
        # A ``pipeline=False`` receipt records every stage already terminal
        # — close its failure indicator now rather than leaving it pending
        # on a receipt that can never fail.
        self._close_indicator(conn, receipt_id)
        return created

    # ------------------------------------------------------------------
    # row access (inside caller conn)
    # ------------------------------------------------------------------

    def _row(
        self, conn: sqlite3.Connection, receipt_id: str, capability: Any
    ) -> Optional[dict[str, Any]]:
        cap = _capability(capability)
        return repos_v4.get(
            conn,
            TABLE,
            {
                "obligation_id": _obligation_id(receipt_id, cap.value),
                "receipt_id": receipt_id,
            },
        )

    def _row_by_id(
        self, conn: sqlite3.Connection, obligation_id: str
    ) -> Optional[dict[str, Any]]:
        return repos_v4.get(conn, TABLE, {"obligation_id": obligation_id})

    def _receipt_rows(
        self, conn: sqlite3.Connection, receipt_id: str
    ) -> list[dict[str, Any]]:
        return repos_v4.query(
            conn, TABLE, {"receipt_id": receipt_id}, order="created_us"
        )

    def _set(
        self,
        conn: sqlite3.Connection,
        row: dict[str, Any],
        state: str,
        error: Optional[str],
    ) -> dict[str, Any]:
        repos_v4.update(
            conn,
            TABLE,
            {"state": state, "error": error, "updated_us": now_us()},
            {"obligation_id": row["obligation_id"]},
        )
        row = dict(row)
        row["state"] = state
        row["error"] = error
        return row

    # ------------------------------------------------------------------
    # transitions (inside caller tx; public forms raise typed errors)
    # ------------------------------------------------------------------

    def fulfill(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        capability: Any,
        *,
        _missing_ok: bool = False,
    ) -> None:
        """Mark an obligation ``succeeded`` — dependencies must be settled.

        Every obligation in ``depends_on`` must be ``succeeded`` or
        ``deferred``; a missing dep is ``INTEGRITY``, an unsettled dep is
        ``STALE_DEPENDENCY``, and a terminal-failure state on the target
        itself is ``INVALID_TRANSITION``. Idempotent on ``succeeded``.
        """
        row = self._row(conn, receipt_id, capability)
        if row is None:
            if _missing_ok:
                return
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no obligation for {receipt_id!r}/{capability!r}",
            )
        self._fulfill_row(conn, row)

    def _fulfill_row(
        self, conn: sqlite3.Connection, row: dict[str, Any]
    ) -> bool:
        state = row["state"]
        if state == ReadinessState.SUCCEEDED.value:
            # Idempotent — still run the indicator roll-up so a receipt
            # whose stages all settled stays consistent.
            self._close_indicator(conn, row["receipt_id"])
            return True
        if state in (
            ReadinessState.FAILED.value,
            ReadinessState.CANCELLED.value,
        ):
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"obligation {row['obligation_id']} is terminally {state}",
            )
        deps = safe_json_loads(row.get("depends_on_json") or "[]") or []
        blockers: list[str] = []
        for dep_id in deps:
            dep = self._row_by_id(conn, dep_id)
            if dep is None:
                raise VerbatimError(
                    ErrorCode.INTEGRITY,
                    f"obligation {row['obligation_id']} depends on missing "
                    f"{dep_id}",
                )
            if dep["state"] not in _SETTLED:
                blockers.append(f"{dep['capability']}:{dep['state']}")
        if blockers:
            raise VerbatimError(
                ErrorCode.STALE_DEPENDENCY,
                f"obligation {row['obligation_id']} blocked by "
                f"unsettled dependencies: {blockers}",
            )
        self._set(conn, row, ReadinessState.SUCCEEDED.value, None)
        self._close_indicator(conn, row["receipt_id"])
        return True

    def try_fulfill(
        self, conn: sqlite3.Connection, receipt_id: str, capability: Any
    ) -> bool:
        """Best-effort fulfill: False when absent, terminal, or dep-blocked.

        Internal orchestration helper — the dep check still runs (no
        enforcement bypass); it reports instead of raising so a fenced
        job commit is never rolled back by a bookkeeping disagreement.
        """
        row = self._row(conn, receipt_id, capability)
        if row is None:
            return False
        try:
            return self._fulfill_row(conn, row)
        except VerbatimError:
            return False

    def fail(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        capability: Any,
        error: str,
        *,
        _missing_ok: bool = False,
    ) -> None:
        """Mark an obligation permanently ``failed`` and cascade.

        Dependents still ``pending``/``running`` transition to
        ``cancelled`` (a cancelled stage never silently pretends work
        happened), and the receipt's ``failed`` indicator fires with the
        error text. ``succeeded`` obligations cannot be failed
        (``INVALID_TRANSITION``) — committed work is not rewritten.
        """
        row = self._row(conn, receipt_id, capability)
        if row is None:
            if _missing_ok:
                return
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no obligation for {receipt_id!r}/{capability!r}",
            )
        self._fail_row(conn, row, error)

    def _fail_row(
        self, conn: sqlite3.Connection, row: dict[str, Any], error: str
    ) -> bool:
        state = row["state"]
        if state == ReadinessState.SUCCEEDED.value:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"obligation {row['obligation_id']} already succeeded",
            )
        if state == ReadinessState.FAILED.value:
            return True
        if state == ReadinessState.CANCELLED.value:
            # Already terminally blocked by an upstream failure — the
            # recorded dependency_failed error stays the honest cause.
            return True
        self._set(conn, row, ReadinessState.FAILED.value, error)
        # Cascade: a failed dependency transitively cancels everything that
        # could still run — a cancelled stage never pretends work happened.
        frontier = [row]
        seen = {row["obligation_id"]}
        while frontier:
            cur = frontier.pop()
            for dep in self._dependents(
                conn, cur["receipt_id"], cur["obligation_id"]
            ):
                if dep["obligation_id"] in seen:
                    continue
                seen.add(dep["obligation_id"])
                if dep["state"] in (
                    ReadinessState.PENDING.value,
                    ReadinessState.RUNNING.value,
                ):
                    dep = self._set(
                        conn,
                        dep,
                        ReadinessState.CANCELLED.value,
                        f"dependency_failed:{cur['capability']}",
                    )
                    frontier.append(dep)
        # The receipt-level failure indicator fires.
        ind = self._row(conn, row["receipt_id"], CapabilityName.FAILED)
        if ind is not None and ind["state"] in (
            ReadinessState.PENDING.value,
            ReadinessState.RUNNING.value,
            ReadinessState.DEFERRED.value,
        ):
            self._set(conn, ind, ReadinessState.FAILED.value, error)
        return True

    def flag_failure(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        error: str,
        *,
        _missing_ok: bool = False,
    ) -> None:
        """Fire the receipt's ``failed`` indicator without touching stages.

        Used when a pipeline job dies while sibling work for the same
        stage is still live: the receipt must report that a failure
        occurred (V4-14.07 visibility) even though the capability itself
        may still be delivered by the siblings.
        """
        ind = self._row(conn, receipt_id, CapabilityName.FAILED)
        if ind is None:
            if _missing_ok:
                return
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no obligation for {receipt_id!r}/failed",
            )
        if ind["state"] in (
            ReadinessState.PENDING.value,
            ReadinessState.RUNNING.value,
            ReadinessState.DEFERRED.value,
        ):
            self._set(conn, ind, ReadinessState.FAILED.value, error)

    def defer(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        capability: Any,
        reason: str,
        *,
        _missing_ok: bool = False,
    ) -> None:
        """Mark an obligation ``deferred`` — owed but not deliverable now.

        Terminal states are never rewritten: an already-succeeded stage
        keeps its commit, an already-failed/cancelled stage keeps its
        error. ``deferred`` is itself terminal but MAY later re-open via
        :meth:`pend` (e.g. a claim approved after the initial drain) or
        fulfill directly (activation delivers the capability).
        """
        row = self._row(conn, receipt_id, capability)
        if row is None:
            if _missing_ok:
                return
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no obligation for {receipt_id!r}/{capability!r}",
            )
        self._defer_row(conn, row, reason)

    def _defer_row(
        self, conn: sqlite3.Connection, row: dict[str, Any], reason: str
    ) -> bool:
        if row["state"] in (
            ReadinessState.PENDING.value,
            ReadinessState.RUNNING.value,
        ):
            self._set(conn, row, ReadinessState.DEFERRED.value, reason)
            self._close_indicator(conn, row["receipt_id"])
            return True
        # Terminal rows are never rewritten — but a fully-settled receipt
        # still closes its indicator through the roll-up.
        self._close_indicator(conn, row["receipt_id"])
        return False

    def state_of(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        capability: Any,
    ) -> Optional[str]:
        """The obligation's current state inside the caller's conn, or
        None when no row exists — a read for orchestration code that must
        branch on state without decoding rows."""
        row = self._row(conn, receipt_id, capability)
        return None if row is None else str(row["state"])

    def pend(
        self,
        conn: sqlite3.Connection,
        receipt_id: str,
        capability: Any,
        *,
        _missing_ok: bool = False,
    ) -> None:
        """Re-open a ``deferred`` obligation to ``pending`` (new owed work).

        Only ``deferred`` re-opens: succeeded work is not un-done, and
        failed/cancelled stages need an explicit operator path, not an
        automatic resurrection.
        """
        row = self._row(conn, receipt_id, capability)
        if row is None:
            if _missing_ok:
                return
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no obligation for {receipt_id!r}/{capability!r}",
            )
        if row["state"] == ReadinessState.DEFERRED.value:
            self._set(conn, row, ReadinessState.PENDING.value, None)
        elif row["state"] in (
            ReadinessState.PENDING.value,
            ReadinessState.RUNNING.value,
        ):
            return
        else:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"obligation {row['obligation_id']} is {row['state']},"
                " cannot re-open to pending",
            )

    def _dependents(
        self, conn: sqlite3.Connection, receipt_id: str, obligation_id: str
    ) -> list[dict[str, Any]]:
        """Obligations on this receipt that directly depend on ``obligation_id``."""
        out = []
        for row in self._receipt_rows(conn, receipt_id):
            deps = safe_json_loads(row.get("depends_on_json") or "[]") or []
            if obligation_id in deps:
                out.append(row)
        return out

    def _close_indicator(
        self, conn: sqlite3.Connection, receipt_id: str
    ) -> None:
        """Settle the ``failed`` indicator once the receipt is terminal.

        All-stage-terminal with zero failures means the receipt completed
        without a permanent failure — the indicator succeeds honestly. A
        real failure sets it directly in ``_fail_row`` before dependents
        settle, so it never reports success over a failure.
        """
        rows = self._receipt_rows(conn, receipt_id)
        ind = None
        stage_states = []
        for row in rows:
            if row["capability"] == CapabilityName.FAILED.value:
                ind = row
            else:
                stage_states.append(row["state"])
        if ind is None or ind["state"] != ReadinessState.PENDING.value:
            return
        if stage_states and all(s in _TERMINAL for s in stage_states):
            if any(
                s in (ReadinessState.FAILED.value, ReadinessState.CANCELLED.value)
                for s in stage_states
            ):
                # A terminal failure exists but the indicator never fired —
                # close it as failed rather than report a clean receipt.
                self._set(
                    conn,
                    ind,
                    ReadinessState.FAILED.value,
                    "stage_failed",
                )
            else:
                self._set(conn, ind, ReadinessState.SUCCEEDED.value, None)

    # ------------------------------------------------------------------
    # receipt → source resolution and lazy convergence
    # ------------------------------------------------------------------

    def ensure_for_source(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        revision: int,
    ) -> list[str]:
        """Return the receipt ids carrying obligations for this source —
        materializing them when the capture path did not record atomically.

        Convergence (not a second capture path): SDK/adapter captures that
        minted a ``cr_*`` receipt without wiring obligations get the same
        durable DAG at the first job/drain touch, derived from the same
        persisted rows. ``pipeline`` is evidenced by the durable job rows
        (or the envelope kind when no job row exists yet).
        """
        if not self._table_present(conn):
            return []
        candidates = receipt_ids_for_source(conn, source_id, revision)
        existing_ids = self._existing_receipts(conn, candidates)
        existing = set(existing_ids)
        missing = list(
            dict.fromkeys(rid for rid in candidates if rid not in existing)
        )
        if not missing:
            # Every resolvable receipt already carries obligations — the
            # evidence-derivation queries below exist only to fill in
            # missing rows, so a fully-materialized source skips them
            # (the jobs-table json_extract scans included).
            return existing_ids
        scope_row = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if scope_row is None:
            return existing_ids
        scope_id = scope_row[0]
        pipeline = self._pipeline_for_source(conn, source_id, revision)
        # Source-branch evidence: durable source_* job rows bound to this
        # revision prove the capture owed projection work even when its
        # obligations were never recorded (crash/SDK path). No job rows
        # → no source obligations: the branch is opt-in, and missing
        # obligations are never fabricated as owed (V5-08.17).
        src_caps = self._source_caps_for_source(conn, source_id, revision)
        # Per-envelope pipeline flags: an agent-authored kind's receipt
        # settles deferred-at-capture even when a sibling envelope on the
        # same revision is pipeline-eligible (V3-13.11 attribution).
        env_kinds = self._envelope_kinds(conn, source_id, revision)
        for rid in missing:
            rid_pipeline = pipeline
            if rid.startswith("cr_"):
                match = [
                    k for k in env_kinds
                    if rid == self._safe_cr_id(source_id, revision, k)
                ]
                if match:
                    rid_pipeline = self._kind_eligible(match[0])
            self.record_obligations(
                conn,
                rid,
                scope_id,
                list(ALL_CAPS) + src_caps,
                pipeline=rid_pipeline,
            )
            existing_ids.append(rid)
        return existing_ids

    @staticmethod
    def _envelope_kinds(
        conn: sqlite3.Connection, source_id: str, revision: int
    ) -> list[str]:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='source_envelopes'"
        ).fetchone() is None:
            return []
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT envelope_kind FROM source_envelopes"
                " WHERE source_id = ? AND revision = ?",
                (source_id, int(revision)),
            ).fetchall()
        ]

    @staticmethod
    def _safe_cr_id(source_id: str, revision: int, kind: str) -> Optional[str]:
        try:
            return _envelope_receipt_id(source_id, int(revision), kind)
        except Exception:
            return None

    def _existing_receipts(
        self, conn: sqlite3.Connection, candidates: list[str]
    ) -> list[str]:
        if not candidates:
            return []
        ph = ",".join("?" for _ in candidates)
        rows = conn.execute(
            f"SELECT DISTINCT receipt_id FROM {TABLE}"
            f" WHERE receipt_id IN ({ph})",
            candidates,
        ).fetchall()
        return [r[0] for r in rows]

    def _pipeline_for_source(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> bool:
        """True when the source revision owes derivation-pipeline work.

        A durable harvest/admit job row is the evidence — it means capture
        queued the pipeline. With no job rows the v3 envelope kind decides
        (a capture that predates job wiring still owes the work when its
        kind is harvest-eligible); a v2 source with neither owes it too.
        """
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs"
            " WHERE kind IN ('harvest','admit')"
            "   AND json_extract(input_refs_json, '$.source_id') = ?"
            "   AND json_extract(input_refs_json, '$.revision') = ?",
            (source_id, int(revision)),
        ).fetchone()
        if row is not None and row[0]:
            return True
        env_rows = conn.execute(
            "SELECT DISTINCT envelope_kind FROM source_envelopes"
            " WHERE source_id = ? AND revision = ?",
            (source_id, int(revision)),
        ).fetchall() if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='source_envelopes'"
        ).fetchone() is not None else []
        if env_rows:
            return any(
                self._kind_eligible(k) for (k,) in env_rows
            )
        # v2 capture contract: ingest always enqueues harvest — a source
        # with no job row still owes the pipeline (capture predates wiring
        # or the job row was inspected before enqueue).
        return True

    def _source_caps_for_source(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> list[CapabilityName]:
        """Source-branch capabilities evidenced by durable job rows.

        A ``source_project``/``source_embed``/``source_backfill`` job row
        bound to this revision proves the capture owed the matching
        source-projection obligation — the same evidence convention as
        ``_pipeline_for_source``. A backfill covers both projections
        (V5-08.16/08.17). No job rows → empty: the source branch is
        opt-in, so absent evidence means absent obligations, never a
        fabricated coverage promise.
        """
        rows = conn.execute(
            "SELECT DISTINCT kind FROM jobs"
            " WHERE kind IN"
            "   ('source_project','source_embed','source_backfill')"
            "   AND json_extract(input_refs_json, '$.source_id') = ?"
            "   AND json_extract(input_refs_json, '$.revision') = ?",
            (source_id, int(revision)),
        ).fetchall()
        kinds = {r[0] for r in rows}
        caps: list[CapabilityName] = []
        if kinds & {"source_project", "source_backfill"}:
            caps.append(CapabilityName.SOURCE_LEXICAL_READY)
        if kinds & {"source_embed", "source_backfill"}:
            caps.append(CapabilityName.SOURCE_VECTOR_READY)
        return caps

    @staticmethod
    def _kind_eligible(kind: str) -> bool:
        try:
            return envelope_pipeline_eligible(kind)
        except Exception:
            return False

    def _ensure_receipt(self, receipt_id: str) -> None:
        """Materialize obligations for a resolvable receipt id, if absent.

        ``rc_ingest:{sid}:{rev}`` parses directly; ``cr_*`` resolves by
        scanning ``source_envelopes`` for the envelope whose deterministic
        id matches — the digest is one-way so the scan is the honest
        reverse lookup. Unresolvable receipts surface as NOT_FOUND at the
        read layer, never as fabricated rows.
        """
        with self._store.read() as conn:
            if not self._table_present(conn):
                return
            if repos_v4.get(
                conn, TABLE, {"receipt_id": receipt_id}
            ) is not None:
                return
        resolved: Optional[tuple[str, int]] = None
        if receipt_id.startswith("rc_ingest:"):
            sid, _, rev = receipt_id[len("rc_ingest:"):].rpartition(":")
            try:
                resolved = (sid, int(rev))
            except ValueError:
                resolved = None
        elif receipt_id.startswith("cr_"):
            with self._store.read() as conn:
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table'"
                    " AND name='source_envelopes'"
                ).fetchone() is not None:
                    envs = conn.execute(
                        "SELECT DISTINCT source_id, revision, envelope_kind"
                        " FROM source_envelopes"
                    ).fetchall()
                    for sid, rev, kind in envs:
                        try:
                            rid = _envelope_receipt_id(sid, int(rev), kind)
                        except Exception:
                            continue
                        if rid == receipt_id:
                            resolved = (sid, int(rev))
                            break
        if resolved is None:
            return
        sid, rev = resolved
        try:
            # Bounded admission: this materialization runs on read paths
            # (the wait/barrier callers), where an unbounded writer-lock
            # wait — or a propagated STORE_BUSY — is wrong. A contended
            # materialization simply stays owed: the receipt reports
            # absent this poll and the owning job's own fenced commit
            # converges the rows later.
            with self._store.tx(budget_ms=5) as conn:
                if not self._table_present(conn):
                    return
                self.ensure_for_source(conn, sid, rev)
        except VerbatimError as exc:
            if exc.retryable:
                return
            raise

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    @staticmethod
    def _check_scope(
        rows: list[dict[str, Any]], scope_id: Optional[str]
    ) -> None:
        """Fail closed when a scoped caller asks across a boundary —
        a receipt in another scope is indistinguishable from unknown."""
        if scope_id is None:
            return
        if any(r["scope_id"] != scope_id for r in rows):
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                "no readiness obligations for receipt",
            )

    def receipt_state(
        self, receipt_id: str, *, scope_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Per-capability durable state for one receipt — plain dicts.

        Never inferred from other receipts or the event journal. Unknown
        receipts raise ``NOT_FOUND_OR_FORBIDDEN`` (fail closed); a missing
        table raises ``CAPABILITY_UNAVAILABLE``. ``scope_id`` binds the
        answer to the caller's partition — a mismatched receipt is
        indistinguishable from unknown (§9).
        """
        require_id(receipt_id, "receipt_id")
        self._require_table()
        self._ensure_receipt(receipt_id)
        with self._store.read() as conn:
            rows = self._receipt_rows(conn, receipt_id)
        if not rows:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                f"no readiness obligations for receipt {receipt_id!r}",
            )
        self._check_scope(rows, scope_id)
        return self._evaluate(receipt_id, rows, None)

    def _wait_commit(self, delay_s: float) -> None:
        """Bounded wait that wakes early when any commit lands on this
        store's file — the readiness settle a barrier is polling for can
        only arrive inside a commit, so a commit signal removes the
        blind-poll latency.

        V6-02.07: waits on the process-wide per-path registry
        (``commit_notify``), not the store's own ``_commit_cond`` — the
        path registry is strictly broader: it fires on commits from ANY
        ``Store`` object opened on the same file in this process,
        including this one (the managed worker drains on a second
        ``Store``; its commits would never touch this object's
        condition). Advisory only: a store without a usable path
        (``None``/``:memory:``) falls back to the per-object condition,
        any wait error degrades to the plain sleep, and spurious wakes
        cost one extra poll, never correctness — the loop re-reads and
        re-evaluates either way."""
        if delay_s <= 0.0:
            return
        path = getattr(self._store, "_path", None)
        if path and path != ":memory:":
            try:
                if commit_notify.wait(path, delay_s):
                    return
                # Timed out with no commit — the bounded wait is spent.
                return
            except Exception:
                pass  # fall through to the per-object condition
        cond = getattr(self._store, "_commit_cond", None)
        if cond is None:
            time.sleep(delay_s)
            return
        try:
            with cond:
                cond.wait(timeout=delay_s)
        except Exception:
            time.sleep(delay_s)

    def wait_ready(
        self,
        receipt_id: str,
        capabilities: Optional[Iterable[Any]] = None,
        deadline_us: Optional[int] = None,
        poll_s: float = 0.05,
        *,
        scope_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Wait on this receipt's DAG; honest pending at the deadline.

        Returns the receipt's per-capability snapshot plus ``ready`` /
        ``complete`` / ``pending`` / ``failed`` / ``deferred``. ``ready``
        means every requested capability settled without failure
        (``succeeded``, or ``deferred`` where the capability is
        unprovisioned — the state field still says which). ``complete``
        means no requested capability still has owed work. Deadline expiry
        returns the honest pending snapshot — never an error, never a
        fabricated readiness claim. ``scope_id`` binds the wait to the
        caller's partition.
        """
        require_id(receipt_id, "receipt_id")
        self._require_table()
        self._ensure_receipt(receipt_id)
        caps = (
            [_capability(c) for c in capabilities]
            if capabilities is not None
            else None
        )
        mkey = (
            (
                receipt_id,
                tuple(c.value for c in caps),
                scope_id,
            )
            if caps is not None
            else None
        )
        while True:
            if mkey is not None:
                hit = self._snap_memo.get(mkey)
                if hit is not None:
                    snap = dict(hit)
                    snap["pending"] = list(hit["pending"])
                    snap["failed"] = list(hit["failed"])
                    snap["deferred"] = list(hit["deferred"])
                    snap["states"] = {
                        k: dict(v) for k, v in hit["states"].items()
                    }
                    return snap
            with self._store.read() as conn:
                rows = self._receipt_rows(conn, receipt_id)
            if not rows:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    f"no readiness obligations for receipt {receipt_id!r}",
                )
            self._check_scope(rows, scope_id)
            snap = self._evaluate(receipt_id, rows, caps)
            if mkey is not None and all(
                snap["states"].get(c.value, {}).get("state")
                in _STABLE_STATES
                for c in caps
            ):
                stored = dict(snap)
                stored["pending"] = list(snap["pending"])
                stored["failed"] = list(snap["failed"])
                stored["deferred"] = list(snap["deferred"])
                stored["states"] = {
                    k: dict(v) for k, v in snap["states"].items()
                }
                self._snap_memo[mkey] = stored
            if snap["ready"] or snap["complete"]:
                return snap
            if deadline_us is not None and now_us() >= deadline_us:
                snap["deadline_exceeded"] = True
                return snap
            if deadline_us is not None:
                remaining = (deadline_us - now_us()) / 1_000_000.0
                self._wait_commit(
                    min(poll_s, _POLL_CAP_S, max(remaining, 0.0))
                )
            else:
                self._wait_commit(min(poll_s, _POLL_CAP_S))

    def wait_ready_many(
        self,
        receipt_ids: Iterable[str],
        capabilities: Optional[Iterable[Any]] = None,
        deadline_us: Optional[int] = None,
        poll_s: float = 0.05,
        *,
        scope_id: Optional[str] = None,
        _shared: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Batched ``wait_ready`` — one read snapshot evaluates every
        receipt per poll instead of one snapshot per receipt.

        Per-receipt semantics are unchanged: each snapshot is evaluated
        over its OWN obligation rows only (V4-14.03), ``scope_id`` still
        fails closed per receipt, deadline expiry returns honest pending
        snapshots, and resolvable-but-unrecorded receipts are
        materialized through the same ``_ensure_receipt`` path before the
        first evaluation. Two deliberate surface differences, both
        fail-closed for the caller:

        - a receipt with no obligation rows yields
          ``{..., "absent": True}`` in the map instead of raising
          ``NOT_FOUND_OR_FORBIDDEN`` — the single-call contract stays;
        - a receipt whose rows violate ``scope_id`` also reports absent
          rather than aborting the whole batch (indistinguishable from
          unknown, same as ``_check_scope``).

        ``_shared`` is a private contract for read-only consumers (the
        ``Memory.search`` barrier loop inspects snapshots, never edits
        them): returned snapshots may BE the memoized verdict objects
        rather than defensive copies. The caller must treat every value
        in the map as immutable — mutating a shared snapshot would
        corrupt the memo for every later wait.
        """
        rids: list[str] = []
        seen: set[str] = set()
        for r in receipt_ids:
            rid = require_id(r, "receipt_id")
            if rid not in seen:
                seen.add(rid)
                rids.append(rid)
        if not rids:
            return {}
        self._require_table()
        caps = (
            [_capability(c) for c in capabilities]
            if capabilities is not None
            else None
        )
        wanted = [c.value for c in caps] if caps is not None else None
        # A memo key exists only for an explicit capability set — the
        # whole-receipt snapshot (caps=None) covers whatever rows exist,
        # and a receipt can still GAIN rows later (``_declare_source_cap``
        # converges legacy captures), so it is never memoized.
        memo_caps = tuple(wanted) if wanted is not None else None

        def _copy_snap(snap: dict[str, Any]) -> dict[str, Any]:
            out = dict(snap)
            out["pending"] = list(snap["pending"])
            out["failed"] = list(snap["failed"])
            out["deferred"] = list(snap["deferred"])
            out["states"] = {k: dict(v) for k, v in snap["states"].items()}
            return out

        _out = (lambda s: s) if _shared else _copy_snap

        # Fast path — every receipt already resolved all wanted
        # capabilities to stable-terminal rows. The memoized snapshots
        # are byte-identical to what the existence check + ensure +
        # evaluation below would produce, so the read snapshot and the
        # materialization attempt are skipped entirely.
        if memo_caps is not None:
            memo_hits = [
                self._snap_memo.get((r, memo_caps, scope_id))
                for r in rids
            ]
            if all(h is not None for h in memo_hits):
                return {
                    r: _out(h) for r, h in zip(rids, memo_hits)
                }

        # Materialize resolvable receipts before the first evaluation —
        # same ensure-before-read ordering as the single-receipt path.
        with self._store.read() as conn:
            have = set(self._existing_receipts(conn, rids))
        for rid in rids:
            if rid not in have:
                self._ensure_receipt(rid)
        absent_snap = {
            "ready": False,
            "complete": False,
            "pending": [],
            "failed": [],
            "deferred": [],
            "states": {},
            "absent": True,
        }

        # Poll ramp: the first re-check runs quickly because a just-
        # committed receipt under an active drain typically settles in
        # tens of ms — a flat poll_s sleep would add up to poll_s of
        # dead time to every almost-ready barrier. Later polls widen to
        # poll_s so a genuinely slow obligation still polls politely.
        delay_s = min(poll_s, 0.005)
        while True:
            by_rid: dict[str, list[dict[str, Any]]] = {}
            scopes: dict[str, set] = {}
            snaps: dict[str, dict[str, Any]] = {}
            # Immutable verdicts come from the memo — they are byte-
            # identical to what a fresh snapshot read would produce.
            live_rids: list[str] = []
            for rid in rids:
                mkey = (
                    (rid, memo_caps, scope_id)
                    if memo_caps is not None
                    else None
                )
                hit = (
                    self._snap_memo.get(mkey)
                    if mkey is not None
                    else None
                )
                if hit is not None:
                    # Memo reference held across polls — memoized verdicts
                    # are stable-terminal and the loop never mutates them;
                    # every snap is copied once at return.
                    snaps[rid] = hit
                else:
                    live_rids.append(rid)
            if not live_rids:
                return {r: _out(s) for r, s in snaps.items()}
            with self._store.read() as conn:
                for i in range(0, len(live_rids), 200):
                    chunk = live_rids[i : i + 200]
                    ph = ",".join("?" for _ in chunk)
                    # Existence + scope evidence over ALL of the receipt's
                    # rows — obligation rows of one receipt share one
                    # scope_id, so DISTINCT pairs carry exactly what
                    # _check_scope would inspect on the unfiltered fetch.
                    for rid, sc in conn.execute(
                        f"SELECT DISTINCT receipt_id, scope_id FROM {TABLE}"
                        f" WHERE receipt_id IN ({ph})",
                        chunk,
                    ):
                        scopes.setdefault(rid, set()).add(sc)
                    # Wanted capabilities restrict the row fetch —
                    # _evaluate already treats a recorded-receipt's
                    # unrecorded capability as absent, so unread rows
                    # can never change the snapshot.
                    q = (
                        f"SELECT receipt_id, capability, state, error,"
                        f" obligation_id, updated_us FROM {TABLE}"
                        f" WHERE receipt_id IN ({ph})"
                    )
                    params: list[Any] = list(chunk)
                    if wanted is not None:
                        ph2 = ",".join("?" for _ in wanted)
                        q += f" AND capability IN ({ph2})"
                        params += wanted
                    for row in conn.execute(q, params):
                        by_rid.setdefault(row[0], []).append({
                            "receipt_id": row[0],
                            "capability": row[1],
                            "state": row[2],
                            "error": row[3],
                            "obligation_id": row[4],
                            "updated_us": row[5],
                        })
            all_done = True
            for rid in live_rids:
                rid_scopes = scopes.get(rid)
                if not rid_scopes:
                    snap = dict(absent_snap)
                    snap["receipt_id"] = rid
                    snaps[rid] = snap
                    continue
                if scope_id is not None and rid_scopes != {scope_id}:
                    # Same fail-closed verdict _check_scope produced —
                    # a receipt in another scope is indistinguishable
                    # from unknown.
                    snap = dict(absent_snap)
                    snap["receipt_id"] = rid
                    snaps[rid] = snap
                    continue
                snap = self._evaluate(rid, by_rid.get(rid) or [], caps)
                snaps[rid] = snap
                if memo_caps is not None and all(
                    snap["states"].get(cap, {}).get("state")
                    in _STABLE_STATES
                    for cap in memo_caps
                ):
                    # Every wanted capability resolved to a stable-
                    # terminal row — this verdict is a durable fact,
                    # identical on any future snapshot.
                    self._snap_memo[(rid, memo_caps, scope_id)] = (
                        _copy_snap(snap)
                    )
                if not (snap["ready"] or snap["complete"]):
                    all_done = False
            if all_done:
                return {r: _out(s) for r, s in snaps.items()}
            if deadline_us is not None and now_us() >= deadline_us:
                for snap in snaps.values():
                    if not (
                        snap.get("ready")
                        or snap.get("complete")
                        or snap.get("absent")
                    ):
                        snap["deadline_exceeded"] = True
                return {r: _out(s) for r, s in snaps.items()}
            if deadline_us is not None:
                remaining = (deadline_us - now_us()) / 1_000_000.0
                self._wait_commit(min(delay_s, max(remaining, 0.0)))
            else:
                self._wait_commit(delay_s)
            delay_s = min(delay_s * 4, poll_s, _POLL_CAP_S)

    def pending(
        self,
        scope_id: Optional[str] = None,
        *,
        states: Iterable[str] = tuple(sorted(_OUTSTANDING)),
    ) -> list[dict[str, Any]]:
        """Outstanding obligations — operator/drain visibility.

        Default states are ``pending``/``running``/``deferred``: deferred
        is owed-but-unprovisioned backlog, not completed work. Rows return
        as plain dicts with ``depends_on`` decoded beside the raw JSON.
        """
        self._require_table()
        state_set = {str(s) for s in states}
        bad = state_set - {s.value for s in ReadinessState}
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown readiness states {sorted(bad)}",
            )
        clauses = ["state IN (%s)" % ",".join("?" for _ in state_set)]
        params: list[Any] = sorted(state_set)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        with self._store.read() as conn:
            if not self._table_present(conn):
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"{TABLE} table absent — store schema predates v4",
                )
            rows = _rows(
                conn.execute(
                    f"SELECT * FROM {TABLE} WHERE {' AND '.join(clauses)}"
                    " ORDER BY created_us, obligation_id",
                    params,
                )
            )
        for row in rows:
            row["depends_on"] = (
                safe_json_loads(row.get("depends_on_json") or "[]") or []
            )
        return rows

    def pending_receipt_ids(
        self,
        receipt_ids: Iterable[str],
        *,
        scope_id: Optional[str] = None,
    ) -> set[str]:
        """The subset of ``receipt_ids`` holding at least one
        pending/running obligation row in ``scope_id`` — one chunked
        ``SELECT DISTINCT`` instead of a per-receipt snapshot.

        Semantics match the per-rid check callers replaced: a receipt
        whose only rows are foreign-scope never qualifies (the
        ``scope_id`` filter excludes them exactly like the absent
        verdicts the snapshot path produced), and a receipt with no
        rows at all likewise never qualifies. Only the compaction
        decision is answered — no snapshot fields are materialized.
        """
        rids: list[str] = []
        seen: set[str] = set()
        for r in receipt_ids:
            rid = require_id(r, "receipt_id")
            if rid not in seen:
                seen.add(rid)
                rids.append(rid)
        if not rids:
            return set()
        self._require_table()
        states = (
            ReadinessState.PENDING.value,
            ReadinessState.RUNNING.value,
        )
        out: set[str] = set()
        with self._store.read() as conn:
            if not self._table_present(conn):
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"{TABLE} table absent — store schema predates v4",
                )
            for i in range(0, len(rids), 200):
                chunk = rids[i : i + 200]
                ph = ",".join("?" for _ in chunk)
                q = (
                    f"SELECT DISTINCT receipt_id FROM {TABLE}"
                    f" WHERE receipt_id IN ({ph})"
                    f" AND state IN (?,?)"
                )
                params: list[Any] = [*chunk, *states]
                if scope_id is not None:
                    q += " AND scope_id = ?"
                    params.append(scope_id)
                for (rid,) in conn.execute(q, params):
                    out.add(str(rid))
        return out

    def pending_count(
        self,
        scope_id: Optional[str] = None,
        *,
        states: Iterable[str] = tuple(sorted(_OUTSTANDING)),
    ) -> int:
        """``len(self.pending(scope_id, states=states))`` without
        materializing the rows — the same ``state IN (...)`` filter as a
        bare ``COUNT(*)``. Drains and status paths that only report the
        depth use this so the obligation table's size never rides the
        per-pass cost."""
        self._require_table()
        state_set = {str(s) for s in states}
        bad = state_set - {s.value for s in ReadinessState}
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unknown readiness states {sorted(bad)}",
            )
        clauses = ["state IN (%s)" % ",".join("?" for _ in state_set)]
        params: list[Any] = sorted(state_set)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        with self._store.read() as conn:
            if not self._table_present(conn):
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"{TABLE} table absent — store schema predates v4",
                )
            row = conn.execute(
                f"SELECT COUNT(*) FROM {TABLE}"
                f" WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        return int(row[0]) if row else 0

    def _evaluate(
        self,
        receipt_id: str,
        rows: list[dict[str, Any]],
        caps: Optional[list[CapabilityName]],
    ) -> dict[str, Any]:
        """Snapshot evaluation over one receipt's obligation rows only."""
        by_cap = {r["capability"]: r for r in rows}
        wanted = [c.value for c in caps] if caps is not None else None
        states: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        failed: list[str] = []
        deferred: list[str] = []
        # Default scope is every capability the receipt actually recorded
        # — claim pipeline, source branch (v5), and the failed indicator.
        # Legacy receipts carry no source rows, so their snapshot is
        # byte-identical to before; source obligations are never invented
        # for them and never hidden when present.
        keys = wanted if wanted is not None else [
            c.value for c in _KNOWN_CAPS if c.value in by_cap
        ]
        for cap in keys:
            row = by_cap.get(cap)
            if row is None:
                # A capability never recorded for this receipt is absent —
                # not owed, not pending.
                states[cap] = {"state": "absent", "error": None}
                continue
            st = row["state"]
            states[cap] = {
                "state": st,
                "error": row.get("error"),
                "obligation_id": row["obligation_id"],
                "updated_us": row.get("updated_us"),
            }
            if st in (
                ReadinessState.PENDING.value,
                ReadinessState.RUNNING.value,
            ):
                pending.append(cap)
            elif st in (
                ReadinessState.FAILED.value,
                ReadinessState.CANCELLED.value,
            ):
                failed.append(cap)
            elif st == ReadinessState.DEFERRED.value:
                deferred.append(cap)
        return {
            "receipt_id": receipt_id,
            "ready": not pending and not failed,
            "complete": not pending,
            "pending": pending,
            "failed": failed,
            "deferred": deferred,
            "states": states,
        }


__all__ = [
    "ALL_CAPS",
    "CAP_SOURCE_LEXICAL",
    "CAP_SOURCE_VECTOR",
    "PIPELINE_CAPS",
    "SOURCE_CAPS",
    "ReadinessEngine",
    "envelope_pipeline_eligible",
    "ingest_receipt_id",
    "job_stage_capability",
    "plan_obligations",
    "receipt_ids_for_source",
]
