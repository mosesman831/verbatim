"""Utility-budgeted refresh scheduling (SPEC_V4_5 §08, V45-08.*;
SPEC_V4 §24/§42; SPEC_V3 §23/§40).

Periodic full reflection re-derives the whole scope every cycle —
consolidation scans every live claim and reflection scans them again —
whether or not anything changed. This module is the measured
alternative: it scans only the *changed* claim regions (claim revisions
opened or closed since the durable refresh watermark, plus explicit
owner requests and still-stale observations), orders them by a declared
utility expression, and enqueues bounded ``consolidate`` jobs through
the real :class:`JobQueue` on its background lane under a declared
budget.

Utility (V45-08.01, declared — not learned weights):

    utility = (1 + staleness) * (1 + demand) * priority
              + stale_cost + readiness_cost

- ``staleness``: event-seq units the region's oldest unhandled change
  has been waiting — deferred-readiness cost grows with age;
- ``demand``: observed access frequency — ``influence`` deliveries that
  named the region's claims;
- ``priority``: explicit owner weight (>= 1.0); an ``owner_requested``
  flag additionally marks work the budget may never evict (V45-08.02);
- ``stale_cost``: stale-answer cost — observations already marked stale
  that cite the region's claims;
- ``readiness_cost``: deferred-readiness cost — pending pipeline
  obligations on the source revisions whose spans back the region's
  claims (V45-08.03).

Owner priorities stay explicit (V45-08.02): owner-requested candidates
schedule *around* the budget — they are never silently demoted or
dropped — but they still count in the spend report, so the owner sees
exactly what their priority work cost.

Budget enforcement (V45-08.04): when ``max_jobs`` or ``max_claims``
binds, the lowest-utility non-owner candidates defer — *reported*, never
dropped: the durable watermark only advances past scheduled work, so
deferred regions re-surface on the next plan (V45-08.03).

Honesty invariants:

- Scheduling is advisory ordering only. Utility scores never rewrite
  provenance, grants, checker identity, evidence digests, or canonical
  bytes — they choose *which* ``consolidate`` jobs enqueue.
- Every job runs the production ``handle_consolidate`` path: windowed
  consolidation with the region's claim ids, plus optional bounded
  reflection — observations, derivations, and ``consolidation_pass`` /
  ``reflection_pass`` events exactly as the v4 pipeline writes them.
- The ``refresh_pass`` event records the whole plan: watermark before
  and after, scheduled vs deferred candidates with utilities and
  reasons, and measured spend (job counts, claims processed, bytes).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

from .core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from .observations.aggregate import fetch_claim_inputs
from .observations.consolidate import (
    ConsolidationWindow,
    _revision_slot_key,
    _slot_key,
    _touched_slots,
)
from .observations.freshness import current_seq
from .storage.repos import EventsRepo

#: Policy id stamped on plans, jobs, and events — measured runs name the
#: exact scheduling rules that produced them.
REFRESH_POLICY_ID = "v45.refresh.v1"

#: Audit event kind for the durable plan/spend report.
REFRESH_EVENT_KIND = "refresh_pass"

#: Producer marker carried in job input_refs so handlers and audits can
#: tell scheduler-emitted jobs from ad-hoc consolidations. The existing
#: ``handle_consolidate`` ignores unknown refs — this is bookkeeping only.
REFRESH_MARKER = "v45_refresh"

#: Bounds that keep the scan honest on large scopes.
_MAX_REGIONS = 512
_MAX_OWNER_REQUESTS = 256
_MAX_READINESS_RECEIPTS = 64
_STALENESS_CAP = 1_000_000


@dataclass(frozen=True)
class RefreshBudget:
    """Declared refresh budget for one planning cycle (V45-08.04).

    - ``max_jobs``: consolidation/reflection jobs enqueued this cycle.
    - ``max_claims``: optional bound on total claim inputs processed
      across scheduled work (bytes are reported, not budgeted).
    - ``min_utility``: optional floor — candidates below it defer even
      when job slots remain.

    Owner-requested candidates are *exempt* from every bound — priority
    is explicit and can never be silently demoted (V45-08.02) — but they
    still land in the spend report.
    """

    max_jobs: int = 8
    max_claims: Optional[int] = None
    min_utility: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_jobs, bool)
            or not isinstance(self.max_jobs, int)
            or self.max_jobs < 0
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION, "max_jobs must be a non-negative int"
            )
        if self.max_claims is not None and (
            isinstance(self.max_claims, bool)
            or not isinstance(self.max_claims, int)
            or self.max_claims < 1
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "max_claims must be a positive int or None",
            )
        if self.min_utility < 0:
            raise VerbatimError(
                ErrorCode.VALIDATION, "min_utility must be >= 0"
            )


@dataclass(frozen=True)
class RefreshCandidate:
    """One unit of refresh work, ordered by declared utility.

    ``kind`` is ``"consolidation"`` (windowed slot re-derivation),
    ``"reflection"`` (bounded rival-slot reflection, charged a full
    input scan — ``reflect()`` reads all live claims), or
    ``"periodic_full"`` (the unwindowed comparator job).
    """

    region_key: str
    kind: str
    scope_id: str
    claim_ids: Tuple[str, ...] = ()
    staleness: int = 0
    demand: int = 0
    priority: float = 1.0
    owner_requested: bool = False
    stale_cost: int = 0
    readiness_cost: int = 0
    first_change_seq: Optional[int] = None
    reasons: Tuple[str, ...] = ()
    cost_claims: int = 0
    cost_bytes: int = 0
    reflect: bool = False
    #: Declared bounded-reflection options passed verbatim into the job's
    #: ``input_refs.reflect`` — never silently dropped.
    reflect_options: Optional[dict[str, int]] = None
    seq_window: Optional[int] = None

    @property
    def utility(self) -> float:
        """Declared utility (V45-08.01): staleness × demand × priority,
        plus deferred-readiness and stale-answer costs (V45-08.03)."""
        staleness = min(max(0, int(self.staleness)), _STALENESS_CAP)
        return (
            (1.0 + staleness)
            * (1.0 + max(0, int(self.demand)))
            * max(1.0, float(self.priority))
            + max(0, int(self.stale_cost))
            + max(0, int(self.readiness_cost))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_key": self.region_key,
            "kind": self.kind,
            "claim_count": len(self.claim_ids),
            "staleness": self.staleness,
            "demand": self.demand,
            "priority": self.priority,
            "owner_requested": self.owner_requested,
            "stale_cost": self.stale_cost,
            "readiness_cost": self.readiness_cost,
            "utility": round(self.utility, 6),
            "first_change_seq": self.first_change_seq,
            "reasons": list(self.reasons),
            "cost_claims": self.cost_claims,
            "cost_bytes": self.cost_bytes,
            "reflect": self.reflect,
            "reflect_options": dict(self.reflect_options or {}),
            "seq_window": self.seq_window,
        }


@dataclass
class RefreshPlan:
    """One planning cycle: what scheduled, what deferred, what it costs."""

    scope_id: str
    watermark_before: int
    watermark_after: int
    current_seq: int
    scheduled: list[RefreshCandidate] = field(default_factory=list)
    deferred: list[RefreshCandidate] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    deferred_reasons: dict[str, str] = field(default_factory=dict)
    policy_id: str = REFRESH_POLICY_ID
    #: Emitted-job count the budget was checked against (packing-aware);
    #: ``spend()`` prefers the realized ``job_ids`` once scheduled.
    projected_jobs: int = 0

    def spend(self) -> dict[str, Any]:
        """Measured spend — counted over scheduled work, owner-priority
        included (D12: priority work still reports its cost). ``jobs``
        is the realized enqueue count when the plan was scheduled, else
        the projected packing-aware count."""
        jobs = (
            len(self.job_ids)
            if self.job_ids
            else (self.projected_jobs or len(self.scheduled))
        )
        priority_jobs = sum(1 for c in self.scheduled if c.owner_requested)
        return {
            "jobs": jobs,
            "work_items": len(self.scheduled),
            "priority_jobs": priority_jobs,
            "consolidation_jobs": sum(
                1 for c in self.scheduled if c.kind == "consolidation"
            ),
            "reflection_jobs": sum(
                1 for c in self.scheduled if c.kind == "reflection"
            ),
            "claims_processed": sum(c.cost_claims for c in self.scheduled),
            "bytes_processed": sum(c.cost_bytes for c in self.scheduled),
            "deferred": len(self.deferred),
            "deferred_claims": sum(c.cost_claims for c in self.deferred),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "policy_id": self.policy_id,
            "watermark_before": self.watermark_before,
            "watermark_after": self.watermark_after,
            "current_seq": self.current_seq,
            "scheduled": [c.as_dict() for c in self.scheduled],
            "deferred": [
                dict(c.as_dict(), deferred_reason=self.deferred_reasons.get(
                    c.region_key, "budget"
                ))
                for c in self.deferred
            ],
            "job_ids": list(self.job_ids),
            "spend": self.spend(),
        }


def _region_key(slot: tuple[str, str, str, str]) -> str:
    """Stable region id for a slot — digest over the slot key so logs
    and dedup keys stay compact and order-insensitive."""
    subj, pred, persp, cond = slot
    digest = hashlib.sha256(
        json_dumps([subj, pred, persp, cond]).encode("utf-8")
    ).hexdigest()[:16]
    return f"slot:{digest}"


def last_watermark(conn: sqlite3.Connection, scope_id: str) -> int:
    """The durable refresh watermark — the ``refresh_pass`` event's
    recorded ``watermark_after``, or 0 before the first plan."""
    row = conn.execute(
        "SELECT payload_json FROM events"
        " WHERE scope_id = ? AND kind = ?"
        " ORDER BY event_seq DESC LIMIT 1",
        (scope_id, REFRESH_EVENT_KIND),
    ).fetchone()
    if row is None:
        return 0
    payload = safe_json_loads(row[0]) or {}
    watermark = payload.get("watermark_after")
    return int(watermark) if isinstance(watermark, int) else 0


def _change_seq(conn: sqlite3.Connection, scope_id: str) -> int:
    """Latest change seq the watermark must cover.

    ``current_seq`` tracks the events/derivations/observations domain,
    but the watermark bounds ``claim_revisions.recorded_from/until`` —
    a revision stamped ahead of the last event (e.g. a writer that
    mints its own seq) is still a real change. Fold both domains so a
    fully-covered cycle can never re-discover its own work.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(r.recorded_from), 0),"
        "       COALESCE(MAX(r.recorded_until), 0)"
        " FROM claim_revisions r"
        " JOIN claims c ON c.claim_id = r.claim_id"
        " WHERE c.scope_id = ?",
        (scope_id,),
    ).fetchone()
    return max(
        current_seq(conn, scope_id),
        int(row[0] or 0),
        int(row[1] or 0),
    )


def _slot_changes(
    conn: sqlite3.Connection, scope_id: str, since_seq: int
) -> dict[tuple[str, str, str, str], int]:
    """Slot → earliest post-watermark change seq (opened or closed
    revision). The same trigger set ``consolidate_windowed`` consumes —
    computed here once so staleness is measured, not assumed."""
    rows = conn.execute(
        "SELECT c.subject_id, c.predicate, r.perspective_id,"
        "       r.condition_json, r.recorded_from, r.recorded_until"
        " FROM claims c JOIN claim_revisions r ON r.claim_id = c.claim_id"
        " WHERE c.scope_id = ?"
        "   AND (r.recorded_from > ? OR r.recorded_until > ?)"
        "   AND c.predicate IS NOT NULL AND c.subject_id IS NOT NULL",
        (scope_id, since_seq, since_seq),
    ).fetchall()
    slots: dict[tuple[str, str, str, str], int] = {}
    for subj, pred, persp, cond, opened, closed in rows:
        key = _revision_slot_key(subj, pred, persp, cond)
        seqs = [
            s for s in (opened, closed) if s is not None and s > since_seq
        ]
        if not seqs:
            continue
        first = min(seqs)
        if key not in slots or first < slots[key]:
            slots[key] = first
    return slots


def _demand_counts(
    conn: sqlite3.Connection, claim_ids: Iterable[str]
) -> dict[str, int]:
    """Deliveries per claim — observed access frequency from the
    influence ledger (V45-08.01 demand signal)."""
    ids = sorted(set(claim_ids))
    if not ids:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT object_id, COUNT(*) FROM influence"
        f" WHERE object_kind = 'claim' AND object_id IN ({ph})"
        " GROUP BY object_id",
        tuple(ids),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _stale_observation_counts(
    conn: sqlite3.Connection, scope_id: str, claim_ids: Iterable[str]
) -> dict[str, int]:
    """Stale-answer cost per claim — stale observations citing it."""
    ids = sorted(set(claim_ids))
    if not ids:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT oe.object_id, COUNT(DISTINCT oe.observation_id)"
        " FROM observation_evidence oe"
        " JOIN observations o ON o.observation_id = oe.observation_id"
        " WHERE o.scope_id = ? AND o.stale_since_seq IS NOT NULL"
        f"   AND oe.object_kind = 'claim' AND oe.object_id IN ({ph})"
        " GROUP BY oe.object_id",
        (scope_id, *ids),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _readiness_claim_ids(
    conn: sqlite3.Connection, scope_id: str
) -> set[str]:
    """Claims whose backing spans belong to source revisions with
    pending pipeline obligations — the deferred-readiness cost region
    (V45-08.03). Bounded; attribution is exact per claim."""
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='readiness_obligations'"
        ).fetchone()
        is None
    ):
        return set()
    receipts = conn.execute(
        "SELECT DISTINCT receipt_id FROM readiness_obligations"
        " WHERE scope_id = ? AND state = 'pending'"
        " LIMIT ?",
        (scope_id, _MAX_READINESS_RECEIPTS),
    ).fetchall()
    pairs: list[tuple[str, int]] = []
    for (receipt_id,) in receipts:
        if not isinstance(receipt_id, str):
            continue
        if receipt_id.startswith("rc_ingest:"):
            body = receipt_id[len("rc_ingest:"):]
            source_id, _, rev = body.rpartition(":")
            if source_id and rev.isdigit():
                pairs.append((source_id, int(rev)))
    if not pairs:
        return set()
    out: set[str] = set()
    for source_id, revision in pairs:
        rows = conn.execute(
            "SELECT DISTINCT ce.claim_id FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id"
            " WHERE s.source_id = ? AND s.revision = ?",
            (source_id, revision),
        ).fetchall()
        out.update(r[0] for r in rows)
    return out


class RefreshScheduler:
    """Plans and enqueues refresh work for one store.

    ``queue`` is the real :class:`JobQueue` — jobs land on its
    background lane under ``JobKind.CONSOLIDATE`` and drain through the
    production ``handle_consolidate`` path (windowed consolidation +
    optional bounded reflection). Nothing in this module executes
    consolidation itself; it only decides *which* durable jobs exist.
    """

    def __init__(
        self,
        store: Any,
        *,
        queue: Any = None,
        regions_per_job: int = 1,
    ) -> None:
        self.store = store
        if queue is None:
            from .jobs.queue import JobQueue

            queue = JobQueue(store)
        self.queue = queue
        if (
            isinstance(regions_per_job, bool)
            or not isinstance(regions_per_job, int)
            or regions_per_job < 1
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "regions_per_job must be a positive int",
            )
        # Consolidation regions pack this many claim_ids windows into one
        # job — identical domain work (the window touches the union of
        # slots) with strictly less queue overhead. Seq-window,
        # reflection, and owner jobs keep their own rows.
        self.regions_per_job = regions_per_job

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def plan(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        *,
        budget: Optional[RefreshBudget] = None,
        owner_requests: Iterable[Any] = (),
        priorities: Optional[dict[str, float]] = None,
        include_reflection: bool = False,
        reflect_budget: Optional[dict[str, int]] = None,
        since_seq: Optional[int] = None,
    ) -> RefreshPlan:
        """Build the refresh plan for one scope (read-only).

        ``owner_requests`` entries are ``{claim_ids: [...], weight?: f}``
        mappings — explicit owner asks that bypass every budget bound
        but still count in spend (V45-08.02). ``priorities`` maps
        ``region_key`` → owner weight for scanner-derived regions.
        ``include_reflection`` adds one scope-wide reflection candidate
        whose cost is honestly charged as a full input scan.
        """
        require_id(scope_id, "scope_id")
        budget = budget or RefreshBudget()
        priorities = dict(priorities or {})
        seq_now = _change_seq(conn, scope_id)
        since = last_watermark(conn, scope_id) if since_seq is None else int(since_seq)

        all_inputs = fetch_claim_inputs(conn, scope_id)
        total_claims = len(all_inputs)
        total_bytes = sum(len(ci.value_text.encode("utf-8")) for ci in all_inputs)

        # Slot → live claim membership (region content).
        slot_claims: dict[tuple[str, str, str, str], list[Any]] = {}
        for ci in all_inputs:
            slot_claims.setdefault(_slot_key(ci), []).append(ci)

        changed = _slot_changes(conn, scope_id, since)
        candidates: dict[str, RefreshCandidate] = {}

        readiness_claims = _readiness_claim_ids(conn, scope_id)

        def _region_key_for(
            slots: list[tuple[str, str, str, str]],
        ) -> str:
            if len(slots) == 1:
                return _region_key(slots[0])
            digest = hashlib.sha256(
                json_dumps(sorted(map(list, slots))).encode("utf-8")
            ).hexdigest()[:16]
            return f"slots:{digest}"

        def _build(
            slots: list[tuple[str, str, str, str]],
            *,
            window_ids: Optional[Iterable[str]] = None,
            first_change: Optional[int],
            reasons: list[str],
            owner_requested: bool = False,
            priority: float = 1.0,
        ) -> RefreshCandidate:
            slot_set = set(slots)
            claims = [
                ci for ci in all_inputs if _slot_key(ci) in slot_set
            ]
            seq_window: Optional[int] = None
            if not claims and first_change is not None:
                # Fully-closed region (slot emptied by deletion/
                # supersession): a claim_ids window cannot reach it —
                # _touched_slots requires a live head revision — so the
                # job falls back to a since_seq window, the same
                # production mechanism a periodic pass uses. The cost is
                # measured through the same touched-set the job will
                # compute, never estimated.
                seq_window = max(0, first_change - 1)
                touched, _ = _touched_slots(
                    conn,
                    scope_id,
                    ConsolidationWindow(since_seq=seq_window),
                )
                claims = [
                    ci for ci in all_inputs if _slot_key(ci) in touched
                ]
            cids = sorted(c.claim_id for c in claims)
            win = (
                tuple(sorted(str(i) for i in window_ids))
                if window_ids is not None
                else tuple(cids)
            )
            demand_map = _demand_counts(conn, cids)
            stale_map = _stale_observation_counts(conn, scope_id, cids)
            staleness = (
                max(0, seq_now - first_change)
                if first_change is not None
                else 0
            )
            return RefreshCandidate(
                region_key=_region_key_for(slots),
                kind="consolidation",
                scope_id=scope_id,
                claim_ids=win,
                staleness=staleness,
                demand=sum(demand_map.get(c, 0) for c in cids),
                priority=max(1.0, float(priority)),
                owner_requested=owner_requested,
                stale_cost=sum(stale_map.get(c, 0) for c in cids),
                readiness_cost=sum(
                    1 for c in cids if c in readiness_claims
                ),
                first_change_seq=first_change,
                reasons=tuple(dict.fromkeys(reasons)),
                cost_claims=len(cids),
                cost_bytes=sum(
                    len(c.value_text.encode("utf-8")) for c in claims
                ),
                seq_window=seq_window,
            )

        # 1) Changed slots since the watermark.
        for slot, first_change in sorted(changed.items()):
            key = _region_key(slot)
            candidates[key] = _build(
                [slot],
                first_change=first_change,
                reasons=["changed_slot"],
                priority=priorities.get(key, 1.0),
            )

        # 2) Explicit owner requests — never evictable (V45-08.02).
        owner_list = list(owner_requests)[:_MAX_OWNER_REQUESTS]
        for i, req in enumerate(owner_list):
            if not isinstance(req, dict):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "owner_requests entries must be mappings",
                )
            req_ids = req.get("claim_ids") or ()
            if not isinstance(req_ids, (list, tuple)):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "owner_requests.claim_ids must be a list",
                )
            req_ids = [str(c) for c in req_ids]
            if not req_ids:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "owner_requests entries need claim_ids",
                )
            weight = float(req.get("weight") or 1.0)
            # Resolve the request to the slot set its claims occupy —
            # the window names the owner's ids and the pass covers the
            # whole touched slots, so the region's cost is its real
            # membership, not just the named ids.
            member = [ci for ci in all_inputs if ci.claim_id in set(req_ids)]
            slots = sorted({_slot_key(ci) for ci in member})
            if not slots:
                # Claims may be pending/held — window on the raw ids so
                # the job still names the owner's region honestly (it
                # reports no_material_change itself).
                slots = [(f"owner:{i}", "request", "", "")]
            key = _region_key_for(slots)
            existing = candidates.get(key)
            first_change = min(
                (changed[s] for s in slots if s in changed),
                default=None,
            )
            candidates[key] = _build(
                slots,
                window_ids=req_ids,
                first_change=first_change,
                reasons=["owner_request"] + (
                    list(existing.reasons) if existing else []
                ),
                owner_requested=True,
                priority=max(
                    weight, existing.priority if existing else 1.0
                ),
            )

        # 3) Still-stale observations whose slots had no recorded change
        # — stale-answer cost keeps them in the plan (V45-08.03).
        stale_obs = conn.execute(
            "SELECT observation_id FROM observations"
            " WHERE scope_id = ? AND stale_since_seq IS NOT NULL"
            " AND recorded_until IS NULL",
            (scope_id,),
        ).fetchall()
        if stale_obs:
            ev_rows = conn.execute(
                "SELECT object_id FROM observation_evidence"
                " WHERE object_kind = 'claim' AND observation_id IN"
                " (SELECT observation_id FROM observations"
                "  WHERE scope_id = ? AND stale_since_seq IS NOT NULL"
                "  AND recorded_until IS NULL)",
                (scope_id,),
            ).fetchall()
            stale_cids = {r[0] for r in ev_rows}
            stale_slots = {
                _slot_key(ci)
                for ci in all_inputs
                if ci.claim_id in stale_cids
            }
            for slot in sorted(stale_slots):
                key = _region_key(slot)
                if key in candidates:
                    continue
                candidates[key] = _build(
                    [slot],
                    first_change=None,
                    reasons=["stale_observation"],
                    priority=priorities.get(key, 1.0),
                )

        # 4) Optional scope-wide reflection candidate — honestly charged
        # a full input scan, since ``reflect()`` reads all live claims.
        if include_reflection:
            ref_claims = tuple(sorted(c.claim_id for c in all_inputs))
            candidates["scope:reflection"] = RefreshCandidate(
                region_key="scope:reflection",
                kind="reflection",
                scope_id=scope_id,
                claim_ids=ref_claims,
                staleness=0,
                demand=sum(_demand_counts(conn, ref_claims).values()),
                priority=1.0,
                owner_requested=False,
                stale_cost=sum(
                    _stale_observation_counts(conn, scope_id, ref_claims).values()
                ),
                readiness_cost=0,
                first_change_seq=None,
                reasons=("rival_reflection",),
                cost_claims=total_claims,
                cost_bytes=total_bytes,
                reflect=True,
                reflect_options=dict(reflect_budget or {}),
            )

        ordered = sorted(
            candidates.values(),
            key=lambda c: (-c.utility, c.region_key),
        )

        # max_jobs bounds *emitted* jobs. Claim-id-windowed consolidation
        # regions pack ``regions_per_job`` per emitted job, so the budget
        # admits packable candidates until the projected job count hits
        # the bound; every other kind still emits its own job.
        rpk = self.regions_per_job
        non_owner_packable = 0
        non_owner_singleton = 0

        def _is_packable(c: RefreshCandidate) -> bool:
            return c.kind == "consolidation" and c.seq_window is None

        def _emitted(extra_packable: int, extra_singleton: int) -> int:
            packed = -(-(non_owner_packable + extra_packable) // rpk)
            return packed + non_owner_singleton + extra_singleton

        scheduled: list[RefreshCandidate] = []
        deferred: list[RefreshCandidate] = []
        deferred_reasons: dict[str, str] = {}
        claims_used = 0
        for cand in ordered:
            if cand.owner_requested:
                scheduled.append(cand)
                claims_used += cand.cost_claims
                continue
            if cand.utility < budget.min_utility:
                deferred.append(cand)
                deferred_reasons[cand.region_key] = "below_min_utility"
                continue
            packable = _is_packable(cand)
            projected = (
                _emitted(1, 0) if packable else _emitted(0, 1)
            )
            if projected > budget.max_jobs:
                deferred.append(cand)
                deferred_reasons[cand.region_key] = "max_jobs"
                continue
            if (
                budget.max_claims is not None
                and claims_used + cand.cost_claims > budget.max_claims
            ):
                deferred.append(cand)
                deferred_reasons[cand.region_key] = "max_claims"
                continue
            scheduled.append(cand)
            if packable:
                non_owner_packable += 1
            else:
                non_owner_singleton += 1
            claims_used += cand.cost_claims

        # Watermark advances only past covered work — deferred regions
        # discovered through the change scan keep their changes above
        # the watermark so they re-surface next cycle; deferred
        # stale-observation regions re-surface through their own signal
        # regardless (V45-08.03: deferral is reported, never silent).
        deferred_seqs = [
            c.first_change_seq for c in deferred
            if c.first_change_seq is not None
        ]
        watermark_after = (
            min(deferred_seqs) - 1 if deferred_seqs else seq_now
        )

        # Projected emitted jobs: non-owner packable regions share jobs
        # ``regions_per_job`` at a time; every other scheduled candidate
        # — seq-window fallbacks, reflection, and every owner request —
        # emits one job each so priority work stays explicitly
        # attributed (V45-08.02).
        projected_jobs = (
            -(-non_owner_packable // rpk)
            + non_owner_singleton
            + sum(1 for c in scheduled if c.owner_requested)
        )

        return RefreshPlan(
            scope_id=scope_id,
            watermark_before=since,
            watermark_after=watermark_after,
            current_seq=seq_now,
            scheduled=scheduled,
            deferred=deferred,
            deferred_reasons=deferred_reasons,
            projected_jobs=projected_jobs,
        )

    def periodic_plan(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        *,
        reflect: bool = True,
        reflect_budget: Optional[dict[str, int]] = None,
    ) -> RefreshPlan:
        """The periodic-full-reflection comparator plan (D11 baseline).

        One unwindowed ``consolidate`` job — a full input scan plus, when
        ``reflect`` is set, a bounded full-scope reflection charged a
        second full scan. Same event/queue path as the budgeted arm;
        only the work selection differs.
        """
        require_id(scope_id, "scope_id")
        seq_now = _change_seq(conn, scope_id)
        all_inputs = fetch_claim_inputs(conn, scope_id)
        claims = len(all_inputs)
        byts = sum(len(ci.value_text.encode("utf-8")) for ci in all_inputs)
        cost_claims = claims * (2 if reflect else 1)
        cost_bytes = byts * (2 if reflect else 1)
        cand = RefreshCandidate(
            region_key="scope:periodic_full",
            kind="periodic_full",
            scope_id=scope_id,
            claim_ids=tuple(sorted(c.claim_id for c in all_inputs)),
            staleness=0,
            demand=0,
            priority=1.0,
            owner_requested=False,
            reasons=("periodic_full",),
            cost_claims=cost_claims,
            cost_bytes=cost_bytes,
            reflect=reflect,
            reflect_options=dict(reflect_budget or {}),
        )
        return RefreshPlan(
            scope_id=scope_id,
            watermark_before=last_watermark(conn, scope_id),
            watermark_after=seq_now,
            current_seq=seq_now,
            scheduled=[cand],
            deferred=[],
        )

    # ------------------------------------------------------------------
    # scheduling — real JobQueue, real handler path
    # ------------------------------------------------------------------

    def _job_refs(self, cand: RefreshCandidate, plan: RefreshPlan) -> dict[str, Any]:
        refs: dict[str, Any] = {
            "refresh": {
                "policy_id": REFRESH_POLICY_ID,
                "region_key": cand.region_key,
                "utility": round(cand.utility, 6),
                "watermark": plan.watermark_before,
            }
        }
        if cand.kind == "periodic_full":
            if cand.reflect:
                refs["reflect"] = {
                    "max_iterations": 3,
                    **dict(cand.reflect_options or {}),
                }
            return refs
        if cand.seq_window is not None:
            # Fully-closed region: the seq window is the only trigger
            # that reaches it (claim_ids need a live head revision).
            refs["window"] = {"since_seq": cand.seq_window}
        else:
            # The claim_ids window bounds which slots re-derive and
            # which observations may retire — no max_inputs: a truncated
            # scan marks the pass partial and defers retirement
            # entirely, so bounded jobs would never clean stale answers.
            refs["window"] = {"claim_ids": list(cand.claim_ids)}
        if cand.reflect:
            refs["reflect"] = {
                "max_iterations": 3,
                "max_regions": 64,
                **dict(cand.reflect_options or {}),
            }
        return refs

    def _dedup_key(self, cand: RefreshCandidate, plan: RefreshPlan) -> bytes:
        return hashlib.blake2b(
            json_dumps(
                [
                    REFRESH_POLICY_ID,
                    plan.scope_id,
                    cand.kind,
                    cand.region_key,
                    plan.current_seq,
                ]
            ).encode("utf-8"),
            digest_size=16,
        ).digest()

    def _pack_refs(
        self, pack: list[RefreshCandidate], plan: RefreshPlan
    ) -> dict[str, Any]:
        """Job refs for a pack of claim-id-windowed regions — the
        window names the union of the pack's claims so one pass touches
        every packed slot (identical domain work to per-region jobs)."""
        claim_ids = sorted(
            {cid for c in pack for cid in c.claim_ids}
        )
        return {
            "window": {"claim_ids": claim_ids},
            "refresh": {
                "policy_id": REFRESH_POLICY_ID,
                "regions": [c.region_key for c in pack],
                "utilities": {
                    c.region_key: round(c.utility, 6) for c in pack
                },
                "watermark": plan.watermark_before,
            },
        }

    def _pack_dedup(
        self, pack: list[RefreshCandidate], plan: RefreshPlan
    ) -> bytes:
        return hashlib.blake2b(
            json_dumps(
                [
                    REFRESH_POLICY_ID,
                    plan.scope_id,
                    "pack",
                    sorted(c.region_key for c in pack),
                    plan.current_seq,
                ]
            ).encode("utf-8"),
            digest_size=16,
        ).digest()

    def schedule(
        self, conn: sqlite3.Connection, plan: RefreshPlan
    ) -> RefreshPlan:
        """Enqueue the plan's scheduled candidates on the real queue.

        Consecutive claim-id-windowed consolidation regions pack
        ``regions_per_job`` into one job; seq-window, reflection,
        periodic, and owner-requested candidates each emit their own
        job. Dedup keys bind (policy, scope, kind, region(s), seq) so a
        retried plan is idempotent and a later seq mints new work.
        Returns the plan with ``job_ids`` filled.
        """
        job_ids: list[str] = []
        pack: list[RefreshCandidate] = []

        def _is_packable(c: RefreshCandidate) -> bool:
            return (
                c.kind == "consolidation"
                and c.seq_window is None
                and not c.owner_requested
            )

        def _flush() -> None:
            if not pack:
                return
            job_ids.append(
                self.queue.enqueue(
                    conn,
                    plan.scope_id,
                    JobKind.CONSOLIDATE,
                    self._pack_refs(pack, plan),
                    dedup_key=self._pack_dedup(pack, plan),
                )
            )
            pack.clear()

        for cand in plan.scheduled:
            if _is_packable(cand):
                pack.append(cand)
                if len(pack) >= self.regions_per_job:
                    _flush()
                continue
            _flush()
            job_ids.append(
                self.queue.enqueue(
                    conn,
                    plan.scope_id,
                    JobKind.CONSOLIDATE,
                    self._job_refs(cand, plan),
                    dedup_key=self._dedup_key(cand, plan),
                )
            )
        _flush()
        plan.job_ids = job_ids
        return plan

    def refresh(
        self,
        scope_id: str,
        *,
        budget: Optional[RefreshBudget] = None,
        owner_requests: Iterable[Any] = (),
        priorities: Optional[dict[str, float]] = None,
        include_reflection: bool = False,
        reflect_budget: Optional[dict[str, int]] = None,
        since_seq: Optional[int] = None,
    ) -> RefreshPlan:
        """Plan + enqueue + audit in one transaction.

        The ``refresh_pass`` event records the full report — watermark,
        scheduled/deferred candidates with utilities and reasons, and
        measured spend — before any job runs, so the schedule is durable
        even if the drain happens later.
        """
        with self.store.tx() as conn:
            plan = self.plan(
                conn,
                scope_id,
                budget=budget,
                owner_requests=owner_requests,
                priorities=priorities,
                include_reflection=include_reflection,
                reflect_budget=reflect_budget,
                since_seq=since_seq,
            )
            self.schedule(conn, plan)
            EventsRepo(self.store).append(
                conn,
                scope_id,
                REFRESH_EVENT_KIND,
                "scheduler:v45.refresh",
                plan.as_dict(),
                REFRESH_POLICY_ID,
            )
        return plan

    def run_periodic(
        self,
        scope_id: str,
        *,
        reflect: bool = True,
        reflect_budget: Optional[dict[str, int]] = None,
    ) -> RefreshPlan:
        """The periodic-full-reflection arm, same queue + audit path.

        One unwindowed ``consolidate`` job per cycle — the D11 baseline:
        every input re-scanned every period (and again by bounded
        reflection), regardless of what changed.
        """
        with self.store.tx() as conn:
            plan = self.periodic_plan(
                conn, scope_id, reflect=reflect,
                reflect_budget=reflect_budget,
            )
            self.schedule(conn, plan)
            EventsRepo(self.store).append(
                conn,
                scope_id,
                REFRESH_EVENT_KIND,
                "scheduler:v45.periodic",
                plan.as_dict(),
                REFRESH_POLICY_ID,
            )
        return plan
