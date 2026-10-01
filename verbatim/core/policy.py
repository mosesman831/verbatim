"""Deterministic admission and transition reducer (SPEC §13, §16, §17, §23).

The reducer — never a decision backend — decides whether any lifecycle write
is permitted. Model judgments are advisory: they may *continue* a pipeline
but every automatic effect still passes deterministic checks, and model
output only ever creates review proposals, never direct transitions (SPEC §17).

Calibration posture (SPEC §26): automatic supersession stays off; vendor
probabilities are treated as uncalibrated; abstention is always a legal
outcome and nothing here fabricates a probability.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from ..storage import repos as _storage_repos
from ..storage import repos_v2 as _storage_repos_v2

from ..config import VerbatimConfig
from .claims import (
    BUILTIN_PREDICATES,
    BUILTIN_REGISTRY_NAMESPACE,
    BUILTIN_REGISTRY_VERSION,
    LITERAL_PREDICATE,
    interpretation_status,
    is_explicit_remember,
    slot_for,
)
from .harvest import SENSITIVE_RE
from .lifecycle import (
    ClaimHead,
    LifecycleMachine,
    read_claim_head,
    read_intervals,
)
from .time import overlaps
from .types import (
    ChangeSignal,
    ClaimProposal,
    DecisionRequest,
    DecisionResult,
    EndpointKind,
    ErrorCode,
    InterpretationStatus,
    Lifecycle,
    Modality,
    PairLabel,
    Polarity,
    Precision,
    Provenance,
    ReviewState,
    SourceEnvelope,
    SourceKind,
    TaskKind,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)

POLICY_VERSION = "policy-1"

#: Actor recorded on engine-initiated events; human actors carry their own id.
ENGINE_ACTOR = "verbatim-engine"

#: Comparison budget per incoming candidate (SPEC §16).
_PAIR_BUDGET = 8

_PAIR_LABELS = tuple(label.value for label in PairLabel)
_CHANGE_LABELS = tuple(label.value for label in ChangeSignal)


@dataclass(frozen=True)
class PolicyContext:
    """Everything the reducer needs: config, optional judge, policy identity.

    ``judge`` is any object exposing ``evaluate(DecisionRequest) ->
    DecisionResult``; ``None`` means the pure-rules deployment (the default
    ``offline_rules`` mode installs no backend).
    """

    cfg: VerbatimConfig
    judge: Optional[Any] = None
    policy_version: str = POLICY_VERSION
    policy_epoch: int = 0
    #: Facts available at admission time for three-valued Condition checks
    #: (SPEC_V2 §17). ``None`` means no context was supplied — every
    #: conditional leaf then evaluates UNKNOWN, which is preserved, never
    #: silently treated as true or false.
    admission_context: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AdmissionOutcome:
    """Result of the admission pipeline; never raises for policy denials."""

    claim_id: Optional[str]
    state: Lifecycle
    review_id: Optional[str]
    reason: str
    decision_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AdmitPlan:
    """Admission inputs computed OUTSIDE the write transaction (V4-09.01).

    Judge evaluation, the sensitivity pre-scan, and the three-valued
    condition check never hold a write tx (V4-09.09); ``_admit_apply``
    consumes the plan inside the caller's fenced commit so the lease,
    dependency re-verification, domain effects, receipt, and job
    completion are one atomic unit.
    """

    early: Optional[AdmissionOutcome] = None
    sensitive: bool = False
    durability: Optional[bool] = None
    judge_req: Optional[DecisionRequest] = None
    judge_res: Optional[DecisionResult] = None
    condition_eval: Optional[bool] = None
    condition_keys: tuple[str, ...] = ()


def _repos(store: Any) -> SimpleNamespace:
    """Repository facade over ``store`` (assumed storage API).

    Isolated so a drifted storage API is adapted in exactly one place.
    """
    return SimpleNamespace(
        claims=_storage_repos.ClaimsRepo(store),
        events=_storage_repos.EventsRepo(store),
        edges=_storage_repos.EdgesRepo(store),
        reviews=_storage_repos.ReviewsRepo(store),
        # Class (not instance): ``repos.decisions(store)`` builds it lazily at
        # the recording call site, matching the other repos' signature style.
        decisions=_storage_repos.DecisionsRepo,
    )


def _ensure_scope(store: Any, conn: sqlite3.Connection, scope: Any) -> str:
    """Scope rows are authorization partitions; create idempotently.

    Delegates to the storage layer's ``ensure_scope`` so the row's
    canonicalization (including ``profile_id``) stays owned by storage.
    """
    return _storage_repos.ensure_scope(store, conn, scope)


def _stored_scope_id(conn: sqlite3.Connection, envelope: Any) -> Optional[str]:
    """The persisted ``sources.scope_id`` for a rebuilt envelope, if any.

    v3 scope ids are opaque partition tokens that need not equal
    ``scope_key(scope)`` — when the envelope was rebuilt from storage the
    stored value is authoritative so derived objects stay inside the
    source's actual partition (and its purge closure)."""
    sid = getattr(envelope, "source_id", None)
    if not sid:
        return None
    try:
        row = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
        ).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row is not None else None


def _claim_for_primary_span(
    conn: sqlite3.Connection, span_id: str
) -> Optional[str]:
    """Earliest claim citing ``span_id`` as primary evidence, if any.

    ``admit`` writes the proposal's first span with role ``primary``, so
    that link is the durable dedup key for the admission itself: a
    redelivered admission for the same span resolves to the claim the
    first attempt already committed — a crash after the claim commit but
    before the job receipt cannot mint a duplicate (the operations
    ledger cannot reach inside ``admit``'s own transaction).
    """
    row = conn.execute(
        "SELECT ce.claim_id FROM claim_evidence ce"
        " JOIN claims c ON c.claim_id = ce.claim_id"
        " WHERE ce.span_id = ? AND ce.evidence_role = 'primary'"
        " ORDER BY c.created_event ASC, ce.claim_id ASC LIMIT 1",
        (span_id,),
    ).fetchone()
    return str(row[0]) if row is not None else None


# ---------------------------------------------------------------------------
# Predicate registry (SPEC_V2 §13.09, §16)
# ---------------------------------------------------------------------------


def _registry_name(predicate: Optional[str]) -> Optional[str]:
    """Map a claim predicate to its registry entry name.

    ``literal:<key>`` predicates resolve to the single ``literal`` family
    entry; unknown predicates stay ``None`` so nothing is coerced into a
    near-enough slot.
    """
    if predicate is None:
        return None
    if predicate.startswith("literal:"):
        # Same well-formedness gate as ``slot_for`` — a malformed literal key
        # is an unknown predicate, not a literal-family member.
        return LITERAL_PREDICATE if slot_for(predicate) is not None else None
    if predicate in {name for name, *_ in BUILTIN_PREDICATES}:
        return predicate
    return None


def _seed_builtin_registry(store: Any, conn: sqlite3.Connection) -> None:
    """Idempotently register the shipped ``vb`` predicate table (SPEC_V2 §16).

    Runs inside the caller's write transaction on first use; a replay is a
    no-op because every entry is checked before insert. Never touches other
    namespaces — third parties register their own.
    """
    reg = _storage_repos_v2.PredicateRegistryRepo(store)
    for name, cardinality, mutable, sensitive in BUILTIN_PREDICATES:
        exists = conn.execute(
            "SELECT 1 FROM predicate_definitions"
            " WHERE namespace = ? AND name = ? AND version = ?",
            (BUILTIN_REGISTRY_NAMESPACE, name, BUILTIN_REGISTRY_VERSION),
        ).fetchone()
        if exists is None:
            reg.register(
                conn,
                BUILTIN_REGISTRY_NAMESPACE,
                name,
                version=BUILTIN_REGISTRY_VERSION,
                cardinality=cardinality,
                mutable=mutable,
                sensitive=sensitive,
                authority_policy="builtin",
            )


def _registry_slot(
    conn: sqlite3.Connection, predicate: Optional[str]
) -> Optional[dict[str, Any]]:
    """Latest-version registry row → slot dict, or ``None``.

    Read on the caller's transaction so freshly seeded rows are visible.
    Unknown predicates return ``None`` — conservative, never coerced.
    """
    name = _registry_name(predicate)
    if name is None:
        return None
    row = conn.execute(
        "SELECT cardinality, mutable, sensitive, version FROM predicate_definitions"
        " WHERE namespace = ? AND name = ? ORDER BY version DESC LIMIT 1",
        (BUILTIN_REGISTRY_NAMESPACE, name),
    ).fetchone()
    if row is None:
        return None
    return {
        "multi_valued": row[0] == "set",
        "mutable": bool(row[1]),
        "sensitive": bool(row[2]),
        "registry_version": row[3],
    }


def _stamp_admission_v2(
    conn: sqlite3.Connection,
    claim_id: str,
    revision: int,
    proposal: ClaimProposal,
    subject_id: Optional[str],
    istat: str,
    registry_version: Optional[int],
) -> None:
    """Fill v2 columns the v1 storage signature does not write.

    Same-transaction patch after ``ClaimsRepo.add_revision``:
    ``claim_revisions`` gains subject/predicate/registry/interpretation/method
    lineage (SPEC_V2 §13.09) and ``valid_intervals`` gains the full endpoint
    metadata of the proposal's applicability interval (SPEC_V2 §16).
    """
    iv = proposal.valid
    conn.execute(
        "UPDATE claim_revisions SET subject_id = ?, predicate = ?,"
        " registry_version = ?, interpretation_status = ?, method = ?"
        " WHERE claim_id = ? AND revision = ?",
        (
            subject_id,
            proposal.predicate,
            registry_version,
            istat,
            proposal.method,
            claim_id,
            revision,
        ),
    )
    conn.execute(
        "UPDATE claims SET interpretation_status = ? WHERE claim_id = ?",
        (istat, claim_id),
    )
    conn.execute(
        "UPDATE valid_intervals SET start_kind = ?, end_kind = ?,"
        " from_us_hi = ?, until_us_hi = ?"
        " WHERE claim_id = ? AND revision = ? AND interval_no = 0",
        (
            iv.start_kind.value,
            iv.end_kind.value,
            iv.from_us_hi,
            iv.until_us_hi,
            claim_id,
            revision,
        ),
    )


def _register_evidence_family(
    store: Any,
    conn: sqlite3.Connection,
    scope_id: str,
    claim_id: str,
    proposal: ClaimProposal,
) -> Optional[str]:
    """Link claims whose evidence bytes *and* payload representation match.

    SPEC_V2 §18: copies of the same statement collapse for ranking while
    remaining distinct evidence records — a family never merges or erases the
    member claims/spans. Returns the family id when a duplicate was found.

    Idempotent: family identity derives from the first-seen claim id and
    membership inserts are ``INSERT OR IGNORE``, so a replay under the same
    operation converges to the same rows.
    """
    if not proposal.evidence:
        return None
    span_ids = [s.span_id for s in proposal.evidence]
    ph = ", ".join("?" for _ in span_ids)
    new_hmacs = {
        r[0]
        for r in conn.execute(
            f"SELECT excerpt_hmac FROM spans WHERE span_id IN ({ph})",
            tuple(span_ids),
        ).fetchall()
    }
    if not new_hmacs:
        return None
    hph = ", ".join("?" for _ in new_hmacs)
    rows = conn.execute(
        f"SELECT DISTINCT ce.claim_id, s.excerpt_hmac FROM claim_evidence ce"
        f" JOIN spans s ON s.span_id = ce.span_id"
        f" JOIN claims c ON c.claim_id = ce.claim_id"
        f" WHERE ce.evidence_role = 'primary' AND ce.claim_id <> ?"
        f"   AND c.scope_id = ? AND s.excerpt_hmac IN ({hph})",
        (claim_id, scope_id, *new_hmacs),
    ).fetchall()
    new_obj = json_dumps(proposal.object_json) if proposal.object_json else None
    partner: Optional[str] = None
    for other_id, _hmac in rows:
        head = read_claim_head(conn, other_id)
        if head is None:
            continue
        if head.object_json == new_obj:
            partner = other_id
            break
    if partner is None:
        return None

    fams = _storage_repos_v2.EvidenceFamiliesRepo(store)
    existing = conn.execute(
        "SELECT f.family_id FROM evidence_families f"
        " JOIN family_members m ON m.family_id = f.family_id"
        " WHERE m.object_kind = 'claim' AND m.object_id = ?",
        (partner,),
    ).fetchone()
    if existing is not None:
        family_id = existing[0]
    else:
        digest = hashlib.sha256(f"family:{partner}".encode("utf-8")).hexdigest()
        family_id = f"ef_{digest[:32]}"
        if (
            conn.execute(
                "SELECT 1 FROM evidence_families WHERE family_id = ?",
                (family_id,),
            ).fetchone()
            is None
        ):
            fams.create(conn, scope_id, "claim", partner, family_id=family_id)
        fams.add_member(conn, family_id, "claim", partner, role="origin")
    fams.add_member(conn, family_id, "claim", claim_id, role="copy")
    for sid in span_ids:
        fams.add_member(conn, family_id, "span", sid, role="copy")
    return family_id


def _conflict_group_for(
    conn: sqlite3.Connection, scope_id: str, a: str, b: str
) -> str:
    """Open conflict group containing both claims (SPEC_V2 §19).

    Joins an existing open group when either claim already belongs to one —
    conflicts are relation clusters, not disjoint pairs — else creates one.
    Membership is ``INSERT OR IGNORE`` so replays converge.
    """
    row = conn.execute(
        "SELECT m.group_id FROM conflict_members m"
        " JOIN conflict_groups g ON g.group_id = m.group_id"
        " WHERE g.status = 'open' AND m.claim_id IN (?, ?) LIMIT 1",
        (a, b),
    ).fetchone()
    if row is not None:
        group_id = row[0]
    else:
        digest = hashlib.sha256(
            f"conflict:{scope_id}:{a}:{b}".encode("utf-8")
        ).hexdigest()
        group_id = f"kg_{digest[:32]}"
        conn.execute(
            "INSERT OR IGNORE INTO conflict_groups"
            "(group_id, scope_id, status, dimensions_json)"
            " VALUES (?, ?, 'open', NULL)",
            (group_id, scope_id),
        )
    for cid in (a, b):
        conn.execute(
            "INSERT OR IGNORE INTO conflict_members(group_id, claim_id)"
            " VALUES (?, ?)",
            (group_id, cid),
        )
    return group_id


def _request_hmac(req: DecisionRequest) -> bytes:
    """Integrity digest of the decision inputs (staleness checks, §20).

    SHA-256 rather than a keyed HMAC: this is a request fingerprint, not a
    persisted low-entropy content dedup key.
    """
    canonical = json_dumps(
        {
            "task": req.task.value,
            "state": req.state,
            "labels": list(req.allowed_labels),
            "rubric": req.rubric_version,
        }
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _record_decision(
    ctx: PolicyContext,
    store: Any,
    repos: SimpleNamespace,
    conn: sqlite3.Connection,
    scope_id: str,
    req: DecisionRequest,
    res: DecisionResult,
    inputs: Sequence[tuple[str, str, Optional[int]]],
) -> str:
    """Persist an advisory outcome under the decisions schema (SPEC §20).

    The decision row plus its content-hashed inputs commit in the caller's
    transaction so a stale-result check can never observe half a record.
    Callers pass ``(object_kind, object_id, revision)``; the storage schema's
    fourth column (``content_hmac``) is reserved for dedup integrity and left
    NULL here — fabricating a fingerprint would be worse than omitting one.
    """
    normalized = [
        (kind, object_id, revision, None) for kind, object_id, revision in inputs
    ]
    return repos.decisions(store).record(
        conn,
        scope_id,
        req.task,
        _request_hmac(req),
        res.backend or "unknown",
        res.model_revision,
        res.rubric_version or req.rubric_version,
        req.policy_epoch,
        {
            "outcome": res.outcome,
            "abstained": res.abstained,
            "reason": res.reason,
            "usage": res.usage,
        },
        normalized,
    )


def _evaluate(
    ctx: PolicyContext, req: DecisionRequest
) -> Optional[DecisionResult]:
    """Call the judge defensively; a backend failure is an abstention, never
    a crash and never an implicit yes (SPEC §25)."""
    if ctx.judge is None:
        return None
    try:
        res = ctx.judge.evaluate(req)
    except Exception:
        return DecisionResult(
            task=req.task,
            backend=getattr(ctx.judge, "backend", "unknown"),
            model_revision=None,
            rubric_version=req.rubric_version,
            outcome={},
            abstained=True,
            reason="backend_error",
        )
    if not isinstance(res, DecisionResult):
        return DecisionResult(
            task=req.task,
            backend="unknown",
            model_revision=None,
            rubric_version=req.rubric_version,
            outcome={},
            abstained=True,
            reason="invalid_result",
        )
    return res


def _durable_vote(res: Optional[DecisionResult]) -> Optional[bool]:
    """Map a durability result conservatively (SPEC §26).

    ``True``  — rule_match or an explicit 'durable' label: pipeline continues.
    ``False`` — rule_no_match / 'not_durable' label: route to review.
    ``None``  — abstention or unparseable outcome: route to review.
    """
    if res is None or res.abstained:
        return None
    out = res.outcome or {}
    for key in ("label", "choice", "rule", "outcome", "answer"):
        v = out.get(key)
        if v in ("durable", "rule_match", True):
            return True
        if v in ("not_durable", "rule_no_match", False):
            return False
    return None


def _pair_label_of(res: Optional[DecisionResult]) -> Optional[PairLabel]:
    if res is None or res.abstained:
        return None
    out = res.outcome or {}
    for key in ("label", "choice", "outcome", "answer"):
        v = out.get(key)
        if isinstance(v, str):
            try:
                return PairLabel(v)
            except ValueError:
                continue
    return None


def _change_signal_of(res: Optional[DecisionResult]) -> Optional[ChangeSignal]:
    if res is None or res.abstained:
        return None
    out = res.outcome or {}
    for key in ("label", "choice", "outcome", "answer"):
        v = out.get(key)
        if isinstance(v, str):
            try:
                return ChangeSignal(v)
            except ValueError:
                continue
    return None


def _span_text(envelope: SourceEnvelope, proposal: ClaimProposal) -> Optional[str]:
    """Exact quotation of the proposal's primary span; ``None`` if the span
    does not decode — never a guessed substring."""
    if not proposal.evidence:
        return None
    span = proposal.evidence[0]
    try:
        return envelope.payload[span.start_byte : span.end_byte].decode("utf-8")
    except (UnicodeDecodeError, IndexError):
        return None


def admit(
    store: Any,
    proposal: ClaimProposal,
    envelope: SourceEnvelope,
    ctx: Optional[PolicyContext] = None,
) -> AdmissionOutcome:
    """Admission pipeline (SPEC §13): capture permission → sensitive hint →
    optional durability judgment → require_review / rules path.

    Order of the outcome ladder (first hit wins):

    1. capture denied            → rejected outcome, nothing written
    2. sensitive hint            → pending + review (never auto-activated)
    3. condition provably false  → pending + review (SPEC_V2 §17)
    4. judge says not-durable    → pending + review
    5. ``require_review`` config → pending + review
    6. explicit "remember …"     → active (privacy/identity checks already ran)
    7. whitelisted pattern       → active
    8. otherwise (abstain)       → pending + review

    Slot behavior is consulted against the durable ``vb`` predicate registry
    (seeded idempotently on first use) with the static table as fallback;
    unknown predicates stay unslotted. The write transaction additionally
    stamps v2 interpretation/registry lineage, the full interval endpoint
    metadata, and evidence-family registration for byte-identical duplicates.

    Every written path runs in one ``store.tx()`` and appends its audit event.
    """
    if ctx is None:
        ctx = PolicyContext(cfg=VerbatimConfig())
    plan = _admit_evaluate(store, proposal, envelope, ctx)
    if plan.early is not None:
        return plan.early
    with store.tx() as conn:
        return _admit_apply(store, conn, proposal, envelope, ctx, plan)


def _admit_evaluate(
    store: Any,
    proposal: ClaimProposal,
    envelope: SourceEnvelope,
    ctx: PolicyContext,
) -> _AdmitPlan:
    """Everything ``admit`` decides before the write transaction opens.

    The durable-registry pre-read, the sensitivity hint, the optional judge
    call, and the three-valued condition check are all produced here so no
    backend ever runs while a write tx is held (V4-09.09). Early capture
    denials short-circuit as ``plan.early`` — they write nothing.
    """
    # --- 1. capture permission ----------------------------------------------
    cap = ctx.cfg.capture
    # Explicit operator kinds bypass the automatic-capture flag (SPEC §10);
    # their consent is the act of importing them.
    explicit = envelope.source_kind in (
        SourceKind.IMPORT,
        SourceKind.OPERATOR_RECORD,
    )
    kind_allowed = {
        SourceKind.USER_MESSAGE: cap.user_messages,
        SourceKind.ASSISTANT_MESSAGE: cap.assistant_context,
        SourceKind.TOOL_OUTPUT: cap.tool_outputs,
        SourceKind.IMPORT: True,
        SourceKind.OPERATOR_RECORD: True,
    }.get(envelope.source_kind, False)
    if not explicit and (not cap.enabled or not kind_allowed):
        return _AdmitPlan(
            early=AdmissionOutcome(
                None,
                Lifecycle.REJECTED,
                None,
                "capture_disabled" if not cap.enabled else "capture_kind_not_allowed",
            )
        )
    if (
        not explicit
        and envelope.provenance == Provenance.ASSISTANT_GENERATED
        and not cap.assistant_context
    ):
        return _AdmitPlan(
            early=AdmissionOutcome(
                None, Lifecycle.REJECTED, None, "capture_provenance_not_allowed"
            )
        )

    text = _span_text(envelope, proposal) or ""

    # --- 2. sensitive hint ----------------------------------------------------
    # The durable registry (once seeded) overrides the static slot table for
    # sensitivity — an operator tightening a predicate must not wait for a
    # code change. Read on a snapshot before the write tx; the tx re-checks.
    slot = slot_for(proposal.predicate)
    try:
        with store.read() as rconn:
            reg_pre = _registry_slot(rconn, proposal.predicate)
    except (sqlite3.Error, AttributeError):
        reg_pre = None
    if reg_pre is not None:
        slot = reg_pre
    sensitive = bool(SENSITIVE_RE.search(text)) or bool(slot and slot["sensitive"])

    # --- 3. optional durability judgment (outside any write transaction) -----
    durability: Optional[bool] = None
    judge_res: Optional[DecisionResult] = None
    judge_req: Optional[DecisionRequest] = None
    if ctx.judge is not None and not sensitive:
        judge_req = DecisionRequest(
            task=TaskKind.DURABILITY,
            state={
                "text": text,
                "predicate": proposal.predicate,
                "modality": proposal.modality.value,
                "negated": proposal.polarity == Polarity.NEGATED,
                "has_condition": proposal.condition is not None,
            },
            allowed_labels=("durable", "not_durable"),
            deadline_s=8.0,
            policy_epoch=ctx.policy_epoch,
            purpose="candidate_curation",
        )
        judge_res = _evaluate(ctx, judge_req)
        durability = _durable_vote(judge_res)

    # --- three-valued condition check (SPEC_V2 §17) ---------------------------
    # The admission context may satisfy or violate a proposal's condition.
    # UNKNOWN is preserved — it is annotated on the interpretation but does
    # not, by itself, change routing (absence of evidence is not evidence of
    # violation). A provably FALSE condition is different: the proposition
    # cannot hold under the supplied context, so it is never auto-activated.
    condition_eval: Optional[bool] = None
    condition_keys: tuple[str, ...] = ()
    if proposal.condition is not None:
        condition_keys = tuple(proposal.condition.required_keys())
        condition_eval = proposal.condition.evaluate(ctx.admission_context or {})

    return _AdmitPlan(
        sensitive=sensitive,
        durability=durability,
        judge_req=judge_req,
        judge_res=judge_res,
        condition_eval=condition_eval,
        condition_keys=condition_keys,
    )


def _admit_apply(
    store: Any,
    conn: sqlite3.Connection,
    proposal: ClaimProposal,
    envelope: SourceEnvelope,
    ctx: PolicyContext,
    plan: _AdmitPlan,
) -> AdmissionOutcome:
    """The atomic write phase of ``admit`` on the CALLER's transaction.

    Every durable effect — claim rows, evidence links, the audit event, the
    recorded decision, the review — lands in ``conn`` and commits with the
    caller's commit, so a lease fence (or dependency re-verification) run
    earlier in the same transaction governs the whole set (V4-09.02/09.03).
    """
    if plan.early is not None:
        return plan.early
    repos = _repos(store)
    scope = envelope.scope
    sensitive = plan.sensitive
    durability = plan.durability
    judge_res = plan.judge_res
    judge_req = plan.judge_req
    condition_eval = plan.condition_eval
    condition_keys = plan.condition_keys
    decision_ids: tuple[str, ...] = ()
    # --- atomic write (caller's transaction) ---
    # The stored source's partition is authoritative for derived
    # objects; ``_ensure_scope`` is the fallback (identical for
    # ordinary v2 writes) and keeps the row-creation side effect.
    scope_id = _stored_scope_id(conn, envelope) or _ensure_scope(
        store, conn, scope
    )
    # --- retried-admission convergence -------------------------------------
    # A redelivery of the same primary-evidence span must resolve to the
    # claim the first attempt committed — otherwise a crash after the
    # claim commit (but before the job receipt landed) mints a duplicate
    # claim plus duplicate reviews. The span→claim link is the durable
    # dedup key, mirroring ``api_ingest.remember``'s ``_claim_for_span``
    # heal. An erased predecessor is different: the proposal is refused
    # rather than resurrecting scrubbed content under a fresh claim id.
    if proposal.evidence:
        prior_id = _claim_for_primary_span(
            conn, proposal.evidence[0].span_id
        )
        if prior_id is not None:
            prior_head = read_claim_head(conn, prior_id)
            if prior_head is None or prior_head.state == Lifecycle.ERASED:
                return AdmissionOutcome(
                    None,
                    Lifecycle.REJECTED,
                    None,
                    "prior_claim_erased",
                )
            prior_review = repos.reviews.find_open(
                conn,
                scope_id,
                {"effect": "admit", "claim_id": prior_id},
            )
            return AdmissionOutcome(
                prior_id,
                prior_head.state,
                prior_review,
                "already_admitted",
            )
    _seed_builtin_registry(store, conn)
    # Registry-consulted slot behavior (SPEC_V2 §16): the durable
    # predicate table wins when it knows the predicate; the static table
    # is the fallback for rows not yet registered. Unknown predicates
    # stay unslotted — nothing coerces them.
    reg_slot = _registry_slot(conn, proposal.predicate)
    eff_slot = (
        reg_slot if reg_slot is not None else slot_for(proposal.predicate)
    )
    registry_version = (
        reg_slot["registry_version"] if reg_slot is not None else None
    )

    # --- outcome decision -------------------------------------------------
    reason: str
    target_state: Lifecycle
    need_review: bool
    if sensitive:
        reason, target_state, need_review = (
            "sensitive_hint",
            Lifecycle.PENDING,
            True,
        )
    elif condition_eval is False:
        reason, target_state, need_review = (
            "condition_violated",
            Lifecycle.PENDING,
            True,
        )
    elif durability is False:
        reason, target_state, need_review = (
            "judge_not_durable",
            Lifecycle.PENDING,
            True,
        )
    elif durability is None and ctx.judge is not None:
        reason = (
            "judge_abstain"
            if (judge_res and judge_res.abstained)
            else "judge_no_label"
        )
        target_state, need_review = Lifecycle.PENDING, True
    elif ctx.cfg.admission.require_review:
        reason, target_state, need_review = (
            "review_required",
            Lifecycle.PENDING,
            True,
        )
    elif is_explicit_remember(proposal):
        reason, target_state, need_review = (
            "explicit_remember",
            Lifecycle.ACTIVE,
            False,
        )
    elif (
        eff_slot is not None
        and proposal.modality == Modality.ASSERTED
        and proposal.predicate is not None
    ):
        reason, target_state, need_review = (
            "rule_whitelist",
            Lifecycle.ACTIVE,
            False,
        )
    else:
        reason, target_state, need_review = (
            "abstained",
            Lifecycle.PENDING,
            True,
        )

    seq = repos.events.append(
        conn,
        scope_id,
        "claim_proposed",
        ENGINE_ACTOR,
        {
            "predicate": proposal.predicate,
            "method": proposal.method,
            "reason": reason,
            "provenance": envelope.provenance.value,
        },
        ctx.policy_version,
    )
    subject_id = proposal.subject_entity_id or envelope.speaker_id
    claim_id = repos.claims.create(
        scope_id,
        subject_id,
        proposal.predicate,
        conn,
    )
    interp = {
        "proposer": proposal.method,
        "policy_version": ctx.policy_version,
        "valid_basis": proposal.valid.basis,
        "source_kind": envelope.source_kind.value,
        "provenance": envelope.provenance.value,
    }
    if condition_eval is not None or condition_keys:
        interp["condition_eval"] = (
            "unknown" if condition_eval is None else condition_eval
        )
        interp["condition_required_keys"] = list(condition_keys)
    revision = repos.claims.add_revision(
        claim_id,
        target_state.value,
        json_dumps(proposal.object_json) if proposal.object_json else None,
        proposal.polarity.value,
        proposal.modality.value,
        json_dumps(proposal.condition.to_json()) if proposal.condition else None,
        json_dumps(interp),
        [proposal.valid],
        [
            (s.span_id, "primary" if i == 0 else "contextual")
            for i, s in enumerate(proposal.evidence)
        ],
        seq,
        conn,
    )
    istat = interpretation_status(proposal)
    _stamp_admission_v2(
        conn, claim_id, revision, proposal, subject_id, istat, registry_version
    )
    _register_evidence_family(store, conn, scope_id, claim_id, proposal)
    if judge_res is not None and judge_req is not None:
        decision_ids = (
            _record_decision(
                ctx,
                store,
                repos,
                conn,
                scope_id,
                judge_req,
                judge_res,
                (
                    ("claim", claim_id, revision),
                    *(
                        ("span", s.span_id, None)
                        for s in proposal.evidence
                    ),
                ),
            ),
        )
    review_id: Optional[str] = None
    if need_review:
        review_id = repos.reviews.create(
            conn,
            scope_id,
            {
                "effect": "admit",
                "claim_id": claim_id,
                "reason": reason,
            },
            {claim_id: revision},
            decision_id=decision_ids[0] if decision_ids else None,
        )
    return AdmissionOutcome(claim_id, target_state, review_id, reason, decision_ids)


# ---------------------------------------------------------------------------
# Pair comparison (SPEC §16)
# ---------------------------------------------------------------------------


def _obj_text(head: ClaimHead) -> Optional[str]:
    if not head.object_json:
        return None
    try:
        obj = safe_json_loads(head.object_json)
    except VerbatimError:
        return None
    if isinstance(obj, dict):
        t = obj.get("text")
        return t if isinstance(t, str) else None
    return None


def _cond_key(condition_json: Optional[str]) -> Optional[str]:
    """Canonical structural key for a stored condition (None = unconditional)."""
    if not condition_json:
        return None
    try:
        obj = safe_json_loads(condition_json)
    except VerbatimError:
        return "unparseable"
    return json_dumps(obj)


def _iv_overlap(a: list[TimeInterval], b: list[TimeInterval]) -> Optional[bool]:
    """Any provable overlap → True; all provably disjoint → False; else None."""
    saw_unknown = False
    for x in a:
        for y in b:
            ov = overlaps(x, y)
            if ov is True:
                return True
            if ov is None:
                saw_unknown = True
    return None if saw_unknown or not a or not b else False


def _claim_text(
    store: Any, conn: sqlite3.Connection, claim_id: str, revision: int
) -> Optional[str]:
    """Reconstruct the exact quotation from span + source bytes.

    Each slice is re-verified against ``spans.excerpt_hmac`` before use —
    the same consumption-time check ``SpansRepo.text`` performs, so a
    tampered excerpt cannot silently feed pair classification. A NULL
    digest marks a legacy row and is skipped; an emptied (purged)
    payload contributes nothing.
    """
    rows = conn.execute(
        "SELECT s.start_byte, s.end_byte, sr.payload, s.excerpt_hmac"
        " FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id"
        " JOIN source_revisions sr"
        "   ON sr.source_id = s.source_id AND sr.revision = s.revision"
        " WHERE ce.claim_id = ? AND ce.revision = ? AND ce.evidence_role = 'primary'"
        " ORDER BY s.start_byte",
        (claim_id, revision),
    ).fetchall()
    parts = []
    for start, end, payload, excerpt_hmac in rows:
        payload = bytes(payload)
        if not payload:
            continue  # purged revision: the excerpt bytes no longer exist
        excerpt = payload[start:end]
        if excerpt_hmac is not None and not hmac.compare_digest(
            store.hmac(excerpt), bytes(excerpt_hmac)
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"claim {claim_id} evidence fails integrity check",
            )
        try:
            parts.append(excerpt.decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return " ".join(parts) if parts else None


def _pair_candidates(conn: sqlite3.Connection, head: ClaimHead) -> list[ClaimHead]:
    """≤8 same-scope comparison candidates (SPEC §16 budget).

    Same-predicate claims first, then high lexical overlap on object text —
    similarity only *generates* candidates, it is never evidence of conflict.
    """
    ids: list[str] = []
    if head.predicate:
        rows = conn.execute(
            "SELECT c.claim_id FROM claims c"
            " JOIN claim_revisions r ON r.claim_id = c.claim_id"
            "   AND r.recorded_until IS NULL"
            " WHERE c.scope_id = ? AND c.claim_id <> ? AND c.predicate = ?"
            "   AND r.state IN ('active','disputed')"
            "   AND r.interpretation_status <> 'unstructured' LIMIT ?",
            (head.scope_id, head.claim_id, head.predicate, _PAIR_BUDGET),
        ).fetchall()
        ids.extend(r[0] for r in rows)
    if len(ids) < _PAIR_BUDGET:
        new_tokens = set(((_obj_text(head) or "")).lower().split())
        rows = conn.execute(
            "SELECT c.claim_id, r.object_json FROM claims c"
            " JOIN claim_revisions r ON r.claim_id = c.claim_id"
            "   AND r.recorded_until IS NULL"
            " WHERE c.scope_id = ? AND c.claim_id <> ?"
            "   AND r.state IN ('active','disputed')"
            "   AND r.interpretation_status <> 'unstructured'",
            (head.scope_id, head.claim_id),
        ).fetchall()
        for cid, obj_json in rows:
            if len(ids) >= _PAIR_BUDGET:
                break
            if cid in ids or not new_tokens or not obj_json:
                continue
            try:
                obj = safe_json_loads(obj_json)
            except VerbatimError:
                continue
            text = obj.get("text") if isinstance(obj, dict) else None
            if not isinstance(text, str):
                continue
            overlap = len(new_tokens & set(text.lower().split()))
            if overlap / len(new_tokens) >= 0.5:
                ids.append(cid)
    out = []
    for cid in ids[:_PAIR_BUDGET]:
        h = read_claim_head(conn, cid)
        if h is not None:
            out.append(h)
    return out


def _classify_pair(
    new: ClaimHead,
    new_ivs: list[TimeInterval],
    old: ClaimHead,
    old_ivs: list[TimeInterval],
) -> tuple[PairLabel, str]:
    """Deterministic pair rules (SPEC §16).

    The labels compare *interpretations*, not truth: 'incompatible' means the
    pair cannot hold under the same scope and interval and must be reviewed —
    it never selects a winner by itself.
    """
    if new.subject_id and old.subject_id and new.subject_id != old.subject_id:
        return PairLabel.DIFFERENT_SCOPE, "different_subject"
    nt, ot = _obj_text(new), _obj_text(old)
    if not nt or not ot:
        return PairLabel.INSUFFICIENT_CONTEXT, "missing_object"
    nc, oc = _cond_key(new.condition_json), _cond_key(old.condition_json)
    same_obj = nt.strip().lower() == ot.strip().lower()
    if same_obj:
        if new.polarity != old.polarity:
            return PairLabel.INCOMPATIBLE, "negation_conflict"
        if nc == oc:
            return PairLabel.EQUIVALENT, "same_proposition"
        if nc and oc and nc != "unparseable" and oc != "unparseable":
            return PairLabel.DIFFERENT_SCOPE, "different_conditions"
        return PairLabel.COMPATIBLE, "one_unconditional"
    if new.predicate != old.predicate:
        return PairLabel.COMPATIBLE, "different_predicate"
    ov = _iv_overlap(new_ivs, old_ivs)
    if ov is False:
        return PairLabel.DIFFERENT_SCOPE, "non_overlapping_time"
    if nc and oc and nc != oc:
        return PairLabel.DIFFERENT_SCOPE, "different_conditions"
    if "unparseable" in (nc or "", oc or ""):
        return PairLabel.INSUFFICIENT_CONTEXT, "unparseable_condition"
    if (nc is None) != (oc is None):
        return PairLabel.INSUFFICIENT_CONTEXT, "condition_mismatch"
    if new.modality != "asserted" or old.modality != "asserted":
        # A hedged statement cannot contradict an asserted one.
        return PairLabel.COMPATIBLE, "modality_differs"
    slot = slot_for(new.predicate)
    if slot and slot["multi_valued"]:
        return PairLabel.COMPATIBLE, "set_valued_slot"
    return PairLabel.INCOMPATIBLE, "same_slot_different_value"


def relate(store: Any, new_claim_id: str, ctx: Optional[PolicyContext] = None) -> list[str]:
    """Compare a new claim against ≤8 same-scope candidates (SPEC §16).

    Deterministic rules classify each pair; when a judge is configured,
    TaskKind.PAIR_RELATION (plus one CHANGE_SIGNAL probe) may replace the
    rules label, with every decision recorded for replay. Only
    ``incompatible`` produces artifacts — a canonical ``conflicts_with`` edge
    and a review *proposing* supersession. Model output never applies a
    transition directly.

    Returns ids of created artifacts (edge ids and review ids).
    """
    if ctx is None:
        ctx = PolicyContext(cfg=VerbatimConfig())
    repos = _repos(store)

    # Phase 1 — read snapshot (no write tx held while a backend runs, §21).
    with store.read() as conn:
        new_head = read_claim_head(conn, new_claim_id)
        if new_head is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found")
        # Unstructured claims are verbatim quotations, not slot
        # interpretations — they never enter slot-conflict detection
        # (SPEC_V2 §13.08). They remain searchable evidence either way.
        if new_head.interpretation_status == InterpretationStatus.UNSTRUCTURED:
            return []
        new_ivs = read_intervals(conn, new_claim_id, new_head.revision)
        new_text = _claim_text(store, conn, new_claim_id, new_head.revision)
        cands = [
            (
                h,
                read_intervals(conn, h.claim_id, h.revision),
                _claim_text(store, conn, h.claim_id, h.revision),
            )
            for h in _pair_candidates(conn, new_head)
        ]

    # Phase 2 — classify (rules first; judge may override with a typed label).
    evaluations: list[tuple[DecisionRequest, DecisionResult, tuple]] = []
    labels: list[tuple[ClaimHead, PairLabel, str]] = []
    for h, ivs, text in cands:
        label, why = _classify_pair(new_head, new_ivs, h, ivs)
        if ctx.judge is not None:
            req = DecisionRequest(
                task=TaskKind.PAIR_RELATION,
                state={
                    "old": text or "",
                    "new": new_text or "",
                    "speaker_relation": (
                        "same"
                        if new_head.subject_id == h.subject_id
                        else "different"
                    ),
                    "time_overlap": {
                        True: "yes",
                        False: "no",
                        None: "unknown",
                    }[_iv_overlap(new_ivs, ivs)],
                },
                allowed_labels=_PAIR_LABELS,
                deadline_s=8.0,
                policy_epoch=ctx.policy_epoch,
                purpose="candidate_curation",
            )
            res = _evaluate(ctx, req)
            if res is not None:
                evaluations.append(
                    (
                        req,
                        res,
                        (
                            ("claim", new_claim_id, new_head.revision),
                            ("claim", h.claim_id, h.revision),
                        ),
                    )
                )
                judge_label = _pair_label_of(res)
                if judge_label is not None:
                    label, why = judge_label, "judge_pair_relation"
        labels.append((h, label, why))

    if ctx.judge is not None:
        req = DecisionRequest(
            task=TaskKind.CHANGE_SIGNAL,
            state={"text": new_text or ""},
            allowed_labels=_CHANGE_LABELS,
            deadline_s=8.0,
            policy_epoch=ctx.policy_epoch,
            purpose="candidate_curation",
        )
        res = _evaluate(ctx, req)
        if res is not None:
            evaluations.append(
                (req, res, (("claim", new_claim_id, new_head.revision),))
            )
            change = _change_signal_of(res)
        else:
            change = None
    else:
        change = None

    # Phase 3 — artifacts + decisions + audit event, one transaction.
    out: list[str] = []
    decision_ids: list[str] = []
    with store.tx() as conn:
        _seed_builtin_registry(store, conn)
        # Mutable single-valued slots may *propose* supersession; immutable
        # or unregistered predicates become disputes instead — the reviewer
        # decides, nothing supersedes automatically (SPEC_V2 §19).
        new_reg = _registry_slot(conn, new_head.predicate)
        new_slot = new_reg if new_reg is not None else slot_for(new_head.predicate)
        supersession_allowed = bool(new_slot and new_slot["mutable"])
        seq = repos.events.append(
            conn,
            new_head.scope_id,
            "pair_comparison",
            ENGINE_ACTOR,
            {
                "new_claim_id": new_claim_id,
                "pairs": [
                    {"claim_id": h.claim_id, "label": lab.value, "rule": why}
                    for h, lab, why in labels
                ],
                "exhausted": len(cands) < _PAIR_BUDGET,
            },
            ctx.policy_version,
        )
        for req, res, inputs in evaluations:
            decision_ids.append(
                _record_decision(ctx, store, repos, conn, new_head.scope_id, req, res, inputs)
            )
        per_pair_decisions: dict[str, str] = {}
        for (req, res, inputs), did in zip(evaluations, decision_ids):
            others = [
                oid for kind, oid, _r in inputs
                if kind == "claim" and oid != new_claim_id
            ]
            if others:
                per_pair_decisions[others[0]] = did
        for h, label, why in labels:
            if label != PairLabel.INCOMPATIBLE:
                continue
            a, b = sorted((new_claim_id, h.claim_id))
            edge_id = repos.edges.add(
                conn,
                new_head.scope_id,
                "claim",
                a,
                "claim",
                b,
                "conflicts_with",
            )
            out.append(edge_id)
            # Conflict-group membership commits with the edge — a pair can
            # never exist as an edge without its cluster row (SPEC_V2 §19).
            _conflict_group_for(conn, new_head.scope_id, a, b)
            if supersession_allowed:
                effect = {
                    "effect": "supersede",
                    "predecessor_id": h.claim_id,
                    "successor_id": new_claim_id,
                    "pair_label": label.value,
                    "change_signal": change.value if change else None,
                    "reason": why,
                }
            else:
                effect = {
                    "effect": "dispute",
                    "claim_id": h.claim_id,
                    "counterparty_id": new_claim_id,
                    "pair_label": label.value,
                    "change_signal": change.value if change else None,
                    "reason": why,
                }
            review_id = repos.reviews.create(
                conn,
                new_head.scope_id,
                effect,
                {h.claim_id: h.revision, new_claim_id: new_head.revision},
                decision_id=per_pair_decisions.get(h.claim_id),
                dedup=True,
            )
            out.append(review_id)
    return out


def propose_supersede(
    store: Any,
    predecessor_id: str,
    successor_id: str,
    actor_id: str,
    reason: str,
    interval: Optional[TimeInterval] = None,
    ctx: Optional[PolicyContext] = None,
) -> str:
    """Create an operator review proposing ``successor`` supersedes ``predecessor``.

    Carries expected versions of both claims so application re-checks them in
    one transaction and stale proposals fail with ``STALE_PROPOSAL`` rather
    than overwriting newer corrections (SPEC §17). Returns the review id.
    """
    if ctx is None:
        ctx = PolicyContext(cfg=VerbatimConfig())
    repos = _repos(store)
    with store.tx() as conn:
        pred = read_claim_head(conn, predecessor_id)
        succ = read_claim_head(conn, successor_id)
        if pred is None or succ is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found"
            )
        if pred.scope_id != succ.scope_id:
            raise VerbatimError(
                ErrorCode.VALIDATION, "claims live in different scopes"
            )
        if pred.state != Lifecycle.ACTIVE:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"predecessor must be active, is {pred.state.value}",
            )
        if succ.state in (Lifecycle.ERASED, Lifecycle.REJECTED, Lifecycle.SUPERSEDED):
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"successor in state {succ.state.value} cannot supersede",
            )
        repos.events.append(
            conn,
            pred.scope_id,
            "review_proposed",
            actor_id,
            {
                "effect": "supersede",
                "predecessor_id": predecessor_id,
                "successor_id": successor_id,
                "reason": reason,
            },
            ctx.policy_version,
        )
        return repos.reviews.create(
            conn,
            pred.scope_id,
            {
                "effect": "supersede",
                "predecessor_id": predecessor_id,
                "successor_id": successor_id,
                "interval": _review_interval_payload(interval),
                "reason": reason,
            },
            {predecessor_id: pred.revision, successor_id: succ.revision},
        )


# ---------------------------------------------------------------------------
# Review application (SPEC_V2 §19, §37)
# ---------------------------------------------------------------------------


def _review_interval_payload(iv: Optional[TimeInterval]) -> Optional[dict[str, Any]]:
    """Serialize an interval into a review payload, keeping endpoint kinds."""
    if iv is None:
        return None
    return {
        "from_us": iv.from_us,
        "until_us": iv.until_us,
        "precision": iv.precision.value,
        "timezone": iv.timezone,
        "basis": iv.basis,
        "start_kind": iv.start_kind.value,
        "end_kind": iv.end_kind.value,
        "from_us_hi": iv.from_us_hi,
        "until_us_hi": iv.until_us_hi,
    }


def _review_interval(payload: Any) -> Optional[TimeInterval]:
    """Rebuild a TimeInterval from a review payload; ``None``/invalid → None."""
    if not isinstance(payload, dict):
        return None
    try:
        return TimeInterval(
            from_us=payload.get("from_us"),
            until_us=payload.get("until_us"),
            precision=payload.get("precision") or Precision.UNKNOWN,
            timezone=payload.get("timezone"),
            basis=payload.get("basis") or "unknown",
            start_kind=payload.get("start_kind") or EndpointKind.EXACT,
            end_kind=payload.get("end_kind") or EndpointKind.EXACT,
            from_us_hi=payload.get("from_us_hi"),
            until_us_hi=payload.get("until_us_hi"),
        )
    except (VerbatimError, ValueError):
        return None


def apply_review(
    store: Any,
    review_id: str,
    actor_id: str,
    ctx: Optional[PolicyContext] = None,
) -> int:
    """Apply an open review's proposed effect in ONE transaction (SPEC_V2 §19).

    The single ``store.tx()`` contains: the expected-version fence, the
    lifecycle transition (with its interval effects and audit event), the
    review state update, and current-generation FTS indexing — a crash can
    never leave a completed transition behind an open review.

    Raises ``STALE_PROPOSAL`` when any fenced revision has moved on (the
    review then stays open for re-triage), ``NOT_FOUND_OR_FORBIDDEN`` for a
    missing review, and ``INVALID_TRANSITION`` for an effect the transition
    table rejects. Returns the transition's event sequence number.
    """
    if ctx is None:
        ctx = PolicyContext(cfg=VerbatimConfig())
    repos = _repos(store)
    machine = LifecycleMachine(store, policy_version=ctx.policy_version)

    review = repos.reviews.get(review_id)
    if review is None:
        raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "review not found")
    if review["state"] != ReviewState.OPEN.value:
        raise VerbatimError(ErrorCode.STALE_PROPOSAL, "review already resolved")
    effect = review["proposed_effect"] or {}
    versions = review["expected_versions"] or {}
    kind = effect.get("effect")

    if kind == "admit":
        claim_id = effect.get("claim_id")
        cmd_effect, successor, interval = "admit", None, None
    elif kind == "supersede":
        claim_id = effect.get("predecessor_id")
        cmd_effect = "supersede"
        successor = effect.get("successor_id")
        interval = _review_interval(effect.get("interval"))
    elif kind == "dispute":
        claim_id = effect.get("claim_id") or effect.get("predecessor_id")
        cmd_effect, successor, interval = "dispute", None, None
    elif kind == "reject":
        claim_id = effect.get("claim_id")
        cmd_effect, successor, interval = "reject", None, None
    else:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"unsupported review effect {kind!r}",
        )
    if not claim_id:
        raise VerbatimError(ErrorCode.VALIDATION, "review effect lacks a claim")
    reason = effect.get("reason") or f"review {review_id}"

    with store.tx() as conn:
        # --- expected-version fence: every pinned revision must still be
        # current, or the whole review is stale (SPEC_V2 §19.07). -----------
        for cid, expected in versions.items():
            h = read_claim_head(conn, cid)
            if h is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found"
                )
            if h.revision != expected:
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    f"claim {cid} at revision {h.revision},"
                    f" review expected {expected}",
                )
        expected = versions.get(claim_id)
        if expected is None:
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL,
                "review does not fence the transitioned claim",
            )
        seq = machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=expected,
                effect=cmd_effect,
                actor_id=actor_id,
                reason=reason,
                successor_claim_id=successor,
                interval=interval,
            ),
            conn,
        )
        repos.reviews.resolve(conn, review_id, ReviewState.APPROVED, seq)
        # No generation bump (V2-19.10): FTS filters on
        # ``projection_generation = current``, so a bump would strand every
        # other indexed claim. The newly-active revision is indexed at the
        # current generation inside this same transaction (V2-39.01).
        head = read_claim_head(conn, claim_id)
        if head is not None and head.state == Lifecycle.ACTIVE:
            from ..storage.repos import FtsRepo

            FtsRepo(store).index_claim_primary(
                conn, claim_id, head.revision, head.scope_id,
                store.projection_generation(),
            )
    if kind == "admit":
        relate(store, claim_id, ctx=ctx)
    return seq
