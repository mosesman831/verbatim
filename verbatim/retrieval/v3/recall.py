"""The v3 recall pipeline (SPEC_V3 §26–§32; V4-10.05/V4-11.04/F4-01).

``recall_v3(store, request)`` runs, in order:

    authorize → query analysis → deterministic route planning →
    lane execution → eligibility-first union → fusion → group assembly →
    abstention → typed pack assembly → influence minting/delivery →
    routing-decision logging

Authorization precedes everything: ``governance.authorize`` is called
defensively; when the parallel governance module is absent the v2
``_allowed_scopes`` path is the documented compatibility shim — retrieval
still runs scope-intersected, never widened (V3-26.07). Authorization and
integrity failures propagate; they are never degraded (V3-28.12).

Scope expansion is purpose-aware (F4-01): every contributing scope is
authorized individually through ``governance.authorize`` for the
request's own purpose — the purpose-blind ``effective_verbs`` union is
diagnostic-only and never consulted (V4-11.04). Cross-scope retrieval is
thereby limited to scopes where the caller holds read-or-quote authority
for that purpose and output kind (V4-10.05); the grant set IS the
explicit search-domain policy — no grant, no contribution, and denial is
indistinguishable from absence (no error, count, or label names the
excluded scope).

The only writes a recall performs are its decision log and influence
exposure rows, inside one ``store.tx()`` after the read snapshot closes —
no synchronous learning side effects (V3-26.04).

Progressive disclosure (V4-32.05): ``detail_tier`` selects the serialized
projection — ``l0`` navigational metadata, ``l1`` grounded overview
(bounded summaries, no payload bytes), ``l2`` exact detail under the
``quote`` verb. Asking for ``l2`` without ``quote`` on any contributing
scope collapses honestly to ``l1`` with a warning. Every delivered item
carries an ``expand`` ref — a caller/scope/purpose/revision/expiry-bound
token whose use re-runs authorization against CURRENT state
(``expand_item``, V4-32.09).

Final-pack cache (V4-33): when ``retrieval.cache.enabled`` is configured
on, each store gets an in-process capacity-bounded LRU over delivered
results. A hit is never an authorization decision — the stored
fingerprint (scope-epoch vector, policy/erasure epochs, projection
generation, schema, ``PRAGMA data_version``, and a content watermark)
must still match, and every delivered object ref is revalidated against
the live snapshot before a cached byte ships.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

from ...core.time import now_us
from ...core.types import (
    ErrorCode,
    Scope,
    VerbatimError,
    Visibility,
    safe_json_loads,
)
from ...core.types_v3 import (
    ActionIntent,
    BudgetTier,
    ContextPack,
    ContextRequestV3,
    GroupSupport,
    PackItem,
    PackKind,
    QueryClass,
    RecallRequestV3,
    RecallResultV3,
    TaskContext,
    Verb,
)
from ...policy import artifacts as _polart
from .. import cache as _rcache
from .. import candidates as _cand
from .. import manifest as _manifest
from ..query import analyze
from .abstain import abstain as _abstain_fn
from . import controller as _ctrl
from . import fusion_v3 as _fuse
from . import learned as _learned
from . import influence as _infl
from . import lanes as _lanes
from . import pack as _pack
from . import union as _union

# ---------------------------------------------------------------------------
# record-write contention
# ---------------------------------------------------------------------------

# The managed worker drains on its *own* Store connection (V5-09.08), so
# cross-connection SQLite contention under live load is expected — a
# retryable STORE_BUSY/BACKPRESSURE from ``store.tx()`` is transient,
# not a failure of the record write. Decision-log and influence rows
# are contract writes (exposure/audit state), never best-effort
# decoration, so bounded retry is the honest path: the transaction is
# atomic, a failed attempt rolls back cleanly, and a non-transient error
# still propagates untouched.
_RECORD_WRITE_ATTEMPTS = 8
_RECORD_WRITE_SLEEP_S = 0.03
#: Per-attempt admission budget: ``budget_ms`` clamps both the in-process
#: writer-lock wait and the connection's SQLite busy_timeout, so a
#: contended attempt fails fast (~60ms) and retries instead of parking
#: the read on the full 250ms busy window (V4-40.02/03 semantics — the
#: clamp restores the configured allowance for the next writer).
_RECORD_WRITE_BUDGET_MS = 60.0


def _record_tx(store: Any, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    """Run ``fn(conn)`` inside ``store.tx()``, retrying transient contention."""
    last: Optional[BaseException] = None
    for _ in range(_RECORD_WRITE_ATTEMPTS):
        try:
            with store.tx(budget_ms=_RECORD_WRITE_BUDGET_MS) as conn:
                return fn(conn)
        except VerbatimError as exc:
            if not exc.retryable:
                raise
            last = exc
        except sqlite3.OperationalError as exc:
            low = str(exc).lower()
            if "locked" not in low and "busy" not in low:
                raise
            last = exc
        time.sleep(_RECORD_WRITE_SLEEP_S)
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def _caller(request: Any) -> Any:
    """Build the governance caller record; None when governance is absent."""
    try:
        from ... import governance  # type: ignore
    except ImportError:
        return None
    task = getattr(request, "task", None)
    return governance.CallerV3(
        principal_id=request.caller_id,
        session_id=(getattr(task, "task_id", None) or "") if task else "",
    )


def _scope_contributes(
    governance: Any,
    conn: sqlite3.Connection,
    caller: Any,
    scope_id: str,
    purpose: Optional[str],
) -> bool:
    """One scope's purpose-aware contribution check (F4-01, V4-11.04).

    Every contributing scope passes ``governance.authorize`` for the
    request's own purpose — never a purpose-blind verb union. ``read``
    is the retrieval floor (metadata + derived views, V4-08.02);
    ``quote`` alone also admits a scope because exact-byte authority
    subsumes retrieval visibility (per-output-kind authorization,
    V4-10.05).

    Denial and absence are publicly indistinguishable: a scope the
    caller cannot use FOR THIS PURPOSE simply does not contribute — no
    error, count, or label names it. A pinned epoch that no longer
    matches the scope cannot be evaluated under the request's epoch
    vector (V4-11.07), so the scope is excluded the same way; other
    failures propagate (authorization is never degraded, V3-28.12).
    """
    for verb in (Verb.READ.value, Verb.QUOTE.value):
        try:
            governance.authorize(
                conn, caller, scope_id, verb, purpose=purpose
            )
            return True
        except VerbatimError as exc:
            if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                continue  # try the other output-kind verb
            if exc.code is ErrorCode.STALE_EPOCH:
                return False  # cannot evaluate under the pinned vector
            raise
    return False


def _authorized_scopes(
    conn: sqlite3.Connection, request: RecallRequestV3
) -> tuple:
    """``(scope_ids, epoch, used_grant_path)``.

    Primary path (F4-01 / V4-11.04): ``governance.authorize`` runs per
    stored scope with the request's own purpose. The requesting scope
    fails closed — denial of the named scope surfaces as the
    indistinguishable ``NOT_FOUND_OR_UNAUTHORIZED`` (§10.05), a stale
    pinned epoch fences loudly and retryably (§09.04). Every other scope
    contributes only when the same purpose-aware check passes; a scope
    the caller may not use for this purpose is excluded silently.

    Compatibility shim (documented, V3 contract allows): when governance
    is absent, the v2 ``_allowed_scopes`` walk over the request scope's
    own row keeps self-scope reads working. Authorization never widens:
    both paths only ever return scope rows the caller may actually use.
    """
    try:
        from ... import governance  # type: ignore
        from ...governance import epochs as _epochs  # type: ignore
    except ImportError:
        # Absent module → the documented v2 shim. A BROKEN governance
        # import (partial module, missing symbol) propagates — silently
        # downgrading grant-based authorization would be worse than a
        # loud failure.
        governance = None
        _epochs = None

    if governance is not None:
        caller = _caller(request)
        # Fail closed: the requesting scope itself must authorize the
        # request's purpose. ``read`` is the floor; ``quote`` alone
        # suffices (exact-byte authority subsumes retrieval visibility).
        try:
            governance.authorize(
                conn, caller, request.scope_id, Verb.READ.value,
                purpose=request.purpose,
            )
        except VerbatimError as exc:
            if exc.code is not ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                raise
            governance.authorize(
                conn, caller, request.scope_id, Verb.QUOTE.value,
                purpose=request.purpose,
            )
        epoch = _epochs.current_epoch(conn, request.scope_id)
        rows = conn.execute(
            "SELECT scope_id FROM scopes ORDER BY scope_id"
        ).fetchall()
        allowed = [
            sid for (sid,) in rows
            if sid == request.scope_id  # authorize() already passed
            or _scope_contributes(
                governance, conn, caller, sid, request.purpose
            )
        ]
        if request.scope_id not in allowed:
            allowed.append(request.scope_id)  # authorize() already passed
        return tuple(sorted(set(allowed))), epoch, True

    # ---- documented compatibility shim (v2 scope model) ----------------
    row = conn.execute(
        "SELECT profile_id, principal_id, workspace_id, conversation_id,"
        " visibility FROM scopes WHERE scope_id = ?",
        (request.scope_id,),
    ).fetchone()
    if row is None:
        # NOT_FOUND_OR_UNAUTHORIZED: existence and denial are
        # indistinguishable (§10.05).
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scope not authorized"
        )
    reader = Scope(
        profile_id=row[0],
        principal_id=row[1],
        workspace_id=row[2],
        conversation_id=row[3],
        visibility=Visibility(row[4]),
    )
    allowed = _cand._allowed_scopes(conn, reader)
    if request.scope_id not in allowed:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "scope not authorized"
        )
    try:
        epoch = conn.execute(
            "SELECT authz_revision FROM scopes WHERE scope_id = ?",
            (request.scope_id,),
        ).fetchone()[0]
    except Exception:
        epoch = 0
    return tuple(allowed), int(epoch or 0), False


# ---------------------------------------------------------------------------
# progressive disclosure (V4-32.05) — tier resolution + collapse
# ---------------------------------------------------------------------------


def _resolve_detail(request: Any, explicit: Any = None) -> Any:
    """The disclosure tier: explicit call-site argument first, then the
    request's own ``detail_tier`` attribute (the facade sets it from the
    caller's budget mapping), defaulting to ``l2``. Unknown tokens are a
    caller error — never a silent downgrade."""
    raw = (
        explicit
        if explicit is not None
        else getattr(request, "detail_tier", None)
    )
    return _pack.normalize_detail_tier(raw)


def _any_quote_scope(conn: sqlite3.Connection, request: Any,
                     scope_ids) -> bool:
    """True when the caller holds ``quote`` on at least one contributing
    scope at the request purpose — the minimum authority under which an
    ``l2`` ask can release any exact byte. When NO contributing scope
    permits quote, the L2 ask collapses honestly to L1 (V4-32.05) rather
    than shipping an all-withheld exact tier.

    Under the governance-absent shim there is no verb model: the
    authorized scope set itself is the evidence (the render gate still
    narrows per-scope), so a non-empty set reports True.
    """
    try:
        from ... import governance  # type: ignore
    except ImportError:
        return bool(scope_ids)
    caller = _caller(request)
    for sid in sorted(set(scope_ids)):
        try:
            governance.authorize(
                conn, caller, sid, Verb.QUOTE.value,
                purpose=request.purpose,
            )
            return True
        except VerbatimError as exc:
            if exc.code in (
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                ErrorCode.STALE_EPOCH,
            ):
                continue
            raise
    return False


def _object_scope(conn: sqlite3.Connection, kind: str,
                  oid: str) -> Optional[str]:
    """The scope an object row lives in — the contribution boundary an
    expansion is re-checked against (claims read their header row; other
    kinds their own table's scope column)."""
    if kind == "claim":
        row = conn.execute(
            "SELECT scope_id FROM claims WHERE claim_id = ?", (oid,)
        ).fetchone()
        return row[0] if row else None
    spec = _union._OBJECT_TABLES.get(kind)
    if spec is not None:
        table, id_col, _rev, scope_col, _b = spec
        if not _cand._has_table(conn, table):
            return None
        row = conn.execute(
            f"SELECT {scope_col} FROM {table} WHERE {id_col} = ?",
            (oid,),
        ).fetchone()
        return row[0] if row else None
    if kind == "working_item":
        row = conn.execute(
            "SELECT ws.scope_id FROM working_set_items i"
            " JOIN working_sets ws ON ws.set_id = i.set_id"
            " WHERE i.item_id = ?",
            (oid,),
        ).fetchone()
        return row[0] if row else None
    return None


def _deny_expand(msg: str = "expansion denied") -> None:
    """Every expansion failure is the same indistinguishable denial —
    revoked, expired, forged, never-delivered, and freshly-withheld all
    surface identically (V4-32.09)."""
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, msg)


def expand_item(store: Any, expand_ref: str, *, caller_id: str,
                detail_tier: Any = "l2", cfg: Any = None,
                now: Optional[int] = None) -> RecallResultV3:
    """Re-ask one delivered item at a deeper tier (V4-32.09).

    The ``expand`` ref embedded in a delivered item is bound to caller,
    scope, purpose, object kind/id/revision, the delivery's epoch, and an
    expiry — verified against the *stored* influence row (the token's own
    claims are never trusted). Expansion then re-runs CURRENT
    authorization: caller match, read on the bound scope, object-scope
    contribution, epoch vector unchanged (any grant edit since delivery
    denies — a revoked grant can never be survived), quarantine and
    purge suppression, the span→source→envelope cascade, and head-
    revision drift for claims. Anything failing → the indistinguishable
    ``NOT_FOUND_OR_UNAUTHORIZED``.

    A successful expansion mints fresh handles for the re-delivered
    items, records the new exposure rows, and returns a one-pack
    ``RecallResultV3`` at the requested tier (default ``l2`` — exact
    detail still gated per item by the render-point quote check).
    """
    now = now_us() if now is None else now
    t_kind, t_oid, t_rev, t_epoch, purpose, mac = _pack._expand_parse(
        expand_ref
    )
    detail = _pack.normalize_detail_tier(detail_tier)
    deny = _deny_expand

    packs: list = []
    warnings: list = []
    with store.read() as conn:
        # The token binds (caller, scope, purpose, epoch, kind, oid, rev)
        # — resolve it against the caller's own delivery records: a row
        # must exist for exactly this object at exactly the token's
        # epoch, and the MAC is re-derived from each candidate row's
        # stored fields (the row, not the token, supplies scope/caller).
        try:
            rows = conn.execute(
                "SELECT handle_id, receipt_id, scope_id, caller_id, epoch,"
                " pack, object_kind, object_id, revision, created_us,"
                " redacted FROM influence"
                " WHERE caller_id = ? AND object_kind = ?"
                " AND object_id = ? AND revision = ? AND epoch = ?"
                " ORDER BY created_us DESC LIMIT 8",
                (caller_id, t_kind, t_oid, t_rev, t_epoch),
            ).fetchall()
        except sqlite3.Error:
            rows = []
        cols = (
            "handle_id", "receipt_id", "scope_id", "caller_id", "epoch",
            "pack", "object_kind", "object_id", "revision", "created_us",
            "redacted",
        )
        token_fields = (t_kind, t_oid, t_rev, t_epoch, purpose, mac)
        verified = []
        for r in rows:
            rec = dict(zip(cols, r))
            if rec["redacted"]:
                continue  # the delivery itself was scrubbed
            if not _pack._expand_verify(
                store, token_fields=token_fields, row=rec,
            ):
                continue  # token fields don't match this row's binding
            verified.append(rec)
        if not verified:
            deny()
        rec = verified[0]
        # Expiry is enforced from stored delivery state — the token
        # carries no clock of its own (deterministic refs).
        if now > int(rec["created_us"]) + _pack._EXPAND_TTL_US:
            deny()

        try:
            from ... import governance  # type: ignore
            from ...governance import epochs as _epochs  # type: ignore
        except ImportError:
            governance = None
            _epochs = None

        # Re-authorize against CURRENT state — the delivery's epoch is
        # only the bound context, never a grant survival permit.
        if governance is not None:
            caller = governance.CallerV3(principal_id=caller_id)
            epoch_now = _epochs.current_epoch(conn, rec["scope_id"])
            if int(epoch_now) != int(rec["epoch"]):
                deny()  # grant set moved since delivery — re-ask
            try:
                governance.authorize(
                    conn, caller, rec["scope_id"], Verb.READ.value,
                    purpose=purpose or None,
                )
            except VerbatimError:
                deny()
        else:
            epoch_now = 0
            try:
                scope_row = conn.execute(
                    "SELECT authz_revision FROM scopes WHERE scope_id = ?",
                    (rec["scope_id"],),
                ).fetchone()
            except sqlite3.Error:
                scope_row = None
            if scope_row is None:
                deny()
            epoch_now = int(scope_row[0] or 0)
            if epoch_now != int(rec["epoch"]):
                deny()

        obj_scope = _object_scope(conn, rec["object_kind"], rec["object_id"])
        if not obj_scope:
            deny()
        if governance is not None and not _scope_contributes(
            governance, conn, caller, obj_scope, purpose or None
        ):
            deny()

        # ---- current-state admissibility of the pinned object ---------
        snap = _union.SnapshotCache(conn)
        kind = rec["object_kind"]
        oid = rec["object_id"]
        rev = int(rec["revision"])
        hit = _union.UnionHit(
            kind, oid, revision=rev, scope_id=obj_scope or "",
        )
        if kind == "claim":
            head = conn.execute(
                "SELECT revision, state, recorded_from FROM claim_revisions"
                " WHERE claim_id = ? AND recorded_until IS NULL",
                (oid,),
            ).fetchone()
            if head is None or int(head[0]) != rev:
                deny()  # corrected/tombstoned since delivery — re-ask
            if head[1] not in _union.CLASS_STATES[QueryClass.ARCHIVE]:
                deny()  # lifecycle moved past readable visibility
            hit.state = head[1]
            hit.recorded_from = int(head[2] or 0)
            if _union._span_suppressed_claims(
                conn, snap, store, {oid: (rev, head[1], 0)}
            ):
                deny()
        elif kind == "working_item":
            meta = _union._working_item_meta(conn, oid, now)
            if meta is None:
                deny()  # expired/absent working item
        else:
            meta = _union._object_meta(
                conn, kind, oid, None, {obj_scope}
            )
            if meta is None or int(meta[1]) != rev:
                deny()  # revision drift or object gone — re-ask
            hit.state = meta[3] or hit.state
            hit.recorded_from = int(meta[2] or 0)
        if _union._should_exclude(conn, kind, oid, rev):
            deny()  # quarantine hold on the object itself
        if _cand._suppressed(store, conn, kind, [oid]):
            deny()  # purge suppression
        if kind not in _union._PACKAGEABLE:
            deny()

        # ---- rebuild the item list at the requested tier --------------
        # The request is only a carrier for scope/caller/purpose/context —
        # expansion never executes query lanes (the object is pinned by
        # the token). The query field must still satisfy validation.
        request2 = RecallRequestV3(
            query=f"expand {kind} {oid}",
            scope_id=rec["scope_id"],
            caller_id=caller_id,
            purpose=purpose or None,
        )
        if (
            detail is _pack.DetailTier.L2
            and not _any_quote_scope(
                conn, request2, {rec["scope_id"], obj_scope}
            )
        ):
            # The expansion renders the same honest metadata view a
            # quote-less L2 recall would — every item withholds bytes and
            # the warning names the collapse.
            warnings.append("detail_tier_collapsed:quote_unauthorized")
        ctx = SimpleNamespace(
            conn=conn,
            store=store,
            request=request2,
            scope_ids=frozenset({rec["scope_id"], obj_scope}),
        )
        gate = _pack._QuoteGate(ctx)
        plan = SimpleNamespace(known_at_seq=None, valid_at_us=None)
        request_context = _union._request_context(request2)
        complete = True
        if kind == "claim":
            items, complete, _refs = _pack._claim_group_items(
                conn, store, hit, None, plan, request_context, gate,
                batch=None, detail=detail,
            )
        else:
            reasons = ("EXPANSION",)
            item = _pack._object_item(conn, hit, reasons)
            items = [item] if item is not None else []
            if kind == "procedure":
                items.extend(
                    _pack._procedure_step_items(
                        conn, hit, reasons, gate, store, detail=detail
                    )
                )
        if not items:
            deny()
        if not complete:
            warnings.append("group_incomplete_on_expand")

        # ---- re-deliver: fresh handles, fresh seals, fresh exposure ---
        receipt2 = f"rx_{now_us():x}"
        seq = [0]
        pack_kind = (
            PackKind(rec["pack"])
            if rec["pack"] in {k.value for k in PackKind}
            else PackKind.EVIDENCE_BUNDLE
        )
        pack_items: list = []
        used_bytes = 0
        withheld_n = 0
        for item in items:
            seq[0] += 1
            handle = _infl.mint_handle(
                receipt2, caller_id, int(epoch_now), pack_kind,
                kind, oid, rev, seq=seq[0],
            )
            ref = _pack._expand_seal(
                store, scope_id=rec["scope_id"], caller_id=caller_id,
                purpose=purpose or None, epoch=int(epoch_now),
                object_kind=kind, object_id=oid,
                revision=rev,
            )
            wrapped, withheld = _pack._render_item(
                item, gate, obj_scope, detail=detail,
                expand_ref=ref, object_kind=kind,
            )
            if _pack._screen_text(item.text):
                warnings.append("screened_secret_item")
                continue
            if withheld:
                withheld_n += 1
            used_bytes += len(wrapped.encode("utf-8")) + 2
            pack_items.append(PackItem(
                handle=handle,
                text=wrapped,
                lifecycle=item.lifecycle.value,
                freshness=_pack._freshness_enum("unknown"),
                security=_pack._security_label(conn, _pack._label_id_for(
                    conn, kind, item.claim_id, item.claim_revision, hit,
                )),
                perspective=_pack._perspective_for(conn, hit, None),
                derived=kind in (
                    "observation", "procedure", "episode", "transition",
                ),
                proof_count=0,
                verify_recommended=False,
            ))
        if withheld_n:
            warnings.append("quote_withheld")
        if not pack_items:
            deny()
        packs.append(ContextPack(
            kind=pack_kind,
            items=tuple(pack_items),
            tokens=max(
                1,
                used_bytes // _pack._APPROX_CHARS_PER_TOKEN,
            ) if pack_items else 0,
            serialized_bytes=used_bytes,
            warnings=tuple(dict.fromkeys(warnings)),
        ))
        generation = 0
        try:
            gen_row = conn.execute(
                "SELECT value_json FROM meta"
                " WHERE key = 'projection_generation'"
            ).fetchone()
            generation = int(safe_json_loads(gen_row[0]) or 0) \
                if gen_row else 0
        except sqlite3.Error:
            generation = 0
        new_handles = [i.handle for i in pack_items]

    # write phase: the re-delivery is a new exposure under fresh handles.
    try:
        _record_tx(
            store,
            lambda wconn: _infl.record_deliveries(
                wconn, new_handles, rec["scope_id"]
            ),
        )
    except VerbatimError:
        raise
    except sqlite3.Error as exc:
        raise VerbatimError(
            ErrorCode.INTEGRITY, f"expand record write failed: {exc}"
        ) from exc

    return RecallResultV3(
        packs=tuple(packs),
        omitted=0,
        warnings=tuple(dict.fromkeys(warnings)),
        capabilities={
            "degraded": [],
            "lanes": {},
            "detail_tier": detail.value,
            "expand": {"handle_id": rec["handle_id"], "receipt": receipt2},
        },
        decision_id=None,
        projection_generation=generation,
        abstained=False,
    )


# ---------------------------------------------------------------------------
# query classification (§27.06)
# ---------------------------------------------------------------------------

_MODE_TO_CLASS = {
    "exact_lookup": QueryClass.EXACT_LOOKUP,
    "exact": QueryClass.EXACT_LOOKUP,
    "current": QueryClass.CURRENT_STATE,
    "current_state": QueryClass.CURRENT_STATE,
    "past": QueryClass.PAST_STATE,
    "past_state": QueryClass.PAST_STATE,
    "timeline": QueryClass.TIMELINE,
    "relationship": QueryClass.RELATIONSHIP,
    "cause": QueryClass.CAUSE,
    "procedure": QueryClass.PROCEDURE,
    "failure": QueryClass.FAILURE,
    "archive": QueryClass.ARCHIVE,
    "exploratory": QueryClass.EXPLORATORY,
}

_BROWSE_CLASSES = frozenset({QueryClass.TIMELINE, QueryClass.ARCHIVE})


def classify(request: RecallRequestV3, plan: Any) -> QueryClass:
    """Request modes first, then conservative inference (§27.06)."""
    for mode in request.modes:
        cls = _MODE_TO_CLASS.get(str(mode).lower())
        if cls is not None:
            return cls
    task = request.task
    if task is not None and (task.stuck or task.session_phase == "stuck"):
        return QueryClass.FAILURE
    if task is not None and task.goal_class in ("procedure", "task"):
        return QueryClass.PROCEDURE
    if request.entity_ids:
        return QueryClass.EXACT_LOOKUP
    if plan.valid_until_us is not None:
        return QueryClass.TIMELINE
    if plan.valid_at_us is not None or request.known_at_seq is not None:
        return QueryClass.PAST_STATE
    if plan.predicates:
        return QueryClass.RELATIONSHIP
    return QueryClass.CURRENT_STATE


# ---------------------------------------------------------------------------
# lane selection for the route set
# ---------------------------------------------------------------------------


def _semantic_possible(conn: sqlite3.Connection, store: Any) -> bool:
    """Encoder + vectors provisioned → a lexical-free query may still
    be answered semantically (V3-29.12)."""
    try:
        if _cand._query_encoder(store) is None:
            return False
        row = conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
        return row is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def _result_sizes(packs: list) -> dict:
    return {
        (p.kind.value if isinstance(p.kind, PackKind) else p.kind):
        len(p.items)
        for p in packs
    }


def recall_v3(store: Any, request: RecallRequestV3,
              cfg: Any = None,
              policy_tables: Any = None,
              detail_tier: Any = None,
              manifest: Any = None,
              pack_mode: Any = None) -> RecallResultV3:
    """Full v3 recall. Read work inside ``store.read()``; the decision log
    and influence rows write afterwards inside one ``store.tx()``.

    ``cfg`` — optional config root exposing ``.v3.retrieval``; absent, the
    deterministic controller and the 128-candidate default apply.

    ``policy_tables`` — optional replay-lab table overrides forwarded to
    ``plan_routes`` (§43 paired sandbox arms); ``None`` is production
    policy. Overrides never mutate the module-level tables. Replay arms
    are never cache-served — each arm must execute its own variant
    tables.

    ``detail_tier`` — disclosure tier ``l0``/``l1``/``l2`` (V4-32.05);
    the request's ``detail_tier`` attribute is honored when no explicit
    argument is given. ``l2`` without ``quote`` on any contributing scope
    collapses to ``l1`` with a warning rather than shipping a fully
    withheld exact tier.

    ``manifest`` — opt-in ``"counterevidence_first"`` (V45-03/I1): the
    result's ``capabilities["manifest"]`` carries the evidence manifest —
    supporting refs, contrary refs included/omitted+reasons, withheld
    presence labels, insufficiency labels, unresolved obligations — and
    contrary groups admit before depth under the shared budget. Default
    is byte-identical v3 behavior. ``pack_mode`` — ``"sufficiency"``
    (V45-05/I3): minimal identifier/condition-covering admission before
    depth. Both fall back to the request's own attributes (dataclass
    fields on RecallRequestV3; the detail_tier setattr convention works
    identically), and both fail closed on unknown tokens.
    """
    warnings: list = []
    plan = analyze(request.query, request, now_us())
    warnings.extend(plan.warnings)
    cfg_root = cfg if cfg is not None else getattr(store, "cfg", None)
    cfg_v3 = getattr(cfg_root, "v3", None)
    retrieval_cfg = getattr(cfg_v3, "retrieval", None)
    controller_kind = getattr(retrieval_cfg, "controller", "deterministic")
    cap_cfg = getattr(retrieval_cfg, "candidate_cap", 128) or 128
    detail = _resolve_detail(request, detail_tier)
    manifest_mode = _manifest.normalize_manifest(
        manifest if manifest is not None
        else getattr(request, "manifest", None)
    )
    pack_mode = _manifest.normalize_pack_mode(
        pack_mode if pack_mode is not None
        else getattr(request, "pack_mode", "standard")
    )
    # The final-pack cache is a performance layer, never an authority:
    # replay arms (policy_tables) bypass it entirely — each arm must run
    # its own variant tables (V4-33.01 keeps table identity out of the
    # key, so skipping is the honest choice).
    cache = (
        _rcache.configured(store, cfg_root)
        if policy_tables is None
        else None
    )
    cfg_bits = (
        controller_kind,
        cap_cfg,
        bool(getattr(retrieval_cfg, "dense", False)),
        bool(getattr(retrieval_cfg, "sparse", False)),
        bool(getattr(retrieval_cfg, "late_interaction", False)),
        bool(getattr(retrieval_cfg, "graph", False)),
        bool(getattr(retrieval_cfg, "causal", False)),
    )

    deadline = _cand.Deadline(getattr(request, "deadline_ms", None))

    receipt_id = f"rc_{now_us():x}"
    decision_id: Optional[str] = None
    stored_key: Optional[str] = None
    packs: list = []
    omitted = 0
    abstained = False
    capabilities: dict = {"degraded": [], "lanes": {}}
    capabilities["detail_tier"] = detail.value
    capabilities["pack_mode"] = pack_mode
    if manifest_mode is not None:
        # Placeholder until the manifest doc is assembled at pack time —
        # the mode declaration alone is honest on early-abstain returns.
        capabilities["manifest"] = {"mode": manifest_mode}
    capabilities["cache"] = {
        "enabled": cache is not None,
        "hit": False,
    }

    with store.read() as conn:
        # The generation must come from THIS snapshot — a bump+rebuild
        # committing between a separate reader probe and the snapshot
        # open would filter lanes against rows the snapshot never saw.
        meta_get = getattr(store, "_meta_get", None)
        if callable(meta_get):
            gen_value = meta_get(conn, "projection_generation")
        else:
            try:
                row = conn.execute(
                    "SELECT value_json FROM meta"
                    " WHERE key = 'projection_generation'"
                ).fetchone()
                gen_value = safe_json_loads(row[0]) if row else None
            except sqlite3.Error:
                gen_value = None
        if gen_value is None:
            warnings.append("projection_generation_unavailable")
        generation = int(gen_value or 0)

        scope_ids, epoch, _grant_path = _authorized_scopes(conn, request)

        # Honest L2 degradation (V4-32.05): an exact-detail ask with no
        # ``quote`` on ANY contributing scope cannot release a single
        # byte — the pack still ships (each exact item renders its
        # read-authorized metadata view with ``quote_withheld``), and a
        # tier-level warning names the collapse so the caller is never
        # left believing exact detail was delivered.
        if detail is _pack.DetailTier.L2 and not _any_quote_scope(
            conn, request, scope_ids
        ):
            warnings.append("detail_tier_collapsed:quote_unauthorized")
            capabilities["delivered_detail"] = "metadata"

        # readiness gate (V3-27.09): min_ready_seq is an event-seq caller
        # value — compare against MAX(event_seq) inside this snapshot
        # (v2 semantics), never the projection generation.
        if request.min_ready_seq is not None:
            try:
                max_seq = conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM events"
                ).fetchone()[0]
            except sqlite3.Error:
                max_seq = 0
            if int(max_seq) < request.min_ready_seq:
                warnings.append("processing_pending")
                return RecallResultV3(
                    packs=(),
                    omitted=0,
                    warnings=tuple(dict.fromkeys(warnings)),
                    capabilities=capabilities,
                    decision_id=None,
                    projection_generation=generation,
                    abstained=True,
                )

        query_class = classify(request, plan)
        state = _ctrl.discretize(conn, request, query_class, list(scope_ids))
        routeset = _ctrl.plan_routes(
            request, plan, request.task, state, tables=policy_tables
        )
        # Learned controller channel (V3-26.03): replay arms carry the
        # artifact inside policy_tables["controller"] — {"kind":
        # "replay_learned"|"learned_shadow", "artifact": ref}. Applied
        # learned plans exist only inside sandbox replays; production
        # config reaches them solely through learned_shadow logging.
        ctrl_override = (policy_tables or {}).get("controller") or {}
        learned_kind = ctrl_override.get("kind")
        artifact = (
            _learned.load_artifact(ctrl_override.get("artifact"))
            if ctrl_override.get("artifact") is not None
            else None
        )
        if learned_kind == "forced":
            # replay arm pins one bounded action directly — documented
            # action support for off-policy estimation (V3-43.04)
            forced = ctrl_override.get("action", "base")
            if forced not in _learned.ACTIONS:
                warnings.append(
                    f"forced_action_unknown:{forced}:deterministic_fallback"
                )
                forced = "base"
            routeset = dataclasses.replace(
                _learned.apply_delta(routeset, state, request, forced),
                policy_revision=f"forced:{forced}",
            )
            if forced != "base":
                warnings.append(f"forced_action:{forced}")
        elif learned_kind == "replay_learned":
            if artifact is not None:
                chosen, action, _scores = _learned.choose(
                    state, request, routeset, artifact
                )
                routeset = dataclasses.replace(
                    chosen,
                    policy_revision=f"learned:{artifact.revision}",
                )
                if action != "base":
                    warnings.append(f"learned_action:{action}")
            else:
                warnings.append(
                    "learned_artifact_missing:deterministic_fallback"
                )
        if controller_kind == "learned_shadow" or (
            learned_kind == "learned_shadow"
        ):
            # The learned policy is hypothetical here; its recorded choice
            # is logged, never applied (V3-26.03).
            if artifact is not None:
                shadow_set, action, scores = _learned.choose(
                    state, request, routeset, artifact
                )
                shadow = {
                    "routes": list(shadow_set.routes),
                    "lanes": list(shadow_set.lanes),
                    "budgets": dict(shadow_set.budgets),
                    "action": action,
                    "scores": scores,
                    "note": f"learned_shadow: artifact {artifact.revision}",
                }
            else:
                shadow = {
                    "routes": list(routeset.routes),
                    "lanes": list(routeset.lanes),
                    "budgets": dict(routeset.budgets),
                    "note": "learned_shadow: deterministic warm start",
                }
            routeset = _ctrl.RouteSet(
                routes=routeset.routes,
                lanes=routeset.lanes,
                budgets=routeset.budgets,
                state_key=routeset.state_key,
                policy_revision=routeset.policy_revision,
                shadow=shadow,
            )

        if learned_kind is None and controller_kind == "learned_active":
            # V6-03.15/16: THE production activation gate — enforced here,
            # at the point the controller's effective routeset is
            # constructed inside the read snapshot. Config validation
            # already required the binding to be DECLARED
            # (v3.retrieval.controller_policy_artifact); the store half
            # checks the bound id against THIS store's policy_artifacts:
            # the artifact must exist, be validation_state='validated',
            # and load as learned_policy_v1. Missing/unvalidated/
            # revoked/corrupt bindings refuse CONFIG_INVALID naming the
            # artifact — never a silent deterministic fallback (the
            # fallback belongs to shadow/replay arms, V3-26.03).
            bound = _polart.bind_for_activation(
                cfg_root,
                getattr(retrieval_cfg, "controller_policy_artifact", None),
                store,
                conn=conn,
            )
            chosen, action, _scores = _learned.choose(
                state, request, routeset, bound
            )
            routeset = dataclasses.replace(
                chosen,
                policy_revision=f"learned:{bound.revision}",
            )
            if action != "base":
                warnings.append(f"learned_action:{action}")

        # A term-less plan only lacks *lexical* signal; identifier lanes,
        # structured lanes, and the semantic lane may still answer.
        if plan.empty and query_class not in _BROWSE_CLASSES and not (
            request.entity_ids
        ) and not _semantic_possible(conn, store):
            warnings.append("no_signal")
            return RecallResultV3(
                packs=(),
                omitted=0,
                warnings=tuple(dict.fromkeys(warnings)),
                capabilities=capabilities,
                decision_id=None,
                projection_generation=generation,
                abstained=True,
            )

        # ---- final-pack cache probe (V4-33): AFTER every early gate so a
        # hit can never bypass readiness/no-signal abstentions. A matching
        # entry was revalidated against the CURRENT snapshot inside probe()
        # — fingerprint (scope epochs, policy/erasure epochs, generation,
        # schema, data_version, content watermark) plus per-ref quarantine/
        # purge/head checks — so a hit is fresh-state confirmation, never
        # an authorization decision (V4-33.02).
        probe = (
            cache.probe(
                conn, store, request, scope_ids=scope_ids,
                generation=generation, detail=detail, cfg_bits=cfg_bits,
            )
            if cache is not None
            else None
        )
        if probe is not None and probe.entry is not None:
            entry = probe.entry
            # A hit still re-delivers: mint FRESH handles per served item
            # (the serialized text never embeds handle ids, so the replay
            # stays byte-identical) — the write phase below then records
            # real new exposure rows. Replaying the original handles
            # would silently fold a distinct delivery into the old rows
            # (INSERT OR IGNORE) and under-report influence (V4-33.02).
            seq_hit = [0]
            packs = []
            for _p in entry.packs:
                _items = []
                for _i in _p.items:
                    seq_hit[0] += 1
                    _h = _infl.mint_handle(
                        receipt_id, request.caller_id, epoch, _p.kind,
                        _i.handle.object_kind, _i.handle.object_id,
                        _i.handle.revision, seq=seq_hit[0],
                    )
                    _items.append(dataclasses.replace(_i, handle=_h))
                packs.append(dataclasses.replace(
                    _p, items=tuple(_items)))
            omitted = entry.omitted
            routeset = entry.routeset
            state = entry.state
            capabilities = dict(entry.capabilities)
            capabilities["cache"] = {"enabled": True, "hit": True}
            warnings.extend(entry.warnings)
            warnings.append("cache_hit")
            learned_kind = (
                "learned_shadow" if controller_kind == "learned_shadow"
                else None
            )
        else:
            lane_ctx = _lanes.LaneContext(
                conn=conn,
                store=store,
                request=request,
                plan=plan,
                query_class=query_class,
                scope_ids=scope_ids,
                generation=generation,
                deadline=deadline,
                lane_cap=routeset.budgets["lane_cap"],
                cfg=retrieval_cfg,
                # V4-30.09: the controller's declared tier hop budget is the
                # graph lane's hop allowance — explicit, never assumed. Node/
                # edge/fanout bounds stay at the spec defaults unless the
                # controller declares wider ones.
                graph_hops=max(0, min(
                    int(routeset.budgets.get("hop_depth") or 0), 4)),
                graph_max_nodes=max(1, min(
                    int(routeset.budgets.get("graph_max_nodes") or 100),
                    1000)),
                graph_max_edges=max(1, min(
                    int(routeset.budgets.get("graph_max_edges") or 200),
                    2000)),
                graph_fanout=max(1, min(
                    int(routeset.budgets.get("graph_fanout") or 32), 256)),
            )
            lane_results = _lanes.run_lanes(lane_ctx, list(routeset.lanes))
            for res in lane_results:
                capabilities["lanes"][res.lane] = res.status
                if res.details:
                    capabilities.setdefault("lane_details", {})[res.lane] = (
                        res.details
                    )
                if res.status in ("unavailable", "degraded", "partial"):
                    capabilities["degraded"].append(res.lane)
                    if res.reason:
                        warnings.append(f"{res.lane}:{res.reason}")

            lane_weights = _fuse.LANE_WEIGHTS
            if policy_tables:
                lane_weights = policy_tables.get("lane_weights", lane_weights)
            union = _union.build_union(
                lane_ctx,
                lane_results,
                candidate_cap=min(cap_cfg, routeset.budgets["candidate_cap"]),
                lane_weights=lane_weights,
            )
            warnings.extend(union.warnings)
            covering = {
                k for k, h in union.items()
                if getattr(h, "time_compat", None) == "compatible"
            }
            ranked = _fuse.fuse(
                union, covering_keys=covering,
                query_point_us=plan.valid_at_us,
            )
            max_items = routeset.budgets["max_items"]
            ranked_top = ranked[: max(max_items * 4, max_items)]

            # group assembly → abstention verdicts → packs. The tier
            # selects the metadata-only projection below L2 — payload
            # bytes are never even read (V4-32.05).
            request_context = _union._request_context(request)
            groups = _pack.assemble_groups(
                lane_ctx, union, ranked_top, request_context,
                detail=detail,
            )
            abstain_overrides = (
                (policy_tables or {}).get("abstain") or {}
            )
            verdict = _abstain_fn(
                conn, union, ranked_top, request, plan, query_class,
                list(scope_ids), generation, groups=groups,
                term_floor_ratio=abstain_overrides.get("term_floor_ratio"),
            )
            warnings.extend(verdict.reason_codes)
            assembled = None
            if verdict.abstained:
                abstained = True
                packs = []
            else:
                # Only supported groups enter answer-support packs (V3-29.05):
                # individually-insufficient groups (weak coverage, broken
                # dependencies for answer classes) are dropped before packing.
                # PARTIAL groups still ship labeled for exploratory classes.
                if verdict.group_verdicts:
                    insuff = {
                        k for k, v in verdict.group_verdicts.items()
                        if v == GroupSupport.INSUFFICIENT
                    }
                    if insuff:
                        ranked_top = [
                            e for e in ranked_top if e.key not in insuff
                        ]
                        warnings.append("insufficient_groups_omitted")
                seq = [0]

                def _mint(pack_kind, kind, oid, rev):
                    seq[0] += 1
                    return _infl.mint_handle(
                        receipt_id, request.caller_id, epoch,
                        pack_kind, kind, oid, rev, seq=seq[0],
                    )

                # Expansion refs (V4-32.09): bound to this delivery's
                # caller/scope/purpose/epoch and the item's pinned
                # revision; deterministic in those fields so identical
                # deliveries render byte-identical items — expanding
                # re-resolves against the caller's stored influence rows.
                def _seal(handle, kind, oid, rev):
                    return _pack._expand_seal(
                        store, scope_id=request.scope_id,
                        caller_id=request.caller_id,
                        purpose=request.purpose, epoch=epoch,
                        object_kind=kind, object_id=oid, revision=rev,
                    )

                assembled = _pack.assemble_packs(
                    lane_ctx, union, ranked_top, request_context, _mint,
                    budgets=routeset.budgets, groups=groups,
                    detail=detail, expand_seal=_seal,
                    manifest_mode=manifest_mode, pack_mode=pack_mode,
                )
                packs = assembled.packs
                omitted = assembled.omitted + union.overflow
                warnings.extend(assembled.warnings)
                if assembled.screened:
                    warnings.append("screened_items")
                if assembled.coverage is not None:
                    # V45-05.02: the declared coverage rule and its honest
                    # outcome ride the result for inspection/measurement.
                    capabilities["sufficiency"] = assembled.coverage
                # drop partial-verdict groups' packs when unsupported
                if verdict.verdict == GroupSupport.PARTIAL:
                    warnings.append("partial_evidence")
            if manifest_mode is not None:
                # V45-03.01/D01: the manifest is the inspectable record of
                # what the delivery is made of — built in BOTH the
                # delivered and abstained branches so an insufficiency
                # answer still declares its contrary accounting. Its
                # labels join the warning surface so callers who never
                # read the manifest still see "contrary_evidence_*".
                mdoc = _manifest.build_manifest(
                    conn=conn, request=request, plan=plan,
                    query_class=query_class, scope_ids=list(scope_ids),
                    union=union, ranked=ranked_top, groups=groups,
                    group_status=(
                        assembled.group_status if assembled is not None
                        else {}
                    ),
                    verdict=verdict, packs=packs,
                    withheld_refs=(
                        assembled.withheld_refs if assembled is not None
                        else []
                    ),
                    screened_refs=(
                        assembled.screened_refs if assembled is not None
                        else []
                    ),
                    detail_tier=detail.value, pack_mode=pack_mode,
                    generation=generation,
                )
                capabilities["manifest"] = mdoc
                warnings.extend(mdoc["insufficiency_labels"])
            # Record the fresh result under the probed key — only
            # delivered (non-abstained, non-empty) packs are cached, so a
            # hit can never replay an abstention or a degraded empty.
            if probe is not None and packs:
                cache.store(
                    conn, store, probe.key, request=request,
                    scope_ids=scope_ids, generation=generation,
                    detail=detail, packs=packs, omitted=omitted,
                    warnings=tuple(dict.fromkeys(warnings)),
                    capabilities=capabilities, routeset=routeset,
                    state=state,
                )
                stored_key = probe.key

    # ---- write phase: decision log + influence rows (one tx) -----------
    if store is not None:
        def _record(wconn: sqlite3.Connection) -> Any:
            did = _ctrl.log_decision(
                wconn, request.scope_id, routeset,
                result_sizes=_result_sizes(packs),
                state=state, request=request,
            )
            if controller_kind == "learned_shadow" or (
                learned_kind == "learned_shadow"
            ):
                _ctrl.record_shadow(wconn, request.scope_id, routeset,
                                    state=state, request=request)
            _infl.record_deliveries(
                wconn,
                (item.handle for p in packs for item in p.items),
                request.scope_id,
            )
            return did

        try:
            decision_id = _record_tx(store, _record)
        except VerbatimError:
            raise
        except sqlite3.Error as exc:
            raise VerbatimError(
                ErrorCode.INTEGRITY, f"recall record write failed: {exc}"
            ) from exc

    # The write phase just committed journal rows — the stored entry's
    # render-time fingerprint carries a pre-write ``data_version`` that
    # can never match a post-write probe. Re-stamp it on a fresh reader:
    # if anything beyond the excluded journal tables changed in the
    # window the entry is evicted rather than risked (V4-33.02). A
    # failed settle keeps the render-time fingerprint — it can only
    # mismatch probes, never serve stale.
    if stored_key is not None and cache is not None:
        try:
            with store.read() as sconn:
                cache.settle(
                    sconn, store, stored_key,
                    scope_ids=scope_ids, generation=generation,
                )
        except (sqlite3.Error, VerbatimError, AttributeError):
            pass

    if not packs and not abstained:
        warnings.append("no_authorized_evidence")
    return RecallResultV3(
        packs=tuple(packs),
        omitted=omitted,
        warnings=tuple(dict.fromkeys(warnings)),
        capabilities=capabilities,
        decision_id=decision_id,
        projection_generation=generation,
        abstained=abstained,
    )


# ---------------------------------------------------------------------------
# context_v3 — task-aware assembly (§27.02)
# ---------------------------------------------------------------------------


def context_v3(store: Any, request: ContextRequestV3,
               cfg: Any = None) -> RecallResultV3:
    """Assemble requested pack kinds under one budget (§27.02, §30).

    Reuses the recall pipeline with a synthesized RecallRequestV3; the
    ``packs`` tuple restricts assembly to the caller's kinds (an explicit
    ask, not a controller decision).
    """
    recall_request = RecallRequestV3(
        query=request.query,
        scope_id=request.scope_id,
        caller_id=request.caller_id,
        purpose=request.purpose,
        task=request.task,
        max_bytes=request.max_bytes,
        target_tokens=request.target_tokens,
        deadline_ms=request.deadline_ms,
        budget_tier=request.budget_tier,
        manifest=getattr(request, "manifest", None),
        pack_mode=getattr(request, "pack_mode", "standard"),
    )
    result = recall_v3(
        store, recall_request, cfg=cfg,
        detail_tier=getattr(request, "detail_tier", None),
    )
    if not request.packs:
        return result
    wanted = set(request.packs)
    kept = tuple(p for p in result.packs if p.kind in wanted)
    return RecallResultV3(
        packs=kept,
        omitted=result.omitted + (
            sum(len(p.items) for p in result.packs) -
            sum(len(p.items) for p in kept)
        ),
        warnings=result.warnings,
        capabilities=result.capabilities,
        decision_id=result.decision_id,
        projection_generation=result.projection_generation,
        abstained=result.abstained,
    )
