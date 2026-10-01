"""Public HTTP API application (SPEC_V4 §44/§45).

A transport adapter, not a memory implementation: every route validates
its JSON body, maps the caller to a ``CallerContext`` whose grants are
*derived from persisted authority* — ``governance.authorize`` /
``effective_verbs`` over ``grants_v3`` is the only grant evaluation this
surface consults, and the ``Engine`` then applies its own scope fencing
on top. The HTTP layer itself decides nothing about who may see what
(single-authority rule).

Contract mapping:

- capture   → ``POST /v1/capture``      (verb ``ingest``; operation id required)
- recall    → ``POST /v1/recall``       (verb ``read``, optional purpose)
- inspect   → ``GET  /v1/claims/{id}``  (verb ``read``)
- remember  → ``POST /v1/remember``     (verb ``derive``)
- feedback  → ``POST /v1/feedback``     (verb ``derive``)
- correct   → ``POST /v1/corrections``  (verb ``derive`` → review proposal)
              ``POST /v1/effects``      (verb ``review`` → idempotent effect)
              ``POST /v1/reviews/{id}/approve|reject`` (verb ``review``)
- forget    → ``POST /v1/forget``       preview (verb ``admin``)
              ``POST /v1/forget/suppress``
              ``POST /v1/forget/{purge_id}/execute|lift``
              ``GET  /v1/purges``       suppression-state inventory
- status    → ``GET  /v1/status``       (verb ``read``)
- caps      → ``GET  /v1/capabilities`` (verb ``read``)
- drain     → ``POST /v1/drain``        (verb ``admin``)
- readiness → ``GET  /v1/readiness/{receipt_id}`` + ``.../wait``

Transport identity is launch-bound: the token file is the operator's
explicit provisioning act (V4-45.03); request bodies can narrow the
addressed partition but can never mint a principal or widen verbs.
"""

from __future__ import annotations

import base64
from typing import Any, Iterable, Optional

from .. import governance
from ..core.identity import scope_key
from ..core.types import (
    CallerContext,
    EffectProposal,
    ErrorCode,
    FeedbackKind,
    GrantKind,
    Provenance,
    RecallMode,
    RecallRequest,
    Scope,
    SourceEnvelope,
    SourceKind,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
    Visibility,
)
from ..core.types_v3 import Verb
from ..readiness import ingest_receipt_id
from ..storage.repos import has_table
from .auth import TokenAuthenticator, TokenCredential
from .httpd import Application, Request, Response, Router, denial

#: GrantKind authority each Verb carries into the Engine's caller fence.
#: ``act``/``hydrate`` have no v2-facade analogue — deliberately unmapped
#: rather than approximated by a wider grant.
_VERB_GRANTS = {
    Verb.READ.value: frozenset({GrantKind.READ_EVIDENCE}),
    Verb.QUOTE.value: frozenset({GrantKind.READ_EVIDENCE}),
    Verb.DERIVE.value: frozenset({GrantKind.PROPOSE}),
    Verb.INGEST.value: frozenset({GrantKind.INGEST}),
    Verb.REVIEW.value: frozenset({GrantKind.RESOLVE}),
    Verb.SHARE.value: frozenset({GrantKind.SHARE}),
    Verb.ADMIN.value: frozenset(GrantKind),
}

#: Grant classes provisioned by shorthand in token files.
GRANT_CLASSES = {
    "agent": frozenset(
        {Verb.READ.value, Verb.QUOTE.value, Verb.DERIVE.value, Verb.INGEST.value}
    ),
    "reviewer": frozenset(
        {
            Verb.READ.value,
            Verb.QUOTE.value,
            Verb.DERIVE.value,
            Verb.INGEST.value,
            Verb.REVIEW.value,
        }
    ),
    "operator": frozenset(v.value for v in Verb),
}

_MAX_ID = 256
_MAX_STR = 8192


def _bad(msg: str) -> VerbatimError:
    return VerbatimError(ErrorCode.VALIDATION, msg)


def _require_obj(body: Any) -> dict:
    if not isinstance(body, dict):
        raise _bad("request body must be a JSON object")
    return body


def _fields(body: dict, allowed: Iterable[str]) -> dict:
    unknown = set(body) - set(allowed)
    if unknown:
        raise _bad(f"unknown request fields {sorted(unknown)}")
    return body


def _str(body: dict, key: str, *, required: bool = False, max_len: int = _MAX_STR) -> Optional[str]:
    v = body.get(key)
    if v is None:
        if required:
            raise _bad(f"{key} is required")
        return None
    if not isinstance(v, str) or not v:
        raise _bad(f"{key} must be a non-empty string")
    if len(v) > max_len:
        raise _bad(f"{key} exceeds {max_len} characters")
    return v


def _int(body: dict, key: str, *, required: bool = False,
         lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[int]:
    v = body.get(key)
    if v is None:
        if required:
            raise _bad(f"{key} is required")
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise _bad(f"{key} must be an integer")
    if lo is not None and v < lo:
        raise _bad(f"{key} must be >= {lo}")
    if hi is not None and v > hi:
        raise _bad(f"{key} must be <= {hi}")
    return v


def _bool(body: dict, key: str) -> Optional[bool]:
    v = body.get(key)
    if v is None:
        return None
    if not isinstance(v, bool):
        raise _bad(f"{key} must be boolean")
    return v


def _enum(enum_cls, body: dict, key: str, default=None):
    v = body.get(key)
    if v is None:
        return default
    try:
        return enum_cls(str(v))
    except ValueError:
        raise _bad(f"unknown {key} {v!r}") from None


def _interval(raw: Any) -> Optional[TimeInterval]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _bad("interval must be an object")
    _fields(raw, ("from_us", "until_us", "precision", "timezone", "basis",
                  "start_kind", "end_kind", "from_us_hi", "until_us_hi"))
    kw: dict[str, Any] = {}
    for k in ("from_us", "until_us", "from_us_hi", "until_us_hi"):
        kw[k] = _int(raw, k, lo=0)
    for k in ("precision", "timezone", "basis", "start_kind", "end_kind"):
        s = _str(raw, k, max_len=64)
        if s is not None:
            kw[k] = s
    try:
        return TimeInterval(**kw)
    except VerbatimError:
        raise
    except Exception as exc:
        raise _bad(f"invalid interval: {exc}") from exc


def _targets(raw: Any) -> list[tuple[str, str, Optional[int]]]:
    """Request-side shape check for forget targets (purge re-validates)."""
    if not isinstance(raw, list) or not raw or len(raw) > 64:
        raise _bad("targets must be a non-empty list (<= 64)")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise _bad("each target must be an object")
        _fields(item, ("object_kind", "kind", "object_id", "revision"))
        kind = item.get("object_kind") or item.get("kind")
        oid = item.get("object_id")
        if not isinstance(kind, str) or not isinstance(oid, str) or not oid:
            raise _bad("targets need object_kind and object_id strings")
        if len(kind) > 64 or len(oid) > _MAX_ID:
            raise _bad("target identifiers exceed bounds")
        rev = item.get("revision")
        if rev is not None and (isinstance(rev, bool) or not isinstance(rev, int) or rev < 1):
            raise _bad("target revision must be an integer >= 1")
        out.append((kind, oid) if rev is None else (kind, oid, rev))
    return out


class ServiceApp(Application):
    """The /v1 JSON API bound to one Engine + one credential set."""

    realm = "verbatim"

    def __init__(self, engine: Any, credentials: Iterable[TokenCredential]) -> None:
        self._engine = engine
        self._auth = TokenAuthenticator(credentials)
        self._router = Router()
        self._routes()

    # -- auth + authorization (transport glue, no policy of its own) ----

    def authenticate(self, request: Request) -> Optional[TokenCredential]:
        return self._auth.authenticate(request.header("authorization"))

    def _home_scope(self, cred: TokenCredential) -> Scope:
        """The credential's bound partition (host default, pin-narrowed)."""
        base = self._engine.host.default_scope()
        pin = cred.scope or {}
        return Scope(
            profile_id=base.profile_id,
            principal_id=cred.principal_id,
            workspace_id=pin.get("workspace_id") or base.workspace_id,
            conversation_id=pin.get("conversation_id") or base.conversation_id,
            visibility=(
                pin["visibility"]
                if isinstance(pin.get("visibility"), Visibility)
                else Visibility(pin["visibility"])
            ) if pin.get("visibility") else base.visibility,
        )

    def _addressed_scope(self, cred: TokenCredential, fields: dict) -> Scope:
        """The request's partition: credential home narrowed by request
        fields — narrowing can never mint a different principal or widen
        visibility past what the grant evaluation then permits."""
        home = self._home_scope(cred)
        vis = fields.get("visibility")
        return Scope(
            profile_id=home.profile_id,
            principal_id=home.principal_id,
            workspace_id=fields.get("workspace_id") or home.workspace_id,
            conversation_id=fields.get("conversation_id") or home.conversation_id,
            visibility=Visibility(str(vis)) if vis else home.visibility,
        )

    def _authorize(
        self,
        cred: TokenCredential,
        addressed: Scope,
        verb: Verb,
        purpose: Optional[str] = None,
    ) -> CallerContext:
        """Grant evaluation through ``governance.authorize`` — the single
        authority — then a CallerContext carrying only the verbs the
        store actually permits (revocation narrows live)."""
        caller_v3 = governance.CallerV3(
            principal_id=cred.principal_id,
            session_id="http",
            host_id="service",
        )
        sid = scope_key(addressed)
        with self._engine.store.read() as conn:
            if not has_table(conn, "grants_v3"):
                # No grant table → no HTTP authority at all (fail closed).
                raise denial()
            governance.authorize(conn, caller_v3, sid, verb.value, purpose=purpose)
            verbs = set(governance.effective_verbs(conn, caller_v3, sid))
        verbs &= set(cred.verbs)  # the token's ceiling, never widened
        grants: set[GrantKind] = set()
        for v in verbs:
            grants |= set(_VERB_GRANTS.get(v, frozenset()))
        return CallerContext(
            profile_id=addressed.profile_id,
            principal_id=cred.principal_id,
            agent_id="http",
            session_id="http",
            workspace_id=addressed.workspace_id,
            conversation_id=addressed.conversation_id,
            grants=frozenset(grants),
            is_operator=Verb.ADMIN.value in verbs,
        )

    def dispatch(self, request: Request, cred: TokenCredential) -> Response:
        handler, params = self._router.match(request.method, request.path)
        data = handler(self, request, cred, params)
        if isinstance(data, Response):
            return data
        return Response(status=200, body=data)

    # -- routes ----------------------------------------------------------

    def _routes(self) -> None:
        r = self._router.add
        r("POST", "/v1/capture", ServiceApp._capture)
        r("POST", "/v1/recall", ServiceApp._recall)
        r("POST", "/v1/remember", ServiceApp._remember)
        r("POST", "/v1/feedback", ServiceApp._feedback)
        r("GET", "/v1/claims/{claim_id}", ServiceApp._inspect)
        r("POST", "/v1/corrections", ServiceApp._propose)
        r("POST", "/v1/effects", ServiceApp._apply_effect)
        r("GET", "/v1/reviews", ServiceApp._reviews_list)
        r("GET", "/v1/reviews/{review_id}", ServiceApp._review_show)
        r("POST", "/v1/reviews/{review_id}/approve", ServiceApp._review_approve)
        r("POST", "/v1/reviews/{review_id}/reject", ServiceApp._review_reject)
        r("POST", "/v1/forget", ServiceApp._forget_preview)
        r("POST", "/v1/forget/suppress", ServiceApp._forget_suppress)
        r("POST", "/v1/forget/{purge_id}/execute", ServiceApp._forget_execute)
        r("POST", "/v1/forget/{purge_id}/lift", ServiceApp._forget_lift)
        r("GET", "/v1/purges", ServiceApp._purges)
        r("POST", "/v1/drain", ServiceApp._drain)
        r("GET", "/v1/status", ServiceApp._status)
        r("GET", "/v1/capabilities", ServiceApp._capabilities)
        r("GET", "/v1/readiness/{receipt_id}", ServiceApp._readiness)
        r("POST", "/v1/readiness/{receipt_id}/wait", ServiceApp._wait_ready)

    # -- endpoint handlers (validation + delegation only) ----------------

    def _capture(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "text", "payload_b64", "kind", "speaker_id", "origin",
            "event_us", "provenance", "external_id", "source_id",
            "revision", "metadata", "operation_id",
            "workspace_id", "conversation_id", "visibility",
        ))
        operation_id = _str(body, "operation_id", required=True, max_len=_MAX_ID)
        text = body.get("text")
        b64 = body.get("payload_b64")
        if (text is None) == (b64 is None):
            raise _bad("capture requires exactly one of text / payload_b64")
        if text is not None:
            if not isinstance(text, str) or not text:
                raise _bad("text must be a non-empty string")
            payload = text.encode("utf-8")
        else:
            if not isinstance(b64, str):
                raise _bad("payload_b64 must be a base64 string")
            try:
                payload = base64.b64decode(b64, validate=True)
            except ValueError:
                raise _bad("payload_b64 is not valid base64") from None
            if not payload:
                raise _bad("payload must be non-empty")
        metadata = body.get("metadata") or {}
        if not isinstance(metadata, dict) or len(metadata) > 32:
            raise _bad("metadata must be an object (<= 32 keys)")
        metadata = {str(k): v for k, v in metadata.items()}
        metadata["operation_id"] = operation_id
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.INGEST)
        envelope = SourceEnvelope(
            origin=_str(body, "origin", max_len=128) or "http",
            source_kind=_enum(SourceKind, body, "kind", SourceKind.USER_MESSAGE),
            scope=scope,
            speaker_id=_str(body, "speaker_id", max_len=_MAX_ID),
            payload=payload,
            event_us=_int(body, "event_us", lo=0) or self._engine.host.now_us(),
            captured_us=self._engine.host.now_us(),
            provenance=_enum(Provenance, body, "provenance", Provenance.UNKNOWN),
            external_id=_str(body, "external_id", max_len=_MAX_ID),
            source_id=_str(body, "source_id", max_len=_MAX_ID),
            revision=_int(body, "revision", lo=1) or 1,
            metadata=metadata,
        )
        receipt = self._engine.ingest(envelope, caller=caller)
        return {
            "accepted": list(receipt.accepted),
            "rejected": [list(r) for r in receipt.rejected],
            "job_ids": list(receipt.job_ids),
            "duplicate": receipt.duplicate,
            "projection_generation": receipt.projection_generation,
            "receipts": {
                sid: ingest_receipt_id(sid, envelope.revision)
                for sid in receipt.accepted
            },
        }

    def _recall(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "query", "mode", "limit", "max_bytes", "deadline_ms",
            "valid_at_us", "valid_until_us", "known_at_seq",
            "entity_ids", "context", "purpose", "target_tokens",
            "workspace_id", "conversation_id", "visibility",
        ))
        query = _str(body, "query", required=True)
        purpose = _str(body, "purpose", max_len=128)
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.READ, purpose=purpose)
        entities = body.get("entity_ids")
        if entities is not None and (
            not isinstance(entities, list)
            or not all(isinstance(e, str) for e in entities)
        ):
            raise _bad("entity_ids must be a list of strings")
        context = body.get("context") or {}
        if not isinstance(context, dict):
            raise _bad("context must be an object")
        req = RecallRequest(
            query=query,
            scope=scope,
            mode=_enum(RecallMode, body, "mode", RecallMode.CURRENT),
            limit=_int(body, "limit", lo=1, hi=32) or 8,
            valid_at_us=_int(body, "valid_at_us", lo=0),
            valid_until_us=_int(body, "valid_until_us", lo=0),
            known_at_seq=_int(body, "known_at_seq", lo=0),
            entity_ids=tuple(entities or ()),
            max_bytes=_int(body, "max_bytes", lo=512, hi=24000) or 6000,
            context=context,
            deadline_ms=_int(body, "deadline_ms", lo=1, hi=10_000) or 200,
            target_tokens=_int(body, "target_tokens", lo=64, hi=8192) or 1536,
        )
        res = self._engine.recall(req, caller=caller)
        items = [
            {
                "claim_id": i.claim_id,
                "claim_revision": i.claim_revision,
                "text": i.text,
                "speaker": i.speaker_id,
                "lifecycle": i.lifecycle.value,
                "valid": i.valid_label,
                "historical": i.historical,
                "disputed": i.disputed,
                "reasons": list(i.reasons),
                "source": {
                    "id": i.span.source_id,
                    "revision": i.span.revision,
                    "start": i.span.start_byte,
                    "end": i.span.end_byte,
                },
            }
            for i in res.items
        ]
        return {
            "items": items,
            "omitted": res.omitted,
            "warnings": list(res.warnings),
            "capabilities": res.capabilities,
            "projection_generation": res.projection_generation,
        }

    def _inspect(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        claim_id = params["claim_id"]
        if len(claim_id) > _MAX_ID:
            raise _bad("claim_id too long")
        scope = self._addressed_scope(cred, request.query)
        caller = self._authorize(cred, scope, Verb.READ)
        return self._engine.inspect(claim_id, scope, caller=caller)

    def _remember(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "source_id", "start_byte", "end_byte", "predicate", "revision",
            "operation_id", "workspace_id", "conversation_id", "visibility",
        ))
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.DERIVE)
        claim_id = self._engine.remember(
            _str(body, "source_id", required=True, max_len=_MAX_ID),
            _int(body, "start_byte", required=True, lo=0),
            _int(body, "end_byte", required=True, lo=1),
            scope,
            predicate_suggestion=_str(body, "predicate", max_len=_MAX_ID),
            caller=caller,
            revision=_int(body, "revision", lo=1) or 1,
            operation_id=_str(body, "operation_id", max_len=_MAX_ID),
        )
        return {"claim_id": claim_id}

    def _feedback(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "claim_id", "kind", "workspace_id", "conversation_id", "visibility",
        ))
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.DERIVE)
        kind = _enum(FeedbackKind, body, "kind")
        if kind is None:
            raise _bad("kind is required")
        self._engine.feedback(
            _str(body, "claim_id", required=True, max_len=_MAX_ID),
            kind,
            scope,
            actor_id=cred.principal_id,
            caller=caller,
        )
        return {"recorded": True}

    def _propose(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        """Create a reviewable transition proposal (the correction path's
        front half — the review queue is the back half)."""
        body = _fields(_require_obj(request.body), (
            "claim_id", "effect", "reason", "successor_claim_id",
            "expected_revision", "interval",
            "workspace_id", "conversation_id", "visibility",
        ))
        effect = _str(body, "effect", required=True, max_len=32)
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.DERIVE)
        cmd = TransitionCommand(
            claim_id=_str(body, "claim_id", required=True, max_len=_MAX_ID),
            expected_revision=_int(body, "expected_revision", lo=1) or 0,
            effect=effect,
            actor_id=cred.principal_id,
            reason=_str(body, "reason", max_len=1024) or "",
            successor_claim_id=_str(body, "successor_claim_id", max_len=_MAX_ID),
            interval=_interval(body.get("interval")),
        )
        review_id = self._engine.propose_transition(cmd, scope, caller=caller)
        return {"review_id": review_id}

    def _apply_effect(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        """Apply one EffectProposal — the idempotent authorized mutation
        (operation_id required, V4-45.01)."""
        body = _require_obj(request.body)
        if not isinstance(body.get("targets"), list):
            raise _bad("targets must be a list of [object_id, revision] pairs")
        targets = []
        for t in body["targets"]:
            if (
                not isinstance(t, (list, tuple)) or len(t) != 2
                or not isinstance(t[0], str)
                or isinstance(t[1], bool) or not isinstance(t[1], int)
            ):
                raise _bad("targets must be [object_id, revision] pairs")
            targets.append((t[0], t[1]))
        proposal = EffectProposal(
            version=1,
            operation_id=str(body.get("operation_id") or ""),
            effect=str(body.get("effect") or ""),
            targets=tuple(targets),
            actor_id=cred.principal_id,
            reason=str(body.get("reason") or ""),
            successor_claim_id=(
                str(body["successor_claim_id"])
                if body.get("successor_claim_id") is not None else None
            ),
            params=body.get("params") if isinstance(body.get("params"), dict) else {},
        )
        # Grant gate before target dereference (engine re-checks inside).
        home = self._home_scope(cred)
        caller = self._authorize(cred, home, Verb.REVIEW)
        receipt = self._engine.apply_proposal(proposal, caller=caller)
        return receipt

    # -- review queue ------------------------------------------------------

    def _reviews_list(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        caller = self._authorize(cred, scope, Verb.REVIEW)
        _ = caller
        from ..storage.repos import ReviewsRepo

        with self._engine.store.read() as conn:
            rows = conn.execute(
                "SELECT review_id, proposed_effect_json, state,"
                " expected_versions_json, resolved_event FROM reviews"
                " WHERE scope_id = ? ORDER BY review_id",
                (scope_key(scope),),
            ).fetchall()
        import json as _json

        out = []
        for rid, effect_json, state, expected_json, resolved in rows:
            try:
                effect = _json.loads(effect_json)
            except ValueError:
                effect = {"unparseable": True}
            out.append({
                "review_id": rid,
                "state": state,
                "effect": effect,
                "expected_versions": _json.loads(expected_json or "{}"),
                "resolved_event": resolved,
            })
        return {"reviews": out}

    def _review_show(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        self._authorize(cred, scope, Verb.REVIEW)
        from ..storage.repos import ReviewsRepo

        row = ReviewsRepo(self._engine.store).get(params["review_id"])
        if row is None or row.get("scope_id") != scope_key(scope):
            raise denial()
        return row

    def _review_approve(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("operation_id", "workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.REVIEW)
        review_id = params["review_id"]
        operation_id = _str(body, "operation_id", max_len=_MAX_ID) or (
            f"review-approve:{review_id}"
        )
        return apply_review(
            self._engine, review_id, scope, operation_id,
            actor_id=cred.principal_id, caller=caller,
        )

    def _review_reject(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        self._authorize(cred, scope, Verb.REVIEW)
        review_id = params["review_id"]
        from ..storage.repos import EventsRepo, ReviewsRepo

        repo = ReviewsRepo(self._engine.store)
        with self._engine.store.read() as conn:
            row = conn.execute(
                "SELECT scope_id, state FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None or row[0] != scope_key(scope) or row[1] != "open":
            raise denial()
        with self._engine.store.tx() as conn:
            seq = EventsRepo(self._engine.store).append(
                conn, scope_key(scope), "review_rejected",
                cred.principal_id, {"review_id": review_id}, "http-1",
            )
            repo.resolve(conn, review_id, "rejected", seq)
        return {"review_id": review_id, "state": "rejected"}

    # -- forget / deletion closure ----------------------------------------

    def _forget_preview(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "targets", "workspace_id", "conversation_id", "visibility",
        ))
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.ADMIN)
        return self._engine.plan_purge(
            _targets(body.get("targets")), scope=scope, caller=caller,
            actor=cred.principal_id,
        )

    def _forget_suppress(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(_require_obj(request.body), (
            "targets", "workspace_id", "conversation_id", "visibility",
        ))
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.ADMIN)
        return self._engine.suppress(
            _targets(body.get("targets")), scope=scope, caller=caller,
            actor=cred.principal_id,
        )

    def _forget_execute(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("confirm", "workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.ADMIN)
        if _bool(body, "confirm") is not True:
            raise _bad("execute requires confirm=true (authenticated confirmation)")
        return self._engine.execute_purge(
            params["purge_id"], scope=scope, caller=caller,
        )

    def _forget_lift(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.ADMIN)
        return self._engine.lift_suppression(
            params["purge_id"], scope=scope, caller=caller,
        )

    def _purges(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        self._authorize(cred, scope, Verb.ADMIN)
        with self._engine.store.read() as conn:
            rows = conn.execute(
                "SELECT p.purge_id, p.state, p.requested_us, p.approved_us,"
                " p.completed_us, pt.object_kind, pt.object_id"
                " FROM purges p LEFT JOIN purge_targets pt"
                "   ON pt.purge_id = p.purge_id"
                " WHERE p.scope_id = ? ORDER BY p.requested_us",
                (scope_key(scope),),
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for pid, state, req, appr, done, kind, oid in rows:
            entry = out.setdefault(pid, {
                "purge_id": pid, "state": state, "requested_us": req,
                "approved_us": appr, "completed_us": done, "targets": [],
            })
            if kind is not None:
                entry["targets"].append({"object_kind": kind, "object_id": oid})
        return {"purges": sorted(out.values(), key=lambda p: p["purge_id"])}

    # -- operations --------------------------------------------------------

    def _drain(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("limit", "workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.ADMIN)
        limit = _int(body, "limit", lo=1, hi=4096) or 64
        return self._engine.drain_report(limit=limit, caller=caller)

    def _status(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        caller = self._authorize(cred, scope, Verb.READ)
        return self._engine.status(caller=caller)

    def _capabilities(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        caller = self._authorize(cred, scope, Verb.READ)
        status = self._engine.status(caller=caller)
        return {
            "capabilities": status["capabilities"],
            "transport": {
                "scheme": "http",
                "tls": "unimplemented",   # loopback-only reference transport
                "bind": "loopback",
                "authn": "bearer-token",
            },
            "integrity": status["integrity"],
        }

    def _readiness(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        scope = self._addressed_scope(cred, request.query)
        caller = self._authorize(cred, scope, Verb.READ)
        return self._engine.receipt_state(
            params["receipt_id"], scope=scope, caller=caller,
        )

    def _wait_ready(self, request: Request, cred: TokenCredential, params: dict) -> dict:
        body = _fields(
            request.body if isinstance(request.body, dict) else {},
            ("timeout_s", "capabilities",
             "workspace_id", "conversation_id", "visibility"),
        )
        scope = self._addressed_scope(cred, body)
        caller = self._authorize(cred, scope, Verb.READ)
        caps = body.get("capabilities")
        if caps is not None and (
            not isinstance(caps, list)
            or not all(isinstance(c, str) for c in caps)
            or len(caps) > 32
        ):
            raise _bad("capabilities must be a list of <= 32 strings")
        timeout = body.get("timeout_s", 30.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
                or not 0 < timeout <= 120:
            raise _bad("timeout_s must be in (0, 120]")
        return self._engine.wait_ready(
            params["receipt_id"],
            capabilities=caps,
            timeout_s=float(timeout),
            scope=scope,
            caller=caller,
        )


def _proposal_for_review(
    review: dict[str, Any], operation_id: str, actor_id: str
) -> EffectProposal:
    """Marshal a stored review row into the unified EffectProposal.

    Shared by the service and workbench — the same translation the CLI
    performs (admission reviews carry ``claim_id``; supersession reviews
    carry ``predecessor_id``/``successor_id``). ``params.review_id`` lets
    the engine resolve the review in the same transaction as the effect.
    """
    effect = review.get("proposed_effect") or {}
    expected = review.get("expected_versions") or {}
    kind = effect.get("effect")
    params: dict[str, Any] = {"review_id": review["review_id"]}
    successor = None
    if kind == "supersede":
        target = effect.get("predecessor_id") or effect.get("claim_id")
        successor = effect.get("successor_id") or effect.get("successor_claim_id")
        if not target or not successor:
            raise _bad("supersede review lacks predecessor/successor")
        targets = ((str(target), int(expected.get(str(target), 1))),)
        pinned = expected.get(str(successor))
        if pinned is not None:
            params["successor_expected_revision"] = int(pinned)
        if isinstance(effect.get("interval"), dict):
            params["interval"] = effect["interval"]
    else:
        target = effect.get("claim_id")
        if not target:
            raise _bad(f"review effect {kind!r} has no claim target")
        if kind == "dispute" and effect.get("conflict_with_claim_id"):
            params["conflict_with_claim_id"] = str(effect["conflict_with_claim_id"])
        targets = ((str(target), int(expected.get(str(target), 1))),)
    return EffectProposal(
        version=1,
        operation_id=operation_id,
        effect=str(kind),
        targets=targets,
        actor_id=actor_id,
        reason=str(effect.get("reason") or "review approval"),
        successor_claim_id=str(successor) if successor else None,
        params=params,
    )


def apply_review(
    engine: Any,
    review_id: str,
    scope: Scope,
    operation_id: str,
    *,
    actor_id: str,
    caller: Optional[CallerContext] = None,
) -> dict:
    """Approve one open review through the engine's real proposal path —
    effect + review resolution commit in ONE transaction (V2-20.04)."""
    from ..storage.repos import ReviewsRepo

    review = ReviewsRepo(engine.store).get(review_id)
    if review is None or review.get("scope_id") != scope_key(scope) \
            or review.get("state") != "open":
        raise denial()
    proposal = _proposal_for_review(review, operation_id, actor_id)
    receipt = engine.apply_proposal(proposal, caller=caller)
    return {
        "review_id": review_id,
        "state": "approved",
        "receipt": receipt,
    }


class _MountedApi(Application):
    """``ServiceApp`` with the V5 consumer surface mounted at /v2/memory.

    One credential set authenticates both surfaces (``authenticate`` is
    the /v1 authenticator); dispatch only forks on the path prefix — the
    memory app applies its own scope-binding check and flat error shape
    inside ``dispatch`` (docs/v6_contracts.md §4).
    """

    realm = "verbatim"

    def __init__(self, api: ServiceApp, memory_app: Application) -> None:
        self._api = api
        self._memory_app = memory_app

    @property
    def memory(self) -> Any:
        """The mounted ``Memory`` facade — the caller owns its close."""
        return self._memory_app.memory

    def authenticate(self, request: Request) -> Optional[TokenCredential]:
        return self._api.authenticate(request)

    def dispatch(self, request: Request, cred: TokenCredential) -> Response:
        if request.path == "/v2/memory" or request.path.startswith("/v2/memory/"):
            return self._memory_app.dispatch(request, cred)
        return self._api.dispatch(request, cred)


def create_api(
    engine: Any,
    credentials: Iterable[TokenCredential],
    *,
    enable_v5_memory: bool = False,
    memory_path: Optional[str] = None,
    memory_user: Optional[str] = None,
    memory_worker: str = "managed",
    memory_config: Any = None,
) -> Application:
    """Build the service application; optionally mount ``/v2/memory/*``.

    Default off keeps the /v1 surface byte-identical to before. When
    enabled, ONE ``Memory`` is constructed at mount time on
    ``memory_path``, bound to ``memory_user`` (or the single principal
    every credential shares) — a credential scoped elsewhere gets the
    memory surface's flat 403, and a config with no usable credential is
    ``CONFIG_INVALID`` at bind time (v6 contracts §4).
    """
    creds = tuple(credentials)
    app = ServiceApp(engine, creds)
    if not enable_v5_memory:
        return app
    from ..memory.facade import Memory
    from .memory_api import MemoryApp, resolve_bound

    bound = resolve_bound(creds, memory_user)
    memory = Memory(memory_path, user_id=bound, worker=memory_worker, config=memory_config)
    mem_app = MemoryApp(memory, creds, bound=bound)
    return _MountedApi(app, mem_app)


__all__ = ["GRANT_CLASSES", "ServiceApp", "apply_review", "create_api"]
