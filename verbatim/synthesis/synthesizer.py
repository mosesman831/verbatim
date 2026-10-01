"""Opt-in grounded synthesis and derived views (SPEC_V4 §20).

This module implements the only synthesis mode this environment can
honestly run: **deterministic grounded composition** — stitching
kernel-verified quotations and object metadata under declared section
rules. There is no neural producer here; requesting a generative view
kind returns ``CAPABILITY_UNAVAILABLE`` rather than simulating one
(V4-20.10, honest-capability rule).

Contract mapping:

- **Opt-in (V4-20.01)** — two explicit provisioning acts, neither of
  which can enable anything else: (1) a ``producer_manifests`` row
  registers the deterministic producer identity; (2) a live ``derive``
  grant on the scope authorizes derivation for the producer principal
  (or the caller in caller-as-producer mode). A ``derive`` grant never
  implies ``read``/``quote``/``share``/remote dispatch — the kernel and
  broker keep their own gates.
- **Labeled derivative, never impersonation (V4-20.02/03)** — every view
  is a registered ``objects`` row of kind ``derived_view`` with an
  ``object_revisions`` digest over a canonical descriptor doc; every
  statement carries ``view_support`` bindings to the exact evidence refs
  (``kind:id@rev`` + byte range) that ground it.
- **One read path (V4-08.09)** — all evidence bytes come from
  ``Kernel.derive_inputs`` (producer path) or ``Kernel.read_verified``
  under the caller's lease (caller path). This module never SELECTs a
  payload/text column: enumeration and freshness use metadata columns
  only (ids, revisions, byte ranges, states, ``length(payload)`` —
  mirroring ``kernel.service._full_range``).
- **Quote gating (V4-08.02)** — persisted view docs carry no canonical
  bytes; quote text is re-materialized at delivery through a fresh
  ``read_verified`` under the *delivering* caller's ``quote`` lease and
  digest-compared to the mint binding. Quote-denied callers receive
  metadata + support refs only.
- **Independent output screening (V4-20.06)** — every statement's
  deliverable text passes ``security.screening.screen_content``
  (rules_v1) at composition; any finding omits the statement.
- **Invalidation (V4-20.07, C49)** — views record ``dependency_edges``
  to each pinned parent revision, so ``kernel.invalidate`` cascades to
  dependent views (``affected`` + ``reevaluate`` obligations) and a
  quarantined ancestor makes the kernel deny the view object itself.
  ``deliver`` additionally re-verifies every bound ref (access +
  pinned-revision equality + absent-byte probe) on each call; any
  withheld, moved, or corrected parent durably suppresses the view
  (``disposition='held'``) and raises ``STALE_DEPENDENCY`` until a
  rebuild mints a new revision. ``note_invalidation`` is the optional
  eager hook that marks views held inside the same transaction as
  ``kernel.invalidate``.
- **Audience/purpose inheritance (V4-10.07/20.06, C48)** — the producer
  path delegates to ``kernel.derive_inputs``: the view's
  ``allowed_purposes`` and ``effective_audience`` are the kernel-computed
  intersections over all contributing scopes, and delivery requires the
  caller's authorization in *every* contributing scope.
- **Failure honesty (V4-20.08)** — a failed compose rolls back its whole
  transaction (no partial view); denials are typed, never silent.

Persistence decision (recorded per worker contract): §41's relation list
names ``view_support`` for grounding but no view-payload table, so this
module deliberately adds NO new table. Views persist on the existing v4
kernel tables: ``objects`` + ``object_revisions`` (revisioned, digest-
bound, disposition-gated lifecycle), ``view_support`` (per-statement
evidence bindings), ``dependency_edges`` (validated ancestry), and
``producer_manifests`` (producer identity). The serialized view document
lives in ``object_revisions.metadata_json`` — statements carry locators,
digests, labels, and field renderings but NEVER canonical bytes, so a
stale view physically cannot leak quoted text.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ..core import time as _time
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import Verb
from ..core.types_v4 import (
    CapabilityReport,
    CapabilityRung,
    EligibilityLease,
    EvidenceLocator,
    LifecycleState,
    PurposeConstraint,
)
from ..governance import CallerV3
from ..governance import grants as _grants
from ..security.screening import screen_content
from ..storage import repos_v4

from .types import (
    Confidence,
    DerivedView,
    StatementForm,
    SupportBinding,
    SupportVerdict,
    SYNTHESIS_MODE,
    SUPPORTED_VIEW_KINDS,
    VIEW_OBJECT_KIND,
    VIEW_SUPPORT_KIND,
    ViewKind,
    ViewStatement,
)

#: Producer manifest identity for the built-in deterministic composer.
DEFAULT_PRODUCER_ID = "producer:verbatim.grounded-composition.v1"

#: Pinned rubric/config identity recorded in the producer manifest and
#: every view doc (V4-20.02) — bump when section/selection rules change.
RUBRIC_REVISION = "grounded-composition.rubric.v1"

#: Registered kinds with a verified byte surface (mirrors the kernel's
#: ``_BYTE_KINDS`` — only these can underwrite quote statements).
_BYTE_KINDS = frozenset({"source", "span"})

#: Claim revision states still visible (mirrors kernel._CLAIM_VISIBLE_STATES).
_CLAIM_VISIBLE_STATES = ("active", "disputed")

#: Bounds so a hostile or pathological scope cannot turn composition into
#: an unbounded scan (fixed-allowlist convention).
_MAX_CANDIDATES = 2048
_MAX_STATEMENTS = 256
_DEFAULT_STATEMENT_CAP = 64
_MAX_QUOTE_BYTES = 16_384
_MAX_SPANS_PER_CLAIM = 64

#: Section tokens a caller may declare (ordering = fill order).
_SECTIONS = ("claims", "sources", "evidence")

_DENIAL = "not found or unauthorized"

_DOC_FORMAT = "derived_view/v1"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _deny() -> "None":
    raise VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, _DENIAL)


def _terms(text: str) -> frozenset:
    return frozenset(t.lower() for t in _WORD_RE.findall(text))


def _doc_digest(hmac_fn, doc: Mapping[str, Any]) -> str:
    """Keyed digest binding the persisted view document (tamper-evident)."""
    return "hmac-sha256:" + hmac_fn(json_dumps(doc).encode("utf-8")).hex()


def _request_digest(parts: Mapping[str, Any]) -> str:
    """Binds the compose request without retaining raw prompt text
    (V4-20.09: structured provenance, not raw prompts)."""
    return "sha256:" + hashlib.sha256(
        json_dumps(parts).encode("utf-8")
    ).hexdigest()


class Synthesizer:
    """Deterministic grounded composer over one ``Store``.

    All authorization, integrity, and lifecycle evaluation is delegated to
    :class:`verbatim.kernel.Kernel`; this class adds selection, statement
    composition, support bookkeeping, and view lifecycle only.
    """

    def __init__(
        self,
        store: Any,
        *,
        kernel: Any = None,
        producer_id: str = DEFAULT_PRODUCER_ID,
    ) -> None:
        self._store = store
        if kernel is None:
            from ..kernel import Kernel

            kernel = Kernel(store)
        self._kernel = kernel
        self._producer_id = require_id(producer_id, "producer_id")

    # ------------------------------------------------------------------
    # producer provisioning (per-producer opt-in, V4-20.01/02)
    # ------------------------------------------------------------------

    def register_producer(
        self,
        conn: Optional[sqlite3.Connection] = None,
        *,
        producer_id: Optional[str] = None,
        license_ref: Optional[str] = None,
        registered_us: Optional[int] = None,
    ) -> str:
        """Register the deterministic producer manifest (idempotent).

        Records artifact/rubric/config digests pinning exactly what
        produced views under this identity (V4-20.02). No artifact bytes
        exist to digest, so ``artifact_digest`` binds the code identity
        instead. Returns the producer id.
        """
        pid = require_id(producer_id or self._producer_id, "producer_id")
        descriptor = {
            "producer_id": pid,
            "synthesis_mode": SYNTHESIS_MODE,
            "rubric": RUBRIC_REVISION,
            "quote_cap_bytes": _MAX_QUOTE_BYTES,
            "statement_cap": _MAX_STATEMENTS,
        }
        row = {
            "producer_id": pid,
            "kind": SYNTHESIS_MODE,
            "artifact_digest": _request_digest(
                {"producer_code": "verbatim.synthesis", **descriptor}
            ),
            "rubric_digest": _request_digest({"rubric": RUBRIC_REVISION}),
            "config_digest": _request_digest(descriptor),
            "schema_version": 4,
            "license_ref": license_ref,
            # A deterministic composer needs no external artifact: once
            # registered it IS available — honest health.
            "health": "available",
            "registered_us": (
                int(registered_us) if registered_us is not None else _time.now_us()
            ),
        }

        def _write(c: sqlite3.Connection) -> None:
            if (
                repos_v4.get(c, "producer_manifests", {"producer_id": pid})
                is None
            ):
                repos_v4.insert(c, "producer_manifests", row)

        if conn is not None:
            _write(conn)
        else:
            with self._store.tx() as c:
                _write(c)
        return pid

    def _producer_row(self, conn: sqlite3.Connection) -> Optional[dict]:
        return repos_v4.get(
            conn, "producer_manifests", {"producer_id": self._producer_id}
        )

    # ------------------------------------------------------------------
    # capabilities — honest rung reporting (V4-50)
    # ------------------------------------------------------------------

    def capabilities(self) -> CapabilityReport:
        """This module's observed rung: grounded composition is real code
        (IMPLEMENTED); neural/hypothesis/user-edited producers are named
        unavailable, never simulated."""
        with self._store.read() as conn:
            provisioned = self._producer_row(conn) is not None
        return CapabilityReport(
            name="synthesis.grounded_composition",
            rung=CapabilityRung.IMPLEMENTED,
            degraded_reason=None if provisioned else "producer_not_registered",
            details={
                "synthesis_mode": SYNTHESIS_MODE,
                "rubric": RUBRIC_REVISION,
                "producer_registered": provisioned,
                "supported_view_kinds": sorted(SUPPORTED_VIEW_KINDS),
                "unavailable_view_kinds": sorted(
                    k.value for k in ViewKind
                    if k.value not in SUPPORTED_VIEW_KINDS
                ),
            },
            observed_us=_time.now_us(),
        )

    # ------------------------------------------------------------------
    # compose (V4-20.01–05)
    # ------------------------------------------------------------------

    def compose(
        self,
        scope_id: str,
        query: Optional[str] = None,
        *,
        topic: Optional[str] = None,
        caller: CallerV3,
        purpose: Optional[str] = "recall",
        view_kind: str = "extractive_digest",
        inputs: Optional[Iterable[Any]] = None,
        sections: Optional[Iterable[str]] = None,
        audience: Iterable[str] = (),
        producer_grant: Any = None,
        grant_check: Optional[Callable[..., bool]] = None,
        max_statements: int = _DEFAULT_STATEMENT_CAP,
        persist: bool = True,
        now_us: Optional[int] = None,
    ) -> DerivedView:
        """Compose a support-bound derived view over scope evidence.

        ``query``/``topic`` (synonyms — at most one) select candidates by
        deterministic term coverage over *verified* text; ``inputs`` pins
        an explicit evidence set instead. Every emitted statement binds
        its supporting refs; unsupported candidates are omitted and
        counted, never shipped (V4-20.05).

        Authorization: the caller must be able to receive the view
        (``read`` on the scope). Composition itself requires either a
        live ``producer_grant`` carrying ``derive`` (producer path —
        ``kernel.derive_inputs`` verifies inputs and computes inherited
        restrictions) or the caller's own ``derive`` grant
        (caller-as-producer). Quote-byte access for composition comes
        from the producer's ``quote`` grants or the caller's ``quote``
        lease; ``typed_summary`` needs metadata only.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        scope_id = self._scope_id(scope_id)
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        vkind = self._view_kind(view_kind)
        if query is not None and topic is not None:
            raise VerbatimError(
                ErrorCode.VALIDATION, "pass query or topic, not both"
            )
        topic_text = query if query is not None else topic
        query_terms: Optional[frozenset] = None
        if topic_text is not None:
            if not isinstance(topic_text, str):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "query/topic must be a string"
                )
            query_terms = _terms(topic_text)
            if not query_terms:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "query/topic has no matchable terms"
                )
        if (
            isinstance(max_statements, bool)
            or not isinstance(max_statements, int)
            or not (1 <= max_statements <= _MAX_STATEMENTS)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"max_statements must be in [1, {_MAX_STATEMENTS}]",
            )
        section_order = self._sections(sections, vkind)
        audience = tuple(
            require_id(str(a), "audience member")
            for a in dict.fromkeys(audience or ())
        )
        explicit = self._normalize_inputs(inputs)

        with self._store.read() as conn:
            # Opt-in gate 1: registered producer (V4-20.01).
            if self._producer_row(conn) is None:
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "synthesis producer not provisioned — "
                    "call register_producer() to opt in",
                )
            # The caller must be able to receive a view on this scope at
            # all — compose never mints views for unauthorized callers.
            self._require_verb(
                conn, caller, (scope_id,), Verb.READ, purpose,
                grant_check, now,
            )

            # Opt-in gate 2: derive authority — a producer grant carrying
            # derive (producer path), or the caller's own derive grant.
            reader: Optional[CallerV3] = None
            read_verb = Verb.QUOTE
            if producer_grant is None:
                self._require_verb(
                    conn, caller, (scope_id,), Verb.DERIVE, purpose,
                    grant_check, now,
                )
                reader = caller
                # Byte access for composition: caller's quote lease when
                # held, else metadata-only read.
                if not self._allowed(
                    conn, caller, (scope_id,), Verb.QUOTE, purpose,
                    grant_check, now,
                ):
                    read_verb = Verb.READ
            if (
                vkind == ViewKind.EXTRACTIVE_DIGEST.value
                and producer_grant is None
                and read_verb is not Verb.QUOTE
            ):
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "extractive composition requires verified byte access "
                    "(caller quote grant or a producer_grant)",
                )
            if (
                query_terms is not None
                and producer_grant is None
                and read_verb is not Verb.QUOTE
            ):
                # Topic selection over canonical text needs verified
                # bytes; without them the match would be fabricated.
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "topic selection requires verified byte access",
                )

            candidates = (
                explicit
                if explicit is not None
                else self._enumerate(conn, scope_id, section_order)
            )

            # --- per-candidate verified reads (kernel path only) -------
            slots: list[dict[str, Any]] = []
            omitted = {"unsupported": 0, "screened": 0, "unmatched": 0}
            contributing: set[str] = set()
            for cand in candidates:
                slot = self._read_candidate(
                    conn,
                    cand,
                    producer_grant=producer_grant,
                    reader=reader,
                    read_verb=read_verb,
                    scope_id=scope_id,
                    purpose=purpose,
                    audience=audience,
                    now=now,
                )
                if slot is None:
                    omitted["unsupported"] += 1
                    continue
                contributing.add(slot["scope_id"])
                contributing.update(
                    s["scope_id"] for s in slot.get("span_slots") or ()
                )
                slots.append(slot)

            # --- selection under declared section rules -----------------
            ranked = self._select(slots, query_terms, vkind)
            if query_terms is not None:
                omitted["unmatched"] += len(slots) - len(ranked)
            ranked = ranked[:max_statements]

            # --- statement build + independent output screen (V4-20.06) -
            statements: list[ViewStatement] = []
            for slot in ranked:
                stmt = self._build_statement(slot, vkind)
                if stmt is None:
                    omitted["unsupported"] += 1
                    continue
                verdict = screen_content(stmt.text or "")
                if verdict.findings:
                    # Generated output is screened independently of its
                    # inputs; any finding omits the proposition.
                    omitted["screened"] += 1
                    continue
                statements.append(stmt)
            statements = self._sequence(statements, section_order)

            # --- inherited restrictions (V4-10.07) ----------------------
            contrib_scopes = tuple(sorted(contributing)) or (scope_id,)
            # C48: the caller must be able to receive the view on EVERY
            # contributing scope — a cross-scope view minted for a caller
            # denied any parent would leak bound-ref existence.
            self._require_verb(
                conn, caller, contrib_scopes, Verb.READ, purpose,
                grant_check, now,
            )
            bound_refs = self._bound_refs(statements)
            byte_refs = sorted(r for r in bound_refs if r[0] in _BYTE_KINDS)
            if producer_grant is not None and byte_refs:
                # One bundle call computes the inherited intersections over
                # the SURVIVING inputs — kernel-owned math, never
                # re-implemented here.
                derived = self._kernel.derive_inputs(
                    conn,
                    producer_grant,
                    [
                        {"kind": k, "id": i, "revision": r}
                        for (k, i, r) in byte_refs
                    ],
                    output_audience=audience,
                    output_purpose=purpose,
                    now_us=now,
                )
                allowed = derived.allowed_purposes
                eff_audience = derived.effective_audience
                epoch_vector = dict(derived.epoch_vector)
            else:
                allowed = PurposeConstraint.any()
                eff_audience = ()
                epoch_vector = {
                    sid: self._epoch(conn, sid) for sid in contrib_scopes
                }

            doc_inputs = self._doc_inputs(conn, statements, slots)
            inputs_digest_list = [
                _request_digest({"input": i}) for i in doc_inputs
            ]
            request_digest = _request_digest(
                {
                    "scope_id": scope_id,
                    "producer_id": self._producer_id,
                    "view_kind": vkind,
                    "query_terms": (
                        sorted(query_terms) if query_terms else None
                    ),
                    "sections": list(section_order),
                    "explicit_inputs": self._inputs_spec(explicit),
                }
            )
            view_id = self._view_id(scope_id, vkind, request_digest)
            doc = {
                "doc": _DOC_FORMAT,
                "view_id": view_id,
                "view_kind": vkind,
                "synthesis_mode": SYNTHESIS_MODE,
                "producer_id": self._producer_id,
                "rubric": RUBRIC_REVISION,
                "scope_id": scope_id,
                "request_digest": request_digest,
                "sections": list(section_order),
                "contributing_scopes": list(contrib_scopes),
                "epoch_vector": epoch_vector,
                "allowed_purposes": allowed.to_json(),
                "effective_audience": list(eff_audience),
                "inputs": doc_inputs,
                "inputs_digests": inputs_digest_list,
                "statements": [s.to_json() for s in statements],
                "omitted": dict(omitted),
                "created_us": now,
            }

        if not persist:
            view = DerivedView(
                view_id=view_id,
                revision=1,
                scope_id=scope_id,
                view_kind=vkind,
                producer_id=self._producer_id,
                statements=tuple(statements),
                inputs_digests=tuple(inputs_digest_list),
                output_digest=_doc_digest(self._store.hmac, doc),
                epoch_vector=epoch_vector,
                allowed_purposes=allowed,
                effective_audience=tuple(eff_audience),
                contributing_scopes=contrib_scopes,
                omitted=dict(omitted),
                created_us=now,
                persisted=False,
                request_digest=request_digest,
            )
            return self._materialize_view(view, caller, purpose, grant_check, now)

        # --- persist: re-verify inside the write tx (§09 race row) -----
        with self._store.tx() as conn:
            existing = repos_v4.get(
                conn, "objects",
                {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
            )
            if existing is not None and (
                existing["disposition"] == LifecycleState.ERASED.value
            ):
                _deny()  # erased views stay absent — never resurrected
            # Freshness re-check inside the commit tx: a parent corrected
            # or withheld since the read phase aborts the compose — typed
            # error, nothing persists (V4-20.08).
            access_verb = (
                Verb.QUOTE
                if (producer_grant is not None or read_verb is Verb.QUOTE)
                else Verb.READ
            )
            identity = caller
            if producer_grant is not None:
                grant_row = self._producer_grant_row(conn, producer_grant)
                if grant_row is None:
                    _deny()
                identity = CallerV3(principal_id=grant_row["principal_id"])
            for inp in doc_inputs:
                kind, oid, rev = inp["kind"], inp["id"], int(inp["rev"])
                iscope = inp.get("scope") or self._resolve_scope(
                    conn, oid, kind
                )
                if iscope is None:
                    _deny()
                lease = self._kernel.resolve_access(
                    conn, identity, access_verb.value, purpose, (iscope,),
                    object_refs=[(kind, oid, rev)], now_us=now,
                )
                if lease.denied:
                    _deny()
                current = self._current_revision(conn, kind, oid)
                if current is None or current != rev:
                    raise VerbatimError(
                        ErrorCode.STALE_DEPENDENCY,
                        "input revision moved during composition — "
                        "view rejected",
                        retryable=True,
                    )
            if existing is not None:
                prior = self._latest_revision_doc(conn, view_id)
                if (
                    prior is not None
                    and existing["disposition"] == LifecycleState.ACTIVE.value
                    and prior[0].get("inputs") == doc["inputs"]
                    and prior[0].get("request_digest") == request_digest
                ):
                    # Identical request over identical pinned inputs —
                    # idempotent: re-deliver the existing revision.
                    revision = int(existing["current_revision"])
                    view = self._doc_to_view(prior[0], revision, digest=prior[1])
                    return self._materialize_view(
                        view, caller, purpose, grant_check, now, conn=conn
                    )
                revision = int(existing["current_revision"]) + 1
                repos_v4.update(
                    conn, "objects",
                    {
                        "current_revision": revision,
                        "disposition": LifecycleState.ACTIVE.value,
                    },
                    {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
                )
            else:
                revision = 1
                repos_v4.insert(
                    conn, "objects",
                    {
                        "object_id": view_id,
                        "kind": VIEW_OBJECT_KIND,
                        "scope_id": scope_id,
                        "current_revision": revision,
                        "disposition": LifecycleState.ACTIVE.value,
                        "created_event": self._store.next_event_us(),
                    },
                )
            doc["revision"] = revision
            digest = _doc_digest(self._store.hmac, doc)
            repos_v4.insert(
                conn, "object_revisions",
                {
                    "kind": VIEW_OBJECT_KIND,
                    "object_id": view_id,
                    "revision": revision,
                    "digest": digest,
                    "recorded_from": self._store.next_event_us(),
                    "producer_ref": self._producer_id,
                    "metadata_json": doc,
                },
            )
            op_id = f"op:{new_id()}"
            for stmt in statements:
                for s in stmt.support:
                    repos_v4.insert(
                        conn, "view_support",
                        {
                            "view_kind": VIEW_SUPPORT_KIND,
                            "view_id": view_id,
                            "view_revision": revision,
                            "proposition_locator": stmt.statement_id,
                            "evidence_kind": s.evidence_kind,
                            "evidence_id": s.evidence_id,
                            "evidence_revision": s.evidence_revision,
                            "verdict": s.verdict,
                        },
                    )
            for seq, (kind, oid, rev) in enumerate(sorted(bound_refs)):
                repos_v4.insert(
                    conn, "dependency_edges",
                    {
                        "child_kind": VIEW_OBJECT_KIND,
                        "child_id": view_id,
                        "child_revision": revision,
                        "parent_kind": kind,
                        "parent_id": oid,
                        "parent_revision": rev,
                        "role": "derived",
                        "producer_id": self._producer_id,
                        "operation_id": op_id,
                        "seq": seq,
                    },
                )

        view = DerivedView(
            view_id=view_id,
            revision=revision,
            scope_id=scope_id,
            view_kind=vkind,
            producer_id=self._producer_id,
            statements=tuple(statements),
            inputs_digests=tuple(inputs_digest_list),
            output_digest=digest,
            epoch_vector=epoch_vector,
            allowed_purposes=allowed,
            effective_audience=tuple(eff_audience),
            contributing_scopes=contrib_scopes,
            omitted=dict(omitted),
            created_us=now,
            persisted=True,
            request_digest=request_digest,
        )
        return self._materialize_view(view, caller, purpose, grant_check, now)

    # ------------------------------------------------------------------
    # deliver / status / erase (V4-20.07)
    # ------------------------------------------------------------------

    def deliver(
        self,
        view_id: str,
        *,
        caller: CallerV3,
        purpose: Optional[str] = "recall",
        revision: Optional[int] = None,
        grant_check: Optional[Callable[..., bool]] = None,
        now_us: Optional[int] = None,
    ) -> DerivedView:
        """Deliver a persisted view revision to an authorized caller.

        Re-authorizes the caller on every contributing scope, re-checks
        every bound evidence ref at its pinned revision (withheld, moved,
        or corrected parents → the view is durably suppressed and
        ``STALE_DEPENDENCY`` raises), verifies the persisted doc digest,
        then materializes quote text under the caller's ``quote`` lease —
        quote-denied callers receive metadata + support refs only.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        require_id(view_id, "view_id")
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        with self._store.read() as conn:
            outcome = self._load_for_delivery(
                conn, view_id, caller, purpose, grant_check, revision, now
            )
        if outcome is None:  # stale — suppress durably, then report
            with self._store.tx() as conn:
                obj = repos_v4.get(
                    conn, "objects",
                    {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
                )
                if obj is not None and obj[
                    "disposition"
                ] == LifecycleState.ACTIVE.value:
                    repos_v4.update(
                        conn, "objects",
                        {"disposition": LifecycleState.HELD.value},
                        {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
                    )
            raise VerbatimError(
                ErrorCode.STALE_DEPENDENCY,
                "derived view is stale — a bound parent moved or was "
                "withheld; recompose to rebuild",
                retryable=True,
            )
        view, _doc = outcome
        return self._materialize_view(view, caller, purpose, grant_check, now)

    def status(
        self,
        view_id: str,
        *,
        caller: CallerV3,
        purpose: Optional[str] = "recall",
        now_us: Optional[int] = None,
    ) -> dict[str, Any]:
        """Metadata-only view inspection for an authorized caller.

        Never returns statement text or canonical bytes — the lifecycle
        record (disposition, revision, counts, digests) only.
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        require_id(view_id, "view_id")
        with self._store.read() as conn:
            obj = repos_v4.get(
                conn, "objects",
                {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
            )
            if obj is None:
                _deny()
            self._require_verb(
                conn, caller, (obj["scope_id"],), Verb.READ, purpose,
                None, now,
            )
            rev = self._latest_revision_doc(conn, view_id)
            doc = rev[0] if rev else {}
            return {
                "view_id": view_id,
                "scope_id": obj["scope_id"],
                "disposition": obj["disposition"],
                "current_revision": int(obj["current_revision"]),
                "statement_count": len(doc.get("statements") or ()),
                "omitted": dict(doc.get("omitted") or {}),
                "producer_id": doc.get("producer_id"),
                "synthesis_mode": doc.get("synthesis_mode"),
                "output_digest": rev[1] if rev else None,
            }

    def erase(
        self,
        view_id: str,
        *,
        caller: CallerV3,
        purpose: Optional[str] = "admin",
        now_us: Optional[int] = None,
    ) -> None:
        """Erase a derived view (lifecycle: revisioned → erased).

        Requires the caller's ``admin`` verb on the view's scope. The
        ``objects`` disposition flips to ``erased`` (delivery denies
        indistinguishably) and the revision doc is scrubbed to a
        tombstone; ``view_support``/``dependency_edges`` rows stay as
        erasure provenance for deletion closure (V4-38).
        """
        now = int(now_us) if now_us is not None else _time.now_us()
        require_id(view_id, "view_id")
        with self._store.tx() as conn:
            obj = repos_v4.get(
                conn, "objects",
                {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
            )
            if obj is None:
                _deny()
            self._require_verb(
                conn, caller, (obj["scope_id"],), Verb.ADMIN, purpose,
                None, now,
            )
            repos_v4.update(
                conn, "objects",
                {"disposition": LifecycleState.ERASED.value},
                {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
            )
            rev = self._latest_revision_doc(conn, view_id)
            if rev is not None:
                doc, _digest = rev
                tombstone = {
                    "doc": _DOC_FORMAT,
                    "view_id": view_id,
                    "revision": doc.get("revision"),
                    "view_kind": doc.get("view_kind"),
                    "synthesis_mode": SYNTHESIS_MODE,
                    "producer_id": doc.get("producer_id"),
                    "scope_id": doc.get("scope_id"),
                    "erased": True,
                    "erased_us": now,
                    "statements": [],
                    "inputs": [],
                    "inputs_digests": [],
                }
                repos_v4.update(
                    conn, "object_revisions",
                    {
                        "metadata_json": tombstone,
                        "digest": _doc_digest(self._store.hmac, tombstone),
                    },
                    {
                        "kind": VIEW_OBJECT_KIND,
                        "object_id": view_id,
                        "revision": doc["revision"],
                    },
                )

    def note_invalidation(
        self, conn: sqlite3.Connection, report: Any
    ) -> int:
        """Eagerly suppress views named by a ``kernel.invalidate`` report.

        Optional hook — call inside the same transaction as
        ``kernel.invalidate`` so dependent views flip to ``held``
        atomically with the epoch bump. Delivery suppresses stale views
        lazily even without this call; the hook makes the suppression
        immediately durable. Returns the count marked.
        """
        affected = getattr(report, "affected", None)
        if affected is None and isinstance(report, Mapping):
            affected = report.get("affected")
        marked = 0
        for item in affected or ():
            try:
                kind, oid, _rev = item
            except (TypeError, ValueError):
                continue
            if kind != VIEW_OBJECT_KIND:
                continue
            obj = repos_v4.get(
                conn, "objects",
                {"kind": VIEW_OBJECT_KIND, "object_id": oid},
            )
            if obj is not None and obj[
                "disposition"
            ] == LifecycleState.ACTIVE.value:
                repos_v4.update(
                    conn, "objects",
                    {"disposition": LifecycleState.HELD.value},
                    {"kind": VIEW_OBJECT_KIND, "object_id": oid},
                )
                marked += 1
        return marked

    # ------------------------------------------------------------------
    # candidate enumeration (metadata-only SQL — never payload columns)
    # ------------------------------------------------------------------

    def _enumerate(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        sections: Sequence[str],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if "evidence" in sections:
            for r in conn.execute(
                "SELECT s.span_id, s.source_id, s.revision,"
                " s.start_byte, s.end_byte"
                " FROM spans s JOIN sources src"
                "   ON src.source_id = s.source_id"
                " WHERE src.scope_id = ?"
                " ORDER BY s.span_id LIMIT ?",
                (scope_id, _MAX_CANDIDATES),
            ).fetchall():
                out.append(
                    {
                        "kind": "span",
                        "id": r[0],
                        "rev": int(r[2]),
                        "start": int(r[3]),
                        "end": int(r[4]),
                        "source_id": r[1],
                    }
                )
        if "sources" in sections:
            for r in conn.execute(
                "SELECT s.source_id,"
                " (SELECT MAX(r.revision) FROM source_revisions r"
                "   WHERE r.source_id = s.source_id)"
                " FROM sources s WHERE s.scope_id = ?"
                " ORDER BY s.source_id LIMIT ?",
                (scope_id, _MAX_CANDIDATES),
            ).fetchall():
                if r[1] is None:
                    continue
                out.append(
                    {
                        "kind": "source",
                        "id": r[0],
                        "rev": int(r[1]),
                        "start": None,
                        "end": None,
                        "source_id": r[0],
                    }
                )
        if "claims" in sections:
            for r in conn.execute(
                "SELECT c.claim_id, cr.revision, cr.state"
                " FROM claims c JOIN claim_revisions cr"
                "   ON cr.claim_id = c.claim_id"
                " WHERE c.scope_id = ?"
                "   AND cr.state IN ('active','disputed')"
                "   AND cr.revision = (SELECT MAX(revision)"
                "       FROM claim_revisions WHERE claim_id = c.claim_id)"
                " ORDER BY c.claim_id LIMIT ?",
                (scope_id, _MAX_CANDIDATES),
            ).fetchall():
                out.append(
                    {
                        "kind": "claim",
                        "id": r[0],
                        "rev": int(r[1]),
                        "start": None,
                        "end": None,
                        "state": r[2],
                    }
                )
        return out

    def _normalize_inputs(
        self, inputs: Optional[Iterable[Any]]
    ) -> Optional[list[dict[str, Any]]]:
        """Explicit input refs → candidate dicts.

        Accepts ``EvidenceLocator``, ``(kind, id, rev)`` triples, or dicts
        (``kind``/``object_kind``, ``id``/``object_id``, ``revision``,
        optional ``start_byte``/``end_byte``/``view_id``). Bare ``(id,
        rev)`` pairs are rejected — synthesis requires the declared kind
        so support bindings stay unambiguous (V4-20.03).
        """
        if inputs is None:
            return None
        out: list[dict[str, Any]] = []
        for ref in inputs:
            kind = oid = rev = start = end = view_id = None
            if isinstance(ref, EvidenceLocator):
                oid, rev = ref.object_id, ref.revision
                start, end, view_id = (
                    ref.start_byte, ref.end_byte, ref.view_id
                )
            elif isinstance(ref, Mapping):
                kind = ref.get("kind") or ref.get("object_kind")
                oid = ref.get("id") or ref.get("object_id")
                rev = ref.get("revision")
                start, end = ref.get("start_byte"), ref.get("end_byte")
                view_id = ref.get("view_id")
            else:
                try:
                    parts = tuple(ref)
                except TypeError:
                    parts = ()
                if len(parts) == 3:
                    kind, oid, rev = parts
                elif len(parts) == 5:
                    kind, oid, rev, start, end = parts
            if oid is None or rev is None:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "input refs need kind + object id + revision",
                )
            if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "input revision must be an int >= 0"
                )
            out.append(
                {
                    "kind": (
                        None if kind is None else require_id(str(kind), "kind")
                    ),
                    "id": require_id(str(oid), "id"),
                    "rev": int(rev),
                    "start": start,
                    "end": end,
                    "view_id": view_id,
                }
            )
        if len(out) > _MAX_CANDIDATES:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"inputs exceed the {_MAX_CANDIDATES} candidate bound",
            )
        return out

    # ------------------------------------------------------------------
    # verified candidate reads (kernel path only)
    # ------------------------------------------------------------------

    def _read_candidate(
        self,
        conn: sqlite3.Connection,
        cand: dict[str, Any],
        *,
        producer_grant: Any,
        reader: Optional[CallerV3],
        read_verb: Verb,
        scope_id: str,
        purpose: Optional[str],
        audience: tuple[str, ...],
        now: int,
    ) -> Optional[dict[str, Any]]:
        """One candidate → a slot {primary slice, support refs, scope}.

        Returns None when the candidate is denied/withheld — its
        statements are then omitted (V4-20.05). Integrity failures
        (``STORE_CORRUPT``) propagate: they are never swallowed into
        "unsupported".
        """
        kind = cand.get("kind")
        oid, rev = cand["id"], cand["rev"]
        if kind is None:
            kind = self._resolve_kind(conn, oid)
            if kind is None:
                return None
            cand = dict(cand, kind=kind)
        if kind == "claim":
            return self._read_claim(
                conn, cand, producer_grant=producer_grant, reader=reader,
                read_verb=read_verb, scope_id=scope_id,
                purpose=purpose, audience=audience, now=now,
            )
        scope = self._resolve_scope(conn, oid, kind)
        if scope is None:
            return None
        if producer_grant is None and scope != scope_id:
            # Caller path is single-scope; cross-scope composition goes
            # through a producer grant (V4-10.07).
            return None
        if kind not in _BYTE_KINDS:
            # Registered no-byte kinds: object-level attested support only.
            if not self._object_ok(
                conn, reader, producer_grant, scope, kind, oid, rev,
                read_verb, purpose, now,
            ):
                return None
            return {
                "kind": kind, "id": oid, "rev": rev, "scope_id": scope,
                "slice": None, "locator": None,
                "supports": [
                    SupportBinding(
                        kind, oid, rev, SupportVerdict.ATTESTED.value
                    )
                ],
                "primary_verified": False,
                "byte_length": None,
            }
        start, end = cand.get("start"), cand.get("end")
        if kind == "span":
            rng = self._span_range(conn, oid)
            if rng is None:
                return None
            start, end, span_source = rng  # whole-excerpt (kernel rule)
            # A span pins the source revision it covers; once the source
            # is corrected past that revision, composing over it would
            # mint an already-stale view — the candidate is omitted, not
            # shipped (V4-20.05/07).
            if self._current_revision(conn, "source", span_source) != rev:
                return None
            cand = dict(cand, source_id=span_source)
        elif kind == "source":
            if self._current_revision(conn, "source", oid) != rev:
                return None
        if start is None or end is None:
            rng = self._full_range(conn, oid, rev)
            if rng is None:
                return None
            start, end = rng
        if int(end) - int(start) > _MAX_QUOTE_BYTES:
            # Too large to quote honestly — degrade to attested field
            # support rather than truncating mid-evidence.
            if not self._object_ok(
                conn, reader, producer_grant, scope, kind, oid, rev,
                read_verb, purpose, now,
            ):
                return None
            return {
                "kind": kind, "id": oid, "rev": rev, "scope_id": scope,
                "slice": None, "locator": None,
                "supports": [
                    SupportBinding(
                        kind, oid, rev, SupportVerdict.ATTESTED.value
                    )
                ],
                "primary_verified": False,
                "byte_length": int(end) - int(start),
            }
        locator = EvidenceLocator(
            object_id=oid, revision=rev, start_byte=int(start),
            end_byte=int(end), view_id=cand.get("view_id"),
        )
        slice_ = self._read_slice(
            conn, kind, locator, producer_grant=producer_grant,
            reader=reader, read_verb=read_verb, scope=scope,
            purpose=purpose, audience=audience, now=now,
        )
        if slice_ is None:
            return None
        verdict = (
            SupportVerdict.VERIFIED.value
            if slice_.verification == "verified"
            else slice_.verification
        )
        supports = [
            SupportBinding(
                kind, oid, rev, verdict=verdict,
                start_byte=int(start), end_byte=int(end),
                view_id=cand.get("view_id"),
            )
        ]
        # Ancestor binding: a span's covering source pins the exact
        # revision so a later correction stales the statement (C49).
        if kind == "span" and cand.get("source_id"):
            supports.append(
                SupportBinding(
                    "source", cand["source_id"], rev,
                    SupportVerdict.ATTESTED.value,
                )
            )
        return {
            "kind": kind, "id": oid, "rev": rev, "scope_id": scope,
            "slice": slice_, "locator": locator, "supports": supports,
            "primary_verified": slice_.verification == "verified",
            "byte_length": int(end) - int(start),
            "provenance": dict(slice_.provenance),
            "source_id": cand.get("source_id"),
        }

    def _read_claim(
        self,
        conn: sqlite3.Connection,
        cand: dict[str, Any],
        *,
        producer_grant: Any,
        reader: Optional[CallerV3],
        read_verb: Verb,
        scope_id: str,
        purpose: Optional[str],
        audience: tuple[str, ...],
        now: int,
    ) -> Optional[dict[str, Any]]:
        """Claims have no byte surface — they contribute field statements
        whose support binds the claim plus its evidence spans."""
        scope = self._resolve_scope(conn, cand["id"], "claim")
        if scope is None or (producer_grant is None and scope != scope_id):
            return None
        if not self._object_ok(
            conn, reader, producer_grant, scope, "claim",
            cand["id"], cand["rev"], read_verb, purpose, now,
        ):
            return None
        # A claim corrected past the pinned revision is stale for
        # composition — minting it would produce an already-stale view.
        if self._current_revision(conn, "claim", cand["id"]) != cand["rev"]:
            return None
        supports = [
            SupportBinding(
                "claim", cand["id"], cand["rev"],
                SupportVerdict.ATTESTED.value,
            )
        ]
        span_slots: list[dict[str, Any]] = []
        for (span_id,) in conn.execute(
            "SELECT span_id FROM claim_evidence"
            " WHERE claim_id = ? AND revision = ? LIMIT ?",
            (cand["id"], cand["rev"], _MAX_SPANS_PER_CLAIM),
        ).fetchall():
            sub = self._read_candidate(
                conn,
                {"kind": "span", "id": span_id, "rev": cand["rev"],
                 "start": None, "end": None},
                producer_grant=producer_grant, reader=reader,
                read_verb=read_verb, scope_id=scope_id,
                purpose=purpose, audience=audience, now=now,
            )
            if sub is None:
                supports.append(
                    SupportBinding("span", span_id, cand["rev"], "withheld")
                )
                continue
            span_slots.append(sub)
            supports.extend(sub["supports"])
        deduped: dict[tuple, SupportBinding] = {}
        for s in supports:
            deduped.setdefault(
                (s.evidence_kind, s.evidence_id, s.evidence_revision,
                 s.verdict), s
            )
        return {
            "kind": "claim", "id": cand["id"], "rev": cand["rev"],
            "scope_id": scope, "slice": None, "locator": None,
            "supports": list(deduped.values()), "span_slots": span_slots,
            "primary_verified": True,  # claim attestation itself verified
            "byte_length": None,
        }

    def _read_slice(
        self,
        conn: sqlite3.Connection,
        kind: str,
        locator: EvidenceLocator,
        *,
        producer_grant: Any,
        reader: Optional[CallerV3],
        read_verb: Verb,
        scope: str,
        purpose: Optional[str],
        audience: tuple[str, ...],
        now: int,
    ) -> Optional[Any]:
        """Verified bytes for one locator — producer derive_inputs or the
        caller's lease. Denied/withheld → None; integrity errors raise."""
        if producer_grant is not None:
            try:
                bundle = self._kernel.derive_inputs(
                    conn,
                    producer_grant,
                    [
                        {
                            "kind": kind,
                            "id": locator.object_id,
                            "revision": locator.revision,
                            "start_byte": locator.start_byte,
                            "end_byte": locator.end_byte,
                            "view_id": locator.view_id,
                        }
                    ],
                    output_audience=audience,
                    output_purpose=purpose,
                    now_us=now,
                )
            except VerbatimError as exc:
                if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                    return None
                raise
            return bundle.inputs[0] if bundle.inputs else None
        assert reader is not None
        lease = self._kernel.resolve_access(
            conn, reader, read_verb.value, purpose, (scope,), now_us=now,
        )
        if lease.denied:
            return None
        try:
            return self._kernel.read_verified(
                conn, lease, [locator], now_us=now
            )[0]
        except VerbatimError as exc:
            if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                return None
            raise

    def _object_ok(
        self,
        conn: sqlite3.Connection,
        reader: Optional[CallerV3],
        producer_grant: Any,
        scope: str,
        kind: str,
        oid: str,
        rev: int,
        verb: Verb,
        purpose: Optional[str],
        now: int,
    ) -> bool:
        """Object-level access check for non-byte refs — the same kernel
        gate (resolve_access evaluates lifecycle/quarantine/suppression
        per object) under the composition's effective verb."""
        identity = reader
        if producer_grant is not None:
            grant_row = self._producer_grant_row(conn, producer_grant)
            if grant_row is None:
                return False
            identity = CallerV3(principal_id=grant_row["principal_id"])
        if identity is None:
            return False
        lease = self._kernel.resolve_access(
            conn, identity, verb.value, purpose, (scope,),
            object_refs=[(kind, oid, rev)], now_us=now,
        )
        return not lease.denied

    # ------------------------------------------------------------------
    # selection + statement build
    # ------------------------------------------------------------------

    def _select(
        self,
        slots: list[dict[str, Any]],
        query_terms: Optional[frozenset],
        vkind: str,
    ) -> list[dict[str, Any]]:
        """Deterministic selection: term coverage over the statement's
        own verified/rendered text; ties break on object identity."""
        scored: list[tuple] = []
        for i, slot in enumerate(slots):
            text = self._slot_text(slot, vkind)
            if query_terms is not None:
                if text is None:
                    continue
                coverage = len(query_terms & _terms(text)) / len(query_terms)
                if coverage <= 0:
                    continue
            else:
                coverage = 1.0
            start = slot["locator"].start_byte if slot["locator"] else -1
            scored.append((-coverage, slot["id"], start, i, slot))
        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [t[4] for t in scored]

    def _slot_text(self, slot: dict[str, Any], vkind: str) -> Optional[str]:
        """The text a statement would carry — decoded slice bytes for
        quotes, the deterministic field rendering otherwise."""
        if (
            vkind == ViewKind.EXTRACTIVE_DIGEST.value
            and slot["slice"] is not None
            and slot["slice"].data
        ):
            try:
                return slot["slice"].data.decode("utf-8")
            except UnicodeDecodeError:
                return None
        return self._field_text(slot)

    def _field_text(self, slot: dict[str, Any]) -> str:
        """Deterministic metadata rendering — never canonical bytes."""
        n = slot.get("byte_length")
        size = f"{n}B" if n is not None else "size unknown"
        if slot["kind"] == "claim":
            return (
                f"claim {slot['id']}@r{slot['rev']}: "
                f"{len(slot.get('span_slots') or ())} evidence span(s)"
            )
        if slot["kind"] == "span":
            return (
                f"span {slot['id']} on {slot.get('source_id') or '?'}@"
                f"r{slot['rev']}: {size}"
            )
        return f"{slot['kind']} {slot['id']}@r{slot['rev']}: {size}"

    def _build_statement(
        self, slot: dict[str, Any], vkind: str
    ) -> Optional[ViewStatement]:
        # One view_support row per (statement, ref) — dedupe so two paths
        # binding the same parent never collide on the support PK.
        deduped: dict[tuple, SupportBinding] = {}
        for s in slot["supports"]:
            deduped.setdefault(
                (s.evidence_kind, s.evidence_id, s.evidence_revision,
                 s.verdict), s,
            )
        supports = tuple(deduped.values())
        if not supports:
            return None  # unsupported → omitted (V4-20.05)
        withheld = any(s.verdict == "withheld" for s in supports)
        if (
            vkind == ViewKind.EXTRACTIVE_DIGEST.value
            and slot["kind"] != "claim"
            and slot["slice"] is not None
            and slot["slice"].data
        ):
            try:
                text = slot["slice"].data.decode("utf-8")
            except UnicodeDecodeError:
                return None  # binary evidence cannot mint a text quote
            form = StatementForm.QUOTE.value
            loc = {
                "object_id": slot["locator"].object_id,
                "revision": slot["locator"].revision,
                "start_byte": slot["locator"].start_byte,
                "end_byte": slot["locator"].end_byte,
                "view_id": slot["locator"].view_id,
            }
            edigest = slot["slice"].digest or None
        else:
            form, text, loc, edigest = (
                StatementForm.FIELD.value,
                self._field_text(slot),
                None,
                None,
            )
        if form == StatementForm.QUOTE.value:
            confidence = (
                Confidence.SUPPORTED.value
                if slot["primary_verified"] and not withheld
                else Confidence.PARTIAL.value
            )
        else:
            confidence = (
                Confidence.PARTIAL.value
                if withheld or all(
                    s.verdict == "withheld" for s in supports
                )
                else Confidence.SUPPORTED.value
            )
        section = {
            "claim": "claims",
            "source": "sources",
            "span": "evidence",
        }.get(slot["kind"], "evidence")
        sid = "st:" + hashlib.sha256(
            json_dumps(
                {
                    "form": form,
                    "supports": [s.to_json() for s in supports],
                    "text": text,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        return ViewStatement(
            statement_id=sid,
            form=form,
            confidence=confidence,
            support=supports,
            text=text,
            section=section,
            locator=loc,
            evidence_digest=edigest,
        )

    @staticmethod
    def _sequence(
        statements: list[ViewStatement], section_order: Sequence[str]
    ) -> list[ViewStatement]:
        order = {s: i for i, s in enumerate(section_order)}
        out = sorted(
            statements,
            key=lambda s: (order.get(s.section, 99), s.statement_id),
        )
        return [
            ViewStatement(
                statement_id=s.statement_id, form=s.form,
                confidence=s.confidence, support=s.support, text=s.text,
                section=s.section, seq=i, locator=s.locator,
                evidence_digest=s.evidence_digest, screen=s.screen,
            )
            for i, s in enumerate(out)
        ]

    # ------------------------------------------------------------------
    # persisted doc <-> DerivedView
    # ------------------------------------------------------------------

    @staticmethod
    def _bound_refs(
        statements: Sequence[ViewStatement],
    ) -> set[tuple[str, str, int]]:
        out: set[tuple[str, str, int]] = set()
        for st in statements:
            for s in st.support:
                if s.verdict == "withheld":
                    continue
                out.add((s.evidence_kind, s.evidence_id, s.evidence_revision))
        return out

    def _doc_inputs(
        self,
        conn: sqlite3.Connection,
        statements: Sequence[ViewStatement],
        slots: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """The pinned input vector the view was minted against — one row
        per bound ref actually grounding a shipped statement."""
        digest_by_ref: dict[tuple, Optional[str]] = {}
        scope_by_ref: dict[tuple, str] = {}
        for slot in slots:
            key = (slot["kind"], slot["id"], slot["rev"])
            scope_by_ref[key] = slot["scope_id"]
            if slot.get("slice") is not None and slot["slice"].digest:
                digest_by_ref[key] = slot["slice"].digest
            for sub in slot.get("span_slots") or ():
                skey = (sub["kind"], sub["id"], sub["rev"])
                scope_by_ref[skey] = sub["scope_id"]
                if sub.get("slice") is not None and sub["slice"].digest:
                    digest_by_ref[skey] = sub["slice"].digest
        out: dict[tuple, dict[str, Any]] = {}
        for st in statements:
            for s in st.support:
                if s.verdict == "withheld":
                    continue
                key = (s.evidence_kind, s.evidence_id, s.evidence_revision)
                if key in out:
                    continue
                scope = scope_by_ref.get(key)
                if scope is None:
                    scope = self._resolve_scope(
                        conn, s.evidence_id, s.evidence_kind
                    )
                out[key] = {
                    "kind": s.evidence_kind,
                    "id": s.evidence_id,
                    "rev": s.evidence_revision,
                    "scope": scope,
                    "digest": digest_by_ref.get(key),
                }
        return [out[k] for k in sorted(out)]

    @staticmethod
    def _inputs_spec(
        explicit: Optional[list[dict[str, Any]]],
    ) -> Optional[list[dict[str, Any]]]:
        if explicit is None:
            return None
        return [
            {
                "kind": c.get("kind"), "id": c["id"], "rev": c["rev"],
                "start": c.get("start"), "end": c.get("end"),
            }
            for c in explicit
        ]

    @staticmethod
    def _view_id(scope_id: str, vkind: str, request_digest: str) -> str:
        """Deterministic view identity: the same declared request over the
        same scope names the same logical view — recomputation mints the
        next revision instead of a parallel object."""
        h = hashlib.sha256(
            json_dumps(
                {
                    "scope": scope_id,
                    "view_kind": vkind,
                    "request": request_digest,
                }
            ).encode("utf-8")
        ).hexdigest()[:40]
        return f"dview:{h}"

    def _doc_to_view(
        self, doc: dict[str, Any], revision: int, *, digest: str
    ) -> DerivedView:
        return DerivedView(
            view_id=doc["view_id"],
            revision=int(doc.get("revision") or revision),
            scope_id=doc["scope_id"],
            view_kind=doc["view_kind"],
            producer_id=doc["producer_id"],
            statements=tuple(
                ViewStatement.from_json(s) for s in doc.get("statements") or ()
            ),
            inputs_digests=tuple(doc.get("inputs_digests") or ()),
            output_digest=digest,
            epoch_vector={
                k: int(v) for k, v in (doc.get("epoch_vector") or {}).items()
            },
            allowed_purposes=PurposeConstraint.from_json(
                doc.get("allowed_purposes")
            ),
            effective_audience=tuple(doc.get("effective_audience") or ()),
            contributing_scopes=tuple(doc.get("contributing_scopes") or ()),
            omitted=dict(doc.get("omitted") or {}),
            created_us=int(doc.get("created_us") or 0),
            persisted=True,
            request_digest=doc.get("request_digest") or "",
        )

    def _latest_revision_doc(
        self, conn: sqlite3.Connection, view_id: str
    ) -> Optional[tuple[dict[str, Any], str]]:
        row = repos_v4.get(
            conn, "objects",
            {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
        )
        if row is None:
            return None
        rev = repos_v4.get(
            conn, "object_revisions",
            {
                "kind": VIEW_OBJECT_KIND,
                "object_id": view_id,
                "revision": int(row["current_revision"]),
            },
        )
        if rev is None:
            return None
        raw = rev.get("metadata_json") or "{}"
        doc = safe_json_loads(raw) if isinstance(raw, str) else dict(raw)
        stored = rev.get("digest") or ""
        if stored and stored != _doc_digest(self._store.hmac, doc):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"derived view {view_id} revision doc fails integrity check",
            )
        return doc, stored

    # ------------------------------------------------------------------
    # delivery-time gating
    # ------------------------------------------------------------------

    def _load_for_delivery(
        self,
        conn: sqlite3.Connection,
        view_id: str,
        caller: CallerV3,
        purpose: Optional[str],
        grant_check: Optional[Callable[..., bool]],
        revision: Optional[int],
        now: int,
    ) -> Optional[tuple[DerivedView, dict[str, Any]]]:
        """Load + authorize + freshness-check a persisted view.

        Returns ``None`` when the view is stale (caller suppresses it
        durably); raises the typed denials for absent/erased/unauthorized.
        """
        obj = repos_v4.get(
            conn, "objects",
            {"kind": VIEW_OBJECT_KIND, "object_id": view_id},
        )
        if obj is None or obj["disposition"] == LifecycleState.ERASED.value:
            _deny()
        rev = (
            int(revision) if revision is not None
            else int(obj["current_revision"])
        )
        row = repos_v4.get(
            conn, "object_revisions",
            {
                "kind": VIEW_OBJECT_KIND,
                "object_id": view_id,
                "revision": rev,
            },
        )
        if row is None:
            _deny()
        raw = row.get("metadata_json") or "{}"
        doc = safe_json_loads(raw) if isinstance(raw, str) else dict(raw)
        stored = row.get("digest") or ""
        if stored and stored != _doc_digest(self._store.hmac, doc):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"derived view {view_id} doc fails integrity check",
            )
        view = self._doc_to_view(doc, rev, digest=stored)
        scopes = view.contributing_scopes or (view.scope_id,)
        # Caller authorization on EVERY contributing scope (C48): a
        # recipient denied any required parent cannot take delivery.
        self._require_verb(
            conn, caller, scopes, Verb.READ, purpose, grant_check, now
        )
        # Inherited purpose restriction (V4-20.06): a SET/NONE constraint
        # narrows which purposes may take delivery.
        if not view.allowed_purposes.permits(purpose):
            _deny()
        if obj["disposition"] != LifecycleState.ACTIVE.value:
            return None
        if not self._fresh(conn, view, caller, purpose, now):
            return None
        return view, doc

    def _fresh(
        self,
        conn: sqlite3.Connection,
        view: DerivedView,
        caller: CallerV3,
        purpose: Optional[str],
        now: int,
    ) -> bool:
        """Every bound ref must still resolve at its pinned revision.

        A held/purged/absent parent (kernel denial at the ref's own
        scope) or a parent whose current revision advanced (correction)
        both stale the view — V4-20.07 suppresses pending recomputation.
        Byte-surface refs additionally probe ``length(payload) > 0`` —
        the metadata-level mirror of the kernel's emptied-payload gate.
        """
        for st in view.statements:
            for s in st.support:
                if s.verdict == "withheld":
                    continue
                scope = self._resolve_scope(
                    conn, s.evidence_id, s.evidence_kind
                )
                if scope is None:
                    return False
                lease = self._kernel.resolve_access(
                    conn, caller, Verb.READ.value, purpose, (scope,),
                    object_refs=[
                        (s.evidence_kind, s.evidence_id, s.evidence_revision)
                    ],
                    now_us=now,
                )
                if lease.denied:
                    return False
                current = self._current_revision(
                    conn, s.evidence_kind, s.evidence_id
                )
                if current is None or current != s.evidence_revision:
                    return False
                if (
                    s.evidence_kind in _BYTE_KINDS
                    and not self._bytes_present(
                        conn, s.evidence_kind, s.evidence_id,
                        s.evidence_revision,
                    )
                ):
                    return False
        return True

    def _materialize_view(
        self,
        view: DerivedView,
        caller: CallerV3,
        purpose: Optional[str],
        grant_check: Optional[Callable[..., bool]],
        now: int,
        conn: Optional[sqlite3.Connection] = None,
    ) -> DerivedView:
        """Materialize quote text for this caller — ``quote`` required for
        canonical bytes (V4-08.02); everyone else gets metadata + support
        refs only."""
        if conn is not None:
            return self._materialize(
                conn, view, caller, purpose, grant_check, now
            )
        with self._store.read() as c:
            return self._materialize(c, view, caller, purpose, grant_check, now)

    def _materialize(
        self,
        conn: sqlite3.Connection,
        view: DerivedView,
        caller: CallerV3,
        purpose: Optional[str],
        grant_check: Optional[Callable[..., bool]],
        now: int,
    ) -> DerivedView:
        scopes = view.contributing_scopes or (view.scope_id,)
        if not self._allowed(
            conn, caller, scopes, Verb.QUOTE, purpose, grant_check, now
        ):
            return view  # metadata + support refs only
        refs = [
            (st.locator["object_id"], int(st.locator["revision"]))
            for st in view.statements
            if st.form == StatementForm.QUOTE.value and st.locator
        ]
        if not refs:
            return view
        lease = self._kernel.resolve_access(
            conn, caller, Verb.QUOTE.value, purpose, scopes,
            object_refs=refs, now_us=now,
        )
        if lease.denied:
            return view
        out: list[ViewStatement] = []
        for st in view.statements:
            if st.form != StatementForm.QUOTE.value or not st.locator:
                out.append(st)
                continue
            loc = EvidenceLocator(
                object_id=st.locator["object_id"],
                revision=int(st.locator["revision"]),
                start_byte=int(st.locator["start_byte"]),
                end_byte=int(st.locator["end_byte"]),
                view_id=st.locator.get("view_id"),
            )
            try:
                slice_ = self._kernel.read_verified(
                    conn, lease, [loc], now_us=now
                )[0]
            except VerbatimError as exc:
                if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                    # Evidence died between freshness and materialization
                    # — withhold the statement's bytes, keep its record.
                    out.append(
                        ViewStatement(
                            statement_id=st.statement_id, form=st.form,
                            confidence=Confidence.PARTIAL.value,
                            support=st.support, text=None,
                            section=st.section, seq=st.seq,
                            locator=st.locator,
                            evidence_digest=st.evidence_digest,
                            screen=st.screen,
                        )
                    )
                    continue
                raise
            if st.evidence_digest and slice_.digest != st.evidence_digest:
                # The bytes under the pinned locator no longer match the
                # mint binding — tamper or in-place rewrite. Fail closed.
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT,
                    "bound evidence bytes diverged from mint digest",
                )
            try:
                text = slice_.data.decode("utf-8") if slice_.data else None
            except UnicodeDecodeError:
                text = None
            out.append(
                ViewStatement(
                    statement_id=st.statement_id, form=st.form,
                    confidence=st.confidence, support=st.support,
                    text=text, section=st.section, seq=st.seq,
                    locator=st.locator, evidence_digest=st.evidence_digest,
                    screen=st.screen,
                )
            )
        return DerivedView(
            view_id=view.view_id, revision=view.revision,
            scope_id=view.scope_id, view_kind=view.view_kind,
            producer_id=view.producer_id, statements=tuple(out),
            inputs_digests=view.inputs_digests,
            output_digest=view.output_digest,
            epoch_vector=view.epoch_vector,
            allowed_purposes=view.allowed_purposes,
            effective_audience=view.effective_audience,
            contributing_scopes=view.contributing_scopes,
            omitted=view.omitted, created_us=view.created_us,
            persisted=view.persisted, stale=view.stale,
            request_digest=view.request_digest,
        )

    # ------------------------------------------------------------------
    # authorization helpers (kernel-mediated; grant_check can only narrow)
    # ------------------------------------------------------------------

    @staticmethod
    def _producer_grant_row(
        conn: sqlite3.Connection, producer_grant: Any
    ) -> Optional[dict]:
        """Resolve a producer grant handle to its row — mirrors the
        shapes ``Kernel._producer_grant`` accepts (id | row | object with
        ``grant_id``) through the public grants repo."""
        if isinstance(producer_grant, str):
            return _grants.get_grant(conn, producer_grant)
        if isinstance(producer_grant, Mapping):
            return dict(producer_grant)
        if hasattr(producer_grant, "grant_id"):
            return _grants.get_grant(conn, producer_grant.grant_id)
        return None

    def _allowed(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        scope_ids: Sequence[str],
        verb: Verb,
        purpose: Optional[str],
        grant_check: Optional[Callable[..., bool]],
        now: int,
    ) -> bool:
        lease = self._kernel.resolve_access(
            conn, caller, verb.value, purpose, scope_ids, now_us=now
        )
        if lease.denied:
            return False
        return self._extra_check(
            conn, grant_check, caller, scope_ids, verb, purpose
        )

    def _require_verb(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        scope_ids: Sequence[str],
        verb: Verb,
        purpose: Optional[str],
        grant_check: Optional[Callable[..., bool]],
        now: int,
    ) -> EligibilityLease:
        lease = self._kernel.resolve_access(
            conn, caller, verb.value, purpose, scope_ids, now_us=now
        )
        if lease.denied:
            _deny()
        if not self._extra_check(
            conn, grant_check, caller, scope_ids, verb, purpose
        ):
            _deny()
        return lease

    @staticmethod
    def _extra_check(
        conn: sqlite3.Connection,
        grant_check: Optional[Callable[..., bool]],
        caller: CallerV3,
        scope_ids: Sequence[str],
        verb: Verb,
        purpose: Optional[str],
    ) -> bool:
        """Optional host-provided gate — can only restrict, never widen.

        Called once per scope as
        ``grant_check(conn, caller, scope_id, verb, purpose) -> bool``;
        a False return narrows the decision to denied.
        """
        if grant_check is None:
            return True
        for sid in scope_ids:
            if not grant_check(conn, caller, sid, verb.value, purpose):
                return False
        return True

    # ------------------------------------------------------------------
    # metadata-only resolution mirrors (no authorization decided here)
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_id(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise VerbatimError(
                ErrorCode.VALIDATION, "scope ids must be non-empty strings"
            )
        return require_id(value, "scope_id")

    @staticmethod
    def _view_kind(value: Any) -> str:
        try:
            v = value if isinstance(value, ViewKind) else ViewKind(str(value))
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown view kind {value!r}"
            ) from exc
        if v.value not in SUPPORTED_VIEW_KINDS:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"view kind {v.value!r} requires a producer class this "
                "environment does not have (grounded_composition only)",
            )
        return v.value

    @staticmethod
    def _sections(
        sections: Optional[Iterable[str]], vkind: str
    ) -> tuple[str, ...]:
        if sections is None:
            return (
                ("evidence",)
                if vkind == ViewKind.EXTRACTIVE_DIGEST.value
                else _SECTIONS
            )
        out = tuple(dict.fromkeys(str(s) for s in sections))
        bad = set(out) - set(_SECTIONS)
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown sections {sorted(bad)}"
            )
        if not out:
            raise VerbatimError(
                ErrorCode.VALIDATION, "sections must be non-empty"
            )
        return out

    def _resolve_kind(
        self, conn: sqlite3.Connection, object_id: str
    ) -> Optional[str]:
        """Object id → kind probe (metadata only; mirrors the tables
        ``kernel._resolve`` consults — no authorization decided here)."""
        r = repos_v4.get(conn, "objects", {"object_id": object_id})
        if r is not None:
            return r["kind"]
        for table, col, kind in (
            ("spans", "span_id", "span"),
            ("sources", "source_id", "source"),
            ("claims", "claim_id", "claim"),
            ("source_envelopes", "envelope_id", "source_envelope"),
        ):
            if conn.execute(
                f"SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1",
                (object_id,),
            ).fetchone() is not None:
                return kind
        return None

    def _resolve_scope(
        self, conn: sqlite3.Connection, object_id: str, kind: str
    ) -> Optional[str]:
        """Object ref → owning scope (metadata only)."""
        row = repos_v4.get(
            conn, "objects", {"object_id": object_id, "kind": kind}
        )
        if row is not None:
            return row["scope_id"]
        if kind == "source":
            r = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?",
                (object_id,),
            ).fetchone()
            return r[0] if r else None
        if kind == "span":
            r = conn.execute(
                "SELECT src.scope_id FROM spans s JOIN sources src"
                " ON src.source_id = s.source_id WHERE s.span_id = ?",
                (object_id,),
            ).fetchone()
            return r[0] if r else None
        if kind == "claim":
            r = conn.execute(
                "SELECT scope_id FROM claims WHERE claim_id = ?",
                (object_id,),
            ).fetchone()
            return r[0] if r else None
        if kind == "source_envelope":
            r = conn.execute(
                "SELECT scope_id FROM source_envelopes WHERE envelope_id = ?",
                (object_id,),
            ).fetchone()
            return r[0] if r else None
        return None

    @staticmethod
    def _span_range(
        conn: sqlite3.Connection, span_id: str
    ) -> Optional[tuple[int, int, str]]:
        row = conn.execute(
            "SELECT start_byte, end_byte, source_id FROM spans"
            " WHERE span_id = ?",
            (span_id,),
        ).fetchone()
        return (int(row[0]), int(row[1]), row[2]) if row else None

    @staticmethod
    def _full_range(
        conn: sqlite3.Connection, source_id: str, revision: int
    ) -> Optional[tuple[int, int]]:
        """Whole-extent byte range (mirrors ``kernel._full_range`` —
        ``length(payload)`` is a metadata probe, not a payload read)."""
        row = conn.execute(
            "SELECT length(payload) FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, int(revision)),
        ).fetchone()
        return (0, int(row[0])) if row and row[0] is not None else None

    def _current_revision(
        self, conn: sqlite3.Connection, kind: str, object_id: str
    ) -> Optional[int]:
        """Latest registered/persisted revision (mirrors
        ``kernel._current_revision`` — used for pinned-revision
        staleness, never for authorization)."""
        row = repos_v4.get(
            conn, "objects", {"object_id": object_id, "kind": kind}
        )
        if row is not None:
            return int(row["current_revision"])
        table, col = {
            "source": ("source_revisions", "source_id"),
            "claim": ("claim_revisions", "claim_id"),
            "span": ("spans", "span_id"),
            "source_envelope": ("source_envelopes", "envelope_id"),
        }.get(kind, (None, None))
        if table is None:
            return None
        r = conn.execute(
            f"SELECT MAX(revision) FROM {table} WHERE {col} = ?",
            (object_id,),
        ).fetchone()
        return int(r[0]) if r and r[0] is not None else None

    @staticmethod
    def _bytes_present(
        conn: sqlite3.Connection, kind: str, object_id: str, revision: int
    ) -> bool:
        """Absent-byte probe mirroring the kernel's emptied-payload gate
        (``length(payload) > 0`` — metadata only)."""
        if kind == "source":
            row = conn.execute(
                "SELECT length(payload) FROM source_revisions"
                " WHERE source_id = ? AND revision = ?",
                (object_id, int(revision)),
            ).fetchone()
        elif kind == "span":
            row = conn.execute(
                "SELECT length(r.payload) FROM spans s"
                " JOIN source_revisions r"
                "   ON r.source_id = s.source_id AND r.revision = s.revision"
                " WHERE s.span_id = ?",
                (object_id,),
            ).fetchone()
        else:
            return True
        return bool(row and row[0])

    @staticmethod
    def _epoch(conn: sqlite3.Connection, scope_id: str) -> int:
        from ..governance import epochs as _epochs

        return _epochs.current_epoch(conn, scope_id)


__all__ = ["DEFAULT_PRODUCER_ID", "RUBRIC_REVISION", "Synthesizer"]
