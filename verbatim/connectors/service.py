"""Connector pull engine (SPEC_V4 §48, V4-48.01–48.10).

``ConnectorService`` drives a connector's ``scan`` output through the
REAL write channel — every accepted item is persisted by
``evidence.envelopes.ingest_envelope`` (or its exact revision mirror),
so imported content gets the same UTF-8 gate, rules_v1 screening label,
quarantine hold, harvest obligation, and capture receipt as any other
capture. Imported content is untrusted input and is never bypassed
(V4-48.04 style: connectors are ingestion *clients* of the authority
services).

Durable bookkeeping (V4-48.04/48.05):

- ``ingest_batches`` — one batch row per pull. The manifest records the
  connector identity, cursor span, retention policy, and the permission
  snapshot (authorization id, authz revision, purpose) at commit time.
- ``connector_cursors`` — the resumable pull position, keyed by
  ``(connector@source-fingerprint, scope_id)`` so two sources on one
  scope never share a cursor. The cursor advances inside the SAME
  transaction that commits the page's accepted items, so a crash
  mid-pull resumes at the last committed page — retries can neither
  skip nor duplicate source changes (item dedup is the second line of
  defense: identical bytes under an identical external id replay to a
  no-op via the ``(scope_id, origin, external_id)`` dedup).

Conflict policy (V4-48.* dedup semantics): a re-imported item whose
content changed mints a NEW revision on the SAME ``sources`` row —
``ingest_envelope``'s own dedup probe pins revision 1, so the revision
case is applied by :func:`_revise_envelope`, which reuses that module's
own private steps verbatim. An identical re-import is a no-op
``duplicate``. The same external id under a *different* origin is not a
dedup hit; it is reported as a ``conflicts`` count (the item still
imports under the connector's own origin — the collision is reported,
never silently merged, V4-48.10).

Consent: a real pull requires an effective ``ingest`` grant for the
principal on the target scope AND a live ``capture_authorizations`` row
covering the ``connector_item`` envelope kind — retention consent, not
tool permission (§11.11). Every accepted envelope carries
``capture_proof`` so the write path re-validates consent per item; a
revoked authorization stops the next page honestly. Dry-runs require
the same ``ingest`` grant (existence probing is partition information)
but no retention consent — they persist nothing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from itertools import islice
from typing import Any, Callable, Mapping, Optional

import re

from ..core.identity import scope_key
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    JobKind,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from ..core.types_v3 import (
    CaptureAuthorization,
    EnvelopeKind,
    Perspective,
    SourceEnvelopeV3,
    TrustClass,
    Verb,
)
from ..evidence import detect
from ..evidence import envelopes as _env
from ..evidence.receipts import mint_receipt
from ..storage.repos import EventsRepo, SourcesRepo, SpansRepo, ensure_scope
from ..storage.repos_v2 import GrantsRepo, IngestBatchesRepo
from ..storage import repos_v3
from .. import governance

from .base import (
    ITEM_DETAIL_LIMIT,
    Connector,
    ConnectorDescriptor,
    PullReport,
    RemoteItem,
)

#: Envelope kind every connector item persists under (§12 universal
#: kinds). ``connector_item`` defaults to ``external_content`` trust;
#: assertion items override to ``imported`` (V4-48.02).
CONNECTOR_ENVELOPE_KIND = EnvelopeKind.CONNECTOR_ITEM

#: Tables the pull engine needs beyond the v1 core — connector imports
#: are a v3+ surface (consent rows, envelope provenance, screening).
_REQUIRED_TABLES = (
    "capture_authorizations",
    "grants_v3",
    "source_envelopes",
    "security_labels",
    "quarantine",
    "connector_cursors",
    "ingest_batches",
    "events",
)

_BATCH_POLICY = "connector-pull-v1"
_MAX_BATCH_SIZE = 512
_DEFAULT_BATCH_SIZE = 64


def _source_fingerprint(source: Mapping[str, Any]) -> str:
    """Stable short digest of the normalized source descriptor — the
    cursor ledger discriminates ``(connector, source, scope)`` so two
    directories/exports on one scope keep independent positions."""
    return hashlib.sha256(json_dumps(dict(source)).encode("utf-8")).hexdigest()[:24]


def ledger_connector_id(connector_id: str, source: Mapping[str, Any]) -> str:
    """The ``connector_cursors.connector_id`` key for this pull."""
    return f"{connector_id}@{_source_fingerprint(source)}"


_ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _actor_id(item: RemoteItem, origin: str) -> str:
    """Engine-id-safe actor attribution.

    A remote author handle (``alice@corp.com``) is not a valid engine id;
    when it cannot bind ``actor_principal``/``perspective.asserter`` the
    connector identity stands in and the raw handle is journaled in
    ``metadata.remote_actor`` — provenance is preserved, never dropped
    and never smuggled through an invalid id.
    """
    if item.author_id and _ID_SAFE.match(item.author_id):
        return item.author_id
    return origin


class ConnectorService:
    """Pull/dry-run engine over a ``Store``.

    ``fence`` — optional callable invoked inside every page transaction
    (the job handler passes a lease assertion so a superseded worker
    abandons the pull at the next page boundary, V4-42.01).
    """

    def __init__(self, store: Any, cfg: Any = None) -> None:
        from ..config import VerbatimConfig
        from ..jobs.queue import JobQueue

        self.store = store
        self.cfg = cfg if cfg is not None else VerbatimConfig()
        self.jobs = JobQueue(store, max_pending=self.cfg.jobs.max_pending)
        self.batches = IngestBatchesRepo(store)
        with self.store.read() as conn:
            self._tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }

    # ------------------------------------------------------------------
    # registry-facing helpers
    # ------------------------------------------------------------------

    def _require_schema(self) -> None:
        missing = [t for t in _REQUIRED_TABLES if t not in self._tables]
        if missing:
            raise VerbatimError(
                ErrorCode.SCHEMA_UNSUPPORTED,
                "connector pulls require the v3 consent/evidence schema; "
                f"missing tables: {', '.join(missing)}",
            )

    def _connector(self, connector_id: str) -> Connector:
        from .registry import get_connector

        return get_connector(connector_id)

    def list_connectors(self) -> list[dict[str, Any]]:
        from .registry import list_connectors

        return [d.to_dict() for d in list_connectors()]

    def describe(self, connector_id: str) -> dict[str, Any]:
        return self._connector(connector_id).descriptor().to_dict()

    def cursor(
        self,
        connector_id: str,
        source: Mapping[str, Any],
        scope_id: str,
    ) -> Optional[str]:
        """The durable pull position for ``(connector, source, scope)``."""
        connector = self._connector(connector_id)
        src = connector.validate_source(source)
        require_id(scope_id, "scope_id")
        with self.store.read() as conn:
            return self.batches.cursor_get(
                conn, ledger_connector_id(connector_id, src), scope_id
            )

    # ------------------------------------------------------------------
    # scope + consent
    # ------------------------------------------------------------------

    def _resolve_scope_id(
        self,
        conn: sqlite3.Connection,
        scope: Optional[Scope],
        scope_id: Optional[str],
        principal_id: str,
    ) -> str:
        if (scope is None) == (scope_id is None):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "pull needs exactly one of scope or scope_id",
            )
        if scope is not None:
            return ensure_scope(self.store, conn, scope)
        assert scope_id is not None
        require_id(scope_id, "scope_id")
        # Opaque v3 partition token: the bare row is created only when
        # absent (INSERT OR IGNORE — a pre-provisioned row wins).
        _env.ensure_scope_row(conn, scope_id, principal_id=principal_id)
        return scope_id

    def _authorize_pull(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        principal_id: str,
        *,
        purpose: Optional[str],
        caller_epoch: Optional[int],
        dry_run: bool,
    ) -> None:
        """The caller must hold an effective ``ingest`` grant on the
        target scope — dry-runs included (existence probing is partition
        information). Denial is indistinguishable (§10.05)."""
        governance.authorize(
            conn,
            governance.CallerV3(principal_id=principal_id, epoch=caller_epoch),
            scope_id,
            Verb.INGEST.value,
            purpose=purpose,
        )

    def _resolve_capture_authorization(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        principal_id: str,
        *,
        authorization: Optional[CaptureAuthorization],
        authorization_id: Optional[str],
    ) -> dict[str, Any]:
        """Resolve the retention-consent record covering
        ``connector_item`` on this scope; returns the permission snapshot
        journaled into the batch manifest (V4-48.04).

        Three forms, strictest first: an explicit ``CaptureAuthorization``
        object, a named ``authorization_id``, or any live row covering
        (principal, connector_item, scope). All paths verify kind, scope,
        expiry, and revocation — missing consent is ``CONSENT_REQUIRED``,
        never a silent downgrade.
        """
        now = now_us()
        if authorization is not None:
            if not isinstance(authorization, CaptureAuthorization):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "authorization must be a CaptureAuthorization",
                )
            if CONNECTOR_ENVELOPE_KIND not in authorization.allowed_kinds:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED,
                    "authorization does not cover envelope kind 'connector_item'",
                )
            if authorization.scope_ids and scope_id not in authorization.scope_ids:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED,
                    "authorization does not cover this scope",
                )
            if (
                authorization.expires_us is not None
                and authorization.expires_us <= now
            ):
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED, "capture authorization expired"
                )
            row = repos_v3.get(
                conn,
                "capture_authorizations",
                {"authorization_id": authorization.authorization_id},
            )
            if row is not None and row["revoked_us"] is not None:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED, "capture authorization revoked"
                )
            return {
                "authorization_id": authorization.authorization_id,
                "retention_policy": authorization.retention_policy,
                "policy_revision": authorization.policy_revision,
                "form": "explicit",
            }
        if authorization_id is not None:
            require_id(authorization_id, "authorization_id")
            row = repos_v3.get(
                conn,
                "capture_authorizations",
                {"authorization_id": authorization_id},
            )
            if row is None or row.get("revoked_us") is not None:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED,
                    "capture authorization absent or revoked",
                )
            exp = row.get("expires_us")
            if exp is not None and int(exp) <= now:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED, "capture authorization expired"
                )
            kinds = repos_v3.json_field(row, "allowed_kinds_json", []) or []
            if CONNECTOR_ENVELOPE_KIND.value not in kinds:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED,
                    "authorization does not cover envelope kind 'connector_item'",
                )
            scopes = repos_v3.json_field(row, "scope_ids_json", []) or []
            if scopes and scope_id not in scopes:
                raise VerbatimError(
                    ErrorCode.CONSENT_REQUIRED,
                    "authorization does not cover this scope",
                )
            return {
                "authorization_id": authorization_id,
                "retention_policy": row.get("retention_policy") or "",
                "policy_revision": row.get("policy_revision") or "",
                "form": "referenced",
            }
        # Probe the principal's live consent rows (same rule
        # capture_submitted applies through require_capture_authorization).
        governance.require_capture_authorization(
            conn, principal_id, CONNECTOR_ENVELOPE_KIND, scope_id
        )
        rows = repos_v3.query(
            conn,
            "capture_authorizations",
            {"principal_id": principal_id, "revoked_us": None},
        )
        for row in rows:
            exp = row.get("expires_us")
            if exp is not None and int(exp) <= now:
                continue
            kinds = repos_v3.json_field(row, "allowed_kinds_json") or []
            if CONNECTOR_ENVELOPE_KIND.value not in kinds:
                continue
            scopes = repos_v3.json_field(row, "scope_ids_json", []) or []
            if scopes and scope_id not in scopes:
                continue
            return {
                "authorization_id": row["authorization_id"],
                "retention_policy": row.get("retention_policy") or "",
                "policy_revision": row.get("policy_revision") or "",
                "form": "probed",
            }
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            "capture consent verified but no live row resolves",
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def pull(
        self,
        connector_id: str,
        source: Mapping[str, Any],
        *,
        scope: Optional[Scope] = None,
        scope_id: Optional[str] = None,
        principal_id: str,
        purpose: Optional[str] = None,
        authorization: Optional[CaptureAuthorization] = None,
        authorization_id: Optional[str] = None,
        caller_epoch: Optional[int] = None,
        dry_run: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_items: Optional[int] = None,
        resume_from: Optional[str] = None,
        fence: Optional[Callable[[sqlite3.Connection], None]] = None,
    ) -> PullReport:
        """Run one pull; returns the honest :class:`PullReport`.

        ``dry_run=True`` enumerates and classifies every item — counts,
        payload digests, duplicates, conflicts, sensitivity findings,
        scope mappings, format losses — and writes nothing (V4-48.03).
        """
        require_id(principal_id, "principal_id")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not (1 <= batch_size <= _MAX_BATCH_SIZE)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"batch_size must be an int in [1, {_MAX_BATCH_SIZE}]",
            )
        if max_items is not None and (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or max_items < 1
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION, "max_items must be an int >= 1"
            )
        connector = self._connector(connector_id)
        desc = connector.descriptor()
        if desc.remote:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"connector {connector_id!r} needs network egress — remote "
                "connectors are declared-unavailable in this build",
            )
        src = connector.validate_source(source)
        self._require_schema()

        if dry_run:
            return self._dry_run(
                connector,
                desc,
                src,
                scope=scope,
                scope_id=scope_id,
                principal_id=principal_id,
                purpose=purpose,
                caller_epoch=caller_epoch,
                batch_size=batch_size,
                max_items=max_items,
                resume_from=resume_from,
            )
        return self._pull(
            connector,
            desc,
            src,
            scope=scope,
            scope_id=scope_id,
            principal_id=principal_id,
            purpose=purpose,
            authorization=authorization,
            authorization_id=authorization_id,
            caller_epoch=caller_epoch,
            batch_size=batch_size,
            max_items=max_items,
            resume_from=resume_from,
            fence=fence,
        )

    def schedule_pull(
        self,
        connector_id: str,
        source: Mapping[str, Any],
        *,
        scope: Optional[Scope] = None,
        scope_id: Optional[str] = None,
        principal_id: str,
        purpose: Optional[str] = None,
        authorization_id: Optional[str] = None,
        caller_epoch: Optional[int] = None,
        dry_run: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_items: Optional[int] = None,
        pull_id: Optional[str] = None,
    ) -> str:
        """Enqueue a durable ``connector_pull`` job; returns ``job_id``.

        Consent is verified at enqueue (fail fast) and re-verified by the
        drain-time handler — a revocation between the two denies the pull
        honestly. ``pull_id`` pins the operation identity: a re-enqueue
        with the same id converges on the existing job (dedup_key) and a
        redelivery after commit replays the recorded receipt
        (operation_key → operations ledger, V2-39.10).
        """
        connector = self._connector(connector_id)
        src = connector.validate_source(source)
        self._require_schema()
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not (1 <= batch_size <= _MAX_BATCH_SIZE)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"batch_size must be an int in [1, {_MAX_BATCH_SIZE}]",
            )
        pid = pull_id or new_id()
        require_id(pid, "pull_id")
        refs = {
            "connector_id": connector_id,
            "source": dict(src),
            "principal_id": principal_id,
            "purpose": purpose,
            "authorization_id": authorization_id,
            "dry_run": bool(dry_run),
            "batch_size": int(batch_size),
            "pull_id": pid,
        }
        if max_items is not None:
            refs["max_items"] = int(max_items)
        with self.store.tx() as conn:
            sid = self._resolve_scope_id(conn, scope, scope_id, principal_id)
            refs["scope_id"] = sid
            self._authorize_pull(
                conn,
                sid,
                principal_id,
                purpose=purpose,
                caller_epoch=caller_epoch,
                dry_run=dry_run,
            )
            if not dry_run:
                self._resolve_capture_authorization(
                    conn,
                    sid,
                    principal_id,
                    authorization=None,
                    authorization_id=authorization_id,
                )
            dedup = self.store.hmac(f"connector_pull:{pid}".encode())
            return self.jobs.enqueue(
                conn,
                sid,
                JobKind.CONNECTOR_PULL,
                refs,
                dedup_key=dedup,
                operation_key=(
                    f"connector_pull:{pid}"
                    if self.jobs.supports_durability
                    else None
                ),
            )

    # ------------------------------------------------------------------
    # dry run (V4-48.03) — writes nothing
    # ------------------------------------------------------------------

    def _dry_run(
        self,
        connector: Connector,
        desc: ConnectorDescriptor,
        src: Mapping[str, Any],
        *,
        scope: Optional[Scope],
        scope_id: Optional[str],
        principal_id: str,
        purpose: Optional[str],
        caller_epoch: Optional[int],
        batch_size: int,
        max_items: Optional[int],
        resume_from: Optional[str],
    ) -> PullReport:
        counts: dict[str, int] = {
            "scanned": 0,
            "would_insert": 0,
            "would_revise": 0,
            "duplicates": 0,
            "would_reject": 0,
            "conflicts": 0,
            "would_quarantine": 0,
            "missing_provenance": 0,
        }
        sensitivity: dict[str, int] = {}
        losses: dict[str, int] = {}
        scope_mappings: dict[str, str] = {}
        items: list[dict[str, Any]] = []
        bytes_seen = 0
        cursor_before: Optional[str] = None
        cursor_after: Optional[str] = None
        truncated = False

        with self.store.read() as conn:
            # Scope resolution for a dry run is read-only: a Scope object
            # derives its digest; a scope_id must already exist.
            if scope is not None:
                sid = scope_key(scope)
            elif scope_id is not None:
                require_id(scope_id, "scope_id")
                sid = scope_id
            else:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "pull needs exactly one of scope or scope_id",
                )
            self._authorize_pull(
                conn,
                sid,
                principal_id,
                purpose=purpose,
                caller_epoch=caller_epoch,
                dry_run=True,
            )
            stored = self.batches.cursor_get(
                conn, ledger_connector_id(desc.connector_id, src), sid
            )
            cursor_before = (
                stored if resume_from is None else (resume_from or None)
            )
            ledger_origin = (
                f"connector:{desc.connector_id}:"
                f"{_source_fingerprint(src)[:12]}"
            )
            for item in connector.scan(src, cursor=cursor_before):
                if max_items is not None and counts["scanned"] >= max_items:
                    truncated = True
                    break
                counts["scanned"] += 1
                cursor_after = item.cursor
                bytes_seen += len(item.content)
                outcome = self._classify(conn, sid, ledger_origin, item)
                counts[outcome["bucket"]] += 1
                if outcome.get("would_quarantine"):
                    counts["would_quarantine"] += 1
                if outcome.get("conflict"):
                    counts["conflicts"] += 1
                for f in outcome.get("findings") or ():
                    sensitivity[f] = sensitivity.get(f, 0) + 1
                for l in item.losses:
                    losses[l] = losses.get(l, 0) + 1
                remote_ns = item.extra.get("scope") or item.extra.get("namespace")
                if isinstance(remote_ns, str) and remote_ns:
                    scope_mappings[remote_ns] = sid
                if item.missing_provenance:
                    counts["missing_provenance"] += 1
                if len(items) < ITEM_DETAIL_LIMIT:
                    entry = {
                        "external_id": item.external_id,
                        "action": outcome["action"],
                        "reason": outcome.get("reason"),
                        "bytes": len(item.content),
                        "digest": hashlib.sha256(item.content).hexdigest()[:24],
                        "item_class": item.item_class,
                        "imported_assertion": item.imported_assertion,
                        # V45-09.06: provenance loss is named per item,
                        # not only in the aggregate count.
                        "missing_provenance": list(
                            item.missing_provenance
                        ),
                        "losses": list(item.losses),
                    }
                    if outcome.get("existing_source_id"):
                        entry["existing_source_id"] = outcome[
                            "existing_source_id"
                        ]
                    items.append(entry)
        pages = (
            (counts["scanned"] + batch_size - 1) // batch_size
            if counts["scanned"]
            else 0
        )
        return PullReport(
            pull_id=f"dryrun:{new_id()}",
            connector_id=desc.connector_id,
            connector_version=desc.version,
            scope_id=sid,
            principal_id=principal_id,
            dry_run=True,
            state="partial" if truncated else "dry_run",
            cursor_before=cursor_before,
            cursor_after=cursor_after if counts["scanned"] else cursor_before,
            scanned=counts["scanned"],
            inserted=counts["would_insert"],
            revised=counts["would_revise"],
            duplicates=counts["duplicates"],
            rejected=counts["would_reject"],
            conflicts=counts["conflicts"],
            quarantined=counts["would_quarantine"],
            bytes_seen=bytes_seen,
            bytes_accepted=0,
            pages=pages,
            missing_provenance=counts["missing_provenance"],
            sensitivity=sensitivity,
            losses=losses,
            scope_mappings=scope_mappings,
            items=tuple(items),
        )

    def _classify(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        ledger_origin: str,
        item: RemoteItem,
    ) -> dict[str, Any]:
        """Dry-run classification — the same probes the write path runs,
        minus every write. Sensitivity findings come from the real
        rules_v1 screener on the item text."""
        from ..security.screening import screen_content

        if item.reject_reason is not None:
            return {
                "action": "would_reject",
                "bucket": "would_reject",
                "reason": item.reject_reason,
            }
        payload = bytes(item.content)
        if not payload:
            return {
                "action": "would_reject",
                "bucket": "would_reject",
                "reason": "empty",
            }
        if len(payload) > self.cfg.capture.max_source_bytes:
            return {
                "action": "would_reject",
                "bucket": "would_reject",
                "reason": "too_large",
            }
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "action": "would_reject",
                "bucket": "would_reject",
                "reason": "invalid_utf8",
            }
        findings: list[str] = []
        would_quarantine = False
        try:
            verdict = screen_content(
                text,
                source_trust=(
                    TrustClass.IMPORTED.value
                    if item.imported_assertion
                    else TrustClass.EXTERNAL_CONTENT.value
                ),
                context_kind=CONNECTOR_ENVELOPE_KIND.value,
            )
            findings = sorted(
                {str(f.get("rule_id")) for f in verdict.findings if f.get("rule_id")}
            )
            would_quarantine = verdict.attack_risk.value in (
                "suspicious",
                "blocked",
            )
        except VerbatimError:
            # A screening failure cannot clear the item — count it as
            # quarantined rather than reporting clean (fail closed).
            would_quarantine = True
        row = conn.execute(
            "SELECT source_id FROM sources"
            " WHERE scope_id = ? AND origin = ? AND external_id = ?",
            (scope_id, ledger_origin, item.external_id),
        ).fetchone()
        collision = conn.execute(
            "SELECT 1 FROM sources"
            " WHERE scope_id = ? AND external_id = ? AND origin <> ?"
            " LIMIT 1",
            (scope_id, item.external_id, ledger_origin),
        ).fetchone()
        out: dict[str, Any] = {
            "findings": findings,
            "would_quarantine": would_quarantine,
        }
        if collision is not None:
            out["conflict"] = True
        if row is None:
            out.update({"action": "would_insert", "bucket": "would_insert"})
            return out
        sid = row[0]
        out["existing_source_id"] = sid
        rev = conn.execute(
            "SELECT revision, payload_hmac FROM source_revisions"
            " WHERE source_id = ? ORDER BY revision DESC LIMIT 1",
            (sid,),
        ).fetchone()
        same = rev is not None and bytes(rev[1]) == self.store.hmac(payload)
        out.update(
            {
                "action": "duplicate" if same else "would_revise",
                "bucket": "duplicates" if same else "would_revise",
            }
        )
        return out

    # ------------------------------------------------------------------
    # real pull
    # ------------------------------------------------------------------

    def _pull(
        self,
        connector: Connector,
        desc: ConnectorDescriptor,
        src: Mapping[str, Any],
        *,
        scope: Optional[Scope],
        scope_id: Optional[str],
        principal_id: str,
        purpose: Optional[str],
        authorization: Optional[CaptureAuthorization],
        authorization_id: Optional[str],
        caller_epoch: Optional[int],
        batch_size: int,
        max_items: Optional[int],
        resume_from: Optional[str],
        fence: Optional[Callable[[sqlite3.Connection], None]],
    ) -> PullReport:
        ledger_id: Optional[str] = None
        batch_id: Optional[str] = None
        permission: dict[str, Any] = {}

        # ---- tx 1: scope + authorization + consent + batch row --------
        with self.store.tx() as conn:
            if fence is not None:
                fence(conn)
            sid = self._resolve_scope_id(conn, scope, scope_id, principal_id)
            self._authorize_pull(
                conn,
                sid,
                principal_id,
                purpose=purpose,
                caller_epoch=caller_epoch,
                dry_run=False,
            )
            permission = self._resolve_capture_authorization(
                conn,
                sid,
                principal_id,
                authorization=authorization,
                authorization_id=authorization_id,
            )
            ledger_id = ledger_connector_id(desc.connector_id, src)
            cursor_before = self.batches.cursor_get(conn, ledger_id, sid)
            if resume_from is not None:
                cursor_before = resume_from or None
            permission["authz_revision"] = GrantsRepo(self.store).authz_revision(
                conn, sid
            )
            permission["principal_id"] = principal_id
            if purpose:
                permission["purpose"] = purpose
            batch_id = self.batches.create_batch(
                conn,
                sid,
                manifest={
                    "connector_id": desc.connector_id,
                    "connector_version": desc.version,
                    "source": dict(src),
                    "ledger_id": ledger_id,
                    "scope_id": sid,
                    "principal_id": principal_id,
                    "cursor_before": cursor_before,
                    "permission": permission,
                    "retention_policy": permission.get("retention_policy", ""),
                    "policy": _BATCH_POLICY,
                    "declared_not_verified": list(desc.declared_not_verified),
                },
            )
            EventsRepo(self.store).append(
                conn,
                sid,
                "connector_pull_started",
                principal_id,
                {
                    "batch_id": batch_id,
                    "connector_id": desc.connector_id,
                    "cursor_before": cursor_before,
                },
                _BATCH_POLICY,
            )
        aid = permission["authorization_id"]
        # Dedup origin discriminates (connector, source): two directories
        # or exports on one scope hold independent external_id spaces —
        # same relpath in two roots is two sources, never a false dedup
        # hit. The ledger key already carries the same fingerprint.
        origin = f"connector:{desc.connector_id}:{_source_fingerprint(src)[:12]}"

        counts: dict[str, int] = {
            "scanned": 0,
            "inserted": 0,
            "revised": 0,
            "duplicates": 0,
            "rejected": 0,
            "conflicts": 0,
            "quarantined": 0,
            "missing_provenance": 0,
        }
        losses: dict[str, int] = {}
        sensitivity: dict[str, int] = {}
        scope_mappings: dict[str, str] = {}
        items: list[dict[str, Any]] = []
        bytes_seen = 0
        bytes_accepted = 0
        pages = 0
        cursor_after = cursor_before
        truncated = False

        iterator = iter(connector.scan(src, cursor=cursor_before))
        while True:
            page = list(
                islice(
                    iterator,
                    min(
                        batch_size,
                        (
                            max_items - counts["scanned"]
                            if max_items is not None
                            else batch_size
                        ),
                    ),
                )
            )
            if not page:
                break
            pages += 1
            with self.store.tx() as conn:
                if fence is not None:
                    fence(conn)
                for item in page:
                    counts["scanned"] += 1
                    bytes_seen += len(item.content)
                    outcome = self._import_item(
                        conn,
                        desc=desc,
                        scope_id=sid,
                        principal_id=principal_id,
                        item=item,
                        batch_id=batch_id,
                        authorization_id=aid,
                        origin=origin,
                    )
                    action = outcome["action"]
                    if action == "inserted":
                        counts["inserted"] += 1
                        bytes_accepted += len(item.content)
                    elif action == "revised":
                        counts["revised"] += 1
                        bytes_accepted += len(item.content)
                    elif action == "duplicate":
                        counts["duplicates"] += 1
                    else:
                        counts["rejected"] += 1
                    if outcome.get("conflict"):
                        counts["conflicts"] += 1
                    if outcome.get("quarantined"):
                        counts["quarantined"] += 1
                    if item.missing_provenance:
                        counts["missing_provenance"] += 1
                    for l in item.losses:
                        losses[l] = losses.get(l, 0) + 1
                    for f in outcome.get("findings") or ():
                        sensitivity[f] = sensitivity.get(f, 0) + 1
                    remote_ns = item.extra.get("scope") or item.extra.get(
                        "namespace"
                    )
                    if isinstance(remote_ns, str) and remote_ns:
                        scope_mappings[remote_ns] = sid
                    if len(items) < ITEM_DETAIL_LIMIT:
                        items.append(
                            {
                                "external_id": item.external_id,
                                "action": action,
                                "reason": outcome.get("reason"),
                                "source_id": outcome.get("source_id"),
                                "revision": outcome.get("revision"),
                                "bytes": len(item.content),
                                "digest": hashlib.sha256(
                                    item.content
                                ).hexdigest()[:24],
                                "item_class": item.item_class,
                                "imported_assertion": item.imported_assertion,
                                # V45-09.06: the accepted receipt names
                                # provenance loss per item, same as the
                                # dry-run preview did.
                                "missing_provenance": list(
                                    item.missing_provenance
                                ),
                                "losses": list(item.losses),
                                "quarantined": bool(
                                    outcome.get("quarantined")
                                ),
                            }
                        )
                # V4-48.05: cursor + accepted batch receipt commit in the
                # SAME transaction as the page's item writes.
                cursor_after = page[-1].cursor
                self.batches.cursor_set(conn, ledger_id, sid, cursor_after)
                self.batches.update_batch(
                    conn,
                    batch_id,
                    received=counts["scanned"],
                    accepted=counts["inserted"] + counts["revised"],
                    state="partial",
                )
            if max_items is not None and counts["scanned"] >= max_items:
                truncated = True
                break

        # ---- final tx: batch close + manifest + completion event ------
        state = "partial" if truncated else "complete"
        with self.store.tx() as conn:
            if fence is not None:
                fence(conn)
            self.batches.update_batch(
                conn,
                batch_id,
                received=counts["scanned"],
                accepted=counts["inserted"] + counts["revised"],
                state=state,
            )
            conn.execute(
                "UPDATE ingest_batches SET manifest_json = ?"
                " WHERE batch_id = ?",
                (
                    json_dumps(
                        {
                            "connector_id": desc.connector_id,
                            "connector_version": desc.version,
                            "source": dict(src),
                            "ledger_id": ledger_id,
                            "scope_id": sid,
                            "principal_id": principal_id,
                            "cursor_before": cursor_before,
                            "cursor_after": cursor_after,
                            "permission": permission,
                            "retention_policy": permission.get(
                                "retention_policy", ""
                            ),
                            "policy": _BATCH_POLICY,
                            "declared_not_verified": list(
                                desc.declared_not_verified
                            ),
                            "counts": dict(counts),
                            "losses": dict(losses),
                            "scope_mappings": dict(scope_mappings),
                            "items": list(items),
                        }
                    ),
                    batch_id,
                ),
            )
            EventsRepo(self.store).append(
                conn,
                sid,
                "connector_pull_completed",
                principal_id,
                {
                    "batch_id": batch_id,
                    "connector_id": desc.connector_id,
                    "state": state,
                    "counts": dict(counts),
                    "cursor_after": cursor_after,
                },
                _BATCH_POLICY,
            )
        return PullReport(
            pull_id=batch_id,
            batch_id=batch_id,
            connector_id=desc.connector_id,
            connector_version=desc.version,
            scope_id=sid,
            principal_id=principal_id,
            dry_run=False,
            state=state,
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            scanned=counts["scanned"],
            inserted=counts["inserted"],
            revised=counts["revised"],
            duplicates=counts["duplicates"],
            rejected=counts["rejected"],
            conflicts=counts["conflicts"],
            quarantined=counts["quarantined"],
            bytes_seen=bytes_seen,
            bytes_accepted=bytes_accepted,
            pages=pages,
            missing_provenance=counts["missing_provenance"],
            sensitivity=sensitivity,
            losses=losses,
            scope_mappings=scope_mappings,
            items=tuple(items),
        )

    # ------------------------------------------------------------------
    # per-item write path
    # ------------------------------------------------------------------

    def _import_item(
        self,
        conn: sqlite3.Connection,
        *,
        desc: ConnectorDescriptor,
        scope_id: str,
        principal_id: str,
        item: RemoteItem,
        batch_id: str,
        authorization_id: str,
        origin: str,
    ) -> dict[str, Any]:
        """Ingest ONE item inside the caller's page transaction.

        Item-level permanent refusals (connector reject flag, empty,
        oversize, malformed UTF-8) are recorded outcomes — they never
        abort the pull and are never silently skipped. Everything else
        propagates so a systemic failure rolls the whole page back.
        """
        if item.reject_reason is not None:
            return {"action": "rejected", "reason": item.reject_reason}
        payload = bytes(item.content)
        if not payload:
            return {"action": "rejected", "reason": "empty"}
        if len(payload) > self.cfg.capture.max_source_bytes:
            return {"action": "rejected", "reason": "too_large"}
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            # C85/V4-13.11: malformed bytes are refused honestly — never
            # replacement-decoded into the store.
            return {"action": "rejected", "reason": "invalid_utf8"}

        row = conn.execute(
            "SELECT source_id FROM sources"
            " WHERE scope_id = ? AND origin = ? AND external_id = ?",
            (scope_id, origin, item.external_id),
        ).fetchone()
        collision = conn.execute(
            "SELECT 1 FROM sources"
            " WHERE scope_id = ? AND external_id = ? AND origin <> ?"
            " LIMIT 1",
            (scope_id, item.external_id, origin),
        ).fetchone() is not None
        payload_hmac = self.store.hmac(payload)

        actor = _actor_id(item, f"connector:{desc.connector_id}")
        envelope = SourceEnvelopeV3(
            kind=CONNECTOR_ENVELOPE_KIND,
            scope_id=scope_id,
            # Attribution honesty: the actor is the remote author when the
            # connector supplies an engine-safe id, else the connector
            # itself — imported content is never recorded as the importing
            # principal's testimony (V3-13.11 analog).
            actor_principal=actor,
            perspective=Perspective(
                asserter=actor,
                observer=principal_id,
            ),
            event_us=(
                int(item.event_us) if item.event_us is not None else now_us()
            ),
            receipt_us=0,
            content=payload,
            media_type=item.media_type or "text/plain",
            # V4-48.02: extracted facts/summaries without originals persist
            # as imported assertions; verbatim originals keep
            # external_content trust.
            trust_class=(
                TrustClass.IMPORTED
                if item.imported_assertion
                else TrustClass.EXTERNAL_CONTENT
            ),
            capture_proof=authorization_id,
            adapter_version=f"{desc.connector_id}/{desc.version}",
            host_id=origin,
            external_id=item.external_id,
            metadata={
                "connector_id": desc.connector_id,
                "connector_version": desc.version,
                "formats": list(desc.formats),
                "declared_not_verified": list(desc.declared_not_verified),
                "remote_actor": item.author_id,
                "pull_id": batch_id,
                "cursor": item.cursor,
                "remote_revision": item.remote_revision,
                "item_class": item.item_class,
                "imported_assertion": item.imported_assertion,
                "original_present": not item.imported_assertion,
                "missing_provenance": list(item.missing_provenance),
                "source_ids": list(item.source_ids),
                "sensitivity_hints": list(item.sensitivity_hints),
                "losses": list(item.losses),
            },
        )

        if row is None:
            receipt = _env.ingest_envelope(conn, self.store, envelope)
            action = "inserted"
        else:
            sid = row[0]
            latest = conn.execute(
                "SELECT revision, payload_hmac FROM source_revisions"
                " WHERE source_id = ? ORDER BY revision DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if latest is not None and bytes(latest[1]) == payload_hmac:
                # Idempotent dedup hit — the same remote bytes under the
                # same identity produce no write (V4-48.05).
                return {
                    "action": "duplicate",
                    "source_id": sid,
                    "revision": int(latest[0]),
                    "conflict": collision,
                }
            receipt = _revise_envelope(conn, self.store, envelope, sid)
            action = "revised"

        quarantined = self._envelope_held(
            conn, receipt.envelope_id, receipt.revision
        )
        # Per-receipt readiness DAG — same convergence call the v3 facade
        # makes; no-op when the v4 table is absent.
        try:
            from ..readiness import ReadinessEngine

            ReadinessEngine(self.store).ensure_for_source(
                conn, receipt.source_id, int(receipt.revision)
            )
        except VerbatimError:
            raise
        except Exception:
            pass  # readiness is additive bookkeeping; capture already holds
        return {
            "action": action,
            "source_id": receipt.source_id,
            "revision": int(receipt.revision),
            "conflict": collision,
            "quarantined": quarantined,
        }

    def _envelope_held(
        self, conn: sqlite3.Connection, envelope_id: str, revision: int
    ) -> bool:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = 'source_envelope' AND object_id = ?"
            " AND revision = ?",
            (envelope_id, revision),
        ).fetchone()
        return row is not None and row[0] in ("pending", "suppressed")


def _revise_envelope(
    conn: sqlite3.Connection,
    store: Any,
    envelope: SourceEnvelopeV3,
    source_id: str,
) -> Any:
    """Mint a NEW revision on an existing source for changed re-imports.

    ``ingest_envelope``'s dedup probe pins revision 1: identical bytes
    replay, different bytes raise a conflict — it cannot express "same
    external identity, new content". The connector conflict policy
    requires the v2 revision semantics (same ``sources`` row, next
    revision), so this mirrors ``ingest_envelope``'s accepted tail
    exactly — same revision row, whole-payload span, envelope row,
    screening label + quarantine, harvest obligation, journal event,
    receipt — reusing that module's own private helpers so the two paths
    cannot drift apart in policy.
    """
    policy_revision = _env._check_capture_permission(conn, envelope, None)
    trust = _env._effective_trust(envelope)
    payload = _env._payload_bytes(envelope)
    dedup_key = _env._dedup_external_id(envelope, payload)
    payload_hmac = store.hmac(payload)
    engine_us = getattr(store, "next_event_us", None) or now_us
    receipt_us = envelope.receipt_us or engine_us()

    row = conn.execute(
        "SELECT COALESCE(MAX(revision), 0) FROM source_revisions"
        " WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    revision = int(row[0]) + 1
    origin = envelope.host_id or "v3"
    SourcesRepo._insert_revision(
        conn,
        source_id,
        _env._v2_envelope(
            envelope, origin, dedup_key, revision, payload, receipt_us
        ),
        payload,
        payload_hmac,
    )
    start, end = detect.whole_payload_span(payload)
    span_id = detect.span_id_for(source_id, revision, start, end)
    SpansRepo(store).insert(
        span_id, source_id, revision, start, end, detect.PARSER_VERSION,
        conn=conn,
    )
    conn.execute(
        "UPDATE spans SET operation_key = ? WHERE span_id = ?",
        (f"capture:{source_id}:{revision}:{envelope.kind.value}", span_id),
    )
    env_row = _env._insert_envelope_row(
        conn,
        envelope,
        envelope_id=_env._envelope_id_for(source_id, revision, envelope.kind),
        source_id=source_id,
        revision=revision,
        perspective_id=_env._persist_perspective(conn, envelope),
        trust=trust,
        receipt_us=receipt_us,
    )
    label_id = _env._maybe_screen(
        conn, envelope, trust,
        envelope_id=env_row["envelope_id"], revision=revision,
    )
    _env._link_label(conn, env_row["envelope_id"], label_id)
    _env._enqueue_harvest(conn, store, envelope, source_id, revision)
    event_seq = _env._record_event(
        conn,
        store,
        envelope,
        {
            "envelope_id": env_row["envelope_id"],
            "source_id": source_id,
            "revision": revision,
            "envelope_kind": envelope.kind.value,
            "dedup_key": dedup_key,
            "span_ids": [span_id],
            "security_label_id": label_id,
            "connector_revision": True,
        },
        policy_revision,
    )
    return mint_receipt(
        conn,
        env_row,
        event_seq=event_seq,
        dedup_key=dedup_key,
        span_ids=(span_id,),
        accepted_bytes=len(payload),
    )


__all__ = [
    "CONNECTOR_ENVELOPE_KIND",
    "ConnectorService",
    "ledger_connector_id",
]
