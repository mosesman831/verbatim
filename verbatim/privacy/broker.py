"""Transport broker — the single egress authority (SPEC_V4 §12, F4-04).

All outbound content — document embeddings, query embeddings, reranking,
judgments, synthesis, connector uploads, diagnostics — crosses THIS broker
(V4-12.01). Configuration permits an *endpoint* to exist; only a current
``DispatchPermit`` authorizes a particular dispatch.

Pipeline position (V4-12.02/03/05):

    caller ──open_dispatch──▶  permit + budget reservation (atomic, durable)
    caller ──encode(permit)─▶  encoder verifies recipient + payload digest,
                               broker.dispatch() rechecks consent/suppression/
                               epochs/reservation and flips open→dispatched
                               INSIDE one transaction — then, and only then,
                               transport I/O runs.
    caller ──reconcile─────▶  reservation settles/overruns/expires; permit
                               row records the outcome for crash correlation.

Enforcement rules:

- ``dispatch_permits`` + ``budget_ledger`` writes happen in ONE transaction
  before any network I/O (V4-12.05). A crash between reservation and
  dispatch leaves a ``reserved`` ledger row that still counts against the
  day — conservative retention until ``reconcile``/``expire_stale``.
- EVERY contributing scope needs an active consent row for
  (recipient processor, purpose) at issuance AND again at dispatch
  (V4-12.03, V4-12.06). The host's default scope is never substituted.
- Zero configured remote budget denies every dispatch, including nominally
  free ones (V4-12.04) — ``max_spend`` must be a positive worst-case bound
  so a reservation is always real.
- Permits are one-use and short-lived (default max age 1 s, V4-09.08):
  a dispatched/expired/denied permit can never replay (C18).
- Endpoints are allowlisted ``EndpointDescriptor`` objects validated at
  construction: TLS required off-loopback, redirects and credential
  forwarding off by default, credentials never carried in origins
  (V4-12.07). The broker validates descriptors; the encoder's own HTTP
  client still executes the transport.
- Receipts persist permit→request correlation (permit id, reservation id,
  consent ids, payload digest, scopes) without retaining raw prompts or
  credentials (V4-12.10).

Consent/suppression/pricing checks are delegated to
:class:`~verbatim.privacy.egress.EgressGate` so there is one consent
predicate and one ledger — the broker adds permit issuance, binding, and
the one-use dispatch transition on top. ``EgressGate.authorize`` remains
the legacy reservation facade for the JEV judge path; both write the same
``budget_ledger`` and count toward the same day spend.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    Mode,
    VerbatimError,
    new_id,
    require_id,
    safe_json_loads,
)
from ..storage import repos_v4
from .egress import EgressGate, _utc_day

# ---------------------------------------------------------------------------
# Payload digests (V4-12.02) — the permit binds the EXACT outbound bytes.
# ---------------------------------------------------------------------------

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def payload_digest(data: bytes) -> str:
    """Canonical payload digest: ``sha256:<hex>`` over the exact bytes that
    will cross the transport. Encoders expose ``request_payload(texts)`` /
    ``payload_digest(texts)`` so callers can mint a permit for the identical
    body the encoder later serializes — a mismatch is detectable, never
    silently re-derived."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        raise VerbatimError(ErrorCode.VALIDATION, "payload_digest needs bytes")
    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


def _check_digest(value: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.match(value):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "payload_digest must be 'sha256:<64 hex>' — use payload_digest()",
        )
    return value


# ---------------------------------------------------------------------------
# Endpoint descriptors (V4-12.07) — allowlist + transport guardrails.
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MAX_REQUEST_BYTES_CAP = 64 << 20


@dataclass(frozen=True)
class EndpointDescriptor:
    """One allowlisted transport endpoint.

    ``origin`` is a bare ``scheme://host[:port]`` — paths, queries,
    fragments, and embedded credentials are rejected (a configured URL is
    attacker-influenceable input, not proof of identity). TLS verification
    is mandatory for every non-loopback origin; ``require_tls=False`` is
    only legal on verified loopback, and ``allow_redirects`` +
    ``forward_credentials`` together are refused (a redirect must never
    carry credentials to a second origin).
    """

    endpoint_id: str
    origin: str
    require_tls: bool = True
    allow_redirects: bool = False
    forward_credentials: bool = False
    max_request_bytes: int = 32 << 20
    account: Optional[str] = None       # provider account/region binding
    processor: Optional[str] = None     # consent processor; default endpoint_id

    def __post_init__(self) -> None:
        require_id(self.endpoint_id, "endpoint_id")
        if not isinstance(self.origin, str) or not self.origin:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "endpoint origin required")
        try:
            parts = urlsplit(self.origin)
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"endpoint origin unparsable: {exc}"
            ) from exc
        if parts.scheme not in ("https", "http"):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"endpoint origin scheme {parts.scheme!r} unsupported",
            )
        host = (parts.hostname or "").lower()
        if not host:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "endpoint origin has no host")
        try:
            parts.port  # noqa: B018 — raises ValueError on malformed ports
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "endpoint origin port invalid"
            ) from exc
        if parts.username or parts.password:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "endpoint origin must not embed credentials",
            )
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "endpoint origin must be an origin only (no path/query/fragment)",
            )
        loopback = host in _LOOPBACK_HOSTS
        object.__setattr__(self, "_loopback", loopback)
        if parts.scheme == "http" and not loopback:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"plain-http endpoint {host!r} is not loopback — TLS required",
            )
        if self.require_tls and parts.scheme != "https":
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "require_tls=True needs an https origin (loopback http may "
                "set require_tls=False explicitly)",
            )
        if not self.require_tls and parts.scheme != "http":
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "require_tls=False is only meaningful for verified-loopback http",
            )
        if self.allow_redirects and self.forward_credentials:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "credential forwarding with redirects enabled is forbidden",
            )
        if not (0 < self.max_request_bytes <= _MAX_REQUEST_BYTES_CAP):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"max_request_bytes must be in 1..{_MAX_REQUEST_BYTES_CAP}",
            )
        if self.account is not None:
            require_id(self.account, "account")
        if self.processor is not None:
            require_id(self.processor, "processor")

    @property
    def is_loopback(self) -> bool:
        return bool(getattr(self, "_loopback", False))

    @property
    def consent_processor(self) -> str:
        """The processor identity consent rows are keyed under."""
        return self.processor or self.endpoint_id


# ---------------------------------------------------------------------------
# Permit states (dispatch_permits.state)
# ---------------------------------------------------------------------------

PERMIT_OPEN = "open"
PERMIT_DISPATCHED = "dispatched"
PERMIT_EXPIRED = "expired"
PERMIT_DENIED = "denied"
PERMIT_RECONCILED = "reconciled"

# reconcile() outcomes → budget_ledger transitions.
_OUTCOME_ALIASES = {
    "settled": "settled",
    "completed": "settled",
    "ok": "settled",
    "overrun": "overrun",
    "expired": "expired",
    "timeout": "expired",
    "failed": "failed",
    "error": "failed",
}


class TransportBroker:
    """The single egress authority (V4-12.01).

    ``store`` is the profile store (``tx()``/``read()``); ``cfg`` supplies
    operating mode and the daily remote budget. ``endpoints`` is the
    allowlist of :class:`EndpointDescriptor` — an empty allowlist denies
    every dispatch (offline profiles ship no endpoints, which is the
    process-level network denial of V4-12.09 made enforceable: no
    descriptor, no permit, no dispatch).
    """

    #: V4-09.08: dispatch permits age out fast — re-issuance is cheap.
    DEFAULT_MAX_PERMIT_AGE_US = 1_000_000
    MAX_SCOPE_IDS = 64

    # Conservative fallback price (microUSD per 1K tokens) when the
    # configured model has no dated pricing row — deliberately above the
    # published embedding rates so an estimate never under-reserves.
    FALLBACK_MICROUSD_PER_1K = 1_000

    def __init__(
        self,
        store: Any,
        cfg: VerbatimConfig,
        *,
        endpoints: Iterable[Any] = (),
        gate: Optional[EgressGate] = None,
        consents: Any = None,
        policy_digest: Optional[str] = None,
        is_scope_suppressed: Optional[Any] = None,
        clock: Optional[Any] = None,
        max_permit_age_us: Optional[int] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._gate = gate or EgressGate(
            store,
            cfg,
            consents=consents,
            policy_digest=policy_digest,
            is_scope_suppressed=is_scope_suppressed,
        )
        self._clock = clock or now_us
        self._max_age = (
            self.DEFAULT_MAX_PERMIT_AGE_US
            if max_permit_age_us is None
            else int(max_permit_age_us)
        )
        if self._max_age <= 0:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "max_permit_age_us must be positive"
            )
        self._endpoints: dict[str, EndpointDescriptor] = {}
        for ep in endpoints or ():
            desc = ep if isinstance(ep, EndpointDescriptor) else EndpointDescriptor(**ep)
            if desc.endpoint_id in self._endpoints:
                raise VerbatimError(
                    ErrorCode.CONFIG_INVALID,
                    f"duplicate endpoint id {desc.endpoint_id!r}",
                )
            self._endpoints[desc.endpoint_id] = desc
        # reservation_id → [(disclosure_id, scope_id)] — in-process receipt
        # correlation; after a crash the disclosures rows themselves still
        # record that egress was authorized at outcome 'reserved'.
        self._receipts: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def gate(self) -> EgressGate:
        """The consent/budget delegate — exposed so hosts can wire ONE
        object for both the legacy gate API and the permit broker."""
        return self._gate

    def endpoint(self, endpoint_id: str) -> Optional[EndpointDescriptor]:
        return self._endpoints.get(endpoint_id)

    def _now(self) -> int:
        return int(self._clock())

    @staticmethod
    def _is_conn(obj: Any) -> bool:
        return isinstance(obj, sqlite3.Connection)

    def _in_tx(self, store_or_conn: Any, fn: Any) -> Any:
        """Run ``fn(conn)`` on the caller's connection, or inside this
        broker's own ``store.tx()`` when handed a store. Permit+reservation
        stay atomic with the caller's transaction when a conn is passed
        (coordinator integration); standalone callers get their own
        committed tx so the reservation is durable BEFORE any I/O."""
        if self._is_conn(store_or_conn):
            return fn(store_or_conn)
        if not hasattr(store_or_conn, "tx"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "open_dispatch needs a store (tx()) or a sqlite3.Connection",
            )
        with store_or_conn.tx() as conn:
            return fn(conn)

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view')"
                " AND name = ? LIMIT 1",
                (name,),
            ).fetchone()
            is not None
        )

    def _permits_table(self, conn: sqlite3.Connection) -> None:
        """Fail closed when the v4 permit ledger is absent — a dispatch that
        cannot be recorded is a dispatch that cannot happen."""
        if not self._table_exists(conn, "dispatch_permits"):
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "dispatch_permits schema unavailable; cannot mint a durable permit",
                retryable=False,
            )

    # ------------------------------------------------------------------
    # estimates
    # ------------------------------------------------------------------

    def estimate_tokens(self, texts: Iterable[str]) -> int:
        """Conservative token estimate (~4 chars/token, floor 1 per text)."""
        total = 0
        for t in texts:
            total += max(1, math.ceil(len(t) / 4))
        return total

    def estimate_spend(self, texts: Iterable[str]) -> float:
        """USD bound for an encode request — the priced rate when the
        configured model has dated metadata, else a conservative fallback
        (unknown pricing never estimates low, V4-12.04)."""
        from .egress import PRICING

        rate = PRICING.get(self._cfg.judge.model, self.FALLBACK_MICROUSD_PER_1K)
        micro = math.ceil(self.estimate_tokens(texts) * rate / 1000)
        return micro / 1_000_000.0

    # ------------------------------------------------------------------
    # open_dispatch — permit issuance (V4-12.02/05/06)
    # ------------------------------------------------------------------

    def open_dispatch(
        self,
        store_or_conn: Any,
        *,
        caller: str,
        recipient: str,
        purpose: str,
        payload_digest: str,
        scope_ids: Iterable[str],
        consent_refs: Iterable[str] = (),
        max_spend: float,
        deadline_us: Optional[int] = None,
        est_tokens: int = 0,
        job_id: Optional[str] = None,
        input_refs: Any = None,
    ) -> Any:
        """Mint a :class:`DispatchPermit` + budget reservation atomically.

        ``store_or_conn`` is either the store (broker opens its own
        committed ``tx()``) or a ``sqlite3.Connection`` inside the caller's
        transaction (coordinator integration). Either way the reservation
        is durable before the caller reaches any transport.

        Raises ``VerbatimError`` (EGRESS_DENIED / BUDGET_EXHAUSTED /
        DEADLINE_EXCEEDED / VALIDATION) on any refusal — never returns a
        degraded permit.
        """
        from ..core.types_v4 import DispatchPermit

        if not isinstance(caller, str) or not caller:
            raise VerbatimError(ErrorCode.VALIDATION, "caller identity required")
        if not isinstance(recipient, str) or not recipient:
            raise VerbatimError(ErrorCode.VALIDATION, "recipient required")
        if not isinstance(purpose, str) or not purpose:
            raise VerbatimError(ErrorCode.VALIDATION, "purpose required")
        digest = _check_digest(payload_digest)
        sids = tuple(dict.fromkeys(scope_ids or ()))
        if not sids:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED, "dispatch needs >=1 contributing scope"
            )
        if len(sids) > self.MAX_SCOPE_IDS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"scope_ids exceed {self.MAX_SCOPE_IDS} bound",
            )
        for sid in sids:
            require_id(sid, "scope_id")
        refs = tuple(consent_refs or ())
        for r in refs:
            if not isinstance(r, str) or not r:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "consent_refs must be non-empty strings"
                )
        try:
            spend = float(max_spend)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, "max_spend must be a number"
            ) from exc
        if not math.isfinite(spend) or spend <= 0.0:
            # V4-12.04: every dispatch carries a positive worst-case
            # reservation; "nominally free" is not a permit class.
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "dispatch requires a positive max_spend bound — nominally "
                "free requests are still dispatches",
                retryable=False,
            )
        reserved_microusd = math.ceil(spend * 1_000_000)
        if est_tokens < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "est_tokens must be >= 0")

        desc = self._endpoints.get(recipient)
        if desc is None:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                f"recipient {recipient!r} is not an allowlisted endpoint",
                retryable=False,
            )
        remote_ok = self._cfg.mode in (Mode.REMOTE_ASSISTED, Mode.JEV_ASSISTED)
        service_ok = remote_ok or self._cfg.mode == Mode.LOCAL_SERVICE
        if desc.is_loopback:
            if not service_ok:
                raise VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    "operating mode permits no transport dispatch",
                    retryable=False,
                )
        elif not remote_ok:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "operating mode does not permit remote egress",
                retryable=False,
            )

        now = self._now()
        if deadline_us is not None and now >= deadline_us:
            raise VerbatimError(
                ErrorCode.DEADLINE_EXCEEDED, "dispatch deadline already passed"
            )
        expires = now + self._max_age
        if deadline_us is not None:
            expires = min(expires, int(deadline_us))
        if expires <= now:
            raise VerbatimError(
                ErrorCode.DEADLINE_EXCEEDED, "permit lifetime is empty"
            )

        def _open(conn: sqlite3.Connection) -> DispatchPermit:
            self._permits_table(conn)
            processor = desc.consent_processor
            verified_consents: list[str] = []
            for sid in sids:
                # V4-12.06: consent for EVERY contributing scope — no
                # host-default-scope substitution.
                if self._gate.scope_suppressed(sid):
                    raise VerbatimError(
                        ErrorCode.EGRESS_DENIED,
                        "scope has a live purge suppression; egress paused",
                        retryable=True,
                    )
                row = self._gate.consent_row(sid, processor, purpose)
                if row is None or not self._gate.consent_epoch_ok(row):
                    raise VerbatimError(
                        ErrorCode.EGRESS_DENIED,
                        f"no active consent for scope {sid!r} processor "
                        f"{processor!r} purpose {purpose!r}",
                        retryable=False,
                    )
                cid = row.get("consent_id") if isinstance(row, dict) else getattr(
                    row, "consent_id", None
                )
                verified_consents.append(str(cid) if cid else f"{sid}:{processor}:{purpose}")
            # Caller-declared consent refs must be backed by verified rows —
            # a caller may narrow to a subset, never assert uncovered consent.
            unknown = [r for r in refs if r not in verified_consents]
            if unknown:
                raise VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    f"consent_refs {unknown!r} do not cover this dispatch",
                    retryable=False,
                )

            daily = self._gate.budget_microusd()
            if daily <= 0:
                raise VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    "zero remote budget prohibits dispatch (V4-12.04)",
                    retryable=False,
                )
            day = _utc_day()
            spent = int(
                conn.execute(
                    "SELECT COALESCE(SUM(COALESCE(actual_cost_microusd,"
                    " reserved_cost_microusd)), 0)"
                    " FROM budget_ledger WHERE day_utc = ?",
                    (day,),
                ).fetchone()[0]
            )
            if spent + reserved_microusd > daily:
                raise VerbatimError(
                    ErrorCode.BUDGET_EXHAUSTED,
                    "daily egress budget lacks room for this reservation",
                    retryable=False,
                )

            permit_id = new_id()
            reservation_id = new_id()
            conn.execute(
                "INSERT INTO budget_ledger (reservation_id, job_id, day_utc,"
                " token_bound, reserved_cost_microusd, state)"
                " VALUES (?,?,?,?,?,'reserved')",
                (reservation_id, job_id, day, int(est_tokens), reserved_microusd),
            )
            repos_v4.insert(
                conn,
                "dispatch_permits",
                {
                    "permit_id": permit_id,
                    "recipient": recipient,
                    "purpose": purpose,
                    "payload_digest": digest,
                    "scope_ids_json": list(sids),
                    "consent_refs_json": sorted(set(verified_consents)),
                    "reservation_id": reservation_id,
                    "max_spend": spend,
                    "issued_us": now,
                    "expires_us": expires,
                    "state": PERMIT_OPEN,
                },
            )
            # V4-12.10 receipt: content-minimized — permit id, caller tag,
            # digest and input refs, never payload text.
            receipt_refs = [f"dispatch_permit:{permit_id}", f"caller:{caller}"]
            receipt_refs.extend(self._gate.normalize_refs(input_refs))
            dids: list[tuple[str, str]] = []
            for sid in sids:
                did = self._gate.record_disclosure(
                    conn, sid, processor, purpose, receipt_refs
                )
                if did:
                    dids.append((did, sid))
            if dids:
                self._receipts[reservation_id] = dids
                while len(self._receipts) > 4096:
                    self._receipts.pop(next(iter(self._receipts)))

            return DispatchPermit(
                permit_id=permit_id,
                recipient=recipient,
                purpose=purpose,
                payload_digest=digest,
                scope_ids=sids,
                consent_refs=tuple(sorted(set(verified_consents))),
                reservation_id=reservation_id,
                max_spend=spend,
                issued_us=now,
                expires_us=expires,
                state=PERMIT_OPEN,
            )

        return self._in_tx(store_or_conn, _open)

    # ------------------------------------------------------------------
    # dispatch — the one-use gate the transport executes through (V4-12.03)
    # ------------------------------------------------------------------

    def _permit_row(self, conn: sqlite3.Connection, permit_id: str) -> dict[str, Any]:
        row = repos_v4.get(conn, "dispatch_permits", {"permit_id": permit_id})
        if row is None:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED, "unknown dispatch permit", retryable=False
            )
        return row

    def _expire_reservation(self, conn: sqlite3.Connection, reservation_id: str) -> None:
        """Charge a still-open reservation at its full estimate — the
        conservative timeout accounting of V4-12.05."""
        cur = conn.execute(
            "UPDATE budget_ledger SET actual_cost_microusd = reserved_cost_microusd,"
            " state = 'expired' WHERE reservation_id = ? AND state = 'reserved'",
            (reservation_id,),
        )
        if cur.rowcount:
            self._close_receipts(conn, reservation_id, "expired")

    def _deny(
        self, conn: sqlite3.Connection, row: dict[str, Any], msg: str
    ) -> VerbatimError:
        """Mark the permit denied and release its reservation at zero actual
        spend — a denied dispatch provably sent nothing, so the honest
        settlement is 0 (the permit, not the budget, carries the denial).

        Returns the error instead of raising so the caller can commit the
        denial inside the same transaction and raise AFTER commit — a
        denial that rolled back would leave the permit looking 'open'.
        """
        conn.execute(
            "UPDATE dispatch_permits SET state = ? WHERE permit_id = ?",
            (PERMIT_DENIED, row["permit_id"]),
        )
        conn.execute(
            "UPDATE budget_ledger SET actual_cost_microusd = 0, state = 'settled'"
            " WHERE reservation_id = ? AND state = 'reserved'",
            (row["reservation_id"],),
        )
        self._close_receipts(conn, row["reservation_id"], "denied")
        return VerbatimError(ErrorCode.EGRESS_DENIED, msg, retryable=False)

    def _recheck(
        self, conn: sqlite3.Connection, row: dict[str, Any]
    ) -> Optional[VerbatimError]:
        """V4-12.03: consent, suppression, and reservation availability are
        re-verified inside the dispatch transaction — immediately before
        ownership transfers to transport."""
        sids = safe_json_loads(row["scope_ids_json"])
        if not isinstance(sids, list) or not sids:
            # Corrupted scope coverage must fail closed — an empty set can
            # never satisfy V4-12.06's per-scope consent requirement.
            return self._deny(conn, row, "permit scope coverage unreadable")
        purpose = row["purpose"]
        desc = self._endpoints.get(row["recipient"])
        if desc is None:
            return self._deny(conn, row, "permit recipient no longer allowlisted")
        processor = desc.consent_processor
        for sid in sids:
            if self._gate.scope_suppressed(sid):
                return self._deny(
                    conn, row, "scope suppressed between permit and dispatch"
                )
            crow = self._gate.consent_row(sid, processor, purpose)
            if crow is None or not self._gate.consent_epoch_ok(crow):
                return self._deny(conn, row, "consent revoked or stale at dispatch")
        res = conn.execute(
            "SELECT state FROM budget_ledger WHERE reservation_id = ?",
            (row["reservation_id"],),
        ).fetchone()
        if res is None or res[0] != "reserved":
            return self._deny(
                conn, row, "budget reservation unavailable at dispatch"
            )
        return None

    def dispatch(
        self,
        permit: Any,
        *,
        recipient: Optional[str] = None,
        payload_digest: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Any:
        """Validate a permit for THIS transport call and atomically consume
        it (open → dispatched). The encoder calls this immediately before
        its HTTP request; a second call with the same permit is a replay
        and denied (C18).

        ``recipient``/``payload_digest`` are the transport's OWN identity
        and the digest of the exact serialized body — a permit minted for
        another endpoint or another payload cannot dispatch here.
        """
        from ..core.types_v4 import DispatchPermit

        if not isinstance(permit, DispatchPermit):
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "remote dispatch requires a TransportBroker-issued DispatchPermit",
                retryable=False,
            )
        if recipient is not None and permit.recipient != recipient:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "permit recipient does not match this transport endpoint",
                retryable=False,
            )
        if recipient is not None and recipient not in self._endpoints:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                f"recipient {recipient!r} is not allowlisted",
                retryable=False,
            )
        if payload_digest is not None and permit.payload_digest != payload_digest:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "permit payload_digest does not cover this request payload",
                retryable=False,
            )

        def _do(
            c: sqlite3.Connection,
        ) -> tuple[Optional[DispatchPermit], Optional[VerbatimError]]:
            self._permits_table(c)
            row = self._permit_row(c, permit.permit_id)
            now = self._now()
            if row["state"] != PERMIT_OPEN:
                return None, VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    f"permit state {row['state']!r} is not dispatchable "
                    "(replay/one-use violation)",
                    retryable=False,
                )
            if now >= int(row["expires_us"]):
                c.execute(
                    "UPDATE dispatch_permits SET state = ? WHERE permit_id = ?",
                    (PERMIT_EXPIRED, row["permit_id"]),
                )
                self._expire_reservation(c, row["reservation_id"])
                return None, VerbatimError(
                    ErrorCode.PERMIT_EXPIRED,
                    "dispatch permit expired before transport handoff",
                    retryable=False,
                )
            err = self._recheck(c, row)
            if err is not None:
                return None, err
            cur = c.execute(
                "UPDATE dispatch_permits SET state = ?"
                " WHERE permit_id = ? AND state = ?",
                (PERMIT_DISPATCHED, row["permit_id"], PERMIT_OPEN),
            )
            if cur.rowcount != 1:
                return None, VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    "permit consumed concurrently — replay denied",
                    retryable=False,
                )
            return self._permit_to_dataclass(
                {**row, "state": PERMIT_DISPATCHED}
            ), None

        if conn is not None:
            # Caller's transaction: mutations are theirs to commit or roll
            # back; the error surfaces either way.
            result, err = _do(conn)
        else:
            with self._store.tx() as c:
                result, err = _do(c)
        if err is not None:
            raise err
        return result

    def mark_dispatched(self, permit_id: str) -> Any:
        """Low-level one-use flip by permit id (C18).

        Equivalent to ``dispatch`` minus the recipient/digest binding —
        intended for transports that already verified binding through
        ``dispatch``'s contract or for tests exercising the state machine.
        """
        row = self._permit_lookup(permit_id)
        permit = self._permit_to_dataclass(row)
        return self.dispatch(permit)

    def _permit_lookup(self, permit_id: str) -> dict[str, Any]:
        with self._store.read() as conn:
            row = repos_v4.get(conn, "dispatch_permits", {"permit_id": permit_id})
        if row is None:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED, "unknown dispatch permit", retryable=False
            )
        return row

    @staticmethod
    def _permit_to_dataclass(row: dict[str, Any]) -> Any:
        from ..core.types_v4 import DispatchPermit

        return DispatchPermit(
            permit_id=row["permit_id"],
            recipient=row["recipient"],
            purpose=row["purpose"],
            payload_digest=row["payload_digest"],
            scope_ids=tuple(safe_json_loads(row["scope_ids_json"]) or ()),
            consent_refs=tuple(safe_json_loads(row["consent_refs_json"]) or ()),
            reservation_id=row["reservation_id"],
            max_spend=float(row["max_spend"]),
            issued_us=int(row["issued_us"]),
            expires_us=int(row["expires_us"]),
            state=row["state"],
        )

    def permit(self, permit_id: str) -> Any:
        """Durable read view of a permit (tests, status, reconciliation)."""
        return self._permit_to_dataclass(self._permit_lookup(permit_id))

    # ------------------------------------------------------------------
    # reconcile — permit→request correlation + settlement (V4-12.05/10)
    # ------------------------------------------------------------------

    def _close_receipts(
        self, conn: sqlite3.Connection, reservation_id: str, outcome: str
    ) -> None:
        """Best-effort disclosure receipt outcome; the ledger stays
        authoritative (same fail-soft rationale as EgressGate)."""
        recs = self._receipts.pop(reservation_id, None)
        if not recs:
            return
        try:
            if self._table_exists(conn, "disclosures"):
                for did, _sid in recs:
                    conn.execute(
                        "UPDATE disclosures SET outcome = ? WHERE disclosure_id = ?",
                        (outcome, did),
                    )
        except Exception:
            pass

    def reconcile(
        self,
        reservation_id: str,
        actual_spend: float,
        outcome: str,
    ) -> str:
        """Settle a reservation and close the permit correlation.

        ``outcome`` ∈ {settled|completed|ok, overrun, expired|timeout,
        failed|error}:

        - ``settled``/``overrun`` — actual spend is recorded; above the
          reservation it lands as ``overrun`` (visible, never concealed).
        - ``expired``/``timeout`` — the provider may have done the work:
          the FULL reservation is charged (V4-12.05 conservative retention).
        - ``failed`` — the transport provably never executed: settles at
          ``actual_spend`` (typically 0). When unsure, report ``expired``.

        Idempotent: an already-reconciled reservation returns its state —
        never double-debits.
        """
        outcome_norm = _OUTCOME_ALIASES.get(str(outcome))
        if outcome_norm is None:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown reconcile outcome {outcome!r}"
            )
        try:
            actual = float(actual_spend)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, "actual_spend must be a number"
            ) from exc
        if not math.isfinite(actual) or actual < 0.0:
            raise VerbatimError(
                ErrorCode.VALIDATION, "actual_spend must be >= 0"
            )
        actual_microusd = math.ceil(actual * 1_000_000)

        with self._store.tx() as conn:
            row = conn.execute(
                "SELECT reservation_id, reserved_cost_microusd, state"
                " FROM budget_ledger WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown budget reservation"
                )
            res_id, reserved, state = row
            if state != "reserved":
                return state  # already reconciled — never double-debit
            if outcome_norm == "expired":
                new_state, debit = "expired", int(reserved)
            elif outcome_norm == "failed":
                new_state, debit = "settled", actual_microusd
            else:
                debit = actual_microusd
                new_state = "overrun" if debit > int(reserved) else "settled"
            conn.execute(
                "UPDATE budget_ledger SET actual_cost_microusd = ?, state = ?"
                " WHERE reservation_id = ?",
                (debit, new_state, res_id),
            )
            # Close permit correlation: a dispatched permit reconciles; an
            # undispatched 'open' permit on a settled/expired reservation is
            # no longer usable — expire it so it can't dispatch against a
            # closed reservation.
            permits = repos_v4.query(
                conn, "dispatch_permits", {"reservation_id": res_id}
            )
            for p in permits:
                if p["state"] == PERMIT_DISPATCHED:
                    target = (
                        PERMIT_EXPIRED if outcome_norm == "expired" else PERMIT_RECONCILED
                    )
                elif p["state"] == PERMIT_OPEN:
                    target = PERMIT_EXPIRED
                else:
                    continue
                conn.execute(
                    "UPDATE dispatch_permits SET state = ? WHERE permit_id = ?",
                    (target, p["permit_id"]),
                )
            self._close_receipts(conn, res_id, new_state)
            return new_state

    # ------------------------------------------------------------------
    # crash recovery — conservative retention sweep (V4-12.05)
    # ------------------------------------------------------------------

    def expire_stale(self, now: Optional[int] = None) -> int:
        """Expire every ``open`` permit past its deadline and charge its
        reservation at the full estimate. Permits that were consumed
        (dispatched) are untouched — their spend belongs to ``reconcile``.
        Returns the number of permits expired."""
        ts = self._now() if now is None else int(now)
        with self._store.tx() as conn:
            if not self._table_exists(conn, "dispatch_permits"):
                return 0
            rows = repos_v4.query(conn, "dispatch_permits", {"state": PERMIT_OPEN})
            n = 0
            for row in rows:
                if ts >= int(row["expires_us"]):
                    conn.execute(
                        "UPDATE dispatch_permits SET state = ? WHERE permit_id = ?",
                        (PERMIT_EXPIRED, row["permit_id"]),
                    )
                    self._expire_reservation(conn, row["reservation_id"])
                    n += 1
            return n

    # ------------------------------------------------------------------
    # encoder seam — one-call permitted encode (F4-04 integration helper)
    # ------------------------------------------------------------------

    def encode_permitted(
        self,
        encoder: Any,
        texts: list[str],
        *,
        scope_ids: Iterable[str],
        purpose: str,
        caller: str,
        deadline_us: Optional[int] = None,
        job_id: Optional[str] = None,
        input_refs: Any = None,
    ) -> list[bytes]:
        """Encode ``texts`` under a fresh dispatch permit.

        Local encoders (``requires_transport_permit`` absent/False — hashing,
        artifact) run directly: they perform no transport I/O. Remote
        encoders get a minted permit whose ``payload_digest`` covers the
        exact request body; the encoder re-verifies it inside ``encode``
        before any socket work, so an unwired call site can never bypass
        the broker — ``encode`` without a permit self-denies.
        """
        if not getattr(encoder, "requires_transport_permit", False):
            return encoder.encode(texts)
        if not texts:
            # Encoders return [] before touching the permit — never mint a
            # reservation for a request that cannot dispatch.
            return []
        digest = encoder.payload_digest(texts)
        est_tokens = self.estimate_tokens(texts)
        permit = self.open_dispatch(
            self._store,
            caller=caller,
            recipient=encoder.endpoint_id,
            purpose=purpose,
            payload_digest=digest,
            scope_ids=scope_ids,
            max_spend=self.estimate_spend(texts),
            deadline_us=deadline_us,
            est_tokens=est_tokens,
            job_id=job_id,
            input_refs=input_refs,
        )
        try:
            blobs = encoder.encode(texts, permit=permit)
        except VerbatimError as exc:
            # Retryable transport failure → the provider may have run the
            # request: expire at full reservation (conservative). Definitive
            # pre-I/O failure settles at zero.
            self.reconcile(
                permit.reservation_id,
                0.0,
                "expired" if exc.retryable else "failed",
            )
            raise
        except Exception:
            self.reconcile(permit.reservation_id, 0.0, "expired")
            raise
        # No usage meter on the embedding wire API — settle at the
        # conservative reserved bound (visible, never under-counted).
        self.reconcile(permit.reservation_id, permit.max_spend, "settled")
        return blobs


__all__ = [
    "EndpointDescriptor",
    "PERMIT_DENIED",
    "PERMIT_DISPATCHED",
    "PERMIT_EXPIRED",
    "PERMIT_OPEN",
    "PERMIT_RECONCILED",
    "TransportBroker",
    "payload_digest",
]
