"""Egress gate: consent, budget, and minimization for remote dispatch (SPEC §27).

Nothing leaves the process unless ALL of these hold, checked at dispatch time
(never assumed from a stale snapshot):

* the operating mode permits remote work (``jev_assisted``);
* an active, unrevoked consent row covers (scope, processor, purpose);
* a positive operator-set daily budget has room after today's spend.

Budget is tracked in ``budget_ledger`` as micro-USD reservations taken
*before* dispatch (SPEC §27 "reservations occur atomically before dispatch").
A timeout leaves the reservation standing — the provider may have done the
work — until reconciled at full estimated cost via ``expire``. Actual usage
larger than the reservation debits the actual amount and marks the row
``overrun`` so the overspend is visible, not concealed.

Price metadata is a dated configuration input, not a correctness invariant:
PRICING maps model revision → microUSD per 1K tokens. An unknown model has
no trustworthy price, so pricing it fails closed with CONFIG_INVALID.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, Union

from ..config import VerbatimConfig
from ..core.types import (
    ErrorCode,
    Mode,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
)

# Dated pricing input (microUSD per 1K tokens), per SPEC §27's rule that
# prices are configuration data — pinned here only as the v1 documented
# value; deployments SHOULD override via reviewed config when terms change.
PRICING: dict[str, int] = {"jev-1.13.0": 42}

_LEDGER_SPEND_SQL = (
    "SELECT COALESCE(SUM(COALESCE(actual_cost_microusd, reserved_cost_microusd)), 0)"
    " FROM budget_ledger WHERE day_utc = ?"
)

# State is what travels on the wire: cap it far below the provider's
# documented limits so a single snapshot can never crowd out the questions.
MAX_STATE_BYTES = 16 * 1024

# --- best-effort secret/PII patterns for minimize() -------------------------
# Heuristic scrubbing, explicitly NOT a guarantee (SPEC §27: minimization is
# selection of authorized fields first; patterns are a second line).
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+\d[\d\s().-]{7,}\d|\d(?:[\d\s().-]{8,})\d)(?!\d)")
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]+-----.*?-----END [A-Z0-9 ]+-----", re.DOTALL
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_APIKEY_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|api[_-]?token|secret|password|access[_-]?token)"
    r"\s*[=:]\s*['\"]?[a-z0-9._~+/=-]{8,}"
)
_SK_RE = re.compile(r"\b(?:sk|pk|ak|xox[baprs])-[a-z0-9-]{10,}\b", re.IGNORECASE)
_AWS_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
REDACTED = "[REDACTED]"


def scope_key(scope: Union[Scope, str]) -> str:
    """Canonical ``scope_id`` rendering — delegates to
    ``core.identity.scope_key`` so consent lookups hit the same ids the CLI
    grants under. Raw strings pass through identifier validation.
    """
    if isinstance(scope, Scope):
        from ..core.identity import scope_key as _canonical

        return _canonical(scope)
    from ..core.types import require_id

    return require_id(scope, "scope_id")


def _utc_day(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(timezone.utc)
    return dt.date().isoformat()


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    """Row → dict via cursor description (store sets no row_factory)."""
    cols = [d[0] for d in cur.description]
    r = cur.fetchone()
    return dict(zip(cols, r)) if r is not None else None


class EgressGate:
    """Consent + budget gate for remote dispatches.

    ``store`` provides ``tx()``/``read()`` connection scopes. ``consents``
    optionally injects a consent lookup — either a callable
    ``f(scope_id, processor, purpose) -> row|None`` or an object exposing
    ``.active(scope_id, processor, purpose)`` (e.g. the parallel
    ConsentsRepo). When omitted, the repo is imported lazily and a direct
    consents-table SELECT is used as the documented fallback.

    Every dispatch rechecks consent epoch, purge suppression, and budget
    (SPEC §27). ``policy_digest`` binds the gate to a consent epoch: when
    set, only consent rows carrying the same digest authorize — a consent
    granted under an older policy cannot silently carry over. Source-
    revision freshness is the caller's responsibility (the gate sees no
    evidence objects). ``is_scope_suppressed`` defaults to a purges-table
    lookup that blocks scopes with a live suppression tombstone; inject a
    callable ``f(scope_id) -> bool`` to override. Suppression-check
    failures fail closed.
    """

    # In-flight purge states that pause new remote dispatch of
    # scope-derived data. 'previewed' is not yet an erasure order, and
    # 'completed' leaves the *objects* tombstoned but the scope clean —
    # it must not permanently disable egress for surviving content.
    _BLOCKING_PURGE_STATES = ("suppressed", "purging")

    def __init__(
        self,
        store: Any,
        cfg: VerbatimConfig,
        consents: Any = None,
        *,
        policy_digest: Optional[str] = None,
        is_scope_suppressed: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._consents = consents
        self._policy_digest = policy_digest
        self._suppressed = is_scope_suppressed or self._scope_suppressed
        # reservation_id → (disclosure_id, scope_id, input_refs): correlates
        # a budget reservation with its content-minimized receipt so settle/
        # expire can close the receipt's outcome. In-process only — after a
        # restart the receipt stays at outcome 'reserved', which still
        # accurately records that egress was authorized.
        self._receipts: dict[str, tuple[str, str, list[str]]] = {}

    def _scope_suppressed(self, scope_id: str) -> bool:
        """True while the scope has a live suppression tombstone.

        Fail-closed: a lookup error is treated as suppressed — a broken
        erasure ledger pauses disclosure rather than silently opening it.
        """
        try:
            states = ",".join("?" for _ in self._BLOCKING_PURGE_STATES)
            with self._store.read() as conn:
                row = conn.execute(
                    "SELECT 1 FROM purges WHERE scope_id = ?"
                    f" AND state IN ({states}) LIMIT 1",
                    (scope_id, *self._BLOCKING_PURGE_STATES),
                ).fetchone()
            return row is not None
        except Exception:
            return True

    def _suppressed_check(self, scope_id: str) -> bool:
        """Wraps the (possibly injected) suppression predicate.

        Any exception — including from a custom callable — is treated as
        suppressed: a broken erasure ledger must pause disclosure, never
        silently open it.
        """
        try:
            return bool(self._suppressed(scope_id))
        except Exception:
            return True

    def _consent_epoch_ok(self, row: Any) -> bool:
        """Consent epoch check: the row must carry the configured digest."""
        if row is None:
            return False
        if self._policy_digest is None:
            return True
        if isinstance(row, dict):
            return row.get("policy_digest") == self._policy_digest
        return getattr(row, "policy_digest", None) == self._policy_digest

    # ------------------------------------------------------------------
    # consent
    # ------------------------------------------------------------------

    def _consent_row(self, scope_id: str, processor: str, purpose: str) -> Any:
        c = self._consents
        if c is not None:
            if hasattr(c, "active"):
                return c.active(scope_id, processor, purpose)
            return c(scope_id, processor, purpose)
        try:
            from ..storage.repos import ConsentsRepo  # built in parallel
        except ImportError:
            ConsentsRepo = None
        if ConsentsRepo is not None:
            return ConsentsRepo(self._store).active(scope_id, processor, purpose)
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM consents WHERE scope_id = ? AND processor = ?"
                    " AND purpose = ? AND revoked_us IS NULL"
                    " ORDER BY granted_us DESC LIMIT 1",
                    (scope_id, processor, purpose),
                )
            )

    def _consent_active(self, scope_id: str, processor: str, purpose: str) -> bool:
        return self._consent_epoch_ok(
            self._consent_row(scope_id, processor, purpose)
        )

    # ------------------------------------------------------------------
    # broker delegate surface (SPEC_V4 §12 / F4-04)
    #
    # ``privacy.broker.TransportBroker`` owns dispatch permits and their
    # reservations; it delegates the consent predicate, epoch binding,
    # suppression check, budget ceiling, and disclosure-receipt writes to
    # this gate so both paths share ONE implementation. These public names
    # are the supported seam — they alias the private internals above.
    # ------------------------------------------------------------------

    def consent_row(self, scope_id: str, processor: str, purpose: str) -> Any:
        """Active consent row for (scope, processor, purpose) or None."""
        return self._consent_row(scope_id, processor, purpose)

    def consent_epoch_ok(self, row: Any) -> bool:
        """Consent row carries the configured policy digest (or none set)."""
        return self._consent_epoch_ok(row)

    def scope_suppressed(self, scope_id: str) -> bool:
        """Fail-closed purge-suppression predicate."""
        return self._suppressed_check(scope_id)

    def budget_microusd(self) -> int:
        """Configured daily remote budget ceiling in microUSD."""
        return self._budget_microusd()

    def normalize_refs(self, input_refs: Any) -> list[str]:
        """Bounded ``kind:id`` receipt references (validation shared)."""
        return self._normalize_refs(input_refs)

    def record_disclosure(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        processor: str,
        purpose: str,
        refs: list[str],
    ) -> Optional[str]:
        """Write a content-minimized disclosure receipt inside the caller's
        transaction (no-op when the disclosures table is absent)."""
        return self._record_disclosure(conn, scope_id, processor, purpose, refs)

    # ------------------------------------------------------------------
    # pricing / spend
    # ------------------------------------------------------------------

    def _price_per_1k(self) -> int:
        model = self._cfg.judge.model
        price = PRICING.get(model)
        if price is None:
            # No dated price for this revision: pricing cannot be trusted,
            # so the safe answer is to refuse rather than under-reserve.
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"no dated price metadata for model {model!r}",
            )
        return price

    def _cost_microusd(self, tokens: int) -> int:
        if tokens < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "token count must be >= 0")
        return math.ceil(tokens * self._price_per_1k() / 1000)

    def _budget_microusd(self) -> int:
        return int((self._cfg.judge.daily_budget_usd * Decimal(1_000_000)).to_integral_value())

    def day_spend(self, scope: Optional[Union[Scope, str]] = None, day: Optional[str] = None) -> int:
        """microUSD committed today (reserved + settled + expired + overrun).

        Schema v1's ledger carries no scope column, so a scope filter can
        only apply through the optional ``job_id`` back-reference; rows
        without a job are attributed to the ledger as a whole (conservative:
        per-process budgets err toward counting more, not less).
        """
        d = day or _utc_day()
        with self._store.read() as conn:
            if scope is None:
                return int(conn.execute(_LEDGER_SPEND_SQL, (d,)).fetchone()[0])
            sid = scope_key(scope)
            return int(
                conn.execute(
                    "SELECT COALESCE(SUM(COALESCE(b.actual_cost_microusd,"
                    " b.reserved_cost_microusd)), 0) FROM budget_ledger b"
                    " JOIN jobs j ON j.job_id = b.job_id"
                    " WHERE b.day_utc = ? AND j.scope_id = ?",
                    (d, sid),
                ).fetchone()[0]
            )

    # ------------------------------------------------------------------
    # gate API
    # ------------------------------------------------------------------

    def check_only(self, scope: Union[Scope, str], processor: str, purpose: str) -> bool:
        """Whether a dispatch would be permitted right now (no reservation)."""
        try:
            sid = scope_key(scope)
            if self._cfg.mode not in (Mode.REMOTE_ASSISTED, Mode.JEV_ASSISTED):
                return False
            budget = self._budget_microusd()
            if budget <= 0:
                return False
            if self._suppressed_check(sid):
                return False
            if not self._consent_active(sid, processor, purpose):
                return False
            return self.day_spend() < budget
        except Exception:
            # A preflight probe must fail closed: a broken ledger or consent
            # table means "not permitted", not an unhandled crash.
            return False

    # ------------------------------------------------------------------
    # disclosure receipts (SPEC_V2 §36): every authorized egress leaves a
    # content-minimized receipt — processor, purpose, input reference ids,
    # never payload text.
    # ------------------------------------------------------------------

    _MAX_RECEIPT_REFS = 512

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view')"
                " AND name = ? LIMIT 1",
                (name,),
            ).fetchone()
            is not None
        )

    def _normalize_refs(self, input_refs: Any) -> list[str]:
        """Input references → bounded ``kind:id`` strings.

        Accepts strings (``"claim:<id>"`` or bare ids) and dicts with
        ``object_kind``/``object_id`` keys; anything else is a validation
        failure rather than a silently mangled audit record.
        """
        refs: list[str] = []
        for item in input_refs or ():
            if isinstance(item, dict):
                kind = item.get("object_kind") or item.get("kind")
                oid = item.get("object_id") or item.get("id")
                if not kind or not oid:
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        "input ref dicts need object_kind/object_id",
                    )
                ref = f"{kind}:{oid}"
            elif isinstance(item, str):
                ref = item
            else:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "input refs must be strings or dicts"
                )
            refs.append(ref[:256])
        if len(refs) > self._MAX_RECEIPT_REFS:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"input_refs exceed {self._MAX_RECEIPT_REFS} bound",
            )
        return refs

    def _record_disclosure(
        self,
        conn: sqlite3.Connection,
        sid: str,
        processor: str,
        purpose: str,
        refs: list[str],
    ) -> Optional[str]:
        """Receipt row inside the authorize transaction — atomic with the
        reservation, so an unrecorded dispatch cannot happen."""
        if not self._table_exists(conn, "disclosures"):
            return None
        from ..storage.repos_v2 import DisclosuresRepo

        return DisclosuresRepo(self._store).record(
            conn,
            sid,
            processor,
            purpose,
            input_refs=list(refs),
            outcome="reserved",
        )

    def _close_disclosure(
        self,
        conn: sqlite3.Connection,
        reservation_id: str,
        outcome: str,
        usage: dict[str, Any],
    ) -> None:
        """Best-effort receipt outcome + usage bumps.

        The reservation ledger is authoritative; a failure to annotate the
        receipt must not roll back a settle/expire (that would risk a
        double-debit), so this path fails soft after the ledger write.
        """
        rec = self._receipts.pop(reservation_id, None)
        if rec is None:
            return
        did, sid, refs = rec
        try:
            if self._table_exists(conn, "disclosures"):
                conn.execute(
                    "UPDATE disclosures SET outcome = ?, usage_json = ?"
                    " WHERE disclosure_id = ?",
                    (outcome, json_dumps(usage), did),
                )
            if self._table_exists(conn, "usage_aggregates"):
                from ..storage.repos_v2 import UsageRepo

                usage_repo = UsageRepo(self._store)
                for ref in refs:
                    kind, _, oid = ref.partition(":")
                    if kind and oid:
                        usage_repo.bump(conn, sid, kind, oid, "egress")
        except Exception:
            pass

    def authorize(
        self,
        scope: Union[Scope, str],
        processor: str,
        purpose: str,
        est_tokens: int,
        *,
        job_id: Optional[str] = None,
        input_refs: Any = None,
    ) -> str:
        """Reserve estimated spend and return the reservation_id.

        Atomic inside ``store.tx()``: consent is rechecked and the day's
        committed spend (reservations + settled + expired estimates) must
        leave room for this reservation before it is written. Failures map
        to EGRESS_DISABLED (mode/consent) or BUDGET_EXHAUSTED (no room).
        """
        sid = scope_key(scope)
        if self._cfg.mode not in (Mode.REMOTE_ASSISTED, Mode.JEV_ASSISTED):
            raise VerbatimError(
                ErrorCode.EGRESS_DISABLED, "mode does not permit remote egress"
            )
        cost = self._cost_microusd(est_tokens)
        day = _utc_day()
        with self._store.tx() as conn:
            if self._suppressed_check(sid):
                raise VerbatimError(
                    ErrorCode.EGRESS_DISABLED,
                    "scope has a live purge suppression; egress paused",
                    retryable=True,
                )
            row = self._consent_row(sid, processor, purpose)
            if row is None:
                raise VerbatimError(
                    ErrorCode.EGRESS_DISABLED,
                    f"no active consent for processor {processor!r} purpose {purpose!r}",
                )
            if not self._consent_epoch_ok(row):
                raise VerbatimError(
                    ErrorCode.EGRESS_DISABLED,
                    "consent predates current policy digest; re-grant required",
                )
            budget = self._budget_microusd()
            if budget <= 0:
                # Remote work stays disabled until the operator chooses a
                # positive daily budget — a zero-cost reservation is still a
                # dispatch (SPEC §27).
                raise VerbatimError(
                    ErrorCode.BUDGET_EXHAUSTED,
                    "no positive daily egress budget configured",
                    retryable=False,
                )
            spent = int(conn.execute(_LEDGER_SPEND_SQL, (day,)).fetchone()[0])
            if spent + cost > budget:
                raise VerbatimError(
                    ErrorCode.BUDGET_EXHAUSTED,
                    "daily egress budget exhausted",
                    retryable=False,
                )
            rid = new_id()
            conn.execute(
                "INSERT INTO budget_ledger (reservation_id, job_id, day_utc,"
                " token_bound, reserved_cost_microusd, state)"
                " VALUES (?,?,?,?,?,'reserved')",
                (rid, job_id, day, est_tokens, cost),
            )
            refs = self._normalize_refs(input_refs)
            did = self._record_disclosure(conn, sid, processor, purpose, refs)
            if did is not None:
                self._receipts[rid] = (did, sid, refs)
                # Bounded: reservations that are never reconciled cannot
                # grow the correlation map without limit.
                while len(self._receipts) > 4096:
                    self._receipts.pop(next(iter(self._receipts)))
        return rid

    def _reservation(
        self, conn: sqlite3.Connection, reservation_id: str
    ) -> dict[str, Any]:
        row = _row(
            conn.execute(
                "SELECT * FROM budget_ledger WHERE reservation_id = ?",
                (reservation_id,),
            )
        )
        if row is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown budget reservation"
            )
        return row

    def settle(self, reservation_id: str, actual_tokens: int) -> None:
        """Record actual usage against a reservation.

        Actual above the reserved bound is debited at the real amount and the
        row marked 'overrun' — the overrun is accounted and visible, and it
        counts fully toward day_spend so further dispatches block sooner
        (SPEC §27).
        """
        if actual_tokens < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "actual_tokens must be >= 0")
        actual_cost = self._cost_microusd(actual_tokens)
        with self._store.tx() as conn:
            row = self._reservation(conn, reservation_id)
            if row["state"] != "reserved":
                return  # already reconciled; never double-debit
            state = "overrun" if actual_cost > row["reserved_cost_microusd"] else "settled"
            conn.execute(
                "UPDATE budget_ledger SET actual_cost_microusd = ?, state = ?"
                " WHERE reservation_id = ?",
                (actual_cost, state, reservation_id),
            )
            self._close_disclosure(
                conn,
                reservation_id,
                state,
                {
                    "actual_tokens": actual_tokens,
                    "actual_cost_microusd": actual_cost,
                },
            )

    def expire(self, reservation_id: str) -> None:
        """Expire a still-open reservation at its full estimated cost.

        The timeout path: work may have happened remotely, so the full
        estimate is charged rather than zeroed (SPEC §27). Already-settled
        rows are left untouched.
        """
        with self._store.tx() as conn:
            row = self._reservation(conn, reservation_id)
            if row["state"] != "reserved":
                return
            conn.execute(
                "UPDATE budget_ledger SET actual_cost_microusd = reserved_cost_microusd,"
                " state = 'expired' WHERE reservation_id = ?",
                (reservation_id,),
            )
            self._close_disclosure(
                conn,
                reservation_id,
                "expired",
                {"token_bound": row["token_bound"]},
            )


# ----------------------------------------------------------------------
# data minimization (SPEC §27)
# ----------------------------------------------------------------------

def minimize(
    text: str, role_map: Optional[dict[str, str]] = None
) -> tuple[str, dict[str, str]]:
    """Replace speaker identifiers and best-effort secrets before egress.

    ``role_map`` maps literal sensitive strings (names, handles) to
    per-request role labels (``SPEAKER_A`` …); replacements are applied
    longest-first to avoid partial-name leaks, and the returned mapping
    records ``label → original`` only for labels actually applied.

    Emails, phones, bearer tokens, API-key-looking assignments, common key
    prefixes, AWS-style keys, and PEM blocks become ``[REDACTED]`` — a
    documented heuristic, never a guarantee; the primary defense is sending
    only selected fields in the first place.
    """
    if not isinstance(text, str):
        raise VerbatimError(ErrorCode.VALIDATION, "minimize() requires text")
    mapping: dict[str, str] = {}
    out = text
    for original in sorted((role_map or {}), key=len, reverse=True):
        label = role_map[original]
        if original and original in out:
            out = out.replace(original, label)
            mapping[label] = original
    for pattern in (_PEM_RE, _BEARER_RE, _APIKEY_RE, _AWS_RE, _SK_RE, _EMAIL_RE, _PHONE_RE):
        out = pattern.sub(REDACTED, out)
    return out, mapping


def build_state(
    quotes: list[str], qualifiers: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Outbound payload: ONLY the selected quotations + qualifiers.

    Never a whole transcript. The serialized size is bounded so a blown-up
    snapshot fails closed with EVIDENCE_TOO_LARGE instead of silently
    shipping an oversized disclosure.
    """
    if not isinstance(quotes, list) or not all(isinstance(q, str) for q in quotes):
        raise VerbatimError(ErrorCode.VALIDATION, "quotes must be a list of strings")
    quals = dict(qualifiers or {})
    state = {"quotes": list(quotes), "qualifiers": quals}
    size = len(json_dumps(state).encode("utf-8"))
    if size > MAX_STATE_BYTES:
        raise VerbatimError(
            ErrorCode.EVIDENCE_TOO_LARGE,
            f"outbound state {size}B exceeds {MAX_STATE_BYTES}B bound",
        )
    return state
