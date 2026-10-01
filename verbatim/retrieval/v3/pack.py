"""Typed context-pack assembly under one shared budget (SPEC_V3 §30,
V3-30.01–30.10; screening §31; quote authorization V4-08.02/F4-02).

Packs are typed containers — ``evidence_bundle``, ``procedure_pack``,
``working_state``, ``verify_pack``, ``conflict_pack``,
``handoff_capsule`` — sharing ONE byte/token ceiling (V3-30.01). Group
atoms (a claim + its required context spans + its conflict members) are
never split: the whole group fits or the whole group is omitted, and
omission is reported (V3-30.02, V3-30.06).

Serialization rules (V3-30.03): quoted spans are emitted byte-exactly —
never paraphrased, never re-encoded — and every item carries a typed
wrapper declaring it untrusted memory evidence, not an instruction to the
host (V3-30.08). Exact strings (commands, identifiers, versions, safety
rules) keep their stored bytes.

Quote authorization (F4-02 / V4-08.02): exact original text — bytes
sliced out of ``source_revisions.payload`` — renders only when the
caller holds ``quote`` on EVERY contributing scope (the object's scope
and the evidence source's scope) for the request's purpose. The check
runs at the point the bytes are serialized (``_render_item``), never
trusted to an upstream flag. A denied contribution degrades the item to
its read-authorized metadata view (``quote_withheld``); a denied
*required* context member invalidates its whole group (V4-08.05,
V3-30.06). Object-row fields — procedure labels, step descriptions,
observation text, working items, environment state — are the objects'
own stored content: derived views governed by ``read``, not quotation
bytes.

Greedy order (V3-30.04): applicability tier, then fused score per byte,
then stable identifier. Lower-priority packs drop first when the shared
budget binds (V3-30.05). Byte accounting covers item text + metadata +
handles + delimiters (V3-30.07). ``verify_recommended`` marks volatile /
stale-anchored / revalidation-due material (§24.06). Retrieval-time
screening scans serialized output for high-entropy secret shapes and
suppresses the containing item (§31).
"""

from __future__ import annotations

import base64
import enum
import hmac as _hmac
import re
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ...core.types import (
    ErrorCode,
    Lifecycle,
    Precision,
    Provenance,
    SpanRef,
    EvidenceItem,
    TimeInterval,
    VerbatimError,
    safe_json_loads,
    json_dumps,
)
from ...core.types_v3 import (
    FreshnessClass,
    GroupSupport,
    InfluenceHandle,
    PackItem,
    PackKind,
    Perspective,
    QueryClass,
    SecurityLabel,
    ContextPack,
    Verb,
)
from .. import candidates as _cand
from .. import manifest as _manifest
from ..package import (
    _KIND_LIFECYCLE,
    _VER_REASONS,
    _claim_items,
    _context_items,
    _item_dict,
    _quote,
    _span_payloads,
    _valid_label,
    _verify_slice,
)
from .union import UnionHit, snapshot_for

# ---------------------------------------------------------------------------
# budgets (§30.01) — per-kind ceilings as fractions of the request budget
# ---------------------------------------------------------------------------

# (kind, fraction-of-max-bytes ceiling, drop-priority) — lower drop
# priority drops FIRST when the shared budget binds (V3-30.05).
PACK_CEILINGS: dict[PackKind, tuple] = {
    PackKind.VERIFY_PACK: (0.20, 0),
    PackKind.HANDOFF_CAPSULE: (0.25, 1),
    PackKind.CONFLICT_PACK: (0.40, 2),
    PackKind.WORKING_STATE: (0.40, 3),
    PackKind.PROCEDURE_PACK: (0.60, 4),
    PackKind.EVIDENCE_BUNDLE: (1.00, 5),
}

# Query class -> ordered pack kinds it may produce (§30.02).
CLASS_PACKS: dict[QueryClass, tuple] = {
    QueryClass.EXACT_LOOKUP: (
        PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK,
    ),
    QueryClass.CURRENT_STATE: (
        PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK,
        PackKind.WORKING_STATE, PackKind.VERIFY_PACK,
    ),
    QueryClass.PAST_STATE: (PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK),
    QueryClass.TIMELINE: (PackKind.EVIDENCE_BUNDLE,),
    QueryClass.RELATIONSHIP: (
        PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK,
    ),
    QueryClass.CAUSE: (
        PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK,
        PackKind.VERIFY_PACK,
    ),
    QueryClass.PROCEDURE: (
        PackKind.PROCEDURE_PACK, PackKind.EVIDENCE_BUNDLE,
        PackKind.WORKING_STATE, PackKind.VERIFY_PACK,
    ),
    QueryClass.FAILURE: (
        PackKind.PROCEDURE_PACK, PackKind.EVIDENCE_BUNDLE,
        PackKind.CONFLICT_PACK, PackKind.VERIFY_PACK,
    ),
    QueryClass.ARCHIVE: (PackKind.EVIDENCE_BUNDLE,),
    QueryClass.EXPLORATORY: (
        PackKind.EVIDENCE_BUNDLE, PackKind.CONFLICT_PACK,
        PackKind.WORKING_STATE,
    ),
}

_OBJECT_PACK: dict[str, PackKind] = {
    "claim": PackKind.EVIDENCE_BUNDLE,
    "procedure": PackKind.PROCEDURE_PACK,
    "episode": PackKind.EVIDENCE_BUNDLE,
    "observation": PackKind.EVIDENCE_BUNDLE,
    "prospective": PackKind.WORKING_STATE,
    "working_item": PackKind.WORKING_STATE,
    "environment_state": PackKind.VERIFY_PACK,
    "transition": PackKind.EVIDENCE_BUNDLE,
}

# Item-level serialization wrapper (V3-30.08): retrieved bytes are
# untrusted typed data, explicitly framed as non-instructions.
UNTRUSTED_HEADER = (
    "<memory_evidence trust=\"untrusted\" role=\"data\" "
    "instruction=\"none\">"
)
UNTRUSTED_FOOTER = "</memory_evidence>"

# §31 screening: high-entropy secret-shaped strings never leave the store
# inside a delivered pack item.
_SECRET_RES = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*"
               r"[A-Za-z0-9_\-./+=]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-.=+/]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
)

GROUP_MEMBER_CAP = 8
_APPROX_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# progressive disclosure tiers (V4-32.05, V4-32.09)
# ---------------------------------------------------------------------------


class DetailTier(str, enum.Enum):
    """Declared detail level of a pack's serialized items.

    - ``L0`` — navigational: identifiers, kinds, span locators, lifecycle,
      and a short ``title`` for derived objects. No payload bytes are read
      or released; the tier is computable from catalog metadata alone.
    - ``L1`` — grounded overview: the full read-authorized metadata view
      plus bounded summaries (claim subject/predicate, truncated derived
      text). Payload bytes are still never materialized — an exact item's
      ``text`` stays empty and carries ``expandable`` + ``expand`` so a
      caller can request deeper detail through a fresh authorization.
    - ``L2`` — exact detail: payload slices released through the ordinary
      render-point ``quote`` gate + digest verification (unchanged).
    """

    L0 = "l0"
    L1 = "l1"
    L2 = "l2"

    @property
    def order(self) -> int:
        return {"l0": 0, "l1": 1, "l2": 2}[self.value]

    @property
    def releases_bytes(self) -> bool:
        """True only for the tier that may materialize payload bytes."""
        return self is DetailTier.L2


def normalize_detail_tier(value: Any) -> DetailTier:
    """Coerce a tier token (``"l1"``/``"L2"``/``DetailTier``); invalid
    tokens are a caller error, never a silent default."""
    if value is None:
        return DetailTier.L2
    if isinstance(value, DetailTier):
        return value
    token = str(value).strip().lower()
    for tier in DetailTier:
        if token == tier.value or token == tier.name.lower():
            return tier
    raise VerbatimError(
        ErrorCode.VALIDATION,
        f"unknown detail_tier {value!r} (expected l0|l1|l2)",
    )


# Effective-budget ceilings as fractions of the caller's declared limits.
# Preview tiers are deliberately cheaper than exact detail — a navigation
# view should never spend the whole transport budget — and every tier is
# a strict tightening of the request's own limits (the request ceiling is
# never widened). Delivered bytes therefore grow monotonically with
# declared detail for the same underlying content.
_DETAIL_CEILING = {
    DetailTier.L0: (0.30, 0.30),
    DetailTier.L1: (0.60, 0.60),
    DetailTier.L2: (1.00, 1.00),
}
# Floors keep a tiny caller budget usable at preview tiers (a floor of one
# group / a few identifiers) without ever exceeding the caller's limits.
_DETAIL_ITEM_FLOOR = {DetailTier.L0: 4, DetailTier.L1: 4}
_DETAIL_BYTE_FLOOR = {DetailTier.L0: 640, DetailTier.L1: 1024}
_L1_TEXT_CAP = 256     # max utf-8 bytes of derived text serialized at L1
_L0_TITLE_CAP = 96     # navigational label cap at L0


def detail_budget_limits(tier: DetailTier, max_items: int,
                         max_bytes: int) -> tuple:
    """Effective ``(max_items, max_bytes)`` under the tier's ceiling.

    Monotone: ``L0 <= L1 <= L2`` for equal caller limits, and every tier's
    result is ``<=`` the caller's own limits — a preview tier can only ever
    shrink the transport budget, never widen it."""
    item_frac, byte_frac = _DETAIL_CEILING[tier]
    items = max(1, min(int(max_items), int(int(max_items) * item_frac) or 1))
    byts = max(1, min(int(max_bytes), int(int(max_bytes) * byte_frac) or 1))
    if tier in _DETAIL_ITEM_FLOOR:
        items = max(items, min(int(max_items), _DETAIL_ITEM_FLOOR[tier]))
        byts = max(byts, min(int(max_bytes), _DETAIL_BYTE_FLOOR[tier]))
    return items, byts


def _truncate_utf8(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", "ignore")


# ---------------------------------------------------------------------------
# expansion handles (V4-32.09) — caller-, scope-, purpose-, revision-,
# expiry-bound capability tokens, self-verifying under the store key
# ---------------------------------------------------------------------------

_EXPAND_TTL_US = 3_600_000_000  # 1h — expansion is a short-lived re-ask
_EXPAND_PREFIX = "ex_"


def _expand_seal(store: Any, *, scope_id: str, caller_id: str,
                 purpose: Optional[str], epoch: int,
                 object_kind: str, object_id: str, revision: int,
                 **_unused: Any) -> Optional[str]:
    """Mint an expansion token bound to the delivery's full context.

    The token carries ``kind|oid|revision|epoch|purpose`` (each
    adversary-proofed against ``.`` splitting by hex/base64) and a keyed
    MAC over every bound field; ``expand`` re-derives the MAC from the
    *stored* influence row (caller/scope are trusted state, never the
    token's say-so) so a forged or rebound token fails verification
    (V4-32.09).

    The token is DETERMINISTIC in ``(caller, scope, purpose, epoch,
    object, revision)`` — no handle id or wall-clock inside — so two
    byte-identical deliveries render byte-identical items (metamorphic
    replay stability); the delivery proof lives in the influence rows
    the token resolves against, not in the token text. Expiry is
    enforced at expand time from the row's ``created_us`` +
    ``_EXPAND_TTL_US``, never trusted from the token.
    """
    hmac_fn = getattr(store, "hmac", None)
    if hmac_fn is None:
        return None  # a store without a keyed digest cannot seal
    purpose_s = purpose or ""
    mac = hmac_fn(
        (
            f"expand|{scope_id}|{caller_id}|{purpose_s}|{epoch}|"
            f"{object_kind}|{object_id}|{revision}"
        ).encode("utf-8")
    ).hex()[:32]
    kh = object_kind.encode("utf-8").hex()
    oh = object_id.encode("utf-8").hex()
    pb = base64.urlsafe_b64encode(purpose_s.encode("utf-8")).decode("ascii")
    return (
        f"{_EXPAND_PREFIX}{kh}.{oh}.{int(revision)}.{int(epoch)}.{pb}.{mac}"
    )


def _expand_parse(token: str) -> tuple:
    """``(object_kind, object_id, revision, epoch, purpose, mac)`` —
    shape check only; every field is re-verified against stored rows."""
    if not isinstance(token, str) or not token.startswith(_EXPAND_PREFIX):
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "expansion denied"
        )
    body = token[len(_EXPAND_PREFIX):]
    parts = body.split(".")
    if len(parts) != 6:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "expansion denied"
        )
    kh, oh, rev_s, epoch_s, pb, mac = parts
    try:
        object_kind = bytes.fromhex(kh).decode("utf-8")
        object_id = bytes.fromhex(oh).decode("utf-8")
        revision = int(rev_s)
        epoch = int(epoch_s)
        purpose = base64.urlsafe_b64decode(pb.encode("ascii")).decode(
            "utf-8"
        )
    except Exception:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "expansion denied"
        )
    return object_kind, object_id, revision, epoch, purpose, mac


def _expand_verify(store: Any, *, token_fields: tuple, row: Any) -> bool:
    """Constant-time MAC check of an expansion token against the stored
    influence row — the row, not the token, supplies caller/scope/epoch/
    object binding (V4-32.09)."""
    _kind, _oid, _rev, _epoch, purpose, mac = token_fields
    hmac_fn = getattr(store, "hmac", None)
    if hmac_fn is None:
        return False
    expected = hmac_fn(
        (
            f"expand|{row['scope_id']}|{row['caller_id']}|{purpose}|"
            f"{row['epoch']}|{row['object_kind']}|{row['object_id']}|"
            f"{row['revision']}"
        ).encode("utf-8")
    ).hex()[:32]
    return _hmac.compare_digest(expected, mac)


@dataclass
class AssembledGroup:
    """One atomic group before budget packing (§29.05 verdict target)."""

    key: tuple                    # ("claim", id) | ("object", kind, id)
    kind: PackKind
    items: list = field(default_factory=list)   # EvidenceItem
    object_refs: list = field(default_factory=list)  # (kind, id, rev)
    complete: bool = True
    score: float = 0.0
    serialized_size: int = 0
    primary_key: Optional[tuple] = None
    has_lane_hit: bool = False
    verify_recommended: bool = False
    conflict_group: Optional[str] = None
    reasons: tuple = ()
    # Transient: rendered (serialized, withheld) pairs computed during the
    # sizing pass so admitted groups are serialized exactly once.
    rendered: Optional[list] = None


@dataclass
class AssemblyResult:
    """pack() output: the packs plus assembly diagnostics."""

    packs: list = field(default_factory=list)          # ContextPack
    omitted: int = 0
    warnings: list = field(default_factory=list)
    groups: list = field(default_factory=list)         # AssembledGroup
    group_verdicts: dict = field(default_factory=dict)
    screened: int = 0
    compacted: int = 0
    items_total: int = 0
    # F4-02: items whose exact bytes were withheld under read-level
    # authority (metadata view delivered instead).
    quote_withheld: int = 0
    # V4-32.05: the disclosure tier the packs were assembled at.
    detail_tier: str = "l2"
    # V45-03/I1: per-group admission status for the manifest
    # ("admitted" | "incomplete" | "empty" | "budget_exceeded" |
    # "kind_disallowed"); groups filtered pre-pack by abstention carry no
    # status — the manifest reads the verdict for their reason.
    group_status: dict = field(default_factory=dict)
    # (kind, oid, rev) refs whose exact bytes were quote-withheld, and
    # refs screened out entirely — the manifest's caller-visible withheld
    # accounting (V45-03.02 reconciliation: only refs the caller can
    # already enumerate are ever named here).
    withheld_refs: list = field(default_factory=list)
    screened_refs: list = field(default_factory=list)
    # V45-05/I3: the admission mode actually used + the coverage report
    # when ``sufficiency`` ran (None otherwise — the field's presence is
    # the honest declaration the mode executed).
    pack_mode: str = "standard"
    manifest_mode: Optional[str] = None
    coverage: Optional[dict] = None


def _ph(n: int) -> str:
    return ",".join("?" * n)


# ---------------------------------------------------------------------------
# quote authorization (F4-02 / V4-08.02 / V4-10.05)
# ---------------------------------------------------------------------------


class _QuoteGate:
    """Render-point ``quote`` authorization for exact evidence bytes.

    V4-08.02: exact original text requires ``quote``; ``read`` permits
    authorized metadata and approved derived views — never a raw-text
    loophole. The pack stage therefore does not trust the upstream scope
    list alone: at the point an item's bytes are serialized, every
    contributing scope — the object's scope AND the evidence source's
    scope — must authorize ``quote`` for the request's purpose
    (V4-10.05, V4-11.04). Denial, absence, and a stale pinned epoch are
    indistinguishable — the contribution is withheld, never error-named.

    An item's text is "exact" when its span ref names a real
    ``source_id`` — i.e. the text was sliced out of a stored payload by
    ``_quote``. Object-row fields (labels, step descriptions,
    observation text, working items) are the objects' own stored
    content — derived views under ``read``.

    Under the documented governance-absent compatibility shim there is
    no verb model: the gate then falls back to the request's authorized
    scope set — a span whose source lives outside it still cannot
    render, which is strictly narrower than the pre-fix behavior.
    """

    def __init__(self, ctx: Any) -> None:
        self._conn = ctx.conn
        self._request = ctx.request
        # The authorized scope set is only a *floor* for the
        # governance-absent shim; with governance present it is never
        # consulted as proof of quote authority.
        self.scope_ids = frozenset(ctx.scope_ids or ())
        try:
            from ... import governance  # type: ignore
        except ImportError:
            governance = None
        self._gov = governance
        self._caller = None
        if governance is not None:
            task = getattr(self._request, "task", None)
            self._caller = governance.CallerV3(
                principal_id=self._request.caller_id,
                session_id=(
                    (getattr(task, "task_id", None) or "") if task else ""
                ),
            )
        self._purpose = getattr(self._request, "purpose", None)
        self._permit: dict = {}       # scope_id -> bool
        self._src_scope: dict = {}    # source_id -> Optional[scope_id]
        self._claim_scope: dict = {}  # claim_id -> Optional[scope_id]

    # -- per-scope verdict -------------------------------------------------

    def permits_scope(self, scope_id: Optional[str]) -> bool:
        """``quote`` on ``scope_id`` for the request's purpose."""
        if not scope_id:
            return False
        cached = self._permit.get(scope_id)
        if cached is not None:
            return cached
        if self._gov is None:
            # Compatibility shim: no verb model exists; the best
            # available evidence of authority is membership in the
            # authorized scope set the request was already scoped to.
            ok = scope_id in self.scope_ids
        else:
            try:
                self._gov.authorize(
                    self._conn, self._caller, scope_id, Verb.QUOTE.value,
                    purpose=self._purpose,
                )
                ok = True
            except VerbatimError as exc:
                if exc.code in (
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    ErrorCode.STALE_EPOCH,
                ):
                    ok = False
                else:
                    raise  # integrity/validation failures stay loud
        self._permit[scope_id] = ok
        return ok

    # -- contributing-scope resolution -------------------------------------

    def source_scope(self, source_id: str) -> Optional[str]:
        if source_id not in self._src_scope:
            row = self._conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            self._src_scope[source_id] = row[0] if row else None
        return self._src_scope[source_id]

    def claim_scope(self, claim_id: str) -> Optional[str]:
        if claim_id not in self._claim_scope:
            row = self._conn.execute(
                "SELECT scope_id FROM claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            self._claim_scope[claim_id] = row[0] if row else None
        return self._claim_scope[claim_id]

    # -- item verdict -------------------------------------------------------

    @staticmethod
    def is_exact(item: EvidenceItem) -> bool:
        """True when ``item.text`` is a source-payload slice (``_quote``).

        Derived items built from object-row fields carry a synthetic span
        ref with an empty ``source_id`` — they are read-level content.
        """
        return bool(item.span.source_id)

    def item_permits(self, item: EvidenceItem, object_scope: str) -> bool:
        """All contributing scopes must permit ``quote`` at the purpose.

        Contributing scopes: the object the item belongs to, the claim
        the item is filed under (context items name their primary
        claim), and the evidence source owning the sliced bytes.
        """
        if not self.is_exact(item):
            return True
        scopes = set()
        if object_scope:
            scopes.add(object_scope)
        claim_scope = self.claim_scope(item.claim_id)
        if claim_scope:
            scopes.add(claim_scope)
        source_scope = self.source_scope(item.span.source_id)
        if source_scope is None:
            # Exact bytes whose source row cannot be resolved have no
            # accountable scope — fail closed.
            return False
        scopes.add(source_scope)
        if not scopes:
            return False
        return all(self.permits_scope(s) for s in scopes)

    def item_permits_span(self, object_scope: str, source_id: str) -> bool:
        """``quote`` verdict for a raw span slice before ``_quote`` runs.

        Used where a caller has a derived fallback (procedure step
        descriptions): the slice renders only when the object's scope
        AND the source's scope both permit ``quote`` at the purpose.
        """
        scopes = set()
        if object_scope:
            scopes.add(object_scope)
        source_scope = self.source_scope(source_id)
        if source_scope is None:
            return False  # unresolvable source — fail closed
        scopes.add(source_scope)
        if not scopes:
            return False
        return all(self.permits_scope(s) for s in scopes)


def _freshness_enum(value: str) -> FreshnessClass:
    try:
        return FreshnessClass(value)
    except ValueError:
        return FreshnessClass.UNKNOWN


def _screen_text(text: str) -> bool:
    """§31 retrieval-time screening: True when text carries a secret shape."""
    return any(rx.search(text) for rx in _SECRET_RES)


def _security_label(conn: sqlite3.Connection, label_id: Optional[str]) -> SecurityLabel:
    """Resolve a stored label; absent → the permissive default (§14.02)."""
    if label_id and _cand._has_table(conn, "security_labels"):
        row = conn.execute(
            "SELECT source_trust, content_form, attack_risk, review_state,"
            " findings_json, method, rules_revision FROM security_labels"
            " WHERE label_id = ?",
            (label_id,),
        ).fetchone()
        if row is not None:
            try:
                return SecurityLabel(
                    source_trust=row[0],
                    content_form=row[1],
                    attack_risk=row[2],
                    review_state=row[3],
                    findings=tuple(safe_json_loads(row[4]) or ()),
                    method=row[5] or "rules",
                    rules_revision=row[6] or "",
                )
            except ValueError:
                pass
    return SecurityLabel()


def _label_id_for(conn: sqlite3.Connection, kind: str, oid: str,
                  rev: int, hit: UnionHit) -> Optional[str]:
    if hit.security_label_id:
        return hit.security_label_id
    if kind == "procedure" and "security_label_id" in _cand._columns(
        conn, "procedures"
    ):
        row = conn.execute(
            "SELECT security_label_id FROM procedures WHERE procedure_id = ?",
            (oid,),
        ).fetchone()
        return row[0] if row else None
    return None


# ---------------------------------------------------------------------------
# object -> EvidenceItem serialization (reuse v2 quoting)
# ---------------------------------------------------------------------------


class _ClaimBatch:
    """Per-assembly prefetch for claim evidence rows, validity intervals,
    and the context-membership probe — one chunked pass over every claim
    the assembly pass will serialize instead of a query per claim.

    Every per-claim item list is built from the same ordered rows the
    per-claim queries produced (evidence-role ordering and interval order
    preserved); digest verification still runs per span slice at
    serialization time (F4-06).
    """

    def __init__(self, conn: sqlite3.Connection, snap: Any,
                 pairs: list, meta_only: bool = False) -> None:
        self.conn = conn
        self.meta_only = meta_only
        self.evidence: dict = {}     # claim_id -> [row cols after claim_id]
        self.intervals: dict = {}    # claim_id -> [TimeInterval]
        self.span_groups: dict = {}  # span_id -> {context group_id}
        self.claim_scopes = snap.claim_scopes([c for c, _r in pairs])
        if not pairs:
            return
        # Preview tiers (V4-32.05): the metadata projection reads the same
        # rows minus ``sr.payload`` — raw bytes are never materialized for
        # L0/L1. ``has_rev``/``has_payload`` stay boolean column probes so
        # purged/unsliced evidence still reports unrepresentable, and the
        # claim's subject/predicate ride along as the bounded L1 summary.
        if meta_only:
            payload_cols = (
                " NULL, sr.provenance, so.speaker_id,"
                " sp.excerpt_hmac, NULL,"
                " (sr.source_id IS NOT NULL), 0,"
                " cr.subject_id, cr.predicate"
            )
            join_rev = (
                " JOIN claim_revisions cr"
                "   ON cr.claim_id = ce.claim_id"
                "   AND cr.revision = ce.revision"
            )
        else:
            payload_cols = (
                " sr.payload, sr.provenance, so.speaker_id,"
                " sp.excerpt_hmac, sr.payload_hmac,"
                " 1, (sr.payload IS NOT NULL AND length(sr.payload) > 0),"
                " NULL, NULL"
            )
            join_rev = ""
        for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
            where = " OR ".join(
                "(ce.claim_id=? AND ce.revision=?)" for _ in part
            )
            flat = [v for p in part for v in p]
            for row in conn.execute(
                "SELECT ce.claim_id, ce.span_id, ce.evidence_role,"
                " sp.source_id, sp.revision, sp.start_byte, sp.end_byte,"
                + payload_cols +
                " FROM claim_evidence ce"
                " JOIN spans sp ON sp.span_id = ce.span_id"
                " JOIN sources so ON so.source_id = sp.source_id"
                " LEFT JOIN source_revisions sr"
                "   ON sr.source_id = sp.source_id"
                "   AND sr.revision = sp.revision"
                + join_rev +
                f" WHERE {where}"
                " ORDER BY ce.claim_id,"
                " CASE ce.evidence_role WHEN 'primary' THEN 0 ELSE 1 END,"
                " ce.span_id",
                flat,
            ).fetchall():
                self.evidence.setdefault(row[0], []).append(row[1:])
        for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
            where = " OR ".join(
                "(claim_id=? AND revision=?)" for _ in part
            )
            flat = [v for p in part for v in p]
            for cid, f, u, p, tz, b in conn.execute(
                "SELECT claim_id, from_us, until_us, precision, timezone,"
                f" basis FROM valid_intervals WHERE {where}"
                " ORDER BY claim_id, interval_no",
                flat,
            ).fetchall():
                self.intervals.setdefault(cid, []).append(
                    TimeInterval(f, u, Precision(p), tz, b)
                )
        # Which seed spans sit inside any context group — the per-claim
        # ``_context_items`` call is skipped entirely when none do (that
        # helper early-returns empty in exactly that case).
        all_spans = sorted({
            r[0] for rows in self.evidence.values() for r in rows
        })
        if all_spans and (
            snap.has_table("context_members")
            and snap.has_table("context_groups")
        ):
            for part in _cand._chunks(all_spans, _cand._IN_CHUNK):
                for sid, gid in conn.execute(
                    "SELECT span_id, group_id FROM context_members"
                    f" WHERE span_id IN ({_ph(len(part))})",
                    part,
                ).fetchall():
                    self.span_groups.setdefault(sid, set()).add(gid)


def _claim_items_batched(batch: _ClaimBatch, store: Any, hit: UnionHit,
                         plan: Any) -> tuple:
    """The ``_claim_items`` body over prefetched rows — identical item
    construction, ordering, digest verification, and unavailable flags;
    no per-claim queries."""
    rows = batch.evidence.get(hit.object_id, [])
    lifecycle = Lifecycle(hit.state)
    reasons: tuple = ()
    label = _valid_label(
        batch.intervals.get(hit.object_id, []), plan.valid_at_us
    )
    items: list = []
    unavailable = False
    span_ids: list = []
    for (span_id, _role, source_id, span_rev, start, end,
         payload, provenance, speaker_id, excerpt_hmac,
         payload_hmac, _has_rev, _has_payload,
         _subject_id, _predicate) in rows:
        span_ids.append(span_id)
        row = (source_id, span_rev, start, end, payload, provenance,
               speaker_id, excerpt_hmac, payload_hmac)
        verification = (
            _verify_slice(store, row, span_id=span_id)
            if store is not None
            else "unverified"
        )
        text, span_ref, speaker, prov, ok = _quote(row)
        if not ok:
            unavailable = True
            continue
        item_reasons = reasons
        if verification != "verified":
            item_reasons = reasons + (_VER_REASONS[verification],)
        items.append(
            EvidenceItem(
                claim_id=hit.object_id,
                claim_revision=hit.revision,
                text=text,
                span=SpanRef(span_id, *span_ref),
                speaker_id=speaker,
                lifecycle=lifecycle,
                valid_label=label,
                recorded_seq=hit.recorded_from,
                reasons=item_reasons,
                provenance=prov,
                disputed=(hit.state == "disputed")
                or lifecycle == Lifecycle.DISPUTED,
                historical=lifecycle in (
                    Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED,
                ),
            )
        )
    if not items:
        unavailable = True
    return items, unavailable, span_ids


def _claim_items_meta(batch: "_ClaimBatch", hit: UnionHit,
                      plan: Any) -> tuple:
    """``_claim_items_batched`` shape over metadata-only rows (V4-32.05).

    Same ordering and per-span item construction, but ``text`` is never
    populated — the span locator, lifecycle, validity label, provenance,
    and speaker are the item. A span whose ``source_revisions`` row is
    absent is unrepresentable even at metadata level (the locator would
    point at nothing); suppressed spans were already filtered upstream.
    """
    rows = batch.evidence.get(hit.object_id, [])
    lifecycle = Lifecycle(hit.state)
    label = _valid_label(
        batch.intervals.get(hit.object_id, []), plan.valid_at_us
    )
    items: list = []
    unavailable = False
    span_ids: list = []
    summary_cache: dict = {}
    for (span_id, _role, source_id, span_rev, start, end,
         _payload, provenance, speaker_id, _excerpt_hmac,
         _payload_hmac, has_rev, _has_payload,
         subject_id, predicate) in rows:
        span_ids.append(span_id)
        if not has_rev:
            unavailable = True
            continue
        try:
            prov = (
                Provenance(provenance) if provenance else Provenance.UNKNOWN
            )
        except ValueError:
            prov = Provenance.UNKNOWN
        # The claim's structured subject/predicate plus FTS-proven term
        # coverage — a bounded read-level summary, never payload bytes.
        # The renderer surfaces it as ``summary`` at L1; the abstention
        # coverage floor reads the same field (V4-32.05).
        if not summary_cache:
            summary_cache["s"] = _meta_summary(
                subject_id, predicate, batch.conn,
                hit.object_id, hit.revision,
                getattr(plan, "terms", ()),
            )
        items.append(
            EvidenceItem(
                claim_id=hit.object_id,
                claim_revision=hit.revision,
                text=summary_cache["s"],
                span=SpanRef(
                    span_id, source_id or "", span_rev, start, end
                ),
                speaker_id=speaker_id,
                lifecycle=lifecycle,
                valid_label=label,
                recorded_seq=hit.recorded_from,
                reasons=(),
                provenance=prov,
                disputed=(hit.state == "disputed")
                or lifecycle == Lifecycle.DISPUTED,
                historical=lifecycle in (
                    Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED,
                ),
            )
        )
    if not items:
        unavailable = True
    return items, unavailable, span_ids


def _context_meta_items(
    conn: sqlite3.Connection,
    store: Any,
    seed_span_ids: list,
    allowed_set: set,
    plan: Any,
    request: Any,
    primary: dict,
    already_emitted: set,
) -> tuple:
    """``_context_items`` at metadata level — same visibility, same
    incomplete-group semantics, zero payload reads (V4-32.05).

    A member is unrepresentable when its span row, source row, or pinned
    ``source_revisions`` row is gone, or when the span is suppressed —
    exactly the conditions under which the exact tier could not quote it,
    minus the byte-level checks that cannot apply without payload reads.
    """
    required: list = []
    optional: list = []
    incomplete = False
    saw_partial = False
    seed_spans = set(seed_span_ids)
    if not seed_span_ids or not (
        _cand._has_table(conn, "context_members")
        and _cand._has_table(conn, "context_groups")
    ):
        return required, optional, incomplete, saw_partial

    group_rows = conn.execute(
        "SELECT DISTINCT cm.group_id FROM context_members cm"
        f" WHERE cm.span_id IN ({_ph(len(seed_span_ids))})",
        seed_span_ids,
    ).fetchall()
    group_ids = [r[0] for r in group_rows]
    if not group_ids:
        return required, optional, incomplete, saw_partial

    known = plan.known_at_seq
    meta_rows = conn.execute(
        "SELECT group_id, scope_id, completeness, recorded_from,"
        " recorded_until FROM context_groups"
        f" WHERE group_id IN ({_ph(len(group_ids))})",
        group_ids,
    ).fetchall()
    suppressed_groups = _cand._suppressed(
        store, conn, "context_group", group_ids
    )
    visible: dict = {}
    for gid, scope_id, completeness, rec_from, rec_until in meta_rows:
        if gid in suppressed_groups:
            continue
        if scope_id not in allowed_set:
            visible[gid] = (False, completeness, rec_from)
            continue
        if known is None:
            known_ok = rec_until is None
        else:
            known_ok = rec_from <= known and (
                rec_until is None or rec_until > known
            )
        visible[gid] = (known_ok, completeness, rec_from)

    member_rows = conn.execute(
        "SELECT group_id, span_id, role, required, ord FROM context_members"
        f" WHERE group_id IN ({_ph(len(group_ids))}) ORDER BY ord, span_id",
        group_ids,
    ).fetchall()
    members: dict = {}
    member_span_ids: list = []
    for gid, span_id, role, req, _ord in member_rows:
        members.setdefault(gid, []).append((span_id, role, req))
        if span_id not in member_span_ids:
            member_span_ids.append(span_id)
    suppressed_spans = _cand._suppressed(
        store, conn, "span", member_span_ids
    )

    # Metadata projection only: span locator + source/revision presence —
    # ``sr.payload`` is never selected (V4-32.05).
    meta: dict = {}
    if member_span_ids:
        for part in _cand._chunks(member_span_ids, _cand._IN_CHUNK):
            for row in conn.execute(
                "SELECT sp.span_id, sp.source_id, sp.revision,"
                " sp.start_byte, sp.end_byte, so.speaker_id,"
                " sr.provenance, (sr.source_id IS NOT NULL)"
                " FROM spans sp"
                " JOIN sources so ON so.source_id = sp.source_id"
                " LEFT JOIN source_revisions sr"
                "   ON sr.source_id = sp.source_id"
                "   AND sr.revision = sp.revision"
                f" WHERE sp.span_id IN ({_ph(len(part))})",
                part,
            ).fetchall():
                meta[row[0]] = row[1:]

    def _make_item(span_id: str, role: str):
        row = meta.get(span_id)
        if row is None or span_id in suppressed_spans:
            return None
        source_id, s_rev, start, end, speaker, provenance, has_rev = row
        if not has_rev:
            return None
        try:
            prov = (
                Provenance(provenance) if provenance else Provenance.UNKNOWN
            )
        except ValueError:
            prov = Provenance.UNKNOWN
        return EvidenceItem(
            claim_id=primary["claim_id"],
            claim_revision=primary["revision"],
            text="",
            span=SpanRef(span_id, source_id or "", s_rev, start, end),
            speaker_id=speaker,
            lifecycle=primary["lifecycle"],
            valid_label=primary["label"],
            recorded_seq=primary["recorded_from"],
            reasons=("REQUIRED_CONTEXT", f"CONTEXT_{role.upper()}"),
            provenance=prov,
            disputed=primary["disputed"],
            historical=primary["historical"],
        )

    for gid, member_list in members.items():
        shown, completeness, _rec = visible.get(gid, (False, "complete", 0))
        if completeness in ("partial", "deferred"):
            saw_partial = True
        for span_id, role, req in member_list:
            if role == "primary" and span_id not in seed_spans:
                continue
            if span_id in already_emitted:
                continue
            if span_id in suppressed_spans or meta.get(span_id) is None:
                if req:
                    incomplete = True
                continue
            if not shown:
                if req:
                    incomplete = True
                continue
            item = _make_item(span_id, role)
            if item is None:
                if req:
                    incomplete = True
                continue
            already_emitted.add(span_id)
            if req:
                required.append(item)
            else:
                optional.append(item)
    return required, optional, incomplete, saw_partial


def _claim_group_items(
    conn: sqlite3.Connection,
    store: Any,
    hit: UnionHit,
    union: Any,
    plan: Any,
    request_context: dict,
    gate: "_QuoteGate",
    batch: Optional[_ClaimBatch] = None,
    detail: DetailTier = DetailTier.L2,
) -> tuple:
    """Items for a claim group: evidence + required context spans.

    Returns ``(items, complete, extra_refs)``. Required context members
    that fail to quote byte-exactly make the group incomplete — the pack
    stage omits it whole (V3-30.06). F4-02: a required member whose exact
    bytes the caller may not quote is the same kind of unrepresentable —
    it invalidates the group (V4-08.05); denied optional members drop
    silently, exactly like unquotable ones.

    ``batch`` is the per-assembly prefetch; absent, the claim's rows are
    fetched directly (unchanged single-claim semantics). ``detail`` below
    L2 switches both evidence and context walks to the metadata-only
    projection — payload columns are never selected (V4-32.05).
    """
    meta_only = detail is not DetailTier.L2
    if batch is not None:
        if batch.meta_only != meta_only:
            # A batch built for the other projection must not answer here:
            # meta rows carry no bytes (harmless) but L2 needs the payload.
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "claim batch built for a different detail tier",
            )
        if meta_only:
            items, unavailable, span_ids = _claim_items_meta(
                batch, hit, plan
            )
        else:
            items, unavailable, span_ids = _claim_items_batched(
                batch, store, hit, plan
            )
    elif meta_only:
        items, unavailable, span_ids = _claim_items_meta_conn(
            conn, hit, plan
        )
    else:
        items, unavailable, span_ids = _claim_items(
            conn, hit.object_id, hit.revision, hit.state,
            hit.recorded_from, None, plan,
            disputed=hit.state == "disputed", store=store,
        )
    complete = not unavailable and bool(items)
    primary = {
        "claim_id": hit.object_id,
        "revision": hit.revision,
        "lifecycle": items[0].lifecycle if items else Lifecycle.ACTIVE,
        "label": items[0].valid_label if items else "unknown",
        "recorded_from": hit.recorded_from,
        "disputed": hit.state == "disputed",
        "historical": hit.state in ("superseded", "archived"),
    }
    # Context-group visibility is read-level metadata: any scope the
    # caller may contribute at this purpose may hold members. Member
    # *bytes* stay quote-gated per item at render.
    allowed = set(gate.scope_ids)
    if batch is not None:
        scope_id = batch.claim_scopes.get(hit.object_id)
        if scope_id:
            allowed.add(scope_id)
    else:
        scope_row = conn.execute(
            "SELECT scope_id FROM claims WHERE claim_id = ?",
            (hit.object_id,),
        ).fetchone()
        if scope_row:
            allowed.add(scope_row[0])
    if batch is not None and not any(
        sid in batch.span_groups for sid in span_ids
    ):
        # No evidence span participates in a context group — the helper
        # would probe and return empty; skip the per-claim query.
        required, optional, incomplete, _partial = [], [], False, False
    elif meta_only:
        required, optional, incomplete, _partial = _context_meta_items(
            conn, store, span_ids, allowed, plan,
            _ReqProxy(request_context), primary, set(span_ids),
        )
    else:
        required, optional, incomplete, _partial = _context_items(
            conn, store, span_ids, allowed, plan,
            _ReqProxy(request_context), primary, set(span_ids),
        )
    if meta_only:
        # Metadata members ship under ``read`` — the quote gate only bites
        # when bytes are actually released (L2). Membership visibility is
        # already enforced inside the meta walk (group scope ∈ allowed).
        req_kept = list(required)
    else:
        req_kept = []
        for item in required:
            if gate.item_permits(item, hit.scope_id):
                req_kept.append(item)
            else:
                # A denied dependency invalidates the dependent group —
                # shipping the claim minus its required context would be a
                # misleading subset (V4-08.05).
                incomplete = True
        optional = [
            item
            for item in optional
            if gate.item_permits(item, hit.scope_id)
        ]
    complete = complete and not incomplete
    return [*items, *req_kept, *optional], complete, []


def _claim_items_meta_conn(conn: sqlite3.Connection, hit: UnionHit,
                           plan: Any) -> tuple:
    """Unbatched metadata walk — the ``batch=None`` L0/L1 path (expansion
    re-reads a single claim without materializing payload bytes)."""
    rows = conn.execute(
        "SELECT ce.span_id, ce.evidence_role, sp.source_id, sp.revision,"
        " sp.start_byte, sp.end_byte, sr.provenance, so.speaker_id,"
        " (sr.source_id IS NOT NULL), cr.subject_id, cr.predicate"
        " FROM claim_evidence ce"
        " JOIN spans sp ON sp.span_id = ce.span_id"
        " JOIN sources so ON so.source_id = sp.source_id"
        " LEFT JOIN source_revisions sr"
        "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
        " JOIN claim_revisions cr"
        "   ON cr.claim_id = ce.claim_id AND cr.revision = ce.revision"
        " WHERE ce.claim_id=? AND ce.revision=?"
        " ORDER BY CASE ce.evidence_role WHEN 'primary' THEN 0 ELSE 1 END,"
        "          ce.span_id",
        (hit.object_id, hit.revision),
    ).fetchall()
    lifecycle = Lifecycle(hit.state)
    intervals = [
        TimeInterval(f, u, Precision(p), tz, b)
        for f, u, p, tz, b in conn.execute(
            "SELECT from_us, until_us, precision, timezone, basis"
            " FROM valid_intervals WHERE claim_id=? AND revision=?"
            " ORDER BY interval_no",
            (hit.object_id, hit.revision),
        ).fetchall()
    ]
    label = _valid_label(intervals, plan.valid_at_us)
    items: list = []
    unavailable = False
    span_ids: list = []
    summary_cache: dict = {}
    for (span_id, _role, source_id, span_rev, start, end,
         provenance, speaker_id, has_rev, subject_id, predicate) in rows:
        span_ids.append(span_id)
        if not has_rev:
            unavailable = True
            continue
        try:
            prov = (
                Provenance(provenance) if provenance else Provenance.UNKNOWN
            )
        except ValueError:
            prov = Provenance.UNKNOWN
        if not summary_cache:
            summary_cache["s"] = _meta_summary(
                subject_id, predicate, conn,
                hit.object_id, hit.revision,
                getattr(plan, "terms", ()),
            )
        items.append(
            EvidenceItem(
                claim_id=hit.object_id,
                claim_revision=hit.revision,
                text=summary_cache["s"],
                span=SpanRef(span_id, source_id or "", span_rev,
                             start, end),
                speaker_id=speaker_id,
                lifecycle=lifecycle,
                valid_label=label,
                recorded_seq=hit.recorded_from,
                reasons=(),
                provenance=prov,
                disputed=(hit.state == "disputed")
                or lifecycle == Lifecycle.DISPUTED,
                historical=lifecycle in (
                    Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED,
                ),
            )
        )
    if not items:
        unavailable = True
    return items, unavailable, span_ids


class _ReqProxy:
    """Minimal request proxy for v2 helpers that read ``.mode``/``.context``."""

    def __init__(self, context: dict, mode: str = "current") -> None:
        self.context = context
        self.mode = mode


def _object_item(
    conn: sqlite3.Connection,
    hit: UnionHit,
    reasons: tuple,
) -> Optional[EvidenceItem]:
    """Serialize a non-claim object byte-exactly (its own stored text)."""
    kind, oid = hit.object_kind, hit.object_id
    text: Optional[str] = None
    if kind == "procedure":
        row = conn.execute(
            "SELECT task_label, state FROM procedures WHERE procedure_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = row[0]
        hit.state = row[1] or hit.state
    elif kind == "episode":
        row = conn.execute(
            "SELECT label, kind FROM episodes WHERE episode_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = row[0] or f"episode:{oid}"
    elif kind == "observation":
        row = conn.execute(
            "SELECT text FROM observations WHERE observation_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = row[0]
    elif kind == "prospective":
        row = conn.execute(
            "SELECT intention_text, status FROM prospective_records"
            " WHERE record_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = row[0]
        hit.state = row[1] or hit.state
    elif kind == "working_item":
        row = conn.execute(
            "SELECT text, kind, object_ref FROM working_set_items"
            " WHERE item_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = row[0] or (row[2] or "")
    elif kind == "environment_state":
        row = conn.execute(
            "SELECT key, value FROM environment_state WHERE key = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = f"{row[0]} = {row[1]}"
    elif kind == "transition":
        row = conn.execute(
            "SELECT episode_id, ord, edge FROM transitions"
            " WHERE transition_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            return None
        text = f"transition {row[0]}#{row[1]} ({row[2]})"
    if text is None:
        return None
    lifecycle = _KIND_LIFECYCLE.get(hit.state, Lifecycle.ACTIVE)
    return EvidenceItem(
        claim_id=oid,
        claim_revision=hit.revision or 1,
        text=text,
        span=SpanRef(oid, "", hit.revision or 1, 0,
                     len(text.encode("utf-8"))),
        speaker_id=None,
        lifecycle=lifecycle,
        valid_label="unknown",
        recorded_seq=hit.recorded_from,
        reasons=reasons,
        provenance=Provenance.UNKNOWN,
        disputed=False,
        historical=lifecycle in (Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED),
    )


def _procedure_step_items(conn: sqlite3.Connection, hit: UnionHit,
                          reasons: tuple, gate: "_QuoteGate",
                          store: Any = None,
                          detail: DetailTier = DetailTier.L2) -> list:
    """Procedure steps: stored descriptions emitted byte-exactly (§21).

    A step's span-sourced text is a payload slice — ``quote``-gated
    (F4-02). When the caller may not quote the span's contributing
    scopes the step falls back to its stored ``description`` field, a
    derived view governed by ``read`` — same as a span that fails
    byte-exact reconstruction.

    Below L2 the span payload is never fetched: the step renders its
    stored ``description`` (read-level content) and keeps the span id as
    a locator for expansion (V4-32.05).
    """
    if not _cand._has_table(conn, "procedure_steps"):
        return []
    rows = conn.execute(
        "SELECT step_no, span_id, description, hazard, verification"
        " FROM procedure_steps WHERE procedure_id = ? AND revision = ?"
        " ORDER BY step_no",
        (hit.object_id, hit.revision or 1),
    ).fetchall()
    items: list = []
    meta_only = detail is not DetailTier.L2
    payloads = (
        {}
        if meta_only
        else (
            _span_payloads(conn, [r[1] for r in rows if r[1]])
            if rows
            else {}
        )
    )
    for step_no, span_id, desc, hazard, verification in rows:
        body = payloads.get(span_id) if span_id else None
        if body is not None and not gate.item_permits_span(
            hit.scope_id, body[0]
        ):
            body = None  # quote denied for this span — derived fallback
        if body is not None:
            text, span_ref, speaker, prov, ok = _quote(body, store)
            if not ok:
                text, span_ref = None, None
        else:
            text, span_ref = None, None
        if text is None:
            desc_text = desc or ""
            span_ref = ("", hit.revision or 1, 0,
                        len(desc_text.encode("utf-8")))
            text = desc_text
        # hazards/verification are safety-relevant exact strings — append
        # as their own items rather than mutating the stored description.
        extras = []
        if hazard:
            extras.append(f"hazard: {hazard}")
        if verification:
            extras.append(f"verify: {verification}")
        for ex in extras:
            items.append(EvidenceItem(
                claim_id=hit.object_id,
                claim_revision=hit.revision or 1,
                text=ex,
                span=SpanRef(hit.object_id, "", hit.revision or 1, 0,
                             len(ex.encode("utf-8"))),
                speaker_id=None,
                lifecycle=Lifecycle.ACTIVE,
                valid_label="unknown",
                recorded_seq=hit.recorded_from,
                reasons=(*reasons, "PROCEDURE_SAFETY"),
                provenance=Provenance.UNKNOWN,
                disputed=False,
                historical=False,
            ))
        items.append(EvidenceItem(
            claim_id=hit.object_id,
            claim_revision=hit.revision or 1,
            text=text,
            span=SpanRef(span_id or hit.object_id, *span_ref),
            speaker_id=None,
            lifecycle=Lifecycle.ACTIVE,
            valid_label="unknown",
            recorded_seq=hit.recorded_from,
            reasons=(*reasons, "PROCEDURE_STEP"),
            provenance=Provenance.UNKNOWN if body is None else Provenance.UNKNOWN,
            disputed=False,
            historical=False,
        ))
    return items


# ---------------------------------------------------------------------------
# group assembly
# ---------------------------------------------------------------------------


def _verify_needed(hit: UnionHit) -> bool:
    """Volatile/stale/expired material gets a verify recommendation (§24)."""
    return bool(
        hit.stale
        or hit.freshness in ("volatile",)
        or (hit.freshness == "revalidate_after" and hit.stale)
    )


def _conflict_member_lists(conn: sqlite3.Connection, snap: Any,
                           gids: list) -> dict:
    """group_id -> [claim_id] ordered by claim_id — one batched pass over
    the touched conflict groups (the per-group ``ORDER BY`` probe folded
    into a single ``IN`` query)."""
    out: dict = {}
    if not gids or not snap.has_table("conflict_members"):
        return out
    for part in _cand._chunks(list(gids), _cand._IN_CHUNK):
        for gid, cid in conn.execute(
            "SELECT group_id, claim_id FROM conflict_members"
            f" WHERE group_id IN ({_ph(len(part))}) ORDER BY claim_id",
            part,
        ).fetchall():
            out.setdefault(gid, []).append(cid)
    return out


def assemble_groups(
    ctx: Any,
    union: Any,
    ranked: list,
    request_context: dict,
    gate: Optional[_QuoteGate] = None,
    detail: DetailTier = DetailTier.L2,
) -> list:
    """Build atomic groups from the union; completeness is tracked, never
    split (V3-28.02, V3-30.06). ``gate`` is the request's quote
    authority (F4-02); absent, one is constructed from ``ctx``.

    Structure pass and serialization pass are separated so every claim
    that will serialize is prefetched in one batched read (``_ClaimBatch``)
    — the group atoms, membership marks, and completeness verdicts are
    identical to the per-claim assembly.

    ``detail`` selects the disclosure tier (V4-32.05): below L2 the batch
    prefetch and every context/dependency walk use the metadata-only
    projection — ``source_revisions.payload`` is never selected.
    """
    conn = ctx.conn
    store = ctx.store
    snap = snapshot_for(ctx)
    if gate is None:
        gate = _QuoteGate(ctx)
    groups: list = []
    plan_proxy = _PlanProxy(ctx)
    grouped_claims: set = set()

    # conflict member lists — one batched pass over the touched groups
    gids_needed = list(dict.fromkeys(
        union.conflict_groups[e.key]
        for e in ranked
        if union.conflict_groups.get(e.key) is not None
        and union.get(e.key) is not None
        and union[e.key].object_kind == "claim"
    ))
    member_lists = _conflict_member_lists(conn, snap, gids_needed)

    # ---- pass A: group structure (which claims serialize where) -------
    conflict_work: list = []
    seen_conflict: set = set()
    for entry in ranked:
        hit = union.get(entry.key)
        if hit is None or hit.object_kind != "claim":
            continue
        gid = union.conflict_groups.get(entry.key)
        if gid is None or gid in seen_conflict:
            continue
        seen_conflict.add(gid)
        member_ids = member_lists.get(gid, [])
        members = [
            union[("claim", m)] for m in member_ids
            if ("claim", m) in union
        ]
        missing = [m for m in member_ids if ("claim", m) not in union]
        conflict_work.append((entry, gid, members, missing))
        for m in members[:GROUP_MEMBER_CAP]:
            grouped_claims.add(("claim", m.object_id))

    single_work: list = []
    for entry in ranked:
        hit = union.get(entry.key)
        if hit is None or hit.object_kind != "claim":
            continue
        if entry.key in grouped_claims:
            continue
        dep_hits = [
            union[d] for d in _dep_members(union, entry.key)
            if d in union and union[d].object_kind == "claim"
        ][:GROUP_MEMBER_CAP]
        single_work.append((entry, hit, dep_hits))
        grouped_claims.add(entry.key)
        for dh in dep_hits:
            grouped_claims.add(dh.key)

    # ---- pass B: one batched read for every claim that serializes ------
    want: list = []
    seen_w: set = set()

    def _w(h: UnionHit) -> None:
        t = (h.object_id, h.revision)
        if t not in seen_w:
            seen_w.add(t)
            want.append(t)

    for _e, _g, members, _m in conflict_work:
        for m in members[:GROUP_MEMBER_CAP]:
            _w(m)
    for _e, hit, dep_hits in single_work:
        _w(hit)
        for dh in dep_hits:
            _w(dh)
    batch = _ClaimBatch(
        conn, snap, want, meta_only=detail is not DetailTier.L2
    )

    # ---- pass C: serialize the planned groups ---------------------------
    for entry, gid, members, missing in conflict_work:
        complete = gid not in union.broken_groups and not missing
        items: list = []
        refs: list = []
        for m in members[:GROUP_MEMBER_CAP]:
            m_items, m_complete, _ = _claim_group_items(
                conn, store, m, union, plan_proxy, request_context, gate,
                batch=batch, detail=detail,
            )
            complete = complete and m_complete
            items.extend(m_items)
            refs.append(("claim", m.object_id, m.revision))
        groups.append(AssembledGroup(
            key=("conflict", gid),
            kind=PackKind.CONFLICT_PACK,
            items=items,
            object_refs=refs,
            complete=complete,
            score=entry.score,
            primary_key=entry.key,
            has_lane_hit=True,
            conflict_group=gid,
            reasons=("CONFLICT_CLOSURE",),
        ))

    for entry, hit, dep_hits in single_work:
        items, complete, _ = _claim_group_items(
            conn, store, hit, union, plan_proxy, request_context, gate,
            batch=batch, detail=detail,
        )
        # dependency claims travel inside the group (failure edges etc.)
        for d_hit in dep_hits:
            d_items, d_complete, _s = _claim_group_items(
                conn, store, d_hit, union, plan_proxy, request_context,
                gate, batch=batch, detail=detail,
            )
            complete = complete and d_complete
            items.extend(d_items)
        failed_deps = union.incomplete.get(entry.key)
        if failed_deps:
            complete = False
        groups.append(AssembledGroup(
            key=entry.key,
            kind=PackKind.EVIDENCE_BUNDLE,
            items=items,
            object_refs=[("claim", hit.object_id, hit.revision)],
            complete=complete,
            score=entry.score,
            primary_key=entry.key,
            has_lane_hit=bool(hit.lane_ranks),
            verify_recommended=_verify_needed(hit),
            reasons=tuple(sorted(hit.lane_ranks.keys())),
        ))

    # non-claim objects
    for entry in ranked:
        hit = union.get(entry.key)
        if hit is None or hit.object_kind == "claim":
            continue
        reasons = tuple(sorted(hit.lane_ranks.keys())) or ("OBJECT",)
        item = _object_item(conn, hit, reasons)
        if item is None:
            continue
        items = [item]
        if hit.object_kind == "procedure":
            items.extend(_procedure_step_items(
                conn, hit, reasons, gate, store, detail=detail
            ))
        kind = _OBJECT_PACK.get(hit.object_kind, PackKind.EVIDENCE_BUNDLE)
        groups.append(AssembledGroup(
            key=entry.key,
            kind=kind,
            items=items,
            object_refs=[(hit.object_kind, hit.object_id, hit.revision)],
            complete=True,
            score=entry.score,
            primary_key=entry.key,
            has_lane_hit=bool(hit.lane_ranks),
            verify_recommended=_verify_needed(hit),
            reasons=reasons,
        ))
    return groups


def _dep_members(union: Any, parent_key: tuple) -> list:
    """Dependency keys the union recorded for ``parent_key`` (V3-28.02)."""
    return sorted(union.deps.get(parent_key, ()))


class _PlanProxy:
    """Minimal plan proxy for v2 helpers reading known_at/valid_at —
    plus ``terms`` so the metadata projection can prove term coverage
    without reading payload bytes (V4-32.05)."""

    def __init__(self, ctx: Any) -> None:
        self.known_at_seq = ctx.request.known_at_seq
        self.valid_at_us = ctx.request.valid_at_us
        self.terms = tuple(getattr(ctx.plan, "terms", ()) or ())


def _meta_matched_terms(conn: sqlite3.Connection, claim_id: str,
                        revision: int, terms, have: str) -> tuple:
    """Query terms the claim's indexed text provably contains (V4-32.05).

    Coverage under the preview tiers is proven through FTS5 ``MATCH``
    EXISTS probes — the index answers membership (0/1 rows) and the
    indexed bytes are never selected, so L0/L1 still read no payload.
    ``have`` is the already-visible summary text; terms present there
    skip their probe. Bounded at 16 probes per claim.
    """
    if not terms or not _cand._has_table(conn, "facts_fts_idx"):
        return ()
    out: list = []
    for t in list(terms)[:16]:
        tc = str(t).casefold()
        if not tc or tc in have:
            continue
        try:
            row = conn.execute(
                "SELECT 1 FROM facts_fts_idx"
                " JOIN fts_rows fr ON fr.row_id = facts_fts_idx.rowid"
                " WHERE facts_fts_idx MATCH ?"
                " AND fr.claim_id = ? AND fr.claim_revision = ?"
                " LIMIT 1",
                ('"' + tc.replace('"', '""') + '"', claim_id, revision),
            ).fetchone()
        except sqlite3.Error:
            break
        if row:
            out.append(tc)
    return tuple(out)


def _meta_summary(subject_id: Any, predicate: Any, conn: Any = None,
                  claim_id: str = "", revision: int = 0,
                  terms: tuple = ()) -> str:
    """The bounded read-level summary a preview-tier claim item carries:
    the claim's structured subject/predicate plus a ``(matched: …)``
    marker naming the query terms the index proved present — all of it
    caller-visible metadata, never payload bytes."""
    core = " ".join(
        str(x) for x in (subject_id, predicate) if x
    ).strip()
    if conn is not None and terms:
        matched = _meta_matched_terms(
            conn, claim_id, revision, terms, core.casefold()
        )
        if matched:
            core = (
                f"{core} (matched: {', '.join(matched)})"
                if core
                else f"(matched: {', '.join(matched)})"
            )
    return _truncate_utf8(core, _L1_TEXT_CAP)


# ---------------------------------------------------------------------------
# pack budget + assembly
# ---------------------------------------------------------------------------


def _applicability_tier(hit: Optional[UnionHit]) -> int:
    """Greedy ordering tier (V3-30.04): applies < unknown < does_not_apply."""
    if hit is None or hit.applicability is None:
        return 0
    return {"applies": 0, "unknown": 1, "does_not_apply": 2}.get(
        hit.applicability, 1
    )


def _render_item(item: EvidenceItem, gate: _QuoteGate,
                 object_scope: str,
                 detail: DetailTier = DetailTier.L2,
                 expand_ref: Optional[str] = None,
                 object_kind: Optional[str] = None,
                 role: Optional[str] = None) -> tuple:
    """``(serialized, withheld)`` — THE exact-byte render path.

    Every serialized item passes through here, so quote authorization is
    consulted at the point bytes are rendered — an upstream flag alone
    is never trusted (F4-02, V4-08.02). Exact items serialize their
    payload slice only when every contributing scope authorizes
    ``quote`` at the request purpose; otherwise the item renders as its
    read-authorized metadata view with ``quote_withheld`` set and no
    raw bytes. Derived items are unaffected.

    ``detail`` is the disclosure tier (V4-32.05): L0 emits the
    navigational projection (identifiers/locator/lifecycle + a bounded
    ``title`` for derived objects), L1 the grounded overview (metadata
    view + bounded summary, no payload bytes — exact items always carry
    ``expandable``), L2 the exact detail. ``expand_ref`` is the
    caller/scope/purpose/revision/expiry-bound expansion handle
    (V4-32.09) embedded so a later ``expand`` re-asks under fresh
    authorization.
    """
    exact = gate.is_exact(item)
    if detail is DetailTier.L0:
        body = {
            "detail_tier": "l0",
            "object_kind": object_kind or "claim",
            "claim_id": item.claim_id,
            "claim_revision": item.claim_revision,
            "span": {
                "span_id": item.span.span_id,
                "source_id": item.span.source_id,
                "revision": item.span.revision,
                "start_byte": item.span.start_byte,
                "end_byte": item.span.end_byte,
            },
            "lifecycle": item.lifecycle.value,
            "valid_label": item.valid_label,
            "recorded_seq": item.recorded_seq,
            "provenance": item.provenance.value,
            "disputed": item.disputed,
            "historical": item.historical,
            "reasons": list(item.reasons),
            "speaker_id": item.speaker_id,
        }
        if not exact and item.text:
            # Derived objects (procedure labels, step descriptions…) are
            # their own stored content — a bounded title is the
            # navigational cue L0 owes the caller.
            body["title"] = _truncate_utf8(item.text, _L0_TITLE_CAP)
        if role is not None:
            body["role"] = role
        if expand_ref is not None:
            body["expand"] = expand_ref
        return (
            f"{UNTRUSTED_HEADER}{json_dumps(body)}{UNTRUSTED_FOOTER}",
            False,
        )
    if detail is DetailTier.L1:
        body = _item_dict(item)
        body["detail_tier"] = "l1"
        if object_kind:
            body["object_kind"] = object_kind
        if exact:
            # The overview never ships payload bytes: the locator stays,
            # the quotation does not, and the bound expansion ref is the
            # documented path to exact detail. Meta-built items carry the
            # claim's subject/predicate as a bounded summary.
            if item.text:
                body["summary"] = item.text
            body["text"] = ""
            body["expandable"] = True
        else:
            body["text"] = _truncate_utf8(item.text, _L1_TEXT_CAP)
            body["truncated"] = len(body["text"]) != len(item.text)
        if role is not None:
            body["role"] = role
        if expand_ref is not None:
            body["expand"] = expand_ref
        return (
            f"{UNTRUSTED_HEADER}{json_dumps(body)}{UNTRUSTED_FOOTER}",
            False,
        )

    body = _item_dict(item)
    body["detail_tier"] = "l2"
    if object_kind:
        body["object_kind"] = object_kind
    if role is not None:
        body["role"] = role
    permitted = gate.item_permits(item, object_scope) if exact else True
    withheld = exact and not permitted
    if exact:
        body["quote_authorized"] = permitted
    if withheld:
        # The serialized metadata view: identity, lifecycle, span
        # locator, and reasons still ship under ``read`` — the raw
        # quotation bytes never do.
        body["text"] = ""
        body["quote_withheld"] = True
        body["expandable"] = True
    if expand_ref is not None:
        body["expand"] = expand_ref
    return f"{UNTRUSTED_HEADER}{json_dumps(body)}{UNTRUSTED_FOOTER}", withheld


def _regroup_for_ranked(groups: list, union: Any, ranked: list) -> list:
    """Re-anchor pre-assembled groups to a filtered ``ranked`` list.

    Groups are assembled over the ranked set BEFORE abstention verdicts
    may drop individually-insufficient entries. A singleton or object
    group whose primary dropped is omitted; a conflict group whose anchor
    entry dropped while a sibling member survives re-anchors to the first
    surviving ranked member — exactly what a fresh assembly pass over the
    filtered list would produce.
    """
    keep = {e.key for e in ranked}
    first_by_gid: dict = {}
    for e in ranked:
        gid = union.conflict_groups.get(e.key)
        if gid is not None and gid not in first_by_gid:
            first_by_gid[gid] = e
    out: list = []
    for g in groups:
        if g.conflict_group is not None:
            anchor = first_by_gid.get(g.conflict_group)
            if anchor is None:
                continue
            g.primary_key = anchor.key
            g.score = anchor.score
            out.append(g)
        elif g.primary_key in keep:
            out.append(g)
    return out


def assemble_packs(
    ctx: Any,
    union: Any,
    ranked: list,
    request_context: dict,
    mint_handles,
    budgets: Optional[dict] = None,
    groups: Optional[list] = None,
    detail: Any = None,
    expand_seal=None,
    manifest_mode: Any = None,
    pack_mode: Any = None,
) -> AssemblyResult:
    """Serialize groups → greedy-pack under the shared budget → typed packs.

    ``mint_handles`` is a callable ``(pack_kind, kind, oid, rev) -> handle``
    injected by ``recall`` (influence minting is a separate concern §32).

    ``budgets`` — the controller's effective budgets (V3-26.13): when
    present they *tighten* the request's transport limits — ``max_items``
    caps delivered items, ``max_bytes``/``target_tokens`` shrink the
    byte/token budgets. ``None`` keeps the request-only limits (direct
    pack-stage callers and pre-43.02 callers).

    ``groups`` — pre-assembled group atoms (``recall`` builds them once
    for the abstention verdict and reuses them here instead of
    serializing every claim twice); ``None`` assembles from ``ranked``.
    Pre-built groups are re-anchored to ``ranked`` in case abstention
    dropped entries between assembly and packing.

    ``detail`` — the disclosure tier (V4-32.05): the tier ceiling further
    tightens the effective item/byte/token budgets (never widens the
    caller's own limits) and selects each item's serialized projection.

    ``expand_seal`` — optional callable ``(handle, kind, oid, rev) ->
    expand_ref`` minting the caller/scope/purpose/revision/expiry-bound
    expansion token embedded in every delivered item (V4-32.09). Without
    it, items carry no ``expand`` field — callers then have no expansion
    handle, which preview tiers report as ``expandable`` without a ref.

    ``manifest_mode`` — opt-in ``counterevidence_first`` (V45-03/D02):
    contrary groups (open conflicts, corrected/contested primaries) admit
    FIRST under the shared budget, and every serialized item carries its
    epistemic ``role``. ``pack_mode`` — ``sufficiency`` (V45-05/D05):
    groups covering required identifiers/conditions admit before
    depth-ordered groups. Both leave the DEFAULT byte-identical — band
    ordering only engages when a mode is set.
    """
    detail = normalize_detail_tier(detail)
    manifest_mode = _manifest.normalize_manifest(manifest_mode)
    pack_mode = _manifest.normalize_pack_mode(pack_mode)
    request = ctx.request
    gate = _QuoteGate(ctx)
    if groups is None:
        groups = assemble_groups(ctx, union, ranked, request_context,
                                 gate=gate, detail=detail)
    else:
        groups = _regroup_for_ranked(groups, union, ranked)
    result = AssemblyResult(groups=groups, detail_tier=detail.value,
                            pack_mode=pack_mode,
                            manifest_mode=manifest_mode)
    # Manifest/sufficiency context computed once over the re-anchored
    # group set. ``roles`` maps every union hit + group to its epistemic
    # role; ``covered_map`` maps each group to the required identifiers
    # its items name (the declared coverage rule's units).
    roles = None
    if manifest_mode is not None:
        roles = _manifest.contrary_index(
            ctx.conn, getattr(ctx, "scope_ids", ()) or (),
            union, groups, ctx.query_class,
        )
    identifiers = ()
    covered_map: dict = {}
    if pack_mode == _manifest.PACK_MODE_SUFFICIENCY:
        identifiers = _manifest.required_identifiers(request)
        covered_map = {
            g.key: _manifest.group_covers(g, identifiers) for g in groups
        }
    max_bytes = request.max_bytes
    target_tokens = request.target_tokens
    max_items = request.max_items
    if budgets:
        max_bytes = min(max_bytes, int(budgets.get("max_bytes", max_bytes)))
        target_tokens = min(
            target_tokens,
            int(budgets.get("target_tokens", target_tokens)),
        )
        max_items = min(
            max_items, int(budgets.get("max_items", max_items))
        )
    # The tier ceiling tightens — never widens — the effective budgets
    # (V4-32.05); final serialized bytes are still checked against the
    # exact transport payload, not an estimate (V4-32.03).
    max_items, max_bytes = detail_budget_limits(detail, max_items, max_bytes)
    token_budget = min(
        target_tokens * _APPROX_CHARS_PER_TOKEN,
        int(target_tokens * _APPROX_CHARS_PER_TOKEN
            * _DETAIL_CEILING[detail][1]) or 1,
    )

    def _band(g: AssembledGroup) -> int:
        # Band 0 admits before band 1 — under the SAME budgets, ceilings,
        # and group atomicity. V45-03/D02: contrary groups first, so
        # counterevidence is crowded out only when it genuinely cannot
        # fit (and the omission lands on the manifest). V45-05/D05:
        # coverage/mandatory groups first — the minimal-sufficient set
        # before depth. Both modes off → every group is band 1 and the
        # key ordering is byte-identical to the v3 greedy order.
        if roles is not None and (
            roles["groups"].get(g.key) == _manifest.Role.CONTRARY.value
        ):
            return 0
        if covered_map and (
            covered_map.get(g.key)
            or _manifest.group_mandatory(g, union, roles)
        ):
            return 0
        return 1

    # deterministic greedy order: applicability, score/byte, stable id.
    # The sizing pass renders every item once; admitted groups reuse the
    # rendered strings rather than serializing a second time. Handles and
    # expansion refs are minted here too so the rendered body can embed
    # them (V4-32.09); minted-but-undelivered handles never gain an
    # influence row, so expansion on them denies.
    for g in groups:
        hit0 = union.get(g.primary_key)
        obj_scope = getattr(hit0, "scope_id", "") or ""
        kind0 = getattr(hit0, "object_kind", "claim") or "claim"
        # V45-05.01/D05: under ``sufficiency`` the depth band renders its
        # navigational projection — identity, locator, lifecycle, and the
        # bound expand ref — instead of eager detail. The pack still
        # enumerates every relevant object; only required identifiers,
        # conditions, contradictions, and safety content ship expanded.
        item_detail = detail
        if pack_mode == _manifest.PACK_MODE_SUFFICIENCY and _band(g):
            item_detail = DetailTier.L0
        rendered: list = []
        for i in g.items:
            handle = mint_handles(
                g.kind, kind0, i.claim_id, i.claim_revision
            )
            ref = (
                expand_seal(handle, kind0, i.claim_id, i.claim_revision)
                if expand_seal is not None
                else None
            )
            wrapped, withheld = _render_item(
                i, gate, obj_scope, detail=item_detail,
                expand_ref=ref, object_kind=kind0,
                role=(
                    _manifest.item_role(i, kind0, roles)
                    if roles is not None else None
                ),
            )
            rendered.append((wrapped, withheld, handle, ref))
        g.rendered = rendered
        g.serialized_size = max(
            1, sum(len(r[0].encode("utf-8")) + 2 for r in rendered)
        )

    ordered = sorted(
        groups,
        key=lambda g: (
            _band(g),
            # greedy set-cover: within a band, groups covering more
            # required identifiers first (sufficiency; 0-width otherwise)
            -len(covered_map.get(g.key) or ()),
            _applicability_tier(union.get(g.primary_key)),
            -(g.score / g.serialized_size),
            -g.score,
            g.key,
        ),
    )

    # per-kind byte ceilings + shared budget; lower drop-priority first
    ceilings = {
        kind: int(max_bytes * frac)
        for kind, (frac, _prio) in PACK_CEILINGS.items()
    }
    used_by_kind: dict = {kind: 0 for kind in PackKind}
    used_total = 0
    allowed_kinds = set(CLASS_PACKS.get(
        ctx.query_class, (PackKind.EVIDENCE_BUNDLE,)
    ))
    pack_items: dict = {kind: [] for kind in PackKind}
    pack_warnings: dict = {kind: [] for kind in PackKind}
    overflow_groups: list = []
    admitted: list = []
    items_admitted = 0

    for g in ordered:
        if g.kind not in allowed_kinds:
            result.group_status[g.key] = "kind_disallowed"
            continue
        if not g.complete:
            result.omitted += 1
            result.group_status[g.key] = "incomplete"
            pack_warnings[g.kind].append("group_incomplete_omitted")
            result.warnings.append("incomplete_group_omitted")
            continue
        if not g.items:
            result.omitted += 1
            result.group_status[g.key] = "empty"
            continue
        cost = g.serialized_size
        ceiling = ceilings.get(g.kind, max_bytes)
        if (
            used_total + cost > max_bytes
            or used_by_kind[g.kind] + cost > ceiling
            or used_total + cost > token_budget
            or items_admitted + len(g.items) > max_items
        ):
            overflow_groups.append(g)
            result.group_status[g.key] = "budget_exceeded"
            continue
        used_total += cost
        used_by_kind[g.kind] += cost
        items_admitted += len(g.items)
        admitted.append(g)
        result.group_status[g.key] = "admitted"

    # lowest drop-priority packs lose overflowed groups first is implicit:
    # ceilings already bound them; re-try overflowed groups into cheaper
    # kinds is NOT allowed — a conflict group stays a conflict pack.
    for g in overflow_groups:
        result.omitted += 1
        result.warnings.append("budget_omitted_group")

    # ---- serialize admitted groups into PackItems -----------------------
    # Per-recall memos: security labels / label ids / perspectives resolve
    # once per distinct id under the open snapshot.
    _label_memo: dict = {}
    _lid_memo: dict = {}
    _persp_memo: dict = {}
    _MISSING = object()
    for g in admitted:
        hit = union.get(g.primary_key)
        obj_scope = getattr(hit, "scope_id", "") or ""
        kind_for_item = (
            hit.object_kind if hit is not None else "claim"
        )
        rendered = g.rendered
        if rendered is None:
            item_detail = detail
            if (
                pack_mode == _manifest.PACK_MODE_SUFFICIENCY
                and _band(g)
            ):
                item_detail = DetailTier.L0
            rendered = [
                (
                    *_render_item(
                        i, gate, obj_scope, detail=item_detail,
                        object_kind=kind_for_item,
                        role=(
                            _manifest.item_role(i, kind_for_item, roles)
                            if roles is not None else None
                        ),
                    ),
                    mint_handles(
                        g.kind, kind_for_item, i.claim_id,
                        i.claim_revision,
                    ),
                    None,
                )
                for i in g.items
            ]
        for item, (wrapped, withheld, handle, _ref) in zip(
            g.items, rendered
        ):
            if _screen_text(item.text):
                result.screened += 1
                result.screened_refs.append(
                    (kind_for_item, item.claim_id, item.claim_revision)
                )
                result.warnings.append("screened_secret_item")
                continue
            if withheld:
                result.quote_withheld += 1
                result.withheld_refs.append(
                    (kind_for_item, item.claim_id, item.claim_revision)
                )
                pack_warnings[g.kind].append("quote_withheld")
                result.warnings.append("quote_withheld")
            lid_key = (kind_for_item, item.claim_id, item.claim_revision)
            lid = _lid_memo.get(lid_key, _MISSING)
            if lid is _MISSING:
                lid = _label_id_for(
                    ctx.conn, kind_for_item, item.claim_id,
                    item.claim_revision,
                    hit or UnionHit(kind_for_item, ""),
                )
                _lid_memo[lid_key] = lid
            if lid not in _label_memo:
                _label_memo[lid] = _security_label(ctx.conn, lid)
            label = _label_memo[lid]
            pid = getattr(hit, "perspective_id", None) if hit is not None else None
            if pid not in _persp_memo:
                _persp_memo[pid] = _perspective_for(
                    ctx.conn, hit, request_perspective=None
                )
            perspective = _persp_memo[pid]
            derived = kind_for_item in (
                "observation", "procedure", "episode", "transition",
            )
            pack_items[g.kind].append(PackItem(
                handle=handle,
                text=wrapped,
                lifecycle=item.lifecycle.value,
                freshness=_freshness_enum(
                    hit.freshness if hit is not None else "unknown"
                ),
                security=label,
                perspective=perspective,
                derived=derived,
                proof_count=(
                    hit.proof_count if hit is not None else 0
                ),
                verify_recommended=g.verify_recommended,
            ))
            result.items_total += 1
        if g.verify_recommended:
            pack_warnings[g.kind].append("verify_recommended")

    # Inspection surface (C08 / V4-08.02): AssemblyResult.groups rides the
    # public return value, so its items must not retain bytes the caller
    # may not quote — scrub denied exact text the same way it was
    # rendered out of the packs. Below L2 items were assembled without
    # payload bytes at all (the meta item's ``text`` is a read-level
    # summary), so the scrub runs only for the exact tier.
    if detail is DetailTier.L2:
        for g in result.groups:
            obj_scope = getattr(
                union.get(g.primary_key), "scope_id", ""
            ) or ""
            g.items = [
                (
                    replace(item, text="")
                    if item.text
                    and gate.is_exact(item)
                    and not gate.item_permits(item, obj_scope)
                    else item
                )
                for item in g.items
            ]

    for kind in allowed_kinds:
        items = tuple(pack_items[kind])
        if not items and not pack_warnings[kind]:
            continue
        blob = "".join(i.text for i in items).encode("utf-8")
        result.packs.append(ContextPack(
            kind=kind,
            items=items,
            tokens=max(1, len(blob) // _APPROX_CHARS_PER_TOKEN) if items else 0,
            serialized_bytes=len(blob),
            warnings=tuple(sorted(set(pack_warnings[kind]))),
        ))
    if pack_mode == _manifest.PACK_MODE_SUFFICIENCY:
        # The declared coverage rule, honestly reported (V45-05.02): the
        # required units, which the admitted set actually covers, and how
        # admission split between coverage/mandatory and depth groups.
        covered: set = set()
        coverage_admitted = 0
        depth_admitted = 0
        for g in admitted:
            cov = covered_map.get(g.key) or frozenset()
            covered |= set(cov)
            if cov or _manifest.group_mandatory(g, union, roles):
                coverage_admitted += 1
            else:
                depth_admitted += 1
        result.coverage = {
            "pack_mode": _manifest.PACK_MODE_SUFFICIENCY,
            "rule": _manifest.SUFFICIENCY_RULE,
            "required_identifiers": sorted(identifiers),
            "covered_identifiers": sorted(covered),
            "uncovered_identifiers": sorted(set(identifiers) - covered),
            "coverage_groups_admitted": coverage_admitted,
            "depth_groups_admitted": depth_admitted,
            "coverage_tier": detail.value,
            "depth_tier": DetailTier.L0.value,
        }
    return result


def _perspective_for(conn: sqlite3.Connection, hit: Optional[UnionHit],
                     request_perspective: Any) -> Perspective:
    """The item's recorded perspective roles (§30.04); empty when absent."""
    pid = getattr(hit, "perspective_id", None) if hit is not None else None
    if not pid or not _cand._has_table(conn, "perspectives"):
        return Perspective()
    row = conn.execute(
        "SELECT asserter, observer, audience_json FROM perspectives"
        " WHERE perspective_id = ?",
        (pid,),
    ).fetchone()
    if row is None:
        return Perspective()
    subjects = tuple(
        r[0]
        for r in conn.execute(
            "SELECT subject_id FROM perspective_subjects"
            " WHERE perspective_id = ?",
            (pid,),
        ).fetchall()
    )
    return Perspective(
        asserter=row[0],
        subjects=subjects,
        observer=row[1],
        audience=tuple(safe_json_loads(row[2]) or ()),
    )
