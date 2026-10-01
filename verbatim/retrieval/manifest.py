"""Counterevidence-first pack manifests (SPEC_V4_5 §03 / §05;
V45-03.01–03.04, V45-05.01–05.04; acceptance D01, D02, D05).

An ordinary pack is a bag of serialized items: a current-state answer can
ship the stale side of a contradiction while the contrary evidence sits
unranked, crowded out of budget, or dropped by an insufficiency verdict —
and nothing on the result says so. A **manifest** is the inspectable
record of what the pack *is made of*: which refs support the answer,
which refs contradict it, which were omitted and why, and which
insufficiency labels fired. It is opt-in
(``request.manifest="counterevidence_first"``) and changes nothing when
unset.

Contract rows implemented here:

- **V45-03.01** — the manifest records supporting refs, contrary refs,
  the as-of state, coverage, the producer kind, and unresolved
  obligations whenever any of those exist and are authorized.
- **V45-03.02** — counterevidence the caller may *not* access (failed
  admission: unauthorized scope, quarantine, suppression, or simply
  absent — these are deliberately indistinguishable on the read path)
  surfaces ONLY as a presence-level insufficiency label
  (``contrary_evidence_inaccessible`` + ``inaccessible_contrary``). No
  title, no identifier, and no count of the hidden side is ever emitted.
  Reconciliation note: ``contrary_withheld`` counts are computed ONLY
  over refs that ARE caller-visible (delivered items whose exact bytes
  were quote-denied, or delivered-then-screened items) — the caller can
  already enumerate those objects under ``read``, so counting them leaks
  nothing. The never-admitted side stays presence-only.
- **V45-03.03** — ``verified_bytes`` (digest-verified payload slices)
  and ``semantic_support`` (probabilistic-lane backing) are distinct
  fields on every included ref; neither implies the other.
- **V45-03.04 / D02** — contrary groups are admitted FIRST under the
  shared budget (counterevidence-first ordering), so contrary evidence
  is crowded out only when it genuinely cannot fit — and every omission
  lands in ``contrary_refs`` with a reason, or in
  ``insufficiency_labels``.
- **V45-05.* (I3)** — ``sufficiency`` pack mode: the packer admits the
  minimal item set covering the query's required identifiers/conditions
  before spending budget on depth. ``required_identifiers`` /
  ``group_covers`` / ``group_mandatory`` implement the declared coverage
  rule; token accounting uses the packer's own chars/4 estimator
  (``_APPROX_CHARS_PER_TOKEN`` — the same number ``ContextPack.tokens``
  reports).

Fail closed: unknown mode tokens raise ``VALIDATION`` — they are never
silently defaulted.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..core.types_v3 import GroupSupport, PackKind, QueryClass
from . import candidates as _cand
from .v3.abstain import extract_hard_identifiers

# ---------------------------------------------------------------------------
# modes (opt-in; validation is fail-closed)
# ---------------------------------------------------------------------------

COUNTEREVIDENCE_FIRST = "counterevidence_first"

#: Manifest modes this build implements. ``None``/``"none"``/``"standard"``
#: are the off positions — current behavior, byte-identical output.
_MANIFEST_OFF = frozenset({None, "", "none", "standard"})
MANIFEST_MODES = frozenset({COUNTEREVIDENCE_FIRST})

#: Pack admission modes. ``standard`` is the v3 greedy order;
#: ``sufficiency`` is the I3 coverage-first order (V45-05).
PACK_MODE_STANDARD = "standard"
PACK_MODE_SUFFICIENCY = "sufficiency"
PACK_MODES = frozenset({PACK_MODE_STANDARD, PACK_MODE_SUFFICIENCY})

MANIFEST_REVISION = "i1_manifest_r1"


def normalize_manifest(value: Any) -> Optional[str]:
    """Coerce the request's manifest flag → a manifest mode or ``None``.

    Off positions return ``None`` (current behavior); unknown tokens are a
    caller error, never a silent default (fail closed).
    """
    if value in _MANIFEST_OFF:
        return None
    token = str(value).strip().lower()
    if token in MANIFEST_MODES:
        return token
    raise VerbatimError(
        ErrorCode.VALIDATION,
        f"unknown manifest mode {value!r} "
        f"(expected {sorted(MANIFEST_MODES)} or none)",
    )


def normalize_pack_mode(value: Any) -> str:
    """Coerce the request's pack-mode flag → ``standard``|``sufficiency``."""
    if value is None or value == "":
        return PACK_MODE_STANDARD
    token = str(value).strip().lower()
    if token in PACK_MODES:
        return token
    raise VerbatimError(
        ErrorCode.VALIDATION,
        f"unknown pack_mode {value!r} (expected {sorted(PACK_MODES)})",
    )


# ---------------------------------------------------------------------------
# roles — what a piece of evidence does FOR a current-state answer
# ---------------------------------------------------------------------------


class Role(str, enum.Enum):
    """An item's epistemic role inside a pack (the declared slot)."""

    SUPPORTING = "supporting"
    CONTRARY = "contrary"
    CONTEXT = "context"


# Lanes whose scoring is model/fuzzy rather than exact — a hit backed only
# by these is probabilistic semantic support, never verified bytes
# (V45-03.03). exact_id/lexical/structured/temporal/browse/working/
# freshness_env/procedural_signature are deterministic surfaces.
SEMANTIC_LANES = frozenset({
    "dense", "sparse", "late", "graph", "causal", "episode_hierarchy",
})

# Query classes where historical lifecycle is expected content, not
# contrary-to-current-state evidence (mirrors union._LENIENT_CLASSES).
_LENIENT_CLASSES = frozenset(
    {QueryClass.PAST_STATE, QueryClass.TIMELINE, QueryClass.ARCHIVE}
)

# Insufficiency labels (V45-03.02 — the ONLY surface for counterevidence
# the caller cannot access; presence-level, never an id/title/count).
L_INSUFFICIENT = "insufficient_evidence"
L_MISSING_IDS = "missing_hard_identifiers"
L_CONTRARY_INCOMPLETE = "contrary_evidence_incomplete"
L_CONTRARY_BUDGET = "contrary_evidence_budget_exceeded"
L_CONTRARY_INSUFFICIENT = "contrary_evidence_insufficient_verdict"
L_CONTRARY_INACCESSIBLE = "contrary_evidence_inaccessible"
L_CONTRARY_WITHHELD = "contrary_evidence_withheld"
L_CONTRARY_SCREENED = "contrary_evidence_screened"
L_NO_CONTRARY = "no_authorized_contrary_evidence"

# Omission reasons recorded per contrary ref (named refs only — these
# objects are authorized-visible; the reason is the audit trail).
R_INCLUDED = "delivered"
R_INCOMPLETE = "group_incomplete"
R_BUDGET = "budget_exceeded"
R_INSUFFICIENT = "insufficient_verdict"
R_KIND = "pack_kind_disallowed"
R_UNREPRESENTABLE = "unrepresentable"
R_SCREENED = "screened_secret"
R_FILTERED = "filtered_or_below_rank"
R_PARTIAL = "partial_verdict"


@dataclass(frozen=True)
class ManifestRef:
    """One named evidence reference on the manifest (V45-03.01).

    Only refs the caller is authorized to see are ever named — the
    never-admitted side appears nowhere here (it is the presence-level
    ``inaccessible_contrary`` label instead).
    """

    object_kind: str
    object_id: str
    revision: int
    role: str                      # Role value
    disposition: str               # included | omitted
    reason: str
    pack: Optional[str] = None
    verified_bytes: bool = False   # digest-verified payload slice shipped
    semantic_support: bool = False # probabilistic-lane backed
    quote_withheld: bool = False   # delivered at read level (bytes denied)

    def to_dict(self) -> dict:
        out = {
            "object_kind": self.object_kind,
            "object_id": self.object_id,
            "revision": self.revision,
            "role": self.role,
            "disposition": self.disposition,
            "reason": self.reason,
            "verified_bytes": self.verified_bytes,
            "semantic_support": self.semantic_support,
        }
        if self.pack is not None:
            out["pack"] = self.pack
        if self.quote_withheld:
            out["quote_withheld"] = True
        return out


# ---------------------------------------------------------------------------
# contrary-edge index — which claims contradict which
# ---------------------------------------------------------------------------


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _contrary_edges(conn: Any, claim_ids: list, scope_ids: list) -> dict:
    """Edge-derived contest map over the claim set.

    Returns ``{"conflicts": set, "corrected": set, "correctors": set,
    "context": set}`` of ``("claim", claim_id)`` keys:

    - ``conflicts_with`` — symmetric contest: BOTH ends are contrary to a
      clean single answer.
    - ``corrects`` (source→target) — the source is the correction
      (supporting, it IS the current answer); the target is the corrected
      stale side (contrary).
    - ``context_of`` — required context, not a contradiction; the
      counterparty role is ``context``.
    - ``supersedes`` is deliberately absent: the predecessor is terminal
      by construction (resolved history, not open contest) and never
      survives admission for strict classes.
    """
    out = {
        "conflicts": set(),
        "corrected": set(),
        "correctors": set(),
        "context": set(),
    }
    if not claim_ids or not scope_ids or not _cand._has_table(conn, "edges"):
        return out
    smarks = _ph(len(scope_ids))
    for part in _cand._chunks(list(dict.fromkeys(claim_ids)), _cand._IN_CHUNK):
        pmarks = _ph(len(part))
        rows = conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges"
            " WHERE retired_event IS NULL"
            " AND source_kind='claim' AND target_kind='claim'"
            f" AND scope_id IN ({smarks})"
            f" AND (source_id IN ({pmarks}) OR target_id IN ({pmarks}))",
            [*scope_ids, *part, *part],
        ).fetchall()
        for source_id, target_id, edge_type in rows:
            if edge_type == "conflicts_with":
                out["conflicts"].add(("claim", source_id))
                out["conflicts"].add(("claim", target_id))
            elif edge_type == "corrects":
                out["correctors"].add(("claim", source_id))
                out["corrected"].add(("claim", target_id))
            elif edge_type == "context_of":
                out["context"].add(("claim", source_id))
                out["context"].add(("claim", target_id))
    return out


def hit_role(hit: Any, edge_map: dict, query_class: QueryClass) -> str:
    """Classify one union hit's epistemic role (Role value)."""
    key = hit.key
    if key in edge_map["conflicts"]:
        return Role.CONTRARY.value
    if key in edge_map["corrected"]:
        return Role.CONTRARY.value
    if getattr(hit, "state", "") == "disputed":
        return Role.CONTRARY.value
    if (
        getattr(hit, "state", "") in ("superseded", "archived")
        and query_class not in _LENIENT_CLASSES
    ):
        # Historical material in a strict current-state answer is the
        # stale side of a contradiction — contrary. Under lenient classes
        # (past/timeline/archive) it is expected content.
        return Role.CONTRARY.value
    if key in edge_map["correctors"]:
        return Role.SUPPORTING.value
    if key in edge_map["context"]:
        return Role.CONTEXT.value
    return Role.SUPPORTING.value


def contrary_index(conn: Any, scope_ids: list, union: Any, groups: list,
                   query_class: QueryClass) -> dict:
    """``{"items": {(kind, oid): role}, "groups": {g.key: role}}``.

    One edges pass over the claim set the groups cover, then per-hit
    classification. Conflict-group members (``union.conflict_groups``)
    are always contrary — the open group IS the contested set.
    """
    scope_ids = sorted(set(scope_ids or ()))
    claim_ids = [
        oid for (k, oid) in union.keys() if k == "claim"
    ]
    edge_map = _contrary_edges(conn, claim_ids, scope_ids)
    item_roles: dict = {}
    for key, hit in union.items():
        if key in union.conflict_groups:
            item_roles[key] = Role.CONTRARY.value
        else:
            item_roles[key] = hit_role(hit, edge_map, query_class)
    group_roles: dict = {}
    for g in groups:
        if g.conflict_group is not None or g.kind == PackKind.CONFLICT_PACK:
            group_roles[g.key] = Role.CONTRARY.value
        elif item_roles.get(g.primary_key) == Role.CONTRARY.value:
            group_roles[g.key] = Role.CONTRARY.value
    return {"items": item_roles, "groups": group_roles}


def item_role(item: Any, object_kind: str, roles: Optional[dict]) -> str:
    """The role stamped on one serialized item under manifest mode."""
    if "REQUIRED_CONTEXT" in getattr(item, "reasons", ()):
        return Role.CONTEXT.value
    if roles is None:
        return Role.SUPPORTING.value
    return roles["items"].get((object_kind, item.claim_id),
                              Role.SUPPORTING.value)


# ---------------------------------------------------------------------------
# sufficiency coverage (V45-05) — the declared coverage rule
# ---------------------------------------------------------------------------


def required_identifiers(request: Any) -> tuple:
    """The query's required identifiers (the abstention hard-identifier
    contract reused verbatim — paths, ``::`` test ids, entity ids)."""
    return extract_hard_identifiers(
        request.query, getattr(request, "entity_ids", ())
    )


def group_covers(group: Any, identifiers: tuple) -> frozenset:
    """Required identifiers this group's items provably name — in the
    evidence text or in the identifier fields themselves (claim/span/
    source ids are identifiers too)."""
    if not identifiers:
        return frozenset()
    blob = " ".join(
        getattr(i, "text", "") or "" for i in group.items
    ).casefold()
    ids = set()
    for i in group.items:
        ids.add(str(getattr(i, "claim_id", "")))
        span = getattr(i, "span", None)
        if span is not None:
            ids.add(str(getattr(span, "span_id", "")))
            ids.add(str(getattr(span, "source_id", "")))
    out = set()
    for u in identifiers:
        u_low = str(u).casefold()
        if u_low in blob or u in ids or u_low in {x.casefold() for x in ids}:
            out.add(u)
    return frozenset(out)


def group_mandatory(group: Any, union: Any,
                    roles: Optional[dict] = None) -> bool:
    """Groups that may never be deferred to depth under ``sufficiency``
    (V45-05.03): contradictions, required context, safety text,
    condition-bearing claims, volatile/stale material, and disputed
    items. Dropping any of these for tokens fails the experiment — they
    are admitted in the coverage band instead."""
    if group.conflict_group is not None or group.kind == PackKind.CONFLICT_PACK:
        return True
    if group.verify_recommended:
        return True
    if roles and roles["groups"].get(group.key) == Role.CONTRARY.value:
        return True
    hit = union.get(group.primary_key)
    if hit is not None and getattr(hit, "cond_keys", ()):
        return True  # condition-bearing claim (applicability constraints)
    for i in group.items:
        if getattr(i, "disputed", False):
            return True
        reasons = getattr(i, "reasons", ())
        if "REQUIRED_CONTEXT" in reasons or "PROCEDURE_SAFETY" in reasons:
            return True
    return False


SUFFICIENCY_RULE = (
    "coverage-first admission: groups carrying >=1 uncovered required "
    "identifier (hard identifiers from the query) or mandatory content "
    "(contradiction/required-context/safety/condition-bearing/volatile) "
    "admit at the requested detail tier before depth-ordered groups; "
    "remaining groups render their navigational (l0) projection — "
    "identity, locator, lifecycle, and a caller/scope/purpose/revision/"
    "expiry-bound expansion ref — instead of eager detail. Group "
    "atomicity, per-kind ceilings, and the shared byte/token budgets "
    "are unchanged; depth is available on expansion, not delivered flat"
)


# ---------------------------------------------------------------------------
# the manifest itself (V45-03.01–03.03)
# ---------------------------------------------------------------------------


def _parse_body(text: str) -> dict:
    """Parse the JSON body out of one serialized item wrapper."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    from ..core.types import safe_json_loads

    body = safe_json_loads(text[start:end + 1])
    return body if isinstance(body, dict) else {}


def _group_membership(groups: list) -> dict:
    """claim_id -> group key — which group an object ships inside."""
    out: dict = {}
    for g in groups:
        for i in g.items:
            out.setdefault(("claim", i.claim_id), g.key)
        for ref in g.object_refs:
            kind, oid = ref[0], ref[1]
            out.setdefault((kind, oid), g.key)
    return out


def build_manifest(
    *,
    conn: Any,
    request: Any,
    plan: Any,
    query_class: QueryClass,
    scope_ids: list,
    union: Any,
    ranked: list,
    groups: list,
    group_status: dict,
    verdict: Any,
    packs: list,
    withheld_refs: list,
    screened_refs: list,
    detail_tier: str,
    pack_mode: str,
    generation: int,
) -> dict:
    """Assemble the counterevidence-first manifest for one delivery.

    Every delivered item is accounted by role; every contrary union hit
    that did NOT ship is accounted with a reason; counterevidence that
    never survived admission appears only as presence-level labels
    (V45-03.02). ``groups`` is the FULL assembled list (pre-regroup), so
    groups filtered by abstention verdicts still get an honest reason.
    """
    roles = contrary_index(conn, scope_ids, union, groups, query_class)
    item_roles = roles["items"]

    # ---- delivered items -------------------------------------------------
    delivered: dict = {}   # (kind, oid) -> ref dict (first pack wins)
    contrary_qwithheld = 0
    max_seq = 0
    per_pack_tokens: dict = {}
    total_tokens = 0
    for p in packs:
        kind_name = p.kind.value if isinstance(p.kind, PackKind) else p.kind
        per_pack_tokens[kind_name] = per_pack_tokens.get(kind_name, 0) + int(
            p.tokens
        )
        total_tokens += int(p.tokens)
        for item in p.items:
            h = item.handle
            key = (h.object_kind, h.object_id)
            if key in delivered:
                continue
            body = _parse_body(item.text)
            role = item_roles.get(key, Role.SUPPORTING.value)
            reasons = body.get("reasons") or ()
            exact = bool((body.get("span") or {}).get("source_id"))
            verified = bool(
                exact
                and body.get("text")
                and body.get("quote_authorized")
                and "UNVERIFIED" not in reasons
                and "LEGACY_UNVERIFIED" not in reasons
            )
            hit = union.get(key)
            semantic = bool(
                hit is not None
                and set(getattr(hit, "lane_ranks", {}) or ()) & SEMANTIC_LANES
            )
            qwith = bool(body.get("quote_withheld"))
            if qwith and role == Role.CONTRARY.value:
                contrary_qwithheld += 1
            seq = int(body.get("recorded_seq") or 0)
            if seq:
                max_seq = max(max_seq, seq)
            delivered[key] = ManifestRef(
                object_kind=h.object_kind,
                object_id=h.object_id,
                revision=int(h.revision),
                role=role,
                disposition="included",
                reason=R_INCLUDED,
                pack=kind_name,
                verified_bytes=verified,
                semantic_support=semantic,
                quote_withheld=qwith,
            )

    # ---- contrary accounting ----------------------------------------------
    membership = _group_membership(groups)
    group_verdicts = getattr(verdict, "group_verdicts", {}) or {}
    group_by_key = {g.key: g for g in groups}
    contrary_included: list = []
    contrary_omitted: list = []
    labels: list = []
    obligations: list = []
    screened_set = {
        (k, o) for (k, o, _r) in (screened_refs or ())
    }

    for key, role in sorted(item_roles.items()):
        if role != Role.CONTRARY.value:
            continue
        kind, oid = key
        hit = union.get(key)
        rev = int(getattr(hit, "revision", 0) or 0) if hit else 0
        if key in delivered:
            ref = delivered[key]
            contrary_included.append(ref)
            continue
        # ---- omitted contrary ref: name it with the reason --------------
        gkey = membership.get(key)
        g = group_by_key.get(gkey)
        status = group_status.get(gkey)
        gverdict = group_verdicts.get(gkey)
        if key in screened_set:
            reason = R_SCREENED
        elif status == "incomplete" or (
            gkey is not None and gkey in getattr(union, "broken_groups", ())
        ):
            reason = R_INCOMPLETE
        elif gverdict == GroupSupport.INSUFFICIENT:
            reason = R_INSUFFICIENT
        elif gverdict == GroupSupport.PARTIAL:
            reason = R_PARTIAL
        elif status == "budget_exceeded":
            reason = R_BUDGET
        elif status == "kind_disallowed":
            reason = R_KIND
        elif status == "empty":
            reason = R_UNREPRESENTABLE
        elif g is None:
            reason = R_FILTERED
        else:
            reason = status or R_FILTERED
        contrary_omitted.append(ManifestRef(
            object_kind=kind, object_id=oid, revision=rev,
            role=Role.CONTRARY.value, disposition="omitted",
            reason=reason,
        ))

    # ---- presence-level withheld accounting (V45-03.02) -------------------
    # Dependency keys that failed admission (unauthorized, held,
    # suppressed, or absent — indistinguishable by design). They never
    # become named refs; a presence label is the whole surface.
    inaccessible_contrary = bool(getattr(union, "incomplete", None)) or bool(
        getattr(union, "broken_groups", None)
    )
    contrary_screened = sum(
        1 for (k, o) in screened_set
        if item_roles.get((k, o)) == Role.CONTRARY.value
    )

    # ---- insufficiency labels (ordered, deduped) --------------------------
    if getattr(verdict, "abstained", False) or (
        getattr(verdict, "verdict", None) == GroupSupport.INSUFFICIENT
    ):
        labels.append(L_INSUFFICIENT)
    if getattr(verdict, "missing_identifiers", ()):
        labels.append(L_MISSING_IDS)
    omitted_reasons = {r.reason for r in contrary_omitted}
    if R_INCOMPLETE in omitted_reasons or getattr(union, "broken_groups", None):
        labels.append(L_CONTRARY_INCOMPLETE)
    if R_INSUFFICIENT in omitted_reasons or R_PARTIAL in omitted_reasons:
        labels.append(L_CONTRARY_INSUFFICIENT)
    if R_BUDGET in omitted_reasons:
        labels.append(L_CONTRARY_BUDGET)
    if inaccessible_contrary:
        labels.append(L_CONTRARY_INACCESSIBLE)
    if contrary_qwithheld:
        labels.append(L_CONTRARY_WITHHELD)
    if contrary_screened:
        labels.append(L_CONTRARY_SCREENED)
    if not contrary_included and not contrary_omitted \
            and not inaccessible_contrary:
        # An explicit coverage statement: within the authorized view the
        # answer is uncontradicted — that fact is itself on the record.
        labels.append(L_NO_CONTRARY)

    # ---- unresolved obligations (V45-03.01) -------------------------------
    verify_group_keys = {
        g.key for g in groups if getattr(g, "verify_recommended", False)
    }
    verify_refs = [
        {"object_kind": k, "object_id": o}
        for (k, o) in sorted(delivered)
        if membership.get((k, o)) in verify_group_keys
    ]
    if verify_refs:
        obligations.append({"kind": "verify_recommended",
                            "refs": verify_refs})
    open_conflicts = sorted({
        gid for gid in union.conflict_groups.values()
    })
    if open_conflicts:
        obligations.append({
            "kind": "open_conflict",
            "members": sorted(
                oid for (k, oid) in union.conflict_groups if k == "claim"
            ),
        })
    missing = list(getattr(verdict, "missing_identifiers", ()) or ())
    if missing:
        obligations.append({
            "kind": "missing_identifiers", "identifiers": missing,
        })
    for g in groups:
        if group_status.get(g.key) == "incomplete":
            pk = g.primary_key or ("", "")
            obligations.append({
                "kind": "incomplete_group",
                "ref": {"object_kind": pk[0], "object_id": pk[1]},
            })

    # ---- coverage + refs partition ----------------------------------------
    gv_counts = {"supported": 0, "partial": 0, "insufficient": 0}
    for v in group_verdicts.values():
        if v == GroupSupport.SUPPORTED:
            gv_counts["supported"] += 1
        elif v == GroupSupport.PARTIAL:
            gv_counts["partial"] += 1
        elif v == GroupSupport.INSUFFICIENT:
            gv_counts["insufficient"] += 1
    supporting_refs = [
        ref.to_dict() for _k, ref in sorted(delivered.items())
        if ref.role != Role.CONTRARY.value
    ]
    contrary_refs = [
        r.to_dict() for r in
        sorted(contrary_included, key=lambda r: (r.object_kind, r.object_id))
    ] + [
        r.to_dict() for r in
        sorted(contrary_omitted, key=lambda r: (r.object_kind, r.object_id))
    ]

    return {
        "manifest": COUNTEREVIDENCE_FIRST,
        "manifest_revision": MANIFEST_REVISION,
        "producer": "verbatim.retrieval.v3.pack",
        "producer_kind": "deterministic_pack",
        "pack_mode": pack_mode,
        "detail_tier": detail_tier,
        "query_class": query_class.value
        if isinstance(query_class, QueryClass) else str(query_class),
        "as_of": {
            "projection_generation": int(generation),
            "valid_at_us": getattr(request, "valid_at_us", None),
            "known_at_seq": getattr(request, "known_at_seq", None),
            "max_recorded_seq": max_seq,
        },
        "coverage": {
            "verdict": (
                verdict.verdict.value
                if getattr(verdict, "verdict", None) is not None
                else "unknown"
            ),
            "reason_codes": list(getattr(verdict, "reason_codes", ()) or ()),
            "hard_identifiers": list(
                getattr(verdict, "hard_identifiers", ()) or ()
            ),
            "missing_identifiers": missing,
            "groups": gv_counts,
        },
        "supporting_refs": supporting_refs,
        "contrary_refs": contrary_refs,
        "contrary_withheld": {
            "quote_withheld": contrary_qwithheld,
            "screened": contrary_screened,
        },
        "inaccessible_contrary": inaccessible_contrary,
        "insufficiency_labels": labels,
        "unresolved_obligations": obligations,
        "tokens": {
            "estimator": "approx_chars_per_token:4",
            "estimated": total_tokens,
            "per_pack": per_pack_tokens,
        },
    }


__all__ = [
    "COUNTEREVIDENCE_FIRST",
    "MANIFEST_MODES",
    "MANIFEST_REVISION",
    "PACK_MODE_STANDARD",
    "PACK_MODE_SUFFICIENCY",
    "PACK_MODES",
    "Role",
    "ManifestRef",
    "SEMANTIC_LANES",
    "SUFFICIENCY_RULE",
    "normalize_manifest",
    "normalize_pack_mode",
    "contrary_index",
    "hit_role",
    "item_role",
    "required_identifiers",
    "group_covers",
    "group_mandatory",
    "build_manifest",
]
