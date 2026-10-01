"""Policy-gated auto-update application (SPEC_V6 V6-03.03).

``add(..., replaces=ref)`` remains the reliable correction path. A
namespace opted into ``update_policy:<ns> = "auto_safe"`` MAY let the
system apply a detected replacement itself — but only for the narrow
class the published twin suite (``eval/v6/twins.py``) bounds: same-type,
same-identifier/entity, unhedged ``newer_value``/``refines`` pairs whose
detector score clears ``AUTO_SAFE_MIN``. Everything else stays advisory
``possible_updates``.

Guards (all must hold or this module returns ``None``):

* ``policy`` — ``meta["update_policy:<ns>"] == "auto_safe"``. Absent or
  unrecognized values default to ``advisory`` — fail closed.
* ``pairing`` — the candidate names *this* new record in *this*
  namespace, and the prior is not the new record itself.
* ``candidate_open`` — a persisted ``update_candidates`` row (when one
  exists for the candidate id) is still ``open``; resolved candidates
  never re-apply.
* ``prior_live`` — the prior's ``source_state`` row exists, belongs to
  this namespace, and is ``disposition='active'``.
* ``prior_asserted`` — the prior record is itself assertive (a hedged
  prior is not a settled value to supersede — the same twin guard the
  detector applies).
* ``same_type`` — the resolved ``type`` representations are identical.
  When the declared field is absent on both sides the resolver still
  classifies the text, so "absent on both" is compatible only when the
  resulting type representations are equal — ``untyped`` vs ``untyped``
  passes, a declared type vs a missing/different one does not.
* ``shared_anchor`` — the records share at least one identifier/entity
  posting key. The anchor set is the resolved view's identifier ∪
  entity values (``enrichment`` ``fields_json`` when projected, else the
  deterministic extractors — the same pick the detector makes), unioned
  with ``entity_postings`` keys when that table exists.
* ``unhedged_new`` — the new record asserts: its polarity is not
  hedged/hypothetical/quoted and no hedge marker appears in its text.
  Two layers: ``updates._nonassertive_text`` (the V5-14.05 twin guard —
  hedge adverbs, hypothetical, future-plan, quoted-majority, hearsay,
  interrogative) then ``evidence.supersession._hedged`` probed at every
  clause-relevant position — the clause-window semantics that catch
  modal+complementizer and intent forms ("confirm that X", "check
  whether X", "we should switch to X") the flat guard misses. When the
  supersession module is unimportable an identical local mirror runs.
* ``relation`` — ``newer_value`` or ``refines`` only. ``contradicts``
  and ``negates`` stay advisory forever: a negation is never safe-auto.
* ``score_floor`` — ``candidate.score >= AUTO_SAFE_MIN``.

``AUTO_SAFE_MIN = 0.9`` is pinned, never learned: under the detector's
deterministic score model a ``newer_value`` pair reaches 0.9 only with a
shared identifier/entity anchor plus near-total subject overlap (the
maximum without an anchor is 0.80), and ``refines`` needs every signal
at maximum. The floor is deliberately near the model's effective top so
only the cleanest value/version moves qualify — the twin suite measures
the residual false-apply rate against the <0.01 upper-95% target.

Application is not a side channel: the effect goes through
``sourcestate.transitions.transition`` — the same fenced coordinator
effect ``Memory.add(replaces=)`` invokes — with the candidate's pinned
``prior_control_version``/``prior_revision`` as the CAS fences,
``superseded_by`` naming ``new_source_id:new_revision``, the namespace
epoch pinned, and a deterministic ``operation_id`` (derived from the
candidate id) so a redelivery replays instead of double-applying. The
advisory row is then resolved through the existing ``adopt_candidate``
machinery. When a ``store`` is supplied the projection generation bumps
inside the same commit exactly like the facade path.

Fence failures observed at apply time (``STALE_DEPENDENCY``,
``STALE_EPOCH``, ``INVALID_TRANSITION``, ``NOT_FOUND_*``) mean the world
moved since candidacy — the pair is no longer provably safe and the
caller keeps it advisory, so they return ``None`` rather than raising.
Malformed input (missing fields, unresolvable new record) still raises
typed ``VerbatimError`` — caller bugs are never silently unsafe.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, Optional

from .. import governance
from ..core.serialize import json_dumps
from ..core.time import now_us as _commit_us
from ..core.time import rfc3339
from ..core.types import ErrorCode, VerbatimError, safe_json_loads
from ..sourcestate import transitions as _stransitions
from ..storage import repos_v5
from ..storage.repos import has_table
from .updates import (
    _NONASSERTIVE,
    _derive_prior,
    _field_of,
    _nonassertive_text,
    _norm,
    _prior_enrichment,
    _resolve_new,
    _source_text,
    adopt_candidate,
)

try:  # supersession's clause-window hedge semantics — reused verbatim
    from ..evidence.supersession import _hedged as _ss_hedged
except Exception:  # pragma: no cover — partial checkouts get the mirror
    _ss_hedged = None


#: Producer label for the auto-apply path — versioned like every
#: deterministic producer so a rule change is a measurable new version.
AUTO_UPDATE_PRODUCER = "auto_update/v1"

#: Pinned score floor (module docstring explains the choice).
AUTO_SAFE_MIN = 0.9

#: The only relations an ``auto_safe`` namespace may apply. ``negates``
#: is excluded unconditionally — a negation is never safe-auto — and
#: ``contradicts`` stays advisory because a conflicting value is exactly
#: the case an operator should see.
SAFE_RELATIONS = frozenset({"newer_value", "refines"})

#: Namespace policy vocabulary (``update_policy:<ns>`` meta rows).
UPDATE_POLICIES = frozenset({"advisory", "auto_safe"})
DEFAULT_UPDATE_POLICY = "advisory"
_POLICY_PREFIX = "update_policy:"

#: Actor label recorded on the transition doc for audit — the policy is
#: the actor; a human caller never impersonated.
_AUTO_ACTOR = "policy:auto_safe"

#: Guard evaluation order — the first failure is ``blocking_guard``.
_GUARD_ORDER = (
    "policy",
    "pairing",
    "candidate_open",
    "prior_live",
    "prior_asserted",
    "same_type",
    "shared_anchor",
    "unhedged_new",
    "relation",
    "score_floor",
)

#: Apply-time fence failures that mean "no longer provably safe" —
#: the pair stays advisory (``None``) rather than erroring.
_STALE_CODES = frozenset(
    {
        ErrorCode.STALE_DEPENDENCY,
        ErrorCode.STALE_EPOCH,
        ErrorCode.INVALID_TRANSITION,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
    }
)

_WORD_RE = re.compile(r"\S+")
_SENTENCE_BOUNDARY_CHARS = frozenset(".!?;:\n")

# --- fallback mirror of supersession._hedged -------------------------
# Used only when verbatim.evidence.supersession is not importable. The
# regexes are copied verbatim from that module so the guard semantics
# are identical (keep in sync — same rule version).
_FB_HEDGE_ADV_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|reportedly|rumou?red(?:ly)?"
    r"|alleged(?:ly)?|apparently|supposedly|seemingly|arguably|unsure"
    r"|uncertain)\b",
    re.IGNORECASE,
)
_FB_MODAL_VERB_RE = re.compile(
    r"\b(?:wonder(?:ing|ed)?|check(?:ing|ed)?|confirm(?:ing|ed)?"
    r"|verif(?:y|ying|ied)|ask(?:ing|ed)?|discuss(?:ing|ed)?"
    r"|question(?:ing|ed)?|consider(?:ing|ed)?|unsure|curious)\b",
    re.IGNORECASE,
)
_FB_COMPLEMENT_RE = re.compile(r"\b(?:whether|if|that|about)\b", re.IGNORECASE)
_FB_SUBORDINATOR_RE = re.compile(
    r"\b(?:if|unless|whenever|in case)\b", re.IGNORECASE
)
_FB_HEARSAY_RE = re.compile(
    r"\b(?:rumou?rs?\s+(?:says|has it)|word is|they say|i heard|"
    r"people say|report has it)\b",
    re.IGNORECASE,
)
_FB_INTENT_RE = re.compile(
    r"(?:\bto|\bwill\b|\bwould\b|\bshall\b|\bshould\b|\bmay\b|\bmight\b"
    r"|\bcould\b|\bmust\b|\bcan\b|\bplease\b|\blet'?s\b|\blets\b)"
    r"\s+(?:[a-z]+\s+){0,2}$",
    re.IGNORECASE,
)
_FB_WHEN_RE = re.compile(r"\bwhen\b", re.IGNORECASE)
_FB_PRESENT_AUX = frozenset({"is", "are", "gets"})
_FB_BOUNDARY_RE = re.compile(r"[.!?;\n:]")


def _fb_hedged(text: str, start: int) -> bool:
    """Local mirror of ``supersession._hedged`` — identical window,
    markers, and modal+complementizer semantics."""
    window = text[max(0, start - 48):start]
    last = None
    for m in _FB_BOUNDARY_RE.finditer(window):
        last = m
    if last is not None:
        window = window[last.end():]
    if _FB_HEDGE_ADV_RE.search(window):
        return True
    if _FB_HEARSAY_RE.search(window):
        return True
    if re.search(r"\bwhether\b", window, re.IGNORECASE):
        return True
    if _FB_SUBORDINATOR_RE.search(window):
        return True
    if _FB_INTENT_RE.search(window):
        return True
    for m in _FB_MODAL_VERB_RE.finditer(window):
        if _FB_COMPLEMENT_RE.search(window, m.end()):
            return True
    return False


def _hedged_text(text: str) -> bool:
    """Whole-text hedge guard for the *new* record.

    ``_nonassertive_text`` first (the detector's own twin guard —
    hedged/hypothetical/quoted/hearsay/future-plan/interrogative text
    asserts nothing), then the supersession clause-window check probed
    at every position where a clause prefix could end: each token start
    (the window then covers everything before it), each sentence
    boundary, and end-of-text. A hedge marker anywhere in the record
    means the record is not an unconditional assertion — unsafe to
    auto-apply. Conservative by construction: a false hedge only ever
    keeps the pair advisory.
    """
    t = str(text or "")
    if not t.strip():
        return False
    if _nonassertive_text(t):
        return True
    positions = [m.start() for m in _WORD_RE.finditer(t)]
    positions.extend(
        i + 1 for i, ch in enumerate(t) if ch in _SENTENCE_BOUNDARY_CHARS
    )
    positions.append(len(t))
    if _ss_hedged is not None:
        return any(_ss_hedged(t, p) for p in positions)
    return any(_fb_hedged(t, p) for p in positions)


# ---------------------------------------------------------------------
# per-namespace update policy (mirrors dedupe_policy: meta rows)
# ---------------------------------------------------------------------


def _policy_key(namespace: str) -> str:
    return _POLICY_PREFIX + namespace


def get_update_policy(conn: sqlite3.Connection, namespace: str) -> str:
    """The namespace's update policy; ``advisory`` when unset or
    unrecognized — the safe default is never auto-apply."""
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?", (_policy_key(namespace),)
    ).fetchone()
    if row is None:
        return DEFAULT_UPDATE_POLICY
    val = safe_json_loads(row[0])
    if isinstance(val, str) and val in UPDATE_POLICIES:
        return val
    return DEFAULT_UPDATE_POLICY


def set_update_policy(
    conn: sqlite3.Connection, namespace: str, policy: str
) -> None:
    """Persist the namespace update policy: ``advisory`` | ``auto_safe``.

    Any other value is a caller error — there is no merge/delete mode
    and no silent mapping to a default.
    """
    if not isinstance(namespace, str) or not namespace:
        raise VerbatimError(ErrorCode.VALIDATION, "namespace is required")
    if policy not in UPDATE_POLICIES:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"update policy must be one of {sorted(UPDATE_POLICIES)}: "
            f"{policy!r}",
        )
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (_policy_key(namespace), json_dumps(policy)),
    )


# ---------------------------------------------------------------------
# guard evaluation
# ---------------------------------------------------------------------


def _candidate_view(candidate: Any) -> Dict[str, Any]:
    """Normalize a ``DetectedCandidate``/mapping/duck-typed candidate to
    the field set the guards and the apply path need."""
    view = {
        "candidate_id": _field_of(candidate, "candidate_id"),
        "namespace": _field_of(candidate, "namespace"),
        "new_source_id": _field_of(candidate, "new_source_id"),
        "new_revision": _field_of(candidate, "new_revision"),
        "prior_source_id": _field_of(candidate, "prior_source_id"),
        "prior_revision": _field_of(candidate, "prior_revision"),
        "prior_control_version": _field_of(
            candidate, "prior_control_version"
        ),
        "relation": _field_of(candidate, "relation"),
        "score": _field_of(candidate, "score"),
        "reason": _field_of(candidate, "reason"),
        "state": _field_of(candidate, "state"),
    }
    missing = [
        k
        for k in (
            "prior_source_id",
            "prior_revision",
            "relation",
            "score",
        )
        if view[k] is None
    ]
    if missing:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"candidate missing required fields: {sorted(missing)}",
        )
    try:
        view["prior_revision"] = int(view["prior_revision"])
        view["score"] = float(view["score"])
        if view["new_revision"] is not None:
            view["new_revision"] = int(view["new_revision"])
        if view["prior_control_version"] is not None:
            view["prior_control_version"] = int(
                view["prior_control_version"]
            )
    except (TypeError, ValueError):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "candidate revision/control_version/score fields must be "
            "numeric",
        )
    return view


def _anchor_keys(conn: sqlite3.Connection, resolved: Any) -> frozenset:
    """Identifier/entity posting keys for one resolved record —
    enrichment-field/extracted values (the detector's own pick) unioned
    with ``entity_postings`` rows when that projection exists."""
    keys = {v for v in resolved.idents | resolved.entities if v}
    if has_table(conn, "entity_postings"):
        try:
            for (entity,) in conn.execute(
                "SELECT entity FROM entity_postings"
                " WHERE source_id = ? AND revision = ?",
                (resolved.source_id, resolved.revision),
            ):
                n = _norm(entity)
                if n:
                    keys.add(n)
        except sqlite3.Error:
            pass
    return frozenset(keys)


def _prior_state_row(conn: sqlite3.Connection, source_id: str):
    return conn.execute(
        "SELECT namespace, control_version, mutation_head, disposition"
        " FROM source_state WHERE source_id = ?",
        (source_id,),
    ).fetchone()


def _resolve_prior_for_candidate(
    conn: sqlite3.Connection, cand: Dict[str, Any]
):
    """Resolve the prior *at the candidate's pinned revision* — the
    record the transition would actually supersede (the CAS fences pin
    it again at apply time)."""
    text = _source_text(conn, cand["prior_source_id"], cand["prior_revision"])
    if not text or not text.strip():
        return None
    enr = _prior_enrichment(
        conn, cand["prior_source_id"], cand["prior_revision"]
    )
    state = _prior_state_row(conn, cand["prior_source_id"])
    cv = int(state[1]) if state is not None else 0
    return _derive_prior(
        cand["prior_source_id"], cand["prior_revision"], text, enr, cv
    )


def evaluate_auto_safe(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    new_record: Any,
    candidate: Any,
) -> Dict[str, Any]:
    """The guard decision for one detected pair — pure (no writes).

    Returns the audit decision dict: per-guard pass/fail under
    ``guards``, the first failure under ``blocking_guard`` (``None``
    when every guard held), and ``safe`` — the conjunction. Callers that
    only need the verdict use ``decision["safe"]`` /
    ``decision["blocking_guard"]``; ``auto_safe_replace`` consumes the
    same evaluation before applying.
    """
    if not namespace:
        raise VerbatimError(ErrorCode.VALIDATION, "namespace is required")
    new = _resolve_new(conn, new_record)
    cand = _candidate_view(candidate)

    decision: Dict[str, Any] = {
        "detector": AUTO_UPDATE_PRODUCER,
        "namespace": namespace,
        "new_source_id": new.source_id,
        "new_revision": new.revision,
        "candidate": cand,
        "guards": {},
        "blocking_guard": None,
        "safe": False,
    }
    guards: Dict[str, Optional[bool]] = decision["guards"]

    # 1. policy — fail closed unless the namespace opted into auto_safe.
    policy = get_update_policy(conn, namespace)
    decision["policy"] = policy
    guards["policy"] = policy == "auto_safe"

    # 2. pairing — the candidate must name this new record in this
    # namespace; self-pairs never apply.
    guards["pairing"] = bool(
        cand["new_source_id"] is not None
        and str(cand["new_source_id"]) == new.source_id
        and (
            cand["new_revision"] is None
            or int(cand["new_revision"]) == new.revision
        )
        and (cand["namespace"] is None or str(cand["namespace"]) == namespace)
        and str(cand["prior_source_id"]) != new.source_id
    )

    # 3. candidate_open — a resolved advisory row never re-applies.
    row_state = None
    if cand["candidate_id"] and has_table(conn, "update_candidates"):
        row = repos_v5.get(
            conn, "update_candidates", {"candidate_id": cand["candidate_id"]}
        )
        row_state = row["state"] if row is not None else None
    declared = cand["state"]
    guards["candidate_open"] = row_state in (None, "open") and declared in (
        None,
        "open",
    )

    # 4. prior_live + prior resolution at the pinned revision.
    state_row = _prior_state_row(conn, cand["prior_source_id"])
    prior = (
        _resolve_prior_for_candidate(conn, cand)
        if state_row is not None
        else None
    )
    guards["prior_live"] = bool(
        state_row is not None
        and state_row[0] == namespace
        and state_row[3] == "active"
        and prior is not None
    )

    # 5. prior_asserted — the prior must itself be an asserted record
    # (same twin guard the detector applies to both sides).
    if prior is not None:
        prior_nonassertive = (
            prior.nonassertive
            if prior.nonassertive is not None
            else _nonassertive_text(prior.text)
        )
        guards["prior_asserted"] = bool(
            prior.polarity not in _NONASSERTIVE and not prior_nonassertive
        )
    else:
        guards["prior_asserted"] = None

    # 6. same_type — identical resolved type representations; both
    # absent → compatible only if the classification is identical.
    guards["same_type"] = (
        bool(prior is not None and new.type == prior.type)
        if prior is not None
        else None
    )

    # 7. shared_anchor — ≥1 shared identifier/entity posting key.
    if prior is not None:
        new_keys = _anchor_keys(conn, new)
        prior_keys = _anchor_keys(conn, prior)
        shared = sorted(new_keys & prior_keys)
        decision["shared_anchors"] = shared
        guards["shared_anchor"] = bool(shared)
    else:
        decision["shared_anchors"] = []
        guards["shared_anchor"] = None

    # 8. unhedged_new — the new record asserts unconditionally.
    guards["unhedged_new"] = bool(
        new.polarity not in _NONASSERTIVE and not _hedged_text(new.text)
    )

    # 9. relation — only value/refinement classes auto-apply.
    guards["relation"] = cand["relation"] in SAFE_RELATIONS

    # 10. score_floor — pinned AUTO_SAFE_MIN.
    guards["score_floor"] = float(cand["score"]) >= AUTO_SAFE_MIN

    blocking = next(
        (g for g in _GUARD_ORDER if guards.get(g) is not True), None
    )
    decision["blocking_guard"] = blocking
    decision["safe"] = blocking is None
    return decision


# ---------------------------------------------------------------------
# application — the same coordinator effect as add(replaces=)
# ---------------------------------------------------------------------


def auto_safe_replace(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    new_record: Any,
    candidate: Any,
    now_us: Optional[int] = None,
    store: Any = None,
    actor: Optional[str] = None,
    operation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Apply ``candidate`` as a supersession iff every auto_safe guard
    holds. Returns the applied-decision dict or ``None`` when unsafe —
    the caller keeps the pair advisory either way.

    The effect is the fenced ``sourcestate.transition`` the
    ``add(replaces=)`` path performs — CAS on the prior's pinned control
    version and mutation head, ``superseded_by`` naming the new
    revision, epoch pin, and immutable transition doc — never a raw
    state-row update. ``store`` (optional) rides the same
    projection-generation bump the facade path triggers.

    ``now_us`` pins the decision timestamp (``known_at``) for
    deterministic replays; ``actor``/``operation_id`` default to the
    policy actor and a candidate-derived idempotent key.
    """
    decision = evaluate_auto_safe(
        conn, namespace=namespace, new_record=new_record, candidate=candidate
    )
    if not decision["safe"]:
        return None

    cand = decision["candidate"]
    new = _resolve_new(conn, new_record)
    prior_sid = str(cand["prior_source_id"])
    prior_rev = int(cand["prior_revision"])

    # CAS expectation: the candidate's pinned control version when it
    # carries one (DetectedCandidate does); a persisted row lacks the
    # column, so fall back to the version verified live inside this same
    # transaction — the head/disposition fences still pin the drift
    # cases.
    expect_cv = cand["prior_control_version"]
    if expect_cv is None:
        row = _prior_state_row(conn, prior_sid)
        if row is None:
            return None
        expect_cv = int(row[1])

    op_id = operation_id or (
        f"auto-update:{cand['candidate_id']}"
        if cand["candidate_id"]
        else "auto-update:"
        + ":".join(
            (namespace, prior_sid, str(prior_rev),
             new.source_id, str(new.revision))
        )
    )

    try:
        state = _stransitions.transition(
            conn,
            prior_sid,
            expected_control_version=int(expect_cv),
            disposition="supersede",
            superseded_by=f"{new.source_id}:{new.revision}",
            producer=AUTO_UPDATE_PRODUCER,
            expected_revision=prior_rev,
            epoch_vector={namespace: governance.current_epoch(conn, namespace)},
            actor=actor or _AUTO_ACTOR,
            operation_id=op_id,
            known_at=now_us,
            store=store,
        )
    except VerbatimError as exc:
        if exc.code in _STALE_CODES:
            return None  # world moved since candidacy — stays advisory
        raise

    # Resolve the advisory row through the existing machinery — an
    # applied candidate is adopted, never left dangling in the open feed.
    adopted = None
    if cand["candidate_id"] and has_table(conn, "update_candidates"):
        row = repos_v5.get(
            conn, "update_candidates", {"candidate_id": cand["candidate_id"]}
        )
        if row is not None and row["state"] == "open":
            try:
                adopted = adopt_candidate(conn, cand["candidate_id"])
            except VerbatimError:
                adopted = None

    stamp = rfc3339(int(now_us) if now_us is not None else _commit_us())
    decision.update(
        {
            "applied": True,
            "relation": cand["relation"],
            "score": cand["score"],
            "candidate_id": cand["candidate_id"],
            "prior_source_id": prior_sid,
            "prior_revision": prior_rev,
            "superseded_by": f"{new.source_id}:{new.revision}",
            "disposition": state.disposition,
            "control_version": state.control_version,
            "producer": AUTO_UPDATE_PRODUCER,
            "actor": actor or _AUTO_ACTOR,
            "operation_id": op_id,
            "adopted_candidate": adopted is not None,
            "applied_at": stamp,
        }
    )
    return {
        "applied": True,
        "decision": decision,
        "reason": (
            f"auto_safe applied {cand['relation']}: "
            f"{prior_sid}@{prior_rev} superseded by "
            f"{new.source_id}@{new.revision} (score {cand['score']})"
        ),
    }


__all__ = [
    "AUTO_SAFE_MIN",
    "AUTO_UPDATE_PRODUCER",
    "DEFAULT_UPDATE_POLICY",
    "SAFE_RELATIONS",
    "UPDATE_POLICIES",
    "auto_safe_replace",
    "evaluate_auto_safe",
    "get_update_policy",
    "set_update_policy",
]
