"""Procedural memory: evidence-backed task experience (SPEC_V2 §23-24).

A procedure records how a task was *attempted* and what was *observed* —
ordered steps that quote recorded spans, an environment fingerprint, and
outcome receipts from named checkers. It is advisory data: remembered
success never authorizes a side effect, and views return quoted recorded
content only (V2-23.01, V2-23.15, V2-24.04).

Matching is honest three-valued logic: a missing environment key is
UNKNOWN (None), never a wildcard match (V2-23.09); a contradictory key is
a mismatch carrying ENVIRONMENT_MISMATCH semantics (V2-24.05).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any, Optional

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
)
from ..storage.repos import _json_parse, _require_int, _require_str
from ..storage.repos_v2 import DependencyRepo, ProceduresRepo
from ..storage.store import Store
from . import repo as _repo

# Environment keys compared case-insensitively: platform spellings vary
# ("Linux" vs "linux") without changing applicability (V2-23.08).
_CASE_INSENSITIVE_ENV_KEYS = frozenset(
    {"os", "arch", "platform", "runtime", "toolchain", "shell", "distro"}
)

# Lifecycle transitions. ``retired`` is terminal: withdrawal is a recorded
# decision, not a silent delete (V2-24.10 keeps the historical trace).
_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"active", "review", "retired"}),
    "review": frozenset({"active", "retired"}),
    "active": frozenset({"review", "retired"}),
    "retired": frozenset(),
}

_OUTCOME_KINDS = ("success", "failure", "partial", "unknown")

# Recent-outcome window for drift detection.
_RECENT_WINDOW = 3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ----------------------------------------------------------------------
# environment normalization / fingerprint / matching
# ----------------------------------------------------------------------


def _normalize_env_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        v = " ".join(value.split())
        if key.lower() in _CASE_INSENSITIVE_ENV_KEYS:
            v = v.lower()
        return v
    if isinstance(value, dict):
        return {str(k).strip(): _normalize_env_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_normalize_env_value(key, v) for v in value]
        # Scalar collections (feature flags, tool lists) are order-insensitive.
        if all(not isinstance(i, (dict, list)) for i in items):
            items = sorted(items, key=json_dumps)
        return items
    return value


def _normalize_env(env: Any) -> dict[str, Any]:
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "environment must be a dict")
    return {str(k).strip(): _normalize_env_value(str(k), v) for k, v in env.items()}


def environment_fingerprint(env: Any) -> str:
    """Canonical hash of recorded environment constraints.

    Normalizes key whitespace, case-insensitive platform fields, and
    order-insensitive scalar lists before hashing so equivalent
    descriptions fingerprint identically.
    """
    normalized = _normalize_env(env)
    return hashlib.sha256(json_dumps(normalized).encode("utf-8")).hexdigest()


def environment_match(
    recorded_env: Any, current_env: Any
) -> tuple[Optional[bool], dict[str, Any]]:
    """Three-valued applicability of a recorded environment (V2-23.09).

    Returns ``(verdict, detail)`` where verdict is True (every recorded key
    present and equal), False (at least one contradictory key →
    ENVIRONMENT_MISMATCH), or None (empty recorded env or missing keys →
    applicability unknown, never a wildcard pass).
    """
    recorded = _normalize_env(recorded_env)
    current = _normalize_env(current_env)
    if not recorded:
        return None, {
            "reason": "no_recorded_environment",
            "missing": [],
            "mismatched": [],
        }
    missing: list[str] = []
    mismatched: list[str] = []
    for key, rv in recorded.items():
        if key not in current:
            missing.append(key)
        elif json_dumps(current[key]) != json_dumps(rv):
            mismatched.append(key)
    if mismatched:
        return False, {"missing": sorted(missing), "mismatched": sorted(mismatched)}
    if missing:
        return None, {"missing": sorted(missing), "mismatched": []}
    return True, {"missing": [], "mismatched": []}


def _env_error_code(verdict: Optional[bool]) -> Optional[str]:
    if verdict is False:
        return ErrorCode.ENVIRONMENT_MISMATCH.value
    if verdict is None:
        return ErrorCode.APPLICABILITY_UNKNOWN.value
    return None


# ----------------------------------------------------------------------
# outcome statistics and drift
# ----------------------------------------------------------------------


def outcome_stats(outcome_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse counts with explicit denominators (V2-23.14)."""
    stats: dict[str, Any] = {k: 0 for k in _OUTCOME_KINDS}
    for row in outcome_rows:
        kind = row.get("outcome")
        if kind in stats:
            stats[kind] += 1
    stats["total"] = len(outcome_rows)
    stats["last_outcome"] = outcome_rows[-1]["outcome"] if outcome_rows else None
    return stats


def _success_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("outcome") == "success") / len(rows)


def drift_status(outcome_rows: list[dict[str, Any]]) -> str:
    """Outcome trend label: degraded performance must surface, not hide.

    - ``no_outcomes`` — nothing observed yet.
    - ``insufficient_data`` — observations exist but no earlier baseline.
    - ``drifted`` — the entire recent window failed after earlier success.
    - ``degraded`` — recent success rate is below the earlier rate.
    - ``stable`` — otherwise.
    """
    if not outcome_rows:
        return "no_outcomes"
    recent = outcome_rows[-_RECENT_WINDOW:]
    earlier = outcome_rows[:-_RECENT_WINDOW]
    if not earlier:
        return "insufficient_data"
    if all(r.get("outcome") == "failure" for r in recent) and any(
        r.get("outcome") == "success" for r in earlier
    ):
        return "drifted"
    if _success_rate(recent) < _success_rate(earlier):
        return "degraded"
    return "stable"


# ----------------------------------------------------------------------
# proposal / lifecycle / outcomes
# ----------------------------------------------------------------------


def _normalize_step(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"description": raw}
    if isinstance(raw, dict):
        if not isinstance(raw.get("description"), str) or not raw["description"]:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"step {index} requires a non-empty description",
            )
        return dict(raw)
    raise VerbatimError(
        ErrorCode.VALIDATION, f"step {index} must be a dict or description string"
    )


def _normalize_evidence_ref(raw: Any) -> tuple[str, str, int]:
    """Evidence input ref → (input_kind, input_id, input_revision)."""
    if isinstance(raw, str):
        return ("span", raw, 1)
    if isinstance(raw, dict):
        kind = raw.get("kind", "span")
        oid = raw.get("id", raw.get("span_id"))
        rev = raw.get("input_revision", raw.get("revision", 1))
        if oid is None:
            raise VerbatimError(
                ErrorCode.VALIDATION, "evidence ref needs an id/span_id"
            )
        _require_int(rev, "input_revision", minimum=1)
        return (str(kind), str(oid), rev)
    raise VerbatimError(
        ErrorCode.VALIDATION, "evidence ref must be a span id or dict"
    )


def propose_procedure(
    store: Store,
    conn: sqlite3.Connection,
    scope_id: str,
    task_label: str,
    steps: Any,
    *,
    environment: Any = None,
    condition: Any = None,
    span_evidence: Any = None,
) -> str:
    """Record a proposed procedure plus its evidence dependencies.

    ``steps`` is an ordered list of dicts (``description`` required;
    ``span_id``, ``precondition``, ``hazard``, ``verification``,
    ``step_no``, ``input_revision`` optional) or bare description strings.
    Steps must quote recorded observations — this layer never invents
    step text (V2-23.06, V2-24.06).

    Every span referenced by a step or listed in ``span_evidence`` gets a
    ``dependency_refs`` edge (procedure → input, ``invalidation=
    'revalidate'``) so a later source edit marks the procedure stale
    instead of silently trusting revised evidence (V2-06.06).
    """
    _require_str(task_label, "task_label")
    if not isinstance(steps, (list, tuple)) or not steps:
        raise VerbatimError(
            ErrorCode.VALIDATION, "procedure requires at least one step"
        )
    normalized = [_normalize_step(s, i) for i, s in enumerate(steps)]
    seen: set[int] = set()
    for i, s in enumerate(normalized):
        no = s.get("step_no", i + 1)
        _require_int(no, "step_no")
        if no in seen:
            raise VerbatimError(ErrorCode.VALIDATION, f"duplicate step_no {no}")
        seen.add(no)
        s["step_no"] = no
    normalized.sort(key=lambda s: s["step_no"])
    # Environment is stored verbatim (canonical JSON); normalization is a
    # comparison-time concern only (environment_match/fingerprint).
    if environment is not None and not isinstance(environment, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "environment must be a dict")

    prepo = ProceduresRepo(store)
    pid = prepo.create(
        conn,
        scope_id,
        task_label,
        environment=environment,
        condition=condition,
        state="proposed",
    )
    revision = 1
    deps = DependencyRepo(store)
    for s in normalized:
        span_id = s.get("span_id")
        if span_id is not None:
            require_id(span_id, "span_id")
        prepo.add_step(
            conn,
            pid,
            revision,
            s["step_no"],
            s["description"],
            span_id=span_id,
            precondition=s.get("precondition"),
            hazard=s.get("hazard"),
            verification=s.get("verification"),
        )
        if span_id is not None:
            deps.add(
                conn,
                scope_id,
                "procedure",
                pid,
                "span",
                span_id,
                derived_revision=revision,
                input_revision=_require_int(
                    s.get("input_revision", 1), "input_revision", minimum=1
                ),
                invalidation="revalidate",
            )
    for raw in span_evidence or ():
        kind, oid, rev = _normalize_evidence_ref(raw)
        deps.add(
            conn,
            scope_id,
            "procedure",
            pid,
            kind,
            oid,
            derived_revision=revision,
            input_revision=rev,
            invalidation="revalidate",
        )
    return pid


def set_procedure_state(
    store: Store,
    conn: sqlite3.Connection,
    procedure_id: str,
    state: str,
    row_version: int,
) -> None:
    """Transition guard + expected-version fence (V2-23.13).

    The ``row_version`` read-modify-write fence makes concurrent
    promotions fail with STALE_PROPOSAL rather than silently interleaving.
    Same-state writes are a no-op; illegal transitions raise
    INVALID_TRANSITION before touching the row.
    """
    row = _repo.procedure_row(conn, procedure_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "procedure not found"
        )
    current = row["state"]
    if state == current:
        return
    if state not in _TRANSITIONS.get(current, frozenset()):
        raise VerbatimError(
            ErrorCode.INVALID_TRANSITION,
            f"procedure state {current} -> {state} is not allowed",
        )
    ProceduresRepo(store).set_state(conn, procedure_id, state, row_version)


def activate_procedure(
    store: Store,
    conn: sqlite3.Connection,
    procedure_id: str,
    row_version: int,
) -> None:
    """proposed|review → active under the expected-version fence."""
    set_procedure_state(store, conn, procedure_id, "active", row_version)


def record_outcome(
    store: Store,
    conn: sqlite3.Connection,
    procedure_id: str,
    revision: int,
    outcome: str,
    checker: str,
    *,
    checked_artifact: Optional[str] = None,
    detail: Any = None,
    recorded_us: int = 0,
) -> str:
    """Append a verification receipt; returns receipt_id.

    Outcomes are recorded separately per checker — an agent saying done is
    not an independent tool/test result (V2-23.04/05).
    """
    if _repo.procedure_row(conn, procedure_id) is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "procedure not found"
        )
    return ProceduresRepo(store).record_outcome(
        conn,
        procedure_id,
        revision,
        outcome=outcome,
        checker=checker,
        checked_artifact=checked_artifact,
        detail=detail,
        recorded_us=recorded_us,
    )


# ----------------------------------------------------------------------
# retrieval / inspection
# ----------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN_RE.findall(text or "") if t.strip("_"))


def _label_score(q_tokens: frozenset[str], l_tokens: frozenset[str]) -> float:
    """Deterministic label match: query-token coverage blended with Jaccard.

    Identical token sets score exactly 1.0; no shared tokens scores 0.
    """
    if not q_tokens:
        return 0.0
    inter = q_tokens & l_tokens
    if not inter:
        return 0.0
    coverage = len(inter) / len(q_tokens)
    jaccard = len(inter) / len(q_tokens | l_tokens)
    return 0.7 * coverage + 0.3 * jaccard


def match_procedure(
    store: Store,
    scope_id: str,
    query: str,
    current_env: Any = None,
    *,
    limit: int = 8,
    include_retired: bool = False,
) -> list[dict[str, Any]]:
    """Candidate procedures for a task intent — MemoryKind.PROCEDURE lane.

    Ranked by deterministic label-token match; each candidate is labeled
    with its three-valued environment verdict, outcome statistics with
    denominators, and a drift flag. Retired procedures stay out of the
    candidate set unless explicitly requested — their history remains
    inspectable through ``procedure_view``.
    """
    require_id(scope_id, "scope_id")
    _require_str(query, "query")
    _require_int(limit, "limit", minimum=1)
    q_tokens = _tokens(query)
    rows = ProceduresRepo(store).for_scope(scope_id)
    candidates: list[dict[str, Any]] = []
    with store.read() as conn:
        for row in rows:
            if row["state"] == "retired" and not include_retired:
                continue
            label = row["task_label"]
            score = _label_score(q_tokens, _tokens(label))
            if q_tokens and score <= 0.0:
                continue
            env = _json_parse(row["environment_json"]) or {}
            verdict, detail = environment_match(env, current_env)
            outcomes = _repo.outcomes_for(conn, row["procedure_id"])
            candidates.append(
                {
                    "kind": "procedure",
                    "advisory": True,
                    "procedure_id": row["procedure_id"],
                    "task_label": label,
                    "state": row["state"],
                    "revision": row["revision"],
                    "row_version": row["row_version"],
                    "match_score": round(score, 6),
                    "matched_tokens": sorted(q_tokens & _tokens(label)),
                    "environment": env,
                    "environment_fingerprint": environment_fingerprint(env),
                    "environment_match": verdict,
                    "environment_detail": detail,
                    "environment_error": _env_error_code(verdict),
                    "outcome_stats": outcome_stats(outcomes),
                    "drift": drift_status(outcomes),
                }
            )
    env_rank = {True: 0, None: 1, False: 2}
    candidates.sort(
        key=lambda c: (
            -c["match_score"],
            env_rank[c["environment_match"]],
            c["procedure_id"],
        )
    )
    return candidates[:limit]


def procedure_view(store: Store, procedure_id: str) -> dict[str, Any]:
    """Full inspectable record — quoted steps, evidence refs, outcomes.

    Advisory data only: the view carries ``advisory: True`` and never
    grants authority to execute anything (V2-24.04/08).
    """
    prepo = ProceduresRepo(store)
    row = prepo.get(procedure_id)
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, "procedure not found"
        )
    revision = row["revision"]
    env = _json_parse(row["environment_json"]) or {}
    steps = [
        {
            "step_no": s["step_no"],
            "description": s["description"],
            "span_id": s["span_id"],
            "precondition": _json_parse(s["precondition_json"]),
            "hazard": s["hazard"],
            "verification": s["verification"],
        }
        for s in prepo.steps(procedure_id, revision)
    ]
    with store.read() as conn:
        outcomes = _repo.outcomes_for(conn, procedure_id)
    deps = DependencyRepo(store).inputs_of("procedure", procedure_id)
    return {
        "kind": "procedure",
        "advisory": True,
        "procedure_id": row["procedure_id"],
        "scope_id": row["scope_id"],
        "revision": revision,
        "row_version": row["row_version"],
        "task_label": row["task_label"],
        "state": row["state"],
        "environment": env,
        "environment_fingerprint": environment_fingerprint(env),
        "condition": _json_parse(row["condition_json"]),
        "steps": steps,
        "evidence_inputs": [
            {
                "input_kind": d["input_kind"],
                "input_id": d["input_id"],
                "input_revision": d["input_revision"],
                "invalidation": d["invalidation"],
            }
            for d in deps
        ],
        "outcomes": [
            {
                "receipt_id": o["receipt_id"],
                "outcome": o["outcome"],
                "checker": o["checker"],
                "checked_artifact": o["checked_artifact"],
                "detail": _json_parse(o["detail_json"]) or {},
                "recorded_us": o["recorded_us"],
            }
            for o in outcomes
        ],
        "outcome_stats": outcome_stats(outcomes),
        "drift": drift_status(outcomes),
        "recorded_from": row["recorded_from"],
        "recorded_until": row["recorded_until"],
    }


def procedures_for_scope(
    store: Store, scope_id: str, *, state: Optional[str] = None
) -> list[dict[str, Any]]:
    """Current procedures in a scope — flat candidate lane for retrieval."""
    require_id(scope_id, "scope_id")
    if state is not None and state not in ProceduresRepo._STATES:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid state {state!r}")
    return ProceduresRepo(store).for_scope(scope_id, state=state)
