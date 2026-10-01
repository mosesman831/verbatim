"""Three-valued procedure applicability (SPEC_V3 §21.07, §22.05, §22.15).

``check_applicability`` evaluates a procedure's recorded
applicability/precondition :class:`TypedCondition` set plus its
environment fingerprint against the caller's current environment and
proposed bindings. The verdict is three-valued — ``applies`` /
``does_not_apply`` / ``unknown`` — and **unknown is never a wildcard**
(V3-22.15): missing environment information, missing bindings, and
unevaluable conditions all yield ``unknown``, never a silent pass.

Environment drift is a first-class verdict (V3-22.05): a fingerprint
mismatch yields ``does_not_apply`` with the ``environment_mismatch``
reason — there is no ``ENVIRONMENT_MISMATCH`` error code on this path; the
caller receives the honest verdict plus reasons instead of an exception
that would hide *which* dimension disagreed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import ApplicabilityVerdict, EnvironmentFingerprint
from ..storage.repos import _json_parse, _row
from .compiler import environment_map

_CONDITION_OPS = ("eq", "neq", "in", "present")


@dataclass(frozen=True)
class ApplicabilityResult:
    """The applicability verdict plus *why* — never just a bool."""

    verdict: ApplicabilityVerdict
    reasons: tuple[str, ...] = ()
    missing_bindings: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, ApplicabilityVerdict):
            object.__setattr__(self, "verdict", ApplicabilityVerdict(self.verdict))

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "missing_bindings": list(self.missing_bindings),
            "mismatched": list(self.mismatched),
            "unknown": list(self.unknown),
        }


def _env_map_of(environment: Any) -> dict[str, Any]:
    """Normalize an environment argument to the flat condition map."""
    if environment is None:
        return {}
    if isinstance(environment, EnvironmentFingerprint):
        return environment_map(environment)
    if isinstance(environment, dict):
        return {str(k): v for k, v in environment.items()}
    raise VerbatimError(
        ErrorCode.VALIDATION,
        "environment must be an EnvironmentFingerprint or dict",
    )


def _lookup(key: str, env: dict[str, Any],
            bindings: dict[str, Any]) -> Any:
    """Resolve a condition key → observed value, or None (unknown)."""
    if key in ("environment.fingerprint", "env.fingerprint"):
        if not env:
            return None
        return hashlib.sha256(json_dumps(env).encode("utf-8")).hexdigest()[:32]
    for prefix in ("environment.", "env."):
        if key.startswith(prefix):
            return env.get(key[len(prefix):])
    if key.startswith("binding."):
        return bindings.get(key[len("binding."):])
    # Bare keys read the environment map first, then bindings — a
    # deterministic order, never a guess at intent.
    if key in env:
        return env[key]
    return bindings.get(key)


def _eval_condition(
    cond: dict[str, Any],
    env: dict[str, Any],
    bindings: dict[str, Any],
) -> Optional[bool]:
    """Three-valued condition evaluation (V3-22.15)."""
    op = cond.get("op")
    if op not in _CONDITION_OPS:
        return None
    key = cond.get("key")
    if not isinstance(key, str) or not key:
        return None
    actual = _lookup(key, env, bindings)
    if op == "present":
        return actual is not None
    if actual is None:
        # Missing information is unknown — never a wildcard (V3-21.07).
        return None
    expected = cond.get("value")
    if op == "eq":
        return str(actual) == str(expected)
    if op == "neq":
        return str(actual) != str(expected)
    # ``in``: the expected value is a list (JSON array or comma string).
    choices: Any = expected
    if isinstance(expected, str):
        try:
            choices = safe_json_loads(expected)
        except VerbatimError:
            choices = [c.strip() for c in expected.split(",") if c.strip()]
    if not isinstance(choices, (list, tuple)):
        choices = [choices]
    return str(actual) in {str(c) for c in choices}


def check_applicability(
    conn: sqlite3.Connection,
    procedure_id: str,
    environment: Any = None,
    bindings: Optional[dict[str, Any]] = None,
) -> ApplicabilityResult:
    """Evaluate a procedure's conditions + environment fingerprint.

    ``environment`` is an :class:`EnvironmentFingerprint` (or an equivalent
    flat dict); ``bindings`` maps binding names to the host's proposed
    refill values. Returns an :class:`ApplicabilityResult` — the verdict
    plus the exact reasons, missing bindings, and mismatched keys.
    """
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?",
            (require_id(procedure_id, "procedure_id"),),
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    env = _env_map_of(environment)
    bound = dict(bindings or {})
    stored_env = _json_parse(row.get("environment_json")) or {}
    conditions = (
        (_json_parse(row.get("applicability_json")) or [])
        + (_json_parse(row.get("preconditions_json")) or [])
    )
    required = [
        b["name"]
        for b in (_json_parse(row.get("bindings_json")) or [])
        if isinstance(b, dict) and b.get("required") and b.get("name")
    ]

    reasons: list[str] = []
    mismatched: list[str] = []
    unknowns: list[str] = []

    # Environment drift check (V3-22.05): a recorded fingerprint only
    # applies to a matching environment.
    if stored_env:
        if not env:
            unknowns.append("environment")
            reasons.append("environment_not_provided")
        elif json_dumps(stored_env) != json_dumps(env):
            mismatched.append("environment")
            reasons.append("environment_mismatch")
    else:
        # No recorded environment is missing information, never a
        # wildcard match (V3-21.07).
        unknowns.append("environment")
        reasons.append("no_environment_evidence")

    missing = [b for b in required if bound.get(b) is None]
    if missing:
        reasons.append("missing_required_bindings")

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        label = str(cond.get("key") or "?")
        verdict = _eval_condition(cond, env, bound)
        if verdict is False:
            mismatched.append(label)
        elif verdict is None:
            unknowns.append(label)

    if mismatched:
        reasons.append("condition_mismatch")
        return ApplicabilityResult(
            ApplicabilityVerdict.DOES_NOT_APPLY,
            reasons=tuple(dict.fromkeys(reasons)),
            missing_bindings=tuple(missing),
            mismatched=tuple(dict.fromkeys(mismatched)),
            unknown=tuple(dict.fromkeys(unknowns)),
        )
    if missing or unknowns:
        return ApplicabilityResult(
            ApplicabilityVerdict.UNKNOWN,
            reasons=tuple(dict.fromkeys(reasons)) or ("insufficient_evidence",),
            missing_bindings=tuple(missing),
            mismatched=(),
            unknown=tuple(dict.fromkeys(unknowns)),
        )
    return ApplicabilityResult(
        ApplicabilityVerdict.APPLIES,
        reasons=("all_conditions_satisfied",),
        missing_bindings=(),
        mismatched=(),
        unknown=(),
    )
