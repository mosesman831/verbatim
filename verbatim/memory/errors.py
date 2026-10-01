"""Typed errors for the V5 consumer facade (SPEC_V5 §06.08–§06.09).

The facade never raises bare exceptions: every failure is a
``VerbatimError`` carrying an existing ``ErrorCode`` so denial stays
indistinguishable (a missing alias, a revoked owner binding, and a
foreign receipt all surface as ``NOT_FOUND_OR_UNAUTHORIZED`` — never a
hint that the object exists but is denied). Messages describe the
*class* of failure and safe recovery, never private payloads, paths,
or another namespace's identifiers.

This module adds no new error taxonomy — it is a thin naming layer so
facade call sites read as policy (``denied()``, ``conflict()``) instead
of repeating enum plumbing.
"""

from __future__ import annotations

from typing import NoReturn

from ..core.types import ErrorCode, VerbatimError

__all__ = [
    "ErrorCode",
    "VerbatimError",
    "invalid",
    "denied",
    "conflict",
    "unavailable",
    "closed",
    "forked",
    "deadline",
    "integrity",
]


def invalid(message: str) -> VerbatimError:
    """Caller-supplied input failed validation (V5-06.08)."""
    return VerbatimError(ErrorCode.VALIDATION, message)


def denied(message: str = "not found or unauthorized") -> VerbatimError:
    """Existence/authorization are publicly indistinguishable (§10.05)."""
    return VerbatimError(ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, message)


def conflict(message: str) -> VerbatimError:
    """Idempotency-key or compare-and-set conflict (V5-07.09, §14.11)."""
    return VerbatimError(ErrorCode.OPERATION_CONFLICT, message)


def unavailable(message: str) -> VerbatimError:
    """A required capability is not provisioned in this build."""
    return VerbatimError(ErrorCode.CAPABILITY_UNAVAILABLE, message)


def closed() -> VerbatimError:
    """Call on a closed/closing facade (V5-09.07) — never resumes."""
    return VerbatimError(
        ErrorCode.INVALID_TRANSITION,
        "facade is closed or closing — open a new Memory instance",
    )


def forked() -> NoReturn:  # pragma: no cover - exercised via pid check
    """Inherited live object used after fork (V5-09.10)."""
    raise VerbatimError(
        ErrorCode.INVALID_TRANSITION,
        "this Memory object was inherited across fork; open a fresh "
        "instance in the child process",
    )


def deadline(message: str) -> VerbatimError:
    """Strict-mode readiness/deadline failure (V5-08.07)."""
    return VerbatimError(ErrorCode.DEADLINE_EXCEEDED, message, retryable=True)


def integrity(message: str) -> VerbatimError:
    """Persisted state failed a re-derivation or consistency check."""
    return VerbatimError(ErrorCode.INTEGRITY, message)
