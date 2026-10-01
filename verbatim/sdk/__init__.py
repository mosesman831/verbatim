"""Verbatim capture SDK — the §13 depth-1 integration path.

Any agent host embeds :class:`CaptureClient` against a local ``Store``:
sessions, §11.11 capture authorizations, agent-submitted sources,
host-attested envelopes, trajectory steps, checker outcomes, and the
``episode_build`` pipeline trigger — one engine, one authorization model,
one receipt contract (§13.01). ``EnvelopeBuilder`` plus the
``step_from``/``outcome_from`` normalizers construct the frozen §12
contracts from typed or plain-dict input.
"""

from .capture import CaptureClient, SESSION_KINDS
from .envelope import (
    SDK_VERSION,
    EnvelopeBuilder,
    checker_from,
    checker_dict,
    environment_from,
    outcome_descriptor,
    outcome_from,
    perspective_from,
    step_from,
)

__all__ = [
    "CaptureClient",
    "SESSION_KINDS",
    "SDK_VERSION",
    "EnvelopeBuilder",
    "checker_from",
    "checker_dict",
    "environment_from",
    "outcome_descriptor",
    "outcome_from",
    "perspective_from",
    "step_from",
]
