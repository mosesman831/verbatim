"""Evidence plane (SPEC_V3 §12–§13, §43): universal source envelopes,
trajectory capture, capture receipts, and replay manifests.

Frozen contract (docs/v3_contracts.md "Envelope-ingest interface"):

    ingest_envelope(conn, store, envelope, *, authorization=None) -> CaptureReceipt

One envelope commits ``sources`` + ``source_revisions`` + ``spans`` +
``source_envelopes`` + screening + receipt atomically; agent-authored text
is never attributed to the human principal.
"""

from __future__ import annotations

from .envelopes import EnvelopeV3, ingest_envelope
from .receipts import CaptureReceipt, mint_receipt, verify_receipt
from .replay import build_replay_manifest
from .trajectories import (
    add_step,
    anchor,
    complete,
    get_trajectory,
    record_trajectory,
    steps,
)

__all__ = [
    "EnvelopeV3",
    "CaptureReceipt",
    "ingest_envelope",
    "mint_receipt",
    "verify_receipt",
    "record_trajectory",
    "add_step",
    "anchor",
    "complete",
    "get_trajectory",
    "steps",
    "build_replay_manifest",
]
