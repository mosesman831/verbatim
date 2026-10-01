"""Payload → span helpers for the v3 evidence plane (SPEC_V3 §12.01, §12.08).

Span conventions mirror the v2 harvester (``verbatim/core/harvest.py``):
spans are exact UTF-8 byte ranges over the *accepted* payload, and span
identity is deterministic —

    sp_<sha256(source_id \\x00 revision \\x00 start \\x00 end \\x00 version)[:32]>

— so replaying a capture under the same parser identity yields identical
span ids (V3-15.12). A v3 envelope persists its payload as ONE whole-payload
span at ingest; the line-level helpers here give later stages (``remember``
span selection, quoting views) bounded sub-spans for textual payloads
without inventing new offset semantics.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# Parser identity recorded on every evidence-plane span (V3-15.12).
PARSER_VERSION = "v3_envelope_v1"

# Bound on line-level spans per payload — a pathological payload cannot
# mint unbounded span rows (SPEC_V3 §44 budget discipline).
MAX_LINE_SPANS = 512

_TEXT_EXACT = frozenset({
    "application/json",
    "application/jsonl",
    "application/xml",
    "application/x-ndjson",
    "application/yaml",
    "application/toml",
    "image/svg+xml",
})


def span_id_for(
    source_id: str,
    revision: int,
    start_byte: int,
    end_byte: int,
    parser_version: str = PARSER_VERSION,
) -> str:
    """Deterministic span identity — identical to the v2 harvester rule
    (SPEC_V2 §12.12, V3-15.12)."""
    digest = hashlib.sha256(
        f"{source_id}\x00{revision}\x00{start_byte}\x00{end_byte}"
        f"\x00{parser_version}".encode("utf-8")
    ).hexdigest()
    return f"sp_{digest[:32]}"


def is_textual(media_type: Optional[str]) -> bool:
    """Whether ``media_type`` denotes a byte-decodable textual payload."""
    mt = (media_type or "").split(";", 1)[0].strip().lower()
    if not mt:
        return True  # envelopes default to text/plain
    return (
        mt.startswith("text/")
        or mt in _TEXT_EXACT
        or mt.endswith("+json")
        or mt.endswith("+xml")
    )


def whole_payload_span(payload: bytes) -> tuple[int, int]:
    """The canonical capture span: the full accepted byte range (§12.08).

    Byte fidelity applies to accepted content; this span is what receipts
    and later span-handle contracts cite (§13.13)."""
    return (0, len(payload))


def line_spans(payload: bytes, *, max_spans: int = MAX_LINE_SPANS) -> list[tuple[int, int]]:
    """Line-level byte spans for textual payloads.

    Offsets are exact UTF-8 boundaries — splitting on ``b"\\n"`` can never
    bisect a multi-byte character (newline is ASCII). Blank lines are
    skipped: a span must satisfy ``end_byte > start_byte`` and empty lines
    carry no quotable content.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in payload.split(b"\n"):
        end = offset + len(line)
        if end > offset and line.strip(b" \t\r"):
            spans.append((offset, end))
            if len(spans) >= max_spans:
                break
        offset = end + 1  # advance past the '\n'
    return spans


def spans_for(
    payload: bytes,
    media_type: Optional[str] = None,
    *,
    include_lines: bool = False,
    max_line_spans: int = MAX_LINE_SPANS,
) -> list[tuple[int, int]]:
    """Candidate spans for a payload: the whole-payload span first, then
    line spans for textual payloads when requested. Order is stable and
    duplicates are removed (a one-line payload's line span equals its
    whole-payload span)."""
    whole = whole_payload_span(payload)
    out = [whole]
    if include_lines and is_textual(media_type):
        for span in line_spans(payload, max_spans=max_line_spans):
            if span != whole:
                out.append(span)
    return out
