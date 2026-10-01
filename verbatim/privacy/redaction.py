"""Placeholder substitution for sensitive values (SPEC_V3 §35.01, §35.07).

``redact_view`` rewrites one accepted view's bytes, replacing each detected
or owner-declared sensitive span with a typed placeholder
(``[VAULT:<CLASS>:<token>]``) and sealing the removed value into the vault:

- ``S2``/``S3`` spans are sealed via :func:`verbatim.privacy.vault.seal` —
  the placeholder stays searchable through a ``vault_refs`` row while the
  exact bytes exist only as ciphertext (§35.02, §35.04).
- ``S4`` spans are zero-retention (§35.07): they are replaced and recorded
  as non-content redaction metadata, but *no* vault entry, no searchable
  ref, and no unkeyed digest of the value is persisted.
- ``S1`` spans are ordinary evidence — passed through untouched and
  reported as ``skipped`` so callers see the class decision honestly.
- ``redaction_spans`` records *distinct* original and accepted byte
  offsets (§35.07): quotations cite the accepted view, and hydrated
  renderings are composed disclosures, never original contiguous
  quotations.

The detector protocol is ``callable(text: str) -> [(start, end,
sensitivity)]`` over the decoded UTF-8 text with character offsets;
``declared`` spans use the same units and are recorded as
``owner_declared`` (§35.01). Overlapping or out-of-bounds spans are a
``VALIDATION`` failure — never silently merged.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import VerbatimConfig
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import DetectionBasis, SensitivityClass
from ..storage import repos_v3
from . import vault as _vault

Detector = Callable[[str], list[tuple[int, int, str]]]

_MAX_PLACEHOLDER_ATTEMPTS = 4


@dataclass(frozen=True)
class Redaction:
    """One applied substitution inside the accepted view."""

    sensitivity: str
    placeholder: str
    entry_id: Optional[str]
    detection: str
    orig_start: int
    orig_end: int
    accepted_start: int
    accepted_end: int


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of ``redact_view``: the accepted view bytes plus the
    non-content bookkeeping for every substitution (§35.07)."""

    view_id: str
    accepted: bytes
    redactions: tuple[Redaction, ...] = ()
    skipped: tuple[tuple[int, int, str], ...] = ()

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(r.placeholder for r in self.redactions)


def _char_to_byte_offsets(text: str) -> list[int]:
    """Prefix table: char index → UTF-8 byte offset (index ``len(text)``
    is the byte length)."""
    offsets = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        acc += len(ch.encode("utf-8"))
        offsets[i + 1] = acc
    return offsets


def _normalize_spans(
    detected: list[tuple[int, int, str]],
    declared: list[tuple[int, int, str]],
    n_chars: int,
) -> list[tuple[int, int, SensitivityClass, str]]:
    """Validate + merge span lists; each item is
    ``(char_start, char_end, sensitivity, detection_basis)``."""
    spans: list[tuple[int, int, SensitivityClass, str]] = []
    for s, e, sens in detected:
        spans.append((s, e, SensitivityClass(sens), DetectionBasis.DETECTED))
    for s, e, sens in declared:
        spans.append(
            (s, e, SensitivityClass(sens), DetectionBasis.OWNER_DECLARED)
        )
    spans.sort(key=lambda t: (t[0], t[1]))
    prev_end = -1
    for s, e, sens, _det in spans:
        if not (0 <= s < e <= n_chars):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"redaction span ({s},{e}) outside view bounds {n_chars}",
            )
        if s < prev_end:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "overlapping redaction spans are not substitutable",
            )
        prev_end = e
    return spans


def _placeholder(scope_id: str, sensitivity: SensitivityClass,
                 entry_id: str) -> str:
    return f"[VAULT:{sensitivity.value.upper()}:{entry_id[:8]}]"


def redact_view(
    conn: sqlite3.Connection,
    scope_id: str,
    view_id: str,
    content: bytes,
    *,
    cfg: VerbatimConfig,
    detector: Optional[Detector] = None,
    declared: Optional[list[tuple[int, int, str]]] = None,
    key_provider: Optional[Callable[..., Any]] = None,
    revision: int = 1,
    created_event: int = 0,
) -> RedactionResult:
    """Substitute sensitive spans in one view with vault placeholders.

    ``content`` is the original view bytes (UTF-8). The returned
    ``accepted`` bytes are what the engine may persist and quote — the
    removed values exist only inside vault ciphertext (or nowhere, for
    ``S4``). ``redaction_spans`` rows record both offset spaces.
    """
    require_id(scope_id, "scope_id")
    require_id(view_id, "view_id")
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise VerbatimError(ErrorCode.VALIDATION, "view content must be bytes")
    try:
        text = bytes(content).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, "view content is not valid UTF-8"
        ) from exc

    detected: list[tuple[int, int, str]] = []
    if detector is not None:
        raw = detector(text)
        if raw is None:
            raw = []
        for item in raw:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "detector must return (start, end, sensitivity) triples",
                )
            detected.append((int(item[0]), int(item[1]), str(item[2])))
    decl = [
        (int(s), int(e), str(sens)) for s, e, sens in (declared or [])
    ]
    spans = _normalize_spans(detected, decl, len(text))
    offsets = _char_to_byte_offsets(text)
    original = bytes(content)

    out = bytearray()
    cursor_b = 0
    redactions: list[Redaction] = []
    skipped: list[tuple[int, int, str]] = []

    for cs, ce, sens, det in spans:
        bs, be = offsets[cs], offsets[ce]
        if sens == SensitivityClass.S1:
            # Ordinary evidence (§35 table): left in place, reported so the
            # class decision is visible rather than silently dropped.
            skipped.append((bs, be, sens.value))
            continue

        out += original[cursor_b:bs]

        if sens == SensitivityClass.S4:
            # Zero retention (§35.07): redacted before persistence — no
            # vault entry, no searchable ref, no keyed/unkeyed digest of
            # the secret. Only non-content span metadata is recorded.
            placeholder = f"[VAULT:S4:{new_id()[:8]}]"
            entry_id: Optional[str] = None
        else:
            # S2/S3: seal the exact bytes; the typed placeholder stays
            # searchable through vault_refs while the value lives only as
            # ciphertext (§35.02, §35.04).
            value = original[bs:be]
            ph_bytes: Optional[bytes] = None
            entry_id = None
            for _attempt in range(_MAX_PLACEHOLDER_ATTEMPTS):
                eid = new_id()
                placeholder = _placeholder(scope_id, sens, eid)
                try:
                    _vault.seal(
                        conn,
                        scope_id,
                        value,
                        sensitivity=sens.value,
                        placeholder=placeholder,
                        cfg=cfg,
                        revision=revision,
                        detection=det.value,
                        key_provider=key_provider,
                        entry_id=eid,
                        created_event=created_event,
                    )
                    entry_id = eid
                    ph_bytes = placeholder.encode("utf-8")
                    break
                except sqlite3.IntegrityError:
                    # (scope_id, placeholder) unique collision — mint a
                    # fresh token for the same entry attempt.
                    continue
            if entry_id is None or ph_bytes is None:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "could not allocate a unique vault placeholder",
                )
            placeholder = ph_bytes.decode("utf-8")
            repos_v3.insert(
                conn,
                "vault_refs",
                {
                    "placeholder": placeholder,
                    "entry_id": entry_id,
                    "scope_id": scope_id,
                    "view_id": view_id,
                    "start_byte": len(out),
                    "end_byte": len(out) + len(ph_bytes),
                },
            )

        accepted_start = len(out)
        out += placeholder.encode("utf-8")
        accepted_end = len(out)
        cursor_b = be

        repos_v3.insert(
            conn,
            "redaction_spans",
            {
                "scope_id": scope_id,
                "view_id": view_id,
                "orig_start": bs,
                "orig_end": be,
                "accepted_start": accepted_start,
                "accepted_end": accepted_end,
                "sensitivity": sens.value,
                "entry_id": entry_id,
            },
        )
        redactions.append(
            Redaction(
                sensitivity=sens.value,
                placeholder=placeholder,
                entry_id=entry_id,
                detection=det.value,
                orig_start=bs,
                orig_end=be,
                accepted_start=accepted_start,
                accepted_end=accepted_end,
            )
        )

    out += original[cursor_b:]
    return RedactionResult(
        view_id=view_id,
        accepted=bytes(out),
        redactions=tuple(redactions),
        skipped=tuple(skipped),
    )


def spans_for_view(
    conn: sqlite3.Connection, scope_id: str, view_id: str
) -> list[dict[str, Any]]:
    """Non-content redaction bookkeeping for one view (§35.07)."""
    return repos_v3.query(
        conn, "redaction_spans", {"scope_id": scope_id, "view_id": view_id}
    )


def refs_for_view(
    conn: sqlite3.Connection, scope_id: str, view_id: str
) -> list[dict[str, Any]]:
    """Searchable placeholder refs inside one accepted view."""
    return repos_v3.query(
        conn, "vault_refs", {"scope_id": scope_id, "view_id": view_id}
    )


def lookup_placeholder(
    conn: sqlite3.Connection, scope_id: str, placeholder: str
) -> Optional[dict[str, Any]]:
    """Resolve a placeholder token back to its vault_refs row (searchable
    view path — never to the value)."""
    return repos_v3.get(
        conn, "vault_refs", {"scope_id": scope_id, "placeholder": placeholder}
    )
