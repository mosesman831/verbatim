"""Proposition grounding for the inspection surface (``grounding/v1`` —
SPEC_V5 §30.6, V5-30.01/30.18).

T2 proposition extraction is not implemented in this build; the honest
inspection view is therefore derived from what IS durable: the T1
``enrichment``/``entity_postings`` rows the pipeline already wrote, and
the ``claim_evidence`` span pins of claims derived from the source.
Each record is proposition-shaped — an assertion about source bytes —
and is *grounded* only when every pin it carries re-verifies against
the retained, integrity-checked payload:

- ``byte_span`` pins (identifier/entity/temporal mentions) verify when
  ``payload[revision][start:end] == value.encode("utf-8")`` — the same
  byte-exact contract the extractor pins (V5-30.17).
- ``span`` pins (claim evidence locators) verify when the span's stored
  ``excerpt_hmac`` matches a fresh profile-keyed digest of the payload
  slice — the kernel's own pin check, recomputed at read.

A record whose producer cannot pin it, whose pins do not verify, or
whose source bytes are gone (purged) reports
``outcome="unsupported_extraction"`` — it stays visible in inspection
and counted in ``proposition_stats`` (V5-30.01: stored unsupported,
never delivered as fact). Nothing here fabricates a T2 extraction; the
view is honest about which producer emitted each record.

This module is pure: no store access, no clock. The caller supplies the
already-authorized rows and a ``payload`` callable that returns the
revision's verified bytes (or ``None``/``b""`` when absent/purged).
"""

from __future__ import annotations

import hmac as _hmac
from typing import Any, Callable, Iterable, Mapping, Optional

#: Version pin for the grounding view (V5-30.18).
GROUNDING_VERSION = "grounding/v1"

OUTCOME_GROUNDED = "grounded"
OUTCOME_UNSUPPORTED = "unsupported_extraction"


def _pin_ok(
    pin: Mapping[str, Any],
    payload: Callable[[int], Optional[bytes]],
    hmac: Callable[[bytes], bytes],
) -> bool:
    """Verify one pin against the retained revision bytes."""
    try:
        rev = int(pin.get("revision"))
        start = int(pin.get("start"))
        end = int(pin.get("end"))
    except (TypeError, ValueError):
        return False
    body = payload(rev)
    if not body or not (0 <= start < end <= len(body)):
        return False
    kind = pin.get("kind")
    if kind == "byte_span":
        expected = pin.get("value")
        if not isinstance(expected, str):
            return False
        return _hmac.compare_digest(
            body[start:end], expected.encode("utf-8")
        )
    if kind == "span":
        digest = pin.get("excerpt_hmac")
        if isinstance(digest, str):
            # Wire form is hex-encoded (controls hex-encodes for
            # JSON-safe to_dict()); unhexlify before compare.
            try:
                digest = bytes.fromhex(digest)
            except ValueError:
                return False
        if not isinstance(digest, (bytes, bytearray)) or not digest:
            return False
        return _hmac.compare_digest(hmac(body[start:end]), bytes(digest))
    return False


def _proposition(
    kind: str,
    *,
    source_id: str,
    revision: int,
    producer: str,
    pins: Iterable[Mapping[str, Any]],
    payload: Callable[[int], Optional[bytes]],
    hmac: Callable[[bytes], bytes],
    **detail: Any,
) -> dict:
    pin_list = [dict(p) for p in pins]
    grounded = bool(pin_list) and all(
        _pin_ok(p, payload, hmac) for p in pin_list
    )
    rec = {
        "kind": kind,
        "source_id": source_id,
        "revision": revision,
        "producer": producer,
        "pins": pin_list,
        "grounded": grounded,
        "outcome": (
            OUTCOME_GROUNDED if grounded else OUTCOME_UNSUPPORTED
        ),
    }
    rec.update({k: v for k, v in detail.items() if v is not None})
    return rec


def _byte_pin(source_id: str, revision: int, start: Any, end: Any, value: str) -> dict:
    return {
        "kind": "byte_span",
        "source_id": source_id,
        "revision": revision,
        "start": start,
        "end": end,
        "value": value,
    }


def build_propositions(
    *,
    source_id: str,
    enrichment_rows: Iterable[Mapping[str, Any]] = (),
    entity_postings: Iterable[Mapping[str, Any]] = (),
    claim_evidence: Iterable[Mapping[str, Any]] = (),
    payload: Callable[[int], Optional[bytes]],
    hmac: Callable[[bytes], bytes],
) -> list:
    """Build the proposition-shaped grounding view for one source.

    ``enrichment_rows`` are the stored ``enrichment`` table dicts
    (``revision``/``producer``/``fields`` — ``fields`` carrying the
    producer-serialized ``identifiers``/``entities``/``temporal``
    records). ``entity_postings`` are the grouped mention index rows
    (``entity``/``entity_kind``/``offsets``/``revision``); they fill in
    mentions the row payload did not serialize. ``claim_evidence`` maps
    each derived claim to its span pins on this source
    (``claim_id``/``span_id``/``start_byte``/``end_byte``/
    ``excerpt_hmac``/``revision``/``evidence_role``).

    Every returned record carries ``pins``, ``grounded``, and
    ``outcome`` — the assertions the inspection surface iterates.
    """
    props: list = []
    covered_mentions: set = set()  # (value, start, end) already pinned

    for row in enrichment_rows or ():
        revision = row.get("revision")
        producer = str(row.get("producer") or "")
        fields = row.get("fields") or {}
        if not isinstance(fields, Mapping):
            fields = {}

        for collection, kind in (
            ("identifiers", "identifier"),
            ("entities", "entity"),
        ):
            for mention in fields.get(collection) or ():
                if not isinstance(mention, Mapping):
                    continue
                value = mention.get("value")
                if not isinstance(value, str) or not value:
                    continue
                pin = _byte_pin(
                    source_id, int(revision or 0),
                    mention.get("start"), mention.get("end"), value,
                )
                covered_mentions.add(
                    (value, pin["start"], pin["end"])
                )
                props.append(_proposition(
                    kind,
                    source_id=source_id,
                    revision=int(revision or 0),
                    producer=producer,
                    pins=[pin],
                    payload=payload,
                    hmac=hmac,
                    value=value,
                    mention_kind=mention.get("kind"),
                ))

        temporal = fields.get("temporal")
        if isinstance(temporal, Mapping):
            expression = temporal.get("expression")
            start = temporal.get("start")
            end = temporal.get("end")
            if isinstance(expression, str) and expression:
                pin = _byte_pin(
                    source_id, int(revision or 0), start, end, expression
                )
                props.append(_proposition(
                    "temporal",
                    source_id=source_id,
                    revision=int(revision or 0),
                    producer=producer,
                    pins=[pin],
                    payload=payload,
                    hmac=hmac,
                    value=expression,
                    resolution={
                        "precision": temporal.get("precision"),
                        "status": temporal.get("status"),
                        "event_at": temporal.get("event_at"),
                        "event_end": temporal.get("event_end"),
                        "anchor_at": temporal.get("anchor_at"),
                    },
                ))

    # entity_postings fill gaps the row payload never serialized — a
    # posting offset is itself a byte pin for the grouped value.
    for posting in entity_postings or ():
        if not isinstance(posting, Mapping):
            continue
        value = posting.get("entity")
        if not isinstance(value, str) or not value:
            continue
        revision = int(posting.get("revision") or 0)
        ekind = str(posting.get("entity_kind") or "")
        kind = "identifier" if "identifier" in ekind else "entity"
        pins = [
            _byte_pin(source_id, revision, off[0], off[1], value)
            for off in (posting.get("offsets") or [])
            if isinstance(off, (list, tuple)) and len(off) == 2
            and (value, off[0], off[1]) not in covered_mentions
        ]
        if not pins:
            continue
        props.append(_proposition(
            kind,
            source_id=source_id,
            revision=revision,
            producer=str(posting.get("producer") or "entity_postings"),
            pins=pins,
            payload=payload,
            hmac=hmac,
            value=value,
            mention_kind=ekind or None,
        ))

    # Derived claims: each is a proposition grounded by its evidence
    # span pins on this source.
    by_claim: dict = {}
    for ev in claim_evidence or ():
        if not isinstance(ev, Mapping):
            continue
        cid = ev.get("claim_id")
        if not cid:
            continue
        by_claim.setdefault(str(cid), []).append(ev)
    for cid in sorted(by_claim):
        rows = by_claim[cid]
        pins = [
            {
                "kind": "span",
                "span_id": ev.get("span_id"),
                "source_id": source_id,
                "revision": int(ev.get("source_revision") or ev.get("revision") or 0),
                "start": ev.get("start_byte"),
                "end": ev.get("end_byte"),
                "excerpt_hmac": ev.get("excerpt_hmac"),
            }
            for ev in rows
        ]
        roles = sorted(
            {str(ev.get("evidence_role")) for ev in rows if ev.get("evidence_role")}
        )
        props.append(_proposition(
            "claim",
            source_id=source_id,
            revision=int(rows[0].get("source_revision") or 0),
            producer=str(rows[0].get("harvester_version") or "harvest"),
            pins=pins,
            payload=payload,
            hmac=hmac,
            claim_id=cid,
            evidence_roles=roles or None,
            suppressed=bool(rows[0].get("suppressed")),
        ))

    return props


def proposition_stats(propositions: Iterable[Mapping[str, Any]]) -> dict:
    """Counts the inspection surface reports — unsupported records are
    visible, never silently folded into the grounded set (V5-30.01)."""
    props = list(propositions or ())
    grounded = sum(1 for p in props if p.get("grounded"))
    return {
        "version": GROUNDING_VERSION,
        "total": len(props),
        "grounded": grounded,
        "unsupported": len(props) - grounded,
    }


__all__ = [
    "GROUNDING_VERSION",
    "OUTCOME_GROUNDED",
    "OUTCOME_UNSUPPORTED",
    "build_propositions",
    "proposition_stats",
]
