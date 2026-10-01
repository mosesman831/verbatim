"""Broader relation discovery for unstructured claims (SPEC_V3 §18.03–04,
gap F26).

``core.policy.relate`` compares only *structured* claims — the ~12
registered predicates — and skips ``unstructured`` claims entirely, so a
verbatim quotation that flatly contradicts an existing belief never
relates to it. This module is the v3 layer V3-18.03 demands:

1. **Deterministic checks first** — candidate pairs come from the same
   scope's live claims; a pair must share an entity neighborhood (a
   common identifier or ≥2 content tokens) and show at least one surface
   contradiction signal (negation asymmetry, numeric mismatch, or an
   explicit correction marker in the new text).
2. **Optional local discriminative judgment (§42)** — when an encoder is
   provisioned, cosine similarity between excerpts acts as a second
   neighborhood signal and is recorded as confidence provenance. The
   encoder never labels a relation alone; it only corroborates.
3. **Then review** — flagged pairs produce a ``conflicts_with`` edge and
   an open ``dispute`` review with pinned expected versions. Detection
   never applies a transition; the reviewer decides (V3-18.06).

Like the supersession detector, precision is the constraint: a
contradiction signal is mandatory, pairs are budgeted, and every review
records its detection method so confidence provenance survives (V3-18.04).
"""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any, Optional

from ..core.lifecycle import read_claim_head
from ..core.types import InterpretationStatus
from ..storage.repos import EdgesRepo, ReviewsRepo
from .supersession import (
    _ENDORSE_RE,
    _NON_IDENT,
    _OBJECT_RE,
    _SUBJECT_RE,
    _claim_text,
    _content_tokens,
    _retired_idents,
)

#: Negation markers — presence in exactly one side of a topically-linked
#: pair is a contradiction signal, not proof.
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|isn't|wasn't|aren't|weren't|don't|doesn't"
    r"|didn't|won't|can't|cannot|without|nor|neither|stopped|fails?"
    r" to|no more)\b",
    re.IGNORECASE,
)

#: End-of-life verbs — a negation attached to one affirms continuity
#: ("X was not retired" agrees with claims still using X), so it never
#: counts as a contradiction signal.
_STATUS_VERB_RE = re.compile(
    r"(?:retire|deprecat|remov|replac|decommission|supersed|obsolet"
    r"|sunset|eol)",
    re.IGNORECASE,
)

#: Successor endorsement lives in supersession._ENDORSE_RE — "X was
#: retired; Y is the successor" affirms a claim already using Y.

#: Explicit self-correction/update markers — the new text declares that
#: something previously believed has changed.
_CORRECTION_RE = re.compile(
    r"\b(?:actually|correction|corrected|i was wrong|turns out|update:"
    r"|scratch that|not anymore|no longer|used to)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")

#: Identifier-shaped token: contains a digit or separator, or is a long
#: capitalized word — the unstructured analogue of entity neighborhood.
_IDENT_SHAPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{2,60}")

_PAIR_BUDGET = 4
_CANDIDATE_LIMIT = 32

#: Encoder corroboration thresholds (hashing encoder's observed spread:
#: ~0.3 topical relation, <0.2 unrelated).
_SIM_NEIGHBORHOOD = 0.55
_SIM_STRONG = 0.65


def _idents(text: str) -> frozenset[str]:
    """Identifier-ish tokens — the entity-neighborhood proxy for text
    that carries no structured entity links."""
    out: set[str] = set()
    for m in _IDENT_SHAPE_RE.finditer(text):
        tok = m.group(0).strip("`'\"").lower()
        if tok in _NON_IDENT:
            continue
        raw = m.group(0)
        if (
            re.search(r"[\d_./-]", raw)
            or (len(raw) >= 6 and raw[0].isupper())
        ):
            out.add(tok)
    return frozenset(out)


def _numbers(text: str) -> frozenset[str]:
    return frozenset(_NUMBER_RE.findall(text))


def _effective_negation(text: str) -> bool:
    """True when the text carries a negation that is NOT a status-
    negation. A negation modifying an end-of-life verb affirms
    continuity ("X was not retired" agrees with claims still using X);
    a negation in the same sentence as a retirement mention resolves or
    echoes that mention ("was it retired? — it wasn't"). Neither
    contradicts a live claim."""
    for m in _NEGATION_RE.finditer(text):
        # bare "do not/don't VERB" is an imperative to the reader ("do
        # not tell the user"), not a state assertion; with a subject it
        # stays a real negation ("we do not use prod-01"). The regex
        # matches ``not`` inside "do not", so detect the ``do`` in the
        # look-back window rather than in the matched group.
        before = text[max(0, m.start() - 24):m.start()]
        if m.group(0).lower() == "don't":
            if not re.search(
                r"(?:we|i|they|you|it|he|she|one|people|teams?)\s*$",
                before, re.IGNORECASE,
            ):
                continue
        elif m.group(0).lower() == "not":
            dm = re.search(r"\bdo\s+$", before, re.IGNORECASE)
            if dm and not re.search(
                r"(?:we|i|they|you|it|he|she|one|people|teams?)\s*$",
                before[:dm.start()], re.IGNORECASE,
            ):
                continue
        after = text[m.end():m.end() + 24].lstrip()
        if _STATUS_VERB_RE.match(after):
            continue
        seg_start = max(
            text.rfind(b, 0, m.start()) + 1
            for b in (".", "!", "?", ";", "\n", ":")
        )
        seg_end = min(
            (e for e in (text.find(b, m.end())
                         for b in (".", "!", "?", ";", "\n", ":"))
             if e != -1),
            default=len(text),
        )
        seg = text[seg_start:seg_end]
        if _SUBJECT_RE.search(seg) or _OBJECT_RE.search(seg):
            continue
        return True
    return False


def _signals(new_text: str, old_text: str) -> list[str]:
    """Deterministic contradiction signals between a pair of excerpts."""
    out: list[str] = []
    if _effective_negation(new_text) != _effective_negation(old_text):
        out.append("negation_asymmetry")
    nn, on = _numbers(new_text), _numbers(old_text)
    if nn and on and nn != on:
        out.append("numeric_mismatch")
    if _CORRECTION_RE.search(new_text):
        out.append("correction_marker")
    return out


def _cosine_pair(
    encoder: Any, new_text: str, old_text: str
) -> Optional[float]:
    """Cosine between two excerpts under the provisioned encoder; None
    when the encoder cannot serve — degradation is honest, never fatal."""
    try:
        from ..embeddings.codec import Float32Codec

        blobs = encoder.encode([new_text, old_text])
        dims = encoder.dimensions
        a = Float32Codec.unpack(blobs[0], dims)
        b = Float32Codec.unpack(blobs[1], dims)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if not na or not nb:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return None


def _existing_conflict(
    conn: sqlite3.Connection, a: str, b: str
) -> bool:
    x, y = sorted((a, b))
    row = conn.execute(
        "SELECT 1 FROM edges WHERE edge_type = 'conflicts_with'"
        " AND source_kind = 'claim' AND target_kind = 'claim'"
        " AND source_id = ? AND target_id = ?"
        " AND retired_event IS NULL LIMIT 1",
        (x, y),
    ).fetchone()
    return row is not None


def propose_unstructured_relations(
    store: Any,
    claim_id: str,
    scope_id: str,
    *,
    encoder: Any = None,
) -> list[str]:
    """Relation discovery for a newly admitted *unstructured* claim.

    Returns created edge/review ids; ``[]`` when the claim is structured
    (``relate`` already owns that path) or no pair clears the gates.
    """
    with store.read() as conn:
        head = read_claim_head(conn, claim_id)
        if head is None:
            return []
        if head.interpretation_status != InterpretationStatus.UNSTRUCTURED:
            return []
        new_text = _claim_text(store, conn, claim_id, head.revision)
        if not new_text:
            return []
        new_topics = _content_tokens(new_text)
        new_idents = _idents(new_text)

        cands = conn.execute(
            "SELECT c.claim_id, r.revision FROM claims c"
            " JOIN claim_revisions r ON r.claim_id = c.claim_id"
            "   AND r.recorded_until IS NULL"
            " WHERE c.scope_id = ? AND c.claim_id <> ?"
            "   AND r.state IN ('active','disputed','pending')"
            " LIMIT ?",
            (scope_id, claim_id, _CANDIDATE_LIMIT),
        ).fetchall()
        texts = {cid: _claim_text(store, conn, cid, rev) for cid, rev in cands}
        existing = {
            (cid,) for cid, rev in cands
            if _existing_conflict(conn, claim_id, cid)
        }

    retired_new = set(_retired_idents(new_text))
    pairs: list[tuple[str, int, list[str], str, Optional[float]]] = []
    for cid, rev in cands:
        text = texts.get(cid)
        if not text or (cid,) in existing or len(pairs) >= _PAIR_BUDGET:
            continue
        shared = new_idents & _idents(text)
        retired_old = set(_retired_idents(text))
        if retired_new & retired_old:
            continue  # both sides declare the same end-of-life — agreement
        if (
            retired_new
            and not (retired_new & shared)
            and _ENDORSE_RE.search(new_text)
        ):
            continue  # "X retired; Y is the successor" affirms a Y-user
        signals = _signals(new_text, text)
        if not signals:
            continue
        sim = _cosine_pair(encoder, new_text, text) if encoder else None
        neighborhood = bool(shared) or len(
            new_topics & _content_tokens(text)
        ) >= 2
        if not neighborhood and not (sim is not None and sim >= _SIM_STRONG):
            continue
        if sim is not None and sim < _SIM_NEIGHBORHOOD and not neighborhood:
            continue
        method = "deterministic_unstructured"
        if sim is not None:
            method += "+encoder"
        pairs.append((cid, rev, signals, method, sim))

    if not pairs:
        return []

    out: list[str] = []
    with store.tx() as conn:
        edges = EdgesRepo(store)
        reviews = ReviewsRepo(store)
        for cid, rev, signals, method, sim in pairs:
            a, b = sorted((claim_id, cid))
            out.append(
                edges.add(
                    conn, scope_id, "claim", a, "claim", b, "conflicts_with"
                )
            )
            reason = (
                f"unstructured pair shares an entity neighborhood and shows "
                f"{', '.join(signals)}"
            )
            if sim is not None:
                reason += f"; cosine {sim:.2f} under {encoder.encoder_id}"
            out.append(
                reviews.create(
                    conn,
                    scope_id,
                    {
                        "effect": "dispute",
                        "claim_id": claim_id,
                        "conflict_with_claim_id": cid,
                        "pair_label": "incompatible",
                        "change_signal": signals[0],
                        "method": method,
                        "confidence": round(sim, 3) if sim is not None else None,
                        "reason": reason,
                    },
                    {claim_id: head.revision, cid: rev},
                    dedup=True,
                )
            )
    return out


__all__ = ["propose_unstructured_relations"]
