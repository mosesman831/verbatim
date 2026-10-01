"""Per-encoder similarity support calibration — ``support_calibration/v1``
(SPEC_V5 §31.3, V5-31.07/31.08; worker contract docs/v5_contracts.md §8).

``ranking/v1`` delivers a fused, *normalized* score; the support verdict
needs the encoder-native similarity value to decide whether a
similarity-only candidate can carry support at all. That bound is a
calibration question, not a ranking one: it must be fitted per pinned
encoder identity (``encoder_id`` — backend + model + algorithm revision)
on development data and validated on held-out no-answer/near-miss
slices — never a global threshold copied between encoders
(V5-31.07/31.08).

This module is the fitted artifact: a small immutable table keyed by
``encoder_id``, plus the deterministic dev slice the numbers were
measured on so the fit stays auditable and re-runnable (``validate``).
Each entry records the floor in the encoder's native score space
(float32-packed cosine, exactly as ``source_lane.vector_candidates``
computes it) and the measured slice statistics behind it — the
``confidence`` family V5-31.08 asks for, not a bare cosine.

Fitted contract for ``hashing:subword-ngram:v1`` — the stdlib
lexical-subword encoder. The dev slice below measured:

- ``no_answer`` (unrelated queries vs the dev corpus): max 0.237
- ``near_miss`` (topically adjacent or morphologically mirrored queries
  the source does not answer — the false-friend envelope): max 0.589
- ``support`` (content genuinely present, including morphological
  variants the term floor cannot see): 0.465–1.00, with the mid band
  *inside* the false-friend envelope

The bands overlap: under a lexical-subword hasher no floor admits
mid-band paraphrases while rejecting the measured false friends, so the
honest fit is conservative — **0.75** sits ~0.16 above the worst
measured false friend, in the band only near-verbatim subword identity
reaches. Similarity corroborates support there; below it a
similarity-only candidate is the least-bad hit V5-31.09 forbids.

Unknown or unlisted encoders return ``None`` — fail-closed: similarity
alone is never support under an encoder with no fitted calibration.

Fitted contract for ``artifact:hash-distilled:v1`` — the locally built
VBV1 word table whose rows are the hashing encoder's per-term vectors
("distilled hashing", ``eval/v6/neural.py::build_dev_artifact``). Its
geometry is *not* the hashing geometry: a text embeds as the pooled
mean of its known-token rows (skip-OOV), so the measured bands differ
and the floor is FIT, never copied from 0.75 (V5-31.07 — a floor
fitted on a different encoder's space is not a fit). The dev slice
below measured, through ``eval/v6/neural.py::fit_calibration``:

- ``no_answer``: max 0.650 (shared stop-word mass only)
- ``near_miss``: max 0.707 — subword false friends vanish under
  word-level pooling, but near-verbatim paraphrases reduced to their
  shared known tokens still reach 1/√2
- ``support``: -0.068–1.00 — pairs whose content survives the pinned
  vocabulary reach verbatim identity (1.00); pairs reduced to OOV fall
  to noise band and honestly cannot be corroborated by similarity

**0.85** sits ~0.14 above the worst measured false friend, in the band
only near-verbatim word-identity reaches — the same conservative shape
as the hashing fit: similarity corroborates near-duplicate content;
below the floor a similarity-only candidate is the least-bad hit
V5-31.09 forbids. Any new artifact revision (new ``encoder_id``) must
re-run the fit — the entry retires automatically with the revision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

#: Version pin for this calibration contract — bumped on any change to
#: the table, the dev slice, or the floor semantics.
CALIBRATION_VERSION = "support_calibration/v1"

#: Query classes a fitted similarity floor may support. ``identifier``
#: and ``no_answer_likely`` are deliberately absent: exact-identifier
#: support is the posting contract (V5-30.17), and a no-answer class
#: suppresses every candidate before calibration is consulted
#: (V5-31.09). A class outside this list resolves to ``None`` — a floor
#: the fit never validated never applies (fail-closed).
SIMILARITY_ELIGIBLE_CLASSES: Tuple[str, ...] = (
    "factual",
    "entity",
    "temporal",
    "preference",
    "procedural",
)


@dataclass(frozen=True)
class SimilarityCalibration:
    """Fitted similarity-support record for one pinned ``encoder_id``.

    ``floor`` is the encoder-native score bound: a similarity-only
    candidate supports its query only when its raw similarity reaches
    it. ``None`` declares "no fitted floor" — similarity never carries
    support under that encoder. ``classes`` names the query classes the
    floor was fitted and validated for. ``measured`` carries the
    dev-slice statistics the floor was fitted from (per-slice envelope
    bounds) so the delivered detail can name the calibration family.
    """

    encoder_id: str
    floor: Optional[float]
    classes: Tuple[str, ...]
    fitted_on: str
    measured: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------
# dev slice — the pinned development data behind the fitted floors
# ---------------------------------------------------------------------
#
# (slice, query, retained-source text). The slices are exactly the
# contract V5-31.08 names: ``no_answer`` and ``near_miss`` are the
# held-out validation slices (a fitted floor MUST sit above every score
# they produce); ``support`` pairs carry genuinely present content and
# bound the band a floor may admit. All scores are cosines over the
# float32-packed vectors, recomputed by ``validate`` — the numbers in
# ``measured`` below were produced by that exact path.

_DEV_SLICE: Tuple[Tuple[str, str, str], ...] = (
    # -- no-answer: unrelated content; any hit is a least-bad hit ------
    ("no_answer", "zyxwv nonexistent-term", "The project linter is ruff."),
    ("no_answer", "what is the wifi password", "the deploy command is deploy-v2"),
    ("no_answer", "kubernetes pod restart count", "grandma's apple pie recipe"),
    ("no_answer", "docker image prune schedule", "quarterly budget review friday"),
    ("no_answer", "mysql root credentials", "sunny weather expected tomorrow"),
    ("no_answer", "quantum entropy budget", "the deploy command is deploy-v2"),
    ("no_answer", "where did I park the car", "The project linter is ruff."),
    # -- near-miss: adjacent/mirrored subwords, unsupported content ----
    ("near_miss", "the deployer commands", "the deploy command is deploy-v2"),
    ("near_miss", "the debugged builds", "the build debugger is pdb"),
    ("near_miss", "deployed commanders deploying", "the deploy command is deploy-v2"),
    ("near_miss", "the databases migrated", "the database migration ran at noon"),
    ("near_miss", "the deployed", "the deploy command is deploy-v2"),
    ("near_miss", "passwords hunter", "the wifi password is hunter2"),
    ("near_miss", "timeouts configuring", "the timeout configuration uses 30s"),
    ("near_miss", "debuggers builds", "the build debugger is pdb"),
    ("near_miss", "migrating databases", "the database migration ran at noon"),
    ("near_miss", "approved vendors", "alice approves all vendor contracts"),
    ("near_miss", "vendor payment terms", "alice approves all vendor contracts"),
    ("near_miss", "password rotation policy", "the wifi password is hunter2"),
    ("near_miss", "linted", "The project linter is ruff."),
    # -- support: content genuinely present in the retained source ----
    ("support", "the deploy command is deploy-v2", "the deploy command is deploy-v2"),
    ("support", "the build debugger", "the build debugger is pdb"),
    ("support", "the timeout config", "the timeout configuration uses 30s"),
    ("support", "migrated database", "the database migration ran at noon"),
    ("support", "jsonserializer", "the json serializer is stdlib"),
    ("support", "deployv2", "the deploy command is deploy-v2"),
    ("support", "postgresql", "the postgre database is v15"),
)

#: Encoder identities the table has fitted floors for. Bumping the
#: encoder's algorithm revision (a new ``encoder_id``) retires the entry
#: automatically — vectors under the new identity have no fitted floor
#: until one is measured and pinned here.
CALIBRATED_ENCODERS: Mapping[str, SimilarityCalibration] = MappingProxyType(
    {
        "hashing:subword-ngram:v1": SimilarityCalibration(
            encoder_id="hashing:subword-ngram:v1",
            floor=0.75,
            classes=SIMILARITY_ELIGIBLE_CLASSES,
            fitted_on=(
                "support_calibration/v1 dev slice "
                f"({len(_DEV_SLICE)} pairs: no_answer/near_miss/support), "
                "cosine over float32le vectors"
            ),
            measured={
                "no_answer_max": 0.23694664100467644,
                "near_miss_max": 0.5893673394803589,
                "support_min": 0.465017783618663,
                "support_max": 1.0,
            },
        ),
        # Fitted via eval/v6/neural.py::fit_calibration — the artifact's
        # pooled word-mean geometry is measured, never assigned the
        # hashing floor (V6-03.09).
        "artifact:hash-distilled:v1": SimilarityCalibration(
            encoder_id="artifact:hash-distilled:v1",
            floor=0.85,
            classes=SIMILARITY_ELIGIBLE_CLASSES,
            fitted_on=(
                "support_calibration/v1 dev slice "
                f"({len(_DEV_SLICE)} pairs: no_answer/near_miss/support) "
                "through the built VBV1 artifact "
                "(eval/v6/neural.py::fit_calibration), cosine over "
                "float32le vectors"
            ),
            measured={
                "no_answer_max": 0.6496639325364868,
                "near_miss_max": 0.7071067877508127,
                "support_min": -0.06798178615129369,
                "support_max": 1.0,
                "support_admitted": 2.0,
            },
        ),
    }
)


def _encoder_key(encoder_id: Any) -> Optional[str]:
    if isinstance(encoder_id, str) and encoder_id.strip():
        return encoder_id.strip()
    return None


def calibration_for(encoder_id: Any) -> Optional[SimilarityCalibration]:
    """The fitted record for ``encoder_id`` — ``None`` when uncalibrated."""
    key = _encoder_key(encoder_id)
    return CALIBRATED_ENCODERS.get(key) if key else None


def is_calibrated(encoder_id: Any) -> bool:
    """True when a fitted floor exists for this encoder identity."""
    rec = calibration_for(encoder_id)
    return rec is not None and rec.floor is not None


def similarity_floor(encoder_id: Any, query_class: Any = None) -> Optional[float]:
    """The calibrated similarity-support floor, or ``None``.

    ``None`` is the fail-closed answer: the encoder is unknown to the
    table, has no fitted floor, or the query class was outside the fitted
    set — in every case similarity alone must not carry support
    (V5-31.07: never a threshold copied from another encoder).
    """
    rec = calibration_for(encoder_id)
    if rec is None or rec.floor is None:
        return None
    if query_class is not None and rec.classes:
        cls = getattr(query_class, "value", query_class)
        if str(cls) not in rec.classes:
            return None
    return rec.floor


def _cosine_pair(encoder: Any, query: str, doc: str) -> Optional[float]:
    """Encoder-native cosine for one dev pair — the same float32-packed,
    float64-reference math ``source_lane.vector_candidates`` applies."""
    from ..embeddings.codec import Float32Codec

    try:
        # ``dimensions`` may RAISE for fail-closed encoders (the artifact
        # backend reports unknown dims until verification passes) — an
        # unverifiable encoder simply yields no pair score.
        dims = int(getattr(encoder, "dimensions", 0) or 0)
    except Exception:
        return None
    if dims < 1:
        return None
    try:
        packed = encoder.encode([query, doc])
    except Exception:
        return None
    if not isinstance(packed, (list, tuple)) or len(packed) != 2:
        return None
    try:
        qv = Float32Codec.unpack(packed[0], dims)
        dv = Float32Codec.unpack(packed[1], dims)
    except Exception:
        return None
    qn = math.fsum(x * x for x in qv)
    dn = math.fsum(x * x for x in dv)
    if qn <= 0.0 or dn <= 0.0:
        return None
    return math.fsum(a * b for a, b in zip(qv, dv)) / math.sqrt(qn * dn)


def validate(encoder: Any, *,
             slice_pairs: Sequence[Tuple[str, str, str]] = _DEV_SLICE
             ) -> dict:
    """Re-run the pinned dev slice through ``encoder`` and report the fit.

    Returns a receipt-loggable dict: per-slice score envelopes, the
    fitted floor for this encoder (``None`` when uncalibrated), and
    ``separates`` — whether every measured no-answer/near-miss pair sits
    strictly below the floor. ``support_admitted``/``support_total``
    honestly report how much of the support band the fitted floor
    admits; a conservative encoder may admit little — that is the fit,
    not a failure of ``validate``.
    """
    key = _encoder_key(getattr(encoder, "encoder_id", None))
    rec = CALIBRATED_ENCODERS.get(key) if key else None
    floor = rec.floor if rec else None
    scores: dict = {}
    for slice_name, query, doc in slice_pairs:
        c = _cosine_pair(encoder, query, doc)
        if c is None:
            continue
        scores.setdefault(slice_name, []).append((query, doc, c))

    def _bounds(name: str) -> dict:
        vals = [c for _q, _d, c in scores.get(name, ())]
        return {
            "n": len(vals),
            "max": max(vals) if vals else None,
            "min": min(vals) if vals else None,
        }

    reject_max = max(
        (c for s in ("no_answer", "near_miss")
         for _q, _d, c in scores.get(s, ())),
        default=None,
    )
    support_vals = [c for _q, _d, c in scores.get("support", ())]
    admitted = (
        sum(1 for c in support_vals if floor is not None and c >= floor)
        if support_vals
        else 0
    )
    return {
        "version": CALIBRATION_VERSION,
        "encoder_id": key,
        "floor": floor,
        "slices": {name: _bounds(name) for name in
                   ("no_answer", "near_miss", "support")},
        "reject_envelope_max": reject_max,
        "separates": (
            floor is not None
            and reject_max is not None
            and reject_max < floor
        ),
        "support_admitted": admitted,
        "support_total": len(support_vals),
    }


__all__ = [
    "CALIBRATED_ENCODERS",
    "CALIBRATION_VERSION",
    "SIMILARITY_ELIGIBLE_CLASSES",
    "SimilarityCalibration",
    "calibration_for",
    "is_calibrated",
    "similarity_floor",
    "validate",
]
