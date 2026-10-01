"""Durable tests for ``support_verdict/v1`` + ``support_calibration/v1``
(SPEC_V5 §31.3, V5-31.07/31.08/31.09).

The contract under test: a similarity-only candidate is never supported
by a copied global threshold — it must meet the *provisioned encoder's*
fitted floor (``hashing:subword-ngram:v1`` → 0.75, fitted on the pinned
dev slice in ``verbatim/querying/calibration.py``); unknown encoders
fail closed. Coverage, identifier, and no-answer rules are unchanged.
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.memory.types import QueryClass, SupportStatus
from verbatim.querying import calibration as cal
from verbatim.querying.verdict import (
    VERDICT_VERSION,
    search_verdict,
)

ENCODER_ID = "hashing:subword-ngram:v1"


class _Hit:
    """Duck-typed lane hit: quote + signal surfaces the verdict reads."""

    def __init__(self, quote=None, signals=None, score_detail=None):
        self.quote = quote
        self.signals = signals or {}
        self.score_detail = score_detail or {}
        self.support_status = "unassessed"


def _verdict(hit, *, terms=("alpha", "beta"), primary="factual", **kw):
    return search_verdict(
        [hit], terms=terms, primary=primary, **kw
    )


# ---------------------------------------------------------------------
# coverage / identifier / no-answer rules (unchanged baseline)
# ---------------------------------------------------------------------


def test_coverage_supports_unsignalled_hit():
    hit = _Hit(quote="alpha beta gamma")
    v = _verdict(hit)  # terms alpha+beta, both covered
    assert v.verdict == SupportStatus.SUPPORTED.value
    assert v.kept == (hit,)
    assert hit.support_status == "supported"


def test_below_term_floor_suppresses():
    hit = _Hit(quote="nothing relevant here")
    v = _verdict(hit)
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert "below_term_floor" in v.reasons


def test_identifier_query_needs_exact_identifier():
    hit = _Hit(
        quote="the deploy command is deploy-v2",
        signals={"similarity": 0.99},
    )
    v = search_verdict(
        [hit],
        terms=("deploy",),
        identifiers=("ticket-9",),
        primary=QueryClass.IDENTIFIER.value,
        encoder=ENCODER_ID,
    )
    # Even a calibrated-above-floor similarity cannot answer an
    # identifier read without the exact surface form (V5-30.17).
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert "missing_exact_identifier" in v.reasons


def test_identifier_query_exact_hit_supported():
    hit = _Hit(quote="the ticket-9 fix shipped")
    v = search_verdict(
        [hit],
        terms=("fix",),
        identifiers=("ticket-9",),
        primary=QueryClass.IDENTIFIER.value,
    )
    assert v.verdict == SupportStatus.SUPPORTED.value


def test_no_answer_class_suppresses_everything():
    hit = _Hit(
        quote="alpha beta", signals={"similarity": 1.0}
    )
    v = _verdict(
        hit,
        primary=QueryClass.NO_ANSWER_LIKELY.value,
        encoder=ENCODER_ID,
    )
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert v.reasons == ("no_answer_query",)


# ---------------------------------------------------------------------
# calibrated similarity support (V5-31.07/31.08)
# ---------------------------------------------------------------------


def test_similarity_only_no_encoder_fails_closed():
    """No encoder arg → no calibrated floor → similarity never supports."""
    hit = _Hit(quote="opaque", signals={"similarity": 0.99})
    v = _verdict(hit)
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert v.detail["calibration"]["similarity_floor"] is None


def test_similarity_only_unknown_encoder_fails_closed():
    hit = _Hit(quote="opaque", signals={"similarity": 0.99})
    v = _verdict(hit, encoder="other-backend:model:v9")
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert v.detail["calibration"]["similarity_floor"] is None


def test_similarity_below_calibrated_floor_suppressed():
    hit = _Hit(quote="opaque", signals={"similarity": 0.60})
    v = _verdict(hit, encoder=ENCODER_ID)
    assert v.verdict == SupportStatus.INSUFFICIENT.value
    assert "below_term_floor" in v.reasons
    assert v.detail["calibration"]["similarity_floor"] == 0.75
    assert v.detail["calibration"]["similarity_supported"] == 0


def test_similarity_at_calibrated_floor_supports():
    """Raw similarity ≥ the encoder's fitted floor carries support even
    with zero term coverage — the V5-31.07 paraphrase path."""
    hit = _Hit(quote="opaque", signals={"similarity": 0.80})
    v = _verdict(hit, encoder=ENCODER_ID)
    assert v.verdict == SupportStatus.SUPPORTED.value
    assert hit.support_status == "supported"
    assert v.detail["calibration"]["similarity_supported"] == 1


def test_similarity_floor_is_encoder_bound_not_global():
    """The same raw score must clear *this* encoder's floor — a second
    encoder identity with no fit rejects the identical value."""
    hit = _Hit(quote="opaque", signals={"similarity": 0.80})
    supported = _verdict(hit, encoder=ENCODER_ID)
    rejected = _verdict(
        _Hit(quote="opaque", signals={"similarity": 0.80}),
        encoder="hashing:subword-ngram:v2",  # unknown revision → no fit
    )
    assert supported.verdict == SupportStatus.SUPPORTED.value
    assert rejected.verdict == SupportStatus.INSUFFICIENT.value


def test_fused_contribution_is_never_treated_as_cosine():
    """``score_detail`` holds normalized contributions — a 0.8
    contribution must NOT satisfy a 0.75 cosine floor (unit guard)."""
    hit = _Hit(quote="opaque", score_detail={"similarity": 0.8})
    v = _verdict(hit, encoder=ENCODER_ID)
    assert v.verdict == SupportStatus.INSUFFICIENT.value


def test_coverage_still_supports_similarity_hit_below_floor():
    """Coverage is independent support: a similarity-only hit that
    covers the term floor is kept without touching calibration."""
    hit = _Hit(quote="alpha beta", signals={"similarity": 0.10})
    v = _verdict(hit, encoder=ENCODER_ID)
    assert v.verdict == SupportStatus.SUPPORTED.value
    assert v.detail["calibration"]["similarity_supported"] == 0


def test_evidence_signal_unchanged():
    hit = _Hit(quote="alpha beta", signals={"lexical": 2.3})
    v = _verdict(hit, encoder=ENCODER_ID)
    assert v.verdict == SupportStatus.SUPPORTED.value


# ---------------------------------------------------------------------
# calibration registry + validate()
# ---------------------------------------------------------------------


def test_similarity_floor_registry():
    assert cal.similarity_floor(ENCODER_ID, "factual") == 0.75
    assert cal.similarity_floor(ENCODER_ID) == 0.75
    # Enum input resolves by value.
    assert cal.similarity_floor(ENCODER_ID, QueryClass.FACTUAL) == 0.75
    # Identifier / no-answer classes are outside the fitted set.
    assert cal.similarity_floor(ENCODER_ID, "identifier") is None
    assert cal.similarity_floor(
        ENCODER_ID, QueryClass.NO_ANSWER_LIKELY) is None
    # Unknown / blank / non-string encoders fail closed.
    assert cal.similarity_floor("other:x:v1", "factual") is None
    assert cal.similarity_floor(None, "factual") is None
    assert cal.similarity_floor("", "factual") is None
    assert cal.is_calibrated(ENCODER_ID) is True
    assert cal.is_calibrated("other:x:v1") is False


def test_calibration_record_carries_fit_provenance():
    rec = cal.calibration_for(ENCODER_ID)
    assert rec is not None and rec.floor == 0.75
    assert rec.measured["near_miss_max"] < rec.floor, (
        "the fitted floor must sit above the measured false-friend "
        "envelope"
    )
    assert rec.measured["no_answer_max"] < rec.floor
    assert "dev slice" in rec.fitted_on


def test_validate_reproduces_fitted_separation():
    """The pinned dev slice re-run through the real encoder must still
    place every no-answer/near-miss pair strictly below the floor —
    the executable form of the V5-31.08 validation requirement."""
    enc = HashingEncoder(VerbatimConfig())
    rep = cal.validate(enc)
    assert rep["encoder_id"] == ENCODER_ID
    assert rep["floor"] == 0.75
    assert rep["separates"] is True
    assert rep["reject_envelope_max"] < rep["floor"]
    # The recorded fit statistics match what the slice reproduces —
    # the table cannot drift silently from its dev data.
    rec = cal.calibration_for(ENCODER_ID)
    assert rep["slices"]["no_answer"]["max"] == pytest.approx(
        rec.measured["no_answer_max"], abs=1e-6)
    assert rep["slices"]["near_miss"]["max"] == pytest.approx(
        rec.measured["near_miss_max"], abs=1e-6)
    # Honest recall bound: the conservative floor admits only the
    # near-verbatim band of the support slice — reported, not hidden.
    assert rep["support_admitted"] >= 1
    assert rep["support_admitted"] < rep["support_total"]


def test_validate_unknown_encoder_reports_uncalibrated():
    class _Other:
        encoder_id = "other:enc:v9"
        dimensions = 384

        def encode(self, texts):
            return HashingEncoder(VerbatimConfig()).encode(texts)

    rep = cal.validate(_Other())
    assert rep["floor"] is None
    assert rep["separates"] is False


def test_verdict_version_and_detail_shape():
    v = search_verdict([], terms=("x",), primary="factual")
    assert v.verdict == "no_candidates"
    assert v.detail["version"] == VERDICT_VERSION
    assert v.detail["calibration"]["version"] == "support_calibration/v1"
