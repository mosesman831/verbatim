"""Tests for ``verbatim/querying/verdict_v2.py`` — ``support_verdict/v2``
(V7-11.01–08) and ``ident_eq/v1`` (§32.8).

Types are constructed directly from ``verbatim.core.types_v7`` — no
sibling worker modules are imported. All generators are seeded.
"""

from __future__ import annotations

import random
import string

import pytest

from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    GroupVerdict,
    IntentClass,
    IntentResult,
    IntervalUs,
    NormAnalysis,
    NormTerm,
    OccurredPrecision,
    OccurredSource,
    QueryViewV7,
    ResultStatus,
    ScoredCandidate,
    SupportLabel,
)
from verbatim.querying import verdict_v2 as vv


# ---------------------------------------------------------------------------
# builders — construct contract types directly
# ---------------------------------------------------------------------------

_POS = [0]


def _term(text: str, channel: str = "text") -> NormTerm:
    _POS[0] += len(text) + 1
    return NormTerm(text, channel, _POS[0] - len(text), _POS[0])


def qv(
    text: str = "",
    *,
    terms=(),
    ids=(),
    stems=(),
    intent: IntentClass = IntentClass.LOOKUP,
    classes=None,
    ents=(),
    window=None,
    facets=(),
) -> QueryViewV7:
    _POS[0] = 0
    t_terms = tuple(_term(t, "text") for t in terms)
    t_terms += tuple(_term(t, "stem") for t in stems)
    t_ids = tuple(_term(i, "identifier") for i in ids)
    norm = NormAnalysis(
        "norm/v2", t_terms + t_ids, t_ids, text
    )
    ir = IntentResult(
        intent, tuple(classes) if classes else (intent,), window
    )
    return QueryViewV7(
        query=text, norm=norm, intent=ir,
        entity_canons=tuple(ents), facets=tuple(facets),
    )


def cand(
    unit: str,
    text=None,
    *,
    score: float = 0.5,
    source: str = None,
    detail=None,
    signals=None,
    group=None,
    lane=None,
) -> ScoredCandidate:
    d = dict(detail or {})
    if signals:
        d.setdefault("signals", dict(signals))
    if text is not None:
        d.setdefault("text", text)
    if group is not None:
        d.setdefault("group", group)
    if lane is not None:
        d.setdefault("lane", lane)
    return ScoredCandidate(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=1,
        score=score,
        score_family="ranking/v7",
        detail=d,
    )


def window(start: int, end: int) -> IntervalUs:
    return IntervalUs(
        start, end,
        precision=OccurredPrecision.DAY,
        source=OccurredSource.EXPLICIT,
        rule_id="T02",
    )


def verdict(items, query, ctx=None):
    groups = vv.classify_groups(items, query, ctx)
    status, missing = vv.result_verdict(groups, query)
    return groups, status, missing


# ===========================================================================
# ident_eq/v1 — every §32.8 rule, positive and negative
# ===========================================================================


class TestIdentEq:
    # -- case-insensitive tickets and handles --
    def test_case_ticket_positive(self):
        assert vv.ident_eq("ABC-123", "abc-123")
        assert vv.ident_eq("Jira-42", "JIRA-42")

    def test_case_ticket_negative(self):
        assert not vv.ident_eq("ABC-123", "abd-123")

    def test_case_handle_positive(self):
        assert vv.ident_eq("@Alice", "@alice")
        assert vv.ident_eq("#Release", "#release")

    def test_case_handle_negative(self):
        assert not vv.ident_eq("@alice", "@bob")

    # -- separators - _ . / <space> interchangeable --
    @pytest.mark.parametrize("b", [
        "abc_123", "abc.123", "abc/123", "abc 123", "ABC-123",
    ])
    def test_separator_positive(self, b):
        assert vv.ident_eq("abc-123", b)

    def test_separator_negative(self):
        assert not vv.ident_eq("abc-123", "abc-124")
        assert not vv.ident_eq("abc-123", "abc123")  # sep vs none

    # -- leading zeros on numeric ids >= 3 digits --
    @pytest.mark.parametrize("pair", [
        ("007", "7"), ("042", "42"), ("000123", "123"), ("010", "10"),
    ])
    def test_leading_zeros_positive(self, pair):
        assert vv.ident_eq(*pair)

    @pytest.mark.parametrize("pair", [
        ("01", "1"),   # < 3 digits: zeros are significant
        ("007", "8"),
        ("123", "0124"),
    ])
    def test_leading_zeros_negative(self, pair):
        assert not vv.ident_eq(*pair)

    # -- URL normalization --
    @pytest.mark.parametrize("pair", [
        ("HTTP://Example.com/", "http://example.com"),
        ("http://x:80/a", "http://x/a"),
        ("https://x:443/a/b/", "https://x/a/b"),
        ("http://x/%2f", "HTTP://X/%2F"),
        ("https://x", "https://x/"),
    ])
    def test_url_positive(self, pair):
        assert vv.ident_eq(*pair)

    @pytest.mark.parametrize("pair", [
        ("http://x:8080/a", "http://x/a"),       # non-default port
        ("http://x/a", "http://x/b"),
        ("http://x/a?p=1", "http://x/a?p=2"),
        ("http://x/a", "https://x/a"),           # scheme differs
    ])
    def test_url_negative(self, pair):
        assert not vv.ident_eq(*pair)

    # -- semver component compare, optional v prefix --
    @pytest.mark.parametrize("pair", [
        ("v1.2.3", "1.2.3"),
        ("1.2.3+build.7", "1.2.3"),
        ("v2.0.0", "2.0.0"),
        ("1.2.3-alpha.1", "1.2.3-alpha.01"),  # numeric pre ids numeric
    ])
    def test_semver_positive(self, pair):
        assert vv.ident_eq(*pair)

    @pytest.mark.parametrize("pair", [
        ("1.2.3", "1.2.4"),
        ("1.2.3-alpha", "1.2.3"),     # release > prerelease
        ("v2.0.0", "1.9.9"),
        ("1.2.3-alpha", "1.2.3-beta"),
        ("1.2.3", "1.2.3.4"),          # not semver on the right
    ])
    def test_semver_negative(self, pair):
        assert not vv.ident_eq(*pair)

    # -- hash >= 7-char prefix, unique in scope --
    def test_hash_full_case(self):
        assert vv.ident_eq("DeAdBeEf0123", "deadbeef0123")

    def test_hash_prefix_unique(self):
        peers = ["abc1234ff00dd", "feedface99"]
        assert vv.ident_eq("abc1234", "abc1234ff00dd", peers=peers)

    def test_hash_prefix_too_short(self):
        assert not vv.ident_eq(
            "abc123", "abc1234ff00dd", peers=["abc1234ff00dd"]
        )

    def test_hash_prefix_not_unique(self):
        peers = ["abc1234ff00dd", "abc1234aa11"]
        assert not vv.ident_eq("abc1234", "abc1234ff00dd", peers=peers)
        assert not vv.ident_eq("abc1234", "abc1234aa11", peers=peers)

    def test_hash_prefix_duplicate_same_full_ok(self):
        # the same full hash appearing twice stays a unique referent
        peers = ["abc1234ff00dd", "abc1234ff00dd", "feedface99"]
        assert vv.ident_eq("abc1234", "abc1234ff00dd", peers=peers)

    def test_hash_negative(self):
        assert not vv.ident_eq("deadbeef", "deadbeee")

    # -- cross-kind never matches --
    @pytest.mark.parametrize("pair", [
        ("http://x/a", "a"),
        ("v1.2.3", "123"),
        ("@alice", "alice-1"),
    ])
    def test_cross_kind_negative(self, pair):
        assert not vv.ident_eq(*pair)

    def test_exact_always(self):
        assert vv.ident_eq("Some-Weird_ID.9", "Some-Weird_ID.9")

    def test_empty_never(self):
        assert not vv.ident_eq("", "x")
        assert not vv.ident_eq("x", "")
        assert not vv.ident_eq("", "")

    def test_identifier_tokens_extraction(self):
        toks = vv.identifier_tokens(
            "see ticket ABC-123 and https://x.io/a, sha deadbeef42, "
            "release v2.0.1, order 66 filed"
        )
        assert "ABC-123" in toks
        assert any(t.startswith("https://") for t in toks)
        assert "deadbeef42" in toks
        assert "v2.0.1" in toks
        assert "order 66" in toks
        # ordinary words are not identifiers
        assert "ticket" not in toks


# ===========================================================================
# Group labeling — signals, never deletion (V7-11.01/11.03)
# ===========================================================================


class TestGroupLabels:
    def test_identifier_exact_supported(self):
        q = qv("ticket ABC-123?", ids=("ABC-123",),
               intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "the fix for ABC-123 shipped")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED
        assert g[0].detail["id_match"] == "exact"
        assert g[0].detail["via"] == "identifier_exact"

    def test_identifier_normalized_supported(self):
        q = qv("ticket ABC-123?", ids=("ABC-123",),
               intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "the fix for abc_123 shipped")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED
        assert g[0].detail["id_match"] == "normalized"

    def test_identifier_signal_supported(self):
        q = qv("id?", ids=("ZX-9",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", None,
                      signals={"identifier_hit": True})]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED

    def test_full_coverage_supported(self):
        q = qv("when did jordan join pottery class",
               terms=("when", "did", "jordan", "join", "pottery",
                      "class"))
        items = [cand(
            "u1",
            "jordan did join the pottery class yesterday when it "
            "opened",
            signals={"lexical": 4.2}, lane="lex")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED

    def test_mid_coverage_with_signal_supported(self):
        q = qv("q", terms=("a", "b", "c", "d", "e"))
        items = [cand("u1", "a b x y z", signals={"bm25": 1.0},
                      lane="lex")]
        g = vv.classify_groups(items, q)
        # 2/5 = 0.4 coverage + real signal -> supported
        assert g[0].label == SupportLabel.SUPPORTED

    def test_partial_label(self):
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [cand("u1", "alpha was mentioned only",
                      signals={"lexical": 1.0}, lane="lex")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.PARTIAL

    def test_weak_label(self):
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [cand("u1", "completely unrelated words here")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.WEAK
        assert g[0].detail["via"] == "weak_coverage"

    def test_entity_canon_supported(self):
        q = qv("tell me about jordan", ents=("jordan",))
        items = [cand("u1", "jordan moved to denver",
                      signals={"entity_overlap": True}, lane="ent")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED

    def test_entity_partial(self):
        q = qv("jordan and casey", ents=("jordan", "casey"))
        items = [cand("u1", "jordan moved", lane="ent")]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.PARTIAL

    def test_window_match_partial(self):
        w = window(1_000_000, 2_000_000)
        q = qv("what happened", intent=IntentClass.HISTORY_OF,
               window=w)
        items = [cand("u1", "stuff", detail={"occurred_us": 1_500_000})]
        g = vv.classify_groups(items, q)
        assert g[0].label in (SupportLabel.PARTIAL, SupportLabel.SUPPORTED)
        assert g[0].detail["window_match"] is True

    def test_nothing_is_deleted(self):
        """D7-04 by construction: zero literal coverage items survive."""
        q = qv("dentist appointment friday",
               terms=("dentist", "appointment", "friday"))
        items = [
            cand("u1", "the tooth doctor visit is on the fifth",
                 signals={"bm25": 2.0}, lane="lex"),
            cand("u2", "unrelated stuff"),
        ]
        g = vv.classify_groups(items, q)
        assert len(g) == 2  # both groups classified, none removed
        assert all(isinstance(x.label, SupportLabel) for x in g)

    def test_uninspectable_text_signal_partial(self):
        q = qv("q", terms=("alpha", "beta", "gamma"))
        items = [cand("u1", None, signals={"lexical": 3.0},
                      lane="lex")]
        g = vv.classify_groups(items, q)
        # real lane match, coverage not measurable -> partial, not weak
        assert g[0].label == SupportLabel.PARTIAL

    def test_grouping_by_session(self):
        q = qv("q", terms=("alpha",))
        items = [
            cand("u1", "alpha", group="s-1"),
            cand("u2", "alpha", group="s-1"),
            cand("u3", "alpha", group="s-2"),
        ]
        g = vv.classify_groups(items, q)
        keys = [x.group_key for x in g]
        assert keys == ["grp:s-1", "grp:s-2"]
        assert g[0].detail["members"] == ["u1", "u2"]

    def test_grouping_fallback_source(self):
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "alpha", source="doc-9"),
                 cand("u2", "alpha", source="doc-9")]
        g = vv.classify_groups(items, q)
        assert len(g) == 1
        assert g[0].group_key == "src:doc-9"

    def test_group_verdict_trigger_field(self):
        q = qv("id", ids=("T-100",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "no id here")]
        g = vv.classify_groups(items, q)
        assert g[0].trigger == "a"


# ===========================================================================
# result_verdict — triggers (a)-(d) only
# ===========================================================================


class TestTriggers:
    def test_b_empty(self):
        q = qv("anything", terms=("x",), ents=("e1",),
               ids=("ID-1",), window=window(1, 2))
        status, missing = vv.result_verdict([], q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=b" in missing.note
        # every query facet reported unsupported
        assert missing.facets["identifier"] == ("ID-1",)
        assert missing.facets["terms"] == ("x",)
        assert missing.facets["entities"] == ("e1",)
        assert missing.facets["time_window"] == ("1..2",)

    def test_a_identifier_absent(self):
        q = qv("ticket?", ids=("ABC-999",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "some other ticket ABC-123 entirely"),
                 cand("u2", "no id")]
        groups, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=a" in missing.note
        assert missing.facets["identifier"] == ("ABC-999",)
        # groups still exist and were labeled — never deleted
        assert len(groups) == 2

    def test_a_identifier_normalized_present(self):
        q = qv("ticket?", ids=("ABC-999",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "tracking abc_999 now")]
        _, status, missing = verdict(items, q)
        assert status == ResultStatus.READY
        assert missing is None

    def test_a_multi_id_partial_match_abstains(self):
        """One of two asked identifiers missing -> the identifier facet
        is partially unsupported: abstain and name it (V7-11.08)."""
        q = qv("ids?", ids=("T-1", "T-2"), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "T-1 done")]
        groups, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=a" in missing.note
        assert missing.facets["identifier"] == ("T-2",)
        # the matched group is labeled partial, not deleted
        assert groups[0].label == SupportLabel.PARTIAL
        assert groups[0].detail["id_match"] == "exact"

    def test_a_all_ids_matched_ready(self):
        q = qv("ids?", ids=("T-1", "T-2"), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "T-1 done"), cand("u2", "T-2 pending")]
        _, status, _ = verdict(items, q)
        assert status == ResultStatus.READY

    def test_a_corpus_existence_blocks_trigger(self):
        """The identifier exists in the eligible corpus (ctx peers)
        even though no delivered item carried it -> not a trigger-(a)
        abstention; groups ship labeled."""
        q = qv("id?", ids=("abc1234",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "some other words entirely")]
        ctx = {"corpus_identifiers": ["abc1234ff00dd"]}
        groups = vv.classify_groups(items, q, ctx)
        status, _ = vv.result_verdict(groups, q)
        assert status == ResultStatus.READY

    def test_a_lane_signal_unattributed_blocks_trigger(self):
        q = qv("id?", ids=("Q-1", "Q-2"), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", None, signals={"identifier_hit": True})]
        _, status, _ = verdict(items, q)
        assert status == ResultStatus.READY

    def test_c_fitted_below_threshold(self):
        q = qv("q", terms=("alpha", "beta"))
        items = [cand("u1", "alpha beta", score=0.31)]
        cal = {"threshold": 0.55, "separates": True,
               "precision": 0.9, "recall": 0.8, "n": 200}
        groups = vv.classify_groups(items, q)
        status, missing = vv.result_verdict(groups, q, cal)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=c" in missing.note
        assert "calibration=fitted" in missing.note

    def test_c_fitted_above_threshold(self):
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "alpha", score=0.9)]
        cal = {"threshold": 0.55, "separates": True}
        status, missing = vv.result_verdict(
            vv.classify_groups(items, q), q, cal)
        assert status == ResultStatus.READY
        assert missing is None

    def test_c_unfitted_none_disabled(self):
        """calibration=None: trigger c disabled — low score still ready."""
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "alpha", score=0.01)]
        status, missing = vv.result_verdict(
            vv.classify_groups(items, q), q, None)
        assert status == ResultStatus.READY
        assert missing is None
        # and the disabled state is reported
        rep = vv.calibration_status(None)
        assert rep["status"] == "unfitted"
        assert rep["report"] == "calibration=unfitted"
        assert rep["fitted"] is False

    @pytest.mark.parametrize("cal", [
        {"threshold": 0.9, "separates": False},   # not separating
        {"threshold": 0.9},                        # separates absent
        {"separates": True},                       # no threshold
        {"threshold": float("nan"), "separates": True},
        {"threshold": "bad", "separates": True},
        {},
    ])
    def test_c_unfitted_variants_disabled(self, cal):
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "alpha", score=0.01)]
        status, _ = vv.result_verdict(
            vv.classify_groups(items, q), q, cal)
        assert status == ResultStatus.READY
        assert vv.calibration_status(cal)["status"] == "unfitted"

    def test_c_unfitted_reported_when_insufficient(self):
        q = qv("id?", ids=("NOPE-1",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "nothing")]
        groups = vv.classify_groups(items, q)
        status, missing = vv.result_verdict(
            groups, q, {"threshold": 0.9, "separates": False})
        assert status == ResultStatus.INSUFFICIENT
        assert "calibration=unfitted" in missing.note

    def test_d_negative_evidence_abstain_likely(self):
        q = qv("when did i visit mars",
               terms=("when", "did", "i", "visit", "mars"),
               intent=IntentClass.ABSTAIN_LIKELY)
        items = [
            cand("u1", "you have never been to mars",
                 detail={"negative_evidence": True},
                 signals={"lexical": 1.0}, lane="lex"),
        ]
        groups, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=d" in missing.note

    def test_d_not_fired_without_negative_evidence(self):
        """abstain_likely + weak topical hits, no negation -> ready
        (weak groups ship labeled; triggers are the only abstain path)."""
        q = qv("when did i visit mars",
               terms=("when", "did", "i", "visit", "mars"),
               intent=IntentClass.ABSTAIN_LIKELY)
        items = [cand("u1", "unrelated words entirely")]
        _, status, _ = verdict(items, q)
        assert status == ResultStatus.READY

    def test_d_not_fired_when_supported(self):
        """corpus states the opposite AND the premise: conflict, not
        abstention — the pack carries the supported group."""
        q = qv("when did i visit mars",
               terms=("when", "did", "i", "visit", "mars"),
               intent=IntentClass.ABSTAIN_LIKELY)
        items = [
            cand("u1", "never went", detail={"negative_evidence": True}),
            cand("u2",
                 "when i did visit mars it was august",
                 signals={"lexical": 9.0}, lane="lex"),
        ]
        groups, status, _ = verdict(items, q)
        assert status == ResultStatus.READY
        assert any(g.label == SupportLabel.SUPPORTED for g in groups)

    def test_d_polarity_negate_counts(self):
        q = qv("q", terms=("x",), intent=IntentClass.ABSTAIN_LIKELY)
        items = [cand("u1", "not the case", detail={"polarity": "negate"})]
        _, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=d" in missing.note

    def test_no_trigger_no_abstain(self):
        """all-weak non-identifier query: still ready (V7-11.01)."""
        q = qv("q", terms=("unfindable", "terms", "here"))
        items = [cand("u1", "nothing matching at all")]
        _, status, _ = verdict(items, q)
        assert status == ResultStatus.READY


# ===========================================================================
# MissingDescriptor completeness (V7-11.08)
# ===========================================================================


class TestMissingDescriptor:
    def test_partial_coverage_listed(self):
        q = qv("id?", ids=("MISS-7",),
               terms=("alpha", "beta", "gamma"),
               ents=("e1", "e2"),
               intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "alpha e1 present only")]
        _, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert missing.facets["identifier"] == ("MISS-7",)
        assert missing.facets["terms"] == ("beta", "gamma")
        assert missing.facets["entities"] == ("e2",)

    def test_fully_covered_facets_omitted(self):
        q = qv("id?", ids=("MISS-7",), terms=("alpha",),
               intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "alpha covered")]
        _, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "terms" not in missing.facets
        assert missing.facets["identifier"] == ("MISS-7",)

    def test_window_facet(self):
        w = window(100, 200)
        q = qv("id?", ids=("MISS-7",), intent=IntentClass.IDENTIFIER,
               window=w)
        items = [cand("u1", "x", detail={"occurred_us": 999})]
        _, status, missing = verdict(items, q)
        assert status == ResultStatus.INSUFFICIENT
        assert missing.facets["time_window"] == ("100..200",)

    def test_window_supported_omitted(self):
        w = window(100, 200)
        q = qv("id?", ids=("MISS-7",), intent=IntentClass.IDENTIFIER,
               window=w)
        items = [cand("u1", "x", detail={"occurred_us": 150})]
        _, _, missing = verdict(items, q)
        assert "time_window" not in missing.facets


# ===========================================================================
# Delivery selection — strict flag (V7-11.03)
# ===========================================================================


class TestDeliverable:
    def _groups(self):
        return [
            GroupVerdict("g1", SupportLabel.SUPPORTED),
            GroupVerdict("g2", SupportLabel.WEAK),
            GroupVerdict("g3", SupportLabel.PARTIAL),
            GroupVerdict("g4", SupportLabel.SUPPORTED),
        ]

    def test_nonstrict_delivers_weak(self):
        out = vv.deliverable_groups(self._groups())
        assert [g.group_key for g in out] == ["g1", "g2", "g3", "g4"]

    def test_strict_only_supported(self):
        out = vv.deliverable_groups(self._groups(), strict=True)
        assert [g.group_key for g in out] == ["g1", "g4"]

    def test_limit_fills_with_weak(self):
        out = vv.deliverable_groups(self._groups(), limit=3)
        assert [g.group_key for g in out] == ["g1", "g2", "g3"]

    def test_limit_supported_saturates(self):
        out = vv.deliverable_groups(self._groups(), limit=2)
        assert [g.group_key for g in out] == ["g1", "g4"]

    def test_strict_with_limit(self):
        out = vv.deliverable_groups(
            self._groups(), strict=True, limit=1)
        assert [g.group_key for g in out] == ["g1"]

    def test_strict_does_not_change_verdict(self):
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "unrelated")]
        groups = vv.classify_groups(items, q)
        kept = vv.deliverable_groups(groups, strict=True)
        assert kept == []  # weak filtered from delivery
        status, _ = vv.result_verdict(groups, q)
        assert status == ResultStatus.READY  # verdict unaffected


# ===========================================================================
# D7-04 regression — paraphrased-but-answerable is never abstained
# ===========================================================================


class TestD704:
    def test_paraphrased_answer_ready(self):
        """Query terms do not literally appear in the evidence; the
        v1 floor abstained here — v2 must deliver (labeled weak)."""
        q = qv("when is my dentist appointment",
               terms=("when", "is", "my", "dentist", "appointment"))
        items = [
            cand("u1", "the tooth doctor visit is on friday",
                 signals={"bm25": 1.2, "similarity": 0.44},
                 lane="dense"),
            cand("u2", "see the orthodontist next week",
                 signals={"lexical": 0.4}, lane="fuzzy"),
        ]
        groups, status, missing = verdict(items, q)
        assert status == ResultStatus.READY
        assert missing is None
        assert len(groups) == 2
        assert all(
            g.label in (SupportLabel.WEAK, SupportLabel.PARTIAL)
            for g in groups
        )

    def test_zero_literal_coverage_delivered(self):
        q = qv("favorite color", terms=("favorite", "color"),
               ents=("alice",))
        items = [cand("u1", "she adores teal and cyan tones",
                      signals={"similarity": 0.61}, lane="dense")]
        groups, status, _ = verdict(items, q)
        assert status == ResultStatus.READY
        assert vv.deliverable_groups(groups)


# ===========================================================================
# Generated suites — >=300 abstention, >=300 answerable cases
# ===========================================================================

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu pottery dentist marathon "
    "algebra bicycle concert dentist embassy falcon galaxy harbor "
    "igloo jungle kitten lantern magnet nectar orchid piano quasar"
).split()


def _rand_text(rng, n=12):
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _rand_ident(rng):
    kind = rng.randrange(5)
    if kind == 0:
        return (f"{rng.choice(['ABC', 'JIRA', 'BUG', 'T'])}-"
                f"{rng.randrange(100, 9999)}")
    if kind == 1:
        return str(rng.randrange(100, 99999))
    if kind == 2:
        return f"@{rng.choice(['alice', 'bob', 'carol'])}{rng.randrange(9)}"
    if kind == 3:
        return (f"{rng.randrange(0, 9)}.{rng.randrange(0, 9)}."
                f"{rng.randrange(0, 20)}")
    return "".join(rng.choice("0123456789abcdef")
                   for _ in range(rng.randrange(8, 16)))


class TestGeneratedAbstention:
    """320 generated abstention cases across all four triggers."""

    def test_generated_abstentions(self):
        rng = random.Random(20260918)
        insufficient = 0
        for i in range(320):
            kind = i % 4
            if kind == 0:
                # (a) identifier query, corpus lacks the identifier
                qid = _rand_ident(rng)
                while rng.random() < 0.3:
                    qid += str(rng.randrange(10))
                items = [
                    cand(f"u{i}-{j}", _rand_text(rng),
                         score=rng.random())
                    for j in range(rng.randrange(1, 4))
                ]
                query = qv("id?", ids=(qid,),
                           intent=IntentClass.IDENTIFIER)
            elif kind == 1:
                # (b) zero eligible candidates
                items = []
                query = qv("q", terms=tuple(
                    rng.sample(_WORDS, rng.randrange(1, 4))))
            elif kind == 2:
                # (c) fitted calibration, all scores below threshold
                th = rng.uniform(0.5, 0.9)
                items = [
                    cand(f"u{i}-{j}", _rand_text(rng),
                         score=rng.uniform(0.0, th - 0.05))
                    for j in range(rng.randrange(1, 4))
                ]
                query = qv("q", terms=tuple(
                    rng.sample(_WORDS, rng.randrange(1, 3))))
            else:
                # (d) abstain_likely + explicit negative evidence;
                # item text drawn from vocabulary disjoint from the
                # query terms so no group reaches SUPPORTED
                qterms = tuple(rng.sample(_WORDS, 3))
                disjoint = [w for w in _WORDS if w not in qterms]
                items = [
                    cand(f"u{i}-{j}",
                         " ".join(rng.sample(disjoint, 6)),
                         detail={"negative_evidence": True})
                    for j in range(rng.randrange(1, 3))
                ]
                query = qv(
                    "abs?", terms=qterms,
                    intent=IntentClass.ABSTAIN_LIKELY)
            groups = vv.classify_groups(items, query)
            cal = (
                {"threshold": th, "separates": True}
                if kind == 2 else None
            )
            status, missing = vv.result_verdict(groups, query, cal)
            assert status == ResultStatus.INSUFFICIENT, (i, kind)
            assert missing is not None
            assert "trigger=" in missing.note
            insufficient += 1
        assert insufficient == 320

    def test_generated_identifier_negatives(self):
        """Mutated identifiers that are NOT equivalent still abstain."""
        rng = random.Random(77)
        n = 0
        for i in range(80):
            qid = f"TKT-{rng.randrange(100, 999)}"
            # different-number ticket present -> no match
            other = rng.randrange(100, 999)
            while f"TKT-{other}" == qid:
                other = rng.randrange(100, 999)
            items = [cand(f"u{i}", f"worked on TKT-{other} today")]
            query = qv("id?", ids=(qid,), intent=IntentClass.IDENTIFIER)
            status, missing = verdict(items, query)[1:]
            assert status == ResultStatus.INSUFFICIENT
            assert missing.facets["identifier"] == (qid,)
            n += 1
        assert n == 80


def _mutate_ident(rng, ident):
    """Apply a random §32.8-equivalent mutation to an identifier."""
    kind = vv.ident_kind(ident)
    r = rng.random()
    if kind == "ticket":
        for sep in "-_./":
            if sep in ident:
                return ident.replace(
                    sep, rng.choice(["-", "_", ".", "/", " "]))
        return ident.swapcase()
    if kind == "numeric" and len(ident) >= 3:
        return "0" * rng.randrange(1, 4) + ident
    if kind == "hash":
        if r < 0.5:
            return ident[: rng.randrange(7, len(ident) + 1)]
        return ident.upper()
    if kind == "handle":
        return ident.upper() if r < 0.5 else ident.lower()
    return ident.swapcase()


class TestGeneratedAnswerable:
    """320 generated answerable cases — none may abstain (D7-04)."""

    def test_generated_answerable(self):
        rng = random.Random(424242)
        ready = 0
        for i in range(320):
            kind = i % 4
            if kind == 0:
                # identifier equivalence: mutated form in the corpus
                base = _rand_ident(rng)
                if vv.ident_kind(base) == "url":
                    continue
                mutated = _mutate_ident(rng, base)
                items = [cand(f"u{i}", f"reference {mutated} here")]
                query = qv("id?", ids=(base,),
                           intent=IntentClass.IDENTIFIER)
            elif kind == 1:
                # term-covered answerable
                terms = tuple(rng.sample(_WORDS, rng.randrange(2, 5)))
                text = " ".join(terms) + " " + _rand_text(rng, 4)
                items = [cand(f"u{i}", text,
                              signals={"lexical": 2.0}, lane="lex")]
                query = qv("q", terms=terms)
            elif kind == 2:
                # paraphrased: zero literal overlap, signal present
                terms = tuple(rng.sample(_WORDS, 3))
                disjoint = [w for w in _WORDS if w not in terms]
                text = " ".join(rng.sample(disjoint, 6))
                items = [cand(
                    f"u{i}", text,
                    signals={"similarity": rng.uniform(0.3, 0.8)},
                    lane="dense")]
                query = qv("q", terms=terms)
            else:
                # entity coverage
                ent = rng.choice(["jordan", "casey", "alice"])
                items = [cand(
                    f"u{i}", f"{ent} went to the market",
                    signals={"entity_overlap": True}, lane="ent")]
                query = qv("q", terms=(), ents=(ent,))
            groups = vv.classify_groups(items, query)
            status, missing = vv.result_verdict(groups, query)
            assert status == ResultStatus.READY, (i, kind, missing)
            assert missing is None
            assert vv.deliverable_groups(groups)  # never empty
            ready += 1
        assert ready >= 300

    def test_generated_equivalence_matrix(self):
        """Every §32.8 equivalence class survives the verdict."""
        rng = random.Random(9)
        cases = [
            ("ABC-123", "abc_123"), ("ABC-123", "abc 123"),
            ("007", "7"), ("042", "42"),
            ("HTTP://Example.com/", "http://example.com"),
            ("v1.2.3", "1.2.3"), ("deadbeef42", "DEADBEEF42"),
            ("@Alice", "@alice"),
        ]
        for i, (qid, corpus) in enumerate(cases):
            items = [cand(f"m{i}", f"see {corpus} inside")]
            query = qv("id?", ids=(qid,), intent=IntentClass.IDENTIFIER)
            groups, status, missing = verdict(items, query)
            assert status == ResultStatus.READY, (qid, corpus, missing)
            assert groups[0].label == SupportLabel.SUPPORTED
            rng.random()  # keep signature seeded


# ===========================================================================
# Misc contract checks
# ===========================================================================


class TestContract:
    def test_determinism(self):
        q = qv("q", terms=("a", "b"), ents=("e1",), ids=("X-1",))
        items = [
            cand("u1", "a e1 X-1", score=0.4, signals={"lexical": 1}),
            cand("u2", "unrelated", score=0.1, group="s"),
        ]
        g1 = vv.classify_groups(items, q)
        g2 = vv.classify_groups(items, q)
        assert g1 == g2
        s1, m1 = vv.result_verdict(g1, q)
        s2, m2 = vv.result_verdict(g2, q)
        assert (s1, m1) == (s2, m2)

    def test_version_pins(self):
        # V8-12.01: v3 adds the answerability/advisory separation —
        # premise doubt is measured and reported but cannot change the
        # status unless the ``verdict.premise_speaker`` flag is armed.
        assert vv.VERDICT_VERSION == "support_verdict/v3"
        assert vv.IDENT_EQ_VERSION == "ident_eq/v1"
        assert vv.FORMULA_STATUS_PROVISIONAL == "provisional/v7-r0"

    def test_mapping_items_accepted(self):
        q = qv("q", terms=("alpha", "beta"))
        items = [{"unit_id": "u1", "score": 0.9,
                  "text": "alpha beta",
                  "signals": {"lexical": 2.0}}]
        g = vv.classify_groups(items, q)
        assert g[0].label == SupportLabel.SUPPORTED

    def test_bytes_quote_decoded(self):
        q = qv("q", terms=("alpha",))
        items = [{"unit_id": "u1", "quote": "alpha text".encode()}]
        g = vv.classify_groups(items, q)
        assert g[0].detail["covered_terms"] == ("alpha",)

    def test_facets_merged(self):
        facet = qv("f", terms=("gamma",), ids=("F-1",))
        q = qv("q", terms=("alpha",), ids=("Q-1",), facets=(facet,))
        items = [cand("u1", "alpha gamma F-1 Q-1")]
        g = vv.classify_groups(items, q)
        assert g[0].detail["id_match"] == "exact"
        assert set(g[0].detail["covered_terms"]) == {"alpha", "gamma"}

    def test_ctx_peers_extend_scope(self):
        """ctx corpus identifiers participate in hash uniqueness."""
        q = qv("id?", ids=("abc1234",), intent=IntentClass.IDENTIFIER)
        # item carries the full hash; ctx adds a DIFFERENT hash sharing
        # the prefix -> prefix no longer unique -> no match -> abstain
        items = [cand("u1", "commit abc1234ff00dd pushed")]
        ctx = {"corpus_identifiers": ["abc1234aa11", "abc1234ff00dd"]}
        groups = vv.classify_groups(items, q, ctx)
        status, missing = vv.result_verdict(groups, q)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=a" in missing.note

    def test_groupverdict_formula_tag(self):
        q = qv("q", terms=("a",))
        g = vv.classify_groups([cand("u1", "a")], q)
        assert g[0].detail["formula"] == "provisional/v7-r0"
        assert g[0].detail["verdict"] == "support_verdict/v3"
