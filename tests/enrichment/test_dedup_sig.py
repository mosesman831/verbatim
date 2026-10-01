"""shingle_signature — k=3 contiguous token shingles for MinHash-class
near-duplicate detection (V5-30.06)."""

from __future__ import annotations

from verbatim.enrichment import normalize_text, shingle_signature
from verbatim.enrichment.dedup_sig import SHINGLE_K, SHINGLE_VERSION


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _tokens(text: str):
    return normalize_text(text).split()


class TestShingles:
    def test_basic(self):
        sig = shingle_signature(["a", "b", "c", "d"])
        assert sig == frozenset({
            ("a", "b", "c"), ("b", "c", "d"),
        })

    def test_k_is_3(self):
        assert SHINGLE_K == 3
        assert SHINGLE_VERSION == "shingle/v1"

    def test_short_sequence_single_shingle(self):
        # shorter than k → one whole-sequence shingle (never empty for
        # non-empty input — short records stay comparable)
        assert shingle_signature(["a", "b"]) == frozenset({("a", "b")})
        assert shingle_signature(["only"]) == frozenset({("only",)})

    def test_empty(self):
        assert shingle_signature([]) == frozenset()
        assert shingle_signature(None) == frozenset()

    def test_returns_frozenset(self):
        assert isinstance(shingle_signature(["a", "b", "c"]), frozenset)

    def test_duplicate_positions_collapse(self):
        # set semantics — repeated n-grams appear once
        sig = shingle_signature(["x", "y", "x", "y", "x"])
        assert ("x", "y", "x") in sig and ("y", "x", "y") in sig
        assert len(sig) == 2

    def test_determinism(self):
        toks = ["deploy", "v2", "last", "week", "went", "fine"]
        assert shingle_signature(toks) == shingle_signature(list(toks))


class TestNearDuplicateBehavior:
    """Adversarial: signatures must differ where §30.07 forbids links
    and overlap where texts are near-identical."""

    def test_deploy_v1_vs_v2_differ(self):
        # "deploy-v1" vs "deploy-v2" — different identifiers must yield
        # different signatures (never duplicates, V5-30.07)
        s1 = shingle_signature(_tokens("deploy-v1 went live"))
        s2 = shingle_signature(_tokens("deploy-v2 went live"))
        assert s1 != s2
        assert _jaccard(s1, s2) < 1.0

    def test_polarity_pair_differs(self):
        # "I like X" vs "I no longer like X" — extra tokens change the
        # shingle set; dedup layers polarity on top (§30.07)
        s1 = shingle_signature(_tokens("I like X"))
        s2 = shingle_signature(_tokens("I no longer like X"))
        assert s1 != s2

    def test_near_identical_high_overlap(self):
        s1 = shingle_signature(_tokens(
            "the deploy to production finished on friday"))
        s2 = shingle_signature(_tokens(
            "the deploy to production finished on saturday"))
        assert _jaccard(s1, s2) > 0.5

    def test_unrelated_low_overlap(self):
        s1 = shingle_signature(_tokens("deploy v2 went live friday"))
        s2 = shingle_signature(_tokens("grandma bakes pies on sundays"))
        assert _jaccard(s1, s2) < 0.2
