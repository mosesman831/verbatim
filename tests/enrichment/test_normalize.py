"""normalize_text / normalized_digest — the pinned ``norm/v1``
projection (V5-30.06). Determinism is the contract: same input, same
output, on every host."""

from __future__ import annotations

import hashlib

import pytest

from verbatim.enrichment import (
    NORMALIZATION_VERSION,
    normalize_text,
    normalized_digest,
)


class TestNormalize:
    def test_case_fold(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_whitespace_collapse(self):
        assert normalize_text("a\t b\n\n  c\r\nd") == "a b c d"
        assert normalize_text("   padded   ") == "padded"

    def test_punctuation_folds_to_space(self):
        assert normalize_text("deploy-v2, now!") == "deploy v2 now"
        assert normalize_text("foo_bar.py") == "foo bar py"
        assert normalize_text("a,b;c:d") == "a b c d"

    def test_unicode_compat_fold(self):
        # NFKC folds full-width and ligature forms
        assert normalize_text("１２３") == "123"          # fullwidth digits
        assert normalize_text("ﬁle") == "file"             # ﬁ ligature
        assert normalize_text("ＡＢＣ") == "abc"           # fullwidth caps

    def test_combining_marks_dropped(self):
        assert normalize_text("Héllo") == "hello"
        assert normalize_text("Café") == "cafe"
        assert normalize_text("naïve") == "naive"
        # precomposed and decomposed é agree
        assert normalize_text("café") == normalize_text("café")

    def test_casefold_forms(self):
        # German sharp-s and dotted-I fold the same way every run
        assert normalize_text("Straße") == "strasse"
        assert normalize_text("İ") == normalize_text("i̇")

    def test_symbols_and_emoji_fold(self):
        out = normalize_text("launch 🚀 now — done ✓")
        assert "🚀" not in out and "✓" not in out
        assert out == "launch now done"

    def test_numbers_preserved(self):
        # version-bearing digits must survive normalization or dedup
        # would collapse "deploy-v1" and "deploy-v2" (V5-30.07)
        assert normalize_text("deploy-v1") == "deploy v1"
        assert normalize_text("deploy-v2") == "deploy v2"
        assert normalize_text("deploy-v1") != normalize_text("deploy-v2")

    def test_empty_and_none(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
        assert normalize_text(None) == ""

    def test_version_pinned(self):
        assert normalize_text("X", version="norm/v1") == "x"
        with pytest.raises(ValueError):
            normalize_text("X", version="norm/v2")
        with pytest.raises(ValueError):
            normalize_text("X", version="")
        assert NORMALIZATION_VERSION == "norm/v1"

    def test_determinism(self):
        text = "Café Müller — DEPLOY-v10.2.3  (ßeta)  ﬁle"
        assert normalize_text(text) == normalize_text(text)


class TestNormalizedDigest:
    def test_digest_is_hex_stable(self):
        d = normalized_digest("Hello, World!")
        assert isinstance(d, str) and len(d) == 32
        assert normalized_digest("Hello, World!") == d
        # independent recomputation
        expect = hashlib.blake2b(
            normalize_text("Hello, World!").encode("utf-8"),
            digest_size=16, person=b"verbatim-n1",
        ).hexdigest()
        assert d == expect

    def test_normalization_equivalence(self):
        # differ only in case/punct/whitespace → same digest
        assert normalized_digest("Deploy V2!") == normalized_digest(
            "deploy-v2")
        assert normalized_digest("Café") == normalized_digest("cafe")

    def test_semantic_difference_survives(self):
        assert normalized_digest("deploy-v1") != normalized_digest(
            "deploy-v2")
        assert normalized_digest("I like X") != normalized_digest(
            "I no longer like X")
