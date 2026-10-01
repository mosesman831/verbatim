"""Tests for `norm/v2` — verbatim/text/norm_v2.py (SPEC_V7 §32.1,
V7-05.10/11).

Coverage: clitic/possessive table (D7-01/D7-03), identifier channel
exactness (V5-30.17 carried), UTF-8 strictness, folded-projection
offsets round-tripping into source bytes, Porter stem channel,
idempotence/determinism, and norm/v1 baseline agreement on non-clitic
text.
"""

from __future__ import annotations

import pytest

from verbatim.core.types_v7 import NormAnalysis, NormTerm
from verbatim.enrichment.normalize import normalize_text
from verbatim.text.norm_v2 import ANALYZER_ID, _porter_stem, analyze, fold


def text_terms(a: NormAnalysis):
    return [t.term for t in a.terms if t.channel == "text"]


def stem_terms(a: NormAnalysis):
    return [t.term for t in a.terms if t.channel == "stem"]


def id_terms(a: NormAnalysis):
    return [t.term for t in a.identifiers]


def surface(text: str, term: NormTerm) -> str:
    return text.encode("utf-8")[term.byte_start:term.byte_end].decode("utf-8")


# ---------------------------------------------------------------------------
# Analyzer identity / result shape
# ---------------------------------------------------------------------------


class TestContract:
    def test_analyzer_id_constant(self):
        assert ANALYZER_ID == "norm/v2"

    def test_result_analyzer_id(self):
        assert analyze("hello").analyzer_id == "norm/v2"

    def test_result_type(self):
        a = analyze("hello world")
        assert isinstance(a, NormAnalysis)
        assert isinstance(a.terms, tuple)
        assert isinstance(a.identifiers, tuple)
        assert all(isinstance(t, NormTerm) for t in a.terms)

    def test_channels_legal(self):
        a = analyze("Caroline's can't https://x.io 3.5 kg")
        assert {t.channel for t in a.terms} <= {"text", "stem"}
        assert all(t.channel == "identifier" for t in a.identifiers)

    def test_identifiers_not_in_terms(self):
        a = analyze("fix ABC-123 today")
        assert "ABC-123" not in {t.term for t in a.terms}
        assert id_terms(a) == ["ABC-123"]


# ---------------------------------------------------------------------------
# §32.1 rule 5 — clitic / possessive table (D7-01, D7-03)
# ---------------------------------------------------------------------------


class TestClitics:
    @pytest.mark.parametrize(
        "src,expected",
        [
            # possessive 's / ’s -> drop clitic (D7-01: no stray `s`)
            ("Caroline's", ["caroline"]),
            ("CAROLINE'S", ["caroline"]),
            ("Caroline’s", ["caroline"]),
            ("that's", ["that"]),
            ("it's", ["it"]),
            ("let's", ["let"]),
            ("dogs'", ["dogs"]),
            ("James'", ["james"]),
            ("James’", ["james"]),
            # 'd -> drop clitic
            ("he'd", ["he"]),
            ("she’d", ["she"]),
            ("i'd", []),          # base "i" ≤1 after split -> dropped
            ("i's", []),
            # n't -> not
            ("can't", ["can", "not"]),
            ("can’t", ["can", "not"]),
            ("CAN'T", ["can", "not"]),
            ("won't", ["won", "not"]),
            ("don't", ["don", "not"]),
            ("didn't", ["didn", "not"]),
            ("isn't", ["isn", "not"]),
            ("aren't", ["aren", "not"]),
            ("ain't", ["ain", "not"]),
            ("shan't", ["shan", "not"]),
            ("couldn't", ["couldn", "not"]),
            # 're -> are
            ("they're", ["they", "are"]),
            ("we're", ["we", "are"]),
            ("you’re", ["you", "are"]),
            # 'll -> will
            ("we'll", ["we", "will"]),
            ("you'll", ["you", "will"]),
            ("i'll", ["will"]),
            # 've -> have
            ("i've", ["have"]),
            ("they've", ["they", "have"]),
            ("could've", ["could", "have"]),
            # 'm -> am
            ("i'm", ["am"]),
            ("i’m", ["am"]),
            ("I'M", ["am"]),
            # stacked clitics
            ("i'd've", ["have"]),
            ("couldn't've", ["couldn", "not", "have"]),
            ("won't've", ["won", "not", "have"]),
            # non-table apostrophes still split (boundary); ≤1 dropped
            ("o'clock", ["clock"]),
            ("y'all", ["all"]),
            ("rock'n'roll", ["rock", "roll"]),
            ("ma'am", ["ma", "am"]),
            ("'tis", ["tis"]),
            ("'em", ["em"]),
            # spaced "’n’" leaves a standalone "n" (no split occurred —
            # the ≤1 guard applies only to post-split pieces, §32.1.5)
            ("rock ’n’ roll", ["rock", "n", "roll"]),
        ],
    )
    def test_clitic_table(self, src, expected):
        assert text_terms(analyze(src)) == expected

    def test_no_stray_s_term(self):
        # D7-03: the dropped clitic never becomes a term.
        a = analyze("Caroline's")
        assert "s" not in {t.term for t in a.terms}
        assert text_terms(a) == ["caroline"]

    def test_sentence(self):
        a = analyze("Caroline's dog can't find they're bone")
        assert text_terms(a) == [
            "caroline", "dog", "can", "not", "find", "they", "are", "bone",
        ]

    def test_clitic_terms_get_stems(self):
        a = analyze("can't")
        assert stem_terms(a) == ["can", "not"]

    def test_contraction_offsets(self):
        # "can" pins bytes 0-3; "not" pins the written "n't" (2-5).
        a = analyze("can't")
        by_term = {t.term: t for t in a.terms if t.channel == "text"}
        assert surface("can't", by_term["can"]) == "can"
        assert surface("can't", by_term["not"]) == "n't"

    def test_possessive_offset_excludes_clitic(self):
        a = analyze("Caroline's book")
        t = a.terms[0]
        assert surface("Caroline's book", t) == "Caroline"
        assert (t.byte_start, t.byte_end) == (0, 8)

    def test_re_clitic_offset(self):
        a = analyze("they're")
        are = [t for t in a.terms if t.term == "are"][0]
        assert surface("they're", are) == "'re"


# ---------------------------------------------------------------------------
# §32.1 rule 2 — identifier channel (exact bytes, excluded from folding)
# ---------------------------------------------------------------------------


class TestIdentifiers:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("go to https://example.com/a?b=1 now",
             ["https://example.com/a?b=1"]),
            ("see https://x.io/a.", ["https://x.io/a"]),
            ("(see https://x.io/a).", ["https://x.io/a"]),
            ("mirror ftp://files.example.com/dir",
             ["ftp://files.example.com/dir"]),
            ("mail bob@corp.io please", ["bob@corp.io"]),
            ("cc jane.doe+x@sub.domain.org", ["jane.doe+x@sub.domain.org"]),
            ("fix ABC-123 today", ["ABC-123"]),
            ("fix TICKET-42 now", ["TICKET-42"]),
            ("commit 0xdeadbeef", ["0xdeadbeef"]),
            ("hash abc1234", ["abc1234"]),
            ("requires v1.2.3", ["v1.2.3"]),
            ("uses 2.0.0-rc.1 ok", ["2.0.0-rc.1"]),
            ("pin 1.2.3+build.7", ["1.2.3+build.7"]),
            ("edit src/main.py", ["src/main.py"]),
            ("check /etc/hosts", ["/etc/hosts"]),
            ("open ~/docs/x.txt", ["~/docs/x.txt"]),
            ("run `rm -rf` now", ["`rm -rf`"]),
            ("weighs 3.5 kg", ["3.5 kg"]),
            ("weighs 3.5kg", ["3.5kg"]),
            ("about 100% sure", ["100%"]),
            ("drive 5 km/h fast", ["5 km/h"]),
            ("bake at 20°C", ["20°C"]),
            ("ping @alice_9", ["@alice_9"]),
            ("love #python", ["#python"]),
            ("see #123", ["#123"]),
            ("read C:\\temp\\f.txt", ["C:\\temp\\f.txt"]),
        ],
    )
    def test_identifier_extracted(self, src, expected):
        assert id_terms(analyze(src)) == expected

    @pytest.mark.parametrize(
        "src",
        [
            "fix ABC-123 today",
            "see https://x.io/a.",
            "mail bob@corp.io",
            "weighs 3.5 kg",
            "run `rm -rf` now",
            "ping @alice_9",
        ],
    )
    def test_identifier_exact_bytes(self, src):
        # V5-30.17 carried: identifier terms keep exact surface bytes.
        a = analyze(src)
        for t in a.identifiers:
            assert surface(src, t) == t.term
            assert t.channel == "identifier"

    def test_identifier_excluded_from_folding(self):
        # ticket letters/digits must not leak into the folded channels
        a = analyze("fix ABC-123 today")
        assert "abc" not in text_terms(a)
        assert "123" not in text_terms(a)
        assert text_terms(a) == ["fix", "today"]

    def test_url_parts_not_terms(self):
        a = analyze("see https://x.io/a now")
        for piece in ("https", "x", "io", "a"):
            assert piece not in text_terms(a)

    def test_code_span_inner_identifier_wins(self):
        # specific kinds beat the quoted wrapper (extract/v1 precedence)
        assert id_terms(analyze("`ABC-123`")) == ["ABC-123"]
        # no inner identifier -> the code span itself is kept verbatim
        assert id_terms(analyze("run `rm -rf`")) == ["`rm -rf`"]

    def test_measure_beats_bare_version(self):
        # "3.5" alone is a version id; "3.5 kg" is one measure id
        assert id_terms(analyze("pi is 3.14")) == ["3.14"]
        assert id_terms(analyze("weighs 3.5 kg")) == ["3.5 kg"]
        assert "kg" not in text_terms(analyze("weighs 3.5 kg"))

    def test_pure_digits_not_hash(self):
        a = analyze("dialed 1234567 twice")
        assert id_terms(a) == []
        assert "1234567" in text_terms(a)

    def test_hex_word_not_hash(self):
        # pure a-f letters are words, not hashes
        a = analyze("facaded acceded")
        assert id_terms(a) == []
        assert text_terms(a) == ["facaded", "acceded"]

    def test_lowercase_not_ticket(self):
        a = analyze("abc-123")
        assert id_terms(a) == []
        assert text_terms(a) == ["abc", "123"]

    def test_and_or_not_path(self):
        a = analyze("and/or")
        assert id_terms(a) == []
        assert text_terms(a) == ["and", "or"]

    def test_date_not_path(self):
        a = analyze("on 2025/03/14")
        assert id_terms(a) == []
        assert text_terms(a) == ["on", "2025", "03", "14"]

    def test_number_without_unit_is_term(self):
        a = analyze("order 42 now")
        assert id_terms(a) == []
        assert "42" in text_terms(a)

    def test_handle_not_glued(self):
        # "x@alice" mid-word is not a handle; email wins whole anyway
        a = analyze("from bob@corp.io to @alice")
        assert id_terms(a) == ["bob@corp.io", "@alice"]

    def test_identifiers_sorted_by_offset(self):
        a = analyze("@b then ABC-1 then https://x.io then 2 kg")
        offsets = [(t.byte_start, t.byte_end) for t in a.identifiers]
        assert offsets == sorted(offsets)


# ---------------------------------------------------------------------------
# §32.1 rule 3 — the matching projection (fold)
# ---------------------------------------------------------------------------


class TestFold:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("CAFÉ", "cafe"),
            ("naïve", "naive"),
            ("Straße", "strasse"),
            ("ﬁle", "file"),          # ﬁ ligature -> fi (NFKC)
            ("①", "1"),              # circled digit -> 1
            ("Ｗｉｄｅ", "wide"),       # fullwidth -> ascii
            ("İ", "i"),              # turkish dotted I -> i
            ("Œuvre", "œuvre"),      # Œ has no NFKC decomposition
            ("Ǆungla", "dzungla"),
            ("Hello, World!", "hello, world!"),  # punct kept in projection
            ("²x", "2x"),
        ],
    )
    def test_fold_values(self, src, expected):
        assert fold(src) == expected

    def test_fold_matches_term_surfaces(self):
        # for plain words the emitted term equals fold(surface)
        for word in ("CAFÉ", "Naïve", "STRASSE", "Running"):
            a = analyze(word)
            assert text_terms(a) == [fold(word)]

    def test_fold_idempotent(self):
        for s in ("Café", "ß", "İ", "ﬁle", "x²"):
            assert fold(fold(s)) == fold(s)

    def test_analyze_folds_diacritics(self):
        assert text_terms(analyze("naïve café")) == ["naive", "cafe"]


# ---------------------------------------------------------------------------
# Byte offsets — every term/id round-trips into original UTF-8 bytes
# ---------------------------------------------------------------------------


class TestOffsets:
    @pytest.mark.parametrize(
        "src",
        [
            "Caroline's dog can't stay",
            "héllo wörld naïve café",
            "fix ABC-123 and mail bob@corp.io",
            "② review the ﬁle", "Straße ②",
        ],
    )
    def test_all_terms_round_trip(self, src):
        a = analyze(src)
        raw = src.encode("utf-8")
        for t in a.terms:
            seg = raw[t.byte_start:t.byte_end]
            seg.decode("utf-8")  # never splits a codepoint
            assert 0 <= t.byte_start < t.byte_end <= len(raw)
        for t in a.identifiers:
            assert raw[t.byte_start:t.byte_end].decode("utf-8") == t.term

    def test_multibyte_exact_offsets(self):
        a = analyze("héllo wörld")
        # h(1) é(2) l l o = 6 bytes; space; w(1) ö(2) r l d = 6 bytes
        t0 = [t for t in a.terms if t.term == "hello"][0]
        t1 = [t for t in a.terms if t.term == "world"][0]
        assert (t0.byte_start, t0.byte_end) == (0, 6)
        assert (t1.byte_start, t1.byte_end) == (7, 13)
        assert surface("héllo wörld", t0) == "héllo"
        assert surface("héllo wörld", t1) == "wörld"

    def test_expansion_offsets(self):
        # "ﬁ" is 3 UTF-8 bytes that fold to 2 chars
        a = analyze("ﬁle")
        t = a.terms[0]
        assert t.term == "file"
        assert surface("ﬁle", t) == "ﬁle"

    def test_offsets_nondecreasing(self):
        a = analyze("Caroline's can't visit https://x.io at 3.5 kg")
        starts = [t.byte_start for t in a.terms]
        assert starts == sorted(starts)

    def test_folded_term_maps_to_surface(self):
        # for non-clitic words: fold(decoded span) == emitted term
        a = analyze("Héllo WORLD")
        for t in a.terms:
            if t.channel == "text":
                assert fold(surface("Héllo WORLD", t)) == t.term


# ---------------------------------------------------------------------------
# §32.1 rules 6-7 — stem channel; stopwords indexed
# ---------------------------------------------------------------------------


class TestStems:
    def test_stem_channel_present(self):
        a = analyze("running")
        assert text_terms(a) == ["running"]
        assert stem_terms(a) == ["run"]

    def test_stem_offsets_match_text(self):
        a = analyze("running quickly")
        texts = [t for t in a.terms if t.channel == "text"]
        stems = [t for t in a.terms if t.channel == "stem"]
        assert len(texts) == len(stems)
        for t, s in zip(texts, stems):
            assert (t.byte_start, t.byte_end) == (s.byte_start, s.byte_end)

    @pytest.mark.parametrize(
        "word,stem",
        [
            # canonical Porter (1980) outputs — tartarus.org reference
            ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
            ("caress", "caress"), ("cats", "cat"), ("feed", "feed"),
            ("agreed", "agre"), ("plastered", "plaster"), ("bled", "bled"),
            ("motoring", "motor"), ("sing", "sing"),
            ("conflated", "conflat"), ("troubled", "troubl"),
            ("sized", "size"), ("hopping", "hop"), ("tanned", "tan"),
            ("falling", "fall"), ("hissing", "hiss"), ("fizzed", "fizz"),
            ("failing", "fail"), ("filing", "file"), ("happy", "happi"),
            ("sky", "sky"), ("relational", "relat"),
            ("conditional", "condit"), ("digitizer", "digit"),
            ("vietnamization", "vietnam"), ("operator", "oper"),
            ("decisiveness", "decis"), ("sensibiliti", "sensibl"),
            ("triplicate", "triplic"), ("formative", "form"),
            ("electriciti", "electr"), ("electrical", "electr"),
            ("hopeful", "hope"), ("goodness", "good"),
            ("revival", "reviv"), ("allowance", "allow"),
            ("inference", "infer"), ("airliner", "airlin"),
            ("adjustable", "adjust"), ("irritant", "irrit"),
            ("replacement", "replac"), ("dependent", "depend"),
            ("adoption", "adopt"), ("communism", "commun"),
            ("activate", "activ"), ("homologous", "homolog"),
            ("effective", "effect"), ("bowdlerize", "bowdler"),
            ("probate", "probat"), ("rate", "rate"), ("cease", "ceas"),
            ("controll", "control"), ("roll", "roll"),
            ("running", "run"), ("caroline", "carolin"),
            ("university", "univers"), ("news", "new"), ("moss", "moss"),
        ],
    )
    def test_porter_reference(self, word, stem):
        assert _porter_stem(word) == stem

    def test_stopwords_indexed_never_dropped(self):
        # §32.1 rule 7 / V7-05.11 — no hard drop list
        a = analyze("the a of and or is it in on at to")
        for w in "the a of and or is it in on at to".split():
            assert w in text_terms(a)


# ---------------------------------------------------------------------------
# Determinism / idempotence / edge cases
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic(self):
        s = "Caroline's can't fix ABC-123 at https://x.io for 3.5 kg"
        assert analyze(s) == analyze(s)

    def test_idempotent_via_text_field(self):
        # analyzing the projection reproduces the same terms/identifiers
        s = "Caroline's can't fix ABC-123 at https://x.io"
        a = analyze(s)
        b = analyze(a.text)
        assert [(t.term, t.channel) for t in b.terms] == [
            (t.term, t.channel) for t in a.terms
        ]
        assert [t.term for t in b.identifiers] == [
            t.term for t in a.identifiers
        ]

    def test_text_field_is_folded_projection(self):
        a = analyze("Hello CAFÉ World")
        assert a.text == "hello cafe world"

    def test_text_field_keeps_identifier_verbatim(self):
        a = analyze("see https://x.io now")
        assert "https://x.io" in a.text
        assert a.text == "see https://x.io now"

    def test_empty(self):
        a = analyze("")
        assert a.terms == () and a.identifiers == () and a.text == ""

    def test_whitespace_only(self):
        assert analyze("   \n\t  ").terms == ()

    def test_punctuation_only(self):
        assert analyze("!!! ??? ...").terms == ()

    def test_single_char_standalone(self):
        # standalone single chars index (only post-split ≤1 pieces drop)
        assert text_terms(analyze("a")) == ["a"]
        assert text_terms(analyze("I")) == ["i"]

    def test_numbers_as_terms(self):
        assert text_terms(analyze("order 66")) == ["order", "66"]

    def test_bytes_input(self):
        assert text_terms(analyze("can't".encode())) == ["can", "not"]

    def test_bytes_invalid_rejected(self):
        with pytest.raises(UnicodeDecodeError):
            analyze(b"\xff\xfe")

    def test_surrogate_rejected(self):
        with pytest.raises(UnicodeEncodeError):
            analyze("bad \ud800 input")

    def test_underscore_word(self):
        # UAX-29: underscore is a word char — one term (differs from
        # norm/v1 which folds Pc to space; documented deviation)
        assert text_terms(analyze("foo_bar")) == ["foo_bar"]

    def test_dotted_unquoted_splits(self):
        # only *quoted* code spans are identifiers (§32.1)
        assert text_terms(analyze("foo.bar")) == ["foo", "bar"]

    def test_hyphen_splits(self):
        assert text_terms(analyze("well-known")) == ["well", "known"]

    def test_mixed_document(self):
        a = analyze("James' ticket ABC-123: can't open `main.py` at 90%")
        assert text_terms(a) == ["james", "ticket", "can", "not", "open", "at"]
        assert id_terms(a) == ["ABC-123", "`main.py`", "90%"]


# ---------------------------------------------------------------------------
# norm/v1 baseline agreement on non-clitic text
# ---------------------------------------------------------------------------


class TestV1Baseline:
    @pytest.mark.parametrize(
        "src",
        [
            "The quick brown fox jumps over 13 lazy dogs",
            "hello world this is a test",
            "alpha beta gamma delta",
            "New York City 2023",
            "order 66 executed",
            "state of the art systems",
            "Hello, World!",
            "well-known facts about plants",
            "multi  word   spacing",
            "end of sentence.",
        ],
    )
    def test_non_clitic_matches_norm_v1(self, src):
        # For text without clitics/identifiers, the text channel equals
        # the norm/v1 projection split on spaces (V7-05.10).
        assert text_terms(analyze(src)) == normalize_text(src).split()
