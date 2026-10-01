"""entities_v2 — canonical entity mentions, query-side extraction, alias
rules A1–A6, expansion bound, entity IDF (V7-08.01–06, §32.7 alias/v1).

NormAnalysis fixtures are built by ``norm(text)`` below: ``text`` carries
the analyzed source text (the field the entity pass consumes for surfaces
and capitalization) and ``terms`` mirrors the analyzer's folded output
with byte offsets into that text.
"""

from __future__ import annotations

import re

import pytest

from verbatim.core.types_v7 import (
    AliasMethod,
    AliasRow,
    AliasState,
    EntityMention,
    MentionRole,
    NormAnalysis,
    NormTerm,
)
from verbatim.enrichment.entities_v2 import (
    EXTRACTOR_ID,
    canon,
    entity_weight,
    expand_query,
    extract_mentions,
    extract_query_entities,
    idf_weight,
    propose_aliases,
    strip_possessive,
)

_WORD = re.compile(r"[^\W_][\w'’.-]*", re.UNICODE)


def norm(text: str) -> NormAnalysis:
    """Stand-in for ``text.norm_v2.analyze``: folded terms, byte offsets
    into the source text, source text carried on ``.text``."""
    terms = []
    for m in _WORD.finditer(text):
        tok = m.group(0)
        start = len(text[: m.start()].encode("utf-8"))
        end = start + len(tok.encode("utf-8"))
        terms.append(NormTerm(tok.casefold(), "text", start, end))
    return NormAnalysis(
        analyzer_id="norm/v2",
        terms=tuple(terms),
        identifiers=(),
        text=text,
    )


def mention(c: str, unit: str = "u1", role: MentionRole = MentionRole.MENTION,
            surface: str = "") -> EntityMention:
    return EntityMention(
        canon=c, surface=surface or c, unit_id=unit,
        byte_start=0, byte_end=0, role=role,
    )


def speaker(c: str, unit: str = "u1") -> EntityMention:
    return mention(c, unit, MentionRole.SPEAKER)


def surfaces(text: str, ms) -> list:
    raw = text.encode("utf-8")
    return [raw[m.byte_start:m.byte_end].decode("utf-8") for m in ms]


# ---------------------------------------------------------------------------
# canon / strip_possessive — D7-02 mechanism
# ---------------------------------------------------------------------------


class TestCanon:
    def test_possessive_straight(self):
        assert canon("Caroline's") == "caroline"

    def test_possessive_curly(self):
        assert canon("Caroline’s") == "caroline"

    def test_all_caps(self):
        assert canon("CAROLINE") == "caroline"

    def test_lowercase(self):
        assert canon("caroline") == "caroline"

    def test_mixed_case(self):
        assert canon("CaRoLiNe") == "caroline"

    def test_trailing_apostrophe(self):
        assert canon("James'") == "james"

    def test_possessive_inside_multiword(self):
        # possessive is stripped per token, not just at the surface end
        assert canon("Caroline's Mural") == "caroline mural"

    def test_multiword(self):
        assert canon("Alice Chen") == "alice chen"

    def test_whitespace_collapse(self):
        assert canon("  Alice   Chen\t") == "alice chen"

    def test_diacritics(self):
        assert canon("José") == "jose"

    def test_nfkc_ligature(self):
        assert canon("ﬁle") == "file"  # ﬁ U+FB01

    def test_nfkc_fullwidth(self):
        assert canon("Ａｌｉｃｅ") == "alice"  # fullwidth ALICE

    def test_punctuation_folds(self):
        assert canon("St. John") == "st john"

    def test_idempotent(self):
        for s in ("Caroline's", "Alice Chen", "José", "St. John"):
            assert canon(canon(s)) == canon(s)

    def test_empty_and_none(self):
        assert canon("") == ""
        assert canon(None) == ""

    def test_d702_trio(self):
        # lowercase, possessive and re-cased query mentions share postings
        assert canon("caroline's") == canon("CAROLINE") == canon("Caroline")

    def test_strip_possessive_internal_apostrophe(self):
        assert strip_possessive("O'Brien") == "O'Brien"

    def test_strip_possessive_uppercase_s(self):
        assert canon("CAROLINE'S") == "caroline"


# ---------------------------------------------------------------------------
# extract_mentions — write-side capitalized-run extraction
# ---------------------------------------------------------------------------


class TestExtractMentions:
    def test_multiword_run(self):
        ms = extract_mentions(norm("Alice Chen called yesterday"), "u1")
        assert [m.canon for m in ms] == ["alice chen"]

    def test_single_capital(self):
        ms = extract_mentions(norm("Caroline painted a mural"), "u1")
        assert [m.canon for m in ms] == ["caroline"]

    def test_connector_run(self):
        ms = extract_mentions(norm("Ruth van der Berg spoke up"), "u1")
        assert [m.canon for m in ms] == ["ruth van der berg"]

    def test_and_splits_two_entities(self):
        ms = extract_mentions(norm("Alice and Bob left early"), "u1")
        assert {m.canon for m in ms} == {"alice", "bob"}

    def test_sentence_initial_stop_suppressed(self):
        ms = extract_mentions(norm("The report was late"), "u1")
        assert "the" not in {m.canon for m in ms}
        assert ms == []

    def test_mid_sentence_the_joins_run(self):
        ms = extract_mentions(norm("I love The Beatles a lot"), "u1")
        assert "the beatles" in {m.canon for m in ms}

    def test_possessive_surface(self):
        ms = extract_mentions(norm("Caroline's painting won"), "u1")
        (m,) = [m for m in ms if m.canon == "caroline"]
        assert m.surface == "Caroline's"

    def test_speaker_mention(self):
        ms = extract_mentions(norm("hello there"), "u1", speaker="Caroline")
        (m,) = [m for m in ms if m.role == MentionRole.SPEAKER]
        assert m.canon == "caroline"
        assert m.surface == "Caroline"
        assert (m.byte_start, m.byte_end) == (0, 0)

    def test_byte_offsets_ascii(self):
        text = "See Alice Chen today."
        ms = extract_mentions(norm(text), "u1")
        (m,) = [m for m in ms if m.canon == "alice chen"]
        assert surfaces(text, [m]) == ["Alice Chen"]

    def test_byte_offsets_unicode(self):
        text = "José met Zoë."
        ms = extract_mentions(norm(text), "u1")
        canons = {m.canon for m in ms}
        assert {"jose", "zoe"} <= canons
        for m in ms:
            # every span slices the source bytes exactly
            assert surfaces(text, [m])[0] == m.surface

    def test_all_lowercase_no_mentions(self):
        assert extract_mentions(norm("nothing to see here"), "u1") == []

    def test_single_char_i_excluded(self):
        assert extract_mentions(norm("I went home"), "u1") == []

    def test_single_char_a_excluded(self):
        assert extract_mentions(norm("A test"), "u1") == []

    def test_empty_text_speaker_only(self):
        ms = extract_mentions(norm(""), "u1", speaker="Bob")
        assert [(m.canon, m.role) for m in ms] == [("bob", MentionRole.SPEAKER)]

    def test_run_not_split(self):
        ms = extract_mentions(norm("New York City is big"), "u1")
        assert [m.canon for m in ms] == ["new york city"]

    def test_surface_case_preserved(self):
        ms = extract_mentions(norm("ALICE left"), "u1")
        assert ms[0].surface == "ALICE"
        assert ms[0].canon == "alice"

    def test_sentence_boundary_breaks_run(self):
        ms = extract_mentions(norm("Alice won. Bob lost."), "u1")
        assert {m.canon for m in ms} == {"alice", "bob"}

    def test_trailing_connector_trimmed(self):
        ms = extract_mentions(norm("Alice van asked about it"), "u1")
        assert [m.canon for m in ms] == ["alice"]
        assert ms[0].surface == "Alice"

    def test_default_role_is_mention(self):
        ms = extract_mentions(norm("Bob left"), "u1")
        assert ms[0].role == MentionRole.MENTION


# ---------------------------------------------------------------------------
# V8-13.03 canon hygiene — contraction boundary + leading vocatives (K76)
# ---------------------------------------------------------------------------


class TestCanonHygiene:
    """D8-20: contraction fragments and vocative pairs never mint canons."""

    # -- (a) apostrophe-contraction boundary -------------------------------

    @pytest.mark.parametrize("text", [
        "Can't believe it", "You're right", "Don't do that",
        "We've met before", "She'll call back", "He'd left early",
        "That'll work", "I'm here", "Isn't it late", "Couldn't agree",
        "Won't help much", "Didn't see it", "Wouldn't know",
        "Shouldn't stay", "Aren't we there", "Weren't they",
    ])
    def test_contraction_mints_no_canon(self, text):
        assert extract_mentions(norm(text), "u1") == []

    def test_curly_apostrophe_contraction(self):
        assert extract_mentions(norm("Can’t stop"), "u1") == []
        assert extract_mentions(norm("You’re right"), "u1") == []

    def test_contraction_fragment_canons_never_minted(self):
        # the audit's measured fragment classes — zero on any input
        texts = ["Can't", "You're", "Don't", "We've", "She'll", "He'd",
                 "That'll", "I'm", "You're welcome, Nate"]
        canons = {m.canon for t in texts
                      for m in extract_mentions(norm(t), "u1")}
        for frag in ("can t", "you re", "don t", "we ve", "she ll",
                     "he d", "that ll", "i m"):
            assert frag not in canons

    def test_contraction_closes_open_run(self):
        # "Can't" ends the run at the boundary — it neither joins nor
        # glues the neighbors into one canon
        ms = extract_mentions(norm("Alice Can't Stop"), "u1")
        assert [m.canon for m in ms] == ["alice", "stop"]

    def test_possessive_s_unaffected(self):
        # 's stays on the possessive path — the canon strips the tail,
        # the surface keeps it byte-exact
        ms = extract_mentions(norm("Caroline's painting won"), "u1")
        (m,) = ms
        assert m.canon == "caroline"
        assert m.surface == "Caroline's"

    def test_possessive_multiword_unaffected(self):
        ms = extract_mentions(norm("Alice's Mural is nice"), "u1")
        assert [m.canon for m in ms] == ["alice mural"]

    def test_internal_apostrophe_name_kept(self):
        # 'b is no contraction suffix — the token participates
        ms = extract_mentions(norm("O'Brien left"), "u1")
        assert [m.canon for m in ms] == ["o brien"]

    def test_contraction_inside_name_kept(self):
        # 'a/'b/'n are not in the owned suffix list
        ms = extract_mentions(norm("D'Artagnan arrived"), "u1")
        assert [m.canon for m in ms] == ["d artagnan"]

    def test_contraction_query_side(self):
        out = extract_query_entities(norm("Can't we ask Nate"), {"nate"})
        assert "nate" in out
        assert "can t" not in out

    # -- (b) leading vocative / greeting / interjection strip --------------

    @pytest.mark.parametrize("text,expected", [
        ("Thanks Nate, see you", "nate"),
        ("Thanks Jon", "jon"),
        ("Thank You Bob", "you bob"),   # "you" is not in the owned list
        ("Hey Gina", "gina"),
        ("Hi Nate", "nate"),
        ("Hello Bob", "bob"),
        ("Congrats Nate", "nate"),
        ("Congratulations Jon", "jon"),
        ("Wow Alice that is great", "alice"),
        ("Oh Alice left", "alice"),
        ("Dear John, write soon", "john"),
        ("Good Morning Alice", "morning alice"),
        ("Well Gina said yes", "gina"),
        ("Sorry Jon I missed it", "jon"),
        ("Please Gina call me", "gina"),
        ("Okay Bob see you", "bob"),
    ])
    def test_vocative_strips_to_real_name(self, text, expected):
        ms = extract_mentions(norm(text), "u1")
        assert [m.canon for m in ms] == [expected]

    @pytest.mark.parametrize("text", [
        "Thanks!", "Hey", "Hello", "Wow", "Oh well", "Sorry.",
    ])
    def test_vocative_alone_no_canon(self, text):
        assert extract_mentions(norm(text), "u1") == []

    def test_vocative_mid_run_does_not_glue(self):
        # a vocative between two names closes the run instead of minting
        # one "alice oh bob" canon
        ms = extract_mentions(norm("Alice Oh Bob left"), "u1")
        assert [m.canon for m in ms] == ["alice", "bob"]

    def test_vocative_case_insensitive(self):
        ms = extract_mentions(norm("THANKS Jon"), "u1")
        assert [m.canon for m in ms] == ["jon"]

    def test_vocative_query_side(self):
        out = extract_query_entities(norm("Hey what about nate"), {"nate"})
        assert "nate" in out
        assert "hey nate" not in out
        assert "hey" not in out

    # -- (c) legitimate names untouched + deriver version -------------------

    def test_legit_multiword_names_still_mint(self):
        ms = extract_mentions(norm("Alice Chen called yesterday"), "u1")
        assert [m.canon for m in ms] == ["alice chen"]
        ms = extract_mentions(norm("Ruth van der Berg spoke up"), "u1")
        assert [m.canon for m in ms] == ["ruth van der berg"]
        ms = extract_mentions(norm("New York City is big"), "u1")
        assert [m.canon for m in ms] == ["new york city"]
        ms = extract_mentions(norm("McDonald's Corporation announced"), "u1")
        assert [m.canon for m in ms] == ["mcdonald corporation"]

    def test_single_token_name_untouched(self):
        # K76: "Will" mints exactly as before (sent-initial function-word
        # suppression is the pre-existing rule, not this change)
        assert [m.canon for m in
                extract_mentions(norm("saw Will yesterday"), "u1")] == ["will"]
        assert extract_mentions(norm("Will went home"), "u1") == []

    def test_k76_scenario(self):
        ms = extract_mentions(norm("Thanks Nate, see you"), "u1")
        assert [m.canon for m in ms] == ["nate"]
        assert extract_mentions(norm("can't"), "u1") == []
        assert extract_mentions(norm("Can't"), "u1") == []

    def test_deriver_version_bumped(self):
        # V8-13.03(c): derived rows change only under a version bump
        assert EXTRACTOR_ID == "entities/v2.1"


# ---------------------------------------------------------------------------
# extract_query_entities — V7-08.02 (no capitalization required)
# ---------------------------------------------------------------------------


class TestExtractQueryEntities:
    def test_lowercase_known_canon(self):
        # D7-02: a lowercase query mention reaches the same canon
        out = extract_query_entities(
            norm("what did caroline paint"), {"caroline"})
        assert out == ["caroline"]

    def test_possessive_query_mention(self):
        out = extract_query_entities(
            norm("what are caroline's hobbies"), {"caroline"})
        assert "caroline" in out

    def test_ngram_match(self):
        out = extract_query_entities(
            norm("where is alice chen"), {"alice chen"})
        assert "alice chen" in out

    def test_longest_match_wins(self):
        out = extract_query_entities(
            norm("where is alice chen"), {"alice chen", "alice", "chen"})
        assert "alice chen" in out
        assert "alice" not in out
        assert "chen" not in out

    def test_capitalized_unknown_emitted(self):
        out = extract_query_entities(norm("What did Zorblax do"), set())
        assert "zorblax" in out

    def test_ngram_bound_four(self):
        # a 5-token canon can never match (n-grams are capped at 4)
        out = extract_query_entities(
            norm("a b c d e f"), {"a b c d e"})
        assert "a b c d e" not in out

    def test_ngram_never_crosses_sentence(self):
        out = extract_query_entities(
            norm("Al went. Chen stayed."), {"al chen"})
        assert "al chen" not in out

    def test_unknown_lowercase_not_emitted(self):
        out = extract_query_entities(
            norm("did the zebra move"), {"caroline"})
        assert out == []

    def test_dedup_repeated(self):
        out = extract_query_entities(
            norm("caroline met caroline"), {"caroline"})
        assert out == ["caroline"]

    def test_empty_text(self):
        assert extract_query_entities(norm(""), {"caroline"}) == []

    def test_known_canons_normalized(self):
        # surfaces in the vocabulary are canon()'d before matching
        out = extract_query_entities(
            norm("tell me about alice chen"), {"Alice Chen"})
        assert "alice chen" in out


# ---------------------------------------------------------------------------
# propose_aliases — §32.7 alias/v1 rules A1–A6
# ---------------------------------------------------------------------------


class TestAliasA1:
    def test_token_subset_active(self):
        rows = propose_aliases([mention("alice")], {"alice chen"})
        (r,) = rows
        assert (r.canon, r.alias_canon) == ("alice chen", "alice")
        assert r.rule_id == "A1"
        assert r.state == AliasState.ACTIVE

    def test_no_superset_no_row(self):
        assert propose_aliases([mention("alice")], {"bob chen"}) == []

    def test_conflict_is_candidate_never_active(self):
        # same-name different-person: two plausible canonicals abstain
        rows = propose_aliases(
            [mention("jordan")], {"jordan smith", "jordan lee"})
        assert len(rows) == 2
        assert all(r.state == AliasState.CANDIDATE for r in rows)
        assert all(r.rule_id == "A6" for r in rows)
        assert {r.canon for r in rows} == {"jordan smith", "jordan lee"}
        assert all(r.alias_canon == "jordan" for r in rows)

    def test_multiword_mention_is_canonical_candidate(self):
        # "alice" co-mentioned with "alice chen" aliases it even before
        # the longer canon lands in the vocabulary
        rows = propose_aliases(
            [mention("alice"), mention("alice chen")], set())
        assert [(r.canon, r.alias_canon, r.state) for r in rows] == [
            ("alice chen", "alice", AliasState.ACTIVE)]

    def test_subset_chain_conflict(self):
        # "alice" ⊂ both "alice chen" and "alice chen wu" → A6 conflict;
        # "alice chen" ⊂ only "alice chen wu" → A1 active
        rows = propose_aliases(
            [mention("alice"), mention("alice chen")],
            {"alice chen wu"})
        got = {(r.canon, r.alias_canon): (r.rule_id, r.state)
               for r in rows}
        assert got == {
            ("alice chen wu", "alice chen"): ("A1", AliasState.ACTIVE),
            ("alice chen", "alice"): ("A6", AliasState.CANDIDATE),
            ("alice chen wu", "alice"): ("A6", AliasState.CANDIDATE),
        }

    def test_evidence_count(self):
        ms = [mention("alice", "u1"), mention("alice", "u2"),
              mention("alice", "u3")]
        (r,) = propose_aliases(ms, {"alice chen"})
        assert r.evidence_count == 3


class TestAliasA2:
    def test_first_last_initial(self):
        rows = propose_aliases([mention("alice c")], {"alice chen"})
        (r,) = rows
        assert (r.canon, r.alias_canon, r.rule_id, r.state) == (
            "alice chen", "alice c", "A2", AliasState.ACTIVE)

    def test_initial_last(self):
        rows = propose_aliases([mention("a chen")], {"alice chen"})
        (r,) = rows
        assert (r.canon, r.alias_canon) == ("alice chen", "a chen")
        assert r.rule_id == "A2"

    def test_no_match(self):
        assert propose_aliases([mention("alice c")], {"bob chen"}) == []

    def test_ambiguous_initial_is_candidate(self):
        rows = propose_aliases(
            [mention("a chen")], {"alice chen", "anna chen"})
        assert len(rows) == 2
        assert all(r.state == AliasState.CANDIDATE for r in rows)
        assert all(r.rule_id == "A6" for r in rows)

    def test_single_token_canonical_never_a2_target(self):
        # "caroline c" must not alias to bare "caroline"
        assert propose_aliases([mention("caroline c")], {"caroline"}) == []

    def test_two_initials_no_match(self):
        assert propose_aliases([mention("a c")], {"alice chen"}) == []


class TestAliasA3:
    def test_call_me(self):
        rows = propose_aliases(
            [speaker("melanie")], {"melanie"}, texts=["call me Mel"])
        (r,) = rows
        assert (r.canon, r.alias_canon, r.rule_id, r.state) == (
            "melanie", "mel", "A3", AliasState.ACTIVE)

    def test_my_name_is(self):
        rows = propose_aliases(
            [speaker("melanie")], {"melanie"}, texts=["my name is Mel"])
        (r,) = rows
        assert (r.canon, r.alias_canon) == ("melanie", "mel")

    def test_goes_by_with_subject(self):
        rows = propose_aliases(
            [mention("melanie")], {"melanie"},
            texts=["Melanie goes by Mel"])
        (r,) = rows
        assert (r.canon, r.alias_canon, r.rule_id) == ("melanie", "mel", "A3")

    def test_goes_by_speaker_fallback(self):
        rows = propose_aliases(
            [speaker("melanie")], {"melanie"}, texts=["I goes by Mel"])
        # "goes by" with no capitalized subject resolves to the speaker
        assert (rows[0].canon, rows[0].alias_canon) == ("melanie", "mel")

    def test_paren_form(self):
        rows = propose_aliases(
            [mention("melanie")], {"melanie"}, texts=["Melanie (Mel) spoke"])
        (r,) = rows
        assert (r.canon, r.alias_canon) == ("melanie", "mel")
        assert r.rule_id == "A3"

    def test_for_short(self):
        rows = propose_aliases(
            [mention("melanie")], {"melanie"},
            texts=["Melanie, Mel for short"])
        (r,) = rows
        assert (r.canon, r.alias_canon) == ("melanie", "mel")

    def test_call_me_needs_unique_speaker(self):
        # two distinct speakers → the pattern cannot pin a canonical
        rows = propose_aliases(
            [speaker("melanie", "u1"), speaker("caroline", "u2")],
            {"melanie", "caroline"},
            texts=["call me Mel"])
        assert all(r.alias_canon != "mel" for r in rows)

    def test_call_me_no_speaker(self):
        rows = propose_aliases([mention("bob")], {"bob"},
                               texts=["call me Mel"])
        assert rows == []

    def test_identity_pair_skipped(self):
        rows = propose_aliases(
            [speaker("melanie")], {"melanie"},
            texts=["my name is Melanie"])
        assert rows == []

    def test_no_texts_no_a3(self):
        assert propose_aliases([speaker("melanie")], {"melanie"}) == []


class TestAliasA4:
    def _mentions(self):
        # "jonathon" and "jonathan" each co-mentioned with alice + bob
        return [
            mention("jonathon", "u1"), mention("alice", "u1"),
            mention("bob", "u1"),
            mention("jonathan", "u2"), mention("alice", "u2"),
            mention("bob", "u2"),
        ]

    def test_near_spelling_candidate(self):
        rows = propose_aliases(self._mentions(), {"jonathan"})
        (r,) = [r for r in rows if r.rule_id == "A4"]
        assert (r.canon, r.alias_canon) == ("jonathan", "jonathon")
        assert r.state == AliasState.CANDIDATE

    def test_edit_distance_two_no_row(self):
        ms = [mention("katherine", "u1"), mention("alice", "u1"),
              mention("bob", "u1"),
              mention("kathryn", "u2"), mention("alice", "u2"),
              mention("bob", "u2")]
        rows = propose_aliases(ms, {"katherine", "kathryn"})
        assert [r for r in rows if r.rule_id == "A4"] == []

    def test_short_names_no_row(self):
        ms = [mention("bob", "u1"), mention("alice", "u1"),
              mention("carol", "u1"),
              mention("bod", "u2"), mention("alice", "u2"),
              mention("carol", "u2")]
        rows = propose_aliases(ms, set())
        assert [r for r in rows if r.rule_id == "A4"] == []

    def test_shared_comention_floor(self):
        # ed ≤ 1 and len ≥ 6 but only one shared co-mention → no row
        ms = [mention("andrea", "u1"), mention("alice", "u1"),
              mention("andreas", "u2"), mention("alice", "u2")]
        rows = propose_aliases(ms, {"andrea", "andreas"})
        assert [r for r in rows if r.rule_id == "A4"] == []

    def test_never_active(self):
        rows = propose_aliases(self._mentions(), {"jonathan"})
        assert all(r.state != AliasState.ACTIVE
                   for r in rows if r.rule_id == "A4")


class TestAliasA5A6:
    def test_caller_alias_active(self):
        rows = propose_aliases(
            [], {"alice chen"}, caller_aliases=[("Alice Chen", "Al")])
        (r,) = rows
        assert (r.canon, r.alias_canon, r.rule_id, r.state, r.method) == (
            "alice chen", "al", "A5", AliasState.ACTIVE,
            AliasMethod.CALLER)

    def test_caller_identity_skipped(self):
        assert propose_aliases([], set(),
                               caller_aliases=[("mel", "Mel")]) == []

    def test_cross_rule_conflict_candidate(self):
        # A1 says alice → alice chen; A3 says alice → bob: conflict → A6
        rows = propose_aliases(
            [mention("alice"), mention("bob")],
            {"alice chen"},
            texts=["Bob (Alice) said hi"])
        al_rows = [r for r in rows if r.alias_canon == "alice"]
        assert len(al_rows) == 2
        assert all(r.rule_id == "A6" for r in al_rows)
        assert all(r.state == AliasState.CANDIDATE for r in al_rows)
        assert {r.canon for r in al_rows} == {"alice chen", "bob"}

    def test_scope_and_generation(self):
        (r,) = propose_aliases([mention("alice")], {"alice chen"},
                               scope_id="s1", now_generation=7)
        assert r.scope_id == "s1"
        assert r.generation == 7

    def test_deterministic_sorted_output(self):
        ms = [mention("alice"), mention("zoe")]
        a = propose_aliases(ms, {"alice chen", "zoe wu"})
        b = propose_aliases(list(reversed(ms)), {"zoe wu", "alice chen"})
        assert a == b
        assert [(r.canon, r.alias_canon) for r in a] == sorted(
            (r.canon, r.alias_canon) for r in a)

    def test_row_fields(self):
        (r,) = propose_aliases([mention("alice")], {"alice chen"})
        assert isinstance(r, AliasRow)
        assert r.method == AliasMethod.RULE
        assert r.evidence_count >= 1


# ---------------------------------------------------------------------------
# expand_query — V7-08.04 bounded alias expansion
# ---------------------------------------------------------------------------


def _alias(cn: str, al: str, state: AliasState = AliasState.ACTIVE
           ) -> AliasRow:
    return AliasRow(
        scope_id="s", canon=cn, alias_canon=al, rule_id="A1",
        evidence_count=1, method=AliasMethod.RULE, state=state,
        generation=0)


class TestExpandQuery:
    def test_alias_to_canon(self):
        out = expand_query(["mel"], [_alias("melanie", "mel")])
        assert out == ["mel", "melanie"]

    def test_canon_to_alias(self):
        # expansion is bidirectional over active links
        out = expand_query(["melanie"], [_alias("melanie", "mel")])
        assert out == ["melanie", "mel"]

    def test_limit_bounds_expansions(self):
        aliases = [_alias("melanie", f"mel{i}") for i in range(12)]
        out = expand_query(["melanie"], aliases, limit=8)
        assert out == ["melanie"] + sorted(f"mel{i}" for i in range(12))[:8]
        assert len(out) == 9

    def test_candidate_never_expands(self):
        out = expand_query(
            ["mel"], [_alias("melanie", "mel", AliasState.CANDIDATE)])
        assert out == ["mel"]

    def test_rejected_never_expands(self):
        out = expand_query(
            ["mel"], [_alias("melanie", "mel", AliasState.REJECTED)])
        assert out == ["mel"]

    def test_dedup_and_order(self):
        aliases = [_alias("alice chen", "alice"), _alias("bob", "alice")]
        out = expand_query(["alice", "bob"], aliases)
        assert out[0] == "alice"
        assert len(out) == len(set(out))

    def test_empty(self):
        assert expand_query([], [_alias("a b", "a")]) == []
        assert expand_query(["x"], []) == ["x"]


# ---------------------------------------------------------------------------
# idf_weight / entity_weight — V7-08.06 dominant-canon cap
# ---------------------------------------------------------------------------


class TestIDF:
    def test_rare_beats_frequent(self):
        assert idf_weight(1, 100) > idf_weight(50, 100)

    def test_monotone_decreasing(self):
        ws = [idf_weight(df, 100) for df in (1, 5, 20, 50, 90)]
        assert ws == sorted(ws, reverse=True)

    def test_dominant_cap(self):
        # df > 30% of units → ≤ 0.1 × the rare-canon weight
        rare = idf_weight(1, 100)
        w = entity_weight("caroline", {"caroline": 40}, 100)
        assert w <= 0.1 * rare + 1e-12
        assert w > 0.0

    def test_exactly_30pct_not_capped(self):
        n = 100
        w = entity_weight("x", {"x": 30}, n)
        assert w == idf_weight(30, n)

    def test_df_zero_is_max(self):
        assert idf_weight(0, 100) >= idf_weight(1, 100)

    def test_zero_units_safe(self):
        assert entity_weight("x", {"x": 5}, 0) == 0.0
        assert idf_weight(5, 0) == 0.0

    def test_surface_key_normalized(self):
        # df_map is keyed by canon; a surface argument folds to it
        assert entity_weight("Caroline's", {"caroline": 5}, 50) == \
            entity_weight("caroline", {"caroline": 5}, 50)

    def test_speaker_dominance_scenario(self):
        # D7-07: the frequent speaker canon's weight is capped relative
        # to a rare evidence canon (H12 mechanism)
        n = 200
        df = {"caroline": 150, "mural": 2}  # speaker in 75% of units
        assert entity_weight("caroline", df, n) <= \
            0.1 * entity_weight("mural", df, n) + 1e-12

    def test_frequent_below_threshold_uncapped(self):
        n = 100
        df = 20  # 20% — below the 30% dominance bar
        assert entity_weight("x", {"x": df}, n) == idf_weight(df, n)
