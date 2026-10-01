"""Harvester behavior tests (SPEC §11): boundaries, hints, fences, overflow."""

from __future__ import annotations

import pytest

from verbatim.core.harvest import HarvestResult, harvest, harvest_source
from verbatim.core.types import Provenance, SourceKind

from tests.core.conftest import make_envelope, make_scope


def texts(payload: bytes, result: HarvestResult) -> list[str]:
    """Decode every candidate back from the payload — proves exact byte
    offsets on UTF-8 boundaries."""
    return [payload[c.start_byte : c.end_byte].decode("utf-8") for c in result.candidates]


def test_abbreviations_do_not_split():
    payload = b"Dr. Smith uses Neovim daily and writes lots of Rust. He lives in Oslo and works from home."
    result = harvest(payload)
    got = texts(payload, result)
    assert got == [
        "Dr. Smith uses Neovim daily and writes lots of Rust.",
        "He lives in Oslo and works from home.",
    ]


def test_abbreviation_mid_sentence():
    payload = b"My editor is etc. not really, but I use vim every day for real work."
    result = harvest(payload)
    assert len(result.candidates) == 1


def test_decimal_point_is_not_a_boundary():
    payload = b"The pi constant is 3.14 in short form. It shows up everywhere in mathematics."
    result = harvest(payload)
    got = texts(payload, result)
    assert got == [
        "The pi constant is 3.14 in short form.",
        "It shows up everywhere in mathematics.",
    ]


def test_url_internal_dots_do_not_split():
    payload = b"See example.com for details about the tool. I use it for all of my work projects."
    result = harvest(payload)
    got = texts(payload, result)
    assert got == [
        "See example.com for details about the tool.",
        "I use it for all of my work projects.",
    ]


def test_ellipsis_is_not_a_boundary():
    payload = b"Well... that is a long story indeed. Ask me later about it."
    result = harvest(payload)
    assert texts(payload, result) == ["Well... that is a long story indeed. Ask me later about it."]


def test_initials_do_not_split():
    payload = b"J. R. R. Tolkien wrote many books over the years."
    result = harvest(payload)
    assert texts(payload, result) == ["J. R. R. Tolkien wrote many books over the years."]


def test_question_exclamation_runs():
    payload = b"What?! That actually changed everything we knew. Now we wait patiently for the next step."
    result = harvest(payload)
    got = texts(payload, result)
    assert got == [
        "What?! That actually changed everything we knew.",
        "Now we wait patiently for the next step.",
    ]


def test_cjk_sentence_punctuation():
    text = "私は毎日Neovimを使っています。仕事ではVS Codeを使います。"
    payload = text.encode("utf-8")
    result = harvest(payload, min_len=8)
    got = texts(payload, result)
    assert got == ["私は毎日Neovimを使っています。", "仕事ではVS Codeを使います。"]
    # byte offsets must be real UTF-8 boundaries (the decode above would fail otherwise)


def test_fullwidth_exclamation_question():
    text = "本当ですか！次の文はここから始まります。最後です"
    payload = text.encode("utf-8")
    result = harvest(payload, min_len=8)
    got = texts(payload, result)
    assert len(got) == 3
    assert got[0].endswith("！")


def test_negation_flag():
    payload = b"I no longer use Docker for this project."
    (c,) = harvest(payload).candidates
    assert c.negated is True


def test_no_negation_flag():
    payload = b"I use Docker for this project every single day."
    (c,) = harvest(payload).candidates
    assert c.negated is False


def test_condition_flag():
    payload = b"At work I use VS Code but at home I use Neovim."
    (c,) = harvest(payload).candidates
    assert c.has_condition is True


@pytest.mark.parametrize(
    "text,hint",
    [
        ("I might switch to Helix next month.", "hypothetical"),
        ("I would use Emacs if it were faster.", "hypothetical"),
        ("I plan to move my config to Nix soon.", "hypothetical"),
        ("I used to run Arch Linux on my laptop.", "habitual"),
        ("I think my editor is Emacs these days.", "uncertain"),
        ("I probably use vim far too often honestly.", "uncertain"),
    ],
)
def test_modality_hints(text, hint):
    (c,) = harvest(text.encode()).candidates
    assert c.modality_hint == hint


def test_no_modality_hint():
    (c,) = harvest(b"I use Neovim for all of my editing.").candidates
    assert c.modality_hint is None


def test_sensitive_hint():
    (c,) = harvest(b"my api key is abc123def456ghi789jklmno").candidates
    assert c.sensitive_hint is True


def test_fenced_code_block_excluded():
    text = (
        "I use Neovim for everything I write.\n"
        "```\n"
        "password = 'hunter2'  # not a personal fact\n"
        "```\n"
        "I live in Oslo and work remotely."
    )
    payload = text.encode()
    result = harvest(payload)
    got = texts(payload, result)
    assert len(got) == 2
    assert all("password" not in g and "hunter2" not in g for g in got)
    assert result.skipped >= 1


def test_unclosed_fence_runs_to_eof():
    payload = b"I use Neovim daily.\n```\nsecret token here\nmore code"
    result = harvest(payload)
    assert texts(payload, result) == ["I use Neovim daily."]


def test_list_items_are_separate_candidates():
    text = "- I use Neovim for personal projects\n- At work I use VS Code daily\n\nAlso I like git a lot."
    result = harvest(text.encode())
    kinds = [c.kind for c in result.candidates]
    assert kinds == ["list_item", "list_item", "paragraph"]
    got = texts(text.encode(), result)
    assert got[0].startswith("- I use Neovim")


def test_numbered_list_items():
    text = "1. First fact worth remembering today\n2. Second fact worth remembering today"
    result = harvest(text.encode())
    assert [c.kind for c in result.candidates] == ["list_item", "list_item"]


@pytest.mark.parametrize("reply", ["Yes.", "the latter", "It.", "I did.", "Nope"])
def test_short_replies_need_context(reply):
    result = harvest(reply.encode())
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.context_needed is True
    assert c.reason == "short_reply"


@pytest.mark.parametrize(
    "text",
    [
        "My name is Bob",
        "my wife is Sarah",
        "moved to Berlin",
        "prefer dark mode",
        "use pnpm not npm",
        "my dog is called Max",
        "born in 1990",
        "allergic to peanuts",
        "call me Rob",
        "timezone is PST",
    ],
)
def test_short_unterminated_chat_facts_are_kept(text):
    result = harvest(text.encode())
    assert texts(text.encode(), result) == [text]


@pytest.mark.parametrize(
    "text",
    [
        "ok hi",
        "okay",
        "thanks",
        "lol",
        "sounds good",
        "got it",
        "nice",
    ],
)
def test_short_chat_noise_is_dropped(text):
    result = harvest(text.encode())
    assert result.candidates == ()
    assert result.skipped == 1


def test_unterminated_tail_is_statement():
    payload = b"I switched to Helix last week. It still needs configuration work"
    result = harvest(payload)
    kinds = [c.kind for c in result.candidates]
    assert kinds == ["sentence", "statement"]


def test_single_sentence_paragraph_kind():
    payload = b"I use Neovim for all of my editing."
    (c,) = harvest(payload).candidates
    assert c.kind == "paragraph"


def test_overflow_beyond_max_candidates():
    text = " ".join(f"Sentence number {i} is recorded here today." for i in range(40))
    result = harvest(text.encode())
    assert len(result.candidates) == 32
    assert result.overflow_count == 8


def test_malformed_utf8_rejected():
    with pytest.raises(UnicodeDecodeError):
        harvest(b"\xff\xfe invalid bytes \x80")


def test_candidate_offsets_reproduce_quotation_exactly():
    text = "I use Neovim. 仕事ではEmacsです。 At work it is VS Code."
    payload = text.encode("utf-8")
    result = harvest(payload, min_len=8)
    for c in result.candidates:
        assert payload[c.start_byte : c.end_byte].decode("utf-8") in text
        # boundary check: decoding must not split a multi-byte char
        payload[: c.start_byte].decode("utf-8")
        payload[c.end_byte :].decode("utf-8")


# ---------------------------------------------------------------------------
# harvest_source provenance/kind policy
# ---------------------------------------------------------------------------


def test_harvest_source_allows_user_message():
    scope = make_scope()
    env = make_envelope(scope, "I use Neovim for all of my editing.")
    result = harvest_source(env)
    assert len(result.candidates) == 1


def test_harvest_source_blocks_tool_output():
    scope = make_scope()
    env = make_envelope(
        scope,
        "I use Neovim for all of my editing.",
        kind=SourceKind.TOOL_OUTPUT,
        provenance=Provenance.APPROVED_TOOL,
    )
    assert harvest_source(env).candidates == ()


def test_harvest_source_tool_output_opt_in():
    scope = make_scope()
    env = make_envelope(
        scope,
        "I use Neovim for all of my editing.",
        kind=SourceKind.TOOL_OUTPUT,
        provenance=Provenance.APPROVED_TOOL,
    )
    result = harvest_source(env, allow_kinds={SourceKind.TOOL_OUTPUT})
    assert len(result.candidates) == 1


def test_harvest_source_blocks_assistant_provenance():
    scope = make_scope()
    env = make_envelope(
        scope,
        "I use Neovim for all of my editing.",
        kind=SourceKind.ASSISTANT_MESSAGE,
        provenance=Provenance.ASSISTANT_GENERATED,
    )
    assert harvest_source(env).candidates == ()


def test_harvest_source_assistant_opt_in():
    scope = make_scope()
    env = make_envelope(
        scope,
        "I use Neovim for all of my editing.",
        kind=SourceKind.ASSISTANT_MESSAGE,
        provenance=Provenance.ASSISTANT_GENERATED,
    )
    result = harvest_source(env, allow_kinds={SourceKind.ASSISTANT_MESSAGE})
    assert len(result.candidates) == 1
