"""extract_identifiers / extract_entities — deterministic mention
extraction with UTF-8 byte offsets (V5-30.14/30.17)."""

from __future__ import annotations

from verbatim.enrichment import (
    extract_entities,
    extract_identifiers,
)
from verbatim.enrichment.identifiers import KNOWN_ENTITIES


def _byte_slice(text, ident):
    return text.encode("utf-8")[ident.start:ident.end].decode("utf-8")


def _kinds(idents):
    return {i.kind for i in idents}


class TestIdentifiers:
    def test_url(self):
        ids = extract_identifiers("see https://example.io/x?y=1 for docs")
        (url,) = [i for i in ids if i.kind == "url"]
        assert url.value == "https://example.io/x?y=1"

    def test_url_trailing_punct_stripped(self):
        ids = extract_identifiers("open https://a.io/x, then go.")
        (url,) = [i for i in ids if i.kind == "url"]
        assert url.value == "https://a.io/x"

    def test_email(self):
        ids = extract_identifiers("mail alice.bob+x@corp.example.com now")
        (em,) = [i for i in ids if i.kind == "email"]
        assert em.value == "alice.bob+x@corp.example.com"

    def test_email_not_handle(self):
        # the @ inside an email must not also yield a handle
        ids = extract_identifiers("reach a@b.co")
        assert "handle" not in _kinds(ids)
        assert "email" in _kinds(ids)

    def test_handle(self):
        ids = extract_identifiers("ping @alice_2 about it")
        (h,) = [i for i in ids if i.kind == "handle"]
        assert h.value == "@alice_2"

    def test_ticket_key(self):
        ids = extract_identifiers("fixed under JIRA-1234 today")
        (t,) = [i for i in ids if i.kind == "ticket"]
        assert t.value == "JIRA-1234"

    def test_version_forms(self):
        ids = extract_identifiers("ship v1.2.3 and deploy-v2 together")
        versions = {i.value for i in ids if i.kind == "version"}
        assert "v1.2.3" in versions
        # the whole name-bearing token is the identifier (V5-30.17)
        assert "deploy-v2" in versions

    def test_version_distinguishes_v1_v2(self):
        # V5-30.07 — the adversarial pair must produce different
        # identifier sets so dedup can never collapse them.
        v1 = extract_identifiers("deploy-v1 went live")
        v2 = extract_identifiers("deploy-v2 went live")
        assert {i.value for i in v1} != {i.value for i in v2}

    def test_semver_bare(self):
        ids = extract_identifiers("python 3.11 required")
        assert any(i.kind == "version" and i.value == "3.11"
                   for i in ids)

    def test_hash_requires_hex_signal(self):
        ids = extract_identifiers("commit deadbee1 landed")
        assert any(i.kind == "hash" and i.value == "deadbee1"
                   for i in ids)
        # 0x prefix
        ids = extract_identifiers("hash 0xabc123 ok")
        assert any(i.kind == "hash" and i.value == "0xabc123"
                   for i in ids)

    def test_hash_rejects_words_and_numbers(self):
        # all-hex-letter word and all-digit number are not hashes
        ids = extract_identifiers("the word acceded and code 1234567")
        assert "hash" not in _kinds(ids)

    def test_quoted_strings(self):
        ids = extract_identifiers('she wrote "ship it friday" down')
        (q,) = [i for i in ids if i.kind == "quoted"]
        assert q.value == '"ship it friday"'

    def test_apostrophe_not_a_quote(self):
        # "don't" must not pair with a later quote to form a span
        ids = extract_identifiers("don't run it, said 'no one'")
        quoted = [i for i in ids if i.kind == "quoted"]
        assert [q.value for q in quoted] == ["'no one'"]

    def test_quoted_yields_inner_specific(self):
        # specific kinds beat the quoted wrapper — the ticket inside a
        # quoted string is still a ticket identifier
        ids = extract_identifiers('the ticket "ABC-123" closed')
        assert any(i.kind == "ticket" and i.value == "ABC-123"
                   for i in ids)
        assert "quoted" not in _kinds(ids)

    def test_code_tokens(self):
        ids = extract_identifiers(
            "call foo_bar.py, run useEffect, bump utf8, upgrade node-18"
        )
        vals = {i.value for i in ids}
        assert {"foo_bar.py", "useEffect", "utf8", "node-18"} <= vals

    def test_code_rejects_plain_words(self):
        ids = extract_identifiers("a well-known fact about code")
        assert "code" not in _kinds(ids)

    def test_paths(self):
        ids = extract_identifiers(
            "edit /etc/hosts, src/main.py, ./rel/x, ~/notes/y, a/b/c"
        )
        vals = {i.value for i in ids if i.kind == "path"}
        assert {"/etc/hosts", "src/main.py", "./rel/x", "~/notes/y",
                "a/b/c"} == vals

    def test_path_rejections(self):
        ids = extract_identifiers("choose and/or, read /etc alone")
        assert "path" not in _kinds(ids)

    def test_windows_path(self):
        ids = extract_identifiers(r"open C:\Users\al\file.txt please")
        assert any(i.kind == "path" and i.value.startswith("C:\\")
                   for i in ids)

    def test_case_preserved(self):
        ids = extract_identifiers("Fix PROJ-99 and Email Bob@Corp.COM")
        vals = {i.value for i in ids}
        assert "PROJ-99" in vals and "Bob@Corp.COM" in vals

    def test_byte_offsets_ascii(self):
        text = "go to src/main.py now"
        ids = extract_identifiers(text)
        (p,) = [i for i in ids if i.kind == "path"]
        assert text[p.start:p.end] == p.value == "src/main.py"

    def test_byte_offsets_multibyte(self):
        text = "café — fix ABC-123 ok"   # é is 2 bytes
        ids = extract_identifiers(text)
        for i in ids:
            assert _byte_slice(text, i) == i.value
        (t,) = [i for i in ids if i.kind == "ticket"]
        byte_start = text.encode("utf-8").find(b"ABC-123")
        assert t.start == byte_start
        # é (2B) + space etc — byte offset > char offset
        assert t.start > text.index("ABC-123")

    def test_no_overlapping_spans(self):
        text = ("mail a@b.co, open https://x.io, ship deploy-v2, "
                "see \"ABC-9\" and read src/x.py")
        ids = extract_identifiers(text)
        for a in ids:
            for b in ids:
                if a is b:
                    continue
                assert a.end <= b.start or b.end <= a.start

    def test_determinism(self):
        text = "Deploy deploy-v2, ping @bob, fix ABC-123 at src/main.py"
        assert extract_identifiers(text) == extract_identifiers(text)

    def test_empty(self):
        assert extract_identifiers("") == []
        assert extract_identifiers(None) == []
        assert extract_identifiers("just some plain words here") == []


class TestEntities:
    def test_capitalized_multiword(self):
        ents = extract_entities("met Alice Johnson in Berlin")
        vals = {e.value for e in ents}
        assert "Alice Johnson" in vals

    def test_connector_names(self):
        ents = extract_entities("spoke with Ruth van der Berg")
        assert any(e.value == "Ruth van der Berg" for e in ents)

    def test_sentence_boundary_breaks_run(self):
        ents = extract_entities("met Alice Berg. Then Bob left.")
        vals = {e.value for e in ents}
        assert "Alice Berg" in vals and "Bob" in vals
        assert not any("Berg." in v or "Berg Then" in v for v in vals)

    def test_sentence_initial_stopwords(self):
        ents = extract_entities("The cat sat. However Bob left.")
        vals = {e.value for e in ents}
        assert "The" not in vals and "However" not in vals
        assert "Bob" in vals

    def test_pronoun_I_never_entity(self):
        ents = extract_entities("I saw Alice")
        assert {e.value for e in ents} == {"Alice"}

    def test_dictionary_hits_case_insensitive(self):
        ents = extract_entities("we moved to redis and postgres")
        vals = {e.value for e in ents if e.kind == "dictionary"}
        assert {"redis", "postgres"} == vals

    def test_dictionary_multiword(self):
        ents = extract_entities("hosted on google cloud now")
        assert any(e.value == "google cloud" for e in ents)

    def test_dictionary_and_span_overlap(self):
        ents = extract_entities("the New York office moved")
        # dictionary hit wins the tie over the same capitalized span
        ny = [e for e in ents if "New York" in e.value]
        assert ny and ny[0].kind == "dictionary"

    def test_list_not_merged(self):
        ents = extract_entities("Alice and Bob paired")
        vals = {e.value for e in ents}
        assert "Alice" in vals and "Bob" in vals
        assert "Alice and Bob" not in vals

    def test_byte_offsets(self):
        text = "Renée met Bob"   # é is 2 bytes
        ents = extract_entities(text)
        for e in ents:
            assert text.encode("utf-8")[e.start:e.end].decode() == e.value
        bob = [e for e in ents if e.value == "Bob"][0]
        assert bob.start == text.encode("utf-8").find(b"Bob")

    def test_dictionary_contents_pinned(self):
        # a sampling of the published dictionary — entries are matching
        # candidates only, never identity merges (V5-30.15)
        assert {"python", "redis", "kubernetes", "new york"} <= \
            KNOWN_ENTITIES

    def test_determinism_and_empty(self):
        text = "Alice met Bob at the New York office"
        assert extract_entities(text) == extract_entities(text)
        assert extract_entities("") == []
        assert extract_entities("all lowercase words here") == []
