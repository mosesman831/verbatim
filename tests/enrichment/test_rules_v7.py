"""V7-20.02/20.05: standing-rule detection, lifecycle, and prefetch."""
import datetime
import re
import sqlite3

import pytest

from verbatim.core.types import json_dumps, safe_json_loads
from verbatim.text.norm_v2 import analyze
from verbatim.storage.schema_v7 import ensure_v7_additive
from verbatim.enrichment.rules_v7 import (
    RULE_PATTERNS_V1,
    apply_revocations,
    detect_revocations,
    detect_rules,
    insert_rules,
    prefetch_block,
    sweep_expired,
)

NOW = int(
    datetime.datetime(2026, 9, 16, 12, 0, tzinfo=datetime.timezone.utc)
    .timestamp() * 1_000_000)


def detect(text, unit="u-1", scope="s1", **kw):
    kw.setdefault("now_us", NOW)
    return detect_rules(
        analyze(text), unit, "speaker:alice",
        raw_text=text, scope_id=scope, generation=1, **kw)


def pin(row):
    return safe_json_loads(row["text_pin"])


def detect_noraw(text, unit="u-1", scope="s1", **kw):
    """detect_rules without raw_text — the term-offsets fallback."""
    kw.setdefault("now_us", NOW)
    return detect_rules(
        analyze(text), unit, "speaker:alice",
        scope_id=scope, generation=1, **kw)


# ---------------------------------------------------------------------------
# pattern registry

class TestPatternRegistry:
    def test_registry_covers_families(self):
        ids = [p[0] for p in RULE_PATTERNS_V1]
        assert len(RULE_PATTERNS_V1) >= 12
        for key in ("always", "never", "only", "dont", "unless",
                    "must", "make_sure", "remember_to", "rule_is",
                    "channel", "path", "until", "time_tail", "prefer"):
            assert any(key in pid for pid in ids), key
        # ordering is stable / ordered tuple
        assert isinstance(RULE_PATTERNS_V1, tuple)

    def test_registry_entries_well_formed(self):
        for pid, rx, fn in RULE_PATTERNS_V1:
            assert isinstance(pid, str) and "/" in pid
            assert isinstance(rx, re.Pattern)
            assert callable(fn)


# ---------------------------------------------------------------------------
# detection per family

@pytest.mark.parametrize("text", [
    "Always post release notes to #Eng-Releases.",
    "Please always run ruff check before pushing.",
])
def test_detect_always(text):
    rows = detect(text)
    assert len(rows) == 1
    assert pin(rows[0])["pattern"] in {"kw/always", "conv/channel"}


@pytest.mark.parametrize("text", [
    "Never deploy on Fridays.",
    "We never merge without review.",
    "On Fridays, never deploy to prod.",
])
def test_detect_never(text):
    rows = detect(text)
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "kw/never"


def test_detect_only_until():
    rows = detect("Only post updates until Friday.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "cond/only_lead"
    assert rows[0]["valid_until_expr"] == "until Friday"


def test_detect_dont_unless():
    rows = detect("Don't commit secrets unless the vault is rotated.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "cond/dont_unless"
    # unless-clauses are validity conditions, not expiry bounds:
    # verbatim in signals.condition, never in valid_until_expr.
    assert rows[0]["valid_until_expr"] is None
    assert p["signals"]["condition"] == \
        "unless the vault is rotated"


def test_detect_must():
    rows = detect("All tests must pass before merge.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "modal/must"
    assert rows[0]["valid_until_expr"] == "before merge"


def test_detect_make_sure():
    rows = detect("Make sure you sign the tag.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "imp/make_sure"


def test_detect_remember_to():
    rows = detect("Remember to run the linter.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "imp/remember_to"


def test_detect_the_rule_is():
    rows = detect("The rule is to never push to main.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "decl/rule_is"


def test_detect_channel_convention():
    rows = detect("Post release notes to #eng-releases.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "conv/channel"
    assert "eng releases" in safe_json_loads(
        rows[0]["trigger_entities"])


def test_detect_path_convention():
    rows = detect("Put tests in tests/ and fixtures in tests/fixtures/.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "conv/path_put"
    assert "tests" in safe_json_loads(
        rows[0]["trigger_entities"])


def test_detect_temporal_bound_leading():
    rows = detect("Until the migration finishes, keep writes off.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "cond/until_lead"
    assert rows[0]["valid_until_expr"] == \
        "Until the migration finishes"


def test_detect_temporal_bound_trailing():
    rows = detect("Deploy to prod before Friday.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "cond/time_tail"
    assert rows[0]["valid_until_expr"] == "before Friday"


def test_detect_temporal_bound_trailing_rejects_narrative():
    assert detect("He left before Friday.") == []
    assert detect("I deployed the build before Friday.") == []
    assert detect("The deadline is before Friday.") == []


def test_detect_preference_as_rule():
    rows = detect("I prefer pytest for unit tests.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "pref/prefer"
    assert "speaker alice" in safe_json_loads(
        rows[0]["trigger_entities"])


def test_detect_we_use():
    rows = detect("We use black for formatting.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "pref/we_use"


def test_detect_negations():
    assert len(detect("No commits on Fridays.")) == 1
    assert len(detect("We are not supposed to push on Fridays.")) == 1
    assert len(detect("Never ever force push.")) == 1


def test_detect_required_and_supposed():
    assert len(detect("Reports are required before close.")) == 1
    assert len(detect("It is mandatory to wear a badge.")) == 1
    assert len(detect("You are supposed to fill out the form.")) == 1


def test_detect_whenever_imperative():
    rows = detect("Whenever you deploy, restart the cache.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "imp/whenever"


def test_detect_ship_unless():
    rows = detect("Ship it unless the build is red.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["pattern"] == "cond/unless_head"
    assert rows[0]["valid_until_expr"] is None
    assert p["signals"]["condition"] == "unless the build is red"


def test_detect_ask_team():
    rows = detect("Always ask the legal team before signing.")
    assert len(rows) == 1
    # not speaker-scoped (speaker flag only on preference family)
    assert safe_json_loads(rows[0]["trigger_entities"]) == []
    assert "legal" in safe_json_loads(rows[0]["trigger_topics"])


# ---------------------------------------------------------------------------
# hedged / questioned / hypothetical — must not detect

@pytest.mark.parametrize("text", [
    "Should we always deploy on Fridays?",
    "Do we always deploy on fridays?",
    "Maybe never touch that file.",
    "we should probably always write tests",
    "i might never use tabs",
    "could we always run the linter first?",
    "let's maybe always deploy early",
    "The docs recommend always running tests.",
    "He mentioned never touching the config.",
    "She said to always check the logs.",
    "rules are meant to be broken.",
    "The policy is unclear.",
    "the rule is probably fine",
    "that must be nice.",
    "it must have been late",
    "he never went to paris",
    "we never saw the error again",
    "the rule is simple",
    "the rules are made to be broken",
    "the convention is outdated",
    "i always loved that restaurant",
    "we always had lunch together",
    "it was never a problem before",
    "she never said anything about it",
])
def test_no_detection_hedged_narrative(text):
    assert detect(text) == [], text


def test_hedge_only_rejects_same_sentence():
    text = "Maybe skip it. Always deploy on Fridays."
    rows = detect(text)
    assert len(rows) == 1
    p = pin(rows[0])
    assert "always" in p["text"].lower()


# ---------------------------------------------------------------------------
# byte-exact pinned spans

def test_pin_byte_exact_raw():
    text = "Ok. Always post notes to #eng-releases. Thanks."
    rows = detect(text)
    p = pin(rows[0])
    raw = text.encode()
    span = raw[p["byte_start"]:p["byte_end"]].decode()
    assert p["text"] == span.strip()
    assert "post notes" in p["text"]


def test_pin_excludes_keyword():
    rows = detect("Always post release notes to #eng-releases.")
    p = pin(rows[0])
    assert not p["text"].lower().startswith("always")
    assert p["text"].startswith("post")


def test_pin_preserves_raw_surface_case():
    # byte-exact slice: the channel surface keeps its source casing
    rows = detect("Always post to #Eng-Releases.")
    p = pin(rows[0])
    assert "#Eng-Releases" in p["text"]


def test_pin_covers_path_surface():
    rows = detect("Put release artifacts in ./dist/ before merge.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert "./dist" in p["text"]
    assert "dist" in safe_json_loads(rows[0]["trigger_entities"])


def test_unit_ref_in_prefetch():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    rows = detect("Always post release notes to #eng-releases.")
    insert_rules(conn, rows)
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           task_text="post to eng releases", now_us=NOW)
    m = re.search(r"\[u-1:(\d+)-(\d+)\]", block)
    assert m
    p = pin(rows[0])
    assert int(m.group(1)) == p["byte_start"]
    assert int(m.group(2)) == p["byte_end"]


def test_noraw_fallback_term_offsets():
    rows = detect_noraw("Always post release notes to #Eng-Releases.")
    assert len(rows) == 1
    p = pin(rows[0])
    assert p["signals"]["pinned"] == "term_offsets"
    # byte span still on norm-aligned raw offsets
    assert p["byte_start"] <= p["byte_end"]
    assert "post" in p["text"]


def test_noraw_and_raw_paths_agree_on_row_keys():
    a = detect("Never deploy on Fridays.")
    b = detect_noraw("Never deploy on Fridays.")
    assert set(a[0]) == set(b[0])
    for k in ("rule_id", "scope_id", "unit_id", "text_pin",
              "trigger_entities", "trigger_topics",
              "valid_until_expr", "status", "generation"):
        assert k in a[0]


# ---------------------------------------------------------------------------
# entity / topic extraction

def test_entities_include_speaker_on_preferences():
    rows = detect("I prefer pytest for unit tests.")
    ents = safe_json_loads(rows[0]["trigger_entities"])
    assert "speaker alice" in ents


def test_channel_entity_canonical():
    rows = detect("Always post to #Eng-Releases.")
    ents = safe_json_loads(rows[0]["trigger_entities"])
    assert "eng releases" in ents


def test_topics_exclude_stopwords():
    rows = detect("Never deploy on Fridays.")
    tops = safe_json_loads(rows[0]["trigger_topics"])
    assert "deploy" in tops and "fridays" in tops
    assert "on" not in tops


def test_topics_capped():
    tops = safe_json_loads(
        detect("Always track alpha beta gamma delta epsilon zeta "
               "eta theta iota kappa lambda mu nu xi omicron pi "
               "rho sigma tau upsilon phi chi psi omega")[0]
        ["trigger_topics"])
    assert len(tops) <= 24


def test_no_phantom_joined_entity():
    rows = detect("Always post to #ops until next Friday.")
    ents = safe_json_loads(rows[0]["trigger_entities"])
    assert "ops" in ents
    assert "ops until" not in ents


def test_hyphen_identifier_joins_entity():
    rows = detect("Always post to #eng-releases.")
    ents = safe_json_loads(rows[0]["trigger_entities"])
    assert "eng releases" in ents


# ---------------------------------------------------------------------------
# deterministic ids

def test_rule_id_deterministic():
    a = detect("Never deploy on Fridays.")
    b = detect("Never deploy on Fridays.")
    assert a[0]["rule_id"] == b[0]["rule_id"]
    assert a[0]["rule_id"].startswith("rule7:")


def test_rule_id_changes_with_unit():
    a = detect("Never deploy on Fridays.", unit="u-1")
    b = detect("Never deploy on Fridays.", unit="u-2")
    assert a[0]["rule_id"] != b[0]["rule_id"]


def test_rule_id_changes_with_pin_position():
    a = detect("Never deploy on Fridays.", unit="u-1")
    b = detect("Ok. Never deploy on Fridays.", unit="u-1")
    assert a[0]["rule_id"] != b[0]["rule_id"]


def test_rule_id_scope_independent():
    # the id binds (unit, pattern, pin) — scope_id lives on the row
    a = detect("Never deploy on Fridays.", scope="s1")
    b = detect("Never deploy on Fridays.", scope="s2")
    assert a[0]["rule_id"] == b[0]["rule_id"]
    assert a[0]["scope_id"] == "s1" and b[0]["scope_id"] == "s2"


def test_unique_rule_ids_per_sentence():
    text = ("Never deploy on Fridays. Always run the linter. "
            "The rule is to never push to main.")
    rows = detect(text)
    ids = [r["rule_id"] for r in rows]
    assert len(rows) == 3 and len(set(ids)) == 3


def test_restatement_gets_own_pin():
    # same words, different byte span → different rule id (each
    # mention keeps its own pin)
    text = "Never deploy on Fridays. Never deploy on Fridays."
    rows = detect(text)
    assert len(rows) == 2
    assert rows[0]["rule_id"] != rows[1]["rule_id"]
    assert pin(rows[0])["byte_start"] != pin(rows[1])["byte_start"]


# ---------------------------------------------------------------------------
# prefetch_block

def _mkconn(rows, **kw):
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    insert_rules(conn, rows)
    return conn


def test_prefetch_renders_block():
    conn = _mkconn(detect(
        "Always post release notes to #Eng-Releases "
        "until deploy is green."))
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           task_text="post the update to eng releases",
                           now_us=NOW)
    assert block.startswith("## STANDING RULES\n- [u-1:")
    assert "post release notes" in block


def _synthetic_global_rule(rid="rule7:global0001", unit="u-g"):
    """A global rule = no trigger_entities/topics — always deliverable.
    Detection always yields some trigger, so the global lane is
    exercised with a hand-built row (schema permits it)."""
    return {
        "rule_id": rid,
        "scope_id": "s1",
        "unit_id": unit,
        "text_pin": json_dumps({
            "byte_start": 0, "byte_end": 14, "unit_id": unit,
            "text": "be civil in review comments",
            "created_us": NOW, "until_end_us": None, "signals": {},
        }),
        "trigger_entities": "[]",
        "trigger_topics": "[]",
        "valid_until_expr": None,
        "status": "active",
        "generation": 1,
    }


def test_prefetch_triggered_above_global():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    trig = detect("Never deploy on Fridays.", unit="u-t")
    insert_rules(conn, [_synthetic_global_rule()] + trig)
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           task_text="deploy the service on friday",
                           now_us=NOW)
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(lines) == 2
    assert "deploy" in lines[0]          # triggered first
    assert "civil" in lines[1]           # global second


def test_prefetch_global_alone_renders():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    insert_rules(conn, [_synthetic_global_rule()])
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           task_text="anything at all",
                           now_us=NOW)
    assert "be civil" in block


def test_prefetch_empty_when_no_rules():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    assert prefetch_block(conn, scope_id="s1", generation=1,
                          task_text="deploy", now_us=NOW) == ""


def test_prefetch_empty_when_no_overlap():
    conn = _mkconn(detect("Never deploy on Fridays."))
    assert prefetch_block(conn, scope_id="s1", generation=1,
                          task_text="water the plants",
                          now_us=NOW) == ""


def test_prefetch_scoped_out():
    conn = _mkconn(detect("Never deploy on Fridays.", scope="s1"))
    assert prefetch_block(conn, scope_id="other", generation=1,
                          task_text="deploy on friday",
                          now_us=NOW) == ""


def test_prefetch_respects_limit():
    rows = []
    for i in range(5):
        rows += detect("Always ask the team about item %d." % i,
                       unit="u-%d" % i)
    conn = _mkconn(rows)
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           task_text="ask team item", now_us=NOW,
                           limit=2)
    assert block.count("\n- ") == 1 or len(
        [ln for ln in block.splitlines() if ln.startswith("- ")]) == 2


def test_prefetch_generation_gate():
    conn = _mkconn(detect("Never deploy on Fridays."))
    assert prefetch_block(conn, scope_id="s1", generation=0,
                          task_text="deploy friday",
                          now_us=NOW) == ""
    assert prefetch_block(conn, scope_id="s1", generation=2,
                          task_text="deploy friday",
                          now_us=NOW) != ""


def test_prefetch_explicit_entities_topics():
    conn = _mkconn(detect("Always post to #eng-releases."))
    block = prefetch_block(conn, scope_id="s1", generation=1,
                           entities=("eng releases",),
                           now_us=NOW)
    assert "eng-releases" in block
    block2 = prefetch_block(conn, scope_id="s1", generation=1,
                            topics=("deploy",),
                            now_us=NOW)
    assert block2 == "" or "post to" in block2


def test_prefetch_revoked_excluded():
    conn = _mkconn(detect("Always post to #eng-releases.", unit="u-9"))
    revs = detect_revocations(
        analyze("Actually, no longer post to #eng-releases."),
        "u-10",
        raw_text="Actually, no longer post to #eng-releases.")
    n = apply_revocations(conn, scope_id="s1", revocations=revs)
    assert n == [r["rule_id"] for r in
                 detect("Always post to #eng-releases.", unit="u-9")]
    assert prefetch_block(conn, scope_id="s1", generation=1,
                          task_text="post to eng releases",
                          now_us=NOW) == ""


def test_prefetch_expired_excluded():
    conn = _mkconn(detect(
        "Always post to #ops until next Friday.", unit="u-1"))
    later = NOW + 20 * 86400 * 1_000_000
    assert prefetch_block(conn, scope_id="s1", generation=1,
                          task_text="post to ops",
                          now_us=later) == ""


# ---------------------------------------------------------------------------
# lifecycle: revocation / expiry

def test_revocation_markers_detected():
    revs = detect_revocations(
        analyze("Actually, no longer post to #eng-releases."),
        "u-10",
        raw_text="Actually, no longer post to #eng-releases.")
    assert len(revs) == 1
    assert revs[0]["marker"] == "rev/no_longer"
    assert "post" in revs[0]["target_text"]


def test_revocation_never_mind():
    revs = detect_revocations(
        analyze("Never mind about the deploys."),
        "u-1",
        raw_text="Never mind about the deploys.")
    assert revs and revs[0]["marker"] == "rev/never_mind"


def test_revocation_requires_overlap():
    conn = _mkconn(detect("Always post to #eng-releases.", unit="u-9"))
    revs = detect_revocations(
        analyze("No longer water the plants daily."),
        "u-10",
        raw_text="No longer water the plants daily.")
    n = apply_revocations(conn, scope_id="s1", revocations=revs)
    assert n == []


def test_revocation_scoped():
    conn = _mkconn(detect("Always post to #eng-releases.", unit="u-9"))
    revs = detect_revocations(
        analyze("No longer post to #eng-releases."),
        "u-10",
        raw_text="No longer post to #eng-releases.")
    assert apply_revocations(
        conn, scope_id="other", revocations=revs) == []


def test_sweep_expired_marks_rows():
    conn = _mkconn(detect("Always post to #ops until next Friday."))
    assert sweep_expired(conn, now_us=NOW + 20 * 86400 * 1_000_000) == 1
    cur = conn.execute(
        "SELECT status FROM standing_rules").fetchone()
    assert cur[0] == "expired"


def test_unresolved_temporal_kept_active():
    rows = detect("Post to #ops until deploy is green.")
    assert rows[0]["status"] == "active"
    assert rows[0]["valid_until_expr"] == "until deploy is green"
    assert safe_json_loads(rows[0]["text_pin"])["signals"][
        "until_unresolved"] is True


def test_resolved_temporal_born_active():
    rows = detect("Post to #ops until next Friday.")
    assert rows[0]["status"] == "active"
    assert safe_json_loads(rows[0]["text_pin"])["until_end_us"] > NOW


def test_resolved_temporal_born_expired():
    # T17 weekday semantics: bare "Friday" = most recent past Friday
    rows = detect("Post to #ops until Friday.")
    assert rows[0]["status"] == "expired"


def test_valid_until_expr_verbatim_not_date():
    rows = detect("Post to #ops until next Friday.")
    assert rows[0]["valid_until_expr"] == "until next Friday"


# ---------------------------------------------------------------------------
# 20-sentence benign corpus — no false positives

BENIGN = [
    "I went to the store yesterday.",
    "The meeting was moved to Thursday.",
    "My manager Sarah approved the budget request.",
    "We hired a new engineer last month.",
    "The concert starts at 8pm on Saturday.",
    "I finished reading that book you recommended.",
    "The train was delayed by an hour.",
    "We had lunch at the new place downtown.",
    "She mentioned the deployment went fine.",
    "The api returns a list of users.",
    "It took us three days to drive there.",
    "My phone number changed last year.",
    "The recipe calls for two cups of flour.",
    "He said the movie was great.",
    "Our flight leaves early tomorrow morning.",
    "The kids started school in September.",
    "I saw a great talk about databases.",
    "They moved the office to Berlin.",
    "The report is due next week.",
    "We celebrated her birthday with cake.",
]


def test_benign_corpus_no_false_positives():
    for c in BENIGN:
        assert detect(c) == [], c


# ---------------------------------------------------------------------------
# enc/dec + insert helpers

def test_json_enc_dec_roundtrip():
    assert safe_json_loads(json_dumps(["a", "b"])) == ["a", "b"]
    assert safe_json_loads(json_dumps([])) == []


def test_insert_rows_carries_pin_until_end():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    rows = detect("Post to #ops until next Friday.")
    insert_rules(conn, rows)
    cur = conn.execute(
        "SELECT text_pin, valid_until_expr FROM standing_rules"
    ).fetchone()
    assert safe_json_loads(cur[0])["until_end_us"] is not None
    assert cur[1] == "until next Friday"
