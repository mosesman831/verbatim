"""Tests for verbatim/extraction/t2.py — SPEC_V7 V7-13.12–18, §30
``t2_facts``.

The model is always a mock ``model_fn`` — the contract under test is the
deterministic machinery around it: honest ``unavailable`` when no model
exists, strict JSON parsing, byte-exact quote verification against the
pinned unit slices, idempotent commit with extraction provenance, and
retirement when a pinned unit leaves the eligible set.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.extraction import t2  # noqa: E402
from verbatim.storage.schema import DDL_V1  # noqa: E402
from verbatim.storage.schema_v7 import ensure_v7_additive  # noqa: E402

T0 = 1_686_787_200_000_000  # 2023-06-15T00:00:00Z, µs
GEN = 1
SCOPE = "sc"


# ---------------------------------------------------------------------------
# fixtures: real DDL (v1 sources/revisions + additive v7 units/t2_facts)
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    c.executescript(DDL_V1)
    ensure_v7_additive(c)
    c.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES (?, 'p', 'owner')",
        (SCOPE,),
    )
    yield c
    c.close()


def seed_unit(conn, unit_id, text, *, scope=SCOPE, generation=GEN,
              source_id=None, revision=1, recorded_at_us=T0, pinned=True):
    """One source + revision + units row pinning the whole payload."""
    payload = text.encode("utf-8") if isinstance(text, str) else text
    sid = source_id or f"src-{unit_id}"
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, origin, external_id, source_kind,"
        " scope_id, speaker_id, created_us)"
        " VALUES (?, 'test', NULL, 'user_message', ?, NULL, ?)",
        (sid, scope, recorded_at_us),
    )
    conn.execute(
        "INSERT OR IGNORE INTO source_revisions (source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, timezone, provenance,"
        " metadata_json)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (sid, revision, payload, b"\x00" * 32, recorded_at_us,
         recorded_at_us, "UTC", "direct_user", "{}"),
    )
    bs, be = (0, len(payload)) if pinned else (None, None)
    conn.execute(
        "INSERT OR IGNORE INTO units (unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, sid, revision, scope, "turn", None, "sess1", 1,
         "alice", "user_stated", recorded_at_us, None, None, "unknown",
         "unknown", bs, be, generation),
    )
    return unit_id


_MISSING = object()


def fact(statement="Alice moved to Berlin.", subject="Alice",
         predicate="moved_to", obj="Berlin", quotes=_MISSING,
         unit_ids=_MISSING, **extra):
    f = {
        "statement": statement,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "quotes": ["moved to Berlin"] if quotes is _MISSING else quotes,
        "unit_ids": ["u1"] if unit_ids is _MISSING else unit_ids,
    }
    f.update(extra)
    return f


def model_returning(payload):
    """model_fn emitting ``payload`` (dict/list → json, str → verbatim)."""
    def _fn(prompt):
        assert "Units:" in prompt and "u1" in prompt
        return payload if isinstance(payload, str) else json.dumps(payload)
    return _fn


# ---------------------------------------------------------------------------
# the honest gate (V7-13.18)
# ---------------------------------------------------------------------------


def test_no_model_reports_unavailable(conn):
    r = t2.propose_facts(None, [{"unit_id": "u1", "text": "hi"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "unavailable"
    assert r.reason == "no_model"
    assert r.facts == []
    assert r.counts["proposed"] == 0


def test_non_callable_model_unavailable(conn):
    r = t2.propose_facts("not-a-fn", [{"unit_id": "u1", "text": "hi"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "unavailable"
    assert r.reason == "model_not_callable"


def test_raising_model_unavailable_never_crashes(conn):
    def boom(prompt):
        raise RuntimeError("endpoint down")

    r = t2.propose_facts(boom, [{"unit_id": "u1", "text": "hi"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "unavailable"
    assert r.reason == "model_error:RuntimeError"
    assert r.facts == []


def test_model_id_defaults_unknown_then_explicit(conn):
    r = t2.propose_facts(model_returning([]), [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.model_id == "unknown"
    r2 = t2.propose_facts(model_returning([]),
                          [{"unit_id": "u1", "text": "x"}],
                          scope_id=SCOPE, generation=GEN, now_us=T0,
                          model_id="ollama:qwen3:8b")
    assert r2.model_id == "ollama:qwen3:8b"


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------


def test_malformed_json_partial_never_crashes(conn):
    r = t2.propose_facts(model_returning("sure! here are facts: [oops"),
                         [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "partial"
    assert r.reason.startswith("parse_error:")
    assert r.facts == []
    assert r.counts["parse_failed"] == 1


def test_non_list_output_partial(conn):
    r = t2.propose_facts(model_returning({"statement": "x"}),
                         [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "partial"
    assert r.reason == "output_not_a_list"


def test_non_text_output_partial(conn):
    r = t2.propose_facts(lambda p: [{"not": "json-text"}],
                         [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "partial"
    assert r.reason == "model_returned_non_text"


def test_empty_array_is_ok(conn):
    r = t2.propose_facts(model_returning([]), [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "ok"
    assert r.facts == []


def test_code_fence_and_prose_salvage(conn):
    body = json.dumps([fact()])
    for wrapped in (f"```json\n{body}\n```", f"Here you go:\n{body}\nDone."):
        r = t2.propose_facts(model_returning(wrapped),
                             [{"unit_id": "u1", "text": "moved to Berlin"}],
                             scope_id=SCOPE, generation=GEN, now_us=T0)
        assert r.status == "ok", wrapped
        assert len(r.facts) == 1


def test_valid_proposal_parsed_and_pending(conn):
    r = t2.propose_facts(model_returning([fact()]),
                         [{"unit_id": "u1", "text": "moved to Berlin"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0,
                         model_id="m1")
    assert r.status == "ok"
    assert r.facts[0]["statement"] == "Alice moved to Berlin."
    # candidates are NOT verified — that verdict belongs to verify_quotes
    assert r.verdicts[0].verified is False
    assert r.verdicts[0].reasons == ["pending_byte_verification"]


def test_mixed_validity_partial(conn):
    items = [fact(), {"statement": "", "quotes": ["q"], "unit_ids": ["u1"]}]
    r = t2.propose_facts(model_returning(items),
                         [{"unit_id": "u1", "text": "moved to Berlin"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "partial"
    assert r.reason == "some_facts_invalid"
    assert len(r.facts) == 1
    assert r.counts["invalid"] == 1


def test_all_invalid_rejected(conn):
    items = [{"statement": ""}, 42]
    r = t2.propose_facts(model_returning(items),
                         [{"unit_id": "u1", "text": "x"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0)
    assert r.status == "rejected"
    assert r.reason == "all_facts_invalid"
    assert r.facts == []


def test_max_facts_cap_drops_extra(conn):
    items = [fact(statement=f"fact {i}") for i in range(5)]
    r = t2.propose_facts(model_returning(items),
                         [{"unit_id": "u1", "text": "moved to Berlin"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0,
                         max_facts=2)
    assert r.status == "partial"
    assert len(r.facts) == 2
    assert r.counts["dropped_over_cap"] == 3


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------


def test_validate_fact_ok():
    assert t2.validate_fact(fact(), known_unit_ids={"u1"}) is True


def test_validate_rejects_empty_statement():
    assert t2.validate_fact(fact(statement="")) is False
    assert t2.validate_fact(fact(statement="   ")) is False
    assert t2.validate_fact(fact(statement=None)) is False


def test_validate_rejects_long_statement():
    assert t2.validate_fact(fact(statement="x" * 513)) is False
    assert t2.validate_fact(fact(statement="x" * 512)) is True


def test_validate_rejects_missing_subject_predicate():
    assert t2.validate_fact(fact(subject=None)) is False
    assert t2.validate_fact(fact(subject="")) is False
    assert t2.validate_fact(fact(predicate=None)) is False
    assert t2.validate_fact(fact(predicate="")) is False


def test_validate_rejects_bad_quotes():
    assert t2.validate_fact(fact(quotes=[])) is False
    assert t2.validate_fact(fact(quotes=None)) is False
    assert t2.validate_fact(fact(quotes=[42])) is False
    assert t2.validate_fact(fact(quotes=[{"unit_id": "u1"}])) is False
    # lone byte bound is malformed
    assert t2.validate_fact(
        fact(quotes=[{"text": "moved", "byte_start": 0}])) is False


def test_validate_rejects_unit_mismatch():
    # quote pins a unit the fact's unit_ids does not claim
    assert t2.validate_fact(
        fact(quotes=[{"unit_id": "u2", "text": "x"}], unit_ids=["u1"])
    ) is False
    # unknown unit ids vs the provided set
    assert t2.validate_fact(fact(unit_ids=["u9"]),
                            known_unit_ids={"u1"}) is False


# ---------------------------------------------------------------------------
# quote verification (V7-13.12)
# ---------------------------------------------------------------------------


def test_verify_exact_quote(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin last week.")
    vs = t2.verify_quotes(conn, [fact()], scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is True
    assert vs[0].quotes[0]["ok"] is True
    assert vs[0].quotes[0]["unit_id"] == "u1"
    assert vs[0].support_unit_ids == ["u1"]


def test_verify_tampered_quote_rejected(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin last week.")
    vs = t2.verify_quotes(
        conn, [fact(quotes=["moved to berlin"])],  # casefolded ≠ verbatim
        scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert "quote_not_verbatim" in vs[0].reasons[0]


def test_verify_one_bad_quote_sinks_fact(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin last week.")
    vs = t2.verify_quotes(
        conn, [fact(quotes=["moved to Berlin", "invented words"])],
        scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert any("quote_1" in r for r in vs[0].reasons)


def test_verify_claimed_byte_range(conn):
    text = "Alice moved to Berlin last week."
    seed_unit(conn, "u1", text)
    start = text.encode().index(b"moved to Berlin")
    good = {"unit_id": "u1", "text": "moved to Berlin",
            "byte_start": start, "byte_end": start + 15}
    vs = t2.verify_quotes(conn, [fact(quotes=[good])],
                          scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is True
    # absolute pin: unit starts at 0 here
    assert vs[0].quotes[0]["byte_start"] == start

    bad = dict(good, byte_start=start + 1, byte_end=start + 16)
    vs = t2.verify_quotes(conn, [fact(quotes=[bad])],
                          scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert "claimed_byte_range_mismatch" in vs[0].reasons[0]


def test_verify_unpinned_unit_cannot_support(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin.", pinned=False)
    vs = t2.verify_quotes(conn, [fact()], scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert "unit_unpinned_or_payload_gone" in vs[0].reasons[0]


def test_verify_missing_unit_rejected(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin.")
    vs = t2.verify_quotes(conn, [fact(unit_ids=["u1", "u-ghost"])],
                          scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert vs[0].reasons == ["unit_not_found:u-ghost"]


def test_verify_quote_searches_all_pinned_units(conn):
    seed_unit(conn, "u1", "Alice spoke first.", source_id="s1")
    seed_unit(conn, "u2", "Bob loves hiking in the Alps.", source_id="s2")
    f = fact(statement="Bob loves hiking.", quotes=["loves hiking"],
             unit_ids=["u1", "u2"])
    vs = t2.verify_quotes(conn, [f], scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is True
    assert vs[0].quotes[0]["unit_id"] == "u2"
    assert vs[0].support_unit_ids == ["u2"]


def test_verify_structurally_invalid_passthrough(conn):
    vs = t2.verify_quotes(conn, [{"statement": ""}],
                          scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is False
    assert "statement_missing_or_too_long" in vs[0].reasons


def test_verify_multibyte_offsets(conn):
    text = "café—Alice said “moved to Berlin” yesterday"
    seed_unit(conn, "u1", text)
    vs = t2.verify_quotes(conn, [fact(quotes=["“moved to Berlin”"])],
                          scope_id=SCOPE, generation=GEN)
    assert vs[0].verified is True
    q = vs[0].quotes[0]
    payload = text.encode("utf-8")
    assert payload[q["byte_start"]:q["byte_end"]] == "“moved to Berlin”".encode()


# ---------------------------------------------------------------------------
# commit (V7-13.13 provenance, idempotent)
# ---------------------------------------------------------------------------


def committed(conn, statements_model="m1", items=None):
    """propose → verify → commit over one seeded unit; returns
    (commit_dict, T2Result)."""
    seed_unit(conn, "u1", "Alice moved to Berlin in March 2023.")
    r = t2.propose_facts(
        model_returning(items if items is not None else [fact()]),
        [{"unit_id": "u1", "text": "Alice moved to Berlin in March 2023."}],
        scope_id=SCOPE, generation=GEN, now_us=T0,
        model_id=statements_model)
    vs = t2.verify_quotes(conn, r.facts, scope_id=SCOPE, generation=GEN)
    return t2.commit_facts(conn, vs, scope_id=SCOPE, generation=GEN,
                           model_id=statements_model, now_us=T0), r


def test_commit_inserts_verified_row(conn):
    out, _ = committed(conn)
    assert out["inserted"] == 1 and out["rejected"] == 0
    row = conn.execute("SELECT * FROM t2_facts").fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM t2_facts").description]
    rec = dict(zip(cols, row))
    assert rec["fact_id"].startswith("t2:")
    assert rec["scope_id"] == SCOPE
    assert rec["verified"] == 1
    assert rec["model_id"] == "m1"
    assert rec["prompt_digest"] == t2.prompt_digest()
    assert rec["subject_canon"] == "alice"
    assert rec["predicate"] == "moved_to"
    assert json.loads(rec["unit_ids_json"]) == ["u1"]
    quotes = json.loads(rec["quotes_json"])
    assert quotes[0]["text"] == "moved to Berlin"
    assert quotes[0]["byte_start"] is not None


def test_commit_idempotent_recommit_skips(conn):
    out1, _ = committed(conn)
    out2, _ = committed(conn)  # same statement/units/model → same fact_id
    assert out1["inserted"] == 1
    assert out2["inserted"] == 0 and out2["skipped_existing"] == 1
    assert conn.execute("SELECT COUNT(*) FROM t2_facts").fetchone()[0] == 1


def test_commit_rejected_verdict_not_inserted(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin.")
    vs = t2.verify_quotes(conn, [fact(quotes=["fabricated"])],
                          scope_id=SCOPE, generation=GEN)
    out = t2.commit_facts(conn, vs, scope_id=SCOPE, generation=GEN,
                          model_id="m1")
    assert out["inserted"] == 0 and out["rejected"] == 1
    assert conn.execute("SELECT COUNT(*) FROM t2_facts").fetchone()[0] == 0


def test_commit_occurred_from_quote_not_model(conn):
    # model claims a bogus occurred value — never trusted (V7-13.13);
    # "in March 2023" in the quote resolves via temporal/v2 instead.
    out, _ = committed(
        conn, items=[fact(occurred_start_us=1, occurred_end_us=2,
                          quotes=["moved to Berlin in March 2023"])])
    assert out["inserted"] == 1
    row = conn.execute(
        "SELECT occurred_start_us, occurred_end_us FROM t2_facts"
    ).fetchone()
    assert row[0] == 1677628800000000 and row[1] == 1680307200000000


# ---------------------------------------------------------------------------
# retire (V7-13.13 lifecycle sweep)
# ---------------------------------------------------------------------------


def test_retire_when_unit_erased(conn):
    committed(conn)
    conn.execute("DELETE FROM units WHERE unit_id = 'u1'")
    out = t2.retire_facts(conn, scope_id=SCOPE, generation=GEN)
    assert out["retired"] == 1
    row = conn.execute(
        "SELECT verified FROM t2_facts").fetchone()
    assert row[0] == 0  # row kept — audit trail, not deleted


def test_retire_via_eligibility_callback(conn):
    committed(conn)
    held = {"u1"}
    out = t2.retire_facts(
        conn, scope_id=SCOPE, generation=GEN,
        is_unit_eligible=lambda uid: uid not in held)
    assert out["retired"] == 1
    assert "unit_ineligible:u1" in out["reasons"][out["retired_ids"][0]]


def test_retire_leaves_supported_fact(conn):
    committed(conn)
    out = t2.retire_facts(conn, scope_id=SCOPE, generation=GEN)
    assert out["checked"] == 1 and out["retired"] == 0
    assert conn.execute("SELECT verified FROM t2_facts").fetchone()[0] == 1


def test_retire_partial_support_sinks_fact(conn):
    seed_unit(conn, "u1", "Alice moved to Berlin.", source_id="s1")
    seed_unit(conn, "u2", "and loves it there.", source_id="s2")
    f = fact(quotes=[{"unit_id": "u1", "text": "moved to Berlin"},
                     {"unit_id": "u2", "text": "loves it there"}],
             unit_ids=["u1", "u2"])
    r = t2.propose_facts(model_returning([f]),
                         [{"unit_id": "u1"}, {"unit_id": "u2"}],
                         scope_id=SCOPE, generation=GEN, now_us=T0,
                         model_id="m1")
    vs = t2.verify_quotes(conn, r.facts, scope_id=SCOPE, generation=GEN)
    assert vs[0].verified and sorted(vs[0].support_unit_ids) == ["u1", "u2"]
    t2.commit_facts(conn, vs, scope_id=SCOPE, generation=GEN, model_id="m1")
    conn.execute("DELETE FROM units WHERE unit_id = 'u2'")
    out = t2.retire_facts(conn, scope_id=SCOPE, generation=GEN)
    assert out["retired"] == 1  # ANY pinned unit going sinks the fact


# ---------------------------------------------------------------------------
# prompt contract (V7-13.17) + end-to-end
# ---------------------------------------------------------------------------


def test_prompt_digest_stable_and_prompt_bound():
    assert t2.prompt_digest() == t2.prompt_digest(t2.T2_PROMPT_V1)
    assert t2.prompt_digest("other prompt") != t2.prompt_digest()
    # generic prompt: no benchmark/dataset hooks (V7-13.17)
    low = t2.T2_PROMPT_V1.lower()
    for banned in ("locomo", "longmemeval", "benchmark", "dataset"):
        assert banned not in low


def test_end_to_end_propose_verify_commit_retire(conn):
    text = "Alice moved to Berlin. Bob stayed in Rome."
    seed_unit(conn, "u1", text)
    items = [
        fact(statement="Alice moved to Berlin.", quotes=["moved to Berlin"]),
        fact(statement="Bob moved to Berlin.", subject="Bob",
             quotes=["moved to Berlin"]),  # false attribution — quote is there
        fact(statement="Alice moved to Paris.", quotes=["moved to Paris"]),
    ]
    r = t2.propose_facts(model_returning(items),
                         [{"unit_id": "u1", "text": text}],
                         scope_id=SCOPE, generation=GEN, now_us=T0,
                         model_id="m1")
    assert r.status == "ok" and len(r.facts) == 3
    vs = t2.verify_quotes(conn, r.facts, scope_id=SCOPE, generation=GEN)
    assert [v.verified for v in vs] == [True, True, False]
    out = t2.commit_facts(conn, vs, scope_id=SCOPE, generation=GEN,
                          model_id="m1", now_us=T0)
    assert out["inserted"] == 2 and out["rejected"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM t2_facts WHERE verified = 1").fetchone()[0] == 2
    conn.execute("DELETE FROM units WHERE unit_id = 'u1'")
    out = t2.retire_facts(conn, scope_id=SCOPE, generation=GEN)
    assert out["retired"] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM t2_facts WHERE verified = 1").fetchone()[0] == 0
    # rows persist for audit — only the verified flag moved
    assert conn.execute(
        "SELECT COUNT(*) FROM t2_facts").fetchone()[0] == 2
