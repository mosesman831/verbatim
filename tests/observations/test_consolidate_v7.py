"""Tests for ``verbatim/observations/consolidate_v7.py`` (w-consolidate,
wave B) — T1 belief consolidation over the V7 unit plane (V7-14.01–08).

Fixtures ride the real DDL: the ``store`` fixture (v1 schema →
``sources``/``source_revisions``/``scopes``) plus
``schema_v7.ensure_v7_additive`` for the §30 tables, so quotes are
genuinely byte-pinned into persisted payloads and every write is a real
SQL row — no SQL-layer mocks. The TestStore connection runs in
autocommit (``isolation_level=None``): each statement is its own commit,
which exercises ``consolidate_scope_v7`` as a pure in-tx function.
"""

from __future__ import annotations

import json

import pytest

from verbatim.observations.consolidate_v7 import (
    BACKLOG_WARN_AGE_US,
    consolidate_scope_v7,
    backlog_v7,
    ensure_consolidation_v7,
)
from verbatim.core.types import VerbatimError
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "s1"
GEN = 1
NOW = 1_800_000_000_000_000


def _bootstrap(conn):
    conn.execute(
        "INSERT INTO scopes(scope_id, profile_id, visibility)"
        " VALUES (?,?,?)",
        (SCOPE, "prof", "conversation"),
    )
    ensure_v7_additive(conn)
    ensure_consolidation_v7(conn)


def _src(conn, source_id, payload: str, scope=SCOPE, rev=1):
    conn.execute(
        "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?,?,?,?,?)",
        (source_id, "test", "user_message", scope, 0),
    )
    pb = payload.encode("utf-8")
    conn.execute(
        "INSERT INTO source_revisions(source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, provenance)"
        " VALUES (?,?,?,?,?,?,?)",
        (source_id, rev, pb, b"h", 0, 0, "direct_user"),
    )


def _unit(conn, unit_id, payload="text", *, scope=SCOPE, gen=GEN,
          occurred=None, recorded=None, session=None):
    _src(conn, f"src-{unit_id}", payload, scope=scope)
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " session_id, seq, recorded_at_us, occurred_start_us,"
        " occurred_end_us, byte_start, byte_end, generation)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, f"src-{unit_id}", 1, scope, "turn", session, 1,
         recorded, occurred, occurred, 0,
         len(payload.encode("utf-8")), gen),
    )


def _state(conn, unit_id, state_key, value, *, status="current",
           vnorm=None, valid_from=None, gen=GEN, scope=SCOPE):
    conn.execute(
        "INSERT INTO state_facts(scope_id, state_key, unit_id,"
        " generation, value_text, value_norm, valid_from_us, status,"
        " producer, pins_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (scope, state_key, unit_id, gen, value,
         vnorm if vnorm is not None else str(value).lower(),
         valid_from, status, "t0", "{}"),
    )


def _pref(conn, unit_id, subject, obj, *, polarity="affirm",
          strength=None, occurred=None, gen=GEN, scope=SCOPE):
    conn.execute(
        "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
        " generation, object_text, polarity, strength,"
        " occurred_start_us, pins_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (scope, subject, unit_id, gen, obj, polarity, strength,
         occurred, "{}"),
    )


def _event(conn, event_id, unit_id, subject, predicate, obj, *,
           polarity="affirm", occurred=None, gen=GEN, scope=SCOPE):
    conn.execute(
        "INSERT INTO events_v7(event_id, unit_id, scope_id,"
        " subject_canon, predicate_lemma, object_text, polarity,"
        " occurred_start_us, occurred_end_us, precision, pins_json,"
        " rule_id, generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (event_id, unit_id, scope, subject, predicate, obj, polarity,
         occurred, occurred, "day", "{}", f"{predicate}/test", gen),
    )


def _mention(conn, unit_id, canon, *, role="mention", gen=GEN, scope=SCOPE):
    conn.execute(
        "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
        " generation, surface, byte_start, byte_end, role)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, canon, unit_id, gen, canon, 0, 1, role),
    )


def _sq(conn, sq_id, filters=None, *, scope=SCOPE, gen=GEN):
    conn.execute(
        "INSERT INTO standing_queries(sq_id, scope_id, principal,"
        " query, filters_json, dirty, generation)"
        " VALUES (?,?,?,?,?,?,?)",
        (sq_id, scope, "alice", "q?", json.dumps(filters or {}), 0, gen),
    )


def _obs_rows(conn, scope=SCOPE, gen=GEN):
    cur = conn.execute(
        "SELECT obs_id, scope_id, slot, text, producer, proof_count,"
        " support_refs_json, contradict_refs_json, first_us, last_us,"
        " stale, generation FROM observations_v7"
        " WHERE scope_id=? AND generation=? ORDER BY obs_id",
        (scope, gen),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _profile_rows(conn, scope=SCOPE, gen=GEN):
    cur = conn.execute(
        "SELECT scope_id, subject_canon, slot, value, status,"
        " support_refs_json, updated_us, generation FROM profiles_v7"
        " WHERE scope_id=? AND generation=? ORDER BY slot",
        (scope, gen),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _run(conn, scope=SCOPE, gen=GEN, budget=10_000, now_us=NOW):
    return consolidate_scope_v7(
        conn, scope_id=scope, generation=gen,
        budget_units=budget, now_us=now_us)


@pytest.fixture()
def conn(store):
    """Bootstrapped shared connection (autocommit — every statement is
    its own committed tx, matching the pure in-tx contract)."""
    with store.tx() as c:
        _bootstrap(c)
    return store._conn


def _seed_berlin_obs(conn):
    """state(u1) + event(u2) agreeing on alice/home_city=berlin."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _run(conn)


# ---------------------------------------------------------------------------
# corroboration threshold + families (V7-14.01)
# ---------------------------------------------------------------------------


def test_two_units_distinct_families_consolidate(conn):
    """state + event agreeing on one slot → 1 obs, proof_count=2, both
    support refs carry the unit's pinned payload slice."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100, session="se1")
    _unit(conn, "u2", "I moved to Berlin.", occurred=200, session="se1")
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    st = _run(conn)
    assert st["status"] == "ok" and st["observations_written"] == 1
    obs = _obs_rows(conn)
    assert len(obs) == 1
    o = obs[0]
    assert o["obs_id"].startswith("obs7:")
    assert o["slot"] == "alice/home_city"
    assert o["text"] == "alice lives in berlin"
    assert o["proof_count"] == 2
    assert o["stale"] == 0
    assert o["first_us"] == 100 and o["last_us"] == 200
    refs = json.loads(o["support_refs_json"])
    assert [r["unit_id"] for r in refs] == ["u1", "u2"]
    assert refs[0]["quote"] == "I live in Berlin."
    assert refs[0]["byte_start"] == 0
    assert refs[0]["byte_end"] == len("I live in Berlin.")
    assert {f for r in refs for f in r["families"]} == {
        "state", "event"}


def test_single_unit_no_obs(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["observations_written"] == 0
    assert _obs_rows(conn) == []


def test_same_family_twice_never_consolidates(conn):
    """Two state_facts from two units are ONE family — the documented
    interpretation of 'distinct evidence families': corroboration needs
    independent evidence kinds, not repeated same-kind rows."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I live in Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _state(conn, "u2", "alice/home_city", "Berlin", valid_from=200)
    st = _run(conn)
    assert st["observations_written"] == 0
    assert _obs_rows(conn) == []


def test_preference_plus_mention_consolidates(conn):
    """preference(u1) + co-mention of subject & value (u2) → obs."""
    _unit(conn, "u1", "I love sushi.", occurred=100)
    _unit(conn, "u2", "Alice ordered sushi again.", occurred=200)
    _pref(conn, "u1", "alice", "sushi", occurred=100)
    _mention(conn, "u2", "alice")
    _mention(conn, "u2", "sushi")
    st = _run(conn)
    assert st["observations_written"] == 1
    (o,) = _obs_rows(conn)
    assert o["slot"] == "alice/preferences.likes"
    assert o["text"] == "alice likes sushi"
    assert o["proof_count"] == 2
    fams = {f for r in json.loads(o["support_refs_json"])
            for f in r["families"]}
    assert fams == {"preference", "mention"}


def test_mentions_alone_never_consolidate(conn):
    """Co-mention is corroboration, not assertion: two units that merely
    co-mention subject+value assert nothing → no obs."""
    _unit(conn, "u1", "Alice and sushi.", occurred=100)
    _unit(conn, "u2", "Alice had sushi.", occurred=200)
    for u in ("u1", "u2"):
        _mention(conn, u, "alice")
        _mention(conn, u, "sushi")
    st = _run(conn)
    assert st["observations_written"] == 0
    assert _obs_rows(conn) == []


def test_negated_event_does_not_corroborate_affirm(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I did not move to Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin",
           polarity="negate", occurred=200)
    st = _run(conn)
    assert st["observations_written"] == 0
    assert _obs_rows(conn) == []


def test_hypothetical_does_not_corroborate(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I might move to Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin",
           polarity="hypothetical", occurred=200)
    st = _run(conn)
    assert st["observations_written"] == 0


def test_historical_state_not_support(conn):
    """status='historical' state rows are slot activity but never support
    a present-tense observation — a lone current event is left at 1."""
    _unit(conn, "u1", "I lived in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin",
           status="historical", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    st = _run(conn)
    assert st["observations_written"] == 0
    assert _obs_rows(conn) == []


def test_single_unit_two_families_still_no_obs(conn):
    """One unit carrying a state fact AND a co-mention contributes both
    families but only one unit — the ≥2-unit gate still fails."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _mention(conn, "u1", "alice")
    _mention(conn, "u1", "berlin")
    st = _run(conn)
    assert st["observations_written"] == 0


def test_event_gate_celebrate_birthday_vs_age(conn):
    """The celebrate predicate splits by object: 'birthday' nouns → the
    birthday slot, a bare number → age; an ungated object lands nowhere."""
    _unit(conn, "u1", "My birthday is June 4.", occurred=100)
    _unit(conn, "u2", "I celebrated my birthday.", occurred=200)
    _state(conn, "u1", "alice/birthday", "june 4", valid_from=100)
    _event(conn, "e1", "u2", "alice", "celebrate", "my birthday",
           occurred=200)
    st = _run(conn)
    # event object 'my birthday' matches the birthday gate; but the
    # values differ ('june 4' vs 'my birthday') → two value groups,
    # each < 2 units → no obs.
    assert st["observations_written"] == 0


# ---------------------------------------------------------------------------
# freshness (V7-14.03)
# ---------------------------------------------------------------------------


def test_newer_different_value_marks_stale(conn):
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I live in Munich.", occurred=300)
    _state(conn, "u3", "alice/home_city", "Munich", valid_from=300)
    st = _run(conn)
    (o,) = _obs_rows(conn)
    assert o["text"] == "alice lives in berlin"
    assert o["stale"] == 1
    assert st["stale_marked"] == 1


def test_stale_then_corroborated_new_value(conn):
    """berlin obs stays stale (labeled history); the corroborated munich
    belief materializes fresh."""
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I live in Munich.", occurred=300)
    _state(conn, "u3", "alice/home_city", "Munich", valid_from=300)
    _run(conn)
    _unit(conn, "u4", "I moved to Munich.", occurred=400)
    _event(conn, "e2", "u4", "alice", "move", "Munich", occurred=400)
    _run(conn)
    obs = {o["text"]: o for o in _obs_rows(conn)}
    assert obs["alice lives in berlin"]["stale"] == 1
    assert obs["alice lives in munich"]["stale"] == 0
    assert obs["alice lives in munich"]["proof_count"] == 2


def test_newer_agreeing_unit_joins_support_no_stale(conn):
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I moved to Berlin.", occurred=300)
    _event(conn, "e2", "u3", "alice", "move", "Berlin", occurred=300)
    st = _run(conn)
    (o,) = _obs_rows(conn)
    assert o["stale"] == 0
    assert o["proof_count"] == 3
    assert st["observations_updated"] == 1
    assert st["stale_marked"] == 0


def test_accumulating_family_new_value_not_stale(conn):
    """pets accumulates: a newer 'has cat' does not stale 'has dog'."""
    _unit(conn, "u1", "I have a dog.", occurred=100)
    _unit(conn, "u2", "I adopted a dog.", occurred=200)
    _state(conn, "u1", "alice/pets", "dog", valid_from=100)
    _event(conn, "e1", "u2", "alice", "adopt", "dog", occurred=200)
    _run(conn)
    _unit(conn, "u3", "I have a cat.", occurred=300)
    _state(conn, "u3", "alice/pets", "cat", valid_from=300)
    _run(conn)
    (o,) = _obs_rows(conn)
    assert o["text"] == "alice has pet dog"
    assert o["stale"] == 0


def test_accumulating_family_negation_stales(conn):
    """'no longer has dog' in an accumulating slot DOES stale 'has dog'
    (same value, conflicting polarity)."""
    _unit(conn, "u1", "I have a dog.", occurred=100)
    _unit(conn, "u2", "I adopted a dog.", occurred=200)
    _state(conn, "u1", "alice/pets", "dog", valid_from=100)
    _event(conn, "e1", "u2", "alice", "adopt", "dog", occurred=200)
    _run(conn)
    _unit(conn, "u3", "I do not have the dog anymore.", occurred=300)
    _event(conn, "e2", "u3", "alice", "adopt", "dog",
           polarity="negate", occurred=300)
    _run(conn)
    (o,) = _obs_rows(conn)
    assert o["stale"] == 1


def test_older_backfill_does_not_stale(conn):
    """An assertion older than obs.last_us is not 'newer' — no stale."""
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I lived in Munich in 2010.", occurred=50)
    _state(conn, "u3", "alice/home_city", "Munich",
           status="historical", valid_from=50)
    _run(conn)
    (o,) = _obs_rows(conn)
    assert o["stale"] == 0


# ---------------------------------------------------------------------------
# near-dup merge (V7-14.04)
# ---------------------------------------------------------------------------


def test_near_dup_merge_folds(conn):
    """'berlin' vs 'berlino' produce text similarity > 0.9 in one slot →
    one canonical obs (the older), support unioned, merge recorded."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I live in Berlino.", occurred=300)
    _unit(conn, "u4", "I moved to Berlino.", occurred=400)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u3", "alice/home_city", "Berlino", valid_from=300)
    _event(conn, "e2", "u4", "alice", "move", "Berlino", occurred=400)
    st = _run(conn)
    assert st["merges"] == 1
    obs = _obs_rows(conn)
    assert len(obs) == 1
    o = obs[0]
    assert o["text"] == "alice lives in berlin"  # older canonical
    assert o["proof_count"] == 4
    refs = json.loads(o["support_refs_json"])
    assert [r["unit_id"] for r in refs] == ["u1", "u2", "u3", "u4"]
    assert o["first_us"] == 100 and o["last_us"] == 400
    mg = conn.execute(
        "SELECT into_obs_id, from_obs_id, from_row_json"
        " FROM obs_merges_v7").fetchall()
    assert len(mg) == 1
    assert mg[0][0] == o["obs_id"]
    from_row = json.loads(mg[0][2])
    assert from_row["text"] == "alice lives in berlino"
    assert from_row["proof_count"] == 2


def test_dissimilar_values_no_merge(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I live in Munich.", occurred=300)
    _unit(conn, "u4", "I moved to Munich.", occurred=400)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u3", "alice/home_city", "Munich", valid_from=300)
    _event(conn, "e2", "u4", "alice", "move", "Munich", occurred=400)
    st = _run(conn)
    assert st["merges"] == 0
    obs = {o["text"]: o["stale"] for o in _obs_rows(conn)}
    assert set(obs) == {"alice lives in berlin", "alice lives in munich"}
    assert obs["alice lives in berlin"] == 1   # superseded value stales
    assert obs["alice lives in munich"] == 0


def test_merge_not_cross_slot(conn):
    """Similar texts in DIFFERENT slots never merge (slot equality is a
    hard precondition of V7-14.04)."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "Bob lives in Berlin.", occurred=300)
    _unit(conn, "u4", "Bob moved to Berlin.", occurred=400)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u3", "bob/home_city", "Berlin", valid_from=300)
    _event(conn, "e2", "u4", "bob", "move", "Berlin", occurred=400)
    st = _run(conn)
    assert st["merges"] == 0
    assert len(_obs_rows(conn)) == 2


# ---------------------------------------------------------------------------
# profiles (V7-14.06)
# ---------------------------------------------------------------------------


def test_profiles_materialize_slots(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I love sushi.", occurred=150)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _pref(conn, "u3", "alice", "sushi", occurred=150)
    st = _run(conn)
    assert st["profiles_written"] >= 2
    prof = {p["slot"]: p for p in _profile_rows(conn)}
    assert prof["home_city"]["value"] == "Berlin"
    assert prof["home_city"]["status"] == "current"
    assert prof["preferences.likes"]["value"] == "sushi"
    assert prof["preferences.likes"]["status"] == "current"
    refs = json.loads(prof["home_city"]["support_refs_json"])
    assert {r["unit_id"] for r in refs} == {"u1", "u2"}
    assert all(r["quote"] for r in refs)


def test_profile_disputed_status(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I live in Munich.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _state(conn, "u2", "alice/home_city", "Munich", valid_from=200)
    _run(conn)
    prof = {p["slot"]: p for p in _profile_rows(conn)}
    assert prof["home_city"]["status"] == "disputed"


def test_profile_historical_when_only_events(conn):
    _unit(conn, "u1", "I moved to Berlin.", occurred=100)
    _event(conn, "e1", "u1", "alice", "move", "Berlin", occurred=100)
    _run(conn)
    prof = {p["slot"]: p for p in _profile_rows(conn)}
    assert prof["home_city"]["status"] == "historical"


def test_profile_scoped_to_subject(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _state(conn, "u1", "bob/home_city", "Paris", valid_from=100)
    _run(conn)
    prof = {(p["subject_canon"], p["slot"]): p
            for p in _profile_rows(conn)}
    assert prof[("alice", "home_city")]["value"] == "Berlin"
    assert prof[("bob", "home_city")]["value"] == "Paris"


# ---------------------------------------------------------------------------
# standing queries (V7-14.07)
# ---------------------------------------------------------------------------


def test_standing_query_dirty_on_scope_change(conn):
    _sq(conn, "sq1", filters={})
    _unit(conn, "u1", "I live in Berlin.", occurred=100, session="se1")
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 1
    assert conn.execute(
        "SELECT dirty FROM standing_queries WHERE sq_id='sq1'"
    ).fetchone()[0] == 1


def test_standing_query_excluded_by_session(conn):
    """A standing query filtered to session 'other' provably excludes the
    changed units in 'se1' → stays clean."""
    _sq(conn, "sq1", filters={"sessions": ["other"]})
    _unit(conn, "u1", "I live in Berlin.", occurred=100, session="se1")
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 0
    assert conn.execute(
        "SELECT dirty FROM standing_queries WHERE sq_id='sq1'"
    ).fetchone()[0] == 0


def test_standing_query_excluded_by_subject(conn):
    _sq(conn, "sq1", filters={"subjects": ["bob"]})
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 0


def test_standing_query_dirty_on_subject_overlap(conn):
    _sq(conn, "sq1", filters={"subjects": ["alice"]})
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 1


def test_standing_query_unknown_filter_key_dirty(conn):
    """An unrecognized filter key is in-doubt → dirty (conservative)."""
    _sq(conn, "sq1", filters={"lanes": ["ent"]})
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 1


def test_standing_query_subjectless_unit_dirty(conn):
    """A changed unit with no extracted subjects cannot be excluded by a
    subject filter — in doubt → dirty."""
    _sq(conn, "sq1", filters={"subjects": ["bob"]})
    _unit(conn, "u1", "no facts here.", occurred=100)
    st = _run(conn)
    assert st["standing_dirty"] == 1


def test_standing_query_other_scope_untouched(conn):
    conn.execute(
        "INSERT INTO scopes(scope_id, profile_id, visibility)"
        " VALUES ('s2','prof','conversation')")
    _sq(conn, "sq2", filters={}, scope="s2")
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["standing_dirty"] == 0
    assert conn.execute(
        "SELECT dirty FROM standing_queries WHERE sq_id='sq2'"
    ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# closure (V7-14.08)
# ---------------------------------------------------------------------------


def test_closure_retires_obs_on_unit_delete(conn):
    _seed_berlin_obs(conn)
    assert len(_obs_rows(conn)) == 1
    conn.execute("DELETE FROM units WHERE unit_id='u2'")
    st = _run(conn)
    assert _obs_rows(conn) == []
    assert any(r.startswith("observations_v7:") for r in st["retired"])


def test_closure_retires_profile_on_unit_delete(conn):
    _seed_berlin_obs(conn)
    assert _profile_rows(conn)
    conn.execute("DELETE FROM units WHERE unit_id='u2'")
    st = _run(conn)
    assert _profile_rows(conn) == []
    assert any(r.startswith("profiles_v7:") for r in st["retired"])


def test_closure_leaves_intact_rows(conn):
    """Deleting a unit that supports NOTHING must not retire the obs."""
    _seed_berlin_obs(conn)
    _unit(conn, "u9", "unrelated chatter.", occurred=50)
    conn.execute("DELETE FROM units WHERE unit_id='u9'")
    _run(conn)
    assert len(_obs_rows(conn)) == 1


def test_support_collapse_retires_obs(conn):
    """Drop a group's family diversity below 2 → the standing obs for
    that value is retired on re-derivation."""
    _seed_berlin_obs(conn)
    conn.execute("DELETE FROM events_v7 WHERE event_id='e1'")
    # touch the slot so it re-derives: a fresh asserting unit
    _unit(conn, "u3", "I live in Berlin.", occurred=300)
    _state(conn, "u3", "alice/home_city", "Berlin", valid_from=300)
    st = _run(conn)
    # group now = {u1 state, u3 state} — one family → obs retires
    assert _obs_rows(conn) == []
    assert st["retired"]


# ---------------------------------------------------------------------------
# determinism / budget / backlog (V7-14.02, 14.05)
# ---------------------------------------------------------------------------


def test_idempotent_second_run(conn):
    _seed_berlin_obs(conn)
    before = conn.execute("SELECT * FROM observations_v7").fetchall()
    st = _run(conn)
    assert st["processed"] == 0
    assert st["observations_written"] == 0
    assert st["observations_updated"] == 0
    assert conn.execute(
        "SELECT * FROM observations_v7").fetchall() == before


def test_byte_identical_rederivation(conn):
    """Delete the artifact + cursors, re-run over the same pins → the
    re-derived row is byte-identical (V7-14.02)."""
    _seed_berlin_obs(conn)
    before = conn.execute("SELECT * FROM observations_v7").fetchall()
    conn.execute("DELETE FROM observations_v7")
    conn.execute("DELETE FROM consolidation_seen_v7")
    conn.execute("DELETE FROM consolidation_cursor_v7")
    _run(conn)
    after = conn.execute("SELECT * FROM observations_v7").fetchall()
    assert after == before


def test_budget_and_cursor_continuation(conn):
    """budget_units bounds the scan; the second call resumes at the
    durable cursor — never reprocesses."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I moved to Berlin.", occurred=300)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _event(conn, "e2", "u3", "alice", "move", "Berlin", occurred=300)
    st = _run(conn, budget=2)
    assert st["processed"] == 2 and st["remaining"] == 1
    assert st["observations_written"] == 1
    st = _run(conn, budget=2)
    assert st["processed"] == 1 and st["remaining"] == 0
    (o,) = _obs_rows(conn)
    assert o["proof_count"] == 3  # u3 folded into support
    st = _run(conn)
    assert st["processed"] == 0


def test_backlog_report(conn):
    _unit(conn, "u1", "a", occurred=100, recorded=NOW - 1000)
    _unit(conn, "u2", "b", occurred=200, recorded=NOW - 500)
    bl = backlog_v7(conn, SCOPE, GEN, now_us=NOW)
    assert bl["status"] == "ok"
    assert bl["unconsolidated_units"] == 2
    assert bl["oldest_age_us"] == 1000
    assert bl["warning"] is False
    _run(conn)
    bl = backlog_v7(conn, SCOPE, GEN, now_us=NOW)
    assert bl["unconsolidated_units"] == 0
    assert bl["oldest_age_us"] is None


def test_backlog_warning_on_age(conn):
    _unit(conn, "u1", "a", occurred=100,
          recorded=NOW - BACKLOG_WARN_AGE_US - 1)
    bl = backlog_v7(conn, SCOPE, GEN, now_us=NOW)
    assert bl["warning"] is True


def test_backlog_throughput_estimate(conn):
    _unit(conn, "u1", "a", occurred=100)
    _unit(conn, "u2", "b", occurred=200)
    _run(conn, now_us=NOW)
    _unit(conn, "u3", "c", occurred=300)
    _run(conn, now_us=NOW + 2_000_000)  # +2 s
    bl = backlog_v7(conn, SCOPE, GEN, now_us=NOW + 2_000_000)
    assert bl["throughput_units_per_s"] == 0.5  # 1 unit / 2 s


def test_side_tables_created_lazily(conn):
    _unit(conn, "u1", "a", occurred=100)
    _run(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"consolidation_cursor_v7", "consolidation_seen_v7",
            "obs_merges_v7"} <= names


# ---------------------------------------------------------------------------
# robustness / fencing
# ---------------------------------------------------------------------------


def test_unavailable_without_units_table(store):
    with store.tx() as conn:
        ensure_consolidation_v7(conn)
        st = consolidate_scope_v7(
            conn, scope_id="s1", generation=1, now_us=NOW)
        assert st["status"] == "unavailable"
        assert st["reason"] == "units_table_missing"
        bl = backlog_v7(conn, "s1", 1, now_us=NOW)
        assert bl["status"] == "unavailable"


def test_empty_scope_ok(conn):
    st = _run(conn)
    assert st["status"] == "ok"
    assert st["processed"] == 0 and st["remaining"] == 0


def test_generation_fencing(conn):
    """Units/facts at another generation are invisible to this pass."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100, gen=2)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200, gen=2)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100, gen=2)
    _event(conn, "e1", "u2", "alice", "move", "Berlin",
           occurred=200, gen=2)
    st = _run(conn, gen=GEN)
    assert st["processed"] == 0
    st = _run(conn, gen=2)
    assert st["processed"] == 2
    assert st["observations_written"] == 1


def test_scope_isolation(conn):
    conn.execute(
        "INSERT INTO scopes(scope_id, profile_id, visibility)"
        " VALUES ('s2','prof','conversation')")
    _unit(conn, "u1", "I live in Berlin.", occurred=100, scope="s2")
    _unit(conn, "u2", "I moved to Berlin.", occurred=200, scope="s2")
    _state(conn, "u1", "alice/home_city", "Berlin",
           valid_from=100, scope="s2")
    _event(conn, "e1", "u2", "alice", "move", "Berlin",
           occurred=200, scope="s2")
    st = _run(conn, scope=SCOPE)
    assert st["processed"] == 0
    st = _run(conn, scope="s2")
    assert st["observations_written"] == 1


def test_unpinned_unit_cannot_support(conn):
    """A unit whose byte pin cannot resolve supplies no quote → it is
    excluded from support; support below 2 → no obs (V7-14.01)."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id,"
        " kind, recorded_at_us, occurred_start_us, occurred_end_us,"
        " byte_start, byte_end, generation)"
        " VALUES ('u2','src-missing',1,?, 'turn',300,300,300,"
        " NULL,NULL,?)",
        (SCOPE, GEN),
    )
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=300)
    st = _run(conn)
    assert st["skipped_unpinned"] >= 1
    assert _obs_rows(conn) == []


def test_validation_errors(conn):
    with pytest.raises(VerbatimError):
        consolidate_scope_v7(conn, scope_id="", generation=GEN)
    with pytest.raises(VerbatimError):
        consolidate_scope_v7(conn, scope_id=SCOPE, generation=GEN,
                             budget_units=0)


def test_contradict_refs_populated(conn):
    """The losing value's units land in contradict_refs_json."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I moved to Munich.", occurred=300)
    _unit(conn, "u4", "I live in Munich.", occurred=400)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u4", "alice/home_city", "Munich", valid_from=400)
    _event(conn, "e2", "u3", "alice", "move", "Munich", occurred=300)
    _run(conn)
    obs = {o["text"]: o for o in _obs_rows(conn)}
    contra = json.loads(
        obs["alice lives in berlin"]["contradict_refs_json"])
    assert {c["value"] for c in contra} == {"munich"}
    assert {c["unit_id"] for c in contra} == {"u3", "u4"}


def test_multiple_subjects_independent(conn):
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I love sushi.", occurred=100)
    _unit(conn, "u4", "Alice had sushi.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _pref(conn, "u3", "alice", "sushi", occurred=100)
    _mention(conn, "u4", "alice")
    _mention(conn, "u4", "sushi")
    _run(conn)
    slots = {o["slot"] for o in _obs_rows(conn)}
    assert slots == {"alice/home_city", "alice/preferences.likes"}


# ---------------------------------------------------------------------------
# per-run trace (V8-13.02, K75; D8-19 diagnosis)
# ---------------------------------------------------------------------------

_TRACE_COUNTERS = ("run_seq", "units_processed", "units_with_assertions",
                   "candidates", "accepted", "inserted", "updated",
                   "unchanged", "written", "merged", "skipped_unpinned",
                   "retired")


def test_trace_emitted_every_run(conn):
    """K75: each consolidation run emits a trace — including a run over
    an empty scope and a run on a scope missing the units table."""
    st = _run(conn)
    tr = st["trace"]
    for k in _TRACE_COUNTERS:
        assert k in tr, k
    assert tr["slots"] == [] and tr["rejected"] == []
    assert tr["profiles"] == {"written": 0, "rejected": []}
    assert tr["run_seq"] == 1


def test_trace_emitted_on_unavailable(store):
    with store.tx() as conn:
        ensure_consolidation_v7(conn)
        st = consolidate_scope_v7(
            conn, scope_id="s1", generation=1, now_us=NOW)
        assert st["status"] == "unavailable"
        assert st["trace"]["candidates"] == 0
        assert st["trace"]["rejected"] == []


def test_trace_candidates_slots_written(conn):
    """Candidate slots carry assertion/candidate counts; the write-path
    counters agree with the durable stats."""
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I moved to Berlin.", occurred=300)
    _event(conn, "e2", "u3", "alice", "move", "Berlin", occurred=300)
    st = _run(conn)
    tr = st["trace"]
    assert tr["units_processed"] == 1
    assert tr["units_with_assertions"] == 1
    (slot,) = tr["slots"]
    assert slot["slot"] == "alice/home_city"
    assert slot["subject"] == "alice" and slot["family"] == "home_city"
    assert slot["assertions"] == 3 and slot["candidates"] == 1
    assert tr["candidates"] == 1
    assert tr["accepted"] == 1
    assert tr["inserted"] == 0 and tr["updated"] == 1
    assert tr["unchanged"] == 0
    assert tr["written"] == 1
    assert tr["written"] == (st["observations_written"]
                             + st["observations_updated"])
    assert tr["rejected"] == []


def test_trace_rejection_names_each_gate(conn):
    """A lone single-family unit fails BOTH corroboration gates; each
    failure lands its own entry naming the gate and the measured
    counts behind the reason."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    rej = st["trace"]["rejected"]
    assert len(rej) == 2
    gates = [r["gate"] for r in rej]
    assert gates == ["min_support_units", "min_support_families"]
    for r in rej:
        assert r["slot"] == "alice/home_city"
        assert r["pol"] == "pos" and r["value"] == "berlin"
        assert r["support_units"] == 1 and r["support_unit_ids"] == ["u1"]
        assert r["families"] == ["state"]
        assert "1" in r["reason"] and "2" in r["reason"]
        assert r["reason"]


def test_trace_family_gate_only(conn):
    """Two units of ONE family: the units gate passes, the families gate
    fails — exactly one rejection entry, named."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I live in Berlin.", occurred=200)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _state(conn, "u2", "alice/home_city", "Berlin", valid_from=200)
    st = _run(conn)
    (r,) = st["trace"]["rejected"]
    assert r["gate"] == "min_support_families"
    assert r["support_units"] == 2
    assert r["families"] == ["state"]
    assert "2" in r["reason"]


def test_trace_proof_pin_gate(conn):
    """Support that cannot lift its quote is dropped by the proof-pin
    gate — V7-14.01 keeps observations quote-pinned; the trace says so."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id,"
        " kind, recorded_at_us, occurred_start_us, occurred_end_us,"
        " byte_start, byte_end, generation)"
        " VALUES ('u2','src-missing',1,?, 'turn',300,300,300,"
        " NULL,NULL,?)",
        (SCOPE, GEN),
    )
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=300)
    st = _run(conn)
    (r,) = st["trace"]["rejected"]
    assert r["gate"] == "proof_pin"
    assert r["support_units"] == 2
    assert r["families"] == ["event", "state"]
    assert "unresolvable" in r["reason"]
    assert st["trace"]["skipped_unpinned"] == st["skipped_unpinned"] == 1
    assert _obs_rows(conn) == []


def test_trace_slot_read_gate(conn):
    """A fact table that exists but cannot be read drops the whole slot
    through the slot_read gate, named in the trace."""
    conn.execute("DROP TABLE entity_mentions")
    conn.execute("CREATE TABLE entity_mentions(x TEXT)")
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    st = _run(conn)
    assert st["skipped_read_errors"] == 1
    (r,) = st["trace"]["rejected"]
    assert r["gate"] == "slot_read"
    assert r["slot"] == "alice/home_city"
    assert "OperationalError" in r["reason"]
    (slot,) = st["trace"]["slots"]
    assert slot["error"] == "slot_read"


def test_trace_profiles_rejections(conn):
    """A slot with no live value group cannot materialize a profile —
    the profile gate names itself in the trace."""
    _unit(conn, "u1", "Alice does not like sushi.", occurred=100)
    _pref(conn, "u1", "alice", "sushi", polarity="negate", occurred=100)
    st = _run(conn)
    tr = st["trace"]
    assert tr["profiles"]["written"] == 0
    (pr,) = tr["profiles"]["rejected"]
    assert pr["gate"] == "profile_live"
    assert pr["subject"] == "alice"
    assert pr["slot"] == "preferences.dislikes"
    assert pr["reason"] == "no live value group"
    assert pr["deleted_prior"] is False
    assert _profile_rows(conn) == []


def test_trace_merged_count(conn):
    _seed_berlin_obs(conn)
    _unit(conn, "u3", "I live in Berlino.", occurred=300)
    _unit(conn, "u4", "I moved to Berlino.", occurred=400)
    _state(conn, "u3", "alice/home_city", "Berlino", valid_from=300)
    _event(conn, "e2", "u4", "alice", "move", "Berlino", occurred=400)
    st = _run(conn)
    assert st["trace"]["merged"] == st["merges"] == 1
    # berlino inserts, berlin updates (its contradict refs gain the
    # berlino group's units) — both count toward written.
    assert st["trace"]["written"] == 2
    assert st["trace"]["inserted"] == 1 and st["trace"]["updated"] == 1


def test_multi_fold_merge_unions_all_support(conn):
    """D8-33 (found by the V8-13.02 trace): three near-dup obs in one
    slot fold in ONE pass — the surviving row MUST union every dropped
    row's support, not just the last fold's. Pre-fix the second fold
    re-read the survivor's ORIGINAL in-memory refs and silently dropped
    the first fold's units."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I live in Berlino.", occurred=300)
    _unit(conn, "u4", "I moved to Berlino.", occurred=400)
    _unit(conn, "u5", "I live in Berlinos.", occurred=500)
    _unit(conn, "u6", "I moved to Berlinos.", occurred=600)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u3", "alice/home_city", "Berlino", valid_from=300)
    _event(conn, "e2", "u4", "alice", "move", "Berlino", occurred=400)
    _state(conn, "u5", "alice/home_city", "Berlinos", valid_from=500)
    _event(conn, "e3", "u6", "alice", "move", "Berlinos", occurred=600)
    st = _run(conn)
    assert st["merges"] == 2 and st["trace"]["merged"] == 2
    (o,) = _obs_rows(conn)
    assert o["text"] == "alice lives in berlin"
    assert o["proof_count"] == 6
    refs = json.loads(o["support_refs_json"])
    assert [r["unit_id"] for r in refs] == [
        "u1", "u2", "u3", "u4", "u5", "u6"]
    assert o["first_us"] == 100 and o["last_us"] == 600
    merges = conn.execute(
        "SELECT into_obs_id, from_obs_id FROM obs_merges_v7"
        " ORDER BY from_obs_id").fetchall()
    assert len(merges) == 2
    assert all(m[0] == o["obs_id"] for m in merges)


def test_trace_behavior_preserving_golden(conn):
    """V8-13.02 is observability-only: every artifact row for a
    multi-path fixture (write + merge + stale + profile + rejection +
    standing-dirty) keeps its exact pre-trace content."""
    _unit(conn, "u1", "I live in Berlin.", occurred=100)
    _unit(conn, "u2", "I moved to Berlin.", occurred=200)
    _unit(conn, "u3", "I live in Berlino.", occurred=300)
    _unit(conn, "u4", "I moved to Berlino.", occurred=400)
    _unit(conn, "u5", "I live in Paris.", occurred=500)
    _unit(conn, "u6", "I might move to Rome.", occurred=600)
    _unit(conn, "u7", "Alice does not like sushi.", occurred=700)
    _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
    _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)
    _state(conn, "u3", "alice/home_city", "Berlino", valid_from=300)
    _event(conn, "e2", "u4", "alice", "move", "Berlino", occurred=400)
    _state(conn, "u5", "alice/home_city", "Paris", valid_from=500)
    _event(conn, "e3", "u6", "alice", "move", "Rome",
           polarity="hypothetical", occurred=600)
    _pref(conn, "u7", "alice", "sushi", polarity="negate", occurred=700)
    _sq(conn, "sq1", filters={})
    st = _run(conn)

    # ---- observations_v7: one merged survivor ---------------------------
    (o,) = _obs_rows(conn)
    assert o["slot"] == "alice/home_city"
    assert o["text"] == "alice lives in berlin"
    assert o["producer"] == "consolidate_v7/t0"
    assert o["proof_count"] == 4
    assert o["stale"] == 1
    assert o["first_us"] == 100 and o["last_us"] == 400
    refs = json.loads(o["support_refs_json"])
    assert [(r["unit_id"], r["quote"], r["byte_start"], r["byte_end"],
             r["families"]) for r in refs] == [
        ("u1", "I live in Berlin.", 0, 17, ["state"]),
        ("u2", "I moved to Berlin.", 0, 18, ["event"]),
        ("u3", "I live in Berlino.", 0, 18, ["state"]),
        ("u4", "I moved to Berlino.", 0, 19, ["event"]),
    ]
    contra = json.loads(o["contradict_refs_json"])
    assert [(c["unit_id"], c["value"], c["pol"]) for c in contra] == [
        ("u5", "paris", "pos")]

    # ---- obs_merges_v7: the absorbed berlino row, prior contents intact -
    merges = conn.execute(
        "SELECT into_obs_id, from_obs_id, from_row_json, merged_run"
        " FROM obs_merges_v7").fetchall()
    assert len(merges) == 1
    assert merges[0][0] == o["obs_id"] and merges[0][3] == 1
    from_row = json.loads(merges[0][2])
    assert from_row["text"] == "alice lives in berlino"
    assert from_row["proof_count"] == 2
    assert from_row["stale"] == 1
    assert [r["unit_id"]
            for r in json.loads(from_row["support_refs_json"])] == [
                "u3", "u4"]

    # ---- profiles_v7: three live currents → disputed, latest value ------
    (p,) = _profile_rows(conn)
    assert p["subject_canon"] == "alice" and p["slot"] == "home_city"
    assert p["value"] == "Paris" and p["status"] == "disputed"
    assert p["updated_us"] == 600
    assert [r["unit_id"]
            for r in json.loads(p["support_refs_json"])] == ["u5"]

    # ---- cursor / seen / standing query ----------------------------------
    cur = conn.execute(
        "SELECT run_seq, last_processed, total_processed"
        " FROM consolidation_cursor_v7").fetchone()
    assert cur == (1, 7, 7)
    assert conn.execute(
        "SELECT COUNT(*) FROM consolidation_seen_v7").fetchone()[0] == 7
    assert conn.execute(
        "SELECT dirty FROM standing_queries WHERE sq_id='sq1'"
    ).fetchone()[0] == 1

    # ---- stats counters ---------------------------------------------------
    assert st["processed"] == 7 and st["observations_written"] == 2
    assert st["merges"] == 1 and st["stale_marked"] == 2
    assert st["profiles_written"] == 1 and st["standing_dirty"] == 1
    assert st["retired"] == []

    # ---- trace mirrors the same run --------------------------------------
    tr = st["trace"]
    assert tr["candidates"] == 5
    assert tr["accepted"] == 2 and tr["written"] == 2
    assert tr["inserted"] == 2 and tr["merged"] == 1
    gates = [r["gate"] for r in tr["rejected"]]
    assert gates == ["min_support_units", "min_support_families"] * 3
    assert tr["profiles"] == {
        "written": 1,
        "rejected": [{"subject": "alice", "slot": "preferences.dislikes",
                      "gate": "profile_live",
                      "reason": "no live value group",
                      "deleted_prior": False}]}
