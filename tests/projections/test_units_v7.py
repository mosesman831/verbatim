"""Tests for verbatim/projections/units_v7.py — SPEC_V7 V7-30.01,
V7-13.02–07, §30 `units`.

The determinism contract is the point: identical (sources row,
source_revisions row, add_args) inputs must reproduce byte-identical
unit rows, including unit_ids.
"""

from __future__ import annotations

import pytest

from verbatim.projections.units_v7 import (
    DEFAULT_SESSION_GAP_US,
    derive_units,
)

# -- row factories: the real persisted shapes ---------------------------
# sources:  source_id, origin, external_id, source_kind, scope_id,
#           speaker_id, created_us
# source_revisions: source_id, revision, payload, payload_hmac,
#           event_us, captured_us, timezone, provenance, metadata_json

T0 = 1_700_000_000_000_000  # arbitrary epoch µs


def src(source_id="src1", kind="user_message", speaker=None, scope="sc"):
    return {
        "source_id": source_id,
        "origin": "test",
        "external_id": None,
        "source_kind": kind,
        "scope_id": scope,
        "speaker_id": speaker,
        "created_us": T0,
    }


def rev(payload: bytes, source_id="src1", revision=1, event_us=T0, meta=None):
    return {
        "source_id": source_id,
        "revision": revision,
        "payload": payload,
        "payload_hmac": b"\x00" * 32,
        "event_us": event_us,
        "captured_us": event_us,
        "timezone": "UTC",
        "provenance": "direct_user",
        "metadata_json": meta if isinstance(meta, str) else "{}",
    }


def transcript(messages):
    """Render messages into a ``speaker: content``-per-line payload."""
    lines = []
    for m in messages:
        spk = m.get("speaker") or m.get("role") or "anon"
        lines.append(f"{spk}: {m['content']}")
    return "\n".join(lines).encode("utf-8")


def kinds(units):
    return [u["kind"] for u in units]


UNIT_COLS = {
    "unit_id",
    "source_id",
    "revision",
    "scope_id",
    "kind",
    "parent_unit_id",
    "session_id",
    "seq",
    "speaker_canon",
    "perspective",
    "recorded_at_us",
    "occurred_start_us",
    "occurred_end_us",
    "occurred_precision",
    "occurred_source",
    "byte_start",
    "byte_end",
    "generation",
}


def test_columns_match_section30():
    u = derive_units(src(), rev(b"Hello there."), {})
    assert set(u[0]) == UNIT_COLS


def test_single_turn_one_unit():
    payload = b"I love pizza."
    u = derive_units(src(), rev(payload), {})
    assert kinds(u) == ["turn"]
    r = u[0]
    assert (r["byte_start"], r["byte_end"]) == (0, len(payload))
    assert payload[r["byte_start"] : r["byte_end"]] == payload
    assert r["session_id"] is None
    assert r["parent_unit_id"] is None
    assert r["perspective"] == "user_stated"  # source_kind user_message


def test_single_turn_with_session_id_gets_session_unit():
    payload = b"I love pizza."
    u = derive_units(src(), rev(payload), {"session_id": "s-9"})
    # units_v7/v1.1: a single-turn session mints no byte-identical
    # container unit — the turn keeps session_id grouping metadata.
    assert kinds(u) == ["turn"]
    (turn,) = u
    assert turn["session_id"] == "s-9"
    assert turn["parent_unit_id"] is None


def test_messages_produce_turns_and_session():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "Hi there."},
        {"role": "assistant", "speaker": "assistant", "content": "Hello!"},
        {"role": "user", "speaker": "alice", "content": "How are you?"},
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s-1"})
    assert kinds(u) == ["session", "turn", "turn", "turn"]
    turns = [r for r in u if r["kind"] == "turn"]
    assert [t["seq"] for t in turns] == [0, 1, 2]
    for t, m in zip(turns, msgs):
        assert payload[t["byte_start"] : t["byte_end"]] == m["content"].encode()
        assert t["session_id"] == "s-1"
    assert turns[0]["perspective"] == "user_stated"
    assert turns[1]["perspective"] == "agent_stated"
    # session unit pins cover first-to-last member bytes
    sess = u[0]
    assert sess["byte_start"] == turns[0]["byte_start"]
    assert sess["byte_end"] == turns[-1]["byte_end"]


def test_long_document_sentence_windows_slice_exactly():
    doc = (
        "S one here. S two here. S three here. S four here. "
        "S five here. S six here. S seven here. S eight done."
    )
    payload = doc.encode()
    u = derive_units(src(kind="import"), rev(payload), {})
    assert kinds(u) == ["turn", "sentence_window", "sentence_window", "sentence_window"]
    turn = u[0]
    wins = u[1:]
    for w in wins:
        assert w["parent_unit_id"] == turn["unit_id"]
        assert payload[w["byte_start"] : w["byte_end"]].startswith(b"S")
        assert payload[w["byte_start"] : w["byte_end"]].endswith(b".")
    # windows are disjoint and tile the document
    assert wins[0]["byte_start"] == 0
    assert wins[-1]["byte_end"] == len(payload)
    for a, b in zip(wins, wins[1:]):
        assert a["byte_end"] < b["byte_start"]


def test_long_turn_gets_windows():
    content = "A one. B two. C three. D four. E five. F six. G seven."
    msgs = [{"role": "user", "speaker": "alice", "content": content}]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s"})
    wins = [r for r in u if r["kind"] == "sentence_window"]
    turn = [r for r in u if r["kind"] == "turn"][0]
    # 7 sentences -> windows of 3, 3, 1
    assert len(wins) == 3
    for w in wins:
        assert w["parent_unit_id"] == turn["unit_id"]
        assert w["session_id"] == "s"
        assert payload[w["byte_start"] : w["byte_end"]].decode().count(".") <= 3


def test_short_document_stays_whole():
    doc = "Only two sentences. Right here."
    u = derive_units(src(kind="import"), rev(doc.encode()), {})
    assert kinds(u) == ["turn"]  # <=3 sentences: no windows


def test_multi_session_grouping():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "one", "session_id": "s-a"},
        {"role": "user", "speaker": "alice", "content": "two", "session_id": "s-a"},
        {"role": "user", "speaker": "alice", "content": "three", "session_id": "s-b"},
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs})
    sess = [r for r in u if r["kind"] == "session"]
    turns = [r for r in u if r["kind"] == "turn"]
    # s-b has a single member — no session twin (units_v7/v1.1); the
    # turn still carries its session_id for grouping.
    assert {s["session_id"] for s in sess} == {"s-a"}
    assert [t["session_id"] for t in turns] == ["s-a", "s-a", "s-b"]


def test_gap_breaks_session():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "one", "at": T0},
        {
            "role": "user",
            "speaker": "alice",
            "content": "two",
            "at": T0 + 600_000_000,
        },
        {
            "role": "user",
            "speaker": "alice",
            "content": "three",
            "at": T0 + 600_000_000 + DEFAULT_SESSION_GAP_US + 1,
        },
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs})
    sess = [r for r in u if r["kind"] == "session"]
    turns = [r for r in u if r["kind"] == "turn"]
    # The second implicit session has a single member — suppressed under
    # units_v7/v1.1 (multi-member sessions still mint container units).
    assert len(sess) == 1
    assert turns[0]["session_id"] == turns[1]["session_id"]
    assert turns[0]["session_id"] == sess[0]["session_id"]
    assert turns[2]["session_id"] != sess[0]["session_id"]


def test_explicit_session_id_wins_over_gap():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "one", "at": T0},
        {
            "role": "user",
            "speaker": "alice",
            "content": "two",
            "at": T0 + 10 * DEFAULT_SESSION_GAP_US,
        },
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "fixed"})
    sess = [r for r in u if r["kind"] == "session"]
    assert len(sess) == 1 and sess[0]["session_id"] == "fixed"
    assert all(t["session_id"] == "fixed" for t in u if t["kind"] == "turn")


def test_episode_segmentation_stable():
    # cat turns share {cat, mochi, sleeps, ...}; quantum turns share
    # {quantum, computing, ...}; the topic shift is the cohesion valley.
    msgs = [
        {
            "role": "user",
            "speaker": "alice",
            "content": "My cat Mochi sleeps all day.",
        },
        {
            "role": "assistant",
            "speaker": "bot",
            "content": "Cats sleep a lot during the day.",
        },
        {
            "role": "user",
            "speaker": "alice",
            "content": "The cat Mochi eats tuna every day.",
        },
        {
            "role": "user",
            "speaker": "alice",
            "content": "Quantum computing fascinates me lately.",
        },
        {
            "role": "assistant",
            "speaker": "bot",
            "content": "Quantum computing and qubits fascinate researchers.",
        },
    ]
    payload = transcript(msgs)
    args = {"messages": msgs, "session_id": "s-e"}
    u = derive_units(src(), rev(payload), args)
    eps = [r for r in u if r["kind"] == "episode"]
    turns = [r for r in u if r["kind"] == "turn"]
    assert len(eps) == 2  # cat episode, then quantum episode
    # first episode covers turns 0-2, second covers 3-4
    assert eps[0]["byte_start"] == turns[0]["byte_start"]
    assert eps[0]["byte_end"] == turns[2]["byte_end"]
    assert eps[1]["byte_start"] == turns[3]["byte_start"]
    assert all(e["parent_unit_id"] == u[0]["unit_id"] for e in eps)
    # turns re-parent to their episode
    assert turns[0]["parent_unit_id"] == eps[0]["unit_id"]
    assert turns[4]["parent_unit_id"] == eps[1]["unit_id"]
    # boundaries are stable across re-derivation
    u2 = derive_units(src(), rev(payload), args)
    assert [(e["byte_start"], e["byte_end"]) for e in u2 if e["kind"] == "episode"] == [
        (e["byte_start"], e["byte_end"]) for e in eps
    ]


def test_cohesive_session_no_episodes():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "The cat sleeps."},
        {"role": "assistant", "speaker": "bot", "content": "The cat naps often."},
        {"role": "user", "speaker": "alice", "content": "The cat dreams of fish."},
        {"role": "assistant", "speaker": "bot", "content": "The cat surely does."},
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s"})
    assert "episode" not in kinds(u)


def test_byte_identical_rederivation():
    msgs = [
        {
            "role": "user",
            "speaker": "alice",
            "content": "First. Second. Third. Fourth.",
            "at": T0,
        },
        {"role": "assistant", "speaker": "bot", "content": "Reply here.", "at": T0 + 1},
    ]
    payload = transcript(msgs)
    args = {"messages": msgs, "session_id": "s", "generation": 7}
    a = derive_units(src(), rev(payload), args)
    b = derive_units(
        dict(src()),
        dict(rev(payload)),
        {
            "messages": [dict(m) for m in msgs],
            "session_id": "s",
            "generation": 7,
        },
    )
    assert a == b
    assert [r["unit_id"] for r in a] == [r["unit_id"] for r in b]


def test_unit_ids_unique_and_content_addressed():
    payload = b"yes yes"
    msgs = [
        {"role": "user", "speaker": "alice", "content": "yes"},
        {"role": "user", "speaker": "alice", "content": "yes"},
    ]
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s"})
    ids = [r["unit_id"] for r in u]
    assert len(ids) == len(set(ids))  # duplicate content -> distinct ids
    # different inputs -> different ids
    u2 = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "other"})
    assert ids != [r["unit_id"] for r in u2]


_ALICE_HI = {"role": "user", "speaker": "alice", "content": "hi"}


@pytest.mark.parametrize(
    "source_kind,msg,persp",
    [
        ("user_message", _ALICE_HI, "user_stated"),
        (
            "user_message",
            {"role": "assistant", "speaker": "bot", "content": "hi"},
            "agent_stated",
        ),
        (
            "user_message",
            {"role": "tool_call", "speaker": "alice", "content": "x"},
            "agent_action",
        ),
        (
            "tool_output",
            {"role": "user", "speaker": "alice", "content": "x"},
            "agent_action",
        ),
        (
            "user_message",
            {"role": "system", "speaker": "sys", "content": "x"},
            "system",
        ),
        (
            "user_message",
            {"role": "user", "speaker": "mom", "content": "x"},
            "user_stated",
        ),
        ("import", {"content": "doc text"}, "document"),
        ("operator_record", {"content": "op note"}, "system"),
    ],
)
def test_perspective_classification(source_kind, msg, persp):
    payload = (msg.get("content") or "").encode()
    u = derive_units(
        src(kind=source_kind),
        rev(payload),
        {"messages": [msg]},
    )
    assert u[0]["perspective"] == persp


def test_perspective_declared_identities():
    # a named speaker matching the declared user -> user_stated; another
    # named speaker -> third_party
    msgs = [
        {"speaker": "Caroline", "content": "I went biking."},
        {"speaker": "Melanie", "content": "Nice!"},
    ]
    payload = transcript(msgs)
    u = derive_units(
        src(), rev(payload), {"messages": msgs, "session_id": "s", "user": "Caroline"}
    )
    turns = [r for r in u if r["kind"] == "turn"]
    assert turns[0]["perspective"] == "user_stated"
    assert turns[1]["perspective"] == "third_party"
    assert turns[0]["speaker_canon"] == "caroline"


def test_recorded_at_and_occurred_passthrough():
    msgs = [
        {
            "role": "user",
            "speaker": "alice",
            "content": "I hiked yesterday.",
            "message_at": "2023-06-04T10:00:00Z",
            "occurred": {
                "start_us": 1,
                "end_us": 2,
                "precision": "day",
                "source": "explicit",
            },
        }
    ]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s"})
    turn = [r for r in u if r["kind"] == "turn"][0]
    assert turn["recorded_at_us"] == 1685872800000000
    assert turn["occurred_start_us"] == 1
    assert turn["occurred_end_us"] == 2
    assert turn["occurred_precision"] == "day"
    assert turn["occurred_source"] == "explicit"


def test_unpinned_message_is_honest_null():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "in payload"},
        {"role": "user", "speaker": "alice", "content": "NOT IN PAYLOAD"},
    ]
    payload = b"alice: in payload"
    u = derive_units(src(), rev(payload), {"messages": msgs, "session_id": "s"})
    turns = [r for r in u if r["kind"] == "turn"]
    assert turns[0]["byte_start"] is not None
    assert turns[1]["byte_start"] is None and turns[1]["byte_end"] is None
    # session still covers the pinned member
    sess = [r for r in u if r["kind"] == "session"][0]
    assert sess["byte_start"] == turns[0]["byte_start"]


def test_when_metadata_parses_to_occurred():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "met for coffee"},
    ]
    payload = transcript(msgs)
    u = derive_units(
        src(), rev(payload),
        {"messages": msgs, "when": "10:04 am on 19 December, 2023"},
    )
    turn = [r for r in u if r["kind"] == "turn"][0]
    # "10:04 am on 19 December, 2023" → 2023-12-19T10:04:00Z instant
    assert turn["occurred_start_us"] == 1702980240_000_000
    assert turn["occurred_end_us"] == 1702980240_000_000
    assert turn["occurred_precision"] == "instant"
    assert turn["occurred_source"] == "explicit"


def test_when_metadata_day_precision():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "met for coffee"},
    ]
    payload = transcript(msgs)
    u = derive_units(
        src(), rev(payload),
        {"messages": msgs, "when": "8 May, 2023"},
    )
    turn = [r for r in u if r["kind"] == "turn"][0]
    # "8 May, 2023" → day-precision half-open interval [day, day+1)
    assert turn["occurred_start_us"] == 1683504000_000_000
    assert turn["occurred_end_us"] == 1683504000_000_000 + 86_400_000_000
    assert turn["occurred_precision"] == "day"
    assert turn["occurred_source"] == "explicit"


def test_when_metadata_unparseable_stays_unknown():
    msgs = [
        {"role": "user", "speaker": "alice", "content": "met for coffee"},
    ]
    payload = transcript(msgs)
    u = derive_units(
        src(), rev(payload),
        {"messages": msgs, "when": "sometime last spring"},
    )
    turn = [r for r in u if r["kind"] == "turn"][0]
    assert turn["occurred_start_us"] is None
    assert turn["occurred_end_us"] is None
    assert turn["occurred_precision"] == "unknown"


def test_utf8_pins_exact():
    content = "J'adore le café — très bon. Encore une phrase."
    msgs = [{"role": "user", "speaker": "alice", "content": content}]
    payload = transcript(msgs)
    u = derive_units(src(), rev(payload), {"messages": msgs})
    t = u[0]
    assert payload[t["byte_start"] : t["byte_end"]] == content.encode("utf-8")


def test_segmentation_modes():
    msgs = [
        {"role": "user", "speaker": "a", "content": "x"},
        {"role": "user", "speaker": "a", "content": "y"},
    ]
    payload = transcript(msgs)
    args = {"messages": msgs, "session_id": "s"}
    assert set(
        kinds(derive_units(src(), rev(payload), args, segmentation="turn"))
    ) == {
        "session",
        "turn",
    }
    assert set(
        kinds(derive_units(src(), rev(payload), args, segmentation="raw"))
    ) == {"turn"}
    doc = derive_units(src(), rev(payload), args, segmentation="document")
    # document mode forces whole-payload treatment: one turn covering
    # [0, len) regardless of the messages arg (a session unit still
    # appears because session_id was given).
    doc_turns = [r for r in doc if r["kind"] == "turn"]
    assert len(doc_turns) == 1
    assert (doc_turns[0]["byte_start"], doc_turns[0]["byte_end"]) == (0, len(payload))
    with pytest.raises(Exception):
        derive_units(src(), rev(payload), args, segmentation="bogus")


def test_canon_fn_used():
    msgs = [{"role": "user", "speaker": "Alice", "content": "hi"}]
    payload = transcript(msgs)
    u = derive_units(
        src(), rev(payload), {"messages": msgs}, canon_fn=lambda s: "C(" + s + ")"
    )
    assert u[0]["speaker_canon"] == "C(Alice)"
    # default fallback folds (entities_v2 may be absent in wave A)
    u2 = derive_units(src(), rev(payload), {"messages": msgs})
    assert u2[0]["speaker_canon"] in ("alice", "Alice")


def test_real_persisted_rows(store):
    """derive_units consumes actual sources/source_revisions rows."""
    from verbatim.storage.repos import SourcesRepo

    from .conftest import add_source, scope_row

    payload = b"Remember this fact."
    with store.tx() as conn:
        scope_row(conn, "sc-real")
        add_source(conn, "src-real", "sc-real", payload, speaker="user")

    repo = SourcesRepo(store)
    srow = repo.get("src-real")
    rrow = repo.get_revision("src-real", 1)
    rrow["payload"] = repo.payload("src-real", 1)
    assert isinstance(srow, dict) and isinstance(rrow["payload"], bytes)

    u = derive_units(srow, rrow, {"session_id": "s-real"})
    # units_v7/v1.1: the single-member session mints no twin row.
    assert kinds(u) == ["turn"]
    t = u[0]
    assert t["session_id"] == "s-real"
    assert payload[t["byte_start"] : t["byte_end"]] == payload
    assert t["scope_id"] == "sc-real"
    assert t["perspective"] == "user_stated"
    assert t["recorded_at_us"] == 1  # revision event_us fallback

    # metadata_json persisted on the revision is honored as add-args.
    with store.tx() as conn:
        conn.execute(
            "UPDATE source_revisions SET metadata_json=? WHERE source_id='src-real'",
            ('{"session_id": "from-meta"}',),
        )
    rrow = repo.get_revision("src-real", 1)
    rrow["payload"] = repo.payload("src-real", 1)
    u = derive_units(srow, rrow, {})
    assert u[0]["session_id"] == "from-meta"


def test_requires_payload():
    from verbatim.core.types import VerbatimError

    with pytest.raises(VerbatimError):
        derive_units(src(), {"source_id": "src1", "revision": 1}, {})
