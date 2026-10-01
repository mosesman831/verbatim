"""V7 turn-capture tests — ``Memory.add`` conversational fields + ``bulk_add``
(D7-26).

Real on-disk ``Store`` per test (``tmp_path``), real ingest, real
``source_revisions.metadata_json``, real ``projections/units_v7.derive_units``
over the persisted rows — the payoff is that turn fields survive the wire and
drive the units projection verbatim. ``worker="external"`` keeps the lifecycle
deterministic; no kernel mocks.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from verbatim import Memory
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.memory.types import Acceptance, AddResult
from verbatim.projections.units_v7 import derive_units


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "m.db")


@pytest.fixture()
def mem(path):
    m = Memory(path=path, worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# store-read helpers
# ---------------------------------------------------------------------------

_SRC_COLS = (
    "source_id", "origin", "external_id", "source_kind", "scope_id",
    "speaker_id", "created_us",
)
_REV_COLS = (
    "source_id", "revision", "payload", "event_us", "captured_us",
    "timezone", "provenance", "metadata_json",
)


def _source_row(mem, source_id):
    with mem._store.read() as conn:
        row = conn.execute(
            f"SELECT {', '.join(_SRC_COLS)} FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
    assert row is not None
    return dict(zip(_SRC_COLS, row))


def _revision_row(mem, source_id, revision=1):
    with mem._store.read() as conn:
        row = conn.execute(
            f"SELECT {', '.join(_REV_COLS)} FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchone()
    assert row is not None
    return dict(zip(_REV_COLS, row))


def _revision_meta(mem, source_id, revision=1):
    return json.loads(_revision_row(mem, source_id, revision)["metadata_json"])


def _units(mem, source_id):
    """The real projection: derive_units over the persisted rows — the
    metadata_json channel is the proof that add-args survived the wire."""
    return derive_units(
        _source_row(mem, source_id), _revision_row(mem, source_id), {}
    )


def _source_count(mem):
    with mem._store.read() as conn:
        return conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]


US_0 = 1_789_700_000_000_000  # 2026-09-17T…Z — fixed µs instant
US_1 = US_0 + 5_000_000
US_2 = US_0 + 10_000_000


# ---------------------------------------------------------------------------
# turn fields persist into metadata_json
# ---------------------------------------------------------------------------


class TestTurnFieldsPersist:
    def test_messages_persist_verbatim(self, mem):
        r = mem.add(
            "user: hi\nassistant: hello there",
            messages=[
                {"speaker": "user", "text": "hi", "at": "2026-09-17T10:00:00Z"},
                {"speaker": "assistant", "text": "hello there", "at": US_1},
            ],
        )
        meta = _revision_meta(mem, r.memory_id)
        assert meta["messages"] == [
            {"speaker": "user", "text": "hi", "at": "2026-09-17T10:00:00Z"},
            {"speaker": "assistant", "text": "hello there", "at": US_1},
        ]
        # the envelope's own provenance markers still land
        assert meta["direct_add"] is True
        assert meta["facade"] == "memory/v5"

    def test_speaker_and_session_persist(self, mem):
        r = mem.add("single turn", speaker="alice", session_id="conv-9")
        meta = _revision_meta(mem, r.memory_id)
        assert meta["speaker"] == "alice"
        assert meta["session_id"] == "conv-9"

    def test_occurred_at_int_us_persists(self, mem):
        r = mem.add("happened then", occurred_at=US_0)
        meta = _revision_meta(mem, r.memory_id)
        assert meta["message_at"] == US_0
        assert meta["occurred"] == {
            "start_us": US_0,
            "end_us": US_0,
            "precision": "instant",
            "source": "explicit",
        }

    def test_occurred_at_rfc3339_normalizes(self, mem):
        r = mem.add("ts str", occurred_at="2026-09-17T10:00:00Z")
        meta = _revision_meta(mem, r.memory_id)
        assert meta["message_at"] == int(
            datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()
            * 1_000_000
        )

    def test_occurred_at_datetime_normalizes(self, mem):
        dt = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        r = mem.add("ts dt", occurred_at=dt)
        meta = _revision_meta(mem, r.memory_id)
        assert meta["message_at"] == int(dt.timestamp() * 1_000_000)
        # naive datetime reads as UTC (same convention as units_v7)
        dt2 = datetime(2026, 9, 17, 10, 0, 0)
        r2 = mem.add("ts dt naive", occurred_at=dt2)
        assert _revision_meta(mem, r2.memory_id)["message_at"] == int(
            dt.timestamp() * 1_000_000
        )

    def test_explicit_turn_params_beat_metadata(self, mem):
        """Explicit params win over same-named caller-metadata keys — the
        precedence derive_units itself gives call args over persisted."""
        r = mem.add(
            "x",
            metadata={"speaker": "meta-speaker", "custom": "kept"},
            speaker="param-speaker",
        )
        meta = _revision_meta(mem, r.memory_id)
        assert meta["speaker"] == "param-speaker"
        assert meta["custom"] == "kept"

    def test_caller_metadata_messages_channel_still_works(self, mem):
        """metadata["messages"] alone (no param) persists through the raw
        channel — bounded by the caller-metadata contract."""
        r = mem.add(
            "a: one\nb: two",
            metadata={"messages": [{"speaker": "a", "text": "one"},
                                   {"speaker": "b", "text": "two"}]},
        )
        meta = _revision_meta(mem, r.memory_id)
        assert [m["speaker"] for m in meta["messages"]] == ["a", "b"]
        units = _units(mem, r.memory_id)
        assert [u["kind"] for u in units].count("turn") == 2


# ---------------------------------------------------------------------------
# derive_units over persisted rows — the payoff
# ---------------------------------------------------------------------------


class TestDeriveUnitsIntegration:
    def test_turn_units_carry_speaker_and_time(self, mem):
        r = mem.add(
            "user: when is the deploy?\nassistant: Tuesday at 3pm.",
            messages=[
                {"speaker": "user", "text": "when is the deploy?",
                 "at": US_0},
                {"speaker": "assistant", "text": "Tuesday at 3pm.",
                 "at": US_1},
            ],
            session_id="conv-42",
        )
        units = _units(mem, r.memory_id)
        turns = [u for u in units if u["kind"] == "turn"]
        assert len(turns) == 2
        assert [t["speaker_canon"] for t in turns] == ["user", "assistant"]
        assert [t["recorded_at_us"] for t in turns] == [US_0, US_1]
        # byte pins locate each message inside the payload
        payload = b"user: when is the deploy?\nassistant: Tuesday at 3pm."
        for t in turns:
            assert t["byte_start"] is not None
            assert payload[t["byte_start"]:t["byte_end"]] in (
                b"when is the deploy?",
                b"Tuesday at 3pm.",
            )

    def test_session_unit_forms_from_session_id(self, mem):
        r = mem.add(
            "user: a\nuser: b",
            messages=[
                {"speaker": "user", "text": "a", "at": US_0},
                {"speaker": "user", "text": "b", "at": US_1},
            ],
            session_id="sess-X",
        )
        units = _units(mem, r.memory_id)
        sessions = [u for u in units if u["kind"] == "session"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-X"
        assert all(
            u["session_id"] == "sess-X" for u in units if u["kind"] == "turn"
        )

    def test_occurred_lands_on_turns(self, mem):
        r = mem.add(
            "user: hello\nassistant: hi",
            messages=[
                {"speaker": "user", "text": "hello"},
                {"speaker": "assistant", "text": "hi"},
            ],
            occurred_at=US_0,
        )
        units = _units(mem, r.memory_id)
        turns = [u for u in units if u["kind"] == "turn"]
        assert turns
        for t in turns:
            # top-level occurred applies to turns lacking their own
            assert t["occurred_start_us"] == US_0
            assert t["occurred_end_us"] == US_0
            assert t["occurred_precision"] == "instant"
            assert t["occurred_source"] == "explicit"
            # occurred_at also feeds the recorded-time lane
            assert t["recorded_at_us"] == US_0

    def test_per_message_at_beats_top_level(self, mem):
        r = mem.add(
            "u: one\nu: two",
            messages=[
                {"speaker": "user", "text": "one", "at": US_1},
                {"speaker": "user", "text": "two", "at": US_2},
            ],
            occurred_at=US_0,
        )
        turns = [u for u in _units(mem, r.memory_id) if u["kind"] == "turn"]
        assert [t["recorded_at_us"] for t in turns] == [US_1, US_2]

    def test_identical_text_turns_distinct_units(self, mem):
        """The repeated-turn defect: identical texts in one session mint
        distinct units (seq + byte pins differ), never collapse."""
        r = mem.add(
            "user: yes\nassistant: ok\nuser: yes",
            messages=[
                {"speaker": "user", "text": "yes", "at": US_0},
                {"speaker": "assistant", "text": "ok", "at": US_1},
                {"speaker": "user", "text": "yes", "at": US_2},
            ],
            session_id="conv-yes",
        )
        turns = [u for u in _units(mem, r.memory_id) if u["kind"] == "turn"]
        assert len(turns) == 3
        assert len({t["unit_id"] for t in turns}) == 3
        yes = [t for t in turns if t["speaker_canon"] == "user"]
        assert len(yes) == 2
        assert yes[0]["byte_start"] != yes[1]["byte_start"]

    def test_same_text_different_time_distinct_sources(self, mem):
        """Two identical-text turns in one session mint distinct sources
        when their declared times differ (turn-aware dedup material)."""
        r1 = mem.add(
            "yes",
            messages=[{"speaker": "user", "text": "yes", "at": US_0}],
            session_id="conv-7",
        )
        r2 = mem.add(
            "yes",
            messages=[{"speaker": "user", "text": "yes", "at": US_2}],
            session_id="conv-7",
        )
        assert r1.memory_id != r2.memory_id
        assert r2.replayed is False

    def test_different_session_distinct_sources(self, mem):
        r1 = mem.add(
            "yes",
            messages=[{"speaker": "user", "text": "yes", "at": US_0}],
            session_id="conv-7",
        )
        r2 = mem.add(
            "yes",
            messages=[{"speaker": "user", "text": "yes", "at": US_0}],
            session_id="conv-8",
        )
        assert r1.memory_id != r2.memory_id

    def test_byte_identical_turn_retry_still_dedups(self, mem):
        """Identical in every declared field = the same logical turn —
        V3-12.04 retry dedup is preserved for the messages path."""
        kwargs = {
            "messages": [{"speaker": "user", "text": "yes", "at": US_0}],
            "session_id": "conv-7",
        }
        r1 = mem.add("yes", **kwargs)
        r2 = mem.add("yes", **kwargs)
        assert r2.memory_id == r1.memory_id
        assert r2.replayed is True
        assert _source_count(mem) == 1

    def test_speakerless_add_falls_back_cleanly(self, mem):
        r = mem.add("bare turn")
        units = _units(mem, r.memory_id)
        assert len(units) == 1
        assert units[0]["kind"] == "turn"
        # speaker falls back to sources.speaker_id (the capture principal)
        assert units[0]["speaker_canon"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestTurnValidation:
    def test_messages_over_limit(self, mem):
        with pytest.raises(VerbatimError) as e:
            mem.add("x", messages=["m"] * 513)
        assert e.value.code == ErrorCode.VALIDATION
        mem.add("x", messages=["m"] * 512)  # boundary accepts

    def test_message_text_over_limit(self, mem):
        big = "t" * (262_144 + 1)
        with pytest.raises(VerbatimError) as e:
            mem.add("x", messages=[{"speaker": "u", "text": big}])
        assert e.value.code == ErrorCode.VALIDATION

    def test_message_string_entry_over_limit(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", messages=["t" * (262_144 + 1)])

    def test_message_bad_entry_type(self, mem):
        for bad in (123, ["nested"], None):
            with pytest.raises(VerbatimError) as e:
                mem.add("x", messages=[bad])
            assert e.value.code == ErrorCode.VALIDATION

    def test_message_bad_text_type(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", messages=[{"text": 42}])

    def test_speaker_must_be_str(self, mem):
        for bad in (123, ["u"], {"s": 1}):
            with pytest.raises(VerbatimError) as e:
                mem.add("x", speaker=bad)
            assert e.value.code == ErrorCode.VALIDATION

    def test_speaker_over_limit(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", speaker="s" * 129)
        mem.add("x", speaker="s" * 128)  # boundary accepts

    def test_speaker_nonprintable(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", speaker="bad\x00speaker")
        with pytest.raises(VerbatimError):
            mem.add("x", speaker="  ")

    def test_per_message_speaker_validated(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", messages=[{"speaker": "a" * 129, "text": "t"}])
        with pytest.raises(VerbatimError):
            mem.add("x", messages=[{"speaker": 7, "text": "t"}])

    def test_occurred_at_bad_values(self, mem):
        for bad in ("not a time", [1, 2], {"x": 1}, 1.5, True):
            with pytest.raises(VerbatimError) as e:
                mem.add("x", occurred_at=bad)
            assert e.value.code == ErrorCode.VALIDATION

    def test_per_message_at_bad_value(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", messages=[{"text": "t", "at": "garbage"}])
        with pytest.raises(VerbatimError):
            mem.add("x", messages=[{"text": "t", "at": 1.5}])

    def test_messages_not_a_list(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", messages=42)

    def test_session_id_validation(self, mem):
        with pytest.raises(VerbatimError):
            mem.add("x", session_id="s" * 257)
        with pytest.raises(VerbatimError):
            mem.add("x", session_id=99)

    def test_message_datetime_at_normalizes(self, mem):
        dt = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        r = mem.add("u: hi", messages=[{"speaker": "u", "text": "hi", "at": dt}])
        meta = _revision_meta(mem, r.memory_id)
        # datetimes are not JSON — they must land as µs ints
        assert meta["messages"][0]["at"] == int(dt.timestamp() * 1_000_000)


# ---------------------------------------------------------------------------
# legacy add unchanged
# ---------------------------------------------------------------------------


class TestLegacyAddUnchanged:
    def test_plain_add_result_shape(self, mem):
        r = mem.add("the wifi password is swordfish")
        assert isinstance(r, AddResult)
        assert r.memory_id and r.receipt_id and r.ref
        assert r.acceptance == Acceptance.ACCEPTED.value
        assert r.error is None

    def test_plain_add_dedup_replays(self, mem):
        """Legacy content-only dedup material is byte-identical — a
        byte-identical second add replays the committed receipt."""
        r1 = mem.add("same bytes")
        r2 = mem.add("same bytes")
        assert r2.memory_id == r1.memory_id
        assert r2.replayed is True

    def test_plain_add_leaves_no_turn_keys(self, mem):
        r = mem.add("plain", metadata={"custom": 1})
        meta = _revision_meta(mem, r.memory_id)
        assert "messages" not in meta
        assert "speaker" not in meta
        assert "occurred" not in meta
        assert "message_at" not in meta
        assert meta["custom"] == 1

    def test_idempotency_conflict_still_typed(self, mem):
        mem.add("a", idempotency_key="k1")
        with pytest.raises(VerbatimError) as e:
            mem.add("different", idempotency_key="k1")
        assert e.value.code == ErrorCode.OPERATION_CONFLICT

    def test_idem_conflict_detects_turn_diff(self, mem):
        """Same key + same content but different turn args → conflict,
        never a silent replay of the other speaker's record."""
        mem.add("yes", idempotency_key="k2", speaker="alice")
        with pytest.raises(VerbatimError) as e:
            mem.add("yes", idempotency_key="k2", speaker="bob")
        assert e.value.code == ErrorCode.OPERATION_CONFLICT
        r = mem.add("yes", idempotency_key="k2", speaker="alice")
        assert r.replayed is True

    def test_turn_add_idempotent_replay(self, mem):
        kw = {
            "messages": [{"speaker": "u", "text": "hi", "at": US_0}],
            "session_id": "s1",
        }
        r1 = mem.add("u: hi", idempotency_key="turnk", **kw)
        r2 = mem.add("u: hi", idempotency_key="turnk", **kw)
        assert r2.replayed is True
        assert r2.memory_id == r1.memory_id


# ---------------------------------------------------------------------------
# bulk_add
# ---------------------------------------------------------------------------


class TestBulkAdd:
    def test_bulk_lands_all_in_order(self, mem):
        items = [
            {"content": f"bulk item {i}", "idempotency_key": f"b-{i}"}
            for i in range(200)
        ]
        res = mem.bulk_add(items)
        assert len(res) == 200
        assert all(isinstance(r, AddResult) for r in res)
        assert all(r.error is None for r in res)
        assert all(r.acceptance == Acceptance.ACCEPTED.value for r in res)
        # order: results[i] corresponds to items[i] — replaying the same
        # key must land on the same memory_id recorded for position i.
        res2 = mem.bulk_add(items)
        assert [r.memory_id for r in res2] == [r.memory_id for r in res]
        assert all(r.replayed for r in res2)

    def test_bulk_per_item_failure_is_honest(self, mem):
        before = _source_count(mem)
        res = mem.bulk_add(
            [
                {"content": "good-0"},
                {"content": 12345},  # invalid payload
                {"content": "good-2"},
                "not-a-dict",  # invalid item
                {"content": "good-4", "nope": 1},  # unknown kwarg
            ]
        )
        assert len(res) == 5
        ok = [0, 2]
        bad = [1, 3, 4]
        for i in ok:
            assert res[i].error is None
            assert res[i].acceptance == Acceptance.ACCEPTED.value
            assert res[i].memory_id
        for i in bad:
            assert res[i].acceptance == Acceptance.FAILED.value
            assert res[i].error
            assert res[i].memory_id == ""
        # only the good items committed — failure is per-item, never the
        # batch and never a phantom success.
        assert _source_count(mem) == before + len(ok)

    def test_bulk_in_tx_failure_rolls_back_only_itself(self, mem):
        """A mid-tx item failure (idempotency conflict) rolls back its own
        record via savepoint; neighbours in the same chunk still commit."""
        mem.add("seeded", idempotency_key="dup")
        res = mem.bulk_add(
            [
                {"content": "c0"},
                {"content": "conflicting", "idempotency_key": "dup"},
                {"content": "c2"},
            ],
            chunk_size=64,
        )
        assert res[0].acceptance == Acceptance.ACCEPTED.value
        assert res[1].acceptance == Acceptance.FAILED.value
        assert "OPERATION_CONFLICT" in res[1].error
        assert res[2].acceptance == Acceptance.ACCEPTED.value

    def test_bulk_turn_items_derive_units(self, mem):
        res = mem.bulk_add(
            [
                {
                    "content": "user: q?\nassistant: a!",
                    "messages": [
                        {"speaker": "user", "text": "q?", "at": US_0},
                        {"speaker": "assistant", "text": "a!", "at": US_1},
                    ],
                    "session_id": "bulk-conv",
                },
                {"content": "plain item"},
            ]
        )
        assert all(r.error is None for r in res)
        units = _units(mem, res[0].memory_id)
        turns = [u for u in units if u["kind"] == "turn"]
        assert len(turns) == 2
        assert {u["session_id"] for u in units} == {"bulk-conv"}
        # the plain item stayed on the legacy single-turn path
        u2 = _units(mem, res[1].memory_id)
        assert len(u2) == 1 and u2[0]["kind"] == "turn"

    def test_bulk_dedup_within_batch(self, mem):
        res = mem.bulk_add(
            [{"content": "identical"}, {"content": "identical"}]
        )
        assert res[0].memory_id == res[1].memory_id
        assert res[1].replayed is True
        assert _source_count(mem) == 1

    def test_bulk_empty(self, mem):
        assert mem.bulk_add([]) == []

    def test_bulk_rejects_bad_items_container(self, mem):
        for bad in ("a string", b"bytes", {"content": "x"}, 42, None):
            with pytest.raises(VerbatimError) as e:
                mem.bulk_add(bad)
            assert e.value.code == ErrorCode.VALIDATION

    def test_bulk_chunk_size_validation(self, mem):
        for bad in (0, -1, 1.5, "64", True, 513):
            with pytest.raises(VerbatimError):
                mem.bulk_add([{"content": "x"}], chunk_size=bad)

    def test_bulk_chunk_boundaries(self, mem):
        """chunk_size=3 forces multi-chunk commits; failures on chunk
        boundaries still isolate per item."""
        items = [{"content": f"cc-{i}"} for i in range(10)]
        items.insert(5, {"content": object()})  # invalid mid-batch
        res = mem.bulk_add(items, chunk_size=3)
        assert len(res) == 11
        assert res[5].acceptance == Acceptance.FAILED.value
        assert sum(
            r.acceptance == Acceptance.ACCEPTED.value for r in res
        ) == 10

    def test_bulk_validation_costs_no_writes(self, mem):
        before = _source_count(mem)
        res = mem.bulk_add([{"content": 1}, {"content": 2}])
        assert all(r.acceptance == Acceptance.FAILED.value for r in res)
        assert _source_count(mem) == before

    def test_bulk_closed_memory(self, mem):
        mem.close()
        with pytest.raises(VerbatimError) as e:
            mem.bulk_add([{"content": "x"}])
        assert e.value.code == ErrorCode.INVALID_TRANSITION

    def test_bulk_perf_not_slower_than_add(self, path):
        """Measured: chunked commits amortize write-lock+fsync so bulk is
        no slower per item than sequential add on the same corpus.

        Two fresh stores per arm, best-of-2 per-item latency (best-case
        comparison is the standard low-noise check — outliers only ever
        inflate both sides). The 1.15x bound tolerates scheduler noise
        while still catching a per-item-tx regression honestly."""
        n = 150
        items = [{"content": f"perf corpus item {i} with text"} for i in range(n)]
        add_t, bulk_t = [], []
        for t in range(2):
            m1 = Memory(path=f"{path}.a{t}", worker="external")
            t0 = time.monotonic()
            for it in items:
                m1.add(it["content"])
            add_t.append((time.monotonic() - t0) / n)
            m1.close()
            m2 = Memory(path=f"{path}.b{t}", worker="external")
            t0 = time.monotonic()
            m2.bulk_add(items)
            bulk_t.append((time.monotonic() - t0) / n)
            m2.close()
        add_best, bulk_best = min(add_t), min(bulk_t)
        # numbers ride in the test log for the wave record
        print(
            f"\nbulk_add perf: add={add_best*1000:.2f}ms/item "
            f"bulk={bulk_best*1000:.2f}ms/item "
            f"({add_best/bulk_best:.2f}x)"
        )
        assert bulk_best <= add_best * 1.15
