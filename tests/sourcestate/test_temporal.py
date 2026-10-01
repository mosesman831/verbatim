"""source_state/v1 — bitemporal evaluation (V5-14.08/12/13).

Two axes: ``at_time`` selects the answer inside the declared validity
semantics (a future ``effective_at`` never makes the successor current
early); ``known_at`` selects which committed doc is visible (known-at
queries return prior information under current authorization).
"""

from __future__ import annotations

import pytest

from verbatim.core import time as _time
from verbatim.sourcestate import (
    apply_erasure,
    current_state,
    ensure_state,
    history,
    is_revision_current,
    transition,
)

from .conftest import NS, SID, add_revision, seed_source

PRODUCER = "verbatim.memory.facade.v1"
HOUR = 3600_000_000


def _setup(conn, store, sid=SID, ns=NS):
    seed_source(conn, store, sid, ns)
    return ensure_state(conn, sid, ns, store=store)


class TestFutureEffectiveAt:
    def test_future_supersede_invisible_until_boundary(self, installed):
        """V5-14.12: scheduling never makes future state current early;
        selection is evaluated at read time — no job has to wake."""
        now = _time.now_us()
        boundary = _time.rfc3339(now + HOUR)
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2,
                effective_at=boundary, producer=PRODUCER,
            )
        with installed.read() as conn:
            # Now (before boundary): predecessor still the answer.
            cur = current_state(conn, SID)
            assert cur.current is True
            assert cur.effective_revision == 1
            assert cur.pending == {
                "successor": "2",
                "effective_at": boundary,
                "control_version": 1,
            }
            assert is_revision_current(conn, SID, 1)
            assert not is_revision_current(conn, SID, 2)
            # At the boundary: the successor answers (pure read-time eval).
            after = current_state(conn, SID, at_time=now + 2 * HOUR)
            assert after.current is True
            assert after.effective_revision == 2
            assert after.pending is None
            assert is_revision_current(conn, SID, 2, at_time=now + 2 * HOUR)
            assert not is_revision_current(
                conn, SID, 1, at_time=now + 2 * HOUR
            )

    def test_immediate_supersede_defaults_to_commit_time(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
        with installed.read() as conn:
            cur = current_state(conn, SID)
        assert cur.effective_revision == 2
        assert cur.pending is None

    def test_supersede_window_can_expire_successor(self, installed):
        now = _time.now_us()
        boundary = _time.rfc3339(now + HOUR)
        close = _time.rfc3339(now + 3 * HOUR)
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2,
                effective_at=boundary, valid_to=close, producer=PRODUCER,
            )
        with installed.read() as conn:
            mid = current_state(conn, SID, at_time=now + 2 * HOUR)
            assert mid.current and mid.effective_revision == 2
            late = current_state(conn, SID, at_time=now + 4 * HOUR)
            assert late.label == "expired"
            assert late.current is False


class TestKnownAt:
    def test_known_at_returns_prior_information(self, installed):
        """V5-14.08/13: known-at before a correction still shows the
        earlier assertion; current queries cannot treat it as settled."""
        with installed.tx() as conn:
            _setup(conn, installed)
        with installed.read() as conn:
            k0 = _time.parse_rfc3339(history(conn, SID)[0]["known_at"])
        k1 = k0 + 5_000_000
        with installed.tx() as conn:
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="correct", superseded_by=2,
                producer=PRODUCER, known_at=k1,
            )
        with installed.read() as conn:
            before = current_state(conn, SID, known_at=k0)
            assert before.found and before.control_version == 0
            assert before.disposition == "active"
            assert before.effective_revision == 1
            assert before.current is True
            after = current_state(conn, SID, known_at=k1)
            assert after.control_version == 1
            assert after.disposition == "corrected"
            assert after.effective_revision == 2
            # before the first record was known, nothing is visible
            k_pre = _time.parse_rfc3339(history(conn, SID)[0]["known_at"]) - 1
            pre = current_state(conn, SID, known_at=k_pre)
            assert pre.found is False and pre.label == "unknown"

    def test_known_at_shows_pre_retraction_state(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
        with installed.read() as conn:
            k0 = _time.parse_rfc3339(history(conn, SID)[0]["known_at"])
        k1 = k0 + 5_000_000
        with installed.tx() as conn:
            transition(
                conn, SID, expected_control_version=0,
                disposition="retract", producer=PRODUCER, known_at=k1,
            )
        with installed.read() as conn:
            hist = current_state(conn, SID, known_at=k0)
            assert hist.disposition == "active" and hist.current
            now_cur = current_state(conn, SID)
            assert now_cur.disposition == "retracted" and not now_cur.current

    def test_known_at_and_valid_time_are_independent_axes(self, installed):
        """A scheduled supersede known now stays invisible at a valid-time
        before the boundary even when evaluated with full knowledge."""
        now = _time.now_us()
        boundary = _time.rfc3339(now + HOUR)
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2,
                effective_at=boundary, producer=PRODUCER,
            )
        with installed.read() as conn:
            # known_at = now (sees the scheduled doc), at_time before boundary
            cur = current_state(
                conn, SID, at_time=now, known_at=now + 10 * HOUR
            )
            assert cur.control_version == 1  # the supersede doc is known…
            assert cur.effective_revision == 1  # …but not yet effective
            assert cur.pending is not None


class TestWindows:
    def test_active_window(self, installed):
        now = _time.now_us()
        with installed.tx() as conn:
            _setup(conn, installed)
            transition(
                conn, SID, expected_control_version=0,
                disposition="activate",
                valid_from=_time.rfc3339(now + HOUR),
                valid_to=_time.rfc3339(now + 2 * HOUR),
                producer=PRODUCER,
            )
        with installed.read() as conn:
            early = current_state(conn, SID, at_time=now)
            assert early.label == "scheduled" and not early.current
            mid = current_state(conn, SID, at_time=now + 90 * 60_000_000)
            assert mid.current and mid.effective_revision == 1
            late = current_state(conn, SID, at_time=now + 3 * HOUR)
            assert late.label == "expired" and not late.current

    def test_history_covers_full_lifecycle(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            transition(
                conn, SID, expected_control_version=1,
                disposition="retract", producer=PRODUCER,
            )
            apply_erasure(conn, SID, producer="privacy.closure")
        with installed.read() as conn:
            docs = history(conn, SID)
        assert [d["control_version"] for d in docs] == [0, 1, 2, 3]
        assert [d["change"] for d in docs] == [
            "create", "supersede", "retract", "erase",
        ]
        assert [d["disposition"] for d in docs] == [
            "active", "superseded", "retracted", "erased",
        ]
        # every doc carries its provenance fields
        for d in docs:
            assert d["effect"] == "source_revision_transition"
            assert d["producer"] is not None
            assert "recorded_event" in d
        # tombstone read: fenced, never reactivated
        cur = current_state(conn, SID)
        assert cur.label == "erased" and cur.current is False
