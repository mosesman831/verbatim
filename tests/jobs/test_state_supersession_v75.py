"""V75-03.01 slot-key supersession at write + V75-04.08 predecessor
delivery (SPEC_V7_5 §03/§04/§09 J01–J03; SPEC_V7 V7-16.03/16.04/16.05).

Write path: ``project_units_v7`` → ``_write_prefs_state`` →
``_supersede_state_fact`` runs inside the projection transaction — each
emitted state fact is compared against the live rows of its
``(scope_id, state_key)`` slot (latest fenced row per natural key) via
``prefs_state.state_compatible``.  Ordered incompatible values demote
the prior claim to ``historical`` with ``valid_to_us`` = successor
occurred start (recorded_at_us fallback); unordered incompatible values
leave both sides ``disputed``; accumulate families never supersede.

Read path: ``lane_typed`` surfaces the demoted predecessor labeled
``historical`` for ``current_value``/``history_of``; ``assemble_pack``
delivers the slot's items as one atomic ``state:<key>`` group rendered
under ``## CONFLICTS``; ``consolidate_scope_v7`` reports the slot's
profile ``disputed`` only for genuinely unordered contradictions.

Real stores throughout — no mocks.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    LifecycleLabel,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
    ScoredCandidate,
)
from verbatim.jobs import units_jobs as uj  # noqa: E402
from verbatim.observations.consolidate_v7 import consolidate_scope_v7  # noqa: E402
from verbatim.retrieval.v7 import typed as tlane  # noqa: E402
from verbatim.retrieval.v7.pack import assemble_pack  # noqa: E402
from verbatim.retrieval.v7.render import render_reader_view  # noqa: E402
from verbatim.storage.store import Store  # noqa: E402

SCOPE_ID = "scope:1"
GEN = 1
DAY_US = 24 * 3600 * 1_000_000


def _us(iso: str) -> int:
    return int(
        datetime.fromisoformat(iso)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000_000
    )


def _occ(day: str) -> dict:
    """add_args carrying an explicit day-precision occurred interval."""
    return {
        "occurred": {
            "start": day,
            "end": day,
            "precision": "day",
            "source": "explicit",
        }
    }


def _src(source_id: str, speaker: str = "alice") -> dict:
    return {
        "source_id": source_id,
        "origin": "test",
        "external_id": None,
        "source_kind": "user_message",
        "scope_id": SCOPE_ID,
        "speaker_id": speaker,
        "created_us": 1_700_000_000_000_000,
    }


def _rev(payload, source_id: str, event_us, captured_us=None):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if captured_us is None and event_us is not None:
        captured_us = event_us + 1000
    return {
        "source_id": source_id,
        "revision": 1,
        "payload": payload,
        "payload_hmac_hex": "00",
        "event_us": event_us,
        "captured_us": captured_us,
        "timezone": None,
        "provenance": "direct_user",
        "metadata_json": "{}",
    }


def _project(store, source_id, text, event_us, args=None, gen=GEN):
    with store.tx() as conn:
        return uj.project_units_v7(
            conn,
            source_row=_src(source_id),
            revision_row=_rev(text, source_id, event_us),
            add_args=args or {},
            generation=gen,
            scope_id=SCOPE_ID,
        )


def _facts(store, state_key=None):
    with store.read() as conn:
        q = ("SELECT unit_id, state_key, value_norm, valid_from_us,"
             " valid_to_us, status, generation FROM state_facts")
        params: tuple = ()
        if state_key is not None:
            q += " WHERE state_key=?"
            params = (state_key,)
        q += " ORDER BY value_norm, unit_id"
        return [
            {
                "unit_id": r[0], "state_key": r[1], "value_norm": r[2],
                "valid_from_us": r[3], "valid_to_us": r[4],
                "status": r[5], "generation": r[6],
            }
            for r in conn.execute(q, params)
        ]


def _seed_revision(conn, source_id, payload, event_us):
    """sources + source_revisions rows so consolidation can lift quotes."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    conn.execute(
        "INSERT INTO scopes(scope_id, profile_id, principal_id,"
        " workspace_id, conversation_id, visibility)"
        " VALUES (?, 'p', 'alice', NULL, 'c1', 'conversation')"
        " ON CONFLICT(scope_id) DO NOTHING",
        (SCOPE_ID,),
    )
    conn.execute(
        "INSERT INTO sources(source_id, origin, external_id, source_kind,"
        " scope_id, speaker_id, created_us)"
        " VALUES (?, 't', NULL, 'user_message', ?, 'alice', 0)",
        (source_id, SCOPE_ID),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, timezone, provenance,"
        " metadata_json) VALUES (?, 1, ?, X'00', ?, ?, NULL,"
        " 'direct_user', '{}')",
        (source_id, payload, event_us or 0, (event_us or 0) + 1000),
    )


def _unit_texts(store):
    with store.read() as conn:
        return {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT r.unit_id, c.text FROM unit_fts_rows r"
                " JOIN unit_fts_content c ON c.fts_row_id = r.row_id"
            )
        }


# -- lane / pack helpers (V7-04.08 delivery path) ---------------------------


def _mk_qv(primary=IntentClass.CURRENT_VALUE, canons=("alice",), terms=()):
    norm = NormAnalysis(
        analyzer_id="norm/v2",
        terms=tuple(NormTerm(t, "text", 0, len(t)) for t in terms),
        identifiers=(),
    )
    return QueryViewV7(
        query="test query",
        norm=norm,
        intent=IntentResult(primary=primary, classes=(primary,)),
        entity_canons=tuple(canons),
        query_time_us=_us("2024-06-01"),
    )


def _mk_ctx(conn, gen=GEN):
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=(LaneName.TYPED,),
        lane_weights={},
    )
    return LaneContextV7(
        store=conn,
        scope_id=SCOPE_ID,
        generation=gen,
        eligible=lambda row: True,
        query_time_us=_us("2024-06-01"),
        profile="test",
        budget=BudgetClass.MID,
        policy=policy,
    )


def _scored(cand, quote):
    """Lane candidate → ScoredCandidate with the pack's declared detail
    contract (``lifecycle``/``state_key``/``state_value``/``valid_from``
    are the keys ``assemble_pack._as_item`` reads off ``detail``)."""
    sig = cand.signals
    detail = {"quote": quote.encode("utf-8")}
    for key in ("lifecycle", "state_key", "state_value"):
        if sig.get(key) is not None:
            detail[key] = sig[key]
    if sig.get("valid_from_us") is not None:
        detail["valid_from_us"] = sig["valid_from_us"]
    return ScoredCandidate(
        unit_id=cand.unit_id,
        source_id=cand.source_id,
        revision=cand.revision,
        score=float(100 - cand.rank),
        score_family="ranking/v7",
        detail=detail,
    )


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v7.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# J01 — ordered incompatible value: predecessor historical, bounded
# ---------------------------------------------------------------------------


class TestJ01Supersession:
    def test_newer_incompatible_supersedes_occurred_start(self, store):
        """J01 core: successor occurred start becomes predecessor
        ``valid_to_us``; the successor stays ``current``."""
        _project(store, "s1", "I live in Oslo.",
                 _us("2020-01-01"), args=_occ("2020-01-01"))
        stats = _project(store, "s2", "I live in Bergen.",
                         _us("2023-06-01"), args=_occ("2023-06-01"))
        assert stats["state_facts"] == 1
        assert stats["state_superseded"] == 1
        rows = _facts(store, "alice/home_city")
        assert len(rows) == 2
        by_val = {r["value_norm"]: r for r in rows}
        assert by_val["bergen"]["status"] == "current"
        assert by_val["bergen"]["valid_to_us"] is None
        assert by_val["oslo"]["status"] == "historical"
        # valid_to_us = successor.occurred_start_us (2023-06-01 00:00Z)
        assert by_val["oslo"]["valid_to_us"] == _us("2023-06-01")

    def test_newer_incompatible_recorded_fallback(self, store):
        """No occurred on the successor → ``valid_to_us`` falls back to
        the successor unit's ``recorded_at_us`` (V75-03.01 step 3)."""
        t_old = _us("2020-01-01")
        t_new = _us("2023-06-01")
        _project(store, "s1", "I live in Oslo.", t_old)
        _project(store, "s2", "I live in Bergen.", t_new)
        by_val = {r["value_norm"]: r for r in _facts(store, "alice/home_city")}
        assert by_val["oslo"]["status"] == "historical"
        assert by_val["oslo"]["valid_to_us"] == t_new
        assert by_val["bergen"]["status"] == "current"

    def test_accumulate_family_never_supersedes(self, store):
        """J01 second leg: ``pets`` is accumulate-mode — dog + cat are
        both live ``current`` claims, no demotion."""
        stats = _project(store, "s1", "I have a dog.",
                         _us("2020-01-01"), args=_occ("2020-01-01"))
        stats = _project(store, "s2", "I have a cat.",
                         _us("2023-06-01"), args=_occ("2023-06-01"))
        assert stats["state_superseded"] == 0
        rows = _facts(store, "alice/pets")
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {"current"}

    def test_same_value_restatement_not_demoted(self, store):
        """Equal values are compatible — a restatement adds a second
        current row, never a false ``historical``."""
        _project(store, "s1", "I live in Oslo.",
                 _us("2020-01-01"), args=_occ("2020-01-01"))
        stats = _project(store, "s2", "I live in Oslo.",
                         _us("2023-06-01"), args=_occ("2023-06-01"))
        assert stats["state_superseded"] == 0
        assert stats["state_disputed"] == 0
        rows = _facts(store, "alice/home_city")
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {"current"}

    def test_incoming_older_value_lands_historical(self, store):
        """Symmetric read of V7-16.03: when the *incoming* claim is the
        prior value, it lands ``historical`` bounded by the nearest
        incompatible successor's anchor."""
        _project(store, "s1", "I live in Bergen.",
                 _us("2023-06-01"), args=_occ("2023-06-01"))
        stats = _project(store, "s2", "I live in Oslo.",
                         _us("2020-01-01"), args=_occ("2020-01-01"))
        by_val = {r["value_norm"]: r for r in _facts(store, "alice/home_city")}
        assert by_val["bergen"]["status"] == "current"
        assert by_val["oslo"]["status"] == "historical"
        assert by_val["oslo"]["valid_to_us"] == _us("2023-06-01")
        assert stats["state_superseded"] == 1  # the incoming row

    def test_generation_fence_not_crossed(self, store):
        """Rows at generation > the write fence are invisible to the
        pass — a gen-2 row is never demoted by a gen-1 write."""
        _project(store, "s1", "I live in Oslo.",
                 _us("2020-01-01"), args=_occ("2020-01-01"))
        with store.tx() as conn:
            # a foreign-producer row at a newer generation
            conn.execute(
                "INSERT INTO units(unit_id, source_id, revision, scope_id,"
                " kind, speaker_canon, recorded_at_us, generation)"
                " VALUES ('u7:future', 'x', 1, ?, 'turn', 'alice',"
                " ?, 2)",
                (SCOPE_ID, _us("2025-01-01")),
            )
            conn.execute(
                "INSERT INTO state_facts(scope_id, state_key, unit_id,"
                " generation, value_text, value_norm, valid_from_us,"
                " valid_to_us, status, producer, pins_json)"
                " VALUES (?, 'alice/home_city', 'u7:future', 2,"
                " 'Tromsø', 'tromsø', ?, NULL, 'current',"
                " 'test/foreign', '{}')",
                (SCOPE_ID, _us("2025-01-01")),
            )
        _project(store, "s2", "I live in Bergen.",
                 _us("2023-06-01"), args=_occ("2023-06-01"), gen=1)
        rows = _facts(store, "alice/home_city")
        by_val = {r["value_norm"]: r for r in rows}
        assert by_val["tromsø"]["status"] == "current"  # untouched
        assert by_val["tromsø"]["valid_to_us"] is None
        assert by_val["oslo"]["status"] == "historical"
        assert by_val["bergen"]["status"] == "current"

    def test_replay_is_convergent(self, store):
        """Reprojecting a slice must reproduce identical lifecycle
        labels — the disputed/historical marks survive a rewrite."""
        args = _occ("2020-01-01")
        _project(store, "s1", "I live in Oslo.", _us("2020-01-01"), args=args)
        _project(store, "s2", "I live in Bergen.", _us("2023-06-01"),
                 args=_occ("2023-06-01"))
        before = _facts(store)
        _project(store, "s1", "I live in Oslo.", _us("2020-01-01"), args=args)
        _project(store, "s2", "I live in Bergen.", _us("2023-06-01"),
                 args=_occ("2023-06-01"))
        after = _facts(store)
        assert before == after

    def test_compat_absent_marks_skipped(self, store, monkeypatch):
        """Without ``state_compatible`` the pass honestly reports
        ``skipped`` — facts still land as plain ``current`` rows."""
        monkeypatch.setattr(uj, "_load_state_compat", lambda: None)
        _project(store, "s1", "I live in Oslo.", _us("2020-01-01"))
        stats = _project(store, "s2", "I live in Bergen.", _us("2023-06-01"))
        assert stats["state_facts"] == 1
        assert stats["state_superseded"] == "skipped"
        assert stats["state_disputed"] == "skipped"
        rows = _facts(store, "alice/home_city")
        assert {r["status"] for r in rows} == {"current"}


# ---------------------------------------------------------------------------
# J02 — unordered incompatible values: both disputed, surfaced as conflicts
# ---------------------------------------------------------------------------


class TestJ02Disputed:
    def test_equal_anchors_both_disputed(self, store):
        """Two different values claiming the same instant — no clear
        temporal order → both rows ``disputed`` (V7-16.05)."""
        _project(store, "s1", "I live in Oslo.",
                 _us("2023-06-01"), args=_occ("2023-06-01"))
        stats = _project(store, "s2", "I live in Bergen.",
                         _us("2023-06-02"), args=_occ("2023-06-01"))
        assert stats["state_disputed"] == 2  # predecessor + incoming
        rows = _facts(store, "alice/home_city")
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {"disputed"}
        assert all(r["valid_to_us"] is None for r in rows)

    def test_unknown_anchors_both_disputed(self, store):
        """Neither side datable (no occurred, no recorded) → unordered
        → disputed."""
        _project(store, "s1", "I have worked at Acme.", None)
        _project(store, "s2", "I have worked at Globex.", None)
        rows = _facts(store, "alice/employer")
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {"disputed"}

    def test_dispute_joins_open_disputed_row(self, store):
        """A third unordered incompatible value joins the existing
        dispute — replay/insertion convergence (the pass sees
        ``disputed`` rows, not just ``current``)."""
        _project(store, "s1", "I live in Oslo.",
                 _us("2023-06-01"), args=_occ("2023-06-01"))
        _project(store, "s2", "I live in Bergen.",
                 _us("2023-06-02"), args=_occ("2023-06-01"))
        _project(store, "s3", "I live in Madrid.",
                 _us("2023-06-03"), args=_occ("2023-06-01"))
        rows = _facts(store, "alice/home_city")
        assert len(rows) == 3
        assert {r["status"] for r in rows} == {"disputed"}

    def test_disputed_profile_and_conflicts_render(self, store):
        """J02 surfacing: the disputed slot lands as a ``disputed``
        ``profiles_v7`` row, and the packed reader view lists the
        state-key group under ``## CONFLICTS``."""
        t1, t2 = _us("2023-06-01"), _us("2023-06-02")
        with store.tx() as conn:
            _seed_revision(conn, "s1", "I live in Oslo.", t1)
            _seed_revision(conn, "s2", "I live in Bergen.", t2)
        _project(store, "s1", "I live in Oslo.", t1, args=_occ("2023-06-01"))
        _project(store, "s2", "I live in Bergen.", t2, args=_occ("2023-06-01"))

        with store.tx() as conn:
            cstats = consolidate_scope_v7(
                conn, scope_id=SCOPE_ID, generation=GEN,
                now_us=_us("2024-06-01"))
        assert cstats["status"] == "ok"
        with store.read() as conn:
            prof = conn.execute(
                "SELECT subject_canon, slot, status FROM profiles_v7"
                " WHERE subject_canon='alice' AND slot='home_city'"
            ).fetchall()
        assert prof == [("alice", "home_city", "disputed")]

        with store.read() as conn:
            out = tlane.lane_typed(
                _mk_ctx(conn),
                _mk_qv(IntentClass.CURRENT_VALUE),
                LaneSlice(deadline_ms=10_000, cap=50),
            )
        assert out.status == LaneStatus.OK
        cands = [c for c in out.candidates
                 if c.signals.get("state_key") == "alice/home_city"]
        assert len(cands) == 2
        assert all(c.signals["lifecycle"] == "disputed" for c in cands)
        assert all(c.signals["status"] == "disputed" for c in cands)

        texts = _unit_texts(store)
        scored = [_scored(c, texts[c.unit_id]) for c in cands]
        pack = assemble_pack(
            scored, _mk_qv(IntentClass.CURRENT_VALUE),
            max_tokens=500, limit=10)
        assert pack.conflicts == [
            ("state:alice/home_city", tuple(i.ref for i in pack.items))
        ]
        view = render_reader_view(pack)
        assert "## CONFLICTS" in view
        assert "- state:alice/home_city:" in view
        # both sides ship under the conflict line
        for i in pack.items:
            assert i.lifecycle is LifecycleLabel.DISPUTED
            assert i.pins.get("state_key") == "alice/home_city"


# ---------------------------------------------------------------------------
# J03 — current_value / history_of delivery keeps the predecessor
# ---------------------------------------------------------------------------


class TestJ03PredecessorDelivery:
    def _seed_change(self, store):
        _project(store, "s1", "I lived in Oslo.",
                 _us("2020-01-01"), args=_occ("2020-01-01"))
        _project(store, "s2", "I live in Bergen.",
                 _us("2023-06-01"), args=_occ("2023-06-01"))

    def test_typed_lane_delivers_current_then_historical(self, store):
        """current_value: the new value's unit ranks first; the
        predecessor unit is still a candidate, labeled historical."""
        self._seed_change(store)
        with store.read() as conn:
            out = tlane.lane_typed(
                _mk_ctx(conn),
                _mk_qv(IntentClass.CURRENT_VALUE),
                LaneSlice(deadline_ms=10_000, cap=50),
            )
        assert out.status == LaneStatus.OK
        slot = [c for c in out.candidates
                if c.signals.get("state_key") == "alice/home_city"]
        assert len(slot) == 2
        first, second = slot[0], slot[1]
        assert first.signals["state_value"] == "Bergen"
        assert first.signals["lifecycle"] == "current"
        assert first.signals["status"] == "current"
        assert second.signals["state_value"] == "Oslo"
        assert second.signals["lifecycle"] == "historical"
        assert second.signals["status"] == "historical"
        # the delivered predecessor carries its closed bound
        assert second.signals["valid_to_us"] == _us("2023-06-01")
        assert second.signals["valid_from_us"] == _us("2020-01-01")

    def test_history_of_chain_orders_newest_first(self, store):
        """history_of: the full chain ships ordered by fact anchor —
        both statuses delivered (V7-16.04)."""
        self._seed_change(store)
        with store.read() as conn:
            out = tlane.lane_typed(
                _mk_ctx(conn),
                _mk_qv(IntentClass.HISTORY_OF),
                LaneSlice(deadline_ms=10_000, cap=50),
            )
        assert out.status == LaneStatus.OK
        slot = [c for c in out.candidates
                if c.signals.get("state_key") == "alice/home_city"]
        assert [c.signals["state_value"] for c in slot] == ["Bergen", "Oslo"]
        assert [c.signals["lifecycle"] for c in slot] == [
            "current", "historical"]

    def test_pack_leads_new_value_and_labels_predecessor(self, store):
        """J03 pack leg: the assembled pack leads with Bergen; Oslo is
        delivered as a protected ``historical`` item and the reader view
        annotates it ``(historical)``."""
        self._seed_change(store)
        with store.read() as conn:
            out = tlane.lane_typed(
                _mk_ctx(conn),
                _mk_qv(IntentClass.CURRENT_VALUE),
                LaneSlice(deadline_ms=10_000, cap=50),
            )
        texts = _unit_texts(store)
        slot = [c for c in out.candidates
                if c.signals.get("state_key") == "alice/home_city"]
        scored = [_scored(c, texts[c.unit_id]) for c in slot]
        pack = assemble_pack(
            scored, _mk_qv(IntentClass.CURRENT_VALUE),
            max_tokens=500, limit=10)

        items = pack.items
        assert len(items) == 2
        assert items[0].pins.get("state_value") == "Bergen"
        assert items[0].lifecycle is LifecycleLabel.CURRENT
        pred = items[1]
        assert pred.pins.get("state_value") == "Oslo"
        assert pred.lifecycle is LifecycleLabel.HISTORICAL
        # the slot forms one atomic group — both delivered together
        assert pack.conflicts == [
            ("state:alice/home_city", tuple(i.ref for i in items))]
        view = render_reader_view(pack)
        assert "(historical)" in view
        # ordering in the rendered evidence: new value first
        assert view.index("Bergen") < view.index("Oslo")

    def test_profile_reports_new_value_not_disputed(self, store):
        """The motivating defect: after an *ordered* change the slot's
        profile must report the new value ``current`` — not a spurious
        ``disputed`` from two live currents."""
        t1, t2 = _us("2020-01-01"), _us("2023-06-01")
        with store.tx() as conn:
            _seed_revision(conn, "s1", "I lived in Oslo.", t1)
            _seed_revision(conn, "s2", "I live in Bergen.", t2)
        self._seed_change(store)
        with store.tx() as conn:
            cstats = consolidate_scope_v7(
                conn, scope_id=SCOPE_ID, generation=GEN,
                now_us=_us("2024-06-01"))
        assert cstats["status"] == "ok"
        with store.read() as conn:
            prof = conn.execute(
                "SELECT value, status FROM profiles_v7"
                " WHERE subject_canon='alice' AND slot='home_city'"
            ).fetchall()
        assert prof == [("Bergen", "current")]
