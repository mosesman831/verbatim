"""V6 typed retrieval lane tests (SPEC_V6 §02, docs/v6_contracts §7).

The typed lane is the V6 hot path: grounded ``enrichment`` records plus
``entity_postings`` identifier hits, ranked through ``ranking/v1`` like
every other candidate.  These tests exercise the REAL pipeline —
``Memory.add`` → managed/external source projection → ``Memory.search``
— never seeded fakes:

* typed candidates surface under ``coverage.lanes["typed"]`` and the
  delivered line carries ``score_family="typed"`` plus re-verified
  ``byte_span`` pins (V6-02.02);
* held, suppressed, stale-generation, and lifecycle-ineligible records
  are excluded;
* an unpinnable record never delivers *as a typed fact* (the honest
  source-lane fallback may still cover the record itself);
* the existing source lane reports ``skipped`` when the grounded index
  covers the whole eligible corpus, ``ok`` when coverage is thin;
* barrier receipts mark their sources for the unblock-first drainer
  (``commit_notify.blocked_sources``);
* delivery exposure emits one record per delivered item when the
  sibling sink exists, and is a silent no-op otherwise;
* the pack-cache fingerprint provably covers source-projection deps
  (V6-02.06 evidence) while staying default-off.
"""

from __future__ import annotations

import pytest

from verbatim import Memory
from verbatim.core.types import JobKind
from verbatim.ingest import Ingester
from verbatim.memory.types import MemoryType
from verbatim.security import open_quarantine
from verbatim.storage import commit_notify
from verbatim.storage.repos import SourcesRepo


def _drain(mem: "Memory") -> None:
    """Run the enqueued source jobs through the real coordinator path —
    the external-worker's honest stand-in for the managed drainer."""
    ingester = Ingester(mem._store, mem._cfg, encoder=mem._encoder)
    report = ingester.drain_report(
        scope=mem._namespace,
        owner="t-typed",
        kinds=(JobKind.SOURCE_PROJECT, JobKind.SOURCE_EMBED),
    )
    assert report["failed"] == 0


@pytest.fixture()
def mem(tmp_path):
    m = Memory(path=str(tmp_path / "m.db"), worker="managed")
    yield m
    try:
        m.close()
    except Exception:
        pass


def _add(mem, text, **kw):
    res = mem.add(text, **kw)
    if getattr(mem._worker_mode, "value", "") == "managed":
        mem.wait_ready(
            res.receipt_id,
            capabilities=["source_lexical_ready"],
            timeout_ms=15000,
        )
    else:
        _drain(mem)
    return res


def _lanes(out):
    return dict((out.coverage or {}).get("lanes") or {})


class TestTypedCandidates:
    def test_typed_lane_reports_in_coverage(self, mem):
        _add(mem, "my favourite café is Bella Vista")
        out = mem.search("favourite café", retrieval="v6")
        lanes = _lanes(out)
        assert lanes.get("typed") == "ok", (
            "the typed lane must report its real status"
        )
        # The whole eligible corpus is grounded — the source lane is
        # honestly skipped rather than silently re-scanning projections.
        assert lanes.get("source") == "skipped"
        assert "+typed" in (out.coverage.get("route") or "")

    def test_delivered_typed_hit_carries_verified_pins(self, mem):
        res = _add(mem, "my favourite café is Bella Vista")
        out = mem.search("café", retrieval="v6")
        hits = [h for h in out.items if h.score_family == "typed"]
        assert hits, "a grounded typed record must deliver as typed"
        hit = hits[0]
        assert hit.memory_id == res.memory_id
        assert hit.type == MemoryType.PREFERENCE.value
        pins = getattr(hit, "pins", None)
        assert pins, "a delivered typed line carries its span pins"
        # Every pin re-verifies against the retained canonical bytes —
        # the same contract grounding/v1 enforces.
        verified, corrupt = SourcesRepo(mem._store).payload_many(
            [(res.memory_id, res.source_revision)]
        )
        assert not corrupt
        body = verified[(res.memory_id, res.source_revision)]
        for pin in pins:
            assert pin["kind"] == "byte_span"
            assert pin["source_id"] == res.memory_id
            assert body[int(pin["start"]): int(pin["end"])] == (
                pin["value"].encode("utf-8")
            )

    def test_identifier_query_surfaces_grounded_hit(self, mem):
        res = _add(mem, "api key: KEY-7781-ALPHA")
        out = mem.search("KEY-7781-ALPHA", retrieval="v6")
        hits = [h for h in out.items if h.memory_id == res.memory_id]
        assert hits, "the exact identifier hit must deliver"
        hit = hits[0]
        assert hit.score_family == "typed"
        values = {p.get("value") for p in getattr(hit, "pins", [])}
        assert "KEY-7781-ALPHA" in values, (
            "the delivered line's pins cover the matched identifier"
        )

    def test_temporal_record_surfaces(self, mem):
        _add(mem, "the meeting is scheduled for 2026-03-04")
        out = mem.search("meeting scheduled", retrieval="v6")
        assert out.items, "a dated typed record must deliver"
        typed = [h for h in out.items if h.score_family == "typed"]
        assert typed
        assert any(
            p.get("value") == "2026-03-04"
            for p in getattr(typed[0], "pins", [])
        )

    def test_held_source_never_delivers(self, mem):
        held = _add(mem, "my favourite café is Bella Vista")
        _add(mem, "my favourite restaurant is Golden Fork")
        with mem._store.tx() as conn:
            open_quarantine(
                conn,
                ("source", held.memory_id, 1),
                ["attack_risk:blocked"],
                [],
                scope_id=mem._namespace,
            )
        out = mem.search("favourite", retrieval="v6")
        assert out.items, "the live record still delivers"
        assert not any(
            h.memory_id == held.memory_id for h in out.items
        ), "a held source is withheld from typed candidates AND delivery"

    def test_suppressed_source_never_delivers(self, mem):
        gone = _add(mem, "my favourite café is Bella Vista")
        _add(mem, "my favourite restaurant is Golden Fork")
        out = mem.forget(gone.ref)
        assert out.suppression_state in (
            "suppressed", "purging", "completed",
        )
        res = mem.search("favourite", retrieval="v6")
        assert not any(
            h.memory_id == gone.memory_id for h in res.items
        ), "a suppressed source is withheld from typed candidates"

    def test_superseded_source_excluded_from_typed(self, mem):
        old = _add(mem, "The deploy command is deploy-v1.")
        _add(mem, "The deploy command is deploy-v2.", replaces=old.ref)
        out = mem.search("deploy command", retrieval="v6")
        assert any(
            "deploy-v2" in (h.quote or "") for h in out.items
        ), "the successor answers"
        assert not any(
            h.lifecycle == "active" and "deploy-v1" in (h.quote or "")
            for h in out.items
        ), "the superseded predecessor never answers as current state"

    def test_unpinnable_record_never_delivers_as_fact(self, mem):
        res = _add(mem, "my favourite café is Bella Vista")
        # Corrupt every pin source — serialized mention offsets AND the
        # posting offsets — so no span can re-verify against the retained
        # bytes (unsupported_extraction, V6-02.02).
        with mem._store.tx() as conn:
            conn.execute(
                "UPDATE enrichment SET fields_json = ?"
                " WHERE source_id = ?",
                (
                    '{"identifiers": [{"kind": "ghost", "value": "GHOST",'
                    ' "start": 0, "end": 1}], "entities": [],'
                    ' "temporal": {}, "producer": "enrich/v1"}',
                    res.memory_id,
                ),
            )
            conn.execute(
                "UPDATE entity_postings SET offsets = '[[9999,10000]]'"
                " WHERE source_id = ?",
                (res.memory_id,),
            )
        out = mem.search("favourite café", retrieval="v6")
        for h in out.items:
            if h.memory_id == res.memory_id:
                assert h.score_family != "typed", (
                    "an unpinnable record never ships as a typed fact"
                )
                assert not getattr(h, "pins", None), (
                    "an unpinnable record carries no delivered pins"
                )

    def test_thin_coverage_runs_source_fallback(self, mem):
        # No extractable mentions → no pins → the typed index cannot
        # cover this record: the source lane covers the gap.
        _add(mem, "the quick brown fox jumps quietly")
        out = mem.search("quick brown fox", retrieval="v6")
        lanes = _lanes(out)
        assert lanes.get("typed") == "ok"
        assert lanes.get("source") == "ok", (
            "an ungroundable eligible record forces the coverage fallback"
        )
        assert "+source" in (out.coverage.get("route") or "")
        assert out.items, "the source lane still answers"

    def test_typed_lane_tables_absent_reports_unavailable(self, mem):
        res = _add(mem, "my favourite café is Bella Vista")
        with mem._store.tx() as conn:
            conn.execute("DROP TABLE enrichment")
            conn.execute("DROP TABLE entity_postings")
        out = mem.search("café", retrieval="v6")
        lanes = _lanes(out)
        assert lanes.get("typed") == "unavailable", (
            "missing typed tables report unavailable honestly"
        )
        assert "typed_lane_unavailable" in out.warnings
        assert lanes.get("source") == "ok", (
            "the source lane covers exactly as before V6"
        )
        assert any(
            h.memory_id == res.memory_id for h in out.items
        ), "the record still delivers through the source path"


class TestLaneUnitContract:
    def test_typed_candidates_contract(self, mem):
        res = _add(mem, "my favourite café is Bella Vista")
        from verbatim.retrieval.v3.typed_lane import typed_candidates

        with mem._store.read() as conn:
            hits, stats = typed_candidates(
                conn,
                namespace=mem._namespace,
                query="favourite café",
                analysis=None,
                generation=None,
                limit=8,
                store=mem._store,
            )
        assert stats.status == "ok"
        assert stats.returned >= 1
        hit = next(h for h in hits if h.source_id == res.memory_id)
        assert hit.pins, "lane candidates carry verified span pins"
        assert hit.signals, "lane candidates carry raw fusion signals"
        assert stats.details["uncovered"] == 0

    def test_unscoped_request_refused(self, mem):
        from verbatim.retrieval.v3.typed_lane import typed_candidates

        with mem._store.read() as conn:
            hits, stats = typed_candidates(
                conn,
                namespace=None,
                query="x",
                analysis=None,
                generation=None,
                limit=8,
                store=mem._store,
            )
        assert hits == []
        assert stats.status == "unavailable"


class TestBarrierSourceMarking:
    def test_pending_receipt_marks_its_source(self, tmp_path):
        path = str(tmp_path / "ext.db")
        m = Memory(path=path, worker="external", ready_timeout_ms=20)
        try:
            res = m.add("my favourite café is Bella Vista")
            out = m.search("café", retrieval="v6")
            # The undrained projection keeps the barrier honestly pending.
            assert out.readiness.get("causal_satisfied") is False
            blocked = commit_notify.blocked_sources(m._store._path)
            assert res.memory_id in blocked, (
                "a pending barrier receipt marks its source for the "
                "unblock-first drainer (V6-02.08)"
            )
            commit_notify.clear_barrier_sources(m._store._path)
        finally:
            m.close()

    def test_met_barrier_marks_nothing(self, mem):
        res = _add(mem, "my favourite café is Bella Vista")
        commit_notify.clear_barrier_sources(mem._store._path)
        out = mem.search("café", retrieval="v6")
        assert out.readiness.get("causal_satisfied") is True
        blocked = commit_notify.blocked_sources(mem._store._path)
        assert res.memory_id not in blocked, (
            "a met barrier marks nothing — the mark is for pending work"
        )


class TestExposureEmission:
    def test_absent_module_is_noop(self, mem):
        import verbatim.memory.facade as facade

        original = facade._exposure
        facade._exposure = None
        try:
            _add(mem, "my favourite café is Bella Vista")
            out = mem.search("café", retrieval="v6")
            assert "exposure_emit_failed" not in out.warnings
        finally:
            facade._exposure = original

    def test_emit_deliveries_called_post_assembly(self, mem):
        import verbatim.memory.facade as facade

        calls = []

        class _Sink:
            @staticmethod
            def emit_deliveries(store, *, receipt_id, namespace,
                                deliveries):
                calls.append(
                    {
                        "receipt_id": receipt_id,
                        "namespace": namespace,
                        "deliveries": list(deliveries),
                    }
                )

        original = facade._exposure
        facade._exposure = _Sink
        try:
            res = _add(mem, "my favourite café is Bella Vista")
            out = mem.search("café", retrieval="v6")
        finally:
            facade._exposure = original
        assert out.items
        assert len(calls) == 1
        call = calls[0]
        assert call["namespace"] == mem._namespace
        assert call["receipt_id"], "the causal token receipts deliveries"
        assert {
            "source_id": res.memory_id,
            "revision": res.source_revision,
            "score_family": "typed",
        } in call["deliveries"], (
            "one delivery record per delivered item, with score family"
        )

    def test_record_source_deliveries_shape_supported(self, mem):
        import verbatim.memory.facade as facade
        from verbatim.storage.schema_v5 import ensure_additive_tables

        class _Sink:
            @staticmethod
            def record_source_deliveries(conn, receipt_id, deliveries):
                ensure_additive_tables(conn)
                for i, dl in enumerate(deliveries):
                    conn.execute(
                        "INSERT OR REPLACE INTO source_exposure"
                        "(receipt_id, ord, source_id, revision,"
                        " score_family, delivered_at_us, namespace)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (
                            receipt_id, i, dl["source_id"],
                            dl["revision"], dl["score_family"], 1,
                            "ns",
                        ),
                    )

        original = facade._exposure
        facade._exposure = _Sink
        try:
            res = _add(mem, "my favourite café is Bella Vista")
            out = mem.search("café", retrieval="v6")
            token = out.causal_token
        finally:
            facade._exposure = original
        assert token
        with mem._store.read() as conn:
            rows = conn.execute(
                "SELECT source_id, revision, score_family"
                " FROM source_exposure WHERE receipt_id = ?",
                (token,),
            ).fetchall()
        assert (res.memory_id, res.source_revision, "typed") in rows

    def test_emit_failure_warns_never_raises(self, mem):
        import verbatim.memory.facade as facade

        class _Sink:
            @staticmethod
            def emit_deliveries(store, *, receipt_id, namespace,
                                deliveries):
                raise RuntimeError("sink down")

        original = facade._exposure
        facade._exposure = _Sink
        try:
            _add(mem, "my favourite café is Bella Vista")
            out = mem.search("café", retrieval="v6")
        finally:
            facade._exposure = original
        assert "exposure_emit_failed" in out.warnings
        assert out.items, "an emit failure never blocks delivery"


class TestPackCacheFingerprint:
    def test_fingerprint_covers_source_projection_deps(self, mem):
        """V6-02.06 evidence: a source-projection commit (projection +
        enrichment + postings rows) moves the cache fingerprint's
        content watermark — stale serves are impossible."""
        from verbatim.retrieval.cache import _watermarks

        res = _add(mem, "my favourite café is Bella Vista")
        with mem._store.read() as conn:
            before = _watermarks(conn)
        with mem._store.tx() as conn:
            conn.execute(
                "INSERT INTO entity_postings"
                "(namespace, entity, entity_kind, source_id, revision,"
                " offsets, generation) VALUES(?,?,?,?,?,?,?)",
                (
                    mem._namespace, "PROBE-1", "identifier:test",
                    res.memory_id, res.source_revision, "[[0,1]]", 0,
                ),
            )
        with mem._store.read() as conn:
            after = _watermarks(conn)
        assert before[0] != after[0], (
            "the content watermark covers the typed-index tables — "
            "a projection commit invalidates cached packs"
        )

    def test_cache_default_off(self):
        """V6-02.06: the pack cache stays opt-in — flipping the default
        is a config-owner decision gated on this coverage evidence."""
        from verbatim.config import VerbatimConfig

        cfg = VerbatimConfig()
        cache_cfg = getattr(getattr(cfg, "retrieval", None), "cache", None)
        assert not bool(getattr(cache_cfg, "enabled", False))
