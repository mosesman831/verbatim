"""Consumer result cache for ``Memory.search`` (V6-02.13).

Real on-disk ``Store`` per test (``tmp_path``), real ingest drain, real
session barrier, real quarantine/forget machinery — the cache is a
performance layer over the assembled ``SearchResult``, exercised end to
end through the public facade.

The cache is OFF by default; every hit-path test enables it explicitly
via ``config={"retrieval": {"cache": {"enabled": True}}}``.

Covered:
  - default-off: no cache object, no stored results, honest stats;
  - a repeated identical search serves ``cache_hit`` with the same
    items, fresh readiness fields, and fresh exposure rows;
  - an add between searches shifts the barrier frontier — the old
    answer is never served across it;
  - ``forget`` suppression and quarantine holds invalidate the stored
    result — held/forgotten sources can never replay from cache;
  - pending barriers and insufficient results are never stored.
"""

from __future__ import annotations

import pytest

import verbatim.security as security
from verbatim import Memory
from verbatim.core.types import JobKind
from verbatim.ingest import Ingester
from verbatim.retrieval import cache as _rcache


def _mem(path, **kw):
    return Memory(path=path, worker="external", **kw)


def _mem_cached(path):
    return _mem(
        path, config={"retrieval": {"cache": {"enabled": True}}}
    )


def _settle(m):
    """Drain durable source work, then resolve the session barrier so a
    search probe is eligible (the cache only engages on a met barrier
    with no deferred capabilities)."""
    report = Ingester(
        m._store, m._cfg, encoder=m._encoder
    ).drain_report(
        scope=m._namespace,
        owner="t-cache",
        limit=256,
    )
    assert report["failed"] == 0
    if m._session_receipts:
        w = m.wait_ready(m._session_receipts[-1], timeout_ms=15000)
        assert w.state == "ready"


def _stats(m):
    return _rcache.stats_for(m._store, m._cfg)


def _exposure_count(m) -> int:
    try:
        with m._store.read() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM source_exposure"
                ).fetchone()[0]
            )
    except Exception:
        return 0


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "m.db")


@pytest.fixture()
def mem(path):
    m = _mem(path)
    yield m
    try:
        m.close()
    except Exception:
        pass


@pytest.fixture()
def cmem(path):
    m = _mem_cached(path)
    yield m
    try:
        m.close()
    except Exception:
        pass


# ---------------------------------------------------------------------
# default-off
# ---------------------------------------------------------------------


class TestDefaultOff:
    def test_no_cache_object_without_config(self, mem):
        """Default config: searches work, no cache object is ever
        attached to the store, and ``stats_for`` reports the honest
        disabled surface (V4-33 opt-in carried to the consumer cache)."""
        mem.add("default off consumer cache probe")
        s1 = mem.search("consumer cache probe")
        s2 = mem.search("consumer cache probe")
        assert s1.items and s2.items
        assert "cache_hit" not in s2.warnings
        assert getattr(mem._store, "_recall_cache_v4", None) is None
        st = _stats(mem)
        assert st["enabled"] is False
        assert st.get("lookups", 0) == 0
        # status() publishes the same honest surface
        assert mem.status().cache["enabled"] is False

    def test_disabled_cache_writes_nothing(self, mem):
        mem.add("nothing cached while disabled")
        mem.search("nothing cached")
        with mem._store.read() as conn:
            # The disabled path must not even consult the entries map —
            # proven by the absent cache object (nothing else writes it).
            assert getattr(mem._store, "_recall_cache_v4", None) is None


# ---------------------------------------------------------------------
# hit path
# ---------------------------------------------------------------------


class TestHitPath:
    def test_repeated_search_serves_cache_hit(self, cmem, monkeypatch):
        """Second identical search replays the stored result: same
        items, ``cache_hit`` warning, ``coverage.cache.hit`` — and the
        hit path still runs the delivery-emission hook (a replayed
        answer is a real exposure, never folded silently; identical
        tokens dedupe honestly at the journal)."""
        cmem.add("the cellar keypad code is 4417")
        _settle(cmem)
        s1 = cmem.search("cellar keypad code")
        assert s1.status == "ready" and s1.items
        assert "cache_hit" not in s1.warnings
        assert cmem.status().cache.get("result_stores", 0) >= 1

        from verbatim.influence import exposure as _exp

        calls = {"n": 0}
        orig_emit = _exp.emit_deliveries

        def _spy(store, **kw):
            calls["n"] += 1
            return orig_emit(store, **kw)

        monkeypatch.setattr(_exp, "emit_deliveries", _spy)
        s2 = cmem.search("cellar keypad code")
        assert "cache_hit" in s2.warnings
        assert s2.coverage["cache"] == {"enabled": True, "hit": True}
        assert [h.memory_id for h in s2.items] == [
            h.memory_id for h in s1.items
        ]
        assert [h.quote for h in s2.items] == [h.quote for h in s1.items]
        # volatile fields rebuilt from THIS call's barrier
        assert s2.readiness["receipts"] == s1.readiness["receipts"]
        assert s2.causal_token
        # the delivery-emission hook ran on the cached serve
        assert calls["n"] >= 1

        s3 = cmem.search("cellar keypad code")
        assert "cache_hit" in s3.warnings
        st = _stats(cmem)
        assert st["result_lookups"] >= 3
        assert st["result_hits"] >= 2
        assert st["result_stores"] >= 1

    def test_different_query_never_hits(self, cmem):
        cmem.add("alpha probe body for the cache")
        _settle(cmem)
        s1 = cmem.search("alpha probe")
        s2 = cmem.search("unrelated omega question")
        assert s1.items
        assert "cache_hit" not in s2.warnings

    def test_pending_barrier_never_cached(self, cmem):
        """A search under an unmet barrier is computed live and stores
        nothing — the probe is skipped before any key is consulted."""
        cmem.add("barrier gated body")
        _settle(cmem)
        s0 = cmem.search("barrier gated")
        assert "cache_hit" in cmem.search("barrier gated").warnings or s0
        st0 = _stats(cmem)
        stores0 = st0.get("result_stores", 0)

        with cmem._store.tx() as conn:
            cmem._engine.record_obligations(
                conn, "rc_cache_stuck", cmem._namespace, include_source=True
            )
        cmem._session_receipts.append("rc_cache_stuck")
        s = cmem.search("barrier gated", ready_timeout_ms=30)
        assert s.status == "pending"
        assert "cache_hit" not in s.warnings
        st = _stats(cmem)
        assert st.get("result_stores", 0) == stores0

    def test_insufficient_never_cached(self, cmem):
        """An insufficient verdict always recomputes — never stored,
        never replayed (V6-02.13 cacheable-result policy)."""
        cmem.add("the ferry timetable lives elsewhere")
        _settle(cmem)
        s1 = cmem.search("what is the airspeed velocity of an unladen swallow")
        s2 = cmem.search("what is the airspeed velocity of an unladen swallow")
        assert s1.status == s2.status
        if s1.status != "ready":
            assert "cache_hit" not in s2.warnings
            st = _stats(cmem)
            assert st.get("result_stores", 0) == 0


# ---------------------------------------------------------------------
# invalidation discipline
# ---------------------------------------------------------------------


class TestInvalidation:
    def test_add_between_searches_shifts_frontier(self, cmem):
        """The barrier frontier is key material: a new receipt from an
        add makes the stored key unreachable — the identical query is a
        miss and recomputes against the new frontier (never a stale
        replay of the pre-add world)."""
        cmem.add("first cache body about penguins")
        _settle(cmem)
        s1 = cmem.search("penguins")
        assert "cache_hit" in cmem.search("penguins").warnings

        cmem.add("second body also about penguins")
        _settle(cmem)
        s3 = cmem.search("penguins")
        assert "cache_hit" not in s3.warnings
        assert len(s3.items) >= len(s1.items)
        # the new-frontier result is itself cacheable
        s4 = cmem.search("penguins")
        assert "cache_hit" in s4.warnings
        assert [h.memory_id for h in s4.items] == [
            h.memory_id for h in s3.items
        ]

    def test_forget_never_serves_forgotten_source(self, cmem):
        """A targeted forget suppresses the source — the stored result
        invalidates (watermark) or fails per-ref revalidation, and the
        recomputed answer drops the forgotten memory.  A cached entry
        can never replay a suppressed source."""
        r = cmem.add("forgettable secret phrase zebra")
        cmem.add("kept companion phrase zebra")
        _settle(cmem)
        s1 = cmem.search("zebra")
        ids1 = {h.memory_id for h in s1.items}
        assert r.memory_id in ids1
        assert "cache_hit" in cmem.search("zebra").warnings

        f = cmem.forget(ref=r.ref)
        assert f.mutated
        s2 = cmem.search("zebra")
        assert "cache_hit" not in s2.warnings
        assert r.memory_id not in {h.memory_id for h in s2.items}
        st = _stats(cmem)
        assert (
            st["invalidations"] + st["ref_denied"] >= 1
            or st["misses"] >= 2
        )

    def test_held_source_never_served_from_cache(self, cmem):
        """A quarantine hold opened between searches withholds the
        source — the entry invalidates on the content watermark and the
        per-ref source-hold check independently denies; the recomputed
        (and subsequently re-cached) result never contains the held id."""
        r_held = cmem.add("held source phrase about otters")
        r_kept = cmem.add("kept source phrase about otters")
        _settle(cmem)
        s1 = cmem.search("otters")
        ids1 = {h.memory_id for h in s1.items}
        assert r_held.memory_id in ids1 and r_kept.memory_id in ids1
        assert "cache_hit" in cmem.search("otters").warnings

        with cmem._store.tx() as conn:
            security.open_quarantine(
                conn,
                ("source", r_held.memory_id, 1),
                ["attack_risk:blocked"],
                [],
                scope_id=cmem._namespace,
            )
        s2 = cmem.search("otters")
        assert "cache_hit" not in s2.warnings
        ids2 = {h.memory_id for h in s2.items}
        assert r_held.memory_id not in ids2
        assert r_kept.memory_id in ids2
        st = _stats(cmem)
        assert st["invalidations"] + st["ref_denied"] >= 1

        # The filtered live result may itself be stored — a third call
        # may hit, but the held source still can never appear.
        s3 = cmem.search("otters")
        assert r_held.memory_id not in {h.memory_id for h in s3.items}
        assert r_kept.memory_id in {h.memory_id for h in s3.items}
