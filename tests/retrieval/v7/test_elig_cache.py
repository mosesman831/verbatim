"""V8-14.01 eligibility snapshot cache — scenarios K79/K80 (SPEC_V8 §14,
§23 ``elig.cache_size`` prior 32, §24/§25).

The materialized ``_Eligible`` snapshot is cached per process under the
FULL input identity: ``(db identity, scope id, caller principal,
purpose, projection generation)`` plus one watermark per governance /
content table the snapshot reads — ``scopes.authz_revision`` (the grant
epoch), ``grants_v3``, ``delegations``, ``quarantine``, ``purges`` +
``purge_targets``, ``source_state``, and the append-plane rowids.

These tests run against the real schema chain (the same DDL ``Store``
applies: V1+V2+V3+V4+V5+``ensure_v7_additive``) with real
``governance`` grants, real ``security.quarantine`` holds, and real
``revocation.revoke_grant`` — the cache is the production
``_SNAP_LRU``/``_snap_lookup``/``_snap_store`` machinery, never a fake.

Covered:

* K79 — two identical ``make_eligible`` calls: the second is a cache
  hit (``stats["cache"] == "hit"``), serves identical verdicts, and the
  measured hit path costs ≤ 1 ms (the §14 target);
* key dimensions — purpose, principal, generation, and scope each key
  the entry (a changed dimension misses);
* K80 — a grant revocation between two identical calls misses the
  cache and the fresh authorization denies
  (``NOT_FOUND_OR_UNAUTHORIZED``);
* governance writes — quarantine hold, suppressing purge, source-state
  change — move the watermark vector, so the next call misses and the
  fresh verdict applies;
* fail-closed key — an unreadable key component bypasses the cache
  entirely (``cache_note`` names it) and still builds a fresh snapshot;
* bounded LRU — the ``elig.cache_size`` arm (prior 32) evicts the
  least-recently-used entry; ``0`` disables;
* immutability/no-payloads — entries are ``_SnapEntry`` verdict data
  only (frozensets + (source, revision) pairs), copied per hit so
  per-call stats never cross searches;
* wholesale invalidation — ``elig_cache_clear()`` drops every entry.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.time import now_us  # noqa: E402
from verbatim.core.types import ErrorCode, VerbatimError  # noqa: E402
from verbatim.core.types_v3 import Verb  # noqa: E402
from verbatim import governance  # noqa: E402
from verbatim.governance.revocation import revoke_grant  # noqa: E402
from verbatim.retrieval.v7.eligibility import (  # noqa: E402
    ELIG_CACHE_SIZE_ARM,
    ELIG_CACHE_SIZE_DEFAULT,
    _SNAP_LRU,
    _SnapEntry,
    elig_cache_clear,
    elig_cache_stats,
    make_eligible,
)
from verbatim.security import quarantine as quar  # noqa: E402
from verbatim.storage.schema import (  # noqa: E402
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
)
from verbatim.storage.schema_v3 import (  # noqa: E402
    DDL_V3,
    DDL_V3_ALTER,
    DDL_V3_JOBS_REBUILD,
    DDL_V3_PROCEDURES_REBUILD,
)
from verbatim.storage.schema_v4 import v4_statements  # noqa: E402
from verbatim.storage.schema_v5 import (  # noqa: E402
    DDL_V5_JOBS_REBUILD,
    v5_statements,
)
from verbatim.storage.schema_v7 import ensure_v7_additive  # noqa: E402
from verbatim.storage.store import _split_alters  # noqa: E402

SC = "ns-alpha"
SC_B = "ns-beta"
ALICE = "alice"
BOB = "bob"


# ---------------------------------------------------------------------------
# mirror store — the real schema chain, in memory (same pattern as
# test_eligibility.py; the cache never sees a mock)
# ---------------------------------------------------------------------------


def _apply_real_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL_V1)
    conn.executescript(DDL_V2)
    for s in _split_alters(DDL_V2_ALTER):
        conn.execute(s)
    for s in _split_alters(DDL_V2_JOBS_REBUILD):
        conn.execute(s)
    conn.executescript(DDL_V3)
    for s in _split_alters(DDL_V3_ALTER):
        conn.execute(s)
    for s in _split_alters(DDL_V3_PROCEDURES_REBUILD):
        conn.execute(s)
    for s in _split_alters(DDL_V3_JOBS_REBUILD):
        conn.execute(s)
    for s in _split_alters(DDL_V5_JOBS_REBUILD):
        conn.execute(s)
    for s in v4_statements():
        conn.execute(s)
    for s in v5_statements():
        conn.execute(s)
    ensure_v7_additive(conn)


class _Db:
    """Thin store shim over a single :memory: connection."""

    def __init__(self, *, schema: bool = True) -> None:
        self._conn = sqlite3.connect(
            ":memory:", isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        if schema:
            _apply_real_schema(self._conn)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    class _Tx:
        def __init__(self, c: sqlite3.Connection) -> None:
            self._c = c

        def __enter__(self) -> sqlite3.Connection:
            return self._c

        def __exit__(self, *exc) -> bool:
            return False

    def tx(self):
        return _Db._Tx(self._conn)

    def read(self):
        return _Db._Tx(self._conn)

    def close(self) -> None:
        self._conn.close()


@pytest.fixture()
def db():
    d = _Db()
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean_cache():
    """The snapshot LRU is process-global — isolate it per test."""
    elig_cache_clear()
    yield
    elig_cache_clear()


# ---------------------------------------------------------------------------
# seed helpers — real governance/security rows
# ---------------------------------------------------------------------------


def _grant(conn, scope_id: str = SC, principal: str = ALICE,
           purposes=None, expires_us=None) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO scopes(scope_id, profile_id, principal_id,"
        " visibility) VALUES (?,?,?,'owner')",
        (scope_id, "prof", principal),
    )
    governance.register_principal(conn, kind="human",
                                  principal_id=principal)
    governance.seed_purposes(conn)
    return governance.create_grant(
        conn,
        scope_id=scope_id,
        principal_id=principal,
        verbs=[Verb.READ.value],
        issuer_id="root",
        purposes=purposes,
        expires_us=expires_us,
        strict=False,
    )


def _source(conn, source_id: str = "src-1", scope_id: str = SC) -> None:
    conn.execute(
        "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, provenance)"
        " VALUES (?,?,?,x'00',1,1,'direct_user')",
        (source_id, 1, b"hello world"),
    )


def _unit(conn, unit_id: str, source_id: str = "src-1",
          scope_id: str = SC, generation: int = 1) -> None:
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " generation) VALUES (?,?,?,?,?,?)",
        (unit_id, source_id, 1, scope_id, "turn", generation),
    )


def _seeded(db: _Db, scope_id: str = SC, principal: str = ALICE) -> str:
    """One scope, one read grant, one source revision, one unit."""
    gid = _grant(db.conn, scope_id, principal)
    _source(db.conn, "src-1", scope_id)
    _unit(db.conn, "u7:one", "src-1", scope_id, 1)
    return gid


def _mk(db: _Db, scope_id: str = SC, generation: int = 1,
        principal: str = ALICE, purpose: str = "recall", **kw):
    return make_eligible(
        db.conn, db, scope_id=scope_id, generation=generation,
        principal_id=principal, purpose=purpose, **kw)


# ---------------------------------------------------------------------------
# K79 — hit path
# ---------------------------------------------------------------------------


class TestK79HitPath:
    def test_k79_second_call_hits_with_identical_verdicts(self, db):
        _seeded(db)
        e1 = _mk(db)
        assert e1.stats["cache"] == "miss"
        assert e1.stats["cache_keyed"] is True
        assert e1("u7:one") is True

        e2 = _mk(db)
        assert e2.stats["cache"] == "hit"
        assert e2("u7:one") is True
        assert e2("u7:ghost") is False
        assert e2.unit_ids == e1.unit_ids

        st = elig_cache_stats()
        assert st["hits"] >= 1 and st["inserts"] >= 1
        assert st["entries"] >= 1

    def test_k79_hit_cost_within_1ms(self, db):
        _seeded(db)
        _mk(db)  # warm — the miss builds + stores the snapshot
        samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            e = _mk(db)
            samples.append((time.perf_counter() - t0) * 1000.0)
            assert e.stats["cache"] == "hit"
        samples.sort()
        p50 = samples[len(samples) // 2]
        p95 = samples[int(len(samples) * 0.95)]
        # V8-14.01 target: hit cost ≤ 1 ms.  The hit path is one
        # watermark-vector probe + one LRU lookup; on this in-memory
        # store p50 sits in the tens of microseconds.  We assert p50 ≤
        # 1 ms (the spec bound) and report p95 for diagnosis — a loaded
        # CI box must not turn a scheduling blip into a red suite.
        assert p50 <= 1.0, f"hit p50 {p50:.3f} ms > 1 ms (p95 {p95:.3f})"

    def test_k79_key_dimensions_each_miss(self, db):
        """purpose / principal / generation / scope are all in the key —
        changing any one misses rather than aliasing another entry."""
        _grant(db.conn, SC, ALICE)
        _grant(db.conn, SC_B, BOB)
        _source(db.conn, "src-1", SC)
        _unit(db.conn, "u7:one", "src-1", SC, 1)
        _source(db.conn, "src-2", SC_B)
        _unit(db.conn, "u7:two", "src-2", SC_B, 1)

        base = _mk(db)
        assert base.stats["cache"] == "miss"
        assert _mk(db).stats["cache"] == "hit"          # identical → hit

        # Different purpose → different key → miss + fresh snapshot.
        assert _mk(db, purpose="evaluate").stats["cache"] == "miss"
        # Different generation → miss.
        assert _mk(db, generation=2).stats["cache"] == "miss"
        # Different scope → miss (and a different verdict universe).
        e_b = _mk(db, scope_id=SC_B, principal=BOB)
        assert e_b.stats["cache"] == "miss"
        assert e_b("u7:two") is True
        assert e_b("u7:one") is False

    def test_k79_append_write_moves_watermark(self, db):
        """A new unit insert moves the append watermark — content drift
        is a miss, never a stale hit."""
        _seeded(db)
        e1 = _mk(db)
        assert e1("u7:two") is False  # not yet written
        _unit(db.conn, "u7:two", "src-1", SC, 1)
        e2 = _mk(db)
        assert e2.stats["cache"] == "miss"
        assert e2("u7:two") is True

    def test_k79_expired_entry_is_a_miss(self, db, monkeypatch):
        """A live grant's ``expires_us`` bounds the entry's
        ``not_after_us`` — once the earliest passive-flip instant
        arrives the cached snapshot is evicted and the call misses."""
        import verbatim.retrieval.v7.eligibility as elig

        expiry = now_us() + 60_000_000      # +60 s — valid at issue
        _grant(db.conn, SC, ALICE, expires_us=expiry)
        _source(db.conn, "src-1", SC)
        _unit(db.conn, "u7:one", "src-1", SC, 1)
        e1 = _mk(db)
        assert e1.stats["cache"] == "miss"
        e1b = _mk(db)
        assert e1b.stats["cache"] == "hit"   # bound not yet reached

        # Advance the module clock past the earliest passive-flip bound —
        # the real ``now_us`` seam, patched, not a fake clock object.
        monkeypatch.setattr(
            elig, "now_us", lambda: expiry + 60_000_000)
        expired0 = elig_cache_stats()["expired"]
        # The entry is evicted as expired — a miss, never a stale hit —
        # and a fresh snapshot is built under the caller's pin.
        e2 = _mk(db)
        assert e2.stats["cache"] == "miss"
        assert elig_cache_stats()["expired"] == expired0 + 1
        assert e2("u7:one") is True


# ---------------------------------------------------------------------------
# K80 — revocation ⇒ miss and deny; governance writes move the key
# ---------------------------------------------------------------------------


class TestK80GovernanceInvalidation:
    def test_k80_revocation_misses_and_denies(self, db):
        gid = _seeded(db)
        e1 = _mk(db)
        assert e1("u7:one") is True
        st0 = elig_cache_stats()

        receipt = revoke_grant(db.conn, gid)
        assert receipt["revoked"] is True

        # The revoked scope's epoch moved → the key differs → miss, and
        # fresh authorization denies with the typed governance error.
        with pytest.raises(VerbatimError) as ei:
            _mk(db)
        assert ei.value.code in (
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, ErrorCode.STALE_EPOCH)
        st1 = elig_cache_stats()
        assert st1["misses"] > st0["misses"]
        # No hit was served after the revocation — the stale entry is
        # unreachable under the moved key.
        assert st1["hits"] == st0["hits"]

    def test_k80_quarantine_hold_misses_and_denies_unit(self, db):
        _seeded(db)
        e1 = _mk(db)
        assert e1("u7:one") is True

        assert quar.open_quarantine(
            db.conn, ("unit", "u7:one", 1), ["rule:test"], [],
            scope_id=SC) is True
        e2 = _mk(db)
        assert e2.stats["cache"] == "miss"   # quarantine watermark moved
        assert e2("u7:one") is False
        assert e2.stats["denied_held"] >= 1

    def test_k80_suppressing_purge_misses_and_denies(self, db):
        _seeded(db)
        assert _mk(db)("u7:one") is True
        db.conn.execute(
            "INSERT INTO purges(purge_id, selection_digest, scope_id,"
            " state, requested_us) VALUES (?,x'00',?,?,1)",
            ("p1", SC, "suppressed"))
        db.conn.execute(
            "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
            " VALUES (?,?,?)", ("p1", "source", "src-1"))
        e2 = _mk(db)
        assert e2.stats["cache"] == "miss"   # purge watermark moved
        assert e2("u7:one") is False

    def test_k80_source_state_change_misses(self, db):
        _seeded(db)
        assert _mk(db)("u7:one") is True
        # Registering a source_state row moves its watermark; a foreign
        # namespace authority also flips the verdict itself.
        db.conn.execute(
            "INSERT INTO source_state(source_id, namespace,"
            " control_version, mutation_head, disposition, known_at,"
            " updated_at, producer) VALUES (?,?,1,'1','active','t','t','test')",
            ("src-1", "ns-elsewhere"))
        e2 = _mk(db)
        assert e2.stats["cache"] == "miss"
        assert e2("u7:one") is False
        assert e2.stats["denied_source"] >= 1

    def test_k80_second_grant_misses_then_hits(self, db):
        _seeded(db)
        _mk(db)
        _grant(db.conn, SC, BOB)     # grants_v3 watermark moves
        assert _mk(db).stats["cache"] == "miss"
        assert _mk(db).stats["cache"] == "hit"

    def test_k80_wholesale_clear(self, db):
        _seeded(db)
        _mk(db)
        assert elig_cache_stats()["entries"] >= 1
        dropped = elig_cache_clear()
        assert dropped >= 1
        assert elig_cache_stats()["entries"] == 0
        assert _mk(db).stats["cache"] == "miss"


# ---------------------------------------------------------------------------
# cache shape — bounded LRU, immutable entries, fail-closed key
# ---------------------------------------------------------------------------


class TestCacheShape:
    def test_lru_bound_evicts_oldest(self, db):
        """``elig.cache_size`` is a bounded LRU — ``cache_size=2`` keeps
        exactly two entries; the third key evicts the least-recent."""
        _grant(db.conn, SC, ALICE)
        _source(db.conn, "src-1", SC)
        _unit(db.conn, "u7:one", "src-1", SC, 1)

        _mk(db, purpose="p1", cache_size=2)
        _mk(db, purpose="p2", cache_size=2)
        _mk(db, purpose="p3", cache_size=2)
        st = elig_cache_stats()
        assert st["entries"] <= 2
        assert st["evictions"] >= 1
        # p1 was evicted → miss on re-request; p3 still resident → hit.
        assert _mk(db, purpose="p1", cache_size=2).stats["cache"] == "miss"
        assert _mk(db, purpose="p3", cache_size=2).stats["cache"] == "hit"

    def test_default_arm_is_32(self, db):
        _seeded(db)
        e = _mk(db)
        assert ELIG_CACHE_SIZE_DEFAULT == 32
        assert e.stats["cache_size"] == 32
        assert ELIG_CACHE_SIZE_ARM == "elig.cache_size"

    def test_cache_disabled(self, db):
        _seeded(db)
        e = _mk(db, cache_size=0)
        assert e.stats["cache"] == "miss"
        assert e.stats["cache_keyed"] is False
        assert e.stats["cache_note"] == "disabled"
        assert elig_cache_stats()["entries"] == 0
        # …and it still answers correctly (a fresh snapshot was built).
        assert e("u7:one") is True

    def test_unreadable_key_fails_closed(self, db):
        """A key component that cannot be read bypasses the cache —
        ``cache_note`` names the failure, no lookup is served, and a
        fresh snapshot is still built (fail closed, never partial)."""
        _seeded(db)

        class _KeyFailingConn(sqlite3.Connection):
            """Real connection whose multi-table ``sqlite_master`` key
            probe raises.  ``_cache_key`` probes the whole key surface
            in one ``IN (…, …)`` statement; ``has_table`` probes one
            name at a time — the failure is scoped to the key probe."""

            def execute(self, sql, parameters=()):
                if "sqlite_master" in str(sql) and len(parameters) > 3:
                    raise sqlite3.OperationalError("injected")
                return super().execute(sql, parameters)

        raw = _Db()          # real schema applied in __init__
        _seeded(raw)
        # Clone the seeded database into a key-failing connection.
        fail = _KeyFailingConn(":memory:", isolation_level=None,
                               check_same_thread=False)
        raw.conn.backup(fail)

        class _FailStore:
            conn = fail

        before = elig_cache_stats()["bypassed"]
        e = make_eligible(
            fail, _FailStore(), scope_id=SC, generation=1,
            principal_id=ALICE, purpose="recall")
        assert e.stats["cache_keyed"] is False
        assert e.stats["cache_note"].startswith("key_unreadable:")
        assert elig_cache_stats()["bypassed"] == before + 1
        assert elig_cache_stats()["entries"] == 0   # nothing stored
        # The fresh snapshot still answers correctly.
        assert e("u7:one") is True
        raw.close()
        fail.close()

    def test_entries_hold_verdict_data_only(self, db):
        """Cache entries are ``_SnapEntry`` — frozensets + (source_id,
        revision) pairs + a stats seed. No payload bytes, no live
        connection, no mutable verdict surface."""
        _seeded(db)
        _mk(db)
        assert _SNAP_LRU, "expected a stored entry"
        for ent in _SNAP_LRU.values():
            assert isinstance(ent, _SnapEntry)
            assert isinstance(ent.ok_pairs, frozenset)
            assert isinstance(ent.unit_holds, frozenset)
            assert isinstance(ent.purged, frozenset)
            assert isinstance(ent.unit_ids, frozenset)
            # units: {unit_id: {generation: (source_id, revision)}} —
            # identity pairs only, never payload bytes.
            for uid, gens in ent.units.items():
                assert isinstance(uid, str)
                for g, pair in gens.items():
                    assert isinstance(g, int)
                    assert isinstance(pair, tuple) and len(pair) == 2
                    assert all(
                        isinstance(x, (str, int)) or x is None
                        for x in pair)
            blob = repr(ent.units) + repr(ent.ok_pairs)
            assert "hello world" not in blob   # no payload text cached

    def test_hit_returns_fresh_stats(self, db):
        """Per-call counters never cross searches — the entry's stats
        seed is copied per hit (the seed is the construction-time
        snapshot, not a shared mutable dict)."""
        _seeded(db)
        e1 = _mk(db)
        e1("u7:one")
        e1("u7:one")
        assert e1.stats["calls"] == 2
        e2 = _mk(db)
        assert e2.stats["cache"] == "hit"
        assert e2.stats["calls"] == 0      # fresh counters on the hit
        e2("u7:one")
        assert e2.stats["calls"] == 1
        assert e1.stats["calls"] == 2      # the miss's stats unmoved
