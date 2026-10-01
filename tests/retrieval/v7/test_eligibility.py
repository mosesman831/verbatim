"""Tests for the V7 eligibility adapter — ``verbatim/retrieval/v7/eligibility.py``
(w-eligibility, wave B).

Two layers of proof:

* **Mirror-plane tests** run against the REAL store schema (the same
  ``DDL_V1`` + ``DDL_V2`` + ``DDL_V3`` + ``v4_statements()`` +
  ``v5_statements()`` + ``ensure_v7_additive`` chain ``Store.create``
  applies) on an in-memory connection, with real ``governance`` grants
  and real ``security.quarantine`` / purge rows — the verdict semantics
  are exercised against the true tables, never mocks.

* **Real-store proofs** run through ``verbatim.storage.store.Store`` +
  ``verbatim.memory.Memory``: ``mem.add`` performs the real ingest,
  ``jobs.units_jobs.project_units_v7`` performs the real units
  projection inside a write tx, and ``open_quarantine``/``release``/
  ``purge.suppress`` run the real security calls.

Coverage: scope + generation fences, source/source-revision existence,
the quarantine cascade (source / span / covering envelope / unit),
suppressing purge targets (source / exact source_revision / span),
purpose-aware construction authorization, retired principals,
revocation, deterministic verdicts, honest unavailable modes, and the
predicate's no-SQL-after-construction guarantee.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types import ErrorCode, VerbatimError  # noqa: E402
from verbatim.core.types_v3 import Verb  # noqa: E402
from verbatim.core.types_v4 import PurposeConstraint  # noqa: E402
from verbatim import governance  # noqa: E402
from verbatim.governance import CallerV3  # noqa: E402
from verbatim.retrieval.v7.eligibility import make_eligible  # noqa: E402
from verbatim.security import quarantine as quar  # noqa: E402
from verbatim.storage.repos import has_table  # noqa: E402
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
# mirror store — the real schema chain, in memory
# ---------------------------------------------------------------------------


def _apply_real_schema(conn: sqlite3.Connection) -> None:
    """The ``Store.create`` DDL chain (minus meta/bootstrap rows), plus the
    lazily-applied V7 additive plane — the real tables the adapter reads."""
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
    """Thin store shim over a single :memory: connection — ``tx``/``read``
    yield the same conn so ``store.read()`` snapshots compose like the
    real Store's thread-local reader reuse."""

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
def db() -> _Db:
    d = _Db()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# seed helpers — real governance/security/state rows
# ---------------------------------------------------------------------------


def _scope(conn, scope_id: str = SC, principal: str = ALICE) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO scopes(scope_id, profile_id, principal_id,"
        " visibility) VALUES (?,?,?,'owner')",
        (scope_id, "prof", principal),
    )


def _grant(
    conn,
    scope_id: str = SC,
    principal: str = ALICE,
    purposes=None,
) -> str:
    _scope(conn, scope_id, principal)
    governance.register_principal(
        conn, kind="human", principal_id=principal
    )
    governance.seed_purposes(conn)
    return governance.create_grant(
        conn,
        scope_id=scope_id,
        principal_id=principal,
        verbs=[Verb.READ.value],
        issuer_id="root",
        purposes=purposes,
        strict=False,
    )


def _source(
    conn, source_id: str = "src-1", scope_id: str = SC, *, payload: bytes = b"hello world", revision: int = 1
) -> None:
    conn.execute(
        "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id),
    )
    if payload is not None:
        conn.execute(
            "INSERT INTO source_revisions(source_id, revision, payload,"
            " payload_hmac, event_us, captured_us, provenance)"
            " VALUES (?,?,?,x'00',1,1,'direct_user')",
            (source_id, revision, payload),
        )


def _state(conn, source_id: str, namespace: str) -> None:
    conn.execute(
        "INSERT INTO source_state(source_id, namespace, control_version,"
        " mutation_head, disposition, known_at, updated_at, producer)"
        " VALUES (?,?,1,'1','active','t','t','test')",
        (source_id, namespace),
    )


def _unit(
    conn,
    unit_id: str,
    source_id: str = "src-1",
    revision: int = 1,
    scope_id: str = SC,
    generation: int = 1,
    kind: str = "turn",
) -> None:
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " generation) VALUES (?,?,?,?,?,?)",
        (unit_id, source_id, revision, scope_id, kind, generation),
    )


def _span(conn, span_id: str, source_id: str = "src-1", revision: int = 1) -> None:
    conn.execute(
        "INSERT INTO spans(span_id, source_id, revision, start_byte,"
        " end_byte, excerpt_hmac, harvester_version)"
        " VALUES (?,?,?,0,5,x'00','h/1')",
        (span_id, source_id, revision),
    )


def _envelope(
    conn, envelope_id: str, source_id: str = "src-1", revision: int = 1,
    scope_id: str = SC,
) -> None:
    conn.execute(
        "INSERT INTO source_envelopes(envelope_id, source_id, revision,"
        " scope_id, envelope_kind, receipt_us) VALUES (?,?,?,?,?,1)",
        (envelope_id, source_id, revision, scope_id, "user_message"),
    )


def _purge(conn, purge_id: str, state: str, targets: list, scope_id: str = SC) -> None:
    conn.execute(
        "INSERT INTO purges(purge_id, selection_digest, scope_id, state,"
        " requested_us) VALUES (?,x'00',?,?,1)",
        (purge_id, scope_id, state),
    )
    for kind, oid in targets:
        conn.execute(
            "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
            " VALUES (?,?,?)",
            (purge_id, kind, oid),
        )


def _mk(db: _Db, scope_id: str = SC, generation: int = 1, principal: str = ALICE,
        purpose: str = "recall"):
    return make_eligible(
        db.conn,
        db,
        scope_id=scope_id,
        generation=generation,
        principal_id=principal,
        purpose=purpose,
    )


def _seeded(db: _Db) -> None:
    """One scope, one read grant, one source revision, one unit."""
    _grant(db.conn)
    _source(db.conn, "src-1", SC)
    _unit(db.conn, "u7:one", "src-1", 1, SC, 1)


# ---------------------------------------------------------------------------
# happy path + input forms
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_unit_row_mapping_eligible(self, db):
        _seeded(db)
        el = _mk(db)
        row = {
            "unit_id": "u7:one", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "generation": 1, "kind": "turn",
        }
        assert el(row) is True
        assert el.stats["mode"] == "ok"

    def test_unit_id_string_eligible(self, db):
        _seeded(db)
        el = _mk(db)
        assert el("u7:one") is True

    def test_protocol_forms_agree(self, db):
        _seeded(db)
        el = _mk(db)
        assert "u7:one" in el
        assert el.is_eligible("u7:one") is True
        assert el.contains("u7:one") is True
        assert el.eligible("u7:one") is True

    def test_attribute_object_accepted(self, db):
        """CandidateV7-style objects (pipeline S3 re-check) resolve via
        attributes — the units row is the identity authority."""
        _seeded(db)
        el = _mk(db)
        from verbatim.core.types_v7 import CandidateV7

        cand = CandidateV7(
            unit_id="u7:one", source_id="src-1", revision=1,
            lane="lex", rank=1, raw_score=1.0,
        )
        assert el(cand) is True

    def test_stats_shape(self, db):
        _seeded(db)
        el = _mk(db)
        for key in ("mode", "checked_sources", "held_sources",
                    "missing_unit_rows"):
            assert key in el.stats
        assert el.stats["checked_sources"] == 1
        assert el.stats["held_sources"] == 0
        assert el.stats["units_fenced"] == 1

    def test_sqlite3_row_form(self, db):
        """A ``sqlite3.Row`` (keys-protocol mapping, not a ``Mapping``
        instance) resolves like a dict — lanes may hand rows straight
        off a cursor."""
        _seeded(db)
        el = _mk(db)
        row_conn = sqlite3.connect(":memory:")
        row_conn.row_factory = sqlite3.Row
        row_conn.execute(
            "CREATE TABLE t(unit_id, source_id, revision, scope_id,"
            " generation)"
        )
        row_conn.execute(
            "INSERT INTO t VALUES ('u7:one','src-1',1,?,1)", (SC,)
        )
        try:
            row = row_conn.execute("SELECT * FROM t").fetchone()
            assert el(row) is True
        finally:
            row_conn.close()

    def test_plain_tuple_denied(self, db):
        """A bare tuple carries no field names — unrecognized input
        fails closed."""
        _seeded(db)
        el = _mk(db)
        assert el(("u7:one", "src-1", 1, SC, 1)) is False


# ---------------------------------------------------------------------------
# scope + generation fences
# ---------------------------------------------------------------------------


class TestFences:
    def test_scope_isolation_other_scope_unit(self, db):
        """A unit stamped under scope B never surfaces for a scope-A
        predicate — it is not in the fenced universe at all."""
        _grant(db.conn)
        _scope(db.conn, SC_B)
        _source(db.conn, "src-1", SC_B)
        _unit(db.conn, "u7:foreign", "src-1", 1, SC_B, 1)
        el = _mk(db, scope_id=SC)
        assert el("u7:foreign") is False
        assert el.stats["missing_unit_rows"] == 1

    def test_scope_isolation_claimed_scope(self, db):
        """A row asserting a foreign scope is denied outright even when the
        unit is otherwise visible."""
        _seeded(db)
        el = _mk(db, scope_id=SC)
        row = {
            "unit_id": "u7:one", "source_id": "src-1", "revision": 1,
            "scope_id": SC_B, "generation": 1,
        }
        assert el(row) is False
        assert el.stats["denied_scope"] == 1

    def test_foreign_source_denied(self, db):
        """In-scope unit row pointing at an out-of-scope source: the
        source-chain check fails closed (cross-partition leak guard)."""
        _grant(db.conn)
        _scope(db.conn, SC_B)
        _source(db.conn, "src-x", SC_B)
        _unit(db.conn, "u7:leak", "src-x", 1, SC, 1)
        el = _mk(db, scope_id=SC)
        assert el("u7:leak") is False
        assert el.stats["denied_source"] >= 1

    def test_namespace_authority_overrides_partition(self, db):
        """``source_state.namespace`` is the scope authority — a source in
        a foreign partition whose authority is our scope still gates the
        same way the write path stamped it."""
        _grant(db.conn)
        _scope(db.conn, SC_B)
        _source(db.conn, "src-ns", SC_B)
        _state(db.conn, "src-ns", SC)  # authority is SC, partition is SC_B
        _unit(db.conn, "u7:ns", "src-ns", 1, SC, 1)
        el = _mk(db, scope_id=SC)
        assert el("u7:ns") is True

    def test_generation_fence_hides_newer(self, db):
        _seeded(db)
        _unit(db.conn, "u7:g2", "src-1", 1, SC, 2)
        el1 = _mk(db, generation=1)
        assert el1("u7:g2") is False
        assert el1.stats["missing_unit_rows"] == 1
        el2 = _mk(db, generation=2)
        assert el2("u7:g2") is True

    def test_generation_field_over_fence_denied(self, db):
        """A row asserting generation above the pin is denied even when a
        same-id row exists at an older generation."""
        _seeded(db)
        el = _mk(db, generation=1)
        row = {
            "unit_id": "u7:one", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "generation": 2,
        }
        assert el(row) is False
        assert el.stats["denied_generation"] == 1

    def test_stale_generation_row_resolves_latest(self, db):
        """Stale-generation nomination (V7-30.02): a row asserting an
        in-fence generation resolves to the latest visible row's verdict."""
        _seeded(db)
        _unit(db.conn, "u7:one", "src-1", 1, SC, 2)  # same unit_id, gen 2
        el = _mk(db, generation=2)
        row = {
            "unit_id": "u7:one", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "generation": 1,
        }
        assert el(row) is True

    def test_missing_unit_row_denied(self, db):
        _seeded(db)
        el = _mk(db)
        assert el("u7:ghost") is False
        assert el.stats["missing_unit_rows"] == 1

    def test_vanished_source_row_denied(self, db):
        """Unit rows whose sources row is absent fail closed — the store
        row can claim anything, the source chain must exist."""
        _grant(db.conn)
        db.conn.execute(
            "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
            " created_us) VALUES ('src-1','t','user_message',?,1)", (SC,))
        db.conn.execute(
            "INSERT INTO source_revisions(source_id, revision, payload,"
            " payload_hmac, event_us, captured_us, provenance)"
            " VALUES ('src-1',1,x'6869',x'00',1,1,'direct_user')")
        _unit(db.conn, "u7:one", "src-1", 1, SC, 1)
        db.conn.execute("PRAGMA foreign_keys = OFF")
        db.conn.execute("DELETE FROM sources WHERE source_id = 'src-1'")
        db.conn.execute("PRAGMA foreign_keys = ON")
        el = _mk(db)
        assert el("u7:one") is False

    def test_vanished_source_revision_denied(self, db):
        _grant(db.conn)
        db.conn.execute(
            "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
            " created_us) VALUES ('src-1','t','user_message',?,1)", (SC,))
        _unit(db.conn, "u7:one", "src-1", 1, SC, 1)
        el = _mk(db)
        assert el("u7:one") is False  # no source_revisions row at all

    def test_emptied_payload_denied(self, db):
        """An emptied (purged) payload reads as absent — the bytes the
        unit pins are gone."""
        _grant(db.conn)
        _source(db.conn, "src-1", SC, payload=b"")
        _unit(db.conn, "u7:one", "src-1", 1, SC, 1)
        el = _mk(db)
        assert el("u7:one") is False


# ---------------------------------------------------------------------------
# quarantine cascade (V3-14.10)
# ---------------------------------------------------------------------------


class TestQuarantineCascade:
    def test_held_source_hides_units(self, db):
        _seeded(db)
        assert quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["attack_risk:blocked"], [],
            scope_id=SC,
        ) is True
        el = _mk(db)
        assert el("u7:one") is False
        assert el.stats["held_sources"] == 1
        assert el.stats["denied_source"] >= 1

    def test_held_span_hides_units(self, db):
        _seeded(db)
        _span(db.conn, "sp-1", "src-1", 1)
        quar.open_quarantine(
            db.conn, ("span", "sp-1", 1), ["rule:test"], [], scope_id=SC,
        )
        el = _mk(db)
        assert el("u7:one") is False

    def test_held_envelope_cascades(self, db):
        """A hold on a covering source_envelope withholds every unit whose
        bytes the envelope covers."""
        _seeded(db)
        _envelope(db.conn, "env-1", "src-1", 1, SC)
        quar.open_quarantine(
            db.conn, ("source_envelope", "env-1", 1), ["rule:test"], [],
            scope_id=SC,
        )
        el = _mk(db)
        assert el("u7:one") is False

    def test_unit_level_hold_denies(self, db):
        _seeded(db)
        quar.open_quarantine(
            db.conn, ("unit", "u7:one", 1), ["rule:test"], [], scope_id=SC,
        )
        el = _mk(db)
        assert el("u7:one") is False
        assert el.stats["denied_held"] == 1

    def test_released_source_visible(self, db):
        """``released`` restores visibility — only pending/suppressed
        hide."""
        _seeded(db)
        quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["rule:test"], [], scope_id=SC,
        )
        quar.release(db.conn, ("source", "src-1", 1), decided_by="rev-1",
                     scope_id=SC)
        el = _mk(db)
        assert el("u7:one") is True

    def test_suppressed_state_hides(self, db):
        _seeded(db)
        quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["rule:test"], [], scope_id=SC,
        )
        quar.suppress(db.conn, ("source", "src-1", 1), decided_by="rev-1",
                      scope_id=SC)
        el = _mk(db)
        assert el("u7:one") is False

    def test_purged_quarantine_tombstone_not_a_hold(self, db):
        """``purged`` quarantine rows are tombstones — privacy's own
        suppression layer (purge targets) is what hides purged content,
        so a bare purged-mark hold does not withhold by itself."""
        _seeded(db)
        quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["rule:test"], [], scope_id=SC,
        )
        quar.mark_purged(db.conn, ("source", "src-1", 1), decided_by="rev-1",
                         scope_id=SC)
        el = _mk(db)
        assert el("u7:one") is True

    def test_unrelated_hold_does_not_hide(self, db):
        _seeded(db)
        _source(db.conn, "src-2", SC)
        quar.open_quarantine(
            db.conn, ("source", "src-2", 1), ["rule:test"], [], scope_id=SC,
        )
        el = _mk(db)
        assert el("u7:one") is True


# ---------------------------------------------------------------------------
# purge / deletion tombstones
# ---------------------------------------------------------------------------


class TestPurge:
    def test_purged_source_invisible(self, db):
        _seeded(db)
        _purge(db.conn, "p1", "suppressed", [("source", "src-1")])
        el = _mk(db)
        assert el("u7:one") is False

    def test_purged_source_revision_invisible(self, db):
        """Exact ``<source_id>:<revision>`` target form."""
        _seeded(db)
        _purge(db.conn, "p1", "suppressed",
               [("source_revision", "src-1:1")])
        el = _mk(db)
        assert el("u7:one") is False

    def test_other_revision_survives(self, db):
        """Revision-scoped erasure is exact — rev 2 of the same source
        stays visible."""
        _seeded(db)
        db.conn.execute(
            "INSERT INTO source_revisions(source_id, revision, payload,"
            " payload_hmac, event_us, captured_us, provenance)"
            " VALUES ('src-1',2,x'6869',x'00',1,1,'direct_user')")
        _unit(db.conn, "u7:two", "src-1", 2, SC, 1)
        _purge(db.conn, "p1", "suppressed",
               [("source_revision", "src-1:1")])
        el = _mk(db)
        assert el("u7:one") is False
        assert el("u7:two") is True

    def test_purging_state_invisible(self, db):
        _seeded(db)
        _purge(db.conn, "p1", "purging", [("source", "src-1")])
        el = _mk(db)
        assert el("u7:one") is False

    def test_completed_purge_invisible(self, db):
        _seeded(db)
        _purge(db.conn, "p1", "completed", [("source", "src-1")])
        el = _mk(db)
        assert el("u7:one") is False

    def test_previewed_purge_does_not_hide(self, db):
        _seeded(db)
        _purge(db.conn, "p1", "previewed", [("source", "src-1")])
        el = _mk(db)
        assert el("u7:one") is True

    def test_purged_span_empties_parent_revision(self, db):
        """``purge._erase_span`` empties the WHOLE parent revision — a
        span tombstone withholds every unit pinned to that revision."""
        _seeded(db)
        _span(db.conn, "sp-1", "src-1", 1)
        _purge(db.conn, "p1", "suppressed", [("span", "sp-1")])
        el = _mk(db)
        assert el("u7:one") is False


# ---------------------------------------------------------------------------
# source-level candidates (the source lane's synthesized rows)
# ---------------------------------------------------------------------------


class TestSourceLevel:
    def test_source_level_row_eligible(self, db):
        """The source lane's synthesized ``"{sid}:{rev}"`` rows resolve
        through the claimed (source_id, revision) — gated by the same
        source-chain verdicts."""
        _seeded(db)
        el = _mk(db)
        row = {
            "unit_id": "src-1:1", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "kind": "source",
        }
        assert el(row) is True

    def test_source_level_held_source_denied(self, db):
        _seeded(db)
        quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["rule:test"], [], scope_id=SC,
        )
        el = _mk(db)
        row = {
            "unit_id": "src-1:1", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "kind": "source",
        }
        assert el(row) is False

    def test_source_level_unknown_source_denied(self, db):
        _seeded(db)
        el = _mk(db)
        row = {
            "unit_id": "ghost:1", "source_id": "ghost", "revision": 1,
            "scope_id": SC, "kind": "source",
        }
        assert el(row) is False

    def test_claimed_source_cannot_override_unit_identity(self, db):
        """The units row is the identity authority: a candidate claiming
        unit u7:one (real, on held src-1) while pointing at clean src-2
        still answers by the REAL source chain — the forge gains nothing."""
        _seeded(db)
        _source(db.conn, "src-2", SC)
        quar.open_quarantine(
            db.conn, ("source", "src-1", 1), ["rule:test"], [], scope_id=SC,
        )
        el = _mk(db)
        forged = {
            "unit_id": "u7:one", "source_id": "src-2", "revision": 1,
            "scope_id": SC, "generation": 1,
        }
        assert el(forged) is False


# ---------------------------------------------------------------------------
# authorization: purpose, principals, epochs
# ---------------------------------------------------------------------------


class TestAuthorization:
    def test_no_grant_denies(self, db):
        _seeded(db)
        with pytest.raises(VerbatimError) as ei:
            make_eligible(
                db.conn, db, scope_id=SC, generation=1,
                principal_id=BOB, purpose="recall",
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_no_scope_at_all_denies(self, db):
        """Absent and forbidden are publicly indistinguishable."""
        with pytest.raises(VerbatimError) as ei:
            make_eligible(
                db.conn, db, scope_id="ns-nowhere", generation=1,
                principal_id=BOB, purpose="recall",
            )
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_purpose_constrained_grant(self, db):
        """Purpose-aware construction: a grant SET-constrained to
        ``review`` cannot mint a ``recall`` predicate."""
        _scope(db.conn)
        governance.register_principal(db.conn, kind="human", principal_id=ALICE)
        governance.seed_purposes(db.conn)
        governance.create_grant(
            db.conn, scope_id=SC, principal_id=ALICE,
            verbs=[Verb.READ.value], issuer_id="root",
            purposes=PurposeConstraint.set({"review"}), strict=False,
        )
        _source(db.conn, "src-1", SC)
        _unit(db.conn, "u7:one", "src-1", 1, SC, 1)
        with pytest.raises(VerbatimError) as ei:
            _mk(db, purpose="recall")
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        el = _mk(db, purpose="review")
        assert el("u7:one") is True

    def test_purpose_any_grant_allows_recall(self, db):
        _seeded(db)
        el = _mk(db, purpose="recall")
        assert el("u7:one") is True

    def test_retired_principal_denied(self, db):
        _seeded(db)
        governance.retire_principal(db.conn, ALICE)
        with pytest.raises(VerbatimError) as ei:
            _mk(db)
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_revoked_grant_denies(self, db):
        _seeded(db)
        gid = governance.create_grant(
            db.conn, scope_id=SC, principal_id=BOB,
            verbs=[Verb.READ.value], issuer_id="root", strict=False,
        )
        assert _mk(db, principal=BOB)("u7:one") is True
        governance.revoke_grant(db.conn, gid)
        with pytest.raises(VerbatimError):
            _mk(db, principal=BOB)

    def test_epoch_bump_does_not_break_unpinned_caller(self, db):
        """An unpinned caller binds current epoch — authority granted
        before a bump still authorizes (check_epoch honored)."""
        _seeded(db)
        governance.bump_epoch(db.conn, SC)
        el = _mk(db)
        assert el("u7:one") is True

    def test_authorization_resolved_once(self, db):
        """The predicate never re-authorizes: revoking the grant AFTER
        construction cannot widen, and the predicate keeps answering from
        its immutable snapshot data (it issues no SQL at all)."""
        _seeded(db)
        el = _mk(db)
        assert el("u7:one") is True
        # Snapshot-pinned verdicts stay consistent even if a later write
        # changes authority — the predicate does not consult the store.
        for row in db.conn.execute(
            "SELECT grant_id FROM grants_v3 WHERE principal_id = ?", (ALICE,)
        ):
            governance.revoke_grant(db.conn, row[0])
        assert el("u7:one") is True


# ---------------------------------------------------------------------------
# robustness: bad inputs, predicate faults, determinism
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_unrecognized_inputs_deny(self, db):
        _seeded(db)
        el = _mk(db)
        for bad in (None, 42, 4.5, [], {}, object(), b"bytes"):
            assert el(bad) is False
        assert el.stats["unrecognized"] >= 5

    def test_empty_mapping_denies(self, db):
        _seeded(db)
        el = _mk(db)
        assert el({"foo": "bar"}) is False

    def test_exception_inside_predicate_returns_false(self, db):
        """Any internal fault is a deny — the class below raises inside
        attribute access, which the predicate converts to ``False``."""

        class Evil:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        _seeded(db)
        el = _mk(db)
        assert el(Evil()) is False
        assert el.stats["predicate_errors"] == 1

    def test_deterministic_verdicts(self, db):
        """Identical inputs under the same pinned snapshot give identical
        verdicts — including after failures."""
        _seeded(db)
        el = _mk(db)
        row = {
            "unit_id": "u7:one", "source_id": "src-1", "revision": 1,
            "scope_id": SC, "generation": 1,
        }
        verdicts = [el(row) for _ in range(5)]
        assert verdicts == [True] * 5
        misses = [el("u7:ghost") for _ in range(3)]
        assert misses == [False] * 3
        assert el.stats["calls"] == 8
        assert el.stats["missing_unit_rows"] == 3

    def test_predicate_issues_no_sql(self, db):
        """Construction reads the snapshot once; the predicate must never
        touch the connection (V7-05.06).  A conn wrapper that starts
        raising on ``execute`` post-construction proves it."""

        class Guard:
            def __init__(self, inner):
                self._inner = inner
                self.broken = False

            def execute(self, *a, **k):
                if self.broken:
                    raise sqlite3.OperationalError("no SQL allowed")
                return self._inner.execute(*a, **k)

        _seeded(db)
        guard = Guard(db.conn)
        el = make_eligible(
            guard, db, scope_id=SC, generation=1, principal_id=ALICE,
        )
        assert el("u7:one") is True
        guard.broken = True
        assert el("u7:one") is True
        assert el("u7:ghost") is False
        assert el.stats["predicate_errors"] == 0


# ---------------------------------------------------------------------------
# honest unavailable modes
# ---------------------------------------------------------------------------


class TestUnavailable:
    def test_missing_units_table_unavailable(self, db):
        """No unit plane → honest all-false predicate with a reason, not a
        silent pass-through."""
        _grant(db.conn)
        _source(db.conn, "src-1", SC)
        db.conn.execute("DROP TABLE units")
        el = _mk(db)
        assert el.stats["mode"] == "unavailable"
        assert "units" in str(el.stats["reason"])
        assert el({"unit_id": "u7:one"}) is False
        assert el("u7:one") is False

    def test_missing_sources_table_unavailable(self, db):
        _grant(db.conn)
        db.conn.execute("PRAGMA foreign_keys = OFF")
        db.conn.execute("DROP TABLE sources")
        el = _mk(db)
        assert el.stats["mode"] == "unavailable"
        assert el({"source_id": "s", "revision": 1}) is False

    def test_missing_governance_unavailable(self, db):
        """A store with no grants table cannot authorize — all-false
        unavailable rather than a crash or a pass."""
        bare = _Db(schema=False)
        try:
            bare.conn.executescript(DDL_V1)
            for s in _split_alters(DDL_V2_ALTER):
                bare.conn.execute(s)
            _scope(bare.conn)
            _source(bare.conn, "src-1", SC)
            ensure_v7_additive(bare.conn)
            _unit(bare.conn, "u7:one", "src-1", 1, SC, 1)
            el = make_eligible(
                bare.conn, bare, scope_id=SC, generation=1,
                principal_id=ALICE,
            )
            assert el.stats["mode"] == "unavailable"
            assert el("u7:one") is False
        finally:
            bare.close()

    def test_unpinned_generation_unavailable(self, db):
        _seeded(db)
        el = make_eligible(
            db.conn, db, scope_id=SC, generation=None,
            principal_id=ALICE,
        )
        assert el.stats["mode"] == "unavailable"
        assert el.stats["reason"] == "generation_unpinned"
        assert el("u7:one") is False

    def test_suppressing_purge_without_targets_unavailable(self, db):
        """A suppressing purge row with no readable target table is an
        unresolved erasure — fail closed to all-false, never widen."""
        _seeded(db)
        db.conn.execute("PRAGMA foreign_keys = OFF")
        db.conn.execute(
            "INSERT INTO purges(purge_id, selection_digest, scope_id,"
            " state, requested_us) VALUES ('p1',x'00',?,'suppressed',1)",
            (SC,),
        )
        db.conn.execute("DROP TABLE purge_targets")
        el = _mk(db)
        assert el.stats["mode"] == "unavailable"
        assert el("u7:one") is False

    def test_no_snapshot_unavailable(self, db):
        _grant(db.conn)
        el = make_eligible(
            None, object(), scope_id=SC, generation=1, principal_id=ALICE,
        )
        assert el.stats["mode"] == "unavailable"
        assert el.stats["reason"] == "no_read_snapshot"
        assert el("u7:one") is False


# ---------------------------------------------------------------------------
# real store proof — Store + Memory + real security calls
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem(tmp_path):
    from verbatim import Memory

    m = Memory(path=str(tmp_path / "m.db"), worker="external")
    yield m
    try:
        m.close()
    except Exception:
        pass


def _project(mem, source_id: str, revision: int = 1, generation: int = 1):
    """Run the REAL units projection job over persisted rows — the same
    ``project_units_v7`` the drain worker calls, inside a write tx."""
    from verbatim.jobs.units_jobs import project_units_v7

    with mem._store.tx() as conn:
        srow = conn.execute(
            "SELECT source_id, origin, external_id, source_kind, scope_id,"
            " speaker_id, created_us FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        assert srow is not None
        rrow = conn.execute(
            "SELECT source_id, revision, payload, event_us, captured_us,"
            " timezone, provenance, metadata_json FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchone()
        assert rrow is not None
        source_row = dict(zip(
            ("source_id", "origin", "external_id", "source_kind",
             "scope_id", "speaker_id", "created_us"), srow,
        ))
        revision_row = dict(zip(
            ("source_id", "revision", "payload", "event_us", "captured_us",
             "timezone", "provenance", "metadata_json"), rrow,
        ))
        stats = project_units_v7(
            conn,
            source_row=source_row,
            revision_row=revision_row,
            add_args={},
            generation=generation,
            scope_id=mem._namespace,
        )
        assert stats["units"] >= 1, stats
        return stats


def _units(mem, source_id: str) -> list:
    with mem._store.read() as conn:
        return conn.execute(
            "SELECT unit_id, source_id, revision, scope_id, generation"
            " FROM units WHERE source_id = ? ORDER BY generation, unit_id",
            (source_id,),
        ).fetchall()


class TestRealStore:
    def test_memory_add_project_eligible(self, mem):
        """End-to-end: real Memory.add ingest → real project_units_v7 →
        adapter verdicts on the pinned snapshot."""
        r = mem.add("hello world. this is a real store test.")
        sid = r.memory_id
        _project(mem, sid)
        rows = _units(mem, sid)
        assert rows
        uid = rows[0][0]
        with mem._store.read() as conn:
            el = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el.stats["mode"] == "ok"
            assert el.stats["checked_sources"] == 1
            row = dict(zip(
                ("unit_id", "source_id", "revision", "scope_id",
                 "generation"), rows[0],
            ))
            assert el(row) is True
            assert el(uid) is True
            assert el("u7:ghost") is False

    def test_real_quarantine_hold_and_release(self, mem):
        """Real ``open_quarantine`` on the source revision hides the unit;
        real ``release`` restores it."""
        r = mem.add("quarantine me. real hold path.")
        sid = r.memory_id
        _project(mem, sid)
        uid = _units(mem, sid)[0][0]
        with mem._store.tx() as conn:
            assert quar.open_quarantine(
                conn, ("source", sid, 1), ["attack_risk:blocked"], [],
                scope_id=mem._namespace,
            ) is True
        with mem._store.read() as conn:
            el = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el(uid) is False
            assert el.stats["held_sources"] == 1
        with mem._store.tx() as conn:
            quar.release(
                conn, ("source", sid, 1), decided_by=mem._owner,
                scope_id=mem._namespace,
            )
        with mem._store.read() as conn:
            el = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el(uid) is True

    def test_real_purge_suppress_invisible(self, mem):
        """Real ``purge.suppress`` on the source tombstones every unit."""
        from verbatim import purge as purge_mod

        r = mem.add("erase me. real purge path.")
        sid = r.memory_id
        _project(mem, sid)
        uid = _units(mem, sid)[0][0]
        purge_mod.suppress(
            mem._store, mem._namespace, [("source", sid)],
            actor="operator",
        )
        with mem._store.read() as conn:
            el = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el(uid) is False
            assert el.stats["held_sources"] == 1

    def test_real_generation_fence(self, mem):
        """Projection stamped at gen 2 is invisible to a gen-1 pin and
        visible at gen 2 — the ``<=`` snapshot fence on real rows."""
        r = mem.add("fence me twice.")
        sid = r.memory_id
        _project(mem, sid, generation=1)
        _project(mem, sid, generation=2)
        rows = _units(mem, sid)
        g1 = [u for u in rows if u[4] == 1]
        g2 = [u for u in rows if u[4] == 2]
        assert g1 and g2
        with mem._store.read() as conn:
            el1 = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el1(g1[0][0]) is True
            assert el1(g2[0][0]) is False  # gen-2 row above the pin
            el2 = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=2, principal_id=mem._owner,
            )
            assert el2(g2[0][0]) is True
            assert el2(g1[0][0]) is True  # older generation still in fence

    def test_real_denied_principal_raises(self, mem):
        """A principal with no grant gets the typed denial, not a
        predicate."""
        r = mem.add("governed bytes.")
        _project(mem, r.memory_id)
        with mem._store.read() as conn:
            with pytest.raises(VerbatimError) as ei:
                make_eligible(
                    conn, mem._store, scope_id=mem._namespace,
                    generation=1, principal_id="mallory",
                )
            assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_real_snapshot_isolation(self, mem):
        """The predicate closes over the CALLER's pinned snapshot — a hold
        committed after the read snapshot opened cannot retroactively
        change its verdicts (deterministic within one snapshot)."""
        r = mem.add("snapshot isolation test.")
        sid = r.memory_id
        _project(mem, sid)
        uid = _units(mem, sid)[0][0]
        with mem._store.read() as conn:
            el = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el(uid) is True
            with mem._store.tx() as wconn:
                quar.open_quarantine(
                    wconn, ("source", sid, 1), ["rule:test"], [],
                    scope_id=mem._namespace,
                )
            # Same pinned snapshot — the new hold is invisible here by
            # construction; the committed-hold path is proven by the
            # hold/release test above.
            assert el(uid) is True
        with mem._store.read() as conn:
            el2 = make_eligible(
                conn, mem._store, scope_id=mem._namespace,
                generation=1, principal_id=mem._owner,
            )
            assert el2(uid) is False
