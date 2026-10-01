"""Regression tests for the beat-it fix wave.

Covers the just-landed fixes end to end, each against the real
implementation (no mocks of the SQL layer):

1. ``_Eligible.unit_ids`` — the materialized eligible-set view must
   reproduce the callable predicate's verdicts exactly across
   generations, unit holds, suppressing purges, and non-OK source
   pairs, and must materialize once (``is``-identical across calls).
2. Dense ``_KeyEligibility`` — when ``eligible.unit_ids`` exists the
   adapter picks ``mode == "set"`` and performs zero ``units`` SELECTs;
   the callable fallback still does the row lookups.
3. ``derive_units`` — a single-turn session mints no byte-identical
   container unit; the turn keeps ``session_id`` with
   ``parent_unit_id`` None, while multi-turn sessions still emit the
   session row.
4. ``_norm_occurred``/``derive_units`` — human ``when``/``date``
   strings ("10:04 am on 19 December, 2023", "8 May, 2023",
   "May 8, 2023") parse deterministically; unparseable stays unknown.
5. ``_cap_spans`` — ":" is a sentence boundary, so "Jon: Gina …"
   yields separate ``jon``/``gina`` mentions, never a fused run.
6. Premise mismatch — a query that names exactly one speaker whose
   canons resolve via the ``units`` table flags groups authored only by
   somebody else (``premise_mismatch == "speaker"``, trigger "d",
   ``INSUFFICIENT``), without deleting deliverable items; an
   Alice-authored supported group never trips it.
7. Lexical df pre-gate — a maintained ``lex_df`` above the fetch cut
   makes ``posts[term] == set()``, is reported in
   ``stats["df_prefetch_gated"]`` and ``stats["nomination_dropped"]``
   (reason ``df_threshold``), and ``stats["df"]`` reports the
   maintained df, not 0.
8. ``project_units_v7`` — writes ``graph_edges`` rows and reports
   ``observations_v7`` stats, following ``tests/jobs/test_units_jobs.py``
   fixture conventions.

Fixture conventions mirror ``tests/retrieval/v7/test_eligibility.py``
(real DDL chain + real governance/quarantine/purge rows),
``tests/retrieval/v7/test_dense.py`` (``_KeyEligibility`` +
``matrix.scan``), ``tests/retrieval/v7/test_lexical.py`` (mirror FTS
DDL + maintained ``lex_stats``/``lex_df``),
``tests/projections/test_units_v7.py`` (``src``/``rev`` row factories),
and ``tests/querying/test_verdict_v2.py`` (``QueryViewV7`` /
``ScoredCandidate`` literals).
"""

from __future__ import annotations

import math
import random
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim import governance  # noqa: E402
from verbatim.core.types_v3 import Verb  # noqa: E402
from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneSlice,
    LaneStatus,
    LaneName,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    ResultStatus,
    RetrievalPolicyV7,
    ScoredCandidate,
    SupportLabel,
)
from verbatim.embeddings import matrix as mx  # noqa: E402
from verbatim.enrichment import entities_v2 as ev2  # noqa: E402
from verbatim.jobs import units_jobs as uj  # noqa: E402
from verbatim.projections.units_v7 import (  # noqa: E402
    UNITS_DERIVER_VERSION,
    _norm_occurred,
    derive_units,
)
from verbatim.querying import verdict_v2 as vv  # noqa: E402
from verbatim.retrieval.v7 import dense as dense_mod  # noqa: E402
from verbatim.retrieval.v7 import lexical as lex  # noqa: E402
from verbatim.retrieval.v7.eligibility import make_eligible  # noqa: E402
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
from verbatim.storage.store import Store, _split_alters  # noqa: E402

SC = "ns-alpha"
SC_B = "ns-beta"
ALICE = "alice"
BOB = "bob"


# ---------------------------------------------------------------------------
# mirror store — the real schema chain, in memory (test_eligibility.py
# convention)
# ---------------------------------------------------------------------------


def _apply_real_schema(conn: sqlite3.Connection) -> None:
    """The ``Store.create`` DDL chain plus the lazily-applied V7 additive
    plane — the real tables the adapter reads."""
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
    conn, source_id: str = "src-1", scope_id: str = SC,
    *, payload: bytes = b"hello world", revision: int = 1,
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


def _unit(
    conn,
    unit_id: str,
    source_id: str = "src-1",
    revision: int = 1,
    scope_id: str = SC,
    generation: int = 1,
    kind: str = "turn",
    speaker: str = "alice",
) -> None:
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
        (unit_id, source_id, revision, scope_id, kind, speaker, generation),
    )


def _purge(conn, purge_id: str, state: str, targets: list,
           scope_id: str = SC) -> None:
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


def _mk(db: _Db, scope_id: str = SC, generation: int = 1,
        principal: str = ALICE, purpose: str = "recall",
        cache_size=None):
    return make_eligible(
        db.conn,
        db,
        scope_id=scope_id,
        generation=generation,
        principal_id=principal,
        purpose=purpose,
        cache_size=cache_size,
    )


class _Spy:
    """``conn.execute`` wrapper that records every SQL statement — used to
    prove the set-mode eligibility path issues zero ``units`` lookups."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.statements: list[str] = []

    def execute(self, sql, params=()):
        self.statements.append(str(sql))
        return self._conn.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def units_selects(self) -> list:
        return [
            s for s in self.statements
            if re.search(r"(?i)\bfrom\s+units\b", s)
        ]


# ===========================================================================
# 1 — _Eligible.unit_ids: set view == callable verdicts, materialized once
# ===========================================================================


def _seed_unit_ids_fixture(db: _Db) -> None:
    """Multiple generations, a unit hold, a suppressing purge, and two
    non-OK source pairs — the equivalence universe the task pins."""
    _grant(db.conn)
    _scope(db.conn, SC_B)
    _source(db.conn, "src-ok", SC)
    _source(db.conn, "src-held", SC)
    _source(db.conn, "src-empty", SC, payload=b"")
    _source(db.conn, "src-foreign", SC_B)
    # eligible units at two generations
    _unit(db.conn, "u7:ok", "src-ok", 1, SC, 1)
    _unit(db.conn, "u7:gen2", "src-ok", 1, SC, 2)
    # one unit_id stamped at both generations — the newest row's source
    # is held, so pin=1 sees it but pin=2 does not
    _unit(db.conn, "u7:dup", "src-ok", 1, SC, 1)
    _unit(db.conn, "u7:dup", "src-held", 1, SC, 2)
    # unit-level hold
    _unit(db.conn, "u7:held", "src-ok", 1, SC, 1)
    quar.open_quarantine(
        db.conn, ("unit", "u7:held", 1), ["rule:test"], [], scope_id=SC,
    )
    # suppressing purge target
    _unit(db.conn, "u7:purged", "src-ok", 1, SC, 1)
    _purge(db.conn, "pg-1", "suppressed", [("unit", "u7:purged")])
    # source-level hold cascades to its units
    quar.open_quarantine(
        db.conn, ("source", "src-held", 1), ["rule:test"], [], scope_id=SC,
    )
    # non-OK source pairs: foreign-partition source + emptied payload
    _unit(db.conn, "u7:badsrc", "src-foreign", 1, SC, 1)
    _unit(db.conn, "u7:emptysrc", "src-empty", 1, SC, 1)


def _universe_ids(conn, scope_id: str, generation: int) -> set:
    return {
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT unit_id FROM units"
            " WHERE scope_id = ? AND generation <= ?",
            (scope_id, generation),
        )
    }


class TestEligibleUnitIds:
    def test_set_view_equivalent_to_callable_universe(self, db):
        _seed_unit_ids_fixture(db)
        el = _mk(db, generation=2)
        universe = _universe_ids(db.conn, SC, 2) | {"u7:ghost", "u7:never"}
        assert el.unit_ids == {u for u in universe if el(u)}
        # spot-check the verdicts the property must reproduce
        assert "u7:ok" in el.unit_ids
        assert "u7:gen2" in el.unit_ids
        assert "u7:dup" not in el.unit_ids      # latest gen → held source
        assert "u7:held" not in el.unit_ids     # unit hold
        assert "u7:purged" not in el.unit_ids   # suppressing purge
        assert "u7:badsrc" not in el.unit_ids   # foreign source pair
        assert "u7:emptysrc" not in el.unit_ids  # emptied payload pair
        assert "u7:ghost" not in el.unit_ids

    def test_generation_pin_changes_the_set(self, db):
        _seed_unit_ids_fixture(db)
        el1 = _mk(db, generation=1)
        # at pin=1 only the gen-1 rows exist; u7:dup's latest in-fence
        # row derives from src-ok and is eligible
        assert "u7:dup" in el1.unit_ids
        assert "u7:gen2" not in el1.unit_ids    # above the pin
        el2 = _mk(db, generation=2)
        assert "u7:dup" not in el2.unit_ids
        assert "u7:gen2" in el2.unit_ids

    def test_materializes_once_and_reports_stats(self, db):
        _seed_unit_ids_fixture(db)
        # V8-14.01: the snapshot cache pre-seeds ``_unit_ids`` — the
        # laziness contract survives only on the non-cached path.
        el = _mk(db, generation=2, cache_size=0)
        assert el._unit_ids is None  # lazy until a consumer asks
        first = el.unit_ids
        assert el.stats["unit_ids_materialized"] == len(first)
        second = el.unit_ids
        assert first is second
        assert isinstance(first, frozenset)


# ===========================================================================
# 2 — dense _KeyEligibility: unit_ids -> set mode, zero units SELECTs
# ===========================================================================


def _rand_vec(rng: random.Random, dims: int = 32) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class TestDenseKeyEligibility:
    def test_unit_ids_property_selects_set_mode_zero_sql(self, db):
        """A callable predicate that also exposes ``unit_ids`` resolves
        keys in memory — no ``units`` SELECT per block."""
        _seed_unit_ids_fixture(db)
        el = _mk(db, generation=2)  # callable AND has .unit_ids
        spy = _Spy(db.conn)
        ke = dense_mod._KeyEligibility(spy, el, SC, 2)
        assert ke._mode == "set"
        got = ke.filter_keys(
            ["u7:ok", "u7:gen2", "u7:held", "u7:purged", "u7:dup",
             "u7:ghost"]
        )
        assert got == set(el.unit_ids) & {
            "u7:ok", "u7:gen2", "u7:held", "u7:purged", "u7:dup",
            "u7:ghost",
        }
        assert got == {"u7:ok", "u7:gen2"}
        assert ke.evaluated == 0
        assert spy.units_selects() == []
        assert spy.statements == []  # no SQL at all on this path

    def test_bare_set_and_attr_stub_choose_set_mode(self, db):
        spy = _Spy(db.conn)
        for elig in (
            frozenset({"u-a", "u-b"}),
            types.SimpleNamespace(unit_ids=frozenset({"u-a", "u-b"})),
            types.SimpleNamespace(unit_ids={"u-a": True, "u-b": True}),
        ):
            ke = dense_mod._KeyEligibility(spy, elig, SC, 1)
            assert ke._mode == "set"
            assert ke.filter_keys(["u-a", "u-c"]) == {"u-a"}
            assert ke.evaluated == 0
        assert spy.units_selects() == []

    def test_callable_still_uses_units_lookup(self, db):
        """The pre-fix path stays intact for callables without
        ``unit_ids``: verdicts come from real ``units`` row fetches."""
        _seed_unit_ids_fixture(db)
        spy = _Spy(db.conn)
        ke = dense_mod._KeyEligibility(
            spy, lambda row: row["unit_id"] == "u7:ok", SC, 2
        )
        assert ke._mode == "callable"
        got = ke.filter_keys(["u7:ok", "u7:held", "u7:ghost"])
        assert got == {"u7:ok"}
        assert ke.evaluated == 3
        assert spy.units_selects() != []
        # second pass over the same keys is memoized — no new SELECTs
        n = len(spy.units_selects())
        assert ke.filter_keys(["u7:ok", "u7:held"]) == {"u7:ok"}
        assert len(spy.units_selects()) == n

    def test_scan_with_key_eligibility_performs_no_units_sql(self, db):
        """End to end: ``matrix.scan`` consumes ``filter_keys`` — the
        whole block scan resolves eligibility with zero ``units`` reads."""
        _seed_unit_ids_fixture(db)
        el = _mk(db, generation=2)
        rng = random.Random(7)
        vecs = {
            "u7:ok": _rand_vec(rng),
            "u7:gen2": _rand_vec(rng),
            "u7:held": _rand_vec(rng),
            "u7:purged": _rand_vec(rng),
        }
        keys = sorted(vecs)
        mx.write_block(
            db.conn, "enc:test:v1", SC, 2, 0,
            [(k, vecs[k]) for k in keys],
        )
        spy = _Spy(db.conn)
        ke = dense_mod._KeyEligibility(spy, el, SC, 2)
        res = mx.scan(
            spy, "enc:test:v1", SC, 2, _rand_vec(random.Random(9)),
            eligible_keys=ke, k=10, use_numpy=False,
        )
        assert res.examined == 4
        assert res.eligible == 2
        assert {k for k, _ in res} <= set(el.unit_ids)
        assert "u7:held" not in {k for k, _ in res}
        assert "u7:purged" not in {k for k, _ in res}
        assert ke.evaluated == 0
        assert spy.units_selects() == []


# ===========================================================================
# 3 — derive_units: single-turn sessions mint no container unit
# ===========================================================================


T0 = 1_700_000_000_000_000  # arbitrary epoch µs


def _src(source_id="src1", kind="user_message", speaker=None, scope="sc"):
    return {
        "source_id": source_id,
        "origin": "test",
        "external_id": None,
        "source_kind": kind,
        "scope_id": scope,
        "speaker_id": speaker,
        "created_us": T0,
    }


def _rev(payload: bytes, source_id="src1", revision=1, event_us=T0,
         meta=None):
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


def _transcript(messages) -> bytes:
    lines = []
    for m in messages:
        spk = m.get("speaker") or m.get("role") or "anon"
        lines.append(f"{spk}: {m['content']}")
    return "\n".join(lines).encode("utf-8")


def _kinds(units) -> list:
    return [u["kind"] for u in units]


class TestSingleTurnSessionSuppression:
    def test_explicit_session_id_single_turn_no_container(self):
        payload = b"I love pizza."
        u = derive_units(_src(), _rev(payload), {"session_id": "s-9"})
        assert _kinds(u) == ["turn"]
        (turn,) = u
        assert turn["session_id"] == "s-9"
        assert turn["parent_unit_id"] is None
        # the turn still pins the whole payload byte range
        assert (turn["byte_start"], turn["byte_end"]) == (0, len(payload))

    def test_single_message_session_suppressed(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "met for coffee"},
        ]
        u = derive_units(
            _src(), _rev(_transcript(msgs)),
            {"messages": msgs, "session_id": "s-one"},
        )
        assert _kinds(u) == ["turn"]
        (turn,) = u
        assert turn["session_id"] == "s-one"
        assert turn["parent_unit_id"] is None

    def test_multi_turn_session_still_emits_container(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "hi gina"},
            {"role": "assistant", "speaker": "gina", "content": "hey alice"},
        ]
        u = derive_units(
            _src(), _rev(_transcript(msgs)),
            {"messages": msgs, "session_id": "s-x"},
        )
        assert _kinds(u) == ["session", "turn", "turn"]
        sess, t0, t1 = u
        assert sess["session_id"] == "s-x"
        assert t0["session_id"] == "s-x" == t1["session_id"]
        assert t0["parent_unit_id"] == sess["unit_id"]
        assert t1["parent_unit_id"] == sess["unit_id"]
        # the container covers its members' byte range exactly
        assert (sess["byte_start"], sess["byte_end"]) == (
            t0["byte_start"], t1["byte_end"])

    def test_multi_session_suppresses_only_the_singleton(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "one",
             "session_id": "s-a"},
            {"role": "user", "speaker": "alice", "content": "two",
             "session_id": "s-a"},
            {"role": "user", "speaker": "alice", "content": "three",
             "session_id": "s-b"},
        ]
        u = derive_units(
            _src(), _rev(_transcript(msgs)), {"messages": msgs},
        )
        sess = [r for r in u if r["kind"] == "session"]
        turns = [r for r in u if r["kind"] == "turn"]
        assert {s["session_id"] for s in sess} == {"s-a"}
        assert [t["session_id"] for t in turns] == ["s-a", "s-a", "s-b"]
        # the suppressed session's turn keeps its grouping id and has
        # no parent container
        assert turns[2]["session_id"] == "s-b"
        assert turns[2]["parent_unit_id"] is None
        assert turns[0]["parent_unit_id"] == sess[0]["unit_id"]

    def test_deriver_version_declares_the_rule(self):
        # the suppression rule is versioned — a silent unit-shape change
        # would break replay determinism (V7-30.01)
        assert UNITS_DERIVER_VERSION == "units_v7/v1.1"


# ===========================================================================
# 4 — _norm_occurred / derive_units: human `when`/`date` strings parse
# ===========================================================================


INSTANT_US = 1_702_980_240_000_000  # 2023-12-19T10:04:00Z
DAY_US = 1_683_504_000_000_000      # 2023-05-08T00:00:00Z
DAY_LEN_US = 86_400_000_000


class TestOccurredParsing:
    def test_when_instant_direct(self):
        assert _norm_occurred(
            None, {"when": "10:04 am on 19 December, 2023"}
        ) == (INSTANT_US, INSTANT_US, "instant", "explicit")

    def test_when_day_dmy_direct(self):
        assert _norm_occurred(None, {"when": "8 May, 2023"}) == (
            DAY_US, DAY_US + DAY_LEN_US, "day", "explicit")

    def test_when_day_mdy_direct(self):
        assert _norm_occurred(None, {"when": "May 8, 2023"}) == (
            DAY_US, DAY_US + DAY_LEN_US, "day", "explicit")

    def test_date_key_also_parses(self):
        assert _norm_occurred(None, {"date": "8 May, 2023"}) == (
            DAY_US, DAY_US + DAY_LEN_US, "day", "explicit")

    def test_bare_string_occurred_parses(self):
        """``occurred`` handed a bare human string takes the same
        grammar — not just the ``when``/``date`` channel."""
        assert _norm_occurred("8 May, 2023", None) == (
            DAY_US, DAY_US + DAY_LEN_US, "day", "explicit")
        assert _norm_occurred(
            "10:04 am on 19 December, 2023", None
        ) == (INSTANT_US, INSTANT_US, "instant", "explicit")

    def test_garbage_when_stays_unknown(self):
        assert _norm_occurred(
            None, {"when": "sometime last spring"}
        ) == (None, None, "unknown", "unknown")
        assert _norm_occurred("sometime last spring", None) == (
            None, None, "unknown", "unknown")

    def test_rfc3339_still_wins_as_instant(self):
        assert _norm_occurred(None, {"when": "2023-12-19T10:04:00Z"}) == (
            INSTANT_US, INSTANT_US, "instant", "explicit")

    def test_derive_units_metadata_when_lands_on_turn(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "met for coffee"},
        ]
        payload = _transcript(msgs)
        u = derive_units(
            _src(), _rev(payload),
            {"messages": msgs, "when": "10:04 am on 19 December, 2023"},
        )
        turn = [r for r in u if r["kind"] == "turn"][0]
        assert turn["occurred_start_us"] == INSTANT_US
        assert turn["occurred_end_us"] == INSTANT_US
        assert turn["occurred_precision"] == "instant"
        assert turn["occurred_source"] == "explicit"

    def test_derive_units_metadata_json_when_lands_on_turn(self):
        """``source_revisions.metadata_json`` merges under add_args —
        a persisted ``when`` reaches the same channel."""
        msgs = [
            {"role": "user", "speaker": "alice", "content": "met for coffee"},
        ]
        payload = _transcript(msgs)
        u = derive_units(
            _src(),
            _rev(payload, meta='{"when": "8 May, 2023"}'),
            {"messages": msgs},
        )
        turn = [r for r in u if r["kind"] == "turn"][0]
        assert turn["occurred_start_us"] == DAY_US
        assert turn["occurred_end_us"] == DAY_US + DAY_LEN_US
        assert turn["occurred_precision"] == "day"
        assert turn["occurred_source"] == "explicit"

    def test_derive_units_per_message_when(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "met for coffee",
             "when": "May 8, 2023"},
        ]
        u = derive_units(
            _src(), _rev(_transcript(msgs)), {"messages": msgs},
        )
        turn = [r for r in u if r["kind"] == "turn"][0]
        assert turn["occurred_start_us"] == DAY_US
        assert turn["occurred_end_us"] == DAY_US + DAY_LEN_US
        assert turn["occurred_precision"] == "day"

    def test_derive_units_unparseable_when_is_honest_null(self):
        msgs = [
            {"role": "user", "speaker": "alice", "content": "met for coffee"},
        ]
        u = derive_units(
            _src(), _rev(_transcript(msgs)),
            {"messages": msgs, "when": "sometime last spring"},
        )
        turn = [r for r in u if r["kind"] == "turn"][0]
        assert turn["occurred_start_us"] is None
        assert turn["occurred_end_us"] is None
        assert turn["occurred_precision"] == "unknown"
        assert turn["occurred_source"] == "unknown"


# ===========================================================================
# 5 — _cap_spans: ":" is a sentence boundary (chat speaker prefix)
# ===========================================================================


def _norm_text(text: str) -> NormAnalysis:
    """Stand-in for ``norm/v2`` analysis carrying the raw source text —
    the shape ``extract_mentions`` consumes (``norm.text``)."""
    terms = []
    for m in ev2._WORD_RE.finditer(text):
        tok = m.group(0)
        start = len(text[: m.start()].encode("utf-8"))
        terms.append(
            NormTerm(tok.casefold(), "text", start,
                     start + len(tok.encode("utf-8")))
        )
    return NormAnalysis(
        analyzer_id="norm/v2", terms=tuple(terms), identifiers=(),
        text=text,
    )


class TestCapSpansColonBoundary:
    def test_speaker_prefix_not_fused(self):
        text = "Jon: Gina went to the store"
        surfaces = [text[s:e] for s, e in ev2._cap_spans(text)]
        assert "Jon" in surfaces
        assert "Gina" in surfaces
        assert "Jon: Gina" not in surfaces
        assert "Jon Gina" not in surfaces

    def test_mentions_canon_split(self):
        text = "Jon: Gina went to the store"
        ms = ev2.extract_mentions(_norm_text(text), "u1")
        canons = {m.canon for m in ms}
        assert {"jon", "gina"} <= canons
        assert "jon gina" not in canons
        assert "jon: gina" not in canons
        # surfaces are byte-exact slices of the source
        by_canon = {m.canon: m for m in ms}
        assert text.encode()[by_canon["jon"].byte_start:
                             by_canon["jon"].byte_end] == b"Jon"
        assert text.encode()[by_canon["gina"].byte_start:
                             by_canon["gina"].byte_end] == b"Gina"

    def test_colon_mid_run_splits(self):
        text = "Alice told Jon: Gina left early"
        ms = ev2.extract_mentions(_norm_text(text), "u1")
        canons = {m.canon for m in ms}
        assert {"alice", "jon", "gina"} <= canons
        assert not any("jon" in c and "gina" in c for c in canons)

    def test_non_boundary_run_still_fuses(self):
        """Regression guard: the colon fix must not break the ordinary
        multi-word capitalized run."""
        text = "New York City is big"
        surfaces = [text[s:e] for s, e in ev2._cap_spans(text)]
        assert "New York City" in surfaces


# ===========================================================================
# 6 — premise mismatch: one resolved speaker, foreign-authored group
# ===========================================================================


_POS = [0]


def _term(text: str, channel: str = "text") -> NormTerm:
    _POS[0] += len(text) + 1
    return NormTerm(text, channel, _POS[0] - len(text), _POS[0])


def _qv(
    text: str = "",
    *,
    terms=(),
    ids=(),
    ents=(),
    intent: IntentClass = IntentClass.LOOKUP,
    speaker=None,
) -> QueryViewV7:
    _POS[0] = 0
    t_terms = tuple(_term(t, "text") for t in terms)
    t_ids = tuple(_term(i, "identifier") for i in ids)
    norm = NormAnalysis("norm/v2", t_terms + t_ids, t_ids, text)
    ir = IntentResult(intent, (intent,), None)
    return QueryViewV7(
        query=text, norm=norm, intent=ir,
        entity_canons=tuple(ents), speaker_canon=speaker,
    )


def _item(unit: str, text=None, *, score: float = 0.5, source=None,
          speaker=None, detail=None, signals=None, group=None, lane=None):
    """A plain-mapping candidate — verdict_v2 duck-types mappings and
    objects identically."""
    d = dict(detail or {})
    if signals:
        d.setdefault("signals", dict(signals))
    if speaker is not None:
        d.setdefault("speaker_canon", speaker)
    if text is not None:
        d.setdefault("text", text)
    if group is not None:
        d.setdefault("group", group)
    it = {
        "unit_id": unit,
        "source_id": source or f"src-{unit}",
        "revision": 1,
        "score": score,
        "detail": d,
    }
    if lane is not None:
        it["lane"] = lane
    return it


def _verdict_ctx(conn, scope_id: str = "scope-v", generation: int = 1,
                 policy=None):
    """Duck-typed ctx — verdict_v2 reads ``conn``/``scope_id``/
    ``generation`` attributes plus ``policy`` for arm flags
    (``verdict.premise_speaker``, V8-12.01 — absent → default off)."""
    return types.SimpleNamespace(
        conn=conn, scope_id=scope_id, generation=generation,
        policy=policy,
    )


def _speaker_units(conn, rows, scope_id: str = "scope-v"):
    """Real ``units`` rows for the speaker-resolution probes."""
    for unit_id, speaker in rows:
        conn.execute(
            "INSERT INTO units(unit_id, source_id, revision, scope_id,"
            " kind, speaker_canon, generation)"
            " VALUES (?,?,?,?,?,?,1)",
            (unit_id, f"src-{unit_id}", 1, scope_id, "turn", speaker),
        )


class TestPremiseMismatch:
    """A query that presupposes exactly one speaker — via the
    ``speaker_canon`` caller hint or via entity canons that resolve to a
    single speaker through the ``units`` table — must flag groups whose
    members were all authored by somebody else."""

    def test_hint_speaker_foreign_authored_group_mismatches(self):
        q = _qv(
            "budget forecast review deck",
            terms=("budget", "forecast", "review", "deck"),
            speaker="alice",
        )
        items = [
            _item("u-b1", "the budget slides are here",
                  speaker="bob", signals={"lexical": 1.0}, lane="lex"),
        ]
        # V8-12.01: the premise measurement is always recorded, but the
        # trigger fires only under ``verdict.premise_speaker=on``.
        ctx = types.SimpleNamespace(policy={"premise_speaker": True})
        groups = vv.classify_groups(items, q, ctx)
        assert len(groups) == 1
        g = groups[0]
        assert g.detail["premise_mismatch"] == "speaker"
        assert g.trigger == "d"
        # the verdict reduces to insufficient — trigger (d) fires even
        # though the intent is a plain lookup, not abstain_likely
        status, missing = vv.result_verdict(groups, q, ctx=ctx)
        assert status == ResultStatus.INSUFFICIENT
        assert "trigger=d" in missing.note
        # nothing is deleted — the group is still deliverable evidence
        delivered = vv.deliverable_groups(groups)
        assert len(delivered) == 1
        assert delivered[0].group_key == g.group_key
        assert g.detail["members"] == ["u-b1"]

        # default-off: same mismatch measured, trigger never fires, and
        # the status is never withheld on premise doubt alone (V8-12.02)
        groups_off = vv.classify_groups(items, q)
        g_off = groups_off[0]
        assert g_off.detail["premise_mismatch"] == "speaker"
        assert g_off.trigger != "d"
        status_off, _ = vv.result_verdict(groups_off, q)
        assert status_off != ResultStatus.INSUFFICIENT

    def test_entity_canons_resolve_speaker_via_units_table(self):
        """``entity_canons=("alice",)`` + a units row naming alice as a
        speaker resolves the premise speaker — no caller hint needed;
        member authorship comes from the units table too."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE units(unit_id TEXT, source_id TEXT,"
            " revision INTEGER, scope_id TEXT, kind TEXT,"
            " speaker_canon TEXT, generation INTEGER)"
        )
        try:
            _speaker_units(conn, [("u-a1", "alice"), ("u-b1", "bob")])
            # V8-12.01: armed flag required for trigger (d) — the
            # measurement itself is unconditional.
            ctx = _verdict_ctx(conn, policy={"premise_speaker": True})
            q = _qv(
                "budget forecast review deck",
                terms=("budget", "forecast", "review", "deck"),
                ents=("alice",),
            )
            items = [
                # no speaker on the item itself — the units-table
                # lookup (``_unit_speakers``) supplies authorship
                _item("u-b1", "the budget slides are here",
                      signals={"lexical": 1.0}, lane="lex"),
            ]
            groups = vv.classify_groups(items, q, ctx)
            (g,) = groups
            assert g.detail["premise_mismatch"] == "speaker"
            assert g.trigger == "d"
            status, missing = vv.result_verdict(groups, q, ctx=ctx)
            assert status == ResultStatus.INSUFFICIENT
            assert "trigger=d" in missing.note
        finally:
            conn.close()

    def test_supported_alice_group_never_triggers_d(self):
        q = _qv(
            "budget deck",
            terms=("budget", "deck"),
            speaker="alice",
        )
        items = [
            _item("u-a1", "the budget deck is ready",
                  speaker="alice", signals={"lexical": 1.0}, lane="lex"),
        ]
        groups = vv.classify_groups(items, q)
        (g,) = groups
        assert g.label == SupportLabel.SUPPORTED
        assert g.detail["premise_mismatch"] is None
        assert g.trigger is None
        status, missing = vv.result_verdict(groups, q)
        assert status == ResultStatus.READY
        assert missing is None

    def test_mixed_authorship_group_rescues_premise(self):
        """A group with both a foreign member and an alice member is not
        a mismatch — the asked speaker did contribute evidence."""
        q = _qv(
            "budget forecast review deck",
            terms=("budget", "forecast", "review", "deck"),
            speaker="alice",
        )
        items = [
            _item("u-b1", "the budget slides are here",
                  speaker="bob", group="g1",
                  signals={"lexical": 1.0}, lane="lex"),
            _item("u-a1", "alice approved the deck",
                  speaker="alice", group="g1",
                  signals={"lexical": 1.0}, lane="lex"),
        ]
        groups = vv.classify_groups(items, q)
        (g,) = groups
        assert g.detail["premise_mismatch"] is None
        assert g.trigger != "d"
        status, _ = vv.result_verdict(groups, q)
        assert status == ResultStatus.READY

    def test_ambiguous_speaker_resolution_stays_off(self):
        """Two resolved speakers → no premise check — a guessed speaker
        is worse than no check (``_premise_speaker`` returns None)."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE units(unit_id TEXT, source_id TEXT,"
            " revision INTEGER, scope_id TEXT, kind TEXT,"
            " speaker_canon TEXT, generation INTEGER)"
        )
        try:
            _speaker_units(conn, [("u-a1", "alice"), ("u-b1", "bob")])
            ctx = _verdict_ctx(conn)
            q = _qv(
                "budget forecast review deck",
                terms=("budget", "forecast", "review", "deck"),
                ents=("alice", "bob"),
            )
            items = [
                _item("u-b1", "the budget slides are here",
                      signals={"lexical": 1.0}, lane="lex"),
            ]
            groups = vv.classify_groups(items, q, ctx)
            (g,) = groups
            assert g.detail["premise_mismatch"] is None
            assert g.trigger != "d"
            status, _ = vv.result_verdict(groups, q)
            assert status == ResultStatus.READY
        finally:
            conn.close()

    def test_unresolved_speaker_stays_off(self):
        """No resolvable subject at all → premise check off — no
        mismatch is reported and no trigger fires."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE units(unit_id TEXT, source_id TEXT,"
            " revision INTEGER, scope_id TEXT, kind TEXT,"
            " speaker_canon TEXT, generation INTEGER)"
        )
        try:
            _speaker_units(conn, [("u-b1", "bob")])  # no alice speaker
            ctx = _verdict_ctx(conn)
            q = _qv(
                "budget forecast review deck",
                terms=("budget", "forecast", "review", "deck"),
            )
            items = [
                _item("u-b1", "the budget slides are here",
                      signals={"lexical": 1.0}, lane="lex"),
            ]
            groups = vv.classify_groups(items, q, ctx)
            (g,) = groups
            assert g.detail["premise_mismatch"] is None
            assert g.trigger != "d"

            # V8 §21.9 row 4: a sole entity canon resolves the asked
            # subject even without a speaker row — the mismatch is then
            # measured and reported, but with ``verdict.premise_speaker``
            # off it stays advisory and can never fire trigger (d).
            q_ents = _qv(
                "budget forecast review deck",
                terms=("budget", "forecast", "review", "deck"),
                ents=("alice",),
            )
            groups = vv.classify_groups(items, q_ents, ctx)
            (g,) = groups
            assert g.detail["premise_mismatch"] == "speaker"
            assert g.trigger != "d"
            status, _ = vv.result_verdict(groups, q_ents, ctx=ctx)
            assert status != ResultStatus.INSUFFICIENT
        finally:
            conn.close()


# ===========================================================================
# 7 — lexical df pre-gate: maintained df above the fetch cut
# ===========================================================================


# §30 mirror DDL — same convention as tests/retrieval/v7/test_lexical.py:
# ``unit_fts.rowid == units.rowid``; lex_stats/lex_df maintained by the
# seed helper exactly as stats_v7 writes them.
_LEX_DDL = """
CREATE TABLE units(
  unit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, kind TEXT NOT NULL, parent_unit_id TEXT,
  session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,
  recorded_at_us INTEGER, occurred_start_us INTEGER, occurred_end_us INTEGER,
  occurred_precision TEXT, occurred_source TEXT, byte_start INTEGER,
  byte_end INTEGER, generation INTEGER NOT NULL);
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when",
  tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE unit_fts_stem USING fts5(
  text, tokenize='porter unicode61');
CREATE VIRTUAL TABLE unit_fts_tri USING fts5(
  text, tokenize='trigram');
CREATE VIRTUAL TABLE unit_fts_vocab USING fts5vocab('unit_fts','col');
CREATE TABLE lex_stats(
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL, field TEXT NOT NULL,
  stats_version TEXT NOT NULL,
  n_units INTEGER NOT NULL, total_len INTEGER NOT NULL,
  PRIMARY KEY(scope_id, generation, field, stats_version));
CREATE TABLE lex_df(
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL, field TEXT NOT NULL,
  term TEXT NOT NULL, stats_version TEXT NOT NULL,
  df INTEGER NOT NULL,
  PRIMARY KEY(scope_id, generation, field, term, stats_version));
"""

_FIELDS = ("text", "speaker", "entities", "session", "when")
_STATS_VERSION = "bm25f/v1"


def _lex_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_LEX_DDL)
    return conn


def _seed_unit(conn, unit_id, *, text="", speaker="", entities="",
               session="", when="", scope="s", gen=1, source_id=None,
               revision=1, kind="turn"):
    """One unit + its FTS shadows, maintaining lex_stats/lex_df exactly
    the way stats_v7 does (V7-06.06)."""
    cur = conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, source_id or unit_id, revision, scope, kind, None,
         "sess", 0, speaker, "user_stated", 0, 0, 0, "instant", "explicit",
         0, len(text.encode()), gen),
    )
    rid = int(cur.lastrowid)
    conn.execute(
        'INSERT INTO unit_fts(rowid,text,speaker,entities,session,"when")'
        " VALUES(?,?,?,?,?,?)",
        (rid, text, speaker, entities, session, when))
    conn.execute("INSERT INTO unit_fts_stem(rowid,text) VALUES(?,?)",
                 (rid, text))
    conn.execute("INSERT INTO unit_fts_tri(rowid,text) VALUES(?,?)",
                 (rid, text))
    field_text = {"text": text, "speaker": speaker, "entities": entities,
                  "session": session, "when": when}
    for f in _FIELDS:
        toks = lex._tokenize(field_text[f])
        conn.execute(
            "INSERT INTO lex_stats(scope_id,generation,field,"
            "stats_version,n_units,total_len) VALUES(?,?,?,?,1,?)"
            " ON CONFLICT(scope_id,generation,field,stats_version)"
            " DO UPDATE SET"
            " n_units=n_units+1, total_len=total_len+excluded.total_len",
            (scope, gen, f, _STATS_VERSION, len(toks)))
        for term in set(toks):
            conn.execute(
                "INSERT INTO lex_df(scope_id,generation,field,term,"
                "stats_version,df) VALUES(?,?,?,?,?,1)"
                " ON CONFLICT(scope_id,generation,field,term,"
                "stats_version) DO UPDATE SET df=df+1",
                (scope, gen, f, term, _STATS_VERSION))
    return rid


def _mkqv(query, *, channels=None):
    """QueryViewV7 literal — folded whitespace tokens, ``channels`` may
    mark a term 'identifier'."""
    channels = channels or {}
    terms = []
    idents = []
    for m in re.finditer(r"[\w./:@#-]+", query):
        tok = m.group(0).strip("\"'")
        if not tok:
            continue
        term = lex._fold(tok)
        start_b = len(query[:m.start(0)].encode())
        end_b = start_b + len(tok.encode())
        ch = channels.get(term, "text")
        nt = NormTerm(term=term, channel=ch, byte_start=start_b,
                      byte_end=end_b)
        terms.append(nt)
        if ch == "identifier":
            idents.append(nt)
    norm = NormAnalysis(analyzer_id="norm/v2", terms=tuple(terms),
                        identifiers=tuple(idents), text=lex._fold(query))
    return QueryViewV7(
        query=query, norm=norm,
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def _mkctx(conn, *, eligible=None, scope="s", gen=1):
    return LaneContextV7(
        store=conn, scope_id=scope, generation=gen, eligible=eligible,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.LEX,), lane_weights={}),
        manifest={})


def _mkslice(cap=50, deadline_ms=60000.0):
    return LaneSlice(deadline_ms=deadline_ms, cap=cap)


class TestDfPrefetchGate:
    def test_collect_postings_gates_flood_term(self):
        """Direct: a maintained df over ``max(1400, θ·N_eligible)``
        skips the MATCH entirely — empty postings, honest stats."""
        conn = _lex_db()
        try:
            rid_rare = _seed_unit(conn, "u-rare", text="the rare bird")
            _seed_unit(conn, "u-flood", text="flood flood flood here")
            uni, truncated = lex._universe(
                conn, "s", 1, lex._Deadline(60000.0))
            assert not truncated
            elig_rowids = set(uni.by_rowid)
            qterms = [lex._QTerm(term="flood"), lex._QTerm(term="rare")]
            stats: dict = {}
            res = lex._collect_postings(
                conn, qterms, [], uni, elig_rowids, True,
                lex._Deadline(60000.0), stats,
                n_eligible=7000,
                df_maintained={"flood": 5000, "rare": 1},
            )
            assert res is not None
            posts, nominated = res
            assert posts["flood"] == set()
            assert posts["rare"] == {rid_rare}
            assert nominated == {rid_rare}
            assert stats["df_prefetch_gated"] == [
                {"term": "flood", "df": 5000}
            ]
            assert {
                "term": "flood", "df": 5000, "reason": "df_threshold"
            } in stats["nomination_dropped"]
            # the gated term consumes no nomination slot
            assert stats["nominated_terms"] == 1
            # scoring reports the maintained df, never the empty-postings 0
            columns = [
                c for c in lex._fts_columns(conn, "unit_fts")
                if c in lex.FIELD_W
            ]
            stats2: dict = {}
            lex._score_candidates(
                conn, qterms, [], posts, nominated, uni, columns,
                {f: 2.0 for f in columns}, len(elig_rowids),
                lex._Deadline(60000.0), stats2,
                df_override={"flood": 5000},
            )
            assert stats2["df"]["flood"] == 5000
            assert stats2["df"]["rare"] == 1
        finally:
            conn.close()

    def test_lane_lexical_end_to_end(self):
        """Through ``lane_lexical``: a real ``lex_df`` row drives the
        gate, and the maintained df reaches ``stats["df"]``."""
        conn = _lex_db()
        try:
            _seed_unit(conn, "u-rare", text="the rare bird sings")
            _seed_unit(conn, "u-flood", text="flood flood flood here")
            # a session-kind unit is leaf-excluded from the universe
            # (same wave): it cannot nominate or deliver
            _seed_unit(conn, "u-sess", text="flood rare container",
                       kind="session")
            # maintained stats claim "flood" lives in 5000 docs — the
            # pre-fetch gate trusts them over the tiny fixture corpus
            conn.execute(
                "INSERT OR REPLACE INTO lex_df(scope_id,generation,field,"
                "term,stats_version,df) VALUES('s',1,'text','flood',?,5000)",
                (_STATS_VERSION,),
            )
            eligible = types.SimpleNamespace(
                unit_ids=frozenset({"u-rare", "u-flood", "u-sess"}))
            out = lex.lane_lexical(
                _mkctx(conn, eligible=eligible),
                _mkqv("flood rare"), _mkslice())
            assert out.status == LaneStatus.OK
            assert out.stats["eligible_via"] == "set"
            # leaf-only universe: the session row is not a candidate
            # surface (its rowid never enters the eligible universe)
            assert out.stats["n_universe"] == 2
            assert out.stats["df_maintained"]["flood"] == 5000
            assert out.stats["df_prefetch_gated"] == [
                {"term": "flood", "df": 5000}
            ]
            assert {
                "term": "flood", "df": 5000, "reason": "df_threshold"
            } in out.stats["nomination_dropped"]
            assert out.stats["df"]["flood"] == 5000
            assert out.stats["df"]["rare"] == 1
            # V8-07.03: the flood term still cannot *nominate* — but the
            # bounded rescue scans the rarest gated term's eligible
            # postings, so u-flood arrives marked ``signals.rescue``.
            rescue = out.stats["coverage_lexical"]["rescue"]
            assert rescue["fired"] is True
            assert rescue["terms"] == ["flood"]
            assert rescue["rows"] <= 4096
            by_id = {c.unit_id: c for c in out.candidates}
            assert set(by_id) == {"u-rare", "u-flood"}
            assert by_id["u-flood"].signals.get("rescue") is True
            assert by_id["u-rare"].signals.get("rescue") is not True
        finally:
            conn.close()


# ===========================================================================
# 8 — project_units_v7: graph_edges rows + observations_v7 stats
# ===========================================================================


def _jrev(payload, source_id="s1", revision=1, meta=None,
          event_us=1_683_417_600_000_000):
    import json as _json

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return {
        "source_id": source_id,
        "revision": revision,
        "payload": payload,
        "payload_hmac_hex": "00",
        "event_us": event_us,
        "captured_us": event_us + 1000,
        "timezone": None,
        "provenance": "direct_user",
        "metadata_json": _json.dumps(meta or {}),
    }


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "v7.db"))
    yield s
    s.close()


class TestProjectUnitsArtifacts:
    def test_graph_edges_written_and_observations_reported(self, store):
        msgs = [
            {"role": "user", "speaker": "alice",
             "content": "Alice moved to Oslo in May."},
            {"role": "assistant", "speaker": "gina",
             "content": "Oslo suits you, Alice."},
        ]
        payload = _transcript(msgs)
        with store.tx() as conn:
            stats = uj.project_units_v7(
                conn,
                source_row=_src("s1", speaker="alice", scope="scope:1"),
                revision_row=_jrev(payload, "s1", 1),
                add_args={"messages": msgs, "session_id": "sess-1"},
                generation=1,
                scope_id="scope:1",
            )
        # stats surface both artifact planes — not "skipped"
        assert isinstance(stats["graph_edges"], int)
        assert stats["graph_edges"] >= 1
        obs = stats["observations_v7"]
        assert isinstance(obs, dict)
        assert obs["status"] == "ok"
        assert obs["processed"] == stats["units"]  # the whole new slice
        assert isinstance(obs["written"], int)
        assert isinstance(obs["updated"], int)
        assert stats["errors"] == []
        with store.read() as conn:
            rows = conn.execute(
                "SELECT type, src_unit, dst_unit, scope_id, generation"
                " FROM graph_edges ORDER BY type, src_unit, dst_unit"
            ).fetchall()
        assert len(rows) >= 1
        assert all(r[3] == "scope:1" and r[4] == 1 for r in rows)
        # the two-turn session yields the family edges
        types_seen = {r[0] for r in rows}
        assert types_seen & {"same_session", "adjacent_turn"}

    def test_replay_does_not_duplicate_edges(self, store):
        """``build_edges`` replaces derived edges incident to the rebuilt
        units — a same-generation replay must not grow the table."""
        msgs = [
            {"role": "user", "speaker": "alice",
             "content": "Alice moved to Oslo."},
            {"role": "assistant", "speaker": "gina",
             "content": "Oslo suits you."},
        ]
        payload = _transcript(msgs)
        for _ in range(2):
            with store.tx() as conn:
                uj.project_units_v7(
                    conn,
                    source_row=_src("s1", speaker="alice", scope="scope:1"),
                    revision_row=_jrev(payload, "s1", 1),
                    add_args={"messages": msgs, "session_id": "sess-1"},
                    generation=1,
                    scope_id="scope:1",
                )
        with store.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0]
            obs_rows = conn.execute(
                "SELECT COUNT(*) FROM consolidation_seen_v7"
            ).fetchone()[0]
        assert n >= 1
        # second replay rewrote the same slice — edge count stable
        with store.tx() as conn:
            uj.project_units_v7(
                conn,
                source_row=_src("s1", speaker="alice", scope="scope:1"),
                revision_row=_jrev(payload, "s1", 1),
                add_args={"messages": msgs, "session_id": "sess-1"},
                generation=1,
                scope_id="scope:1",
            )
        with store.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0] == n
        assert obs_rows >= 1
