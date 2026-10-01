"""F4-17/F4-18/F4-21: the registered provider is the consolidated route
(SPEC_V4 V4-07.09, V4-07.10, V4-05.13, V4-14.07/11; C81/C82/C83).

``verbatim.register`` installs ``VerbatimMemoryProvider``. That provider
opens ONE store through the shared resolver and binds the v3 capture
surface (``HermesV3Adapter`` + ``CaptureClient``) to the engine's own
store — capture, drain, recall, correction, and deletion all flow through
the same database and job queue. Session-end drain failures are visible
and durable, never swallowed.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

import verbatim
from verbatim.adapters.hermes_v3 import HermesV3Adapter
from verbatim.core.identity import scope_key
from verbatim.core.types import (
    ErrorCode,
    TransitionCommand,
    VerbatimError,
)
from verbatim.provider import VerbatimMemoryProvider
from verbatim.storage.store import Store


class _Ctx:
    """The Hermes registration context — captures the installed provider."""

    def __init__(self) -> None:
        self.provider = None

    def register_memory_provider(self, provider) -> None:
        self.provider = provider


def _register():
    ctx = _Ctx()
    verbatim.register(ctx)
    assert isinstance(ctx.provider, VerbatimMemoryProvider)
    return ctx.provider


def _home(tmp_path, *, capture=True):
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    cfg = (
        "memory:\n"
        "  verbatim:\n"
        "    capture:\n"
        f"      enabled: {'true' if capture else 'false'}\n"
        "      assistant_context: true\n"
        "      tool_outputs: true\n"
    )
    (home / "config.yaml").write_text(cfg, encoding="utf-8")
    return str(home)


def _profile_id(home: str) -> str:
    return "p" + hashlib.sha256(os.path.realpath(home).encode()).hexdigest()[:24]


def _data_dir(home: str) -> str:
    return os.path.join(home, "verbatim")


def _jobs(store, state=None):
    sql = "SELECT kind, state FROM jobs"
    params = ()
    if state is not None:
        sql += " WHERE state = ?"
        params = (state,)
    with store.read() as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _pending(store):
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM jobs"
            " WHERE state IN ('queued','retry_wait','leased')"
        ).fetchone()[0]


def _events(store, kind):
    with store.read() as conn:
        return conn.execute(
            "SELECT kind, payload_json FROM events WHERE kind = ?", (kind,)
        ).fetchall()


def _admit_all(prov):
    """Operator correction: admit every pending claim through the engine's
    own transition path (C81 — correction runs the consolidated route)."""
    with prov._engine.store.read() as conn:
        heads = dict(
            conn.execute(
                "SELECT claim_id, revision FROM claim_revisions"
                " WHERE state='pending'"
            ).fetchall()
        )
    for cid, rev in heads.items():
        prov._engine.apply_transition(
            TransitionCommand(
                claim_id=cid,
                expected_revision=rev,
                effect="admit",
                actor_id="op",
                reason="operator approval",
            ),
            scope=prov._session_scope("sess-1"),
        )
    prov._engine._ingester.run_pending(scope=None, limit=64)
    return heads


def _recall(prov, query, session_id="sess-1"):
    return json.loads(
        prov.handle_tool_call(
            "verbatim_recall", {"query": query}, session_id=session_id
        )
    )


# ---------------------------------------------------------------------
# C81 — the registered provider IS the consolidated engine route
# ---------------------------------------------------------------------


def test_register_constructs_consolidated_route(tmp_path):
    home = _home(tmp_path)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")
    try:
        # Exactly ONE store file, named by the profile convention — the
        # adapter never mints a parallel v3.db (F4-18).
        dbs = {
            n
            for n in os.listdir(_data_dir(home))
            if n.endswith(".db")
        }
        assert dbs == {f"{_profile_id(home)}.db"}
        # Capture runs the SAME store the engine recalls/corrects from.
        assert isinstance(prov._capture, HermesV3Adapter)
        assert prov._capture_client.store is prov._engine.store
        assert prov._engine.store.path == os.path.join(
            _data_dir(home), f"{_profile_id(home)}.db"
        )
    finally:
        prov.shutdown()


def test_registered_roundtrip_capture_drain_recall_correct_restart_delete(
    tmp_path,
):
    home = _home(tmp_path)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")
    try:
        prov.sync_turn(
            "I prefer aisle seats on flights",
            "Noted — aisle seats Friday.",
            session_id="sess-1",
        )
        # v3 envelopes landed in the engine's store (same transaction).
        with prov._engine.store.read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] >= 1
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_envelopes"
                ).fetchone()[0]
                >= 1
            )
        assert _pending(prov._engine.store) >= 1  # harvest obligation queued

        # Session end drains through the engine's provisioned ingester.
        prov.on_session_end([])
        assert prov.drain_error is None
        assert prov._last_drain["drained"] >= 1
        assert prov.drain_status()["pending_jobs"] == 0

        # Correction through the same engine admits the pending claims.
        heads = _admit_all(prov)
        assert heads, "capture produced no claims to correct"
        out = _recall(prov, "aisle seats")
        assert out["ok"] is True
        assert out["data"]["items"], "admitted evidence must recall"
    finally:
        prov.shutdown()

    # Restart: a fresh registered provider over the same home recalls the
    # same store — persistence across provider lifecycle (C81).
    prov2 = _register()
    prov2.initialize("sess-2", hermes_home=home, user_id="alice")
    try:
        out = _recall(prov2, "aisle seats", session_id="sess-2")
        assert out["ok"] is True
        # Deletion runs the consolidated engine too: tombstone the claim
        # itself and the recall surface honors the suppression.
        with prov2._engine.store.read() as conn:
            cid = conn.execute(
                "SELECT claim_id FROM claims LIMIT 1"
            ).fetchone()[0]
        prov2._engine.suppress(
            [("claim", cid)], scope=prov2._session_scope("sess-1")
        )
        prov2._engine._ingester.run_pending(scope=None, limit=64)
        out2 = _recall(prov2, "aisle seats", session_id="sess-1")
        assert out2["ok"] is True
        assert not out2["data"]["items"]
    finally:
        prov2.shutdown()


def test_provider_store_conflict_requires_operator_decision(tmp_path):
    home = _home(tmp_path)
    data = _data_dir(home)
    os.makedirs(data, exist_ok=True)
    legacy = Store.create(os.path.join(data, f"{_profile_id(home)}.db"))
    legacy.close()
    v3 = Store.create(os.path.join(data, "v3.db"))
    v3.close()

    prov = _register()
    # Both conventions on disk → typed conflict, never a silent pick.
    with pytest.raises(VerbatimError) as exc:
        prov.initialize("sess-1", hermes_home=home, user_id="alice")
    assert exc.value.code == ErrorCode.STORE_CONFLICT

    # The operator decision channels resolve it — same consolidated route.
    prov.initialize(
        "sess-1",
        hermes_home=home,
        user_id="alice",
        store_path=os.path.join(data, "v3.db"),
    )
    try:
        assert prov._engine.store.path.endswith("v3.db")
        assert prov._capture.client.store is prov._engine.store
    finally:
        prov.shutdown()

    prov.initialize(
        "sess-1", hermes_home=home, user_id="alice", prefer_store="legacy"
    )
    try:
        assert prov._engine.store.path.endswith(f"{_profile_id(home)}.db")
    finally:
        prov.shutdown()


# ---------------------------------------------------------------------
# C83 / V4-14.07/11 — drain failure is visible + durable, never fatal
# ---------------------------------------------------------------------


def test_drain_failure_is_visible_nonfatal_and_durable(tmp_path, caplog):
    home = _home(tmp_path)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")
    try:
        prov.sync_turn("I prefer aisle seats on flights", "noted")
        assert _pending(prov._engine.store) >= 1

        def boom(*a, **k):
            raise VerbatimError(ErrorCode.STORE_BUSY, "injected drain failure")

        real = prov._engine._ingester.drain_report
        prov._engine._ingester.drain_report = boom
        try:
            # The hook does not raise into the host — but it is not a
            # success either.
            prov.on_session_end([])
        finally:
            prov._engine._ingester.drain_report = real

        # Visible: status surface carries the typed error + pending count.
        err = prov.drain_error
        assert err is not None
        assert err["code"] == ErrorCode.STORE_BUSY.value
        assert "injected drain failure" in err["error"]
        assert err["pending_jobs"] >= 1
        status = prov.drain_status()
        assert status["drain_error"] is err or status["drain_error"] == err
        assert status["pending_jobs"] >= 1

        # Durable: the failure is journaled; the obligation stays queued.
        ev = _events(prov._engine.store, "session_end_drain_failed")
        assert ev, "drain failure was not journaled"
        payload = json.loads(ev[0][1])
        assert payload["code"] == ErrorCode.STORE_BUSY.value
        assert _pending(prov._engine.store) >= 1

        # Non-fatal: read surfaces keep working after the failure, and
        # capture still lands (a second builtin-predicate turn).
        assert prov.prefetch("deploy") is not None
        assert _recall(prov, "deploy")["ok"] is True
        prov.sync_turn("I prefer window seats too", "noted")

        # And once the fault clears, the same seam completes honestly.
        prov.on_session_end([])
        assert prov.drain_error is None
        assert prov.drain_status()["pending_jobs"] == 0
    finally:
        prov.shutdown()


def test_drain_failure_reaches_host_log(tmp_path, caplog):
    """Diagnostics flow through ``host.log`` — the stdlib fallback emits
    a real record outside Hermes (V4-14.11)."""
    home = _home(tmp_path)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")

    class Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("drain exploded")

    try:
        prov._capture.client.drain_pending = Boom()
        # Provider drains via engine._ingester — patch that too so BOTH
        # drain channels prove non-silent independently of which runs.
        real = prov._engine._ingester.drain_report
        prov._engine._ingester.drain_report = Boom()
        with caplog.at_level("ERROR", logger="verbatim"):
            prov.on_session_end([])
        prov._engine._ingester.drain_report = real
        assert prov.drain_error is not None
        assert any(
            "verbatim_session_end_drain_failed" in r.getMessage()
            for r in caplog.records
        )
    finally:
        prov.shutdown()


def test_capture_disabled_declines_writes_but_drains(tmp_path):
    home = _home(tmp_path, capture=False)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")
    try:
        assert prov._capture is None  # no capture surface armed
        prov.sync_turn("secret", "reply")  # silent decline — no rows
        with prov._engine.store.read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
        prov.on_session_end([])  # drain seam still runs, cleanly
        assert prov.drain_error is None
        assert prov._last_drain["drained"] == 0
    finally:
        prov.shutdown()


def test_subagent_context_never_mints_write_records(tmp_path):
    home = _home(tmp_path)
    prov = _register()
    prov.initialize(
        "sess-1", hermes_home=home, user_id="alice", agent_context="subagent"
    )
    try:
        assert prov._capture is None
        prov.sync_turn("hi", "hello")
        with prov._engine.store.read() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
            )
            # No grant/authorization provisioning either (SPEC §33).
            for table in ("grants_v3", "capture_authorizations"):
                assert (
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                    == 0
                )
    finally:
        prov.shutdown()


def test_session_switch_rebinds_capture_scope(tmp_path):
    home = _home(tmp_path)
    prov = _register()
    prov.initialize("sess-1", hermes_home=home, user_id="alice")
    try:
        prov.sync_turn("I prefer aisle seats", "ok")
        prov.on_session_switch("sess-2")
        prov.sync_turn("I prefer window seats", "ok")
        prov.on_session_end([])
        assert prov.drain_error is None
        # Both turns captured; the second lands under the new scope.
        with prov._engine.store.read() as conn:
            scopes = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT scope_id FROM sources"
                ).fetchall()
            }
        assert len(scopes) == 2
    finally:
        prov.shutdown()
