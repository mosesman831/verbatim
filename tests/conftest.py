"""Shared test fixtures: an in-memory store implementing the store contract.

The durable store (``verbatim/storage/store.py``) is built in parallel; its
contract is ``tx()``/``read()`` context managers yielding an
``sqlite3.Connection`` plus ``hmac(bytes) -> bytes``. This shim implements
exactly that contract over the real DDL so these tests exercise genuine
schema constraints, transactions, and fencing — no mocks of the SQL layer.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import pytest

from verbatim.core.types import Scope
from verbatim.storage.schema import DDL_V1


class TestStore:
    """In-memory store honoring the tx()/read()/hmac() contract.

    A single shared connection over :memory: keeps the tests simple while
    BEGIN IMMEDIATE inside tx() preserves real transactional semantics
    (isolation_level=None → manual transaction control).
    """

    def __init__(self) -> None:
        # No row_factory: the real Store yields plain tuples, so tests here
        # exercise the same cursor-description normalization the repos use.
        self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(DDL_V1)
        self._key = b"test-store-hmac-key"
        self._tx_lock = threading.RLock()

    @contextmanager
    def tx(self):
        """Write transaction: BEGIN IMMEDIATE … COMMIT/ROLLBACK."""
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @contextmanager
    def read(self):
        """Read scope: autocommit SELECTs see the latest committed snapshot."""
        yield self._conn

    def hmac(self, data: bytes) -> bytes:
        return _hmac.new(self._key, data, hashlib.sha256).digest()


class FakeClock:
    """Deterministic microsecond clock."""

    def __init__(self, start_us: int = 1_700_000_000_000_000) -> None:
        self.t = start_us

    def __call__(self) -> int:
        return self.t

    def advance(self, seconds: float) -> int:
        self.t += int(seconds * 1_000_000)
        return self.t


@pytest.fixture()
def store() -> TestStore:
    return TestStore()


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def scope() -> Scope:
    return Scope(profile_id="prof", principal_id="alice", conversation_id="conv1")


@pytest.fixture()
def scope_id(scope: Scope) -> str:
    from verbatim.jobs.queue import scope_key

    return scope_key(scope)


def qrows(conn, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Test-side row→dict helper mirroring repos._rows (tuple cursors)."""
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def qrow(conn, sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    rows = qrows(conn, sql, params)
    return rows[0] if rows else None


def grant_consent(
    store: TestStore,
    scope_id: str,
    processor: str = "typesafe",
    purpose: str = "candidate_curation",
    revoked_us: Optional[int] = None,
) -> str:
    """Insert a consent row directly (trusted-setup simulation)."""
    cid = f"consent-{scope_id}-{processor}-{purpose}"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO consents (consent_id, scope_id, processor, purpose,"
            " granted_us, revoked_us, policy_digest) VALUES (?,?,?,?,?,?,?)",
            (cid, scope_id, processor, purpose, int(time.time() * 1e6), revoked_us, "digest-1"),
        )
    return cid
