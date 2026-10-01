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
from pathlib import Path
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


# --- spec corpus guard ------------------------------------------------------
# The internal spec corpus (SPEC v1-v8.5, REQUIREMENTS, THREAT_MODEL) is
# maintained privately and is deliberately not part of this distribution, so
# evidence-audit tests that parse it cannot run from a released tree. Each
# such test names the spec file it audits; the audit is skipped only when that
# file is genuinely absent, so a development tree carrying the corpus still
# runs every one of them unchanged.
_CORPUS_AUDITS: dict[str, str] = {
    "tests/eval/test_v3_harness.py::test_registry_parses_all_requirements": "SPEC_V3.md",
    "tests/eval/test_v3_harness.py::test_seed_ledger_is_idempotent": "SPEC_V3.md",
    "tests/eval/test_v4_ledger.py::test_real_check_passes_on_seed_map": "SPEC_V4.md",
    "tests/eval/test_v4_ledger.py::test_real_spec_ids_all_enumerated": "SPEC_V4.md",
    "tests/eval/test_v5_ledger.py::test_live_overlay_claims_carry_evidence": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_repo_check_passes": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_spec_all_requirements_parsed": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_spec_gate_and_scenario_anchors": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_spec_honest_start_state": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_spec_scenarios_and_gates": "SPEC_V5.md",
    "tests/eval/test_v5_ledger.py::test_real_spec_stage_inference_spot_checks": "SPEC_V5.md",
    "tests/eval/test_v6_ledger.py::test_carried_real_repo_v5_tail": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_live_overlay_validate_clean": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_real_repo_check_passes": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_real_spec_all_requirements_parsed": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_real_spec_honest_start_state": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_real_spec_phase_and_anchor_inference": "SPEC_V6.md",
    "tests/eval/test_v6_ledger.py::test_real_spec_scenarios_and_gates": "SPEC_V6.md",
    "tests/eval/test_v7_ledger.py::test_gate_aggregation_rules": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_id_section_semantics_for_playbook_definitions": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_ledger_json_on_disk_parses_and_matches_spec": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_real_repo_check_passes": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_real_spec_all_requirements_parsed": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_real_spec_honest_start_state": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_real_spec_named_collections": "SPEC_V7.md",
    "tests/eval/test_v7_ledger.py::test_real_spec_validate_clean": "SPEC_V7.md",
    "tests/v45/test_scenarios_d2.py::test_d22_m5_report_names_every_parent_registry_comparator": "SPEC_V4_5.md",
    "tests/v5/test_scenarios_d.py::test_e72_ledger_covers_all_scenarios_nothing_falsely_verified": "SPEC_V5.md",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(__file__).resolve().parents[1]
    for item in items:
        spec = _CORPUS_AUDITS.get(item.nodeid)
        if spec and not (root / spec).exists():
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{spec} is not part of this distribution (private spec corpus)"
                )
            )
