"""Connector test fixtures (SPEC_V4 §48).

Real ``Store.create`` (full v3/v4 schema) — connector pulls exercise the
production write path: grants_v3 + capture_authorizations consent,
``ingest_envelope`` screening/quarantine, source dedup, cursor ledger.
"""

from __future__ import annotations

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types_v3 import EnvelopeKind
from verbatim.governance import (
    create_grant,
    issue_capture_authorization,
    register_principal,
)
from verbatim.storage.store import Store

from verbatim.connectors import ConnectorService

PRINCIPAL = "alice"


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "connectors.db"))
    yield s
    s.close()


@pytest.fixture
def service(store):
    return ConnectorService(store, VerbatimConfig())


def provision(
    store: Store,
    scope_id: str = "scopeA",
    principal: str = PRINCIPAL,
    *,
    grant: bool = True,
    consent: bool = True,
) -> str | None:
    """Scope row + ``ingest`` grant + ``connector_item`` capture auth.

    Returns the authorization_id (or None when ``consent=False``).
    """
    aid = None
    with store.tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO scopes"
            "(scope_id,profile_id,principal_id,visibility)"
            " VALUES(?,?,?,'owner')",
            (scope_id, "v3", principal),
        )
        register_principal(conn, kind="human", principal_id=principal)
        if grant:
            create_grant(
                conn,
                scope_id=scope_id,
                principal_id=principal,
                verbs={"ingest"},
                issuer_id=principal,
            )
        if consent:
            aid = issue_capture_authorization(
                conn,
                principal_id=principal,
                issuer_id=principal,
                allowed_kinds={EnvelopeKind.CONNECTOR_ITEM},
                retention_policy="keep",
                policy_revision="pol1",
                scope_ids={scope_id},
            )
    return aid


def write_file(root, rel: str, content: bytes) -> None:
    from pathlib import Path

    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
