"""Fixtures for experience-layer tests: a real Store with the v2 schema.

Uses ``Store.create`` so the genuine v2 DDL, foreign keys, and transaction
fencing apply — no SQL-level mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types import (
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    Visibility,
    new_id,
)
from verbatim.storage.repos import SourcesRepo, SpansRepo, ensure_scope
from verbatim.storage.store import Store


@pytest.fixture()
def store_path(tmp_path: Path) -> str:
    return str(tmp_path / "data" / "v.db")


@pytest.fixture()
def store(store_path: str):
    s = Store.create(store_path)
    yield s
    s.close()


@pytest.fixture()
def scope() -> Scope:
    return Scope(
        profile_id="prof",
        principal_id="alice",
        workspace_id="ws1",
        conversation_id="conv1",
        visibility=Visibility.CONVERSATION,
    )


@pytest.fixture()
def scope_id(store: Store, scope: Scope) -> str:
    with store.tx() as conn:
        sid = ensure_scope(store, conn, scope)
    return sid


def make_span(store: Store, scope: Scope, payload: bytes = b"evidence text"):
    """Persist a source + span inside one tx; returns (source_id, span_id)."""
    env = SourceEnvelope(
        origin="test-harness",
        source_kind=SourceKind.TOOL_OUTPUT,
        scope=scope,
        speaker_id="tool",
        payload=payload,
        event_us=1_700_000_000_000_000,
        captured_us=1_700_000_000_000_001,
        provenance=Provenance.APPROVED_TOOL,
    )
    source_id, _ = SourcesRepo(store).insert(env)
    span_id = new_id()
    SpansRepo(store).insert(span_id, source_id, 1, 0, len(payload), "harv-1")
    return source_id, span_id
