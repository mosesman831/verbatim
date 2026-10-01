"""Shared fixtures for storage-layer tests."""

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
)
from verbatim.storage.repos import ensure_scope
from verbatim.storage.store import Store


@pytest.fixture()
def store_path(tmp_path: Path) -> str:
    return str(tmp_path / "data" / "v.db")


@pytest.fixture()
def store(store_path: str):
    s = Store.create(store_path)
    yield s
    s.close()


def make_scope(**kw) -> Scope:
    base = dict(
        profile_id="prof",
        principal_id="alice",
        workspace_id="ws1",
        conversation_id="conv1",
        visibility=Visibility.CONVERSATION,
    )
    base.update(kw)
    return Scope(**base)


def make_envelope(scope: Scope, payload: bytes = b"hello", **kw) -> SourceEnvelope:
    base = dict(
        origin="test-harness",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id="alice",
        payload=payload,
        event_us=1_700_000_000_000_000,
        captured_us=1_700_000_000_000_001,
        timezone="UTC",
        provenance=Provenance.DIRECT_USER,
        external_id=None,
    )
    base.update(kw)
    return SourceEnvelope(**base)


@pytest.fixture()
def scope() -> Scope:
    return make_scope()


@pytest.fixture()
def scope_id(store: Store, scope: Scope) -> str:
    """A persisted scope row id for scope-constrained repo tests."""
    with store.tx() as conn:
        sid = ensure_scope(store, conn, scope)
    return sid
