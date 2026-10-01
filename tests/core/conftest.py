"""Shared fixtures for core tests: the real ``Store`` on a ``tmp_path`` plus
seed helpers that go through the real repositories.

Using the production storage layer (not a stub) means these tests verify the
core modules against the actual contracts: scope upserts with ``profile_id``,
bitemporal claim revisions, UTF-8-boundary-checked spans, keyed HMACs, and the
append-only event journal. Seed helpers compose repo calls in single
transactions — ``Store.tx()`` is deliberately non-reentrant, so helpers never
nest one tx inside another.
"""

from __future__ import annotations

from typing import Optional

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import (
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    SpanRef,
    TimeInterval,
    json_dumps,
    new_id,
)
from verbatim.storage.repos import (
    ClaimsRepo,
    EventsRepo,
    SourcesRepo,
    SpansRepo,
    ensure_scope,
    scope_id_for,
)
from verbatim.storage.store import Store


def make_scope(
    profile_id: str = "prof",
    principal_id: str = "alice",
    workspace_id: str = "ws",
    conversation_id: str = "conv",
) -> Scope:
    return Scope(
        profile_id=profile_id,
        principal_id=principal_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )


def scope_id_of(store: Store, scope: Scope) -> str:
    return scope_id_for(store, scope)


def make_envelope(
    scope: Scope,
    text: str,
    *,
    kind: SourceKind = SourceKind.USER_MESSAGE,
    provenance: Provenance = Provenance.DIRECT_USER,
    speaker_id: Optional[str] = "alice",
    event_us: Optional[int] = None,
) -> SourceEnvelope:
    now = now_us()
    return SourceEnvelope(
        origin="test",
        source_kind=kind,
        scope=scope,
        speaker_id=speaker_id,
        payload=text.encode("utf-8"),
        event_us=event_us or now,
        captured_us=now,
        provenance=provenance,
    )


def insert_source(store: Store, envelope: SourceEnvelope) -> str:
    """Persist an envelope; the repo ensures the scope row in the same tx."""
    source_id, _created = SourcesRepo(store).insert(envelope)
    return source_id


def insert_span(
    store: Store,
    source_id: str,
    revision: int,
    start_byte: int,
    end_byte: int,
) -> str:
    return SpansRepo(store).insert(
        new_id(), source_id, revision, start_byte, end_byte, "harvest-1"
    )


def seed_span_for(store: Store, envelope: SourceEnvelope, text: Optional[str] = None) -> SpanRef:
    """Persist an envelope and a span covering its whole payload."""
    source_id = insert_source(store, envelope)
    payload = envelope.payload if text is None else text.encode("utf-8")
    span_id = insert_span(store, source_id, envelope.revision, 0, len(payload))
    return SpanRef(span_id, source_id, envelope.revision, 0, len(payload))


def seed_claim(
    store: Store,
    scope: Scope,
    *,
    predicate: Optional[str] = "editor",
    object_text: Optional[str] = "neovim",
    state: str = "active",
    condition_json: Optional[str] = None,
    polarity: str = "affirmative",
    modality: str = "asserted",
    subject_id: Optional[str] = "alice",
    with_evidence: bool = True,
    intervals: Optional[list] = None,
    span_text: str = "I use neovim for everything.",
) -> tuple[str, int]:
    """Create a claim with one revision; returns ``(claim_id, revision)``."""
    envelope = make_envelope(scope, span_text, speaker_id=subject_id)
    span = seed_span_for(store, envelope) if with_evidence else None
    with store.tx() as conn:
        sid = ensure_scope(store, conn, scope)
        seq = EventsRepo(store).append(
            conn, sid, "claim_proposed", "test", {"seed": True}, "test-1"
        )
        claim_id = ClaimsRepo(store).create(sid, subject_id, predicate, conn)
        rev = ClaimsRepo(store).add_revision(
            claim_id,
            state,
            json_dumps({"kind": "literal", "text": object_text})
            if object_text
            else None,
            polarity,
            modality,
            condition_json,
            json_dumps({"seed": True}),
            intervals if intervals is not None else [TimeInterval()],
            [(span.span_id, "primary")] if span else [],
            seq,
            conn,
        )
        return claim_id, rev


def q(store: Store, sql: str, params: tuple = ()) -> list:
    """Run a read query inside a consistent snapshot."""
    with store.read() as conn:
        return conn.execute(sql, params).fetchall()


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store.create(str(tmp_path / "v.db"))
    yield s
    s.close()


@pytest.fixture
def scope() -> Scope:
    return make_scope()
