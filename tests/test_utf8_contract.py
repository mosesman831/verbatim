"""F4-19 / V4-05.11 / V4-13.11/13.12: strict UTF-8 contract across every
public ingestion and harvesting surface.

The text channel accepts only well-formed UTF-8. Malformed bytes produce a
typed ``VerbatimError(ErrorCode.VALIDATION)`` — never a replacement decode,
never a silent skip, never an empty candidate set:

- V2 ``Ingester.ingest`` / ``Engine.ingest``: the gate rejects before any
  persistence, so screening, quarantine, and harvest all consume exactly
  the bytes that were accepted (V4-13.11).
- V3 ``ingest_envelope``: ``_payload_bytes`` validates inline content —
  including structural kinds whose harvester computes byte offsets.
- Drain time: a payload that somehow persisted malformed (corruption or a
  pre-fix row) fails the harvest job with a typed, non-retryable
  VALIDATION error — a decode failure never implies clean content
  (V4-13.12) on either the v2 prose path or the v3 structural path.
- ``harvest_v3``: the public structural segmenter raises VALIDATION on
  malformed bytes instead of returning ``[]``.
- ``Engine.remember`` byte locators: a range splitting a UTF-8 character
  is rejected; offsets are byte-exact by construction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from verbatim.api import Engine
from verbatim.config import VerbatimConfig, config_from_mapping
from verbatim.core.identity import scope_key
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
)
from verbatim.core.types_v3 import (
    CaptureAuthorization,
    EnvelopeKind,
    Perspective,
    SourceEnvelopeV3,
)
from verbatim.evidence import ingest_envelope
from verbatim.host import LocalHost
from verbatim.ingest import Ingester, _harvest_v3_result
from verbatim.storage.repos import SourcesRepo
from verbatim.storage.store import Store


SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")

MALFORMED = b"\xff\xfe invalid \x80 utf-8"


def _cfg() -> VerbatimConfig:
    cfg = VerbatimConfig()
    return replace(
        cfg,
        capture=replace(
            cfg.capture,
            enabled=True,
            user_messages=True,
            assistant_context=True,
            tool_outputs=True,
        ),
    )


@pytest.fixture()
def store(tmp_path):
    s = Store.create(str(tmp_path / "utf8.db"))
    yield s
    s.close()


def _env(
    payload: bytes,
    kind: SourceKind = SourceKind.USER_MESSAGE,
    scope: Scope = SCOPE,
) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test:utf8",
        source_kind=kind,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=payload,
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )


def _v3_env(kind: EnvelopeKind, content: bytes) -> SourceEnvelopeV3:
    return SourceEnvelopeV3(
        kind=kind,
        scope_id="sA",
        actor_principal="u1",
        perspective=Perspective(asserter="u1"),
        content=content,
        media_type="text/plain",
        host_id="h1",
        session_id="ss1",
        external_id=None,
        event_us=1000,
        receipt_us=1001,
        metadata={},
    )


def _count(store: Store, table: str) -> int:
    with store.read() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --------------------------------------------------------------------------
# V2 ingestion gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        SourceKind.USER_MESSAGE,
        SourceKind.IMPORT,
        SourceKind.OPERATOR_RECORD,
        SourceKind.TOOL_OUTPUT,
        SourceKind.ASSISTANT_MESSAGE,
    ],
)
def test_v2_gate_rejects_malformed_payload(store, kind):
    """Every source kind on the text channel is UTF-8-strict — the check is
    independent of the optional screening tables and the capture flags."""
    ing = Ingester(store, _cfg())
    with pytest.raises(VerbatimError) as ei:
        ing.ingest(_env(MALFORMED, kind))
    assert ei.value.code is ErrorCode.VALIDATION
    # Rejected before persistence: no source, revision, job, or event.
    assert _count(store, "sources") == 0
    assert _count(store, "source_revisions") == 0
    assert _count(store, "jobs") == 0


def test_engine_ingest_rejects_malformed_payload(store):
    """The Engine write path runs the same gate — the public API surface is
    covered identically to the library ingester."""
    engine = Engine(
        store,
        _cfg(),
        LocalHost(profile_id="p", principal_id="alice",
                  conversation_id="c1"),
    )
    with pytest.raises(VerbatimError) as ei:
        engine.ingest(_env(MALFORMED))
    assert ei.value.code is ErrorCode.VALIDATION
    assert _count(store, "sources") == 0


def test_valid_multibyte_payload_accepted_and_byte_exact(store):
    """Screening and harvest consume exactly the accepted bytes: a multibyte
    payload yields spans whose stored byte offsets decode to the candidate
    text — no replacement characters, no drifted offsets (V4-13.11)."""
    ing = Ingester(store, _cfg())
    text = "Mon éditeur est néovim — 日本語も大丈夫。"
    receipt = ing.ingest(_env(text.encode("utf-8")))
    assert receipt.accepted
    ing.run_pending(scope=SCOPE)
    with store.read() as conn:
        rows = conn.execute(
            "SELECT s.start_byte, s.end_byte, sr.payload"
            " FROM spans s JOIN source_revisions sr"
            "  ON sr.source_id = s.source_id AND sr.revision = s.revision"
        ).fetchall()
    assert rows, "expected harvested spans"
    payload = text.encode("utf-8")
    for sb, eb, raw in rows:
        assert bytes(raw) == payload
        # Each span decodes strictly and is a verbatim substring.
        assert payload[sb:eb].decode("utf-8")


# --------------------------------------------------------------------------
# V3 inline envelopes (incl. structural kinds)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        EnvelopeKind.USER_MESSAGE,
        EnvelopeKind.SYSTEM_EVENT,
        EnvelopeKind.TOOL_RESULT,
        EnvelopeKind.FILE_DIFF,
        EnvelopeKind.TEST_RESULT,
        EnvelopeKind.IMPORT,
    ],
)
def test_v3_inline_envelope_rejects_malformed(store, kind):
    """``_payload_bytes`` strictly validates inline content — a malformed
    envelope never persists a source whose harvest offsets would be
    meaningless (V4-05.11 inline envelope + structural event coverage).

    Agent-submitted kinds need an explicit capture authorization to reach
    the payload check (§11.11), so one is supplied for every kind."""
    auth = CaptureAuthorization(
        authorization_id="authz-utf8",
        issuer_id="op-1",
        principal_id="u1",
        allowed_kinds=frozenset(EnvelopeKind),
        scope_ids=frozenset({"sA"}),
        retention_policy="task",
        policy_revision="pol-1",
        issued_us=1,
    )
    with pytest.raises(VerbatimError) as ei:
        with store.tx() as conn:
            ingest_envelope(
                conn, store, _v3_env(kind, MALFORMED), authorization=auth
            )
    assert ei.value.code is ErrorCode.VALIDATION
    assert _count(store, "sources") == 0
    assert _count(store, "source_envelopes") == 0


# --------------------------------------------------------------------------
# Drain-time decode failures
# --------------------------------------------------------------------------


def _persist_bypassing_gate(
    store: Store, payload: bytes, *, envelope_kind: str | None = None
) -> tuple[str, int]:
    """Persist a malformed source as if it predated strict acceptance (or
    was corrupted on disk), then enqueue its harvest job."""
    ing = Ingester(store, _cfg())
    env = _env(payload)
    with store.tx() as conn:
        sid, created = ing.sources.insert(env, conn=conn)
        assert created
        if envelope_kind is not None:
            scope_id = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO source_envelopes"
                "(envelope_id, source_id, revision, scope_id, envelope_kind)"
                " VALUES (?, ?, ?, ?, ?)",
                (f"env_{sid[:12]}", sid, 1, scope_id, envelope_kind),
            )
        ing.jobs.enqueue(
            conn,
            scope_key(env.scope),
            JobKind.HARVEST,
            {"source_id": sid, "revision": 1},
            dedup_key=store.hmac(f"harvest:{sid}:1".encode()),
            operation_key=(
                f"harvest:{sid}:1" if ing.jobs.supports_durability else None
            ),
        )
    return sid, 1


def test_persisted_malformed_payload_fails_harvest_typed(store):
    """v2 prose path: a malformed persisted payload is a permanent typed
    VALIDATION failure — never retried, never harvested into guessed
    offsets (V4-13.12)."""
    ing = Ingester(store, _cfg())
    sid, rev = _persist_bypassing_gate(store, MALFORMED)
    ing.run_pending(scope=SCOPE, kinds=[JobKind.HARVEST])
    with store.read() as conn:
        job = conn.execute(
            "SELECT state, error_code FROM jobs WHERE kind = 'harvest'"
        ).fetchone()
    assert job == ("failed", ErrorCode.VALIDATION.value)
    assert _count(store, "spans") == 0
    assert _count(store, "jobs") >= 1  # failed row remains auditable


def test_persisted_malformed_structural_payload_fails_typed(store):
    """v3 structural path: a ``tool_result``-kind source whose stored bytes
    are malformed also fails typed — ``_harvest_v3_result`` decodes
    strictly before ``harvest_v3`` computes offsets (V4-13.11/13.12)."""
    ing = Ingester(store, _cfg())
    sid, rev = _persist_bypassing_gate(
        store, MALFORMED, envelope_kind="tool_result"
    )
    ing.run_pending(scope=SCOPE, kinds=[JobKind.HARVEST])
    with store.read() as conn:
        job = conn.execute(
            "SELECT state, error_code FROM jobs WHERE kind = 'harvest'"
        ).fetchone()
    assert job == ("failed", ErrorCode.VALIDATION.value)
    assert _count(store, "spans") == 0


def test_harvest_v3_result_raises_decode_error_on_malformed(store):
    """The routing helper surfaces the raw ``UnicodeDecodeError`` so the
    worker maps it once — both v3-structural and v2-prose branches."""
    ing = Ingester(store, _cfg())
    env = _env(MALFORMED)
    with store.tx() as conn:
        sid, _ = ing.sources.insert(env, conn=conn)
        scope_id = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO source_envelopes"
            "(envelope_id, source_id, revision, scope_id, envelope_kind)"
            " VALUES (?, ?, ?, ?, ?)",
            (f"env_{sid[:12]}", sid, 1, scope_id, "file_diff"),
        )
    from verbatim.ingest import envelope_for

    envelope = envelope_for(store, sid, 1)
    with pytest.raises(UnicodeDecodeError):
        _harvest_v3_result(store, sid, 1, envelope)


def test_screen_job_over_malformed_source_fails_typed(store):
    """A ``screen`` job whose ``source`` ref resolves to undecodable bytes
    fails VALIDATION at the ingest-side preflight — screening never
    consumes a replacement-decoded view of the payload (V4-13.12)."""
    ing = Ingester(store, _cfg())
    sid, rev = _persist_bypassing_gate(store, MALFORMED)
    with store.tx() as conn:
        ing.jobs.enqueue(
            conn,
            scope_key(SCOPE),
            JobKind.SCREEN,
            {"object_kind": "source", "object_id": sid, "revision": rev},
            dedup_key=store.hmac(f"screen:{sid}:{rev}".encode()),
            operation_key=(
                f"screen:{sid}:{rev}"
                if ing.jobs.supports_durability
                else None
            ),
        )
    ing.run_pending(scope=SCOPE, kinds=[JobKind.SCREEN])
    with store.read() as conn:
        job = conn.execute(
            "SELECT state, error_code FROM jobs WHERE kind = 'screen'"
        ).fetchone()
        # No label attached — malformed bytes never screened as clean.
        n_labels = conn.execute(
            "SELECT COUNT(*) FROM security_labels"
        ).fetchone()[0]
    assert job == ("failed", ErrorCode.VALIDATION.value)
    assert n_labels == 0


# --------------------------------------------------------------------------
# harvest_v3 public surface
# --------------------------------------------------------------------------


def test_harvest_v3_public_surface_rejects_malformed():
    """The public segmenter is a typed-failure surface too (F4-19): bytes
    that cannot decode raise VALIDATION rather than an empty list that
    would read as clean-but-empty content."""
    from verbatim.harvest_v3 import harvest_v3

    with pytest.raises(VerbatimError) as ei:
        harvest_v3(MALFORMED, envelope_kind="tool_call")
    assert ei.value.code is ErrorCode.VALIDATION


# --------------------------------------------------------------------------
# byte locators (grounded remember)
# --------------------------------------------------------------------------


def test_byte_locator_splitting_utf8_char_rejected(store):
    """``Engine.remember`` byte ranges are validated strictly: a range that
    splits a multibyte character or exceeds the payload is VALIDATION,
    never a best-effort decode (V4-05.11 byte-locator coverage)."""
    engine = Engine(
        store,
        config_from_mapping(
            {"capture": {"enabled": True, "user_messages": True}}
        ),
        LocalHost(profile_id="p", principal_id="alice",
                  conversation_id="c1"),
    )
    text = "Mon éditeur est néovim."
    receipt = engine.ingest(_env(text.encode("utf-8")))
    (sid,) = receipt.accepted
    scope = engine.host.default_scope()
    data = text.encode("utf-8")
    # 'é' occupies bytes [4,6) — quoting [0,5) splits it.
    assert data[4:6] == "é".encode("utf-8")
    with pytest.raises(VerbatimError) as ei:
        engine.remember(sid, 0, 5, scope)
    assert ei.value.code is ErrorCode.VALIDATION
    # Out-of-range and inverted ranges are rejected the same way.
    with pytest.raises(VerbatimError):
        engine.remember(sid, 0, len(data) + 1, scope)
    with pytest.raises(VerbatimError):
        engine.remember(sid, 6, 4, scope)
    # A boundary-aligned quote succeeds — the check is strict, not broader.
    claim_id = engine.remember(sid, 0, len(data), scope)
    assert isinstance(claim_id, str) and claim_id
