"""Encoder contract, factory, and the span-embedding write path.

Semantic retrieval is optional and independent of any remote decision
service (SPEC §28). An ``Encoder`` turns already-authorized quotation text
into packed little-endian float32 vectors. Two identities matter:

- ``encoder_id`` — stable identity ``backend:model:revision`` recorded on
  every embeddings row and manifest so query/document vectors can never be
  silently mixed across model generations (SPEC §28: query and document
  vectors must use the same encoder identity).
- ``preprocessing_version`` — the engine-side request/normalization
  contract revision, stored per row so a preprocessing change produces a
  distinguishable vector generation.

``get_encoder`` is the only construction entry point. The ``artifact``
backend is deliberately unimplemented: standalone local encoding requires
a *reviewed* offline runtime and pinned artifact (SPEC §28), and a model
loader that executes repository code or fetches at runtime is a remote-code
and supply-chain risk. Until that review lands, the honest answer is
``ENCODER_UNAVAILABLE`` — a visible degraded capability, not a silent
fallback to an unreviewed loader.

``encode_spans`` is the write path: encode first (network/inference work
happens before any SQL write), validate every returned vector through the
codec gate, then upsert rows keyed ``(span_id, encoder_id)`` and register
the encoder manifest if absent.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Optional, Protocol, runtime_checkable

from ..core.types import ErrorCode, VerbatimError, require_id
from .codec import Float32Codec

_MAX_TEXTS_PER_CALL = 256


@runtime_checkable
class Encoder(Protocol):
    """Vector producer contract (SPEC §28).

    Implementations return packed float32le blobs — never raw floats — so
    validation always passes through :class:`Float32Codec` on the exact
    bytes that will be persisted.
    """

    @property
    def encoder_id(self) -> str:
        """Stable identity ``backend:model:artifact_revision|'unpinned'``."""
        ...

    @property
    def dimensions(self) -> int:
        """Declared vector dimension.

        May raise :class:`VerbatimError` ``ENCODER_UNAVAILABLE`` when the
        backend discovers dimensions on first encode and has not encoded
        yet — an explicit unknown rather than a placeholder zero (SPEC §8).
        """
        ...

    @property
    def normalization(self) -> Optional[str]:
        """``'l2'`` when consumers must L2-normalize before cosine, ``'none'``
        when vectors arrive normalized, ``None`` when unrecorded."""
        ...

    def encode(self, texts: list[str]) -> list[bytes]:
        """Return one packed float32le blob per input text, same order.

        Raises :class:`VerbatimError` ``ENCODER_UNAVAILABLE`` when the
        backend cannot serve the request. Output is NOT trusted — callers
        must run :func:`Float32Codec.validate_blob` before persisting.
        """
        ...

    def available(self) -> bool:
        """Local availability check only; never raises, never mutates state."""
        ...

    def manifest(self) -> dict[str, Any]:
        """``encoder_manifests`` row data plus ``preprocessing_version``."""
        ...


def encoder_identity(backend: str, model: str, artifact_revision: Optional[str]) -> str:
    """Stable encoder identity.

    An unpinned revision is recorded as the literal ``'unpinned'`` — never
    elided — so an alias like ``latest`` can never masquerade as a reviewed
    pin in stored vector identity or benchmark reporting (SPEC §28).
    """
    return f"{backend}:{model}:{artifact_revision or 'unpinned'}"


# ---------------------------------------------------------------------------
# Transport permit seam (SPEC_V4 §12, F4-04)
# ---------------------------------------------------------------------------


class RemoteEncoder:
    """Base for encoders whose ``encode`` performs transport I/O.

    Remote dispatch is impossible-by-construction without a permit: the
    concrete ``encode`` MUST funnel through :meth:`_authorize_transport`
    immediately before its socket work. The check needs both halves — a
    broker attached at construction AND a per-call ``DispatchPermit`` —
    so an encoder built without a broker denies every dispatch rather than
    falling back to ungated transport.

    ``request_payload(texts)`` returns the exact body bytes the transport
    will send; ``payload_digest`` digests those bytes so the permit minted
    by the caller covers THIS request — not a paraphrase of it.
    """

    requires_transport_permit: bool = True
    _broker: Any = None

    @property
    def endpoint_id(self) -> str:
        """The allowlisted endpoint identity this encoder dispatches to."""
        raise NotImplementedError

    def endpoint_descriptor(self) -> Any:
        """``EndpointDescriptor`` for broker allowlist construction."""
        raise NotImplementedError

    def request_payload(self, texts: list[str]) -> bytes:
        """The exact request body ``encode`` will POST for ``texts``."""
        raise NotImplementedError

    def payload_digest(self, texts: list[str]) -> str:
        """``payload_digest(request_payload(texts))`` — mint-permit input."""
        from ..privacy.broker import payload_digest as _pd

        return _pd(self.request_payload(texts))

    def _authorize_transport(self, permit: Any, payload: bytes) -> None:
        """Consume ``permit`` for this exact request body.

        ``broker.dispatch`` re-verifies consent/suppression/epochs and the
        reservation inside its own transaction, then flips the permit
        open→dispatched atomically (one-use — replay is denied). No broker
        attached ⇒ remote dispatch is permanently disabled (fail closed).
        """
        from ..privacy.broker import payload_digest as _pd

        if self._broker is None:
            raise VerbatimError(
                ErrorCode.EGRESS_DENIED,
                "remote encoder has no transport broker — dispatch disabled "
                "(wire a TransportBroker and pass permit= to encode)",
                retryable=False,
            )
        self._broker.dispatch(
            permit,
            recipient=self.endpoint_id,
            payload_digest=_pd(payload),
        )


def encoder_requires_permit(encoder: Any) -> bool:
    """True when ``encoder.encode`` dispatches over a transport and must be
    called with ``permit=`` (or via ``TransportBroker.encode_permitted``)."""
    return bool(getattr(encoder, "requires_transport_permit", False))


def get_encoder(
    cfg: Any,
    http: Optional[Any] = None,
    secret_getter: Optional[Any] = None,
    broker: Optional[Any] = None,
) -> Optional[Encoder]:
    """Resolve the configured embedding backend to an ``Encoder`` or None.

    - ``none`` → ``None``; semantic retrieval degrades to lexical with an
      explicit capability indicator (SPEC §4).
    - ``hashing`` → :class:`HashingEncoder` — deterministic pure-stdlib
      lexical-subword encoder (feature hashing over word unigrams +
      char n-grams). Always available; the algorithm revision is the
      artifact pin.
    - ``ollama`` → :class:`OllamaEncoder` bound to a verified loopback
      endpoint (local-service only).
    - ``cloudflare`` → :class:`CloudflareEncoder` for remote_assisted; the
      API token resolves through ``secret_getter`` at encode time.

    - ``artifact`` → :class:`ArtifactEncoder` — a fail-closed contract
      implementation. Construction succeeds so the store opens and the
      capability report shows ``available()=False``; ``encode`` raises
      ``ENCODER_UNAVAILABLE`` until a reviewed, hash-pinned runtime
      ships (``artifact.py`` documents the trust contract). Raising at
      construction would kill ``open_store`` instead of degrading the
      capability, which is the wrong failure mode (SPEC_V2 §8).

    ``broker`` attaches a :class:`~verbatim.privacy.broker.TransportBroker`
    to transport-bound encoders (ollama, cloudflare). Without one those
    encoders are fail-closed: ``encode`` raises ``EGRESS_DENIED`` before
    any socket work (F4-04 — an unwired call site cannot dispatch).

    ``http`` is an optional transport injected for tests; production passes
    nothing and gets the real transport.
    """
    backend = cfg.embedding.backend
    if backend == "none":
        return None
    if backend == "hashing":
        from .hashing import HashingEncoder

        return HashingEncoder(cfg.embedding)
    if backend == "ollama":
        from .ollama import OllamaEncoder

        return OllamaEncoder(cfg.embedding, http=http, broker=broker)
    if backend == "cloudflare":
        from .cloudflare import CloudflareEncoder

        if secret_getter is None:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "cloudflare backend requires a host secret_getter",
            )
        return CloudflareEncoder(
            cfg.embedding,
            account_id=cfg.embedding.account_id or "",
            secret_getter=secret_getter,
            http=http,
            broker=broker,
        )
    if backend == "artifact":
        from .artifact import ArtifactEncoder

        return ArtifactEncoder(cfg.embedding)
    raise VerbatimError(
        ErrorCode.CONFIG_INVALID, f"embedding.backend {backend!r} unsupported"
    )


class SpanText(NamedTuple):
    """One already-authorized span quotation to embed.

    The caller owns scope authorization — this type deliberately carries no
    scope fields so the embedding path cannot be mistaken for an
    authorization boundary (SPEC §28: encode *contextual* quotations the
    caller already cleared).
    """

    span_id: str
    text: str


def _row_parts(row: Any) -> tuple[str, str]:
    """Accept ``SpanText`` or a mapping with ``span_id``/``text`` keys.

    Duck-typing keeps the call site adaptable to whatever row shape the
    storage layer hands over, while still validating the span identifier
    before it touches SQL.
    """
    try:
        if isinstance(row, Mapping):
            span_id, text = row["span_id"], row["text"]
        else:
            span_id, text = row.span_id, row.text
    except (KeyError, AttributeError, TypeError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, "span row needs span_id and text"
        ) from exc
    require_id(span_id, "span_id")
    if not isinstance(text, str):
        raise VerbatimError(ErrorCode.VALIDATION, "span text must be str")
    return span_id, text


def _manifest_row(encoder: Encoder) -> dict[str, Any]:
    m = encoder.manifest()
    require_id(encoder.encoder_id, "encoder_id")
    artifact_revision = m.get("artifact_revision") or "unpinned"
    dims = m.get("dimensions")
    if not isinstance(dims, int) or dims < 1:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID, f"encoder manifest dimensions {dims!r} invalid"
        )
    return {
        "encoder_id": encoder.encoder_id,
        "artifact_revision": artifact_revision,
        "dimensions": dims,
        "normalization": m.get("normalization"),
        "license_id": m.get("license_id"),
        "preprocessing_version": m.get("preprocessing_version") or "v1",
        "manifest_json": m.get("manifest_json") or {},
    }


def _register_manifest(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert the manifest only when absent.

    ``DO NOTHING`` on conflict is deliberate: an existing manifest may
    carry a reviewed ``license_id`` or corrected fields from operator
    review, and silently overwriting it would erase that audit trail.
    Compatibility drift is detected by comparing rows, not by mutation.
    """
    conn.execute(
        """
        INSERT INTO encoder_manifests
            (encoder_id, artifact_revision, dimensions, normalization,
             license_id, manifest_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(encoder_id) DO NOTHING
        """,
        (
            row["encoder_id"],
            row["artifact_revision"],
            row["dimensions"],
            row["normalization"],
            row["license_id"],
            json.dumps(row["manifest_json"], sort_keys=True, allow_nan=False),
        ),
    )


def encode_spans(
    store: Any,
    conn: sqlite3.Connection,
    encoder: Encoder,
    span_rows: Sequence[Any],
) -> int:
    """Embed authorized span texts and upsert embedding rows.

    ``span_rows`` are :class:`SpanText` items (or mappings with
    ``span_id``/``text``) whose text the CALLER has already authorized for
    the target scope — this function performs no scope checks and must not
    be reachable with unauthorized text.

    Ordering contract (SPEC §21): ``encoder.encode`` runs FIRST, before any
    SQL on ``conn`` — model inference must never hold a database write
    transaction. Callers should open ``conn``'s transaction immediately
    before calling so the encode does not sit inside a long-lived tx.

    Returns the number of embedding rows upserted. ``source_hmac`` is the
    store's profile-keyed HMAC of the exact UTF-8 bytes embedded, so stale
    rows are detectable when a span's authorized text changes (SPEC §28:
    embedding jobs key on source/span hash, encoder identity, and
    preprocessing revision).
    """
    pairs = [_row_parts(r) for r in span_rows]
    if not pairs:
        return 0
    if len(pairs) > _MAX_TEXTS_PER_CALL:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"embedding batch {len(pairs)} exceeds {_MAX_TEXTS_PER_CALL}",
        )

    # Phase 1: inference only — no SQL has run yet inside this call.
    texts = [t for _, t in pairs]
    blobs = encoder.encode(texts)
    if len(blobs) != len(pairs):
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"encoder returned {len(blobs)} vectors for {len(pairs)} texts",
        )

    # Phase 2: validate every blob against the encoder's declared dimension
    # BEFORE any write — a partially valid batch must not persist partially.
    manifest = _manifest_row(encoder)
    dims = manifest["dimensions"]
    for blob in blobs:
        Float32Codec.validate_blob(blob, dims)

    # Phase 3: writes only — upsert rows + register manifest if absent.
    _register_manifest(conn, manifest)
    upserted = 0
    for (span_id, text), blob in zip(pairs, blobs):
        conn.execute(
            """
            INSERT INTO embeddings
                (span_id, encoder_id, preprocessing_version, dimensions,
                 dtype, vector, source_hmac)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(span_id, encoder_id) DO UPDATE SET
                preprocessing_version = excluded.preprocessing_version,
                dimensions = excluded.dimensions,
                dtype = excluded.dtype,
                vector = excluded.vector,
                source_hmac = excluded.source_hmac
            """,
            (
                span_id,
                encoder.encoder_id,
                manifest["preprocessing_version"],
                dims,
                Float32Codec.dtype,
                blob,
                store.hmac(text.encode("utf-8")),
            ),
        )
        upserted += 1
    return upserted


__all__ = [
    "Encoder",
    "RemoteEncoder",
    "SpanText",
    "encoder_identity",
    "encoder_requires_permit",
    "get_encoder",
    "encode_spans",
]
