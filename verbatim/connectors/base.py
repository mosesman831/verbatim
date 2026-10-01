"""Connector framework types (SPEC_V4 §48, V4-48.*).

A connector is an *importer*: it enumerates remote/host artifacts and the
pull engine drives each item through the real write channel
(``evidence.envelopes.ingest_envelope`` — screening, consent, dedup,
receipts included). Connectors are ingestion clients of the same
authority services, never trusted exceptions (§48 intro).

Contracts fixed here:

- ``ConnectorDescriptor`` — the connector's self-declared identity,
  formats, and honest capability flags. ``remote=True`` marks a connector
  that would need network egress; this build ships no remote connectors
  (declared-unavailable, never silently absent).
- ``RemoteItem`` — one enumerated artifact. ``external_id`` is the stable
  remote identity that binds the item to one durable ``sources`` row
  (re-imports revise it); ``cursor`` is the connector's opaque resume
  token carried per item so the ledger can checkpoint at item
  granularity (V4-48.05).
- ``PullReport`` — the honest result envelope shared by real pulls and
  dry-runs. Counts are exact; ``items`` is a bounded detail list (the
  durable batch row + source metadata carry the full provenance).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Protocol, runtime_checkable

from ..core.types import ErrorCode, VerbatimError, require_id

# Item classes a connector may declare (mirrors the §48 import-class table:
# documents carry verbatim bytes; episodes/entities/claims without originals
# are imported ASSERTIONS — V4-48.02).
ITEM_CLASSES = frozenset(
    {"document", "episode", "entity", "claim", "note", "event"}
)

# Actions a pull/dry-run assigns to each enumerated item.
ITEM_ACTIONS = frozenset(
    {
        "inserted",        # new source row created
        "revised",         # same external identity, new revision minted
        "duplicate",       # same identity + same bytes — dedup hit, no write
        "rejected",        # item refused (malformed, oversize, unreadable…)
        "would_insert",    # dry-run only
        "would_revise",    # dry-run only
        "would_reject",    # dry-run only
    }
)

#: Bound on per-item detail retained in reports/manifests — a bulk import
#: never materializes unbounded item rows into a single receipt.
ITEM_DETAIL_LIMIT = 512


class ItemClass(str, enum.Enum):
    DOCUMENT = "document"
    EPISODE = "episode"
    ENTITY = "entity"
    CLAIM = "claim"
    NOTE = "note"
    EVENT = "event"


@dataclass(frozen=True)
class ConnectorDescriptor:
    """A connector's self-description (V4-48.01 per-provider coverage).

    ``declared_not_verified`` lists claims the connector makes that this
    build has not verified — e.g. fidelity to an external provider's
    private schema. Honest by construction: unverified claims are named,
    never implied by a successful import count (V4-48.10).
    """

    connector_id: str
    version: str
    display_name: str
    remote: bool
    formats: tuple[str, ...]
    item_classes: tuple[str, ...] = ("document",)
    capabilities: dict[str, Any] = field(default_factory=dict)
    declared_not_verified: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.connector_id, "connector_id")
        if not isinstance(self.version, str) or not self.version:
            raise VerbatimError(ErrorCode.VALIDATION, "connector version required")
        object.__setattr__(self, "formats", tuple(self.formats))
        object.__setattr__(self, "item_classes", tuple(self.item_classes))
        object.__setattr__(
            self, "declared_not_verified", tuple(self.declared_not_verified)
        )
        bad = set(self.item_classes) - ITEM_CLASSES
        if bad:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown item classes {sorted(bad)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "version": self.version,
            "display_name": self.display_name,
            "remote": self.remote,
            "formats": list(self.formats),
            "item_classes": list(self.item_classes),
            "capabilities": dict(self.capabilities),
            "declared_not_verified": list(self.declared_not_verified),
        }


@dataclass(frozen=True)
class RemoteItem:
    """One enumerated artifact from a connector ``scan``.

    ``content`` is the raw item bytes BEFORE acceptance — the pull engine
    applies the real write-channel gates (UTF-8 validity, size bound,
    screening) per item; a malformed item is recorded ``rejected`` rather
    than aborting the pull (a rejection is a report outcome, never a
    silent skip — V4-48.05).

    ``imported_assertion`` marks extracted facts/summaries whose original
    bytes are absent (V4-48.02): they persist with ``imported`` trust and
    ``original_present=False`` metadata — never reconstructed as
    byte-exact evidence.
    """

    external_id: str
    cursor: str
    content: bytes
    media_type: str = "text/plain"
    item_class: str = "document"
    event_us: Optional[int] = None
    author_id: Optional[str] = None
    imported_assertion: bool = False
    remote_revision: Optional[str] = None
    missing_provenance: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    sensitivity_hints: tuple[str, ...] = ()
    # Connector-level refusal: the item occupies a cursor position but is
    # not importable (e.g. a symlink escaping the import root). The pull
    # engine counts it rejected and records the reason — the cursor still
    # advances past it because the rejection is durable in the manifest.
    reject_reason: Optional[str] = None
    # Connector-reported format losses for this item (unsupported fields
    # the export carried that this import cannot preserve — V4-48.10).
    losses: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # external_id is a remote identity token, not an engine id —
        # file paths and provider keys carry separators, so the only
        # contract is non-empty string (it lands in sources.external_id
        # TEXT, keyed by exact equality).
        if not isinstance(self.external_id, str) or not self.external_id:
            raise VerbatimError(
                ErrorCode.VALIDATION, "external_id must be a non-empty string"
            )
        if not isinstance(self.cursor, str) or not self.cursor:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "item cursor must be a non-empty string — the cursor "
                "ledger cannot checkpoint an empty token",
            )
        if not isinstance(self.content, (bytes, bytearray)):
            raise VerbatimError(ErrorCode.VALIDATION, "item content must be bytes")
        object.__setattr__(self, "content", bytes(self.content))
        if self.item_class not in ITEM_CLASSES:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown item class {self.item_class!r}"
            )
        if self.event_us is not None and (
            isinstance(self.event_us, bool)
            or not isinstance(self.event_us, int)
            or self.event_us < 0
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION, "item event_us must be an int >= 0"
            )
        for name in ("author_id", "remote_revision", "reject_reason"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, str):
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"item {name} must be a string"
                )
        object.__setattr__(
            self, "missing_provenance", tuple(self.missing_provenance)
        )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(
            self, "sensitivity_hints", tuple(self.sensitivity_hints)
        )
        object.__setattr__(self, "losses", tuple(self.losses))


@runtime_checkable
class Connector(Protocol):
    """Importer contract (§48).

    ``scan`` enumerates items in a stable cursor order and must be
    restartable: given ``cursor`` (the last committed item token) it
    yields strictly-later items. Items are plain data — the connector
    never writes to the store; the pull engine owns every write.
    """

    def descriptor(self) -> ConnectorDescriptor:
        ...

    def validate_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the caller's source descriptor into a JSON-safe dict.

        Raises ``VerbatimError(VALIDATION)`` on a malformed or unreachable
        source; ``VerbatimError(CAPABILITY_UNAVAILABLE)`` when the source
        needs transport this build does not have (remote endpoints).
        """
        ...

    def scan(
        self,
        source: Mapping[str, Any],
        *,
        cursor: Optional[str],
        limit: Optional[int] = None,
    ) -> Iterator[RemoteItem]:
        """Yield items strictly after ``cursor`` in stable order."""
        ...


@dataclass(frozen=True)
class PullReport:
    """The pull's honest result — also the dry-run shape (V4-48.03).

    Real pulls persist the same identity set in ``ingest_batches`` +
    ``connector_cursors`` + per-item ``source_revisions.metadata_json``;
    dry-runs write nothing and return this report alone.
    """

    pull_id: str
    connector_id: str
    connector_version: str
    scope_id: str
    principal_id: str
    dry_run: bool
    state: str                     # complete | partial | dry_run | failed
    cursor_before: Optional[str]
    cursor_after: Optional[str]
    scanned: int = 0
    inserted: int = 0
    revised: int = 0
    duplicates: int = 0
    rejected: int = 0
    conflicts: int = 0             # same external_id under a foreign origin
    quarantined: int = 0
    bytes_seen: int = 0
    bytes_accepted: int = 0
    pages: int = 0
    missing_provenance: int = 0
    sensitivity: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    scope_mappings: dict[str, str] = field(default_factory=dict)
    items: tuple[dict[str, Any], ...] = ()
    batch_id: Optional[str] = None
    job_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pull_id": self.pull_id,
            "batch_id": self.batch_id,
            "connector_id": self.connector_id,
            "connector_version": self.connector_version,
            "scope_id": self.scope_id,
            "principal_id": self.principal_id,
            "dry_run": self.dry_run,
            "state": self.state,
            "cursor_before": self.cursor_before,
            "cursor_after": self.cursor_after,
            "scanned": self.scanned,
            "inserted": self.inserted,
            "revised": self.revised,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "conflicts": self.conflicts,
            "quarantined": self.quarantined,
            "bytes_seen": self.bytes_seen,
            "bytes_accepted": self.bytes_accepted,
            "pages": self.pages,
            "missing_provenance": self.missing_provenance,
            "sensitivity": dict(self.sensitivity),
            "losses": dict(self.losses),
            "scope_mappings": dict(self.scope_mappings),
            "items": [dict(i) for i in self.items],
            "job_id": self.job_id,
            "error": self.error,
        }


__all__ = [
    "Connector",
    "ConnectorDescriptor",
    "ITEM_ACTIONS",
    "ITEM_CLASSES",
    "ITEM_DETAIL_LIMIT",
    "ItemClass",
    "PullReport",
    "RemoteItem",
]
