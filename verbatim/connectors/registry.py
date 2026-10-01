"""Connector registry (SPEC_V4 §48).

Two local reference connectors ship in this build — ``local_files`` and
``holographic`` — plus the ``RemoteConnectorStub`` machinery for
declaring a networked provider UNAVAILABLE without implementing it
(V4-48.06: no allowlist machinery, no credential store, no redirect
handling exists yet, so remote transport is refused honestly rather than
faked). ``get_connector`` on an unknown id fails
``NOT_FOUND_OR_UNAUTHORIZED`` — never an implicit default importer.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError, require_id
from .base import Connector, ConnectorDescriptor, RemoteItem
from .local_files import LocalFilesConnector
from .holographic import HolographicConnector


class RemoteConnectorStub:
    """Declared-unavailable remote provider placeholder.

    Exists so a provider name can be listed/described honestly —
    ``descriptor().remote=True`` — while every pull refuses with
    ``CAPABILITY_UNAVAILABLE``. V4-48.06 requires allowlists, redirect
    limits, SSRF protection, content bounds, and separately-scoped
    credentials before any remote fetcher may run; none of that exists
    in this build, so the stub is the honest declaration, not a tease.
    """

    def __init__(self, connector_id: str, display_name: str) -> None:
        self._id = connector_id
        self._name = display_name

    def descriptor(self) -> ConnectorDescriptor:
        return ConnectorDescriptor(
            connector_id=self._id,
            version="0",
            display_name=self._name,
            remote=True,
            formats=(),
            capabilities={
                "remote_fetch": False,
                "unavailable_reason": (
                    "no network egress, credential, or SSRF machinery "
                    "in this build"
                ),
            },
            declared_not_verified=("provider_format_fidelity",),
        )

    def validate_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"remote connector {self._id!r} is declared-unavailable in "
            "this build (no egress/credential machinery)",
        )

    def scan(
        self,
        source: Mapping[str, Any],
        *,
        cursor: Optional[str],
        limit: Optional[int] = None,
    ) -> Iterator[RemoteItem]:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"remote connector {self._id!r} cannot scan — unavailable",
        )
        yield  # pragma: no cover - keep this a generator


_REGISTRY: dict[str, Connector] = {}


def register(connector: Connector) -> ConnectorDescriptor:
    """Register a connector instance; returns its validated descriptor."""
    if not isinstance(connector, Connector):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "connector must satisfy the Connector protocol "
            "(descriptor/validate_source/scan)",
        )
    desc = connector.descriptor()
    _REGISTRY[desc.connector_id] = connector
    return desc


def get_connector(connector_id: str) -> Connector:
    require_id(connector_id, "connector_id")
    conn = _REGISTRY.get(connector_id)
    if conn is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            f"unknown connector {connector_id!r}",
        )
    return conn


def list_connectors() -> list[ConnectorDescriptor]:
    return [c.descriptor() for c in _REGISTRY.values()]


register(LocalFilesConnector())
register(HolographicConnector())

__all__ = [
    "RemoteConnectorStub",
    "get_connector",
    "list_connectors",
    "register",
]
