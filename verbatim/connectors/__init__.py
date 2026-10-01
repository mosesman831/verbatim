"""verbatim.connectors — importer framework (SPEC_V4 §48, V4-48.*).

Connectors are ingestion *clients*: they enumerate remote/host
artifacts, and the pull engine drives each item through the real write
channel (``evidence.envelopes.ingest_envelope`` — screening, consent,
dedup, receipts). This build ships two local reference connectors
(``local_files``, ``holographic``); remote providers are
declared-unavailable stubs — no network, no credentials, no downloads.

Public surface::

    from verbatim.connectors import ConnectorService, get_connector
    report = ConnectorService(store, cfg).pull(
        "local_files", {"dir": "/path"},
        scope_id=sid, principal_id="alice",
        authorization_id=aid,
    )
"""

from .base import (
    Connector,
    ConnectorDescriptor,
    ItemClass,
    PullReport,
    RemoteItem,
)
from .registry import (
    RemoteConnectorStub,
    get_connector,
    list_connectors,
    register,
)
from .service import ConnectorService

__all__ = [
    "Connector",
    "ConnectorDescriptor",
    "ConnectorService",
    "ItemClass",
    "PullReport",
    "RemoteConnectorStub",
    "RemoteItem",
    "get_connector",
    "list_connectors",
    "register",
]
