"""Decision backends: advisory typed judges over fixed label sets (SPEC §23-25)."""

from .backend import (
    BackendRegistry,
    DecisionBackend,
    FakeBackend,
    default_registry,
    request_fingerprint,
)

__all__ = [
    "BackendRegistry",
    "DecisionBackend",
    "FakeBackend",
    "default_registry",
    "request_fingerprint",
]
