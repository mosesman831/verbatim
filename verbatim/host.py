"""Host adapter protocol — the seam that makes Hermes just one plugin.

The engine core never imports Hermes. Any host (Hermes adapter, standalone
CLI, a future agent) supplies identity, time, secrets, and thread-spawning
through this interface (SPEC §6, §7, §36).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .core.types import Scope


@runtime_checkable
class HostAdapter(Protocol):
    """What a host environment must provide to run the Verbatim engine."""

    def host_name(self) -> str:
        """Stable host identifier, e.g. 'hermes', 'cli'."""

    def profile_id(self) -> str:
        """Canonical storage partition for this host session."""

    def default_scope(self) -> Scope:
        """The scope evidence captured in this session belongs to."""

    def now_us(self) -> int:
        """Current wall-clock UTC in microseconds."""

    def secret(self, name: str) -> Optional[str]:
        """Scoped secret lookup (e.g. TYPESAFE_API_KEY). None when absent.

        Implementations MUST NOT fall back to a different scope's credentials
        when the scoped accessor rejects the call (SPEC §41).
        """

    def spawn_thread(self, fn: Callable[[], Any], name: str) -> threading.Thread:
        """Context-bound background thread. Hermes uses spawn_context_thread."""

    def log(self, level: str, event: str, **fields: Any) -> None:
        """Structured local logging; implementations omit private payloads."""


class LocalHost:
    """Standalone host for `python -m verbatim` and library use.

    Single-operator context: the local user is the principal; secrets resolve
    from a caller-supplied mapping or the process environment when explicitly
    allowed by the caller (never silently for remote egress).
    """

    def __init__(
        self,
        profile_id: str = "local",
        principal_id: Optional[str] = "local-owner",
        conversation_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        secrets: Optional[dict[str, str]] = None,
        allow_env_secrets: bool = False,
    ) -> None:
        self._profile_id = profile_id
        self._scope = Scope(
            profile_id=profile_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        self._secrets = dict(secrets or {})
        self._allow_env = allow_env_secrets

    def host_name(self) -> str:
        return "cli"

    def profile_id(self) -> str:
        return self._profile_id

    def default_scope(self) -> Scope:
        return self._scope

    def now_us(self) -> int:
        from .core.time import now_us

        return now_us()

    def secret(self, name: str) -> Optional[str]:
        if name in self._secrets:
            return self._secrets[name]
        if self._allow_env:
            return os.environ.get(name)
        return None

    def spawn_thread(self, fn: Callable[[], Any], name: str) -> threading.Thread:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        return t

    def log(self, level: str, event: str, **fields: Any) -> None:
        import sys

        print(f"[{level}] {event} {fields if fields else ''}", file=sys.stderr)
