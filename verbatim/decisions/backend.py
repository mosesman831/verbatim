"""Decision backend contract, registry, and test fake (SPEC §23).

A backend is an *advisory judge over fixed label sets* — never a generator.
Results are typed judgments; the deterministic policy reducer (outside this
package) decides what any output is allowed to do. Backends may abstain but
may not enlarge their candidate set, context budget, or spending cap.

``available()`` is a local preflight only: it inspects configuration,
credentials presence, and consent state — it must never spend tokens or send
private content over the network.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..core.types import (
    DecisionRequest,
    DecisionResult,
    ErrorCode,
    TaskKind,
    VerbatimError,
    json_dumps,
)

ALL_TASKS: tuple[str, ...] = tuple(t.value for t in TaskKind)


@runtime_checkable
class DecisionBackend(Protocol):
    """Interface every advisory decision backend implements (SPEC §23)."""

    name: str

    def capabilities(self) -> dict[str, Any]:
        """Static metadata: supported tasks, calibration status, revisions."""
        ...

    def available(self) -> bool:
        """Local-only readiness check — never performs network I/O."""
        ...

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        """Bounded evaluation over the supplied authorized snapshot."""
        ...

    def close(self) -> None:
        """Release any held resources; must be safe to call repeatedly."""
        ...


# Factory signature: (cfg, scope, deps) -> DecisionBackend. ``deps`` is an
# opaque mapping carrying host-provided wiring — keys used by built-ins:
#   'store'         durable store (tx/read context managers, hmac)
#   'egress'        EgressGate instance (jev only)
#   'secret_getter' callable(name) -> Optional[str] (jev only)
#   'http'          transport override for tests (jev only)
BackendFactory = Callable[..., DecisionBackend]


class BackendRegistry:
    """Name → factory map; construction is explicit, never ambient."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        if not name or not callable(factory):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "backend registration needs name+factory")
        self._factories[name] = factory

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(
        self,
        name: str,
        cfg: Any,
        scope: Any,
        deps: Optional[dict[str, Any]] = None,
    ) -> DecisionBackend:
        factory = self._factories.get(name)
        if factory is None:
            # CONFIG_INVALID, not a lookup error: selecting an unknown backend
            # must fail activation exactly like a bad config key (SPEC §43).
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"unknown decision backend {name!r}"
            )
        return factory(cfg=cfg, scope=scope, deps=dict(deps or {}))


def default_registry() -> BackendRegistry:
    """Registry with the two built-in backends wired lazily.

    Imports happen here so that importing the decisions package (or the
    rules backend alone) never requires the jev adapter's dependencies.
    """
    reg = BackendRegistry()

    def _rules(cfg: Any, scope: Any, deps: dict[str, Any]) -> DecisionBackend:
        from .rules import RulesBackend

        return RulesBackend()

    def _jev(cfg: Any, scope: Any, deps: dict[str, Any]) -> DecisionBackend:
        from .jev import JevBackend

        try:
            egress = deps["egress"]
            secret_getter = deps["secret_getter"]
        except KeyError as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"jev backend requires deps[{exc.args[0]!r}]",
            ) from exc
        return JevBackend(
            cfg,
            scope,
            egress_gate=egress,
            secret_getter=secret_getter,
            http=deps.get("http"),
        )

    reg.register("rules", _rules)
    reg.register("jev", _jev)
    return reg


def request_fingerprint(request: DecisionRequest) -> str:
    """Deterministic hash of the decision-relevant request content.

    Used by the fake backend for fixture lookup and suitable as the request
    dedup/cache key seed elsewhere. Canonical JSON makes keying stable across
    processes and dict orderings.
    """
    canon = json_dumps(
        {
            "task": request.task.value,
            "state": request.state,
            "allowed_labels": list(request.allowed_labels),
            "rubric_version": request.rubric_version,
        }
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class FakeBackend:
    """Deterministic fixture backend for tests (SPEC §23).

    Returns canned results keyed by :func:`request_fingerprint`. A missing
    fixture yields a deterministic abstention — never a fabricated answer.
    Records every request it saw so tests can assert dispatch behavior.
    Contains no network code paths at all.
    """

    name = "fake"
    MODEL_REVISION = "fake-0"

    def __init__(
        self,
        fixtures: Optional[dict[str, Any]] = None,
        *,
        available: bool = True,
    ) -> None:
        self._fixtures = dict(fixtures or {})
        self._available = available
        self.calls: list[DecisionRequest] = []
        self.closed = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "tasks": ALL_TASKS,
            "calibrated": False,
            "backend": self.name,
            "model_revision": self.MODEL_REVISION,
            "rubric_version": "fake-1",
        }

    def available(self) -> bool:
        return self._available

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        self.calls.append(request)
        key = request_fingerprint(request)
        fixture = self._fixtures.get(key)
        if isinstance(fixture, DecisionResult):
            return fixture
        if isinstance(fixture, dict):
            return DecisionResult(
                task=request.task,
                backend=self.name,
                model_revision=self.MODEL_REVISION,
                rubric_version=request.rubric_version,
                outcome=dict(fixture),
            )
        return DecisionResult(
            task=request.task,
            backend=self.name,
            model_revision=self.MODEL_REVISION,
            rubric_version=request.rubric_version,
            outcome={"fixture": "absent"},
            abstained=True,
            reason="no fixture",
        )

    def close(self) -> None:
        self.closed = True
