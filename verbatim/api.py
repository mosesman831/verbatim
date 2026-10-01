"""Public engine API — the host-agnostic facade (SPEC §36, V3-06.01).

Every entry point takes an explicit scope or a validated scoped handle; no
implicit global profile. Hosts (Hermes adapter, CLI, other agents) bind a
HostAdapter and call this — the core never sees Hermes types.

Facade-plus-siblings (V3-06.01): ``Engine`` composes topic mixins —
``api_ingest`` (evidence write path), ``api_recall`` (bounded recall),
``api_review`` (transitions/proposals), ``api_governance`` (caller
binding + grants), ``api_privacy`` (purge/export/handoff), and
``api_experience`` (declared experience owner). Siblings own their topic
and never import this facade; the facade owns construction, capability
status, lifecycle, and ``open_store``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .api_experience import ExperienceMixin
from .api_governance import GovernanceMixin
from .api_ingest import IngestMixin
from .api_privacy import PrivacyMixin
from .api_recall import RecallMixin
from .api_review import ReviewMixin
from .config import VerbatimConfig
from .core.identity import scope_key
from .core.time import now_us
from .core.types import (
    CallerContext,
    ErrorCode,
    GrantKind,
    VerbatimError,
)
from .host import HostAdapter
from .ingest import Ingester
from .storage.store import Store


def _retrieval_cache_status(store: Any, cfg: Any) -> dict[str, Any]:
    """``status()`` block for the V4-33 final-pack cache — real counters
    or an honest "off/unavailable", never a fabricated hit rate."""
    try:
        from .retrieval import cache as _rcache

        stats = _rcache.stats_for(store, cfg)
    except Exception:
        return {
            "state": "unavailable",
            "degraded_reason": "verbatim.retrieval.cache not importable",
            "details": {},
        }
    state = stats.pop("state", "implemented")
    reason = stats.pop("degraded_reason", None)
    stats.pop("enabled", None)
    return {
        "state": state,
        "degraded_reason": reason,
        "details": stats,
    }


class Engine(
    IngestMixin,
    RecallMixin,
    ReviewMixin,
    GovernanceMixin,
    PrivacyMixin,
    ExperienceMixin,
):
    """An opened store bound to a host context.

    All methods enforce scope authorization before touching data. Unknown IDs
    and access-denied are deliberately indistinguishable (SPEC §9).
    """

    def __init__(
        self,
        store: Store,
        cfg: VerbatimConfig,
        host: HostAdapter,
        judge: Any = None,
        encoder: Any = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.host = host
        self.encoder = encoder
        if encoder is not None:
            # Query-side encoder identity travels with the store so the
            # semantic lane's ``_query_encoder`` probe resolves the same
            # encoder that wrote the embedding rows (SPEC §28: query and
            # document vectors must share encoder identity).
            store.encoder = encoder
        self.transport_broker = getattr(store, "transport_broker", None)
        self._ingester = Ingester(
            store,
            cfg,
            judge,
            encoder=encoder,
            transport_broker=self.transport_broker,
        )
        self._closed = False

    def status(self, *, caller: Optional[CallerContext] = None) -> dict[str, Any]:
        """Mode, integrity, and per-capability truthfulness (V2-05, §43).

        ``caller`` is optional: when supplied it must hold READ_EVIDENCE —
        diagnostic output stays inside the caller's authority.
        """
        self._require_open()
        if caller is not None:
            c = self._resolve_caller(caller, self.host.default_scope())
            self._require_grant(c, GrantKind.READ_EVIDENCE)
        judge = self._ingester.judge
        encoder = self.encoder
        # States follow the CapabilityRung ladder (V4-50.01): a capability
        # reports its highest *observed* rung plus a degraded_reason —
        # never a bare "ok" for something the build cannot do, and never
        # "healthy" merely because a flag is set or a module imports
        # (V4-50.02). The provider's own ``available()`` probe decides
        # encoder health; object existence alone caps at "configured".
        if self.store.fts_enabled:
            fts = ("healthy", None)
        else:
            fts = ("implemented", "sqlite build lacks FTS5")
        if judge is not None:
            jud = ("healthy", None)
        elif self.cfg.judge.backend == "rules":
            jud = ("configured", "rules judge not instantiated for this engine")
        else:
            jud = ("configured", f"judge backend {self.cfg.judge.backend!r} not instantiated")
        encoder_provider = None
        if encoder is not None:
            eid = getattr(encoder, "encoder_id", None)
            if callable(eid):
                try:
                    eid = eid()
                except Exception:
                    eid = None
            encoder_provider = eid or type(encoder).__name__
            probe = getattr(encoder, "available", None)
            if not callable(probe):
                enc = (
                    "configured",
                    "bound encoder exposes no availability probe",
                )
            else:
                try:
                    enc_ok = bool(probe())
                except Exception:
                    enc_ok = False
                enc = (
                    ("healthy", None)
                    if enc_ok
                    else (
                        "unavailable",
                        f"bound encoder {encoder_provider!r} reports "
                        "unavailable",
                    )
                )
        elif self.cfg.embedding.backend == "none":
            enc = ("implemented", "embedding backend disabled (backend='none')")
        else:
            enc = (
                "configured",
                f"embedding backend {self.cfg.embedding.backend!r} configured "
                "but no usable encoder provider is bound",
            )
        from .storage.repos import has_table
        from .storage.repos_v2 import GrantsRepo

        with self.store.read() as conn:
            authz_rev = GrantsRepo(self.store).authz_revision(
                conn, scope_key(self.host.default_scope())
            )
            embedding_rows = (
                conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                if has_table(conn, "embeddings")
                else 0
            )
        return {
            "mode": self.cfg.mode.value,
            "capture_enabled": self.cfg.capture.enabled,
            "projection_generation": self.store.projection_generation(),
            "policy_epoch": self.store.policy_epoch(),
            "authz_revision": authz_rev,
            "integrity": self.store.check_integrity(),
            "capabilities": {
                "lexical_search": {
                    "state": fts[0],
                    "degraded_reason": fts[1],
                    "details": {"backend": "sqlite-fts5", "fts5": self.store.fts_enabled},
                },
                "judge": {
                    "state": jud[0],
                    "degraded_reason": jud[1],
                    "details": {
                        "backend": self.cfg.judge.backend,
                        "provider": type(judge).__name__ if judge else None,
                    },
                },
                "encoder": {
                    "state": enc[0],
                    "degraded_reason": enc[1],
                    "details": {
                        "backend": self.cfg.embedding.backend,
                        "provider": encoder_provider,
                    },
                },
                "semantic_recall": {
                    "state": enc[0],
                    "degraded_reason": enc[1],
                    "details": {
                        "rerank": "rrf",
                        "requires": "encoder",
                        "embedding_rows": embedding_rows,
                    },
                },
                # V4-33.07: honest cache state — real counters, and an
                # explicit "off" report rather than silence (the cache is
                # a performance layer, never an authority).
                "retrieval_cache": _retrieval_cache_status(
                    self.store, self.cfg
                ),
            },
        }

    # ------------------------------------------------------------------
    # durable readiness (SPEC_V4 §14)
    # ------------------------------------------------------------------

    def _resolve_receipt_id(
        self,
        receipt_id: Optional[str],
        source_id: Optional[str],
        revision: int,
    ) -> str:
        """The receipt identity a caller can hold: the ``cr_*``/``rc_*``
        id returned at capture, or ``(source_id, revision)`` resolved to
        the deterministic v2 ingest receipt."""
        from .readiness import ingest_receipt_id

        if receipt_id is not None:
            return receipt_id
        if source_id is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "wait_ready needs receipt_id or (source_id, revision)",
            )
        return ingest_receipt_id(source_id, int(revision))

    def wait_ready(
        self,
        receipt_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
        revision: int = 1,
        capabilities: Any = None,
        timeout_s: Optional[float] = 30.0,
        scope: Any = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Wait on one receipt's durable obligation DAG (V4-14.03).

        ``capabilities`` selects the rungs to await (default: every
        recorded capability on the receipt). The answer is the receipt's
        own snapshot — ``ready``/``complete``/``pending``/``failed``/
        ``deferred`` per capability — never inferred from the event
        journal or unrelated work (C26/C88). A deadline that expires
        returns the honest pending snapshot rather than an error or a
        fabricated ready.
        """
        self._require_open()
        scope = scope if scope is not None else self.host.default_scope()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.READ_EVIDENCE)
        rid = self._resolve_receipt_id(receipt_id, source_id, revision)
        deadline = (
            now_us() + int(timeout_s * 1_000_000)
            if timeout_s is not None
            else None
        )
        return self._ingester.readiness_engine().wait_ready(
            rid,
            capabilities,
            deadline_us=deadline,
            scope_id=scope_key(scope),
        )

    def receipt_state(
        self,
        receipt_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
        revision: int = 1,
        scope: Any = None,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """One receipt's per-capability durable snapshot (V4-14.02)."""
        self._require_open()
        scope = scope if scope is not None else self.host.default_scope()
        c = self._resolve_caller(caller, scope)
        self._require_grant(c, GrantKind.READ_EVIDENCE)
        rid = self._resolve_receipt_id(receipt_id, source_id, revision)
        return self._ingester.readiness_engine().receipt_state(
            rid, scope_id=scope_key(scope)
        )

    def drain_report(
        self,
        limit: int = 64,
        *,
        caller: Optional[CallerContext] = None,
    ) -> dict[str, Any]:
        """Drain due jobs and report the honest per-outcome breakdown —
        processed/succeeded/failed/deferred/still-pending plus outstanding
        readiness obligations (V4-14.05)."""
        self._require_open()
        c = self._resolve_caller(caller, self.host.default_scope())
        self._require_grant(c, GrantKind.OPERATOR)
        return self._ingester.drain_report(
            scope=self.host.default_scope(), limit=limit
        )

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "engine is closed")


def open_store(
    data_dir: str,
    cfg: VerbatimConfig,
    host: HostAdapter,
    *,
    create: bool = False,
    judge: Any = None,
    encoder: Any = None,
    deps: Optional[dict[str, Any]] = None,
    store_path: Optional[str] = None,
    prefer_store: Optional[str] = None,
) -> Engine:
    """Open (or create) a profile store and bind it to a host context.

    The store path is selected by ``storage.resolver.resolve_store_path``
    — the single profile-store policy every surface shares (V4-07.10,
    F4-18): the profile store ``{profile_id}.db`` is the canonical name, a
    discovered ``v3.db`` is adopted rather than replaced, and a directory
    holding BOTH conventions is a ``STORE_CONFLICT`` until the operator
    decides via ``store_path`` (explicit path) or ``prefer_store``
    ("legacy"/"v3"). Never a silent merge or empty replacement (V4-05.10).

    When ``judge``/``encoder`` are not injected, they are constructed from
    the validated config — the same path the CLI and Hermes adapter use —
    so a configured backend is never silently absent (v1 gap F09).
    """
    cfg = cfg.validate()
    from .storage.resolver import require_store_path, resolve_store_path

    resolution = resolve_store_path(
        data_dir,
        profile_id=host.profile_id(),
        explicit_path=store_path,
        create=create,
        prefer=prefer_store,
    )
    path = require_store_path(resolution)
    if os.path.exists(path):
        store = Store.open(path)
    elif create:
        store = Store.create(path)
    else:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_FORBIDDEN, f"no store at {path}"
        )
    try:
        if judge is None:
            judge = _build_judge(cfg, store, host, deps or {})
        if encoder is None:
            encoder = _build_encoder(cfg, deps or {}, host, store=store)
    except Exception:
        store.close()
        raise
    return Engine(store, cfg, host, judge=judge, encoder=encoder)


def _build_judge(
    cfg: VerbatimConfig, store: Store, host: HostAdapter, deps: dict[str, Any]
) -> Any:
    """Construct the configured decision backend through the registry.

    ``rules`` needs no wiring; ``jev`` gets the egress gate and the host's
    scoped secret accessor — never a raw environment fallback (SPEC §41).
    """
    from .decisions.backend import default_registry

    name = cfg.judge.backend
    if name == "rules":
        return default_registry().create("rules", cfg, host.default_scope(), {})
    build_deps = dict(deps)
    if "egress" not in build_deps:
        from .privacy.egress import EgressGate

        build_deps["egress"] = EgressGate(store, cfg)
    if "secret_getter" not in build_deps:
        build_deps["secret_getter"] = host.secret
    return default_registry().create(name, cfg, host.default_scope(), build_deps)


def _build_encoder(
    cfg: VerbatimConfig,
    deps: dict[str, Any],
    host: HostAdapter,
    store: Optional[Store] = None,
) -> Any:
    """Construct the configured encoder; ``none`` returns None honestly.

    A transport-bound encoder additionally gets the profile's
    ``TransportBroker`` attached (F4-04): the broker is constructed with
    the encoder's own ``endpoint_descriptor`` allowlisted, bound to the
    encoder, and exposed as ``store.transport_broker`` so the document
    (``_do_embed``) and query (``_semantic``) paths dispatch under permits
    — an encoder built through any other path stays fail-closed.
    """
    from .embeddings.encoder import encoder_requires_permit, get_encoder

    enc = get_encoder(cfg, http=deps.get("http"), secret_getter=host.secret)
    if enc is not None and store is not None and encoder_requires_permit(enc):
        from .privacy.broker import TransportBroker

        try:
            desc = enc.endpoint_descriptor()
        except Exception:
            # No allowlisted endpoint → every dispatch denies (fail closed).
            desc = None
        broker = TransportBroker(
            store, cfg, endpoints=[desc] if desc is not None else ()
        )
        enc._broker = broker
        store.transport_broker = broker
    return enc
