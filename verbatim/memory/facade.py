"""V5 consumer facade — ``Memory`` (SPEC_V5 §05–§09, §15; docs/v5_contracts.md §10).

The small evidence-first surface: ``Memory() -> add -> wait_ready -> search``,
plus ``inspect``/``forget``/``status``/``close``.  Everything composes the
existing kernel — governance for authority, ``ingest_envelope`` for the atomic
capture, ``ReadinessEngine`` for readiness, the V3 governed recall/evidence
lanes plus the v5 source lane for retrieval, ``source_state`` for control
versions, and the existing suppression/purge machinery for forget.  This
module owns no second store, queue, policy, or retrieval implementation.

Honesty rules wired in here:

* ``user_id`` is an alias *label*, never authority — the caller is bound to
  the host's authenticated principal only (V5-05.04).
* ``host`` selects a registered binding — ``"local"`` (default) or
  ``"hermes"`` — that supplies the trusted identity/secrets/thread seam;
  the profile stays ``local_memory`` and store+worker sharing is the same
  honest path (V5-06.14, §19).
* ``worker='external'`` never starts a helper; ``'managed'`` acquires the
  shared per-store worker and is released on ``close`` (V5-09).
* Capabilities that are not provisioned in this build report as such —
  ``deferred`` obligations, ``unavailable`` lanes, warnings on the result —
  never fabricated readiness (V5-06.04, V5-08.13).
* ``Memory`` is importable with zero side effects; every public method on a
  closed/closing or fork-inherited object fails typed (V5-09.07/09.10).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import governance
from ..api_v3.facade import VerbatimV3
from ..config import VerbatimConfig, config_from_mapping
from ..core.serialize import json_dumps
from ..core.time import now_us, rfc3339
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import (
    EnvelopeKind,
    Perspective,
    SourceEnvelopeV3,
    TrustClass,
    Verb,
)
from ..governance import CallerV3
from ..core.types_v4 import CapabilityName
from ..evidence.envelopes import ingest_envelope
from ..host import HostAdapter, LocalHost
from ..jobs.queue import JobQueue
from ..readiness import (
    CAP_SOURCE_LEXICAL,
    SOURCE_CAPS,
    ReadinessEngine,
)
from ..sourcestate import state as _sstate
from ..sourcestate import transitions as _stransitions
from ..storage.repos import SourcesRepo, has_table
from ..storage.resolver import require_store_path, resolve_store_path
from ..storage.store import Store
from . import aliases as _aliases
from .bootstrap import ROOT_PURPOSES, ROOT_VERBS, ensure_bootstrap
from .controls import MemoryControls, object_ref_to_string
from .errors import (
    closed,
    conflict,
    deadline,
    denied,
    forked,
    invalid,
    unavailable,
)
from .types import (
    Acceptance,
    AddResult,
    CloseReport,
    Consistency,
    ForgetResult,
    Hit,
    Inspection,
    MemoryRef,
    MemoryStatus,
    MemoryType,
    Readiness,
    SearchResult,
    SearchStatus,
    SupportStatus,
    UpdateCandidate,
    WorkerMode,
    monotonic_ms,
)

# --- optional seams: report, never silently swallow ------------------------
try:  # query_analysis/v1 (§31) — deterministic classifier
    from ..querying.analyze import analyze as _analyze_query
except Exception:  # pragma: no cover - absent on partial checkouts
    _analyze_query = None

try:  # §30.5 advisory update candidates — needs the v5 tables
    from ..querying.updates import (
        NewRecord,
        detect_update_candidates,
        list_open_candidates as _list_open_candidates,
    )
except Exception:  # pragma: no cover
    NewRecord = None
    detect_update_candidates = None
    _list_open_candidates = None

try:  # V6-03.03 auto_safe namespace policy — same effect as replaces=
    from ..querying.auto_update import auto_safe_replace as _auto_safe_replace
except Exception:  # pragma: no cover
    _auto_safe_replace = None

try:  # §30.2 dedup link plane — collapse views + submission counters
    from ..dedup import links as _dedup_links
except Exception:  # pragma: no cover
    _dedup_links = None

try:  # §31.3 per-class support verdict (support_verdict/v1)
    from ..querying.verdict import search_verdict as _search_verdict
except Exception:  # pragma: no cover
    _search_verdict = None

try:  # the v5 source lane (lexical projection + similarity + postings)
    from ..retrieval.v3.source_lane import source_candidates as _source_candidates
except Exception:  # pragma: no cover
    _source_candidates = None

try:  # ranking/v1 fusion over admitted candidates
    from ..retrieval.v3.fusion_v1 import fuse as _fuse
except Exception:  # pragma: no cover
    _fuse = None

try:  # v6 typed lane — grounded-fact index first (V6-02.01/02.02)
    from ..retrieval.v3.typed_lane import typed_candidates as _typed_candidates
except Exception:  # pragma: no cover
    _typed_candidates = None

try:  # advisory barrier→source marking (docs/v6_contracts §2)
    from ..storage import commit_notify as _commit_notify
except Exception:  # pragma: no cover
    _commit_notify = None

try:  # deterministic ``cr_*`` receipt ids (barrier→source resolution)
    from ..evidence.receipts import receipt_id_for as _receipt_id_for
except Exception:  # pragma: no cover
    _receipt_id_for = None

try:  # delivery exposure ledger (docs/v6_contracts §8) — sibling module
    from ..influence import exposure as _exposure
except Exception:  # pragma: no cover
    _exposure = None

try:  # V7 retrieval pipeline (§04.2 S0–S8, docs/v7_contracts.md)
    from ..core.types_v7 import (
        BudgetClass,
        IntervalUs,
        LaneContextV7,
        OccurredPrecision,
        OccurredSource,
    )
    from ..retrieval.v7.pipeline import run_search as _v7_run_search
    from ..retrieval.v7.policy import load_policy as _v7_load_policy
except Exception:  # pragma: no cover - absent on partial checkouts
    BudgetClass = None
    IntervalUs = None
    LaneContextV7 = None
    OccurredPrecision = None
    OccurredSource = None
    _v7_run_search = None
    _v7_load_policy = None

try:  # §21.6 temporal resolver — the V8-09.03 T04 year guard replays its
    # candidates; absent on partial checkouts → guard is a no-op.
    from ..enrichment.temporal_v2 import resolve as _v7_temporal_resolve
except Exception:  # pragma: no cover
    _v7_temporal_resolve = None

try:  # V7 S1 query analysis (build_query_view) — query_view.py
    from ..querying.query_view import build_query_view as _v7_query_view
except Exception:  # pragma: no cover
    _v7_query_view = None

try:  # V7 eligibility adapter (quarantine/purge/scope cascade)
    from ..retrieval.v7.eligibility import make_eligible as _v7_make_eligible
except Exception:  # pragma: no cover
    _v7_make_eligible = None

try:  # per-store result cache (V4-33 machinery; V6-02.13 consumer path)
    from ..retrieval import cache as _rcache
except Exception:  # pragma: no cover
    _rcache = None

try:  # default local encoder — pure stdlib, deterministic identity
    from ..embeddings.hashing import HashingEncoder
except Exception:  # pragma: no cover
    HashingEncoder = None


__all__ = ["Memory"]


_PROFILE = "local_memory"
_PRODUCER = "memory/v5"
_DEFAULT_ALIAS = "default"
_STATE_TABLE = "source_state"

_MAX_CONTENT_BYTES = 1_048_576
_MAX_METADATA_BYTES = 8_192
_MAX_METADATA_DEPTH = 8
_MAX_LIMIT = 64
_MAX_IDEMPOTENCY_KEY = 128
_SESSION_RECEIPT_MAX = 512
_CAUSAL_PREFIX = "memory.causal."
_CAUSAL_TTL_US = 7 * 24 * 3600 * 1_000_000
#: Post-delivery writes (causal-token mint, exposure rows) run AFTER the
#: result is assembled; a 5ms starve budget made them silently degrade
#: whenever the managed worker held a short job tx. 50ms stays bounded —
#: far under the 250ms writer busy cap — while surviving normal drain.
_POST_DELIVERY_WRITE_MS = 50
_IDEM_PREFIX = "memory.idem."
_SESSION_PREFIX = "memory.session."

#: Turn-capture bounds (V7 D7-26): ``Memory.add`` conversational fields
#: persist into ``source_revisions.metadata_json`` under the
#: ``projections/units_v7`` add-args key contract (``messages``,
#: ``speaker``, ``message_at``, ``occurred``, ``session_id``). The bounds
#: keep one add from minting unbounded metadata/unit fan-out.
_MAX_TURN_MESSAGES = 512
_MAX_TURN_TEXT_BYTES = 262_144  # 256 KiB per message text
_MAX_SPEAKER_CHARS = 128
_MAX_SESSION_CHARS = 256
#: ``bulk_add`` commits one ``store.tx()`` per chunk of items; a failed
#: item rolls back through its own SAVEPOINT, never the chunk.
_BULK_CHUNK = 64
_MAX_BULK_CHUNK = 512

_TERM_RE = re.compile(r"[a-z0-9_]+")

#: Caller-declared retention/sensitivity control keys inside ``metadata``
#: (V5-07.03, V5-06.18). The declared vocabulary is deliberately finite —
#: an unrecognized value is a typed rejection, never a silently-ignored
#: security/retention option (V5-06.08). Protected classes keep the bytes
#: as durable evidence but exclude them from every plaintext projection.
_RETENTION_ORDINARY = frozenset({"default", "standard", "persistent", "normal"})
_RETENTION_PROTECTED = frozenset({"none", "zero", "ephemeral", "protected"})
_SENSITIVITY_ORDINARY = frozenset({"public", "normal", "low", "internal"})
_SENSITIVITY_PROTECTED = frozenset(
    {"secret", "sensitive", "confidential", "private", "restricted", "high"}
)

#: Registered host bindings for ``Memory(host=...)`` (V5-06.14): the name
#: selects a fixed trusted binding — it is never a request-supplied
#: identity object and never mints caller authority.
_HOST_NAMES = frozenset({"local", "hermes"})

#: Process-local constructor serialization per resolved store path
#: (V5-05.09): racing in-process ``Memory()`` constructors on a fresh path
#: must serialize through create → bootstrap so the loser observes the
#: winner's committed binding and converges — a divergent second owner or
#: a torn file from concurrent creates is never an allowed outcome.
_INIT_LOCKS: Dict[str, threading.Lock] = {}
_INIT_LOCKS_GUARD = threading.Lock()
_INIT_LOCKS_PID = os.getpid()


def _init_lock(path: str) -> threading.Lock:
    """The construction lock for one resolved store path (fork-safe)."""
    global _INIT_LOCKS_PID
    pid = os.getpid()
    key = os.path.abspath(path)
    with _INIT_LOCKS_GUARD:
        if _INIT_LOCKS_PID != pid:
            # A forked child must not inherit lock objects the parent's
            # dead threads may have held — rebuild in the child.
            _INIT_LOCKS.clear()
            _INIT_LOCKS_PID = pid
        return _INIT_LOCKS.setdefault(key, threading.Lock())


def _open_once(target: str, create: bool) -> Store:
    """One open/create attempt with the raw driver surface translated.

    ``OperationalError``/``sqlite3.Error``/``OSError`` (incl.
    ``FileNotFoundError`` races on the path or sidecars) map into the
    typed taxonomy — the constructor never leaks them (V5-05.09).
    """
    try:
        if create:
            return Store.create(target)
        return Store.open(target)
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            raise VerbatimError(
                ErrorCode.STORE_BUSY,
                "store is busy — a peer holds the write lock",
                retryable=True,
            ) from exc
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED, f"store open failed: {exc}"
        ) from exc
    except sqlite3.Error as exc:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT, f"store open failed: {exc}"
        ) from exc
    except OSError as exc:
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED,
            f"store path failed during open/create: {exc}",
        ) from exc


def _converged(exc: BaseException) -> bool:
    """True when the failure shape is a peer's still-finishing create —
    worth a bounded retry rather than a verdict."""
    if isinstance(exc, VerbatimError):
        if exc.code == ErrorCode.CONFIG_INVALID and (
            "does not exist" in str(exc) or "already exists" in str(exc)
        ):
            return True
        if exc.code == ErrorCode.STORE_CORRUPT and "empty" in str(exc):
            return True
        return exc.code in (ErrorCode.STORE_BUSY, ErrorCode.LOCKED)
    return False


def _open_or_create(target: str, create: bool) -> Store:
    """Open-or-create with peer convergence (V5-05.09).

    The exists/size probe is advisory — a peer may be mid-create between
    the probe and our own call. A create that loses to a peer converges
    by opening the winner's store, and a loser that arrives before the
    winner's first commit retries a bounded window instead of
    condemning a half-visible file. Every outcome is either a Store
    handle or a typed VerbatimError.
    """
    try:
        ready = os.path.exists(target) and os.path.getsize(target) > 0
    except OSError:
        ready = False
    deadline = time.monotonic() + 5.0  # bounded convergence window
    while True:
        try:
            return _open_once(target, create=create if not ready else False)
        except VerbatimError as exc:
            if (
                create
                and exc.code == ErrorCode.CONFIG_INVALID
                and "already exists" in str(exc)
            ):
                # We lost a create race — converge on the winner's store.
                # Its first commit may still be in flight; the retry loop
                # covers that window honestly.
                try:
                    return _open_once(target, create=False)
                except VerbatimError as inner:
                    if not _converged(inner) or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
                    ready = True
                    continue
            if not _converged(exc) or time.monotonic() >= deadline:
                raise
            if (
                not create
                and exc.code == ErrorCode.CONFIG_INVALID
                and "does not exist" in str(exc)
            ):
                # create=False on a genuinely missing store fails fast —
                # convergence waits are only owed while a peer may be
                # creating on our behalf.
                raise
            time.sleep(0.02)


def _protection_declared(meta: Dict[str, Any]) -> bool:
    """True when caller metadata declares a protected retention or
    sensitivity class, or the reserved ``protected`` flag is set."""
    retention = meta.get("retention")
    if isinstance(retention, str) and (
        retention.strip().lower() in _RETENTION_PROTECTED
    ):
        return True
    sensitivity = meta.get("sensitivity")
    if isinstance(sensitivity, str) and (
        sensitivity.strip().lower() in _SENSITIVITY_PROTECTED
    ):
        return True
    return meta.get("protected") is True


def _check_retention_policy(meta: Dict[str, Any]) -> bool:
    """Validate declared retention/sensitivity markers; return True when
    the write is protected (V5-07.03).

    The control keys accept only the declared vocabulary: an unknown
    retention or sensitivity value, or a non-bool ``protected`` flag, is
    a typed VALIDATION rejection — never a silently-ignored
    security/retention option (V5-06.08).
    """
    protected = _protection_declared(meta)
    for key, known in (
        ("retention", _RETENTION_PROTECTED | _RETENTION_ORDINARY),
        ("sensitivity", _SENSITIVITY_PROTECTED | _SENSITIVITY_ORDINARY),
    ):
        value = meta.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or value.strip().lower() not in known:
            raise invalid(
                f"metadata.{key} {value!r} is not a declared policy value"
            )
    flag = meta.get("protected")
    if flag is not None and not isinstance(flag, bool):
        raise invalid("metadata.protected must be a bool")
    return protected


def _resolve_host(host: Any) -> HostAdapter:
    """Resolve ``host`` to a trusted HostAdapter binding (V5-06.14).

    ``None``/``"local"`` binds the standalone :class:`LocalHost`; a
    registered name selects the fixed binding for that host integration
    (``"hermes"`` → the ``adapters/hermes_v3`` host binding); a live
    ``HostAdapter`` instance binds directly. Anything else is a typed
    rejection — a name or object can never mint caller identity.
    """
    if host is None:
        return LocalHost()
    if isinstance(host, str):
        name = host.strip().lower()
        if name == "local":
            return LocalHost()
        if name == "hermes":
            return _hermes_host()
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            f"unknown host binding {host!r} — registered bindings: "
            f"{sorted(_HOST_NAMES)}",
        )
    if not isinstance(host, HostAdapter):
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "host must satisfy the HostAdapter protocol",
        )
    return host


def _started_spawn(spawn: Any) -> Any:
    """Wrap a host's ``spawn_thread`` so the returned thread is running.

    The HostAdapter contract is "already started" — LocalHost honors it,
    and Hermes' ``spawn_context_thread`` does too — but the provider's
    standalone fallback (no Hermes agent package installed) returns an
    unstarted ``threading.Thread``. Enforcing the contract at this seam
    keeps the managed worker honest under ``host="hermes"`` without
    distinguishing which host produced the callable.
    """

    def _spawn(fn: Any, name: str) -> Any:
        t = spawn(fn, name)
        if isinstance(t, threading.Thread) and t.ident is None:
            # ident is None only before start(); an already-started or
            # finished thread keeps it set — no double-start risk.
            t.start()
        return t

    return _spawn


def _hermes_host() -> HostAdapter:
    """The registered ``"hermes"`` binding (V5-19.01/19.04).

    Reuses the same ``_HermesHost`` HostAdapter the provider and
    ``adapters/hermes_v3`` bind — Hermes scoped secrets (fail-closed when
    absent), context-thread spawning, hermes_logging, and the profile's
    canonical storage identity — so the facade shares the store, the
    managed worker, and the prefetch constraints through one host seam
    rather than impersonating a session. The bound principal is the
    adapter's documented session fallback (``hermes-principal``): a
    standalone facade is trusted process code, and the host's
    authenticated identity is never caller-supplied.
    """
    from ..provider import _HermesHost

    return _HermesHost(
        hermes_home=os.path.expanduser("~/.hermes"),
        session_id="",
        principal_id="hermes-principal",
        workspace_id=None,
    )


_LIFECYCLE_MAP = {
    "active": "active",
    "recorded": "active",
    "superseded": "superseded",
    "corrected": "corrected",
    "retracted": "retracted",
    "archived": "retracted",
    "erased": "retracted",
    "expired": "expired",
}

#: V7 group support labels (V7-11.01: supported|partial|weak) projected
#: onto the governed ``SupportStatus`` enum. ``weak`` evidence is not
#: adequate support — it maps to ``insufficient`` (honest "not backed"),
#: never silently to ``supported``. ``partial`` carries verified support.
_V7_SUPPORT_TO_STATUS = {
    "supported": SupportStatus.SUPPORTED.value,
    "partial": SupportStatus.SUPPORTED.value,
    "weak": SupportStatus.INSUFFICIENT.value,
}

#: Consumer result cache (V6-02.13): the Hit dataclass fields are the
#: stored serialization shape (same surface ``Hit.to_dict`` ships); the
#: non-field extras a delivered hit can carry (typed ``pins``, raw lane
#: ``signals``) ride beside them under ``"+"``-prefixed keys.
_PACK_CACHE_HIT_FIELDS = tuple(f.name for f in dataclasses.fields(Hit))
_PACK_CACHE_HIT_FIELD_SET = frozenset(_PACK_CACHE_HIT_FIELDS)
_PACK_CACHE_HIT_EXTRAS = ("pins", "signals")
_FILTER_KEYS = frozenset(
    {"type", "kind", "lifecycle", "source_id", "created_after", "created_before"}
)


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonneg_ms(value: Any, field: str) -> float:
    if not _is_num(value) or value < 0:
        raise invalid(f"{field} must be a non-negative number")
    return float(value)


def _payload_bytes(content: Any) -> bytes:
    """Strict UTF-8 — bytes are decoded to prove it, never coerced."""
    if isinstance(content, str):
        payload = content.encode("utf-8")
    elif isinstance(content, (bytes, bytearray, memoryview)):
        payload = bytes(content)
        try:
            payload.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise invalid(f"content is not valid UTF-8: {exc}") from exc
    else:
        raise invalid("content must be str or UTF-8 bytes")
    if not payload:
        raise invalid("content must not be empty")
    if len(payload) > _MAX_CONTENT_BYTES:
        raise invalid(f"content exceeds {_MAX_CONTENT_BYTES} bytes")
    return payload


def _transcript_payload(messages: Any) -> bytes:
    """Synthesize the content payload for ``add(messages=...)`` with no
    explicit content: one ``speaker: text`` line per message.

    The persisted bytes are the units projection's byte-pin ground
    truth — each message text must appear verbatim so the deriver's
    sequential ``_locate`` search pins every emitted turn inside the
    source (V85-03.02's one-source-per-turn shape). ``messages`` arrives
    already normalized by ``_turn_meta``/``_turn_messages``.
    """
    if messages is None:
        raise invalid(
            "content is required (str or UTF-8 bytes) — or supply messages="
        )
    lines: List[str] = []
    for m in messages:
        if isinstance(m, str):
            text, speaker = m, None
        elif isinstance(m, dict):
            text = next(
                (m[k] for k in _MSG_TEXT_KEYS if m.get(k)),
                None,
            )
            speaker = next(
                (m[k] for k in _MSG_SPEAKER_KEYS if m.get(k)), None
            )
        else:
            continue
        if not text:
            continue
        lines.append(f"{speaker}: {text}" if speaker else str(text))
    payload = "\n".join(lines).encode("utf-8")
    if not payload:
        raise invalid("messages carry no text to persist")
    if len(payload) > _MAX_CONTENT_BYTES:
        raise invalid(
            f"synthesized transcript exceeds {_MAX_CONTENT_BYTES} bytes"
        )
    return payload


def _metadata_ok(value: Any, depth: int = 0) -> None:
    if depth > _MAX_METADATA_DEPTH:
        raise invalid("metadata is nested too deeply")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise invalid("metadata keys must be strings")
            _metadata_ok(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _metadata_ok(v, depth + 1)
    elif value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:  # NaN
            raise invalid("metadata must be finite JSON values")
    else:
        raise invalid(f"metadata value of type {type(value).__name__} unsupported")


def _metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise invalid("metadata must be a mapping")
    _metadata_ok(value)
    encoded = json_dumps(value)
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise invalid(f"metadata exceeds {_MAX_METADATA_BYTES} bytes")
    return dict(value)


def _key_ok(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise invalid(f"{field} must be a non-empty string")
    if len(value) > _MAX_IDEMPOTENCY_KEY or not re.match(
        r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$", value
    ):
        raise invalid(f"{field} is not a portable token")
    return value


# ---------------------------------------------------------------------------
# turn-capture validation (V7 D7-26) — the ``units_v7`` add-args contract
# ---------------------------------------------------------------------------
#
# The keys emitted here are the persisted add-args channel the V7 units
# projection reads verbatim out of ``source_revisions.metadata_json``:
# top-level ``messages`` (per-message ``speaker``/``text``/``at`` …),
# ``speaker``, ``message_at``, ``occurred``, ``session_id`` — see the
# ``verbatim/projections/units_v7.py`` module docstring for the contract.

_MSG_TEXT_KEYS = ("text", "content", "message")
_MSG_SPEAKER_KEYS = ("speaker", "speaker_id", "name", "author")
#: Human-facing time keys (magnitude/RFC3339 heuristic lane) + raw-µs keys.
_MSG_TIME_KEYS = (
    "at",
    "message_at",
    "timestamp",
    "ts",
    "recorded_at_us",
    "event_us",
    "message_at_us",
)
_MSG_SESSION_KEYS = ("session_id", "session")
_MSG_OCCURRED_FLAT = (
    "occurred_start_us",
    "occurred_end_us",
    "occurred_precision",
    "occurred_source",
)
_INT_TOKEN_RE = re.compile(r"[+-]?\d+")


def _printable_token(value: Any, field: str, max_chars: int) -> str:
    """Speaker/session identity tokens: non-empty, printable, bounded."""
    if not isinstance(value, str) or not value.strip():
        raise invalid(f"{field} must be a non-empty string")
    if len(value) > max_chars:
        raise invalid(f"{field} exceeds {max_chars} characters")
    if not value.isprintable():
        raise invalid(f"{field} must be printable text")
    return value


def _rfc3339_to_us(text: str, field: str) -> int:
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise invalid(
            f"{field} must be int µs, RFC3339 text, or a datetime"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def _occurred_to_us(value: Any, field: str = "occurred_at") -> int:
    """``occurred_at`` → int µs. Accepts int µs (literal — never
    magnitude-guessed), integral floats, RFC3339 strings, and datetimes
    (naive reads as UTC, mirroring ``units_v7._as_us``)."""
    if isinstance(value, bool):
        raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 10**21:
            return int(value)
        raise invalid(f"{field} float must be an exact µs integer")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")
        if _INT_TOKEN_RE.fullmatch(s):
            return int(s)
        return _rfc3339_to_us(s, field)
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")


def _as_of_in_range(v: int) -> int:
    """An ``as_of`` anchor must land on a representable instant — absurd
    magnitudes are ``VALIDATION``, never silently folded into windows."""
    try:
        datetime.fromtimestamp(v / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise invalid("as_of is out of representable range") from exc
    return v


def _as_of_to_us(value: Any) -> int:
    """V8-09.01/V8-20.01 — strict ``as_of`` → int µs (never a wall-clock
    guess). Accepted shapes mirror ``_occurred_to_us`` — int µs, integral
    float, numeric string, RFC3339 string, datetime — except that the
    anchor must be unambiguous: a naive datetime or an offset-less date
    string raises ``VALIDATION`` naming ``as_of`` instead of silently
    reading as UTC (V8-20.05)."""
    field = "as_of"
    bad = (
        f"{field} must be int µs, an RFC3339 string with an explicit"
        " offset/'Z', or a timezone-aware datetime"
    )
    if isinstance(value, bool) or value is None:
        raise invalid(bad)
    if isinstance(value, int):
        return _as_of_in_range(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 10**21:
            return _as_of_in_range(int(value))
        raise invalid(f"{field} float must be an exact µs integer")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise invalid(f"{field} datetime must be timezone-aware")
        return int(value.timestamp() * 1_000_000)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise invalid(bad)
        if _INT_TOKEN_RE.fullmatch(s):
            return _as_of_in_range(int(s))
        iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError as exc:
            raise invalid(bad) from exc
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise invalid(
                f"{field} string must carry an explicit offset or 'Z'"
            )
        return int(dt.timestamp() * 1_000_000)
    raise invalid(bad)


def _dig(obj: Any, *keys: str) -> Any:
    """Defensive nested lookup over dicts/attrs — absent keys → None."""
    for key in keys:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def _v7_answerability(result: Any) -> Optional[str]:
    """V8-20.02 — the verdict report's ``answerability``, extracted
    defensively from whatever the pipeline currently carries (a direct
    attribute, a verdict report object, or the coverage/explain verdict
    blocks). Absent → ``None``; a value is never fabricated from status."""
    if result is None:
        return None
    coverage = getattr(result, "coverage", None)
    explain = getattr(result, "explain", None)
    for probe in (
        getattr(result, "answerability", None),
        _dig(getattr(result, "verdict_report", None), "answerability"),
        _dig(getattr(result, "verdict", None), "answerability"),
        _dig(coverage, "verdict", "answerability"),
        _dig(coverage, "rerank", "stages", "verdict", "answerability"),
        _dig(explain, "verdict", "answerability"),
    ):
        if isinstance(probe, str) and probe:
            return probe
    return None


# V8-09.03 — bare four-digit years (resolver rule T04): a year window is
# legitimate only when the surface follows a temporal cue from the owned
# list below, or lies within [anchor_year - 150, anchor_year + 5] and is
# not adjacent to a capitalized title token. "Cyberpunk 2077" must never
# become a window while "in 2019" still does.
_YEAR_CUE_TAIL_RE = re.compile(
    r"(?:in|since|before|after|during|until|by|around|from|year|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)\s+(?:(?:the|a|an|of)\s+)?$",
    re.IGNORECASE,
)
_WORD_BEFORE_RE = re.compile(r"[A-Za-z']+\s*$")
_WORD_AFTER_RE = re.compile(r"\s*[A-Za-z']+")


def _v7_year_legit(raw: str, start: int, end: int, year: int, anchor_year: int) -> bool:
    """One ``YYYY`` occurrence in the raw query → keep its T04 window?"""
    if _YEAR_CUE_TAIL_RE.search(raw[:start]):
        return True
    if not (anchor_year - 150 <= year <= anchor_year + 5):
        return False
    prev = _WORD_BEFORE_RE.search(raw[:start])
    if prev is not None and prev.group(0).strip()[:1].isupper():
        return False  # capitalized title token, no cue ("Cyberpunk 2077")
    nxt = _WORD_AFTER_RE.match(raw[end:])
    if nxt is not None and nxt.group(0).strip()[:1].isupper():
        return False
    return True


def _v7_window_union(rts: List[Any], anchor_us: int) -> Optional[Any]:
    """Rebuild ``resolve_query_window``'s union over a filtered candidate
    set — identical fold rules (coarsest precision, explicit-only source,
    sorted rule ids), just without the dropped T04 spans."""
    usable = [
        r for r in rts
        if r.interval.start_us is not None or r.interval.end_us is not None
    ]
    if not usable:
        return None
    starts = [r.interval.start_us for r in usable
              if r.interval.start_us is not None]
    ends = [r.interval.end_us for r in usable
            if r.interval.end_us is not None]
    rank = {
        OccurredPrecision.INSTANT: 0, OccurredPrecision.DAY: 1,
        OccurredPrecision.WEEK: 2, OccurredPrecision.MONTH: 3,
        OccurredPrecision.SEASON: 4, OccurredPrecision.YEAR: 5,
        OccurredPrecision.DECADE: 6, OccurredPrecision.UNKNOWN: 7,
    }
    prec = max(
        (r.interval.precision for r in usable),
        key=lambda p: rank.get(p, 7),
    )
    src = (
        OccurredSource.EXPLICIT
        if all(r.interval.source == OccurredSource.EXPLICIT for r in usable)
        else OccurredSource.RESOLVED_RELATIVE
    )
    rules = "+".join(sorted({r.rule_id for r in usable}))
    return IntervalUs(
        min(starts) if starts else None,
        max(ends) if ends else None,
        precision=prec,
        source=src,
        rule_id=rules,
        anchor_us=int(anchor_us),
    )


def _v7_guard_year_window(qv: Any) -> Any:
    """V8-09.03 post-pass on a built ``QueryViewV7``: drop illegitimate
    T04 bare-year candidates and rebuild the union window. The resolver
    (unowned) emits every ``YYYY`` — this is the owned consumer-side
    guard; facets recurse. A resolver failure keeps the upstream window —
    the guard never invents a window of its own."""
    if (
        qv is None
        or _v7_temporal_resolve is None
        or IntervalUs is None
        or OccurredPrecision is None
    ):
        return qv
    facets = tuple(getattr(qv, "facets", None) or ())
    if facets:
        guarded = tuple(_v7_guard_year_window(f) for f in facets)
        if guarded != facets:
            qv = dataclasses.replace(qv, facets=guarded)
    intent = getattr(qv, "intent", None)
    window = getattr(intent, "window", None)
    if window is None or "T04" not in str(window.rule_id or "").split("+"):
        return qv
    anchor = getattr(qv, "query_time_us", None)
    if not isinstance(anchor, int) or isinstance(anchor, bool):
        anchor = now_us()
    norm = getattr(qv, "norm", None)
    text = getattr(norm, "text", "") or " ".join(
        getattr(t, "term", "")
        for t in (getattr(norm, "terms", ()) or ())
    )
    try:
        rts = _v7_temporal_resolve(text, int(anchor))
    except Exception:
        return qv
    raw = getattr(qv, "query", "") or text
    anchor_year = datetime.fromtimestamp(
        int(anchor) / 1_000_000, tz=timezone.utc
    ).year
    kept: List[Any] = []
    dropped = False
    for rt in rts:
        if getattr(rt, "rule_id", None) != "T04":
            kept.append(rt)
            continue
        surface = (getattr(rt, "text", "") or "").strip()
        if not re.fullmatch(r"(?:19|20)\d{2}", surface):
            kept.append(rt)  # "1990s" decades stay — an explicit marker
            continue
        year = int(surface)
        found = False
        legit = False
        for m in re.finditer(rf"(?<!\d){year}(?!\d)", raw):
            found = True
            if _v7_year_legit(raw, m.start(), m.end(), year, anchor_year):
                legit = True
                break
        if legit or not found:
            kept.append(rt)  # unseen in raw text → nothing to prove against
        else:
            dropped = True
    if not dropped:
        return qv
    return dataclasses.replace(
        qv,
        intent=dataclasses.replace(
            intent, window=_v7_window_union(kept, int(anchor))
        ),
    )


def _msg_time_ok(value: Any, field: str) -> Any:
    """Validate a per-message time value; returns the JSON-safe form.

    Ints / integral floats / numeric strings / RFC3339 strings pass
    verbatim (``units_v7._as_us`` re-reads them by documented magnitude).
    Datetimes convert to µs ints (JSON cannot carry a datetime). Other
    types — and non-integral floats, whose re-interpretation would
    silently mis-scale — are typed rejections."""
    if isinstance(value, bool) or value is None:
        raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise invalid(
            f"{field} float must be an exact integer — use int µs or RFC3339"
        )
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")
        if _INT_TOKEN_RE.fullmatch(s):
            return value
        _rfc3339_to_us(s, field)  # validates; keeps the caller's text
        return value
    raise invalid(f"{field} must be int µs, RFC3339 text, or a datetime")


def _occurred_arg_ok(value: Any, field: str) -> Dict[str, Any]:
    """Validate a message/add-args ``occurred`` value → plain JSON dict
    (``{start_us, end_us, precision, source}``). Accepts the dict shape
    ``units_v7._norm_occurred`` reads; an ``IntervalUs`` is converted to
    it (dataclasses are not JSON-serializable in metadata)."""
    from ..core.types_v7 import IntervalUs  # local: parallel-worker seam

    if isinstance(value, IntervalUs):
        return {
            "start_us": value.start_us,
            "end_us": value.end_us,
            "precision": getattr(value.precision, "value", value.precision),
            "source": getattr(value.source, "value", value.source),
        }
    if not isinstance(value, dict):
        raise invalid(f"{field} must be a dict or IntervalUs")
    out: Dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise invalid(f"{field} keys must be strings")
        if key in ("start", "end", "start_us", "end_us"):
            if raw is None:
                out[key] = None
            else:
                out[key] = _msg_time_ok(raw, f"{field}.{key}")
        elif key in ("precision", "source"):
            if raw is not None and not isinstance(raw, str):
                raise invalid(f"{field}.{key} must be a string")
            out[key] = raw
        else:
            _metadata_ok(raw)  # pass-through keys stay JSON-bounded
            out[key] = raw
    return out


def _turn_messages(value: Any) -> List[Any]:
    """Validate/normalize the ``messages`` add-arg.

    Accepts a list/tuple of strings or dicts (``speaker``/``text``/``at``
    optional per message; a lone dict or string wraps to one entry).
    Returns a JSON-safe list preserving caller keys verbatim — entries
    carry the ``units_v7`` per-message contract through untouched."""
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise invalid("messages must be a list of strings or dicts")
    if len(value) > _MAX_TURN_MESSAGES:
        raise invalid(f"messages exceeds {_MAX_TURN_MESSAGES} entries")
    out: List[Any] = []
    for i, msg in enumerate(value):
        field = f"messages[{i}]"
        if isinstance(msg, str):
            if len(msg.encode("utf-8")) > _MAX_TURN_TEXT_BYTES:
                raise invalid(f"{field} exceeds {_MAX_TURN_TEXT_BYTES} bytes")
            out.append(msg)
            continue
        if not isinstance(msg, dict):
            raise invalid(f"{field} must be a dict or string")
        rec: Dict[str, Any] = {}
        for k, v in msg.items():
            if not isinstance(k, str):
                raise invalid(f"{field} keys must be strings")
            rec[k] = v
        for key in _MSG_TEXT_KEYS:
            text = rec.get(key)
            if text is None:
                continue
            if not isinstance(text, str):
                raise invalid(f"{field}.{key} must be a string")
            if len(text.encode("utf-8")) > _MAX_TURN_TEXT_BYTES:
                raise invalid(
                    f"{field}.{key} exceeds {_MAX_TURN_TEXT_BYTES} bytes"
                )
        for key in _MSG_SPEAKER_KEYS:
            if rec.get(key) is not None:
                rec[key] = _printable_token(
                    rec[key], f"{field}.{key}", _MAX_SPEAKER_CHARS
                )
        for key in _MSG_TIME_KEYS:
            if rec.get(key) is not None:
                rec[key] = _msg_time_ok(rec[key], f"{field}.{key}")
        for key in _MSG_SESSION_KEYS:
            if rec.get(key) is not None:
                rec[key] = _printable_token(
                    rec[key], f"{field}.{key}", _MAX_SESSION_CHARS
                )
        if rec.get("occurred") is not None:
            rec["occurred"] = _occurred_arg_ok(
                rec["occurred"], f"{field}.occurred"
            )
        for key in ("byte_start", "byte_end"):
            pin = rec.get(key)
            if pin is not None and (
                not isinstance(pin, int) or isinstance(pin, bool) or pin < 0
            ):
                raise invalid(f"{field}.{key} must be a non-negative int")
        out.append(rec)
    return out


def _turn_meta(
    *,
    speaker: Optional[Any],
    occurred_at: Optional[Any],
    messages: Optional[Any],
    session_id: Optional[Any],
) -> Dict[str, Any]:
    """Turn-capture params → the persisted-metadata fragment (the
    ``units_v7`` add-args contract). Empty dict when no turn fields were
    supplied — the legacy add shape is then byte-identical.

    ``occurred_at`` (the instant the turn happened) lands on both the
    recorded-time lane (``message_at``, µs int) and the event-time lane
    (``occurred`` as a degenerate ``[t, t]`` instant/explicit interval) —
    a turn's saying-time and its occurrence coincide; callers needing
    distinct values set the keys directly via ``metadata``/``messages``.
    ``session_id`` is the caller's conversational-session label for unit
    grouping — the envelope's own ``session_id`` stays the facade's
    capture-session identity."""
    out: Dict[str, Any] = {}
    if speaker is not None:
        out["speaker"] = _printable_token(speaker, "speaker", _MAX_SPEAKER_CHARS)
    if session_id is not None:
        out["session_id"] = _printable_token(
            session_id, "session_id", _MAX_SESSION_CHARS
        )
    if occurred_at is not None:
        us = _occurred_to_us(occurred_at, "occurred_at")
        out["message_at"] = us
        out["occurred"] = {
            "start_us": us,
            "end_us": us,
            "precision": "instant",
            "source": "explicit",
        }
    if messages is not None:
        out["messages"] = _turn_messages(messages)
    return out


def _failed_add_result(exc: BaseException) -> AddResult:
    """Honest per-item failure record for ``bulk_add`` — the item's own
    writes rolled back (savepoint); nothing committed under this result."""
    if isinstance(exc, VerbatimError):
        detail = f"{exc.code.value}: {exc}"
    else:
        detail = f"{type(exc).__name__}: {exc}"
    return AddResult(
        acceptance=Acceptance.FAILED.value,
        inference="not_requested",
        error=detail,
        warnings=["item_failed"],
    )


def _add_result_dict(result: AddResult) -> Dict[str, Any]:
    d = result.to_dict() if hasattr(result, "to_dict") else {}
    if d:
        return d
    return {
        "memory_id": result.memory_id,
        "ref": result.ref,
        "source_revision": result.source_revision,
        "receipt_id": result.receipt_id,
        "acceptance": result.acceptance,
        "replayed": result.replayed,
        "readiness": dict(result.readiness),
        "warnings": list(result.warnings),
        "possible_updates": [
            {
                "ref": u.ref,
                "relation": u.relation,
                "reason": u.reason,
                "score": u.score,
            }
            for u in result.possible_updates
        ],
        "inference": result.inference,
    }


def _add_result_from(data: Dict[str, Any]) -> AddResult:
    if not isinstance(data, dict):
        raise denied("idempotency record is not readable")
    updates = [
        UpdateCandidate(
            ref=str(u.get("ref", "")),
            relation=str(u.get("relation", "")),
            reason=str(u.get("reason", "")),
            score=float(u.get("score", 0.0)),
        )
        for u in data.get("possible_updates") or []
        if isinstance(u, dict)
    ]
    return AddResult(
        memory_id=str(data.get("memory_id", "")),
        ref=str(data.get("ref", "")),
        source_revision=int(data.get("source_revision", 0)),
        receipt_id=str(data.get("receipt_id", "")),
        acceptance=str(data.get("acceptance", "accepted")),
        replayed=True,
        readiness={str(k): str(v) for k, v in (data.get("readiness") or {}).items()},
        warnings=[str(w) for w in data.get("warnings") or []],
        possible_updates=updates,
        inference=str(data.get("inference", "not_requested")),
        error=(str(data["error"]) if data.get("error") is not None else None),
    )


# ---------------------------------------------------------------------------
# the facade
# ---------------------------------------------------------------------------


class Memory:
    """The V5 consumer surface (docs/v5_contracts.md §10)."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        profile: str = _PROFILE,
        worker: str = "managed",
        encoder: str = "hashing",
        ready_timeout_ms: float = 200,
        create: bool = True,
        config: Optional[Any] = None,
        host: Optional[Any] = None,
    ) -> None:
        # ---- declaration validation (typed, before any side effect) ----
        if path is not None and not isinstance(path, str):
            raise invalid("path must be a string or None")
        if user_id is not None:
            _aliases.validate_label(user_id, kind="user")
        if profile != _PROFILE:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"profile {profile!r} unsupported — this facade serves "
                f"{_PROFILE!r} only",
            )
        try:
            worker_mode = WorkerMode(worker)
        except ValueError:
            raise invalid("worker must be 'managed' or 'external'")
        if not isinstance(encoder, str) or encoder not in ("hashing", "none", "artifact"):
            raise invalid("encoder must be 'hashing', 'none', or 'artifact'")
        ready_timeout = _nonneg_ms(ready_timeout_ms, "ready_timeout_ms")
        if not isinstance(create, bool):
            raise invalid("create must be a bool")

        if config is None:
            cfg = VerbatimConfig()
        elif isinstance(config, VerbatimConfig):
            cfg = config.validate()
        elif isinstance(config, dict):
            cfg = config_from_mapping(config)
        else:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "config must be a VerbatimConfig or mapping",
            )

        host = _resolve_host(host)

        # The trusted identity is the host's authenticated principal —
        # user_id is only an alias label and never authenticates (V5-05.04).
        scope = host.default_scope()
        owner = getattr(scope, "principal_id", None)
        if not isinstance(owner, str) or not owner:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "host must supply an authenticated principal "
                "(default_scope().principal_id)",
            )

        profile_id = host.profile_id()
        resolution = resolve_store_path(
            cfg.data_dir,
            profile_id=profile_id,
            explicit_path=path,
            create=create,
        )
        target = require_store_path(resolution)

        # V5-05.09: racing constructors on one path serialize through the
        # per-path construction lock — open/create, bootstrap, and worker
        # acquisition run as one critical section so a same-process peer
        # converges on the committed binding instead of tearing a
        # half-created store or minting a divergent owner.
        with _init_lock(target):
            store = _open_or_create(target, create)

            self._store = store
            self._cfg = cfg
            self._host = host
            self._owner = owner
            self._profile_id = profile_id
            self._pid = os.getpid()
            self._lock = threading.RLock()
            self._closed = False
            self._closing = False
            self._session_id = f"sess_{new_id()}"
            self._session_receipts: List[str] = []
            self._ready_timeout_ms = ready_timeout
            self._worker_mode = worker_mode
            self._worker_handle = None
            self._close_report: Optional[CloseReport] = None
            self._warnings: List[str] = []
            try:
                self._init_after_store(encoder, worker_mode, user_id)
            except BaseException:
                # Constructor failure must not leak a store or a shared
                # worker registration (V5-09.12) — unwind what was
                # acquired.
                try:
                    if self._worker_handle is not None:
                        self._worker_handle.release(timeout_ms=1000)
                except Exception:
                    pass
                try:
                    store.close()
                except Exception:
                    pass
                self._closed = True
                raise

    # ---- construction internals ----------------------------------------

    def _init_after_store(
        self, encoder: str, worker_mode: WorkerMode, user_id: Optional[str]
    ) -> None:
        store = self._store
        host = self._host

        self._tag = store.db_id() or store.hmac(b"verbatim-store-tag").hex()[:16]
        self._alias_label = user_id if user_id is not None else _DEFAULT_ALIAS
        info = ensure_bootstrap(
            store,
            owner=self._owner,
            owner_kind="human",
            profile_id=self._profile_id,
            host_name=host.host_name(),
            alias_label=self._alias_label,
        )
        self._namespace = info.namespace
        self._created = info.created

        # The governed read surface — bound to the same store, same verbs.
        self._v3 = VerbatimV3(
            store,
            self._cfg,
            bound_verbs=ROOT_VERBS,
            host_id=host.host_name(),
            adapter_version=_PRODUCER,
            principal_kind="human",
        )
        self._engine = ReadinessEngine(store)
        self._jobs = JobQueue(store)
        # Inspect/forget delegate to the bound-caller controls surface —
        # one inspect/closure implementation, never a parallel path.
        self._controls = MemoryControls(
            store,
            caller=CallerV3(
                principal_id=self._owner,
                session_id=self._session_id,
                host_id=host.host_name(),
            ),
            namespace=self._namespace,
            store_tag=self._tag,
        )

        # Encoder: hashing is the stdlib-only deterministic path; 'none' is
        # an explicit capability downgrade, reported on status/search.
        self._encoder = None
        self._encoder_id = "none"
        if encoder == "hashing":
            if HashingEncoder is None:
                self._warnings.append("encoder_unavailable")
            else:
                try:
                    self._encoder = HashingEncoder(self._cfg)
                    self._encoder_id = self._encoder.encoder_id
                except Exception:
                    self._encoder = None
                    self._warnings.append("encoder_unavailable")
        elif encoder == "artifact":
            # Pinned local artifact (V6-03.06–10): available() re-verifies
            # the manifest each call; unverified artifacts degrade to an
            # honest encoder_unavailable, never a silent hashing pretend.
            try:
                from ..embeddings.artifact import ArtifactEncoder
            except Exception:
                ArtifactEncoder = None  # type: ignore[assignment]
            if ArtifactEncoder is None:
                self._warnings.append("encoder_unavailable")
            else:
                try:
                    data_dir = getattr(self._cfg, "data_dir", None) or None
                    self._encoder = ArtifactEncoder(
                        self._cfg.embedding, data_dir=data_dir
                    )
                    if self._encoder.available():
                        # The id is claimed only for a VERIFIED artifact —
                        # an unverifiable pin must not read as the active
                        # encoder on status/capabilities (V6-03.06).
                        self._encoder_id = self._encoder.encoder_id
                    else:
                        self._encoder = None
                        self._warnings.append("encoder_unavailable")
                except Exception:
                    self._encoder = None
                    self._warnings.append("encoder_unavailable")

        # The source-projection jobs seam (SPEC_V5 §5): when the module is
        # absent in this build the facade performs the jobs layer's defer
        # decision at capture — obligations are declared, then deferred
        # with the honest reason (V5-08.15).
        self._source_jobs = None
        try:
            from ..jobs import source_jobs as _sj  # type: ignore

            fn = getattr(_sj, "enqueue_source_jobs", None)
            if callable(fn):
                self._source_jobs = fn
        except Exception:
            self._source_jobs = None
        if self._source_jobs is None:
            self._warnings.append("source_jobs_unprovisioned")
        if _source_candidates is None:
            self._warnings.append("source_lane_unavailable")
        if detect_update_candidates is None:
            self._warnings.append("update_detection_unavailable")

        if worker_mode is WorkerMode.MANAGED:
            from . import worker as _worker_mod

            self._worker_handle = _worker_mod.acquire(
                store,
                self._cfg,
                scope_ids=(self._namespace,),
                spawn_thread=_started_spawn(host.spawn_thread),
                encoder=self._encoder,
            )

        # Upgrade backfill (V5-07.13, §26 P1): an existing store that gained
        # v5 schema via migration carries sources with no projection rows.
        # One dedup-keyed plan per namespace; the cursor is durable, so a
        # crash or external-worker mode leaves honest partial coverage.
        self._maybe_backfill()

    def _maybe_backfill(self) -> None:
        if self._source_jobs is None:
            return
        try:
            from ..jobs.source_jobs import enqueue_source_backfill
            from ..ingest import Ingester
        except ImportError:
            return
        try:
            with self._store.tx() as conn:
                missing = conn.execute(
                    "SELECT s.source_id FROM sources s WHERE s.scope_id = ?"
                    " AND NOT EXISTS (SELECT 1 FROM source_lexical_projection p"
                    "  WHERE p.source_id = s.source_id)"
                    " ORDER BY s.source_id LIMIT 1",
                    (self._namespace,),
                ).fetchone()
                if missing is None:
                    return
                # job_key keyed by the first missing source: identical across
                # opens while the same gap persists (dedup converges), and a
                # later-arriving gap mints a fresh cursor once done=1.
                ing = Ingester(self._store, self._cfg, encoder=self._encoder)
                enqueue_source_backfill(
                    conn,
                    ing,
                    scope_id=self._namespace,
                    namespace=self._namespace,
                    job_key=f"source_backfill:{self._namespace}:{missing[0]}",
                )
        except VerbatimError:
            self._warnings.append("backfill_enqueue_failed")
        except Exception:
            self._warnings.append("backfill_enqueue_failed")

    # ---- guards -----------------------------------------------------------

    def _require_live(self) -> None:
        if self._closed or self._closing:
            raise closed()
        if os.getpid() != self._pid:
            forked()

    def _caller(self) -> CallerV3:
        return CallerV3(
            principal_id=self._owner,
            session_id=self._session_id,
            host_id=self._host.host_name(),
        )

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    def add(
        self,
        content: Any = None,
        *,
        infer: bool = True,
        metadata: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
        replaces: Optional[Any] = None,
        change: str = "supersede",
        effective_at: Optional[Any] = None,
        speaker: Optional[str] = None,
        occurred_at: Optional[Any] = None,
        messages: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> AddResult:
        """Evidence-preserving direct add; atomic with obligations (§06/§07).

        Turn capture (V7 D7-26): ``speaker``, ``session_id``,
        ``occurred_at`` (int µs / RFC3339 str / datetime → µs), and
        ``messages`` (≤512 strings or ``{speaker, text, at, ...}`` dicts)
        persist into the revision's ``metadata_json`` under the
        ``projections/units_v7`` add-args contract, so the units
        projection derives turn/session units carrying real speaker and
        time pins. ``session_id`` is the caller's conversational-session
        label for unit grouping — the envelope's own ``session_id`` field
        stays the facade's capture-session identity. With no turn fields
        supplied the write is byte-identical to the legacy shape.

        V85-02/03: ``content`` may be omitted when ``messages`` is
        supplied — the source payload is then synthesized as a
        ``speaker: text`` transcript, one line per message, so the
        projection emits one turn unit per message with in-source
        ordinals 0..n-1 (``_transcript_payload`` keeps every message
        text byte-locatable).  ``add()`` with neither raises the same
        typed ``invalid`` as before.
        """
        self._require_live()
        plan = self._add_plan(
            content,
            infer=infer,
            metadata=metadata,
            idempotency_key=idempotency_key,
            replaces=replaces,
            change=change,
            effective_at=effective_at,
            speaker=speaker,
            occurred_at=occurred_at,
            messages=messages,
            session_id=session_id,
        )
        with self._store.tx() as conn:
            governance.authorize(
                conn,
                self._caller(),
                self._namespace,
                Verb.INGEST.value,
                purpose="ingest",
            )
            result, fresh = self._add_capture_tx(conn, plan)
        if fresh:
            self._add_settled(result)
        return result

    def _add_plan(
        self,
        content: Any,
        *,
        infer: bool,
        metadata: Optional[dict],
        idempotency_key: Optional[str],
        replaces: Optional[Any],
        change: str,
        effective_at: Optional[Any],
        speaker: Optional[str],
        occurred_at: Optional[Any],
        messages: Optional[Any],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Validate every add argument and assemble the capture plan.

        Pure — no store I/O — so ``bulk_add`` runs it per item before any
        tx opens and a malformed item fails without touching the store.
        """
        if not isinstance(infer, bool):
            raise invalid("infer must be a bool")
        meta = _metadata(metadata)
        turn = _turn_meta(
            speaker=speaker,
            occurred_at=occurred_at,
            messages=messages,
            session_id=session_id,
        )
        if content is None:
            # V85-02: ``add(messages=[...])`` — synthesize the source
            # payload from the normalized message list so the projection
            # emits one pinned turn per message.
            payload = _transcript_payload(turn.get("messages"))
        else:
            payload = _payload_bytes(content)
        idem = (
            _key_ok(idempotency_key, "idempotency_key")
            if idempotency_key is not None
            else None
        )
        if not isinstance(change, str) or change not in ("supersede", "correct"):
            raise invalid("change must be 'supersede' or 'correct'")
        if replaces is None and change != "supersede":
            # A nondefault transition kind with no target is meaningless —
            # typed rejection, never a silent no-op (V5-06.15).
            raise invalid(
                "change applies only with replaces= (a transition target)"
            )

        # V5-07.03/06.18: caller-declared sensitive/zero-retention policy
        # is decided BEFORE the durable write — protected bytes persist as
        # evidence but never enter plaintext projections or receipts.
        protected = _check_retention_policy(meta)

        pred_ref: Optional[MemoryRef] = None
        if replaces is not None:
            pred_ref = self._bound_ref(replaces, what="replaces")
        elif effective_at is not None:
            raise invalid("effective_at requires replaces= (a transition target)")
        if effective_at is not None and change == "correct":
            raise invalid("effective_at applies only to change='supersede'")

        store = self._store
        ns = self._namespace
        if "messages" in turn:
            # Turn-aware dedup material (D7-26): the content-only
            # ``external_id`` collapses identical repeated turns — a second
            # "yes" in one conversation folds onto the first source and can
            # never mint its own turn unit. For turn-carrying adds the
            # material folds in the session identity and the indexed
            # message list (per-message ordinal + speaker + declared
            # times): distinct sessions/positions/times mint distinct
            # sources while a byte-identical retry still replays the
            # committed capture (V3-12.04 dedup preserved). Adds without
            # ``messages`` keep the exact legacy material.
            canon = json_dumps(
                {
                    "session": turn.get("session_id") or self._session_id,
                    "speaker": turn.get("speaker"),
                    "occurred": turn.get("occurred"),
                    "message_at": turn.get("message_at"),
                    "messages": [
                        (
                            dict(m, _i=i)
                            if isinstance(m, dict)
                            else {"text": m, "_i": i}
                        )
                        for i, m in enumerate(turn["messages"])
                    ],
                }
            ).encode("utf-8")
            external_id = "v5m:" + store.hmac(
                b"memory-add-turns|"
                + ns.encode()
                + b"|"
                + canon
                + b"|"
                + payload
            ).hex()[:32]
        else:
            external_id = "v5m:" + store.hmac(
                b"memory-add|" + ns.encode() + b"|" + payload
            ).hex()[:32]

        # Explicit turn params override same-named caller-metadata keys —
        # the same "explicit wins" precedence ``derive_units`` gives call
        # args over persisted metadata.
        merged = {**meta, **turn} if turn else meta
        # Idempotency semantic material: the legacy repr is byte-identical
        # when no turn fields exist; turn fields extend it so a same-key
        # retry with different turn arguments is a conflict, never a
        # silent re-execution.
        sem_extra = b"|" + json_dumps(turn).encode("utf-8") if turn else b""
        return {
            "payload": payload,
            "meta": meta,
            "merged_meta": merged,
            "idem": idem,
            "pred_ref": pred_ref,
            "replaces": replaces,
            "change": change,
            "effective_at": effective_at,
            "protected": protected,
            "external_id": external_id,
            "infer": infer,
            "sem_extra": sem_extra,
        }

    def _add_capture_tx(
        self, conn, plan: Dict[str, Any], *, savepoint: Optional[str] = None
    ) -> Tuple[AddResult, bool]:
        """The add's in-transaction capture body (~35 statements), shared by
        ``add`` (one tx per call) and ``bulk_add`` (one tx per chunk, a
        SAVEPOINT per item via ``_item_savepoint``).

        The caller owns the tx boundary AND the INGEST authorization —
        ``add`` authorizes inside its own tx; ``bulk_add`` authorizes once
        per call. Returns ``(result, fresh)`` where ``fresh=False`` marks
        an idempotency replay of a committed result — no new write, no
        obligations owed, no session-frontier update.

        NOTE: an advisory update-detection prescan (WAL snapshot ahead of
        the commit, like the source jobs use) was measured and reverted —
        the counter fingerprint almost never survives the write-lock wait
        under a draining worker, so adds paid the ~15ms scan twice
        (snapshot + fused in-tx fallback). The fused
        ``detect_update_candidates`` inside the commit is the cheaper
        honest path here.
        """
        store = self._store
        ns = self._namespace
        owner = self._owner
        payload = plan["payload"]
        meta = plan["meta"]
        merged_meta = plan["merged_meta"]
        idem = plan["idem"]
        replaces = plan["replaces"]
        pred_ref = plan["pred_ref"]
        change = plan["change"]
        effective_at = plan["effective_at"]
        protected = plan["protected"]
        infer = plan["infer"]
        external_id = plan["external_id"]
        sem_extra = plan["sem_extra"]

        replay_result: Optional[AddResult] = None
        result: Optional[AddResult] = None

        with self._item_savepoint(conn, savepoint):

            # --- durable idempotency (V5-07.09): same key + same semantics
            # replays the committed result; same key + different semantics
            # is a conflict, never a silent re-execution. ---
            idem_key = None
            semantic = store.hmac(
                b"memory-add-sem|"
                + payload
                + b"|"
                + repr(
                    (
                        infer,
                        meta,
                        str(replaces) if replaces is not None else None,
                        change,
                        repr(effective_at),
                    )
                ).encode("utf-8")
                + sem_extra
            ).hex()
            if idem is not None:
                idem_key = _IDEM_PREFIX + store.hmac(
                    f"{idem}|{ns}|{owner}".encode("utf-8")
                ).hex()
                prior = store._meta_get(conn, idem_key)
                if prior is not None:
                    if not isinstance(prior, dict) or prior.get("digest") != semantic:
                        raise conflict(
                            "idempotency_key was used with different arguments"
                        )
                    replay_result = _add_result_from(prior.get("result") or {})
                    replay_result.replayed = True

            if replay_result is None:
                # Content-level dedup pre-check: the envelope's external_id
                # is derived from (namespace, payload) so a byte-identical
                # retry replays the committed receipt instead of writing a
                # twin (V3-12.04 semantics ride the existing dedup path).
                prior_src = conn.execute(
                    "SELECT source_id FROM sources"
                    " WHERE scope_id = ? AND origin = ? AND external_id = ?",
                    (ns, self._host.host_name(), external_id),
                ).fetchone()

                envelope = SourceEnvelopeV3(
                    # V5-05.14: a standalone add is an operator-submitted
                    # source record — `operator` provenance (SYSTEM_EVENT →
                    # OPERATOR_RECORD), principal_reported trust, harvest-
                    # eligible. Claimed speaker/role metadata can never
                    # mint direct-human provenance (V5-05.08).
                    kind=EnvelopeKind.SYSTEM_EVENT,
                    scope_id=ns,
                    actor_principal=owner,
                    perspective=Perspective(asserter=owner, observer=owner),
                    event_us=self._host.now_us(),
                    receipt_us=0,
                    content=payload,
                    trust_class=TrustClass.PRINCIPAL_REPORTED,
                    adapter_version=_PRODUCER,
                    host_id=self._host.host_name(),
                    session_id=self._session_id,
                    external_id=external_id,
                    metadata={
                        **merged_meta,
                        "facade": _PRODUCER,
                        "profile": _PROFILE,
                        "direct_add": True,
                        "infer": infer,
                        # Durable protection marker — the projection plane
                        # (and backfill) must honor it on every later read
                        # of this envelope, not only this session.
                        **({"protected": True} if protected else {}),
                    },
                )
                receipt = ingest_envelope(conn, store, envelope)
                replayed = prior_src is not None and prior_src[0] == receipt.source_id
                if replayed and _dedup_links is not None:
                    # V5-30.05/30.09: the content-dedup replay IS the
                    # collapse event — a byte-identical submission folds
                    # onto the committed record, so no second member row
                    # exists for ``duplicate_links`` to group. Record the
                    # collapse durably inside this commit so delivered
                    # hits report ``collapsed_duplicates`` honestly
                    # (never corroboration — same submitter, same bytes).
                    try:
                        _dedup_links.record_duplicate_submission(
                            conn, receipt.source_id, int(receipt.revision)
                        )
                    except Exception:
                        self._warnings.append("duplicate_record_failed")

                if protected:
                    # The hold rides the existing quarantine authority
                    # (V3-34) inside the same fenced commit: every later
                    # drain, projection gate, backfill window, retrieval
                    # cascade, and quote path re-checks it, so declared-
                    # protected bytes persist as durable evidence yet can
                    # never enter a plaintext projection or quote
                    # (V5-07.03, V5-06.18). When the schema cannot record
                    # the hold the add fails typed — protection is never
                    # silently weakened.
                    if not has_table(conn, "quarantine"):
                        raise unavailable(
                            "protected retention requires the quarantine "
                            "schema (v3+)"
                        )
                    from ..security.quarantine import open_quarantine

                    codes: List[str] = []
                    ret = meta.get("retention")
                    if isinstance(ret, str) and (
                        ret.strip().lower() in _RETENTION_PROTECTED
                    ):
                        codes.append(f"policy:retention.{ret.strip().lower()}")
                    sen = meta.get("sensitivity")
                    if isinstance(sen, str) and (
                        sen.strip().lower() in _SENSITIVITY_PROTECTED
                    ):
                        codes.append(
                            f"policy:sensitivity.{sen.strip().lower()}"
                        )
                    if not codes:
                        codes = ["policy:protected_retention"]
                    findings = [
                        {
                            "rule": "facade.metadata",
                            "detail": "caller-declared retention/sensitivity",
                        }
                    ]
                    open_quarantine(
                        conn,
                        ("source", receipt.source_id, receipt.revision),
                        codes,
                        findings,
                        scope_id=ns,
                    )
                    open_quarantine(
                        conn,
                        (
                            "source_envelope",
                            receipt.envelope_id,
                            receipt.revision,
                        ),
                        codes,
                        findings,
                        scope_id=ns,
                    )

                # --- readiness obligations, declared atomically (V5-08.15)
                if self._engine.available:
                    if protected:
                        # Protected writes owe the DAG honestly: accepted +
                        # screened settle at capture, claim stages are
                        # excluded like a non-derivation kind, and the
                        # source-projection obligations are declared then
                        # deferred under the retention reason — owed work
                        # is recorded, never silently absent (V5-08.15).
                        self._engine.record_obligations(
                            conn,
                            receipt.receipt_id,
                            ns,
                            capabilities=(
                                CapabilityName.ACCEPTED,
                                CapabilityName.SCREENED,
                                CapabilityName.FAILED,
                            ),
                            pipeline=False,
                            include_source=True,
                        )
                        for cap in SOURCE_CAPS:
                            self._engine.defer(
                                conn,
                                receipt.receipt_id,
                                cap,
                                "protected_retention",
                            )
                    elif infer:
                        self._engine.record_obligations(
                            conn,
                            receipt.receipt_id,
                            ns,
                            pipeline=True,
                            include_source=True,
                        )
                    else:
                        self._engine.record_obligations(
                            conn,
                            receipt.receipt_id,
                            ns,
                            capabilities=(
                                CapabilityName.ACCEPTED,
                                CapabilityName.SCREENED,
                                CapabilityName.FAILED,
                            ),
                            pipeline=False,
                            include_source=True,
                        )
                    if not protected:
                        self._declare_source_obligations(conn, receipt.receipt_id)

                if not infer or protected:
                    # infer=False asks for source visibility only — the
                    # harvest→claim obligation the envelope path enqueues is
                    # cancelled inside the same tx (V5-08.16): the write is
                    # still owed source-search readiness, never stranded.
                    self._cancel_harvest(conn, receipt.source_id, receipt.revision)

                # --- v5 control artifact: source_state/v1 registration.
                state = None
                if has_table(conn, _STATE_TABLE):
                    state = _sstate.ensure_state(
                        conn,
                        receipt.source_id,
                        ns,
                        head=receipt.revision,
                        producer=_PRODUCER,
                        actor=owner,
                        operation_id=receipt.receipt_id,
                        store=store,
                    )
                else:
                    self._warnings.append("source_state_table_absent")

                # --- explicit replacement: the predecessor's control record
                # fences to the successor head — CAS-bound, atomic with the
                # capture (V5-06.11, V5-14.10). ---
                if pred_ref is not None and receipt.source_id != pred_ref.source_id:
                    if state is None:
                        raise unavailable(
                            "replacement requires source_state (schema v5)"
                        )
                    if not has_table(conn, _STATE_TABLE):
                        raise unavailable("replacement requires source_state")
                    _stransitions.transition(
                        conn,
                        pred_ref.source_id,
                        expected_control_version=pred_ref.control_version,
                        disposition=change,
                        superseded_by=f"{receipt.source_id}:{receipt.revision}",
                        effective_at=effective_at,
                        producer=_PRODUCER,
                        expected_revision=pred_ref.expected_revision,
                        epoch_vector={ns: governance.current_epoch(conn, ns)},
                        actor=owner,
                        operation_id=receipt.receipt_id,
                        store=store,
                    )

                candidates: List[UpdateCandidate] = []
                if (
                    not protected
                    and detect_update_candidates is not None
                    and NewRecord is not None
                ):
                    try:
                        new_record = NewRecord(
                            source_id=receipt.source_id,
                            revision=receipt.revision,
                            text=payload.decode("utf-8"),
                        )
                        detected = detect_update_candidates(
                            conn,
                            ns,
                            new_record,
                            store=store,
                        )
                        for d in detected:
                            # V6-03.03: namespaces opted into `auto_safe`
                            # apply guarded replacements through the same
                            # coordinator effect as `replaces=` — applied
                            # candidates never surface as advisory rows.
                            applied = False
                            if _auto_safe_replace is not None:
                                try:
                                    applied = bool(
                                        _auto_safe_replace(
                                            conn,
                                            namespace=ns,
                                            new_record=new_record,
                                            candidate=d,
                                            store=store,
                                        )
                                    )
                                except VerbatimError:
                                    applied = False
                            if applied:
                                continue
                            ref = MemoryRef(
                                store_tag=self._tag,
                                namespace=ns,
                                source_id=d.prior_source_id,
                                expected_revision=d.prior_revision,
                                control_version=d.prior_control_version,
                            )
                            candidates.append(
                                UpdateCandidate(
                                    ref=ref.to_string(),
                                    relation=d.relation,
                                    reason=d.reason,
                                    score=d.score,
                                )
                            )
                    except VerbatimError:
                        candidates = []
                        self._warnings.append("update_detection_unavailable")

                acceptance = (
                    Acceptance.PROTECTED.value
                    if protected
                    else self._acceptance(conn, receipt)
                )
                result = AddResult(
                    memory_id=receipt.source_id,
                    ref=MemoryRef(
                        store_tag=self._tag,
                        namespace=ns,
                        source_id=receipt.source_id,
                        expected_revision=receipt.revision,
                        control_version=(
                            state.control_version if state is not None else 0
                        ),
                    ).to_string(),
                    source_revision=receipt.revision,
                    receipt_id=receipt.receipt_id,
                    acceptance=acceptance,
                    replayed=replayed,
                    warnings=[],
                    possible_updates=candidates,
                    inference=(
                        "deferred"
                        if protected
                        else (
                            "queued"
                            if infer and self._worker_handle is not None
                            else ("not_requested" if not infer else "deferred")
                        )
                    ),
                )
                if idem_key is not None:
                    store._meta_set(
                        conn,
                        idem_key,
                        {
                            "v": 1,
                            "digest": semantic,
                            "result": _add_result_dict(result),
                            "created_us": now_us(),
                        },
                    )

                # session frontier — persisted inside the same tx so the
                # causal record survives process restart (V5-08.10).
                self._record_session_receipt(conn, receipt.receipt_id)

        result = result if result is not None else replay_result
        assert result is not None
        # fresh=False marks an idempotency replay — no write happened, so
        # the caller skips the settled bookkeeping (session frontier,
        # worker wake, readiness snapshot) exactly as before.
        return result, replay_result is None

    @contextlib.contextmanager
    def _item_savepoint(self, conn, name: Optional[str]):
        """Per-item rollback fence inside a shared tx (``bulk_add``).

        ``None`` → a bare yield (``add`` needs no fence — the tx itself is
        the item boundary). A name → ``SAVEPOINT``/``RELEASE`` wrapping
        one item's capture; an item failure issues ``ROLLBACK TO`` +
        ``RELEASE`` then re-raises, so a bad item undoes only its own
        writes and neighbours still commit. Savepoint names are fixed
        internal identifiers — never caller data (they cannot be bound
        parameters)."""
        if name is None:
            yield
            return
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            raise
        else:
            conn.execute(f"RELEASE SAVEPOINT {name}")

    def _add_settled(self, result: AddResult) -> None:
        """Post-commit bookkeeping for a fresh capture — session frontier,
        worker wake, readiness snapshot. Idempotency replays skip all
        three (nothing new committed)."""
        with self._lock:
            if result.receipt_id not in self._session_receipts:
                self._session_receipts.append(result.receipt_id)
                self._compact_session()
        if self._worker_handle is not None:
            self._worker_handle.wake()
        result.readiness = self._receipt_caps(result.receipt_id)

    def bulk_add(
        self,
        items: Iterable[Dict[str, Any]],
        *,
        infer: bool = True,
        chunk_size: int = _BULK_CHUNK,
    ) -> List[AddResult]:
        """Batch form of ``add`` (V7 D7-26) — one result per item, in
        input order.

        Each item is a dict of ``add`` arguments: ``content`` plus
        optional ``metadata``, ``idempotency_key``, ``replaces``,
        ``change``, ``effective_at``, ``speaker``, ``occurred_at``,
        ``messages``, ``session_id``, and ``infer`` (per-item override of
        the call-level default). Unknown keys are a per-item typed
        failure, never silently dropped.

        Amortization + atomicity: items commit ``chunk_size`` (default
        64) per ``store.tx()`` — the amortized part is the write-lock
        acquisition + commit/fsync, not the per-item capture semantics.
        Inside a chunk each item runs under its own ``SAVEPOINT``: a
        failed item rolls back only its own writes and is returned as
        ``acceptance="failed"`` with the typed reason on ``error`` while
        its neighbours commit. A chunk-level commit failure marks every
        not-already-failed item in that chunk failed — a rolled-back
        write never reports success. ``governance.authorize`` runs ONCE
        per bulk call (INGEST on this facade's bound namespace, inside
        the first chunk's tx); an authorization denial condemns the whole
        call and propagates exactly as ``add`` raises — call-scoped
        failures are never flattened into per-item results. Validation is
        per-item up front (``_add_plan`` is pure), so a malformed item
        fails without entering a tx at all.
        """
        self._require_live()
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or not 1 <= chunk_size <= _MAX_BULK_CHUNK
        ):
            raise invalid(f"chunk_size must be an int in [1, {_MAX_BULK_CHUNK}]")
        if items is None or isinstance(items, (str, bytes, dict)):
            raise invalid("items must be an iterable of add-argument dicts")
        try:
            item_list = list(items)
        except TypeError:
            raise invalid("items must be an iterable of add-argument dicts")

        # --- phase 1: per-item validation, no store writes -------------
        plans: List[Optional[Dict[str, Any]]] = []
        results: List[Optional[AddResult]] = [None] * len(item_list)
        known = {
            "infer", "metadata", "idempotency_key", "replaces", "change",
            "effective_at", "speaker", "occurred_at", "messages", "session_id",
        }
        for i, item in enumerate(item_list):
            try:
                if not isinstance(item, dict):
                    raise invalid(f"items[{i}] must be a dict of add arguments")
                kwargs = dict(item)
                content = kwargs.pop("content", None)
                unknown = sorted(k for k in kwargs if k not in known)
                if unknown:
                    raise invalid(
                        f"items[{i}] has unknown add arguments: {unknown}"
                    )
                plans.append(
                    self._add_plan(
                        content,
                        infer=kwargs.pop("infer", infer),
                        metadata=kwargs.pop("metadata", None),
                        idempotency_key=kwargs.pop("idempotency_key", None),
                        replaces=kwargs.pop("replaces", None),
                        change=kwargs.pop("change", "supersede"),
                        effective_at=kwargs.pop("effective_at", None),
                        speaker=kwargs.pop("speaker", None),
                        occurred_at=kwargs.pop("occurred_at", None),
                        messages=kwargs.pop("messages", None),
                        session_id=kwargs.pop("session_id", None),
                    )
                )
            except VerbatimError as exc:
                results[i] = _failed_add_result(exc)
                plans.append(None)

        # --- phase 2: chunked commit, SAVEPOINT per item ---------------
        pending = [i for i, plan in enumerate(plans) if plan is not None]
        fresh: List[AddResult] = []
        authorized = False
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            try:
                with self._store.tx() as conn:
                    if not authorized:
                        governance.authorize(
                            conn,
                            self._caller(),
                            self._namespace,
                            Verb.INGEST.value,
                            purpose="ingest",
                        )
                        authorized = True
                    for i in chunk:
                        try:
                            res, is_fresh = self._add_capture_tx(
                                conn, plans[i], savepoint="v5_bulk_item"
                            )
                        except Exception as exc:
                            # The savepoint already rolled the item's own
                            # writes back (or the tx aborted — the outer
                            # handler marks the whole chunk then).
                            results[i] = _failed_add_result(exc)
                        else:
                            results[i] = res
                            if is_fresh:
                                fresh.append(res)
            except VerbatimError as exc:
                if exc.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                    raise
                # The chunk's commit failed — every not-already-failed
                # item in it rolled back; report that, never a phantom
                # success. Items that already carry a per-item error keep
                # the truer cause.
                for i in chunk:
                    if results[i] is None or results[i].error is None:
                        results[i] = _failed_add_result(exc)
            except Exception as exc:
                for i in chunk:
                    if results[i] is None or results[i].error is None:
                        results[i] = _failed_add_result(exc)

        # --- phase 3: settled bookkeeping, once per call ---------------
        if fresh:
            # A chunk-commit failure replaces its items' results with
            # failure records — dropped objects must not enter the
            # session frontier (their receipts never committed).
            live = {id(r) for r in results if r is not None}
            fresh = [r for r in fresh if id(r) in live]
        if fresh:
            with self._lock:
                for res in fresh:
                    if res.receipt_id not in self._session_receipts:
                        self._session_receipts.append(res.receipt_id)
                self._compact_session()
            if self._worker_handle is not None:
                self._worker_handle.wake()
            for res in fresh:
                res.readiness = self._receipt_caps(res.receipt_id)
        return [
            r
            if r is not None
            else _failed_add_result(
                VerbatimError(ErrorCode.VALIDATION, "item was not attempted")
            )
            for r in results
        ]

    def _declare_source_obligations(self, conn, receipt_id: str) -> None:
        """Declare-or-defer the v5 source branch (V5-08.15).

        When the source-jobs module is provisioned it enqueues real work
        and the obligations stay ``pending``; otherwise the jobs layer's
        honest decision — ``deferred: source_jobs_unprovisioned`` — is
        recorded here, durably, never silently absent.
        """
        if self._source_jobs is not None:
            try:
                self._source_jobs(conn, self._store, receipt_id=receipt_id)
                return
            except TypeError:
                # Signature drift on a provisional seam — fall through to
                # the honest deferral rather than guessing argument shapes.
                pass
            except VerbatimError:
                pass
        for cap in SOURCE_CAPS:
            try:
                self._engine.defer(
                    conn, receipt_id, cap, "source_jobs_unprovisioned"
                )
            except VerbatimError:
                pass

    def _cancel_harvest(self, conn, source_id: str, revision: int) -> None:
        """Cancel the envelope path's harvest job for infer=False adds."""
        try:
            dedup = self._store.hmac(
                f"harvest:{source_id}:{revision}".encode()
            )
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE dedup_key = ? AND state NOT IN"
                " ('succeeded','failed','cancelled')",
                (dedup,),
            ).fetchone()
            if row is not None:
                self._jobs.cancel(conn, row[0])
        except VerbatimError:
            pass

    def _acceptance(self, conn, receipt) -> str:
        """accepted | held — quarantine-held evidence reports honestly."""
        try:
            row = conn.execute(
                "SELECT state FROM quarantine WHERE object_kind = 'source_envelope'"
                " AND object_id = ? AND revision = ?",
                (receipt.envelope_id, receipt.revision),
            ).fetchone()
            if row is not None and row[0] in ("pending", "suppressed"):
                return "held"
        except Exception:
            pass
        return "accepted"

    def _receipt_caps(self, receipt_id: str) -> Dict[str, str]:
        if not self._engine.available:
            return {}
        try:
            snap = self._engine.receipt_state(receipt_id, scope_id=self._namespace)
        except VerbatimError:
            return {}
        return {
            cap: str(entry.get("state", "unknown"))
            for cap, entry in (snap.get("states") or {}).items()
        }

    def _record_session_receipt(self, conn, receipt_id: str) -> None:
        key = f"{_SESSION_PREFIX}{self._session_id}"
        rec = self._store._meta_get(conn, key)
        if not isinstance(rec, dict):
            rec = {
                "v": 1,
                "owner": self._owner,
                "namespace": self._namespace,
                "receipts": [],
            }
        receipts = [r for r in rec.get("receipts") or [] if isinstance(r, str)]
        if receipt_id not in receipts:
            receipts.append(receipt_id)
        rec["receipts"] = receipts[-_SESSION_RECEIPT_MAX * 2 :]
        rec["updated_us"] = now_us()
        self._store._meta_set(conn, key, rec)

    def _compact_session(self) -> None:
        """Bound the in-memory frontier: settled receipts compact away."""
        if len(self._session_receipts) <= _SESSION_RECEIPT_MAX:
            return
        if not self._engine.available:
            self._session_receipts = self._session_receipts[-_SESSION_RECEIPT_MAX:]
            return
        keep: List[str] = []
        try:
            # One chunked ``SELECT DISTINCT`` finds the receipts still
            # holding pending/running rows — per-rid ``receipt_state``
            # calls made each >_SESSION_RECEIPT_MAX add an O(session)
            # scan (the measured add-ack tail). Foreign-scope and
            # row-less receipts never qualify — the same drop the old
            # per-rid except/absent paths produced.
            pending = self._engine.pending_receipt_ids(
                self._session_receipts[-_SESSION_RECEIPT_MAX * 2 :],
                scope_id=self._namespace,
            )
        except VerbatimError:
            pending = set()
        keep = [
            rid
            for rid in self._session_receipts[
                -_SESSION_RECEIPT_MAX * 2 :
            ]
            if rid in pending
        ]
        if len(keep) > _SESSION_RECEIPT_MAX:
            keep = keep[-_SESSION_RECEIPT_MAX:]
        # Never compact away receipts that have no snapshot yet.
        if not keep:
            keep = self._session_receipts[-_SESSION_RECEIPT_MAX:]
        self._session_receipts = keep

    # ------------------------------------------------------------------
    # wait_ready
    # ------------------------------------------------------------------

    def wait_ready(
        self,
        receipt: Any,
        *,
        capabilities: Optional[Iterable[str]] = None,
        timeout_ms: float = 2000,
    ) -> Readiness:
        """Bounded readiness wait; honest pending/blocked at the deadline."""
        self._require_live()
        rid = self._receipt_id_of(receipt)
        timeout = _nonneg_ms(timeout_ms, "timeout_ms")
        caps: Optional[List[str]] = None
        if capabilities is not None:
            caps = [str(c) for c in capabilities]
            if not caps:
                raise invalid("capabilities must be non-empty when given")
        else:
            caps = [CAP_SOURCE_LEXICAL]

        started = monotonic_ms()
        if not self._engine.available:
            return Readiness(
                receipt_id=rid,
                state=SearchStatus.UNAVAILABLE.value,
                capabilities={c: "unavailable" for c in caps},
                causal_satisfied=False,
                waited_ms=monotonic_ms() - started,
            )
        snap = self._engine.wait_ready(
            rid,
            capabilities=caps,
            deadline_us=now_us() + int(timeout * 1000),
            scope_id=self._namespace,
        )
        waited = monotonic_ms() - started
        states = {
            cap: str(entry.get("state", "unknown"))
            for cap, entry in (snap.get("states") or {}).items()
        }
        if snap.get("failed"):
            state = SearchStatus.BLOCKED.value
        elif snap.get("pending"):
            state = SearchStatus.PENDING.value
        elif snap.get("deferred"):
            # Owed but unprovisioned in this build — not "ready" claims.
            state = SearchStatus.PARTIAL.value
        else:
            state = SearchStatus.READY.value
        return Readiness(
            receipt_id=rid,
            state=state,
            capabilities=states,
            causal_satisfied=not snap.get("pending") and not snap.get("failed"),
            waited_ms=waited,
        )

    def _receipt_id_of(self, receipt: Any) -> str:
        """Receipt references: AddResult/Readiness/dict/receipt string."""
        if isinstance(receipt, str):
            if receipt.startswith("mref1."):
                raise invalid(
                    "wait_ready takes a receipt/AddResult, not a MemoryRef"
                )
            return require_id(receipt, "receipt_id")
        if isinstance(receipt, dict):
            rid = receipt.get("receipt_id")
            if isinstance(rid, str) and rid:
                return rid
            raise invalid("dict receipt references need a receipt_id")
        rid = getattr(receipt, "receipt_id", None)
        if isinstance(rid, str) and rid:
            return rid
        raise invalid(
            "receipt must be an AddResult, Readiness, dict with receipt_id,"
            " or a receipt id string"
        )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Any,
        *,
        limit: int = 8,
        filters: Optional[dict] = None,
        after: Optional[Any] = None,
        consistency: str = "session",
        ready_timeout_ms: Optional[float] = None,
        timeout_ms: float = 500,
        strict: bool = False,
        retrieval: str = "auto",
        as_of: Optional[Any] = None,
    ) -> SearchResult:
        """Governed search → ``SearchResult`` (never a bare list).

        ``retrieval`` selects the engine: ``"auto"`` (default) runs V7's
        peer-lane pipeline when this store carries V7 projections, else
        the V6 path; ``"v7"`` forces V7 (its source lane covers stores
        whose units were never projected); ``"v6"`` forces the legacy
        governed path. Engine selection is reported in coverage.engine.

        ``as_of`` (V8-09.01/V8-20.01) is the caller's question-time
        anchor for temporal queries: an aware ``datetime``, an RFC3339
        string with an explicit offset, or int µs. It beats the wall
        clock for that call only (never stored as process state) and is
        reported in ``coverage.temporal.anchor``. Invalid or ambiguous
        (naive) values raise ``VALIDATION`` naming ``as_of``.
        """
        self._require_live()
        if not isinstance(query, str):
            raise invalid("query must be a string")
        query = query.strip()
        if not query:
            raise invalid("query must not be empty")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise invalid("limit must be a positive int")
        limit = min(limit, _MAX_LIMIT)
        try:
            cons = Consistency(consistency)
        except ValueError:
            raise invalid("consistency must be 'session' or 'eventual'")
        if not isinstance(strict, bool):
            raise invalid("strict must be a bool")
        if retrieval not in ("auto", "v7", "v6"):
            raise invalid("retrieval must be 'auto', 'v7', or 'v6'")
        # V8-09.01/20.05 — validate the caller anchor at admission so a
        # bad ``as_of`` always raises (barrier, cache, and engine paths
        # can never observe it); ``None`` means wall-clock anchoring.
        anchor_us: Optional[int] = (
            _as_of_to_us(as_of) if as_of is not None else None
        )
        anchor_eff: Optional[int] = anchor_us
        timeout = _nonneg_ms(timeout_ms, "timeout_ms")
        ready_to = (
            self._ready_timeout_ms
            if ready_timeout_ms is None
            else _nonneg_ms(ready_timeout_ms, "ready_timeout_ms")
        )
        flt = self._filters(filters)

        barrier_receipts: List[str] = []
        warnings: List[str] = []
        if after is not None:
            barrier_receipts.extend(self._resolve_after(after, strict, warnings))
            if any(w == "causal_token_expired" for w in warnings):
                out = SearchResult(
                    status=SearchStatus.UNAVAILABLE.value,
                    warnings=warnings,
                    readiness={"causal_satisfied": False},
                    coverage={},
                )
                out.answerability = None  # V8-20.02 — no verdict ran
                return out
        if cons is Consistency.SESSION:
            with self._lock:
                barrier_receipts.extend(self._session_receipts)

        # The causal token is content-addressed over the resolved
        # frontier — mint it once up front. Its one-time meta write then
        # precedes the result-cache probe fingerprint below (a mint that
        # landed mid-pipeline would invalidate the freshly stored entry
        # once per frontier), and the cache-hit path reuses the same
        # token for its own exposure emission.
        token = self._mint_causal_token(barrier_receipts)

        started = monotonic_ms()
        deadline_us = now_us() + int(min(ready_to, timeout or ready_to) * 1000)

        # ---- causal barrier (V5-08): bounded, per-receipt, never global ----
        barrier_state = "met"
        unresolved: List[str] = []
        pending_receipts: List[str] = []
        deferred_caps: List[str] = []
        if barrier_receipts and self._engine.available:
            snaps = self._engine.wait_ready_many(
                barrier_receipts,
                capabilities=[CAP_SOURCE_LEXICAL],
                deadline_us=deadline_us,
                scope_id=self._namespace,
                # The barrier loop below only inspects snapshots
                # (``snap.get`` reads) — the shared read-only return
                # skips 512 defensive copies per search. Snapshots must
                # never be mutated here (``_shared`` contract).
                _shared=True,
            )
            for rid in dict.fromkeys(barrier_receipts):
                snap = snaps.get(rid)
                if snap is None or snap.get("absent"):
                    # Foreign/unknown receipts are indistinguishable.
                    continue
                if snap.get("failed"):
                    barrier_state = "failed"
                    unresolved.append(rid)
                    continue
                if snap.get("pending"):
                    if barrier_state == "met":
                        barrier_state = "pending"
                    unresolved.append(rid)
                    pending_receipts.append(rid)
                deferred_caps.extend(snap.get("deferred") or [])
        elif barrier_receipts and not self._engine.available:
            barrier_state = "unavailable"
            warnings.append("readiness_unavailable")
        if deferred_caps:
            warnings.append("source_projection_deferred")

        waited_ms = monotonic_ms() - started

        # V6-02.08 + docs/v6_contracts §2: advisory barrier→source marking.
        # Sources behind *pending* barrier receipts are noted for the
        # drain pass's unblock-first read; the mark never changes the
        # barrier verdict, and a marking failure is swallowed — advisory
        # only, never a search failure.
        if pending_receipts and _commit_notify is not None:
            try:
                _path = getattr(self._store, "_path", None)
                if _path and _path != ":memory:":
                    _src_ids = self._barrier_source_ids(pending_receipts)
                    if _src_ids:
                        _commit_notify.note_barrier_sources(
                            _path, _src_ids
                        )
            except Exception:
                pass

        if strict and barrier_state in ("pending", "failed"):
            raise deadline(
                f"causal barrier not met within budget ({barrier_state})"
            )
        if strict and barrier_state == "unavailable":
            raise unavailable("readiness subsystem is not provisioned")

        # ---- consumer result cache (V6-02.13 + docs/v6_contracts §7.5):
        # consulted AFTER the barrier — a cached answer can never bypass
        # the causal gate, and only a fully met barrier serves one
        # (pending/blocked/failed/unavailable and deferred-capability
        # results are computed live, never cached). probe_result()
        # revalidates the fingerprint — barrier/projection generation,
        # scope epochs, policy/erasure epochs, watermarks — plus every
        # delivered item against the current snapshot before a stored
        # byte ships.
        _result_cache = (
            _rcache.configured(self._store, self._cfg)
            if _rcache is not None
            else None
        )
        _cprobe = None
        if _result_cache is not None and (
            barrier_state == "met" and not deferred_caps
        ):
            try:
                # V8-09.01: a caller ``as_of`` changes what a temporal
                # query resolves to — it is request material, so it joins
                # the key (wall-clock anchoring keeps the V7 contract).
                _ckey = _rcache.consumer_request_key(
                    query=query,
                    limit=limit,
                    filters=(
                        flt
                        if anchor_us is None
                        else {**(flt or {}), "_as_of_us": anchor_us}
                    ),
                    consistency=cons.value,
                    namespace=self._namespace,
                    caller_id=self._owner,
                    encoder_id=self._encoder_id,
                    frontier=barrier_receipts,
                )
                with self._store.read() as conn:
                    _cprobe = _result_cache.probe_result(
                        conn,
                        self._store,
                        key=_ckey,
                        scope_ids=self._pack_cache_scope_ids(conn),
                        generation=self._pack_cache_generation(conn),
                    )
            except Exception:
                _cprobe = None
            if _cprobe is not None and _cprobe.entry is not None:
                result = self._pack_cache_result(
                    _cprobe.entry,
                    receipts=barrier_receipts,
                    waited_ms=waited_ms,
                    unresolved=unresolved,
                    deferred=deferred_caps,
                    cons=cons,
                )
                result.causal_token = token
                # V8-20.02 — a replayed answer carries whatever
                # answerability the stored verdict block recorded.
                result.answerability = _dig(
                    result.coverage, "verdict", "answerability"
                )
                # Fresh delivery rows on a hit — mirroring the v3 path's
                # fresh-handle discipline (V4-33.02): a replayed answer
                # is a real exposure, never folded into the original.
                self._emit_source_exposure(result, token)
                return result

        # ---- governed retrieval: derived pipeline + declared evidence lane
        # (V4-08.07 — raw sources ship only through the governed lane, which
        # applies read/quote authorization, purge suppression, and the
        # quarantine cascade) plus the v5 source lane when provisioned. ----
        hits: List[Hit] = []
        lanes: Dict[str, Any] = {}
        coverage: Dict[str, Any] = {}
        omitted = 0
        retrieval_degraded = False
        remaining_ms = max(0.0, timeout - (monotonic_ms() - started))
        # query_analysis/v1 — pure, no I/O; the verdict needs it even
        # when the source lane below is unprovisioned.  The analysis
        # object itself feeds the typed lane (identifier/entity/temporal
        # markers) so both consumers share one classification.
        analysis = (
            _analyze_query(query) if _analyze_query is not None else None
        )
        terms, idents, ents, qclass = self._query_terms(query, analysis)

        # ---- retrieval engine dispatch (V7-05.01: lanes as peers) ----
        # ``auto`` runs the V7 peer-lane pipeline when this store carries
        # the V7 projection plane; explicit ``v7`` always runs it (its
        # source lane covers unprojected stores); ``v6`` forces the
        # legacy governed path.  The shared tail (merge/verdict/status/
        # exposure/cache) applies to whichever engine produced hits.
        engine = retrieval
        v7_result = None
        v7_status_hint: Optional[str] = None
        lane_encoder: Optional[str] = None
        if engine in ("auto", "v7"):
            _rem = max(0.0, timeout - (monotonic_ms() - started))
            # V8-09.01/09.02 — one temporal anchor per call: the caller's
            # ``as_of`` wins; absent it the wall clock is read exactly
            # once and that value drives S1 resolution + every lane.
            if anchor_eff is None:
                anchor_eff = now_us()
            v7_result = self._search_v7(
                query,
                limit=_MAX_LIMIT,
                deadline_ms=_rem,
                strict=strict,
                require_units=(engine == "auto"),
                anchor_us=anchor_eff,
            )
            if v7_result is not None:
                engine = "v7"
                hits.extend(self._v7_hits(v7_result, limit))
                for _ln, _lo in (
                    getattr(v7_result, "lanes", None) or {}
                ).items():
                    lanes[f"v7.{_ln}"] = getattr(
                        getattr(_lo, "status", None), "value", None
                    ) or str(getattr(_lo, "status", "ok"))
                _v = getattr(v7_result, "verdict", None)
                v7_status_hint = getattr(_v, "value", _v)
                _miss = getattr(v7_result, "missing", None)
                if _miss is not None:
                    coverage["missing"] = {
                        "kind": getattr(_miss, "kind", None),
                        "detail": getattr(_miss, "detail", None),
                    }
                # V8-20.03 — the pipeline grafts the verdict report's
                # detail (status_trigger/triggers/answerability/
                # premise_speaker/deadline_cut_lanes) onto
                # ``PipelineResult.coverage.verdict``; surface it on the
                # public coverage so a cached replay keeps answerability
                # and callers can read the verdict block directly.
                _vdet = _dig(
                    getattr(v7_result, "coverage", None), "verdict"
                )
                if isinstance(_vdet, dict) and _vdet:
                    coverage["verdict"] = dict(_vdet)
                # V85-03.03 — context propagation/injection stats ride
                # ``coverage.context`` (boosted/injected/window counts)
                # so callers can see the stage actually engaged.
                _ctx_stats = _dig(
                    getattr(v7_result, "coverage", None), "context"
                )
                if isinstance(_ctx_stats, dict) and _ctx_stats:
                    coverage["context"] = dict(_ctx_stats)
                if getattr(v7_result, "explain", None):
                    coverage["explain"] = v7_result.explain
                coverage["engine"] = "v7"
                coverage["formula_status"] = getattr(
                    getattr(v7_result, "pack", None),
                    "formula_status",
                    None,
                )
            else:
                # auto falls back silently to the v6 path; explicit v7
                # reports the miss and still covers via v6 (a store
                # without V7 tables is honest unavailability, not a
                # fabricated lane).
                if engine == "v7":
                    lanes["v7"] = "unavailable"
                    warnings.append("retrieval_v7_unavailable")
                engine = "v6"
        if engine == "v6":

            recall = None
            try:
                recall = self._v3.recall(
                    self._namespace,
                    query,
                    principal_id=self._owner,
                    purpose="recall",
                    budget={
                        "modes": ("evidence",),
                        "max_items": 32,  # RecallRequestV3 caps at 32
                        "max_bytes": 24_000,  # contract bound 512..24000
                        "target_tokens": 6_000,
                        "deadline_ms": int(remaining_ms) or 1,
                        "session_id": self._session_id,
                    },
                    session_id=self._session_id,
                )
            except VerbatimError as exc:
                if exc.code == ErrorCode.CAPABILITY_UNAVAILABLE:
                    retrieval_degraded = True
                    lanes["governed_recall"] = "unavailable"
                    warnings.append("retrieval_v3_unavailable")
                    if strict:
                        raise
                else:
                    raise
            if recall is not None:
                warnings.extend(str(w) for w in recall.warnings)
                caps = recall.capabilities or {}
                lanes.update(caps.get("lanes") or {})
                if caps.get("degraded"):
                    retrieval_degraded = True
                omitted += int(recall.omitted or 0)
                hits.extend(self._hits_from_packs(recall.packs))

            # ---- v6 typed lane: grounded-fact index FIRST (V6-02.01/02) ----
            # Typed candidates are additive ranking input — the same
            # ``ranking/v1`` fusion below weighs them; the lane never admits.
            # The source lane stays as coverage fallback: it runs whenever the
            # typed index cannot provision (pre-v5 store), is partial or
            # truncated, leaves eligible records uncovered, or answers nothing
            # (identifier miss included) — the honest coverage contract of
            # docs/v6_contracts §7.3.
            lane_encoder: Optional[str] = None  # pinned by the vector sub-lane
            lane_hits: List[Any] = []
            typed_pool: List[Any] = []
            typed_map: Dict[Tuple[str, int], Any] = {}
            run_source = True
            if _typed_candidates is not None and _fuse is not None:
                try:
                    remaining_ms = max(0.0, timeout - (monotonic_ms() - started))
                    with self._store.read() as conn:
                        typed_hits, typed_stats = _typed_candidates(
                            conn,
                            namespace=self._namespace,
                            query=query,
                            analysis=analysis,
                            generation=None,
                            limit=_MAX_LIMIT,
                            store=self._store,
                            deadline_ms=int(remaining_ms) or 1,
                        )
                    lanes["typed"] = (
                        getattr(typed_stats, "status", None) or "ok"
                    )
                    warnings.extend(
                        str(w)
                        for w in (getattr(typed_stats, "warnings", None) or [])
                    )
                    if lanes["typed"] == "unavailable":
                        warnings.append("typed_lane_unavailable")
                    typed_pool = list(typed_hits or ())
                    for h in typed_pool:
                        typed_map[(h.source_id, int(h.revision))] = h
                    t_details = getattr(typed_stats, "details", None) or {}
                    uncovered = int(t_details.get("uncovered") or 0)
                    thin = (
                        lanes["typed"] != "ok"
                        or getattr(typed_stats, "truncated", False)
                        or getattr(typed_stats, "deadline_exceeded", False)
                        or uncovered > 0
                        or not typed_pool
                    )
                    run_source = thin
                except VerbatimError:
                    lanes["typed"] = "unavailable"
                    warnings.append("typed_lane_unavailable")
                    if strict:
                        raise
                except Exception:
                    lanes["typed"] = "unavailable"
                    warnings.append("typed_lane_unavailable")
            else:
                # Unprovisioned lane (partial checkout / pre-V5 store) — the
                # source path covers exactly as it did before V6, so this is
                # honest reporting, never a fabricated capability.
                lanes["typed"] = "unavailable"
                warnings.append("typed_lane_unavailable")

            # ---- v5 source lane — coverage fallback over projections -------
            if run_source:
                if _source_candidates is not None and _fuse is not None:
                    try:
                        remaining_ms = max(
                            0.0, timeout - (monotonic_ms() - started)
                        )
                        with self._store.read() as conn:
                            qvec = None
                            if self._encoder is not None:
                                try:
                                    qvec = self._encoder.encode([query])[0]
                                except Exception:
                                    qvec = None
                            lane_hits, lane_stats = _source_candidates(
                                self._store,
                                conn,
                                query_terms=terms,
                                identifiers=idents,
                                entities=ents,
                                query_vector=qvec,
                                namespace=self._namespace,
                                limit=_MAX_LIMIT,
                                deadline_ms=int(remaining_ms) or 1,
                            )
                        lanes["source"] = "ok"
                        # The encoder identity the vector sub-lane actually
                        # pinned for this scan — it selects that encoder's
                        # calibrated similarity-support floor in the verdict
                        # (V5-31.07/31.08).
                        vec_stats = (
                            getattr(lane_stats, "details", None) or {}
                        ).get("vector") or {}
                        if vec_stats.get("encoder"):
                            lane_encoder = str(vec_stats["encoder"])
                        stats = getattr(lane_stats, "stats", None) or {}
                        for name, st in (
                            stats.items() if isinstance(stats, dict) else []
                        ):
                            if isinstance(st, dict) and st.get("status") not in (
                                None, "ok"
                            ):
                                lanes[f"source.{name}"] = st.get("status")
                                retrieval_degraded = True
                        lane_warnings = getattr(lane_stats, "warnings", None) or []
                        warnings.extend(str(w) for w in lane_warnings)
                    except VerbatimError:
                        lanes["source"] = "unavailable"
                        retrieval_degraded = True
                        warnings.append("source_lane_unavailable")
                        if strict:
                            raise
                    except Exception:
                        lanes["source"] = "unavailable"
                        retrieval_degraded = True
                        warnings.append("source_lane_unavailable")
                else:
                    lanes["source"] = "unavailable"
                    retrieval_degraded = True
                    warnings.append("source_lane_unavailable")
            elif "source" not in lanes:
                lanes["source"] = "skipped"

            # ---- ranking/v1 fusion over the union of lane candidates -------
            # Both lanes feed one fused ranking (V5-31.05): a record surfaced
            # by both merges signals per-key (typed carrier keeps the verified
            # span pins that make the delivered line a fact, V6-02.02).
            merged_candidates = self._merge_lane_candidates(typed_pool, lane_hits)
            if merged_candidates and _fuse is not None:
                try:
                    remaining_ms = max(0.0, timeout - (monotonic_ms() - started))
                    fused = _fuse(
                        merged_candidates,
                        query_class=qclass,
                        limit=_MAX_LIMIT,
                        deadline=int(remaining_ms) or 1,
                    )
                    if getattr(fused, "stats", {}).get("partial"):
                        warnings.append("ranking_partial")
                        retrieval_degraded = True
                    hits.extend(
                        self._hits_from_source_lane(
                            list(fused), typed_map=typed_map
                        )
                    )
                except VerbatimError:
                    retrieval_degraded = True
                    warnings.append("ranking_unavailable")
                    if strict:
                        raise
                except Exception:
                    retrieval_degraded = True
                    warnings.append("ranking_unavailable")

        # ---- merge, filter, bound -----------------------------------------
        hits = self._dedup_hits(hits)
        hits = self._apply_filters(hits, flt)
        # support_verdict/v1 (V5-31.07/31.09): a candidate that carries no
        # support for the asked query never ships as an answer — the
        # honest ``insufficient`` verdict replaces the least-bad hit.
        # Skipped on the V7 engine: verdict_v2 already ran inside S7 and
        # its hits carry assessed support_status (``v7_status_hint``).
        verdict = (
            _search_verdict(
                hits, terms=terms, identifiers=idents,
                entities=ents, primary=qclass,
                encoder=lane_encoder or self._encoder_id or None,
            )
            if _search_verdict is not None and engine != "v7"
            else None
        )
        if verdict is not None:
            hits = list(verdict.kept)
            warnings.extend(verdict.warnings)
        # Duplicate collapse (V5-10.04/30.05/30.09): after every member
        # has been assessed on its own support, duplicate-linked sources
        # and replay-collapsed submissions report under one
        # representative carrying collapsed_duplicates + independent
        # corroboration; unlinked contrary records are never grouped.
        hits = self._collapse_duplicate_groups(hits)
        # Conflict-group labeling (V5-13.02/13.03): a source hit with an
        # open contradiction/update candidate against a still-active
        # record is a member of an unresolved group — it ships labeled
        # ``disputed``, never as a lone settled answer.
        self._label_conflict_groups(hits)
        for h in hits:
            # An unassessed hit may never ship silently (V5-13.03) —
            # when the support verdict could not run, the missing
            # assessment is itself a warning.
            if h.support_status == SupportStatus.UNASSESSED.value:
                h.warnings.append("support_unassessed")
        if len(hits) > limit:
            omitted += len(hits) - limit
            hits = hits[:limit]

        # ---- status precedence (V5-06.04) ---------------------------------
        if barrier_state == "unavailable":
            status = SearchStatus.UNAVAILABLE.value
        elif barrier_state == "failed":
            status = SearchStatus.BLOCKED.value
        elif barrier_state == "pending":
            status = SearchStatus.PENDING.value
        elif retrieval_degraded or deferred_caps:
            status = SearchStatus.PARTIAL.value
        elif engine == "v7" and v7_status_hint is not None:
            # The V7 structural verdict (S7, verdict_v2) is the status —
            # READY/INSUFFICIENT map directly; a degraded pipeline
            # (partial lanes) surfaces through coverage.lanes.
            status = (
                SupportStatus.INSUFFICIENT.value
                if v7_status_hint == "insufficient"
                else SearchStatus.READY.value
            )
        elif verdict is not None and verdict.verdict == SupportStatus.INSUFFICIENT.value:
            # Completed search, examined candidates, no support —
            # distinct from both "ready" and "not found" (V5-13.10).
            status = SupportStatus.INSUFFICIENT.value
        else:
            status = SearchStatus.READY.value

        # Route honesty (V6-02.01 + docs/v6_contracts §7): lanes that
        # actually produced candidates — a skipped or unavailable lane is
        # reported in ``coverage.lanes`` but is not part of the route.
        route = "consolidated+evidence"
        if engine == "v7":
            # The V7 pack IS the route — its peer lanes already fused.
            route = "v7"
            _producing = sorted(
                k
                for k, v in lanes.items()
                if k.startswith("v7.")
                and v not in (None, "unavailable", "skipped", "deadline")
            )
            if _producing:
                route += "(" + ",".join(_producing) + ")"
        else:
            if lanes.get("typed") not in (None, "unavailable", "skipped"):
                route += "+typed"
            if lanes.get("source") not in (None, "unavailable", "skipped"):
                route += "+source"
        coverage.update(
            {
                "lanes": lanes,
                "omitted": omitted,
                "route": route,
                "engine": engine,
            }
        )
        if engine == "v7" or anchor_us is not None:
            # V8-09.01/20.03 — the temporal anchor this search resolved
            # against: caller's ``as_of`` when given (it wins), else the
            # per-call wall clock. Reported even when the temporal lane
            # itself skipped; the v6 engine simply never consumed it.
            coverage.setdefault("temporal", {})["anchor"] = {
                "source": "caller" if anchor_us is not None else "wall",
                "us": anchor_eff,
            }
        if verdict is not None:
            coverage["support"] = verdict.detail
        # V6-02.10: consistency="eventual" skips the session barrier
        # ENTIRELY — the result must say so plainly: causal_satisfied
        # stays false and the result carries an explicit warning.  Only
        # a session barrier that actually ran and met earns True.
        if cons is Consistency.EVENTUAL:
            warnings.append("consistency_eventual")
        if _result_cache is not None:
            coverage["cache"] = {"enabled": True, "hit": False}
        result = SearchResult(
            status=status,
            items=hits,
            warnings=list(dict.fromkeys(warnings)),
            readiness={
                "causal_satisfied": (
                    barrier_state in ("met",)
                    and cons is Consistency.SESSION
                ),
                "waited_ms": waited_ms,
                "receipts": len(barrier_receipts),
                "unresolved": unresolved[:32],
                "deferred": sorted(set(deferred_caps)),
            },
            coverage=coverage,
            causal_token=token,
        )
        # V8-20.02 — verdict answerability rides beside status; ``None``
        # when the pipeline did not carry a report (never fabricated).
        result.answerability = _v7_answerability(v7_result)
        # V6 exposure emission (docs/v6_contracts §7.4/§8): one delivery
        # record per delivered item, in its own short write AFTER the
        # result is fully assembled — never inside the read tx.  Absent
        # sibling module → no-op; emit failure → warning, never a raise.
        self._emit_source_exposure(result, token)
        # A completed READY result with delivered items is the only
        # shape the consumer cache may record — pending/blocked/
        # partial/degraded/insufficient answers and empty hit lists
        # always recompute live (V6-02.13).
        self._pack_cache_store(_result_cache, _cprobe, result)
        return result

    # ---- search internals --------------------------------------------------

    def _query_terms(
        self, query: str, analysis: Any = None
    ) -> Tuple[List[str], List[str], List[str], Optional[str]]:
        if analysis is None and _analyze_query is not None:
            analysis = _analyze_query(query)
        if analysis is not None:
            return (
                list(getattr(analysis, "terms", ()) or ()),
                [v for _k, v in (getattr(analysis, "identifiers", ()) or ())],
                list(getattr(analysis, "entities", ()) or ()),
                getattr(analysis, "primary", None),
            )
        terms = [t for t in _TERM_RE.findall(query.lower()) if len(t) > 1]
        return terms, [], [], None

    # ---- V7 retrieval engine (§04.2 S0–S8, docs/v7_contracts.md) ----------

    def _v7_ready(self) -> bool:
        """True when this store carries the V7 projection plane."""
        if _v7_run_search is None or LaneContextV7 is None:
            return False
        try:
            with self._store.read() as conn:
                return has_table(conn, "units")
        except Exception:
            return False

    def _v7_query_view(self, query: str, now: int, conn: Any) -> Any:
        """S1 query analysis → QueryViewV7 (w-queryview lands the full
        builder; the fallback composes norm/intent directly so the V7
        path works before that module lands). ``now`` is the per-call
        temporal anchor — the caller's ``as_of`` or the wall clock
        (V8-09.01/09.02). The V8-09.03 bare-year guard runs here so every
        consumer — the temporal lane, verdict windows, facets — sees the
        corrected window."""
        qv = None
        if _v7_query_view is not None:
            qv = _v7_query_view(
                query,
                now_us=now,
                known_canons=lambda: self._v7_known_canons(conn),
            )
        elif LaneContextV7 is not None:
            from ..core.types_v7 import QueryViewV7
            from ..text.norm_v2 import analyze as _norm_analyze
            try:
                norm = _norm_analyze(query)
            except Exception:
                return None
            intent = None
            try:
                from ..querying.intent_v2 import classify as _intent
                idents = tuple(
                    getattr(t, "term", "")
                    for t in (getattr(norm, "identifiers", ()) or ())
                )
                intent = _intent(norm, (), idents)
            except Exception:
                intent = None
            qv = QueryViewV7(
                query=query, norm=norm,
                intent=intent or getattr(norm, "intent", None),
                entity_canons=(), query_time_us=now,
            )
        return _v7_guard_year_window(qv)

    def _v7_known_canons(self, conn: Any) -> List[str]:
        """Distinct active entity canons for query-side matching."""
        try:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT canon FROM entity_canon WHERE scope_id = ? "
                    "ORDER BY canon LIMIT 2000",
                    (self._namespace,),
                )
            ]
        except Exception:
            return []

    def _search_v7(
        self,
        query: str,
        *,
        limit: int,
        deadline_ms: float,
        strict: bool,
        require_units: bool = True,
        anchor_us: Optional[int] = None,
    ) -> Optional[Any]:
        """Run the V7 pipeline → PipelineResult, or ``None`` when the
        caller's mode can't provision it.

        ``require_units=True`` (auto mode) declines when the V7
        projection plane (``units``) was never applied — auto falls
        back to the V6 path. ``require_units=False`` (explicit ``v7``)
        runs anyway: the source lane reads ``source_fts``/units-less
        stores directly, so V7 answers honestly on unprojected stores.

        ``anchor_us`` is the per-call temporal anchor (V8-09.01/09.02) —
        the caller's ``as_of`` when supplied, else the wall clock read
        once by the caller. It flows to ``build_query_view(now_us=…)``,
        ``LaneContextV7.query_time_us``, and the lane/verdict manifest —
        never to process state.
        """
        if _v7_run_search is None or LaneContextV7 is None:
            return None
        now = anchor_us if anchor_us is not None else now_us()
        try:
            with self._store.read() as conn:
                if require_units and not has_table(conn, "units"):
                    return None
                generation = self._store.projection_generation()
                if _v7_make_eligible is not None:
                    eligible = _v7_make_eligible(
                        conn, self._store,
                        scope_id=self._namespace,
                        generation=generation,
                        principal_id=self._owner,
                    )
                else:
                    # Fallback fence: lanes already gate scope+generation;
                    # without the adapter every in-fence row is eligible.
                    eligible = lambda _row: True  # noqa: E731
                qv = self._v7_query_view(query, now, conn)
                if qv is None:
                    return None
                profile = str(
                    getattr(self._cfg, "retrieval_profile", "default")
                    or "default"
                )
                policy = (
                    _v7_load_policy(profile)
                    if _v7_load_policy is not None
                    else None
                )
                ctx = LaneContextV7(
                    store=conn,
                    scope_id=self._namespace,
                    generation=generation,
                    eligible=eligible,
                    query_time_us=now,
                    profile=profile,
                    budget=(
                        BudgetClass.MID if BudgetClass is not None else "mid"
                    ),
                    policy=policy,
                    manifest={
                        "limit": limit,
                        "profile": profile,
                        # Dense lane resolves its query encoder + vector
                        # space through the manifest (LaneContextV7 is
                        # frozen — no encoder slot). ``None`` when the
                        # store has no encoder → lane reports unavailable.
                        "query_encoder": self._encoder,
                        "encoder_id": self._encoder_id,
                        # V8-09.01 — the resolved per-call temporal anchor
                        # for lanes/verdict stages that need it.
                        "as_of_us": int(now),
                    },
                )
                return _v7_run_search(ctx, qv, deadline_ms=deadline_ms)
        except VerbatimError:
            if strict:
                raise
            return None
        except Exception:
            return None

    def _v7_hits(self, result: Any, limit: int) -> List[Hit]:
        """Map a PipelineResult's pack items to governed Hits.

        Each pack item's unit_id resolves through the `units` row to its
        (source_id, revision) — the delivered artifact is the pinned
        quote, so the Hit carries the unit's byte-verified slice. One
        read snapshot for the whole assembly (the §33 rule).
        """
        pack = getattr(result, "pack", None)
        items = getattr(pack, "items", None) or []
        if not items:
            return []
        # one covering units join for every delivered item
        unit_ids = [
            getattr(it, "unit_id", None) for it in items
            if getattr(it, "unit_id", None)
        ]
        unit_rows: Dict[str, Dict[str, Any]] = {}
        states: Dict[str, Dict[str, Any]] = {}
        times: Dict[Tuple[str, int], int] = {}
        try:
            with self._store.read() as conn:
                if unit_ids:
                    ph = ",".join("?" * len(unit_ids))
                    try:
                        for row in conn.execute(
                            "SELECT unit_id, source_id, revision, kind, "
                            "speaker_canon, occurred_start_us, recorded_at_us, "
                            "byte_start, byte_end, session_id "
                            f"FROM units WHERE unit_id IN ({ph})",
                            unit_ids,
                        ):
                            unit_rows[row[0]] = {
                                "source_id": row[1],
                                "revision": int(row[2]),
                                "kind": row[3],
                                "speaker_canon": row[4],
                                "occurred_start_us": row[5],
                                "recorded_at_us": row[6],
                                "byte_start": row[7],
                                "byte_end": row[8],
                                "session_id": row[9],
                            }
                    except Exception:
                        unit_rows = {}
                sids = sorted(
                    {u["source_id"] for u in unit_rows.values()}
                )
                pairs = [
                    (u["source_id"], u["revision"])
                    for u in unit_rows.values()
                ]
                if sids:
                    try:
                        states = self._source_states_in(conn, sids)
                    except Exception:
                        states = {}
                    try:
                        times = self._revision_times_in(conn, pairs)
                    except Exception:
                        times = {}
        except Exception:
            unit_rows = {}
        # score lookup from the scored candidates
        scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}
        for sc in getattr(result, "scored", ()) or ():
            uid = getattr(sc, "unit_id", None)
            if uid:
                scores[uid] = float(getattr(sc, "score", 0.0) or 0.0)
                feats = getattr(sc, "features", None) or {}
                details[uid] = {
                    str(k): float(v) for k, v in feats.items()
                    if isinstance(v, (int, float))
                }
        out: List[Hit] = []
        for item in items[:limit]:
            uid = getattr(item, "unit_id", None)
            u = unit_rows.get(uid) or {}
            sid = u.get("source_id") or uid or ""
            rev = int(u.get("revision") or 0)
            st = states.get(sid) or {}
            ref = MemoryRef(
                store_tag=self._tag,
                namespace=self._namespace,
                source_id=sid,
                expected_revision=rev,
                control_version=int(st.get("control_version") or 0),
            ).to_string()
            quote = getattr(item, "quote", b"")
            if isinstance(quote, bytes):
                quote = quote.decode("utf-8", "replace")
            support = getattr(item, "support", None)
            support_v = getattr(support, "value", support) or "supported"
            # V7 group labels (V7-11.01: supported|partial|weak) project onto
            # the governed SupportStatus enum — ``weak`` evidence is NOT
            # adequate support, so it maps to ``insufficient`` rather than
            # silently claiming ``supported``; ``partial`` has verified
            # support and maps to ``supported``. The fine label stays on the
            # pack item / explain payload for consumers that want it.
            support_status = _V7_SUPPORT_TO_STATUS.get(
                str(support_v), SupportStatus.SUPPORTED.value
            )
            recorded = times.get((sid, rev)) or u.get("recorded_at_us")
            occurred_start = u.get("occurred_start_us")
            hit = Hit(
                memory_id=sid,
                ref=ref,
                object_ref=object_ref_to_string("unit", uid, rev),
                kind="source",
                quote=quote or "",
                score=scores.get(uid, 0.0),
                score_family="ranking/v7",
                lifecycle=_LIFECYCLE_MAP.get(
                    str(st.get("disposition") or "active"), "active"
                ),
                support_status=support_status,
                role="supporting",
                type=MemoryType.UNTYPED.value,
                valid_time=(
                    rfc3339(occurred_start) if occurred_start else None
                ),
                recorded_time=rfc3339(recorded) if recorded else None,
                score_detail=details.get(uid) or {},
            )
            out.append(hit)
        return out

    @staticmethod
    def _merge_lane_candidates(
        typed_pool: Iterable[Any], lane_hits: Iterable[Any]
    ) -> List[Any]:
        """Union of typed-lane and source-lane candidates, one per key.

        A record surfaced by both lanes merges into a single admitted
        candidate on the typed carrier — per-signal ``max`` keeps the
        stronger raw value (``ranking/v1`` normalizes each signal across
        the set, so this is a union, never a double count), ``lanes``
        provenance unions, and the carrier keeps the re-verified span
        pins that make the delivered line a fact (V6-02.02).  Typed
        candidates enter first so the merge order stays deterministic.
        """
        merged: Dict[Tuple[str, int], Any] = {}
        order: List[Tuple[str, int]] = []
        for hit in typed_pool or ():
            key = (hit.source_id, int(hit.revision))
            merged[key] = hit
            order.append(key)
        for hit in lane_hits or ():
            key = (hit.source_id, int(hit.revision))
            entry = merged.get(key)
            if entry is None:
                merged[key] = hit
                order.append(key)
                continue
            for name, value in (getattr(hit, "signals", None) or {}).items():
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                prev = entry.signals.get(name)
                entry.signals[name] = (
                    fv if prev is None else max(float(prev), fv)
                )
            for lane_name, rank in (getattr(hit, "lanes", None) or {}).items():
                entry.lanes.setdefault(lane_name, rank)
        return [merged[k] for k in order]

    def _barrier_source_ids(self, receipt_ids: Iterable[str]) -> set:
        """Resolve readiness receipt ids to their ``source_id``s.

        ``rc_ingest:<sid>:<rev>`` parses directly; ``cr_*`` ids are
        deterministic digests over ``(source_id, revision, envelope_kind)``
        — reversed by recomputing ids over this namespace's
        ``source_envelopes`` rows (the same honest lookup
        ``ReadinessEngine._ensure_receipt`` and the source-job dispatcher
        use; a foreign or unknown receipt resolves to nothing).  Advisory
        only — every failure path returns the partial set, never raises.
        """
        out: set = set()
        cr_needed: set = set()
        for rid in receipt_ids or ():
            rid = str(rid or "")
            if rid.startswith("rc_ingest:"):
                sid, _, _rev = rid[len("rc_ingest:"):].rpartition(":")
                if sid:
                    out.add(sid)
            elif rid.startswith("cr_"):
                cr_needed.add(rid)
        if cr_needed and _receipt_id_for is not None:
            try:
                with self._store.read() as conn:
                    if has_table(conn, "source_envelopes"):
                        rows = conn.execute(
                            "SELECT DISTINCT source_id, revision,"
                            " envelope_kind FROM source_envelopes"
                            " WHERE scope_id = ? LIMIT 8192",
                            (self._namespace,),
                        ).fetchall()
                        for sid, rev, kind in rows:
                            try:
                                rid = _receipt_id_for(
                                    str(sid), int(rev), str(kind)
                                )
                            except Exception:
                                continue
                            if rid in cr_needed:
                                out.add(str(sid))
            except Exception:
                pass
        return out

    def _emit_source_exposure(
        self, result: SearchResult, token: str
    ) -> None:
        """Post-delivery source-exposure rows (V6-02, docs/v6_contracts §8).

        One delivery record per delivered source-backed item —
        ``{source_id, revision, score_family}`` — emitted under the
        search's causal token (or a minted exposure receipt when no
        causal frontier exists) in its own short write AFTER the result
        is fully assembled, never inside a read tx.  The sink lands with
        the feedback worker (``verbatim.influence.exposure``): an absent
        or unshaped module is an honest no-op, and an emit failure
        becomes a result warning, never a search failure.
        """
        if _exposure is None:
            return
        deliveries: List[Dict[str, Any]] = []
        for h in result.items:
            # Source-backed hits only — V6 lanes ship ``kind="source"``
            # and V7 ships source-pinned unit slices under the same
            # ``kind="source"`` (granularity rides ``object_ref``).
            if h.kind != "source" or not h.memory_id:
                continue
            rev = self._hit_revision(h)
            if rev is None:
                continue
            deliveries.append(
                {
                    "source_id": h.memory_id,
                    "revision": int(rev),
                    "score_family": h.score_family or "unranked",
                }
            )
        if not deliveries:
            return
        receipt_id = token or f"search:{new_id()}"
        try:
            emit = getattr(_exposure, "emit_deliveries", None)
            record = getattr(_exposure, "record_source_deliveries", None)
            if callable(emit):
                emit(
                    self._store,
                    receipt_id=receipt_id,
                    namespace=self._namespace,
                    deliveries=deliveries,
                )
            elif callable(record):
                with self._store.tx(budget_ms=_POST_DELIVERY_WRITE_MS) as conn:
                    record(conn, receipt_id, deliveries)
            else:
                return  # unshaped sibling module — honest no-op
        except Exception:
            if "exposure_emit_failed" not in result.warnings:
                result.warnings.append("exposure_emit_failed")

    # ---- consumer result cache helpers (V6-02.13, §7.5) ----------------
    #
    # The cache reuses the V4-33 machinery (``retrieval.cache``) at the
    # consumer level — same probe→fingerprint→per-ref revalidation
    # discipline, over the assembled ``SearchResult`` instead of v3
    # packs.  Entries live in the same per-store LRU; ``stats_for``
    # reports the combined counters the A0-cache envelope publishes.

    def _pack_cache_scope_ids(self, conn: sqlite3.Connection) -> List[str]:
        """Every scope row — the fingerprint's epoch vector covers the
        whole authorization plane, so a grant edit on ANY scope
        (governed packs can draw on more than the bound namespace)
        invalidates rather than serving across changed authority."""
        try:
            return [
                str(r[0])
                for r in conn.execute(
                    "SELECT scope_id FROM scopes ORDER BY scope_id"
                ).fetchall()
            ]
        except sqlite3.Error:
            return [self._namespace]

    def _pack_cache_generation(self, conn: sqlite3.Connection) -> int:
        """The committed projection generation on this snapshot — the
        barrier generation the result is computed under."""
        try:
            value = self._store._meta_get(conn, "projection_generation")
            return int(value or 0)
        except Exception:
            return 0

    @staticmethod
    def _pack_cache_object_ref(
        object_ref: Any,
    ) -> Optional[Tuple[str, str, int]]:
        """``vobj1.<kind>.<hex id>.<rev|->`` → ``(kind, id, rev)``."""
        parts = str(object_ref or "").split(".")
        if len(parts) != 4 or parts[0] != "vobj1":
            return None
        try:
            oid = bytes.fromhex(parts[2]).decode("utf-8")
            rev = 0 if parts[3] == "-" else int(parts[3])
        except (ValueError, UnicodeDecodeError):
            return None
        return (parts[1], oid, rev)

    def _pack_cache_refs(
        self, items: List[Hit]
    ) -> Tuple[frozenset, Dict[str, Any]]:
        """Delivered object refs + the ``source_state`` snapshot each
        delivered source carried — the hit-time revalidation contract
        (quarantine/envelope holds, purge suppression, lifecycle and
        head-revision drift are all re-checked live)."""
        refs = set()
        src_ids = set()
        for h in items:
            if h.kind == "source" and h.memory_id:
                rev = self._hit_revision(h)
                refs.add(("source", h.memory_id, int(rev or 0)))
                src_ids.add(h.memory_id)
                continue
            parts = self._pack_cache_object_ref(h.object_ref)
            if parts is not None:
                refs.add(parts)
        expect: Dict[str, Any] = {}
        if src_ids:
            states = self._source_states(sorted(src_ids))
            for sid in sorted(src_ids):
                row = states.get(sid)
                expect[sid] = (
                    (
                        int(row["control_version"]),
                        str(row["disposition"]),
                        str(row["mutation_head"]),
                    )
                    if row
                    else None
                )
        return frozenset(refs), expect

    def _pack_cache_hit_dict(self, hit: Hit) -> Dict[str, Any]:
        """Serialize a delivered hit — the same fields ``to_dict()``
        ships plus the non-field extras (``pins``/``signals``) the
        typed/fusion path attaches for delivery."""
        d = {
            f: copy.deepcopy(getattr(hit, f))
            for f in _PACK_CACHE_HIT_FIELDS
        }
        for extra in _PACK_CACHE_HIT_EXTRAS:
            value = getattr(hit, extra, None)
            if value is not None:
                d["+" + extra] = copy.deepcopy(value)
        return d

    def _pack_cache_hit(self, d: Dict[str, Any]) -> Hit:
        """Rehydrate a stored hit — a fresh object, never aliased to the
        entry's frozen payload."""
        fields = {
            k: copy.deepcopy(v)
            for k, v in d.items()
            if k in _PACK_CACHE_HIT_FIELD_SET
        }
        h = Hit(**fields)
        for extra in _PACK_CACHE_HIT_EXTRAS:
            key = "+" + extra
            if key in d:
                setattr(h, extra, copy.deepcopy(d[key]))
        return h

    def _pack_cache_result(
        self,
        entry: Any,
        *,
        receipts: List[str],
        waited_ms: float,
        unresolved: List[str],
        deferred: List[str],
        cons: Consistency,
    ) -> SearchResult:
        """Rebuild a ``SearchResult`` from a revalidated entry.

        The stored payload replays (items, status, warnings, coverage)
        while the volatile delivery fields come from THIS call's
        barrier — waited_ms, the receipt frontier, and the causal
        token (the caller sets it).  ``cache_hit`` and the
        ``coverage.cache`` flag mark the serve, matching the v3 path's
        hit reporting.
        """
        items = [self._pack_cache_hit(d) for d in entry.items]
        coverage = copy.deepcopy(entry.coverage) if entry.coverage else {}
        coverage["cache"] = {"enabled": True, "hit": True}
        return SearchResult(
            status=entry.status,
            items=items,
            warnings=list(dict.fromkeys([*entry.warnings, "cache_hit"])),
            readiness={
                "causal_satisfied": cons is Consistency.SESSION,
                "waited_ms": waited_ms,
                "receipts": len(receipts),
                "unresolved": list(unresolved)[:32],
                "deferred": sorted(set(deferred)),
            },
            coverage=coverage,
            causal_token="",
        )

    def _pack_cache_store(
        self, cache: Any, probe: Any, result: SearchResult
    ) -> None:
        """Record an assembled result — READY + non-empty only.

        Pending/blocked/partial/degraded/insufficient answers and
        empty hit lists are never recorded: a cached entry can only
        ever replay a complete delivery.  The entry stores under the
        probe-snapshot fingerprint, so a commit that raced the
        pipeline invalidates on next probe rather than blessing a
        result computed across a write boundary.  Store failures stay
        silent — the cache is a performance layer, never an authority.
        """
        if cache is None or probe is None or probe.fingerprint is None:
            return
        if result.status != SearchStatus.READY.value or not result.items:
            return
        try:
            refs, expect = self._pack_cache_refs(result.items)
            items = tuple(
                self._pack_cache_hit_dict(h) for h in result.items
            )
            cache.store_result(
                probe.key,
                fingerprint=probe.fingerprint,
                items=items,
                status=result.status,
                warnings=tuple(result.warnings),
                coverage=result.coverage,
                refs=refs,
                source_expect=expect,
            )
        except Exception:
            pass

    def _filters(self, filters: Optional[dict]) -> Dict[str, Any]:
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise invalid("filters must be a mapping")
        bad = set(filters) - _FILTER_KEYS
        if bad:
            raise invalid(f"unknown filter keys: {sorted(bad)}")
        out: Dict[str, Any] = {}
        for k, v in filters.items():
            if k in ("created_after", "created_before"):
                if not isinstance(v, str):
                    raise invalid(f"filter {k} must be an RFC3339 string")
                out[k] = v
            else:
                vals = v if isinstance(v, (list, tuple)) else [v]
                if not all(isinstance(x, str) and x for x in vals):
                    raise invalid(f"filter {k} must be a string or list of strings")
                out[k] = [str(x) for x in vals]
        return out

    def _apply_filters(self, hits: List[Hit], flt: Dict[str, Any]) -> List[Hit]:
        if not flt:
            return hits

        def ok(hit: Hit) -> bool:
            for key, want in flt.items():
                if key == "type":
                    if hit.type not in want:
                        return False
                elif key == "kind":
                    if hit.kind not in want:
                        return False
                elif key == "lifecycle":
                    if hit.lifecycle not in want:
                        return False
                elif key == "source_id":
                    if hit.memory_id not in want:
                        return False
                elif key == "created_after":
                    if hit.recorded_time is None or hit.recorded_time <= want:
                        return False
                elif key == "created_before":
                    if hit.recorded_time is None or hit.recorded_time >= want:
                        return False
            return True

        return [h for h in hits if ok(h)]

    def _source_states_in(
        self, conn: sqlite3.Connection, ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        if not has_table(conn, _STATE_TABLE):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        ph = ",".join("?" for _ in ids)
        for row in conn.execute(
            f"SELECT source_id, control_version, disposition,"
            f" mutation_head FROM {_STATE_TABLE}"
            f" WHERE source_id IN ({ph})",
            ids,
        ):
            out[row[0]] = {
                "control_version": int(row[1]),
                "disposition": row[2],
                "mutation_head": row[3],
            }
        return out

    def _source_states(self, source_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids = sorted(set(source_ids))
        if not ids:
            return {}
        try:
            with self._store.read() as conn:
                return self._source_states_in(conn, ids)
        except Exception:
            return {}

    def _revision_times_in(
        self, conn: sqlite3.Connection, rows: List[Tuple[str, int]]
    ) -> Dict[Tuple[str, int], int]:
        out: Dict[Tuple[str, int], int] = {}
        sids = sorted({s for s, _ in rows})
        ph = ",".join("?" for _ in sids)
        for sid, rev, cap in conn.execute(
            "SELECT source_id, revision, captured_us FROM source_revisions"
            f" WHERE source_id IN ({ph})",
            sids,
        ):
            out[(sid, int(rev))] = int(cap or 0)
        return out

    def _revision_times(self, pairs: Iterable[Tuple[str, int]]) -> Dict[Tuple[str, int], int]:
        rows = [(s, int(r)) for s, r in pairs]
        if not rows:
            return {}
        try:
            with self._store.read() as conn:
                return self._revision_times_in(conn, rows)
        except Exception:
            return {}

    def _hits_from_packs(self, packs: Iterable[Any]) -> List[Hit]:
        pairs: List[Tuple[Any, Any]] = []
        for pack in packs or ():
            for item in getattr(pack, "items", ()) or ():
                handle = item.handle
                pairs.append((item, handle))
        # One read snapshot for the whole assembly — SPEC_V5 §33's one
        # snapshot per query (was two). Each part keeps its own
        # failure semantics: a failed lookup degrades to {} alone.
        state_ids = sorted(
            {h.object_id for _i, h in pairs if h.object_kind == "source"}
        )
        time_pairs = [
            (h.object_id, int(h.revision))
            for _i, h in pairs
            if h.object_kind == "source"
        ]
        states: Dict[str, Dict[str, Any]] = {}
        times: Dict[Tuple[str, int], int] = {}
        if state_ids:
            try:
                with self._store.read() as conn:
                    try:
                        states = self._source_states_in(conn, state_ids)
                    except Exception:
                        states = {}
                    try:
                        times = self._revision_times_in(conn, time_pairs)
                    except Exception:
                        times = {}
            except Exception:
                states = {}
                times = {}
        out: List[Hit] = []
        for item, handle in pairs:
            if handle.object_kind == "source":
                st = states.get(handle.object_id) or {}
                ref = MemoryRef(
                    store_tag=self._tag,
                    namespace=self._namespace,
                    source_id=handle.object_id,
                    expected_revision=int(handle.revision),
                    control_version=int(st.get("control_version") or 0),
                ).to_string()
                lifecycle = _LIFECYCLE_MAP.get(
                    str(st.get("disposition") or "active"), "active"
                )
                recorded = times.get((handle.object_id, int(handle.revision)))
                warnings = ["verify_recommended"] if item.verify_recommended else []
                out.append(
                    Hit(
                        memory_id=handle.object_id,
                        ref=ref,
                        object_ref=object_ref_to_string(
                            "source", handle.object_id, int(handle.revision)
                        ),
                        kind="source",
                        quote=item.text,
                        score=0.0,
                        score_family="unranked",
                        lifecycle=lifecycle,
                        support_status="unassessed",
                        role="supporting",
                        recorded_time=rfc3339(recorded) if recorded else None,
                        corroboration=max(1, int(getattr(item, "proof_count", 0) or 1)),
                        warnings=warnings,
                    )
                )
            else:
                out.append(
                    Hit(
                        memory_id="",
                        ref="",
                        object_ref=object_ref_to_string(
                            str(handle.object_kind),
                            handle.object_id,
                            int(handle.revision),
                        ),
                        kind=str(handle.object_kind),
                        quote=item.text,
                        score=0.0,
                        score_family="unranked",
                        lifecycle=_LIFECYCLE_MAP.get(
                            str(item.lifecycle or "active"), "active"
                        ),
                        support_status="unassessed",
                        role="supporting",
                        warnings=(
                            ["verify_recommended"] if item.verify_recommended else []
                        ),
                    )
                )
        return out

    def _held_source_ids_in(self, conn: sqlite3.Connection, ids: List[str]) -> set:
        """The ``_held_source_ids`` queries on the caller's snapshot —
        same fail-closed contract (raises propagate to the caller's
        exception policy)."""
        if not has_table(conn, "quarantine"):
            return set()
        # No live hold anywhere → nothing can be withheld: the two
        # scoped joins below can only map existing hold rows onto
        # source ids, and an empty hold set maps to nothing.
        if conn.execute(
            "SELECT 1 FROM quarantine"
            " WHERE state IN ('pending','suppressed') LIMIT 1"
        ).fetchone() is None:
            return set()
        ph = ",".join("?" for _ in ids)
        held = {
            str(r[0])
            for r in conn.execute(
                "SELECT object_id FROM quarantine"
                f" WHERE object_kind = 'source' AND object_id IN ({ph})"
                " AND state IN ('pending','suppressed')",
                ids,
            )
        }
        held |= {
            str(r[0])
            for r in conn.execute(
                "SELECT se.source_id FROM quarantine q"
                " JOIN source_envelopes se"
                "  ON se.envelope_id = q.object_id"
                f" WHERE q.object_kind = 'source_envelope'"
                f"  AND se.source_id IN ({ph})"
                "  AND q.state IN ('pending','suppressed')",
                ids,
            )
        }
        return held

    def _held_source_ids(self, source_ids: Iterable[str]) -> set:
        """Source ids under a live quarantine hold (``pending``/``suppressed``).

        The source lane reads projection rows, not the hold table — a
        source held after its projection materialized (policy tightened
        post-capture, or a declared-protected write whose projection a
        pre-hold job committed) must still be withheld here, mirroring
        the claim-lane quarantine cascade (V3-34.10). Fail-closed on the
        read itself: when the hold table cannot be checked, every
        candidate id is reported held rather than leaking.
        """
        ids = sorted(set(source_ids))
        if not ids:
            return set()
        try:
            with self._store.read() as conn:
                return self._held_source_ids_in(conn, ids)
        except Exception:
            return set(ids)

    def _hits_from_source_lane(
        self,
        fused_hits: Iterable[Any],
        typed_map: Optional[Dict[Tuple[str, int], Any]] = None,
    ) -> List[Hit]:
        fused = list(fused_hits or ())
        if not fused:
            return []
        # One read snapshot for hold-check + metadata + verified quotes —
        # the §33 single-snapshot rule (was four snapshots). Each part
        # keeps its own failure semantics: a failed hold check withholds
        # every candidate (fail closed), the metadata lookups degrade to
        # {}, and a quote-assembly error propagates so the lane reports
        # unavailable — the same taxonomy the per-snapshot calls had.
        acquired = False
        try:
            with self._store.read() as conn:
                acquired = True
                try:
                    held = self._held_source_ids_in(
                        conn, sorted({h.source_id for h in fused})
                    )
                except Exception:
                    held = {h.source_id for h in fused}
                if held:
                    fused = [h for h in fused if h.source_id not in held]
                if fused:
                    sids = sorted({h.source_id for h in fused})
                    pairs = [(h.source_id, int(h.revision)) for h in fused]
                    try:
                        states = self._source_states_in(conn, sids)
                    except Exception:
                        states = {}
                    try:
                        times = self._revision_times_in(conn, pairs)
                    except Exception:
                        times = {}
                    quotes = self._lane_quotes_in(conn, pairs)
                else:
                    states, times, quotes = {}, {}, {}
        except Exception:
            if not acquired:
                # A dead snapshot maps onto the old per-part outcome:
                # the fail-closed hold check withheld everything.
                return []
            raise
        out: List[Hit] = []
        for h in fused:
            st = states.get(h.source_id) or {}
            ref = MemoryRef(
                store_tag=self._tag,
                namespace=self._namespace,
                source_id=h.source_id,
                expected_revision=int(h.revision),
                control_version=int(st.get("control_version") or 0),
            ).to_string()
            recorded = times.get((h.source_id, int(h.revision)))
            detail = getattr(h, "score_detail", None) or {}
            trec = (
                typed_map.get((h.source_id, int(h.revision)))
                if typed_map
                else None
            )
            hit = Hit(
                memory_id=h.source_id,
                ref=ref,
                object_ref=object_ref_to_string("source", h.source_id, int(h.revision)),
                kind="source",
                quote=quotes.get((h.source_id, int(h.revision)), ""),
                score=float(getattr(h, "score", 0.0) or 0.0),
                # A grounded typed record delivers under its own score
                # family (V6-02); a projection-only candidate stays
                # "ranking/v1". Both were ranked by the same fusion pass.
                score_family="typed" if trec is not None else "ranking/v1",
                lifecycle=_LIFECYCLE_MAP.get(
                    str(st.get("disposition") or "active"), "active"
                ),
                support_status="unassessed",
                role="supporting",
                type=(
                    str(trec.mem_type)
                    if trec is not None and trec.mem_type
                    else MemoryType.UNTYPED.value
                ),
                valid_time=(
                    trec.valid_time if trec is not None else None
                ),
                recorded_time=rfc3339(recorded) if recorded else None,
                score_detail={
                    str(k): float(v.get("contribution", 0.0))
                    for k, v in detail.items()
                    if isinstance(v, dict)
                },
            )
            if trec is not None:
                # The delivered typed line carries its re-verified span
                # pins — the byte-exact grounding evidence (V6-02.02).
                # ``pins`` rides beside ``signals`` as a delivery-time
                # attribute (the Hit schema itself is owned by the core
                # types contract; ``to_dict`` keeps the frozen shape).
                hit.pins = [dict(p) for p in (trec.pins or ())]
            # Raw lane signals ride beside the fused contributions so the
            # support verdict can apply the pinned encoder's calibrated
            # floor to encoder-native values (V5-31.07/31.08) — the
            # normalized ``score_detail`` contributions are a different
            # unit and can never be compared to a cosine floor.
            raw_signals = getattr(h, "signals", None)
            if isinstance(raw_signals, dict):
                hit.signals = dict(raw_signals)
            out.append(hit)
        return out

    def _lane_quotes_in(
        self, conn: sqlite3.Connection, pairs: List[Tuple[str, int]]
    ) -> Dict[Tuple[str, int], str]:
        """Exact bytes for lane hits — the same verified-read + quote-gate
        the evidence lane applies (never a raw bypass), on the caller's
        snapshot. Per-pair corruption still withholds only that pair;
        an unexpected read failure propagates like the old per-call
        loop's did."""
        out: Dict[Tuple[str, int], str] = {}
        if not pairs:
            return out
        try:
            verbs = governance.effective_verbs(
                conn, self._caller(), self._namespace
            )
        except Exception:
            return out
        # quote is required for payload text; read-only callers get refs.
        if Verb.QUOTE.value not in {str(v) for v in verbs}:
            return out
        repo = SourcesRepo(self._store)
        verified, _corrupt = repo.payload_many(pairs, conn=conn)
        for (sid, rev), payload in verified.items():
            if payload:
                out[(sid, rev)] = payload.decode("utf-8", "replace")
        return out

    def _lane_quotes(self, pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], str]:
        """Standalone-snapshot form of ``_lane_quotes_in``."""
        with self._store.read() as conn:
            return self._lane_quotes_in(conn, pairs)

    def _dedup_hits(self, hits: List[Hit]) -> List[Hit]:
        """One row per object — highest score wins, then deterministic order."""
        best: Dict[str, Hit] = {}
        order: List[str] = []
        for h in hits:
            key = h.object_ref or h.ref or h.memory_id
            if not key:
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = h
                order.append(key)
            elif h.score > prev.score:
                best[key] = h
        ranked = sorted(
            (best[k] for k in order),
            key=lambda h: (-h.score, h.memory_id, h.object_ref),
        )
        return ranked

    def _hit_revision(self, hit: Hit) -> Optional[int]:
        """Expected revision of a source-backed hit — the duplicate-link
        key. The bound ``ref`` is authoritative; ``object_ref`` is the
        fallback for hits whose ref was never minted."""
        if hit.ref:
            try:
                return int(MemoryRef.parse(hit.ref).expected_revision)
            except Exception:
                pass
        tail = str(hit.object_ref or "").rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else None

    def _collapse_duplicate_groups(self, hits: List[Hit]) -> List[Hit]:
        """Duplicate collapse on delivery (V5-10.04, V5-30.05/30.09).

        Two collapse shapes share one rule — duplicates report under one
        representative, never as independent corroboration, and never
        crowd out unlinked (independent or contrary) records:

        * ``duplicate_links`` groups: delivered members of one group ship
          as a single representative hit; ``collapsed_duplicates`` counts
          every *live* member hidden under it, and ``corroboration``
          counts only distinct (origin, speaker, provenance) attributions
          across the group.
        * acceptance-time collapse: a byte-identical re-add replays onto
          the committed record — no member row exists to link — so the
          recorded submission count contributes ``collapsed_duplicates``
          directly (same submitter ⇒ corroboration stays 1).

        Collapse is reporting, never removal: on any read failure the
        un-collapsed hit list ships unchanged.
        """
        if _dedup_links is None:
            return hits
        keyed: List[Tuple[Hit, str, int]] = []
        for h in hits:
            if h.kind != "source" or not h.memory_id:
                continue
            rev = self._hit_revision(h)
            if rev is not None:
                keyed.append((h, h.memory_id, rev))
        if not keyed:
            return hits
        try:
            with self._store.read() as conn:
                # No link rows and no replay-count markers anywhere →
                # nothing can collapse or contribute submissions: the
                # per-hit probes below would only re-derive emptiness.
                try:
                    quiet = (
                        conn.execute(
                            "SELECT 1 FROM duplicate_links LIMIT 1"
                        ).fetchone() is None
                        and conn.execute(
                            "SELECT 1 FROM meta WHERE key LIKE"
                            " 'dedup.submissions.%' LIMIT 1"
                        ).fetchone() is None
                    )
                except Exception:
                    quiet = False
                if quiet:
                    return hits
                gid_of: Dict[int, str] = {}
                views: Dict[str, dict] = {}
                subs_of: Dict[int, int] = {}
                pairs = [(sid, rev) for _h, sid, rev in keyed]
                gid_map = _dedup_links.link_groups(conn, pairs)
                for h, sid, rev in keyed:
                    gid = gid_map.get((sid, rev))
                    if gid:
                        gid_of[id(h)] = gid
                        if gid not in views:
                            views[gid] = (
                                _dedup_links.collapse_for_hit(conn, sid, rev)
                                or {}
                            )
                member_pairs = [
                    (msid, mrev)
                    for v in views.values()
                    for msid, mrev in (v or {}).get("member_refs", ())
                ]
                subs_map = _dedup_links.duplicate_submission_counts(
                    conn, pairs + member_pairs
                )
                for h, sid, rev in keyed:
                    subs_of[id(h)] = subs_map.get((sid, rev), 0)
                # Submissions collapsed onto any live member roll up to
                # the group's representative (V5-30.09).
                group_subs: Dict[str, int] = {}
                for gid, view in views.items():
                    group_subs[gid] = sum(
                        subs_map.get((str(msid), int(mrev)), 0)
                        for msid, mrev in (view or {}).get("member_refs", ())
                    )
        except Exception:
            return hits

        groups: Dict[str, List[Hit]] = {}
        for h, _sid, _rev in keyed:
            gid = gid_of.get(id(h))
            if gid:
                groups.setdefault(gid, []).append(h)

        keep: set = set()
        for gid, members in groups.items():
            view = views.get(gid) or {}
            rep = view.get("representative") or {}
            chosen = next(
                (m for m in members if m.memory_id == rep.get("source_id")),
                None,
            )
            if chosen is None:
                # The canonical rep was not delivered — the best-ranked
                # member carries the group's report instead.
                chosen = sorted(
                    members, key=lambda m: (-m.score, m.memory_id)
                )[0]
            chosen.collapsed_duplicates = int(
                view.get("collapsed_duplicates") or 0
            ) + int(group_subs.get(gid) or 0)
            chosen.corroboration = max(
                1, int(view.get("corroboration") or 1)
            )
            keep.add(id(chosen))

        out: List[Hit] = []
        for h in hits:
            gid = gid_of.get(id(h))
            if gid is not None:
                if id(h) in keep:
                    out.append(h)
                continue  # non-representative members collapse under it
            n = subs_of.get(id(h)) or 0
            if n:
                h.collapsed_duplicates = n
            out.append(h)
        return out

    #: Advisory update relations that mark an unresolved conflict between
    #: live records. ``refines`` admits a compatible refinement — it does
    #: not dispute the prior's standing answer.
    _CONFLICT_RELATIONS = frozenset({"contradicts", "negates", "newer_value"})

    def _label_conflict_groups(self, hits: List[Hit]) -> None:
        """Source-side conflict-group labeling (V5-13.02/13.03, V5-30.21).

        A delivered source hit with an OPEN ``contradicts``/``negates``/
        ``newer_value`` candidate against a still-active record is a
        member of a required conflict group — it ships labeled
        ``disputed`` with a warning, never as a lone settled answer,
        whether or not the counterpart is itself delivered. Resolved
        groups stay clean: ``replaces=``/corrected/retracted priors are
        non-current and deletion closure sweeps a forgotten endpoint's
        candidate rows. The label is opaque — it reports that an
        unresolved group exists, never what the counterparty says
        (V5-13.05).
        """
        src_hits = [h for h in hits if h.kind == "source" and h.memory_id]
        if not src_hits or _list_open_candidates is None:
            return
        delivered = {h.memory_id for h in src_hits}
        try:
            with self._store.read() as conn:
                # The open-candidate materialization is the expensive
                # part; a namespace with no open conflict relation can
                # never label a hit, so probe before materializing.
                try:
                    probe = conn.execute(
                        "SELECT 1 FROM update_candidates"
                        " WHERE namespace = ? AND state = 'open'"
                        " AND relation IN"
                        " ('contradicts','negates','newer_value')"
                        " LIMIT 1",
                        (self._namespace,),
                    ).fetchone()
                except Exception:
                    probe = True  # unreadable → take the honest slow path
                if probe is None:
                    return
                # Only pairs touching a delivered hit can label anything —
                # push that filter into SQL so a large open-candidate set
                # does not materialize (and so the 512-cap of the list
                # helper can never truncate a pair that matters).
                rels = sorted(self._CONFLICT_RELATIONS)
                rel_ph = ",".join("?" for _ in rels)
                ids = sorted(delivered)
                rows: list = []
                for i in range(0, len(ids), 200):
                    chunk = ids[i : i + 200]
                    id_ph = ",".join("?" for _ in chunk)
                    rows.extend(conn.execute(
                        "SELECT new_source_id, prior_source_id, relation"
                        " FROM update_candidates"
                        " WHERE namespace = ? AND state = 'open'"
                        f" AND relation IN ({rel_ph})"
                        f" AND (new_source_id IN ({id_ph})"
                        f"  OR prior_source_id IN ({id_ph}))",
                        (
                            self._namespace,
                            *rels,
                            *chunk,
                            *chunk,
                        ),
                    ).fetchall())
                pairs: List[Tuple[str, str]] = []
                other_ids: set = set()
                for na, nb, _rel in rows:
                    a = str(na or "")
                    b = str(nb or "")
                    if a and b and (a in delivered or b in delivered):
                        pairs.append((a, b))
                        other_ids.update((a, b))
                if not pairs:
                    return
                # Same snapshot as the candidate read — the dispute
                # label and its evidence come from one consistent view.
                states = self._source_states_in(conn, sorted(other_ids))
        except Exception:
            return
        unresolved: set = set()
        for a, b in pairs:
            for sid, other in ((a, b), (b, a)):
                if sid not in delivered:
                    continue
                disp = (states.get(other) or {}).get("disposition")
                # An open candidate whose counterpart is still a current
                # answer — or whose resolution cannot be verified — keeps
                # this member visibly disputed.
                if disp is None or disp == "active":
                    unresolved.add(sid)
        for h in src_hits:
            if h.memory_id in unresolved:
                h.support_status = SupportStatus.DISPUTED.value
                if "conflict_unresolved" not in h.warnings:
                    h.warnings.append("conflict_unresolved")

    # ---- causal tokens -------------------------------------------------------

    def _mint_causal_token(self, receipts: List[str]) -> str:
        """A reattach token for cross-process causal promises (V5-08.19)."""
        if not receipts:
            return ""
        frontier = sorted(dict.fromkeys(receipts))[-256:]
        digest = self._store.hmac(
            b"memory-causal|" + "|".join(frontier).encode("utf-8")
        ).hex()[:24]
        key = f"{_CAUSAL_PREFIX}{digest}"
        try:
            # The token is content-addressed over the frontier — a stored
            # record with >TTL/2 of validity left already resolves it
            # identically, so the refresh write (a synchronous=FULL
            # commit) is skipped until renewal could actually matter.
            with self._store.read() as conn:
                rec = self._store._meta_get(conn, key)
            if (
                isinstance(rec, dict)
                and int(rec.get("expires_us") or 0) - now_us()
                >= _CAUSAL_TTL_US // 2
            ):
                return f"mc1.{digest}"
            with self._store.tx(budget_ms=_POST_DELIVERY_WRITE_MS) as conn:
                rec = {
                    "v": 1,
                    "owner": self._owner,
                    "namespace": self._namespace,
                    "receipts": frontier,
                    "created_us": now_us(),
                    "expires_us": now_us() + _CAUSAL_TTL_US,
                }
                self._store._meta_set(conn, key, rec)
        except VerbatimError:
            return ""
        return f"mc1.{digest}"

    def _resolve_after(
        self, after: Any, strict: bool, warnings: List[str]
    ) -> List[str]:
        """Explicit causal handles: ``mc1.*`` tokens or receipt references."""
        if isinstance(after, str) and after.startswith("mc1."):
            key = f"{_CAUSAL_PREFIX}{after[4:]}"
            with self._store.read() as conn:
                rec = self._store._meta_get(conn, key)
            if not isinstance(rec, dict):
                raise denied("unknown or foreign causal token")
            if (
                rec.get("owner") != self._owner
                or rec.get("namespace") != self._namespace
            ):
                raise denied("causal token belongs to another partition")
            if now_us() >= int(rec.get("expires_us") or 0):
                if strict:
                    raise unavailable("causal token expired")
                warnings.append("causal_token_expired")
                return []
            return [r for r in rec.get("receipts") or [] if isinstance(r, str)]
        # A receipt reference is honored only when it resolves inside this
        # namespace — foreign receipts are indistinguishable from unknown.
        rid = self._receipt_id_of(after)
        if self._engine.available:
            try:
                self._engine.receipt_state(rid, scope_id=self._namespace)
            except VerbatimError:
                raise denied("receipt is not in this namespace")
        return [rid]

    # ------------------------------------------------------------------
    # inspect / forget — delegated to the bound-caller controls surface
    # (verbatim.memory.controls): one inspect/closure implementation, one
    # ref grammar, one confirmation-token format — never a parallel path.
    # ------------------------------------------------------------------

    def inspect(self, ref: Any, *, detail: str = "evidence") -> Inspection:
        """Explain one authorized memory (V5-15.01).  Accepts a serialized
        or object ``MemoryRef``, an object ref (``vobj1.…``, tuple, or
        mapping), or a bare ``memory_id``.  Missing, foreign, and
        unauthorized targets all deny identically."""
        self._require_live()
        return self._controls.inspect(ref, detail=detail)

    def forget(
        self,
        ref: Optional[Any] = None,
        *,
        query: Optional[str] = None,
        confirmation: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ForgetResult:
        """Suppression-and-closure through the existing privacy services
        (V5-15.03/04): ``forget(ref)`` is targeted, ``forget(query=…)``
        returns a confirmation-bound preview, and
        ``forget(confirmation=…)`` executes exactly the pinned selection.
        """
        self._require_live()
        return self._controls.forget(
            ref,
            query=query,
            confirmation=confirmation,
            idempotency_key=idempotency_key,
        )

    def _bound_ref(self, ref: Any, *, what: str = "ref") -> MemoryRef:
        """Resolve a caller ref to a namespace/store-bound MemoryRef —
        the only form mutation paths accept (V5-06.11)."""
        if isinstance(ref, MemoryRef):
            mref = ref
        elif isinstance(ref, str) and ref.startswith(MemoryRef.PREFIX + "."):
            try:
                mref = MemoryRef.parse(ref)
            except (ValueError, TypeError) as exc:
                raise invalid(f"malformed MemoryRef for {what}: {exc}")
        else:
            raise invalid(
                f"{what} requires a source-backed MemoryRef — bare "
                "claim/view refs are rejected (V5-06.11)"
            )
        if mref.store_tag != self._tag or mref.namespace != self._namespace:
            raise denied()
        return mref


    # ------------------------------------------------------------------
    # status / close
    # ------------------------------------------------------------------

    def status(self) -> MemoryStatus:
        """Honest operational snapshot — never config-implied health."""
        self._require_live()
        warnings = list(self._warnings)
        if self._worker_handle is not None:
            worker = self._worker_handle.status()
        elif self._worker_mode is WorkerMode.EXTERNAL:
            from .worker import external_status

            worker = external_status()
        else:
            worker = {"mode": self._worker_mode.value, "running": False}
        counts: Dict[str, int] = {}
        if self._engine.available:
            try:
                for row in self._engine.pending(self._namespace):
                    st = str(row.get("state", "pending"))
                    counts[st] = counts.get(st, 0) + 1
                for row in self._engine.pending(
                    self._namespace, states=("failed", "cancelled")
                ):
                    st = str(row.get("state", "failed"))
                    counts[st] = counts.get(st, 0) + 1
            except VerbatimError:
                warnings.append("readiness_unavailable")
        else:
            warnings.append("readiness_unavailable")
        capabilities = {
            "source_lane": "available" if _source_candidates is not None else "unavailable",
            "source_jobs": "available" if self._source_jobs is not None else "unavailable",
            "update_detection": (
                "available" if detect_update_candidates is not None else "unavailable"
            ),
            "encoder": self._encoder_id,
            "readiness": "available" if self._engine.available else "unavailable",
        }
        cache_stats: Dict[str, Any] = {}
        if _rcache is not None:
            try:
                cache_stats = _rcache.stats_for(self._store, self._cfg)
            except Exception:
                cache_stats = {"state": "error"}
        return MemoryStatus(
            profile=_PROFILE,
            store_tag=self._tag,
            namespace=self._namespace,
            caller=self._owner,
            worker=worker,
            encoder=self._encoder_id,
            cache=cache_stats,
            readiness_counts=counts,
            capabilities=capabilities,
            warnings=list(dict.fromkeys(warnings)),
        )

    def close(self, *, timeout_ms: float = 5000, strict: bool = False) -> CloseReport:
        """Bounded, idempotent shutdown; honest about what did not finish."""
        timeout = _nonneg_ms(timeout_ms, "timeout_ms")
        if not isinstance(strict, bool):
            raise invalid("strict must be a bool")
        if os.getpid() != self._pid:
            # Fork-inherited object: detach locally without touching the
            # parent's connections or worker registry (V5-09.10).
            self._closed = True
            return CloseReport(
                closed=True,
                drained=False,
                worker_stopped=False,
                incomplete=["forked_child"],
                warnings=[
                    "object inherited across fork — detached in the child "
                    "only; the parent's resources keep running"
                ],
            )
        with self._lock:
            if self._closed:
                return self._close_report or CloseReport(
                    closed=True, drained=True, worker_stopped=True
                )
            self._closing = True
        incomplete: List[str] = []
        warnings: List[str] = []
        worker_stopped = self._worker_mode is WorkerMode.EXTERNAL
        drained = self._worker_mode is WorkerMode.EXTERNAL
        pending = 0

        if self._worker_handle is not None:
            try:
                report = self._worker_handle.release(timeout_ms=timeout)
                worker_stopped = bool(report.worker_stopped)
                drained = bool(report.drained)
                pending = max(pending, int(report.pending_obligations or 0))
                incomplete.extend(str(i) for i in report.incomplete)
                warnings.extend(str(w) for w in report.warnings)
            except Exception as exc:
                worker_stopped = False
                incomplete.append("worker_release_error")
                warnings.append(f"worker_release_error:{type(exc).__name__}")

        if self._engine.available:
            try:
                pending = max(pending, len(self._engine.pending(self._namespace)))
            except VerbatimError:
                warnings.append("readiness_unavailable")

        try:
            self._store.close()
        except Exception as exc:
            incomplete.append("store_close_error")
            warnings.append(f"store_close_error:{type(exc).__name__}")

        report = CloseReport(
            closed="store_close_error" not in incomplete,
            drained=drained and pending == 0,
            worker_stopped=worker_stopped,
            pending_obligations=pending,
            incomplete=incomplete,
            warnings=warnings,
        )
        if "store_close_error" not in incomplete:
            self._closed = True
            self._close_report = report
        if strict and (incomplete or pending):
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                "close did not meet the strict contract: "
                + ", ".join(incomplete or ["pending obligations"]),
            )
        return report

    # ------------------------------------------------------------------
    # context manager + fork safety
    # ------------------------------------------------------------------

    def __enter__(self) -> "Memory":
        self._require_live()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close()
        except Exception:
            # Never mask an in-flight exception with a close failure.
            if exc_type is None:
                raise
        return False

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            if os.getpid() == self._pid and not self._closed:
                self.close(timeout_ms=500)
        except Exception:
            pass
