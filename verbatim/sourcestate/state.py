"""``source_state/v1`` — the registered control artifact for raw-memory
lifecycle (SPEC_V5 §14.3, V5-14.09 … V5-14.16; registration duties per
V5-03.06).

A *logical memory* is one ``source_id`` plus its immutable
``source_revisions`` chain. V5 adds this control record so replacements,
corrections, retractions, and erasure have a compare-and-set anchor that
byte revisions alone cannot express. It is **not** a new evidence plane,
grant evaluator, or memory database — the API ``memory_id`` stays the
source id and canonical bytes stay in ``source_revisions`` untouched.

Storage model (one authority, existing interfaces):

* ``source_state`` — the frozen docs/v5_contracts.md §3 table. One row per
  ``source_id``: the *latest committed control record* (CAS projection).
* ``objects`` / ``object_revisions`` — the artifact registers under kind
  ``source_state`` with ``object_id = artifact_id(source_id)``
  (``"source_state:<sid>"``, digest-mangled when that would exceed the id
  alphabet). The derived id is deliberate: registering the raw
  ``source_id`` under a second kind would make ``Kernel._resolve`` see two
  kinds for one id and deny *every* kernel access to the source as
  ambiguous — the control artifact is its own object bound to the source.
* Each committed control version ``cv`` serializes to one immutable doc at
  ``object_revisions.revision == cv + 1`` (registry revisions are 1-based
  so every doc — including the ``cv=0`` creation doc — can carry
  ``derivations`` provenance edges). ``objects.current_revision`` mirrors
  the doc axis. ``source_state.control_version`` remains the CAS counter
  callers pin in ``MemoryRef`` (V5-06.11).
* ``derivations`` edges bind each doc to its predecessor doc and to the
  ``source_revision`` the head names — deletion closure traverses these
  edges (V5-14.16) and provenance never dangles into an unregistered kind
  (``source_state`` is in ``derivations.OBJECT_KINDS``).
* ``producer_manifests`` row + per-transition ``producer``/``actor``/
  ``operation_id`` fields carry decision provenance (§14.3 "Decision/
  provenance").

Dispositions (§14.3 revision dispositions + schema comment):

* ``recorded`` — registered, pre-admission/legacy-adopted; never the
  current answer (V5-14.09: ``active`` means *admitted*).
* ``active`` — head is the current answer inside its valid window.
* ``superseded`` — a ``supersede`` transition: ``mutation_head`` names the
  successor, ``effective_at`` the declared boundary. Before the boundary
  the *predecessor* still answers; at/after it the successor does
  (V5-14.12 — evaluated at read time, no background job required).
* ``corrected`` — a ``correct`` transition: the predecessor assertion is
  retracted over the declared validity interval; the successor (when one
  is supplied) answers. Correction takes effect at record time —
  ``effective_at`` is rejected for it (V5-14.13).
* ``retracted`` — a ``retract``: no successor, nothing current.
* ``archived`` — retired but retained; nothing current.
* ``erased`` — erasure tombstone. Terminal: ``transition`` refuses it and
  ``ensure_state`` will not resurrect it (V5-14.16 — erasure may clear a
  head but never reactivates an older revision).

The seven registration handlers V5-03.06/V5-14.09 require *before use*:

* **access** — the ``objects`` registry row (scope = the source's
  namespace token) is what ``Kernel.resolve_access`` evaluates; the
  artifact has no byte surface so ``read_verified`` correctly denies
  byte reads on it.
* **provenance** — ``derivations`` edges + ``producer_manifests`` +
  per-doc ``producer``/``actor``/``operation_id``/``recorded_event``.
* **invalidation** — writers accept ``store=`` and bump the store
  projection generation inside the same transaction, the existing
  signal that claims/views/index/cache dependents are stale; pending
  producers re-fence through ``assert_publishable`` (V5-14.14).
* **erasure** — ``transitions.apply_erasure`` is the closure hook:
  deletion/purge paths call it inside their closure transaction to fence
  the row to the ``erased`` tombstone (membership per V5-14.16); CAS
  options enforce V5-15.03's version-bound suppression.
* **serialization** — ``serialize_doc``/``parse_doc``/``doc_digest`` and
  ``SourceState.to_dict``/``from_dict`` give the canonical
  ``source_state/v1`` encoding; digests are ``sha256:`` over the canonical
  JSON doc (the row is control metadata, not kernel byte egress, so no
  store key is needed to verify it).
* **migration** — ``install_schema`` creates the §3 table idempotently
  (identical DDL to the v4→v5 storage migration, so it composes with it)
  and records its own install marker in ``meta``; ``adopt`` registers a
  pre-v5 source at ``recorded``/``unresolved`` — unknown legacy lifecycle
  stays unresolved rather than inferred approval (V5-14.16, §08).
* **recovery** — ``recover`` rebuilds the CAS row from the verified doc
  history (docs are the durable artifact; the row is a projection), and
  ``verify`` re-checks every doc digest (``STORE_CORRUPT`` on tamper).

Transaction discipline: writers take the caller's ``conn`` inside
``store.tx()`` — transition, doc append, registry bump, provenance edges,
and the generation bump commit or roll back together (coordinator-effect
style, V5-14.10). The logical effect kind is ``source_revision_transition``;
its persisted encoding is the ``source_state/v1`` doc (the
``EffectKind``/``_EFFECT_TABLES`` registration in ``jobs/coordinator.py``
belongs to the coordinator owner — this module supplies the artifact
encoding it will name).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..core import time as _time
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..memory.types import SOURCE_STATE_KIND
from ..storage import repos_v4
from ..storage.repos import has_table
from ..derivations import record_edge

try:  # storage worker owns verbatim/storage/repos_v5.py (v5_contracts §3)
    from ..storage import repos_v5 as _repos_v5
except Exception:  # pragma: no cover - absent until the storage wave lands
    _repos_v5 = None


# ----------------------------------------------------------------------
# contract constants
# ----------------------------------------------------------------------

# ``SOURCE_STATE_KIND`` (``source_state/v1``) is imported from the frozen
# ``verbatim/memory/types.py`` contract — the doc-format marker stored
# inside every serialized state doc.

#: ``objects.kind`` / ``derivations`` kind token for this artifact.
OBJECT_KIND = "source_state"

#: The v5 contract table this module owns.
TABLE = "source_state"

#: Logical coordinator effect this module encodes (V5-14.10). Registration
#: of the persisted encoding inside ``jobs/coordinator.py``'s effect map is
#: the coordinator owner's wiring step; every transition doc names it.
EFFECT_KIND = "source_revision_transition"

PRODUCER_ID = "producer:verbatim.sourcestate.v1"
_POLICY_VERSION = "sourcestate.v1"

#: Frozen §3 DDL — column-identical to the v4→v5 storage migration's
#: definition (``storage/schema_v5.py``), including its namespace index.
DDL_SOURCE_STATE = """
CREATE TABLE IF NOT EXISTS source_state (
    source_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    control_version INTEGER NOT NULL,
    mutation_head TEXT NOT NULL,
    disposition TEXT NOT NULL,
    superseded_by TEXT,
    effective_at TEXT,
    known_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    updated_at TEXT NOT NULL,
    producer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_state_ns
    ON source_state(namespace, disposition);
"""

#: Full disposition vocabulary (§14.3 revision dispositions).
DISPOSITIONS = frozenset({
    "recorded",
    "active",
    "superseded",
    "corrected",
    "retracted",
    "archived",
    "erased",
})

#: Dispositions a caller may request through ``transition`` —
#: ``erased`` is reachable only through the erasure path (V5-14.16).
TRANSITION_DISPOSITIONS = DISPOSITIONS - {"erased"}

#: Change-kind aliases accepted by ``transition`` → stored disposition.
CHANGE_ALIASES = {
    "supersede": "superseded",
    "correct": "corrected",
    "retract": "retracted",
    "archive": "archived",
    "record": "recorded",
    "adopt": "recorded",
    "activate": "active",
}

#: ``mutation_head`` sentinel — "unresolved/none" per §14.3 (a legacy or
#: pre-admission record with no approved head; never silently inferred).
HEAD_UNRESOLVED = "unresolved"

#: objects.disposition mirror for the *artifact's* availability. The
#: registry disposition gates kernel access to the control object itself —
#: it is not the answer-selection label (that lives in ``current_state``),
#: so terminal-but-inspectable states stay ``active``.
_OBJECT_DISPOSITION = {
    "recorded": "held",
    "active": "active",
    "superseded": "active",
    "corrected": "active",
    "retracted": "active",
    "archived": "archived",
    "erased": "erased",
}

_MAX_HEAD_LEN = 128


# ----------------------------------------------------------------------
# storage seam — repos_v5 when the storage worker's module covers the
#: table, an identical allowlist shim until then (same frozen columns,
#: same parameterized statements; every access funnels through here).
# ----------------------------------------------------------------------

_COLUMNS = frozenset({
    "source_id", "namespace", "control_version", "mutation_head",
    "disposition", "superseded_by", "effective_at", "known_at",
    "valid_from", "valid_to", "updated_at", "producer",
})


def _v5_repo() -> Any:
    """repos_v5 when it registers ``source_state``; else ``None``."""
    cols = getattr(_repos_v5, "_COLUMNS", None) if _repos_v5 is not None else None
    if isinstance(cols, Mapping) and TABLE in cols:
        return _repos_v5
    return None


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return dict(zip(cols, rows[0])) if rows else None


def _check_cols(row: Mapping[str, Any]) -> None:
    bad = set(row) - _COLUMNS
    if bad:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"source_state: unknown columns {sorted(bad)}"
        )


def _insert_row(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    repo = _v5_repo()
    if repo is not None:
        repo.insert(conn, TABLE, dict(row))
        return
    _check_cols(row)
    names = sorted(row)
    conn.execute(
        f"INSERT INTO {TABLE} ({', '.join(names)}) "
        f"VALUES ({', '.join('?' for _ in names)})",
        [row[n] for n in names],
    )


def _fetch_row(conn: sqlite3.Connection, source_id: str) -> Optional[dict[str, Any]]:
    repo = _v5_repo()
    if repo is not None:
        return repo.get(conn, TABLE, {"source_id": source_id})
    return _row(
        conn.execute(
            f"SELECT * FROM {TABLE} WHERE source_id = ?", (source_id,)
        )
    )


def _cas_update(
    conn: sqlite3.Connection,
    source_id: str,
    expected_control_version: int,
    set_: Mapping[str, Any],
) -> int:
    """Guarded UPDATE — the WHERE clause *is* the compare-and-set."""
    repo = _v5_repo()
    if repo is not None:
        return repo.update(
            conn,
            TABLE,
            dict(set_),
            {"source_id": source_id, "control_version": expected_control_version},
        )
    _check_cols(set_)
    set_sql = ", ".join(f"{k} = ?" for k in sorted(set_))
    params = [set_[k] for k in sorted(set_)]
    params += [source_id, expected_control_version]
    return conn.execute(
        f"UPDATE {TABLE} SET {set_sql}"
        " WHERE source_id = ? AND control_version = ?",
        params,
    ).rowcount


def _delete_row(conn: sqlite3.Connection, source_id: str) -> int:
    """Row removal exists only for tests/recovery — never a lifecycle op."""
    repo = _v5_repo()
    if repo is not None:
        return repo.delete(conn, TABLE, {"source_id": source_id})
    return conn.execute(
        f"DELETE FROM {TABLE} WHERE source_id = ?", (source_id,)
    ).rowcount


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _require_table(conn: sqlite3.Connection) -> None:
    """Fail closed when the artifact's table was never installed."""
    if not has_table(conn, TABLE):
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            "source_state table absent — run install_schema/migration first",
        )


def artifact_id(source_id: str) -> str:
    """The registered ``objects.object_id`` for a source's state artifact.

    Readable ``source_state:<sid>`` form when it fits the id alphabet;
    a deterministic digest-mangled form otherwise. Derived, never minted,
    so recovery recomputes it.
    """
    require_id(source_id, "source_id")
    base = f"source_state:{source_id}"
    if len(base) <= 120:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"ss1.{digest[:40]}"


def _event_us(store: Any) -> int:
    """Logical event seq: store's monotone clock when available."""
    if store is not None:
        return int(store.next_event_us())
    return _time.now_us()


def _rfc3339(us: int) -> str:
    return _time.rfc3339(int(us))


def _to_us(value: Any, field_name: str) -> int:
    """Normalize a time argument: int µs or RFC3339 text → µs."""
    if isinstance(value, bool):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{field_name} must be RFC3339 or int µs"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _time.parse_rfc3339(value)
    raise VerbatimError(
        ErrorCode.VALIDATION, f"{field_name} must be RFC3339 or int µs"
    )


def _norm_time(value: Any, field_name: str) -> Optional[str]:
    """Normalize an optional persisted time field to RFC3339 text."""
    if value is None:
        return None
    return _rfc3339(_to_us(value, field_name))


def parse_head(head: Optional[str]) -> Optional[int]:
    """Numeric head token → int revision; ``unresolved``/other → None."""
    if head is None or head == HEAD_UNRESOLVED:
        return None
    try:
        rev = int(head)
    except (TypeError, ValueError):
        return None
    return rev if rev >= 1 else None


def _norm_head(value: Any, field_name: str = "mutation_head") -> str:
    if value is None:
        return HEAD_UNRESOLVED
    if isinstance(value, bool):
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid {field_name}")
    if isinstance(value, int):
        if value < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{field_name} revision must be >= 1"
            )
        return str(value)
    if not isinstance(value, str) or not value:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid {field_name}")
    if value != HEAD_UNRESOLVED and len(value) > _MAX_HEAD_LEN:
        raise VerbatimError(ErrorCode.VALIDATION, f"{field_name} too long")
    return value


def _head_revision(conn: sqlite3.Connection, source_id: str) -> Optional[int]:
    """Latest *persisted* source revision — used only as the initial bind
    at ensure/adopt time inside the caller's transaction, never as an
    answer-selection rule (§14.3: not blindly ``max(revision)``)."""
    if not has_table(conn, "source_revisions"):
        return None
    row = conn.execute(
        "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


# ----------------------------------------------------------------------
# SourceState — the row shape (times are RFC3339 text, matching schema)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SourceState:
    """One committed control record for a logical memory (§14.3)."""

    source_id: str
    namespace: str
    control_version: int
    mutation_head: str
    disposition: str
    known_at: str
    updated_at: str
    producer: str
    superseded_by: Optional[str] = None
    effective_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    @property
    def head_revision(self) -> Optional[int]:
        return parse_head(self.mutation_head)

    @property
    def tombstoned(self) -> bool:
        return self.disposition == "erased"

    def pending_scheduled(self, now_us: Optional[int] = None) -> bool:
        """A supersede whose boundary has not yet been reached."""
        if self.disposition != "superseded" or self.effective_at is None:
            return False
        now = _time.now_us() if now_us is None else int(now_us)
        return _to_us(self.effective_at, "effective_at") > now

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": SOURCE_STATE_KIND,
            "source_id": self.source_id,
            "namespace": self.namespace,
            "control_version": self.control_version,
            "mutation_head": self.mutation_head,
            "disposition": self.disposition,
            "superseded_by": self.superseded_by,
            "effective_at": self.effective_at,
            "known_at": self.known_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "updated_at": self.updated_at,
            "producer": self.producer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SourceState":
        return cls(
            source_id=str(d["source_id"]),
            namespace=str(d["namespace"]),
            control_version=int(d["control_version"]),
            mutation_head=str(d["mutation_head"]),
            disposition=str(d["disposition"]),
            superseded_by=d.get("superseded_by"),
            effective_at=d.get("effective_at"),
            known_at=str(d["known_at"]),
            valid_from=d.get("valid_from"),
            valid_to=d.get("valid_to"),
            updated_at=str(d["updated_at"]),
            producer=str(d["producer"]),
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SourceState":
        return cls.from_dict(row)


@dataclass(frozen=True)
class CurrentState:
    """Read-time evaluation of a source's control record (V5-14.08/12).

    ``effective_revision`` is the source revision answering at ``at_time``
    under the doc visible at ``known_at``; ``current`` marks whether it may
    compete as the current answer (V5-14.14 — superseded/retracted/etc.
    revisions may only appear as labeled historical/contrary evidence).
    """

    found: bool
    source_id: str
    namespace: Optional[str] = None
    control_version: Optional[int] = None
    mutation_head: Optional[str] = None
    disposition: Optional[str] = None
    effective_revision: Optional[int] = None
    predecessor_revision: Optional[int] = None
    label: str = "unknown"
    current: bool = False
    pending: Optional[dict[str, Any]] = None
    at_time: Optional[str] = None
    known_at: Optional[str] = None
    detail: str = ""
    state: Optional[SourceState] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "source_id": self.source_id,
            "namespace": self.namespace,
            "control_version": self.control_version,
            "mutation_head": self.mutation_head,
            "disposition": self.disposition,
            "effective_revision": self.effective_revision,
            "predecessor_revision": self.predecessor_revision,
            "label": self.label,
            "current": self.current,
            "pending": self.pending,
            "at_time": self.at_time,
            "known_at": self.known_at,
            "detail": self.detail,
        }


# ----------------------------------------------------------------------
# doc serialization / integrity (the `source_revision_transition` encoding)
# ----------------------------------------------------------------------


def serialize_doc(doc: Mapping[str, Any]) -> str:
    """Canonical ``source_state/v1`` encoding (sorted-keys JSON)."""
    if doc.get("kind") != SOURCE_STATE_KIND:
        raise VerbatimError(
            ErrorCode.VALIDATION, "source_state doc kind mismatch"
        )
    return json_dumps(dict(doc))


def doc_digest(doc: Mapping[str, Any]) -> str:
    """``sha256:`` over the canonical encoding — the doc's integrity tag."""
    return "sha256:" + hashlib.sha256(serialize_doc(doc).encode("utf-8")).hexdigest()


def parse_doc(text: Any) -> dict[str, Any]:
    """Inverse of ``serialize_doc``; rejects foreign doc kinds."""
    doc = safe_json_loads(text) if isinstance(text, str) else dict(text)
    if not isinstance(doc, dict) or doc.get("kind") != SOURCE_STATE_KIND:
        raise VerbatimError(
            ErrorCode.INTEGRITY, "not a source_state/v1 doc"
        )
    return doc


def _write_doc(
    conn: sqlite3.Connection,
    store: Any,
    doc: dict[str, Any],
    *,
    disposition: str,
) -> int:
    """Persist ``doc`` as the artifact's next object revision.

    Doc revision == ``control_version + 1`` (see module docstring). The
    registry row is created on first write; afterwards only
    ``current_revision``/``disposition`` advance.
    """
    oid = artifact_id(doc["source_id"])
    revision = int(doc["control_version"]) + 1
    obj = repos_v4.get(
        conn, "objects", {"kind": OBJECT_KIND, "object_id": oid}
    )
    if obj is None:
        repos_v4.insert(
            conn,
            "objects",
            {
                "object_id": oid,
                "kind": OBJECT_KIND,
                "scope_id": doc["namespace"],
                "current_revision": revision,
                "disposition": _OBJECT_DISPOSITION[disposition],
                "created_event": _event_us(store),
            },
        )
    else:
        repos_v4.update(
            conn,
            "objects",
            {
                "current_revision": revision,
                "disposition": _OBJECT_DISPOSITION[disposition],
            },
            {"kind": OBJECT_KIND, "object_id": oid},
        )
    doc["registry_revision"] = revision
    repos_v4.insert(
        conn,
        "object_revisions",
        {
            "kind": OBJECT_KIND,
            "object_id": oid,
            "revision": revision,
            "digest": doc_digest(doc),
            "recorded_from": _event_us(store),
            "producer_ref": doc.get("producer") or PRODUCER_ID,
            "metadata_json": serialize_doc(doc),
        },
    )
    return revision


def _docs(
    conn: sqlite3.Connection, source_id: str, *, verify: bool = True
) -> list[dict[str, Any]]:
    """All committed docs, ascending control_version, digest-verified."""
    oid = artifact_id(source_id)
    rows = repos_v4.query(
        conn,
        "object_revisions",
        {"kind": OBJECT_KIND, "object_id": oid},
        order="revision",
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        doc = parse_doc(r.get("metadata_json") or "{}")
        if verify and r.get("digest") and r["digest"] != doc_digest(doc):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"source_state {source_id} doc rev {r['revision']} fails "
                "integrity check",
            )
        out.append(doc)
    out.sort(key=lambda d: int(d["control_version"]))
    return out


def _record_edges(
    conn: sqlite3.Connection,
    doc: Mapping[str, Any],
    *,
    predecessor_doc_rev: Optional[int],
) -> None:
    """Provenance edges for one committed doc (V5-14.09, V3-17.02).

    Child = this doc; parents = the predecessor state doc (chain) and the
    ``source_revision`` the head binds (evidence anchor) when numeric.
    """
    oid = artifact_id(str(doc["source_id"]))
    child = (OBJECT_KIND, oid, int(doc["registry_revision"]))
    seq = 0
    if predecessor_doc_rev is not None:
        record_edge(
            conn,
            child,
            (OBJECT_KIND, oid, predecessor_doc_rev),
            "source_state",
            PRODUCER_ID,
            str(doc["namespace"]),
            seq,
        )
        seq += 1
    head_rev = parse_head(doc.get("mutation_head"))
    if head_rev is not None:
        record_edge(
            conn,
            child,
            ("source_revision", str(doc["source_id"]), head_rev),
            "source_state",
            PRODUCER_ID,
            str(doc["namespace"]),
            seq,
        )


# ----------------------------------------------------------------------
# registration / migration / recovery handlers (V5-03.06, V5-14.09)
# ----------------------------------------------------------------------


def install_schema(conn: sqlite3.Connection) -> None:
    """Idempotent migration handler: create the frozen §3 table.

    Identical DDL to the storage worker's v4→v5 migration — whichever runs
    first wins, the other no-ops. Records its own install marker in
    ``meta`` for recovery diagnostics (schema-version bookkeeping itself
    stays with ``storage/migrations.py``).
    """
    for stmt in DDL_SOURCE_STATE.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt.strip())
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (
            "source_state.install",
            json_dumps(
                {
                    "kind": SOURCE_STATE_KIND,
                    "ddl_sha256": hashlib.sha256(
                        DDL_SOURCE_STATE.encode("utf-8")
                    ).hexdigest(),
                }
            ),
        ),
    )


def register_producer(conn: sqlite3.Connection) -> str:
    """Idempotent producer-manifest registration (V5-14.09 provenance)."""
    row = repos_v4.get(
        conn, "producer_manifests", {"producer_id": PRODUCER_ID}
    )
    if row is None:
        descriptor = {
            "producer_id": PRODUCER_ID,
            "kind": OBJECT_KIND,
            "doc": SOURCE_STATE_KIND,
            "effect": EFFECT_KIND,
            "dispositions": sorted(DISPOSITIONS),
        }
        digest = "sha256:" + hashlib.sha256(
            json_dumps(descriptor).encode("utf-8")
        ).hexdigest()
        repos_v4.insert(
            conn,
            "producer_manifests",
            {
                "producer_id": PRODUCER_ID,
                "kind": OBJECT_KIND,
                "artifact_digest": digest,
                "rubric_digest": "sha256:" + hashlib.sha256(
                    json_dumps({"rubric": _POLICY_VERSION}).encode("utf-8")
                ).hexdigest(),
                "config_digest": digest,
                "schema_version": 5,
                "license_ref": None,
                "health": "available",
                "registered_us": _time.now_us(),
            },
        )
    return PRODUCER_ID


def verify(conn: sqlite3.Connection, source_id: str) -> int:
    """Integrity handler: re-check every persisted doc digest.

    Returns the number of verified docs; a tampered doc raises
    ``STORE_CORRUPT``.
    """
    _require_table(conn)
    return len(_docs(conn, source_id, verify=True))


def recover(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    store: Any = None,
) -> Optional[SourceState]:
    """Recovery handler: rebuild the CAS row from the doc history.

    The ``object_revisions`` docs are the durable artifact; the
    ``source_state`` row is their latest-state projection. A missing row is
    rebuilt; a row disagreeing with the latest verified doc raises
    ``INTEGRITY`` (recovery reports the divergence — it never silently
    overwrites control data, V5-03.06).
    """
    _require_table(conn)
    docs = _docs(conn, source_id, verify=True)
    if not docs:
        return None
    latest = docs[-1]
    rebuilt = SourceState.from_dict(latest)
    row = _fetch_row(conn, source_id)
    if row is None:
        _insert_row(
            conn, {k: v for k, v in rebuilt.to_dict().items() if k != "kind"}
        )
        return rebuilt
    current = SourceState.from_row(row)
    if current != rebuilt:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"source_state row for {source_id} diverges from doc history "
            f"(row cv={current.control_version} {current.disposition} vs "
            f"doc cv={rebuilt.control_version} {rebuilt.disposition})",
        )
    return current


# ----------------------------------------------------------------------
# reads
# ----------------------------------------------------------------------


def get_state(
    conn: sqlite3.Connection, source_id: str
) -> Optional[SourceState]:
    """Latest committed control record, or ``None`` when unregistered."""
    require_id(source_id, "source_id")
    _require_table(conn)
    row = _fetch_row(conn, source_id)
    return SourceState.from_row(row) if row is not None else None


def get(
    conn: sqlite3.Connection, source_id: str
) -> Optional[dict[str, Any]]:
    """Dict-shaped reader — the seam ``memory/controls.py`` probes for
    (``get``/``current``/``read``/``state``). Returns the raw contract row
    rather than a ``SourceState``; ``None`` when unregistered or when the
    artifact's table is absent (the facade then falls back cleanly)."""
    require_id(source_id, "source_id")
    if not has_table(conn, TABLE):
        return None
    return _fetch_row(conn, source_id)


def history(
    conn: sqlite3.Connection, source_id: str, *, verify_docs: bool = True
) -> list[dict[str, Any]]:
    """Disposition/revision history for inspect (V5-15.01).

    One entry per committed control version, ascending — each the full
    serialized ``source_state/v1`` doc (state fields + ``change``,
    ``predecessor_head``, ``producer``, ``actor``, ``operation_id``,
    ``recorded_event``, ``registry_revision``).
    """
    require_id(source_id, "source_id")
    _require_table(conn)
    return _docs(conn, source_id, verify=verify_docs)


def _window(
    doc: Mapping[str, Any], t_us: int, head: Optional[int]
) -> tuple[str, bool]:
    """Declared valid-window check for the effective revision."""
    vf = doc.get("valid_from")
    vt = doc.get("valid_to")
    if vf is not None and t_us < _to_us(vf, "valid_from"):
        return "scheduled", False
    if vt is not None and t_us >= _to_us(vt, "valid_to"):
        return "expired", False
    return "active", True


def _evaluate(
    doc: Mapping[str, Any],
    prev_doc: Optional[Mapping[str, Any]],
    t_us: int,
    k_us: Optional[int],
) -> CurrentState:
    """Validity evaluation of one doc at valid-time ``t_us``.

    Bitemporal axes (V5-14.08): ``known_at`` selects *which* committed doc
    is visible (system/record time); ``at_time`` selects the answer inside
    that doc's declared validity semantics (valid time).
    """
    disp = str(doc["disposition"])
    head = parse_head(doc.get("mutation_head"))
    pred = parse_head(doc.get("predecessor_head"))
    base = dict(
        found=True,
        source_id=str(doc["source_id"]),
        namespace=str(doc["namespace"]),
        control_version=int(doc["control_version"]),
        mutation_head=str(doc["mutation_head"]),
        disposition=disp,
        predecessor_revision=pred,
        at_time=_rfc3339(t_us),
        known_at=_rfc3339(k_us) if k_us is not None else None,
        state=SourceState.from_dict(doc),
    )
    if disp == "superseded":
        # effective_at is guaranteed by transition(); a doc without it
        # (hand-built/legacy) degenerates to an immediate boundary at its
        # recorded time rather than failing the read.
        boundary = (
            _to_us(doc["effective_at"], "effective_at")
            if doc.get("effective_at") is not None
            else _to_us(doc["known_at"], "known_at")
        )
        if t_us < boundary:
            # Scheduled (V5-14.12): the predecessor still answers, under
            # its own doc's declared window, until the boundary.
            label, current = (
                _window(prev_doc, t_us, pred)
                if prev_doc is not None
                else ("active", True)
            )
            return CurrentState(
                effective_revision=pred,
                label=label,
                current=current,
                pending={
                    "successor": doc["mutation_head"],
                    "effective_at": doc["effective_at"],
                    "control_version": int(doc["control_version"]),
                },
                detail="supersede scheduled; predecessor remains current "
                "until effective_at",
                **base,
            )
        label, current = _window(doc, t_us, head)
        return CurrentState(
            effective_revision=head,
            label=label,
            current=current,
            detail="successor effective",
            **base,
        )
    if disp == "corrected":
        # Correction lands at record time (no effective_at — V5-14.13);
        # valid_from/valid_to bound the *predecessor's* retracted interval.
        if doc.get("superseded_by") is not None:
            return CurrentState(
                effective_revision=head,
                label="active",
                current=True,
                detail="correction successor is the current answer",
                **base,
            )
        return CurrentState(
            effective_revision=head,
            label="corrected",
            current=False,
            detail="assertion retracted by correction; no successor",
            **base,
        )
    if disp == "active":
        label, current = _window(doc, t_us, head)
        return CurrentState(
            effective_revision=head, label=label, current=current, **base
        )
    # recorded / retracted / archived / erased — inspectable, never the
    # current answer (V5-14.14).
    detail = {
        "recorded": "registered, not admitted",
        "retracted": "retracted by owner",
        "archived": "retired, retained",
        "erased": "erasure tombstone — fenced",
    }.get(disp, disp)
    return CurrentState(
        effective_revision=head,
        label=disp,
        current=False,
        detail=detail,
        **base,
    )


def current_state(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    at_time: Any = None,
    known_at: Any = None,
) -> CurrentState:
    """Evaluate the source's current answer at ``at_time``.

    ``at_time`` — valid time (RFC3339 or int µs; default now). A future
    ``effective_at`` NEVER makes the successor current early: before the
    boundary the predecessor answers with ``pending`` set (V5-14.12).
    ``known_at`` — system time (same encodings; default latest): evaluates
    the newest doc whose ``known_at`` ≤ the bound, returning prior
    information under current authorization (V5-14.08).
    """
    require_id(source_id, "source_id")
    _require_table(conn)
    t = _to_us(at_time, "at_time") if at_time is not None else _time.now_us()
    k = _to_us(known_at, "known_at") if known_at is not None else None
    docs = _docs(conn, source_id)
    sel: Optional[dict[str, Any]] = None
    for d in docs:
        if k is None or _to_us(d["known_at"], "known_at") <= k:
            sel = d
    if sel is None:
        return CurrentState(
            found=False,
            source_id=source_id,
            label="unknown",
            current=False,
            at_time=_rfc3339(t),
            known_at=_rfc3339(k) if k is not None else None,
            detail="no control record known at the requested time",
        )
    idx = docs.index(sel)
    prev = docs[idx - 1] if idx > 0 else None
    return _evaluate(sel, prev, t, k)


# ----------------------------------------------------------------------
# creation / adoption (the atomic add binding — V5-14.09)
# ----------------------------------------------------------------------


def ensure_state(
    conn: sqlite3.Connection,
    source_id: str,
    namespace: str,
    *,
    head: Any = None,
    disposition: str = "active",
    producer: str = PRODUCER_ID,
    actor: Optional[str] = None,
    operation_id: Optional[str] = None,
    known_at: Any = None,
    store: Any = None,
) -> SourceState:
    """Idempotent create of the control record at ``control_version`` 0.

    The initial-add binding (V5-14.09): writes the ``source_state`` row,
    the registered ``objects``/``object_revisions`` artifact doc, the
    producer manifest, and the provenance edge to the bound source
    revision — all inside the caller's transaction.

    ``head`` is the approved head revision for CAS binding (int or token);
    omitted, it binds the latest persisted revision *inside this
    transaction* (the just-appended revision for an atomic add), else
    ``unresolved``. A second call returns the existing record unchanged —
    a tombstoned record is returned as-is and never reactivated — but a
    conflicting ``namespace`` is an ``INTEGRITY`` failure (a state row
    cannot silently switch partitions).
    """
    require_id(source_id, "source_id")
    require_id(namespace, "namespace")
    _require_table(conn)
    register_producer(conn)

    if disposition in CHANGE_ALIASES:
        disposition = CHANGE_ALIASES[disposition]
    if disposition not in TRANSITION_DISPOSITIONS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"initial disposition must be one of "
            f"{sorted(TRANSITION_DISPOSITIONS)}",
        )
    if not isinstance(producer, str) or not producer:
        raise VerbatimError(ErrorCode.VALIDATION, "producer required")

    existing = _fetch_row(conn, source_id)
    if existing is not None:
        state = SourceState.from_row(existing)
        if state.namespace != namespace:
            raise VerbatimError(
                ErrorCode.INTEGRITY,
                f"source_state for {source_id} already bound to namespace "
                f"{state.namespace!r}",
            )
        return state
    if _docs(conn, source_id, verify=False):
        # Registry docs exist without a CAS row — that is a recovery
        # situation, not a fresh create; fail typed instead of colliding
        # on the doc primary key.
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"source_state docs exist for {source_id} but the row is "
            "absent — use recover()",
        )

    if head is None:
        rev = _head_revision(conn, source_id)
        head_token = str(rev) if rev is not None else HEAD_UNRESOLVED
    else:
        head_token = _norm_head(head)
    now = _time.now_us()
    known = (
        _rfc3339(_to_us(known_at, "known_at"))
        if known_at is not None
        else _rfc3339(now)
    )
    state = SourceState(
        source_id=source_id,
        namespace=namespace,
        control_version=0,
        mutation_head=head_token,
        disposition=disposition,
        known_at=known,
        updated_at=_rfc3339(now),
        producer=producer,
    )
    _insert_row(conn, {k: v for k, v in state.to_dict().items() if k != "kind"})
    doc = state.to_dict() | {
        "change": "adopt" if disposition == "recorded" else "create",
        "predecessor_head": None,
        "actor": actor,
        "operation_id": operation_id,
        "recorded_event": _event_us(store),
        "effect": EFFECT_KIND,
    }
    _write_doc(conn, store, doc, disposition=disposition)
    _record_edges(conn, doc, predecessor_doc_rev=None)
    return state


def adopt(
    conn: sqlite3.Connection,
    source_id: str,
    namespace: str,
    *,
    producer: str = PRODUCER_ID,
    store: Any = None,
) -> SourceState:
    """Migration/adoption path for pre-v5 sources (V5-14.16, §08).

    Unknown legacy lifecycle stays ``recorded`` with an ``unresolved``
    head — never inferred approval. Explicitly equivalent to
    ``ensure_state(..., disposition="recorded", head=HEAD_UNRESOLVED)``;
    kept as a named operation so callers cannot silently adopt as active.
    """
    return ensure_state(
        conn,
        source_id,
        namespace,
        head=HEAD_UNRESOLVED,
        disposition="recorded",
        producer=producer,
        store=store,
    )


__all__ = [
    "CHANGE_ALIASES",
    "DISPOSITIONS",
    "DDL_SOURCE_STATE",
    "EFFECT_KIND",
    "HEAD_UNRESOLVED",
    "OBJECT_KIND",
    "PRODUCER_ID",
    "SOURCE_STATE_KIND",
    "TABLE",
    "TRANSITION_DISPOSITIONS",
    "CurrentState",
    "SourceState",
    "adopt",
    "artifact_id",
    "current_state",
    "doc_digest",
    "ensure_state",
    "get",
    "get_state",
    "history",
    "install_schema",
    "parse_doc",
    "parse_head",
    "recover",
    "register_producer",
    "serialize_doc",
    "verify",
]
