"""Repositories: the only code that speaks SQL for each table (SPEC §19-20).

Mutable rows never escape as objects — reads return plain ``dict`` snapshots
(SPEC §8). Methods that mutate take the caller's transaction ``conn`` so a
higher layer can commit source + revision + event + projection atomically
(SPEC §21: "a claim transition atomically updates events, revisions,
interval interpretations, edges, review state, and projection generation").
Read methods without a ``conn`` parameter open their own WAL snapshot.

All SQL is parameterized. Placeholder lists are built from fixed column
allowlists and value counts — never from caller text (SPEC §41).
"""

from __future__ import annotations

import hmac
import sqlite3
from typing import Any, Iterable, Optional, Sequence

from ..core.identity import can_read, scope_key
from ..core.time import wall_us
from ..core.types import (
    EdgeType,
    ErrorCode,
    FeedbackKind,
    Lifecycle,
    Modality,
    Polarity,
    ReviewState,
    Scope,
    SourceEnvelope,
    TaskKind,
    TimeInterval,
    VerbatimError,
    Visibility,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from .store import Store

# Event sequence cutoff meaning "+infinity" for known_at queries. Event
# sequences are AUTOINCREMENT ints and never reach this; using a concrete int
# keeps the recorded_from/recorded_until predicate identical between the
# "latest belief" and "belief at K" paths (SPEC §14).
_K_INF = 1 << 62

_EVIDENCE_ROLES = ("primary", "contextual")


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rows = _rows(cur)
    return rows[0] if rows else None


def _json_text(value: Any, field: str) -> Optional[str]:
    """Normalize a JSON-bearing argument to canonical text for storage."""
    if value is None:
        return None
    if isinstance(value, str):
        safe_json_loads(value)  # validates: rejects NaN/dupes/over-depth
        return value
    if isinstance(value, (dict, list)):
        return _dump(value, field)
    raise VerbatimError(ErrorCode.VALIDATION, f"{field} must be dict, json text, or None")


def _json_parse(text: Optional[str]) -> Any:
    return safe_json_loads(text) if isinstance(text, str) else None


def _dump(value: Any, field: str) -> str:
    """Canonical JSON for persistence, as a typed error on bad input."""
    try:
        return json_dumps(value)
    except (TypeError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{field} is not json-serializable: {exc}"
        ) from exc


def _next_event_seq(conn: sqlite3.Connection) -> int:
    """Estimated next event_seq for created_event columns.

    Callers append the transition event inside the same transaction, so the
    estimate materializes as the real sequence number at commit.
    """
    row = conn.execute("SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events").fetchone()
    return int(row[0])


def _enum_value(value: Any, enum_cls: Any, field: str) -> str:
    try:
        return enum_cls(value).value
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid {field}: {value!r}"
        ) from exc


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerbatimError(ErrorCode.VALIDATION, f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerbatimError(ErrorCode.VALIDATION, f"{field} must be an int >= {minimum}")
    return value


def _placeholders(values: Sequence[Any]) -> str:
    return ", ".join("?" for _ in values)


# ----------------------------------------------------------------------
# scope helpers
# ----------------------------------------------------------------------


#: Per-connection cache of confirmed-present schema objects —
#: ``id(conn) -> (conn, {name, ...})``. Holding the connection itself
#: keeps its ``id`` from being recycled while the entry lives (an
#: ``id``-keyed cache without the strong ref could serve a recycled
#: connection the old one's entries). Only True results are memoized:
#: schema objects are created at open/migration time before
#: connections serve queries and are never dropped while a connection
#: lives (the only DROPs sit in open-time migration DDL), so a
#: confirmed table stays present for the connection's lifetime.
#: ``False`` is re-probed every call — a later-created object is seen
#: on the very next check. Bounded: eviction drops the strong ref.
_HAS_TABLE_CACHE: dict = {}
_HAS_TABLE_CACHE_MAX = 64


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    """Whether a schema object (table or view) exists — the single shared
    introspection helper; v2-only relations are absent on v1 stores and
    vice versa, so callers gate on presence rather than schema version."""
    ent = _HAS_TABLE_CACHE.get(id(conn))
    if ent is None or ent[0] is not conn:
        ent = (conn, set())
        _HAS_TABLE_CACHE[id(conn)] = ent
        while len(_HAS_TABLE_CACHE) > _HAS_TABLE_CACHE_MAX:
            _HAS_TABLE_CACHE.pop(next(iter(_HAS_TABLE_CACHE)))
    cache = ent[1]
    if name in cache:
        return True
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    if row is not None:
        cache.add(name)
        return True
    return False


def scope_id_for(store: Store, scope: Scope) -> str:
    """Stable scope identifier — delegates to ``core.identity.scope_key``.

    One canonical derivation for every scope-keyed table (scopes, sources,
    jobs, events, consents). ``store`` is accepted for call-site symmetry and
    ignored: the id is a deterministic digest of the scope tuple, which is
    itself stored in plaintext on the row, so keying it adds no secrecy.
    A visibility change still yields a distinct partition (SPEC §9).
    """
    return scope_key(scope)


def ensure_scope(store: Store, conn: sqlite3.Connection, scope: Scope) -> str:
    """Upsert the scope row inside the caller's write tx; returns scope_id."""
    sid = scope_id_for(store, scope)
    conn.execute(
        "INSERT OR IGNORE INTO scopes"
        "(scope_id, profile_id, principal_id, workspace_id, conversation_id, visibility)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            sid,
            scope.profile_id,
            scope.principal_id,
            scope.workspace_id,
            scope.conversation_id,
            scope.visibility.value,
        ),
    )
    return sid


def readable_scope_ids(conn: sqlite3.Connection, reader: Scope) -> list[str]:
    """Scope partitions ``reader`` may see under ``can_read`` (SPEC §9).

    Candidate generation is scoped before ranking; retrieval layers call
    this first and only then query inside the returned partitions.
    """
    rows = conn.execute(
        "SELECT scope_id, profile_id, principal_id, workspace_id,"
        " conversation_id, visibility FROM scopes"
    ).fetchall()
    out: list[str] = []
    for sid, profile_id, principal_id, workspace_id, conversation_id, vis in rows:
        owner = Scope(
            profile_id=profile_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            visibility=Visibility(vis),
        )
        if can_read(reader, owner):
            out.append(sid)
    return out


# ----------------------------------------------------------------------
# sources and spans
# ----------------------------------------------------------------------


class SourcesRepo:
    """Stable source identity plus immutable revision bytes (SPEC §10, §19)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def insert(
        self,
        envelope: SourceEnvelope,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> tuple[str, bool]:
        """Persist source + revision atomically; returns ``(source_id, created)``.

        Dedup key is ``(origin, external_id)`` when ``external_id`` is
        present; ``created`` is False only when the exact revision already
        exists. A new revision on a known source is an edit — prior revisions
        keep their offsets (SPEC §10). Identical dedup keys carrying
        different bytes are a conflict, never a silent merge. Opens its own
        tx unless ``conn`` is supplied — ingest needs source insert + job
        creation in ONE transaction (SPEC §21).
        """
        if conn is None:
            with self._store.tx() as owned:
                return self._insert(owned, envelope)
        return self._insert(conn, envelope)

    def _insert(self, conn: sqlite3.Connection, envelope: SourceEnvelope) -> tuple[str, bool]:
        store = self._store
        payload = bytes(envelope.payload)
        payload_hmac = store.hmac(payload)
        if True:
            scope_id = ensure_scope(store, conn, envelope.scope)
            source_id: Optional[str] = None
            if envelope.external_id is not None:
                # Partition-local dedup (V3-12.04): the same host key in
                # a different scope is an independent source, never a
                # revision grafted across the partition boundary.
                found = conn.execute(
                    "SELECT source_id FROM sources"
                    " WHERE scope_id = ? AND origin = ? AND external_id = ?",
                    (scope_id, envelope.origin, envelope.external_id),
                ).fetchone()
                if found is not None:
                    source_id = found[0]

            if source_id is not None:
                existing = conn.execute(
                    "SELECT payload_hmac FROM source_revisions"
                    " WHERE source_id = ? AND revision = ?",
                    (source_id, envelope.revision),
                ).fetchone()
                if existing is not None:
                    if bytes(existing[0]) != payload_hmac:
                        raise VerbatimError(
                            ErrorCode.VALIDATION,
                            "conflicting payload for dedup key "
                            "(origin, external_id, revision)",
                        )
                    return source_id, False
                self._insert_revision(conn, source_id, envelope, payload, payload_hmac)
                return source_id, True

            source_id = envelope.source_id or new_id()
            require_id(source_id, "source_id")
            conn.execute(
                "INSERT INTO sources"
                "(source_id, origin, external_id, source_kind, scope_id,"
                " speaker_id, created_us)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    envelope.origin,
                    envelope.external_id,
                    envelope.source_kind.value,
                    scope_id,
                    envelope.speaker_id,
                    envelope.captured_us,
                ),
            )
            self._insert_revision(conn, source_id, envelope, payload, payload_hmac)
            return source_id, True

    @staticmethod
    def _insert_revision(
        conn: sqlite3.Connection,
        source_id: str,
        envelope: SourceEnvelope,
        payload: bytes,
        payload_hmac: bytes,
    ) -> None:
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us, captured_us,"
            " timezone, provenance, metadata_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                envelope.revision,
                payload,
                payload_hmac,
                envelope.event_us,
                envelope.captured_us,
                envelope.timezone,
                envelope.provenance.value,
                _dump(envelope.metadata, "metadata"),
            ),
        )

    def get(self, source_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT source_id, origin, external_id, source_kind, scope_id,"
                    " speaker_id, created_us FROM sources WHERE source_id = ?",
                    (source_id,),
                )
            )

    def get_revision(self, source_id: str, revision: int) -> Optional[dict[str, Any]]:
        """Revision metadata; payload bytes excluded (use ``payload``)."""
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT source_id, revision, length(payload) AS payload_bytes,"
                    " hex(payload_hmac) AS payload_hmac_hex, event_us, captured_us,"
                    " timezone, provenance, metadata_json"
                    " FROM source_revisions WHERE source_id = ? AND revision = ?",
                    (source_id, revision),
                )
            )

    def payload(self, source_id: str, revision: int) -> Optional[bytes]:
        """Exact accepted bytes; None when absent or purged (SPEC §40).

        The persisted ``payload_hmac`` is re-verified on every read: a row
        whose bytes no longer match their digest is corruption, never
        content — ``STORE_CORRUPT``, not a silent byte serving. A NULL
        digest marks a legacy row written before the column existed; it is
        returned unverified rather than condemned (nothing to check
        against). A purged revision keeps ``payload = X''`` *with* a
        matching resealed digest, so it still verifies and returns ``b''``.
        """
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT payload, payload_hmac FROM source_revisions"
                " WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        payload = bytes(row[0])
        if row[1] is not None and not hmac.compare_digest(
            self._store.hmac(payload), bytes(row[1])
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"source {source_id}@{revision} payload fails integrity check",
            )
        return payload

    def payload_many(
        self,
        pairs: Iterable[tuple],
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> tuple[dict, list]:
        """Batched verified reads — identical per-row discipline as
        ``payload()`` under one snapshot.

        Returns ``(verified, corrupt)``: ``verified`` maps each retained
        ``(source_id, revision)`` to its exact accepted bytes (a purged
        revision still verifies and maps to ``b''``); absent rows and
        NULL payloads are simply missing, matching ``payload()``'s
        ``None``. ``corrupt`` lists the pairs whose stored bytes fail
        their persisted digest — the caller decides whether corruption
        is an error (``STORE_CORRUPT``) or a per-row withhold, exactly
        the policy it applies to ``payload()`` today. Every returned byte
        string passed the same HMAC check; nothing is served unverified.

        ``conn`` joins the caller's snapshot instead of opening a new one
        — the same one-snapshot-per-query shape the read path requires.
        """
        wanted = [(str(s), int(r)) for s, r in pairs]
        if conn is not None:
            return self._payload_many(conn, wanted)
        with self._store.read() as conn:
            return self._payload_many(conn, wanted)

    def _payload_many(
        self, conn: sqlite3.Connection, wanted: list
    ) -> tuple[dict, list]:
        verified: dict = {}
        corrupt: list = []
        for i in range(0, len(wanted), 200):
            chunk = wanted[i : i + 200]
            where = " OR ".join(
                "(source_id = ? AND revision = ?)" for _ in chunk
            )
            params = [v for pair in chunk for v in pair]
            for sid, rev, payload, hm in conn.execute(
                "SELECT source_id, revision, payload, payload_hmac"
                f" FROM source_revisions WHERE {where}",
                params,
            ):
                if payload is None:
                    continue
                blob = bytes(payload)
                if hm is not None and not hmac.compare_digest(
                    self._store.hmac(blob), bytes(hm)
                ):
                    corrupt.append((sid, int(rev)))
                    continue
                verified[(sid, int(rev))] = blob
        return verified, corrupt

    def latest_revision(self, source_id: str) -> Optional[int]:
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else None


class SpansRepo:
    """Exact excerpts with reproducible UTF-8 byte offsets (SPEC §11)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def insert(
        self,
        span_id: str,
        source_id: str,
        revision: int,
        start_byte: int,
        end_byte: int,
        harvester_version: str,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> str:
        """Insert a span; opens its own tx unless ``conn`` is supplied.

        Offsets must satisfy ``0 <= start < end <= len(payload)`` and land on
        UTF-8 character boundaries — verified by a strict decode of the exact
        excerpt, never by guessing (SPEC §11). The excerpt HMAC is
        profile-keyed so the stored fingerprint cannot confirm low-entropy
        text by dictionary attack.
        """
        if conn is None:
            with self._store.tx() as owned:
                return self._insert(
                    owned, span_id, source_id, revision, start_byte, end_byte,
                    harvester_version,
                )
        return self._insert(
            conn, span_id, source_id, revision, start_byte, end_byte,
            harvester_version,
        )

    def _insert(
        self,
        conn: sqlite3.Connection,
        span_id: str,
        source_id: str,
        revision: int,
        start_byte: int,
        end_byte: int,
        harvester_version: str,
    ) -> str:
        require_id(span_id, "span_id")
        _require_str(harvester_version, "harvester_version")
        _require_int(revision, "revision", minimum=1)
        _require_int(start_byte, "start_byte")
        _require_int(end_byte, "end_byte")
        row = conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchone()
        if row is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "source revision not found"
            )
        payload = bytes(row[0])
        if not (0 <= start_byte < end_byte <= len(payload)):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"span offsets [{start_byte}, {end_byte}) outside "
                f"payload length {len(payload)}",
            )
        excerpt = payload[start_byte:end_byte]
        try:
            excerpt.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "span offsets must fall on UTF-8 character boundaries",
            ) from exc
        conn.execute(
            "INSERT INTO spans"
            "(span_id, source_id, revision, start_byte, end_byte, excerpt_hmac,"
            " harvester_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                span_id,
                source_id,
                revision,
                start_byte,
                end_byte,
                self._store.hmac(excerpt),
                harvester_version,
            ),
        )
        return span_id

    def get(self, span_id: str) -> Optional[dict[str, Any]]:
        """Span row (offsets, source refs); None when missing."""
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT span_id, source_id, revision, start_byte, end_byte,"
                " view_id, operation_key"
                " FROM spans WHERE span_id = ?",
                (span_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "span_id": row[0],
            "source_id": row[1],
            "revision": row[2],
            "start_byte": row[3],
            "end_byte": row[4],
            "view_id": row[5],
            "operation_key": row[6],
        }

    def text(self, span_id: str) -> Optional[str]:
        """Decoded excerpt; None when the span or its revision was purged.

        The persisted ``excerpt_hmac`` is re-verified over the exact stored
        slice before decoding — a tampered payload or shifted offsets that
        still decode to valid UTF-8 are corruption (``STORE_CORRUPT``), not
        content. A NULL digest marks a legacy row and is skipped rather
        than condemned. An emptied payload (``X''``) only exists after a
        purge scrub — the excerpt is gone, so the answer is absent rather
        than a digest failure.
        """
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT s.start_byte, s.end_byte, r.payload, s.excerpt_hmac"
                " FROM spans s"
                " JOIN source_revisions r"
                "   ON r.source_id = s.source_id AND r.revision = s.revision"
                " WHERE s.span_id = ?",
                (span_id,),
            ).fetchone()
        if row is None or row[2] is None:
            return None
        payload = bytes(row[2])
        if not payload:
            # ``source_revisions.payload`` is only ever emptied by a purge
            # scrub (span insert requires end_byte <= len(payload), so no
            # span could have been created over empty bytes). The span
            # skeleton survives for lineage; its excerpt is gone.
            return None
        excerpt = payload[row[0] : row[1]]
        if row[3] is not None and not hmac.compare_digest(
            self._store.hmac(excerpt), bytes(row[3])
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"span {span_id} excerpt fails integrity check",
            )
        try:
            return excerpt.decode("utf-8")
        except UnicodeDecodeError as exc:
            # The insert-time boundary check makes this unreachable unless
            # the stored bytes were damaged or rewritten outside the repo.
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT, "span excerpt fails to decode"
            ) from exc


# ----------------------------------------------------------------------
# claims (bitemporal)
# ----------------------------------------------------------------------


class ClaimsRepo:
    """Stable claim identity plus revisioned belief history (SPEC §14-15)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        scope_id: str,
        subject_id: Optional[str],
        predicate: Optional[str],
        conn: sqlite3.Connection,
    ) -> str:
        """Create claim identity; ``created_event`` estimates the next seq."""
        claim_id = new_id()
        conn.execute(
            "INSERT INTO claims"
            "(claim_id, scope_id, subject_id, predicate, created_event, row_version)"
            " VALUES (?, ?, ?, ?, ?, 1)",
            (claim_id, scope_id, subject_id, predicate, _next_event_seq(conn)),
        )
        return claim_id

    def add_revision(
        self,
        claim_id: str,
        state: Lifecycle | str,
        object_json: Any,
        polarity: Polarity | str,
        modality: Modality | str,
        condition_json: Any,
        interpretation_json: Any,
        valid_intervals: list[TimeInterval],
        evidence: list[tuple[str, str]],
        recorded_from: int,
        conn: sqlite3.Connection,
    ) -> int:
        """Append revision N+1 with its intervals and evidence links.

        ``recorded_from`` is an event sequence, not a timestamp (SPEC §14).
        All enum fields are validated before write so unsupported values
        never reach the CHECK constraints.
        """
        state_v = _enum_value(state, Lifecycle, "state")
        pol_v = _enum_value(polarity, Polarity, "polarity")
        mod_v = _enum_value(modality, Modality, "modality")
        _require_int(recorded_from, "recorded_from")
        require_id(claim_id, "claim_id")

        if conn.execute(
            "SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone() is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim not found")

        revision = int(
            conn.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM claim_revisions"
                " WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO claim_revisions"
            "(claim_id, revision, state, object_json, polarity, modality,"
            " condition_json, interpretation_json, recorded_from, recorded_until)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                claim_id,
                revision,
                state_v,
                _json_text(object_json, "object_json"),
                pol_v,
                mod_v,
                _json_text(condition_json, "condition_json"),
                _json_text(interpretation_json, "interpretation_json"),
                recorded_from,
            ),
        )
        for interval_no, iv in enumerate(valid_intervals or ()):
            if not isinstance(iv, TimeInterval):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "valid_intervals must be TimeInterval"
                )
            conn.execute(
                "INSERT INTO valid_intervals"
                "(claim_id, revision, interval_no, from_us, until_us, precision,"
                " timezone, basis, uncertainty_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    claim_id,
                    revision,
                    interval_no,
                    iv.from_us,
                    iv.until_us,
                    iv.precision.value,
                    iv.timezone,
                    iv.basis,
                ),
            )
        for item in evidence or ():
            try:
                span_id, role = item
            except (TypeError, ValueError) as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "evidence entries must be (span_id, role) pairs",
                ) from exc
            if role not in _EVIDENCE_ROLES:
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"invalid evidence role {role!r}"
                )
            conn.execute(
                "INSERT INTO claim_evidence(claim_id, revision, span_id, evidence_role)"
                " VALUES (?, ?, ?, ?)",
                (claim_id, revision, span_id, role),
            )
        conn.execute(
            "UPDATE claims SET row_version = row_version + 1 WHERE claim_id = ?",
            (claim_id,),
        )
        return revision

    _CURRENT_COLS = (
        "c.claim_id, c.scope_id, c.subject_id, c.predicate, c.created_event,"
        " c.row_version, r.revision, r.state, r.object_json, r.polarity,"
        " r.modality, r.condition_json, r.interpretation_json, r.recorded_from,"
        " r.recorded_until"
    )
    _KNOWN_PRED = (
        "r.recorded_from <= ? AND (r.recorded_until IS NULL OR r.recorded_until > ?)"
    )
    _KNOWN_PRED_R2 = (
        "r2.recorded_from <= ? AND (r2.recorded_until IS NULL OR r2.recorded_until > ?)"
    )

    def _detail(self, conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
        claim_id, revision = row["claim_id"], row["revision"]
        row["object"] = _json_parse(row.get("object_json"))
        row["condition"] = _json_parse(row.get("condition_json"))
        row["interpretation"] = _json_parse(row.get("interpretation_json"))
        row["valid_intervals"] = _rows(
            conn.execute(
                "SELECT interval_no, from_us, until_us, precision, timezone, basis,"
                " uncertainty_json FROM valid_intervals"
                " WHERE claim_id = ? AND revision = ? ORDER BY interval_no",
                (claim_id, revision),
            )
        )
        row["evidence"] = _rows(
            conn.execute(
                "SELECT span_id, evidence_role FROM claim_evidence"
                " WHERE claim_id = ? AND revision = ? ORDER BY span_id",
                (claim_id, revision),
            )
        )
        return row

    def current(
        self, claim_id: str, known_at_seq: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        """Latest revision known at cutoff K (SPEC §14).

        A revision counts exactly when ``recorded_from <= K`` and
        ``recorded_until`` is absent or ``> K``; default K is +inf, i.e. the
        latest non-erased recorded belief.
        """
        k = _K_INF if known_at_seq is None else _require_int(known_at_seq, "known_at_seq")
        with self._store.read() as conn:
            row = _row(
                conn.execute(
                    f"SELECT {self._CURRENT_COLS}"
                    " FROM claim_revisions r JOIN claims c ON c.claim_id = r.claim_id"
                    f" WHERE r.claim_id = ? AND {self._KNOWN_PRED}"
                    " ORDER BY r.revision DESC LIMIT 1",
                    (claim_id, k, k),
                )
            )
            return self._detail(conn, row) if row is not None else None

    def list_current(
        self,
        scope_id: str,
        states: tuple[str, ...],
        known_at_seq: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Current-revision rows for a scope restricted to ``states``.

        Scope filtering happens in SQL before any ranking — filtering only
        after global retrieval is forbidden (SPEC §6, §9).
        """
        if not states:
            return []
        state_vals = tuple(_enum_value(s, Lifecycle, "state") for s in states)
        _require_int(limit, "limit", minimum=1)
        k = _K_INF if known_at_seq is None else _require_int(known_at_seq, "known_at_seq")
        ph = _placeholders(state_vals)
        with self._store.read() as conn:
            rows = _rows(
                conn.execute(
                    f"SELECT {self._CURRENT_COLS}"
                    " FROM claims c JOIN claim_revisions r ON r.claim_id = c.claim_id"
                    f" WHERE c.scope_id = ? AND r.state IN ({ph})"
                    f" AND {self._KNOWN_PRED}"
                    " AND NOT EXISTS ("
                    "   SELECT 1 FROM claim_revisions r2"
                    "   WHERE r2.claim_id = r.claim_id AND r2.revision > r.revision"
                    f"   AND {self._KNOWN_PRED_R2}"
                    " )"
                    " ORDER BY c.claim_id LIMIT ?",
                    (scope_id, *state_vals, k, k, k, k, limit),
                )
            )
            return [self._detail(conn, row) for row in rows]

    def set_recorded_until(
        self,
        claim_id: str,
        revision: int,
        until_seq: int,
        conn: sqlite3.Connection,
    ) -> None:
        """Close a revision's recorded-time bound at ``until_seq``."""
        _require_int(until_seq, "until_seq")
        cur = conn.execute(
            "UPDATE claim_revisions SET recorded_until = ?"
            " WHERE claim_id = ? AND revision = ?",
            (until_seq, claim_id, revision),
        )
        if cur.rowcount == 0:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "claim revision not found"
            )

    def get(self, claim_id: str) -> Optional[dict[str, Any]]:
        """The stable claim row (identity fields, not a revision)."""
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT claim_id, scope_id, subject_id, predicate,"
                    " created_event, row_version FROM claims WHERE claim_id = ?",
                    (claim_id,),
                )
            )


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------


class EventsRepo:
    """Append-only transition journal (SPEC §20).

    ``recorded_us`` is a nondecreasing logical clock from
    ``Store.next_event_us``; ``observed_wall_us`` keeps the real wall-clock
    reading so clock regressions stay visible instead of being hidden.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def append(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        kind: str,
        actor_id: str,
        payload: dict[str, Any],
        policy_version: str,
    ) -> int:
        _require_str(kind, "kind")
        _require_str(actor_id, "actor_id")
        _require_str(policy_version, "policy_version")
        if not isinstance(payload, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "payload must be a dict")
        cur = conn.execute(
            "INSERT INTO events"
            "(event_id, scope_id, kind, actor_id, recorded_us, observed_wall_us,"
            " policy_version, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id(),
                scope_id,
                kind,
                actor_id,
                self._store.next_event_us(),
                wall_us(),
                policy_version,
                _dump(payload, "payload"),
            ),
        )
        return int(cur.lastrowid)

    def latest_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(event_seq), 0) FROM events").fetchone()
        return int(row[0])

    def get(self, event_seq: int) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            row = _row(
                conn.execute(
                    "SELECT event_seq, event_id, scope_id, kind, actor_id,"
                    " recorded_us, observed_wall_us, policy_version, payload_json"
                    " FROM events WHERE event_seq = ?",
                    (event_seq,),
                )
            )
            if row is not None:
                row["payload"] = _json_parse(row.get("payload_json"))
            return row


# ----------------------------------------------------------------------
# entities
# ----------------------------------------------------------------------


class EntitiesRepo:
    """Scoped entities with evidence-backed aliases (SPEC §12).

    ``find_or_create`` matches on a case-normalized label inside one scope —
    two strings that look alike do not prove two entities are the same
    person, so matching never crosses scope partitions (SPEC §9). It is a
    write-path convenience (dedup a repeated mention), NOT a resolver:
    it returns the oldest same-label row. Resolution that must preserve
    ambiguity — competing same-name entities staying distinct — goes
    through ``all_for_label`` / ``verbatim.evidence.entities``
    (V4-17.02); callers minting deliberately distinct entities use
    ``create`` so the dedup helper cannot collapse them.
    """

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        label: str,
        kind: Optional[str] = None,
    ) -> str:
        """Always mint a new entity — the explicit-distinct path."""
        label = _require_str(label, "label").strip()
        entity_id = new_id()
        conn.execute(
            "INSERT INTO entities"
            "(entity_id, scope_id, kind, label, created_event, row_version)"
            " VALUES (?, ?, ?, ?, ?, 1)",
            (entity_id, scope_id, kind, label, _next_event_seq(conn)),
        )
        return entity_id

    def all_for_label(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        label: str,
    ) -> list:
        """Every entity in scope whose label case-matches — ambiguity
        preserved, oldest first (V4-17.02: never pick a winner)."""
        label = _require_str(label, "label").strip()
        rows = conn.execute(
            "SELECT entity_id FROM entities"
            " WHERE scope_id = ? AND LOWER(TRIM(label)) = LOWER(TRIM(?))"
            " ORDER BY created_event, entity_id",
            (scope_id, label),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def __init__(self, store: Store) -> None:
        self._store = store

    def find_or_create(
        self,
        scope_id: str,
        label: str,
        kind: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> str:
        """Match on case-normalized label inside one scope, else create."""
        if conn is None:
            with self._store.tx() as owned:
                return self._find_or_create(owned, scope_id, label, kind)
        return self._find_or_create(conn, scope_id, label, kind)

    def _find_or_create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        label: str,
        kind: Optional[str],
    ) -> str:
        label = _require_str(label, "label").strip()
        row = conn.execute(
            "SELECT entity_id FROM entities"
            " WHERE scope_id = ? AND LOWER(TRIM(label)) = LOWER(TRIM(?))"
            " ORDER BY created_event LIMIT 1",
            (scope_id, label),
        ).fetchone()
        if row is not None:
            return str(row[0])
        entity_id = new_id()
        conn.execute(
            "INSERT INTO entities"
            "(entity_id, scope_id, kind, label, created_event, row_version)"
            " VALUES (?, ?, ?, ?, ?, 1)",
            (entity_id, scope_id, kind, label, _next_event_seq(conn)),
        )
        return entity_id

    def add_alias(
        self,
        entity_id: str,
        normalized_alias: str,
        span_id: Optional[str],
        conn: sqlite3.Connection,
    ) -> None:
        """Record an evidence-backed naming association (idempotent)."""
        _require_str(normalized_alias, "normalized_alias")
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases"
            "(entity_id, normalized_alias, source_span_id, approval_event)"
            " VALUES (?, ?, ?, NULL)",
            (entity_id, normalized_alias, span_id),
        )

    def link_claim(
        self,
        conn: sqlite3.Connection,
        claim_id: str,
        entity_id: str,
        role: str = "mention",
        span_id: Optional[str] = None,
    ) -> None:
        """Join a claim to an entity for exact candidate retrieval."""
        _require_str(role, "role")
        conn.execute(
            "INSERT OR REPLACE INTO claim_entities(claim_id, entity_id, role, span_id)"
            " VALUES (?, ?, ?, ?)",
            (claim_id, entity_id, role, span_id),
        )

    def claims_for_entities(
        self,
        scope_id: str,
        entity_ids: Iterable[str],
        conn: sqlite3.Connection,
    ) -> list[str]:
        """Claim ids joined to any of ``entity_ids``, restricted to scope."""
        ids = list(dict.fromkeys(entity_ids or ()))
        if not ids:
            return []
        ph = _placeholders(ids)
        rows = conn.execute(
            "SELECT DISTINCT ce.claim_id FROM claim_entities ce"
            " JOIN claims c ON c.claim_id = ce.claim_id"
            f" WHERE c.scope_id = ? AND ce.entity_id IN ({ph})"
            " ORDER BY ce.claim_id",
            (scope_id, *ids),
        ).fetchall()
        return [str(r[0]) for r in rows]


# ----------------------------------------------------------------------
# edges
# ----------------------------------------------------------------------


class EdgesRepo:
    """Evidence graph lineage (SPEC §18-19).

    ``conflicts_with`` is an undirected relation stored once: endpoints are
    canonicalized to lexical order so adding B→A cannot duplicate A→B.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def add(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        source_kind: str,
        source_id: str,
        target_kind: str,
        target_id: str,
        edge_type: EdgeType | str,
        decision_id: Optional[str] = None,
    ) -> str:
        et = _enum_value(edge_type, EdgeType, "edge_type")
        _require_str(source_kind, "source_kind")
        _require_str(source_id, "source_id")
        _require_str(target_kind, "target_kind")
        _require_str(target_id, "target_id")
        s = (source_kind, source_id)
        t = (target_kind, target_id)
        if s == t:
            raise VerbatimError(ErrorCode.VALIDATION, "self-edge is not meaningful")
        if et == EdgeType.CONFLICTS_WITH.value and t < s:
            s, t = t, s
        row = conn.execute(
            "SELECT edge_id FROM edges"
            " WHERE scope_id = ? AND source_kind = ? AND source_id = ?"
            "   AND target_kind = ? AND target_id = ? AND edge_type = ?"
            "   AND retired_event IS NULL",
            (scope_id, s[0], s[1], t[0], t[1], et),
        ).fetchone()
        if row is not None:
            return str(row[0])
        edge_id = new_id()
        conn.execute(
            "INSERT INTO edges"
            "(edge_id, scope_id, source_kind, source_id, target_kind, target_id,"
            " edge_type, decision_id, created_event, retired_event)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (edge_id, scope_id, s[0], s[1], t[0], t[1], et, decision_id,
             _next_event_seq(conn)),
        )
        return edge_id

    def retire(self, edge_id: str, event_seq: int, conn: sqlite3.Connection) -> None:
        """Retire an edge at an event seq; retiring twice is STALE_PROPOSAL."""
        _require_int(event_seq, "event_seq")
        cur = conn.execute(
            "UPDATE edges SET retired_event = ?"
            " WHERE edge_id = ? AND retired_event IS NULL",
            (event_seq, edge_id),
        )
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM edges WHERE edge_id = ?", (edge_id,)
            ).fetchone()
            if exists is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "edge not found"
                )
            raise VerbatimError(ErrorCode.STALE_PROPOSAL, "edge already retired")

    def neighbors(
        self,
        scope_id: str,
        kind: str,
        id: str,
        edge_types: Iterable[EdgeType | str],
        conn: sqlite3.Connection,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Unretired edges touching ``(kind, id)`` in either direction."""
        ets = tuple(
            _enum_value(e, EdgeType, "edge_type") for e in (edge_types or ())
        )
        if not ets:
            return []
        _require_int(limit, "limit", minimum=1)
        ph = _placeholders(ets)
        return _rows(
            conn.execute(
                "SELECT edge_id, scope_id, source_kind, source_id, target_kind,"
                " target_id, edge_type, decision_id, created_event, retired_event"
                " FROM edges"
                " WHERE scope_id = ? AND retired_event IS NULL"
                f" AND edge_type IN ({ph})"
                " AND ((source_kind = ? AND source_id = ?)"
                "   OR (target_kind = ? AND target_id = ?))"
                " ORDER BY edge_id LIMIT ?",
                (scope_id, *ets, kind, id, kind, id, limit),
            )
        )


# ----------------------------------------------------------------------
# decisions and reviews
# ----------------------------------------------------------------------


class DecisionsRepo:
    """Advisory decision records with stale-check inputs (SPEC §20, §23).

    A decision row plus its content-hashed inputs commits in the caller's tx
    so a stale-result check can never observe half a record.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def record(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        task: TaskKind | str,
        request_hmac: bytes,
        backend: str,
        model_revision: Optional[str],
        rubric_version: str,
        policy_epoch: int,
        result: dict[str, Any],
        inputs: list[tuple[str, str, Optional[int], Optional[bytes]]],
    ) -> str:
        task_v = _enum_value(task, TaskKind, "task")
        _require_str(backend, "backend")
        _require_str(rubric_version, "rubric_version")
        _require_int(policy_epoch, "policy_epoch")
        if not isinstance(request_hmac, (bytes, bytearray)) or not request_hmac:
            raise VerbatimError(
                ErrorCode.VALIDATION, "request_hmac must be non-empty bytes"
            )
        if not isinstance(result, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "result must be a dict")
        decision_id = new_id()
        conn.execute(
            "INSERT INTO decisions"
            "(decision_id, scope_id, task, request_hmac, backend, model_revision,"
            " rubric_version, policy_epoch, result_json, created_us)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                scope_id,
                task_v,
                bytes(request_hmac),
                backend,
                model_revision,
                rubric_version,
                int(policy_epoch),
                _dump(result, "result"),
                wall_us(),
            ),
        )
        for item in inputs or ():
            try:
                object_kind, object_id, revision, content_hmac = item
            except (TypeError, ValueError) as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "inputs must be (object_kind, object_id, revision, content_hmac)",
                ) from exc
            _require_str(object_kind, "object_kind")
            _require_str(object_id, "object_id")
            conn.execute(
                "INSERT OR REPLACE INTO decision_inputs"
                "(decision_id, object_kind, object_id, revision, content_hmac)"
                " VALUES (?, ?, ?, ?, ?)",
                (decision_id, object_kind, object_id, revision, content_hmac),
            )
        return decision_id

    def get(self, decision_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            row = _row(
                conn.execute(
                    "SELECT decision_id, scope_id, task, hex(request_hmac) AS"
                    " request_hmac_hex, backend, model_revision, rubric_version,"
                    " policy_epoch, result_json, created_us"
                    " FROM decisions WHERE decision_id = ?",
                    (decision_id,),
                )
            )
            if row is not None:
                row["result"] = _json_parse(row.get("result_json"))
            return row


def _review_subjects(effect: dict[str, Any]) -> tuple[str, ...]:
    """Canonical claim subjects of a proposed review effect.

    Pairing producers name the counterparty under different keys
    (``counterparty_id`` in ``core.policy.relate``, ``conflict_with_claim_id``
    in ``evidence.relations``), so the dedup identity is the claim-id set —
    not the payload's field shape. ``supersede`` keeps direction:
    (predecessor, successor) is a different proposal than the converse.
    """
    if not isinstance(effect, dict):
        return ()
    if effect.get("effect") == "supersede":
        pair = (effect.get("predecessor_id"), effect.get("successor_id"))
        return tuple(x for x in pair if isinstance(x, str) and x)
    ids = (
        effect.get(k)
        for k in (
            "claim_id",
            "counterparty_id",
            "conflict_with_claim_id",
            "predecessor_id",
            "successor_id",
        )
    )
    return tuple(sorted({x for x in ids if isinstance(x, str) and x}))


class ReviewsRepo:
    """Operator review queue: proposals, expected versions, resolution."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def find_open(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        proposed_effect: dict[str, Any],
    ) -> Optional[str]:
        """review_id of an open review in ``scope_id`` proposing the same
        effect on the same claim subjects; ``None`` when none exists.

        The presence guard for redelivered proposals (mirrors
        ``evidence.relations._existing_conflict``): an equivalent *open*
        review already queued means this proposal is a retry, not new
        work. Resolved reviews never match — a re-proposal after a
        decision is legitimate new work.
        """
        kind = (
            proposed_effect.get("effect")
            if isinstance(proposed_effect, dict)
            else None
        )
        subjects = _review_subjects(proposed_effect)
        if not kind or not subjects:
            return None
        rows = conn.execute(
            "SELECT review_id, proposed_effect_json FROM reviews"
            " WHERE scope_id = ? AND state = 'open' ORDER BY review_id",
            (scope_id,),
        ).fetchall()
        for review_id, raw in rows:
            try:
                eff = _json_parse(raw)
            except VerbatimError:
                continue  # unparseable effect can never equal a valid one
            if (
                isinstance(eff, dict)
                and eff.get("effect") == kind
                and _review_subjects(eff) == subjects
            ):
                return str(review_id)
        return None

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        proposed_effect: dict[str, Any],
        expected_versions: dict[str, Any],
        decision_id: Optional[str] = None,
        *,
        dedup: bool = False,
    ) -> str:
        if not isinstance(proposed_effect, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "proposed_effect must be a dict")
        if not isinstance(expected_versions, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "expected_versions must be a dict")
        if dedup:
            existing = self.find_open(conn, scope_id, proposed_effect)
            if existing is not None:
                return existing
        review_id = new_id()
        conn.execute(
            "INSERT INTO reviews"
            "(review_id, scope_id, decision_id, proposed_effect_json, state,"
            " expected_versions_json, resolved_event)"
            " VALUES (?, ?, ?, ?, 'open', ?, NULL)",
            (
                review_id,
                scope_id,
                decision_id,
                _dump(proposed_effect, "proposed_effect"),
                _dump(expected_versions, "expected_versions"),
            ),
        )
        return review_id

    @staticmethod
    def _decode(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        row["proposed_effect"] = _json_parse(row.get("proposed_effect_json"))
        row["expected_versions"] = _json_parse(row.get("expected_versions_json"))
        return row

    def get(self, review_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return self._decode(
                _row(
                    conn.execute(
                        "SELECT review_id, scope_id, decision_id,"
                        " proposed_effect_json, state, expected_versions_json,"
                        " resolved_event FROM reviews WHERE review_id = ?",
                        (review_id,),
                    )
                )
            )

    def list_open(self, scope_id: str, limit: int = 100) -> list[dict[str, Any]]:
        _require_int(limit, "limit", minimum=1)
        with self._store.read() as conn:
            rows = _rows(
                conn.execute(
                    "SELECT review_id, scope_id, decision_id, proposed_effect_json,"
                    " state, expected_versions_json, resolved_event"
                    " FROM reviews WHERE scope_id = ? AND state = 'open'"
                    " ORDER BY review_id LIMIT ?",
                    (scope_id, limit),
                )
            )
        return [self._decode(r) for r in rows if r is not None]

    def resolve(
        self,
        conn: sqlite3.Connection,
        review_id: str,
        state: ReviewState | str,
        event_seq: int,
    ) -> None:
        """Resolve an open review; re-resolving is STALE_PROPOSAL (SPEC §15)."""
        state_v = _enum_value(state, ReviewState, "state")
        if state_v == ReviewState.OPEN.value:
            raise VerbatimError(
                ErrorCode.VALIDATION, "resolve requires a terminal state"
            )
        _require_int(event_seq, "event_seq")
        cur = conn.execute(
            "UPDATE reviews SET state = ?, resolved_event = ?"
            " WHERE review_id = ? AND state = 'open'",
            (state_v, event_seq, review_id),
        )
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
            if exists is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "review not found"
                )
            raise VerbatimError(ErrorCode.STALE_PROPOSAL, "review already resolved")


# ----------------------------------------------------------------------
# feedback, consents
# ----------------------------------------------------------------------


class FeedbackRepo:
    """Usefulness signals, kept separate from truth judgments (SPEC §20)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def add(
        self,
        conn: sqlite3.Connection,
        claim_id: str,
        actor_id: str,
        kind: FeedbackKind | str,
        event_id: Optional[str] = None,
    ) -> str:
        kind_v = _enum_value(kind, FeedbackKind, "kind")
        _require_str(actor_id, "actor_id")
        feedback_id = new_id()
        conn.execute(
            "INSERT INTO feedback"
            "(feedback_id, claim_id, actor_id, kind, created_us, event_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_id, claim_id, actor_id, kind_v, wall_us(), event_id),
        )
        return feedback_id

    def counts(self, claim_id: str) -> dict[str, int]:
        out: dict[str, int] = {k.value: 0 for k in FeedbackKind}
        total = 0
        with self._store.read() as conn:
            for kind, n in conn.execute(
                "SELECT kind, COUNT(*) FROM feedback WHERE claim_id = ? GROUP BY kind",
                (claim_id,),
            ).fetchall():
                out[kind] = int(n)
                total += int(n)
        out["total"] = total
        return out


class ConsentsRepo:
    """Explicit egress authorization grants (SPEC §27).

    An active consent is a grant with no revocation; lookups return the most
    recent active grant so revoke→re-grant cycles behave predictably.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def grant(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        processor: str,
        purpose: str,
        policy_digest: str,
    ) -> str:
        _require_str(processor, "processor")
        _require_str(purpose, "purpose")
        _require_str(policy_digest, "policy_digest")
        consent_id = new_id()
        conn.execute(
            "INSERT INTO consents"
            "(consent_id, scope_id, processor, purpose, granted_us, revoked_us,"
            " policy_digest)"
            " VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (consent_id, scope_id, processor, purpose, wall_us(), policy_digest),
        )
        return consent_id

    def revoke(self, conn: sqlite3.Connection, consent_id: str) -> None:
        """Revoke a grant; revoking a missing consent fails closed."""
        cur = conn.execute(
            "UPDATE consents SET revoked_us = ?"
            " WHERE consent_id = ? AND revoked_us IS NULL",
            (wall_us(), consent_id),
        )
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM consents WHERE consent_id = ?", (consent_id,)
            ).fetchone()
            if exists is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "consent not found"
                )

    def active(
        self, scope_id: str, processor: str, purpose: str
    ) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT consent_id, scope_id, processor, purpose, granted_us,"
                    " revoked_us, policy_digest FROM consents"
                    " WHERE scope_id = ? AND processor = ? AND purpose = ?"
                    "   AND revoked_us IS NULL"
                    " ORDER BY granted_us DESC, consent_id DESC LIMIT 1",
                    (scope_id, processor, purpose),
                )
            )


# ----------------------------------------------------------------------
# purges
# ----------------------------------------------------------------------


class PurgesRepo:
    """Two-phase erasure: exact preview, then suppression, then cleanup.

    Suppression commits before any physical delete so recall, history
    replay, review screens, and late model responses all observe the same
    tombstone set (SPEC §40). ``suppressed_ids`` is the single predicate
    every layer consults.
    """

    _SUPPRESSED = ("suppressed", "purging", "completed")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create_preview(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        targets: list[tuple[str, str]],
        digest: bytes,
    ) -> str:
        if not isinstance(digest, (bytes, bytearray)) or not digest:
            raise VerbatimError(
                ErrorCode.VALIDATION, "selection digest must be non-empty bytes"
            )
        if not targets:
            raise VerbatimError(ErrorCode.VALIDATION, "purge needs at least one target")
        purge_id = new_id()
        conn.execute(
            "INSERT INTO purges"
            "(purge_id, selection_digest, scope_id, state, requested_us,"
            " approved_us, completed_us)"
            " VALUES (?, ?, ?, 'previewed', ?, NULL, NULL)",
            (purge_id, bytes(digest), scope_id, wall_us()),
        )
        for item in targets:
            try:
                object_kind, object_id = item
            except (TypeError, ValueError) as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "targets must be (object_kind, object_id) pairs",
                ) from exc
            _require_str(object_kind, "object_kind")
            _require_str(object_id, "object_id")
            conn.execute(
                "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
                " VALUES (?, ?, ?)",
                (purge_id, object_kind, object_id),
            )
        return purge_id

    def _transition(
        self,
        conn: sqlite3.Connection,
        purge_id: str,
        to_state: str,
        allowed_from: tuple[str, ...],
        extra: str,
    ) -> None:
        ph = _placeholders(allowed_from)
        cur = conn.execute(
            f"UPDATE purges SET state = ?, {extra}"
            f" WHERE purge_id = ? AND state IN ({ph})",
            (to_state, wall_us(), purge_id, *allowed_from),
        )
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT state FROM purges WHERE purge_id = ?", (purge_id,)
            ).fetchone()
            if exists is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge not found"
                )
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL,
                f"purge in state {exists[0]!r} cannot move to {to_state!r}",
            )

    def confirm_suppress(self, conn: sqlite3.Connection, purge_id: str) -> None:
        """previewed → suppressed: tombstones activate in this same tx."""
        self._transition(conn, purge_id, "suppressed", ("previewed",), "approved_us = ?")

    def complete(self, conn: sqlite3.Connection, purge_id: str) -> None:
        """suppressed/purging → completed after physical cleanup finished."""
        self._transition(
            conn, purge_id, "completed", ("suppressed", "purging"), "completed_us = ?"
        )

    def get(self, purge_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            row = _row(
                conn.execute(
                    "SELECT purge_id, hex(selection_digest) AS selection_digest_hex,"
                    " scope_id, state, requested_us, approved_us, completed_us"
                    " FROM purges WHERE purge_id = ?",
                    (purge_id,),
                )
            )
            if row is not None:
                row["targets"] = _rows(
                    conn.execute(
                        "SELECT object_kind, object_id FROM purge_targets"
                        " WHERE purge_id = ? ORDER BY object_kind, object_id",
                        (purge_id,),
                    )
                )
            return row

    def suppressed_ids(self, object_kind: str, object_ids: Iterable[str]) -> set[str]:
        """Objects under active suppression — the erasure tombstone set."""
        ids = list(dict.fromkeys(object_ids or ()))
        if not ids:
            return set()
        ph = _placeholders(ids)
        states = _placeholders(self._SUPPRESSED)
        with self._store.read() as conn:
            rows = conn.execute(
                "SELECT DISTINCT pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                f" WHERE pt.object_kind = ? AND pt.object_id IN ({ph})"
                f" AND p.state IN ({states})",
                (object_kind, *ids, *self._SUPPRESSED),
            ).fetchall()
        return {str(r[0]) for r in rows}


# ----------------------------------------------------------------------
# lexical index
# ----------------------------------------------------------------------


class FtsRepo:
    """fts_rows + facts_fts mapping into the FTS5 index (SPEC §20).

    Facts stay searchable per scope and projection generation so a rebuild
    can populate generation N+1 beside live generation N and switch
    atomically — search never depends on a current-only index.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def _require_fts(self) -> None:
        if not self._store.fts_enabled:
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED, "FTS5 is unavailable in this build"
            )

    def index(
        self,
        conn: sqlite3.Connection,
        claim_id: str,
        claim_revision: int,
        scope_id: str,
        generation: int,
        text: str,
    ) -> int:
        """(Re)index one claim revision under a generation; returns row_id."""
        self._require_fts()
        _require_int(claim_revision, "claim_revision", minimum=1)
        _require_int(generation, "generation", minimum=1)
        if not isinstance(text, str):
            raise VerbatimError(ErrorCode.VALIDATION, "index text must be str")
        # REPLACE semantics without breaking the facts_fts FK: delete the
        # old row pair first, then insert fresh under the same unique key.
        conn.execute(
            "DELETE FROM facts_fts WHERE fts_row_id IN"
            " (SELECT row_id FROM fts_rows WHERE claim_id = ?"
            "   AND claim_revision = ? AND projection_generation = ?)",
            (claim_id, claim_revision, generation),
        )
        conn.execute(
            "DELETE FROM fts_rows WHERE claim_id = ? AND claim_revision = ?"
            " AND projection_generation = ?",
            (claim_id, claim_revision, generation),
        )
        cur = conn.execute(
            "INSERT INTO fts_rows"
            "(claim_id, claim_revision, scope_id, projection_generation)"
            " VALUES (?, ?, ?, ?)",
            (claim_id, claim_revision, scope_id, generation),
        )
        row_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO facts_fts(fts_row_id, text) VALUES (?, ?)", (row_id, text)
        )
        return row_id

    def index_claim_primary(
        self,
        conn: sqlite3.Connection,
        claim_id: str,
        claim_revision: int,
        scope_id: str,
        generation: int,
    ) -> bool:
        """Index a claim revision's primary evidence span — in the caller's tx.

        Reconstructs the exact quotation text from persisted source bytes
        (same join retrieval uses). Returns False when the claim has no
        indexable primary span or the bytes are not valid UTF-8 — callers
        treat that as "no projection", not an error. Must run inside the
        transition's own transaction (V2-39.01).
        """
        if not self._store.fts_enabled:
            return False
        row = conn.execute(
            "SELECT ce.span_id"
            " FROM claim_evidence ce"
            " JOIN spans sp ON sp.span_id = ce.span_id"
            " JOIN source_revisions sr"
            "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
            " WHERE ce.claim_id = ? AND ce.revision = ?"
            "   AND ce.evidence_role = 'primary'"
            "   AND sr.payload IS NOT NULL"
            " ORDER BY ce.span_id LIMIT 1",
            (claim_id, claim_revision),
        ).fetchone()
        if row is None:
            return False
        # Verified slice through the span authority: excerpt_hmac is
        # re-checked over the stored bytes — a tampered revision fails
        # STORE_CORRUPT rather than projecting forged text (V4-07.02).
        # Absent/purged excerpts simply have no projection.
        text = SpansRepo(self._store).text(row[0])
        if text is None:
            return False
        self.index(conn, claim_id, claim_revision, scope_id, generation, text)
        return True

    def search(
        self,
        scope_ids: list[str],
        match_query: str,
        generation: int,
        limit: int = 32,
    ) -> list[tuple[str, int, float]]:
        """bm25-ranked (claim_id, claim_revision, rank) inside scope+generation.

        The scope partition filter runs inside the query — an unauthorized
        audience's corpus can never affect returned term statistics
        (SPEC §20).
        """
        if not self._store.fts_enabled:
            return []
        ids = list(dict.fromkeys(scope_ids or ()))
        if not ids or not isinstance(match_query, str) or not match_query.strip():
            return []
        _require_int(limit, "limit", minimum=1)
        ph = _placeholders(ids)
        with self._store.read() as conn:
            try:
                rows = conn.execute(
                    "SELECT fr.claim_id, fr.claim_revision,"
                    "       bm25(facts_fts_idx) AS rank"
                    " FROM facts_fts_idx"
                    " JOIN fts_rows fr ON fr.row_id = facts_fts_idx.rowid"
                    f" WHERE facts_fts_idx MATCH ?"
                    "   AND fr.projection_generation = ?"
                    f"   AND fr.scope_id IN ({ph})"
                    " ORDER BY rank LIMIT ?",
                    (match_query, generation, *ids, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"invalid FTS query: {exc}"
                ) from exc
        return [(str(r[0]), int(r[1]), float(r[2])) for r in rows]

    def delete_for_generation(self, conn: sqlite3.Connection, generation: int) -> int:
        """Drop every index row of a generation (rebuild cleanup)."""
        self._require_fts()
        _require_int(generation, "generation", minimum=1)
        conn.execute(
            "DELETE FROM facts_fts WHERE fts_row_id IN"
            " (SELECT row_id FROM fts_rows WHERE projection_generation = ?)",
            (generation,),
        )
        cur = conn.execute(
            "DELETE FROM fts_rows WHERE projection_generation = ?", (generation,)
        )
        return int(cur.rowcount)

    def verify(self, generation: Optional[int] = None) -> dict[str, Any]:
        """Rebuild validation: mapping table, content table, and index counts.

        After a rebuild the three sides must agree; a mismatch means the
        triggers or the swap were interrupted and the generation is unsafe.
        """
        if not self._store.fts_enabled:
            return {"ok": False, "reason": "fts5 unavailable"}
        with self._store.read() as conn:
            if generation is None:
                fr = conn.execute("SELECT COUNT(*) FROM fts_rows").fetchone()[0]
                ff = conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
                ix = conn.execute("SELECT COUNT(*) FROM facts_fts_idx").fetchone()[0]
            else:
                fr = conn.execute(
                    "SELECT COUNT(*) FROM fts_rows WHERE projection_generation = ?",
                    (generation,),
                ).fetchone()[0]
                ff = conn.execute(
                    "SELECT COUNT(*) FROM facts_fts f"
                    " JOIN fts_rows r ON r.row_id = f.fts_row_id"
                    " WHERE r.projection_generation = ?",
                    (generation,),
                ).fetchone()[0]
                ix = conn.execute(
                    "SELECT COUNT(*) FROM facts_fts_idx i"
                    " JOIN fts_rows r ON r.row_id = i.rowid"
                    " WHERE r.projection_generation = ?",
                    (generation,),
                ).fetchone()[0]
        return {
            "ok": fr == ff == ix,
            "fts_rows": int(fr),
            "facts_fts": int(ff),
            "index_rows": int(ix),
        }
