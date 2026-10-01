"""V5 consumer controls — ``inspect`` + ``forget`` (SPEC_V5 §06.11, §14, §15).

This module is the single-authority implementation behind the facade's
``Memory.inspect`` / ``Memory.forget`` methods (facade worker composes
it).  It owns no policy of its own: authorization flows through the
existing ``governance.authorize`` grant evaluator, suppression through
the existing purge registry, physical cleanup through the existing
resumable ``ClosureEngine``, and span/source bytes through the existing
integrity-checked repositories.  Nothing here mints authority, bypasses
quote permission, or fabricates ready/complete states.

Bindings (V5-06.11, §05.12):
  * ``namespace`` is the facade-bound namespace — an opaque scope row id
    minted by ``aliases.provision`` (``ns_…``); it IS the partition a
    ``MemoryRef`` is pinned to.
  * ``store_tag`` is the opaque store identity carried in refs
    (defaults to ``Store.db_id()`` — a random id, never a path).

Forget-by-ref is a targeted suppression-and-closure request (V5-15.03):
a version-bound ``MemoryRef`` must match the approved mutation head and
``source_state`` control version *inside the suppression commit*, then
the closure engine tombstones the root and drains the derived-data
boundary.  Query forget is two-phase (V5-15.04): a mutation-free preview
returns the authorized selection plus an HMAC-bound, caller/namespace/
store-bound, single-use, short-TTL confirmation token; only the exact
pinned selection can execute, never "whatever matches now".
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import sqlite3
from typing import Any, Iterable, Mapping, Optional

from ..core.time import now_us, rfc3339
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import Verb
from ..core.types_v4 import ClosurePhase
from .. import derivations
from ..governance import CallerV3, authorize, effective_verbs
from ..privacy.closure import ClosureEngine
from ..storage.repos import has_table as _has_table
from ..storage import repos_v4

from .errors import conflict, denied, invalid
from .types import ForgetResult, Inspection, MemoryRef

# Source-state control plane (the ``source_state/v1`` artifact).  The
# module's ``get``/``history``/``current_state`` readers and the
# ``apply_erasure`` tombstone writer are preferred; when the module is
# absent (pre-v5 builds) reads fall back to the contract-frozen table.
try:
    from .. import sourcestate as _sourcestate
except Exception:  # pragma: no cover - absent in pre-v5 builds
    _sourcestate = None

try:  # §30.6 proposition grounding view (enrichment worker surface)
    from ..enrichment import grounding as _grounding
except Exception:  # pragma: no cover - absent in pre-v5 builds
    _grounding = None


# ---------------------------------------------------------------------------
# constants / small types
# ---------------------------------------------------------------------------

#: ``inspect.detail`` levels admitted by the v5 surface (V5-06.17).
#: ``enrichment`` exposes the T1/projection plane: stored enrichment
#: rows, entity postings, and the proposition-grounding view (§30.6).
_DETAIL_LEVELS = ("evidence", "metadata", "enrichment")

#: Confirmation-token wire prefix.
_TOKEN_PREFIX = "vfc1"
#: HMAC domain separator so a forget token can never collide with any
#: other store-keyed MAC (expand refs, pack tokens, …).
_TOKEN_DOMAIN = b"verbatim:v5:forget-confirm\x00"
#: Confirmation token lifetime — short-lived by contract (V5-15.04).
TOKEN_TTL_US = 300_000_000  # 5 minutes
#: Bound on sources scanned for a query-forget selection.
_SCAN_LIMIT = 4096
#: Bound on members a single confirmation token may select.
_SELECT_LIMIT = 256
#: Query-length bound mirroring the request caps used by recall.
_QUERY_LIMIT = 8192
#: Suppression states that fence delivery — mirrors PurgesRepo._SUPPRESSED.
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

#: V5 derived tables keyed by ``source_id`` — regenerable projections that
#: deletion closure must cover (V5-11.11, §15, contracts §3).  All writes
#: here are ``DELETE … WHERE source_id = ?`` guarded by table+column
#: presence so a pre-v5 store simply has nothing to sweep.  The
#: ``source_fts`` external-content table is handled first (its rowid
#: references ``source_fts_rows`` and its delete trigger maintains the
#: ``source_fts_idx`` shadow); ``update_candidates`` and the
#: ``duplicate_links.group_id`` anchor are swept separately below.
_V5_SOURCE_TABLES = (
    "source_lexical_projection",
    "source_fts_rows",
    "source_vectors",
    "entity_postings",
    "enrichment",
    "duplicate_links",
)

_PRODUCER = "memory.controls:1"


def _hex_enc(text: str) -> str:
    return text.encode("utf-8").hex()


def _hex_dec(text: str) -> str:
    try:
        return bytes.fromhex(text).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise invalid("object ref carries an undecodable id") from exc


def object_ref_to_string(kind: str, object_id: str, revision: Optional[int]) -> str:
    """Serialize an exact claim/view object ref (Hit.object_ref wire form).

    ``vobj1.<kind>.<hex(utf8 id)>.<rev|->`` — the id is hex-encoded so
    arbitrary identifier bytes survive the dotted envelope unchanged.
    """
    require_id(kind, "object_kind")
    if not isinstance(object_id, str) or not object_id:
        raise invalid("object ref requires a non-empty object_id")
    rev = "-" if revision is None else str(int(revision))
    return f"vobj1.{kind}.{_hex_enc(object_id)}.{rev}"


def _coerce_ref(value: Any) -> tuple:
    """Parse a caller-supplied ref.

    Returns ``("memory", MemoryRef)`` for source-backed refs or
    ``("object", (kind, object_id, revision|None))`` for claim/view refs.
    A bare identifier string is the consumer ``memory_id`` — an alias of
    ``source_id`` (V5-06.11) — and resolves as ``("bare", source_id)``.
    """
    if isinstance(value, MemoryRef):
        return ("memory", value)
    if isinstance(value, str):
        if value.startswith(MemoryRef.PREFIX + "."):
            try:
                return ("memory", MemoryRef.parse(value))
            except (ValueError, TypeError) as exc:
                raise invalid(f"malformed MemoryRef: {exc}") from exc
        if value.startswith("vobj1."):
            parts = value.split(".")
            if len(parts) not in (3, 4):
                raise invalid("object ref must be vobj1.<kind>.<id>.<rev>")
            kind = parts[1]
            oid = _hex_dec(parts[2])
            rev: Optional[int] = None
            if len(parts) == 4 and parts[3] != "-":
                try:
                    rev = int(parts[3])
                except ValueError:
                    raise invalid("object ref revision must be an integer")
            return ("object", (kind, oid, rev))
        if not value:
            raise invalid("ref must not be empty")
        # Bare logical memory id (alias of source_id — V5-06.11).
        return ("bare", require_id(value, "memory_id"))
    if isinstance(value, Mapping):
        if isinstance(value.get("ref"), str):
            return _coerce_ref(value["ref"])
        if {"store_tag", "namespace", "source_id"} <= set(value):
            try:
                return (
                    "memory",
                    MemoryRef(
                        store_tag=str(value["store_tag"]),
                        namespace=str(value["namespace"]),
                        source_id=str(value["source_id"]),
                        expected_revision=int(value["expected_revision"]),
                        control_version=int(value["control_version"]),
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise invalid(f"malformed MemoryRef mapping: {exc}") from exc
        kind = value.get("object_kind", value.get("kind"))
        oid = value.get("object_id", value.get("id"))
        rev = value.get("object_revision", value.get("revision"))
        if isinstance(kind, str) and isinstance(oid, str):
            return ("object", (kind, oid, _norm_rev(rev)))
        raise invalid("unrecognized ref mapping")
    if isinstance(value, (tuple, list)) and len(value) in (2, 3):
        kind, oid = value[0], value[1]
        rev = value[2] if len(value) == 3 else None
        if not isinstance(kind, str) or not isinstance(oid, str):
            raise invalid("object ref entries must be (kind, id[, revision])")
        return ("object", (kind, oid, _norm_rev(rev)))
    raise invalid("ref must be a MemoryRef, object ref, or memory_id")


def _norm_rev(rev: Any) -> Optional[int]:
    if rev is None:
        return None
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1:
        raise invalid("object revision must be an int >= 1")
    return rev


# ---------------------------------------------------------------------------
# confirmation tokens (V5-15.04)
# ---------------------------------------------------------------------------


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text.encode("ascii"))
    except Exception as exc:
        raise invalid("malformed confirmation token encoding") from exc


class MemoryControls:
    """Bound-caller inspect/forget surface for one facade namespace.

    The facade constructs one of these per ``Memory`` binding:

    >>> controls = MemoryControls(store, caller=CallerV3("alice"),
    ...                           namespace="ns_abc123")
    >>> controls.inspect(ref); controls.forget(ref, idempotency_key="k")
    """

    def __init__(
        self,
        store: Any,
        *,
        caller: CallerV3,
        namespace: str,
        scope_id: Optional[str] = None,
        store_tag: Optional[str] = None,
        purpose: Optional[str] = None,
        confirm_ttl_us: int = TOKEN_TTL_US,
        scan_limit: int = _SCAN_LIMIT,
        select_limit: int = _SELECT_LIMIT,
        drain_budget: Optional[int] = None,
        drain_max_steps: Optional[int] = None,
    ) -> None:
        self._store = store
        self._caller = caller
        # The namespace IS the partition (aliases mint scope ids ns_…);
        # an explicit scope_id override exists for stores whose namespace
        # predates the alias layer, never to widen the binding.
        self._namespace = require_id(namespace, "namespace")
        self._scope_id = scope_id or self._namespace
        self._store_tag = store_tag if store_tag is not None else store.db_id()
        self._purpose = purpose
        self._confirm_ttl_us = int(confirm_ttl_us)
        self._scan_limit = int(scan_limit)
        self._select_limit = int(select_limit)
        self._drain_budget = drain_budget
        self._drain_max_steps = drain_max_steps
        self._engine = ClosureEngine(store)

    # ------------------------------------------------------------------
    # shared primitives
    # ------------------------------------------------------------------

    def _deny(self) -> None:
        raise denied()

    def _authorize(self, conn: sqlite3.Connection, verb: Verb) -> None:
        """Bound-caller authorization — denial is indistinguishable."""
        authorize(conn, self._caller, self._scope_id, verb.value, self._purpose)

    @staticmethod
    def _source_row(conn: sqlite3.Connection, source_id: str) -> Optional[dict]:
        row = conn.execute(
            "SELECT source_id, origin, external_id, source_kind, scope_id,"
            " speaker_id, created_us FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "source_id": row[0],
            "origin": row[1],
            "external_id": row[2],
            "source_kind": row[3],
            "scope_id": row[4],
            "speaker_id": row[5],
            "created_us": row[6],
        }

    @staticmethod
    def _suppressed_ids(
        conn: sqlite3.Connection, kind: str, ids: Iterable[str]
    ) -> set[str]:
        """Conn-local purge-registry predicate (same states as PurgesRepo)."""
        wanted = sorted(set(ids))
        if not wanted or not _has_table(conn, "purge_targets"):
            return set()
        ph = ",".join("?" * len(wanted))
        states = ",".join("?" * len(_SUPPRESSING_STATES))
        rows = conn.execute(
            "SELECT DISTINCT pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            f" WHERE pt.object_kind = ? AND pt.object_id IN ({ph})"
            f" AND p.state IN ({states})",
            [kind, *wanted, *_SUPPRESSING_STATES],
        ).fetchall()
        return {str(r[0]) for r in rows}

    @staticmethod
    def _held(conn: sqlite3.Connection, kind: str, oid: str, rev: int) -> bool:
        """Quarantine invisibility check (V3-14.10) — fail closed."""
        if not _has_table(conn, "quarantine"):
            return False
        try:
            from ..security.quarantine import should_exclude

            return bool(should_exclude(conn, kind, oid, int(rev)))
        except VerbatimError:
            raise
        except Exception:
            return True  # unreadable quarantine state — fail closed

    def _check_binding(self, mref: MemoryRef) -> None:
        """Store/namespace binding — foreign refs deny indistinguishably."""
        if (
            not isinstance(mref, MemoryRef)
            or mref.store_tag != self._store_tag
            or mref.namespace != self._namespace
        ):
            self._deny()

    # ------------------------------------------------------------------
    # source-state control head (via verbatim.sourcestate if importable)
    # ------------------------------------------------------------------

    def _source_state(self, conn: sqlite3.Connection, source_id: str) -> Optional[dict]:
        """The ``source_state/v1`` control row, or None when unprovisioned."""
        if _sourcestate is not None:  # pragma: no cover - module lands later
            for name in ("current", "get", "read", "state"):
                fn = getattr(_sourcestate, name, None)
                if callable(fn):
                    try:
                        row = fn(conn, source_id)
                    except TypeError:
                        continue
                    if isinstance(row, dict):
                        return row
        if not _has_table(conn, "source_state"):
            return None
        row = conn.execute(
            "SELECT source_id, namespace, control_version, mutation_head,"
            " disposition, superseded_by, effective_at, known_at,"
            " valid_from, valid_to, updated_at, producer"
            " FROM source_state WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "source_id": row[0],
            "namespace": row[1],
            "control_version": int(row[2]),
            "mutation_head": row[3],
            "disposition": row[4],
            "superseded_by": row[5],
            "effective_at": row[6],
            "known_at": row[7],
            "valid_from": row[8],
            "valid_to": row[9],
            "updated_at": row[10],
            "producer": row[11],
        }

    def _control_head(
        self, conn: sqlite3.Connection, source_id: str
    ) -> tuple[Optional[int], int, str]:
        """(approved head revision, control version, disposition).

        With a ``source_state`` row the head is the recorded
        ``mutation_head`` (an integer revision serialized as TEXT).  On a
        pre-v5 store there is no separate approval ledger — the observable
        head is ``MAX(source_revisions.revision)`` and the control version
        is the minted default ``0``.
        """
        state = self._source_state(conn, source_id)
        if state is not None:
            head_raw = state.get("mutation_head")
            head: Optional[int] = None
            # ``sourcestate.parse_head`` semantics: the "unresolved"
            # sentinel and any non-integer token mean *no approved head* —
            # never an integrity failure at read time.
            if isinstance(head_raw, bool):
                head = None
            elif isinstance(head_raw, int):
                head = head_raw if head_raw >= 1 else None
            elif isinstance(head_raw, str) and head_raw:
                try:
                    parsed = int(head_raw)
                except ValueError:
                    head = None
                else:
                    head = parsed if parsed >= 1 else None
            return (
                head,
                int(state.get("control_version") or 0),
                str(state.get("disposition") or "recorded"),
            )
        row = conn.execute(
            "SELECT MAX(revision) FROM source_revisions WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        head = int(row[0]) if row and row[0] is not None else None
        return (head, 0, "recorded")

    # ------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------

    def inspect(self, ref: Any, *, detail: str = "evidence") -> Inspection:
        """Authorized evidence/metadata inspection (V5-15.01).

        Accepts a serialized or object ``MemoryRef``, an object ref
        (``vobj1.…`` string, ``(kind, id[, rev])`` tuple, or mapping), or
        a bare ``memory_id`` (source alias).  Missing, foreign, and
        unauthorized targets all deny identically.
        """
        if detail not in _DETAIL_LEVELS:
            raise invalid(
                f"inspect detail must be one of {_DETAIL_LEVELS}, "
                f"got {detail!r}"
            )
        kind, parsed = _coerce_ref(ref)
        if kind == "memory":
            return self._inspect_source(parsed, detail, ref_string=parsed.to_string())
        if kind == "bare":
            return self._inspect_source_id(parsed, detail, ref_string=None)
        return self._inspect_object(parsed, detail)

    def _inspect_source(
        self, mref: MemoryRef, detail: str, *, ref_string: str
    ) -> Inspection:
        self._check_binding(mref)
        return self._inspect_source_id(mref.source_id, detail, ref_string=ref_string)

    def _inspect_source_id(
        self, source_id: str, detail: str, *, ref_string: Optional[str]
    ) -> Inspection:
        store = self._store
        with store.read() as conn:
            row = self._source_row(conn, source_id)
            if row is None or row["scope_id"] != self._scope_id:
                self._deny()
            self._authorize(conn, Verb.READ)
            quote_ok = Verb.QUOTE.value in effective_verbs(
                conn, self._caller, self._scope_id
            )

            suppressed = self._suppressed_ids(conn, "source", [source_id])
            rev_suppressed = self._suppressed_ids(
                conn, "source_revision", [f"{source_id}:{r[0]}" for r in conn.execute(
                    "SELECT revision FROM source_revisions WHERE source_id = ?",
                    (source_id,),
                ).fetchall()]
            )
            state = self._source_state(conn, source_id)
            head_rev, ctl_version, disposition = self._control_head(conn, source_id)

            revisions = []
            for r in conn.execute(
                "SELECT revision, length(payload), payload_hmac IS NOT NULL,"
                " event_us, captured_us, timezone, provenance, metadata_json"
                " FROM source_revisions WHERE source_id = ? ORDER BY revision",
                (source_id,),
            ).fetchall():
                rev_sup = f"{source_id}:{r[0]}" in rev_suppressed
                revisions.append(
                    {
                        "revision": r[0],
                        "payload_bytes": int(r[1] or 0),
                        "integrity": (
                            "purged"
                            if int(r[1] or 0) == 0 and r[2]
                            else ("hmac-verified" if r[2] else "legacy")
                        ),
                        "event_us": r[3],
                        "captured_us": r[4],
                        "timezone": r[5],
                        "provenance": r[6],
                        "metadata": safe_json_loads(r[7]) if r[7] else {},
                        "suppressed": rev_sup,
                        "held": self._held(conn, "source", source_id, r[0]),
                    }
                )

            envelopes = []
            if _has_table(conn, "source_envelopes"):
                for e in conn.execute(
                    "SELECT envelope_id, revision, envelope_kind,"
                    " actor_principal, trust_class, adapter_version,"
                    " media_type, redaction_status, event_us, receipt_us,"
                    " host_id, session_id, task_id"
                    " FROM source_envelopes WHERE source_id = ?"
                    " ORDER BY revision, envelope_id",
                    (source_id,),
                ).fetchall():
                    envelopes.append(
                        {
                            "envelope_id": e[0],
                            "revision": e[1],
                            "kind": e[2],
                            "actor_principal": e[3],
                            "trust_class": e[4],
                            "adapter_version": e[5],
                            "media_type": e[6],
                            "redaction_status": e[7],
                            "event_us": e[8],
                            "receipt_us": e[9],
                            "host_id": e[10],
                            "session_id": e[11],
                            "task_id": e[12],
                        }
                    )

            evidence = self._source_evidence(
                conn, store, source_id, detail, quote_ok
            )
            lifecycle = {
                "disposition": disposition,
                "mutation_head": head_rev,
                "control_version": ctl_version,
                "suppressed": source_id in suppressed,
                "revision_suppressed": sorted(rev_suppressed),
                "quarantined": any(r["held"] for r in revisions),
                # Top-level forward pointer (V5-14.10): consumers walk the
                # supersession chain without digging into source_state.
                "superseded_by": (
                    state.get("superseded_by") if state is not None else None
                ),
                "effective_at": (
                    state.get("effective_at") if state is not None else None
                ),
                "valid_from": (
                    state.get("valid_from") if state is not None else None
                ),
                "valid_to": (
                    state.get("valid_to") if state is not None else None
                ),
                "source_state": (
                    {k: v for k, v in state.items() if k != "source_id"}
                    if state is not None
                    else None
                ),
            }
            lifecycle.update(self._control_ledger(conn, source_id))
            enrichment = self._source_enrichment(conn, source_id)
            derived_claims = self._derived_claims(conn, source_id)
            if derived_claims:
                enrichment["derived_claims"] = derived_claims

        if ref_string is None:
            # Bare-id inspect: return a CAS-ready MemoryRef pinned at the
            # current approved mutation head + control version (V5-06.11).
            head_int = head_rev if isinstance(head_rev, int) else None
            if head_int is None and revisions:
                head_int = int(revisions[-1]["revision"])
            ref_string = MemoryRef(
                store_tag=str(self._store_tag),
                namespace=self._namespace,
                source_id=source_id,
                expected_revision=int(head_int or 0),
                control_version=int(ctl_version or 0),
            ).to_string()

        out = Inspection(
            ref=ref_string,
            found=True,
            detail=detail,
            provenance={
                "source_id": source_id,
                "origin": row["origin"],
                "external_id": row["external_id"],
                "source_kind": row["source_kind"],
                "speaker_id": row["speaker_id"],
                "created_us": row["created_us"],
                "attribution": sorted(
                    {e["actor_principal"] for e in envelopes if e["actor_principal"]}
                ),
                "trust_class": sorted(
                    {e["trust_class"] for e in envelopes if e["trust_class"]}
                ),
                "producer": sorted(
                    {e["adapter_version"] for e in envelopes if e["adapter_version"]}
                    | {
                        s["harvester_version"]
                        for s in evidence
                        if s.get("harvester_version")
                    }
                ),
                "envelopes": envelopes,
            },
            revisions=revisions,
            lifecycle=lifecycle,
            evidence=evidence,
            enrichment=enrichment,
            warnings=[],
        )
        if detail == "evidence" and not quote_ok:
            out.warnings.append(
                "quote_not_granted: evidence locators returned without "
                "excerpt text (V5-15.01)"
            )
        if lifecycle["suppressed"]:
            out.warnings.append("suppressed: memory is under an active erasure tombstone")
        if lifecycle["quarantined"]:
            out.warnings.append("held: one or more revisions under quarantine hold")
        return out

    def _source_evidence(
        self,
        conn: sqlite3.Connection,
        store: Any,
        source_id: str,
        detail: str,
        quote_ok: bool,
    ) -> list[dict]:
        """Evidence locators; excerpt text only for evidence+quote."""
        spans = conn.execute(
            "SELECT span_id, revision, start_byte, end_byte, view_id,"
            " harvester_version FROM spans"
            " WHERE source_id = ? ORDER BY revision, start_byte, span_id",
            (source_id,),
        ).fetchall()
        if not spans:
            return []
        held_spans = self._suppressed_ids(conn, "span", [s[0] for s in spans])
        env_of: dict = {}
        if _has_table(conn, "source_envelopes"):
            for srev in {s[1] for s in spans}:
                env_of[srev] = [
                    r[0]
                    for r in conn.execute(
                        "SELECT envelope_id FROM source_envelopes"
                        " WHERE source_id = ? AND revision = ?",
                        (source_id, srev),
                    ).fetchall()
                ]
        items = []
        for span_id, srev, sb, eb, view_id, hver in spans:
            held = (
                span_id in held_spans
                or self._held(conn, "span", span_id, srev)
                or self._held(conn, "source", source_id, srev)
                or any(
                    self._held(conn, "source_envelope", eid, srev)
                    for eid in env_of.get(srev, ())
                )
            )
            item = {
                "span_id": span_id,
                "revision": srev,
                "start_byte": sb,
                "end_byte": eb,
                "view_id": view_id,
                "harvester_version": hver,
                "held": held,
            }
            if detail == "evidence":
                text: Optional[str] = None
                if held:
                    text = None
                elif quote_ok:
                    # SpansRepo.text re-verifies the excerpt digest —
                    # corruption raises rather than renders (F4-06).
                    from ..storage.repos import SpansRepo

                    text = SpansRepo(store).text(span_id)
                item["text"] = text if text is not None else "[unavailable]"
            items.append(item)
        return items

    def _source_enrichment(
        self, conn: sqlite3.Connection, source_id: str
    ) -> dict:
        """Enrichment/duplicate/update surfaces — empty when unprovisioned."""
        out: dict[str, Any] = {}
        if _has_table(conn, "enrichment"):
            out["enrichment"] = [
                {
                    "revision": r[0],
                    "producer": r[1],
                    "type": r[2],
                    "polarity": r[3],
                    "time_precision": r[4],
                    "time_status": r[5],
                    "event_at": r[6],
                    "anchor_at": r[7],
                    "fields": safe_json_loads(r[8]) if r[8] else {},
                }
                for r in conn.execute(
                    "SELECT revision, producer, type, polarity,"
                    " time_precision, time_status, event_at, anchor_at,"
                    " fields_json FROM enrichment WHERE source_id = ?"
                    " ORDER BY revision, producer",
                    (source_id,),
                ).fetchall()
            ]
        if _has_table(conn, "duplicate_links"):
            links = conn.execute(
                "SELECT revision, group_id, method, score, created_at"
                " FROM duplicate_links WHERE source_id = ?"
                " ORDER BY revision, method",
                (source_id,),
            ).fetchall()
            out["duplicate_links"] = [
                {
                    "revision": r[0],
                    "group_id": r[1],
                    "method": r[2],
                    "score": r[3],
                    "created_at": r[4],
                }
                for r in links
            ]
            groups = {r[1] for r in links}
            members: dict = {}
            for gid in sorted(groups):
                members[gid] = [
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT source_id FROM duplicate_links"
                        " WHERE group_id = ? ORDER BY source_id",
                        (gid,),
                    ).fetchall()
                ]
            if members:
                out["duplicate_groups"] = members
        if _has_table(conn, "update_candidates"):
            out["update_candidates"] = [
                {
                    "candidate_id": r[0],
                    "role": (
                        "prior"
                        if r[2] == source_id
                        else "new"
                    ),
                    "counterpart": (
                        r[1] if r[2] == source_id else r[2]
                    ),
                    "relation": r[3],
                    "state": r[4],
                    "score": r[5],
                }
                for r in conn.execute(
                    "SELECT candidate_id, new_source_id, prior_source_id,"
                    " relation, state, score FROM update_candidates"
                    " WHERE new_source_id = ? OR prior_source_id = ?"
                    " ORDER BY candidate_id",
                    (source_id, source_id),
                ).fetchall()
            ]
        # Entity postings (§30.4): the grouped identifier/entity index —
        # mention offsets are byte pins into the pinned revision.
        postings: list = []
        if _has_table(conn, "entity_postings"):
            postings = [
                {
                    "entity": r[0],
                    "entity_kind": r[1],
                    "revision": r[2],
                    "offsets": safe_json_loads(r[3]) if r[3] else [],
                }
                for r in conn.execute(
                    "SELECT entity, entity_kind, revision, offsets"
                    " FROM entity_postings WHERE source_id = ?"
                    " ORDER BY entity, revision",
                    (source_id,),
                ).fetchall()
            ]
            out["entity_postings"] = postings
        # Proposition grounding (§30.6, V5-30.01/30.18): every extraction
        # and derived claim is a proposition whose pins must re-verify
        # against the retained bytes — unverifiable records stay visible
        # as unsupported_extraction, never silently facts.
        if _grounding is not None:
            props = _grounding.build_propositions(
                source_id=source_id,
                enrichment_rows=out.get("enrichment") or (),
                entity_postings=postings,
                claim_evidence=self._claim_evidence_pins(conn, source_id),
                payload=self._revision_payloads(conn, source_id).get,
                hmac=self._store.hmac,
            )
            out["propositions"] = props
            out["proposition_stats"] = _grounding.proposition_stats(props)
        return out

    def _revision_payloads(
        self, conn: sqlite3.Connection, source_id: str
    ) -> dict:
        """``{revision: bytes}`` with the same integrity check
        ``SourcesRepo.payload`` applies — the stored HMAC is re-verified
        on every read; a mismatch is ``STORE_CORRUPT``, never content."""
        out: dict = {}
        for r in conn.execute(
            "SELECT revision, payload, payload_hmac FROM source_revisions"
            " WHERE source_id = ?",
            (source_id,),
        ).fetchall():
            blob = bytes(r[1] or b"")
            if r[2] is not None and not _hmac.compare_digest(
                self._store.hmac(blob), bytes(r[2])
            ):
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT,
                    f"source {source_id}@{r[0]} payload fails integrity check",
                )
            out[int(r[0])] = blob
        return out

    def _claim_evidence_pins(
        self, conn: sqlite3.Connection, source_id: str
    ) -> list:
        """Claims derived from this source with their span pins — the
        grounding view's proposition records for derived memory."""
        if not _has_table(conn, "claim_evidence"):
            return []
        rows = conn.execute(
            "SELECT ce.claim_id, ce.revision, ce.span_id, ce.evidence_role,"
            " s.revision, s.start_byte, s.end_byte, s.excerpt_hmac,"
            " s.harvester_version"
            " FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id"
            " WHERE s.source_id = ?"
            " ORDER BY ce.claim_id, ce.revision, s.start_byte",
            (source_id,),
        ).fetchall()
        if not rows:
            return []
        suppressed = self._suppressed_ids(
            conn, "claim", [r[0] for r in rows]
        )
        return [
            {
                "claim_id": r[0],
                "claim_revision": r[1],
                "span_id": r[2],
                "evidence_role": r[3],
                "source_revision": r[4],
                "start_byte": r[5],
                "end_byte": r[6],
                # Hex over the wire — to_dict() must stay JSON-safe.
                "excerpt_hmac": r[7].hex() if isinstance(r[7], (bytes, bytearray)) else r[7],
                "harvester_version": r[8],
                "suppressed": r[0] in suppressed,
            }
            for r in rows
        ]

    def _derived_claims(
        self, conn: sqlite3.Connection, source_id: str
    ) -> list[dict]:
        """Interpretation ancestry — claims derived from this source
        (V5-15.01)."""
        if not _has_table(conn, "claim_evidence"):
            return []
        claims = conn.execute(
            "SELECT DISTINCT ce.claim_id, c.scope_id FROM claim_evidence ce"
            " JOIN spans s ON s.span_id = ce.span_id"
            " JOIN claims c ON c.claim_id = ce.claim_id"
            " WHERE s.source_id = ? ORDER BY ce.claim_id",
            (source_id,),
        ).fetchall()
        if not claims:
            return []
        suppressed = self._suppressed_ids(
            conn, "claim", [c[0] for c in claims]
        )
        return [
            {
                "claim_id": c[0],
                "scope_id": c[1],
                "suppressed": c[0] in suppressed,
                "object_ref": object_ref_to_string("claim", c[0], None),
            }
            for c in claims
        ]

    def _control_ledger(
        self, conn: sqlite3.Connection, source_id: str
    ) -> dict:
        """Committed ``source_state`` history + read-time evaluation.

        The artifact ledger (``sourcestate.history``) is the durable
        record of every committed control transition — ``inspect`` reports
        it verbatim rather than synthesizing a story from the latest row
        (V5-15.01).  ``current_state`` adds the label/effective-revision
        view (V5-14.08/14.14).  Both degrade to ``None`` fields, never an
        exception, when the control plane or its registry is absent; a
        digest failure still surfaces as STORE_CORRUPT via the caller.
        """
        out: dict[str, Any] = {"history": None, "evaluation": None}
        if _sourcestate is None or not _has_table(conn, "source_state"):
            return out
        hist_fn = getattr(_sourcestate, "history", None)
        if callable(hist_fn):
            docs = hist_fn(conn, source_id)
            out["history"] = [
                {
                    "control_version": d.get("control_version"),
                    "disposition": d.get("disposition"),
                    "change": d.get("change"),
                    "mutation_head": d.get("mutation_head"),
                    "predecessor_head": d.get("predecessor_head"),
                    "superseded_by": d.get("superseded_by"),
                    "effective_at": d.get("effective_at"),
                    "known_at": d.get("known_at"),
                    "producer": d.get("producer"),
                    "operation_id": d.get("operation_id"),
                }
                for d in docs
            ]
        cur_fn = getattr(_sourcestate, "current_state", None)
        if callable(cur_fn):
            cur = cur_fn(conn, source_id)
            if getattr(cur, "found", False):
                out["evaluation"] = {
                    "label": cur.label,
                    "current": cur.current,
                    "effective_revision": cur.effective_revision,
                    "pending": cur.pending,
                    "disposition": cur.disposition,
                }
        return out

    def _inspect_object(
        self, ref: tuple[str, str, Optional[int]], detail: str
    ) -> Inspection:
        kind, oid, rev = ref
        # normalize_kind is the single kind authority: unknown kinds are
        # a typed VALIDATION, ``fact`` resolves to ``claim``.
        kind = derivations.normalize_kind(kind)
        if kind in ("source", "source_revision"):
            return self._inspect_source_id(
                oid, detail, ref_string=object_ref_to_string(kind, oid, rev)
            )
        if kind == "claim":
            return self._inspect_claim(oid, detail)
        # Any other registered kind (episode, observation, source_state …)
        # resolves through recorded derivation ancestry to its evidence
        # roots — never a guessed parent.
        return self._inspect_via_ancestry(kind, oid, rev, detail)

    def _inspect_claim(self, claim_id: str, detail: str) -> Inspection:
        """Claim object inspection under bound-caller governance.

        Same authority chain as ``retrieval.inspect.inspect_claim`` —
        authorization precedes detail, suppressed/held content is fenced —
        evaluated through the v3 grant surface so opaque namespace scope
        ids (which the v2 ``can_read`` tuple predicate cannot express)
        resolve correctly.
        """
        store = self._store
        with store.read() as conn:
            row = conn.execute(
                "SELECT claim_id, scope_id, subject_id, predicate, row_version"
                " FROM claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if row is None:
                self._deny()
            claim_scope = row[1]
            # Authorize on the object's own scope — grants may reach
            # beyond the bound namespace; denial stays indistinguishable.
            authorize(conn, self._caller, claim_scope, Verb.READ.value, self._purpose)
            if claim_id in self._suppressed_ids(conn, "claim", [claim_id]):
                self._deny()
            quote_ok = Verb.QUOTE.value in effective_verbs(
                conn, self._caller, claim_scope
            )
            revisions = [
                {
                    "revision": r[0],
                    "state": r[1],
                    "polarity": r[2],
                    "modality": r[3],
                    "recorded_from": r[4],
                    "recorded_until": r[5],
                }
                for r in conn.execute(
                    "SELECT revision, state, polarity, modality,"
                    " recorded_from, recorded_until FROM claim_revisions"
                    " WHERE claim_id = ? ORDER BY revision",
                    (claim_id,),
                ).fetchall()
            ]
            intervals = [
                {
                    "revision": r[0],
                    "from": rfc3339(r[1]) if r[1] is not None else None,
                    "until": rfc3339(r[2]) if r[2] is not None else None,
                    "precision": r[3],
                    "basis": r[5],
                }
                for r in conn.execute(
                    "SELECT revision, from_us, until_us, precision, timezone,"
                    " basis FROM valid_intervals WHERE claim_id = ?"
                    " ORDER BY revision, interval_no",
                    (claim_id,),
                ).fetchall()
            ]
            evidence_rows = conn.execute(
                "SELECT ce.revision, ce.span_id, ce.evidence_role,"
                " s.source_id, s.revision, s.start_byte, s.end_byte"
                " FROM claim_evidence ce"
                " JOIN spans s ON s.span_id = ce.span_id"
                " WHERE ce.claim_id = ? ORDER BY ce.revision",
                (claim_id,),
            ).fetchall()
            held_spans = self._suppressed_ids(
                conn, "span", [e[1] for e in evidence_rows]
            )
            env_of: dict = {}
            if _has_table(conn, "source_envelopes"):
                for src, srev in {(e[3], e[4]) for e in evidence_rows}:
                    env_of[(src, srev)] = [
                        r[0]
                        for r in conn.execute(
                            "SELECT envelope_id FROM source_envelopes"
                            " WHERE source_id = ? AND revision = ?",
                            (src, srev),
                        ).fetchall()
                    ]
            evidence = []
            for _r, span_id, role, src, srev, sb, eb in evidence_rows:
                held = (
                    span_id in held_spans
                    or self._held(conn, "span", span_id, srev)
                    or self._held(conn, "source", src, srev)
                    or any(
                        self._held(conn, "source_envelope", eid, srev)
                        for eid in env_of.get((src, srev), ())
                    )
                )
                item = {
                    "span_id": span_id,
                    "role": role,
                    "source_id": src,
                    "source_revision": srev,
                    "start_byte": sb,
                    "end_byte": eb,
                    "held": held,
                }
                if detail == "evidence":
                    text: Optional[str] = None
                    if not held and quote_ok:
                        from ..storage.repos import SpansRepo

                        text = SpansRepo(store).text(span_id)
                    item["text"] = text if text is not None else "[unavailable]"
                evidence.append(item)

        out = Inspection(
            ref=object_ref_to_string("claim", claim_id, None),
            found=True,
            detail=detail,
            provenance={
                "kind": "claim",
                "claim_id": claim_id,
                "scope_id": claim_scope,
                "subject_id": row[2],
                "predicate": row[3],
                "row_version": row[4],
            },
            revisions=revisions,
            lifecycle={
                "valid_intervals": intervals,
                "suppressed": False,
            },
            evidence=evidence,
            enrichment={},
            warnings=(
                []
                if detail != "evidence" or quote_ok
                else [
                    "quote_not_granted: evidence locators returned without "
                    "excerpt text (V5-15.01)"
                ]
            ),
        )
        return out

    def _evidence_root_sources(
        self, conn: sqlite3.Connection, kind: str, oid: str
    ) -> list[str]:
        """Map an evidence-plane derivation root to its source_id(s)."""
        if kind in ("source", "source_revision"):
            return [oid]
        if kind == "span" and _has_table(conn, "spans"):
            return [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT source_id FROM spans WHERE span_id = ?",
                    (oid,),
                ).fetchall()
            ]
        if kind == "envelope" and _has_table(conn, "source_envelopes"):
            return [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT source_id FROM source_envelopes"
                    " WHERE envelope_id = ?",
                    (oid,),
                ).fetchall()
            ]
        # Other evidence kinds (artifact/trajectory/…) resolve only when
        # the id names a live source row directly — no inference.
        row = conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (oid,)
        ).fetchone()
        return [oid] if row is not None else []

    def _inspect_via_ancestry(
        self, kind: str, oid: str, rev: Optional[int], detail: str
    ) -> Inspection:
        """Inspect a non-claim object through its derivation ancestry.

        The object's registered scope (``objects`` row, when present) is
        checked first — a foreign registration denies indistinguishably.
        Resolution itself walks ``derivations.roots`` to evidence-plane
        ancestors; only sources in this binding's scope are inspected.
        """
        with self._store.read() as conn:
            if _has_table(conn, "objects"):
                reg = conn.execute(
                    "SELECT scope_id, disposition, current_revision"
                    " FROM objects WHERE kind = ? AND object_id = ?",
                    (kind, oid),
                ).fetchone()
                if reg is not None and reg[0] != self._scope_id:
                    self._deny()
            if not _has_table(conn, "derivations"):
                raise VerbatimError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"inspection of {kind!r} objects is unavailable — "
                    "no derivation graph on this store",
                )
            roots = derivations.roots(conn, (kind, oid, rev))
            source_ids: list[str] = []
            for rk, rid, _rrev in roots:
                for sid in self._evidence_root_sources(conn, rk, rid):
                    if sid not in source_ids:
                        source_ids.append(sid)
            ours = [
                sid
                for sid in source_ids
                if self._source_visible(conn, sid)
            ]
            if not ours:
                self._deny()
        out = self._inspect_source_id(
            ours[0], detail, ref_string=object_ref_to_string(kind, oid, rev)
        )
        out.provenance["via"] = "derivations"
        out.provenance["object_kind"] = kind
        out.provenance["object_id"] = oid
        out.provenance["object_revision"] = rev
        out.provenance["resolved_source"] = ours[0]
        if len(ours) > 1:
            out.warnings.append(
                f"object ancestry resolves to {len(ours)} sources — "
                "inspection shows the first; inspect siblings by ref"
            )
        return out

    def _source_visible(
        self, conn: sqlite3.Connection, source_id: str
    ) -> bool:
        """Scope+grant visibility probe for ancestry resolution."""
        row = self._source_row(conn, source_id)
        if row is None or row["scope_id"] != self._scope_id:
            return False
        try:
            self._authorize(conn, Verb.READ)
        except VerbatimError:
            return False
        return True

    # ------------------------------------------------------------------
    # forget — argument validation + dispatch (V5-06.18)
    # ------------------------------------------------------------------

    def forget(
        self,
        ref: Any = None,
        *,
        query: Optional[str] = None,
        confirmation: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ForgetResult:
        """Two-surface forget (V5-15.03/04).

        ``forget(ref)`` — targeted suppression + deletion closure for a
        source-backed ``MemoryRef`` (a bare claim/view ref is rejected;
        a bare ``memory_id`` resolves to its current ref and keeps the
        CAS guard). ``forget(query=…)`` — preview only: authorized
        selection + confirmation token, ``mutated=False``.
        ``forget(confirmation=token)`` — executes exactly the pinned
        selection. Replays and conflicts are typed, never silent.
        """
        if confirmation is not None:
            if ref is not None or query is not None:
                raise invalid(
                    "confirmation token stands alone — do not pass "
                    "ref/query with it"
                )
            if not isinstance(confirmation, str) or not confirmation:
                raise invalid("confirmation must be a token string")
            return self._forget_confirmed(confirmation)
        if (ref is None) == (query is None):
            raise invalid("forget needs exactly one of ref or query")
        if query is not None:
            if idempotency_key is not None:
                raise invalid(
                    "idempotency_key applies to executed operations — "
                    "previews mutate nothing"
                )
            return self._forget_preview(query)
        return self._forget_ref(ref, idempotency_key)

    # -- forget: ref path ------------------------------------------------

    @staticmethod
    def _run_id(scope_id: str, source_id: str) -> str:
        """Deterministic continuation id — replay-safe, content-free."""
        return f"v5mem-forget-{scope_id}-{source_id}"

    @staticmethod
    def _op_id(scope_id: str, source_id: str) -> str:
        return f"v5mem:forget:{scope_id}:{source_id}"

    @staticmethod
    def _key_op_id(scope_id: str, key: str) -> str:
        return f"v5mem:forget-key:{scope_id}:{key}"

    @staticmethod
    def _input_digest(doc: Mapping[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(
            json_dumps(doc).encode("utf-8")
        ).hexdigest()

    def _forget_ref(
        self, ref: Any, idempotency_key: Optional[str]
    ) -> ForgetResult:
        kind, parsed = _coerce_ref(ref)
        if kind == "object":
            raise invalid(
                "forget requires a source-backed MemoryRef — bare "
                "claim/view refs are rejected (V5-06.11)"
            )
        if kind == "memory":
            mref: MemoryRef = parsed
            self._check_binding(mref)
            source_id = mref.source_id
            expected_rev: Optional[int] = mref.expected_revision
            expected_ctl: Optional[int] = mref.control_version
        else:  # bare memory_id — resolve current ref, CAS guard kept (V5-15.03)
            mref = None  # type: ignore[assignment]
            source_id = parsed
            expected_rev = None
            expected_ctl = None

        scope_id = self._scope_id
        run_id = self._run_id(scope_id, source_id)
        input_doc = {
            "op": "memory.forget",
            "scope_id": scope_id,
            "ref": mref.to_string() if mref is not None else source_id,
        }
        input_digest = self._input_digest(input_doc)
        key_op = (
            self._key_op_id(scope_id, idempotency_key)
            if idempotency_key is not None
            else None
        )
        replayed = False
        applied_ref = (
            mref.to_string() if mref is not None else ""
        )

        with self._store.tx() as conn:
            row = self._source_row(conn, source_id)
            if row is None or row["scope_id"] != scope_id:
                self._deny()
            self._authorize(conn, Verb.ADMIN)

            if key_op is not None:
                prior = repos_v4.get(
                    conn, "operation_receipts", {"operation_id": key_op}
                )
                if prior is not None:
                    if prior["input_digest"] != input_digest:
                        raise conflict(
                            "idempotency key replayed with a different "
                            "request"
                        )
                    replayed = True

            existing = repos_v4.get(
                conn, "closure_runs", {"run_id": run_id}
            )
            if existing is not None:
                # Deterministic continuation: this exact suppression was
                # already committed — resume it rather than re-CAS an
                # erased tombstone (V5-15.08 restart-safe idempotency).
                replayed = True
            else:
                head_rev, ctl_version, _disp = self._control_head(conn, source_id)
                # Bare-id resolution pins the live head inside this tx —
                # the CAS guard still runs at the suppression commit.
                want_rev = expected_rev if expected_rev is not None else head_rev
                want_ctl = expected_ctl if expected_ctl is not None else ctl_version
                if want_rev != head_rev or want_ctl != ctl_version:
                    raise conflict(
                        "stale memory ref — the memory moved since this "
                        "ref was issued; re-inspect and retry (V5-14.11)"
                    )
                if mref is None:
                    applied_ref = MemoryRef(
                        store_tag=str(self._store_tag),
                        namespace=self._namespace,
                        source_id=source_id,
                        expected_revision=int(head_rev or 0),
                        control_version=ctl_version,
                    ).to_string()
                # Erasure tombstone + suppression commit atomically: the
                # control record fences stale refs/ledger history while
                # the closure run owns physical cleanup (V5-14.16/15.03).
                self._apply_erasure(
                    conn, source_id,
                    expected_rev=want_rev, expected_ctl=want_ctl,
                )
                self._engine.begin(
                    roots=[("source", source_id, None)],
                    scope_id=scope_id,
                    run_id=run_id,
                    conn=conn,
                )
                self._write_receipt(conn, scope_id, self._op_id(scope_id, source_id),
                                    input_digest, run_id)
            if key_op is not None:
                prior_key = repos_v4.get(
                    conn, "operation_receipts", {"operation_id": key_op}
                )
                if prior_key is None:
                    self._write_receipt(
                        conn, scope_id, key_op, input_digest,
                        self._op_id(scope_id, source_id),
                    )

        run, sweep_state = self._drive(run_id, source_id)
        result = self._result_for_run(
            run, sweep_state, receipt_id=self._op_id(scope_id, source_id)
        )
        result.selection = [applied_ref] if applied_ref else []
        if replayed:
            result.warnings.append("replayed: prior forget resumed/reported")
        return result

    def _write_receipt(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        op_id: str,
        input_digest: str,
        result_ref: str,
    ) -> None:
        seq_row = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) FROM events"
        ).fetchone()
        repos_v4.insert(
            conn,
            "operation_receipts",
            {
                "operation_id": op_id,
                "scope_id": scope_id,
                "input_digest": input_digest,
                "result_ref": result_ref,
                "effects_applied": 1,
                "jobs_json": [],
                "applied_seq": int(seq_row[0]) if seq_row else 0,
                "created_us": now_us(),
            },
        )

    def _apply_erasure(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        *,
        expected_rev: Optional[int] = None,
        expected_ctl: Optional[int] = None,
    ) -> None:
        """Fence the source's ``source_state`` record to ``erased``.

        The authoritative path is ``sourcestate.apply_erasure`` — it
        CAS-verifies the fences, bumps the control version, writes the
        immutable ledger doc + provenance edges, and bumps the projection
        generation inside this same transaction (V5-14.10/14.16/15.03).
        The tombstone never deletes the row — it IS the durable proof —
        and an already-erased record returns unchanged (replay-safe).
        The raw-row fallback exists only for builds predating the module.
        """
        if not _has_table(conn, "source_state"):
            return
        eraser = getattr(_sourcestate, "apply_erasure", None)
        if callable(eraser):
            try:
                eraser(
                    conn,
                    source_id,
                    producer=_PRODUCER,
                    namespace=self._namespace,
                    expected_control_version=expected_ctl,
                    expected_revision=expected_rev,
                    store=self._store,
                )
            except VerbatimError as exc:
                if exc.code in (
                    ErrorCode.STALE_DEPENDENCY,
                    ErrorCode.STALE_EPOCH,
                    ErrorCode.INVALID_TRANSITION,
                ):
                    raise conflict(
                        "stale memory ref — the memory moved since this "
                        "ref was issued; re-inspect and retry (V5-14.11)"
                    ) from exc
                raise
            return
        # Pre-v5 fallback: minimal fenced tombstone, same CAS semantics.
        now = rfc3339(now_us())
        if expected_ctl is not None:
            cur = conn.execute(
                "UPDATE source_state SET disposition = 'erased',"
                " control_version = control_version + 1, updated_at = ?"
                " WHERE source_id = ? AND control_version = ?",
                (now, source_id, expected_ctl),
            )
            if cur.rowcount == 0:
                raise conflict(
                    "stale memory ref — control version moved before "
                    "commit (V5-14.11)"
                )
        else:
            cur = conn.execute(
                "UPDATE source_state SET disposition = 'erased',"
                " control_version = control_version + 1, updated_at = ?"
                " WHERE source_id = ?",
                (now, source_id),
            )
            if cur.rowcount == 0:
                conn.execute(
                    "INSERT INTO source_state (source_id, namespace,"
                    " control_version, mutation_head, disposition,"
                    " known_at, updated_at, producer)"
                    " VALUES (?, ?, 1, 'unresolved', 'erased', ?, ?, ?)",
                    (source_id, self._namespace, now, now, _PRODUCER),
                )

    # -- forget: query preview -------------------------------------------

    def _query_terms(self, query: str) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            raise invalid("forget query must be a non-empty string")
        if len(query) > _QUERY_LIMIT:
            raise invalid("forget query exceeds the request bound")
        terms = [t for t in query.lower().split() if len(t) > 1]
        if not terms:
            raise invalid(
                "forget query has no usable terms — a selection that "
                "matches everything is not previewable"
            )
        return terms

    def _forget_preview(self, query: str) -> ForgetResult:
        terms = self._query_terms(query)
        warnings: list[str] = []
        with self._store.read() as conn:
            # Forget authority is required for the preview itself — the
            # token it mints only makes sense under an admin grant, and a
            # read-only caller must not be able to probe the selection.
            self._authorize(conn, Verb.ADMIN)
            candidates = [
                r[0]
                for r in conn.execute(
                    "SELECT source_id FROM sources WHERE scope_id = ?"
                    " ORDER BY source_id LIMIT ?",
                    (self._scope_id, self._scan_limit),
                ).fetchall()
            ]
            suppressed = self._suppressed_ids(conn, "source", candidates)
            selection: list[tuple[str, int, int]] = []
            scanned = 0
            for sid in candidates:
                scanned += 1
                if sid in suppressed:
                    continue
                state = self._source_state(conn, sid)
                if state is not None and state.get("disposition") == "erased":
                    continue
                head_rev, ctl_version, _ = self._control_head(conn, sid)
                if head_rev is None:
                    continue
                if self._held(conn, "source", sid, head_rev):
                    # Held sources stay out of ordinary selection — the
                    # quarantine surface handles them; forget-by-ref
                    # remains the explicit path.
                    continue
                payload_row = conn.execute(
                    "SELECT payload FROM source_revisions"
                    " WHERE source_id = ? AND revision = ?",
                    (sid, head_rev),
                ).fetchone()
                text = (
                    bytes(payload_row[0]).decode("utf-8", errors="replace").lower()
                    if payload_row and payload_row[0]
                    else ""
                )
                if not all(t in text for t in terms):
                    continue
                selection.append((sid, head_rev, ctl_version))
                if len(selection) >= self._select_limit:
                    warnings.append(
                        "selection_truncated: preview bound reached — the "
                        "token covers only the listed refs"
                    )
                    break
            if scanned >= self._scan_limit:
                warnings.append("scan_truncated: source scan bound reached")

        token = self._mint_token(selection)
        refs = [
            MemoryRef(
                store_tag=str(self._store_tag),
                namespace=self._namespace,
                source_id=sid,
                expected_revision=rev,
                control_version=ctl,
            ).to_string()
            for sid, rev, ctl in selection
        ]
        return ForgetResult(
            mode="preview",
            mutated=False,
            confirmation_token=token,
            selection=refs,
            suppression_state="",
            closure_state="",
            receipt_id="",
            warnings=warnings,
        )

    # -- confirmation tokens ----------------------------------------------

    def _mint_token(self, selection: list[tuple[str, int, int]]) -> str:
        payload = {
            "v": 1,
            "kind": "v5-forget-confirm",
            "store": self._store_tag,
            "ns": self._namespace,
            "scope": self._scope_id,
            "caller": self._caller.principal_id,
            "sel": [[s, r, c] for s, r, c in sorted(selection)],
            "exp": now_us() + self._confirm_ttl_us,
            "nonce": new_id(),
        }
        raw = json_dumps(payload).encode("utf-8")
        mac = self._store.hmac(_TOKEN_DOMAIN + raw).hex()
        return f"{_TOKEN_PREFIX}.{_b64e(raw)}.{mac}"

    def _parse_token(self, token: str) -> dict:
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
            raise invalid("malformed confirmation token")
        raw = _b64d(parts[1])
        mac = self._store.hmac(_TOKEN_DOMAIN + raw).hex()
        if not _hmac.compare_digest(mac, parts[2]):
            raise denied("confirmation token failed integrity check")
        try:
            payload = safe_json_loads(raw.decode("utf-8"))
        except (VerbatimError, UnicodeDecodeError) as exc:
            raise invalid("confirmation token payload is not valid json") from exc
        if not isinstance(payload, dict):
            raise invalid("confirmation token payload malformed")
        if (
            payload.get("v") != 1
            or payload.get("kind") != "v5-forget-confirm"
            or not isinstance(payload.get("sel"), list)
            or not isinstance(payload.get("exp"), int)
            or not isinstance(payload.get("nonce"), str)
        ):
            raise invalid("confirmation token fields malformed")
        return payload

    def _forget_confirmed(self, token: str) -> ForgetResult:
        payload = self._parse_token(token)
        # Rebinding checks — every identity in the token must equal this
        # binding; any mismatch denies indistinguishably (V5-06.12).
        if (
            payload.get("store") != self._store_tag
            or payload.get("ns") != self._namespace
            or payload.get("scope") != self._scope_id
            or payload.get("caller") != self._caller.principal_id
        ):
            raise denied("confirmation token is bound to another caller/namespace")
        if now_us() >= int(payload["exp"]):
            raise VerbatimError(
                ErrorCode.PERMIT_EXPIRED,
                "confirmation token expired — re-run the forget preview",
            )

        nonce = payload["nonce"]
        op_id = f"v5mem:forget-confirm:{self._scope_id}:{nonce}"
        members: list[tuple[str, int, int]] = []
        for entry in payload["sel"]:
            if not (
                isinstance(entry, list)
                and len(entry) == 3
                and isinstance(entry[0], str)
                and isinstance(entry[1], int)
                and isinstance(entry[2], int)
            ):
                raise invalid("confirmation token selection malformed")
            members.append((entry[0], entry[1], entry[2]))
        if len(members) > self._select_limit:
            raise invalid("confirmation selection exceeds the bound")

        with self._store.tx() as conn:
            prior = repos_v4.get(
                conn, "operation_receipts", {"operation_id": op_id}
            )
            if prior is not None:
                raise conflict(
                    "confirmation token already consumed — tokens are "
                    "single-use (V5-15.04)"
                )
            self._authorize(conn, Verb.ADMIN)

            run_ids: list[str] = []
            for sid, rev, ctl in members:
                row = self._source_row(conn, sid)
                if row is None or row["scope_id"] != self._scope_id:
                    raise VerbatimError(
                        ErrorCode.STALE_PROPOSAL,
                        "selection changed since preview — re-run the "
                        "forget preview",
                    )
                run_id = self._run_id(self._scope_id, sid)
                existing = repos_v4.get(
                    conn, "closure_runs", {"run_id": run_id}
                )
                if existing is None:
                    head_rev, ctl_version, _ = self._control_head(conn, sid)
                    if head_rev != rev or ctl_version != ctl:
                        raise VerbatimError(
                            ErrorCode.STALE_PROPOSAL,
                            "selection changed since preview — re-run "
                            "the forget preview",
                        )
                run_ids.append(run_id)

            # Consume + suppress atomically: either the whole selection
            # tombstones and the token is spent, or nothing happens and
            # the token stays usable after a re-preview.
            self._write_receipt(
                conn,
                self._scope_id,
                op_id,
                self._input_digest(
                    {"op": "memory.forget-confirm", "token_sha": hashlib.sha256(
                        token.encode("utf-8")).hexdigest()}
                ),
                json_dumps(run_ids),
            )
            for sid, mrev, mctl in members:
                run_id = self._run_id(self._scope_id, sid)
                existing = repos_v4.get(
                    conn, "closure_runs", {"run_id": run_id}
                )
                if existing is None:
                    self._apply_erasure(
                        conn, sid,
                        expected_rev=mrev, expected_ctl=mctl,
                    )
                    self._engine.begin(
                        roots=[("source", sid, None)],
                        scope_id=self._scope_id,
                        run_id=run_id,
                        conn=conn,
                    )
                member_op = self._op_id(self._scope_id, sid)
                if (
                    repos_v4.get(
                        conn, "operation_receipts", {"operation_id": member_op}
                    )
                    is None
                ):
                    self._write_receipt(
                        conn,
                        self._scope_id,
                        member_op,
                        self._input_digest(
                            {
                                "op": "memory.forget",
                                "scope_id": self._scope_id,
                                "ref": sid,
                                "via": op_id,
                            }
                        ),
                        run_id,
                    )

        if not members:
            return ForgetResult(
                mode="operation",
                mutated=False,
                confirmation_token="",
                selection=[],
                suppression_state="none",
                closure_state="none",
                receipt_id=op_id,
                warnings=["empty_selection: preview selected nothing"],
            )
        runs = []
        sweeps = []
        for sid, _r, _c in members:
            run, sweep = self._drive(self._run_id(self._scope_id, sid), sid)
            runs.append(run)
            sweeps.append(sweep)
        result = self._result_for_runs(runs, sweeps, receipt_id=op_id)
        result.selection = [
            MemoryRef(
                store_tag=str(self._store_tag),
                namespace=self._namespace,
                source_id=sid,
                expected_revision=rev,
                control_version=ctl,
            ).to_string()
            for sid, rev, ctl in members
        ]
        return result

    # -- closure drive + derived-data sweep --------------------------------

    def _drive(
        self, run_id: str, source_id: str
    ) -> tuple[Any, str]:
        """Drain the closure run then sweep v5 derived rows.

        ``_drive`` is restart-safe: a redelivered call resumes the same
        deterministic run, and the sweep is an idempotent delete set.
        Returns ``(run, sweep_state)``.
        """
        kwargs: dict = {}
        if self._drain_budget is not None:
            kwargs["budget"] = self._drain_budget
        if self._drain_max_steps is not None:
            kwargs["max_steps"] = self._drain_max_steps
        try:
            run = self._engine.drain(run_id, **kwargs)
        except VerbatimError as exc:
            if exc.code == ErrorCode.CLOSURE_PENDING:
                run = self._engine.status(run_id)
            else:
                raise
        if run.phase == ClosurePhase.FAILED:
            # One resume attempt heals a transient failed unit; a second
            # failure reports failed_cleanup honestly (V5-15.07).
            try:
                self._engine.resume(run_id)
                run = self._engine.drain(run_id, **kwargs)
            except VerbatimError as exc:
                if exc.code == ErrorCode.CLOSURE_PENDING:
                    run = self._engine.status(run_id)
                else:
                    run = self._engine.status(run_id)
        sweep_state = self._sweep_derived(source_id)
        return run, sweep_state

    def _sweep_derived(self, source_id: str) -> str:
        """Delete the v5 derived rows + re-assert the ``source_state`` tombstone.

        These tables are regenerable projections (V5-11.11): the closure
        engine erases the canonical store layers (payloads, spans, claims,
        FTS, envelopes); this sweep covers the v5 control-plane layers it
        does not own — lexical projection, the ``source_fts``/
        ``source_fts_rows`` external-content pair (whose delete trigger
        maintains the ``source_fts_idx`` shadow), vectors, entity
        postings, enrichment, duplicate links (memberships *and* rows
        where this source anchored the group — the anchor id must not
        outlive it), and update candidates on either endpoint.  The
        ``source_state`` erasure tombstone is re-asserted idempotently so
        a resumed run heals the commit/sweep window; the ledger docs and
        provenance edges under ``objects``/``object_revisions``/
        ``derivations`` are the fence proof and are never swept.
        """
        with self._store.tx() as conn:
            swept: list[str] = []
            # FTS pair first: the content row's delete trigger cleans the
            # shadow index, and it must go before its carrier (FK).
            if _has_table(conn, "source_fts") and _has_table(
                conn, "source_fts_rows"
            ):
                cur = conn.execute(
                    "DELETE FROM source_fts WHERE fts_row_id IN"
                    " (SELECT row_id FROM source_fts_rows"
                    "  WHERE source_id = ?)",
                    (source_id,),
                )
                if cur.rowcount:
                    swept.append("source_fts")
            for base in _V5_SOURCE_TABLES:
                swept += self._delete_source_rows(conn, base, source_id)
            if _has_table(conn, "duplicate_links"):
                cols = self._table_cols(conn, "duplicate_links")
                if "group_id" in cols:
                    conn.execute(
                        "DELETE FROM duplicate_links WHERE group_id = ?",
                        (source_id,),
                    )
            if _has_table(conn, "update_candidates"):
                cols = self._table_cols(conn, "update_candidates")
                if {"new_source_id", "prior_source_id"} <= cols:
                    conn.execute(
                        "DELETE FROM update_candidates"
                        " WHERE new_source_id = ? OR prior_source_id = ?",
                        (source_id, source_id),
                    )
                    swept.append("update_candidates")
            # V7 derived plane (SPEC_V7 §30): the closure drain already
            # swept each emptied revision through purge._empty_revision;
            # this whole-source pass catches rows minted between drain
            # and sweep and is a no-op when nothing remains (idempotent).
            if _has_table(conn, "units"):
                from ..privacy import closure_v7 as _closure_v7

                v7 = _closure_v7.delete_source_v7(conn, source_id)
                if v7["units_removed"]:
                    swept.append("v7")
            # Idempotent tombstone re-assert — the closure-membership
            # hook returns the existing tombstone unchanged (V5-14.16).
            if _has_table(conn, "source_state"):
                self._apply_erasure(conn, source_id)
                swept.append("source_state")
        return "swept:" + ",".join(sorted(set(swept))) if swept else "swept:none"

    @staticmethod
    def _table_cols(conn: sqlite3.Connection, table: str) -> set:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    def _delete_source_rows(
        self, conn: sqlite3.Connection, base: str, source_id: str
    ) -> list[str]:
        """DELETE base-table + shadow-table rows for a source.

        Shadow discovery covers the ``<base>_*`` names an FTS5/postings
        shadow would take; only tables that actually carry a
        ``source_id`` column are touched — every check is live catalog
        state, never an assumed schema.
        """
        touched: list[str] = []
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                " AND (name = ? OR name LIKE ?)",
                (base, base + r"\_%"),
            ).fetchall()
        ]
        for name in names:
            if "source_id" not in self._table_cols(conn, name):
                continue
            conn.execute(
                f'DELETE FROM "{name}" WHERE source_id = ?', (source_id,)
            )
            touched.append(name)
        return touched

    # -- result assembly ---------------------------------------------------

    def _result_for_run(
        self, run: Any, sweep_state: str, *, receipt_id: str
    ) -> ForgetResult:
        return self._result_for_runs([run], [sweep_state], receipt_id=receipt_id)

    def _result_for_runs(
        self, runs: list, sweeps: list[str], *, receipt_id: str
    ) -> ForgetResult:
        warnings: list[str] = []
        phases = {r.phase for r in runs}
        if ClosurePhase.FAILED in phases:
            closure_state = ClosurePhase.FAILED.value
            warnings.append(
                "closure failed_cleanup — suppression holds; resume via "
                "the receipt run id"
            )
        elif ClosurePhase.CLEANING in phases:
            closure_state = ClosurePhase.CLEANING.value
            warnings.append(
                "closure pending — suppression committed; physical "
                "cleanup continues on resume"
            )
        elif ClosurePhase.VERIFYING in phases:
            closure_state = ClosurePhase.VERIFYING.value
        elif all(r.phase == ClosurePhase.COMPLETED for r in runs):
            closure_state = ClosurePhase.COMPLETED.value
        else:
            closure_state = "pending"
        if any(s != "swept:none" for s in sweeps):
            closure_state = f"{closure_state}+derived_swept"
        suppression = self._purge_state(runs)
        return ForgetResult(
            mode="operation",
            mutated=True,
            confirmation_token="",
            suppression_state=suppression,
            closure_state=closure_state,
            receipt_id=receipt_id,
            warnings=warnings,
        )

    def _purge_state(self, runs: list) -> str:
        """Aggregate suppression state from the engine-owned purge rows."""
        states: set = set()
        with self._store.read() as conn:
            for run in runs:
                row = conn.execute(
                    "SELECT state FROM purges WHERE purge_id = ?",
                    (f"purge-{run.run_id}",),
                ).fetchone()
                if row is not None:
                    states.add(row[0])
        if not states:
            return "suppressed"
        if states == {"completed"}:
            return "completed"
        if "purging" in states:
            return "purging"
        return "suppressed"


__all__ = [
    "MemoryControls",
    "TOKEN_TTL_US",
    "object_ref_to_string",
]
