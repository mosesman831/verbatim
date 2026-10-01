"""Scope export/import: portable evidence bundles (SPEC_V2 §10, §42).

``export_scope`` produces a versioned JSON bundle of one scope's evidence —
claims + revisions, intervals, spans, source metadata with payload bytes
(base64), edges, entities, and event *metadata* — each record carrying an
integrity digest, plus a manifest with counts and the scope identity
(V2-42.09). The export never contains credentials, consent rows, HMAC key
material, decision/outbound bodies, jobs, or other scopes' data (V2-42.10),
and suppressed evidence is excluded by default.

``import_bundle`` re-owns bundle objects into a target scope: new local ids
are minted and mapped (foreign grants never become local authority,
V2-42.13), origin provenance is preserved as ``legacy_import`` plus an
explicit import-provenance field, and the erasure ledger fences every
object — a purged id in a restored bundle is skipped, never recreated
(V2-40.14, V2-41.15). Repeating an import is idempotent by bundle
origin/object via the operations ledger (V2-42.15).

Import performs no network access and runs inside the caller's
transaction: validation (manifest, counts, per-record digests, payload
hashes, enum membership, span bounds) happens before any write, so a
rejected bundle leaves the store untouched.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
from typing import Any, Optional, Union

from .core.time import wall_us
from .core.types import (
    CallerContext,
    EdgeType,
    ErrorCode,
    GrantKind,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from .storage.repos import (
    EventsRepo,
    PurgesRepo,
    SourcesRepo,
    SpansRepo,
    ensure_scope,
    has_table as _has_table,
)
from .storage.repos_v2 import ErasureRepo, OperationsRepo, SourceViewsRepo

EXPORT_FORMAT_VERSION = 1

#: Import acceptance bounds (V2-42.11): oversized or unbounded bundles are
#: rejected before anything is written.
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_MAX_RECORDS = 200_000

_SOURCE_KINDS = (
    "user_message",
    "assistant_message",
    "tool_output",
    "import",
    "operator_record",
)
_LIFECYCLE_STATES = (
    "pending", "active", "disputed", "superseded", "rejected", "archived", "erased",
)
_EVIDENCE_ROLES = ("primary", "contextual")
_PRECISIONS = ("instant", "day", "month", "year", "unknown")
_ENDPOINT_KINDS = ("exact", "uncertain_range", "unbounded", "unknown")
_INTERVAL_BASES_MIN = 0
_IMPORT_POLICY_VERSION = "policy-1"

#: Sections never exported — credentials, authority, derived caches, and
#: other operational tables stay in the store (V2-42.10). Documented here so
#: the exclusion is auditable rather than implicit.
NEVER_EXPORTED = (
    "consents",
    "budget_ledger",
    "disclosures",
    "jobs",
    "job_events",
    "decisions",
    "decision_inputs",
    "reviews",
    "purges",
    "purge_targets",
    "erasure_ledger",
    "embeddings",
    "embedding_inputs",
    "encoder_manifests",
    "policy_artifacts",
    "fts_rows",
    "facts_fts",
    "handoff_capsules",
    "capsule_members",
    "operations",
    "connector_cursors",
    "ingest_batches",
    "migration_history",
    "meta",
    # v5 source-projection plane — derived caches rebuilt by the project
    # jobs; bundles ship the evidence sections, not the projections.
    "source_state",
    "source_lexical_projection",
    "source_fts_rows",
    "source_fts",
    "source_vectors",
    "entity_postings",
    "duplicate_links",
    "enrichment",
    "update_candidates",
    "backfill_cursor",
    # v7 unit-projection plane (schema_v7 §30 — physical names incl. the
    # events/entity_aliases collision renames): the entire plane is
    # recomputable from sources + source_revisions, so it never rides a
    # bundle. screening_log/run_manifests are operational/eval journals,
    # not evidence — same exclusion class as ``decisions``/``operations``.
    "units",
    "unit_fts_rows",
    "unit_fts_content",
    "unit_fts",
    "unit_fts_stem",
    "unit_fts_tri",
    "lex_stats",
    "lex_df",
    "unit_vectors_block",
    # Provisional per-row f32 oracle (embeddings.matrix.ORACLE_TABLE) —
    # same derived-cache class as the block matrix: rebuilt by
    # source_embed, never shipped in a bundle.
    "unit_vectors",
    "entity_canon",
    "entity_mentions",
    "entity_aliases_v7",
    "graph_edges",
    "events_v7",
    "state_facts",
    "preferences",
    "standing_rules",
    "observations_v7",
    "profiles_v7",
    "standing_queries",
    "t2_facts",
    "screening_log",
    "run_manifests",
)


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(
        r[1] == col for r in conn.execute(f"PRAGMA table_info({table})")
    )


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json_dumps(obj).encode("utf-8")).hexdigest()


def _record_digest(record: dict[str, Any]) -> str:
    """Integrity digest over the record minus its own digest field."""
    body = {k: v for k, v in record.items() if k != "digest"}
    return _sha256_json(body)


def _scope_id_for(store: Any, scope: Union[Scope, str]) -> str:
    if isinstance(scope, Scope):
        from .core.identity import scope_key

        return scope_key(scope)
    return require_id(scope, "scope_id")


def _scope_row(conn: sqlite3.Connection, scope_id: str) -> Optional[dict[str, Any]]:
    rows = _rows(
        conn.execute(
            "SELECT scope_id, profile_id, principal_id, workspace_id,"
            " conversation_id, visibility FROM scopes WHERE scope_id = ?",
            (scope_id,),
        )
    )
    return rows[0] if rows else None


def _fenced(
    store: Any,
    conn: sqlite3.Connection,
    scope_ids: set[str],
    object_kind: str,
    object_id: str,
) -> bool:
    """Erasure-ledger check across every scope the object could be filed in."""
    if not _has_table(conn, "erasure_ledger"):
        return False
    repo = ErasureRepo(store)
    return any(
        repo.is_erased(conn, sid, object_kind, object_id) for sid in scope_ids
    )


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


def export_scope(
    store: Any,
    scope: Union[Scope, str],
    out_path: Optional[str] = None,
    *,
    portable: bool = True,
    caller: Optional[CallerContext] = None,
) -> dict[str, Any]:
    """Export one scope's evidence as a versioned JSON bundle.

    ``portable=True`` embeds payload bytes (base64) so the bundle can
    re-materialize evidence elsewhere; ``False`` emits a metadata/digest-only
    manifest. When ``caller`` is supplied it must hold ``GrantKind.EXPORT``
    and belong to the scope's profile (V2-09.04) — a None caller is the
    trusted in-process boundary (V2-09.03).
    """
    sid = _scope_id_for(store, scope)
    with store.read() as conn:
        srow = _scope_row(conn, sid)
        if srow is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "scope not found"
            )
        if caller is not None:
            if not caller.has(GrantKind.EXPORT):
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                    "caller missing export grant",
                )
            if caller.profile_id != srow["profile_id"]:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "caller outside profile"
                )
        purges = PurgesRepo(store)
        sections = _collect_sections(conn, store, purges, sid, portable)
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "scope_id": sid,
        "scope": {
            "profile_id": srow["profile_id"],
            "principal_id": srow["principal_id"],
            "workspace_id": srow["workspace_id"],
            "conversation_id": srow["conversation_id"],
            "visibility": srow["visibility"],
        },
        "db_id": _safe_db_id(store),
        "exported_us": wall_us(),
        "portable": bool(portable),
        "counts": {k: len(v) for k, v in sections.items()},
        "records_sha256": _sha256_json(sections),
    }
    bundle: dict[str, Any] = {"manifest": manifest}
    bundle.update(sections)
    if out_path is not None:
        _write_bundle(out_path, bundle)
    return bundle


def _safe_db_id(store: Any) -> Optional[str]:
    try:
        return store.db_id()
    except Exception:
        return None


def _suppressed(
    purges: PurgesRepo, kind: str, ids: list[str]
) -> set[str]:
    try:
        return purges.suppressed_ids(kind, ids)
    except Exception:
        # Fail closed on the exclusion side: a broken tombstone lookup must
        # hide everything rather than leak suppressed evidence.
        return set(ids)


#: Quarantine states that hide an object revision (mirrors
#: ``security.EXCLUDING_STATES``; re-declared so the fallback path never
#: imports a parallel module — same convention as retrieval/v3 union).
_HELD_STATES = frozenset({"pending", "suppressed"})


def _held(conn: sqlite3.Connection, object_kind: str, object_id: str,
          revision: int) -> bool:
    """Quarantine exclusion for one object revision (V3-14.10).

    ``security.should_exclude`` when the parallel module is provisioned,
    the quarantine-table state otherwise — the same defensive pair
    ``retrieval/v3/union.py::_should_exclude`` uses. Fails closed on
    unreadable state; callers gate on ``_has_table(conn, "quarantine")``
    so schemas that predate the table are unaffected.
    """
    try:
        from . import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(
                _security.should_exclude(conn, object_kind, object_id, revision)
            )
        except Exception:
            pass  # fall back to the local table check below
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (object_kind, object_id, revision),
        ).fetchone()
    except sqlite3.Error:
        return True  # no readable quarantine state — fail closed
    return row is not None and row[0] in _HELD_STATES


def _source_revision_held(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> bool:
    """Quarantine hold on a source revision or a covering envelope.

    A hold on ``("source", sid, rev)`` or on any ``source_envelopes`` row
    for ``(sid, rev)`` withholds the revision's bytes (V3-14.10) — the
    same resolution retrieval/v3's evidence cascade performs.
    """
    if not _has_table(conn, "quarantine"):
        return False  # schema predates quarantine — no holds exist
    if _held(conn, "source", source_id, revision):
        return True
    # A span-level hold withholds the containing revision's bytes too —
    # the bundle ships whole payloads (``payload_b64`` under portable)
    # and cannot splice out a held byte range, so fail closed (V4-36.03).
    if _has_table(conn, "spans"):
        span_rows = conn.execute(
            "SELECT span_id FROM spans WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchall()
        if any(_held(conn, "span", r[0], revision) for r in span_rows):
            return True
    if not _has_table(conn, "source_envelopes"):
        return False
    rows = conn.execute(
        "SELECT envelope_id FROM source_envelopes"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchall()
    return any(
        _held(conn, "source_envelope", r[0], revision) for r in rows
    )


def _claim_held(
    conn: sqlite3.Connection, purges: PurgesRepo, claim_id: str
) -> bool:
    """Evidence-chain hold cascade for claim export (V3-14.10).

    Withhold the whole claim record when any cited span — its source
    revision or a covering source envelope — is tombstoned or
    quarantined, or a revision of the claim itself is quarantined. The
    claim's ``object_json`` carries the same text the hold covers, so a
    hold anywhere in the chain taints the interpretation too.
    """
    ev = conn.execute(
        "SELECT ce.span_id, s.source_id, s.revision FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id WHERE ce.claim_id = ?",
        (claim_id,),
    ).fetchall()
    span_ids = [r[0] for r in ev]
    if span_ids and _suppressed(purges, "span", span_ids):
        return True
    src_ids = list({r[1] for r in ev})
    if src_ids and _suppressed(purges, "source", src_ids):
        return True
    rev_keys = [f"{r[1]}:{r[2]}" for r in ev]
    if rev_keys and _suppressed(purges, "source_revision", rev_keys):
        return True
    if not _has_table(conn, "quarantine"):
        return False
    for span_id, src, rev in ev:
        if _held(conn, "span", span_id, rev):
            return True
        if _source_revision_held(conn, src, rev):
            return True
    revisions = [
        r[0]
        for r in conn.execute(
            "SELECT revision FROM claim_revisions WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    ]
    return any(_held(conn, "claim", claim_id, r) for r in revisions)


def _collect_sections(
    conn: sqlite3.Connection,
    store: Any,
    purges: PurgesRepo,
    sid: str,
    portable: bool,
) -> dict[str, list[dict[str, Any]]]:
    sources = _rows(
        conn.execute(
            "SELECT source_id, origin, external_id, source_kind, scope_id,"
            " speaker_id, created_us FROM sources WHERE scope_id = ?"
            " ORDER BY source_id",
            (sid,),
        )
    )
    src_supp = _suppressed(purges, "source", [s["source_id"] for s in sources])
    live_sources = [s for s in sources if s["source_id"] not in src_supp]
    sources_repo = SourcesRepo(store)
    revs_of: dict[str, list[dict[str, Any]]] = {}
    rev_tombstone_keys: list[str] = []
    for s in live_sources:
        # Metadata columns only — evidence bytes come through
        # ``SourcesRepo.payload``, which re-verifies ``payload_hmac``
        # before releasing them (V4-07.02/V4-08.03). The compat surface
        # never selects the plaintext column directly.
        revs = _rows(
            conn.execute(
                "SELECT source_id, revision, event_us, captured_us,"
                " timezone, provenance, metadata_json FROM source_revisions"
                " WHERE source_id = ? ORDER BY revision",
                (s["source_id"],),
            )
        )
        revs_of[s["source_id"]] = revs
        rev_tombstone_keys.extend(
            f"{s['source_id']}:{r['revision']}" for r in revs
        )
    # Revision-scoped tombstones name the exact erased bytes as
    # "<source_id>:<revision>" (V2-41.04) — the same object-id convention
    # purge.py writes into purge_targets.
    rev_supp = _suppressed(purges, "source_revision", rev_tombstone_keys)
    out_sources: list[dict[str, Any]] = []
    span_rows_all: list[dict[str, Any]] = []
    held_revs: set[tuple[str, int]] = set()
    for s in live_sources:
        rev_out: list[dict[str, Any]] = []
        for r in revs_of[s["source_id"]]:
            rev_key = f"{s['source_id']}:{r['revision']}"
            if rev_key in rev_supp or _source_revision_held(
                conn, s["source_id"], r["revision"]
            ):
                # Tombstoned in the suppress→execute window, or under a
                # quarantine hold on the revision or its covering
                # envelope — nothing to export (V3-14.10). The payload
                # (and even its fingerprint) stays out of the bundle.
                held_revs.add((s["source_id"], r["revision"]))
                continue
            # Digest-verified read (SourcesRepo.payload re-checks
            # payload_hmac): a tampered revision fails the whole export
            # STORE_CORRUPT rather than shipping forged bytes (C03).
            payload = sources_repo.payload(s["source_id"], r["revision"])
            if payload is None or len(payload) == 0:
                continue  # absent or physically purged — nothing to export
            rec = {
                "revision": r["revision"],
                "event_us": r["event_us"],
                "captured_us": r["captured_us"],
                "timezone": r["timezone"],
                "provenance": r["provenance"],
                "metadata": safe_json_loads(r["metadata_json"] or "{}"),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
            if portable:
                rec["payload_b64"] = base64.b64encode(payload).decode("ascii")
            rev_out.append(rec)
        if not rev_out:
            continue
        rec = {
            "source_id": s["source_id"],
            "origin": s["origin"],
            "external_id": s["external_id"],
            "source_kind": s["source_kind"],
            "speaker_id": s["speaker_id"],
            "created_us": s["created_us"],
            "revisions": rev_out,
        }
        rec["digest"] = _record_digest(rec)
        out_sources.append(rec)
        span_cols = "span_id, source_id, revision, start_byte, end_byte, harvester_version"
        if _has_col(conn, "spans", "view_id"):
            span_cols += ", view_id, operation_key"
        span_rows_all.extend(
            _rows(
                conn.execute(
                    f"SELECT {span_cols} FROM spans WHERE source_id = ?"
                    " ORDER BY span_id",
                    (s["source_id"],),
                )
            )
        )

    span_supp = _suppressed(
        purges, "span", [r["span_id"] for r in span_rows_all]
    )
    has_quarantine = _has_table(conn, "quarantine")
    out_spans: list[dict[str, Any]] = []
    for sp in span_rows_all:
        if sp["span_id"] in span_supp:
            continue
        if (sp["source_id"], sp["revision"]) in held_revs:
            continue  # span sits on a held revision — withheld with it
        if has_quarantine and _held(
            conn, "span", sp["span_id"], sp["revision"]
        ):
            continue  # quarantine hold on the span itself (V3-14.10)
        rec = dict(sp)
        rec["digest"] = _record_digest(rec)
        out_spans.append(rec)

    claim_cols = "claim_id, scope_id, subject_id, predicate, created_event"
    if _has_col(conn, "claims", "interpretation_status"):
        claim_cols += ", interpretation_status"
    claim_rows = _rows(
        conn.execute(
            f"SELECT {claim_cols} FROM claims WHERE scope_id = ?"
            " ORDER BY claim_id",
            (sid,),
        )
    )
    claim_supp = _suppressed(
        purges, "claim", [c["claim_id"] for c in claim_rows]
    )
    out_claims: list[dict[str, Any]] = []
    for c in claim_rows:
        cid = c["claim_id"]
        if cid in claim_supp:
            continue
        if _claim_held(conn, purges, cid):
            continue  # evidence chain under a hold — withheld whole
        rec = _export_claim(conn, c)
        if rec is None:
            continue  # erased head — a tombstone, not evidence
        out_claims.append(rec)

    entity_rows = _rows(
        conn.execute(
            "SELECT entity_id, scope_id, kind, label, created_event"
            " FROM entities WHERE scope_id = ? ORDER BY entity_id",
            (sid,),
        )
    )
    out_entities: list[dict[str, Any]] = []
    for e in entity_rows:
        aliases = _rows(
            conn.execute(
                "SELECT normalized_alias, source_span_id FROM entity_aliases"
                " WHERE entity_id = ? ORDER BY normalized_alias",
                (e["entity_id"],),
            )
        )
        rec = {
            "entity_id": e["entity_id"],
            "kind": e["kind"],
            "label": e["label"],
            "created_event": e["created_event"],
            "aliases": aliases,
        }
        rec["digest"] = _record_digest(rec)
        out_entities.append(rec)

    edge_rows = _rows(
        conn.execute(
            "SELECT edge_id, source_kind, source_id, target_kind, target_id,"
            " edge_type, decision_id, created_event, retired_event FROM edges"
            " WHERE scope_id = ? ORDER BY edge_id",
            (sid,),
        )
    )
    out_edges: list[dict[str, Any]] = []
    for e in edge_rows:
        rec = dict(e)
        rec["digest"] = _record_digest(rec)
        out_edges.append(rec)

    event_rows = _rows(
        conn.execute(
            "SELECT event_seq, kind, actor_id, recorded_us, policy_version"
            " FROM events WHERE scope_id = ? ORDER BY event_seq",
            (sid,),
        )
    )
    # Metadata only — event payloads can carry derived values and are not
    # part of the portable evidence set.
    out_events = [dict(r) for r in event_rows]

    return {
        "sources": out_sources,
        "spans": out_spans,
        "claims": out_claims,
        "entities": out_entities,
        "edges": out_edges,
        "events": out_events,
    }


def _export_claim(conn: sqlite3.Connection, c: dict[str, Any]) -> Optional[dict[str, Any]]:
    cid = c["claim_id"]
    rev_cols = (
        "revision, state, object_json, polarity, modality,"
        " condition_json, interpretation_json, recorded_from, recorded_until"
    )
    for extra in ("subject_id", "predicate", "registry_version",
                  "interpretation_status", "method"):
        if _has_col(conn, "claim_revisions", extra):
            rev_cols += f", {extra}"
    revs = _rows(
        conn.execute(
            f"SELECT {rev_cols} FROM claim_revisions"
            " WHERE claim_id = ? ORDER BY revision",
            (cid,),
        )
    )
    if not revs:
        return None
    if revs[-1]["state"] == "erased":
        return None  # purged tombstone — never exported (V2-42.10)
    out_revs: list[dict[str, Any]] = []
    iv_cols = (
        "interval_no, from_us, until_us, precision, timezone, basis,"
        " uncertainty_json"
    )
    for extra in ("start_kind", "end_kind", "from_us_hi", "until_us_hi"):
        if _has_col(conn, "valid_intervals", extra):
            iv_cols += f", {extra}"
    for r in revs:
        intervals = _rows(
            conn.execute(
                f"SELECT {iv_cols} FROM valid_intervals"
                " WHERE claim_id = ? AND revision = ? ORDER BY interval_no",
                (cid, r["revision"]),
            )
        )
        for iv in intervals:
            iv["uncertainty"] = safe_json_loads(iv.pop("uncertainty_json") or "null")
        evidence = _rows(
            conn.execute(
                "SELECT span_id, evidence_role FROM claim_evidence"
                " WHERE claim_id = ? AND revision = ? ORDER BY span_id",
                (cid, r["revision"]),
            )
        )
        out_revs.append(
            {
                "revision": r["revision"],
                "state": r["state"],
                "object": safe_json_loads(r["object_json"]) if r["object_json"] else None,
                "polarity": r["polarity"],
                "modality": r["modality"],
                "condition": safe_json_loads(r["condition_json"]) if r["condition_json"] else None,
                "interpretation": safe_json_loads(r["interpretation_json"]) if r["interpretation_json"] else None,
                "recorded_from": r["recorded_from"],
                "recorded_until": r["recorded_until"],
                "subject_id": r.get("subject_id"),
                "predicate": r.get("predicate"),
                "registry_version": r.get("registry_version"),
                "interpretation_status": r.get("interpretation_status"),
                "method": r.get("method"),
                "intervals": intervals,
                "evidence": evidence,
            }
        )
    entity_links = _rows(
        conn.execute(
            "SELECT entity_id, role, span_id FROM claim_entities"
            " WHERE claim_id = ? ORDER BY entity_id, role",
            (cid,),
        )
    )
    rec = {
        "claim_id": cid,
        "subject_id": c["subject_id"],
        "predicate": c["predicate"],
        "created_event": c["created_event"],
        "interpretation_status": c.get("interpretation_status") or "structured",
        "entity_links": entity_links,
        "revisions": out_revs,
    }
    rec["digest"] = _record_digest(rec)
    return rec


def _write_bundle(out_path: str, bundle: dict[str, Any]) -> str:
    """Write the bundle with store-artifact permissions (0o600)."""
    abspath = os.path.abspath(out_path)
    if os.path.islink(abspath):
        raise VerbatimError(ErrorCode.VALIDATION, "export path is a symlink")
    if os.path.isdir(abspath):
        raise VerbatimError(ErrorCode.VALIDATION, "export path is a directory")
    parent = os.path.dirname(abspath) or "."
    if not os.path.isdir(parent):
        try:
            os.makedirs(parent, mode=0o700)
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED,
                f"cannot create export directory {parent}: {exc}",
            ) from exc
    data = json_dumps(bundle).encode("utf-8")
    if len(data) > _MAX_BUNDLE_BYTES:
        raise VerbatimError(
            ErrorCode.EVIDENCE_TOO_LARGE, "export bundle exceeds size bound"
        )
    try:
        fd = os.open(abspath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    except OSError as exc:
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED, f"cannot write export: {exc}"
        ) from exc
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED, f"cannot write export: {exc}"
        ) from exc
    return abspath


def _table_cols(conn: sqlite3.Connection, cache: dict[str, set[str]], table: str) -> set[str]:
    """Actual columns of a table — INSERTs are filtered to what exists so the
    same import path works on v1 and v2 stores."""
    cols = cache.get(table)
    if cols is None:
        cols = {
            r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        cache[table] = cols
    return cols


def _insert_filtered(
    conn: sqlite3.Connection,
    cache: dict[str, set[str]],
    table: str,
    row: dict[str, Any],
) -> None:
    """INSERT only the keys that are real columns of ``table``."""
    cols = _table_cols(conn, cache, table)
    items = [(k, v) for k, v in row.items() if k in cols]
    names = ", ".join(k for k, _v in items)
    ph = ", ".join("?" for _ in items)
    conn.execute(
        f"INSERT INTO {table}({names}) VALUES ({ph})", tuple(v for _k, v in items)
    )


# ----------------------------------------------------------------------
# import
# ----------------------------------------------------------------------


def import_bundle(
    store: Any,
    conn: Optional[sqlite3.Connection],
    bundle: Any,
    target_scope: Union[Scope, str],
    actor: str,
) -> dict[str, Any]:
    """Import a bundle into ``target_scope`` with remapped ownership.

    Runs inside the caller's ``conn`` transaction (or opens one when
    ``conn`` is None): validation completes before any write, and per-object
    work records an operations-ledger receipt so a retried import replays
    durable results instead of duplicating them.
    """
    require_id(actor, "actor")
    parsed = _parse_bundle(bundle)
    if conn is None:
        with store.tx() as owned:
            return _import(store, owned, parsed, target_scope, actor)
    return _import(store, conn, parsed, target_scope, actor)


def _parse_bundle(bundle: Any) -> dict[str, Any]:
    """Accept a dict, JSON text/bytes, or a filesystem path to JSON."""
    obj = bundle
    if isinstance(obj, (str, bytes)):
        if isinstance(obj, str) and not obj.lstrip().startswith("{"):
            # Treat as a path; refuse directories/symlinks like other stores.
            abspath = os.path.abspath(obj)
            if os.path.islink(abspath):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "bundle path is a symlink"
                )
            if not os.path.isfile(abspath):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "bundle is neither JSON nor a file"
                )
            if os.path.getsize(abspath) > _MAX_BUNDLE_BYTES:
                raise VerbatimError(
                    ErrorCode.EVIDENCE_TOO_LARGE, "bundle file exceeds bound"
                )
            with open(abspath, "rb") as fh:
                obj = fh.read()
        if isinstance(obj, bytes):
            if len(obj) > _MAX_BUNDLE_BYTES:
                raise VerbatimError(
                    ErrorCode.EVIDENCE_TOO_LARGE, "bundle exceeds bound"
                )
            try:
                obj = obj.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "bundle is not UTF-8 JSON"
                ) from exc
        obj = safe_json_loads(obj, max_bytes=_MAX_BUNDLE_BYTES)
    if not isinstance(obj, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "bundle must be a JSON object")
    return obj


def _validate_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "bundle missing manifest")
    if manifest.get("format_version") != EXPORT_FORMAT_VERSION:
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            f"unsupported bundle format_version {manifest.get('format_version')!r}",
        )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "manifest missing counts")
    for section in ("sources", "spans", "claims", "entities", "edges", "events"):
        rows = bundle.get(section)
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"bundle section {section!r} must be a list"
            )
        declared = counts.get(section)
        if (
            len(rows) > _MAX_RECORDS
            or not isinstance(declared, int)
            or declared != len(rows)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"manifest count mismatch for {section!r}",
            )
    return manifest


def _verify_digest(record: dict[str, Any], what: str) -> None:
    digest = record.get("digest")
    if not isinstance(digest, str) or digest != _record_digest(record):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"integrity digest mismatch on {what}"
        )


def _import(
    store: Any,
    conn: sqlite3.Connection,
    bundle: dict[str, Any],
    target_scope: Union[Scope, str],
    actor: str,
) -> dict[str, Any]:
    manifest = _validate_manifest(bundle)
    if isinstance(target_scope, Scope):
        sid = ensure_scope(store, conn, target_scope)
    else:
        sid = _scope_id_for(store, target_scope)
        if _scope_row(conn, sid) is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_FORBIDDEN, "target scope not found"
            )
    origin_sid = manifest.get("scope_id")
    fence_scopes = {sid}
    if isinstance(origin_sid, str) and origin_sid:
        fence_scopes.add(origin_sid)

    fp = str(manifest.get("records_sha256") or _sha256_json(bundle))[:16]
    ops = OperationsRepo(store) if _has_table(conn, "operations") else None
    views = SourceViewsRepo(store) if _has_table(conn, "source_views") else None
    seq = _next_seq(conn)
    cols_cache: dict[str, set[str]] = {}

    id_map: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    skipped_erased: list[dict[str, str]] = []
    imported = {"sources": 0, "spans": 0, "claims": 0, "entities": 0, "edges": 0}
    any_replayed = False  # set when a committed operation receipt replays

    def map_id(kind: str, orig: str) -> Optional[str]:
        return id_map.get(kind, {}).get(orig)

    def remember(kind: str, orig: str, new: str) -> None:
        id_map.setdefault(kind, {})[orig] = new

    def opkey(kind: str, orig: str) -> str:
        return f"import:{fp}:{kind}:{orig}"

    def fenced(kind: str, orig: str) -> bool:
        if _fenced(store, conn, fence_scopes, kind, orig):
            skipped_erased.append({"object_kind": kind, "object_id": orig})
            warnings.append(f"{kind} {orig} skipped: erasure fence")
            return True
        return False

    def prior(kind: str, orig: str, record: dict[str, Any]) -> Optional[str]:
        """Idempotent replay: a committed operation receipt maps orig→new."""
        nonlocal any_replayed
        if ops is None:
            return None
        existing = ops.check(
            conn, sid, opkey(kind, orig), store.hmac(str(record.get("digest")).encode())
        )
        if existing is not None:
            receipt = safe_json_loads(existing.get("receipt_json") or "{}")
            if isinstance(receipt, dict) and isinstance(receipt.get("new_id"), str):
                remember(kind, orig, receipt["new_id"])
                any_replayed = True
                return receipt["new_id"]
        return None

    def commit_op(kind: str, orig: str, record: dict[str, Any], new: str) -> None:
        if ops is None:
            return
        ops.record(
            conn,
            sid,
            opkey(kind, orig),
            input_digest=store.hmac(str(record.get("digest")).encode()),
            effect_kind="import_object",
            receipt={"new_id": new, "kind": kind, "origin_id": orig},
            committed_event=seq,
        )

    def require_record(rec: Any, what: str) -> dict[str, Any]:
        if not isinstance(rec, dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"{what} record must be an object"
            )
        return rec

    # --- sources + revisions ------------------------------------------------
    for rec in bundle.get("sources") or []:
        rec = require_record(rec, "source")
        _verify_digest(rec, "source")
        orig = require_id(rec.get("source_id"), "source_id")
        if fenced("source", orig):
            continue
        got = prior("source", orig, rec)
        if got is not None:
            continue
        # Dedup on the (origin, external_id) uniqueness even when the
        # operations ledger is unavailable — replay never inserts twice.
        ext_id = f"{fp}:{sid}:{orig}"
        existing_src = conn.execute(
            "SELECT source_id FROM sources WHERE origin = 'verbatim-import'"
            " AND external_id = ?",
            (ext_id,),
        ).fetchone()
        if existing_src is not None:
            remember("source", orig, existing_src[0])
            any_replayed = True
            continue
        new_sid_obj = new_id()
        kind_v = rec.get("source_kind")
        if kind_v not in _SOURCE_KINDS:
            kind_v = "import"
        _insert_filtered(
            conn,
            cols_cache,
            "sources",
            {
                "source_id": new_sid_obj,
                "origin": "verbatim-import",
                # external_id embeds (bundle, target, origin object) so
                # per-scope dedup holds and cross-scope imports never collide
                # on the (origin, external_id) unique index.
                "external_id": f"{fp}:{sid}:{orig}",
                "source_kind": kind_v,
                "scope_id": sid,
                "speaker_id": rec.get("speaker_id"),
                "created_us": int(rec.get("created_us") or wall_us()),
            },
        )
        for rev in rec.get("revisions") or []:
            _import_revision(
                store, conn, cols_cache, views, sid, new_sid_obj, orig, rev,
                fence_scopes, skipped_erased, warnings,
            )
        remember("source", orig, new_sid_obj)
        commit_op("source", orig, rec, new_sid_obj)
        imported["sources"] += 1

    # --- spans ----------------------------------------------------------------
    for rec in bundle.get("spans") or []:
        rec = require_record(rec, "span")
        _verify_digest(rec, "span")
        orig = require_id(rec.get("span_id"), "span_id")
        if fenced("span", orig):
            continue
        got = prior("span", orig, rec)
        if got is not None:
            continue
        new_source = map_id("source", str(rec.get("source_id")))
        if new_source is None:
            warnings.append(f"span {orig} skipped: source not in import")
            continue
        new_span = _import_span(store, conn, cols_cache, rec, new_source, warnings, orig)
        if new_span is None:
            continue
        remember("span", orig, new_span)
        commit_op("span", orig, rec, new_span)
        imported["spans"] += 1

    # --- entities ---------------------------------------------------------------
    for rec in bundle.get("entities") or []:
        rec = require_record(rec, "entity")
        _verify_digest(rec, "entity")
        orig = require_id(rec.get("entity_id"), "entity_id")
        if fenced("entity", orig):
            continue
        got = prior("entity", orig, rec)
        if got is not None:
            continue
        new_eid = new_id()
        _insert_filtered(
            conn,
            cols_cache,
            "entities",
            {
                "entity_id": new_eid,
                "scope_id": sid,
                "kind": rec.get("kind"),
                "label": str(rec.get("label") or "?"),
                "created_event": seq,
                "row_version": 1,
            },
        )
        for alias in rec.get("aliases") or []:
            span_ref = alias.get("source_span_id")
            mapped_span = map_id("span", str(span_ref)) if span_ref else None
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases"
                "(entity_id, normalized_alias, source_span_id, approval_event)"
                " VALUES (?, ?, ?, NULL)",
                (new_eid, str(alias.get("normalized_alias") or ""), mapped_span),
            )
        remember("entity", orig, new_eid)
        commit_op("entity", orig, rec, new_eid)
        imported["entities"] += 1

    # --- claims ---------------------------------------------------------------
    for rec in bundle.get("claims") or []:
        rec = require_record(rec, "claim")
        _verify_digest(rec, "claim")
        orig = require_id(rec.get("claim_id"), "claim_id")
        if fenced("claim", orig):
            continue
        got = prior("claim", orig, rec)
        if got is not None:
            continue
        new_cid = _import_claim(
            store, conn, cols_cache, rec, sid, seq, map_id, warnings, orig
        )
        if new_cid is None:
            continue
        remember("claim", orig, new_cid)
        commit_op("claim", orig, rec, new_cid)
        imported["claims"] += 1

    # --- edges ------------------------------------------------------------------
    for rec in bundle.get("edges") or []:
        rec = require_record(rec, "edge")
        _verify_digest(rec, "edge")
        orig = require_id(rec.get("edge_id"), "edge_id")
        if fenced("edge", orig):
            continue
        got = prior("edge", orig, rec)
        if got is not None:
            continue
        et = rec.get("edge_type")
        try:
            EdgeType(et)
        except ValueError:
            warnings.append(f"edge {orig} skipped: unknown edge_type {et!r}")
            continue
        skind, sid_orig = str(rec.get("source_kind")), str(rec.get("source_id"))
        tkind, tid_orig = str(rec.get("target_kind")), str(rec.get("target_id"))
        new_source = map_id(skind, sid_orig)
        new_target = map_id(tkind, tid_orig)
        if new_source is None or new_target is None:
            warnings.append(f"edge {orig} skipped: endpoint not imported")
            continue
        new_eid = new_id()
        _insert_filtered(
            conn,
            cols_cache,
            "edges",
            {
                "edge_id": new_eid,
                "scope_id": sid,
                "source_kind": skind,
                "source_id": new_source,
                "target_kind": tkind,
                "target_id": new_target,
                "edge_type": et,
                "decision_id": None,  # foreign decisions never import
                "created_event": seq,
                "retired_event": seq if rec.get("retired_event") is not None else None,
            },
        )
        remember("edge", orig, new_eid)
        commit_op("edge", orig, rec, new_eid)
        imported["edges"] += 1

    bump = getattr(store, "bump_generation", None)
    if callable(bump):
        bump(conn)
    EventsRepo(store).append(
        conn,
        sid,
        "bundle_imported",
        actor,
        {
            "bundle_fp": fp,
            "origin_scope_id": origin_sid,
            "imported": dict(imported),
            "skipped_erased": len(skipped_erased),
        },
        _IMPORT_POLICY_VERSION,
    )
    return {
        "scope_id": sid,
        "imported": imported,
        "skipped_erased": skipped_erased,
        "warnings": warnings,
        "id_map": id_map,
        "duplicate": any_replayed and not any(imported.values()),
    }


def _next_seq(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute("SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events").fetchone()[0]
    )


def _import_revision(
    store: Any,
    conn: sqlite3.Connection,
    cols_cache: dict[str, set[str]],
    views: Optional[SourceViewsRepo],
    scope_id: str,
    new_source_id: str,
    orig_source_id: str,
    rev: dict[str, Any],
    fence_scopes: set[str],
    skipped_erased: list[dict[str, str]],
    warnings: list[str],
) -> None:
    rev_no = rev.get("revision")
    if not isinstance(rev_no, int) or rev_no < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "revision must be int >= 1")
    if _fenced(store, conn, fence_scopes, "source_revision", f"{orig_source_id}:{rev_no}"):
        skipped_erased.append(
            {"object_kind": "source_revision", "object_id": f"{orig_source_id}:{rev_no}"}
        )
        warnings.append(
            f"source_revision {orig_source_id}:{rev_no} skipped: erasure fence"
        )
        return
    payload_b64 = rev.get("payload_b64")
    if payload_b64 is None:
        warnings.append(
            f"revision {orig_source_id}:{rev_no} skipped: non-portable bundle"
        )
        return
    try:
        payload = base64.b64decode(payload_b64, validate=True)
    except Exception as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid payload_b64 on {orig_source_id}:{rev_no}"
        ) from exc
    if not payload:
        raise VerbatimError(ErrorCode.VALIDATION, "empty payload in bundle")
    if len(payload) > _MAX_BUNDLE_BYTES:
        raise VerbatimError(ErrorCode.EVIDENCE_TOO_LARGE, "payload exceeds bound")
    declared = rev.get("payload_sha256")
    if isinstance(declared, str) and declared != hashlib.sha256(payload).hexdigest():
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"payload sha256 mismatch on {orig_source_id}:{rev_no}",
        )
    meta = dict(rev.get("metadata") or {})
    # Explicit provenance (V2-42.13): the origin object's identity and
    # provenance live under import_provenance; the local provenance column is
    # always legacy_import — cross-profile imports never inherit foreign
    # labels verbatim (V2-42.14).
    meta["import_provenance"] = {
        "origin_source_id": orig_source_id,
        "origin_revision": rev_no,
        "origin_provenance": rev.get("provenance"),
    }
    _insert_filtered(
        conn,
        cols_cache,
        "source_revisions",
        {
            "source_id": new_source_id,
            "revision": rev_no,
            "payload": payload,
            "payload_hmac": store.hmac(payload),
            "event_us": int(rev.get("event_us") or wall_us()),
            "captured_us": int(rev.get("captured_us") or wall_us()),
            "timezone": rev.get("timezone"),
            "provenance": "legacy_import",
            "metadata_json": json_dumps(meta),
        },
    )
    if views is not None:
        views.ensure_primary(conn, new_source_id, rev_no)


def _import_span(
    store: Any,
    conn: sqlite3.Connection,
    cols_cache: dict[str, set[str]],
    rec: dict[str, Any],
    new_source_id: str,
    warnings: list[str],
    orig: str,
) -> Optional[str]:
    rev = rec.get("revision")
    start = rec.get("start_byte")
    end = rec.get("end_byte")
    if not (isinstance(rev, int) and isinstance(start, int) and isinstance(end, int)):
        warnings.append(f"span {orig} skipped: malformed offsets")
        return None
    if conn.execute(
        "SELECT 1 FROM source_revisions WHERE source_id = ? AND revision = ?",
        (new_source_id, rev),
    ).fetchone() is None:
        warnings.append(f"span {orig} skipped: revision not imported")
        return None
    new_span = new_id()
    # Span construction goes through the authority repo (V4-07.07): it
    # reads the payload on this tx conn, bounds-checks the offsets,
    # strict-decodes the excerpt, and derives excerpt_hmac — no parallel
    # slicing/HMAC implementation in the import surface.
    try:
        SpansRepo(store).insert(
            new_span,
            new_source_id,
            rev,
            start,
            end,
            str(rec.get("harvester_version") or "import-1"),
            conn=conn,
        )
    except VerbatimError as exc:
        warnings.append(f"span {orig} skipped: {exc.message}")
        return None
    # Optional projection columns ride along only when the schema has them.
    extra = {
        k: rec.get(k)
        for k in ("view_id", "operation_key")
        if k in _table_cols(conn, cols_cache, "spans")
        and rec.get(k) is not None
    }
    if extra:
        conn.execute(
            f"UPDATE spans SET {', '.join(f'{k} = ?' for k in extra)}"
            " WHERE span_id = ?",
            (*extra.values(), new_span),
        )
    return new_span


def _import_claim(
    store: Any,
    conn: sqlite3.Connection,
    cols_cache: dict[str, set[str]],
    rec: dict[str, Any],
    scope_id: str,
    seq: int,
    map_id: Any,
    warnings: list[str],
    orig: str,
) -> Optional[str]:
    revs = rec.get("revisions")
    if not isinstance(revs, list) or not revs:
        warnings.append(f"claim {orig} skipped: no revisions")
        return None
    subject = rec.get("subject_id")
    new_subject = map_id("entity", str(subject)) if subject else None
    istatus = rec.get("interpretation_status") or "structured"
    if istatus not in ("unstructured", "partial", "structured"):
        istatus = "structured"
    new_cid = new_id()
    _insert_filtered(
        conn,
        cols_cache,
        "claims",
        {
            "claim_id": new_cid,
            "scope_id": scope_id,
            "subject_id": new_subject,
            "predicate": rec.get("predicate"),
            "created_event": seq,
            "row_version": 1,
            "interpretation_status": istatus,
        },
    )
    dropped_evidence = 0
    for r in revs:
        rev_no = r.get("revision")
        state = r.get("state")
        if not isinstance(rev_no, int) or rev_no < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "claim revision must be int >= 1")
        if state not in _LIFECYCLE_STATES:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown claim state {state!r}"
            )
        rev_istatus = r.get("interpretation_status") or "structured"
        if rev_istatus not in ("unstructured", "partial", "structured"):
            rev_istatus = "structured"
        rev_subject = r.get("subject_id")
        new_rev_subject = map_id("entity", str(rev_subject)) if rev_subject else None
        _insert_filtered(
            conn,
            cols_cache,
            "claim_revisions",
            {
                "claim_id": new_cid,
                "revision": rev_no,
                "state": state,
                "object_json": json_dumps(r["object"]) if r.get("object") is not None else None,
                "polarity": r.get("polarity") if r.get("polarity") in ("affirmative", "negated") else "affirmative",
                "modality": r.get("modality") if r.get("modality") in ("asserted", "hypothetical", "habitual", "uncertain") else "asserted",
                "condition_json": json_dumps(r["condition"]) if r.get("condition") is not None else None,
                "interpretation_json": json_dumps(r["interpretation"]) if r.get("interpretation") is not None else None,
                "recorded_from": seq,
                "recorded_until": seq if r.get("recorded_until") is not None else None,
                "subject_id": new_rev_subject,
                "predicate": r.get("predicate"),
                "registry_version": r.get("registry_version"),
                "interpretation_status": rev_istatus,
                "method": r.get("method"),
            },
        )
        for iv in r.get("intervals") or []:
            _import_interval(conn, cols_cache, new_cid, rev_no, iv)
        for ev in r.get("evidence") or []:
            mapped = map_id("span", str(ev.get("span_id")))
            role = ev.get("evidence_role")
            if mapped is None:
                dropped_evidence += 1
                continue
            if role not in _EVIDENCE_ROLES:
                role = "primary"
            conn.execute(
                "INSERT INTO claim_evidence(claim_id, revision, span_id,"
                " evidence_role) VALUES (?, ?, ?, ?)",
                (new_cid, rev_no, mapped, role),
            )
    for link in rec.get("entity_links") or []:
        ent = map_id("entity", str(link.get("entity_id")))
        if ent is None:
            continue
        span_ref = link.get("span_id")
        mapped_span = map_id("span", str(span_ref)) if span_ref else None
        conn.execute(
            "INSERT OR REPLACE INTO claim_entities"
            "(claim_id, entity_id, role, span_id) VALUES (?, ?, ?, ?)",
            (new_cid, ent, str(link.get("role") or "mention"), mapped_span),
        )
    if dropped_evidence:
        warnings.append(
            f"claim {orig}: {dropped_evidence} evidence link(s) unmapped"
        )
    return new_cid


def _import_interval(
    conn: sqlite3.Connection,
    cols_cache: dict[str, set[str]],
    claim_id: str,
    rev_no: int,
    iv: dict[str, Any],
) -> None:
    interval_no = iv.get("interval_no")
    if not isinstance(interval_no, int) or interval_no < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "interval_no must be int >= 0")
    prec = iv.get("precision")
    if prec not in _PRECISIONS:
        prec = "unknown"
    skind = iv.get("start_kind")
    ekind = iv.get("end_kind")
    if skind not in _ENDPOINT_KINDS:
        skind = "exact"
    if ekind not in _ENDPOINT_KINDS:
        ekind = "exact"
    _insert_filtered(
        conn,
        cols_cache,
        "valid_intervals",
        {
            "claim_id": claim_id,
            "revision": rev_no,
            "interval_no": interval_no,
            "from_us": iv.get("from_us"),
            "until_us": iv.get("until_us"),
            "precision": prec,
            "timezone": iv.get("timezone"),
            "basis": str(iv.get("basis") or "unknown"),
            "uncertainty_json": json_dumps(iv["uncertainty"]) if iv.get("uncertainty") is not None else None,
            "start_kind": skind,
            "end_kind": ekind,
            "from_us_hi": iv.get("from_us_hi"),
            "until_us_hi": iv.get("until_us_hi"),
        },
    )


__all__ = [
    "EXPORT_FORMAT_VERSION",
    "NEVER_EXPORTED",
    "export_scope",
    "import_bundle",
]
