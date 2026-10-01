"""V7 derived-plane closure (SPEC_V7 §30, V7-06.11/V7-19.09, V7-30.02).

The V7 schema adds a second derived plane (``schema_v7.V7_TABLES`` — 21
§30 tables plus the ``unit_fts_*`` wiring).  Every byte of it is
*recomputable projection* of ``sources``/``source_revisions``: no V7
table holds a foreign key into the evidence plane, so erasure is scoped
DELETEs over the covering indexes — exactly like the v5 projection
tables.  This module is the single allowlist + sweep those deletes run
through; the existing erasure surfaces register onto it additively:

* ``verbatim/purge.py::_empty_revision`` calls ``delete_source_v7`` for
  every emptied ``(source_id, revision)`` — so span erasure, revision
  erasure, whole-source purge (``execute_purge``), the resumable
  ``ClosureEngine`` (via ``deletion._delete_object`` →
  ``purge._erase_*``), and ``handle_purge_derived`` all sweep the unit
  plane in the same transaction that empties the bytes.
* ``verbatim/memory/controls.py::_sweep_derived`` re-runs the idempotent
  whole-source sweep on the forget path, closing the drain↔sweep commit
  window.
* ``verbatim/storage/store.py::_COUNT_TABLES`` names every V7 table so
  ``check_integrity`` reports live counts (``None`` on pre-V7 stores).
* ``verbatim/export.py::NEVER_EXPORTED`` documents the V7 set as
  recomputable derived caches — bundles ship evidence sections only.

Table disposition (mirrors ``schema_v7``'s own comments):

* ``V7_CLOSURE_TABLES`` — content-bearing derived rows deleted by the
  sweeps below.
* ``V8_CLOSURE_TABLES`` — the SPEC_V8 §19 additions
  (``unit_time_mentions``, ``unit_doclen``): same unit-keyed derived
  shape, swept identically (V8-19.03).  ``events_v7.subject_source``
  needs no handling of its own — it rides the event row, which already
  dies with its unit.
* ``V7_FTS_INDEXES`` — the three §30 FTS5 virtual tables.  They are never
  written to directly: deleting ``unit_fts_content`` rows fires the
  ``*_ad`` mirror triggers that remove index terms (the v5
  ``source_fts`` convention — content first, carrier second, so the
  ``REFERENCES unit_fts_rows(row_id)`` pair never dangles under
  ``PRAGMA foreign_key_check``).
* ``V7_AUDIT_TABLES`` — ``screening_log`` is the write-channel screening
  journal: its rows survive evidence closure exactly like the v1
  ``events`` journal (§30 note on the table).  ``run_manifests`` is
  scope-less eval provenance.  Both are counted under
  ``audit_retained`` — reported, never claimed as erased.

Ordering discipline inside a unit sweep: mentions are read first (the
``entity_canon``/``entity_aliases_v7`` fixup needs them), lexical stats
are decremented while the field text still exists, then content rows,
then carriers, then the unit rows themselves — a crashed sweep resumed
by a second call still converges because every statement is idempotent.

Generation fencing (V7-30.02): erasure removes the unit's rows at EVERY
generation — the source bytes are gone, so no generation of their
projection may outlive them.  Rebuild side writes a new generation
beside the live one; ``sweep_stale_generations_v7`` is the post-flip
sweeper and refuses to touch rows at or above the committed
``meta.projection_generation`` fence unless the caller pins a higher
``keep_generation`` itself.

Quarantine seam (V7-19.09 / V3-14.10 cascade): ``held_unit_ids`` maps
live holds — ``quarantine`` rows in excluding states plus active purge
tombstones — onto unit ids through the covering chain
unit → source → source_revision → span → source_envelope.  It is the
query layer the eligibility adapter
(``verbatim/retrieval/v7/eligibility.py::make_eligible``, imported by
``memory/facade.py``) should call:

    held = held_unit_ids(conn, scope_id)
    eligible = lambda row: row["unit_id"] not in held

so a hold on a source, its envelope, or a covering source revision
withholds every dependent unit from every V7 lane.

This module never opens its own transaction and never commits — every
helper runs on the caller's ``conn`` inside the caller's write tx.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from typing import Any, Iterable, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError, safe_json_loads
from ..storage.repos import has_table
from ..storage.stats_v7 import decrement_stats

try:  # the frozen §30 table list — drift-checked, not imported blindly
    from ..storage import schema_v7 as _schema_v7
except Exception:  # pragma: no cover - partial tree
    _schema_v7 = None


#: Content-bearing V7 derived tables the sweeps own (fixed allowlist —
#: identifiers here are literals, never caller text).
V7_CLOSURE_TABLES: tuple[str, ...] = (
    "units",
    "unit_fts_rows",
    "unit_fts_content",
    "lex_stats",
    "lex_df",
    "unit_vectors_block",
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
)

#: Content-bearing V8 derived tables the sweeps own (SPEC_V8 §19,
#: V8-19.03 — same unit-keyed derived shape; ``events_v7.subject_source``
#: rides the event row, which is already a closure member).
V8_CLOSURE_TABLES: tuple[str, ...] = (
    "unit_time_mentions",
    "unit_doclen",
)

#: The §30 FTS5 MATCH targets — maintained only through the
#: ``unit_fts_content`` delete triggers (never a direct DELETE on a
#: virtual table or its ``*_data``/``*_idx`` shadows).
V7_FTS_INDEXES: tuple[str, ...] = ("unit_fts", "unit_fts_stem", "unit_fts_tri")

#: Audit/provenance V7 tables: counted by every sweep under
#: ``audit_retained``, never deleted (v1 ``events``-journal rule).
V7_AUDIT_TABLES: tuple[str, ...] = ("screening_log", "run_manifests")

#: Flat unit-keyed tables: ``DELETE … WHERE unit_id IN (…)``.  The V8
#: §19 tables ride the same keyed shape (V8-19.03).
_UNIT_KEYED_TABLES: tuple[str, ...] = (
    "entity_mentions",
    "events_v7",
    "state_facts",
    "preferences",
    "standing_rules",
    "unit_time_mentions",
    "unit_doclen",
)

#: Scope+generation keyed tables — the scope sweep and the stale-
#: generation sweep delete ``WHERE scope_id = ?`` / ``generation < ?``.
_SCOPE_GENERATION_TABLES: tuple[str, ...] = (
    "units",
    "unit_fts_rows",
    "lex_stats",
    "lex_df",
    "unit_vectors_block",
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
    "unit_time_mentions",
    "unit_doclen",
)

#: Quarantine object kinds that name a unit directly.
_UNIT_HOLD_KINDS = frozenset({"unit", "v7_unit", "units"})
#: Kinds whose object_id resolves through ``source_envelopes``.
_ENVELOPE_HOLD_KINDS = frozenset({"envelope", "source_envelope"})

#: Quarantine states that hide content (mirrors security.EXCLUDING_STATES;
#: re-declared so this module stays importable without the v3 security
#: stack — the same convention ``retrieval/v3/union.py`` uses).
_HELD_STATES = frozenset({"pending", "suppressed"})
#: Purge states whose targets are tombstoned for reads.
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

_IN_CHUNK = 400

_DROP = object()  # sentinel for _strip_refs


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _chunks(seq: list, n: int = _IN_CHUNK) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _ph(n: int) -> str:
    return ",".join("?" for _ in range(n))


def _new_result(source_id: Optional[str] = None, scope_id: Optional[str] = None) -> dict:
    return {
        "units_removed": 0,
        "deleted": {},
        "audit_retained": {},
        "unhandled_v7": [],
        "scopes": [],
        "source_id": source_id,
        "scope_id": scope_id,
    }


def _unhandled_v7(conn: sqlite3.Connection) -> list[str]:
    """Present V7/V8 tables outside the covered set — reported, never
    claimed.

    If ``schema_v7`` grows a table this list names it in every sweep
    receipt instead of silently skipping it (the purge ``unhandled``
    convention: coverage is only ever claimed for the mapped set).
    """
    if _schema_v7 is None:
        return []
    covered = (
        set(V7_CLOSURE_TABLES) | set(V7_FTS_INDEXES)
        | set(V7_AUDIT_TABLES) | set(V8_CLOSURE_TABLES)
    )
    universe = (
        _schema_v7.V7_TABLES + _schema_v7.V7_INTERNAL_TABLES
        + getattr(_schema_v7, "V8_TABLES", ())
    )
    present = {t for t in universe if has_table(conn, t)}
    return sorted(present - covered)


def _fields_for_stats(
    text: Any, speaker: Any, entities: Any, session: Any, when: Any
) -> dict:
    """Rebuild a stats row's field terms from stored FTS content —
    analyzer terms for ``text`` (write-time parity with the projection
    writer), the stored token strings elsewhere.  Mirrors
    ``jobs/units_jobs._fields_for_stats`` so a decrement subtracts
    exactly what the add counted.
    """
    try:
        from ..text.norm_v2 import analyze

        terms = [t.term for t in analyze(text or "").terms if t.channel == "text"]
    except Exception:
        terms = (text or "").split()
    return {
        "text": terms,
        "speaker": speaker,
        "entities": entities,
        "session": session,
        "when": when,
    }


def _refs_in_json(value: Any) -> set[str]:
    """Every unit id a refs JSON value names — bare strings and
    ``{"unit_id": …}`` leaves at any nesting depth (the acceptance
    envelope ``retrieval/v7/obs.py::_parse_refs`` defines)."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.add(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k == "unit_id" and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(safe_json_loads(value))
    return out


def _strip_refs_json(value: Any, removed: set[str]) -> Optional[str]:
    """``removed`` ids dropped from a refs JSON, structure preserved.

    Returns ``None`` when the value never referenced a removed unit —
    callers can skip the UPDATE entirely.
    """
    data = safe_json_loads(value)
    hit = False

    def walk(node: Any) -> Any:
        nonlocal hit
        if isinstance(node, str):
            if node in removed:
                hit = True
                return _DROP
            return node
        if isinstance(node, list):
            kept = [walk(item) for item in node]
            return [item for item in kept if item is not _DROP]
        if isinstance(node, dict):
            uid = node.get("unit_id")
            if isinstance(uid, str) and uid in removed:
                hit = True
                return _DROP
            out = {}
            for k, v in node.items():
                nv = walk(v)
                if nv is _DROP:
                    continue
                out[k] = nv
            return out
        return node

    cleaned = walk(data)
    if not hit:
        return None
    return json.dumps(None if cleaned is _DROP else cleaned)


# ---------------------------------------------------------------------------
# vector-block surgery (unit_vectors_block — embeddings.matrix layout)
# ---------------------------------------------------------------------------
#
# ``rowmap_blob`` = u32 count + per-key (u32 len, utf8 bytes); ``data_blob``
# = n_rows × dims × width (f32: 4, int8: 1); ``scale_blob`` = n_rows ×
# 8 bytes for f32 (one f64 norm) or n_rows × 16 for int8 (scale + norm).
# Slices are byte-exact — surviving rows keep their stored bytes, nothing
# is re-quantized.

_SCALE_WIDTHS = {"f32": 8, "int8": 16}
_DATA_WIDTHS = {"f32": 4, "int8": 1}


def _rowmap_keys(blob: bytes, expected: int) -> Optional[list[str]]:
    """Decode a rowmap blob; ``None`` when it doesn't parse or the count
    disagrees with the declared row count (corrupt → caller fails closed)."""
    try:
        pos = 0
        (n,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        keys: list[str] = []
        for _ in range(n):
            (klen,) = struct.unpack_from("<I", blob, pos)
            pos += 4
            keys.append(blob[pos : pos + klen].decode("utf-8"))
            pos += klen
    except (struct.error, UnicodeDecodeError, IndexError):
        return None
    if n != expected:
        return None
    return keys


def _pack_rowmap(keys: list[str]) -> bytes:
    out = bytearray(struct.pack("<I", len(keys)))
    for key in keys:
        kb = key.encode("utf-8")
        out += struct.pack("<I", len(kb))
        out += kb
    return bytes(out)


def _sweep_vector_blocks(
    conn: sqlite3.Connection,
    buckets: dict[tuple[str, int], list[str]],
    out: dict,
) -> None:
    """Remove deleted unit keys from vector blocks, byte-exact.

    Every block in an affected *scope* is scanned across ALL
    generations — the block's ``generation`` column is the writer's
    snapshot tag, and a rowmap can legitimately carry unit keys minted
    under an earlier units generation, so filtering blocks by the
    units' own generation would leak. A block whose rowmap intersects
    the removed set is rewritten minus those rows; a block that decodes
    empty is dropped; a block whose rowmap or blobs cannot be decoded is
    dropped outright — the block is recomputable, and an opaque blob
    cannot prove it does not carry an erased unit's vector (fail
    closed, never leak).
    """
    if not has_table(conn, "unit_vectors_block"):
        return
    by_scope: dict[str, set[str]] = {}
    for (scope_id, _gen), ids in buckets.items():
        by_scope.setdefault(scope_id, set()).update(ids)
    for scope_id in sorted(by_scope):
        removed = by_scope[scope_id]
        for row in conn.execute(
            "SELECT encoder_id, generation, block_no, n_rows, dims, quant,"
            " scale_blob, data_blob, rowmap_blob"
            " FROM unit_vectors_block"
            " WHERE scope_id = ?",
            (scope_id,),
        ).fetchall():
            encoder_id, generation, block_no, n_rows, dims, quant, scale, data, rowmap = row
            keys = _rowmap_keys(bytes(rowmap), int(n_rows))
            if keys is None:
                conn.execute(
                    "DELETE FROM unit_vectors_block"
                    " WHERE encoder_id = ? AND scope_id = ?"
                    "   AND generation = ? AND block_no = ?",
                    (encoder_id, scope_id, generation, block_no),
                )
                out["deleted"]["unit_vectors_block"] = (
                    out["deleted"].get("unit_vectors_block", 0) + 1
                )
                continue
            keep = [i for i, k in enumerate(keys) if k not in removed]
            if len(keep) == len(keys):
                continue  # no overlap — block untouched
            if not keep:
                conn.execute(
                    "DELETE FROM unit_vectors_block"
                    " WHERE encoder_id = ? AND scope_id = ?"
                    "   AND generation = ? AND block_no = ?",
                    (encoder_id, scope_id, generation, block_no),
                )
                out["deleted"]["unit_vectors_block"] = (
                    out["deleted"].get("unit_vectors_block", 0) + 1
                )
                continue
            dw = _DATA_WIDTHS.get(quant)
            sw = _SCALE_WIDTHS.get(quant)
            data_b, scale_b = bytes(data), bytes(scale or b"")
            if (
                dw is None
                or sw is None
                or len(data_b) != int(n_rows) * int(dims) * dw
                or len(scale_b) != int(n_rows) * sw
            ):
                # Unknown quant or torn blobs — cannot prove the kept
                # rows; drop the recomputable block (fail closed).
                conn.execute(
                    "DELETE FROM unit_vectors_block"
                    " WHERE encoder_id = ? AND scope_id = ?"
                    "   AND generation = ? AND block_no = ?",
                    (encoder_id, scope_id, generation, block_no),
                )
                out["deleted"]["unit_vectors_block"] = (
                    out["deleted"].get("unit_vectors_block", 0) + 1
                )
                continue
            row_bytes = int(dims) * dw
            new_data = b"".join(
                data_b[i * row_bytes : (i + 1) * row_bytes] for i in keep
            )
            new_scale = b"".join(
                scale_b[i * sw : (i + 1) * sw] for i in keep
            )
            conn.execute(
                "UPDATE unit_vectors_block SET n_rows = ?, scale_blob = ?,"
                " data_blob = ?, rowmap_blob = ?"
                " WHERE encoder_id = ? AND scope_id = ?"
                "   AND generation = ? AND block_no = ?",
                (
                    len(keep),
                    new_scale,
                    new_data,
                    _pack_rowmap([keys[i] for i in keep]),
                    encoder_id,
                    scope_id,
                    generation,
                    block_no,
                ),
            )
            out["deleted"]["unit_vectors_rewritten"] = (
                out["deleted"].get("unit_vectors_rewritten", 0) + 1
            )


def _sweep_vector_oracle(
    conn: sqlite3.Connection, unit_ids: Iterable[str], out: dict
) -> None:
    """Remove the swept units' per-row f32 oracle vectors.

    The provisional ``unit_vectors`` oracle (``embeddings.matrix``'s
    V7-07.04 durable per-row BLOB home) is keyed ``(unit_key,
    encoder_id)`` — no scope/generation columns. ``unit_id`` is
    content-addressed over ``(source_id, revision, generation, …)``, so
    a swept id belongs to exactly the erased units; deleting by key
    removes its rows under every encoder space they were written into
    and leaves coexisting units' oracle rows untouched.
    """
    ids = sorted(set(unit_ids))
    if not ids or not has_table(conn, "unit_vectors"):
        return
    n = 0
    for chunk in _chunks(ids):
        cur = conn.execute(
            f"DELETE FROM unit_vectors WHERE unit_key IN ({_ph(len(chunk))})",
            chunk,
        )
        n += cur.rowcount
    if n:
        out["deleted"]["unit_vectors"] = (
            out["deleted"].get("unit_vectors", 0) + n
        )


# ---------------------------------------------------------------------------
# the unit sweep — every V7 artifact standing on a removed unit set
# ---------------------------------------------------------------------------


def _sweep_units(
    conn: sqlite3.Connection, unit_rows: list[dict], out: dict
) -> None:
    """Delete all V7 rows derived from ``unit_rows`` (dicts carrying
    ``unit_id``/``scope_id``/``generation``).  Idempotent; count-reported."""
    if not unit_rows:
        return
    unit_ids = sorted({u["unit_id"] for u in unit_rows})
    buckets: dict[tuple[str, int], list[str]] = {}
    for u in unit_rows:
        buckets.setdefault((u["scope_id"], int(u["generation"])), []).append(
            u["unit_id"]
        )
    scopes = sorted({s for s, _g in buckets})
    out["scopes"] = scopes
    out["units_removed"] = len(unit_ids)

    # -- entity mentions: collect the affected canon keys BEFORE deleting
    #    (the canon/alias fixup needs them), and per-key mention counts.
    affected_canons: dict[tuple[str, str, int], int] = {}
    if has_table(conn, "entity_mentions"):
        for chunk in _chunks(unit_ids):
            for canon, scope_id, gen in conn.execute(
                "SELECT canon, scope_id, generation FROM entity_mentions"
                f" WHERE unit_id IN ({_ph(len(chunk))})",
                chunk,
            ).fetchall():
                key = (scope_id, canon, int(gen))
                affected_canons[key] = affected_canons.get(key, 0) + 1
        n = 0
        for chunk in _chunks(unit_ids):
            cur = conn.execute(
                f"DELETE FROM entity_mentions WHERE unit_id IN ({_ph(len(chunk))})",
                chunk,
            )
            n += cur.rowcount
        out["deleted"]["entity_mentions"] = n

    # -- lexical stats + FTS pair --------------------------------------
    # Decrement while the field text still exists (V7-06.06 contract:
    # the same-tx decrement mirrors the add); then content rows (the
    # *_ad triggers mirror into unit_fts/_stem/_tri), then carriers.
    fts_rows: list[tuple] = []
    if has_table(conn, "unit_fts_rows"):
        for chunk in _chunks(unit_ids):
            fts_rows.extend(
                conn.execute(
                    "SELECT row_id, unit_id, scope_id, generation"
                    " FROM unit_fts_rows"
                    f" WHERE unit_id IN ({_ph(len(chunk))})",
                    chunk,
                ).fetchall()
            )
    if fts_rows:
        row_ids = [int(r[0]) for r in fts_rows]
        content_by_uid: dict[str, tuple] = {}
        if has_table(conn, "unit_fts_content"):
            for chunk in _chunks(row_ids):
                for uid, text, sp, ent, sess, whn in conn.execute(
                    "SELECT r.unit_id, c.text, c.speaker, c.entities,"
                    " c.session, c.\"when\""
                    " FROM unit_fts_content c"
                    " JOIN unit_fts_rows r ON r.row_id = c.fts_row_id"
                    f" WHERE c.fts_row_id IN ({_ph(len(chunk))})",
                    chunk,
                ).fetchall():
                    content_by_uid[uid] = (text, sp, ent, sess, whn)
        # decrement per (scope, generation, stats_version) — only
        # versions with live stats rows get a decrement (a version that
        # never counted the scope has nothing to give back).
        if has_table(conn, "lex_stats"):
            for (scope_id, generation), ids in sorted(buckets.items()):
                versions = [
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT stats_version FROM lex_stats"
                        " WHERE scope_id = ? AND generation = ?",
                        (scope_id, generation),
                    ).fetchall()
                ]
                dec_rows = [
                    {
                        "unit_id": uid,
                        "fields": _fields_for_stats(*content_by_uid[uid]),
                    }
                    for uid in ids
                    if uid in content_by_uid
                ]
                if not dec_rows:
                    continue
                for version in versions:
                    decrement_stats(
                        conn,
                        scope_id,
                        generation,
                        dec_rows,
                        stats_version=version,
                    )
                out["deleted"]["lex_stats"] = (
                    out["deleted"].get("lex_stats", 0) + len(dec_rows)
                )
        if has_table(conn, "unit_fts_content"):
            n = 0
            for chunk in _chunks(row_ids):
                cur = conn.execute(
                    "DELETE FROM unit_fts_content"
                    f" WHERE fts_row_id IN ({_ph(len(chunk))})",
                    chunk,
                )
                n += cur.rowcount
            out["deleted"]["unit_fts_content"] = n
        n = 0
        for chunk in _chunks(row_ids):
            cur = conn.execute(
                f"DELETE FROM unit_fts_rows WHERE row_id IN ({_ph(len(chunk))})",
                chunk,
            )
            n += cur.rowcount
        out["deleted"]["unit_fts_rows"] = n

    # -- flat unit-keyed tables -----------------------------------------
    for table in _UNIT_KEYED_TABLES:
        if not has_table(conn, table):
            continue
        n = 0
        for chunk in _chunks(unit_ids):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE unit_id IN ({_ph(len(chunk))})",
                chunk,
            )
            n += cur.rowcount
        out["deleted"][table] = n

    # -- graph edges, both directions (V7-08.11 incident-edge closure) --
    if has_table(conn, "graph_edges"):
        n = 0
        for chunk in _chunks(unit_ids):
            cur = conn.execute(
                "DELETE FROM graph_edges"
                f" WHERE src_unit IN ({_ph(len(chunk))})"
                f"    OR dst_unit IN ({_ph(len(chunk))})",
                [*chunk, *chunk],
            )
            n += cur.rowcount
        out["deleted"]["graph_edges"] = n

    # -- unit-pinning artifacts (JSON ref columns) ----------------------
    # t2_facts: the fact stands on its pinned units AND carries verbatim
    # quotes — the row goes with its support (V7-13.13 erasure form).
    if has_table(conn, "t2_facts"):
        n = 0
        for scope_id in scopes:
            rows = conn.execute(
                "SELECT fact_id, unit_ids_json FROM t2_facts"
                " WHERE scope_id = ?",
                (scope_id,),
            ).fetchall()
            doomed = [
                r[0]
                for r in rows
                if _refs_in_json(r[1]) & set(unit_ids)
            ]
            for chunk in _chunks(doomed):
                cur = conn.execute(
                    f"DELETE FROM t2_facts WHERE fact_id IN ({_ph(len(chunk))})",
                    chunk,
                )
                n += cur.rowcount
        out["deleted"]["t2_facts"] = n

    # observations_v7: a support ref going away retires the row
    # (V7-14.08); a *contradict* ref going away strips the dead pin but
    # keeps the observation — its support still stands.
    if has_table(conn, "observations_v7"):
        n = 0
        removed = set(unit_ids)
        for scope_id in scopes:
            rows = conn.execute(
                "SELECT obs_id, support_refs_json, contradict_refs_json"
                " FROM observations_v7 WHERE scope_id = ?",
                (scope_id,),
            ).fetchall()
            for obs_id, support, contradict in rows:
                if _refs_in_json(support) & removed:
                    conn.execute(
                        "DELETE FROM observations_v7 WHERE obs_id = ?",
                        (obs_id,),
                    )
                    n += 1
                    continue
                stripped = _strip_refs_json(contradict, removed)
                if stripped is not None:
                    conn.execute(
                        "UPDATE observations_v7 SET contradict_refs_json = ?"
                        " WHERE obs_id = ?",
                        (stripped, obs_id),
                    )
        out["deleted"]["observations_v7"] = n

    # profiles_v7: a slot stands on its support refs — retire on loss.
    if has_table(conn, "profiles_v7"):
        n = 0
        removed = set(unit_ids)
        for scope_id in scopes:
            rows = conn.execute(
                "SELECT subject_canon, slot, generation, support_refs_json"
                " FROM profiles_v7 WHERE scope_id = ?",
                (scope_id,),
            ).fetchall()
            for subject, slot, gen, refs in rows:
                if _refs_in_json(refs) & removed:
                    conn.execute(
                        "DELETE FROM profiles_v7 WHERE scope_id = ?"
                        " AND subject_canon = ? AND slot = ?"
                        " AND generation = ?",
                        (scope_id, subject, slot, gen),
                    )
                    n += 1
        out["deleted"]["profiles_v7"] = n

    # standing_queries: the stored pack may embed the erased content
    # verbatim (opaque blob — cannot prove otherwise), so packs in every
    # affected scope are scrubbed and marked dirty for rebuild
    # (V7-14.07).  The query row itself is user state and survives.
    if has_table(conn, "standing_queries"):
        cur = conn.execute(
            "UPDATE standing_queries SET dirty = 1, pack_blob = NULL,"
            " pack_digest = NULL"
            f" WHERE scope_id IN ({_ph(len(scopes))})",
            scopes,
        )
        out["deleted"]["standing_queries"] = cur.rowcount

    # -- dense matrix blocks + the per-row f32 oracle --------------------
    _sweep_vector_blocks(conn, buckets, out)
    _sweep_vector_oracle(conn, unit_ids, out)

    # -- entity vocabulary fixup ----------------------------------------
    # df_units is recomputed (not decremented) so mixed-source scopes
    # stay exact; a canon with no surviving mentions goes, and its
    # rule-derived aliases go with it.  Caller/review/rejected alias
    # state is never resurrected or deleted (projection-writer rule).
    if has_table(conn, "entity_canon") and affected_canons:
        canon_dropped = 0
        by_scope_gen: dict[tuple[str, int], list[str]] = {}
        for (scope_id, canon, gen), _n in affected_canons.items():
            by_scope_gen.setdefault((scope_id, gen), []).append(canon)
        for (scope_id, gen), canons in by_scope_gen.items():
            for chunk in _chunks(sorted(set(canons))):
                conn.execute(
                    "UPDATE entity_canon SET df_units = ("
                    "  SELECT COUNT(DISTINCT unit_id) FROM entity_mentions"
                    "  WHERE scope_id = ? AND canon = entity_canon.canon"
                    "    AND generation = entity_canon.generation)"
                    f" WHERE scope_id = ? AND generation = ?"
                    f"   AND canon IN ({_ph(len(chunk))})",
                    (scope_id, scope_id, gen, *chunk),
                )
                cur = conn.execute(
                    "DELETE FROM entity_canon WHERE scope_id = ?"
                    " AND generation = ?"
                    f" AND canon IN ({_ph(len(chunk))}) AND df_units <= 0",
                    (scope_id, gen, *chunk),
                )
                canon_dropped += cur.rowcount
                if has_table(conn, "entity_aliases_v7"):
                    cur = conn.execute(
                        "DELETE FROM entity_aliases_v7 WHERE scope_id = ?"
                        " AND generation = ? AND method = 'rule'"
                        f" AND canon IN ({_ph(len(chunk))})"
                        " AND canon NOT IN (SELECT canon FROM entity_canon"
                        "                  WHERE scope_id = ?"
                        "                    AND generation = ?)",
                        (scope_id, gen, *chunk, scope_id, gen),
                    )
                    out["deleted"]["entity_aliases_v7"] = (
                        out["deleted"].get("entity_aliases_v7", 0)
                        + cur.rowcount
                    )
        out["deleted"]["entity_canon"] = canon_dropped

    # -- the unit rows themselves, last among derived rows --------------
    n = 0
    for chunk in _chunks(unit_ids):
        cur = conn.execute(
            f"DELETE FROM units WHERE unit_id IN ({_ph(len(chunk))})",
            chunk,
        )
        n += cur.rowcount
    out["deleted"]["units"] = n


def _audit_count(
    conn: sqlite3.Connection,
    *,
    unit_ids: Optional[list[str]] = None,
    source_id: Optional[str] = None,
) -> int:
    """``screening_log`` rows pinned to the swept objects — counted, never
    deleted (the write-channel journal survives evidence closure like the
    v1 ``events`` journal)."""
    if not has_table(conn, "screening_log"):
        return 0
    n = 0
    if source_id is not None:
        n += conn.execute(
            "SELECT COUNT(*) FROM screening_log WHERE source_id = ?",
            (source_id,),
        ).fetchone()[0]
    if unit_ids:
        for chunk in _chunks(sorted(unit_ids)):
            n += conn.execute(
                "SELECT COUNT(*) FROM screening_log"
                f" WHERE unit_id IN ({_ph(len(chunk))})"
                "   AND (source_id IS NULL OR source_id <> ?)",
                [*chunk, source_id or ""],
            ).fetchone()[0]
    return n


# ---------------------------------------------------------------------------
# public erasure surface
# ---------------------------------------------------------------------------


def delete_source_v7(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    revision: Optional[int] = None,
    generation: Optional[int] = None,
) -> dict:
    """Erase every V7 artifact derived from ``source_id``.

    ``revision`` narrows to one source revision (span/revision-scoped
    erasure).  ``generation`` bounds the sweep to ``generation <=`` the
    given value — the re-derivation form; ``None`` removes the source's
    units at EVERY generation (erasure: the bytes are gone, so no
    generation of their projection may outlive them — V7-30.02's
    coexistence rule governs rebuild visibility, not erasure).

    Returns per-table counts under ``deleted`` plus ``audit_retained``
    and ``unhandled_v7`` (present-but-uncovered V7 tables — reported,
    never claimed).  No-op on pre-V7 stores.
    """
    out = _new_result(source_id=source_id)
    if not isinstance(source_id, str) or not source_id:
        raise VerbatimError(ErrorCode.VALIDATION, "source_id required")
    if not has_table(conn, "units"):
        out["unhandled_v7"] = _unhandled_v7(conn)
        return out
    where = ["source_id = ?"]
    params: list[Any] = [source_id]
    if revision is not None:
        where.append("revision = ?")
        params.append(int(revision))
    if generation is not None:
        where.append("generation <= ?")
        params.append(int(generation))
    unit_rows = [
        {"unit_id": r[0], "scope_id": r[1], "generation": r[2]}
        for r in conn.execute(
            "SELECT unit_id, scope_id, generation FROM units"
            f" WHERE {' AND '.join(where)}",
            params,
        ).fetchall()
    ]
    _sweep_units(conn, unit_rows, out)
    audit = _audit_count(
        conn,
        source_id=source_id,
        unit_ids=[u["unit_id"] for u in unit_rows],
    )
    if audit:
        out["audit_retained"]["screening_log"] = audit
    out["unhandled_v7"] = _unhandled_v7(conn)
    return out


def delete_scope_v7(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    generation: Optional[int] = None,
) -> dict:
    """Erase the scope's whole V7 derived plane (or one generation of it).

    Scope deletion removes every scope-keyed V7 row — stats rows included
    (no per-unit decrement: the corpus partition itself is going away).
    ``generation`` narrows to exactly that generation's plane; ``None``
    sweeps every generation.  Audit journals are counted, not deleted.
    """
    out = _new_result(scope_id=scope_id)
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION, "scope_id required")
    if not has_table(conn, "units"):
        out["unhandled_v7"] = _unhandled_v7(conn)
        return out
    gen_clause = ""
    params: list[Any] = [scope_id]
    if generation is not None:
        gen_clause = " AND generation = ?"
        params.append(int(generation))
    # Content before carrier — the AFTER-DELETE triggers clean the FTS
    # indexes and the REFERENCES pair never dangles under fk_check.
    if has_table(conn, "unit_fts_content") and has_table(conn, "unit_fts_rows"):
        cur = conn.execute(
            "DELETE FROM unit_fts_content WHERE fts_row_id IN"
            " (SELECT row_id FROM unit_fts_rows WHERE scope_id = ?"
            f"{gen_clause})",
            params,
        )
        out["deleted"]["unit_fts_content"] = cur.rowcount
    # The per-row vector oracle (embeddings.matrix's provisional
    # ``unit_vectors``) is keyed by ``unit_key`` alone — capture the
    # doomed unit ids while the ``units`` rows still exist (the
    # scope-table loop below deletes them). A unit id is unique to its
    # (source, revision, generation), so this removes exactly the
    # swept plane's vectors under every encoder.
    doomed: list[str] = [
        r[0]
        for r in conn.execute(
            "SELECT unit_id FROM units WHERE scope_id = ?"
            f"{gen_clause}",
            params,
        ).fetchall()
    ]
    if has_table(conn, "unit_vectors") and doomed:
        n = 0
        for chunk in _chunks(sorted(set(doomed))):
            cur = conn.execute(
                "DELETE FROM unit_vectors WHERE unit_key IN"
                f" ({_ph(len(chunk))})",
                chunk,
            )
            n += cur.rowcount
        out["deleted"]["unit_vectors"] = n
    for table in _SCOPE_GENERATION_TABLES:
        if not has_table(conn, table):
            continue
        cur = conn.execute(
            f"DELETE FROM {table} WHERE scope_id = ?{gen_clause}",
            params,
        )
        out["deleted"][table] = cur.rowcount
    # A generation-narrowed sweep must also scrub surviving-generation
    # blocks: the producer stamps a block with its commit generation,
    # which can be newer than the units its rowmap carries. (A whole-scope
    # delete already removed every block in the scope — nothing survives
    # to rewrite.)
    if doomed and generation is not None:
        _sweep_vector_blocks(conn, {(scope_id, 0): doomed}, out)
    if has_table(conn, "screening_log"):
        out["audit_retained"]["screening_log"] = conn.execute(
            "SELECT COUNT(*) FROM screening_log WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()[0]
    out["unhandled_v7"] = _unhandled_v7(conn)
    return out


def _committed_generation(conn: sqlite3.Connection) -> int:
    """``meta.projection_generation`` read inside the caller's tx."""
    if not has_table(conn, "meta"):
        return 0
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = 'projection_generation'"
    ).fetchone()
    value = safe_json_loads(row[0]) if row else None
    return int(value) if isinstance(value, int) else 0


def sweep_stale_generations_v7(
    conn: sqlite3.Connection,
    *,
    scope_id: Optional[str] = None,
    keep_generation: Optional[int] = None,
) -> dict:
    """Post-flip sweeper: delete V7 rows at ``generation < keep``.

    ``keep_generation`` defaults to the committed
    ``meta.projection_generation`` — the live fence read inside the
    caller's transaction.  Rows AT the keep generation are never
    touched, so calling this while generation N is still live removes
    only generations strictly below N (i.e. nothing, when N is the
    generation the rows were written at); after the fence flips to N+1
    the same call reclaims the retired plane.  Rebuild callers that bump
    the fence in the same transaction see the new value here directly.

    Audit journals (``screening_log``/``run_manifests`` — generation
    defaults 0 by design) are excluded: they are not generation planes.
    """
    keep = (
        int(keep_generation)
        if keep_generation is not None
        else _committed_generation(conn)
    )
    out: dict[str, Any] = {
        "keep_generation": keep,
        "scope_id": scope_id,
        "deleted": {},
        "audit_retained": {},
        "unhandled_v7": [],
    }
    if not has_table(conn, "units"):
        out["unhandled_v7"] = _unhandled_v7(conn)
        return out
    params: list[Any] = [keep]
    scope_clause = ""
    if scope_id is not None:
        scope_clause = " AND scope_id = ?"
        params.append(scope_id)
    if has_table(conn, "unit_fts_content") and has_table(conn, "unit_fts_rows"):
        cur = conn.execute(
            "DELETE FROM unit_fts_content WHERE fts_row_id IN"
            " (SELECT row_id FROM unit_fts_rows WHERE generation < ?"
            f"{scope_clause})",
            params,
        )
        out["deleted"]["unit_fts_content"] = cur.rowcount
    # Per-row oracle rows carry no generation column — delete via the
    # doomed unit ids before the table loop removes ``units``. The same
    # doomed-id set also drives block surgery below: a block is stamped
    # with its commit generation, which can be NEWER than the units its
    # rowmap carries, so the flat ``generation < keep`` delete does not
    # reach every vector of a swept unit.
    doomed_units: dict[tuple[str, int], list[str]] = {}
    for uid, sc in conn.execute(
        "SELECT unit_id, scope_id FROM units WHERE generation < ?"
        f"{scope_clause}",
        params,
    ).fetchall():
        doomed_units.setdefault((sc, 0), []).append(uid)
    if has_table(conn, "unit_vectors") and doomed_units:
        n = 0
        for ids in doomed_units.values():
            for chunk in _chunks(sorted(set(ids))):
                cur = conn.execute(
                    "DELETE FROM unit_vectors WHERE unit_key IN"
                    f" ({_ph(len(chunk))})",
                    chunk,
                )
                n += cur.rowcount
        out["deleted"]["unit_vectors"] = n
    for table in _SCOPE_GENERATION_TABLES:
        if not has_table(conn, table):
            continue
        cur = conn.execute(
            f"DELETE FROM {table} WHERE generation < ?{scope_clause}",
            params,
        )
        out["deleted"][table] = cur.rowcount
    # Surviving-generation blocks can still carry swept unit ids —
    # byte-exact rowmap surgery strips them (the flat delete above only
    # reached blocks stamped inside the swept window).
    if doomed_units:
        _sweep_vector_blocks(conn, doomed_units, out)
    if has_table(conn, "screening_log"):
        q = "SELECT COUNT(*) FROM screening_log WHERE generation < ?"
        if scope_id is not None:
            q += " AND scope_id = ?"
        out["audit_retained"]["screening_log"] = conn.execute(
            q, params
        ).fetchone()[0]
    out["unhandled_v7"] = _unhandled_v7(conn)
    return out


# ---------------------------------------------------------------------------
# residual verifier — the V8-19.03 closure check
# ---------------------------------------------------------------------------

#: Every table a residual check can name a unit id in directly — the
#: flat unit-keyed set plus the two carriers. ``unit_fts_content`` pairs
#: through ``unit_fts_rows`` (a swept carrier drains it), and the
#: ref-JSON tables strip by reference rather than key, so neither is a
#: direct ``unit_id`` residual.
_VERIFY_UNIT_TABLES: tuple[str, ...] = (
    "units",
    "unit_fts_rows",
) + _UNIT_KEYED_TABLES


def verify_unit_closure_v7(
    conn: sqlite3.Connection, unit_ids: Iterable[str]
) -> dict[str, Any]:
    """Residual-row check for swept units — the V8-19.03 verifier half.

    Counts rows still keyed by any of ``unit_ids`` across
    ``_VERIFY_UNIT_TABLES`` (every table with a direct ``unit_id``
    column in the closure set: the v7 flat-keyed tables, the
    ``unit_fts_rows`` carrier, and the §19 ``unit_time_mentions`` /
    ``unit_doclen`` rows).  ``closed`` is True only when every count is
    zero; per-table residuals are reported under ``residual`` so a leak
    names its table.  Erasure removes a unit's rows at every
    generation, so the check is generation-blind — same rule as the
    sweep.
    """
    ids = sorted({str(u) for u in unit_ids if u})
    residual: dict[str, int] = {}
    if not ids:
        return {"closed": True, "residual": residual}
    for table in _VERIFY_UNIT_TABLES:
        if not has_table(conn, table):
            continue
        n = 0
        for chunk in _chunks(ids):
            n += conn.execute(
                f"SELECT COUNT(*) FROM {table}"
                f" WHERE unit_id IN ({_ph(len(chunk))})",
                chunk,
            ).fetchone()[0]
        if n:
            residual[table] = n
    return {"closed": not residual, "residual": residual}


# ---------------------------------------------------------------------------
# quarantine / suppression cascade — the eligibility query seam
# ---------------------------------------------------------------------------


def _norm_hold_ref(item: Any) -> Optional[tuple[str, str, Optional[int]]]:
    """(kind, id, revision|None) — accepts triples, pairs, or dicts."""
    kind: Any = None
    oid: Any = None
    rev: Any = None
    if isinstance(item, dict):
        kind = item.get("kind") or item.get("object_kind")
        oid = item.get("id") or item.get("object_id")
        rev = item.get("revision", item.get("object_revision"))
    elif isinstance(item, (tuple, list)):
        if len(item) == 2:
            kind, oid = item
        elif len(item) == 3:
            kind, oid, rev = item
    if not isinstance(kind, str) or not kind or not isinstance(oid, str) or not oid:
        return None
    if rev is not None:
        try:
            rev = int(rev)
        except (TypeError, ValueError):
            rev = None
    return (kind, oid, rev)


def _live_hold_refs(conn: sqlite3.Connection) -> set[tuple[str, str, Optional[int]]]:
    """The store's own hold set: excluding-state quarantine rows plus
    active purge tombstones (the suppress→execute window hides the
    units before the sweep lands — V2-41.03 tombstone semantics)."""
    refs: set[tuple[str, str, Optional[int]]] = set()
    if has_table(conn, "quarantine"):
        for kind, oid, rev in conn.execute(
            "SELECT object_kind, object_id, revision FROM quarantine"
            " WHERE state IN ('pending','suppressed')"
        ).fetchall():
            refs.add((kind, oid, int(rev)))
    if has_table(conn, "purges") and has_table(conn, "purge_targets"):
        states = _ph(len(_SUPPRESSING_STATES))
        for kind, oid in conn.execute(
            "SELECT pt.object_kind, pt.object_id FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            f" WHERE p.state IN ({states})",
            _SUPPRESSING_STATES,
        ).fetchall():
            rev: Optional[int] = None
            if kind == "source_revision":
                _src, sep, rt = oid.rpartition(":")
                if sep and rt.isdigit():
                    rev = int(rt)
            refs.add((kind, oid, rev))
    return refs


def held_unit_ids(
    conn: sqlite3.Connection,
    scope_id: str,
    hold_refs: Optional[Iterable[Any]] = None,
) -> set[str]:
    """Unit ids in ``scope_id`` hidden by holds on the covering chain.

    ``hold_refs`` is an optional iterable of ``(kind, id, revision)``
    triples/dicts; ``None`` reads the store's live holds (excluding-state
    ``quarantine`` rows + active purge tombstones).  Cascade rules —
    a unit is held when ANY of these is held:

    * the unit itself (``unit``/``v7_unit``/``units`` kinds, by unit_id);
    * ``("source", source_id, rev)`` — exact source revision, or every
      revision when the hold carries ``rev`` 0/None;
    * ``("source_revision", "sid:rev" | sid, rev)``;
    * ``("span", span_id, rev)`` — resolved through ``spans`` to the
      covering ``(source_id, revision)``;
    * ``("envelope"|"source_envelope", envelope_id, rev)`` — resolved
      through ``source_envelopes`` the same way.

    This is the query layer the V7 eligibility adapter calls — see the
    module docstring for the ``make_eligible`` contract.
    """
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION, "scope_id required")
    if not has_table(conn, "units"):
        return set()
    if hold_refs is None:
        refs = _live_hold_refs(conn)
    else:
        refs = {
            r for r in (_norm_hold_ref(i) for i in hold_refs) if r is not None
        }
    if not refs:
        return set()

    direct = {i for k, i, _r in refs if k in _UNIT_HOLD_KINDS}
    src_revs: set[tuple[str, Optional[int]]] = set()
    for kind, oid, rev in refs:
        if kind == "source":
            src_revs.add((oid, rev if rev else None))
        elif kind == "source_revision":
            src, sep, rt = oid.rpartition(":")
            if sep and rt.isdigit():
                src_revs.add((src, int(rt)))
            else:
                src_revs.add((oid, rev))
        elif kind == "span" and has_table(conn, "spans"):
            for sid, rev_ in conn.execute(
                "SELECT source_id, revision FROM spans WHERE span_id = ?",
                (oid,),
            ).fetchall():
                src_revs.add((sid, int(rev_)))
        elif kind in _ENVELOPE_HOLD_KINDS and has_table(conn, "source_envelopes"):
            for sid, rev_ in conn.execute(
                "SELECT source_id, revision FROM source_envelopes"
                " WHERE envelope_id = ?",
                (oid,),
            ).fetchall():
                src_revs.add((sid, int(rev_)))

    held: set[str] = set()
    for chunk in _chunks(sorted(direct)):
        held.update(
            r[0]
            for r in conn.execute(
                "SELECT unit_id FROM units"
                f" WHERE scope_id = ? AND unit_id IN ({_ph(len(chunk))})",
                [scope_id, *chunk],
            ).fetchall()
        )
    for sid, rev in sorted(src_revs):
        if rev is None:
            rows = conn.execute(
                "SELECT unit_id FROM units WHERE scope_id = ? AND source_id = ?",
                (scope_id, sid),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT unit_id FROM units"
                " WHERE scope_id = ? AND source_id = ? AND revision = ?",
                (scope_id, sid, rev),
            ).fetchall()
        held.update(r[0] for r in rows)
    return held


# ---------------------------------------------------------------------------
# registration record
# ---------------------------------------------------------------------------


def register_v7_erasers(registry: Optional[Any] = None) -> dict[str, Any]:
    """The V7 erasure registration record.

    V7 rows are *projections*, not derivation-graph members — they are
    swept where their source's bytes are erased, not walked as closure
    objects (a ``v7_unit`` member would land in ``outside_boundary`` —
    noise without coverage).  This record declares the wired surfaces so
    audit/reporting code can enumerate them; when ``registry`` is a
    mutable mapping it gains a ``"v7"`` entry (the
    ``register_side_walker`` convention, mapping form).
    """
    wiring: dict[str, Any] = {
        "tables": V7_CLOSURE_TABLES,
        "tables_v8": V8_CLOSURE_TABLES,
        "fts_indexes_via_triggers": V7_FTS_INDEXES,
        "audit_retained": V7_AUDIT_TABLES,
        "hooks": {
            "verbatim/purge.py::_empty_revision": (
                "delete_source_v7(conn, source_id, revision=…)"
            ),
            "verbatim/purge.py::_scrub_derived_content": (
                "delete_source_v7(conn, source_id) per purged source"
            ),
            "verbatim/memory/controls.py::_sweep_derived": (
                "delete_source_v7(conn, source_id)"
            ),
            "verbatim/storage/store.py::_COUNT_TABLES": (
                "integrity counts for every V7 table"
            ),
            "verbatim/export.py::NEVER_EXPORTED": (
                "V7 rows excluded as recomputable derived caches"
            ),
        },
        "rebuild_sweep": "sweep_stale_generations_v7 (post-fence-flip)",
        "eligibility_seam": (
            "held_unit_ids(conn, scope_id, hold_refs) — the query layer "
            "retrieval/v7/eligibility.py::make_eligible should call"
        ),
    }
    if isinstance(registry, dict):
        registry["v7"] = wiring
    return wiring


def v7_closure_coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    """Presence/coverage report for the derived plane — which §30/§19
    tables exist, which the sweeps own, which are audit-retained, and
    any present table outside the covered set (must be empty)."""
    expected = (
        V7_CLOSURE_TABLES + V7_FTS_INDEXES + V7_AUDIT_TABLES
        + V8_CLOSURE_TABLES
    )
    present = [t for t in expected if has_table(conn, t)]
    return {
        "present": present,
        "swept": [t for t in V7_CLOSURE_TABLES if has_table(conn, t)],
        "swept_v8": [t for t in V8_CLOSURE_TABLES if has_table(conn, t)],
        "fts_via_triggers": [t for t in V7_FTS_INDEXES if has_table(conn, t)],
        "audit_retained": [t for t in V7_AUDIT_TABLES if has_table(conn, t)],
        "unhandled": _unhandled_v7(conn),
        "complete": not _unhandled_v7(conn)
        and len(present) == len(expected),
    }


__all__ = [
    "V7_AUDIT_TABLES",
    "V7_CLOSURE_TABLES",
    "V7_FTS_INDEXES",
    "V8_CLOSURE_TABLES",
    "delete_scope_v7",
    "delete_source_v7",
    "held_unit_ids",
    "register_v7_erasers",
    "sweep_stale_generations_v7",
    "v7_closure_coverage",
    "verify_unit_closure_v7",
]
