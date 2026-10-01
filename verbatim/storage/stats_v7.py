"""Incremental corpus statistics for the V7 lexical engine (V7-06.06).

BM25F needs, per (scope, generation): the eligible-corpus document count N,
per-field average length ``avglen(f) = total_len[f] / N``, and per-field
document frequency ``df(t, f)`` (§32.2). V7-06.06 requires these to be
maintained INCREMENTALLY — df and length sums update in the same
generation-fenced transaction as the posting write, in O(terms of the add),
never by full-corpus recomputation at query time — and keyed by
``(scope, generation, stats_version)``, not by a corpus fingerprint that
every add invalidates (D7-10 closed by design).

Storage:

* ``lex_stats(scope_id, generation, field, stats_version, n_units,
  total_len)`` — ``n_units`` is the number of units contributing to the
  corpus, identical across fields by construction (each unit increments
  every declared field's n_units, empty fields included — BM25F's avglen
  denominator is the corpus N, not the count of units where the field is
  non-empty). ``total_len`` is the field's token-length sum.
* ``lex_df(scope_id, generation, field, term, stats_version, df)`` —
  ``df`` counts units containing the term at least once in that field.

Unit-row shape for ``update_stats``/``decrement_stats``: each row is a dict
``{"unit_id": <str>, "fields": {field: terms}}`` where ``terms`` is either
the analyzer's term iterable for that field or the field's whitespace-joined
token string (the same text that lands in ``unit_fts_content`` — both are
accepted so a caller holding either form updates identical stats). Fields
outside ``LEX_FIELDS`` are ignored; absent fields contribute length 0 but
still count the unit in ``n_units``.

``decrement_stats`` applies the same aggregation with negative sign —
unit deletion adjusts the same sums in O(terms of the removal) (closure,
V7-06.11). df/total_len/n_units floor at 0: a caller double-decrementing
must not push the persisted aggregates negative (a negative df would
poison the BM25F idf log-domain); the floor is the honest bound — the
row stays, at 0, rather than fabricating a still-positive count.

Both functions execute on the caller's connection inside the caller's
transaction — they never commit (the same-tx-as-posting-write contract).
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Mapping

#: Stats-version tag written under ``provisional/v7-r0`` constants — the
#: BM25F field set and definition are §32.2's (tagged ``bm25f/v1``).
STATS_VERSION_V1 = "bm25f/v1"

#: The fielded-FTS column set (§30 ``unit_fts``): the universe ``lex_stats``
#: maintains per unit. Deliberately fixed — stats and index agree by
#: construction.
LEX_FIELDS: tuple[str, ...] = (
    "text",
    "speaker",
    "entities",
    "session",
    "when",
)


def _field_terms(value: object) -> list[str]:
    """Terms for one field: accept a whitespace-joined token string or an
    iterable of terms; anything else is a length-0 field (honest empty)."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    try:
        return [str(t) for t in value]  # type: ignore[union-attr]
    except TypeError:
        return []


def _aggregate(
    unit_rows: Iterable[Mapping],
) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, str], int], int]:
    """Fold a batch of unit rows into per-field (n_units, total_len) deltas
    and per-(field, term) df deltas.

    df counts units, not occurrences: each unit contributes ≤ 1 per term per
    field. ``n_units`` is incremented for every declared field per unit —
    the BM25F avglen denominator is corpus N, identical across fields.
    """
    field_delta: dict[str, int] = {f: 0 for f in LEX_FIELDS}
    df_delta: dict[tuple[str, str], int] = {}
    n = 0
    for row in unit_rows:
        fields = row.get("fields", {})
        if not isinstance(fields, Mapping):
            fields = {}
        n += 1
        for field in LEX_FIELDS:
            terms = _field_terms(fields.get(field))
            field_delta[field] += len(terms)
            # distinct terms per (unit, field) — df is a unit count.
            for term in set(terms):
                key = (field, term)
                df_delta[key] = df_delta.get(key, 0) + 1
    stats_delta = {
        f: (n, field_delta[f]) for f in LEX_FIELDS
    }
    return stats_delta, df_delta, n


def _apply(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_rows: Iterable[Mapping],
    stats_version: str,
    sign: int,
) -> None:
    """Apply the aggregated deltas with ``sign`` (+1 add, -1 remove).

    Upserts, never replaces: the change rides in O(terms of the batch)
    regardless of corpus size (V7-06.06). Persisted sums floor at 0 — a
    caller over-decrementing cannot push aggregates negative.
    """
    stats_delta, df_delta, n = _aggregate(unit_rows)
    if n == 0:
        return
    # UPDATE-first, then floored INSERT: the upsert form cannot floor a
    # fresh insert without flooring ``excluded.*`` too (which would turn
    # every decrement into a no-op on existing rows). A decrement onto a
    # missing row mints 0 — never a negative aggregate (V7-06.06 floor).
    for field, (dn, dlen) in stats_delta.items():
        cur = conn.execute(
            "UPDATE lex_stats SET"
            "   n_units = MAX(0, n_units + ?),"
            "   total_len = MAX(0, total_len + ?)"
            " WHERE scope_id = ? AND generation = ? AND field = ?"
            "   AND stats_version = ?",
            (
                sign * dn,
                sign * dlen,
                scope_id,
                int(generation),
                field,
                stats_version,
            ),
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO lex_stats("
                " scope_id, generation, field, stats_version,"
                " n_units, total_len)"
                " VALUES (?, ?, ?, ?, MAX(0, ?), MAX(0, ?))",
                (
                    scope_id,
                    int(generation),
                    field,
                    stats_version,
                    sign * dn,
                    sign * dlen,
                ),
            )
    for (field, term), ddf in df_delta.items():
        cur = conn.execute(
            "UPDATE lex_df SET df = MAX(0, df + ?)"
            " WHERE scope_id = ? AND generation = ? AND field = ?"
            "   AND term = ? AND stats_version = ?",
            (
                sign * ddf,
                scope_id,
                int(generation),
                field,
                term,
                stats_version,
            ),
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO lex_df("
                " scope_id, generation, field, term, stats_version, df)"
                " VALUES (?, ?, ?, ?, ?, MAX(0, ?))",
                (
                    scope_id,
                    int(generation),
                    field,
                    term,
                    stats_version,
                    sign * ddf,
                ),
            )


def update_stats(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_rows: Iterable[Mapping],
    stats_version: str = STATS_VERSION_V1,
) -> None:
    """Increment corpus stats for newly indexed unit rows — in the SAME
    transaction as the posting writes (V7-06.06), O(terms of the add)."""
    _apply(conn, scope_id, generation, unit_rows, stats_version, +1)


def decrement_stats(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_rows: Iterable[Mapping],
    stats_version: str = STATS_VERSION_V1,
) -> None:
    """Decrement corpus stats for removed units — same O(terms) contract,
    floored at 0 (deletion closure, V7-06.11)."""
    _apply(conn, scope_id, generation, unit_rows, stats_version, -1)


def corpus_stats(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    stats_version: str = STATS_VERSION_V1,
) -> dict[str, dict[str, int]]:
    """``{field: {"n": n_units, "total_len": total_len}}`` for the scope's
    current generation — every declared field present (zeros when the field
    has no stats row yet) so consumers never guess at a missing key."""
    rows = conn.execute(
        "SELECT field, n_units, total_len FROM lex_stats"
        " WHERE scope_id = ? AND generation = ? AND stats_version = ?",
        (scope_id, int(generation), stats_version),
    ).fetchall()
    by_field = {
        field: {"n": int(n), "total_len": int(tl)} for field, n, tl in rows
    }
    return {
        f: by_field.get(f, {"n": 0, "total_len": 0}) for f in LEX_FIELDS
    }


def term_df(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    field: str,
    term: str,
    stats_version: str = STATS_VERSION_V1,
) -> int:
    """Document frequency of ``term`` in ``field`` — 0 when no row exists
    (the fuzzy lane's df-floor probe, V7-06.04)."""
    row = conn.execute(
        "SELECT df FROM lex_df"
        " WHERE scope_id = ? AND generation = ? AND field = ?"
        "   AND term = ? AND stats_version = ?",
        (scope_id, int(generation), field, term, stats_version),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def term_dfs(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    terms: Iterable[str],
    field: str = "text",
    stats_version: str = STATS_VERSION_V1,
) -> dict[str, int]:
    """Batch df lookup — one statement for the term set (query-time hot
    path keeps its SQL-statement budget, V7-17.03)."""
    term_list = sorted(set(terms))
    if not term_list:
        return {}
    marks = ",".join("?" for _ in term_list)
    rows = conn.execute(
        "SELECT term, df FROM lex_df"
        " WHERE scope_id = ? AND generation = ? AND field = ?"
        f"  AND stats_version = ? AND term IN ({marks})",
        (scope_id, int(generation), field, stats_version, *term_list),
    ).fetchall()
    found = {term: int(df) for term, df in rows}
    return {t: found.get(t, 0) for t in term_list}
