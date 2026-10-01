"""SQL statement census — SPEC_V8 V8-14.02.

Counts the SQL statements a search issues, bucketed by *normalized
shape* — ``verb:table(key-columns)`` — through SQLite's connection
trace callback (``Connection.set_trace_callback``).  FTS5 shadow
statements (``unit_fts_docsize``, ``unit_fts_content``, …) fire the same
callback, so the census sees exactly the churn D8-11 measured.

Scope honesty: a trace callback sees only statements prepared on *that*
connection.  :meth:`SqlCensus.attach_store` covers the store's
thread-local reader (``Store._reader()`` via ``store.read()``) plus the
writer connection when reachable; the emitted ``scopes`` records which
connections were actually traced.  Attach and query on the same thread.

Output::

    {"statements": 117,
     "by_shape": {"select:units(unit_id,generation)": 42, "tx:begin": 3, …},
     "scopes": ["reader", "writer"]}

The ``statements`` total equals ``sum(by_shape.values())`` — ``tx:*``
shapes (BEGIN/COMMIT/ROLLBACK) are included so the count is complete;
consumers subtract them when they want data statements only.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

SCHEMA = "forensics/sql_census-v1"

_WS = re.compile(r"\s+")
_STR_LIT = re.compile(r"'(?:[^']|'')*'")
_NUM_LIT = re.compile(r"\b\d+(?:\.\d+)?\b")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DQUOTED = re.compile(r"\"([^\"]|\"\")*\"")
_BQUOTED = re.compile(r"`([^`]|``)*`")
_SQUOTED_IDENT = re.compile(r"\[([^\]]|\]\])*\]")

_VERBS_WITH_TABLE = {
    "select": (r"\bfrom\s+(?:only\s+)?", "from"),
    "insert": (r"\binto\s+", "into"),
    "replace": (r"\binto\s+", "into"),
    "update": (r"\bupdate\s+(?:or\s+\w+\s+)?", "update"),
    "delete": (r"\bfrom\s+", "from"),
}
_CLAUSE_END = re.compile(
    r"\b(group\s+by|order\s+by|limit|offset|having|union|intersect|"
    r"except|returning|window)\b"
)
_CMP_COL = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?:=|!=|<>|<=?|>=?|\bis\b|\bin\b|\blike\b|\bglob\b|\bmatch\b|\bbetween\b)",
    re.I,
)
_SET_COL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", re.I)
_MAX_COLS = 8


def _first_ident(text: str, pos: int) -> Optional[str]:
    """Read the identifier at ``pos`` (quoted or bare, optionally
    schema-qualified) — returns ``schema.table`` or ``table``."""
    seg = text[pos:].lstrip()
    m = re.match(
        r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))*",
        seg,
    )
    if not m:
        return None
    name = m.group(0)
    name = name.replace(" ", "")
    parts = [
        p.strip('"`[]') for p in name.split(".") if p.strip('"`[]')
    ]
    return ".".join(parts) if parts else None


def _columns(stmt_norm: str, verb: str, table: Optional[str]) -> List[str]:
    """Key columns: WHERE predicates for reads, the column list for
    writes, SET targets for updates — sorted, deduped, capped."""
    low = stmt_norm.lower()
    cols: List[str] = []
    if verb == "insert" or verb == "replace":
        m = re.search(r"\(([^)]*)\)", stmt_norm)
        if m:
            cols += _IDENT.findall(m.group(1))
    elif verb == "update":
        mset = re.search(r"\bset\b(.*?)(?:\bwhere\b|$)", low)
        if mset:
            cols += _SET_COL.findall(mset.group(1))
    if " where " in low:
        where = low.split(" where ", 1)[1]
        end = _CLAUSE_END.search(where)
        if end:
            where = where[: end.start()]
        for hit in _CMP_COL.finditer(where):
            col = hit.group(1)
            # strip table qualifier; keep the column
            cols.append(col.rsplit(".", 1)[-1])
    seen: List[str] = []
    for c in cols:
        c = c.strip('"`[]').lower()
        if c and c not in seen:
            seen.append(c)
    if len(seen) > _MAX_COLS:
        seen = seen[:_MAX_COLS] + [f"+{len(seen) - _MAX_COLS}"]
    return sorted(seen) if len(seen) <= _MAX_COLS else seen


def normalize_shape(stmt: str) -> str:
    """One SQL statement → its census shape ``verb:table(col,…)``.

    Literals collapse to ``?`` implicitly by never being read; shape
    components are lowercase, whitespace-folded, and column-sorted so
    identical statement *patterns* share a bucket.  Unknown forms still
    get a verb bucket — the census never drops a statement.
    """
    s = stmt.strip().rstrip(";")
    if not s:
        return "empty:"
    s = _WS.sub(" ", s)
    s = _STR_LIT.sub("?", s)
    low = s.lower()

    m = _IDENT.match(s)
    verb = m.group(0).lower() if m else "other"
    if verb == "with":
        # CTE-prefix statement — the payload verb follows the CTEs
        for v in ("insert", "update", "delete", "replace", "select"):
            if re.search(rf"\)\s*{v}\b|,\s*{v}\b|^{v}\b", low):
                verb = f"with_{v}"
                break
        else:
            verb = "with_select" if " select " in low else "with"

    if verb in ("begin", "commit", "rollback", "savepoint", "release",
                "end", "vacuum", "analyze", "reindex"):
        return f"tx:{verb}" if verb in ("begin", "commit", "rollback") \
            else f"{verb}:"

    if verb == "pragma":
        m2 = re.match(r"pragma\s+(?:\w+\s*\.\s*)?([A-Za-z_][A-Za-z0-9_]*)", low)
        return f"pragma:{m2.group(1) if m2 else '?'}"

    if verb == "create":
        m2 = re.match(
            r"create\s+(?:virtual\s+)?(?:temp\s+|temporary\s+)?"
            r"(table|index|view|trigger)\b", low)
        kind = m2.group(1) if m2 else "object"
        rest = low[m2.end():] if m2 else ""
        rest = re.sub(r"^\s*if\s+(?:not\s+)?exists\s+", "", rest)
        name = _first_ident(rest, 0)
        return f"create:{kind}:{name or '?'}"

    if verb == "explain":
        inner = normalize_shape(s[m.end():].strip()) if m else "other:"
        return f"explain:{inner}"

    base_verb = verb.split("_", 1)[-1] if verb.startswith("with_") else verb
    spec = _VERBS_WITH_TABLE.get(base_verb)
    if spec is None:
        return f"{verb}:"
    pattern, _kw = spec
    mm = re.search(pattern, low)
    table = _first_ident(low, mm.end()) if mm else None
    cols = _columns(low, base_verb, table)
    colpart = f"({','.join(cols)})" if cols else ""
    return f"{verb}:{table or '?'}{colpart}"


class SqlCensus:
    """Per-query statement census over one or more traced connections.

    Usage::

        census = SqlCensus()
        census.attach_store(mem._store)          # reader + writer
        result, snap = census.wrap(mem.search, "query", limit=10)

    ``reset()`` / ``snapshot()`` bracket any code region; ``wrap`` is the
    one-call convenience.  Thread affinity: trace callbacks live on the
    connection object — attach and measure on the same thread.
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._total = 0
        self._conns: List[sqlite3.Connection] = []
        self.scopes: List[str] = []
        self.attach_thread: Optional[int] = None

    # -- attach/detach --------------------------------------------------

    def attach(self, conn: sqlite3.Connection, *, scope: str = "reader") -> None:
        """Register the trace callback on ``conn`` (replaces any existing
        callback — the production path installs none)."""
        conn.set_trace_callback(self._on_statement)
        self._conns.append(conn)
        self.scopes.append(scope)
        self.attach_thread = threading.get_ident()

    def attach_store(self, store: Any) -> "SqlCensus":
        """Trace the store's thread-local reader plus its writer conn.

        The reader is materialized through a real ``store.read()`` block;
        the writer is reached via the ``_writer`` attribute when present
        (recorded, never assumed)."""
        with store.read() as conn:
            self.attach(conn, scope="reader")
        writer = getattr(store, "_writer", None)
        if isinstance(writer, sqlite3.Connection):
            self.attach(writer, scope="writer")
        return self

    def detach(self) -> None:
        for conn in self._conns:
            try:
                conn.set_trace_callback(None)
            except Exception:  # noqa: BLE001 — closed conn: nothing to undo
                pass
        self._conns = []

    # -- counting ---------------------------------------------------------

    def _on_statement(self, stmt: str) -> None:
        shape = normalize_shape(stmt)
        self._counts[shape] = self._counts.get(shape, 0) + 1
        self._total += 1

    def reset(self) -> None:
        self._counts = {}
        self._total = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "statements": self._total,
            "by_shape": dict(sorted(self._counts.items())),
            "scopes": list(self.scopes),
        }

    def wrap(self, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        """``(result, census)`` — reset, invoke, snapshot.  The fn's
        exception propagates with the partial census attached to
        ``exc.census`` so failed calls still report their statements."""
        self.reset()
        try:
            res = fn(*a, **kw)
        except Exception as exc:
            exc.census = self.snapshot()  # type: ignore[attr-defined]
            raise
        return res, self.snapshot()

    def counting(self) -> Any:
        """Context-manager form: ``with census.counting() as c: …`` —
        ``c.snapshot()`` afterwards holds the region's counts."""
        census = self

        class _Region:
            def __enter__(self) -> "SqlCensus":
                census.reset()
                return census

            def __exit__(self, *exc: Any) -> bool:
                return False

        return _Region()


# ---------------------------------------------------------------------------
# convenience: one governed search under the census
# ---------------------------------------------------------------------------


def census_search(mem: Any, query: str, **kw: Any) -> Any:
    """Run ``mem.search`` under a census bound to ``mem``'s store.

    Returns ``(SearchResult, census_dict)`` — the census counts the
    call's statements on the store's reader (+writer) connections."""
    census = SqlCensus()
    store = getattr(mem, "_store", None)
    if store is not None:
        census.attach_store(store)
    try:
        return census.wrap(mem.search, query, **kw)
    finally:
        census.detach()


# ---------------------------------------------------------------------------
# CLI — measure one query (or a corpus's task stream) on a real store
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.forensics.sql_census",
        description="V8-14.02 statement census over real Memory.search calls",
    )
    ap.add_argument("--dataset", default=None,
                    help="dataset-registry id; tasks' queries are measured")
    ap.add_argument("--split", default=None)
    ap.add_argument("--query", default=None,
                    help="single ad-hoc query (ignores --dataset)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--timeout-ms", type=float, default=500.0)
    ap.add_argument("--items-json", default=None,
                    help="inline corpus: JSON list of items (no download)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ..arms import DictCorpus, corpus_tasks, item_ref
    from ._common import ForensicVerbatimArm

    if args.query is not None:
        corpus = DictCorpus(
            [{"id": "i1", "text": "placeholder item for ad-hoc census"}],
            [{"task_id": "q0", "query": args.query}],
            name="adhoc", dataset_id="adhoc",
        )
    elif args.items_json:
        items = json.loads(args.items_json)
        corpus = DictCorpus(items, [], name="adhoc", dataset_id="adhoc")
    elif args.dataset:
        from ..corpora import load_corpus

        corpus = load_corpus(args.dataset, args.split)
    else:
        ap.error("one of --query, --items-json, --dataset is required")

    census = SqlCensus()
    arm = ForensicVerbatimArm(census=census)
    try:
        arm.ingest(corpus)
        census.attach_store(arm._mem._store)
        rows = []
        for t in corpus_tasks(corpus):
            q = str(getattr(t, "query", t.get("query") if isinstance(t, Mapping) else ""))
            res, snap = census.wrap(
                arm._mem.search, q, limit=args.limit,
                timeout_ms=args.timeout_ms,
            )
            rows.append({
                "task_id": getattr(t, "task_id", None),
                "query": q,
                "status": str(getattr(res, "status", "")),
                "sql": snap,
            })
    finally:
        census.detach()
        arm.close()

    stmts = [r["sql"]["statements"] for r in rows]
    stmts.sort()
    n = len(stmts)
    report = {
        "schema": SCHEMA,
        "tool": "sql_census",
        "status": "executed" if rows else "not_run",
        "dataset": getattr(corpus, "dataset_id", None),
        "queries": rows,
        "summary": {
            "n": n,
            "p50": stmts[n // 2] if n else None,
            "p95": stmts[min(n - 1, int(0.95 * (n - 1) + 0.5))] if n else None,
            "scopes": census.scopes,
        },
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
    print(json.dumps(report["summary"], indent=1))
    return 0


__all__ = ["SCHEMA", "SqlCensus", "census_search", "normalize_shape"]


if __name__ == "__main__":
    raise SystemExit(main())
