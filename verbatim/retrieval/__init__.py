"""Retrieval core: scoped candidates, rank fusion, budgeted evidence.

``search`` orchestrates the recall flow of SPEC §6: scope gate → query
analysis → eligible candidates → fusion → evidence expansion → budgeted
output. It returns evidence bundles only — never ungrounded prose — and
degrades missing optional sources into warnings instead of failing.
"""

from __future__ import annotations

from ..core.time import now_us
from ..core.types import (
    RecallMode,
    RecallRequest,
    RecallResult,
    safe_json_loads,
)
from .candidates import (
    CandidateHit,
    CandidateMap,
    Deadline,
    _allowed_scopes,
    gather,
)
from .fusion import Ranked, rrf
from .package import package
from .query import QueryPlan, analyze

__all__ = [
    "search",
    "analyze",
    "QueryPlan",
    "gather",
    "CandidateHit",
    "CandidateMap",
    "Deadline",
    "rrf",
    "Ranked",
    "package",
]

# Modes that may list eligible memory without a keyword query (V2-25.11):
# the typed empty-result contract applies to lookup modes only.
_BROWSE_MODES = frozenset({RecallMode.TIMELINE, RecallMode.ARCHIVE})


def _dedup(warnings: list) -> list:
    seen: set = set()
    out: list = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _capabilities(hits: CandidateMap) -> dict:
    return {
        "semantic": bool(getattr(hits, "semantic_active", False)),
        "rerank": False,
        "degraded": list(getattr(hits, "degraded", []) or []),
        "notes": dict(getattr(hits, "capability_notes", {}) or {}),
    }


def _uncovered_terms(conn, scope_ids: list, generation: int,
                     plan: QueryPlan) -> tuple:
    """Plan terms that appear nowhere in authorized indexed evidence.

    Coverage is substring containment over the sanctioned projection text
    (``facts_fts`` rows at this generation within readable scopes) —
    tokenizer-agnostic, so it works for CJK and unindexed builds alike.
    Bounded: one ``LIMIT 1`` probe per term. Terms absent from the index
    entirely mean no stored span can satisfy them.
    """
    terms = tuple(t for t in plan.terms if t)
    if not terms or not scope_ids:
        return ()
    uncovered = []
    marks = ",".join("?" * len(scope_ids))
    structured_pairs = tuple(getattr(plan, "predicate_terms", ()))
    requested_predicates = sorted({p for _, p in structured_pairs})
    existing_predicates = set()
    if requested_predicates:
        pred_marks = ",".join("?" * len(requested_predicates))
        existing_predicates = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT predicate FROM claims"
                f" WHERE scope_id IN ({marks})"
                f" AND predicate IN ({pred_marks})",
                [*scope_ids, *requested_predicates],
            ).fetchall()
        }
    for term in terms:
        term_key = term.casefold()
        if any(
            alias == term_key and predicate in existing_predicates
            for alias, predicate in structured_pairs
        ):
            continue
        probe = term.strip(".,;:!?\"'()[]{}") or term
        row = conn.execute(
            "SELECT 1 FROM fts_rows fr"
            " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
            f" WHERE fr.projection_generation = ?"
            f" AND fr.scope_id IN ({marks})"
            " AND instr(lower(ft.text), lower(?)) > 0"
            " LIMIT 1",
            [generation, *scope_ids, probe],
        ).fetchone()
        if row is None:
            uncovered.append(term)
    return tuple(uncovered)


def _semantic_possible(conn, store) -> bool:
    """Whether the semantic channel could still satisfy uncovered terms.

    Abstention is a lexical-coverage signal only: a working encoder plus
    indexed vectors can legitimately surface evidence whose text shares
    no query term (synonyms, paraphrases), so uncovered terms must not
    suppress it.
    """
    from .candidates import _query_encoder

    if _query_encoder(store) is None:
        return False
    row = conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
    return row is not None


def search(store, request: RecallRequest) -> RecallResult:
    """Run the full recall pipeline inside one consistent read snapshot.

    Never raises for missing optional sources: each source guards itself
    and reports through ``capabilities.degraded`` and ``warnings``. An
    empty/stopword-only query returns a typed ``no_signal`` result rather
    than scanning all memory (SPEC §29) — except in explicit timeline/
    archive list modes, where browsing without a keyword is a supported
    operation (V2-25.11). A zero-result distinction between "nothing
    matched" and "nothing authorized" is deliberately not exposed
    (SPEC §9, §31).

    ``request.deadline_ms`` is a cooperative work budget shared by
    candidate gathering and packaging: on expiry the pipeline stops
    gathering, packages what it has, and flags ``DEADLINE_EXCEEDED``
    (V2-25).
    """
    plan = analyze(request.query, request, now_us())
    warnings = list(plan.warnings)
    deadline = Deadline(getattr(request, "deadline_ms", None))
    with store.read() as conn:
        # The generation must come from THIS snapshot — a bump+rebuild
        # committing between a separate reader probe and the snapshot
        # open would filter lanes against rows the snapshot never saw.
        meta_get = getattr(store, "_meta_get", None)
        if callable(meta_get):
            gen_value = meta_get(conn, "projection_generation")
        else:
            try:
                row = conn.execute(
                    "SELECT value_json FROM meta"
                    " WHERE key = 'projection_generation'"
                ).fetchone()
                gen_value = safe_json_loads(row[0]) if row else None
            except sqlite3.Error:
                gen_value = None
        if gen_value is None:
            warnings.append("projection_generation_unavailable")
        generation = int(gen_value or 0)

        # A term-less plan only means "no lexical signal": the semantic
        # channel encodes the raw query text, so it can still answer.
        if (plan.empty and request.mode not in _BROWSE_MODES
                and not _semantic_possible(conn, store)):
            return RecallResult(
                items=(),
                omitted=0,
                warnings=tuple(_dedup(["no_signal", *warnings])),
                capabilities={"semantic": False, "rerank": False,
                              "degraded": []},
                projection_generation=generation,
            )
        scope_ids = _allowed_scopes(conn, request.scope)
        uncovered = _uncovered_terms(conn, scope_ids, generation, plan)
        if (uncovered and request.mode not in _BROWSE_MODES
                and not _semantic_possible(conn, store)):
            # Evidence-first abstention (V2-29): when a content term of
            # the query appears nowhere in authorized indexed evidence,
            # no stored span can satisfy the full query — OR-matching the
            # remaining generic terms would return evidence for a
            # different question (the measured no-answer FP source).
            warnings.append("abstained_uncovered_terms")
            return RecallResult(
                items=(),
                groups=(),
                omitted=0,
                warnings=tuple(_dedup(warnings)),
                capabilities={"semantic": False, "rerank": False,
                              "degraded": []},
                projection_generation=generation,
            )
        hits = gather(conn, store, plan, request, generation, deadline,
                      scope_ids=scope_ids)
        ranked = rrf(hits)
        result = package(
            conn, store, ranked, request, plan, generation, deadline
        )

    warnings.extend(getattr(hits, "warnings", []) or [])
    warnings.extend(result.warnings)
    if not result.items and "no_signal" not in warnings:
        # Covers both "no matching evidence" and "nothing readable": the
        # distinction must stay invisible to unauthorized callers (§9).
        warnings.append("no_authorized_evidence")

    return RecallResult(
        items=result.items,
        groups=result.groups,
        omitted=result.omitted,
        warnings=tuple(_dedup(warnings)),
        capabilities=result.capabilities or _capabilities(hits),
        projection_generation=generation,
    )
