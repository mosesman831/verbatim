"""Calibrated abstention (SPEC_V3 §29.05, §29.06, §29.12, §29.13 — REVISED).

Abstention is a *group-level support* decision, never an all-query-word
veto:

- **Hard constraints**: explicit required identifiers — exact file paths,
  ``::``-separated test ids, and entity ids — present in the query but
  absent from *authorized* evidence force abstention. Ordinary content-word
  overlap, aliases, structured matches, and semantic relevance are fallible
  signals; an absent descriptive term never by itself proves no answer.
- **Group verdicts**: every complete evidence group classifies as
  ``SUPPORTED`` | ``PARTIAL`` | ``INSUFFICIENT``. Only supported groups
  enter answer-support packs; explicit exploratory/archive requests may
  return labeled partial evidence (V3-29.05).
- The engine returns fewer items rather than padding to K; an empty
  authorized result with reason codes is a valid outcome (V3-29.06).
- A query with no lexical signal may still be served by the semantic lane
  when provisioned — coverage probes never disable non-lexical evidence
  (V3-29.12).
"""

from __future__ import annotations

import math as _math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.types_v3 import GroupSupport, QueryClass
from ..candidates import _has_table

# ---- hard-identifier extraction ----------------------------------------

# `::`-separated test ids: pytest/node/gtest style selectors.
_TEST_ID_RE = re.compile(r"[\w.\-]+::[\w.\-]+")
# File paths: contains a slash (abs/rel/`~/`) or ends in a known source
# extension. Applied to whitespace-separated tokens only — a bare prose
# word never qualifies.
_PATH_TOKEN_RE = re.compile(
    r"^(?:[~.]?/[\w.\-/]+|(?:[\w.\-]+/)+[\w.\-]+|"
    r"[\w\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|c|cc|cpp|h|hpp|cs|sh|"
    r"json|ya?ml|toml|xml|sql|md|txt|ini|cfg|css|html|proto|lock))$"
)

# Query classes that may deliver labeled PARTIAL groups (V3-29.05).
_PARTIAL_OK = frozenset({QueryClass.EXPLORATORY, QueryClass.ARCHIVE})


@dataclass
class AbstentionVerdict:
    """The abstention engine's output."""

    verdict: GroupSupport
    hard_identifiers: tuple = ()
    missing_identifiers: tuple = ()
    group_verdicts: dict = field(default_factory=dict)  # group key -> verdict
    reason_codes: tuple = ()

    @property
    def abstained(self) -> bool:
        return self.verdict == GroupSupport.INSUFFICIENT


def extract_hard_identifiers(query: str, entity_ids: tuple) -> tuple:
    """Query tokens that name concrete artifacts the answer must cover.

    Three classes (V3-29.05): ``::`` test ids, file paths / file names,
    and explicit ``entity_ids`` from the request. Ordinary content words —
    even rare ones — are *fallible signals*, never hard constraints.
    """
    found: list = []
    seen: set = set()

    def _add(value: str) -> None:
        v = value.strip().strip("\"'`(),;:…")
        if len(v) >= 2 and v not in seen:
            seen.add(v)
            found.append(v)

    for eid in entity_ids or ():
        _add(eid)
    for token in query.split():
        t = token.strip().strip("\"'`()[]{}<>,;")
        if not t:
            continue
        if _TEST_ID_RE.search(t):
            _add(t)
            continue
        if _PATH_TOKEN_RE.match(t):
            _add(t)
    return tuple(found)


def _identifier_covered(
    conn: sqlite3.Connection,
    identifier: str,
    scope_ids: list,
    generation: int,
) -> bool:
    """One bounded ``LIMIT 1`` probe per authorized evidence surface.

    Presence anywhere in the authorized corpus satisfies the constraint;
    absence is meaningful because every probe is scope-intersected
    (V3-28.04 — hidden-corpus rows can never satisfy it either).
    """
    marks = _ph_sql(len(scope_ids))
    # FTS projection text (claims' indexed evidence)
    if _has_table(conn, "facts_fts"):
        row = conn.execute(
            "SELECT 1 FROM fts_rows fr"
            " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
            " WHERE fr.projection_generation = ?"
            f" AND fr.scope_id IN ({marks})"
            " AND instr(ft.text, ?) > 0 LIMIT 1",
            [generation, *scope_ids, identifier],
        ).fetchone()
        if row is not None:
            return True
    # entity registry (labels + aliases)
    if _has_table(conn, "entities"):
        row = conn.execute(
            "SELECT 1 FROM entities e"
            f" WHERE e.scope_id IN ({marks})"
            " AND (instr(e.label, ?) > 0 OR e.entity_id = ?) LIMIT 1",
            [*scope_ids, identifier, identifier],
        ).fetchone()
        if row is not None:
            return True
        if _has_table(conn, "entity_aliases"):
            row = conn.execute(
                "SELECT 1 FROM entity_aliases ea"
                " JOIN entities e ON e.entity_id = ea.entity_id"
                f" WHERE e.scope_id IN ({marks})"
                " AND instr(ea.normalized_alias, lower(?)) > 0 LIMIT 1",
                [*scope_ids, identifier],
            ).fetchone()
            if row is not None:
                return True
    # learning-plane objects: procedure/episode/observation labels
    for table, col in (
        ("procedures", "task_label"),
        ("episodes", "label"),
        ("observations", "text"),
        ("prospective_records", "intention_text"),
    ):
        if not _has_table(conn, table):
            continue
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE scope_id IN ({marks})"
            f" AND instr({col}, ?) > 0 LIMIT 1",
            [*scope_ids, identifier],
        ).fetchone()
        if row is not None:
            return True
    # exact span payloads not yet interpreted into claims
    if _has_table(conn, "spans"):
        row = conn.execute(
            "SELECT 1 FROM spans sp"
            " JOIN sources so ON so.source_id = sp.source_id"
            " JOIN source_revisions sr"
            "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
            f" WHERE so.scope_id IN ({marks})"
            " AND instr(CAST(sr.payload AS TEXT), ?) > 0 LIMIT 1",
            [*scope_ids, identifier],
        ).fetchone()
        if row is not None:
            return True
    return False


def _ph_sql(n: int) -> str:
    return ",".join("?" * n)


def coverage_probe(
    conn: sqlite3.Connection,
    identifiers: tuple,
    scope_ids: list,
    generation: int,
) -> tuple:
    """``(covered, missing)`` partition of the hard identifiers."""
    covered: list = []
    missing: list = []
    for ident in identifiers:
        if _identifier_covered(conn, ident, scope_ids, generation):
            covered.append(ident)
        else:
            missing.append(ident)
    return tuple(covered), tuple(missing)


def _covered_count(group: Any, terms: tuple) -> int:
    """Query content terms present in the group's evidence text.

    Aggregate support signal (§29.05): not a per-term veto — a group
    whose evidence covers less than half the query's content terms is
    weakly supported even when a lane scored it.
    """
    blob = " ".join(
        getattr(i, "text", "") or "" for i in getattr(group, "items", ())
    ).casefold()
    return sum(1 for t in terms if t in blob)


def _term_floor(terms: tuple, ratio: Optional[float] = None) -> int:
    """Minimum covered content terms for a SUPPORTED verdict.

    ``ratio`` is the coverage fraction required — default 0.5 (the
    published threshold); replay variants may raise or lower it
    (V3-43.02 abstention-threshold replay).
    """
    if len(terms) < 2:
        return 0
    r = 0.5 if ratio is None else max(0.0, min(1.0, float(ratio)))
    return _math.ceil(len(terms) * r)


def group_support_verdicts(
    groups: list,
    ranked_keys: set,
    missing_identifiers: tuple,
    query_class: QueryClass,
    *,
    terms: tuple = (),
    identifiers: tuple = (),
    term_floor_ratio: Optional[float] = None,
) -> dict:
    """Classify each atomic group SUPPORTED | PARTIAL | INSUFFICIENT.

    A group is SUPPORTED when it is complete (all mandatory members
    deliverable) and its primary member came from a fallible-but-real
    lane signal; PARTIAL when it is incomplete or only dependency-ranked;
    INSUFFICIENT is reserved for the request level — a group alone is
    never the abstention trigger.

    Term-coverage floor (§29.05 aggregate support): when the query
    carries no hard identifiers and its evidence covers under half the
    content terms, the group's lane hit was generic overlap, not answer
    support — PARTIAL for exploratory classes, INSUFFICIENT otherwise.
    Queries carrying satisfied hard identifiers are already anchored by
    the coverage probe and skip this check.
    """
    verdicts: dict = {}
    hard_unsatisfied = bool(missing_identifiers)
    floor = _term_floor(terms, term_floor_ratio) if not identifiers else 0
    for g in groups:
        key = g.key
        if hard_unsatisfied:
            verdicts[key] = GroupSupport.INSUFFICIENT
            continue
        if not g.complete:
            verdicts[key] = (
                GroupSupport.PARTIAL
                if query_class in _PARTIAL_OK
                else GroupSupport.INSUFFICIENT
            )
            continue
        if floor and _covered_count(g, terms) < floor:
            verdicts[key] = (
                GroupSupport.PARTIAL
                if query_class in _PARTIAL_OK
                else GroupSupport.INSUFFICIENT
            )
            continue
        verdicts[key] = (
            GroupSupport.SUPPORTED
            if g.primary_key in ranked_keys or g.has_lane_hit
            else GroupSupport.PARTIAL
        )
    return verdicts


def abstain(
    conn: sqlite3.Connection,
    union: Any,
    ranked: list,
    request: Any,
    plan: Any,
    query_class: QueryClass,
    scope_ids: list,
    generation: int,
    groups: Optional[list] = None,
    term_floor_ratio: Optional[float] = None,
) -> AbstentionVerdict:
    """The abstention decision for one recall.

    Order: extract hard identifiers → coverage probe vs authorized
    evidence → per-group verdicts → request verdict. ``groups`` are the
    pack-stage atomic units; when absent (pre-pack call) per-group
    verdicts stay empty and the request verdict uses candidate presence.
    """
    identifiers = extract_hard_identifiers(
        request.query, getattr(request, "entity_ids", ())
    )
    _covered, missing = coverage_probe(
        conn, identifiers, scope_ids, generation
    )
    reasons: list = []
    if missing:
        reasons.append("missing_hard_identifiers")

    terms = tuple(
        t.casefold() for t in (getattr(plan, "terms", ()) or ()) if t
    )
    group_verdicts: dict = {}
    if groups is not None:
        ranked_keys = {e.key for e in ranked}
        group_verdicts = group_support_verdicts(
            groups, ranked_keys, missing, query_class,
            terms=terms, identifiers=identifiers,
            term_floor_ratio=term_floor_ratio,
        )
        floor = _term_floor(terms, term_floor_ratio) if not identifiers else 0
        if floor and any(
            v == GroupSupport.INSUFFICIENT for v in group_verdicts.values()
        ) and all(
            _covered_count(g, terms) < floor for g in groups
        ):
            reasons.append("weak_term_coverage")

    if missing:
        # Hard constraints are unsatisfiable — fail closed (V3-29.05).
        verdict = GroupSupport.INSUFFICIENT
    elif not union:
        verdict = GroupSupport.INSUFFICIENT
        reasons.append("no_authorized_evidence")
    elif groups is not None:
        supported = any(
            v == GroupSupport.SUPPORTED for v in group_verdicts.values()
        )
        partial = any(
            v == GroupSupport.PARTIAL for v in group_verdicts.values()
        )
        if supported:
            verdict = GroupSupport.SUPPORTED
        elif partial and query_class in _PARTIAL_OK:
            verdict = GroupSupport.PARTIAL
            reasons.append("partial_groups_only")
        else:
            verdict = GroupSupport.INSUFFICIENT
            reasons.append("no_supported_groups")
    else:
        # candidates exist; no group assembly yet — the request is at
        # least partially answerable
        verdict = GroupSupport.SUPPORTED
    return AbstentionVerdict(
        verdict=verdict,
        hard_identifiers=identifiers,
        missing_identifiers=missing,
        group_verdicts=group_verdicts,
        reason_codes=tuple(reasons),
    )
