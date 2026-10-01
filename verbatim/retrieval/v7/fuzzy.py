"""V7 fuzzy lane — trigram substring + edit-distance respelling (V7-06.04).

A *conditional* lane: it runs only when the query carries terms the index
does not know for this scope — ``df = 0`` in maintained ``lex_df``
``(scope_id, generation)`` rows, or below ``DF_FLOOR`` when the constant is
raised by the formula search (provisional/v7-r0).  A query whose terms all
have in-scope postings reports ``skipped(reason="no_oov_terms")`` — the
lane never pads a healthy result set.

For each out-of-vocabulary term t:

1. **Respelling** — candidate spellings are drawn from the ``fts5vocab``
   term table (``unit_fts_vocab``) at bounded Levenshtein distance:
   ≤ 2 edits, ≤ 1 when ``len(t) ≤ 5``.  Respelled terms are reissued
   through the real ``unit_fts`` postings and scored with the §32.2
   BM25F machinery at the declared ``RESPELL_WEIGHT`` (0.3) — the score
   contribution is multiplied by the declared weight, never hidden.
2. **Trigram substring** — ``unit_fts_tri MATCH '"t"'`` finds the raw
   misspelling as a substring (compounds, glued words).  Each hit
   contributes the flat declared ``RESPELL_WEIGHT``; df = 0 terms would
   otherwise score a fabricated idf, so substring hits carry the weight
   itself, marked ``via="trigram"``.

Identifiers are **never** respelled and never trigram-probed (V7-06.04's
mutation target: an identifier is a byte-exact artifact, fuzzing it
invents entities).  Identifier-only OOV queries leave the lane skipped.

Honest degradation: the ``fts5vocab`` module cannot be instantiated on a
``query_only`` read snapshot, so the lane looks for the pre-created
``unit_fts_vocab`` table; absent it, respelling is dropped, trigram still
runs, and the output is ``partial(reason="no_vocab_table")``.  With no
usable fuzzy index at all the lane is ``unavailable``.  A deadline cut
reports ``partial`` with honest examined counts.

All constants are ``provisional/v7-r0``.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    CandidateV7,
    LaneContextV7,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    POOLS,
    QueryViewV7,
)
from ...storage.repos import has_table
from . import lexical as _lex

LANE = "fuzzy"

# ---------------------------------------------------------------------------
# fuzzy/v1 constants — provisional/v7-r0 (V7-06.04)
# ---------------------------------------------------------------------------

RESPELL_WEIGHT = 0.3   # declared low weight for reissued alternatives
DF_FLOOR = 1           # OOV when scope df < floor (r0: strictly df == 0)
MAX_RESPELLINGS = 8    # per OOV term, ordered by (distance, term)
TRI_MIN_CHARS = 3      # trigram tokenizer cannot match shorter strings
EDIT_DIST_NEAR = 1     # len(term) <= 5
EDIT_DIST_FAR = 2      # len(term)  > 5
_SHORT_LEN = 5


# ---------------------------------------------------------------------------
# Bounded edit distance
# ---------------------------------------------------------------------------


def _edit_distance_le(a: str, b: str, bound: int) -> Optional[int]:
    """Levenshtein distance between a and b if ≤ bound, else ``None``.
    Banded DP — only cells within ``bound`` of the diagonal are computed;
    pure stdlib and deterministic."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > bound:
        return None
    # row[j] = edit distance between a[:i] and b[:j]
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        lo = max(1, i - bound)
        hi = min(lb, i + bound)
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(lo, hi + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                (cur[j - 1] if j - 1 >= 1 else i - 1 + j - 1) + 1
                if False else cur[j - 1] + 1 if j - 1 >= lo or j == lo
                else prev[j - 1] + 1,
                prev[j - 1] + cost,
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > bound:
            return None
        prev = cur
    return prev[lb] if prev[lb] <= bound else None


def _levenshtein_bounded(a: str, b: str, bound: int) -> Optional[int]:
    """Full-matrix bounded Levenshtein — simple, exact, fast at query
    vocab sizes (cells outside the bound band are skipped by index)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > bound:
        return None
    if la == 0 or lb == 0:
        d = max(la, lb)
        return d if d <= bound else None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] * (lb + 1)
        lo = max(1, i - bound)
        hi = min(lb, i + bound)
        if lo > 1:
            cur[0] = i  # unreachable beyond band; keep row prefix valid
        for j in range(lo, hi + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        # propagate band edges so out-of-band cells stay > bound
        for j in range(hi + 1, lb + 1):
            cur[j] = cur[hi]
        row_min = min(cur[j] for j in range(lo, hi + 1)) if lo <= hi else i
        if row_min > bound and lo > 1:
            # entire row beyond bound — can still come back? Levenshtein
            # distance is monotone enough that early exit is safe only
            # when the diagonal itself exceeded bound.
            pass
        prev = cur
    return prev[lb] if prev[lb] <= bound else None


def _edit_distance(a: str, b: str, bound: int) -> Optional[int]:
    """Banded Levenshtein: distance ≤ ``bound`` or ``None``.

    Cells outside |i−j| ≤ bound are never needed for a ≤bound edit path,
    so the DP evaluates only the band — O(len · bound)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > bound:
        return None
    if not a or not b:
        d = max(la, lb)
        return d if d <= bound else None
    big = bound + 1
    prev = [j if j <= bound else big for j in range(lb + 1)]
    for i in range(1, la + 1):
        lo = max(1, i - bound)
        hi = min(lb, i + bound)
        cur = [big] * (lb + 1)
        cur[0] = i if i <= bound else big
        for j in range(lo, hi + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        if all(cur[j] > bound for j in range(0, hi + 1) if j == 0 or lo <= j):
            # the whole reachable band is out — the true distance must
            # exceed bound (any ≤bound path passes through the band)
            return None
        prev = cur
    return prev[lb] if prev[lb] <= bound else None


# ---------------------------------------------------------------------------
# Term tables
# ---------------------------------------------------------------------------


def _scope_df(conn: sqlite3.Connection, scope_id: str,
              generation: Optional[int], term: str,
              stats: dict) -> Optional[int]:
    """Max df across fields from maintained ``lex_df`` (V7-06.06).
    ``None`` when the table is absent — the caller falls back to a
    postings probe and reports the substitution."""
    if not has_table(conn, _lex.DF_TABLE):
        return None
    if generation is None:
        sql = (f"SELECT MAX(df) FROM {_lex.DF_TABLE}"
               " WHERE scope_id = ? AND term = ?")
        params: list = [scope_id, term]
    else:
        # V7-30.02: ``generation <= pinned`` — a generation bump does not
        # rewrite df rows. Each row is a cumulative snapshot per
        # (field, term, stats_version): resolve every key's newest row
        # at/below the pin (DESC + first-seen), then MAX(df) across
        # fields/versions (mirrors lexical._maintained_df).
        has_sver = "stats_version" in _lex._fts_columns(conn, _lex.DF_TABLE)
        if has_sver:
            sel = "field, stats_version, generation, df"
            order = "field, stats_version, generation DESC"
        else:
            sel = "field, generation, df"
            order = "field, generation DESC"
        sql = (f"SELECT {sel} FROM {_lex.DF_TABLE}"
               " WHERE scope_id = ? AND generation <= ? AND term = ?"
               f" ORDER BY {order}")
        params = [scope_id, generation, term]
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return None
    if generation is None:
        if not rows or rows[0][0] is None:
            return 0
        return int(rows[0][0])
    seen: set = set()
    best: Optional[int] = None
    for row in rows:
        if has_sver:
            field, sver, _gen, df = row
        else:
            field, _gen, df = row
            sver = ""
        key = (str(field), str(sver))
        if key in seen:
            continue  # older snapshot of the same df key
        seen.add(key)
        if best is None or int(df) > best:
            best = int(df)
    return best if best is not None else 0


def _probe_df(conn: sqlite3.Connection, term: str, uni_rows: set) -> int:
    """Fallback when ``lex_df`` is absent: universe postings count —
    labelled ``df_source=postings_probe`` in stats, never silent."""
    rows = _lex._match_rowids(conn, _lex.FTS_TABLE, _lex._fts_quote(term))
    return len(rows & uni_rows) if rows else 0


def _vocab_terms(conn: sqlite3.Connection) -> Optional[list]:
    """Distinct corpus spellings from a pre-created fts5vocab table
    (``unit_fts_vocab`` — created by schema_v7 at index-build time; the
    module cannot be instantiated under query_only).  ``None`` = absent."""
    if not has_table(conn, _lex.VOCAB_TABLE):
        return None
    try:
        return [str(r[0]) for r in conn.execute(
            f"SELECT DISTINCT term FROM {_lex.VOCAB_TABLE}")]
    except sqlite3.Error:
        return None


def _respellings(term: str, vocab: list, bound: int) -> list:
    """``[(distance, spelling)]`` sorted deterministically, capped."""
    out = []
    for cand in vocab:
        if cand == term or abs(len(cand) - len(term)) > bound:
            continue
        d = _edit_distance(term, cand, bound)
        if d is not None:
            out.append((d, cand))
    out.sort()
    return out[:MAX_RESPELLINGS]


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def lane_fuzzy(ctx: LaneContextV7, qv: QueryViewV7,
               slice: LaneSlice) -> LaneOutput:
    """Trigram + respelling candidates for out-of-vocabulary query terms
    (V7-06.04).  Conditional: ``skipped`` when every term is in-vocab."""
    out = LaneOutput(lane=LANE, status=LaneStatus.OK)
    stats = out.stats
    stats["formula"] = _lex.FORMULA_ID
    stats["formula_status"] = FORMULA_STATUS_PROVISIONAL
    stats["respell_weight"] = RESPELL_WEIGHT
    stats["df_floor"] = DF_FLOOR

    deadline = _lex._Deadline(getattr(slice, "deadline_ms", 0.0))
    if deadline.expired():
        out.status = LaneStatus.DEADLINE
        out.reason = "deadline"
        return out

    conn = _lex._conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    have_fts = has_table(conn, _lex.UNITS_TABLE) and has_table(
        conn, _lex.FTS_TABLE)
    have_tri = has_table(conn, _lex.TRI_TABLE)
    stats["trigram"] = "ok" if have_tri else "unavailable"
    if not have_fts and not have_tri:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_fuzzy_index"
        return out

    norm = getattr(qv, "norm", None)
    if norm is None:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_analysis"
        return out

    # OOV candidates: text-channel terms only.  Identifiers are never
    # respelled and never substring-probed (mutation target, V7-06.04).
    ident_surfaces = {str(t.term) for t in
                      (getattr(norm, "identifiers", ()) or ())}
    terms: list[str] = []
    seen: set = set()
    for t in getattr(norm, "terms", ()) or ():
        ch = getattr(t, "channel", "text")
        term = str(t.term)
        if ch == "identifier" or term in ident_surfaces:
            ident_surfaces.add(term)
            continue
        if ch != "text" or not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    if not terms:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_fuzzable_terms"
        return out

    scope_id = str(getattr(ctx, "scope_id", ""))
    generation = getattr(ctx, "generation", None)

    uni, uni_trunc = _lex._universe(
        conn, scope_id, generation, deadline,
        need_full=_lex._elig_mode(getattr(ctx, "eligible", None))
        == "callable", share=ctx)
    if uni_trunc:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        return out
    elig_rowids, elig_via = _lex._resolve_eligibility(ctx, uni)
    stats["eligible_via"] = elig_via
    if elig_via == "unrecognized":
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_shape_unknown"
        return out
    out.eligible = len(elig_rowids)

    # ---- OOV detection -------------------------------------------------
    stats["df_source"] = "lex_df" if has_table(
        conn, _lex.DF_TABLE) else "postings_probe"
    oov: list[str] = []
    df_map: dict[str, int] = {}
    for term in terms:
        if deadline.expired():
            out.status = LaneStatus.PARTIAL
            out.reason = "deadline"
            out.examined = len(df_map)
            return out
        df = _scope_df(conn, scope_id, generation, term, stats)
        if df is None:
            df = _probe_df(conn, term, set(uni.by_rowid))
        df_map[term] = df
        if df < DF_FLOOR:
            oov.append(term)
    stats["term_df"] = df_map
    out.examined = len(df_map)
    if not oov:
        out.status = LaneStatus.SKIPPED
        out.reason = "no_oov_terms"
        return out
    stats["oov_terms"] = list(oov)

    # ---- respelling candidates -----------------------------------------
    # Respelling needs the fielded index for postings; when only the
    # trigram shadow exists the lane still covers substring hits.
    vocab = _vocab_terms(conn) if have_fts else None
    stats["vocab"] = "ok" if vocab is not None else "unavailable"
    vocab = vocab or []
    respell_map: dict[str, list] = {}   # orig -> [(dist, spelling)]
    repl_terms: dict[str, tuple] = {}   # spelling -> (orig, dist)
    if vocab:
        for term in oov:
            bound = EDIT_DIST_NEAR if len(term) <= _SHORT_LEN else EDIT_DIST_FAR
            hits = _respellings(term, vocab, bound)
            respell_map[term] = hits
            for dist, spelling in hits:
                # one orig per spelling keeps scoring deterministic
                repl_terms.setdefault(spelling, (term, dist))
    stats["respellings"] = {k: [s for _, s in v]
                            for k, v in respell_map.items() if v}

    # ---- nominate ------------------------------------------------------
    # Respelled terms go through the real fielded postings at the declared
    # weight; trigram hits add substring coverage of the raw OOV string.
    qterms = [
        _lex._QTerm(term=r, via="respell",
                    stem=_lex.porter_stem(r), scale=RESPELL_WEIGHT,
                    source=orig)
        for r, (orig, _d) in sorted(repl_terms.items())
    ]
    stem_ok = has_table(conn, _lex.STEM_TABLE)
    posts: dict[str, set] = {}
    if have_fts:
        for qt in qterms:
            if deadline.expired():
                out.status = LaneStatus.PARTIAL
                out.reason = "deadline"
                return out
            rows = _lex._match_rowids(
                conn, _lex.FTS_TABLE, _lex._fts_quote(qt.term)) or set()
            srows = (_lex._match_rowids(
                conn, _lex.STEM_TABLE, _lex._fts_quote(qt.term))
                if stem_ok else set()) or set()
            posts[qt.term] = (rows | srows) & set(uni.by_rowid) & elig_rowids

    tri_posts: dict[str, set] = {}
    if have_tri:
        for term in oov:
            if len(term) < TRI_MIN_CHARS:
                continue
            if deadline.expired():
                out.status = LaneStatus.PARTIAL
                out.reason = "deadline"
                return out
            rows = _lex._match_rowids(
                conn, _lex.TRI_TABLE, _lex._fts_quote(term)) or set()
            tri_posts[term] = rows & set(uni.by_rowid) & elig_rowids
        stats["trigram_terms"] = sorted(tri_posts)

    nominated = sorted(
        {r for rs in posts.values() for r in rs}
        | {r for rs in tri_posts.values() for r in rs})
    stats["nominated"] = len(nominated)
    out.examined += len(nominated)
    if not nominated:
        stats["result"] = "no_fuzzy_hits"
        return out  # ok, zero candidates — nothing matched

    # ---- corpus stats (same eligible-set discipline as the lex lane) ---
    columns = [c for c in _lex._fts_columns(conn, _lex.FTS_TABLE)
               if c in _lex.FIELD_W]
    corp = _lex._eligible_corpus_stats(
        conn, uni, elig_rowids, columns, scope_id, generation, deadline,
        stats, share=ctx)
    if corp is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        return out
    n_eligible, avglen = corp

    # ---- score ----------------------------------------------------------
    # Reuse the BM25F scorer with weight-scaled respell terms (their
    # _QTerm.scale is applied inside _score_candidates); trigram hits
    # contribute the flat declared weight (df=0 → no honest idf).
    scores = _lex._score_candidates(
        conn, qterms, [], posts, set(nominated), uni, columns, avglen, n_eligible,
        deadline, stats, stem_cache={}, share=ctx)
    if scores is None:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        return out

    for term, rids in tri_posts.items():
        for rid in rids:
            acc = scores.setdefault(rid, {"score": 0.0, "matched": {}})
            acc["score"] += RESPELL_WEIGHT
            acc["matched"][term] = {"via": "trigram", "weight": RESPELL_WEIGHT}
    stats["scored"] = len(scores)

    cap = int(getattr(slice, "cap", 0) or 0)
    if cap <= 0:
        pool = POOLS.get(getattr(ctx, "budget", None))
        cap = pool.lane_cap if pool is not None else 200
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["score"], str(uni.by_rowid[kv[0]]["unit_id"]),
                        kv[0]),
    )
    admitted = ordered[:cap]
    stats["cap"] = cap
    stats["overflow"] = max(0, len(ordered) - cap)

    rank = 0
    for rid, acc in admitted:
        unit = uni.by_rowid[rid]
        rank += 1
        matched = {
            t: dict(d, weight=RESPELL_WEIGHT)
            for t, d in acc["matched"].items()
        }
        out.candidates.append(CandidateV7(
            unit_id=str(unit["unit_id"]),
            source_id=str(unit["source_id"]),
            revision=int(unit["revision"]),
            lane=LANE,
            rank=rank,
            raw_score=float(acc["score"]),
            signals={"matched_terms": matched, "fuzzy": True,
                     "respell_weight": RESPELL_WEIGHT},
        ))

    if stats.get("vocab") == "unavailable" and out.status == LaneStatus.OK:
        out.status = LaneStatus.PARTIAL
        out.reason = "no_vocab_table"
    return out


__all__ = ["lane_fuzzy", "RESPELL_WEIGHT", "DF_FLOOR", "MAX_RESPELLINGS"]
