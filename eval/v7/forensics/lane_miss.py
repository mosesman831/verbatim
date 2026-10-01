"""Lane-miss forensic — SPEC_V8 V8-07.01 (scenario K37).

Population: every *answerable* task whose gold was indexed by the arm
but never surfaced into the observable pool — attribution stage
``lane_miss`` (``eval/v7/attribution.py``; withheld rows still enter via
``miss_stage`` — a withheld verdict does not hide where the gold died).

For each such question the tool replays the *recorded* evidence plus a
bounded read-only store probe and assigns exactly one first-loss class:

* ``a`` — df prefetch gate: every gold-matching nominating query term was
  removed by the pre-fetch df gate (``coverage_lexical.df_gate.gated`` /
  ``stats["df_prefetch_gated"]``) and the bounded rescue did not produce
  the gold unit;
* ``b`` — nomination budget: the gold's matching terms were collected
  but *all* were dropped at nomination (``stats["nomination_dropped"]`` —
  ``reason`` recorded verbatim; ``"budget"`` vs ``"df_threshold"`` stays
  distinguishable in the evidence);
* ``c`` — lane cap: the gold was nominated and scored (``stats["scored"]``,
  readable field bytes) but sits outside the admitted candidate list
  while ``stats["overflow"]``/``stats["cap_truncated"]`` record a cut;
* ``d`` — zero lexical overlap: no nominating query term's posting hits
  any eligible gold unit (verified against the real FTS postings, not a
  text guess);
* ``e`` — deadline: the lane's slice expired before the stage that would
  have decided the gold (phase recorded: ``entry`` / ``universe_scan`` /
  ``corpus_stats`` / ``postings`` / ``scoring`` / ``pre_lane``);
* ``f`` — eligibility: a gold unit is denied by the request-time
  eligibility predicate (rebuilt through the same ``make_eligible`` seam
  the facade installs on ``ctx.eligible``), or the lane lost it at the
  S3 eligibility re-check.  **Must be zero** — a nonzero count is a
  defect (K37), surfaced under ``report["defects"]``;
* ``g`` — other / insufficient evidence: the gold got past every
  recorded nomination stage but the captured data cannot order the
  surviving stages (records the raw evidence, never a guess).

Evidence discipline: probes run inside one ``store.read()`` snapshot —
the same generation fence + eligibility snapshot the lane ran under —
and re-execute the lane's *own* primitives (``MATCH`` rowid sets, the
``ORDER BY rowid LIMIT`` rescue scan, the units fencing ORDER BY).  Rows
carry ids, term strings, counts and status keys only — never payload
text, gold document text, or explain blobs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..attribution import attribute_detail
from ._common import (
    ArmSpec,
    CONSTANTS_TAG,
    DETERMINISM_EXEMPT,
    ForensicVerbatimArm,
    PolicyPatch,
    _jsonable,
    arm_report_kwargs,
    category_id_of,
    dataset_block,
    env_block,
    ref_sessions,
    run_spec,
    task_views,
    write_report,
)

SCHEMA = "forensics/lane_miss-v1"
REQUIREMENT = "V8-07.01"
LANE = "lex"
LANE_STATUS_KEY = "v7.lex"

CLASS_ORDER = ("a", "b", "c", "d", "e", "f", "g")
CLASS_LABELS = {
    "a": "df_prefetch_gate",
    "b": "nomination_budget",
    "c": "lane_cap",
    "d": "zero_lexical_overlap",
    "e": "deadline",
    "f": "eligibility",
    "g": "other",
}

FTS_TABLE = "unit_fts"
STEM_TABLE = "unit_fts_stem"
UNITS_TABLE = "units"


# ---------------------------------------------------------------------------
# local mirrors of the lane's tiny primitives — kept byte-identical so the
# probes measure the same sets the lane measured (no private imports; the
# bodies are verbatim copies of retrieval/v7/lexical.py helpers).
# ---------------------------------------------------------------------------


def _fts_quote(term: str) -> str:
    return '"' + str(term).replace('"', '""') + '"'


def _match_rowids(conn: sqlite3.Connection, table: str,
                  match: str) -> Optional[set]:
    """Universe-wide rowids matching an FTS5 query; ``None`` on error."""
    try:
        cur = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (match,)
        )
        return {int(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return None


def _fts_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [str(r[1]) for r in conn.execute(
            f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    try:
        from verbatim.storage.repos import has_table
        return bool(has_table(conn, name))
    except Exception:  # noqa: BLE001 — probe degrades, never crashes
        try:
            conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
            return True
        except sqlite3.Error:
            return False


# ---------------------------------------------------------------------------
# term-list reconstruction — the lane's own _query_terms replay
# ---------------------------------------------------------------------------


def _lane_terms(
    query_text: str,
    stats: Mapping[str, Any],
    explain: Optional[Mapping[str, Any]],
) -> Tuple[List[str], List[str], Dict[str, bool], str, Optional[str]]:
    """``(terms, idents, content_flags, source, note)`` — the lane's term
    split, rebuilt with the same ``_query_terms`` the lane ran.

    Preferred source: replay ``norm/v2`` over the query text and run the
    real ``_query_terms`` — byte-identical terms + content flags.
    Fallbacks: ``stats["df"]``/``stats["df_maintained"]`` keys ∪ the
    gated/dropped/rescue term lists (still the lane's own records), then
    ``explain.query.terms ∪ identifiers`` (approximate — may carry
    stem-channel twins; flagged).  ``content_flags`` is ``term -> bool``
    for nominating terms; empty when the filter is unavailable.
    """
    note: Optional[str] = None
    try:
        from types import SimpleNamespace

        from verbatim.retrieval.v7.lexical import _query_terms
        from verbatim.text.norm_v2 import analyze as _analyze

        norm = _analyze(str(query_text))
        qterms, idents = _query_terms(SimpleNamespace(norm=norm))
        terms = sorted({str(q.term) for q in qterms})
        id_list = sorted({str(q.term) for q in idents})
        content = {str(q.term): bool(q.content) for q in qterms}
        return terms, id_list, content, "norm_v2_replay", None
    except Exception as exc:  # noqa: BLE001 — degrade honestly
        note = f"norm_v2_replay_failed:{type(exc).__name__}"

    # stats-key fallback — every key here is a term the lane itself parsed
    terms: set = set()
    for key in ("df", "df_maintained"):
        block = stats.get(key)
        if isinstance(block, Mapping):
            terms.update(str(t) for t in block)
    cov = stats.get("coverage_lexical") or {}
    gate = cov.get("df_gate") or {}
    for block in (gate.get("gated") or (), stats.get("df_prefetch_gated") or ()):
        for e in block:
            if isinstance(e, Mapping) and e.get("term") is not None:
                terms.add(str(e["term"]))
    for e in stats.get("nomination_dropped") or ():
        if isinstance(e, Mapping) and e.get("term") is not None:
            terms.add(str(e["term"]))
    rescue = cov.get("rescue") or {}
    for t in rescue.get("terms") or ():
        terms.add(str(t))
    query_ex = (explain or {}).get("query") or {}
    idents = sorted(str(x) for x in (query_ex.get("identifiers") or ()))
    if not terms:
        # last resort — explain's term list may include stem-channel
        # twins the lane never nominated with; flagged approximate.
        terms = {str(x) for x in (query_ex.get("terms") or ())}
        if terms:
            return (sorted(terms), idents, {}, "explain.query",
                    (note or "") + "|terms_approximate")
        return [], idents, {}, "unavailable", note
    try:
        from verbatim.retrieval.v7.lexical import _content_filter
        noncontent = _content_filter()
        content = {t: t not in noncontent for t in terms}
    except Exception:  # noqa: BLE001
        content = {}
        note = (note or "") + "|content_filter_unavailable"
    return sorted(terms), idents, content, "stats", note


# ---------------------------------------------------------------------------
# store probe — one pinned read snapshot per run
# ---------------------------------------------------------------------------


class _LaneProbe:
    """Read-only replays of the lane's fencing/matching primitives.

    Built once per run inside ``store.read()`` — the identical snapshot
    discipline the lane ran under (``ctx.conn`` is a pinned read).  All
    per-question probes reuse ``uni_rows``/``elig``/``stem_ok`` so the
    evidence is consistent across the report.
    """

    def __init__(self, arm: ForensicVerbatimArm) -> None:
        self.arm = arm
        self.mem = getattr(arm, "_mem", None)
        self.store = getattr(self.mem, "_store", None)
        self.scope_id = str(getattr(self.mem, "_namespace", "") or "")
        self.owner = str(getattr(self.mem, "_owner", "") or "")
        self.generation: Optional[int] = None
        self.elig_ids: Optional[frozenset] = None
        self.elig_stats: Dict[str, Any] = {}
        self.elig_error: Optional[str] = None
        self.uni_rows: Dict[int, str] = {}          # rowid -> unit_id
        self.elig_rids: Optional[set] = None         # None = unverifiable
        self.stem_ok = False
        self.fts_ok = False
        self.fts_columns: List[str] = []
        self._post_cache: Dict[Tuple[str, bool], Optional[set]] = {}
        self._units_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._fields_cache: Dict[int, Optional[bool]] = {}
        self._conn: Any = None
        self._stack: Any = None

    # -- snapshot plumbing --------------------------------------------------

    def open(self) -> Tuple[Any, Optional[str]]:
        """Enter the pinned read snapshot; returns ``(conn, error)``."""
        if self.store is None:
            return None, "no_store"
        self._stack = contextlib.ExitStack()
        try:
            conn = self._stack.enter_context(self.store.read())
        except Exception as exc:  # noqa: BLE001
            self._stack = None
            return None, f"read_snapshot:{type(exc).__name__}: {exc}"
        self._conn = conn
        try:
            try:
                self.generation = int(self.store._meta_get(
                    conn, "projection_generation") or 0)
            except Exception:  # noqa: BLE001
                self.generation = int(self.store.projection_generation())
            self.fts_ok = _has_table(conn, FTS_TABLE)
            self.stem_ok = _has_table(conn, STEM_TABLE)
            self.fts_columns = (
                _fts_columns(conn, FTS_TABLE) if self.fts_ok else []
            )
            self.uni_rows = self._universe_rows(conn)
            self._eligibility(conn)
            self.elig_rids = (
                None if self.elig_ids is None
                else {r for r, uid in self.uni_rows.items()
                      if uid in self.elig_ids}
            )
        except Exception as exc:  # noqa: BLE001
            self._stack.close()
            self._stack = None
            self._conn = None
            return None, f"probe_init:{type(exc).__name__}: {exc}"
        return conn, None

    def close(self, conn: Any = None) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._conn = None

    def _universe_rows(self, conn: Any) -> Dict[int, str]:
        """``rowid -> unit_id`` — the lane's ``_universe`` fence:
        ``scope_id`` + ``kind='turn'`` + ``generation <= pinned``, latest
        row per unit_id (DESC first-seen)."""
        out: Dict[int, str] = {}
        seen: set = set()
        try:
            if self.generation is None:
                cur = conn.execute(
                    f"SELECT rowid, unit_id FROM {UNITS_TABLE}"
                    " WHERE scope_id = ? AND kind = 'turn'",
                    (self.scope_id,),
                )
            else:
                cur = conn.execute(
                    f"SELECT rowid, unit_id FROM {UNITS_TABLE}"
                    " WHERE scope_id = ? AND generation <= ?"
                    " AND kind = 'turn'"
                    " ORDER BY unit_id, generation DESC",
                    (self.scope_id, self.generation),
                )
        except sqlite3.Error:
            return out
        for rid, uid in cur.fetchall():
            uid = str(uid)
            if uid in seen:
                continue
            seen.add(uid)
            out[int(rid)] = uid
        return out

    def _eligibility(self, conn: Any) -> None:
        """Rebuild the facade's ``ctx.eligible`` on this snapshot."""
        if self.generation is None:
            self.elig_error = "generation_unpinned"
            return
        try:
            from verbatim.retrieval.v7.eligibility import make_eligible
            elig = make_eligible(
                conn, self.store, scope_id=self.scope_id,
                generation=self.generation, principal_id=self.owner,
            )
            self.elig_stats = dict(getattr(elig, "stats", {}) or {})
            ids = getattr(elig, "unit_ids", None)
            self.elig_ids = frozenset(str(u) for u in ids) \
                if ids is not None else None
            if self.elig_stats.get("mode") == "unavailable":
                self.elig_error = str(
                    self.elig_stats.get("reason") or "unavailable"
                )
        except Exception as exc:  # noqa: BLE001
            self.elig_error = f"{type(exc).__name__}: {exc}"
            self.elig_ids = None

    def eligible_of(self, unit_id: str) -> Optional[bool]:
        """``True/False`` from the predicate, ``None`` unverifiable."""
        if self.elig_ids is None:
            return None
        return str(unit_id) in self.elig_ids

    # -- per-term postings ---------------------------------------------------

    def postings(self, term: str, *, ident: bool) -> Optional[set]:
        """Unfenced MATCH rowid set for one term — exact channel plus the
        stem channel for non-identifiers, mirroring ``_collect_postings``'s
        per-term probes.  ``None`` when a channel errored."""
        key = (str(term), bool(ident))
        if key in self._post_cache:
            return self._post_cache[key]
        conn = self._conn
        if conn is None:
            return None
        ex = _match_rowids(conn, FTS_TABLE, _fts_quote(term))
        if ident:
            self._post_cache[key] = ex
            return ex
        st = (
            _match_rowids(conn, STEM_TABLE, _fts_quote(term))
            if self.stem_ok else set()
        )
        out = None if ex is None or st is None else (ex | st)
        self._post_cache[key] = out
        return out

    # -- gold units ------------------------------------------------------------

    def gold_units(self, source_id: str) -> List[Dict[str, Any]]:
        """Latest in-fence ``kind='turn'`` unit rows for one source."""
        if source_id in self._units_cache:
            return self._units_cache[source_id]
        out: List[Dict[str, Any]] = []
        if self._conn is None:
            return out
        try:
            if self.generation is None:
                cur = self._conn.execute(
                    f"SELECT rowid, unit_id, generation FROM {UNITS_TABLE}"
                    " WHERE source_id = ? AND scope_id = ?"
                    " AND kind = 'turn'",
                    (str(source_id), self.scope_id),
                )
            else:
                cur = self._conn.execute(
                    f"SELECT rowid, unit_id, generation FROM {UNITS_TABLE}"
                    " WHERE source_id = ? AND scope_id = ?"
                    " AND generation <= ? AND kind = 'turn'"
                    " ORDER BY unit_id, generation DESC",
                    (str(source_id), self.scope_id, self.generation),
                )
            seen: set = set()
            for rid, uid, gen in cur.fetchall():
                uid = str(uid)
                if uid in seen:
                    continue
                seen.add(uid)
                out.append({"rowid": int(rid), "unit_id": uid,
                            "generation": int(gen)})
        except sqlite3.Error:
            out = []
        self._units_cache[str(source_id)] = out
        return out

    # -- field readability (scored-membership proxy) ---------------------------

    def fields_readable(self, rowids: Iterable[int]) -> Optional[set]:
        """Rowids whose ``unit_fts`` row exists with ≥1 non-NULL field —
        the lane's ``_fetch_fields``/null-row test.  ``None`` on error."""
        rids = sorted({int(r) for r in rowids})
        if not rids:
            return set()
        if self._conn is None:
            return None
        if not self.fts_ok or not self.fts_columns:
            return set()
        cols = ",".join(
            '"' + c.replace('"', '""') + '"' for c in self.fts_columns
        )
        out: set = set()
        try:
            for i in range(0, len(rids), 400):
                chunk = rids[i:i + 400]
                ph = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"SELECT rowid, {cols} FROM {FTS_TABLE}"
                    f" WHERE rowid IN ({ph})",
                    chunk,
                )
                for row in cur.fetchall():
                    if any(row[j + 1] is not None
                           for j in range(len(self.fts_columns))):
                        out.add(int(row[0]))
        except sqlite3.Error:
            return None
        return out

    # -- rescue replay ---------------------------------------------------------

    def rescue_replay(self, rescue: Mapping[str, Any],
                      posts_fenced: Mapping[str, set]) -> set:
        """Replay the §21.4 bounded rescue's visit order for exactly the
        measured ``rows`` count — identical prefix ⇒ identical produced
        set.  ``posts_fenced`` holds the lane-time fenced postings for
        terms that were fetched (non-empty ⇒ sorted reuse, matching
        ``if posts.get(t)`` in the lane)."""
        terms = [str(t) for t in (rescue.get("terms") or ())]
        budget = int(rescue.get("rows") or 0)
        rescued: set = set()
        visited = 0
        uni = self.uni_rows.keys()
        elig = self.elig_rids
        for t in terms:
            if visited >= budget:
                break
            stop = False
            reuse = posts_fenced.get(t)
            if reuse:
                sources: List[Any] = [iter(sorted(int(r) for r in reuse))]
            else:
                sources = []
                tables = ((FTS_TABLE, STEM_TABLE) if self.stem_ok
                          else (FTS_TABLE,))
                for table in tables:
                    remaining = budget - visited
                    if remaining <= 0:
                        break
                    try:
                        sources.append(self._conn.execute(
                            f"SELECT rowid FROM {table}"
                            f" WHERE {table} MATCH ?"
                            " ORDER BY rowid LIMIT ?",
                            (_fts_quote(t), remaining)))
                    except sqlite3.Error:
                        continue
            for it in sources:
                for row in it:
                    if visited >= budget:
                        stop = True
                        break
                    visited += 1
                    rid = int(row[0]) if isinstance(row, tuple) else int(row)
                    if rid in uni and (elig is None or rid in elig):
                        rescued.add(rid)
                if stop:
                    break
        return rescued


# ---------------------------------------------------------------------------
# deadline phase — where the lane's slice actually expired
# ---------------------------------------------------------------------------


def _deadline_phase(status: Optional[str], reason: Optional[str],
                    stats: Mapping[str, Any]) -> Optional[str]:
    deadlineish = (
        status == "deadline"
        or str(reason or "").startswith("deadline")
        or reason == "no_slice_budget"
    )
    if not deadlineish:
        return None
    if status == "deadline":
        return "entry" if reason == "deadline" else "pre_lane"
    if stats.get("universe_scan") == "truncated":
        return "universe_scan"
    if stats.get("stats_phase") == "deadline_cut":
        return "corpus_stats"
    if "nominated_terms" not in stats and "nomination_dropped" not in stats:
        return "postings"
    if "scored" not in stats and "nominated" in stats:
        return "scoring"
    return "unknown"


# ---------------------------------------------------------------------------
# the pure classifier — everything it needs is the assembled evidence dict
# ---------------------------------------------------------------------------


def classify_lane_miss(ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(letter, detail)`` — the first recorded stage that lost the gold.

    ``ev`` is the probe-assembled evidence mapping (see
    ``_assemble_evidence``); the function itself is pure so unit tests can
    exercise every branch without a store.
    """
    stats = ev.get("stats") or {}
    cov = stats.get("coverage_lexical") or {}
    gate = cov.get("df_gate") or {}
    gated: set = {
        str(e.get("term"))
        for e in (gate.get("gated") or ()) if isinstance(e, Mapping)
    }
    gated |= {
        str(e.get("term"))
        for e in (stats.get("df_prefetch_gated") or ())
        if isinstance(e, Mapping)
    }
    dropped: Dict[str, Mapping] = {
        str(e.get("term")): e
        for e in (stats.get("nomination_dropped") or ())
        if isinstance(e, Mapping) and e.get("term") is not None
    }
    nominating = {str(t) for t in (ev.get("nominating") or ())}
    status = ev.get("lane_status")
    reason = ev.get("lane_reason")
    deadlineish = (
        status == "deadline"
        or str(reason or "").startswith("deadline")
        or reason == "no_slice_budget"
    )
    gate_ran = bool(gate) or "df_prefetch_gated" in stats
    nom_ran = "nominated_terms" in stats or "nomination_dropped" in stats
    scored_done = "scored" in stats
    rescue_fired = bool((cov.get("rescue") or {}).get("fired"))
    overflow = int(stats.get("overflow") or 0)
    cap_trunc = int(stats.get("cap_truncated") or 0)
    post = ev.get("post_pool") or {}
    fused_n, scored_n = post.get("fused"), post.get("scored")
    dphase = _deadline_phase(status, reason, stats)

    units = list(ev.get("gold_units") or ())
    detail: Dict[str, Any] = {
        "lane_status": status,
        "lane_reason": reason,
        "gate_ran": gate_ran,
        "nomination_ran": nom_ran,
        "scoring_done": scored_done,
        "deadline_phase": dphase,
    }

    if ev.get("probe_error"):
        detail["note"] = "probe_unavailable"
        detail["probe_error"] = ev.get("probe_error")
        return "g", detail
    if not units:
        detail["note"] = "no_turn_unit"
        return "g", detail

    # Lane-level eligibility shape defect — the lane itself refused.
    if stats.get("eligible_via") == "unrecognized" \
            or reason == "eligibility_shape_unknown":
        detail["note"] = "eligibility_shape_unknown"
        return "f", detail

    def _unit_stage(u: Mapping[str, Any]) -> Tuple[str, str]:
        """``(letter, note)`` — where this one gold unit died."""
        if u.get("eligible") is False:
            return "f", "unit_denied"
        matching = [t for t in (u.get("matching") or ())
                    if t in nominating]
        if not matching:
            return "d", "no_nominating_overlap"
        if not gate_ran:
            if deadlineish:
                return "e", f"deadline_{dphase or 'unknown'}"
            if not ev.get("lane_present"):
                return "g", "lane_block_absent"
            return "g", "df_gate_block_absent"
        if all(t in gated for t in matching):
            if u.get("rescued"):
                pass  # produced by the bounded rescue → nominated path
            elif u.get("rescued") is None and rescue_fired:
                # rescue produced something but replay could not prove
                # this unit wasn't it — (a) is not provable
                return "g", "rescue_membership_unverifiable"
            else:
                return "a", "all_matching_gated"
        elif nom_ran:
            kept = [t for t in matching
                    if t not in gated and t not in dropped]
            if not kept and not u.get("rescued"):
                if u.get("rescued") is None and rescue_fired:
                    return "g", "rescue_membership_unverifiable"
                return "b", "all_matching_dropped"
        else:
            if deadlineish:
                return "e", f"deadline_{dphase or 'postings'}"
            return "g", "nomination_record_absent"
        # nominated — did scoring complete?
        if not scored_done:
            if deadlineish:
                return "e", f"deadline_{dphase or 'scoring'}"
            return "g", "scoring_incomplete"
        if u.get("admitted"):
            if u.get("in_items"):
                return "g", "scored_below_delivery"
            trimmed = (
                fused_n is not None and scored_n is not None
                and int(fused_n) > int(scored_n)
            )
            if trimmed:
                return "g", ("post_pool_trim_or_recheck"
                             if ev.get("recheck_dropped")
                             else "post_pool_trim")
            if ev.get("recheck_dropped") and u.get("eligible") is not True:
                # S3 dropped ≥1 admitted candidate; gold's own
                # eligibility can't be verified — honest (g).
                return "g", "recheck_dropped_unverifiable"
            return "g", "admitted_not_scored"
        if u.get("readable") is False:
            return "g", "no_field_content"
        if u.get("readable") is None:
            return "g", "ordered_membership_unverifiable"
        if overflow or cap_trunc:
            return "c", "cap_cut"
        return "g", "nominated_scored_unadmitted"

    # deepest-surviving unit decides — the question dies where its last
    # live gold candidate died.
    _DEPTH = {
        "unit_denied": 1,                      # f — never saw a pool
        "no_nominating_overlap": 2,            # d
        "lane_block_absent": 3,
        "df_gate_block_absent": 3,
        "all_matching_gated": 4,               # a
        "all_matching_dropped": 5,             # b
        "nomination_record_absent": 5,
        "scoring_incomplete": 6,
        "no_field_content": 6,
        "ordered_membership_unverifiable": 6,
        "cap_cut": 7,                          # c
        "nominated_scored_unadmitted": 7,
        "post_pool_trim": 8,
        "post_pool_trim_or_recheck": 8,
        "recheck_dropped_unverifiable": 8,
        "admitted_not_scored": 8,
        "scored_below_delivery": 9,
    }
    _DEADLINE_DEPTH = {
        "entry": 3, "pre_lane": 3, "unknown": 3,
        "universe_scan": 3, "corpus_stats": 3,
        "postings": 5, "scoring": 6,
    }

    def _depth(letter: str, note: str) -> float:
        if note.startswith("deadline_"):
            return float(_DEADLINE_DEPTH.get(note[len("deadline_"):], 3))
        return float(_DEPTH.get(note, 3))

    staged: List[Tuple[float, str, str, Mapping[str, Any]]] = []
    for u in units:
        letter, note = _unit_stage(u)
        staged.append((_depth(letter, note), letter, note, u))
    staged.sort(
        key=lambda x: (-x[0], x[1], x[2],
                       str(x[3].get("ref")), str(x[3].get("unit_id")))
    )
    depth, letter, note, unit = staged[0]
    detail["note"] = note
    detail["unit_stages"] = [
        {"ref": u.get("ref"), "unit_id": u.get("unit_id"),
         "class": l, "note": n, "pack": u.get("pack")}
        for _d, l, n, u in staged
    ]
    return letter, detail


# ---------------------------------------------------------------------------
# evidence assembly — per lane_miss question
# ---------------------------------------------------------------------------


def _assemble_evidence(
    probe: _LaneProbe,
    tv: Any,
    row: Mapping[str, Any],
    explain: Optional[Mapping[str, Any]],
    coverage: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the classifier's evidence dict from explain + store probes."""
    lanes = (explain or {}).get("lanes") or {}
    lane = lanes.get(LANE)
    lstats = dict((lane or {}).get("stats") or {})
    # the pipeline grafts stats["coverage_lexical"] → coverage.lexical
    cov_lex = (coverage or {}).get("lexical") or {}
    cov_block = lstats.get("coverage_lexical") or cov_lex
    stats = dict(lstats)
    if cov_block and "coverage_lexical" not in stats:
        stats["coverage_lexical"] = cov_block

    lane_status = (lane or {}).get("status")
    lane_reason = (lane or {}).get("reason")
    if lane_status is None:
        lane_status = ((coverage or {}).get("lanes") or {}).get(
            LANE_STATUS_KEY)

    terms, idents, content, terms_source, terms_note = _lane_terms(
        tv.query, stats, explain)
    ident_set = set(idents)
    qterm_terms = [t for t in terms if t not in ident_set]
    has_content = any(content.get(t) for t in qterm_terms)
    if content:
        nominating = sorted(
            ident_set | {t for t in qterm_terms
                         if content.get(t) or not has_content}
        )
    else:
        # content flags unavailable — treat every non-ident term as
        # nominating and flag it (over-inclusive, evidence keeps terms).
        nominating = sorted(ident_set | set(qterm_terms))

    gate = (cov_block or {}).get("df_gate") or {}
    gated_terms = {
        str(e.get("term"))
        for e in (gate.get("gated") or ()) if isinstance(e, Mapping)
    } | {
        str(e.get("term"))
        for e in (stats.get("df_prefetch_gated") or ())
        if isinstance(e, Mapping)
    }
    dropped = {
        str(e.get("term")): e
        for e in (stats.get("nomination_dropped") or ())
        if isinstance(e, Mapping) and e.get("term") is not None
    }

    # unfenced postings per term (lane MATCH semantics)
    posts: Dict[str, Optional[set]] = {}
    for t in terms:
        posts[t] = probe.postings(t, ident=t in ident_set)

    # fenced posts for the rescue replay (lane's posts[t] shape)
    elig_rids = probe.elig_rids
    uni_rids = set(probe.uni_rows)
    posts_fenced: Dict[str, set] = {}
    for t, s in posts.items():
        if not s:
            posts_fenced[t] = set()
            continue
        fenced = s & uni_rids
        if elig_rids is not None:
            fenced &= elig_rids
        posts_fenced[t] = fenced

    # rescue replay — only when the lane reported it fired
    rescue = (cov_block or {}).get("rescue") or {}
    rescue_replayed: Optional[set] = None
    if rescue.get("fired"):
        try:
            rescue_replayed = probe.rescue_replay(rescue, posts_fenced)
        except Exception:  # noqa: BLE001
            rescue_replayed = None

    # gold units
    ref_source = getattr(probe.arm, "_ref_source", {}) or {}
    gold_refs = sorted(str(r) for r in tv.gold_item)
    units: List[Dict[str, Any]] = []
    unmapped: List[str] = []
    for ref in gold_refs:
        sid = ref_source.get(ref)
        if not sid:
            unmapped.append(ref)
            continue
        for u in probe.gold_units(sid):
            units.append({"ref": ref, **u})
    admitted_uids = {
        str(c.get("unit_id"))
        for c in (lane or {}).get("candidates") or ()
        if isinstance(c, Mapping) and c.get("unit_id") is not None
    }
    item_uids = {
        str(i.get("unit_id"))
        for i in (explain or {}).get("items") or ()
        if isinstance(i, Mapping) and i.get("unit_id") is not None
    }
    item_packs = {
        str(i.get("unit_id")): i.get("pack")
        for i in (explain or {}).get("items") or ()
        if isinstance(i, Mapping) and i.get("unit_id") is not None
    }
    readable = probe.fields_readable(u["rowid"] for u in units)

    # Open-set inference: the lane's own n_eligible/n_universe stats
    # prove every fenced unit was eligible — the per-unit predicate is
    # then known-True even when the probe could not materialize the set.
    elig_state = "verified" if probe.elig_ids is not None \
        else "unverifiable"
    if probe.elig_ids is None:
        n_e, n_u = stats.get("n_eligible"), stats.get("n_universe")
        if isinstance(n_e, int) and isinstance(n_u, int) \
                and n_e == n_u and not isinstance(n_e, bool):
            elig_state = "open_set"

    for u in units:
        rid = u["rowid"]
        if probe.elig_ids is not None:
            u["eligible"] = probe.eligible_of(u["unit_id"])
        elif elig_state == "open_set":
            u["eligible"] = True
        else:
            u["eligible"] = None
        u["matching"] = sorted(
            t for t in terms
            if posts.get(t) and rid in posts[t]
        )
        if rescue_replayed is not None:
            u["rescued"] = rid in rescue_replayed
        elif int(rescue.get("produced") or 0) == 0:
            # the lane's own produced count proves the unit was not
            # rescue-admitted regardless of replay fidelity
            u["rescued"] = False
        else:
            u["rescued"] = None   # fired, produced>0, replay failed
        u["readable"] = (
            None if readable is None else rid in readable
        )
        u["admitted"] = u["unit_id"] in admitted_uids
        u["in_items"] = u["unit_id"] in item_uids
        u["pack"] = item_packs.get(u["unit_id"])

    ev: Dict[str, Any] = {
        "lane_present": lane is not None,
        "explain_present": explain is not None,
        "lane_status": lane_status,
        "lane_reason": lane_reason,
        "stats": stats,
        "terms": terms,
        "terms_source": terms_source,
        "terms_note": terms_note,
        "idents": idents,
        "content_flags": content or "unavailable",
        "nominating": nominating,
        "gated_terms": sorted(gated_terms),
        "dropped_terms": sorted(dropped),
        "post_pool": (explain or {}).get("post_pool"),
        "gold_units": units,
        "unmapped_refs": unmapped,
        "rescue_replay": (
            "ok" if rescue_replayed is not None
            else ("not_fired" if not rescue.get("fired") else "failed")
        ),
        "recheck_dropped": (
            ((coverage or {}).get("security") or {})
            .get("eligibility_recheck_dropped")
        ),
        "elig_state": elig_state,
        "surfaced": list(row.get("surfaced") or ()),
        "lane_candidate_uids": sorted(admitted_uids),
        "item_uids": sorted(item_uids),
    }
    return ev


def _evidence_row(
    ev: Mapping[str, Any],
    letter: str,
    detail: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project assembled + classified evidence into the report row's
    ``evidence`` block — ids, terms, counts, statuses only."""
    stats = ev.get("stats") or {}
    cov = stats.get("coverage_lexical") or {}
    gate = cov.get("df_gate") or {}
    rescue = cov.get("rescue") or {}
    dropped_matching: List[Dict[str, Any]] = []
    gated_matching: List[Dict[str, Any]] = []
    gated = set(ev.get("gated_terms") or ())
    dropped = {str(e.get("term")): e
               for e in (stats.get("nomination_dropped") or ())
               if isinstance(e, Mapping)}
    nom_matching: set = set()
    for u in ev.get("gold_units") or ():
        for t in u.get("matching") or ():
            if t in set(ev.get("nominating") or ()):
                nom_matching.add(t)
    for t in sorted(nom_matching):
        if t in gated:
            df = next((e.get("df") for e in (gate.get("gated") or ())
                       if isinstance(e, Mapping) and str(e.get("term")) == t),
                      None)
            gated_matching.append({"term": t, "df": df})
        if t in dropped:
            e = dropped[t]
            dropped_matching.append({
                "term": t,
                "reason": e.get("reason"),
                "df": e.get("df"),
            })
    overlap_all = sorted({
        t for u in (ev.get("gold_units") or ())
        for t in (u.get("matching") or ())
    })
    out: Dict[str, Any] = {
        "lane": {
            "present": ev.get("lane_present"),
            "status": ev.get("lane_status"),
            "reason": ev.get("lane_reason"),
            "examined": (ev.get("stats") or {}).get("examined"),
        },
        "terms": {
            "count": len(ev.get("terms") or ()),
            "source": ev.get("terms_source"),
            "note": ev.get("terms_note"),
            "nominating_matching": sorted(nom_matching),
            "overlap_all": overlap_all,
            "content_flags": (
                "ok" if isinstance(ev.get("content_flags"), Mapping)
                else "unavailable"
            ),
        },
        "df_gate": {
            "floor": gate.get("floor"),
            "gated": [dict(e) for e in (gate.get("gated") or ())
                      if isinstance(e, Mapping)],
            "exempt": [dict(e) for e in (gate.get("exempt") or ())
                       if isinstance(e, Mapping)],
            "gated_matching": gated_matching,
        },
        "nomination": {
            "budget": stats.get("nominate_terms_max"),
            "eligible_terms": stats.get("nomination_eligible_terms"),
            "nominated_terms": stats.get("nominated_terms"),
            "dropped_matching": dropped_matching,
        },
        "rescue": {
            "fired": rescue.get("fired"),
            "terms": list(rescue.get("terms") or ()),
            "rows": rescue.get("rows"),
            "produced": rescue.get("produced"),
            "gold_rescued": any(
                u.get("rescued") for u in (ev.get("gold_units") or ())
            ),
            "replay": ev.get("rescue_replay"),
        },
        "cap": {
            "cap": stats.get("cap"),
            "overflow": stats.get("overflow"),
            "cap_truncated": stats.get("cap_truncated"),
            "scored": stats.get("scored"),
            "nominated": stats.get("nominated"),
            "scored_docs": stats.get("scored_docs"),
        },
        "deadline": {"phase": detail.get("deadline_phase")},
        "eligibility": {
            "state": ev.get("elig_state"),
            "via": stats.get("eligible_via"),
            "n_eligible": stats.get("n_eligible"),
            "n_universe": stats.get("n_universe"),
            "denied_units": [
                u.get("unit_id") for u in (ev.get("gold_units") or ())
                if u.get("eligible") is False
            ],
            "recheck_dropped": ev.get("recheck_dropped"),
        },
        "gold": {
            "refs": [u.get("ref") for u in (ev.get("gold_units") or ())],
            "unmapped_refs": list(ev.get("unmapped_refs") or ()),
            "n_units": len(ev.get("gold_units") or ()),
            "in_lane_candidates": any(
                u.get("admitted") for u in (ev.get("gold_units") or ())
            ),
            "in_scored_items": any(
                u.get("in_items") for u in (ev.get("gold_units") or ())
            ),
            "pack_decisions": sorted({
                str(u.get("pack")) for u in (ev.get("gold_units") or ())
                if u.get("pack") is not None
            }),
        },
        "post_pool": ev.get("post_pool"),
        "unit_stages": detail.get("unit_stages"),
        "note": detail.get("note"),
    }
    return out


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------


def run_lane_miss_forensic(
    corpus: Any,
    *,
    limit: int = 10,
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    policy_overrides: Optional[Mapping[str, Any]] = None,
    policy_doc: Optional[Mapping[str, Any]] = None,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """V8-07.01 — classify every answerable lane_miss question by the
    first stage that lost its gold.

    Runs the real ``VerbatimArm`` write/read path with forced explain
    capture, then replays the recorded lane evidence plus a bounded
    read-only store probe.  ``limit`` is the delivery cut used for the
    measured call (the surfaced-pool probe stays the arm's own).
    """
    views = task_views(corpus)
    ref_session = ref_sessions(corpus)
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "constants_tag": CONSTANTS_TAG,
        "tool": "lane_miss_forensic",
        "requirement": REQUIREMENT,
        "determinism_exempt": list(DETERMINISM_EXEMPT),
        "dataset": dataset_block(corpus, views),
        "limit": int(limit),
        "status": "executed",
        "not_run": [],
        "manifest": {
            "arm_class": "eval.v7.forensics.ForensicVerbatimArm",
            "arm_kwargs": dict(arm_kwargs or {}),
            "policy_overrides": _jsonable(policy_overrides),
            "policy_doc": _jsonable(policy_doc),
            "shared_store": True,
            "explain_forced": True,
            "environment": env_block(),
            "evidence_policy": (
                "ids, query-term strings, counts and status keys only — "
                "no payload text, no gold document text, no explain "
                "blobs"
            ),
            "classes_legend": dict(CLASS_LABELS),
        },
        "population": {
            "definition": (
                "answerable tasks whose gold was indexed but no lane "
                "surfaced it into the observable pool — attribution "
                "miss_stage == 'lane_miss' (withheld 'abstain' labels "
                "enter via their recorded stage)"
            ),
            "n_tasks": len(views),
            "n_answerable": sum(
                1 for v in views if v.answerable and v.gold_item),
            "n_lane_miss": 0,
            "n_classified": 0,
        },
        "classes": {c: 0 for c in CLASS_ORDER},
        "class_labels": dict(CLASS_LABELS),
        "eligibility": {
            "probe": ("make_eligible(conn, store, scope_id, generation, "
                      "principal_id=arm owner, purpose='recall') — the "
                      "predicate the facade installs on ctx.eligible"),
            "state": "unmeasured",
            "class_f_rows": 0,
        },
        "defects": {
            "eligibility_losses": 0,
            "impossible_states": [],
            "unverifiable": [],
        },
        "questions": [],
    }

    arm = ForensicVerbatimArm(
        name="verbatim",
        full_explain=True,
        policy_overrides=dict(policy_overrides or {}),
        **dict(arm_kwargs or {}),
    )
    t0 = time.time()
    try:
        try:
            report["manifest"]["ingest"] = arm.ingest(corpus)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "not_run"
            report["not_run"].append(
                f"ingest failed: {type(exc).__name__}: {exc}")
            write_report(report, out)
            return report

        patch = PolicyPatch(policy_doc=policy_doc) \
            if policy_doc is not None else None
        spec = ArmSpec(label="baseline", patch=patch)
        res = run_spec(views, arm, spec, k=int(limit),
                       ref_session=ref_session)
        report["manifest"]["patch"] = res.get("patch")
        if res.get("error"):
            report["status"] = "not_run"
            report["not_run"].append(f"query loop: {res['error']}")
            write_report(report, out)
            return report
        rows_by_id = res.get("rows") or {}
        explains = res.get("explains") or {}
        coverages = res.get("coverages") or {}
        indexed = set(arm.indexed_refs())

        # ---- probe context: one pinned snapshot for every question ----
        probe = _LaneProbe(arm)
        conn, err = probe.open()
        if conn is None:
            report["eligibility"]["state"] = "unavailable"
            report["not_run"].append(f"store probe: {err}")
        else:
            report["eligibility"]["state"] = (
                "verified" if probe.elig_ids is not None
                else "unverifiable"
            )
            report["eligibility"]["mode"] = probe.elig_stats.get("mode")
            report["eligibility"]["n_universe_units"] = len(probe.uni_rows)
            if probe.elig_ids is not None:
                report["eligibility"]["n_eligible_units"] = \
                    len(probe.elig_ids)
            if probe.elig_error:
                report["eligibility"]["reason"] = probe.elig_error
            report["eligibility"]["generation"] = probe.generation

        rows: List[Dict[str, Any]] = []
        counts = {c: 0 for c in CLASS_ORDER}
        try:
            for tv in views:
                if not tv.answerable or not tv.gold_item:
                    continue
                row = rows_by_id.get(tv.task_id) or {}
                detail = attribute_detail(tv, row, indexed=indexed,
                                          k=int(limit))
                stage = detail.get("miss_stage")
                if stage != "lane_miss" and \
                        detail.get("attribution") != "lane_miss":
                    continue
                if conn is not None:
                    ev = _assemble_evidence(
                        probe, tv, row,
                        explains.get(tv.task_id),
                        coverages.get(tv.task_id))
                else:
                    ev = {
                        "lane_present": False, "explain_present": False,
                        "lane_status": None, "lane_reason": None,
                        "stats": {}, "terms": [], "idents": [],
                        "nominating": [], "post_pool": None,
                        "gold_units": [], "unmapped_refs": [],
                        "elig_state": "unverifiable",
                        "probe_error": err,
                        "surfaced": list(row.get("surfaced") or ()),
                    }
                letter, cdetail = classify_lane_miss(ev)
                counts[letter] += 1
                evidence = _evidence_row(ev, letter, cdetail)
                rows.append({
                    "task_id": tv.task_id,
                    "conv_id": tv.group_id,
                    "category": tv.category,
                    "category_id": category_id_of(tv.raw),
                    "attribution": detail.get("attribution"),
                    "miss_stage": stage,
                    "withheld": detail.get("withheld"),
                    "class": letter,
                    "class_label": CLASS_LABELS[letter],
                    "evidence": evidence,
                })
                if letter == "f":
                    report["defects"]["eligibility_losses"] += 1
                if cdetail.get("note") in (
                        "nominated_scored_unadmitted",
                        "df_gate_block_absent", "lane_block_absent",
                        "nomination_record_absent", "scoring_incomplete",
                        "ordered_membership_unverifiable",
                        "admitted_not_scored"):
                    report["defects"]["impossible_states"].append(
                        tv.task_id)
                if ev.get("elig_state") == "unverifiable":
                    report["defects"]["unverifiable"].append(tv.task_id)
        finally:
            probe.close()

        report["population"]["n_lane_miss"] = len(rows)
        report["population"]["n_classified"] = len(rows)
        report["classes"] = counts
        report["eligibility"]["class_f_rows"] = counts["f"]
        report["questions"] = rows
        report["wall_ms"] = round((time.time() - t0) * 1000.0, 1)
    finally:
        arm.close()
    write_report(report, out)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.forensics.lane_miss",
        description="V8-07.01 lane-miss forensic — first-loss class per "
                    "answerable lane_miss question",
    )
    ap.add_argument("--dataset", required=True,
                    help="dataset-registry id (see eval.v7.corpora)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=10,
                    help="delivery cut k for the measured query")
    ap.add_argument("--timeout-ms", type=float, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--settle-timeout", type=float, default=None)
    ap.add_argument("--policy-overrides", default=None,
                    help="JSON object — arm policy_overrides")
    ap.add_argument("--policy-doc", default=None,
                    help="JSON policy document (lanes/params/nominate_*)")
    ap.add_argument("--out", default="eval/v8/lane_miss_forensic.json")
    args = ap.parse_args(argv)

    from ..corpora import load_corpus

    corpus = load_corpus(args.dataset, args.split)
    rep = run_lane_miss_forensic(
        corpus,
        limit=args.limit,
        arm_kwargs=arm_report_kwargs(
            timeout_ms=args.timeout_ms,
            pool_limit=args.pool_limit,
            settle_timeout_s=args.settle_timeout,
        ),
        policy_overrides=(
            json.loads(args.policy_overrides)
            if args.policy_overrides else None
        ),
        policy_doc=(
            json.loads(args.policy_doc) if args.policy_doc else None
        ),
        out=args.out,
    )
    classes = rep.get("classes") or {}
    pop = rep.get("population") or {}
    print(
        f"lane_miss={pop.get('n_lane_miss')}"
        f"/{pop.get('n_answerable')} "
        + " ".join(f"{c}={classes.get(c, 0)}" for c in CLASS_ORDER)
        + f" status={rep.get('status')}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "CLASS_LABELS",
    "CLASS_ORDER",
    "REQUIREMENT",
    "SCHEMA",
    "classify_lane_miss",
    "run_lane_miss_forensic",
]


if __name__ == "__main__":
    raise SystemExit(main())
