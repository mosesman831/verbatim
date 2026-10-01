"""Offline AMB proxy — SPEC_V8_5 §4 (V85-04.01/04.02/04.04).

The fast, no-LLM loop that answers the diagnostic question: *does the
gold evidence actually reach the delivered context?*

Pipeline:

* Load LoCoMo through the Track R loader + split
  (``eval.v7.corpora.load_corpus("locomo", split=…)`` — the same five-
  conversation dev split, never redefined here).
* Build AMB-shaped session ``Document`` dicts exactly as the pinned AMB
  LoCoMo loader does (``memory_bench/datasets/locomo.py`` — code-read
  citation in the manifest): ``id`` = ``{sample_id}_{session_N}``,
  ``content`` = ``json.dumps`` of the cleaned turn list, ``user_id`` =
  ``sample_id``, ``timestamp`` = the session's date ISO-8601 UTC,
  ``context`` = ``"Conversation between A and B (session_N of id)"``.
* Drive the **real** provider v2 (``eval.amb.provider.VerbatimAMBProvider``)
  — ``prepare`` → ``ingest`` → per-question ``retrieve``; retrieval
  latency comes from the provider's own ``query_records``.
* For GEIC@B: replay the question's rank-ordered hit stream through the
  V85-02.04 delivery rule at every budget — **the provider's own
  ``_expand`` when it is exposed** (the same code the real retrieve ran;
  ``expansion_path`` records which ran), else the shared parity
  implementation ``eval.v7.arms.geic_expand`` that Track R's
  ``session_messages`` arm uses.  Delivered positions map back to
  ``dia_id`` through the provider's persisted ``SessionIndex``
  (``unit-*.sessions.json``) — engine internals stay out of the loop.

Scores GEIC-any / GEIC-all at B ∈ {1K, 2K, 4.5K, 9K, unbounded} plus
context tokens and retrieve ms, reported overall, by LoCoMo category,
and by AMB label (open-domain=4, single-hop=1, temporal=2, multi-hop=3).

Usage::

    VERBATIM_EVAL_LOCOMO=1 python -m eval.v8.amb_proxy --split dev
    VERBATIM_EVAL_LOCOMO=1 python -m eval.v8.amb_proxy --split full

Artifacts land in ``eval/v8/results/`` (JSON + Markdown).  No network,
no LLM — the reader is never invoked; GEIC is a ceiling, not accuracy
(§9).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "amb_proxy/v85-04"

#: AMB label crosswalk (SPEC_V8_5 §0): LoCoMo numeric category id → AMB
#: category label.  Category 5 (adversarial) has no AMB label — it is
#: reported under ``adversarial`` and excluded from answerable scopes.
AMB_LABELS: Dict[int, str] = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

#: Budget sweep (V85-04.01); ``None`` = unbounded.
DEFAULT_BUDGETS: Tuple[Optional[int], ...] = (1000, 2000, 4500, 9000, None)

_LTS = "%I:%M %p on %d %B, %Y"   # locomo "1:56 pm on 8 May, 2023"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_SESSION_KEY_PREFIX = "session_"


# ---------------------------------------------------------------------------
# LoCoMo raw → AMB-shaped session documents
# ---------------------------------------------------------------------------


def _session_n(name: str) -> int:
    """``session_12`` → 12; non-matching keys sort last."""
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 1 << 30


def parse_session_dt(value: Any) -> Optional[_dt.datetime]:
    """LoCoMo ``session_N_date_time`` → UTC datetime (naive treated as
    UTC — the AMB loader's convention, locomo.py:205-279)."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        dt = value
    else:
        s = str(value).strip()
        dt = None
        for fmt in (_LTS, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = _dt.datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            try:
                dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def _iso(dt: Optional[_dt.datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00") if dt is not None else None


def session_documents(conv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One AMB session ``Document``-shaped dict per LoCoMo session —
    the pinned loader's shape (``content`` = the JSON turn list; turns
    keep their raw keys: ``speaker``/``dia_id``/``text``/``img_url``/
    ``blip_caption``)."""
    sid = str(conv.get("sample_id") or "unknown")
    conv_obj = conv.get("conversation") or {}
    speaker_a = conv_obj.get("speaker_a") or "speaker_a"
    speaker_b = conv_obj.get("speaker_b") or "speaker_b"
    docs: List[Dict[str, Any]] = []
    names = sorted(
        (k for k in conv_obj
         if isinstance(k, str) and k.startswith(_SESSION_KEY_PREFIX)
         and not k.endswith("_date_time")),
        key=_session_n,
    )
    for sname in names:
        raw_turns = conv_obj.get(sname) or []
        turns = [
            dict(t) for t in raw_turns
            if isinstance(t, Mapping) and t.get("dia_id")
        ]
        ts = parse_session_dt(conv_obj.get(f"{sname}_date_time"))
        docs.append({
            "id": f"{sid}_{sname}",
            "content": json.dumps(turns, ensure_ascii=False),
            "user_id": sid,
            "timestamp": _iso(ts),
            "context": (
                f"Conversation between {speaker_a} and {speaker_b} "
                f"({sname} of {sid})"
            ),
        })
    return docs


def question_timestamp(conv: Mapping[str, Any]) -> Optional[str]:
    """The AMB LoCoMo ``query_timestamp`` — the conversation's last
    session date, ISO-8601 UTC (locomo.py:205-279)."""
    conv_obj = conv.get("conversation") or {}
    dts = [
        parse_session_dt(v)
        for k, v in conv_obj.items()
        if isinstance(k, str) and k.endswith("_date_time")
    ]
    dts = [d for d in dts if d is not None]
    return _iso(max(dts)) if dts else None


def _gold_dias(task: Any, sample_id: str) -> Tuple[List[str], int]:
    """(resolvable gold dia_ids, raw gold count) — ``evidence_ids`` are
    ``{sample_id}:{dia}`` refs the loader already verified against the
    corpus; ``metadata.raw_evidence`` keeps the pre-normalization list
    for the honest ``n_gold_raw`` count."""
    eids = list(getattr(task, "evidence_ids", None) or ())
    prefix = f"{sample_id}:"
    dias = sorted({
        e[len(prefix):] for e in eids
        if isinstance(e, str) and e.startswith(prefix)
    })
    meta = getattr(task, "metadata", None) or {}
    raw = meta.get("raw_evidence")
    n_raw = len(raw) if isinstance(raw, (list, tuple)) else len(dias)
    return dias, n_raw


# ---------------------------------------------------------------------------
# provider access — documented surface + disclosed internals
# ---------------------------------------------------------------------------


def _provider_search_kwargs(provider: Any, mem: Any, query_timestamp: Any) -> Dict[str, Any]:
    """The provider's own search kwargs (limit=L, timeout, as_of when
    the facade accepts it) — falls back to the documented contract."""
    fn = getattr(provider, "_search_kwargs", None)
    if callable(fn):
        try:
            kwargs, _rec = fn(mem, query_timestamp)
            return dict(kwargs)
        except Exception:  # noqa: BLE001 — contract drift, degrade
            pass
    kwargs: Dict[str, Any] = {"limit": int(
        getattr(provider, "search_limit", 64) or 64)}
    try:
        import inspect
        if "as_of" in inspect.signature(mem.search).parameters and (
            query_timestamp is not None
        ):
            kwargs["as_of"] = query_timestamp
    except Exception:  # noqa: BLE001
        pass
    return kwargs


def _provider_index(provider: Any, unit_key: str, store_dir: Path) -> Any:
    """The bank's ``SessionIndex`` — in-memory map first, then the
    persisted ``unit-*.sessions.json`` sidecar (provider contract for
    resume runs)."""
    indexes = getattr(provider, "_indexes", None)
    if isinstance(indexes, Mapping) and unit_key in indexes:
        return indexes[unit_key]
    try:
        from eval.amb._turns import SessionIndex
        from eval.amb.provider import _index_filename

        path = Path(store_dir) / _index_filename(unit_key)
        if path.is_file():
            return SessionIndex.load(path)
    except Exception:  # noqa: BLE001
        pass
    return None


def _unit_key(user_id: Any) -> str:
    try:
        from eval.amb.provider import _unit_key as _f

        return _f(user_id)
    except Exception:  # noqa: BLE001 — the documented convention
        return str(user_id) if user_id not in (None, "") else "_shared"


def _resolve_via_index(
    hit: Any,
    index: Any,
    unit_rows: Mapping[str, Mapping[str, Any]],
    claim_source: Callable[[str], Optional[str]],
) -> Optional[Dict[str, Any]]:
    """``provider._resolve_hit`` replayed over the SessionIndex's public
    methods — used when the provider's internal resolver isn't exposed.
    Returns ``{"doc_id", "ordinals"}`` or ``None`` (stray)."""
    from eval.v7.arms import _parse_hit_object_ref

    parsed = _parse_hit_object_ref(getattr(hit, "object_ref", ""))
    quote = getattr(hit, "quote", "") or ""
    if isinstance(quote, (bytes, bytearray)):
        quote = bytes(quote).decode("utf-8", "replace")

    if parsed is not None and parsed[0] == "unit":
        got = index.resolve_unit(parsed[1])
        if got is not None:
            doc_id, ordinal = got
            return {"doc_id": doc_id, "ordinals": {int(ordinal)}}
        row = unit_rows.get(parsed[1]) or {}
        src = row.get("source_id")
        doc_id = index.doc_for_source(src)
        if doc_id is not None:
            covered = index.covered_ordinals(
                doc_id, row.get("byte_start"), row.get("byte_end"),
                source_id=src)
            if not covered and row.get("kind") == "turn":
                by_seq = index.ordinal_by_seq(
                    doc_id, row.get("seq"), source_id=src)
                covered = [by_seq] if by_seq is not None else []
            if not covered:
                by_q = index.ordinal_by_quote(doc_id, quote)
                covered = [by_q] if by_q is not None else [0]
            return {"doc_id": doc_id, "ordinals": set(covered)}
        # unit row unresolved → stray (falls through to source check
        # only when parsed is a bare-source ref, matching the provider)
        return None

    if parsed is not None and parsed[0] == "claim":
        sid = claim_source(parsed[1])
        doc_id = index.doc_for_source(sid) if sid else None
        if doc_id is not None:
            by_q = index.ordinal_by_quote(doc_id, quote)
            return {"doc_id": doc_id,
                    "ordinals": {by_q if by_q is not None else 0}}
        return None

    sid = parsed[1] if parsed is not None else None
    if not sid:
        sid = str(getattr(hit, "memory_id", "") or "") or None
    doc_id = index.doc_for_source(sid) if sid else None
    if doc_id is None:
        return None
    by_q = index.ordinal_by_quote(doc_id, quote)
    return {"doc_id": doc_id,
            "ordinals": {by_q if by_q is not None else 0}}


def _index_sessions(index: Any, meter: Callable[[str], int]) -> Dict[str, Any]:
    """``{doc_id: {"costs", "header"}}`` — metered per-position line
    costs + once-per-session header, the ``geic_expand`` session table.
    Line/header strings are the provider's ``render_turn_line`` /
    ``render_header`` (V85-02.05)."""
    from eval.amb._turns import render_header, render_turn_line

    sessions: Dict[str, Any] = {}
    for doc_id, sess in (getattr(index, "sessions", None) or {}).items():
        turns = sess.get("turns") or []
        sessions[str(doc_id)] = {
            "costs": [meter(render_turn_line(t)) for t in turns],
            "header": meter(render_header(sess)),
            "turns": turns,
        }
    return sessions


def _expand_via_provider(
    provider: Any, index: Any, hits: List[Any],
    resolved: List[Any], budget: Optional[int],
) -> Tuple[Dict[str, set], int]:
    """The provider's own ``_expand`` replayed at ``budget`` — the real
    delivery code, with ``token_budget`` swapped and restored."""
    original = provider.token_budget
    provider.token_budget = budget
    try:
        ex = provider._expand(index, hits, resolved)
    finally:
        provider.token_budget = original
    delivered = {
        str(doc_id): set(ords)
        for doc_id, ords in (ex.get("delivered") or {}).items()
    }
    return delivered, int(ex.get("spent") or 0)


def _expand_via_geic(
    provider: Any,
    resolved: List[Any],
    hits: List[Any],
    sessions: Mapping[str, Any],
    window: int,
    budget: Optional[int],
    meter: Callable[[str], int],
) -> Tuple[Dict[str, set], int]:
    """The shared parity implementation (``eval.v7.arms.geic_expand`` —
    Track R's ``session_messages`` arm runs the same function)."""
    from eval.v7.arms import (
        GEIC_STRAY_MARGIN, GEIC_STRAY_PREFIX, geic_expand,
    )

    blocks: List[Tuple[Any, ...]] = []
    for hit, r in zip(hits, resolved):
        if r is None or r.get("doc_id") is None or (
            sessions.get(str(r["doc_id"])) is None
        ):
            quote = getattr(hit, "quote", "") or ""
            if isinstance(quote, (bytes, bytearray)):
                quote = bytes(quote).decode("utf-8", "replace")
            cost = meter(GEIC_STRAY_PREFIX + str(quote)) + GEIC_STRAY_MARGIN
            blocks.append((None, None, cost))
            continue
        doc_id = str(r["doc_id"])
        blocks.append((doc_id, set(r.get("ordinals") or ()), 0))
    delivered, spent = geic_expand(
        blocks, sessions, window=window, budget=budget)
    return {k: set(v) for k, v in delivered.items()}, spent


def _delivered_dias(
    delivered: Mapping[str, Iterable[int]], index: Any
) -> List[str]:
    """``{doc_id: ordinals}`` → sorted unique ``dia_id`` list — turns
    without a dia_id (or a missing session) contribute nothing, never a
    fabricated id."""
    out: set = set()
    for doc_id, ords in delivered.items():
        sess = index.session(doc_id) if hasattr(index, "session") else None
        if sess is None:
            continue
        turns = sess.get("turns") or []
        for o in ords:
            if 0 <= int(o) < len(turns):
                dia = turns[int(o)].get("dia_id")
                if dia:
                    out.add(str(dia))
    return sorted(out)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _budget_key(b: Optional[int]) -> str:
    return "unbounded" if b is None else str(int(b))


def _score_budgets(
    *,
    budgets: Sequence[Optional[int]],
    expand: Callable[[Optional[int]], Tuple[Dict[str, set], int]],
    index: Any,
    gold: set,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for b in budgets:
        delivered, spent = expand(b)
        dias = _delivered_dias(delivered, index)
        hit = len(gold & set(dias))
        n_positions = sum(len(v) for v in delivered.values())
        out[_budget_key(b)] = {
            "geic_any": bool(hit),
            "geic_all": bool(gold) and hit == len(gold),
            "geic_prop": (hit / len(gold)) if gold else None,
            "n_gold_hit": hit,
            "delivered_dias": dias,
            "n_sessions": len(delivered),
            "n_turns": n_positions,
            "tokens": spent,
        }
    return out


def _agg(records: List[Dict[str, Any]], bkey: str) -> Dict[str, Any]:
    """geic@B aggregate over one slice — answerable questions with
    resolvable gold (T3's denominator)."""
    scored = [r for r in records if r["answerable"] and r["n_gold"]]
    out: Dict[str, Any] = {
        "n": len(scored),
        "n_all_gold": sum(1 for r in records if r["n_gold"]),
    }
    if not scored:
        out.update({
            "geic_any": None, "geic_all": None, "geic_prop": None,
            "turns_mean": None, "tokens_mean": None, "tokens_p95": None,
        })
        return out
    vals = [r["budgets"].get(bkey) or {} for r in scored]
    anys = sum(1.0 for v in vals if v.get("geic_any")) / len(vals)
    alls = sum(1.0 for v in vals if v.get("geic_all")) / len(vals)
    props = [float(v.get("geic_prop") or 0.0) for v in vals]
    turns = [float(v.get("n_turns") or 0) for v in vals]
    toks = sorted(float(v.get("tokens") or 0) for v in vals)
    p95 = toks[min(len(toks) - 1, int(round(0.95 * (len(toks) - 1))))] \
        if toks else None
    out.update({
        "geic_any": anys,
        "geic_all": alls,
        "geic_prop": sum(props) / len(props),
        "turns_mean": sum(turns) / len(turns),
        "tokens_mean": sum(toks) / len(toks) if toks else None,
        "tokens_p95": p95,
    })
    return out


def _latency_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = sorted(float(r.get("retrieve_ms") or 0.0) for r in records)
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "mean": None}

    def _pct(q: float) -> float:
        return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]

    return {
        "n": len(vals),
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "mean": sum(vals) / len(vals),
    }


def run_proxy(
    corpus: Any,
    conversations: List[Mapping[str, Any]],
    provider: Any,
    *,
    budgets: Sequence[Optional[int]] = DEFAULT_BUDGETS,
    k: int = 10,
    window: Optional[int] = None,
    search_limit: Optional[int] = None,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """Drive the real provider end-to-end and score GEIC@B per question.

    ``provider`` is a prepared (``prepare()`` done) provider v2 instance;
    ingest happens here so the ingest record lands in the report.
    ``max_tasks`` truncates the (task_id-sorted) question list — a
    deterministic smoke cap, recorded in the run record.
    """
    from eval.v7.arms import UnitSpans, geic_meter

    budgets = tuple(budgets)
    tasks = list(getattr(corpus, "tasks", None) or [])
    tasks.sort(key=lambda t: str(getattr(t, "task_id", "")))
    n_tasks_total = len(tasks)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    # ---- ingest: one provider for the whole split; per-conversation
    # banks via user_id (AMB isolation convention).
    docs: List[Dict[str, Any]] = []
    conv_by_sid: Dict[str, Mapping[str, Any]] = {}
    for conv in conversations:
        sid = str(conv.get("sample_id") or "")
        conv_by_sid[sid] = conv
        docs.extend(session_documents(conv))
    ingest_rep = provider.ingest(docs)

    meter_name = getattr(provider, "_meter_name", None) or geic_meter()[0]
    meter = getattr(provider, "_meter", None) or geic_meter()[1]
    window = (
        int(window) if window is not None
        else int(getattr(provider, "neighbor_w", 1) or 1)
    )
    limit = (
        int(search_limit) if search_limit is not None
        else int(getattr(provider, "search_limit", 64) or 64)
    )

    # Expansion path — prefer the provider's real ``_expand`` replayed
    # per budget; fall back to the shared ``geic_expand`` (Track R's
    # parity implementation) when internals aren't exposed.
    has_provider_expand = all(
        callable(getattr(provider, m, None))
        for m in ("_expand", "_resolve_hit")
    )
    expansion_path = (
        "provider._expand replayed per budget"
        if has_provider_expand
        else "eval.v7.arms.geic_expand (provider internals absent)"
    )
    parity = {"checked": 0, "matched": 0, "mismatched": []}
    units_cache: Dict[str, Any] = {}

    records: List[Dict[str, Any]] = []
    n_retrieve_errors = 0
    for task in tasks:
        sid = str(getattr(task, "group_id", None) or "")
        query = str(getattr(task, "query", "") or "")
        conv = conv_by_sid.get(sid)
        qt = question_timestamp(conv) if conv else None
        gold, n_raw = _gold_dias(task, sid)
        meta = getattr(task, "metadata", None) or {}
        cat_id = meta.get("category_id")
        try:
            cat_id = int(cat_id) if cat_id is not None else None
        except (TypeError, ValueError):
            cat_id = None
        amb_label = AMB_LABELS.get(cat_id, f"cat{cat_id}")

        rec: Dict[str, Any] = {
            "task_id": str(getattr(task, "task_id", "")),
            "sample_id": sid,
            "query": query,
            "category": str(getattr(task, "category", "unknown")),
            "category_id": cat_id,
            "amb_label": amb_label,
            "answerable": bool(getattr(task, "answerable", True)),
            "gold_dias": list(gold),
            "n_gold": len(gold),
            "n_gold_raw": n_raw,
            "query_timestamp": qt,
        }

        bkey = _unit_key(sid)
        mem = (provider.banks() or {}).get(bkey)
        index = _provider_index(
            provider, bkey,
            getattr(provider, "_store_dir", None) or Path("."))

        # -- the real measured retrieve (fills provider.query_records)
        try:
            before = len(provider.query_records)
            _docs, _raw = provider.retrieve(
                query, k=int(k), user_id=sid, query_timestamp=qt)
            qrecs = provider.query_records
            qrec = qrecs[-1] if len(qrecs) > before else {}
            rec["retrieve_ms"] = qrec.get("retrieve_ms")
            rec["provider"] = {
                "status": qrec.get("status"),
                "n_hits": qrec.get("n_hits"),
                "n_docs": qrec.get("n_docs"),
                "doc_ids": qrec.get("doc_ids"),
                "est_context_tokens": qrec.get("est_context_tokens"),
                "expansion": qrec.get("expansion"),
                "resolution": qrec.get("resolution"),
                "token_budget": qrec.get("token_budget"),
                "warnings": qrec.get("warnings"),
            }
        except Exception as exc:  # noqa: BLE001 — recorded, scored 0
            n_retrieve_errors += 1
            rec["retrieve_ms"] = None
            rec["provider"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rec["budgets"] = {
                _budget_key(b): {
                    "geic_any": False,
                    "geic_all": False,
                    "geic_prop": 0.0 if gold else None,
                    "n_gold_hit": 0,
                    "delivered_dias": [],
                    "n_sessions": 0,
                    "n_turns": 0,
                    "tokens": 0,
                    "error": True,
                }
                for b in budgets
            }
            records.append(rec)
            continue

        # -- GEIC replay: the same hit stream through the V85-02.04
        # delivery rule at every budget (unmeasured — the proxy loop).
        hits: List[Any] = []
        resolved: List[Any] = []
        if mem is not None and index is not None:
            try:
                kwargs = _provider_search_kwargs(provider, mem, qt)
                res = mem.search(query, **kwargs)
                hits = list(getattr(res, "items", None) or [])
                if has_provider_expand:
                    resolved = [
                        provider._resolve_hit(mem, index, h)
                        for h in hits
                    ]
                else:
                    if bkey not in units_cache:
                        units_cache[bkey] = UnitSpans(mem._store)
                    units = units_cache[bkey]
                    uids = []
                    from eval.v7.arms import _parse_hit_object_ref

                    for h in hits:
                        p = _parse_hit_object_ref(
                            getattr(h, "object_ref", ""))
                        if p is not None and p[0] == "unit":
                            uids.append(p[1])
                    rows = units.unit_rows(uids)

                    def _claim_src(cid: str, _u=units) -> Optional[str]:
                        for sid_, _bs, _be in _u.claim_ranges(
                            [cid]).get(cid, ()):
                            return sid_
                        return None

                    resolved = [
                        _resolve_via_index(h, index, rows, _claim_src)
                        for h in hits
                    ]
            except Exception as exc:  # noqa: BLE001 — recorded honestly
                rec.setdefault("warnings", []).append(
                    f"geic_replay: {type(exc).__name__}: {exc}")
                hits, resolved = [], []

        rec["n_hits"] = len(hits)
        rec["n_resolved"] = sum(
            1 for r in resolved if r is not None and r.get("doc_id"))

        if index is not None:
            sessions = _index_sessions(index, meter)

            def _expand(b: Optional[int]) -> Tuple[Dict[str, set], int]:
                if has_provider_expand:
                    return _expand_via_provider(
                        provider, index, hits, resolved, b)
                return _expand_via_geic(
                    provider, resolved, hits, sessions, window, b, meter)

            rec["budgets"] = _score_budgets(
                budgets=budgets, expand=_expand, index=index,
                gold=set(gold))

            # parity check: the shared implementation vs the provider's
            # expansion, at the provider's configured budget.
            if has_provider_expand:
                pb = getattr(provider, "token_budget", None)
                try:
                    p_del, p_spent = _expand_via_provider(
                        provider, index, hits, resolved, pb)
                    g_del, g_spent = _expand_via_geic(
                        provider, resolved, hits, sessions, window, pb,
                        meter)
                    parity["checked"] += 1
                    same = (
                        p_spent == g_spent
                        and {k: sorted(v) for k, v in p_del.items()}
                        == {k: sorted(v) for k, v in g_del.items()}
                    )
                    if same:
                        parity["matched"] += 1
                    elif len(parity["mismatched"]) < 10:
                        parity["mismatched"].append({
                            "task_id": rec["task_id"],
                            "provider_spent": p_spent,
                            "geic_spent": g_spent,
                        })
                except Exception:  # noqa: BLE001
                    parity["checked"] += 1
        else:
            rec["budgets"] = {
                _budget_key(b): {
                    "geic_any": False, "geic_all": False,
                    "geic_prop": 0.0 if gold else None,
                    "n_gold_hit": 0, "delivered_dias": [],
                    "n_sessions": 0, "n_turns": 0, "tokens": 0,
                    "error": "no_session_index",
                }
                for b in budgets
            }
        records.append(rec)

    # ---- aggregates ----------------------------------------------------
    bkeys = [_budget_key(b) for b in budgets]
    overall = {bk: _agg(records, bk) for bk in bkeys}
    cats = sorted({r["category"] for r in records})
    by_category = {
        c: {"category_id": next(
            (r["category_id"] for r in records
             if r["category"] == c and r["category_id"] is not None),
            None),
            "budgets": {
                bk: _agg([r for r in records if r["category"] == c], bk)
                for bk in bkeys}}
        for c in cats
    }
    labels = sorted({r["amb_label"] for r in records})
    by_amb_label = {
        lab: {"budgets": {
            bk: _agg([r for r in records if r["amb_label"] == lab], bk)
            for bk in bkeys}}
        for lab in labels
    }

    return {
        "records": records,
        "ingest": ingest_rep,
        "aggregate": {
            "scope": "answerable+gold",
            "n_tasks_total": n_tasks_total,
            "max_tasks": max_tasks,
            "overall": overall,
            "by_category": by_category,
            "by_amb_label": by_amb_label,
            "retrieve_ms": _latency_summary(records),
            "n_retrieve_errors": n_retrieve_errors,
        },
        "expansion": {
            "path": expansion_path,
            "parity": parity,
            "window": window,
            "search_limit": limit,
            "meter": meter_name,
        },
        "provider_records": list(provider.query_records),
    }


# ---------------------------------------------------------------------------
# manifest + report assembly
# ---------------------------------------------------------------------------


def _sha256_file(path: Any) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _manifest(
    corpus: Any,
    provider: Any,
    args: argparse.Namespace,
    conversations: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    try:
        from eval.v7.manifest import collect_environment, git_state

        git = git_state(str(_REPO_ROOT))
        env_block = collect_environment()
    except Exception:  # noqa: BLE001
        git, env_block = {"status": "unavailable"}, {}
    mf = {}
    try:
        mf = provider.manifest_fields()
    except Exception:  # noqa: BLE001 — older providers lack it
        mf = {
            "provider_revision": getattr(
                provider, "PROVIDER_VERSION", None)
            or getattr(provider, "provider_version", None),
        }
    src = getattr(corpus, "source_path", "") or ""
    return {
        "schema": SCHEMA,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "id": getattr(corpus, "dataset_id", None)
            or getattr(corpus, "name", None),
            "name": getattr(corpus, "name", None),
            "split": getattr(corpus, "split", None),
            "source_path": src,
            "sha256": _sha256_file(src) if src else None,
            "n_conversations": len(conversations),
            "n_items": len(getattr(corpus, "items", None) or ()),
            "n_tasks": len(getattr(corpus, "tasks", None) or ()),
        },
        "provider": mf,
        "engine": git,
        "run": {
            "budgets": [
                "unbounded" if b is None else int(b)
                for b in args.budgets_parsed
            ],
            "neighbor_w": args.window,
            "search_limit": args.search_limit,
            "k": args.k,
            "seed": args.seed,
            "split": args.split,
            "max_tasks": getattr(args, "max_tasks", None),
        },
        "environment": env_block,
        "env_flags": {
            "VERBATIM_EVAL_LOCOMO": bool(
                os.environ.get("VERBATIM_EVAL_LOCOMO")),
        },
        "determinism": (
            "single-threaded; store read-only during the query phase; "
            "identical input+config reproduces identical results aside "
            "from measured latency fields"
        ),
        "code_citations": {
            "doc_shape": (
                "memory_bench/datasets/locomo.py:281-331 via "
                "research/v8_final_pack/c1-amb-provider.md (doc.id "
                "{sample_id}_{session_N}; content=json.dumps(turns); "
                "context 'Conversation between A and B (session_N of "
                "sample_id)')"
            ),
            "query_timestamp": (
                "memory_bench/datasets/locomo.py:205-279 — last session "
                "date, ISO-8601 UTC"
            ),
            "expansion_rule": "SPEC_V8_5 §2 V85-02.04",
            "split": "eval.v7.dataset_registry SplitSpec(conversation, "
                     "dev_pct=40) via eval.v7.corpora.load_corpus",
        },
        "assumptions": [
            "provider interface: eval.amb.provider.VerbatimAMBProvider "
            "with prepare()/ingest()/retrieve()/query_records/"
            "manifest_fields(); _expand/_resolve_hit used when exposed "
            "(expansion_path records which ran)",
            "tokenizer: provider's meter (cl100k_base via tiktoken when "
            "installed, else the repo's tok/v1 estimator) — the meter "
            "name is recorded, never silently substituted",
            "gold = corpus evidence_ids (loader-resolvable) → dia_ids; "
            "raw unresolvable evidence ids counted in n_gold_raw only",
            "cat-5 questions are answerable=False (premise evidence, "
            "not answer support) — GEIC denominates answerable+gold",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Deterministic markdown render of the proxy report."""
    agg = report.get("aggregate") or {}
    man = report.get("manifest") or {}
    run = man.get("run") or {}
    lines: List[str] = []
    lines.append("# AMB offline proxy — GEIC@B (V85-04)")
    lines.append("")
    ds = man.get("dataset") or {}
    lines.append(
        f"- dataset `{ds.get('id')}` split `{ds.get('split')}` — "
        f"{ds.get('n_conversations')} conversations, "
        f"{ds.get('n_tasks')} tasks, sha256 `{(ds.get('sha256') or '')[:12]}`"
    )
    prov = man.get("provider") or {}
    lines.append(
        f"- provider `{prov.get('provider_revision')}` "
        f"engine `{(man.get('engine') or {}).get('sha')}`"
    )
    ex = report.get("expansion") or {}
    lines.append(
        f"- expansion `{ex.get('path')}` W_r={ex.get('window')} "
        f"L={ex.get('search_limit')} meter=`{ex.get('meter')}`"
    )
    par = ex.get("parity") or {}
    if par.get("checked"):
        lines.append(
            f"- parity provider._expand vs geic_expand: "
            f"{par.get('matched')}/{par.get('checked')} identical"
        )
    lat = agg.get("retrieve_ms") or {}
    lines.append(
        f"- retrieve_ms p50={lat.get('p50')} p95={lat.get('p95')} "
        f"mean={lat.get('mean')} n={lat.get('n')} "
        f"errors={agg.get('n_retrieve_errors')}"
    )
    lines.append("")

    def _f(x: Any, nd: int = 3) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    head = ["scope", "budget", "n", "geic_any", "geic_all", "geic_prop",
            "turns_mean", "tokens_mean", "tokens_p95"]

    def _table(rows: List[List[Any]]) -> None:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("| " + " | ".join("---" for _ in head) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(c) for c in r) + " |")

    rows = []
    for bk, g in (agg.get("overall") or {}).items():
        rows.append(["overall", bk, g.get("n"), _f(g.get("geic_any")),
                     _f(g.get("geic_all")), _f(g.get("geic_prop")),
                     _f(g.get("turns_mean"), 1),
                     _f(g.get("tokens_mean"), 1),
                     _f(g.get("tokens_p95"), 1)])
    lines.append("## Overall (answerable+gold)")
    lines.append("")
    _table(rows)
    lines.append("")

    lines.append("## By AMB label")
    lines.append("")
    rows = []
    for lab, blk in (agg.get("by_amb_label") or {}).items():
        for bk, g in (blk.get("budgets") or {}).items():
            rows.append([lab, bk, g.get("n"), _f(g.get("geic_any")),
                         _f(g.get("geic_all")), _f(g.get("geic_prop")),
                         _f(g.get("turns_mean"), 1),
                         _f(g.get("tokens_mean"), 1),
                         _f(g.get("tokens_p95"), 1)])
    _table(rows)
    lines.append("")

    lines.append("## By LoCoMo category")
    lines.append("")
    rows = []
    for cat, blk in (agg.get("by_category") or {}).items():
        cid = blk.get("category_id")
        for bk, g in (blk.get("budgets") or {}).items():
            rows.append([f"{cat} (cat {cid})", bk, g.get("n"),
                         _f(g.get("geic_any")), _f(g.get("geic_all")),
                         _f(g.get("geic_prop")),
                         _f(g.get("turns_mean"), 1),
                         _f(g.get("tokens_mean"), 1),
                         _f(g.get("tokens_p95"), 1)])
    _table(rows)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_budgets(raw: str) -> Tuple[Optional[int], ...]:
    out: List[Optional[int]] = []
    for tok in str(raw).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("none", "unbounded", "inf"):
            out.append(None)
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("--budgets parsed to an empty list")
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v8.amb_proxy",
        description="Offline AMB proxy — GEIC@B over the real provider "
                    "v2 (V85-04.01)",
    )
    ap.add_argument("--split", default="dev",
                    choices=("dev", "test", "full"),
                    help="registry split; 'full' = the whole dataset")
    ap.add_argument("--dataset", default="locomo",
                    help="registry dataset id (default: locomo)")
    ap.add_argument("--budgets", default="1000,2000,4500,9000,unbounded",
                    help="comma list; 'unbounded'/'none' = no cap")
    ap.add_argument("--window", type=int, default=None,
                    help="W_r neighbor window (default: provider's)")
    ap.add_argument("--search-limit", type=int, default=None,
                    help="L = search hit cap (default: provider's, 64)")
    ap.add_argument("--k", type=int, default=10,
                    help="AMB k recorded on retrieve (never obeyed, "
                         "V85-02.03; default 10)")
    ap.add_argument("--provider-budget", default="4500",
                    help="the provider's own token_budget for the "
                         "measured retrieve (default 4500; "
                         "'unbounded' allowed)")
    ap.add_argument("--workdir", default=None,
                    help="provider store dir (default: tempdir, deleted)")
    ap.add_argument("--keep-store", action="store_true",
                    help="keep the workdir store for forensics")
    ap.add_argument("--out", default=None,
                    help="JSON report path (default: "
                         "eval/v8/results/amb_proxy_<split>.json)")
    ap.add_argument("--md", default=None,
                    help="Markdown report path (default: alongside --out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="debug/smoke cap — first N tasks by task_id "
                         "(deterministic; the manifest records it)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        args.budgets_parsed = _parse_budgets(args.budgets)
        args.provider_budget_parsed = _parse_budgets(
            args.provider_budget)[0]
    except ValueError as exc:
        print(f"amb_proxy: {exc}", file=sys.stderr)
        return 2

    from eval.v7 import corpora as _corpora

    split = None if args.split == "full" else args.split
    try:
        corpus = _corpora.load_corpus(
            args.dataset, split=split,
            seed=args.seed if args.seed else None)
    except Exception as exc:  # noqa: BLE001 — honest exit, never silent
        print(
            f"amb_proxy: dataset {args.dataset!r} unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    src = getattr(corpus, "source_path", "") or ""
    if not src or not os.path.isfile(src):
        print(f"amb_proxy: corpus has no readable source_path {src!r}",
              file=sys.stderr)
        return 2
    with open(src, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    keep_sids = {
        str(getattr(i, "group_id", "") or "")
        for i in (getattr(corpus, "items", None) or ())
    }
    keep_sids |= {
        str(getattr(t, "group_id", "") or "")
        for t in (getattr(corpus, "tasks", None) or ())
    }
    keep_sids.discard("")
    conversations = [
        c for c in raw
        if str(c.get("sample_id") or "") in keep_sids
    ]
    if not conversations:
        print("amb_proxy: no conversations in split — nothing to run",
              file=sys.stderr)
        return 2

    from eval.amb.provider import VerbatimAMBProvider

    owns_dir = args.workdir is None
    workdir = Path(
        args.workdir or tempfile.mkdtemp(prefix="v85-amb-proxy-"))
    workdir.mkdir(parents=True, exist_ok=True)
    provider = VerbatimAMBProvider(
        store_dir=workdir,
        token_budget=args.provider_budget_parsed,
        neighbor_w=(args.window if args.window is not None else
                    __import__("eval.amb.provider", fromlist=["x"])._UNSET),
        search_limit=(args.search_limit if args.search_limit is not None
                      else __import__("eval.amb.provider",
                                      fromlist=["x"])._UNSET),
        doc_mode="pack",
    )
    t0 = time.perf_counter()
    try:
        provider.prepare(workdir, unit_ids=sorted(keep_sids), reset=True)
        result = run_proxy(
            corpus, conversations, provider,
            budgets=args.budgets_parsed, k=int(args.k),
            window=args.window, search_limit=args.search_limit,
            max_tasks=args.max_tasks,
        )
    finally:
        try:
            provider.cleanup()
        except Exception:  # noqa: BLE001
            pass
    wall_s = time.perf_counter() - t0
    # effective values back into the manifest — None args resolved to
    # the provider's own knobs inside run_proxy.
    args.window = result["expansion"]["window"]
    args.search_limit = result["expansion"]["search_limit"]

    report = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "wall_s": round(wall_s, 3),
        "manifest": _manifest(corpus, provider, args, conversations),
        "expansion": result["expansion"],
        "ingest": result["ingest"],
        "aggregate": result["aggregate"],
        "records": result["records"],
    }
    # provider_records are bulky; keep counts in the report tail, not
    # the full copy (query_records stay inspectable via the provider
    # object in-process / a --keep-store forensics run).
    report["n_provider_records"] = len(result["provider_records"])

    out = Path(args.out) if args.out else (
        _RESULTS_DIR / f"amb_proxy_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=str)
    md_path = Path(args.md) if args.md else out.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    if owns_dir and not args.keep_store:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    ov = (report["aggregate"].get("overall") or {})
    line = " | ".join(
        f"B={bk}: any={g.get('geic_any')} all={g.get('geic_all')}"
        for bk, g in ov.items()
    )
    print(f"amb_proxy: {len(report['records'])} questions — {line}")
    print(f"  json: {out}")
    print(f"  md:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
