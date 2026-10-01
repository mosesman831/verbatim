"""AMB run driver — SPEC_V8 §15.2 (V8-15.06–15.13; D8-27).

Authorization gate FIRST (V7-00.04 carried; §26): the owner records O
decisions in ``eval/amb/authorizations.json`` (shape documented below;
copy ``authorizations.example.json``).  Every requested row whose
required decisions are unrecorded emits
``status="blocked_on_authorization"`` with the missing ids — O1/O2/O3
dataset licenses, O5 reader+judge spend, O6 comparator infrastructure,
``amb_license`` + ``amb_pin`` for the V8-15.11 license/pin check — and
the run exits nonzero **with the report written** (not a crash, never a
fabricated result — the not_run discipline).

With the required records present the row runs the ingest → retrieve →
answer → judge loop:

* ingest/retrieve go through :class:`eval.amb.provider.VerbatimAMBProvider`
  (real ``Memory.add`` / ``Memory.search`` — V8-15.06);
* the reader/judge model calls go through a pluggable
  :class:`ModelBackend` — **the default raises
  :class:`AuthorizationBlocked`**, never a silent stub (a real backend
  only exists when the owner wires the pinned AMB checkout's own
  GeminiLLM — prompts and models unmodified, V8-15.07);
* reader/judge prompts come from a :class:`PromptSource` — the pinned
  AMB checkout's ``_DEFAULT_OPEN_PROMPT`` / ``_DEFAULT_MCQ_PROMPT`` /
  judge ``_PROMPT`` (digests pinned in the manifest), or an explicitly
  injected source; a missing prompt blocks the row on ``amb_pin``;
* retrieval-mode rows score returned id sets (PrecisionMemBench
  semantics — ``assert_include``/``assert_exclude``/``gold_ids``), no
  model anywhere (V8-15.09 step 1).

``authorizations.json`` shape::

    {
      "schema": "amb_authorizations/v1",
      "decisions": {"O1": {"granted": true, "note": "...", "recorded_at": "..."},
                    "O2": {"granted": false}, "O3": {...}, "O5": {...},
                    "O6": {...}, "O11": {...}, "O13": {...}},
      "amb_license": {"verified": true, "commit": "<sha>", "digest": "..."},
      "amb_pin": {"commit": "<sha>", "root": "/path/to/checkout"}
    }

``granted`` must be explicitly ``true``; an absent file means every
decision is undecided (the §26 defaults — all rows blocked).

Run order follows V8-15.09 (``RUN_ORDER``); the comparator row
(V8-15.12, O6) is declared but never executed here — it runs inside the
pinned AMB checkout itself.

Usage::

    python -m eval.amb.runner --dataset locomo10 --mode rag \
        --dataset-file data.json --token-budget 4500 --out report.json

    python -m eval.amb.runner --plan        # the authorization board
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .manifest import build_manifest
from .provider import (
    PROVIDER_VERSION,
    TOKEN_BUDGETS,
    VerbatimAMBProvider,
    context_string,
    harness_patch_records,
    provider_arm_fields,
    retrieval_context,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SCHEMA = "amb_run/v8"

DEFAULT_AUTH_PATH = os.path.join("eval", "amb", "authorizations.json")

#: Exit codes — 0 all requested rows executed; 3 any row blocked on
#: authorization (report written, nonzero-with-report); 4 execution
#: error without blocks; 2 CLI usage errors (argparse default).
EXIT_OK = 0
EXIT_BLOCKED = 3
EXIT_ERROR = 4

#: V8-15.09 run order — every row the program will ever quote, with its
#: §26 requirement set.  ``amb_license``/``amb_pin`` implement V8-15.11.
RUN_ORDER: tuple = (
    {
        "row": "precisionmembench.retrieval",
        "dataset": "precisionmembench",
        "mode": "retrieval",
        "requires": ("amb_license", "amb_pin"),
        "note": "no model in the loop — V8-15.09 step 1",
    },
    {
        "row": "locomo10.oracle",
        "dataset": "locomo10",
        "mode": "rag",
        "oracle": True,
        "requires": ("amb_license", "amb_pin", "O1", "O5"),
        "note": "reader ceiling on gold documents — V8-15.09 step 2",
    },
    {
        "row": "locomo10.rag",
        "dataset": "locomo10",
        "mode": "rag",
        "requires": ("amb_license", "amb_pin", "O1", "O5"),
    },
    {
        "row": "longmemeval_s.rag",
        "dataset": "longmemeval_s",
        "mode": "rag",
        "requires": ("amb_license", "amb_pin", "O2", "O5"),
    },
    {
        "row": "personamem_32k.rag",
        "dataset": "personamem_32k",
        "mode": "rag",
        "requires": ("amb_license", "amb_pin", "O3", "O5"),
    },
    {
        "row": "lifebench_en.rag",
        "dataset": "lifebench_en",
        "mode": "rag",
        "requires": ("amb_license", "amb_pin", "O3", "O5"),
    },
    {
        "row": "beam.rag",
        "dataset": "beam",
        "mode": "rag",
        "requires": ("amb_license", "amb_pin", "O3", "O5"),
    },
    {
        "row": "hindsight.comparator",
        "dataset": "hindsight",
        "mode": "rag",
        "comparator": True,
        "requires": ("amb_license", "amb_pin", "O5", "O6"),
        "note": (
            "V8-15.12 — runs inside the pinned AMB checkout itself, "
            "never this runner"
        ),
    },
)


# ---------------------------------------------------------------------------
# authorizations
# ---------------------------------------------------------------------------


def load_authorizations(path: Optional[str]) -> Dict[str, Any]:
    """Read ``authorizations.json``; absent/unparseable → every
    decision undecided (the §26 defaults — never assumed granted)."""
    out: Dict[str, Any] = {
        "path": path,
        "present": False,
        "decisions": {},
        "amb_license": False,
        "amb_pin": None,
    }
    if not path:
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(doc, Mapping):
        return out
    out["present"] = True
    dec = doc.get("decisions") or {}
    out["decisions"] = {
        str(k): (dict(v) if isinstance(v, Mapping) else {"granted": v})
        for k, v in dec.items()
    }
    lic = doc.get("amb_license") or {}
    out["amb_license"] = bool(
        isinstance(lic, Mapping) and lic.get("verified") is True
    )
    out["amb_license_record"] = lic if isinstance(lic, Mapping) else {}
    pin = doc.get("amb_pin") or {}
    out["amb_pin"] = dict(pin) if isinstance(pin, Mapping) else None
    return out


def requirement_met(auths: Mapping[str, Any], req: str) -> bool:
    """One requirement → granted?  ``O*`` ids read the decisions map;
    ``amb_license`` needs ``verified:true``; ``amb_pin`` needs a
    recorded commit."""
    if req.startswith("O"):
        rec = (auths.get("decisions") or {}).get(req) or {}
        return rec.get("granted") is True
    if req == "amb_license":
        return auths.get("amb_license") is True
    if req == "amb_pin":
        pin = auths.get("amb_pin") or {}
        return bool(pin.get("commit"))
    return False


def _blocked_order(req: str) -> tuple:
    """Canonical ``blocked_on`` ordering: owner decisions (``O<n>``) in
    numeric order first — they are what a human must decide — then
    harness prerequisites (``amb_license``, ``amb_pin``) alphabetically.
    """
    if req.startswith("O") and req[1:].isdigit():
        return (0, int(req[1:]), req)
    return (1, 0, req)


def missing_requirements(auths: Mapping[str, Any],
                         requires: Iterable[str]) -> List[str]:
    missing = [r for r in requires if not requirement_met(auths, r)]
    return sorted(missing, key=_blocked_order)


def evaluate_plan(auths: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The full V8-15.09 board — every declared row with its current
    authorization state (``ready`` only when every requirement met)."""
    out = []
    for row in RUN_ORDER:
        missing = missing_requirements(auths, row["requires"])
        out.append({
            "row": row["row"],
            "dataset": row.get("dataset"),
            "mode": row.get("mode"),
            "requires": list(row["requires"]),
            "status": ("ready" if not missing
                       else "blocked_on_authorization"),
            "blocked_on": missing,
            **({"note": row["note"]} if row.get("note") else {}),
        })
    return out


# ---------------------------------------------------------------------------
# pluggable model backends + prompt sources
# ---------------------------------------------------------------------------


class AuthorizationBlocked(Exception):
    """A run step required an authorization (or provisioned backend)
    that is not present — carries the decision/requirement id so the
    row can report ``blocked_on_authorization`` honestly."""

    def __init__(self, decision: str, detail: str = "") -> None:
        super().__init__(f"{decision}: {detail}" if detail else decision)
        self.decision = decision
        self.detail = detail


class ModelBackend:
    """Reader + judge seam (V8-15.07 — AMB's pinned Gemini models run
    unmodified; this runner never ships a fake)."""

    def answer(self, prompt: str, *, query: str = "",
               context: str = "") -> Mapping:
        """→ ``{"answer": str, "reasoning": str}``."""
        raise NotImplementedError

    def judge(self, query: str, answer: str,
              gold_answers: Sequence[str], *,
              context: str = "") -> Mapping:
        """→ ``{"correct": bool, "reason": str}``."""
        raise NotImplementedError


class BlockedModelBackend(ModelBackend):
    """Default backend — raises the authorization error rather than
    fabricating a model call (never a silent stub).  A real backend is
    the pinned AMB checkout's own GeminiLLM, wired by the owner after
    O5 is recorded."""

    def __init__(self, decision: str = "O5") -> None:
        self.decision = decision

    def answer(self, prompt: str, *, query: str = "",
               context: str = "") -> Mapping:
        raise AuthorizationBlocked(
            self.decision,
            "no reader backend provisioned — the default raises; plug "
            "the pinned AMB checkout's GeminiLLM (V8-15.07)",
        )

    def judge(self, query: str, answer: str,
              gold_answers: Sequence[str], *,
              context: str = "") -> Mapping:
        raise AuthorizationBlocked(
            self.decision,
            "no judge backend provisioned — the default raises; plug "
            "the pinned AMB checkout's GeminiJudge (V8-15.07)",
        )


class PromptSource:
    """AMB prompt bytes + their digests (K88: prompt/model/scoring
    digests equal the pinned upstream commit)."""

    available = False

    def answer_prompt(self, query: str, context: str,
                      task_type: str = "open") -> Optional[str]:
        return None

    def judge_prompt(self, query: str, gold_answers: Sequence[str],
                     answer: str) -> Optional[str]:
        return None

    def digests(self) -> Dict[str, Optional[str]]:
        return {"reader_open": None, "reader_mcq": None, "judge": None}


class StaticPromptSource(PromptSource):
    """Prompts supplied explicitly (tests, or owner-pasted pinned text);
    digests are computed over the supplied strings."""

    def __init__(self, *, open_prompt: Optional[str] = None,
                 mcq_prompt: Optional[str] = None,
                 judge_prompt: Optional[str] = None) -> None:
        self._open = open_prompt
        self._mcq = mcq_prompt
        self._judge = judge_prompt
        self.available = bool(open_prompt or mcq_prompt)
        self._dig = {
            "reader_open": self._sha(open_prompt),
            "reader_mcq": self._sha(mcq_prompt),
            "judge": self._sha(judge_prompt),
        }

    @staticmethod
    def _sha(text: Optional[str]) -> Optional[str]:
        import hashlib

        return (hashlib.sha256(text.encode("utf-8")).hexdigest()
                if isinstance(text, str) else None)

    def answer_prompt(self, query: str, context: str,
                      task_type: str = "open") -> Optional[str]:
        tpl = self._mcq if task_type == "mcq" else self._open
        if tpl is None:
            return None
        return tpl.format(context=context, query=query)

    def judge_prompt(self, query: str, gold_answers: Sequence[str],
                     answer: str) -> Optional[str]:
        if self._judge is None:
            return None
        gold_str = "\n".join(f"- {a}" for a in gold_answers)
        return self._judge.format(
            query=query, gold_answers=gold_str, answer=answer)

    def digests(self) -> Dict[str, Optional[str]]:
        return dict(self._dig)


class AMBCheckoutPromptSource(PromptSource):
    """Prompts read from the pinned AMB checkout (``pin.root``) —
    imports ``memory_bench`` from that tree and extracts the unmodified
    ``_DEFAULT_OPEN_PROMPT`` / ``_DEFAULT_MCQ_PROMPT`` /
    judge ``_PROMPT`` (V8-15.07).  Unavailable without a checkout —
    honestly reported, never substituted."""

    def __init__(self, amb_root: Optional[str]) -> None:
        self.amb_root = amb_root
        self.available = False
        self._open: Optional[str] = None
        self._mcq: Optional[str] = None
        self._judge: Optional[str] = None
        self.error: Optional[str] = None
        if amb_root:
            self._load()

    def _load(self) -> None:
        import importlib
        import sys as _sys

        src = os.path.join(str(self.amb_root), "src")
        if not os.path.isdir(src):
            src = str(self.amb_root)
        _sys.path.insert(0, src)
        try:
            base = importlib.import_module("memory_bench.dataset.base")
            judge = importlib.import_module("memory_bench.judge")
            self._open = getattr(base, "_DEFAULT_OPEN_PROMPT", None)
            self._mcq = getattr(base, "_DEFAULT_MCQ_PROMPT", None)
            self._judge = getattr(judge, "_PROMPT", None)
            self.available = bool(self._open or self._mcq)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                _sys.path.remove(src)
            except ValueError:
                pass

    def answer_prompt(self, query: str, context: str,
                      task_type: str = "open") -> Optional[str]:
        tpl = self._mcq if task_type == "mcq" else self._open
        if tpl is None:
            return None
        return tpl.format(context=context, query=query)

    def judge_prompt(self, query: str, gold_answers: Sequence[str],
                     answer: str) -> Optional[str]:
        if self._judge is None:
            return None
        gold_str = "\n".join(f"- {a}" for a in gold_answers)
        return self._judge.format(
            query=query, gold_answers=gold_str, answer=answer)

    def digests(self) -> Dict[str, Optional[str]]:
        return {
            "reader_open": StaticPromptSource._sha(self._open),
            "reader_mcq": StaticPromptSource._sha(self._mcq),
            "judge": StaticPromptSource._sha(self._judge),
        }


# ---------------------------------------------------------------------------
# dataset view (JSON file source — AMB checkout wiring is a HANDOFF;
# the file mirrors the upstream Document/Query field names)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryView:
    """One scored query — the AMB ``Query`` fields plus the per-query
    assertions retrieval-mode datasets ship (gold is scorer-side only)."""

    id: str
    query: str
    gold_answers: tuple = ()
    user_id: Optional[str] = None
    meta: Mapping = field(default_factory=dict)
    gold_ids: tuple = ()
    assert_include: tuple = ()
    assert_exclude: tuple = ()
    task_type: str = "open"


@dataclass
class DatasetView:
    dataset_id: str
    split: Optional[str]
    task_type: str
    documents: List[Any]
    queries: List[QueryView]
    digest: Optional[str] = None
    license: Optional[str] = None


def _qview(q: Any) -> QueryView:
    def g(*names, default=None):
        if isinstance(q, Mapping):
            for n in names:
                if q.get(n) is not None:
                    return q[n]
            return default
        for n in names:
            v = getattr(q, n, None)
            if v is not None:
                return v
        return default

    meta = g("meta", "metadata", default={}) or {}
    return QueryView(
        id=str(g("id", "query_id", "task_id", default="q")),
        query=str(g("query", "question", default="")),
        gold_answers=tuple(
            str(x) for x in (g("gold_answers", "answers", default=()) or ())
        ),
        user_id=(None if g("user_id") is None else str(g("user_id"))),
        meta=meta if isinstance(meta, Mapping) else {},
        gold_ids=tuple(str(x) for x in (g("gold_ids", default=()) or ())),
        assert_include=tuple(
            str(x) for x in (g("assert_include", "must_include",
                               default=()) or ())
        ),
        assert_exclude=tuple(
            str(x) for x in (g("assert_exclude", "must_exclude",
                               default=()) or ())
        ),
        task_type=str(g("task_type", default="open")),
    )


def load_dataset_file(path: str) -> DatasetView:
    """``{"documents":[...], "queries":[...]}`` JSON → DatasetView;
    sha256-pinned for the manifest."""
    from eval.v7.manifest import sha256_file

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, Mapping):
        raise SystemExit(f"{path}: expected a JSON object")
    queries = [_qview(q) for q in (doc.get("queries") or ())]
    return DatasetView(
        dataset_id=str(doc.get("dataset") or doc.get("dataset_id")
                       or os.path.basename(path)),
        split=doc.get("split"),
        task_type=str(doc.get("task_type") or "open"),
        documents=list(doc.get("documents") or ()),
        queries=queries,
        digest=sha256_file(path),
        license=doc.get("license"),
    )


# ---------------------------------------------------------------------------
# the row loop
# ---------------------------------------------------------------------------


def _score_mcq(answer: str, gold_answers: Sequence[str]) -> tuple:
    """AMB's exact-letter MCQ scoring (``amb-mcq/v1`` — mirrors
    ``memory_bench/runner.py::_score_mcq``; no judge call involved)."""
    def norm(s: str) -> str:
        return s.strip().lower().strip("(). ")[:1]

    letter = norm(answer)
    for gold in gold_answers:
        if norm(gold) == letter:
            return True, "letter match"
    return False, f"expected one of {list(gold_answers)!r}, got {answer!r}"


def _pct(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    i = min(len(xs) - 1, max(0, round((p / 100.0) * (len(xs) - 1))))
    return xs[i]


def execute_row(
    row: Mapping[str, Any],
    dataset: DatasetView,
    provider: VerbatimAMBProvider,
    *,
    reader: ModelBackend,
    judge: ModelBackend,
    prompts: PromptSource,
    query_limit: Optional[int] = None,
    k: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """The ingest → retrieve → answer → judge loop for one run row.

    Raises nothing for routine outcomes — per-query errors land in the
    query record; :class:`AuthorizationBlocked` propagates (the row
    reports blocked, never a fabricated score)."""
    queries = list(dataset.queries)
    if query_limit is not None:
        queries = queries[: int(query_limit)]

    # Oracle mode ingests ONLY gold documents (upstream runner
    # convention: load_documents(ids=gold_ids) before ingest) — the
    # reader's ceiling is measured without retrieval loss in the loop.
    documents = list(dataset.documents)
    if row.get("oracle"):
        gold = {g for q in queries for g in (q.gold_ids or ())}
        documents = [
            d for d in documents
            if str(
                d.get("id") if isinstance(d, Mapping)
                else getattr(d, "id", "")
            ) in gold
        ]

    t0 = time.perf_counter()
    ingest_rep = provider.ingest(documents)
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    out_queries: List[Dict[str, Any]] = []
    for q in queries:
        meta = dict(q.meta or {})
        ts = meta.get("query_timestamp")
        t1 = time.perf_counter()
        try:
            docs, raw = provider.retrieve(
                q.query, k=int(k), user_id=q.user_id,
                query_timestamp=ts,
            )
        except AuthorizationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 — typed, kept visible
            out_queries.append({
                "query_id": q.id, "query": q.query,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        retrieve_ms = (time.perf_counter() - t1) * 1000.0
        rec = (raw or {}).get("verbatim") or {}
        if not rec:
            # provider v2 (V85-02.05): ``raw_response`` is ALWAYS None —
            # upstream ``build_rag_prompt`` would serialize it into the
            # reader prompt.  The per-query record instead lives on the
            # provider's own ``query_records``; take the row this call
            # just appended (same keys as the old raw block).
            _recs = getattr(provider, "query_records", None) or []
            rec = dict(_recs[-1]) if _recs else {}

        if str(row.get("mode")) == "retrieval":
            ctx = retrieval_context(docs)
            ids = [
                str(getattr(d, "id", "") if not isinstance(d, Mapping)
                    else d.get("id", ""))
                for d in docs
            ]
            must = tuple(q.assert_include or q.gold_ids or ())
            inc = all(g in ids for g in must)
            exc = not any(g in ids for g in (q.assert_exclude or ()))
            correct = inc and exc
            answer = f"{len(docs)} memories retrieved"
            reasoning = ""
            missed = ",".join(g for g in must if g not in ids)
            judge_reason = (
                f"id-set: include={'ok' if inc else 'MISS:' + missed} "
                f"exclude={'ok' if exc else 'VIOLATED'}"
            )
        else:
            ctx = context_string(docs)
            ttype = q.task_type or dataset.task_type
            prompt = prompts.answer_prompt(q.query, ctx, ttype)
            if prompt is None:
                raise AuthorizationBlocked(
                    "amb_pin",
                    "no unmodified AMB reader prompt available — pin "
                    "the checkout (amb_pin) or inject a prompt source",
                )
            ans = reader.answer(prompt, query=q.query, context=ctx)
            answer = str(
                (ans or {}).get("answer", "")
                if isinstance(ans, Mapping) else ans
            )
            reasoning = str(
                (ans or {}).get("reasoning", "")
                if isinstance(ans, Mapping) else ""
            )
            if ttype == "mcq":
                correct, judge_reason = _score_mcq(answer, q.gold_answers)
            else:
                jr = judge.judge(
                    q.query, answer, q.gold_answers, context=ctx
                )
                correct = bool((jr or {}).get("correct"))
                judge_reason = str((jr or {}).get("reason", ""))
        out_queries.append({
            "query_id": q.id,
            "query": q.query,
            "status": "ok",
            "answer": answer,
            "reasoning": reasoning,
            "context": ctx,
            "est_context_tokens": rec.get("est_context_tokens"),
            "retrieve_ms": round(retrieve_ms, 3),
            "provider_retrieve_ms": rec.get("retrieve_ms"),
            "n_docs": len(docs),
            "doc_ids": rec.get("doc_ids"),
            "gold_answers": list(q.gold_answers),
            "correct": correct,
            "judge_reason": judge_reason,
            "provider_status": rec.get("status"),
            "as_of": rec.get("as_of"),
            "warnings": rec.get("warnings"),
        })

    n = len(out_queries)
    n_ok = sum(1 for r in out_queries if r.get("status") == "ok")
    n_correct = sum(1 for r in out_queries if r.get("correct"))
    lat = [r["retrieve_ms"] for r in out_queries if "retrieve_ms" in r]
    toks = [
        r["est_context_tokens"] for r in out_queries
        if isinstance(r.get("est_context_tokens"), int)
    ]
    return {
        "summary": {
            "n_queries": n,
            "n_answered": n_ok,
            "n_correct": n_correct,
            "accuracy": (n_correct / n_ok) if n_ok else None,
            "ingest_ms": round(ingest_ms, 3),
            "ingest_docs": ingest_rep.get("indexed"),
            "ingest_report": ingest_rep,
            "retrieve_ms": {
                "avg": (sum(lat) / len(lat)) if lat else None,
                "p50": _pct(lat, 50),
                "p95": _pct(lat, 95),
            },
            "est_context_tokens": {
                "avg": (sum(toks) / len(toks)) if toks else None,
                "p50": _pct(toks, 50),
                "p95": _pct(toks, 95),
                "meter": "tok/v1 (provider estimate; AMB meters "
                        "cl100k_base on the real run)",
            },
            "token_budget": provider.token_budget,
            "doc_mode": provider.doc_mode,
            "timeout_ms": provider.timeout_ms,
            "k": int(k),
            "seed": int(seed),
            "token_curve": provider.token_curve(),
        },
        "queries": out_queries,
        "notes": list(provider.notes),
    }


# ---------------------------------------------------------------------------
# provider + row construction
# ---------------------------------------------------------------------------


def _row_id(row: Mapping[str, Any], budget: Optional[int]) -> str:
    b = "unbounded" if budget is None else str(budget)
    return f"{row['row']}@{b}"


def _model_label(backend: Any, env_var: str) -> str:
    """Reader/judge id for the manifest (V85-02.07): the env-declared
    model id when present, else the backend class name — an honest
    ``backend:BlockedModelBackend`` beats a fabricated model id."""
    env = os.environ.get(env_var)
    if env:
        return env
    return f"backend:{type(backend).__name__}"


def make_provider(
    row: Mapping[str, Any],
    *,
    store_root: str,
    token_budget: Optional[int],
    timeout_ms: float = 500.0,
    memory_factory: Optional[Any] = None,
    memory_kwargs: Optional[dict] = None,
    **kw: Any,
) -> VerbatimAMBProvider:
    """Construct the verbatim provider for a row — ``doc_mode`` follows
    the run's mode flag (a run-level arm knob, never a dataset branch)."""
    doc_mode = "items" if str(row.get("mode")) == "retrieval" else "pack"
    store_dir = os.path.join(
        store_root,
        row["row"].replace(".", "_"),
        "unbounded" if token_budget is None else f"t{token_budget}",
    )
    return VerbatimAMBProvider(
        store_dir=store_dir,
        token_budget=token_budget,
        timeout_ms=timeout_ms,
        doc_mode=doc_mode,
        memory_factory=memory_factory,
        memory_kwargs=memory_kwargs,
        **kw,
    )


def select_rows(dataset: Optional[str], mode: Optional[str],
                oracle: bool) -> List[Dict[str, Any]]:
    """Requested rows from RUN_ORDER — comparator rows are declared but
    never selected (they run inside the pinned AMB checkout).  The
    dedicated ``*.oracle`` RUN_ORDER entries are board declarations for
    ``evaluate_plan``; selecting them directly would double-count the
    rag row — oracle variants are minted from rag rows only when
    ``oracle`` is requested.
    """
    rows = [r for r in RUN_ORDER
            if not r.get("comparator") and not r.get("oracle")]
    if dataset:
        rows = [r for r in rows if r.get("dataset") == dataset]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    if oracle:
        rows = [dict(r, oracle=True, row=r["row"].replace(".rag", ".oracle"))
                for r in rows if r.get("mode") == "rag"]
        for r in rows:
            r["requires"] = tuple(dict.fromkeys(
                (*r["requires"], "O5")))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_budgets(text: Optional[str]) -> List[Optional[int]]:
    if not text or text.strip() in ("all", "*"):
        return list(TOKEN_BUDGETS)
    out: List[Optional[int]] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(
            None if tok in ("unbounded", "inf", "none") else int(tok)
        )
    return out or list(TOKEN_BUDGETS)


def main(argv: Optional[Sequence[str]] = None, *,
         provider_factory: Optional[Any] = None,
         reader: Optional[ModelBackend] = None,
         judge: Optional[ModelBackend] = None,
         prompts: Optional[PromptSource] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.amb.runner",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("--dataset", default=None,
                    help="AMB dataset name from the V8-15.09 plan")
    ap.add_argument("--dataset-file", default=None,
                    help="JSON file with documents+queries (local mirror "
                         "of an AMB dataset; pinned by sha256)")
    ap.add_argument("--mode", default=None,
                    choices=["rag", "retrieval"])
    ap.add_argument("--oracle", action="store_true",
                    help="oracle mode — gold documents only (V8-15.09)")
    ap.add_argument("--token-budget", default="all",
                    help="comma list or 'all' — the V8-15.08 sweep "
                         "{1000,2000,4500,9000,unbounded}")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="cap queries (--smoke pins 20 per K87)")
    ap.add_argument("--smoke", action="store_true",
                    help="V8-15.06 conformance smoke: --query-limit 20")
    ap.add_argument("--k", type=int, default=10,
                    help="retrieval depth passed to Memory.search limit")
    ap.add_argument("--timeout-ms", type=float, default=500.0,
                    help="search deadline (product default 500, V8-15.06)")
    ap.add_argument("--workdir", default=os.path.join(
        "eval", "amb", "stores"),
        help="provider store root (default eval/amb/stores)")
    ap.add_argument("--authorizations", default=DEFAULT_AUTH_PATH,
                    help=f"O-decision records (default {DEFAULT_AUTH_PATH})")
    ap.add_argument("--amb-root", default=None,
                    help="pinned AMB checkout root (prompt/model pin)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", action="store_true",
                    help="print the evaluated V8-15.09 run-order board")
    ap.add_argument("--out", default=None, help="write report JSON here")
    args = ap.parse_args(argv)

    # ---- authorization gate FIRST -------------------------------------
    auths = load_authorizations(args.authorizations)
    plan = evaluate_plan(auths)
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "provider_version": PROVIDER_VERSION,
        "authorizations": {
            "path": auths.get("path"),
            "present": auths.get("present"),
            "decisions": {
                k: (v.get("granted") is True)
                for k, v in (auths.get("decisions") or {}).items()
            },
            "amb_license": auths.get("amb_license"),
            "amb_pin": bool((auths.get("amb_pin") or {}).get("commit")),
        },
        "plan": plan,
        "rows": [],
        "manifest": None,
        "environment": None,
    }
    if args.plan:
        report["exit"] = 0
        print(json.dumps(report["plan"], indent=2, sort_keys=True))
        return 0

    rows = select_rows(args.dataset, args.mode, args.oracle)
    if not rows:
        ap.error(
            f"no plan rows match dataset={args.dataset!r} "
            f"mode={args.mode!r} (declared datasets: "
            f"{sorted({r.get('dataset') for r in RUN_ORDER})})"
        )

    budgets = _parse_budgets(args.token_budget)
    dataset: Optional[DatasetView] = None
    if args.dataset_file:
        dataset = load_dataset_file(args.dataset_file)
    query_limit = 20 if args.smoke else args.query_limit

    reader = reader or BlockedModelBackend("O5")
    judge = judge or reader
    if prompts is None:
        pin_root = args.amb_root or (
            (auths.get("amb_pin") or {}).get("root")
        )
        prompts = AMBCheckoutPromptSource(pin_root)

    exit_code = EXIT_OK
    ran_dataset: Optional[DatasetView] = None
    for row in rows:
        for budget in budgets:
            rid = _row_id(row, budget)
            missing = missing_requirements(auths, row["requires"])
            if missing:
                report["rows"].append({
                    "row": rid,
                    "status": "blocked_on_authorization",
                    "blocked_on": missing,
                })
                exit_code = EXIT_BLOCKED
                continue
            if row.get("mode") != "retrieval" and not getattr(
                prompts, "available", False
            ):
                # an authorized rag row without prompt bytes cannot
                # form AMB's unmodified prompt — the pin is ineffective
                report["rows"].append({
                    "row": rid,
                    "status": "blocked_on_authorization",
                    "blocked_on": ["amb_pin"],
                    "detail": (
                        "amb_pin granted but the checkout produced no "
                        "prompt bytes"
                        + (f" ({prompts.error})"
                           if getattr(prompts, "error", None) else "")
                    ),
                })
                exit_code = EXIT_BLOCKED
                continue
            if dataset is None:
                report["rows"].append({
                    "row": rid,
                    "status": "not_executed",
                    "reason": "no --dataset-file supplied; the AMB "
                              "checkout dataset wiring is a handoff",
                })
                if exit_code != EXIT_BLOCKED:
                    exit_code = EXIT_ERROR
                continue
            ran_dataset = dataset
            try:
                mk = provider_factory or make_provider
                provider = mk(
                    row,
                    store_root=args.workdir,
                    token_budget=budget,
                    timeout_ms=args.timeout_ms,
                )
                provider.prepare(
                    getattr(provider, "_store_dir", None) or args.workdir,
                    reset=True,
                )
                out = execute_row(
                    row, dataset, provider,
                    reader=reader, judge=judge, prompts=prompts,
                    query_limit=query_limit, k=args.k, seed=args.seed,
                )
                report["rows"].append({
                    "row": rid,
                    "status": "executed",
                    **out,
                })
                try:
                    provider.cleanup()
                except Exception:
                    pass
            except AuthorizationBlocked as exc:
                report["rows"].append({
                    "row": rid,
                    "status": "blocked_on_authorization",
                    "blocked_on": [exc.decision],
                    "detail": exc.detail,
                })
                exit_code = EXIT_BLOCKED
            except Exception as exc:  # noqa: BLE001 — typed row, not a crash
                report["rows"].append({
                    "row": rid,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if exit_code == EXIT_OK:
                    exit_code = EXIT_ERROR

    # ---- manifest (every row pinned, V7-22.24 carried) -----------------
    try:
        manifest = build_manifest(
            root=None,
            run_id=None,
            provider_version=PROVIDER_VERSION,
            arm={
                "name": "verbatim",
                "timeout_ms": args.timeout_ms,
                "token_budgets": [
                    ("unbounded" if b is None else b) for b in budgets
                ],
                "k": args.k,
                "doc_mode": (
                    "items" if args.mode == "retrieval" else "pack"
                ),
                "oracle": bool(args.oracle),
                # V85-02.07 — the provider's arm record (B/W_r/L,
                # tokenizer, concurrency, provider revision).
                **provider_arm_fields(),
            },
            dataset={
                "id": (ran_dataset.dataset_id if ran_dataset
                       else args.dataset),
                "split": ran_dataset.split if ran_dataset else None,
                "digest": (
                    ran_dataset.digest if ran_dataset else "no_dataset"
                ),
                "license": (ran_dataset.license if ran_dataset else None),
            },
            amb={
                "commit": (auths.get("amb_pin") or {}).get("commit"),
                "prompt_digests": prompts.digests(),
                # V85-02.07 — the three amb_patches.md harness patches.
                "harness_patches": harness_patch_records(),
            },
            models={
                "reader": {"id": _model_label(
                    reader, "OMB_ANSWER_MODEL")},
                "judge": {"id": _model_label(
                    judge, "OMB_JUDGE_MODEL")},
            },
            authorizations=report["authorizations"],
            seeds=[args.seed],
        )
        report["manifest"] = {
            "digest": manifest["digest"], "status": "pinned",
            "manifest": manifest,
        }
    except Exception as exc:  # noqa: BLE001
        report["manifest"] = {
            "digest": None, "status": "unpinned",
            "error": f"{type(exc).__name__}: {exc}",
        }

    report["exit"] = exit_code
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                    exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "AMBCheckoutPromptSource",
    "AuthorizationBlocked",
    "BlockedModelBackend",
    "DatasetView",
    "EXIT_BLOCKED",
    "EXIT_ERROR",
    "EXIT_OK",
    "ModelBackend",
    "PromptSource",
    "QueryView",
    "RUN_ORDER",
    "SCHEMA",
    "StaticPromptSource",
    "evaluate_plan",
    "execute_row",
    "load_authorizations",
    "load_dataset_file",
    "main",
    "make_provider",
    "missing_requirements",
    "requirement_met",
    "select_rows",
]
