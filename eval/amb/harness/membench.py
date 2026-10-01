"""
MemBench dataset (https://github.com/import-myself/Membench).
Paper: https://arxiv.org/abs/2506.21605 (ACL 2025 Findings)

Comprehensive memory benchmark for LLM-based agents — factual memory and
reflective memory across participation (first-person) and observation
(third-person) interactive scenarios.

Data ships via Google Drive (link in the repo README); the harness cannot
auto-download it — fetch + unzip manually once, then either set
MEMBENCH_DATA_DIR to the directory containing the four split files or drop
them in the membench dataset cache dir:

  FirstAgentDataHighLevel.json    participation-reflective
  FirstAgentDataLowLevel.json     participation-factual
  ThirdAgentDataHighLevel.json    observation-reflective
  ThirdAgentDataLowLevel.json     observation-factual

Structure
---------
Each split file is {category: {subcategory: [dialogue, ...]}} where every
dialogue is {tid|gid, message_list, QA}:
  - message_list: list of sessions; each session is a list of turns.
    Turn schemas vary: {sid, user_message, assistant_message},
    {mid, user, assistant}, or {mid, message} (third-person), all with
    'time'/'place'.  Low-level third-person files use a FLAT list of
    turns (no session nesting).
  - QA: {qid, question, answer, target_step_id, choices{A-D},
    ground_truth, time} — MCQ with a gold letter; answer/choice values
    may be lists.

Documents = one per dialogue (messages carried as canonical
speaker/text/at/dia_id utterance dicts so the verbatim provider
materializes real turn units — the blob path falls back to v6 recall
whose deadline_ms contract rejects the det-timeout budget).
Queries   = one per QA, scoped to its dialogue (isolation per persona).
Splits    = the four data files; categories = top-level categories
            (doc-partitioned: each category owns its dialogues).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

_SPLIT_FILES = {
    "first_high": "FirstAgentDataHighLevel.json",
    "first_low": "FirstAgentDataLowLevel.json",
    "third_high": "ThirdAgentDataHighLevel.json",
    "third_low": "ThirdAgentDataLowLevel.json",
}
SPLITS = list(_SPLIT_FILES)

_GDRIVE_URL = "https://drive.google.com/file/d/112Zraj4pTPH4Idph6i1uMOLA_LPFdGr0/view"


def _has_zh(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


class MemBenchDataset(Dataset):
    """MemBench — factual + reflective memory benchmark (MCQ)."""

    name = "membench"
    published = True
    description = "MemBench memory benchmark: participation/observation × factual/reflective."
    splits = SPLITS
    task_type = "mcq"
    isolation_unit = "dialogue"
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2506.21605"},
        {"label": "GitHub", "url": "https://github.com/import-myself/Membench"},
    ]

    def __init__(self) -> None:
        env = os.environ.get("MEMBENCH_DATA_DIR")
        self._data_dir = Path(env) if env else dataset_cache_dir("membench")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_path(self, split: str) -> Path:
        path = self._data_dir / _SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"MemBench split file not found: {path}\n"
                f"Download the dataset zip from {_GDRIVE_URL}, unzip it, and set "
                f"MEMBENCH_DATA_DIR to the 'data/' directory (or copy the four "
                f"split files into {self._data_dir})."
            )
        return path

    def _load_raw(self, split: str) -> dict:
        with open(self._data_path(split), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _iter_items(data: dict):
        """Yield (cat, sub, item) over the nested {cat:{sub:[item]}} structure."""
        for cat, subs in data.items():
            if isinstance(subs, dict):
                for sub, items in subs.items():
                    for item in items:
                        yield cat, sub, item
            elif isinstance(subs, list):
                for item in subs:
                    yield cat, cat, item

    @staticmethod
    def _unit_id(split: str, cat: str, sub: str, tid) -> str:
        return f"{split}:{cat}:{sub}:{tid}"

    @staticmethod
    def _doc_id(cat: str, sub: str, tid) -> str:
        return f"{cat}/{sub}/{tid}"

    @staticmethod
    def _item_tid(item: dict):
        """FirstAgent uses 'tid', ThirdAgent 'gid'."""
        return item.get("tid", item.get("gid"))

    @staticmethod
    def _parse_time(raw: str | None) -> str | None:
        """Parse MemBench time strings like \"'2024-10-01 08:49' Friday\" → ISO."""
        if not raw:
            return None
        s = str(raw).strip().strip("'\"")
        parts = s.split()
        s = parts[0] + " " + parts[1] if len(parts) >= 2 else parts[0]
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                continue
        return None

    @classmethod
    def _turn_utterances(cls, turn: dict, si: int) -> list[dict]:
        """One MemBench turn → canonical utterance dicts for
        ``Document.messages`` (speaker/text/at/dia_id — the keys the
        verbatim provider's ``turns_from`` recognizes, so the engine
        materializes real turn units instead of a blob doc).
        ``target_step_id`` addresses turns as [mid-or-sid, session_idx],
        so ``dia_id`` carries both for honest traceability."""
        tkey = turn.get("mid", turn.get("sid"))
        dia = f"s{si}m{tkey}" if tkey is not None else None
        at = cls._parse_time(turn.get("time"))
        place = turn.get("place")
        outs: list[dict] = []

        def emit(speaker: str, text) -> None:
            if text is None:
                return
            s = str(text)
            if place:
                s = f"({place}) {s}"
            d = {"speaker": speaker, "text": s}
            if at:
                d["at"] = at
            if dia:
                d["dia_id"] = dia
            outs.append(d)

        um, am = turn.get("user_message"), turn.get("assistant_message")
        u, a = turn.get("user"), turn.get("assistant")
        m = turn.get("message")
        if um or am:
            emit("user", um)
            emit("assistant", am)
        elif u or a:
            emit("user", u)
            emit("assistant", a)
        elif m:
            emit("narrator", m)
        return outs

    @classmethod
    def _item_messages(cls, item: dict) -> list[dict]:
        """All sessions flattened to one utterance stream — the engine
        chunks >_MAX_TURN_MESSAGES adds itself; each turn keeps its own
        'time'/'place' so per-turn timestamps survive into the store.
        Third-person low-level files carry a flat turn list — each turn
        then becomes its own one-utterance 'session' (si = turn index)."""
        msgs: list[dict] = []
        for si, session in enumerate(item.get("message_list") or []):
            turns = session if isinstance(session, list) else [session]
            for t in turns:
                if isinstance(t, dict):
                    msgs.extend(cls._turn_utterances(t, si))
        return msgs

    @classmethod
    def _item_content(cls, item: dict) -> str:
        """Readable transcript (used as context/diagnostics; the engine
        payload is rebuilt from ``messages`` by the provider)."""
        parts: list[str] = []
        for si, session in enumerate(item.get("message_list") or []):
            turns = session if isinstance(session, list) else [session]
            lines: list[str] = []
            for t in turns:
                if not isinstance(t, dict):
                    continue
                for u in cls._turn_utterances(t, si):
                    lines.append(f"{u['speaker']}: {u['text']}")
            parts.append(f"## Session {si}\n" + "\n".join(lines))
        return "\n\n".join(parts)

    @staticmethod
    def _first_time(item: dict) -> str | None:
        for session in item.get("message_list") or []:
            turns = session if isinstance(session, list) else [session]
            for turn in turns:
                if isinstance(turn, dict) and turn.get("time"):
                    return turn["time"]
        return None

    @staticmethod
    def _choices_text(choices: dict | None) -> str:
        if not choices:
            return ""

        def fmt(v):
            return ", ".join(v) if isinstance(v, list) else str(v)

        return "\n".join(f"{k}. {fmt(choices[k])}" for k in sorted(choices))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def categories(self, split: str) -> list[str] | None:
        return sorted(self._load_raw(split).keys())

    def category_type(self, split: str, category: str) -> str:
        return "doc"

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        if meta.get("category"):
            axes["Category"] = [meta["category"]]
        if meta.get("subcategory"):
            axes["Subcategory"] = [meta["subcategory"]]
        return axes

    def build_rag_prompt(self, query: str, context: str, task_type: str, split: str, category: str | None = None, meta: dict | None = None) -> str:
        if task_type == "mcq":
            return (
                f"The following is the user's memory/history:\n\n{context}\n\n"
                f"{query}\n\n"
                f"Find the most appropriate answer given the user's history. Answer with only the letter (a), (b), (c), or (d)."
            )
        from .base import _DEFAULT_OPEN_PROMPT
        return _DEFAULT_OPEN_PROMPT.format(context=context, query=query)

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        data = self._load_raw(split)
        queries: list[Query] = []
        for cat, sub, item in self._iter_items(data):
            if category and cat != category:
                continue
            qa = item.get("QA") or {}
            question = qa.get("question", "")
            truth = qa.get("ground_truth")
            if not question or not truth:
                continue
            tid = self._item_tid(item)
            unit = self._unit_id(split, cat, sub, tid)
            opts = self._choices_text(qa.get("choices"))
            query_text = f"{question}\n\n{opts}" if opts else question
            queries.append(Query(
                id=f"{unit}#q{qa.get('qid', 0)}",
                query=query_text,
                gold_ids=[self._doc_id(cat, sub, tid)],
                gold_answers=[str(truth)],
                user_id=unit,
                meta={
                    "category": cat,
                    "subcategory": sub,
                    "retrieval_query": question,
                    **({"query_timestamp": ts} if (ts := self._parse_time(qa.get("time"))) else {}),
                },
            ))
        if limit:
            queries = queries[:limit]
        return queries

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        data = self._load_raw(split)
        documents: list[Document] = []
        for cat, sub, item in self._iter_items(data):
            if category and cat != category:
                continue
            tid = self._item_tid(item)
            doc_id = self._doc_id(cat, sub, tid)
            if ids is not None and doc_id not in ids:
                continue
            unit = self._unit_id(split, cat, sub, tid)
            if user_ids is not None and unit not in user_ids:
                continue
            ts = self._parse_time(self._first_time(item))
            documents.append(Document(
                id=doc_id,
                content=self._item_content(item),
                user_id=unit,
                messages=self._item_messages(item),
                timestamp=ts,
                context=f"MemBench {split} {cat}/{sub} dialogue {tid}",
            ))
        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="MemBench dataset stats")
        table.add_column("Split", style="bold")
        table.add_column("Categories", justify="right")
        table.add_column("Dialogues", justify="right")
        table.add_column("QA pairs", justify="right")
        table.add_column("ZH QA", justify="right")
        for split in SPLITS:
            try:
                data = self._load_raw(split)
            except FileNotFoundError:
                continue
            n_items = n_qa = n_zh = 0
            for cat, sub, item in self._iter_items(data):
                n_items += 1
                qa = item.get("QA") or {}
                if qa.get("question"):
                    n_qa += 1
                    if _has_zh(qa["question"]):
                        n_zh += 1
            table.add_row(split, str(len(data)), str(n_items), str(n_qa), str(n_zh))
        console.print(table)
