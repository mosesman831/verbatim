"""
MSC-MemFuse-MC10 dataset (https://huggingface.co/datasets/Percena/msc-memfuse-mc10).

Structure
---------
JSONL, 500 rows. Each row is a self-contained item:
  question_id, question, answer, choices[10], correct_choice_index,
  haystack_session_ids[5], haystack_sessions[5]

Each question owns a unique haystack: 5 multi-session chat dialogs
(OpenAI {role, content} turns, ~13 turns/session, ~6.6k chars/question).

Documents = one per session, id "{question_id}_s{i}".
Queries   = the question + ten lettered options "(a) .. (j)".
            gold_ids = all five session doc ids (the haystack is the corpus).
Task      = 10-way multiple choice over episodic memory. Letter-match
            scoring (runner._score_mcq), no judge needed.
"""
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._cache import dataset_cache_dir
from .base import Dataset
from ..models import Document, Query

SPLITS = ["main"]

_FILE = "msc_memfuse_mc10.json"
_REPO = "Percena/msc-memfuse-mc10"
_URL = (
    "https://huggingface.co/datasets/Percena/msc-memfuse-mc10/"
    "resolve/main/data/msc_memfuse_mc10.json"
)
_LETTERS = "abcdefghij"


class MscMemfuseDataset(Dataset):
    """MSC-MemFuse-MC10 — 10-way MCQ episodic memory over MSC dialogs."""

    name = "msc_memfuse"
    published = True
    description = "10-option MCQ episodic memory QA over multi-session chats."
    splits = SPLITS
    task_type = "mcq"
    isolation_unit = "question"
    links = [
        {"label": "HuggingFace", "url": "https://huggingface.co/datasets/Percena/msc-memfuse-mc10"},
    ]

    def _data_path(self) -> Path:
        import os
        env = os.environ.get("MSC_MEMFUSE_DATA_PATH")
        if env:
            return Path(env)
        cache = dataset_cache_dir("msc_memfuse")
        path = cache / _FILE
        if not path.exists():
            import urllib.request
            print(f"Downloading MSC-MemFuse-MC10 (~5MB) to {path}…")
            urllib.request.urlretrieve(_URL, path)
        return path

    def _load_rows(self) -> list[dict]:
        with open(self._data_path(), encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    @staticmethod
    def _format_session(turns: list[dict]) -> str:
        parts: list[str] = []
        for t in turns:
            content = (t.get("content") or "").strip()
            if content:
                parts.append(f"[{(t.get('role') or '?').upper()}] {content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def build_rag_prompt(self, query: str, context: str, task_type: str, split: str, category: str | None = None, meta: dict | None = None) -> str:
        if task_type == "mcq":
            return (
                f"The following is the user's memory/history:\n\n{context}\n\n"
                f"{query}\n\n"
                f"Answer with only the letter of the correct option, one of (a) through (j)."
            )
        from .base import _DEFAULT_OPEN_PROMPT
        return _DEFAULT_OPEN_PROMPT.format(context=context, query=query)

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        queries: list[Query] = []
        for row in self._load_rows():
            qid = row["question_id"]
            choices = row["choices"]
            letter = _LETTERS[row["correct_choice_index"]]
            options = "\n".join(
                f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))
            query_text = f"{row['question']}\n\n{options}"
            queries.append(Query(
                id=qid,
                query=query_text,
                gold_ids=[f"{qid}_s{i}" for i in range(len(row["haystack_sessions"]))],
                gold_answers=[letter, row["answer"]],
                user_id=qid,
                meta={
                    "retrieval_query": row["question"],
                    "n_sessions": len(row["haystack_sessions"]),
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
        documents: list[Document] = []
        for row in self._load_rows():
            qid = row["question_id"]
            if user_ids is not None and qid not in user_ids:
                continue
            for i, session in enumerate(row["haystack_sessions"]):
                doc_id = f"{qid}_s{i}"
                if ids is not None and doc_id not in ids:
                    continue
                content = self._format_session(session)
                if not content:
                    continue
                turns = [
                    {"role": t["role"], "content": t["content"]}
                    for t in session
                    if t.get("content", "").strip()
                ]
                documents.append(Document(
                    id=doc_id, content=content, user_id=qid, messages=turns))
        if limit and ids is None:
            documents = documents[:limit]
        return documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="MSC-MemFuse-MC10 dataset stats")
        table.add_column("Questions", justify="right")
        table.add_column("Sessions/q", justify="right")
        table.add_column("Turns/q", justify="right")
        table.add_column("Choices", justify="right")
        rows = self._load_rows()
        ns = [len(r["haystack_sessions"]) for r in rows]
        nt = [sum(len(s) for s in r["haystack_sessions"]) for r in rows]
        table.add_row(
            str(len(rows)),
            f"{min(ns)}-{max(ns)}",
            f"{sum(nt)/len(nt):.1f}",
            "10",
        )
        console.print(table)
