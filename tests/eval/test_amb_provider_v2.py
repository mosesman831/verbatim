"""Durable tests for the AMB provider v2 — SPEC_V8.5 §2
(V85-02.01–02.07).

Pins the new contract:

* turn-list session documents become **turn-level memory** — one
  ``Memory.add(messages=[{speaker, text, at}], session_id=<doc.id>,
  occurred_at=<doc.timestamp>)`` per session, never a whole-document
  blob (the blob path stays only for genuine prose documents);
* ``Memory.search(query, limit=64, as_of=<query_timestamp>)`` — AMB's
  ``k`` recorded, never obeyed;
* delivery expansion walks hits in rank order adding ±``W_r`` session
  neighbors, deduped, never exceeding the metered token budget ``B``;
* render = one AMB ``Document`` per session excerpt, sessions by best
  hit rank, turns in dialogue order, ``[<conversation> · session <n> ·
  <YYYY-MM-DD, Weekday>]`` headers, ``Speaker: text`` lines with image
  captions, ``source_ids=[doc id]``, ``raw_response=None``;
* retrieval is byte-identical at concurrency 1 and 4;
* the provider-side session index (``unit-*.sessions.json`` sidecar)
  makes neighbor expansion independent of engine internals and
  survives ``reset=False`` resume.

No network, no model calls — real ``verbatim.Memory`` stores under
``tmp_path`` plus spy factories for the fallback paths.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from eval.amb import provider as P
from eval.amb._turns import parse_turn_list


# ---------------------------------------------------------------------------
# fixtures + spies
# ---------------------------------------------------------------------------


def _session_doc(
    doc_id="conv-9_session_1",
    user_id="conv-9",
    ts="2023-05-08T13:56:00+00:00",
    context="Conversation between Caroline and Melanie (session_1 of conv-9)",
    turns=None,
):
    turns = turns if turns is not None else [
        {"speaker": "Caroline", "dia_id": "D1:1",
         "text": "Hey Mel! Good to see you! How have you been?"},
        {"speaker": "Melanie", "dia_id": "D1:2",
         "text": "Caroline! I adopted a rescue dog named Biscuit "
                 "last week."},
        {"speaker": "Caroline", "dia_id": "D1:3",
         "text": "That is wonderful news! Dogs are the best."},
        {"speaker": "Melanie", "dia_id": "D1:4",
         "text": "He sleeps on the pottery studio couch.",
         "blip_caption": "a dog sleeping on a couch"},
        {"speaker": "Caroline", "dia_id": "D1:5",
         "text": "I went to a LGBTQ support group yesterday and it "
                 "was so powerful."},
    ]
    return {
        "id": doc_id,
        "content": json.dumps(turns),
        "user_id": user_id,
        "timestamp": ts,
        "context": context,
    }


def _session2():
    return _session_doc(
        doc_id="conv-9_session_2", ts="2023-07-12T16:33:00+00:00",
        context=("Conversation between Caroline and Melanie "
                 "(session_2 of conv-9)"),
        turns=[
            {"speaker": "Melanie", "dia_id": "D2:1",
             "text": "Biscuit won a ribbon at the dog show yesterday."},
            {"speaker": "Caroline", "dia_id": "D2:2",
             "text": "Amazing! Send me the photos."},
            {"speaker": "Melanie", "dia_id": "D2:3",
             "text": "The pottery wheel broke though."},
        ],
    )


class SpyHit:
    def __init__(self, i, quote, kind="source", oid=None):
        self.memory_id = oid or f"src-{i}"
        self.ref = ""
        self.object_ref = (
            f"vobj1.{kind}.{self.memory_id.encode().hex()}.1")
        self.kind = kind
        self.quote = quote
        self.lifecycle = "active"
        self.support_status = "supported"
        self.recorded_time = "2024-01-01T00:00:00+00:00"
        self.score = 1.0 - i * 0.01
        self.warnings = []


class SpyResult:
    def __init__(self, items):
        self.status = "ready"
        self.items = list(items)
        self.warnings = []
        self.readiness = {}
        self.coverage = {"support": {"verdict": "supported"}}
        self.causal_token = ""


class SpyMemory:
    """``Memory``-shaped spy — ``**kwargs`` on add/search means the
    provider's feature-detects pass everything through."""

    def __init__(self, hits=None):
        self.add_calls = []
        self.search_calls = []
        self._hits = hits if hits is not None else [
            SpyHit(0, "Biscuit won a ribbon at the dog show yesterday."),
        ]

    def add(self, content, **kw):
        self.add_calls.append((content, kw))
        return SimpleNamespace(
            memory_id=f"src-{len(self.add_calls) - 1}",
            ref="", receipt_id=f"r{len(self.add_calls) - 1}",
            acceptance="accepted")

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return SpyResult(self._hits)

    def wait_ready(self, receipt, timeout_ms=None):
        return SimpleNamespace(state="ready")

    def close(self):
        pass


class NoMessagesSpy(SpyMemory):
    """A facade without the ``messages`` add-arg — forces the per-turn
    fallback path (still turn-level, never a blob)."""

    def add(self, content, *, speaker=None, session_id=None,
            occurred_at=None, metadata=None, infer=None,
            idempotency_key=None):
        self.add_calls.append((content, {
            "speaker": speaker, "session_id": session_id,
            "occurred_at": occurred_at, "metadata": metadata,
            "idempotency_key": idempotency_key,
        }))
        return SimpleNamespace(
            memory_id=f"src-{len(self.add_calls) - 1}",
            ref="", receipt_id=f"r{len(self.add_calls) - 1}",
            acceptance="accepted")


def _spy_factory(spies, cls=SpyMemory, **kw):
    def make(path, unit_key):
        mem = cls(**kw)
        spies[unit_key] = mem
        return mem
    return make


# ---------------------------------------------------------------------------
# V85-02.02 — turn ingest
# ---------------------------------------------------------------------------


def test_turn_list_ingest_emits_turn_units_not_blob(tmp_path):
    """A JSON turn-list Document becomes ONE ``add(messages=...)`` whose
    projection emits one ``turn`` unit per message with real seq —
    never a whole-document blob (V85-02.02)."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    rep = prov.ingest([_session_doc()])
    assert rep["add_errors"] == 0
    assert rep["turn_documents"] == 1
    assert rep["blob_documents"] == 0
    assert rep["turns_indexed"] == 5
    assert rep["ingest_path"] == "messages"

    mem = prov.banks()["conv-9"]
    with mem._store.read() as conn:
        rows = conn.execute(
            "SELECT unit_id, kind, session_id, seq, speaker_canon"
            " FROM units WHERE kind='turn' ORDER BY seq"
        ).fetchall()
    assert len(rows) == 5                      # 5 turn units, not a blob
    assert {r[3] for r in rows} == {0, 1, 2, 3, 4}   # real ordinals
    assert all(r[2] == "conv-9_session_1" for r in rows)
    assert [r[4] for r in rows] == [
        "caroline", "melanie", "caroline", "melanie", "caroline"]

    # the provider-side session index attached every turn's unit_id
    idx = prov._indexes["conv-9"]
    sess = idx.session("conv-9_session_1")
    assert sess["units_attached"] is True
    assert all(t["unit_id"] for t in sess["turns"])
    assert sess["turns"][0]["dia_id"] == "D1:1"
    prov.cleanup()


def test_dia_id_persisted_in_revision_metadata(tmp_path):
    """``dia_id`` rides inside the message dicts → revision
    ``metadata_json`` (the store's unit-metadata channel) AND the
    provider session index (V85-02.02)."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    mem = prov.banks()["conv-9"]
    with mem._store.read() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM source_revisions LIMIT 1"
        ).fetchone()
    meta = json.loads(row[0])
    assert meta["messages"][0]["dia_id"] == "D1:1"
    assert meta["messages"][3]["blip_caption"] == "a dog sleeping on a couch"
    idx = prov._indexes["conv-9"]
    assert idx.session("conv-9_session_1")["turns"][2]["dia_id"] == "D1:3"
    prov.cleanup()


def test_session_index_sidecar_written(tmp_path):
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    prov.cleanup()
    files = list(tmp_path.glob("unit-*.sessions.json"))
    assert len(files) == 1
    blob = json.loads(files[0].read_text())
    assert blob["schema"] == "amb_session_index/v1"
    sess = blob["sessions"]["conv-9_session_1"]
    assert len(sess["turns"]) == 5
    assert sess["turns"][0]["speaker"] == "Caroline"


def test_per_turn_fallback_without_messages_param(tmp_path):
    """A facade lacking ``messages`` gets per-turn adds sharing
    ``session_id=<doc.id>`` — still turn-level, never a blob."""
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path,
        memory_factory=_spy_factory(spies, cls=NoMessagesSpy))
    prov.prepare(tmp_path)
    rep = prov.ingest([_session_doc()])
    assert rep["add_errors"] == 0
    mem = spies["conv-9"]
    assert len(mem.add_calls) == 5            # one add per turn
    for _content, kw in mem.add_calls:
        assert kw["session_id"] == "conv-9_session_1"
        assert kw["occurred_at"] == "2023-05-08T13:56:00+00:00"
    assert mem.add_calls[1][1]["speaker"] == "Melanie"
    assert mem.add_calls[1][1]["metadata"]["amb_dia_id"] == "D1:2"
    prov.cleanup()


def test_prose_document_keeps_blob_path(tmp_path):
    """The blob path is forbidden only for turn-list payloads — a
    genuine prose document stays a single ``add`` (V85-02.02)."""
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies))
    prov.prepare(tmp_path)
    rep = prov.ingest([{
        "id": "d-prose", "content": "Caroline adopted a dog.",
        "user_id": "u1", "timestamp": "2024-01-05T09:00:00Z"}])
    assert rep["add_errors"] == 0
    assert rep["blob_documents"] == 1
    (content, kw), = spies["u1"].add_calls
    assert content == "Caroline adopted a dog."
    assert "messages" not in kw
    prov.cleanup()


# ---------------------------------------------------------------------------
# V85-02.03 — query
# ---------------------------------------------------------------------------


def test_search_uses_engine_cap_limit_and_as_of(tmp_path):
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path, memory_factory=_spy_factory(spies))
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    prov.retrieve("dog", k=10, user_id="conv-9",
                  query_timestamp="2023-07-17T14:31:00+00:00")
    (_q, kwargs), = spies["conv-9"].search_calls
    assert kwargs["limit"] == 64          # engine cap, not AMB's k=10
    assert kwargs["as_of"] == "2023-07-17T14:31:00+00:00"
    rec = prov.query_records[-1]
    assert rec["k"] == 10                  # recorded, never obeyed
    assert rec["search_limit"] == 64
    prov.cleanup()


# ---------------------------------------------------------------------------
# V85-02.04/02.05 — expansion, budget, render
# ---------------------------------------------------------------------------


def test_session_excerpt_render_and_source_ids(tmp_path):
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    docs, raw = prov.retrieve(
        "what dog did Melanie adopt", user_id="conv-9",
        query_timestamp="2023-07-17T00:00:00Z")
    assert raw is None                              # V85-02.05 critical
    assert docs, "expected at least one session excerpt"
    d = docs[0]
    assert d.id == "conv-9_session_1"
    assert d.source_ids == ["conv-9_session_1"]
    head, *lines = d.content.splitlines()
    assert head == ("[Conversation between Caroline and Melanie "
                    "· session 1 · 2023-05-08, Monday]")
    # turns keep dialogue order, Speaker: text, image caption inline
    assert any(l.startswith("Caroline:") for l in lines)
    assert "[image: a dog sleeping on a couch]" in d.content
    prov.cleanup()


def test_sessions_ordered_by_best_hit_rank(tmp_path):
    """Two sessions: the one whose hit ranked first ships first; turns
    inside each excerpt stay in dialogue order."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc(), _session2()])
    docs, _ = prov.retrieve("dog show ribbon", user_id="conv-9")
    ids = [d.id for d in docs]
    assert "conv-9_session_2" in ids
    assert ids[0] == "conv-9_session_2"     # best hit's session first
    s2 = docs[0]
    body = s2.content.splitlines()[1:]
    assert body[0].startswith("Melanie: Biscuit won a ribbon")
    prov.cleanup()


def test_neighbor_window_arms(tmp_path):
    """``W_r=0`` delivers only the hit's covered turns; ``W_r=1`` adds
    one neighbor each side, ``W_r=2`` two (arm {0,1,2}, V85-02.04).

    A spy hit with a source-level ref and a quote contained in exactly
    one turn resolves to ``{3}`` deterministically — expansion width
    then depends only on the neighbor radius."""
    hits = [SpyHit(0, "pottery studio couch", oid="src-0")]
    counts = {}
    for w in (0, 1, 2):
        d = tmp_path / f"w{w}"
        prov = P.VerbatimAMBProvider(
            store_dir=d, neighbor_w=w,
            memory_factory=_spy_factory({}, hits=hits))
        prov.prepare(d)
        prov.ingest([_session_doc()])
        docs, _ = prov.retrieve("pottery", user_id="conv-9")
        counts[w] = sum(len(d.content.splitlines()) - 1 for d in docs)
        prov.cleanup()
    assert counts[0] == 1      # {3}
    assert counts[1] == 3      # {2,3,4}
    assert counts[2] == 4      # {1,2,3,4} — clamped at session end


def test_budget_boundary_never_exceeded(tmp_path):
    """``est_context_tokens`` (metered over the delivered AMB context
    string) never exceeds ``B`` (V85-02.04)."""
    big_turns = [
        {"speaker": "A", "text": f"turn {i} " + "padding " * 40}
        for i in range(12)
    ]
    prov = P.VerbatimAMBProvider(store_dir=tmp_path, token_budget=120)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc(turns=big_turns)])
    docs, _ = prov.retrieve("padding turn", user_id="conv-9")
    rec = prov.query_records[-1]
    assert rec["est_context_tokens"] <= 120
    ctx = P.context_string(docs)
    assert prov._meter(ctx) <= 120
    prov.cleanup()


def test_unbounded_budget_delivers_all_hit_sessions(tmp_path):
    prov = P.VerbatimAMBProvider(store_dir=tmp_path, token_budget=None)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    docs, _ = prov.retrieve("dog", user_id="conv-9")
    assert docs
    assert prov.query_records[-1]["token_budget"] is None
    prov.cleanup()


def test_unresolved_hit_delivered_as_itself(tmp_path):
    """A hit that maps to no session is never silently dropped — it
    renders as a ``verbatim-hit-*`` document with its quote."""
    spies = {}
    prov = P.VerbatimAMBProvider(
        store_dir=tmp_path,
        memory_factory=_spy_factory(
            spies, hits=[SpyHit(0, "orphan evidence", oid="src-x")]))
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    docs, _ = prov.retrieve("orphan", user_id="conv-9")
    assert docs and docs[0].id == "verbatim-hit-00000"
    assert "orphan evidence" in docs[0].content
    rec = prov.query_records[-1]
    assert rec["resolution"]["unresolved"] == 1
    prov.cleanup()


# ---------------------------------------------------------------------------
# V85-02.06 — concurrency determinism
# ---------------------------------------------------------------------------


def test_retrieval_byte_identical_concurrency_1_and_4(tmp_path):
    """Same query set through a 1-thread and 4-thread executor →
    byte-identical delivered documents (V85-02.06)."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path, concurrency=4)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc(), _session2()])
    queries = [
        "what dog did Melanie adopt",
        "dog show ribbon",
        "support group",
        "pottery studio",
    ]

    def run(workers):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(
                lambda q: prov.retrieve(
                    q, k=10, user_id="conv-9",
                    query_timestamp="2023-07-17T00:00:00Z"),
                queries))

    seq = run(1)
    par = run(4)
    assert len(seq) == len(par) == len(queries)
    for (d1, r1), (d2, r2) in zip(seq, par):
        assert r1 is None and r2 is None
        assert [d.id for d in d1] == [d.id for d in d2]
        assert [d.content for d in d1] == [d.content for d in d2]
    prov.cleanup()


def test_concurrency_env_default():
    prov = P.VerbatimAMBProvider()
    assert prov.concurrency == 4            # env default
    os.environ["VERBATIM_AMB_CONCURRENCY"] = "7"
    try:
        assert P._env_concurrency() == 7
    finally:
        del os.environ["VERBATIM_AMB_CONCURRENCY"]
    prov2 = P.VerbatimAMBProvider(concurrency=2)
    assert prov2.concurrency == 2


# ---------------------------------------------------------------------------
# resume + records + manifest surface
# ---------------------------------------------------------------------------


def test_resume_reload_keeps_expansion(tmp_path):
    """``reset=False`` reloads the persisted session index — neighbor
    expansion still works without re-ingest (V85-02.02 sidecar)."""
    prov = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    prov.cleanup()

    prov2 = P.VerbatimAMBProvider(store_dir=tmp_path)
    prov2.prepare(tmp_path, reset=False)
    idx = prov2._indexes.get("conv-9")
    assert idx is not None
    assert len(idx.session("conv-9_session_1")["turns"]) == 5
    # bank opens lazily on the same store — search + expand still work
    docs, _ = prov2.retrieve("pottery studio", user_id="conv-9")
    assert docs
    assert docs[0].source_ids == ["conv-9_session_1"]
    prov2.cleanup()


def test_records_and_manifest_fields(tmp_path):
    prov = P.VerbatimAMBProvider(store_dir=tmp_path, token_budget=4500,
                               neighbor_w=2, search_limit=64)
    prov.prepare(tmp_path)
    prov.ingest([_session_doc()])
    docs, _ = prov.retrieve("dog", k=7, user_id="conv-9",
                            query_timestamp="2023-07-17T00:00:00Z")
    rec = prov.query_records[-1]
    assert rec["provider_version"] == "verbatim-amb/2"
    assert rec["token_budget"] == 4500
    assert rec["neighbor_w"] == 2
    assert rec["token_meter"]
    assert rec["as_of"]["applied"] is True
    assert rec["expansion"]["turns_delivered"] >= 1
    curve = prov.token_curve()
    assert curve["token_budget"] == 4500
    assert curve["est_context_tokens"] == [rec["est_context_tokens"]]
    fields = prov.manifest_fields()
    assert fields["provider_revision"] == "verbatim-amb/2"
    assert fields["token_budget"] == 4500
    assert fields["neighbor_w"] == 2
    assert fields["search_limit"] == 64
    patches = P.harness_patch_records()
    assert len(patches["patches"]) == 3     # the three amb_patches.md
    assert patches["doc_sha256"]            # pins the patch document
    prov.cleanup()


def test_turn_parser_contract():
    """The session-doc parser accepts the AMB LoCoMo shape and rejects
    non-dialogue JSON."""
    turns = parse_turn_list(json.dumps([
        {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
        {"speaker": "B", "dia_id": "D1:2", "text": "hello"},
    ]))
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert turns[0]["dia_id"] == "D1:1"
    assert parse_turn_list("plain prose, not json") is None
    assert parse_turn_list(json.dumps([1, 2, 3])) is None
    assert parse_turn_list(json.dumps({"a": 1})) is None
