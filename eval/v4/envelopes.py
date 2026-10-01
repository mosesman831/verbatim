"""SPEC_V4 §56 workload-envelope measurement harness (V4-56.*).

Measures the *real public path* end-to-end — ``ingest_envelope`` capture →
``Ingester.drain_report`` job drain → ``VerbatimV3.recall`` (which runs
``governance.authorize`` + ``retrieval.v3.recall_v3``) — on a real
``Store.create`` database. No lane is simulated; every number here is a
``time.perf_counter_ns`` sample around the production call.

Envelope definitions (SPEC_V4 §56 table):

* **S0** — 100 claims / 500 spans / 10 episodes / 0 procedures, no neural
  artifacts (``embedding.backend="none"``). One recall client, one capture
  client, one drain. May qualify with 1,000 measured queries (V4-56.01).
* **S1** — 1K claims / 5K spans / 100 episodes / 20 procedures. Four recall
  clients, two captures/s, one drain. MUST use >= 10,000 measured queries.

Honesty contract (V4-55.09/55.10, V4-56.01/02/06):

* A declared warmup runs per restart repetition and is EXCLUDED from the
  reported samples (the count is disclosed, not folded in).
* Restart repetitions close and reopen the ``Store`` between repetitions;
  each repetition is reported beside the pooled distribution.
* A missed target is reported with ``met: false`` — never hidden. Latency
  samples include failed queries' elapsed time; the failure count is
  reported beside them.
* S1's ≥10% historical revisions / 5% holds / heterogeneous scopes mix is
  seeded through the real transition/quarantine paths and the realized
  percentages are disclosed in ``mix``.
* ``hashing`` is a deterministic pure-stdlib encoder (no neural artifact);
  S1 provisions it so the semantic lane and ``semantic_ready`` obligations
  are real measured work rather than CAPABILITY_UNAVAILABLE slots. S0 runs
  ``backend="none"`` per the no-model envelope.

CLI: ``python -m eval.v4.envelopes --envelope s0|s1 --out <dir>`` writes
``envelope_<id>_report.json`` and ``envelope_<id>_report.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import random
import resource
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# spec table (SPEC_V4 §56)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeSpec:
    """One §56 envelope: evidence volumes, concurrent load, and targets.

    ``targets`` maps a report metric path to its millisecond bound; every
    declared target is evaluated into ``report["targets"]`` — a miss is
    data, not a crash (V4-56.11).
    """

    envelope: str
    claims: int
    spans: int
    episodes: int
    procedures: int
    recall_clients: int
    captures_per_s: float
    drains: int
    queries: int
    warmup: int
    restarts: int
    embedding_backend: str
    capture_payload_bytes: int
    scopes: int
    historical_revision_pct: float
    hold_pct: float
    targets: dict[str, float]
    readiness_deadline_s: float = 30.0
    note: str = ""


#: S0 — the M0 no-model envelope. "One capture" client is paced at 2/s like
#: S1 (the spec gives no S0 rate; the declared rate is disclosed).
S0_SPEC = EnvelopeSpec(
    envelope="s0",
    claims=100,
    spans=500,
    episodes=10,
    procedures=0,
    recall_clients=1,
    captures_per_s=2.0,
    drains=1,
    queries=1000,
    warmup=50,
    restarts=2,
    embedding_backend="none",
    capture_payload_bytes=4096,
    scopes=1,
    historical_revision_pct=0.0,
    hold_pct=0.0,
    targets={
        "recall.p95_ms": 25.0,
        "recall.p99_ms": 75.0,
        "capture.ack_p95_ms": 50.0,
    },
    note="M0 no-model envelope: rules recall only, embeddings unprovisioned.",
)

#: S1 — the first sustained-load envelope.
S1_SPEC = EnvelopeSpec(
    envelope="s1",
    claims=1000,
    spans=5000,
    episodes=100,
    procedures=20,
    recall_clients=4,
    captures_per_s=2.0,
    drains=1,
    queries=10000,
    warmup=50,
    restarts=2,
    embedding_backend="hashing",
    capture_payload_bytes=4096,
    scopes=2,
    historical_revision_pct=0.10,
    hold_pct=0.05,
    targets={
        "recall.p95_ms": 25.0,
        "recall.p99_ms": 75.0,
        "capture.ack_p95_ms": 50.0,
        "capture.ack_p95_ms_ingest": 100.0,
        "readiness.lexical_p95_ms": 2000.0,
        "readiness.semantic_p95_ms": 10000.0,
    },
    note=(
        "Sustained load: 4 recall clients, 2 captures/s, 1 drain; "
        "deterministic hashing encoder provisions the semantic lane."
    ),
)

SPECS = {"s0": S0_SPEC, "s1": S1_SPEC}

_CALLER = "envelope-agent"
_EVALUATOR = "envelope-harness"


# ---------------------------------------------------------------------------
# percentiles + environment helpers (pure functions — unit-tested)
# ---------------------------------------------------------------------------


def percentiles(samples: Iterable[float]) -> dict[str, Any]:
    """Nearest-rank percentiles over ``samples`` (ms).

    p95 of n samples is ``sorted[ceil(0.95 n) - 1]`` — the classic SLO
    convention; ``n`` is always reported beside the figures (V4-55.10).
    """
    s = sorted(float(x) for x in samples)
    n = len(s)
    if n == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "max_ms": None, "mean_ms": None}

    def _pct(q: float) -> float:
        import math

        return s[min(n - 1, max(0, math.ceil(q * n) - 1))]

    return {
        "n": n,
        "p50_ms": _pct(0.50),
        "p95_ms": _pct(0.95),
        "p99_ms": _pct(0.99),
        "max_ms": s[-1],
        "mean_ms": sum(s) / n,
    }


def _loadavg() -> Optional[list[float]]:
    try:
        return [round(x, 3) for x in os.getloadavg()]
    except OSError:
        return None


def _environment(db_path: Optional[str] = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if db_path and os.path.exists(db_path):
        env["db_bytes"] = os.path.getsize(db_path)
        wal = db_path + "-wal"
        env["wal_bytes"] = os.path.getsize(wal) if os.path.exists(wal) else 0
    return env


# ---------------------------------------------------------------------------
# deterministic corpus generator
# ---------------------------------------------------------------------------

_SERVICES = [
    "billing", "auth", "search", "ingest", "ledger", "notify", "cache",
    "router", "scheduler", "indexer",
]
_COMPONENTS = [
    "deploy", "failover", "backup", "migration", "rollout", "compaction",
    "replication", "throttling", "gc", "snapshot",
]
_PATHS = [
    "src/engine/recall.py", "src/storage/wal.py", "src/net/balancer.py",
    "tools/runbooks/deploy.md", "configs/prod.yaml", "src/jobs/drain.py",
]
_TOOLS = ("read_file", "grep", "apply_patch", "run_check")


class _Corpus:
    """Seeded UTF-8 claim/note generator.

    ``make_claim(i)`` returns ``(text, meta)`` where meta carries the
    identifier tokens, a paraphrase query, and entity ids the query-mix
    builder draws from. Texts mix plain sentences, hard identifiers, and
    code-ish lines; every claim-bearing text opens with a ``remember``
    stem so admission activates it under ``require_review=False`` — the
    seeded corpus needs no synthetic operator pass.
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def make_claim(self, i: int) -> tuple[str, dict[str, Any]]:
        r = self.rng
        svc = _SERVICES[i % len(_SERVICES)]
        comp = _COMPONENTS[i % len(_COMPONENTS)]
        path = _PATHS[i % len(_PATHS)]
        tag = f"v{r.randint(1, 4)}.{r.randint(0, 9)}.{r.randint(0, 9)}"
        build = f"build-{r.randint(1, 997)}"
        ttl = r.randint(30, 3600)
        ticket = f"OPS-{r.randint(100, 9999)}"
        stem = r.choice(
            ("remember that ", "please remember: ", "remember: ",
             "remember this: ")
        )
        form = i % 4
        if form == 0:
            text = (
                f"{stem}the {svc} {comp} runbook pins {build} and tag {tag}; "
                f"ticket {ticket}, cache ttl {ttl}s, owner file {path}"
            )
            paraphrase = f"how is the {svc} {comp} done"
        elif form == 1:
            text = (
                f"{stem}`def run_{comp}_{i}(ctx)` in {path} drives the "
                f"{svc} {comp}; deploy tag {tag}, see {ticket}"
            )
            paraphrase = f"which function handles {svc} {comp}"
        elif form == 2:
            text = (
                f"{stem}release {tag} moved the {svc} endpoint; {comp} "
                f"runs every {ttl} seconds under {build} ({ticket})"
            )
            paraphrase = f"current {svc} release and {comp} cadence"
        else:
            text = (
                f"{stem}for {ticket} the {svc} team keeps {comp} on "
                f"{build}; rollback doc lives at {path} (tag {tag})"
            )
            paraphrase = f"{svc} {comp} rollback instructions"
        meta = {
            "id_terms": [build, tag, ticket],
            "entities": [svc, comp],
            "paraphrase": paraphrase,
            "path": path,
        }
        return text, meta

    def make_note(self, i: int) -> str:
        """Agent-authored span filler — captured, never claim-derived."""
        r = self.rng
        return (
            f"working note {i}: scanned {_PATHS[i % len(_PATHS)]} while "
            f"checking {_SERVICES[r.randrange(len(_SERVICES))]} "
            f"{_COMPONENTS[r.randrange(len(_COMPONENTS))]} logs; nothing "
            f"actionable, marker {uuid.uuid5(uuid.NAMESPACE_DNS, f'n{i}').hex[:8]}"
        )

    def make_tool_op(self, i: int) -> dict[str, Any]:
        tool = _TOOLS[i % len(_TOOLS)]
        if tool == "read_file":
            args = {"path": _PATHS[i % len(_PATHS)]}
        elif tool == "grep":
            args = {"pattern": f"def run_{i}", "path": "src"}
        elif tool == "apply_patch":
            args = {"path": _PATHS[i % len(_PATHS)],
                    "patch": f"@@ -{i} +{i} @@ fix {i}"}
        else:
            args = {"argv": ["pytest", f"tests/test_{i}.py", "-x"]}
        return {"tool": tool, "args": args}

    def make_capture_text(self, i: int, nbytes: int) -> str:
        """A measured-load capture body ~= ``nbytes`` UTF-8.

        Multi-paragraph on purpose: the v3 harvester drops prose segments
        above ``max_len`` (1200B), so a monolithic 4KiB blob yields no
        candidates — no claim, and the receipt's ``lexical_ready`` would
        settle ``deferred`` rather than measuring the real admit→index
        path. Paragraphs keep the byte count honest AND harvestable.
        """
        out: list[str] = []
        total = 0
        p = 0
        while total < nbytes:
            piece = (
                f"remember that live capture {i} part {p} for "
                f"{_SERVICES[(i + p) % len(_SERVICES)]} records marker "
                f"{uuid.uuid5(uuid.NAMESPACE_DNS, f'cap{i}.{p}').hex} "
                f"with {_COMPONENTS[(i + p) % len(_COMPONENTS)]} notes"
            )
            out.append(piece)
            total += len(piece.encode("utf-8")) + 2
            p += 1
        return "\n\n".join(out)[:nbytes]

    def make_note_capture(self, i: int, nbytes: int) -> str:
        """4KiB-ish agent-authored note — the bulk of the measured capture
        stream: real write-path work (envelope + span + screen + DAG) but
        no claim derivation, so the envelope's declared claim volume is
        not inflated mid-run."""
        base = f"live note {i}: observation filler "
        pad = nbytes - len(base.encode("utf-8"))
        chunk = uuid.uuid5(uuid.NAMESPACE_DNS, f"capnote{i}").hex
        return base + (chunk * ((pad // len(chunk)) + 1))[:pad]


# ---------------------------------------------------------------------------
# engine plumbing (real public paths only)
# ---------------------------------------------------------------------------


def _config(spec: EnvelopeSpec):
    from verbatim.config import config_from_mapping

    return config_from_mapping(
        {
            "mode": "offline_rules",
            "capture": {
                "enabled": True,
                "user_messages": True,
                "tool_outputs": True,
            },
            # Explicit-remember texts admit ACTIVE on drain — the corpus
            # needs no synthetic operator pass; the flag is a shipped
            # config knob (disclosed in the report).
            "admission": {"require_review": False},
            "embedding": {"backend": spec.embedding_backend},
        }
    )


class _EngineBundle:
    """Store + Engine + VerbatimV3 facade + Ingester over one DB path."""

    def __init__(self, path: str, cfg: Any, host: Any, *, create: bool) -> None:
        from verbatim.api import Engine
        from verbatim.api_v3.facade import VerbatimV3
        from verbatim.embeddings.encoder import get_encoder
        from verbatim.storage.store import Store

        self.path = path
        self.store = (
            Store.create(path) if create else Store.open(path)
        )
        encoder = get_encoder(cfg)
        self.engine = Engine(self.store, cfg, host, encoder=encoder)
        self.ingester = self.engine._ingester
        self.facade = VerbatimV3(self.store, cfg)

    def close(self) -> None:
        self.store.close()


def _capture(
    bundle: _EngineBundle,
    scope_id: str,
    text: str,
    *,
    kind: Any,
    external_id: str,
    metadata: Optional[dict[str, Any]] = None,
    proof: Optional[str] = None,
    event_us: int = 0,
) -> Any:
    """One real capture: ``ingest_envelope`` + readiness DAG materialization
    in the SAME write tx — exactly what ``api_v3.capture_submitted`` does
    after its grant checks (V4-14.01/14.02)."""
    from verbatim.core.types_v3 import (
        EnvelopeKind,
        Perspective,
        SourceEnvelopeV3,
    )
    from verbatim.evidence import ingest_envelope
    from verbatim.readiness import ReadinessEngine

    env = SourceEnvelopeV3(
        kind=kind if isinstance(kind, EnvelopeKind) else EnvelopeKind(kind),
        scope_id=scope_id,
        actor_principal="envelope-corpus",
        perspective=Perspective(asserter="envelope-corpus"),
        event_us=event_us,
        receipt_us=0,
        content=text.encode("utf-8"),
        media_type="text/plain",
        host_id="envelope-harness",
        external_id=external_id,
        capture_proof=proof,
        metadata=metadata or {},
    )
    with bundle.store.tx() as conn:
        receipt = ingest_envelope(conn, bundle.store, env)
        ReadinessEngine(bundle.store).ensure_for_source(
            conn, receipt.source_id, int(receipt.revision)
        )
    return receipt


def _drain(ingester: Any, limit: int = 512, max_rounds: int = 200) -> dict:
    """Loop ``drain_report`` until the queue is empty (or rounds run out);
    returns the summed honest breakdown plus the last report's backlog."""
    totals = {"processed": 0, "succeeded": 0, "failed": 0,
              "deferred": 0, "expired": 0}
    last: dict[str, Any] = {}
    for _ in range(max_rounds):
        rep = ingester.drain_report(limit=limit)
        last = rep
        for k in totals:
            totals[k] += int(rep.get(k, 0))
        if rep.get("processed", 0) == 0:
            break
    totals["still_pending"] = last.get("still_pending")
    totals["pending_obligations"] = last.get("pending_obligations")
    return totals


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def _provision_auth(bundle: _EngineBundle, scope_ids: list[str]) -> str:
    """Purposes + recall grants for the eval caller + capture consent for
    the corpus principal — the real governance rows every surface checks."""
    from verbatim.governance import (
        create_grant,
        issue_capture_authorization,
        register_principal,
        seed_purposes,
    )

    with bundle.store.tx() as conn:
        seed_purposes(conn)
        register_principal(conn, kind="agent", principal_id=_CALLER)
        register_principal(conn, kind="human", principal_id="envelope-corpus")
        for sid in scope_ids:
            create_grant(
                conn,
                scope_id=sid,
                principal_id=_CALLER,
                verbs={"read", "quote"},
                issuer_id=_EVALUATOR,
                purposes=["recall"],
            )
        return issue_capture_authorization(
            conn,
            principal_id="envelope-corpus",
            issuer_id="envelope-corpus",
            allowed_kinds=[
                "agent_note", "lesson", "tool_call", "tool_result",
            ],
            retention_policy="envelope-corpus",
            policy_revision="env-pol-1",
            scope_ids=scope_ids,
        )


def _seed_procedure_episode(
    bundle: _EngineBundle,
    scope_id: str,
    idx: int,
    corpus: _Corpus,
    proof: str,
    seq: list[int],
) -> tuple[Optional[str], Optional[str]]:
    """One coding trajectory → episode → transitions → ``compile_episode``.

    Returns ``(episode_id, procedure_id_or_status)``. Every step is a real
    ``tool_call`` envelope (consented under ``proof``); the last step carries
    a ``test_result`` checker observation — the §20/§22 producer chain, no
    table shortcuts.
    """
    from verbatim.core.types import new_id
    from verbatim.core.types_v3 import (
        EnvelopeKind,
        TrajectoryRecord,
        TrajectoryStep,
    )
    from verbatim.evidence import trajectories as traj
    from verbatim.experience import episodes_v3
    from verbatim.experience.episodes import close_episode
    from verbatim.experience.transitions import build_transitions
    from verbatim.procedures import compile_episode

    trajectory_id = f"traj:env{idx}:{new_id()[:12]}"
    goal = f"fix-tests-{idx}"
    store = bundle.store

    with store.tx() as conn:
        traj.record_trajectory(
            conn,
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                scope_id=scope_id,
                host_id="envelope-harness",
                task_id=f"task-{idx}",
                metadata={"goal_class": goal, "label": goal},
            ),
            store=store,
        )

    ops = [corpus.make_tool_op(idx * 4 + j) for j in range(4)]
    for i, op in enumerate(ops):
        r = _capture(
            bundle,
            scope_id,
            f"call {op['tool']} step {i} for {goal}",
            kind=EnvelopeKind.TOOL_CALL,
            external_id=f"proc{idx}-op{i}",
            metadata={"tool": op["tool"], "args": op["args"]},
            proof=proof,
            event_us=seq[0],
        )
        seq[0] += 1
        obs: tuple[str, ...] = ()
        if i == len(ops) - 1:
            cr = _capture(
                bundle,
                scope_id,
                f"pytest passed for {goal}: 4 tests, 0 failures",
                kind=EnvelopeKind.TEST_RESULT,
                external_id=f"proc{idx}-check",
                metadata={
                    "checker_receipt": {
                        "checker_id": "pytest",
                        "outcome": "success",
                    }
                },
                event_us=seq[0],
            )
            seq[0] += 1
            obs = (cr.envelope_id,)
        with store.tx() as conn:
            traj.add_step(
                conn,
                TrajectoryStep(
                    step_id=f"step:{new_id()[:12]}",
                    trajectory_id=trajectory_id,
                    ord=i,
                    action_envelope_id=r.envelope_id,
                    observation_envelope_ids=obs,
                ),
            )

    with store.tx() as conn:
        traj.complete(conn, trajectory_id, store=store)
        episode_id = episodes_v3.build_episode(conn, trajectory_id)
        build_transitions(conn, episode_id)
        close_episode(store, conn, episode_id)
        result = compile_episode(conn, episode_id, hmac_fn=store.hmac)
    pid = result.procedure_id
    return episode_id, pid or f"not_compiled:{result.status}:{result.reason}"


def _seed_plain_episode(
    bundle: _EngineBundle,
    scope_id: str,
    idx: int,
    corpus: _Corpus,
    proof: str,
    seq: list[int],
) -> str:
    """A non-procedure episode: recorded trajectory (agent-note actions),
    completed, grouped by ``episodes_v3.build_episode`` with transitions —
    the same derivation chain procedure episodes use."""
    from verbatim.core.types import new_id
    from verbatim.core.types_v3 import (
        EnvelopeKind,
        TrajectoryRecord,
        TrajectoryStep,
    )
    from verbatim.evidence import trajectories as traj
    from verbatim.experience import episodes_v3
    from verbatim.experience.episodes import close_episode
    from verbatim.experience.transitions import build_transitions

    trajectory_id = f"traj:env{idx}:{new_id()[:12]}"
    store = bundle.store
    with store.tx() as conn:
        traj.record_trajectory(
            conn,
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                scope_id=scope_id,
                host_id="envelope-harness",
                task_id=f"task-{idx}",
                metadata={"label": f"session-{idx}"},
            ),
            store=store,
        )
    for i in range(2):
        r = _capture(
            bundle,
            scope_id,
            corpus.make_note(idx * 100 + i),
            kind=EnvelopeKind.AGENT_NOTE,
            external_id=f"ep{idx}-note{i}",
            proof=proof,
            event_us=seq[0],
        )
        seq[0] += 1
        with store.tx() as conn:
            traj.add_step(
                conn,
                TrajectoryStep(
                    step_id=f"step:{new_id()[:12]}",
                    trajectory_id=trajectory_id,
                    ord=i,
                    action_envelope_id=r.envelope_id,
                ),
            )
    with store.tx() as conn:
        traj.complete(conn, trajectory_id, store=store)
        episode_id = episodes_v3.build_episode(conn, trajectory_id)
        build_transitions(conn, episode_id)
        close_episode(store, conn, episode_id)
    return episode_id


def _seed(
    bundle: _EngineBundle,
    spec: EnvelopeSpec,
    corpus: _Corpus,
    scope_ids: list[str],
    notes: list[str],
) -> dict[str, Any]:
    """Seed the envelope volumes through the real write path.

    Order: governance → procedure/plain episodes (their envelopes add
    spans) → claim envelopes → drain → span top-up with non-harvested
    agent notes → historical revisions + holds (the S1+ mix) → counts.
    """
    from verbatim.core.types import TransitionCommand
    from verbatim.core.types_v3 import EnvelopeKind

    store = bundle.store
    proof = _provision_auth(bundle, scope_ids)
    seq = [1_700_000_000_000_000]

    # --- procedures (each also builds one episode) -------------------------
    procedure_ids: list[str] = []
    episode_ids: list[str] = []
    proc_failures: list[str] = []
    for i in range(spec.procedures):
        sid = scope_ids[i % len(scope_ids)]
        try:
            ep, pid = _seed_procedure_episode(
                bundle, sid, i, corpus, proof, seq
            )
            episode_ids.append(ep)
            if pid and not str(pid).startswith("not_compiled"):
                procedure_ids.append(pid)
            else:
                proc_failures.append(str(pid))
        except Exception as exc:  # honest: record, keep seeding
            proc_failures.append(f"{type(exc).__name__}: {exc}")
    if proc_failures:
        notes.append(f"procedure compile failures: {proc_failures[:5]}")

    for i in range(spec.episodes - spec.procedures):
        sid = scope_ids[i % len(scope_ids)]
        try:
            episode_ids.append(
                _seed_plain_episode(
                    bundle, sid, spec.procedures + i, corpus, proof, seq
                )
            )
        except Exception as exc:
            notes.append(
                f"episode {i} seed failed: {type(exc).__name__}: {exc}"
            )

    # --- claim-bearing envelopes (batched per tx, still one logical
    # capture each — ingest_envelope is the caller-tx API) ---------------
    claim_meta: list[dict[str, Any]] = []
    n_claim_env = spec.claims
    batch: list[tuple[str, str, dict[str, Any]]] = []
    for i in range(n_claim_env):
        text, meta = corpus.make_claim(i)
        sid = scope_ids[i % len(scope_ids)]
        batch.append((sid, text, meta))
        meta["scope_id"] = sid
        claim_meta.append(meta)
    # commit in chunks so no single tx holds thousands of statements
    CH = 200
    for off in range(0, len(batch), CH):
        chunk = batch[off : off + CH]
        with store.tx() as conn:
            from verbatim.core.types_v3 import (
                Perspective,
                SourceEnvelopeV3,
            )
            from verbatim.evidence import ingest_envelope
            from verbatim.readiness import ReadinessEngine

            reng = ReadinessEngine(store)
            for j, (sid, text, _m) in enumerate(chunk):
                env = SourceEnvelopeV3(
                    kind=EnvelopeKind.USER_MESSAGE,
                    scope_id=sid,
                    actor_principal="envelope-corpus",
                    perspective=Perspective(asserter="envelope-corpus"),
                    event_us=seq[0],
                    receipt_us=0,
                    content=text.encode("utf-8"),
                    media_type="text/plain",
                    host_id="envelope-harness",
                    external_id=f"claim-{off + j}",
                    metadata={},
                )
                seq[0] += 1
                receipt = ingest_envelope(conn, store, env)
                reng.ensure_for_source(
                    conn, receipt.source_id, int(receipt.revision)
                )

    drain0 = _drain(bundle.ingester)

    # --- span top-up: agent-authored notes are captured (1 span each)
    # but never harvested into claims (V3-13.11) --------------------------
    def _count(t: str) -> int:
        with store.read() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])

    span_now = _count("spans")
    filler = max(0, spec.spans - span_now)
    for off in range(0, filler, CH):
        chunk = range(off, min(off + CH, filler))
        with store.tx() as conn:
            from verbatim.core.types_v3 import (
                Perspective,
                SourceEnvelopeV3,
            )
            from verbatim.evidence import ingest_envelope

            for j in chunk:
                env = SourceEnvelopeV3(
                    kind=EnvelopeKind.AGENT_NOTE,
                    scope_id=scope_ids[j % len(scope_ids)],
                    actor_principal="envelope-corpus",
                    perspective=Perspective(asserter="envelope-corpus"),
                    event_us=seq[0],
                    receipt_us=0,
                    content=corpus.make_note(j).encode("utf-8"),
                    media_type="text/plain",
                    host_id="envelope-harness",
                    external_id=f"note-{j}",
                    capture_proof=proof,
                    metadata={},
                )
                seq[0] += 1
                ingest_envelope(conn, store, env)
    drain1 = _drain(bundle.ingester)

    # --- S1+ mix: >=10% historical revisions, ~5% holds --------------------
    with store.read() as conn:
        heads = conn.execute(
            "SELECT cr.claim_id, MAX(cr.revision) AS rev, c.scope_id FROM"
            " claim_revisions cr JOIN claims c ON c.claim_id = cr.claim_id"
            " WHERE cr.state = 'active' GROUP BY cr.claim_id"
        ).fetchall()
    by_scope: dict[str, list[tuple[str, int]]] = {}
    for cid, rev, sid in heads:
        by_scope.setdefault(sid, []).append((cid, int(rev)))

    superseded = 0
    want_hist = int(spec.claims * spec.historical_revision_pct)
    if want_hist:
        for sid, pairs in by_scope.items():
            for k in range(0, len(pairs) - 1, 2):
                if superseded >= want_hist:
                    break
                (cid_a, rev_a), (cid_b, _rev_b) = pairs[k], pairs[k + 1]
                try:
                    bundle.engine.apply_transition(
                        TransitionCommand(
                            claim_id=cid_a,
                            expected_revision=rev_a,
                            effect="supersede",
                            actor_id=_EVALUATOR,
                            reason="envelope mix: superseded by later fact",
                            successor_claim_id=cid_b,
                        ),
                        scope=_scope_obj(sid, scope_ids),
                    )
                    superseded += 1
                except Exception as exc:
                    notes.append(
                        f"supersede failed for {cid_a}:"
                        f" {type(exc).__name__}: {exc}"
                    )
                    break

    held = 0
    want_holds = int(spec.claims * spec.hold_pct)
    if want_holds:
        from verbatim.security.quarantine import open_quarantine

        flat = [
            (cid, rev, sid)
            for sid, cs in by_scope.items()
            for cid, rev in cs
        ]
        with store.tx() as conn:
            for cid, rev, sid in flat[:want_holds]:
                try:
                    if open_quarantine(
                        conn,
                        ("claim", cid, rev),
                        ["envelope_mix:hold"],
                        None,
                        scope_id=sid,
                    ):
                        held += 1
                except Exception as exc:
                    notes.append(
                        f"hold failed for {cid}: {type(exc).__name__}: {exc}"
                    )

    observed = {
        "claims_total": _count("claims"),
        "claims_active": int(
            _scalar(
                store,
                "SELECT COUNT(*) FROM claim_revisions cr"
                " JOIN (SELECT claim_id, MAX(revision) r FROM"
                "       claim_revisions GROUP BY claim_id) h"
                " ON h.claim_id = cr.claim_id AND h.r = cr.revision"
                " WHERE cr.state = 'active'",
            )
            or 0
        ),
        "claim_revisions": _count("claim_revisions"),
        "spans": _count("spans"),
        "episodes": _count("episodes"),
        "procedures": _count("procedures"),
        "sources": _count("sources"),
        "embeddings": _count("embeddings")
        if _has_table(store, "embeddings")
        else 0,
        "superseded_claims": superseded,
        "held_claims": held,
        "seed_drain": {
            "claims_pass": drain0,
            "filler_pass": drain1,
        },
    }
    if observed["claims_active"] < spec.claims:
        notes.append(
            f"active claims {observed['claims_active']} below spec volume"
            f" {spec.claims} — admissions that rejected are reported, not"
            " forced"
        )
    return {"claim_meta": claim_meta, "observed": observed,
            "procedure_ids": procedure_ids, "episode_ids": episode_ids,
            "proof": proof}


def _has_table(store: Any, name: str) -> bool:
    with store.read() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
            is not None
        )


def _scalar(store: Any, sql: str, params: tuple = ()) -> Any:
    with store.read() as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


_SCOPE_CACHE: dict[str, Any] = {}


def _scope_obj(scope_id: str, scope_ids: list[str]) -> Any:
    """Recover the ``Scope`` for a partition id (needed by the v2-style
    ``apply_transition`` API, which takes Scope objects)."""
    return _SCOPE_CACHE[scope_id]


# ---------------------------------------------------------------------------
# query bank (V4-56.02 mix)
# ---------------------------------------------------------------------------

_QUERY_MIX = (
    ("exact", 0.30),
    ("paraphrase", 0.20),
    ("multi_entity", 0.15),
    ("temporal", 0.10),
    ("no_answer", 0.10),
    ("cross_scope", 0.10),
    ("procedural", 0.05),
)


def _build_query_bank(
    spec: EnvelopeSpec,
    corpus: _Corpus,
    claim_meta: list[dict[str, Any]],
    scope_ids: list[str],
    procedure_labels: list[str],
) -> list[dict[str, Any]]:
    """Deterministic per-category query list covering the §56 mix.

    ``cross_scope`` entries run against the secondary partition — the
    caller holds grants on every seeded scope, so these are *authorized*
    cross-scope cases per V4-56.02.
    """
    rng = random.Random(99173)
    bank: list[dict[str, Any]] = []
    n = max(64, min(512, spec.queries // 4))
    primaries = [m for m in claim_meta if m["scope_id"] == scope_ids[0]]
    secondary = (
        [m for m in claim_meta if m["scope_id"] != scope_ids[0]]
        or primaries
    )
    for i in range(n):
        cat = _QUERY_MIX[i % len(_QUERY_MIX)][0]
        m = primaries[i % len(primaries)] if primaries else {
            "id_terms": ["x"], "entities": ["x"], "paraphrase": "x",
            "scope_id": scope_ids[0],
        }
        if cat == "exact":
            q = f"{m['id_terms'][0]} {m['entities'][0]}"
            req: dict[str, Any] = {"query": q, "scope_id": m["scope_id"]}
        elif cat == "paraphrase":
            req = {"query": m["paraphrase"], "scope_id": m["scope_id"]}
        elif cat == "multi_entity":
            m2 = primaries[(i * 7 + 1) % len(primaries)]
            req = {
                "query": (
                    f"{m['entities'][0]} {m2['entities'][0]} "
                    f"{m['id_terms'][0]}"
                ),
                "scope_id": scope_ids[0],
            }
        elif cat == "temporal":
            req = {
                "query": (
                    f"what is the current {m['entities'][0]} "
                    f"{m['entities'][1]} status"
                ),
                "scope_id": m["scope_id"],
                "valid_at_us": 1_800_000_000_000_000,
            }
        elif cat == "no_answer":
            req = {
                "query": (
                    f"zqx{rng.randrange(10**6)} wobble"
                    f" {rng.randrange(10**6)} nonexistent"
                ),
                "scope_id": scope_ids[0],
            }
        elif cat == "cross_scope":
            mm = secondary[i % len(secondary)]
            req = {
                "query": f"{mm['id_terms'][0]} {mm['entities'][0]}",
                "scope_id": mm["scope_id"],
            }
        else:  # procedural
            label = (
                procedure_labels[i % len(procedure_labels)]
                if procedure_labels
                else "fix-tests-0"
            )
            req = {
                "query": f"procedure for {label} steps runbook",
                "scope_id": scope_ids[0],
                "memory_kinds": ("procedure",),
            }
        req["category"] = cat
        bank.append(req)
    rng.shuffle(bank)
    return bank


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _recall_once(bundle: _EngineBundle, q: dict[str, Any]) -> float:
    """One timed recall through the public facade; returns elapsed ms.

    The latency of a *failed* query is still returned — suppressing it
    would flatter the distribution (V4-55.10). The caller records the
    failure separately.
    """
    budget: dict[str, Any] = {}
    if q.get("memory_kinds"):
        budget["memory_kinds"] = list(q["memory_kinds"])
    if q.get("valid_at_us") is not None:
        budget["valid_at_us"] = q["valid_at_us"]
    t0 = time.perf_counter_ns()
    bundle.facade.recall(
        q["scope_id"],
        q["query"],
        principal_id=_CALLER,
        purpose="recall",
        budget=budget or None,
    )
    return (time.perf_counter_ns() - t0) / 1e6


def _measure_repetition(
    bundle: _EngineBundle,
    spec: EnvelopeSpec,
    queries: list[dict[str, Any]],
    capture_items: list[tuple[str, str, bool]],
    proof: str,
    rep_index: int,
    notes: list[str],
) -> dict[str, Any]:
    """One repetition: warmup (excluded) → declared concurrent load →
    measured recall samples → backlog + drain totals."""
    from verbatim.core.types_v3 import EnvelopeKind
    from verbatim.core.time import now_us
    from verbatim.readiness import CapabilityName

    store = bundle.store
    cap_scope_cycle = sorted({q["scope_id"] for q in queries})

    # --- warmup: same path, samples discarded by contract (V4-56.01) -----
    for i in range(spec.warmup):
        q = queries[i % len(queries)]
        try:
            _recall_once(bundle, q)
        except Exception:
            pass

    recall_samples: list[float] = []
    recall_by_cat: dict[str, list[float]] = {}
    recall_failures: list[str] = []
    cat_counts: dict[str, int] = {}
    lock = threading.Lock()
    stop = threading.Event()
    receipts_q: "queue.Queue[tuple[str, str, float]]" = queue.Queue()

    # --- capture client (paced) -------------------------------------------
    capture_ack: list[float] = []
    capture_failures: list[str] = []
    captured: list[tuple[str, str, float]] = []

    def _capture_loop() -> None:
        period = 1.0 / spec.captures_per_s if spec.captures_per_s > 0 else 0
        i = 0
        next_t = time.monotonic()
        while not stop.is_set():
            next_t += period
            kind_s, text, claim_bearing = capture_items[
                i % len(capture_items)
            ]
            sid = cap_scope_cycle[i % len(cap_scope_cycle)]
            t0 = time.perf_counter_ns()
            try:
                # user_message is a host-produced kind — passing the proof
                # would fail closed (kind not in allowed_kinds); agent_note
                # requires it (V3-11.11).
                r = _capture(
                    bundle,
                    sid,
                    text,
                    kind=EnvelopeKind(kind_s),
                    external_id=f"live-r{rep_index}-{i}",
                    proof=proof if kind_s == "agent_note" else None,
                    event_us=0,
                )
                ack = (time.perf_counter_ns() - t0) / 1e6
                t_commit = time.monotonic()
                with lock:
                    capture_ack.append(ack)
                    captured.append((r.receipt_id, sid, t_commit))
                # readiness is measured on claim-bearing receipts — an
                # agent note's DAG settles deferred by design (V3-13.11)
                if claim_bearing:
                    receipts_q.put((r.receipt_id, sid, t_commit))
            except Exception as exc:
                with lock:
                    capture_failures.append(
                        f"{type(exc).__name__}: {exc}"
                    )
            i += 1
            delay = next_t - time.monotonic()
            if delay > 0:
                stop.wait(delay)

    # --- readiness collector ----------------------------------------------
    lex_wait: list[float] = []
    sem_wait: list[float] = []
    ready_states: dict[str, int] = {}
    ready_sem_states: dict[str, int] = {}
    ready_deadlines = [0]

    def _ready_loop() -> None:
        eng = bundle.ingester.readiness_engine()
        while True:
            try:
                rid, sid, t_commit = receipts_q.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            deadline = now_us() + int(spec.readiness_deadline_s * 1e6)
            snap: dict[str, Any] = {}
            snap2: dict[str, Any] = {}
            try:
                snap = eng.wait_ready(
                    rid,
                    [CapabilityName.LEXICAL_READY],
                    deadline_us=deadline,
                    poll_s=0.01,
                    scope_id=sid,
                )
                st = snap["states"]["lexical_ready"]["state"]
                lex_ms = (time.monotonic() - t_commit) * 1000.0
            except Exception:
                st = "error"
                lex_ms = None
            # semantic rung is measured separately so lexical latency is
            # not padded by embed work (V4-56.06 keeps them distinct)
            try:
                snap2 = eng.wait_ready(
                    rid,
                    [CapabilityName.SEMANTIC_READY],
                    deadline_us=deadline,
                    poll_s=0.01,
                    scope_id=sid,
                )
                st2 = snap2["states"]["semantic_ready"]["state"]
                sem_ms = (time.monotonic() - t_commit) * 1000.0
            except Exception:
                st2 = "error"
                sem_ms = None
            with lock:
                ready_states[st] = ready_states.get(st, 0) + 1
                ready_sem_states[st2] = ready_sem_states.get(st2, 0) + 1
                if lex_ms is not None:
                    lex_wait.append(lex_ms)
                if sem_ms is not None:
                    sem_wait.append(sem_ms)
                if snap.get("deadline_exceeded") or snap2.get(
                    "deadline_exceeded"
                ):
                    ready_deadlines[0] += 1

    # --- drain worker -------------------------------------------------------
    drain_totals = {"processed": 0, "succeeded": 0, "failed": 0,
                    "deferred": 0, "expired": 0}
    drain_last: dict[str, Any] = {}

    def _drain_loop() -> None:
        while not stop.is_set():
            rep = bundle.ingester.drain_report(limit=64)
            with lock:
                for k in drain_totals:
                    drain_totals[k] += int(rep.get(k, 0))
                drain_last.clear()
                drain_last.update(rep)
            if rep.get("processed", 0) == 0:
                stop.wait(0.01)

    # --- recall clients ------------------------------------------------------
    per_client: list[int] = []
    slices = [queries[i:: spec.recall_clients]
              for i in range(spec.recall_clients)]

    def _client(idx: int) -> None:
        n = 0
        for q in slices[idx]:
            cat = q["category"]
            try:
                ms = _recall_once(bundle, q)
            except Exception as exc:
                ms = -1.0
                err = f"{type(exc).__name__}: {exc}"
            else:
                err = None
            with lock:
                if ms >= 0:
                    recall_samples.append(ms)
                    recall_by_cat.setdefault(cat, []).append(ms)
                else:
                    recall_failures.append(err or "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                n += 1
        per_client.append(n)

    t_start = time.monotonic()
    threads = [
        threading.Thread(target=_client, args=(i,), daemon=True,
                         name=f"recall-{i}")
        for i in range(spec.recall_clients)
    ]
    cap_t = threading.Thread(target=_capture_loop, daemon=True,
                             name="capture")
    dra_t = threading.Thread(target=_drain_loop, daemon=True, name="drain")
    red_t = threading.Thread(target=_ready_loop, daemon=True,
                             name="readiness")
    for t in (cap_t, dra_t, red_t, *threads):
        t.start()
    for t in threads:
        t.join()
    stop.set()
    cap_t.join(timeout=5)
    # Tail: keep draining in-line until the queue is empty or the cap hits
    # (bounded at 2x the readiness deadline + 10s — leftover receipts are
    # reported as backlog, not waited out serially).
    tail_deadline = time.monotonic() + min(
        60.0, spec.readiness_deadline_s * 2 + 10.0
    )
    while time.monotonic() < tail_deadline:
        rep = bundle.ingester.drain_report(limit=64)
        with lock:
            for k in drain_totals:
                drain_totals[k] += int(rep.get(k, 0))
            drain_last.clear()
            drain_last.update(rep)
        if rep.get("still_pending", 0) == 0 and receipts_q.empty():
            break
        time.sleep(0.02)
    dra_t.join(timeout=5)
    # Readiness collector drains its queue with the same deadline budget;
    # whatever remains unsettled is the reported backlog.
    red_t.join(timeout=spec.readiness_deadline_s + 5)
    wall_s = time.monotonic() - t_start

    backlog = {
        "jobs_pending": drain_last.get("still_pending"),
        "obligations_pending": drain_last.get("pending_obligations"),
        "receipts_unsettled": receipts_q.qsize(),
    }
    if recall_failures:
        notes.append(
            f"rep{rep_index}: {len(recall_failures)} recall failures, e.g."
            f" {recall_failures[0]}"
        )
    if capture_failures:
        notes.append(
            f"rep{rep_index}: {len(capture_failures)} capture failures,"
            f" e.g. {capture_failures[0]}"
        )
    return {
        "repetition": rep_index,
        "wall_s": wall_s,
        "recall_samples": recall_samples,
        "recall_by_cat": {
            k: percentiles(v) for k, v in sorted(recall_by_cat.items())
        },
        "recall_failures": len(recall_failures),
        "recall_failure_examples": recall_failures[:5],
        "per_client": sorted(per_client),
        "query_categories": dict(sorted(cat_counts.items())),
        "capture_ack_ms": capture_ack,
        "capture_failures": len(capture_failures),
        "capture_failure_examples": capture_failures[:5],
        "captures": len(captured),
        "capture_rate_s": (
            len(captured) / wall_s if wall_s > 0 else 0.0
        ),
        "readiness": {
            "lexical_states": dict(sorted(ready_states.items())),
            "semantic_states": dict(sorted(ready_sem_states.items())),
            "lexical_wait_ms": lex_wait,
            "semantic_wait_ms": sem_wait,
            "deadline_exceeded": ready_deadlines[0],
        },
        "drain": dict(drain_totals),
        "backlog_end": backlog,
    }


# ---------------------------------------------------------------------------
# top-level runner
# ---------------------------------------------------------------------------


def run_envelope(
    spec: EnvelopeSpec,
    root: str,
    *,
    seed: int = 42,
    extra_notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run one §56 envelope under ``root``; returns the report dict.

    Layout: ``root/<envelope>/env.db`` holds the profile store. Seeding and
    each restart repetition use the real public path; the corpus RNG is
    seeded (``seed``) so content is reproducible — only timings vary.
    """
    from verbatim.core.identity import scope_key
    from verbatim.core.types import Scope
    from verbatim.host import LocalHost

    notes: list[str] = list(extra_notes or [])
    t_run = time.monotonic()
    work = os.path.join(root, spec.envelope)
    os.makedirs(work, exist_ok=True)
    db_path = os.path.join(work, "env.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
        if os.path.exists(db_path + ".key"):
            os.remove(db_path + ".key")

    cfg = _config(spec)
    host = LocalHost(
        profile_id="env", principal_id="envelope-corpus",
        conversation_id="env-conv",
    )
    scope_objs = [
        Scope(
            profile_id="env",
            principal_id="envelope-corpus"
            if i == 0 else f"tenant-{i}",
            conversation_id="env-conv" if i == 0 else f"env-conv-{i}",
        )
        for i in range(spec.scopes)
    ]
    scope_ids = [scope_key(s) for s in scope_objs]
    _SCOPE_CACHE.clear()
    _SCOPE_CACHE.update(dict(zip(scope_ids, scope_objs)))

    load_start = _loadavg()
    corpus = _Corpus(seed)

    bundle = _EngineBundle(db_path, cfg, host, create=True)
    try:
        seeded = _seed(bundle, spec, corpus, scope_ids, notes)
        observed = seeded["observed"]
        proof = seeded["proof"]
        proc_labels = [
            f"fix-tests-{i}" for i in range(len(seeded["procedure_ids"]))
        ]
        bank = _build_query_bank(
            spec, corpus, seeded["claim_meta"], scope_ids, proc_labels
        )
    finally:
        bundle.close()

    # --- restart repetitions (V4-56.01) ------------------------------------
    per_rep: list[dict[str, Any]] = []
    all_recall: list[float] = []
    all_ack: list[float] = []
    all_lex: list[float] = []
    all_sem: list[float] = []
    rep_summaries: list[dict[str, Any]] = []
    queries_per_rep = spec.queries // spec.restarts
    remainder = spec.queries % spec.restarts
    # Measured capture stream: 1 in 5 is a claim-bearing user_message
    # (full capture → harvest → admit → index pipeline → lexical readiness);
    # the rest are 4KiB agent-authored notes — real write-path + ack work,
    # no claim derivation — so the measured phase does not inflate the
    # envelope's declared claim volume mid-run.
    capture_items: list[tuple[str, str, bool]] = []
    for i in range(int(spec.captures_per_s * 120) + 64):
        if i % 5 == 4:
            # ~1 paragraph → ~1 claim: keeps measured-phase claim growth
            # bounded (~20% of captures × ~1 claim) while still exercising
            # the full admit→index readiness path.
            capture_items.append(
                ("user_message", corpus.make_capture_text(i, 128), True)
            )
        else:
            capture_items.append(
                (
                    "agent_note",
                    corpus.make_note_capture(i, spec.capture_payload_bytes),
                    False,
                )
            )
    notes.append(
        "measured captures: 20% claim-bearing user_messages (~128B, ~1 "
        "claim each, full pipeline, lexical readiness measured) + 80% "
        "agent_notes (4KiB, write-path ack load, no claim derivation — "
        "envelope volumes stay at spec); all acks are real accepted-input "
        "timings"
    )

    for rep in range(spec.restarts):
        bundle = _EngineBundle(db_path, cfg, host, create=False)
        try:
            rep_queries = queries_per_rep + (1 if rep < remainder else 0)
            qslice = [
                bank[(rep * queries_per_rep + i) % len(bank)]
                for i in range(rep_queries)
            ]
            m = _measure_repetition(
                bundle, spec, qslice, capture_items, proof, rep, notes
            )
        finally:
            bundle.close()
        per_rep.append(m)
        all_recall.extend(m["recall_samples"])
        all_ack.extend(m["capture_ack_ms"])
        all_lex.extend(m["readiness"]["lexical_wait_ms"])
        all_sem.extend(m["readiness"]["semantic_wait_ms"])
        rep_summaries.append(
            {
                "repetition": rep,
                "measured": len(m["recall_samples"]),
                "failures": m["recall_failures"],
                "p50_ms": percentiles(m["recall_samples"])["p50_ms"],
                "p95_ms": percentiles(m["recall_samples"])["p95_ms"],
                "p99_ms": percentiles(m["recall_samples"])["p99_ms"],
                "wall_s": m["wall_s"],
            }
        )

    # readiness waits were collected per-receipt across reps
    lex_states: dict[str, int] = {}
    sem_states: dict[str, int] = {}
    deadlines = 0
    for m in per_rep:
        for k, v in m["readiness"]["lexical_states"].items():
            lex_states[k] = lex_states.get(k, 0) + v
        for k, v in m["readiness"]["semantic_states"].items():
            sem_states[k] = sem_states.get(k, 0) + v
        deadlines += m["readiness"]["deadline_exceeded"]

    recall_stats = percentiles(all_recall)
    ack_stats = percentiles(all_ack)
    drain_totals = {"processed": 0, "succeeded": 0, "failed": 0,
                    "deferred": 0, "expired": 0}
    backlog_end = {"jobs_pending": 0, "obligations_pending": 0,
                   "receipts_unsettled": 0}
    for m in per_rep:
        for k in drain_totals:
            drain_totals[k] += int(m["drain"].get(k, 0))
        backlog_end["receipts_unsettled"] += int(
            m["backlog_end"].get("receipts_unsettled") or 0
        )
    if per_rep:
        backlog_end["jobs_pending"] = per_rep[-1]["backlog_end"].get(
            "jobs_pending"
        )
        backlog_end["obligations_pending"] = per_rep[-1]["backlog_end"].get(
            "obligations_pending"
        )

    metrics = {
        "recall": {
            **recall_stats,
            # per-category latency rollups, per repetition (V4-56.02 mix)
            "by_category": {
                rep["repetition"]: rep["recall_by_cat"] for rep in per_rep
            },
            "failures": sum(m["recall_failures"] for m in per_rep),
            "failure_examples": [
                e for m in per_rep for e in m["recall_failure_examples"]
            ][:10],
            "clients": spec.recall_clients,
            "measured": len(all_recall),
        },
        "capture": {
            "ack": ack_stats,
            "captures": sum(m["captures"] for m in per_rep),
            "failures": sum(m["capture_failures"] for m in per_rep),
            "failure_examples": [
                e for m in per_rep for e in m["capture_failure_examples"]
            ][:10],
            "declared_rate_s": spec.captures_per_s,
            "measured_rate_s": (
                sum(m["capture_rate_s"] for m in per_rep) / len(per_rep)
                if per_rep else 0.0
            ),
        },
        "readiness": {
            "lexical_p95_ms": percentiles(all_lex)["p95_ms"],
            "semantic_p95_ms": percentiles(all_sem)["p95_ms"],
            "lexical": {
                "states": lex_states,
                "settled": sum(lex_states.values()),
                **percentiles(all_lex),
            },
            "semantic": {
                "states": sem_states,
                "settled": sum(sem_states.values()),
                **percentiles(all_sem),
            },
            "deadline_exceeded": deadlines,
            "deadline_s": spec.readiness_deadline_s,
            "wait_definition": (
                "ms from capture-tx commit to wait_ready() settle on the"
                " receipt's own obligation DAG (V4-56.06)"
            ),
        },
        "drain": drain_totals,
        "backlog_end": backlog_end,
    }

    # target evaluation — every declared bound becomes a row
    target_rows = []
    for name, bound in spec.targets.items():
        measured = _metric_lookup(metrics, name)
        ok = measured is not None and measured <= bound
        target_rows.append(
            {
                "target": name,
                "bound_ms": bound,
                "measured_ms": measured,
                "met": bool(ok),
            }
        )
    met = all(r["met"] for r in target_rows) and all(
        m["recall_failures"] == 0 and m["capture_failures"] == 0
        for m in per_rep
    )

    report = {
        "envelope": spec.envelope,
        "spec": "SPEC_V4 §56 (V4-56.01/56.02/56.05/56.06)",
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "seed": seed,
        "config": {
            "claims": spec.claims,
            "spans": spec.spans,
            "episodes": spec.episodes,
            "procedures": spec.procedures,
            "recall_clients": spec.recall_clients,
            "captures_per_s": spec.captures_per_s,
            "drains": spec.drains,
            "embedding_backend": spec.embedding_backend,
            "admission_require_review": False,
            "capture_payload_bytes": spec.capture_payload_bytes,
            "scopes": spec.scopes,
        },
        "observed_volumes": observed,
        "mix": {
            "historical_revisions_pct": (
                observed["superseded_claims"] / max(1, spec.claims)
            ),
            "hold_pct": observed["held_claims"] / max(1, spec.claims),
            "scopes": len(scope_ids),
            "query_mix_declared": dict(_QUERY_MIX),
            "query_mix_realized": _merge_counts(per_rep),
        },
        "samples": {
            "measured_queries": len(all_recall),
            "warmup_queries": spec.warmup * spec.restarts,
            "warmup_excluded": True,
            "restart_repetitions": spec.restarts,
            "per_repetition": rep_summaries,
        },
        "metrics": metrics,
        "targets": target_rows,
        "met": bool(met),
        "environment": {
            **_environment(db_path),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
        },
        "duration_s": time.monotonic() - t_run,
        "notes": notes + ([spec.note] if spec.note else []),
    }
    return report


def _merge_counts(per_rep: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in per_rep:
        for k, v in m["query_categories"].items():
            out[k] = out.get(k, 0) + v
    return out


def _metric_lookup(metrics: dict[str, Any], name: str) -> Optional[float]:
    """Resolve a ``targets`` key like ``recall.p95_ms`` into the measured
    value; ``None`` means the metric has no samples (reported as a miss)."""
    if name in ("capture.ack_p95_ms", "capture.ack_p95_ms_ingest"):
        return metrics["capture"]["ack"]["p95_ms"]
    parts = name.split(".")
    node: Any = metrics
    for p in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(p)
    return node if isinstance(node, (int, float)) else None


def run_s0(root: str, **kwargs: Any) -> dict[str, Any]:
    """The §56 S0 envelope (M0 no-model)."""
    return run_envelope(S0_SPEC, root, **kwargs)


def run_s1(root: str, **kwargs: Any) -> dict[str, Any]:
    """The §56 S1 envelope (1K claims, 10K measured queries)."""
    return run_envelope(S1_SPEC, root, **kwargs)


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable envelope report — same numbers as the JSON."""
    r = report
    m = r["metrics"]
    lines = [
        f"# Workload envelope {r['envelope'].upper()} — SPEC_V4 §56",
        "",
        f"Generated: {r['generated_at']}  |  seed {r['seed']}  |  "
        f"duration {r['duration_s']:.1f}s",
        "",
        f"**Verdict: {'MET' if r['met'] else 'MISSED'}** (all declared"
        " targets AND zero query/capture failures required)",
        "",
        "## Volumes (declared → observed)",
        "",
        "| volume | spec | observed |",
        "| --- | ---: | ---: |",
    ]
    obs = r["observed_volumes"]
    for key, label in (
        ("claims", "claims_total"),
        ("claims", "claims_active"),
        ("spans", "spans"),
        ("episodes", "episodes"),
        ("procedures", "procedures"),
    ):
        lines.append(
            f"| {label} | {r['config'][key]} | {obs.get(label)} |"
        )
    lines += [
        f"| claim_revisions | — | {obs.get('claim_revisions')} |",
        f"| sources | — | {obs.get('sources')} |",
        f"| embeddings | — | {obs.get('embeddings')} |",
        f"| superseded (historical) | "
        f"{r['config']['claims'] and ''}≥"
        f"{100 * r['mix']['historical_revisions_pct']:.0f}% seeded | "
        f"{obs.get('superseded_claims')} |",
        f"| held (quarantine) | — | {obs.get('held_claims')} |",
        "",
        "## Recall latency (ms) — measured samples only",
        "",
        f"- clients: {m['recall']['clients']}, measured: "
        f"{m['recall']['measured']}, failures: {m['recall']['failures']}",
        f"- p50 {_fmt(m['recall']['p50_ms'])} / p95 "
        f"{_fmt(m['recall']['p95_ms'])} / p99 "
        f"{_fmt(m['recall']['p99_ms'])} / max "
        f"{_fmt(m['recall']['max_ms'])} / mean "
        f"{_fmt(m['recall']['mean_ms'])}",
        "",
        "## Capture + readiness",
        "",
        f"- captures: {m['capture']['captures']} "
        f"(declared {m['capture']['declared_rate_s']}/s, measured "
        f"{_fmt(m['capture']['measured_rate_s'])}/s), failures: "
        f"{m['capture']['failures']}",
        f"- capture ack p50 {_fmt(m['capture']['ack']['p50_ms'])} / p95 "
        f"{_fmt(m['capture']['ack']['p95_ms'])} / p99 "
        f"{_fmt(m['capture']['ack']['p99_ms'])} / max "
        f"{_fmt(m['capture']['ack']['max_ms'])}",
        f"- lexical_ready states: {m['readiness']['lexical']['states']}"
        f" — p95 {_fmt(m['readiness']['lexical_p95_ms'])}ms",
        f"- semantic_ready states: {m['readiness']['semantic']['states']}"
        f" — p95 {_fmt(m['readiness']['semantic_p95_ms'])}ms",
        f"- readiness deadlines exceeded: "
        f"{m['readiness']['deadline_exceeded']}",
        "",
        "## Load discipline",
        "",
        f"- measured queries: {r['samples']['measured_queries']} "
        f"(warmup {r['samples']['warmup_queries']} excluded; "
        f"{r['samples']['restart_repetitions']} restart repetitions)",
        f"- per-repetition: {r['samples']['per_repetition']}",
        f"- query mix realized: {r['mix']['query_mix_realized']}",
        f"- drain totals: {m['drain']}",
        f"- backlog at end: {m['backlog_end']}",
        "",
        "## Targets",
        "",
        "| target | bound ms | measured ms | met |",
        "| --- | ---: | ---: | --- |",
    ]
    for t in r["targets"]:
        lines.append(
            f"| {t['target']} | {_fmt(t['bound_ms'])} | "
            f"{_fmt(t['measured_ms'])} | {'yes' if t['met'] else 'NO'} |"
        )
    lines += [
        "",
        "## Environment",
        "",
        "```",
        json.dumps(r["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {n}" for n in r["notes"]]
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write ``envelope_<id>_report.{json,md}`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    env = report["envelope"]
    jpath = os.path.join(out_dir, f"envelope_{env}_report.json")
    mpath = os.path.join(out_dir, f"envelope_{env}_report.md")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return {"json": jpath, "md": mpath}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v4.envelopes",
        description="SPEC_V4 §56 workload-envelope measurement (S0/S1).",
    )
    ap.add_argument("--envelope", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--work",
        default=None,
        help="scratch dir for the seeded store (default: <out>/work)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--queries",
        type=int,
        default=None,
        help="override measured-query count (reported honestly; below the"
        " spec minimum keeps the envelope unqualified)",
    )
    args = ap.parse_args(argv)

    spec = SPECS[args.envelope]
    if args.queries is not None:
        spec = replace(spec, queries=int(args.queries))
    report = run_envelope(
        spec, args.work or os.path.join(args.out, "work"), seed=args.seed
    )
    paths = write_reports(report, args.out)
    print(
        f"[{spec.envelope}] met={report['met']} "
        f"recall p95={_fmt(report['metrics']['recall']['p95_ms'])}ms "
        f"p99={_fmt(report['metrics']['recall']['p99_ms'])}ms "
        f"n={report['samples']['measured_queries']}"
    )
    for t in report["targets"]:
        flag = "ok" if t["met"] else "MISS"
        print(f"  {flag} {t['target']}: {_fmt(t['measured_ms'])} "
              f"vs bound {_fmt(t['bound_ms'])}")
    print(f"wrote {paths['json']} + {paths['md']}")
    return 0 if report["met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
