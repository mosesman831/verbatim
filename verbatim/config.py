"""Validated configuration model (SPEC §37).

Precedence: defaults → selected profile/file → explicit CLI overrides.
Unknown keys, invalid types, and contradictory privacy settings are rejected
with actionable errors — never silently coerced.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from .core.types import ErrorCode, Mode, VerbatimError


@dataclass(frozen=True)
class CaptureConfig:
    enabled: bool = False
    user_messages: bool = True
    assistant_context: bool = False
    tool_outputs: bool = False
    max_source_bytes: int = 262_144


@dataclass(frozen=True)
class AdmissionConfig:
    require_review: bool = True


@dataclass(frozen=True)
class SupersessionConfig:
    automatic: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "none"  # none | artifact | hashing | ollama | cloudflare
    artifact_revision: Optional[str] = None
    model: str = "nomic-embed-text"
    endpoint: str = "http://127.0.0.1:11434"
    account_id: Optional[str] = None  # cloudflare backend only


@dataclass(frozen=True)
class JudgeConfig:
    backend: str = "rules"  # rules | jev
    model: str = "jev-1.13.0"
    daily_budget_usd: Decimal = Decimal("0")
    max_attempts: int = 3
    transport: str = "typesafe"  # typesafe | cloudflare
    account_id: Optional[str] = None  # cloudflare transport only


@dataclass(frozen=True)
class RecallConfig:
    max_items: int = 8
    max_bytes: int = 6000
    target_tokens: int = 1536
    remote_rerank: bool = False


@dataclass(frozen=True)
class JobsConfig:
    max_pending: int = 10_000
    workers_per_profile: int = 1
    # V8-13.05 ingest coalescing: same-scope same-kind pending siblings
    # commit inside the leased job's fenced write tx.  0 disables;
    # ``source_coalesce_tx_ms`` bounds how long the shared commit may
    # hold the writer before unprocessed claimed siblings are released
    # back to ``queued`` (never lost) so foreground writes never starve.
    source_coalesce: int = 16
    source_coalesce_tx_ms: float = 100.0
    # V8 §23 dense write-path arms (deployment-level carriers — a job's
    # ``input_refs`` or a retrieval-policy params map overrides these;
    # resolution lives in ``retrieval/v7/dense_compact.job_arm``).
    # ``dense.B_max``: compact a vector space once it holds more than
    # this many blocks (V8-08.01). ``dense.embed_batch``: coalesced
    # ``source_embed`` commits bound merged pending unit rows at
    # ``dense_embed_batch_rows`` and close the batch once an admitted
    # sibling's queue age reaches ``dense_embed_batch_max_age_ms``
    # (V8-08.02).
    dense_compact_b_max: int = 64
    dense_embed_batch_rows: int = 512
    dense_embed_batch_max_age_ms: float = 250.0


@dataclass(frozen=True)
class RetentionConfig:
    pending_days: int = 30
    audit_days: int = 90
    max_store_bytes: int = 1_073_741_824


# --- V3 sections (SPEC_V3 §05, §35, §40) -----------------------------------

@dataclass(frozen=True)
class VaultConfig:
    """Sensitive-value vault (§35). Key material is never stored in the
    database; ``key_source`` names where scope wrapping keys come from.
    ``allow_plaintext_hydration`` stays False: model-visible answers carry
    opaque handles, and plaintext requires a declared downstream processor
    plus consent (§35.09)."""
    enabled: bool = False
    key_source: str = "env"  # env | file | external
    key_file: Optional[str] = None
    allow_plaintext_hydration: bool = False
    handle_ttl_s: int = 3600


@dataclass(frozen=True)
class RetrievalV3Config:
    """Optional retrieval lanes (§28–§31). Lexical/structured/temporal are
    always on; the rest are gated capabilities reported honestly by
    capability_report — never silently enabled (V3-62.04)."""
    dense: bool = False
    sparse: bool = False
    late_interaction: bool = False
    graph: bool = False
    causal: bool = False
    controller: str = "deterministic"  # deterministic | learned_shadow | learned_active
    # learned_active binds to exactly one registered policy artifact id
    # (V6-03.15): config validation requires the binding to be declared;
    # the artifact's validation_state is probed in the bound store at
    # controller construction (verbatim.policy.artifacts.bind_for_activation).
    controller_policy_artifact: Optional[str] = None
    candidate_cap: int = 128


@dataclass(frozen=True)
class V3Config:
    """V3 profile + feature gates (SPEC_V3 §05). The profile is declared, not
    inferred; capabilities unavailable to the profile report as such."""
    profile: str = "embedded"
    # embedded | local_semantic | team_service | scaled_service
    #          | split_privacy | portable_workspace | local_memory (V5 facade)
    capture_depth: int = 0          # 0 explicit | 1 SDK | 2 adapter | 3 host-native
    procedure_promotion: str = "manual"   # manual | auto (auto needs G5 evidence)
    governance_strict: bool = True  # purposes enforced from registry (§11.06)
    vault: VaultConfig = field(default_factory=VaultConfig)
    retrieval: RetrievalV3Config = field(default_factory=RetrievalV3Config)


# --- V4 retrieval surface (SPEC_V4 §32–§33) ---------------------------------


@dataclass(frozen=True)
class RetrievalCacheConfig:
    """Final-pack recall cache (V4-33). In-process, per-store,
    capacity-bounded LRU — never persisted, never an authorization
    decision. ``enabled`` defaults OFF: the invalidation model
    (epoch vector + projection generation + ``PRAGMA data_version`` +
    content watermark + per-ref revalidation on every hit) is
    conservative, but operators opt in explicitly rather than trusting a
    silent default."""
    enabled: bool = False
    capacity: int = 256


@dataclass(frozen=True)
class RetrievalConfig:
    """Top-level retrieval surface knobs (V4 §32–§33) — distinct from the
    legacy ``recall`` bounds, which govern the v2 path, and from
    ``v3.retrieval``, which gates v3 lanes/controller."""
    cache: RetrievalCacheConfig = field(default_factory=RetrievalCacheConfig)


@dataclass(frozen=True)
class VerbatimConfig:
    """Top-level engine configuration (SPEC §37 defaults)."""

    mode: Mode = Mode.OFFLINE_RULES
    data_dir: str = "verbatim"
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    supersession: SupersessionConfig = field(default_factory=SupersessionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    v3: V3Config = field(default_factory=V3Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def validate(self) -> "VerbatimConfig":
        if not isinstance(self.mode, Mode):
            object.__setattr__(self, "mode", Mode(self.mode))
        # jev_assisted is a restricted remote_assisted alias (SPEC_V2 §4):
        # it may reach remote processors but only for Jev purposes.
        remote_ok = self.mode in (Mode.REMOTE_ASSISTED, Mode.JEV_ASSISTED)
        service_ok = remote_ok or self.mode == Mode.LOCAL_SERVICE
        c = self.capture
        if c.max_source_bytes < 1024:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "capture.max_source_bytes < 1024")
        j = self.judge
        if j.backend not in ("rules", "jev"):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"judge.backend {j.backend!r} unsupported")
        if not 1 <= j.max_attempts <= 3:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "judge.max_attempts must be 1..3")
        if j.backend == "jev" and not remote_ok:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "judge.backend=jev requires mode=remote_assisted",
            )
        if j.backend == "jev" and j.daily_budget_usd <= 0:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "judge.backend=jev requires a positive daily_budget_usd",
            )
        if j.transport not in ("typesafe", "cloudflare"):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"judge.transport {j.transport!r} unsupported"
            )
        if j.transport == "cloudflare" and not j.account_id:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "judge.transport=cloudflare requires judge.account_id",
            )
        r = self.recall
        if not 1 <= r.max_items <= 32:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "recall.max_items must be 1..32")
        if not 512 <= r.max_bytes <= 24_000:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "recall.max_bytes must be 512..24000")
        if not 64 <= r.target_tokens <= 8192:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "recall.target_tokens must be 64..8192")
        if r.remote_rerank and not remote_ok:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "recall.remote_rerank requires mode=remote_assisted",
            )
        e = self.embedding
        if e.backend not in ("none", "artifact", "hashing", "ollama", "cloudflare"):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"embedding.backend {e.backend!r} unsupported")
        if e.backend == "ollama" and not service_ok:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "loopback embeddings require local_service or remote_assisted mode",
            )
        if e.backend == "cloudflare":
            if not remote_ok:
                raise VerbatimError(
                    ErrorCode.CONFIG_INVALID,
                    "cloudflare embeddings require mode=remote_assisted",
                )
            if not e.account_id:
                raise VerbatimError(
                    ErrorCode.CONFIG_INVALID,
                    "embedding.backend=cloudflare requires embedding.account_id",
                )
        if e.backend == "artifact" and not e.artifact_revision:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "embedding.backend=artifact requires artifact_revision",
            )
        if self.supersession.automatic:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "supersession.automatic requires a validated policy artifact; "
                "leave false until the calibration gate passes (SPEC §26)",
            )
        if self.jobs.max_pending < 1 or self.jobs.workers_per_profile < 1:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "jobs limits must be positive")
        if self.retention.pending_days < 1 or self.retention.audit_days < 1:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "retention windows must be positive")
        v = self.v3
        if v.profile not in (
            "embedded", "local_semantic", "team_service", "scaled_service",
            "split_privacy", "portable_workspace", "local_memory",
        ):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"v3.profile {v.profile!r} unsupported")
        if not 0 <= v.capture_depth <= 3:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "v3.capture_depth must be 0..3")
        if v.procedure_promotion not in ("manual", "auto"):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "v3.procedure_promotion must be manual|auto"
            )
        if v.profile == "split_privacy" and not remote_ok:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "v3.profile=split_privacy requires mode=remote_assisted",
            )
        if v.profile in ("team_service", "scaled_service") and not service_ok:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"v3.profile={v.profile} requires local_service or remote_assisted mode",
            )
        rv = v.retrieval
        if rv.controller not in ("deterministic", "learned_shadow", "learned_active"):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"v3.retrieval.controller {rv.controller!r} unsupported"
            )
        if rv.controller == "learned_active":
            # Learned routing activates only against a policy artifact that
            # passed the paired-execution gate (V3-26/G8, V6-03.15/16).
            # Config cannot probe the store, so validation here requires the
            # binding to be DECLARED; whether the bound artifact is actually
            # validation_state='validated' in the store is enforced where
            # the controller is constructed — verbatim.policy.artifacts.
            # bind_for_activation, called from recall_v3's controller
            # selection block.
            if not rv.controller_policy_artifact:
                raise VerbatimError(
                    ErrorCode.CONFIG_INVALID,
                    "v3.retrieval.controller=learned_active requires a "
                    "validated policy artifact bound via "
                    "v3.retrieval.controller_policy_artifact=<artifact_id>; "
                    "use deterministic or learned_shadow until the paired "
                    "gate passes",
                )
        if not 1 <= rv.candidate_cap <= 1024:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "v3.retrieval.candidate_cap must be 1..1024"
            )
        vt = v.vault
        if vt.key_source not in ("env", "file", "external"):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"v3.vault.key_source {vt.key_source!r} unsupported"
            )
        if vt.key_source == "file" and not vt.key_file:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "v3.vault.key_source=file requires v3.vault.key_file"
            )
        if not 60 <= vt.handle_ttl_s <= 86400:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "v3.vault.handle_ttl_s must be 60..86400"
            )
        rc = self.retrieval.cache
        if not 1 <= rc.capacity <= 65_536:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                "retrieval.cache.capacity must be 1..65536",
            )
        return self


_SECTIONS = {
    "capture": CaptureConfig,
    "admission": AdmissionConfig,
    "supersession": SupersessionConfig,
    "embedding": EmbeddingConfig,
    "judge": JudgeConfig,
    "recall": RecallConfig,
    "jobs": JobsConfig,
    "retention": RetentionConfig,
    "v3": V3Config,
    "retrieval": RetrievalConfig,
}

# Nested dataclass sections inside ``v3`` (vault, retrieval) and the
# top-level ``retrieval`` surface (cache) — keyed by the dotted path they
# appear under so unknown keys are still rejected.
_NESTED_SECTIONS = {
    "v3": {"vault": VaultConfig, "retrieval": RetrievalV3Config},
    "retrieval": {"cache": RetrievalCacheConfig},
}

# ``from __future__ import annotations`` stores f.type as a string, so the
# coerce checks must use resolved hints — otherwise every isinstance test
# fails silently and ``"false"`` would land in a bool field (v1 defect).
_FIELD_TYPES: dict[str, dict[str, Any]] = {
    name: typing.get_type_hints(cls) for name, cls in _SECTIONS.items()
}
for _parent, _children in _NESTED_SECTIONS.items():
    for _child_name, _child_cls in _children.items():
        _FIELD_TYPES[f"{_parent}.{_child_name}"] = typing.get_type_hints(_child_cls)


def _coerce(section: str, key: str, value: Any) -> Any:
    ftypes = _FIELD_TYPES[section]
    if key not in ftypes:
        raise VerbatimError(ErrorCode.CONFIG_INVALID, f"unknown key {section}.{key}")
    expected = ftypes[key]
    if expected is bool:
        if not isinstance(value, bool):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"{section}.{key} must be boolean")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"{section}.{key} must be int")
        return value
    if expected is Decimal:
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"{section}.{key} must be decimal") from exc
    if expected is str or expected == Optional[str]:
        if not isinstance(value, str):
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"{section}.{key} must be string")
        return value
    return value


def config_from_mapping(mapping: dict[str, Any]) -> VerbatimConfig:
    """Build a validated config from a `memory.verbatim`-style mapping.

    Rejects unknown keys at any level (SPEC §37).
    """
    if not isinstance(mapping, dict):
        raise VerbatimError(ErrorCode.CONFIG_INVALID, "config must be a mapping")
    top: dict[str, Any] = {}
    sections: dict[str, dict[str, Any]] = {}
    for key, value in mapping.items():
        if key in ("mode", "data_dir"):
            top[key] = value
        elif key in _SECTIONS:
            if not isinstance(value, dict):
                raise VerbatimError(ErrorCode.CONFIG_INVALID, f"{key} must be a mapping")
            nested = _NESTED_SECTIONS.get(key, {})
            out: dict[str, Any] = {}
            for k, v in value.items():
                if k in nested:
                    if not isinstance(v, dict):
                        raise VerbatimError(
                            ErrorCode.CONFIG_INVALID, f"{key}.{k} must be a mapping"
                        )
                    child_path = f"{key}.{k}"
                    out[k] = nested[k](
                        **{ck: _coerce(child_path, ck, cv) for ck, cv in v.items()}
                    )
                else:
                    out[k] = _coerce(key, k, v)
            sections[key] = out
        else:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, f"unknown key {key}")

    mode_raw = top.get("mode", Mode.OFFLINE_RULES.value)
    try:
        mode = Mode(mode_raw)
    except ValueError as exc:
        raise VerbatimError(ErrorCode.CONFIG_INVALID, f"unknown mode {mode_raw!r}") from exc
    data_dir = top.get("data_dir", "verbatim")
    if not isinstance(data_dir, str) or not data_dir:
        raise VerbatimError(ErrorCode.CONFIG_INVALID, "data_dir must be a non-empty string")

    cfg = VerbatimConfig(
        mode=mode,
        data_dir=data_dir,
        **{name: _SECTIONS[name](**sections.get(name, {})) for name in _SECTIONS},
    )
    return cfg.validate()


def load_config_file(path: str) -> VerbatimConfig:
    """Load a standalone JSON config file (YAML subset: flat JSON only for now)."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return config_from_mapping(raw)
