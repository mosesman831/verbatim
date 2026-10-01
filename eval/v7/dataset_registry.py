"""V7 dataset registry (SPEC_V7 §22 — V7-22.01, V7-22.03, V7-22.08).

The registry is the single source of truth for which evaluation datasets
exist, under what license, where they live locally, which owner decision
(O1/O2/O3, §37) gates them, and how their fixed dev/test partition is
computed.  It records; it does not load — :mod:`eval.v7.corpora` turns an
``available`` entry into a :class:`Corpus`.

Authorization model (§37 defaults — an undecided O-question blocks):

* ``gate`` names the owner decision (``"O1"``/``"O2"``/``"O3"``) or
  ``None``.  A gated dataset is usable only when its ``env_flag`` is set
  to a truthy value (``1|true|yes|on``); the flag IS the recorded
  authorization — nobody sets it without the decision landing.
* Status values (V7-22.01 gives the license disposition; ``status()``
  gives the *operational* answer):

  - ``available`` — gate satisfied (or ungated), loader present, file
    present, pinned digest matching.
  - ``blocked_on_authorization`` — an O-gate is set and its env flag is
    unset.  §22.4 renders this verbatim on the scoreboard.
  - ``missing_file`` — authorized but the data file is absent at every
    configured path.
  - ``unavailable(<reason>)`` — anything else honest: license-blocked
    with no gate mechanism, adapter not implemented, twin generator not
    landed, sha256 mismatch.

* Nothing here ever downloads: ``paths`` are local-only, ``env_path``
  overrides them, and a missing network is never a status input.

Splits (V7-22.08): each entry carries a :class:`SplitSpec` — a *fixed*
deterministic partition by conversation or question-group, never by
random question.  :func:`split_for_group` assigns a group to ``dev`` or
``test`` by hashing ``{tag}:{dataset_id}:{group_key}``; the partition is
recorded in the entry so a tuning artifact can pin the dev-split digest
(V7-22.09).  :func:`materialize_partition` turns a group-id set into a
:class:`Partition` (sorted dev/test + recorded digest) — ids only, so
the machinery is live before O1; :func:`assert_not_tuned_on_test` is
the J16 integrity gate a tuning artifact's manifest must pass.

No dataset content is committed to the repo.  LoCoMo is CC BY-NC 4.0 —
the registry stores its path, license, and digest only (V7-22.03).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# status values (operational — distinct from the V7-22.01 license disposition)
# ---------------------------------------------------------------------------

STATUS_AVAILABLE = "available"
STATUS_BLOCKED = "blocked_on_authorization"
STATUS_MISSING = "missing_file"


def unavailable(reason: str) -> str:
    """The ``unavailable(reason)`` status form."""
    return f"unavailable({reason})"


def is_unavailable(status: str) -> bool:
    return status.startswith("unavailable(") and status.endswith(")")


#: Owner decisions that gate datasets (SPEC_V7 §37).
GATES: Dict[str, str] = {
    "O1": "LoCoMo CC BY-NC 4.0 local-only evaluation permitted?",
    "O2": "LongMemEval dataset download — confirm dataset card terms",
    "O3": "BEAM and DolphinBench license review",
}

#: Truthy env-flag values.  ``VERBATIM_EVAL_LOCOMO=1`` is the documented
#: form; the others exist so a stray ``=0`` never silently authorizes.
_FLAG_TRUE = frozenset({"1", "true", "yes", "on"})

#: Deterministic split-partition revision (V7-22.08).  Changing the tag
#: repartitions every dataset — bump it only with a spec revision.
SPLIT_TAG = "v7-split-v1"

#: Repo root — configured ``paths`` may be repo-relative (e.g. the
#: bundle's ``research/v7_formula_search/locomo10.json``, V75-04.07);
#: they resolve against the caller's CWD first, then against this root,
#: so the registry answers identically from any working directory.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """Fixed dev/test partition rule (V7-22.08).

    ``unit`` names the grouping key the corpus records carry in
    ``group_id`` — ``"conversation"`` (LoCoMo sample_id, one dialogue per
    group) or ``"question_group"`` (LongMemEval/owned corpora group ids).
    ``dev_pct`` percent of hashed groups land in ``dev``; the rest are
    ``test``.  ``tag`` is the partition revision — recorded so a test
    run can prove its tuning artifacts never saw test groups.
    """

    unit: str = "question_group"  # conversation | question_group
    dev_pct: int = 30
    strategy: str = "hash"  # deterministic sha256 partition
    tag: str = SPLIT_TAG

    def __post_init__(self) -> None:
        if self.unit not in ("conversation", "question_group"):
            raise ValueError(f"unknown split unit {self.unit!r}")
        if self.strategy != "hash":
            raise ValueError(f"unknown split strategy {self.strategy!r}")
        if not (0 < self.dev_pct < 100):
            raise ValueError(f"dev_pct {self.dev_pct} outside (0,100)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "dev_pct": self.dev_pct,
            "strategy": self.strategy,
            "tag": self.tag,
        }


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset (V7-22.01 fields + operational surface).

    ``registry_status`` is the V7-22.01 license disposition
    (``permitted | permitted_local_only | blocked | unverified``);
    :func:`status` is the operational answer used by loaders and reports.
    ``loader_kind`` selects the corpora.py path:

    - ``"twin"`` — ``loader`` is a twin module name
      (``eval.v7.twins_*``), imported lazily at load time.
    - ``"locomo_json"`` / ``"longmemeval_json"`` — ``loader`` names the
      ``eval.v7.corpora`` adapter; the dataset is file-backed.
    - ``"none"`` — named but unimplemented (never silently absent,
      V7-23.05-style honesty for datasets).
    """

    dataset_id: str
    source_url: str
    data_license: str
    code_license: str
    redistribution: str  # yes | local_only | no
    local_storage_policy: str
    registry_status: str  # permitted|permitted_local_only|blocked|unverified
    gate: Optional[str]  # "O1"|"O2"|"O3"|None
    loader_kind: str  # twin | locomo_json | longmemeval_json | none
    loader: str
    paths: Tuple[str, ...] = ()
    env_flag: Optional[str] = None
    env_path: Optional[str] = None
    sha256: Optional[str] = None
    splits: Optional[SplitSpec] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset entry requires dataset_id")
        if self.registry_status not in (
            "permitted",
            "permitted_local_only",
            "blocked",
            "unverified",
        ):
            raise ValueError(
                f"{self.dataset_id}: bad registry_status "
                f"{self.registry_status!r}"
            )
        if self.gate is not None and self.gate not in GATES:
            raise ValueError(f"{self.dataset_id}: unknown gate {self.gate!r}")
        if self.gate is not None and not self.env_flag:
            raise ValueError(
                f"{self.dataset_id}: gated entries need env_flag"
            )
        if self.loader_kind not in (
            "twin",
            "locomo_json",
            "longmemeval_json",
            "none",
        ):
            raise ValueError(
                f"{self.dataset_id}: bad loader_kind {self.loader_kind!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """The V7-22.01 record shape (what ``datasets.json`` renders)."""
        return {
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "data_license": self.data_license,
            "code_license": self.code_license,
            "redistribution": self.redistribution,
            "local_storage_policy": self.local_storage_policy,
            "status": self.registry_status,
            "gate": self.gate,
            "env_flag": self.env_flag,
            "env_path": self.env_path,
            "paths": list(self.paths),
            "sha256": self.sha256,
            "loader_kind": self.loader_kind,
            "loader": self.loader,
            "splits": self.splits.to_dict() if self.splits else None,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# the registry (V7-22.03 dispositions as of the spec snapshot)
# ---------------------------------------------------------------------------

_TWIN_POLICY = "committed owned corpus — generated deterministically, no external data"
_LOCOMO_POLICY = (
    "permitted_local_only: never committed, never redistributed, "
    "derived artifacts redacted (V7-22.03)"
)

DATASETS: Tuple[DatasetEntry, ...] = (
    # --- owned twins (V7-22.05): license-free, gate None, always legal ---
    DatasetEntry(
        dataset_id="owned_locomo_like",
        source_url="owned",
        data_license="owned (verbatim project)",
        code_license="owned (verbatim project)",
        redistribution="yes",
        local_storage_policy=_TWIN_POLICY,
        registry_status="permitted",
        gate=None,
        loader_kind="twin",
        loader="eval.v7.twins_locomo_like",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="LoCoMo-category twin: multi-session dialogue, planted "
        "evidence, cat-like slices; seeded generator, no benchmark text.",
    ),
    DatasetEntry(
        dataset_id="owned_lme_like",
        source_url="owned",
        data_license="owned (verbatim project)",
        code_license="owned (verbatim project)",
        redistribution="yes",
        local_storage_policy=_TWIN_POLICY,
        registry_status="permitted",
        gate=None,
        loader_kind="twin",
        loader="eval.v7.twins_lme_like",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="LongMemEval-category twin: temporal, knowledge-update, "
        "abstention items; seeded generator.",
    ),
    DatasetEntry(
        dataset_id="owned_actions",
        source_url="owned",
        data_license="owned (verbatim project)",
        code_license="owned (verbatim project)",
        redistribution="yes",
        local_storage_policy=_TWIN_POLICY,
        registry_status="permitted",
        gate=None,
        loader_kind="twin",
        loader="eval.v7.twins_actions",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Track-A actions twin (DolphinBench-shaped categories).",
    ),
    DatasetEntry(
        dataset_id="owned_prefs",
        source_url="owned",
        data_license="owned (verbatim project)",
        code_license="owned (verbatim project)",
        redistribution="yes",
        local_storage_policy=_TWIN_POLICY,
        registry_status="permitted",
        gate=None,
        loader_kind="twin",
        loader="eval.v7.twins_prefs",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Preference-following twin corpus.",
    ),
    DatasetEntry(
        dataset_id="owned_scale",
        source_url="owned",
        data_license="owned (verbatim project)",
        code_license="owned (verbatim project)",
        redistribution="yes",
        local_storage_policy=_TWIN_POLICY,
        registry_status="permitted",
        gate=None,
        loader_kind="twin",
        loader="eval.v7.twins_scale",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Scale/envelope generator — volume filler for B-class rows.",
    ),
    # --- LoCoMo: CC BY-NC 4.0, O1-gated, local-only --------------------
    DatasetEntry(
        dataset_id="locomo",
        source_url="https://github.com/snap-research/locomo",
        data_license="CC BY-NC 4.0",
        code_license="see repo LICENSE",
        redistribution="no",
        local_storage_policy=_LOCOMO_POLICY,
        registry_status="permitted_local_only",
        gate="O1",
        loader_kind="locomo_json",
        loader="eval.v7.corpora:load_locomo",
        # V75-04.07 / J20: path list extended, not replaced — the
        # bundle copy is sha256-identical to the pin, so it resolves
        # through the same verification path (no file copied).
        paths=(
            "/tmp/locomo/locomo10.json",
            "research/v7_formula_search/locomo10.json",
        ),
        env_flag="VERBATIM_EVAL_LOCOMO",
        env_path="VERBATIM_EVAL_LOCOMO_PATH",
        # pinned digest of the authorized local artifact (V7-22.01)
        sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        splits=SplitSpec(unit="conversation", dev_pct=40),
        notes="locomo10.json: 10 conversations, ~5,882 turns, 1,986 QA "
        "(cats 1-5 per V7-22.07).  Loader only — no content committed.",
    ),
    # --- LongMemEval: code MIT, dataset terms unconfirmed → O2 ---------
    DatasetEntry(
        dataset_id="longmemeval_s",
        source_url="https://github.com/xiaowu0162/LongMemEval",
        data_license="see HF dataset card (unconfirmed at spec snapshot)",
        code_license="MIT",
        redistribution="local_only",
        local_storage_policy="download local-only pending O2; never committed",
        registry_status="unverified",
        gate="O2",
        loader_kind="longmemeval_json",
        loader="eval.v7.corpora:load_longmemeval",
        paths=("/tmp/longmemeval/longmemeval_s.json",),
        env_flag="VERBATIM_EVAL_LONGMEMEVAL",
        env_path="VERBATIM_EVAL_LONGMEMEVAL_S_PATH",
        sha256=None,  # unpinned until the O2 download lands
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="500 Q / ~115K tokens; adapter provisional until file reviewed.",
    ),
    DatasetEntry(
        dataset_id="longmemeval_m",
        source_url="https://github.com/xiaowu0162/LongMemEval",
        data_license="see HF dataset card (unconfirmed at spec snapshot)",
        code_license="MIT",
        redistribution="local_only",
        local_storage_policy="download local-only pending O2; never committed",
        registry_status="unverified",
        gate="O2",
        loader_kind="longmemeval_json",
        loader="eval.v7.corpora:load_longmemeval",
        paths=("/tmp/longmemeval/longmemeval_m.json",),
        env_flag="VERBATIM_EVAL_LONGMEMEVAL",
        env_path="VERBATIM_EVAL_LONGMEMEVAL_M_PATH",
        sha256=None,
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="~500 sessions/question scale probe (V7-24.05).",
    ),
    DatasetEntry(
        dataset_id="longmemeval_oracle",
        source_url="https://github.com/xiaowu0162/LongMemEval",
        data_license="see HF dataset card (unconfirmed at spec snapshot)",
        code_license="MIT",
        redistribution="local_only",
        local_storage_policy="download local-only pending O2; never committed",
        registry_status="unverified",
        gate="O2",
        loader_kind="longmemeval_json",
        loader="eval.v7.corpora:load_longmemeval",
        paths=("/tmp/longmemeval/longmemeval_oracle.json",),
        env_flag="VERBATIM_EVAL_LONGMEMEVAL",
        env_path="VERBATIM_EVAL_LONGMEMEVAL_ORACLE_PATH",
        sha256=None,
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Oracle-evidence LongMemEval variant.",
    ),
    # --- O3-gated: license unverified, default blocked -----------------
    DatasetEntry(
        dataset_id="beam",
        source_url="unverified — license review pending",
        data_license="unverified",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="blocked pending O3 review",
        registry_status="blocked",
        gate="O3",
        loader_kind="none",
        loader="",
        env_flag="VERBATIM_EVAL_BEAM",
        env_path="VERBATIM_EVAL_BEAM_PATH",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Ten abilities, 128K-10M buckets (V7-24.06/07).",
    ),
    DatasetEntry(
        dataset_id="dolphinbench",
        source_url="unverified — license review pending",
        data_license="unverified",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="blocked pending O3 review",
        registry_status="blocked",
        gate="O3",
        loader_kind="none",
        loader="",
        env_flag="VERBATIM_EVAL_DOLPHINBENCH",
        env_path="VERBATIM_EVAL_DOLPHINBENCH_PATH",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Runs through the Hermes harness once licensed (V7-22.06).",
    ),
    # --- named-but-blocked/unverified rows (V7-22.02/03 completeness) --
    DatasetEntry(
        dataset_id="halumem",
        source_url="https://github.com/MemTensor/HaluMem",
        data_license="CC BY-NC-ND 4.0",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="blocked: ND clause forbids derived artifacts",
        registry_status="blocked",
        gate=None,
        loader_kind="none",
        loader="",
        notes="Blocked outright per V7-22.03 — no O-gate exists for it.",
    ),
    DatasetEntry(
        dataset_id="memoryagentbench",
        source_url="unverified",
        data_license="MIT tag with inherited source terms",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="unverified pending license review",
        registry_status="unverified",
        gate=None,
        loader_kind="none",
        loader="",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="V7-22.02 adapter listed; status unverified (V7-22.03).",
    ),
    DatasetEntry(
        dataset_id="convomem",
        source_url="unverified",
        data_license="unverified",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="unverified pending license review",
        registry_status="unverified",
        gate=None,
        loader_kind="none",
        loader="",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="V7-22.02 adapter listed; review pending.",
    ),
    DatasetEntry(
        dataset_id="longmemeval_v2",
        source_url="unreleased",
        data_license="unverified",
        code_license="unverified",
        redistribution="no",
        local_storage_policy="awaiting released data + review",
        registry_status="unverified",
        gate=None,
        loader_kind="none",
        loader="",
        splits=SplitSpec(unit="question_group", dev_pct=30),
        notes="Adapter lands when released data is reviewed (V7-22.02).",
    ),
)

_BY_ID: Dict[str, DatasetEntry] = {e.dataset_id: e for e in DATASETS}

if len(_BY_ID) != len(DATASETS):
    raise AssertionError("duplicate dataset_id in registry")


# ---------------------------------------------------------------------------
# lookup + status
# ---------------------------------------------------------------------------


def get(dataset_id: str) -> DatasetEntry:
    """The registry entry for ``dataset_id`` (raises ``KeyError``)."""
    return _BY_ID[dataset_id]


def all_entries() -> Tuple[DatasetEntry, ...]:
    return DATASETS


def dataset_ids() -> Tuple[str, ...]:
    return tuple(e.dataset_id for e in DATASETS)


def _flag_on(name: Optional[str], env: Mapping[str, str]) -> bool:
    if not name:
        return False
    return env.get(name, "").strip().lower() in _FLAG_TRUE


def resolved_path(
    entry: DatasetEntry, env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """The local file path this entry resolves to right now.

    ``env_path`` overrides the configured ``paths``; otherwise the first
    existing configured path wins; otherwise the first configured path is
    returned for error reporting (``None`` when the entry has none).

    Relative configured paths are tried as given (caller CWD) and then
    relative to the repo root — the registry resolves identically from
    any working directory.
    """
    env = os.environ if env is None else env
    if entry.env_path:
        override = env.get(entry.env_path, "").strip()
        if override:
            return override
    for p in entry.paths:
        if os.path.isfile(p):
            return p
        if not os.path.isabs(p):
            rooted = os.path.join(_REPO_ROOT, p)
            if os.path.isfile(rooted):
                return rooted
    return entry.paths[0] if entry.paths else None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(
    ref: "str | DatasetEntry", env: Optional[Mapping[str, str]] = None
) -> str:
    """Operational status: ``available | blocked_on_authorization |
    missing_file | unavailable(<reason>)``.

    Order matters: the authorization gate is checked *before* the file —
    a blocked dataset never even looks at the filesystem.
    """
    e = get(ref) if isinstance(ref, str) else ref
    env = os.environ if env is None else env

    # 1. authorization gate (§37 default: undecided O-question blocks)
    if e.gate and not _flag_on(e.env_flag, env):
        return STATUS_BLOCKED

    # 2. license-blocked with no gate mechanism (e.g. HaluMem ND)
    if e.registry_status == "blocked" and not e.gate:
        return unavailable(f"license-blocked:{e.data_license}")

    # 3. twin generator presence (lazy — never imported here)
    if e.loader_kind == "twin":
        if _module_present(e.loader):
            return STATUS_AVAILABLE
        return unavailable(f"twin generator {e.loader} not landed")

    # 4. named-but-unimplemented adapter
    if e.loader_kind == "none":
        return unavailable("adapter not implemented")

    # 5. file-backed datasets: presence then pinned digest
    path = resolved_path(e, env)
    if path is None or not os.path.isfile(path):
        return STATUS_MISSING
    if e.sha256 and file_sha256(path) != e.sha256:
        return unavailable("sha256-mismatch")
    return STATUS_AVAILABLE


def _module_present(module_name: str) -> bool:
    """Importable check without executing the module."""
    if module_name in sys.modules:
        return getattr(sys.modules[module_name], "__spec__", None) is not None
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def status_detail(
    dataset_id: str, env: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Status plus the why: gate, flag state, resolved path, loader."""
    e = get(dataset_id)
    env = os.environ if env is None else env
    return {
        "dataset_id": e.dataset_id,
        "status": status(e, env),
        "registry_status": e.registry_status,
        "gate": e.gate,
        "flag_set": _flag_on(e.env_flag, env) if e.env_flag else None,
        "env_flag": e.env_flag,
        "resolved_path": resolved_path(e, env),
        "loader": e.loader,
        "splits": e.splits.to_dict() if e.splits else None,
    }


def summaries(env: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Any], ...]:
    """Per-dataset status rows for report/status tables."""
    return tuple(status_detail(e.dataset_id, env) for e in DATASETS)


# ---------------------------------------------------------------------------
# fixed dev/test partition (V7-22.08, V75-04.07)
# ---------------------------------------------------------------------------


def split_for_group_spec(
    spec: SplitSpec, dataset_id: str, group_key: str
) -> str:
    """``"dev"`` or ``"test"`` under an explicit :class:`SplitSpec`.

    Same hash formula as :func:`split_for_group`
    (``sha256({tag}:{dataset_id}:{group_key})`` mod 100 vs
    ``dev_pct``) — the spec is the *config*: ``dev_pct`` is the split
    fraction, ``tag`` the seed (a new tag = a new partition, V7-22.08).
    """
    blob = f"{spec.tag}:{dataset_id}:{group_key}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % 100
    return "dev" if bucket < spec.dev_pct else "test"


def split_for_group(dataset_id: str, group_key: str) -> str:
    """``"dev"`` or ``"test"`` for ``group_key`` — deterministic.

    The partition unit is the *conversation* or *question-group* id,
    never an individual question (V7-22.08).  Hashing
    ``{tag}:{dataset_id}:{group_key}`` makes the assignment fixed across
    runs and machines; only a bumped ``SplitSpec.tag`` repartitions.
    """
    e = get(dataset_id)
    if e.splits is None:
        raise ValueError(f"{dataset_id}: no split spec registered")
    return split_for_group_spec(e.splits, dataset_id, group_key)


def split_groups(
    dataset_id: str, group_keys: Iterable[str]
) -> Dict[str, Tuple[str, ...]]:
    """Partition ``group_keys`` into ``{"dev": ..., "test": ...}``."""
    dev, test = [], []
    for g in group_keys:
        (dev if split_for_group(dataset_id, g) == "dev" else test).append(g)
    return {"dev": tuple(dev), "test": tuple(test)}


def _partition_blob(
    dataset_id: str, spec: Optional[SplitSpec], dev, test
) -> bytes:
    """The canonical split-assignment serialization — the digest input."""
    return json.dumps(
        {
            "dataset_id": dataset_id,
            "spec": spec.to_dict() if spec else None,
            "dev": sorted(dev),
            "test": sorted(test),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def split_digest(dataset_id: str, group_keys: Iterable[str]) -> str:
    """Digest of a split assignment — pinned on tuning artifacts
    (V7-22.09 dev-split digest)."""
    e = get(dataset_id)
    parts = split_groups(dataset_id, group_keys)
    return hashlib.sha256(
        _partition_blob(dataset_id, e.splits, parts["dev"], parts["test"])
    ).hexdigest()


@dataclass(frozen=True)
class Partition:
    """A materialized dev/test split over a group-id set (V7-22.08).

    ``dev`` / ``test`` are sorted group-id tuples; ``digest`` is the
    canonical assignment digest a tuning artifact pins (V7-22.09 — the
    same value :func:`split_digest` reports for the registered spec).
    Nothing here reads dataset *content*: partitioning operates on ids
    alone, so the machinery exists and is tested before O1 lands; a
    committed partition of the gated file waits for the grant
    (V75-04.07).
    """

    dataset_id: str
    spec: SplitSpec
    dev: Tuple[str, ...]
    test: Tuple[str, ...]
    digest: str

    @property
    def groups(self) -> Tuple[str, ...]:
        """Every partitioned group id, sorted."""
        return tuple(sorted(self.dev + self.test))

    def assignment(self, group_key: str) -> Optional[str]:
        """``"dev"`` / ``"test"`` / ``None`` (not partitioned)."""
        if group_key in self.dev:
            return "dev"
        if group_key in self.test:
            return "test"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "spec": self.spec.to_dict(),
            "dev": list(self.dev),
            "test": list(self.test),
            "digest": self.digest,
        }


def materialize_partition(
    dataset_id: str,
    group_keys: Iterable[str],
    spec: Optional[SplitSpec] = None,
) -> Partition:
    """Materialize the fixed dev/test partition over ``group_keys``
    (V75-04.07).

    ``spec`` defaults to the registered entry's :class:`SplitSpec`
    (LoCoMo: ``unit="conversation"``, ``dev_pct=40`` — the *conversation*
    ``sample_id`` is the unit, never a question).  A caller may pass an
    explicit spec to vary the fraction (``dev_pct``) or seed (``tag``)
    for sensitivity checks; the partition digest changes accordingly and
    is recorded on the returned object.
    """
    if spec is None:
        e = get(dataset_id)
        if e.splits is None:
            raise ValueError(f"{dataset_id}: no split spec registered")
        spec = e.splits
    dev: list = []
    test: list = []
    for g in group_keys:
        (dev if split_for_group_spec(spec, dataset_id, g) == "dev"
         else test).append(g)
    digest = hashlib.sha256(
        _partition_blob(dataset_id, spec, dev, test)
    ).hexdigest()
    return Partition(
        dataset_id=dataset_id,
        spec=spec,
        dev=tuple(sorted(dev)),
        test=tuple(sorted(test)),
        digest=digest,
    )


# ---------------------------------------------------------------------------
# tuning-integrity check (V7-22.09, J16)
# ---------------------------------------------------------------------------


class SplitIntegrityError(RuntimeError):
    """A tuning artifact touched test data — or cannot prove it didn't
    (V7-22.09).  Raised by :func:`assert_not_tuned_on_test`; fails the
    run's integrity gate."""


#: Manifest keys a tuning artifact uses to declare the group ids its
#: fitting data contained.  All present keys union.
FITTED_GROUP_KEYS: Tuple[str, ...] = (
    "fitted_groups",
    "fitted_group_ids",
    "fit_groups",
    "tuned_on",
    "training_groups",
)

#: Manifest keys that pin the dev-split digest the artifact was fitted
#: under (V7-22.09).  All present keys must equal the partition digest.
SPLIT_DIGEST_KEYS: Tuple[str, ...] = (
    "dev_split_digest",
    "split_digest",
    "partition_digest",
)


def assert_not_tuned_on_test(
    manifest: Mapping[str, Any], partition: Partition
) -> None:
    """Integrity check (V7-22.09 / J16): a tuning artifact whose
    declared fitting data contains a test-partition group id fails.

    The artifact's *split provenance* is read from a flat manifest
    mapping: group ids under any :data:`FITTED_GROUP_KEYS` name, pinned
    split digest under any :data:`SPLIT_DIGEST_KEYS` name.  Failures —
    each a :class:`SplitIntegrityError`:

    * a declared fitted group is in the partition's ``test`` set;
    * a declared fitted group is in *neither* partition set (the
      artifact saw data this partition cannot account for);
    * a pinned digest differs from ``partition.digest`` (fitted under
      a different partition — the digest mismatch is the detection);
    * no provenance at all — V7-22.09 requires tuning artifacts to
      carry the dev-split digest, so an undeclared artifact is
      unverifiable and fails closed.
    """
    if not isinstance(manifest, Mapping):
        raise SplitIntegrityError(
            f"tuning artifact manifest must be a mapping, got "
            f"{type(manifest).__name__}"
        )
    fitted: set = set()
    for key in FITTED_GROUP_KEYS:
        for g in manifest.get(key) or ():
            fitted.add(str(g))
    digests = [
        str(manifest[k]) for k in SPLIT_DIGEST_KEYS if manifest.get(k)
    ]
    if not fitted and not digests:
        raise SplitIntegrityError(
            "tuning artifact declares no split provenance "
            f"(none of {FITTED_GROUP_KEYS + SPLIT_DIGEST_KEYS}) — "
            "V7-22.09 requires the dev-split digest on every fitted "
            "artifact"
        )
    test_set = set(partition.test)
    leaked = sorted(fitted & test_set)
    if leaked:
        raise SplitIntegrityError(
            f"tuning artifact fitted on test-partition group(s) "
            f"{leaked} of {partition.dataset_id!r} (V7-22.09)"
        )
    unknown = sorted(fitted - set(partition.dev) - test_set)
    if unknown:
        raise SplitIntegrityError(
            f"tuning artifact fitted on group(s) outside the "
            f"{partition.dataset_id!r} partition universe: {unknown}"
        )
    for d in digests:
        if d != partition.digest:
            raise SplitIntegrityError(
                f"tuning artifact pins split digest {d} but the "
                f"{partition.dataset_id!r} partition digest is "
                f"{partition.digest} — fitted under a different "
                "partition (V7-22.09)"
            )


# ---------------------------------------------------------------------------
# datasets.json rendering (V7-22.01) + CLI status table
# ---------------------------------------------------------------------------


def render_datasets_json() -> str:
    """The V7-22.01 ``eval/v7/datasets.json`` payload.

    Rendered on demand — the module is the registry; the JSON is a
    derived artifact a later wave may commit via ``--write``.
    """
    doc = {
        "registry": "eval/v7/dataset_registry.py",
        "spec": "SPEC_V7 V7-22.01/22.03/22.08",
        "split_tag": SPLIT_TAG,
        "datasets": [e.to_dict() for e in DATASETS],
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write",
        action="store_true",
        help="write eval/v7/datasets.json from the registry",
    )
    args = ap.parse_args(argv)
    if args.write:
        out = os.path.join(os.path.dirname(__file__), "datasets.json")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_datasets_json())
        print(f"wrote {out}")
        return 0
    for row in summaries():
        gate = row["gate"] or "-"
        flag = (
            f"{row['env_flag']}={'set' if row['flag_set'] else 'unset'}"
            if row["env_flag"]
            else "-"
        )
        print(f"{row['dataset_id']:<22} {row['status']:<28} gate={gate} {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
