"""V6 neural-artifact dev path — offline build, paired quality gate, and
calibration fitting (SPEC_V6 V6-03.07/03.09/03.10, docs/v6_contracts.md §6).

No neural model can be downloaded here (no network, no numpy/torch), so
the honest V6 neural path is a locally built, hash-pinned, parse-only
artifact. This module ships the three eval pieces:

* :func:`build_dev_artifact` — the V6-03.07 trainer. The pinned
  vocabulary is every normalized token of the seeded V5 dev corpus
  (item texts + task queries); each term's row is the *hashing
  encoder's* own vector for that term — a real deterministic word
  table, honestly "distilled hashing", not a pretend neural model.
  Training inputs (corpus digest), code identity, and seed travel in
  the manifest so the artifact is reproducible byte-for-byte.

* :func:`paired_quality` — the V6-03.10 gate. ``verbatim_memory`` runs
  on the same corpus/tasks under two arms: the production ``hashing``
  encoder and the built artifact encoder (injected into the facade's
  encoder slot — the same object ``encoder="artifact"`` will construct
  once the facade allowlist lands; see CONFIG_SNIPPET in
  ``verbatim/embeddings/artifact_build.py``). recall@k and precision@k
  are measured per arm; ``neural_recommended`` is True only when the
  artifact *strictly* improves recall with non-inferior precision —
  otherwise the loss is published with numbers.

* :func:`fit_calibration` — the V6-03.09 fit. Runs the pinned
  ``support_calibration/v1`` dev slice through the artifact encoder and
  reports the measured envelopes + ``separates`` for the fitted floor
  recorded in ``verbatim/querying/calibration.py``. The floor must be
  FIT here — even though the table is hashing-derived, the pooled
  word-mean geometry differs from the feature-hashed text geometry, so
  copying the 0.75 hashing floor would be exactly the copied-threshold
  failure V5-31.07 forbids.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Sequence

from eval.v5.corpus import ConsumerCorpus, seed_corpus
from eval.v5.harness import (
    ConsumerEnv,
    run_task,
    seed_corpus_env,
    settle,
)
from verbatim.config import EmbeddingConfig
from verbatim.embeddings.artifact import ArtifactEncoder
from verbatim.embeddings.artifact_build import build_artifact
from verbatim.embeddings.codec import Float32Codec
from verbatim.embeddings.hashing import HashingEncoder, _tokens

#: Pinned artifact identity — bump revision on any table/format change
#: (a new ``encoder_id`` then needs its own fitted calibration).
MODEL = "hash-distilled"
REVISION = "v1"
ENCODER_ID = f"artifact:{MODEL}:{REVISION}"

#: Table width — the hashing encoder's native 384, so distilled rows are
#: the hashing vectors verbatim (no projection layer to validate).
DIM = 384


# ---------------------------------------------------------------------
# build path (V6-03.07)
# ---------------------------------------------------------------------

def dev_vocabulary(corpus: ConsumerCorpus) -> list:
    """The pinned vocabulary: every normalized token in item texts and
    task queries — query terms are covered, so OOV at eval time means
    genuinely unseen text, not a vocabulary oversight."""
    toks = set()
    for item in corpus.items:
        toks.update(_tokens(item.text))
    for task in corpus.tasks:
        toks.update(_tokens(task.query))
    return sorted(toks)


def distilled_table(vocab: Sequence[str], *, dim: int = DIM) -> Dict:
    """One row per term: the hashing encoder's own float32 vector.

    Going through ``encode`` + ``Float32Codec.unpack`` (never the private
    featurizer) means the table stores exactly the float32 values the
    production encoder emits — the "distillation" is honest byte-level
    reuse, not a parallel reimplementation.
    """
    enc = HashingEncoder(EmbeddingConfig(backend="hashing"))
    table: Dict[str, list] = {}
    for tok in vocab:
        blob = enc.encode([tok])[0]
        table[tok] = list(Float32Codec.unpack(blob, dim))
    return table


def build_dev_artifact(
    workdir: str,
    *,
    memories: int = 32,
    seed: int = 42,
    model: str = MODEL,
    revision: str = REVISION,
) -> dict:
    """Train + write the pinned dev artifact under ``<workdir>/models``.

    Returns a receipt record: artifact paths, manifest, vocabulary size,
    corpus digest, and the ``encoder_id`` the artifact pins. Deterministic
    — same ``(memories, seed, model, revision)`` → identical bytes.
    """
    corpus = seed_corpus(memories=memories, seed=seed)
    vocab = dev_vocabulary(corpus)
    table = distilled_table(vocab, dim=DIM)
    models_root = os.path.join(workdir, "models")
    manifest = build_artifact(
        models_root,
        model=model,
        revision=revision,
        table=table,
        dim=DIM,
        seed=seed,
    )
    return {
        "workdir": workdir,
        "models_root": models_root,
        "model": model,
        "revision": revision,
        "encoder_id": f"artifact:{model}:{revision}",
        "dim": DIM,
        "vocab_size": len(vocab),
        "corpus": {"name": corpus.name, "seed": corpus.seed,
                   "items": len(corpus.items), "tasks": len(corpus.tasks),
                   "digest": corpus.digest()},
        "training": {
            "recipe": "distilled-hashing",
            "teacher": "hashing:subword-ngram:v1",
            "seed": seed,
            "note": (
                "each table row is the hashing encoder's float32 vector "
                "for that term — a real deterministic word table, not a "
                "downloaded or fabricated neural model"
            ),
        },
        "manifest": manifest,
    }


def dev_encoder(workdir: str, *, model: str = MODEL,
                revision: str = REVISION) -> ArtifactEncoder:
    """An ``ArtifactEncoder`` bound to ``<workdir>``'s built artifact."""
    cfg = EmbeddingConfig(
        backend="artifact", model=model, artifact_revision=revision,
    )
    return ArtifactEncoder(cfg, data_dir=Path(workdir))


# ---------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------

def _seed_artifact_env(corpus: ConsumerCorpus, workdir: str,
                       encoder: ArtifactEncoder) -> ConsumerEnv:
    """Seed a consumer env whose facade runs the artifact encoder.

    ``Memory(encoder=...)`` accepts only ``"hashing"``/``"none"`` today
    (facade.py's ``encoder not in ("hashing", "none")`` allowlist in
    ``Memory.__init__`` — the documented integration point), so the
    eval wires the encoder the same way the facade will: open with
    ``encoder="none"`` and *no* drain/forget inside ``seed_corpus_env``,
    inject the constructed ``ArtifactEncoder`` into the encoder slot
    (plus ``store.encoder`` so the v3 lane's ``_query_encoder`` probe
    resolves the same identity), then settle — the durable job queue's
    ``source_embed`` handler picks the injected encoder up at drain time
    exactly as it does for hashing. The forget distractor still goes
    through the real ``memory.forget`` route after a live settle.
    """
    env = seed_corpus_env(
        corpus,
        workdir=workdir,
        worker="external",
        drain=False,
        forget=False,
        memory_kwargs={"encoder": "none"},
    )
    mem = env.memory
    mem._encoder = encoder            # the facade's own slot
    mem._encoder_id = encoder.encoder_id
    mem._store.encoder = encoder      # v3-lane _query_encoder probe
    settle(env)                       # drain: projections + vectors
    for item in corpus.items:
        if not item.forget:
            continue
        ref = env.refs.get(item.id)
        if ref is None:
            env.add_errors.setdefault(item.id, "forget item never added")
            continue
        try:
            env.forget_results[item.id] = mem.forget(ref)
        except Exception as exc:  # noqa: BLE001 — recorded, kept visible
            env.add_errors[item.id] = (
                f"forget: {type(exc).__name__}: {exc}"
            )
    settle(env)
    return env


def _run_arm(corpus: ConsumerCorpus, env: ConsumerEnv, *, k: int) -> dict:
    scored = [run_task(env, t, k=k) for t in corpus.tasks]
    recalls = [s.recall for s in scored if s.recall is not None]
    precs = [s.precision for s in scored if s.precision is not None]
    abstain_tasks = [s for s in scored if s.expected_abstain]
    return {
        "tasks": len(scored),
        "recall_at_k": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "precision_at_k": (
            sum(precs) / len(precs) if precs else None
        ),
        "recall_n": len(recalls),
        "precision_n": len(precs),
        "abstain_correct": (
            sum(1 for s in abstain_tasks if s.abstained)
            if abstain_tasks else None
        ),
        "abstain_n": len(abstain_tasks),
        "forbidden_hits": sum(len(s.forbidden_hits) for s in scored),
        "errors": sum(1 for s in scored if s.error),
        # seeding diagnostics — the harness's honest-support surface
        "add_errors": dict(env.add_errors),
        "drain": dict(env.drain),
        "notes": list(env.notes),
        "per_task": {
            s.task_id: {
                "recall": s.recall,
                "precision": s.precision,
                "returned": list(s.returned_ids),
                "abstained": s.abstained,
                "error": s.error,
            }
            for s in scored
        },
    }


# ---------------------------------------------------------------------
# paired gate (V6-03.10)
# ---------------------------------------------------------------------

def paired_quality(
    workdir: str,
    *,
    memories: int = 32,
    seed: int = 42,
    k: int = 8,
) -> dict:
    """Run the paired hashing-vs-artifact quality comparison.

    Both arms seed the same corpus through the real consumer route and
    answer the same tasks; the only difference is the encoder pinned in
    the facade's slot. ``neural_recommended`` follows the V6-03.10 gate:
    recall@k must *strictly* improve and precision@k must be
    non-inferior; otherwise the artifact keeps the label unrecommended
    and the numbers are the report.
    """
    corpus = seed_corpus(memories=memories, seed=seed)
    built = build_dev_artifact(
        workdir, memories=memories, seed=seed,
    )
    artifact_enc = dev_encoder(workdir)
    artifact_available = artifact_enc.available()

    arms: Dict[str, dict] = {}
    hash_dir = os.path.join(workdir, "arm-hashing")
    try:
        env = seed_corpus_env(
            corpus,
            workdir=hash_dir,
            worker="external",
            memory_kwargs={"encoder": "hashing"},
        )
        try:
            arms["hashing"] = _run_arm(corpus, env, k=k)
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 — arm failure is data
        arms["hashing"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }

    art_dir = os.path.join(workdir, "arm-artifact")
    if artifact_available:
        try:
            env = _seed_artifact_env(corpus, art_dir, artifact_enc)
            try:
                arms["artifact"] = _run_arm(corpus, env, k=k)
            finally:
                env.close()
        except Exception as exc:  # noqa: BLE001 — arm failure is data
            arms["artifact"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        arms["artifact"] = {
            "unavailable": True,
            "reason": "artifact failed verification — arm not run",
        }

    h = arms["hashing"]
    a = arms["artifact"]
    if a.get("unavailable") or a.get("error") or h.get("error"):
        recommended = False
        delta = None
        if a.get("unavailable"):
            reason = (
                "artifact encoder unavailable — cannot earn the "
                "local_memory_neural label"
            )
        else:
            reason = (
                "paired run incomplete "
                f"(hashing={h.get('error') or 'ok'}, "
                f"artifact={a.get('error') or 'ok'}) — gate undecidable"
            )
    else:
        hr, ar = h["recall_at_k"], a["recall_at_k"]
        hp, ap = h["precision_at_k"], a["precision_at_k"]
        delta = {
            "recall": (
                (ar - hr) if (ar is not None and hr is not None) else None
            ),
            "precision": (
                (ap - hp) if (ap is not None and hp is not None) else None
            ),
        }
        if hr is None or ar is None:
            recommended = False
            reason = "no recall-gold tasks measured — gate undecidable"
        else:
            strict_recall = ar > hr
            noninferior_precision = hp is None or ap is None or (
                ap >= hp - 1e-9
            )
            recommended = bool(strict_recall and noninferior_precision)
            reason = (
                f"recall@k artifact={ar:.4f} vs hashing={hr:.4f} "
                f"({'strictly better' if strict_recall else 'not strictly better'}); "
                f"precision@k artifact={ap if ap is not None else float('nan'):.4f} "
                f"vs hashing={hp if hp is not None else float('nan'):.4f} "
                f"({'non-inferior' if noninferior_precision else 'INFERIOR'})"
            )

    return {
        "suite": "v6-neural-paired",
        "measured": True,
        "qualification": "locally_measured",
        "corpus": {
            "name": corpus.name,
            "seed": corpus.seed,
            "items": len(corpus.items),
            "tasks": len(corpus.tasks),
            "digest": corpus.digest(),
        },
        "artifact": {
            "model": MODEL,
            "revision": REVISION,
            "encoder_id": ENCODER_ID,
            "dim": DIM,
            "vocab_size": built["vocab_size"],
            "available": artifact_available,
            "recipe": built["training"]["recipe"],
        },
        "k": k,
        "arms": {
            "verbatim_memory-hashing": h,
            "verbatim_memory-artifact": a,
        },
        "delta": delta,
        "verdict": {
            "neural_recommended": recommended,
            "reason": reason,
            "gate": (
                "V6-03.10: recall@k strictly improves AND precision@k "
                "non-inferior, else the label stays unrecommended"
            ),
        },
    }


# ---------------------------------------------------------------------
# calibration fit (V6-03.09)
# ---------------------------------------------------------------------

def fit_calibration(workdir: str, *, memories: int = 32,
                    seed: int = 42) -> dict:
    """Run the pinned ``support_calibration/v1`` dev slice through the
    built artifact encoder and report the measured fit.

    The floor recorded in ``verbatim/querying/calibration.py`` for
    ``artifact:hash-distilled:v1`` was fitted by this function — pooled
    word-mean geometry differs from feature-hashed text geometry, so the
    hashing floor is never copied across.
    """
    from verbatim.querying.calibration import validate

    build_dev_artifact(workdir, memories=memories, seed=seed)
    enc = dev_encoder(workdir)
    rep = validate(enc)
    rep["artifact_available"] = enc.available()
    rep["fitted_encoder_id"] = ENCODER_ID
    return rep


def main() -> None:  # pragma: no cover — manual run entry
    workdir = tempfile.mkdtemp(prefix="verbatim-v6-neural-")
    rep = paired_quality(workdir)
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "DIM",
    "ENCODER_ID",
    "MODEL",
    "REVISION",
    "build_dev_artifact",
    "dev_encoder",
    "dev_vocabulary",
    "distilled_table",
    "fit_calibration",
    "paired_quality",
]
