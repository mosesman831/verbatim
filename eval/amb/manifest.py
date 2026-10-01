"""AMB run-manifest builder — SPEC_V8 §15.2 (V8-15.06/15.08/15.11,
V7-22.10 conventions carried).

Every AMB row is produced from a manifest whose digest is printed beside
it; ``report_v8.md`` refuses to render a number without one
(V8-15.17/V7-22.24 carried).  The manifest records:

* ``provider`` — ``eval/amb/provider.py``'s ``PROVIDER_VERSION`` plus the
  verbatim head commit (``vcs.sha``).  A dirty tree is recorded, never
  laundered; a non-git root reports ``unavailable``.
* ``amb`` — the pinned harness coordinates: repository, commit,
  version, license state, leaderboard JSON digest, and the reader /
  judge / scorer prompt digests (V8-15.11, K88/K93).  Until the owner
  pins a checkout these carry ``pending_authorization`` — the manifest
  records the expectation honestly instead of inventing pins.
* ``dataset`` — dataset id, split, license, and sha256 pin
  (``pending_authorization`` while undelivered).
* ``models`` — AMB's pinned reader + judge model ids and decoding
  (``pending_authorization`` until O5), plus both context-token meters:
  AMB's authoritative ``cl100k_base`` (the meter behind Hindsight's
  36,235) and the provider-side ``tok/v1`` estimator used for budget
  enforcement (V8-15.08).
* ``arm`` — the verbatim arm config: timeout_ms=500 (product default,
  V8-15.06), token budget, k, doc_mode, encoder, worker, infer.
* ``authorizations`` — the O-decision snapshot the run validated
  against, with the ``blocked_on`` list when blocked.

Digest convention is ``eval/v7/manifest.py``'s: sha256 over the
canonical ``verbatim.core.serialize.json_dumps`` of the manifest sans
``digest`` — identical inputs, identical digest.

Manifests persist as ``eval/amb/manifests/<digest>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Iterable, Optional

#: Manifest schema tag (versioned like the v7 manifest tags).
MANIFEST_SCHEMA = "run_manifest/amb-v8"

#: Where manifests persist.
MANIFEST_DIR = os.path.join("eval", "amb", "manifests")

#: Placeholder marker for values gated on owner authorization — the
#: manifest carries the marker, never a fabricated pin.
PENDING = "pending_authorization"


def _json_dumps(value: Any) -> str:
    """Canonical JSON via the repo serialization seam (V5-06.01) — the
    eval.v7 convention duplicated so this module stands alone."""
    try:
        from verbatim.core.serialize import json_dumps
        return json_dumps(value)
    except Exception:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)


def _env_and_vcs(root: Optional[str], environment: Optional[dict],
                 vcs: Optional[dict]) -> tuple:
    """Reuse the eval.v7 collectors when present; degrade honestly."""
    if environment is None:
        try:
            from eval.v7.manifest import collect_environment
            environment = collect_environment()
        except Exception:
            environment = {"status": "unavailable"}
    if vcs is None:
        try:
            from eval.v7.manifest import git_state
            vcs = git_state(root) if root else {
                "system": "git", "status": "unavailable",
                "reason": "no root given", "sha": None, "dirty": None,
            }
        except Exception:
            vcs = {
                "system": "git", "status": "unavailable",
                "reason": "vcs collector unavailable",
                "sha": None, "dirty": None,
            }
    return environment, vcs


def default_amb_block() -> dict:
    """The pinned-harness coordinate block (V8-15.11) — placeholders
    until the owner records the pin in ``authorizations.json``."""
    return {
        "harness": "agent-memory-benchmark",
        "repo": "github.com/vectorize-io/agent-memory-benchmark",
        "commit": PENDING,
        "version": PENDING,
        "license": "pending_verification",
        "leaderboard_digest": None,
        "prompt_digests": {
            "reader_open": None,
            "reader_mcq": None,
            "judge": None,
        },
        "scorer_digest": None,
    }


def default_models_block() -> dict:
    """Reader/judge model coordinates — ``pending_authorization`` until
    O5 records AMB's pinned Gemini ids and decoding settings."""
    return {
        "reader": {
            "id": PENDING,
            "decoding": PENDING,
            "authorization": "O5",
        },
        "judge": {
            "id": PENDING,
            "decoding": PENDING,
            "authorization": "O5",
        },
        "context_meter": {
            "authoritative": "cl100k_base (AMB tiktoken meter)",
            "provider_estimate": "tok/v1",
            "note": (
                "AMB computes context_tokens itself on a real run; "
                "tok/v1 is the provider-side budget-enforcement "
                "estimator — both are pinned so the curve's meter is "
                "unambiguous"
            ),
        },
    }


def build_manifest(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    provider_version: str,
    arm: Optional[dict] = None,
    dataset: Optional[dict] = None,
    amb: Optional[dict] = None,
    models: Optional[dict] = None,
    authorizations: Optional[dict] = None,
    seeds: Iterable[int] = (),
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    environment: Optional[dict] = None,
    vcs: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build one AMB run manifest (V8-15.06/15.08/15.11 field set)."""
    environment, vcs = _env_and_vcs(root, environment, vcs)

    amb_block = default_amb_block()
    for k, v in (amb or {}).items():
        if isinstance(v, dict) and isinstance(amb_block.get(k), dict):
            amb_block[k] = {**amb_block[k], **v}
        else:
            amb_block[k] = v

    models_block = default_models_block()
    for k, v in (models or {}).items():
        if isinstance(v, dict) and isinstance(models_block.get(k), dict):
            models_block[k] = {**models_block[k], **v}
        else:
            models_block[k] = v

    ds = dict(dataset or {})
    ds.setdefault("id", None)
    ds.setdefault("split", None)
    ds.setdefault("digest", PENDING)
    ds.setdefault("license", PENDING)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "vcs": vcs,
        "environment": environment,
        "provider": {
            "name": "verbatim",
            "provider_version": provider_version,
            "verbatim_commit": vcs.get("sha"),
            "verbatim_dirty": vcs.get("dirty"),
        },
        "amb": amb_block,
        "dataset": ds,
        "models": models_block,
        "arm": dict(sorted((arm or {}).items())),
        "authorizations": dict(authorizations or {}),
        "seeds": sorted({int(s) for s in seeds}),
        "started_at": started_at,
        "ended_at": ended_at,
        "extra": dict(extra) if extra else {},
    }
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def manifest_digest(manifest: dict) -> str:
    """Canonical digest: sha256 over ``json_dumps`` of the manifest
    with ``digest`` removed — the V7-22.10 convention carried."""
    body = {k: v for k, v in manifest.items() if k != "digest"}
    return hashlib.sha256(_json_dumps(body).encode("utf-8")).hexdigest()


def verify_manifest(manifest: dict) -> bool:
    got = manifest.get("digest")
    return isinstance(got, str) and got == manifest_digest(manifest)


def write_manifest(manifest: dict, root: str) -> str:
    """Persist to ``eval/amb/manifests/<digest>.json``; returns the
    relative path.  Byte-stable canonical pretty JSON."""
    digest = manifest.get("digest") or manifest_digest(manifest)
    rel = os.path.join(MANIFEST_DIR, f"{digest}.json")
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True,
                           ensure_ascii=False) + "\n")
    return rel


def load_manifest(path: str) -> dict:
    """Strict load via the repo JSON seam when present."""
    try:
        from verbatim.core.serialize import json_loads
        with open(path, "r", encoding="utf-8") as f:
            return json_loads(f.read(), max_bytes=4 << 20, max_depth=64)
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.amb.manifest",
        description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        help="repo root (default: auto)")
    ap.add_argument("--example", action="store_true",
                    help="print a skeleton manifest (no execution)")
    args = ap.parse_args(argv)
    m = build_manifest(
        root=args.root, run_id="example",
        provider_version="verbatim-amb/1",
        arm={"name": "verbatim", "timeout_ms": 500.0,
             "token_budget": 4500, "k": 10, "doc_mode": "pack"},
        dataset={"id": "locomo10", "split": "test",
                 "license": "CC BY-NC 4.0"},
        authorizations={"decisions": {}, "blocked_on": ["O5"]},
    )
    print(json.dumps(m, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
