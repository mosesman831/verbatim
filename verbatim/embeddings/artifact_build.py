"""Offline builder for pinned VBV1 word-table artifacts (SPEC_V6
V6-03.06/03.07, docs/v6_contracts.md §6).

No neural model can be downloaded in this environment — the honest V6
neural path is a *locally built, hash-pinned, parse-only* artifact. This
module is the build path V6-03.07 requires: given a deterministic
``{key: vector}`` table (produced offline by a recorded trainer — see
``eval/v6/neural.py::build_dev_artifact`` for the shipped "distilled
hashing" recipe), it writes:

- ``<out_dir>/<model>/<revision>/vectors.bin`` — the VBV1 payload::

      offset 0   magic            b"VBV1"            (4 bytes)
      offset 4   dim              u32 little-endian  (vector width)
      offset 8   count            u64 little-endian  (row count)
      offset 16  count records, sorted by key (UTF-8 byte order):
                 u16le key_len | key bytes (UTF-8) | dim x float32le

  Every record carries its key — the format is self-describing and the
  loader needs no side vocabulary. Parse-only: no pickle, no eval, no
  code objects; ``struct`` bounds every read.

- ``<out_dir>/artifact_manifest.json`` — the pin::

      {"format_version": 1,
       "models": {"<model>": {"<revision>": {
           "files": {"vectors.bin": {"sha256": ..., "size": ...}},
           "format": "VBV1", "dim": ..., "count": ...,
           "loader_source_sha256": <sha256 of artifact.py>,
           "build": {"builder": ..., "seed": ...}}}}}

``out_dir`` is the *models root*: the encoder resolves files under
``<data_dir>/models/<model>/<revision>/`` and accepts a manifest at
``<data_dir>/models/artifact_manifest.json`` (built artifacts) or
``<data_dir>/artifact_manifest.json`` (shipped-package convention), so
call ``build_artifact(<data_dir>/models, ...)`` — or simply pass the
workdir that ``eval/v6/neural.py::build_dev_artifact`` provisions.

Determinism: keys are serialized in UTF-8 byte order and the manifest is
``sort_keys`` JSON, so identical ``(model, revision, table, dim)``
inputs produce byte-identical outputs on every platform — the pinned
sha256 is stable across rebuilds (V6-03.07 reproducibility). ``seed``
does not alter the bytes (the table is already the deterministic
product); it is recorded in the manifest so the training seed travels
with the artifact.

Consumer wiring (facade integration — a later step, see
``eval/v6/neural.py`` for the eval-side injection that measures the
same path today):

.. code-block:: python

    Memory(
        path,
        encoder="artifact",            # facade.py encoder allowlist
        config={
            "data_dir": "<data_dir>",  # manifest+models resolved here
            "embedding": {
                "backend": "artifact",
                "model": "<model>",
                "artifact_revision": "<revision>",   # never "latest"
            },
        },
    )

requires ``verbatim/memory/facade.py`` to accept ``"artifact"`` in the
``encoder`` allowlist (``Memory.__init__``'s
``encoder not in ("hashing", "none")`` check) and ``_init_after_store``'s
``if encoder == "hashing":`` block to gain a sibling branch constructing
``ArtifactEncoder(cfg.embedding, data_dir=Path(cfg.data_dir))``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping, Optional, Sequence

MAGIC = b"VBV1"
FORMAT_ID = "VBV1"
MANIFEST_NAME = "artifact_manifest.json"
VECTORS_NAME = "vectors.bin"
_HEADER = struct.Struct("<4sIQ")          # magic, u32 dim, u64 count
_KEY_LEN = struct.Struct("<H")            # u16 key length
_MAX_KEY_BYTES = 4096
_MAX_DIM = 65536

#: Consumer-wiring snippet kept importable for docs/tests — the exact
#: config overrides the facade integration step needs.
CONFIG_SNIPPET = """\
Memory(path, encoder="artifact", config={
    "data_dir": "<data_dir>",
    "embedding": {"backend": "artifact",
                  "model": "<model>",
                  "artifact_revision": "<revision>"},
})
# facade.py must allow "artifact" in Memory.__init__'s encoder allowlist;
# _init_after_store constructs
# ArtifactEncoder(cfg.embedding, data_dir=Path(cfg.data_dir)).
"""


def _check_name(value: object, what: str) -> str:
    """Reject empty/traversing model+revision names — the pair is a path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    name = value.strip()
    if (
        name in (".", "..", "latest")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(
            f"{what} {value!r} is not a safe pinned name "
            "(no separators, '.', '..', or 'latest')"
        )
    return name


def _loader_source_sha256() -> str:
    """sha256 of the reviewed loader module (``artifact.py``).

    The manifest pins the exact loader source the artifact was built and
    reviewed against; the encoder re-derives it from its own ``__file__``
    at verify time.
    """
    loader = Path(__file__).with_name("artifact.py")
    return hashlib.sha256(loader.read_bytes()).hexdigest()


def _coerce_table(
    table: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    if not isinstance(table, Mapping) or not table:
        raise ValueError("table must be a non-empty mapping of key -> vector")
    out: dict[str, list[float]] = {}
    for key, vec in table.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"table key {key!r} must be a non-empty string")
        kb = key.encode("utf-8")
        if not kb or len(kb) > _MAX_KEY_BYTES:
            raise ValueError(
                f"table key {key!r} encodes to {len(kb)} bytes "
                f"(max {_MAX_KEY_BYTES})"
            )
        try:
            row = [float(x) for x in vec]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"table row for {key!r} is not numeric: {exc}"
            ) from exc
        if not row:
            raise ValueError(f"table row for {key!r} is empty")
        for x in row:
            if x != x or x in (float("inf"), float("-inf")):
                raise ValueError(
                    f"table row for {key!r} contains a non-finite component"
                )
        out[key] = row
    return out


def encode_vectors_bin(
    table: Mapping[str, Sequence[float]], dim: int
) -> bytes:
    """Serialize ``table`` to VBV1 bytes — deterministic, sorted by key."""
    rows = sorted(table.items(), key=lambda kv: kv[0].encode("utf-8"))
    parts = [_HEADER.pack(MAGIC, dim, len(rows))]
    for key, vec in rows:
        kb = key.encode("utf-8")
        parts.append(_KEY_LEN.pack(len(kb)))
        parts.append(kb)
        parts.append(struct.pack(f"<{dim}f", *vec))
    return b"".join(parts)


def build_artifact(
    out_dir: str,
    *,
    model: str,
    revision: str,
    table: Mapping[str, Sequence[float]],
    dim: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Write a pinned VBV1 artifact + manifest; return the manifest dict.

    ``out_dir`` is the models root — the artifact lands at
    ``<out_dir>/<model>/<revision>/vectors.bin`` with the manifest at
    ``<out_dir>/artifact_manifest.json``. To wire an
    :class:`~verbatim.embeddings.artifact.ArtifactEncoder`, point its
    ``data_dir`` at ``out_dir``'s parent (i.e. build into
    ``<data_dir>/models``).

    ``dim`` declares the vector width; when ``None`` it is inferred from
    the table (which must then be nonempty and uniform). Every row must
    match ``dim`` exactly — a ragged table is a build error, never a
    silently truncated artifact.
    """
    model = _check_name(model, "model")
    revision = _check_name(revision, "revision")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    rows = _coerce_table(table)

    widths = {len(v) for v in rows.values()}
    if len(widths) != 1:
        raise ValueError(
            f"table rows have inconsistent dims {sorted(widths)}"
        )
    inferred = widths.pop()
    if dim is None:
        dim = inferred
    if not isinstance(dim, int) or isinstance(dim, bool) or not (
        1 <= dim <= _MAX_DIM
    ):
        raise ValueError(f"dim {dim!r} must be an int in 1..{_MAX_DIM}")
    if dim != inferred:
        raise ValueError(
            f"dim {dim} does not match table row width {inferred}"
        )

    payload = encode_vectors_bin(rows, dim)
    sha = hashlib.sha256(payload).hexdigest()

    root = Path(out_dir)
    rev_dir = root / model / revision
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / VECTORS_NAME).write_bytes(payload)

    manifest = {
        "format_version": 1,
        "models": {
            model: {
                revision: {
                    "files": {
                        VECTORS_NAME: {"sha256": sha, "size": len(payload)},
                    },
                    "format": FORMAT_ID,
                    "dim": dim,
                    "count": len(rows),
                    "loader_source_sha256": _loader_source_sha256(),
                    "build": {
                        "builder": (
                            "verbatim.embeddings.artifact_build."
                            "build_artifact"
                        ),
                        "format": FORMAT_ID,
                        "seed": seed,
                        "keys_sorted": "utf8-bytes",
                    },
                }
            }
        },
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "CONFIG_SNIPPET",
    "FORMAT_ID",
    "MAGIC",
    "MANIFEST_NAME",
    "VECTORS_NAME",
    "build_artifact",
    "encode_vectors_bin",
]
