"""Deterministic text normalization — ``norm/v1`` (SPEC_V5 §30.2, V5-30.06).

``norm/v1`` is a pinned matching projection, never a rewrite of stored
bytes:

    NFKC compatibility fold  →  NFKD decomposition with combining marks
    dropped  →  casefold  →  punctuation / symbol / separator / control
    categories folded to a single space  →  whitespace collapse + strip.

Every artifact that consumes this projection records the version string;
an unknown ``version`` raises ``ValueError`` rather than silently
producing an unlabeled projection, so a future ``norm/v2`` can never be
confused with this output (V5-30.06 "pinned normalization version").

Also hosts the two byte-level helpers shared by the extraction modules:
``utf8_offsets`` (char-index → UTF-8 byte-offset table — every span this
package emits is a byte offset into the original text, V5-30.14) and
``normalized_digest`` (the content digest used for exact/near-duplicate
linking, V5-30.05/30.06).

Pure functions, stdlib only, no store access (§7 contract).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import List

NORMALIZATION_VERSION = "norm/v1"
SUPPORTED_VERSIONS = frozenset({NORMALIZATION_VERSION})

_WS_RE = re.compile(r"\s+")

#: Unicode general-category groups folded to a single space:
#: P* punctuation, S* symbols, C* control/format/surrogate/unassigned,
#: Z* separators. Letters (L*), numbers (N*) stay; marks (M*) are dropped
#: entirely after NFKD decomposition so accents fold ("é" → "e").
_FOLD_GROUP = frozenset({"P", "S", "C", "Z"})

#: Domain-separated personalization for the normalized-content digest —
#: distinct from embeddings/hashing.py's ``verbatim-h1`` so a dedup digest
#: can never collide with a feature hash by construction.
_DIGEST_PERSON = b"verbatim-n1"
_DIGEST_SIZE = 16


def normalize_text(text: str, *, version: str = NORMALIZATION_VERSION) -> str:
    """Canonical matching projection of ``text`` under ``norm/v1``.

    Case-folded, punctuation/Unicode-compat folded, whitespace collapsed.
    Deterministic: same input + same version ⇒ same output on every host.
    """
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported normalization version {version!r}; "
            f"supported: {sorted(SUPPORTED_VERSIONS)}"
        )
    s = str(text if text is not None else "")
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFKD", s)
    out: List[str] = []
    for ch in s:
        group = unicodedata.category(ch)[0]
        if group == "M":
            continue  # combining mark — dropped after decomposition
        if group in _FOLD_GROUP:
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out).casefold()
    return _WS_RE.sub(" ", s).strip()


def normalized_digest(text: str) -> str:
    """blake2b-128 hex digest over the ``norm/v1`` projection (V5-30.06).

    Byte-identical-after-normalization texts share a digest; the digest
    is stable across hosts and runs (fixed personalization, no salt).
    """
    normed = normalize_text(text)
    return hashlib.blake2b(
        normed.encode("utf-8"),
        digest_size=_DIGEST_SIZE,
        person=_DIGEST_PERSON,
    ).hexdigest()


def utf8_offsets(text: str) -> List[int]:
    """Cumulative UTF-8 byte offset for each char index.

    ``offsets[i]`` is the byte offset of char ``i`` in
    ``text.encode("utf-8")``; ``offsets[len(text)]`` is the encoded
    length. Extraction modules match on ``str`` and translate through
    this table so every emitted ``start``/``end`` is a byte offset into
    the original bytes (V5-30.14).
    """
    offsets = [0] * (len(text) + 1)
    pos = 0
    for i, ch in enumerate(text):
        pos += len(ch.encode("utf-8"))
        offsets[i + 1] = pos
    return offsets
