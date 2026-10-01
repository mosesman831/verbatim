"""Near-duplicate signature — ``shingle/v1`` (§7, V5-30.06).

``shingle_signature(tokens)`` returns the frozenset of contiguous
k-token shingles (``SHINGLE_K = 3``) used for MinHash-class similarity
in the dedup lane. Deterministic: same token sequence ⇒ same signature.

Edge conventions (pinned, documented):

- ``len(tokens) < SHINGLE_K`` and non-empty → a single shingle covering
  the whole sequence (short records still get a comparable signature).
- empty token list → empty signature; similarity between an empty and
  any signature is the caller's problem — the signature itself stays
  honest.

Shingles are tuples of the *given* tokens — normalization upstream
(``normalize_text``) decides what "same" means; this layer never
re-normalizes.
"""

from __future__ import annotations

from typing import List

SHINGLE_VERSION = "shingle/v1"
SHINGLE_K = 3


def shingle_signature(tokens: List[str]) -> frozenset:
    """Frozenset of contiguous ``SHINGLE_K``-token shingles."""
    toks = [str(t) for t in (tokens or [])]
    if not toks:
        return frozenset()
    if len(toks) < SHINGLE_K:
        return frozenset({tuple(toks)})
    return frozenset(
        tuple(toks[i:i + SHINGLE_K])
        for i in range(len(toks) - SHINGLE_K + 1)
    )
