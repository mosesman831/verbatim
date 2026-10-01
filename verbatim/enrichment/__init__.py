"""T1 deterministic enrichment (docs/v5_contracts.md §7).

Pure, versioned, recomputable functions over raw text — no store access,
no models, no network. Every artifact downstream stamps
``ENRICHMENT_VERSION = "enrich/v1"`` as producer. Contract surface:

- ``normalize_text(text, version="norm/v1")`` — pinned matching
  projection (case / whitespace / punctuation / Unicode-compat fold).
- ``normalized_digest(text)`` — blake2b over the normalized projection.
- ``extract_identifiers(text)`` — ``Identifier`` spans (paths, urls,
  emails, handles, versions, hashes, ticket keys, quoted strings, code
  tokens) with UTF-8 byte offsets, case preserved.
- ``extract_entities(text)`` — ``Entity`` candidates (capitalized spans,
  dictionary hits) with byte offsets.
- ``parse_temporal(text, anchor)`` — ``TemporalResult``: precision,
  status, ``event_at``/``event_end`` interval, ``anchor_at``; ambiguity
  resolves to ``unknown``, never a guessed date (V5-30.10/30.11).
- ``polarity(text)`` — ``Polarity`` from surface markers.
- ``classify_type(text)`` — ``MemoryType`` marker rules.
- ``shingle_signature(tokens)`` — k-shingle frozenset for MinHash dedup.
"""

from verbatim.memory.types import ENRICHMENT_VERSION

from .normalize import (
    NORMALIZATION_VERSION,
    SUPPORTED_VERSIONS,
    normalize_text,
    normalized_digest,
    utf8_offsets,
)
from .identifiers import (
    EXTRACT_VERSION,
    KNOWN_ENTITIES,
    Entity,
    Identifier,
    extract_entities,
    extract_identifiers,
)
from .temporal import (
    PARSER_VERSION,
    TemporalResult,
    parse_temporal,
)
from .polarity import polarity
from .typing import TYPING_PRODUCER, classify_type
from .dedup_sig import SHINGLE_K, SHINGLE_VERSION, shingle_signature

__all__ = [
    "ENRICHMENT_VERSION",
    "NORMALIZATION_VERSION",
    "SUPPORTED_VERSIONS",
    "normalize_text",
    "normalized_digest",
    "utf8_offsets",
    "EXTRACT_VERSION",
    "KNOWN_ENTITIES",
    "Identifier",
    "Entity",
    "extract_identifiers",
    "extract_entities",
    "PARSER_VERSION",
    "TemporalResult",
    "parse_temporal",
    "polarity",
    "TYPING_PRODUCER",
    "classify_type",
    "SHINGLE_K",
    "SHINGLE_VERSION",
    "shingle_signature",
]
