"""Native host adapters — the §15 depth-2 integration path (SPEC_V3 §13,
§49).

Every adapter here is a *thin edge*: it translates host events into the
canonical capture vocabulary and delegates to
:class:`~verbatim.sdk.CaptureClient`. No adapter implements admission,
authorization, screening, or storage (V3-49.04) — an adapter obtains
nothing the engine would deny a direct caller (V3-13.01).

- :class:`NativeAdapter` — the typed contract (``identity``,
  ``translate_event``, ``capture_auth``, ``emit``, ``capability_matrix``).
- :class:`GenericEventAdapter` — the canonical event-stream reference.
- :class:`HermesV3Adapter` — Hermes provider hooks → SDK (§13.03).
- :class:`AdkMemoryAdapter` — Google ADK MemoryService ops → SDK; the
  injectable ``AdkTransport`` is the only live-host dependency.
"""

from .adk import AdkMemoryAdapter, AdkTransport, adk_event_to_envelope, adk_scope_id
from .base import EVENT_OPS, GenericEventAdapter, NativeAdapter, dispatch
from .hermes_v3 import HermesV3Adapter
from .langchain import VerbatimRetriever

__all__ = [
    "EVENT_OPS",
    "AdkMemoryAdapter",
    "AdkTransport",
    "GenericEventAdapter",
    "HermesV3Adapter",
    "NativeAdapter",
    "adk_event_to_envelope",
    "adk_scope_id",
    "VerbatimRetriever",
    "dispatch",
]
