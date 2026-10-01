"""Privacy layer: egress consent, budget reservations, minimization (SPEC §27),
and the v3 sensitive-value vault (SPEC_V3 §35–§37).

The vault surface keeps plaintext out of ordinary evidence: values are
sealed under per-entry AES-256-GCM data keys wrapped by externally
provisioned scope keys; hydration is consent-gated and defaults to opaque
one-use handles; action tickets are one-use gateway-checked authority;
erasure reports are honest about what key destruction can prove.
"""

from .broker import EndpointDescriptor, TransportBroker, payload_digest
from .egress import EgressGate, build_state, minimize, scope_key
from .erasure import erase_entry, report as erasure_report
from .hydration import (
    LOCAL_TRUSTED_PROCESSORS,
    HydrationResult,
    hydrate,
    issue_handle,
    resolve_handle,
)
from .keys import (
    KeyNotProvisioned,
    load_wrap_key,
    scope_slug,
)
from .redaction import (
    Redaction,
    RedactionResult,
    lookup_placeholder,
    redact_view,
    refs_for_view,
    spans_for_view,
)
from .tickets import issue as issue_ticket, verify as verify_ticket
from .vault import ALGORITHM, Vault, get_entry, open as vault_open, rewrap, seal

__all__ = [
    # v2 egress (unchanged) + v4 transport broker (§12)
    "EgressGate",
    "TransportBroker",
    "EndpointDescriptor",
    "payload_digest",
    "build_state",
    "minimize",
    "scope_key",
    # v3 vault (§35)
    "ALGORITHM",
    "Vault",
    "seal",
    "vault_open",
    "rewrap",
    "get_entry",
    "redact_view",
    "spans_for_view",
    "refs_for_view",
    "lookup_placeholder",
    "Redaction",
    "RedactionResult",
    # hydration + handles (§35.04)
    "hydrate",
    "issue_handle",
    "resolve_handle",
    "HydrationResult",
    "LOCAL_TRUSTED_PROCESSORS",
    # action tickets (§09.07)
    "issue_ticket",
    "verify_ticket",
    # erasure (§35.05, §36)
    "erase_entry",
    "erasure_report",
    # keys (§35.11, §37.02)
    "KeyNotProvisioned",
    "load_wrap_key",
    "scope_slug",
]
