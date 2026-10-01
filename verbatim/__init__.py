"""Verbatim: evidence-first, zero-generation memory engine for Hermes Agent.

Importing this package MUST NOT create databases, start workers, fetch
models, or make network calls (SPEC §6, §38). ``register`` is the Hermes
plugin entry point; it lazily imports the provider adapter.
"""

__version__ = "0.1.0"
__all__ = ["register", "Memory", "__version__"]

# Marker for Hermes plugin discovery: this file exposes register_memory_provider
# wiring through ``register`` and the MemoryProvider subclass in provider.py.


def __getattr__(name: str):
    """Lazy public exports — ``import verbatim`` stays side-effect free
    (no store creation, no worker startup, no model fetch)."""
    if name == "Memory":
        from .memory.facade import Memory

        return Memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register(ctx) -> None:
    """Register the Verbatim memory provider with Hermes.

    Performs registration only — no migration, inference, or background
    startup (SPEC §38).
    """
    from .provider import VerbatimMemoryProvider

    ctx.register_memory_provider(VerbatimMemoryProvider())
