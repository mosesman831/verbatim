"""Mem0-shaped compatibility shims (SPEC_V5 §18.2).

Importing this package is side-effect free: the v5 facade is bound lazily
inside ``verbatim.compat.mem0`` so the shim remains importable while the
facade worker's modules are mid-flight.
"""

__all__ = [
    "Memory",
    "MemoryClient",
    "Mem0CompatError",
    "UnsupportedOptionError",
    "ConfirmationRequiredError",
    "NamespaceBindingError",
    "MemoryNotFoundError",
    "MemoryConflictError",
    "FacadeUnavailableError",
    "facade_available",
    "MEM0_COMPAT_SCHEMA",
    "MEM0_SURFACE_PIN",
]


def __getattr__(name):
    if name in __all__:
        from . import mem0

        return getattr(mem0, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
