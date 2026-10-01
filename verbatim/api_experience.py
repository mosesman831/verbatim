"""Engine sibling — experience-memory API surface (V3-06.01).

Declared topic owner for episodes, transitions, and procedures on the
v2 ``Engine`` facade. The v2 Engine currently exposes no experience
entry points — v3 experience flows live under ``verbatim/experience/``
and the ``api_v3`` facade — so this mixin carries no methods yet. When
a v2-facade experience method is added it belongs here, not in
``api.py``.
"""

from __future__ import annotations


class ExperienceMixin:
    """Experience-memory topic (composed by the facade; no methods yet)."""

    __slots__ = ()


__all__ = ["ExperienceMixin"]
