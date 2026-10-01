"""Persistent job queue package (SPEC §22)."""

from .queue import JobQueue, scope_key

__all__ = ["JobQueue", "scope_key"]
