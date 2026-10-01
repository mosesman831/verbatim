"""``local_files`` connector (SPEC_V4 §48) — import UTF-8 text/markdown
files from a local directory.

The reference connector for the pull framework: a directory tree of
``.txt``/``.md`` (configurable) files becomes ``connector_item``
envelopes through the real write channel.

Honesty rules:

- Path containment is enforced by ``realpath`` + ``commonpath``: a
  symlink or ``..`` escape is refused at ``validate_source``/``scan``
  time — escaping entries surface as rejected items, never followed.
- Bytes are passed through verbatim. Malformed UTF-8 is NOT decoded here
  — the pull engine's write-channel gate rejects it per item and the
  rejection is recorded (C85 / V4-13.11), never silently skipped.
- Cursor = the last committed relative POSIX path; scan resumes with
  strictly-later paths in sorted order. A file added later that sorts
  behind the cursor is not picked up until the cursor is reset —
  documented bound of a name-ordered ledger (``resume_from=""``
  rescans; dedup makes it cheap).
- ``max_file_bytes`` is the connector-level read bound; the pull
  engine's ``capture.max_source_bytes`` bound still applies per item.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError
from .base import ConnectorDescriptor, RemoteItem

CONNECTOR_ID = "local_files"
CONNECTOR_VERSION = "1.0"

_DEFAULT_EXTENSIONS = (".txt", ".md", ".markdown")
_DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024
_HARD_MAX_FILE_BYTES = 64 * 1024 * 1024

_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
}


class LocalFilesConnector:
    """Directory importer. ``source`` descriptor::

        {"dir": "/abs/path",            # required
         "extensions": [".txt", ".md"], # optional, default text/markdown
         "recursive": true,             # optional, default true
         "max_file_bytes": 4194304}     # optional read bound
    """

    def descriptor(self) -> ConnectorDescriptor:
        return ConnectorDescriptor(
            connector_id=CONNECTOR_ID,
            version=CONNECTOR_VERSION,
            display_name="Local UTF-8 text/markdown directory",
            remote=False,
            formats=("text/plain", "text/markdown"),
            item_classes=("document",),
            capabilities={
                "revisions": True,
                "dry_run": True,
                "deletions": False,
                "remote_fetch": False,
            },
        )

    def validate_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(source, Mapping):
            raise VerbatimError(
                ErrorCode.VALIDATION, "local_files source must be a mapping"
            )
        raw_dir = source.get("dir")
        if not isinstance(raw_dir, str) or not raw_dir:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "local_files source requires a non-empty 'dir'",
            )
        # realpath collapses ``..`` and symlink hops; the root itself must
        # exist and be a directory — an unreachable source is a pull-spec
        # error, not an empty pull.
        root = os.path.realpath(raw_dir)
        if not os.path.isdir(root):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"local_files dir {raw_dir!r} is not a directory",
            )
        exts = source.get("extensions", _DEFAULT_EXTENSIONS)
        if not isinstance(exts, (list, tuple)) or not exts:
            raise VerbatimError(
                ErrorCode.VALIDATION, "extensions must be a non-empty list"
            )
        norm_exts = []
        for e in exts:
            if not isinstance(e, str) or not e.startswith("."):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "extensions entries must be dotted suffixes like '.txt'",
                )
            norm_exts.append(e.lower())
        recursive = source.get("recursive", True)
        if not isinstance(recursive, bool):
            raise VerbatimError(
                ErrorCode.VALIDATION, "recursive must be a boolean"
            )
        max_bytes = source.get("max_file_bytes", _DEFAULT_MAX_FILE_BYTES)
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not (1 <= max_bytes <= _HARD_MAX_FILE_BYTES)
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"max_file_bytes must be an int in [1, {_HARD_MAX_FILE_BYTES}]",
            )
        return {
            "dir": root,
            "extensions": sorted(set(norm_exts)),
            "recursive": recursive,
            "max_file_bytes": max_bytes,
        }

    def scan(
        self,
        source: Mapping[str, Any],
        *,
        cursor: Optional[str],
        limit: Optional[int] = None,
    ) -> Iterator[RemoteItem]:
        root = source["dir"]
        exts = frozenset(source.get("extensions") or _DEFAULT_EXTENSIONS)
        recursive = bool(source.get("recursive", True))
        max_bytes = int(source.get("max_file_bytes") or _DEFAULT_MAX_FILE_BYTES)

        paths, escaped = self._enumerate(root, exts, recursive)
        emitted = 0
        for rel in paths:
            if cursor is not None and rel <= cursor:
                continue
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield self._item(root, rel, max_bytes, escaped)

    # ------------------------------------------------------------------

    def _enumerate(
        self, root: str, exts: frozenset, recursive: bool
    ) -> tuple[list[str], frozenset]:
        """Sorted relative POSIX paths of in-root files + the escapee set.

        Entries whose realpath escapes the root (or that are symlinks)
        keep a cursor position and surface as rejected items — the pull
        reports them instead of skipping silently.
        """
        out: list[str] = []
        escapees: list[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(
                root, followlinks=False
            ):
                # followlinks=False already refuses to descend into
                # symlinked dirs; removing them keeps the listing honest
                # and records the escape position.
                for d in list(dirnames):
                    full = os.path.join(dirpath, d)
                    if os.path.islink(full):
                        dirnames.remove(d)
                        escapees.append(self._rel(root, full))
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = self._rel(root, full)
                    if os.path.islink(full) or not self._contained(
                        root, full
                    ):
                        escapees.append(rel)
                        continue
                    if os.path.splitext(name)[1].lower() in exts:
                        out.append(rel)
        else:
            try:
                names = sorted(os.listdir(root))
            except OSError as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"local_files cannot list {root!r}: {exc}",
                ) from exc
            for name in names:
                full = os.path.join(root, name)
                if os.path.islink(full) or not self._contained(root, full):
                    escapees.append(name)
                    continue
                if os.path.isfile(full) and (
                    os.path.splitext(name)[1].lower() in exts
                ):
                    out.append(name)
        # Escapees merge into the same sorted order so cursor positions
        # stay consistent between scans.
        merged = sorted(set(out) | set(escapees))
        return merged, frozenset(escapees)

    @staticmethod
    def _rel(root: str, full: str) -> str:
        return os.path.relpath(full, root).replace(os.sep, "/")

    @staticmethod
    def _contained(root: str, full: str) -> bool:
        try:
            return (
                os.path.commonpath((root, os.path.realpath(full))) == root
            )
        except ValueError:
            return False

    def _item(
        self, root: str, rel: str, max_bytes: int, escaped: frozenset
    ) -> RemoteItem:
        full = os.path.join(root, rel)
        if rel in escaped:
            return RemoteItem(
                external_id=rel,
                cursor=rel,
                content=b"",
                reject_reason="path_escapes_root",
            )
        try:
            st = os.stat(full)
            if st.st_size > max_bytes:
                return RemoteItem(
                    external_id=rel,
                    cursor=rel,
                    content=b"",
                    reject_reason="too_large",
                    event_us=int(st.st_mtime * 1_000_000),
                )
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            return RemoteItem(
                external_id=rel,
                cursor=rel,
                content=b"",
                reject_reason=f"unreadable:{exc.__class__.__name__}",
            )
        ext = os.path.splitext(rel)[1].lower()
        return RemoteItem(
            external_id=rel,
            cursor=rel,
            content=data,
            media_type=_MEDIA_TYPES.get(ext, "text/plain"),
            item_class="document",
            event_us=int(st.st_mtime * 1_000_000),
        )


__all__ = ["CONNECTOR_ID", "LocalFilesConnector"]
