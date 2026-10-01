"""Projection-root filesystem safety (SPEC_V4 V4-47.06).

Every path a projection reads or writes is validated here: no symlink
escapes, no ``..`` traversal, no hidden/hook names (``.git``/``.hooks``
style directories are refused outright), no external fetches — this
package performs no I/O outside the validated root and never touches
the network.

Layout under a projection root::

    <root>/
        .staging-<build_id>/    staging generation — never reader-visible
            _checkpoint.json    resume state (deleted before publish)
            <files...>
        gen-000001/             immutable published generation
            manifest.json       commit marker — written LAST in staging
            index.md
            scope-*.md
        current -> gen-000001   atomic symlink flip at publication

Readers only ever traverse ``current``; ``.staging-*`` is scratch space.
Publishing is one ``os.rename`` of the staging dir plus one atomic
symlink replace — a reader sees the prior generation or the new one,
never a half-written tree (V4-43.02).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from ..core.types import ErrorCode, VerbatimError

GEN_PREFIX = "gen-"
STAGING_PREFIX = ".staging-"
CURRENT_LINK = "current"
CHECKPOINT_FILE = "_checkpoint.json"
MANIFEST_FILE = "manifest.json"
INDEX_FILE = "index.md"

#: Shipped file names: flat generation trees, ``.md``/``.json`` only.
_SHIPPED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(md|json)$")
_GEN_NAME = re.compile(r"^gen-(\d{6,})$")


def _fail(msg: str) -> "VerbatimError":
    return VerbatimError(ErrorCode.VALIDATION, msg)


def prepare_root(root: str) -> str:
    """Validate/create the projection root; returns its absolute path.

    The root itself must not be a symlink and must resolve to itself;
    every existing component is checked for links so a planted symlink
    cannot redirect writes outside the tree (V4-47.06).
    """
    if not isinstance(root, str) or not root.strip():
        raise _fail("projection root must be a path string")
    abspath = os.path.abspath(root)
    if os.path.islink(abspath):
        raise _fail("projection root is a symlink")
    if os.path.exists(abspath):
        if not os.path.isdir(abspath):
            raise _fail("projection root is not a directory")
        if os.path.realpath(abspath) != abspath:
            raise _fail("projection root does not resolve to itself")
    else:
        parent = os.path.dirname(abspath) or "."
        if os.path.exists(parent) and not os.path.isdir(parent):
            raise _fail("projection root parent is not a directory")
        try:
            os.makedirs(abspath, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED,
                f"cannot create projection root: {exc}",
            ) from exc
        try:
            os.chmod(abspath, 0o700)
        except OSError:
            pass
    return abspath


def safe_relpath(relpath: str) -> str:
    """Validate a single-level file name shipped inside a generation."""
    if not isinstance(relpath, str) or not relpath:
        raise _fail("relpath must be a non-empty string")
    if relpath != os.path.basename(relpath) or relpath in (".", ".."):
        raise _fail(f"relpath {relpath!r} escapes the generation tree")
    if relpath.startswith(".") or not _SHIPPED_NAME.match(relpath):
        raise _fail(f"relpath {relpath!r} is not a projection file name")
    return relpath


def check_inside(root: str, path: str) -> str:
    """Resolve ``path`` and require containment inside ``root`` — every
    existing ancestor is checked for symlinks."""
    real_root = os.path.realpath(root)
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        candidate = os.path.realpath(candidate)
    if candidate != real_root and not candidate.startswith(real_root + os.sep):
        raise _fail(f"path {path!r} escapes projection root")
    # Reject symlinked components between root and target.
    rel = os.path.relpath(candidate, real_root)
    if rel.startswith(".."):
        raise _fail(f"path {path!r} escapes projection root")
    cursor = real_root
    for part in rel.split(os.sep):
        if part in (".", "..", ""):
            continue
        if part.startswith(".") and not part.startswith(STAGING_PREFIX):
            raise _fail(f"hidden path component {part!r} rejected")
        cursor = os.path.join(cursor, part)
        if os.path.islink(cursor):
            # ``current`` is the one link the layout owns — it must point
            # inside the root and is only ever created by publish().
            if os.path.basename(cursor) != CURRENT_LINK:
                raise _fail(f"symlink component {cursor!r} rejected")
            target = os.path.realpath(cursor)
            if not target.startswith(real_root + os.sep):
                raise _fail("current link escapes the projection root")
    return candidate


def generation_name(seq: int) -> str:
    return f"{GEN_PREFIX}{seq:06d}"


def generation_seq(name: str) -> Optional[int]:
    m = _GEN_NAME.match(name)
    return int(m.group(1)) if m else None


def list_generations(root: str) -> list[tuple[int, str]]:
    """``[(seq, abspath)]`` for published generation dirs, oldest first."""
    out: list[tuple[int, str]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        seq = generation_seq(name)
        if seq is None:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            out.append((seq, path))
    out.sort(key=lambda t: t[0])
    return out


def list_staging(root: str) -> list[str]:
    """Absolute paths of ``.staging-*`` dirs (crash residue)."""
    out: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if not name.startswith(STAGING_PREFIX):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            out.append(path)
    return sorted(out)


def current_target(root: str) -> Optional[str]:
    """Generation dir ``current`` points at; ``None`` when unpublished.

    A ``current`` that is anything but an in-root symlink is reported as
    absent rather than followed blindly.
    """
    link = os.path.join(root, CURRENT_LINK)
    if not os.path.islink(link):
        return None
    target = os.path.realpath(link)
    real_root = os.path.realpath(root)
    if not target.startswith(real_root + os.sep):
        return None
    return target if os.path.isdir(target) else None


def write_file(path: str, data: bytes) -> None:
    """Write one file 0o600 + fsync, refusing symlink targets."""
    if os.path.islink(path):
        raise _fail(f"refusing to write through symlink {path!r}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED, f"cannot write {path}: {exc}"
        ) from exc


def remove_tree(path: str) -> None:
    """Best-effort recursive delete of a staging/generation dir.

    Refuses to follow a symlinked root. Symlinks *inside* the tree are
    unlinked, never traversed.
    """
    import shutil

    if os.path.islink(path):
        try:
            os.unlink(path)
        except OSError:
            pass
        return
    if not os.path.isdir(path):
        return
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "CHECKPOINT_FILE",
    "CURRENT_LINK",
    "GEN_PREFIX",
    "INDEX_FILE",
    "MANIFEST_FILE",
    "STAGING_PREFIX",
    "check_inside",
    "current_target",
    "generation_name",
    "generation_seq",
    "list_generations",
    "list_staging",
    "prepare_root",
    "remove_tree",
    "safe_relpath",
    "write_file",
]
