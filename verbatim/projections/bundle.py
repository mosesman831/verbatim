"""Portable projection bundles (SPEC_V4 V4-47.03/04, §47).

A bundle is a deterministic tar archive of one published projection
generation: every Markdown file plus ``manifest.json`` carrying the
store id, projection generation, builder identity, and per-file content
digests. It is a *view transfer* — never a mutation channel:

* **Export** (``export_bundle``) packages an existing verified
  generation dir; the files were already authorized per object at build
  time (V4-47.02) — the archive ships exactly those bytes, member names
  validated, digests re-verified before packing.
* **Verify** (``verify_bundle``) is pure inspection: size bounds,
  regular-file members only (no links/devices/dirs), single-level safe
  names (no ``..``, no absolute paths, no hidden/hook names), manifest
  shape, and sha256 over every shipped byte. Nothing is fetched and
  nothing executes — archives never carry hooks (V4-47.06).
* **Import** (``import_bundle``) verifies the archive, then classifies
  each projected claim against the *local* store: ``verified`` (same id
  + revision + object digest), ``foreign-unverifiable`` (no local row —
  honestly labeled, never promoted to evidence), or ``suppressed``
  (a live local tombstone — deleted content can never resurrect,
  V4-47.03/C68). Foreign provenance stays foreign: nothing written by
  import acquires local authority, and optional document ingest lands
  each ``scope-*.md`` as an ordinary ``import``-kind source through the
  real ingester — not as claims, not as grants (V4-47.04).

Archive determinism: uncompressed tar, sorted member names, zeroed
mtime/uid/gid/uname/gname, fixed modes. Identical input generations
produce byte-identical archives.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from typing import Any, Optional

from ..core.time import now_us as _now_us
from ..core.types import (
    ErrorCode,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from ..storage.repos import PurgesRepo
from . import markdown as _md
from . import paths as _paths

BUNDLE_FORMAT = "verbatim-projection-bundle"
BUNDLE_VERSION = 1

#: Hard bounds — a bundle is a transfer artifact, not a backup format.
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_MEMBERS = 10_000

#: Tar member modes — data files only; nothing in a bundle is runnable.
_MEMBER_MODE = 0o644


def _fail(code: ErrorCode, msg: str) -> VerbatimError:
    return VerbatimError(code, msg)


def _integrity(msg: str) -> VerbatimError:
    # V4-43.06: projection corruption is reported as projection
    # corruption (INTEGRITY) — never as canonical STORE_CORRUPT.
    return VerbatimError(ErrorCode.INTEGRITY, msg)


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


def _resolve_generation_dir(path: str) -> str:
    """Accept a projection root (resolve ``current``) or a gen dir."""
    if not isinstance(path, str) or not path.strip():
        raise _fail(ErrorCode.VALIDATION, "generation path required")
    ap = os.path.abspath(path)
    if os.path.isfile(os.path.join(ap, _paths.MANIFEST_FILE)):
        return ap
    target = _paths.current_target(ap)
    if target is not None and os.path.isfile(
        os.path.join(target, _paths.MANIFEST_FILE)
    ):
        return target
    raise _fail(
        ErrorCode.VALIDATION,
        f"{path!r} is neither a projection generation nor a root with "
        "a live 'current' pointer",
    )


def _load_manifest(gen_dir: str) -> dict[str, Any]:
    try:
        with open(
            os.path.join(gen_dir, _paths.MANIFEST_FILE), "rb"
        ) as fh:
            obj = safe_json_loads(fh.read().decode("utf-8", "replace"))
    except OSError as exc:
        raise _integrity(f"generation manifest unreadable: {exc}") from exc
    if not isinstance(obj, dict):
        raise _integrity("generation manifest is not an object")
    _validate_manifest(obj)
    return obj


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Structural manifest validation (V4-47.03)."""
    if manifest.get("format") != _md.FORMAT_TAG:
        raise _integrity("manifest format tag mismatch")
    if manifest.get("format_version") != _md.FORMAT_VERSION:
        raise _integrity("manifest format_version unsupported")
    if manifest.get("kind") != "projection-manifest":
        raise _integrity("manifest kind must be 'projection-manifest'")
    for key in ("store_id", "projection_generation", "builder", "files"):
        if key not in manifest:
            raise _integrity(f"manifest missing {key!r}")
    builder = manifest.get("builder")
    if not isinstance(builder, dict) or not builder.get("name"):
        raise _integrity("manifest builder identity malformed")
    files = manifest["files"]
    if not isinstance(files, dict) or len(files) > _MAX_MEMBERS:
        raise _integrity("manifest files map malformed")
    for name, meta in files.items():
        _paths.safe_relpath(name)
        if not isinstance(meta, dict):
            raise _integrity(f"manifest file entry {name!r} malformed")
        sha = meta.get("sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise _integrity(f"manifest file {name!r} digest malformed")
        nbytes = meta.get("bytes")
        if (
            isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes < 0
            or nbytes > _MAX_MEMBER_BYTES
        ):
            raise _integrity(f"manifest file {name!r} size malformed")
    if not isinstance(manifest.get("scopes"), list):
        raise _integrity("manifest scopes list malformed")


def export_bundle(gen_or_root: str, dest: str) -> dict[str, Any]:
    """Pack a published generation dir into a deterministic tar bundle.

    Every shipped member is re-hashed against the generation's
    ``manifest.json`` before packing — a corrupted projection file fails
    the export as ``INTEGRITY``, never silently repackages.
    """
    gen_dir = _resolve_generation_dir(gen_or_root)
    manifest = _load_manifest(gen_dir)

    members: list[tuple[str, bytes]] = []
    files_map: dict[str, dict[str, Any]] = dict(manifest["files"])
    for name in sorted(files_map):
        _paths.check_inside(gen_dir, os.path.join(gen_dir, name))
        try:
            with open(os.path.join(gen_dir, name), "rb") as fh:
                data = fh.read(_MAX_MEMBER_BYTES + 1)
        except OSError as exc:
            raise _integrity(f"bundle member {name!r} unreadable: {exc}") from exc
        if len(data) > _MAX_MEMBER_BYTES:
            raise _integrity(f"bundle member {name!r} exceeds size bound")
        meta = files_map[name]
        actual = hashlib.sha256(data).hexdigest()
        if actual != meta["sha256"] or len(data) != meta["bytes"]:
            raise _integrity(
                f"bundle member {name!r} digest mismatch — generation "
                "corrupted; rebuild before export"
            )
        members.append((name, data))

    manifest_bytes = (json_dumps(manifest) + "\n").encode("utf-8")
    members.append((_paths.MANIFEST_FILE, manifest_bytes))
    members.sort(key=lambda t: t[0])

    if not isinstance(dest, str) or not dest.strip():
        raise _fail(ErrorCode.VALIDATION, "destination path required")
    dest_abs = os.path.abspath(dest)
    tmp = dest_abs + ".tmp"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = _MEMBER_MODE
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    if len(blob) > _MAX_BUNDLE_BYTES:
        raise _fail(
            ErrorCode.EVIDENCE_TOO_LARGE, "bundle exceeds size bound"
        )
    _paths.write_file(tmp, blob)
    os.replace(tmp, dest_abs)
    return {
        "path": dest_abs,
        "bundle_sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "members": len(members),
        "store_id": manifest.get("store_id"),
        "projection_generation": manifest.get("projection_generation"),
        "builder": manifest.get("builder"),
    }


# ----------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------


def verify_bundle(path: str) -> dict[str, Any]:
    """Statically verify a bundle archive; returns manifest + digests.

    Raises ``INTEGRITY``/``VALIDATION`` on any failure — a bundle either
    verifies completely or not at all.
    """
    if not isinstance(path, str) or not path.strip():
        raise _fail(ErrorCode.VALIDATION, "bundle path required")
    ap = os.path.abspath(path)
    try:
        size = os.path.getsize(ap)
    except OSError as exc:
        raise _fail(ErrorCode.VALIDATION, f"bundle unreadable: {exc}") from exc
    if size > _MAX_BUNDLE_BYTES or size == 0:
        raise _integrity("bundle size out of bounds")
    bundle_sha = hashlib.sha256()
    contents: dict[str, bytes] = {}
    try:
        with open(ap, "rb") as fh:
            stream = fh.read()
    except OSError as exc:
        raise _fail(ErrorCode.VALIDATION, f"bundle unreadable: {exc}") from exc
    bundle_sha.update(stream)
    try:
        tf = tarfile.open(fileobj=io.BytesIO(stream), mode="r:*")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise _integrity(f"bundle is not a readable tar: {exc}") from exc
    with tf:
        members = tf.getmembers()
        if len(members) > _MAX_MEMBERS:
            raise _integrity("bundle member count out of bounds")
        names = []
        for m in members:
            # Regular files only — no links, dirs, devices, or any
            # other member type can ship in a bundle (V4-47.06).
            if m.type != tarfile.REGTYPE:
                raise _integrity(
                    f"bundle member {m.name!r} is not a regular file"
                )
            _paths.safe_relpath(m.name)
            if m.size > _MAX_MEMBER_BYTES or m.size < 0:
                raise _integrity(
                    f"bundle member {m.name!r} size out of bounds"
                )
            names.append(m.name)
        if len(set(names)) != len(names):
            raise _integrity("bundle contains duplicate member names")
        for m in members:
            extracted = tf.extractfile(m)
            if extracted is None:
                raise _integrity(f"bundle member {m.name!r} unreadable")
            data = extracted.read(_MAX_MEMBER_BYTES + 1)
            if len(data) > _MAX_MEMBER_BYTES or len(data) != m.size:
                raise _integrity(
                    f"bundle member {m.name!r} size mismatch"
                )
            contents[m.name] = data

    raw_manifest = contents.get(_paths.MANIFEST_FILE)
    if raw_manifest is None:
        raise _integrity("bundle lacks manifest.json")
    manifest = safe_json_loads(raw_manifest.decode("utf-8", "replace"))
    if not isinstance(manifest, dict):
        raise _integrity("bundle manifest is not an object")
    _validate_manifest(manifest)

    listed = set(manifest["files"])
    shipped = set(contents) - {_paths.MANIFEST_FILE}
    if shipped != listed:
        raise _integrity(
            "bundle member set != manifest file set "
            f"(extra={sorted(shipped - listed)}, "
            f"missing={sorted(listed - shipped)})"
        )
    files_out = []
    for name in sorted(listed):
        meta = manifest["files"][name]
        actual = hashlib.sha256(contents[name]).hexdigest()
        if actual != meta["sha256"] or len(contents[name]) != meta["bytes"]:
            raise _integrity(
                f"bundle member {name!r} content digest mismatch"
            )
        files_out.append(
            {"path": name, "sha256": actual, "bytes": len(contents[name])}
        )

    # Envelope-level consistency: every shipped scope file must carry a
    # projection header agreeing with the manifest.
    for srow in manifest.get("scopes") or []:
        if not isinstance(srow, dict):
            raise _integrity("manifest scope row malformed")
        fname = srow.get("file")
        if fname not in contents:
            raise _integrity(f"scope file {fname!r} not shipped")
        env = _md.parse_header(contents[fname].decode("utf-8", "replace"))
        if env is None:
            raise _integrity(f"scope file {fname!r} lacks a header envelope")
        if env.get("scope_id") != srow.get("scope_id"):
            raise _integrity(
                f"scope file {fname!r} header disagrees with manifest"
            )
        if env.get("store") != manifest.get("store_id"):
            raise _integrity(
                f"scope file {fname!r} store id disagrees with manifest"
            )

    return {
        "valid": True,
        "path": ap,
        "bundle_sha256": bundle_sha.hexdigest(),
        "manifest": manifest,
        "files": files_out,
        "contents": contents,  # caller may re-inspect; never persisted raw
    }


# ----------------------------------------------------------------------
# import — verify + classify against the local store
# ----------------------------------------------------------------------

_ENTRY_OPEN = f"<!-- {_md.ENTRY_MARK} "
_ENTRY_CLOSE = " -->"


def _iter_entry_markers(text: str):
    """Yield parsed ``verbatim:entry`` metadata dicts in file order.

    An unparseable marker yields ``None`` — the importer records the row
    as ``malformed`` rather than silently dropping it (honest labeling —
    a foreign file we can't verify is never quietly skipped).
    """
    pos = 0
    while True:
        start = text.find(_ENTRY_OPEN, pos)
        if start < 0:
            return
        end = text.find(_ENTRY_CLOSE, start + len(_ENTRY_OPEN))
        if end < 0:
            return
        try:
            meta = safe_json_loads(text[start + len(_ENTRY_OPEN):end])
        except Exception:
            meta = None
        yield meta if isinstance(meta, dict) else None
        pos = end + len(_ENTRY_CLOSE)


def _classify_claim(
    conn: Any,
    purges: PurgesRepo,
    meta: dict[str, Any],
) -> str:
    """Verifiability of one projected claim row on THIS store.

    ``verified`` — same id + revision + object digest present and live;
    ``suppressed`` — the claim (or any cited span/source) is under a live
    tombstone; ``foreign-unverifiable`` — no local row exists. The last
    is the honest label: foreign bytes prove foreign origin, never local
    authority (V4-47.04).
    """
    cid = meta.get("claim_id")
    rev = meta.get("revision")
    if not isinstance(cid, str) or not isinstance(rev, int):
        return "malformed"
    if cid in purges.suppressed_ids("claim", [cid]):
        return "suppressed"
    row = conn.execute(
        "SELECT cr.state, cr.object_json FROM claim_revisions cr"
        " WHERE cr.claim_id = ? AND cr.revision = ?",
        (cid, rev),
    ).fetchone()
    if row is None:
        return "foreign-unverifiable"
    state, object_json = row
    # Tombstoned deps also suppress the projected row.
    dep_ids = [
        e.get("span_id")
        for e in meta.get("evidence") or []
        if isinstance(e, dict) and isinstance(e.get("span_id"), str)
    ]
    src_ids = [
        e.get("source_id")
        for e in meta.get("evidence") or []
        if isinstance(e, dict) and isinstance(e.get("source_id"), str)
    ]
    if dep_ids and purges.suppressed_ids("span", dep_ids):
        return "suppressed"
    if src_ids and purges.suppressed_ids("source", src_ids):
        return "suppressed"
    want = meta.get("object_sha256")
    if want is not None:
        local_obj = safe_json_loads(object_json) if object_json else None
        if local_obj is None:
            return "digest-mismatch"
        have = hashlib.sha256(
            json_dumps(local_obj).encode("utf-8")
        ).hexdigest()
        if have != want:
            return "digest-mismatch"
    if state in ("erased",):
        return "suppressed"
    return "verified"


def import_bundle(
    store: Any,
    path: str,
    *,
    ingest: bool = False,
    target_scope: Optional[Scope] = None,
    ingester: Any = None,
) -> dict[str, Any]:
    """Verify a bundle and classify its rows against this store.

    Default mode is verification-only — the report is the product. With
    ``ingest=True`` each scope file whose claims carry no suppressed
    references is ingested as a *new document* (``SourceKind.IMPORT``,
    foreign provenance) into ``target_scope_id`` — the document itself is
    new input; the claims it *describes* are never minted locally, so a
    stale erased row can never resurrect (C68). Files whose rows touch a
    live local tombstone are refused outright (V4-47.03 deletion policy).
    """
    verified = verify_bundle(path)
    manifest = verified["manifest"]
    contents = verified["contents"]
    purges = PurgesRepo(store)

    claims_report: list[dict[str, Any]] = []
    scope_rows = manifest.get("scopes") or []
    blocked_files: set[str] = set()
    with store.read() as conn:
        for srow in scope_rows:
            fname = srow["file"]
            text = contents[fname].decode("utf-8", "replace")
            for meta in _iter_entry_markers(text):
                if meta is None:
                    claims_report.append(
                        {
                            "scope_id": srow.get("scope_id"),
                            "claim_id": None,
                            "revision": None,
                            "status": "malformed",
                        }
                    )
                    blocked_files.add(fname)
                    continue
                status = _classify_claim(conn, purges, meta)
                claims_report.append(
                    {
                        "scope_id": srow.get("scope_id"),
                        "claim_id": meta.get("claim_id"),
                        "revision": meta.get("revision"),
                        "status": status,
                    }
                )
                if status in ("suppressed", "digest-mismatch", "malformed"):
                    # Deletion policy + integrity: files asserting rows
                    # that contradict this store (or can't be verified
                    # at all) are refused for ingest — the foreign view
                    # is reported, never re-landed.
                    blocked_files.add(fname)

    # Self-consistency of shipped excerpts: declared quote digests must
    # recompute — a tampered quote fails even on a foreign store.
    quote_checks = 0
    for srow in scope_rows:
        text = contents[srow["file"]].decode("utf-8", "replace")
        for declared_sha, _declared_bytes, quote in _md.iter_quote_blocks(
            text
        ):
            if declared_sha is None:
                raise _integrity("quote block lacks declared digest")
            actual = hashlib.sha256(quote.encode("utf-8")).hexdigest()
            if actual != declared_sha:
                raise _integrity(
                    "quote block digest mismatch — bundle self-inconsistent"
                )
            quote_checks += 1

    ingested: list[dict[str, Any]] = []
    if ingest:
        if target_scope is None or not isinstance(target_scope, Scope):
            raise _fail(
                ErrorCode.VALIDATION,
                "ingest requires an explicit target Scope — imported "
                "documents never inherit the foreign partition",
            )
        if ingester is None:
            raise _fail(
                ErrorCode.VALIDATION,
                "ingest requires an Ingester instance",
            )
        for srow in scope_rows:
            fname = srow["file"]
            if fname in blocked_files:
                continue  # deletion policy: never re-land suppressed refs
            data = contents[fname]
            env = SourceEnvelope(
                origin="projection-bundle",
                source_kind=SourceKind.IMPORT,
                scope=target_scope,
                speaker_id=None,
                payload=data,
                event_us=_now_us(),
                captured_us=_now_us(),
                provenance=Provenance.LEGACY_IMPORT,
                external_id=f"bundle:{verified['bundle_sha256'][:16]}:{fname}",
                metadata={
                    "foreign_store_id": manifest.get("store_id"),
                    "foreign_scope_id": srow.get("scope_id"),
                    "projection_generation": manifest.get(
                        "projection_generation"
                    ),
                    "builder": manifest.get("builder"),
                    "bundle_sha256": verified["bundle_sha256"],
                    "foreign": True,
                },
            )
            receipt = ingester.ingest(env)
            ingested.append(
                {
                    "file": fname,
                    "accepted": list(receipt.accepted),
                    "duplicate": receipt.duplicate,
                }
            )

    counts = {"verified": 0, "foreign-unverifiable": 0, "suppressed": 0,
              "digest-mismatch": 0, "malformed": 0}
    for c in claims_report:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {
        "bundle_sha256": verified["bundle_sha256"],
        "foreign_store_id": manifest.get("store_id"),
        "projection_generation": manifest.get("projection_generation"),
        "builder": manifest.get("builder"),
        "claims": claims_report,
        "counts": counts,
        "files": verified["files"],
        "blocked_files": sorted(blocked_files),
        "quote_blocks_verified": quote_checks,
        "ingested": ingested,
        "authority": (
            "foreign rows carry foreign provenance only — no local "
            "authority, grant, or claim was created from them"
        ),
    }


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "export_bundle",
    "import_bundle",
    "verify_bundle",
]
