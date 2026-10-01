"""Generation-scoped Markdown projection builder (SPEC_V4 §43, §47).

A *projection generation* is a complete, immutable directory tree under
``<root>/gen-NNNNNN/`` that readers reach only through the ``current``
symlink. Builds stage into ``<root>/.staging-<build_id>/`` — never
reader-visible (V4-43.02) — and publish by one ``os.rename`` plus one
atomic symlink replace: a reader sees the prior generation or the new
one, never a half-written tree.

Lifecycle (maps onto ``projection_builds.status`` — the persisted
vocabulary is the v2 CHECK set):

    building   → staging dir + ``_checkpoint.json`` being written
    ready      → every staged file re-hashed clean; ``manifest.json`` last
    published  → generation dir + ``current`` flipped; CAS into
                 ``active_projections`` (kind ``markdown-files`` per scope)
    failed     → build abandoned; staging left for inspection/resume
    retired    → generation superseded; ``reclaim`` deletes the dir

Resume (V4-43.05): a checkpoint records ``build_id``, ``db_id``, builder
identity, projection generation, and the input-set digest. ``build()``
re-derives the input digest in a fresh snapshot — a match resumes the
staged work (per-file digests re-verified), a mismatch discards the
partial artifacts safely and reports them.

Suppression (V4-43.09): rendering reads through the same quarantine +
purge-tombstone cascade as export, and excerpt bytes only ever come
from ``Kernel.read_verified`` under a live lease — a ``quote``-less
authority produces metadata+reference files, never withheld bytes
(V4-08.02, §47).

Scope-partition rebuilds (V4-43.04): ``build(scope_ids=[...])`` re-renders
only those scopes' files; sibling scope files are carried into the new
generation verbatim — but only when the build authority still holds
``read`` on them *and* the prior bytes re-verify against the recorded
digest + fresh eligibility digest. A scope the authority can no longer
read is dropped and reported, never carried.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from ..core.time import now_us as _now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import Verb
from ..core.types_v4 import EvidenceLocator
from ..governance import CallerV3
from ..kernel import Kernel
from ..storage.repos import (
    EventsRepo,
    has_table as _has_table,
)
from ..storage.repos_v2 import ProjectionRepo
from . import paths as _paths
from . import markdown as _md
from . import _withhold as _wh

#: ``active_projections.projection_kind`` for file projections.
PROJECTION_KIND = "markdown-files"

#: Bounded lease window for a whole build pass — re-resolved per scope.
_LEASE_MAX_AGE_US = 1_000_000

#: Refuse absurd outputs before they hit the filesystem.
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_FILES = 10_000

_EVENT_POLICY = "projection-1"


@dataclass(frozen=True)
class ProjectionAuthority:
    """The explicit authority a projection build runs under (§47).

    ``principal_id`` + ``purpose`` are evaluated through
    ``governance.authorize`` per scope: ``quote`` releases exact excerpt
    bytes into the files; ``read`` alone yields metadata + references
    with an explicit withheld marker; neither → the scope is omitted and
    counted. There is no ambient default — a caller that passes nothing
    gets nothing.
    """

    principal_id: str
    purpose: str = "recall"
    session_id: str = ""
    host_id: str = ""
    epoch: Optional[int] = None

    def caller(self) -> CallerV3:
        require_id(self.principal_id, "principal_id")
        if not isinstance(self.purpose, str) or not self.purpose:
            raise VerbatimError(
                ErrorCode.VALIDATION, "authority purpose must be non-empty"
            )
        return CallerV3(
            principal_id=self.principal_id,
            session_id=self.session_id,
            host_id=self.host_id,
            epoch=self.epoch,
        )


# ----------------------------------------------------------------------
# plan — one store.read() snapshot
# ----------------------------------------------------------------------


def _all_scope_ids(conn: sqlite3.Connection) -> list[str]:
    return sorted(
        r[0] for r in conn.execute("SELECT scope_id FROM scopes").fetchall()
    )


def _plan_scope(
    conn: sqlite3.Connection,
    store: Any,
    kernel: Kernel,
    authority: ProjectionAuthority,
    scope_id: str,
    now: int,
) -> tuple[Optional[_md.ScopePlan], str]:
    """Collect a scope's renderable content under its lease.

    Returns ``(plan, mode)`` where mode is ``quote``/``read``/``denied``.
    All withholding logic runs inside the caller's snapshot ``conn``;
    ``PurgesRepo``/repo helpers join the same snapshot via ``store.read``
    nesting, so the verdict set is snapshot-consistent.
    """
    caller = authority.caller()
    lease = kernel.resolve_access(
        conn, caller, Verb.QUOTE, authority.purpose, [scope_id],
        now_us=now, max_age_us=_LEASE_MAX_AGE_US,
    )
    quote = not lease.denied
    if not quote:
        lease = kernel.resolve_access(
            conn, caller, Verb.READ, authority.purpose, [scope_id],
            now_us=now, max_age_us=_LEASE_MAX_AGE_US,
        )
        if lease.denied:
            return None, "denied"

    wh = _wh.collect_withholding(conn, store, scope_id)
    entries: list[_md.ClaimEntry] = []
    withheld = 0
    inactive = 0
    input_items: list[str] = [f"quote={1 if quote else 0}"]

    for row in wh["claims"]:
        (
            claim_id, revision, state, object_json,
            recorded_from, recorded_until, subject_id, predicate,
        ) = row
        if state not in _md.RENDERABLE_STATES:
            inactive += 1
            continue
        if _wh.claim_withheld(claim_id, wh):
            withheld += 1
            continue
        ev_entries: list[_md.EvidenceEntry] = []
        claim_blocked = False
        for (_cid, _rev, span_id, role, src, srev, st, en) in wh["evidence"]:
            if _cid != claim_id or int(_rev) != int(revision):
                continue
            quote_text: Optional[str] = None
            quote_sha: Optional[str] = None
            if quote and role == "primary":
                # The only byte path: kernel-verified slice under the
                # quote lease. A denied/held span withholds the whole
                # claim (atomic omission — a claim without its evidence
                # is a misleading remainder).
                try:
                    slices = kernel.read_verified(
                        conn,
                        lease,
                        [
                            EvidenceLocator(
                                object_id=span_id,
                                revision=int(srev),
                                start_byte=int(st),
                                end_byte=int(en),
                            )
                        ],
                        now_us=now,
                    )
                except VerbatimError as exc:
                    if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                        claim_blocked = True
                        break
                    raise
                data = slices[0].data
                try:
                    quote_text = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise VerbatimError(
                        ErrorCode.STORE_CORRUPT,
                        f"span {span_id} excerpt fails to decode",
                    ) from exc
                quote_sha = hashlib.sha256(data).hexdigest()
            ev_entries.append(
                _md.EvidenceEntry(
                    span_id=span_id,
                    source_id=src,
                    source_revision=int(srev),
                    role=role,
                    start_byte=int(st),
                    end_byte=int(en),
                    quote=quote_text,
                    quote_sha256=quote_sha,
                )
            )
        if claim_blocked:
            withheld += 1
            continue
        entries.append(
            _md.ClaimEntry(
                claim_id=claim_id,
                revision=int(revision),
                state=state,
                recorded_from=int(recorded_from),
                recorded_until=(
                    int(recorded_until) if recorded_until is not None else None
                ),
                subject_id=subject_id,
                predicate=predicate,
                obj=safe_json_loads(object_json) if object_json else None,
                evidence=tuple(ev_entries),
            )
        )
        input_items.append(f"claim:{claim_id}@{revision}:{state}")

    # Source index: live sources only, revisions verified the same way.
    src_meta: dict[str, dict[str, Any]] = {}
    rev_rows: dict[str, list[dict[str, Any]]] = {}
    for (sid0, origin, _ext, kind, speaker, created) in wh["sources"]:
        if sid0 in wh["supp_sources"]:
            continue
        src_meta[sid0] = {
            "source_id": sid0,
            "origin": origin,
            "source_kind": kind,
            "speaker_id": speaker,
            "created_us": int(created),
        }
    for (src, rev, event_us, provenance, has_payload, nbytes) in wh[
        "rev_meta"
    ]:
        if src not in src_meta:
            continue
        if f"{src}:{int(rev)}" in wh["supp_rev_keys"]:
            withheld += 1
            continue
        if _wh.source_revision_withheld(
            conn, src, int(rev), held_refs=wh["held_refs"]
        ):
            withheld += 1
            continue
        if not has_payload or int(nbytes) <= 0:
            withheld += 1
            continue  # purged/empty revision — absent, not evidence
        rmeta: dict[str, Any] = {
            "revision": int(rev),
            "event_us": int(event_us),
            "provenance": provenance,
            "byte_length": int(nbytes),
        }
        if quote:
            # The payload digest comes from the kernel's verified byte
            # path — the hmac is checked inside read_verified; the
            # sha256 here binds the file's provenance line.
            try:
                slices = kernel.read_verified(
                    conn,
                    lease,
                    [
                        EvidenceLocator(
                            object_id=src,
                            revision=int(rev),
                            start_byte=0,
                            end_byte=int(nbytes),
                        )
                    ],
                    now_us=now,
                )
            except VerbatimError as exc:
                if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED:
                    withheld += 1
                    continue
                raise
            rmeta["payload_sha256"] = hashlib.sha256(slices[0].data).hexdigest()
        rev_rows.setdefault(src, []).append(rmeta)
        input_items.append(f"source:{src}@{int(rev)}")

    sources: list[dict[str, Any]] = []
    for sid0 in sorted(src_meta):
        revs = rev_rows.get(sid0, [])
        if not revs:
            withheld += 1
            continue  # every revision withheld/absent — omit the source
        rec = dict(src_meta[sid0])
        rec["revisions"] = revs
        sources.append(rec)

    input_items.sort()
    inputs_sha = hashlib.sha256(
        json_dumps(input_items).encode("utf-8")
    ).hexdigest()
    return (
        _md.ScopePlan(
            scope_id=scope_id,
            quote=quote,
            claims=tuple(entries),
            withheld=withheld,
            inactive=inactive,
            sources=tuple(sources),
            inputs_sha256=inputs_sha,
        ),
        "quote" if quote else "read",
    )


def _plan(
    store: Any,
    authority: ProjectionAuthority,
    scope_ids: Optional[Iterable[str]],
    now: int,
    carry_scopes: Iterable[str] = (),
) -> dict[str, Any]:
    """Snapshot: collect per-scope plans + the input manifest digest.

    ``carry_scopes`` are scopes covered by the live generation but not in
    ``scope_ids`` — they are planned too (under the same authority) so the
    build can carry a byte-identical file or re-render it when the scope
    moved; they are never silently dropped (V4-43.04).
    """
    kernel = Kernel(store)
    with store.read() as conn:
        if scope_ids is None:
            wanted = _all_scope_ids(conn)
        else:
            wanted = sorted({require_id(s, "scope_id") for s in scope_ids})
        wanted = sorted(set(wanted) | set(carry_scopes))
        generation = int(
            store._meta_get(conn, "projection_generation") or 0
        )
        snapshot_seq = int(
            conn.execute(
                "SELECT COALESCE(MAX(event_seq), 0) FROM events"
            ).fetchone()[0]
        )
        plans: dict[str, _md.ScopePlan] = {}
        denied: list[str] = []
        for sid in wanted:
            plan, _mode = _plan_scope(
                conn, store, kernel, authority, sid, now
            )
            if plan is None:
                denied.append(sid)
            else:
                plans[sid] = plan
        store_id = store.db_id()
    inputs_sha = hashlib.sha256(
        json_dumps(
            {
                "db": store_id,
                "generation": generation,
                "builder": f"{_md.RENDERER_NAME}/{_md.RENDERER_VERSION}",
                "purpose": authority.purpose,
                "scopes": {
                    sid: p.inputs_sha256 for sid, p in sorted(plans.items())
                },
                "denied": sorted(denied),
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "generation": generation,
        "snapshot_seq": snapshot_seq,
        "store_id": store_id,
        "plans": plans,
        "denied": denied,
        "inputs_sha256": inputs_sha,
    }


# ----------------------------------------------------------------------
# checkpoint (resume) plumbing
# ----------------------------------------------------------------------


def _checkpoint_load(staging: str) -> Optional[dict[str, Any]]:
    path = os.path.join(staging, _paths.CHECKPOINT_FILE)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    obj = safe_json_loads(raw.decode("utf-8", "replace"))
    return obj if isinstance(obj, dict) else None


def _checkpoint_write(staging: str, state: dict[str, Any]) -> None:
    tmp = os.path.join(staging, _paths.CHECKPOINT_FILE + ".tmp")
    _paths.write_file(tmp, (json_dumps(state) + "\n").encode("utf-8"))
    os.replace(tmp, os.path.join(staging, _paths.CHECKPOINT_FILE))


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------------
# publish helpers
# ----------------------------------------------------------------------


def _current_manifest(root: str) -> Optional[dict[str, Any]]:
    target = _paths.current_target(root)
    if target is None:
        return None
    path = os.path.join(target, _paths.MANIFEST_FILE)
    try:
        with open(path, "rb") as fh:
            obj = safe_json_loads(fh.read().decode("utf-8", "replace"))
    except OSError:
        return None
    return obj if isinstance(obj, dict) else None


def _flip_current(root: str, gen_name: str) -> None:
    """Atomically point ``current`` at ``gen_name`` (relative symlink)."""
    link = os.path.join(root, _paths.CURRENT_LINK)
    tmp = os.path.join(
        root, f".{_paths.CURRENT_LINK}-{os.getpid()}-{new_id()[:8]}"
    )
    if os.path.exists(link) and not os.path.islink(link):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "current exists and is not a symlink — refusing to overwrite",
        )
    try:
        os.symlink(gen_name, tmp)
        os.replace(tmp, link)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise VerbatimError(
            ErrorCode.STORE_WRITE_FAILED,
            f"cannot publish projection generation: {exc}",
        ) from exc


def _next_gen_seq(root: str) -> int:
    gens = _paths.list_generations(root)
    return (gens[-1][0] + 1) if gens else 1


def _build_row_manifest(
    plan: dict[str, Any],
    scope_files: dict[str, str],
    carried: list[str],
) -> dict[str, Any]:
    """The V4-43.01 record: watermark, producer identity, coverage."""
    return {
        "builder": f"{_md.RENDERER_NAME}/{_md.RENDERER_VERSION}",
        "format": _md.FORMAT_TAG,
        "format_version": _md.FORMAT_VERSION,
        "store_id": plan["store_id"],
        "projection_generation": plan["generation"],
        "snapshot_seq": plan["snapshot_seq"],
        "inputs_sha256": plan["inputs_sha256"],
        "scopes": sorted(scope_files),
        "carried_scopes": sorted(carried),
        "denied_scopes": len(plan["denied"]),
        "claims": sum(len(p.claims) for p in plan["plans"].values()),
        "withheld": sum(p.withheld for p in plan["plans"].values()),
        "inactive": sum(p.inactive for p in plan["plans"].values()),
    }


def _record_build(
    store: Any,
    build_id: str,
    manifest: dict[str, Any],
    scope_partition: str,
    snapshot_seq: int,
) -> bool:
    """Create the ``projection_builds`` row; False when the table is absent."""
    with store.tx() as conn:
        if not _has_table(conn, "projection_builds"):
            return False
        ProjectionRepo(store).build_create(
            conn,
            PROJECTION_KIND,
            scope_partition,
            manifest=manifest,
            snapshot_seq=snapshot_seq,
            build_id=build_id,
        )
    return True


def _set_build_state(
    store: Any,
    build_id: str,
    status: str,
    *,
    caught_up_seq: Optional[int] = None,
    validation_digest: Optional[bytes] = None,
) -> None:
    with store.tx() as conn:
        if not _has_table(conn, "projection_builds"):
            return
        ProjectionRepo(store).build_set_state(
            conn,
            build_id,
            status,
            caught_up_seq=caught_up_seq,
            validation_digest=validation_digest,
        )


def _publish_rows(
    store: Any,
    build_id: str,
    scope_ids: list[str],
    snapshot_seq: int,
    manifest_sha256: str,
    actor: str,
) -> None:
    """CAS-publish each scope's markdown pointer + append the event."""
    with store.tx() as conn:
        if not _has_table(conn, "projection_builds"):
            return
        repo = ProjectionRepo(store)
        repo.build_set_state(
            conn, build_id, "published", caught_up_seq=snapshot_seq
        )
        for sid in scope_ids:
            cur = repo.active_get(sid, PROJECTION_KIND)
            repo.active_publish(
                conn,
                sid,
                PROJECTION_KIND,
                build_id,
                snapshot_seq,
                expected_cas=cur["cas_revision"] if cur else -1,
            )
            EventsRepo(store).append(
                conn,
                sid,
                "projection_published",
                actor,
                {
                    "build_id": build_id,
                    "projection_kind": PROJECTION_KIND,
                    "manifest_sha256": manifest_sha256,
                    "snapshot_seq": snapshot_seq,
                },
                _EVENT_POLICY,
            )


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------


def build(
    store: Any,
    root: str,
    *,
    authority: ProjectionAuthority,
    scope_ids: Optional[Iterable[str]] = None,
    resume: bool = True,
    now_us: Optional[int] = None,
    on_file: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Build (or resume) a Markdown projection generation under ``root``.

    ``on_file`` is a deterministic progress hook invoked after each
    staged file commits — the test seam for crash/resume.

    Returns a result dict: build id, published generation dir, per-file
    digests, scope coverage, withheld/denied counts, resume/discard
    detail. Errors propagate as typed ``VerbatimError``; a build that
    dies mid-stage leaves ``.staging-<id>`` + checkpoint for ``resume``.
    """
    if not isinstance(authority, ProjectionAuthority):
        raise VerbatimError(
            ErrorCode.VALIDATION, "authority must be a ProjectionAuthority"
        )
    now = int(now_us) if now_us is not None else _now_us()
    root_path = _paths.prepare_root(root)

    # Coverage = requested scopes ∪ scopes the live generation already
    # covers — a scope-partition rebuild re-evaluates siblings rather
    # than stranding them behind the new generation (V4-43.04).
    prev = _current_manifest(root_path)
    prev_target = _paths.current_target(root_path)
    prev_scope_rows = {
        s["scope_id"]: s
        for s in (prev or {}).get("scopes") or []
        if isinstance(s, dict) and isinstance(s.get("scope_id"), str)
    }
    requested = (
        None
        if scope_ids is None
        else {require_id(s, "scope_id") for s in scope_ids}
    )
    carry_candidates = (
        set(prev_scope_rows) - set(requested) if requested else set()
    )
    plan = _plan(
        store, authority, requested, now, carry_scopes=carry_candidates
    )

    # ---- resume bookkeeping ------------------------------------------
    staging: Optional[str] = None
    build_id: Optional[str] = None
    done_files: dict[str, dict[str, Any]] = {}
    discarded: list[str] = []
    resumed = False
    builder_id = f"{_md.RENDERER_NAME}/{_md.RENDERER_VERSION}"
    for cand in _paths.list_staging(root_path):
        cp = _checkpoint_load(cand) if resume else None
        compatible = (
            cp is not None
            and cp.get("db_id") == plan["store_id"]
            and cp.get("builder") == builder_id
            and cp.get("generation") == plan["generation"]
            and cp.get("inputs_sha256") == plan["inputs_sha256"]
            and isinstance(cp.get("build_id"), str)
        )
        if compatible and staging is None:
            # The recorded build row must still be in-flight — a
            # 'published' row means this staging dir is orphan residue.
            row = (
                ProjectionRepo(store).build_get(cp["build_id"])
                if _table(store, "projection_builds")
                else None
            )
            if row is not None and row["status"] in ("published", "retired"):
                compatible = False
        if compatible:
            staging = cand
            build_id = cp["build_id"]
            done_files = dict(cp.get("files") or {})
            resumed = True
        else:
            _paths.remove_tree(cand)
            discarded.append(os.path.basename(cand))
            if cp is not None and isinstance(cp.get("build_id"), str):
                _mark_failed_quiet(store, cp["build_id"])

    # ---- render the file set ------------------------------------------
    # A carry-candidate scope ships the prior file verbatim ONLY when its
    # fresh eligibility digest matches the recorded one — identical input
    # renders identical bytes, so carrying is provably equivalent to
    # re-rendering (and a suppressed/moved sibling can never ride through
    # on stale bytes, V4-43.09). Any mismatch re-renders from the live plan.
    renders: list[tuple[str, bytes]] = []  # (relpath, bytes) in write order
    scope_files: dict[str, str] = {}
    carried: list[str] = []
    prev_files = (prev or {}).get("files") or {}
    for sid, splan in sorted(plan["plans"].items()):
        relpath = _md.scope_filename(sid)
        _paths.safe_relpath(relpath)
        prior = prev_scope_rows.get(sid)
        if (
            sid in carry_candidates
            and prior is not None
            and prior.get("inputs_sha256") == splan.inputs_sha256
            and prior.get("file") == relpath
            and prev_target is not None
            and isinstance(prev_files.get(relpath), dict)
        ):
            src_path = os.path.join(prev_target, relpath)
            actual = _sha256_file(src_path)
            if actual == prev_files[relpath].get("sha256"):
                with open(src_path, "rb") as fh:
                    renders.append((relpath, fh.read()))
                scope_files[sid] = relpath
                carried.append(sid)
                continue
        data = _md.render_scope_document(
            splan,
            store_id=plan["store_id"],
            generation=plan["generation"],
        )
        renders.append((relpath, data))
        scope_files[sid] = relpath

    data_by_path = {rp: d for rp, d in renders}
    scope_rows = []
    for sid in sorted(scope_files):
        built = plan["plans"].get(sid)
        prior = prev_scope_rows.get(sid, {})
        scope_rows.append(
            {
                "scope_id": sid,
                "file": scope_files[sid],
                "claims": (
                    len(built.claims)
                    if built is not None
                    else int(prior.get("claims") or 0)
                ),
                "withheld": (
                    built.withheld
                    if built is not None
                    else int(prior.get("withheld") or 0)
                ),
                "quote": (
                    bool(built.quote)
                    if built is not None
                    else bool(prior.get("quote"))
                ),
                "inputs_sha256": (
                    built.inputs_sha256
                    if built is not None
                    else prior.get("inputs_sha256")
                ),
                "sha256": hashlib.sha256(
                    data_by_path[scope_files[sid]]
                ).hexdigest(),
                "bytes": len(data_by_path[scope_files[sid]]),
            }
        )

    index_data = _md.render_index(
        scope_rows,
        store_id=plan["store_id"],
        generation=plan["generation"],
        snapshot_seq=plan["snapshot_seq"],
    )
    renders.append((_paths.INDEX_FILE, index_data))

    files_meta = [
        {
            "path": rp,
            "sha256": hashlib.sha256(d).hexdigest(),
            "bytes": len(d),
        }
        for rp, d in renders
    ]
    manifest = {
        "format": _md.FORMAT_TAG,
        "format_version": _md.FORMAT_VERSION,
        "kind": "projection-manifest",
        "store_id": plan["store_id"],
        "projection_generation": plan["generation"],
        "snapshot_seq": plan["snapshot_seq"],
        "builder": {"name": _md.RENDERER_NAME, "version": _md.RENDERER_VERSION},
        "purpose": authority.purpose,
        "inputs_sha256": plan["inputs_sha256"],
        "scopes": scope_rows,
        "built_scopes": sorted(set(scope_files) - set(carried)),
        "carried_scopes": sorted(carried),
        "denied_scopes": len(plan["denied"]),
        "files": {f["path"]: f for f in files_meta},
        "counts": {
            "files": len(files_meta) + 1,  # + manifest.json itself
            "claims": sum(len(p.claims) for p in plan["plans"].values()),
            "withheld": sum(p.withheld for p in plan["plans"].values()),
            "inactive": sum(p.inactive for p in plan["plans"].values()),
            "scopes": len(scope_rows),
        },
    }
    manifest["content_sha256"] = hashlib.sha256(
        json_dumps(files_meta).encode("utf-8")
    ).hexdigest()
    manifest_data = (json_dumps(manifest) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()

    if len(files_meta) + 1 > _MAX_FILES:
        raise VerbatimError(
            ErrorCode.EVIDENCE_TOO_LARGE, "projection exceeds file bound"
        )
    for f in files_meta:
        if f["bytes"] > _MAX_FILE_BYTES:
            raise VerbatimError(
                ErrorCode.EVIDENCE_TOO_LARGE,
                f"projection file {f['path']} exceeds size bound",
            )

    # ---- record + stage ------------------------------------------------
    # ``build_id`` always names the staging dir + checkpoint; ``recorded``
    # tracks whether a ``projection_builds`` row exists (absent on stores
    # without the v2 projection tables — the fs generation still works).
    recorded = False
    if build_id is None:
        build_id = new_id()
        manifest_row = _build_row_manifest(plan, scope_files, carried)
        recorded = _record_build(
            store,
            build_id,
            manifest_row,
            ",".join(sorted(scope_files)) or "*",
            plan["snapshot_seq"],
        )
    else:
        recorded = (
            _table(store, "projection_builds")
            and ProjectionRepo(store).build_get(build_id) is not None
        )

    if staging is None:
        staging = os.path.join(
            root_path, f"{_paths.STAGING_PREFIX}{build_id}"
        )
        os.makedirs(staging, mode=0o700, exist_ok=False)

    checkpoint = {
        "build_id": build_id,
        "db_id": plan["store_id"],
        "builder": builder_id,
        "generation": plan["generation"],
        "inputs_sha256": plan["inputs_sha256"],
        "files": done_files,
    }
    _checkpoint_write(staging, checkpoint)

    write_order = [(rp, d) for rp, d in renders]
    write_order.append((_paths.MANIFEST_FILE, manifest_data))
    expected_names = {rp for rp, _d in write_order} | {
        _paths.CHECKPOINT_FILE
    }
    # Stray staged files from an abandoned render set never publish.
    for name in os.listdir(staging):
        if name not in expected_names:
            _paths.remove_tree(os.path.join(staging, name))

    for relpath, data in write_order:
        sha = hashlib.sha256(data).hexdigest()
        prev_done = done_files.get(relpath)
        staged_path = os.path.join(staging, relpath)
        if (
            prev_done is not None
            and prev_done.get("sha256") == sha
            and _sha256_file(staged_path) == sha
        ):
            if on_file is not None:
                on_file(relpath)  # still report progress on resume
            continue  # staged bytes already verified — resume skips them
        _paths.write_file(staged_path, data)
        actual = _sha256_file(staged_path)
        if actual != sha:
            _mark_failed_quiet(store, build_id)
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED,
                f"staged file {relpath} failed verification",
            )
        done_files[relpath] = {"sha256": sha, "bytes": len(data)}
        checkpoint["files"] = done_files
        _checkpoint_write(staging, checkpoint)
        if on_file is not None:
            on_file(relpath)

    # ---- validate + publish --------------------------------------------
    if recorded:
        _set_build_state(
            store,
            build_id,
            "ready",
            validation_digest=bytes.fromhex(manifest["content_sha256"]),
        )

    gen_name = _paths.generation_name(_next_gen_seq(root_path))
    gen_dir = os.path.join(root_path, gen_name)
    os.remove(os.path.join(staging, _paths.CHECKPOINT_FILE))
    os.rename(staging, gen_dir)
    _flip_current(root_path, gen_name)

    if recorded:
        _publish_rows(
            store, build_id, sorted(scope_files), plan["snapshot_seq"],
            manifest_sha, authority.principal_id,
        )

    return {
        "build_id": build_id,
        "recorded": recorded,
        "root": root_path,
        "generation": plan["generation"],
        "snapshot_seq": plan["snapshot_seq"],
        "published": gen_name,
        "current": os.path.join(root_path, _paths.CURRENT_LINK),
        "files": files_meta + [
            {
                "path": _paths.MANIFEST_FILE,
                "sha256": manifest_sha,
                "bytes": len(manifest_data),
            }
        ],
        "scopes": scope_rows,
        "built_scopes": sorted(set(scope_files) - set(carried)),
        "carried_scopes": sorted(carried),
        "denied_scopes": len(plan["denied"]),
        "withheld": sum(p.withheld for p in plan["plans"].values()),
        "inactive": sum(p.inactive for p in plan["plans"].values()),
        "inputs_sha256": plan["inputs_sha256"],
        "manifest_sha256": manifest_sha,
        "resumed": resumed,
        "discarded": discarded,
    }


def _table(store: Any, name: str) -> bool:
    try:
        with store.read() as conn:
            return _has_table(conn, name)
    except Exception:
        return False


def _mark_failed_quiet(store: Any, build_id: Optional[str]) -> None:
    if not build_id:
        return
    try:
        _set_build_state(store, build_id, "failed")
    except Exception:
        pass  # best-effort bookkeeping; the staging residue remains


# ----------------------------------------------------------------------
# status / read / reclaim
# ----------------------------------------------------------------------


def suppress(
    store: Any,
    root: str,
    targets: Iterable[tuple[str, str]],
    *,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Immediate deletion suppression on rendered bytes (V4-43.09).

    Called by the projection owner alongside ``purge.suppress``/closure:
    the canonical tombstones already gate the store; this call removes
    the *projected copies* without waiting for a rebuild.

    * ``.staging-*`` residue: every already-written file containing an
      attributable block is physically deleted and un-marked in the
      checkpoint, so a resume re-renders the scope under current policy.
    * every published ``gen-*`` (current and retained): attributable
      claim blocks are excised in place and replaced by withheld
      markers; ``index.md`` and ``manifest.json`` are rewritten so all
      digests stay internally consistent, and the manifest records a
      ``suppression`` event. Ambiguous files collapse to a whole-scope
      tombstone rather than leaking a partial excerpt.

    ``targets`` are ``(kind, object_id)`` pairs — ``claim``, ``span`` or
    ``source``; a claim citing a suppressed span/source is attributable
    and excised too. Other kinds cannot appear in a file projection and
    are reported under ``not_applicable`` (the canonical suppression in
    the store is unaffected either way). ``canon_unsuppressed`` honestly
    reports excised claims that carry no canonical tombstone yet —
    excision is always safe (rebuild restores), but the caller should
    normally have run ``purge.suppress``/closure first.
    """
    claim_ids: set[str] = set()
    span_ids: set[str] = set()
    source_ids: set[str] = set()
    not_applicable: list[str] = []
    for t in targets:
        kind, oid = t
        if kind == "claim":
            claim_ids.add(require_id(str(oid), "claim_id"))
        elif kind == "span":
            span_ids.add(require_id(str(oid), "span_id"))
        elif kind == "source":
            source_ids.add(require_id(str(oid), "source_id"))
        else:
            not_applicable.append(f"{kind}:{oid}")
    if not (claim_ids or span_ids or source_ids):
        return {
            "suppressed_claims": [],
            "generations": [],
            "staging_files": [],
            "not_applicable": sorted(not_applicable),
            "canon_unsuppressed": [],
        }
    now = int(now_us) if now_us is not None else _now_us()
    root_path = os.path.abspath(root)
    staged_removed: list[str] = []
    if os.path.isdir(root_path):
        for cand in _paths.list_staging(root_path):
            cp = _checkpoint_load(cand)
            dirty_cp = False
            for name in sorted(os.listdir(cand)):
                if not name.endswith(".md"):
                    continue
                fpath = os.path.join(cand, name)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if not _file_attributable(
                    text, claim_ids, span_ids, source_ids
                ):
                    continue
                os.remove(fpath)  # physically remove attributable bytes
                staged_removed.append(
                    f"{os.path.basename(cand)}/{name}"
                )
                if (
                    cp is not None
                    and isinstance(cp.get("files"), dict)
                    and name in cp["files"]
                ):
                    del cp["files"][name]
                    dirty_cp = True
            if dirty_cp:
                _checkpoint_write(cand, cp)
        changed_gens: list[str] = []
        removed_claims: list[str] = []
        for _seq, gen in _paths.list_generations(root_path):
            removed = _suppress_in_generation(
                gen, claim_ids, span_ids, source_ids, now
            )
            if removed:
                changed_gens.append(os.path.basename(gen))
                removed_claims.extend(removed)
    else:
        changed_gens = []
        removed_claims = []
    canon: Optional[set[str]] = None
    try:
        with store.read() as conn:
            if _has_table(conn, "purges") and _has_table(
                conn, "purge_targets"
            ):
                from ..storage.repos import PurgesRepo

                canon = PurgesRepo(store).suppressed_ids(
                    "claim", sorted(claim_ids)
                )
    except Exception:
        canon = None
    result = {
        "suppressed_claims": sorted(set(removed_claims)),
        "generations": changed_gens,
        "staging_files": staged_removed,
        "not_applicable": sorted(not_applicable),
    }
    if canon is not None:
        result["canon_unsuppressed"] = sorted(claim_ids - canon)
    else:
        result["canon_unsuppressed"] = "unknown"
    return result


def _file_attributable(
    text: str,
    claim_ids: set[str],
    span_ids: set[str],
    source_ids: set[str],
) -> bool:
    for cid, meta, _s, _e in _md.iter_claim_blocks(text):
        if cid is not None and cid in claim_ids:
            return True
        if meta is None:
            if span_ids or source_ids:
                return True  # cannot disprove attribution — fail closed
            continue
        for ev in meta.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            if ev.get("span_id") in span_ids or (
                ev.get("source_id") in source_ids
            ):
                return True
    return False


def _suppress_in_generation(
    gen_dir: str,
    claim_ids: set[str],
    span_ids: set[str],
    source_ids: set[str],
    now: int,
) -> list[str]:
    """Excise attributable blocks in one published generation."""
    mpath = os.path.join(gen_dir, _paths.MANIFEST_FILE)
    try:
        with open(mpath, encoding="utf-8") as fh:
            manifest = safe_json_loads(fh.read())
    except OSError:
        manifest = None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != _md.FORMAT_TAG
        or manifest.get("kind") != "projection-manifest"
    ):
        return []  # not ours — never mutate an unrecognized tree
    scope_rows = [
        r
        for r in (manifest.get("scopes") or [])
        if isinstance(r, dict) and isinstance(r.get("file"), str)
    ]
    row_by_file = {r["file"]: r for r in scope_rows}
    removed_all: list[str] = []
    for name in sorted(os.listdir(gen_dir)):
        if not name.endswith(".md") or name == _paths.INDEX_FILE:
            continue
        fpath = os.path.join(gen_dir, name)
        try:
            with open(fpath, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        new_text, removed = _md.excise_claims(
            text,
            claim_ids=frozenset(claim_ids),
            span_ids=frozenset(span_ids),
            source_ids=frozenset(source_ids),
        )
        if not removed or new_text is None:
            continue
        data = new_text.encode("utf-8")
        _paths.write_file(fpath, data)
        sha = hashlib.sha256(data).hexdigest()
        meta = _md.parse_header(new_text) or {}
        row = row_by_file.get(name)
        if row is not None:
            row["claims"] = int(meta.get("claims") or 0)
            row["withheld"] = int(meta.get("withheld") or 0)
            row["sha256"] = sha
            row["bytes"] = len(data)
        files = manifest.get("files")
        if isinstance(files, dict):
            files[name] = {"path": name, "sha256": sha, "bytes": len(data)}
        removed_all.extend(removed)
    if not removed_all:
        return []
    # index.md renders the scope table — regenerate it from the updated
    # rows so the shipped view and the manifest agree (V4-43.01/47.01).
    idx_rows = sorted(scope_rows, key=lambda r: str(r.get("scope_id")))
    index_data = _md.render_index(
        idx_rows,
        store_id=manifest.get("store_id"),
        generation=int(manifest.get("projection_generation") or 0),
        snapshot_seq=int(manifest.get("snapshot_seq") or 0),
    )
    _paths.write_file(os.path.join(gen_dir, _paths.INDEX_FILE), index_data)
    files = manifest.get("files")
    if isinstance(files, dict):
        files[_paths.INDEX_FILE] = {
            "path": _paths.INDEX_FILE,
            "sha256": hashlib.sha256(index_data).hexdigest(),
            "bytes": len(index_data),
        }
        counts = manifest.get("counts")
        if isinstance(counts, dict):
            counts["claims"] = sum(
                int(r.get("claims") or 0) for r in scope_rows
            )
            counts["withheld"] = sum(
                int(r.get("withheld") or 0) for r in scope_rows
            )
        events = manifest.setdefault("events", [])
        if isinstance(events, list):
            events.append(
                {
                    "type": "suppression",
                    "claims": sorted(set(removed_all)),
                    "recorded_us": now,
                }
            )
        manifest["content_sha256"] = hashlib.sha256(
            json_dumps(list(files.values())).encode("utf-8")
        ).hexdigest()
        _paths.write_file(
            mpath, (json_dumps(manifest) + "\n").encode("utf-8")
        )
    return removed_all


def status(root: str) -> dict[str, Any]:
    """Observable projection-root state (V4-43.01 reader surface)."""
    root_path = os.path.abspath(root)
    gens = _paths.list_generations(root_path)
    current = _paths.current_target(root_path)
    manifest = _current_manifest(root_path)
    return {
        "root": root_path,
        "current": os.path.basename(current) if current else None,
        "generations": [os.path.basename(p) for _s, p in gens],
        "staging": [
            os.path.basename(p) for p in _paths.list_staging(root_path)
        ],
        "manifest": manifest,
    }


def read_current(root: str, relpath: str) -> Optional[bytes]:
    """Read one file from the live generation; ``None`` when absent.

    Path safety is enforced here too — a reader cannot escape through
    ``..`` or a planted symlink (V4-47.06).
    """
    root_path = os.path.abspath(root)
    target = _paths.current_target(root_path)
    if target is None:
        return None
    _paths.safe_relpath(relpath)
    path = _paths.check_inside(root_path, os.path.join(target, relpath))
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def reclaim(
    store: Any,
    root: str,
    *,
    keep: int = 1,
) -> dict[str, Any]:
    """Reclaim old generations (V4-43.08).

    Removes published ``gen-*`` dirs except the ``keep`` newest and the
    one ``current`` points at — no deletion under an active reader
    pointer. Staging residue is cleaned separately by ``build`` resume
    validation or ``prune_staging``. Retired generations are marked
    ``retired`` in ``projection_builds`` when the row is identifiable.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
        raise VerbatimError(ErrorCode.VALIDATION, "keep must be an int >= 0")
    root_path = os.path.abspath(root)
    current = _paths.current_target(root_path)
    gens = _paths.list_generations(root_path)
    survivors = {p for _s, p in gens[len(gens) - keep:]} if keep else set()
    if current is not None:
        survivors.add(current)
    removed: list[str] = []
    for _seq, path in gens:
        if path in survivors:
            continue
        manifest_path = os.path.join(path, _paths.MANIFEST_FILE)
        manifest = None
        try:
            with open(manifest_path, "rb") as fh:
                manifest = safe_json_loads(
                    fh.read().decode("utf-8", "replace")
                )
        except OSError:
            manifest = None
        _paths.remove_tree(path)
        removed.append(os.path.basename(path))
        _retire_matching_builds(store, manifest)
    return {"removed": removed, "kept": sorted(os.path.basename(p) for p in survivors)}


def _retire_matching_builds(store: Any, manifest: Any) -> None:
    """Best-effort: mark published builds for a removed generation."""
    if not isinstance(manifest, dict):
        return
    try:
        with store.tx() as conn:
            if not _has_table(conn, "projection_builds"):
                return
            rows = conn.execute(
                "SELECT build_id, manifest_json FROM projection_builds"
                " WHERE kind = ? AND status = 'published'",
                (PROJECTION_KIND,),
            ).fetchall()
            for build_id, raw in rows:
                rec = safe_json_loads(raw or "{}")
                if (
                    isinstance(rec, dict)
                    and rec.get("inputs_sha256") == manifest.get("inputs_sha256")
                ):
                    ProjectionRepo(store).build_set_state(
                        conn, build_id, "retired"
                    )
    except Exception:
        pass  # bookkeeping is best-effort; the dir is already gone


def prune_staging(store: Any, root: str) -> dict[str, Any]:
    """Discard every staging residue dir; their build rows go 'failed'."""
    root_path = os.path.abspath(root)
    removed: list[str] = []
    for cand in _paths.list_staging(root_path):
        cp = _checkpoint_load(cand)
        _paths.remove_tree(cand)
        removed.append(os.path.basename(cand))
        if cp is not None and isinstance(cp.get("build_id"), str):
            _mark_failed_quiet(store, cp["build_id"])
    return {"discarded": removed}


__all__ = [
    "PROJECTION_KIND",
    "ProjectionAuthority",
    "build",
    "prune_staging",
    "read_current",
    "reclaim",
    "status",
    "suppress",
]
