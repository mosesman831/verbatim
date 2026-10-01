"""File projections, portable bundles, and handoff (SPEC_V4 §43, §47).

SQLite remains the canonical authority for policy and evidence; this
package renders *derived, reviewable* Markdown views into generation-
scoped directories, packs them into deterministic tar bundles, verifies
and classifies bundles on a second store, mints bounded read-only
handoff capsules, and turns file edits into review proposals.

Submodules:

* ``builder``   — generation-scoped staging/atomic-publish/resume engine
  (V4-43.01–09) that renders through ``Kernel.read_verified`` and the
  export-grade suppression/quarantine cascade.
* ``markdown``  — pure deterministic renderer (V4-47.01).
* ``bundle``    — directory→tar portable bundles with manifest digest
  verification, honest foreign-row labeling, and deletion-policy checks
  on import (V4-47.03/04).
* ``handoff``   — bound read-only capsules + honest offline presentation
  (V4-47.07/08).
* ``edits``     — file edits → review proposals with conflict detection
  (V4-47.05); generated text never overwrites original evidence.
* ``paths``     — projection-root filesystem safety (V4-47.06).
* ``_withhold`` — the shared suppression/quarantine cascade.

Nothing here touches the network, fetches remotely, or executes hooks.
"""

from .builder import (
    PROJECTION_KIND,
    ProjectionAuthority,
    build,
    prune_staging,
    read_current,
    reclaim,
    status,
    suppress,
)
from .bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    export_bundle,
    import_bundle,
    verify_bundle,
)
from .edits import audit_file, propose_edit
from .handoff import (
    CAPSULE_VERBS,
    DEFAULT_OFFLINE_LEASE_US,
    PROHIBITED_VERBS,
    create_capsule,
    open_capsule,
    present_offline,
)
from .markdown import (
    FORMAT_TAG,
    FORMAT_VERSION,
    parse_header,
    render_index,
    render_scope_document,
    scope_filename,
)
from .paths import (
    check_inside,
    current_target,
    list_generations,
    prepare_root,
    safe_relpath,
)

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "CAPSULE_VERBS",
    "DEFAULT_OFFLINE_LEASE_US",
    "FORMAT_TAG",
    "FORMAT_VERSION",
    "PROHIBITED_VERBS",
    "PROJECTION_KIND",
    "ProjectionAuthority",
    "audit_file",
    "build",
    "check_inside",
    "create_capsule",
    "current_target",
    "export_bundle",
    "import_bundle",
    "list_generations",
    "open_capsule",
    "parse_header",
    "prepare_root",
    "present_offline",
    "propose_edit",
    "prune_staging",
    "read_current",
    "reclaim",
    "render_index",
    "render_scope_document",
    "safe_relpath",
    "scope_filename",
    "status",
    "suppress",
    "verify_bundle",
]
