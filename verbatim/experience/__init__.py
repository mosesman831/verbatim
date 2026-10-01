"""Experience memory: episodes, procedures, and prospective records.

The v2 differentiator (SPEC_V2 §21-24): typed task experience that other
memory systems lack — episodes grouping evidence, evidence-backed
procedures with environment matching and outcome drift, and prospective
records for plans/commitments that never masquerade as facts.

Importing this package is side-effect free: no stores opened, no threads,
no network (V2-06.07). All mutations take the caller's transaction
``conn``; reads open ``store.read()`` snapshots.
"""

from .episodes import (
    attach,
    close_episode,
    detach,
    episode_summary,
    episodes_for_scope,
    members,
    open_episode,
)
from .procedures import (
    activate_procedure,
    drift_status,
    environment_fingerprint,
    environment_match,
    match_procedure,
    outcome_stats,
    procedure_view,
    procedures_for_scope,
    propose_procedure,
    record_outcome,
    set_procedure_state,
)
from .prospective import (
    cancel,
    complete,
    due,
    plan,
    plan_view,
    plans_for_scope,
    recurrence_next,
    reschedule,
    start,
)
from .scenes import (
    SCENE_KIND,
    SCENE_PRODUCER_ID,
    assign_episode,
    build_scenes,
    episode_sequence,
    episodes_for_family,
    family_key_for,
    remove_member,
    scene_for_episode,
    scene_id_for,
    scene_members,
    scene_row,
    scenes_for_scope,
    trajectory_of_episode,
)
from .hierarchy import (
    HIERARCHY_PRODUCER_ID,
    build_overview,
    build_overviews,
    expand,
    overview_for,
    overview_id_for,
)

__all__ = [
    # episodes (§21)
    "open_episode",
    "attach",
    "detach",
    "close_episode",
    "members",
    "episode_summary",
    "episodes_for_scope",
    # procedures (§23-24)
    "propose_procedure",
    "activate_procedure",
    "set_procedure_state",
    "record_outcome",
    "environment_fingerprint",
    "environment_match",
    "outcome_stats",
    "drift_status",
    "match_procedure",
    "procedure_view",
    "procedures_for_scope",
    # prospective (§22)
    "plan",
    "due",
    "complete",
    "cancel",
    "start",
    "reschedule",
    "recurrence_next",
    "plan_view",
    "plans_for_scope",
    # scenes + hierarchy (SPEC_V4 §22)
    "SCENE_KIND",
    "SCENE_PRODUCER_ID",
    "HIERARCHY_PRODUCER_ID",
    "scene_id_for",
    "assign_episode",
    "build_scenes",
    "scene_row",
    "scene_members",
    "scene_for_episode",
    "scenes_for_scope",
    "episodes_for_family",
    "episode_sequence",
    "family_key_for",
    "trajectory_of_episode",
    "remove_member",
    "overview_id_for",
    "build_overview",
    "build_overviews",
    "overview_for",
    "expand",
]
