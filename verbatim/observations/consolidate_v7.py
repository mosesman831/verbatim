"""T1 belief consolidation for the V7 unit plane (SPEC_V7 §14,
V7-14.01–14.08).

``consolidate_scope_v7`` is a pure in-transaction maintenance-lane pass:
it scans the scope's unconsolidated ``units`` (durable high-water cursor
in ``consolidation_cursor_v7`` plus a per-unit seen table so a reused
SQLite rowid can never strand a unit), re-derives slot assertions from
the §30 fact tables (``state_facts`` / ``preferences`` / ``events_v7`` /
``entity_mentions``), and emits grounded ``observations_v7`` rows where
≥ 2 distinct units from ≥ 2 distinct *evidence families* agree on one
(subject, slot, value). It then materializes ``profiles_v7`` slots and
marks ``standing_queries`` rows dirty when the scope's changed units are
not provably excluded by the stored filters.

Evidence families (documented interpretation of V7-14.01's "distinct
evidence families", V6-03.11/12 carried): the fact-kind table an
assertion came from — ``state`` (state_facts), ``preference``
(preferences), ``event`` (events_v7), ``mention`` (entity_mentions
co-mention of subject+value). Two rows of one kind are ONE family: two
state_facts alone never consolidate — the spec's corroboration contract
demands independent evidence kinds. Each unit contributes ≤ 1 support
ref regardless of how many kinds it carries; ``proof_count`` is the
number of distinct supporting units.

Determinism (V7-14.02): ``obs_id`` = ``"obs7:" + sha256(canonical_json
{version, scope, slot, polarity, value_norm, generation})[:24]`` — the
identity of a *belief*, stable across support growth; support refs,
proof count, first/last times, and the stale flag are mutable columns
updated in place. Text is a fixed template per slot family —
``"{subject} {phrase} {value}"`` (negated/hypothetical variants use the
family's base phrase) — never free-form. Re-running a pass over the
same inputs writes zero rows and yields byte-identical text/refs.

Freshness (V7-14.03): inside each pass, a slot assertion that is not in
an observation's support and is *newer* than ``obs.last_us`` marks the
observation ``stale=1`` — for replacing families (``home_city``, …) any
different value marks stale; for accumulating families (``pets``,
``preferences.*``, …) only a *conflicting polarity on the same value*
marks stale (a new pet does not invalidate "has dog"); a unit that
agrees is folded into support and clears staleness. A lone new fact is
enough to flag that the belief may have moved even before it reaches
the corroboration threshold itself.

Near-dup merge (V7-14.04): within a touched slot, observation pairs
whose generated text matches at ``difflib.SequenceMatcher`` ratio
≥ ``MERGE_TEXT_SIM`` (0.9) fold into the *older* observation (earliest
``first_us``, tie by ``obs_id``): support refs union, ``proof_count``
becomes the union size, ``first_us``/``last_us`` span both, and the fold
is recorded in ``obs_merges_v7`` with the absorbed row's full prior
contents — a merge is a durable, reversible link, not a rewrite.

Profiles (V7-14.06): per touched subject, ``profiles_v7`` rows
materialize per slot: value = the current group's representative
(distinct live currents → ``disputed``; only event/historical evidence
→ ``historical``; else ``current``), ``support_refs_json`` pinned,
``updated_us`` = latest assertion occurred (deterministic — no wall
clock lands in the artifact). Profile rows with no pinned support are
removed — an ungrounded slot is never materialized.

Standing queries (V7-14.07): after the pass, every ``standing_queries``
row in the scope whose ``filters_json`` cannot provably exclude ALL
changed units is marked ``dirty=1``. The provable exclusion keys are
``sessions``/``session_id``/``session``, ``subjects``/``subject``/
``subject_canon``/``entities``/``canons``, and ``units``/``unit_ids``;
any other filter key, an unparseable blob, or an empty filter is
in-doubt → dirty (conservative, per the wave brief).

Closure (V7-14.08): every pass sweeps the scope's observations and
profile rows; a row whose support refs name a unit_id no longer present
in ``units`` at this generation is retired (deleted) and counted in
``stats["retired"]``. The sweep is deliberately not limited to touched
slots — a deletion does not move the insert cursor, so a touched-only
check would strand orphans.

Budget / backlog (V7-14.05): each call scans ≤ ``budget_units`` units
ordered by rowid, persists the cursor in the same transaction, and
reports ``processed``/``remaining``. ``backlog_v7`` reports
``unconsolidated_units``, ``oldest_age_us``, an inter-call throughput
estimate, and ``warning=True`` past the spec's surfaced-health bounds
(> 10,000 units or > 1 hour old).

Per-run trace (V8-13.02, D8-19 diagnosis): ``stats["trace"]`` carries
structured observability for the pass — ``slots`` (candidate slots
considered, with assertion/candidate counts), ``candidates`` (total
``(slot, polarity, value)`` groups evaluated), ``rejected`` (one entry
per candidate per failed gate, each naming ``gate`` + ``reason`` and
the measured counts that failed), ``accepted``/``inserted``/
``updated``/``unchanged``/``written`` write-path counters, ``merged``
(near-dup folds), ``profiles`` (materialization writes + gate-named
rejections), and mirror counters ``skipped_unpinned``/``retired``/
``units_with_assertions``. The trace is diagnosis-only: no gate reads
it, and nothing about what consolidates changes — observations stay
proof-pinned (V7-14.01).

Tables read are all gated through ``has_table`` — a scope missing the
fact tables consolidates nothing and reports honestly, and a missing
``units`` table yields ``status="unavailable"`` rather than a traceback.
Side tables (``consolidation_cursor_v7``, ``consolidation_seen_v7``,
``obs_merges_v7``) live in this file's own lazy DDL — executed
statement-by-statement inside the caller's transaction, never via
``executescript`` (schema_v7 discipline carried).
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError, json_dumps, safe_json_loads
from ..storage.repos import has_table

CONSOLIDATE_VERSION = "consolidate_v7/v1"
PRODUCER_T0 = "consolidate_v7/t0"
FORMULA_STATUS = "provisional/v7-r0"

#: V7-14.04 merge gate — canonical slot equality + text similarity.
MERGE_TEXT_SIM = 0.9
#: V7-14.01 — minimum distinct units AND distinct evidence families.
MIN_SUPPORT_UNITS = 2
MIN_SUPPORT_FAMILIES = 2
#: Bound on stored contradiction refs per observation.
MAX_CONTRADICT_REFS = 8
#: V8-13.02 — bound on support unit ids echoed per trace rejection.
MAX_TRACE_UNITS = 8
#: IN-clause chunk width for batched lookups.
MAX_BATCH_IN = 400
#: V7-14.05 surfaced-health thresholds.
BACKLOG_WARN_UNITS = 10_000
BACKLOG_WARN_AGE_US = 3_600 * 1_000_000

EVIDENCE_FAMILIES = ("state", "preference", "event", "mention")

# ---------------------------------------------------------------------------
# Lazy side-table DDL (owned by this module; never in schema_v7 digests)
# ---------------------------------------------------------------------------

DDL_CONSOLIDATION_V7: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS consolidation_cursor_v7 (
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    last_unit_rowid INTEGER NOT NULL DEFAULT 0,
    run_seq INTEGER NOT NULL DEFAULT 0,
    last_run_us INTEGER,
    prev_run_us INTEGER,
    last_processed INTEGER NOT NULL DEFAULT 0,
    total_processed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, generation)
)""",
    """CREATE TABLE IF NOT EXISTS consolidation_seen_v7 (
    scope_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    seen_run INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, unit_id, generation)
)""",
    """CREATE TABLE IF NOT EXISTS obs_merges_v7 (
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    into_obs_id TEXT NOT NULL,
    from_obs_id TEXT NOT NULL,
    from_row_json TEXT NOT NULL,
    merged_run INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, generation, into_obs_id, from_obs_id)
)""",
)


def ensure_consolidation_v7(conn: sqlite3.Connection) -> None:
    """Create this module's side tables inside the caller's transaction.
    Statement-by-statement ``conn.execute`` — ``executescript`` would
    implicitly COMMIT the surrounding write tx (schema_v7 rule)."""
    for stmt in DDL_CONSOLIDATION_V7:
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Slot-family lexicon (§32.11 names + consolidation-only extras)
# ---------------------------------------------------------------------------
#
# ``phrase`` = affirm template predicate, ``base`` = bare form used by the
# negated/hypothetical variants, ``accumulates`` = multi-valued family (a
# different value does NOT stale an existing observation), ``predicates``
# = events_v7 predicate_lemma families mapped into the slot. Mappings are
# deliberately conservative: ambiguous event families (buy, attend, meet,
# host, born, die, call, book, start, retire) map to no slot — they stay
# events, never inflated into beliefs.

@dataclass(frozen=True)
class _SlotSpec:
    phrase: str
    base: str
    accumulates: bool = False
    predicates: tuple[str, ...] = ()


SLOT_FAMILIES: dict[str, _SlotSpec] = {
    # §32.11 state-key attribute families ------------------------------
    "home_city": _SlotSpec("lives in", "live in", predicates=("move",)),
    "home_country": _SlotSpec("is based in", "be based in"),
    "address": _SlotSpec("lives at", "live at"),
    "employer": _SlotSpec("works at", "work at", predicates=("hire", "quit")),
    "job_title": _SlotSpec("works as", "work as"),
    "team": _SlotSpec("is on team", "be on team"),
    "manager": _SlotSpec("reports to", "report to"),
    "school": _SlotSpec("attends", "attend", predicates=("graduate",)),
    "major": _SlotSpec("studies", "study"),
    "degree": _SlotSpec("holds degree", "hold degree"),
    "relationship_status": _SlotSpec("has relationship status",
                                     "have relationship status"),
    "partner": _SlotSpec("is with", "be with", predicates=("marry",)),
    "children": _SlotSpec("has child", "have child", accumulates=True),
    "pets": _SlotSpec("has pet", "have pet", accumulates=True,
                      predicates=("adopt",)),
    "pet_names": _SlotSpec("has pet named", "have pet named",
                           accumulates=True),
    "birthday": _SlotSpec("has birthday", "have birthday"),
    "age": _SlotSpec("is aged", "be aged"),
    "nationality": _SlotSpec("has nationality", "have nationality"),
    "languages": _SlotSpec("speaks", "speak", accumulates=True),
    "phone": _SlotSpec("has phone", "have phone"),
    "email": _SlotSpec("has email", "have email"),
    "favorite_food": _SlotSpec("favorite food is", "favorite food be"),
    "favorite_color": _SlotSpec("favorite color is", "favorite color be"),
    "favorite_music": _SlotSpec("favorite music is", "favorite music be"),
    "favorite_book": _SlotSpec("favorite book is", "favorite book be"),
    "favorite_movie": _SlotSpec("favorite movie is", "favorite movie be"),
    "hobbies": _SlotSpec("enjoys", "enjoy", accumulates=True,
                         predicates=("cook", "paint", "learn",
                                     "volunteer", "plant")),
    "sport": _SlotSpec("plays sport", "play sport", accumulates=True,
                       predicates=("run",)),
    "diet": _SlotSpec("follows diet", "follow diet"),
    "allergies": _SlotSpec("is allergic to", "be allergic to",
                           accumulates=True),
    "health_condition": _SlotSpec("has health condition",
                                  "have health condition",
                                  accumulates=True, predicates=("sick",)),
    "medication": _SlotSpec("takes medication", "take medication",
                            accumulates=True),
    "car": _SlotSpec("drives", "drive"),
    "device": _SlotSpec("uses device", "use device"),
    "os": _SlotSpec("uses os", "use os"),
    "editor": _SlotSpec("uses editor", "use editor"),
    "programming_language": _SlotSpec("codes in", "code in",
                                      accumulates=True),
    "project_current": _SlotSpec("works on", "work on", accumulates=True,
                                 predicates=("launch", "renovate")),
    "goal_current": _SlotSpec("aims for", "aim for", accumulates=True),
    "plan_upcoming": _SlotSpec("plans", "plan", accumulates=True),
    "travel_upcoming": _SlotSpec("plans travel to", "plan travel to",
                                 accumulates=True),
    "schedule_regular": _SlotSpec("keeps schedule", "keep schedule",
                                  accumulates=True),
    "subscription": _SlotSpec("subscribes to", "subscribe to",
                              accumulates=True),
    "bank_or_payment": _SlotSpec("uses payment", "use payment"),
    "timezone": _SlotSpec("is in timezone", "be in timezone"),
    "preferred_name": _SlotSpec("goes by", "go by"),
    "pronouns": _SlotSpec("uses pronouns", "use pronouns"),
    "workout_routine": _SlotSpec("follows workout", "follow workout",
                                 accumulates=True),
    "reading_current": _SlotSpec("is reading", "be reading",
                                 accumulates=True, predicates=("read",)),
    "show_current": _SlotSpec("is watching", "be watching",
                              accumulates=True),
    # Consolidation-only extras beyond §32.11: past travel, and the
    # preferences.* pseudo-families derived from polarity/strength.
    "travel": _SlotSpec("traveled to", "travel to", accumulates=True,
                        predicates=("visit",)),
    "preferences.likes": _SlotSpec("likes", "like", accumulates=True),
    "preferences.dislikes": _SlotSpec("dislikes", "dislike",
                                      accumulates=True),
    "preferences.habits": _SlotSpec("usually", "usually",
                                    accumulates=True),
    "preferences.favorites": _SlotSpec("has favorite", "have favorite",
                                       accumulates=True),
    "preferences.constraints": _SlotSpec("avoids", "avoid",
                                         accumulates=True),
}

#: ``celebrate`` is the only object-gated predicate: birthday nouns →
#: ``birthday``, a bare number → ``age``, anything else → no slot.
_EVENT_OBJ_GATES: dict[str, tuple[tuple[str, "re.Pattern[str]"], ...]] = {
    "celebrate": (
        ("birthday", re.compile(r"birthday|anniversary", re.IGNORECASE)),
        ("age", re.compile(r"^\d{1,3}$")),
    ),
}

#: predicate_lemma → slot families, built once from SLOT_FAMILIES.
_PREDICATE_TO_SLOTS: dict[str, tuple[str, ...]] = {}
for _fname, _spec in SLOT_FAMILIES.items():
    for _pred in _spec.predicates:
        _PREDICATE_TO_SLOTS[_pred] = (
            _PREDICATE_TO_SLOTS.get(_pred, ()) + (_fname,))

_PREF_STRENGTH_SLOTS = {
    "constraint": "preferences.constraints",
    "favorite": "preferences.favorites",
    "habitual": "preferences.habits",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _canon_fn():
    try:
        from ..enrichment.entities_v2 import canon
        return canon
    except Exception:  # pragma: no cover - documented fallback
        return lambda s: str(s if s is not None else "").strip().casefold()


def _in_clause(name: str, ids: Iterable[str]) -> tuple[str, list[str]]:
    ids = list(ids)
    return f"{name} IN ({','.join('?' for _ in ids)})", ids


def _rows(cur: sqlite3.Cursor) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _now_us(now_us: Optional[int]) -> int:
    if now_us is not None:
        return int(now_us)
    return int(time.time() * 1_000_000)


# ---------------------------------------------------------------------------
# assertions — the per-unit slot claims consolidation works over
# ---------------------------------------------------------------------------

_POL_AFFIRM = "pos"
_POL_NEG = "neg"
_POL_HYP = "hyp"


@dataclass(frozen=True)
class _Assertion:
    """One unit asserting (slot, polarity, value). ``eligible`` marks
    support-eligibility — e.g. historical state_facts are slot activity
    but never support a present-tense observation."""

    unit_id: str
    family: str            # state | preference | event | mention
    subject: str
    family_name: str       # slot family (slot == "{subject}/{family_name}")
    pol: str               # pos | neg | hyp
    value_norm: str        # folded comparison value
    value_text: str        # display surface stored on the fact row
    eligible: bool
    occurred_us: Optional[int]

    @property
    def slot(self) -> str:
        return f"{self.subject}/{self.family_name}"

    @property
    def value_key(self) -> tuple[str, str]:
        return (self.pol, self.value_norm)


def _pref_slot(row: Mapping[str, Any]) -> str:
    pol = str(row.get("polarity") or "").strip().lower()
    if pol in ("negate", "neg", "negative", "dislike"):
        return "preferences.dislikes"
    strength = str(row.get("strength") or "").strip().lower()
    return _PREF_STRENGTH_SLOTS.get(strength, "preferences.likes")


def _pref_pol(row: Mapping[str, Any]) -> str:
    pol = str(row.get("polarity") or "").strip().lower()
    if pol in ("negate", "neg", "negative", "dislike"):
        return _POL_NEG
    if pol in ("hypothetical", "hyp", "modal"):
        return _POL_HYP
    return _POL_AFFIRM


def _event_pol(row: Mapping[str, Any]) -> str:
    pol = str(row.get("polarity") or "").strip().lower()
    if pol in ("negate", "neg", "negative"):
        return _POL_NEG
    if pol in ("hypothetical", "hyp", "modal", "uncertain"):
        return _POL_HYP
    return _POL_AFFIRM


def _event_families(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Slot families for one events_v7 row — the predicate→family map plus
    the gated ``celebrate`` split (an ungated object lands nowhere —
    conservative)."""
    pred = row.get("predicate_lemma")
    if pred is None:
        return ()
    pred = str(pred)
    gates = _EVENT_OBJ_GATES.get(pred)
    if gates is not None:
        obj = str(row.get("object_text") or "")
        return tuple(f for f, rx in gates if rx.search(obj))
    return _PREDICATE_TO_SLOTS.get(pred, ())


def _state_assertion(r: Mapping[str, Any], canon) -> Optional[_Assertion]:
    sk = r.get("state_key") or ""
    if "/" not in sk:
        return None
    subj, fam = sk.split("/", 1)
    if not subj or not fam:
        return None
    status = str(r.get("status") or "current").strip().lower()
    vtext = r.get("value_text") or ""
    return _Assertion(
        str(r["unit_id"]), "state", subj, fam, _POL_AFFIRM,
        str(r.get("value_norm") or canon(vtext)), vtext,
        eligible=status in ("current", "disputed"),
        occurred_us=r.get("valid_from_us"),
    )


def _pref_assertion(r: Mapping[str, Any], canon) -> Optional[_Assertion]:
    subj = r.get("subject_canon")
    if not subj:
        return None
    obj = r.get("object_text") or ""
    return _Assertion(
        str(r["unit_id"]), "preference", str(subj), _pref_slot(r),
        _pref_pol(r), canon(obj), obj, eligible=True,
        occurred_us=r.get("occurred_start_us"),
    )


def _event_assertions(r: Mapping[str, Any], canon) -> list[_Assertion]:
    subj = r.get("subject_canon")
    if not subj:
        return []
    obj = r.get("object_text") or ""
    out = []
    for fam in _event_families(r):
        out.append(_Assertion(
            str(r["unit_id"]), "event", str(subj), fam, _event_pol(r),
            canon(obj), obj, eligible=True,
            occurred_us=r.get("occurred_start_us"),
        ))
    return out


def _batch_assertions(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    unit_ids: Iterable[str],
    canon,
) -> dict[str, list[_Assertion]]:
    """Slot assertions for a batch of units — one chunked IN query per
    fact table (never N per-unit statements)."""
    out: dict[str, list[_Assertion]] = {u: [] for u in unit_ids}
    ids = sorted(set(unit_ids))
    if not ids:
        return out

    def _add(a: Optional[_Assertion]) -> None:
        if a is not None and a.unit_id in out:
            out[a.unit_id].append(a)

    if has_table(conn, "state_facts"):
        for i in range(0, len(ids), MAX_BATCH_IN):
            where, params = _in_clause("unit_id", ids[i:i + MAX_BATCH_IN])
            for r in _rows(conn.execute(
                "SELECT unit_id, state_key, value_text, value_norm,"
                " valid_from_us, status FROM state_facts"
                f" WHERE scope_id=? AND generation=? AND {where}",
                (scope_id, generation, *params),
            )):
                _add(_state_assertion(r, canon))
    if has_table(conn, "preferences"):
        for i in range(0, len(ids), MAX_BATCH_IN):
            where, params = _in_clause("unit_id", ids[i:i + MAX_BATCH_IN])
            for r in _rows(conn.execute(
                "SELECT unit_id, subject_canon, object_text, polarity,"
                " strength, occurred_start_us FROM preferences"
                f" WHERE scope_id=? AND generation=? AND {where}",
                (scope_id, generation, *params),
            )):
                _add(_pref_assertion(r, canon))
    if has_table(conn, "events_v7"):
        for i in range(0, len(ids), MAX_BATCH_IN):
            where, params = _in_clause("unit_id", ids[i:i + MAX_BATCH_IN])
            for r in _rows(conn.execute(
                "SELECT unit_id, subject_canon, predicate_lemma,"
                " object_text, polarity, occurred_start_us"
                " FROM events_v7"
                f" WHERE scope_id=? AND generation=? AND {where}",
                (scope_id, generation, *params),
            )):
                for a in _event_assertions(r, canon):
                    _add(a)
    return out


def _load_units_by_id(
    conn: sqlite3.Connection, scope_id: str, generation: int,
    unit_ids: Iterable[str],
) -> dict[str, dict]:
    ids = sorted(set(unit_ids))
    if not ids or not has_table(conn, "units"):
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(ids), MAX_BATCH_IN):
        where, params = _in_clause("unit_id", ids[i:i + MAX_BATCH_IN])
        cur = conn.execute(
            "SELECT unit_id, session_id, speaker_canon, recorded_at_us,"
            " occurred_start_us, occurred_end_us, byte_start, byte_end,"
            " source_id, revision FROM units"
            f" WHERE scope_id=? AND generation=? AND {where}",
            (scope_id, generation, *params),
        )
        for row in _rows(cur):
            out[row["unit_id"]] = row
    return out


def _slot_assertions(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    subject: str,
    family_name: str,
    canon,
) -> list[_Assertion]:
    """ALL current assertions in one (subject, family) slot — the full
    recompute that makes re-derivation byte-identical (never an
    incremental fold of only-new inputs)."""
    slot = f"{subject}/{family_name}"
    out: list[_Assertion] = []
    unit_ids: set[str] = set()

    if has_table(conn, "state_facts") \
            and not family_name.startswith("preferences."):
        for r in _rows(conn.execute(
            "SELECT unit_id, state_key, value_text, value_norm,"
            " valid_from_us, status FROM state_facts"
            " WHERE scope_id=? AND generation=? AND state_key=?",
            (scope_id, generation, slot),
        )):
            a = _state_assertion(r, canon)
            if a is not None:
                out.append(a)
                unit_ids.add(a.unit_id)

    if has_table(conn, "preferences") \
            and family_name.startswith("preferences."):
        for r in _rows(conn.execute(
            "SELECT unit_id, subject_canon, object_text, polarity,"
            " strength, occurred_start_us FROM preferences"
            " WHERE scope_id=? AND generation=? AND subject_canon=?",
            (scope_id, generation, subject),
        )):
            a = _pref_assertion(r, canon)
            if a is not None and a.family_name == family_name:
                out.append(a)
                unit_ids.add(a.unit_id)

    if has_table(conn, "events_v7"):
        preds = [p for p, fams in _PREDICATE_TO_SLOTS.items()
                 if family_name in fams]
        if preds:
            where, params = _in_clause("predicate_lemma", preds)
            for r in _rows(conn.execute(
                "SELECT unit_id, subject_canon, predicate_lemma,"
                " object_text, polarity, occurred_start_us"
                " FROM events_v7"
                f" WHERE scope_id=? AND generation=? AND"
                f" subject_canon=? AND {where}",
                (scope_id, generation, subject, *params),
            )):
                for a in _event_assertions(r, canon):
                    if a.family_name == family_name:
                        out.append(a)
                        unit_ids.add(a.unit_id)

    # backfill occurred from the units rows (fact occurred wins)
    umap = _load_units_by_id(conn, scope_id, generation, unit_ids)
    fixed: list[_Assertion] = []
    for a in out:
        u = umap.get(a.unit_id)
        if u is None:
            continue  # fact outlived its unit — closure owns the artifact
        fixed.append(_Assertion(
            a.unit_id, a.family, a.subject, a.family_name, a.pol,
            a.value_norm, a.value_text, a.eligible,
            a.occurred_us if a.occurred_us is not None else (
                u.get("occurred_start_us") or u.get("recorded_at_us")),
        ))

    # Mention corroboration (family "mention"): a unit co-mentioning the
    # subject canon AND the normalized value corroborates the affirmative
    # value group. Co-mention is documented weak evidence — it joins the
    # family count but never substitutes for a second unit.
    if has_table(conn, "entity_mentions"):
        pos_values = {a.value_norm for a in fixed
                      if a.pol == _POL_AFFIRM and a.value_norm}
        for vnorm in sorted(pos_values):
            cur = conn.execute(
                "SELECT unit_id FROM entity_mentions"
                " WHERE scope_id=? AND generation=? AND canon IN (?, ?)"
                " GROUP BY unit_id HAVING COUNT(DISTINCT canon) = 2",
                (scope_id, generation, subject, vnorm),
            )
            mids = [r[0] for r in cur.fetchall()]
            mmap = _load_units_by_id(conn, scope_id, generation, mids)
            for m_uid in sorted(mids):
                u = mmap.get(m_uid)
                if u is None:
                    continue
                fixed.append(_Assertion(
                    m_uid, "mention", subject, family_name, _POL_AFFIRM,
                    vnorm, vnorm, eligible=True,
                    occurred_us=(u.get("occurred_start_us")
                                 or u.get("recorded_at_us")),
                ))
    return fixed


# ---------------------------------------------------------------------------
# quotes / support refs — byte-grounded (V7-13.06, V7-14.01)
# ---------------------------------------------------------------------------


def _unit_quote(
    conn: sqlite3.Connection,
    unit: Optional[Mapping[str, Any]],
    payload_cache: dict,
) -> Optional[str]:
    """The unit's pinned payload slice — ``payload[byte_start:byte_end]``
    strict-decoded. ``None`` when the pin cannot resolve: the unit then
    contributes nothing to support (never a quote-less ref)."""
    if unit is None:
        return None
    src = unit.get("source_id")
    rev = unit.get("revision")
    bs = unit.get("byte_start")
    be = unit.get("byte_end")
    if src is None or rev is None or bs is None or be is None:
        return None
    if not has_table(conn, "source_revisions"):
        return None
    key = (src, int(rev))
    if key not in payload_cache:
        row = conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id=? AND revision=?",
            key,
        ).fetchone()
        payload = row[0] if row else None
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        payload_cache[key] = None if payload is None else bytes(payload)
    payload = payload_cache[key]
    if payload is None:
        return None
    try:
        return payload[int(bs):int(be)].decode("utf-8")
    except (UnicodeDecodeError, ValueError, TypeError):
        return None


def _support_refs(
    conn: sqlite3.Connection,
    members: list[_Assertion],
    umap: Mapping[str, Mapping[str, Any]],
    quote_cache: dict,
) -> tuple[list[dict], int]:
    """One pinned ref per supporting unit (dedup by unit_id, families
    unioned), sorted by unit_id — deterministic byte-identical output.
    Units whose quote cannot be lifted are excluded and counted."""
    per_unit: dict[str, set[str]] = {}
    for a in members:
        if not a.eligible:
            continue
        per_unit.setdefault(a.unit_id, set()).add(a.family)
    refs: list[dict] = []
    skipped = 0
    for uid in sorted(per_unit):
        quote = _unit_quote(conn, umap.get(uid), quote_cache)
        if quote is None:
            skipped += 1
            continue
        u = umap[uid]
        refs.append({
            "unit_id": uid,
            "quote": quote,
            "byte_start": u.get("byte_start"),
            "byte_end": u.get("byte_end"),
            "families": sorted(per_unit[uid]),
        })
    return refs, skipped


def _refs_unit_ids(refs_json: Any) -> set[str]:
    data = safe_json_loads(refs_json) if isinstance(refs_json, str) else refs_json
    out: set[str] = set()
    if isinstance(data, list):
        for r in data:
            if isinstance(r, Mapping) and r.get("unit_id"):
                out.add(str(r["unit_id"]))
    return out


# ---------------------------------------------------------------------------
# observation materialization
# ---------------------------------------------------------------------------


def _obs_id(scope_id: str, slot: str, pol: str, value_norm: str,
            generation: int) -> str:
    canon = {
        "v": CONSOLIDATE_VERSION,
        "scope": scope_id,
        "slot": slot,
        "pol": pol,
        "value": value_norm,
        "generation": int(generation),
    }
    return "obs7:" + hashlib.sha256(
        json_dumps(canon).encode("utf-8")).hexdigest()[:24]


def _obs_text(subject: str, family_name: str, pol: str,
              value_norm: str) -> str:
    spec = SLOT_FAMILIES.get(family_name)
    phrase = spec.phrase if spec else f"has {family_name}"
    base = spec.base if spec else f"have {family_name}"
    if pol == _POL_NEG:
        return f"{subject} does not {base} {value_norm}"
    if pol == _POL_HYP:
        return f"{subject} may {base} {value_norm}"
    return f"{subject} {phrase} {value_norm}"


def _obs_parse(obs: Mapping[str, Any]) -> tuple[str, str]:
    """Invert the fixed template → (pol, value_norm). Exact for every row
    this module writes; foreign rows fall back to (pos, tail heuristic)."""
    text = obs.get("text") or ""
    slot = obs.get("slot") or ""
    subject, _, fam = slot.partition("/")
    spec = SLOT_FAMILIES.get(fam)
    base = spec.base if spec else f"have {fam}"
    phrase = spec.phrase if spec else f"has {fam}"
    for pol, ph in ((_POL_NEG, f"does not {base}"),
                    (_POL_HYP, f"may {base}"), (_POL_AFFIRM, phrase)):
        prefix = f"{subject} {ph}"
        if text == prefix:
            return pol, ""
        if text.startswith(prefix + " "):
            return pol, text[len(prefix) + 1:]
    # unrecognized shape — last-token heuristic
    tail = text[len(subject):].strip() if text.startswith(subject) else text
    parts = tail.rsplit(" ", 1)
    return _POL_AFFIRM, (parts[-1] if parts else tail)


_OBS_COLS = ("obs_id", "scope_id", "slot", "text", "producer",
             "proof_count", "support_refs_json", "contradict_refs_json",
             "first_us", "last_us", "stale", "generation")


def _obs_row_for(
    scope_id: str, generation: int,
    subject: str, family_name: str, pol: str, value_norm: str,
    refs: list[dict], contradict: list[dict],
    members: list[_Assertion],
) -> dict:
    occs = [a.occurred_us for a in members
            if a.eligible and a.occurred_us is not None]
    return {
        "obs_id": _obs_id(scope_id, f"{subject}/{family_name}", pol,
                          value_norm, generation),
        "scope_id": scope_id,
        "slot": f"{subject}/{family_name}",
        "text": _obs_text(subject, family_name, pol, value_norm),
        "producer": PRODUCER_T0,
        "proof_count": len(refs),
        "support_refs_json": json_dumps(refs),
        "contradict_refs_json": json_dumps(
            contradict[:MAX_CONTRADICT_REFS]),
        "first_us": min(occs) if occs else None,
        "last_us": max(occs) if occs else None,
        "stale": 0,
        "generation": int(generation),
    }


def _fetch_obs(
    conn: sqlite3.Connection, scope_id: str, generation: int, slot: str
) -> list[dict]:
    if not has_table(conn, "observations_v7"):
        return []
    return _rows(conn.execute(
        f"SELECT {','.join(_OBS_COLS)} FROM observations_v7"
        " WHERE scope_id=? AND generation=? AND slot=?",
        (scope_id, generation, slot),
    ))


def _write_obs(conn: sqlite3.Connection, row: dict) -> str:
    """Insert-or-update the deterministic row. Returns
    'inserted'|'updated'|'unchanged' — identical fields write nothing
    (idempotent re-derivation)."""
    existing = conn.execute(
        f"SELECT {','.join(_OBS_COLS)} FROM observations_v7"
        " WHERE obs_id=?",
        (row["obs_id"],),
    ).fetchone()
    if existing is None:
        conn.execute(
            f"INSERT INTO observations_v7({','.join(_OBS_COLS)})"
            f" VALUES ({','.join('?' for _ in _OBS_COLS)})",
            tuple(row[c] for c in _OBS_COLS),
        )
        return "inserted"
    old = dict(zip(_OBS_COLS, existing))
    changed = any(old.get(c) != row[c] for c in _OBS_COLS
                  if c != "obs_id")
    if not changed:
        return "unchanged"
    conn.execute(
        "UPDATE observations_v7 SET "
        + ", ".join(f"{c}=?" for c in _OBS_COLS if c != "obs_id")
        + " WHERE obs_id=?",
        tuple(row[c] for c in _OBS_COLS if c != "obs_id")
        + (row["obs_id"],),
    )
    return "updated"


def _set_stale(conn: sqlite3.Connection, obs_id: str, stale: int) -> None:
    conn.execute(
        "UPDATE observations_v7 SET stale=? WHERE obs_id=? AND stale!=?",
        (stale, obs_id, stale),
    )


def _delete_obs(conn: sqlite3.Connection, obs_id: str) -> None:
    conn.execute("DELETE FROM observations_v7 WHERE obs_id=?", (obs_id,))


# ---------------------------------------------------------------------------
# stale marking + merges inside one slot
# ---------------------------------------------------------------------------


def _unit_newer(a: _Assertion, obs_last_us: Optional[int]) -> bool:
    """The occurred-vs-last_us compare: an assertion is 'new' when its
    time is after the observation's last support time; un-timed
    assertions are conservatively new (in-doubt → stale)."""
    t = a.occurred_us
    if t is None:
        return True
    if obs_last_us is None:
        return True
    return t > obs_last_us


def _slot_stale_flags(
    slot_assertions: list[_Assertion],
    obs_rows: list[dict],
    spec: _SlotSpec,
) -> dict[str, int]:
    """For each obs row in the slot: 1 iff some slot assertion is newer,
    absent from the obs's support, and meaningful for the family —
    replacing families stale on any different value; accumulating
    families stale only on a conflicting-polarity assertion of the SAME
    value (a fresh accumulated item is not a revision)."""
    flags: dict[str, int] = {}
    parsed = {o["obs_id"]: _obs_parse(o) for o in obs_rows}
    for obs in obs_rows:
        pol_o, val_o = parsed[obs["obs_id"]]
        support = _refs_unit_ids(obs["support_refs_json"])
        stale = 0
        for a in slot_assertions:
            if a.unit_id in support:
                continue
            if not _unit_newer(a, obs["last_us"]):
                continue
            if a.value_norm == val_o:
                if a.pol != pol_o:
                    stale = 1  # polarity conflict on the same value
                    break
                continue  # agrees but unpinned — belief unchanged
            if spec.accumulates:
                continue  # different accumulated item — not a revision
            stale = 1  # replacing family, different value
            break
        flags[obs["obs_id"]] = stale
    return flags


def _merge_slot_obs(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    obs_rows: list[dict],
    run_seq: int,
) -> tuple[int, list[dict]]:
    """Near-duplicate reconciliation (V7-14.04): slot-equal pairs at text
    similarity ≥ 0.9 fold into the older row; the fold is durable in
    ``obs_merges_v7`` (the absorbed row's prior contents — reversible)."""
    rows = sorted(obs_rows, key=lambda o: (
        o["first_us"] if o["first_us"] is not None else 0, o["obs_id"]))
    merged: set[str] = set()
    merges: list[dict] = []
    for i, keep in enumerate(rows):
        if keep["obs_id"] in merged:
            continue
        for drop in rows[i + 1:]:
            if drop["obs_id"] in merged:
                continue
            ratio = difflib.SequenceMatcher(
                None, keep["text"], drop["text"]).ratio()
            if ratio < MERGE_TEXT_SIM:
                continue
            by_uid: dict[str, dict] = {}
            for r in (safe_json_loads(keep["support_refs_json"]) or []) \
                    + (safe_json_loads(drop["support_refs_json"]) or []):
                if not isinstance(r, Mapping) or not r.get("unit_id"):
                    continue
                uid = str(r["unit_id"])
                if uid in by_uid:
                    fams = set(by_uid[uid].get("families") or [])
                    fams.update(r.get("families") or [])
                    by_uid[uid]["families"] = sorted(fams)
                else:
                    by_uid[uid] = dict(r)
            refs = [by_uid[u] for u in sorted(by_uid)]
            # Contradict refs follow the same union rule (keep ∪ drop,
            # dedup'd, bounded) MINUS (a) units that just joined support
            # and (b) refs naming either row's own (pol, value) — a
            # contradict ref must name an OTHER value group, never the
            # observation's own belief (D8-33).
            kpol, kval = _obs_parse(keep)
            dpol, dval = _obs_parse(drop)
            ckey: set = set()
            contradict: list[dict] = []
            for r in (safe_json_loads(keep["contradict_refs_json"])
                      or []) + (safe_json_loads(
                          drop["contradict_refs_json"]) or []):
                if not isinstance(r, Mapping) or not r.get("unit_id"):
                    continue
                if str(r["unit_id"]) in by_uid:
                    continue
                if (r.get("pol"), r.get("value")) in (
                        (kpol, kval), (dpol, dval)):
                    continue
                k = (str(r["unit_id"]), r.get("value"), r.get("pol"))
                if k in ckey:
                    continue
                ckey.add(k)
                contradict.append(dict(r))
                if len(contradict) >= MAX_CONTRADICT_REFS:
                    break
            firsts = [t for t in (keep["first_us"], drop["first_us"])
                      if t is not None]
            lasts = [t for t in (keep["last_us"], drop["last_us"])
                     if t is not None]
            new_refs = json_dumps(refs)
            new_contra = json_dumps(contradict)
            new_proof = max(len(refs), int(keep["proof_count"] or 0),
                            int(drop["proof_count"] or 0))
            new_first = min(firsts) if firsts else None
            new_last = max(lasts) if lasts else None
            conn.execute(
                "UPDATE observations_v7 SET support_refs_json=?,"
                " contradict_refs_json=?, proof_count=?, first_us=?,"
                " last_us=? WHERE obs_id=?",
                (new_refs, new_contra, new_proof, new_first, new_last,
                 keep["obs_id"]),
            )
            # Fold-forward (D8-33): the surviving row's in-memory copy
            # must reflect each fold — a second drop in the same pass
            # unions against the post-merge refs, not the originals,
            # or earlier drops' support silently vanishes.
            keep["support_refs_json"] = new_refs
            keep["contradict_refs_json"] = new_contra
            keep["proof_count"] = new_proof
            keep["first_us"] = new_first
            keep["last_us"] = new_last
            conn.execute(
                "INSERT OR REPLACE INTO obs_merges_v7(scope_id, generation,"
                " into_obs_id, from_obs_id, from_row_json, merged_run)"
                " VALUES (?,?,?,?,?,?)",
                (scope_id, generation, keep["obs_id"], drop["obs_id"],
                 json_dumps({c: drop[c] for c in _OBS_COLS}),
                 run_seq),
            )
            _delete_obs(conn, drop["obs_id"])
            merged.add(drop["obs_id"])
            merges.append({"into": keep["obs_id"], "from": drop["obs_id"],
                           "similarity": round(ratio, 6)})
    return len(merges), merges


# ---------------------------------------------------------------------------
# profiles_v7 materialization (V7-14.06)
# ---------------------------------------------------------------------------


def _subject_assertions(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    subject: str,
    canon,
) -> list[_Assertion]:
    """Every asserted fact for one subject — all §32.11 families plus
    preferences.* — from state_facts / preferences / events_v7."""
    out: list[_Assertion] = []
    if has_table(conn, "state_facts"):
        prefix = f"{subject}/"
        for r in _rows(conn.execute(
            "SELECT state_key, unit_id, value_text, value_norm,"
            " valid_from_us, status FROM state_facts"
            " WHERE scope_id=? AND generation=?"
            " AND substr(state_key, 1, ?) = ?",
            (scope_id, generation, len(prefix), prefix),
        )):
            fam = (r["state_key"] or "")[len(prefix):]
            if not fam:
                continue
            status = str(r.get("status") or "current").strip().lower()
            vtext = r.get("value_text") or ""
            out.append(_Assertion(
                str(r["unit_id"]), "state", subject, fam, _POL_AFFIRM,
                str(r.get("value_norm") or canon(vtext)), vtext,
                eligible=status in ("current", "disputed"),
                occurred_us=r.get("valid_from_us"),
            ))
    if has_table(conn, "preferences"):
        for r in _rows(conn.execute(
            "SELECT unit_id, subject_canon, object_text, polarity,"
            " strength, occurred_start_us FROM preferences"
            " WHERE scope_id=? AND generation=? AND subject_canon=?",
            (scope_id, generation, subject),
        )):
            a = _pref_assertion(r, canon)
            if a is not None:
                out.append(a)
    if has_table(conn, "events_v7"):
        for r in _rows(conn.execute(
            "SELECT unit_id, subject_canon, predicate_lemma, object_text,"
            " polarity, occurred_start_us FROM events_v7"
            " WHERE scope_id=? AND generation=? AND subject_canon=?",
            (scope_id, generation, subject),
        )):
            out.extend(_event_assertions(r, canon))
    # backfill occurred from units
    umap = _load_units_by_id(
        conn, scope_id, generation, {a.unit_id for a in out})
    fixed = []
    for a in out:
        u = umap.get(a.unit_id)
        if u is None:
            continue
        fixed.append(_Assertion(
            a.unit_id, a.family, a.subject, a.family_name, a.pol,
            a.value_norm, a.value_text, a.eligible,
            a.occurred_us if a.occurred_us is not None else (
                u.get("occurred_start_us") or u.get("recorded_at_us")),
        ))
    return fixed


def _profile_live(a: _Assertion) -> bool:
    """Does this assertion make its value a *live* profile candidate?
    Affirmed facts only — a negated or hypothetical claim never becomes
    the slot's face value."""
    if a.pol != _POL_AFFIRM:
        return False
    if a.family == "state":
        return a.eligible  # current/disputed, never historical
    return a.family in ("preference", "event")


def _materialize_profiles(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    subjects: Iterable[str],
    canon,
    quote_cache: dict,
) -> tuple[int, list[str], list[dict]]:
    """Upsert ``profiles_v7`` per (subject, slot). Face value = the live
    group's representative (latest occurred wins; ties → smallest value);
    multiple live values → ``disputed``; only events/historical state →
    ``historical``; else ``current``. Rows rewrite only on a real diff;
    rows left with zero pinned support are removed (never materialize an
    ungrounded slot). Returns (written, retired_keys, rejected) — the
    third element is the V8-13.02 trace channel: one gate-named entry
    per candidate slot that produced no materialization."""
    if not has_table(conn, "profiles_v7"):
        return 0, [], []
    written = 0
    retired: list[str] = []
    rejected: list[dict] = []
    for subject in sorted(set(subjects)):
        assertions = _subject_assertions(
            conn, scope_id, generation, subject, canon)
        umap = _load_units_by_id(
            conn, scope_id, generation,
            {a.unit_id for a in assertions})
        by_slot: dict[str, list[_Assertion]] = {}
        for a in assertions:
            by_slot.setdefault(a.family_name, []).append(a)
        for slot_name in sorted(by_slot):
            members = by_slot[slot_name]
            groups: dict[str, list[_Assertion]] = {}
            for a in members:
                groups.setdefault(a.value_norm, []).append(a)

            currents: list[tuple[str, int]] = []   # (value_norm, latest)
            historical: list[tuple[str, int]] = []
            for vnorm, ms in groups.items():
                live = [m for m in ms if _profile_live(m)]
                if not live:
                    continue  # neg/hyp-only groups never face a slot
                latest = max((m.occurred_us or 0) for m in ms)
                is_current = any(
                    (m.family == "state" and m.eligible)
                    or (m.family == "preference" and m.pol == _POL_AFFIRM)
                    for m in ms)
                (currents if is_current else historical).append(
                    (vnorm, latest))
            if currents:
                status = "disputed" if len(
                    {v for v, _ in currents}) > 1 else "current"
                pool = currents
            elif historical:
                status = "historical"
                pool = historical
            else:
                pool = []
            chosen: Optional[str] = None
            if pool:
                chosen = sorted(pool, key=lambda c: (-c[1], c[0]))[0][0]

            if chosen is None:
                # nothing live left — remove any stale materialization
                res = conn.execute(
                    "DELETE FROM profiles_v7 WHERE scope_id=? AND"
                    " subject_canon=? AND slot=? AND generation=?",
                    (scope_id, subject, slot_name, generation),
                )
                if res.rowcount:
                    retired.append(f"profiles_v7:{subject}|{slot_name}")
                rejected.append({
                    "subject": subject, "slot": slot_name,
                    "gate": "profile_live",
                    "reason": "no live value group",
                    "deleted_prior": bool(res.rowcount)})
                continue

            chosen_members = [m for m in groups[chosen] if m.eligible]
            refs, _sk = _support_refs(conn, chosen_members, umap,
                                      quote_cache)
            if not refs:
                res = conn.execute(
                    "DELETE FROM profiles_v7 WHERE scope_id=? AND"
                    " subject_canon=? AND slot=? AND generation=?",
                    (scope_id, subject, slot_name, generation),
                )
                if res.rowcount:
                    retired.append(f"profiles_v7:{subject}|{slot_name}")
                rejected.append({
                    "subject": subject, "slot": slot_name,
                    "gate": "profile_pin",
                    "reason": "zero pinned support refs",
                    "deleted_prior": bool(res.rowcount)})
                continue
            updated = max((m.occurred_us or 0) for m in members)
            value_text = next(
                (m.value_text for m in sorted(
                    chosen_members, key=lambda m: m.unit_id)
                  if m.value_text), chosen)
            new_refs = json_dumps(refs)
            existing = conn.execute(
                "SELECT value, status, support_refs_json, updated_us"
                " FROM profiles_v7 WHERE scope_id=? AND subject_canon=?"
                " AND slot=? AND generation=?",
                (scope_id, subject, slot_name, generation),
            ).fetchone()
            if existing is not None and tuple(existing) == (
                    value_text, status, new_refs, updated):
                continue
            conn.execute(
                "INSERT INTO profiles_v7(scope_id, subject_canon, slot,"
                " generation, value, status, support_refs_json,"
                " updated_us) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(scope_id, subject_canon, slot, generation)"
                " DO UPDATE SET value=excluded.value,"
                " status=excluded.status,"
                " support_refs_json=excluded.support_refs_json,"
                " updated_us=excluded.updated_us",
                (scope_id, subject, slot_name, generation, value_text,
                 status, new_refs, updated),
            )
            written += 1
    return written, retired, rejected


# ---------------------------------------------------------------------------
# standing queries — scoped-change dirty marking (V7-14.07)
# ---------------------------------------------------------------------------

_SQ_SESSION_KEYS = ("sessions", "session_id", "session")
_SQ_SUBJECT_KEYS = ("subjects", "subject", "subject_canon", "entities",
                    "canons")
_SQ_UNIT_KEYS = ("units", "unit_ids")
_SQ_KNOWN_KEYS = frozenset(
    _SQ_SESSION_KEYS + _SQ_SUBJECT_KEYS + _SQ_UNIT_KEYS)


def _filter_set(filters: Mapping[str, Any],
                keys: tuple[str, ...]) -> Optional[set]:
    vals: set = set()
    seen = False
    for k in keys:
        v = filters.get(k)
        if v is None:
            continue
        seen = True
        if isinstance(v, (list, tuple, set)):
            vals.update(str(x) for x in v)
        else:
            vals.add(str(v))
    return vals if seen else None


def _sq_excluded(filters: Any, changed: list[Mapping[str, Any]]) -> bool:
    """True only when the stored filters provably exclude EVERY changed
    unit. A unit is excluded when a constrained dimension misses it:
    session filter disjoint, subject filter disjoint, or an explicit unit
    list not containing it. Unknown filter keys / unparseable blobs /
    empty filters are in-doubt → not excluded (conservative → dirty)."""
    f = safe_json_loads(filters) if isinstance(filters, str) else filters
    if not isinstance(f, Mapping) or not f:
        return False
    if any(k not in _SQ_KNOWN_KEYS for k in f):
        return False
    sessions = _filter_set(f, _SQ_SESSION_KEYS)
    subjects = _filter_set(f, _SQ_SUBJECT_KEYS)
    units = _filter_set(f, _SQ_UNIT_KEYS)
    if sessions is None and subjects is None and units is None:
        return False
    for ch in changed:
        excluded = False
        if sessions is not None and (
                ch.get("session_id") is None
                or str(ch["session_id"]) not in sessions):
            excluded = True  # session membership is structural — provable
        if subjects is not None:
            ch_subjects = set(ch.get("subjects") or ())
            # a unit with NO extracted subjects is "in doubt" — its text
            # may still mention the filtered subject — never excluded.
            if ch_subjects and not (ch_subjects & subjects):
                excluded = True
        if units is not None and str(ch.get("unit_id")) not in units:
            excluded = True
        if not excluded:
            return False  # this unit may matter → cannot exclude
    return True


def _dirty_standing_queries(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    changed: list[Mapping[str, Any]],
) -> int:
    if not has_table(conn, "standing_queries") or not changed:
        return 0
    rows = _rows(conn.execute(
        "SELECT sq_id, filters_json FROM standing_queries"
        " WHERE scope_id=? AND generation=? AND dirty=0",
        (scope_id, generation),
    ))
    n = 0
    for r in rows:
        if _sq_excluded(r["filters_json"], changed):
            continue
        conn.execute(
            "UPDATE standing_queries SET dirty=1 WHERE sq_id=?",
            (r["sq_id"],))
        n += 1
    return n


# ---------------------------------------------------------------------------
# closure (V7-14.08) — support vanished ⇒ retire
# ---------------------------------------------------------------------------


def _existing_unit_ids(
    conn: sqlite3.Connection, generation: int, unit_ids: Iterable[str]
) -> set[str]:
    ids = sorted(set(unit_ids))
    if not ids:
        return set()
    found: set[str] = set()
    for i in range(0, len(ids), MAX_BATCH_IN):
        where, params = _in_clause("unit_id", ids[i:i + MAX_BATCH_IN])
        cur = conn.execute(
            f"SELECT unit_id FROM units WHERE generation=? AND {where}",
            (generation, *params),
        )
        found.update(r[0] for r in cur.fetchall())
    return found


def _closure_sweep(
    conn: sqlite3.Connection, scope_id: str, generation: int
) -> list[str]:
    """Retire observations/profile rows whose support names a unit absent
    from ``units`` at this generation. Whole-scope every pass — a delete
    does not move the insert cursor, so a touched-only check would leave
    orphans."""
    retired: list[str] = []
    obs_rows: list[dict] = []
    prof_rows: list[dict] = []
    if has_table(conn, "observations_v7"):
        obs_rows = _rows(conn.execute(
            "SELECT obs_id, support_refs_json FROM observations_v7"
            " WHERE scope_id=? AND generation=?",
            (scope_id, generation),
        ))
    if has_table(conn, "profiles_v7"):
        prof_rows = _rows(conn.execute(
            "SELECT subject_canon, slot, support_refs_json"
            " FROM profiles_v7 WHERE scope_id=? AND generation=?",
            (scope_id, generation),
        ))
    pool: set[str] = set()
    for r in obs_rows + prof_rows:
        pool |= _refs_unit_ids(r["support_refs_json"])
    existing = _existing_unit_ids(conn, generation, pool)
    for r in obs_rows:
        if _refs_unit_ids(r["support_refs_json"]) - existing:
            _delete_obs(conn, r["obs_id"])
            retired.append(f"observations_v7:{r['obs_id']}")
    for r in prof_rows:
        if _refs_unit_ids(r["support_refs_json"]) - existing:
            conn.execute(
                "DELETE FROM profiles_v7 WHERE scope_id=? AND"
                " subject_canon=? AND slot=? AND generation=?",
                (scope_id, r["subject_canon"], r["slot"], generation),
            )
            retired.append(f"profiles_v7:{r['subject_canon']}|{r['slot']}")
    return retired


# ---------------------------------------------------------------------------
# cursor
# ---------------------------------------------------------------------------

_CURSOR_COLS = ("scope_id", "generation", "last_unit_rowid", "run_seq",
                "last_run_us", "prev_run_us", "last_processed",
                "total_processed")


def _cursor_row(conn: sqlite3.Connection, scope_id: str,
                generation: int) -> dict:
    row = conn.execute(
        f"SELECT {','.join(_CURSOR_COLS)} FROM consolidation_cursor_v7"
        " WHERE scope_id=? AND generation=?",
        (scope_id, generation),
    ).fetchone()
    if row is None:
        return {"scope_id": scope_id, "generation": generation,
                "last_unit_rowid": 0, "run_seq": 0, "last_run_us": None,
                "prev_run_us": None, "last_processed": 0,
                "total_processed": 0}
    return dict(zip(_CURSOR_COLS, row))


def _write_cursor(conn: sqlite3.Connection, cur: Mapping[str, Any]) -> None:
    conn.execute(
        "INSERT INTO consolidation_cursor_v7(scope_id, generation,"
        " last_unit_rowid, run_seq, last_run_us, prev_run_us,"
        " last_processed, total_processed)"
        " VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(scope_id, generation) DO UPDATE SET"
        " last_unit_rowid=excluded.last_unit_rowid,"
        " run_seq=excluded.run_seq,"
        " last_run_us=excluded.last_run_us,"
        " prev_run_us=excluded.prev_run_us,"
        " last_processed=excluded.last_processed,"
        " total_processed=excluded.total_processed",
        tuple(cur[c] for c in _CURSOR_COLS),
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def consolidate_scope_v7(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    generation: int,
    budget_units: int = 10_000,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """One bounded consolidation pass over ``scope_id`` at ``generation``.

    Pure in-transaction work — never commits, never blocks on a lock: a
    row that fails to read is skipped and counted. Returns honest counts;
    a second call continues from the durable cursor.
    """
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION,
                            "scope_id must be a non-empty string")
    if isinstance(budget_units, bool) or not isinstance(budget_units, int) \
            or budget_units < 1:
        raise VerbatimError(ErrorCode.VALIDATION,
                            "budget_units must be a positive int")
    generation = int(generation)
    stats: dict[str, Any] = {
        "status": "ok",
        "scope_id": scope_id,
        "generation": generation,
        "producer": PRODUCER_T0,
        "formula_status": FORMULA_STATUS,
        "processed": 0,
        "remaining": 0,
        "observations_written": 0,
        "observations_updated": 0,
        "profiles_written": 0,
        "merges": 0,
        "merge_log": [],
        "stale_marked": 0,
        "stale_cleared": 0,
        "retired": [],
        "standing_dirty": 0,
        "slots_touched": 0,
        "skipped_read_errors": 0,
        "skipped_unpinned": 0,
        # V8-13.02 — per-run consolidation trace (diagnosis only; no
        # gate reads it). Populated below, emitted even on early exit.
        "trace": {
            "run_seq": 0,
            "units_processed": 0,
            "units_with_assertions": 0,
            "slots": [],
            "candidates": 0,
            "accepted": 0,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "written": 0,
            "merged": 0,
            "skipped_unpinned": 0,
            "retired": 0,
            "rejected": [],
            "profiles": {"written": 0, "rejected": []},
        },
    }
    trace = stats["trace"]
    ensure_consolidation_v7(conn)
    if not has_table(conn, "units"):
        stats["status"] = "unavailable"
        stats["reason"] = "units_table_missing"
        return stats

    now = _now_us(now_us)
    cur = _cursor_row(conn, scope_id, generation)
    run_seq = int(cur["run_seq"]) + 1
    trace["run_seq"] = run_seq

    # -- batch scan: unconsolidated units (rowid high-water OR never-seen;
    #    the seen-table covers SQLite's deleted-max-rowid reuse) -----------
    batch = _rows(conn.execute(
        "SELECT u.rowid AS _rid, unit_id, source_id, revision, scope_id,"
        " kind, session_id, seq, speaker_canon, recorded_at_us,"
        " occurred_start_us, occurred_end_us, byte_start, byte_end"
        " FROM units u"
        " WHERE scope_id=? AND generation=?"
        " AND (u.rowid > ? OR NOT EXISTS("
        "   SELECT 1 FROM consolidation_seen_v7 s"
        "   WHERE s.scope_id=u.scope_id AND s.unit_id=u.unit_id"
        "     AND s.generation=u.generation))"
        " ORDER BY u.rowid LIMIT ?",
        (scope_id, generation, cur["last_unit_rowid"], budget_units),
    ))
    stats["processed"] = len(batch)
    trace["units_processed"] = len(batch)

    canon = _canon_fn()
    touched: dict[str, tuple[str, str]] = {}   # slot -> (subject, family)
    changed_units: list[dict] = []
    max_rid = int(cur["last_unit_rowid"])

    by_unit = _batch_assertions(
        conn, scope_id, generation, [u["unit_id"] for u in batch], canon)
    for u in batch:
        max_rid = max(max_rid, int(u["_rid"]))
        assertions = by_unit.get(u["unit_id"], [])
        if assertions:
            trace["units_with_assertions"] += 1
        for a in assertions:
            touched[a.slot] = (a.subject, a.family_name)
        changed_units.append({
            "unit_id": u["unit_id"],
            "session_id": u.get("session_id"),
            "subjects": {a.subject for a in assertions},
        })

    quote_cache: dict = {}
    touched_subjects = {s for (s, _f) in touched.values()}

    # -- per touched slot: full recompute → obs upserts → stale → merge ----
    for slot in sorted(touched):
        subject, family_name = touched[slot]
        spec = SLOT_FAMILIES.get(family_name, _SlotSpec(
            phrase=f"has {family_name}", base=f"have {family_name}"))
        slot_rec: dict[str, Any] = {"slot": slot, "subject": subject,
                                    "family": family_name}
        try:
            slot_assertions = _slot_assertions(
                conn, scope_id, generation, subject, family_name, canon)
        except sqlite3.Error as exc:
            stats["skipped_read_errors"] += 1
            slot_rec["assertions"] = None
            slot_rec["candidates"] = 0
            slot_rec["error"] = "slot_read"
            trace["slots"].append(slot_rec)
            trace["rejected"].append({
                "slot": slot, "subject": subject, "family": family_name,
                "gate": "slot_read",
                "reason": f"{type(exc).__name__}: {exc}"})
            continue
        uids = {a.unit_id for a in slot_assertions}
        umap = _load_units_by_id(conn, scope_id, generation, uids)

        groups: dict[tuple[str, str], list[_Assertion]] = {}
        for a in slot_assertions:
            groups.setdefault(a.value_key, []).append(a)
        slot_rec["assertions"] = len(slot_assertions)
        slot_rec["candidates"] = len(groups)
        trace["slots"].append(slot_rec)
        trace["candidates"] += len(groups)

        existing = _fetch_obs(conn, scope_id, generation, slot)
        by_id = {o["obs_id"]: o for o in existing}

        # contradiction refs = the OTHER affirmed/negated value groups in
        # the same slot (hypotheticals never contradict), bounded.
        for (pol, vnorm), members in sorted(groups.items()):
            support_units = {a.unit_id for a in members if a.eligible}
            fams = {a.family for a in members if a.eligible}
            obs_id = _obs_id(scope_id, slot, pol, vnorm, generation)
            cand = {"slot": slot, "pol": pol, "value": vnorm,
                    "support_units": len(support_units),
                    "support_unit_ids": sorted(support_units)[
                        :MAX_TRACE_UNITS],
                    "families": sorted(fams)}
            # V7-14.01 corroboration gates — each failed gate lands its
            # own trace entry so per-gate drop counts are enumerable.
            failed: list[tuple[str, str]] = []
            if len(support_units) < MIN_SUPPORT_UNITS:
                failed.append(("min_support_units",
                               f"{len(support_units)} distinct eligible "
                               f"units < {MIN_SUPPORT_UNITS}"))
            if len(fams) < MIN_SUPPORT_FAMILIES:
                failed.append(("min_support_families",
                               f"{len(fams)} distinct evidence families "
                               f"< {MIN_SUPPORT_FAMILIES}"))
            if failed:
                # support collapsed below threshold → a prior obs for
                # this exact value cannot stand.
                if obs_id in by_id:
                    _delete_obs(conn, obs_id)
                    stats["retired"].append(f"observations_v7:{obs_id}")
                    cand["retired_obs_id"] = obs_id
                for gate, reason in failed:
                    trace["rejected"].append(
                        {**cand, "gate": gate, "reason": reason})
                continue
            refs, skipped = _support_refs(
                conn, members, umap, quote_cache)
            stats["skipped_unpinned"] += skipped
            if len(refs) < MIN_SUPPORT_UNITS:
                # quotes could not be lifted — never write a
                # quote-less observation (V7-14.01 requires them);
                # a prior row loses its grounding → retire.
                if obs_id in by_id:
                    _delete_obs(conn, obs_id)
                    stats["retired"].append(
                        f"observations_v7:{obs_id}")
                    cand["retired_obs_id"] = obs_id
                trace["rejected"].append({
                    **cand, "gate": "proof_pin",
                    "reason": f"{len(refs)} pinned support refs "
                              f"< {MIN_SUPPORT_UNITS} "
                              f"({skipped} quotes unresolvable)"})
                continue
            contradict: list[dict] = []
            for (p2, v2), m2 in sorted(groups.items()):
                if (p2, v2) == (pol, vnorm) or p2 == _POL_HYP:
                    continue
                for a in m2:
                    u = umap.get(a.unit_id)
                    if u is None:
                        continue
                    contradict.append({
                        "unit_id": a.unit_id,
                        "value": v2,
                        "pol": p2,
                        "byte_start": u.get("byte_start"),
                        "byte_end": u.get("byte_end"),
                    })
                if len(contradict) >= MAX_CONTRADICT_REFS:
                    break
            row = _obs_row_for(scope_id, generation, subject,
                               family_name, pol, vnorm, refs,
                               contradict, members)
            res = _write_obs(conn, row)
            if res == "inserted":
                stats["observations_written"] += 1
                trace["inserted"] += 1
            elif res == "updated":
                stats["observations_updated"] += 1
                trace["updated"] += 1
            else:
                trace["unchanged"] += 1
            trace["accepted"] += 1

        # stale flags over ALL obs rows now present in the slot
        live = _fetch_obs(conn, scope_id, generation, slot)
        flags = _slot_stale_flags(slot_assertions, live, spec)
        for o in live:
            new = flags[o["obs_id"]]
            old = int(o["stale"] or 0)
            if new != old:
                _set_stale(conn, o["obs_id"], new)
                if new:
                    stats["stale_marked"] += 1
                else:
                    stats["stale_cleared"] += 1

        # near-dup merge inside the slot
        live = _fetch_obs(conn, scope_id, generation, slot)
        n_merges, log = _merge_slot_obs(
            conn, scope_id, generation, live, run_seq)
        stats["merges"] += n_merges
        stats["merge_log"].extend(log)
        trace["merged"] += n_merges

    stats["slots_touched"] = len(touched)

    # -- profiles for touched subjects --------------------------------------
    try:
        written, prof_retired, prof_rejected = _materialize_profiles(
            conn, scope_id, generation, touched_subjects, canon,
            quote_cache)
        stats["profiles_written"] = written
        stats["retired"].extend(prof_retired)
        trace["profiles"]["written"] = written
        trace["profiles"]["rejected"] = prof_rejected
    except sqlite3.Error:
        stats["skipped_read_errors"] += 1

    # -- standing queries dirty on non-excluded change -----------------------
    try:
        stats["standing_dirty"] = _dirty_standing_queries(
            conn, scope_id, generation, changed_units)
    except sqlite3.Error:
        stats["skipped_read_errors"] += 1

    # -- closure sweep (V7-14.08) --------------------------------------------
    try:
        stats["retired"].extend(_closure_sweep(conn, scope_id, generation))
    except sqlite3.Error:
        stats["skipped_read_errors"] += 1

    # -- mark seen + advance cursor (same tx) --------------------------------
    for u in batch:
        conn.execute(
            "INSERT OR IGNORE INTO consolidation_seen_v7"
            "(scope_id, unit_id, generation, seen_run) VALUES (?,?,?,?)",
            (scope_id, u["unit_id"], generation, run_seq),
        )
    cur["run_seq"] = run_seq
    cur["last_unit_rowid"] = max_rid
    cur["prev_run_us"] = cur["last_run_us"]
    cur["last_run_us"] = now
    cur["last_processed"] = len(batch)
    cur["total_processed"] = int(cur["total_processed"]) + len(batch)
    _write_cursor(conn, cur)

    stats["remaining"] = conn.execute(
        "SELECT COUNT(*) FROM units u WHERE scope_id=? AND generation=?"
        " AND NOT EXISTS(SELECT 1 FROM consolidation_seen_v7 s"
        "  WHERE s.scope_id=u.scope_id AND s.unit_id=u.unit_id"
        "    AND s.generation=u.generation)",
        (scope_id, generation),
    ).fetchone()[0]

    # -- trace finalization (V8-13.02) — mirrors of the durable counters --
    trace["written"] = trace["inserted"] + trace["updated"]
    trace["skipped_unpinned"] = stats["skipped_unpinned"]
    trace["retired"] = len(stats["retired"])
    return stats


# ---------------------------------------------------------------------------
# backlog (V7-14.05)
# ---------------------------------------------------------------------------


def backlog_v7(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    *,
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    """Maintenance-lane health: unconsolidated unit count, oldest
    unprocessed age, and an inter-call throughput estimate. The spec's
    surfaced warning fires past > 10,000 units or > 1 hour."""
    if not isinstance(scope_id, str) or not scope_id:
        raise VerbatimError(ErrorCode.VALIDATION,
                            "scope_id must be a non-empty string")
    if not has_table(conn, "units"):
        return {"status": "unavailable", "reason": "units_table_missing",
                "scope_id": scope_id, "generation": int(generation),
                "unconsolidated_units": 0, "oldest_age_us": None,
                "throughput_units_per_s": None, "warning": False}
    generation = int(generation)
    now = _now_us(now_us)

    if has_table(conn, "consolidation_seen_v7"):
        row = conn.execute(
            "SELECT COUNT(*), MIN(recorded_at_us) FROM units u"
            " WHERE scope_id=? AND generation=?"
            " AND NOT EXISTS(SELECT 1 FROM consolidation_seen_v7 s"
            "  WHERE s.scope_id=u.scope_id AND s.unit_id=u.unit_id"
            "    AND s.generation=u.generation)",
            (scope_id, generation),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*), MIN(recorded_at_us) FROM units"
            " WHERE scope_id=? AND generation=?",
            (scope_id, generation),
        ).fetchone()
    unconsolidated = int(row[0] or 0)
    oldest_recorded = row[1]
    oldest_age = (now - int(oldest_recorded)) if oldest_recorded else None

    throughput: Optional[float] = None
    if has_table(conn, "consolidation_cursor_v7"):
        cur = conn.execute(
            "SELECT last_processed, last_run_us, prev_run_us"
            " FROM consolidation_cursor_v7"
            " WHERE scope_id=? AND generation=?",
            (scope_id, generation),
        ).fetchone()
        if cur is not None:
            last_p, last_t, prev_t = cur
            if last_t and prev_t and last_t > prev_t:
                throughput = round(
                    (last_p or 0) / ((last_t - prev_t) / 1_000_000.0), 3)

    warning = (unconsolidated > BACKLOG_WARN_UNITS) or (
        oldest_age is not None and oldest_age > BACKLOG_WARN_AGE_US)
    return {
        "status": "ok",
        "scope_id": scope_id,
        "generation": generation,
        "unconsolidated_units": unconsolidated,
        "oldest_age_us": oldest_age,
        "throughput_units_per_s": throughput,
        "warning": bool(warning),
        "warn_thresholds": {"units": BACKLOG_WARN_UNITS,
                            "age_us": BACKLOG_WARN_AGE_US},
    }


__all__ = [
    "CONSOLIDATE_VERSION",
    "PRODUCER_T0",
    "FORMULA_STATUS",
    "SLOT_FAMILIES",
    "EVIDENCE_FAMILIES",
    "MERGE_TEXT_SIM",
    "MIN_SUPPORT_UNITS",
    "MIN_SUPPORT_FAMILIES",
    "BACKLOG_WARN_UNITS",
    "BACKLOG_WARN_AGE_US",
    "DDL_CONSOLIDATION_V7",
    "ensure_consolidation_v7",
    "consolidate_scope_v7",
    "backlog_v7",
]
