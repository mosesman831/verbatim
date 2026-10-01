"""Declared retrieval policy table — ``retrieval_policy/v7`` (SPEC_V7 §32.3).

Lane enablement is a declared, versioned *table* keyed by profile and
query intent class — never a code-path conditional (V7-05.02). This
module ships the ``provisional/v7-r0`` snapshot (V7-32.01): the §32.3
lane-weight matrix verbatim as the named arm
:data:`POLICY_ID_INTENT_WEIGHTED`, plus the §04.2 per-lane cost bounds
that ``retrieval/v7/deadline.py`` uses for S2 slice allocation (V7-05.05).
Both constants are pre-search hypotheses, reproducible but not selected.

V75-03.03 (pre-O9 default): V7-10.01's text governs — the *default* lane
weights are flat ``1.0`` on every lane. The §32.3 intent-weighted matrix
(:data:`LANE_WEIGHTS_V1`) is reachable only by selecting the arm tag
``retrieval_policy/v7-intent-weighted`` as the policy document's
``policy_id``; it MUST NOT ship as the default before owner decision O9.

V75-04.03 (weak-lane containment): a policy document may carry
``lane_gates`` — ``{lane: "exclude" | <top_n int> | {"top_n": n}}``,
top-level and under ``profiles[profile]`` — declaring per-profile fusion
gates. The resolved map rides the returned policy as ``lane_gates`` (a
:class:`GatedPolicyV7`); ``retrieval/v7/fusion.py`` applies it, and a
gated lane still answers when it is the only lane with candidates (J13).

V75-04.02 (df-ordered nomination budget): a policy document may carry
``nominate_terms_max`` (positive int, default ``NOMINATE_TERMS_MAX_R0``)
and ``nominate_df_theta`` (``null`` or a finite number in ``(0, 1]``,
default disarmed), top-level and under ``profiles[profile]``. Both are
Q1 formula-search constants; the lexical lane reads them off
``ctx.policy`` and bounds which query terms may nominate (identifiers
first, then content terms by ascending eligible df).

V8 §23 (flag and arm register): a policy document may carry ``params``
— a flat ``{<area>.<knob>: value}`` map (nested ``{area: {knob: value}}``
is flattened one level), top-level and under ``profiles[profile]``.
This is the single overrides channel every §23 flag/arm travels on:
``scheduler.R_post``, ``scheduler.post_pool``, ``graph.gate`` and their
peers resolve off ``ctx.policy.params`` at query time, so no flag needs
its own dataclass field or config path. Defaults live at the read site
(the §23 prior); the map only carries what the document declared.

Layering notes:

- The table declares *enablement* and *weights*. Lane *availability* —
  e.g. an unprovisioned dense tier on ``local_memory`` — is reported at
  run time via ``LaneStatus.UNAVAILABLE``/``SKIPPED`` with a reason
  (V7-04.03); runtime state never rewrites the declared table.
- Intent classes select lane weights and pack shape; they never gate
  eligibility or lane enablement (V7-05.12, V7-05.01), so
  :func:`lanes_for` returns every enabled lane for every intent.
- The weight rows cover the eight §32.3 S2 lanes (``LANES_V1``).
  ``LaneName.EXACT_ID`` and ``LaneName.SOURCE`` have no §32.3 column and
  no §04.2 budget slice; a ``policy_json`` may still enable them — they
  then run at :data:`DEFAULT_LANE_WEIGHT` and are costed for slicing at
  :data:`DEFAULT_LANE_COST_MS` (deadline module).
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    NOMINATE_TERMS_MAX_R0,
    POOLS,
    BudgetClass,
    IntentClass,
    LaneName,
    PoolProfile,
    RetrievalPolicyV7,
)

POLICY_ID_V1 = "retrieval_policy/v7"

#: Named r0 arm tag selecting the §32.3 intent-weighted base table
#: (V75-03.03: the §32.3 matrix is ``retrieval_policy/v7-intent-weighted``,
#: an arm for the Q1 fusion-form decision — never the pre-O9 default).
POLICY_ID_INTENT_WEIGHTED = "retrieval_policy/v7-intent-weighted"

#: Weight used when ``lane_weights[intent][lane]`` is absent
#: (``RetrievalPolicyV7`` contract: default 1.0).
DEFAULT_LANE_WEIGHT = 1.0

#: The eight S2 lanes of §04.2/§32.3, in the spec's declared column order —
#: the full enablement set a policy document may name.
LANES_V1: tuple[LaneName, ...] = (
    LaneName.LEX,
    LaneName.FUZZY,
    LaneName.DENSE,
    LaneName.ENT,
    LaneName.TIME,
    LaneName.GRAPH,
    LaneName.TYPED,
    LaneName.OBS,
)

#: V85-05.01 — the measured ship-set lane tuple (SPEC_V8_5 §05): the four
#: earning lanes are the DEFAULT enablement set (b5 earners
#: +0.095 any@10 / +0.148 MRR; c4 −typed−obs best).  ``ent``, ``graph``,
#: ``typed`` and ``obs`` stay registered (``pipeline._LANE_MODULES``) and
#: remain reachable through a declared ``lanes`` list or a profile
#: overlay — only the *default* changed, and ``LANES_V1`` still names
#: the full §32.3 column order the weight/cost tables are written in.
LANES_V85: tuple[LaneName, ...] = (
    LaneName.LEX,
    LaneName.FUZZY,
    LaneName.DENSE,
    LaneName.TIME,
)


def _w(
    lex: float,
    fuzzy: float,
    dense: float,
    ent: float,
    time: float,
    graph: float,
    typed: float,
    obs: float,
) -> dict[LaneName, float]:
    """One §32.3 weight row in declared column order."""

    return {
        LaneName.LEX: float(lex),
        LaneName.FUZZY: float(fuzzy),
        LaneName.DENSE: float(dense),
        LaneName.ENT: float(ent),
        LaneName.TIME: float(time),
        LaneName.GRAPH: float(graph),
        LaneName.TYPED: float(typed),
        LaneName.OBS: float(obs),
    }


# §32.3 grouped rows — the spec table writes one row per intent *group*;
# each member class resolves to the group's row.
_ROW_TEMPORAL = _w(0.8, 0.2, 0.8, 0.8, 1.5, 0.4, 0.8, 0.3)   # temporal_* / duration
_ROW_CURRENT = _w(0.9, 0.2, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8)    # current_value / history_of
_ROW_MULTIHOP = _w(1.0, 0.3, 1.0, 1.0, 0.5, 1.2, 0.8, 0.6)   # multi_hop / comparison

_TEMPORAL_INTENTS = (
    IntentClass.TEMPORAL_POINT,
    IntentClass.TEMPORAL_RANGE,
    IntentClass.TEMPORAL_ORDER,
    IntentClass.DURATION,
)
_CURRENT_INTENTS = (IntentClass.CURRENT_VALUE, IntentClass.HISTORY_OF)
_MULTIHOP_INTENTS = (IntentClass.MULTI_HOP, IntentClass.COMPARISON)

#: The §32.3 intent-weighted lane matrix verbatim (``provisional/v7-r0``)
#: — the named arm :data:`POLICY_ID_INTENT_WEIGHTED`, NOT the pre-O9
#: default (V75-03.03: the default is flat ``1.0`` per V7-10.01's text).
#: Every mapped intent owns an independent row dict.
#: ``IntentClass.ABSTAIN_LIKELY`` has no §32.3 row: it falls back to
#: :data:`DEFAULT_LANE_WEIGHT` on every enabled lane.
LANE_WEIGHTS_V1: dict[IntentClass, dict[LaneName, float]] = {
    IntentClass.LOOKUP: _w(1.0, 0.3, 1.0, 0.8, 0.3, 0.5, 0.8, 0.5),
    IntentClass.IDENTIFIER: _w(1.5, 0.2, 0.3, 1.0, 0.1, 0.2, 1.0, 0.2),
    **{ic: dict(_ROW_TEMPORAL) for ic in _TEMPORAL_INTENTS},
    IntentClass.COUNT_AGGREGATE: _w(1.0, 0.2, 0.8, 1.0, 0.8, 0.6, 0.8, 0.6),
    **{ic: dict(_ROW_CURRENT) for ic in _CURRENT_INTENTS},
    IntentClass.PREFERENCE: _w(0.7, 0.2, 1.2, 0.6, 0.3, 0.4, 0.8, 1.2),
    **{ic: dict(_ROW_MULTIHOP) for ic in _MULTIHOP_INTENTS},
    IntentClass.OPEN_DOMAIN: _w(0.8, 0.3, 1.4, 0.6, 0.3, 0.8, 0.6, 1.0),
    IntentClass.WHY_CAUSAL: _w(0.9, 0.2, 1.0, 0.8, 0.4, 1.2, 0.6, 0.6),
}

#: Declared per-lane relative cost for deadline slicing (V7-05.05) — the
#: §04.2 per-lane millisecond budgets, used both as the proportionality
#: weights and as each lane's slice cap. ``provisional/v7-r0``: real costs
#: are measured by the stage profiler (§32.17) before selection.
#: Recalibrated against the real LoCoMo dev split (~7k units, beat-it
#: r2): the undifferentiated 2–6ms table under-declared heavy lanes by
#: ~30–50×, so slices expired mid-scan.  Lex/dense are the measured
#: heavy pair (~150ms each post eligibility-batching); the rest are
#: index-probe lanes.
LANE_COSTS_V1: dict[LaneName, float] = {
    LaneName.LEX: 200.0,
    LaneName.FUZZY: 25.0,
    LaneName.DENSE: 200.0,
    LaneName.ENT: 20.0,
    LaneName.TIME: 15.0,
    LaneName.GRAPH: 40.0,
    LaneName.TYPED: 25.0,
    LaneName.OBS: 15.0,
}

#: Slice cost for a ``LaneName`` with no §04.2 budget row (``exact_id``,
#: ``source``): the cheapest declared tier. ``provisional/v7-r0``.
DEFAULT_LANE_COST_MS = 15.0

_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "formula_status",
        "lanes",
        "lane_weights",
        "lane_gates",
        "nominate_terms_max",
        "nominate_df_theta",
        "params",
        "profiles",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "lanes",
        "lane_weights",
        "lane_gates",
        "nominate_terms_max",
        "nominate_df_theta",
        "params",
    }
)


@dataclass(frozen=True)
class GatedPolicyV7(RetrievalPolicyV7):
    """``RetrievalPolicyV7`` carrying resolved V75-04.03 weak-lane gates.

    ``lane_gates[lane]`` is the highest lane rank allowed to contribute to
    fusion: ``0`` excludes the lane's candidates from fusion entirely,
    ``N`` admits only its top-N ranks. Gates are *containment*, not
    enablement — the lane still runs, still reports honest coverage, and
    fusion bypasses every gate when no ungated lane produced candidates
    (J13: a gated lane still answers standalone). An empty map means
    nothing is gated (the default).

    ``RetrievalPolicyV7`` is a frozen contract type; the gate map rides
    this subclass so ``isinstance(policy, RetrievalPolicyV7)`` holds and
    ``dataclasses.replace`` (ablation) preserves the declared gates.

    ``params`` is the V8 §23 flag/arm register channel: a flat
    ``{<area>.<knob>: value}`` map declared by the policy document.
    Query-time consumers resolve their knob with :func:`policy_param`
    (or ``policy.params.get``) against the §23 prior default — an absent
    key means the prior, never a fabricated override.
    """

    lane_gates: dict[LaneName, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


def _fail(message: str) -> None:
    raise VerbatimError(ErrorCode.VALIDATION, message)


def _nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"policy {field} must be a non-empty string, got {value!r}")
    return value.strip()


def _as_lane(value: Any, *, field: str = "lane") -> LaneName:
    if isinstance(value, LaneName):
        return value
    try:
        return LaneName(str(value))
    except ValueError:
        member = _extension_lane(str(value))
        if member is None:
            _fail(f"unknown {field}: {value!r}")
        return member


# ---------------------------------------------------------------------------
# Extension lanes — post-freeze lane names (separate region; never weights)
# ---------------------------------------------------------------------------
#
# ``LaneName`` is a contract-frozen enum (docs/v7_contracts.md, wave A), so
# a lane added after the freeze cannot grow a member in types_v7.py.  Such
# a lane mints a ``LaneName`` *instance* carrying its value inside its own
# module — a str-enum instance that compares and hashes as the value and
# passes every ``LaneName(x)`` coercion site unchanged — and registers the
# *value* here so a declared policy table (``lanes``/``lane_weights`` JSON)
# or ``lanes_disabled`` ablation can name it.  Resolution is lazy: the
# module is imported only when its lane name actually appears.
#
# "scope" — the V75-04.01 speaker/entity scoped candidate-seed lane
# (``retrieval/v7/scope.py``), the Q1 scope-form arm.  Off by default (not
# in ``LANES_V1``); enabled only by naming ``"scope"`` in the declared
# lanes table, exactly like the optional ``exact_id``/``source`` lanes.
_EXTENSION_LANES: dict[str, str] = {
    "scope": "verbatim.retrieval.v7.scope",
}


def _extension_lane(value: str) -> Optional[LaneName]:
    """Resolve an extension lane name to its module-minted LaneName.

    ``None`` when the name is not a declared extension lane or its module
    cannot be imported — the caller then fails validation honestly.
    """
    modname = _EXTENSION_LANES.get(value)
    if modname is None:
        return None
    try:
        mod = importlib.import_module(modname)
    except Exception:  # noqa: BLE001 — absent module → unknown name
        return None
    member = getattr(mod, "LANE_ENUM", None)
    if isinstance(member, LaneName) and getattr(member, "value", None) == value:
        return member
    return None


def _as_intent(value: Any, *, field: str = "intent") -> IntentClass:
    if isinstance(value, IntentClass):
        return value
    try:
        return IntentClass(str(value))
    except ValueError:
        _fail(f"unknown {field}: {value!r}")


def _maybe_intent(value: Any) -> Optional[IntentClass]:
    if isinstance(value, IntentClass):
        return value
    try:
        return IntentClass(str(value))
    except (ValueError, TypeError):
        return None


def _parse_lanes(value: Any) -> tuple[LaneName, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(f"policy lanes must be a list of lane names, got {value!r}")
    out: list[LaneName] = []
    for item in value:
        lane = _as_lane(item)
        if lane in out:
            _fail(f"duplicate lane in policy table: {lane.value!r}")
        out.append(lane)
    if not out:
        _fail("policy table enables no lanes")
    return tuple(out)


def _parse_weight(value: Any, *, intent: str, lane: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"lane weight {intent!r}/{lane!r} must be a number, got {value!r}")
    w = float(value)
    if not math.isfinite(w) or w < 0.0:
        _fail(
            f"lane weight {intent!r}/{lane!r} must be finite and >= 0, "
            f"got {value!r}"
        )
    return w


def _parse_weight_rows(value: Any) -> dict[IntentClass, dict[LaneName, float]]:
    if not isinstance(value, dict):
        _fail("policy lane_weights must be an object {intent: {lane: weight}}")
    out: dict[IntentClass, dict[LaneName, float]] = {}
    for raw_intent, row in value.items():
        intent = _as_intent(raw_intent, field="lane_weights intent")
        if not isinstance(row, dict):
            _fail(f"lane_weights row for {raw_intent!r} must be an object")
        bucket = out.setdefault(intent, {})
        for raw_lane, raw_w in row.items():
            lane = _as_lane(raw_lane, field="lane_weights lane")
            bucket[lane] = _parse_weight(
                raw_w, intent=intent.value, lane=lane.value
            )
    return out


def _parse_gate(value: Any, *, lane: str) -> int:
    """One ``lane_gates`` entry → the max contributing rank (0 = exclude).

    Accepted forms: ``"exclude"`` (→ 0), a non-negative int (top-N cap;
    0 is identical to ``"exclude"``), or ``{"top_n": n}``.
    """

    if isinstance(value, str):
        if value.strip() == "exclude":
            return 0
        _fail(f"lane_gates[{lane!r}] must be \"exclude\", a non-negative "
              f"int, or {{\"top_n\": n}} — got {value!r}")
    if isinstance(value, dict):
        unknown = set(value) - {"top_n"}
        if unknown or "top_n" not in value:
            _fail(
                f"lane_gates[{lane!r}] object form accepts only "
                f"\"top_n\", got {sorted(value)!r}"
            )
        value = value["top_n"]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"lane_gates[{lane!r}] must be an int rank cap, got {value!r}")
    if value < 0:
        _fail(f"lane_gates[{lane!r}] must be >= 0, got {value!r}")
    return int(value)


def _parse_gates(value: Any) -> dict[LaneName, int]:
    if not isinstance(value, dict):
        _fail(f"policy lane_gates must be an object {{lane: gate}}, got {value!r}")
    out: dict[LaneName, int] = {}
    for raw_lane, raw_gate in value.items():
        lane = _as_lane(raw_lane, field="lane_gates lane")
        out[lane] = _parse_gate(raw_gate, lane=lane.value)
    return out


def _merge_weights(
    base: dict[IntentClass, dict[LaneName, float]],
    over: dict[IntentClass, dict[LaneName, float]],
) -> None:
    for intent, row in over.items():
        base.setdefault(intent, {}).update(row)


# --- V75-04.02 lexical-nomination knobs (Q1 constants) -------------------
# Separate knob region: these govern *which query terms may nominate* in
# the lexical lane (``retrieval/v7/lexical.py``), not fusion weights.  The
# r0 budget default is the contract's ``NOMINATE_TERMS_MAX_R0`` — the JSON
# channel only overrides it.


def _parse_nominate_terms_max(value: Any) -> int:
    """``nominate_terms_max`` — the nomination budget, a positive int.
    Strict: a count knob rejects bools/floats/strings outright."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"nominate_terms_max must be a positive int, got {value!r}")
    if value < 1:
        _fail(f"nominate_terms_max must be >= 1, got {value!r}")
    return int(value)


def _parse_nominate_df_theta(value: Any) -> Optional[float]:
    """``nominate_df_theta`` — the optional ``df > θ·N_E`` nomination
    exclusion arm.  ``null``/absent disarms it (the r0 default); any other
    value must be a finite number in (0, 1]."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"nominate_df_theta must be null or a number, got {value!r}")
    theta = float(value)
    if not math.isfinite(theta) or not 0.0 < theta <= 1.0:
        _fail(
            f"nominate_df_theta must be finite and in (0, 1], "
            f"got {value!r}"
        )
    return theta


# --- V8 §23 flag/arm register channel -----------------------------------
# ``params`` carries the dotted flag names verbatim (``scheduler.R_post``,
# ``graph.gate`` …); a one-level nested form (``{"graph": {"gate": true}}``)
# flattens to the same dotted key so areas group naturally in documents.
# Values are scalars or flat scalar lists — arms are constants, never
# structures.  Consumers resolve domain rules (range, type) at the read
# site; this parser only guarantees a clean, JSON-able map.


def _param_leaf_ok(value: Any, *, key: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        x is None or isinstance(x, (bool, int, float, str)) for x in value
    ):
        return list(value)
    _fail(
        f"policy params[{key!r}] must be a scalar or a flat list of "
        f"scalars, got {value!r}"
    )


def _parse_params(value: Any) -> dict[str, Any]:
    """``params`` — the V8 §23 flag/arm map.  Flat dotted keys verbatim;
    one level of nesting flattens to ``<area>.<knob>``."""
    if not isinstance(value, dict):
        _fail(f"policy params must be an object {{flag: value}}, got {value!r}")
    out: dict[str, Any] = {}
    for raw_key, raw_val in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            _fail(f"policy params keys must be non-empty strings, got {raw_key!r}")
        key = raw_key.strip()
        if isinstance(raw_val, dict):
            for sub_key, sub_val in raw_val.items():
                if not isinstance(sub_key, str) or not sub_key.strip():
                    _fail(
                        f"policy params[{key!r}] nested keys must be "
                        f"non-empty strings, got {sub_key!r}"
                    )
                flat = f"{key}.{sub_key.strip()}"
                out[flat] = _param_leaf_ok(sub_val, key=flat)
            continue
        out[key] = _param_leaf_ok(raw_val, key=key)
    return out


#: The V8.5 §05 params register — the declared shipped defaults for the
#: §23-style ``params`` channel, keyed by the dotted flag name exactly as
#: a policy document declares it.  This is a *declaration* surface: a
#: resolved ``params`` map still carries only what the document wrote
#: (``policy_view`` prints the applied overrides, never the priors), and
#: read sites keep their own §23 priors as call-site defaults.  When a
#: consumer calls :func:`policy_param` *without* an explicit default the
#: register supplies the prior — the register is the single declaration
#: of what shipped (V85-05.02/05.03/05.06/05.07; forensic ``PolicyPatch``
#: arms toggle the same keys through ``params``).
PARAM_DEFAULTS: dict[str, Any] = {
    # V85-05.02 — lexical co-occurrence nomination: a candidate must
    # match >= ``lexical.cooc_min`` nominated terms (read: lexical
    # ``_cooc_arms`` → ``_select_nomination``); an empty filtered set
    # falls back to the union and says so in ``coverage_lexical.cooc``.
    "lexical.nom_cooc": True,
    "lexical.cooc_min": 2,
    # V85-05.03 — rare-term rescue: a term whose measured eligible df is
    # below this threshold skips the co-occurrence requirement.
    "lexical.rare_df": 4,
    # V85-05.06 — ``as_of`` anchors relative-expression window resolution
    # and the verdict only; it never re-anchors recency or fallback
    # ordering.  (Declaration for the temporal reader landing with the
    # as_of work — SPEC_V8_5 §05.)
    "temporal.as_of_scope": "window",
    # V85-05.07 — the scheduler's shipped shape.  ``deadline.py`` runs
    # the S2a/S2b two-phase split today; the declaration pins that prior
    # so the paired single-phase arm can toggle it.
    "scheduler.two_phase": True,
    # SPEC_V8_5 §06 don't-ship — ``dense.N_d`` guaranteed fused-pool
    # slots measured 0/137 unique gold rescues; ships at 0.  The
    # ``dense_N5``/``dense_N20`` ablation arms sweep the on-direction.
    "dense.N_d": 0,
}

#: Sentinel marking "caller gave no explicit default" — distinguishes an
#: omitted default (register applies) from an explicit ``None``.
_PARAM_UNSET = object()


def policy_param(
    policy: RetrievalPolicyV7, name: str, default: Any = _PARAM_UNSET
) -> Any:
    """Resolve one V8 §23 flag/arm off the policy object.

    ``policy.params`` exists on :class:`GatedPolicyV7` (every
    ``load_policy`` product); a bare ``RetrievalPolicyV7`` — hand-built
    fixtures — has no map and yields ``default``.  An explicit
    ``default`` is the caller's §23 prior and always wins; when it is
    omitted the V8.5 declaration register :data:`PARAM_DEFAULTS`
    supplies the prior (``None`` when the flag is undeclared).  An
    absent key is never an override.
    """

    if default is _PARAM_UNSET:
        default = PARAM_DEFAULTS.get(name)
    params = getattr(policy, "params", None)
    if not isinstance(params, dict):
        return default
    return params.get(name, default)


def load_policy(
    profile: str, policy_json: Optional[dict] = None
) -> RetrievalPolicyV7:
    """Resolve the declared policy table for ``profile`` (V7-05.02).

    ``policy_json``, when given, is the parsed table document::

        {
          "policy_id": "retrieval_policy/v7",      # default POLICY_ID_V1
          "formula_status": "provisional/v7-r0",   # default r0 tag
          "lanes": ["lex", "fuzzy", ...],          # default: LANES_V85 ship set
          "lane_weights": {"<intent>": {"<lane>": <weight>}},
          "lane_gates": {"<lane>": "exclude" | <top_n> | {"top_n": n}},
          "nominate_terms_max": 32,                # V75-04.02 budget (r0)
          "nominate_df_theta": null,               # V75-04.02 θ arm (off)
          "params": {"scheduler.R_post": 60.0,     # V8 §23 flag/arm map
                     "graph.gate": true},
          "profiles": {"<profile>": {"lanes": [...], "lane_weights": {...},
                                    "lane_gates": {...},
                                    "nominate_terms_max": ...,
                                    "params": {...}}}
        }

    Resolution order: built-in r0 defaults → top-level keys → the
    ``profiles[profile]`` overlay. ``lane_weights`` entries merge per
    (intent, lane) cell over the base rows; ``lane_gates`` entries merge
    per lane; ``params`` entries merge per dotted key; the V75-04.02
    nomination knobs resolve as last-write-wins scalars (top level, then
    the profile overlay). Weight entries naming valid-but-disabled lanes
    are validated but inert — lanes absent from ``lanes`` never run.

    Base weight rows (V75-03.03): the default base is FLAT — every
    (intent, lane) cell starts at :data:`DEFAULT_LANE_WEIGHT`, per
    V7-10.01's "all 1.0 unless tuned on the dev split". Only when the
    document's ``policy_id`` is :data:`POLICY_ID_INTENT_WEIGHTED`
    (``retrieval_policy/v7-intent-weighted``) does the base become the
    §32.3 matrix :data:`LANE_WEIGHTS_V1` — that is the named Q1 arm, and
    ``policy_id`` then honestly reports which table was applied.

    Every returned policy is a :class:`GatedPolicyV7` stamped
    ``formula_status`` — ``provisional/v7-r0`` for the r0 table
    (V7-32.01).
    """

    if not isinstance(profile, str) or not profile.strip():
        _fail(f"profile must be a non-empty string, got {profile!r}")
    profile = profile.strip()

    policy_id = POLICY_ID_V1
    formula_status = FORMULA_STATUS_PROVISIONAL
    lanes: tuple[LaneName, ...] = LANES_V85   # V85-05.01 ship set
    overrides: dict[IntentClass, dict[LaneName, float]] = {}
    # V75-04.02 nomination knobs — r0 defaults unless the document sets them.
    nominate_terms_max = NOMINATE_TERMS_MAX_R0
    nominate_df_theta: Optional[float] = None
    lane_gates: dict[LaneName, int] = {}
    # V8 §23 flag/arm register — ``{<area>.<knob>: value}``, top level then
    # the profile overlay merges per key (last write wins).
    params: dict[str, Any] = {}

    if policy_json is not None:
        if not isinstance(policy_json, dict):
            _fail(f"policy_json must be an object, got {policy_json!r}")
        unknown = set(policy_json) - _POLICY_KEYS
        if unknown:
            _fail(f"unknown policy_json keys: {sorted(unknown)!r}")

        if "policy_id" in policy_json:
            policy_id = _nonempty_str(policy_json["policy_id"], "policy_id")
        if "formula_status" in policy_json:
            formula_status = _nonempty_str(
                policy_json["formula_status"], "formula_status"
            )
        if "lanes" in policy_json:
            lanes = _parse_lanes(policy_json["lanes"])
        if "lane_weights" in policy_json:
            _merge_weights(
                overrides, _parse_weight_rows(policy_json["lane_weights"])
            )
        if "lane_gates" in policy_json:
            lane_gates.update(_parse_gates(policy_json["lane_gates"]))
        if "nominate_terms_max" in policy_json:
            nominate_terms_max = _parse_nominate_terms_max(
                policy_json["nominate_terms_max"]
            )
        if "nominate_df_theta" in policy_json:
            nominate_df_theta = _parse_nominate_df_theta(
                policy_json["nominate_df_theta"]
            )
        if "params" in policy_json:
            params.update(_parse_params(policy_json["params"]))

        profiles = policy_json.get("profiles")
        if profiles is not None:
            if not isinstance(profiles, dict):
                _fail("policy profiles must be an object {profile: table}")
            sub = profiles.get(profile)
            if sub is not None:
                if not isinstance(sub, dict):
                    _fail(f"policy profiles[{profile!r}] must be an object")
                unknown_sub = set(sub) - _PROFILE_KEYS
                if unknown_sub:
                    _fail(
                        f"unknown profiles[{profile!r}] keys: "
                        f"{sorted(unknown_sub)!r}"
                    )
                if "lanes" in sub:
                    lanes = _parse_lanes(sub["lanes"])
                if "lane_weights" in sub:
                    _merge_weights(
                        overrides, _parse_weight_rows(sub["lane_weights"])
                    )
                if "lane_gates" in sub:
                    lane_gates.update(_parse_gates(sub["lane_gates"]))
                if "nominate_terms_max" in sub:
                    nominate_terms_max = _parse_nominate_terms_max(
                        sub["nominate_terms_max"]
                    )
                if "nominate_df_theta" in sub:
                    nominate_df_theta = _parse_nominate_df_theta(
                        sub["nominate_df_theta"]
                    )
                if "params" in sub:
                    params.update(_parse_params(sub["params"]))

    # Materialize the resolved table: every intent class × every enabled
    # lane, so coverage.policy prints the effective table exactly as
    # applied (V7-05.02, V7-31.07). The base is flat unless the named
    # intent-weighted arm tag was selected (V75-03.03).
    base_rows = (
        LANE_WEIGHTS_V1 if policy_id == POLICY_ID_INTENT_WEIGHTED else {}
    )
    lane_weights: dict[IntentClass, dict[LaneName, float]] = {}
    for intent in IntentClass:
        default_row = base_rows.get(intent, {})
        over_row = overrides.get(intent, {})
        lane_weights[intent] = {
            lane: over_row.get(
                lane, default_row.get(lane, DEFAULT_LANE_WEIGHT)
            )
            for lane in lanes
        }

    return GatedPolicyV7(
        policy_id=policy_id,
        profile=profile,
        lanes=tuple(lanes),
        lane_weights=lane_weights,
        formula_status=formula_status,
        lane_gates=lane_gates,
        nominate_terms_max=nominate_terms_max,
        nominate_df_theta=nominate_df_theta,
        params=params,
    )


def pool_for(budget: Any) -> PoolProfile:
    """The §32.3 pool profile for a budget class (V7-05.07)."""

    if isinstance(budget, PoolProfile):
        return budget
    try:
        cls = (
            budget
            if isinstance(budget, BudgetClass)
            else BudgetClass(str(budget))
        )
    except ValueError:
        _fail(f"unknown budget class: {budget!r}")
    return POOLS[cls]


def lanes_for(
    intent: Any, policy: RetrievalPolicyV7
) -> tuple[LaneName, ...]:
    """Lanes enabled for this intent under ``policy``.

    The r0 table enables lanes per profile, not per intent — every enabled
    lane runs on every query (V7-05.01) and the intent class modulates
    *weights*, never enablement or eligibility (V7-05.12). ``intent`` is
    accepted so callers code against the intent-keyed contract; an
    unparseable intent still returns the declared set.
    """

    return tuple(policy.lanes)


def weights_for(intent: Any, policy: RetrievalPolicyV7) -> dict[LaneName, float]:
    """Effective fusion weights for ``intent`` over the enabled lanes.

    Intents with no declared row — including ``abstain_likely`` and any
    intent string that does not parse — resolve to
    :data:`DEFAULT_LANE_WEIGHT` on every enabled lane.
    """

    ic = _maybe_intent(intent)
    row = policy.lane_weights.get(ic) if ic is not None else None
    if row is None:
        return {lane: DEFAULT_LANE_WEIGHT for lane in policy.lanes}
    return {lane: row.get(lane, DEFAULT_LANE_WEIGHT) for lane in policy.lanes}


def ablation_lanes(
    policy: RetrievalPolicyV7, disabled: Iterable[Any]
) -> RetrievalPolicyV7:
    """``lanes_disabled=[...]`` ablation switch (V7-05.09).

    Returns a new policy with the named lanes removed from ``lanes`` and
    every weight row filtered to the survivors — the printed table then
    shows exactly the lanes the leave-one-lane-out arm ran. Unknown lane
    *names* fail validation (a typo must not silently no-op an arm);
    disabling a valid lane the policy does not enable is a no-op.
    """

    drop: set[LaneName] = set()
    for item in disabled:
        drop.add(_as_lane(item, field="disabled lane"))
    lanes = tuple(lane for lane in policy.lanes if lane not in drop)
    lane_weights = {
        intent: {lane: w for lane, w in row.items() if lane in lanes}
        for intent, row in policy.lane_weights.items()
    }
    # V75-04.03: gates on surviving lanes carry over; gates naming an
    # ablated lane are dropped with it (they were inert anyway).
    if hasattr(policy, "lane_gates"):
        gates = {
            lane: g
            for lane, g in getattr(policy, "lane_gates", {}).items()
            if lane in lanes
        }
        return replace(
            policy, lanes=lanes, lane_weights=lane_weights, lane_gates=gates
        )
    return replace(policy, lanes=lanes, lane_weights=lane_weights)


def policy_view(policy: RetrievalPolicyV7) -> dict[str, Any]:
    """The JSON-able ``coverage.policy`` block (V7-05.02): the declared
    table exactly as applied — id (the applied weight-table arm tag,
    V75-03.03), profile, provisional tag, enabled lanes in table order,
    the full weight matrix, the V75-04.02 nomination knobs
    (``nominate_terms_max``/``nominate_df_theta``), and any declared
    ``lane_gates`` (V75-04.03; gate values are the max contributing
    rank — 0 means excluded)."""

    view = {
        "policy_id": policy.policy_id,
        "profile": policy.profile,
        "formula_status": policy.formula_status,
        "lanes": [lane.value for lane in policy.lanes],
        "lane_weights": {
            intent.value: {lane.value: w for lane, w in row.items()}
            for intent, row in policy.lane_weights.items()
        },
        # V75-04.02 — the lexical nomination knobs exactly as applied.
        "nominate_terms_max": policy.nominate_terms_max,
        "nominate_df_theta": policy.nominate_df_theta,
    }
    gates = getattr(policy, "lane_gates", None)
    if gates:
        view["lane_gates"] = {
            getattr(lane, "value", str(lane)): int(gate)
            for lane, gate in gates.items()
        }
    # V8 §23 — the declared flag/arm overrides exactly as applied (absent
    # map = every knob sits at its §23 prior; nothing to disclose).
    params = getattr(policy, "params", None)
    if params:
        view["params"] = dict(sorted(params.items()))
    return view


__all__ = [
    "DEFAULT_LANE_COST_MS",
    "DEFAULT_LANE_WEIGHT",
    "GatedPolicyV7",
    "LANES_V1",
    "LANES_V85",
    "LANE_COSTS_V1",
    "LANE_WEIGHTS_V1",
    "PARAM_DEFAULTS",
    "POLICY_ID_INTENT_WEIGHTED",
    "POLICY_ID_V1",
    "ablation_lanes",
    "lanes_for",
    "load_policy",
    "policy_param",
    "policy_view",
    "pool_for",
    "weights_for",
]
