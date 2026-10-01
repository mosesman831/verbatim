"""Owned DolphinBench-like agent-action twin generator (V7-20.06, V7-24.10).

Seeded, fully-invented corpus of agent/workspace message histories: tool-call /
tool-result turn sequences, standing rules planted mid-stream (§32.13
``rules_detect/v1`` imperative/normative forms), runbook documents for
procedure-recall tasks, and recorded agent actions used by compliance checks.

Task kinds (``CorpusTask.kind``-style ``kind`` field on each task dict):

* ``implicit_rule_recall`` — the probe text never names the remembered rule
  (DolphinBench's "post the update to the appropriate channel"); gold is the
  governing rule unit. Sized ≥ 200 by default per V7-20.06, and each task
  carries ``naive_action`` / ``expected_action`` so the harness can certify
  pass-with-history / fail-without (V7-24.10 solvability rule adopted).
* ``procedure_recall`` — "how do I deploy X" probes whose gold is the full
  ordered runbook unit set.
* ``rule_compliance`` — a later ``agent_action`` unit either respected or
  violated an earlier user-stated rule; gold is BOTH units and
  ``expect.compliant`` records the ground truth.
* ``action_fact_lookup`` — a fact stated inside an agent action ("I booked
  the 09:15 ferry to Harwick for March 3") retrievable as ``agent_stated`` /
  ``agent_action`` evidence (H60).

Every rule is keyed by a unique (artifact, channel, service) combination so
gold sets stay unambiguous when hundreds of rules share one history. Filler
text deliberately avoids every §32.13 cue lexeme (``always``, ``never``,
``unless``, ``until``, ``whenever``, ``only``, ``policy``, ``rule``,
``remember``, ``make sure``, ``by default``, ``from now on``,
``going forward``, ``don't``/``do not``) so planted rules — not filler — are
the only normative statements in the corpus.

Determinism: identical ``(seed, counts)`` produce byte-identical corpora
(``corpus_digest``). Stdlib only; no network; no benchmark text.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

GENERATOR_ID = "twins_actions/v1"
CONSTANTS_TAG = "provisional/v7-r0"
CORPUS_NAME = "owned_actions"
DEFAULT_SEED = 20260922

#: 2024-01-01T00:00:00Z — deterministic timeline origin. Sessions are spread
#: across ~24 months so rules land "years before" the task-time probes
#: (V7-20.06 "rule stated once, task years later without cue").
BASE_US = 1_704_067_200_000_000
DAY_US = 86_400_000_000
HOUR_US = 3_600_000_000

# ---------------------------------------------------------------------------
# invented entity pools (license-free; deliberately unlike public-benchmark
# persona names so no twin item can be mistaken for benchmark content)
# ---------------------------------------------------------------------------

PEOPLE: Tuple[str, ...] = (
    "Arden", "Bexley", "Calloway", "Darby", "Ellery", "Finch", "Greeley",
    "Hollis", "Iver", "Junia", "Kestrel", "Larkin", "Marlow", "Niven",
    "Ossian", "Pryce", "Quill", "Rennick", "Sorrel", "Tamsin",
)

CHANNELS: Tuple[str, ...] = (
    "#ship-notes", "#team-eng", "#ops-alerts", "#design-sync", "#incidents",
    "#growth-metrics", "#qa-reports", "#cust-feedback", "#infra-log",
    "#weekly-roundup", "#mobile-team", "#data-quality",
)

DEFAULT_CHANNEL = "#general"

SERVICES: Tuple[str, ...] = (
    "osprey-api", "tern-worker", "magpie-web", "puffin-ml", "wren-db",
    "lark-gateway", "heron-queue", "ibis-search", "crane-billing",
    "swift-auth", "robin-mobile", "falcon-edge",
)

TOOLS: Tuple[str, ...] = (
    "deploy_status", "fetch_metrics", "search_tickets", "get_calendar",
    "read_channel", "run_report", "check_quota", "list_deploys",
)

#: Artifacts that get posted/routed/filed — the noun a rule scopes to.
ARTIFACTS: Tuple[str, ...] = (
    "changelog digest", "release notes", "incident summary", "deploy report",
    "metrics snapshot", "design notes", "sync minutes", "build report",
    "uptime bulletin", "retro summary", "capacity review", "cost report",
    "support digest", "perf profile", "rollout plan", "status brief",
)

WORK_NOUNS: Tuple[str, ...] = (
    "dashboard", "runbook", "playbook", "checklist", "backlog", "handoff",
    "inventory", "roadmap", "postmortem", "scorecard",
)

FILE_PATHS: Tuple[str, ...] = (
    "src/payments/**", "src/search/**", "docs/runbooks/**", "infra/tf/**",
    "web/client/**", "libs/proto/**", "scripts/release/**", "db/migrations/**",
)

WEEKDAYS: Tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
)


def corpus_digest(corpus: Dict[str, Any]) -> str:
    """Stable fingerprint over corpus content (manifest identity)."""
    canon = {
        "name": corpus.get("name"),
        "generator": corpus.get("generator"),
        "seed": corpus.get("seed"),
        "units": corpus.get("units"),
        "tasks": corpus.get("tasks"),
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iso_day(us: int) -> str:
    return _dt.datetime.fromtimestamp(
        us / 1_000_000, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# rule construction (§32.13 forms)
# ---------------------------------------------------------------------------
#
# Each builder returns ``{text, pattern, action, condition, entities}``.
# ``condition`` feeds ``valid_until_expr``/``condition`` annotations; the rule
# text always carries at least one §32.13 cue lexeme so the T0 detector has a
# true positive to find.

def _rule_route_artifact(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": f"From now on, always post the {a} to {ch}.",
        "pattern": "from_now_on_always",
        "action": f"post the {a} to {ch}",
        "condition": None,
        "entities": [a, ch],
    }


def _rule_route_until(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Post the {a} to {ch} only until the {svc} deploy is green."
        ),
        "pattern": "only_until",
        "action": f"post the {a} to {ch}",
        "condition": f"until the {svc} deploy is green",
        "entities": [a, ch, svc],
    }


def _rule_never_unless(a: str, ch: str, svc: str) -> Dict[str, Any]:
    other = "#ops-alerts" if ch != "#ops-alerts" else "#ship-notes"
    return {
        "text": (
            f"Never post the {a} to {other} unless the {svc} "
            "incident is still open."
        ),
        "pattern": "never_unless",
        "action": f"keep the {a} out of {other}",
        "condition": f"unless the {svc} incident is still open",
        "entities": [a, other, svc],
    }


def _rule_dont_unless(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Don't file the {a} under {svc} unless Arden signs off."
        ),
        "pattern": "dont_unless",
        "action": f"file the {a} under {svc}",
        "condition": "unless Arden signs off",
        "entities": [a, svc, "Arden"],
    }


def _rule_by_default(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"By default, send the {a} to {ch} when the {svc} "
            "pipeline finishes."
        ),
        "pattern": "by_default_when",
        "action": f"send the {a} to {ch}",
        "condition": f"when the {svc} pipeline finishes",
        "entities": [a, ch, svc],
    }


def _rule_make_sure(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Make sure to attach the {a} before tagging a {svc} "
            "release."
        ),
        "pattern": "make_sure_before",
        "action": f"attach the {a}",
        "condition": f"before tagging a {svc} release",
        "entities": [a, svc],
    }


def _rule_whenever(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Whenever the {svc} build turns red, drop the {a} in {ch}."
        ),
        "pattern": "whenever",
        "action": f"drop the {a} in {ch}",
        "condition": f"whenever the {svc} build turns red",
        "entities": [a, ch, svc],
    }


def _rule_going_forward(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Going forward, post the {a} to {ch} instead of "
            f"{DEFAULT_CHANNEL}."
        ),
        "pattern": "going_forward",
        "action": f"post the {a} to {ch}",
        "condition": None,
        "entities": [a, ch],
    }


def _rule_as_a_rule(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"As a rule, run fetch_metrics on {svc} before sharing "
            f"the {a}."
        ),
        "pattern": "as_a_rule",
        "action": f"run fetch_metrics on {svc}",
        "condition": f"before sharing the {a}",
        "entities": [a, svc, "fetch_metrics"],
    }


def _rule_policy(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Team policy: the {a} goes to {ch} only after two "
            "approvals."
        ),
        "pattern": "policy_only_after",
        "action": f"send the {a} to {ch}",
        "condition": "only after two approvals",
        "entities": [a, ch],
    }


def _rule_remember_to(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Remember to cc Bexley whenever the {a} mentions {svc}."
        ),
        "pattern": "remember_to_whenever",
        "action": "cc Bexley",
        "condition": f"whenever the {a} mentions {svc}",
        "entities": [a, svc, "Bexley"],
    }


def _rule_only_after(a: str, ch: str, svc: str) -> Dict[str, Any]:
    return {
        "text": (
            f"Only merge the {svc} change after the {a} is published."
        ),
        "pattern": "only_after",
        "action": f"merge the {svc} change",
        "condition": f"after the {a} is published",
        "entities": [a, svc],
    }


_RULE_BUILDERS: Tuple[Any, ...] = (
    _rule_route_artifact,
    _rule_route_until,
    _rule_never_unless,
    _rule_dont_unless,
    _rule_by_default,
    _rule_make_sure,
    _rule_whenever,
    _rule_going_forward,
    _rule_as_a_rule,
    _rule_policy,
    _rule_remember_to,
    _rule_only_after,
)

#: §32.13 cue lexemes — asserted on every planted rule in tests and used to
#: keep filler text cue-free.
RULE_CUES: Tuple[str, ...] = (
    "always", "never", "don't", "do not", "only", "make sure",
    "remember to", "from now on", "going forward", "whenever", "until",
    "unless", "by default", "as a rule", "policy",
)

#: Filler small-talk templates — verified cue-free in tests. ``{p}`` person,
#: ``{s}`` service, ``{w}`` work noun, ``{d}`` weekday.
_FILLER_TEMPLATES: Tuple[str, ...] = (
    "{p} moved the {s} review to {d} afternoon.",
    "The {s} rollout finished ahead of schedule.",
    "Can someone glance at the {w} draft before {d}?",
    "{p} is out {d}; sync shifts to the afternoon.",
    "The {s} dashboard shows a quiet week.",
    "Heads up: {s} maintenance window moved to {d} evening.",
    "{p} asked for a second pair of eyes on the {w}.",
    "Lunch-and-learn on {d} covers the {s} internals.",
    "The {w} got a fresh coat of paint — feedback welcome.",
    "{p} wrapped the {s} cleanup; diff is small.",
    "Quiet morning — {s} graphs all look flat.",
    "Reminder-free note: the {w} lives in the shared drive.",
)

_TOOL_CALL_TEMPLATES: Tuple[str, ...] = (
    "Tool call {tool}: {{\"service\": \"{s}\"}}",
    "Tool call {tool}: {{\"channel\": \"{c}\"}}",
    "Tool call {tool}: {{\"service\": \"{s}\", \"window\": \"7d\"}}",
)

_TOOL_RESULT_TEMPLATES: Tuple[str, ...] = (
    "Tool result {tool}: {{\"status\": \"ok\", \"rows\": {n}}}",
    "Tool result {tool}: {{\"status\": \"green\", \"service\": \"{s}\"}}",
    "Tool result {tool}: {{\"found\": {n}, \"service\": \"{s}\"}}",
)

_TASK_TEXT_TEMPLATES: Tuple[str, ...] = (
    "Post the {a}.",
    "Share the {a} with the team.",
    "Put up this week's {a}.",
    "Send out the {a}.",
    "Publish the {a} where it goes.",
)

_COMPLIANCE_TASK_TEMPLATES: Tuple[str, ...] = (
    "On {date} you posted the {a}. Did that follow my standing instructions?",
    "You handled the {a} on {date}. Was that consistent with what I asked earlier?",
    "Check the {a} post from {date} — did it respect my standing instructions?",
)

_PROCEDURE_STEPS: Tuple[str, ...] = (
    "Run `make build SERVICE={s}` and wait for the artifact hash.",
    "Deploy the canary with `deployctl canary {s} --pct 5`.",
    "Watch the {s} error budget for ten minutes.",
    "Promote with `deployctl promote {s}` once the canary is clean.",
    "Post the deploy report where the team expects it.",
)

_ACTION_FACTS: Tuple[Tuple[str, str], ...] = (
    (
        "I booked the 09:15 ferry to Harwick for March 3.",
        "When is my ferry to Harwick?",
    ),
    (
        "I scheduled the {s} failover drill for the second Tuesday.",
        "When is the {s} failover drill?",
    ),
    (
        "I renewed the {s} certificate; it expires in 400 days.",
        "When does the {s} certificate expire?",
    ),
    (
        "I reserved the north conference room for the {s} review.",
        "Which room is the {s} review in?",
    ),
    (
        "I ordered forty sensor kits for the {s} testbed.",
        "How many sensor kits were ordered for the {s} testbed?",
    ),
    (
        "I moved the {s} maintenance window to Sunday 02:00.",
        "When is the {s} maintenance window?",
    ),
    (
        "I filed ticket PLAT-{n} for the {s} latency spike.",
        "Which ticket covers the {s} latency spike?",
    ),
    (
        "I set the {s} dashboard refresh to five minutes.",
        "How often does the {s} dashboard refresh?",
    ),
)


class _Builder:
    """Accumulates units/sessions/tasks in chronological order."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.units: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self._seq = 0
        self._sess_seq = 0
        self._us = BASE_US

    # -- time ---------------------------------------------------------------

    def _next_session_us(self, min_gap_days: int = 1, max_gap_days: int = 6) -> int:
        self._us += self.rng.randint(min_gap_days, max_gap_days) * DAY_US
        self._us += self.rng.randint(8, 19) * HOUR_US
        return self._us

    def _tick(self, us: int) -> int:
        return us + self.rng.randint(60, 900) * 1_000_000

    # -- units ---------------------------------------------------------------

    def _session(self) -> str:
        self._sess_seq += 1
        return f"sess-{self._sess_seq:04d}"

    def _add_unit(
        self,
        *,
        kind: str,
        speaker: str,
        session_id: str,
        text: str,
        occurred_us: int,
        perspective: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._seq += 1
        unit = {
            "id": f"u-{self._seq:05d}",
            "kind": kind,
            "speaker": speaker,
            "session_id": session_id,
            "text": text,
            "occurred_us": occurred_us,
            "perspective": perspective,
        }
        if meta:
            unit["meta"] = meta
        self.units.append(unit)
        return unit

    def add_filler(self, session_id: str, us: int) -> Dict[str, Any]:
        tpl = self.rng.choice(_FILLER_TEMPLATES)
        text = tpl.format(
            p=self.rng.choice(PEOPLE),
            s=self.rng.choice(SERVICES),
            w=self.rng.choice(WORK_NOUNS),
            d=self.rng.choice(WEEKDAYS),
        )
        speaker = self.rng.choice(("user",) + PEOPLE[:6])
        perspective = "user_stated" if speaker == "user" else "third_party"
        return self._add_unit(
            kind="turn",
            speaker=speaker,
            session_id=session_id,
            text=text,
            occurred_us=us,
            perspective=perspective,
        )

    def add_rule(
        self, session_id: str, us: int, rule: Dict[str, Any], rule_id: str
    ) -> Dict[str, Any]:
        return self._add_unit(
            kind="turn",
            speaker="user",
            session_id=session_id,
            text=rule["text"],
            occurred_us=us,
            perspective="user_stated",
            meta={
                "standing_rule": {
                    "rule_id": rule_id,
                    "pattern": rule["pattern"],
                    "action": rule["action"],
                    "condition": rule["condition"],
                    "trigger_entities": list(rule["entities"]),
                }
            },
        )

    def add_tool_exchange(
        self, session_id: str, us: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tool = self.rng.choice(TOOLS)
        call_text = self.rng.choice(_TOOL_CALL_TEMPLATES).format(
            tool=tool,
            s=self.rng.choice(SERVICES),
            c=self.rng.choice(CHANNELS),
        )
        call = self._add_unit(
            kind="tool_call",
            speaker="assistant",
            session_id=session_id,
            text=call_text,
            occurred_us=us,
            perspective="agent_action",
            meta={"tool": tool},
        )
        result_text = self.rng.choice(_TOOL_RESULT_TEMPLATES).format(
            tool=tool, s=self.rng.choice(SERVICES), n=self.rng.randint(0, 40)
        )
        result = self._add_unit(
            kind="tool_result",
            speaker=f"tool:{tool}",
            session_id=session_id,
            text=result_text,
            occurred_us=self._tick(us),
            perspective="system",
            meta={"tool": tool},
        )
        return call, result


# ---------------------------------------------------------------------------
# scenario emitters
# ---------------------------------------------------------------------------


def _emit_implicit_rule(b: _Builder, index: int) -> None:
    """One rule planted mid-stream + a cue-free task probe."""
    builder = _RULE_BUILDERS[index % len(_RULE_BUILDERS)]
    artifact = ARTIFACTS[index % len(ARTIFACTS)]
    # rotate qualifiers through the artifact so hundreds of planted rules
    # keep unique trigger entities and unambiguous gold sets
    cycle = index // len(ARTIFACTS)
    if cycle:
        svc_scope = SERVICES[cycle % len(SERVICES)]
        artifact = f"{artifact} for {svc_scope}"
        if cycle >= len(SERVICES):
            owner = PEOPLE[(cycle // len(SERVICES)) % len(PEOPLE)]
            artifact = f"{owner}'s {artifact}"
    channel = CHANNELS[index % len(CHANNELS)]
    service = SERVICES[index % len(SERVICES)]
    rule = builder(artifact, channel, service)
    rule_id = f"rule-imp-{index:04d}"

    session = b._session()
    us = b._next_session_us()
    rule_unit = b.add_rule(session, us, rule, rule_id)
    for _ in range(b.rng.randint(2, 4)):
        us = b._tick(us)
        b.add_filler(session, us)
    # a tool exchange in the same era adds realistic agent-action mass
    us = b._tick(us)
    b.add_tool_exchange(session, us)

    task_text = b.rng.choice(_TASK_TEXT_TEMPLATES).format(a=artifact)
    naive = f"post the {artifact} to {DEFAULT_CHANNEL}"
    expected = rule["action"]
    if rule["condition"]:
        expected = f"{rule['action']} ({rule['condition']})"
    b.tasks.append(
        {
            "task_id": f"act-imp-{index:04d}",
            "kind": "implicit_rule_recall",
            "task_text": task_text,
            "query": task_text,
            "gold_unit_ids": [rule_unit["id"]],
            "gold_rule_ids": [rule_id],
            "expected_action": expected,
            "naive_action": naive,
            "needs_history": True,
            "meta": {
                "rule_pattern": rule["pattern"],
                "trigger_entities": list(rule["entities"]),
            },
        }
    )


def _emit_compliance(b: _Builder, index: int) -> None:
    """Rule, then a later agent action that complies or violates."""
    builder = _RULE_BUILDERS[index % len(_RULE_BUILDERS)]
    artifact = ARTIFACTS[(index * 3 + 1) % len(ARTIFACTS)]
    service_scope = SERVICES[(index * 5 + 2) % len(SERVICES)]
    artifact = f"team {artifact} for {service_scope}"
    if index >= 48:  # pair cycle repeats every lcm(16,12)=48 — uniquify
        artifact = f"{artifact} wave-{index // 48}"
    channel = CHANNELS[(index * 7 + 3) % len(CHANNELS)]
    service = SERVICES[(index * 11 + 5) % len(SERVICES)]
    rule = builder(artifact, channel, service)
    rule_id = f"rule-cmp-{index:04d}"

    s1 = b._session()
    us = b._next_session_us()
    rule_unit = b.add_rule(s1, us, rule, rule_id)
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(s1, us)

    # weeks/months later: the agent acts on the artifact. The action is
    # dated after its rule but does not drag the global clock — the next
    # planted scenario interleaves naturally into the same history.
    s2 = b._session()
    us2 = us + b.rng.randint(40, 300) * DAY_US
    compliant = b.rng.random() < 0.55
    uses_channel = rule["pattern"] in (
        "from_now_on_always",
        "only_until",
        "by_default_when",
        "whenever",
        "going_forward",
        "policy_only_after",
    )
    if uses_channel:
        posted_to = channel if compliant else DEFAULT_CHANNEL
        action_text = f"I posted the {artifact} to {posted_to}."
    else:
        # non-channel rules: compliance = honoring the named condition verb
        did = "did" if compliant else "did not"
        action_text = f"I handled the {artifact}; I {did} apply your condition."
    action_unit = b._add_unit(
        kind="turn",
        speaker="assistant",
        session_id=s2,
        text=action_text,
        occurred_us=us2,
        perspective="agent_action",
        meta={"acts_on": rule_id, "compliant": compliant},
    )
    for _ in range(b.rng.randint(1, 2)):
        us2 = b._tick(us2)
        b.add_filler(s2, us2)

    task_text = b.rng.choice(_COMPLIANCE_TASK_TEMPLATES).format(
        a=artifact, date=_iso_day(us2)
    )
    b.tasks.append(
        {
            "task_id": f"act-cmp-{index:04d}",
            "kind": "rule_compliance",
            "task_text": task_text,
            "query": task_text,
            "gold_unit_ids": [rule_unit["id"], action_unit["id"]],
            "gold_rule_ids": [rule_id],
            "expect": {"compliant": compliant},
            "expected_action": (
                f"{rule['action']}"
                + (f" ({rule['condition']})" if rule["condition"] else "")
            ),
            "naive_action": "report compliance without consulting the rule",
            "needs_history": True,
            "meta": {
                "rule_pattern": rule["pattern"],
                "trigger_entities": list(rule["entities"]),
            },
        }
    )


def _emit_procedure(b: _Builder, index: int) -> None:
    """A runbook document for 'how do I deploy X' probes."""
    service = SERVICES[index % len(SERVICES)]
    proc_id = f"proc-{index:04d}"
    session = b._session()
    us = b._next_session_us()
    gold: List[str] = []
    intro = b._add_unit(
        kind="document",
        speaker="user",
        session_id=session,
        text=f"Runbook: how to deploy {service} safely.",
        occurred_us=us,
        perspective="document",
        meta={"procedure": proc_id, "step": 0},
    )
    gold.append(intro["id"])
    for step_no, tpl in enumerate(_PROCEDURE_STEPS, start=1):
        us = b._tick(us)
        u = b._add_unit(
            kind="document",
            speaker="user",
            session_id=session,
            text=f"Step {step_no}: " + tpl.format(s=service),
            occurred_us=us,
            perspective="document",
            meta={"procedure": proc_id, "step": step_no},
        )
        gold.append(u["id"])
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(session, us)

    b.tasks.append(
        {
            "task_id": f"act-pro-{index:04d}",
            "kind": "procedure_recall",
            "task_text": f"How do I deploy {service}?",
            "query": f"how do I deploy {service}",
            "gold_unit_ids": gold,
            "gold_rule_ids": [],
            "expected_action": f"follow the {service} runbook steps in order",
            "naive_action": f"run deployctl promote {service} directly",
            "needs_history": True,
            "meta": {"procedure": proc_id, "service": service},
        }
    )


def _emit_action_fact(b: _Builder, index: int) -> None:
    """An agent action stating a fact; probe retrieves it (H60 twin)."""
    text_tpl, q_tpl = _ACTION_FACTS[index % len(_ACTION_FACTS)]
    service = SERVICES[(index * 3 + 7) % len(SERVICES)]
    text = text_tpl.format(s=service, n=1000 + index * 37 % 9000)
    query = q_tpl.format(s=service)
    session = b._session()
    us = b._next_session_us()
    unit = b._add_unit(
        kind="turn",
        speaker="assistant",
        session_id=session,
        text=text,
        occurred_us=us,
        perspective="agent_action",
        meta={"action_fact": True},
    )
    for _ in range(b.rng.randint(1, 3)):
        us = b._tick(us)
        b.add_filler(session, us)
    b.tasks.append(
        {
            "task_id": f"act-fact-{index:04d}",
            "kind": "action_fact_lookup",
            "task_text": query,
            "query": query,
            "gold_unit_ids": [unit["id"]],
            "gold_rule_ids": [],
            "expected_action": "answer from the recorded action",
            "naive_action": "guess or say unknown",
            "needs_history": True,
            "meta": {},
        }
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate(
    seed: int = DEFAULT_SEED,
    *,
    n_implicit: int = 220,
    n_compliance: int = 120,
    n_procedure: int = 90,
    n_action_lookup: int = 70,
    filler_sessions: int = 40,
) -> Dict[str, Any]:
    """Generate the owned action corpus.

    Defaults yield 500 tasks (V7-24.10 ``owned_actions`` floor) with
    220 implicit-rule probes (V7-20.06 floor is 200). Smaller counts are
    fine for tests; the corpus stays deterministic per ``(seed, counts)``.
    """
    for name, val in (
        ("n_implicit", n_implicit),
        ("n_compliance", n_compliance),
        ("n_procedure", n_procedure),
        ("n_action_lookup", n_action_lookup),
        ("filler_sessions", filler_sessions),
    ):
        if not isinstance(val, int) or val < 0:
            raise ValueError(f"{name} must be a non-negative int, got {val!r}")

    b = _Builder(seed)
    for i in range(n_implicit):
        _emit_implicit_rule(b, i)
    for i in range(n_compliance):
        _emit_compliance(b, i)
    for i in range(n_procedure):
        _emit_procedure(b, i)
    for i in range(n_action_lookup):
        _emit_action_fact(b, i)
    for _ in range(filler_sessions):
        session = b._session()
        us = b._next_session_us()
        for _ in range(b.rng.randint(3, 7)):
            us = b._tick(us)
            b.add_filler(session, us)
        if b.rng.random() < 0.6:
            us = b._tick(us)
            b.add_tool_exchange(session, us)

    b.units.sort(key=lambda u: (u["occurred_us"], u["id"]))
    corpus = {
        "name": CORPUS_NAME,
        "generator": GENERATOR_ID,
        "constants": CONSTANTS_TAG,
        "seed": seed,
        "units": b.units,
        "tasks": b.tasks,
        "stats": {
            "n_units": len(b.units),
            "n_tasks": len(b.tasks),
            "n_rules": sum(1 for t in b.tasks if t["gold_rule_ids"]),
            "task_kinds": {
                k: sum(1 for t in b.tasks if t["kind"] == k)
                for k in sorted({t["kind"] for t in b.tasks})
            },
            "span_days": (
                (b.units[-1]["occurred_us"] - b.units[0]["occurred_us"])
                // DAY_US
                if b.units
                else 0
            ),
        },
    }
    corpus["digest"] = corpus_digest(corpus)
    return corpus


__all__ = [
    "GENERATOR_ID",
    "CONSTANTS_TAG",
    "CORPUS_NAME",
    "DEFAULT_SEED",
    "RULE_CUES",
    "corpus_digest",
    "generate",
]
