"""Deterministic rules decision backend (SPEC §23, §26).

This backend is the always-available floor: stdlib-only, no network, no
numerical dependencies. It answers *only* what can be decided from explicit,
named rules — when the evidence does not match a rule it abstains rather
than inventing a probability (SPEC §26: fabricated confidence is worse than
honest abstention).

Every outcome names the rule that fired (``outcome['rule']``) so decisions
remain auditable, and reports RuleOutcome semantics: ``rule_match`` /
``rule_no_match`` / ``abstain``. It never emits probability distributions.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..core.types import (
    ChangeSignal,
    DecisionRequest,
    DecisionResult,
    PairLabel,
    RuleOutcome,
    TaskKind,
)

RUBRIC_VERSION = "rules-1"

# --- durability ------------------------------------------------------------
# Explicit retention intent: imperative "remember" markers plus a narrow
# whitelist of unambiguous declarative identity facts. Absence of a marker is
# NOT evidence of ephemerality — it yields abstention, not rule_no_match,
# because lexical silence cannot prove content is not worth keeping.
_REMEMBER_MARKERS = re.compile(
    r"\b(remember\b(?:\s+this|\s+that)?|don't forget|do not forget|never forget"
    r"|keep in mind|make a note|note this down|write (this|it) down|save this"
    r"|store this)\b",
    re.IGNORECASE,
)
_DECLARATIVE_DURABLE = re.compile(
    r"\b(my (name|birthday|date of birth|address|phone( number)?|email)"
    r"\s+is|i was born)\b",
    re.IGNORECASE,
)

# --- change signal ---------------------------------------------------------
# Correction is checked first: "I was wrong" is stronger than a mere switch,
# and a text containing both a change and a correction cue is conservatively
# classed as a correction.
_CORRECTION_CUES = re.compile(
    r"\b(was wrong|were wrong|i meant|correction|correct that|"
    r"that'?s not what i said|misremembered|lied)\b",
    re.IGNORECASE,
)
_CHANGE_CUES = re.compile(
    r"\b(switched|no longer|moved to|used to|changed|now use|now uses|"
    r"stopped|quit|gave up)\b",
    re.IGNORECASE,
)

# --- importance ------------------------------------------------------------
_IMPORTANCE_CUES = re.compile(
    r"\b(never forget|important|crucial|critical|essential|always)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class RulesBackend:
    """Zero-dependency deterministic backend; never fabricates confidence."""

    name = "rules"
    RUBRIC = RUBRIC_VERSION

    # Lexical overlap above this ratio earns a 'relevant' rule label; below
    # it the honest answer is abstention (a ratio is not a probability of
    # relevance and is labeled as overlap in the outcome).
    RELEVANCE_THRESHOLD = 0.5

    def capabilities(self) -> dict[str, Any]:
        return {
            "tasks": tuple(t.value for t in TaskKind),
            "calibrated": False,
            "backend": self.name,
            "model_revision": None,
            "rubric_version": self.RUBRIC,
        }

    def available(self) -> bool:
        # Pure rules: always usable, even with no optional deps installed.
        return True

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        handler = {
            TaskKind.DURABILITY: self._durability,
            TaskKind.PAIR_RELATION: self._pair_relation,
            TaskKind.CHANGE_SIGNAL: self._change_signal,
            TaskKind.CONDITION_MATCH: self._condition_match,
            TaskKind.RELEVANCE: self._relevance,
            TaskKind.IMPORTANCE: self._importance,
        }[request.task]
        return handler(request)

    def _result(
        self,
        request: DecisionRequest,
        outcome: dict[str, Any],
        *,
        abstained: bool = False,
        reason: Optional[str] = None,
    ) -> DecisionResult:
        outcome.setdefault("backend_rule", True)
        return DecisionResult(
            task=request.task,
            backend=self.name,
            model_revision=None,
            rubric_version=self.RUBRIC,
            outcome=outcome,
            abstained=abstained,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # task handlers
    # ------------------------------------------------------------------

    def _durability(self, request: DecisionRequest) -> DecisionResult:
        text = request.state.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._result(
                request,
                {"rule": "durability_input", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="missing text",
            )
        if _REMEMBER_MARKERS.search(text):
            return self._result(
                request,
                {
                    "rule": "durability_remember_marker",
                    "outcome": RuleOutcome.RULE_MATCH.value,
                    "label": "durable",
                },
            )
        if _DECLARATIVE_DURABLE.search(text):
            return self._result(
                request,
                {
                    "rule": "durability_declarative_identity",
                    "outcome": RuleOutcome.RULE_MATCH.value,
                    "label": "durable",
                },
            )
        return self._result(
            request,
            {"rule": "durability_no_marker", "outcome": RuleOutcome.ABSTAIN.value},
            abstained=True,
            reason="no explicit retention marker",
        )

    def _pair_relation(self, request: DecisionRequest) -> DecisionResult:
        """Classify pairs from caller-precomputed boolean flags only.

        The caller supplies, per state['pairs'][i]: subject_equal,
        predicate_equal, condition_overlap, object_equal. Semantic judgment
        beyond these flags is out of scope for a rules backend — anything
        unresolved abstains with 'insufficient_context' rather than guessing.
        """
        pairs = request.state.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            return self._result(
                request,
                {"rule": "pair_flags", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="no pairs supplied",
            )
        decided: list[Optional[str]] = []
        per_pair: list[dict[str, Any]] = []
        for i, pair in enumerate(pairs):
            label = self._classify_pair(pair if isinstance(pair, dict) else {})
            decided.append(label)
            per_pair.append(
                {
                    "index": i,
                    "label": label,
                    "reason": None if label else "insufficient_context",
                }
            )
        any_abstain = any(lbl is None for lbl in decided)
        outcome = {
            "rule": "pair_flags",
            "outcome": (
                RuleOutcome.ABSTAIN.value if any_abstain else RuleOutcome.RULE_MATCH.value
            ),
            "pairs": per_pair,
            "labels": decided,
        }
        return self._result(
            request,
            outcome,
            abstained=any_abstain,
            reason="insufficient_context" if any_abstain else None,
        )

    @staticmethod
    def _classify_pair(flags: dict[str, Any]) -> Optional[str]:
        subject_eq = bool(flags.get("subject_equal"))
        predicate_eq = bool(flags.get("predicate_equal"))
        condition_overlap = bool(flags.get("condition_overlap"))
        object_eq = bool(flags.get("object_equal"))

        if not (subject_eq and predicate_eq):
            # Different subjects/predicates cannot be ordered against each
            # other by flags alone — that is insufficient context, not proof
            # of a scope difference.
            return None
        if not condition_overlap:
            # Same subject+predicate under non-overlapping conditions: the
            # propositions live in different applicable scopes.
            return PairLabel.DIFFERENT_SCOPE.value
        if object_eq:
            # Same subject, predicate, object, overlapping conditions: the
            # same proposition restated.
            return PairLabel.EQUIVALENT.value
        # Same subject+predicate, overlapping conditions, different object:
        # both cannot hold in the same scope and interval.
        return PairLabel.INCOMPATIBLE.value

    def _change_signal(self, request: DecisionRequest) -> DecisionResult:
        text = request.state.get("text")
        if not isinstance(text, str) and isinstance(request.state.get("pairs"), list):
            parts = []
            for pair in request.state["pairs"]:
                if isinstance(pair, dict) and isinstance(pair.get("new"), str):
                    parts.append(pair["new"])
            text = " ".join(parts)
        if not isinstance(text, str) or not text.strip():
            return self._result(
                request,
                {"rule": "change_lexicon", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="missing text",
            )
        if _CORRECTION_CUES.search(text):
            label = ChangeSignal.STATES_CORRECTION.value
            rule = "change_correction_cue"
        elif _CHANGE_CUES.search(text):
            label = ChangeSignal.STATES_CHANGE.value
            rule = "change_transition_cue"
        else:
            label = ChangeSignal.NEITHER.value
            rule = "change_no_cue"
        return self._result(
            request,
            {"rule": rule, "outcome": RuleOutcome.RULE_MATCH.value, "label": label},
        )

    def _condition_match(self, request: DecisionRequest) -> DecisionResult:
        """Evaluate a bounded condition tree against supplied attributes.

        state['condition'] is a Condition.to_json() tree; state['attributes']
        is a flat mapping. Any missing attribute makes the result unknown —
        abstain, never assume false or true.
        """
        cond = request.state.get("condition")
        attrs = request.state.get("attributes")
        if not isinstance(cond, dict) or not isinstance(attrs, dict):
            return self._result(
                request,
                {"rule": "condition_eval", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="missing condition or attributes",
            )
        verdict = self._eval_condition(cond, attrs)
        if verdict is None:
            return self._result(
                request,
                {"rule": "condition_eval", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="insufficient_context",
            )
        return self._result(
            request,
            {
                "rule": "condition_eval",
                "outcome": RuleOutcome.RULE_MATCH.value,
                "label": "match" if verdict else "no_match",
            },
        )

    @staticmethod
    def _eval_condition(node: Any, attrs: dict[str, Any]) -> Optional[bool]:
        """Three-valued evaluation: True / False / None (unknown)."""
        if not isinstance(node, dict):
            return None
        op = node.get("op")
        if op == "eq":
            key = node.get("key")
            if key not in attrs:
                return None
            return attrs[key] == node.get("value")
        children = node.get("children")
        if not isinstance(children, list) or not children:
            return None
        vals = [RulesBackend._eval_condition(c, attrs) for c in children]
        if op == "all":
            if any(v is False for v in vals):
                return False
            if any(v is None for v in vals):
                return None
            return True
        if op == "any":
            if any(v is True for v in vals):
                return True
            if any(v is None for v in vals):
                return None
            return False
        if op == "not":
            if len(vals) != 1:
                return None
            return None if vals[0] is None else (not vals[0])
        return None

    def _relevance(self, request: DecisionRequest) -> DecisionResult:
        """Query-coverage overlap ratio; NOT a probability of relevance."""
        query = request.state.get("query")
        text = request.state.get("text")
        if not isinstance(query, str) or not isinstance(text, str):
            return self._result(
                request,
                {"rule": "rule_overlap", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="missing query or text",
            )
        q = _tokens(query)
        if not q:
            return self._result(
                request,
                {"rule": "rule_overlap", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="empty query tokens",
            )
        overlap = len(q & _tokens(text)) / len(q)
        if overlap < self.RELEVANCE_THRESHOLD:
            return self._result(
                request,
                {
                    "rule": "rule_overlap",
                    "outcome": RuleOutcome.ABSTAIN.value,
                    "overlap": overlap,
                },
                abstained=True,
                reason="overlap below threshold",
            )
        return self._result(
            request,
            {
                "rule": "rule_overlap",
                "outcome": RuleOutcome.RULE_MATCH.value,
                "label": "relevant",
                "overlap": overlap,
            },
        )

    def _importance(self, request: DecisionRequest) -> DecisionResult:
        text = request.state.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._result(
                request,
                {"rule": "importance_emphasis", "outcome": RuleOutcome.ABSTAIN.value},
                abstained=True,
                reason="missing text",
            )
        if _IMPORTANCE_CUES.search(text):
            return self._result(
                request,
                {
                    "rule": "importance_emphasis",
                    "outcome": RuleOutcome.RULE_MATCH.value,
                    "label": "important",
                },
            )
        return self._result(
            request,
            {"rule": "importance_no_cue", "outcome": RuleOutcome.ABSTAIN.value},
            abstained=True,
            reason="no emphasis cue",
        )
