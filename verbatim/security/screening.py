"""Deterministic content screener — rules_v1 (SPEC_V3 §14.02, §31.01, §31.06).

Two independent judgments, never one scale (V3-14.01):

* ``content_form`` — descriptive / instructional / mixed / unknown. Imperative
  density, numbered steps, and command-bearing code blocks mark instructional
  form. Instructional form is NOT an attack signal: install guides, runbooks,
  test recipes, and safety procedures are legitimately instructional and must
  screen clean (V3-14.02, V3-14.11, B45 "Run pnpm test before merging").
* ``attack_risk`` — unassessed / no_findings / suspicious / blocked. Raised
  only by the specific boundary-violation patterns below: redirecting the
  receiving agent's instructions or authority (ignore/disregard/override your
  rules, "you are now <role>", "new instructions:", prompt-exfiltration
  demands), claiming privileges the text cannot grant (grant yourself admin,
  disable authorization checks, mark this as trusted), and persistence
  attacks (hide this from the user, plant backdoors, leak data through future
  responses, tamper with memory). One weak pattern → ``suspicious``; a strong
  boundary/authority/persistence pattern → ``blocked`` (§34 stage-1 control).

Rules are pure functions of the input text — no ML, no network, deterministic
(V3-31.06). Every finding carries ``{rule_id, span, excerpt}`` so a block is
auditable: it names the attempted violation and its location (V3-14.05).

Negation guard: a rule match whose governing clause is negated ("Never
ignore safety instructions", "you must not disable the checks") produces no
finding — safety documentation that *prohibits* the behavior is not itself
an attempt. The guard is deliberately narrow (same sentence only, sentence
boundaries stop it) so it cannot launder real attacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..core.types_v3 import (
    AttackRisk,
    ContentForm,
    TrustClass,
)

#: Bump-visible rule revision stamped on every verdict and persisted label.
#: Screening replays compare revisions; findings without a matching revision
#: are not portable evidence (V3-14.02, V3-15.12 analog for screening).
RULES_REVISION = "rules_v1:2025-01"

_FINDING_EXCERPT_MAX = 120


@dataclass(frozen=True)
class SecurityVerdict:
    """Outcome of ``screen_content`` (§14.01).

    ``findings`` is a tuple of ``{"rule_id", "span", "excerpt"}`` dicts —
    ``span`` is a ``[start, end]`` character-offset pair into the screened
    text. ``no_findings`` means no listed attack signal matched, never
    certified safety.
    """

    content_form: ContentForm
    attack_risk: AttackRisk
    findings: tuple[dict[str, Any], ...] = ()
    rules_revision: str = RULES_REVISION
    method: str = "rules"

    def __post_init__(self) -> None:
        if not isinstance(self.content_form, ContentForm):
            object.__setattr__(self, "content_form", ContentForm(self.content_form))
        if not isinstance(self.attack_risk, AttackRisk):
            object.__setattr__(self, "attack_risk", AttackRisk(self.attack_risk))
        object.__setattr__(self, "findings", tuple(self.findings))


# ---------------------------------------------------------------------------
# content_form: instruction/evidence classification (§14.02)
# ---------------------------------------------------------------------------

# Imperative sentence-openers. Deliberately limited to verbs that begin
# commands/procedures; declarative openers ("Reports", "Shows") stay out.
_IMPERATIVE_VERBS = frozenset({
    "run", "execute", "install", "open", "set", "add", "create", "check",
    "verify", "restart", "start", "stop", "copy", "download", "follow",
    "ensure", "configure", "edit", "update", "delete", "remove", "enable",
    "disable", "enter", "type", "click", "press", "select", "choose", "use",
    "perform", "apply", "save", "build", "test", "deploy", "commit", "push",
    "pull", "clone", "navigate", "launch", "invoke", "call", "read", "write",
    "make", "keep", "place", "put", "move", "rename", "change", "confirm",
    "review", "inspect", "repeat", "rerun", "rebuild", "retry", "log",
    "sign", "wait", "note", "remember", "mount", "unmount", "insert",
    "append", "prepend", "replace", "substitute", "grab", "fetch", "send",
    "grant", "allow", "permit", "deny", "reject", "accept", "complete",
    "mark", "record", "store", "load", "unload", "initialize", "reset",
    "disconnect", "connect", "attach", "detach", "compile", "generate",
    "ignore", "disregard", "forget", "pretend", "reveal", "show", "display",
    "output", "tell", "share", "give", "override", "bypass", "hide",
    "conceal", "plant", "exfiltrate", "leak", "obey", "comply", "refuse",
    "continue", "proceed", "remain", "stay", "become", "assume", "visit",
    "go", "try", "attempt", "practice", "print", "repeat", "recite",
    "list", "name", "provide", "supply", "specify", "indicate", "state",
})

# Shell/CLI command leaders for code-block command detection.
_COMMAND_LEADERS = re.compile(
    r"^\s*(?:\$\s*|>\s*|#\s*)?(?:sudo\s+|env\s+\w+=\S*\s+)?"
    r"(?:pnpm|npm|npx|yarn|bun|deno|pip|pip3|pipx|python|python3|pytest|apt"
    r"|apt-get|brew|git|cd|make|cmake|docker|kubectl|helm|cargo|go|rustc"
    r"|java|mvn|gradle|curl|wget|ssh|scp|rsync|tar|zip|unzip|chmod|chown"
    r"|mkdir|rm|mv|cp|ln|touch|cat|echo|grep|find|sed|awk|systemctl"
    r"|service|ps|kill|killall|top|htop|df|du|mount|umount|lsmod|insmod"
    r"|modprobe|sudo|su|bash|sh|zsh|fish|powershell|pwsh|cmd|node|tsc"
    r"|eslint|prettier|black|flake8|mypy|ruff|tox|nox|ansible|terraform"
    r"|vagrant|nix|bundle|rake|rails|gem|composer|php|artisan|symfony"
    r"|dotnet|nuget|msbuild|xcodebuild|fastlane|pod|swift|kotlinc|mysql"
    r"|psql|sqlite3|redis-cli|mongo|mongosh|openssl|gpg|ffmpeg|convert"
    r"|magick|pandoc|latex|pdflatex|xelatex|jupyter|nbconvert)\b",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_NUMBERED_STEP_RE = re.compile(r"^\s*(?:step\s+\d+[:.)]?|\d{1,2}[.)])\s+\S", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_NEGATION_RE = re.compile(
    r"\b(?:never|not|don't|do not|does not|doesn't|cannot|can't|cant"
    r"|must not|mustn't|should not|shouldn't|won't|will not|shall not"
    r"|avoid|avoiding|refrain|refrain|without)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?;\n]")


def _code_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(1), m.end(1)) for m in _CODE_FENCE_RE.finditer(text)]


def _count_command_lines(block: str) -> int:
    return sum(
        1 for line in block.splitlines() if line.strip() and _COMMAND_LEADERS.match(line)
    )


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace code-fence regions with blanks so fenced commands don't
    count toward prose sentence/imperative analysis."""
    chars = list(text)
    for s, e in spans:
        for i in range(s, e):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _sentence_starts(text: str) -> list[str]:
    """Candidate imperative openers: sentence-initial words."""
    verbs: list[str] = []
    for chunk in _SENTENCE_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"[\W_]*([A-Za-z]+)", chunk)
        if m and m.group(1).lower() in _IMPERATIVE_VERBS:
            verbs.append(m.group(1).lower())
    return verbs


def _post_comma_imperatives(text: str) -> int:
    """Imperative main clauses after a subordinate opener.

    "If tests fail, check the lockfile." — the instruction is real even
    though the sentence starts with "If".
    """
    n = 0
    for m in re.finditer(
        r"[,;]\s*(?:then\s+|please\s+)?([a-zA-Z]+)\s", text, re.IGNORECASE
    ):
        if m.group(1).lower() in _IMPERATIVE_VERBS:
            n += 1
    return n


def classify_form(text: str) -> ContentForm:
    """Content form from imperative/command density (§14.02).

    ``instructional`` needs either structural evidence (numbered steps,
    command-bearing code blocks) or high imperative density; a lone
    imperative inside declarative prose is ``mixed`` — honest, and it keeps
    runbooks instructional without making every imperative suspicious.
    """
    if not text or not text.strip():
        return ContentForm.UNKNOWN

    code_spans = _code_spans(text)
    command_lines = sum(
        _count_command_lines(text[s:e]) for s, e in code_spans
    )
    numbered = sum(
        1 for line in text.splitlines() if _NUMBERED_STEP_RE.match(line)
    )
    inline_commands = sum(
        1 for m in _BACKTICK_RE.finditer(text) if _COMMAND_LEADERS.match(m.group(1).strip())
    )

    prose = _mask_spans(text, code_spans)
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(prose) if s.strip()]
    n_sent = max(1, len(sentences))
    imperative = len(_sentence_starts(prose)) + _post_comma_imperatives(prose)
    strong = command_lines + numbered

    signals = imperative + strong + inline_commands
    if signals == 0:
        return ContentForm.DESCRIPTIVE
    if strong >= 2 or imperative >= 3 or imperative / n_sent >= 0.5:
        return ContentForm.INSTRUCTIONAL
    return ContentForm.MIXED


# ---------------------------------------------------------------------------
# attack_risk: boundary-violation rules (§14.02, §34 stage 1)
# ---------------------------------------------------------------------------

# Each rule: (rule_id, compiled pattern, tier) where tier is "strong"
# (a match → blocked) or "weak" (one match → suspicious). Every id names
# the attempted violation per §14.05.

_STRONG_RULES: list[tuple[str, "re.Pattern[str]"]] = []
_WEAK_RULES: list[tuple[str, "re.Pattern[str]"]] = []


def _strong(rule_id: str, pattern: str) -> None:
    _STRONG_RULES.append((rule_id, re.compile(pattern, re.IGNORECASE | re.DOTALL)))


def _weak(rule_id: str, pattern: str) -> None:
    _WEAK_RULES.append((rule_id, re.compile(pattern, re.IGNORECASE | re.DOTALL)))


_INST_TARGETS = (
    r"instructions?|rules?|guidelines?|directives?|policies|policy"
    r"|restrictions?|safeguards?|guardrails?|protocols?|programming"
    r"|training|constraints?|filters?|prompt\b"
)

# --- (a) boundary redirection -------------------------------------------
_strong(
    "boundary_redirection.ignore_prior",
    r"\b(?:ignore|disregard|forget|overlook|abandon|override|overrule|bypass"
    r"|disobey|violate|discard|dismiss|suppress|circumvent|neglect|omit)\b"
    r"[^.!?;\n]{0,60}?\b(?:" + _INST_TARGETS + r")\b",
)
_strong(
    "boundary_redirection.role_reset",
    r"\byou\s+(?:are|'re)\s+now\s+(?:in|a|an|the|operating|running|acting"
    r"|functioning|working|switched|being)\b",
)
_strong(
    "boundary_redirection.new_instructions",
    r"\b(?:new|updated|real|actual|true|revised|replacement|correct"
    r"|authentic|fresh)\s+(?:system\s+|hidden\s+|secret\s+)?instructions?\s*:",
)
_strong(
    "boundary_redirection.unrestricted_persona",
    r"\bpretend\b[^.!?;\n]{0,60}\b(?:no|not any|without|free of|lacking"
    r"|zero|unbound by|not bound by)\b[^.!?;\n]{0,40}\b(?:restrictions?"
    r"|rules?|limits?|limitations?|guidelines?|filters?|safeguards?"
    r"|policies|boundaries|constraints?)\b",
)
_strong(
    "boundary_redirection.act_unrestricted",
    r"\b(?:act|behave|respond|answer|operate|function|reply|perform)\s+"
    r"(?:as\s+)?(?:if|like|though)\s+you\s+(?:have|had|are|were|got|were given)"
    r"\s+(?:no|not any|zero|without)\s+(?:restrictions?|rules?|limits?"
    r"|limitations?|guidelines?|filters?|safeguards?|constraints?)\b",
)
_strong(
    "boundary_redirection.prompt_exfil",
    r"\b(?:reveal|print|show|display|output|repeat|recite|tell\s+me|share"
    r"|give\s+me|disclose|expose|dump|reproduce|write\s+out|transcribe|quote"
    r"|leak|list)\b[^.!?;\n]{0,40}\b(?:system\s+prompt|system\s+instructions?"
    r"|system\s+message|system\s+directives?|initial\s+instructions?"
    r"|original\s+instructions?|base\s+instructions?|master\s+instructions?"
    r"|hidden\s+(?:prompt|instructions?)|secret\s+(?:prompt|instructions?)"
    r"|your\s+(?:full\s+|entire\s+|complete\s+|exact\s+|verbatim\s+)?"
    r"(?:prompt|instructions?|programming|directives?|system\s+prompt"
    r"|system\s+instructions?)"
    r"|the\s+(?:full|entire|complete|exact|verbatim)\s+(?:prompt|instructions?"
    r"|programming|directives?|system\s+prompt|system\s+instructions?))\b",
)
_strong(
    "boundary_redirection.do_not_follow",
    r"\b(?:do\s+not|don't|dont|stop|cease|quit|never)\s+"
    r"(?:following|follow|obey|obeying|heeding|adhering\s+to|listening\s+to"
    r"|complying\s+with|respecting)\b[^.!?;\n]{0,40}\b(?:your\s+|the\s+|all\s+)?"
    r"(?:" + _INST_TARGETS + r")\b",
)

# --- (b) authority claims -------------------------------------------------
_strong(
    "authority_claim.self_grant",
    r"\b(?:grant|give|assign|award|bestow|confer)\s+"
    r"(?:yourself|you|the\s+agent|the\s+assistant|the\s+model|itself|the\s+system"
    r"|this\s+agent|this\s+assistant)\b[^.!?;\n]{0,40}"
    r"\b(?:admin|administrator|administrative|root|elevated|full|unrestricted"
    r"|superuser|sudo|all|total|complete|write|execute|any|new|additional"
    r"|extra|more|higher|special)?\s*(?:access|privileges?|permissions?"
    r"|rights?|control|authority|power|capabilities|ability)\b",
)
_strong(
    "authority_claim.privilege_escalation",
    r"\b(?:escalate|elevate|raise|increase|expand|extend|upgrade)\s+"
    r"(?:your|its|the\s+agent'?s|the\s+assistant'?s|the\s+model'?s)?\s*"
    r"(?:privileges?|permissions?|access|rights?|authority|control"
    r"|capabilities|clearance)\b",
)
_strong(
    "authority_claim.disable_checks",
    r"\b(?:disable|turn\s+off|switch\s+off|shut\s+off|deactivate|remove"
    r"|bypass|skip|override|neutralize|defeat|circumvent|eliminate|kill)\b"
    r"[^.!?;\n]{0,40}\b(?:authorization|authentication|access[- ]control"
    r"|permission|safety|security|content|moderation|integrity|audit|all|the"
    r"|your|its|every|any|these|those)\s*(?:checks?|filters?|restrictions?"
    r"|guardrails?|mechanisms?|policies|controls?|systems?|measures?"
    r"|protocols?|features?|limits?|barriers?)\b",
)
_strong(
    "authority_claim.mark_trusted",
    r"\b(?:mark|flag|label|tag|classify|register|record|treat|consider|count"
    r"|certify|whitelist|white-list|trust)\s+(?:this|it|these|the\s+content"
    r"|this\s+content|this\s+text|this\s+document|this\s+source|the\s+source"
    r"|this\s+file|these\s+instructions)\s+as\s+(?:trusted|trustworthy|safe"
    r"|verified|approved|authorized|principal_direct|admin|privileged"
    r"|system|internal|secure|benign|whitelisted)\b",
)
_strong(
    "authority_claim.granted_access",
    r"\byou\s+(?:now\s+)?(?:have|hold|possess|enjoy|wield)\s+(?:been\s+)?"
    r"(?:granted\s+|given\s+|awarded\s+)?(?:admin|administrator|root"
    r"|unrestricted|full|elevated|superuser|sudo|god[- ]mode|total|complete)"
    r"\s+(?:access|privileges?|permissions?|rights?|control|authority|power"
    r"|clearance)\b",
)

# --- (c) persistence / concealment / exfiltration -------------------------
_strong(
    "persistence.future_responses",
    r"\b(?:always|every\s+time|from\s+now\s+on|in\s+(?:all|every|each)"
    r"|each\s+time|whenever)\b[^.!?;\n]{0,60}\b(?:include|add|append|insert"
    r"|embed|attach|mention|say|state|write|put|respond|reply|answer|leak"
    r"|reveal|disclose|send|encode|hide|place|inject|sneak)\b"
    r"[^.!?;\n]{0,60}\b(?:future|every|each|your|all|subsequent|next"
    r"|upcoming|later|following)?\s*(?:responses?|answers?|replies|outputs?"
    r"|messages?|completions?|generations?)\b",
)
_strong(
    "persistence.inject_into_responses",
    r"\b(?:include|add|append|insert|embed|hide|leak|send|inject|smuggle"
    r"|sneak|encode|conceal|stash)\b[^.!?;\n]{0,50}\bin\s+(?:all|every|each"
    r"|your|future|subsequent|the\s+next)\s+(?:future\s+)?"
    r"(?:responses?|answers?|replies|outputs?|messages?|completions?)\b",
)
_strong(
    "persistence.backdoor",
    r"\b(?:plant|install|insert|create|leave|hide|place|deploy|drop"
    r"|establish|build)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:backdoors?|malware|trojans?|rootkits?|keyloggers?|payloads?"
    r"|implants?|beacons?|web\s*shells?)\b",
)
_strong(
    "persistence.exfiltration",
    r"\b(?:send|upload|transmit|post|exfiltrate|leak|forward|ship|relay"
    r"|smuggle|siphon|transfer|copy|move|write|save|store|push|report|phone"
    r"|beacon|encode)\b[^.!?;\n]{0,80}\b(?:passwords?|secrets?|tokens?|keys?"
    r"|credentials?|cookies?|history|conversations?|messages?|data|contents?"
    r"|files?|memory|prompts?|outputs?|plaintext|archives?|dumps?|snapshots?"
    r"|backups?)\b[^.!?;\n]{0,80}\bto\s+(?:https?|ftp|ssh|smtp)://",
)
_strong(
    "persistence.hide_from_user",
    r"\b(?:do\s+not|don't|dont|never)\s+(?:tell|inform|notify|let|show"
    r"|reveal|disclose|alert|warn|advise|mention\s+to|report\s+to"
    r"|disclose\s+to)\b[^.!?;\n]{0,40}\b(?:the\s+|your\s+|any\s+|this\s+)?"
    r"(?:user|owner|operator|administrator|human|person|reviewer|victim"
    r"|target|principal|anyone|anybody)\b",
)
_strong(
    "persistence.hide_activity",
    r"\b(?:hide|conceal|cover|mask|obscure|bury|suppress)\b[^.!?;\n]{0,40}"
    r"\b(?:this|these|your|its|the|any|all|their)\s+"
    r"(?:tracks?|activity|actions?|presence|traces?|footprints?|behavior"
    r"|behaviour|operations?|instructions?|intent|exfiltration|payload)\b"
    r"(?:[^.!?;\n]{0,40}\bfrom\s+(?:the\s+)?(?:user|operator|administrator"
    r"|reviewer|logs?|audit|monitoring|detection|oversight))?",
)
_strong(
    "persistence.keep_secret",
    r"\bkeep\b[^.!?;\n]{0,30}\b(?:this|these|it|our|the)\s+"
    r"(?:a\s+)?secret\b[^.!?;\n]{0,40}\b(?:from\s+(?:the\s+)?"
    r"(?:user|operator|owner|human|reviewer|principal|anyone|everyone)"
    r"|between\s+us)\b",
)
_strong(
    "persistence.memory_tamper",
    r"\b(?:modify|edit|alter|rewrite|change|tamper\s+with|corrupt|poison"
    r"|overwrite|inject\s+into|write\s+to|update|patch)\b[^.!?;\n]{0,40}"
    r"\b(?:your|the|its|agent'?s|assistant'?s|model'?s|long[- ]term"
    r"|persistent|stored)\s*(?:memory|memories|knowledge\s+base|facts?"
    r"|beliefs?|stored\s+data|database|state)\b[^.!?;\n]{0,80}"
    r"\b(?:leak|exfiltrat|send|reveal|disclose|hide|conceal|backdoor"
    r"|exfiltrate|steer|bias|manipulate|plant|suppress|inject)\b",
)
_strong(
    "persistence.exfil_verb",
    r"\bexfiltrat\w+\b",
)

# --- weak patterns → suspicious -------------------------------------------
_weak(
    "role_claim.roleplay",
    r"\b(?:pretend|act|roleplay|role-play|imagine|behave)\s+"
    r"(?:to\s+be|as|like|you'?re|you\s+are|that\s+you'?re|that\s+you\s+are)\b",
)
_weak(
    "role_claim.prompt_probe",
    r"\byour\s+(?:system|initial|original|base|hidden|secret|real|true"
    r"|actual|underlying|core|full|entire)\s+(?:prompt|instructions?"
    r"|programming|directives?|guidelines?|system\s+prompt)\b",
)
_weak(
    "boundary_redirection.repeat_above",
    r"\brepeat\b[^.!?;\n]{0,40}\b(?:words?|text|content|everything|all)"
    r"\s+(?:above|back|verbatim|exactly)\b",
)
_weak(
    "boundary_redirection.jailbreak_terms",
    r"\b(?:jailbreak|jail\s*break|do\s+anything\s+now|\bDAN\b\s+mode)\b",
)
_weak(
    "authority_claim.no_restrictions",
    r"\b(?:with|have|has|having|got|given|granted|enjoy)\s+"
    r"(?:no|zero|not\s+any)\s+(?:restrictions?|limits?|limitations?|rules?"
    r"|guidelines?|filters?|safeguards?|boundaries|constraints?)\b",
)
_weak(
    "persistence.send_to_url",
    r"\b(?:send|upload|transmit|post|forward|ship|relay|push|report"
    r"|exfiltrate|leak)\b[^.!?;\n]{0,60}\bto\s+https?://",
)
_weak(
    "persistence.erase_evidence",
    r"\b(?:delete|erase|remove|clear|wipe|destroy|purge|shred|scrub)\b"
    r"[^.!?;\n]{0,40}\b(?:the\s+|your\s+|all\s+|these\s+|any\s+|its\s+)?"
    r"(?:logs?|traces?|evidence|history|audit\s+(?:trail|logs?)|records?"
    r"|footprints?|transcripts?)\b",
)
_weak(
    "persistence.memory_write",
    r"\b(?:store|save|write|record|persist|commit|memorize|remember"
    r"|implant|inject)\s+(?:this|it|these|that|the\s+following)"
    r"\s+(?:in|to|into|inside|within)\s+(?:your|the|its|long[- ]?term"
    r"|permanent|agent'?s)\s*(?:memory|memories|knowledge\s+base|database"
    r"|long[- ]term\s+storage|state)\b",
)
_weak(
    "tool_invocation.dangerous_command",
    r"\b(?:rm\s+-[rfRF]{1,3}\s+/|sudo\s+rm\b|format\s+[a-zA-Z]:|dd\s+if="
    r"|mkfs\b|:\(\)\s*\{\s*:\|:|drop\s+table\b|shutdown\s+-h|del\s+/[fsq]"
    r"|rd\s+/s|cipher\s+/w|chmod\s+-R\s+777\s+/|chown\s+-R\b[^.!?;\n]{0,30}/)",
)


def _negated(text: str, start: int) -> bool:
    """True when the match's own clause negates it ("never ignore safety
    instructions"). Scoped to the same sentence — a boundary character
    between the negation and the match breaks the guard."""
    window = text[max(0, start - 60) : start]
    m = None
    for m in _NEGATION_RE.finditer(window):
        pass
    if m is None:
        return False
    tail = window[m.end() :]
    return not _SENTENCE_BOUNDARY_RE.search(tail)


def _findings(text: str) -> tuple[tuple[dict[str, Any], ...], bool, bool]:
    """(findings, saw_strong, saw_weak), deduped per (rule_id, span)."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    strong = False
    weak = False
    for rules, tier in ((_STRONG_RULES, True), (_WEAK_RULES, False)):
        for rule_id, pat in rules:
            for m in pat.finditer(text):
                if _negated(text, m.start()):
                    continue
                key = (rule_id, m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                excerpt = text[m.start() : m.end()]
                if len(excerpt) > _FINDING_EXCERPT_MAX:
                    excerpt = excerpt[: _FINDING_EXCERPT_MAX - 1] + "…"
                out.append(
                    {
                        "rule_id": rule_id,
                        "tier": "strong" if tier else "weak",
                        "span": [m.start(), m.end()],
                        "excerpt": excerpt,
                    }
                )
                if tier:
                    strong = True
                else:
                    weak = True
    out.sort(key=lambda f: (f["span"][0], f["rule_id"]))
    return tuple(out), strong, weak


def screen_content(
    text: str,
    *,
    source_trust: Any = TrustClass.UNKNOWN,
    context_kind: Optional[str] = None,
) -> SecurityVerdict:
    """Screen one text item under rules_v1 (§14.02, §31.06, §34.01).

    ``source_trust`` is validated and carried for callers — rules_v1 keeps
    findings purely content-driven so identical text yields identical
    verdicts regardless of claimed provenance (V3-14.09: claimed trust
    cannot clear a finding). ``context_kind`` is accepted for callers that
    route different surfaces (evidence, procedure pack, handoff capsule);
    rules_v1 applies the same rigor to every surface (V3-31.08).
    """
    if not isinstance(text, str):
        raise VerbatimError(ErrorCode.VALIDATION, "screen_content requires text")
    if isinstance(source_trust, TrustClass):
        trust = source_trust
    else:
        try:
            trust = TrustClass(source_trust)
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"unknown source_trust {source_trust!r}"
            ) from exc
    if context_kind is not None and not isinstance(context_kind, str):
        raise VerbatimError(ErrorCode.VALIDATION, "context_kind must be a string")

    form = classify_form(text)
    findings, strong, weak = _findings(text)
    if strong:
        risk = AttackRisk.BLOCKED
    elif weak:
        risk = AttackRisk.SUSPICIOUS
    else:
        risk = AttackRisk.NO_FINDINGS
    return SecurityVerdict(
        content_form=form,
        attack_risk=risk,
        findings=findings,
        rules_revision=RULES_REVISION,
    )


__all__ = [
    "RULES_REVISION",
    "SecurityVerdict",
    "classify_form",
    "screen_content",
]
