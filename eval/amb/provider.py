"""Verbatim memory provider v2 for the Agent Memory Benchmark —
SPEC_V8.5 §2 (V85-02.01–02.07; retires the v1 pack provider, which ran
the legacy v2 claim engine and measured 35.0% / 494-token contexts on
locomo10 — cited only as that baseline).

Upstream contract (pinned: github.com/vectorize-io/agent-memory-benchmark,
``src/memory_bench/memory/base.py::MemoryProvider`` — code-read
2026-09-24, pinned commit recorded in the run manifest):

* class attrs ``name``/``description``/``kind``/``provider``/``variant``/
  ``concurrency``/``supports_filters``;
* hooks ``initialize()`` / ``prepare(store_dir, unit_ids, reset)`` /
  ``cleanup()`` / ``set_extraction_labels(labels)``;
* ``ingest(documents: list[Document]) -> None``;
* ``retrieve(query, k=10, user_id=None, query_timestamp=None,
  filters=None) -> (documents, raw_response)`` — called *positionally*
  by ``base.async_retrieve`` when ``supports_filters`` is falsy, so the
  parameter order is load-bearing;
* LoCoMo's ``build_rag_prompt`` ``json.dumps``-es ``_raw_response``
  when non-None — ``retrieve`` therefore returns ``(docs, None)``
  ALWAYS (V85-02.05; never serialize the raw pack).  Per-query stats
  live on :attr:`query_records` / :meth:`token_curve` instead, which is
  what the local runner reads.

V8.5 bindings:

* **Ingest (V85-02.02)** — each AMB session document (``content`` =
  ``json.dumps`` of a turn list, or a structured ``messages`` field)
  becomes ONE ``Memory.add(messages=[{speaker, text, at, ...}],
  session_id=<doc.id>, occurred_at=<doc.timestamp>)`` — one turn unit
  per message with a real ordinal ``seq``.  ``dia_id`` and image
  captions ride inside the message dicts (the facade persists caller
  keys into revision metadata) and inside the provider-side
  :class:`~eval.amb._turns.SessionIndex`.  The whole-document blob path
  is forbidden for turn-list payloads (it stays available for genuine
  prose documents).  When a facade without a ``messages`` parameter is
  injected, the provider falls back to per-turn ``Memory.add`` calls
  sharing ``session_id=<doc.id>`` — feature-detected, disclosed in
  ``last_ingest["ingest_path"]`` and the manifest.
* **Session index (V85-02.02)** — ``unit-<hash>.sessions.json`` next to
  each bank store: ``doc.id -> ordered [(ordinal, speaker, text,
  dia_id, caption, unit_id, seq, byte pins)]`` so ±``W_r`` neighbor
  expansion is independent of engine internals and survives
  ``--skip-ingestion`` resume runs.
* **Query (V85-02.03)** — ``Memory.search(query, limit=L, as_of=
  query_timestamp, timeout_ms=…)`` with ``L`` = the engine cap (64).
  AMB's ``k`` is recorded, never obeyed — the token budget bounds the
  output, not the hit count.
* **Delivery expansion (V85-02.04)** — hits walk in rank order; each
  contributes the turn ordinals its unit covers ±``W_r`` session
  neighbors (``VERBATIM_AMB_NEIGHBOR_W``, default 1; arm {0,1,2});
  dedup; stop before exceeding ``B`` tokens
  (``VERBATIM_AMB_TOKEN_BUDGET``, default 4000; arm {2000, 4500};
  ``unbounded``/``none`` → no cap) counted with cl100k via tiktoken when
  installed, else the repo's pinned ``tok/v1`` estimator — the meter is
  recorded in the manifest either way.
* **Render (V85-02.05)** — one AMB ``Document`` per session excerpt:
  sessions ordered by best hit rank, turns in dialogue order, header
  ``[<conversation> · session <n> · <YYYY-MM-DD, Weekday>]``, each line
  ``Speaker: text`` plus `` [image: <caption>]`` when present,
  ``source_ids=[session doc id]``, ``raw_response=None``.
* **Concurrency (V85-02.06)** — ``concurrency`` from
  ``VERBATIM_AMB_CONCURRENCY`` (then ``AMB_VERBATIM_CONCURRENCY``,
  default 4).  Writes hold :attr:`_write_lock`; reads run on the
  store's per-thread reader connections — retrieval is byte-identical
  at concurrency 1 and 4 (durable test).
* **Manifest (V85-02.07)** — :meth:`manifest_fields` exposes provider
  revision, B, W_r, L, tokenizer, concurrency; the runner merges it
  into the manifest's ``arm`` block together with the AMB commit and
  the three ``amb_patches.md`` harness patches
  (:func:`harness_patch_records`).

Per-unit isolation (AMB ``isolation_unit`` / ``user_id`` scoping) uses
one ``Memory`` bank per unit id under ``prepare()``'s ``store_dir`` —
the same shape as AMB's per-unit bank convention.

No network, no model calls, stdlib only.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple,
)

from ._turns import (
    SessionIndex,
    messages_for_add,
    parse_chat_lines,
    parse_turn_list,
    render_header,
    render_turn_line,
    transcript_for_add,
    turns_from,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Provider implementation version — pinned into every run manifest.
PROVIDER_VERSION = "verbatim-amb/2"

#: Registry name inside AMB (``amb providers`` listing, K87).
PROVIDER_NAME = "verbatim"

#: V8-15.06 pins the product default deadline on every search call.
DEFAULT_TIMEOUT_MS = 500.0

#: V8-15.08 token-budget sweep; ``None`` renders the unbounded point.
TOKEN_BUDGETS = (1000, 2000, 4500, 9000, None)

#: ``doc_mode`` values — "pack" (rag context) | "items" (retrieval
#: mode).  Both emit the same session-excerpt documents under v2; the
#: knob is kept for the manifest/record surface.
DOC_MODES = ("pack", "items")

#: Facade ``Memory.search`` clamps ``limit`` to ``_MAX_LIMIT`` (64);
#: V85-02.03 pins L to the engine cap.
DEFAULT_SEARCH_LIMIT = 64
DEFAULT_NEIGHBOR_W = 1
DEFAULT_TOKEN_BUDGET = 4000
DEFAULT_CONCURRENCY = 4

#: ``Memory.add(messages=...)`` bound (facade ``_MAX_TURN_MESSAGES``).
_MAX_MESSAGES_PER_ADD = 512

#: Enumerative/multi-evidence question shapes — the session-diversity
#: ordering (``VERBATIM_AMB_DDIVERSE=auto``) applies only to these:
#: list-style questions need every evidence session inside the token
#: budget, while single-fact queries are hurt by breadth (measured:
#: global diversity cost open-domain accuracy on the conv-42 board).
_ENUMERATIVE_RE = re.compile(
    r"\b(how many|how much|which (?:activities|sports|hobbies|books|"
    r"movies|songs|places|cities|countries|states|foods|games|pets|"
    r"people|friends|subjects|topics|things|items|kinds|types|sorts)|"
    r"what (?:activities|sports|hobbies|books|movies|songs|places|"
    r"cities|countries|states|foods|games|pets|people|friends|subjects|"
    r"topics|things|items|kinds|types|sorts|all|other)|"
    r"\bhas (?:\w+ ){0,3}(?:done|been|read|met|seen|watched|written|"
    r"visited|attended|tried|played)|"
    r"\bhave (?:\w+ ){0,3}(?:done|been|read|met|seen|watched|written|"
    r"visited|attended|tried|played)|"
    r"\bdoes (?:\w+ ){0,2}(?:have|own)|"
    r"\bwhat (?:\w+ )?does (?:\w+ ){0,2}(?:use|play|like|own)|"
    r"\bname (?:all|the)|\ball of the|\bevery\b|\bbesides\b|"
    r"\bwhich (?:\w+ )?(?:city|cities|state|states|country|countries|"
    r"area|areas|place|places|location|locations)\b)",
    re.IGNORECASE)

# --- datecalc probe -----------------------------------------------------
#: month-name + ordinal-date + offset patterns for VERBATIM_AMB_DATECALC —
#: resolves "the 44th day after March 20" / "226 days before New Year"
#: against the question-time header into absolute probe dates.
_DC_MONTHS = {
    m: i for i, m in enumerate(
        ("january february march april may june july august "
         "september october november december").split(), 1)}
_DC_MDATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE)
_DC_WORDN = {
    w: i for i, w in enumerate(
        ("one two three four five six seven eight nine ten eleven "
         "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen").split(), 1)}
_DC_WORDN.update({w: i * 10 for i, w in enumerate(
    ("twenty thirty forty fifty sixty seventy eighty ninety "
     "hundred").split(), 2)})
_DC_OFF_RE = re.compile(
    r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred)(?:st|nd|rd|th)?\s*"
    r"(day|days|week|weeks|month|months)\s+(before|after|ago)",
    re.IGNORECASE)
_DC_QTIME_RE = re.compile(
    r"question time:\s*(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE)


def _dc_qdates(qtext: Any, query_timestamp: Any) -> List[Any]:
    """Absolute dates referenced by a date-arithmetic question.

    Explicit ``<Month> <D>`` mentions are themselves probes (the day's
    session usually IS evidence), and ``N days|weeks|months
    before|after|ago`` offsets are resolved against explicit dates in
    the question, named anchors (new year, christmas), or — only for
    'ago' — the question-time header.  Returns ≤4 ``date`` objects."""
    from datetime import date as _d, timedelta as _td

    qtext = str(qtext or "")
    t0 = None
    m = _DC_QTIME_RE.search(qtext)
    if m:
        try:
            t0 = _d(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            t0 = None
    if t0 is None and query_timestamp is not None:
        try:
            t0 = _d.fromisoformat(str(query_timestamp)[:10])
        except (ValueError, TypeError):
            t0 = None
    yr = t0.year if t0 else 2025
    out: List[Any] = []
    explicits: List[Any] = []
    for mm in _DC_MDATE_RE.finditer(qtext):
        try:
            explicits.append(
                _d(yr, _DC_MONTHS[mm.group(1).lower()],
                   int(mm.group(2))))
        except (ValueError, KeyError):
            pass
    out.extend(explicits)
    anchors = list(explicits)
    ql = qtext.lower()
    if "new year" in ql:
        anchors.append(_d(yr + 1, 1, 1))
    if "christmas" in ql:
        anchors.append(_d(yr, 12, 25))
    for mm in _DC_OFF_RE.finditer(qtext):
        raw_n = mm.group(1).lower()
        n = (int(raw_n) if raw_n.isdigit()
             else _DC_WORDN.get(raw_n, 0))
        if not n:
            continue
        unit = mm.group(2).lower()
        days = (n if unit.startswith("day")
                else n * 7 if unit.startswith("week") else n * 30)
        direc = mm.group(3).lower()
        if direc == "ago":
            if t0 is None:
                continue
            use = [t0]
        elif anchors:
            use = list(anchors)
        else:
            continue  # 'before/after <event>' — event is undated here
        sign = -1 if direc in ("before", "ago") else 1
        for a in use:
            try:
                out.append(a + _td(days=sign * days))
            except (OverflowError, ValueError):
                pass
    seen: set = set()
    res: List[Any] = []
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        res.append(d)
    return res[:4]

#: entity_canon noise guard — sentence-initial fragments the write
#: path minted as "entities" ('Awesome, Jolene', 'Bye Joanna',
#: 'Woah Joanna').  Real anchors ('Paris', 'Talkeetna', 'Turtles')
#: never match either arm.
_GREETING_ENT_RE = re.compile(
    r"[,!]\s*[A-Z]"
    r"|^(?:bye|woah|wow|woohoo|yum|yep|yeah|yup|aww+|ugh|anytime|"
    r"anyways|alright|appreciated?|agreed|bummer|cheers|cool|dang|"
    r"definitely|glad|gotta|haha|hopefully|thanks|thank|sounds|"
    r"sure|nice|worries|connecting|believing?|brings)\b",
    re.IGNORECASE)

#: temporal-shaped question detector — gates the TORDER reorder's
#: 'auto' arm (and reported in query records).  Duration ("how many
#: days/weeks ago"), ordering ("which happened first"), and
#: recency ("most recent") shapes.
_TEMPORAL_RE = re.compile(
    r"\b(how many (?:days|weeks|months|years|hours|times)|how long|"
    r"how often|\bdays? ago\b|\bweeks? ago\b|\bmonths? ago\b|"
    r"\byears? ago\b|\bago\b|most recent|latest|earliest|last time|"
    r"first time|\bthe last\b|\bthe first\b|\bbefore\b|\bafter\b|"
    r"\bsince\b|\buntil\b|\bwhen\b|\bwhich (?:day|date|week|month|"
    r"year)\b|\bin what order\b|from first to last|from last to "
    r"first|\bchronolog|\bprevious|\bprior\b|\bearlier\b|"
    r"\b(?:the )?(?:next|following) (?:day|week|month|time)\b|"
    r"\bhow (?:many|much) (?:time|long)\b|passed between|"
    r"\bfrequency\b|\bmore recently\b|\brecently\b|"
    r"\border\b|\bsequence\b|\bfirst\b|\blast\b|\bsecond\b|"
    r"\bthird\b|\bconsecutiv|\bsimultaneous|\bfollowed by\b|"
    r"\bmonday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday\b|\bweekend\b|\bjanuary|february|march|april|"
    r"may\b|\bjune|july|august|september|october|november|"
    r"december\b|valentine|christmas|halloween|thanksgiving|"
    r"new year|birthday|\btonight\b|\byesterday\b|"
    r"\btomorrow\b|\btonight\b|\bweekday\b|\bfortnight\b)",
    re.IGNORECASE)


# --- conditional QX gate --------------------------------------------------
#: VERBATIM_AMB_QX_COND — lexical gate that skips LLM query expansion on
#: questions whose phrasing signals preference-reasoning rather than fact
#: recall.  Three signals on the question text:
#:   * judgment-seeking stems ("should I", "do you think", "been
#:     considering", "planning") — the user proposes and wants a verdict;
#:   * change-narrative stems ("I've decided", "no longer", "used to");
#:   * suggestion-shaped options without recall-shaped options.
#: Expansion terms pull distractor sessions on reasoning shapes (pmem
#: measured: generalizing +10.6 / recommendations +9.0 / reasons +4.1 with
#: QX off) while recall shapes need it (shared_facts −9.3, sni −7.5,
#: facts_mentioned −5.9 when off).
_QXC_JUDGE_RE = re.compile(
    r"\b(should i\b|not sure (?:if|whether)|do you think|any thoughts|"
    r"what do you think|whether i should|right (?:fit|choice|decision)|"
    r"good (?:idea|fit)|is it (?:something|a good)|does it "
    r"(?:make sense|suit)|worth (?:it|pursu)|or just|or should)\b",
    re.IGNORECASE)
_QXC_CHG_RE = re.compile(
    r"\b(i'?ve ?(?:decided|found|realized|felt|noticed|started|"
    r"stopped|quit|rejoined|restarted|stepped back|moved on|shifted)|"
    r"i (?:decided|found|realized|felt|noticed|started|stopped|quit|"
    r"rejoined|restarted|stepped back|moved on|shifted)|anymore|"
    r"change of heart|no longer|used to)\b",
    re.IGNORECASE)
_QXC_OPT_SUG_RE = re.compile(
    r"\([a-d]\)[^\n]*(?:consider|explor|try|might|could|suggest|"
    r"recommend|why not|how about|imagine|picture)", re.IGNORECASE)
_QXC_OPT_REC_RE = re.compile(
    r"\([a-d]\)[^\n]*(?:you (?:mentioned|said|told|recall|felt|enjoyed|"
    r"liked|disliked)|your (?:preference|interest|love|enjoy|dislike|"
    r"passion)|you'?ve|you have|i understand your|previously,)",
    re.IGNORECASE)


def _qx_cond_off(query: str) -> bool:
    """True when the conditional gate suppresses query expansion for this
    question: a reasoning-shaped stem, or suggestion options with no
    recall options."""
    q = str(query)
    stem = re.split(r"\n\s*\(a\)", q, 1)[0]
    if _QXC_JUDGE_RE.search(stem) or _QXC_CHG_RE.search(stem):
        return True
    return bool(_QXC_OPT_SUG_RE.search(q)) and not _QXC_OPT_REC_RE.search(q)


def _sess_dt(sess: Mapping[str, Any]) -> Any:
    """Session datetime (aware) or None — reuses _turns' tolerant
    ``date_label`` parser by way of isoformat strings and datetime
    objects; numeric µs/ms/s magnitudes follow the same heuristic."""
    ts = sess.get("timestamp")
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        v = float(ts)
        if v >= 10**14:
            v = v / 1_000_000.0
        elif v >= 10**11:
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    s = str(ts).strip()
    if not s:
        return None
    if s.isdigit():
        return _sess_dt({"timestamp": int(s)})
    cleaned = s.split("(")[0].strip() if "(" in s else s
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(
                cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_SNIP_STOP = frozenset(
    "the a an and or of to in on for with at by from as is are was "
    "were be been i you your my me we they he she it its this that "
    "these those do does did have has had can could will would should "
    "what which when where who whom how many much ago last past next "
    "since until before after between first latest most recent order "
    "happened happen did day days week weeks month months year years "
    "time times long often".split())


def _q_terms(query: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", query.lower())
            if len(w) >= 3 and w not in _SNIP_STOP]


def _snip_turn(sess: Mapping[str, Any], terms: List[str],
               span: int = 90) -> str:
    """Best-snippet for a session vs the query: turn with the most
    distinct query terms, window centered on the first match.  Falls
    back to the first user turn (then turn 0) when nothing matches."""
    turns = sess.get("turns") or []
    if not turns:
        return ""
    best_i, best_score = -1, 0
    for t in turns:
        txt = str(t.get("text") or "")
        low = txt.lower()
        score, pos = 0, -1
        for w in terms:
            j = low.find(w)
            if j >= 0:
                score += 1
                if pos < 0:
                    pos = j
        if score > best_score:
            best_i, best_score = t["i"], score
    if best_i >= 0:
        txt = re.sub(r"\s+", " ", str(
            turns[best_i].get("text") or "")).strip()
        # map the match position onto the whitespace-normalized text
        low2 = txt.lower()
        pos = min((low2.find(w) for w in terms if low2.find(w) >= 0),
                  default=0)
        start = max(0, pos - 20)
        out = txt[start:start + span]
        return ("…" if start else "") + out
    uords = [t["i"] for t in turns
             if str(t.get("speaker") or "") == "user"]
    pick = uords[0] if uords else 0
    return re.sub(r"\s+", " ", str(
        turns[pick].get("text") or "")).strip()[:span]


def _clip_turn(line: str, cap: Any = None,
               terms: Any = ()) -> str:
    """``VERBATIM_AMB_TURN_CLIP`` (chars, 0=off) — head-clips a rendered
    turn line at a word boundary.  Long assistant list answers carry
    filler that starves co-evidence sessions out of the token budget;
    the evidence claim almost always leads the turn.

    ``VERBATIM_AMB_TURN_CLIP_NT`` overrides the cap for non-temporal
    questions (measured: single-session-assistant answers often live
    mid-way through multi-KB pasted documents — clipping at 400 chars
    loses them; pass 0 to disable there).

    ``VERBATIM_AMB_TERM_CLIP`` — when set and ``terms`` is non-empty,
    long turns clip to a window centered on the first query-term
    occurrence instead of the head: evidence rarely sits at char 0
    (e.g. 'construction began in 2014' at char 900 of a pasted doc)
    while the cap still bounds the token spend."""
    if cap is None:
        cap = _env_int("VERBATIM_AMB_TURN_CLIP", 0)
    if cap and len(line) > cap:
        if terms:
            low = line.lower()
            occ = []
            for w in terms:
                j = low.find(w)
                while j >= 0:
                    occ.append(j)
                    j = low.find(w, j + 1)
            if occ:
                occ.sort()
                # densest window: the cap-wide span containing the most
                # distinct query terms (evidence clusters; a header
                # mention at char 0 shouldn't win over the answer)
                span = cap * 3 // 4
                best_i, best = 0, -1
                for i, p in enumerate(occ):
                    hit = {w for w in terms
                           if low.find(w, p, p + span) >= 0}
                    if len(hit) > best:
                        best, best_i = len(hit), i
                pos = occ[best_i]
                start = max(0, pos - cap // 4)
                end = min(len(line), start + cap)
                start = max(0, end - cap)
                seg = line[start:end]
                sp = seg.find(" ")
                if start and sp >= 0 and sp < cap // 3:
                    seg = seg[sp + 1:]
                return ("…" if start else "") + seg.rstrip() + " […]"
        cut = line[:cap]
        sp = cut.rfind(" ")
        if sp > cap * 2 // 3:
            cut = cut[:sp]
        return cut.rstrip() + " […]"
    return line


def _rel_tag(sess_dt: Any, ask_dt: Any) -> str:
    """'· N days (W weeks, ~M months) before the question' — the
    reader does no date arithmetic: day/week/month distances to the
    question anchor are read off the header directly."""
    d = int(round((ask_dt - sess_dt).total_seconds() / 86400.0))
    if d == 0:
        return "on the same day as the question"
    if d < 0:
        n = -d
        return (f"{n} day{'s' if n != 1 else ''} AFTER the question "
                f"date")
    w = d // 7
    m = int(round(d / 30.4375))
    if d == 1:
        return "1 day before the question"
    if d < 7:
        return f"{d} days before the question"
    if d < 45:
        return f"{d} days ({w} weeks) before the question"
    return f"{d} days ({w} weeks, ~{m} months) before the question"


#: query-domain lexicons for the entity-anchored splice — evidence
#: turns phrase the answer in domain verbs the question never repeats
#: ('took my turtles to the beach in Tampa' vs 'what state did Nate
#: visit').  A query matching a domain regex merges that lexicon into
#: the relevance bag so the right entity's turn scores.
_ENT_DOMAIN_LEX: Tuple[Tuple[Any, frozenset], ...] = (
    (re.compile(
        r"\b(state|states|city|cities|country|countries|town|place|"
        r"places|location|locations|where|visit|visits|visited|trip|"
        r"trips|move|moved|live|lives|living|travel|travels|traveled|"
        r"vacation|vacations|hometown)\b", re.IGNORECASE),
     frozenset({"took", "trip", "visit", "visited", "went", "go",
                "going", "flew", "fly", "flying", "drive", "drove",
                "driving", "beach", "vacation", "weekend", "stay",
                "stayed", "travel", "traveled", "flight", "airport",
                "hotel", "hometown", "moved", "moving", "lives",
                "living", "from", "to", "in", "spent", "spending"})),
    (re.compile(
        r"\b(career|careers|job|jobs|work|works|working|degree|"
        r"degrees|major|majors|study|studies|studied|profession|"
        r"professions|occupation|occupations|employ|employed|"
        r"intern|internship|hire|hired)\b", re.IGNORECASE),
     frozenset({"job", "work", "working", "career", "hired", "hire",
                "offer", "offered", "position", "interview", "degree",
                "major", "graduated", "graduating", "college",
                "university", "study", "studied", "studying", "class",
                "intern", "internship", "boss", "salary", "employee",
                "employer", "keeper", "keeper", "taking", "care"})),
    (re.compile(
        r"\b(health|sick|problem|problems|disease|diseases|doctor|"
        r"doctors|hospital|pain|ill|illness|symptom|symptoms|"
        r"diagnos\w*|weight|diet|exercise)\b", re.IGNORECASE),
     frozenset({"doctor", "hospital", "sick", "pain", "diagnosed",
                "disease", "illness", "medicine", "symptom",
                "symptoms", "health", "checkup", "checkups",
                "appointment", "treatment", "surgery", "weight",
                "overweight", "diet", "lose", "losing", "lost",
                "blood", "pressure", "cholesterol", "sleep",
                "sleeping", "tired", "fatigue", "gym", "exercise"})),
    (re.compile(
        r"\b(book|books|read|reads|reading|author|authors|novel|"
        r"novels|movie|movies|film|films|show|shows|watch|watched|"
        r"watching|song|songs|album|albums|game|games|gaming)\b",
        re.IGNORECASE),
     frozenset({"read", "reading", "book", "novel", "author",
                "chapter", "page", "watch", "watched", "watching",
                "movie", "film", "show", "episode", "season", "song",
                "album", "listen", "listened", "play", "played",
                "playing", "game", "gaming", "story", "series"})),
)

#: irregular + suffix morphology for entity-splice scoring —
#: evidence turns inflect the query's verbs ('bought' vs 'buy',
#: 'studied' vs 'study'); without variants the right entity's
#: mention turn scores zero and never gets spliced.
_VERB_IRREG: Dict[str, frozenset] = {
    w: frozenset(v.split()) for w, v in {
        "buy": "bought", "go": "went gone", "take": "took taken",
        "see": "saw seen", "make": "made", "get": "got gotten",
        "come": "came", "run": "ran", "eat": "ate",
        "write": "wrote written", "speak": "spoke spoken",
        "keep": "kept", "teach": "taught", "bring": "brought",
        "find": "found", "feel": "felt", "leave": "left",
        "mean": "meant", "pay": "paid", "say": "said",
        "tell": "told", "think": "thought", "have": "had",
        "do": "did done", "meet": "met", "win": "won",
        "lose": "lost", "sit": "sat", "give": "gave given",
        "drive": "drove driven", "fly": "flew flown",
        "grow": "grew grown", "know": "knew known",
        "throw": "threw thrown", "wear": "wore worn",
        "sleep": "slept", "swim": "swam", "begin": "began begun",
        "drink": "drank", "sing": "sang", "ride": "rode ridden",
        "study": "studied studying studies", "try": "tried tries",
        "apply": "applied applies", "carry": "carried carries",
        "hike": "hiked hikes", "bake": "baked bakes",
        "exercise": "exercised exercises",
        "volunteer": "volunteered volunteers",
    }.items()
}


_VERB_REV: Dict[str, frozenset] = {
    _v: frozenset(
        _l for _l, _vs in _VERB_IRREG.items() if _v in _vs)
    for _v in {x for _vs in _VERB_IRREG.values() for x in _vs}
}


def _morph(w: str) -> frozenset:
    # curated irregulars only — generic suffix expansion ('play' →
    # 'played'/'playing' on every rel term) floods the bag and
    # rescues junk entities, a measured net-negative on the board.
    out = {w}
    out.update(_VERB_IRREG.get(w, ()))
    out.update(_VERB_REV.get(w, ()))
    return frozenset(out)


#: bidirectional place↔container map — evidence turns name the CITY
#: ('adopted a pup from a shelter in Stamford') while questions name
#: the STATE ('Does James live in Connecticut'), and 'what country'
#: questions need the reverse edge (Paris → France).  Fires on the
#: expanded rel bag (QX terms included), not just the raw query.
_GEO_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("connecticut", "stamford"), ("connecticut", "hartford"),
    ("connecticut", "new haven"), ("connecticut", "bridgeport"),
    ("minnesota", "minneapolis"), ("minnesota", "duluth"),
    ("minnesota", "rochester"), ("minnesota", "saint paul"),
    ("minnesota", "st paul"), ("alaska", "talkeetna"),
    ("alaska", "anchorage"), ("alaska", "juneau"),
    ("france", "paris"), ("france", "lyon"), ("france", "nice"),
    ("brazil", "rio de janeiro"), ("brazil", "sao paulo"),
    ("thailand", "bangkok"), ("thailand", "phuket"),
    ("thailand", "chiang mai"), ("indonesia", "bali"),
    ("indonesia", "jakarta"), ("greenland", "nuuk"),
    ("italy", "rome"), ("italy", "venice"), ("italy", "milan"),
    ("italy", "florence"), ("spain", "barcelona"),
    ("spain", "madrid"), ("england", "london"),
    ("england", "liverpool"), ("england", "manchester"),
    ("united kingdom", "london"), ("scotland", "edinburgh"),
    ("scotland", "glasgow"), ("washington", "seattle"),
    ("washington", "spokane"), ("washington", "tacoma"),
    ("oregon", "portland"), ("california", "los angeles"),
    ("california", "san francisco"), ("california", "san diego"),
    ("california", "sacramento"), ("florida", "tampa"),
    ("florida", "miami"), ("florida", "orlando"),
    ("arizona", "phoenix"), ("arizona", "tucson"),
    ("arizona", "sedona"), ("new york", "manhattan"),
    ("new york", "brooklyn"), ("new york", "buffalo"),
    ("texas", "austin"), ("texas", "dallas"), ("texas", "houston"),
    ("texas", "san antonio"), ("pennsylvania", "philadelphia"),
    ("pennsylvania", "pittsburgh"), ("massachusetts", "boston"),
    ("ohio", "columbus"), ("ohio", "cleveland"),
    ("colorado", "denver"), ("colorado", "boulder"),
    ("nevada", "las vegas"), ("nevada", "reno"),
    ("georgia", "atlanta"), ("illinois", "chicago"),
    ("michigan", "detroit"), ("tennessee", "nashville"),
    ("tennessee", "memphis"), ("louisiana", "new orleans"),
    ("missouri", "kansas city"), ("virginia", "richmond"),
    ("maryland", "baltimore"), ("north carolina", "charlotte"),
    ("north carolina", "raleigh"), ("utah", "salt lake city"),
    ("new jersey", "newark"), ("japan", "tokyo"),
    ("japan", "osaka"), ("japan", "kyoto"), ("china", "beijing"),
    ("china", "shanghai"), ("south korea", "seoul"),
    ("mexico", "mexico city"), ("mexico", "cancun"),
    ("canada", "toronto"), ("canada", "vancouver"),
    ("canada", "montreal"), ("australia", "sydney"),
    ("australia", "melbourne"), ("egypt", "cairo"),
    ("india", "mumbai"), ("india", "delhi"), ("germany", "berlin"),
    ("germany", "munich"), ("netherlands", "amsterdam"),
    ("portugal", "lisbon"), ("ireland", "dublin"),
    ("greece", "athens"), ("peru", "lima"), ("peru", "cusco"),
    ("iceland", "reykjavik"), ("norway", "oslo"),
    ("sweden", "stockholm"), ("denmark", "copenhagen"),
    ("poland", "warsaw"), ("argentina", "buenos aires"),
    ("hawaii", "honolulu"), ("hawaii", "maui"),
    ("vermont", "burlington"), ("wisconsin", "milwaukee"),
    ("indiana", "indianapolis"),
)
_GEO_REL: Dict[str, frozenset] = {}
for _ga, _gb in _GEO_PAIRS:
    _GEO_REL.setdefault(_ga, set()).add(_gb)
    _GEO_REL.setdefault(_gb, set()).add(_ga)
_GEO_REL = {k: frozenset(v) for k, v in _GEO_REL.items()}
_GEO_SINGLE = frozenset(k for k in _GEO_REL if " " not in k)
# place -> container (pairs are (container, place)-ordered)
_GEO_CONTAINER: Dict[str, str] = {b: a for a, b in _GEO_PAIRS}
_GEO_PLACE_Q_RE = re.compile(
    r"\b(state|country|city|town|province|region|county|place)\b")


def _is_place(disp_l: str, dws_l: List[str]) -> bool:
    """Is this entity display a known place name?"""
    if disp_l in _GEO_REL:
        return True
    if len(dws_l) == 1 and dws_l[0] in _GEO_SINGLE:
        return True
    return any(w in _GEO_SINGLE for w in dws_l)


_UNSET = object()

MAP_GLOB = "unit-*.sessions.json"
LEGACY_MAP_FILE = "_verbatim_amb_map.json"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_budget() -> Optional[int]:
    raw = os.environ.get("VERBATIM_AMB_TOKEN_BUDGET", "")
    if raw.strip().lower() in ("unbounded", "none", "inf", "off"):
        return None
    try:
        return int(raw) if raw.strip() else DEFAULT_TOKEN_BUDGET
    except ValueError:
        return DEFAULT_TOKEN_BUDGET


def _env_concurrency() -> int:
    return _env_int(
        "VERBATIM_AMB_CONCURRENCY",
        _env_int("AMB_VERBATIM_CONCURRENCY", DEFAULT_CONCURRENCY),
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "off", "false", "no")


def _env_chat_format() -> bool:
    """``VERBATIM_AMB_FORMAT_CHAT`` (default on): parse ``_format_chat``-
    style prose (``[anchor | Turn n] Role: text`` blocks) into real
    turns so dialogue-format corpora take the turn-unit ingest path
    instead of the whole-document blob (V85-02.02 covers BEAM-shaped
    prose, not just JSON turn lists)."""
    return _env_flag("VERBATIM_AMB_FORMAT_CHAT", True)


def _env_operator() -> str:
    """``VERBATIM_AMB_OPERATOR``: eval-context quarantine policy —
    ``weak`` (default) releases holds whose findings are all weak-tier
    rules_v1 signals (benign-prose false positives), ``all`` releases
    every pending hold, ``off`` keeps every hold.  The release is a real
    review-surface decision (``decided_by=eval-amb-operator``) — labels
    and findings stay recorded; nothing is relabeled."""
    return os.environ.get(
        "VERBATIM_AMB_OPERATOR", "weak").strip().lower()


def _env_release_strong() -> bool:
    """``VERBATIM_AMB_RELEASE_STRONG`` (default off): extend the
    eval-context operator sweep to also release holds whose findings are
    all rules_v1-tiered signals (weak *or* strong tier) — the
    benign-prose strong-FP class (e.g. ``authority_claim.granted_access``
    on "you have full control").  Holds carrying untiered/non-rules_v1
    findings still need ``VERBATIM_AMB_OPERATOR=all``; this knob stays
    narrower than ``all`` on purpose."""
    return _env_flag("VERBATIM_AMB_RELEASE_STRONG", False)


def _env_admit() -> bool:
    """``VERBATIM_AMB_ADMIT`` (default on): after the drain, admit
    pending claim heads through ``Engine.apply_transition`` — the same
    operator sweep ``eval.v5.consolidation._admit_pending`` performs;
    the ``Memory.add`` consumer contract leaves claims ``pending`` by
    design (``admission.require_review``)."""
    return _env_flag("VERBATIM_AMB_ADMIT", True)


def provider_arm_fields() -> Dict[str, Any]:
    """Env-resolved arm knobs — the values a zero-arg provider would
    use (the AMB REGISTRY ctor takes no config).  The runner merges
    these into the run manifest's ``arm`` block (V85-02.07)."""
    return {
        "provider_revision": PROVIDER_VERSION,
        "token_budget": _env_budget(),
        "neighbor_w": _env_int(
            "VERBATIM_AMB_NEIGHBOR_W", DEFAULT_NEIGHBOR_W),
        "search_limit": _env_int(
            "VERBATIM_AMB_SEARCH_LIMIT", DEFAULT_SEARCH_LIMIT),
        "concurrency": _env_concurrency(),
        "max_turns_per_doc": _env_int("VERBATIM_AMB_MAX_TURNS_PER_DOC", 0),
        "token_meter": default_token_meter()[1],
        "ingest_path": "Memory.add(messages=[{speaker,text,at}],"
                       " session_id=<doc.id>, occurred_at=<doc.ts>)",
        "format_chat": _env_chat_format(),
        "quarantine_release": _env_operator(),
        "release_strong": _env_release_strong(),
        "admit_pending": _env_admit(),
        "expansion": "session_neighbors/v1",
    }


#: The three harness patches from ``research/v8_final_pack/
#: amb_patches.md`` — recorded beside the AMB commit in every manifest
#: (V85-02.07).  ``digest`` pins the patch document itself.
HARNESS_PATCHES = [
    {
        "id": "amb_patches#1",
        "file": "src/memory_bench/llm/openai.py",
        "change": (
            "generate() replaced: 6-attempt retry ladder, "
            "response_format downgrade json_schema→json_object→prompt "
            "on INVALID_REQUEST_BODY, special-token strip, first-JSON-"
            "object extraction"
        ),
    },
    {
        "id": "amb_patches#2",
        "file": "src/memory_bench/runner.py",
        "change": (
            "_process_one: 4-attempt per-query retry; final failure "
            "recorded as a failed QueryResult — one bad query never "
            "kills the run"
        ),
    },
    {
        "id": "amb_patches#3",
        "file": "src/memory_bench/memory/verbatim.py",
        "change": (
            "provider class attr concurrency = "
            "int(os.environ['AMB_VERBATIM_CONCURRENCY']) — the runner's "
            "asyncio.Semaphore knob"
        ),
    },
]


def harness_patch_records() -> Dict[str, Any]:
    """``{patches, doc, doc_sha256}`` for the manifest's ``amb`` block."""
    doc = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "research", "v8_final_pack", "amb_patches.md",
    )
    digest = None
    try:
        with open(doc, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        doc = None
    return {
        "patches": [dict(p) for p in HARNESS_PATCHES],
        "doc": doc,
        "doc_sha256": digest,
    }


# ---------------------------------------------------------------------------
# token meter — cl100k via tiktoken when installed, else the repo's
# pinned ``tok/v1`` estimator (SPEC: record the choice in the manifest)
# ---------------------------------------------------------------------------


def default_token_meter() -> Tuple[Callable[[str], int], str]:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")

        def _cl100k(text: Any) -> int:
            if text is None:
                return 0
            if isinstance(text, (bytes, bytearray)):
                text = bytes(text).decode("utf-8", "replace")
            if not isinstance(text, str):
                text = str(text)
            return len(enc.encode(text or ""))

        return _cl100k, "cl100k_base(tiktoken)"
    except Exception:
        from verbatim.retrieval.v7.pack import estimate_tokens

        return (
            lambda t: int(estimate_tokens(t)),
            "tok/v1 (verbatim.retrieval.v7.pack.estimate_tokens — "
            "tiktoken not installed)",
        )


# ---------------------------------------------------------------------------
# AMB document mirror
# ---------------------------------------------------------------------------


@dataclass
class AMBDoc:
    """Field-exact mirror of ``memory_bench.models.Document`` (pinned
    upstream).  Used whenever the real class is not importable; the
    upstream modes only read attributes, so the duck type is
    interchangeable inside the harness."""

    id: str
    content: str
    user_id: Optional[str] = None
    messages: Optional[list] = None
    timestamp: Optional[str] = None
    context: Optional[str] = None
    source_ids: Optional[list] = None
    tags: Optional[list] = None


_DOC_CLS_CACHE: Any = ...


def _doc_cls() -> type:
    """Real ``memory_bench.models.Document`` inside a pinned AMB
    checkout, else the local mirror — resolved once per process."""
    global _DOC_CLS_CACHE
    if _DOC_CLS_CACHE is ...:
        try:
            from memory_bench.models import Document as _Doc
            _DOC_CLS_CACHE = _Doc
        except Exception:
            _DOC_CLS_CACHE = AMBDoc
    return _DOC_CLS_CACHE


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """First present, non-None field across dict keys and attributes —
    the eval.v7 corpus-protocol convention."""
    if isinstance(obj, Mapping):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def parse_query_time(value: Any) -> Tuple[Any, Optional[int]]:
    """``(raw, µs|None)`` for an AMB ``query_timestamp`` — same tolerant
    contract as v1 (int µs/seconds, RFC3339 str, datetime); unparseable
    input keeps its raw value and ``None`` µs."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, None
    if isinstance(value, (int, float)):
        v = int(value)
        return value, (v if v >= 10**12 else v * 1_000_000)
    try:
        from datetime import datetime, timezone
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(
                tzinfo=timezone.utc)
            return value, int(dt.timestamp() * 1_000_000)
        if isinstance(value, str):
            s = value.strip()
            if s.isdigit():
                v = int(s)
                return value, (
                    v if v >= 10**12 else v * 1_000_000)
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return value, None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return value, int(dt.timestamp() * 1_000_000)
    except Exception:
        return value, None
    return value, None


def parse_object_ref(ref: Any) -> Optional[Tuple[str, str, Optional[int]]]:
    """``vobj1.<kind>.<hex(utf8 id)>.<rev|->`` → ``(kind, id, rev|None)``
    — the ``Hit.object_ref`` wire form (verbatim/memory/controls.py)."""
    if not isinstance(ref, str) or not ref.startswith("vobj1."):
        return None
    parts = ref.split(".")
    if len(parts) not in (3, 4):
        return None
    try:
        oid = bytes.fromhex(parts[2]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    rev: Optional[int] = None
    if len(parts) == 4 and parts[3] != "-":
        try:
            rev = int(parts[3])
        except ValueError:
            return None
    return parts[1], oid, rev


def _accepts_param(fn: Any, name: str) -> bool:
    """True when ``fn`` takes ``name`` as a parameter or accepts
    ``**kwargs`` — the ``messages``/``as_of`` feature-detect."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def _unit_key(user_id: Any) -> str:
    return str(user_id) if user_id not in (None, "") else "_shared"


def _bank_label(unit_key: str) -> str:
    """``Memory(user_id=...)`` label for a bank — sanitized to the
    alias-label charset, deterministic per unit."""
    if unit_key == "_shared":
        return "amb"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", unit_key)[:48] or "u"
    return f"amb-{cleaned}"


def _bank_filename(unit_key: str) -> str:
    digest = hashlib.sha256(unit_key.encode("utf-8")).hexdigest()[:16]
    return f"unit-{digest}.db"


def _index_filename(unit_key: str) -> str:
    digest = hashlib.sha256(unit_key.encode("utf-8")).hexdigest()[:16]
    return f"unit-{digest}.sessions.json"


# ---------------------------------------------------------------------------
# drain env (eval.v5.harness.drain_memory convention — the consumer
# route's 8-line protocol shim)
# ---------------------------------------------------------------------------


class _DrainEnv:
    def __init__(self, *, worker: str, memory: Any, notes: List[str]) -> None:
        self.worker = worker
        self.memory = memory
        self.notes = notes
        self.drain: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# the provider
# ---------------------------------------------------------------------------


class VerbatimAMBProvider:
    """``verbatim`` — the real ``Memory`` write + read path behind the
    AMB ``MemoryProvider`` surface (V8-15.06 carried, V85-02).

    ``memory_factory(path, unit_key) -> Memory-like`` builds a bank; the
    default constructs ``verbatim.Memory`` on disk under ``prepare()``'s
    ``store_dir``.  Tests inject spy factories — construction kwargs
    therefore flow only through this seam.
    """

    # -- AMB class attrs (base.py contract) ------------------------------
    name = PROVIDER_NAME
    description = (
        "Verbatim governed memory (local SQLite store; model-free write "
        "path; session-excerpt context at a pinned token budget)"
    )
    kind = "local"
    provider = "verbatim"
    variant = "local"
    #: Runner reads this attr for its asyncio.Semaphore.  Retrieval runs
    #: on the store's per-thread reader connections; only writes take
    #: the lock, so parallelism is safe (V85-02.06).
    concurrency = _env_concurrency()
    #: AMB tag-group filters have no verbatim lane — declared honestly
    #: so the harness never passes ``filters``.
    supports_filters = False

    def __init__(
        self,
        *,
        store_dir: Optional[Any] = None,
        token_budget: Any = _UNSET,
        neighbor_w: Any = _UNSET,
        search_limit: Any = _UNSET,
        concurrency: Any = _UNSET,
        timeout_ms: Any = _UNSET,
        doc_mode: str = "pack",
        infer: bool = True,
        encoder: str = "hashing",
        worker: str = "external",
        user_id: str = "amb",
        settle_timeout_s: float = 120.0,
        memory_factory: Optional[Callable] = None,
        memory_kwargs: Optional[Dict[str, Any]] = None,
        token_meter: Optional[Callable[[str], int]] = None,
    ) -> None:
        if doc_mode not in DOC_MODES:
            raise ValueError(
                f"doc_mode must be one of {DOC_MODES}, got {doc_mode!r}"
            )
        # ``VERBATIM_AMB_SEARCH_TIMEOUT_MS`` overrides the 500ms product
        # deadline — at concurrency>1 lanes hit slice deadlines under
        # CPU contention and return timing-dependent PARTIAL tails, so
        # eval reruns drift.  Raise it for deterministic measurement
        # runs; the default stays product-faithful.  <=0 maps to a
        # 24-hour budget (effectively unbounded — ``inf`` would break
        # the facade's ``int(remaining_ms)`` casts downstream).
        if timeout_ms is _UNSET:
            timeout_ms = _env_int(
                "VERBATIM_AMB_SEARCH_TIMEOUT_MS", int(DEFAULT_TIMEOUT_MS))
        if timeout_ms <= 0:
            timeout_ms = 86_400_000.0
        if not isinstance(timeout_ms, (int, float)) or timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative number")
        self._store_dir = Path(store_dir) if store_dir is not None else None
        # ``_UNSET`` → env → default; an explicit None means unbounded.
        self.token_budget = (
            _env_budget() if token_budget is _UNSET else token_budget
        )
        if self.token_budget is not None:
            self.token_budget = int(self.token_budget)
            if self.token_budget < 0:
                raise ValueError("token_budget must be non-negative")
        self.neighbor_w = (
            _env_int("VERBATIM_AMB_NEIGHBOR_W", DEFAULT_NEIGHBOR_W)
            if neighbor_w is _UNSET else int(neighbor_w)
        )
        if self.neighbor_w < 0:
            raise ValueError("neighbor_w must be non-negative")
        self.search_limit = (
            _env_int("VERBATIM_AMB_SEARCH_LIMIT", DEFAULT_SEARCH_LIMIT)
            if search_limit is _UNSET else int(search_limit)
        )
        if self.search_limit < 1:
            raise ValueError("search_limit must be positive")
        if concurrency is not _UNSET:
            self.concurrency = int(concurrency)
        self.timeout_ms = float(timeout_ms)
        self.doc_mode = doc_mode
        self.infer = bool(infer)
        self.encoder = encoder
        self.worker = worker
        self.user_id = user_id
        self.settle_timeout_s = float(settle_timeout_s)
        self._memory_factory = memory_factory or self._default_factory
        self._memory_kwargs = dict(memory_kwargs or {})
        if token_meter is not None:
            self._meter = token_meter
            self._meter_name = "custom"
        else:
            self._meter, self._meter_name = default_token_meter()

        #: writes (ingest + index mutations) serialize on this lock;
        #: reads never take it (V85-02.06).
        self._write_lock = threading.Lock()
        #: engine ``Memory.search`` is not reentrant — concurrent calls
        #: return differently-ordered hit lists (the fused lane merge
        #: mutates shared state), which made every concurrent eval run
        #: nondeterministic.  All search calls go through this lock.
        self._search_lock = threading.Lock()
        #: tiny lock for lazy session-index mutations during retrieve
        #: (unit-hit cache fills) — reads of the index are GIL-atomic
        #: dict gets and never block.
        self._idx_lock = threading.Lock()
        self._rec_lock = threading.Lock()
        self._prepared = False
        self._unit_ids: Optional[List[str]] = None
        self._banks: Dict[str, Any] = {}
        self._bank_paths: Dict[str, str] = {}
        self._bank_docs: Dict[str, List[str]] = {}
        #: bank_key -> SessionIndex (persisted as
        #: ``unit-<hash>.sessions.json`` next to the bank store).
        self._indexes: Dict[str, SessionIndex] = {}
        self._queries: List[Dict[str, Any]] = []
        self._extraction_labels: List[dict] = []
        self.notes: List[str] = []
        self.last_ingest: Dict[str, Any] = {}

    # -- construction seams ------------------------------------------------

    def _default_factory(self, path: str, unit_key: str) -> Any:
        """Real ``verbatim.Memory`` bank — product defaults except the
        eval-declared worker/encoder (same surface VerbatimArm uses)."""
        from verbatim import Memory

        return Memory(
            path,
            user_id=(
                _bank_label(unit_key)
                if unit_key != "_shared" else self.user_id
            ),
            worker=self.worker,
            encoder=self.encoder,
            **self._memory_kwargs,
        )

    # -- AMB lifecycle hooks -----------------------------------------------

    def initialize(self) -> None:
        """No external processes — banks open lazily at first use."""

    def set_extraction_labels(self, labels: Optional[list]) -> None:
        """Optional runner hook (hasattr-guarded upstream); stored for
        inspection only — verbatim has no entity-label concept."""
        self._extraction_labels = list(labels or [])

    def prepare(
        self,
        store_dir: Any,
        unit_ids: Optional[Iterable[str]] = None,
        reset: bool = True,
    ) -> None:
        """AMB ``prepare``: bind the persistent store dir; ``reset``
        clears prior banks + session indexes (fresh run) while
        ``reset=False`` reloads the persisted sidecars for a resume run
        (``--skip-ingestion``)."""
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            for key, mem in list(self._banks.items()):
                try:
                    mem.close()
                except Exception:
                    pass
            self._banks.clear()
            self._indexes.clear()
            for path in list(self._bank_paths.values()):
                # the store file plus SQLite sidecars (a half-deleted
                # bank is worse than a stale one)
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        os.unlink(path + suffix)
                    except OSError:
                        pass
            self._bank_paths.clear()
            self._bank_docs.clear()
            # banks left by an earlier provider instance on this dir —
            # reset must be total, not just for banks this process opened
            for p in self._store_dir.glob("unit-*.db*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            for p in self._store_dir.glob(MAP_GLOB):
                try:
                    p.unlink()
                except OSError:
                    pass
            legacy = self._store_dir / LEGACY_MAP_FILE
            if legacy.exists():
                try:
                    legacy.unlink()
                except OSError:
                    pass
        else:
            self._load_indexes()
        self._unit_ids = (
            sorted(str(u) for u in unit_ids) if unit_ids else None
        )
        self._prepared = True

    def cleanup(self) -> None:
        """Close every bank (``cleanup`` is AMB's inverse-of-initialize
        hook; ``close`` aliases it for direct use)."""
        for mem in list(self._banks.values()):
            try:
                mem.close()
            except Exception:
                pass
        self._banks.clear()

    close = cleanup

    # -- banks + session indexes ---------------------------------------------

    def _bank(self, user_id: Any) -> Any:
        return self._banks.get(_unit_key(user_id))

    def _index(self, unit_key: str) -> SessionIndex:
        idx = self._indexes.get(unit_key)
        if idx is None:
            idx = SessionIndex(unit_key)
            self._indexes[unit_key] = idx
        return idx

    def _bank_or_create(self, user_id: Any) -> Any:
        key = _unit_key(user_id)
        mem = self._banks.get(key)
        if mem is None:
            if self._store_dir is None:
                raise RuntimeError(
                    "verbatim provider: prepare(store_dir) has not run"
                )
            path = str(self._store_dir / _bank_filename(key))
            mem = self._memory_factory(path, key)
            self._banks[key] = mem
            self._bank_paths[key] = path
        return mem

    def banks(self) -> Dict[str, Any]:
        return dict(self._banks)

    # -- ingest (V85-02.02: turn-level Memory.add path) -------------------

    def ingest(self, documents: Iterable[Any]) -> Dict[str, Any]:
        """``Memory.add`` per AMB session document — turn-list payloads
        go through ``messages=[{speaker, text, at}]`` (one turn unit per
        message, real ``seq``); prose documents keep the single-add
        path.  Settle = durable-queue drain + ``wait_ready`` frontier —
        no fixture shortcuts."""
        from eval.v5.harness import add_with_retry, drain_memory

        if self._store_dir is None:
            raise RuntimeError(
                "verbatim provider: prepare(store_dir) has not run"
            )
        docs = list(documents or [])
        t0 = time.perf_counter()
        touched: Dict[str, Tuple[Any, Any]] = {}
        errors: Dict[str, str] = {}
        n_turn_docs = 0
        n_blob_docs = 0
        n_turns = 0

        with self._write_lock:
            for i, d in enumerate(docs):
                doc_id = str(_get(d, "id", "doc_id", "ref",
                                  default=f"doc-{i:05d}"))
                user_id = _get(d, "user_id")
                bkey = _unit_key(user_id)
                try:
                    mem = self._bank_or_create(user_id)
                except Exception as exc:  # noqa: BLE001 — recorded, visible
                    errors[doc_id] = f"bank: {type(exc).__name__}: {exc}"
                    continue
                index = self._index(bkey)
                try:
                    res = self._ingest_doc(
                        mem, index, d, doc_id, user_id, add_with_retry
                    )
                except Exception as exc:  # noqa: BLE001 — per-doc error,
                    # recorded honestly, the batch continues
                    errors[doc_id] = f"{type(exc).__name__}: {exc}"
                    continue
                if res.get("turns"):
                    n_turn_docs += 1
                    n_turns += int(res["turns"])
                else:
                    n_blob_docs += 1
                if res.get("last") is not None:
                    touched[bkey] = (mem, res["last"])
                if res.get("added"):
                    self._bank_docs.setdefault(bkey, []).append(doc_id)

            # settle: external worker drains the durable queue per bank;
            # the readiness frontier is confirmed through wait_ready —
            # never assumed (VerbatimArm.ingest convention).
            drains: Dict[str, Any] = {}
            settles: Dict[str, Any] = {}
            operator: Dict[str, Any] = {}
            for bkey, (mem, last) in sorted(touched.items()):
                env = _DrainEnv(worker=self.worker, memory=mem,
                                notes=self.notes)
                drains[bkey] = dict(drain_memory(env))
                # eval-context operator: release weak-tier quarantine
                # holds (re-drive their jobs, drain again) and admit
                # pending claim heads — same review-surface semantics
                # the eval suites run themselves.
                op = self._operator_pass(mem, env)
                if op:
                    operator[bkey] = op
                    drains[bkey] = dict(env.drain)
                settle: Dict[str, Any] = {"state": "no_receipts"}
                if last is not None and hasattr(mem, "wait_ready"):
                    deadline = time.monotonic() + self.settle_timeout_s
                    state = "pending"
                    while True:
                        try:
                            rd = mem.wait_ready(last, timeout_ms=2000)
                            state = str(getattr(rd, "state", "unknown"))
                        except Exception as exc:  # noqa: BLE001
                            state = f"error:{type(exc).__name__}"
                        if state in ("ready", "blocked", "unavailable") or (
                            time.monotonic() > deadline
                        ):
                            break
                        time.sleep(0.05)
                    settle = {
                        "state": state,
                        "waited_s": round(
                            self.settle_timeout_s
                            - max(0.0, deadline - time.monotonic()),
                            3,
                        ),
                    }
                settles[bkey] = settle

            # unit attach: learn the projected turn unit_ids / byte pins
            # so a hit resolves to a session ordinal without trusting
            # engine internals (V85-02.02 session index).
            attach_notes = 0
            for bkey, (mem, _last) in sorted(touched.items()):
                index = self._indexes.get(bkey)
                if index is None:
                    continue
                attach_notes += self._attach_bank_units(
                    mem, index, self._bank_docs.get(bkey, ()))
            for idx in self._indexes.values():
                self.notes.extend(idx.notes)
                idx.notes.clear()
            self._save_indexes()

        self.last_ingest = {
            "documents": len(docs),
            "indexed": sum(len(v) for v in self._bank_docs.values()),
            "add_errors": len(errors),
            "add_error_detail": dict(errors),
            "turn_documents": n_turn_docs,
            "blob_documents": n_blob_docs,
            "turns_indexed": n_turns,
            "banks": sorted(touched),
            "drain": drains,
            "settle": settles,
            "operator": operator,
            "ingest_path": "messages" if n_turn_docs else "add",
            "ingest_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return dict(self.last_ingest)

    def _ingest_doc(
        self,
        mem: Any,
        index: SessionIndex,
        d: Any,
        doc_id: str,
        user_id: Any,
        add_with_retry: Callable,
    ) -> Dict[str, Any]:
        """One AMB document → session record + ``Memory.add`` calls."""
        content = str(_get(d, "content", "text", default="") or "")
        ts = _get(d, "timestamp", "when", "occurred_at")
        ctx = _get(d, "context")
        tags = _get(d, "tags")
        meta: Dict[str, Any] = {
            "amb_doc_id": doc_id,
            "amb_user_id": (None if user_id is None else str(user_id)),
            "amb_provider": PROVIDER_VERSION,
        }
        if ctx is not None:
            meta["amb_context"] = str(ctx)
        if tags is not None:
            meta["amb_tags"] = list(tags) if isinstance(
                tags, (list, tuple)) else tags
        if ts is not None:
            meta["amb_timestamp"] = str(ts)

        # turn extraction: structured ``messages`` first, then the
        # ``json.dumps``-ed turn list in ``content`` (the AMB LoCoMo
        # shape), then ``_format_chat``-style prose (BEAM: ``[anchor |
        # Turn n] Role: text`` blocks).  None of the three → prose
        # document (legitimate blob path).
        msgs = _get(d, "messages")
        turns = None
        turn_source = None
        if isinstance(msgs, (list, tuple)) and msgs:
            turns = turns_from(list(msgs))
            if turns is not None:
                turn_source = "messages"
        if turns is None:
            turns = parse_turn_list(content)
            if turns is not None:
                turn_source = "json"
        if turns is None and _env_chat_format():
            turns = parse_chat_lines(content)
            if turns is not None:
                turn_source = "chat_lines"
        if turn_source is not None:
            meta["amb_turn_source"] = turn_source

        sess = index.add_session(
            doc_id,
            turns if turns is not None else
            [{"speaker": None, "text": content, "at": ts,
              "dia_id": None, "caption": None, "raw": {}}],
            user_id=(None if user_id is None else str(user_id)),
            timestamp=(None if ts is None else str(ts)),
            context=(None if ctx is None else str(ctx)),
        )

        # AMB filter contract: caller-supplied Document.tags (e.g.
        # ``scope:domain-code``) plus LLM-derived extraction-label tags
        # (``state:*`` / ``name:*`` when VERBATIM_AMB_TAGS is set) are
        # persisted on the session record so retrieve() can evaluate
        # ``filters`` without touching the engine store.
        if tags is not None:
            sess["amb_tags"] = (
                [str(t) for t in tags]
                if isinstance(tags, (list, tuple)) else [str(tags)])
        if _env_int("VERBATIM_AMB_TAGS", 0) and self._extraction_labels:
            label_text = content
            if not label_text.strip() and turns:
                label_text = " ".join(
                    str(t.get("text") or "")
                    for t in turns if isinstance(t, dict))
            try:
                sess["label_tags"] = self._label_tags(label_text)
            except Exception as exc:  # noqa: BLE001 — recorded, not
                # swallowed: a failed tagger must never corrupt ingest
                sess["label_tags_err"] = f"{type(exc).__name__}: {exc}"
            # canonical-name recovery: the doc text's leading
            # snake_case token is the caller's canonical name —
            # normalize it into a ``name:`` tag so coverage never rests
            # on the extractor model alone.
            head = (content.split() or [""])[0]
            if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", head):
                sess.setdefault("label_tags", [])
                ws = self._tag_norm(head).split()
                variants = [" ".join(ws)]
                # a 3+ word canonical decomposes into its 2-3 word
                # sub-runs ('error_handling_explicit' → 'error
                # handling') — never into lone words.
                for w in (2, 3):
                    if w < len(ws):
                        variants += [" ".join(ws[i:i + w])
                                     for i in range(len(ws) - w + 1)]
                for v in variants:
                    cn = f"name:{v}"
                    if cn not in sess["label_tags"]:
                        sess["label_tags"].append(cn)
            # name-region sub-runs: the text's opening tokens (before
            # the first sentence boundary) are the caller's alias list
            # written out — every contiguous 2-3 word span inside it is
            # a lookup form ('auth … session backend …' carries
            # 'session backend').  Single words never emit: a lone noun
            # must come from the extractor or the canonical, or a
            # generic word like 'chapter' would detach from 'chapter 7'.
            region = re.split(r"[.\n!?;:]", content, 1)[0]
            raw_toks = region.split()[:16]
            rws = self._tag_norm(region).split()[:16]
            if len(rws) >= 2:
                sess.setdefault("label_tags", [])
                stop = VerbatimAMBProvider._STOPWORDS
                for w in (2, 3):
                    if w > len(rws):
                        continue
                    for i in range(len(rws) - w + 1):
                        run = rws[i:i + w]
                        # skip pure-stopword spans ('in the') and runs
                        # ending on a function word ('redis for') — they
                        # collide with ordinary phrasing.
                        if run[-1] in stop or all(
                                x in stop for x in run):
                            continue
                        rn = f"name:{' '.join(run)}"
                        if rn not in sess["label_tags"]:
                            sess["label_tags"].append(rn)
                # lone-word emission only for STYLIZED tokens — interior
                # capitals or digits ('ReactJS', 'MongoDB', 'k8s'), the
                # identifiers people actually type.  Plain words like
                # 'chapter' stay compound-only.
                stylized = set()
                for tok in raw_toks:
                    tn = self._tag_norm(tok)
                    if (tn not in stop
                            and (len(tn) >= 3
                                 or (len(tn) == 2
                                     and any(c.isdigit() for c in tn)))
                            and (any(c.isupper() for c in tok[1:])
                                 or any(c.isdigit() for c in tn))):
                        stylized.add(tn)
                        rn = f"name:{tn}"
                        if rn not in sess["label_tags"]:
                            sess["label_tags"].append(rn)
                # canonical-head trap: a bare single word that is just
                # the head of a ≥3-word canonical name is a shared
                # concept, not a name ('composition' must not resolve
                # to 'composition_over_inheritance').  Kept when the
                # caller wrote it stylized ('MongoDB' → 'mongodb').
                head = (content.split() or [""])[0]
                if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", head):
                    cws = self._tag_norm(head).split()
                    if len(cws) >= 3:
                        trap = f"name:{cws[0]}"
                        if cws[0] not in stylized:
                            sess["label_tags"] = [
                                t for t in sess["label_tags"]
                                if t != trap]

        # portable idempotency tokens (facade ``_key_ok`` charset) —
        # doc ids with other characters are folded deterministically.
        idem_doc = re.sub(r"[^A-Za-z0-9_.:-]", "_", doc_id)[:200]

        last = None
        added = False
        if turns is not None and _accepts_param(mem.add, "messages"):
            # V85-02.02 primary path: ONE add per session document.
            payload = transcript_for_add(turns)
            if content.strip() and parse_turn_list(content) is None:
                # prose ``content`` alongside a structured ``messages``
                # field — preserved in the payload so it stays
                # searchable; turn byte pins still resolve verbatim.
                payload = payload + "\n\n" + content
            msgs_for_add = messages_for_add(turns)
            chunks = [
                msgs_for_add[o:o + _MAX_MESSAGES_PER_ADD]
                for o in range(0, len(msgs_for_add),
                               _MAX_MESSAGES_PER_ADD)
            ]
            for ci, chunk in enumerate(chunks):
                kw = {
                    "infer": self.infer,
                    "metadata": meta,
                    "messages": chunk,
                    "session_id": doc_id,
                    "occurred_at": ts,
                    "idempotency_key": (
                        f"amb:{idem_doc}" if len(chunks) == 1
                        else f"amb:{idem_doc}_c{ci:03d}"
                    ),
                }
                res = add_with_retry(mem, payload, **kw)
                last = res
                sid = str(getattr(res, "memory_id", "") or "")
                if not sid:
                    sid = self._result_source_id(res)
                if sid:
                    index.attach_source(doc_id, sid)
                    added = True
        elif turns is not None:
            # fallback: facade without ``messages`` — per-turn adds
            # sharing ``session_id=<doc.id>`` (still turn-level memory,
            # never a whole-document blob).
            for ti, t in enumerate(turns):
                tmeta = dict(meta)
                tmeta["amb_turn"] = ti
                if t.get("dia_id"):
                    tmeta["amb_dia_id"] = t["dia_id"]
                kw = {
                    "infer": self.infer,
                    "metadata": tmeta,
                    "session_id": doc_id,
                    "occurred_at": t.get("at") if t.get(
                        "at") is not None else ts,
                    "idempotency_key": f"amb:{idem_doc}_t{ti:04d}",
                }
                if t.get("speaker"):
                    kw["speaker"] = t["speaker"]
                res = add_with_retry(mem, t["text"], **kw)
                last = res
                sid = str(getattr(res, "memory_id", "") or "")
                if not sid:
                    sid = self._result_source_id(res)
                if sid:
                    index.attach_source(doc_id, sid)
                    added = True
        else:
            # genuine prose document — the single-add path (the blob
            # path is forbidden only for turn-list payloads).
            kw = {
                "infer": self.infer,
                "metadata": meta,
                "occurred_at": ts,
                "session_id": doc_id,
                "idempotency_key": f"amb:{idem_doc}",
            }
            res = add_with_retry(mem, content, **kw)
            last = res
            sid = str(getattr(res, "memory_id", "") or "")
            if not sid:
                sid = self._result_source_id(res)
            if sid:
                index.attach_source(doc_id, sid)
                added = True

        sess["ingest_path"] = (
            "messages" if turns is not None and _accepts_param(
                mem.add, "messages")
            else ("per_turn" if turns is not None else "blob")
        )
        return {
            "turns": len(turns) if turns is not None else 0,
            "last": last,
            "added": added,
        }

    @staticmethod
    def _result_source_id(res: Any) -> str:
        """``AddResult.ref`` → source_id when ``memory_id`` is absent."""
        try:
            from verbatim.memory.types import MemoryRef

            return MemoryRef.parse(
                getattr(res, "ref", "") or "").source_id
        except Exception:
            return ""

    # -- eval-context operator pass ---------------------------------------

    @staticmethod
    def _quarantined_source(conn: Any, row: Mapping) -> Optional[str]:
        """Quarantine row → its ``source_id`` (envelope/span refs resolve
        through their owning tables)."""
        kind = str(row.get("object_kind") or "")
        oid = row.get("object_id")
        rev = row.get("revision")
        try:
            if kind == "source_envelope":
                r = conn.execute(
                    "SELECT source_id FROM source_envelopes"
                    " WHERE envelope_id = ? AND revision = ?",
                    (oid, rev)).fetchone()
                return str(r[0]) if r else None
            if kind == "source":
                return str(oid) if oid else None
            if kind == "span":
                r = conn.execute(
                    "SELECT source_id FROM spans WHERE span_id = ?",
                    (oid,)).fetchone()
                return str(r[0]) if r else None
        except Exception:
            return None
        return None

    def _operator_pass(self, mem: Any, env: Any) -> Dict[str, Any]:
        """Eval-context operator for a drained bank.

        (a) ``VERBATIM_AMB_OPERATOR`` — release quarantine holds whose
        findings are all weak-tier rules_v1 signals (the benign-prose
        false-positive class, e.g. ``role_claim.roleplay`` on "behave
        as"); the release is a real ``quarantine.release`` decision so
        labels/findings stay on record.  Released sources get their
        durable jobs re-enqueued (harvest + the source-projection
        branch — a dead job's dedup key vacates for the fresh attempt)
        and the queue drains again.
        (b) ``VERBATIM_AMB_ADMIT`` — admit every pending claim head via
        ``Engine.apply_transition`` (the eval suites' own admission
        sweep; ``admission.require_review`` mints claims ``pending`` by
        design).
        """
        from eval.v5.harness import drain_memory

        mode = _env_operator()
        release_strong = _env_release_strong()
        admit = _env_admit()
        out: Dict[str, Any] = {
            "mode": mode, "release_strong": release_strong, "admit": admit,
            "released": 0, "held": 0, "requeued": 0,
            "released_reasons": {},
            "admitted": 0, "pending_heads": 0,
            "errors": [],
        }
        if mode in ("off", "0", "false", "no") and not admit:
            return {}
        store = getattr(mem, "_store", None)
        cfg = getattr(mem, "_cfg", None)
        if store is None:
            return {"error": "bank carries no _store"}

        released_sources: List[Tuple[str, int]] = []
        if mode not in ("off", "0", "false", "no"):
            try:
                from verbatim.security.quarantine import (
                    pending_items, release,
                )
                with store.tx() as conn:
                    for row in pending_items(conn, limit=10000):
                        findings = row.get("findings") or []
                        tiers = [
                            str(f.get("tier")) for f in findings
                        ]
                        weak_only = bool(findings) and all(
                            t == "weak" for t in tiers
                        )
                        # VERBATIM_AMB_RELEASE_STRONG: every finding is a
                        # rules_v1-tiered signal (weak|strong) — still
                        # narrower than mode=all (untiered/empty findings
                        # stay held).
                        rules_tiered = bool(findings) and all(
                            t in ("weak", "strong") for t in tiers
                        )
                        if not (mode == "all" or weak_only
                                or (release_strong and rules_tiered)):
                            out["held"] += 1
                            continue
                        release(
                            conn,
                            (row["object_kind"], row["object_id"],
                             int(row["revision"])),
                            decided_by="eval-amb-operator",
                            decision={
                                "context": "amb-eval",
                                "reason": (
                                    "rules_v1-tiered finding (incl. "
                                    "strong) on benign benchmark prose"
                                    if not weak_only else
                                    "weak-tier rules_v1 finding "
                                    "on benign benchmark prose"),
                                "reason_codes": list(
                                    row.get("reason_codes") or []),
                            },
                            scope_id=row.get("scope_id"),
                        )
                        out["released"] += 1
                        for rc in row.get("reason_codes") or []:
                            out["released_reasons"][rc] = (
                                out["released_reasons"].get(rc, 0) + 1)
                        sid = self._quarantined_source(conn, row)
                        if sid is not None:
                            released_sources.append(
                                (sid, int(row["revision"])))
            except Exception as exc:  # noqa: BLE001 — recorded, visible
                out["errors"].append(
                    f"release: {type(exc).__name__}: {exc}")
            if released_sources:
                try:
                    from verbatim.core.types import JobKind
                    from verbatim.ingest import Ingester
                    from verbatim.jobs.source_jobs import (
                        enqueue_source_jobs,
                    )
                    ing = Ingester(
                        store, cfg,
                        encoder=getattr(mem, "_encoder", None))
                    with store.tx() as conn:
                        for sid, rev in released_sources:
                            srow = conn.execute(
                                "SELECT scope_id FROM sources"
                                " WHERE source_id = ?", (sid,)
                            ).fetchone()
                            if srow is None:
                                out["errors"].append(
                                    f"requeue: no source row {sid}")
                                continue
                            op_key = f"harvest:{sid}:{rev}"
                            ing.jobs.enqueue(
                                conn, srow[0], JobKind.HARVEST,
                                {"source_id": sid, "revision": rev},
                                dedup_key=store.hmac(op_key.encode()),
                                operation_key=(
                                    op_key
                                    if ing.jobs.supports_durability
                                    else None),
                            )
                            enqueue_source_jobs(
                                conn, store,
                                receipt_id=f"rc_ingest:{sid}:{rev}",
                                queue=ing.jobs,
                            )
                            out["requeued"] += 1
                except Exception as exc:  # noqa: BLE001
                    out["errors"].append(
                        f"requeue: {type(exc).__name__}: {exc}")
                drain_memory(env)

        if admit:
            try:
                from eval.v5.consolidation import _admit_pending
                rep = _admit_pending(env)
                out["pending_heads"] = int(rep.get("pending_heads", 0))
                out["admitted"] = int(rep.get("admitted", 0))
                if rep.get("failed"):
                    out["admit_failed"] = list(rep["failed"])[:8]
                if rep.get("error"):
                    out["errors"].append(f"admit: {rep['error']}")
            except Exception as exc:  # noqa: BLE001
                out["errors"].append(
                    f"admit: {type(exc).__name__}: {exc}")
            if out["admitted"]:
                drain_memory(env)
        return out

    def _attach_bank_units(
        self, mem: Any, index: SessionIndex, doc_ids: Iterable[str]
    ) -> int:
        """Read the projected turn units for each session's sources and
        fill ``unit_id``/``seq``/byte pins into the session index."""
        store = getattr(mem, "_store", None)
        if store is None:
            return 0
        try:
            generation = store.projection_generation()
        except Exception:
            generation = None
        try:
            from verbatim.storage.repos import has_table
            with store.read() as conn:
                if not has_table(conn, "units"):
                    return 0
                for doc_id in doc_ids:
                    sess = index.session(doc_id)
                    if sess is None:
                        continue
                    for sid in sess["source_ids"]:
                        rows = [
                            r for r in conn.execute(
                                "SELECT unit_id, seq, byte_start,"
                                " byte_end, source_id, generation"
                                " FROM units WHERE source_id = ?"
                                " AND kind = 'turn' AND generation <= ?"
                                " ORDER BY generation DESC, seq,"
                                " byte_start, unit_id",
                                (sid, generation if generation is not
                                 None else 1 << 62),
                            )
                        ]
                        if not rows:
                            continue
                        top = max(r[5] for r in rows)
                        rows = [r for r in rows if r[5] == top]
                        index.attach_units(
                            doc_id,
                            [
                                {
                                    "unit_id": r[0], "seq": r[1],
                                    "byte_start": r[2],
                                    "byte_end": r[3],
                                    "source_id": r[4],
                                }
                                for r in rows
                            ],
                            generation=top,
                            source_id=sid,
                        )
        except Exception as exc:  # noqa: BLE001 — honest note, not a
            # silent gap: expansion then runs on source/quote fallbacks
            self.notes.append(
                f"unit attach failed: {type(exc).__name__}: {exc}")
            return 1
        return 0

    # -- index persistence (resume runs) ---------------------------------

    def _save_indexes(self) -> None:
        if self._store_dir is None:
            return
        for bkey, idx in self._indexes.items():
            path = self._store_dir / _index_filename(bkey)
            try:
                idx.save(path)
            except OSError as exc:
                self.notes.append(
                    f"session index persist failed ({bkey}): {exc}")

    def _load_indexes(self) -> None:
        if self._store_dir is None:
            return
        for p in sorted(self._store_dir.glob(MAP_GLOB)):
            idx = SessionIndex.load(p)
            if idx is not None:
                self._indexes[idx.bank_key] = idx

    # -- retrieve (V85-02.03/02.04/02.05) --------------------------------

    def _search_kwargs(
        self, mem: Any, query_timestamp: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """``Memory.search`` kwargs + the honest as_of record — the pin
        goes on the wire only when the dataset supplied a time AND the
        bound facade accepts the parameter (feature-detected)."""
        kwargs: Dict[str, Any] = {
            "limit": int(self.search_limit),
            "timeout_ms": self.timeout_ms,
        }
        rec: Dict[str, Any] = {"requested": query_timestamp}
        if query_timestamp is None:
            rec["applied"] = None
            return kwargs, rec
        if _accepts_param(mem.search, "as_of"):
            kwargs["as_of"] = query_timestamp
            rec["applied"] = True
        else:
            rec["applied"] = False
            rec["reason"] = (
                "Memory.search has no as_of parameter at this head"
            )
        return kwargs, rec

    def _unit_row(self, mem: Any, unit_id: str) -> Optional[Dict[str, Any]]:
        """Generation-fenced ``units`` read for lazy hit resolution —
        the same snapshot discipline as the facade's hit-mapper."""
        store = getattr(mem, "_store", None)
        if store is None:
            return None
        cols = (
            "unit_id", "source_id", "revision", "kind", "session_id",
            "seq", "speaker_canon", "byte_start", "byte_end",
            "parent_unit_id", "generation",
        )
        try:
            generation = store.projection_generation()
            from verbatim.storage.repos import has_table

            with store.read() as conn:
                if not has_table(conn, "units"):
                    return None
                r = conn.execute(
                    "SELECT " + ", ".join(cols) +
                    " FROM units WHERE unit_id = ? AND generation <= ?"
                    " ORDER BY generation DESC LIMIT 1",
                    (unit_id, generation),
                ).fetchone()
            return dict(zip(cols, r)) if r else None
        except Exception:
            return None

    def _claim_source(self, mem: Any, claim_id: str) -> Optional[str]:
        """claim hit → evidence span → source_id (the eval.v7
        source-mapping convention)."""
        store = getattr(mem, "_store", None)
        if store is None:
            return None
        try:
            from verbatim.storage.repos import has_table
            with store.read() as conn:
                if not (has_table(conn, "claim_evidence")
                        and has_table(conn, "spans")):
                    return None
                row = conn.execute(
                    "SELECT s.source_id FROM claim_evidence ce"
                    " JOIN spans s ON s.span_id = ce.span_id"
                    " WHERE ce.claim_id = ? LIMIT 1",
                    (claim_id,),
                ).fetchone()
            return str(row[0]) if row and row[0] else None
        except Exception:
            return None

    def _resolve_hit(
        self, mem: Any, index: SessionIndex, hit: Any
    ) -> Optional[Dict[str, Any]]:
        """Delivered hit → ``{doc_id, ordinals, method}`` or ``None``
        (a stray rendered as itself, never silently dropped).

        Ladder: ``vobj1.unit`` → session-index unit map → lazy units-row
        read (byte-range containment for session/episode/window units,
        ``seq`` for turns, quote containment last); ``vobj1.claim`` →
        ``claim_evidence`` → source → session; ``vobj1.source``/bare
        ``memory_id`` → source → session (quote match → ordinal 0)."""
        parsed = parse_object_ref(getattr(hit, "object_ref", ""))
        quote = getattr(hit, "quote", "") or ""
        if isinstance(quote, (bytes, bytearray)):
            quote = bytes(quote).decode("utf-8", "replace")

        if parsed is not None and parsed[0] == "unit":
            uid = parsed[1]
            got = index.resolve_unit(uid)
            if got is not None:
                doc_id, ordinal = got
                return {"doc_id": doc_id, "ordinals": {ordinal},
                        "method": "unit"}
            row = self._unit_row(mem, uid)
            if row is not None:
                doc_id = index.doc_for_source(row.get("source_id"))
                if doc_id is not None:
                    covered = index.covered_ordinals(
                        doc_id, row.get("byte_start"),
                        row.get("byte_end"),
                        source_id=row.get("source_id"))
                    if not covered and row.get("kind") == "turn":
                        by_seq = index.ordinal_by_seq(
                            doc_id, row.get("seq"),
                            source_id=row.get("source_id"))
                        covered = [by_seq] if by_seq is not None else []
                    if not covered:
                        by_q = index.ordinal_by_quote(doc_id, quote)
                        covered = ([by_q] if by_q is not None else [0])
                    # NOTE: deliberately no lazy ``unit_index`` fill
                    # here.  ``resolve_unit``'s warm path returns a
                    # single ordinal while this cold path resolves the
                    # full covered set — under concurrent retrieves a
                    # race on who fills the cache changed delivered
                    # ordinals run to run.  Keeping the cold path the
                    # only path makes resolution identical every call.
                    return {"doc_id": doc_id,
                            "ordinals": set(covered),
                            "method": f"unit_row:{row.get('kind')}"}

        if parsed is not None and parsed[0] == "claim":
            sid = self._claim_source(mem, parsed[1])
            doc_id = index.doc_for_source(sid) if sid else None
            if doc_id is not None:
                by_q = index.ordinal_by_quote(doc_id, quote)
                return {"doc_id": doc_id,
                        "ordinals": {by_q if by_q is not None else 0},
                        "method": "claim"}

        # source-level / bare-id hit
        sid = parsed[1] if parsed is not None else None
        if not sid:
            sid = str(getattr(hit, "memory_id", "") or "") or None
        doc_id = index.doc_for_source(sid) if sid else None
        if doc_id is None:
            return None
        by_q = index.ordinal_by_quote(doc_id, quote)
        return {"doc_id": doc_id,
                "ordinals": {by_q if by_q is not None else 0},
                "method": "source"}

    def _expand(
        self, index: SessionIndex, hits: List[Any],
        resolved: List[Optional[Dict[str, Any]]],
        query: str = "",
    ) -> Dict[str, Any]:
        """V85-02.04 — hits in rank order, each contributing its covered
        turns ± ``neighbor_w`` session neighbors, deduped, stopping
        before the metered context would exceed ``token_budget``."""
        budget = self.token_budget
        w = int(self.neighbor_w)
        _is_temporal = bool(_TEMPORAL_RE.search(str(query or "")))
        clip_cap = _env_int("VERBATIM_AMB_TURN_CLIP", 0)
        if not _is_temporal:
            clip_cap = _env_int("VERBATIM_AMB_TURN_CLIP_NT", clip_cap)
        clip_terms = (_q_terms(str(query or ""))
                      if _env_int("VERBATIM_AMB_TERM_CLIP", 0) else ())
        max_turns = _env_int("VERBATIM_AMB_MAX_TURNS_PER_DOC", 0)
        meter = self._meter
        delivered: Dict[str, set] = {}
        covered_by: Dict[str, set] = {}
        rank_of: Dict[str, int] = {}
        add_order: List[Tuple[str, int]] = []
        full_docs: set = set()
        strays: List[Tuple[int, Any]] = []
        spent = 0
        stopped = False
        new_turns = 0
        n_hits_mapped = 0

        # ``VERBATIM_AMB_ENUM`` — enumeration-shaped queries ("how many
        # X", "all the Y") get a breadth coverage pass.  Hit sessions
        # spend only TOKEN_BUDGET - ENUM_RESERVE; every un-nominated
        # session then contributes its first ENUM_TURNS turns in
        # chronological order until the real budget fills.  Enumeration
        # answers need every instance session in-context, and single-
        # pass lexical hits structurally miss zero-overlap sessions.
        enum_cov = (_env_int("VERBATIM_AMB_ENUM", 0)
                    and bool(_ENUMERATIVE_RE.search(str(query or ""))))
        cov_reserve = _env_int("VERBATIM_AMB_ENUM_RESERVE", 4000)
        hit_budget = budget
        if enum_cov and budget is not None:
            hit_budget = max(budget - cov_reserve, 2000)

        for rank, (hit, r) in enumerate(zip(hits, resolved)):
            if stopped:
                break
            if r is None or r.get("doc_id") is None or (
                index.session(r["doc_id"]) is None
            ):
                # unresolved hit — delivered as itself, never dropped
                text = str(getattr(hit, "quote", "") or "")
                if isinstance(text, (bytes, bytearray)):
                    text = bytes(text).decode("utf-8", "replace")
                cost = meter("[verbatim · retrieved]\n" + text) + 8
                if hit_budget is not None and spent + cost > hit_budget:
                    stopped = True
                    break
                strays.append((rank, hit))
                spent += cost
                continue
            doc_id = r["doc_id"]
            sess = index.session(doc_id)
            n_turns = len(sess["turns"])
            if not n_turns:
                continue
            n_hits_mapped += 1
            covered = sorted(o for o in r["ordinals"]
                             if 0 <= o < n_turns) or [0]
            needed = set()
            for c in covered:
                lo = max(0, c - w)
                hi = min(n_turns - 1, c + w)
                needed.update(range(lo, hi + 1))
            # ``VERBATIM_AMB_USERCTX`` = W2 — a wider reach that applies
            # to *user* turns only: the factual claims ("I received the
            # chandelier from my aunt") live in user statements while
            # lexical hits usually land on assistant boilerplate.  W2
            # pulls in user turns up to W2 ordinals away from a hit,
            # ordered by distance so the nearest join first.
            uw = _env_int("VERBATIM_AMB_USERCTX", 0)
            if uw:
                for c in covered:
                    for d in range(1, uw + 1):
                        for o in (c - d, c + d):
                            if 0 <= o < n_turns and str(
                                    sess["turns"][o].get("speaker")
                                    or "") == "user":
                                needed.add(o)
            # ``VERBATIM_AMB_SESS_FULL`` = N — when a hit session has
            # <= N turns, deliver ALL of them: evidence often lives in
            # turns far from any hit (a pasted document at turn 0
            # answering "what year did construction begin"), and
            # small sessions are cheap once clipped.
            # ``VERBATIM_AMB_SESS_FULL_BYTES`` = B — additionally
            # requires the session's raw text to fit B chars;
            # oversized sessions fall back to covered±w delivery so one
            # giant session cannot starve the budget.  Full turns are
            # still clipped (measured: unclipped full sessions collapse
            # temporal-reasoning to baseline by starving the context of
            # co-evidence sessions) — pair with VERBATIM_AMB_TERM_CLIP
            # so the clip window lands on evidence, not position 0.
            sf = _env_int("VERBATIM_AMB_SESS_FULL", 0)
            sfb = _env_int("VERBATIM_AMB_SESS_FULL_BYTES", 0)
            # ``VERBATIM_AMB_SESS_FULL_MIN_TURN`` = chars — full delivery
            # only fires when the hit session contains a turn at least
            # this long (a pasted document / long-form answer).  Short
            # chat sessions keep normal covered±w density: delivering
            # their every turn just adds distractors (measured: ss-
            # preference lost 5/30 to whole-session spillover).
            sfmt = _env_int("VERBATIM_AMB_SESS_FULL_MIN_TURN", 0)
            if sf and n_turns <= sf and (
                    not sfb or sum(len(str(t.get("text") or ""))
                                   for t in sess["turns"]) <= sfb) and (
                    not sfmt or max(
                        (len(str(t.get("text") or ""))
                         for t in sess["turns"]), default=0) >= sfmt):
                needed.update(range(n_turns))
                full_docs.add(doc_id)
            # hit's own coverage first, then neighbors by expanding ring
            ordered = sorted(
                needed,
                key=lambda o: (min(abs(o - c) for c in covered), o),
            )
            if doc_id not in rank_of:
                rank_of[doc_id] = rank
            got = delivered.setdefault(doc_id, set())
            covered_ords = covered_by.setdefault(doc_id, set())
            for o in ordered:
                if o in got:
                    continue
                if max_turns and len(got) >= max_turns:
                    break
                line = _clip_turn(
                    render_turn_line(sess["turns"][o]),
                    cap=clip_cap, terms=clip_terms)
                cost = meter(line) + 1
                if not got:
                    # first delivered turn pays the header + the
                    # "## Memory i\n" wrapper + join margin
                    cost += meter(render_header(sess)) + 8
                if hit_budget is not None and spent + cost > hit_budget:
                    stopped = True
                    break
                got.add(o)
                if o in covered:
                    covered_ords.add(o)
                add_order.append((doc_id, o))
                spent += cost
                new_turns += 1

        if enum_cov and index is not None:
            # ``stopped`` here means hits spent the reserve — expected;
            # coverage has its own ``cov_full`` flag on the real budget.
            cov_turns = _env_int("VERBATIM_AMB_ENUM_TURNS", 2)
            cov_clip = (_env_int("VERBATIM_AMB_ENUM_CLIP", 160)
                        or clip_cap)
            cov_full = False
            _min_dt = datetime.min.replace(tzinfo=timezone.utc)
            for d, sess in sorted(
                    index.sessions.items(),
                    key=lambda kv: (_sess_dt(kv[1]) or _min_dt)):
                if cov_full:
                    break
                d = str(d)
                if d in delivered:
                    continue
                turns = sess.get("turns") or []
                for o in range(min(cov_turns, len(turns))):
                    line = _clip_turn(
                        render_turn_line(turns[o]),
                        cap=cov_clip, terms=clip_terms)
                    cost = meter(line) + 1
                    if d not in delivered or not delivered[d]:
                        cost += meter(render_header(sess)) + 8
                    if budget is not None and spent + cost > budget:
                        cov_full = True
                        break
                    delivered.setdefault(d, set()).add(o)
                    add_order.append((d, o))
                    spent += cost
                    new_turns += 1
                if d in delivered:
                    rank_of.setdefault(d, 1 << 29)

        return {
            "delivered": delivered,
            "covered": covered_by,
            "rank_of": rank_of,
            "add_order": add_order,
            "strays": strays,
            "spent": spent,
            "new_turns": new_turns,
            "stopped_early": stopped,
            "hits_mapped": n_hits_mapped,
            "full_docs": full_docs,
        }

    def _render_docs(
        self, user_id: Any, index: SessionIndex, ex: Dict[str, Any],
        query: str = "", ask_dt: Any = None,
        ent_tags: Optional[Dict[str, List[str]]] = None,
    ) -> List[Any]:
        """One AMB ``Document`` per session excerpt (V85-02.05):
        sessions by best hit rank, turns in dialogue order.

        Temporal surface (all env-gated, off by default):
        ``VERBATIM_AMB_REL_DATE``=1 tags each header with the session's
        day/week/month distance to ``ask_dt`` (the question date);
        =2/'poles' additionally marks the EARLIEST and MOST RECENT
        delivered memories.  ``VERBATIM_AMB_TORDER`` = 'chrono' /
        'recent' reorders emitted session docs by session date (with
        'auto' it applies only to temporal-shaped queries).
        ``VERBATIM_AMB_TIMELINE`` = N prepends a synthetic doc listing
        the N best-ranked dated sessions chronologically with their
        relative offsets.  ``VERBATIM_AMB_TEMPORAL_ONLY`` — when set,
        all temporal decorations (rel tags, pole marks, reordering,
        timeline) apply only to queries matching ``_TEMPORAL_RE``;
        other questions render exactly as the knobs-off baseline.
        """
        Doc = _doc_cls()
        uid = None if user_id is None else str(user_id)
        entries: List[Tuple[int, str, Any]] = []
        delivered = ex["delivered"]
        rank_of = ex["rank_of"]

        temporal_only = _env_int("VERBATIM_AMB_TEMPORAL_ONLY", 0)
        is_temporal = bool(_TEMPORAL_RE.search(query))
        temporal_ok = not temporal_only or is_temporal
        clip_cap = _env_int("VERBATIM_AMB_TURN_CLIP", 0)
        if not is_temporal:
            clip_cap = _env_int("VERBATIM_AMB_TURN_CLIP_NT", clip_cap)
        clip_terms = (_q_terms(str(query or ""))
                      if _env_int("VERBATIM_AMB_TERM_CLIP", 0) else ())

        rel_mode = (os.environ.get("VERBATIM_AMB_REL_DATE", "")
                    or "").strip().lower()
        rel_on = (rel_mode not in ("", "0", "off")
                  and ask_dt is not None and temporal_ok)
        poles = rel_on and rel_mode in ("2", "pole", "poles")
        torder = (os.environ.get("VERBATIM_AMB_TORDER", "")
                  or "").strip().lower()
        if not temporal_ok:
            torder = ""
        tl_mode = (os.environ.get("VERBATIM_AMB_TIMELINE", "")
                   or "").strip().lower()
        tl_n = (_env_int("VERBATIM_AMB_TIMELINE", 0)
                if temporal_ok else 0)
        if temporal_ok and tl_mode in ("all", "full"):
            tl_n = max(tl_n, 1)  # 'all' mode enables the block

        sdt_of: Dict[str, Any] = {}
        for doc_id in delivered:
            sess = index.session(doc_id)
            if sess is not None:
                sdt_of[doc_id] = _sess_dt(sess)

        pole_mark: Dict[str, str] = {}
        if poles:
            dated = {d: t for d, t in sdt_of.items() if t is not None}
            if len(dated) >= 2:
                pole_mark[min(dated, key=dated.get)] = (
                    "the EARLIEST of these memories")
                pole_mark[max(dated, key=dated.get)] = (
                    "the MOST RECENT of these memories")

        def _header(doc_id: str, sess: Mapping[str, Any]) -> str:
            h = render_header(sess)
            if not h.endswith("]"):
                return h
            extras: List[str] = []
            if rel_on and sdt_of.get(doc_id) is not None:
                extras.append(_rel_tag(sdt_of[doc_id], ask_dt))
            if doc_id in pole_mark:
                extras.append(pole_mark[doc_id])
            return (h[:-1] + " · " + " · ".join(extras) + "]"
                    if extras else h)

        for doc_id, ords in delivered.items():
            if not ords:
                continue
            sess = index.session(doc_id)
            if sess is None:
                continue
            covered = (ex.get("covered") or {}).get(doc_id, set())
            marker = os.environ.get("VERBATIM_AMB_HIT_MARK", "")
            lines = [
                (marker + _clip_turn(
                    render_turn_line(sess["turns"][o]),
                    cap=clip_cap, terms=clip_terms)
                 if marker and o in covered else
                 _clip_turn(render_turn_line(sess["turns"][o]),
                            cap=clip_cap, terms=clip_terms))
                for o in sorted(ords)
            ]
            tagline = ""
            _tags = ent_tags or {}
            if _tags.get(doc_id):
                tagline = ("\n[entities: "
                           + ", ".join(_tags[doc_id]) + "]")
            content = (_header(doc_id, sess) + tagline + "\n"
                       + "\n".join(lines))
            entries.append((
                rank_of.get(doc_id, 1 << 30),
                str(doc_id),
                Doc(
                    id=str(doc_id),
                    content=content,
                    user_id=uid,
                    timestamp=sess.get("timestamp"),
                    context=sess.get("context"),
                    source_ids=[str(doc_id)],
                ),
            ))
        for rank, hit in ex["strays"]:
            text = getattr(hit, "quote", "") or ""
            if isinstance(text, (bytes, bytearray)):
                text = bytes(text).decode("utf-8", "replace")
            src = str(getattr(hit, "memory_id", "") or "")
            entries.append((
                rank,
                f"~hit-{rank:05d}",
                Doc(
                    id=f"verbatim-hit-{rank:05d}",
                    content=f"[verbatim · retrieved]\n{text}",
                    user_id=uid,
                    source_ids=[src] if src else None,
                ),
            ))
        entries.sort(key=lambda e: (e[0], e[1]))

        torder_active = bool(torder) and torder not in ("0", "off")
        if torder_active and "auto" in torder and not _TEMPORAL_RE.search(
                str(query)):
            torder_active = False
        if torder_active:
            # re-sort only the session docs (stray hits keep their
            # rank-ordered tail slots); undated sessions go last for
            # 'chrono', keep min-date sink for 'recent'.
            dt_min = datetime.min.replace(tzinfo=timezone.utc)
            sess_idx = [i for i, e in enumerate(entries)
                        if not e[1].startswith("~hit-")]
            if "recent" in torder:
                sess_entries = sorted(
                    (entries[i] for i in sess_idx),
                    key=lambda e: sdt_of.get(e[1]) or dt_min,
                    reverse=True)
            else:
                sess_entries = sorted(
                    (entries[i] for i in sess_idx),
                    key=lambda e: (
                        sdt_of.get(e[1]) is None,
                        sdt_of.get(e[1]) or dt_min,
                        e[0], e[1]))
            for slot, e in zip(sess_idx, sess_entries):
                entries[slot] = e

        if tl_n and sdt_of:
            emitted = {e[1] for e in entries
                       if not e[1].startswith("~hit-")}
            tl_mode = (os.environ.get("VERBATIM_AMB_TIMELINE", "")
                       or "").strip().lower()
            tl_all = tl_mode in ("all", "full")
            if tl_all:
                # enumerate every dated session in the bank, not just
                # delivered ones — enumeration/ordering questions
                # ("the three trips in order") need sessions retrieval
                # never hit
                dated = sorted(
                    ((str(d), _sess_dt(s))
                     for d, s in index.sessions.items()
                     if _sess_dt(s) is not None),
                    key=lambda kv: kv[1])
                keep = {d for d, _t in dated}
            else:
                dated = sorted(
                    ((str(d), t) for d, t in sdt_of.items()
                     if t is not None and str(d) in emitted),
                    key=lambda kv: kv[1])
                # keep the best-ranked N sessions, oldest→newest
                keep = {d for d, _t in sorted(
                    dated, key=lambda kv: rank_of.get(kv[0], 1 << 30)
                )[:tl_n]}
            tl_lines = []
            # +2: the timeline itself occupies "## Memory 1", shifting
            # every session doc one slot later in the joined context.
            pos = {e[1]: i + 2 for i, e in enumerate(entries)}
            deliver_str = {str(k): v for k, v in delivered.items()}
            q_terms = _q_terms(str(query or ""))
            for d, t in dated:
                if d not in keep:
                    continue
                sess = index.session(d)
                snip = ""
                ords = sorted(deliver_str.get(d) or ())
                if sess is not None:
                    if q_terms:
                        # query-relevant snippet: the turn that shares
                        # the most query terms, centered on the match —
                        # surfaces buried evidence (e.g. "ukulele"
                        # inside a keyboard-shopping session)
                        txt = _snip_turn(sess, q_terms)
                    elif ords:
                        txt = str(
                            sess["turns"][ords[0]].get("text") or "")
                    else:
                        txt = str(sess["turns"][0].get("text") or ""
                                  ) if sess["turns"] else ""
                    snip = re.sub(r"\s+", " ", txt).strip()[:90]
                    if snip:
                        snip = f" — {snip}"
                lab = t.strftime("%Y-%m-%d, %A")
                rel = (f" — {_rel_tag(t, ask_dt)}" if ask_dt is not None
                       else "")
                if d in pos:
                    tl_lines.append(
                        f"- Memory {pos[d]}: {lab}{rel}{snip}")
                else:
                    tl_lines.append(f"- {lab}{rel}{snip}")
            if tl_lines:
                scope = ("every dated session in this bank"
                         if tl_all else
                         "the dated sessions in the memories below")
                tl_content = (
                    f"[timeline — {scope}, oldest to newest; offsets "
                    "are relative to the question date]\n"
                    + "\n".join(tl_lines))
                if _env_int("VERBATIM_AMB_TL_HINT", 0):
                    tl_content += (
                        "\nHow to use this timeline: each date is when "
                        "that conversation happened. When the user "
                        "described doing something 'today', 'yesterday' "
                        "or recently in a session, treat that session's "
                        "date as the event's date. Count distinct "
                        "events by their session rows.")
                tl_doc = Doc(
                    id="verbatim-timeline",
                    content=tl_content,
                    user_id=uid,
                )
                entries.insert(0, (-1, "verbatim-timeline", tl_doc))
        return [e[2] for e in entries]

    def _deliver(
        self, user_id: Any, index: SessionIndex, hits: List[Any],
        resolved: List[Optional[Dict[str, Any]]],
        query: str = "", ask_dt: Any = None,
        ent_tags: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Expand → render → budget-verifying trim (the meter is not
        perfectly additive, so the final context string is re-metered
        and trailing additions dropped until it fits — deterministic)."""
        ex = self._expand(index, hits, resolved, query=query)
        trimmed = 0
        while True:
            docs = self._render_docs(user_id, index, ex,
                                     query=query, ask_dt=ask_dt,
                                     ent_tags=ent_tags)
            ctx = context_string(docs)
            est = self._meter(ctx)
            if self.token_budget is None or est <= self.token_budget:
                break
            if ex["add_order"]:
                doc_id, o = ex["add_order"].pop()
                ex["delivered"][doc_id].discard(o)
                trimmed += 1
            elif ex["strays"]:
                # only stray (unmapped) hits left — drop the last so the
                # metered context still respects B.
                ex["strays"].pop()
                trimmed += 1
            else:
                break  # nothing left to drop — honest overflow recorded
        ex["trimmed"] = trimmed
        ex["est_context_tokens"] = est
        return docs, ex

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: Optional[str] = None,
        query_timestamp: Optional[Any] = None,
        filters: Optional[dict] = None,
    ) -> Tuple[List[Any], Optional[dict]]:
        """AMB ``retrieve`` → ``(documents, None)`` — ``raw_response`` is
        ALWAYS ``None`` (V85-02.05: LoCoMo json.dumps-es it into the
        reader prompt; the pack is never serialized).  Per-query stats
        land on :attr:`query_records`."""
        t_start = time.perf_counter()
        rec: Dict[str, Any] = {
            "query": str(query),
            "k": int(k),  # recorded, never obeyed (V85-02.03)
            "user_id": user_id,
            "query_timestamp": query_timestamp,
            "search_limit": self.search_limit,
            "neighbor_w": self.neighbor_w,
            "token_budget": self.token_budget,
            "token_meter": self._meter_name,
            "doc_mode": self.doc_mode,
            "pack_source": "session_expansion/v1",
            "provider_version": PROVIDER_VERSION,
        }
        if filters is not None and not _env_int(
                "VERBATIM_AMB_FILTERS", 0):
            # VERBATIM_AMB_FILTERS off → supports_filters=False is
            # declared, so the harness never sends this — a direct
            # caller gets the ignore recorded.
            rec["filters_ignored"] = True

        if not str(query).strip():
            # upstream convention: a blank query gets an empty context
            # (PrecisionMemBench asserts exactly that).
            rec["status"] = "empty_query"
            rec.update(n_hits=0, n_docs=0, doc_ids=[],
                       est_context_tokens=0, retrieve_ms=0.0)
            self._record(rec)
            return [], None

        mem = self._bank(user_id)
        bkey = _unit_key(user_id)
        index = self._indexes.get(bkey)
        if mem is None and index is not None and self._store_dir:
            # resume run (reset=False): the persisted sidecar says this
            # bank exists — reopen it lazily on its deterministic path.
            mem = self._bank_or_create(user_id)
        if mem is None or index is None:
            if not self._banks and not self._indexes:
                # upstream convention (bm25 provider): explicit failure
                # when nothing was ever ingested.
                raise RuntimeError(
                    "verbatim: no documents ingested yet"
                )
            rec["status"] = "no_bank"
            rec["note"] = "no memory bank exists for this isolation unit"
            rec.update(n_hits=0, n_docs=0, doc_ids=[],
                       est_context_tokens=0, retrieve_ms=0.0)
            self._record(rec)
            return [], None

        kwargs, as_of_rec = self._search_kwargs(mem, query_timestamp)
        rec["as_of"] = as_of_rec
        _ask_raw, ask_us = parse_query_time(query_timestamp)
        ask_dt = (None if ask_us is None else
                  datetime.fromtimestamp(ask_us / 1_000_000,
                                         tz=timezone.utc))
        rec["temporal_shape"] = bool(_TEMPORAL_RE.search(str(query)))
        try:
            t0 = time.perf_counter()
            with self._search_lock:
                res = mem.search(str(query), **kwargs)
            search_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001 — per-query error,
            # recorded, never swallowed
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["retrieve_ms"] = round(
                (time.perf_counter() - t_start) * 1000.0, 3)
            self._record(rec)
            raise

        hits = list(getattr(res, "items", None) or [])
        resolved = [self._resolve_hit(mem, index, h) for h in hits]

        seen = {
            getattr(h, "object_ref", None)
            or getattr(h, "memory_id", None)
            for h in hits
        }
        rfx_k = _env_int("VERBATIM_AMB_RFX", 0)
        if rfx_k and hits:
            terms = self._feedback_terms(index, str(query), resolved)
            if terms:
                try:
                    with self._search_lock:
                        res2 = mem.search(
                            str(query) + " " + " ".join(terms),
                            **kwargs)
                    extra = list(getattr(res2, "items", None) or [])
                    added = 0
                    for h in extra:
                        key = (getattr(h, "object_ref", None)
                               or getattr(h, "memory_id", None))
                        if key in seen:
                            continue
                        hits.append(h)
                        resolved.append(
                            self._resolve_hit(mem, index, h))
                        seen.add(key)
                        added += 1
                    rec["rfx"] = {"terms": terms, "added": added}
                except Exception as exc:  # noqa: BLE001
                    rec["rfx"] = {"terms": terms,
                                  "error": f"{type(exc).__name__}: {exc}"}

        # Per-call expansion terms + entity tagmap: concurrent
        # retrieves share one provider instance (concurrency>1), so
        # these must never live on ``self`` — cross-query pollution
        # made contexts differ run-to-run.
        qx_last: List[str] = []
        qx_model = (os.environ.get("VERBATIM_AMB_QX", "")
                    or "").strip()
        if (qx_model and _env_int("VERBATIM_AMB_QX_COND", 0)
                and _qx_cond_off(query)):
            rec["qx_cond"] = "off"
            qx_model = ""
        if qx_model and hits:
            # LLM query expansion: the question's category words
            # ("console", "meat", "state", "technique") almost never
            # lexically overlap the answer turn ("Xeonoblade",
            # "Chicken Pot Pie", "Tampa", "25 minutes on").  One cheap
            # model call proposes ≤QX_TOP surface terms the answer turn
            # plausibly contains; a second search appends dedup'd tail
            # candidates (RFX pattern — never reorders the ranked head).
            # Expansions are cached per-query on disk so reruns are free.
            try:
                terms = self._qx_terms(str(query), qx_model)
                if _env_int("VERBATIM_AMB_QX_TWO", 0):
                    for t in self._qx_terms(str(query), qx_model,
                                            alt=True):
                        if t not in terms:
                            terms.append(t)
            except Exception as exc:  # noqa: BLE001
                terms, qx_err = [], f"{type(exc).__name__}: {exc}"
            else:
                qx_err = None
            qx_last = list(terms)[
                :max(1, _env_int("VERBATIM_AMB_QX_HINT_TOP", 16))]
            if terms:
                try:
                    # split the expansion into ≤SPLIT-term chunks — one
                    # long query dilutes the match and buries the
                    # evidence session; each chunk keeps its own pull
                    chunk = max(1, _env_int("VERBATIM_AMB_QX_SPLIT", 8))
                    extra: List = []
                    seen_extra: set = set()
                    for ci in range(0, len(terms), chunk):
                        ts = terms[ci:ci + chunk]
                        with self._search_lock:
                            res2 = mem.search(
                                str(query) + " " + " ".join(ts),
                                **kwargs)
                        for h in (getattr(res2, "items", None) or []):
                            ek = (getattr(h, "object_ref", None)
                                  or getattr(h, "memory_id", None))
                            if ek in seen_extra:
                                continue
                            seen_extra.add(ek)
                            extra.append(h)
                    # singleton pass: rare candidate terms ('tampa',
                    # 'mafia') lose the chunk's 8-way match — query+
                    # single term surfaces their exact-match session
                    single_k = _env_int("VERBATIM_AMB_QX_SINGLE", 0)
                    if single_k:
                        for t in terms:
                            with self._search_lock:
                                res3 = mem.search(
                                    str(query) + " " + t, **kwargs)
                            for h in (getattr(res3, "items", None)
                                      or [])[:single_k]:
                                ek = (getattr(h, "object_ref", None)
                                      or getattr(h, "memory_id", None))
                                if ek in seen_extra:
                                    continue
                                seen_extra.add(ek)
                                extra.append(h)
                    splice_n = _env_int("VERBATIM_AMB_QX_SPLICE", 12)
                    splice_at = _env_int("VERBATIM_AMB_QX_AT", 8)
                    added = 0
                    ins: List[int] = []
                    for h in extra:
                        key = (getattr(h, "object_ref", None)
                               or getattr(h, "memory_id", None))
                        if key in seen:
                            continue
                        hits.append(h)
                        resolved.append(
                            self._resolve_hit(mem, index, h))
                        seen.add(key)
                        ins.append(len(resolved) - 1)
                        added += 1
                    # nomination-vote: docs whose covered turns match
                    # MORE DISTINCT expansion terms win the splice —
                    # raw expanded-search rank lets common-word docs
                    # bury the evidence session
                    if ins:
                        def _votes(i: int) -> int:
                            r = resolved[i]
                            if not r or not r.get("doc_id"):
                                return 0
                            sess = index.session(r["doc_id"])
                            if not sess:
                                return 0
                            bag: set = set()
                            for t in sess["turns"]:
                                tx = (t.get("text", "")
                                      if isinstance(t, dict)
                                      else str(t)).lower()
                                for term in terms:
                                    if term in tx:
                                        bag.add(term)
                            return len(bag)
                        ins.sort(key=lambda i: (-_votes(i), i))
                    # QX candidates are the only bridge across the
                    # vocabulary gap — appended at the tail they land
                    # below the pack cutoff, so splice the top few into
                    # the mid-pack (bounded displacement, ~SPLICE slots).
                    # A QX hit nominates the right SESSION but its unit
                    # coverage often points at the wrong turns, so each
                    # spliced doc rescored turn-wise against the
                    # expansion terms takes its top matches (falling
                    # back to the hit's ordinals), then widens ±QX_NBR
                    # for conversational adjacency.
                    nbr = _env_int("VERBATIM_AMB_QX_NBR", 1)
                    doc_turns = _env_int("VERBATIM_AMB_QX_DOC_TURNS", 3)
                    if ins and splice_n:
                        front = ins[:splice_n]
                        take_h = [hits[i] for i in front]
                        take_r = [resolved[i] for i in front]
                        drop = set(front)
                        for r in take_r:
                            if not r or not r.get("doc_id"):
                                continue
                            sess = index.session(r["doc_id"])
                            n_t = (len(sess["turns"])
                                   if sess else 0)
                            if not n_t:
                                continue
                            scored = []
                            for o, t in enumerate(sess["turns"]):
                                tx = (t.get("text", "")
                                      if isinstance(t, dict)
                                      else str(t)).lower()
                                s = sum(1 for term in terms
                                        if term in tx)
                                scored.append((s, o))
                            scored.sort(key=lambda x: (-x[0], x[1]))
                            pick = [o for s, o in scored[:doc_turns]
                                    if s > 0]
                            # the hit's own ordinals stay — their ±nbr
                            # ring often reaches the evidence turn the
                            # rescore misses (adjacent chatter)
                            pick += list(r["ordinals"])
                            wide = set()
                            for o in pick:
                                for d in range(-nbr, nbr + 1):
                                    if 0 <= o + d < n_t:
                                        wide.add(o + d)
                            r["ordinals"] = wide
                        hits = [h for i, h in enumerate(hits)
                                if i not in drop]
                        resolved = [r for i, r in enumerate(resolved)
                                    if i not in drop]
                        at = min(splice_at, len(hits))
                        hits[at:at] = take_h
                        resolved[at:at] = take_r
                    rec["qx"] = {"terms": terms, "added": added,
                                 "spliced": min(len(ins), splice_n)}
                except Exception as exc:  # noqa: BLE001
                    rec["qx"] = {"terms": terms,
                                 "error": f"{type(exc).__name__}: {exc}"}
            elif qx_err:
                rec["qx"] = {"error": qx_err}

        ent_k = _env_int("VERBATIM_AMB_ENT_SPLICE", 0)
        ent_tags: Optional[Dict[str, List[str]]] = None
        if ent_k or _env_int("VERBATIM_AMB_ENT_TAGS", 0):
            # entity-anchored turn splice: RAW_ENT surfaces capitalized
            # entity names but leaves them floating as bare names in a
            # JSON list — the reader can't reach the turn that grounds
            # them.  This picks rare entities whose MENTION TURN text
            # overlaps query+QX terms and injects those turns (±ENT_NBR
            # ring) into the pack — deterministic SQL via the write
            # path's own entity_mentions/unit_index, zero LLM calls.
            try:
                estore = getattr(mem, "_store", None)
                if estore is not None:
                    rel = {
                        w for w in re.findall(
                            r"[a-z0-9']+", str(query).lower())
                        if len(w) >= 3 and w not in self._ENT_STOP
                    }
                    rel |= {str(t).lower() for t in qx_last}
                    # speaker names are in the query AND in nearly
                    # every mention turn — without exclusion every
                    # entity scores ≥1 and the gate is meaningless.
                    spk: Any = None
                    try:
                        spk = index.speakers() if hasattr(
                            index, "speakers") else self._speakers(
                            mem, index)
                        rel -= {str(s).lower() for s in (spk or ())}
                    except Exception:  # noqa: BLE001
                        pass
                    # evidence turns phrase answers in domain verbs
                    # the question never repeats — merge the matched
                    # domain lexicon so such turns still score.
                    qlow = str(query).lower()
                    for rx, lex in _ENT_DOMAIN_LEX:
                        if rx.search(qlow):
                            rel |= lex
                    # inflected evidence: 'bought' answers 'buy',
                    # 'studied' answers 'study'.
                    rel |= {
                        v for w in tuple(rel) for v in _morph(w)
                        if len(v) >= 3 and v not in self._ENT_STOP
                    }
                    rows = []
                    with estore.read() as conn:
                        from verbatim.storage.repos import has_table
                        if (has_table(conn, "entity_mentions")
                                and has_table(conn, "entity_canon")):
                            rows = conn.execute(
                                "SELECT m.canon, e.display, e.df_units,"
                                " m.unit_id"
                                " FROM entity_mentions m"
                                " JOIN entity_canon e"
                                "   ON e.canon = m.canon"
                                "  AND e.scope_id = m.scope_id"
                                " WHERE m.role = 'mention'"
                                " ORDER BY m.canon, m.unit_id").fetchall()
                    packed: set = set()
                    for r in resolved:
                        if r and r.get("doc_id"):
                            packed.update(
                                (r["doc_id"], o) for o in r["ordinals"])
                    by_canon: Dict[str, Dict[str, Any]] = {}
                    for canon, disp, _dfu, uid in rows:
                        if (not disp or not disp[0].isupper()
                                or not 3 <= len(disp) <= 40
                                or disp.lower() in self._ENT_STOP
                                or disp.lower().strip("'")
                                in self._ENT_STOP
                                or _GREETING_ENT_RE.search(disp)):
                            continue
                        ent = by_canon.setdefault(
                            canon, {"display": disp, "uids": []})
                        if uid:
                            ent["uids"].append(str(uid))
                    # place↔container: 'connecticut' in the question
                    # must surface the 'stamford' turn; 'paris' in the
                    # bag must surface 'france'.
                    if _env_int("VERBATIM_AMB_GEO", 1):
                        geo_add: set = set()
                        for w in tuple(rel):
                            geo_add |= _GEO_REL.get(w, frozenset())
                        for gk, gv in _GEO_REL.items():
                            if " " in gk and gk in qlow:
                                geo_add |= gv
                        rel |= {
                            g for g in geo_add
                            if g not in self._ENT_STOP
                        }
                    min_s = _env_int("VERBATIM_AMB_ENT_MINSCORE", 1)
                    enbr = _env_int("VERBATIM_AMB_ENT_NBR", 1)
                    # proper-noun proxy: real entities ('Tampa',
                    # 'Talkeetna') never surface lowercased mid-
                    # sentence; extractor junk ('Heard', 'Nature')
                    # is a sentence-initial common word — lowercase
                    # occurrences exist somewhere in the corpus.
                    # Cached per bank.
                    bw = getattr(self, "_bank_lowers", None)
                    if bw is None:
                        bw = self._bank_lowers = {}
                    bkey2 = getattr(index, "bank_key", "_shared")
                    entry = bw.get(bkey2)
                    if entry is None:
                        lowers: set = set()
                        low_text: List[str] = []
                        dfmap: Dict[str, int] = {}
                        n_turns2 = 0
                        for s2 in index.sessions.values():
                            for t in s2["turns"]:
                                tt = str(t.get("text") or "")
                                n_turns2 += 1
                                low_text.append(tt.lower())
                                ws = re.findall(
                                    r"[A-Za-z']+", tt)
                                for w in ws[1:]:
                                    if w and w[0].islower():
                                        lowers.add(w.lower())
                                for w in set(
                                        w.lower() for w in ws):
                                    dfmap[w] = dfmap.get(w, 0) + 1
                        entry = (lowers, "\x00".join(low_text),
                                 dfmap, max(1, n_turns2))
                        bw[bkey2] = entry
                    lowers, corpus_low, dfmap, n_turns2 = entry

                    def _idf(w: str) -> float:
                        import math
                        return math.log(
                            (n_turns2 + 1.0) / (dfmap.get(w, 0) + 1.0))

                    seen_docs = {
                        r["doc_id"] for r in resolved
                        if r and r.get("doc_id")}
                    place_q = bool(_GEO_PLACE_Q_RE.search(qlow))
                    place_bonus = float(_env_int(
                        "VERBATIM_AMB_PLACE_BONUS", 8))
                    scored_e: List[Tuple[float, str, list]] = []
                    for e in by_canon.values():
                        disp = e["display"]
                        dws = re.findall(r"[A-Za-z']+", disp)
                        if not dws:
                            continue
                        is_pl = _is_place(
                            disp.lower(), [w.lower() for w in dws])
                        # junk penalty: single gerund/verb-shaped names
                        # and phrases whose every word is a common
                        # lowercase word — concrete-noun anchors keep
                        # a survivable −1, extractor junk goes below
                        # the novel-session bonus.
                        pen = 0
                        if len(dws) == 1:
                            dlw = dws[0].lower()
                            if re.search(
                                    r"(ing|ed|ive|ly|est|n't|'s)$",
                                    dlw):
                                pen += 1
                            if dlw in lowers:
                                pen += 1
                        elif all(w.lower() in lowers or
                                 w.lower() in self._ENT_STOP
                                 for w in dws):
                            pen += 1
                        locs = []
                        best = -99.0
                        for uid in e["uids"]:
                            loc = index.unit_index.get(uid)
                            if not loc:
                                continue
                            did2, ord2 = loc[0], int(loc[1])
                            sess2 = index.session(did2)
                            if (not sess2
                                    or ord2 >= len(sess2["turns"])):
                                continue
                            tx = str(
                                sess2["turns"][ord2].get("text")
                                or "").lower()
                            # idf-weighted overlap — rare-domain terms
                            # ('beach', 'hatha') discriminate real
                            # evidence turns from chatty ones
                            # ('trip', 'visit').
                            # sorted() — FP accumulation over a
                            # str-set jitters in the last ulp across
                            # runs and flips scored_e sort ties.
                            s = sum(_idf(term) for term in sorted(rel)
                                    if term and term in tx)
                            # packed-but-undelivered: a turn sitting in
                            # a tail-ranked session dies at the budget
                            # wall — being "covered" is not being
                            # delivered.  Bonus only when the session
                            # isn't already near the top.
                            if did2 not in seen_docs:
                                s += 2.0
                            # QX/query names the entity itself — for
                            # 'in what <category>' questions the answer
                            # IS an entity name; a display token inside
                            # the expansion bag is the strongest
                            # anchor signal there is.
                            if _env_int(
                                    "VERBATIM_AMB_ENT_QXBONUS", 12):
                                dl = {w.lower() for w in dws}
                                if dl & rel:
                                    s += _env_int(
                                        "VERBATIM_AMB_ENT_QXBONUS",
                                        12)
                            # 'in what state/country/city?' — the
                            # answer IS a place entity; its mention
                            # turns are the evidence.  A place canon
                            # gets a flat bonus so its session
                            # surfaces.
                            if place_q and is_pl:
                                s += place_bonus
                            s -= pen
                            locs.append((s, did2, ord2))
                            if s > best:
                                best = s
                        if locs and best >= min_s:
                            scored_e.append((best, e["display"], locs,
                                             is_pl))
                    scored_e.sort(key=lambda x: (-x[0], x[1]))
                    if place_q:
                        # the answer is a place — deliver place
                        # sessions ahead of the df giants.
                        scored_e.sort(
                            key=lambda x: (not x[3], -x[0], x[1]))
                    per = max(1, _env_int("VERBATIM_AMB_ENT_TURNS", 2))
                    added_e = 0
                    ent_ins: List[int] = []
                    ins_set: set = set()
                    for _s, _disp, locs, _pl in scored_e:
                        if added_e >= ent_k:
                            break
                        took = 0
                        for s2, did2, ord2 in sorted(
                                locs, key=lambda x: (-x[0], x[2])):
                            if took >= per:
                                break
                            idx2 = next(
                                (i for i, r in enumerate(resolved)
                                 if r and r.get("doc_id") == did2),
                                None)
                            if idx2 is None:
                                r2 = {"doc_id": did2,
                                      "ordinals": set(),
                                      "method": "ent"}
                                resolved.append(r2)
                                hits.append(hits[-1] if hits else None)
                                idx2 = len(resolved) - 1
                            else:
                                r2 = resolved[idx2]
                            # every anchored session — new OR packed-
                            # but-tail-ranked — must reach the front
                            # or its evidence dies at the budget wall.
                            if idx2 not in ins_set:
                                ins_set.add(idx2)
                                ent_ins.append(idx2)
                            wide = set()
                            n_t2 = len(
                                (index.session(did2) or {}).get(
                                    "turns") or [])
                            for d in range(-enbr, enbr + 1):
                                o2 = ord2 + d
                                # r2-local, not the global packed set —
                                # a second resolved entry for the same
                                # session can "cover" the turn while
                                # this elevated entry renders without
                                # it (the global check lost Tampa).
                                if 0 <= o2 < n_t2 and \
                                        o2 not in r2["ordinals"]:
                                    wide.add(o2)
                            if not wide:
                                continue
                            r2["ordinals"].update(wide)
                            packed.update((did2, o) for o in wide)
                            took += 1
                        if took:
                            added_e += 1
                            if _env_int("VERBATIM_AMB_DEBUG", 0):
                                rec.setdefault(
                                    "ent_spliced", []).append(
                                    {"display": _disp,
                                     "score": round(_s, 1),
                                     "turns": took})
                    if ent_ins:
                        pos = min(
                            _env_int("VERBATIM_AMB_ENT_AT", 10),
                            len(hits))
                        take_h = [hits[i] for i in ent_ins]
                        take_r = [resolved[i] for i in ent_ins]
                        drop2 = set(ent_ins)
                        hits = [h for i, h in enumerate(hits)
                                if i not in drop2]
                        resolved = [r for i, r in enumerate(resolved)
                                    if i not in drop2]
                        hits[pos:pos] = take_h
                        resolved[pos:pos] = take_r
                    if _env_int("VERBATIM_AMB_ENT_TAGS", 0):
                        # reader-side anchors: list each delivered
                        # session's extracted entities in its header —
                        # connective signal ('Paris' next to 'Seraphim')
                        # without spending turn budget.  Strict
                        # proper-noun proxy only: a display that ever
                        # appears lowercase mid-sentence is extractor
                        # noise ('Here's', 'Last Friday'), and speakers
                        # are already in the conversation header.
                        spk_low = {str(s).lower()
                                   for s in (spk or ())}
                        ord_sc = {
                            disp: s for s, disp, _, _p in scored_e}
                        tagmap: Dict[str, List[str]] = {}
                        for e in by_canon.values():
                            disp_t = e["display"]
                            dwt = re.findall(r"[A-Za-z']+", disp_t)
                            if not dwt:
                                continue
                            if len(dwt) == 1 and \
                                    dwt[0].lower() in lowers:
                                continue
                            if disp_t.lower() in spk_low:
                                continue
                            if any(w.lower() in spk_low
                                   for w in dwt):
                                continue
                            for uid in e["uids"]:
                                loc = index.unit_index.get(uid)
                                if not loc:
                                    continue
                                lst = tagmap.setdefault(
                                    loc[0], [])
                                cont = _GEO_CONTAINER.get(
                                    disp_t.lower())
                                tag = (f"{disp_t} ({cont})"
                                       if cont else disp_t)
                                if tag not in lst:
                                    lst.append(tag)
                        tcap = _env_int(
                            "VERBATIM_AMB_ENT_TAGS_N", 8)
                        for did, names in tagmap.items():
                            names.sort(
                                key=lambda d: -ord_sc.get(d, -99))
                            del names[tcap:]
                        ent_tags = tagmap
                    else:
                        ent_tags = None
                    rec["ent_splice"] = {"added": added_e,
                                         "cands": len(scored_e)}
            except Exception as exc:  # noqa: BLE001
                rec["ent_splice"] = {
                    "error": f"{type(exc).__name__}: {exc}"}

        sfx_mode = (os.environ.get("VERBATIM_AMB_SFX", "")
                    or "").strip().lower()
        if sfx_mode and sfx_mode != "0" and hits:
            # subject-probe: when the query names a speaker of this
            # bank, a bare-name search surfaces that person's own
            # statements across every session — covers "what has X
            # done/been/read/met" enumerative misses whose answer
            # terms never enter the query's lexical space.  The
            # probe appends are harmless tail candidates in any
            # mode; speaker-turn coverage inflates nominated docs'
            # turn cost, so under 'auto' it fires only for
            # enumerative-shaped questions.
            try:
                speakers = index.speakers() if hasattr(
                    index, "speakers") else self._speakers(mem, index)
                qlow = str(query).lower()
                names = [s for s in speakers
                         if s and s.lower() in qlow]
                added = 0
                sfx_new: List[int] = []
                for name in names[:2]:
                    with self._search_lock:
                        res3 = mem.search(name, **kwargs)
                    for h in (getattr(res3, "items", None) or []):
                        key = (getattr(h, "object_ref", None)
                               or getattr(h, "memory_id", None))
                        if key in seen:
                            continue
                        hits.append(h)
                        resolved.append(
                            self._resolve_hit(mem, index, h))
                        seen.add(key)
                        sfx_new.append(len(resolved) - 1)
                        added += 1
                speaker_cov = (sfx_mode not in ("auto", "enum") or
                               bool(_ENUMERATIVE_RE.search(str(query))))
                if names and speaker_cov:
                    # speaker-turn coverage — ONLY on sessions the
                    # name probe itself nominated: their generic
                    # ordinals miss the answer turns, which are the
                    # named speakers' own lines ("I'm doing
                    # kickboxing").  Up to SFX_TURNS per such doc.
                    nlow = {n.lower() for n in names}
                    cap = _env_int("VERBATIM_AMB_SFX_TURNS", 6)
                    spk_cov = 0
                    seen_docs: set = set()
                    for i in sfx_new:
                        r = resolved[i]
                        if not r or not r.get("doc_id"):
                            continue
                        if r["doc_id"] in seen_docs:
                            continue
                        sess = index.session(r["doc_id"])
                        if not sess:
                            continue
                        own = [o for o, t in enumerate(sess["turns"])
                               if isinstance(t, dict)
                               and str(t.get("speaker") or "").lower()
                               in nlow][:cap]
                        if own:
                            r["ordinals"] = set(r["ordinals"]) | set(own)
                            seen_docs.add(r["doc_id"])
                            spk_cov += 1
                    rec["sfx"] = {"names": names, "added": added,
                                  "speaker_docs": spk_cov}
            except Exception as exc:  # noqa: BLE001
                rec["sfx"] = {"error": f"{type(exc).__name__}: {exc}"}

        ddi_mode = (os.environ.get("VERBATIM_AMB_DDIVERSE", "")
                    or "").strip().lower()
        if ddi_mode and ddi_mode != "0" and resolved:
            if "auto" in ddi_mode or "enum" in ddi_mode:
                # enumerative-shaped questions get the diversity
                # reorder; 'diffuse' mode additionally fires it when
                # pass-1 evidence already spans many sessions.  In
                # measurement the diffuse heuristic re-opened the
                # non-enum regression, so it stays opt-in.
                ok = bool(_ENUMERATIVE_RE.search(str(query)))
                if not ok and "diffuse" in ddi_mode:
                    top_docs = {str(r.get("doc_id")) for r in resolved[:10]
                                if r and r.get("doc_id")}
                    ok = len(top_docs) >= _env_int(
                        "VERBATIM_AMB_DDI_MIN_DOCS", 6)
                if not ok:
                    ddi_mode = "0"
        if ddi_mode and ddi_mode != "0" and resolved:
            # session-diversity ordering: round-robin over distinct
            # doc_ids (each session's best hit first) instead of raw
            # rank order — enumeration needs every evidence session's
            # covered turns inside the token budget before any one
            # session double-dips.  Unresolved hits keep rank order
            # after the doc round-robin.  'tail' mode keeps the top
            # DDI_TOP hits in raw rank order and diversifies only the
            # remainder — strong evidence keeps rank while the tail
            # spreads across sessions.
            keep_top = (_env_int("VERBATIM_AMB_DDI_TOP", 8)
                        if "tail" in ddi_mode else 0)
            per_doc: Dict[str, List[int]] = {}
            stray_idx: List[int] = []
            for i, r in enumerate(resolved):
                if r and r.get("doc_id"):
                    per_doc.setdefault(r["doc_id"], []).append(i)
                else:
                    stray_idx.append(i)
            order: List[int] = []
            depth = 0
            while True:
                tier = [idxs[depth] for _d, idxs in
                        sorted(per_doc.items(),
                               key=lambda kv: kv[1][0])
                        if depth < len(idxs)]
                if not tier:
                    break
                order.extend(tier)
                depth += 1
            order.extend(stray_idx)
            if keep_top:
                order = sorted(order[:keep_top]) + order[keep_top:]
            hits = [hits[i] for i in order]
            resolved = [resolved[i] for i in order]

        if filters and _env_int("VERBATIM_AMB_FILTERS", 0):
            # Honor the AMB filter contract: sessions whose stored tags
            # (caller-supplied ``amb_tags`` + extraction-label-derived
            # ``label_tags``) fail any group never deliver — excluded
            # memories don't just rank low, they never enter the pack.
            # And a doc the query names is delivered even when search
            # never surfaced it (a misspelled name has no lexical
            # overlap to score) — name-matched entries lead the pack.
            elig = {
                did for did, s in index.sessions.items()
                if self._sess_eligible(s, filters)
            }
            paired = [
                (h, r) for h, r in zip(hits, resolved)
                if r is not None and r.get("doc_id") in elig]
            have = {r["doc_id"] for _, r in paired}
            if _env_int("VERBATIM_AMB_FILTER_FILL", 1):
                front = [
                    (None, {"doc_id": did, "ordinals": {0},
                            "method": "filter_eligible"})
                    for did in sorted(elig - have)]
                n_fill = len(front)
                paired = front + paired
            else:
                n_fill = 0
            rec["filters_applied"] = {
                "eligible_docs": sorted(elig),
                "kept": len(paired), "filled": n_fill,
                "dropped": (len(hits) - (len(paired) - n_fill))}
            hits = [h for h, _ in paired]
            resolved = [r for _, r in paired]

        tscore_k = _env_int("VERBATIM_AMB_TSCORE", 0)
        if tscore_k and resolved:
            # turn-level rescore: a hit nominates its session but ships
            # only its unit's covered ordinals, so the answer-bearing
            # turn of an already-packed session never renders (the
            # 'right session, wrong turns' failure).  Rescore every
            # turn of each nominated session against the query's
            # content tokens + the QX expansion terms, keep the top-K
            # scorers plus the hit's own ordinals, then widen
            # ±TSCORE_NBR.  Gated on the enumerative question shape
            # unless TSCORE_ALL=1.
            if (_env_int("VERBATIM_AMB_TSCORE_ALL", 0)
                    or _ENUMERATIVE_RE.search(str(query))):
                bag = {
                    w for w in
                    re.findall(r"[a-z0-9']+", str(query).lower())
                    if len(w) > 2 and w not in self._ENT_STOP
                }
                bag |= {str(t).lower() for t in qx_last}
                for w in list(bag):
                    var = _VERB_IRREG.get(w)
                    if var:
                        bag |= set(var)
                tnbr = _env_int("VERBATIM_AMB_TSCORE_NBR", 1)
                rescored = 0
                for r in resolved:
                    if not r or not r.get("doc_id"):
                        continue
                    sess = index.session(r["doc_id"])
                    n_t = len(sess["turns"]) if sess else 0
                    if not n_t:
                        continue
                    scored = []
                    for o, t in enumerate(sess["turns"]):
                        tx = (t.get("text", "")
                              if isinstance(t, dict)
                              else str(t)).lower()
                        s = sum(1 for term in bag if term in tx)
                        if s:
                            scored.append((s, o))
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    pick = {o for _s, o in scored[:tscore_k]}
                    pick |= set(r["ordinals"])
                    wide = set()
                    for o in pick:
                        for dd in range(-tnbr, tnbr + 1):
                            if 0 <= o + dd < n_t:
                                wide.add(o + dd)
                    r["ordinals"] = wide
                    rescored += 1
                rec["tscore"] = {"docs": rescored,
                                 "keep": tscore_k}

        _dc_mode = _env_int("VERBATIM_AMB_DATECALC", 0)
        if _dc_mode and resolved is not None:
            # date-arithmetic probe: questions like "what did I do 44
            # days after buying the MacBook on March 20" or "226 days
            # before New Year" anchor on a computable date — resolve it
            # and inject the day's session into the candidate list so
            # budget breadth decides inclusion, not lexical luck.
            try:
                from datetime import date as _dc_date, timedelta as _dc_td
                qtext = str(query)
                qd = _dc_qdates(qtext, query_timestamp)
                if qd:
                    by_n = {}
                    by_d = {}
                    for did, s in (
                            getattr(index, "sessions", {}) or {}).items():
                        try:
                            by_n[int(s.get("session_n"))] = did
                        except (TypeError, ValueError):
                            pass
                        _ts = str(s.get("timestamp") or "")[:10]
                        if _ts:
                            try:
                                by_d.setdefault(
                                    _dc_date.fromisoformat(_ts),
                                    []).append(did)
                            except ValueError:
                                continue
                    first_d = None
                    for s in (getattr(index, "sessions", {}) or {}).values():
                        ts = str(s.get("timestamp") or "")[:10]
                        if ts:
                            try:
                                cand = _dc_date.fromisoformat(ts)
                            except ValueError:
                                continue
                            if first_d is None or cand < first_d:
                                first_d = cand
                    injected = []
                    promoted = []
                    seen_did: set = set()
                    pos_ = min(4, len(resolved))
                    for d in qd:
                        # LifeBench path: daily sessions indexed by
                        # (date − first_date)+1.  Corpora without a
                        # session_N label (lme) fall back to matching the
                        # session whose own date equals the computed one
                        # (±1d slack for loose phrasing like "two weeks
                        # ago" that sits between session days; a date may
                        # hold several sessions — all are candidates).
                        if first_d is not None:
                            _one = by_n.get((d - first_d).days + 1)
                            cands = [_one] if _one else []
                        else:
                            cands = []
                        if not cands:
                            for _cand in (d, d + _dc_td(days=1),
                                          d - _dc_td(days=1)):
                                cands = [
                                    did_ for did_ in (by_d.get(_cand)
                                                      or [])
                                    if did_ not in seen_did]
                                if cands:
                                    break
                        for did in cands:
                            if did in seen_did or (
                                    len(injected) + len(promoted) >= 4):
                                continue
                            seen_did.add(did)
                            # mode 2 also PROMOTES already-resolved
                            # sessions to the probe slot; mode 1 leaves
                            # them at natural rank (promotion measured
                            # at best neutral, −2.1pt on Sun Yuwei)
                            ei = next(
                                (i for i, r in enumerate(resolved)
                                 if r and r.get("doc_id") == did), None)
                            if ei is not None:
                                if _dc_mode >= 2 and ei > pos_:
                                    r = resolved.pop(ei)
                                    resolved.insert(pos_, r)
                                    hits.insert(
                                        pos_, hits.pop(ei))
                                    promoted.append(did)
                                continue
                            injected.append({
                                "doc_id": did,
                                "ordinals": set(range(
                                    len((index.session(did) or {})
                                        .get("turns") or ()))),
                                "method": "datecalc",
                                "hit_ordinals": set(),
                                "dc_date": d.isoformat()})
                    if injected:
                        pos_ = min(4, len(resolved))
                        resolved[pos_:pos_] = injected
                        hits[pos_:pos_] = [None] * len(injected)
                    rec["datecalc"] = {
                        "dates": [d.isoformat() for d in qd],
                        "injected": [r["doc_id"] for r in injected],
                        "promoted": promoted}
            except Exception as exc:  # noqa: BLE001
                rec["datecalc"] = {"error": f"{type(exc).__name__}: {exc}"}

        docs, ex = self._deliver(user_id, index, hits, resolved,
                                 query=str(query), ask_dt=ask_dt,
                                 ent_tags=ent_tags)
        delivered_ids = [
            str(getattr(d, "id", "")) for d in docs
            if not str(getattr(d, "id", "")).startswith("verbatim-hit-")
        ]

        coverage = getattr(res, "coverage", None) or {}
        support = coverage.get("support") or {}
        resolution: Dict[str, int] = {}
        for r in resolved:
            if r is None:
                bucket = "unresolved"
            else:
                m = str(r.get("method") or "source")
                bucket = "unit_row" if m.startswith("unit_row") else m
            resolution[bucket] = resolution.get(bucket, 0) + 1

        retrieve_ms = (time.perf_counter() - t_start) * 1000.0
        rec.update({
            "status": str(getattr(res, "status", "unknown")),
            "verdict": support.get("verdict"),
            "n_hits": len(hits),
            "n_sessions": len(ex["delivered"]),
            "n_strays": len(ex["strays"]),
            "n_docs": len(docs),
            "doc_ids": delivered_ids,
            "est_context_tokens": ex["est_context_tokens"],
            "search_ms": round(search_ms, 3),
            "retrieve_ms": round(retrieve_ms, 3),
            "expansion": {
                "hits_mapped": ex["hits_mapped"],
                "turns_delivered": ex["new_turns"],
                "stopped_early": ex["stopped_early"],
                "trimmed": ex["trimmed"],
                "est_meter_tokens": ex["spent"],
            },
            "resolution": resolution,
            "warnings": [
                str(w) for w in (getattr(res, "warnings", None) or ())
            ],
        })
        self._record(rec)
        if (_env_int("VERBATIM_AMB_RAW_ENT", 0)
                or _env_int("VERBATIM_AMB_QX_HINT", 0)
                or _env_int("VERBATIM_AMB_RAW_DOCS", 0)):
            try:
                return docs, self._raw_with_entities(
                    mem, index, docs,
                    with_entities=bool(_env_int("VERBATIM_AMB_RAW_ENT", 0)),
                    clues=(list(qx_last)
                           if _env_int("VERBATIM_AMB_QX_HINT", 0)
                           else None))
            except Exception:  # noqa: BLE001 — raw path is additive only
                if _env_int("VERBATIM_AMB_DEBUG", 0):
                    import traceback
                    traceback.print_exc()
                return docs, None
        return docs, None  # raw_response=None is REQUIRED for LoCoMo

    def _raw_with_entities(
        self, mem: Any, index: SessionIndex, docs: List[Any],
        with_entities: bool = True,
        clues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Recall-result JSON for LoCoMo's raw path — the same packed
        turn content plus the per-session entity surface forms the
        write path extracted (names the evidence turn itself may only
        imply: 'Talkeetna', 'Xenoblade Chronicles', 'Stamford')."""
        import sqlite3  # noqa: F401 — store.read() supplies the conn

        store = getattr(mem, "_store", None) if with_entities else None
        if store is None:
            out = {
                "memories": [
                    {"id": str(getattr(d, "id", "")),
                     "content": getattr(d, "content", "") or ""}
                    for d in docs],
                **({"clues": clues} if clues else {}),
            }
            if _env_int("VERBATIM_AMB_RAW_DOCS", 0):
                _dn = _env_int("VERBATIM_AMB_RAW_DOCS_N", 0)
                out["documents"] = list(docs[:_dn] if _dn else docs)
            return out
        per_doc: Dict[str, List[str]] = {}
        try:
            with store.read() as conn:
                from verbatim.storage.repos import has_table
                if not (has_table(conn, "entity_mentions")
                        and has_table(conn, "entity_canon")
                        and has_table(conn, "units")):
                    raise RuntimeError("no entity tables")
                for d in docs:
                    did = str(getattr(d, "id", ""))
                    sess = index.session(did)
                    if not sess:
                        continue
                    srcs = tuple(sess.get("source_ids") or ())
                    if not srcs:
                        continue
                    marks = ",".join("?" * len(srcs))
                    rows = conn.execute(
                        "SELECT DISTINCT e.display, e.df_units"
                        " FROM entity_mentions m"
                        " JOIN entity_canon e"
                        "   ON e.canon = m.canon AND e.scope_id = m.scope_id"
                        " JOIN units u ON u.unit_id = m.unit_id"
                        " WHERE u.source_id IN (" + marks + ")"
                        "   AND m.role = 'mention'",
                        srcs).fetchall()
                    names = [
                        disp for disp, _df in rows
                        if disp and disp[0].isupper()
                        and 2 <= len(disp) <= 40
                        and disp.lower() not in self._ENT_STOP
                        and disp.strip("'").lower() not in self._ENT_STOP
                    ]
                    # rare-and-capitalized first — proper-noun signal
                    names.sort(key=lambda n: (
                        0 if any(c.isupper() for c in n[1:]) else 1, n))
                    per_doc[did] = names[:40]
        except Exception:  # noqa: BLE001
            if _env_int("VERBATIM_AMB_DEBUG", 0):
                import traceback
                traceback.print_exc()
            per_doc = {}
        out = {
            "memories": [
                {"id": str(getattr(d, "id", "")),
                 "content": getattr(d, "content", "") or ""}
                for d in docs],
            "entities_per_memory": per_doc,
            **({"clues": clues} if clues else {}),
        }
        if _env_int("VERBATIM_AMB_RAW_DOCS", 0):
            # retrieval-scored datasets (PrecisionMemBench) read
            # raw_response["documents"] — real Document objects with
            # .id/.content/.context so belief-id resolution can fire.
            # Never on LoCoMo: raw_response gets json.dumps-ed there.
            # RAW_DOCS_N caps the list — the bench counts noise as a
            # hard failure, so only the top-N ranked docs report.
            _dn = _env_int("VERBATIM_AMB_RAW_DOCS_N", 0)
            out["documents"] = list(docs[:_dn] if _dn else docs)
        return out

    _ENT_STOP = frozenset(
        "the a an and or of to in on for with is are was were be been "
        "it its that this those these i you he she we they my your his "
        "her our their what which who when where why how have has had "
        "do does did not no yes so but if then than just really very "
        "can could would should will shall may might also get got go "
        "went going come came see saw say said know think like want "
        "need take make made feel felt look looks looking seem seems "
        "one two three four five six seven eight nine ten first second "
        "last next new old good bad great nice cool fun big small long "
        "short high low right left early late hard easy same different "
        "sure okay ok yeah yep nope hey hi hello thanks thank please "
        "anything everything something nothing anyone everyone someone "
        "always never sometimes often usually ever already still yet "
        "again once twice ago now today yesterday tomorrow tonight "
        "here there where everywhere anywhere somewhere nowhere "
        "much many more most less least few several each every all "
        "some none any both either neither other others another such "
        "own same only even just quite rather pretty too very enough "
        "bit lot lots kind sort type part side way thing things stuff "
        "time times day days week weeks month months year years "
        "morning afternoon evening night today tonight "
        "let's let's let us i'm i've i'll i'd you're you've you'll "
        "you'd he's he's she'll it's it'll we'd we've we'll we'd "
        "they're they've they ' ll ' d ' re ' ve m s t don didn't "
        "doesn't isn't aren't wasn't weren't won't wouldn't can't "
        "couldn't shouldn't hasn't haven't hadn't mustn't "
        "keep keeps keeping kept try tries trying tried want wants "
        "wanting wanted need needs needing needed start starts "
        "starting started stop stops stopping stopped remember "
        "remembers remembering remembered forget forgets forgetting "
        "forgot happen happens happening happened mean means meaning "
        "meant guess guesses guessing guessed hope hopes hoping hoped "
        "wish wishes wishing wished love loves loving loved hate hates "
        "hating hated enjoy enjoys enjoying enjoyed mind minds minding "
        "minded care cares caring cared sounds sound looks seem "
        "awesome certainly definitely probably maybe actually "
        "basically literally honestly seriously totally completely "
        "absolutely exactly especially generally normally obviously "
        "recently currently finally suddenly immediately "
        "plus also anyway besides however therefore otherwise "
        "meanwhile afterward afterwards before after during while "
        "since until unless though although despite within without "
        "around about above below under over between among through "
        "against along across behind beyond beside besides near far "
        "upon onto off out up down back away forward together apart "
        "march april may june july august september october november "
        "december january february monday tuesday wednesday thursday "
        "friday saturday sunday weekend weekdays"
        .split())

    _RFX_STOP = frozenset(
        "the a an and or of to in on for with is are was were be been "
        "it its that this those these i you he she we they my your his "
        "her our their what which who when where why how have has had "
        "do does did not no yes so but if then than just really very "
        "can could would should will shall may might also get got go "
        "went going come came see saw say said know think like want "
        "need make take time year years day days way thing things one "
        "two some any much more most other another each all both few "
        "many same such only even still back own over under again once "
        "here there out up down about into through during before after "
        "between while because until them him us me".split())

    def _feedback_terms(
        self, index: SessionIndex, query: str,
        resolved: List[Optional[Dict[str, Any]]],
    ) -> List[str]:
        """Relevance feedback (``VERBATIM_AMB_RFX``): rare content terms
        mined from pass-1 covered turn texts — co-occurrence inside
        evidence surfaces answer-side vocabulary the query lacks
        (enumerative + semantic-gap misses).  Pass-2 hits are APPENDED
        after pass-1 hits, so nomination is a pure union: nothing
        pass-1 found can be displaced."""
        qterms = set(re.findall(r"[a-z']+", query.lower()))
        freq: Dict[str, int] = {}
        n_docs = 0
        for r in resolved[:24]:
            if not r or not r.get("doc_id"):
                continue
            sess = index.session(r["doc_id"])
            if not sess:
                continue
            n_docs += 1
            for o in r["ordinals"]:
                if 0 <= o < len(sess["turns"]):
                    t = sess["turns"][o]
                    text = (t.get("text") or "") if isinstance(t, dict) else str(t)
                    for w in set(re.findall(r"[a-z']{4,}", text.lower())):
                        if (w not in self._RFX_STOP
                                and w not in qterms
                                and "'" not in w):
                            freq[w] = freq.get(w, 0) + 1
        topk = _env_int("VERBATIM_AMB_RFX", 8)
        return [w for w, c in sorted(
            freq.items(), key=lambda kv: (-kv[1], kv[0]))
            if c >= 2][:topk]

    def _speakers(self, mem: Any, index: SessionIndex) -> List[str]:
        """Distinct speaker canons present in the bank's sessions
        (cached per index)."""
        cached = getattr(index, "_speaker_cache", None)
        if cached is not None:
            return cached
        names: List[str] = []
        for sess in getattr(index, "sessions", {}).values():
            for t in (sess or {}).get("turns") or []:
                s = t.get("speaker") if isinstance(t, dict) else None
                if s and s not in names:
                    names.append(s)
        try:
            index._speaker_cache = names
        except Exception:  # noqa: BLE001
            pass
        return names

    _QX_SYS = (
        "You expand a memory-retrieval query over raw conversation "
        "turns. The turn carrying the evidence usually does NOT contain "
        "the answer word itself — it describes it indirectly. List up "
        "to {top} terms likely to appear VERBATIM inside that evidence "
        "turn, split into two lists: \"candidates\" — plausible answers "
        "to the question (for 'which national park': real park names; "
        "for 'which US state': state and city names, incl. less famous "
        "ones like 'stamford', 'boston', 'talkeetna'; for 'composer': "
        "composer names AND the works they scored like 'harry potter'); "
        "and \"context\" — the surrounding everyday words the turn "
        "would use (e.g. 'hiking', 'trail', 'map', 'dogs'; 'playing', "
        "'rpg'; 'trip', 'flew'; 'chicken', 'grill'; 'soundtrack'). "
        "Prefer concrete proper nouns and everyday words over the "
        "question's own abstract category terms. Never emit the "
        "question's own words. Output JSON only: "
        '{{"candidates": ["a1"], "context": ["c1"]}}')

    _QX_SYS_ALT = (
        "You expand a memory-retrieval query over raw conversation "
        "turns. Describe what the evidence turn concretely SAYS using "
        "everyday words and specific names — the casual phrasing a "
        "friend would use, not the question's formal category words. "
        "List up to {top} terms: everyday activities, specific place "
        "or product names, and the words adjacent to the topic. "
        "Output JSON only: {{\"terms\": [\"t1\", \"t2\"]}}")

    # -- AMB filter contract (VERBATIM_AMB_FILTERS / VERBATIM_AMB_TAGS) ----
    #
    # PrecisionMemBench sends a provider-neutral filter at retrieve time
    # (``filters``): ``any`` = OR-groups AND-ed, ``all`` = every tag,
    # ``none`` = exclusion, ``narrow_any`` = OR of tagged spec leaves
    # (each leaf has its own ``resolve``: literal ``exact`` match or
    # ``fuzzy`` trigram resolve).  Tags live on the session record:
    # caller-supplied ``Document.tags`` land in ``sess["amb_tags"]``;
    # extraction-label values land in ``sess["label_tags"]`` (the LLM
    # tagger honours the dataset's own label spec — state values are
    # classified, multi-text keys become ``<key>:<normalized>`` tags).

    _TAG_WORD_RE = re.compile(r"[a-z0-9]+")

    _STOPWORDS = frozenset(
        "a an the in on of for to and or is are was were be been by at "
        "as it its with from that this these those do does did not no "
        "so if then than but we you he she they us our your my me him "
        "her them what which who how when where why should could would "
        "can may might will shall must about into over under between "
        "through during before after up down out off again further once "
        "here there all any both each few more most other some such "
        "only own same too very just also".split())

    @staticmethod
    def _tag_norm(value: str) -> str:
        """Candidate-tag normalization — identical to the dataset's:
        lowercase ``[a-z0-9]+`` words joined by single spaces."""
        return " ".join(
            VerbatimAMBProvider._TAG_WORD_RE.findall(
                str(value).lower()))

    @staticmethod
    def _trigrams(s: str) -> set:
        s = "  " + s + " "
        return {s[i:i + 3] for i in range(len(s) - 2)}

    @classmethod
    def _tri_jac(cls, a: str, b: str) -> float:
        A, B = cls._trigrams(a), cls._trigrams(b)
        return len(A & B) / len(A | B) if A and B else 0.0

    @classmethod
    def _fuzzy_hit(cls, cand: str, tag_val: str) -> bool:
        """Trigram resolve for one ``name:`` candidate vs one stored
        tag value.  Guards keep morphological noise out of a raw
        threshold: the stored tag may not be a strict string prefix of
        the candidate (a plural like 'errors' must not resolve to
        'error'), word counts must match ('chapter' must not resolve to
        'chapter 7'), and multi-word candidates resolve word-aligned —
        position-wise trigram Jaccard — so 'chapter with' cannot reach
        'chapter 7' through the shared first word."""
        cw, tw = cand.split(), tag_val.split()
        if not cw or len(cw) != len(tw):
            return False
        if cand.startswith(tag_val):
            return False
        if len(cw) > 1:
            thr = _env_float("VERBATIM_AMB_TAG_FUZZ_RUN", 0.6)
            return all(
                cls._tri_jac(a, b) >= thr or cls._is_subseq(a, b)
                for a, b in zip(cw, tw))
        thr = _env_float("VERBATIM_AMB_TAG_FUZZ", 0.45)
        # a dropped-letter candidate ('eror') is a subsequence of its
        # target ('error') — that resolves even below the trigram
        # floor, while a vowel-swapped one ('mango' vs 'mongo') is not
        # a subsequence and stays blocked.  The subsequence escape is
        # gated to exactly one dropped letter so filler words like
        # 'here' (a subsequence of 'inheritance') can't leak through.
        return (cls._tri_jac(cand, tag_val) >= thr
                or cls._is_subseq(cand, tag_val))

    @staticmethod
    def _is_subseq(cand: str, val: str) -> bool:
        """True when ``cand`` equals ``val`` minus exactly one
        character — a dropped-letter typo, not a letter swap or an
        accidental word-inside-a-word."""
        if len(val) - len(cand) != 1 or len(cand) < 3:
            return False
        it = iter(val)
        return all(ch in it for ch in cand)

    @classmethod
    def _narrow_hit(cls, tags: set, spec: dict) -> bool:
        """One ``narrow_any`` spec leaf against a session's tag set.
        ``resolve=exact`` is literal membership; ``resolve=fuzzy``
        compares only ``name:``-namespaced values on both sides."""
        spec_tags = spec.get("tags") or []
        if spec.get("resolve") == "fuzzy":
            tag_names = [
                t[5:] for t in tags
                if isinstance(t, str) and t.startswith("name:")]
            for cand in spec_tags:
                c = str(cand)
                c = c[5:] if c.startswith("name:") else c
                if any(cls._fuzzy_hit(c, tv) for tv in tag_names):
                    return True
            return False
        return any(str(t) in tags for t in spec_tags)

    @classmethod
    def _sess_eligible(cls, sess: dict, flt: dict) -> bool:
        """AND-of-groups filter semantics — the same mapping other
        providers hand to their own tag-group WHERE: ``any`` is OR
        inside each group but AND across groups, ``narrow_any`` is one
        OR-ed leaf block, ``all``/``narrow`` are single groups, ``none``
        excludes."""
        tags = set(sess.get("amb_tags") or ()) | set(
            sess.get("label_tags") or ())
        for grp in flt.get("any") or []:
            if grp and not (tags & set(grp)):
                return False
        for t in flt.get("all") or []:
            if t not in tags:
                return False
        if tags & set(flt.get("none") or ()):
            return False
        narrow_any = [
            s for s in (flt.get("narrow_any") or []) if s.get("tags")]
        if narrow_any:
            if not any(cls._narrow_hit(tags, s) for s in narrow_any):
                return False
        if flt.get("narrow"):
            if not (tags & set(flt["narrow"])):
                return False
        return True

    def _label_tags(self, text: str) -> List[str]:
        """LLM extraction-label tagger (``VERBATIM_AMB_TAGS``): builds
        its prompt from the dataset's ``extraction_labels`` spec — key,
        type ('value' picks one of ``values``; 'multi-text' returns a
        list) and description — so the contract stays generic rather
        than pmb-specific.  Only labels flagged ``tag`` emit tags;
        normalized the way the lookup normalizes candidates."""
        import json as _json
        import urllib.request

        labels = [lb for lb in (self._extraction_labels or [])
                  if isinstance(lb, dict) and lb.get("tag")]
        if not labels:
            return []
        model = (os.environ.get("VERBATIM_AMB_TAG_MODEL")
                 or os.environ.get("OMB_ANSWER_MODEL")
                 or "").strip()
        if not model:
            raise RuntimeError(
                "VERBATIM_AMB_TAGS needs VERBATIM_AMB_TAG_MODEL "
                "or OMB_ANSWER_MODEL")
        spec_lines, out_shape = [], {}
        for lb in labels:
            key = str(lb.get("key") or "").strip()
            if not key:
                continue
            desc = str(lb.get("description") or "")
            if lb.get("type") == "value":
                vals = [str(v.get("value"))
                        for v in (lb.get("values") or [])
                        if isinstance(v, dict) and v.get("value")]
                spec_lines.append(
                    f'- "{key}": one of {vals}. {desc}')
                out_shape[key] = vals[0] if vals else ""
            else:
                spec_lines.append(
                    f'- "{key}": a list of strings. {desc}')
                out_shape[key] = []
        if not out_shape:
            return []
        sys_prompt = (
            "You are a memory-index tagger. Read the memory text and "
            "return a JSON object with exactly these keys:\n"
            + "\n".join(spec_lines)
            + "\nFor multi-text keys: the text's opening tokens before "
            "the first sentence ARE the subject's names — copy EVERY "
            "one verbatim (normalized), including abbreviations, "
            "acronyms, shorthand and even misspellings. Emit each "
            "separately — NEVER fuse several names into one compound. "
            "Keep atomic single-word forms ('error', 'pov', 'mongodb') "
            "alongside multi-word ones ('error handling', 'point of "
            "view'); if the fact explicitly rejects an alternative "
            "('use X, not Y'), Y is a name too. A pattern, convention, "
            "style or preference IS a nameable subject — when the text "
            "opens with a canonical name, always include it and never "
            "return an empty list. Then add common short forms a user "
            "might type. Exhaustive coverage beats brevity.\n"
            "Respond with JSON only.")
        key = hashlib.sha256(
            (model + "\x00" + sys_prompt + "\x00" + text).encode()
        ).hexdigest()[:24]
        cdir = Path(os.environ.get(
            "VERBATIM_AMB_TAG_CACHE", "/tmp/amb-tag-cache"))
        cfile = cdir / f"{key}.json"
        try:
            if cfile.exists():
                return list(_json.loads(cfile.read_text())
                            .get("tags") or [])
        except Exception:  # noqa: BLE001
            pass
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        api = os.environ.get("OPENAI_API_KEY", "")
        if not base or not api:
            raise RuntimeError(
                "tag extraction needs OPENAI_BASE_URL+OPENAI_API_KEY")
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text[:4000]},
            ],
            "temperature": 0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        obj: Dict[str, Any] = {}
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    base + "/chat/completions",
                    data=_json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {api}",
                             "Content-Type": "application/json",
                             # CF error 1010 bans urllib's default UA
                             "User-Agent": "curl/8.5.0"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=60) as r:
                    payload = _json.loads(r.read().decode())
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
                time.sleep(0.5)
                continue
            content_txt = (payload.get("choices") or [{}])[0].get(
                "message", {}).get("content", "") or ""
            m = re.search(r"\{.*\}", content_txt, re.S)
            if m:
                try:
                    obj = _json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    obj = {}
            if obj:
                break
        tags: List[str] = []
        for lb in labels:
            key = str(lb.get("key") or "")
            val = obj.get(key)
            if lb.get("type") == "value":
                allowed = {str(v.get("value"))
                           for v in (lb.get("values") or [])
                           if isinstance(v, dict)}
                v = str(val).strip().lower() if val is not None else ""
                if v and (not allowed or v in allowed):
                    tags.append(f"{key}:{v}")
            elif isinstance(val, list):
                for item in val:
                    n = self._tag_norm(item)
                    if n:
                        tags.append(f"{key}:{n}")
            elif isinstance(val, str):
                n = self._tag_norm(val)
                if n:
                    tags.append(f"{key}:{n}")
        # multi-word name decomposition: 'auth redis session backend'
        # also carries 'session backend' / 'redis session' as candidates
        # — contiguous 2-3 word sub-runs of each emitted name.  Single
        # words are never decomposed (a lone noun like 'chapter' must
        # not leak out of a compound like 'chapter 7').
        if _env_int("VERBATIM_AMB_TAG_SUBRUNS", 1):
            for t in list(tags):
                if not t.startswith("name:"):
                    continue
                ws = t[5:].split()
                if len(ws) < 3:
                    continue
                for w in (2, 3):
                    if w >= len(ws):
                        continue
                    for i in range(len(ws) - w + 1):
                        tags.append("name:" + " ".join(ws[i:i + w]))
        tags = list(dict.fromkeys(tags))
        if tags:
            try:
                cdir.mkdir(parents=True, exist_ok=True)
                tmp = cdir / (
                    f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(_json.dumps({"tags": tags}))
                tmp.replace(cfile)
            except Exception:  # noqa: BLE001
                pass
        return tags

    def _qx_terms(self, query: str, model: str,
                  alt: bool = False) -> List[str]:
        """LLM query expansion (``VERBATIM_AMB_QX``): one cheap call
        against the OpenAI-compatible endpoint (OPENAI_BASE_URL /
        OPENAI_API_KEY), disk-cached per (model, query) so reruns cost
        nothing.  Returns ≤``VERBATIM_AMB_QX_TOP`` surface terms."""
        import json as _json
        import urllib.request

        top = _env_int("VERBATIM_AMB_QX_TOP", 10)
        sys_prompt = (self._QX_SYS_ALT if alt else
                      self._QX_SYS).format(top=top)
        key = hashlib.sha256(
            f"{model}\x00{sys_prompt}\x00{query}".encode()).hexdigest()[:24]
        cdir = Path(os.environ.get(
            "VERBATIM_AMB_QX_CACHE", "/tmp/amb-qx-cache"))
        cfile = cdir / f"{key}.json"
        try:
            if cfile.exists():
                cached = _json.loads(cfile.read_text())
                return list(cached.get("terms") or [])
        except Exception:  # noqa: BLE001
            pass
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        api = os.environ.get("OPENAI_API_KEY", "")
        if not base or not api:
            raise RuntimeError("qx needs OPENAI_BASE_URL+OPENAI_API_KEY")
        body = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": sys_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        terms: List[str] = []
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    base + "/chat/completions",
                    data=_json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {api}",
                             "Content-Type": "application/json",
                             # CF error 1010 bans urllib's default UA
                             "User-Agent": "curl/8.5.0"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    payload = _json.loads(r.read().decode())
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
                time.sleep(0.5)
                continue
            content = (payload.get("choices") or [{}])[0].get(
                "message", {}).get("content", "") or ""
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    obj = _json.loads(m.group(0))
                    cand = (obj.get("candidates") or [])
                    ctx = (obj.get("context") or [])
                    if cand or ctx:
                        # keep both halves — a global cap starves the
                        # context list (models emit candidates first)
                        half = max(1, top // 2)
                        raw = cand[:half] + ctx[:top - half]
                    else:
                        raw = (obj.get("terms") or obj.get("expansions")
                               or [])
                    qt = set(re.findall(r"[a-z0-9]+", query.lower()))
                    for t in raw:
                        t = str(t).strip().lower()
                        if (3 <= len(t) <= 40 and t not in qt
                                and re.fullmatch(r"[a-z0-9' \-]+", t)):
                            terms.append(t)
                except Exception:  # noqa: BLE001
                    pass
            if terms:
                break
        terms = terms[:top]
        if terms:
            # cache only real expansions — transient empties must not
            # be sticky
            try:
                cdir.mkdir(parents=True, exist_ok=True)
                tmp = cdir / (
                    f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(_json.dumps({"q": query,
                                            "terms": terms}))
                tmp.replace(cfile)
            except Exception:  # noqa: BLE001
                pass
        return terms

    def _record(self, rec: Dict[str, Any]) -> None:
        with self._rec_lock:
            self._queries.append(rec)

    # -- records ------------------------------------------------------------

    @property
    def query_records(self) -> List[Dict[str, Any]]:
        """Per-retrieve records — the token-curve rows and the runner's
        per-query stats source (raw_response is None by contract)."""
        return list(self._queries)

    def token_curve(self) -> Dict[str, Any]:
        """``{token_budget, meter, est_context_tokens}`` for the run —
        the accuracy-vs-tokens curve's x values at this budget."""
        return {
            "token_budget": self.token_budget,
            "neighbor_w": self.neighbor_w,
            "search_limit": self.search_limit,
            "meter": self._meter_name,
            "amb_meter": "cl100k_base (harness-side on a real run)",
            "est_context_tokens": [
                r.get("est_context_tokens") for r in self._queries
            ],
            "n_docs": [r.get("n_docs") for r in self._queries],
            "turns_delivered": [
                (r.get("expansion") or {}).get("turns_delivered")
                for r in self._queries
            ],
            "max_context_tokens": [
                (r.get("expansion") or {}).get("est_meter_tokens")
                for r in self._queries
            ],
        }

    def manifest_fields(self) -> Dict[str, Any]:
        """V85-02.07 — the provider's arm record for the run manifest."""
        return {
            "provider_revision": PROVIDER_VERSION,
            "token_budget": self.token_budget,
            "neighbor_w": self.neighbor_w,
            "search_limit": self.search_limit,
            "concurrency": self.concurrency,
            "timeout_ms": self.timeout_ms,
            "doc_mode": self.doc_mode,
            "encoder": self.encoder,
            "worker": self.worker,
            "infer": self.infer,
            "token_meter": self._meter_name,
            "expansion": "session_neighbors/v1",
            "ingest_path": self.last_ingest.get("ingest_path"),
        }


# ---------------------------------------------------------------------------
# AMB context assembly + registration
# ---------------------------------------------------------------------------


def context_string(docs: Iterable[Any]) -> str:
    """AMB rag-mode context assembly (``amb-rag/v1`` — the upstream
    runner's own join, ``memory_bench/modes/rag.py``):
    ``## Memory i\\n{content}`` separated by blank lines.

    The harness performs this join itself on a real run; the local
    runner reproduces it byte-identically for per-query token records
    and the mirror answer loop."""
    return "\n\n".join(
        f"## Memory {i + 1}\n{_get(d, 'content', default='')}"
        for i, d in enumerate(docs)
    )


def retrieval_context(docs: Iterable[Any]) -> str:
    """AMB retrieval-mode context dump (``memory_bench/modes/
    retrieval.py``): ``## Retrieved memories (N)`` then
    ``i. [id] ← source_ids\\n{content}`` per document."""
    docs = list(docs)
    lines = [f"## Retrieved memories ({len(docs)})"]
    for i, d in enumerate(docs):
        src_ids = _get(d, "source_ids") or []
        src = f" ← {', '.join(str(s) for s in src_ids)}" if src_ids else ""
        lines.append(
            f"{i + 1}. [{_get(d, 'id', default='')}]{src}\n"
            f"{_get(d, 'content', default='')}"
        )
    return "\n\n".join(lines)


def register(registry: Optional[dict] = None) -> dict:
    """Install ``verbatim`` into an AMB provider registry (V8-15.06 —
    the provider appears in ``amb providers``, scenario K87)."""
    if registry is None:
        try:
            from memory_bench import memory as _amb_memory

            registry = _amb_memory.REGISTRY
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "AMB registry unavailable — run inside the pinned "
                f"agent-memory-benchmark checkout ({exc})"
            ) from exc
    registry[PROVIDER_NAME] = VerbatimAMBProvider
    return registry


__all__ = [
    "AMBDoc",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_NEIGHBOR_W",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_TOKEN_BUDGET",
    "DOC_MODES",
    "HARNESS_PATCHES",
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "TOKEN_BUDGETS",
    "VerbatimAMBProvider",
    "context_string",
    "default_token_meter",
    "harness_patch_records",
    "parse_object_ref",
    "parse_query_time",
    "provider_arm_fields",
    "register",
    "retrieval_context",
]
