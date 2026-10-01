"""T2 grounded LLM-fact tier — deterministic machinery around the model
call (SPEC_V7 V7-13.12–18, §30 ``t2_facts``).

T2 is the *optional* tier competitors call "memory": an LLM proposes
short factual statements over a session window, each carrying **required
quote spans copied verbatim** from the source units. This module owns
everything around that proposal — the model call itself is injected as
``model_fn`` (a plain ``prompt -> str`` callable) so this module never
performs I/O, never reads a clock, and never hides a missing model:

- ``model_fn=None`` (or a raising callable) → ``status="unavailable"``
  (``reason="no_model"`` / ``"model_error:<Type>"``). V7-13.18: absence
  never blocks, never fabricates, never retries unboundedly. The open
  question O8 resolves local-only; this module is model-agnostic — the
  *caller* decides which callable is authorized.
- Malformed output (unparseable JSON, non-list, non-dict elements) →
  ``status="partial"`` with the parse failure recorded — never a crash.
- Every fact is a **candidate** until each of its quotes byte-verifies
  against the pinned unit slices in ``source_revisions.payload``:
  ``payload[unit.byte_start : unit.byte_end]`` must contain the quote's
  UTF-8 bytes exactly (V7-13.12). A claimed ``byte_start``/``byte_end``
  on a quote is a stricter pin: it must satisfy
  ``unit_slice[s:e] == quote`` — caller hints are never trusted blindly
  (``projections/units_v7`` precedent). ANY unverifiable quote → the
  fact is rejected and counted; it is never inserted as verified.
- ``commit_facts`` inserts verified candidates into ``t2_facts`` with
  ``verified=1``, ``model_id`` and ``prompt_digest`` (V7-13.13
  provenance). ``fact_id`` is content-addressed —
  ``t2:<blake2b(statement | sorted support unit_ids | model_id)>`` — so
  re-committing the same proposal is an idempotent no-op.
- ``occurred_*`` is resolved by ``temporal/v2`` **from the verified
  quote text** anchored on the pinned unit's ``recorded_at_us`` — a
  model-supplied date field is never stored (V7-13.13, V7-34.04).
- ``retire_facts`` is the V7-13.13 lifecycle sweep: a fact whose ANY
  pinned unit is no longer eligible — row gone (erased), or
  held/superseded per the injected ``is_unit_eligible`` callback
  (default: ``units``-table presence at ≤ ``generation``) — gets
  ``verified=0``. The row is kept: it stays auditable, is counted, and
  the delivery lane reads ``verified``. T2 is ADD-only (V7-13.14): no
  path here updates a statement or deletes a row.

The prompt template is versioned and published (V7-13.17):
``T2_PROMPT_V1`` is generic — no benchmark names, no question-type
hints, no dataset-specific examples — and ``prompt_digest`` is the
blake2b digest of its exact bytes, so two deployments on different
prompt text can never mint indistinguishable facts.

Unit dicts handed to :func:`propose_facts` carry at least
``{"unit_id", "text"}`` where ``text`` is the unit's verbatim pinned
slice (the caller slices ``source_revisions.payload`` — this module does
not fetch bytes for prompt construction). Optional ``speaker``/
``session_id``/``occurred`` keys are rendered as context lines.

Pure module: no store handle, no clock reads, no network. ``conn`` is
the caller's authorized read/write connection inside its transaction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ..core.types import json_dumps, safe_json_loads

# ---------------------------------------------------------------------------
# prompt + digests
# ---------------------------------------------------------------------------

#: Versioned extraction prompt (V7-13.17 — generic; no benchmark names,
#: no question-type hints, no dataset examples). ``{units_block}`` is the
#: only substitution point and is filled by ``propose_facts``.
T2_PROMPT_V1 = """\
You are extracting grounded facts from a conversation transcript.

You are given numbered units. Each unit is shown as `[unit_id] text`
where `text` is verbatim source bytes. Extract short factual statements
that are DIRECTLY supported by those bytes.

Output ONLY a JSON array. Each element MUST be an object:

{
  "statement": "<one short factual sentence>",
  "subject": "<canonical subject entity or speaker>",
  "predicate": "<short predicate, e.g. lives_in, prefers, works_at>",
  "object": "<object value, may be empty>",
  "state_key": "<optional stable key for an evolving fact, e.g. residence>",
  "unit_ids": ["<unit_id>", "..."],
  "quotes": [
    {"unit_id": "<unit_id>", "text": "<verbatim substring of that unit>",
     "byte_start": <optional int>, "byte_end": <optional int>}
  ]
}

Hard rules:
- Every quote `text` MUST be a verbatim substring of the unit's bytes —
  copied exactly, including casing and punctuation. Do not paraphrase,
  translate, normalize, or complete truncated words.
- `quotes` is REQUIRED and non-empty: a fact without verbatim support is
  worthless here and will be discarded.
- Every `unit_id` you cite MUST come from the provided units.
- If you supply `byte_start`/`byte_end` on a quote, they are offsets into
  that unit's text and must slice exactly to the quote text.
- Do not invent dates: if the text says "last March" keep the words in
  the quote and let the system resolve the interval.
- Emit nothing when nothing is supported: `[]` is a valid answer.

Units:
{units_block}

JSON:"""

T2_PROMPT_VERSION = "t2-extract/v1"

#: Domain separation for digests — distinct from normalize.py's
#: ``verbatim-n1`` and embeddings' ``verbatim-h1``.
_PERSON_PROMPT = b"verbatim-t2p"
_PERSON_FACT = b"verbatim-t2f"
_DIGEST_SIZE = 16

#: Bounds (V7-13.15 — T2 output is bounded and reported).
MAX_FACTS_PER_CALL = 64
MAX_UNITS_PER_CALL = 128
MAX_UNIT_TEXT_CHARS = 4096
MAX_STATEMENT_CHARS = 512
MAX_FIELD_CHARS = 256
MAX_OBJECT_CHARS = 1024
MAX_QUOTES_PER_FACT = 16
MAX_QUOTE_CHARS = 4096
MAX_UNIT_IDS_PER_FACT = 64
#: Upper bound on model output bytes we will attempt to parse.
MAX_OUTPUT_CHARS = 1 << 20

#: Digest of ``T2_PROMPT_V1`` — the default ``prompt_digest`` stamped on
#: rows when the caller doesn't supply one (custom prompts must pass
#: their own so provenance stays honest).
_PROMPT_DIGEST = hashlib.blake2b(
    T2_PROMPT_V1.encode("utf-8"), digest_size=_DIGEST_SIZE,
    person=_PERSON_PROMPT,
).hexdigest()

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_REJECTED = "rejected"


def prompt_digest(prompt: str = T2_PROMPT_V1) -> str:
    """blake2b hex digest of the exact prompt bytes (V7-13.13)."""
    return hashlib.blake2b(
        str(prompt).encode("utf-8"),
        digest_size=_DIGEST_SIZE,
        person=_PERSON_PROMPT,
    ).hexdigest()


def _fact_id(statement: str, unit_ids: Sequence[str], model_id: str) -> str:
    """Content-addressed fact id: ``t2:<hmac(statement|unit_ids|model)>``.

    Sorted unit ids make the id order-stable; identical re-proposals of
    the same statement over the same support set by the same model
    collapse to one row (idempotent commit).
    """
    canon = {
        "statement": statement,
        "unit_ids": sorted(str(u) for u in unit_ids),
        "model_id": str(model_id),
    }
    return "t2:" + hashlib.blake2b(
        json_dumps(canon).encode("utf-8"),
        digest_size=_DIGEST_SIZE,
        person=_PERSON_FACT,
    ).hexdigest()


def _canon(surface: str) -> str:
    """Subject canon via enrichment entities_v2 (lazy — parallel worker);
    documented fallback matches units_v7's."""
    try:
        from ..enrichment.entities_v2 import canon

        return canon(surface)
    except Exception:
        return str(surface if surface is not None else "").strip().casefold()


# ---------------------------------------------------------------------------
# result contracts
# ---------------------------------------------------------------------------


@dataclass
class FactVerdict:
    """Per-fact outcome of quote verification (or structural screening).

    ``verified`` is True only when the fact is structurally valid AND
    every quote byte-verified — it is then a ``verified=1`` insert
    candidate. ``quotes`` carries per-quote pin records:
    ``{text, unit_id, ok, reason, source_id, revision, byte_start,
    byte_end}`` where the byte range is the *resolved absolute* pin into
    ``source_revisions.payload`` (present only on success).
    """

    index: int
    fact: dict
    verified: bool
    reasons: list = field(default_factory=list)
    quotes: list = field(default_factory=list)
    support_unit_ids: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verified


@dataclass
class T2Result:
    """Honest extraction status (V7-13.18).

    ``status``: ``ok`` | ``partial`` | ``unavailable`` | ``rejected``.
    ``unavailable`` means the model did not produce usable output at all
    (absent or raising callable); ``partial`` means output existed but
    was malformed or partly invalid; ``rejected`` means parseable
    candidates that all failed structural validation. ``counts`` is a
    plain dict of ints so callers can journal it verbatim.
    """

    status: str
    reason: Optional[str] = None
    facts: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    model_id: Optional[str] = None
    prompt_digest: str = ""
    scope_id: Optional[str] = None
    generation: int = 0
    counts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


# ---------------------------------------------------------------------------
# structural validation (pre-byte-check, V7-13.12 "candidate until verified")
# ---------------------------------------------------------------------------


def _is_nonempty_str(v: Any, cap: int) -> bool:
    return isinstance(v, str) and 0 < len(v.strip()) <= cap


def _norm_quote(q: Any) -> Optional[dict]:
    """Normalize one quote item: ``str`` or ``{text|quote|span,
    unit_id?, byte_start?, byte_end?}`` → ``{text, unit_id, byte_start,
    byte_end}``; ``None`` when malformed."""
    if isinstance(q, str):
        text: Any = q
        unit_id = None
        bs = be = None
    elif isinstance(q, Mapping):
        text = q.get("text", q.get("quote", q.get("span")))
        unit_id = q.get("unit_id")
        bs = q.get("byte_start")
        be = q.get("byte_end")
    else:
        return None
    if not _is_nonempty_str(text, MAX_QUOTE_CHARS):
        return None
    if unit_id is not None and not isinstance(unit_id, str):
        return None
    if (bs is None) != (be is None):
        return None  # byte hints are a pair — a lone bound is malformed
    if bs is not None:
        if (
            not isinstance(bs, int)
            or not isinstance(be, int)
            or isinstance(bs, bool)
            or isinstance(be, bool)
            or not (0 <= bs < be)
        ):
            return None
    return {
        "text": text,
        "unit_id": unit_id,
        "byte_start": bs,
        "byte_end": be,
    }


def _structural_errors(f: Any, *, known_unit_ids=None) -> list:
    """All structural violations of one proposed fact (empty ⇒ valid)."""
    errs: list[str] = []
    if not isinstance(f, Mapping):
        return ["fact_not_object"]

    statement = f.get("statement")
    if not _is_nonempty_str(statement, MAX_STATEMENT_CHARS):
        errs.append("statement_missing_or_too_long")
    subject = f.get("subject", f.get("subject_canon"))
    if not _is_nonempty_str(subject, MAX_FIELD_CHARS):
        errs.append("subject_missing")
    predicate = f.get("predicate")
    if not _is_nonempty_str(predicate, MAX_FIELD_CHARS):
        errs.append("predicate_missing")
    obj = f.get("object")
    if obj is not None and not isinstance(obj, str):
        errs.append("object_not_text")
    elif isinstance(obj, str) and len(obj) > MAX_OBJECT_CHARS:
        errs.append("object_too_long")
    state_key = f.get("state_key")
    if state_key is not None and not _is_nonempty_str(
        state_key, MAX_FIELD_CHARS
    ):
        errs.append("state_key_invalid")

    unit_ids = f.get("unit_ids")
    if (
        not isinstance(unit_ids, (list, tuple))
        or not unit_ids
        or len(unit_ids) > MAX_UNIT_IDS_PER_FACT
        or any(not isinstance(u, str) or not u for u in unit_ids)
    ):
        errs.append("unit_ids_invalid")
        unit_id_set: set = set()
    else:
        unit_id_set = set(unit_ids)
        if known_unit_ids is not None and not unit_id_set <= set(
            known_unit_ids
        ):
            errs.append("unit_ids_unknown")

    quotes = f.get("quotes")
    if not isinstance(quotes, (list, tuple)) or not quotes:
        errs.append("quotes_missing")
    elif len(quotes) > MAX_QUOTES_PER_FACT:
        errs.append("quotes_too_many")
    else:
        for i, q in enumerate(quotes):
            nq = _norm_quote(q)
            if nq is None:
                errs.append(f"quote_{i}_malformed")
            elif (
                nq["unit_id"] is not None
                and unit_id_set
                and nq["unit_id"] not in unit_id_set
            ):
                errs.append(f"quote_{i}_unit_not_pinned")
    return errs


def validate_fact(f: Any, *, known_unit_ids=None) -> bool:
    """Structural validation BEFORE any byte check (V7-13.12).

    ``statement`` non-empty ≤512 chars; ``subject``/``predicate``
    present; ``quotes`` a non-empty list of well-formed quote items;
    ``unit_ids`` non-empty and ⊆ ``known_unit_ids`` when that set is
    provided. Returns a bare bool — ``_structural_errors`` carries the
    reasons for verdicts/journals.
    """
    return not _structural_errors(f, known_unit_ids=known_unit_ids)


def _normalize_fact(f: Mapping) -> dict:
    """Canonical in-memory fact shape (normalized, still a *candidate*)."""
    obj = f.get("object")
    state_key = f.get("state_key")
    return {
        "statement": str(f.get("statement")).strip(),
        "subject": str(f.get("subject", f.get("subject_canon"))).strip(),
        "predicate": str(f.get("predicate")).strip(),
        "object": obj.strip() if isinstance(obj, str) else None,
        "state_key": state_key.strip() if isinstance(state_key, str) else None,
        "unit_ids": [str(u) for u in f.get("unit_ids")],
        "quotes": [_norm_quote(q) for q in f.get("quotes")],
    }


# ---------------------------------------------------------------------------
# model call + output parsing
# ---------------------------------------------------------------------------


def _render_units_block(units: Sequence[Mapping]) -> str:
    """Deterministic ``[unit_id] text`` rendering of the provided units."""
    lines: list[str] = []
    for u in units[:MAX_UNITS_PER_CALL]:
        uid = u.get("unit_id") or u.get("id") or "?"
        text = str(u.get("text") if u.get("text") is not None else "")
        if len(text) > MAX_UNIT_TEXT_CHARS:
            text = text[:MAX_UNIT_TEXT_CHARS] + "…"
        ctx = []
        if u.get("speaker"):
            ctx.append(f"speaker={u['speaker']}")
        if u.get("session_id"):
            ctx.append(f"session={u['session_id']}")
        if u.get("occurred"):
            ctx.append(f"occurred={u['occurred']}")
        head = f"[{uid}]" + (" (" + ", ".join(ctx) + ")" if ctx else "")
        lines.append(f"{head} {text}")
    return "\n".join(lines)


def _strip_fence(text: str) -> str:
    """Remove one enclosing ``` fence if the model wrapped its JSON."""
    t = text.strip()
    if t.startswith("```"):
        # drop first line (``` or ```json) and a trailing ```
        nl = t.find("\n")
        if nl >= 0:
            t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    return t.strip()


def _parse_output(raw: Any) -> tuple:
    """``(parsed_list_or_None, error_or_None)`` — never raises."""
    if not isinstance(raw, str):
        return None, "model_returned_non_text"
    if len(raw) > MAX_OUTPUT_CHARS:
        return None, "model_output_too_large"
    text = _strip_fence(raw)
    if not text:
        return None, "model_output_empty"
    try:
        value = safe_json_loads(text)
    except Exception as exc:
        # Salvage path: prose-wrapped output — try the outermost [...]
        # span once, then give up honestly.
        lo, hi = text.find("["), text.rfind("]")
        if 0 <= lo < hi:
            try:
                value = safe_json_loads(text[lo : hi + 1])
            except Exception:
                return None, f"parse_error:{exc}"
        else:
            return None, f"parse_error:{exc}"
    if not isinstance(value, list):
        return None, "output_not_a_list"
    return value, None


def propose_facts(
    model_fn: Optional[Callable[[str], str]],
    units: Sequence[Mapping],
    *,
    scope_id: str,
    generation: int,
    now_us: int,
    model_id: Optional[str] = None,
    max_facts: int = MAX_FACTS_PER_CALL,
    prompt: str = T2_PROMPT_V1,
) -> T2Result:
    """Run the injected model over ``units`` → structurally-validated
    candidate facts (V7-13.12 ``extract`` step).

    ``model_fn`` is a plain ``prompt -> str`` callable — the caller owns
    transport, auth, and budgets; ``None`` (or a raising callable)
    reports ``unavailable`` honestly. Output is parsed strictly; a
    malformed response yields ``status="partial"`` and ``facts=[]`` with
    the failure recorded in ``reason`` — the caller is never crashed and
    nothing is fabricated.
    """
    digest = prompt_digest(prompt)
    mid = model_id or getattr(model_fn, "model_id", None) or (
        "unknown" if model_fn is not None else None
    )
    base = {
        "facts": [],
        "verdicts": [],
        "model_id": mid,
        "prompt_digest": digest,
        "scope_id": scope_id,
        "generation": generation,
    }
    if model_fn is None:
        return T2Result(
            status=STATUS_UNAVAILABLE,
            reason="no_model",
            counts={"proposed": 0, "valid": 0, "invalid": 0},
            **base,
        )
    if not callable(model_fn):
        return T2Result(
            status=STATUS_UNAVAILABLE,
            reason="model_not_callable",
            counts={"proposed": 0, "valid": 0, "invalid": 0},
            **base,
        )

    block = _render_units_block(units)
    if "{units_block}" in prompt:
        full_prompt = prompt.replace("{units_block}", block)
    else:
        # custom prompt without the slot — units go last, still verbatim
        full_prompt = prompt.rstrip() + "\n\nUnits:\n" + block + "\n\nJSON:\n"
    try:
        raw = model_fn(full_prompt)
    except Exception as exc:
        return T2Result(
            status=STATUS_UNAVAILABLE,
            reason=f"model_error:{type(exc).__name__}",
            counts={"proposed": 0, "valid": 0, "invalid": 0},
            **base,
        )

    parsed, err = _parse_output(raw)
    if err is not None:
        return T2Result(
            status=STATUS_PARTIAL,
            reason=err,
            counts={
                "proposed": 0,
                "valid": 0,
                "invalid": 0,
                "parse_failed": 1,
            },
            **base,
        )

    known_ids = {
        str(u.get("unit_id") or u.get("id"))
        for u in units
        if isinstance(u, Mapping) and (u.get("unit_id") or u.get("id"))
    }
    facts: list[dict] = []
    verdicts: list[FactVerdict] = []
    invalid = 0
    dropped = 0
    for i, item in enumerate(parsed):
        errs = _structural_errors(item, known_unit_ids=known_ids or None)
        if errs:
            invalid += 1
            verdicts.append(
                FactVerdict(
                    index=i,
                    fact=dict(item) if isinstance(item, Mapping) else {},
                    verified=False,
                    reasons=errs,
                )
            )
            continue
        if len(facts) >= max_facts:
            dropped += 1
            verdicts.append(
                FactVerdict(
                    index=i,
                    fact={},
                    verified=False,
                    reasons=["dropped_over_max_facts"],
                )
            )
            continue
        nf = _normalize_fact(item)
        facts.append(nf)
        verdicts.append(
            FactVerdict(
                index=i,
                fact=nf,
                verified=False,
                reasons=["pending_byte_verification"],
            )
        )
        # ^ a proposed fact is a *candidate* until ``verify_quotes``
        # byte-verifies every quote — ``verified=True`` is only ever set
        # there, so piping propose-verdicts straight into ``commit_facts``
        # can never insert an unchecked fact.

    counts = {
        "proposed": len(parsed),
        "valid": len(facts),
        "invalid": invalid,
        "dropped_over_cap": dropped,
        "parse_failed": 0,
    }
    if not parsed:
        status, reason = STATUS_OK, None  # empty extraction is a valid answer
    elif not facts:
        status, reason = STATUS_REJECTED, "all_facts_invalid"
    elif invalid or dropped:
        status, reason = STATUS_PARTIAL, "some_facts_invalid"
    else:
        status, reason = STATUS_OK, None
    return T2Result(
        status=status,
        reason=reason,
        facts=facts,
        verdicts=verdicts,
        counts=counts,
        **{k: v for k, v in base.items() if k not in ("facts", "verdicts")},
    )


# ---------------------------------------------------------------------------
# quote verification — the deterministic core (V7-13.12)
# ---------------------------------------------------------------------------


def _unit_row(conn, scope_id: str, unit_id: str, generation: int):
    """Latest units row for ``unit_id`` at ≤ ``generation`` in scope."""
    row = conn.execute(
        "SELECT source_id, revision, byte_start, byte_end, recorded_at_us"
        " FROM units"
        " WHERE scope_id = ? AND unit_id = ? AND generation <= ?"
        " ORDER BY generation DESC LIMIT 1",
        (scope_id, unit_id, int(generation)),
    ).fetchone()
    if row is None:
        return None
    return {
        "source_id": row[0],
        "revision": row[1],
        "byte_start": row[2],
        "byte_end": row[3],
        "recorded_at_us": row[4],
    }


def _payload(conn, source_id: str, revision: int) -> Optional[bytes]:
    row = conn.execute(
        "SELECT payload FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (source_id, int(revision)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return bytes(row[0])


def _unit_slice(conn, urow: Mapping) -> Optional[bytes]:
    """The unit's pinned byte slice of its revision payload; ``None``
    when the unit carries no pins (``unsupported_extraction`` — it can
    never back a quote) or the payload is gone."""
    bs, be = urow.get("byte_start"), urow.get("byte_end")
    if bs is None or be is None or not (0 <= int(bs) < int(be)):
        return None
    body = _payload(conn, urow["source_id"], urow["revision"])
    if body is None or int(be) > len(body):
        return None
    return body[int(bs) : int(be)]


def _verify_one_quote(
    qtext: str,
    qbs: Optional[int],
    qbe: Optional[int],
    candidate_unit_ids: Iterable[str],
    unit_rows: Mapping,
    conn,
) -> tuple:
    """``(ok, record)`` for one quote against the pinned unit slices.

    Byte offsets inside a quote are interpreted relative to the unit's
    pinned slice (what the model was shown as that unit's text).
    """
    qbytes = qtext.encode("utf-8")
    saw_unpinned = False
    saw_range_mismatch = False
    for uid in candidate_unit_ids:
        urow = unit_rows.get(uid)
        if urow is None:
            continue
        uslice = _unit_slice(conn, urow)
        if uslice is None:
            saw_unpinned = True
            continue
        if qbs is not None:
            # Claimed range is a strict pin relative to the unit slice —
            # the quote must slice exactly there; substring elsewhere
            # does not rescue a false pin claim.
            if qbe <= len(uslice) and uslice[qbs:qbe] == qbytes:
                return True, {
                    "unit_id": uid,
                    "source_id": urow["source_id"],
                    "revision": urow["revision"],
                    "byte_start": int(urow["byte_start"]) + qbs,
                    "byte_end": int(urow["byte_start"]) + qbe,
                }
            if qbytes in uslice:
                saw_range_mismatch = True
            continue
        idx = uslice.find(qbytes)
        if idx >= 0:
            return True, {
                "unit_id": uid,
                "source_id": urow["source_id"],
                "revision": urow["revision"],
                "byte_start": int(urow["byte_start"]) + idx,
                "byte_end": int(urow["byte_start"]) + idx + len(qbytes),
            }
    if saw_range_mismatch:
        reason = "claimed_byte_range_mismatch"
    elif saw_unpinned:
        reason = "unit_unpinned_or_payload_gone"
    else:
        reason = "quote_not_verbatim"
    return False, {"reason": reason}


def verify_quotes(conn, proposed: Sequence[Mapping], *, scope_id: str,
                  generation: int) -> list:
    """Byte-verify every quote of every proposed fact against the pinned
    unit slices (V7-13.12).

    A fact is ``verified`` only when it is structurally valid AND every
    quote is an exact substring of at least one pinned unit's byte slice
    (and satisfies its claimed byte range when one was asserted). Any
    unverifiable quote → the fact is rejected with reasons; nothing is
    inserted here — ``commit_facts`` consumes these verdicts.
    """
    verdicts: list[FactVerdict] = []
    for i, raw in enumerate(proposed):
        errs = _structural_errors(raw)
        if errs:
            verdicts.append(
                FactVerdict(
                    index=i,
                    fact=dict(raw) if isinstance(raw, Mapping) else {},
                    verified=False,
                    reasons=errs,
                )
            )
            continue
        fact = _normalize_fact(raw)

        # Resolve every claimed support unit up front: a fact pinning a
        # unit that does not resolve is claiming support we cannot check.
        unit_rows: dict = {}
        missing: list[str] = []
        for uid in fact["unit_ids"]:
            urow = _unit_row(conn, scope_id, uid, generation)
            if urow is None:
                missing.append(uid)
            else:
                unit_rows[uid] = urow
        if missing:
            verdicts.append(
                FactVerdict(
                    index=i,
                    fact=fact,
                    verified=False,
                    reasons=[f"unit_not_found:{u}" for u in missing],
                )
            )
            continue

        quote_records: list[dict] = []
        reasons: list[str] = []
        support: set = set()
        for qi, q in enumerate(fact["quotes"]):
            cands = [q["unit_id"]] if q["unit_id"] else fact["unit_ids"]
            ok, rec = _verify_one_quote(
                q["text"], q["byte_start"], q["byte_end"], cands,
                unit_rows, conn,
            )
            qrec = {"text": q["text"], "ok": ok, **rec}
            quote_records.append(qrec)
            if ok:
                support.add(rec["unit_id"])
            else:
                reasons.append(f"quote_{qi}:{rec['reason']}")
        verdicts.append(
            FactVerdict(
                index=i,
                fact=fact,
                verified=not reasons,
                reasons=reasons,
                quotes=quote_records,
                support_unit_ids=sorted(support),
            )
        )
    return verdicts


# ---------------------------------------------------------------------------
# commit + retire
# ---------------------------------------------------------------------------


def _resolve_occurred(verdict: FactVerdict, conn, scope_id: str,
                      generation: int, now_us: Optional[int]):
    """``(start_us, end_us)`` resolved by ``temporal/v2`` from the
    verified quote texts — the model's own date claims are never stored
    (V7-13.13). Anchor: the pinned unit's ``recorded_at_us``, then
    ``now_us``. ``(None, None)`` when nothing resolves."""
    try:
        from ..enrichment import temporal_v2
    except Exception:
        return None, None
    starts: list[int] = []
    ends: list[int] = []
    anchor_cache: dict = {}
    for q in verdict.quotes:
        if not q.get("ok"):
            continue
        uid = q.get("unit_id")
        anchor = anchor_cache.get(uid)
        if uid is not None and uid not in anchor_cache:
            urow = _unit_row(conn, scope_id, uid, generation)
            anchor = (
                urow.get("recorded_at_us") if urow else None
            ) or now_us
            anchor_cache[uid] = anchor
        if anchor is None:
            continue
        try:
            for rt in temporal_v2.resolve(q["text"], int(anchor)):
                iv = rt.interval
                if iv.start_us is not None and iv.end_us is not None:
                    starts.append(iv.start_us)
                    ends.append(iv.end_us)
        except Exception:
            continue
    if not starts:
        return None, None
    return min(starts), max(ends)


def commit_facts(conn, verdicts: Sequence[FactVerdict], *, scope_id: str,
                 generation: int, model_id: str,
                 prompt_digest: Optional[str] = None,
                 now_us: Optional[int] = None) -> dict:
    """Insert ``verified`` verdicts into ``t2_facts`` (V7-13.12/13).

    Idempotent: ``fact_id`` is content-addressed, so re-committing an
    identical proposal inserts nothing new (``skipped_existing``).
    Rejected verdicts are counted with their reasons — never stored as
    verified. ``unit_ids_json`` records the *actual* support set (units
    that pinned a quote), so retirement sweeps track real dependencies.
    Returns ``{inserted, skipped_existing, rejected, reasons, fact_ids}``.
    """
    pdigest = prompt_digest if prompt_digest is not None else _PROMPT_DIGEST
    inserted = 0
    skipped = 0
    rejected = 0
    reasons: dict = {}
    fact_ids: list[str] = []
    for v in verdicts:
        if not v.verified:
            rejected += 1
            reasons[str(v.index)] = list(v.reasons)
            continue
        f = v.fact
        support = v.support_unit_ids or list(f["unit_ids"])
        fid = _fact_id(f["statement"], support, model_id)
        exists = conn.execute(
            "SELECT 1 FROM t2_facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        if exists is not None:
            skipped += 1
            fact_ids.append(fid)
            continue
        occ_start, occ_end = _resolve_occurred(
            v, conn, scope_id, generation, now_us
        )
        quotes_json = json_dumps(
            [
                {
                    "text": q["text"],
                    "unit_id": q.get("unit_id"),
                    "source_id": q.get("source_id"),
                    "revision": q.get("revision"),
                    "byte_start": q.get("byte_start"),
                    "byte_end": q.get("byte_end"),
                }
                for q in v.quotes
                if q.get("ok")
            ]
        )
        conn.execute(
            "INSERT INTO t2_facts (fact_id, scope_id, unit_ids_json,"
            " statement, quotes_json, subject_canon, predicate, object,"
            " occurred_start_us, occurred_end_us, state_key, model_id,"
            " prompt_digest, verified, generation)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
            (
                fid,
                scope_id,
                json_dumps(sorted(support)),
                f["statement"],
                quotes_json,
                _canon(f["subject"]),
                f["predicate"],
                f["object"],
                occ_start,
                occ_end,
                f["state_key"],
                model_id,
                pdigest,
                int(generation),
            ),
        )
        inserted += 1
        fact_ids.append(fid)
    return {
        "inserted": inserted,
        "skipped_existing": skipped,
        "rejected": rejected,
        "reasons": reasons,
        "fact_ids": fact_ids,
    }


def _unit_present(conn, scope_id: str, unit_id: str, generation: int) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM units"
            " WHERE scope_id = ? AND unit_id = ? AND generation <= ?"
            " LIMIT 1",
            (scope_id, unit_id, int(generation)),
        ).fetchone()
        is not None
    )


def retire_facts(conn, *, scope_id: str, generation: int,
                 is_unit_eligible: Optional[Callable[[str], bool]] = None
                 ) -> dict:
    """V7-13.13 retirement sweep: a verified fact whose ANY pinned unit
    is no longer eligible — row gone (erased), or held/superseded per the
    injected ``is_unit_eligible`` predicate (default: ``units`` presence
    at ≤ ``generation``) — gets ``verified=0``. The row is never deleted:
    it stays auditable and counted; the delivery lane reads ``verified``.

    Returns ``{checked, retired, retired_ids, reasons}`` where ``reasons``
    maps fact_id → the ineligible unit ids that retired it.
    """
    rows = conn.execute(
        "SELECT fact_id, unit_ids_json FROM t2_facts"
        " WHERE scope_id = ? AND verified = 1",
        (scope_id,),
    ).fetchall()

    def eligible(uid: str) -> bool:
        if is_unit_eligible is not None:
            try:
                return bool(is_unit_eligible(uid))
            except Exception:
                return False  # a failing eligibility probe is not proof
        return _unit_present(conn, scope_id, uid, generation)

    checked = 0
    retired_ids: list[str] = []
    reasons: dict = {}
    for fact_id, ujson in rows:
        checked += 1
        try:
            unit_ids = json.loads(ujson) if ujson else []
        except Exception:
            unit_ids = []
        bad = [u for u in unit_ids if not eligible(str(u))]
        if not bad:
            continue
        conn.execute(
            "UPDATE t2_facts SET verified = 0 WHERE fact_id = ?",
            (fact_id,),
        )
        retired_ids.append(fact_id)
        reasons[fact_id] = [f"unit_ineligible:{u}" for u in bad]
    return {
        "checked": checked,
        "retired": len(retired_ids),
        "retired_ids": retired_ids,
        "reasons": reasons,
    }
