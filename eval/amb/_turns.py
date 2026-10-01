"""AMB turn-list parsing + the provider-side session index — SPEC_V8.5
§2 (V85-02.02/02.04/02.05).

Two responsibilities, both pure-stdlib and deterministic:

* **Turn parsing.**  An AMB session ``Document`` carries its dialogue as
  ``json.dumps(turns)`` in ``content`` (LoCoMo ``session_N`` lists:
  ``{speaker, dia_id, text[, blip_caption, img_url, ...]}``) or, on some
  datasets, a structured ``messages`` field.  :func:`parse_turn_list`
  recognizes both shapes and normalizes each turn to
  ``{speaker, text, at, dia_id, caption, raw}`` — ``raw`` preserving the
  original dict keys verbatim for the ``Memory.add(messages=...)``
  contract (the facade persists caller keys into revision metadata, so
  ``dia_id`` survives inside the store *and* inside this index).

* **Session index.**  :class:`SessionIndex` is the provider-owned map
  ``doc.id -> ordered [(ordinal, speaker, text, dia_id, caption,
  unit_id, seq, byte_start, byte_end)]`` plus reverse indexes
  ``unit_id -> (doc_id, ordinal)`` and ``source_id -> doc_id``.  It is
  what makes ±``W_r`` neighbor expansion independent of engine
  internals: retrieval resolves a hit's ``vobj1.unit.<id>.<rev>`` ref to
  a session turn without trusting claim/claim-engine ids.  The index
  persists as one JSON sidecar per bank next to the store file, so a
  ``--skip-ingestion`` resume run expands correctly without re-adding.

Byte ranges (``bs``/``be``) come from the real ``units`` projection rows
attached after the settle drain — the same pins the delivered quotes
were sliced from — which lets a ``session``/``episode``/
``sentence_window`` unit hit resolve to *all* the turn ordinals its
byte span covers, not a guessed center.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# turn field conventions
# ---------------------------------------------------------------------------

#: text-bearing keys, in priority order (mirrors the facade's
#: ``_MSG_TEXT_KEYS`` so what we render is what the engine indexed).
TEXT_KEYS = ("text", "content", "message")

#: speaker identity keys (mirrors ``_MSG_SPEAKER_KEYS`` + ``role``).
SPEAKER_KEYS = ("speaker", "speaker_id", "name", "author", "role")

#: per-message time keys the facade validates (``_MSG_TIME_KEYS``).
TIME_KEYS = (
    "at",
    "message_at",
    "timestamp",
    "ts",
    "recorded_at_us",
    "event_us",
    "message_at_us",
)

#: image-caption keys seen in the wild (LoCoMo ``blip_caption``).
CAPTION_KEYS = (
    "blip_caption",
    "img_caption",
    "image_caption",
    "caption",
)

#: turn identity keys (LoCoMo ``dia_id`` first-class).
DIA_KEYS = ("dia_id", "turn_id", "dialog_id")


def _first_key(d: Mapping, keys: Iterable[str]) -> Any:
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return None


def normalize_turn(raw: Any, i: int = 0) -> Optional[Dict[str, Any]]:
    """One raw turn element → the normalized record, or ``None`` when
    the element carries no text (mirrors ``units_v7._turn_record``'s
    skip rule — ``None``/empty content emits no turn unit)."""
    if isinstance(raw, str):
        if raw == "":
            return None
        return {
            "speaker": None, "text": raw, "at": None,
            "dia_id": None, "caption": None, "raw": {"text": raw},
        }
    if not isinstance(raw, Mapping):
        return None
    text = _first_key(raw, TEXT_KEYS)
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if text == "":
        return None
    speaker = _first_key(raw, SPEAKER_KEYS)
    if speaker is not None:
        speaker = str(speaker).strip() or None
    at = _first_key(raw, TIME_KEYS)
    dia = _first_key(raw, DIA_KEYS)
    caption = _first_key(raw, CAPTION_KEYS)
    msg = dict(raw)  # caller keys pass through to the facade verbatim
    # canonical keys so the facade's contract finds them regardless of
    # which synonym the dataset used
    if "text" not in msg:
        msg["text"] = text
    if speaker is not None and not any(
        k in raw for k in ("speaker", "speaker_id", "name", "author")
    ):
        msg["speaker"] = speaker
    return {
        "speaker": speaker,
        "text": text,
        "at": at,
        "dia_id": None if dia is None else str(dia),
        "caption": None if caption is None else str(caption),
        "raw": msg,
    }


def parse_turn_list(content: Any) -> Optional[List[Dict[str, Any]]]:
    """``json.dumps``-ed turn list → normalized turns; ``None`` when the
    content is not a turn list (prose documents keep the blob path).

    A list qualifies only when every element is a dict or string and at
    least one element carries text — a bare list of numbers/tags is not
    dialogue and must not be split into pseudo-turns."""
    if not isinstance(content, str) or not content.strip():
        return None
    s = content.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        return None
    return turns_from(obj)


def turns_from(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize a decoded ``messages``/turn-list value; ``None`` unless
    it is a non-empty list of dicts/strings with ≥1 text-carrying turn."""
    if isinstance(obj, (dict, str)):
        obj = [obj]
    if not isinstance(obj, (list, tuple)) or not obj:
        return None
    turns: List[Dict[str, Any]] = []
    any_text = False
    for i, raw in enumerate(obj):
        if not isinstance(raw, (dict, str)):
            return None
        t = normalize_turn(raw, i)
        if t is not None:
            any_text = True
            turns.append(t)
    if not any_text:
        return None
    return turns


def messages_for_add(turns: List[Dict[str, Any]]) -> List[Any]:
    """The ``Memory.add(messages=...)`` payload: the original dicts with
    canonical ``text``/``speaker`` keys (``dia_id``, ``blip_caption`` and
    friends pass through into revision metadata — that is where
    ``dia_id`` lives inside the store)."""
    return [t["raw"] if isinstance(t["raw"], dict) else t["text"]
            for t in turns]


def transcript_for_add(turns: List[Dict[str, Any]]) -> str:
    """The ``content`` payload: one ``Speaker: text`` line per turn so
    every message text appears in the payload verbatim and the units
    projection's byte pins resolve (``_locate`` over the payload)."""
    return "\n".join(render_turn_line(t, caption=False) for t in turns)


# ---------------------------------------------------------------------------
# chat-prose parsing (BEAM ``_format_chat`` shape)
# ---------------------------------------------------------------------------

#: One ``_format_chat`` line: ``[<meta>] <Role>: <text>`` where ``<meta>``
#: is ``<anchor> | Turn <n>`` (either part optional).  Turn blocks split
#: on blank lines; a block failing the pattern folds into the preceding
#: turn's body (message text may itself contain blank lines).
_CHAT_TURN_RE = re.compile(
    r"^(?:\[(?P<meta>[^\]\n]+)\][ \t]*)?"
    r"(?P<role>[A-Za-z][A-Za-z0-9_]{0,39})"
    r":[ \t]*(?P<text>.*)$",
    re.DOTALL,
)
_CHAT_META_TURN_RE = re.compile(r"^Turn\s+(?P<tid>\S+)\s*$")


def _anchor_iso(anchor: Any) -> Optional[str]:
    """BEAM ``time_anchor`` (``March-15-2024``) → ISO-8601 day; ``None``
    when unparseable (never a fabricated date)."""
    if not anchor:
        return None
    a = re.sub(
        r"\s+",
        " ",
        str(anchor).strip().replace("_", " ").replace("-", " ")
        .replace(",", " "),
    )
    for fmt in ("%B %d %Y", "%b %d %Y", "%Y %m %d", "%d %B %Y",
                "%d %b %Y"):
        try:
            return datetime.strptime(a, fmt).replace(
                tzinfo=timezone.utc
            ).strftime("%Y-%m-%dT00:00:00+00:00")
        except ValueError:
            continue
    return None


def _chat_meta(meta: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``<anchor> | Turn <n>`` → ``(anchor, turn_id)``; either may be
    ``None``."""
    if not meta:
        return None, None
    anchor_parts: List[str] = []
    tid: Optional[str] = None
    for part in meta.split("|"):
        part = part.strip()
        m = _CHAT_META_TURN_RE.match(part)
        if m is not None and tid is None:
            tid = m.group("tid")
        elif part:
            anchor_parts.append(part)
    return (" | ".join(anchor_parts) or None), tid


def parse_chat_lines(content: Any) -> Optional[List[Dict[str, Any]]]:
    """``_format_chat``-style prose → normalized turns; ``None`` when the
    content is not turn-formatted (plain prose keeps the blob path).

    A document qualifies only when the first ``\\n\\n``-delimited block
    parses as ``[<meta>] <Role>: <text>`` and the run yields ≥2 turns,
    or a single turn carrying a bracket prefix.  The ``[anchor | Turn
    n]`` prefix is kept inside ``text`` so unit bytes and delivered
    lines preserve the ordering markers the dataset embeds
    deliberately."""
    if not isinstance(content, str) or not content.strip():
        return None
    blocks = [b for b in re.split(r"\n\n+", content.strip()) if b.strip()]
    parsed: List[Dict[str, Any]] = []
    bracketed = 0
    for i, block in enumerate(blocks):
        m = _CHAT_TURN_RE.match(block)
        if m is None or not m.group("text").strip():
            if i == 0 or not parsed:
                return None
            # continuation of the previous turn's body — appended
            # verbatim, blank-line separator included.
            parsed[-1]["text"] += "\n\n" + block
            parsed[-1]["raw"]["text"] = parsed[-1]["text"]
            continue
        meta = m.group("meta")
        anchor, tid = _chat_meta(meta)
        prefix = f"[{meta.strip()}] " if meta else ""
        text = prefix + m.group("text").strip()
        if meta:
            bracketed += 1
        raw: Dict[str, Any] = {
            "speaker": m.group("role"),
            "text": text,
            "turn_id": tid,
            "dia_id": tid,
            "time_anchor": anchor,
        }
        at = _anchor_iso(anchor)
        if at is not None:
            raw["at"] = at
        t = normalize_turn(raw, i)
        if t is None:
            return None
        parsed.append(t)
    if len(parsed) < 2 and not bracketed:
        return None
    return parsed or None


# ---------------------------------------------------------------------------
# header + line rendering (V85-02.05)
# ---------------------------------------------------------------------------

_CTX_RE = re.compile(
    r"^\s*(?P<conv>.*?)\s*"
    r"\(\s*(?P<label>session[_\s-]?\d+)\s+of\s+(?P<cid>[^)]+)\)\s*$",
    re.IGNORECASE,
)
_ID_RE = re.compile(
    r"^(?P<conv>.+?)[_\-\s]?session[_\s-]?(?P<n>\d+)\s*$", re.IGNORECASE
)
_LABEL_N_RE = re.compile(r"(\d+)")


def session_label(doc_id: Any, context: Any) -> Dict[str, Any]:
    """``(conversation, session_label, session_n, conv_id)`` for the
    ``[<conversation> · session <n> · <date>]`` header.

    AMB LoCoMo supplies ``context`` = ``"Conversation between A and B
    (session_7 of conv-26)"`` and ``id`` = ``"conv-26_session_7"``; both
    parse to the same parts.  Unparseable inputs degrade honestly to the
    doc id as the conversation name with no session ordinal."""
    did = str(doc_id or "")
    ctx = str(context or "").strip()
    out: Dict[str, Any] = {
        "conversation": ctx or did or "verbatim",
        "session_label": None,
        "session_n": None,
        "conv_id": None,
    }
    m = _CTX_RE.match(ctx) if ctx else None
    if m:
        conv = m.group("conv").strip()
        out["conversation"] = conv or m.group("cid").strip()
        out["conv_id"] = m.group("cid").strip()
        n = _LABEL_N_RE.search(m.group("label"))
        if n:
            out["session_n"] = int(n.group(1))
            out["session_label"] = f"session {int(n.group(1))}"
        return out
    m = _ID_RE.match(did)
    if m:
        out["conversation"] = ctx or m.group("conv")
        out["session_n"] = int(m.group("n"))
        out["session_label"] = f"session {int(m.group('n'))}"
        out["conv_id"] = m.group("conv")
        return out
    return out


def date_label(timestamp: Any) -> Optional[str]:
    """``YYYY-MM-DD, Weekday`` for the header; ``None`` when the
    timestamp does not parse (never a fabricated date)."""
    if timestamp is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(timestamp, datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)) and not isinstance(
        timestamp, bool
    ):
        v = float(timestamp)
        # µs / ms / s magnitude heuristic (same convention as _as_us)
        if v >= 10**14:
            v = v / 1_000_000.0
        elif v >= 10**11:
            v = v / 1000.0
        try:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(timestamp, str):
        s = timestamp.strip()
        if not s:
            return None
        if s.isdigit():
            return date_label(int(s))
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d, %A")


def render_turn_line(turn: Mapping[str, Any], *, caption: bool = True) -> str:
    """``Speaker: text`` plus `` [image: <caption>]`` when present."""
    text = str(turn.get("text") or "")
    spk = turn.get("speaker")
    line = f"{spk}: {text}" if spk else text
    cap = turn.get("caption")
    if caption and cap:
        line = f"{line} [image: {cap}]"
    return line


def render_header(sess: Mapping[str, Any]) -> str:
    """``[<conversation> · session <n> · <YYYY-MM-DD, Weekday>]`` —
    parts that failed to parse are omitted rather than invented."""
    parts: List[str] = []
    conv = sess.get("conversation") or sess.get("doc_id") or "verbatim"
    parts.append(str(conv))
    label = sess.get("session_label")
    if label:
        parts.append(str(label))
    dl = sess.get("date_label")
    if dl:
        parts.append(str(dl))
    elif sess.get("timestamp"):
        parts.append(str(sess["timestamp"]))
    return "[" + " · ".join(parts) + "]"


# ---------------------------------------------------------------------------
# SessionIndex — the provider-side neighbor-expansion map
# ---------------------------------------------------------------------------

INDEX_SCHEMA = "amb_session_index/v1"

#: In-memory session record shape (serialized identically):
#:   {"doc_id", "source_ids": [...], "conversation", "conv_id",
#:    "session_label", "session_n", "timestamp", "date_label",
#:    "context", "user_id", "generation", "units_attached",
#:    "turns": [{"i", "speaker", "text", "dia_id", "caption",
#:               "unit_id", "seq", "bs", "be"}]}


class SessionIndex:
    """``doc.id -> session record`` for one memory bank.

    ``unit_index`` maps a projected turn ``unit_id`` to
    ``"<doc_id>:<ordinal>"`` so a V7 ``vobj1.unit.*`` hit resolves to a
    session position in O(1); ``source_index`` maps a verbatim
    ``source_id`` to ``doc_id`` for source/claim-level hits.  Everything
    serializes to plain JSON — the sidecar is authoritative on resume
    and never reaches into engine internals.
    """

    def __init__(self, bank_key: str = "_shared") -> None:
        self.bank_key = bank_key
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.unit_index: Dict[str, str] = {}
        self.source_index: Dict[str, str] = {}
        self.generation: Optional[int] = None
        self.notes: List[str] = []

    # -- ingest-side ---------------------------------------------------

    def add_session(
        self,
        doc_id: str,
        turns: List[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register one ingested session document.  ``turns`` are the
        normalized records (already filtered to text-carrying)."""
        parts = session_label(doc_id, context)
        sess = self.sessions.get(doc_id)
        if sess is None:
            sess = {
                "doc_id": doc_id,
                "source_ids": [],
                "conversation": parts["conversation"],
                "conv_id": parts["conv_id"],
                "session_label": parts["session_label"],
                "session_n": parts["session_n"],
                "timestamp": timestamp,
                "date_label": date_label(timestamp),
                "context": context,
                "user_id": user_id,
                "generation": None,
                "units_attached": 0,
                "turns": [],
            }
            self.sessions[doc_id] = sess
        base = len(sess["turns"])
        for j, t in enumerate(turns):
            sess["turns"].append({
                "i": base + j,
                "speaker": t.get("speaker"),
                "text": t.get("text"),
                "dia_id": t.get("dia_id"),
                "caption": t.get("caption"),
                "at": (str(t["at"]) if t.get("at") is not None else None),
                "unit_id": None,
                "seq": None,
                "bs": None,
                "be": None,
            })
        return sess

    def attach_source(self, doc_id: str, source_id: str) -> None:
        sess = self.sessions.get(doc_id)
        if sess is not None:
            if source_id and source_id not in sess["source_ids"]:
                sess["source_ids"].append(source_id)
        if source_id:
            self.source_index[source_id] = doc_id

    def attach_units(
        self,
        doc_id: str,
        unit_rows: Iterable[Mapping[str, Any]],
        *,
        generation: Optional[int] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """Zip the store's turn-unit rows onto this session's unattached
        turns, in emitted order.  ``unit_rows`` items carry
        ``unit_id``/``seq``/``byte_start``/``byte_end`` (and optionally
        ``source_id`` for the source back-map).  Returns the number of
        turns that received ids this call."""
        sess = self.sessions.get(doc_id)
        if sess is None:
            return 0
        rows = list(unit_rows)
        pending = [t for t in sess["turns"] if t.get("unit_id") is None]
        if not pending:
            return 0  # already attached (e.g. a dedup replay) — no note
        n = 0
        for t, row in zip(pending, rows):
            uid = row.get("unit_id")
            if not uid:
                continue
            t["unit_id"] = str(uid)
            t["seq"] = row.get("seq")
            t["bs"] = row.get("byte_start")
            t["be"] = row.get("byte_end")
            self.unit_index[str(uid)] = [doc_id, int(t["i"])]
            sid = row.get("source_id") or source_id
            if sid:
                # byte pins + seq are relative to THEIR source payload —
                # tag the turn so covered_ordinals/ordinal_by_seq stay
                # unambiguous under chunked (>512-message) adds.
                t["src"] = str(sid)
                self.attach_source(doc_id, str(sid))
            n += 1
        sess["units_attached"] = bool(sess["turns"]) and all(
            t.get("unit_id") for t in sess["turns"]
        )
        if generation is not None:
            sess["generation"] = generation
            self.generation = generation
        if len(rows) != len(pending):
            self.notes.append(
                f"{doc_id}: {len(rows)} turn units vs "
                f"{len(pending)} pending turns — index partially attached"
            )
        return n

    # -- query-side -----------------------------------------------------

    def session(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        return self.sessions.get(str(doc_id))

    def doc_for_source(self, source_id: Any) -> Optional[str]:
        return self.source_index.get(str(source_id))

    def resolve_unit(self, unit_id: Any) -> Optional[Tuple[str, int]]:
        """``unit_id -> (doc_id, ordinal)`` for a projected turn unit."""
        ref = self.unit_index.get(str(unit_id))
        if ref is None:
            return None
        try:
            doc_id, i = ref
            return str(doc_id), int(i)
        except (TypeError, ValueError, IndexError):
            return None

    def covered_ordinals(
        self, doc_id: str, bs: Optional[int], be: Optional[int],
        source_id: Any = None,
    ) -> List[int]:
        """Turn ordinals whose byte range intersects ``[bs, be)`` — how
        a session/episode/sentence-window unit hit maps to the turns it
        actually covers.  Byte pins are relative to their own source
        payload, so turns tagged with a different ``src`` are skipped."""
        sess = self.sessions.get(doc_id)
        if sess is None or bs is None or be is None:
            return []
        sid = None if source_id is None else str(source_id)
        out = [
            t["i"]
            for t in sess["turns"]
            if t.get("bs") is not None and t.get("be") is not None
            and (sid is None or t.get("src") in (None, sid))
            and t["bs"] < be and bs < t["be"]
        ]
        return out

    def ordinal_by_seq(
        self, doc_id: str, seq: Any, source_id: Any = None
    ) -> Optional[int]:
        sess = self.sessions.get(doc_id)
        if sess is None or not isinstance(seq, int):
            return None
        sid = None if source_id is None else str(source_id)
        fallback: Optional[int] = None
        for t in sess["turns"]:
            if t.get("seq") == seq:
                if sid is None or t.get("src") == sid:
                    return t["i"]
                if fallback is None and t.get("src") is None:
                    fallback = t["i"]
        return fallback

    def ordinal_by_quote(self, doc_id: str, quote: Any) -> Optional[int]:
        """Match a delivered quote back to a turn ordinal: exact first,
        then containment either way (sentence-window slices are strict
        substrings of their parent turn's text)."""
        sess = self.sessions.get(doc_id)
        if sess is None:
            return None
        q = str(quote or "").strip()
        if not q:
            return None
        turns = sess["turns"]
        for t in turns:
            if str(t.get("text") or "").strip() == q:
                return t["i"]
        for t in turns:
            text = str(t.get("text") or "").strip()
            if text and (q in text or text in q):
                return t["i"]
        return None

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": INDEX_SCHEMA,
            "bank": self.bank_key,
            "generation": self.generation,
            "sessions": self.sessions,
            "unit_index": self.unit_index,
            "source_index": self.source_index,
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> "SessionIndex":
        idx = cls(str(blob.get("bank") or "_shared"))
        idx.sessions = {
            str(k): dict(v)
            for k, v in (blob.get("sessions") or {}).items()
            if isinstance(v, Mapping)
        }
        idx.unit_index = {
            str(k): [str(v[0]), int(v[1])]
            for k, v in (blob.get("unit_index") or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        }
        idx.source_index = {
            str(k): str(v)
            for k, v in (blob.get("source_index") or {}).items()
        }
        g = blob.get("generation")
        idx.generation = g if isinstance(g, int) else None
        return idx

    def save(self, path: Any) -> None:
        """Atomic write (tmp + replace) — a torn sidecar must never
        leave half a session map behind."""
        import os

        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, sort_keys=True,
                      ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Any) -> Optional["SessionIndex"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(blob, Mapping) or "sessions" not in blob:
            return None
        return cls.from_dict(blob)


__all__ = [
    "CAPTION_KEYS",
    "DIA_KEYS",
    "INDEX_SCHEMA",
    "SessionIndex",
    "SPEAKER_KEYS",
    "TEXT_KEYS",
    "TIME_KEYS",
    "date_label",
    "messages_for_add",
    "normalize_turn",
    "parse_turn_list",
    "render_header",
    "render_turn_line",
    "session_label",
    "transcript_for_add",
    "turns_from",
]
