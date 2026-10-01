"""Mem0-shaped compatibility surface over the verbatim v5 facade.

SPEC_V5 §18.2 / V5-18.06..13 — translate, never re-implement. Every call
delegates to a bound :class:`verbatim.memory.facade.Memory` (imported lazily
so this module loads while the facade worker is mid-flight). Same store,
same authority, same closure: the shim owns no parallel identity, grant, or
storage path (V5-18.06).

Pinned surface (V5-18.07)
-------------------------
``mem0ai/mem0`` OSS ``mem0.memory.main.Memory`` plus the method subset of
``mem0.client.main.MemoryClient``, as documented on the project README /
``LLM.md`` / ``docs.mem0.ai`` (OSS 1.x line, output format ``v1.1`` —
dict ``{"results": [...]}`` responses):

    Memory(config=MemoryConfig())
    add(messages, *, user_id, agent_id, run_id, metadata, infer,
        memory_type, prompt)
    get(memory_id)
    get_all(*, user_id, agent_id, run_id, filters, limit=100)
    search(query, *, user_id, agent_id, run_id, limit=100, filters,
           threshold)
    update(memory_id, data)
    delete(memory_id)
    delete_all(user_id, agent_id, run_id)
    history(memory_id)
    MemoryClient(api_key, host, org_id, project_id) — platform client;
    here bound to a local store path; platform-only arguments raise.

Documented migration differences (V5-18.08 — visible, not hidden)
-----------------------------------------------------------------
* **Extraction.** ``infer=True`` in Mem0 runs an LLM that extracts facts and
  decides ADD/UPDATE/DELETE/NOOP itself. verbatim never fabricates facts:
  ``infer=True`` only *requests* the caller-permitted deterministic
  interpretation pipeline (claims stay revision-pinned views over retained
  bytes); ``infer=False`` stores source-only and is still searchable
  (V5-07.04). One verbatim source is retained per message — a list of
  messages yields one result item per message, never LLM-merged splits.
  Model-decided update/delete events do not exist; advisory
  ``verbatim_possible_updates`` are surfaced instead (V5-30.18/19).
* **Scoping.** ``user_id``/``agent_id``/``run_id`` select *namespace
  aliases* bound under the local owner (V5-18.09). The facade constructor
  takes a single ``user_id`` alias label; a bare user id passes through
  unchanged (so compat and native callers naming the same user share one
  namespace) while multi-id tuples compose one deterministic escaped
  label. This is strictly narrower than Mem0's metadata-AND semantics —
  a memory added under ``(user_id=u, agent_id=a)`` is not returned by
  ``search(user_id=u)``. No call mints a principal or broadens a search;
  run-scoped TTL needs a facade-side run-kind binding the contract
  signature does not expose, so ``run_id`` currently scopes like any
  other composite label (no TTL).
* **Honest states.** verbatim searches may be ``pending``/``partial``/
  ``blocked``/``unavailable``. Those states surface in every response as
  ``status`` + ``verbatim_status`` (+ ``verbatim_warnings``/
  ``verbatim_readiness``/``verbatim_coverage``); they are never collapsed
  into an empty "successful" list (V5-18.10).
* **Updates are retained sources with CAS.** verbatim replacements mint
  a *new* source and mark the predecessor ``superseded`` with a
  ``superseded_by = '<new_sid>:<rev>'`` pointer (V5-14.10) — a Mem0
  logical memory is therefore the supersession *chain*, and its live
  identity is the chain head. ``update``/``get``/``delete`` on a bare id
  follow that pointer (bounded, cycle-safe) so callers holding a
  pre-update id reach the current memory; ``verbatim_current_id``
  surfaces the live source id. Literal ``mref1.…`` addresses never
  follow — an explicit version fence pins exactly that version, so stale
  writes conflict with ``MemoryConflictError`` rather than silently
  overwriting the head (V5-18.11).
* **Deletion is suppression + closure,** not row erasure. ``delete``
  goes through ``forget(ref)`` with the CAS guard intact on the resolved
  live head; superseded predecessors remain as retained audit history
  (the facade cannot CAS-fence a composite mutation head — reported via
  ``verbatim_chain_links``/``verbatim_audit_remnants``, never hidden).
  ``delete_all`` requires an explicit ``confirm=True`` (defaults safe,
  V5-18.13) and then runs one of two honest flows: with the verbatim
  ``query=`` extension it drives the facade's real two-phase
  query-forget — ``forget(query=…)`` mints a bounded-selection
  confirmation token and ``forget(confirmation=token)`` executes exactly
  that selection (the token stands alone; V5-15.04). Without ``query``
  (Mem0 match-all semantics) the shim enumerates the bound namespace via
  bounded ``search`` and runs ordinary per-ref closure on every live
  source-backed hit; if enumeration is ``pending``/``blocked``/
  ``unavailable`` the call performs nothing and reports the real status
  rather than deleting a partial selection under an "all deleted" claim,
  and a ``partial``/``degraded`` enumeration reports incomplete coverage
  instead of claiming success.
* **Unsupported options raise** ``UnsupportedOptionError`` naming every
  rejected key (V5-18.10): LLM/graph knobs (``prompt``, ``llm``,
  ``memory_type``, ``custom_instructions``…), provider configs, platform
  tenancy (``api_key``, ``org_id``, ``project_id``), ``version="v1"`` /
  ``output_format="v1.0"`` bare-list output (it would discard status
  fields), OR/NOT filter operators, multi-namespace ``in`` scope filters,
  and per-call retention knobs (``expiration_date``).
* ``verbatim_*`` keys mark verbatim extensions. Mem0-named fields are
  populated only from real facade data; fields verbatim cannot produce
  honestly are omitted rather than fabricated.

Enumeration note: Mem0 ``get_all`` has no verbatim counterpart verb; it is
implemented as a bounded ``search(_ENUMERATION_QUERY)`` against the bound
namespace (see ``_ENUMERATION_QUERY``) — the landed facade honors the
match-all token, and pending/partial/blocked states surface verbatim
instead of a fake empty list. Match-all ``delete_all`` reuses the same
bounded enumeration for its selection rather than the query-forget path:
a previewed selection pins each ref's *current* fence, and superseded
links' composite heads can never satisfy it — the confirm would abort the
whole transaction (V5-15.04).
"""

from __future__ import annotations

import inspect as _inspect
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from verbatim.core.types import ErrorCode, VerbatimError

MEM0_COMPAT_SCHEMA = "verbatim.compat.mem0/v1"
MEM0_SURFACE_PIN = (
    "mem0ai/mem0 OSS Memory/MemoryClient documented surface, 1.x line, "
    "output_format v1.1 (dict {'results': [...]})"
)

#: Query sent to ``facade.search`` when the Mem0 caller asks for
#: "everything in this scope" (get_all / match-all delete_all selection).
#: The facade honors the match-all token and enumerates the bound
#: namespace; if it ever rejects it the typed error propagates — we never
#: fake an empty listing. It is NOT sent to ``facade.forget(query=…)``:
#: the previewed selection pins each ref's current fence and a superseded
#: link's composite head can never satisfy it — the confirm would abort
#: the whole transaction (V5-15.04).
_ENUMERATION_QUERY = "*"

_MAX_BOUND_FACADES = 64

#: Hit/source lifecycles that are no longer "the current memory" — Mem0
#: results enumerate live memories only; these stay visible via verbatim_*
#: fields, history, and get-by-ref audit rather than masquerading as live.
_NONCURRENT_LIFECYCLES = frozenset(
    {"superseded", "corrected", "retracted", "expired", "erased", "purged"}
)

#: Bound on supersession hops when resolving a logical memory_id to its
#: live chain head — cycles/pathological chains fail closed.
_MAX_CHAIN_HOPS = 32

#: VerbatimError codes that mean "your CAS fence lost" when raised through
#: update/delete.
_CONFLICT_CODES = frozenset(
    {
        ErrorCode.OPERATION_CONFLICT,
        ErrorCode.STALE_PROPOSAL,
        ErrorCode.STALE_EPOCH,
        ErrorCode.STALE_DEPENDENCY,
        ErrorCode.STORE_CONFLICT,
    }
)
_NOT_FOUND_CODES = frozenset(
    {ErrorCode.NOT_FOUND_OR_FORBIDDEN, ErrorCode.NOT_FOUND_OR_UNAUTHORIZED}
)


# ------------------------------------------------------------------ errors


class Mem0CompatError(VerbatimError):
    """Base class for typed compat-layer failures."""

    _CODE = ErrorCode.VALIDATION

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ErrorCode] = None,
        retryable: bool = False,
        detail_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            code or self._CODE, message, retryable=retryable, detail_id=detail_id
        )


class UnsupportedOptionError(Mem0CompatError):
    """A Mem0 option verbatim cannot honor — raised, never dropped."""

    _CODE = ErrorCode.VALIDATION


class ConfirmationRequiredError(Mem0CompatError):
    """A bulk/irreversible call was attempted without explicit confirmation."""

    _CODE = ErrorCode.VALIDATION


class NamespaceBindingError(Mem0CompatError):
    """Scope ids could not be bound to an accessible namespace alias."""

    _CODE = ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


class MemoryNotFoundError(Mem0CompatError):
    """memory_id resolved to nothing deliverable (or is not authorized).

    Denial indistinguishability: missing and forbidden look identical.
    """

    _CODE = ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


class MemoryConflictError(Mem0CompatError):
    """A version-fenced update/delete lost its CAS race."""

    _CODE = ErrorCode.OPERATION_CONFLICT


class FacadeUnavailableError(Mem0CompatError):
    """The v5 facade is not importable/constructible in this build."""

    _CODE = ErrorCode.CAPABILITY_UNAVAILABLE


# ------------------------------------------------------------ facade bind


def _load_facade_class():
    """Resolve the facade ``Memory`` class lazily (mid-flight tolerant)."""
    errors: List[str] = []
    try:
        from verbatim.memory.facade import Memory as cls  # noqa: WPS433

        return cls
    except Exception as exc:  # module absent or itself mid-flight
        errors.append(f"verbatim.memory.facade: {exc!r}")
    try:
        import verbatim

        cls = getattr(verbatim, "Memory")  # lazy __getattr__ export
        return cls
    except Exception as exc:
        errors.append(f"verbatim.Memory: {exc!r}")
    raise FacadeUnavailableError(
        "v5 facade Memory is not available in this build "
        f"({'; '.join(errors)})"
    )


def facade_available() -> bool:
    """True when a facade ``Memory`` class can be resolved."""
    try:
        _load_facade_class()
    except FacadeUnavailableError:
        return False
    return True


_REF_PREFIX = "mref1"


def _load_memory_ref():
    """Lazy import of the frozen ``MemoryRef`` type (types.py may lag)."""
    try:
        from verbatim.memory.types import MemoryRef  # noqa: WPS433

        return MemoryRef
    except Exception:
        return None


def _decode_ref_component(value: str) -> str:
    return value.replace("%2E", ".").replace("%25", "%")


def _parse_ref_fields(
    text: Any,
) -> Optional[Tuple[str, str, str, int, int]]:
    """``mref1.<store>.<ns>.<src>.<rev>.<ctl>`` → tuple, or None.

    Prefers the frozen ``MemoryRef.parse``; falls back to the documented
    wire format so the shim still works if ``verbatim.memory.types`` cannot
    be imported yet.
    """
    if not isinstance(text, str) or not text.startswith(_REF_PREFIX + "."):
        return None
    cls = _load_memory_ref()
    if cls is not None:
        try:
            parsed = cls.parse(text)
            return (
                parsed.store_tag,
                parsed.namespace,
                parsed.source_id,
                parsed.expected_revision,
                parsed.control_version,
            )
        except Exception:
            pass
    parts = text.split(".")
    if len(parts) != 6:
        return None
    try:
        return (
            _decode_ref_component(parts[1]),
            _decode_ref_component(parts[2]),
            _decode_ref_component(parts[3]),
            int(parts[4]),
            int(parts[5]),
        )
    except (ValueError, IndexError):
        return None


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {}


def _is_gone(lifecycle: Dict[str, Any]) -> bool:
    """Suppressed/erased lifecycle → Mem0 deleted-is-absent parity."""
    if lifecycle.get("suppressed"):
        return True
    return str(lifecycle.get("disposition") or "") in ("erased", "purged")


def _int_or_none(value: Any) -> Optional[int]:
    """Coerce a maybe-stringified revision/control number to int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _iso_time(value: Any) -> Optional[str]:
    """Normalize a verbatim timestamp for the Mem0 ``*_at`` fields.

    Store-side fields are epoch microseconds (``*_us`` ints); facade-side
    fields (``recorded_time``/``valid_time``/``updated_at``) are already
    ISO strings — passed through. Anything else yields ``None`` rather
    than a fabricated date.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    import datetime

    try:
        dt = datetime.datetime.fromtimestamp(
            value / 1_000_000, tz=datetime.timezone.utc
        )
    except (OverflowError, OSError, ValueError):
        return None
    return dt.isoformat().replace("+00:00", "Z")


# ------------------------------------------------------- namespace aliasing
#
# user_id/agent_id/run_id map ONLY to namespace aliases (V5-18.09). The
# facade binds one ``user_id`` alias label per instance (an opaque,
# label-free ``ns_…`` namespace is provisioned per label by
# ``verbatim.memory.aliases``); a scope tuple with agent/run components is
# composed into a single deterministic label — strictly narrower than
# Mem0's metadata-AND semantics and never a new principal. Alias labels
# are free-form printable strings (facade ``validate_label``): non-empty,
# ≤256 bytes, no control characters. ``:`` inside a component is escaped
# so composite labels stay unambiguous.

_MAX_LABEL_BYTES = 256
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_TAG_NAMES = {"u": "user_id", "a": "agent_id", "r": "run_id"}


def _enc_component(value: str) -> str:
    return value.replace("%", "%25").replace(":", "%3A")


def _dec_component(value: str) -> str:
    return value.replace("%3A", ":").replace("%25", "%")


def _compose_alias(
    user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
) -> Optional[str]:
    provided = [
        (tag, value)
        for tag, value in (("u", user_id), ("a", agent_id), ("r", run_id))
        if value is not None
    ]
    if not provided:
        return None
    if len(provided) == 1 and provided[0][0] == "u":
        # Bare user scope keeps the facade's own user_id label convention,
        # so compat and native facade callers naming the same user_id share
        # one namespace.
        return provided[0][1]
    return ":".join(f"{tag}:{_enc_component(value)}" for tag, value in provided)


def _validate_alias_label(alias: Optional[str]) -> Optional[str]:
    """Mirror facade ``aliases.validate_label`` — fail before binding."""
    if alias is None:
        return None
    text = alias.strip()
    if not text:
        raise Mem0CompatError("scope alias resolves to an empty label")
    if len(text.encode("utf-8", "strict")) > _MAX_LABEL_BYTES:
        raise Mem0CompatError(
            f"scope alias exceeds {_MAX_LABEL_BYTES} bytes"
        )
    if _CONTROL_CHARS.search(text):
        raise Mem0CompatError("scope alias contains control characters")
    return text


def _ids_from_alias(alias: Optional[str]) -> Dict[str, str]:
    """Inverse of the tagged composite form only — opaque namespaces and
    bare labels decode to nothing rather than a guessed id."""
    if not alias:
        return {}
    parts = alias.split(":")
    if (
        len(parts) >= 2
        and len(parts) % 2 == 0
        and all(parts[i] in _TAG_NAMES for i in range(0, len(parts), 2))
    ):
        return {
            _TAG_NAMES[parts[i]]: _dec_component(parts[i + 1])
            for i in range(0, len(parts), 2)
        }
    return {}


# --------------------------------------------------------------- validation


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Mem0CompatError(f"{field} must be a non-empty string")
    return value


def _require_id_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Mem0CompatError(f"{field} must be a non-empty string")
    return value


def _reject_unknown(kwargs: Dict[str, Any], where: str) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise UnsupportedOptionError(f"unsupported {where} option(s): {names}")


def _facade_call(where: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a bound facade method with compat error translation.

    ``VerbatimError`` passes through ``_translate_error`` (CAS → conflict,
    denial → not-found, rest preserved). A mid-flight facade raising
    ``ImportError``/``NotImplementedError`` from inside a call means the
    capability is not provisioned in this build — surfaced as
    ``FacadeUnavailableError`` (CAPABILITY_UNAVAILABLE), never raw.
    """
    try:
        return fn(*args, **kwargs)
    except VerbatimError as exc:
        raise _translate_error(exc, where) from exc
    except (ImportError, NotImplementedError) as exc:
        raise FacadeUnavailableError(
            f"{where}: facade path is mid-flight/unprovisioned — {exc!r}"
        ) from exc


def _reject_unsupported(
    provided: Dict[str, Any], where: str, *, allow_false_flags: bool = True
) -> None:
    """Raise for Mem0 knobs verbatim does not implement.

    ``allow_false_flags`` lets callers pass ``flag=False`` (requesting that
    an absent feature stay absent) without an error; any truthy/non-None
    non-bool value always fails loudly.
    """
    rejected = []
    for name, value in provided.items():
        if value is None:
            continue
        if allow_false_flags and isinstance(value, bool) and value is False:
            continue
        rejected.append(name)
    if rejected:
        raise UnsupportedOptionError(
            f"unsupported {where} option(s): {', '.join(sorted(rejected))}"
        )


def _check_version(version: Any, where: str) -> None:
    """Mem0 ``version``/``output_format``: only dict-shape outputs survive.

    v1.0 returns bare lists that would discard verbatim status fields —
    rejected rather than silently downgraded (V5-18.10).
    """
    if version is None:
        return
    if str(version) in ("v2", "v1.1", "1.1"):
        return
    raise UnsupportedOptionError(
        f"unsupported {where} version/output_format {version!r}: only the "
        "dict {'results': [...]} shape is emitted (bare v1.0 lists would "
        "drop verbatim status fields)"
    )


def _normalize_content(content: Any) -> str:
    """One message's content → text. Multimodal parts are unsupported."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            else:
                raise UnsupportedOptionError(
                    "non-text message content parts are unsupported"
                )
        return "\n".join(texts)
    raise Mem0CompatError("message content must be a string or text parts")


def _normalize_messages(messages: Any, text: Optional[str]) -> List[Dict[str, str]]:
    """Mem0 ``messages`` (str | message dict | list) → validated list."""
    if text is not None:
        if messages is not None:
            raise Mem0CompatError("pass either messages or text, not both")
        messages = text
    if messages is None:
        raise Mem0CompatError("add() requires messages or text")
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise Mem0CompatError(
            "messages must be a string or a non-empty list of "
            "{'role','content'} dicts"
        )
    out: List[Dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise Mem0CompatError(f"messages[{i}] is not a dict")
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            raise Mem0CompatError(f"messages[{i}] needs a string 'role'")
        out.append({"role": role, "content": _normalize_content(msg.get("content"))})
    return out


def _merge_limit(limit: Any, top_k: Any, default: int) -> int:
    if limit is not None and top_k is not None and limit != top_k:
        raise Mem0CompatError(
            f"conflicting limit={limit!r} and top_k={top_k!r}"
        )
    value = limit if limit is not None else top_k
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Mem0CompatError(f"limit must be a positive integer, got {value!r}")
    return value


_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


#: Facade ``search(filters=)`` allowlist — membership keys take a string
#: or list of strings; the time bounds take RFC3339 strings.
_FACADE_MEMBERSHIP_KEYS = frozenset(
    {"type", "kind", "lifecycle", "source_id"}
)
_FACADE_TIME_KEYS = frozenset({"created_after", "created_before"})

#: Mem0 filter keys that map onto facade time bounds (recorded_time).
_TIME_KEY_MAP = {
    "created_at": True,
    "recorded_time": True,
}


def _translate_filters(
    filters: Any, where: str
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str], List[str]]:
    """Mem0 filter dict → (facade filter dict, scope ids, notes).

    The facade's filter language is a bounded conjunction over a fixed
    allowlist — ``type``/``kind``/``lifecycle``/``source_id`` membership
    and ``created_after``/``created_before`` RFC3339 bounds (V5-06.13).
    ``eq``/``in`` collapse to membership values; ``gt``/``gte``/``lt``/
    ``lte`` on ``created_at`` map to the strict facade bounds (the
    boundary is exclusive — reported as a note). OR/NOT combinators,
    negation/substring operators, and any key the facade cannot evaluate
    raise ``UnsupportedOptionError`` — silently dropping a caller's
    filter would *broaden* the result set. ``user_id``/``agent_id``/
    ``run_id`` inside filters are extracted as scope ids (single values
    only — ``in`` over scope ids would be a multi-namespace search, i.e.
    a broadening).
    """
    scope: Dict[str, str] = {}
    notes: List[str] = []
    if filters is None:
        return None, scope, notes
    if not isinstance(filters, dict):
        raise Mem0CompatError(f"{where} filters must be a dict")

    clauses: Dict[str, Any] = {}

    def add_scope(key: str, value: Any) -> None:
        if isinstance(value, dict):
            ops = set(value)
            if ops == {"eq"}:
                value = value["eq"]
            elif "in" in value and len(ops) == 1:
                raise UnsupportedOptionError(
                    f"{where}: '{key}' membership filters would span "
                    "multiple namespaces — unsupported (V5-18.09)"
                )
            else:
                raise UnsupportedOptionError(
                    f"{where}: unsupported operators on scope key {key!r}: "
                    f"{sorted(ops)}"
                )
        if not isinstance(value, str) or not value:
            raise Mem0CompatError(
                f"{where}: scope key {key!r} needs a non-empty string"
            )
        if key in scope and scope[key] != value:
            raise Mem0CompatError(f"{where}: conflicting {key} values in filters")
        scope[key] = value

    def _membership(key: str, value: Any) -> Any:
        """eq/in → scalar-or-list of strings for the facade allowlist."""
        if isinstance(value, dict):
            ops = set(value)
            if ops == {"eq"}:
                value = value["eq"]
            elif ops == {"in"} and isinstance(value["in"], (list, tuple)):
                value = list(value["in"])
            else:
                raise UnsupportedOptionError(
                    f"{where}: unsupported filter operator(s) on {key!r}: "
                    f"{sorted(ops)} — the facade supports membership only"
                )
        vals = value if isinstance(value, (list, tuple)) else [value]
        if not vals or not all(isinstance(x, str) and x for x in vals):
            raise Mem0CompatError(
                f"{where}: filter {key!r} needs a non-empty string or "
                "list of strings"
            )
        return list(vals) if isinstance(value, (list, tuple)) else vals[0]

    def add_time_bound(key: str, value: Any) -> None:
        if not isinstance(value, dict) or not value:
            raise UnsupportedOptionError(
                f"{where}: filter {key!r} needs an operator dict "
                "(gt/gte/lt/lte) — equality on timestamps is not a "
                "facade filter"
            )
        ops = set(value)
        if not ops <= {"gt", "gte", "lt", "lte"}:
            raise UnsupportedOptionError(
                f"{where}: unsupported filter operator(s) on {key!r}: "
                f"{sorted(ops)}"
            )
        for op in ("gt", "gte"):
            if op in value:
                if not isinstance(value[op], str) or not value[op]:
                    raise Mem0CompatError(
                        f"{where}: {key}.{op} needs an RFC3339 string"
                    )
                if "created_after" in clauses and clauses["created_after"] != value[op]:
                    raise Mem0CompatError(
                        f"{where}: conflicting {key} lower bounds"
                    )
                clauses["created_after"] = value[op]
                if op == "gte":
                    notes.append(
                        "verbatim: 'gte' on "
                        f"{key} maps to a strict created_after — items "
                        "recorded exactly at the bound are excluded"
                    )
        for op in ("lt", "lte"):
            if op in value:
                if not isinstance(value[op], str) or not value[op]:
                    raise Mem0CompatError(
                        f"{where}: {key}.{op} needs an RFC3339 string"
                    )
                if (
                    "created_before" in clauses
                    and clauses["created_before"] != value[op]
                ):
                    raise Mem0CompatError(
                        f"{where}: conflicting {key} upper bounds"
                    )
                clauses["created_before"] = value[op]
                if op == "lte":
                    notes.append(
                        "verbatim: 'lte' on "
                        f"{key} maps to a strict created_before — items "
                        "recorded exactly at the bound are excluded"
                    )

    def add_clause(key: str, value: Any) -> None:
        low = key.lower()
        if low in _FACADE_MEMBERSHIP_KEYS:
            clauses[low] = _membership(key, value)
            return
        if low in _FACADE_TIME_KEYS:
            # Direct facade bound names — scalar RFC3339 or {"eq": …}.
            if isinstance(value, dict):
                ops = set(value)
                if ops == {"eq"}:
                    value = value["eq"]
                else:
                    raise UnsupportedOptionError(
                        f"{where}: filter {key!r} takes an RFC3339 "
                        f"string, got operators {sorted(ops)}"
                    )
            if not isinstance(value, str) or not value:
                raise Mem0CompatError(
                    f"{where}: filter {key!r} needs an RFC3339 string"
                )
            clauses[low] = value
            return
        if low in _TIME_KEY_MAP:
            add_time_bound(key, value)
            return
        if low == "updated_at":
            raise UnsupportedOptionError(
                f"{where}: 'updated_at' filtering is not expressible — "
                "the facade bounds recorded_time only (valid_time "
                "filters would be a semantic mismatch, not an "
                "approximation)"
            )
        raise UnsupportedOptionError(
            f"{where}: filter key {key!r} is not in the facade's "
            f"allowlist {sorted(_FACADE_MEMBERSHIP_KEYS | _FACADE_TIME_KEYS)} "
            "— dropping it would silently broaden the search"
        )

    def walk(node: Dict[str, Any]) -> None:
        for key, value in node.items():
            low = key.lower()
            if low in ("and",):
                if not isinstance(value, (list, tuple)):
                    raise Mem0CompatError(f"{where}: 'AND' expects a list")
                for sub in value:
                    if not isinstance(sub, dict):
                        raise Mem0CompatError(f"{where}: AND entries must be dicts")
                    walk(sub)
            elif low in ("or", "not"):
                raise UnsupportedOptionError(
                    f"{where}: {key!r} filters exceed verbatim's bounded "
                    "conjunction (V5-06.13)"
                )
            elif key in _SCOPE_KEYS:
                add_scope(key, value)
            else:
                add_clause(key, value)

    walk(filters)
    return (clauses or None), scope, notes


# ------------------------------------------------------------------- shim


class Memory:
    """Mem0 OSS-shaped local memory over the verbatim v5 facade.

    One facade ``Memory`` is bound lazily per scope tuple
    ``(user_id, agent_id, run_id)`` — all of them on *this* instance's
    store path, so compat callers share one store and one authority.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Any = None,
        profile: str = "local_memory",
        worker: str = "managed",
        encoder: str = "hashing",
        ready_timeout_ms: int = 200,
        create: bool = True,
        host: Any = None,
        **unsupported: Any,
    ) -> None:
        if config is not None:
            cfg = _as_dict(config) if not isinstance(config, dict) else dict(config)
            if not cfg and hasattr(config, "dict"):
                cfg = dict(config.dict())
            if not cfg and hasattr(config, "model_dump"):
                cfg = dict(config.model_dump())
            allowed = {
                "path",
                "user_id",
                "agent_id",
                "run_id",
                "profile",
                "worker",
                "encoder",
                "ready_timeout_ms",
                "create",
                "host",
            }
            foreign = sorted(k for k in cfg if k not in allowed)
            if foreign:
                raise UnsupportedOptionError(
                    "unsupported Mem0 config key(s): "
                    + ", ".join(foreign)
                    + " (verbatim takes path/user_id/profile/worker/encoder/"
                    "ready_timeout_ms/create/host; provider/LLM/vector-store "
                    "configs do not translate)"
                )
            path = cfg.get("path", path)
            user_id = cfg.get("user_id", user_id)
            agent_id = cfg.get("agent_id", agent_id)
            run_id = cfg.get("run_id", run_id)
            profile = cfg.get("profile", profile)
            worker = cfg.get("worker", worker)
            encoder = cfg.get("encoder", encoder)
            ready_timeout_ms = cfg.get("ready_timeout_ms", ready_timeout_ms)
            create = cfg.get("create", create)
            host = cfg.get("host", host)
        _reject_unknown(unsupported, "constructor")

        self._path = path
        self._ctor_kwargs = {
            "profile": profile,
            "worker": worker,
            "encoder": encoder,
            "ready_timeout_ms": ready_timeout_ms,
            "create": create,
            "host": host,
        }
        self._default_scope = (user_id, agent_id, run_id)
        self._facades: "OrderedDict[Tuple[Optional[str], ...], Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------ plumbing

    @classmethod
    def from_config(cls, config: Any) -> "Memory":
        """Mem0 ``Memory.from_config`` parity (verbatim-shaped keys only)."""
        return cls(config=config)

    def _scope(
        self,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        extra: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Merge call ids + filter-carried ids over the instance default."""
        extra = extra or {}
        for key in _SCOPE_KEYS:
            if key in extra and extra[key] is not None:
                call_value = {
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                }[key]
                if call_value is not None and call_value != extra[key]:
                    raise Mem0CompatError(
                        f"conflicting {key}: argument {call_value!r} vs "
                        f"filters {extra[key]!r}"
                    )
        user_id = user_id if user_id is not None else extra.get("user_id")
        agent_id = agent_id if agent_id is not None else extra.get("agent_id")
        run_id = run_id if run_id is not None else extra.get("run_id")
        default = self._default_scope
        return (
            user_id if user_id is not None else default[0],
            agent_id if agent_id is not None else default[1],
            run_id if run_id is not None else default[2],
        )

    def _facade(self, scope: Tuple[Optional[str], Optional[str], Optional[str]]):
        """One bound facade per scope tuple, all on the same store path."""
        if self._closed:
            raise Mem0CompatError("Memory is closed", code=ErrorCode.INVALID_TRANSITION)
        with self._lock:
            cached = self._facades.get(scope)
            if cached is not None:
                self._facades.move_to_end(scope)
                return cached
            facade_cls = _load_facade_class()
            kwargs = self._facade_kwargs(facade_cls, scope)
            try:
                instance = (
                    facade_cls(self._path, **kwargs)
                    if self._path is not None
                    else facade_cls(**kwargs)
                )
            except VerbatimError:
                raise
            except TypeError as exc:
                raise NamespaceBindingError(
                    f"facade rejected scope binding {kwargs!r}: {exc}"
                ) from exc
            except Exception as exc:
                raise FacadeUnavailableError(
                    f"facade construction failed: {exc!r}"
                ) from exc
            self._facades[scope] = instance
            if len(self._facades) > _MAX_BOUND_FACADES:
                _, evicted = self._facades.popitem(last=False)
                try:
                    evicted.close()
                except Exception:
                    pass
            return instance

    def _facade_kwargs(self, facade_cls: Any, scope) -> Dict[str, Any]:
        """Bind scope ids to facade constructor args (mid-flight tolerant).

        Preferred: structured ``user_id``/``agent_id``/``run_id`` params if
        the facade signature grew them. Contract baseline: ``user_id`` only
        → pass the composed namespace alias. If the facade knows no alias
        parameter at all, scoped calls fail loudly instead of silently
        landing in the default namespace (never broaden, V5-18.09).
        """
        user_id, agent_id, run_id = scope
        try:
            params = _inspect.signature(facade_cls).parameters
            accepts_kwargs = any(
                p.kind is _inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            params, accepts_kwargs = {}, True

        kwargs: Dict[str, Any] = {}
        wanted = dict(self._ctor_kwargs)
        wanted = {k: v for k, v in wanted.items() if v is not None or k != "host"}
        for key, value in wanted.items():
            if accepts_kwargs or key in params:
                kwargs[key] = value
        if {"user_id", "agent_id", "run_id"} <= set(params):
            kwargs["user_id"] = user_id
            kwargs["agent_id"] = agent_id
            kwargs["run_id"] = run_id
        elif "user_id" in params or accepts_kwargs:
            # Contract-baseline facade: a single user_id alias label carries
            # the whole scope tuple.
            alias = _validate_alias_label(
                _compose_alias(user_id, agent_id, run_id)
            )
            if alias is not None:
                kwargs["user_id"] = alias
        elif scope != (None, None, None):
            raise FacadeUnavailableError(
                "facade exposes no user_id/alias binding — cannot honor "
                "scoped Mem0 calls without risking scope broadening"
            )
        return kwargs

    # ------------------------------------------------------------ Mem0 API

    def add(
        self,
        messages: Any = None,
        *,
        text: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm: Any = None,
        timestamp: Any = None,
        expiration_date: Any = None,
        filters: Optional[Dict[str, Any]] = None,
        categories: Any = None,
        includes: Any = None,
        excludes: Any = None,
        custom_instructions: Any = None,
        app_id: Any = None,
        output_format: Any = None,
        async_mode: Any = None,
        idempotency_key: Optional[str] = None,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``add`` → one verbatim source per message (event ``ADD``)."""
        _reject_unknown(unsupported, "add")
        _check_version(output_format, "add")
        _reject_unsupported(
            {
                "memory_type": memory_type,
                "prompt": prompt,
                "llm": llm,
                "timestamp": timestamp,
                "expiration_date": expiration_date,
                "categories": categories,
                "includes": includes,
                "excludes": excludes,
                "custom_instructions": custom_instructions,
                "app_id": app_id,
                "async_mode": async_mode,
                "filters": filters,
            },
            "add",
        )
        msgs = _normalize_messages(messages, text)
        if metadata is not None and not isinstance(metadata, dict):
            raise Mem0CompatError("metadata must be a dict")
        scope = self._scope(user_id, agent_id, run_id, None)
        bound = self._facade(scope)

        results: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for msg in msgs:
            meta = dict(metadata or {})
            meta["verbatim_role"] = msg["role"]
            kwargs: Dict[str, Any] = {"infer": bool(infer), "metadata": meta}
            if idempotency_key is not None:
                kwargs["idempotency_key"] = idempotency_key
            res = _facade_call("add", bound.add, msg["content"], **kwargs)
            rd = _as_dict(res)
            warnings.extend(rd.get("warnings") or [])
            item = {
                "id": rd.get("memory_id") or rd.get("ref"),
                "memory": msg["content"],
                "event": "ADD",
                "metadata": meta,
                "verbatim_ref": rd.get("ref"),
                "verbatim_revision": rd.get("source_revision"),
                "verbatim_receipt_id": rd.get("receipt_id"),
                "verbatim_acceptance": rd.get("acceptance"),
                "verbatim_readiness": rd.get("readiness"),
                "verbatim_replayed": rd.get("replayed", False),
                "verbatim_inference": rd.get("inference"),
                "verbatim_possible_updates": rd.get("possible_updates") or [],
            }
            item.update({k: v for k, v in zip(_SCOPE_KEYS, scope) if v is not None})
            results.append(item)
        acceptances = {
            r.get("verbatim_acceptance") or "accepted" for r in results
        }
        status = (
            acceptances.pop() if len(acceptances) == 1 else "mixed"
        )
        return {
            "results": results,
            "status": status,
            "verbatim_status": status,
            "verbatim_warnings": warnings,
            "verbatim_schema": MEM0_COMPAT_SCHEMA,
        }

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        version: Any = None,
        rerank: Any = None,
        keyword_search: Any = None,
        filter_memories: Any = None,
        fields: Any = None,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``search`` → facade ``search`` with status preserved."""
        _reject_unknown(unsupported, "search")
        _check_version(version, "search")
        _reject_unsupported(
            {
                "rerank": rerank,
                "keyword_search": keyword_search,
                "filter_memories": filter_memories,
                "fields": fields,
            },
            "search",
        )
        _require_text(query, "query")
        limit_value = _merge_limit(limit, top_k, default=100)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                raise Mem0CompatError(
                    f"threshold must be a number in [0,1], got {threshold!r}"
                ) from None
            if not 0.0 <= threshold <= 1.0:
                raise Mem0CompatError(
                    f"threshold must be in [0,1], got {threshold!r}"
                )
        clauses, filter_scope, notes = _translate_filters(filters, "search")
        scope = self._scope(user_id, agent_id, run_id, filter_scope)
        bound = self._facade(scope)

        call: Dict[str, Any] = {"limit": limit_value}
        if clauses is not None:
            call["filters"] = clauses
        res = _facade_call("search", bound.search, query, **call)
        return _mem0_search_result(
            res, scope, threshold=threshold, extra_warnings=notes
        )

    def get(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mem0 ``get`` → facade ``inspect``; None when not found/forbidden.

        Scope hint kwargs are a verbatim extension: a logical id resolves
        under the hinted/default scope, then scopes this instance already
        bound; ``mref1.…`` refs are authorized against whichever bound
        scope can see them (the ref itself carries the namespace).
        Suppressed/erased memories read as ``None`` — Mem0 parity treats a
        deletion tombstone as absent (``history`` still audits it).
        """
        _require_id_string(memory_id, "memory_id")
        scope = self._scope(user_id, agent_id, run_id, None)
        located = self._resolve(memory_id, scope, need_cas=False)
        if located is None:
            return None
        lifecycle = located[4].get("lifecycle") or {}
        disposition = str(lifecycle.get("disposition") or "")
        if lifecycle.get("suppressed") or disposition in ("erased", "purged"):
            return None
        return _mem0_record(memory_id, located[4], scope=located[0])

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        version: Any = None,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``get_all`` → bounded namespace enumeration via facade search.

        ``page``/``page_size`` (platform client parity) are a marked
        client-side slice over one bounded enumeration — verbatim has no
        offset cursor (V5-18.13).
        """
        _reject_unknown(unsupported, "get_all")
        _check_version(version, "get_all")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise Mem0CompatError(f"limit must be a positive integer, got {limit!r}")
        paging = page is not None or page_size is not None
        if paging:
            page = 1 if page is None else page
            page_size = limit if page_size is None else page_size
            for name, v in (("page", page), ("page_size", page_size)):
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise Mem0CompatError(f"{name} must be a positive integer")
            fetch = page * page_size
        else:
            fetch = limit
        clauses, filter_scope, notes = _translate_filters(filters, "get_all")
        scope = self._scope(user_id, agent_id, run_id, filter_scope)
        bound = self._facade(scope)

        call: Dict[str, Any] = {"limit": fetch}
        if clauses is not None:
            call["filters"] = clauses
        res = _facade_call("get_all", bound.search, _ENUMERATION_QUERY, **call)
        out = _mem0_search_result(
            res, scope, extra_warnings=notes, enumeration=True
        )
        if paging:
            start = (page - 1) * page_size
            out["results"] = out["results"][start : start + page_size]
            out["verbatim_paging"] = {
                "mode": "client_side_slice",
                "page": page,
                "page_size": page_size,
                "fetched": fetch,
            }
        return out

    def update(
        self,
        memory_id: str,
        data: Optional[str] = None,
        *,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``update`` → version-fenced ``add(replaces=ref)`` (V5-18.11).

        ``memory_id`` may be the logical id we returned (resolved to the
        authorized current ref via ``inspect``) or a verbatim ``mref1.…``
        string carrying an explicit expected revision — the CAS fence is
        retained either way; conflicts raise ``MemoryConflictError``.
        ``user_id``/``agent_id``/``run_id`` are verbatim scope-hint
        extensions for bare-id resolution.
        """
        _reject_unknown(unsupported, "update")
        _require_id_string(memory_id, "memory_id")
        if data is not None and text is not None:
            raise Mem0CompatError("pass either data or text, not both")
        new_text = _require_text(data if data is not None else text, "data/text")
        if metadata is not None and not isinstance(metadata, dict):
            raise Mem0CompatError("metadata must be a dict")
        scope = self._scope(user_id, agent_id, run_id, None)
        located = self._resolve(memory_id, scope)
        if located is None:
            raise MemoryNotFoundError(f"memory_id {memory_id!r} not found")
        found_scope, bound, ref, logical_id, _ = located

        kwargs: Dict[str, Any] = {"replaces": ref, "change": "supersede"}
        if metadata is not None:
            kwargs["metadata"] = metadata
        res = _facade_call("update", bound.add, new_text, **kwargs)
        rd = _as_dict(res)
        ref_fields = _parse_ref_fields(ref)
        return {
            "message": "Memory updated successfully!",
            "id": logical_id,
            "memory": new_text,
            "verbatim_ref": rd.get("ref"),
            "verbatim_current_id": rd.get("memory_id"),
            "verbatim_supersedes": (
                ref_fields[2] if ref_fields is not None else None
            ),
            "verbatim_receipt_id": rd.get("receipt_id"),
            "verbatim_revision": rd.get("source_revision"),
            "verbatim_acceptance": rd.get("acceptance"),
            "verbatim_readiness": rd.get("readiness"),
            "verbatim_possible_updates": rd.get("possible_updates") or [],
            "verbatim_status": rd.get("acceptance", "accepted"),
            **{k: v for k, v in zip(_SCOPE_KEYS, found_scope) if v is not None},
        }

    def delete(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``delete`` → facade ``forget`` with CAS on the resolved ref.

        A bare id resolves through the supersession chain to the live
        head — deleting a memory means closing what is *current*, not a
        stale predecessor the facade can no longer CAS-fence. Literal
        ``mref1.…`` addresses fence exactly that version (a superseded
        link's fence is a typed conflict, not a blind wipe).
        """
        _reject_unknown(unsupported, "delete")
        _require_id_string(memory_id, "memory_id")
        scope = self._scope(user_id, agent_id, run_id, None)
        located = self._resolve(memory_id, scope)
        if located is None:
            raise MemoryNotFoundError(f"memory_id {memory_id!r} not found")
        _found_scope, bound, ref, logical_id, d = located
        res = _facade_call("delete", bound.forget, ref)
        rd = _as_dict(res)
        ref_fields = _parse_ref_fields(ref)
        head_sid = ref_fields[2] if ref_fields is not None else None
        chain_links = []
        if head_sid and head_sid != logical_id and not logical_id.startswith(
            _REF_PREFIX + "."
        ):
            chain_links = [logical_id]
        if rd.get("mode", "operation") != "operation" or rd.get("mutated") is False:
            return {
                "message": "Memory deletion requires confirmation.",
                "id": logical_id,
                "verbatim_status": rd.get("mode", "preview"),
                "verbatim_mutated": bool(rd.get("mutated", False)),
                "verbatim_confirmation_token": rd.get("confirmation_token"),
                "verbatim_warnings": rd.get("warnings") or [],
            }
        warnings = list(rd.get("warnings") or [])
        out = {
            "message": "Memory deleted successfully!",
            "id": logical_id,
            "verbatim_status": "deleted",
            "verbatim_suppression_state": rd.get("suppression_state"),
            "verbatim_closure_state": rd.get("closure_state"),
            "verbatim_receipt_id": rd.get("receipt_id"),
        }
        if chain_links:
            out["verbatim_current_id"] = head_sid
            out["verbatim_chain_links"] = chain_links
            warnings.append(
                "verbatim: superseded predecessor link(s) "
                + ", ".join(chain_links)
                + " remain as retained audit history — closure covered "
                "the live chain head"
            )
        out["verbatim_warnings"] = warnings
        return out

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        *,
        confirm: bool = False,
        query: Optional[str] = None,
        limit: int = 256,
        **unsupported: Any,
    ) -> Dict[str, Any]:
        """Mem0 ``delete_all`` → bounded selection + explicit confirm.

        ``confirm`` defaults False and there is no Mem0-side equivalent —
        bulk deletion must be opt-in (V5-18.13). Without it the call raises
        ``ConfirmationRequiredError`` and touches nothing.

        ``query`` (verbatim extension): drive the facade's real two-phase
        query-forget — ``forget(query=…)`` previews a bounded term-matched
        selection and mints a short-lived confirmation token;
        ``forget(confirmation=token)`` then executes exactly that pinned
        selection (the token stands alone — V5-15.04). ``query="*"`` is
        term-free — it routes to the match-all path below, because a
        previewed selection containing superseded links can never
        CAS-confirm under the current facade.

        Without ``query`` this is Mem0's match-all. The shim enumerates
        the bound namespace through a bounded ``search`` and runs
        ordinary per-ref closure on every live source-backed hit. If
        enumeration is ``pending``/``blocked``/``unavailable`` nothing is
        deleted — a missing selection must not be reported as "all
        deleted". A ``partial``/``degraded`` enumeration still executes
        but reports honestly that coverage was incomplete.
        """
        _reject_unknown(unsupported, "delete_all")
        if not confirm:
            raise ConfirmationRequiredError(
                "delete_all requires confirm=True; the bounded selection "
                "and closure run only after that opt-in"
            )
        if query is not None:
            _require_text(query, "query")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise Mem0CompatError(
                f"limit must be a positive integer, got {limit!r}"
            )
        scope = self._scope(user_id, agent_id, run_id, None)
        if scope == (None, None, None):
            raise Mem0CompatError(
                "delete_all needs at least one scope id — a whole-store "
                "wipe is not a Mem0-scope operation"
            )
        bound = self._facade(scope)
        if query is not None and query.strip() != _ENUMERATION_QUERY:
            return self._delete_all_by_query(bound, query)
        return self._delete_all_enumerated(bound, scope, limit)

    def _delete_all_by_query(self, bound: Any, query: str) -> Dict[str, Any]:
        """Facade two-phase forget: preview → token → confirmed execute."""
        preview = _facade_call("delete_all", bound.forget, query=query)
        pd = _as_dict(preview)
        if pd.get("mutated") or pd.get("mode", "preview") == "operation":
            raise FacadeUnavailableError(
                "forget(query=…) mutated without a confirmation token — "
                "refusing to treat a preview call as executed"
            )
        token = pd.get("confirmation_token")
        if not token:
            raise FacadeUnavailableError(
                "delete_all preview minted no confirmation_token"
            )
        res = _facade_call("delete_all", bound.forget, confirmation=token)
        rd = _as_dict(res)
        if rd.get("mutated") is False or rd.get("mode", "operation") != "operation":
            raise FacadeUnavailableError(
                "confirmed forget reported a non-mutating result — "
                "selection may not have executed"
            )
        selection = rd.get("selection") or pd.get("selection") or []
        return {
            "message": "Memories deleted successfully!",
            "verbatim_status": "deleted",
            "verbatim_deleted": len(selection),
            "verbatim_selection": list(selection),
            "verbatim_suppression_state": rd.get("suppression_state"),
            "verbatim_closure_state": rd.get("closure_state"),
            "verbatim_receipt_id": rd.get("receipt_id"),
            "verbatim_warnings": (pd.get("warnings") or [])
            + (rd.get("warnings") or []),
        }

    def _delete_all_enumerated(
        self,
        bound: Any,
        scope: Tuple[Optional[str], Optional[str], Optional[str]],
        limit: int,
    ) -> Dict[str, Any]:
        """Match-all: bounded enumerate → per-ref closure (V5-15.04-safe).

        The facade's query-forget refuses term-free selections, so the
        bounded ``search(_ENUMERATION_QUERY)`` result *is* the explicit
        selection; each member then goes through ``forget(ref)`` — the
        ordinary suppression + closure path with its CAS guard.
        """
        res = _facade_call(
            "delete_all", bound.search, _ENUMERATION_QUERY, limit=limit
        )
        sd = _as_dict(res)
        status = str(sd.get("status", "pending"))
        # A support-verdict miss ("insufficient"/"no_answer") on a
        # match-all enumeration means coverage cannot be established —
        # items may exist that the verdict withheld. In enumeration
        # vocabulary that is degraded coverage, never a clean listing.
        enum_status = (
            "degraded"
            if status in ("insufficient", "no_answer", "no_candidates")
            else status
        )
        if status in ("pending", "blocked", "unavailable"):
            return {
                "message": (
                    "Deletion not performed — namespace enumeration is "
                    f"{status}; nothing was deleted"
                ),
                "verbatim_status": status,
                "verbatim_mutated": False,
                "verbatim_deleted": 0,
                "verbatim_readiness": sd.get("readiness") or {},
                "verbatim_coverage": sd.get("coverage") or {},
                "verbatim_warnings": sd.get("warnings") or [],
            }
        refs: List[str] = []
        unclosable: List[str] = []
        seen = set()
        for hit in sd.get("items") or []:
            hd = hit if isinstance(hit, dict) else _as_dict(hit)
            ref = hd.get("ref")
            # Only source-backed refs close through forget — claim/view
            # object refs are rejected by design (V5-06.11).
            if not (isinstance(ref, str) and ref) or ref in seen:
                continue
            seen.add(ref)
            lifecycle = str(hd.get("lifecycle") or "").strip().lower()
            if lifecycle in _NONCURRENT_LIFECYCLES:
                # A superseded/retracted link's composite mutation head
                # cannot be fenced by its pinned integer revision — the
                # facade CAS-guards it as always-stale. Report as
                # retained audit history, not a failure.
                unclosable.append(ref)
                continue
            refs.append(ref)
        deleted = 0
        errors: List[Dict[str, Any]] = []
        warnings: List[str] = list(sd.get("warnings") or [])
        for ref in refs:
            try:
                out = _as_dict(bound.forget(ref))
            except VerbatimError as exc:
                errors.append({"ref": ref, "error": exc.to_dict()})
                continue
            except (ImportError, NotImplementedError) as exc:
                raise FacadeUnavailableError(
                    "delete_all: facade forget() path is mid-flight/"
                    f"unprovisioned — {exc!r}"
                ) from exc
            warnings.extend(out.get("warnings") or [])
            if out.get("mutated"):
                deleted += 1
        truncated = (len(refs) + len(unclosable)) >= limit
        if truncated:
            warnings.append(
                "selection_truncated: enumeration hit the limit — the "
                "selection may not cover the whole namespace"
            )
        if status != "ready":
            warnings.append(
                f"enumeration_incomplete: search reported {status} — "
                "the selection may not cover every live memory; re-run "
                "delete_all to confirm closure"
            )
        if unclosable:
            warnings.append(
                f"audit_remnants: {len(unclosable)} superseded/retracted "
                "link(s) cannot be CAS-closed under this facade — they "
                "remain as superseded audit history, payloads retained "
                "(see verbatim_audit_remnants)"
            )
        if errors:
            warnings.append(
                f"partial_closure: {len(errors)} member(s) failed "
                "closure — see verbatim_errors"
            )
        honest = not errors and not truncated and status == "ready"
        result = {
            "message": (
                "Memories deleted successfully!"
                if honest
                else "Deletion incomplete — see verbatim_errors / "
                "verbatim_truncated / verbatim_audit_remnants"
            ),
            "verbatim_status": "deleted" if honest else "partial",
            "verbatim_enumeration_status": enum_status,
            "verbatim_deleted": deleted,
            "verbatim_selected": len(refs),
            "verbatim_errors": errors,
            "verbatim_audit_remnants": unclosable,
            "verbatim_warnings": warnings,
            "verbatim_truncated": truncated,
            "verbatim_readiness": sd.get("readiness") or {},
            "verbatim_coverage": sd.get("coverage") or {},
        }
        result.update(
            {k: v for k, v in zip(_SCOPE_KEYS, scope) if v is not None}
        )
        return result

    def history(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Mem0 ``history`` → revision trail across the supersession chain.

        verbatim mints a new source per replacement; the Mem0 history of a
        logical memory is every chain link's revision trail, oldest → live
        head. Inspection exposes a forward ``superseded_by`` pointer only,
        so the walk starts at the addressed link — history of a mid-chain
        id covers that link forward (no backward pointer exists to fake).
        """
        _require_id_string(memory_id, "memory_id")
        scope = self._scope(user_id, agent_id, run_id, None)
        located = self._resolve(
            memory_id,
            scope,
            detail="evidence",
            need_cas=False,
            include_suppressed=True,
            follow_chain=False,
        )
        if located is None:
            return []
        bound = located[1]
        hops: List[Dict[str, Any]] = [located[4]]
        seen = set()
        fields = _parse_ref_fields(hops[0].get("ref"))
        if fields is not None:
            seen.add(fields[2])
        for _ in range(_MAX_CHAIN_HOPS):
            nxt = (hops[-1].get("lifecycle") or {}).get("superseded_by")
            if not nxt:
                break
            sid = str(nxt).rsplit(":", 1)[0]
            if not sid or sid in seen:
                break
            seen.add(sid)
            try:
                nd = _as_dict(bound.inspect(sid, detail="evidence"))
            except (VerbatimError, ImportError, NotImplementedError):
                break
            if not nd.get("found", bool(nd.get("ref"))):
                break
            hops.append(nd)

        events: List[Dict[str, Any]] = []
        for hop_no, hop in enumerate(hops):
            hop_fields = _parse_ref_fields(hop.get("ref"))
            hop_sid = hop_fields[2] if hop_fields is not None else None
            revisions = hop.get("revisions") or []
            lifecycle = hop.get("lifecycle") or {}
            head = _int_or_none(lifecycle.get("mutation_head"))
            suppressed = bool(lifecycle.get("suppressed"))
            head_disposition = str(
                lifecycle.get("disposition") or lifecycle.get("state") or ""
            )
            current_text = _extract_text(hop)
            is_last_hop = hop_no == len(hops) - 1
            for i, rev in enumerate(revisions):
                if not isinstance(rev, dict):
                    rev = _as_dict(rev)
                rev_no = rev.get("revision")
                rev_no_i = _int_or_none(rev_no)
                is_head = (
                    rev_no_i is not None
                    and head is not None
                    and rev_no_i == head
                ) or (head is None and i == len(revisions) - 1)
                # Per-revision disposition is computed, not stored:
                # anything below the approved mutation head is superseded;
                # the head carries the lifecycle disposition (V5-14).
                disposition = rev.get("disposition") or rev.get("state")
                if disposition is None:
                    if is_head:
                        disposition = head_disposition or "recorded"
                    elif (
                        head is not None
                        and rev_no_i is not None
                        and rev_no_i < head
                    ):
                        disposition = "superseded"
                event = "ADD" if (hop_no == 0 and i == 0) else "UPDATE"
                entry = {
                    "id": f"{memory_id}:{hop_no}:{rev_no if rev_no is not None else i}",
                    "memory_id": memory_id,
                    "event": event,
                    "created_at": _iso_time(
                        rev.get("captured_us")
                        or rev.get("event_us")
                        or rev.get("recorded_time")
                        or rev.get("created_at")
                    ),
                    "verbatim_revision": rev_no,
                    "verbatim_disposition": disposition,
                    "verbatim_chain_hop": hop_no,
                    "verbatim_source_id": hop_sid,
                    "verbatim_integrity": rev.get("integrity"),
                    "verbatim_suppressed": rev.get("suppressed"),
                    "verbatim_held": rev.get("held"),
                    "verbatim_payload_bytes": rev.get("payload_bytes")
                    or rev.get("accepted_bytes"),
                }
                # Revision payloads are not echoed per-rev (verbatim
                # retains bytes but history reports the control trail);
                # the live head may carry the current text — no
                # fabrication for older revs.
                if is_head and current_text:
                    entry["new_memory"] = current_text
                events.append(
                    {k: v for k, v in entry.items() if v is not None}
                )
        # A suppressed/terminal head means the memory is deleted — mark
        # the terminal event rather than inventing a deletion entry.
        if events:
            last_lifecycle = hops[-1].get("lifecycle") or {}
            last_disp = str(
                last_lifecycle.get("disposition")
                or last_lifecycle.get("state")
                or ""
            )
            if last_lifecycle.get("suppressed") or last_disp in (
                "retracted",
                "erased",
                "purged",
            ):
                events[-1]["event"] = "DELETE"
                if last_lifecycle.get("suppressed"):
                    events[-1]["verbatim_suppressed"] = True
        return events

    def reset(self, **_kw: Any) -> None:
        raise UnsupportedOptionError(
            "reset() would wipe the whole store — not a scoped Mem0 "
            "operation here; use delete_all(confirm=True) per scope or "
            "destroy the store explicitly"
        )

    # ------------------------------------------------ verbatim extras

    def status(self) -> Dict[str, Any]:
        bound = self._facade(self._default_scope)
        d = _as_dict(_facade_call("status", bound.status))
        d["verbatim_status"] = "ok"
        return d

    def wait_ready(self, receipt: Any, **kwargs: Any) -> Dict[str, Any]:
        bound = self._facade(self._default_scope)
        return _as_dict(
            _facade_call("wait_ready", bound.wait_ready, receipt, **kwargs)
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            facades = list(self._facades.values())
            self._facades.clear()
        for facade in facades:
            try:
                facade.close()
            except Exception:
                pass

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ internals

    def _candidate_scopes(
        self, explicit: Optional[Tuple[Optional[str], Optional[str], Optional[str]]]
    ) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
        """Scopes an id may resolve under: the explicit/default scope first,
        then scopes this instance already bound. Never binds a scope the
        caller did not name or previously use — an id can only re-enter
        namespaces this application already accessed (V5-18.09)."""
        first = explicit if explicit is not None else self._default_scope
        order = [first]
        for scope in self._facades:
            if scope != first:
                order.append(scope)
        return order

    def _resolve(
        self,
        memory_id: str,
        scope: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None,
        *,
        detail: str = "evidence",
        need_cas: bool = True,
        include_suppressed: bool = False,
        follow_chain: bool = True,
    ) -> Optional[Tuple[Tuple[Optional[str], ...], Any, str, str, Dict[str, Any]]]:
        """memory_id → (scope, bound facade, ref-for-CAS, logical id, inspection).

        Tries the hinted/default scope, then every scope this instance
        already bound — the first facade whose ``inspect`` authorizes the
        id wins. For ``mref1.…`` refs the *caller's literal* ref is returned
        as the CAS fence (its embedded expected revision/control version is
        the fence — re-resolving to the current head would be blind
        last-write-wins, V5-18.11). For bare ids the facade's ``inspect``
        resolves to a live ``mref1.…`` pinned at the current mutation
        head/control version — CAS-ready as returned. A mid-flight facade
        that echoes no parseable ref falls back to ``_mint_ref``: the
        fence is synthesized from the authorized inspection's live head +
        control version and the bound facade's declared store/namespace
        identity, so a concurrent change between inspect and mutate still
        raises ``MemoryConflictError`` (V5-14.11). ``None`` when no bound
        scope can see the id.

        ``need_cas=False`` (get/history) skips ref minting entirely —
        read paths never need a mutation fence. ``include_suppressed``
        lets ``history`` audit a deleted memory's chain while
        get/update/delete treat suppressed/erased as absent (Mem0
        deleted-is-absent parity).

        ``follow_chain=True`` walks ``lifecycle.superseded_by`` pointers to
        the live chain head for bare ids — verbatim mints a *new* source
        per replacement, so a Mem0 logical memory is the supersession
        chain, and its live identity is the head. Literal ``mref1.…``
        addresses never follow: an explicit version fence resolves to
        exactly that version (stale writes must conflict, V5-14.11).
        """
        fields = _parse_ref_fields(memory_id)
        if memory_id.startswith(_REF_PREFIX + ".") and fields is None:
            raise Mem0CompatError(f"invalid verbatim ref: {memory_id!r}")
        candidates = self._candidate_scopes(scope)
        if fields is not None:
            ids = _ids_from_alias(fields[1])
            if ids:
                decoded = (
                    ids.get("user_id"),
                    ids.get("agent_id"),
                    ids.get("run_id"),
                )
                if decoded not in candidates:
                    candidates.insert(0, decoded)
        for sc in candidates:
            bound = self._facade(sc)
            try:
                d = _as_dict(bound.inspect(memory_id, detail=detail))
            except VerbatimError as exc:
                if exc.code in _NOT_FOUND_CODES:
                    continue
                raise _translate_error(exc, "resolve") from exc
            except (ImportError, NotImplementedError) as exc:
                raise FacadeUnavailableError(
                    f"resolve: facade inspect() is mid-flight/"
                    f"unprovisioned — {exc!r}"
                ) from exc
            if not d.get("found", bool(d.get("ref"))):
                continue
            lifecycle = d.get("lifecycle") or {}
            if not include_suppressed and _is_gone(lifecycle):
                continue
            if fields is not None:
                return sc, bound, memory_id, fields[2], d
            head_sid = memory_id
            if follow_chain:
                d, head_sid = self._follow_supersession(
                    bound, memory_id, d, detail
                )
                lifecycle = d.get("lifecycle") or {}
                if not include_suppressed and _is_gone(lifecycle):
                    continue
            ref = d.get("ref")
            if not isinstance(ref, str) or not ref:
                if need_cas:
                    raise FacadeUnavailableError(
                        "inspect() returned no ref for "
                        f"{memory_id!r} — cannot carry CAS"
                    )
                ref = memory_id
            elif need_cas and _parse_ref_fields(ref) is None:
                # Bare-id inspect echoes the caller's id (V5-06.11 alias
                # semantics), not a version-bound ref. Mutation needs a
                # source-backed MemoryRef (V5-14.11): mint it from the
                # authorized inspection's live head + control version and
                # the facade's declared store/namespace identity — for
                # the *resolved head*, never the stale predecessor.
                ref = self._mint_ref(bound, head_sid, d)
            return sc, bound, ref, memory_id, d
        return None

    def _follow_supersession(
        self, bound: Any, start_sid: str, d: Dict[str, Any], detail: str
    ) -> Tuple[Dict[str, Any], str]:
        """Walk ``superseded_by`` pointers to the live chain head.

        verbatim replacements mint a *new* source and mark the predecessor
        ``superseded`` with ``superseded_by = '<new_sid>:<rev>'``. The Mem0
        logical memory is the whole chain — its live identity is the head.
        Bounded and cycle-safe; a broken/denied hop keeps the last good
        inspection rather than guessing. Returns ``(inspection, head_sid)``
        — the sid the walk landed on (for CAS minting when the facade
        echoed no parseable ref).
        """
        seen = set()
        fields = _parse_ref_fields(d.get("ref"))
        last_sid = fields[2] if fields is not None else start_sid
        if last_sid:
            seen.add(last_sid)
        for _ in range(_MAX_CHAIN_HOPS):
            lifecycle = d.get("lifecycle") or {}
            nxt = lifecycle.get("superseded_by")
            if not nxt:
                return d, last_sid
            # '<sid>:<rev>' composite — revision is the final segment.
            sid = str(nxt).rsplit(":", 1)[0]
            if not sid or sid in seen:
                return d, last_sid
            seen.add(sid)
            try:
                nd = _as_dict(bound.inspect(sid, detail=detail))
            except VerbatimError as exc:
                if exc.code in _NOT_FOUND_CODES:
                    return d, last_sid
                raise _translate_error(exc, "resolve") from exc
            except (ImportError, NotImplementedError) as exc:
                raise FacadeUnavailableError(
                    "resolve: facade inspect() is mid-flight/unprovisioned "
                    f"— {exc!r}"
                ) from exc
            if not nd.get("found", bool(nd.get("ref"))):
                return d, last_sid
            d = nd
            last_sid = sid
        return d, last_sid

    def _mint_ref(
        self, bound: Any, source_id: str, inspection: Dict[str, Any]
    ) -> str:
        """Version-bound ref from authorized inspect + status data.

        ``lifecycle.mutation_head``/``control_version`` come from the
        authorized ``inspect``; ``store_tag``/``namespace`` from the bound
        facade's ``status()`` — never fabricated, never broadened.
        """
        cls = _load_memory_ref()
        lifecycle = inspection.get("lifecycle") or {}
        head = _int_or_none(lifecycle.get("mutation_head"))
        ctl = _int_or_none(lifecycle.get("control_version"))
        if cls is None or head is None or ctl is None:
            raise FacadeUnavailableError(
                "inspection lacks a control head — cannot mint a "
                "version-bound CAS ref"
            )
        status = _as_dict(_facade_call("status", bound.status))
        store_tag = status.get("store_tag")
        namespace = status.get("namespace")
        if not store_tag or not namespace:
            raise FacadeUnavailableError(
                "facade status() does not expose store_tag/namespace — "
                "cannot mint a version-bound CAS ref"
            )
        return cls(
            store_tag=str(store_tag),
            namespace=str(namespace),
            source_id=str(source_id),
            expected_revision=head,
            control_version=ctl,
        ).to_string()


class MemoryClient(Memory):
    """Mem0 platform-client shape bound to a *local* verbatim store.

    Platform-only constructor arguments (``api_key``, ``org_id``,
    ``project_id``, remote ``host`` URL, injected ``client``) raise
    ``UnsupportedOptionError`` — there is no remote tenancy here
    (V5-18.09). ``host`` may only be a verbatim ``HostAdapter`` binding.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        host: Any = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        _reject_unsupported(
            {
                "api_key": api_key,
                "org_id": org_id,
                "project_id": project_id,
                "client": client,
            },
            "MemoryClient",
        )
        if isinstance(host, str):
            raise UnsupportedOptionError(
                "MemoryClient host= is a remote Mem0 API URL in the "
                "platform client — verbatim binds an in-process "
                "HostAdapter or None"
            )
        super().__init__(path, host=host, **kwargs)

    # -------- platform parity helpers that are real (loops over scoped ops)

    def batch_update(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(memories, (list, tuple)):
            raise Mem0CompatError("batch_update expects a list of dicts")
        results = []
        for i, item in enumerate(memories):
            if not isinstance(item, dict):
                results.append({"index": i, "error": "not a dict"})
                continue
            mid = item.get("id") or item.get("memory_id")
            text = item.get("text") or item.get("data") or item.get("memory")
            try:
                out = self.update(mid, text)
                results.append({"index": i, "id": out.get("id"), "status": "ok"})
            except VerbatimError as exc:
                results.append(
                    {"index": i, "id": mid, "error": exc.to_dict()}
                )
        return {"results": results, "verbatim_status": "batch_complete"}

    def batch_delete(self, memories: Any) -> Dict[str, Any]:
        if not isinstance(memories, (list, tuple)):
            raise Mem0CompatError("batch_delete expects a list")
        results = []
        for i, item in enumerate(memories):
            if isinstance(item, dict):
                mid = item.get("id") or item.get("memory_id")
            else:
                mid = item
            try:
                self.delete(mid)
                results.append({"index": i, "id": mid, "status": "ok"})
            except VerbatimError as exc:
                results.append(
                    {"index": i, "id": mid, "error": exc.to_dict()}
                )
        return {"results": results, "verbatim_status": "batch_complete"}

    # -------- platform-only surface: explicit unsupported, never silent

    def _unsupported_method(self, name: str):
        def _raise(*_a: Any, **_k: Any) -> None:
            raise UnsupportedOptionError(
                f"MemoryClient.{name}() is a Mem0 platform-tenancy "
                "operation with no verbatim equivalent"
            )

        return _raise

    def users(self, *a: Any, **k: Any):
        return self._unsupported_method("users")(*a, **k)

    def delete_users(self, *a: Any, **k: Any):
        return self._unsupported_method("delete_users")(*a, **k)

    def feedback(self, *a: Any, **k: Any):
        return self._unsupported_method("feedback")(*a, **k)

    def get_project(self, *a: Any, **k: Any):
        return self._unsupported_method("get_project")(*a, **k)

    def update_project(self, *a: Any, **k: Any):
        return self._unsupported_method("update_project")(*a, **k)

    def create_memory_export(self, *a: Any, **k: Any):
        return self._unsupported_method("create_memory_export")(*a, **k)

    def get_memory_export(self, *a: Any, **k: Any):
        return self._unsupported_method("get_memory_export")(*a, **k)

    def get_memory_summary(self, *a: Any, **k: Any):
        return self._unsupported_method("get_memory_summary")(*a, **k)

    def list_entities(self, *a: Any, **k: Any):
        return self._unsupported_method("list_entities")(*a, **k)

    def delete_entity(self, *a: Any, **k: Any):
        return self._unsupported_method("delete_entity")(*a, **k)


# ------------------------------------------------------------- translation


def _translate_error(exc: VerbatimError, where: str) -> VerbatimError:
    """Map facade errors onto compat typed errors, preserving the code."""
    if exc.code in _CONFLICT_CODES:
        return MemoryConflictError(
            f"{where}: version conflict — the memory changed under the "
            f"supplied/resolved ref ({exc.message})",
            code=exc.code,
            retryable=exc.retryable,
            detail_id=exc.detail_id,
        )
    if exc.code in _NOT_FOUND_CODES:
        return MemoryNotFoundError(
            f"{where}: not found or not authorized ({exc.message})",
            code=exc.code,
            retryable=exc.retryable,
            detail_id=exc.detail_id,
        )
    return exc


def _mem0_search_result(
    res: Any, scope: Tuple[Optional[str], Optional[str], Optional[str]],
    *, threshold: Optional[float] = None,
    extra_warnings: Optional[List[str]] = None,
    enumeration: bool = False,
) -> Dict[str, Any]:
    d = _as_dict(res)
    status = str(d.get("status", "pending"))
    # On enumeration surfaces (get_all) a support-verdict miss is a
    # coverage statement — the verdict withheld candidates, so the
    # listing is partial, never a clean "insufficient answer".
    wire_status = (
        "partial"
        if enumeration
        and status in ("insufficient", "no_answer", "no_candidates")
        else status
    )
    warnings = list(d.get("warnings") or [])
    warnings.extend(extra_warnings or [])
    items: List[Dict[str, Any]] = []
    dropped_noncurrent = 0
    for hit in d.get("items") or []:
        hd = hit if isinstance(hit, dict) else _as_dict(hit)
        lifecycle = str(hd.get("lifecycle") or "").strip().lower()
        if lifecycle in _NONCURRENT_LIFECYCLES:
            # Mem0 results = current memories. Superseded/retracted
            # predecessors stay auditable via get-by-ref and history()
            # rather than masquerading as live hits.
            dropped_noncurrent += 1
            continue
        score = hd.get("score", 0.0) or 0.0
        if threshold is not None and score < float(threshold):
            continue
        item = {
            "id": hd.get("memory_id") or hd.get("ref"),
            "memory": hd.get("quote", ""),
            "score": score,
            "created_at": hd.get("recorded_time"),
            "updated_at": hd.get("valid_time") or hd.get("recorded_time"),
            "metadata": hd.get("metadata") or {},
            "verbatim_ref": hd.get("ref"),
            "verbatim_object_ref": hd.get("object_ref"),
            "verbatim_kind": hd.get("kind"),
            "verbatim_lifecycle": hd.get("lifecycle"),
            "verbatim_support": hd.get("support_status"),
            "verbatim_role": hd.get("role"),
            "verbatim_type": hd.get("type"),
            "verbatim_score_family": hd.get("score_family"),
            "verbatim_collapsed_duplicates": hd.get("collapsed_duplicates"),
            "verbatim_corroboration": hd.get("corroboration"),
            "verbatim_warnings": hd.get("warnings") or [],
        }
        item = {k: v for k, v in item.items() if v is not None}
        item.update({k: v for k, v in zip(_SCOPE_KEYS, scope) if v is not None})
        items.append(item)
    if threshold is not None:
        warnings.append(
            f"verbatim: applied client-side score threshold {threshold}"
        )
    if dropped_noncurrent:
        warnings.append(
            "verbatim: dropped "
            f"{dropped_noncurrent} non-current hit(s) (superseded/"
            "retracted predecessors remain auditable via history/get-by-ref)"
        )
    return {
        "results": items,
        "status": status,
        "verbatim_status": wire_status,
        "verbatim_warnings": warnings,
        "verbatim_readiness": d.get("readiness") or {},
        "verbatim_coverage": d.get("coverage") or {},
        "verbatim_dropped_noncurrent": dropped_noncurrent,
        "verbatim_schema": MEM0_COMPAT_SCHEMA,
    }


def _mem0_record(
    memory_id: str,
    inspection: Dict[str, Any],
    *,
    scope: Optional[Tuple[Optional[str], ...]] = None,
) -> Dict[str, Any]:
    """Inspection → Mem0 get() record shape + marked verbatim extras."""
    text = _extract_text(inspection)
    lifecycle = inspection.get("lifecycle") or {}
    provenance = inspection.get("provenance") or {}
    revisions = [
        r if isinstance(r, dict) else _as_dict(r)
        for r in (inspection.get("revisions") or [])
    ]
    first = revisions[0] if revisions else {}
    head_no = _int_or_none(lifecycle.get("mutation_head"))
    head_rev = next(
        (r for r in revisions if _int_or_none(r.get("revision")) == head_no),
        revisions[-1] if revisions else {},
    )
    source_state = lifecycle.get("source_state") or {}
    ids: Dict[str, str] = {}
    if scope:
        ids = {k: v for k, v in zip(_SCOPE_KEYS, scope) if v is not None}
    if not ids:
        ids = _ids_from_ref(inspection.get("ref"))
    metadata = dict(provenance.get("metadata") or {})
    if isinstance(head_rev.get("metadata"), dict):
        metadata.update(head_rev["metadata"])
    record = {
        "id": memory_id,
        "memory": text,
        "created_at": _iso_time(
            first.get("captured_us")
            or first.get("event_us")
            or provenance.get("created_us")
            or first.get("created_at")
        ),
        "updated_at": _iso_time(
            source_state.get("updated_at")
            or head_rev.get("captured_us")
            or head_rev.get("event_us")
            or lifecycle.get("effective_at")
            or lifecycle.get("updated_at")
        ),
        "metadata": metadata,
        "verbatim_ref": inspection.get("ref"),
        "verbatim_lifecycle": lifecycle.get("disposition")
        or lifecycle.get("state"),
        "verbatim_mutation_head": lifecycle.get("mutation_head"),
        "verbatim_superseded_by": lifecycle.get("superseded_by"),
        "verbatim_revisions": len(revisions),
        "verbatim_inspection_detail": inspection.get("detail"),
        "verbatim_warnings": inspection.get("warnings") or [],
    }
    if not text:
        record["verbatim_text_missing"] = True
    ref_fields = _parse_ref_fields(inspection.get("ref"))
    resolved_sid = ref_fields[2] if ref_fields is not None else None
    if (
        isinstance(resolved_sid, str)
        and resolved_sid
        and resolved_sid != memory_id
        and not memory_id.startswith(_REF_PREFIX + ".")
    ):
        # The addressed id resolved through supersession to a newer
        # source — surface the live identity without renaming the
        # caller's logical id (Mem0 keeps one id per memory).
        record["verbatim_current_id"] = resolved_sid
    record.update(ids)
    return {k: v for k, v in record.items() if v is not None}


def _ids_from_ref(ref: Any) -> Dict[str, str]:
    fields = _parse_ref_fields(ref)
    if fields is None:
        return {}
    return _ids_from_alias(fields[1])


#: Controls emit this sentinel on evidence locators when the QUOTE verb is
#: not granted or the span is held — it is a marker, never memory text.
_NO_TEXT_MARKERS = frozenset({"[unavailable]", "[held]", "[suppressed]"})


def _extract_text(node: Any, _depth: int = 0) -> str:
    """Find the retained text inside an Inspection (shape-tolerant)."""
    if _depth > 4:
        return ""
    if isinstance(node, dict):
        for key in ("quote", "text", "content", "memory", "payload"):
            value = node.get(key)
            if (
                isinstance(value, str)
                and value
                and value not in _NO_TEXT_MARKERS
            ):
                return value
        for key in (
            "evidence",
            "revisions",
            "items",
            "quotes",
            "spans",
            "provenance",
            "enrichment",
        ):
            value = node.get(key)
            found = _extract_text(value, _depth + 1)
            if found:
                return found
    elif isinstance(node, (list, tuple)):
        for item in node:
            found = _extract_text(item, _depth + 1)
            if found:
                return found
    return ""


__all__ = [
    "Memory",
    "MemoryClient",
    "Mem0CompatError",
    "UnsupportedOptionError",
    "ConfirmationRequiredError",
    "NamespaceBindingError",
    "MemoryNotFoundError",
    "MemoryConflictError",
    "FacadeUnavailableError",
    "facade_available",
    "MEM0_COMPAT_SCHEMA",
    "MEM0_SURFACE_PIN",
]
