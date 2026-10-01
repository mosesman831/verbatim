"""``holographic`` connector (SPEC_V4 §30/§48) — import a structured
export of a Holographic-style provider's episodes/entities/claims.

Format: ``holographic-export-v1`` — a DECLARED interchange format owned
by this build, not a verified mirror of the provider's private schema
(Holographic is a Hermes memory plugin whose on-disk store is
SQLite/phase-vector HRR [R02/R03]; no public JSON export contract is
pinned in the spec sources). The descriptor marks
``provider_format_fidelity`` declared-not-verified — a successful import
count never claims semantic equivalence it cannot prove (V4-48.10).

Wire shapes accepted (``source = {"path": ...}``):

* ``*.json`` — one object ``{"format": "holographic-export-v1",
  "provider": "holographic", "exported_us": int, "source_scope": str?,
  "items": [item, ...]}`` (a bare list of items is also accepted and
  reported as a missing-format-header loss).
* ``*.jsonl`` — one JSON object per line; an optional first line carries
  the ``{"format": ...}`` header; every other line is one item.

Item fields::

    {"id": str (required — stable remote identity),
     "kind": "episode"|"entity"|"claim"|"document" (required),
     "text"|"summary"|"content": str (one required — verbatim bytes),
     "original": bool            — true only for byte-exact originals;
                                   everything else imports as an
                                   imported ASSERTION (V4-48.02),
     "actor": str?, "event_us"|"occurred_us": int?,
     "revision": str?,           — upstream version marker
     "refs": [str, ...],         — upstream ids this item derives from
     "scope": str?               — remote namespace → reported in
                                   dry-run scope_mappings
     ...}                        — unknown top-level fields are counted
                                   as import losses, never hidden.

Provenance honesty: items without ``original: true`` persist with
``trust_class=imported`` and ``imported_assertion`` metadata — extracted
facts/summaries are labeled assertions, never reconstructed as
byte-exact original evidence (V4-48.02). Missing ids are rejected items;
missing timestamps/actors are ``missing_provenance`` reports, not
fabricated values.

Cursor = decimal item ordinal (position in the export stream); resume
yields items with ordinal strictly greater than the committed cursor.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Mapping, Optional

from ..core.types import ErrorCode, VerbatimError, safe_json_loads
from .base import ConnectorDescriptor, RemoteItem

CONNECTOR_ID = "holographic"
CONNECTOR_VERSION = "1.0"
FORMAT_ID = "holographic-export-v1"

_ITEM_KINDS = frozenset({"episode", "entity", "claim", "document"})
_KNOWN_ITEM_FIELDS = frozenset(
    {
        "id",
        "kind",
        "text",
        "summary",
        "content",
        "original",
        "actor",
        "event_us",
        "occurred_us",
        "valid_from_us",
        "valid_until_us",
        "revision",
        "version",
        "refs",
        "scope",
        "meta",
    }
)
_MAX_EXPORT_BYTES = 256 * 1024 * 1024


class HolographicConnector:
    """Holographic structured-export importer. ``source`` descriptor::

        {"path": "/abs/export.jsonl",            # required
         "format": "holographic-export-v1"}      # optional assert
    """

    def descriptor(self) -> ConnectorDescriptor:
        return ConnectorDescriptor(
            connector_id=CONNECTOR_ID,
            version=CONNECTOR_VERSION,
            display_name="Holographic structured export (episodes/entities/claims)",
            remote=False,
            formats=(FORMAT_ID,),
            item_classes=("episode", "entity", "claim", "document"),
            capabilities={
                "revisions": True,
                "dry_run": True,
                "deletions": False,
                "remote_fetch": False,
            },
            # The provider's real store is SQLite + HRR phase vectors
            # [R03]; this JSON interchange is our declared mapping, not a
            # verified round-trip of the provider's private schema.
            declared_not_verified=("provider_format_fidelity",),
        )

    def validate_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(source, Mapping):
            raise VerbatimError(
                ErrorCode.VALIDATION, "holographic source must be a mapping"
            )
        raw = source.get("path")
        if not isinstance(raw, str) or not raw:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "holographic source requires a non-empty 'path'",
            )
        path = os.path.realpath(raw)
        if not os.path.isfile(path):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"holographic path {raw!r} is not a file",
            )
        fmt = source.get("format", FORMAT_ID)
        if fmt != FORMAT_ID:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"holographic source format must be {FORMAT_ID!r}",
            )
        return {"path": path, "format": FORMAT_ID}

    # ------------------------------------------------------------------

    def scan(
        self,
        source: Mapping[str, Any],
        *,
        cursor: Optional[str],
        limit: Optional[int] = None,
    ) -> Iterator[RemoteItem]:
        path = source["path"]
        exported_us, source_scope, raw_items = self._load(path)
        start = 0
        if cursor is not None:
            try:
                start = int(cursor) + 1
            except (TypeError, ValueError) as exc:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"holographic cursor {cursor!r} is not an ordinal",
                ) from exc
        emitted = 0
        for ordinal in range(start, len(raw_items)):
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield self._item(
                raw_items[ordinal], ordinal, exported_us, source_scope
            )

    # ------------------------------------------------------------------

    def _load(
        self, path: str
    ) -> tuple[Optional[int], Optional[str], list[Any]]:
        """Parse the export file → (exported_us, source_scope, items).

        The export FILE itself must be well-formed UTF-8 JSON/JSONL — a
        malformed export is a connector-level VALIDATION failure (the
        pull cannot honestly enumerate it), distinct from per-item
        content rejections.
        """
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"cannot stat export {path!r}: {exc}"
            ) from exc
        if size > _MAX_EXPORT_BYTES:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"export exceeds {_MAX_EXPORT_BYTES} bytes",
            )
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"cannot read export {path!r}: {exc}"
            ) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "export file is not valid UTF-8",
            ) from exc

        header: dict[str, Any] = {}
        items: list[Any] = []
        try:
            parsed = safe_json_loads(text)
        except VerbatimError:
            parsed = None  # not a single JSON document — try JSONL
        if isinstance(parsed, dict) and "items" in parsed:
            header = parsed
            items = parsed["items"]
        elif isinstance(parsed, list):
            items = parsed
            header = {}
        else:
            # JSONL: one object per non-blank line; an optional first
            # object may carry the format header (no item fields).
            lines = [ln for ln in text.splitlines() if ln.strip()]
            first = True
            for ln in lines:
                obj = safe_json_loads(ln)
                if not isinstance(obj, dict):
                    raise VerbatimError(
                        ErrorCode.VALIDATION,
                        "export line is not a JSON object",
                    )
                if first and "format" in obj and "kind" not in obj:
                    header = obj
                    first = False
                    continue
                first = False
                items.append(obj)
        if not isinstance(items, list):
            raise VerbatimError(
                ErrorCode.VALIDATION, "export 'items' must be a list"
            )
        fmt = header.get("format")
        if fmt is not None and fmt != FORMAT_ID:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"unsupported export format {fmt!r} — expected {FORMAT_ID!r}",
            )
        exported_us = header.get("exported_us")
        if exported_us is not None and (
            isinstance(exported_us, bool)
            or not isinstance(exported_us, int)
            or exported_us < 0
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "header exported_us must be an int >= 0",
            )
        source_scope = header.get("source_scope")
        if source_scope is not None and not isinstance(source_scope, str):
            raise VerbatimError(
                ErrorCode.VALIDATION, "header source_scope must be a string"
            )
        return exported_us, source_scope, items

    def _item(
        self,
        raw: Any,
        ordinal: int,
        exported_us: Optional[int],
        source_scope: Optional[str],
    ) -> RemoteItem:
        cursor = str(ordinal)
        if not isinstance(raw, dict):
            return RemoteItem(
                external_id=f"ord:{ordinal}",
                cursor=cursor,
                content=b"",
                reject_reason="item_not_object",
            )
        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id:
            return RemoteItem(
                external_id=f"ord:{ordinal}",
                cursor=cursor,
                content=b"",
                reject_reason="missing_id",
            )
        kind = raw.get("kind")
        if kind not in _ITEM_KINDS:
            return RemoteItem(
                external_id=item_id,
                cursor=cursor,
                content=b"",
                reject_reason=f"unknown_kind:{kind!r}"[:64],
            )
        text = raw.get("text") or raw.get("summary") or raw.get("content")
        if not isinstance(text, str) or not text:
            return RemoteItem(
                external_id=item_id,
                cursor=cursor,
                content=b"",
                reject_reason="missing_text",
            )
        # V4-48.02: only an explicit ``original: true`` marks byte-exact
        # evidence; entities/claims/summaries default to imported
        # assertions.
        original = raw.get("original") is True and kind in (
            "episode",
            "document",
        )
        missing = []
        if raw.get("event_us") is None and raw.get("occurred_us") is None:
            missing.append("event_us")
        if not isinstance(raw.get("actor"), str) or not raw.get("actor"):
            missing.append("actor")
        event_us = raw.get("event_us") or raw.get("occurred_us")
        if event_us is not None and (
            isinstance(event_us, bool)
            or not isinstance(event_us, int)
            or event_us < 0
        ):
            event_us = None
            missing.append("event_us")
        refs = raw.get("refs")
        source_ids = (
            tuple(str(r) for r in refs)
            if isinstance(refs, (list, tuple))
            else ()
        )
        losses = sorted(
            k for k in raw if k not in _KNOWN_ITEM_FIELDS
        )
        revision = raw.get("revision") or raw.get("version")
        extra: dict[str, Any] = {"export_ordinal": ordinal}
        if source_scope:
            extra["scope"] = source_scope
        if isinstance(raw.get("scope"), str) and raw["scope"]:
            extra["scope"] = raw["scope"]
        meta = raw.get("meta")
        if isinstance(meta, dict):
            extra["export_meta"] = meta
        for tf in ("valid_from_us", "valid_until_us"):
            if isinstance(raw.get(tf), int) and not isinstance(
                raw.get(tf), bool
            ):
                extra[tf] = raw[tf]
        return RemoteItem(
            external_id=item_id,
            cursor=cursor,
            content=text.encode("utf-8"),
            media_type="text/plain",
            item_class=kind,
            event_us=event_us if event_us is not None else exported_us,
            author_id=(
                raw["actor"] if isinstance(raw.get("actor"), str) else None
            ),
            imported_assertion=not original,
            remote_revision=(
                str(revision) if revision is not None else None
            ),
            missing_provenance=tuple(missing),
            source_ids=source_ids,
            losses=tuple(losses),
            extra=extra,
        )


__all__ = ["CONNECTOR_ID", "FORMAT_ID", "HolographicConnector"]
