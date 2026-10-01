"""Reference extraction for v3 harvest candidates (SPEC_V3 §15.05, §15.09).

Deterministic, zero-ML, zero-network extraction of machine-meaningful tokens
from a candidate's text. The result attaches to each candidate dict under
``refs`` and feeds retrieval lanes later (exact lookup, identifier lane,
environment-state revalidation). Every entry is a *hint* — a verbatim token
found in the text — never a semantic judgment.

Ref kinds (all lists deduplicated, order-preserving, bounded):

- ``paths``        — file paths (``src/a.py``, ``/etc/x``, ``C:\\x\\y``,
  ``~/f``, ``./rel/x``) and bare filenames with recognized dev extensions
  (``test_x.py``, ``config.yaml``).
- ``tests``        — test node ids (``tests/test_x.py::test_y``).
- ``identifiers``  — snake_case / CamelCase code tokens.
- ``urls``         — http(s) URLs, trailing punctuation stripped.
- ``commands``     — command invocations: ``$ ...`` prompt lines, backticked
  commands, shell-verb invocations (``pytest tests/ -x``), and call-shaped
  tool uses (``EditFile(path="a.py")``).
- ``error_codes``  — exception class names (``ValueError``), alphanumeric
  codes (``E501``, ``TS2345``), ``Errno N``, ``exit code N``, ``HTTP NNN``.
- ``versions``     — semver-shaped versions (``1.2.3``, ``v1.2``) covering the
  V3-15.09 "versions" slice; numbers without version shape are not claimed.

``REF_EXTRACTOR_VERSION`` versions the rules so a refs set is reproducible
(V3-15.12).
"""

from __future__ import annotations

import re
from typing import Iterator

REF_EXTRACTOR_VERSION = "refs-v3-1"

#: Bound per ref kind — hints are advisory; unbounded lists would be a
#: resource hazard on adversarial payloads.
_MAX_PER_KIND = 64

_REF_KINDS = (
    "paths",
    "tests",
    "identifiers",
    "urls",
    "commands",
    "error_codes",
    "versions",
)

# ---------------------------------------------------------------------------
# Versioned patterns
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\bhttps?://[^\s<>'\")\]}]+")
_URL_TRAIL = ".,;:!?)]}>\"'"

#: Paths require at least one separator. Optional prefixes cover absolute,
#: home-relative, dot-relative, and drive-letter roots; bare ``a/b`` is enough.
_PATH_RE = re.compile(
    r"(?<![\w/\\])"
    r"(?:[A-Za-z]:[\\/]|~[\\/]|\.{1,2}[\\/]|/)?"
    r"[\w.@~+\-]+(?:[\\/][\w.@~+\-]+)+[\\/]?"
)

#: Bare filenames are paths only when the extension is a recognized
#: development/document type — ``os.path`` or ``example.com`` must not
#: masquerade as files.
_FILE_EXTS = frozenset(
    "py pyi js jsx ts tsx mts cts mjs cjs json jsonl yaml yml toml ini cfg "
    "conf xml html htm css scss less md markdown rst txt log sql rs go java "
    "kt kts scala c h cc cpp cxx hpp hh cs fs fsx rb php pl pm jl lua sh "
    "bash zsh fish ps1 bat cmd swift m mm dart ex exs erl hrl clj cljs edn "
    "hs lhs ml mli nim zig sv tcl vim el lisp scm groovy gradle tf hcl proto "
    "graphql gql vue svelte astro ipynb diff patch mk cmake ninja bzl lock "
    "csv tsv parquet wasm wat so dll dylib exe bin dat db sqlite sqlite3 "
    "png jpg jpeg gif webp svg ico pdf zip tar gz bz2 xz 7z whl apk deb rpm "
    "dmg iso env gitignore gitattributes dockerignore editorconfig".split()
)
_FILENAME_RE = re.compile(r"\b[\w\-]+\.([A-Za-z0-9]{1,10})\b")

#: pytest-style node ids and class::method references.
_NODEID_RE = re.compile(r"\b[\w./\\\-]+::[\w.\[\]\-]+(?:::[\w.\[\]\-]+)*")

#: ``Name(args)`` call-shaped invocations (tool calls, function calls).
_CALL_RE = re.compile(r"\b[A-Za-z_][\w.]*\([^()\n]{0,160}\)")

#: ``$ cmd args`` / ``> cmd`` prompt lines.
_PROMPT_CMD_RE = re.compile(r"^[ \t]*[$#>][ \t]+(.{1,200}?)[ \t]*$", re.MULTILINE)

_BACKTICK_RE = re.compile(r"`([^`\n]{1,160})`")

#: Shell/devtool verbs that mark a command invocation when word-initial.
_SHELL_VERBS = frozenset(
    "git npm pnpm yarn bun deno node python python3 python2 pip pip3 uv "
    "poetry pipenv conda pytest tox nox cargo rustc rustup go gofmt make "
    "cmake ninja meson bazel buck docker docker-compose kubectl helm k9s "
    "terraform ansible vagrant apt apt-get brew yum dnf pacman zypper "
    "systemctl service journalctl curl wget ssh scp sftp rsync tar gzip "
    "gunzip zip unzip grep rg find sed awk xargs cat ls cd pwd mkdir rmdir "
    "rm cp mv ln chmod chown sudo env which echo printf head tail wc sort "
    "uniq diff patch touch kill ps df du mount ping netstat ip gcc g++ "
    "clang javac java mvn gradle dotnet ruby gem bundle rake php composer "
    "perl julia tsc eslint prettier webpack vite jest mocha vitest ava "
    "playwright cypress ffmpeg convert magick sqlite3 psql mysql redis-cli "
    "aws gcloud az gh glab pandoc".split()
)

#: A word that could begin a shell command (position-anchored: line start or
#: after whitespace/command separators).
_WORD_AT_RE = re.compile(r"(?:(?<=^)|(?<=[ \t;|&>`\"'(:]))([a-z][a-z0-9_+.\-]{0,30})")

_SNAKE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*\b")
_CAMEL_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")

_ERRCODE_RE = re.compile(r"\b[A-Z]{1,5}\d{2,5}\b")
_EXC_NAME_RE = re.compile(
    r"\b[A-Za-z_][\w.]*(?:Error|Exception|Warning|Fault|Failure)\b"
)
_EXIT_RE = re.compile(
    r"\b(?:exit(?:ed)?(?:\s+code)?|status|errno|signal|code)\s+\d+\b",
    re.IGNORECASE,
)
_HTTP_CODE_RE = re.compile(r"\bHTTP/?\s?\d{3}\b|\bHTTP\s+\d{3}\b", re.IGNORECASE)

#: Version shapes: x.y.z (any depth >= 2 dots) or v-prefixed x.y. Bare ``3.14``
#: is a decimal, not a version claim.
_VERSION_RE = re.compile(r"\bv\d+\.\d+(?:\.\d+)*\b|\b\d+\.\d+\.\d+(?:\.\d+)*\b")


def _dedupe(items: list[str], cap: int = _MAX_PER_KIND) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
            if len(out) >= cap:
                break
    return out


def _mask_spans(mask: list[str], rx: "re.Pattern[str]", src: str) -> Iterator[re.Match]:
    """Yield matches of ``rx`` on ``src`` while blanking their spans in
    ``mask`` so later extractors do not re-tokenize the same bytes."""
    for m in rx.finditer(src):
        for k in range(m.start(), m.end()):
            mask[k] = " "
        yield m


def _command_tail(text: str, start: int) -> int:
    """End index of a command invocation beginning at ``start``: the verb plus
    up to 12 whitespace-separated argument tokens, stopping at command
    separators, quotes, brackets, or newline."""
    n = min(len(text), start + 200)
    i = start
    while i < n and not text[i].isspace() and text[i] not in "|&;`'\"<>":
        i += 1
    args = 0
    while args < 12:
        j = i
        while j < n and text[j] in " \t":
            j += 1
        if j >= n or text[j] in "\n|&;`'\"<>(){}":
            break
        k = j
        while k < n and text[k] not in " \t\n|&;`'\"<>{}":
            k += 1
        i = k
        args += 1
    return i


def _commands(text: str) -> list[str]:
    cmds: list[str] = []
    for m in _PROMPT_CMD_RE.finditer(text):
        cmds.append(m.group(1))
    for m in _BACKTICK_RE.finditer(text):
        s = m.group(1).strip()
        if not s:
            continue
        first = re.split(r"[ \t]", s, 1)[0]
        if first in _SHELL_VERBS or " " in s or "/" in s:
            cmds.append(s)
    for m in _WORD_AT_RE.finditer(text):
        if m.group(1) in _SHELL_VERBS:
            end = _command_tail(text, m.start(1))
            cmds.append(text[m.start(1) : end].rstrip())
    for m in _CALL_RE.finditer(text):
        cmds.append(m.group(0))
    return cmds


def _is_camel(tok: str) -> bool:
    return (
        tok[0].isupper()
        and sum(1 for c in tok if c.isupper()) >= 2
        and any(c.islower() for c in tok)
    )


def extract_refs(text: str) -> dict[str, list[str]]:
    """Extract structured reference hints from candidate text.

    Deterministic and total: any input returns the fixed-key dict (possibly
    all-empty lists) and never raises. Extraction order matters: URLs are
    masked first so their inner path-like text is not double-counted, then
    paths/filenames are masked before identifier scanning so ``src/a.py``
    does not also emit the fragment identifiers ``a`` and ``py``.
    """
    refs: dict[str, list[str]] = {k: [] for k in _REF_KINDS}
    if not isinstance(text, str) or not text:
        return refs

    mask = list(text)

    urls = [
        m.group(0).rstrip(_URL_TRAIL)
        for m in _mask_spans(mask, _URL_RE, text)
    ]
    url_masked = "".join(mask)

    # Paths run on the url-masked text while node ids are still visible, so
    # ``tests/test_x.py::test_y`` contributes ``tests/test_x.py`` to paths.
    paths: list[str] = []
    for m in _mask_spans(mask, _PATH_RE, url_masked):
        paths.append(m.group(0).rstrip("/\\"))
    # Filenames run on the *post*-path snapshot so ``a.py`` inside
    # ``src/a.py`` is not double-counted as a bare filename.
    post_path = "".join(mask)
    for m in _mask_spans(mask, _FILENAME_RE, post_path):
        if m.group(1).lower() in _FILE_EXTS:
            paths.append(m.group(0))

    # Node ids are masked for the identifier scan; their path prefix was
    # already captured above.
    tests = [m.group(0) for m in _mask_spans(mask, _NODEID_RE, url_masked)]

    ident_masked = "".join(mask)
    identifiers = [m.group(0) for m in _SNAKE_RE.finditer(ident_masked)]
    identifiers += [
        m.group(0)
        for m in _CAMEL_CANDIDATE_RE.finditer(ident_masked)
        if _is_camel(m.group(0))
    ]

    error_codes = [m.group(0) for m in _EXC_NAME_RE.finditer(text)]
    error_codes += [m.group(0) for m in _ERRCODE_RE.finditer(text)]
    error_codes += [m.group(0) for m in _EXIT_RE.finditer(text)]
    error_codes += [m.group(0) for m in _HTTP_CODE_RE.finditer(text)]

    refs["paths"] = _dedupe(paths)
    refs["tests"] = _dedupe(tests)
    refs["identifiers"] = _dedupe(identifiers)
    refs["urls"] = _dedupe(urls)
    refs["commands"] = _dedupe(_commands(text))
    refs["error_codes"] = _dedupe(error_codes)
    refs["versions"] = _dedupe([m.group(0) for m in _VERSION_RE.finditer(text)])
    return refs
