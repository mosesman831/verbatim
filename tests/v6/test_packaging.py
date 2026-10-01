"""V6 packaging + distribution contract tests (docs/v6_contracts.md §4–§5).

Covers the ship-surface requirements: the wheel carries ``py.typed`` and
``LICENSE``, both new console scripts resolve to real callables, and the
TypeScript memory client exists, parses, loads under Node (type-stripped),
and speaks the frozen ``/v2/memory`` wire format against a stub server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = ROOT / "clients" / "ts" / "memory-client"
CLIENT_SRC = CLIENT_DIR / "src" / "index.ts"
NODE = shutil.which("node")

_needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")


# ---------------------------------------------------------------------------
# Repo markers
# ---------------------------------------------------------------------------


def test_py_typed_marker_exists():
    assert (ROOT / "verbatim" / "py.typed").is_file()


def test_license_file_is_mit():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Permission is hereby granted" in text


# ---------------------------------------------------------------------------
# Console entry points
# ---------------------------------------------------------------------------


def test_console_entry_points_importable():
    from verbatim.api_v3.mcp import main as mcp_main
    from verbatim.service.__main__ import main as service_main

    assert callable(service_main)
    assert callable(mcp_main)


def test_pyproject_declares_entry_points():
    import tomllib

    doc = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = doc["project"]["scripts"]
    assert scripts["verbatim-service"] == "verbatim.service.__main__:main"
    assert scripts["verbatim-mcp"] == "verbatim.api_v3.mcp:main"
    assert doc["project"]["name"] == "verbatim-memory"
    providers = doc["project"]["entry-points"]["hermes_agent.memory_providers"]
    assert providers["verbatim"] == "verbatim:register"
    assert "py.typed" in doc["tool"]["setuptools"]["package-data"]["verbatim"]


def test_service_main_help_parses(capsys):
    from verbatim.service.__main__ import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--path", "--user", "--bind", "--port", "--token", "--token-file"):
        assert flag in out


# ---------------------------------------------------------------------------
# Wheel contents
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = list(out.glob("hermes_verbatim-*.whl"))
    assert wheels, proc.stdout + proc.stderr
    return wheels[0]


def test_wheel_contains_markers_and_entry_points(wheel_path):
    names = set()
    entry_points = ""
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
        for name in names:
            if name.endswith("entry_points.txt"):
                entry_points = zf.read(name).decode("utf-8")
    assert "verbatim/py.typed" in names
    assert "verbatim/plugin.yaml" in names
    # setuptools >= 69 lays license files under dist-info/licenses/; older
    # releases drop them at dist-info top level — accept either location.
    assert any(
        n.endswith(".dist-info/LICENSE") or n.endswith(".dist-info/licenses/LICENSE")
        for n in names
    ), sorted(n for n in names if "dist-info" in n)
    assert 'verbatim-service = "verbatim.service.__main__:main"' in entry_points.replace(
        "'", '"'
    ) or "verbatim.service.__main__:main" in entry_points
    assert "verbatim.api_v3.mcp:main" in entry_points


# ---------------------------------------------------------------------------
# TypeScript client — files, package metadata, Node load + wire format
# ---------------------------------------------------------------------------


def test_ts_client_files_and_package_json():
    assert CLIENT_SRC.is_file()
    pkg = json.loads((CLIENT_DIR / "package.json").read_text(encoding="utf-8"))
    assert pkg["name"] == "@verbatim/memory-client"
    assert pkg["main"] == "src/index.ts"
    assert pkg["types"] == "src/index.ts"


@_needs_node
def test_ts_client_loads_under_node():
    """``node --check`` cannot parse TS — the honest syntax pass is an
    import through Node's type stripping (Node >= 22.6)."""
    proc = subprocess.run(
        [
            NODE,
            "-e",
            f"import({json.dumps(CLIENT_SRC.as_uri())}).then(m => {{"
            " if (typeof m.MemoryClient !== 'function') process.exit(3);"
            " console.log('loaded'); })",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "loaded" in proc.stdout


class _StubHandler(BaseHTTPRequestHandler):
    """Records the request and answers the frozen /v2/memory shapes."""

    seen: list = []

    def _reply(self, doc: dict, status: int = 200) -> None:
        payload = json.dumps(doc).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        entry = {
            "method": self.command,
            "path": self.path,
            "authorization": self.headers.get("authorization"),
            "content_type": self.headers.get("content-type"),
            "body": json.loads(body) if body else None,
        }
        type(self).seen.append(entry)
        return entry

    def do_POST(self) -> None:  # noqa: N802
        entry = self._record()
        if entry["authorization"] != "Bearer stub-token":
            self._reply({"error": "unauthorized", "code": "not_found_or_unauthorized", "retryable": False}, 401)
            return
        if self.path == "/v2/memory/add":
            self._reply(
                {
                    "schema": "v5/1",
                    "memory_id": "m1",
                    "ref": "mref1.abc",
                    "source_revision": 1,
                    "receipt_id": "rcpt-1",
                    "acceptance": "accepted",
                    "replayed": False,
                    "readiness": {"source_lexical": "pending"},
                    "warnings": [],
                    "possible_updates": [],
                    "inference": "not_requested",
                }
            )
        elif self.path == "/v2/memory/search":
            self._reply(
                {
                    "schema": "v5/1",
                    "status": "ready",
                    "items": [],
                    "warnings": [],
                    "readiness": {},
                    "coverage": {},
                    "causal_token": "ct-1",
                }
            )
        else:
            self._reply({"error": "nope", "code": "not_found_or_unauthorized", "retryable": False}, 404)

    def do_GET(self) -> None:  # noqa: N802
        entry = self._record()
        if entry["authorization"] != "Bearer stub-token":
            self._reply({"error": "unauthorized", "code": "not_found_or_unauthorized", "retryable": False}, 401)
            return
        if self.path == "/v2/memory/readiness/rcpt-1":
            self._reply(
                {
                    "schema": "v5/1",
                    "receipt_id": "rcpt-1",
                    "state": "ready",
                    "capabilities": {"source_lexical": "ready"},
                    "causal_satisfied": True,
                    "waited_ms": 3.0,
                }
            )
        else:
            self._reply({"error": "nope", "code": "not_found_or_unauthorized", "retryable": False}, 404)

    def log_message(self, *args) -> None:
        pass


@_needs_node
def test_ts_client_wire_format_against_stub():
    """Drive MemoryClient against a Python stub server and verify the
    Authorization header, method/path, and JSON bodies it emits."""
    _StubHandler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    script = f"""
    import({json.dumps(CLIENT_SRC.as_uri())}).then(async (m) => {{
      const c = new m.MemoryClient({{
        baseUrl: 'http://127.0.0.1:{port}', token: 'stub-token', timeoutMs: 5000,
      }});
      const a = await c.add('hello evidence', {{ infer: true, metadata: {{ k: 'v' }} }});
      const s = await c.search('evidence', {{ limit: 3, consistency: 'session' }});
      const r = await c.waitReady(a.receipt_id, 2000, 25);
      if (a.receipt_id !== 'rcpt-1' || s.causal_token !== 'ct-1' || r.state !== 'ready') {{
        console.error('unexpected payloads', JSON.stringify({{ a, s, r }}));
        process.exit(4);
      }}
      console.log('wire-ok');
    }}).catch((e) => {{ console.error(String(e)); process.exit(5); }});
    """
    try:
        proc = subprocess.run(
            [NODE, "-e", script], capture_output=True, text=True, timeout=60
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wire-ok" in proc.stdout

    by_path = {e["path"]: e for e in _StubHandler.seen}
    add = by_path["/v2/memory/add"]
    assert add["method"] == "POST"
    assert add["authorization"] == "Bearer stub-token"
    assert add["content_type"] == "application/json"
    assert add["body"]["text"] == "hello evidence"
    assert add["body"]["infer"] is True
    assert add["body"]["metadata"] == {"k": "v"}

    search = by_path["/v2/memory/search"]
    assert search["body"]["query"] == "evidence"
    assert search["body"]["limit"] == 3
    assert search["body"]["consistency"] == "session"

    readiness = by_path["/v2/memory/readiness/rcpt-1"]
    assert readiness["method"] == "GET"
    assert readiness["authorization"] == "Bearer stub-token"
