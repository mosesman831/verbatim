"""Packaging smoke test: drive an installed Verbatim through its public API.

Run against a wheel installed into a clean environment (no source tree on
the path). Two checks:

1. Library path: capture -> readiness -> recall -> inspect -> delete closure.
2. HTTP path: start the documented ``verbatim-service`` launcher and prove
   that it serves the bearer-authenticated ``/v2/memory/*`` surface, failing
   closed (401) without a token.
"""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

TOKEN = "smoke-token"


def check_library(store_dir: pathlib.Path) -> str:
    from verbatim import Memory

    mem = Memory(str(store_dir / "store.vdb"))
    try:
        added = mem.add("the release tag is v6.2", metadata={"topic": "release"})
        ready = mem.wait_ready(added.receipt_id)
        assert ready.state == "ready", f"readiness state {ready.state!r}"

        res = mem.search("what is the release tag?")
        assert res.status == "ready", f"search status {res.status!r}"
        assert res.items, "recall returned no items"
        quote = res.items[0].quote
        assert quote == "the release tag is v6.2", f"unexpected quote {quote!r}"

        detail = mem.inspect(res.items[0].ref)
        assert detail.found, "inspect did not resolve the reference"

        mem.forget(res.items[0].ref)
        after = mem.search("what is the release tag?")
        assert not after.items, "deletion did not suppress the memory"
    finally:
        mem.close()
    return f"library OK: recalled {quote!r}, deletion closed cleanly"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def check_http(store_dir: pathlib.Path) -> str:
    port = _free_port()
    base = f"http://127.0.0.1:{port}/v2/memory"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "verbatim.service",
            "--path",
            str(store_dir / "service.vdb"),
            "--user",
            "me",
            "--token",
            TOKEN,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 30
        last: Exception | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read() if proc.stderr else "")[-800:]
                raise AssertionError(f"verbatim-service exited early (rc={proc.returncode}): {err}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError as exc:  # not listening yet
                last = exc
                time.sleep(0.25)
        else:
            raise AssertionError(f"verbatim-service never accepted connections: {last}")

        # fail closed without a token
        try:
            urllib.request.urlopen(f"{base}/capabilities", timeout=10)
            raise AssertionError("unauthenticated request was NOT rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, f"expected 401 unauthenticated, got {exc.code}"

        status, caps = _request(f"{base}/capabilities")
        assert status == 200, f"capabilities -> {status}"
        assert "capabilities" in caps, f"unexpected capabilities payload: {caps}"

        status, added = _request(f"{base}/add", {"text": "the release tag is v6.2"})
        assert status == 200 and added.get("acceptance") == "accepted", f"add -> {status} {added}"

        status, found = _request(f"{base}/search", {"query": "what is the release tag?"})
        assert status == 200, f"search -> {status}"
        assert found.get("answerability") in {"supported", "insufficient"}, found
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return f"http OK: 401 without a token, capabilities + add (accepted) + search over loopback :{port}"


def main() -> int:
    store_dir = pathlib.Path(tempfile.mkdtemp(prefix="verbatim-smoke-"))
    print("wheel smoke OK:", check_library(store_dir))
    print("wheel smoke OK:", check_http(store_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())