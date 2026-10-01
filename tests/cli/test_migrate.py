"""`verbatim migrate <path>`: explicit operator migration (SPEC §21)."""

from __future__ import annotations

import json
import os
import sqlite3

from verbatim.cli import main as cli_main
from verbatim.core.time import now_us
from verbatim.core.types import json_dumps
from verbatim.storage.schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
    FTS_TRIGGERS,
    SCHEMA_VERSION,
)


def _make_v2_db(path: str) -> None:
    """Build a genuine schema-v2 database file (pre-v3 DDL)."""
    key_path = path + ".key"
    with open(key_path, "wb") as fh:
        fh.write(os.urandom(32))
    os.chmod(key_path, 0o600)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL_V1)
    conn.executescript(DDL_V2)
    for stmt in DDL_V2_ALTER.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    for stmt in DDL_V2_JOBS_REBUILD.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.executescript(DDL_FTS5)
    conn.executescript(FTS_TRIGGERS)
    for key, value in (
        ("schema_version", 2),
        ("db_id", "db-cli-fixture"),
        ("created_us", now_us()),
        ("projection_generation", 1),
        ("policy_epoch", 0),
    ):
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (key, json_dumps(value)),
        )
    conn.execute(
        "INSERT INTO scopes (scope_id, profile_id, visibility)"
        " VALUES ('s1', 'prof', 'owner')"
    )
    conn.commit()
    conn.close()


def test_cli_migrate_upgrades_v2_store(tmp_path, capsys):
    db = str(tmp_path / "v2.db")
    _make_v2_db(db)

    rc = cli_main(["--json", "migrate", db])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["fts_enabled"] is True
    assert out["path"] == db

    conn = sqlite3.connect(db)
    names = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "vault_entries" in names
    hist = [
        r[0]
        for r in conn.execute("SELECT version FROM migration_history")
    ]
    assert 3 in hist
    conn.close()


def test_cli_migrate_is_noop_on_current(tmp_path, capsys):
    db = str(tmp_path / "cur.db")
    from verbatim.storage.store import Store

    store = Store.create(db)
    store.close()
    capsys.readouterr()  # drain nothing — create prints nothing

    rc = cli_main(["--json", "migrate", db])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == SCHEMA_VERSION


def test_cli_migrate_missing_db_errors(tmp_path, capsys):
    rc = cli_main(["--json", "migrate", str(tmp_path / "nope.db")])
    assert rc == 2  # CONFIG_INVALID -> usage/config exit code
    assert "error" in capsys.readouterr().err
