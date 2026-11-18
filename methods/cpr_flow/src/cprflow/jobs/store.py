import json
import sqlite3
import time
from pathlib import Path

from cprflow.data.cache import digest


class Store:
    def __init__(self, plan):
        root = Path(plan["state_dir"])
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "state.sqlite", timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,state TEXT,attempt INTEGER DEFAULT 0,failures INTEGER DEFAULT 0,pid INTEGER,pid_start TEXT,reason TEXT,log TEXT,result TEXT,next_retry REAL DEFAULT 0)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY,time REAL,job TEXT,state TEXT,details TEXT)"
        )
        expected = digest(plan)
        old = self.db.execute("SELECT value FROM meta WHERE key='plan_hash'").fetchone()
        if old and old[0] != expected:
            raise ValueError("Manifest changed after submission; create a new campaign instead")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('plan_hash',?)", (expected,))
        for job in plan["jobs"]:
            self.db.execute(
                "INSERT OR IGNORE INTO jobs(id,state) VALUES (?,'pending')", (job["id"],)
            )
        self.hash = expected

    def rows(self):
        self.db.row_factory = sqlite3.Row
        return {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM jobs")}

    def set(self, name, state, **values):
        if set(values) - {
            "attempt",
            "failures",
            "pid",
            "pid_start",
            "reason",
            "log",
            "result",
            "next_retry",
        }:
            raise ValueError("Unknown state fields")
        old = self.rows()[name]
        if old["state"] == state and all(old.get(k) == v for k, v in values.items()):
            return
        values = {"state": state, **values}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?",
                (*values.values(), name),
            )
            self.db.execute(
                "INSERT INTO events(time,job,state,details) VALUES (?,?,?,?)",
                (time.time(), name, state, json.dumps(values)),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
