"""SQLite checkpointer: every run is persisted so it can pause for approval and resume later."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .graph import Run, Step


class SQLiteCheckpointer:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY, status TEXT, next_node TEXT, state TEXT, trace TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    def save(self, run: Run) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id,status,next_node,state,trace,updated_at) "
            "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            (run.run_id, run.status, run.next_node, json.dumps(run.state, default=str),
             json.dumps([asdict(s) for s in run.trace])))
        self.conn.commit()

    def load(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT run_id,status,next_node,state,trace FROM runs WHERE run_id=?",
                                (run_id,)).fetchone()
        if not row:
            return None
        return Run(row[0], json.loads(row[3]), row[2], row[1], [Step(**s) for s in json.loads(row[4])])

    def list(self) -> list[dict]:
        rows = self.conn.execute("SELECT run_id,status,state,updated_at FROM runs ORDER BY updated_at DESC").fetchall()
        return [{"run_id": r[0], "status": r[1], "state": json.loads(r[2]), "updated_at": r[3]} for r in rows]
