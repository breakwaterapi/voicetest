"""
store.py — BlobStore (disk) + Store (SQLite) for voicetest.

BlobStore: persists call artifacts (report.json, recording.wav) to local disk.
Store: SQLite-backed persistence for runs, calls, and reports.

Tables
------
runs    — one row per test run (N calls against a scenario+number)
calls   — one row per individual call within a run
reports — one row per scored report (1-to-1 with a completed call)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voicetest.models import CallOutcome, CallResult, Event, Report, Turn

if TYPE_CHECKING:
    from voicetest.agent_model.models import AgentModel

# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj: object) -> object:
    """Recursively convert dataclasses / enums to JSON-safe primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_json_safe(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, CallOutcome):
        return str(obj)
    return obj


def call_result_to_dict(result: CallResult) -> dict:
    return _to_json_safe(result)  # type: ignore[return-value]


def transcript_hash(transcript: str) -> str:
    """Content hash used to dedup production calls on ingest."""
    return hashlib.sha256(transcript.strip().encode("utf-8")).hexdigest()


def call_result_from_dict(data: dict) -> CallResult:
    turns = [Turn(**t) for t in data.get("turns", [])]
    events = [Event(**e) for e in data.get("events", [])]
    return CallResult(
        scenario_id=data["scenario_id"],
        outcome=CallOutcome(data["outcome"]),
        turns=turns,
        events=events,
        recording_path=data.get("recording_path"),
        duration_ms=data["duration_ms"],
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# BlobStore
# ---------------------------------------------------------------------------


class BlobStore:
    """
    Writes call artifacts to a per-call subdirectory under root_dir.

    Layout::
        <root_dir>/
            <scenario_id>-<call_index>/
                report.json
                recording.wav   ← written by recorder (P0-T5)
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is None:
            from voicetest.config import Settings
            root_dir = Settings().voicetest_blob_dir
        self.root_dir = Path(root_dir)

    def call_dir(self, scenario_id: str, call_index: int = 0) -> Path:
        name = f"{scenario_id}-{call_index:04d}"
        d = self.root_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_result(self, result: CallResult, call_index: int = 0) -> Path:
        """Serialise CallResult to JSON; return the path written."""
        d = self.call_dir(result.scenario_id, call_index)
        path = d / "report.json"
        path.write_text(json.dumps(call_result_to_dict(result), indent=2))
        return path

    def read_result(self, path: Path | str) -> CallResult:
        data = json.loads(Path(path).read_text())
        return call_result_from_dict(data)

    def recording_path(self, scenario_id: str, call_index: int = 0) -> Path:
        """Return the expected WAV path (may not exist yet — written by recorder)."""
        return self.call_dir(scenario_id, call_index) / "recording.wav"


# ---------------------------------------------------------------------------
# Run status enum
# ---------------------------------------------------------------------------


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Store — SQLite persistence
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS batteries (
    id          TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    number      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    total       INTEGER NOT NULL DEFAULT 0,
    completed   INTEGER NOT NULL DEFAULT 0,
    report_json TEXT,
    agent_model_id TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    battery_id  TEXT,
    scenario_id TEXT NOT NULL,
    scenario_group TEXT NOT NULL DEFAULT '',
    scenario_title TEXT NOT NULL DEFAULT '',
    number      TEXT NOT NULL,
    n           INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    completed   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calls (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    call_index  INTEGER NOT NULL,
    outcome     TEXT,
    duration_ms INTEGER,
    error       TEXT,
    result_json TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
    id          TEXT PRIMARY KEY,
    call_id     TEXT NOT NULL REFERENCES calls(id),
    run_id      TEXT NOT NULL REFERENCES runs(id),
    passed      INTEGER NOT NULL,
    dimensions  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_models (
    id          TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    confirmed   INTEGER NOT NULL DEFAULT 0,
    model_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS production_calls (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    transcript      TEXT NOT NULL,
    transcript_hash TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    error           TEXT,
    root_cause      TEXT NOT NULL DEFAULT '',
    detected_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(agent_id, transcript_hash)
);
"""


class Store:
    """
    SQLite-backed store for runs, calls, and reports.

    All methods are synchronous — call from a thread or via
    asyncio.to_thread() when used from async code.
    """

    def __init__(self, db_url: str | None = None) -> None:
        if db_url is None:
            from voicetest.config import Settings
            db_url = Settings().voicetest_db_url
        # Strip "sqlite:///" prefix if present
        db_path = db_url.removeprefix("sqlite:///")
        self._path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema (idempotent)."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
        for name, ddl in (
            ("battery_id", "ALTER TABLE runs ADD COLUMN battery_id TEXT"),
            ("scenario_group", "ALTER TABLE runs ADD COLUMN scenario_group TEXT NOT NULL DEFAULT ''"),
            ("scenario_title", "ALTER TABLE runs ADD COLUMN scenario_title TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in cols:
                conn.execute(ddl)

        battery_cols = {r["name"] for r in conn.execute("PRAGMA table_info(batteries)").fetchall()}
        if "agent_model_id" not in battery_cols:
            conn.execute("ALTER TABLE batteries ADD COLUMN agent_model_id TEXT")

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        scenario_id: str,
        number: str,
        n: int,
        *,
        battery_id: str | None = None,
        scenario_group: str = "",
        scenario_title: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs"
                " (id, battery_id, scenario_id, scenario_group, scenario_title,"
                "  number, n, status, completed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    run_id,
                    battery_id,
                    scenario_id,
                    scenario_group,
                    scenario_title,
                    number,
                    n,
                    RunStatus.PENDING,
                ),
            )

    def update_run_status(self, run_id: str, status: RunStatus, completed: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, completed=?, updated_at=datetime('now') WHERE id=?",
                (status, completed, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def create_call(self, call_id: str, run_id: str, call_index: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO calls (id, run_id, call_index) VALUES (?, ?, ?)",
                (call_id, run_id, call_index),
            )

    def update_call(self, call_id: str, result: CallResult) -> None:
        result_json = json.dumps(call_result_to_dict(result))
        with self._connect() as conn:
            conn.execute(
                "UPDATE calls SET outcome=?, duration_ms=?, error=?, result_json=? WHERE id=?",
                (result.outcome, result.duration_ms, result.error, result_json, call_id),
            )

    def get_calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calls WHERE run_id=? ORDER BY call_index",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def save_report(self, report_id: str, call_id: str, run_id: str, report: Report) -> None:
        dimensions_json = json.dumps(report.dimensions)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reports (id, call_id, run_id, passed, dimensions, summary)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (report_id, call_id, run_id, int(report.passed), dimensions_json, report.summary),
            )

    def get_reports_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT r.*, c.call_index, c.outcome, c.duration_ms"
                " FROM reports r JOIN calls c ON r.call_id = c.id"
                " WHERE r.run_id=? ORDER BY c.call_index",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_report_for_call(self, call_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE call_id=?", (call_id,)
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Batteries
    # ------------------------------------------------------------------

    def create_battery(
        self,
        battery_id: str,
        description: str,
        number: str,
        total: int,
        *,
        agent_model_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batteries"
                " (id, description, number, total, status, completed, agent_model_id)"
                " VALUES (?, ?, ?, ?, ?, 0, ?)",
                (battery_id, description, number, total, RunStatus.PENDING, agent_model_id),
            )

    def update_battery_status(
        self,
        battery_id: str,
        status: RunStatus,
        completed: int,
        report_json: str | None = None,
    ) -> None:
        with self._connect() as conn:
            if report_json is not None:
                conn.execute(
                    "UPDATE batteries SET status=?, completed=?, report_json=?,"
                    " updated_at=datetime('now') WHERE id=?",
                    (status, completed, report_json, battery_id),
                )
            else:
                conn.execute(
                    "UPDATE batteries SET status=?, completed=?,"
                    " updated_at=datetime('now') WHERE id=?",
                    (status, completed, battery_id),
                )

    def increment_battery_completed(self, battery_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE batteries SET completed=completed+1,"
                " updated_at=datetime('now') WHERE id=?",
                (battery_id,),
            )

    def get_battery(self, battery_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batteries WHERE id=?", (battery_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_runs_for_battery(self, battery_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE battery_id=? ORDER BY created_at",
                (battery_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_reports_for_battery(self, battery_id: str) -> list[dict[str, Any]]:
        """One report row per call across every run in the battery, with run meta."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rp.*, c.call_index, c.outcome, c.duration_ms,"
                " ru.scenario_id, ru.scenario_group, ru.scenario_title"
                " FROM reports rp"
                " JOIN calls c ON rp.call_id = c.id"
                " JOIN runs ru ON rp.run_id = ru.id"
                " WHERE ru.battery_id=? ORDER BY ru.scenario_group, ru.created_at",
                (battery_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent models
    # ------------------------------------------------------------------

    def save_agent_model(self, model: "AgentModel") -> None:
        """Insert or update an AgentModel (upsert on id)."""
        model_json = model.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_models (id, description, confirmed, model_json)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   description=excluded.description,"
                "   confirmed=excluded.confirmed,"
                "   model_json=excluded.model_json,"
                "   updated_at=datetime('now')",
                (
                    model.id,
                    model.source_description,
                    int(model.confirmed),
                    model_json,
                ),
            )

    def get_agent_model(self, model_id: str) -> "AgentModel | None":
        from voicetest.agent_model.models import AgentModel

        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_json FROM agent_models WHERE id=?", (model_id,)
            ).fetchone()
        if row is None:
            return None
        return AgentModel.model_validate_json(row["model_json"])

    # ------------------------------------------------------------------
    # Production calls
    # ------------------------------------------------------------------

    def production_call_exists(self, agent_id: str, transcript: str) -> bool:
        """Dedup check — has a call with this transcript already been ingested?"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM production_calls WHERE agent_id=? AND transcript_hash=?",
                (agent_id, transcript_hash(transcript)),
            ).fetchone()
        return row is not None

    def save_production_call(
        self,
        call_id: str,
        agent_id: str,
        transcript: str,
        outcome: str,
        error: str | None,
        *,
        root_cause: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO production_calls"
                " (id, agent_id, transcript, transcript_hash, outcome, error, root_cause)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(agent_id, transcript_hash) DO NOTHING",
                (
                    call_id,
                    agent_id,
                    transcript,
                    transcript_hash(transcript),
                    outcome,
                    error,
                    root_cause,
                ),
            )

    def get_production_failures(self, agent_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Failures only (outcome != 'completed' or error set), most recent first."""
        with self._connect() as conn:
            if agent_id:
                rows = conn.execute(
                    "SELECT * FROM production_calls"
                    " WHERE agent_id=? AND (outcome != 'completed' OR error IS NOT NULL)"
                    " ORDER BY detected_at DESC, rowid DESC LIMIT ?",
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM production_calls"
                    " WHERE outcome != 'completed' OR error IS NOT NULL"
                    " ORDER BY detected_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_production_call(self, call_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM production_calls WHERE id=?", (call_id,)
            ).fetchone()
        return dict(row) if row else None
