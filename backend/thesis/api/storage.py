from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .payload import strip_payload_metadata

RunStatus = Literal["success", "error"]

DEFAULT_RUNS_DB_PATH = (Path(__file__).resolve().parents[2] / "data" / "scenario_runs.sqlite3").resolve()

__all__ = [
    "DEFAULT_RUNS_DB_PATH",
    "RunStatus",
    "ScenarioRunStore",
    "StoredRunRecord",
]


@dataclass(frozen=True, slots=True)
class StoredRunRecord:
    id: str
    created_at: str
    status: RunStatus
    input_mode_hint: str
    resolved_input_mode: str | None
    submitted_kind: str
    input_preview: str
    model_name: str | None
    deterministic_best_action: str | None
    dominant_framework: str | None
    rag_runtime_available: bool
    reasoning_runtime_available: bool
    error_code: str | None
    error_message: str | None
    replay_stage_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "input_mode_hint": self.input_mode_hint,
            "resolved_input_mode": self.resolved_input_mode,
            "submitted_kind": self.submitted_kind,
            "input_preview": self.input_preview,
            "model_name": self.model_name,
            "deterministic_best_action": self.deterministic_best_action,
            "dominant_framework": self.dominant_framework,
            "rag_runtime_available": self.rag_runtime_available,
            "reasoning_runtime_available": self.reasoning_runtime_available,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "replay_stage_count": self.replay_stage_count,
        }


class ScenarioRunStore:
    def __init__(self, database_path: Path | str | None = None) -> None:
        self.database_path = Path(database_path or DEFAULT_RUNS_DB_PATH).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_mode_hint TEXT NOT NULL,
                    resolved_input_mode TEXT,
                    submitted_kind TEXT NOT NULL,
                    input_preview TEXT NOT NULL,
                    model_name TEXT,
                    deterministic_best_action TEXT,
                    dominant_framework TEXT,
                    rag_runtime_available INTEGER NOT NULL DEFAULT 0,
                    reasoning_runtime_available INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    replay_stage_count INTEGER NOT NULL DEFAULT 0,
                    input_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scenario_runs_created_at
                ON scenario_runs(created_at DESC)
                """
            )

    def save_run(
        self,
        *,
        request_input: str | dict[str, Any],
        input_mode_hint: str,
        payload: dict[str, Any],
        status: RunStatus,
        model_name: str | None,
    ) -> StoredRunRecord:
        stored_input = strip_payload_metadata(request_input)
        summary = payload.get("summary", {})
        error = payload.get("error", {})
        replay = payload.get("replay", [])

        record = StoredRunRecord(
            id=str(uuid4()),
            created_at=_utc_now(),
            status=status,
            input_mode_hint=input_mode_hint,
            resolved_input_mode=summary.get("input_mode"),
            submitted_kind="json" if isinstance(request_input, dict) else "text",
            input_preview=_build_input_preview(stored_input),
            model_name=model_name,
            deterministic_best_action=summary.get("deterministic_best_action"),
            dominant_framework=summary.get("dominant_framework"),
            rag_runtime_available=bool(summary.get("rag_runtime_available", False)),
            reasoning_runtime_available=bool(summary.get("reasoning_runtime_available", False)),
            error_code=error.get("code"),
            error_message=error.get("message"),
            replay_stage_count=len(replay) if isinstance(replay, list) else 0,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scenario_runs (
                    id,
                    created_at,
                    status,
                    input_mode_hint,
                    resolved_input_mode,
                    submitted_kind,
                    input_preview,
                    model_name,
                    deterministic_best_action,
                    dominant_framework,
                    rag_runtime_available,
                    reasoning_runtime_available,
                    error_code,
                    error_message,
                    replay_stage_count,
                    input_json,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.created_at,
                    record.status,
                    record.input_mode_hint,
                    record.resolved_input_mode,
                    record.submitted_kind,
                    record.input_preview,
                    record.model_name,
                    record.deterministic_best_action,
                    record.dominant_framework,
                    int(record.rag_runtime_available),
                    int(record.reasoning_runtime_available),
                    record.error_code,
                    record.error_message,
                    record.replay_stage_count,
                    _dump_json(stored_input),
                    _dump_json(payload),
                ),
            )

        return record

    def list_runs(self, *, limit: int = 25) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    status,
                    input_mode_hint,
                    resolved_input_mode,
                    submitted_kind,
                    input_preview,
                    model_name,
                    deterministic_best_action,
                    dominant_framework,
                    rag_runtime_available,
                    reasoning_runtime_available,
                    error_code,
                    error_message,
                    replay_stage_count
                FROM scenario_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_runs,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS failed_runs
                FROM scenario_runs
                """
            ).fetchone()

        return {
            "runs": [_record_from_row(row).to_dict() for row in rows],
            "total_runs": int(totals["total_runs"] or 0),
            "success_runs": int(totals["success_runs"] or 0),
            "failed_runs": int(totals["failed_runs"] or 0),
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    status,
                    input_mode_hint,
                    resolved_input_mode,
                    submitted_kind,
                    input_preview,
                    model_name,
                    deterministic_best_action,
                    dominant_framework,
                    rag_runtime_available,
                    reasoning_runtime_available,
                    error_code,
                    error_message,
                    replay_stage_count,
                    input_json,
                    payload_json
                FROM scenario_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(row["payload_json"])
        return {
            "run": _record_from_row(row).to_dict(),
            "input": json.loads(row["input_json"]),
            **payload,
        }


def _record_from_row(row: sqlite3.Row) -> StoredRunRecord:
    status = str(row["status"])
    return StoredRunRecord(
        id=str(row["id"]),
        created_at=str(row["created_at"]),
        status="error" if status == "error" else "success",
        input_mode_hint=str(row["input_mode_hint"]),
        resolved_input_mode=str(row["resolved_input_mode"]) if row["resolved_input_mode"] is not None else None,
        submitted_kind=str(row["submitted_kind"]),
        input_preview=str(row["input_preview"]),
        model_name=str(row["model_name"]) if row["model_name"] is not None else None,
        deterministic_best_action=(
            str(row["deterministic_best_action"]) if row["deterministic_best_action"] is not None else None
        ),
        dominant_framework=str(row["dominant_framework"]) if row["dominant_framework"] is not None else None,
        rag_runtime_available=bool(row["rag_runtime_available"]),
        reasoning_runtime_available=bool(row["reasoning_runtime_available"]),
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        error_message=str(row["error_message"]) if row["error_message"] is not None else None,
        replay_stage_count=int(row["replay_stage_count"]),
    )


def _build_input_preview(request_input: str | dict[str, Any]) -> str:
    if isinstance(request_input, dict):
        serialized = json.dumps(request_input, ensure_ascii=True, separators=(",", ":"))
        return _truncate(serialized, 180)

    return _truncate(" ".join(request_input.split()), 180)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
