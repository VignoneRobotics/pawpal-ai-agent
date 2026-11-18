"""JSONL trace logging for agent runs (one line per event, grouped by request_id)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = "agent_trace.jsonl"
MAX_RESULT_CHARS = 500


class TraceLogger:
    """Append agent events to logs/agent_trace.jsonl and keep an in-memory copy for the caller."""

    def __init__(self, log_dir: str | Path | None = "logs", request_id: str | None = None) -> None:
        """Start a trace (a new request_id unless one is given); log_dir=None keeps events in memory only."""
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self.path = Path(log_dir) / TRACE_FILE if log_dir is not None else None

    def log(self, event: str, **data: object) -> None:
        """Record one event with a UTC timestamp and write it as a JSON line."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": self.request_id,
            "event": event,
            **data,
        }
        self.events.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Shorten long tool results for the log."""
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


MEDICATION_LOG_FILE = "medication_verification.jsonl"


def log_medication_verification(log_dir: str | Path | None, record: dict) -> dict:
    """Append one label-verification record (never image bytes) to logs/medication_verification.jsonl."""
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    if log_dir is not None:
        path = Path(log_dir) / MEDICATION_LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return record
