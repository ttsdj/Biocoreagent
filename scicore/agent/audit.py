"""Small JSONL audit logger for agent actions and provenance."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import EventType

SECRET_KEYS = ("api_key", "token", "password", "secret", "authorization")


def default_audit_path(cwd: str | None = None) -> Path:
    root = Path(cwd or os.getcwd())
    return root / ".scicore" / "audit" / "events.jsonl"


def _preview(value: Any, limit: int = 2000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[:limit] + f"... truncated ({len(text)} chars total)"
    return text


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(secret in key.lower() for secret in SECRET_KEYS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class AuditLogger:
    """Append-only JSONL logger with conservative content previews."""

    def __init__(self, path: str | Path | None = None, session_id: str | None = None):
        self.path = Path(path) if path else default_audit_path()
        self.session_id = session_id or str(uuid.uuid4())

    def log(
        self,
        event_type: EventType | str,
        *,
        tool_name: str | None = None,
        arguments: dict | None = None,
        result_preview: Any = None,
        files: list[str] | None = None,
        cwd: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type),
            "session_id": self.session_id,
            "tool_name": tool_name,
            "arguments": _redact(arguments or {}),
            "result_preview": _preview(result_preview or ""),
            "files": files or [],
            "cwd": cwd or os.getcwd(),
            "success": success,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def tail_events(path: str | Path | None = None, limit: int = 5) -> list[dict]:
    audit_path = Path(path) if path else default_audit_path()
    if not audit_path.exists():
        return []
    lines = audit_path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[dict] = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event_type": "ERROR", "result_preview": line})
    return events
