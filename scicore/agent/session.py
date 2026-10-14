"""Session persistence - save and resume conversations.

Session state is a JSON dump of messages + model config.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path.home() / ".scicore_agent" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _sessions_dir() -> Path:
    home = os.environ.get("SCICORE_AGENT_HOME")
    if home:
        return Path(home).expanduser() / "sessions"
    return SESSIONS_DIR


def _session_path(session_id: str) -> Path:
    sessions_dir = _sessions_dir()
    path = (sessions_dir / f"{_normalize_session_id(session_id)}.json").resolve()
    root = sessions_dir.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """Save conversation to disk. Returns the session ID."""
    _sessions_dir().mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    data = {
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """Load a saved session. Returns (messages, model) or None."""
    path = _session_path(session_id)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    return data["messages"], data["model"]


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    sessions_dir = _sessions_dir()
    if not sessions_dir.exists():
        return []

    sessions = []
    for f in sorted(sessions_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            # grab first user message as preview
            preview = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions[:20]  # cap at 20
