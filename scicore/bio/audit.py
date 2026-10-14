"""Session-level audit trail for SciCoreAgent.

The low-level kernel runtime already writes per-run traces.  This module adds a
workspace-level evidence chain that is easier to inspect across normal runs,
shortcuts, approvals, and multi-agent jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


SECRET_KEYWORDS = ("api_key", "apikey", "token", "password", "secret", "authorization", "bearer")
SECRET_TEXT_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_.-]{8,}|api[_-]?key\s*[:=]\s*['\"]?[^'\"\s]+|authorization\s*[:=]\s*['\"]?[^'\"\s]+)"
)
MAX_PREVIEW_CHARS = 4000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).resolve() / ".scicore" / "audit"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _clip(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    text = _stringify(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... truncated ({len(text)} chars total)"


def redact(value: Any) -> Any:
    """Conservatively redact secrets while keeping useful audit structure."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(secret in key_text.lower() for secret in SECRET_KEYWORDS):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact(item)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_TEXT_RE.sub("[REDACTED]", value)
    return value


class AuditTrail:
    """Append-only session audit writer.

    Files are intentionally plain JSONL so interrupted runs still leave usable
    evidence.  `finalize()` creates an index with hashes for tamper-evident
    review and a short Markdown report for humans.
    """

    FILE_BY_TYPE = {
        "message": "messages.jsonl",
        "tool_call": "tool_calls.jsonl",
        "approval": "approvals.jsonl",
        "state_transition": "state_transitions.jsonl",
        "artifact": "artifacts.jsonl",
        "error": "errors.jsonl",
    }

    def __init__(
        self,
        workspace_root: str | Path,
        session_id: str,
        *,
        actor: str = "main",
        root: str | Path | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = str(session_id or "unknown")
        self.actor = str(actor or "main")
        self.root = Path(root).resolve() if root else audit_root(self.workspace_root)
        self.session_dir = self.root / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def path_for(self, filename: str) -> Path:
        return self.session_dir / filename

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": "audit_" + hashlib.sha1(f"{time.time_ns()}-{os.getpid()}".encode()).hexdigest()[:12],
            "event_type": event_type,
            "session_id": self.session_id,
            "actor": self.actor,
            "timestamp": utc_now(),
            **dict(redact(payload)),
        }
        with self._lock:
            self._append_jsonl(self.path_for("events.jsonl"), event)
            typed_file = self.FILE_BY_TYPE.get(event_type)
            if typed_file:
                self._append_jsonl(self.path_for(typed_file), event)
        return event

    def log_message(self, role: str, content: Any, *, run_id: str = "", task_id: str = "") -> None:
        text = _stringify(content)
        self.append(
            "message",
            {
                "role": role,
                "run_id": run_id,
                "task_id": task_id,
                "content_preview": _clip(text),
                "content_sha256": sha256_text(text),
            },
        )

    def log_tool_call(
        self,
        *,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any = "",
        metadata: dict[str, Any] | None = None,
        run_id: str = "",
        task_id: str = "",
    ) -> None:
        metadata = dict(metadata or {})
        status = str(metadata.get("tool_status") or "unknown")
        self.append(
            "tool_call",
            {
                "run_id": run_id,
                "task_id": task_id,
                "tool": name,
                "args": args or {},
                "result_preview": _clip(result),
                "result_sha256": sha256_text(_stringify(result)),
                "status": status,
                "risk_level": metadata.get("risk_level", ""),
                "read_only": metadata.get("read_only", ""),
                "affected_paths": metadata.get("affected_paths", []),
                "workspace_changed": metadata.get("workspace_changed", False),
                "diff_summary": metadata.get("diff_summary", []),
                "tool_error_code": metadata.get("tool_error_code", ""),
                "security_event_type": metadata.get("security_event_type", ""),
            },
        )
        if status in {"error", "rejected", "partial_success"} or metadata.get("tool_error_code"):
            self.log_error(
                category=str(metadata.get("tool_error_code") or status),
                source=f"tool:{name}",
                message=_clip(result, 1200),
                run_id=run_id,
                task_id=task_id,
            )

    def log_approval(
        self,
        *,
        action: str,
        args: dict[str, Any] | None = None,
        decision: str,
        reason: str = "",
        risk_level: str = "",
        run_id: str = "",
        task_id: str = "",
    ) -> None:
        self.append(
            "approval",
            {
                "run_id": run_id,
                "task_id": task_id,
                "action": action,
                "args": args or {},
                "decision": decision,
                "reason": reason,
                "risk_level": risk_level,
            },
        )

    def log_state_transition(self, transition: dict[str, Any]) -> None:
        self.append("state_transition", transition)

    def log_artifact(
        self,
        *,
        path: str | Path,
        kind: str,
        description: str = "",
        run_id: str = "",
        task_id: str = "",
        job_id: str = "",
        team_id: str = "",
        source: str = "",
        verified: bool | None = None,
    ) -> None:
        artifact_path = Path(path)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task_id,
            "job_id": job_id,
            "team_id": team_id,
            "kind": kind,
            "path": str(artifact_path),
            "description": description,
            "source": source,
        }
        if verified is not None:
            payload["verified"] = bool(verified)
        try:
            if artifact_path.exists() and artifact_path.is_file():
                payload["sha256"] = sha256_file(artifact_path)
                payload["size_bytes"] = artifact_path.stat().st_size
        except Exception as exc:
            payload["hash_error"] = str(exc)
        self.append("artifact", payload)

    def log_error(
        self,
        *,
        category: str,
        source: str,
        message: Any,
        run_id: str = "",
        task_id: str = "",
        repair_policy: str = "",
    ) -> None:
        self.append(
            "error",
            {
                "run_id": run_id,
                "task_id": task_id,
                "category": category,
                "source": source,
                "message": _clip(message, 1600),
                "repair_policy": repair_policy,
            },
        )

    def log_trace_event(self, event: str, payload: dict[str, Any], *, run_id: str = "", task_id: str = "") -> None:
        if event == "tool_executed":
            self.log_tool_call(
                name=str(payload.get("name", "")),
                args=dict(payload.get("args", {}) or {}),
                result=payload.get("result", ""),
                metadata=payload,
                run_id=run_id,
                task_id=task_id,
            )
        elif event == "run_finished":
            status = str(payload.get("status", ""))
            if status not in {"completed", "success"}:
                self.log_error(
                    category=str(payload.get("stop_reason") or status or "run_stopped"),
                    source="agent_loop",
                    message=payload.get("final_answer", ""),
                    run_id=run_id,
                    task_id=task_id,
                )

    def register_run_artifacts(self, run_dir: str | Path, *, run_id: str = "", task_id: str = "") -> None:
        path = Path(run_dir)
        for name, kind in (
            ("task_state.json", "task_state"),
            ("trace.jsonl", "trace"),
            ("report.json", "run_report"),
            ("evidence_index.json", "evidence_index"),
            ("source_manifest.jsonl", "source_manifest"),
            ("git_before.json", "git_snapshot_before"),
            ("git_after.json", "git_snapshot_after"),
            ("environment.json", "environment"),
        ):
            candidate = path / name
            if candidate.exists():
                self.log_artifact(
                    path=candidate,
                    kind=kind,
                    description=f"kernel {kind} artifact",
                    run_id=run_id,
                    task_id=task_id,
                    source="scicore_kernel_run_store",
                    verified=True,
                )

    def finalize(self, *, final_status: str = "", final_answer: str = "") -> dict[str, Any]:
        index = self.build_index(final_status=final_status, final_answer=final_answer)
        index_path = self.path_for("audit_index.json")
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = self.path_for("final_report.md")
        report_path.write_text(self.render_report(index), encoding="utf-8")
        return index

    def build_index(self, *, final_status: str = "", final_answer: str = "") -> dict[str, Any]:
        files = {}
        counts = {}
        for path in sorted(self.session_dir.glob("*.jsonl")):
            lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            files[path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "events": len(lines),
            }
            counts[path.stem] = len(lines)
        return {
            "schema_version": "scicore.bio.audit.v1",
            "session_id": self.session_id,
            "actor": self.actor,
            "workspace_root": str(self.workspace_root),
            "generated_at": utc_now(),
            "final_status": final_status,
            "final_answer_preview": _clip(final_answer, 1200),
            "final_answer_sha256": sha256_text(_stringify(final_answer)) if final_answer else "",
            "files": files,
            "counts": counts,
        }

    @staticmethod
    def render_report(index: dict[str, Any]) -> str:
        counts = index.get("counts", {})
        lines = [
            "# SciCoreAgent Audit Report",
            "",
            f"- Session: {index.get('session_id', '')}",
            f"- Actor: {index.get('actor', '')}",
            f"- Workspace: {index.get('workspace_root', '')}",
            f"- Generated at: {index.get('generated_at', '')}",
            f"- Final status: {index.get('final_status', '') or '(unknown)'}",
            "",
            "## Evidence Counts",
        ]
        for key in ("messages", "tool_calls", "approvals", "state_transitions", "artifacts", "errors", "events"):
            lines.append(f"- {key}: {counts.get(key, 0)}")
        if index.get("final_answer_preview"):
            lines.extend(["", "## Final Answer Preview", "", str(index.get("final_answer_preview", ""))])
        lines.extend(["", "## Files"])
        for name, meta in sorted((index.get("files") or {}).items()):
            lines.append(f"- {name}: {meta.get('events', 0)} events, sha256={meta.get('sha256', '')}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
