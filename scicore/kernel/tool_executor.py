"""Structured tool execution for the agent runtime."""

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum

from .workspace import clip


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    metadata: dict


class ToolStatus(str, Enum):
    OK = "ok"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    REJECTED = "rejected"


def _metadata(
    tool_status,
    tool_error_code="",
    security_event_type="",
    risk_level="low",
    read_only=True,
    affected_paths=None,
    workspace_changed=False,
    workspace_fingerprint="",
    diff_summary=None,
    execution_id="",
    duration_ms=0,
):
    result = {
        "tool_status": tool_status,
        "tool_error_code": tool_error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level,
        "read_only": read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
        "execution_id": execution_id,
        "duration_ms": max(0, int(duration_ms)),
    }
    if workspace_fingerprint:
        result["workspace_fingerprint"] = workspace_fingerprint
    return result


class ToolManager:
    """Single execution gateway for every model-visible runtime tool.

    Registries may still be assembled by a domain runtime, but lookup, policy,
    validation, approval, execution classification, cancellation and evidence
    metadata all pass through this manager.
    """

    def __init__(self, agent):
        self.agent = agent
        self._cancel_requested = threading.Event()
        self._execution_lock = threading.RLock()

    @property
    def registry(self):
        return self.agent.tools

    def begin_run(self):
        self._cancel_requested.clear()

    def cancel(self):
        self._cancel_requested.set()

    @property
    def cancel_requested(self):
        return self._cancel_requested.is_set()

    def execute(self, name, args):
        execution_id = "tool_" + uuid.uuid4().hex[:12]
        started_at = time.monotonic()
        with self._execution_lock:
            return self._execute(
                name,
                args,
                execution_id=execution_id,
                started_at=started_at,
            )

    def _execute(self, name, args, *, execution_id, started_at):
        agent = self.agent
        args = _normalize_model_tool_args(args)
        if self.cancel_requested:
            return ToolExecutionResult(
                content=f"error: tool '{name}' was cancelled before execution",
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="tool_cancelled",
                    risk_level="high",
                    read_only=False,
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )
        if agent.allowed_tools is not None and name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )

        tool = agent.tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )

        try:
            agent.validate_tool(name, args)
        except Exception as exc:
            example = agent.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            return ToolExecutionResult(
                content=message,
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="invalid_arguments",
                    security_event_type=security_event_type,
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )

        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )

        if tool["risky"] and not agent.approve(name, args):
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    ToolStatus.REJECTED.value,
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block" if agent.read_only else "approval_denied",
                    risk_level="high",
                    read_only=False,
                    execution_id=execution_id,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                ),
            )

        before_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            content = clip(tool["run"](args))
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = ToolStatus.OK.value
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", content)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = ToolStatus.PARTIAL_SUCCESS.value
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = ToolStatus.ERROR.value
                    tool_error_code = "tool_failed"
            agent.update_memory_after_tool(name, args, content)
            metadata = _metadata(
                tool_status,
                tool_error_code=tool_error_code,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                execution_id=execution_id,
                duration_ms=(time.monotonic() - started_at) * 1000,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            metadata = _metadata(
                ToolStatus.PARTIAL_SUCCESS.value if workspace_changed else ToolStatus.ERROR.value,
                tool_error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                execution_id=execution_id,
                duration_ms=(time.monotonic() - started_at) * 1000,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)


# Backwards-compatible import for extensions built against the earlier name.
ToolExecutor = ToolManager


def _normalize_model_tool_args(args):
    if not isinstance(args, dict):
        return args
    nested = args.get("args")
    if len(args) == 1 and isinstance(nested, dict):
        return nested
    if len(args) == 1 and isinstance(nested, str):
        try:
            parsed = json.loads(nested)
        except json.JSONDecodeError:
            return args
        if isinstance(parsed, dict):
            return parsed
    return args
