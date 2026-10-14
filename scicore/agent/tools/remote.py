"""SSH remote execution tool with SciCoreAgent policy checks."""

from __future__ import annotations

import json
import subprocess

from .base import Tool
from ..audit import AuditLogger
from ..events import EventType
from ..policy import PolicyDecision, evaluate_command, is_remote_path_allowed
from ..remote import RemoteConfig, build_ssh_command, shell_cd_and_run


class SSHBashTool(Tool):
    name = "ssh_bash"
    description = (
        "Execute a command on the configured remote SSH server inside BIO_REMOTE_WORK_DIR. "
        "Requires BIO_REMOTE_HOST, BIO_REMOTE_USER, BIO_REMOTE_WORK_DIR, and optional BIO_REMOTE_KEY."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Remote shell command to run."},
            "cwd": {
                "type": "string",
                "description": "Absolute remote working directory. Must be inside BIO_REMOTE_WORK_DIR.",
            },
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default 3600."},
        },
        "required": ["command"],
    }

    def execute(self, command: str, cwd: str | None = None, timeout: int = 3600) -> str:
        audit = AuditLogger()
        config = RemoteConfig.from_env()
        error = config.validate()
        if error:
            return f"Error: {error}"

        remote_cwd = cwd or config.work_dir
        if not is_remote_path_allowed(remote_cwd, config.work_dir):
            audit.log(
                EventType.POLICY_BLOCK,
                tool_name=self.name,
                arguments={"command": command, "cwd": remote_cwd, "remote_work_dir": config.work_dir},
                success=False,
                error="remote cwd outside BIO_REMOTE_WORK_DIR",
            )
            return "Blocked by remote sandbox: cwd must be inside BIO_REMOTE_WORK_DIR"

        decision, reason = evaluate_command(command)
        if decision != PolicyDecision.ALLOW:
            audit.log(
                EventType.POLICY_BLOCK,
                tool_name=self.name,
                arguments={"command": command, "cwd": remote_cwd, "decision": decision, "reason": reason},
                success=False,
                error=reason,
            )
            return f"Blocked by policy: {reason}\nCommand: {command}"

        remote_command = shell_cd_and_run(remote_cwd, command)
        ssh_cmd = build_ssh_command(config, remote_command)
        try:
            proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
            output = {
                "host": config.host,
                "user": config.user,
                "cwd": remote_cwd,
                "returncode": proc.returncode,
                "stdout": _truncate(proc.stdout),
                "stderr": _truncate(proc.stderr),
            }
            audit.log(
                EventType.COMMAND_RUN,
                tool_name=self.name,
                arguments={"host": config.host, "user": config.user, "cwd": remote_cwd, "command": command},
                result_preview=output,
                success=proc.returncode == 0,
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            )
            return json.dumps(output, indent=2)
        except subprocess.TimeoutExpired:
            audit.log(
                EventType.COMMAND_RUN,
                tool_name=self.name,
                arguments={"host": config.host, "user": config.user, "cwd": remote_cwd, "command": command},
                success=False,
                error=f"timed out after {timeout}s",
            )
            return f"Error: remote command timed out after {timeout}s"
        except Exception as e:
            audit.log(
                EventType.ERROR,
                tool_name=self.name,
                arguments={"host": config.host, "user": config.user, "cwd": remote_cwd, "command": command},
                success=False,
                error=str(e),
            )
            return f"Error running remote command: {e}"


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:8000] + f"\n... truncated ({len(text)} chars total) ...\n" + text[-2000:]
