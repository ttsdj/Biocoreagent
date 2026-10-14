"""Shell command execution with safety checks.

Features:
- Output capture with truncation (head+tail preserved)
- Timeout support
- Dangerous command detection
- Working directory tracking (cd awareness)
"""

import os
import re
import subprocess
from .base import Tool
from ..audit import AuditLogger
from ..events import EventType
from ..policy import PolicyDecision, evaluate_command

# track cwd across commands
_cwd: str | None = None

# patterns that could wreck the filesystem or leak secrets
_DANGEROUS_PATTERNS = [
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    (r"\brm\s+(-\w*)?-rf\s", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?bash", "pipe curl to bash"),
    (r"\bwget\b.*\|\s*(sudo\s+)?bash", "pipe wget to bash"),
]


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 120) -> str:
        global _cwd
        audit = AuditLogger()

        decision, reason = evaluate_command(command)
        if decision == PolicyDecision.BLOCK:
            audit.log(
                EventType.POLICY_BLOCK,
                tool_name=self.name,
                arguments={"command": command, "decision": decision, "reason": reason},
                success=False,
                error=reason,
            )
            return f"Blocked by policy: {reason}\nCommand: {command}"
        if decision == PolicyDecision.CONFIRM_REQUIRED:
            audit.log(
                EventType.POLICY_BLOCK,
                tool_name=self.name,
                arguments={"command": command, "decision": decision, "reason": reason},
                success=False,
                error=reason,
            )
            return (
                f"Confirmation required by policy: {reason}\n"
                "This first version cannot ask interactively inside tool execution; rerun only after explicit configuration.\n"
                f"Command: {command}"
            )

        # safety check
        warning = _check_dangerous(command)
        if warning:
            audit.log(
                EventType.POLICY_BLOCK,
                tool_name=self.name,
                arguments={"command": command, "decision": "legacy_block", "reason": warning},
                success=False,
                error=warning,
            )
            return f"⚠ Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # use tracked working directory
        cwd = _cwd or os.getcwd()

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            # track cd commands so next command runs in the right place
            if proc.returncode == 0:
                _update_cwd(command, cwd)
            out = proc.stdout
            if proc.stderr:
                out += f"\n[stderr]\n{proc.stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            # keep head + tail to preserve the most useful info
            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            audit.log(
                EventType.COMMAND_RUN,
                tool_name=self.name,
                arguments={"command": command, "timeout": timeout},
                result_preview=out,
                success=proc.returncode == 0,
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            )
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            audit.log(
                EventType.COMMAND_RUN,
                tool_name=self.name,
                arguments={"command": command, "timeout": timeout},
                success=False,
                error=f"timed out after {timeout}s",
            )
            return f"Error: timed out after {timeout}s"
        except Exception as e:
            audit.log(
                EventType.COMMAND_RUN,
                tool_name=self.name,
                arguments={"command": command, "timeout": timeout},
                success=False,
                error=str(e),
            )
            return f"Error running command: {e}"


def _check_dangerous(cmd: str) -> str | None:
    """Return a warning string if the command looks destructive, else None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    """Track directory changes from cd commands."""
    global _cwd
    # simple heuristic: look for cd at the end of a && chain or standalone
    parts = command.split("&&")
    for part in parts:
        part = part.strip()
        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
            if target:
                new_dir = os.path.normpath(os.path.join(current_cwd, os.path.expanduser(target)))
                if os.path.isdir(new_dir):
                    _cwd = new_dir
