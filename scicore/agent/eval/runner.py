"""Controlled task runner for SciCoreAgent evaluations."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import argparse
from pathlib import Path
from typing import Any

from ..audit import default_audit_path
from ..policy import PolicyDecision, evaluate_command
from ..prompt import system_prompt
from ..tools import get_tool
from .grader import grade_verification
from .schemas import EvalTask
from .task import load_task


def run_task(task_path: str | Path, agent: Any | None = None, run_root: str | Path | None = None) -> dict:
    task = load_task(task_path)
    missing = _missing_required_files(task)
    if missing:
        return _result(task, "skipped", "SKIPPED_MISSING_INPUT", missing_inputs=missing)

    blocked_pattern = _first_disallowed_pattern(task)
    if blocked_pattern:
        return _result(task, "failed", f"DISALLOWED_PATTERN_IN_PROMPT: {blocked_pattern}")

    isolated = _create_isolated_workspace(task, run_root)
    audit_log_path = str(default_audit_path(isolated))
    commands: list[dict] = []
    expected_files: list[dict] = []
    agent_response = None
    initial_files = _files_changed(isolated)

    old_cwd = os.getcwd()
    try:
        os.chdir(isolated)
        if agent is None:
            return _result(
                task,
                "failed",
                "NO_AGENT_PROVIDED",
                isolated_workspace=str(isolated),
                audit_log_path=audit_log_path,
            )
        if hasattr(agent, "audit") and hasattr(agent.audit, "path"):
            agent.audit.path = Path(audit_log_path)
        _apply_allowed_tools(agent, task)
        agent_response = agent.chat(task.prompt)
        commands = [_run_verification_command(cmd, isolated) for cmd in task.verification.commands]
        expected_files = [_check_expected_file(item.path, item.must_exist, isolated) for item in task.verification.expected_files]
    finally:
        os.chdir(old_cwd)

    verification = grade_verification(commands, expected_files)
    status = "passed" if verification["passed"] else "failed"
    result = _result(
        task,
        status,
        "verification passed" if verification["passed"] else "verification failed",
        commands=commands,
        expected_files=expected_files,
        files_changed=_new_files(isolated, initial_files),
        audit_log_path=audit_log_path,
        isolated_workspace=str(isolated),
        agent_response=agent_response,
    )
    result["verification"] = verification
    _write_results_json(isolated, result)
    return result


def _missing_required_files(task: EvalTask) -> list[str]:
    workspace = Path(task.workspace.path)
    missing = []
    for item in task.input_contract:
        path = Path(item.path)
        if not path.is_absolute():
            path = workspace / path
        if item.must_exist and not path.exists():
            missing.append(str(path))
    return missing


def _first_disallowed_pattern(task: EvalTask) -> str | None:
    for pattern in task.disallowed_patterns:
        if re.search(pattern, task.prompt):
            return pattern
    return None


def _create_isolated_workspace(task: EvalTask, run_root: str | Path | None) -> Path:
    root = Path(run_root) if run_root else Path(tempfile.mkdtemp(prefix="scicore_bio_eval_"))
    run_id = f"{task.id}_{time.strftime('%Y%m%d_%H%M%S')}"
    isolated = root / run_id
    isolated.mkdir(parents=True, exist_ok=False)

    source_workspace = Path(task.workspace.path)
    for required in task.input_contract:
        source = Path(required.path)
        if not source.is_absolute():
            source = source_workspace / source
        relative = Path(required.path)
        destination = isolated / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return isolated


def _run_verification_command(command: str, cwd: Path) -> dict:
    decision, reason = evaluate_command(command)
    if decision != PolicyDecision.ALLOW:
        return {
            "command": command,
            "returncode": 126,
            "stdout": "",
            "stderr": f"{decision}: {reason}",
            "policy_decision": decision,
        }
    proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": _truncate(proc.stdout),
        "stderr": _truncate(proc.stderr),
        "policy_decision": str(PolicyDecision.ALLOW),
    }


def _check_expected_file(path: str, must_exist: bool, cwd: Path) -> dict:
    target = Path(path)
    if not target.is_absolute():
        target = cwd / target
    return {"path": str(target), "must_exist": must_exist, "exists": target.exists()}


def _apply_allowed_tools(agent: Any, task: EvalTask) -> None:
    if not task.allowed_tools or not hasattr(agent, "tools"):
        return
    allowed = []
    for name in task.allowed_tools:
        tool = get_tool(name)
        if tool is not None:
            allowed.append(tool)
    agent.tools = allowed
    if hasattr(agent, "_system"):
        agent._system = system_prompt(agent.tools)


def _files_changed(workspace: Path) -> list[str]:
    return sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file())


def _new_files(workspace: Path, initial_files: list[str]) -> list[str]:
    initial = set(initial_files)
    return [path for path in _files_changed(workspace) if path not in initial]


def _result(task: EvalTask, status: str, reason: str, **extra) -> dict:
    result = {
        "task_id": task.id,
        "status": status,
        "reason": reason,
        "commands": extra.pop("commands", []),
        "files_changed": extra.pop("files_changed", []),
        "audit_log_path": extra.pop("audit_log_path", ""),
        "verification": extra.pop("verification", {"passed": status == "passed", "details": []}),
    }
    result.update(extra)
    return result


def _write_results_json(workspace: Path, result: dict) -> None:
    out = workspace / "results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... truncated ({len(text)} chars total)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SciCoreAgent eval task file.")
    parser.add_argument("task", help="Path to task.yaml or task.json")
    parser.add_argument("--run-root", help="Directory for isolated eval workspaces")
    args = parser.parse_args()
    result = run_task(args.task, agent=None, run_root=args.run_root)
    print(json.dumps(result, indent=2))
