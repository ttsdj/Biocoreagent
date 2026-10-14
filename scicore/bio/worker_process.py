"""Killable subprocess entry point for multi-agent workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

from scicore.kernel.cli import _build_model_client, _configured_secret_names
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext

from .cli import load_scicore_bio_env
from .domain import ensure_workspace_state
from .runtime import BioRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal isolated SciCoreAgent worker.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--job-workspace", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--config-file", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    job_workspace = Path(args.job_workspace).resolve()
    result_path = job_workspace / "worker_process_result.json"
    try:
        config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
        task = Path(args.task_file).read_text(encoding="utf-8")
        load_scicore_bio_env(workspace_root)
        from scicore.agent.mcp_client import start_configured_mcp_servers

        start_configured_mcp_servers(workspace_root)
        runtime_args = Namespace(**config)
        state_root = ensure_workspace_state(workspace_root)
        agent = BioRuntime(
            model_client=_build_model_client(runtime_args),
            workspace=WorkspaceContext.build(workspace_root),
            session_store=SessionStore(state_root / "sessions"),
            run_store=RunStore(state_root / "runs"),
            approval_policy="auto",
            max_steps=int(config.get("max_steps", 20)),
            max_new_tokens=int(config.get("max_new_tokens", 4096)),
            secret_env_names=_configured_secret_names(runtime_args),
            role=args.role,
            orchestrator=None,
            allow_orchestration=False,
        )
        progress_path = job_workspace / "worker_progress.json"
        previous_emit_trace = agent.emit_trace

        def worker_emit_trace(task_state, event, payload=None):
            progress_temp = progress_path.with_suffix(".json.tmp")
            progress_temp.write_text(
                json.dumps(
                    {
                        "event": str(event),
                        "run_id": str(getattr(task_state, "run_id", "") or ""),
                        "updated_at": time.time(),
                    }
                ),
                encoding="utf-8",
            )
            progress_temp.replace(progress_path)
            return previous_emit_trace(task_state, event, payload)

        agent.emit_trace = worker_emit_trace
        result = agent.ask(task)
        task_state = agent.current_task_state
        payload = {
            "status": "completed",
            "result": result,
            "session_id": str(agent.session.get("id", "")),
            "run_id": str(getattr(task_state, "run_id", "") or ""),
        }
        exit_code = 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
        exit_code = 1
    job_workspace.mkdir(parents=True, exist_ok=True)
    temp = result_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(result_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
