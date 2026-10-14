"""Persistent asyncio-based multi-agent orchestration for SciCoreAgent."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .audit import AuditTrail
from .lifecycle import (
    JOB_TRANSITIONS,
    TEAM_TRANSITIONS,
    JobStatus,
    TeamStatus,
    ensure_job_transition,
    ensure_team_transition,
    validate_state,
)


LIFECYCLE_STATES = (
    "queued",
    "waiting_dependencies",
    "running",
    "verifying",
    "replanning",
    "degraded",
    "synthesizing",
    "completed",
    "failed",
)
EXCEPTION_STATES = {JobStatus.CANCELLED.value, JobStatus.BLOCKED.value}
TERMINAL_STATES = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    *EXCEPTION_STATES,
}
TERMINAL_TEAM_STATES = {
    TeamStatus.COMPLETED.value,
    TeamStatus.FAILED.value,
    TeamStatus.CANCELLED.value,
    TeamStatus.DEGRADED.value,
}
VALID_ROLES = {"explorer", "planner", "executor", "verifier", "bio_worker"}


@dataclass
class AgentJob:
    job_id: str
    task: str
    role: str
    status: str = "queued"
    team_id: str = ""
    name: str = ""
    dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    attempt: int = 0
    max_retries: int = 1
    session_id: str = ""
    run_id: str = ""
    workspace_path: str = ""
    result: str = ""
    error: str = ""
    cancel_requested: bool = False
    timeout_seconds: float = 600.0
    failure_type: str = ""
    degradation_level: int = 0
    replan_generation: int = 0
    heartbeat_at: float | None = None
    progress_at: float | None = None
    deadline_at: float | None = None
    stalled_timeout_seconds: float = 600.0
    execution_mode: str = "thread"
    artifact_paths: list[str] = field(default_factory=list)
    parent_session_id: str = ""

    def __post_init__(self) -> None:
        validate_state("job", self.status, JOB_TRANSITIONS)


@dataclass
class AgentTeam:
    team_id: str
    objective: str
    job_ids: list[str]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    synthesis: str = ""
    started_at: float = field(default_factory=time.time)
    global_timeout_seconds: float = 1800.0
    failure_replan_threshold: float = 0.5
    max_replans: int = 1
    replan_count: int = 0
    degradation_level: int = 0
    forced_synthesis: bool = False
    verification_passed: bool | None = None
    replan_history: list[dict[str, Any]] = field(default_factory=list)
    parent_session_id: str = ""

    def __post_init__(self) -> None:
        validate_state("team", self.status, TEAM_TRANSITIONS)


@dataclass
class AgentArtifact:
    artifact_id: str
    job_id: str
    team_id: str
    kind: str
    path: str
    description: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentTransition:
    transition_id: str
    job_id: str
    team_id: str
    from_status: str
    to_status: str
    reason: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class TeamTransition:
    transition_id: str
    team_id: str
    from_status: str
    to_status: str
    reason: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ReplanNode:
    name: str
    role: str
    task: str
    depends_on: tuple[str, ...] = ()
    max_retries: int = 0
    timeout_seconds: float = 600.0
    stalled_timeout_seconds: float | None = None
    hard_timeout: bool | None = None
    artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplanPatch:
    reason: str
    nodes: tuple[ReplanNode, ...]


class JsonStateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve() / ".scicore" / "multiagent"
        self.jobs_dir = self.root / "jobs"
        self.teams_dir = self.root / "teams"
        self.messages_dir = self.root / "messages"
        self.artifacts_dir = self.root / "artifacts"
        self.transitions_dir = self.root / "transitions"
        self.team_transitions_dir = self.root / "team_transitions"
        self.workspaces_dir = self.root / "workspaces"
        self.leases_dir = self.root / "write_leases"
        self.completions_dir = self.root / "completion_inbox"
        for path in (
            self.jobs_dir,
            self.teams_dir,
            self.messages_dir,
            self.artifacts_dir,
            self.transitions_dir,
            self.team_transitions_dir,
            self.workspaces_dir,
            self.leases_dir,
            self.completions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save_job(self, job: AgentJob) -> None:
        self._atomic_json(self.jobs_dir / f"{job.job_id}.json", asdict(job))

    def load_job(self, job_id: str) -> AgentJob | None:
        data = self._read_json(self.jobs_dir / f"{job_id}.json")
        return AgentJob(**data) if data else None

    def list_jobs(self) -> list[AgentJob]:
        jobs = []
        for path in self.jobs_dir.glob("job_*.json"):
            data = self._read_json(path)
            if data:
                jobs.append(AgentJob(**data))
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def save_team(self, team: AgentTeam) -> None:
        self._atomic_json(self.teams_dir / f"{team.team_id}.json", asdict(team))

    def load_team(self, team_id: str) -> AgentTeam | None:
        data = self._read_json(self.teams_dir / f"{team_id}.json")
        return AgentTeam(**data) if data else None

    def append_message(self, team_id: str, message: dict[str, Any]) -> None:
        path = self.messages_dir / f"{team_id}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    def list_messages(self, team_id: str, recipient: str = "") -> list[dict[str, Any]]:
        path = self.messages_dir / f"{team_id}.jsonl"
        if not path.exists():
            return []
        messages = []
        with self._lock:
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if not recipient or item.get("recipient") in {"", "*", recipient}:
                    messages.append(item)
        return messages

    def append_transition(self, transition: AgentTransition) -> None:
        path = self.transitions_dir / f"{transition.job_id}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(transition), ensure_ascii=False) + "\n")

    def list_transitions(self, job_id: str) -> list[AgentTransition]:
        path = self.transitions_dir / f"{job_id}.jsonl"
        if not path.exists():
            return []
        transitions = []
        with self._lock:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    transitions.append(AgentTransition(**json.loads(line)))
        return transitions

    def append_team_transition(self, transition: TeamTransition) -> None:
        path = self.team_transitions_dir / f"{transition.team_id}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(transition), ensure_ascii=False) + "\n")

    def list_team_transitions(self, team_id: str) -> list[TeamTransition]:
        path = self.team_transitions_dir / f"{team_id}.jsonl"
        if not path.exists():
            return []
        transitions = []
        with self._lock:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    transitions.append(TeamTransition(**json.loads(line)))
        return transitions

    def save_artifact(self, artifact: AgentArtifact) -> None:
        self._atomic_json(self.artifacts_dir / f"{artifact.artifact_id}.json", asdict(artifact))

    def list_artifacts(self, job_id: str = "", team_id: str = "") -> list[AgentArtifact]:
        artifacts = []
        for path in self.artifacts_dir.glob("artifact_*.json"):
            data = self._read_json(path)
            if not data:
                continue
            artifact = AgentArtifact(**data)
            if job_id and artifact.job_id != job_id:
                continue
            if team_id and artifact.team_id != team_id:
                continue
            artifacts.append(artifact)
        return sorted(artifacts, key=lambda item: item.created_at)

    def job_workspace(self, job_id: str) -> Path:
        path = self.workspaces_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def acquire_write_lease(
        self,
        raw_path: str | Path,
        job_id: str,
        team_id: str = "",
        ttl_seconds: float = 600.0,
    ) -> tuple[bool, dict[str, Any]]:
        resolved = str(Path(raw_path).expanduser().resolve())
        lease_name = hashlib.sha256(resolved.lower().encode("utf-8")).hexdigest() + ".json"
        lease_path = self.leases_dir / lease_name
        now_value = time.time()
        with self._lock:
            current = self._read_json(lease_path)
            if (
                current
                and current.get("job_id") != job_id
                and float(current.get("expires_at", 0)) > now_value
            ):
                return False, current
            lease = {
                "path": resolved,
                "job_id": job_id,
                "team_id": team_id,
                "acquired_at": now_value,
                "expires_at": now_value + max(1.0, float(ttl_seconds)),
            }
            self._atomic_json(lease_path, lease)
            return True, lease

    def renew_write_leases(self, job: AgentJob) -> None:
        for raw_path in job.artifact_paths:
            self.acquire_write_lease(
                raw_path,
                job.job_id,
                job.team_id,
                ttl_seconds=max(job.timeout_seconds, 1.0),
            )

    def release_write_leases(self, job_id: str) -> None:
        with self._lock:
            for lease_path in self.leases_dir.glob("*.json"):
                current = self._read_json(lease_path)
                if current and current.get("job_id") == job_id:
                    current["expires_at"] = 0
                    current["released_at"] = time.time()
                    self._atomic_json(lease_path, current)

    def list_write_leases(self) -> list[dict[str, Any]]:
        leases = []
        for lease_path in self.leases_dir.glob("*.json"):
            current = self._read_json(lease_path)
            if current:
                leases.append(current)
        return sorted(leases, key=lambda item: item.get("acquired_at", 0))

    def append_completion(self, parent_session_id: str, completion: dict[str, Any]) -> dict[str, Any]:
        """Append a durable, recipient-scoped parent-session notification."""
        session_id = str(parent_session_id).strip()
        if not session_id:
            return {}
        path = self.completions_dir / f"{session_id}.jsonl"
        with self._lock:
            sequence = 1
            if path.exists():
                sequence += sum(
                    1
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            payload = {
                "sequence": sequence,
                "completion_id": "completion_" + uuid.uuid4().hex[:12],
                "parent_session_id": session_id,
                "created_at": time.time(),
                **completion,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def list_completions(
        self,
        parent_session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        path = self.completions_dir / f"{str(parent_session_id).strip()}.jsonl"
        if not path.exists():
            return []
        with self._lock:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return [
            item
            for item in rows
            if int(item.get("sequence", 0)) > int(after_sequence)
        ][: max(0, int(limit))]

    def _atomic_json(self, path: Path, data: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))


class AsyncMultiAgentOrchestrator:
    """Central scheduler for independent kernel runs.

    The scheduler owns lifecycle and persistence. Each worker owns its own
    model client, session, task state, checkpoint, trace, and role tool policy.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        agent_factory: Callable[[str, bool], Any],
        max_concurrency: int = 4,
        process_command_factory: Callable[[AgentJob, Path], list[str]] | None = None,
        default_execution_mode: str = "thread",
        watchdog_interval_seconds: float = 0.5,
        replan_strategy: Callable[[AgentTeam, list[AgentJob], float], ReplanPatch | dict[str, Any]]
        | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.agent_factory = agent_factory
        self.store = JsonStateStore(self.workspace_root)
        self.audit_trail = AuditTrail(self.workspace_root, "multiagent", actor="orchestrator")
        self.max_concurrency = max(1, int(max_concurrency))
        self.process_command_factory = process_command_factory
        if default_execution_mode not in {"thread", "process"}:
            raise ValueError("default_execution_mode must be thread or process")
        self.default_execution_mode = (
            "process"
            if default_execution_mode == "process" and process_command_factory is not None
            else "thread"
        )
        self.watchdog_interval_seconds = max(0.05, float(watchdog_interval_seconds))
        self.replan_strategy = replan_strategy or self._default_replan_patch
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="scicore-orchestrator", daemon=True)
        self._tasks: dict[str, Future] = {}
        self._agents: dict[str, Any] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._job_terminal_events: dict[str, asyncio.Event] = {}
        self._team_update_events: dict[str, asyncio.Event] = {}
        self._lock = threading.RLock()
        self._thread.start()
        self._semaphore = self._submit(self._make_semaphore()).result(timeout=5)
        self._recover_interrupted_jobs()
        self._watchdog_future = self._submit(self._watchdog_loop())

    @property
    def tool_names(self) -> set[str]:
        return {
            "agent_start", "agent_status", "agent_cancel", "agent_retry",
            "agent_team_start", "agent_team_status", "agent_team_cancel",
            "agent_message",
            "agent_artifacts",
        }

    async def _make_semaphore(self):
        return asyncio.Semaphore(self.max_concurrency)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coroutine) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def start_job(
        self,
        task: str,
        role: str = "explorer",
        *,
        team_id: str = "",
        name: str = "",
        dependencies: list[str] | None = None,
        max_retries: int = 1,
        timeout_seconds: float = 600,
        replan_generation: int = 0,
        stalled_timeout_seconds: float | None = None,
        hard_timeout: bool | None = None,
        artifact_paths: list[str] | None = None,
        parent_session_id: str = "",
    ) -> AgentJob:
        if role not in VALID_ROLES:
            raise ValueError(f"unknown role: {role}")
        if not task.strip():
            raise ValueError("task must not be empty")
        timeout_value = max(0.1, float(timeout_seconds))
        use_process = self.default_execution_mode == "process" if hard_timeout is None else bool(hard_timeout)
        if use_process and self.process_command_factory is None:
            use_process = False
        job = AgentJob(
            job_id="job_" + uuid.uuid4().hex[:12],
            task=task.strip(),
            role=role,
            team_id=team_id,
            name=name or role,
            dependencies=list(dependencies or []),
            max_retries=max(0, int(max_retries)),
            timeout_seconds=timeout_value,
            replan_generation=max(0, int(replan_generation)),
            heartbeat_at=time.time(),
            progress_at=time.time(),
            stalled_timeout_seconds=max(
                0.1,
                float(stalled_timeout_seconds)
                if stalled_timeout_seconds is not None
                else timeout_value,
            ),
            execution_mode="process" if use_process else "thread",
            artifact_paths=[
                str(
                    (Path(item).expanduser() if Path(item).expanduser().is_absolute()
                     else self.workspace_root / Path(item).expanduser()).resolve()
                )
                for item in (artifact_paths or [])
            ],
            parent_session_id=str(parent_session_id),
        )
        self.store.save_job(job)
        self._transition(job, "created", "queued", "job accepted by supervisor")
        self.audit_trail.append(
            "message",
            {
                "role": "supervisor",
                "content_preview": "agent job accepted",
                "job_id": job.job_id,
                "team_id": job.team_id,
                "agent_role": job.role,
                "task": job.task,
            },
        )
        future = self._submit(self._schedule_job(job.job_id))
        with self._lock:
            self._tasks[job.job_id] = future
        return job

    async def _schedule_job(self, job_id: str) -> None:
        job = self.store.load_job(job_id)
        if job is None or job.cancel_requested:
            if job is not None:
                self._finish_job(job, JobStatus.CANCELLED.value, error="cancel requested before execution")
            return

        dependencies = [self.store.load_job(dep) for dep in job.dependencies]
        if any(dep is None for dep in dependencies):
            self._finish_job(
                job,
                JobStatus.BLOCKED.value,
                error="dependency record is missing",
                reason="DAG dependency evidence is incomplete",
            )
            return
        if dependencies and not all(dep.status in TERMINAL_STATES for dep in dependencies):
            self._set_job_status(
                job,
                JobStatus.WAITING_DEPENDENCIES.value,
                "waiting for DAG predecessors",
            )
            self.store.save_job(job)
            await asyncio.gather(
                *(self._wait_for_job_terminal(dep.job_id) for dep in dependencies)
            )

        job = self.store.load_job(job_id)
        if job is None:
            return
        if job.cancel_requested:
            self._finish_job(job, JobStatus.CANCELLED.value, error="cancel requested before execution")
            return
        dependencies = [self.store.load_job(dep) for dep in job.dependencies]
        if any(
            dep is None
            or dep.status
            in {
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
                JobStatus.BLOCKED.value,
            }
            for dep in dependencies
        ):
            self._finish_job(
                job,
                JobStatus.BLOCKED.value,
                error="dependency did not complete successfully",
            )
            return

        async with self._semaphore:
            job = self.store.load_job(job_id)
            if job is None:
                return
            for raw_path in job.artifact_paths:
                acquired, lease = self.store.acquire_write_lease(
                    raw_path,
                    job.job_id,
                    job.team_id,
                    ttl_seconds=job.timeout_seconds,
                )
                if not acquired:
                    self.audit_trail.append(
                        "trace",
                        {
                            "event": "watchdog_tick_action",
                            "action": "write_conflict_blocked",
                            "job_id": job.job_id,
                            "path": raw_path,
                            "owner_job_id": lease.get("job_id", ""),
                        },
                    )
                    self._finish_job(
                        job,
                        "blocked",
                        error=f"write lease conflict for {raw_path}; owned by {lease.get('job_id', '')}",
                        reason="artifact write conflict detected before execution",
                    )
                    return
            job.started_at = time.time()
            job.deadline_at = job.started_at + job.timeout_seconds
            job.heartbeat_at = job.started_at
            job.progress_at = job.started_at
            self.store.save_job(job)
            try:
                if job.execution_mode == "process":
                    await asyncio.to_thread(self._execute_job, job_id)
                else:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._execute_job, job_id),
                        timeout=job.timeout_seconds,
                    )
            except asyncio.TimeoutError:
                job = self.store.load_job(job_id)
                if job is None or job.status in TERMINAL_STATES:
                    return
                job.cancel_requested = True
                job.failure_type = "timeout"
                job.degradation_level = max(job.degradation_level, 1)
                self._request_worker_cancellation(job.job_id)
                self._finish_job(
                    job,
                    "failed",
                    error=f"worker exceeded timeout_seconds={job.timeout_seconds:g}",
                    reason="level-1 degradation: single worker timeout",
                )
            finally:
                self.store.release_write_leases(job_id)

    async def _wait_for_job_terminal(self, job_id: str) -> None:
        current = self.store.load_job(job_id)
        if current is None or current.status in TERMINAL_STATES:
            return
        event = self._job_terminal_events.setdefault(job_id, asyncio.Event())
        # Close the load/register race: a worker may finish between the first
        # state read and event creation.
        current = self.store.load_job(job_id)
        if current is None or current.status in TERMINAL_STATES:
            event.set()
            return
        await event.wait()

    def _notify_job_update(self, job_id: str, team_id: str = "", *, terminal: bool = False) -> None:
        def notify() -> None:
            if terminal:
                self._job_terminal_events.setdefault(job_id, asyncio.Event()).set()
            if team_id:
                self._team_update_events.setdefault(team_id, asyncio.Event()).set()

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(notify)

    def _request_worker_cancellation(self, job_id: str, *, hard: bool = False) -> None:
        with self._lock:
            agent = self._agents.get(job_id)
            process = self._processes.get(job_id)
        manager = getattr(agent, "tool_manager", None)
        if manager is not None:
            manager.cancel()
        if hard and process is not None:
            self._terminate_process_tree(process)

    def _execute_job(self, job_id: str) -> None:
        job = self.store.load_job(job_id)
        if job is None:
            return
        if job.cancel_requested:
            self._finish_job(job, "cancelled", error="cancel requested")
            return

        self._set_job_status(job, "running", "worker slot acquired")
        job.started_at = time.time()
        job.attempt += 1
        workspace = self.store.job_workspace(job.job_id)
        job.workspace_path = str(workspace)
        self.store.save_job(job)
        try:
            if job.execution_mode == "process":
                self._execute_process_job(job, workspace)
                return
            agent = self._build_worker_agent(job.role, workspace)
            with self._lock:
                self._agents[job_id] = agent
            dependency_context = self._dependency_context(job)
            message_context = self._message_context(job)
            prompt = (
                f"Team objective task assigned to agent '{job.name}' ({job.role}).\n"
                f"{job.task}\n{dependency_context}{message_context}\n"
                "Return observed facts, actions taken, evidence paths, missing information, and recommended next steps."
            )
            previous_progress_callback = getattr(agent, "progress_callback", None)

            def worker_progress(message: str) -> None:
                self._touch_job(job_id, progress=True)
                if callable(previous_progress_callback):
                    previous_progress_callback(message)

            agent.progress_callback = worker_progress
            previous_emit_trace = getattr(agent, "emit_trace", None)
            if callable(previous_emit_trace):
                def worker_emit_trace(task_state, event, payload=None):
                    self._touch_job(job_id, progress=True)
                    return previous_emit_trace(task_state, event, payload)

                agent.emit_trace = worker_emit_trace
            self._touch_job(job_id, progress=True)
            result = agent.ask(prompt)
            job = self.store.load_job(job_id) or job
            if job.status in TERMINAL_STATES:
                return
            if job.cancel_requested:
                if job.failure_type != "timeout":
                    self._finish_job(job, "cancelled", error="cancel requested while worker was running")
                return
            job.session_id = str(agent.session.get("id", ""))
            if agent.current_task_state is not None:
                job.run_id = agent.current_task_state.run_id
            self._write_result_artifact(job, result)
            self._finish_job(job, "completed", result=result, reason="worker returned final result")
        except Exception as exc:
            job = self.store.load_job(job_id) or job
            if job.status in TERMINAL_STATES:
                return
            if job.attempt <= job.max_retries and not job.cancel_requested:
                self._set_job_status(job, "queued", "retry scheduled after worker error")
                job.error = str(exc)
                job.failure_type = exc.__class__.__name__
                self.store.save_job(job)
                future = self._submit(self._schedule_job(job_id))
                with self._lock:
                    self._tasks[job_id] = future
            else:
                job.failure_type = exc.__class__.__name__
                self._finish_job(job, "failed", error=str(exc), reason="worker error exceeded retry budget")
        finally:
            with self._lock:
                self._agents.pop(job_id, None)

    def _execute_process_job(self, job: AgentJob, workspace: Path) -> None:
        if self.process_command_factory is None:
            raise RuntimeError("process execution requested without process_command_factory")
        command = self.process_command_factory(job, workspace)
        if not command:
            raise ValueError("process_command_factory returned an empty command")
        stdout_path = workspace / "worker_process.stdout.txt"
        stderr_path = workspace / "worker_process.stderr.txt"
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        popen_kwargs: dict[str, Any] = {
            "cwd": str(workspace),
            "stdout": stdout_path.open("w", encoding="utf-8"),
            "stderr": stderr_path.open("w", encoding="utf-8"),
            "text": True,
            "creationflags": creationflags,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        with self._lock:
            self._processes[job.job_id] = process
        progress_path = workspace / "worker_progress.json"
        last_progress_mtime = 0.0
        try:
            while process.poll() is None:
                current = self.store.load_job(job.job_id)
                if current is None:
                    self._terminate_process_tree(process)
                    return
                progress_mtime = (
                    progress_path.stat().st_mtime if progress_path.is_file() else 0.0
                )
                made_progress = progress_mtime > last_progress_mtime
                if made_progress:
                    last_progress_mtime = progress_mtime
                self._touch_job(job.job_id, progress=made_progress)
                if current.cancel_requested or (
                    current.deadline_at is not None and time.time() >= current.deadline_at
                ):
                    self._terminate_process_tree(process)
                    current = self.store.load_job(job.job_id) or current
                    if current.status not in TERMINAL_STATES:
                        current.failure_type = "timeout" if not current.cancel_requested else "cancelled"
                        current.degradation_level = max(current.degradation_level, 1)
                        self._finish_job(
                            current,
                            "failed" if current.failure_type == "timeout" else "cancelled",
                            error="hard-timeout worker process was terminated",
                            reason="watchdog terminated isolated worker process",
                        )
                    return
                time.sleep(min(0.1, self.watchdog_interval_seconds))
            result_path = workspace / "worker_process_result.json"
            if process.returncode != 0:
                detail = ""
                if result_path.is_file():
                    try:
                        detail = str(json.loads(result_path.read_text(encoding="utf-8")).get("error", ""))
                    except Exception:
                        detail = ""
                raise RuntimeError(
                    f"isolated worker exited with code {process.returncode}"
                    + (f": {detail}" if detail else "")
                )
            if not result_path.is_file():
                raise FileNotFoundError("isolated worker did not write worker_process_result.json")
            process_result = json.loads(result_path.read_text(encoding="utf-8"))
            if process_result.get("status") != "completed":
                raise RuntimeError(str(process_result.get("error", "isolated worker failed")))
            current = self.store.load_job(job.job_id) or job
            if current.status in TERMINAL_STATES:
                return
            current.session_id = str(process_result.get("session_id", ""))
            current.run_id = str(process_result.get("run_id", ""))
            result = str(process_result.get("result", ""))
            self._write_result_artifact(current, result)
            self._finish_job(
                current,
                "completed",
                result=result,
                reason="isolated worker process returned final result",
            )
        finally:
            with self._lock:
                self._processes.pop(job.job_id, None)
            for handle_key in ("stdout", "stderr"):
                handle = popen_kwargs.get(handle_key)
                if hasattr(handle, "close"):
                    handle.close()

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def _touch_job(self, job_id: str, *, progress: bool) -> None:
        current = self.store.load_job(job_id)
        if current is None or current.status in TERMINAL_STATES:
            return
        timestamp = time.time()
        current.heartbeat_at = timestamp
        if progress:
            current.progress_at = timestamp
        self.store.save_job(current)

    async def _watchdog_loop(self) -> None:
        while True:
            now_value = time.time()
            for job in self.store.list_jobs():
                if job.status != "running":
                    continue
                self.store.renew_write_leases(job)
                deadline_reached = job.deadline_at is not None and now_value >= job.deadline_at
                stalled = (
                    job.progress_at is not None
                    and now_value - job.progress_at >= job.stalled_timeout_seconds
                )
                if not deadline_reached and not stalled:
                    continue
                reason = "deadline_exceeded" if deadline_reached else "progress_stalled"
                with self._lock:
                    process = self._processes.get(job.job_id)
                if process is not None:
                    self._terminate_process_tree(process)
                current = self.store.load_job(job.job_id) or job
                if current.status in TERMINAL_STATES:
                    continue
                current.cancel_requested = True
                current.failure_type = "timeout" if deadline_reached else "stalled"
                current.degradation_level = max(current.degradation_level, 1)
                self._request_worker_cancellation(
                    current.job_id,
                    hard=current.execution_mode == "process",
                )
                self.audit_trail.append(
                    "trace",
                    {
                        "event": "watchdog_tick_action",
                        "action": reason,
                        "job_id": current.job_id,
                        "execution_mode": current.execution_mode,
                        "heartbeat_at": current.heartbeat_at,
                        "progress_at": current.progress_at,
                        "deadline_at": current.deadline_at,
                    },
                )
                self._finish_job(
                    current,
                    "failed",
                    error=f"watchdog action: {reason}",
                    reason=f"level-1 degradation: {reason}",
                )
            await asyncio.sleep(self.watchdog_interval_seconds)

    def _finish_job(self, job: AgentJob, status: str, result: str = "", error: str = "", reason: str = "") -> None:
        self._set_job_status(job, status, reason or status)
        job.completed_at = time.time()
        job.result = result
        job.error = error
        self.store.save_job(job)
        self.store.append_completion(
            job.parent_session_id,
            {
                "kind": "agent_job_completed",
                "job_id": job.job_id,
                "team_id": job.team_id,
                "name": job.name,
                "role": job.role,
                "status": job.status,
                "result": job.result[:6000],
                "error": job.error[:2000],
                "artifact_paths": [
                    item.path for item in self.store.list_artifacts(job_id=job.job_id)
                ],
            },
        )
        self._notify_job_update(job.job_id, job.team_id, terminal=True)

    def _set_job_status(self, job: AgentJob, status: str, reason: str = "") -> None:
        previous = job.status
        ensure_job_transition(previous, status)
        job.status = status
        timestamp = time.time()
        job.heartbeat_at = timestamp
        job.progress_at = timestamp
        if previous != status:
            self._transition(job, previous, status, reason)

    def _transition(self, job: AgentJob, from_status: str, to_status: str, reason: str = "") -> None:
        transition = AgentTransition(
            transition_id="tr_" + uuid.uuid4().hex[:12],
            job_id=job.job_id,
            team_id=job.team_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
        )
        self.store.append_transition(transition)
        self.audit_trail.log_state_transition(asdict(transition))

    def _build_worker_agent(self, role: str, workspace: Path):
        signature = inspect.signature(self.agent_factory)
        if len(signature.parameters) >= 3:
            return self.agent_factory(role, True, workspace)
        return self.agent_factory(role, True)

    def _write_result_artifact(self, job: AgentJob, result: str) -> None:
        workspace = Path(job.workspace_path) if job.workspace_path else self.store.job_workspace(job.job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        result_path = workspace / "result.md"
        result_path.write_text(result or "", encoding="utf-8")
        artifact = AgentArtifact(
            artifact_id="artifact_" + uuid.uuid4().hex[:12],
            job_id=job.job_id,
            team_id=job.team_id,
            kind="worker_result",
            path=str(result_path),
            description=f"{job.name} ({job.role}) final response",
        )
        self.store.save_artifact(artifact)
        self.audit_trail.log_artifact(
            path=artifact.path,
            kind=artifact.kind,
            description=artifact.description,
            job_id=artifact.job_id,
            team_id=artifact.team_id,
            source="multiagent_worker",
            verified=True,
        )

    def start_team(
        self,
        objective: str,
        agents: list[dict[str, Any]],
        *,
        global_timeout_seconds: float = 1800,
        failure_replan_threshold: float = 0.5,
        max_replans: int = 1,
        parent_session_id: str = "",
    ) -> AgentTeam:
        if not objective.strip():
            raise ValueError("objective must not be empty")
        if not 2 <= len(agents) <= 12:
            raise ValueError("a team requires 2-12 agents")

        names = self._validate_team_specs(agents)
        team_id = "team_" + uuid.uuid4().hex[:12]
        name_to_id = {
            name: "job_" + uuid.uuid4().hex[:12]
            for name in names
        }
        jobs = []
        for index, spec in enumerate(agents, start=1):
            role = str(spec.get("role", "explorer"))
            if role not in VALID_ROLES:
                raise ValueError(f"unknown role: {role}")
            name = str(spec.get("name") or f"{role}_{index}")
            dependencies = [name_to_id[item] for item in spec.get("depends_on", [])]
            timeout_value = max(0.1, float(spec.get("timeout_seconds", 600)))
            requested_process = bool(
                spec.get("hard_timeout", self.default_execution_mode == "process")
            )
            use_process = requested_process and self.process_command_factory is not None
            artifact_paths = []
            for raw_path in spec.get("artifact_paths", []):
                candidate = Path(str(raw_path)).expanduser()
                if not candidate.is_absolute():
                    candidate = self.workspace_root / candidate
                artifact_paths.append(str(candidate.resolve()))
            timestamp = time.time()
            job = AgentJob(
                job_id=name_to_id[name],
                task=str(spec.get("task") or spec.get("focus") or objective),
                role=role,
                team_id=team_id,
                name=name,
                dependencies=dependencies,
                max_retries=max(0, int(spec.get("max_retries", 1))),
                timeout_seconds=timeout_value,
                heartbeat_at=timestamp,
                progress_at=timestamp,
                stalled_timeout_seconds=max(
                    0.1,
                    float(spec.get("stalled_timeout_seconds", timeout_value)),
                ),
                execution_mode="process" if use_process else "thread",
                artifact_paths=artifact_paths,
                parent_session_id=str(parent_session_id),
            )
            self.store.save_job(job)
            self._transition(
                job,
                "created",
                JobStatus.QUEUED.value,
                "team DAG node accepted by supervisor",
            )
            jobs.append(job)

        team = AgentTeam(
            team_id=team_id,
            objective=objective.strip(),
            job_ids=[job.job_id for job in jobs],
            global_timeout_seconds=max(0.1, float(global_timeout_seconds)),
            failure_replan_threshold=min(1.0, max(0.0, float(failure_replan_threshold))),
            max_replans=max(0, int(max_replans)),
            parent_session_id=str(parent_session_id),
        )
        self.store.save_team(team)
        self._transition_team(team, "created", "queued", "team accepted by supervisor")
        self.audit_trail.append(
            "message",
            {
                "role": "supervisor",
                "content_preview": "agent team accepted",
                "team_id": team.team_id,
                "objective": team.objective,
                "job_ids": team.job_ids,
            },
        )
        for job in jobs:
            with self._lock:
                self._tasks[job.job_id] = self._submit(self._schedule_job(job.job_id))
        self._submit(self._monitor_team(team_id))
        return team

    @staticmethod
    def _validate_team_specs(agents: list[dict[str, Any]]) -> list[str]:
        names = [
            str(spec.get("name") or f"{spec.get('role', 'explorer')}_{index}")
            for index, spec in enumerate(agents, start=1)
        ]
        if len(set(names)) != len(names):
            raise ValueError("team member names must be unique")
        known = set(names)
        graph = {}
        for name, spec in zip(names, agents):
            dependencies = [str(item) for item in spec.get("depends_on", [])]
            unknown = sorted(set(dependencies) - known)
            if unknown:
                raise ValueError(f"unknown dependencies for {name}: {', '.join(unknown)}")
            if name in dependencies:
                raise ValueError(f"team dependency cycle includes {name}")
            graph[name] = dependencies

        visiting = set()
        visited = set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"team dependency cycle includes {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in graph[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in names:
            visit(name)
        return names

    async def _monitor_team(self, team_id: str) -> None:
        update_event = self._team_update_events.setdefault(team_id, asyncio.Event())
        while True:
            update_event.clear()
            team = self.store.load_team(team_id)
            if team is None:
                return
            if team.status in TERMINAL_TEAM_STATES:
                return
            jobs = [self.store.load_job(job_id) for job_id in team.job_ids]
            jobs = [job for job in jobs if job is not None]
            elapsed = time.time() - team.started_at
            if elapsed >= team.global_timeout_seconds:
                self._force_team_synthesis(team, jobs)
                return
            if jobs and all(job.status in TERMINAL_STATES for job in jobs):
                failures = [job for job in jobs if job.status in {"failed", "blocked"}]
                original_jobs = [job for job in jobs if job.replan_generation == 0]
                failure_ratio = len(
                    [job for job in original_jobs if job.status in {"failed", "blocked"}]
                ) / max(1, len(original_jobs))
                if (
                    failures
                    and team.replan_count < team.max_replans
                    and failure_ratio >= team.failure_replan_threshold
                    and self._replan_team(team, jobs, failure_ratio)
                ):
                    continue
                self._finalize_team(team, jobs)
                return
            if team.status not in {"running", "replanning"}:
                self._set_team_status(team, "running", "team has active workers")
            remaining = max(0.0, team.global_timeout_seconds - elapsed)
            try:
                await asyncio.wait_for(update_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                team = self.store.load_team(team_id)
                if team is None or team.status in TERMINAL_TEAM_STATES:
                    return
                jobs = [self.store.load_job(job_id) for job_id in team.job_ids]
                self._force_team_synthesis(team, [job for job in jobs if job is not None])
                return

    def _default_replan_patch(
        self,
        team: AgentTeam,
        jobs: list[AgentJob],
        failure_ratio: float,
    ) -> ReplanPatch:
        evidence = []
        for job in jobs:
            evidence.append(
                f"- {job.name} ({job.role}) [{job.status}]: "
                f"{(job.result or job.error or '(no result)')[:1200]}"
            )
        remaining = max(
            0.1,
            min(600.0, team.global_timeout_seconds - (time.time() - team.started_at)),
        )
        generation = team.replan_count + 1
        return ReplanPatch(
            reason=f"failure ratio {failure_ratio:.2%} reached threshold",
            nodes=(
                ReplanNode(
                    name=f"replan_{generation}",
                    role="executor",
                    task=(
                        "Recover the team objective using the completed evidence below. "
                        "Do not repeat failed actions blindly. Produce a verified partial result when full "
                        "recovery is impossible, and name every unresolved blocker.\n\n"
                        f"Objective: {team.objective}\n"
                        + "\n".join(evidence)
                    ),
                    max_retries=0,
                    timeout_seconds=remaining,
                    stalled_timeout_seconds=remaining,
                ),
            ),
        )

    @staticmethod
    def _coerce_replan_patch(raw_patch: ReplanPatch | dict[str, Any]) -> ReplanPatch:
        if isinstance(raw_patch, ReplanPatch):
            return raw_patch
        if not isinstance(raw_patch, dict):
            raise TypeError("replan strategy must return ReplanPatch or a dictionary")
        nodes = []
        for raw_node in raw_patch.get("nodes", []):
            if not isinstance(raw_node, dict):
                raise TypeError("every replan node must be a dictionary")
            nodes.append(
                ReplanNode(
                    name=str(raw_node.get("name", "")),
                    role=str(raw_node.get("role", "executor")),
                    task=str(raw_node.get("task", "")),
                    depends_on=tuple(str(item) for item in raw_node.get("depends_on", [])),
                    max_retries=max(0, int(raw_node.get("max_retries", 0))),
                    timeout_seconds=max(0.1, float(raw_node.get("timeout_seconds", 600))),
                    stalled_timeout_seconds=(
                        None
                        if raw_node.get("stalled_timeout_seconds") is None
                        else max(0.1, float(raw_node["stalled_timeout_seconds"]))
                    ),
                    hard_timeout=raw_node.get("hard_timeout"),
                    artifact_paths=tuple(
                        str(item) for item in raw_node.get("artifact_paths", [])
                    ),
                )
            )
        return ReplanPatch(reason=str(raw_patch.get("reason", "")), nodes=tuple(nodes))

    @staticmethod
    def _validate_replan_patch(jobs: list[AgentJob], patch: ReplanPatch) -> None:
        if not patch.nodes:
            raise ValueError("replan patch must add at least one node")
        existing_names = {job.name for job in jobs}
        new_names = [node.name for node in patch.nodes]
        if any(not name.strip() for name in new_names):
            raise ValueError("replan node names must not be empty")
        if len(set(new_names)) != len(new_names):
            raise ValueError("replan node names must be unique")
        conflicts = sorted(existing_names.intersection(new_names))
        if conflicts:
            raise ValueError(f"replan node names already exist: {', '.join(conflicts)}")

        known = existing_names.union(new_names)
        graph = {job.name: [] for job in jobs}
        for node in patch.nodes:
            if node.role not in VALID_ROLES:
                raise ValueError(f"unknown replan role: {node.role}")
            if not node.task.strip():
                raise ValueError(f"replan task must not be empty: {node.name}")
            unknown = sorted(set(node.depends_on) - known)
            if unknown:
                raise ValueError(
                    f"unknown replan dependencies for {node.name}: {', '.join(unknown)}"
                )
            if node.name in node.depends_on:
                raise ValueError(f"replan dependency cycle includes {node.name}")
            graph[node.name] = list(node.depends_on)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"replan dependency cycle includes {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in graph.get(name, []):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in new_names:
            visit(name)

    def _replan_team(
        self,
        team: AgentTeam,
        jobs: list[AgentJob],
        failure_ratio: float,
    ) -> bool:
        try:
            patch = self._coerce_replan_patch(
                self.replan_strategy(team, jobs, failure_ratio)
            )
            self._validate_replan_patch(jobs, patch)
        except Exception as exc:
            team.replan_history.append(
                {
                    "generation": team.replan_count + 1,
                    "status": "rejected",
                    "failure_ratio": failure_ratio,
                    "error": str(exc),
                    "created_at": time.time(),
                }
            )
            self.store.save_team(team)
            self.audit_trail.append(
                "trace",
                {
                    "event": "replan_patch_rejected",
                    "team_id": team.team_id,
                    "failure_ratio": failure_ratio,
                    "error": str(exc),
                },
            )
            return False

        team.replan_count += 1
        generation = team.replan_count
        team.degradation_level = max(team.degradation_level, 2)
        self._set_team_status(
            team,
            TeamStatus.REPLANNING.value,
            f"level-2 degradation: {patch.reason}",
        )
        name_to_id = {job.name: job.job_id for job in jobs}
        name_to_id.update(
            {node.name: "job_" + uuid.uuid4().hex[:12] for node in patch.nodes}
        )
        created_jobs = []
        remaining_global = max(
            0.1,
            team.global_timeout_seconds - (time.time() - team.started_at),
        )
        for node in patch.nodes:
            requested_process = (
                self.default_execution_mode == "process"
                if node.hard_timeout is None
                else bool(node.hard_timeout)
            )
            use_process = requested_process and self.process_command_factory is not None
            timeout_value = max(0.1, min(node.timeout_seconds, remaining_global))
            artifact_paths = []
            for raw_path in node.artifact_paths:
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = self.workspace_root / candidate
                artifact_paths.append(str(candidate.resolve()))
            timestamp = time.time()
            recovery_job = AgentJob(
                job_id=name_to_id[node.name],
                task=node.task.strip(),
                role=node.role,
                team_id=team.team_id,
                name=node.name,
                dependencies=[name_to_id[item] for item in node.depends_on],
                max_retries=node.max_retries,
                timeout_seconds=timeout_value,
                replan_generation=generation,
                degradation_level=2,
                heartbeat_at=timestamp,
                progress_at=timestamp,
                stalled_timeout_seconds=max(
                    0.1,
                    min(
                        node.stalled_timeout_seconds or timeout_value,
                        remaining_global,
                    ),
                ),
                execution_mode="process" if use_process else "thread",
                artifact_paths=artifact_paths,
                parent_session_id=team.parent_session_id,
            )
            self.store.save_job(recovery_job)
            self._transition(
                recovery_job,
                "created",
                JobStatus.QUEUED.value,
                "dynamic DAG replan node created",
            )
            created_jobs.append(recovery_job)
            team.job_ids.append(recovery_job.job_id)

        team.replan_history.append(
            {
                "generation": generation,
                "status": "applied",
                "failure_ratio": failure_ratio,
                "reason": patch.reason,
                "nodes": [
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "role": job.role,
                        "dependencies": job.dependencies,
                    }
                    for job in created_jobs
                ],
                "created_at": time.time(),
            }
        )
        self.store.save_team(team)
        with self._lock:
            for job in created_jobs:
                self._tasks[job.job_id] = self._submit(self._schedule_job(job.job_id))
        self._notify_job_update("", team.team_id)
        return True

    def _finalize_team(self, team: AgentTeam, jobs: list[AgentJob]) -> None:
        self._set_team_status(team, "verifying", "checking worker result artifacts")
        team.verification_passed = self._verify_team_artifacts(jobs)
        self._set_team_status(team, "synthesizing", "supervisor is composing final evidence")
        has_failures = any(job.status in {"failed", "blocked"} for job in jobs)
        recovery_succeeded = any(
            job.replan_generation > 0 and job.status == "completed"
            for job in jobs
        )
        if has_failures and recovery_succeeded:
            final_status = "degraded"
            team.degradation_level = max(team.degradation_level, 2)
        elif has_failures:
            final_status = "failed"
        elif any(job.status == "cancelled" for job in jobs):
            final_status = "cancelled"
        elif team.verification_passed:
            final_status = "completed"
        else:
            final_status = "failed"
        team.completed_at = time.time()
        team.synthesis = self._synthesize(team, jobs)
        self._write_team_synthesis_artifact(team, final_status=final_status)
        self._set_team_status(team, final_status, "team lifecycle reached terminal state")
        self._publish_team_completion(team)

    def _force_team_synthesis(self, team: AgentTeam, jobs: list[AgentJob]) -> None:
        team.forced_synthesis = True
        team.degradation_level = 3
        for job in jobs:
            if job.status not in TERMINAL_STATES:
                job.cancel_requested = True
                job.failure_type = "global_timeout"
                job.degradation_level = 3
                self._request_worker_cancellation(
                    job.job_id,
                    hard=job.execution_mode == "process",
                )
                self._finish_job(
                    job,
                    "failed",
                    error=f"team exceeded global_timeout_seconds={team.global_timeout_seconds:g}",
                    reason="level-3 degradation: global timeout",
                )
        jobs = [self.store.load_job(job_id) for job_id in team.job_ids]
        jobs = [job for job in jobs if job is not None]
        self._set_team_status(
            team,
            "synthesizing",
            "level-3 degradation: force synthesis from completed evidence",
        )
        team.verification_passed = self._verify_team_artifacts(
            [job for job in jobs if job.status == "completed"]
        )
        team.completed_at = time.time()
        team.synthesis = self._synthesize(team, jobs)
        self._write_team_synthesis_artifact(team, final_status="degraded")
        self._set_team_status(team, "degraded", "partial synthesis completed after global timeout")
        self._publish_team_completion(team)

    def _publish_team_completion(self, team: AgentTeam) -> None:
        self.store.append_completion(
            team.parent_session_id,
            {
                "kind": "agent_team_completed",
                "team_id": team.team_id,
                "status": team.status,
                "degradation_level": team.degradation_level,
                "forced_synthesis": team.forced_synthesis,
                "verification_passed": team.verification_passed,
                "synthesis": team.synthesis[:12000],
                "artifact_paths": [
                    item.path for item in self.store.list_artifacts(team_id=team.team_id)
                ],
            },
        )

    def _verify_team_artifacts(self, jobs: list[AgentJob]) -> bool:
        completed = [job for job in jobs if job.status == "completed"]
        if not completed:
            return False
        for job in completed:
            artifacts = self.store.list_artifacts(job_id=job.job_id)
            if not any(Path(item.path).is_file() for item in artifacts):
                return False
        return True

    def cancel_job(self, job_id: str) -> bool:
        job = self.store.load_job(job_id)
        if job is None or job.status in TERMINAL_STATES:
            return False
        job.cancel_requested = True
        self.store.save_job(job)
        future = self._tasks.get(job_id)
        if job.status in {
            JobStatus.QUEUED.value,
            JobStatus.WAITING_DEPENDENCIES.value,
        } and future:
            future.cancel()
            self._finish_job(job, JobStatus.CANCELLED.value, error="cancel requested")
        else:
            self._request_worker_cancellation(
                job_id,
                hard=job.execution_mode == "process",
            )
            self._notify_job_update(job.job_id, job.team_id)
        return True

    def cancel_team(self, team_id: str) -> bool:
        team = self.store.load_team(team_id)
        if team is None or team.status in TERMINAL_TEAM_STATES:
            return False
        changed = False
        for job_id in team.job_ids:
            changed = self.cancel_job(job_id) or changed
        if changed:
            team = self.store.load_team(team_id) or team
            if team.status not in TERMINAL_TEAM_STATES:
                team.completed_at = time.time()
                self._set_team_status(
                    team,
                    TeamStatus.CANCELLED.value,
                    "team cancellation requested by supervisor",
                )
            self._notify_job_update("", team_id)
        return changed

    def retry_job(self, job_id: str) -> AgentJob:
        old = self.store.load_job(job_id)
        if old is None:
            raise ValueError(f"unknown job: {job_id}")
        if old.status not in TERMINAL_STATES:
            raise ValueError("only terminal jobs can be retried")
        return self.start_job(
            old.task,
            old.role,
            team_id=old.team_id,
            name=old.name,
            dependencies=old.dependencies,
            max_retries=old.max_retries,
            timeout_seconds=old.timeout_seconds,
            stalled_timeout_seconds=old.stalled_timeout_seconds,
            hard_timeout=old.execution_mode == "process",
            artifact_paths=old.artifact_paths,
        )

    def send_message(self, team_id: str, sender: str, recipient: str, content: str) -> dict[str, Any]:
        if self.store.load_team(team_id) is None:
            raise ValueError(f"unknown team: {team_id}")
        message = {
            "message_id": "msg_" + uuid.uuid4().hex[:12],
            "team_id": team_id,
            "sender": sender,
            "recipient": recipient or "*",
            "content": content,
            "created_at": time.time(),
        }
        self.store.append_message(team_id, message)
        self.audit_trail.append("message", {"role": "agent_message", **message})
        return message

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.load_job(job_id)
        if not job:
            return None
        data = asdict(job)
        data["artifacts"] = [asdict(item) for item in self.store.list_artifacts(job_id=job_id)]
        data["transitions"] = [asdict(item) for item in self.store.list_transitions(job_id)]
        return data

    def team_status(self, team_id: str) -> dict[str, Any] | None:
        team = self.store.load_team(team_id)
        if team is None:
            return None
        data = asdict(team)
        data["jobs"] = [self.job_status(job_id) for job_id in team.job_ids]
        data["messages"] = self.store.list_messages(team_id)
        data["artifacts"] = [asdict(item) for item in self.store.list_artifacts(team_id=team_id)]
        data["transitions"] = [
            asdict(item) for item in self.store.list_team_transitions(team_id)
        ]
        return data

    def wait_for_job(self, job_id: str, timeout: float = 30) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.job_status(job_id)
            if status is None or status["status"] in TERMINAL_STATES:
                return status
            time.sleep(0.05)
        return self.job_status(job_id)

    def wait_for_team(self, team_id: str, timeout: float = 60) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.team_status(team_id)
            if status is None or status["status"] in TERMINAL_TEAM_STATES:
                return status
            time.sleep(0.05)
        return self.team_status(team_id)

    def completion_updates(
        self,
        parent_session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return completed child/team results ready for parent reinjection."""
        return self.store.list_completions(
            parent_session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def _dependency_context(self, job: AgentJob) -> str:
        if not job.dependencies:
            return ""
        lines = ["\nCompleted dependency results:"]
        for dep_id in job.dependencies:
            dep = self.store.load_job(dep_id)
            if dep:
                lines.append(f"- {dep.name} ({dep.role}): {dep.result[:3000]}")
        return "\n".join(lines)

    def _message_context(self, job: AgentJob) -> str:
        if not job.team_id:
            return ""
        messages = self.store.list_messages(job.team_id, recipient=job.name)
        if not messages:
            return ""
        return "\nTeam messages:\n" + "\n".join(
            f"- from {item['sender']}: {item['content']}" for item in messages[-10:]
        )

    @staticmethod
    def _synthesize(team: AgentTeam, jobs: list[AgentJob]) -> str:
        lines = [
            f"# Team synthesis: {team.objective}",
            "",
            f"- degradation_level: {team.degradation_level}",
            f"- replan_count: {team.replan_count}",
            f"- forced_synthesis: {str(team.forced_synthesis).lower()}",
            f"- verification_passed: {team.verification_passed}",
            "",
        ]
        for job in jobs:
            lines.extend([
                f"## {job.name} ({job.role}) [{job.status}]",
                job.result or job.error or "(no result)",
                "",
            ])
        lines.append("## Decision boundary")
        if team.forced_synthesis:
            lines.append(
                "The global timeout forced a partial synthesis. Completed evidence is retained, "
                "but unresolved work must not be reported as successful."
            )
        elif any(job.status != "completed" for job in jobs):
            lines.append("The team has incomplete or failed work; do not treat the objective as verified.")
        else:
            lines.append("All assigned agents completed. Final claims still require artifact-level verification.")
        return "\n".join(lines)

    def _set_team_status(self, team: AgentTeam, status: str, reason: str = "") -> None:
        previous = team.status
        ensure_team_transition(previous, status)
        team.status = status
        if previous != status:
            self._transition_team(team, previous, status, reason)
        self.store.save_team(team)

    def _transition_team(
        self,
        team: AgentTeam,
        from_status: str,
        to_status: str,
        reason: str = "",
    ) -> None:
        transition = TeamTransition(
            transition_id="team_tr_" + uuid.uuid4().hex[:12],
            team_id=team.team_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
        )
        self.store.append_team_transition(transition)
        self.audit_trail.log_state_transition(
            {
                **asdict(transition),
                "job_id": "",
            }
        )

    def _write_team_synthesis_artifact(
        self,
        team: AgentTeam,
        final_status: str | None = None,
    ) -> None:
        team_dir = self.store.root / "teams" / team.team_id
        team_dir.mkdir(parents=True, exist_ok=True)
        synthesis_path = team_dir / "synthesis.md"
        synthesis_path.write_text(team.synthesis or "", encoding="utf-8")
        artifact = AgentArtifact(
            artifact_id="artifact_" + uuid.uuid4().hex[:12],
            job_id="",
            team_id=team.team_id,
            kind="team_synthesis",
            path=str(synthesis_path),
            description=f"Supervisor synthesis for {team.team_id}",
        )
        self.store.save_artifact(artifact)
        self.audit_trail.log_artifact(
            path=artifact.path,
            kind=artifact.kind,
            description=artifact.description,
            team_id=artifact.team_id,
            source="multiagent_supervisor",
            verified=True,
        )
        self.audit_trail.finalize(
            final_status=final_status or team.status,
            final_answer=team.synthesis,
        )

    def _recover_interrupted_jobs(self) -> None:
        for job in self.store.list_jobs():
            if job.status == "running":
                self._finish_job(
                    job,
                    JobStatus.FAILED.value,
                    error="orchestrator restarted while job was running; retry explicitly",
                    reason="orchestrator restart interrupted running worker",
                )

    def tool_registry(self, parent_agent) -> dict[str, dict[str, Any]]:
        parent_session_id = str(parent_agent.session.get("id", ""))
        return {
            "agent_start": self._spec(
                {
                    "task": "string",
                    "role": "string=optional",
                    "max_retries": "integer=optional",
                    "timeout_seconds": "number=optional",
                    "stalled_timeout_seconds": "number=optional",
                    "hard_timeout": "boolean=optional",
                    "artifact_paths": "array=optional",
                },
                "Start a background agent with optional hard timeout and artifact write leases.",
                lambda args: json.dumps(
                    asdict(self.start_job(parent_session_id=parent_session_id, **args)),
                    ensure_ascii=False,
                ),
                risky=True,
            ),
            "agent_status": self._spec(
                {"job_id": "string"},
                "Read durable status for a background agent.",
                lambda args: json.dumps(self.job_status(args["job_id"]), ensure_ascii=False, indent=2),
            ),
            "agent_cancel": self._spec(
                {"job_id": "string"},
                "Request cancellation of a queued or running agent.",
                lambda args: json.dumps({"cancelled": self.cancel_job(args["job_id"])}),
                risky=True,
            ),
            "agent_retry": self._spec(
                {"job_id": "string"},
                "Create a new attempt for a terminal agent job.",
                lambda args: json.dumps(asdict(self.retry_job(args["job_id"])), ensure_ascii=False),
                risky=True,
            ),
            "agent_team_start": self._spec(
                {
                    "objective": "string",
                    "agents": "array",
                    "global_timeout_seconds": "number=optional",
                    "failure_replan_threshold": "number=optional",
                    "max_replans": "integer=optional",
                },
                "Start a concurrent or dependency-ordered role-based agent team.",
                lambda args: json.dumps(
                    asdict(self.start_team(parent_session_id=parent_session_id, **args)),
                    ensure_ascii=False,
                ),
                risky=True,
            ),
            "agent_team_status": self._spec(
                {"team_id": "string"},
                "Read team, member, synthesis, and message state.",
                lambda args: json.dumps(self.team_status(args["team_id"]), ensure_ascii=False, indent=2),
            ),
            "agent_team_cancel": self._spec(
                {"team_id": "string"},
                "Request cancellation of every non-terminal member of a team.",
                lambda args: json.dumps({"cancel_requested": self.cancel_team(args["team_id"])}),
                risky=True,
            ),
            "agent_message": self._spec(
                {"team_id": "string", "sender": "string", "recipient": "string", "content": "string"},
                "Persist a message for another member of an agent team.",
                lambda args: json.dumps(self.send_message(**args), ensure_ascii=False),
            ),
            "agent_artifacts": self._spec(
                {"job_id": "string=optional", "team_id": "string=optional"},
                "List registered multi-agent artifacts for a job or team.",
                lambda args: json.dumps(
                    [asdict(item) for item in self.store.list_artifacts(args.get("job_id", ""), args.get("team_id", ""))],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        }

    @staticmethod
    def _spec(schema, description, runner, risky=False):
        return {"schema": schema, "description": description, "risky": risky, "run": runner}

    def validate_tool(self, name: str, args: dict[str, Any]) -> None:
        required = {
            "agent_start": {"task"},
            "agent_status": {"job_id"},
            "agent_cancel": {"job_id"},
            "agent_retry": {"job_id"},
            "agent_team_start": {"objective", "agents"},
            "agent_team_status": {"team_id"},
            "agent_team_cancel": {"team_id"},
            "agent_message": {"team_id", "sender", "recipient", "content"},
            "agent_artifacts": set(),
        }[name]
        missing = sorted(key for key in required if args.get(key) in (None, ""))
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")

    def shutdown(self) -> None:
        try:
            self._submit(self._cancel_pending_tasks()).result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _cancel_pending_tasks(self) -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
