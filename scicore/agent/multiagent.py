"""Role-aware asynchronous sub-agent runtime."""

from __future__ import annotations

import concurrent.futures
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .prompt import system_prompt
from .progress import ProgressState


AGENT_ROLES = {
    "explorer": [
        "read_file",
        "grep",
        "glob",
        "bio_seq_inspect",
        "bio_sample_sheet_inspect",
        "bio_count_matrix_inspect",
        "file_hash",
    ],
    "planner": ["read_file", "grep", "glob", "write_file", "bio_workflow_sketch"],
    "executor": [
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "bio_seq_inspect",
        "bio_sample_sheet_inspect",
        "bio_count_matrix_inspect",
        "bio_rnaseq_compare",
        "bio_report",
        "file_hash",
        "bio_workflow_sketch",
    ],
    "verifier": ["read_file", "grep", "glob", "bash", "file_hash"],
    "bio_worker": [
        "read_file",
        "bash",
        "bio_seq_inspect",
        "bio_sample_sheet_inspect",
        "bio_count_matrix_inspect",
        "bio_rnaseq_compare",
        "bio_report",
        "file_hash",
        "bio_workflow_sketch",
    ],
}


@dataclass
class AgentJob:
    job_id: str
    role: str
    task: str
    status: str
    created_at: float
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None
    progress: ProgressState = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.progress is None:
            self.progress = ProgressState()


@dataclass
class AgentTeam:
    team_id: str
    objective: str
    jobs: list[str]
    created_at: float


class MultiAgentManager:
    """Small background job manager for role-restricted sub-agents."""

    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._jobs: dict[str, AgentJob] = {}
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._teams: dict[str, AgentTeam] = {}
        self._lock = Lock()

    def start(self, parent_agent: Any, task: str, role: str = "explorer") -> str:
        if role not in AGENT_ROLES:
            raise ValueError(f"unknown role '{role}'. Known roles: {', '.join(sorted(AGENT_ROLES))}")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = AgentJob(job_id=job_id, role=role, task=task, status="running", created_at=time.time())
        job.progress.update(percent=5, label="starting", current_step="creating sub-agent")
        with self._lock:
            self._jobs[job_id] = job
        future = self._pool.submit(self._run_job, parent_agent, job_id, task, role)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def status(self, job_id: str) -> AgentJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[AgentJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def start_team(self, parent_agent: Any, objective: str, agents: list[dict[str, str]]) -> AgentTeam:
        if not agents:
            raise ValueError("team requires at least one agent")
        if len(agents) > 6:
            raise ValueError("team cannot exceed 6 agents in this version")

        team_id = f"team_{uuid.uuid4().hex[:12]}"
        job_ids = []
        for index, spec in enumerate(agents, start=1):
            role = spec.get("role", "explorer")
            name = spec.get("name") or f"{role}_{index}"
            focus = spec.get("focus") or spec.get("task") or objective
            task = (
                f"Team objective: {objective}\n"
                f"Your team role name: {name}\n"
                f"Your assigned focus: {focus}\n"
                "Return a concise result with observed facts, missing information, and recommended next steps."
            )
            job_ids.append(self.start(parent_agent, task=task, role=role))

        team = AgentTeam(team_id=team_id, objective=objective, jobs=job_ids, created_at=time.time())
        with self._lock:
            self._teams[team_id] = team
        return team

    def team_status(self, team_id: str, auto_synthesize: bool = False) -> dict[str, Any] | None:
        with self._lock:
            team = self._teams.get(team_id)
            if team is None:
                return None
            jobs = [self._jobs[job_id] for job_id in team.jobs if job_id in self._jobs]

        completed = sum(1 for job in jobs if job.status == "completed")
        failed = sum(1 for job in jobs if job.status == "failed")
        running = sum(1 for job in jobs if job.status == "running")

        result = {
            "team_id": team.team_id,
            "objective": team.objective,
            "created_at": team.created_at,
            "status": "failed" if failed else "completed" if completed == len(jobs) else "running",
            "summary": {"total": len(jobs), "running": running, "completed": completed, "failed": failed},
            "jobs": [
                {
                    "job_id": job.job_id,
                    "role": job.role,
                    "status": job.status,
                    "progress": job.progress.to_dict(),
                    "result": job.result,
                    "error": job.error,
                }
                for job in jobs
            ],
        }

        # 自动合成团队结果
        if auto_synthesize and running == 0:
            try:
                from .agent_team import TeamSynthesizer

                agent_results = {}
                for job in jobs:
                    agent_results[job.job_id] = {
                        "role": job.role,
                        "status": job.status,
                        "result": job.result,
                        "error": job.error,
                    }

                synthesizer = TeamSynthesizer()
                synthesis = synthesizer.synthesize(
                    team_id=team.team_id,
                    objective=team.objective,
                    agent_results=agent_results,
                )
                result["synthesis"] = synthesis.to_dict()
            except Exception as e:
                result["synthesis_error"] = str(e)

        return result

    def _run_job(self, parent_agent: Any, job_id: str, task: str, role: str) -> None:
        try:
            from .agent import Agent

            self.update_progress(job_id, percent=20, label="planning", current_step=f"applying {role} role tools")
            tools = _tools_for_role(parent_agent.tools, role)
            sub = Agent(
                llm=parent_agent.llm,
                tools=tools,
                max_context_tokens=parent_agent.context.max_tokens,
                max_rounds=20,
            )
            sub._system = _role_system_prompt(role, sub.tools)
            self.update_progress(job_id, percent=45, label="running", current_step="sub-agent chat loop")
            result = sub.chat(task)
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            self.update_progress(job_id, percent=95, label="finalizing", current_step="collecting result")
            self._finish(job_id, status="completed", result=result)
        except Exception as e:
            self._finish(job_id, status="failed", error=str(e))

    def _finish(self, job_id: str, *, status: str, result: str | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.completed_at = time.time()
            job.result = result
            job.error = error
            job.progress.update(
                percent=100 if status == "completed" else job.progress.percent,
                label=status,
                current_step="done" if status == "completed" else str(error or status),
            )

    def update_progress(
        self,
        job_id: str,
        *,
        percent: int | None = None,
        label: str | None = None,
        current_step: str | None = None,
        total_steps: int | None = None,
        completed_steps: int | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.progress.update(
                    percent=percent,
                    label=label,
                    current_step=current_step,
                    total_steps=total_steps,
                    completed_steps=completed_steps,
                )


def _tools_for_role(parent_tools: list[Any], role: str) -> list[Any]:
    allowed = set(AGENT_ROLES[role])
    return [tool for tool in parent_tools if tool.name in allowed]


def _role_system_prompt(role: str, tools: list[Any]) -> str:
    base = system_prompt(tools)
    return (
        base
        + "\n# Multi-agent role boundary\n"
        + f"You are running as the '{role}' sub-agent role. Stay within this role's tools and responsibility.\n"
        + "If the task requires a forbidden action, report the missing capability instead of working around it.\n"
    )


GLOBAL_MULTIAGENT_MANAGER = MultiAgentManager()
