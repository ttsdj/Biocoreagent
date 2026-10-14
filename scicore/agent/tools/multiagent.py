"""Asynchronous role-aware sub-agent tools."""

from __future__ import annotations

import json

from .base import Tool
from ..multiagent import AGENT_ROLES, GLOBAL_MULTIAGENT_MANAGER


class AgentStartTool(Tool):
    name = "agent_start"
    description = (
        "Start a role-restricted sub-agent in the background. "
        "Use agent_status to check completion."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task for the sub-agent."},
            "role": {
                "type": "string",
                "enum": sorted(AGENT_ROLES),
                "description": "Sub-agent role and tool boundary.",
            },
        },
        "required": ["task"],
    }

    _parent_agent = None

    def execute(self, task: str, role: str = "explorer") -> str:
        if self._parent_agent is None:
            return "Error: agent_start tool not initialized (no parent agent)"
        try:
            job_id = GLOBAL_MULTIAGENT_MANAGER.start(self._parent_agent, task=task, role=role)
            return json.dumps({"job_id": job_id, "role": role, "status": "running"}, indent=2)
        except Exception as e:
            return f"Error: {e}"


class AgentStatusTool(Tool):
    name = "agent_status"
    description = "Check the status and result of a background sub-agent job."
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Background job id returned by agent_start."}
        },
        "required": ["job_id"],
    }

    def execute(self, job_id: str) -> str:
        job = GLOBAL_MULTIAGENT_MANAGER.status(job_id)
        if job is None:
            return f"Error: unknown agent job '{job_id}'"
        return json.dumps(
            {
                "job_id": job.job_id,
                "role": job.role,
                "task": job.task,
                "status": job.status,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "progress": job.progress.to_dict(),
                "result": job.result,
                "error": job.error,
            },
            indent=2,
        )


class AgentTeamStartTool(Tool):
    name = "agent_team_start"
    description = (
        "Start a small role-restricted team of background sub-agents. "
        "Use agent_team_status to inspect aggregate progress and results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "Shared team objective."},
            "agents": {
                "type": "array",
                "description": "Team members with name, role, and focus.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Human-readable member name."},
                        "role": {"type": "string", "enum": sorted(AGENT_ROLES), "description": "Role boundary."},
                        "focus": {"type": "string", "description": "Specific assigned focus/task."},
                    },
                    "required": ["role", "focus"],
                },
            },
        },
        "required": ["objective", "agents"],
    }

    _parent_agent = None

    def execute(self, objective: str, agents: list[dict]) -> str:
        if self._parent_agent is None:
            return "Error: agent_team_start tool not initialized (no parent agent)"
        try:
            team = GLOBAL_MULTIAGENT_MANAGER.start_team(self._parent_agent, objective=objective, agents=agents)
            return json.dumps(
                {"team_id": team.team_id, "objective": team.objective, "job_ids": team.jobs, "status": "running"},
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"


class AgentTeamStatusTool(Tool):
    name = "agent_team_status"
    description = "Inspect aggregate status and results for a background sub-agent team."
    parameters = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id returned by agent_team_start."}
        },
        "required": ["team_id"],
    }

    def execute(self, team_id: str) -> str:
        status = GLOBAL_MULTIAGENT_MANAGER.team_status(team_id)
        if status is None:
            return f"Error: unknown agent team '{team_id}'"
        return json.dumps(status, indent=2)
