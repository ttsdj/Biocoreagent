"""Versioned lifecycle state machines used by the SciCoreAgent runtime.

Persisted state remains a plain string for JSON compatibility.  The enums and
transition maps provide a single authoritative vocabulary and reject invalid
runtime transitions before they can corrupt the audit trail.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class LifecycleTransitionError(ValueError):
    """Raised when a runtime object attempts an illegal state transition."""


class JobStatus(str, Enum):
    QUEUED = "queued"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TeamStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    REPLANNING = "replanning"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


JOB_TRANSITIONS: Mapping[str, frozenset[str]] = {
    JobStatus.QUEUED.value: frozenset(
        {
            JobStatus.WAITING_DEPENDENCIES.value,
            JobStatus.RUNNING.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.BLOCKED.value,
        }
    ),
    JobStatus.WAITING_DEPENDENCIES.value: frozenset(
        {
            JobStatus.RUNNING.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.BLOCKED.value,
        }
    ),
    JobStatus.RUNNING.value: frozenset(
        {
            JobStatus.QUEUED.value,
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }
    ),
    JobStatus.COMPLETED.value: frozenset(),
    JobStatus.FAILED.value: frozenset(),
    JobStatus.CANCELLED.value: frozenset(),
    JobStatus.BLOCKED.value: frozenset(),
}


TEAM_TRANSITIONS: Mapping[str, frozenset[str]] = {
    TeamStatus.QUEUED.value: frozenset(
        {
            TeamStatus.RUNNING.value,
            TeamStatus.REPLANNING.value,
            TeamStatus.VERIFYING.value,
            TeamStatus.SYNTHESIZING.value,
            TeamStatus.FAILED.value,
            TeamStatus.CANCELLED.value,
        }
    ),
    TeamStatus.RUNNING.value: frozenset(
        {
            TeamStatus.REPLANNING.value,
            TeamStatus.VERIFYING.value,
            TeamStatus.SYNTHESIZING.value,
            TeamStatus.FAILED.value,
            TeamStatus.CANCELLED.value,
        }
    ),
    TeamStatus.REPLANNING.value: frozenset(
        {
            TeamStatus.RUNNING.value,
            TeamStatus.VERIFYING.value,
            TeamStatus.SYNTHESIZING.value,
            TeamStatus.FAILED.value,
            TeamStatus.CANCELLED.value,
        }
    ),
    TeamStatus.VERIFYING.value: frozenset(
        {
            TeamStatus.SYNTHESIZING.value,
            TeamStatus.FAILED.value,
            TeamStatus.CANCELLED.value,
        }
    ),
    TeamStatus.SYNTHESIZING.value: frozenset(
        {
            TeamStatus.COMPLETED.value,
            TeamStatus.DEGRADED.value,
            TeamStatus.FAILED.value,
            TeamStatus.CANCELLED.value,
        }
    ),
    TeamStatus.COMPLETED.value: frozenset(),
    TeamStatus.DEGRADED.value: frozenset(),
    TeamStatus.FAILED.value: frozenset(),
    TeamStatus.CANCELLED.value: frozenset(),
}


def validate_state(machine: str, state: str, transitions: Mapping[str, frozenset[str]]) -> None:
    if state not in transitions:
        raise LifecycleTransitionError(f"unknown {machine} lifecycle state: {state!r}")


def ensure_transition(
    machine: str,
    current: str,
    target: str,
    transitions: Mapping[str, frozenset[str]],
) -> None:
    """Validate a transition, allowing an idempotent assignment."""

    validate_state(machine, current, transitions)
    validate_state(machine, target, transitions)
    if current == target:
        return
    if target not in transitions[current]:
        raise LifecycleTransitionError(
            f"illegal {machine} lifecycle transition: {current!r} -> {target!r}"
        )


def ensure_job_transition(current: str, target: str) -> None:
    ensure_transition("job", current, target, JOB_TRANSITIONS)


def ensure_team_transition(current: str, target: str) -> None:
    ensure_transition("team", current, target, TEAM_TRANSITIONS)
