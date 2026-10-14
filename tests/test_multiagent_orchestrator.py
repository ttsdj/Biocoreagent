import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scicore.bio.lifecycle import (
    LifecycleTransitionError,
    ensure_job_transition,
    ensure_team_transition,
)
from scicore.bio.orchestrator import (
    LIFECYCLE_STATES,
    AgentJob,
    AgentTeam,
    AsyncMultiAgentOrchestrator,
    JsonStateStore,
    ReplanNode,
    ReplanPatch,
)
from scicore.bio.runtime import BioRuntime
from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.task_state import TaskState
from scicore.kernel.workspace import WorkspaceContext


def build_factory(tmp_path: Path, prompts: dict[str, list[str]]):
    def factory(role: str, worker: bool, worker_workspace: Path | None = None):
        client = FakeModelClient([f"<final>{role} completed</final>"])
        prompts.setdefault(role, []).append(client)
        root = worker_workspace or tmp_path
        return BioRuntime(
            model_client=client,
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            orchestrator=None,
            allow_orchestration=not worker,
        )

    return factory


def test_team_runs_independent_agents_and_persists_state(tmp_path):
    clients = {}
    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        build_factory(tmp_path, clients),
        max_concurrency=2,
    )
    try:
        team = manager.start_team(
            "Review RNA-seq inputs",
            [
                {"name": "inspect", "role": "explorer", "task": "Inspect files"},
                {"name": "plan", "role": "planner", "task": "Plan analysis"},
            ],
        )
        status = manager.wait_for_team(team.team_id, timeout=10)

        assert status["status"] == "completed"
        assert {job["role"] for job in status["jobs"]} == {"explorer", "planner"}
        assert all(job["session_id"] for job in status["jobs"])
        assert len({job["session_id"] for job in status["jobs"]}) == 2
        assert len({job["run_id"] for job in status["jobs"]}) == 2
        assert all(Path(job["workspace_path"]).is_dir() for job in status["jobs"])
        assert len({job["workspace_path"] for job in status["jobs"]}) == 2
        assert all(any(item["kind"] == "worker_result" for item in job["artifacts"]) for job in status["jobs"])
        assert all([item["to_status"] for job in status["jobs"] for item in job["transitions"]])
        assert any(item["kind"] == "team_synthesis" for item in status["artifacts"])
        assert "All assigned agents completed" in status["synthesis"]
        audit_dir = tmp_path / ".scicore" / "audit" / "sessions" / "multiagent"
        assert (audit_dir / "state_transitions.jsonl").exists()
        assert (audit_dir / "artifacts.jsonl").exists()
        assert (audit_dir / "audit_index.json").exists()
        transitions = [
            json.loads(line)
            for line in (audit_dir / "state_transitions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(item["to_status"] == "running" for item in transitions)
        artifacts = [
            json.loads(line)
            for line in (audit_dir / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(item["kind"] == "team_synthesis" for item in artifacts)

        reloaded = JsonStateStore(tmp_path)
        assert reloaded.load_team(team.team_id).status == "completed"
        assert all(reloaded.load_job(job_id).status == "completed" for job_id in team.job_ids)
    finally:
        manager.shutdown()


def test_semaphore_runs_ready_dag_nodes_concurrently_without_exceeding_limit(tmp_path):
    guard = threading.Lock()
    two_workers_active = threading.Event()
    active = 0
    max_active = 0
    sequence = 0

    class ProbeAgent:
        def __init__(self, role):
            nonlocal sequence
            with guard:
                sequence += 1
                identifier = sequence
            self.role = role
            self.session = {"id": f"session_{identifier}"}
            self.current_task_state = SimpleNamespace(run_id=f"run_{identifier}")
            self.progress_callback = None

        def ask(self, prompt):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    two_workers_active.set()
            two_workers_active.wait(timeout=1)
            time.sleep(0.05)
            with guard:
                active -= 1
            return f"{self.role} probe result"

    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        lambda role, worker, worker_workspace=None: ProbeAgent(role),
        max_concurrency=2,
    )
    try:
        team = manager.start_team(
            "measure semaphore",
            [
                {"name": "one", "role": "explorer"},
                {"name": "two", "role": "planner"},
                {"name": "three", "role": "executor"},
            ],
            max_replans=0,
        )
        status = manager.wait_for_team(team.team_id, timeout=5)

        assert status["status"] == "completed"
        assert max_active == 2
    finally:
        manager.shutdown()


def test_dag_dependency_result_is_injected_into_downstream_prompt(tmp_path):
    clients = {}
    manager = AsyncMultiAgentOrchestrator(tmp_path, build_factory(tmp_path, clients), max_concurrency=2)
    try:
        team = manager.start_team(
            "Validate analysis",
            [
                {"name": "inspect", "role": "explorer", "task": "Inspect"},
                {
                    "name": "verify",
                    "role": "verifier",
                    "task": "Verify",
                    "depends_on": ["inspect"],
                },
            ],
        )
        status = manager.wait_for_team(team.team_id, timeout=10)

        assert status["status"] == "completed"
        verifier_prompt = clients["verifier"][0].prompts[0]
        assert "Completed dependency results" in verifier_prompt
        assert "explorer completed" in verifier_prompt
    finally:
        manager.shutdown()


def test_completed_child_result_is_reinjected_once_into_parent_session(tmp_path):
    clients = {}
    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        build_factory(tmp_path, clients),
        max_concurrency=1,
    )
    try:
        parent = BioRuntime(
            model_client=FakeModelClient(["<final>parent</final>"]),
            workspace=WorkspaceContext.build(tmp_path),
            session_store=SessionStore(tmp_path / ".scicore" / "parent-sessions"),
            run_store=RunStore(tmp_path / ".scicore" / "parent-runs"),
            approval_policy="auto",
            role="executor",
            orchestrator=manager,
        )
        job = manager.start_job(
            "Inspect retained evidence",
            "explorer",
            parent_session_id=parent.session["id"],
        )

        assert manager.wait_for_job(job.job_id, timeout=10)["status"] == "completed"
        injected = parent.consume_agent_completions()

        assert job.job_id in injected
        assert "explorer completed" in injected
        assert parent.consume_agent_completions() == ""
        assert parent.session["orchestrator"]["completion_cursor"] == 1
        assert parent.session["history"][-1]["metadata"]["kind"] == "orchestrator_completion_reinjection"
    finally:
        manager.shutdown()


def test_message_bus_is_durable_and_recipient_scoped(tmp_path):
    clients = {}
    manager = AsyncMultiAgentOrchestrator(tmp_path, build_factory(tmp_path, clients))
    try:
        team = manager.start_team(
            "Message test",
            [
                {"name": "one", "role": "explorer", "task": "One"},
                {"name": "two", "role": "planner", "task": "Two"},
            ],
        )
        sent = manager.send_message(team.team_id, "one", "two", "check sample labels")
        messages = manager.store.list_messages(team.team_id, recipient="two")

        assert messages == [sent]
        assert manager.store.list_messages(team.team_id, recipient="one") == []
        manager.wait_for_team(team.team_id, timeout=5)
    finally:
        manager.shutdown()


def test_cancel_queued_job_and_retry_terminal_job(tmp_path):
    class SlowClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            time.sleep(0.3)
            return super().complete(prompt, max_new_tokens, **kwargs)

    def factory(role, worker):
        return BioRuntime(
            model_client=SlowClient([f"<final>{role} done</final>"]),
            workspace=WorkspaceContext.build(tmp_path),
            session_store=SessionStore(tmp_path / ".scicore" / "sessions"),
            run_store=RunStore(tmp_path / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    manager = AsyncMultiAgentOrchestrator(tmp_path, factory, max_concurrency=1)
    try:
        first = manager.start_job("occupy worker", "explorer")
        second = manager.start_job("cancel me", "planner")
        assert manager.cancel_job(second.job_id)
        cancelled = manager.wait_for_job(second.job_id, timeout=5)
        assert cancelled["status"] == "cancelled"

        retried = manager.retry_job(second.job_id)
        done = manager.wait_for_job(retried.job_id, timeout=10)
        assert done["status"] == "completed"
        assert retried.job_id != second.job_id
        assert manager.wait_for_job(first.job_id, timeout=10)["status"] == "completed"
    finally:
        manager.shutdown()


def test_team_rejects_unknown_and_cyclic_dependencies_before_persisting(tmp_path):
    manager = AsyncMultiAgentOrchestrator(tmp_path, build_factory(tmp_path, {}))
    try:
        try:
            manager.start_team(
                "bad",
                [
                    {"name": "one", "role": "explorer", "depends_on": ["missing"]},
                    {"name": "two", "role": "planner"},
                ],
            )
            raise AssertionError("unknown dependency was accepted")
        except ValueError as exc:
            assert "unknown dependencies" in str(exc)

        try:
            manager.start_team(
                "cycle",
                [
                    {"name": "one", "role": "explorer", "depends_on": ["two"]},
                    {"name": "two", "role": "planner", "depends_on": ["one"]},
                ],
            )
            raise AssertionError("dependency cycle was accepted")
        except ValueError as exc:
            assert "cycle" in str(exc)
        assert manager.store.list_jobs() == []
    finally:
        manager.shutdown()


def test_root_orchestration_tools_require_approval(tmp_path):
    clients = {}
    holder = {}

    def factory(role, worker):
        return BioRuntime(
            model_client=FakeModelClient(["<final>done</final>"]),
            workspace=WorkspaceContext.build(tmp_path),
            session_store=SessionStore(tmp_path / ".scicore" / "sessions"),
            run_store=RunStore(tmp_path / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            orchestrator=holder.get("manager"),
            allow_orchestration=not worker,
        )

    manager = AsyncMultiAgentOrchestrator(tmp_path, factory)
    holder["manager"] = manager
    try:
        root = factory("executor", False)
        assert root.tools["agent_start"]["risky"] is True
        assert root.tools["agent_team_start"]["risky"] is True
        assert "agent_artifacts" in root.tools
        worker = factory("bio_worker", True)
        assert "agent_start" not in worker.tools
        assert "agent_team_start" not in worker.tools
    finally:
        manager.shutdown()


def test_lifecycle_declares_nine_operational_states():
    assert len(LIFECYCLE_STATES) == 9
    assert LIFECYCLE_STATES == (
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


def test_three_lifecycle_state_machines_reject_illegal_transitions():
    task = TaskState.create(task_id="task_test", user_request="test")
    task.finish_success("done")
    with pytest.raises(ValueError, match="illegal task lifecycle transition"):
        task.stop("too_late")

    with pytest.raises(LifecycleTransitionError, match="illegal job lifecycle transition"):
        ensure_job_transition("completed", "running")
    with pytest.raises(LifecycleTransitionError, match="illegal team lifecycle transition"):
        ensure_team_transition("completed", "replanning")
    with pytest.raises(LifecycleTransitionError, match="unknown job lifecycle state"):
        AgentJob(job_id="job_bad", task="bad", role="executor", status="mystery")
    with pytest.raises(LifecycleTransitionError, match="unknown team lifecycle state"):
        AgentTeam(team_id="team_bad", objective="bad", job_ids=[], status="mystery")


def test_single_worker_timeout_is_classified_as_level_one_degradation(tmp_path):
    class SlowClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            time.sleep(0.2)
            return "<final>late result</final>"

    def factory(role, worker, worker_workspace=None):
        root = worker_workspace or tmp_path
        return BioRuntime(
            model_client=SlowClient([]),
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    manager = AsyncMultiAgentOrchestrator(tmp_path, factory, max_concurrency=1)
    try:
        team = manager.start_team(
            "timeout classification",
            [
                {
                    "name": "slow",
                    "role": "explorer",
                    "task": "wait",
                    "timeout_seconds": 0.05,
                    "max_retries": 0,
                },
                {
                    "name": "also_slow",
                    "role": "planner",
                    "task": "wait",
                    "timeout_seconds": 0.05,
                    "max_retries": 0,
                },
            ],
            max_replans=0,
            global_timeout_seconds=2,
        )
        status = manager.wait_for_team(team.team_id, timeout=3)

        assert status["status"] == "failed"
        assert all(job["failure_type"] == "timeout" for job in status["jobs"])
        assert all(job["degradation_level"] == 1 for job in status["jobs"])
    finally:
        manager.shutdown()


def test_batch_failure_triggers_dynamic_replan_and_degraded_synthesis(tmp_path):
    class FailingClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            raise RuntimeError("simulated worker failure")

    def factory(role, worker, worker_workspace=None):
        root = worker_workspace or tmp_path
        client = (
            FailingClient([])
            if role == "explorer"
            else FakeModelClient([f"<final>{role} recovered evidence</final>"])
        )
        return BioRuntime(
            model_client=client,
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    manager = AsyncMultiAgentOrchestrator(tmp_path, factory, max_concurrency=2)
    try:
        team = manager.start_team(
            "recover failed exploration",
            [
                {
                    "name": "inspect",
                    "role": "explorer",
                    "task": "fail",
                    "max_retries": 0,
                },
                {
                    "name": "plan",
                    "role": "planner",
                    "task": "succeed",
                    "max_retries": 0,
                },
            ],
            failure_replan_threshold=0.5,
            max_replans=1,
            # Level-3 behavior has a dedicated short-deadline test below.
            # Keep this deadline generous so host load cannot mask Level 2.
            global_timeout_seconds=10,
        )
        status = manager.wait_for_team(team.team_id, timeout=11)

        assert status["status"] == "degraded"
        assert status["replan_count"] == 1
        assert status["degradation_level"] == 2
        assert status["replan_history"][0]["status"] == "applied"
        assert any(job["replan_generation"] == 1 and job["status"] == "completed" for job in status["jobs"])
        assert any(item["to_status"] == "replanning" for item in status["transitions"])
        assert "replan_count: 1" in status["synthesis"]
    finally:
        manager.shutdown()


def test_dynamic_replan_applies_a_validated_multi_node_dag_patch(tmp_path):
    class FailingClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            raise RuntimeError("simulated discovery failure")

    prompts = {}

    def factory(role, worker, worker_workspace=None):
        root = worker_workspace or tmp_path
        client = (
            FailingClient([])
            if role == "explorer"
            else FakeModelClient([f"<final>{role} patch evidence</final>"])
        )
        prompts.setdefault(role, []).append(client)
        return BioRuntime(
            model_client=client,
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    def strategy(team, jobs, failure_ratio):
        return ReplanPatch(
            reason="replace failed discovery with recovery and verification",
            nodes=(
                ReplanNode(
                    name="recover",
                    role="executor",
                    task="Recover from retained evidence",
                ),
                ReplanNode(
                    name="verify_recovery",
                    role="verifier",
                    task="Verify recovered evidence",
                    depends_on=("recover",),
                ),
            ),
        )

    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        factory,
        max_concurrency=2,
        replan_strategy=strategy,
    )
    try:
        team = manager.start_team(
            "recover and verify",
            [
                {"name": "discover", "role": "explorer", "max_retries": 0},
                {"name": "plan", "role": "planner"},
            ],
            failure_replan_threshold=0.5,
            max_replans=1,
            global_timeout_seconds=15,
        )
        status = manager.wait_for_team(team.team_id, timeout=16)

        assert status["status"] == "degraded"
        patched = [job for job in status["jobs"] if job["replan_generation"] == 1]
        assert [job["name"] for job in patched] == ["recover", "verify_recovery"]
        assert all(job["status"] == "completed" for job in patched)
        assert patched[1]["dependencies"] == [patched[0]["job_id"]]
        assert "executor patch evidence" in prompts["verifier"][0].prompts[0]
        assert len(status["replan_history"][0]["nodes"]) == 2
    finally:
        manager.shutdown()


def test_invalid_replan_patch_is_rejected_and_audited(tmp_path):
    class FailingClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            raise RuntimeError("simulated failure")

    def factory(role, worker, worker_workspace=None):
        root = worker_workspace or tmp_path
        client = (
            FailingClient([])
            if role == "explorer"
            else FakeModelClient(["<final>available evidence</final>"])
        )
        return BioRuntime(
            model_client=client,
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        factory,
        replan_strategy=lambda team, jobs, ratio: {
            "reason": "invalid patch",
            "nodes": [
                {
                    "name": "recover",
                    "role": "executor",
                    "task": "recover",
                    "depends_on": ["missing"],
                }
            ],
        },
    )
    try:
        team = manager.start_team(
            "reject invalid patch",
            [
                {"name": "discover", "role": "explorer", "max_retries": 0},
                {"name": "plan", "role": "planner"},
            ],
            max_replans=1,
            global_timeout_seconds=10,
        )
        status = manager.wait_for_team(team.team_id, timeout=11)

        assert status["status"] == "failed"
        assert status["replan_count"] == 0
        assert status["replan_history"][0]["status"] == "rejected"
        assert "unknown replan dependencies" in status["replan_history"][0]["error"]
    finally:
        manager.shutdown()


def test_global_timeout_forces_partial_synthesis(tmp_path):
    class SlowClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            time.sleep(0.3)
            return "<final>too late</final>"

    def factory(role, worker, worker_workspace=None):
        root = worker_workspace or tmp_path
        return BioRuntime(
            model_client=SlowClient([]),
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".scicore" / "sessions"),
            run_store=RunStore(root / ".scicore" / "runs"),
            approval_policy="auto",
            role=role,
            allow_orchestration=False,
        )

    manager = AsyncMultiAgentOrchestrator(tmp_path, factory, max_concurrency=2)
    try:
        team = manager.start_team(
            "force synthesis",
            [
                {"name": "one", "role": "explorer", "task": "wait"},
                {"name": "two", "role": "planner", "task": "wait"},
            ],
            max_replans=0,
            global_timeout_seconds=0.05,
        )
        status = manager.wait_for_team(team.team_id, timeout=2)

        assert status["status"] == "degraded"
        assert status["forced_synthesis"] is True
        assert status["degradation_level"] == 3
        assert "global timeout forced a partial synthesis" in status["synthesis"]
        assert any(item["to_status"] == "synthesizing" for item in status["transitions"])
    finally:
        manager.shutdown()
