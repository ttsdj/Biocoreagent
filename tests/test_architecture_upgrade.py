import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scicore.bio.analysis_router import AnalysisRoute
from scicore.bio.capabilities import default_capability_registry
from scicore.bio.orchestrator import AsyncMultiAgentOrchestrator
from scicore.bio.runtime import BioRuntime
from scicore.bio.workflow_ir import FinalSynthesis, WorkflowManifest, WorkflowManifestStore
from scicore.kernel.features.memory import LayeredMemory
from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext


def test_workflow_ir_rejects_missing_fields_and_unverified_completion(tmp_path):
    with pytest.raises(ValidationError):
        WorkflowManifest.model_validate({})

    route = AnalysisRoute(
        analysis_type="bulk_rnaseq",
        intent="run_analysis",
        risk_level="medium",
        requires_plan=True,
        preferred_backend="omicverse",
        fallback_allowed=True,
    )
    store = WorkflowManifestStore(tmp_path)
    manifest = store.create("run RNA-seq", route)
    manifest.final = FinalSynthesis(
        status="completed",
        summary="claimed completion",
        evidence_paths=["missing.csv"],
    )

    with pytest.raises(ValidationError, match="passed verification"):
        store.save(manifest)


def test_capability_contract_downgrades_missing_artifacts():
    class Runtime:
        def execute_tool(self, name, parameters):
            return SimpleNamespace(content=json.dumps({"status": "completed"}))

    registry = default_capability_registry()
    invocation, execution, verification, _ = registry.invoke(
        Runtime(),
        "omicverse_bulk_deg",
        {
            "count_matrix_path": "counts.tsv",
            "target_group": "el",
            "output_dir": "outputs",
        },
    )

    assert invocation.deterministic is True
    assert execution.status == "failed"
    assert verification.status == "failed"
    assert any(check.name.startswith("artifact:") for check in verification.checks)


def test_controlled_analysis_persists_verified_workflow_manifest(tmp_path):
    table = tmp_path / "protein_intensity.csv"
    table.write_text("protein,el1,el2\nP1,10,12\n", encoding="utf-8")
    agent = BioRuntime(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".scicore" / "sessions"),
        run_store=RunStore(tmp_path / ".scicore" / "runs"),
        approval_policy="auto",
        role="executor",
        allow_orchestration=False,
    )

    agent.ask(f'"{table}" run proteomics analysis')

    manifests = list((tmp_path / ".scicore" / "workflows").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = WorkflowManifest.model_validate_json(manifests[0].read_text(encoding="utf-8"))
    assert manifest.route.analysis_type == "proteomics"
    assert manifest.plan.status == "approved"
    assert manifest.executions[0].backend == "generic_script"
    assert manifest.verification.status == "passed"
    assert manifest.final.status == "completed"
    assert manifest.final.evidence_paths


def test_process_worker_hard_timeout_prevents_late_side_effect(tmp_path):
    def command_factory(job, workspace):
        script = workspace / "slow_worker.py"
        script.write_text(
            "from pathlib import Path\n"
            "import time\n"
            "time.sleep(1.5)\n"
            "Path(__file__).with_name('late_marker.txt').write_text('late')\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        agent_factory=lambda *args: None,
        process_command_factory=command_factory,
        default_execution_mode="process",
        watchdog_interval_seconds=0.05,
    )
    try:
        job = manager.start_job(
            "hang",
            "explorer",
            timeout_seconds=0.2,
            hard_timeout=True,
            max_retries=0,
        )
        status = manager.wait_for_job(job.job_id, timeout=3)
        time.sleep(0.3)

        assert status["status"] == "failed"
        assert status["execution_mode"] == "process"
        assert status["failure_type"] == "timeout"
        assert manager._processes == {}
        assert not (Path(status["workspace_path"]) / "late_marker.txt").exists()
    finally:
        manager.shutdown()


def test_artifact_write_lease_blocks_conflicting_agent(tmp_path):
    class Worker:
        def __init__(self):
            self.session = {"id": "session-" + str(time.time_ns())}
            self.current_task_state = SimpleNamespace(run_id="run-" + str(time.time_ns()))
            self.progress_callback = None

        def ask(self, prompt):
            time.sleep(0.35)
            return "done"

    def factory(role, worker, worker_workspace=None):
        return Worker()

    manager = AsyncMultiAgentOrchestrator(
        tmp_path,
        factory,
        max_concurrency=2,
        watchdog_interval_seconds=0.05,
    )
    output_path = tmp_path / "shared.csv"
    try:
        first = manager.start_job(
            "first",
            "executor",
            artifact_paths=[str(output_path)],
            timeout_seconds=2,
        )
        deadline = time.time() + 1
        while time.time() < deadline:
            active = [
                lease
                for lease in manager.store.list_write_leases()
                if lease.get("expires_at", 0) > time.time()
            ]
            if active:
                break
            time.sleep(0.01)
        second = manager.start_job(
            "second",
            "executor",
            artifact_paths=[str(output_path)],
            timeout_seconds=2,
        )

        second_status = manager.wait_for_job(second.job_id, timeout=3)
        first_status = manager.wait_for_job(first.job_id, timeout=3)
        assert first_status["status"] == "completed"
        assert second_status["status"] == "blocked"
        assert "write lease conflict" in second_status["error"]
    finally:
        manager.shutdown()


def test_four_dimension_memory_feedback_forgetting_and_supersession(tmp_path):
    memory = LayeredMemory(workspace_root=tmp_path)
    memory.promote_durable(
        [
            ("dependency-facts", "DESeq2 requires a compatible rlang version."),
            ("user-preferences", "Reports should be written in Chinese."),
        ]
    )

    results = memory.retrieval_candidates(
        "DESeq2 rlang dependency",
        scope="bulk-rnaseq dependency",
        role="executor",
    )
    scoring = results[0]["retrieval"]
    assert scoring["scoring_model"] == "four_dimension_v1"
    assert set(("relevance", "recency", "reliability", "scope_fit")) <= set(scoring)
    assert sum(scoring["weights"].values()) == pytest.approx(1.0, abs=1e-5)

    updated_reliability = memory.record_retrieval_feedback(
        results[0]["memory_id"],
        verified=True,
        success=True,
    )
    assert updated_reliability > scoring["reliability"]

    with sqlite3.connect(memory.durable_store.db_path) as connection:
        connection.execute(
            """
            UPDATE memories
            SET created_at = ?, last_accessed_at = '', reliability = 0.05, access_count = 0
            WHERE topic = 'dependency-facts'
            """,
            ("2020-01-01T00:00:00+00:00",),
        )
        connection.commit()
    forgotten = memory.apply_forgetting(
        threshold=0.2,
        at_timestamp=datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp(),
    )
    assert any(item["topic"] == "dependency-facts" for item in forgotten)
    assert not any(item["topic"] == "user-preferences" for item in forgotten)

    memory.promote_durable(
        [("project-conventions", "Output files should use TSV format.")]
    )
    _, superseded = memory.promote_durable(
        [("project-conventions", "Output files should use CSV format.")]
    )
    assert superseded
    with sqlite3.connect(memory.durable_store.db_path) as connection:
        state = connection.execute(
            "SELECT state FROM memories WHERE text = ?",
            ("Output files should use TSV format.",),
        ).fetchone()[0]
    assert state == "superseded"
