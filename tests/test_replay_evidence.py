import json
from pathlib import Path

import pytest

from scicore.bio.replay import (
    ReplayError,
    build_scorecard,
    explain_artifact,
    extract_case,
    lint_case,
    materialize_source,
    run_case,
    validate_trace,
)
from scicore.bio.runtime import BioRuntime
from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext


def build_agent(root: Path, outputs: list[str]) -> BioRuntime:
    return BioRuntime(
        model_client=FakeModelClient(outputs),
        workspace=WorkspaceContext.build(root, repo_root_override=root),
        session_store=SessionStore(root / ".scicore" / "sessions"),
        run_store=RunStore(root / ".scicore" / "runs"),
        approval_policy="auto",
        role="executor",
        allow_orchestration=False,
    )


def test_source_snapshot_and_deliverable_survive_later_file_change(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"write_file","args":{"path":"analysis.py","content":"print(\'v1\')\\n"}}</tool>',
            "<final>created analysis</final>",
        ],
    )

    assert agent.ask("create analysis code") == "created analysis"
    run_dir = agent.run_store.run_dir(agent.current_task_state)
    trace_check = validate_trace(run_dir)
    assert trace_check["valid"] is True
    assert trace_check["canonical"] is True
    assert (run_dir / "evidence_index.json").is_file()
    assert (run_dir / "git_before.json").is_file()
    assert (run_dir / "git_after.json").is_file()
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert {"python", "r", "container", "packages"} <= set(environment)

    explanation = explain_artifact(run_dir, "analysis.py")
    assert explanation["manifest"]["execution"]["working_directory"] == str(tmp_path)
    assert explanation["manifest"]["execution"]["status"] == "ok"
    assert explanation["manifest"]["verifier"]["status"] == "not_run"
    sources = explanation["code_manifest"]["sources"]
    old_version = next(
        row for row in sources
        if row["path"] == "analysis.py" and row["version_role"] == "tool_output"
    )
    object_digest = old_version["object_id"].split(":", 1)[1]
    object_path = tmp_path / ".scicore" / "objects" / "sha256" / object_digest[:2] / object_digest
    assert object_path.read_text(encoding="utf-8") == "print('v1')\n"

    (tmp_path / "analysis.py").write_text("print('v2')\n", encoding="utf-8")
    restored = tmp_path / "restored"
    materialize_source(run_dir, restored)
    assert (restored / "analysis.py").read_text(encoding="utf-8") == "print('v1')\n"


def test_historical_trace_extracts_draft_case(tmp_path):
    (tmp_path / "input.tsv").write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"input.tsv","start":1,"end":20}}</tool>',
            "<final>input inspected</final>",
        ],
    )
    agent.ask("inspect input.tsv")
    run_dir = agent.run_store.run_dir(agent.current_task_state)

    result = extract_case(run_dir, tmp_path / "cases", "inspect-input")
    case_dir = Path(result["case_dir"])
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert case["status"] == "draft"
    assert case["input"]["prompt"] == "inspect input.tsv"
    assert (case_dir / "fixtures" / "input.tsv").is_file()
    assert lint_case(case_dir)["ready"] is False


def test_ready_case_runs_current_runtime_and_scores(tmp_path):
    case_dir = tmp_path / "case"
    (case_dir / "fixtures").mkdir(parents=True)
    case = {
        "schema_version": 1,
        "case_id": "simple-answer",
        "status": "ready",
        "input": {"prompt": "say ok"},
        "fixtures": [],
        "runtime": {
            "allowed_tools": ["list_files"],
            "max_steps": 3,
            "max_new_tokens": 100,
            "forbid_network": True,
        },
        "verification": {
            "final_nonempty": True,
            "final_contains": ["ok"],
            "final_regex": [],
            "expected_files": [],
            "commands": [],
            "expect_recovery": False,
        },
        "budgets": {"max_tool_steps": 1},
    }
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")

    result = run_case(
        case_dir,
        tmp_path / "replays",
        model_client=FakeModelClient(["<final>ok</final>"]),
    )
    assert result["verification"]["passed"] is True
    trace = [
        json.loads(line)
        for line in (Path(result["fresh_run_dir"]) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["source_case_id"] for row in trace} == {"simple-answer"}
    scorecard = build_scorecard([result], tmp_path / "scorecard.json")
    assert scorecard["summary"]["pass_rate"] == 1.0
    assert (tmp_path / "scorecard.md").is_file()


def test_materialize_refuses_nonempty_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ReplayError, match="empty directory"):
        materialize_source(tmp_path / "missing-run", target)
