"""Trace validation, replay case extraction, fresh runs, and scorecards."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from corecoder.policy import PolicyDecision, evaluate_command
from pico.providers.clients import FakeModelClient
from pico.run_store import RunStore
from pico.session_store import SessionStore
from pico.workspace import WorkspaceContext, now

from .audit import redact
from .evidence import (
    MAX_CODE_BYTES,
    SECRET_FILE_RE,
    ContentAddressedStore,
    sha256_file,
    write_json_atomic,
)
from .runtime import BioPico


REPLAY_CASE_SCHEMA_VERSION = 1
SCORECARD_SCHEMA_VERSION = 1
FORBIDDEN_REPLAY_TOOLS = {
    "ssh_bash",
    "agent_start",
    "agent_team_start",
    "pubmed_search",
    "pubmed_fetch_details",
    "pubmed_literature_review",
}
NETWORK_COMMAND_RE = re.compile(
    r"(?i)(https?://|\bcurl\b|\bwget\b|\bssh\b|\bscp\b|\brsync\b|\bnc\b|\btelnet\b)"
)


class ReplayError(RuntimeError):
    pass


class LocalReplayBioPico(BioPico):
    """BioPico constrained to local, non-network replay execution."""

    def validate_tool(self, name, args):
        super().validate_tool(name, args)
        if name == "run_shell":
            command = str((args or {}).get("command", ""))
            if NETWORK_COMMAND_RE.search(command):
                raise ValueError("network and remote commands are forbidden in local replay")

    def _try_csv_export_shortcut(self, user_message):
        # 确定性回放必须走纯工具循环，不能短路到需要真实后端/真实数据的域捷径。
        return None

    def _try_bio_shortcut(self, user_message, force=False):
        return None

    def _try_fallback_analysis(self, user_message, route):
        return None


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_trace(path_or_run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_run_dir)
    if path.is_dir():
        path = path / "trace.jsonl"
    if not path.exists():
        raise ReplayError(f"trace does not exist: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"invalid JSON at trace line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReplayError(f"trace line {number} is not an object")
        rows.append(row)
    return rows


def validate_trace(path_or_run_dir: str | Path) -> dict[str, Any]:
    original = Path(path_or_run_dir)
    try:
        rows = read_trace(path_or_run_dir)
    except ReplayError as exc:
        return {"valid": False, "canonical": False, "errors": [str(exc)], "warnings": []}
    errors = []
    warnings = []
    canonical = bool(rows) and all(int(row.get("schema_version", 0) or 0) >= 3 for row in rows)
    terminals = [
        row for row in rows
        if _event_type(row) == "run_finished"
    ]
    if len(terminals) != 1:
        errors.append(f"expected exactly one run_finished event, found {len(terminals)}")
    if canonical:
        sequences = [int(row.get("sequence", 0) or 0) for row in rows]
        if sequences != list(range(1, len(rows) + 1)):
            errors.append("canonical trace sequence is not contiguous")
        event_ids = [str(row.get("event_id", "")) for row in rows]
        if any(not value for value in event_ids) or len(set(event_ids)) != len(event_ids):
            errors.append("canonical trace event IDs are missing or duplicated")
        trace_ids = {str(row.get("trace_id", "")) for row in rows}
        if len(trace_ids) != 1 or "" in trace_ids:
            errors.append("canonical trace must contain one non-empty trace_id")
    else:
        warnings.append("legacy trace: replay extraction is best-effort")
    evidence = None
    if original.is_dir() and (original / "evidence_index.json").is_file():
        evidence = validate_evidence_index(original)
        if not evidence["valid"]:
            errors.extend(evidence["errors"])
    return {
        "valid": not errors,
        "canonical": canonical,
        "event_count": len(rows),
        "terminal_event_count": len(terminals),
        "errors": errors,
        "warnings": warnings,
        "evidence_index": evidence,
    }


def validate_evidence_index(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    index_path = run_dir / "evidence_index.json"
    if not index_path.is_file():
        return {"valid": False, "errors": ["evidence_index.json is missing"]}
    try:
        index = load_json(index_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"invalid evidence index: {exc}"]}
    errors = []
    for relative, descriptor in (index.get("files") or {}).items():
        path = run_dir / relative
        if not path.is_file():
            errors.append(f"indexed evidence file is missing: {relative}")
            continue
        if sha256_file(path) != descriptor.get("sha256"):
            errors.append(f"indexed evidence hash mismatch: {relative}")
    return {"valid": not errors, "errors": errors, "indexed_file_count": len(index.get("files") or {})}


def extract_case(
    run_dir: str | Path,
    output_root: str | Path,
    case_id: str,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,79}", case_id):
        raise ReplayError("case_id must be 2-80 safe filename characters")
    task_state_path = run_dir / "task_state.json"
    if not task_state_path.exists():
        raise ReplayError(f"missing task state: {task_state_path}")
    task_state = load_json(task_state_path)
    trace = read_trace(run_dir)
    source_root = _source_workspace(run_dir)
    case_dir = Path(output_root).resolve() / case_id
    if case_dir.exists():
        raise ReplayError(f"case directory already exists: {case_dir}")
    fixtures_dir = case_dir / "fixtures"
    fixtures_dir.mkdir(parents=True)

    candidates = _input_path_candidates(trace, source_root)
    fixtures = []
    omissions = []
    for candidate in candidates:
        relative = _relative_inside(candidate, source_root)
        if relative is None:
            omissions.append({"path": str(candidate), "reason": "outside_workspace"})
            continue
        if SECRET_FILE_RE.search(candidate.name):
            omissions.append({"path": relative, "reason": "sensitive_filename"})
            continue
        if not candidate.is_file():
            omissions.append({"path": relative, "reason": "missing_or_not_file"})
            continue
        if candidate.stat().st_size > MAX_CODE_BYTES:
            omissions.append({"path": relative, "reason": "larger_than_10_mib"})
            continue
        destination = fixtures_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_redacted_fixture(candidate, destination)
        fixtures.append(
            {
                "path": relative,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    allowed_tools = sorted(
        {
            str(row.get("name", ""))
            for row in trace
            if _event_type(row) == "tool_call_completed" and row.get("name")
        }
        - FORBIDDEN_REPLAY_TOOLS
    )
    if not allowed_tools:
        allowed_tools = ["list_files"]
    expected_files = _historical_deliverables(run_dir)
    case = {
        "schema_version": REPLAY_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "draft",
        "title": case_id.replace("-", " ").replace("_", " "),
        "tags": ["historical-trace"],
        "source": {
            "run_id": str(task_state.get("run_id", run_dir.name)),
            "run_dir": str(run_dir),
            "trace_sha256": sha256_file(run_dir / "trace.jsonl"),
            "extracted_at": now(),
        },
        "input": {"prompt": str(redact(_trace_prompt(trace) or task_state.get("user_request", "")))},
        "fixtures": fixtures,
        "runtime": {
            "allowed_tools": allowed_tools,
            "max_steps": max(int(task_state.get("tool_steps", 0)) + 3, 6),
            "max_new_tokens": 4096,
            "forbid_network": True,
        },
        "verification": {
            "final_nonempty": True,
            "final_contains": [],
            "final_regex": [],
            "expected_files": expected_files,
            "commands": [],
            "expect_recovery": False,
        },
        "budgets": {
            "max_tool_steps": max(int(task_state.get("tool_steps", 0)) + 3, 6),
        },
        "extraction": {
            "omissions": omissions,
            "requires_human_verifier_review": True,
        },
    }
    write_json_atomic(case_dir / "case.json", case)
    (case_dir / "README.md").write_text(
        f"# {case_id}\n\n"
        "This case was extracted as `draft`. Review fixtures and verification "
        "rules, resolve every omission, set `requires_human_verifier_review` "
        "to false, then explicitly change `status` to `ready`.\n",
        encoding="utf-8",
    )
    return {"case_dir": str(case_dir), "case": case}


def lint_case(case_dir_or_file: str | Path) -> dict[str, Any]:
    case_file = _case_file(case_dir_or_file)
    errors = []
    warnings = []
    try:
        case = load_json(case_file)
    except Exception as exc:
        return {"valid": False, "ready": False, "errors": [str(exc)], "warnings": []}
    if int(case.get("schema_version", 0) or 0) != REPLAY_CASE_SCHEMA_VERSION:
        errors.append("unsupported replay case schema_version")
    if not str(case.get("case_id", "")):
        errors.append("case_id is required")
    if not str((case.get("input") or {}).get("prompt", "")).strip():
        errors.append("input.prompt is required")
    runtime = case.get("runtime") or {}
    tools = set(runtime.get("allowed_tools", []) or [])
    forbidden = sorted(tools & FORBIDDEN_REPLAY_TOOLS)
    if forbidden:
        errors.append(f"external/orchestration tools are forbidden in local replay: {forbidden}")
    if not runtime.get("forbid_network", True):
        errors.append("runtime.forbid_network must be true for v1 local replay")
    fixture_root = case_file.parent / "fixtures"
    for fixture in case.get("fixtures", []) or []:
        path = fixture_root / str(fixture.get("path", ""))
        if not path.is_file():
            errors.append(f"fixture is missing: {fixture.get('path', '')}")
            continue
        if fixture.get("sha256") and sha256_file(path) != fixture["sha256"]:
            errors.append(f"fixture hash mismatch: {fixture.get('path', '')}")
    omissions = ((case.get("extraction") or {}).get("omissions") or [])
    if omissions:
        errors.append(f"case extraction has {len(omissions)} unresolved omissions")
    if (case.get("extraction") or {}).get("requires_human_verifier_review"):
        errors.append("human verifier review has not been acknowledged")
    if not isinstance(case.get("verification"), dict):
        errors.append("verification rules are required")
    ready = case.get("status") == "ready"
    if not ready:
        warnings.append("case status is not ready")
    return {
        "valid": not errors,
        "ready": ready and not errors,
        "errors": errors,
        "warnings": warnings,
        "case": case,
        "case_file": str(case_file),
    }


def run_case(
    case_dir_or_file: str | Path,
    output_root: str | Path,
    *,
    model_client: Any,
) -> dict[str, Any]:
    lint = lint_case(case_dir_or_file)
    if not lint["ready"]:
        raise ReplayError("case is not runnable: " + "; ".join(lint["errors"] + lint["warnings"]))
    case = lint["case"]
    case_file = Path(lint["case_file"])
    replay_id = "replay_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8]
    replay_dir = Path(output_root).resolve() / replay_id / case["case_id"]
    workspace = replay_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    fixture_root = case_file.parent / "fixtures"
    for fixture in case.get("fixtures", []) or []:
        source = fixture_root / fixture["path"]
        destination = workspace / fixture["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    runtime = case.get("runtime") or {}
    allowed_tools = set(runtime.get("allowed_tools", []) or []) - FORBIDDEN_REPLAY_TOOLS
    if not allowed_tools:
        allowed_tools = {"list_files"}
    agent = LocalReplayBioPico(
        model_client=model_client,
        workspace=WorkspaceContext.build(workspace, repo_root_override=workspace),
        session_store=SessionStore(workspace / ".biocoreagent" / "sessions"),
        run_store=RunStore(workspace / ".biocoreagent" / "runs"),
        approval_policy="auto",
        max_steps=int(runtime.get("max_steps", 6)),
        max_new_tokens=int(runtime.get("max_new_tokens", 4096)),
        allowed_tools=allowed_tools,
        trace_context={
            "source_case_id": case["case_id"],
            "source_run_id": str((case.get("source") or {}).get("run_id", "")),
            "replay_id": replay_id,
        },
        role="executor",
        allow_orchestration=False,
    )
    answer = agent.ask(str(case["input"]["prompt"]))
    fresh_run_dir = agent.run_store.run_dir(agent.current_task_state)
    verification = verify_fresh_run(case, fresh_run_dir, workspace, answer)
    evidence_index = load_json(fresh_run_dir / "evidence_index.json")
    write_json_atomic(replay_dir / "verification.json", verification)
    result = {
        "schema_version": 1,
        "replay_id": replay_id,
        "case_id": case["case_id"],
        "source_run_id": str((case.get("source") or {}).get("run_id", "")),
        "fresh_run_id": agent.current_task_state.run_id,
        "fresh_run_dir": str(fresh_run_dir),
        "workspace": str(workspace),
        "answer": answer,
        "runtime_identity": evidence_index.get("runtime_identity", {}),
        "verification": verification,
    }
    write_json_atomic(replay_dir / "case_score.json", result)
    return result


def run_suite(
    suite_dir: str | Path,
    output_root: str | Path,
    *,
    model_client: Any,
) -> dict[str, Any]:
    suite_dir = Path(suite_dir).resolve()
    case_files = sorted(suite_dir.glob("*/case.json"))
    if not case_files:
        raise ReplayError(f"suite contains no replay cases: {suite_dir}")
    results = []
    for case_file in case_files:
        try:
            results.append(run_case(case_file, output_root, model_client=model_client))
        except (ReplayError, RuntimeError, ValueError, OSError) as exc:
            case_id = case_file.parent.name
            results.append(
                {
                    "schema_version": 1,
                    "replay_id": "",
                    "case_id": case_id,
                    "fresh_run_id": "",
                    "fresh_run_dir": "",
                    "runtime_identity": {},
                    "verification": {
                        "passed": False,
                        "hard_gates_passed": False,
                        "score": 0.0,
                        "tool_steps": 0,
                        "stop_reason": "replay_error",
                        "error": str(exc),
                    },
                }
            )
    scorecard_path = Path(output_root).resolve() / (
        "suite_scorecard_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    scorecard = build_scorecard(results, scorecard_path)
    return {
        "suite": str(suite_dir),
        "results": results,
        "scorecard": scorecard,
        "scorecard_path": str(scorecard_path),
    }


def run_suite_deterministic(
    suite_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run every case in a suite, each driven by its OWN fake_outputs.json.

    与 run_suite 的唯一区别：run_suite 让所有 case 共享同一个 model_client（模拟一次
    长会话里的多次 ask），而 deterministic 套件里每个 case 的模型输出都固化在自己的
    fake_outputs.json，因此这里为每个 case 单独构建一个 FakeModelClient，保证「这条 case
    的输出序列」完全由它自己决定——这正是「每条用例都能独立、确定性地跑通」的语义。
    """
    suite_dir = Path(suite_dir).resolve()
    case_files = sorted(suite_dir.glob("*/case.json"))
    if not case_files:
        raise ReplayError(f"suite contains no replay cases: {suite_dir}")
    missing = [
        str(path.parent / "fake_outputs.json")
        for path in case_files
        if not (path.parent / "fake_outputs.json").is_file()
    ]
    if missing:
        raise ReplayError(
            "deterministic suite requires a fake_outputs.json per case; missing: "
            + ", ".join(missing)
        )
    results = []
    for case_file in case_files:
        case_id = case_file.parent.name
        try:
            client = fake_model_from_file(case_file.parent / "fake_outputs.json")
            results.append(run_case(case_file, output_root, model_client=client))
        except (ReplayError, RuntimeError, ValueError, OSError) as exc:
            results.append(
                {
                    "schema_version": 1,
                    "replay_id": "",
                    "case_id": case_id,
                    "fresh_run_id": "",
                    "fresh_run_dir": "",
                    "runtime_identity": {},
                    "verification": {
                        "passed": False,
                        "hard_gates_passed": False,
                        "score": 0.0,
                        "tool_steps": 0,
                        "stop_reason": "replay_error",
                        "error": str(exc),
                    },
                }
            )
    scorecard_path = Path(output_root).resolve() / (
        "suite_scorecard_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    scorecard = build_scorecard(results, scorecard_path)
    return {
        "suite": str(suite_dir),
        "deterministic": True,
        "results": results,
        "scorecard": scorecard,
        "scorecard_path": str(scorecard_path),
    }


def verify_fresh_run(
    case: dict[str, Any],
    fresh_run_dir: str | Path,
    workspace: str | Path,
    answer: str,
) -> dict[str, Any]:
    fresh_run_dir = Path(fresh_run_dir)
    workspace = Path(workspace)
    checks = []

    trace_result = validate_trace(fresh_run_dir)
    checks.append(_check("trace_integrity", trace_result["valid"], trace_result))
    events = read_trace(fresh_run_dir) if trace_result["valid"] else []
    types = {_event_type(row) for row in events}
    security_events = [
        row for row in events
        if str(row.get("security_event_type", "")) in {"path_escape", "read_only_block"}
    ]
    checks.append(_check("security_boundary", not security_events, {"events": security_events}))

    verification = case.get("verification") or {}
    if verification.get("final_nonempty", True):
        checks.append(_check("final_nonempty", bool(str(answer).strip()), {}))
    for text in verification.get("final_contains", []) or []:
        checks.append(_check(f"final_contains:{text}", str(text) in answer, {}))
    for pattern in verification.get("final_regex", []) or []:
        checks.append(_check(f"final_regex:{pattern}", bool(re.search(pattern, answer)), {}))

    expected_file_checks = []
    for expected in verification.get("expected_files", []) or []:
        path = workspace / str(expected.get("path", ""))
        passed = path.is_file() if expected.get("must_exist", True) else True
        details = {"path": str(path), "exists": path.is_file()}
        if passed and expected.get("sha256"):
            details["actual_sha256"] = sha256_file(path)
            passed = details["actual_sha256"] == expected["sha256"]
        if passed and expected.get("json_keys"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                missing = [key for key in expected["json_keys"] if key not in payload]
                details["missing_json_keys"] = missing
                passed = not missing
            except Exception as exc:
                details["json_error"] = str(exc)
                passed = False
        expected_file_checks.append(_check(f"expected_file:{expected.get('path', '')}", passed, details))
    checks.extend(expected_file_checks)

    command_checks = []
    for command in verification.get("commands", []) or []:
        decision, reason = evaluate_command(str(command))
        if decision != PolicyDecision.ALLOW or NETWORK_COMMAND_RE.search(str(command)):
            command_checks.append(
                _check(
                    f"verifier_command:{command}",
                    False,
                    {"policy_decision": str(decision), "reason": reason},
                )
            )
            continue
        process = subprocess.run(
            str(command),
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        command_checks.append(
            _check(
                f"verifier_command:{command}",
                process.returncode == 0,
                {
                    "returncode": process.returncode,
                    "stdout": process.stdout[-2000:],
                    "stderr": process.stderr[-2000:],
                },
            )
        )
    checks.extend(command_checks)

    task_state = load_json(fresh_run_dir / "task_state.json")
    result_checks = [
        row for row in checks
        if row["name"].startswith(("final_", "expected_file:", "verifier_command:"))
    ]
    outcome_ratio = _pass_ratio(result_checks)
    outcome_score = round(50 * outcome_ratio, 2)

    evidence_exists = (fresh_run_dir / "evidence_index.json").is_file()
    deliverables_exist = any((fresh_run_dir / "deliverables").glob("*/manifest.json"))
    artifact_required = bool(verification.get("expected_files"))
    provenance_ratio = (int(evidence_exists) + int(deliverables_exist or not artifact_required)) / 2
    provenance_score = round(20 * provenance_ratio, 2)

    required_trace = {
        "run_started", "user_input_recorded", "model_requested",
        "model_response_parsed", "final_delivery", "run_finished",
    }
    trace_score = round(15 * len(types & required_trace) / len(required_trace), 2)
    expect_recovery = bool(verification.get("expect_recovery"))
    recovery_score = 10.0 if not expect_recovery else (10.0 if "recovery_completed" in types else 0.0)
    max_tool_steps = int((case.get("budgets") or {}).get("max_tool_steps", 0) or 0)
    within_budget = not max_tool_steps or int(task_state.get("tool_steps", 0)) <= max_tool_steps
    budget_score = 5.0 if within_budget else 0.0

    total = round(outcome_score + provenance_score + trace_score + recovery_score + budget_score, 2)
    hard_gate_checks = [
        row for row in checks
        if row["name"] in {"trace_integrity", "security_boundary"}
        or row["name"].startswith(("final_", "expected_file:", "verifier_command:"))
    ]
    hard_gates_passed = all(row["passed"] for row in hard_gate_checks)
    passed = hard_gates_passed and total >= 80
    completed_tool_events = [row for row in events if _event_type(row) == "tool_call_completed"]
    rejected_tools = sum(
        1 for row in completed_tool_events
        if str(row.get("tool_status", "")) in {"rejected", "error", "partial_success"}
    )
    terminal_event = next((row for row in reversed(events) if _event_type(row) == "run_finished"), {})
    return {
        "schema_version": 1,
        "passed": passed,
        "hard_gates_passed": hard_gates_passed,
        "score": total,
        "scores": {
            "outcome": outcome_score,
            "provenance": provenance_score,
            "trace": trace_score,
            "recovery": recovery_score,
            "budget": budget_score,
        },
        "checks": checks,
        "tool_steps": int(task_state.get("tool_steps", 0)),
        "attempts": int(task_state.get("attempts", 0)),
        "stop_reason": str(task_state.get("stop_reason", "")),
        "run_duration_ms": int(terminal_event.get("run_duration_ms", 0) or 0),
        "tool_rejection_count": rejected_tools,
        "recovery_triggered": "recovery_started" in types,
        "recovery_completed": "recovery_completed" in types,
    }


def build_scorecard(results: list[dict[str, Any]], output: str | Path | None = None) -> dict[str, Any]:
    rows = []
    for result in results:
        verification = result.get("verification") or {}
        rows.append(
            {
                "case_id": result.get("case_id", ""),
                "replay_id": result.get("replay_id", ""),
                "fresh_run_id": result.get("fresh_run_id", ""),
                "passed": bool(verification.get("passed")),
                "score": float(verification.get("score", 0)),
                "tool_steps": int(verification.get("tool_steps", 0)),
                "stop_reason": verification.get("stop_reason", ""),
                "run_duration_ms": int(verification.get("run_duration_ms", 0)),
                "tool_rejection_count": int(verification.get("tool_rejection_count", 0)),
                "recovery_triggered": bool(verification.get("recovery_triggered")),
                "recovery_completed": bool(verification.get("recovery_completed")),
                "runtime_identity": result.get("runtime_identity", {}),
            }
        )
    passed = sum(1 for row in rows if row["passed"])
    failure_categories = {}
    for row in rows:
        if row["passed"]:
            continue
        category = str(row.get("stop_reason") or "verification_failed")
        failure_categories[category] = failure_categories.get(category, 0) + 1
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "created_at": now(),
        "summary": {
            "total_cases": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "pass_rate": passed / len(rows) if rows else 0.0,
            "average_score": sum(row["score"] for row in rows) / len(rows) if rows else 0.0,
            "average_tool_steps": sum(row["tool_steps"] for row in rows) / len(rows) if rows else 0.0,
            "average_run_duration_ms": sum(row["run_duration_ms"] for row in rows) / len(rows) if rows else 0.0,
            "tool_rejection_count": sum(row["tool_rejection_count"] for row in rows),
            "recovery_success_rate": (
                sum(1 for row in rows if row["recovery_completed"])
                / sum(1 for row in rows if row["recovery_triggered"])
                if any(row["recovery_triggered"] for row in rows)
                else 1.0
            ),
            "failure_category_counts": failure_categories,
        },
        "cases": rows,
        "runtime_identities": [
            row["runtime_identity"]
            for row in rows
            if row.get("runtime_identity")
        ],
    }
    if output is not None:
        output = Path(output)
        write_json_atomic(output, scorecard)
        output.with_suffix(".md").write_text(_render_scorecard(scorecard), encoding="utf-8")
    return scorecard


def compare_scorecard(scorecard: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current = scorecard.get("summary") or {}
    previous = baseline.get("summary") or {}
    current_cases = {row["case_id"]: row for row in scorecard.get("cases", [])}
    baseline_cases = {row["case_id"]: row for row in baseline.get("cases", [])}
    regressions = [
        case_id
        for case_id in sorted(set(current_cases) & set(baseline_cases))
        if baseline_cases[case_id].get("passed") and not current_cases[case_id].get("passed")
    ]
    result = {
        "comparable_case_count": len(set(current_cases) & set(baseline_cases)),
        "pass_rate_delta": float(current.get("pass_rate", 0)) - float(previous.get("pass_rate", 0)),
        "average_score_delta": float(current.get("average_score", 0)) - float(previous.get("average_score", 0)),
        "average_tool_steps_delta": float(current.get("average_tool_steps", 0)) - float(previous.get("average_tool_steps", 0)),
        "case_regressions": regressions,
        "runtime_identity_changed": (
            scorecard.get("runtime_identities", [])
            != baseline.get("runtime_identities", [])
        ),
    }
    result["passed"] = not regressions and result["pass_rate_delta"] >= 0
    return result


def explain_artifact(search_root: str | Path, artifact: str) -> dict[str, Any]:
    root = Path(search_root)
    matches = []
    for manifest_path in root.rglob("deliverables/*/manifest.json"):
        manifest = load_json(manifest_path)
        descriptor = manifest.get("artifact") or {}
        if artifact in {
            str(manifest.get("deliverable_id", "")),
            str(descriptor.get("path", "")),
            Path(str(descriptor.get("path", ""))).name,
        }:
            bundle = manifest_path.parent
            matches.append(
                {
                    "manifest": manifest,
                    "environment": load_json(bundle / "environment.json"),
                    "lineage": load_json(bundle / "lineage.json"),
                    "code_manifest": load_json(bundle / "code_manifest.json"),
                    "bundle": str(bundle),
                }
            )
    if not matches:
        raise ReplayError(f"artifact was not found: {artifact}")
    if len(matches) > 1:
        return {"ambiguous": True, "matches": matches}
    return matches[0]


def source_versions(run_dir: str | Path, *, at_event: str = "") -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest = run_dir / "source_manifest.jsonl"
    if not manifest.exists():
        raise ReplayError(f"source manifest does not exist: {manifest}")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not at_event:
        return rows
    events = read_trace(run_dir)
    event_sequence = {str(row.get("event_id", "")): int(row.get("sequence", 0)) for row in events}
    if at_event not in event_sequence:
        raise ReplayError(f"event does not exist in trace: {at_event}")
    target = event_sequence[at_event]
    return [
        row for row in rows
        if event_sequence.get(str(row.get("captured_at_event", "")), target) <= target
    ]


def materialize_source(
    run_dir: str | Path,
    target: str | Path,
    *,
    at_event: str = "",
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    target = Path(target).resolve()
    if target.exists() and any(target.iterdir()):
        raise ReplayError("source materialization target must be an empty directory")
    target.mkdir(parents=True, exist_ok=True)
    state_root = run_dir.parent.parent
    store = ContentAddressedStore(state_root / "objects")
    rows = source_versions(run_dir, at_event=at_event)
    latest = {}
    for row in rows:
        latest[str(row["path"])] = row
    written = []
    for relative, row in latest.items():
        destination = (target / relative).resolve()
        try:
            destination.relative_to(target)
        except ValueError as exc:
            raise ReplayError(f"unsafe source path: {relative}") from exc
        source = store.object_path(row["object_id"])
        if not source.is_file():
            raise ReplayError(f"source object is missing: {row['object_id']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written.append(relative)
    return {"target": str(target), "written": sorted(written), "at_event": at_event}


def fake_model_from_file(path: str | Path) -> FakeModelClient:
    payload = Path(path).read_text(encoding="utf-8")
    try:
        decoded = json.loads(payload)
        outputs = decoded if isinstance(decoded, list) else [str(decoded)]
    except json.JSONDecodeError:
        outputs = [payload]
    return FakeModelClient(outputs)


def _event_type(row: dict[str, Any]) -> str:
    value = str(row.get("event_type") or row.get("event") or "")
    aliases = {
        "model_parsed": "model_response_parsed",
        "tool_executed": "tool_call_completed",
        "recovery_gate_triggered": "recovery_completed",
    }
    return aliases.get(value, value)


def _source_workspace(run_dir: Path) -> Path:
    for name in ("git_before.json", "evidence_index.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = load_json(path)
        candidate = payload.get("repo_root") or payload.get("workspace_root")
        if candidate:
            return Path(candidate).resolve()
    return run_dir.parent.parent.parent.resolve()


def _input_path_candidates(trace: list[dict[str, Any]], source_root: Path) -> list[Path]:
    paths = set()
    key_re = re.compile(r"(?i)(^|_)(path|file|input|matrix|metadata|counts|table)(_|$)")
    for row in trace:
        if _event_type(row) != "tool_call_completed":
            continue
        for key, value in (row.get("args") or {}).items():
            if not key_re.search(str(key)) or not isinstance(value, str):
                continue
            if not row.get("read_only", False) and not re.search(
                r"(?i)(input|source|matrix|metadata|counts|table)", str(key)
            ):
                continue
            path = Path(value)
            paths.add((path if path.is_absolute() else source_root / path).resolve())
    return sorted(paths, key=str)


def _relative_inside(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _historical_deliverables(run_dir: Path) -> list[dict[str, Any]]:
    expected = []
    for manifest_path in run_dir.glob("deliverables/*/manifest.json"):
        artifact = (load_json(manifest_path).get("artifact") or {})
        if artifact.get("path"):
            # Hash is deliberately omitted: behavior-equivalent outputs may differ.
            expected.append({"path": artifact["path"], "must_exist": True})
    return sorted(expected, key=lambda item: item["path"])


def _trace_prompt(trace: list[dict[str, Any]]) -> str:
    for row in trace:
        if _event_type(row) == "user_input_recorded":
            return str(row.get("user_input", ""))
    for row in trace:
        if _event_type(row) == "run_started":
            return str(row.get("user_request", ""))
    return ""


def _copy_redacted_fixture(source: Path, destination: Path) -> None:
    try:
        text = source.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        shutil.copy2(source, destination)
        return
    destination.write_text(str(redact(text)), encoding="utf-8")


def _case_file(value: str | Path) -> Path:
    path = Path(value).resolve()
    return path / "case.json" if path.is_dir() else path


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _pass_ratio(checks: list[dict[str, Any]]) -> float:
    if not checks:
        return 1.0
    return sum(1 for row in checks if row["passed"]) / len(checks)


def _render_scorecard(scorecard: dict[str, Any]) -> str:
    summary = scorecard["summary"]
    lines = [
        "# BioCoreAgent Replay Scorecard",
        "",
        f"- Cases: {summary['total_cases']}",
        f"- Passed: {summary['passed']}",
        f"- Pass rate: {summary['pass_rate']:.1%}",
        f"- Average score: {summary['average_score']:.2f}",
        "",
        "| Case | Passed | Score | Tool steps | Stop reason |",
        "|---|---:|---:|---:|---|",
    ]
    for row in scorecard["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['passed']} | {row['score']:.2f} | "
            f"{row['tool_steps']} | {row['stop_reason']} |"
        )
    return "\n".join(lines) + "\n"
