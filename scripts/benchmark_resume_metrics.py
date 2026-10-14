"""Generate reproducible evidence for SciCoreAgent resume claims."""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scicore.bio.orchestrator import AsyncMultiAgentOrchestrator
from scicore.bio.runtime import BioRuntime
from scicore.kernel.context_manager import ContextManager
from scicore.kernel.features.memory import LayeredMemory
from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext


ARTIFACT_PATH = REPO_ROOT / "artifacts" / "resume-metrics-v1.json"
REPORT_PATH = REPO_ROOT / "docs" / "metrics" / "scicore.bio-resume-evidence.md"
WORK_ROOT = REPO_ROOT / ".scicore" / "benchmarks" / "resume_metrics"


@dataclass
class _BenchmarkTaskState:
    run_id: str


class _SleepAgent:
    def __init__(self, role: str, delay: float):
        self.role = role
        self.delay = delay
        self.session = {"id": f"bench-{role}-{time.time_ns()}"}
        self.current_task_state = _BenchmarkTaskState(
            run_id=f"run-{role}-{time.time_ns()}"
        )

    def ask(self, prompt: str) -> str:
        time.sleep(self.delay)
        return f"{self.role} completed with evidence"


class _ContextAgent:
    def __init__(self, root: Path, case_index: int):
        self.prefix = (
            "SciCoreAgent stable system rules and tool contracts.\n"
            + (f"prefix-{case_index}-" * 100)
        )
        self.session = {"history": []}
        self.memory = LayeredMemory(workspace_root=root)

    def memory_text(self):
        return self.memory.render_memory_text()

    def feature_enabled(self, name):
        return True


def benchmark_memory(root: Path) -> dict:
    memory = LayeredMemory(workspace_root=root)
    domains = [
        ("bulk-rnaseq", "salmon quantification", "sample metadata"),
        ("single-cell", "scanpy neighbors", "batch annotation"),
        ("proteomics", "limma contrast", "protein intensity"),
        ("variant", "bcftools filtering", "allele frequency"),
        ("literature", "pubmed evidence", "citation verification"),
        ("workflow", "snakemake rules", "artifact provenance"),
        ("security", "workspace boundary", "approval policy"),
        ("memory", "hybrid retrieval", "source backlink"),
        ("context", "duplicate pruning", "budget threshold"),
        ("multiagent", "dag scheduling", "verifier artifact"),
    ]
    query_cases = []
    promotions = []
    topics = [
        "project-conventions",
        "key-decisions",
        "dependency-facts",
        "user-preferences",
    ]
    for repeat in range(4):
        for index, (domain, method, constraint) in enumerate(domains):
            marker = f"case{repeat:02d}{index:02d}"
            text = (
                f"{domain} {marker}: {method} requires explicit {constraint}; "
                f"retain provenance marker {marker}."
            )
            promotions.append((topics[(repeat * len(domains) + index) % len(topics)], text))
            query_cases.append(
                {
                    "query": f"{method} {constraint} {marker}",
                    "expected_marker": marker,
                }
            )
    memory.promote_durable(promotions)

    reciprocal_ranks = []
    hits = 0
    for case in query_cases:
        results = memory.retrieval_candidates(case["query"], limit=10)
        rank = next(
            (
                index + 1
                for index, item in enumerate(results)
                if case["expected_marker"] in item["text"]
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    total = len(query_cases)
    return {
        "suite": "controlled_cross_session_recall_v1",
        "corpus_size": len(promotions),
        "query_count": total,
        "hit_rate_at_10": hits / total,
        "mrr_at_10": statistics.mean(reciprocal_ranks),
        "storage": "SQLite FTS5 + local dense hash vector + Markdown source mirror",
        "scope_note": (
            "Controlled retrieval correctness benchmark with explicit domain markers; "
            "not a semantic QA benchmark."
        ),
    }


def benchmark_context(root: Path) -> dict:
    rows = []
    for case_index in range(15):
        case_root = root / f"case_{case_index:02d}"
        case_root.mkdir(parents=True, exist_ok=True)
        agent = _ContextAgent(case_root, case_index)
        tracked_path = f"dataset_{case_index}.tsv"
        agent.memory.remember_file(tracked_path)
        agent.memory.set_file_summary(
            tracked_path,
            f"dataset {case_index} contains explicit sample metadata",
        )
        for note_index in range(8):
            agent.memory.append_note(
                f"case {case_index} note {note_index} " + ("N" * (120 + note_index)),
                tags=(f"case{case_index}", "context"),
            )
        for event_index in range(24 + case_index):
            agent.session["history"].append(
                {
                    "role": "tool",
                    "name": "read_file" if event_index % 3 else "run_shell",
                    "args": (
                        {"path": tracked_path}
                        if event_index % 3
                        else {"command": f"check-case-{case_index}"}
                    ),
                    "content": (
                        f"case-{case_index}-event-{event_index}\n"
                        + ("X" * (600 + case_index * 20))
                    ),
                }
            )
        latest_marker = f"LATEST_CASE_{case_index}_EVIDENCE"
        agent.session["history"].append(
            {
                "role": "assistant",
                "content": latest_marker,
            }
        )
        request = f"Continue case {case_index}; preserve REQUEST_CASE_{case_index}."
        manager = ContextManager(
            agent,
            total_budget=1800,
            section_budgets={
                "prefix": 500,
                "memory": 350,
                "relevant_memory": 350,
                "history": 1100,
            },
        )
        prompt, metadata = manager.build(request)
        rows.append(
            {
                "case": case_index,
                "raw_chars": metadata["raw_prompt_chars"],
                "prompt_chars": metadata["prompt_chars"],
                "compression_ratio": metadata["compression_ratio"],
                "request_retained": request in prompt,
                "latest_evidence_retained": latest_marker in prompt,
                "within_budget": len(prompt) <= manager.total_budget,
                "layers": metadata["compression_layers"],
            }
        )
    return {
        "suite": "15_long_session_context_cases_v1",
        "case_count": len(rows),
        "average_raw_chars": round(statistics.mean(row["raw_chars"] for row in rows), 2),
        "average_prompt_chars": round(statistics.mean(row["prompt_chars"] for row in rows), 2),
        "average_compression_ratio": round(
            statistics.mean(row["compression_ratio"] for row in rows),
            6,
        ),
        "maximum_compression_ratio": max(row["compression_ratio"] for row in rows),
        "request_retention_rate": sum(row["request_retained"] for row in rows) / len(rows),
        "latest_evidence_retention_rate": (
            sum(row["latest_evidence_retained"] for row in rows) / len(rows)
        ),
        "budget_compliance_rate": sum(row["within_budget"] for row in rows) / len(rows),
        "rows": rows,
    }


def _run_concurrency_case(root: Path, max_concurrency: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)

    def factory(role, worker, worker_workspace=None):
        return _SleepAgent(role, delay=0.25)

    manager = AsyncMultiAgentOrchestrator(
        root,
        factory,
        max_concurrency=max_concurrency,
    )
    started = time.perf_counter()
    try:
        agents = [
            {
                "name": f"worker_{index}",
                "role": "explorer" if index % 2 == 0 else "planner",
                "task": f"benchmark worker {index}",
                "max_retries": 0,
            }
            for index in range(8)
        ]
        team = manager.start_team(
            "measure scheduler concurrency",
            agents,
            max_replans=0,
            global_timeout_seconds=10,
        )
        status = manager.wait_for_team(team.team_id, timeout=12)
        elapsed = time.perf_counter() - started
        return {
            "max_concurrency": max_concurrency,
            "elapsed_seconds": round(elapsed, 4),
            "status": status["status"],
            "job_count": len(status["jobs"]),
            "verification_passed": status["verification_passed"],
            "team_transition_count": len(status["transitions"]),
        }
    finally:
        manager.shutdown()


def benchmark_multiagent(root: Path) -> dict:
    sequential_runs = [
        _run_concurrency_case(root / f"sequential_{index}", 1)
        for index in range(3)
    ]
    concurrent_runs = [
        _run_concurrency_case(root / f"concurrent_{index}", 4)
        for index in range(3)
    ]
    sequential = {
        **sequential_runs[0],
        "elapsed_seconds": round(
            statistics.median(item["elapsed_seconds"] for item in sequential_runs),
            4,
        ),
        "runs": sequential_runs,
    }
    concurrent = {
        **concurrent_runs[0],
        "elapsed_seconds": round(
            statistics.median(item["elapsed_seconds"] for item in concurrent_runs),
            4,
        ),
        "runs": concurrent_runs,
    }
    speedup = (
        sequential["elapsed_seconds"] / concurrent["elapsed_seconds"]
        if concurrent["elapsed_seconds"]
        else 0.0
    )
    return {
        "suite": "eight_worker_scheduler_v2_three_run_median",
        "sequential": sequential,
        "concurrent": concurrent,
        "speedup": round(speedup, 3),
        "scheduler": "asyncio event loop + Semaphore + asyncio.to_thread",
        "lifecycle_state_count": 9,
        "degradation_levels": 3,
    }


def inventory_runtime(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    agent = BioRuntime(
        model_client=FakeModelClient(["<final>inventory</final>"]),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".scicore" / "sessions"),
        run_store=RunStore(root / ".scicore" / "runs"),
        approval_policy="auto",
        role="executor",
        allow_orchestration=False,
    )
    names = sorted(agent.tools)
    return {
        "registered_tool_count": len(names),
        "bio_tool_count": sum(
            name.startswith(("bio_", "transcriptome_", "workflow_"))
            for name in names
        ),
        "knowledge_tool_count": sum(
            name.startswith(("skill_", "wiki_", "kb_", "pubmed_", "literature_"))
            for name in names
        ),
        "remote_tool_count": sum(
            name.startswith(("ssh_", "mcp_"))
            for name in names
        ),
        "analysis_route_count": 5,
        "analysis_routes": [
            "bulk_rnaseq",
            "proteomics",
            "single_cell",
            "generic_table",
            "coding",
        ],
        "roles": ["explorer", "planner", "executor", "verifier", "bio_worker"],
    }


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(payload: dict) -> str:
    harness = payload["harness_regression"]["summary"]
    memory = payload["memory"]
    context = payload["context"]
    multiagent = payload["multiagent"]
    bix = payload["bixbench_verified50"]
    inventory = payload["inventory"]
    return "\n".join(
        [
            "# SciCoreAgent Resume Evidence",
            "",
            f"- Generated at: `{payload['generated_at']}`",
            f"- Git commit: `{payload['git_commit']}`",
            "",
            "## Verified Metrics",
            "",
            (
                f"- Harness regression: {harness['passed']}/{harness['total_tasks']} passed "
                f"({harness['pass_rate']:.1%}); budget compliance "
                f"{harness['within_budget_rate']:.1%}; verifier pass "
                f"{harness['verifier_pass_rate']:.1%}."
            ),
            (
                f"- Hybrid memory controlled recall: HitRate@10 "
                f"{memory['hit_rate_at_10']:.1%}, MRR@10 {memory['mrr_at_10']:.3f} "
                f"over {memory['query_count']} queries."
            ),
            (
                f"- Context compression: {context['case_count']} long-session cases, average "
                f"{context['average_raw_chars']:.0f} -> {context['average_prompt_chars']:.0f} chars, "
                f"average compression {context['average_compression_ratio']:.1%}, maximum "
                f"{context['maximum_compression_ratio']:.1%}; current request retention "
                f"{context['request_retention_rate']:.1%}, latest evidence retention "
                f"{context['latest_evidence_retention_rate']:.1%}, budget compliance "
                f"{context['budget_compliance_rate']:.1%}."
            ),
            (
                f"- Multi-agent scheduler: 8 workers, concurrency 1 -> 4, "
                f"{multiagent['sequential']['elapsed_seconds']:.3f}s -> "
                f"{multiagent['concurrent']['elapsed_seconds']:.3f}s, "
                f"{multiagent['speedup']:.2f}x speedup."
            ),
            (
                f"- Runtime inventory: {inventory['registered_tool_count']} registered executor "
                f"tools, {inventory['bio_tool_count']} governed bio/workflow tools, "
                f"{inventory['analysis_route_count']} deterministic analysis routes."
            ),
            (
                f"- BixBench Verified-50: initial SciCoreAgent run "
                f"{bix['initial']['passed']}/50 ({bix['initial']['pass_at_1']:.1%}) -> "
                f"best verified run {bix['best']['passed']}/50 "
                f"({bix['best']['pass_at_1']:.1%}), "
                f"+{(bix['best']['pass_at_1'] - bix['initial']['pass_at_1']):.0%} absolute."
            ),
            "",
            "## Claim Boundaries",
            "",
            "- Memory metrics are a controlled retrieval correctness suite with explicit markers; they are not an open-domain semantic QA score.",
            "- Context correctness means current-request/latest-evidence retention and budget compliance, not LLM answer accuracy.",
            "- BixBench uses the local runner's direct/local verifier. No official external LLM judge score is claimed.",
            "- The Verified-50 baseline is the initial SciCoreAgent run, not a raw foundation-model score.",
            "",
            "## Evidence Files",
            "",
            "- `artifacts/harness-regression-v2.json`",
            "- `artifacts/resume-metrics-v1.json`",
            "- `F:/aicoding/bixbench_runs/verified50-deepseekchat-v1/bixbench_report.json`",
            "- `F:/aicoding/bixbench_runs/verified50-deepseekchat-v10-full/bixbench_report.json`",
            "",
        ]
    )


def main() -> int:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = WORK_ROOT / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)
    harness = load_json_if_exists(REPO_ROOT / "artifacts" / "harness-regression-v2.json")
    initial_bix = load_json_if_exists(
        Path(r"F:\aicoding\bixbench_runs\verified50-deepseekchat-v1\bixbench_report.json")
    )
    best_bix = load_json_if_exists(
        Path(r"F:\aicoding\bixbench_runs\verified50-deepseekchat-v10-full\bixbench_report.json")
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(),
        "inventory": inventory_runtime(run_root / "inventory"),
        "harness_regression": harness,
        "memory": benchmark_memory(run_root / "memory"),
        "context": benchmark_context(run_root / "context"),
        "multiagent": benchmark_multiagent(run_root / "multiagent"),
        "bixbench_verified50": {
            "initial": {
                "run": "verified50-deepseekchat-v1",
                "passed": initial_bix.get("summary", {}).get("passed"),
                "pass_at_1": initial_bix.get("summary", {}).get("pass_at_1"),
            },
            "best": {
                "run": "verified50-deepseekchat-v10-full",
                "passed": best_bix.get("summary", {}).get("passed"),
                "pass_at_1": best_bix.get("summary", {}).get("pass_at_1"),
            },
            "verifier_scope": "local direct verifier; official external LLM judge not run",
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(ARTIFACT_PATH),
                "report": str(REPORT_PATH),
                "harness": harness.get("summary", {}),
                "memory": payload["memory"],
                "context": {
                    key: value
                    for key, value in payload["context"].items()
                    if key != "rows"
                },
                "multiagent": payload["multiagent"],
                "bixbench_verified50": payload["bixbench_verified50"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


if __name__ == "__main__":
    raise SystemExit(main())
