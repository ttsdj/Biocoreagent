import json

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The docs/ directory is gitignored (user's explicit "忽略全部docs" decision), so a
# fresh clone has no docs.  These tests assert on the resume/ledger content; when that
# content is absent they must skip cleanly rather than hard-fail the committed suite.
RESUME_DOC = REPO_ROOT / "docs" / "resume" / "biocoreagent_resume_star.md"
LEDGER_DOC = REPO_ROOT / "docs" / "claim_evidence_ledger.md"


def _skip_if_missing(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"docs/ is gitignored in this checkout; missing: {path}")


def test_resume_metrics_artifact_matches_documented_claims():
    metrics_path = REPO_ROOT / "artifacts" / "resume-metrics-v1.json"
    harness_path = REPO_ROOT / "artifacts" / "harness-regression-v2.json"
    resume_path = RESUME_DOC
    memory_scorecard_path = REPO_ROOT / "artifacts" / "memory-recall-synthetic-v1.json"
    graph_scorecard_path = REPO_ROOT / "artifacts" / "research-graph-curated-v1.json"

    _skip_if_missing(resume_path)
    assert metrics_path.is_file()
    assert harness_path.is_file()
    assert resume_path.is_file()
    assert memory_scorecard_path.is_file()
    assert graph_scorecard_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    resume = resume_path.read_text(encoding="utf-8")
    memory_scorecard = json.loads(memory_scorecard_path.read_text(encoding="utf-8"))
    graph_scorecard = json.loads(graph_scorecard_path.read_text(encoding="utf-8"))

    assert harness["summary"]["passed"] == 14
    assert harness["summary"]["pass_rate"] == 1.0
    assert harness["summary"]["within_budget_rate"] == 1.0
    assert harness["summary"]["verifier_pass_rate"] == 1.0

    assert metrics["memory"]["query_count"] == 40
    assert metrics["memory"]["hit_rate_at_10"] == 1.0
    assert metrics["memory"]["mrr_at_10"] == 1.0
    assert memory_scorecard["query_count"] == 200
    assert memory_scorecard["hit_rate_at_10"] >= 0.994
    assert memory_scorecard["mrr_at_10"] >= 0.67
    assert memory_scorecard["scope_leakage_rate"] == 0.0
    assert memory_scorecard["benchmark_class"] == "synthetic_engineering_qualification"
    assert memory_scorecard["embedding_backend"].startswith("sentence_transformers:")
    assert graph_scorecard["case_count"] == 8
    assert graph_scorecard["entity"]["f1"] == 1.0
    assert graph_scorecard["event_relation"]["f1"] == 1.0
    assert graph_scorecard["effective_time_accuracy"] == 1.0
    assert graph_scorecard["evidence_coverage"] == 1.0
    assert graph_scorecard["total_score"] == 100.0
    assert graph_scorecard["hard_gates_passed"] is True

    assert metrics["context"]["case_count"] == 15
    assert metrics["context"]["average_raw_chars"] == 26928
    assert metrics["context"]["average_prompt_chars"] == 1795.33
    assert metrics["context"]["average_compression_ratio"] == 0.929541
    assert metrics["context"]["request_retention_rate"] == 1.0
    assert metrics["context"]["latest_evidence_retention_rate"] == 1.0
    assert metrics["context"]["budget_compliance_rate"] == 1.0

    assert metrics["multiagent"]["suite"] == "eight_worker_scheduler_v2_three_run_median"
    assert metrics["multiagent"]["speedup"] >= 2.0
    assert metrics["multiagent"]["lifecycle_state_count"] == 9
    assert metrics["multiagent"]["degradation_levels"] == 3

    assert metrics["bixbench_verified50"]["initial"]["passed"] == 22
    assert metrics["bixbench_verified50"]["best"]["passed"] == 38

    for claim in (
        "14/14",
        "26,928",
        "1,795",
        "92.95%",
        "至少 2.0",
        "44%",
        "76%",
    ):
        assert claim in resume


def test_resume_document_states_material_claim_boundaries():
    _skip_if_missing(RESUME_DOC)
    resume = RESUME_DOC.read_text(encoding="utf-8")

    assert "当前 Postgres 实现将 embedding 保存为 JSONB" in resume
    assert "当前 100%/1.000 来自 200 条冻结合成工程资格集" in resume
    assert "官方外部 LLM judge 未运行" in resume
    assert "44% 是 BioCoreAgent 初版" in resume


def test_claim_evidence_ledger_matches_artifacts():
    # 权威 ledger 作为全仓库唯一真值来源,必须与 artifacts 关键数字一致,且不残留旧口径。
    ledger_path = LEDGER_DOC
    _skip_if_missing(ledger_path)
    ledger = ledger_path.read_text(encoding="utf-8")

    # 真值标记必须出现在权威表里
    for token in (
        "26,928",
        "1,795",
        "92.95%",
        "95.23%",
        "14/14",
        "100%",
        "1.000",
        "99.4%",
        "0.67",
        "22/50",
        "38/50",
        "76%",
        "32 个百分点",
        "100/100",
        "8 个冻结策划场景",
        "验收门槛",
        "不是实测结果",
    ):
        assert token in ledger, f"claim_evidence_ledger.md 缺少真值标记: {token}"

    # 旧口径(91.42% / 提升20% / 把 99.4%当实测)不得作为独立真值残留
    assert "91.42%" not in ledger
    assert "提升 20%" not in ledger

    # 双向校验:artifact 里的标志性数字必须与权威表口径一致
    metrics = json.loads(
        (REPO_ROOT / "artifacts" / "resume-metrics-v1.json").read_text(encoding="utf-8")
    )
    assert metrics["context"]["average_compression_ratio"] == 0.929541  # 92.95%
    assert metrics["bixbench_verified50"]["initial"]["passed"] == 22
    assert metrics["bixbench_verified50"]["best"]["passed"] == 38

    memory_scorecard = json.loads(
        (REPO_ROOT / "artifacts" / "memory-recall-synthetic-v1.json").read_text(encoding="utf-8")
    )
    assert memory_scorecard["query_count"] == 200
    assert memory_scorecard["hit_rate_at_10"] >= 0.994
    assert memory_scorecard["mrr_at_10"] >= 0.67
