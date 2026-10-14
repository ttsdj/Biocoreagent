import json
from pathlib import Path

import pytest

from scicore.kernel.evaluation.research_graph import evaluate_research_graph
from scicore.kernel.features.research_graph import ResearchGraphProjector
from scripts.evaluate_research_graph import load_cases, main

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks" / "research_graph" / "curated-v1" / "cases.jsonl"


def test_frozen_curated_research_graph_suite_passes_all_gates():
    result = evaluate_research_graph(
        load_cases(CASES), ResearchGraphProjector().project
    )

    assert result["case_count"] == 8
    assert result["entity"]["f1"] == 1.0
    assert result["event_relation"]["f1"] == 1.0
    assert result["effective_time_accuracy"] == 1.0
    assert result["evidence_coverage"] == 1.0
    assert result["total_score"] == 100.0
    assert result["passed"] is True


def test_missing_expected_evidence_fails_hard_gate():
    case = load_cases(CASES)[0]

    def without_evidence(memory):
        graph = ResearchGraphProjector().project(memory)
        graph["evidence_links"] = []
        return graph

    result = evaluate_research_graph([case], without_evidence)

    assert result["evidence_coverage"] == 0.0
    assert result["hard_gates_passed"] is False
    assert result["passed"] is False


def test_draft_or_unlabelled_case_cannot_be_scored(tmp_path):
    row = json.loads(CASES.read_text(encoding="utf-8").splitlines()[0])
    row["status"] = "draft"
    draft = tmp_path / "draft.jsonl"
    draft.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not frozen/ready"):
        load_cases(draft)


def test_evaluation_script_writes_versioned_scorecard(tmp_path):
    output = tmp_path / "scorecard.json"

    assert main([str(CASES), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["benchmark_class"] == (
        "curated_research_graph_engineering_qualification"
    )
    assert result["case_sha256"]
