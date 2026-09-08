"""Replay smoke suite 的真值测试。

对应简历声明「100 条 Replay Smoke 用例通过率 100%」。

关键点：这 100 条**不是**把 Scorecard 的 100 分满分当成“100 条用例”，而是：
- 生成器（scripts/generate_replay_case_suite.py）原子性地产出 100 个 ready case；
- 每条 case 的 fake_outputs.json 会真实驱动 Pico 工具循环（list_files / read_file /
  search / write_file），真的在 workspace 里产生可验证产物；
- 用 `run_suite_deterministic` 跑完后，scorecard 的 `passed == 100` 字面成立、可复现。

改 case 请先改生成器再重跑，不要直接改生成出来的 case.json / fake_outputs.json。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biocoreagent.replay import lint_case, run_suite_deterministic

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO_ROOT / "benchmarks" / "replay_cases" / "engineering"


def _ready_cases() -> list[Path]:
    return sorted(SUITE_DIR.glob("*/case.json"))


def test_suite_has_exactly_100_ready_cases():
    case_files = _ready_cases()
    assert len(case_files) == 100, f"expected 100 replay cases, found {len(case_files)}"

    manifest_path = SUITE_DIR / "manifest.json"
    assert manifest_path.is_file()
    data = manifest_path.read_text(encoding="utf-8")
    assert '"total_cases": 100' in data
    assert '"suite": "engineering"' in data

    # 每条 case 都必须带 fake_outputs.json，并且 lint 通过、状态 ready。
    for case_file in case_files:
        assert (case_file.parent / "fake_outputs.json").is_file(), (
            f"{case_file.parent.name} is missing fake_outputs.json"
        )
        result = lint_case(case_file)
        assert result["ready"], (
            f"{case_file.parent.name} is not ready: {result['errors'] + result['warnings']}"
        )


@pytest.mark.slow
def test_suite_deterministic_passes_100(tmp_path):
    result = run_suite_deterministic(SUITE_DIR, tmp_path / "runs")
    scorecard = result["scorecard"]["summary"]

    assert scorecard["total_cases"] == 100
    assert scorecard["passed"] == 100
    assert scorecard["failed"] == 0
    assert scorecard["pass_rate"] == 1.0
    assert scorecard["failure_category_counts"] == {}

    # 逐条硬门禁断言：每条都真实跑通、真实产生产物、无被拒工具 / 未恢复的恢复。
    for row in result["results"]:
        verification = row["verification"]
        assert verification["passed"], f"{row['case_id']} failed"
        assert verification["hard_gates_passed"], (
            f"{row['case_id']} did not pass hard gates"
        )
        assert verification["score"] >= 80, f"{row['case_id']} scored below 80"
