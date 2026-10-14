from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _wilson_ci(correct: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"lo": 0.0, "hi": 0.0}
    phat = correct / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return {"lo": round(100 * (centre - margin) / denom, 1), "hi": round(100 * (centre + margin) / denom, 1)}


def _metric(correct: int, total: int, label: str) -> dict[str, Any]:
    pct = round(100 * correct / total, 1) if total else 0.0
    return {"label": label, "correct": correct, "total": total, "pct": pct, "ci_95": _wilson_ci(correct, total)}


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or "summary" not in report or "tasks" not in report:
        raise ValueError(f"not a scicore.bio BixBench report: {path}")
    return report


def _result_rows(report: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    rows = []
    benchmark = report.get("benchmark", {})
    for task in report.get("tasks", []):
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "benchmark": benchmark.get("name", "unknown"),
                "run_id": run_id,
                "repeat_index": 1,
                "test": "bio-data-analysis",
                "question_id": task.get("id", ""),
                "task_group": task.get("short_id", ""),
                "status": "completed" if task.get("status") in {"pass", "fail"} else task.get("status", ""),
                "success": str(bool(task.get("passed"))),
                "attempts": 1,
                "uploaded_files": "",
                "poll_count": "",
                "latency_ms": round(float(task.get("elapsed_seconds", 0)) * 1000, 1),
                "error": task.get("error", ""),
                "question": "",
                "ground_truth": task.get("ideal", ""),
                "direct_answer": task.get("prediction", ""),
                "answer": task.get("prediction", ""),
                "direct_correct": str(bool(task.get("passed"))),
                "direct_reasoning": task.get("failure_category", ""),
                "mcq_with_refusal_correct": "",
                "mcq_with_refusal_reasoning": "",
                "mcq_with_refusal_mapped_answer": "",
                "mcq_with_refusal_options": "",
                "mcq_without_refusal_correct": "",
                "mcq_without_refusal_reasoning": "",
                "mcq_without_refusal_mapped_answer": "",
                "mcq_without_refusal_options": "",
                "grader_model": "scicore_bio_local_verifier",
                "option_chooser_model": "",
                "judge_error": "",
                "api_task_id": "",
                "data_folder": task.get("data_folder", ""),
                "local_data_dir": task.get("workspace", ""),
            }
        )
    return rows


def _summary(report: dict[str, Any], result_csv: Path, run_id: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    benchmark = report.get("benchmark", {})
    correct = int(summary.get("passed", 0))
    total = int(summary.get("total_tasks", 0))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": f"{benchmark.get('name', 'BixBench')} Results",
        "source_scicore_bio_report": str(Path(report.get("artifacts", {}).get("report_json", "")).resolve())
        if report.get("artifacts", {}).get("report_json")
        else "",
        "verified_50": benchmark.get("verified_50", {}),
        "totals": {"rows": total, "repeats": 1, "questions": total, "task_groups": len(report.get("by_short_id", {}))},
        "source_files": [
            {
                "filename": result_csv.name,
                "file_code": run_id,
                "date": datetime.now(timezone.utc).date().isoformat(),
                "rows": total,
                "repeats": 1,
            }
        ],
        "overall": {
            "direct": _metric(correct, total, "Direct"),
            "mcq_with_refusal": _metric(0, 0, "MCQ with refusal"),
            "mcq_without_refusal": _metric(0, 0, "MCQ without refusal"),
        },
        "by_short_id": report.get("by_short_id", {}),
        "by_eval_mode": report.get("by_eval_mode", {}),
        "by_category": report.get("by_category", {}),
    }


def _html(summary: dict[str, Any]) -> str:
    direct = summary["overall"]["direct"]
    title = summary["title"]
    pct = direct["pct"]
    ci = direct["ci_95"]
    width = 760
    height = 360
    bar_width = 120
    chart_top = 60
    chart_height = 220
    x = 320
    y = chart_top + chart_height * (1 - pct / 100)
    h = chart_height * pct / 100
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #18352a; background: #f7fbf8; }}
.panel {{ max-width: 860px; border: 1px solid #9cc9ad; background: white; padding: 24px; }}
.note {{ color: #4b6358; font-size: 14px; }}
svg {{ width: 100%; max-width: {width}px; height: auto; }}
</style>
<body>
<div class="panel">
<h1>{title}</h1>
<p class="note">Generated from a SciCoreAgent BixBench report. Direct pass@1 only; MCQ modes are not evaluated by SciCoreAgent's local runner.</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="SciCoreAgent BixBench direct score">
  <line x1="120" y1="{chart_top}" x2="120" y2="{chart_top + chart_height}" stroke="#527463"/>
  <line x1="120" y1="{chart_top + chart_height}" x2="640" y2="{chart_top + chart_height}" stroke="#527463"/>
  <text x="82" y="{chart_top + 5}" font-size="13">100%</text>
  <text x="92" y="{chart_top + chart_height}" font-size="13">0%</text>
  <rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="#2f8f5b"/>
  <text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" font-size="24" font-weight="700">{pct:.1f}%</text>
  <text x="{x + bar_width / 2}" y="{chart_top + chart_height + 34}" text-anchor="middle" font-size="16">Direct</text>
  <text x="120" y="330" font-size="14">correct={direct["correct"]}, total={direct["total"]}, 95% CI {ci["lo"]:.1f}-{ci["hi"]:.1f}%</text>
</svg>
</div>
</body>
</html>
"""


def build(report_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    report_path = Path(report_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _load_report(report_path)
    run_id = report_path.parent.name
    result_csv = output_dir / f"{run_id}.csv"
    rows = _result_rows(report, run_id)
    fieldnames = list(rows[0].keys()) if rows else ["timestamp_utc"]
    with result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(report, result_csv, run_id)
    summary_json = output_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html = output_dir / "bio_benchmark_chart.html"
    html.write_text(_html(summary), encoding="utf-8")
    return {"csv": str(result_csv), "summary_json": str(summary_json), "html": str(html)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bio-benchmark-style chart artifacts from a SciCoreAgent BixBench report.")
    parser.add_argument("--report", required=True, help="Path to scicore.bio bixbench_report.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV, summary JSON, and HTML chart.")
    args = parser.parse_args()
    print(json.dumps(build(args.report, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
