"""Evaluate research graph projection against a frozen labelled JSONL suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scicore.kernel.evaluation.research_graph import evaluate_research_graph
from scicore.kernel.features.research_graph import ResearchGraphProjector


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        missing = sorted({"case_id", "status", "memory", "expected"} - set(row))
        if missing:
            raise ValueError(
                f"research graph case {line_number} is missing: {', '.join(missing)}"
            )
        if str(row["status"]).lower() not in {"frozen", "ready"}:
            raise ValueError(f"research graph case {line_number} is not frozen/ready")
        if not dict(row["memory"]).get("evidence", {}).get("source_uri"):
            raise ValueError(f"research graph case {line_number} has no source evidence")
        if not list(dict(row["expected"]).get("evidence", [])):
            raise ValueError(
                f"research graph case {line_number} has no independently labelled evidence"
            )
        rows.append(row)
    if not rows:
        raise ValueError("research graph benchmark is empty")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="Frozen labelled JSONL research graph cases.")
    parser.add_argument("--output", required=True, help="JSON scorecard output path.")
    parser.add_argument("--min-score", type=float, default=80.0)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    result = evaluate_research_graph(
        cases, ResearchGraphProjector().project, min_score=args.min_score
    )
    result.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "case_source": str(Path(args.cases)),
            "case_sha256": file_sha256(args.cases),
            "benchmark_class": "curated_research_graph_engineering_qualification",
            "claim_boundary": (
                "This score qualifies only the frozen curated research scenarios; "
                "it is not evidence of open-domain extraction quality."
            ),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "details"},
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
