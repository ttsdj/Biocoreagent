"""Evaluate frozen cross-session memory cases against configured Postgres."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Direct ``python scripts/...`` execution otherwise may import a stale editable
# installation instead of the source tree being evaluated.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scicore.kernel.evaluation.memory_recall import evaluate_memory_recall
from scicore.kernel.features.postgres_memory import PostgresLongTermMemoryStore


def load_cases(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        required = {"case_id", "query", "relevant_memory_ids"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"case {line_number} is missing: {', '.join(missing)}")
        if str(row.get("status", "frozen")).lower() not in {"frozen", "ready"}:
            raise ValueError(f"case {line_number} is not frozen/ready for scoring")
        if not str(row["query"]).strip():
            raise ValueError(f"case {line_number} has no independently authored query")
        if not list(row["relevant_memory_ids"]):
            raise ValueError(f"case {line_number} has no labelled relevant_memory_ids")
        rows.append(row)
    if not rows:
        raise ValueError("benchmark case file is empty")
    return rows


def load_corpus(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"memory_id", "statement", "evidence", "user_scope", "project_scope"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"corpus row {line_number} is missing: {', '.join(missing)}")
        rows.append(row)
    return rows


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="Frozen JSONL cases with relevant memory IDs.")
    parser.add_argument(
        "--corpus",
        default="",
        help="Optional versioned JSONL corpus to upsert before evaluation.",
    )
    parser.add_argument("--output", required=True, help="Path for the JSON scorecard.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-hit-rate", type=float, default=0.994)
    parser.add_argument("--min-mrr", type=float, default=0.67)
    parser.add_argument("--max-scope-leakage", type=float, default=0.0)
    args = parser.parse_args(argv)

    store = PostgresLongTermMemoryStore.from_environment()
    if store is None:
        parser.error("SCICORE_MEMORY_POSTGRES_DSN must be configured")
    cases = load_cases(args.cases)
    seeded = 0
    if args.corpus:
        corpus = load_corpus(args.corpus)
        for memory in corpus:
            store.upsert(memory)
        seeded = len(corpus)
    result = evaluate_memory_recall(cases, store.retrieve, k=args.k)
    result["schema_version"] = 1
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    result["seeded_memory_count"] = seeded
    result["case_source"] = str(Path(args.cases))
    result["corpus_source"] = str(Path(args.corpus)) if args.corpus else ""
    result["case_sha256"] = file_sha256(args.cases)
    result["corpus_sha256"] = file_sha256(args.corpus) if args.corpus else ""
    provider = getattr(store, "embedding_provider", None)
    result["embedding_backend"] = getattr(provider, "name", "unknown")
    synthetic = bool(cases) and all(
        str(item.get("fixture_class", "")).startswith("synthetic_")
        for item in cases
    )
    result["benchmark_class"] = (
        "synthetic_engineering_qualification"
        if synthetic
        else "human_reviewed_cross_session"
    )
    result["claim_boundary"] = (
        "This score qualifies the frozen synthetic named-entity fixture only; "
        "it is not open-domain semantic-QA evidence."
        if synthetic
        else "This score is valid only for the supplied frozen, labelled cases."
    )
    result["thresholds"] = {
        "min_hit_rate": args.min_hit_rate,
        "min_mrr": args.min_mrr,
        "max_scope_leakage": args.max_scope_leakage,
    }
    result["passed"] = bool(
        result[f"hit_rate_at_{args.k}"] >= args.min_hit_rate
        and result[f"mrr_at_{args.k}"] >= args.min_mrr
        and result["scope_leakage_rate"] <= args.max_scope_leakage
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
