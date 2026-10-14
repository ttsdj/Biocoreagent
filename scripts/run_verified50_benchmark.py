from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ID = "phylobio/BixBench-Verified-50"
JSONL_NAME = "BixBench-Verified-50.jsonl"


def _token_present() -> bool:
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    return bool(get_token())


def _dataset_ready(dataset_dir: Path) -> bool:
    return (dataset_dir / JSONL_NAME).is_file() and any(dataset_dir.glob("CapsuleFolder-*.zip"))


def _download_dataset(dataset_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(dataset_dir))


def _run(command: list[str], cwd: Path, *, allow_nonzero: bool = False) -> int:
    result = subprocess.run(command, cwd=str(cwd), text=True)
    if result.returncode != 0 and not allow_nonzero:
        raise SystemExit(result.returncode)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download BixBench-Verified-50, run SciCoreAgent, and build bio-benchmark-style chart artifacts."
    )
    parser.add_argument("--dataset-dir", default=r"F:\aicoding\bixbench_verified50")
    parser.add_argument("--run-dir", default=r"F:\aicoding\bixbench_runs\verified50-deepseekchat-v1")
    parser.add_argument(
        "--chart-dir",
        default=r"D:\aicoding\00.project\scicore.bio\V3\.scicore\bixbench_runs\reports\bio_benchmark_verified50",
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--task-timeout-seconds", type=int, default=600)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(args.dataset_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    chart_dir = Path(args.chart_dir).resolve()

    if not args.skip_download and not _dataset_ready(dataset_dir):
        if not _token_present():
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "blocker": "huggingface_gated_dataset_auth_required",
                        "dataset": REPO_ID,
                        "message": (
                            "BixBench-Verified-50 is gated and this machine has no HuggingFace token. "
                            "Run `hf auth login`, accept the dataset terms on HuggingFace, then rerun this script."
                        ),
                        "expected_jsonl": str(dataset_dir / JSONL_NAME),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _download_dataset(dataset_dir)

    if not _dataset_ready(dataset_dir):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blocker": "verified50_dataset_files_missing",
                    "dataset_dir": str(dataset_dir),
                    "expected_jsonl": str(dataset_dir / JSONL_NAME),
                    "expected_capsules": str(dataset_dir / "CapsuleFolder-*.zip"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    command = [
        sys.executable,
        "-m",
        "scicore.agent.eval.bixbench",
        "--official-dir",
        str(dataset_dir),
        "--official-jsonl",
        JSONL_NAME,
        "--strategy",
        "scicore_bio_agent",
        "--extract-capsules",
        "--resume",
        "--max-steps",
        str(args.max_steps),
        "--task-timeout-seconds",
        str(args.task_timeout_seconds),
        "--output-dir",
        str(run_dir),
        "--provider",
        args.provider,
        "--model",
        args.model,
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    eval_returncode = _run(command, repo_root, allow_nonzero=True)

    report = run_dir / "bixbench_report.json"
    if not report.is_file():
        print(
            json.dumps(
                {
                    "status": "error",
                    "blocker": "bixbench_report_missing",
                    "eval_returncode": eval_returncode,
                    "expected_report": str(report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return eval_returncode or 1
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_bio_benchmark_chart.py"),
            "--report",
            str(report),
            "--output-dir",
            str(chart_dir),
        ],
        repo_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset_dir": str(dataset_dir),
                "report": str(report),
                "chart_dir": str(chart_dir),
                "chart_html": str(chart_dir / "bio_benchmark_chart.html"),
                "eval_returncode": eval_returncode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
