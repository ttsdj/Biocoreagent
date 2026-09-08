"""Command line interface for BioCoreAgent trace and replay operations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from pico.cli import _build_model_client
from pico.config import load_project_env

from .replay import (
    ReplayError,
    build_scorecard,
    compare_scorecard,
    explain_artifact,
    extract_case,
    fake_model_from_file,
    lint_case,
    load_json,
    materialize_source,
    run_case,
    run_suite,
    run_suite_deterministic,
    source_versions,
    validate_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biocore-replay",
        description="Inspect BioCoreAgent evidence and run local fresh-trace regressions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    trace = commands.add_parser("trace", help="Trace evidence operations.")
    trace_commands = trace.add_subparsers(dest="trace_command", required=True)
    trace_validate = trace_commands.add_parser("validate")
    trace_validate.add_argument("run_dir")

    case = commands.add_parser("case", help="Replay case operations.")
    case_commands = case.add_subparsers(dest="case_command", required=True)
    case_extract = case_commands.add_parser("extract")
    case_extract.add_argument("run_dir")
    case_extract.add_argument("--case-id", required=True)
    case_extract.add_argument("--output-root", default="benchmarks/replay_cases")
    case_lint = case_commands.add_parser("lint")
    case_lint.add_argument("case")

    run = commands.add_parser("run", help="Run one ready case with the current runtime.")
    run.add_argument("case")
    run.add_argument(
        "--output-root",
        default=os.environ.get("BIOCOREAGENT_REPLAY_ROOT", ".biocoreagent/replay_runs"),
    )
    run.add_argument("--fake-output-file")
    run.add_argument(
        "--fake-per-case",
        action="store_true",
        help="Run a directory of cases, each driven by its own fake_outputs.json (deterministic suite).",
    )
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--base-url")
    run.add_argument("--host", default="http://127.0.0.1:11434")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--ollama-timeout", type=int, default=300)
    run.add_argument("--openai-timeout", type=int, default=300)
    run.add_argument("--scorecard")

    compare = commands.add_parser("compare", help="Compare a scorecard with a baseline.")
    compare.add_argument("scorecard")
    compare.add_argument("--baseline", required=True)

    baseline = commands.add_parser("baseline", help="Explicitly update a baseline.")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_update = baseline_commands.add_parser("update")
    baseline_update.add_argument("scorecard")
    baseline_update.add_argument("--target", required=True)

    artifact = commands.add_parser("artifact", help="Explain artifact lineage.")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    artifact_explain = artifact_commands.add_parser("explain")
    artifact_explain.add_argument("artifact")
    artifact_explain.add_argument("--search-root", default=".biocoreagent/runs")

    source = commands.add_parser("source", help="Inspect or reconstruct captured source.")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    source_show = source_commands.add_parser("show")
    source_show.add_argument("run_dir")
    source_show.add_argument("--at", default="")
    source_materialize = source_commands.add_parser("materialize")
    source_materialize.add_argument("run_dir")
    source_materialize.add_argument("--at", default="")
    source_materialize.add_argument("--target", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "trace":
            result = validate_trace(args.run_dir)
            _print(result)
            return 0 if result["valid"] else 1
        if args.command == "case" and args.case_command == "extract":
            return _emit(extract_case(args.run_dir, args.output_root, args.case_id))
        if args.command == "case" and args.case_command == "lint":
            result = lint_case(args.case)
            _print(result)
            return 0 if result["ready"] else 1
        if args.command == "run":
            load_project_env(Path(args.case).resolve(), override=False)
            model_client = (
                fake_model_from_file(args.fake_output_file)
                if args.fake_output_file
                else _build_model_client(_model_args(args))
            )
            candidate = Path(args.case).resolve()
            if candidate.is_dir() and not (candidate / "case.json").is_file():
                suite_cases = sorted(candidate.glob("*/case.json"))
                deterministic = bool(suite_cases) and all(
                    (case_file.parent / "fake_outputs.json").is_file()
                    for case_file in suite_cases
                )
                if args.fake_per_case or deterministic:
                    result = run_suite_deterministic(candidate, args.output_root)
                else:
                    result = run_suite(candidate, args.output_root, model_client=model_client)
                passed = not result["scorecard"]["summary"]["failed"]
            else:
                result = run_case(args.case, args.output_root, model_client=model_client)
                passed = result["verification"]["passed"]
                if args.scorecard:
                    result["scorecard"] = build_scorecard([result], args.scorecard)
            _print(result)
            return 0 if passed else 1
        if args.command == "compare":
            result = compare_scorecard(load_json(args.scorecard), load_json(args.baseline))
            _print(result)
            return 0 if result["passed"] else 1
        if args.command == "baseline":
            source = Path(args.scorecard).resolve()
            target = Path(args.target).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return _emit({"updated": str(target), "source": str(source)})
        if args.command == "artifact":
            return _emit(explain_artifact(args.search_root, args.artifact))
        if args.command == "source" and args.source_command == "show":
            return _emit({"sources": source_versions(args.run_dir, at_event=args.at)})
        if args.command == "source" and args.source_command == "materialize":
            return _emit(materialize_source(args.run_dir, args.target, at_event=args.at))
    except (ReplayError, RuntimeError, ValueError, OSError) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2
    return 2


def _model_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        ollama_timeout=args.ollama_timeout,
        openai_timeout=args.openai_timeout,
        secret_env_names=[],
    )


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _emit(value) -> int:
    _print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
