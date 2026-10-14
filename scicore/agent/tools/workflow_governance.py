"""OmicOS-style workflow governance tools.

These tools keep complex research/coding tasks from becoming an unstructured
LLM loop. They provide preflight gates, a primitive ledger, option composition,
and durable plan artifacts that both humans and agents can inspect.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Tool
from .bio.r_bridge import _find_rscript


PRIMITIVES = [
    {
        "id": "inspect_count_matrix",
        "tool": "bio_count_matrix_inspect",
        "stage": "eda_gate",
        "description": "Inspect gene-by-sample count matrices without biological claims.",
        "risk": "low",
    },
    {
        "id": "infer_sample_groups",
        "tool": "workflow_preflight_check",
        "stage": "eda_gate",
        "description": "Infer sample group prefixes and replicate counts from sample names.",
        "risk": "low",
    },
    {
        "id": "check_r_deseq2_env",
        "tool": "workflow_preflight_check",
        "stage": "environment_gate",
        "description": "Check Rscript, DESeq2, and rlang compatibility before execution.",
        "risk": "low",
    },
    {
        "id": "check_omicverse_bulk_backend",
        "tool": "transcriptome_omicverse_check",
        "stage": "environment_gate",
        "description": "Check OmicVerse bulk backend and pyDEG capability availability.",
        "risk": "low",
    },
    {
        "id": "run_omicverse_pydge_tissue_vs_rest",
        "tool": "transcriptome_omicverse_deg",
        "stage": "execute",
        "description": "Run target-vs-rest differential expression through OmicVerse ov.bulk.pyDEG.",
        "risk": "high",
    },
    {
        "id": "run_deseq2_tissue_vs_rest",
        "tool": "bio_deseq2_tissue_vs_rest",
        "stage": "execute",
        "description": "Fallback deterministic DESeq2 target-vs-rest workflow.",
        "risk": "high",
    },
    {
        "id": "export_literature_xlsx",
        "tool": "literature_export_xlsx",
        "stage": "emit_artifacts",
        "description": "Write literature/evidence tables to xlsx.",
        "risk": "high",
    },
    {
        "id": "red_blue_review",
        "tool": "literature_red_blue_review",
        "stage": "self_audit",
        "description": "Attack and repair evidence synthesis across factuality, logic, and citations.",
        "risk": "low",
    },
    {
        "id": "save_reusable_skill",
        "tool": "skill_save",
        "stage": "sedimentation",
        "description": "Persist reusable workflows as local skills.",
        "risk": "high",
    },
    {
        "id": "save_project_wiki",
        "tool": "wiki_save",
        "stage": "sedimentation",
        "description": "Persist durable project knowledge in the local wiki.",
        "risk": "high",
    },
    {
        "id": "link_code_literature",
        "tool": "code_literature_link_save",
        "stage": "provenance",
        "description": "Link scientific code to DOI/PMID evidence for later retrieval.",
        "risk": "high",
    },
]


class WorkflowPreflightCheckTool(Tool):
    name = "workflow_preflight_check"
    description = (
        "Run a preflight/EDA gate before complex research work. Checks input files, "
        "count-matrix shape, sample groups, target replicates, and DESeq2 R environment "
        "when requested."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "User task or scientific question."},
            "input_files": {"type": "array", "items": {"type": "string"}, "description": "Input files to inspect."},
            "analysis_type": {"type": "string", "description": "Task type, e.g. deseq2, rnaseq, literature."},
            "target_group": {"type": "string", "description": "Optional target group, e.g. el."},
        },
        "required": ["task"],
    }

    def execute(
        self,
        task: str,
        input_files: list[str] | None = None,
        analysis_type: str = "",
        target_group: str = "",
    ) -> str:
        result = build_preflight(task, input_files or [], analysis_type, target_group)
        return json.dumps(result, ensure_ascii=False, indent=2)


class WorkflowPrimitiveLedgerTool(Tool):
    name = "workflow_primitive_ledger"
    description = "Return an OmicOS-style primitive ledger and recommended primitive picks for a task."
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "User task or scientific question."},
            "analysis_type": {"type": "string", "description": "Task type filter."},
            "pick_count": {"type": "integer", "description": "Number of primitives to recommend. Default 3."},
        },
        "required": ["task"],
    }

    def execute(self, task: str, analysis_type: str = "", pick_count: int = 3) -> str:
        result = build_primitive_ledger(task, analysis_type, pick_count)
        return json.dumps(result, ensure_ascii=False, indent=2)


class WorkflowPlanPrepareTool(Tool):
    name = "workflow_plan_prepare"
    description = (
        "Prepare durable plan artifacts for a complex task: plan.md for humans, "
        "plan.json for agents, and delegation.json for sub-agent roles. Includes "
        "preflight gates, primitive ledger, and quick/standard/strict options."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "User task or scientific question."},
            "input_files": {"type": "array", "items": {"type": "string"}, "description": "Input files to inspect."},
            "analysis_type": {"type": "string", "description": "Task type, e.g. deseq2, rnaseq, literature."},
            "target_group": {"type": "string", "description": "Optional target group, e.g. el."},
            "mode": {"type": "string", "description": "quick, standard, or strict. Default quick."},
            "output_dir": {"type": "string", "description": "Artifact directory. Defaults to .scicore/plans."},
        },
        "required": ["task"],
    }

    def execute(
        self,
        task: str,
        input_files: list[str] | None = None,
        analysis_type: str = "",
        target_group: str = "",
        mode: str = "quick",
        output_dir: str = ".scicore/plans",
    ) -> str:
        result = prepare_plan_artifacts(task, input_files or [], analysis_type, target_group, mode, output_dir)
        return json.dumps(result, ensure_ascii=False, indent=2)


def build_preflight(task: str, input_files: list[str], analysis_type: str = "", target_group: str = "") -> dict[str, Any]:
    analysis = _classify(task, analysis_type)
    checks: list[dict[str, Any]] = []
    observed: dict[str, Any] = {
        "analysis_type": analysis,
        "input_files": input_files,
        "target_group": str(target_group or "").strip().lower(),
    }

    if not input_files:
        checks.append(_check("input_files", "warning", "No explicit input file was provided."))
    for raw_path in input_files:
        path = Path(raw_path).expanduser().resolve()
        checks.append(_check("input_exists", "pass" if path.exists() else "error", str(path)))
        checks.append(_check("input_is_file", "pass" if path.is_file() else "error", str(path)))
        if path.is_file() and _looks_like_count_matrix(path, analysis):
            matrix = _inspect_count_matrix(path, target_group)
            observed.setdefault("count_matrices", []).append(matrix)
            checks.extend(matrix["checks"])

    if analysis in {"deseq2", "rnaseq"} or "deseq2" in str(task).lower():
        r_check = _check_deseq2_environment()
        observed["r_environment"] = r_check["observed"]
        checks.extend(r_check["checks"])

    status = "pass"
    if any(item["status"] == "error" for item in checks):
        status = "error"
    elif any(item["status"] == "warning" for item in checks):
        status = "warning"
    return {
        "status": status,
        "stop_confirm_required": status == "error",
        "checks": checks,
        "observed": observed,
    }


def build_primitive_ledger(task: str, analysis_type: str = "", pick_count: int = 3) -> dict[str, Any]:
    analysis = _classify(task, analysis_type)
    scored = []
    for primitive in PRIMITIVES:
        score = 0
        text = f"{primitive['id']} {primitive['tool']} {primitive['description']}".lower()
        if analysis and analysis in text:
            score += 3
        if "deseq2" in analysis and primitive["id"] in {
            "inspect_count_matrix",
            "infer_sample_groups",
            "check_omicverse_bulk_backend",
            "run_omicverse_pydge_tissue_vs_rest",
            "check_r_deseq2_env",
            "run_deseq2_tissue_vs_rest",
        }:
            score += 5
        if "literature" in analysis and primitive["id"] in {"red_blue_review", "export_literature_xlsx"}:
            score += 5
        if primitive["stage"] in {"eda_gate", "environment_gate", "self_audit"}:
            score += 1
        scored.append({**primitive, "score": score})
    recommended = sorted(scored, key=lambda item: (-item["score"], item["id"]))[: max(1, int(pick_count or 3))]
    return {
        "analysis_type": analysis,
        "ledger": scored,
        "recommended_primitives": recommended,
        "selection_rule": "Prefer gate/check primitives before execution primitives; pick distinct stages when possible.",
    }


def compose_options(task: str, analysis_type: str, target_group: str = "") -> list[dict[str, Any]]:
    analysis = _classify(task, analysis_type)
    target = target_group or "target"
    if analysis in {"deseq2", "rnaseq"}:
        return [
            {
                "name": "quick",
                "thesis": f"Run {target}-vs-rest bulk differential expression through OmicVerse pyDEG first, then use DESeq2 fallback when the backend is unavailable.",
                "stages": ["preflight", "OmicVerse backend gate", "pyDEG", "CSV summary"],
                "yield": ["omicverse_full_results.csv", "omicverse_significant.csv", "omicverse_summary.json"],
                "hardness_check": "Requires count matrix, >=2 target replicates, >=2 rest samples, and a working OmicVerse bulk backend.",
            },
            {
                "name": "standard",
                "thesis": "Add QC summaries, common diagnostic plots, and DESeq2 fallback verification after OmicVerse succeeds.",
                "stages": ["preflight", "OmicVerse pyDEG", "DESeq2 fallback check", "PCA/MA/volcano", "human report"],
                "yield": ["quick outputs", "plots", "summary.md"],
                "hardness_check": "Requires OmicVerse plus plotting packages; DESeq2/R is used as a fallback or verifier when configured.",
            },
            {
                "name": "strict",
                "thesis": "Use explicit metadata, richer covariates, OmicVerse execution, independent fallback checks, and verifier review before interpretation.",
                "stages": ["metadata validation", "design review", "OmicVerse", "fallback/verifier", "report"],
                "yield": ["validated design", "audited results", "limitations"],
                "hardness_check": "Requires user-confirmed design and biological metadata.",
            },
        ]
    return [
        {"name": "quick", "thesis": "Produce the smallest useful artifact.", "stages": ["preflight", "execute", "summary"], "yield": ["primary artifact"], "hardness_check": "Inputs must be sufficient."},
        {"name": "standard", "thesis": "Add validation and human-readable reporting.", "stages": ["preflight", "execute", "verify", "report"], "yield": ["artifact", "report"], "hardness_check": "Tools and data must pass preflight."},
        {"name": "strict", "thesis": "Use explicit plan approval and independent verification.", "stages": ["plan", "approve", "execute", "audit"], "yield": ["plan", "artifact", "audit"], "hardness_check": "May require user approval."},
    ]


def prepare_plan_artifacts(
    task: str,
    input_files: list[str],
    analysis_type: str = "",
    target_group: str = "",
    mode: str = "quick",
    output_dir: str = ".scicore/plans",
) -> dict[str, Any]:
    analysis = _classify(task, analysis_type)
    preflight = build_preflight(task, input_files, analysis, target_group)
    ledger = build_primitive_ledger(task, analysis, 4)
    options = compose_options(task, analysis, target_group)
    selected = next((item for item in options if item["name"] == str(mode or "quick").lower()), options[0])
    delegation = _delegation_for(analysis, selected)

    base = Path(output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    plan_id = datetime.now().strftime("plan_%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    plan_dir = base / plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "plan_id": plan_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "analysis_type": analysis,
        "mode": selected["name"],
        "input_files": input_files,
        "target_group": target_group,
        "preflight": preflight,
        "primitive_ledger": ledger,
        "options": options,
        "selected_option": selected,
        "delegation": delegation,
        "artifacts": {},
    }
    plan_md = _render_plan_markdown(plan)
    plan_json_path = plan_dir / "plan.json"
    plan_md_path = plan_dir / "plan.md"
    delegation_path = plan_dir / "delegation.json"
    plan_json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_md_path.write_text(plan_md, encoding="utf-8")
    delegation_path.write_text(json.dumps(delegation, ensure_ascii=False, indent=2), encoding="utf-8")
    plan["artifacts"] = {
        "plan_md": str(plan_md_path),
        "plan_json": str(plan_json_path),
        "delegation_json": str(delegation_path),
    }
    plan_json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "prepared",
        "plan_id": plan_id,
        "preflight_status": preflight["status"],
        "stop_confirm_required": preflight["stop_confirm_required"],
        "selected_option": selected,
        "recommended_primitives": ledger["recommended_primitives"],
        "artifacts": plan["artifacts"],
    }


def _classify(task: str, analysis_type: str = "") -> str:
    explicit = str(analysis_type or "").strip().lower()
    if explicit:
        return explicit
    lowered = str(task or "").lower()
    if "deseq2" in lowered:
        return "deseq2"
    if any(term in lowered for term in ("rnaseq", "rna-seq", "transcriptome", "转录组")):
        return "rnaseq"
    if any(term in lowered for term in ("literature", "pubmed", "文献")):
        return "literature"
    return "general"


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    item = {"name": name, "status": status, "detail": detail}
    item.update(extra)
    return item


def _looks_like_count_matrix(path: Path, analysis: str) -> bool:
    return analysis in {"deseq2", "rnaseq"} or path.suffix.lower() in {".txt", ".tsv", ".csv"}


def _inspect_count_matrix(path: Path, target_group: str = "", max_rows: int = 10000) -> dict[str, Any]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    checks: list[dict[str, Any]] = []
    groups: Counter[str] = Counter()
    non_integer_values = 0
    all_zero_rows = 0
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        sample_cols = [col.strip() for col in header[1:] if col.strip()]
        checks.append(_check("count_matrix_has_gene_and_samples", "pass" if len(sample_cols) >= 2 else "error", f"{len(sample_cols)} sample columns"))
        for sample in sample_cols:
            match = re.match(r"([A-Za-z]+)", sample)
            if match:
                groups[match.group(1).lower()] += 1
        checks.append(_check("sample_groups_inferred", "pass" if groups else "warning", json.dumps(dict(groups), ensure_ascii=False)))
        for row_count, row in enumerate(reader, start=1):
            if row_count > max_rows:
                break
            values = row[1:]
            parsed = []
            for value in values:
                try:
                    numeric = float(value)
                    parsed.append(numeric)
                    if numeric != int(numeric):
                        non_integer_values += 1
                except ValueError:
                    non_integer_values += 1
            if parsed and all(value == 0 for value in parsed):
                all_zero_rows += 1
    checks.append(_check("count_values_integer_like", "pass" if non_integer_values == 0 else "error", f"{non_integer_values} non-integer/non-numeric values in scanned rows"))
    if target_group:
        target = target_group.lower()
        target_count = groups.get(target, 0)
        rest_count = sum(groups.values()) - target_count
        checks.append(_check("target_replicates", "pass" if target_count >= 2 else "error", f"{target} samples: {target_count}"))
        checks.append(_check("rest_replicates", "pass" if rest_count >= 2 else "error", f"rest samples: {rest_count}"))
    return {
        "path": str(path),
        "delimiter": delimiter,
        "sample_count": len(sample_cols),
        "sample_groups": dict(groups),
        "rows_scanned": min(row_count, max_rows),
        "all_zero_rows_seen": all_zero_rows,
        "checks": checks,
    }


def _check_deseq2_environment() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rscript = _find_rscript()
    if not rscript:
        checks.append(_check("rscript_available", "error", "Rscript not found on PATH or common install paths."))
        return {"checks": checks, "observed": {"rscript": ""}}
    checks.append(_check("rscript_available", "pass", rscript))
    code = (
        "cat('rlang=', as.character(packageVersion('rlang')), '\\n', sep=''); "
        "suppressPackageStartupMessages(library(DESeq2)); "
        "cat('DESeq2=', as.character(packageVersion('DESeq2')), '\\n', sep='')"
    )
    try:
        proc = subprocess.run(
            [rscript, "--no-save", "--no-restore", "--no-environ", "-e", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        checks.append(_check("deseq2_loads", "error", f"R environment check failed: {exc}"))
        return {"checks": checks, "observed": {"rscript": rscript}}
    status = "pass" if proc.returncode == 0 else "error"
    detail = (proc.stdout + proc.stderr).strip()[-1200:]
    checks.append(_check("deseq2_loads", status, detail or f"exit_code={proc.returncode}"))
    return {
        "checks": checks,
        "observed": {
            "rscript": rscript,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-1200:],
            "stderr": proc.stderr[-1200:],
        },
    }


def _delegation_for(analysis: str, selected: dict[str, Any]) -> dict[str, Any]:
    jobs = [
        {"name": "preflight", "role": "explorer", "task": "Inspect inputs and report missing information."},
        {"name": "execute", "role": "bio_worker", "task": f"Execute selected {selected['name']} plan using approved deterministic tools."},
        {"name": "verify", "role": "verifier", "task": "Inspect artifacts and verify the result before final claims."},
    ]
    if analysis == "literature":
        jobs.insert(1, {"name": "red_blue", "role": "planner", "task": "Run red-blue evidence critique before synthesis."})
    return {"coordination": "centralized_orchestrator", "jobs": jobs}


def _render_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# SciCoreAgent Plan: {plan['plan_id']}",
        "",
        f"- Task: {plan['task']}",
        f"- Analysis type: {plan['analysis_type']}",
        f"- Mode: {plan['mode']}",
        f"- Preflight status: {plan['preflight']['status']}",
        "",
        "## Clarify Checklist / EDA Gate",
        "",
    ]
    for check in plan["preflight"]["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['detail']}")
    lines.extend(["", "## Three Options", ""])
    for option in plan["options"]:
        lines.append(f"### {option['name']}")
        lines.append(f"- Thesis: {option['thesis']}")
        lines.append(f"- Stages: {', '.join(option['stages'])}")
        lines.append(f"- Yield: {', '.join(option['yield'])}")
        lines.append(f"- Hardness check: {option['hardness_check']}")
        lines.append("")
    lines.extend(["## Recommended Primitives", ""])
    for primitive in plan["primitive_ledger"]["recommended_primitives"]:
        lines.append(f"- `{primitive['id']}` via `{primitive['tool']}` ({primitive['stage']})")
    lines.extend(["", "## Delegation Plan", ""])
    for job in plan["delegation"]["jobs"]:
        lines.append(f"- {job['name']}: {job['role']} - {job['task']}")
    return "\n".join(lines) + "\n"
