"""Bioinformatics workflow skeleton helper."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..base import Tool


class BioWorkflowSketchTool(Tool):
    name = "bio_workflow_sketch"
    description = (
        "Draft a Snakemake or Nextflow workflow skeleton from explicitly provided "
        "bioinformatics requirements. Reports missing metadata instead of guessing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "workflow_type": {
                "type": "string",
                "enum": ["snakemake", "nextflow"],
                "description": "Workflow manager to sketch.",
            },
            "task_description": {
                "type": "string",
                "description": "User-provided task details and explicit metadata.",
            },
            "input_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit input file paths provided by the user.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory where workflow outputs should be placed.",
            },
        },
        "required": ["workflow_type", "task_description", "input_files", "output_dir"],
    }

    def execute(
        self,
        workflow_type: str,
        task_description: str,
        input_files: list[str],
        output_dir: str,
    ) -> str:
        workflow_type = workflow_type.lower()
        if workflow_type not in {"snakemake", "nextflow"}:
            return "Error: unsupported workflow_type. Supported: snakemake, nextflow"
        if not input_files:
            return "Error: input_files must contain at least one explicit path"

        missing = _missing_information(task_description)
        suggested_path = str(Path(output_dir) / ("Snakefile" if workflow_type == "snakemake" else "main.nf"))
        result = {
            "can_generate": not missing,
            "workflow_type": workflow_type,
            "suggested_file_path": suggested_path,
            "missing_information": missing,
            "uncertainties": missing,
            "input_files": input_files,
            "output_dir": output_dir,
            "message": "",
            "workflow_draft": None,
        }

        if missing:
            result["message"] = "Cannot generate executable workflow until required inputs are provided."
            return json.dumps(result, indent=2)

        if workflow_type == "snakemake":
            result["workflow_draft"] = _snakemake_draft(input_files, output_dir, task_description)
        else:
            result["workflow_draft"] = _nextflow_draft(input_files, output_dir, task_description)
        result["message"] = "Draft is a non-executed skeleton. Review all parameters before running."
        return json.dumps(result, indent=2)


def _missing_information(text: str) -> list[str]:
    checks = {
        "reference_genome_path": r"(reference_genome_path|reference genome|参考基因组)\s*[:=]\s*\S+",
        "single_end_or_paired_end": r"\b(single-end|single end|single_end|paired-end|paired end|paired_end|SE|PE)\b",
        "tool_choice": r"(tool_choice|tool choice|aligner|mapper|quantifier|工具)\s*[:=]\s*\S+",
        "species": r"(species|物种)\s*[:=]\s*\S+",
        "threads_or_resources": r"(threads|cores|cpus|memory|内存|线程)\s*[:=]\s*\S+",
    }
    missing = []
    for name, pattern in checks.items():
        if not re.search(pattern, text, flags=re.IGNORECASE):
            missing.append(name)
    return missing


def _snakemake_draft(input_files: list[str], output_dir: str, task_description: str) -> str:
    input_list = ",\n        ".join(repr(path) for path in input_files)
    return f"""# Generated skeleton only; not executed.
# Explicit task description:
# {task_description}

INPUT_FILES = [
        {input_list}
]
OUTPUT_DIR = {output_dir!r}

rule all:
    input:
        f"{{OUTPUT_DIR}}/complete.txt"

rule placeholder_step:
    input:
        INPUT_FILES
    output:
        f"{{OUTPUT_DIR}}/complete.txt"
    shell:
        "echo Review tool_choice, reference_genome_path, resources, and sample metadata before replacing this placeholder > {{output}}"
"""


def _nextflow_draft(input_files: list[str], output_dir: str, task_description: str) -> str:
    input_list = ", ".join(repr(path) for path in input_files)
    return f"""// Generated skeleton only; not executed.
// Explicit task description:
// {task_description}

params.input_files = [{input_list}]
params.outdir = {output_dir!r}

workflow {{
    Channel
        .fromList(params.input_files)
        .set {{ input_files_ch }}

    placeholder_step(input_files_ch)
}}

process placeholder_step {{
    publishDir params.outdir, mode: 'copy'

    input:
    path input_file

    output:
    path "complete.txt"

    script:
    '''
    echo "Review tool_choice, reference_genome_path, resources, and sample metadata before replacing this placeholder" > complete.txt
    '''
}}
"""
