"""Tools exposing the bulk transcriptome capability layer."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Tool
from scicore.agent.transcriptome import (
    get_capability,
    inspect_transcriptome_state,
    list_capabilities,
    plan_transcriptome_workflow,
    verify_plan,
)
from scicore.agent.transcriptome.provenance import append_transcriptome_provenance
from scicore.agent.transcriptome.omicverse_backend import (
    check_omicverse_bulk_backend,
    run_omicverse_pydge_tissue_vs_rest,
)


class TranscriptomeCapabilityListTool(Tool):
    name = "transcriptome_capability_list"
    description = "List bulk transcriptome capability contracts with requires/produces/tool metadata."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Optional capability name to inspect."},
        },
        "required": [],
    }

    def execute(self, name: str = "") -> str:
        if name:
            result = get_capability(name)
        else:
            result = {
                "capabilities": [
                    {
                        "name": item["name"],
                        "tool": item["tool"],
                        "aliases": item.get("aliases", [])[:3],
                        "requires": item.get("requires", {}),
                        "produces": item.get("produces", {}),
                    }
                    for item in list_capabilities()
                ],
                "note": "Pass name=<capability> to inspect the full capability contract.",
            }
        return json.dumps(result, ensure_ascii=False, indent=2)


class TranscriptomeStateInspectTool(Tool):
    name = "transcriptome_state_inspect"
    description = "Inspect current bulk transcriptome object/file state before selecting capabilities."
    parameters = {
        "type": "object",
        "properties": {
            "count_matrix_path": {"type": "string", "description": "Gene-by-sample raw count matrix."},
            "metadata_path": {"type": "string", "description": "Optional sample metadata table."},
            "target_group": {"type": "string", "description": "Optional target group prefix, e.g. el."},
            "max_rows": {"type": "integer", "description": "Rows to scan. Default 10000."},
        },
        "required": ["count_matrix_path"],
    }

    def execute(
        self,
        count_matrix_path: str,
        metadata_path: str = "",
        target_group: str = "",
        max_rows: int = 10000,
    ) -> str:
        state = inspect_transcriptome_state(count_matrix_path, metadata_path, target_group, int(max_rows or 10000))
        return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)


class TranscriptomePlanTool(Tool):
    name = "transcriptome_plan"
    description = "Plan a bulk transcriptome workflow from capability contracts and current state."
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "User task or scientific question."},
            "count_matrix_path": {"type": "string", "description": "Optional raw count matrix."},
            "target_group": {"type": "string", "description": "Optional target group prefix."},
            "mode": {"type": "string", "description": "quick, standard, or strict. Default quick."},
        },
        "required": ["task"],
    }

    def execute(self, task: str, count_matrix_path: str = "", target_group: str = "", mode: str = "quick") -> str:
        result = plan_transcriptome_workflow(task, count_matrix_path, target_group, mode)
        state = result.get("state", {})
        result["state"] = {
            "count_matrix_path": state.get("count_matrix_path", ""),
            "sample_groups": state.get("sample_groups", {}),
            "target_group": state.get("target_group", ""),
            "target_replicates": state.get("target_replicates", 0),
            "rest_replicates": state.get("rest_replicates", 0),
            "blockers": state.get("blockers", []),
            "warnings": state.get("warnings", []),
        }
        result["capabilities"] = [
            {
                "name": item["name"],
                "tool": item["tool"],
                "failure_modes": item.get("failure_modes", []),
            }
            for item in result.get("capabilities", [])
        ]
        verification = result.get("verification", {})
        result["verification"] = {
            "status": verification.get("status", ""),
            "steps": [
                {
                    "capability": item.get("capability", ""),
                    "tool": item.get("tool", ""),
                    "status": item.get("status", ""),
                    "missing_state": item.get("missing_state", []),
                }
                for item in verification.get("steps", [])
            ],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


class TranscriptomePlanVerifyTool(Tool):
    name = "transcriptome_plan_verify"
    description = "Verify a transcriptome capability chain against a supplied transcriptome state."
    parameters = {
        "type": "object",
        "properties": {
            "steps_json": {"type": "string", "description": "JSON list of capability names."},
            "state_json": {"type": "string", "description": "Transcriptome state JSON object."},
        },
        "required": ["steps_json", "state_json"],
    }

    def execute(self, steps_json: str, state_json: str) -> str:
        steps = json.loads(steps_json)
        state = json.loads(state_json)
        result = verify_plan(steps, state)
        return json.dumps(result, ensure_ascii=False, indent=2)


class TranscriptomeProvenanceAppendTool(Tool):
    name = "transcriptome_provenance_append"
    description = "Append a transcriptome capability execution lineage record."
    parameters = {
        "type": "object",
        "properties": {
            "root": {"type": "string", "description": "Workspace root. Defaults to current directory."},
            "capability": {"type": "string", "description": "Capability name."},
            "tool": {"type": "string", "description": "Tool used to execute the capability."},
            "params_json": {"type": "string", "description": "JSON object of execution parameters."},
            "input_state_json": {"type": "string", "description": "JSON object before execution."},
            "output_state_json": {"type": "string", "description": "JSON object after execution."},
            "artifacts_json": {"type": "string", "description": "Optional JSON object of produced artifacts."},
        },
        "required": ["capability", "tool", "params_json", "input_state_json", "output_state_json"],
    }

    def execute(
        self,
        capability: str,
        tool: str,
        params_json: str,
        input_state_json: str,
        output_state_json: str,
        root: str = "",
        artifacts_json: str = "{}",
    ) -> str:
        entry = append_transcriptome_provenance(
            root=Path(root or "."),
            capability=capability,
            tool=tool,
            params=json.loads(params_json),
            input_state=json.loads(input_state_json),
            output_state=json.loads(output_state_json),
            artifacts=json.loads(artifacts_json or "{}"),
        )
        return json.dumps(entry, ensure_ascii=False, indent=2)


class TranscriptomeOmicVerseCheckTool(Tool):
    name = "transcriptome_omicverse_check"
    description = "Check whether OmicVerse bulk transcriptome backend is installed and exposes required pyDEG APIs."
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        return json.dumps(check_omicverse_bulk_backend(), ensure_ascii=False, indent=2)


class TranscriptomeOmicVerseDEGTool(Tool):
    name = "transcriptome_omicverse_deg"
    description = (
        "Run bulk target-vs-rest differential expression through OmicVerse ov.bulk.pyDEG. "
        "Returns backend_missing if omicverse is not installed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "count_matrix_path": {"type": "string", "description": "Gene-by-sample raw count matrix."},
            "target_group": {"type": "string", "description": "Target group prefix, e.g. el."},
            "output_dir": {"type": "string", "description": "Output directory. Defaults to matrix directory."},
            "method": {"type": "string", "description": "OmicVerse pyDEG method: DEseq2, edger, limma, ttest."},
            "alpha": {"type": "number", "description": "FDR/significance threshold. Default 0.05."},
            "n_cpus": {"type": "integer", "description": "CPU count for backend. Default 2."},
        },
        "required": ["count_matrix_path", "target_group"],
    }

    def execute(
        self,
        count_matrix_path: str,
        target_group: str,
        output_dir: str = "",
        method: str = "DEseq2",
        alpha: float = 0.05,
        n_cpus: int = 2,
    ) -> str:
        result = run_omicverse_pydge_tissue_vs_rest(
            count_matrix_path=count_matrix_path,
            target_group=target_group,
            output_dir=output_dir,
            method=method or "DEseq2",
            alpha=float(alpha),
            n_cpus=int(n_cpus or 2),
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
