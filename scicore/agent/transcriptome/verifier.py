"""Verify transcriptome capability plans against state contracts."""

from __future__ import annotations

from typing import Any

from .registry import get_capability
from .state import TranscriptomeState


STATE_FIELD_MAP = {
    "count_matrix_inspected": "count_matrix_inspected",
    "sample_columns": "sample_columns",
    "sample_groups": "sample_groups",
    "replicate_counts": "sample_groups",
    "target_has_replicates": "target_has_replicates",
    "rest_has_replicates": "rest_has_replicates",
    "deseq2_environment_checked": "deseq2_environment_checked",
    "omicverse_backend_checked": "omicverse_backend_checked",
    "differential_expression_completed": "differential_expression_completed",
}


def verify_plan(steps: list[str], state: TranscriptomeState | dict[str, Any]) -> dict[str, Any]:
    current = state if isinstance(state, TranscriptomeState) else TranscriptomeState(**state)
    results = []
    simulated = current.to_dict()
    for step in steps:
        capability = get_capability(step)
        missing = []
        for required in capability.get("requires", {}).get("state", []):
            field = STATE_FIELD_MAP.get(required, required)
            value = simulated.get(field)
            if not value:
                missing.append(required)
        status = "pass" if not missing else "blocked"
        results.append({
            "capability": step,
            "tool": capability["tool"],
            "status": status,
            "missing_state": missing,
            "requires": capability.get("requires", {}),
            "produces": capability.get("produces", {}),
        })
        if status == "pass":
            for produced in capability.get("produces", {}).get("state", []):
                field = STATE_FIELD_MAP.get(produced, produced)
                simulated[field] = True
    return {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "blocked",
        "steps": results,
        "final_state": simulated,
    }
