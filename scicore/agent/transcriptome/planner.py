"""Planner for bulk transcriptome capability chains."""

from __future__ import annotations

from typing import Any

from .registry import get_capability
from .state import inspect_transcriptome_state
from .verifier import verify_plan


def plan_transcriptome_workflow(
    task: str,
    count_matrix_path: str = "",
    target_group: str = "",
    mode: str = "quick",
) -> dict[str, Any]:
    lowered = str(task or "").lower()
    selected_mode = str(mode or "quick").lower()
    state = inspect_transcriptome_state(count_matrix_path=count_matrix_path, target_group=target_group)
    steps = ["inspect_count_matrix", "infer_sample_groups"]
    deg_keywords = ("deseq2", "differential", "deg", "omicverse", "差异", "比较", "表达")
    if any(keyword in lowered for keyword in deg_keywords):
        steps.extend(["check_omicverse_bulk_backend", "omicverse_pydge_tissue_vs_rest"])
        if selected_mode in {"standard", "strict"}:
            steps.extend(["check_deseq2_environment", "deseq2_tissue_vs_rest"])
    else:
        steps.append("lightweight_group_compare")
    if selected_mode in {"standard", "strict"}:
        steps.append("emit_transcriptome_report")
    contracts = [get_capability(step) for step in steps]
    verification = verify_plan(steps, state)
    return {
        "task": task,
        "mode": selected_mode,
        "state": state.to_dict(),
        "steps": steps,
        "capabilities": contracts,
        "verification": verification,
        "execution_policy": (
            "Execute only capabilities whose requires are satisfied. If blocked, run the missing prerequisite "
            "or return a stop-confirm diagnostic instead of generating free-form code."
        ),
    }
