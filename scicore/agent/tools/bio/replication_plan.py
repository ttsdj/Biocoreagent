"""Generate a structured replication plan from protocol evidence.

Consolidates ProtocolCards, TaskCards, and ResourceCards from the evidence
store into a ReplicationPlan that separates automatable steps from those
requiring manual confirmation, and explicitly lists blockers.

Reference: SciCoreAgent 需求文档, Section 9.4.
"""

from __future__ import annotations

import json
import uuid

from ..base import Tool
from ...bio.evidence_store import EvidenceStore
from ...bio.schemas import ReplicationPlan, EvidenceSpan


class BioReplicationPlanTool(Tool):
    name = "bio_replication_plan"
    description = (
        "Generate a structured replication plan from evidence stored in "
        "the evidence store.  The LLM should construct the ReplicationPlan "
        "as JSON from the ProtocolCards, TaskCards, and ResourceCards "
        "retrieved via bio_query_evidence.  The plan explicitly separates "
        "what can be automated from what needs manual confirmation and "
        "what blocks execution entirely."
    )
    parameters = {
        "type": "object",
        "properties": {
            "protocol_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Protocol IDs to include in the replication plan.",
            },
            "plan_json": {
                "type": "string",
                "description": (
                    "Full ReplicationPlan as a JSON string. Must include: plan_id, "
                    "protocol_ids, research_goal, required_inputs, expected_outputs, "
                    "tasks (array of TaskCard), resources (array of ResourceCard), "
                    "missing_information, automatable_steps, manual_confirmation_required, "
                    "cannot_proceed_reasons, and evidence_summary (array of EvidenceSpan)."
                ),
            },
        },
        "required": ["protocol_ids", "plan_json"],
    }

    def execute(
        self,
        protocol_ids: list[str],
        plan_json: str,
    ) -> str:
        # --- validate input -------------------------------------------------
        if not protocol_ids:
            return json.dumps({
                "success": False,
                "error": "protocol_ids must not be empty.",
            })

        store = EvidenceStore()

        # Verify all protocol_ids exist in the store.
        missing_ids = []
        for pid in protocol_ids:
            if store.get_protocol(pid) is None:
                missing_ids.append(pid)

        if missing_ids:
            return json.dumps({
                "success": False,
                "error": f"Protocol(s) not found in evidence store: {missing_ids}",
                "hint": "Run bio_ingest_protocol and bio_extract_protocol first.",
            })

        # --- parse plan JSON ------------------------------------------------
        try:
            plan_raw = json.loads(plan_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"plan_json is not valid JSON: {e}",
            })

        # Auto-fill protocol_id on tasks and resources that lack it.
        default_pid = protocol_ids[0] if protocol_ids else ""
        for task in plan_raw.get("tasks", []):
            if not task.get("protocol_id"):
                task["protocol_id"] = default_pid
        for resource in plan_raw.get("resources", []):
            if not resource.get("protocol_id"):
                resource["protocol_id"] = default_pid

        # --- validate with Pydantic -----------------------------------------
        try:
            plan = ReplicationPlan(**plan_raw)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"ReplicationPlan validation failed: {e}",
                "hint": (
                    "Ensure plan_json has all required ReplicationPlan fields: "
                    "plan_id, protocol_ids, research_goal, required_inputs, "
                    "expected_outputs, tasks, resources, missing_information, "
                    "automatable_steps, manual_confirmation_required, "
                    "cannot_proceed_reasons, evidence_summary."
                ),
            })

        # --- hard safety checks ---------------------------------------------
        warnings: list[str] = []

        # Check that every task has evidence spans.
        for task in plan.tasks:
            if not task.evidence_spans:
                warnings.append(
                    f"Task '{task.task_name}' has no evidence_spans. "
                    "Every task must cite its source."
                )

        # Check that every resource has evidence spans.
        for resource in plan.resources:
            if not resource.evidence_spans:
                warnings.append(
                    f"Resource '{resource.resource_name}' has no evidence_spans. "
                    "Every resource must cite its source."
                )

        # Check that missing_information is populated if evidence is thin.
        if plan.missing_information and plan.cannot_proceed_reasons:
            # Both are populated — good, the LLM is being honest.
            pass
        elif plan.missing_information and not plan.cannot_proceed_reasons:
            # Has missing info but no blockers — may be acceptable.
            pass

        # Check that automatable steps don't claim execution.
        for step in plan.automatable_steps:
            if any(
                kw in step.lower()
                for kw in ["executed", "completed", "ran successfully"]
            ):
                warnings.append(
                    f"Automatable step claims execution: '{step}'. "
                    "Phrasing should describe what CAN be done, not what WAS done."
                )

        # --- determine workflow readiness -----------------------------------
        can_generate_workflow = (
            len(plan.cannot_proceed_reasons) == 0
            and len(plan.missing_information) == 0
            and len(plan.tasks) > 0
        )

        reason = ""
        if not can_generate_workflow:
            parts = []
            if plan.cannot_proceed_reasons:
                parts.append(
                    f"{len(plan.cannot_proceed_reasons)} blocker(s): "
                    + "; ".join(plan.cannot_proceed_reasons[:3])
                )
            if plan.missing_information:
                parts.append(
                    f"{len(plan.missing_information)} missing information items"
                )
            if not plan.tasks:
                parts.append("no tasks extracted")
            reason = ". ".join(parts)

        return json.dumps({
            "success": True,
            "plan_id": plan.plan_id,
            "plan": plan.model_dump(),
            "can_generate_workflow": can_generate_workflow,
            "reason": reason or "All evidence is sufficient for workflow generation.",
            "warnings": warnings,
        }, indent=2, ensure_ascii=False, default=str)
