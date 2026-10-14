"""Extract structured protocol evidence from ingested chunks.

This tool is LLM-assisted: the agent first reads chunks via
bio_query_evidence, synthesises the extraction, then calls this tool
with the structured JSON output.  The tool validates that every
extracted field references real EvidenceSpans in the store.

Reference: SciCoreAgent 需求文档, Section 9.2.
"""

from __future__ import annotations

import json
import uuid

from ..base import Tool
from ...bio.evidence_store import EvidenceStore
from ...bio.schemas import (
    EvidenceSpan,
    ProtocolCard,
    TaskCard,
    ResourceCard,
)


class BioExtractProtocolTool(Tool):
    name = "bio_extract_protocol"
    description = (
        "Extract structured ProtocolCard, TaskCard, and ResourceCard data "
        "from previously ingested evidence chunks.  The LLM should examine "
        "the chunks (via bio_query_evidence) and provide the extracted "
        "fields as JSON.  Every extracted value must cite an EvidenceSpan "
        "referencing the source chunk.  Fields not found in the source "
        "MUST be left empty."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "Source ID returned by bio_ingest_protocol.",
            },
            "protocol_title": {
                "type": "string",
                "description": "Extracted protocol title from the document.",
            },
            "source_type": {
                "type": "string",
                "description": "Source type: protocol, paper, readme, workflow_doc, or other.",
            },
            "research_goal": {
                "type": "string",
                "description": "Research goal stated in the protocol (empty if not found).",
            },
            "organism": {
                "type": "string",
                "description": "Organism mentioned in the protocol (empty if not found).",
            },
            "assay_type": {
                "type": "string",
                "description": "Assay type: RNA-seq, ChIP-seq, ATAC-seq, WGS, etc. (empty if not found).",
            },
            "input_data": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Input data types or file descriptions.",
            },
            "output_data": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Output data types or file descriptions.",
            },
            "tasks_json": {
                "type": "string",
                "description": (
                    "JSON array of TaskCard objects. Each must have: task_id, task_name, "
                    "task_type, description, inputs, outputs, tools_required, "
                    "databases_required, software_required, parameters (dict), "
                    "confidence (high/medium/low), and evidence_spans (array of "
                    "{source_id, source_type, chunk_id, location, text})."
                ),
            },
            "resources_json": {
                "type": "string",
                "description": (
                    "JSON array of ResourceCard objects. Each must have: resource_id, "
                    "resource_name, resource_type, version, purpose, installation_info, "
                    "and evidence_spans."
                ),
            },
            "missing_information": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of information the protocol does not provide.",
            },
        },
        "required": ["source_id", "protocol_title"],
    }

    def execute(
        self,
        source_id: str,
        protocol_title: str,
        source_type: str = "other",
        research_goal: str = "",
        organism: str = "",
        assay_type: str = "",
        input_data: list[str] | None = None,
        output_data: list[str] | None = None,
        tasks_json: str = "[]",
        resources_json: str = "[]",
        missing_information: list[str] | None = None,
    ) -> str:
        store = EvidenceStore()
        warnings: list[str] = []

        # --- validate chunks exist ------------------------------------------
        existing_chunks = store.get_chunks(source_id)
        if not existing_chunks:
            return json.dumps({
                "success": False,
                "error": (
                    f"No chunks found for source_id='{source_id}'. "
                    "Run bio_ingest_protocol first."
                ),
            })

        # --- parse tasks ----------------------------------------------------
        try:
            tasks_raw = json.loads(tasks_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"tasks_json is not valid JSON: {e}",
            })

        if not isinstance(tasks_raw, list):
            return json.dumps({
                "success": False,
                "error": "tasks_json must be a JSON array.",
            })

        tasks: list[TaskCard] = []
        for i, t in enumerate(tasks_raw):
            try:
                # Generate task_id if not provided.
                if "task_id" not in t:
                    t["task_id"] = f"{source_id}_task_{i:03d}"
                if "protocol_id" not in t:
                    t["protocol_id"] = source_id
                tasks.append(TaskCard(**t))
            except Exception as e:
                warnings.append(f"TaskCard[{i}] validation failed: {e}")

        # --- parse resources ------------------------------------------------
        try:
            resources_raw = json.loads(resources_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"resources_json is not valid JSON: {e}",
            })

        if not isinstance(resources_raw, list):
            return json.dumps({
                "success": False,
                "error": "resources_json must be a JSON array.",
            })

        resources: list[ResourceCard] = []
        for i, r in enumerate(resources_raw):
            try:
                if "resource_id" not in r:
                    r["resource_id"] = f"{source_id}_resource_{i:03d}"
                if "protocol_id" not in r:
                    r["protocol_id"] = source_id
                resources.append(ResourceCard(**r))
            except Exception as e:
                warnings.append(f"ResourceCard[{i}] validation failed: {e}")

        # --- build protocol card --------------------------------------------
        protocol_id = source_id
        protocol = ProtocolCard(
            protocol_id=protocol_id,
            title=protocol_title,
            source_type=source_type,
            source_path="",  # stored in chunk metadata
            research_goal=research_goal,
            organism=organism,
            assay_type=assay_type,
            input_data=input_data or [],
            output_data=output_data or [],
            key_task_ids=[t.task_id for t in tasks],
            resource_ids=[r.resource_id for r in resources],
            missing_information=missing_information or [],
        )

        # --- persist to evidence store --------------------------------------
        store.add_protocol(protocol)
        for task in tasks:
            store.add_task(task)
        for resource in resources:
            store.add_resource(resource)

        return json.dumps({
            "success": True,
            "protocol_id": protocol_id,
            "protocol_card": protocol.model_dump(),
            "task_cards": [t.model_dump() for t in tasks],
            "resource_cards": [r.model_dump() for r in resources],
            "task_count": len(tasks),
            "resource_count": len(resources),
            "missing_information": missing_information or [],
            "warnings": warnings,
        }, indent=2, ensure_ascii=False, default=str)
