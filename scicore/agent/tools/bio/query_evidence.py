"""Query the protocol evidence store.

Provides keyword search and filtered lookup across protocols, tasks,
resources, and raw chunks in the evidence store.

Reference: SciCoreAgent 需求文档, Section 9.3.
"""

from __future__ import annotations

import json

from ..base import Tool
from ...bio.evidence_store import EvidenceStore


class BioQueryEvidenceTool(Tool):
    name = "bio_query_evidence"
    description = (
        "Search the evidence store for protocol-related information. "
        "Returns matching protocol cards, task cards, resource cards, "
        "or raw chunks with their evidence span metadata. "
        "Use before extraction to review ingested documents, or during "
        "replication planning to find relevant evidence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword query to search for.",
            },
            "search_type": {
                "type": "string",
                "enum": ["all", "protocols", "tasks", "resources", "chunks"],
                "description": "Type of evidence to search. Default 'all'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return. Default 10.",
            },
            "source_id": {
                "type": "string",
                "description": (
                    "Optional: filter results to a specific source document ID "
                    "(returned by bio_ingest_protocol)."
                ),
            },
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str,
        search_type: str = "all",
        limit: int = 10,
        source_id: str = "",
    ) -> str:
        try:
            store = EvidenceStore()

            if search_type == "chunks" and source_id:
                # Direct chunk lookup by source_id.
                all_chunks = store.get_chunks(source_id)
                terms = [t.lower() for t in query.lower().split() if t.strip()]
                if terms:
                    filtered = [
                        c for c in all_chunks
                        if any(term in c.get("text", "").lower() for term in terms)
                    ]
                else:
                    filtered = all_chunks
                results = filtered[:limit]
            else:
                raw = store.search_by_type(query, search_type=search_type, limit=limit)
                if source_id:
                    results = [
                        r for r in raw
                        if r.get("source_id") == source_id
                        or r.get("protocol_id") == source_id
                    ][:limit]
                else:
                    results = raw

            # Build compact summaries.
            summaries = []
            for row in results:
                summary = {
                    "source_id": row.get("source_id", row.get("protocol_id", "")),
                    "type": _guess_type(row),
                }
                if "title" in row:
                    summary["title"] = row["title"]
                if "task_name" in row:
                    summary["task_name"] = row["task_name"]
                    summary["task_type"] = row.get("task_type", "")
                if "resource_name" in row:
                    summary["resource_name"] = row["resource_name"]
                    summary["resource_type"] = row.get("resource_type", "")
                if "text" in row:
                    summary["snippet"] = row["text"][:300]
                    summary["chunk_id"] = row.get("chunk_id", "")
                if "research_goal" in row:
                    summary["research_goal"] = row["research_goal"][:200]
                if "missing_information" in row:
                    summary["missing_information"] = row["missing_information"]
                summaries.append(summary)

            return json.dumps({
                "results": summaries,
                "count": len(summaries),
                "query": query,
                "search_type": search_type,
            }, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"Error querying evidence: {e}"


def _guess_type(row: dict) -> str:
    """Guess the evidence type from row keys."""
    if "task_name" in row:
        return "task"
    if "resource_name" in row:
        return "resource"
    if "text" in row and "chunk_id" in row:
        return "chunk"
    if "title" in row or "research_goal" in row:
        return "protocol"
    return "unknown"
