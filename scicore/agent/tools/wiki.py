"""Wiki knowledge-base tools for SciCoreAgent.

Three tools that let the agent persist, search, and list factual
knowledge points in a local JSONL wiki.  The wiki is designed for
cross-session knowledge accumulation: things the user has explained,
questions that have been answered, and concepts worth remembering.

Follows the same Tool subclass pattern as tools/skills.py.
"""

from __future__ import annotations

import json

from .base import Tool
from ..wiki import save_entry, search_entries, list_entries, get_entry


class WikiSaveTool(Tool):
    name = "wiki_save"
    description = (
        "Save a factual knowledge point or FAQ to the local wiki for "
        "future reference across sessions. Use this when the user explains "
        "a concept, answers a question that may come up again, or describes "
        "a bioinformatics workflow worth remembering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the knowledge entry. Should be concise and descriptive.",
            },
            "content": {
                "type": "string",
                "description": "The knowledge content in Markdown. Include relevant details, references, and context.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization (e.g. rna-seq, alignment, quality-control).",
            },
            "source_session": {
                "type": "string",
                "description": "Brief note about what prompted this entry (e.g. 'user asked about STAR alignment parameters').",
            },
        },
        "required": ["title", "content"],
    }

    def execute(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        source_session: str = "",
    ) -> str:
        try:
            entry = save_entry(
                title=title,
                content=content,
                tags=tags or [],
                source_session=source_session,
            )
            return json.dumps(
                {
                    "saved": True,
                    "id": entry["id"],
                    "title": entry["title"],
                    "tags": entry["tags"],
                },
                indent=2,
            )
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error saving wiki entry: {e}"


class WikiSearchTool(Tool):
    name = "wiki_search"
    description = (
        "Search the local wiki for knowledge entries matching a query. "
        "Use before answering questions about bioinformatics concepts, "
        "tools, or workflows that may have been previously discussed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword query to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Default 10.",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, limit: int = 10) -> str:
        try:
            entries = search_entries(query, limit=limit)
            # Return a compact summary rather than full content
            results = [
                {
                    "id": e["id"],
                    "title": e["title"],
                    "tags": e.get("tags", []),
                    "snippet": e.get("content", "")[:200],
                    "created_at": e.get("created_at", ""),
                }
                for e in entries
            ]
            return json.dumps({"entries": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error searching wiki: {e}"


class WikiListTool(Tool):
    name = "wiki_list"
    description = (
        "List recent entries in the local wiki knowledge base. "
        "Returns titles, tags, and snippets sorted by recency."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries. Default 20.",
            },
        },
        "required": [],
    }

    def execute(self, limit: int = 20) -> str:
        try:
            entries = list_entries(limit=limit)
            results = [
                {
                    "id": e["id"],
                    "title": e["title"],
                    "tags": e.get("tags", []),
                    "snippet": e.get("content", "")[:120],
                    "created_at": e.get("created_at", ""),
                }
                for e in entries
            ]
            return json.dumps({"entries": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error listing wiki: {e}"
