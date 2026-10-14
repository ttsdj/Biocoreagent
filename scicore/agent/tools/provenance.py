"""Tools for code-literature provenance memory."""

from __future__ import annotations

import json

from .base import Tool
from ..provenance import list_code_literature_links, save_code_literature_link, search_code_literature_links


class CodeLiteratureLinkSaveTool(Tool):
    name = "code_literature_link_save"
    description = (
        "Persist a traceable link between a code file/symbol and literature evidence. "
        "Use after generating or modifying scientific code based on papers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code_path": {"type": "string", "description": "Code file path inside the workspace."},
            "code_symbol": {"type": "string", "description": "Function/class/rule name if applicable."},
            "purpose": {"type": "string", "description": "Why this code exists or was modified."},
            "evidence_summary": {"type": "string", "description": "Short summary of the literature basis."},
            "literature_json": {"type": "string", "description": "JSON list of cited papers/articles."},
        },
        "required": ["code_path", "purpose", "literature_json"],
    }

    def execute(
        self,
        code_path: str,
        purpose: str,
        literature_json: str,
        code_symbol: str = "",
        evidence_summary: str = "",
    ) -> str:
        literature = json.loads(literature_json)
        if isinstance(literature, dict):
            literature = [literature]
        if not isinstance(literature, list):
            return "Error: literature_json must be a JSON list or object"
        entry = save_code_literature_link(
            code_path=code_path,
            code_symbol=code_symbol,
            purpose=purpose,
            evidence_summary=evidence_summary,
            literature=literature,
        )
        return json.dumps({"saved": True, "entry": entry}, ensure_ascii=False, indent=2)


class CodeLiteratureLinkSearchTool(Tool):
    name = "code_literature_link_search"
    description = "Search persisted code-literature provenance links by code path, DOI, PMID, title, or purpose."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }

    def execute(self, query: str, limit: int = 10) -> str:
        rows = search_code_literature_links(query=query, limit=int(limit or 10))
        return json.dumps({"links": rows, "count": len(rows)}, ensure_ascii=False, indent=2)


class CodeLiteratureLinkListTool(Tool):
    name = "code_literature_link_list"
    description = "List recent code-literature provenance links."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20},
        },
        "required": [],
    }

    def execute(self, limit: int = 20) -> str:
        rows = list_code_literature_links()[: int(limit or 20)]
        return json.dumps({"links": rows, "count": len(rows)}, ensure_ascii=False, indent=2)
