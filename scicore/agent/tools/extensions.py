"""Tools for inspecting reserved Skill and MCP extension interfaces."""

from __future__ import annotations

import json

from .base import Tool
from ..extensions import GLOBAL_EXTENSION_REGISTRY


class ExtensionsTool(Tool):
    name = "extensions"
    description = "Inspect registered Skill providers and MCP providers."
    parameters = {
        "type": "object",
        "properties": {
            "detail": {
                "type": "boolean",
                "description": "Whether to include detailed provider metadata. Default true.",
            }
        },
        "required": [],
    }

    def execute(self, detail: bool = True) -> str:
        data = GLOBAL_EXTENSION_REGISTRY.describe()
        if not detail:
            data = {
                "skill_count": len(data["skills"]),
                "mcp_server_count": len(data["mcp_servers"]),
            }
        return json.dumps(data, indent=2)
