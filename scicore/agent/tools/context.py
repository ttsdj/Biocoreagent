"""Context window observability tool."""

from __future__ import annotations

import json

from .base import Tool


class ContextStatusTool(Tool):
    name = "context_status"
    description = "Show estimated context usage, compaction thresholds, and the next compaction action."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    _parent_agent = None

    def execute(self) -> str:
        if self._parent_agent is None:
            return "Error: context_status tool not initialized (no parent agent)"
        return json.dumps(self._parent_agent.context.status(self._parent_agent.messages), indent=2)
