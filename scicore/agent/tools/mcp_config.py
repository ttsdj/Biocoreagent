"""Tools for registering external MCP server configurations."""

from __future__ import annotations

import json

from .base import Tool
from ..mcp_config import list_mcp_servers, register_mcp_server, unregister_mcp_server


class MCPRegisterTool(Tool):
    name = "mcp_register"
    description = (
        "Register an external MCP server so SciCoreAgent can connect to it "
        "and use its tools. The server will auto-connect on the next agent "
        "session. Store only command and env var names — never secret values."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Stable server name (letters, numbers, -, _)."},
            "command": {"type": "string", "description": "Executable command, e.g. npx or python."},
            "args": {"type": "array", "items": {"type": "string"}, "description": "Command arguments."},
            "description": {"type": "string", "description": "What this MCP server provides."},
            "env_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Environment variable names required by the server. Values are not stored.",
            },
            "overwrite": {"type": "boolean", "description": "Overwrite an existing registration. Default false."},
        },
        "required": ["name", "command"],
    }

    def execute(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        description: str = "",
        env_keys: list[str] | None = None,
        overwrite: bool = False,
    ) -> str:
        try:
            config = register_mcp_server(
                name=name,
                command=command,
                args=args or [],
                description=description,
                env_keys=env_keys or [],
                overwrite=overwrite,
            )
            return json.dumps({
                "registered": True,
                "name": config.name,
                "command": config.command,
                "args": config.args,
                "description": config.description,
                "hint": "This MCP server will connect on the next agent session.",
            }, indent=2)
        except Exception as e:
            return f"Error: {e}"


class MCPListTool(Tool):
    name = "mcp_list"
    description = (
        "List registered external MCP servers with their configuration "
        "and live connection status (if currently connected)."
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        configs = list_mcp_servers()
        result = []

        # Try to get live status from the running provider.
        live_status: dict[str, dict] = {}
        try:
            from ..mcp_client import get_mcp_provider
            provider = get_mcp_provider()
            for s in provider.list_servers():
                live_status[s["name"]] = s
        except Exception:
            pass

        for cfg in configs:
            entry = {
                "name": cfg.name,
                "command": cfg.command,
                "args": cfg.args,
                "description": cfg.description,
                "env_keys": cfg.env_keys or [],
                "configured": True,
            }
            if cfg.name in live_status:
                ls = live_status[cfg.name]
                entry["connected"] = ls.get("alive", False)
                entry["tool_count"] = ls.get("tool_count", 0)
                entry["server_info"] = ls.get("server_info", {})
            else:
                entry["connected"] = False
                entry["tool_count"] = 0
            result.append(entry)

        return json.dumps({"mcp_servers": result}, indent=2)


class MCPUnregisterTool(Tool):
    name = "mcp_unregister"
    description = "Remove an external MCP server registration by name."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Server name to unregister."}},
        "required": ["name"],
    }

    def execute(self, name: str) -> str:
        try:
            # Also disconnect if currently running.
            try:
                from ..mcp_client import get_mcp_provider
                get_mcp_provider().remove_client(name)
            except Exception:
                pass
            removed = unregister_mcp_server(name)
            return json.dumps({"name": name, "removed": removed}, indent=2)
        except Exception as e:
            return f"Error: {e}"
