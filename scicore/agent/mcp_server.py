"""Minimal MCP stdio adapter for SciCoreAgent tools.

This adapter exposes the existing Tool registry through the MCP JSON-RPC
methods needed by common clients: initialize, tools/list, and tools/call.
It intentionally avoids a runtime SDK dependency so the npm package remains
easy to install on machines that only have Python and Node.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .tools import ALL_TOOLS
from .tools.base import Tool

JSON = dict[str, Any]


PROTOCOL_VERSION = "2025-06-18"


def tool_to_mcp(tool: Tool) -> JSON:
    schema = tool.schema()["function"]
    return {
        "name": schema["name"],
        "title": getattr(tool, "title", schema["name"]),
        "description": schema.get("description", ""),
        "inputSchema": schema.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }


def list_mcp_tools(tools: list[Tool] | None = None) -> list[JSON]:
    return [tool_to_mcp(tool) for tool in (tools or ALL_TOOLS)]


def call_mcp_tool(name: str, arguments: dict[str, Any] | None = None, tools: list[Tool] | None = None) -> JSON:
    arguments = arguments or {}
    registry = {tool.name: tool for tool in (tools or ALL_TOOLS)}
    tool = registry.get(name)
    if tool is None:
        return _error_content(f"unknown tool: {name}")
    try:
        result = tool.execute(**arguments)
    except TypeError as e:
        return _error_content(f"bad arguments for {name}: {e}")
    except Exception as e:
        return _error_content(f"tool execution failed for {name}: {e}")

    text = str(result)
    is_error = text.startswith(("Error:", "Blocked", "Confirmation required"))
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_jsonrpc(request: JSON, tools: list[Tool] | None = None) -> JSON | None:
    if request.get("method", "").startswith("notifications/"):
        return None

    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    try:
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "serverInfo": {"name": "scicore.bio", "version": __version__},
                "capabilities": {"tools": {"listChanged": False}},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": list_mcp_tools(tools)}
        elif method == "tools/call":
            result = call_mcp_tool(
                name=params.get("name", ""),
                arguments=params.get("arguments") or {},
                tools=tools,
            )
        else:
            return _jsonrpc_error(req_id, -32601, f"method not found: {method}")
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as e:
        return _jsonrpc_error(req_id, -32603, str(e))


def run_stdio() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = _jsonrpc_error(None, -32700, f"parse error: {e}")
        else:
            response = handle_jsonrpc(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scicore-mcp", description="Expose SciCoreAgent tools as an MCP stdio server.")
    parser.add_argument("--stdio", action="store_true", help="Run the MCP server over JSON-RPC stdio.")
    parser.add_argument("--list-tools", action="store_true", help="Print MCP tool metadata as JSON and exit.")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps({"tools": list_mcp_tools()}, indent=2))
        return
    run_stdio()


def _error_content(message: str) -> JSON:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> JSON:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    main()
