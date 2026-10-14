"""Persistent external MCP server configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MCP_CONFIG = Path(".scicore") / "mcp_servers.json"
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$")


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    description: str = ""
    env_keys: list[str] | None = None


def mcp_config_path(root: str | Path | None = None) -> Path:
    return Path(root or ".").resolve() / DEFAULT_MCP_CONFIG


def register_mcp_server(
    name: str,
    command: str,
    args: list[str] | None = None,
    description: str = "",
    env_keys: list[str] | None = None,
    overwrite: bool = False,
    root: str | Path | None = None,
) -> MCPServerConfig:
    name = validate_mcp_name(name)
    if not command.strip():
        raise ValueError("command is required")
    cfg = MCPServerConfig(
        name=name,
        command=command.strip(),
        args=[str(arg) for arg in (args or [])],
        description=description.strip(),
        env_keys=[str(key) for key in (env_keys or [])],
    )
    data = _load(root)
    if name in data and not overwrite:
        raise FileExistsError(f"MCP server already registered: {name}")
    data[name] = _to_dict(cfg)
    _save(data, root)
    return cfg


def list_mcp_servers(root: str | Path | None = None) -> list[MCPServerConfig]:
    data = _load(root)
    return [_from_dict(item) for _, item in sorted(data.items())]


def unregister_mcp_server(name: str, root: str | Path | None = None) -> bool:
    name = validate_mcp_name(name)
    data = _load(root)
    existed = name in data
    if existed:
        data.pop(name)
        _save(data, root)
    return existed


def mcp_prompt_context(root: str | Path | None = None, limit: int = 8) -> str:
    servers = list_mcp_servers(root)[:limit]
    if not servers:
        return "No external MCP servers registered."
    lines = []
    for server in servers:
        env = f" env_keys={','.join(server.env_keys or [])}" if server.env_keys else ""
        lines.append(f"- {server.name}: {server.description or server.command}{env}")
    return "\n".join(lines)


def validate_mcp_name(name: str) -> str:
    clean = name.strip()
    if not _NAME_RE.match(clean):
        raise ValueError("MCP server name must be 2-64 chars: letters, numbers, '-' or '_'")
    return clean


def _load(root: str | Path | None = None) -> dict:
    path = mcp_config_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict, root: str | Path | None = None) -> None:
    path = mcp_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _to_dict(config: MCPServerConfig) -> dict:
    return {
        "name": config.name,
        "command": config.command,
        "args": config.args,
        "description": config.description,
        "env_keys": config.env_keys or [],
    }


def _from_dict(data: dict) -> MCPServerConfig:
    return MCPServerConfig(
        name=data["name"],
        command=data["command"],
        args=list(data.get("args") or []),
        description=data.get("description", ""),
        env_keys=list(data.get("env_keys") or []),
    )
