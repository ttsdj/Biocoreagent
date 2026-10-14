"""Reserved Skill and MCP extension interfaces.

These interfaces are intentionally lightweight. They make the extension points
explicit without binding SciCoreAgent to one plugin framework too early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .tools.base import Tool


class SkillProvider(Protocol):
    name: str
    description: str

    def tools(self) -> list[Tool]:
        """Return tools contributed by this skill."""
        ...


class MCPProvider(Protocol):
    name: str
    description: str

    def tool_schemas(self) -> list[dict]:
        """Return MCP-backed tool schemas available to the agent."""
        ...

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP-backed tool and return a text observation."""
        ...


@dataclass
class ExtensionRegistry:
    skill_providers: dict[str, SkillProvider] = field(default_factory=dict)
    mcp_providers: dict[str, MCPProvider] = field(default_factory=dict)

    def register_skill(self, provider: SkillProvider) -> None:
        self.skill_providers[provider.name] = provider

    def register_mcp(self, provider: MCPProvider) -> None:
        self.mcp_providers[provider.name] = provider

    def skill_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for provider in self.skill_providers.values():
            tools.extend(provider.tools())
        return tools

    def mcp_tool_schemas(self) -> list[dict]:
        schemas: list[dict] = []
        for provider in self.mcp_providers.values():
            schemas.extend(provider.tool_schemas())
        return schemas

    def call_mcp_tool(self, name: str, arguments: dict) -> str | None:
        for provider in self.mcp_providers.values():
            for schema in provider.tool_schemas():
                # MCP tool schemas have "name" at the top level.
                # The agent wraps them in OpenAI function-calling format
                # ({type: "function", function: {name: ...}}) before sending
                # to the LLM, but the raw schema stores name directly.
                schema_name = schema.get("name") or schema.get("function", {}).get("name")
                if schema_name == name:
                    return provider.call_tool(name, arguments)
        return None

    def describe(self) -> dict:
        return {
            "skills": [
                {"name": p.name, "description": p.description, "tool_count": len(p.tools())}
                for p in self.skill_providers.values()
            ],
            "mcp_servers": [
                {"name": p.name, "description": p.description, "tool_count": len(p.tool_schemas())}
                for p in self.mcp_providers.values()
            ],
        }


GLOBAL_EXTENSION_REGISTRY = ExtensionRegistry()
