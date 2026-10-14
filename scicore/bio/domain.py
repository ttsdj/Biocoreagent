"""Adapters that bring SciCoreAgent tools into the kernel execution contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scicore.agent.tools import ALL_TOOLS
from scicore.agent.mcp_client import get_mcp_provider


EXCLUDED_LEGACY_TOOLS = {
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "agent",
    "agent_start",
    "agent_status",
    "agent_team_start",
    "agent_team_status",
    "team_synthesize",
    "team_workspace",
    "context_status",
}

RISKY_PREFIXES = (
    "ssh_",
    "mcp_register",
    "mcp_unregister",
    "skill_install",
    "skill_save",
    "wiki_save",
    "user_profile_set",
    "bio_ingest",
    "bio_report",
    "bio_r_bridge",
    "bio_workflow",
    "bio_deseq2",
    "bio_contingency",
    "bio_regression",
    "workflow_plan_prepare",
    "transcriptome_provenance_append",
    "transcriptome_omicverse_deg",
    "pubmed_literature_review",
    "literature_export_xlsx",
    "code_literature_link_save",
)

PATH_KEYS = {
    "file_path",
    "source_path",
    "output_dir",
    "output_path",
    "output_docx",
    "metadata_path",
    "count_matrix_path",
    "input_file",
    "csv_path",
    "merge_csv_path",
    "counts_path",
    "code_path",
}


@dataclass(frozen=True)
class DomainToolAdapter:
    tool: Any
    risky: bool

    @property
    def name(self) -> str:
        return self.tool.name

    def scicore_kernel_spec(self, context) -> dict[str, Any]:
        return {
            "schema": _compact_schema(self.tool.parameters),
            "risky": self.risky,
            "description": self.tool.description,
            "run": lambda args: self.execute(context, args),
        }

    def validate(self, context, args: dict[str, Any]) -> None:
        schema = self.tool.parameters or {}
        required = schema.get("required", [])
        missing = [name for name in required if name not in args or args[name] in (None, "")]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")

        properties = schema.get("properties", {})
        unknown = sorted(set(args) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")

        if self.name == "ssh_bash":
            return
        for key, value in args.items():
            if key == "input_files" and isinstance(value, list):
                for item in value:
                    context.path(str(item))
            elif key in PATH_KEYS and value:
                context.path(str(value))

    def execute(self, context, args: dict[str, Any]) -> str:
        normalized = dict(args)
        if self.name != "ssh_bash":
            for key, value in list(normalized.items()):
                if key == "input_files" and isinstance(value, list):
                    normalized[key] = [str(context.path(str(item))) for item in value]
                elif key in PATH_KEYS and value:
                    normalized[key] = str(context.path(str(value)))
        result = self.tool.execute(**normalized)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@dataclass(frozen=True)
class MCPToolAdapter:
    provider: Any
    raw_schema: dict[str, Any]
    risky: bool = True

    @property
    def name(self) -> str:
        return str(
            self.raw_schema.get("name")
            or self.raw_schema.get("function", {}).get("name")
            or ""
        )

    @property
    def parameters(self) -> dict[str, Any]:
        if self.raw_schema.get("type") == "function":
            return self.raw_schema.get("function", {}).get("parameters", {})
        return self.raw_schema.get("inputSchema", {})

    def scicore_kernel_spec(self, context) -> dict[str, Any]:
        description = (
            self.raw_schema.get("description")
            or self.raw_schema.get("function", {}).get("description", "")
        )
        return {
            "schema": _compact_schema(self.parameters),
            "risky": True,
            "description": f"External MCP tool. {description}".strip(),
            "run": lambda args: self.execute(context, args),
        }

    def validate(self, context, args: dict[str, Any]) -> None:
        required = self.parameters.get("required", [])
        missing = [name for name in required if args.get(name) in (None, "")]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")
        _validate_nested_paths(context, args)

    def execute(self, context, args: dict[str, Any]) -> str:
        result = self.provider.call_tool(self.name, args)
        if result is None:
            raise RuntimeError(f"MCP provider no longer serves tool: {self.name}")
        return str(result)


def build_domain_adapters() -> dict[str, DomainToolAdapter | MCPToolAdapter]:
    adapters = {}
    for tool in ALL_TOOLS:
        if tool.name in EXCLUDED_LEGACY_TOOLS:
            continue
        risky = tool.name.startswith(RISKY_PREFIXES)
        adapters[tool.name] = DomainToolAdapter(tool=tool, risky=risky)
    provider = get_mcp_provider()
    for schema in provider.tool_schemas():
        adapter = MCPToolAdapter(provider=provider, raw_schema=schema)
        if not adapter.name or adapter.name in adapters:
            continue
        adapters[adapter.name] = adapter
    return adapters


def _compact_schema(parameters: dict[str, Any]) -> dict[str, str]:
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    rendered = {}
    for name, spec in properties.items():
        type_name = spec.get("type", "any")
        suffix = "" if name in required else "=optional"
        rendered[name] = f"{type_name}{suffix}"
    return rendered


def ensure_workspace_state(root: str | Path) -> Path:
    state = Path(root).resolve() / ".scicore"
    state.mkdir(parents=True, exist_ok=True)
    return state


def _validate_nested_paths(context, value: Any, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _validate_nested_paths(context, child_value, str(child_key))
        return
    if isinstance(value, list):
        for child in value:
            _validate_nested_paths(context, child, key)
        return
    if not isinstance(value, str) or not value:
        return
    lowered = key.lower()
    if lowered in PATH_KEYS or lowered.endswith(("_path", "_file", "_dir")):
        context.path(value)
