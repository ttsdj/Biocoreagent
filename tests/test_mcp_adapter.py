from pathlib import Path

from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext

import scicore.agent.mcp_client as mcp_client
from scicore.bio.runtime import BioRuntime


class FakeMCPProvider:
    def tool_schemas(self):
        return [
            {
                "name": "external_lookup",
                "description": "Look up an external record.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

    def call_tool(self, name, arguments):
        if name == "external_lookup":
            return f"record:{arguments['query']}"
        return None


def test_external_mcp_tool_uses_scicore_kernel_approval_and_trace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MCP_PROVIDER", FakeMCPProvider())
    agent = BioRuntime(
        model_client=FakeModelClient(
            [
                '<tool>{"name":"external_lookup","args":{"query":"TP53"}}</tool>',
                "<final>External evidence inspected.</final>",
            ]
        ),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".scicore" / "sessions"),
        run_store=RunStore(tmp_path / ".scicore" / "runs"),
        approval_policy="auto",
        role="executor",
        allow_orchestration=False,
    )

    assert agent.tools["external_lookup"]["risky"] is True
    assert agent.ask("Find TP53") == "External evidence inspected."
    tool_event = next(item for item in agent.session["history"] if item["role"] == "tool")
    assert tool_event["content"] == "record:TP53"
