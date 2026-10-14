"""Dataclasses for the SciCoreAgent evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RequiredFile:
    path: str
    must_exist: bool = True
    readonly: bool = True


@dataclass
class ExpectedFile:
    path: str
    must_exist: bool = True


@dataclass
class Verification:
    commands: list[str] = field(default_factory=list)
    expected_files: list[ExpectedFile] = field(default_factory=list)


@dataclass
class WorkspaceSpec:
    path: str


@dataclass
class GradingSpec:
    type: str = "command_exit_code_and_diff"


@dataclass
class EvalTask:
    id: str
    title: str
    description: str
    workspace: WorkspaceSpec
    prompt: str
    input_contract: list[RequiredFile] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_patterns: list[str] = field(default_factory=list)
    verification: Verification = field(default_factory=Verification)
    grading: GradingSpec = field(default_factory=GradingSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "EvalTask":
        workspace_data = data.get("workspace") or {}
        workspace_path = workspace_data.get("path", ".")
        if base_dir and not Path(workspace_path).is_absolute():
            workspace_path = str((base_dir / workspace_path).resolve())

        required_files = [
            RequiredFile(**item)
            for item in (data.get("input_contract") or {}).get("required_files", [])
        ]
        verification_data = data.get("verification") or {}
        verification = Verification(
            commands=list(verification_data.get("commands", [])),
            expected_files=[
                ExpectedFile(**item)
                for item in verification_data.get("expected_files", [])
            ],
        )
        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            description=data.get("description", ""),
            workspace=WorkspaceSpec(path=workspace_path),
            input_contract=required_files,
            prompt=data.get("prompt", ""),
            allowed_tools=list(data.get("allowed_tools", [])),
            disallowed_patterns=list(data.get("disallowed_patterns", [])),
            verification=verification,
            grading=GradingSpec(**(data.get("grading") or {})),
        )
