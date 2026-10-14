"""Versioned workflow intermediate representation for governed analyses."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> float:
    return time.time()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskEnvelope(StrictModel):
    task_id: str
    user_request: str
    input_paths: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)


class RouteDecision(StrictModel):
    analysis_type: str
    intent: str
    risk_level: Literal["low", "medium", "high"]
    requires_plan: bool
    preferred_backend: str
    fallback_allowed: bool


class ApprovedPlan(StrictModel):
    plan_id: str
    status: Literal["planned", "approved", "rejected", "not_required"]
    steps: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    approved_by: str = ""
    approved_at: float | None = None


class CapabilityInvocation(StrictModel):
    invocation_id: str
    capability_name: str
    backend: str
    parameters: dict[str, Any]
    expected_artifacts: list[str] = Field(default_factory=list)
    deterministic: bool = True
    risk_level: Literal["low", "medium", "high"] = "medium"
    approval_required: bool = False
    created_at: float = Field(default_factory=_now)


class EvidenceArtifact(StrictModel):
    path: str
    kind: str = "analysis_artifact"
    exists: bool
    size_bytes: int = 0
    sha256: str = ""


class ExecutionResult(StrictModel):
    invocation_id: str
    status: Literal["completed", "failed", "blocked", "degraded"]
    backend: str
    exit_code: int | None = None
    artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    stdout_path: str = ""
    stderr_path: str = ""
    error_type: str = ""
    blocker: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    completed_at: float = Field(default_factory=_now)


class VerificationCheck(StrictModel):
    name: str
    status: Literal["pass", "fail", "warning"]
    message: str = ""


class VerificationResult(StrictModel):
    status: Literal["passed", "failed", "blocked"]
    checks: list[VerificationCheck] = Field(default_factory=list)
    verified_artifacts: list[str] = Field(default_factory=list)
    verified_at: float = Field(default_factory=_now)


class FinalSynthesis(StrictModel):
    status: Literal["completed", "failed", "blocked", "degraded"]
    summary: str
    unresolved_blockers: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)


class WorkflowManifest(StrictModel):
    schema_version: Literal["scicore.bio.workflow.v1"] = "scicore.bio.workflow.v1"
    workflow_id: str
    task: TaskEnvelope
    route: RouteDecision
    plan: ApprovedPlan
    invocations: list[CapabilityInvocation] = Field(default_factory=list)
    executions: list[ExecutionResult] = Field(default_factory=list)
    verification: VerificationResult | None = None
    final: FinalSynthesis | None = None
    updated_at: float = Field(default_factory=_now)

    @model_validator(mode="after")
    def completed_requires_verified_evidence(self):
        if self.final is None or self.final.status != "completed":
            return self
        if self.verification is None or self.verification.status != "passed":
            raise ValueError("completed workflow requires passed verification")
        if not any(item.status == "completed" for item in self.executions):
            raise ValueError("completed workflow requires a completed execution")
        if not self.final.evidence_paths:
            raise ValueError("completed workflow requires evidence paths")
        return self


class WorkflowManifestStore:
    """Atomic JSON persistence for workflow manifests."""

    def __init__(self, workspace_root: str | Path):
        self.root = Path(workspace_root).resolve() / ".scicore" / "workflows"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, user_request: str, route: Any) -> WorkflowManifest:
        route_data = route.to_dict() if hasattr(route, "to_dict") else dict(route)
        input_path = str(route_data.pop("input_path", "") or "")
        route_data.pop("target_group", None)
        workflow_id = "workflow_" + uuid.uuid4().hex[:12]
        manifest = WorkflowManifest(
            workflow_id=workflow_id,
            task=TaskEnvelope(
                task_id="task_" + uuid.uuid4().hex[:12],
                user_request=str(user_request),
                input_paths=[str(Path(input_path).expanduser())] if input_path else [],
            ),
            route=RouteDecision(**route_data),
            plan=ApprovedPlan(
                plan_id="plan_" + uuid.uuid4().hex[:12],
                status="planned" if route_data["requires_plan"] else "not_required",
            ),
        )
        self.save(manifest)
        return manifest

    def save(self, manifest: WorkflowManifest) -> Path:
        manifest.updated_at = _now()
        validated = WorkflowManifest.model_validate(manifest.model_dump())
        directory = self.root / validated.workflow_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(path)
        return path

    def load(self, workflow_id: str) -> WorkflowManifest | None:
        path = self.root / str(workflow_id) / "manifest.json"
        if not path.is_file():
            return None
        return WorkflowManifest.model_validate_json(path.read_text(encoding="utf-8"))


def evidence_artifact(path: str | Path, kind: str = "analysis_artifact") -> EvidenceArtifact:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return EvidenceArtifact(path=str(resolved), kind=kind, exists=False)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return EvidenceArtifact(
        path=str(resolved),
        kind=kind,
        exists=True,
        size_bytes=resolved.stat().st_size,
        sha256=digest.hexdigest(),
    )


def manifest_as_dict(manifest: WorkflowManifest) -> dict[str, Any]:
    return json.loads(manifest.model_dump_json())
