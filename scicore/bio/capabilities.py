"""Typed deterministic capability contracts used by governed workflows."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .workflow_ir import (
    CapabilityInvocation,
    EvidenceArtifact,
    ExecutionResult,
    VerificationCheck,
    VerificationResult,
    evidence_artifact,
)


Executor = Callable[[Any, dict[str, Any]], dict[str, Any]]
Verifier = Callable[[dict[str, Any]], VerificationResult]


@dataclass(frozen=True)
class CapabilityContract:
    name: str
    analysis_type: str
    backend: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    executor: Executor
    verifier: Verifier
    fallback: str = ""
    risk_level: str = "medium"
    approval_policy: str = "plan"
    expected_artifact_keys: tuple[str, ...] = ()
    deterministic: bool = True

    def validate_input(self, parameters: dict[str, Any]) -> None:
        _validate_object(parameters, self.input_schema, "capability input")

    def validate_output(self, payload: dict[str, Any]) -> None:
        _validate_object(payload, self.output_schema, "capability output")


class CapabilityRegistry:
    def __init__(self):
        self._contracts: dict[str, CapabilityContract] = {}

    def register(self, contract: CapabilityContract) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"capability already registered: {contract.name}")
        self._contracts[contract.name] = contract

    def get(self, name: str) -> CapabilityContract:
        try:
            return self._contracts[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {name}") from exc

    def list(self, analysis_type: str = "") -> list[CapabilityContract]:
        values = list(self._contracts.values())
        if analysis_type:
            values = [item for item in values if item.analysis_type == analysis_type]
        return values

    def invoke(self, runtime: Any, name: str, parameters: dict[str, Any]):
        contract = self.get(name)
        contract.validate_input(parameters)
        invocation = CapabilityInvocation(
            invocation_id="invocation_" + uuid.uuid4().hex[:12],
            capability_name=contract.name,
            backend=contract.backend,
            parameters=parameters,
            expected_artifacts=list(contract.expected_artifact_keys),
            deterministic=contract.deterministic,
            risk_level=contract.risk_level,
            approval_required=contract.approval_policy == "explicit",
        )
        try:
            payload = contract.executor(runtime, dict(parameters))
            if not isinstance(payload, dict):
                raise TypeError("capability executor must return a mapping")
            contract.validate_output(payload)
            verification = contract.verifier(payload)
            artifacts = _artifacts_from_payload(payload, contract.expected_artifact_keys)
            execution_status = _execution_status(payload, verification)
            execution = ExecutionResult(
                invocation_id=invocation.invocation_id,
                status=execution_status,
                backend=contract.backend,
                exit_code=_optional_int(payload.get("returncode", payload.get("exit_code"))),
                artifacts=artifacts,
                stdout_path=str(payload.get("stdout_path", "")),
                stderr_path=str(payload.get("stderr_path", "")),
                error_type=str(payload.get("error_type", "")),
                blocker=str(payload.get("blocker", payload.get("error", ""))),
                metrics=dict(payload.get("summary", {}) or {}),
            )
        except Exception as exc:
            payload = {
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "blocker": str(exc),
            }
            verification = VerificationResult(
                status="failed",
                checks=[VerificationCheck(name="capability_exception", status="fail", message=str(exc))],
            )
            execution = ExecutionResult(
                invocation_id=invocation.invocation_id,
                status="failed",
                backend=contract.backend,
                error_type=exc.__class__.__name__,
                blocker=str(exc),
            )
        return invocation, execution, verification, payload


def _tool_executor(tool_name: str) -> Executor:
    def execute(runtime: Any, parameters: dict[str, Any]) -> dict[str, Any]:
        result = runtime.execute_tool(tool_name, parameters)
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, dict):
            return content
        try:
            parsed = json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{tool_name} returned non-JSON output") from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"{tool_name} returned JSON that is not an object")
        return parsed

    return execute


def _artifact_verifier(required_keys: tuple[str, ...]) -> Verifier:
    def verify(payload: dict[str, Any]) -> VerificationResult:
        checks = []
        verified = []
        backend_status = str(payload.get("status", "")).lower()
        checks.append(
            VerificationCheck(
                name="backend_status",
                status="pass" if backend_status == "completed" else "fail",
                message=backend_status or "missing status",
            )
        )
        for key in required_keys:
            raw_path = payload.get(key)
            artifact = evidence_artifact(raw_path) if raw_path else None
            exists = bool(artifact and artifact.exists)
            checks.append(
                VerificationCheck(
                    name=f"artifact:{key}",
                    status="pass" if exists else "fail",
                    message=str(raw_path or "missing path"),
                )
            )
            if exists:
                verified.append(artifact.path)
        return VerificationResult(
            status="passed" if checks and all(item.status == "pass" for item in checks) else "failed",
            checks=checks,
            verified_artifacts=verified,
        )

    return verify


def default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    common_input = {
        "type": "object",
        "required": ["count_matrix_path", "target_group", "output_dir"],
        "properties": {
            "count_matrix_path": {"type": "string"},
            "target_group": {"type": "string"},
            "output_dir": {"type": "string"},
            "method": {"type": "string"},
            "alpha": {"type": "number"},
            "n_cpus": {"type": "integer"},
        },
    }
    output = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
        "allow_additional": True,
    }
    registry.register(
        CapabilityContract(
            name="omicverse_bulk_deg",
            analysis_type="bulk_rnaseq",
            backend="omicverse",
            input_schema=common_input,
            output_schema=output,
            executor=_tool_executor("transcriptome_omicverse_deg"),
            verifier=_artifact_verifier(
                ("full_results_path", "significant_results_path", "summary_path")
            ),
            fallback="deseq2_bulk_deg",
            expected_artifact_keys=(
                "full_results_path",
                "significant_results_path",
                "summary_path",
            ),
        )
    )
    registry.register(
        CapabilityContract(
            name="deseq2_bulk_deg",
            analysis_type="bulk_rnaseq",
            backend="deseq2",
            input_schema={
                "type": "object",
                "required": ["count_matrix_path", "target_tissue", "output_dir"],
                "properties": {
                    "count_matrix_path": {"type": "string"},
                    "target_tissue": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "padj_threshold": {"type": "number"},
                    "timeout": {"type": "integer"},
                },
            },
            output_schema=output,
            executor=_tool_executor("bio_deseq2_tissue_vs_rest"),
            verifier=_artifact_verifier(
                ("full_results_path", "significant_results_path", "summary_path")
            ),
            expected_artifact_keys=(
                "metadata_path",
                "script_path",
                "full_results_path",
                "significant_results_path",
                "summary_path",
            ),
        )
    )
    return registry


def _validate_object(value: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    required = set(schema.get("required", []))
    missing = sorted(key for key in required if key not in value or value[key] in (None, ""))
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")
    properties = schema.get("properties", {})
    if not schema.get("allow_additional", False):
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
    expected_types = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, field_schema in properties.items():
        if key not in value:
            continue
        expected = expected_types.get(field_schema.get("type"))
        if expected is not None and not isinstance(value[key], expected):
            raise TypeError(f"{label}.{key} must be {field_schema.get('type')}")


def _artifacts_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> list[EvidenceArtifact]:
    artifacts = []
    for key in keys:
        raw_path = payload.get(key)
        if raw_path:
            artifacts.append(evidence_artifact(str(raw_path), kind=key))
    return artifacts


def _execution_status(payload: dict[str, Any], verification: VerificationResult) -> str:
    raw = str(payload.get("status", "")).lower()
    if raw == "completed" and verification.status == "passed":
        return "completed"
    if raw in {"blocked", "backend_missing", "script_written", "script_written"}:
        return "blocked"
    if raw == "degraded":
        return "degraded"
    return "failed"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
