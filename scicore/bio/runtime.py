"""kernel runtime extended with SciCoreAgent domain and orchestration tools."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from pathlib import Path

from scicore.kernel import tools as scicore_kernel_tools
from scicore.kernel.prompt_prefix import PromptPrefix
from scicore.kernel.runtime import kernel
from scicore.kernel.task_state import TaskState
from scicore.kernel.workspace import clip, now

from .analysis_router import route_analysis_task
from .analysis_verifier import verify_final_answer
from .audit import AuditTrail
from .capabilities import default_capability_registry
from .domain import build_domain_adapters
from .evidence import EvidenceManager
from .fallback_runner import run_fallback_analysis
from .result_exporter import export_result_table_to_csv, is_csv_export_request
from .workflow_ir import (
    CapabilityInvocation,
    EvidenceArtifact,
    ExecutionResult,
    FinalSynthesis,
    VerificationCheck,
    VerificationResult,
    WorkflowManifest,
    WorkflowManifestStore,
    evidence_artifact,
)


DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;\"'<>]+")
PMID_PATTERN = re.compile(r"\bPMID[:\s]*([0-9]{5,12})\b", re.I)
REMEMBER_PATTERN = re.compile(r"(记住|记下来|保存这个|沉淀|写入wiki|存到wiki|remember this|save this|note this)", re.I)
SKILL_PATTERN = re.compile(r"(保存为skill|沉淀为skill|沉淀成skill|以后复用|复用流程|reusable skill|save.*skill)", re.I)


ROLE_TOOL_POLICIES = {
    "explorer": {
        "list_files", "read_file", "search", "bio_seq_inspect",
        "bio_sample_sheet_inspect", "bio_count_matrix_inspect", "file_hash",
        "skill_list", "skill_read", "skill_search", "wiki_search", "wiki_list",
        "kb_cross_search", "bio_query_evidence", "pubmed_search",
        "pubmed_fetch_details", "literature_red_blue_review",
        "code_literature_link_search", "code_literature_link_list",
        "workflow_preflight_check", "workflow_primitive_ledger",
        "transcriptome_capability_list", "transcriptome_state_inspect",
        "transcriptome_plan", "transcriptome_plan_verify", "transcriptome_omicverse_check",
    },
    "planner": {
        "list_files", "read_file", "search", "bio_workflow_sketch",
        "bio_pipeline_plan", "bio_pipeline_supported", "bio_experiment_plan",
        "bio_replication_plan", "skill_list", "skill_read", "skill_search",
        "wiki_search", "kb_cross_search", "pubmed_search",
        "pubmed_fetch_details", "literature_red_blue_review",
        "code_literature_link_search", "code_literature_link_list",
        "workflow_preflight_check", "workflow_primitive_ledger", "workflow_plan_prepare",
        "transcriptome_capability_list", "transcriptome_state_inspect",
        "transcriptome_plan", "transcriptome_plan_verify", "transcriptome_omicverse_check",
    },
    "executor": {"*"},
    "verifier": {
        "list_files", "read_file", "search", "run_shell", "file_hash",
        "bio_seq_inspect", "bio_sample_sheet_inspect",
        "bio_count_matrix_inspect", "bio_rnaseq_compare",
        "bio_contingency_test", "bio_regression",
        "bio_deseq2_tissue_vs_rest",
        "workflow_preflight_check", "workflow_primitive_ledger",
        "transcriptome_capability_list", "transcriptome_state_inspect",
        "transcriptome_plan", "transcriptome_plan_verify", "transcriptome_omicverse_check",
    },
    "bio_worker": {
        "list_files", "read_file", "search", "run_shell", "ssh_bash",
        "bio_seq_inspect", "bio_sample_sheet_inspect",
        "bio_count_matrix_inspect", "bio_rnaseq_compare", "bio_report",
        "bio_pipeline_plan", "bio_experiment_plan", "bio_rds_inspect",
        "bio_r_bridge", "bio_contingency_test", "bio_regression",
        "bio_deseq2_quick", "bio_deseq2_tissue_vs_rest", "file_hash", "skill_search", "skill_read",
        "wiki_search", "kb_cross_search", "bio_query_evidence",
        "pubmed_search", "pubmed_fetch_details", "pubmed_literature_review",
        "literature_red_blue_review", "literature_export_xlsx",
        "code_literature_link_save", "code_literature_link_search",
        "code_literature_link_list",
        "workflow_preflight_check", "workflow_primitive_ledger", "workflow_plan_prepare",
        "transcriptome_capability_list", "transcriptome_state_inspect",
        "transcriptome_plan", "transcriptome_plan_verify",
        "transcriptome_provenance_append", "transcriptome_omicverse_check",
        "transcriptome_omicverse_deg",
    },
}


class BioRuntime(kernel):
    def __init__(
        self,
        *args,
        role: str = "executor",
        orchestrator=None,
        allow_orchestration: bool = True,
        **kwargs,
    ):
        if role not in ROLE_TOOL_POLICIES:
            raise ValueError(f"unknown agent role: {role}")
        self.role = role
        self.orchestrator = orchestrator
        self.allow_orchestration = allow_orchestration
        self._domain_adapters = build_domain_adapters()
        self.last_auto_sedimentations = []
        self._dream_manager = None
        super().__init__(*args, **kwargs)
        self.audit_trail = AuditTrail(
            self.root,
            str(self.session.get("id", "")),
            actor=f"agent:{self.role}",
        )
        self.evidence_manager = EvidenceManager(
            self.root,
            self.run_store.root.parent,
            redactor=self.redact_text,
        )
        self._pending_evidence_tool = None
        self._last_direct_artifacts: list[str] = []
        self.last_direct_task_state: TaskState | None = None
        self.audit_trail.append(
            "message",
            {
                "role": "system",
                "content_preview": "session started",
                "workspace_root": str(self.root),
                "role_name": self.role,
                "approval_policy": self.approval_policy,
                "max_steps": self.max_steps,
                "tool_count": len(self.tools),
            },
        )
        self.capability_registry = default_capability_registry()
        self.workflow_store = WorkflowManifestStore(self.root)
        self.current_workflow_manifest: WorkflowManifest | None = None
        self.current_workflow_path: Path | None = None

    def _apply_tool_allowlist(self, tools):
        """Validate allowlists against the fused registry, including domain tools."""
        if self.allowed_tools is None:
            return tools
        unknown = [name for name in self.allowed_tools if name not in tools]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {name: tool for name, tool in tools.items() if name in allowed}

    def record(self, item):
        super().record(item)
        role = str(item.get("role", ""))
        if role in {"user", "assistant", "tool"}:
            task_state = getattr(self, "current_task_state", None)
            self.audit_trail.log_message(
                role,
                item.get("content", ""),
                run_id=getattr(task_state, "run_id", "") or "",
                task_id=getattr(task_state, "task_id", "") or "",
            )

    def consume_agent_completions(self) -> str:
        """Persist and inject newly completed child-agent results once."""
        if self.orchestrator is None or not self.allow_orchestration:
            return ""
        state = self.session.setdefault("orchestrator", {})
        cursor = int(state.get("completion_cursor", 0))
        updates = self.orchestrator.completion_updates(
            str(self.session.get("id", "")),
            after_sequence=cursor,
        )
        if not updates:
            return ""
        lines = [
            "Supervisor completion inbox (new durable child-agent results):",
        ]
        for item in updates:
            status = item.get("status", "unknown")
            if item.get("kind") == "agent_team_completed":
                lines.append(
                    f"- team {item.get('team_id')} [{status}], "
                    f"degradation={item.get('degradation_level', 0)}, "
                    f"verified={item.get('verification_passed')}"
                )
                lines.append(str(item.get("synthesis", ""))[:6000])
            else:
                lines.append(
                    f"- job {item.get('job_id')} {item.get('name')} "
                    f"({item.get('role')}) [{status}]"
                )
                lines.append(
                    str(item.get("result") or item.get("error") or "(no result)")[:3000]
                )
            artifacts = item.get("artifact_paths", [])
            if artifacts:
                lines.append("  artifacts: " + ", ".join(str(path) for path in artifacts[:12]))
        state["completion_cursor"] = max(int(item["sequence"]) for item in updates)
        content = "\n".join(lines)
        self.record(
            {
                "role": "system",
                "content": content,
                "created_at": now(),
                "metadata": {
                    "kind": "orchestrator_completion_reinjection",
                    "completion_ids": [item.get("completion_id", "") for item in updates],
                },
            }
        )
        self.audit_trail.append(
            "message",
            {
                "role": "supervisor",
                "content_preview": content[:1200],
                "kind": "orchestrator_completion_reinjection",
                "completion_ids": [item.get("completion_id", "") for item in updates],
            },
        )
        return content

    def emit_trace(self, task_state, event, payload=None):
        emitted = super().emit_trace(task_state, event, payload)
        if event == "run_started":
            self.evidence_manager.begin_run(
                str(getattr(task_state, "run_id", "") or ""),
                self.run_store.run_dir(task_state),
                str(emitted.get("event_id", "")),
            )
        else:
            self.evidence_manager.note_event(emitted)
        if event == "tool_executed" and self._pending_evidence_tool is not None:
            pending = self._pending_evidence_tool
            self._pending_evidence_tool = None
            bundles = self.evidence_manager.record_tool_completion(
                name=pending["name"],
                args=pending["args"],
                result=pending["result"].content,
                metadata=pending["result"].metadata,
                event_id=str(emitted.get("event_id", "")),
            )
            for bundle in bundles:
                self.emit_trace(
                    task_state,
                    "artifact_created",
                    {
                        "artifact_path": bundle["artifact_path"],
                        "deliverable_bundle": bundle["bundle_path"],
                        "source_tool_event": emitted.get("event_id", ""),
                    },
                )
        if event == "run_finished":
            self.evidence_manager.finish_run(str(emitted.get("event_id", "")))
        self.audit_trail.log_trace_event(
            event,
            emitted,
            run_id=getattr(task_state, "run_id", "") or "",
            task_id=getattr(task_state, "task_id", "") or "",
        )
        if event == "run_finished":
            # The kernel loop writes report.json immediately after this trace
            # event.  BioRuntime.ask finalizes the audit after super().ask()
            # returns, so all run artifacts are present before hashing.
            pass
        return emitted

    def execute_tool(self, name, args):
        if getattr(self, "current_task_state", None) is not None:
            if name == "run_shell":
                self.evidence_manager.capture_code_tree(
                    version_role="pre_shell",
                    captured_at_event=self.evidence_manager.last_event_id,
                )
            else:
                self.evidence_manager.capture_paths(
                    self._tool_path_arguments(args or {}),
                    version_role="pre_tool",
                    captured_at_event=self.evidence_manager.last_event_id,
                )
        result = super().execute_tool(name, args)
        if getattr(self, "current_task_state", None) is not None:
            self._pending_evidence_tool = {
                "name": name,
                "args": dict(args or {}),
                "result": result,
            }
        if getattr(self, "current_task_state", None) is None:
            self.audit_trail.log_tool_call(
                name=name,
                args=args or {},
                result=result.content,
                metadata=result.metadata,
            )
        return result

    @staticmethod
    def _tool_path_arguments(args: dict) -> list[str]:
        values = []
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            if re.search(r"(?i)(path|file|script|input|output)", str(key)):
                values.append(value)
        return values

    def approve(self, name, args):
        task_state = getattr(self, "current_task_state", None)
        if task_state is not None:
            self.emit_trace(
                task_state,
                "approval_requested",
                {"action": name, "args": args or {}, "risk_level": "high"},
            )
        decision = super().approve(name, args)
        if task_state is not None:
            self.emit_trace(
                task_state,
                "approval_decided",
                {
                    "action": name,
                    "decision": "approved" if decision else "denied",
                    "approval_policy": self.approval_policy,
                },
            )
        self.audit_trail.log_approval(
            action=name,
            args=args or {},
            decision="approved" if decision else "denied",
            reason=f"approval_policy={self.approval_policy}",
            risk_level="high",
            run_id=getattr(task_state, "run_id", "") or "",
            task_id=getattr(task_state, "task_id", "") or "",
        )
        return decision

    def build_tools(self):
        tools = scicore_kernel_tools.build_tool_registry(self.tool_context())
        tools.pop("delegate", None)
        for name, adapter in self._domain_adapters.items():
            tools[name] = adapter.scicore_kernel_spec(self.tool_context())
        if self.orchestrator is not None and self.allow_orchestration:
            tools.update(self.orchestrator.tool_registry(self))

        policy = ROLE_TOOL_POLICIES[self.role]
        if "*" not in policy:
            tools = {name: spec for name, spec in tools.items() if name in policy}
        return tools

    def _apply_tool_allowlist(self, tools):
        if self.allowed_tools is None:
            return tools
        unknown = sorted(set(self.allowed_tools) - set(tools))
        if unknown:
            raise ValueError(f"unknown or role-forbidden tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {name: tool for name, tool in tools.items() if name in allowed}

    def validate_tool(self, name, args):
        if name in scicore_kernel_tools.BASE_TOOL_SPECS:
            return scicore_kernel_tools.validate_tool(self.tool_context(), name, args)
        if name in self._domain_adapters:
            return self._domain_adapters[name].validate(self.tool_context(), args or {})
        if (
            self.orchestrator is not None
            and self.allow_orchestration
            and name in self.orchestrator.tool_names
        ):
            return self.orchestrator.validate_tool(name, args or {})
        raise ValueError(f"unknown tool: {name}")

    def build_prefix(self) -> PromptPrefix:
        prefix = super().build_prefix()
        identity = (
            "You are SciCoreAgent, an auditable multi-agent research and coding "
            "harness. You are built on the kernel execution kernel, but your user-facing "
            "identity is SciCoreAgent.\n\n"
            "Core responsibilities:\n"
            "- Help users inspect, modify, test, and package local projects with a safe tool harness.\n"
            "- Orchestrate role-based sub-agents for exploration, planning, execution, and verification.\n"
            "- Manage local skills, MCP extensions, durable project knowledge, and domain tools.\n"
            "- Support rigorous scientific and bioinformatics workflows when those domain tools are relevant.\n"
            "- Explain uncertainty, missing data, tool limits, and verification status explicitly.\n"
        )
        text = prefix.text.replace(
            "You are scicore.kernel, a small local coding agent working inside a local repository.",
            identity,
            1,
        )
        bio_rules = (
            "\n\nSciCoreAgent domain contract:\n"
            f"- Active role: {self.role}.\n"
            "- Treat biological input data as immutable evidence.\n"
            "- Distinguish observed facts, computed results, assumptions, and missing information.\n"
            "- Record reference genome/build, sample groups, software parameters, and output provenance.\n"
            "- Never claim a biological analysis succeeded without inspecting its result artifact.\n"
            "- Multi-agent tasks are centrally scheduled; use team tools for genuinely independent work.\n"
            "- For literature reviews, evidence synthesis, mechanism claims, or research strategy, use the literature_red_blue_review tool when evidence rows are available. Red Agent attacks factuality, logical consistency, and citation quality; Blue Agent applies ADD/DELETE/MODIFY/VERIFY repairs; inspect score convergence, oscillation detection, and JSON fallback status before presenting a calibrated conclusion.\n"
            "- When writing or modifying scientific code based on papers, persist a code-literature provenance link with code_literature_link_save so future turns can find the code by DOI/PMID/title/purpose and modify it with the original evidence context.\n"
            "- For literature-review deliverables, prefer literature_export_xlsx over ad-hoc spreadsheet scripts.\n"
            "- For complex research tasks, use OmicOS-style governance: run workflow_preflight_check, inspect workflow_primitive_ledger, compare quick/standard/strict options, and emit workflow_plan_prepare artifacts before high-risk execution.\n"
            "- The CLI/runtime supports streaming final-answer text through provider SSE when the backend supports streaming. Do not claim streaming is impossible; tool-call markup is withheld from the user until parsed, while <final> content may stream to the terminal.\n"
            "- If tools fail repeatedly or no artifacts appear after substantial progress, stop blind retries and enter recovery: summarize failed tool calls, missing environment/data, available partial artifacts, and the smallest next action.\n"
        )
        text = text + bio_rules
        import hashlib

        return replace(prefix, text=text, hash=hashlib.sha256(text.encode("utf-8")).hexdigest())

    def ask(self, user_message, stream_callback=None):
        # A new request must not attach shortcut/tool evidence to a prior run.
        self.current_task_state = None
        self.current_run_dir = None
        shortcut_user_logged = False
        export_result = self._try_csv_export_shortcut(user_message)
        if export_result is not None:
            export_completed = export_result.startswith("CSV 导出完成")
            self._record_direct_delivery(
                user_message,
                export_result,
                kind="csv_export_shortcut",
                completed=export_completed,
            )
            self.audit_trail.log_message("user", user_message)
            shortcut_user_logged = True
            self.audit_trail.log_message("assistant", export_result)
            self.audit_trail.finalize(
                final_status="completed" if export_completed else "stopped",
                final_answer=export_result,
            )
            self.last_auto_sedimentations = []
            return export_result
        route = route_analysis_task(user_message)
        self.last_analysis_route = route.to_dict()
        if route.analysis_type != "coding" and route.intent == "run_analysis":
            self.current_workflow_manifest = self.workflow_store.create(user_message, route)
            self.current_workflow_path = self.workflow_store.save(self.current_workflow_manifest)
        else:
            self.current_workflow_manifest = None
            self.current_workflow_path = None
        self.audit_trail.append(
            "message",
            {
                "role": "system",
                "content_preview": "analysis route selected",
                "route": self.last_analysis_route,
            },
        )
        if route.analysis_type == "bulk_rnaseq" and route.intent == "run_analysis":
            shortcut = self._try_bio_shortcut(user_message, force=True)
            if shortcut is not None:
                shortcut = verify_final_answer(shortcut)
                self._finalize_current_workflow(shortcut)
                workflow_status = self._current_workflow_status("completed")
                self._record_direct_delivery(
                    user_message,
                    shortcut,
                    kind="bulk_rnaseq_shortcut",
                    completed=workflow_status == "completed",
                )
                if not shortcut_user_logged:
                    self.audit_trail.log_message("user", user_message)
                self.audit_trail.log_message("assistant", shortcut)
                self.audit_trail.finalize(
                    final_status=self._current_workflow_status("completed"),
                    final_answer=shortcut,
                )
                self.last_auto_sedimentations = []
                return shortcut
        if route.analysis_type in {"proteomics", "single_cell", "generic_table"} and route.intent == "run_analysis":
            fallback = self._try_fallback_analysis(user_message, route)
            if fallback is not None:
                fallback = verify_final_answer(fallback)
                self._finalize_current_workflow(fallback)
                workflow_status = self._current_workflow_status("completed")
                self._record_direct_delivery(
                    user_message,
                    fallback,
                    kind=f"{route.analysis_type}_fallback",
                    completed=workflow_status == "completed",
                )
                if not shortcut_user_logged:
                    self.audit_trail.log_message("user", user_message)
                self.audit_trail.log_message("assistant", fallback)
                self.audit_trail.finalize(
                    final_status=self._current_workflow_status("completed"),
                    final_answer=fallback,
                )
                self.last_auto_sedimentations = []
                return fallback
        try:
            final = super().ask(user_message, stream_callback=stream_callback)
            final = verify_final_answer(final)
            self.last_auto_sedimentations = self._auto_sediment(user_message, final)
            task_state = getattr(self, "current_task_state", None)
            if task_state is not None:
                self.run_store.write_report(
                    task_state,
                    self.redact_artifact(self.build_report(task_state)),
                )
                self.evidence_manager.build_evidence_index(self._runtime_identity())
                self.audit_trail.register_run_artifacts(
                    self.run_store.run_dir(task_state),
                    run_id=getattr(task_state, "run_id", "") or "",
                    task_id=getattr(task_state, "task_id", "") or "",
                )
                self.audit_trail.finalize(
                    final_status=getattr(task_state, "status", ""),
                    final_answer=final,
                )
            return final
        except Exception as exc:
            task_state = getattr(self, "current_task_state", None)
            if task_state is not None and getattr(task_state, "status", "") == "running":
                task_state.stop_model_error(str(exc))
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "exception_raised",
                    {
                        "exception_type": exc.__class__.__name__,
                        "message": str(exc),
                        "recovery_status": "unrecovered",
                    },
                )
                self.emit_trace(
                    task_state,
                    "final_delivery",
                    {"final_answer": str(exc), "delivery_status": "failed"},
                )
                self.emit_trace(
                    task_state,
                    "run_finished",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "final_answer": str(exc),
                    },
                )
                self.run_store.write_report(
                    task_state,
                    self.redact_artifact(self.build_report(task_state)),
                )
                self.evidence_manager.build_evidence_index(self._runtime_identity())
                self.audit_trail.register_run_artifacts(
                    self.run_store.run_dir(task_state),
                    run_id=getattr(task_state, "run_id", "") or "",
                    task_id=getattr(task_state, "task_id", "") or "",
                )
            self.audit_trail.log_error(
                category=exc.__class__.__name__,
                source="BioRuntime.ask",
                message=str(exc),
                run_id=getattr(task_state, "run_id", "") or "",
                task_id=getattr(task_state, "task_id", "") or "",
            )
            self.audit_trail.finalize(final_status="error", final_answer=str(exc))
            raise

    def finalize_answer(self, answer):
        return verify_final_answer(str(answer))

    def _runtime_identity(self) -> dict:
        return {
            "agent_role": self.role,
            "approval_policy": self.approval_policy,
            "max_steps": self.max_steps,
            "max_new_tokens": self.max_new_tokens,
            "model_client": self.model_client.__class__.__name__,
            "model": str(getattr(self.model_client, "model", "")),
            "tool_names": sorted(self.tools),
            "tool_signature": str(getattr(self.prefix_state, "tool_signature", "")),
        }

    def _record_direct_delivery(
        self,
        user_message: str,
        answer: str,
        *,
        kind: str,
        completed: bool,
    ) -> None:
        """Give deterministic shortcuts the same evidence lifecycle as AgentLoop."""
        task_state = TaskState.create(
            task_id=self.new_task_id(),
            run_id=self.new_run_id(),
            user_request=user_message,
        )
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {"task_id": task_state.task_id, "user_request": clip(user_message, 300)},
        )
        self.emit_trace(task_state, "user_input_recorded", {"user_input": user_message})
        self.emit_trace(task_state, "shortcut_selected", {"shortcut_kind": kind})

        artifact_paths = list(self._last_direct_artifacts)
        manifest = self.current_workflow_manifest
        if self.current_workflow_path is not None:
            artifact_paths.append(str(self.current_workflow_path))
        if manifest is not None and manifest.verification is not None:
            artifact_paths.extend(manifest.verification.verified_artifacts)
        for value in dict.fromkeys(artifact_paths):
            path = Path(value)
            absolute = path if path.is_absolute() else self.root / path
            if not absolute.is_file():
                continue
            bundle = self.evidence_manager.create_deliverable(
                absolute,
                tool_name=kind,
                tool_args={},
                tool_result=answer,
                tool_metadata={
                    "tool_status": "ok" if completed else "error",
                    "execution_id": kind,
                    "duration_ms": 0,
                },
                event_id=self.evidence_manager.last_event_id,
            )
            if bundle is not None:
                self.emit_trace(
                    task_state,
                    "artifact_created",
                    {
                        "artifact_path": str(absolute),
                        "deliverable_bundle": str(bundle),
                        "source_shortcut": kind,
                    },
                )
        self._last_direct_artifacts = []

        if completed:
            task_state.finish_success(answer)
        else:
            task_state.stop(f"{kind}_stopped", final_answer=answer)
        self.run_store.write_task_state(task_state)
        self.emit_trace(
            task_state,
            "final_delivery",
            {
                "final_answer": answer,
                "delivery_status": "completed" if completed else "stopped",
            },
        )
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": answer,
            },
        )
        self.run_store.write_report(
            task_state,
            self.redact_artifact(self.build_report(task_state)),
        )
        self.evidence_manager.build_evidence_index(self._runtime_identity())
        self.audit_trail.register_run_artifacts(
            self.run_store.run_dir(task_state),
            run_id=task_state.run_id,
            task_id=task_state.task_id,
        )
        self.last_direct_task_state = task_state
        self.current_task_state = None
        self.current_run_dir = None

    def _try_csv_export_shortcut(self, user_message: str) -> str | None:
        if not is_csv_export_request(user_message):
            return None
        self._emit_progress("识别为结果表 CSV 导出任务，使用确定性转换路径，不进入工具循环。")
        result = export_result_table_to_csv(self.root, user_message)
        if result.status != "completed":
            self._last_direct_artifacts = []
            return (
                "CSV 导出未完成：没有进入模型工具循环，已在确定性导出层停止。\n\n"
                f"- 原因: {result.message}\n"
                "- 请明确提供源文件路径，例如：deseq2_el_vs_rest_results.txt 输出成 csv。"
            )
        self._last_direct_artifacts = [result.source_path, result.output_path]
        return (
            "CSV 导出完成。\n\n"
            f"- 源文件: {result.source_path}\n"
            f"- CSV 文件: {result.output_path}\n"
            f"- 数据行数: {result.rows}\n"
            f"- 列数: {result.columns}\n"
            f"- 识别分隔符: {result.delimiter}\n\n"
            "这类格式转换现在会直接走确定性路径，不再反复 read_file 或拼 run_shell。"
        )

    def _try_bio_shortcut(self, user_message: str, force: bool = False) -> str | None:
        request = str(user_message)
        lowered = request.lower()
        if not force and not _is_bulk_rnaseq_request(request):
            return None
        self._emit_progress("识别为 bulk RNA-seq / 差异表达任务，进入固定工作流。")
        path = _extract_count_matrix_path(request)
        target = _extract_target_tissue(request)
        route = route_analysis_task(request)
        if not path:
            path = route.input_path
        if not target:
            target = route.target_group
        if path and not target:
            target = _infer_target_tissue_from_matrix_and_text(path, request)
        if not path:
            self._record_workflow_blocker("count_matrix_missing")
            return (
                "Bulk RNA-seq deterministic workflow requires a count matrix path before execution.\n"
                "Please provide a .txt/.tsv/.csv count matrix and the target group, for example: "
                "C:\\path\\counts.txt, target group el, compare el vs rest."
            )
        if not target:
            self._record_workflow_blocker("target_group_missing")
            return (
                "Bulk RNA-seq deterministic workflow found the count matrix but could not determine the target group.\n"
                f"Count matrix: {path}\n"
                "Please specify the comparison explicitly, for example: el vs rest."
            )
        plan_payload = {}
        transcriptome_plan = {}
        omicverse_payload = {}
        if "transcriptome_plan" in self.tools:
            self._emit_progress("生成轻量能力链计划：检查 count matrix、推断分组、检查 OmicVerse。")
            plan_result = self.execute_tool(
                "transcriptome_plan",
                {
                    "task": request,
                    "count_matrix_path": path,
                    "target_group": target,
                    "mode": "quick",
                },
            )
            try:
                transcriptome_plan = json.loads(plan_result.content)
            except Exception:
                transcriptome_plan = {"raw_result": plan_result.content}
        if "transcriptome_omicverse_check" in self.tools:
            self._emit_progress("检查 OmicVerse 后端是否可用。")
            check_result = self.execute_tool("transcriptome_omicverse_check", {})
            try:
                omicverse_payload = json.loads(check_result.content)
            except Exception:
                omicverse_payload = {"raw_result": check_result.content}
        if "workflow_plan_prepare" in self.tools:
            plan_result = self.execute_tool(
                "workflow_plan_prepare",
                {
                    "task": request,
                    "input_files": [path],
                    "analysis_type": "deseq2",
                    "target_group": target,
                    "mode": "quick",
                    "output_dir": str(Path(path).resolve().parent / ".scicore" / "plans"),
                },
            )
            try:
                plan_payload = json.loads(plan_result.content)
            except Exception:
                plan_payload = {"raw_result": plan_result.content}
        else:
            plan_payload = {
                "steps": list(transcriptome_plan.get("steps", [])),
                "artifacts": {},
                "message": "Execute the deterministic transcriptome capability chain.",
            }
        approved = self._approve_generated_plan(plan_payload)
        self._record_workflow_plan(plan_payload, approved)
        if not approved:
            return _format_plan_rejected_result(
                target=target,
                transcriptome_plan=transcriptome_plan,
                omicverse_payload=omicverse_payload,
                plan_payload=plan_payload,
            )
        if omicverse_payload.get("available") and "transcriptome_omicverse_deg" in self.tools:
            self._emit_progress("OmicVerse 可用，执行 OmicVerse pyDEG 差异分析。")
            omicverse_result = self._invoke_registered_capability(
                "omicverse_bulk_deg",
                {
                    "count_matrix_path": path,
                    "target_group": target,
                    "output_dir": str(Path(path).resolve().parent),
                    "method": "DEseq2",
                    "alpha": 0.05,
                    "n_cpus": 2,
                },
            )
            if omicverse_result.get("status") == "completed":
                return (
                    f"OmicVerse-backed bulk RNA-seq analysis completed for {target}-vs-rest.\n"
                    f"- transcriptome steps: {transcriptome_plan.get('steps', [])}\n"
                    f"- full results: {omicverse_result.get('full_results_path')}\n"
                    f"- significant results: {omicverse_result.get('significant_results_path')}\n"
                    f"- summary: {json.dumps(omicverse_result.get('summary', {}), ensure_ascii=False)}"
                )
        if "bio_deseq2_tissue_vs_rest" not in self.tools:
            self._record_workflow_blocker("no_executable_de_backend")
            return (
                "Bulk RNA-seq deterministic workflow was selected, but no executable DE backend is available.\n"
                f"- transcriptome plan: {json.dumps(transcriptome_plan.get('steps', []), ensure_ascii=False)}\n"
                f"- OmicVerse backend: {json.dumps(omicverse_payload, ensure_ascii=False)}"
            )
        if "workflow_plan_prepare" in self.tools and not plan_payload:
            self._emit_progress("生成正式计划文档 plan.md / plan.json / delegation.json。")
            plan_result = self.execute_tool(
                "workflow_plan_prepare",
                {
                    "task": request,
                    "input_files": [path],
                    "analysis_type": "deseq2",
                    "target_group": target,
                    "mode": "quick",
                    "output_dir": str(Path(path).resolve().parent / ".scicore" / "plans"),
                },
            )
            try:
                plan_payload = json.loads(plan_result.content)
            except Exception:
                plan_payload = {"raw_result": plan_result.content}
            approved = self._approve_generated_plan(plan_payload)
            self._record_workflow_plan(plan_payload, approved)
            if not approved:
                return _format_plan_rejected_result(
                    target=target,
                    transcriptome_plan=transcriptome_plan,
                    omicverse_payload=omicverse_payload,
                    plan_payload=plan_payload,
                )
        self._emit_progress("计划已通过，执行 DESeq2 fallback 并验证结果文件。")
        payload = self._invoke_registered_capability(
            "deseq2_bulk_deg",
            {
                "count_matrix_path": path,
                "target_tissue": target,
                "output_dir": str(Path(path).resolve().parent),
                "padj_threshold": 0.05,
                "timeout": 600,
            },
        )
        if payload.get("status") == "completed":
            return _format_bulk_rnaseq_result(
                target=target,
                payload=payload,
                transcriptome_plan=transcriptome_plan,
                omicverse_payload=omicverse_payload,
                plan_payload=plan_payload,
            )
        if payload.get("status") == "script_written":
            return _format_bulk_rnaseq_result(
                target=target,
                payload=payload,
                transcriptome_plan=transcriptome_plan,
                omicverse_payload=omicverse_payload,
                plan_payload=plan_payload,
            )
        return _format_bulk_rnaseq_result(
            target=target,
            payload=payload,
            transcriptome_plan=transcriptome_plan,
            omicverse_payload=omicverse_payload,
            plan_payload=plan_payload,
        )

    def _try_fallback_analysis(self, user_message: str, route) -> str | None:
        if not route.fallback_allowed:
            return None
        self._emit_progress(
            f"识别为 {route.analysis_type} 分析任务，当前没有专用封装能力，进入受控 fallback。"
        )
        if not route.input_path:
            self._record_workflow_blocker("input_path_missing")
            return (
                "分析未完成：已进入受控 fallback，但没有找到输入文件路径。\n\n"
                f"- 路由结果: {json.dumps(route.to_dict(), ensure_ascii=False)}\n"
                "- 需要你提供 .csv/.tsv/.txt/.xlsx 等表格路径。\n"
                "- 示例：\"C:\\path\\protein_intensity.csv\" 做蛋白组差异分析。"
            )

        plan_payload = {
            "route": route.to_dict(),
            "preflight_status": "fallback",
            "artifacts": {},
            "message": (
                "No registered domain backend is available. SciCoreAgent will generate one "
                "controlled table-summary script, run it once, and stop on classified errors."
            ),
        }
        approved = not route.requires_plan or self._approve_generated_plan(plan_payload)
        self._record_workflow_plan(plan_payload, approved)
        if route.requires_plan and not approved:
            return (
                "已停止：fallback 计划已生成，但用户未审批继续执行。\n\n"
                f"- 路由结果: {json.dumps(route.to_dict(), ensure_ascii=False)}\n"
                "- 没有运行脚本，也没有生成分析结果。"
            )

        self._emit_progress("写入一个主脚本并立即运行；不进入反复读脚本/改脚本循环。")
        result = run_fallback_analysis(self.root, route.analysis_type, route.input_path)
        payload = result.to_dict()
        self._record_fallback_result(route.analysis_type, payload)
        if result.status == "completed":
            return (
                "受控 fallback 已完成：已生成表格结构与数值列摘要。\n\n"
                "注意：这不是正式领域分析结论；它用于在专用后端缺失时给出可交接的最小结果。\n\n"
                f"- 分析类型: {route.analysis_type}\n"
                f"- 输入文件: {result.input_path}\n"
                f"- 工作目录: {result.work_dir}\n"
                f"- 主脚本: {result.script_path}\n"
                f"- 摘要文件: {result.summary_path}\n"
                f"- stdout: {result.stdout_path}\n"
                f"- stderr: {result.stderr_path}"
            )
        return (
            "分析未完成：受控 fallback 已停止，并给出可诊断 blocker。\n\n"
            f"- 分析类型: {route.analysis_type}\n"
            f"- 输入文件: {result.input_path}\n"
            f"- 错误类型: {result.error_type}\n"
            f"- blocker: {result.blocker}\n"
            f"- 工作目录: {result.work_dir}\n"
            f"- 主脚本: {result.script_path}\n"
            f"- stdout: {result.stdout_path}\n"
            f"- stderr: {result.stderr_path}\n"
            f"- summary: {result.summary_path}\n\n"
            "下一步最小动作：根据 error_type 修复输入路径、数据格式或环境后重跑；不会继续盲目循环。"
        )

    def _invoke_registered_capability(self, name: str, parameters: dict) -> dict:
        invocation, execution, verification, payload = self.capability_registry.invoke(
            self, name, parameters
        )
        manifest = self.current_workflow_manifest
        if manifest is not None:
            manifest.invocations.append(invocation)
            manifest.executions.append(execution)
            manifest.verification = verification
            self.current_workflow_path = self.workflow_store.save(manifest)
        return payload

    def _record_workflow_plan(self, plan_payload: dict, approved: bool) -> None:
        manifest = self.current_workflow_manifest
        if manifest is None:
            return
        artifacts = dict(plan_payload.get("artifacts", {}) or {})
        steps = []
        if isinstance(plan_payload.get("steps"), list):
            steps = [str(item) for item in plan_payload["steps"]]
        manifest.plan.status = "approved" if approved else "rejected"
        manifest.plan.approved_by = "user" if approved else ""
        manifest.plan.approved_at = manifest.updated_at if approved else None
        manifest.plan.steps = steps
        manifest.plan.expected_artifacts = [str(value) for value in artifacts.values() if value]
        self.current_workflow_path = self.workflow_store.save(manifest)

    def _record_workflow_blocker(self, blocker: str) -> None:
        manifest = self.current_workflow_manifest
        if manifest is None:
            return
        manifest.verification = VerificationResult(
            status="blocked",
            checks=[VerificationCheck(name="preflight", status="fail", message=str(blocker))],
        )
        self.current_workflow_path = self.workflow_store.save(manifest)

    def _record_fallback_result(self, analysis_type: str, payload: dict) -> None:
        manifest = self.current_workflow_manifest
        if manifest is None:
            return
        invocation = CapabilityInvocation(
            invocation_id="invocation_" + uuid.uuid4().hex[:12],
            capability_name="controlled_fallback_script",
            backend="generic_script",
            parameters={
                "analysis_type": str(analysis_type),
                "input_path": str(payload.get("input_path", "")),
            },
            expected_artifacts=["script_path", "summary_path", "stdout_path", "stderr_path"],
            deterministic=True,
            risk_level="medium",
        )
        artifact_keys = ("script_path", "summary_path", "stdout_path", "stderr_path")
        artifacts: list[EvidenceArtifact] = [
            evidence_artifact(payload[key], kind=key)
            for key in artifact_keys
            if payload.get(key)
        ]
        summary_artifact = next((item for item in artifacts if item.kind == "summary_path"), None)
        success = str(payload.get("status", "")) == "completed" and bool(
            summary_artifact and summary_artifact.exists
        )
        verification = VerificationResult(
            status="passed" if success else "failed",
            checks=[
                VerificationCheck(
                    name="fallback_status",
                    status="pass" if str(payload.get("status", "")) == "completed" else "fail",
                    message=str(payload.get("status", "")),
                ),
                VerificationCheck(
                    name="summary_artifact",
                    status="pass" if summary_artifact and summary_artifact.exists else "fail",
                    message=str(payload.get("summary_path", "")),
                ),
            ],
            verified_artifacts=[
                item.path for item in artifacts if item.exists and item.kind == "summary_path"
            ],
        )
        execution = ExecutionResult(
            invocation_id=invocation.invocation_id,
            status="completed" if success else "failed",
            backend="generic_script",
            exit_code=payload.get("returncode"),
            artifacts=artifacts,
            stdout_path=str(payload.get("stdout_path", "")),
            stderr_path=str(payload.get("stderr_path", "")),
            error_type=str(payload.get("error_type", "")),
            blocker=str(payload.get("blocker", "")),
        )
        manifest.invocations.append(invocation)
        manifest.executions.append(execution)
        manifest.verification = verification
        self.current_workflow_path = self.workflow_store.save(manifest)

    def _finalize_current_workflow(self, answer: str) -> None:
        manifest = self.current_workflow_manifest
        if manifest is None or manifest.final is not None:
            return
        completed = any(item.status == "completed" for item in manifest.executions)
        degraded = any(item.status == "degraded" for item in manifest.executions)
        passed = bool(manifest.verification and manifest.verification.status == "passed")
        if completed and passed:
            status = "completed"
        elif degraded:
            status = "degraded"
        elif manifest.plan.status == "rejected" or not manifest.executions:
            status = "blocked"
        else:
            status = "failed"
        blockers = [item.blocker for item in manifest.executions if item.blocker]
        if manifest.verification is not None:
            blockers.extend(
                item.message
                for item in manifest.verification.checks
                if item.status == "fail" and item.message
            )
        if status == "completed":
            blockers = []
        evidence_paths = (
            list(manifest.verification.verified_artifacts)
            if manifest.verification is not None
            else []
        )
        manifest.final = FinalSynthesis(
            status=status,
            summary=clip(str(answer), 1200),
            unresolved_blockers=list(dict.fromkeys(blockers)),
            evidence_paths=evidence_paths,
        )
        self.current_workflow_path = self.workflow_store.save(manifest)
        self.audit_trail.log_artifact(
            path=self.current_workflow_path,
            kind="workflow_manifest",
            description="Versioned governed analysis workflow manifest",
            source="workflow_runtime",
            verified=status == "completed",
        )

    def _current_workflow_status(self, default: str) -> str:
        manifest = self.current_workflow_manifest
        if manifest is None or manifest.final is None:
            return default
        return manifest.final.status

    def _emit_progress(self, message: str) -> None:
        callback = getattr(self, "progress_callback", None)
        if callable(callback):
            callback(message)

    def _approve_generated_plan(self, plan_payload: dict) -> bool:
        callback = getattr(self, "plan_approval_callback", None)
        if callable(callback):
            return bool(callback(plan_payload))
        return True

    def _approve_environment_repair(self, command: str, reason: str = "") -> bool:
        callback = getattr(self, "repair_approval_callback", None)
        if callable(callback):
            return bool(callback(command, reason))
        return False

    def maybe_recovery_diagnosis(self, task_state):
        domain_recovery = self._bio_recovery_diagnosis(task_state)
        if domain_recovery:
            return domain_recovery
        return super().maybe_recovery_diagnosis(task_state)

    def _bio_recovery_diagnosis(self, task_state):
        if int(getattr(task_state, "tool_steps", 0) or 0) < self.recovery_threshold():
            return None
        tool_events = [item for item in self.session.get("history", []) if item.get("role") == "tool"]
        recent_text = "\n".join(str(item.get("content", "")) for item in tool_events[-8:])
        recent_names = [str(item.get("name", "")) for item in tool_events[-8:]]
        if "rlang" in recent_text and "DESeq2" in recent_text:
            blocker = _summarize_bulk_rnaseq_blocker({"stderr_tail": recent_text}, {"status": "backend_missing"})
            return (
                "已进入 Recovery Mode：环境修复兜底。\n\n"
                "继续重复运行 R 脚本不会解决问题。\n\n"
                f"- 当前步数: {task_state.tool_steps}/{self.max_steps}\n"
                f"- 最近工具: {recent_names}\n"
                f"- 主要阻塞原因: {blocker}\n\n"
                "为什么没有自动修复：更新 R 包会修改当前 R 环境，属于外部环境变更，必须经过用户审批。\n\n"
                "建议的最小修复动作：\n"
                "1. 批准后运行：Rscript -e \"install.packages('rlang', repos='https://cloud.r-project.org')\"\n"
                "2. 验证：Rscript -e \"packageVersion('rlang'); packageVersion('DESeq2')\"\n"
                "3. 重新运行同一个转录组任务。\n\n"
                "注意：这不是 count 矩阵或 DESeq2 脚本逻辑错误，而是 R 包版本不满足 DESeq2 的加载要求。"
            )
        if any("invalid arguments for read_file" in str(item.get("content", "")) for item in tool_events[-8:]):
            return (
                "已进入 Recovery Mode：工具参数修复兜底。\n\n"
                "模型连续提交了错误的 read_file 参数。\n\n"
                "正确格式是：\n"
                '<tool>{"name":"read_file","args":{"path":"deseq2_analysis.R","start":1,"end":200}}</tool>\n\n'
                "已启用自动解包兜底：如果模型误传 {\"args\":\"{...}\"}，运行时会尝试解包成真正参数。\n"
                "下一步请不要重复读取同一文件；应运行确定性转录组工具或报告当前 blocker。"
            )
        return None

    def _auto_sediment(self, user_message: str, final_answer: str) -> list[dict]:
        events = []
        turn_tools = self._current_turn_tool_events()
        used_tools = {item.get("name", "") for item in turn_tools}

        if REMEMBER_PATTERN.search(user_message) and "wiki_save" not in used_tools:
            saved = self._auto_save_wiki(user_message, final_answer)
            if saved:
                events.append(saved)

        if SKILL_PATTERN.search(user_message) and "skill_save" not in used_tools:
            saved = self._auto_save_skill(user_message, final_answer)
            if saved:
                events.append(saved)

        if "code_literature_link_save" not in used_tools:
            events.extend(self._auto_save_code_literature_links(user_message, final_answer, turn_tools))

        try:
            dream = self._get_dream_manager().submit(
                user_message,
                final_answer,
                turn_tools,
                source_session=str(self.session.get("id", "")),
            )
            events.append({"kind": "dream_background_review", **dream})
        except Exception as exc:
            events.append({"kind": "background_review_error", "error": str(exc)})

        if events:
            self.memory.append_note(
                "Auto-sedimentation: " + ", ".join(event["kind"] for event in events),
                tags=("auto-sedimentation",),
                source="SciCoreAgent",
                kind="process",
            )
            self.session["memory"] = self.memory.to_dict()
            self.session_path = self.session_store.save(self.session)
        return events

    def _get_dream_manager(self):
        if self._dream_manager is None:
            from scicore.agent.sedimentation import DreamSedimentationManager

            self._dream_manager = DreamSedimentationManager(self.root)
        return self._dream_manager

    def dream_status(self, job_id: str) -> dict | None:
        if self._dream_manager is None:
            return None
        return self._dream_manager.status(job_id)

    def shutdown(self) -> None:
        if self._dream_manager is not None:
            self._dream_manager.close()
            self._dream_manager = None
        super().shutdown()

    def _current_turn_tool_events(self) -> list[dict]:
        history = list(self.session.get("history", []))
        start = 0
        for index in range(len(history) - 1, -1, -1):
            if history[index].get("role") == "user":
                start = index
                break
        return [item for item in history[start:] if item.get("role") == "tool"]

    def _auto_save_wiki(self, user_message: str, final_answer: str) -> dict | None:
        try:
            from scicore.agent.wiki import save_entry

            entry = save_entry(
                title=_title_from_text(user_message, "Auto Memory"),
                content=(
                    "## User request\n"
                    f"{clip(user_message, 600)}\n\n"
                    "## SciCoreAgent answer\n"
                    f"{clip(final_answer, 1800)}\n"
                ),
                tags=["auto", "memory"],
                source_session=str(self.session.get("id", "")),
                root=self.root,
            )
            return {"kind": "wiki", "id": entry["id"], "title": entry["title"]}
        except Exception as exc:
            return {"kind": "wiki_error", "error": str(exc)}

    def _auto_save_skill(self, user_message: str, final_answer: str) -> dict | None:
        try:
            from scicore.agent.skills import save_skill

            slug = _slug_from_text(user_message)
            meta = save_skill(
                slug=slug,
                title=_title_from_text(user_message, "Auto Skill"),
                summary=clip(final_answer.replace("\n", " "), 180) or "Auto-saved reusable workflow.",
                trigger=clip(user_message, 500),
                procedure=clip(final_answer, 2400),
                pitfalls="Auto-saved from user request; review before relying on it.",
                source_task=clip(user_message, 800),
                tags=["auto", "reusable"],
                overwrite=True,
                root=self.root,
            )
            return {"kind": "skill", "slug": meta.slug, "path": meta.path}
        except Exception as exc:
            return {"kind": "skill_error", "error": str(exc)}

    def _auto_save_code_literature_links(self, user_message: str, final_answer: str, turn_tools: list[dict]) -> list[dict]:
        literature = _extract_literature_mentions(user_message + "\n" + final_answer)
        if not literature:
            return []

        write_events = [
            item for item in turn_tools
            if item.get("name") in {"write_file", "patch_file"} and item.get("args", {}).get("path")
        ]
        if not write_events:
            return []

        results = []
        try:
            from scicore.agent.provenance import save_code_literature_link
        except Exception as exc:
            return [{"kind": "code_literature_error", "error": str(exc)}]

        for item in write_events:
            raw_path = str(item.get("args", {}).get("path", ""))
            code_path = Path(raw_path)
            absolute = code_path if code_path.is_absolute() else self.root / code_path
            try:
                entry = save_code_literature_link(
                    code_path=str(absolute),
                    code_symbol=_infer_symbol_from_path(raw_path),
                    purpose=clip(user_message, 500),
                    evidence_summary=clip(final_answer, 700),
                    literature=literature,
                    root=self.root,
                )
                results.append({"kind": "code_literature", "id": entry["id"], "code_path": raw_path})
            except Exception as exc:
                results.append({"kind": "code_literature_error", "code_path": raw_path, "error": str(exc)})
        return results

    def build_report(self, task_state):
        report = super().build_report(task_state)
        report["agent_role"] = self.role
        sedimentations = []
        for event in self.last_auto_sedimentations:
            hydrated = dict(event)
            if event.get("kind") == "dream_background_review":
                current = self.dream_status(str(event.get("job_id", "")))
                if current:
                    hydrated.update(current)
            sedimentations.append(hydrated)
        report["auto_sedimentations"] = sedimentations
        return report


def _title_from_text(text: str, fallback: str) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    return (clean[:80] or fallback).strip()


def _format_bulk_rnaseq_result(
    target: str,
    payload: dict,
    transcriptome_plan: dict,
    omicverse_payload: dict,
    plan_payload: dict,
) -> str:
    status = str(payload.get("status") or "failed")
    artifacts = dict(plan_payload.get("artifacts", {}) or {})
    blocker = _summarize_bulk_rnaseq_blocker(payload, omicverse_payload)
    lines = []
    if status == "completed":
        lines.append(f"分析完成：{target} vs rest 的 bulk RNA-seq 差异表达分析已完成。")
    elif status == "script_written":
        lines.append("分析未完成：已生成可运行的 DESeq2 脚本，但当前环境还不能执行。")
    else:
        lines.append("分析未完成：固定转录组流程已经运行，但在执行后端时失败。")

    lines.extend(
        [
            "",
            "流程标签：DESeq2 tissue-vs-rest workflow",
            "",
            "本次执行路径：",
            f"- 任务类型：bulk RNA-seq / DESeq2 tissue-vs-rest",
            f"- 比较设计：{target} vs rest",
            f"- OmicVerse：{omicverse_payload.get('status', 'unknown')}",
            f"- DESeq2 fallback：{status}",
        ]
    )
    steps = transcriptome_plan.get("steps") or []
    if steps:
        lines.append(f"- 能力链：{' -> '.join(str(step) for step in steps)}")

    generated = [
        ("metadata", payload.get("metadata_path")),
        ("R script", payload.get("script_path")),
        ("plan.md", artifacts.get("plan_md")),
        ("plan.json", artifacts.get("plan_json")),
        ("delegation.json", artifacts.get("delegation_json")),
    ]
    if status == "completed":
        generated.extend(
            [
                ("full results", payload.get("full_results_path")),
                ("significant results", payload.get("significant_results_path")),
                ("summary", payload.get("summary_path")),
            ]
        )
    existing = [(name, path) for name, path in generated if path]
    if existing:
        lines.extend(["", "已生成："])
        lines.extend(f"- {name}: {path}" for name, path in existing)
    if artifacts:
        lines.append(f"Plan artifacts: {json.dumps(artifacts, ensure_ascii=False)}")

    if status != "completed":
        missing = [
            ("full results", payload.get("full_results_path")),
            ("significant results", payload.get("significant_results_path")),
            ("summary", payload.get("summary_path")),
        ]
        lines.extend(["", "尚未生成或尚未验证："])
        lines.extend(f"- {name}: {path}" for name, path in missing if path)

    if blocker:
        lines.extend(["", "主要阻塞原因：", f"- {blocker}"])

    if status == "completed":
        summary = payload.get("summary") or {}
        if summary:
            lines.extend(["", "结果摘要：", f"- {json.dumps(summary, ensure_ascii=False)}"])
        lines.extend(["", "下一步：", "- 检查 significant results 和 summary，再决定是否绘制火山图/富集分析。"])
    else:
        lines.extend(
            [
                "",
                "下一步：",
                "- 先修复上述环境问题，然后重新运行同一条任务。",
                "- 不需要手动改 R 脚本；当前脚本已经由确定性工具生成。",
            ]
        )
    return "\n".join(lines)


def _format_plan_rejected_result(
    target: str,
    transcriptome_plan: dict,
    omicverse_payload: dict,
    plan_payload: dict,
) -> str:
    artifacts = dict(plan_payload.get("artifacts", {}) or {})
    lines = [
        "已停止：计划已生成，但用户未审批继续执行。",
        "",
        "本次执行路径：",
        "- 任务类型：bulk RNA-seq / DESeq2 tissue-vs-rest",
        f"- 比较设计：{target} vs rest",
        f"- OmicVerse：{omicverse_payload.get('status', 'unknown')}",
    ]
    steps = transcriptome_plan.get("steps") or []
    if steps:
        lines.append(f"- 能力链：{' -> '.join(str(step) for step in steps)}")
    if artifacts:
        lines.extend(
            [
                "",
                "已生成计划：",
                f"- plan.md: {artifacts.get('plan_md')}",
                f"- plan.json: {artifacts.get('plan_json')}",
                f"- delegation.json: {artifacts.get('delegation_json')}",
            ]
        )
    lines.extend(
        [
            "",
            "下一步：",
            "- 检查 plan.md / plan.json。",
            "- 如果认可计划，重新运行同一任务并在审批提示处输入 y。",
        ]
    )
    return "\n".join(lines)


def _summarize_bulk_rnaseq_blocker(payload: dict, omicverse_payload: dict) -> str:
    stderr = str(payload.get("stderr_tail") or "")
    error = str(payload.get("error") or "")
    if "rlang" in stderr and ">=" in stderr:
        match = re.search(r"namespace 'rlang' ([^ ]+) .*>=\s*([0-9.]+)", stderr)
        if match:
            return f"R 包 rlang 版本过低：当前 {match.group(1)}，需要 >= {match.group(2)}。"
        return "R 包 rlang 版本过低，导致 DESeq2 无法加载。"
    if "DESeq2" in stderr and "there is no package" in stderr:
        return "R 环境缺少 DESeq2 包。"
    if "Rscript not found" in error:
        return "当前环境没有找到 Rscript。"
    if omicverse_payload.get("status") == "backend_missing":
        if error:
            return f"OmicVerse 未安装，且 DESeq2 fallback 失败：{error}"
        if stderr:
            return "OmicVerse 未安装，且 DESeq2 fallback 执行失败。"
        return "OmicVerse 未安装。"
    if error:
        return error
    if stderr:
        return stderr.strip().splitlines()[-1]
    return ""


def _slug_from_text(text: str) -> str:
    lowered = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if tokens:
        slug = "-".join(tokens[:8])
    else:
        slug = "auto-skill-" + str(abs(hash(text)))[:8]
    slug = re.sub(r"[^a-z0-9_-]+", "-", slug).strip("-")
    if len(slug) < 2:
        slug = "auto-skill"
    return slug[:64]


def _extract_literature_mentions(text: str) -> list[dict]:
    items = []
    seen = set()
    for doi in DOI_PATTERN.findall(text):
        clean = doi.rstrip(".。)")
        key = ("doi", clean.lower())
        if key in seen:
            continue
        seen.add(key)
        items.append({"doi": clean})
    for pmid in PMID_PATTERN.findall(text):
        key = ("pmid", pmid)
        if key in seen:
            continue
        seen.add(key)
        items.append({"pmid": pmid})
    return items


def _infer_symbol_from_path(path: str) -> str:
    name = Path(path).stem
    return name if name else ""


def _extract_count_matrix_path(text: str) -> str:
    quoted = re.findall(r'"([^"]+\.(?:txt|tsv|csv))"', text, flags=re.I)
    if quoted:
        return quoted[0]
    unquoted = re.findall(r"([A-Za-z]:\\[^\s\"']+\.(?:txt|tsv|csv))", text, flags=re.I)
    if unquoted:
        return unquoted[0]
    relative = re.findall(r"([^\s\"']+\.(?:txt|tsv|csv))", text, flags=re.I)
    return relative[0] if relative else ""


def _is_bulk_rnaseq_request(text: str) -> bool:
    lowered = str(text or "").lower()
    if "inspect" in lowered and not any(term in lowered for term in ("deseq2", "差异", "比较", "分析", "vs", "与其余")):
        return False
    strong_keywords = (
        "deseq2",
        "rna-seq",
        "rnaseq",
        "bulk rna",
        "转录组",
        "差异表达",
        "表达矩阵",
        "基因表达",
    )
    if any(keyword in lowered for keyword in strong_keywords):
        return True
    count_keywords = ("count matrix", "counts", "count矩阵")
    action_keywords = ("分析", "比较", "计算", "差异", "运行", "run", "compare", "vs", "与其余")
    return any(keyword in lowered for keyword in count_keywords) and any(keyword in lowered for keyword in action_keywords)


def _extract_target_tissue(text: str) -> str:
    patterns = [
        r"计算\s*([A-Za-z]{1,12})\s*与",
        r"([A-Za-z]{1,12})\s*与其余",
        r"([A-Za-z]{1,12})\s*(?:vs|versus)\s*(?:rest|others)",
        r"target[_\s-]*tissue[:=]\s*([A-Za-z]{1,12})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).lower()
    return ""


def _extract_target_tissue(text: str) -> str:
    patterns = [
        r"计算\s*([A-Za-z]{1,12})\s*与",
        r"([A-Za-z]{1,12})\s*与其余",
        r"([A-Za-z]{1,12})\s*和其余",
        r"([A-Za-z]{1,12})\s*对其余",
        r"([A-Za-z]{1,12})\s*(?:vs|versus)\s*(?:rest|others)",
        r"target[_\s-]*group[:=]\s*([A-Za-z]{1,12})",
        r"target[_\s-]*tissue[:=]\s*([A-Za-z]{1,12})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).lower()
    return ""


def _infer_target_tissue_from_matrix_and_text(path: str, text: str) -> str:
    try:
        matrix = Path(path).resolve()
        header = matrix.read_text(encoding="utf-8-sig").splitlines()[0]
    except Exception:
        return ""
    columns = header.split("\t")
    if len(columns) < 3:
        return ""
    prefixes: set[str] = set()
    for sample in columns[1:]:
        match = re.match(r"([A-Za-z]+)", sample.strip())
        if match:
            prefixes.add(match.group(1).lower())
    lowered = str(text).lower()
    candidates = [prefix for prefix in sorted(prefixes, key=len, reverse=True) if re.search(rf"\b{re.escape(prefix)}\b", lowered)]
    return candidates[0] if len(candidates) == 1 else ""
