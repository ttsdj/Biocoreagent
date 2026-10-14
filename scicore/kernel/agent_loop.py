"""Agent control loop extracted from the runtime facade."""

import time

from .checkpoint import (
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_PARTIAL_STALE_STATUS,
    CHECKPOINT_WORKSPACE_MISMATCH_STATUS,
)
from .task_state import (
    PHASE_CONTEXT_BUILDING,
    PHASE_FINALIZING,
    PHASE_MODEL_CALLING,
    PHASE_OUTPUT_PARSING,
    PHASE_RECOVERING,
    PHASE_TERMINATED,
    PHASE_TOOL_EXECUTING,
    TaskState,
)
from .workspace import clip, now


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def run(self, user_message, stream_callback=None):
        agent = self.agent
        agent.tool_manager.begin_run()
        run_started_at = time.monotonic()
        agent.memory.set_task_summary(user_message)
        agent.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=agent.new_run_id(), task_id=agent.new_task_id(), user_request=user_message)
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_state = task_state
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )
        agent.emit_trace(task_state, "user_input_recorded", {"user_input": user_message})

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < agent.max_steps and attempts < max_attempts:
            attempts += 1
            _transition_phase(agent, task_state, PHASE_CONTEXT_BUILDING)
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            _emit_progress(agent, f"准备第 {attempts} 轮上下文。")
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            _transition_phase(agent, task_state, PHASE_MODEL_CALLING)
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            _emit_progress(agent, "请求模型生成下一步动作。")
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            safe_stream = _FinalStreamFilter(stream_callback) if callable(stream_callback) else None
            try:
                raw = _request_model(
                    agent,
                    prompt,
                    safe_stream,
                    prompt_cache_key,
                    prompt_cache_retention,
                )
            except Exception as exc:
                _finish_failed_model_run(
                    agent,
                    task_state,
                    exc,
                    run_started_at=run_started_at,
                )
                raise
            _transition_phase(agent, task_state, PHASE_OUTPUT_PARSING)
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind, payload = agent.parse(raw)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                _transition_phase(agent, task_state, PHASE_TOOL_EXECUTING)
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                _emit_progress(agent, f"执行工具：{name}。")
                task_state.record_tool(name)
                agent.emit_trace(
                    task_state,
                    "tool_call_started",
                    {"name": name, "args": args},
                )
                tool_started_at = time.monotonic()
                tool_result = agent.execute_tool(name, args)
                result = tool_result.content
                agent.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                recovery_final = agent.maybe_recovery_diagnosis(task_state)
                if recovery_final:
                    _transition_phase(agent, task_state, PHASE_RECOVERING)
                    _emit_progress(agent, "检测到低进展循环，进入 Recovery Mode。")
                    agent.emit_trace(
                        task_state,
                        "recovery_started",
                        {
                            "reason": "low_progress_loop",
                            "tool_steps": task_state.tool_steps,
                            "attempts": task_state.attempts,
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "recovery_action",
                        {"action": "stop_and_diagnose"},
                    )
                    _transition_phase(agent, task_state, PHASE_FINALIZING)
                    agent.record({"role": "assistant", "content": recovery_final, "created_at": now()})
                    task_state.stop("recovery_gate_triggered", final_answer=recovery_final)
                    agent.promote_durable_memory(user_message, recovery_final)
                    agent.run_store.write_task_state(task_state)
                    agent.emit_trace(
                        task_state,
                        "recovery_gate_triggered",
                        {
                            "final_answer": recovery_final,
                            "tool_steps": task_state.tool_steps,
                            "attempts": task_state.attempts,
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "final_delivery",
                        {"final_answer": recovery_final, "delivery_status": "stopped"},
                    )
                    _transition_phase(agent, task_state, PHASE_TERMINATED)
                    agent.emit_trace(
                        task_state,
                        "run_finished",
                        {
                            "status": task_state.status,
                            "stop_reason": task_state.stop_reason,
                            "final_answer": recovery_final,
                            "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                        },
                    )
                    agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
                    return recovery_final
                continue

            if kind == "retry":
                agent.record({"role": "assistant", "content": payload, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            _transition_phase(agent, task_state, PHASE_FINALIZING)
            final = agent.finalize_answer((payload or raw).strip())
            _emit_progress(agent, "形成最终回答。")
            agent.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            agent.promote_durable_memory(user_message, final)
            checkpoint = agent.create_checkpoint(task_state, user_message, trigger="run_finished")
            agent.run_store.write_task_state(task_state)
            agent.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            agent.emit_trace(
                task_state,
                "final_delivery",
                {"final_answer": final, "delivery_status": "completed"},
            )
            _transition_phase(agent, task_state, PHASE_TERMINATED)
            agent.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
            return final

        _transition_phase(agent, task_state, PHASE_FINALIZING)
        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = (
                "已停止：模型连续多次没有返回有效的工具调用或最终答案。"
                "这通常是模型没有遵守工具协议，或者任务提示过于开放。"
                "请把任务拆小，或换一个更稳定的模型后重试。"
            )
            task_state.stop_retry_limit(final)
        else:
            final = (
                f"已停止：达到本次任务的步骤上限 max_steps={agent.max_steps}，"
                "但还没有形成最终答案。复杂任务可以用 --max-steps 调高上限，"
                "或把任务拆成“先调研/再整理/再导出文件”几个阶段。"
            )
            task_state.stop_step_limit(final)
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.promote_durable_memory(user_message, final)
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "final_delivery",
            {"final_answer": final, "delivery_status": task_state.status},
        )
        _transition_phase(agent, task_state, PHASE_TERMINATED)
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final


def _transition_phase(agent, task_state, phase):
    previous = task_state.phase
    task_state.transition_phase(phase)
    agent.run_store.write_task_state(task_state)
    if previous != phase:
        agent.emit_trace(
            task_state,
            "task_phase_changed",
            {"from_phase": previous, "to_phase": phase},
        )


def _request_model(
    agent,
    prompt,
    safe_stream,
    prompt_cache_key,
    prompt_cache_retention,
):
    if safe_stream and hasattr(agent.model_client, "complete_stream"):
        try:
            raw = agent.model_client.complete_stream(
                prompt,
                agent.max_new_tokens,
                on_delta=safe_stream.feed,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            safe_stream.finish()
            return raw
        except Exception:
            if safe_stream.emitted:
                raise
    return agent.model_client.complete(
        prompt,
        agent.max_new_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )


def _finish_failed_model_run(agent, task_state, exc, *, run_started_at):
    _transition_phase(agent, task_state, PHASE_FINALIZING)
    final = f"model error: {exc.__class__.__name__}: {clip(str(exc), 300)}"
    task_state.stop_model_error(final)
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "exception_raised",
        {
            "stage": PHASE_MODEL_CALLING,
            "exception_type": exc.__class__.__name__,
            "message": clip(str(exc), 300),
        },
    )
    _transition_phase(agent, task_state, PHASE_TERMINATED)
    agent.emit_trace(
        task_state,
        "run_finished",
        {
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": final,
            "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
        },
    )
    agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))


def _emit_progress(agent, message: str) -> None:
    callback = getattr(agent, "progress_callback", None)
    if callable(callback):
        callback(message)


class _FinalStreamFilter:
    """Stream only content inside <final>...</final>, never internal tool calls."""

    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.in_final = False
        self.closed = False
        self.emitted = False

    def feed(self, delta: str) -> None:
        if self.closed or not delta:
            return
        self.buffer += str(delta)
        while self.buffer:
            if not self.in_final:
                start = self.buffer.find("<final>")
                tool_start = self.buffer.find("<tool")
                if tool_start != -1 and (start == -1 or tool_start < start):
                    self.closed = True
                    self.buffer = ""
                    return
                if start == -1:
                    self.buffer = self.buffer[-16:]
                    return
                self.buffer = self.buffer[start + len("<final>") :]
                self.in_final = True
            end = self.buffer.find("</final>")
            if end == -1:
                if self.buffer:
                    self.callback(self.buffer)
                    self.emitted = True
                    self.buffer = ""
                return
            chunk = self.buffer[:end]
            if chunk:
                self.callback(chunk)
                self.emitted = True
            self.buffer = self.buffer[end + len("</final>") :]
            self.closed = True
            return

    def finish(self) -> None:
        if self.in_final and not self.closed and self.buffer:
            self.callback(self.buffer)
            self.emitted = True
            self.buffer = ""
