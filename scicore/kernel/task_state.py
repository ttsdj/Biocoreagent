"""一次 ask() 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

TASK_TRANSITIONS = {
    STATUS_RUNNING: frozenset({STATUS_COMPLETED, STATUS_STOPPED, STATUS_FAILED}),
    STATUS_COMPLETED: frozenset(),
    STATUS_STOPPED: frozenset(),
    STATUS_FAILED: frozenset(),
}

PHASE_INITIALIZING = "initializing"
PHASE_CONTEXT_BUILDING = "context_building"
PHASE_MODEL_CALLING = "model_calling"
PHASE_OUTPUT_PARSING = "output_parsing"
PHASE_TOOL_EXECUTING = "tool_executing"
PHASE_RECOVERING = "recovering"
PHASE_FINALIZING = "finalizing"
PHASE_TERMINATED = "terminated"

TASK_PHASE_TRANSITIONS = {
    PHASE_INITIALIZING: frozenset({PHASE_CONTEXT_BUILDING, PHASE_FINALIZING}),
    PHASE_CONTEXT_BUILDING: frozenset({PHASE_MODEL_CALLING, PHASE_FINALIZING}),
    PHASE_MODEL_CALLING: frozenset({PHASE_OUTPUT_PARSING, PHASE_FINALIZING}),
    PHASE_OUTPUT_PARSING: frozenset(
        {PHASE_CONTEXT_BUILDING, PHASE_TOOL_EXECUTING, PHASE_FINALIZING}
    ),
    PHASE_TOOL_EXECUTING: frozenset(
        {PHASE_CONTEXT_BUILDING, PHASE_RECOVERING, PHASE_FINALIZING}
    ),
    PHASE_RECOVERING: frozenset({PHASE_FINALIZING}),
    PHASE_FINALIZING: frozenset({PHASE_TERMINATED}),
    PHASE_TERMINATED: frozenset(),
}

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_DELEGATE_FAILED = "delegate_failed"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""
    phase: str = PHASE_INITIALIZING

    def __post_init__(self):
        if self.status not in TASK_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle state: {self.status!r}")
        if self.phase not in TASK_PHASE_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle phase: {self.phase!r}")

    @classmethod
    def create(cls, task_id, user_request, run_id=""):
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data):
        status = str(data.get("status", STATUS_RUNNING))
        default_phase = PHASE_INITIALIZING if status == STATUS_RUNNING else PHASE_TERMINATED
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            user_request=str(data.get("user_request", "")),
            status=status,
            tool_steps=int(data.get("tool_steps", 0)),
            attempts=int(data.get("attempts", 0)),
            last_tool=str(data.get("last_tool", "")),
            stop_reason=str(data.get("stop_reason", "")),
            final_answer=str(data.get("final_answer", "")),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            resume_status=str(data.get("resume_status", "")),
            phase=str(data.get("phase", default_phase)),
        )

    def record_attempt(self):
        # attempt 统计的是“模型被调用了几轮”，不等于 tool_steps。
        self.attempts += 1
        return self

    def record_tool(self, name):
        # tool_steps 只统计真正进入执行阶段的工具调用次数。
        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def stop(self, stop_reason, status=STATUS_STOPPED, final_answer=""):
        # stop_reason 和 status 分开存，是为了区分“怎么停的”和“停下时是什么状态”。
        self.transition_to(status)
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer=""):
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer=""):
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_model_error(self, final_answer=""):
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def finish_success(self, final_answer):
        self.transition_to(STATUS_COMPLETED)
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
        return self

    def transition_to(self, status):
        status = str(status)
        if self.status not in TASK_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle state: {self.status!r}")
        if status not in TASK_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle state: {status!r}")
        if status != self.status and status not in TASK_TRANSITIONS[self.status]:
            raise ValueError(f"illegal task lifecycle transition: {self.status!r} -> {status!r}")
        self.status = status
        return self

    def transition_phase(self, phase):
        phase = str(phase)
        if self.phase not in TASK_PHASE_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle phase: {self.phase!r}")
        if phase not in TASK_PHASE_TRANSITIONS:
            raise ValueError(f"unknown task lifecycle phase: {phase!r}")
        if phase != self.phase and phase not in TASK_PHASE_TRANSITIONS[self.phase]:
            raise ValueError(
                f"illegal task lifecycle phase transition: {self.phase!r} -> {phase!r}"
            )
        self.phase = phase
        return self

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
            "phase": self.phase,
        }
