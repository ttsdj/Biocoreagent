"""Core agent loop.

The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import logging
from .llm import LLM
from .tools import ALL_TOOLS, get_tool
from .tools.base import Tool
from .tools.agent import AgentTool
from .tools.multiagent import AgentStartTool, AgentTeamStartTool
from .tools.context import ContextStatusTool
from .prompt import system_prompt
from .context import ContextManager
from .audit import AuditLogger
from .events import EventType
from .extensions import GLOBAL_EXTENSION_REGISTRY
from .mcp_client import start_configured_mcp_servers, get_mcp_provider
from .wiki import (
    QuestionTracker,
    detect_explicit_remember,
    search_entries,
    is_duplicate_question,
)

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS + GLOBAL_EXTENSION_REGISTRY.skill_tools()
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.audit = AuditLogger()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
            if isinstance(t, AgentStartTool):
                t._parent_agent = self
            if isinstance(t, AgentTeamStartTool):
                t._parent_agent = self
            if isinstance(t, ContextStatusTool):
                t._parent_agent = self

        # Auto-start configured external MCP servers.
        # Best-effort: if a server fails to start, the agent still works.
        try:
            mcp_provider = start_configured_mcp_servers()
            if mcp_provider.client_count > 0:
                logger.info(
                    "Connected to %d external MCP server(s)",
                    mcp_provider.client_count,
                )
        except Exception as e:
            logger.warning("Failed to start MCP servers: %s", e)

        # Wiki auto-trigger: question tracker for duplicate detection.
        self._wiki_tracker = QuestionTracker()

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        """Return tool schemas for the LLM in OpenAI function-calling format.

        Built-in tools already produce the correct format via Tool.schema().
        External MCP tools use the MCP format (name/description/inputSchema)
        and are wrapped here into the OpenAI function-calling shape.
        """
        builtin = [t.schema() for t in self.tools]
        mcp_schemas = GLOBAL_EXTENSION_REGISTRY.mcp_tool_schemas()
        # Wrap MCP-format schemas into OpenAI function-calling format.
        wrapped_mcp = []
        for schema in mcp_schemas:
            if schema.get("type") == "function" and "function" in schema:
                wrapped_mcp.append(schema)
                continue
            wrapped_mcp.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema.get("name", "unknown"),
                        "description": schema.get("description", ""),
                        "parameters": schema.get(
                            "inputSchema",
                            {"type": "object", "properties": {}, "required": []},
                        ),
                    },
                }
            )
        return builtin + wrapped_mcp

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds.

        Wiki auto-trigger pipeline:
        1. Auto-search wiki for relevant past knowledge → inject as context.
        2. Detect explicit "remember this" → hint LLM to call wiki_save.
        3. Track question for duplicate detection → auto-save when count >= 2.
        """
        # ---- Wiki: auto-search before answering ---------------------------
        wiki_hint = self._build_wiki_hint(user_input)
        effective_input = user_input
        if wiki_hint:
            effective_input = f"{user_input}\n\n[System note: the local wiki contains relevant past knowledge. Consider using wiki_search to retrieve full details if needed.]\n{wiki_hint}"

        self.messages.append({"role": "user", "content": effective_input})
        self.audit.log(EventType.USER_MESSAGE, result_preview=user_input)
        self._maybe_compress()

        # ---- Wiki: detect explicit "remember this" intent -----------------
        explicit_remember = detect_explicit_remember(user_input)

        # ---- Main agent loop ----------------------------------------------
        final_answer = ""
        for _ in range(self.max_rounds):
            self.audit.log(
                EventType.LLM_REQUEST,
                result_preview={"message_count": len(self._full_messages()), "tool_count": len(self.tools)},
            )
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )
            self.audit.log(
                EventType.LLM_RESPONSE,
                result_preview={"has_tool_calls": bool(resp.tool_calls), "content": resp.content},
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                final_answer = resp.content
                break

            # tool calls -> execute (parallel when multiple)
            self.messages.append(resp.message)

            if len(resp.tool_calls) == 1:
                tc = resp.tool_calls[0]
                if on_tool:
                    on_tool(tc.name, tc.arguments)
                result = self._exec_tool(tc)
                self._check_wiki_auto_save(tc, result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            else:
                results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                for tc, result in zip(resp.tool_calls, results):
                    self._check_wiki_auto_save(tc, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            self._maybe_compress()

        if not final_answer:
            final_answer = "(reached maximum tool-call rounds)"

        # ---- Wiki: post-turn duplicate detection --------------------------
        self._maybe_auto_save_wiki(user_input, final_answer, explicit_remember)

        return final_answer

    def _build_wiki_hint(self, user_input: str) -> str:
        """Search wiki for relevant past knowledge.  Returns a compact
        context string to inject, or empty string if nothing relevant."""
        try:
            entries = search_entries(user_input, limit=3)
            if not entries:
                return ""
            lines = ["Relevant past wiki knowledge:"]
            for e in entries:
                snippet = e.get("content", "")[:150].replace("\n", " ")
                lines.append(f"- [{e.get('title', '?')}] {snippet}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _maybe_auto_save_wiki(
        self,
        user_input: str,
        answer: str,
        explicit_remember: bool,
    ) -> None:
        """After the LLM has finished responding, check if wiki auto-save
        should fire: (a) explicit user intent, or (b) duplicate threshold.

        Appends a follow-up system message prompting the LLM to call
        wiki_save — this lets the LLM formulate the title and content
        naturally rather than us trying to auto-generate them.
        """
        # Don't trigger if LLM already saved on its own this turn.
        if getattr(self, "_wiki_already_saved_this_turn", False):
            self._wiki_already_saved_this_turn = False
            return

        # Record the question and check duplicates.
        try:
            dup_info = self._wiki_tracker.record(
                user_input,
                answer_summary=answer[:200],
            )
        except Exception:
            return

        should_save = explicit_remember or dup_info["is_duplicate"]

        if not should_save:
            return

        trigger_reason = (
            "the user explicitly asked to remember this"
            if explicit_remember
            else f"this question has been asked {dup_info['count']} times (similar: {dup_info['similar_questions']})"
        )

        hint = (
            f"[System auto-trigger: {trigger_reason}. "
            "Please call wiki_save NOW to persist this knowledge. "
            "First check with wiki_search if similar entries already exist "
            "to avoid duplicates. If a similar entry exists, update it instead "
            "of creating a near-duplicate.]"
        )
        self.messages.append({"role": "user", "content": hint})

    def _check_wiki_auto_save(self, tc, result: str) -> None:
        """Called after each tool execution.  If the LLM already called
        wiki_save on its own (without auto-trigger), note it so we don't
        double-trigger."""
        # Track whether wiki_save was already called this turn.
        # This prevents the auto-trigger from double-saving.
        if tc.name == "wiki_save":
            if not hasattr(self, "_wiki_already_saved_this_turn"):
                self._wiki_already_saved_this_turn = False
            self._wiki_already_saved_this_turn = True

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._find_tool(tc.name)
        self.audit.log(
            EventType.TOOL_CALL,
            tool_name=tc.name,
            arguments=tc.arguments,
            files=_argument_files(tc.arguments),
        )
        try:
            if tool is not None:
                result = tool.execute(**tc.arguments)
            else:
                mcp_result = GLOBAL_EXTENSION_REGISTRY.call_mcp_tool(tc.name, tc.arguments)
                if mcp_result is None:
                    return f"Error: unknown tool '{tc.name}'"
                result = mcp_result
            self.audit.log(
                EventType.TOOL_RESULT,
                tool_name=tc.name,
                arguments=tc.arguments,
                result_preview=result,
                files=_argument_files(tc.arguments),
                success=not str(result).startswith("Error:"),
            )
            return result
        except TypeError as e:
            self.audit.log(EventType.ERROR, tool_name=tc.name, arguments=tc.arguments, success=False, error=str(e))
            return f"Error: bad arguments for {tc.name}: {e}"
        except Exception as e:
            self.audit.log(EventType.ERROR, tool_name=tc.name, arguments=tc.arguments, success=False, error=str(e))
            return f"Error executing {tc.name}: {e}"

    def _find_tool(self, name: str):
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        When the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()

    def _maybe_compress(self):
        before = len(self.messages)
        compressed = self.context.maybe_compress(self.messages, self.llm)
        if compressed:
            self.audit.log(
                EventType.CONTEXT_COMPRESS,
                result_preview={"messages_before": before, "messages_after": len(self.messages)},
            )
        return compressed


def _argument_files(arguments: dict) -> list[str]:
    files: list[str] = []
    for key, value in arguments.items():
        if key in {"file_path", "path", "output_dir"} and isinstance(value, str):
            files.append(value)
        elif key == "input_files" and isinstance(value, list):
            files.extend(str(item) for item in value)
    return files
