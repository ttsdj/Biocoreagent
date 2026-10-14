"""Unified CLI for the fused SciCoreAgent runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from scicore.kernel.cli import (
    _build_model_client,
    _configured_secret_names,
    _effective_model,
    _effective_provider,
)
from scicore.kernel.cli import (
    build_arg_parser as build_scicore_kernel_parser,
)
from scicore.kernel.config import _parse_env_line, load_project_env
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext

from .domain import ensure_workspace_state
from .orchestrator import AsyncMultiAgentOrchestrator
from .runtime import BioRuntime

try:
    from colorama import just_fix_windows_console
except Exception:
    just_fix_windows_console = None


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def enable_terminal_style() -> None:
    if just_fix_windows_console is not None:
        just_fix_windows_console()


def assistant_line(text: str) -> str:
    return f"{GREEN}\u25cf{RESET} {text}"


def progress_line(text: str) -> str:
    return f"{DIM}\u25e6 {text}{RESET}"


def print_agent_response(agent, text: str) -> None:
    streamed = {"started": False, "chars": 0}

    def on_progress(message: str) -> None:
        if message:
            print(progress_line(message), flush=True)

    def on_plan_approval(plan: dict) -> bool:
        artifacts = plan.get("artifacts", {}) if isinstance(plan, dict) else {}
        preflight = plan.get("preflight_status", "unknown") if isinstance(plan, dict) else "unknown"
        print(progress_line("计划已生成，等待用户审批后再执行。"), flush=True)
        if artifacts:
            print(progress_line(f"plan.md: {artifacts.get('plan_md', '(none)')}"), flush=True)
            print(progress_line(f"plan.json: {artifacts.get('plan_json', '(none)')}"), flush=True)
        print(progress_line(f"preflight: {preflight}"), flush=True)
        try:
            answer = input(f"{YELLOW}Approve plan and continue? [y/N]{RESET} ").strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes"}

    def on_repair_approval(command: str, reason: str = "") -> bool:
        if reason:
            print(progress_line(reason), flush=True)
        print(progress_line(f"repair command: {command}"), flush=True)
        try:
            answer = input(f"{YELLOW}Approve environment repair? [y/N]{RESET} ").strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes"}

    def on_delta(delta: str) -> None:
        if not delta:
            return
        if not streamed["started"]:
            print(f"{GREEN}\u25cf{RESET} ", end="", flush=True)
            streamed["started"] = True
        print(delta, end="", flush=True)
        streamed["chars"] += len(delta)

    old_progress = getattr(agent, "progress_callback", None)
    old_plan_approval = getattr(agent, "plan_approval_callback", None)
    old_repair_approval = getattr(agent, "repair_approval_callback", None)
    agent.progress_callback = on_progress
    agent.plan_approval_callback = on_plan_approval
    agent.repair_approval_callback = on_repair_approval
    try:
        result = agent.ask(text, stream_callback=on_delta)
    finally:
        agent.progress_callback = old_progress
        agent.plan_approval_callback = old_plan_approval
        agent.repair_approval_callback = old_repair_approval
    if streamed["started"]:
        print()
    else:
        print(assistant_line(result))


def user_prompt() -> str:
    return f"{CYAN}{BOLD}You{RESET} {DIM}>{RESET} "


def global_env_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("SCICORE_ENV")
    if explicit:
        return [Path(explicit).expanduser()]
    home = Path.home()
    candidates.append(home / ".scicore" / ".env")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "SciCoreAgent" / ".env")
    return candidates


def load_env_file(path: Path, override: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def load_scicore_bio_env(root: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for path in global_env_candidates():
        loaded.update(load_env_file(path, override=False))
    loaded.update(load_project_env(root, override=True))
    return loaded


WELCOME_ART = (
    "  ____  _       ____                 _                    _   ",
    " | __ )(_) ___ / ___|___  _ __ ___  / \\   __ _  ___ _ __ | |_ ",
    " |  _ \\| |/ _ \\ |   / _ \\| '__/ _ \\/ _ \\ / _` |/ _ \\ '_ \\| __|",
    " | |_) | | (_) | |__| (_) | | |  __/ ___ \\ (_| |  __/ | | | |_ ",
    " |____/|_|\\___/ \\____\\___/|_|  \\___/_/   \\_\\__, |\\___|_| |_|\\__|",
    "                                            |___/               ",
)


def build_welcome(agent, provider: str, model: str, max_agents: int) -> str:
    width = 78
    inner = width - 4

    def divider(char: str = "-") -> str:
        return "+" + char * (width - 2) + "+"

    def row(text: str = "") -> str:
        text = text[:inner]
        return f"| {text.ljust(inner)} |"

    def center(text: str) -> str:
        text = text[:inner]
        return f"| {text.center(inner)} |"

    lines = [divider("=")]
    lines.extend(center(line) for line in WELCOME_ART)
    lines.extend(
        [
            center("SciCoreAgent"),
            center("auditable multi-agent research and coding harness"),
            divider("-"),
            row(f"Workspace : {agent.workspace.repo_root}"),
            row(f"Provider  : {provider}"),
            row(f"Model     : {model}"),
            row("Role      : executor"),
            row(f"Agents    : max {max_agents} concurrent workers"),
            row(f"Session   : {agent.session['id']}"),
            divider("-"),
            row("Commands  : /help  /memory  /session  /agents  /reset  /exit"),
            row("Identity  : I am SciCoreAgent, a research-oriented coding agent."),
            divider("="),
        ]
    )
    return "\n".join(lines)


def is_workspace_query(text: str) -> bool:
    normalized = text.strip().lower().rstrip("？?。.")
    return normalized in {
        "pwd",
        "cwd",
        "/pwd",
        "/cwd",
        "现在的目录是",
        "现在目录是",
        "当前目录",
        "当前目录是",
        "现在的工作目录是哪里",
        "当前工作目录是哪里",
        "工作目录是哪里",
    }


def build_arg_parser():
    parser = build_scicore_kernel_parser()
    parser.prog = "scicore"
    parser.description = "SciCoreAgent: auditable multi-agent research coding harness."
    parser.set_defaults(max_steps=20, max_new_tokens=4096)
    parser.add_argument("--max-agents", type=int, default=4, help="Maximum concurrent sub-agents.")
    parser.add_argument(
        "--memory-source",
        nargs=3,
        metavar=("SESSION_ID", "START_MESSAGE", "END_MESSAGE"),
        help="Print the exact persisted messages behind a memory source backlink and exit.",
    )
    parser.add_argument(
        "--memory-timeline",
        nargs=2,
        metavar=("USER_SCOPE", "PROJECT_SCOPE"),
        help="Print the effective-time research event timeline for one scope and exit.",
    )
    parser.add_argument(
        "--memory-event",
        metavar="EVENT_ID",
        help="Explain one research graph event with its original evidence links and exit.",
    )
    parser.add_argument(
        "--memory-causal",
        metavar="EVENT_ID",
        help="Print the caused-by/recovery/supersession chain for one event and exit.",
    )
    return parser


def build_runtime(args):
    workspace = WorkspaceContext.build(args.cwd)
    root = Path(workspace.repo_root).resolve()
    os.chdir(root)
    load_scicore_bio_env(root)
    state_root = ensure_workspace_state(root)
    from scicore.agent.mcp_client import start_configured_mcp_servers

    start_configured_mcp_servers(root)
    secret_names = _configured_secret_names(args)
    holder = {}
    worker_config = {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "host": getattr(args, "host", None),
        "ollama_timeout": getattr(args, "ollama_timeout", 300),
        "openai_timeout": getattr(args, "openai_timeout", 300),
        "temperature": getattr(args, "temperature", 0.2),
        "top_p": getattr(args, "top_p", 0.9),
        "max_steps": getattr(args, "max_steps", 20),
        "max_new_tokens": getattr(args, "max_new_tokens", 4096),
        "secret_env_names": list(getattr(args, "secret_env_names", []) or []),
    }

    def create_worker_command(job, worker_workspace: Path) -> list[str]:
        task_path = worker_workspace / "worker_task.txt"
        config_path = worker_workspace / "worker_config.json"
        task_path.write_text(job.task, encoding="utf-8")
        config_path.write_text(
            json.dumps(worker_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [
            sys.executable,
            "-m",
            "scicore.bio.worker_process",
            "--workspace-root",
            str(root),
            "--job-workspace",
            str(worker_workspace),
            "--role",
            job.role,
            "--task-file",
            str(task_path),
            "--config-file",
            str(config_path),
        ]

    def create_agent(role: str = "executor", worker: bool = True, worker_workspace: Path | None = None):
        agent_root = Path(worker_workspace).resolve() if worker_workspace is not None else root
        agent_state_root = ensure_workspace_state(agent_root)
        session_store = SessionStore(agent_state_root / "sessions")
        run_store = RunStore(agent_state_root / "runs")
        return BioRuntime(
            model_client=_build_model_client(args),
            workspace=WorkspaceContext.build(agent_root),
            session_store=session_store,
            run_store=run_store,
            approval_policy="auto" if worker else args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=secret_names,
            role=role,
            orchestrator=holder.get("orchestrator"),
            allow_orchestration=not worker,
        )

    orchestrator = AsyncMultiAgentOrchestrator(
        root,
        agent_factory=create_agent,
        max_concurrency=args.max_agents,
        process_command_factory=create_worker_command,
        default_execution_mode="process",
    )
    holder["orchestrator"] = orchestrator

    if args.resume:
        session_store = SessionStore(state_root / "sessions")
        session_id = session_store.latest() if args.resume == "latest" else args.resume
        if not session_id:
            raise ValueError("no session is available to resume")
        agent = BioRuntime.from_session(
            model_client=_build_model_client(args),
            workspace=workspace,
            session_store=session_store,
            session_id=session_id,
            run_store=RunStore(state_root / "runs"),
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=secret_names,
            role="executor",
            orchestrator=orchestrator,
        )
    else:
        agent = create_agent("executor", worker=False)
    return agent, orchestrator


def main(argv=None):
    enable_terminal_style()
    args = build_arg_parser().parse_args(argv)
    if args.memory_source:
        session_id, start, end = args.memory_source
        cwd_root = Path(args.cwd).resolve()
        repo_root = Path(WorkspaceContext.build(args.cwd).repo_root).resolve()
        state_roots = [
            cwd_root / ".scicore" / "sessions",
            cwd_root / ".scicore/kernel" / "sessions",
            repo_root / ".scicore" / "sessions",
            repo_root / ".scicore/kernel" / "sessions",
        ]
        store_root = next((path for path in state_roots if path.exists()), None)
        if store_root is None:
            print("memory source error: no local session store exists", file=sys.stderr)
            return 2
        try:
            excerpt = SessionStore(store_root).source_excerpt(session_id, int(start), int(end))
        except (ValueError, TypeError) as exc:
            print(f"memory source error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"session_id": session_id, "messages": excerpt}, ensure_ascii=False, indent=2))
        return 0
    if args.memory_timeline or args.memory_event or args.memory_causal:
        root = Path(WorkspaceContext.build(args.cwd).repo_root).resolve()
        load_scicore_bio_env(root)
        from scicore.kernel.features.postgres_memory import PostgresLongTermMemoryStore

        backend = PostgresLongTermMemoryStore.from_environment()
        if backend is None:
            print(
                "research graph error: SCICORE_MEMORY_POSTGRES_DSN is not configured",
                file=sys.stderr,
            )
            return 2
        if args.memory_timeline:
            user_scope, project_scope = args.memory_timeline
            result = backend.research_graph.timeline(
                user_scope=user_scope, project_scope=project_scope
            )
        elif args.memory_event:
            result = backend.research_graph.explain_event(args.memory_event)
        else:
            result = backend.research_graph.causal_chain(args.memory_causal)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    try:
        agent, orchestrator = build_runtime(args)
    except (RuntimeError, ValueError) as exc:
        print(f"startup error: {exc}", file=sys.stderr)
        return 2

    provider = _effective_provider(args)
    model = _effective_model(args, provider)
    print(build_welcome(agent, provider=provider, model=model, max_agents=args.max_agents))

    try:
        if args.prompt:
            prompt = " ".join(args.prompt).strip()
            if prompt:
                if is_workspace_query(prompt):
                    print(assistant_line(agent.workspace.repo_root))
                    return 0
                try:
                    print_agent_response(agent, prompt)
                except RuntimeError as exc:
                    print(f"runtime error: {exc}", file=sys.stderr)
                    return 1
                except KeyboardInterrupt:
                    print(assistant_line("interrupted; use /exit in interactive mode when you want to leave."))
                    return 130
            return 0

        while True:
            try:
                text = input(user_prompt()).strip()
            except EOFError:
                print(assistant_line("input closed; use /exit next time for a clean shutdown."))
                return 0
            except KeyboardInterrupt:
                print(assistant_line("cleared current input. Type /exit to leave SciCoreAgent."))
                continue
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                return 0
            if text == "/help":
                print(assistant_line(
                    "/pwd  /memory  /session  /agents  /job ID  /team ID  "
                    "/cancel ID  /reset  /exit"
                ))
                continue
            if is_workspace_query(text):
                print(assistant_line(agent.workspace.repo_root))
                continue
            if text == "/memory":
                print(assistant_line(agent.memory_text()))
                continue
            if text == "/session":
                print(assistant_line(str(agent.session_path)))
                continue
            if text == "/agents":
                jobs = orchestrator.store.list_jobs()
                print(assistant_line("\n".join(f"{job.job_id} {job.role} {job.status} {job.name}" for job in jobs) or "(none)"))
                continue
            if text.startswith("/job "):
                print(assistant_line(orchestrator.job_status(text.split(maxsplit=1)[1])))
                continue
            if text.startswith("/team "):
                print(assistant_line(orchestrator.team_status(text.split(maxsplit=1)[1])))
                continue
            if text.startswith("/cancel "):
                print(assistant_line(str({"cancel_requested": orchestrator.cancel_job(text.split(maxsplit=1)[1])})))
                continue
            if text == "/reset":
                agent.reset()
                print(assistant_line("session reset"))
                continue
            try:
                print_agent_response(agent, text)
            except RuntimeError as exc:
                print(assistant_line(f"runtime error: {exc}"), file=sys.stderr)
            except KeyboardInterrupt:
                print(assistant_line("interrupted; back to prompt. Type /exit to leave SciCoreAgent."))
    finally:
        agent.shutdown()
        orchestrator.shutdown()
        try:
            from scicore.agent.mcp_client import get_mcp_provider

            get_mcp_provider().shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
