"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .session import save_session, load_session, list_sessions
from . import __version__
from .audit import default_audit_path, tail_events
from .workspace import workspace_summary
from .multiagent import GLOBAL_MULTIAGENT_MANAGER
from .skills import list_skills

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="scicore.agent",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $SCICORE_AGENT_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    executable = os.path.splitext(os.path.basename(sys.argv[0]))[0].lower()
    if executable.startswith("scicore.bio"):
        from scicore.bio.cli import main as scicore_bio_main

        return scicore_bio_main()

    args = _parse_args()
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or SCICORE_AGENT_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 SCICORE_AGENT_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config)


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    print()


def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]SciCoreAgent[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.scicore_agent_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent._maybe_compress()
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/context":
            status = agent.context.status(agent.messages)
            console.print(
                f"Context: [cyan]{status['estimated_tokens']}[/cyan] / "
                f"[cyan]{status['max_tokens']}[/cyan] tokens "
                f"({status['utilization']:.1%}); next compact action: "
                f"[bold]{status['next_compaction_action']}[/bold]"
            )
            console.print(f"Messages: {status['message_count']}  Thresholds: {status['thresholds']}")
            continue
        if user_input == "/audit":
            path = default_audit_path()
            console.print(f"[bold]Audit log:[/bold] [cyan]{path}[/cyan]")
            events = tail_events(path, limit=5)
            if not events:
                console.print("[dim]No audit events yet.[/dim]")
            else:
                for event in events:
                    console.print(
                        f"  [cyan]{event.get('event_type')}[/cyan] "
                        f"{event.get('tool_name') or ''} "
                        f"success={event.get('success')}"
                    )
            continue
        if user_input == "/workspace":
            console.print(workspace_summary())
            continue
        if user_input == "/bio-help":
            _show_bio_help()
            continue
        if user_input == "/skills":
            _show_skills()
            continue
        if user_input == "/jobs":
            _show_jobs()
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: scicore.agent -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /context       Show context usage and compaction thresholds\n"
        "  /audit         Show audit log path and recent events\n"
        "  /workspace     Show SciCoreAgent workspace conventions\n"
        "  /bio-help      Show bioinformatics tool support\n"
        "  /skills        List persisted local skills\n"
        "  /jobs          Show background agent progress\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit SciCoreAgent\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="SciCoreAgent Help",
        border_style="dim",
    ))


def _show_jobs():
    jobs = GLOBAL_MULTIAGENT_MANAGER.list_jobs()
    if not jobs:
        console.print("[dim]No background agent jobs.[/dim]")
        return
    for job in jobs[:20]:
        progress = job.progress
        console.print(
            f"[cyan]{job.job_id}[/cyan] "
            f"role={job.role} status={job.status} "
            f"{progress.bar()} step={progress.current_step}"
        )


def _show_skills():
    skills = list_skills()
    if not skills:
        console.print("[dim]No persisted skills yet.[/dim]")
        return
    for item in skills[:30]:
        tags = f" tags={','.join(item.tags)}" if item.tags else ""
        console.print(f"[cyan]{item.slug}[/cyan] {item.title} - {item.summary}{tags}")


def _show_bio_help():
    console.print(Panel(
        "[bold]Supported first-phase bioinformatics features:[/bold]\n"
        "  bio_seq_inspect  Inspect FASTA/FASTQ metadata without printing sequences\n"
        "  bio_sample_sheet_inspect  Validate sample metadata TSV without inferring groups\n"
        "  bio_count_matrix_inspect  Inspect gene-by-sample count matrices\n"
        "  bio_rnaseq_compare  Lightweight observed-count two-group comparison\n"
        "  bio_report      Write observed bioinformatics results to reports/*.md\n"
        "  file_hash        Compute sha256 provenance for input files\n"
        "  bio_workflow_sketch  Draft Snakemake/Nextflow skeletons without guessing metadata\n"
        "  agent_start / agent_status  Run role-restricted sub-agents in the background\n"
        "  agent_team_start / agent_team_status  Run small role-restricted background teams\n"
        "  /jobs          Show background sub-agent progress bars\n"
        "  context_status Show context usage and compaction thresholds\n"
        "  ssh_bash        Execute configured remote SSH commands inside BIO_REMOTE_WORK_DIR\n"
        "  extensions      Inspect registered Skill and MCP extension providers\n"
        "  audit log        Record tool calls, command runs, and policy blocks\n"
        "  policy           Block destructive commands and require confirmation for costly workflow commands\n"
        "  eval harness     Run task.yaml checks with: scicore-eval path/to/task.yaml\n"
        "\n"
        "[bold]Not supported in this first phase:[/bold]\n"
        "  BAM/CRAM/VCF parsing, clinical interpretation, automatic public data download,\n"
        "  running large pipelines by default, or inventing missing biological metadata.",
        title="SciCoreAgent Help",
        border_style="green",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
