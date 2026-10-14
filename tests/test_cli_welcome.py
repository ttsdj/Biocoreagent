from types import SimpleNamespace

import scicore.bio.cli as cli
from scicore.bio.cli import assistant_line, build_arg_parser, build_welcome, is_workspace_query, load_scicore_bio_env


def test_welcome_banner_uses_scicore_bio_identity():
    agent = SimpleNamespace(
        workspace=SimpleNamespace(repo_root="D:\\workspace"),
        session={"id": "session-1"},
    )

    banner = build_welcome(agent, provider="deepseek", model="deepseek-chat", max_agents=4)

    assert "SciCoreAgent" in banner
    assert "auditable multi-agent research and coding harness" in banner
    assert "research-oriented coding agent" in banner
    assert "local coding agent" not in banner


def test_global_env_loads_when_workspace_has_no_env(tmp_path, monkeypatch):
    global_env = tmp_path / "global.env"
    global_env.write_text(
        "SCICORE_PROVIDER=deepseek\nSCICORE_DEEPSEEK_API_KEY=global-key\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("SCICORE_ENV", str(global_env))
    monkeypatch.delenv("SCICORE_PROVIDER", raising=False)
    monkeypatch.delenv("SCICORE_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_project_env", lambda root, override=True: {})

    loaded = load_scicore_bio_env(workspace)

    assert loaded["SCICORE_PROVIDER"] == "deepseek"
    assert loaded["SCICORE_DEEPSEEK_API_KEY"] == "global-key"


def test_workspace_env_overrides_global_env(tmp_path, monkeypatch):
    global_env = tmp_path / "global.env"
    global_env.write_text("SCICORE_DEEPSEEK_MODEL=global-model\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SCICORE_DEEPSEEK_MODEL=project-model\n", encoding="utf-8")

    monkeypatch.setenv("SCICORE_ENV", str(global_env))
    monkeypatch.delenv("SCICORE_DEEPSEEK_MODEL", raising=False)
    real_load_project_env = cli.load_project_env
    monkeypatch.setattr(
        cli,
        "load_project_env",
        lambda root, override=True: real_load_project_env(workspace, override=override),
    )

    load_scicore_bio_env(workspace)

    assert __import__("os").environ["SCICORE_DEEPSEEK_MODEL"] == "project-model"


def test_workspace_query_recognizes_common_chinese_and_shell_forms():
    assert is_workspace_query("pwd")
    assert is_workspace_query("/pwd")
    assert is_workspace_query("现在的目录是")
    assert is_workspace_query("当前工作目录是哪里？")
    assert not is_workspace_query("请解释当前目录里的项目架构")


def test_scicore_bio_default_step_budget_handles_longer_tasks():
    args = build_arg_parser().parse_args([])

    assert args.max_steps == 20
    assert args.max_new_tokens == 4096


def test_assistant_line_uses_real_bullet():
    assert "\u25cf" in assistant_line("hello")
