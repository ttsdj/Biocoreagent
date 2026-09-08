"""会话语义切块 (C3) 的测试.

`create_chunks` 新增 ``strategy="fixed"|"semantic"``：
  - "fixed"    —— 默认，按 `max_chars` 在消息边界处贪心打包（行为与改造前一致）。
  - "semantic" —— 在角色/工具切换、Markdown 标题行、段落空行（\\n\\n）这类自然
    边界处优先断块，避免把跨段落/跨语轮的上下文从中间切开。

语义版与 fixed 版产出相同的 chunk_id / source_uri 寻址格式（内容 hash 不变）。
"""

from __future__ import annotations

import re
import shutil
import tempfile

import pytest

from pico.session_store import SessionStore


@pytest.fixture
def chunk_root():
    """一个干净的临时目录，避开 pytest 的 basetemp 权限问题。"""
    root = tempfile.mkdtemp(prefix="biocoreagent-chunks-")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _store_with(root, history, session_id="sess-semantic-1"):
    store = SessionStore(root)
    store.save(
        {
            "id": session_id,
            "workspace_root": str(root),
            "created_at": "",
            "history": history,
        }
    )
    return store, session_id


def _msg(role, content, name=""):
    item = {"role": role, "content": content}
    if name:
        item["name"] = name
    return item


def _assert_addressable(chunk):
    # chunk_id 与 source_uri 均携带内容 hash 与消息区间，可回溯到不动原文。
    assert re.match(r"^chunk_.*_\d+_\d+_[0-9a-f]{16}$", chunk["chunk_id"])
    assert chunk["source_uri"].startswith("session://")
    assert "#sha256=" in chunk["source_uri"]
    assert chunk["content_sha256"] == chunk["source_uri"].split("sha256=", 1)[1]


# ---------------------------------------------------------------------------
# fixed（默认）行为保持不变
# ---------------------------------------------------------------------------


def test_default_strategy_packs_messages_into_few_chunks(chunk_root):
    history = [
        _msg("user", "a"),
        _msg("assistant", "b"),
        _msg("user", "c"),
        _msg("assistant", "d"),
    ]
    store, sid = _store_with(chunk_root, history)

    default_chunks = store.create_chunks(sid)
    fixed_chunks = store.create_chunks(sid, strategy="fixed")

    # 默认 = 显式 fixed，且（默认 max_chars 足够大）4 条合成 1 个 chunk。
    assert [c["chunk_id"] for c in default_chunks] == [c["chunk_id"] for c in fixed_chunks]
    assert len(default_chunks) == 1
    assert len(default_chunks[0]["content"]) < 1800


# ---------------------------------------------------------------------------
# semantic —— 角色/工具切换处断块
# ---------------------------------------------------------------------------


def test_semantic_breaks_at_role_switches(chunk_root):
    history = [
        _msg("user", "a"),
        _msg("assistant", "b"),
        _msg("user", "c"),
        _msg("assistant", "d"),
    ]
    store, sid = _store_with(chunk_root, history)

    fixed = store.create_chunks(sid, strategy="fixed")
    semantic = store.create_chunks(sid, strategy="semantic")

    # fixed：4 条合成 1 块；semantic：每个角色切换都成独立块。
    assert len(fixed) == 1
    assert len(semantic) == 4

    lines = [s["content"] for s in semantic]
    assert lines[0] == "[user] a"
    assert lines[1] == "[assistant] b"
    assert lines[2] == "[user] c"
    assert lines[3] == "[assistant] d"
    for chunk in semantic:
        _assert_addressable(chunk)


def test_semantic_breaks_at_tool_switch_within_same_role(chunk_root):
    history = [
        _msg("assistant", "reading", name="read_file"),
        _msg("assistant", "writing", name="write_file"),
    ]
    store, sid = _store_with(chunk_root, history)

    # 同为 assistant，但工具名不同 → semantic 仍应断块。
    semantic = store.create_chunks(sid, strategy="semantic")
    assert [s["content"] for s in semantic] == [
        "[assistant:read_file] reading",
        "[assistant:write_file] writing",
    ]


# ---------------------------------------------------------------------------
# semantic —— 标题行 / 段落空行断块
# ---------------------------------------------------------------------------


def test_semantic_breaks_at_markdown_heading(chunk_root):
    history = [
        _msg("assistant", "plain paragraph"),
        _msg("assistant", "# Title Section"),
    ]
    store, sid = _store_with(chunk_root, history)

    fixed = store.create_chunks(sid, strategy="fixed")
    semantic = store.create_chunks(sid, strategy="semantic")

    # fixed：两条同角色消息合为 1 块；semantic：第二条以标题开头 → 另起一块。
    assert len(fixed) == 1
    assert len(semantic) == 2
    assert semantic[1]["content"] == "[assistant] # Title Section"


def test_semantic_breaks_at_paragraph_blankline(chunk_root):
    history = [
        _msg("assistant", "para one"),
        _msg("assistant", "para two\n\npara three"),
    ]
    store, sid = _store_with(chunk_root, history)

    semantic = store.create_chunks(sid, strategy="semantic")
    assert len(semantic) == 2
    assert semantic[0]["content"] == "[assistant] para one"


# ---------------------------------------------------------------------------
# semantic —— 同一段落内仍可打包且地址格式兼容
# ---------------------------------------------------------------------------


def test_semantic_packs_within_same_section(chunk_root):
    """同一角色、无标题、无空行的连续消息应仍可合并。"""
    history = [
        _msg("assistant", "step one"),
        _msg("assistant", "step two"),
    ]
    store, sid = _store_with(chunk_root, history)

    semantic = store.create_chunks(sid, strategy="semantic")
    # 两条同角色、无断块点 → 合为 1 块，而不是强行切成 1 块/条。
    assert len(semantic) == 1
    assert "step one" in semantic[0]["content"]
    assert "step two" in semantic[0]["content"]
    _assert_addressable(semantic[0])
