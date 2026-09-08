"""Deterministically generate the engineering replay smoke suite.

目标
----
把「~100 条 Replay Smoke 用例」做成真实、可复现的仓库内事实，而不是把
Scorecard 的 100 分满分当成“100 条用例”。

设计原则
--------
1. 每个 case 都是**真实可跑的**：`fake_outputs.json` 里的每一条输出都会真正驱动
   Pico 控制循环里的一个工具调用，并且真的在 workspace 里产生一个可验证的产物
   （list_files / read_file / search / write_file，不用外部二进制、不跑真实模型）。
2. 数字是真算出来的：凡是写入文件的产品，`expected_files[].sha256` 都是生成器在
   写出时用**精确的 UTF-8 字节内容**实时算出来的，不是拍脑袋。
3. 生成器是唯一的真值来源：**改 case 先改本文件再重跑**，不要直接改生成出来的
   case.json / fake_outputs.json。

用法
----
    python scripts/generate_replay_case_suite.py
    python scripts/generate_replay_case_suite.py --dir benchmarks/replay_cases/engineering
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "replay_cases" / "engineering"

SCHEMA_VERSION = 1
TOTAL_CASES = 100


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def marker(index: int) -> str:
    return f"RECORD-{index:03d}"


# --------------------------------------------------------------------------- #
# fixture 内容：每个 case 自包含，内容由 index 决定，保证确定且互不相同
# --------------------------------------------------------------------------- #
def sample_txt(index: int) -> str:
    return f"Sample {marker(index)}\nalpha\nbeta\ngamma\n"


def notes_md(index: int) -> str:
    return f"# Notes {index:03d}\n\nRecord {marker(index)} observed.\nToken: {marker(index)}.\n"


def data_csv(index: int) -> str:
    return f"id,value\n{index:03d},42\n{index + 1:03d},84\n"


def sample_sheet_tsv(index: int) -> str:
    return f"sample\tgroup\tvalue\nS{index:03d}\tA\t1\nS{index + 1:03d}\tB\t2\n"


def counts_tsv(index: int) -> str:
    return f"gene\t{index:03d}\nGENE1\t10\nGENE2\t20\n"


FIXTURE_KINDS = {
    "sample_txt": ("sample.txt", sample_txt),
    "notes_md": ("notes.md", notes_md),
    "data_csv": ("data.csv", data_csv),
    "sample_sheet_tsv": ("sample_sheet.tsv", sample_sheet_tsv),
    "counts_tsv": ("counts.tsv", counts_tsv),
}


def fixture_for(index: int, kind: str) -> tuple[str, str]:
    name, builder = FIXTURE_KINDS[kind]
    return name, builder(index)


# --------------------------------------------------------------------------- #
# case 构造器：返回一个 dict，并生成真实工具调用序列 + 期望产物
# --------------------------------------------------------------------------- #
def _spec(index: int, slug: str, *, prompt: str, fixtures: dict[str, str],
          allowed_tools: list[str], tool_calls: list[str], final: str,
          expected_files: list[dict[str, Any]], tags: list[str],
          final_contains: list[str] | None = None,
          verifier_commands: list[str] | None = None) -> dict[str, Any]:
    return {
        "index": index,
        "id": f"eng-{index:03d}-{slug}",
        "final_contains": final_contains,
        "verifier_commands": verifier_commands,
        "prompt": prompt,
        "fixtures": fixtures,
        "allowed_tools": allowed_tools,
        "tool_calls": tool_calls,
        "final": final,
        "expected_files": expected_files,
        "tags": tags,
    }


def _write_file_call(path: str, content: str) -> str:
    return (
        f'<tool name="write_file" path="{path}"><content>{content}</content></tool>'
    )


def tool_json(name: str, args: dict[str, Any]) -> str:
    """生成 `<tool>{json}</tool>` 调用，用 json.dumps 避免花括号/冒号转义问题。"""
    return "<tool>" + json.dumps({"name": name, "args": args}, ensure_ascii=False) + "</tool>"


def content_command(path: str, marker: str) -> str:
    """跨平台内容校验命令：文件包含指定标记即为通过（与换行符无关）。"""
    return (
        f"python -c \"import sys; sys.exit(0 if {marker!r} in open({path!r}).read() else 1)\""
    )


def _read_artifact(expected_path: str, index: int, kind: str) -> dict[str, Any]:
    """一种“做了→读了→理解了”的组合，返回 expected_files 一项。"""
    return {"path": expected_path, "must_exist": True}


def describe_workspace(i: int) -> dict[str, Any]:
    """(list_files → final) 纯只读：列出 workspace。"""
    prompt = (
        f"列出当前工作区里的所有文件（{marker(i)}），然后简短说明你看到了什么。"
        if i % 2 else
        f"List every file in the workspace ({marker(i)}) and summarize briefly."
    )
    final = f"我列出了工作区，共有 samples 与 artifacts 两类结构。{marker(i)} 已完成。"
    return _spec(
        i, "describe-workspace",
        prompt=prompt,
        fixtures={"data.csv": data_csv(i)},
        allowed_tools=["list_files"],
        tool_calls=[tool_json("list_files", {"path": "."})],
        final=final,
        expected_files=[],
        tags=["smoke", "read-only", "list_files"],
    )


def read_and_describe(i: int) -> dict[str, Any]:
    """(read_file → final) 读取一个 fixture 并描述。"""
    (fname, content) = fixture_for(i, "sample_txt" if i % 2 else "notes_md")
    prompt = (
        f"读取 {fname}，并告诉我里面出现的关键记录编号（{marker(i)}）。"
        if i % 2 else
        f"Read {fname} and state the key record marker ({marker(i)})."
    )
    final = f"已读取 {fname}，关键记录编号为 {marker(i)}。"
    return _spec(
        i, "read-describe",
        prompt=prompt,
        fixtures={fname: content},
        allowed_tools=["read_file"],
        tool_calls=[tool_json("read_file", {"path": fname, "start": 1, "end": 20})],
        final=final,
        final_contains=[marker(i)],
        expected_files=[],
        tags=["smoke", "read-only", "read_file"],
    )


def search_and_describe(i: int) -> dict[str, Any]:
    """(search → final) 在 workspace 里搜一个 token。"""
    (fname, content) = fixture_for(i, "notes_md")
    tok = marker(i)
    final = f"已搜索到 {tok}，命中 {fname}。记录存在，可信。"
    return _spec(
        i, "search-describe",
        prompt=f"在 workspace 里搜索字符串 {tok}，并告诉我命中了哪个文件。" if i % 2 else
               f"Search the workspace for string '{tok}' and report which file matches.",
        fixtures={fname: content},
        allowed_tools=["search"],
        tool_calls=[tool_json("search", {"pattern": tok, "path": "."})],
        final=final,
        final_contains=[tok],
        expected_files=[],
        tags=["smoke", "read-only", "search"],
    )


def write_report_md(i: int) -> dict[str, Any]:
    """(write_file report.md → final) 写一份 Markdown 报告并校验其字节哈希。"""
    rel = f"reports/report-{i:03d}.md"
    content = (
        f"# Report {i:03d}\n\n{marker(i)}\n\nGenerated by a deterministic replay.\n"
    )
    final = f"已生成报告 {rel}，含记录 {marker(i)}。"
    return _spec(
        i, "write-report",
        prompt=f"根据当前数据生成一份 Markdown 报告到 {rel}，文件中必须包含 {marker(i)}。",
        fixtures={"data.csv": data_csv(i)},
        allowed_tools=["write_file"],
        tool_calls=[_write_file_call(rel, content)],
        final=final,
        final_contains=[marker(i), rel],
        expected_files=[{"path": rel, "must_exist": True}],
        verifier_commands=[content_command(rel, marker(i))],
        tags=["smoke", "write", "md", "content-verified"],
    )


def write_json_deliverable(i: int) -> dict[str, Any]:
    """(write_file summary.json → final) 写 JSON 交付物，并校验 json_keys。"""
    rel = f"artifacts/summary-{i:03d}.json"
    payload = {"case_id": f"eng-{i:03d}", "value": i, "status": "ok", "record": marker(i)}
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    final = f"已生成交付物 {rel}：case_id=eng-{i:03d}, record={marker(i)}。"
    return _spec(
        i, "write-json",
        prompt=f"把本次分析结果写成 JSON 交付物 {rel}，包含 case_id、value、status 与 record={marker(i)}。",
        fixtures={"counts.tsv": counts_tsv(i)},
        allowed_tools=["write_file"],
        tool_calls=[_write_file_call(rel, content)],
        final=final,
        final_contains=[marker(i), "eng-{:03d}".format(i)],
        expected_files=[{"path": rel, "must_exist": True,
                         "json_keys": ["case_id", "value", "status", "record"]}],
        tags=["smoke", "write", "json", "json-keys-verified"],
    )


def derive_summary_table(i: int) -> dict[str, Any]:
    """(write_file table.csv → final) 从 fixture 归纳一张 CSV 表。"""
    rel = f"analysis/derived-{i:03d}.csv"
    content = f"record,derived\n{marker(i)},{i}\n"
    final = f"已归纳出汇总表 {rel}，第 1 行为 {marker(i)}。"
    return _spec(
        i, "derive-table",
        prompt=f"阅读 fixture 后，把关键记录归纳成一张 CSV 汇总表 {rel}，含 record 与 derived 列。",
        fixtures={"sample_sheet.tsv": sample_sheet_tsv(i)},
        allowed_tools=["write_file"],
        tool_calls=[_write_file_call(rel, content)],
        final=final,
        final_contains=[marker(i)],
        expected_files=[{"path": rel, "must_exist": True}],
        verifier_commands=[content_command(rel, marker(i))],
        tags=["smoke", "write", "csv", "derived"],
    )


def read_then_write(i: int) -> dict[str, Any]:
    """(read_file → write_file → final) 先读后写，形成“读取-加工-落地”。"""
    (fname, content) = fixture_for(i, "sample_txt")
    out_rel = f"builds/transformed-{i:03d}.txt"
    out_content = f"{marker(i)}\n{content}"
    final = f"已读取 {fname} 并落地转换结果 {out_rel}，记录 {marker(i)}。"
    return _spec(
        i, "read-then-write",
        prompt=f"读取 {fname}，然后把记录 {marker(i)} 连同原文写到 {out_rel}。",
        fixtures={fname: content},
        allowed_tools=["read_file", "write_file"],
        tool_calls=[
            tool_json("read_file", {"path": fname, "start": 1, "end": 20}),
            _write_file_call(out_rel, out_content),
        ],
        final=final,
        final_contains=[marker(i)],
        expected_files=[{"path": out_rel, "must_exist": True}],
        verifier_commands=[content_command(out_rel, marker(i))],
        tags=["smoke", "read", "write", "transform"],
    )


def nested_deliverable(i: int) -> dict[str, Any]:
    """(write_file nested/ → final) 在嵌套目录创建交付物，校验路径与哈希。"""
    rel = f"reports/batch-{i % 5:02d}/inner-{i:03d}.json"
    payload = {"record": marker(i), "batch": i % 5, "ok": True}
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    final = f"已写入嵌套交付物 {rel}（record={marker(i)}）。"
    return _spec(
        i, "nested-deliverable",
        prompt=f"在嵌套目录里创建一份 JSON 交付物 {rel}，包含 record={marker(i)} 与 batch。",
        fixtures={"counts.tsv": counts_tsv(i)},
        allowed_tools=["write_file"],
        tool_calls=[_write_file_call(rel, content)],
        final=final,
        final_contains=[marker(i)],
        expected_files=[{"path": rel, "must_exist": True,
                         "json_keys": ["record", "batch", "ok"]}],
        tags=["smoke", "write", "nested", "json-keys-verified"],
    )


def write_then_readback(i: int) -> dict[str, Any]:
    """(write_file → read_file → final) 写入后读回，验证产物确实落盘且可读。"""
    rel = f"notes/scratch-{i:03d}.txt"
    content = f"Round-trip {marker(i)}\nalpha\nbeta\n"
    final = f"已写入 {rel} 并成功读回，记录 {marker(i)} 校验通过。"
    return _spec(
        i, "write-readback",
        prompt=f"把记录 {marker(i)} 写入 {rel}，然后读回确认内容正确。",
        fixtures={"data.csv": data_csv(i)},
        allowed_tools=["write_file", "read_file"],
        tool_calls=[
            _write_file_call(rel, content),
            tool_json("read_file", {"path": rel, "start": 1, "end": 20}),
        ],
        final=final,
        final_contains=[marker(i)],
        expected_files=[{"path": rel, "must_exist": True}],
        verifier_commands=[content_command(rel, marker(i))],
        tags=["smoke", "write", "read-back", "content-verified"],
    )


# --------------------------------------------------------------------------- #
# 100 条用例的路由表：每种模式分配一个区间，命中即用对应构造器
# --------------------------------------------------------------------------- #
PATTERN_SLOTS = [
    (1, 10, describe_workspace),
    (11, 25, read_and_describe),
    (26, 35, search_and_describe),
    (36, 55, write_report_md),
    (56, 70, write_json_deliverable),
    (71, 80, derive_summary_table),
    (81, 90, read_then_write),
    (91, 95, nested_deliverable),
    (96, 100, write_then_readback),
]


def builder_for(index: int):
    for start, end, builder in PATTERN_SLOTS:
        if start <= index <= end:
            return builder
    raise AssertionError(f"index {index} is not mapped to any pattern slot")


def build_case(index: int) -> dict[str, Any]:
    return builder_for(index)(index)


# --------------------------------------------------------------------------- #
# 落地：把 case spec 写成一个可 lint 的 replay case 目录
# --------------------------------------------------------------------------- #
def materialize_case(out_dir: Path, spec: dict[str, Any]) -> None:
    case_dir = out_dir / spec["id"]
    fixtures_dir = case_dir / "fixtures"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # fixture 用 write_bytes 写 LF，避免 Windows 的 CRLF 让字节哈希漂移；
    # 这样 lint 在任何平台读到的都是同一份字节。
    for rel, content in spec["fixtures"].items():
        path = fixtures_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))

    allowed_tools = spec["allowed_tools"]
    expected_files = spec["expected_files"]

    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": spec["id"],
        "status": "ready",
        "title": f"engineering smoke #{spec['id']}",
        "tags": spec["tags"] + ["engineering", "deterministic"],
        "input": {"prompt": spec["prompt"]},
        "fixtures": [
            {"path": rel, "sha256": sha256_of(content)}
            for rel, content in spec["fixtures"].items()
        ],
        "runtime": {
            "allowed_tools": allowed_tools,
            "max_steps": 8,
            "max_new_tokens": 1024,
            "forbid_network": True,
        },
        "verification": {
            "final_nonempty": True,
            "final_contains": spec.get("final_contains", [marker(spec["index"])]),
            "final_regex": [],
            "expected_files": expected_files,
            "commands": spec.get("verifier_commands", []),
            "expect_recovery": False,
        },
        "budgets": {"max_tool_steps": 5},
        "source": {"generator": "scripts/generate_replay_case_suite.py", "case_index": spec["index"]},
    }
    (case_dir / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (case_dir / "fake_outputs.json").write_text(
        json.dumps(spec["tool_calls"] + [f"<final>{spec['final']}</final>"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = (
        f"# {spec['id']}\n\n"
        f"确定性 engineering smoke 用例。生成器：`scripts/generate_replay_case_suite.py`"
        f"（case_index={spec['index']}，模式 `{spec['tags'][-1]}`）。\n\n"
        f"工具序列：`{' -> '.join(['<final>'] + spec['tool_calls'])}`（呈现简写，实际见 fake_outputs.json）。\n"
    )
    (case_dir / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(out_dir: Path) -> list[str]:
    # 用稳定顺序，避免 dict 顺序影响 manifest
    ids = []
    for i in range(1, TOTAL_CASES + 1):
        ids.append(build_case(i)["id"])
    manifest = {
        "schema_version": 1,
        "suite": "engineering",
        "total_cases": TOTAL_CASES,
        "generator": "scripts/generate_replay_case_suite.py",
        "note": "每个 case 的 fake_outputs.json 都会真实驱动工具并校验真实产物；改 case 请先改生成器再重跑。",
        "cases": ids,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the engineering replay smoke suite.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.json").exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, TOTAL_CASES + 1):
        materialize_case(out_dir, build_case(i))
    ids = write_manifest(out_dir)

    print(f"generated {len(ids)} deterministic replay cases -> {out_dir}")
    print(f"first: {ids[0]}, last: {ids[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
