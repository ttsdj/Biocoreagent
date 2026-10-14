"""Task file loading for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import EvalTask


def load_task(path: str | Path) -> EvalTask:
    task_path = Path(path).resolve()
    data = _load_mapping(task_path)
    return EvalTask.from_dict(data, base_dir=task_path.parent)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("task file must contain a mapping")
        return data
    except ImportError:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the task.yaml subset used by this project without adding PyYAML."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            item_text = line[2:].strip()
            if not isinstance(parent, list):
                raise ValueError("list item found outside a list")
            if ": " in item_text:
                key, value = item_text.split(": ", 1)
                item = {key: _scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_scalar(item_text))
            continue

        key, value = _split_key_value(line)
        if value == "":
            next_container: Any = []
            if key in {"workspace", "input_contract", "verification", "grading"}:
                next_container = {}
            if isinstance(parent, dict):
                parent[key] = next_container
            else:
                raise ValueError("nested mapping under non-mapping parent")
            stack.append((indent, next_container))
        else:
            if isinstance(parent, dict):
                parent[key] = _scalar(value)
            else:
                raise ValueError("key-value pair under non-mapping parent")
    return result


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"invalid yaml line: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"[]", ""}:
        return [] if value == "[]" else ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",")]
    return value.strip("'\"")
