"""Simple result grading for eval runs."""

from __future__ import annotations


def grade_verification(commands: list[dict], expected_files: list[dict]) -> dict:
    command_failures = [cmd for cmd in commands if cmd.get("returncode") != 0]
    missing_files = [item for item in expected_files if item.get("must_exist") and not item.get("exists")]
    passed = not command_failures and not missing_files
    details = []
    for cmd in command_failures:
        details.append(f"command failed: {cmd.get('command')}")
    for item in missing_files:
        details.append(f"expected file missing: {item.get('path')}")
    return {"passed": passed, "details": details}
