"""Execution policy for commands and bioinformatics workspace safety."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    CONFIRM_REQUIRED = "CONFIRM_REQUIRED"
    BLOCK = "BLOCK"


BLOCK_PATTERNS = [
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    (r"\brm\s+(-\w*)?-rf\s", "force recursive delete"),
    (r"\brm\s+[^;&|]*[*?][^;&|]*", "bulk delete using wildcard"),
    (r"\b(del|erase)\s+[^;&|]*[*?][^;&|]*", "bulk delete using wildcard"),
    (r"\b(rmdir|rd)\s+/(s|S)\b", "recursive directory delete"),
    (r"\bRemove-Item\b[^\n;&|]*(-Recurse|-r\b)", "recursive Remove-Item"),
    (r"\bRemove-Item\b[^\n;&|]*[*?][^\n;&|]*", "bulk Remove-Item using wildcard"),
    (r"\bsudo\b", "sudo is not allowed by default"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?bash", "pipe curl to bash"),
    (r"\bwget\b.*\|\s*(sudo\s+)?bash", "pipe wget to bash"),
    (r"\b(upload|scp|rsync)\b.*\b(data/raw|\.fastq|\.fq|\.bam|\.cram|\.vcf)\b", "possible raw data upload"),
    (r">\s*.*data[/\\]raw[/\\].*", "overwrite in data/raw"),
]

CONFIRM_PATTERNS = [
    (r"(^|[;&|]\s*)(cd|chdir)\b", "changing shell working directory requires confirmation"),
    (r"\b(Set-Location|Push-Location|Pop-Location|pushd|popd)\b", "changing shell working directory requires confirmation"),
    (r"\bnextflow\s+run\b", "workflow execution can be expensive"),
    (r"\bsnakemake\b.*--cores\b", "workflow execution can be expensive"),
    (r"\b(conda|mamba)\s+install\b", "environment changes require confirmation"),
    (r"\bpip\s+install\b", "environment changes require confirmation"),
    (r"\bdocker\s+run\b", "container execution requires confirmation"),
    (r"\bsingularity\s+exec\b", "container execution requires confirmation"),
]


def evaluate_command(command: str) -> tuple[PolicyDecision, str | None]:
    for pattern, reason in BLOCK_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return PolicyDecision.BLOCK, reason
    for pattern, reason in CONFIRM_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return PolicyDecision.CONFIRM_REQUIRED, reason
    return PolicyDecision.ALLOW, None


def is_remote_path_allowed(path: str, remote_work_dir: str) -> bool:
    """Return True only when path is inside the configured remote work directory."""
    try:
        target = _normalize_posix(path)
        root = _normalize_posix(remote_work_dir)
    except ValueError:
        return False
    return target == root or target.startswith(root.rstrip("/") + "/")


def _normalize_posix(path: str) -> str:
    if not path or "\x00" in path:
        raise ValueError("invalid path")
    pure = PurePosixPath(path)
    if not pure.is_absolute():
        raise ValueError("remote path must be absolute")
    parts = []
    for part in pure.parts:
        if part in {"", "/"}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        if part == ".":
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def evaluate_file_write(path: str, workspace_root: str | None = None) -> tuple[PolicyDecision, str | None]:
    """Policy for local file writes.

    SciCoreAgent keeps source-code edits possible, but protects biological input
    zones by default. The protected zones are relative to the current workspace:
    data/raw and data/reference.
    """
    target = Path(path).expanduser().resolve()
    root = Path(workspace_root or ".").resolve()
    protected = [root / "data" / "raw", root / "data" / "reference"]
    for protected_root in protected:
        try:
            target.relative_to(protected_root.resolve())
            return PolicyDecision.BLOCK, f"refusing to write protected input path: {protected_root}"
        except ValueError:
            continue
    return PolicyDecision.ALLOW, None
