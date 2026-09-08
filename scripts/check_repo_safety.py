"""Repository safety gate for BiocoreagentV2.0 releases."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
BLOCKED_NAMES = {
    ".env",
}
BLOCKED_PARTS = {
    ".biocoreagent",
    ".pytest_cache",
    ".tmp",
    "tmp",
    "output",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
BLOCKED_SUFFIXES = {
    ".bam",
    ".bai",
    ".cram",
    ".crai",
    ".fastq",
    ".fq",
    ".h5ad",
    ".rds",
    ".rdata",
    ".sqlite",
    ".db",
    ".whl",
}
BLOCKED_DOUBLE_SUFFIXES = {
    ".fastq.gz",
    ".fq.gz",
    ".tar.gz",
}
SECRET_PATTERNS = [
    re.compile(r"sk-ws-[A-Za-z0-9_.-]{24,}"),
    re.compile(r"sk-(?!benchmark-secret\b|test\b)[A-Za-z0-9_.-]{24,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{24,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*=\s*['\"]?(?!replace-me|changeme|your-|example|os\.environ|args\.|config\.|api_key|token|secret|password)([A-Za-z0-9_.-]{32,})"),
]


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if ".git" in parts:
        return True
    if parts & BLOCKED_PARTS:
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    if "docs" in parts and "biocore-v1" in parts:
        return True
    return False


def release_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        relative_paths = [
            Path(item.decode("utf-8", errors="replace"))
            for item in result.stdout.split(b"\0")
            if item
        ]
        return [ROOT / path for path in relative_paths]
    except Exception:
        return [path for path in ROOT.rglob("*") if path.is_file()]


def main() -> int:
    problems: list[str] = []
    for path in release_files():
        if path.is_dir() or should_skip(path):
            continue
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        lower = rel_text.lower()
        parts = set(rel.parts)
        if path.name in BLOCKED_NAMES:
            problems.append(f"blocked local secret file: {rel_text}")
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"file exceeds 5MB: {rel_text}")
        if path.suffix.lower() in BLOCKED_SUFFIXES or any(lower.endswith(item) for item in BLOCKED_DOUBLE_SUFFIXES):
            problems.append(f"blocked data/build artifact extension: {rel_text}")
        if path.suffix.lower() in {".py", ".toml", ".md", ".txt", ".yml", ".yaml", ".json", ".example", ".ps1"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"possible secret in {rel_text}: {pattern.pattern}")
                    break

    if problems:
        print("Repository safety check failed:")
        for item in problems:
            print(f"- {item}")
        return 1
    print("Repository safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
