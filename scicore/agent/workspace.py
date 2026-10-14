"""Workspace conventions for SciCoreAgent projects."""

from __future__ import annotations

from pathlib import Path


def workspace_paths(root: str | Path | None = None) -> dict[str, str]:
    base = Path(root or ".").resolve()
    return {
        "raw": str(base / "data" / "raw"),
        "reference": str(base / "data" / "reference"),
        "work": str(base / "work"),
        "results": str(base / "results"),
        "reports": str(base / "reports"),
        "audit": str(base / ".scicore" / "audit" / "events.jsonl"),
    }


def workspace_summary(root: str | Path | None = None) -> str:
    paths = workspace_paths(root)
    return (
        "SciCoreAgent workspace convention:\n"
        f"- data/raw: {paths['raw']} (read-only input data)\n"
        f"- data/reference: {paths['reference']} (provided or explicit reference files)\n"
        f"- work: {paths['work']} (intermediate files)\n"
        f"- results: {paths['results']} (analysis outputs)\n"
        f"- reports: {paths['reports']} (human-readable reports)\n"
        f"- audit log: {paths['audit']}\n"
    )
