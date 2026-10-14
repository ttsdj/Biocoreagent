"""Persistent links between generated code and literature evidence."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROVENANCE_DIR = Path(".scicore") / "provenance"
LINKS_FILE = "code_literature_links.jsonl"


def provenance_dir(root: str | Path | None = None) -> Path:
    return Path(root or ".").resolve() / DEFAULT_PROVENANCE_DIR


def links_path(root: str | Path | None = None) -> Path:
    return provenance_dir(root) / LINKS_FILE


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    return hashlib.sha256(target.read_bytes()).hexdigest()


def save_code_literature_link(
    code_path: str,
    literature: list[dict[str, Any]],
    purpose: str,
    code_symbol: str = "",
    evidence_summary: str = "",
    generated_by: str = "SciCoreAgent",
    root: str | Path | None = None,
) -> dict[str, Any]:
    base = provenance_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    path = links_path(root)
    entry = {
        "id": uuid.uuid4().hex[:12],
        "code_path": str(code_path),
        "code_symbol": str(code_symbol),
        "code_sha256": file_sha256(code_path),
        "purpose": str(purpose),
        "evidence_summary": str(evidence_summary),
        "literature": [_compact_literature(item) for item in literature],
        "generated_by": generated_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def list_code_literature_links(root: str | Path | None = None) -> list[dict[str, Any]]:
    path = links_path(root)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def search_code_literature_links(query: str, limit: int = 10, root: str | Path | None = None) -> list[dict[str, Any]]:
    terms = [term.lower() for term in str(query).split() if term.strip()]
    if not terms:
        return list_code_literature_links(root)[:limit]
    scored = []
    for item in list_code_literature_links(root):
        text = json.dumps(item, ensure_ascii=False).lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, item.get("created_at", ""), item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in scored[:limit]]


def _compact_literature(item: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "Title": "title",
        "Year": "year",
        "Journal": "journal",
        "DOI": "doi",
        "PMID": "pmid",
        "Core Conclusion": "core_conclusion",
    }
    normalized = {}
    for key, value in dict(item).items():
        normalized[aliases.get(key, key)] = value
    keep = (
        "pmid",
        "doi",
        "title",
        "journal",
        "year",
        "core_conclusion",
        "evidence_strength",
        "limitations",
        "url",
    )
    return {key: normalized.get(key, "") for key in keep if normalized.get(key, "")}
