"""Execution lineage helpers for transcriptome capability runs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def append_transcriptome_provenance(
    root: str | Path,
    capability: str,
    tool: str,
    params: dict[str, Any],
    input_state: dict[str, Any],
    output_state: dict[str, Any],
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve() / ".scicore" / "provenance"
    base.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": "txprov_" + uuid.uuid4().hex[:10],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "capability": capability,
        "tool": tool,
        "params": params,
        "input_state": input_state,
        "output_state": output_state,
        "artifacts": artifacts or {},
    }
    path = base / "transcriptome_lineage.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    entry["path"] = str(path)
    return entry
