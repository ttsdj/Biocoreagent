"""File hash provenance tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base import Tool


class FileHashTool(Tool):
    name = "file_hash"
    description = "Compute a file hash for provenance without printing file contents."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "algorithm": {
                "type": "string",
                "enum": ["sha256"],
                "description": "Hash algorithm. First version supports sha256.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, algorithm: str = "sha256") -> str:
        if algorithm != "sha256":
            return "Error: unsupported hash algorithm. Supported: sha256"
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            digest = hashlib.sha256()
            size = 0
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)

            return json.dumps(
                {
                    "file_path": str(p),
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"
