"""JSONL evidence store for Protocol RAG.

Stores structured protocol evidence (ProtocolCards, TaskCards, ResourceCards)
and raw document chunks in plain JSONL files under .scicore/evidence/.

Design:
- Simple: plain JSONL, one record per line, human-readable and grep-friendly.
- Portable: no database, no server, works wherever Python works.
- Auditable: every record has a created_at timestamp.
- Phase 1 search: keyword matching (like skills.py:search_skills).
  Upgrade path: SQLite FTS, Chroma, LanceDB, FAISS, or pgvector later.

Reference: SciCoreAgent 需求文档, Section 7.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .schemas import EvidenceSpan, ProtocolCard, TaskCard, ResourceCard

DEFAULT_EVIDENCE_DIR = Path(".scicore") / "evidence"

# JSONL file names under the evidence directory.
PROTOCOLS_FILE = "protocols.jsonl"
TASKS_FILE = "tasks.jsonl"
RESOURCES_FILE = "resources.jsonl"
CHUNKS_FILE = "chunks.jsonl"


class EvidenceStore:
    """Append-only JSONL store for protocol evidence.

    Each method that writes appends a single line to the corresponding
    JSONL file.  Reads scan the files linearly — acceptable for Phase 1
    dataset sizes (hundreds to low thousands of records).
    """

    def __init__(self, root: str | Path | None = None):
        self._base = (Path(root or ".").resolve() / DEFAULT_EVIDENCE_DIR).resolve()
        # Ensure the directory exists so tools can write immediately.
        self._base.mkdir(parents=True, exist_ok=True)

    # -- write helpers -------------------------------------------------------

    def _append_jsonl(self, filename: str, obj: dict | BaseModel) -> None:
        """Append one JSON record to a JSONL file, creating it if needed."""
        data = obj.model_dump() if isinstance(obj, BaseModel) else obj
        with (self._base / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def _read_jsonl(self, filename: str) -> list[dict]:
        """Read all records from a JSONL file.  Returns empty list if missing."""
        path = self._base / filename
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    # -- public write API ----------------------------------------------------

    def add_protocol(self, card: ProtocolCard) -> None:
        """Persist a ProtocolCard to protocols.jsonl."""
        self._append_jsonl(PROTOCOLS_FILE, card)

    def add_task(self, card: TaskCard) -> None:
        """Persist a TaskCard to tasks.jsonl."""
        self._append_jsonl(TASKS_FILE, card)

    def add_resource(self, card: ResourceCard) -> None:
        """Persist a ResourceCard to resources.jsonl."""
        self._append_jsonl(RESOURCES_FILE, card)

    def add_chunk(
        self,
        source_id: str,
        chunk_id: str,
        text: str,
        source_type: str = "unknown",
        location: str = "",
    ) -> dict:
        """Persist a raw text chunk to chunks.jsonl.

        Returns the chunk dict that was written (includes auto-generated id).
        """
        chunk = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_type": source_type,
            "location": location,
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._append_jsonl(CHUNKS_FILE, chunk)
        return chunk

    # -- public read API -----------------------------------------------------

    def get_protocol(self, protocol_id: str) -> ProtocolCard | None:
        """Look up a single ProtocolCard by its protocol_id."""
        for row in self._read_jsonl(PROTOCOLS_FILE):
            if row.get("protocol_id") == protocol_id:
                return ProtocolCard(**row)
        return None

    def list_protocols(self) -> list[ProtocolCard]:
        """Return all stored ProtocolCards."""
        return [ProtocolCard(**row) for row in self._read_jsonl(PROTOCOLS_FILE)]

    def get_tasks(self, protocol_id: str) -> list[TaskCard]:
        """Return all TaskCards belonging to *protocol_id*."""
        return [
            TaskCard(**row)
            for row in self._read_jsonl(TASKS_FILE)
            if row.get("protocol_id") == protocol_id
        ]

    def get_resources(self, protocol_id: str) -> list[ResourceCard]:
        """Return all ResourceCards belonging to *protocol_id*."""
        return [
            ResourceCard(**row)
            for row in self._read_jsonl(RESOURCES_FILE)
            if row.get("protocol_id") == protocol_id
        ]

    def get_chunk(self, source_id: str, chunk_id: str) -> dict | None:
        """Retrieve a single chunk by its composite key."""
        for row in self._read_jsonl(CHUNKS_FILE):
            if row.get("source_id") == source_id and row.get("chunk_id") == chunk_id:
                return row
        return None

    def get_chunks(self, source_id: str) -> list[dict]:
        """Return all chunks belonging to a source document, in order."""
        chunks = [
            row
            for row in self._read_jsonl(CHUNKS_FILE)
            if row.get("source_id") == source_id
        ]
        chunks.sort(key=lambda c: c.get("chunk_id", ""))
        return chunks

    # -- search --------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search across all evidence files.

        Phase 1: simple term-frequency scoring, identical in spirit to
        skills.py:search_skills().  Each unique word in *query* is counted
        against the lowercased JSON line; the sum is the relevance score.

        Upgrade path: replace this method body with embedding-based
        retrieval (Chroma, LanceDB, FAISS, pgvector) without changing
        the call signature.
        """
        terms = [t.lower() for t in re.findall(r"[\w-]+", query) if t.strip()]
        if not terms:
            return []

        results: list[tuple[int, dict]] = []
        for filename in [PROTOCOLS_FILE, TASKS_FILE, RESOURCES_FILE, CHUNKS_FILE]:
            for row in self._read_jsonl(filename):
                lower = json.dumps(row, ensure_ascii=False, default=str).lower()
                score = sum(lower.count(term) for term in terms)
                if score > 0:
                    results.append((score, row))

        results.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in results[:limit]]

    def search_by_type(
        self,
        query: str,
        search_type: str = "all",
        limit: int = 10,
    ) -> list[dict]:
        """Search within a specific evidence type.

        *search_type*: "all", "protocols", "tasks", "resources", or "chunks".
        """
        terms = [t.lower() for t in re.findall(r"[\w-]+", query) if t.strip()]
        if not terms:
            return []

        file_map = {
            "all": [PROTOCOLS_FILE, TASKS_FILE, RESOURCES_FILE, CHUNKS_FILE],
            "protocols": [PROTOCOLS_FILE],
            "tasks": [TASKS_FILE],
            "resources": [RESOURCES_FILE],
            "chunks": [CHUNKS_FILE],
        }
        filenames = file_map.get(search_type, file_map["all"])

        results: list[tuple[int, dict]] = []
        for filename in filenames:
            for row in self._read_jsonl(filename):
                lower = json.dumps(row, ensure_ascii=False, default=str).lower()
                score = sum(lower.count(term) for term in terms)
                if score > 0:
                    results.append((score, row))

        results.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in results[:limit]]
