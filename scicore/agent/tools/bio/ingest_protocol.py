"""Ingest a protocol document into the evidence store.

Reads a bioinformatics protocol file (Markdown or plain text), splits it
into overlapping chunks, and stores them in the evidence store for later
structured extraction.

Phase 1 supports Markdown and plain text.  PDF support requires an
optional dependency (pdfplumber or PyPDF2) and will be added later.

Reference: SciCoreAgent 需求文档, Section 9.1.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from ..base import Tool
from ...bio.evidence_store import EvidenceStore

# Characters per chunk (soft target; chunks respect paragraph boundaries).
DEFAULT_CHUNK_SIZE = 2000
# Overlap between consecutive chunks to avoid splitting key sentences.
CHUNK_OVERLAP = 100


class BioIngestProtocolTool(Tool):
    name = "bio_ingest_protocol"
    description = (
        "Read a bioinformatics protocol document (Markdown or plain text) "
        "and ingest it into the evidence store.  The document is split into "
        "overlapping chunks for later structured extraction.  "
        "Use before bio_extract_protocol to prepare the evidence base."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the protocol document (Markdown .md or plain text .txt).",
            },
            "source_type": {
                "type": "string",
                "enum": ["markdown", "text", "pdf"],
                "description": "Format of the source document. PDF support requires an optional dependency.",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Approximate characters per chunk. Default 2000.",
            },
        },
        "required": ["file_path", "source_type"],
    }

    def execute(
        self,
        file_path: str,
        source_type: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> str:
        # --- validate input -------------------------------------------------
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return json.dumps({
                "success": False,
                "error": f"File not found: {file_path}",
            })

        if source_type == "pdf":
            return json.dumps({
                "success": False,
                "error": (
                    "PDF reading is not supported in Phase 1. "
                    "Please convert the PDF to Markdown or plain text first, "
                    "then re-run with source_type='markdown' or 'text'."
                ),
            })

        # --- read content ---------------------------------------------------
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps({
                "success": False,
                "error": "File could not be decoded as UTF-8.  Please re-save as UTF-8.",
            })

        if not raw_text.strip():
            return json.dumps({
                "success": False,
                "error": "File is empty.",
            })

        # --- chunk ----------------------------------------------------------
        source_id = uuid.uuid4().hex[:12]
        chunks = _chunk_text(raw_text, chunk_size=chunk_size, overlap=CHUNK_OVERLAP)

        # --- persist to evidence store --------------------------------------
        store = EvidenceStore()
        warnings: list[str] = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"chunk_{i:04d}"
            location = f"chunk_{i:04d}"
            store.add_chunk(
                source_id=source_id,
                chunk_id=chunk_id,
                text=chunk_text,
                source_type=source_type,
                location=location,
            )

        if len(chunks) > 100:
            warnings.append(
                f"Document produced {len(chunks)} chunks. "
                "Consider reviewing chunk relevance before extraction."
            )

        return json.dumps({
            "success": True,
            "source_id": source_id,
            "source_type": source_type,
            "source_path": str(path),
            "chunks_created": len(chunks),
            "chunk_size_target": chunk_size,
            "warnings": warnings,
        }, indent=2)


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks respecting paragraph boundaries.

    Strategy:
    1. Split on double-newlines (paragraph boundaries).
    2. Accumulate paragraphs until reaching *chunk_size*.
    3. Start the next chunk with the last *overlap* characters of the
       previous chunk.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        if current_len + para_len > chunk_size and current:
            # Flush current chunk.
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # Start new chunk with overlap from the tail of the previous.
            tail = chunk_text[-overlap:] if len(chunk_text) > overlap else ""
            current = [tail, para] if tail else [para]
            current_len = len(tail) + para_len if tail else para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks
