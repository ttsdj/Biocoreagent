"""Protocol collection data schemas for Protocol MCP + RAG.

Defines the structured types used by the Protocol MCP server
and the protocol collection CLI commands.

Reference: SciCoreAgent 需求文档, Section 5.2 & 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ProtocolEvidence(BaseModel):
    """Evidence span for a specific field in a protocol record."""

    field_name: str
    source_text: str = ""
    source_location: str = ""


class ProtocolStep(BaseModel):
    """A single step within a protocol."""

    step_number: Optional[int] = None
    description: str = ""
    expected_duration: Optional[str] = None
    notes: Optional[str] = None


class BioProtocolRecord(BaseModel):
    """A single protocol record ingested from an external source.

    Fields not present in the source MUST be left as None / empty.
    """

    protocol_id: str
    source: str = "protocols_io"  # "protocols_io" | "bio_protocol" | "manual" | "other"
    external_id: str = ""
    title: str = ""
    url: str = ""
    domain: str = ""
    subdomain: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    last_author: Optional[str] = None
    summary: Optional[str] = None
    materials: list[str] = Field(default_factory=list)
    reagents: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    software: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    steps: list[ProtocolStep] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    access: str = "unknown"  # "public" | "restricted" | "unknown"
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    missing_information: list[str] = Field(default_factory=list)
    evidence: list[ProtocolEvidence] = Field(default_factory=list)


class ProtocolCollectionRunReport(BaseModel):
    """Complete run report for a protocol collection execution."""

    model_config = {"extra": "allow"}

    run_id: str
    source: str = "protocols_io"
    domain: str = ""
    target_count: int = 100
    searched_count: int = 0
    fetched_count: int = 0
    ingested_count: int = 0
    duplicate_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    rag_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    report_path: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    failed_items: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    field_completeness: dict = Field(default_factory=dict)
    search_keywords: list[str] = Field(default_factory=list)
    reason_not_full: list[str] = Field(default_factory=list)
