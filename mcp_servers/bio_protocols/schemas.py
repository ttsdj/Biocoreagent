"""Pydantic schemas for Protocol MCP tools.

Reference: SciCoreAgent 需求文档, Section 4.2.1.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProtocolSearchResult(BaseModel):
    """A single result from a protocol search."""

    external_id: str
    title: str = ""
    url: str = ""
    authors: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    access: str = "unknown"  # "public" | "restricted" | "unknown"


class ProtocolsIOSearchOutput(BaseModel):
    """Output for protocols_io_search tool."""

    results: list[ProtocolSearchResult] = Field(default_factory=list)
    searched_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ProtocolFetchResult(BaseModel):
    """Full protocol record from protocols.io."""

    external_id: str
    title: str = ""
    url: str = ""
    authors: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    materials: list[str] = Field(default_factory=list)
    steps: list[dict] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    access: str = "unknown"


class ProtocolsIOFetchOutput(BaseModel):
    """Output for protocols_io_fetch tool."""

    protocol: Optional[ProtocolFetchResult] = None
    fetch_status: str = "success"  # "success" | "failed" | "skipped"
    warnings: list[str] = Field(default_factory=list)


class ProtocolCollectOutput(BaseModel):
    """Output for protocols_io_collect_cell_biology tool."""

    success: bool = True
    target_count: int = 100
    searched_count: int = 0
    fetched_count: int = 0
    ingested_count: int = 0
    duplicate_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    rag_path: str = ""
    report_path: str = ""
    warnings: list[str] = Field(default_factory=list)


class ProtocolQueryRagOutput(BaseModel):
    """Output for protocol_query_rag tool."""

    results: list[dict] = Field(default_factory=list)
    count: int = 0
