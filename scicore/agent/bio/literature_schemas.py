"""Literature review data schemas for PubMed literature RAG.

Defines the structured types used by the PubMed literature MCP server
and the literature review CLI commands. Every field follows the rule:
"原文未出现则不填，禁止凭空猜测" — if the source document does not
explicitly state a value, leave it empty/None rather than guessing.

Reference: SciCoreAgent 需求文档, Section 5.1 & 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class LiteratureArticle(BaseModel):
    """A single PubMed article with its metadata.

    Fields that are not available from PubMed metadata MUST be left
    as None / empty, rather than guessed.
    """

    pmid: str
    doi: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    last_author: Optional[str] = None
    corresponding_author: Optional[str] = None
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LiteratureReviewRow(BaseModel):
    """One extracted method/software/parameter row from a literature article.

    Each field must be traceable to evidence_text from the source.
    If a field cannot be found in the source, it MUST be None / empty.
    """

    pmid: str
    method_name: Optional[str] = None
    software_name: Optional[str] = None
    parameters: Optional[str] = None
    lab_info: Optional[str] = None
    paper_title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    evidence_text: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)


class MethodSummaryRow(BaseModel):
    """Summary statistics for one method across the current review.

    Includes frequency, confidence, difficulty, and main evidence/limitations.
    """

    method_name: str
    frequency_in_current_review: int = 0
    supporting_article_count: int = 0
    confidence: str = "medium"  # "high" | "medium" | "low"
    difficulty: str = "medium"  # "low" | "medium" | "high"
    main_evidence: Optional[str] = None
    main_missing_information: Optional[str] = None


class LiteratureReviewReport(BaseModel):
    """High-level report of a literature review run."""

    review_id: str
    topic: str = ""
    query: str = ""
    max_results: int = 50
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    pmids_found: int = 0
    articles_fetched: int = 0
    rows_extracted: int = 0
    methods_identified: int = 0
    docx_path: Optional[str] = None
    markdown_path: Optional[str] = None
    missing_parameter_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class LiteratureRunReport(BaseModel):
    """Complete run report for a literature review execution.

    Includes the review summary plus detailed failure tracking.
    """

    model_config = {"extra": "allow"}

    run_id: str
    query: str = ""
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    pmids_found: int = 0
    articles_fetched: int = 0
    rows_extracted: int = 0
    rag_records_written: int = 0
    docx_path: Optional[str] = None
    markdown_path: Optional[str] = None
    failed_items: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
