"""Pydantic schemas for PubMed Literature MCP tools.

Defines input/output validation for all MCP tools exposed by the
pubmed_literature MCP server.

Reference: SciCoreAgent 需求文档, Section 4.1.1.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# PubMed search
# ---------------------------------------------------------------------------


class PubMedSearchInput(BaseModel):
    """Input for pubmed_search tool."""

    query: str
    max_results: int = Field(default=50, ge=1, le=500)
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None  # YYYY-MM-DD
    sort: str = Field(default="relevance", pattern="^(relevance|pub_date)$")


class PubMedSearchOutput(BaseModel):
    """Output for pubmed_search tool."""

    query: str
    pmids: list[str] = Field(default_factory=list)
    count: int = 0
    api_status: str = "success"  # "success" | "failed"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PubMed fetch details
# ---------------------------------------------------------------------------


class PubMedFetchDetailsInput(BaseModel):
    """Input for pubmed_fetch_details tool."""

    pmids: list[str]
    include_abstract: bool = True


class PubMedArticle(BaseModel):
    """A single article fetched from PubMed."""

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


class PubMedFetchDetailsOutput(BaseModel):
    """Output for pubmed_fetch_details tool."""

    articles: list[PubMedArticle] = Field(default_factory=list)
    fetched_count: int = 0
    failed_pmids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extract RNA-seq methods
# ---------------------------------------------------------------------------


class ExtractRNASeqMethodsInput(BaseModel):
    """Input for pubmed_extract_rnaseq_methods tool."""

    articles: list[dict] = Field(default_factory=list)


class RNASeqMethodRow(BaseModel):
    """One extracted method/software/parameter row."""

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


class ExtractRNASeqMethodsOutput(BaseModel):
    """Output for pubmed_extract_rnaseq_methods tool."""

    rows: list[RNASeqMethodRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Save to RAG
# ---------------------------------------------------------------------------


class SaveToRagInput(BaseModel):
    """Input for pubmed_save_to_rag tool."""

    articles: list[dict] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    collection_name: str = "literature_review"


class SaveToRagOutput(BaseModel):
    """Output for pubmed_save_to_rag tool."""

    success: bool = True
    articles_written: int = 0
    rows_written: int = 0
    rag_path: str = ""


# ---------------------------------------------------------------------------
# Literature review (one-stop)
# ---------------------------------------------------------------------------


class LiteratureReviewInput(BaseModel):
    """Input for pubmed_literature_review tool."""

    topic: str = "RNA-seq workflow"
    query: str = "RNA-seq workflow bioinformatics pipeline differential expression"
    max_results: int = Field(default=50, ge=1, le=500)
    output_docx: str = "reports/rnaseq_literature_review.docx"


class LiteratureReviewOutput(BaseModel):
    """Output for pubmed_literature_review tool."""

    success: bool = True
    review_id: str = ""
    pmids_found: int = 0
    articles_fetched: int = 0
    rows_extracted: int = 0
    docx_path: str = ""
    markdown_path: str = ""
    run_report_path: str = ""
    warnings: list[str] = Field(default_factory=list)
