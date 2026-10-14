"""PubMed Literature MCP Server.

Implements MCP tools for PubMed literature search, fetch, and
RNA-seq method extraction.

Runs as an MCP stdio server using JSON-RPC 2.0 over stdin/stdout.
Also supports direct CLI invocation for testing.

Reference: SciCoreAgent 需求文档, Section 4.1.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .schemas import (
    PubMedSearchInput,
    PubMedSearchOutput,
    PubMedFetchDetailsInput,
    PubMedFetchDetailsOutput,
    PubMedArticle,
    ExtractRNASeqMethodsInput,
    ExtractRNASeqMethodsOutput,
    RNASeqMethodRow,
    SaveToRagInput,
    SaveToRagOutput,
    LiteratureReviewInput,
    LiteratureReviewOutput,
)

logger = logging.getLogger(__name__)

# NCBI E-utilities base URL
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Rate limiting: 3 requests/second without API key, 10 requests/second with key
RATE_NO_KEY = 3
RATE_WITH_KEY = 10

# Tool definitions for MCP
TOOLS = [
    {
        "name": "pubmed_search",
        "title": "PubMed Search",
        "description": "Search PubMed for articles matching a query. Returns PMIDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search query string"},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 50},
                "date_from": {"type": "string", "description": "Start date YYYY-MM-DD (optional)"},
                "date_to": {"type": "string", "description": "End date YYYY-MM-DD (optional)"},
                "sort": {"type": "string", "description": "Sort order: relevance or pub_date", "default": "relevance"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "pubmed_fetch_details",
        "title": "PubMed Fetch Details",
        "description": "Fetch detailed article metadata (title, abstract, authors, journal, etc.) for given PMIDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pmids": {"type": "array", "items": {"type": "string"}, "description": "List of PMIDs to fetch"},
                "include_abstract": {"type": "boolean", "description": "Whether to include abstracts", "default": True},
            },
            "required": ["pmids"],
        },
    },
    {
        "name": "pubmed_extract_rnaseq_methods",
        "title": "Extract RNA-seq Methods",
        "description": "Extract RNA-seq methods, software, and parameters from PubMed article abstracts and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "articles": {"type": "array", "description": "List of article dicts with pmid, title, abstract, authors, etc."},
            },
            "required": ["articles"],
        },
    },
    {
        "name": "pubmed_save_to_rag",
        "title": "Save to Literature RAG",
        "description": "Save articles and review rows to the literature RAG store (SQLite + JSONL).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "articles": {"type": "array", "description": "List of article dicts"},
                "rows": {"type": "array", "description": "List of review row dicts"},
                "collection_name": {"type": "string", "description": "Collection label", "default": "literature_review"},
            },
            "required": ["articles", "rows", "collection_name"],
        },
    },
    {
        "name": "pubmed_literature_review",
        "title": "PubMed Literature Review",
        "description": "One-stop tool: search PubMed, fetch articles, extract RNA-seq methods, generate Word + Markdown reports.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Review topic, e.g. RNA-seq workflow"},
                "query": {"type": "string", "description": "PubMed search query"},
                "max_results": {"type": "integer", "description": "Max results", "default": 50},
                "output_docx": {"type": "string", "description": "Output .docx path"},
            },
            "required": ["query"],
        },
    },
]


class PubMedServer:
    """PubMed literature MCP server.

    Handles all PubMed API communication, result parsing, and tool execution.
    """

    def __init__(self):
        self._email = os.environ.get("NCBI_EMAIL", "")
        self._api_key = os.environ.get("NCBI_API_KEY", "")
        self._tool_name = os.environ.get("NCBI_TOOL", "SciCoreAgent")
        self._last_request_time = 0.0
        self._min_interval = 1.0 / (RATE_WITH_KEY if self._api_key else RATE_NO_KEY)

    @property
    def ready(self) -> bool:
        """Check if the server can make PubMed API calls."""
        return bool(self._email)

    def status(self) -> dict:
        """Return server configuration status."""
        return {
            "email_configured": bool(self._email),
            "api_key_configured": bool(self._api_key),
            "rate_limit": RATE_WITH_KEY if self._api_key else RATE_NO_KEY,
            "tool_name": self._tool_name,
        }

    # ------------------------------------------------------------------
    # Tool: pubmed_search
    # ------------------------------------------------------------------

    def pubmed_search(self, query: str, max_results: int = 50,
                      date_from: Optional[str] = None,
                      date_to: Optional[str] = None,
                      sort: str = "relevance") -> dict:
        """Search PubMed and return PMIDs."""
        warnings = []
        pmids = []

        if not self._email:
            return {
                "query": query, "pmids": [], "count": 0,
                "api_status": "failed",
                "warnings": ["NCBI_EMAIL not configured. Set NCBI_EMAIL environment variable."],
            }

        # Build query with date range
        full_query = query
        if date_from or date_to:
            if date_from:
                full_query += f" AND (\"{date_from}\"[Date - Publication] : "
            if date_to:
                full_query += f"\"{date_to}\"[Date - Publication])"
            elif date_from:
                full_query += "\"3000\"[Date - Publication])"

        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": sort,
            "tool": self._tool_name,
            "email": self._email,
        }

        try:
            xml_text = self._eutils_get("esearch.fcgi", params)
            root = ET.fromstring(xml_text)

            count_elem = root.find(".//Count")
            count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

            for id_elem in root.findall(".//Id"):
                if id_elem.text:
                    pmids.append(id_elem.text)

        except Exception as e:
            logger.error("PubMed search failed: %s", e)
            warnings.append(f"Search failed: {e}")
            return {
                "query": query, "pmids": pmids, "count": 0,
                "api_status": "failed", "warnings": warnings,
            }

        if not pmids:
            warnings.append("No PMIDs found for the given query.")

        return {
            "query": query, "pmids": pmids, "count": len(pmids),
            "api_status": "success", "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Tool: pubmed_fetch_details
    # ------------------------------------------------------------------

    def pubmed_fetch_details(self, pmids: list[str],
                              include_abstract: bool = True) -> dict:
        """Fetch detailed metadata for given PMIDs from PubMed."""
        warnings = []
        articles = []
        failed_pmids = []

        if not pmids:
            return {
                "articles": [], "fetched_count": 0,
                "failed_pmids": [], "warnings": ["No PMIDs provided."],
            }

        if not self._email:
            return {
                "articles": [], "fetched_count": 0,
                "failed_pmids": pmids,
                "warnings": ["NCBI_EMAIL not configured."],
            }

        # Fetch in batches of 100 (NCBI limit)
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            try:
                batch_articles, batch_failed = self._fetch_batch(batch, include_abstract)
                articles.extend(batch_articles)
                failed_pmids.extend(batch_failed)
            except Exception as e:
                logger.error("Failed to fetch batch: %s", e)
                failed_pmids.extend(batch)
                warnings.append(f"Batch fetch failed: {e}")

        return {
            "articles": articles,
            "fetched_count": len(articles),
            "failed_pmids": failed_pmids,
            "warnings": warnings,
        }

    def _fetch_batch(self, pmids: list[str], include_abstract: bool) -> tuple[list[dict], list[str]]:
        """Fetch one batch of PMIDs."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": self._tool_name,
            "email": self._email,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        xml_text = self._eutils_get("efetch.fcgi", params)
        root = ET.fromstring(xml_text)

        articles = []
        failed = list(pmids)  # Start with all, remove successes

        for article_elem in root.findall(".//PubmedArticle"):
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if pmid and pmid in failed:
                failed.remove(pmid)

            article = self._parse_pubmed_article(article_elem, include_abstract)
            articles.append(article)

        return articles, failed

    def _parse_pubmed_article(self, elem, include_abstract: bool) -> dict:
        """Parse a PubmedArticle XML element into a dict."""
        medline = elem.find(".//MedlineCitation")
        article_elem = elem.find(".//Article")

        pmid = ""
        pmid_elem = medline.find(".//PMID") if medline is not None else None
        if pmid_elem is not None and pmid_elem.text:
            pmid = pmid_elem.text

        # Title
        title = None
        if article_elem is not None:
            title_elem = article_elem.find(".//ArticleTitle")
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

        # Journal
        journal = None
        if article_elem is not None:
            journal_elem = article_elem.find(".//Journal/Title")
            if journal_elem is not None and journal_elem.text:
                journal = journal_elem.text.strip()

        # Year
        year = None
        if article_elem is not None:
            year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass

        # Abstract
        abstract = None
        if include_abstract and article_elem is not None:
            abstract_parts = []
            for abs_elem in article_elem.findall(".//AbstractText"):
                label = abs_elem.get("Label", "")
                text = abs_elem.text or ""
                prefix = f"{label}: " if label else ""
                abstract_parts.append(f"{prefix}{text}")
            if abstract_parts:
                abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        last_author = None
        corresponding_author = None
        if article_elem is not None:
            author_list = article_elem.findall(".//Author")
            for auth in author_list:
                last_name = auth.findtext("LastName", "")
                fore_name = auth.findtext("ForeName", "")
                name = f"{last_name} {fore_name}".strip()
                if name:
                    authors.append(name)
            if authors:
                last_author = authors[-1]
            # Corresponding author is NOT extracted from PubMed XML by default.
            # We leave it as None unless explicitly found in metadata.

        # DOI
        doi = None
        if article_elem is not None:
            for eid in article_elem.findall(".//ELocationID"):
                if eid.get("EIdType") == "doi" and eid.text:
                    doi = eid.text.strip()

        # MeSH terms
        mesh_terms = []
        if medline is not None:
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text.strip())

        # Publication types
        pub_types = []
        if article_elem is not None:
            for pt in article_elem.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())

        return {
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "authors": authors,
            "last_author": last_author,
            "corresponding_author": corresponding_author,  # Always None unless found
            "mesh_terms": mesh_terms,
            "publication_types": pub_types,
        }

    # ------------------------------------------------------------------
    # Tool: pubmed_extract_rnaseq_methods
    # ------------------------------------------------------------------

    def pubmed_extract_rnaseq_methods(self, articles: list[dict]) -> dict:
        """Extract RNA-seq methods from article metadata using regex patterns.

        This is a deterministic extraction based on known RNA-seq tool names
        and patterns. It does NOT use an LLM.
        """
        rows = []

        # Known RNA-seq software/tools for regex matching
        TOOL_PATTERNS = {
            "STAR": r"\bSTAR\b(?!\s+Protocol)",
            "HISAT2": r"\bHISAT2\b",
            "Salmon": r"\b[Ss]almon\b",
            "kallisto": r"\bkallisto\b",
            "DESeq2": r"\bDESeq2\b",
            "edgeR": r"\bedgeR\b",
            "limma": r"\blimma\b",
            "voom": r"\bvoom\b",
            "featureCounts": r"\bfeatureCounts\b",
            "RSEM": r"\bRSEM\b",
            "Trimmomatic": r"\bTrimmomatic\b",
            "Cutadapt": r"\b[Cc]utadapt\b",
            "fastp": r"\bfastp\b",
            "FastQC": r"\bFastQC\b",
            "bowtie2": r"\bbowtie2\b",
            "BWA": r"\bBWA\b",
            "GATK": r"\bGATK\b",
            "Picard": r"\bPicard\b",
            "Samtools": r"\b[Ss]amtools\b",
            "StringTie": r"\bStringTie\b",
            "Cufflinks": r"\bCufflinks\b",
            "HTSeq": r"\bHTSeq\b",
            "TopHat": r"\bTopHat\b",
            "clusterProfiler": r"\bclusterProfiler\b",
            "GSEA": r"\bGSEA\b",
            "GOseq": r"\bGOseq\b",
            "ChIPseeker": r"\bChIPseeker\b",
            "MACS2": r"\bMACS2\b",
            "ggplot2": r"\bggplot2\b",
            "Seurat": r"\bSeurat\b",
            "Scanpy": r"\bScanpy\b",
        }

        METHOD_PATTERNS = {
            "Quality Control": r"\bquality\s*control\b|\bQC\b|\bfastq\s*quality\b",
            "Read Alignment": r"\b(?:read\s*)?align(?:ment)?\b|\bmapping\b|\breference\s*genome\b",
            "Pseudoalignment": r"\bpseudoalign(?:ment)?\b|\btranscriptome\s*index\b",
            "Quantification": r"\bquantif(?:y|ication)\b|\bgene\s*counts?\b|\btranscript\s*abundance\b",
            "Differential Expression": r"\bdifferential\s*expression\b|\bDE\b|\blog\s*fold\s*change\b",
            "Functional Enrichment": r"\b(?:functional\s*)?enrichment\b|\bGO\s*(?:term|analysis)\b|\bKEGG\b|\bpathway\b",
            "Normalization": r"\bnormaliz(?:e|ation)\b|\bTPM\b|\bFPKM\b|\bRPKM\b|\bCPM\b",
            "Variant Calling": r"\bvariant\s*call(?:ing)?\b|\bSNP\b|\bINDEL\b",
            "Peak Calling": r"\bpeak\s*call(?:ing)?\b|\bChIP.seq\b|\bATAC.seq\b",
            "Clustering": r"\bclustering\b|\bhierarchical\b|\bk.means\b|\bt.SNE\b|\bUMAP\b",
            "Visualization": r"\bvisualiz(?:e|ation)\b|\bheatmap\b|\bvolcano\s*plot\b|\bPCA\b|\bMA\s*plot\b",
        }

        for article in articles:
            pmid = article.get("pmid", "")
            title = article.get("title", "") or ""
            abstract = article.get("abstract", "") or ""
            journal = article.get("journal", "") or ""
            year = article.get("year")
            doi = article.get("doi")
            authors = article.get("authors", [])
            last_author = article.get("last_author")

            text_to_search = f"{title} {abstract}"
            if not text_to_search.strip():
                continue

            # Find methods mentioned
            found_methods = {}
            for method, pattern in METHOD_PATTERNS.items():
                match = re.search(pattern, text_to_search, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 50)
                    end = min(len(text_to_search), match.end() + 50)
                    context = text_to_search[start:end].strip()
                    found_methods[method] = context

            # Find software mentioned
            found_software = {}
            for tool, pattern in TOOL_PATTERNS.items():
                match = re.search(pattern, text_to_search)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text_to_search), match.end() + 30)
                    context = text_to_search[start:end].strip()
                    found_software[tool] = context

            # Create rows for each method found
            if found_methods or found_software:
                for method, evidence in found_methods.items():
                    software_for_method = [
                        s for s, ctx in found_software.items()
                        if s.lower() in evidence.lower()
                        or any(s.lower() in mctx.lower() for mctx in found_software.values())
                    ]
                    row = {
                        "pmid": pmid,
                        "method_name": method,
                        "software_name": ", ".join(software_for_method) if software_for_method else None,
                        "parameters": None,  # Cannot reliably extract from regex
                        "lab_info": self._format_lab_info(authors, last_author, article),
                        "paper_title": title,
                        "journal": journal,
                        "doi": doi,
                        "year": year,
                        "evidence_text": evidence[:500] if evidence else None,
                        "missing_fields": ["parameters"] if not software_for_method else (
                            ["parameters", "software_version"] if software_for_method else ["parameters"]
                        ),
                    }
                    rows.append(row)

                # Also create rows for software not matched to a method
                method_software_set = set()
                for row in rows:
                    if row.get("software_name"):
                        for s in row["software_name"].split(", "):
                            method_software_set.add(s)

                for tool, evidence in found_software.items():
                    if tool not in method_software_set:
                        rows.append({
                            "pmid": pmid,
                            "method_name": f"Software: {tool}",
                            "software_name": tool,
                            "parameters": None,
                            "lab_info": self._format_lab_info(authors, last_author, article),
                            "paper_title": title,
                            "journal": journal,
                            "doi": doi,
                            "year": year,
                            "evidence_text": evidence[:500] if evidence else None,
                            "missing_fields": ["parameters", "method_context", "software_version"],
                        })

        # If no methods found, still create a row noting this
        if not rows:
            for article in articles:
                rows.append({
                    "pmid": article.get("pmid", ""),
                    "method_name": None,
                    "software_name": None,
                    "parameters": None,
                    "lab_info": self._format_lab_info(
                        article.get("authors", []),
                        article.get("last_author"),
                        article,
                    ),
                    "paper_title": article.get("title"),
                    "journal": article.get("journal"),
                    "doi": article.get("doi"),
                    "year": article.get("year"),
                    "evidence_text": None,
                    "missing_fields": [
                        "method_name", "software_name", "parameters",
                        "未在 PubMed metadata / abstract 中明确提供",
                    ],
                })

        return {"rows": rows}

    @staticmethod
    def _format_lab_info(authors: list[str], last_author: Optional[str],
                          article: dict) -> Optional[str]:
        """Format lab/group information from author metadata.

        Rules:
        - If corresponding_author is explicitly provided → "通讯作者: xxx"
        - Otherwise → "末尾作者候选: xxx"
        - Never write "通讯作者" without explicit metadata confirmation.
        """
        corr = article.get("corresponding_author")
        if corr:
            return f"通讯作者: {corr}"
        if last_author:
            return f"末尾作者候选: {last_author}"
        return None

    # ------------------------------------------------------------------
    # Tool: pubmed_save_to_rag
    # ------------------------------------------------------------------

    def pubmed_save_to_rag(self, articles: list[dict], rows: list[dict],
                            collection_name: str = "literature_review") -> dict:
        """Save articles and rows to the literature RAG store."""
        try:
            from scicore.agent.bio.rag_store import default_rag_store
            from scicore.agent.bio.literature_schemas import LiteratureArticle, LiteratureReviewRow

            store = default_rag_store()

            articles_written = 0
            for art in articles:
                try:
                    article = LiteratureArticle(**art)
                    if store.add_literature_article(article):
                        articles_written += 1
                except Exception as e:
                    logger.warning("Failed to save article %s: %s", art.get("pmid", "?"), e)

            rows_written = 0
            for row in rows:
                try:
                    review_row = LiteratureReviewRow(**row)
                    if store.add_literature_review_row(review_row):
                        rows_written += 1
                except Exception as e:
                    logger.warning("Failed to save row for PMID %s: %s", row.get("pmid", "?"), e)

            return {
                "success": True,
                "articles_written": articles_written,
                "rows_written": rows_written,
                "rag_path": str(store._rag_dir),
            }
        except Exception as e:
            logger.error("Failed to save to RAG: %s", e)
            return {
                "success": False,
                "articles_written": 0,
                "rows_written": 0,
                "rag_path": "",
            }

    # ------------------------------------------------------------------
    # Tool: pubmed_literature_review (one-stop)
    # ------------------------------------------------------------------

    def pubmed_literature_review(self, topic: str = "RNA-seq workflow",
                                   query: str = "RNA-seq workflow bioinformatics pipeline differential expression",
                                   max_results: int = 50,
                                   output_docx: str = "reports/rnaseq_literature_review.docx") -> dict:
        """Run a complete literature review pipeline."""
        review_id = uuid.uuid4().hex[:12]
        warnings = []

        # Step 1: Search
        search_result = self.pubmed_search(query=query, max_results=max_results)
        pmids = search_result.get("pmids", [])
        warnings.extend(search_result.get("warnings", []))

        # Step 2: Fetch details
        articles = []
        if pmids:
            fetch_result = self.pubmed_fetch_details(pmids=pmids)
            articles = fetch_result.get("articles", [])
            warnings.extend(fetch_result.get("warnings", []))

        # Step 3: Extract methods
        rows = []
        if articles:
            extract_result = self.pubmed_extract_rnaseq_methods(articles=articles)
            rows = extract_result.get("rows", [])

        # Step 4: Save to RAG
        rag_result = self.pubmed_save_to_rag(
            articles=articles,
            rows=rows,
            collection_name=f"review_{review_id}",
        )

        # Step 5: Generate reports
        docx_path = None
        markdown_path = None
        try:
            from scicore.agent.bio.literature_schemas import (
                LiteratureReviewReport, LiteratureReviewRow,
                MethodSummaryRow,
            )
            from scicore.agent.bio.report_writer import (
                generate_literature_review_docx,
                generate_literature_review_markdown,
            )

            # Build review report
            methods_set = set(r.get("method_name") for r in rows if r.get("method_name"))
            missing_count = sum(len(r.get("missing_fields", [])) for r in rows)

            review = LiteratureReviewReport(
                review_id=review_id,
                topic=topic,
                query=query,
                max_results=max_results,
                started_at=datetime.now(timezone.utc).isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                pmids_found=len(pmids),
                articles_fetched=len(articles),
                rows_extracted=len(rows),
                methods_identified=len(methods_set),
                docx_path=output_docx,
                markdown_path=output_docx.replace(".docx", ".md"),
                missing_parameter_count=missing_count,
                warnings=warnings,
            )

            # Build method summaries
            from collections import Counter
            method_counts = Counter(r.get("method_name") for r in rows if r.get("method_name"))
            method_summaries = []
            for method, count in method_counts.most_common():
                method_rows = [r for r in rows if r.get("method_name") == method]
                supporting_pmids = set(r.get("pmid") for r in method_rows)
                all_missing = set()
                for r in method_rows:
                    all_missing.update(r.get("missing_fields", []))

                method_summaries.append(MethodSummaryRow(
                    method_name=method,
                    frequency_in_current_review=count,
                    supporting_article_count=len(supporting_pmids),
                    confidence="high" if len(supporting_pmids) >= 3 else (
                        "medium" if len(supporting_pmids) >= 2 else "low"
                    ),
                    difficulty="low" if not all_missing else (
                        "medium" if len(all_missing) <= 2 else "high"
                    ),
                    main_evidence=method_rows[0].get("evidence_text") if method_rows else None,
                    main_missing_information=", ".join(all_missing) if all_missing else None,
                ))

            # Convert dict rows to LiteratureReviewRow objects
            row_objects = [LiteratureReviewRow(**r) for r in rows]
            article_objects = []
            from scicore.agent.bio.literature_schemas import LiteratureArticle
            for art in articles:
                try:
                    article_objects.append(LiteratureArticle(**art))
                except Exception:
                    pass

            # Generate files
            docx_path = str(generate_literature_review_docx(
                review, article_objects, row_objects, method_summaries, output_docx
            ))
            markdown_path = output_docx.replace(".docx", ".md")
            generate_literature_review_markdown(review, row_objects, method_summaries, markdown_path)

            # Run report
            run_report = {
                "run_id": review_id,
                "query": query,
                "started_at": review.started_at,
                "ended_at": review.ended_at,
                "pmids_found": len(pmids),
                "articles_fetched": len(articles),
                "rows_extracted": len(rows),
                "rag_records_written": rag_result.get("articles_written", 0) + rag_result.get("rows_written", 0),
                "docx_path": docx_path,
                "markdown_path": markdown_path,
                "failed_items": [],
                "warnings": warnings,
            }
            run_report_path = output_docx.replace(".docx", "_run_report.json")
            Path(run_report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(run_report_path).write_text(
                json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        except ImportError as e:
            warnings.append(f"Report generation skipped: {e}")
        except Exception as e:
            logger.error("Failed to generate reports: %s", e)
            warnings.append(f"Report generation failed: {e}")

        return {
            "success": True,
            "review_id": review_id,
            "pmids_found": len(pmids),
            "articles_fetched": len(articles),
            "rows_extracted": len(rows),
            "docx_path": docx_path or "",
            "markdown_path": markdown_path or "",
            "run_report_path": output_docx.replace(".docx", "_run_report.json"),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call and return JSON result."""
        try:
            if tool_name == "pubmed_search":
                result = self.pubmed_search(**arguments)
            elif tool_name == "pubmed_fetch_details":
                result = self.pubmed_fetch_details(**arguments)
            elif tool_name == "pubmed_extract_rnaseq_methods":
                result = self.pubmed_extract_rnaseq_methods(**arguments)
            elif tool_name == "pubmed_save_to_rag":
                result = self.pubmed_save_to_rag(**arguments)
            elif tool_name == "pubmed_literature_review":
                result = self.pubmed_literature_review(**arguments)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return json.dumps({"error": f"Tool {tool_name} failed: {e}"})

    def get_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return TOOLS

    # ------------------------------------------------------------------
    # NCBI E-utilities HTTP helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce rate limiting between NCBI API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _eutils_get(self, endpoint: str, params: dict) -> str:
        """Make a GET request to NCBI E-utilities with rate limiting."""
        self._rate_limit()
        url = f"{NCBI_EUTILS_BASE}/{endpoint}?{urlencode(params)}"
        req = Request(url)
        req.add_header("User-Agent", f"SciCoreAgent/1.0 (mailto:{self._email})")
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as e:
            raise RuntimeError(f"NCBI API HTTP {e.code}: {e.reason}")
        except URLError as e:
            raise RuntimeError(f"NCBI API connection error: {e.reason}")


# ---------------------------------------------------------------------------
# MCP stdio loop
# ---------------------------------------------------------------------------


def run_stdio():
    """Run the PubMed MCP server via JSON-RPC 2.0 over stdin/stdout."""
    server = PubMedServer()
    logger.info("PubMed MCP server started (stdio)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {
                        "name": "pubmed_literature",
                        "version": "1.0.0",
                    },
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": server.get_tools()},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result_text = server.call_tool(tool_name, tool_args)
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }
        elif method == "ping":
            response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method.startswith("notifications/"):
            continue  # No response for notifications
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="PubMed Literature MCP Server")
    p.add_argument("--stdio", action="store_true", help="Run in MCP stdio mode")
    p.add_argument("--test-search", help="Test PubMed search with a query")
    p.add_argument("--status", action="store_true", help="Show server status")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.stdio:
        run_stdio()
    elif args.test_search:
        server = PubMedServer()
        result = server.pubmed_search(query=args.test_search, max_results=5)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.status:
        server = PubMedServer()
        print(json.dumps(server.status(), ensure_ascii=False, indent=2))
    else:
        p.print_help()
