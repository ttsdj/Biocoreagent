"""Word report generation for SciCoreAgent.

Uses python-docx to generate .docx reports from literature reviews
and protocol collection runs.

Reference: SciCoreAgent 需求文档, Section 7, 8, 9, 10.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .literature_schemas import (
    LiteratureArticle,
    LiteratureReviewRow,
    LiteratureReviewReport,
    LiteratureRunReport,
    MethodSummaryRow,
)
from .protocol_schemas import BioProtocolRecord, ProtocolCollectionRunReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Literature review Word report
# ---------------------------------------------------------------------------


def generate_literature_review_docx(
    review: LiteratureReviewReport,
    articles: list[LiteratureArticle],
    rows: list[LiteratureReviewRow],
    method_summaries: list[MethodSummaryRow],
    output_path: str | Path,
) -> Path:
    """Generate a Word (.docx) report for an RNA-seq literature review.

    The report includes:
      1. Title and metadata
      2. Overview statistics
      3. Detailed table (method/software/parameters/evidence per article)
      4. Summary table (method frequency, confidence, difficulty)
      5. Replication workflow (evidence-supported vs. general)
      6. Missing information section
      7. Appendix: PMID/DOI/evidence_text
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, Cm, RGBColor
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
    except ImportError:
        raise ImportError(
            "python-docx is required for Word report generation. "
            "Install it with: pip install python-docx"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # -- Style setup --
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(10)

    # -- Title --
    title = doc.add_heading("RNA-seq Literature Review Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f"Topic: {review.topic}")
    doc.add_paragraph(f"PubMed Query: {review.query}")
    doc.add_paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    doc.add_paragraph(f"Review ID: {review.review_id}")

    # -- Section 1: Overview --
    doc.add_heading("1. Overview", level=1)
    _add_kv_table(doc, [
        ("Max results requested", str(review.max_results)),
        ("PMIDs found", str(review.pmids_found)),
        ("Articles fetched", str(review.articles_fetched)),
        ("Method rows extracted", str(review.rows_extracted)),
        ("Methods identified", str(review.methods_identified)),
        ("Missing parameter count", str(review.missing_parameter_count)),
    ])

    if review.warnings:
        doc.add_heading("Warnings", level=2)
        for w in review.warnings:
            doc.add_paragraph(w, style="List Bullet")

    # -- Section 2: Detailed Table --
    doc.add_heading("2. Detailed Method Extraction Table", level=1)
    _add_detailed_table(doc, rows)

    # -- Section 3: Summary Table --
    doc.add_heading("3. Method Summary Table", level=1)
    _add_summary_table(doc, method_summaries)

    # -- Section 4: Replication Workflow --
    doc.add_heading("4. RNA-seq Replication Workflow", level=1)

    doc.add_heading("4.1 Literature-Evidence-Supported Steps", level=2)
    doc.add_paragraph(
        "The following steps are supported by explicit evidence from the reviewed literature. "
        "Each step references specific PMID(s) where the method/software/parameter was described."
    )
    # Group rows by method_name
    methods = {}
    for row in rows:
        mn = row.method_name or "Unnamed method"
        if mn not in methods:
            methods[mn] = []
        methods[mn].append(row)

    for method_name, method_rows in sorted(methods.items()):
        pmids = sorted(set(r.pmid for r in method_rows))
        software = sorted(set(r.software_name for r in method_rows if r.software_name))
        doc.add_heading(f"Method: {method_name}", level=3)
        doc.add_paragraph(f"Supporting PMID(s): {', '.join(pmids[:10])}")
        if software:
            doc.add_paragraph(f"Software mentioned: {', '.join(software)}")
        for r in method_rows[:3]:
            if r.evidence_text:
                doc.add_paragraph(f"Evidence: {r.evidence_text[:300]}", style="Intense Quote")

    doc.add_heading("4.2 General Bioinformatics Workflow Recommendations", level=2)
    doc.add_paragraph(
        "The following steps represent a standard RNA-seq bioinformatics workflow. "
        "These are general recommendations and NOT attributed to any specific reviewed article. "
        "They are provided for context only."
    )
    general_steps = [
        "1. Quality Control: FastQC or similar for raw FASTQ assessment",
        "2. Adapter Trimming: Trimmomatic, Cutadapt, or fastp",
        "3. Alignment: STAR, HISAT2, or Salmon (pseudoalignment)",
        "4. Quantification: featureCounts, Salmon, kallisto, or RSEM",
        "5. Differential Expression: DESeq2, edgeR, or limma-voom",
        "6. Functional Enrichment: clusterProfiler, GSEA, or GOseq",
        "7. Visualization: PCA plots, heatmaps, volcano plots, MA plots",
    ]
    for step in general_steps:
        doc.add_paragraph(step)

    doc.add_heading("4.3 Steps with Insufficient Evidence", level=2)
    missing_steps = [r for r in rows if r.missing_fields]
    if missing_steps:
        for r in missing_steps[:10]:
            doc.add_paragraph(
                f"Method '{r.method_name or 'unknown'}' (PMID: {r.pmid}): "
                f"Missing: {', '.join(r.missing_fields)}"
            )
    else:
        doc.add_paragraph("All extracted methods had sufficient evidence for the fields attempted.")

    # -- Section 5: Missing Information --
    doc.add_heading("5. Missing Information Summary", level=1)
    all_missing = []
    for r in rows:
        all_missing.extend(r.missing_fields or [])
    if all_missing:
        from collections import Counter
        missing_counts = Counter(all_missing)
        for field, count in missing_counts.most_common():
            doc.add_paragraph(f"{field}: missing in {count} row(s)")
    else:
        doc.add_paragraph("No missing fields reported.")

    # -- Section 6: Appendix --
    doc.add_heading("6. Appendix: Article Reference List", level=1)
    for article in articles[:50]:
        doc.add_paragraph(
            f"PMID: {article.pmid}  |  DOI: {article.doi or 'N/A'}\n"
            f"Title: {article.title or 'N/A'}\n"
            f"Journal: {article.journal or 'N/A'} ({article.year or 'N/A'})\n"
            f"Authors: {', '.join(article.authors[:5])}{'...' if len(article.authors) > 5 else ''}"
        )

    # -- Save --
    doc.save(str(output_path))
    logger.info("Literature review docx saved to %s", output_path)
    return output_path


def generate_protocol_collection_report_docx(
    run_report: ProtocolCollectionRunReport,
    protocols: list[BioProtocolRecord],
    output_path: str | Path,
) -> Path:
    """Generate a Word report for a protocol collection run."""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        raise ImportError("python-docx is required. Install with: pip install python-docx")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Title
    title = doc.add_heading("Protocol Collection Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f"Domain: {run_report.domain}")
    doc.add_paragraph(f"Source: {run_report.source}")
    doc.add_paragraph(f"Run ID: {run_report.run_id}")
    doc.add_paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Overview
    doc.add_heading("1. Collection Statistics", level=1)
    _add_kv_table(doc, [
        ("Target count", str(run_report.target_count)),
        ("Searched", str(run_report.searched_count)),
        ("Fetched", str(run_report.fetched_count)),
        ("Ingested", str(run_report.ingested_count)),
        ("Duplicates", str(run_report.duplicate_count)),
        ("Skipped", str(run_report.skipped_count)),
        ("Failed", str(run_report.failed_count)),
    ])

    if run_report.reason_not_full:
        doc.add_heading("Reasons for Not Reaching Target", level=2)
        for reason in run_report.reason_not_full:
            doc.add_paragraph(reason, style="List Bullet")

    # Field completeness
    doc.add_heading("2. Field Completeness", level=1)
    if run_report.field_completeness:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text = "Field"
        hdr[1].text = "Non-null Count"
        hdr[2].text = "Missing Count"
        hdr[3].text = "Completeness %"
        for field, stats in run_report.field_completeness.items():
            row = table.add_row().cells
            row[0].text = field
            row[1].text = str(stats.get("non_null", 0))
            row[2].text = str(stats.get("missing", 0))
            rate = stats.get("completeness", 0)
            row[3].text = f"{rate:.1%}" if isinstance(rate, float) else str(rate)

    # Protocol list
    doc.add_heading("3. Collected Protocols", level=1)
    if protocols:
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text = "Protocol ID"
        hdr[1].text = "Title"
        hdr[2].text = "Source"
        hdr[3].text = "Tags"
        hdr[4].text = "Missing Fields"
        for p in protocols[:50]:
            row = table.add_row().cells
            row[0].text = p.protocol_id[:20]
            row[1].text = (p.title or "")[:60]
            row[2].text = p.source
            row[3].text = ", ".join(p.tags[:5])
            row[4].text = ", ".join(p.missing_information[:5])
    else:
        doc.add_paragraph("No protocols collected.")

    # Failed items
    if run_report.failed_items:
        doc.add_heading("4. Failed / Skipped Items", level=1)
        for item in run_report.failed_items:
            doc.add_paragraph(
                f"External ID: {item.get('external_id', 'N/A')} | "
                f"Title: {item.get('title', 'N/A')} | "
                f"Reason: {item.get('reason', 'Unknown')}"
            )

    doc.save(str(output_path))
    logger.info("Protocol collection report saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def generate_literature_review_markdown(
    review: LiteratureReviewReport,
    rows: list[LiteratureReviewRow],
    method_summaries: list[MethodSummaryRow],
    output_path: str | Path,
) -> Path:
    """Generate a Markdown report for the literature review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# RNA-seq Literature Review Report",
        "",
        f"**Topic:** {review.topic}",
        f"**PubMed Query:** `{review.query}`",
        f"**Review ID:** {review.review_id}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Max results requested | {review.max_results} |",
        f"| PMIDs found | {review.pmids_found} |",
        f"| Articles fetched | {review.articles_fetched} |",
        f"| Method rows extracted | {review.rows_extracted} |",
        f"| Methods identified | {review.methods_identified} |",
        f"| Missing parameter count | {review.missing_parameter_count} |",
        "",
    ]

    if review.warnings:
        lines.append("### Warnings")
        for w in review.warnings:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    # Detailed table
    lines.append("## 2. Detailed Method Extraction Table")
    lines.append("")
    if rows:
        lines.append(
            "| Method | Software | Parameters | Lab Info | Article | Journal | PMID | DOI | Year | Evidence | Missing |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|"
        )
        for r in rows[:100]:
            method = (r.method_name or "-")[:30]
            software = (r.software_name or "-")[:30]
            params = (r.parameters or "-")[:30]
            lab = (r.lab_info or "-")[:25]
            title = (r.paper_title or "-")[:40]
            journal = (r.journal or "-")[:20]
            pmid = r.pmid
            doi = (r.doi or "-")[:20]
            year = str(r.year) if r.year else "-"
            evidence = (r.evidence_text or "-")[:60]
            missing = ", ".join(r.missing_fields or []) or "-"
            lines.append(
                f"| {method} | {software} | {params} | {lab} | {title} | {journal} | {pmid} | {doi} | {year} | {evidence} | {missing} |"
            )
    else:
        lines.append("No method rows extracted.")
    lines.append("")

    # Summary table
    lines.append("## 3. Method Summary Table")
    lines.append("")
    if method_summaries:
        lines.append(
            "| Method | Frequency | Articles | Confidence | Difficulty | Main Evidence | Missing Info |"
        )
        lines.append(
            "|---|---|---|---|---|---|"
        )
        for ms in method_summaries:
            lines.append(
                f"| {ms.method_name} | {ms.frequency_in_current_review} | {ms.supporting_article_count} | "
                f"{ms.confidence} | {ms.difficulty} | {(ms.main_evidence or '-')[:60]} | "
                f"{(ms.main_missing_information or '-')[:60]} |"
            )
    else:
        lines.append("No method summaries generated.")
    lines.append("")

    # Replication workflow
    lines.append("## 4. Replication Workflow")
    lines.append("")
    lines.append("### 4.1 Literature-Evidence-Supported Steps")
    methods = {}
    for row in rows:
        mn = row.method_name or "Unnamed"
        if mn not in methods:
            methods[mn] = []
        methods[mn].append(row)
    for method_name, method_rows in sorted(methods.items()):
        pmids = sorted(set(r.pmid for r in method_rows))
        lines.append(f"- **{method_name}**: supported by PMID(s): {', '.join(pmids[:5])}")
    lines.append("")
    lines.append("### 4.2 General Recommendations (not article-attributed)")
    lines.append(
        "QC (FastQC) → Trim (Trimmomatic/Cutadapt) → Align (STAR/HISAT2) → "
        "Quantify (featureCounts/Salmon) → DE (DESeq2/edgeR) → Enrich (clusterProfiler/GSEA)"
    )
    lines.append("")

    # Missing info
    lines.append("## 5. Missing Information")
    all_missing = []
    for r in rows:
        all_missing.extend(r.missing_fields or [])
    if all_missing:
        from collections import Counter
        for field, count in Counter(all_missing).most_common():
            lines.append(f"- **{field}**: missing in {count} row(s)")
    else:
        lines.append("No missing fields reported.")
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Literature review markdown saved to %s", output_path)
    return output_path


def generate_protocol_collection_markdown(
    run_report: ProtocolCollectionRunReport,
    protocols: list[BioProtocolRecord],
    output_path: str | Path,
) -> Path:
    """Generate a Markdown report for protocol collection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Cell Biology Protocol Collection Report",
        "",
        f"## 1. Run Information",
        f"- **Run ID:** {run_report.run_id}",
        f"- **Started:** {run_report.started_at or 'N/A'}",
        f"- **Ended:** {run_report.ended_at or 'N/A'}",
        f"- **Source:** {run_report.source}",
        f"- **Domain:** {run_report.domain}",
        f"- **Target count:** {run_report.target_count}",
        "",
        f"## 2. Collection Statistics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Searched | {run_report.searched_count} |",
        f"| Fetched | {run_report.fetched_count} |",
        f"| Ingested | {run_report.ingested_count} |",
        f"| Duplicates | {run_report.duplicate_count} |",
        f"| Skipped | {run_report.skipped_count} |",
        f"| Failed | {run_report.failed_count} |",
        f"",
        f"## 3. Data Source",
        f"- protocols.io public API",
        f"",
        f"## 4. Search Keywords",
    ]
    for kw in run_report.search_keywords:
        lines.append(f"- {kw}")
    lines.append("")

    # Field completeness
    lines.append("## 5. Field Completeness")
    lines.append("")
    if run_report.field_completeness:
        lines.append("| Field | Non-null | Missing | Completeness |")
        lines.append("|---|---|---|---|")
        for field, stats in run_report.field_completeness.items():
            nn = stats.get("non_null", 0)
            miss = stats.get("missing", 0)
            rate = stats.get("completeness", 0)
            rate_str = f"{rate:.1%}" if isinstance(rate, float) else str(rate)
            lines.append(f"| {field} | {nn} | {miss} | {rate_str} |")
    else:
        lines.append("No field completeness data available.")
    lines.append("")

    # Protocol list
    lines.append("## 6. Collected Protocol List")
    lines.append("")
    if protocols:
        lines.append("| Protocol ID | Title | Source | URL | Tags | Missing Fields |")
        lines.append("|---|---|---|---|---|---|")
        for p in protocols[:50]:
            tags = ", ".join(p.tags[:5])
            missing = ", ".join(p.missing_information[:5])
            lines.append(
                f"| {p.protocol_id[:20]} | {(p.title or '')[:60]} | {p.source} | "
                f"{p.url[:50]} | {tags} | {missing} |"
            )
    else:
        lines.append("No protocols collected.")
    lines.append("")

    # Failed items
    if run_report.failed_items:
        lines.append("## 7. Failed / Skipped Items")
        lines.append("")
        lines.append("| External ID | Title | Reason |")
        lines.append("|---|---|---|")
        for item in run_report.failed_items:
            lines.append(
                f"| {item.get('external_id', 'N/A')} | "
                f"{item.get('title', 'N/A')[:60]} | "
                f"{item.get('reason', 'Unknown')} |"
            )
        lines.append("")

    # RAG paths
    lines.append("## 8. RAG Paths")
    lines.append(f"- SQLite: `{run_report.rag_path or 'N/A'}`")
    lines.append(f"- JSONL: `{run_report.jsonl_path or 'N/A'}`")
    lines.append("")

    # Query examples
    lines.append("## 9. Query Examples")
    lines.append("```bash")
    lines.append(
        'python -m scicore.agent.bio_cli query-protocols '
        '--query "immunofluorescence staining cell culture" --top-k 10'
    )
    lines.append("```")
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Protocol collection markdown saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_kv_table(doc, pairs: list[tuple[str, str]]) -> None:
    """Add a two-column key-value table."""
    from docx.shared import Pt
    table = doc.add_table(rows=len(pairs), cols=2)
    table.style = "Light Grid Accent 1"
    for i, (key, value) in enumerate(pairs):
        table.rows[i].cells[0].text = key
        table.rows[i].cells[1].text = value


def _add_detailed_table(doc, rows: list[LiteratureReviewRow]) -> None:
    """Add the detailed method extraction table."""
    if not rows:
        doc.add_paragraph("No method rows extracted.")
        return

    headers = [
        "Method", "Software", "Parameters", "Lab Info",
        "Article", "Journal", "PMID", "DOI", "Year",
        "Evidence", "Missing Fields",
    ]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = 1  # center

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h

    for r in rows[:100]:
        row_cells = table.add_row().cells
        row_cells[0].text = (r.method_name or "-")[:30]
        row_cells[1].text = (r.software_name or "-")[:30]
        row_cells[2].text = (r.parameters or "-")[:30]
        row_cells[3].text = (r.lab_info or "-")[:25]
        row_cells[4].text = (r.paper_title or "-")[:40]
        row_cells[5].text = (r.journal or "-")[:20]
        row_cells[6].text = r.pmid
        row_cells[7].text = (r.doi or "-")[:20]
        row_cells[8].text = str(r.year) if r.year else "-"
        row_cells[9].text = (r.evidence_text or "-")[:60]
        row_cells[10].text = ", ".join(r.missing_fields or []) or "-"


def _add_summary_table(doc, summaries: list[MethodSummaryRow]) -> None:
    """Add the method summary table."""
    if not summaries:
        doc.add_paragraph("No method summaries generated.")
        return

    headers = [
        "Method", "Frequency", "Articles", "Confidence",
        "Difficulty", "Main Evidence", "Missing Info",
    ]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h

    for ms in summaries:
        row_cells = table.add_row().cells
        row_cells[0].text = ms.method_name
        row_cells[1].text = str(ms.frequency_in_current_review)
        row_cells[2].text = str(ms.supporting_article_count)
        row_cells[3].text = ms.confidence
        row_cells[4].text = ms.difficulty
        row_cells[5].text = (ms.main_evidence or "-")[:60]
        row_cells[6].text = (ms.main_missing_information or "-")[:60]
