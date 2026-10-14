"""SciCoreAgent CLI - literature review, protocol collection, and system status.

Usage:
    python -m scicore.agent.bio_cli doctor
    python -m scicore.agent.bio_cli literature-review --topic "RNA-seq workflow" ...
    python -m scicore.agent.bio_cli collect-protocols --source protocols_io --domain cell_biology ...
    python -m scicore.agent.bio_cli query-protocols --query "immunofluorescence" --top-k 10
    python -m scicore.agent.bio_cli report-status

Reference: SciCoreAgent 需求文档, Section 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass
from datetime import datetime, timezone


def main():
    """Main entry point for bio_cli."""
    p = argparse.ArgumentParser(
        prog="scicore-bio",
        description="SciCoreAgent Bioinformatics CLI",
    )
    sub = p.add_subparsers(dest="command", help="Available commands")

    # doctor
    sub.add_parser("doctor", help="Check system environment and configuration")

    # literature-review
    lit = sub.add_parser("literature-review", help="Run PubMed RNA-seq literature review")
    lit.add_argument("--topic", default="RNA-seq workflow")
    lit.add_argument("--query", required=True, help="PubMed search query")
    lit.add_argument("--max-results", type=int, default=50)
    lit.add_argument("--out", default="reports/rnaseq_literature_review.docx",
                     help="Output .docx path")

    # collect-protocols
    cp = sub.add_parser("collect-protocols", help="Collect protocols from external sources")
    cp.add_argument("--source", default="protocols_io")
    cp.add_argument("--domain", default="cell_biology")
    cp.add_argument("--max-results", type=int, default=100)
    cp.add_argument("--collection-name", default="cell_biology_protocols")

    # query-protocols
    qp = sub.add_parser("query-protocols", help="Query the protocol RAG store")
    qp.add_argument("--query", required=True)
    qp.add_argument("--top-k", type=int, default=10)

    # report-status
    sub.add_parser("report-status", help="Show data report and system status")

    args = p.parse_args()

    if not args.command:
        p.print_help()
        return

    # Dispatch
    if args.command == "doctor":
        cmd_doctor()
    elif args.command == "literature-review":
        cmd_literature_review(args)
    elif args.command == "collect-protocols":
        cmd_collect_protocols(args)
    elif args.command == "query-protocols":
        cmd_query_protocols(args)
    elif args.command == "report-status":
        cmd_report_status()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor():
    """Check system environment and configuration."""
    results = []

    def check(label: str, ok: bool, detail: str = "", warn: bool = False) -> str:
        if warn:
            status = "[WARN]"
            results.append((status, label, detail))
        else:
            status = "[OK]" if ok else "[FAIL]"
            results.append((status, label, detail))
        line = f"{status} {label}{' - ' + detail if detail else ''}"
        print(line)
        return line

    print("SciCoreAgent Doctor Report\n")

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check("Python", sys.version_info >= (3, 10), py_ver)

    # 2. python-docx
    try:
        import docx  # noqa: F401
        check("python-docx", True, "installed")
    except ImportError:
        check("python-docx", False, "NOT INSTALLED - run: pip install python-docx")

    # 3. SQLite FTS5
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS test_fts USING fts5(content)")
        conn.close()
        check("SQLite FTS5", True, "available")
    except Exception as e:
        check("SQLite FTS5", False, str(e))

    # 4. PubMed MCP server importable
    try:
        from mcp_servers.pubmed_literature.server import PubMedServer  # noqa: F401
        pubmed = PubMedServer()
        check("PubMed MCP server", True, "importable")
    except Exception as e:
        check("PubMed MCP server", False, str(e))
        pubmed = None

    # 5. Protocol MCP server importable
    try:
        from mcp_servers.bio_protocols.server import ProtocolServer  # noqa: F401
        prot = ProtocolServer()
        check("Protocol MCP server", True, "importable")
    except Exception as e:
        check("Protocol MCP server", False, str(e))
        prot = None

    # 6. NCBI_EMAIL
    ncbi_email = os.environ.get("NCBI_EMAIL", "")
    check("NCBI_EMAIL", bool(ncbi_email), ncbi_email if ncbi_email else "MISSING")

    # 7. NCBI_API_KEY
    ncbi_key = os.environ.get("NCBI_API_KEY", "")
    if ncbi_key:
        check("NCBI_API_KEY", True, "configured")
    else:
        check("NCBI_API_KEY", True, "not configured; rate limit will use 3 requests/second", warn=True)

    # 8. PROTOCOLS_IO_CLIENT_TOKEN
    prot_token = os.environ.get("PROTOCOLS_IO_CLIENT_TOKEN", "")
    check("PROTOCOLS_IO_CLIENT_TOKEN", bool(prot_token), "configured" if prot_token else "MISSING")

    # 9. reports/ directory writable
    reports_dir = Path("reports")
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        test_file = reports_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        check("reports/ directory", True, "writable")
    except Exception as e:
        check("reports/ directory", False, str(e))

    # 10. .scicore/rag/ directory writable
    rag_dir = Path(".scicore") / "rag"
    try:
        rag_dir.mkdir(parents=True, exist_ok=True)
        test_file = rag_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        check("rag/ directory", True, "writable")
    except Exception as e:
        check("rag/ directory", False, str(e))

    # Summary
    print()
    all_ok = all(r[0] == "[OK]" for r in results if r[0] in ("[OK]", "[FAIL]"))
    warns = [r for r in results if r[0] == "[WARN]"]

    critical_failures = [
        r for r in results
        if r[0] == "[FAIL]" and r[1] in ("Python", "SQLite FTS5", "reports/ directory", "rag/ directory")
    ]

    if critical_failures:
        print("Status: NOT_READY")
        print("Reason:")
    elif warns:
        print("Status: READY (with warnings)")
        for _, label, detail in warns:
            print(f"  - {label}: {detail}")
    else:
        print("Status: READY")

    if critical_failures:
        for _, label, detail in critical_failures:
            print(f"  - {label}: {detail}")


# ---------------------------------------------------------------------------
# literature-review
# ---------------------------------------------------------------------------


def cmd_literature_review(args):
    """Run a PubMed RNA-seq literature review."""
    print(f"Running PubMed Literature Review...")
    print(f"  Topic: {args.topic}")
    print(f"  Query: {args.query}")
    print(f"  Max Results: {args.max_results}")
    print(f"  Output: {args.out}")
    print()

    try:
        from mcp_servers.pubmed_literature.server import PubMedServer

        server = PubMedServer()

        if not server.ready:
            print("ERROR: NCBI_EMAIL not configured.")
            print("Set NCBI_EMAIL environment variable and try again.")
            print("export NCBI_EMAIL=\"your_email@example.com\"")
            sys.exit(1)

        result = server.pubmed_literature_review(
            topic=args.topic,
            query=args.query,
            max_results=args.max_results,
            output_docx=args.out,
        )

        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

        if result.get("success"):
            print("Literature review completed successfully.")
            print(f"  PMIDs found: {result.get('pmids_found')}")
            print(f"  Articles fetched: {result.get('articles_fetched')}")
            print(f"  Rows extracted: {result.get('rows_extracted')}")
            print(f"  DOCX: {result.get('docx_path')}")
            print(f"  Markdown: {result.get('markdown_path')}")
            print(f"  Run report: {result.get('run_report_path')}")
        else:
            print("Literature review completed with issues.")
            for w in result.get("warnings", []):
                print(f"  Warning: {w}")

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Make sure the PubMed MCP server is in the Python path.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# collect-protocols
# ---------------------------------------------------------------------------


def cmd_collect_protocols(args):
    """Collect protocols from external sources."""
    print(f"Collecting protocols...")
    print(f"  Source: {args.source}")
    print(f"  Domain: {args.domain}")
    print(f"  Max Results: {args.max_results}")
    print(f"  Collection Name: {args.collection_name}")
    print()

    token = os.environ.get("PROTOCOLS_IO_CLIENT_TOKEN", "")
    if not token:
        print("Protocol collection status: NOT_RUN_MISSING_TOKEN")
        print("Reason: PROTOCOLS_IO_CLIENT_TOKEN is required for protocols.io public API access.")
        print("Next step: set PROTOCOLS_IO_CLIENT_TOKEN and rerun collect-protocols.")
        print()
        print("You can get a token from: https://www.protocols.io/developers")
        return

    try:
        from mcp_servers.bio_protocols.server import ProtocolServer

        server = ProtocolServer()
        result = server.protocols_io_collect_cell_biology(
            max_results=args.max_results,
            collection_name=args.collection_name,
        )

        print(f"Collection Results:")
        print(f"  Target: {result.get('target_count')}")
        print(f"  Searched: {result.get('searched_count')}")
        print(f"  Fetched: {result.get('fetched_count')}")
        print(f"  Ingested: {result.get('ingested_count')}")
        print(f"  Duplicates: {result.get('duplicate_count')}")
        print(f"  Skipped: {result.get('skipped_count')}")
        print(f"  Failed: {result.get('failed_count')}")
        print(f"  RAG path: {result.get('rag_path')}")
        print(f"  Report: {result.get('report_path')}")

        if result.get("warnings"):
            print()
            print("Warnings:")
            for w in result["warnings"]:
                print(f"  - {w}")

        if result["ingested_count"] < args.max_results:
            print()
            print("Note: Collected fewer protocols than the target. See reasons above.")
            print(f"  target_count: {args.max_results}")
            print(f"  searched_count: {result['searched_count']}")
            print(f"  fetched_count: {result['fetched_count']}")
            print(f"  ingested_count: {result['ingested_count']}")
            print(f"  skipped_count: {result['skipped_count']}")
            print("  Reasons may include:")
            print("    - API returned fewer public records")
            print("    - Duplicate records removed")
            print("    - Records without accessible public content skipped")

    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# query-protocols
# ---------------------------------------------------------------------------


def cmd_query_protocols(args):
    """Query the protocol RAG store."""
    print(f"Querying protocol RAG: '{args.query}' (top {args.top_k})\n")

    try:
        from scicore.agent.bio.rag_store import default_rag_store

        store = default_rag_store()
        results = store.search_protocols(query=args.query, top_k=args.top_k)

        if not results:
            print("No results found.")
            return

        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.get('protocol_id', '?')[:16]}] {r.get('title', 'No title')}")
            print(f"   Source: {r.get('source', '?')}")
            print(f"   URL: {r.get('url', 'N/A')}")
            print(f"   Domain: {r.get('domain', 'N/A')}")
            missing = r.get("missing_information", [])
            if missing:
                print(f"   Missing fields: {', '.join(missing[:5])}")
            print(f"   Score: {r.get('score', 'N/A')}")
            print()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# report-status
# ---------------------------------------------------------------------------


def cmd_report_status():
    """Show comprehensive data report and system status."""
    print("SciCoreAgent Data Report\n")

    # 1. MCP tools
    print("## MCP Tools Built")
    print()
    try:
        from mcp_servers.pubmed_literature.server import TOOLS as PUBMED_TOOLS
        print(f"PubMed Literature MCP: {len(PUBMED_TOOLS)} tools")
        for t in PUBMED_TOOLS:
            print(f"  - {t['name']}: {t['description'][:80]}")
    except Exception:
        print("PubMed Literature MCP: NOT IMPORTABLE")

    try:
        from mcp_servers.bio_protocols.server import TOOLS as PROTOCOL_TOOLS
        print(f"Protocol MCP: {len(PROTOCOL_TOOLS)} tools")
        for t in PROTOCOL_TOOLS:
            print(f"  - {t['name']}: {t['description'][:80]}")
    except Exception:
        print("Protocol MCP: NOT IMPORTABLE")
    print()

    # 2. CLI commands
    print("## CLI Commands Built")
    commands = [
        ("doctor", "Check system environment and configuration"),
        ("literature-review", "Run PubMed RNA-seq literature review"),
        ("collect-protocols", "Collect protocols from external sources"),
        ("query-protocols", "Query the protocol RAG store"),
        ("report-status", "Show this data report"),
    ]
    for cmd, desc in commands:
        print(f"  - {cmd}: {desc}")
    print()

    # 3. RAG status
    print("## RAG Status")
    try:
        from scicore.agent.bio.rag_store import default_rag_store
        store = default_rag_store()
        stats = store.stats()

        lit = stats.get("literature", {})
        prot = stats.get("protocols", {})

        print(f"  Literature RAG: {lit.get('db_path', 'N/A')}")
        print(f"    Articles: {lit.get('articles', 0)}")
        print(f"    Review rows: {lit.get('review_rows', 0)}")
        print(f"  Protocol RAG: {prot.get('db_path', 'N/A')}")
        print(f"    Protocol records: {prot.get('records', 0)}")
        by_source = prot.get("by_source", {})
        if by_source:
            for source, count in by_source.items():
                print(f"      {source}: {count}")
    except Exception as e:
        print(f"  RAG status unavailable: {e}")
    print()

    # 4. Recent run results (check for existing reports)
    print("## Report Files")
    reports_dir = Path("reports")
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*")):
            size = f.stat().st_size if f.is_file() else 0
            print(f"  {f.name} ({_format_size(size)})")
    else:
        print("  No reports directory found.")
    print()

    # 5. Environment
    print("## Environment")
    print(f"  NCBI_EMAIL: {'configured' if os.environ.get('NCBI_EMAIL') else 'MISSING'}")
    print(f"  NCBI_API_KEY: {'configured' if os.environ.get('NCBI_API_KEY') else 'not set (3 req/s limit)'}")
    print(f"  PROTOCOLS_IO_CLIENT_TOKEN: {'configured' if os.environ.get('PROTOCOLS_IO_CLIENT_TOKEN') else 'MISSING'}")
    print()

    # 6. Config status
    print("## Configuration Status")
    issues = []
    if not os.environ.get("NCBI_EMAIL"):
        issues.append("NCBI_EMAIL missing - PubMed MCP will not work")
    if not os.environ.get("PROTOCOLS_IO_CLIENT_TOKEN"):
        issues.append("PROTOCOLS_IO_CLIENT_TOKEN missing - Protocol collection will not run")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  All required configuration present.")


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


if __name__ == "__main__":
    main()
