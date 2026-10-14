"""Protocol MCP Server.

Implements MCP tools for protocol collection from protocols.io and
RAG storage/querying.

Runs as an MCP stdio server using JSON-RPC 2.0 over stdin/stdout.

Reference: SciCoreAgent 需求文档, Section 4.2.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .clients.protocols_io import ProtocolsIOClient

logger = logging.getLogger(__name__)

# Tool definitions for MCP
TOOLS = [
    {
        "name": "protocols_io_search",
        "title": "Search protocols.io",
        "description": "Search protocols.io for public protocols matching a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "protocols_io_fetch",
        "title": "Fetch protocol from protocols.io",
        "description": "Fetch a single protocol's details by its external ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "external_id": {"type": "string", "description": "Protocol external ID"},
            },
            "required": ["external_id"],
        },
    },
    {
        "name": "protocols_io_collect_cell_biology",
        "title": "Collect Cell Biology Protocols",
        "description": "Collect cell biology protocols from protocols.io and save to RAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Max total results", "default": 100},
                "collection_name": {"type": "string", "description": "Collection name", "default": "cell_biology_protocols"},
            },
        },
    },
    {
        "name": "protocol_query_rag",
        "title": "Query Protocol RAG",
        "description": "Query the protocol RAG store for matching protocols.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 10},
            },
            "required": ["query"],
        },
    },
]

# Search keywords for cell biology collection
CELL_BIO_KEYWORDS = [
    "cell biology",
    "cell culture",
    "immunofluorescence",
    "immunostaining",
    "confocal microscopy",
    "cell viability",
    "cell transfection",
    "flow cytometry",
    "cell migration",
    "cell proliferation",
]


class ProtocolServer:
    """Protocol collection MCP server."""

    def __init__(self):
        self._client = ProtocolsIOClient()

    @property
    def ready(self) -> bool:
        return self._client.ready

    def status(self) -> dict:
        return {
            "token_configured": self._client.ready,
        }

    # ------------------------------------------------------------------
    # Tool: protocols_io_search
    # ------------------------------------------------------------------

    def protocols_io_search(self, query: str, max_results: int = 20) -> dict:
        """Search protocols.io."""
        return self._client.search(query=query, max_results=max_results)

    # ------------------------------------------------------------------
    # Tool: protocols_io_fetch
    # ------------------------------------------------------------------

    def protocols_io_fetch(self, external_id: str) -> dict:
        """Fetch a single protocol."""
        return self._client.fetch(external_id=external_id)

    # ------------------------------------------------------------------
    # Tool: protocols_io_collect_cell_biology
    # ------------------------------------------------------------------

    def protocols_io_collect_cell_biology(self, max_results: int = 100,
                                            collection_name: str = "cell_biology_protocols") -> dict:
        """Collect cell biology protocols from protocols.io."""
        run_id = uuid.uuid4().hex[:12]
        warnings = []
        failed_items = []

        if not self._client.ready:
            return {
                "success": False, "target_count": max_results,
                "searched_count": 0, "fetched_count": 0, "ingested_count": 0,
                "duplicate_count": 0, "skipped_count": 0, "failed_count": 0,
                "rag_path": "", "report_path": "",
                "warnings": ["PROTOCOLS_IO_CLIENT_TOKEN not configured. "
                             "Protocol collection status: NOT_RUN_MISSING_TOKEN"],
            }

        # Step 1: Search across all cell biology keywords
        all_results = []
        seen_ids = set()
        per_keyword = max_results // len(CELL_BIO_KEYWORDS) + 5

        for keyword in CELL_BIO_KEYWORDS:
            if len(all_results) >= max_results:
                break
            try:
                result = self._client.search(query=keyword, max_results=per_keyword)
                for r in result.get("results", []):
                    eid = r.get("external_id")
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        all_results.append(r)
                warnings.extend(result.get("warnings", []))
            except Exception as e:
                warnings.append(f"Search for '{keyword}' failed: {e}")

        searched_count = len(all_results)
        if searched_count < max_results:
            warnings.append(
                f"Searched {searched_count} unique protocols, target was {max_results}. "
                "API returned fewer unique public records."
            )

        # Step 2: Fetch details for each
        fetched = []
        for r in all_results[:max_results]:
            eid = r.get("external_id")
            try:
                fetch_result = self._client.fetch(eid)
                if fetch_result.get("fetch_status") == "success":
                    protocol = fetch_result.get("protocol")
                    if protocol:
                        protocol["tags"] = list(set(protocol.get("tags", []) + r.get("tags", [])))
                        fetched.append(protocol)
                    else:
                        failed_items.append({
                            "external_id": eid,
                            "title": r.get("title", "N/A"),
                            "reason": "No protocol data returned",
                        })
                elif fetch_result.get("fetch_status") == "skipped":
                    failed_items.append({
                        "external_id": eid,
                        "title": r.get("title", "N/A"),
                        "reason": "Skipped: " + ", ".join(fetch_result.get("warnings", ["unknown"])),
                    })
                else:
                    failed_items.append({
                        "external_id": eid,
                        "title": r.get("title", "N/A"),
                        "reason": "Fetch failed",
                    })
                warnings.extend(fetch_result.get("warnings", []))
            except Exception as e:
                failed_items.append({
                    "external_id": eid,
                    "title": r.get("title", "N/A"),
                    "reason": str(e),
                })

        fetched_count = len(fetched)
        failed_count = len(failed_items)

        # Step 3: Save to RAG
        ingested_count = 0
        duplicate_count = 0
        skipped_count = 0
        rag_path = ""
        protocols_ingested = []

        try:
            from scicore.agent.bio.rag_store import default_rag_store
            from scicore.agent.bio.protocol_schemas import BioProtocolRecord, ProtocolEvidence

            store = default_rag_store()

            for protocol_data in fetched:
                try:
                    protocol_id = f"prot_{protocol_data.get('external_id', uuid.uuid4().hex[:8])}"

                    # Check for duplicates (by external_id)
                    existing = store.search_protocols(
                        protocol_data.get("title", ""), top_k=1
                    )
                    is_duplicate = any(
                        e.get("external_id") == protocol_data.get("external_id")
                        for e in existing
                    )

                    if is_duplicate:
                        duplicate_count += 1
                        continue

                    record = BioProtocolRecord(
                        protocol_id=protocol_id,
                        source="protocols_io",
                        external_id=protocol_data.get("external_id", ""),
                        title=protocol_data.get("title", ""),
                        url=protocol_data.get("url", ""),
                        domain="cell_biology",
                        authors=protocol_data.get("authors", []),
                        summary=protocol_data.get("summary"),
                        materials=protocol_data.get("materials", []),
                        steps=protocol_data.get("steps", []),
                        tags=protocol_data.get("tags", []),
                        access=protocol_data.get("access", "unknown"),
                    )

                    # Check skip conditions
                    if record.access == "restricted":
                        skipped_count += 1
                        failed_items.append({
                            "external_id": record.external_id,
                            "title": record.title,
                            "reason": "Restricted access",
                        })
                        continue

                    # Track missing information
                    missing = []
                    if not record.summary:
                        missing.append("summary")
                    if not record.materials:
                        missing.append("materials")
                    if not record.steps:
                        missing.append("steps")
                    if not record.authors:
                        missing.append("authors")
                    record.missing_information = missing

                    if store.add_protocol(record):
                        ingested_count += 1
                        protocols_ingested.append(record)

                except Exception as e:
                    failed_items.append({
                        "external_id": protocol_data.get("external_id", "N/A"),
                        "title": protocol_data.get("title", "N/A"),
                        "reason": f"Ingest error: {e}",
                    })

            rag_path = str(store._rag_dir)

        except ImportError as e:
            warnings.append(f"RAG store import failed: {e}")
        except Exception as e:
            warnings.append(f"RAG ingest failed: {e}")

        # Step 4: Generate collection report
        report_path = ""
        try:
            from scicore.agent.bio.report_writer import generate_protocol_collection_markdown
            from scicore.agent.bio.protocol_schemas import ProtocolCollectionRunReport

            # Compute field completeness
            field_stats = self._compute_field_completeness(protocols_ingested)

            # Build reason_not_full
            reason_not_full = []
            if ingested_count < max_results:
                if searched_count < max_results:
                    reason_not_full.append(
                        f"API returned fewer unique public records ({searched_count} found, {max_results} targeted)"
                    )
                if duplicate_count > 0:
                    reason_not_full.append(f"{duplicate_count} duplicate records removed")
                if skipped_count > 0:
                    reason_not_full.append(f"{skipped_count} records without accessible public content skipped")
                if failed_count > 0:
                    reason_not_full.append(f"{failed_count} records failed to fetch")

            run_report = ProtocolCollectionRunReport(
                run_id=run_id,
                source="protocols_io",
                domain="cell_biology",
                target_count=max_results,
                searched_count=searched_count,
                fetched_count=fetched_count,
                ingested_count=ingested_count,
                duplicate_count=duplicate_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                rag_path=rag_path,
                jsonl_path=str(Path(rag_path) / "protocols.jsonl") if rag_path else "",
                report_path="",
                started_at=datetime.now(timezone.utc).isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                failed_items=failed_items,
                warnings=warnings,
                field_completeness=field_stats,
                search_keywords=CELL_BIO_KEYWORDS,
                reason_not_full=reason_not_full,
            )

            report_dir = Path("reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(report_dir / f"{collection_name}_report.md")
            json_report_path = str(report_dir / f"{collection_name}_report.json")

            generate_protocol_collection_markdown(run_report, protocols_ingested, report_path)

            # JSON report
            Path(json_report_path).write_text(
                run_report.model_dump_json(indent=2, exclude_none=False),
                encoding="utf-8",
            )

        except Exception as e:
            warnings.append(f"Report generation failed: {e}")

        return {
            "success": True,
            "target_count": max_results,
            "searched_count": searched_count,
            "fetched_count": fetched_count,
            "ingested_count": ingested_count,
            "duplicate_count": duplicate_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "rag_path": rag_path,
            "report_path": report_path or "",
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Tool: protocol_query_rag
    # ------------------------------------------------------------------

    def protocol_query_rag(self, query: str, top_k: int = 10) -> dict:
        """Query the protocol RAG store."""
        try:
            from scicore.agent.bio.rag_store import default_rag_store
            store = default_rag_store()
            results = store.search_protocols(query=query, top_k=top_k)

            # Format for output with matched text highlighting
            formatted = []
            for r in results:
                matched_text = None
                section = None
                # Determine which section matched
                for field in ["title", "summary", "domain", "tags", "materials"]:
                    val = r.get(field)
                    if val and isinstance(val, (str, list)):
                        val_str = " ".join(val) if isinstance(val, list) else val
                        if query.lower() in val_str.lower():
                            matched_text = val_str[:200]
                            section = field
                            break

                formatted.append({
                    "protocol_id": r.get("protocol_id", ""),
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "url": r.get("url", ""),
                    "matched_text": matched_text,
                    "score": r.get("score"),
                    "section": section,
                    "missing_fields": r.get("missing_information", []),
                })

            return {"results": formatted, "count": len(formatted)}
        except Exception as e:
            logger.error("Protocol RAG query failed: %s", e)
            return {"results": [], "count": 0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_field_completeness(protocols: list) -> dict:
        """Compute per-field completeness statistics."""
        if not protocols:
            return {}

        fields = [
            "title", "url", "domain", "authors", "last_author",
            "summary", "materials", "reagents", "instruments",
            "software", "databases", "steps", "tags", "access",
        ]
        stats = {}
        total = len(protocols)

        for field in fields:
            non_null = sum(
                1 for p in protocols
                if getattr(p, field, None) and (
                    not isinstance(getattr(p, field), (list, str)) or
                    len(getattr(p, field)) > 0
                )
            )
            missing = total - non_null
            stats[field] = {
                "non_null": non_null,
                "missing": missing,
                "completeness": non_null / total if total > 0 else 0,
            }
        return stats

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call and return JSON result."""
        try:
            if tool_name == "protocols_io_search":
                result = self.protocols_io_search(**arguments)
            elif tool_name == "protocols_io_fetch":
                result = self.protocols_io_fetch(**arguments)
            elif tool_name == "protocols_io_collect_cell_biology":
                result = self.protocols_io_collect_cell_biology(**arguments)
            elif tool_name == "protocol_query_rag":
                result = self.protocol_query_rag(**arguments)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return json.dumps({"error": f"Tool {tool_name} failed: {e}"})

    def get_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return TOOLS


# ---------------------------------------------------------------------------
# MCP stdio loop
# ---------------------------------------------------------------------------


def run_stdio():
    """Run the Protocol MCP server via JSON-RPC 2.0 over stdin/stdout."""
    server = ProtocolServer()
    logger.info("Protocol MCP server started (stdio)")

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
                        "name": "bio_protocols",
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
            continue
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

    p = argparse.ArgumentParser(description="Protocol MCP Server")
    p.add_argument("--stdio", action="store_true", help="Run in MCP stdio mode")
    p.add_argument("--test-search", help="Test protocol search with a query")
    p.add_argument("--status", action="store_true", help="Show server status")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.stdio:
        run_stdio()
    elif args.test_search:
        server = ProtocolServer()
        result = server.protocols_io_search(query=args.test_search, max_results=5)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.status:
        server = ProtocolServer()
        print(json.dumps(server.status(), ensure_ascii=False, indent=2))
    else:
        p.print_help()
