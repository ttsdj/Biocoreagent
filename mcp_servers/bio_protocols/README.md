# Bio Protocols MCP Server

Protocol collection from protocols.io public API with SQLite RAG storage.

## Quickstart

```bash
# Set token (required)
export PROTOCOLS_IO_CLIENT_TOKEN="your_token_here"

# Test search
python -m mcp_servers.bio_protocols.server --test-search "cell culture"

# Check status
python -m mcp_servers.bio_protocols.server --status

# Run as MCP server
python -m mcp_servers.bio_protocols.server --stdio
```

Get a token: https://www.protocols.io/developers

## Tools

| Tool | Description |
|---|---|
| `protocols_io_search` | Search protocols.io for public protocols |
| `protocols_io_fetch` | Fetch a single protocol's full details |
| `protocols_io_collect_cell_biology` | Batch collect cell biology protocols across 10 keywords |
| `protocol_query_rag` | Query the protocol RAG store (SQLite FTS5) |

## Cell Biology Collection Keywords

The `collect_cell_biology` tool searches across 10 keywords:
cell biology, cell culture, immunofluorescence, immunostaining,
confocal microscopy, cell viability, cell transfection, flow cytometry,
cell migration, cell proliferation

## Configuration

| Env Var | Required | Description |
|---|---|---|
| `PROTOCOLS_IO_CLIENT_TOKEN` | Yes | API token for protocols.io |

## Output

Each collected protocol is stored as a `BioProtocolRecord` with:
- Protocol ID, title, URL, authors, summary
- Materials, reagents, instruments, software, databases
- Steps (step_number + description + duration)
- Tags, access level
- `missing_information` tracking per field

## RAG Storage

```
.scicore/rag/
├── protocols.sqlite    # SQLite FTS5
└── protocols.jsonl     # JSONL backup
```

## Failure Transparency

If fewer than `target_count` protocols are collected, the run report explicitly lists reasons:
- API returned fewer public records
- Duplicate records removed
- Records without accessible public content skipped
- Fetch failures
